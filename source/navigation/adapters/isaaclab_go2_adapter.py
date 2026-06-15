"""Isaac Lab locomotion-policy adapter for Go2-X5 navigation.

Imports are intentionally lazy so pure navigation tests do not require Isaac
Lab, Torch, or a GPU runtime.
"""

from __future__ import annotations

import math
from typing import Any

from .frame_utils import yaw_to_quat_wxyz


ARM_JOINT_NAMES = [f"arm_joint{index}" for index in range(1, 7)]
GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]
DOG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]


def _item(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _quat_to_yaw(quat_wxyz: Any) -> float:
    w, x, y, z = (_item(value) for value in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_to_roll_pitch(quat_wxyz: Any) -> tuple[float, float]:
    w, x, y, z = (_item(value) for value in quat_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


class Go2LocomotionAdapter:
    """Bridge DWA body commands to an Isaac Lab command-conditioned policy."""

    def __init__(self, env: Any, policy: Any, observations: Any):
        self.env = env
        self.policy = policy
        self.observations = observations
        self.runtime = env.unwrapped
        self.robot = self.runtime.scene["robot"]
        self.base_cmd_term = self.runtime.command_manager._terms.get("base_velocity")
        if self.base_cmd_term is None:
            raise RuntimeError("Isaac Lab task is missing the base_velocity command term.")
        self.arm_term = self.runtime.command_manager._terms.get("arm_joint_pos")
        self.joint_pos_action_term = self.runtime.action_manager._terms.get("joint_pos")
        self.dog_joint_ids, _ = self.robot.find_joints(DOG_JOINT_NAMES, preserve_order=True)
        self.arm_joint_ids, _ = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.gripper_joint_ids, _ = self.robot.find_joints(GRIPPER_JOINT_NAMES, preserve_order=True)
        self.ee_body_ids, _ = self.robot.find_bodies(["arm_link6"])
        self.dog_action_indices = self._resolve_action_indices(DOG_JOINT_NAMES)
        self.arm_action_indices = self._resolve_action_indices(ARM_JOINT_NAMES)
        self.direct_arm_action_override = False
        self._base_pose_lock_xyzyaw: tuple[float, float, float, float] | None = None
        self._dog_joint_lock_target = None
        self._command = (0.0, 0.0, 0.0)
        self._arm_joint_target = None
        self._gripper_joint_target = None
        self._last_actions = None

    def _write_root_pose_xyzyaw(self, x: float, y: float, z: float, yaw: float) -> None:
        """Write a level root pose and zero root velocity directly to simulation."""
        import torch

        quat = yaw_to_quat_wxyz(yaw)
        pose = torch.tensor([[x, y, z, *quat]], dtype=torch.float32, device=self.runtime.device)
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.runtime.device)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(velocity)

    def reset_to_pose(self, x: float, y: float, yaw: float) -> None:
        """Write a root pose and zero root velocity directly to simulation."""

        current_z = _item(self.robot.data.root_pos_w[0][2])
        self._write_root_pose_xyzyaw(x, y, current_z, yaw)

    def set_base_pose_lock(self, enabled: bool = True, pose_xyyaw: tuple[float, float, float] | None = None) -> dict[str, Any]:
        """Pin the floating base to a level world x/y/z/yaw pose during manipulation."""

        if enabled:
            pose = pose_xyyaw if pose_xyyaw is not None else self.get_base_pose()
            z = _item(self.robot.data.root_pos_w[0][2])
            self._base_pose_lock_xyzyaw = (float(pose[0]), float(pose[1]), float(z), float(pose[2]))
        else:
            self._base_pose_lock_xyzyaw = None
        pose_xyzyaw = list(self._base_pose_lock_xyzyaw) if self._base_pose_lock_xyzyaw is not None else None
        pose_xyyaw_report = (
            [pose_xyzyaw[0], pose_xyzyaw[1], pose_xyzyaw[3]]
            if pose_xyzyaw is not None
            else None
        )
        return {
            "enabled": self._base_pose_lock_xyzyaw is not None,
            "pose_xyzyaw": pose_xyzyaw,
            "pose_xyyaw": pose_xyyaw_report,
        }

    def _apply_base_pose_lock(self) -> None:
        if self._base_pose_lock_xyzyaw is None:
            return
        self._write_root_pose_xyzyaw(*self._base_pose_lock_xyzyaw)

    def apply_base_pose_lock(self) -> dict[str, Any]:
        """应用已捕获的 root pose；不推进仿真。"""

        if self._base_pose_lock_xyzyaw is None:
            return {"applied": False, "reason": "base_pose_lock_disabled"}
        self._apply_base_pose_lock()
        return {
            "applied": True,
            "pose_xyzyaw": list(self._base_pose_lock_xyzyaw),
            "uses_direct_root_state": True,
        }

    def set_support_joint_lock(self, enabled: bool = True) -> dict[str, Any]:
        """冻结当前四足支撑关节姿态，不直接改写关节状态。"""

        if enabled:
            if len(self.dog_joint_ids) != len(DOG_JOINT_NAMES):
                self._dog_joint_lock_target = None
            else:
                self._dog_joint_lock_target = self.robot.data.joint_pos[0, self.dog_joint_ids].detach().clone().reshape(1, -1)
        else:
            self._dog_joint_lock_target = None
        return {
            "enabled": self._dog_joint_lock_target is not None,
            "joint_names": list(DOG_JOINT_NAMES) if self._dog_joint_lock_target is not None else [],
            "joint_ids": [int(index) for index in self.dog_joint_ids] if self._dog_joint_lock_target is not None else [],
            "action_indices": list(self.dog_action_indices or []),
            "uses_direct_joint_state": False,
        }

    def _apply_support_joint_lock(self) -> None:
        if self._dog_joint_lock_target is None:
            return
        import torch

        target = self._dog_joint_lock_target.to(device=self.runtime.device, dtype=torch.float32)
        velocity = torch.zeros_like(target)
        # IsaacLab 的 write_joint_state_to_sim(joint_ids=...) 会把整条 articulation
        # 的 DOF buffer 写回 PhysX。这里不能用它锁腿，否则会把 arm_joint1~6
        # 也反复写回旧状态，导致 pick/place 看起来完全没有执行。
        self.robot.set_joint_position_target(target, joint_ids=self.dog_joint_ids)
        self.robot.set_joint_velocity_target(velocity, joint_ids=self.dog_joint_ids)

    def apply_support_joint_lock(self) -> dict[str, Any]:
        """应用已捕获的支撑关节姿态；不推进仿真。"""

        if self._dog_joint_lock_target is None:
            return {"applied": False, "reason": "support_joint_lock_disabled"}
        self._apply_support_joint_lock()
        return {
            "applied": True,
            "joint_names": list(DOG_JOINT_NAMES),
            "joint_ids": [int(index) for index in self.dog_joint_ids],
            "action_indices": list(self.dog_action_indices or []),
            "target_positions": [
                float(value)
                for value in self._dog_joint_lock_target.reshape(-1).detach().cpu().tolist()
            ],
            "uses_direct_joint_state": False,
            "lock_mode": "position_velocity_target_only",
        }

    def _apply_gripper_joint_target(self) -> None:
        if len(self.gripper_joint_ids) != 2:
            return
        import torch

        gripper_target = (
            torch.zeros((1, 2), dtype=torch.float32, device=self.runtime.device)
            if self._gripper_joint_target is None
            else torch.as_tensor(
                self._gripper_joint_target,
                dtype=torch.float32,
                device=self.runtime.device,
            ).reshape(1, -1)
        )
        self.robot.set_joint_position_target(gripper_target, joint_ids=self.gripper_joint_ids)

    def apply_gripper_joint_target(self) -> dict[str, Any]:
        """把当前夹爪目标写成 articulation position target，不改写关节状态。"""

        if self._gripper_joint_target is None:
            return {"applied": False, "reason": "gripper_joint_target_disabled"}
        if len(self.gripper_joint_ids) != len(GRIPPER_JOINT_NAMES):
            return {
                "applied": False,
                "reason": "gripper_joint_id_count_mismatch",
                "joint_ids": [int(index) for index in self.gripper_joint_ids],
            }
        import torch

        target = torch.as_tensor(
            self._gripper_joint_target,
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape(1, -1)
        if target.shape[1] != len(self.gripper_joint_ids):
            return {
                "applied": False,
                "reason": "gripper_joint_target_count_mismatch",
                "target_count": int(target.shape[1]),
                "joint_count": len(self.gripper_joint_ids),
            }
        velocity_target = torch.zeros_like(target)
        self.robot.set_joint_position_target(target, joint_ids=self.gripper_joint_ids)
        self.robot.set_joint_velocity_target(velocity_target, joint_ids=self.gripper_joint_ids)
        return {
            "applied": True,
            "joint_names": list(GRIPPER_JOINT_NAMES),
            "joint_ids": [int(index) for index in self.gripper_joint_ids],
            "target_positions": [
                float(value) for value in target.reshape(-1).detach().cpu().tolist()
            ],
            "uses_direct_joint_state": False,
        }

    def get_base_pose(self) -> tuple[float, float, float]:
        """Return world-frame ``x, y, yaw``."""

        position = self.robot.data.root_pos_w[0]
        return _item(position[0]), _item(position[1]), _quat_to_yaw(self.robot.data.root_quat_w[0])

    def get_base_pose_full(self) -> dict[str, Any]:
        """Return the root pose fields used by nav-result handoff."""

        position = self.robot.data.root_pos_w[0]
        quat = self.robot.data.root_quat_w[0]
        return {
            "x": _item(position[0]),
            "y": _item(position[1]),
            "z": _item(position[2]),
            "yaw": _quat_to_yaw(quat),
            "quat_wxyz": [_item(value) for value in quat],
        }

    def get_base_velocity(self) -> tuple[float, float]:
        """Return measured body-frame ``vx, wz`` for DWA dynamic windows."""

        linear = self.robot.data.root_lin_vel_b[0]
        angular = self.robot.data.root_ang_vel_b[0]
        return _item(linear[0]), _item(angular[2])

    def get_base_velocity_full(self) -> tuple[float, float, float]:
        """Return measured body-frame ``vx, vy, wz``."""

        linear = self.robot.data.root_lin_vel_b[0]
        angular = self.robot.data.root_ang_vel_b[0]
        return _item(linear[0]), _item(linear[1]), _item(angular[2])

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        """Store a body command; it is injected before the next policy step."""

        self._command = float(vx), float(vy), float(wz)

    def set_arm_joint_target(self, target: Any | None) -> None:
        """Optionally hold arm joints while the locomotion policy steps."""

        self._arm_joint_target = target

    def set_gripper_joint_target(self, target: Any | None) -> None:
        """Optionally hold gripper joints while the locomotion policy steps."""

        self._gripper_joint_target = target

    def apply_arm_joint_target(self) -> dict[str, Any]:
        """把当前机械臂目标写成 articulation position target，不改写关节状态。"""

        if self._arm_joint_target is None:
            return {"applied": False, "reason": "arm_joint_target_disabled"}
        if len(self.arm_joint_ids) != len(ARM_JOINT_NAMES):
            return {
                "applied": False,
                "reason": "arm_joint_id_count_mismatch",
                "joint_ids": [int(index) for index in self.arm_joint_ids],
            }
        import torch

        target = torch.as_tensor(
            self._arm_joint_target,
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape(1, -1)
        if target.shape[1] != len(self.arm_joint_ids):
            return {
                "applied": False,
                "reason": "arm_joint_target_count_mismatch",
                "target_count": int(target.shape[1]),
                "joint_count": len(self.arm_joint_ids),
            }
        self.robot.set_joint_position_target(target, joint_ids=self.arm_joint_ids)
        return {
            "applied": True,
            "joint_names": list(ARM_JOINT_NAMES),
            "joint_ids": [int(index) for index in self.arm_joint_ids],
            "target_positions": [
                float(value) for value in target.reshape(-1).detach().cpu().tolist()
            ],
            # baseline 只通过 position action 驱动机械臂；运动阶段不能把 velocity
            # target 每拍写成 0，否则阻尼项会持续抵抗 cuRobo 轨迹跟踪。
            "control_mode": "position_target_only",
            "velocity_target_written": False,
            "uses_direct_joint_state": False,
        }

    def set_direct_arm_action_override(self, enabled: bool = True) -> dict[str, Any]:
        """Override policy arm action slots with externally supplied joint targets.

        The Go2-X5 locomotion policy action controls both dog joints and
        arm_joint1~6. Writing only the arm command term asks the policy to track
        a target, but does not guarantee execution. Contact manipulation needs
        the cuRobo trajectory to own the arm slots while the policy still owns
        the legs, so we replace just those action dimensions before env.step().
        """

        self.direct_arm_action_override = bool(enabled)
        return {
            "enabled": self.direct_arm_action_override,
            "action_term_available": self.joint_pos_action_term is not None,
            "arm_action_indices": list(self.arm_action_indices or []),
            "arm_joint_names": list(ARM_JOINT_NAMES),
        }

    def _resolve_action_indices(self, joint_name_order: list[str]) -> list[int] | None:
        """Map ordered joint names into the joint_pos action vector."""

        action_term = self.joint_pos_action_term
        if action_term is None:
            return None
        joint_names = list(getattr(action_term, "_joint_names", []))
        if not joint_names:
            return None
        indices = []
        for joint_name in joint_name_order:
            if joint_name not in joint_names:
                return None
            indices.append(joint_names.index(joint_name))
        return indices

    def _term_values_for_indices(self, value: Any, indices: Any, actions: Any, *, default: float):
        """Return a tensor [num_envs, len(indices)] for action scale/offset values."""

        import torch

        if hasattr(value, "detach"):
            tensor = value.to(device=actions.device, dtype=actions.dtype)
            if tensor.ndim == 0:
                return tensor.reshape(1, 1).expand(actions.shape[0], indices.numel())
            if tensor.ndim == 1:
                return tensor[indices].reshape(1, -1).expand(actions.shape[0], -1)
            return tensor[: actions.shape[0], :][:, indices]
        return torch.full(
            (actions.shape[0], indices.numel()),
            float(value if value is not None else default),
            dtype=actions.dtype,
            device=actions.device,
        )

    def _override_target_actions(self, actions: Any, action_indices: list[int] | None, target: Any | None):
        """Replace selected policy action dimensions with direct joint targets."""

        import torch

        if target is None:
            return actions
        if self.joint_pos_action_term is None or not action_indices:
            return actions
        indices = torch.as_tensor(action_indices, dtype=torch.long, device=actions.device)
        target = torch.as_tensor(target, dtype=actions.dtype, device=actions.device).reshape(1, -1)
        if target.shape[1] != indices.numel():
            return actions
        scale = self._term_values_for_indices(
            getattr(self.joint_pos_action_term, "_scale", 1.0),
            indices,
            actions,
            default=1.0,
        )
        offset = self._term_values_for_indices(
            getattr(self.joint_pos_action_term, "_offset", 0.0),
            indices,
            actions,
            default=0.0,
        )
        raw_arm_action = (target.expand(actions.shape[0], -1) - offset) / torch.clamp(scale.abs(), min=1.0e-6)
        overridden = actions.clone()
        overridden[:, indices] = raw_arm_action
        return overridden

    def _override_arm_actions(self, actions: Any):
        """Replace policy action dimensions for dog locks and arm targets."""

        actions = self._override_target_actions(actions, self.dog_action_indices, self._dog_joint_lock_target)
        if not self.direct_arm_action_override or self._arm_joint_target is None:
            return actions
        return self._override_target_actions(actions, self.arm_action_indices, self._arm_joint_target)

    def compute_policy_action(self, *, refresh_observations: bool = True) -> Any:
        """写入速度命令并执行策略推理，但不推进仿真。"""

        import torch

        command = torch.tensor([self._command], dtype=torch.float32, device=self.base_cmd_term.device)
        self.base_cmd_term.vel_command_b[:] = command
        if hasattr(self.base_cmd_term, "is_heading_env"):
            self.base_cmd_term.is_heading_env[:] = False
        if hasattr(self.base_cmd_term, "is_standing_env"):
            self.base_cmd_term.is_standing_env[:] = torch.linalg.norm(command, dim=1) < 1.0e-6
        if hasattr(self.base_cmd_term, "heading_target"):
            self.base_cmd_term.heading_target[:] = 0.0
        if self.arm_term is not None:
            if self._arm_joint_target is None:
                self.arm_term.command_buffer[:] = 0.0
            else:
                arm_target = torch.as_tensor(
                    self._arm_joint_target,
                    dtype=torch.float32,
                    device=self.arm_term.command_buffer.device,
                ).reshape(1, -1)
                self.arm_term.command_buffer[:, : arm_target.shape[1]] = arm_target

        with torch.inference_mode():
            # 新 pipeline 的导航阶段不会开启这些锁；保留调用是为了兼容旧 manipulation 路径。
            self._apply_base_pose_lock()
            self._apply_support_joint_lock()
            self._apply_gripper_joint_target()
            if refresh_observations:
                self.observations = self.env.get_observations()
            actions = self.policy(self.observations)
            clip_actions = getattr(self.env, "clip_actions", None)
            if clip_actions is not None:
                actions = torch.clamp(actions, -clip_actions, clip_actions)
            # 先裁剪 locomotion policy 输出，再写入 cuRobo 直接关节目标。
            # Foundation 配置中 arm action scale 只有 0.10；如果 override 后再 clip，
            # 1 rad 级别的机械臂目标会被等效压成约 0.1 rad，表现为 pick 阶段原地等待。
            actions = self._override_arm_actions(actions)
            self._last_actions = actions.detach()
        return actions

    def update_observations(self, observations: Any | None = None) -> Any:
        """在 simulation runtime 完成物理步后更新策略观测。"""

        self.observations = self.env.get_observations() if observations is None else observations
        return self.observations

    def step(self) -> Any:
        """旧流程兼容入口；新 full-physics pipeline 禁止调用此方法。"""

        import torch

        actions = self.compute_policy_action()
        with torch.inference_mode():
            self.observations, _, _, _ = self.env.step(actions)
            self._apply_gripper_joint_target()
            self._apply_support_joint_lock()
            self._apply_base_pose_lock()
        return self.observations

    def settle(self, steps: int = 120) -> None:
        """Hold zero body command for a fixed number of policy steps."""

        self.apply_base_command(0.0, 0.0, 0.0)
        for _ in range(max(0, steps)):
            self.step()

    def is_stable(self, linear_tolerance: float = 0.05, angular_tolerance: float = 0.10) -> bool:
        """Check whether body-frame planar velocity is sufficiently small."""

        vx, vy, wz = self.get_base_velocity_full()
        return math.hypot(vx, vy) <= linear_tolerance and abs(wz) <= angular_tolerance

    def snapshot(self, *, timestamp: float, phase: str) -> dict[str, Any]:
        """Return one stable-schema recorder row."""

        pose = self.get_base_pose_full()
        vx, vy, wz = self.get_base_velocity_full()
        values: dict[str, Any] = {
            "timestamp": timestamp,
            "phase": phase,
            "base_pos_x": pose["x"],
            "base_pos_y": pose["y"],
            "base_pos_z": pose["z"],
            "base_yaw": pose["yaw"],
            "base_quat_w": pose["quat_wxyz"][0],
            "base_quat_x": pose["quat_wxyz"][1],
            "base_quat_y": pose["quat_wxyz"][2],
            "base_quat_z": pose["quat_wxyz"][3],
            "base_lin_vel_x_body": vx,
            "base_lin_vel_y_body": vy,
            "base_ang_vel_z_body": wz,
            "cmd_vx": self._command[0],
            "cmd_vy": self._command[1],
            "cmd_wz": self._command[2],
        }
        if len(self.arm_joint_ids) == 6:
            arm_values = self.robot.data.joint_pos[0, self.arm_joint_ids]
            values.update({f"arm_joint{index + 1}": _item(value) for index, value in enumerate(arm_values)})
        if len(self.gripper_joint_ids) == 2:
            gripper_values = self.robot.data.joint_pos[0, self.gripper_joint_ids]
            values["gripper"] = sum(_item(value) for value in gripper_values) / 2.0
        return values

    def diagnostics(self) -> dict[str, float]:
        """Return compact locomotion-policy diagnostics for smoke-test logs."""

        roll, pitch = _quat_to_roll_pitch(self.robot.data.root_quat_w[0])
        values = {
            "base_z": _item(self.robot.data.root_pos_w[0][2]),
            "base_roll": roll,
            "base_pitch": pitch,
            "measured_vx": self.get_base_velocity_full()[0],
            "measured_vy": self.get_base_velocity_full()[1],
            "measured_wz": self.get_base_velocity_full()[2],
            "command_seen_vx": _item(self.base_cmd_term.vel_command_b[0][0]),
            "command_seen_vy": _item(self.base_cmd_term.vel_command_b[0][1]),
            "command_seen_wz": _item(self.base_cmd_term.vel_command_b[0][2]),
        }
        if self._last_actions is not None:
            values["action_abs_max"] = _item(self._last_actions[0].abs().max())
            if len(self.dog_joint_ids) == 12:
                values["dog_action_abs_mean"] = _item(self._last_actions[0, :12].abs().mean())
        try:
            contact_sensor = self.runtime.scene.sensors["contact_forces"]
            contact_forces = contact_sensor.data.net_forces_w[0].norm(dim=-1)
            foot_ids = [index for index, name in enumerate(contact_sensor.body_names) if "foot" in name.lower()]
            nonfoot_ids = [index for index, name in enumerate(contact_sensor.body_names) if "foot" not in name.lower()]
            values["contact_force_max"] = _item(contact_forces.max())
            values["foot_contact_force_max"] = _item(contact_forces[foot_ids].max()) if foot_ids else 0.0
            values["nonfoot_contact_force_max"] = _item(contact_forces[nonfoot_ids].max()) if nonfoot_ids else 0.0
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
            pass
        return values

    def get_front_rgb(self) -> Any | None:
        """Return the head-camera RGB tensor if configured."""

        try:
            return self.runtime.scene["head_camera"].data.output["rgb"][0, :, :, :3]
        except (KeyError, TypeError):
            return None
