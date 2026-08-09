"""Go2-X5 导航使用的 Isaac Lab locomotion-policy adapter。

Isaac Lab、Torch 和 GPU runtime 均保持延迟导入，避免纯导航测试依赖仿真环境。
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


def _shape_tuple(value: Any) -> tuple[int, ...]:
    """把 tensor/array 的 shape 转为可 JSON 化的整数元组。"""

    return tuple(int(item) for item in getattr(value, "shape", ()))


def _flat_float_values(value: Any) -> list[float]:
    """把单环境 tensor/array 展平为 Python float，避免遥测持有 GPU 引用。"""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()

    flattened: list[float] = []

    def _append(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                _append(child)
            return
        flattened.append(float(item))

    _append(value)
    return flattened


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    """给观测项生成完整有限值统计，并显式报告非有限值数量。"""

    finite_values = [value for value in values if math.isfinite(value)]
    report: dict[str, Any] = {
        "value_count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
    }
    if finite_values:
        report.update(
            {
                "min": min(finite_values),
                "max": max(finite_values),
                "mean": sum(finite_values) / len(finite_values),
            }
        )
    else:
        report.update({"min": None, "max": None, "mean": None})
    return report


class Go2LocomotionAdapter:
    """把机体系速度命令写入 Isaac Lab command-conditioned policy。"""

    def __init__(
        self,
        env: Any,
        policy: Any,
        observations: Any,
        *,
        standing_command_threshold: float = 0.0,
        policy_action_warmup_steps: int = 0,
    ):
        if standing_command_threshold < 0.0:
            raise ValueError("standing_command_threshold 不能为负数。")
        if policy_action_warmup_steps < 0:
            raise ValueError("policy_action_warmup_steps 不能为负数。")
        self.env = env
        self.policy = policy
        self.observations = observations
        self.standing_command_threshold = float(standing_command_threshold)
        self.policy_action_warmup_steps = int(policy_action_warmup_steps)
        self._policy_action_step = 0
        self._policy_action_warmup_scale = 1.0
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
        self._navigation_joint_lock_target = None
        self._navigation_joint_lock_joint_ids: tuple[int, ...] = ()
        self._navigation_joint_lock_joint_names: tuple[str, ...] = ()
        self._command = (0.0, 0.0, 0.0)
        self._effective_command = (0.0, 0.0, 0.0)
        self._command_is_standing = True
        self._arm_joint_target = None
        self._arm_joint_velocity_hold_enabled = False
        self._gripper_joint_target = None
        self._last_actions = None
        self._last_stair_probe_policy_pre_step: dict[str, Any] | None = None

    def _write_root_pose_xyzyaw(self, x: float, y: float, z: float, yaw: float) -> None:
        """把水平 root pose 写入仿真，并清零 root 速度。"""
        import torch

        quat = yaw_to_quat_wxyz(yaw)
        pose = torch.tensor([[x, y, z, *quat]], dtype=torch.float32, device=self.runtime.device)
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.runtime.device)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(velocity)

    def reset_to_pose(self, x: float, y: float, yaw: float) -> None:
        """把 root pose 写入仿真，并清零 root 速度。"""

        current_z = _item(self.robot.data.root_pos_w[0][2])
        self._write_root_pose_xyzyaw(x, y, current_z, yaw)

    def set_base_pose_lock(
        self,
        enabled: bool = True,
        pose_xyyaw: tuple[float, float, float] | None = None,
        pose_xyzyaw: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """锁定 floating base；可用于 manipulation，也可用于 PCT 楼梯漂移。"""

        if enabled:
            if pose_xyzyaw is not None:
                self._base_pose_lock_xyzyaw = (
                    float(pose_xyzyaw[0]),
                    float(pose_xyzyaw[1]),
                    float(pose_xyzyaw[2]),
                    float(pose_xyzyaw[3]),
                )
            else:
                pose = pose_xyyaw if pose_xyyaw is not None else self.get_base_pose()
                z = _item(self.robot.data.root_pos_w[0][2])
                self._base_pose_lock_xyzyaw = (
                    float(pose[0]),
                    float(pose[1]),
                    float(z),
                    float(pose[2]),
                )
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

    def _dog_joint_target_tensor(
        self,
        dog_joint_target: Any | None,
        dog_joint_names: Any | None,
    ) -> Any:
        """把外部传入的四足站立姿态转换为当前设备上的张量。"""

        if dog_joint_names is not None:
            names = tuple(str(name) for name in dog_joint_names)
            if names != tuple(DOG_JOINT_NAMES):
                return {
                    "error": "dog_joint_names_mismatch",
                    "expected": list(DOG_JOINT_NAMES),
                    "received": list(names),
                }
        if dog_joint_target is None:
            return (
                self.robot.data.joint_pos[0, self.dog_joint_ids]
                .detach()
                .clone()
                .reshape(1, -1)
            )
        import torch

        target = torch.as_tensor(
            dog_joint_target,
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape(1, -1)
        if target.shape[1] != len(DOG_JOINT_NAMES):
            return {
                "error": "dog_joint_target_count_mismatch",
                "target_count": int(target.shape[1]),
                "joint_count": len(DOG_JOINT_NAMES),
            }
        return target.detach().clone()

    def set_support_joint_lock(
        self,
        enabled: bool = True,
        *,
        dog_joint_target: Any | None = None,
        dog_joint_names: Any | None = None,
    ) -> dict[str, Any]:
        """冻结当前四足支撑关节姿态，不直接改写关节状态。"""

        if enabled:
            if len(self.dog_joint_ids) != len(DOG_JOINT_NAMES):
                self._dog_joint_lock_target = None
            else:
                target = self._dog_joint_target_tensor(
                    dog_joint_target,
                    dog_joint_names,
                )
                if isinstance(target, dict):
                    self._dog_joint_lock_target = None
                    return {
                        "enabled": False,
                        "reason": target.get("error", "dog_joint_target_invalid"),
                        **target,
                        "uses_direct_joint_state": False,
                    }
                self._dog_joint_lock_target = target
        else:
            self._dog_joint_lock_target = None
        return {
            "enabled": self._dog_joint_lock_target is not None,
            "joint_names": list(DOG_JOINT_NAMES) if self._dog_joint_lock_target is not None else [],
            "joint_ids": [int(index) for index in self.dog_joint_ids] if self._dog_joint_lock_target is not None else [],
            "action_indices": list(self.dog_action_indices or []),
            "target_positions": (
                [
                    float(value)
                    for value in self._dog_joint_lock_target.reshape(-1)
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if self._dog_joint_lock_target is not None
                else []
            ),
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

    def set_navigation_joint_pose_lock(
        self,
        enabled: bool = True,
        *,
        arm_joint_target: Any | None = None,
        dog_joint_target: Any | None = None,
        dog_joint_names: Any | None = None,
    ) -> dict[str, Any]:
        """楼梯漂移期间锁住腿部和机械臂姿态，避免 root 漂移时关节被拉歪。"""

        if not enabled:
            self._navigation_joint_lock_target = None
            self._navigation_joint_lock_joint_ids = ()
            self._navigation_joint_lock_joint_names = ()
            return {
                "enabled": False,
                "joint_names": [],
                "joint_ids": [],
                "uses_direct_joint_state": False,
            }
        if len(self.dog_joint_ids) != len(DOG_JOINT_NAMES):
            self._navigation_joint_lock_target = None
            self._navigation_joint_lock_joint_ids = ()
            self._navigation_joint_lock_joint_names = ()
            return {
                "enabled": False,
                "reason": "dog_joint_id_count_mismatch",
                "joint_ids": [int(index) for index in self.dog_joint_ids],
                "uses_direct_joint_state": False,
            }

        joint_ids: list[int] = [int(index) for index in self.dog_joint_ids]
        joint_names: list[str] = list(DOG_JOINT_NAMES)
        dog_target = self._dog_joint_target_tensor(dog_joint_target, dog_joint_names)
        if isinstance(dog_target, dict):
            return {
                "enabled": False,
                "reason": dog_target.get("error", "dog_joint_target_invalid"),
                **dog_target,
                "uses_direct_joint_state": False,
            }
        target_parts = [dog_target]

        if len(self.arm_joint_ids) == len(ARM_JOINT_NAMES):
            joint_ids.extend(int(index) for index in self.arm_joint_ids)
            joint_names.extend(ARM_JOINT_NAMES)
            if arm_joint_target is None:
                arm_target = (
                    self.robot.data.joint_pos[0, self.arm_joint_ids]
                    .detach()
                    .clone()
                    .reshape(1, -1)
                )
            else:
                import torch

                arm_target = torch.as_tensor(
                    arm_joint_target,
                    dtype=torch.float32,
                    device=self.runtime.device,
                ).reshape(1, -1)
                if arm_target.shape[1] != len(ARM_JOINT_NAMES):
                    return {
                        "enabled": False,
                        "reason": "arm_joint_target_count_mismatch",
                        "target_count": int(arm_target.shape[1]),
                        "joint_count": len(ARM_JOINT_NAMES),
                        "uses_direct_joint_state": False,
                    }
            target_parts.append(arm_target)

        if len(self.gripper_joint_ids) == len(GRIPPER_JOINT_NAMES):
            joint_ids.extend(int(index) for index in self.gripper_joint_ids)
            joint_names.extend(GRIPPER_JOINT_NAMES)
            target_parts.append(
                self.robot.data.joint_pos[0, self.gripper_joint_ids]
                .detach()
                .clone()
                .reshape(1, -1)
            )

        self._navigation_joint_lock_target = torch.cat(target_parts, dim=1).detach().clone()
        self._navigation_joint_lock_joint_ids = tuple(joint_ids)
        self._navigation_joint_lock_joint_names = tuple(joint_names)
        return {
            "enabled": True,
            "joint_names": list(self._navigation_joint_lock_joint_names),
            "joint_ids": list(self._navigation_joint_lock_joint_ids),
            "target_positions": [
                float(value)
                for value in self._navigation_joint_lock_target.reshape(-1)
                .detach()
                .cpu()
                .tolist()
            ],
            "uses_direct_joint_state": True,
            "lock_mode": "stair_float_full_body_pose",
        }

    def apply_navigation_joint_pose_lock(self) -> dict[str, Any]:
        """把楼梯漂移锁定姿态写入 articulation target 和 joint state。"""

        if self._navigation_joint_lock_target is None:
            return {
                "applied": False,
                "reason": "navigation_joint_pose_lock_disabled",
            }
        import torch

        target = self._navigation_joint_lock_target.to(
            device=self.runtime.device,
            dtype=torch.float32,
        )
        joint_ids = list(self._navigation_joint_lock_joint_ids)
        velocity = torch.zeros_like(target)
        self.robot.set_joint_position_target(target, joint_ids=joint_ids)
        self.robot.set_joint_velocity_target(velocity, joint_ids=joint_ids)
        self.robot.write_joint_state_to_sim(target, velocity, joint_ids=joint_ids)
        return {
            "applied": True,
            "joint_names": list(self._navigation_joint_lock_joint_names),
            "joint_ids": joint_ids,
            "target_positions": [
                float(value) for value in target.reshape(-1).detach().cpu().tolist()
            ],
            "uses_direct_joint_state": True,
            "lock_mode": "stair_float_full_body_pose",
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
        """返回实测机体系 ``vx、wz``。"""

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

    def get_effective_base_command(self) -> tuple[float, float, float]:
        """返回经过站立死区处理、实际写入 policy observation 的速度命令。"""

        return tuple(float(value) for value in self._effective_command)

    def reset_policy_warmup(self) -> None:
        """重置每个 episode 的 locomotion action 渐入状态。"""

        self._policy_action_step = 0
        self._policy_action_warmup_scale = (
            1.0 if self.policy_action_warmup_steps <= 0 else 0.0
        )

    def set_arm_joint_target(
        self,
        target: Any | None,
        *,
        hold_velocity: bool = False,
    ) -> None:
        """在 locomotion policy step 时可选保持机械臂关节目标。"""

        self._arm_joint_target = target
        self._arm_joint_velocity_hold_enabled = bool(hold_velocity and target is not None)

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
        velocity_target_written = False
        if getattr(self, "_arm_joint_velocity_hold_enabled", False):
            velocity = torch.zeros_like(target)
            self.robot.set_joint_velocity_target(velocity, joint_ids=self.arm_joint_ids)
            velocity_target_written = True
        return {
            "applied": True,
            "joint_names": list(ARM_JOINT_NAMES),
            "joint_ids": [int(index) for index in self.arm_joint_ids],
            "target_positions": [
                float(value) for value in target.reshape(-1).detach().cpu().tolist()
            ],
            # 运动段只写 position target；post-motion hold 才写零速度，避免到位等待时关节继续漂移。
            "control_mode": (
                "position_velocity_target"
                if velocity_target_written
                else "position_target_only"
            ),
            "velocity_target_written": velocity_target_written,
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

    @staticmethod
    def _clip_count_report(values: list[float]) -> dict[str, Any]:
        """统计 height-scan 在 policy clip 边界的饱和值。"""

        count = len(values)
        clipped_low_count = sum(value <= -1.0 + 1.0e-6 for value in values)
        clipped_high_count = sum(value >= 1.0 - 1.0e-6 for value in values)
        return {
            "clip_low_value": -1.0,
            "clip_high_value": 1.0,
            "clipped_low_count": clipped_low_count,
            "clipped_low_ratio": (
                float(clipped_low_count) / count if count else 0.0
            ),
            "clipped_high_count": clipped_high_count,
            "clipped_high_ratio": (
                float(clipped_high_count) / count if count else 0.0
            ),
            "interpretation": (
                "-1 是 policy term clip 下界；需结合 ray_miss_count 区分射线未命中"
            ),
        }

    def _height_scan_front_subset_report(
        self,
        *,
        values: list[float],
        policy_flat_index_start: int | None,
    ) -> dict[str, Any]:
        """按 GridPattern 实际生成顺序提取传感器局部 ``x>=0`` 前区。"""

        report: dict[str, Any] = {
            "available": False,
            "selection_semantics": "height_scanner GridPattern local x >= 0",
            "index_source": "height_scanner.cfg.pattern_cfg.func generated order",
        }
        try:
            scene = self.runtime.scene
            sensors = getattr(scene, "sensors", None)
            if sensors is not None and "height_scanner" in sensors:
                sensor = sensors["height_scanner"]
            else:
                sensor = scene["height_scanner"]
            pattern_cfg = sensor.cfg.pattern_cfg
            pattern_func = pattern_cfg.func
            ray_starts, _ = pattern_func(pattern_cfg, "cpu")
            local_x = _flat_float_values(ray_starts[:, 0])
            local_y = _flat_float_values(ray_starts[:, 1])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            report["unavailable_reason"] = (
                "height_scanner_grid_pattern_unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            return report
        if len(local_x) != len(values):
            report.update(
                {
                    "unavailable_reason": (
                        "height_scan_value_count_does_not_match_grid_pattern"
                    ),
                    "height_scan_value_count": len(values),
                    "grid_ray_count": len(local_x),
                }
            )
            return report

        relative_indices = [
            index for index, x_value in enumerate(local_x)
            if x_value >= -1.0e-9
        ]
        front_values = [values[index] for index in relative_indices]
        global_indices = (
            None
            if policy_flat_index_start is None
            else [
                int(policy_flat_index_start) + index
                for index in relative_indices
            ]
        )
        ray_miss_relative_indices: list[int] | None = None
        try:
            ray_hits = sensor.data.ray_hits_w[0]
            ray_miss_relative_indices = []
            for index in range(int(ray_hits.shape[0])):
                hit = _flat_float_values(ray_hits[index])
                if not hit or not all(math.isfinite(value) for value in hit):
                    ray_miss_relative_indices.append(index)
        except (AttributeError, IndexError, TypeError, ValueError):
            # 老版本 RayCaster 可能不公开 hit tensor；前区仍可按 cfg 可靠提取。
            ray_miss_relative_indices = None

        ray_miss_index_set = (
            set(ray_miss_relative_indices)
            if ray_miss_relative_indices is not None
            else None
        )
        front_miss_count = (
            None
            if ray_miss_index_set is None
            else sum(
                index in ray_miss_index_set
                for index in relative_indices
            )
        )
        ordering = str(getattr(pattern_cfg, "ordering", "unknown"))
        x_count = len({round(value, 9) for value in local_x})
        y_count = len({round(value, 9) for value in local_y})
        inner_axis = "x" if ordering == "xy" else "y"
        outer_axis = "y" if ordering == "xy" else "x"
        inner_count = x_count if inner_axis == "x" else y_count
        outer_count = y_count if outer_axis == "y" else x_count
        report.update(
            {
                "available": True,
                "unavailable_reason": None,
                "pattern": {
                    "type": type(pattern_cfg).__name__,
                    "size_xy_m": [
                        float(value) for value in getattr(pattern_cfg, "size", ())
                    ],
                    "resolution_m": float(
                        getattr(pattern_cfg, "resolution", 0.0)
                    ),
                    "ordering": ordering,
                    "x_sample_count": x_count,
                    "y_sample_count": y_count,
                    "flatten_inner_axis": inner_axis,
                    "flatten_outer_axis": outer_axis,
                    "flatten_shape_outer_inner": [outer_count, inner_count],
                    "flatten_order_note": (
                        f"ordering={ordering}: outer={outer_axis}({outer_count}), "
                        f"inner={inner_axis}({inner_count}); indices 来自 cfg.func 实际输出"
                    ),
                    "ray_count": len(local_x),
                },
                "relative_height_scan_indices": relative_indices,
                "policy_flat_indices": global_indices,
                "value_count": len(front_values),
                "values": front_values,
                "statistics": _numeric_summary(front_values),
                "clip_diagnostics": self._clip_count_report(front_values),
                "ray_miss_count": (
                    None
                    if ray_miss_relative_indices is None
                    else len(ray_miss_relative_indices)
                ),
                "front_ray_miss_count": front_miss_count,
                "ray_miss_relative_indices": ray_miss_relative_indices,
                "ray_miss_available": ray_miss_relative_indices is not None,
            }
        )
        return report

    def _policy_observation_term_report(
        self,
        *,
        include_selected_values: bool,
        values_are_exact_policy_input: bool = False,
    ) -> dict[str, Any]:
        """按 ObservationManager 的 term 名称和维度解析真实 policy 输入。

        当前 checkpoint 的 policy group 是一维拼接向量。这里仍先读取 manager
        公开的 ``active_terms/group_obs_term_dim``，只有确认每项均为一维后才给出
        连续 flat index；不把任何历史 observation 偏移写死在 adapter 中。
        """

        selected_names = ("velocity_commands", "height_scan")
        report: dict[str, Any] = {
            "available": False,
            "group": "policy",
            "layout_source": (
                "observation_manager.active_terms+group_obs_term_dim"
            ),
            "values_are_exact_policy_input": bool(
                values_are_exact_policy_input
            ),
            "values_after_term_noise_clip_scale": True,
            "selected_terms": {},
        }
        try:
            manager = self.runtime.observation_manager
            active_terms = manager.active_terms
            term_dims_by_group = manager.group_obs_term_dim
            concatenate_by_group = manager.group_obs_concatenate
            term_names = list(active_terms["policy"])
            raw_term_dims = list(term_dims_by_group["policy"])
            concatenated = bool(concatenate_by_group["policy"])
            policy_observation = self.observations["policy"]
        except (AttributeError, KeyError, TypeError) as exc:
            report["unavailable_reason"] = (
                "observation_manager_layout_unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            return report

        term_dims: list[tuple[int, ...]] = []
        try:
            for raw_dims in raw_term_dims:
                if isinstance(raw_dims, int):
                    term_dims.append((int(raw_dims),))
                else:
                    term_dims.append(tuple(int(value) for value in raw_dims))
        except (TypeError, ValueError) as exc:
            report["unavailable_reason"] = (
                "observation_term_dimension_invalid: "
                f"{type(exc).__name__}: {exc}"
            )
            return report

        report.update(
            {
                "policy_tensor_shape": list(_shape_tuple(policy_observation)),
                "group_concatenated": concatenated,
                "term_order": term_names,
                "term_shapes": [list(dims) for dims in term_dims],
            }
        )
        if len(term_names) != len(term_dims):
            report["unavailable_reason"] = (
                "observation_term_name_dimension_count_mismatch"
            )
            return report

        selected_reports: dict[str, Any] = {}
        term_layout: list[dict[str, Any]] = []
        if concatenated:
            # 对多维 term 沿任意 axis 拼接时，reshape 后未必仍是连续区间；此时
            # 宁可显式降级，也不伪造 flat index。当前 locomotion policy 的每项
            # 都是一维，因而能够可靠给出 [start, end) 范围。
            if any(len(dims) != 1 for dims in term_dims):
                report["unavailable_reason"] = (
                    "concatenated_policy_contains_non_vector_term"
                )
                return report
            try:
                flat_values = _flat_float_values(policy_observation[0])
            except (IndexError, TypeError, ValueError) as exc:
                report["unavailable_reason"] = (
                    "policy_tensor_flatten_failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                return report
            expected_count = sum(dims[0] for dims in term_dims)
            report["policy_flat_value_count"] = len(flat_values)
            report["expected_flat_value_count"] = expected_count
            if len(flat_values) != expected_count:
                report["unavailable_reason"] = (
                    "policy_tensor_length_does_not_match_manager_layout"
                )
                return report
            start = 0
            for name, dims in zip(term_names, term_dims):
                end = start + dims[0]
                layout = {
                    "name": name,
                    "shape": list(dims),
                    "policy_flat_index_start": start,
                    "policy_flat_index_end_exclusive": end,
                }
                term_layout.append(layout)
                if name in selected_names:
                    values = flat_values[start:end]
                    selected = {
                        **layout,
                        "statistics": _numeric_summary(values),
                    }
                    if include_selected_values:
                        selected["values"] = values
                    if name == "height_scan":
                        selected["clip_diagnostics"] = (
                            self._clip_count_report(values)
                        )
                        if include_selected_values:
                            selected["front_subset"] = (
                                self._height_scan_front_subset_report(
                                    values=values,
                                    policy_flat_index_start=start,
                                )
                            )
                    selected_reports[name] = selected
                start = end
        else:
            if not isinstance(policy_observation, dict) and not hasattr(
                policy_observation,
                "keys",
            ):
                report["unavailable_reason"] = (
                    "nonconcatenated_policy_observation_is_not_mapping"
                )
                return report
            for name, dims in zip(term_names, term_dims):
                layout = {
                    "name": name,
                    "shape": list(dims),
                    "policy_flat_index_start": None,
                    "policy_flat_index_end_exclusive": None,
                }
                term_layout.append(layout)
                if name not in selected_names:
                    continue
                try:
                    values = _flat_float_values(policy_observation[name][0])
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    selected_reports[name] = {
                        **layout,
                        "available": False,
                        "unavailable_reason": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                    continue
                selected = {
                    **layout,
                    "statistics": _numeric_summary(values),
                }
                if include_selected_values:
                    selected["values"] = values
                if name == "height_scan":
                    selected["clip_diagnostics"] = self._clip_count_report(
                        values
                    )
                    if include_selected_values:
                        selected["front_subset"] = (
                            self._height_scan_front_subset_report(
                                values=values,
                                policy_flat_index_start=None,
                            )
                        )
                selected_reports[name] = selected

        missing_terms = [
            name for name in selected_names if name not in selected_reports
        ]
        report["term_layout"] = term_layout
        report["selected_terms"] = selected_reports
        report["missing_selected_terms"] = missing_terms
        report["available"] = not missing_terms and all(
            term.get("available", True)
            for term in selected_reports.values()
        )
        report["unavailable_reason"] = (
            None
            if report["available"]
            else "required_policy_observation_term_unavailable"
        )
        return report

    def _dog_action_probe_report(
        self,
        *,
        raw_policy_actions: Any,
        submitted_actions: Any,
    ) -> dict[str, Any]:
        """提取与当前关节名映射对应的 12 维腿部 action。"""

        indices = list(self.dog_action_indices or [])
        report: dict[str, Any] = {
            "available": False,
            "dog_joint_names": list(DOG_JOINT_NAMES),
            "dog_action_indices": [int(index) for index in indices],
            "raw_policy_action_shape": list(_shape_tuple(raw_policy_actions)),
            "submitted_action_shape": list(_shape_tuple(submitted_actions)),
            "submitted_action_semantics": (
                "clip/warmup/optional_joint_override 后传给 ActionManager 的 action"
            ),
        }
        if len(indices) != len(DOG_JOINT_NAMES):
            report["unavailable_reason"] = "dog_action_index_count_mismatch"
            return report
        try:
            raw_values = [
                _item(raw_policy_actions[0, index]) for index in indices
            ]
            submitted_values = [
                _item(submitted_actions[0, index]) for index in indices
            ]
        except (IndexError, TypeError, ValueError) as exc:
            report["unavailable_reason"] = (
                f"dog_action_extract_failed: {type(exc).__name__}: {exc}"
            )
            return report
        report.update(
            {
                "available": True,
                "unavailable_reason": None,
                "raw_policy_dog_action": raw_values,
                "submitted_dog_action": submitted_values,
                "raw_statistics": _numeric_summary(raw_values),
                "submitted_statistics": _numeric_summary(submitted_values),
            }
        )
        return report

    def _capture_stair_probe_policy_pre_step(
        self,
        *,
        raw_policy_actions: Any,
        submitted_actions: Any,
    ) -> None:
        """冻结一次 policy 推理的输入、命令 buffer 与腿部 action。"""

        try:
            command_readback = [
                _item(value) for value in self.base_cmd_term.vel_command_b[0]
            ]
            command_report = {
                "available": len(command_readback) >= 3,
                "requested_adapter_command": [
                    float(value) for value in self._command
                ],
                "written_effective_command": [
                    float(value) for value in self._effective_command
                ],
                "command_buffer_attribute": "base_velocity.vel_command_b",
                "command_buffer_readback": command_readback,
                "write_readback_match": (
                    len(command_readback) >= 3
                    and all(
                        math.isclose(
                            float(written),
                            float(readback),
                            rel_tol=0.0,
                            abs_tol=1.0e-6,
                        )
                        for written, readback in zip(
                            self._effective_command,
                            command_readback[:3],
                        )
                    )
                ),
            }
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            command_report = {
                "available": False,
                "unavailable_reason": (
                    f"command_buffer_readback_failed: {type(exc).__name__}: {exc}"
                ),
            }

        observation_report = self._policy_observation_term_report(
            include_selected_values=True,
            values_are_exact_policy_input=True,
        )
        try:
            velocity_values = observation_report["selected_terms"][
                "velocity_commands"
            ]["values"]
            command_readback = command_report["command_buffer_readback"]
            command_report["policy_velocity_commands"] = list(
                velocity_values
            )
            command_report["policy_observation_matches_command_buffer"] = (
                len(velocity_values) >= 3
                and len(command_readback) >= 3
                and all(
                    math.isclose(
                        float(observed),
                        float(readback),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    for observed, readback in zip(
                        velocity_values[:3],
                        command_readback[:3],
                    )
                )
            )
        except (KeyError, TypeError, ValueError):
            command_report["policy_observation_matches_command_buffer"] = None
        action_report = self._dog_action_probe_report(
            raw_policy_actions=raw_policy_actions,
            submitted_actions=submitted_actions,
        )
        self._last_stair_probe_policy_pre_step = {
            "available": bool(
                command_report.get("available")
                and observation_report.get("available")
                and action_report.get("available")
            ),
            "capture_phase": (
                "policy inference 前写 command；同一推理输入与推理后 submitted action"
            ),
            "command_buffer": command_report,
            "policy_observation": observation_report,
            "dog_action": action_report,
            "policy_action_step": int(self._policy_action_step),
            "policy_action_warmup_scale": float(
                self._policy_action_warmup_scale
            ),
        }

    def get_stair_probe_policy_pre_step(self) -> dict[str, Any]:
        """返回最近一次固定命令探针冻结的 pre-step 遥测。"""

        if self._last_stair_probe_policy_pre_step is None:
            return {
                "available": False,
                "unavailable_reason": "stair_probe_capture_not_requested",
            }
        return self._last_stair_probe_policy_pre_step

    def _contact_force_probe_report(self) -> dict[str, Any]:
        """读取最后一个 physics 子步的逐刚体净接触力与足端状态。"""

        try:
            contact_sensor = self.runtime.scene.sensors["contact_forces"]
            body_names = [str(name) for name in contact_sensor.body_names]
            force_vectors = contact_sensor.data.net_forces_w[0]
            current_air_time = getattr(
                contact_sensor.data,
                "current_air_time",
                None,
            )
            current_contact_time = getattr(
                contact_sensor.data,
                "current_contact_time",
                None,
            )
            if len(body_names) != int(force_vectors.shape[0]):
                return {
                    "available": False,
                    "unavailable_reason": "contact_body_name_force_count_mismatch",
                    "body_name_count": len(body_names),
                    "force_count": int(force_vectors.shape[0]),
                }
            contacts = []
            for index, name in enumerate(body_names):
                vector = _flat_float_values(force_vectors[index])
                norm = math.sqrt(sum(value * value for value in vector))
                contact = {
                    "body_name": name,
                    "net_force_world_xyz_n": vector,
                    "net_force_norm_n": norm,
                }
                if current_air_time is not None:
                    try:
                        contact["current_air_time_s"] = _item(
                            current_air_time[0, index]
                        )
                    except (IndexError, TypeError, ValueError):
                        contact["current_air_time_s"] = None
                if current_contact_time is not None:
                    try:
                        contact["current_contact_time_s"] = _item(
                            current_contact_time[0, index]
                        )
                    except (IndexError, TypeError, ValueError):
                        contact["current_contact_time_s"] = None
                contacts.append(contact)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            return {
                "available": False,
                "unavailable_reason": (
                    f"contact_sensor_unavailable: {type(exc).__name__}: {exc}"
                ),
            }

        foot_contacts = [
            contact for contact in contacts
            if "foot" in contact["body_name"].lower()
        ]
        foot_state_report: dict[str, Any]
        try:
            robot_body_names = [str(name) for name in self.robot.body_names]
            robot_body_index = {
                name: index for index, name in enumerate(robot_body_names)
            }
            foot_states = []
            missing_foot_bodies = []
            for contact in foot_contacts:
                name = contact["body_name"]
                if name not in robot_body_index:
                    missing_foot_bodies.append(name)
                    continue
                body_index = robot_body_index[name]
                foot_states.append(
                    {
                        "body_name": name,
                        "position_world_xyz_m": _flat_float_values(
                            self.robot.data.body_pos_w[0, body_index]
                        ),
                        "linear_velocity_world_xyz_mps": _flat_float_values(
                            self.robot.data.body_lin_vel_w[0, body_index]
                        ),
                    }
                )
            foot_state_report = {
                "available": not missing_foot_bodies,
                "feet": foot_states,
                "missing_body_names": missing_foot_bodies,
                "unavailable_reason": (
                    None
                    if not missing_foot_bodies
                    else "contact_foot_not_found_in_robot_body_names"
                ),
            }
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            foot_state_report = {
                "available": False,
                "feet": [],
                "unavailable_reason": (
                    f"foot_body_state_unavailable: {type(exc).__name__}: {exc}"
                ),
            }

        return {
            "available": True,
            "unavailable_reason": None,
            "force_frame": "world",
            "force_sample_semantics": (
                "完整 control step 最后一个 physics 子步后 ContactSensor 净力"
            ),
            "all_body_contacts": contacts,
            "foot_contacts": foot_contacts,
            "foot_states": foot_state_report,
            "contact_force_max_n": max(
                (contact["net_force_norm_n"] for contact in contacts),
                default=0.0,
            ),
            "foot_contact_force_max_n": max(
                (contact["net_force_norm_n"] for contact in foot_contacts),
                default=0.0,
            ),
        }

    def capture_stair_probe_post_step(self) -> dict[str, Any]:
        """捕获完整 decimation 后的 root、腿部关节、足端和接触状态。"""

        report: dict[str, Any] = {
            "available": True,
            "capture_phase": (
                "完整 control step 的全部 physics 子步与 scene.update 完成后"
            ),
        }
        try:
            pose = self.get_base_pose_full()
            report["root_pose_world"] = {
                "position_xyz_m": [pose["x"], pose["y"], pose["z"]],
                "quaternion_wxyz": list(pose["quat_wxyz"]),
                "yaw_rad": pose["yaw"],
            }
            report["root_velocity_body_vx_vy_wz"] = list(
                self.get_base_velocity_full()
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            report["available"] = False
            report["root_state_unavailable_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )

        if len(self.dog_joint_ids) == len(DOG_JOINT_NAMES):
            try:
                dog_joint_state = {
                    "available": True,
                    "joint_names": list(DOG_JOINT_NAMES),
                    "joint_ids": [int(index) for index in self.dog_joint_ids],
                    "positions_rad": _flat_float_values(
                        self.robot.data.joint_pos[0, self.dog_joint_ids]
                    ),
                    "velocities_rad_s": _flat_float_values(
                        self.robot.data.joint_vel[0, self.dog_joint_ids]
                    ),
                }
                try:
                    dog_joint_state["applied_torque_nm"] = (
                        _flat_float_values(
                            self.robot.data.applied_torque[
                                0, self.dog_joint_ids
                            ]
                        )
                    )
                    dog_joint_state["applied_torque_available"] = True
                    dog_joint_state["applied_torque_unavailable_reason"] = None
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    dog_joint_state["applied_torque_available"] = False
                    dog_joint_state["applied_torque_unavailable_reason"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                try:
                    dog_joint_state["position_targets_rad"] = (
                        _flat_float_values(
                            self.robot.data.joint_pos_target[
                                0, self.dog_joint_ids
                            ]
                        )
                    )
                    dog_joint_state["position_targets_available"] = True
                    dog_joint_state["position_targets_unavailable_reason"] = None
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    dog_joint_state["position_targets_available"] = False
                    dog_joint_state["position_targets_unavailable_reason"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                report["dog_joint_state"] = dog_joint_state
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                report["dog_joint_state"] = {
                    "available": False,
                    "unavailable_reason": f"{type(exc).__name__}: {exc}",
                }
        else:
            report["dog_joint_state"] = {
                "available": False,
                "unavailable_reason": "dog_joint_id_count_mismatch",
                "joint_ids": [int(index) for index in self.dog_joint_ids],
            }
        report["contacts"] = self._contact_force_probe_report()
        dog_state = report.get("dog_joint_state", {})
        contacts = report.get("contacts", {})
        foot_states = (
            contacts.get("foot_states", {})
            if isinstance(contacts, dict)
            else {}
        )
        component_availability = {
            "root_state": "root_pose_world" in report,
            "dog_joint_state": bool(
                isinstance(dog_state, dict)
                and dog_state.get("available") is True
            ),
            "applied_torque": bool(
                isinstance(dog_state, dict)
                and dog_state.get("applied_torque_available") is True
            ),
            "position_targets": bool(
                isinstance(dog_state, dict)
                and dog_state.get("position_targets_available") is True
            ),
            "contacts": bool(
                isinstance(contacts, dict)
                and contacts.get("available") is True
            ),
            "foot_states": bool(
                isinstance(foot_states, dict)
                and foot_states.get("available") is True
            ),
        }
        report["component_availability"] = component_availability
        report["available"] = all(component_availability.values())
        report["unavailable_reason"] = (
            None
            if report["available"]
            else "required_post_step_component_unavailable"
        )
        return report

    def compute_policy_action(
        self,
        *,
        refresh_observations: bool = True,
        capture_stair_probe_telemetry: bool = False,
    ) -> Any:
        """写入速度命令并执行策略推理，但不推进仿真。"""

        import torch

        command = torch.tensor([self._command], dtype=torch.float32, device=self.base_cmd_term.device)
        command_magnitude = torch.max(torch.abs(command), dim=1).values
        standing_threshold = max(self.standing_command_threshold, 1.0e-6)
        is_standing = command_magnitude <= standing_threshold
        effective_command = command.clone()
        effective_command[is_standing] = 0.0
        self.base_cmd_term.vel_command_b[:] = effective_command
        self._effective_command = tuple(
            _item(value) for value in effective_command[0]
        )
        self._command_is_standing = bool(is_standing[0].item())
        if hasattr(self.base_cmd_term, "is_heading_env"):
            self.base_cmd_term.is_heading_env[:] = False
        if hasattr(self.base_cmd_term, "is_standing_env"):
            self.base_cmd_term.is_standing_env[:] = is_standing
        if hasattr(self.base_cmd_term, "heading_target"):
            self.base_cmd_term.heading_target[:] = 0.0
        if self.arm_term is not None:
            if self._arm_joint_target is not None:
                arm_target = torch.as_tensor(
                    self._arm_joint_target,
                    dtype=torch.float32,
                    device=self.arm_term.command_buffer.device,
                ).reshape(1, -1)
                self.arm_term.command_buffer[:, : arm_target.shape[1]] = arm_target
            # 无外部目标时保留 command manager 已按
            # use_default_offset=true 采样出的训练默认姿态。把 buffer 强制清零
            # 会将 pct_multifloor 的 arm2/arm3 从 0.3/0.5 拉到 0，改变质心与观测分布。

        with torch.inference_mode():
            # 新 pipeline 的导航阶段不会开启这些锁；保留调用是为了兼容旧 manipulation 路径。
            self._apply_base_pose_lock()
            self._apply_support_joint_lock()
            self._apply_gripper_joint_target()
            if refresh_observations:
                self.observations = self.env.get_observations()
            actions = self.policy(self.observations)
            raw_policy_actions = (
                actions.detach().clone()
                if capture_stair_probe_telemetry
                else None
            )
            clip_actions = getattr(self.env, "clip_actions", None)
            if clip_actions is not None:
                actions = torch.clamp(actions, -clip_actions, clip_actions)
            if self.policy_action_warmup_steps > 0:
                # reset 后从默认关节姿态平滑接管，避免首帧 policy target 阶跃
                # 在不规则碰撞网格上产生明显弹跳和侧滑。
                self._policy_action_warmup_scale = min(
                    1.0,
                    float(self._policy_action_step + 1)
                    / float(self.policy_action_warmup_steps),
                )
                actions = actions * self._policy_action_warmup_scale
            else:
                self._policy_action_warmup_scale = 1.0
            self._policy_action_step += 1
            # 先裁剪 locomotion policy 输出，再写入 cuRobo 直接关节目标。
            # Foundation 配置中 arm action scale 只有 0.10；如果 override 后再 clip，
            # 1 rad 级别的机械臂目标会被等效压成约 0.1 rad，表现为 pick 阶段原地等待。
            actions = self._override_arm_actions(actions)
            self._last_actions = actions.detach()
            if capture_stair_probe_telemetry:
                try:
                    self._capture_stair_probe_policy_pre_step(
                        raw_policy_actions=raw_policy_actions,
                        submitted_actions=actions,
                    )
                except Exception as exc:
                    # 遥测绝不能中断真实 policy 控制；失败必须留下可审计原因。
                    self._last_stair_probe_policy_pre_step = {
                        "available": False,
                        "unavailable_reason": (
                            "stair_probe_pre_step_capture_failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
            else:
                self._last_stair_probe_policy_pre_step = None
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

    def diagnostics(self) -> dict[str, Any]:
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
            "standing_command_threshold": self.standing_command_threshold,
            "command_is_standing": self._command_is_standing,
            "policy_action_step": self._policy_action_step,
            "policy_action_warmup_steps": self.policy_action_warmup_steps,
            "policy_action_warmup_scale": self._policy_action_warmup_scale,
        }
        if self._last_actions is not None:
            values["action_abs_max"] = _item(self._last_actions[0].abs().max())
            if len(self.dog_action_indices or []) == len(DOG_JOINT_NAMES):
                dog_indices = self.dog_action_indices or []
                dog_actions = self._last_actions[0, dog_indices]
                values["dog_action_abs_mean"] = _item(
                    dog_actions.abs().mean()
                )
        # 常规诊断同样从 ObservationManager 元数据解析，但不逐拍复制完整
        # height-scan values；固定命令 probe 才会保存完整选中项。
        values["policy_observation_report"] = (
            self._policy_observation_term_report(
                include_selected_values=False,
                values_are_exact_policy_input=False,
            )
        )
        try:
            contact_sensor = self.runtime.scene.sensors["contact_forces"]
            contact_forces = contact_sensor.data.net_forces_w[0].norm(dim=-1)
            foot_ids = [index for index, name in enumerate(contact_sensor.body_names) if "foot" in name.lower()]
            nonfoot_ids = [index for index, name in enumerate(contact_sensor.body_names) if "foot" not in name.lower()]
            values["contact_force_max"] = _item(contact_forces.max())
            if foot_ids:
                foot_forces = contact_forces[foot_ids]
                foot_local_index = int(foot_forces.argmax().item())
                values["foot_contact_force_max"] = _item(
                    foot_forces[foot_local_index]
                )
                values["foot_contact_body_name"] = contact_sensor.body_names[
                    foot_ids[foot_local_index]
                ]
            else:
                values["foot_contact_force_max"] = 0.0
                values["foot_contact_body_name"] = None
            if nonfoot_ids:
                nonfoot_forces = contact_forces[nonfoot_ids]
                nonfoot_local_index = int(nonfoot_forces.argmax().item())
                values["nonfoot_contact_force_max"] = _item(
                    nonfoot_forces[nonfoot_local_index]
                )
                values["nonfoot_contact_body_name"] = contact_sensor.body_names[
                    nonfoot_ids[nonfoot_local_index]
                ]
            else:
                values["nonfoot_contact_force_max"] = 0.0
                values["nonfoot_contact_body_name"] = None
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError):
            pass
        return values

    def get_front_rgb(self) -> Any | None:
        """Return the head-camera RGB tensor if configured."""

        try:
            return self.runtime.scene["head_camera"].data.output["rgb"][0, :, :, :3]
        except (KeyError, TypeError):
            return None
