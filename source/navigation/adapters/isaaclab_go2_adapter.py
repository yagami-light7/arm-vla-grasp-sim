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
        self.dog_joint_ids, _ = self.robot.find_joints(DOG_JOINT_NAMES, preserve_order=True)
        self.arm_joint_ids, _ = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.gripper_joint_ids, _ = self.robot.find_joints(GRIPPER_JOINT_NAMES, preserve_order=True)
        self.ee_body_ids, _ = self.robot.find_bodies(["arm_link6"])
        self._command = (0.0, 0.0, 0.0)
        self._last_actions = None

    def reset_to_pose(self, x: float, y: float, yaw: float) -> None:
        """Write a root pose and zero root velocity directly to simulation."""

        import torch

        current_z = _item(self.robot.data.root_pos_w[0][2])
        quat = yaw_to_quat_wxyz(yaw)
        pose = torch.tensor([[x, y, current_z, *quat]], dtype=torch.float32, device=self.runtime.device)
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.runtime.device)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(velocity)

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

    def step(self) -> Any:
        """Inject the command term, run policy inference, and advance simulation."""

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
            self.arm_term.command_buffer[:] = 0.0

        self.observations = self.env.get_observations()
        with torch.inference_mode():
            actions = self.policy(self.observations)
            self._last_actions = actions.detach()
            self.observations, _, _, _ = self.env.step(actions)
            if len(self.gripper_joint_ids) == 2:
                closed = torch.zeros((1, 2), dtype=torch.float32, device=self.runtime.device)
                self.robot.set_joint_position_target(closed, joint_ids=self.gripper_joint_ids)
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
            "measured_wz": self.get_base_velocity_full()[2],
            "command_seen_vx": _item(self.base_cmd_term.vel_command_b[0][0]),
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
