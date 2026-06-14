"""Deterministic simulation double used by the first-phase dry run."""

from __future__ import annotations

import math
import time
from typing import Any

from source.interfaces import EpisodeSpec, RobotAction, SimulationState


def _pose7_from_planar(x: float, y: float, yaw: float) -> tuple[float, float, float, float, float, float, float]:
    return (x, y, 0.35, math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _object_pose7(
    pose: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, float, float, float, float, float, float] | None:
    if pose is None:
        return None
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        x,
        y,
        z,
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


class InMemorySimulationRuntime:
    """Apply deterministic mock effects while preserving the real loop shape."""

    def __init__(self, *, dt: float = 0.05):
        self.dt = float(dt)
        self.step_calls = 0
        self.apply_calls = 0
        self.built = False
        self.closed = False
        self._episode_spec: EpisodeSpec | None = None
        self._action = RobotAction.idle()
        self._robot_pose = _pose7_from_planar(0.0, 0.0, 0.0)
        self._object_pose = None
        self._joint_positions = (0.0,) * 8
        self._metadata: dict[str, Any] = {
            "object_lifted": False,
            "object_placed": False,
            "object_attached": False,
            "last_arm_tracking_report": None,
            "arm_tracking_report": {
                "sample_count": 0,
                "max_abs_error": 0.0,
                "peak_report": None,
            },
            "last_gripper_action_report": None,
        }

    def build(self, episode_spec: EpisodeSpec) -> None:
        if self.closed:
            raise RuntimeError("simulation runtime is closed")
        self.built = True
        self._episode_spec = episode_spec
        self._metadata["scene_usd"] = episode_spec.scene_usd

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        if not self.built:
            raise RuntimeError("stage must be built before reset")
        self._episode_spec = episode_spec
        self._robot_pose = _pose7_from_planar(
            episode_spec.start.x,
            episode_spec.start.y,
            episode_spec.start.yaw,
        )
        self._object_pose = _object_pose7(episode_spec.object_initial_pose)
        self._joint_positions = (0.0,) * 8
        self._metadata = {
            "seed": int(seed),
            "scene_usd": episode_spec.scene_usd,
            "object_pose_debug_after_reset": {
                "available": True,
                "within_tolerance": True,
                "read_only": True,
                "source": "in_memory_runtime",
            },
            "object_lifted": False,
            "object_placed": False,
            "object_attached": False,
            "last_arm_tracking_report": None,
            "arm_tracking_report": {
                "sample_count": 0,
                "max_abs_error": 0.0,
                "peak_report": None,
            },
            "last_gripper_action_report": None,
        }
        self._action = RobotAction.idle(source="reset")

    def read(self) -> SimulationState:
        return SimulationState(
            step_index=self.step_calls,
            timestamp=time.time(),
            robot_root_pose=self._robot_pose,
            robot_root_velocity=(
                self._action.base_velocity[0],
                self._action.base_velocity[1],
                0.0,
                0.0,
                0.0,
                self._action.base_velocity[2],
            ),
            joint_positions=self._joint_positions,
            joint_velocities=(0.0,) * len(self._joint_positions),
            tcp_pose=(0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
            object_pose=self._object_pose,
            object_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            metadata=dict(self._metadata),
        )

    def apply(self, action: RobotAction) -> None:
        self.apply_calls += 1
        self._action = action
        if action.metadata.get("manipulation_base_lock"):
            self._metadata["used_manipulation_base_lock"] = True
            self._metadata["execution_provenance_verified"] = True
        if action.metadata.get("manipulation_support_joint_lock"):
            self._metadata["used_manipulation_support_joint_lock"] = True
            self._metadata["execution_provenance_verified"] = True
        self._record_gripper_target(action)

    def step(self, *, render: bool) -> None:
        del render
        self.step_calls += 1
        if self._action.arm_joint_positions is not None:
            self._joint_positions = tuple(self._action.arm_joint_positions)
            self._record_arm_tracking(self._action)
        effect = self._action.metadata.get("dry_run_effect")
        if effect == "nav_reached":
            goal = self._action.metadata["goal"]
            self._robot_pose = _pose7_from_planar(float(goal[0]), float(goal[1]), float(goal[2]))
        elif effect == "pick_lifted":
            self._metadata["object_lifted"] = True
            self._metadata["object_attached"] = True
            if self._object_pose is not None:
                self._object_pose = (
                    self._object_pose[0],
                    self._object_pose[1],
                    self._object_pose[2] + 0.12,
                    *self._object_pose[3:],
                )
        elif effect == "place_completed":
            target = self._action.metadata.get("target_pose")
            if target is not None:
                self._object_pose = _object_pose7(tuple(target))
            self._metadata["object_placed"] = True
            self._metadata["object_attached"] = False
        self._apply_manipulation_smoke_effects()

    def prepare_object_for_pick(self, episode_spec: EpisodeSpec) -> dict[str, Any]:
        self._object_pose = _object_pose7(episode_spec.object_initial_pose)
        report = {
            "applied": self._object_pose is not None,
            "pose_reset": self._object_pose is not None,
            "velocity_zeroed": True,
            "woken": True,
            "wake_policy": "physx_contact",
            "source": "in_memory_runtime",
        }
        self._metadata["object_prepare_for_pick_report"] = report
        return report

    def pause(self) -> dict[str, Any]:
        report = {"paused": True, "source": "in_memory_runtime"}
        self._metadata["terminal_hold_report"] = report
        return report

    def _record_arm_tracking(self, action: RobotAction) -> None:
        """内存后端精确跟随 target，只用于验证真实 runtime 的诊断合同。"""

        target = tuple(float(value) for value in action.arm_joint_positions or ())
        joint_names = tuple(
            str(name)
            for name in action.metadata.get(
                "arm_joint_names",
                tuple(f"arm_joint{index}" for index in range(1, len(target) + 1)),
            )
        )
        report = {
            "available": True,
            "source": action.source,
            "pipeline_state": action.metadata.get("carry_arm_home_phase"),
            "joint_names": joint_names,
            "target_positions": target,
            "actual_positions": target,
            "position_errors": (0.0,) * len(target),
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "applied_step_index": self.step_calls - 1,
            "sample_step_index": self.step_calls,
        }
        aggregate = dict(self._metadata.get("arm_tracking_report") or {})
        sample_count = int(aggregate.get("sample_count") or 0) + 1
        peak_report = aggregate.get("peak_report") or report
        self._metadata["last_arm_tracking_report"] = report
        self._metadata["arm_tracking_report"] = {
            "sample_count": sample_count,
            "max_abs_error": 0.0,
            "peak_report": peak_report,
            "latest_report": report,
        }

    def _record_gripper_target(self, action: RobotAction) -> None:
        metadata = action.metadata
        positions = metadata.get("gripper_joint_positions")
        if positions is None and action.gripper_command == "close":
            positions = (0.0, 0.0)
        elif positions is None and action.gripper_command == "open":
            positions = (0.04, 0.04)
        if positions is None:
            return
        self._metadata["last_gripper_action_report"] = {
            "target_staged": True,
            "gripper_command": action.gripper_command,
            "gripper_joint_names": tuple(
                metadata.get("gripper_joint_names", ("arm_joint7", "arm_joint8"))
            ),
            "gripper_joint_positions": tuple(float(value) for value in positions),
            "world_step_owned_by_pipeline": True,
        }

    def close(self) -> None:
        self.closed = True

    def _apply_manipulation_smoke_effects(self) -> None:
        metadata = self._action.metadata
        operation = metadata.get("operation")
        segment_name = metadata.get("segment_name")
        event_marker = metadata.get("event_marker")

        if operation == "pick" and event_marker == "gripper_close":
            # smoke 只验证 action 合同；这里不声称物理抓取成功。
            self._metadata["object_attached"] = True
        if (
            operation == "pick"
            and segment_name == "lift_object"
            and self._metadata.get("object_attached")
            and not self._metadata.get("object_lifted")
        ):
            self._metadata["object_lifted"] = True
            if self._object_pose is not None:
                self._object_pose = (
                    self._object_pose[0],
                    self._object_pose[1],
                    self._object_pose[2] + 0.12,
                    *self._object_pose[3:],
                )
        if operation == "place" and event_marker == "gripper_open":
            if self._episode_spec is not None and self._episode_spec.place_target_pose is not None:
                self._object_pose = _object_pose7(self._episode_spec.place_target_pose)
            self._metadata["object_placed"] = True
            self._metadata["object_attached"] = False
