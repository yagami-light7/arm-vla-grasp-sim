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
        self._action = RobotAction.idle()
        self._robot_pose = _pose7_from_planar(0.0, 0.0, 0.0)
        self._object_pose = None
        self._joint_positions = (0.0,) * 8
        self._metadata: dict[str, Any] = {
            "object_lifted": False,
            "object_placed": False,
            "object_attached": False,
        }

    def build(self, episode_spec: EpisodeSpec) -> None:
        if self.closed:
            raise RuntimeError("simulation runtime is closed")
        self.built = True
        self._metadata["scene_usd"] = episode_spec.scene_usd

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        if not self.built:
            raise RuntimeError("stage must be built before reset")
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
            "object_lifted": False,
            "object_placed": False,
            "object_attached": False,
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

    def step(self, *, render: bool) -> None:
        del render
        self.step_calls += 1
        if self._action.arm_joint_positions is not None:
            self._joint_positions = tuple(self._action.arm_joint_positions)
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

    def close(self) -> None:
        self.closed = True
