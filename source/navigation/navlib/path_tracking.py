from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathTrackingConfig:
    """Simple waypoint follower used as the first online navigation controller."""

    lookahead_distance: float = 0.6
    waypoint_tolerance: float = 0.2
    goal_tolerance: float = 0.3
    slowdown_distance: float = 1.0
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.0
    linear_gain: float = 0.8
    angular_gain: float = 1.5
    rotate_in_place_angle: float = 0.9


@dataclass(frozen=True)
class PathTrackingDebug:
    target_index: int
    target_point: tuple[float, float]
    distance_to_target: float
    distance_to_goal: float
    heading_error: float
    reached_goal: bool


class PathTrackingController:
    """Tracks a piecewise-linear path with vx + wz commands.

    This is intentionally simpler than DWA so the first navigation loop can be
    validated before adding local obstacle evaluation.
    """

    def __init__(self, path_world: list[tuple[float, float]], config: PathTrackingConfig):
        if len(path_world) < 2:
            raise ValueError("path tracker requires at least two world-frame waypoints.")
        self.path_world = np.asarray(path_world, dtype=np.float64)
        self.config = config
        self.target_index = 1

    def compute_command(self, pose_xyyaw: tuple[float, float, float]) -> tuple[np.ndarray, PathTrackingDebug]:
        x, y, yaw = pose_xyyaw
        position = np.array([x, y], dtype=np.float64)

        goal_vector = self.path_world[-1] - position
        distance_to_goal = float(np.linalg.norm(goal_vector))
        if distance_to_goal <= self.config.goal_tolerance:
            debug = PathTrackingDebug(
                target_index=len(self.path_world) - 1,
                target_point=tuple(float(v) for v in self.path_world[-1]),
                distance_to_target=0.0,
                distance_to_goal=distance_to_goal,
                heading_error=0.0,
                reached_goal=True,
            )
            return np.zeros(3, dtype=np.float32), debug

        self._advance_target(position)
        target = self.path_world[self.target_index]
        delta = target - position
        distance_to_target = float(np.linalg.norm(delta))
        target_heading = math.atan2(delta[1], delta[0])
        heading_error = _wrap_angle(target_heading - yaw)

        if abs(heading_error) > self.config.rotate_in_place_angle:
            linear_velocity = 0.0
        else:
            heading_scale = max(0.0, math.cos(heading_error))
            distance_scale = min(1.0, distance_to_goal / max(self.config.slowdown_distance, 1.0e-6))
            linear_velocity = self.config.linear_gain * distance_to_target * heading_scale * distance_scale
            linear_velocity = min(linear_velocity, self.config.max_linear_velocity)

        angular_velocity = np.clip(
            self.config.angular_gain * heading_error,
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity,
        )

        debug = PathTrackingDebug(
            target_index=self.target_index,
            target_point=(float(target[0]), float(target[1])),
            distance_to_target=distance_to_target,
            distance_to_goal=distance_to_goal,
            heading_error=heading_error,
            reached_goal=False,
        )
        return np.array([linear_velocity, 0.0, angular_velocity], dtype=np.float32), debug

    def _advance_target(self, position: np.ndarray):
        while self.target_index < len(self.path_world) - 1:
            target = self.path_world[self.target_index]
            if np.linalg.norm(target - position) > self.config.waypoint_tolerance:
                break
            self.target_index += 1

        while self.target_index < len(self.path_world) - 1:
            target = self.path_world[self.target_index]
            if np.linalg.norm(target - position) >= self.config.lookahead_distance:
                break
            self.target_index += 1


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
