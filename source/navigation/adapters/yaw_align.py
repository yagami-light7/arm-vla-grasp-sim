"""Yaw-alignment helpers for command-conditioned quadruped navigation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class YawAlignConfig:
    """Tunable yaw-alignment command parameters."""

    kp: float = 2.0
    min_wz: float = 0.55
    max_wz: float = 1.00
    activation_vx: float = 0.16
    activation_yaw_error: float = 0.0
    allow_reverse: bool = False


@dataclass(frozen=True)
class YawAlignDiagnostics:
    """Progress diagnostics for yaw-alignment stall detection."""

    sample_count: int
    start_abs_error: float
    current_abs_error: float
    error_reduction: float


def body_goal_forward_projection(
    robot_pose_xyyaw: tuple[float, float, float],
    goal_xy: tuple[float, float],
) -> float:
    """Return goal displacement projected onto the robot body x axis."""

    return body_goal_components(robot_pose_xyyaw, goal_xy)[0]


def body_goal_components(
    robot_pose_xyyaw: tuple[float, float, float],
    goal_xy: tuple[float, float],
) -> tuple[float, float]:
    """Return goal displacement in the robot body frame as ``x, y``."""

    x, y, yaw = robot_pose_xyyaw
    dx = goal_xy[0] - x
    dy = goal_xy[1] - y
    body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return body_x, body_y


def compute_yaw_align_command(
    *,
    yaw_error: float,
    yaw_tolerance: float,
    body_goal_x: float,
    config: YawAlignConfig,
) -> tuple[float, float, float]:
    """Return a body-frame ``vx, vy, wz`` command for terminal yaw alignment."""

    abs_error = abs(yaw_error)
    if abs_error <= yaw_tolerance:
        return 0.0, 0.0, 0.0

    wz_abs = min(config.max_wz, max(config.kp * abs_error, config.min_wz))
    wz = math.copysign(wz_abs, yaw_error)

    vx = 0.0
    if abs_error >= config.activation_yaw_error and config.activation_vx > 0.0:
        if body_goal_x < -1.0e-3:
            vx = -config.activation_vx if config.allow_reverse else 0.0
        else:
            vx = config.activation_vx
    return vx, 0.0, wz


class YawAlignStallDetector:
    """Detect terminal yaw alignment that stops making angular progress."""

    def __init__(self, *, window_steps: int, min_progress_rad: float):
        self.window_steps = max(2, int(window_steps))
        self.min_progress_rad = float(min_progress_rad)
        self._errors: deque[float] = deque(maxlen=self.window_steps)

    def reset(self) -> None:
        """Clear the yaw-error history."""

        self._errors.clear()

    def diagnostics(self) -> YawAlignDiagnostics:
        """Return the current yaw-progress window summary."""

        if not self._errors:
            return YawAlignDiagnostics(0, 0.0, 0.0, 0.0)
        start = self._errors[0]
        current = self._errors[-1]
        return YawAlignDiagnostics(len(self._errors), start, current, start - current)

    def update(self, abs_yaw_error: float) -> tuple[bool, YawAlignDiagnostics]:
        """Append one yaw-error sample and return ``(stalled, diagnostics)``."""

        self._errors.append(abs(float(abs_yaw_error)))
        diagnostics = self.diagnostics()
        if diagnostics.sample_count < self.window_steps:
            return False, diagnostics
        return diagnostics.error_reduction < self.min_progress_rad, diagnostics
