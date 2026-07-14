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
class TerminalPoseConfig:
    """Tunable final-pose controller parameters.

    The locomotion policy tracks small translational commands more reliably
    than a pure in-place high-rate yaw command.  This controller therefore
    keeps a small gait command available near the goal and ramps yaw speed down
    as the final heading error becomes small.
    """

    position_tolerance: float = 0.08
    position_acceptance_tolerance: float = 0.18
    yaw_tolerance: float = 0.08
    position_kp: float = 0.8
    max_vx: float = 0.35
    min_vx: float = 0.16
    allow_reverse: bool = True
    lateral_kp: float = 0.8
    lateral_deadband: float = 0.03
    max_vy: float = 0.18
    min_vy: float = 0.0
    yaw_kp: float = 2.0
    yaw_min_wz: float = 0.40
    yaw_max_wz: float = 0.65
    yaw_slowdown_error: float = 0.65
    yaw_slowdown_min_wz: float = 0.20
    yaw_slowdown_max_wz: float = 0.45
    large_yaw_error: float = 1.0
    large_yaw_position_scale: float = 0.45
    gait_activation_vx: float = 0.04
    recovery_yaw_max_wz: float = 0.35
    recovery_gait_vx: float = 0.08
    yaw_polish_gait_vx: float = 0.08
    yaw_polish_min_wz: float = 0.45
    yaw_polish_max_wz: float = 0.55
    # 携物终端段优先使用“转向剩余位置误差 -> 前向收敛”，避免 locomotion
    # policy 在最终 yaw 已对齐、但仍有横向位置误差时持续低速踏步。
    prefer_forward_translation: bool = False
    forward_heading_deadband: float = 0.20
    # 大航向误差时必须先原地转向；非零 vx 与固定 wz 会形成最小转弯半径，
    # 在短程 place goal 周围产生稳定圆周轨迹。
    forward_turn_gait_vx: float = 0.0
    forward_turn_max_wz: float = 0.50


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


def compute_terminal_pose_command(
    *,
    body_goal_x: float,
    body_goal_y: float,
    yaw_error: float,
    distance_to_goal: float,
    config: TerminalPoseConfig,
    recovery: bool = False,
) -> tuple[float, float, float]:
    """Return a body-frame command for final XY + yaw convergence."""

    if (
        config.prefer_forward_translation
        and distance_to_goal > config.position_tolerance
    ):
        return _compute_forward_translation_command(
            body_goal_x=body_goal_x,
            body_goal_y=body_goal_y,
            distance_to_goal=distance_to_goal,
            config=config,
        )

    abs_yaw_error = abs(yaw_error)
    position_scale = 1.0
    if abs_yaw_error >= config.large_yaw_error:
        position_scale = max(0.0, min(1.0, config.large_yaw_position_scale))

    inside_position_acceptance = distance_to_goal <= max(config.position_tolerance, config.position_acceptance_tolerance)

    inside_terminal_position = distance_to_goal <= config.position_tolerance
    inside_large_yaw_arc = inside_position_acceptance and abs_yaw_error >= config.large_yaw_error
    inside_yaw_polish_arc = (
        inside_position_acceptance
        and not inside_terminal_position
        and config.yaw_tolerance < abs_yaw_error < config.large_yaw_error
    )

    if inside_large_yaw_arc:
        vx = _axis_velocity(
            error=body_goal_x,
            kp=config.position_kp,
            max_abs=config.max_vx,
            min_abs=config.min_vx,
            deadband=config.lateral_deadband,
            allow_negative=config.allow_reverse,
            scale=1.0,
        )
        vy = _axis_velocity(
            error=body_goal_y,
            kp=config.lateral_kp,
            max_abs=config.max_vy,
            min_abs=config.min_vy,
            deadband=config.lateral_deadband,
            allow_negative=True,
            scale=1.0,
        )
        if abs(vx) < 1.0e-6 and abs(vy) < 1.0e-6:
            vx = max(config.min_vx, config.yaw_polish_gait_vx)
    elif inside_yaw_polish_arc:
        polish_max_vx = min(config.max_vx, max(config.yaw_polish_gait_vx, config.min_vx))
        polish_max_vy = min(config.max_vy, max(config.yaw_polish_gait_vx, config.min_vy))
        vx = _axis_velocity(
            error=body_goal_x,
            kp=config.position_kp,
            max_abs=polish_max_vx,
            min_abs=min(config.yaw_polish_gait_vx, polish_max_vx),
            deadband=config.lateral_deadband,
            allow_negative=config.allow_reverse,
            scale=1.0,
        )
        vy = _axis_velocity(
            error=body_goal_y,
            kp=config.lateral_kp,
            max_abs=polish_max_vy,
            min_abs=min(max(config.yaw_polish_gait_vx, config.min_vy), polish_max_vy),
            deadband=config.lateral_deadband,
            allow_negative=True,
            scale=1.0,
        )
        if abs(vx) < 1.0e-6 and abs(vy) < 1.0e-6:
            vx = config.yaw_polish_gait_vx
    elif inside_terminal_position or inside_position_acceptance:
        vx = 0.0
        vy = 0.0
    else:
        vx = _axis_velocity(
            error=body_goal_x,
            kp=config.position_kp,
            max_abs=config.max_vx,
            min_abs=config.min_vx,
            deadband=config.lateral_deadband,
            allow_negative=config.allow_reverse,
            scale=position_scale,
        )
        vy = _axis_velocity(
            error=body_goal_y,
            kp=config.lateral_kp,
            max_abs=config.max_vy,
            min_abs=config.min_vy,
            deadband=config.lateral_deadband,
            allow_negative=True,
            scale=position_scale,
        )

    if abs_yaw_error > config.yaw_tolerance and abs(vx) < 1.0e-6 and abs(vy) < 1.0e-6:
        if inside_position_acceptance:
            activation_vx = config.yaw_polish_gait_vx
        else:
            activation_vx = config.recovery_gait_vx if recovery else config.gait_activation_vx
        if activation_vx > 0.0:
            if inside_position_acceptance:
                vx = activation_vx
            elif body_goal_x < -config.lateral_deadband:
                vx = -activation_vx if config.allow_reverse else 0.0
            elif body_goal_x > config.lateral_deadband:
                vx = activation_vx
            else:
                vx = activation_vx

    if abs_yaw_error <= config.yaw_tolerance:
        wz = 0.0
    else:
        max_wz = config.yaw_max_wz
        min_wz = config.yaw_min_wz
        if inside_position_acceptance:
            max_wz = min(max_wz, config.yaw_polish_max_wz)
            min_wz = min(max_wz, max(min_wz, config.yaw_polish_min_wz))
        elif abs_yaw_error <= config.yaw_slowdown_error:
            max_wz = min(max_wz, config.yaw_slowdown_max_wz)
            min_wz = min(min_wz, config.yaw_slowdown_min_wz)
        if recovery and not inside_position_acceptance:
            max_wz = min(max_wz, config.recovery_yaw_max_wz)
            min_wz = min(min_wz, max_wz)
        wz_abs = min(max_wz, max(config.yaw_kp * abs_yaw_error, min_wz))
        wz = math.copysign(wz_abs, yaw_error)
    return vx, vy, wz


def _compute_forward_translation_command(
    *,
    body_goal_x: float,
    body_goal_y: float,
    distance_to_goal: float,
    config: TerminalPoseConfig,
) -> tuple[float, float, float]:
    """携物时用非完整约束式小步收敛，避免依赖低速横移能力。"""

    heading_error = math.atan2(body_goal_y, body_goal_x)
    abs_heading_error = abs(heading_error)
    max_wz = min(
        max(0.0, float(config.yaw_max_wz)),
        max(0.0, float(config.forward_turn_max_wz)),
    )
    if abs_heading_error <= 1.0e-6 or max_wz <= 0.0:
        wz = 0.0
    else:
        proportional_wz = min(max_wz, config.yaw_kp * abs_heading_error)
        if abs_heading_error > config.forward_heading_deadband:
            min_wz = min(max_wz, max(0.0, config.yaw_min_wz))
            proportional_wz = max(proportional_wz, min_wz)
        wz = math.copysign(proportional_wz, heading_error)

    max_vx = max(0.0, float(config.max_vx))
    if abs_heading_error > config.forward_heading_deadband:
        vx = min(max_vx, max(0.0, config.forward_turn_gait_vx))
    else:
        vx = min(max_vx, max(0.0, config.position_kp * distance_to_goal))
        min_vx = min(max_vx, max(0.0, config.min_vx))
        if 0.0 < vx < min_vx:
            vx = min_vx
    return vx, 0.0, wz


def _axis_velocity(
    *,
    error: float,
    kp: float,
    max_abs: float,
    min_abs: float,
    deadband: float,
    allow_negative: bool,
    scale: float,
) -> float:
    """Proportional axis command with optional minimum active gait speed."""

    if abs(error) <= deadband:
        return 0.0
    if error < 0.0 and not allow_negative:
        return 0.0
    value = kp * error * scale
    max_abs = max(0.0, max_abs)
    value = max(-max_abs, min(max_abs, value))
    min_abs = min(max(0.0, min_abs), max_abs)
    if min_abs > 0.0 and 0.0 < abs(value) < min_abs:
        value = math.copysign(min_abs, value)
    return value


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
