"""Progress-based stall detection for commanded base navigation."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StallDiagnostics:
    """Statistics for the current sliding navigation window."""

    sample_count: int
    max_displacement_m: float
    forward_command_ratio: float
    max_yaw_displacement_rad: float
    angular_command_ratio: float


class NavigationStallDetector:
    """Detect physical stalls without being reset by occasional slow commands."""

    def __init__(
        self,
        *,
        window_steps: int,
        min_progress_m: float,
        min_forward_command: float = 0.05,
        min_forward_ratio: float = 0.25,
        min_angular_progress_rad: float = 0.05,
        min_angular_command: float = 0.05,
        min_angular_ratio: float = 0.25,
    ) -> None:
        if window_steps < 2:
            raise ValueError("window_steps must be at least 2.")
        if min_progress_m <= 0.0:
            raise ValueError("min_progress_m must be positive.")
        if not 0.0 <= min_forward_ratio <= 1.0:
            raise ValueError("min_forward_ratio must be between 0 and 1.")
        self.window_steps = window_steps
        self.min_progress_m = min_progress_m
        self.min_forward_command = min_forward_command
        self.min_forward_ratio = min_forward_ratio
        self.min_angular_progress_rad = float(min_angular_progress_rad)
        self.min_angular_command = float(min_angular_command)
        self.min_angular_ratio = float(min_angular_ratio)
        self._samples: deque[tuple[float, float, float, bool, bool]] = deque(
            maxlen=window_steps
        )

    def reset(self) -> None:
        """Clear the active sliding window."""

        self._samples.clear()

    def update(
        self,
        x: float,
        y: float,
        cmd_vx: float,
        yaw: float = 0.0,
        cmd_wz: float = 0.0,
    ) -> tuple[bool, StallDiagnostics]:
        """Record one navigation step and report whether progress has stalled."""

        self._samples.append(
            (
                x,
                y,
                yaw,
                cmd_vx >= self.min_forward_command,
                abs(cmd_wz) >= self.min_angular_command,
            )
        )
        diagnostics = self.diagnostics()
        translation_stalled = (
            diagnostics.sample_count == self.window_steps
            and diagnostics.forward_command_ratio >= self.min_forward_ratio
            and diagnostics.max_displacement_m < self.min_progress_m
        )
        angular_stalled = (
            diagnostics.sample_count == self.window_steps
            and diagnostics.angular_command_ratio >= self.min_angular_ratio
            and diagnostics.max_yaw_displacement_rad < self.min_angular_progress_rad
            and diagnostics.max_displacement_m < self.min_progress_m
        )
        return translation_stalled or angular_stalled, diagnostics

    def diagnostics(self) -> StallDiagnostics:
        """Return statistics for the current window."""

        if not self._samples:
            return StallDiagnostics(
                sample_count=0,
                max_displacement_m=0.0,
                forward_command_ratio=0.0,
                max_yaw_displacement_rad=0.0,
                angular_command_ratio=0.0,
            )
        origin_x, origin_y, origin_yaw, _, _ = self._samples[0]
        max_displacement = max(
            math.hypot(x - origin_x, y - origin_y)
            for x, y, _, _, _ in self._samples
        )
        max_yaw_displacement = max(
            abs(math.atan2(math.sin(yaw - origin_yaw), math.cos(yaw - origin_yaw)))
            for _, _, yaw, _, _ in self._samples
        )
        forward_ratio = sum(
            is_forward for _, _, _, is_forward, _ in self._samples
        ) / len(self._samples)
        angular_ratio = sum(
            is_angular for _, _, _, _, is_angular in self._samples
        ) / len(self._samples)
        return StallDiagnostics(
            sample_count=len(self._samples),
            max_displacement_m=max_displacement,
            forward_command_ratio=forward_ratio,
            max_yaw_displacement_rad=max_yaw_displacement,
            angular_command_ratio=angular_ratio,
        )
