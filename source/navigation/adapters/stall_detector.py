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


class NavigationStallDetector:
    """Detect physical stalls without being reset by occasional slow commands."""

    def __init__(
        self,
        *,
        window_steps: int,
        min_progress_m: float,
        min_forward_command: float = 0.05,
        min_forward_ratio: float = 0.25,
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
        self._samples: deque[tuple[float, float, bool]] = deque(maxlen=window_steps)

    def reset(self) -> None:
        """Clear the active sliding window."""

        self._samples.clear()

    def update(self, x: float, y: float, cmd_vx: float) -> tuple[bool, StallDiagnostics]:
        """Record one navigation step and report whether progress has stalled."""

        self._samples.append((x, y, cmd_vx >= self.min_forward_command))
        diagnostics = self.diagnostics()
        stalled = (
            diagnostics.sample_count == self.window_steps
            and diagnostics.forward_command_ratio >= self.min_forward_ratio
            and diagnostics.max_displacement_m < self.min_progress_m
        )
        return stalled, diagnostics

    def diagnostics(self) -> StallDiagnostics:
        """Return statistics for the current window."""

        if not self._samples:
            return StallDiagnostics(sample_count=0, max_displacement_m=0.0, forward_command_ratio=0.0)
        origin_x, origin_y, _ = self._samples[0]
        max_displacement = max(math.hypot(x - origin_x, y - origin_y) for x, y, _ in self._samples)
        forward_ratio = sum(is_forward for _, _, is_forward in self._samples) / len(self._samples)
        return StallDiagnostics(
            sample_count=len(self._samples),
            max_displacement_m=max_displacement,
            forward_command_ratio=forward_ratio,
        )
