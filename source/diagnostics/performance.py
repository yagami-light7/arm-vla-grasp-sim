"""Thread-safe wall-time profiling shared by the pipeline and its components."""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without a numpy dependency."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class _OperationSamples:
    durations: list[float] = field(default_factory=list)
    work_units: int = 0


class WallTimeProfiler:
    """Collect nested operation wall times using a single monotonic clock.

    Nested measurements intentionally overlap.  The report therefore exposes
    operation totals as diagnostic attribution, not as an additive partition of
    episode wall time.
    """

    SCHEMA_VERSION = "wall_time_profile_v1"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.perf_counter()
        self._operations: dict[str, _OperationSamples] = {}

    @contextmanager
    def measure(self, operation: str, *, work_units: int = 1) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                operation,
                time.perf_counter() - started_at,
                work_units=work_units,
            )

    def record(self, operation: str, duration_seconds: float, *, work_units: int = 1) -> None:
        duration = float(duration_seconds)
        if duration < 0.0 or not math.isfinite(duration):
            raise ValueError("profile duration must be finite and non-negative")
        units = int(work_units)
        if units < 0:
            raise ValueError("profile work_units must be non-negative")
        with self._lock:
            samples = self._operations.setdefault(str(operation), _OperationSamples())
            samples.durations.append(duration)
            samples.work_units += units

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started_at

    def report(self, **metadata: Any) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                name: (list(samples.durations), samples.work_units)
                for name, samples in self._operations.items()
            }
        operations: dict[str, Any] = {}
        for name in sorted(snapshot):
            durations, work_units = snapshot[name]
            total = sum(durations)
            count = len(durations)
            operations[name] = {
                "count": count,
                "work_units": work_units,
                "total_seconds": total,
                "mean_seconds": total / count if count else 0.0,
                "min_seconds": min(durations, default=0.0),
                "max_seconds": max(durations, default=0.0),
                "p50_seconds": _percentile(durations, 0.50),
                "p95_seconds": _percentile(durations, 0.95),
            }
        return {
            "schema_version": self.SCHEMA_VERSION,
            "wall_seconds": self.elapsed_seconds(),
            "operations_are_non_additive": True,
            "operations": operations,
            **metadata,
        }

