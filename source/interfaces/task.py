"""Normalized task contracts for the full-physics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .navigation import NavGoal


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: int
    episode_id: int
    instruction: str
    scene_usd: str
    nav_map: str
    start: NavGoal
    pick_goal: NavGoal
    place_goal: NavGoal | None
    object_prim_path: str | None
    object_initial_pose: tuple[float, float, float, float, float, float] | None
    place_target_pose: tuple[float, float, float, float, float, float] | None
    raw_task: dict[str, Any] = field(default_factory=dict)


class TaskProvider(Protocol):
    def load(self, path: str | Path) -> EpisodeSpec:
        ...
