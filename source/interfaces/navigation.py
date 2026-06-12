"""Navigation planning and execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .simulation import RobotAction, SimulationState


@dataclass(frozen=True)
class NavGoal:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class NavPlan:
    goal: NavGoal
    waypoints: tuple[tuple[float, float], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class NavPlanner(Protocol):
    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        ...


class NavExecutor(Protocol):
    def reset(self, plan: NavPlan) -> None:
        ...

    def compute_action(self, state: SimulationState) -> RobotAction:
        ...

    def is_done(self, state: SimulationState) -> bool:
        ...

    def status(self) -> dict[str, Any]:
        ...
