"""Manipulation planning and execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .simulation import RobotAction, SimulationState
from .task import EpisodeSpec


@dataclass(frozen=True)
class ArmPlan:
    operation: str
    joint_trajectory: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class ManipulationPlanner(Protocol):
    def plan_pick(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        ...

    def plan_place(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        ...


class ArmExecutor(Protocol):
    def reset(self, plan: ArmPlan) -> None:
        ...

    def compute_action(self, state: SimulationState) -> RobotAction:
        ...

    def is_done(self, state: SimulationState) -> bool:
        ...

    def status(self) -> dict[str, Any]:
        ...


class GripperController(Protocol):
    def command_open(self) -> str:
        ...

    def command_close(self) -> str:
        ...

    def command_hold(self) -> str:
        ...
