"""Episode recording contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .simulation import RobotAction, SimulationState
from .task import EpisodeSpec


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    timestamp: float
    pipeline_state: str
    observation: SimulationState
    action: RobotAction
    post_step_observation: SimulationState
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder(Protocol):
    @property
    def output_dir(self) -> Path:
        ...

    def save_task(self, episode_spec: EpisodeSpec) -> Path:
        ...

    def record_event(self, event: dict[str, Any]) -> None:
        ...

    def record_step(self, record: StepRecord) -> None:
        ...

    def prepare_lerobot_export(self) -> dict[str, Any]:
        ...

    def close(self, summary: dict[str, Any]) -> Path:
        ...
