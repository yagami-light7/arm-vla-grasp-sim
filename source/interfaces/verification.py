"""Success and reachability verification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .simulation import SimulationState
from .task import EpisodeSpec


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodeVerifier(Protocol):
    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        ...

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        ...

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        ...

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        ...
