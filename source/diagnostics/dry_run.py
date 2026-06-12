"""Verification implementation for deterministic dry-run state."""

from __future__ import annotations

import math

from source.interfaces import EpisodeSpec, SimulationState, VerificationResult


class DryRunEpisodeVerifier:
    @staticmethod
    def _goal_distance(state: SimulationState, x: float, y: float) -> float:
        return math.hypot(state.robot_root_pose[0] - x, state.robot_root_pose[1] - y)

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        distance = self._goal_distance(state, episode_spec.pick_goal.x, episode_spec.pick_goal.y)
        return VerificationResult(
            success=distance <= 1.0e-6,
            failure_reason="pick_target_unreachable",
            metadata={"goal_distance": distance},
        )

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del episode_spec
        lifted = bool(state.metadata.get("object_lifted", False))
        return VerificationResult(
            success=lifted,
            failure_reason="grasp_failed",
            metadata={"object_lifted": lifted},
        )

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        if episode_spec.place_goal is None:
            return VerificationResult(False, "place_target_unreachable")
        distance = self._goal_distance(state, episode_spec.place_goal.x, episode_spec.place_goal.y)
        return VerificationResult(
            success=distance <= 1.0e-6,
            failure_reason="place_target_unreachable",
            metadata={"goal_distance": distance},
        )

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del episode_spec
        placed = bool(state.metadata.get("object_placed", False))
        return VerificationResult(
            success=placed,
            failure_reason="object_out_of_place",
            metadata={"object_placed": placed},
        )
