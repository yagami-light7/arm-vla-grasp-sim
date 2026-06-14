"""Isaac manipulation action apply smoke 的验证器。"""

from __future__ import annotations

from source.interfaces import EpisodeSpec, SimulationState, VerificationResult


class ManipulationApplySmokeVerifier:
    """只验证 arm/gripper action 被 Isaac runtime 接收，不验证抓放物理结果。"""

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del state, episode_spec
        return VerificationResult(success=True, metadata={"skipped": "navigation_not_executed"})

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del episode_spec
        metadata = _apply_metadata(state)
        arm_count = int(metadata.get("arm_joint_action_apply_count", 0))
        close_count = int(metadata.get("gripper_close_apply_count", 0))
        success = arm_count > 0 and close_count > 0
        return VerificationResult(
            success=success,
            failure_reason="pick_tracking_failed",
            metadata={
                **metadata,
                "validation_mode": "joint_action_apply_only",
                "requires_object_lift": False,
            },
        )

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del state, episode_spec
        return VerificationResult(success=True, metadata={"skipped": "navigation_not_executed"})

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del episode_spec
        metadata = _apply_metadata(state)
        open_count = int(metadata.get("gripper_open_apply_count", 0))
        success = int(metadata.get("arm_joint_action_apply_count", 0)) > 0 and open_count > 0
        return VerificationResult(
            success=success,
            failure_reason="place_tracking_failed",
            metadata={
                **metadata,
                "validation_mode": "joint_action_apply_only",
                "requires_object_place": False,
            },
        )


def _apply_metadata(state: SimulationState) -> dict:
    keys = (
        "joint_action_apply_count",
        "arm_joint_action_apply_count",
        "gripper_joint_action_apply_count",
        "gripper_close_apply_count",
        "gripper_open_apply_count",
        "last_joint_action_report",
        "last_arm_tracking_report",
        "arm_tracking_peak_report",
        "arm_tracking_report",
        "arm_tracking_sample_count",
        "arm_tracking_max_abs_error",
    )
    return {key: state.metadata.get(key) for key in keys if key in state.metadata}
