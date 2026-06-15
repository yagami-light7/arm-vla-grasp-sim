"""完整物理 pipeline 的物体级验证器。"""

from __future__ import annotations

import math

from source.interfaces import EpisodeSpec, SimulationState, VerificationResult

from .navigation import NavigationEpisodeVerifier


class FullPhysicsVerifier:
    """验证导航可达、真实抓取保持和最终放置结果。"""

    def __init__(
        self,
        navigation: NavigationEpisodeVerifier,
        *,
        grasp_tcp_distance_tolerance_m: float = 0.08,
        object_lift_height_tolerance_m: float = 0.04,
        object_retreat_distance_tolerance_m: float = 0.03,
        pick_linear_velocity_tolerance_mps: float = 0.30,
        place_xy_tolerance_m: float = 0.12,
        place_z_tolerance_m: float = 0.10,
        place_linear_velocity_tolerance_mps: float = 0.20,
    ):
        self.navigation = navigation
        self.grasp_tcp_distance_tolerance_m = float(grasp_tcp_distance_tolerance_m)
        self.object_lift_height_tolerance_m = float(object_lift_height_tolerance_m)
        self.object_retreat_distance_tolerance_m = float(object_retreat_distance_tolerance_m)
        self.pick_linear_velocity_tolerance_mps = float(pick_linear_velocity_tolerance_mps)
        self.place_xy_tolerance_m = float(place_xy_tolerance_m)
        self.place_z_tolerance_m = float(place_z_tolerance_m)
        self.place_linear_velocity_tolerance_mps = float(place_linear_velocity_tolerance_mps)

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.navigation.verify_pick_reachable(state, episode_spec)

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        close_count = int(state.metadata.get("gripper_close_apply_count", 0) or 0)
        if close_count <= 0:
            return VerificationResult(
                success=False,
                failure_reason="gripper_close_not_applied",
                metadata={
                    "reason": "gripper_close_not_applied",
                    "gripper_close_apply_count": close_count,
                },
            )
        if (
            state.object_pose is None
            or state.tcp_pose is None
            or episode_spec.object_initial_pose is None
        ):
            return VerificationResult(
                success=False,
                failure_reason="object_or_tcp_pose_missing",
                metadata={
                    "reason": "object_or_tcp_pose_missing",
                    "object_pose": state.object_pose,
                    "tcp_pose": state.tcp_pose,
                    "object_initial_pose": episode_spec.object_initial_pose,
                },
            )
        distance = _distance_xyz(state.object_pose[:3], state.tcp_pose[:3])
        initial_object_z = float(episode_spec.object_initial_pose[2])
        object_lift_height = float(state.object_pose[2]) - initial_object_z
        object_retreat_displacement = _distance_xyz(
            state.object_pose[:3],
            episode_spec.object_initial_pose[:3],
        )
        peak_object_lift_height = state.metadata.get(
            "pick_peak_object_lift_height_m"
        )
        verified_lift_height = max(
            object_lift_height,
            (
                float(peak_object_lift_height)
                if peak_object_lift_height is not None
                else object_lift_height
            ),
        )
        linear_speed = _linear_speed(state.object_velocity)
        has_lift_segment = bool(state.metadata.get("pick_has_lift_segment", True))
        has_retreat_segment = bool(state.metadata.get("pick_has_retreat_segment", False))
        use_side_retreat_validation = has_retreat_segment and not has_lift_segment
        if (
            use_side_retreat_validation
            and object_retreat_displacement < self.object_retreat_distance_tolerance_m
        ):
            failure_reason = "object_not_retreated"
        elif (
            not use_side_retreat_validation
            and verified_lift_height < self.object_lift_height_tolerance_m
        ):
            failure_reason = "object_not_lifted"
        elif distance > self.grasp_tcp_distance_tolerance_m:
            failure_reason = "object_too_far_from_tcp"
        elif linear_speed > self.pick_linear_velocity_tolerance_mps:
            failure_reason = "object_unstable_after_pick"
        else:
            failure_reason = ""
        success = not failure_reason
        return VerificationResult(
            success=success,
            failure_reason=failure_reason,
            metadata={
                "validation_mode": (
                    "side_retreat_contact_and_stability"
                    if use_side_retreat_validation
                    else "main_pick_lift_contact_and_stability"
                ),
                "object_tcp_distance_m": distance,
                "object_lift_height_m": object_lift_height,
                "object_retreat_displacement_m": object_retreat_displacement,
                "pick_peak_object_lift_height_m": peak_object_lift_height,
                "verified_object_lift_height_m": verified_lift_height,
                "pick_peak_object_pose": state.metadata.get("pick_peak_object_pose"),
                "pick_peak_step_index": state.metadata.get("pick_peak_step_index"),
                "object_linear_speed_mps": linear_speed,
                "initial_object_z_m": initial_object_z,
                "grasp_tcp_distance_tolerance_m": self.grasp_tcp_distance_tolerance_m,
                "object_lift_height_tolerance_m": self.object_lift_height_tolerance_m,
                "object_retreat_distance_tolerance_m": (
                    self.object_retreat_distance_tolerance_m
                ),
                "pick_has_lift_segment": has_lift_segment,
                "pick_has_retreat_segment": has_retreat_segment,
                "pick_linear_velocity_tolerance_mps": (
                    self.pick_linear_velocity_tolerance_mps
                ),
                "gripper_close_apply_count": close_count,
                "object_pose": state.object_pose,
                "tcp_pose": state.tcp_pose,
                "failure_reason": failure_reason,
            },
        )

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.navigation.verify_place_reachable(state, episode_spec)

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        open_count = int(state.metadata.get("gripper_open_apply_count", 0) or 0)
        if open_count <= 0:
            return VerificationResult(
                success=False,
                failure_reason="place_tracking_failed",
                metadata={"reason": "gripper_open_not_applied", "gripper_open_apply_count": open_count},
            )
        if state.object_pose is None or episode_spec.place_target_pose is None:
            return VerificationResult(
                success=False,
                failure_reason="object_out_of_place",
                metadata={
                    "reason": "missing_object_or_place_pose",
                    "object_pose": state.object_pose,
                    "place_target_pose": episode_spec.place_target_pose,
                },
            )
        object_xyz = state.object_pose[:3]
        target_xyz = episode_spec.place_target_pose[:3]
        xy_error = math.hypot(object_xyz[0] - target_xyz[0], object_xyz[1] - target_xyz[1])
        z_error = abs(object_xyz[2] - target_xyz[2])
        linear_speed = _linear_speed(state.object_velocity)
        success = (
            xy_error <= self.place_xy_tolerance_m
            and z_error <= self.place_z_tolerance_m
            and linear_speed <= self.place_linear_velocity_tolerance_mps
        )
        return VerificationResult(
            success=success,
            failure_reason="" if success else "object_out_of_place",
            metadata={
                "validation_mode": "object_final_pose_and_stability",
                "object_pose": state.object_pose,
                "place_target_pose": episode_spec.place_target_pose,
                "place_xy_error_m": xy_error,
                "place_z_error_m": z_error,
                "object_linear_speed_mps": linear_speed,
                "place_xy_tolerance_m": self.place_xy_tolerance_m,
                "place_z_tolerance_m": self.place_z_tolerance_m,
                "place_linear_velocity_tolerance_mps": self.place_linear_velocity_tolerance_mps,
                "gripper_open_apply_count": open_count,
            },
        )


def _distance_xyz(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _linear_speed(velocity: tuple[float, ...] | None) -> float:
    if velocity is None or len(velocity) < 3:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in velocity[:3]))
