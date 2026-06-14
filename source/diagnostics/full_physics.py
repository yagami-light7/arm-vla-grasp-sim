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
        grasp_tcp_distance_tolerance_m: float = 0.16,
        place_xy_tolerance_m: float = 0.12,
        place_z_tolerance_m: float = 0.10,
        place_linear_velocity_tolerance_mps: float = 0.20,
    ):
        self.navigation = navigation
        self.grasp_tcp_distance_tolerance_m = float(grasp_tcp_distance_tolerance_m)
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
        del episode_spec
        close_count = int(state.metadata.get("gripper_close_apply_count", 0) or 0)
        if close_count <= 0:
            return VerificationResult(
                success=False,
                failure_reason="grasp_failed",
                metadata={"reason": "gripper_close_not_applied", "gripper_close_apply_count": close_count},
            )
        if state.object_pose is None or state.tcp_pose is None:
            return VerificationResult(
                success=False,
                failure_reason="grasp_failed",
                metadata={
                    "reason": "missing_object_or_tcp_pose",
                    "object_pose": state.object_pose,
                    "tcp_pose": state.tcp_pose,
                },
            )
        distance = _distance_xyz(state.object_pose[:3], state.tcp_pose[:3])
        success = distance <= self.grasp_tcp_distance_tolerance_m
        return VerificationResult(
            success=success,
            failure_reason="" if success else "grasp_failed",
            metadata={
                "validation_mode": "object_tcp_contact_window",
                "object_tcp_distance_m": distance,
                "grasp_tcp_distance_tolerance_m": self.grasp_tcp_distance_tolerance_m,
                "gripper_close_apply_count": close_count,
                "object_pose": state.object_pose,
                "tcp_pose": state.tcp_pose,
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
