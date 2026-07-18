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
        place_xy_tolerance_m: float = 0.06,
        place_z_tolerance_m: float = 0.03,
        place_linear_velocity_tolerance_mps: float = 0.05,
        place_angular_velocity_tolerance_rps: float = 0.50,
        place_release_xy_tolerance_m: float = 0.03,
        place_release_z_tolerance_m: float = 0.015,
        place_release_peak_linear_speed_tolerance_mps: float = 0.35,
        place_release_peak_horizontal_speed_tolerance_mps: float = 0.15,
        place_release_peak_angular_speed_tolerance_rps: float = 12.0,
        place_release_horizontal_displacement_tolerance_m: float = 0.05,
    ):
        self.navigation = navigation
        self.grasp_tcp_distance_tolerance_m = float(grasp_tcp_distance_tolerance_m)
        self.object_lift_height_tolerance_m = float(object_lift_height_tolerance_m)
        self.object_retreat_distance_tolerance_m = float(object_retreat_distance_tolerance_m)
        self.pick_linear_velocity_tolerance_mps = float(pick_linear_velocity_tolerance_mps)
        self.place_xy_tolerance_m = float(place_xy_tolerance_m)
        self.place_z_tolerance_m = float(place_z_tolerance_m)
        self.place_linear_velocity_tolerance_mps = float(place_linear_velocity_tolerance_mps)
        self.place_angular_velocity_tolerance_rps = float(
            place_angular_velocity_tolerance_rps
        )
        self.place_release_xy_tolerance_m = float(place_release_xy_tolerance_m)
        self.place_release_z_tolerance_m = float(place_release_z_tolerance_m)
        self.place_release_peak_linear_speed_tolerance_mps = float(
            place_release_peak_linear_speed_tolerance_mps
        )
        self.place_release_peak_horizontal_speed_tolerance_mps = float(
            place_release_peak_horizontal_speed_tolerance_mps
        )
        self.place_release_peak_angular_speed_tolerance_rps = float(
            place_release_peak_angular_speed_tolerance_rps
        )
        self.place_release_horizontal_displacement_tolerance_m = float(
            place_release_horizontal_displacement_tolerance_m
        )

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
        release_open_count = int(state.metadata.get("place_open_apply_count_delta", 0) or 0)
        release_observed = state.metadata.get("place_release_observed") is True
        if open_count <= 0 or release_open_count <= 0 or not release_observed:
            return VerificationResult(
                success=False,
                failure_reason="place_release_not_observed",
                metadata={
                    "reason": "place_gripper_open_not_observed",
                    "gripper_open_apply_count": open_count,
                    "place_open_apply_count_delta": release_open_count,
                    "place_release_observed": release_observed,
                },
            )
        place_target_pose, place_target_source = _runtime_place_target_pose(
            state,
            episode_spec,
        )
        if state.object_pose is None or place_target_pose is None:
            return VerificationResult(
                success=False,
                failure_reason="object_out_of_place",
                metadata={
                    "reason": "missing_object_or_place_pose",
                    "object_pose": state.object_pose,
                    "place_target_pose": place_target_pose,
                    "place_target_pose_source": place_target_source,
                },
            )

        object_xyz = _xyz(state.object_pose)
        target_xyz = _xyz(place_target_pose)
        if object_xyz is None or target_xyz is None:
            return VerificationResult(
                success=False,
                failure_reason="object_out_of_place",
                metadata={
                    "reason": "non_finite_object_or_place_pose",
                    "object_pose": state.object_pose,
                    "place_target_pose": place_target_pose,
                    "place_target_pose_source": place_target_source,
                },
            )
        object_velocity = _finite_vector(state.object_velocity, minimum_length=6)
        if object_velocity is None:
            return VerificationResult(
                success=False,
                failure_reason="place_dynamics_unavailable",
                metadata={
                    "reason": "object_velocity_unavailable_or_non_finite",
                    "object_velocity": state.object_velocity,
                },
            )

        try:
            validation_config = _place_validation_config(
                episode_spec,
                default_xy_tolerance_m=self.place_xy_tolerance_m,
                default_z_tolerance_m=self.place_z_tolerance_m,
                default_linear_velocity_tolerance_mps=(
                    self.place_linear_velocity_tolerance_mps
                ),
                default_angular_velocity_tolerance_rps=(
                    self.place_angular_velocity_tolerance_rps
                ),
                default_release_peak_linear_speed_tolerance_mps=(
                    self.place_release_peak_linear_speed_tolerance_mps
                ),
            )
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                success=False,
                failure_reason="place_validation_config_invalid",
                metadata={
                    "reason": "place_validation_config_invalid",
                    "detail": str(exc),
                },
            )
        xy_tolerance = validation_config["place_xy_tolerance_m"]
        z_tolerance = validation_config["place_z_tolerance_m"]
        linear_velocity_tolerance = float(
            validation_config["place_linear_velocity_tolerance_mps"]
        )
        angular_velocity_tolerance = float(
            validation_config["place_angular_velocity_tolerance_rps"]
        )
        region = validation_config["placement_region_world"]
        peak_linear_tolerance = float(
            validation_config[
                "place_release_peak_linear_speed_tolerance_mps"
            ]
        )
        peak_upward_tolerance = float(
            validation_config[
                "place_release_peak_upward_speed_tolerance_mps"
            ]
        )
        peak_downward_tolerance = float(
            validation_config[
                "place_release_peak_downward_speed_tolerance_mps"
            ]
        )

        xy_error = math.hypot(
            object_xyz[0] - target_xyz[0],
            object_xyz[1] - target_xyz[1],
        )
        z_error = abs(object_xyz[2] - target_xyz[2])
        linear_speed = _linear_speed(object_velocity)
        angular_speed = _angular_speed(object_velocity)
        region_contains_object_center = None
        if region is not None:
            region_contains_object_center = bool(
                region["x_min"] <= float(object_xyz[0]) <= region["x_max"]
                and region["y_min"] <= float(object_xyz[1]) <= region["y_max"]
            )

        release_pose = state.metadata.get("place_release_object_pose")
        expected_release_center = state.metadata.get(
            "place_expected_release_object_center"
        )
        if _xyz(release_pose) is None or _xyz(expected_release_center) is None:
            return VerificationResult(
                success=False,
                failure_reason="place_release_pose_unavailable",
                metadata={
                    "reason": "release_pose_or_target_unavailable",
                    "place_release_object_pose": release_pose,
                    "place_expected_release_object_center": expected_release_center,
                },
            )
        release_xyz = _xyz(release_pose)
        expected_release_xyz = _xyz(expected_release_center)
        assert release_xyz is not None and expected_release_xyz is not None
        release_xy_error = math.hypot(
            release_xyz[0] - expected_release_xyz[0],
            release_xyz[1] - expected_release_xyz[1],
        )
        release_z_error_to_planned_center = abs(
            release_xyz[2] - expected_release_xyz[2]
        )
        release_z_error_to_final_target = abs(release_xyz[2] - target_xyz[2])
        if release_z_error_to_final_target < release_z_error_to_planned_center:
            release_z_error = release_z_error_to_final_target
            release_z_reference = "final_supported_target"
        else:
            release_z_error = release_z_error_to_planned_center
            release_z_reference = "planned_release_center"
        peak_linear_speed = _optional_finite_float(
            state.metadata.get("place_peak_object_linear_speed_mps")
        )
        peak_horizontal_speed = _optional_finite_float(
            state.metadata.get("place_peak_object_horizontal_speed_mps")
        )
        peak_upward_speed = _optional_finite_float(
            state.metadata.get("place_peak_object_upward_speed_mps")
        )
        peak_downward_speed = _optional_finite_float(
            state.metadata.get("place_peak_object_downward_speed_mps")
        )
        directional_speed_source = "signed_vertical_velocity_peaks"
        if peak_upward_speed is None or peak_downward_speed is None:
            # 兼容旧 episode：没有有向速度时继续按原三轴峰值严格判断，
            # 不会因为升级 schema 而放宽历史数据质量门。
            peak_upward_speed = peak_linear_speed
            peak_downward_speed = peak_linear_speed
            directional_speed_source = "legacy_linear_speed_fallback"
        peak_angular_speed = _optional_finite_float(
            state.metadata.get("place_peak_object_angular_speed_rps")
        )
        max_horizontal_displacement = _optional_finite_float(
            state.metadata.get("place_max_horizontal_displacement_m")
        )
        velocity_samples = int(
            state.metadata.get("place_release_velocity_sample_count", 0) or 0
        )

        if (
            peak_linear_speed is None
            or peak_horizontal_speed is None
            or peak_upward_speed is None
            or peak_downward_speed is None
            or peak_angular_speed is None
            or max_horizontal_displacement is None
            or velocity_samples <= 0
        ):
            failure_reason = "place_dynamics_unavailable"
        elif (
            release_xy_error > self.place_release_xy_tolerance_m
            or release_z_error > self.place_release_z_tolerance_m
        ):
            failure_reason = "place_release_pose_error"
        elif (
            peak_upward_speed > peak_upward_tolerance
            or peak_downward_speed > peak_downward_tolerance
            or peak_horizontal_speed
            > self.place_release_peak_horizontal_speed_tolerance_mps
            or peak_angular_speed > self.place_release_peak_angular_speed_tolerance_rps
            or max_horizontal_displacement
            > self.place_release_horizontal_displacement_tolerance_m
        ):
            failure_reason = "place_release_ejected"
        elif (
            xy_error > xy_tolerance
            or z_error > z_tolerance
            or region_contains_object_center is False
        ):
            failure_reason = "object_out_of_place"
        elif (
            linear_speed > linear_velocity_tolerance
            or angular_speed > angular_velocity_tolerance
        ):
            failure_reason = "object_unstable_after_place"
        else:
            failure_reason = ""
        success = not failure_reason
        return VerificationResult(
            success=success,
            failure_reason=failure_reason,
            metadata={
                "validation_mode": (
                    "contact_release_pose_dynamics_region_and_final_stability"
                    if region is not None
                    else "contact_release_pose_dynamics_and_final_stability"
                ),
                "object_pose": state.object_pose,
                "place_target_pose": place_target_pose,
                "place_target_pose_source": place_target_source,
                "configured_place_target_pose": episode_spec.place_target_pose,
                "place_xy_error_m": xy_error,
                "place_z_error_m": z_error,
                "object_linear_speed_mps": linear_speed,
                "object_angular_speed_rps": angular_speed,
                "place_xy_tolerance_m": xy_tolerance,
                "place_z_tolerance_m": z_tolerance,
                "place_linear_velocity_tolerance_mps": linear_velocity_tolerance,
                "place_angular_velocity_tolerance_rps": (
                    angular_velocity_tolerance
                ),
                "placement_region_world": region,
                "placement_region_contains_object_center": (
                    region_contains_object_center
                ),
                "place_release_object_pose": release_pose,
                "place_expected_release_object_center": expected_release_center,
                "place_release_xy_error_m": release_xy_error,
                "place_release_z_error_m": release_z_error,
                "place_release_z_error_to_planned_center_m": (
                    release_z_error_to_planned_center
                ),
                "place_release_z_error_to_final_target_m": (
                    release_z_error_to_final_target
                ),
                "place_release_z_reference": release_z_reference,
                "place_peak_object_linear_speed_mps": peak_linear_speed,
                "place_peak_object_horizontal_speed_mps": peak_horizontal_speed,
                "place_peak_object_upward_speed_mps": peak_upward_speed,
                "place_peak_object_downward_speed_mps": peak_downward_speed,
                "place_directional_speed_source": directional_speed_source,
                "place_peak_object_angular_speed_rps": peak_angular_speed,
                "place_max_horizontal_displacement_m": max_horizontal_displacement,
                "place_release_velocity_sample_count": velocity_samples,
                "place_release_xy_tolerance_m": self.place_release_xy_tolerance_m,
                "place_release_z_tolerance_m": self.place_release_z_tolerance_m,
                "place_release_peak_linear_speed_tolerance_mps": (
                    peak_linear_tolerance
                ),
                "place_release_peak_upward_speed_tolerance_mps": (
                    peak_upward_tolerance
                ),
                "place_release_peak_downward_speed_tolerance_mps": (
                    peak_downward_tolerance
                ),
                "place_release_peak_horizontal_speed_tolerance_mps": (
                    self.place_release_peak_horizontal_speed_tolerance_mps
                ),
                "place_release_peak_angular_speed_tolerance_rps": (
                    self.place_release_peak_angular_speed_tolerance_rps
                ),
                "place_release_horizontal_displacement_tolerance_m": (
                    self.place_release_horizontal_displacement_tolerance_m
                ),
                "gripper_open_apply_count": open_count,
                "place_open_apply_count_delta": release_open_count,
                "failure_reason": failure_reason,
            },
        )


def _runtime_place_target_pose(
    state: SimulationState,
    episode_spec: EpisodeSpec,
) -> tuple[tuple[float, ...] | None, str]:
    """优先采用 CuRobo 已审计的 runtime Mesh-truth 最终物体中心。"""

    configured = episode_spec.place_target_pose
    export = state.metadata.get("last_current_state_curobo_place_export")
    export = export if isinstance(export, dict) else {}
    report = export.get("mesh_truth_place_target_report")
    report = report if isinstance(report, dict) else {}
    derived = report.get("derived_place_pose_world")
    if (
        report.get("verified") is True
        and report.get("xyz_source") == "runtime_mesh_truth"
        and isinstance(derived, dict)
    ):
        try:
            xyz = tuple(float(derived[axis]) for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError):
            xyz = ()
        if len(xyz) == 3 and all(math.isfinite(value) for value in xyz):
            orientation = () if configured is None else tuple(configured[3:])
            return xyz + orientation, "runtime_mesh_truth"
    return configured, "episode_spec"


def _distance_xyz(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _linear_speed(velocity: tuple[float, ...] | None) -> float:
    if velocity is None or len(velocity) < 3:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in velocity[:3]))


def _angular_speed(velocity: tuple[float, ...] | None) -> float:
    if velocity is None or len(velocity) < 6:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in velocity[3:6]))


def _place_validation_config(
    episode_spec: EpisodeSpec,
    *,
    default_xy_tolerance_m: float,
    default_z_tolerance_m: float,
    default_linear_velocity_tolerance_mps: float,
    default_angular_velocity_tolerance_rps: float,
    default_release_peak_linear_speed_tolerance_mps: float,
) -> dict[str, object]:
    """解析任务级放置容差与可选的世界坐标安全区域。"""

    raw_place = (episode_spec.raw_task or {}).get("place") or {}
    if not isinstance(raw_place, dict):
        raise TypeError("task.place 必须是对象")
    xy_tolerance = _positive_finite_float(
        raw_place.get("place_xy_tolerance", default_xy_tolerance_m),
        field_name="task.place.place_xy_tolerance",
    )
    z_tolerance = _positive_finite_float(
        raw_place.get("place_z_tolerance", default_z_tolerance_m),
        field_name="task.place.place_z_tolerance",
    )
    linear_velocity_tolerance = _positive_finite_float(
        raw_place.get(
            "place_linear_velocity_tolerance_mps",
            default_linear_velocity_tolerance_mps,
        ),
        field_name="task.place.place_linear_velocity_tolerance_mps",
    )
    angular_velocity_tolerance = _positive_finite_float(
        raw_place.get(
            "place_angular_velocity_tolerance_rps",
            default_angular_velocity_tolerance_rps,
        ),
        field_name="task.place.place_angular_velocity_tolerance_rps",
    )
    linear_peak_tolerance = _positive_finite_float(
        raw_place.get(
            "place_release_peak_linear_speed_tolerance_mps",
            default_release_peak_linear_speed_tolerance_mps,
        ),
        field_name=(
            "task.place.place_release_peak_linear_speed_tolerance_mps"
        ),
    )
    upward_peak_tolerance = _positive_finite_float(
        raw_place.get(
            "place_release_peak_upward_speed_tolerance_mps",
            linear_peak_tolerance,
        ),
        field_name=(
            "task.place.place_release_peak_upward_speed_tolerance_mps"
        ),
    )
    downward_peak_tolerance = _positive_finite_float(
        raw_place.get(
            "place_release_peak_downward_speed_tolerance_mps",
            linear_peak_tolerance,
        ),
        field_name=(
            "task.place.place_release_peak_downward_speed_tolerance_mps"
        ),
    )
    raw_region = raw_place.get("placement_region")
    region: dict[str, float] | None = None
    if raw_region is not None:
        if not isinstance(raw_region, dict):
            raise TypeError("task.place.placement_region 必须是对象")
        bound_keys = ("x_min", "x_max", "y_min", "y_max")
        configured_keys = [key for key in bound_keys if key in raw_region]
        if configured_keys and len(configured_keys) != len(bound_keys):
            raise ValueError("task.place.placement_region 的 XY 边界必须完整配置")
        if configured_keys:
            if str(raw_region.get("frame", "world")) != "world":
                raise ValueError("task.place.placement_region 当前只支持 world frame")
            region = {
                key: _strict_finite_float(
                    raw_region[key],
                    field_name=f"task.place.placement_region.{key}",
                )
                for key in bound_keys
            }
            if region["x_min"] >= region["x_max"]:
                raise ValueError("task.place.placement_region x_min 必须小于 x_max")
            if region["y_min"] >= region["y_max"]:
                raise ValueError("task.place.placement_region y_min 必须小于 y_max")
    return {
        "place_xy_tolerance_m": xy_tolerance,
        "place_z_tolerance_m": z_tolerance,
        "place_linear_velocity_tolerance_mps": linear_velocity_tolerance,
        "place_angular_velocity_tolerance_rps": angular_velocity_tolerance,
        "place_release_peak_linear_speed_tolerance_mps": (
            linear_peak_tolerance
        ),
        "place_release_peak_upward_speed_tolerance_mps": (
            upward_peak_tolerance
        ),
        "place_release_peak_downward_speed_tolerance_mps": (
            downward_peak_tolerance
        ),
        "placement_region_world": region,
    }


def _positive_finite_float(value: object, *, field_name: str) -> float:
    """要求配置值为有限正数。"""

    parsed = _strict_finite_float(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} 必须大于 0")
    return parsed


def _strict_finite_float(value: object, *, field_name: str) -> float:
    """严格解析有限浮点数。"""

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} 必须是有限数值")
    return parsed


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_vector(
    value: object,
    *,
    minimum_length: int,
) -> tuple[float, ...] | None:
    if not isinstance(value, (tuple, list)) or len(value) < minimum_length:
        return None
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return parsed if all(math.isfinite(item) for item in parsed) else None


def _xyz(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        xyz = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in xyz):
        return None
    return (xyz[0], xyz[1], xyz[2])
