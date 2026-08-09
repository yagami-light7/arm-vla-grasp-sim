from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from source.interfaces import RobotAction
from source.navigation.cmd_vel_to_policy import (
    BodyVelocityCommand,
    CmdVelToPolicyAdapter,
    CmdVelToPolicyConfig,
    NavigationSafetyPermit,
    PolicyCommandInput,
    PolicyCommandWriteReport,
)
from source.navigation.isaac_depth_point_cloud import DepthPointCloudConfig
from source.navigation.isaac_ros2_ogn_bridge import (
    IsaacRos2OgnBridgeConfig,
    OgnBoolSample,
    OgnBsplineDiagnosticsSample,
    OgnControllerStatusSample,
    OgnGridMapObservationDiagnosticsSample,
    OgnPCTGoalSample,
    OgnPathSample,
    OgnStairExecutionFreezePublicationReport,
    OgnTwistSample,
)
from source.recording.jsonl_recorder import _compact_simulation_metadata
from source.simulation.isaaclab_runtime import (
    ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT,
    IsaacLabNavigationRuntime,
    IsaacLabNavigationRuntimeConfig,
    _stair_execution_frozen_from_action,
)
from source.simulation.dynamic_obstacles import resolve_dynamic_obstacle_plan


def _runtime_navigation_permit(
    timestamp: float,
    *,
    status_sequence: int = 1,
    allow: bool = True,
) -> NavigationSafetyPermit:
    return NavigationSafetyPermit(
        header_stamp_ns=max(1, round(timestamp * 1_000_000_000)),
        received_at=timestamp,
        status_sequence=status_sequence,
        state_revision=1,
        goal_id=10,
        active_path_stamp_ns=20,
        state=3 if allow else 5,
        allow_tracking_command=allow,
        force_zero_velocity=not allow,
        identity_valid=True,
        reason="测试许可有效" if allow else "测试强制停车",
    )


class _FakeBridge:
    def __init__(self) -> None:
        self.config = IsaacRos2OgnBridgeConfig(
            enable_pct_goal_publisher=True,
        )
        self.odometry_calls: list[tuple[object, ...]] = []
        self.cloud_calls: list[tuple[np.ndarray, float | None]] = []
        self.path_samples: list[OgnPathSample | None] = []
        self.path_poll_count = 0
        self.controller_status_samples: list[
            OgnControllerStatusSample | None
        ] = []
        self.controller_status_poll_timestamps: list[float] = []
        self.goal_calls: list[tuple[tuple[float, float, float], float, int, str]] = []
        self._goal_generation_sequence = 0
        self._last_goal_sample: OgnPCTGoalSample | None = None
        self._pct_goal_transport_attempt_count = 0

    def update_odometry(self, *args: object) -> None:
        self.odometry_calls.append(args)

    def update_point_cloud(
        self,
        points: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        self.cloud_calls.append((points, timestamp))

    def poll_reference_path(self) -> OgnPathSample | None:
        self.path_poll_count += 1
        return self.path_samples.pop(0) if self.path_samples else None

    def poll_controller_status(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnControllerStatusSample | None:
        self.controller_status_poll_timestamps.append(receipt_timestamp)
        return (
            self.controller_status_samples.pop(0)
            if self.controller_status_samples
            else None
        )

    def publish_pct_goal(
        self,
        position_base_xyz: tuple[float, float, float],
        yaw: float,
        *,
        stamp_ns: int,
        frame_id: str,
    ) -> OgnPCTGoalSample:
        position = tuple(float(value) for value in position_base_xyz)
        self.goal_calls.append((position, float(yaw), stamp_ns, frame_id))
        self._goal_generation_sequence += 1
        sample = OgnPCTGoalSample(
            position_base_xyz=position,
            yaw=float(yaw),
            source_topic="/pct/goal",
            frame_id=frame_id,
            stamp_sec=stamp_ns // 1_000_000_000,
            stamp_nanosec=stamp_ns % 1_000_000_000,
            sequence=self._goal_generation_sequence,
        )
        self._last_goal_sample = sample
        self._pct_goal_transport_attempt_count = 1
        return sample

    def republish_last_pct_goal(self) -> OgnPCTGoalSample:
        sample = self._last_goal_sample
        if sample is None:
            raise RuntimeError("缺少首发 PCT goal")
        stamp_ns = sample.stamp_sec * 1_000_000_000 + sample.stamp_nanosec
        self.goal_calls.append(
            (sample.position_base_xyz, sample.yaw, stamp_ns, sample.frame_id)
        )
        self._pct_goal_transport_attempt_count += 1
        return sample

    @property
    def pct_goal_transport_attempt_count(self) -> int:
        return self._pct_goal_transport_attempt_count


class _FakeFreshnessGate:
    def __init__(self) -> None:
        self.odometry_stamps: list[float] = []
        self.point_cloud_stamps: list[float] = []

    def mark_odometry(self, *, owner_id: str, received_at: float) -> None:
        assert owner_id == "scan_cmd_vel"
        self.odometry_stamps.append(received_at)

    def mark_point_cloud(self, *, owner_id: str, received_at: float) -> None:
        assert owner_id == "scan_cmd_vel"
        self.point_cloud_stamps.append(received_at)


def _controller_status_sample(
    *,
    candidate_present: bool = False,
) -> OgnControllerStatusSample:
    return OgnControllerStatusSample(
        source_topic="/planning/controller_status",
        receipt_timestamp=0.02,
        rx_sequence=4,
        frame_id="world",
        header_stamp_sec=20,
        header_stamp_nanosec=1,
        status_sequence=9,
        acceptance_sequence=3,
        event=2 if candidate_present else 1,
        reference_path_stamp_sec=18,
        reference_path_stamp_nanosec=999_999_999,
        bspline_header_stamp_sec=19,
        bspline_header_stamp_nanosec=2,
        start_time_sec=19,
        start_time_nanosec=3,
        traj_id=42,
        accepted=True,
        trajectory_valid=True,
        is_final=False,
        emergency_stop=False,
        state=10,
        reason=(
            "候选轨迹已拒绝" if candidate_present else "B-spline 已接受"
        ),
        candidate_present=candidate_present,
        candidate_reference_path_stamp_sec=20 if candidate_present else 0,
        candidate_reference_path_stamp_nanosec=10 if candidate_present else 0,
        candidate_bspline_header_stamp_sec=20 if candidate_present else 0,
        candidate_bspline_header_stamp_nanosec=11 if candidate_present else 0,
        candidate_start_time_sec=20 if candidate_present else 0,
        candidate_start_time_nanosec=12 if candidate_present else 0,
        candidate_traj_id=43 if candidate_present else 0,
    )


def _dynamic_evidence_test_plan(
    *,
    start_delay_s: float = 0.0,
    speed_mps: float = 1.0,
):
    return resolve_dynamic_obstacle_plan(
        {
            "dynamic_obstacles": [
                {
                    "id": "crossing_cart",
                    "shape": "cuboid",
                    "floor_id": "F1",
                    "surface_class": "flat",
                    "size_xyz_m": [0.5, 0.5, 1.0],
                    "waypoints_world_xyz": [
                        [0.0, 0.0, 0.5],
                        [2.0, 0.0, 0.5],
                    ],
                    "speed_mps": speed_mps,
                    "start_delay_s": start_delay_s,
                    "yaw_rad": 0.0,
                    "motion": "ping_pong",
                    "color_rgb": [0.9, 0.2, 0.1],
                    "mass_kg": 20.0,
                    "collision_enabled": True,
                    "visible": True,
                }
            ],
            "dynamic_obstacle_safety": {
                "minimum_stair_clearance_m": 1.0,
                "stair_exclusion_aabbs_world": [
                    {
                        "id": "remote_stair",
                        "min_xyz": [100.0, 100.0, -1.0],
                        "max_xyz": [101.0, 101.0, 2.0],
                    }
                ],
            },
        }
    )


def _grid_diagnostics_sample(
    *,
    observation_sequence: int,
    timestamp_s: float,
    hit_points: tuple[tuple[float, float, float], ...] = (),
    explicit_free_points: tuple[tuple[float, float, float], ...] = (),
    explicit_free_miss_voxel_count: int = 0,
    occupied_to_free_count: int = 0,
    sliding_reset_count: int = 0,
    transition_hit_sequences: tuple[int, ...] | None = None,
) -> OgnGridMapObservationDiagnosticsSample:
    stamp_ns = round(timestamp_s * 1_000_000_000)
    map_resolution = 0.05
    voxel_index = lambda point: tuple(  # noqa: E731
        math.floor(float(value) / map_resolution) for value in point
    )
    hit_count = len(hit_points)
    free_count = 1 if explicit_free_points else 0
    accepted_count = hit_count + free_count
    resolved_transition_sequences = (
        transition_hit_sequences
        if transition_hit_sequences is not None
        else tuple(
            max(0, observation_sequence - 1)
            for _point in explicit_free_points
        )
    )
    return OgnGridMapObservationDiagnosticsSample(
        source_topic="/planning/grid_map_observation_diagnostics",
        receipt_timestamp=timestamp_s,
        rx_sequence=observation_sequence,
        frame_id="world",
        header_stamp_sec=stamp_ns // 1_000_000_000,
        header_stamp_nanosec=stamp_ns % 1_000_000_000,
        observation_sequence=observation_sequence,
        sensor_pose_stamp_sec=stamp_ns // 1_000_000_000,
        sensor_pose_stamp_nanosec=stamp_ns % 1_000_000_000,
        sensor_origin=(-1.0, 0.0, 0.5),
        canonical_empty=False,
        map_fusion_performed=True,
        map_resolution=map_resolution,
        input_point_count=accepted_count,
        accepted_endpoint_count=accepted_count,
        hit_endpoint_count=hit_count,
        explicit_free_endpoint_count=free_count,
        hit_endpoint_samples_truncated=False,
        hit_endpoint_samples=hit_points,
        hit_endpoint_sample_voxel_indices=tuple(
            voxel_index(point) for point in hit_points
        ),
        free_to_occupied_transition_count=hit_count,
        free_to_occupied_transition_samples_truncated=False,
        free_to_occupied_transition_hit_samples=hit_points,
        free_to_occupied_transition_voxel_indices=tuple(
            voxel_index(point) for point in hit_points
        ),
        explicit_free_miss_voxel_count=explicit_free_miss_voxel_count,
        occupied_to_free_by_explicit_miss_count=occupied_to_free_count,
        occupied_to_free_samples_truncated=False,
        occupied_to_free_by_explicit_miss_samples=explicit_free_points,
        occupied_to_free_sample_voxel_indices=tuple(
            voxel_index(point) for point in explicit_free_points
        ),
        occupied_to_free_transition_hit_observation_sequences=tuple(
            resolved_transition_sequences
        ),
        occupied_to_free_transition_hit_samples=explicit_free_points,
        occupied_to_free_transition_hit_header_stamp_ns=tuple(
            (
                round(max(0.0, timestamp_s - 1.0) * 1_000_000_000)
                if sequence > 0
                else 0
            )
            for sequence in resolved_transition_sequences
        ),
        occupied_removed_by_sliding_reset_count=sliding_reset_count,
    )


def _bspline_diagnostics_sample(
    *,
    diagnostic_sequence: int,
    timestamp_s: float,
    traj_id: int,
    maximum_deviation_m: float,
) -> OgnBsplineDiagnosticsSample:
    stamp_ns = round(timestamp_s * 1_000_000_000)
    trajectory = (
        (0.5, 1.0, 0.5),
        (1.5, 1.0, 0.5),
        (2.5, 1.0, 0.5),
    )
    reference = (
        (0.5, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (2.5, 0.0, 0.0),
    )
    return OgnBsplineDiagnosticsSample(
        source_topic="/planning/bspline_diagnostics",
        receipt_timestamp=timestamp_s,
        rx_sequence=diagnostic_sequence,
        frame_id="world",
        header_stamp_sec=stamp_ns // 1_000_000_000,
        header_stamp_nanosec=stamp_ns % 1_000_000_000,
        diagnostic_sequence=diagnostic_sequence,
        start_time_sec=stamp_ns // 1_000_000_000,
        start_time_nanosec=stamp_ns % 1_000_000_000,
        reference_path_stamp_sec=1,
        reference_path_stamp_nanosec=1,
        traj_id=traj_id,
        is_final=False,
        emergency_stop=False,
        stationary=False,
        ordered_reference_checked=True,
        ordered_reference_safe=True,
        maximum_trajectory_deviation=maximum_deviation_m,
        maximum_guide_anchor_deviation=maximum_deviation_m,
        maximum_guide_progress_lead=0.01,
        maximum_deviation_limit=0.10,
        maximum_progress_lead_limit=0.02,
        trajectory_duration=0.02,
        maximum_velocity_upper_bound=1.0,
        double_cylinder_radius=0.27,
        double_cylinder_offset=0.16,
        trajectory_sample_count_total=len(trajectory),
        trajectory_samples_truncated=False,
        trajectory_samples=trajectory,
        ordered_reference_sample_count_total=len(reference),
        ordered_reference_samples_truncated=False,
        ordered_reference_samples=reference,
    )


def _active_bspline_diagnostics_sample(
    *,
    diagnostic_sequence: int,
    receipt_timestamp: float,
    event: int,
    fusion_current: int = 0,
    fusion_distinct: int = 0,
) -> OgnBsplineDiagnosticsSample:
    """构造同一完整 identity 的主动感知 typed 事件。"""

    sample = _bspline_diagnostics_sample(
        diagnostic_sequence=diagnostic_sequence,
        timestamp_s=10.0,
        traj_id=70,
        maximum_deviation_m=0.0,
    )
    settled = event in {3, 4, 5}
    return replace(
        sample,
        receipt_timestamp=receipt_timestamp,
        rx_sequence=diagnostic_sequence,
        stationary=True,
        ordered_reference_checked=False,
        ordered_reference_safe=False,
        active_sensing=True,
        active_sensing_event=event,
        active_sensing_start_yaw=0.1,
        active_sensing_target_yaw=0.3,
        active_sensing_yaw_offset=0.2,
        active_sensing_yaw_rate=0.2,
        active_sensing_settle_stamp_sec=11 if settled else 0,
        active_sensing_settle_stamp_nanosec=0,
        active_sensing_settle_yaw_error=0.01 if settled else 0.0,
        active_sensing_settle_angular_speed=0.04 if settled else 0.0,
        active_sensing_stable_duration=0.1 if settled else 0.0,
        active_sensing_fusion_baseline=10 if settled else 0,
        active_sensing_fusion_current=fusion_current if settled else 0,
        active_sensing_fusion_distinct=fusion_distinct if settled else 0,
        active_sensing_fusion_required=3,
        active_sensing_completed=event == 5,
        active_sensing_failed=False,
        active_sensing_reason=f"主动感知事件 {event}",
    )


def _configured_runtime(*, publish_interval: int = 2) -> tuple[
    IsaacLabNavigationRuntime,
    _FakeBridge,
]:
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={
                "distance_to_image_plane": np.asarray(
                    [[[[2.0]]]],
                    dtype=np.float32,
                )
            },
            intrinsic_matrices=np.asarray(
                [np.eye(3, dtype=np.float32)]
            ),
            pos_w=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            quat_w_ros=np.asarray(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
            root_quat_w=np.asarray(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            root_lin_vel_b=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
            root_ang_vel_b=np.asarray([[0.4, 0.5, 0.6]], dtype=np.float32),
        )
    )
    bridge = _FakeBridge()
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=8.0,
            minimum_valid_points=1,
            publish_interval_control_steps=publish_interval,
        ),
    )
    runtime._runtime = SimpleNamespace(  # type: ignore[attr-defined]
        physics_dt=0.0025,
        scene={"head_camera": sensor},
    )
    runtime._adapter = SimpleNamespace(robot=robot)  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 8  # type: ignore[attr-defined]
    runtime._ros2_odometry_publish_count = 0  # type: ignore[attr-defined]
    runtime._ros2_point_cloud_publish_count = 0  # type: ignore[attr-defined]
    runtime._last_camera_render_step = 2  # type: ignore[attr-defined]
    runtime._last_pct_goal_request_identity = None  # type: ignore[attr-defined]
    runtime._last_pct_goal_sample = None  # type: ignore[attr-defined]
    runtime._step_calls = 0  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]
    return runtime, bridge


def test_runtime_publishes_body_velocity_and_fresh_cloud_with_same_time() -> None:
    runtime, bridge = _configured_runtime()

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    assert len(bridge.odometry_calls) == 1
    assert bridge.odometry_calls[0] == (
        (4.0, 5.0, 6.0),
        (1.0, 0.0, 0.0, 0.0),
        pytest.approx((0.1, 0.2, 0.3)),
        pytest.approx((0.4, 0.5, 0.6)),
        0.02,
    )
    assert len(bridge.cloud_calls) == 1
    points, cloud_timestamp = bridge.cloud_calls[0]
    np.testing.assert_allclose(points, [[1.0, 2.0, 5.0]])
    assert cloud_timestamp == 0.02
    report = runtime._metadata["navigation_ros2_last_publish_report"]  # type: ignore[attr-defined]
    assert report["odometry_published"] is True
    assert report["point_cloud_published"] is True
    assert report["timestamp"] == 0.02
    assert bridge.path_poll_count == 1


def test_runtime_exposes_live_reference_path_report_in_simulation_metadata() -> None:
    runtime, bridge = _configured_runtime()
    bridge.path_samples.append(
        OgnPathSample(
            points_ground_xyz=((0.0, 0.0, 0.0), (1.0, -0.2, 0.3)),
            terminal_yaw=0.4,
            source_topic="/initial_path",
            frame_id="world",
            stamp_sec=8,
            stamp_nanosec=125,
            sequence=4,
            points_sha256="a" * 64,
        )
    )

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    assert runtime._metadata["scan_reference_path_last_report"] == {  # type: ignore[attr-defined]
        "points_ground_xyz": [[0.0, 0.0, 0.0], [1.0, -0.2, 0.3]],
        "terminal_yaw": 0.4,
        "source": "ros2_nav_msgs_path",
        "topic": "/initial_path",
        "frame_id": "world",
        "stamp": {"sec": 8, "nanosec": 125},
        "sequence": 4,
        "points_sha256": "a" * 64,
        "cleared": False,
    }


@pytest.mark.parametrize("candidate_present", [False, True])
def test_runtime_exposes_typed_controller_status_with_exact_identity(
    candidate_present: bool,
) -> None:
    runtime, bridge = _configured_runtime()
    bridge.config = IsaacRos2OgnBridgeConfig(
        enable_pct_goal_publisher=True,
        enable_controller_status_subscription=True,
    )
    bridge.controller_status_samples.append(
        _controller_status_sample(candidate_present=candidate_present)
    )

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    assert bridge.controller_status_poll_timestamps == [0.02]
    report = runtime._metadata["scan_controller_status_last_report"]  # type: ignore[attr-defined]
    assert report == {
        "source": "ros2_scan_planner_msgs_controller_status",
        "topic": "/planning/controller_status",
        "receipt_timestamp": 0.02,
        "rx_sequence": 4,
        "header": {
            "frame_id": "world",
            "stamp": {"sec": 20, "nanosec": 1},
            "stamp_ns": 20_000_000_001,
        },
        "status_sequence": 9,
        "acceptance_sequence": 3,
        "event": 2 if candidate_present else 1,
        "state": 10,
        "reason": (
            "候选轨迹已拒绝" if candidate_present else "B-spline 已接受"
        ),
        "accepted": True,
        "trajectory_valid": True,
        "is_final": False,
        "emergency_stop": False,
        "active_sensing_yaw_only": False,
        "command_aggregate": {
            "sample_count": 0,
            "first_command": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "max_abs_vx": 0.0,
            "max_abs_vy": 0.0,
            "max_abs_wz": 0.0,
            "violation_count": 0,
        },
        "identity": {
            "reference_path_stamp": {"sec": 18, "nanosec": 999_999_999},
            "reference_path_stamp_ns": 18_999_999_999,
            "bspline_header_stamp": {"sec": 19, "nanosec": 2},
            "bspline_header_stamp_ns": 19_000_000_002,
            "start_time": {"sec": 19, "nanosec": 3},
            "start_time_ns": 19_000_000_003,
            "traj_id": 42,
        },
        "candidate": (
            {
                "reference_path_stamp": {"sec": 20, "nanosec": 10},
                "reference_path_stamp_ns": 20_000_000_010,
                "bspline_header_stamp": {"sec": 20, "nanosec": 11},
                "bspline_header_stamp_ns": 20_000_000_011,
                "start_time": {"sec": 20, "nanosec": 12},
                "start_time_ns": 20_000_000_012,
                "traj_id": 43,
            }
            if candidate_present
            else None
        ),
    }
    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    assert lifecycle["schema"] == "scan_controller_status_lifecycle_v1"
    assert lifecycle["sample_count"] == 1
    assert lifecycle["accepted_status_count"] == 1
    assert lifecycle["tracking_status_count"] == 1
    assert lifecycle["distinct_accepted_trajectory_count"] == 1
    assert lifecycle["candidate_rejection_count"] == int(candidate_present)


def test_controller_status_lifecycle_records_replacement_and_recovery() -> None:
    runtime, bridge = _configured_runtime()
    bridge.config = IsaacRos2OgnBridgeConfig(
        enable_pct_goal_publisher=True,
        enable_controller_status_subscription=True,
    )
    bridge.controller_status_samples.append(_controller_status_sample())
    runtime._publish_navigation_ros2_observation(completed_control_step=2)
    accepted = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_last_report"
    ]

    emergency = deepcopy(accepted)
    emergency.update(
        {
            "status_sequence": 10,
            "event": 4,
            "state": 8,
            "reason": "预测碰撞急停",
            "emergency_stop": True,
        }
    )
    runtime._update_scan_controller_status_lifecycle_report(  # type: ignore[attr-defined]
        emergency
    )
    recovered = deepcopy(accepted)
    recovered.update(
        {
            "status_sequence": 11,
            "acceptance_sequence": 4,
            "event": 1,
            "state": 10,
            "reason": "新局部轨迹已接受",
            "identity": {
                **accepted["identity"],
                "bspline_header_stamp": {"sec": 21, "nanosec": 2},
                "bspline_header_stamp_ns": 21_000_000_002,
                "start_time": {"sec": 21, "nanosec": 3},
                "start_time_ns": 21_000_000_003,
                "traj_id": 43,
            },
        }
    )
    runtime._update_scan_controller_status_lifecycle_report(  # type: ignore[attr-defined]
        recovered
    )

    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    assert lifecycle["sample_count"] == 3
    assert lifecycle["distinct_accepted_trajectory_count"] == 2
    assert lifecycle["trajectory_replacement_count"] == 1
    assert lifecycle["emergency_stop_status_count"] == 1
    assert lifecycle["tracking_after_emergency_stop_observed"] is True
    assert lifecycle["emergency_stop_recovery_count"] == 1
    assert lifecycle["emergency_stop_pending_recovery"] is False


def _dynamic_evidence_runtime(
    *,
    start_delay_s: float = 0.0,
    speed_mps: float = 1.0,
) -> IsaacLabNavigationRuntime:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    plan = _dynamic_evidence_test_plan(
        start_delay_s=start_delay_s,
        speed_mps=speed_mps,
    )
    runtime._dynamic_obstacle_plan = plan  # type: ignore[attr-defined]
    runtime._metadata = {  # type: ignore[attr-defined]
        "grid_map_observation_lifecycle_report": (
            runtime._new_grid_map_observation_lifecycle_report()
        ),
        "bspline_diagnostics_lifecycle_report": (
            runtime._new_bspline_diagnostics_lifecycle_report()
        ),
        "scan_controller_status_lifecycle_report": (
            runtime._new_scan_controller_status_lifecycle_report()
        ),
        "navigation_policy_gate_lifecycle_report": (
            runtime._new_navigation_policy_gate_lifecycle_report()
        ),
        "dynamic_navigation_evidence_report": (
            runtime._new_dynamic_navigation_evidence_report(plan)
        ),
    }
    return runtime


def _accepted_status_report(
    identity: dict[str, object],
    *,
    status_sequence: int,
    acceptance_sequence: int,
) -> dict[str, object]:
    return {
        "source": "ros2_scan_planner_msgs_controller_status",
        "topic": "/planning/controller_status",
        "receipt_timestamp": float(identity["bspline_header_stamp_ns"])
        * 1.0e-9,
        "rx_sequence": status_sequence,
        "header": {
            "frame_id": "world",
            "stamp": identity["bspline_header_stamp"],
            "stamp_ns": identity["bspline_header_stamp_ns"],
        },
        "status_sequence": status_sequence,
        "acceptance_sequence": acceptance_sequence,
        "event": 1,
        "state": 10,
        "reason": "B-spline 已接受并进入 TRACKING",
        "accepted": True,
        "trajectory_valid": True,
        "is_final": False,
        "emergency_stop": False,
        "identity": identity,
        "candidate": None,
    }


def _active_controller_status_report(
    identity: dict[str, object],
    *,
    status_sequence: int,
    command_sample_count: int,
    max_abs_wz: float,
) -> dict[str, object]:
    """构造与主动感知 identity 一致的 controller 命令聚合。"""

    report = _accepted_status_report(
        identity,
        status_sequence=status_sequence,
        acceptance_sequence=1,
    )
    report.update(
        {
            "state": 9,
            "reason": "主动感知 yaw-only 已接受",
            "active_sensing_yaw_only": True,
            "command_aggregate": {
                "sample_count": command_sample_count,
                "first_command": [0.0] * 6,
                "max_abs_vx": 0.0,
                "max_abs_vy": 0.0,
                "max_abs_wz": max_abs_wz,
                "violation_count": 0,
            },
        }
    )
    return report


def test_controller_lifecycle_classifies_only_goal_latched_same_path_rejection(
) -> None:
    """只把到达后同一 Path 的递增迟到候选归为可解释的安全拒绝。"""

    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._metadata = {  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report": (
            runtime._new_scan_controller_status_lifecycle_report()
        )
    }

    def identity_for(
        trajectory_id: int,
        *,
        stamp_ns: int,
    ) -> dict[str, object]:
        return {
            "reference_path_stamp": {"sec": 20, "nanosec": 1},
            "reference_path_stamp_ns": 20_000_000_001,
            "bspline_header_stamp": {
                "sec": stamp_ns // 1_000_000_000,
                "nanosec": stamp_ns % 1_000_000_000,
            },
            "bspline_header_stamp_ns": stamp_ns,
            "start_time": {
                "sec": stamp_ns // 1_000_000_000,
                "nanosec": stamp_ns % 1_000_000_000,
            },
            "start_time_ns": stamp_ns,
            "traj_id": trajectory_id,
        }

    identity = identity_for(3, stamp_ns=30_000_000_000)
    safe_rejection = _accepted_status_report(
        identity,
        status_sequence=1,
        acceptance_sequence=3,
    )
    safe_rejection.update(
        {
            "event": 2,
            "state": 12,
            "reason": "任意语言的到达后拒绝原因",
            "is_final": True,
            "candidate": {
                **identity_for(4, stamp_ns=31_000_000_000),
                "reference_path_stamp": identity["reference_path_stamp"],
                "reference_path_stamp_ns": identity[
                    "reference_path_stamp_ns"
                ],
            },
        }
    )

    runtime._update_scan_controller_status_lifecycle_report(safe_rejection)

    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    assert lifecycle["candidate_rejection_count"] == 1
    assert (
        lifecycle["goal_latched_same_path_candidate_rejection_count"] == 1
    )
    assert lifecycle["unexpected_candidate_rejection_count"] == 0

    running_rejection = deepcopy(safe_rejection)
    running_rejection.update(
        {
            "status_sequence": 2,
            "state": 10,
            "is_final": False,
            "reason": "运行中候选被拒绝",
        }
    )
    runtime._update_scan_controller_status_lifecycle_report(running_rejection)

    assert lifecycle["candidate_rejection_count"] == 2
    assert (
        lifecycle["goal_latched_same_path_candidate_rejection_count"] == 1
    )
    assert lifecycle["unexpected_candidate_rejection_count"] == 1


def _policy_write_report(
    command: tuple[float, float, float],
    *,
    timestamp: float,
) -> PolicyCommandWriteReport:
    """构造已经由唯一 adapter 实际写入 policy 的命令报告。"""

    body_command = BodyVelocityCommand(*command)
    return PolicyCommandWriteReport(
        timestamp=timestamp,
        owner_id="scan_cmd_vel",
        requested_command=body_command,
        limited_target=body_command,
        written_command=body_command,
        motion_allowed=True,
        stop_reasons=(),
        clipped_axes=(),
        rate_limited_axes=(),
    )


def test_active_sensing_lifecycle_aggregates_typed_chain_and_recovery() -> None:
    runtime, bridge = _configured_runtime()
    runtime._dynamic_obstacle_plan = resolve_dynamic_obstacle_plan({})  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = None  # type: ignore[attr-defined]
    observed_status = {
        "schema": "navigation_status_observed_diagnostics_v1",
        "status": {
            "status_sequence": 1,
            "state": 3,
            "pct_plan_id": 7,
            "active_path_stamp_ns": 1_000_000_001,
            "global_replan_requested": False,
            "global_replan_in_flight": False,
            "global_replan_request_id": 0,
            "consecutive_scan_failures": 0,
            "reason": "主动感知窗口",
            "identity_valid": True,
        },
    }
    bridge.navigation_status_observed_diagnostics = (  # type: ignore[attr-defined]
        lambda: deepcopy(observed_status)
    )

    started = _active_bspline_diagnostics_sample(
        diagnostic_sequence=1,
        receipt_timestamp=10.1,
        event=1,
    )
    runtime._update_bspline_diagnostics_lifecycle_report(started)
    identity = runtime._bspline_diagnostics_identity(started)
    first_controller = _active_controller_status_report(
        identity,
        status_sequence=1,
        command_sample_count=1,
        max_abs_wz=0.0,
    )
    runtime._metadata["scan_controller_status_last_report"] = first_controller  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(first_controller)

    # planner ACCEPTED typed 事件可能晚于 controller 到达；首条
    # 真实零写入先有界 pending，ACCEPTED 后按 write_sequence 回填。
    runtime._record_scan_policy_write_report(
        action=RobotAction.idle(source="active_pre_accepted_zero"),
        command_report=_policy_write_report((0.0, 0.0, 0.0), timestamp=10.15),
        temporary_inhibit_reason=None,
    )

    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=2,
            receipt_timestamp=10.2,
            event=2,
        )
    )
    runtime._record_scan_policy_write_report(
        action=RobotAction.idle(source="active_first_effective_zero"),
        command_report=_policy_write_report((0.0, 0.0, 0.0), timestamp=10.25),
        temporary_inhibit_reason=None,
    )
    runtime._record_scan_policy_write_report(
        action=RobotAction.idle(source="active_yaw_command"),
        command_report=_policy_write_report((0.0, 0.0, 0.15), timestamp=10.3),
        temporary_inhibit_reason=None,
    )
    final_controller = _active_controller_status_report(
        identity,
        status_sequence=2,
        command_sample_count=3,
        max_abs_wz=0.18,
    )
    runtime._metadata["scan_controller_status_last_report"] = final_controller  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(final_controller)

    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=3,
            receipt_timestamp=11.05,
            event=3,
            fusion_current=10,
        )
    )
    for sequence, timestamp in enumerate((11.1, 11.2, 11.3), start=1):
        runtime._update_grid_map_observation_lifecycle_report(
            _grid_diagnostics_sample(
                observation_sequence=sequence,
                timestamp_s=timestamp,
                hit_points=((1.0 + 0.1 * sequence, 0.0, 0.5),),
            )
        )
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=4,
            receipt_timestamp=11.35,
            event=4,
            fusion_current=13,
            fusion_distinct=3,
        )
    )
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=5,
            receipt_timestamp=11.4,
            event=5,
            fusion_current=13,
            fusion_distinct=3,
        )
    )

    recovery_sample = _bspline_diagnostics_sample(
        diagnostic_sequence=6,
        timestamp_s=12.0,
        traj_id=71,
        maximum_deviation_m=0.01,
    )
    runtime._update_bspline_diagnostics_lifecycle_report(recovery_sample)
    recovery_identity = runtime._bspline_diagnostics_identity(recovery_sample)
    recovery_status = _accepted_status_report(
        recovery_identity,
        status_sequence=3,
        acceptance_sequence=2,
    )
    recovery_status.update(
        {
            "active_sensing_yaw_only": False,
            "command_aggregate": {
                "sample_count": 1,
                "first_command": [0.0] * 6,
                "max_abs_vx": 0.0,
                "max_abs_vy": 0.0,
                "max_abs_wz": 0.0,
                "violation_count": 0,
            },
        }
    )
    runtime._metadata["scan_controller_status_last_report"] = recovery_status  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(recovery_status)
    observed_status["status"]["status_sequence"] = 2
    observed_status["status"]["reason"] = "主动感知后恢复运动"
    runtime._record_scan_policy_write_report(
        action=RobotAction.idle(source="active_recovery_tracking"),
        command_report=_policy_write_report((0.1, 0.0, 0.0), timestamp=12.1),
        temporary_inhibit_reason=None,
    )

    lifecycle = runtime._metadata["active_sensing_lifecycle_report"]  # type: ignore[attr-defined]
    assert lifecycle["schema"] == "active_sensing_lifecycle_v1"
    assert lifecycle["attempt_count"] == 1
    assert lifecycle["completed_attempt_count"] == 1
    assert lifecycle["failed_attempt_count"] == 0
    assert lifecycle["active_attempt_identity"] is None
    attempt = lifecycle["attempts"][0]
    assert attempt["identity"] == identity
    assert attempt["events"] == [
        "STARTED",
        "ACCEPTED",
        "YAW_STABLE",
        "FUSION_PROGRESS",
        "COMPLETED",
    ]
    assert all(attempt[name] is not None for name in (
        "started",
        "accepted",
        "yaw_stable",
        "completed",
    ))
    assert attempt["failed"] is None
    assert attempt["planner_fusion"] == {
        "baseline": 10,
        "current": 13,
        "distinct": 3,
        "required": 3,
    }
    fused = attempt["post_settle_fused_observations"]
    assert len(fused) == 3
    assert len({item["header_stamp_ns"] for item in fused}) == 3
    assert all(
        item["map_fusion_performed"] is True
        and item["accepted_endpoint_count"] > 0
        and item["header_stamp_ns"] > 11_000_000_000
        for item in fused
    )
    assert attempt["controller_command_aggregate"]["sample_count"] == 3
    assert attempt["controller_command_aggregate"]["first_command"] == [0.0] * 6
    assert attempt["controller_command_aggregate"]["max_abs_wz"] == pytest.approx(0.18)
    assert attempt["policy_command_aggregate"]["sample_count"] == 3
    assert attempt["policy_command_aggregate"]["first_command"] == [0.0, 0.0, 0.0]
    assert attempt["policy_command_aggregate"]["max_abs_wz"] == pytest.approx(0.15)
    assert attempt["policy_command_aggregate"]["violation_count"] == 0
    assert attempt["policy_command_aggregate"]["first_rotation_write"][
        "written_command"
    ] == [0.0, 0.0, 0.15]
    assert attempt["policy_command_aggregate"]["maximum_abs_wz_write"][
        "written_command"
    ] == [0.0, 0.0, 0.15]
    assert attempt["policy_command_aggregate"]["first_write"][
        "scan_controller_status_snapshot"
    ]["identity"] == identity
    assert attempt["policy_command_aggregate"]["last_write"][
        "scan_controller_status_snapshot"
    ]["active_sensing_yaw_only"] is True
    assert attempt["pct_plan_ids"] == [7]
    assert attempt["recovery"] == {
        "identity": recovery_identity,
        "reference_path_stamp_ns": 1_000_000_001,
        "pct_plan_id": 7,
        "stationary": False,
        "controller_state": 10,
    }

    bspline_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "bspline_diagnostics_lifecycle_report"
    ]
    assert bspline_lifecycle["sample_count"] == 1
    assert bspline_lifecycle["active_sensing_diagnostic_count"] == 5
    assert bspline_lifecycle["distinct_trajectory_identity_count"] == 1
    controller_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    assert controller_lifecycle["sample_count"] == 1
    assert controller_lifecycle["active_sensing_status_count"] == 2
    assert controller_lifecycle["accepted_status_count"] == 1
    assert controller_lifecycle["distinct_accepted_trajectory_count"] == 1
    assert controller_lifecycle["trajectory_replacement_count"] == 0


def test_active_sensing_cross_topic_races_backfill_after_terminal() -> None:
    """独立 topic 乱序只按完整 identity 回填，终态立即关闭命令窗口。"""

    runtime, _bridge = _configured_runtime()
    runtime._dynamic_obstacle_plan = resolve_dynamic_obstacle_plan({})  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = None  # type: ignore[attr-defined]
    started = _active_bspline_diagnostics_sample(
        diagnostic_sequence=1,
        receipt_timestamp=10.1,
        event=1,
    )
    identity = runtime._bspline_diagnostics_identity(started)
    early_controller = _active_controller_status_report(
        identity,
        status_sequence=1,
        command_sample_count=1,
        max_abs_wz=0.0,
    )

    # ControllerStatus 先于 planner STARTED 到达。
    runtime._metadata["scan_controller_status_last_report"] = early_controller  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(early_controller)
    lifecycle = runtime._metadata["active_sensing_lifecycle_report"]  # type: ignore[attr-defined]
    assert lifecycle["attempts"] == []
    assert lifecycle["pending_active_controller_statuses"] == [
        early_controller
    ]

    runtime._update_bspline_diagnostics_lifecycle_report(started)
    attempt = lifecycle["attempts"][0]
    assert lifecycle["pending_active_controller_statuses"] == []
    assert attempt["controller_command_aggregate"]["sample_count"] == 1
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=2,
            receipt_timestamp=10.2,
            event=2,
        )
    )
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=3,
            receipt_timestamp=11.05,
            event=3,
            fusion_current=10,
        )
    )
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=4,
            receipt_timestamp=11.35,
            event=4,
            fusion_current=13,
            fusion_distinct=3,
        )
    )
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=5,
            receipt_timestamp=11.4,
            event=5,
            fusion_current=13,
            fusion_distinct=3,
        )
    )

    # planner 已 COMPLETED 而 controller topic 仍是旧 active 快照时，
    # 普通恢复命令不得被误计为 active 命令。
    runtime._record_scan_policy_write_report(
        action=RobotAction.idle(source="normal_recovery_before_status_catchup"),
        command_report=_policy_write_report(
            (0.1, 0.0, 0.0),
            timestamp=11.45,
        ),
        temporary_inhibit_reason=None,
    )
    assert attempt["policy_window_open"] is False
    assert attempt["policy_command_aggregate"]["sample_count"] == 0

    # COMPLETED 先于 GridMap 诊断被 runtime 观测；三条真实
    # 后 settle 融合仍必须回填到已完成 attempt。
    assert attempt["post_settle_fused_observations"] == []
    future_source = replace(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=11.41,
            hit_points=((1.05, 0.0, 0.5),),
        ),
        receipt_timestamp=11.6,
    )
    runtime._update_grid_map_observation_lifecycle_report(future_source)
    assert attempt["post_settle_fused_observations"] == []
    for sequence, timestamp in enumerate((11.1, 11.2, 11.3), start=2):
        late_sample = replace(
            _grid_diagnostics_sample(
                observation_sequence=sequence,
                timestamp_s=timestamp,
                hit_points=((1.0 + 0.1 * sequence, 0.0, 0.5),),
            ),
            receipt_timestamp=11.5 + 0.1 * sequence,
        )
        runtime._update_grid_map_observation_lifecycle_report(late_sample)
    assert len(attempt["post_settle_fused_observations"]) == 3

    # recovery ControllerStatus 先于普通 B-spline diagnostics 到达。
    recovery_sample = _bspline_diagnostics_sample(
        diagnostic_sequence=6,
        timestamp_s=12.0,
        traj_id=71,
        maximum_deviation_m=0.01,
    )
    recovery_identity = runtime._bspline_diagnostics_identity(recovery_sample)
    recovery_status = _accepted_status_report(
        recovery_identity,
        status_sequence=2,
        acceptance_sequence=2,
    )
    runtime._metadata["scan_controller_status_last_report"] = recovery_status  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(recovery_status)
    assert lifecycle["pending_recovery_controller_statuses"] == [
        recovery_status
    ]

    runtime._update_bspline_diagnostics_lifecycle_report(recovery_sample)
    assert lifecycle["pending_recovery_controller_statuses"] == []
    assert attempt["recovery"]["identity"] == recovery_identity
    assert attempt["recovery"]["controller_state"] == 10


def test_active_sensing_controller_pending_overflow_fails_closed() -> None:
    """长时缺少 planner STARTED 不能无界缓存 controller 证据。"""

    runtime, _bridge = _configured_runtime()
    started = _active_bspline_diagnostics_sample(
        diagnostic_sequence=1,
        receipt_timestamp=10.1,
        event=1,
    )
    identity = runtime._bspline_diagnostics_identity(started)
    for sequence in range(1, ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT + 1):
        runtime._update_active_sensing_from_controller_status(
            _active_controller_status_report(
                identity,
                status_sequence=sequence,
                command_sample_count=sequence,
                max_abs_wz=0.1,
            )
        )

    lifecycle = runtime._metadata["active_sensing_lifecycle_report"]  # type: ignore[attr-defined]
    assert len(lifecycle["pending_active_controller_statuses"]) == (
        ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT
    )
    with pytest.raises(RuntimeError, match="active controller pending"):
        runtime._update_active_sensing_from_controller_status(
            _active_controller_status_report(
                identity,
                status_sequence=(
                    ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT + 1
                ),
                command_sample_count=(
                    ACTIVE_SENSING_PENDING_CONTROLLER_STATUS_LIMIT + 1
                ),
                max_abs_wz=0.1,
            )
        )


def test_active_sensing_zero_gate_records_real_first_policy_write() -> None:
    """KeepLast(1) 即使只留下 yaw，新 active identity 的首次实写仍为零。"""

    runtime, _unused_bridge = _configured_runtime()
    runtime._dynamic_obstacle_plan = resolve_dynamic_obstacle_plan({})  # type: ignore[attr-defined]
    started = _active_bspline_diagnostics_sample(
        diagnostic_sequence=1,
        receipt_timestamp=10.1,
        event=1,
    )
    runtime._update_bspline_diagnostics_lifecycle_report(started)
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=2,
            receipt_timestamp=10.2,
            event=2,
        )
    )
    identity = runtime._bspline_diagnostics_identity(started)
    active_status = _active_controller_status_report(
        identity,
        status_sequence=1,
        command_sample_count=1,
        max_abs_wz=0.0,
    )
    runtime._metadata["scan_controller_status_last_report"] = active_status  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(active_status)

    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=100.0,
            max_vy_rate=100.0,
            max_wz_rate=100.0,
            require_odometry=False,
            require_point_cloud=False,
            require_navigation_status=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    bridge = _FakeCommandBridge(
        [
            # controller 首拍零已被 KeepLast(1) 后续 yaw 覆盖。
            OgnTwistSample(
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.15),
                receipt_timestamp=0.02,
                sequence=1,
            )
        ]
    )
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    baseline_write_count = len(sink.commands)
    try:
        first = runtime._apply_scan_cmd_vel_to_policy(
            environment_terminated=False,
        )
        runtime._record_scan_policy_write_report(
            action=RobotAction.idle(source="active_identity_zero_gate"),
            command_report=first,
            temporary_inhibit_reason=None,
        )

        assert bridge.samples == []
        assert first.written_command.as_tuple() == (0.0, 0.0, 0.0)
        assert first.stop_reasons == ("active_sensing_identity_zero_gate",)
        assert sink.commands[baseline_write_count:] == [(0.0, 0.0, 0.0)]
        lifecycle = runtime._metadata["active_sensing_lifecycle_report"]  # type: ignore[attr-defined]
        assert lifecycle["policy_zero_gate"] is None
        assert lifecycle["policy_zero_gate_armed_count"] == 1
        assert lifecycle["policy_zero_gate_consumed_count"] == 1

        # 同 identity 的后续 controller 累计快照不能重新布署零门。
        later_status = _active_controller_status_report(
            identity,
            status_sequence=2,
            command_sample_count=2,
            max_abs_wz=0.15,
        )
        runtime._metadata["scan_controller_status_last_report"] = later_status  # type: ignore[attr-defined]
        runtime._update_scan_controller_status_lifecycle_report(later_status)
        assert lifecycle["policy_zero_gate"] is None

        bridge.samples.append(
            OgnTwistSample(
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.15),
                receipt_timestamp=0.04,
                sequence=2,
            )
        )
        runtime._ros2_physics_step_count = 16  # type: ignore[attr-defined]
        yaw = runtime._apply_scan_cmd_vel_to_policy(
            environment_terminated=False,
        )
        runtime._record_scan_policy_write_report(
            action=RobotAction.idle(source="active_yaw_after_zero_gate"),
            command_report=yaw,
            temporary_inhibit_reason=None,
        )
        assert yaw.written_command.as_tuple() == pytest.approx(
            (0.0, 0.0, 0.15)
        )
        attempt = lifecycle["attempts"][0]
        aggregate = attempt["policy_command_aggregate"]
        assert aggregate["sample_count"] == 2
        assert aggregate["first_command"] == [0.0, 0.0, 0.0]
        assert aggregate["max_abs_wz"] == pytest.approx(0.15)
        assert aggregate["violation_count"] == 0
        assert aggregate["first_write"]["written_command"] == [
            0.0,
            0.0,
            0.0,
        ]
        assert aggregate["first_write"]["navigation_gate_diagnostics"] == (
            aggregate["first_write"][
                "policy_navigation_gate_consumed_report"
            ]
        )
        assert aggregate["first_write"]["motion_allowed"] is False
        assert aggregate["first_write"]["stop_reasons"] == [
            "active_sensing_identity_zero_gate"
        ]
        assert aggregate["first_write"]["navigation_gate_diagnostics"][
            "command_identity"
        ] is None
        assert aggregate["first_write"]["navigation_gate_diagnostics"][
            "command_identity_matches_permit"
        ] is False
        assert aggregate["first_rotation_write"]["written_command"] == [
            0.0,
            0.0,
            0.15,
        ]
        assert aggregate["maximum_abs_wz_write"] == (
            aggregate["first_rotation_write"]
        )
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.04)


def test_active_sensing_zero_gate_without_pending_twist_is_valid() -> None:
    """没有待 drain Twist 时，active 首拍仍必须由唯一 owner 真写零。"""

    runtime, _unused_bridge = _configured_runtime()
    runtime._dynamic_obstacle_plan = resolve_dynamic_obstacle_plan({})  # type: ignore[attr-defined]
    started = _active_bspline_diagnostics_sample(
        diagnostic_sequence=1,
        receipt_timestamp=10.1,
        event=1,
    )
    runtime._update_bspline_diagnostics_lifecycle_report(started)
    runtime._update_bspline_diagnostics_lifecycle_report(
        _active_bspline_diagnostics_sample(
            diagnostic_sequence=2,
            receipt_timestamp=10.2,
            event=2,
        )
    )
    identity = runtime._bspline_diagnostics_identity(started)
    active_status = _active_controller_status_report(
        identity,
        status_sequence=1,
        command_sample_count=1,
        max_abs_wz=0.0,
    )
    runtime._metadata["scan_controller_status_last_report"] = active_status  # type: ignore[attr-defined]
    runtime._update_scan_controller_status_lifecycle_report(active_status)

    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=100.0,
            max_vy_rate=100.0,
            max_wz_rate=100.0,
            require_odometry=False,
            require_point_cloud=False,
            require_navigation_status=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    runtime._ros2_ogn_bridge = _FakeCommandBridge([])  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    try:
        report = runtime._apply_scan_cmd_vel_to_policy(
            environment_terminated=False,
        )
        runtime._record_scan_policy_write_report(
            action=RobotAction.idle(source="active_zero_without_twist"),
            command_report=report,
            temporary_inhibit_reason=None,
        )

        attempt = runtime._metadata["active_sensing_lifecycle_report"][  # type: ignore[attr-defined]
            "attempts"
        ][0]
        first_write = attempt["policy_command_aggregate"]["first_write"]
        assert report.written_command.as_tuple() == (0.0, 0.0, 0.0)
        assert first_write["cmd_vel_sample_drained_this_tick"] is False
        assert first_write["motion_allowed"] is False
        assert first_write["navigation_gate_diagnostics"][
            "command_identity"
        ] is None
        assert first_write["navigation_gate_diagnostics"][
            "command_identity_matches_permit"
        ] is False
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.04)


def test_dynamic_typed_evidence_requires_hit_detour_clear_and_recovery() -> None:
    runtime = _dynamic_evidence_runtime()
    hit = _grid_diagnostics_sample(
        observation_sequence=1,
        timestamp_s=1.0,
        hit_points=((1.0, 0.0, 0.5),),
    )
    clear = _grid_diagnostics_sample(
        observation_sequence=2,
        timestamp_s=2.0,
        explicit_free_points=((1.0, 0.0, 0.5),),
        explicit_free_miss_voxel_count=3,
        occupied_to_free_count=1,
    )
    detour = _bspline_diagnostics_sample(
        diagnostic_sequence=1,
        timestamp_s=1.5,
        traj_id=11,
        maximum_deviation_m=0.08,
    )
    recovery = _bspline_diagnostics_sample(
        diagnostic_sequence=2,
        timestamp_s=2.5,
        traj_id=12,
        maximum_deviation_m=0.01,
    )

    runtime._update_grid_map_observation_lifecycle_report(hit)
    runtime._update_grid_map_observation_lifecycle_report(clear)
    runtime._update_bspline_diagnostics_lifecycle_report(detour)
    runtime._update_bspline_diagnostics_lifecycle_report(recovery)
    detour_identity = runtime._bspline_diagnostics_identity(detour)
    recovery_identity = runtime._bspline_diagnostics_identity(recovery)
    runtime._update_scan_controller_status_lifecycle_report(
        _accepted_status_report(
            detour_identity,
            status_sequence=1,
            acceptance_sequence=1,
        )
    )
    runtime._update_scan_controller_status_lifecycle_report(
        _accepted_status_report(
            recovery_identity,
            status_sequence=2,
            acceptance_sequence=2,
        )
    )
    policy_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "navigation_policy_gate_lifecycle_report"
    ]
    controller_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    detour_tracking_write = {
        "write_sequence": 19,
        "timestamp": 1.75,
        "written_command": [0.1, 0.0, 0.0],
        "navigation_gate_diagnostics": {"identity_valid": True},
        "scan_controller_status_snapshot": controller_lifecycle[
            "tracking_status_reports"
        ][0],
    }
    recovery_tracking_write = {
        "write_sequence": 20,
        "timestamp": 2.6,
        "written_command": [0.1, 0.0, 0.0],
        "navigation_gate_diagnostics": {"identity_valid": True},
        "scan_controller_status_snapshot": controller_lifecycle[
            "tracking_status_reports"
        ][1],
    }
    policy_lifecycle["identity_verified_tracking_write_count"] = 2
    policy_lifecycle["first_identity_verified_tracking_write"] = (
        detour_tracking_write
    )
    policy_lifecycle["last_identity_verified_tracking_write"] = (
        recovery_tracking_write
    )
    policy_lifecycle["identity_verified_tracking_write_reports"] = [
        detour_tracking_write,
        recovery_tracking_write,
    ]

    evidence = runtime._refresh_dynamic_navigation_evidence_report()

    assert evidence["schema"] == "dynamic_navigation_evidence_v1"
    assert evidence["enabled"] is True
    assert evidence["verified"] is True
    assert evidence["post_filter_hit"]["verified"] is True
    assert evidence["post_filter_hit"]["observation_sequence"] == 1
    assert evidence["ordered_detour"]["verified"] is True
    assert evidence["ordered_detour"]["identity"] == detour_identity
    assert evidence["ordered_detour"]["controller_tracking_status"] == (
        controller_lifecycle["tracking_status_reports"][0]
    )
    assert evidence["ordered_detour"][
        "policy_identity_verified_tracking_write"
    ] == detour_tracking_write
    assert evidence["current_obstacle_clearance"]["verified"] is True
    assert evidence["current_obstacle_clearance"]["obstacle_clearances"][0][
        "continuous_clearance_verified"
    ] is True
    assert evidence["explicit_miss_ghost_clear"]["verified"] is True
    assert evidence["explicit_miss_ghost_clear"][
        "occupied_removed_by_sliding_reset_count"
    ] == 0
    clear_match = evidence["explicit_miss_ghost_clear"]["clear_matches"][0]
    assert clear_match["obstacle_path_distance_separation_m"] == (
        pytest.approx(1.0)
    )
    assert clear_match["obstacle_pose_position_separation_m"] == (
        pytest.approx(1.0)
    )
    assert clear_match["minimum_obstacle_motion_separation_m"] == (
        pytest.approx(0.05)
    )
    assert clear_match["obstacle_motion_separation_verified"] is True
    assert evidence["trajectory_recovery"]["verified"] is True
    assert evidence["trajectory_recovery"][
        "before_detour_identity"
    ] == detour_identity
    assert evidence["trajectory_recovery"][
        "after_recovery_identity"
    ] == recovery_identity


@pytest.mark.parametrize(
    ("start_delay_s", "speed_mps", "point_world_xyz"),
    (
        (10.0, 1.0, (0.299, 0.0, 0.5)),
        (0.0, 0.04, (-0.215, 0.0, 0.5)),
    ),
    ids=("obstacle_not_started", "motion_below_one_voxel"),
)
def test_dynamic_typed_evidence_rejects_clear_without_measurable_motion(
    start_delay_s: float,
    speed_mps: float,
    point_world_xyz: tuple[float, float, float],
) -> None:
    runtime = _dynamic_evidence_runtime(
        start_delay_s=start_delay_s,
        speed_mps=speed_mps,
    )
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=1.0,
            hit_points=(point_world_xyz,),
        )
    )
    clear_report = runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=2,
            timestamp_s=2.0,
            explicit_free_points=(point_world_xyz,),
            explicit_free_miss_voxel_count=3,
            occupied_to_free_count=1,
        )
    )

    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "grid_map_observation_lifecycle_report"
    ]
    assert lifecycle["first_hit_report"] is not None
    assert clear_report[
        "dynamic_obstacle_explicit_miss_clear_matches"
    ] == []
    assert lifecycle["last_explicit_miss_clear_report"] is None
    assert lifecycle[
        "dynamic_obstacle_explicit_miss_clear_match_count"
    ] == 0
    evidence = runtime._refresh_dynamic_navigation_evidence_report()
    assert evidence["explicit_miss_ghost_clear"]["verified"] is False


def test_dynamic_typed_evidence_recomputes_clear_motion_fail_closed() -> None:
    runtime = _dynamic_evidence_runtime()
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=1.0,
            hit_points=((1.0, 0.0, 0.5),),
        )
    )
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=2,
            timestamp_s=2.0,
            explicit_free_points=((1.0, 0.0, 0.5),),
            explicit_free_miss_voxel_count=3,
            occupied_to_free_count=1,
        )
    )
    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "grid_map_observation_lifecycle_report"
    ]
    clear_report = lifecycle["last_explicit_miss_clear_report"]
    clear_match = clear_report[
        "dynamic_obstacle_explicit_miss_clear_matches"
    ][0]
    clear_match["obstacle_state_after_clear"]["path_distance_m"] = (
        clear_match["obstacle_state_at_hit"]["path_distance_m"]
    )

    evidence = runtime._refresh_dynamic_navigation_evidence_report()

    assert evidence["explicit_miss_ghost_clear"]["verified"] is False


def test_bspline_clearance_uses_continuous_curve_lower_bound() -> None:
    runtime = _dynamic_evidence_runtime()
    sample = replace(
        _bspline_diagnostics_sample(
            diagnostic_sequence=1,
            timestamp_s=1.5,
            traj_id=11,
            maximum_deviation_m=0.08,
        ),
        maximum_velocity_upper_bound=100.0,
    )

    report = runtime._update_bspline_diagnostics_lifecycle_report(sample)

    clearance = report["dynamic_obstacle_clearances"][0]
    assert clearance["sampling_clearance_margin_m"] == pytest.approx(0.5)
    assert clearance["continuous_clearance_lower_bound_m"] < 0.43
    assert clearance["continuous_clearance_verified"] is False
    assert report["ordered_detour_candidate"] is False


def test_detour_policy_write_at_clear_receipt_is_not_pre_clear_execution() -> None:
    runtime = _dynamic_evidence_runtime()
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=1.0,
            hit_points=((1.0, 0.0, 0.5),),
        )
    )
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=2,
            timestamp_s=2.0,
            explicit_free_points=((1.0, 0.0, 0.5),),
            explicit_free_miss_voxel_count=3,
            occupied_to_free_count=1,
        )
    )
    detour = _bspline_diagnostics_sample(
        diagnostic_sequence=1,
        timestamp_s=1.5,
        traj_id=11,
        maximum_deviation_m=0.08,
    )
    runtime._update_bspline_diagnostics_lifecycle_report(detour)
    identity = runtime._bspline_diagnostics_identity(detour)
    runtime._update_scan_controller_status_lifecycle_report(
        _accepted_status_report(
            identity,
            status_sequence=1,
            acceptance_sequence=1,
        )
    )
    controller_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_lifecycle_report"
    ]
    write = {
        "write_sequence": 1,
        "timestamp": 2.0,
        "written_command": [0.1, 0.0, 0.0],
        "navigation_gate_diagnostics": {"identity_valid": True},
        "scan_controller_status_snapshot": controller_lifecycle[
            "last_tracking_status"
        ],
    }
    policy_lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "navigation_policy_gate_lifecycle_report"
    ]
    policy_lifecycle["identity_verified_tracking_write_count"] = 1
    policy_lifecycle["first_identity_verified_tracking_write"] = write
    policy_lifecycle["last_identity_verified_tracking_write"] = write
    policy_lifecycle["identity_verified_tracking_write_reports"] = [write]

    evidence = runtime._refresh_dynamic_navigation_evidence_report()

    assert evidence["ordered_detour"]["verified"] is False
    assert evidence["current_obstacle_clearance"]["verified"] is False


def test_dynamic_typed_evidence_keeps_sample_provenance_with_unrelated_sliding_reset() -> None:
    runtime = _dynamic_evidence_runtime()
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=1.0,
            hit_points=((1.0, 0.0, 0.5),),
        )
    )
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=2,
            timestamp_s=2.0,
            explicit_free_points=((1.0, 0.0, 0.5),),
            explicit_free_miss_voxel_count=3,
            occupied_to_free_count=1,
            sliding_reset_count=1,
        )
    )

    evidence = runtime._refresh_dynamic_navigation_evidence_report()

    assert evidence["post_filter_hit"]["verified"] is True
    assert evidence["explicit_miss_ghost_clear"]["verified"] is True
    lifecycle = runtime._metadata[  # type: ignore[attr-defined]
        "grid_map_observation_lifecycle_report"
    ]
    assert lifecycle[
        "dynamic_obstacle_explicit_miss_clear_match_count"
    ] == 1
    assert lifecycle[
        "total_occupied_removed_by_sliding_reset_count"
    ] == 1


def test_dynamic_typed_evidence_rejects_sliding_reset_without_transition_provenance() -> None:
    runtime = _dynamic_evidence_runtime()
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=1,
            timestamp_s=1.0,
            hit_points=((1.0, 0.0, 0.5),),
        )
    )
    runtime._update_grid_map_observation_lifecycle_report(
        _grid_diagnostics_sample(
            observation_sequence=2,
            timestamp_s=2.0,
            explicit_free_points=((1.0, 0.0, 0.5),),
            explicit_free_miss_voxel_count=3,
            occupied_to_free_count=1,
            sliding_reset_count=1,
            transition_hit_sequences=(0,),
        )
    )

    evidence = runtime._refresh_dynamic_navigation_evidence_report()

    assert evidence["explicit_miss_ghost_clear"]["verified"] is False


def test_runtime_exposes_tombstone_without_terminal_yaw() -> None:
    runtime, bridge = _configured_runtime()
    bridge.path_samples.append(
        OgnPathSample(
            points_ground_xyz=(),
            terminal_yaw=None,
            source_topic="/initial_path",
            frame_id="world",
            stamp_sec=9,
            stamp_nanosec=250,
            sequence=5,
            points_sha256="b" * 64,
        )
    )

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    report = runtime._metadata["scan_reference_path_last_report"]  # type: ignore[attr-defined]
    assert report["points_ground_xyz"] == []
    assert report["terminal_yaw"] is None
    assert report["cleared"] is True


def test_runtime_publishes_each_pipeline_pct_goal_generation_exactly_once() -> None:
    runtime, bridge = _configured_runtime()
    runtime._step_calls = 2  # type: ignore[attr-defined]
    runtime._last_pct_goal_request_identity = None  # type: ignore[attr-defined]
    request = {
        "generation": 1,
        "frame_id": "world",
        "position_base_xyz": (1.0, 2.0, 0.48),
        "yaw": 0.4,
        "height_semantics": "base",
    }
    action = RobotAction(
        source="scan_pct_goal_publish",
        metadata={"navigation_pct_goal_request": request},
    )

    runtime._publish_pct_goal_request(action)
    runtime._publish_pct_goal_request(action)

    assert bridge.goal_calls == [((1.0, 2.0, 0.48), 0.4, 20_000_000, "world")]
    assert runtime._metadata["scan_pct_goal_last_report"] == {  # type: ignore[attr-defined]
        "published": True,
        "source": "isaac_ros2_ogn_pose_stamped",
        "topic": "/pct/goal",
        "frame_id": "world",
        "stamp": {"sec": 0, "nanosec": 20_000_000},
        "sequence": 1,
        "generation": 1,
        "position_base_xyz": [1.0, 2.0, 0.48],
        "yaw": 0.4,
        "height_semantics": "base",
        "published_at_control_step": 2,
        "transport_attempt_count": 1,
        "first_transport_attempt_control_step": 2,
        "last_transport_attempt_control_step": 2,
        "dds_acknowledged": False,
    }

    next_request = dict(request)
    next_request.update(
        {
            "generation": 2,
            "position_base_xyz": (3.0, 4.0, 3.62),
        }
    )
    runtime._ros2_physics_step_count = 9  # type: ignore[attr-defined]
    runtime._publish_pct_goal_request(
        RobotAction(
            source="scan_pct_goal_publish",
            metadata={"navigation_pct_goal_request": next_request},
        )
    )
    assert len(bridge.goal_calls) == 2
    assert bridge.goal_calls[-1][0] == (3.0, 4.0, 3.62)


def test_runtime_retries_same_stamped_pct_goal_without_new_generation() -> None:
    runtime, bridge = _configured_runtime()
    runtime._step_calls = 2  # type: ignore[attr-defined]
    request = {
        "generation": 1,
        "frame_id": "world",
        "position_base_xyz": (1.0, 2.0, 0.48),
        "yaw": 0.4,
        "height_semantics": "base",
    }
    runtime._publish_pct_goal_request(
        RobotAction(metadata={"navigation_pct_goal_request": request})
    )
    first_sample = runtime._last_pct_goal_sample  # type: ignore[attr-defined]

    runtime._step_calls = 7  # type: ignore[attr-defined]
    runtime._publish_pct_goal_request(
        RobotAction(
            metadata={
                "navigation_pct_goal_request": {
                    **request,
                    "transport_retry": True,
                }
            }
        )
    )

    assert len(bridge.goal_calls) == 2
    assert bridge.goal_calls[0] == bridge.goal_calls[1]
    assert runtime._last_pct_goal_sample == first_sample  # type: ignore[attr-defined]
    report = runtime._metadata["scan_pct_goal_last_report"]  # type: ignore[attr-defined]
    assert report["sequence"] == 1
    assert report["stamp"] == {"sec": 0, "nanosec": 20_000_000}
    assert report["transport_attempt_count"] == 2
    assert report["first_transport_attempt_control_step"] == 2
    assert report["last_transport_attempt_control_step"] == 7


def _effective_goal_provenance(
    position_base_xyz: tuple[float, float, float],
) -> dict[str, object]:
    configured_height = 0.30
    collision_sha = "a" * 64
    return {
        "schema": "pct_effective_goal_height_v1",
        "height_semantics": "collision_ground_plus_configured_body_height",
        "formula": "effective_base_z=collision_ground_z+configured_body_height_m",
        "configured_body_height_m": configured_height,
        "raw_task_goal_z": 0.62,
        "raw_task_z_used_as_height_evidence": False,
        "calibration": {
            "configured_body_height_hint_m": configured_height,
            "collision_ply_sha256": collision_sha,
        },
        "projection": {
            "configured_body_height_hint_m": configured_height,
            "collision_ply_sha256": collision_sha,
            "projected_base_sim_xyz": list(position_base_xyz),
        },
    }


def test_runtime_requires_and_echoes_effective_goal_provenance() -> None:
    runtime, bridge = _configured_runtime()
    runtime._step_calls = 2  # type: ignore[attr-defined]
    runtime._last_pct_goal_request_identity = None  # type: ignore[attr-defined]
    position = (1.0, 2.0, 0.48)
    provenance = _effective_goal_provenance(position)
    request = {
        "generation": 1,
        "frame_id": "world",
        "position_base_xyz": position,
        "yaw": 0.4,
        "height_semantics": "base",
        "effective_goal_provenance_required": True,
        "effective_goal_provenance": provenance,
    }

    runtime._publish_pct_goal_request(
        RobotAction(metadata={"navigation_pct_goal_request": request})
    )

    assert len(bridge.goal_calls) == 1
    report = runtime._metadata["scan_pct_goal_last_report"]  # type: ignore[attr-defined]
    assert report["effective_goal_provenance_required"] is True
    assert report["effective_goal_provenance"] == provenance

    rejected_runtime, rejected_bridge = _configured_runtime()
    rejected_runtime._step_calls = 2  # type: ignore[attr-defined]
    rejected_runtime._last_pct_goal_request_identity = None  # type: ignore[attr-defined]
    tampered = dict(request)
    tampered["position_base_xyz"] = (1.0, 2.0, 0.49)
    with pytest.raises(ValueError, match="provenance 不一致"):
        rejected_runtime._publish_pct_goal_request(
            RobotAction(metadata={"navigation_pct_goal_request": tampered})
        )
    assert rejected_bridge.goal_calls == []


def test_runtime_rejects_changed_payload_with_same_pct_goal_generation() -> None:
    runtime, _bridge = _configured_runtime()
    runtime._step_calls = 2  # type: ignore[attr-defined]
    runtime._last_pct_goal_request_identity = None  # type: ignore[attr-defined]
    base = {
        "generation": 1,
        "frame_id": "world",
        "position_base_xyz": (1.0, 2.0, 0.48),
        "yaw": 0.4,
        "height_semantics": "base",
    }
    runtime._publish_pct_goal_request(
        RobotAction(metadata={"navigation_pct_goal_request": base})
    )
    changed = dict(base)
    changed["yaw"] = 0.5
    with pytest.raises(RuntimeError, match="不能改变"):
        runtime._publish_pct_goal_request(
            RobotAction(metadata={"navigation_pct_goal_request": changed})
        )


def test_runtime_does_not_retimestamp_stale_depth() -> None:
    runtime, bridge = _configured_runtime()
    runtime._last_camera_render_step = 3  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 16  # type: ignore[attr-defined]

    runtime._publish_navigation_ros2_observation(completed_control_step=4)

    assert len(bridge.odometry_calls) == 1
    assert bridge.odometry_calls[0][-1] == 0.04
    assert bridge.cloud_calls == []
    report = runtime._metadata["navigation_ros2_last_publish_report"]  # type: ignore[attr-defined]
    assert report["point_cloud_due"] is True
    assert report["point_cloud_published"] is False
    assert report["point_cloud_skip_reason"] == "stale_or_unrendered_depth_rejected"


def test_runtime_does_not_publish_or_refresh_sparse_depth_cloud() -> None:
    runtime, bridge = _configured_runtime()
    gate = _FakeFreshnessGate()
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=8.0,
            minimum_valid_points=2,
            publish_interval_control_steps=2,
        ),
    )

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    assert bridge.cloud_calls == []
    assert gate.odometry_stamps == [0.02]
    assert gate.point_cloud_stamps == []
    report = runtime._metadata["navigation_ros2_last_publish_report"]  # type: ignore[attr-defined]
    assert report["point_cloud_due"] is True
    assert report["point_cloud_published"] is False
    assert report["point_cloud_point_count"] == 1
    assert report["point_cloud_skip_reason"] == "insufficient_valid_points"


def test_runtime_does_not_publish_or_refresh_empty_raw_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bridge = _configured_runtime()
    gate = _FakeFreshnessGate()
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(
            pixel_stride=1,
            min_depth_m=0.0,
            max_depth_m=8.0,
            minimum_valid_points=64,
            publish_interval_control_steps=2,
        ),
    )
    monkeypatch.setattr(
        "source.simulation.isaaclab_runtime.camera_sensor_to_world_points",
        lambda _sensor, _config: np.empty((0, 3), dtype=np.float32),
    )

    runtime._publish_navigation_ros2_observation(completed_control_step=2)

    assert bridge.cloud_calls == []
    assert gate.point_cloud_stamps == []
    report = runtime._metadata["navigation_ros2_last_publish_report"]  # type: ignore[attr-defined]
    assert report["point_cloud_published"] is False
    assert report["point_cloud_point_count"] == 0
    assert report["point_cloud_skip_reason"] == "insufficient_valid_points"


def test_ros2_config_is_paired_and_world_framed() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
    )
    with pytest.raises(ValueError, match="同时配置"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            odom_frame_id="map",
            point_cloud_frame_id="map",
        ),
        depth_point_cloud_config=DepthPointCloudConfig(),
        hide_navigation_collision_visual=False,
    )
    with pytest.raises(ValueError, match="world"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(),
        hide_navigation_collision_visual=True,
    )
    with pytest.raises(ValueError, match="collision"):
        runtime._validate_navigation_ros2_config()


def test_cmd_vel_subscription_and_policy_gate_must_be_paired() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    common = {
        "depth_point_cloud_config": DepthPointCloudConfig(),
        "hide_navigation_collision_visual": False,
    }
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_controller_status_subscription=True,
            enable_pct_goal_publisher=True,
        ),
        **common,
    )
    with pytest.raises(ValueError, match="同时启用或关闭"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="同时启用或关闭"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="goal_reached"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_pct_goal_publisher=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="controller_status"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_controller_status_subscription=True,
            enable_grid_map_diagnostics_subscription=True,
            enable_bspline_diagnostics_subscription=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="pct/goal"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_controller_status_subscription=True,
            enable_pct_goal_publisher=True,
            enable_stair_execution_frozen_publisher=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="typed diagnostics"):
        runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_controller_status_subscription=True,
            enable_grid_map_diagnostics_subscription=True,
            enable_bspline_diagnostics_subscription=True,
            enable_pct_goal_publisher=True,
            enable_stair_execution_frozen_publisher=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    runtime._validate_navigation_ros2_config()

    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_goal_reached_subscription=True,
            enable_controller_status_subscription=True,
            enable_grid_map_diagnostics_subscription=True,
            enable_bspline_diagnostics_subscription=True,
            enable_pct_goal_publisher=True,
        ),
        cmd_vel_to_policy_config=CmdVelToPolicyConfig(),
        **common,
    )
    with pytest.raises(ValueError, match="楼梯执行冻结"):
        runtime._validate_navigation_ros2_config()


def test_navigation_cli_enables_typed_controller_status_by_default() -> None:
    from scripts.pipeline.run_full_physics_pipeline import (
        _navigation_ros2_runtime_kwargs,
        _parse_args,
    )

    runtime_kwargs = _navigation_ros2_runtime_kwargs(_parse_args([]))
    bridge_config = runtime_kwargs["ros2_ogn_bridge_config"]

    assert isinstance(bridge_config, IsaacRos2OgnBridgeConfig)
    assert bridge_config.enable_controller_status_subscription is True
    assert bridge_config.controller_status_topic == "/planning/controller_status"
    assert bridge_config.enable_grid_map_diagnostics_subscription is True
    assert bridge_config.grid_map_diagnostics_topic == (
        "/planning/grid_map_observation_diagnostics"
    )
    assert bridge_config.enable_bspline_diagnostics_subscription is True
    assert bridge_config.bspline_diagnostics_topic == (
        "/planning/bspline_diagnostics"
    )


def test_enabled_navigation_bridge_requires_reference_path_subscription() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(
            enable_reference_path_subscription=False,
        ),
        depth_point_cloud_config=DepthPointCloudConfig(),
        hide_navigation_collision_visual=False,
    )

    with pytest.raises(ValueError, match="参考 Path"):
        runtime._validate_navigation_ros2_config()


class _FakeCommandSink:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float, float]] = []

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        self.commands.append((float(vx), float(vy), float(wz)))


class _FakeCommandBridge:
    def __init__(
        self,
        samples: list[OgnTwistSample | None],
        goal_samples: list[OgnBoolSample | None] | None = None,
        *,
        enable_stair_execution_frozen_publisher: bool = False,
    ) -> None:
        self.config = SimpleNamespace(
            enable_goal_reached_subscription=True,
            enable_stair_execution_frozen_publisher=(
                enable_stair_execution_frozen_publisher
            ),
            odom_frame_id="world",
        )
        self.active_reference_path_stamp_ns = 20
        self.samples = samples
        self.goal_samples = list(goal_samples or [])
        self.poll_timestamps: list[float] = []
        self.goal_poll_timestamps: list[float] = []
        self.stair_frozen_calls: list[tuple[bool, float]] = []

    def poll_twist(self, *, receipt_timestamp: float) -> OgnTwistSample | None:
        self.poll_timestamps.append(receipt_timestamp)
        return self.samples.pop(0) if self.samples else None

    def poll_goal_reached(
        self,
        *,
        receipt_timestamp: float,
    ) -> OgnBoolSample | None:
        self.goal_poll_timestamps.append(receipt_timestamp)
        return self.goal_samples.pop(0) if self.goal_samples else None

    def publish_stair_execution_frozen(
        self,
        value: bool,
        *,
        timestamp: float,
    ) -> OgnStairExecutionFreezePublicationReport:
        self.stair_frozen_calls.append((value, timestamp))
        return OgnStairExecutionFreezePublicationReport(
            frozen=value,
            source_topic="/planning/stair_execution_frozen",
            publish_timestamp=timestamp,
            header_stamp_sec=int(timestamp),
            header_stamp_nanosec=int((timestamp % 1.0) * 1_000_000_000),
            reference_path_stamp_sec=0,
            reference_path_stamp_nanosec=20,
            writer_id="isaac",
            writer_epoch="epoch-a",
            sequence=len(self.stair_frozen_calls),
        )


def test_runtime_records_goal_event_with_receipt_sequence() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    bridge = _FakeCommandBridge(
        [],
        [
            OgnBoolSample(
                value=False,
                receipt_timestamp=0.1,
                sequence=7,
            )
        ],
    )
    runtime._runtime = SimpleNamespace(physics_dt=0.02)  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 5  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]

    runtime._poll_scan_goal_reached()

    assert bridge.goal_poll_timestamps == [0.1]
    assert runtime._metadata["scan_goal_reached_last_sample"] == {  # type: ignore[attr-defined]
        "value": False,
        "receipt_timestamp": 0.1,
        "sequence": 7,
    }


def test_compact_episode_metadata_keeps_ros2_navigation_evidence() -> None:
    """常规帧也必须保存 SCAN 命令、完成事件与传感器发布证据。"""

    metadata = {
        "scan_cmd_vel_last_write_report": {"write_sequence": 17},
        "navigation_policy_gate_lifecycle_report": {
            "motion_allowed_write_count": 4,
        },
        "scan_goal_reached_last_sample": {
            "value": True,
            "receipt_timestamp": 1.2,
            "sequence": 9,
        },
        "navigation_ros2_last_publish_report": {
            "odometry_published": True,
            "point_cloud_published": True,
        },
        "navigation_stair_execution_frozen_last_publish_report": {
            "sequence": 18,
            "publish_timestamp": 1.18,
            "frozen": False,
        },
        "static_stage_report": {"large": "不应逐帧重复"},
    }

    compact = _compact_simulation_metadata(metadata)

    assert compact == {
        "scan_cmd_vel_last_write_report": {"write_sequence": 17},
        "navigation_policy_gate_lifecycle_report": {
            "motion_allowed_write_count": 4,
        },
        "scan_goal_reached_last_sample": {
            "value": True,
            "receipt_timestamp": 1.2,
            "sequence": 9,
        },
        "navigation_ros2_last_publish_report": {
            "odometry_published": True,
            "point_cloud_published": True,
        },
        "navigation_stair_execution_frozen_last_publish_report": {
            "sequence": 18,
            "publish_timestamp": 1.18,
            "frozen": False,
        },
    }


def test_compact_episode_metadata_drops_repeated_lifecycle_histories() -> None:
    """逐拍诊断保留首末证据和计数，但不重复终态完整历史。"""

    metadata = {
        "navigation_policy_gate_lifecycle_report": {
            "policy_write_count": 10,
            "first_identity_verified_tracking_write": {"write_sequence": 1},
            "last_identity_verified_tracking_write": {"write_sequence": 10},
            "identity_verified_tracking_write_reports": [{"large": True}],
        },
        "grid_map_observation_lifecycle_report": {
            "sample_count": 5,
            "first_report": {"observation_sequence": 1},
            "last_report": {"observation_sequence": 5},
            "diagnostic_reports": [{"large": True}],
        },
        "bspline_diagnostics_lifecycle_report": {
            "sample_count": 3,
            "last_report": {"diagnostic_sequence": 3},
            "diagnostic_reports": [{"large": True}],
            "trajectory_identities": [{"traj_id": 3}],
        },
        "active_sensing_lifecycle_report": {
            "attempt_count": 1,
            "completed_attempt_count": 1,
            "attempts": [{"large": True}],
        },
        "scan_controller_status_lifecycle_report": {
            "sample_count": 4,
            "last_status": {"status_sequence": 4},
            "accepted_status_reports": [{"large": True}],
            "tracking_status_reports": [{"large": True}],
            "accepted_trajectory_identities": [{"traj_id": 3}],
        },
    }

    compact = _compact_simulation_metadata(metadata)

    assert compact["navigation_policy_gate_lifecycle_report"] == {
        "policy_write_count": 10,
        "first_identity_verified_tracking_write": {"write_sequence": 1},
        "last_identity_verified_tracking_write": {"write_sequence": 10},
    }
    assert compact["grid_map_observation_lifecycle_report"] == {
        "sample_count": 5,
        "first_report": {"observation_sequence": 1},
        "last_report": {"observation_sequence": 5},
    }
    assert compact["bspline_diagnostics_lifecycle_report"] == {
        "sample_count": 3,
        "last_report": {"diagnostic_sequence": 3},
    }
    assert compact["active_sensing_lifecycle_report"] == {
        "attempt_count": 1,
        "completed_attempt_count": 1,
    }
    assert compact["scan_controller_status_lifecycle_report"] == {
        "sample_count": 4,
        "last_status": {"status_sequence": 4},
    }


def test_runtime_scan_command_gate_is_the_only_policy_velocity_writer() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=10.0,
            max_vy_rate=10.0,
            max_wz_rate=10.0,
            require_odometry=False,
            require_point_cloud=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    bridge = _FakeCommandBridge(
        [
            OgnTwistSample(
                linear_velocity=(0.2, -0.1, 9.0),
                angular_velocity=(8.0, 7.0, 0.3),
                receipt_timestamp=0.1,
                sequence=1,
                navigation_permit=_runtime_navigation_permit(0.1),
            )
        ]
    )
    runtime._runtime = SimpleNamespace(physics_dt=0.02)  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 5  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]

    report = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
    )

    assert bridge.poll_timestamps == [0.1]
    assert report.motion_allowed is True
    assert report.written_command.as_tuple() == pytest.approx(
        (0.2, -0.1, 0.3)
    )
    assert sink.commands[-1] == pytest.approx((0.2, -0.1, 0.3))

    bridge.samples.append(
        OgnTwistSample(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            receipt_timestamp=0.12,
            sequence=1,
            command_present=False,
            navigation_permit=_runtime_navigation_permit(
                0.12,
                status_sequence=2,
                allow=False,
            ),
        )
    )
    runtime._ros2_physics_step_count = 6  # type: ignore[attr-defined]
    forced_zero = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
    )
    assert "navigation_status_force_zero" in forced_zero.stop_reasons
    assert forced_zero.written_command.as_tuple() == (0.0, 0.0, 0.0)
    assert sink.commands[-1] == (0.0, 0.0, 0.0)

    stopped = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=True,
    )
    assert stopped.stop_reasons == ("environment_terminated",)
    assert sink.commands[-1] == (0.0, 0.0, 0.0)

    bridge.samples.append(
        OgnTwistSample(
            linear_velocity=(0.3, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            receipt_timestamp=0.1,
            sequence=2,
            navigation_permit=_runtime_navigation_permit(
                0.1,
                status_sequence=2,
            ),
        )
    )
    emergency_stopped = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
        emergency_stop_reason="locomotion_stall",
    )
    assert emergency_stopped.motion_allowed is False
    assert emergency_stopped.stop_reasons == ("locomotion_stall",)
    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    # 锁存停车分支不得轮询或接纳仍在发布的 ROS 非零 Twist。
    assert len(bridge.samples) == 1
    gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_runtime_stair_freeze_drains_twist_and_requires_fresh_resume() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=10.0,
            max_vy_rate=10.0,
            max_wz_rate=10.0,
            require_odometry=False,
            require_point_cloud=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    bridge = _FakeCommandBridge(
        [
            OgnTwistSample(
                linear_velocity=(0.3, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                receipt_timestamp=0.1,
                sequence=1,
                navigation_permit=_runtime_navigation_permit(0.1),
            )
        ]
    )
    runtime._runtime = SimpleNamespace(physics_dt=0.02)  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 5  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]

    inhibited = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
        temporary_inhibit_reason="scan_stair_freeze",
    )

    assert bridge.poll_timestamps == [0.1]
    assert bridge.samples == []
    assert inhibited.motion_allowed is False
    assert inhibited.stop_reasons == ("scan_stair_freeze",)
    assert inhibited.written_command.as_tuple() == (0.0, 0.0, 0.0)
    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    assert runtime._last_scan_cmd_vel_source_sequence == 1  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_source_receipt_timestamp == pytest.approx(  # type: ignore[attr-defined]
        0.1
    )
    assert runtime._scan_cmd_vel_sample_received_this_tick is False  # type: ignore[attr-defined]
    assert runtime._scan_cmd_vel_sample_drained_this_tick is True  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_drain_sequence == 1  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_drain_receipt_timestamp == pytest.approx(  # type: ignore[attr-defined]
        0.1
    )

    runtime._ros2_physics_step_count = 6  # type: ignore[attr-defined]
    no_fresh_command = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
    )
    assert no_fresh_command.motion_allowed is False
    assert "missing_cmd_vel" in no_fresh_command.stop_reasons
    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    assert runtime._last_scan_cmd_vel_source_sequence == 1  # type: ignore[attr-defined]
    assert runtime._scan_cmd_vel_sample_received_this_tick is False  # type: ignore[attr-defined]
    assert runtime._scan_cmd_vel_sample_drained_this_tick is False  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_drain_sequence == 1  # type: ignore[attr-defined]

    bridge.samples.append(
        OgnTwistSample(
            linear_velocity=(0.2, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            receipt_timestamp=0.14,
            sequence=2,
            navigation_permit=_runtime_navigation_permit(0.14),
        )
    )
    runtime._ros2_physics_step_count = 7  # type: ignore[attr-defined]
    resumed = runtime._apply_scan_cmd_vel_to_policy(
        environment_terminated=False,
    )
    assert resumed.motion_allowed is True
    assert resumed.written_command.as_tuple() == pytest.approx((0.2, 0.0, 0.0))
    assert runtime._last_scan_cmd_vel_source_sequence == 2  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_source_receipt_timestamp == pytest.approx(  # type: ignore[attr-defined]
        0.14
    )
    assert runtime._scan_cmd_vel_sample_received_this_tick is True  # type: ignore[attr-defined]
    assert runtime._scan_cmd_vel_sample_drained_this_tick is False  # type: ignore[attr-defined]
    assert runtime._last_scan_cmd_vel_drain_sequence == 1  # type: ignore[attr-defined]
    gate.release(owner_id="scan_cmd_vel", now=0.14)


class _FakePolicyAction:
    def to(self, _device: object) -> "_FakePolicyAction":
        return self


def _runtime_with_nonzero_scan_command() -> tuple[
    IsaacLabNavigationRuntime,
    _FakeCommandSink,
    CmdVelToPolicyAdapter,
    _FakeCommandBridge,
    list[_FakePolicyAction],
]:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=10.0,
            max_vy_rate=10.0,
            max_wz_rate=10.0,
            require_odometry=False,
            require_point_cloud=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    gate.receive(
        PolicyCommandInput(
            command=(0.3, 0.0, 0.0),
            navigation_permit=_runtime_navigation_permit(0.08),
        ),
        0.08,
        "scan_cmd_vel",
    )
    gate.tick(0.08, "scan_cmd_vel")
    bridge = _FakeCommandBridge(
        [
            OgnTwistSample(
                linear_velocity=(0.25, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                receipt_timestamp=0.1,
                sequence=2,
                navigation_permit=_runtime_navigation_permit(
                    0.1,
                    status_sequence=2,
                ),
            )
        ],
        enable_stair_execution_frozen_publisher=True,
    )
    processed_actions: list[_FakePolicyAction] = []
    runtime._runtime = SimpleNamespace(  # type: ignore[attr-defined]
        physics_dt=0.02,
        device="cpu",
        action_manager=SimpleNamespace(
            process_action=processed_actions.append,
        ),
    )
    runtime._ros2_physics_step_count = 5  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._environment_terminated = False  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]
    runtime._adapter = SimpleNamespace(  # type: ignore[attr-defined]
        compute_policy_action=lambda **_kwargs: _FakePolicyAction(),
    )
    runtime._require_ready = lambda: None  # type: ignore[method-assign]
    runtime._apply_object_initialization_pose_stabilization = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._configure_manipulation_base_lock = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._stage_arm_target = lambda _action: {}  # type: ignore[method-assign]
    runtime._stage_gripper_target = lambda _action: {}  # type: ignore[method-assign]
    runtime._poll_scan_goal_reached = lambda: None  # type: ignore[method-assign]
    runtime._update_velocity_command_visualization = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._record_joint_action_apply = (  # type: ignore[method-assign]
        lambda _action, _arm, _gripper: None
    )
    assert sink.commands[-1] == (0.3, 0.0, 0.0)
    return runtime, sink, gate, bridge, processed_actions


def _controller_policy_write_snapshot(
    *,
    status_sequence: int,
    state: int = 10,
    traj_id: int = 42,
) -> dict[str, object]:
    """构造 policy 写入时已接收的不可变 typed controller 快照。"""

    return {
        "source": "ros2_scan_planner_msgs_controller_status",
        "topic": "/planning/controller_status",
        "status_sequence": status_sequence,
        "state": state,
        "identity": {
            "reference_path_stamp": {"sec": 18, "nanosec": 1},
            "reference_path_stamp_ns": 18_000_000_001,
            "bspline_header_stamp": {"sec": 19, "nanosec": 2},
            "bspline_header_stamp_ns": 19_000_000_002,
            "start_time": {"sec": 19, "nanosec": 3},
            "start_time_ns": 19_000_000_003,
            "traj_id": traj_id,
        },
    }


def _record_identity_verified_policy_write(
    runtime: IsaacLabNavigationRuntime,
    gate: CmdVelToPolicyAdapter,
    snapshot: object,
) -> None:
    runtime._metadata[  # type: ignore[attr-defined]
        "scan_controller_status_last_report"
    ] = deepcopy(snapshot)
    report = gate.tick(0.1, "scan_cmd_vel")
    assert report.motion_allowed is True
    runtime._record_scan_policy_write_report(  # type: ignore[attr-defined]
        action=RobotAction.idle(source="scan_tracking"),
        command_report=report,
        temporary_inhibit_reason=None,
    )


def _scan_stair_freeze_action() -> RobotAction:
    return RobotAction(
        base_velocity=(0.2, 0.0, 0.0),
        source="scan_stair_freeze_active",
        metadata={
            "navigation_base_pose_lock": True,
            "navigation_base_pose_lock_xyzyaw": (1.0, 2.0, 0.3, 0.0),
            "navigation_cmd_vel_inhibit": True,
            "navigation_cmd_vel_inhibit_reason": "scan_stair_freeze",
        },
    )


def test_runtime_records_identity_verified_tracking_gate_lifecycle() -> None:
    runtime, sink, gate, bridge, processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    try:
        runtime.apply(RobotAction.idle(source="scan_tracking"))

        assert sink.commands[-1] == pytest.approx((0.25, 0.0, 0.0))
        assert bridge.poll_timestamps == [0.1]
        assert len(processed_actions) == 1
        write = runtime._metadata[  # type: ignore[attr-defined]
            "scan_cmd_vel_last_write_report"
        ]
        diagnostics = write["policy_navigation_gate_consumed_report"]
        assert write["motion_allowed"] is True
        assert diagnostics["permit"]["state"] == 3
        assert diagnostics["permit"]["status_sequence"] == 2
        assert diagnostics["permit"]["identity_valid"] is True
        assert diagnostics["command_identity"] == [10, 20, 1]
        assert diagnostics["command_identity_matches_permit"] is True

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        assert lifecycle["policy_write_count"] == 1
        assert lifecycle["motion_allowed_write_count"] == 1
        assert lifecycle["identity_verified_tracking_write_count"] == 1
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 0
        assert lifecycle["forced_zero_write_count"] == 0
        assert lifecycle[
            "first_identity_verified_tracking_write"
        ] == lifecycle["last_identity_verified_tracking_write"]
        assert lifecycle["last_stop_reasons"] == []
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_tracking_write_ring_pins_first_write_per_controller_snapshot() -> None:
    runtime, _sink, gate, _bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    snapshot = _controller_policy_write_snapshot(status_sequence=7)
    try:
        for _ in range(20):
            _record_identity_verified_policy_write(runtime, gate, snapshot)

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        reports = lifecycle["identity_verified_tracking_write_reports"]
        assert lifecycle["identity_verified_tracking_write_count"] == 20
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 1
        assert len(reports) == 1
        assert reports[0]["write_sequence"] == 1
        assert reports[0]["scan_controller_status_snapshot"] == snapshot
        assert lifecycle["last_identity_verified_tracking_write"][
            "write_sequence"
        ] == 20
        assert (
            lifecycle[
                "dropped_identity_verified_tracking_write_report_count"
            ]
            == 0
        )
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_same_trajectory_later_tracking_state_keeps_distinct_snapshot() -> None:
    runtime, _sink, gate, _bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    waiting = _controller_policy_write_snapshot(
        status_sequence=7,
        state=0,
    )
    tracking = _controller_policy_write_snapshot(
        status_sequence=8,
        state=10,
    )
    try:
        _record_identity_verified_policy_write(runtime, gate, waiting)
        _record_identity_verified_policy_write(runtime, gate, tracking)

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        reports = lifecycle["identity_verified_tracking_write_reports"]
        assert lifecycle["identity_verified_tracking_write_count"] == 2
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 2
        assert [
            report["scan_controller_status_snapshot"]["state"]
            for report in reports
        ] == [0, 10]
        assert [
            report["scan_controller_status_snapshot"]["status_sequence"]
            for report in reports
        ] == [7, 8]
        assert reports[0]["scan_controller_status_snapshot"]["identity"] == (
            reports[1]["scan_controller_status_snapshot"]["identity"]
        )
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_detour_tracking_evidence_survives_many_control_ticks() -> None:
    runtime, _sink, gate, _bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    detour_snapshot = _controller_policy_write_snapshot(
        status_sequence=31,
        state=10,
        traj_id=77,
    )
    try:
        for _ in range(1000):
            _record_identity_verified_policy_write(
                runtime,
                gate,
                detour_snapshot,
            )

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        reports = lifecycle["identity_verified_tracking_write_reports"]
        assert lifecycle["identity_verified_tracking_write_count"] == 1000
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 1
        assert len(reports) == 1
        assert reports[0]["write_sequence"] == 1
        assert reports[0]["scan_controller_status_snapshot"] == (
            detour_snapshot
        )
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_tracking_write_ring_keeps_128_distinct_snapshots_with_fifo_eviction(
) -> None:
    runtime, _sink, gate, _bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    try:
        for status_sequence in range(1, 131):
            _record_identity_verified_policy_write(
                runtime,
                gate,
                _controller_policy_write_snapshot(
                    status_sequence=status_sequence,
                ),
            )

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        reports = lifecycle["identity_verified_tracking_write_reports"]
        retained_sequences = [
            report["scan_controller_status_snapshot"]["status_sequence"]
            for report in reports
        ]
        assert lifecycle["identity_verified_tracking_write_count"] == 130
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 130
        assert len(reports) == 128
        assert retained_sequences == list(range(3, 131))
        assert (
            lifecycle[
                "dropped_identity_verified_tracking_write_report_count"
            ]
            == 2
        )
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_tracking_write_ring_rejects_untyped_snapshot_fail_closed() -> None:
    runtime, _sink, gate, _bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    try:
        _record_identity_verified_policy_write(
            runtime,
            gate,
            {"status_sequence": 1, "state": 10},
        )

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        assert lifecycle["identity_verified_tracking_write_count"] == 1
        assert lifecycle["identity_verified_tracking_snapshot_count"] == 0
        assert lifecycle["identity_verified_tracking_write_reports"] == []
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_runtime_retains_global_replan_and_tracking_recovery_lifecycle() -> None:
    """终态回到 TRACKING 后仍须保留曾经重规划及新 PCT 代际的证据。"""

    runtime, _sink, gate, bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    observed = {
        "schema": "navigation_status_observed_diagnostics_v1",
        "status": {
            "status_sequence": 10,
            "state": 4,
            "global_replan_requested": True,
            "global_replan_in_flight": True,
            "global_replan_request_id": 7,
            "pct_plan_id": 3,
            "consecutive_scan_failures": 5,
            "reason": "scan_failure_threshold_reached",
            "identity_valid": False,
        },
    }
    bridge.navigation_status_observed_diagnostics = lambda: deepcopy(observed)
    try:
        runtime._record_scan_policy_write_report(  # type: ignore[attr-defined]
            action=RobotAction.idle(source="global_replan"),
            command_report=gate.tick(0.10, "scan_cmd_vel"),
            temporary_inhibit_reason=None,
        )
        observed["status"] = {
            "status_sequence": 11,
            "state": 3,
            "global_replan_requested": False,
            "global_replan_in_flight": False,
            "global_replan_request_id": 7,
            "pct_plan_id": 4,
            "consecutive_scan_failures": 0,
            "reason": "local_trajectory_tracking",
            "identity_valid": True,
        }
        runtime._record_scan_policy_write_report(  # type: ignore[attr-defined]
            action=RobotAction.idle(source="tracking_recovered"),
            command_report=gate.tick(0.12, "scan_cmd_vel"),
            temporary_inhibit_reason=None,
        )

        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        assert lifecycle["observed_status_sequence_count"] == 2
        assert lifecycle["observed_state_transition_count"] == 1
        assert lifecycle["observed_state_counts"] == {
            "global_replan": 1,
            "tracking": 1,
        }
        assert lifecycle["maximum_consecutive_scan_failures"] == 5
        assert lifecycle["global_replan_requested_status_count"] == 1
        assert lifecycle["global_replan_in_flight_status_count"] == 1
        assert lifecycle["distinct_global_replan_request_ids"] == [7]
        assert lifecycle["distinct_pct_plan_ids"] == [3, 4]
        assert lifecycle["global_replan_pending_recovery"] is False
        assert lifecycle["tracking_after_global_replan_observed"] is True
        assert lifecycle["global_replan_recovery_count"] == 1
        assert lifecycle["first_global_replan_status"] is not None
        assert lifecycle["last_global_replan_status"] is not None
        assert lifecycle["last_observed_state"] == 3
        assert lifecycle["last_observed_status_sequence"] == 11
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.12)


@pytest.mark.parametrize(
    ("metadata", "expected", "expected_phase"),
    [
        ({}, False, None),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "active",
            },
            True,
            "active",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "full_lock_settle",
            },
            True,
            "full_lock_settle",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "root_release_settle",
            },
            True,
            "root_release_settle",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "released",
            },
            True,
            "released",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": (
                    "post_release_stabilizing"
                ),
            },
            True,
            "post_release_stabilizing",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "terminal_hold",
            },
            True,
            "terminal_hold",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "released_stable",
            },
            False,
            "released_stable",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
                "navigation_scan_stair_freeze_phase": "resume",
            },
            False,
            "resume",
        ),
        (
            {
                "navigation_scan_stair_freeze": True,
            },
            True,
            None,
        ),
        (
            {
                "navigation_stair_emergency_hold": True,
                "navigation_scan_stair_freeze_phase": "released_stable",
            },
            True,
            "released_stable",
        ),
    ],
)
def test_stair_frozen_derivation_covers_every_freeze_and_resume_phase(
    metadata: dict[str, object],
    expected: bool,
    expected_phase: str | None,
) -> None:
    value, phase, _reason = _stair_execution_frozen_from_action(
        RobotAction(source="test_action", metadata=metadata)
    )

    assert value is expected
    assert phase == expected_phase


def test_runtime_stair_frozen_publish_records_fixed_report_for_ordinary_action(
) -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    bridge = _FakeCommandBridge(
        [],
        enable_stair_execution_frozen_publisher=True,
    )
    runtime._runtime = SimpleNamespace(physics_dt=0.02)  # type: ignore[attr-defined]
    runtime._ros2_physics_step_count = 1  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]

    report = runtime._publish_stair_execution_frozen_for_action(
        RobotAction.idle(source="episode_reset")
    )

    assert report is not None
    assert bridge.stair_frozen_calls == [(False, 0.02)]
    assert runtime._metadata[  # type: ignore[attr-defined]
        "navigation_stair_execution_frozen_last_publish_report"
    ] == {
        "schema": "isaac_stair_execution_frozen_v1",
        "message_type": "scan_planner_msgs/msg/StairExecutionFreeze",
        "source": "isaac_action_metadata",
        "topic": "/planning/stair_execution_frozen",
        "publish_timestamp": 0.02,
        "header": {
            "frame_id": "world",
            "stamp": {"sec": 0, "nanosec": 20_000_000},
        },
        "reference_path_stamp": {"sec": 0, "nanosec": 20},
        "reference_path_stamp_ns": 20,
        "writer_id": "isaac",
        "writer_epoch": "epoch-a",
        "sequence": 1,
        "value": False,
        "frozen": False,
        "action_source": "episode_reset",
        "action_phase": None,
        "decision_reason": "ordinary_action",
    }


def test_runtime_apply_refreshes_inactive_stair_heartbeat_each_control_tick(
) -> None:
    runtime, _sink, gate, bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    heartbeat_reports: list[dict[str, object]] = []
    post_step_ages: list[float] = []
    try:
        # 不等待 wall time，连续推进 0.32 秒仿真时间，复现 v6 的快速
        # 控制循环。inactive false 必须每个 control tick 获得新
        # sequence/header；即使 SCAN 在紧随其后的 post-step /clock 上
        # 检查，新鲜度也只能老化一拍。
        for _tick in range(16):
            runtime.apply(RobotAction.idle(source="exec_nav_to_pick"))
            report = runtime._metadata[  # type: ignore[attr-defined]
                "navigation_stair_execution_frozen_last_publish_report"
            ]
            assert report["value"] is False
            heartbeat_reports.append(deepcopy(report))

            publish_timestamp = float(report["publish_timestamp"])
            runtime._ros2_physics_step_count += 1  # type: ignore[attr-defined]
            post_step_timestamp = runtime._navigation_ros2_timestamp()
            post_step_age = post_step_timestamp - publish_timestamp
            post_step_ages.append(post_step_age)
            assert post_step_age == pytest.approx(0.02)
            assert post_step_age < 0.25
            runtime._action_prepared = False  # type: ignore[attr-defined]
    finally:
        gate.release(
            owner_id="scan_cmd_vel",
            now=runtime._navigation_ros2_timestamp(),
        )

    assert [value for value, _timestamp in bridge.stair_frozen_calls] == [
        False
    ] * 16
    assert [timestamp for _value, timestamp in bridge.stair_frozen_calls] == (
        pytest.approx([0.10 + 0.02 * index for index in range(16)])
    )
    assert [report["sequence"] for report in heartbeat_reports] == list(
        range(1, 17)
    )
    assert max(post_step_ages) < 0.25


def test_runtime_stair_frozen_skips_publish_without_current_path_identity(
) -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    bridge = _FakeCommandBridge(
        [],
        enable_stair_execution_frozen_publisher=True,
    )
    bridge.active_reference_path_stamp_ns = 0
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]

    assert runtime._publish_stair_execution_frozen_for_action(
        RobotAction.idle(source="episode_reset")
    ) is None
    assert bridge.stair_frozen_calls == []


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    (
        ("object_initialization", ["object_initialization"]),
        ("joint_lock", ["object_initialization", "joint_lock"]),
        (
            "arm_staging",
            ["object_initialization", "joint_lock", "arm_staging"],
        ),
    ),
)
def test_runtime_stair_freeze_zeroes_before_fallible_action_preparation(
    failure_stage: str,
    expected_calls: list[str],
) -> None:
    runtime, sink, gate, bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    preparation_calls: list[str] = []

    def prepare(stage: str) -> dict[str, object] | None:
        # 每一个可能失败的准备步骤开始前，旧 locomotion 命令都必须已清零。
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        preparation_calls.append(stage)
        if stage == failure_stage:
            raise RuntimeError(f"injected_{stage}_failure")
        return {}

    runtime._apply_object_initialization_pose_stabilization = (  # type: ignore[method-assign]
        lambda _action: prepare("object_initialization")
    )
    runtime._configure_manipulation_base_lock = (  # type: ignore[method-assign]
        lambda _action: prepare("joint_lock")
    )
    runtime._stage_arm_target = (  # type: ignore[method-assign]
        lambda _action: prepare("arm_staging")
    )
    writes_before = len(sink.commands)
    try:
        with pytest.raises(RuntimeError, match=f"injected_{failure_stage}_failure"):
            runtime.apply(_scan_stair_freeze_action())

        assert preparation_calls == expected_calls
        assert len(sink.commands) == writes_before + 1
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        assert bridge.poll_timestamps == [0.1]
        assert bridge.samples == []
        report = runtime._metadata[  # type: ignore[attr-defined]
            "scan_cmd_vel_last_write_report"
        ]
        assert report["write_sequence"] == 1
        assert report["owner_id"] == "scan_cmd_vel"
        assert report["written_command"] == [0.0, 0.0, 0.0]
        assert report["stop_reasons"] == ["scan_stair_freeze"]
        assert bridge.stair_frozen_calls == [(True, 0.1)]
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_runtime_stair_freeze_success_writes_policy_only_once() -> None:
    runtime, sink, gate, bridge, processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    bridge.config.enable_stair_execution_frozen_publisher = True
    writes_before = len(sink.commands)
    try:
        runtime.apply(_scan_stair_freeze_action())

        assert len(sink.commands) == writes_before + 1
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        assert bridge.poll_timestamps == [0.1]
        assert bridge.samples == []
        assert runtime._scan_policy_write_sequence == 1  # type: ignore[attr-defined]
        assert len(processed_actions) == 1
        report = runtime._metadata[  # type: ignore[attr-defined]
            "scan_cmd_vel_last_write_report"
        ]
        assert report["cmd_vel_source_sequence"] == 2
        assert report["cmd_vel_source_receipt_timestamp"] == pytest.approx(0.1)
        assert report["cmd_vel_sample_received_this_tick"] is False
        assert report["cmd_vel_sample_drained_this_tick"] is True
        assert report["last_cmd_vel_drain_sequence"] == 2
        assert report["last_cmd_vel_drain_receipt_timestamp"] == pytest.approx(
            0.1
        )
        assert report["policy_navigation_gate_consumed_report"][
            "permit_received"
        ] is True
        assert report["navigation_status_observed_report"] is None
        lifecycle = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_policy_gate_lifecycle_report"
        ]
        assert lifecycle["policy_write_count"] == 1
        assert lifecycle["motion_allowed_write_count"] == 0
        assert lifecycle["identity_verified_tracking_write_count"] == 0
        assert lifecycle["forced_zero_write_count"] == 1
        assert bridge.stair_frozen_calls == [(True, 0.1)]
        frozen_report = runtime._metadata[  # type: ignore[attr-defined]
            "navigation_stair_execution_frozen_last_publish_report"
        ]
        assert frozen_report == {
            "schema": "isaac_stair_execution_frozen_v1",
            "message_type": "scan_planner_msgs/msg/StairExecutionFreeze",
            "source": "isaac_action_metadata",
            "topic": "/planning/stair_execution_frozen",
            "publish_timestamp": 0.1,
            "header": {
                "frame_id": "world",
                "stamp": {"sec": 0, "nanosec": 100_000_000},
            },
            "reference_path_stamp": {"sec": 0, "nanosec": 20},
            "reference_path_stamp_ns": 20,
            "writer_id": "isaac",
            "writer_epoch": "epoch-a",
            "sequence": 1,
            "value": True,
            "frozen": True,
            "action_source": "scan_stair_freeze_active",
            "action_phase": None,
            "decision_reason": "stair_metadata_fail_closed",
        }
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_navigation_support_lock_enable_failure_is_fail_closed() -> None:
    """导航支撑锁未启用时必须在当拍抛错，不能继续 root-only 运行。"""

    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._metadata = {}  # type: ignore[attr-defined]
    runtime._manipulation_base_lock_active = False  # type: ignore[attr-defined]
    runtime._manipulation_support_joint_lock_active = False  # type: ignore[attr-defined]
    runtime._navigation_joint_pose_lock_active = False  # type: ignore[attr-defined]
    runtime._adapter = SimpleNamespace(  # type: ignore[attr-defined]
        set_support_joint_lock=lambda *_args, **_kwargs: {
            "enabled": False,
            "reason": "injected_support_lock_failure",
        }
    )
    action = RobotAction(
        source="scan_stair_freeze_active",
        metadata={
            "navigation_support_joint_lock": True,
            "navigation_support_joint_lock_phase": "active",
            "navigation_dog_joint_names": ("joint",),
            "navigation_dog_joint_positions": (0.0,),
        },
    )

    with pytest.raises(RuntimeError, match="failed to enable support joint lock"):
        runtime._configure_manipulation_base_lock(action)  # type: ignore[attr-defined]

    report = runtime._metadata[  # type: ignore[attr-defined]
        "last_navigation_support_joint_lock_report"
    ]
    assert report["enabled"] is False
    assert report["reason"] == "injected_support_lock_failure"


def test_navigation_full_body_lock_enable_failure_is_fail_closed() -> None:
    """导航全身锁未启用时必须立即失败关闭。"""

    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._metadata = {}  # type: ignore[attr-defined]
    runtime._navigation_joint_pose_lock_active = False  # type: ignore[attr-defined]
    runtime._adapter = SimpleNamespace(  # type: ignore[attr-defined]
        set_navigation_joint_pose_lock=lambda *_args, **_kwargs: {
            "enabled": False,
            "reason": "injected_full_body_lock_failure",
        }
    )
    action = RobotAction(
        source="scan_stair_freeze_active",
        metadata={
            "navigation_full_body_joint_lock": True,
            "navigation_full_body_joint_lock_phase": "active",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="failed to enable navigation joint pose lock",
    ):
        runtime._configure_navigation_joint_pose_lock(  # type: ignore[attr-defined]
            action
        )

    report = runtime._metadata[  # type: ignore[attr-defined]
        "last_navigation_joint_pose_lock_report"
    ]
    assert report["enabled"] is False
    assert report["reason"] == "injected_full_body_lock_failure"


def test_runtime_emergency_stop_precedes_inhibit_and_fallible_preparation() -> None:
    runtime, sink, gate, bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )

    def fail_object_initialization(_action: RobotAction) -> None:
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        raise RuntimeError("injected_object_initialization_failure")

    runtime._apply_object_initialization_pose_stabilization = (  # type: ignore[method-assign]
        fail_object_initialization
    )
    writes_before = len(sink.commands)
    action = _scan_stair_freeze_action()
    action.metadata.update(
        {
            "navigation_emergency_stop": True,
            "navigation_emergency_stop_reason": "predicted_collision",
        }
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="injected_object_initialization_failure",
        ):
            runtime.apply(action)

        assert len(sink.commands) == writes_before + 1
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        # emergency stop 优先于临时 inhibit，不得先消费桥上的新 Twist。
        assert bridge.poll_timestamps == []
        assert len(bridge.samples) == 1
        report = runtime._metadata[  # type: ignore[attr-defined]
            "scan_cmd_vel_last_write_report"
        ]
        assert report["write_sequence"] == 1
        assert report["stop_reasons"] == ["predicted_collision"]
        assert report["navigation_emergency_stop_latched"] is True
        assert report["navigation_emergency_stop_reason"] == "predicted_collision"
        assert bridge.stair_frozen_calls == [(True, 0.1)]
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_runtime_stair_inhibit_zeroes_before_failing_twist_drain() -> None:
    runtime, sink, gate, bridge, _processed_actions = (
        _runtime_with_nonzero_scan_command()
    )

    def fail_drain(*, receipt_timestamp: float) -> None:
        assert receipt_timestamp == pytest.approx(0.1)
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        raise RuntimeError("injected_twist_drain_failure")

    bridge.poll_twist = fail_drain  # type: ignore[method-assign]
    writes_before = len(sink.commands)
    try:
        with pytest.raises(RuntimeError, match="injected_twist_drain_failure"):
            runtime.apply(_scan_stair_freeze_action())

        assert len(sink.commands) == writes_before + 1
        assert sink.commands[-1] == (0.0, 0.0, 0.0)
        assert bridge.stair_frozen_calls == [(True, 0.1)]
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_runtime_apply_latches_navigation_emergency_stop_until_reset() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    sink = _FakeCommandSink()
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx_rate=10.0,
            max_vy_rate=10.0,
            max_wz_rate=10.0,
            require_odometry=False,
            require_point_cloud=False,
        ),
    )
    gate.claim("scan_cmd_vel", 0.0)
    bridge = _FakeCommandBridge(
        [
            OgnTwistSample(
                linear_velocity=(0.3, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                receipt_timestamp=0.1,
                sequence=1,
            )
        ],
        enable_stair_execution_frozen_publisher=True,
    )
    processed_actions: list[_FakePolicyAction] = []
    runtime._runtime = SimpleNamespace(  # type: ignore[attr-defined]
        physics_dt=0.02,
        device="cpu",
        action_manager=SimpleNamespace(
            process_action=processed_actions.append,
        ),
    )
    runtime._ros2_physics_step_count = 5  # type: ignore[attr-defined]
    runtime._ros2_ogn_bridge = bridge  # type: ignore[attr-defined]
    runtime._cmd_vel_to_policy = gate  # type: ignore[attr-defined]
    runtime._cmd_vel_owner_id = "scan_cmd_vel"  # type: ignore[attr-defined]
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    runtime._environment_terminated = False  # type: ignore[attr-defined]
    runtime._scan_policy_write_sequence = 0  # type: ignore[attr-defined]
    runtime._metadata = {}  # type: ignore[attr-defined]
    runtime._adapter = SimpleNamespace(  # type: ignore[attr-defined]
        compute_policy_action=lambda **_kwargs: _FakePolicyAction(),
    )
    runtime._require_ready = lambda: None  # type: ignore[method-assign]
    runtime._apply_object_initialization_pose_stabilization = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._configure_manipulation_base_lock = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._stage_arm_target = lambda _action: {}  # type: ignore[method-assign]
    runtime._stage_gripper_target = lambda _action: {}  # type: ignore[method-assign]
    runtime._poll_scan_goal_reached = lambda: None  # type: ignore[method-assign]
    runtime._update_velocity_command_visualization = (  # type: ignore[method-assign]
        lambda _action: None
    )
    runtime._record_joint_action_apply = (  # type: ignore[method-assign]
        lambda _action, _arm, _gripper: None
    )

    runtime.apply(
        RobotAction(
            source="exec_nav_to_pick",
            metadata={
                "navigation_emergency_stop": True,
                "navigation_emergency_stop_reason": "locomotion_stall",
            },
        )
    )

    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    assert bridge.poll_timestamps == []
    assert runtime._metadata["scan_cmd_vel_last_write_report"] == {  # type: ignore[attr-defined]
        "write_sequence": 1,
        "timestamp": 0.1,
        "owner_id": "scan_cmd_vel",
        "requested_command": None,
        "limited_target": [0.0, 0.0, 0.0],
        "written_command": [0.0, 0.0, 0.0],
        "motion_allowed": False,
        "stop_reasons": ["locomotion_stall"],
        "clipped_axes": [],
        "rate_limited_axes": [],
        "navigation_emergency_stop_latched": True,
        "navigation_emergency_stop_reason": "locomotion_stall",
        "navigation_cmd_vel_inhibited": False,
        "navigation_cmd_vel_inhibit_reason": None,
        "cmd_vel_source_sequence": None,
        "cmd_vel_source_receipt_timestamp": None,
        "cmd_vel_sample_received_this_tick": False,
        "cmd_vel_sample_drained_this_tick": False,
        "last_cmd_vel_drain_sequence": None,
        "last_cmd_vel_drain_receipt_timestamp": None,
        "navigation_status_observed_report": None,
        "policy_navigation_gate_consumed_report": {
            "schema": "navigation_policy_gate_diagnostics_v1",
            "required": True,
            "timeout_s": 0.25,
            "status_fault": None,
            "permit_received": False,
            "permit": None,
            "command_identity": None,
            "command_identity_matches_permit": False,
        },
        "pipeline_base_velocity_ignored": [0.0, 0.0, 0.0],
    }

    # 后续普通 action 不能解除锁存，也不能消费桥上等待的非零 Twist。
    runtime._ros2_physics_step_count = 6  # type: ignore[attr-defined]
    runtime.apply(RobotAction.idle(source="terminal_retry"))
    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    assert bridge.poll_timestamps == []
    assert len(bridge.samples) == 1
    assert runtime._scan_policy_write_sequence == 2  # type: ignore[attr-defined]
    assert len(processed_actions) == 2
    assert bridge.stair_frozen_calls == [(True, 0.1), (True, 0.12)]
    frozen_report = runtime._metadata[  # type: ignore[attr-defined]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    assert frozen_report["value"] is True
    assert frozen_report["action_source"] == "terminal_retry"
    assert (
        frozen_report["decision_reason"]
        == "navigation_emergency_stop_latched"
    )

    # reset 的核心顺序是安全门先写零清输入，再解除 runtime 锁存。
    gate.reset(owner_id="scan_cmd_vel", now=0.12)
    runtime._navigation_emergency_stop_reason = None  # type: ignore[attr-defined]
    assert sink.commands[-1] == (0.0, 0.0, 0.0)
    gate.release(owner_id="scan_cmd_vel", now=0.12)


def test_runtime_no_physics_action_still_refreshes_stair_freeze_snapshot(
) -> None:
    runtime, sink, gate, bridge, processed_actions = (
        _runtime_with_nonzero_scan_command()
    )
    try:
        runtime.apply(
            RobotAction(
                source="reset_audit_frame",
                metadata={"skip_physics_step": True},
            )
        )

        assert bridge.stair_frozen_calls == [(False, 0.1)]
        assert processed_actions == []
        assert sink.commands[-1] == (0.3, 0.0, 0.0)
        assert runtime._action_prepared is False  # type: ignore[attr-defined]
    finally:
        gate.release(owner_id="scan_cmd_vel", now=0.1)


def test_depth_only_mode_enables_front_sensor_without_rgb_recording() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        enable_front_camera=False,
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(),
    )

    assert runtime._navigation_depth_camera_enabled() is True
    assert runtime._front_camera_sensor_enabled() is True
    assert runtime._camera_sensors_enabled() is True


def test_depth_publish_interval_can_raise_render_rate_without_changing_rgb_rate() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        camera_render_interval_control_steps=10,
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(
            publish_interval_control_steps=5,
        ),
    )

    assert runtime._effective_camera_render_interval_control_steps() == 5
    assert runtime._config.camera_render_interval_control_steps == 10  # type: ignore[attr-defined]


def test_render_interval_covers_non_divisible_rgb_and_cloud_grids() -> None:
    runtime = object.__new__(IsaacLabNavigationRuntime)
    runtime._config = IsaacLabNavigationRuntimeConfig(  # type: ignore[attr-defined]
        camera_render_interval_control_steps=10,
        ros2_ogn_bridge_config=IsaacRos2OgnBridgeConfig(),
        depth_point_cloud_config=DepthPointCloudConfig(
            publish_interval_control_steps=4,
        ),
    )

    assert runtime._effective_camera_render_interval_control_steps() == 2
    assert 10 % runtime._effective_camera_render_interval_control_steps() == 0
    assert 4 % runtime._effective_camera_render_interval_control_steps() == 0


def test_episode_reset_methods_do_not_reset_continuous_ros_physics_count() -> None:
    reset_source = inspect.getsource(IsaacLabNavigationRuntime.reset)
    prepare_source = inspect.getsource(IsaacLabNavigationRuntime.prepare_episode)

    assert "_ros2_physics_step_count" not in reset_source
    assert "_ros2_physics_step_count" not in prepare_source
