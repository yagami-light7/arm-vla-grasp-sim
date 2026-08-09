"""phase184 live summary 离线验收器的纯 Python 回归测试。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.navigation.validate_pct_scan_live_summary import (
    SummaryInputError,
    load_summary,
    main,
    validate_pct_scan_live_summary,
)
from source.navigation.scan_stair_freeze_profile import (
    load_scan_stair_freeze_profile,
)
from source.simulation.dynamic_obstacles import resolve_dynamic_obstacle_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL_STAMP_NS = 100
PATH_STAMP_NS = 200
PATH_HASH = "a" * 64
SCAN_STAIR_FREEZE_PROFILE_RUNTIME = load_scan_stair_freeze_profile(
    PROJECT_ROOT
    / "configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json",
    expected_scene="multi_floor",
    expected_robot="go2_x5",
).audit_report()


def _stamp(value_ns: int) -> dict[str, int]:
    return {"sec": value_ns // 1_000_000_000, "nanosec": value_ns % 1_000_000_000}


def _controller_identity(trajectory_id: int) -> dict[str, object]:
    bspline_stamp = {
        1: 8_500_000_000,
        2: 12_000_000_000,
    }.get(trajectory_id, 8_500_000_000 + trajectory_id)
    start_stamp = bspline_stamp + 1
    return {
        "reference_path_stamp": _stamp(PATH_STAMP_NS),
        "reference_path_stamp_ns": PATH_STAMP_NS,
        "bspline_header_stamp": _stamp(bspline_stamp),
        "bspline_header_stamp_ns": bspline_stamp,
        "start_time": _stamp(start_stamp),
        "start_time_ns": start_stamp,
        "traj_id": trajectory_id,
    }


def _grid_map_diagnostic_report(
    sequence: int,
    *,
    timestamp_s: float,
    hit_samples: list[list[float]],
    clear_samples: list[list[float]],
) -> dict[str, object]:
    hit_count = len(hit_samples)
    clear_count = len(clear_samples)
    explicit_free_count = 1 if clear_count else 0
    accepted_count = hit_count + explicit_free_count
    header_stamp = int(round(timestamp_s * 1_000_000_000))
    map_resolution = 0.10
    hit_voxel_indices = [
        [math.floor(value / map_resolution) for value in point]
        for point in hit_samples
    ]
    clear_voxel_indices = [
        [math.floor(value / map_resolution) for value in point]
        for point in clear_samples
    ]
    transition_sequences = [max(sequence - 1, 0) for _ in clear_samples]
    transition_hit_header_stamps_ns = [
        max(header_stamp - 1_000_000_000, 0)
        if transition_sequence > 0
        else 0
        for transition_sequence in transition_sequences
    ]
    return {
        "source": "ros2_scan_grid_map_observation_diagnostics",
        "topic": "/planning/grid_map_observation_diagnostics",
        "receipt_timestamp": timestamp_s + 0.01,
        "rx_sequence": sequence,
        "ros_time_offset_s": 0.0,
        "header": {
            "frame_id": "world",
            "stamp": _stamp(header_stamp),
            "stamp_ns": header_stamp,
        },
        "episode_elapsed_time_s": timestamp_s,
        "observation_sequence": sequence,
        "sensor_pose_stamp": {
            **_stamp(header_stamp - 1),
            "stamp_ns": header_stamp - 1,
        },
        "sensor_origin_world_xyz": [-4.2, 3.0, 0.4],
        "canonical_empty": False,
        "map_fusion_performed": accepted_count > 0,
        "map_resolution_m": map_resolution,
        "input_point_count": accepted_count,
        "accepted_endpoint_count": accepted_count,
        "hit_endpoint_count": hit_count,
        "explicit_free_endpoint_count": explicit_free_count,
        "hit_endpoint_samples_truncated": False,
        "hit_endpoint_samples_world_xyz": hit_samples,
        "hit_endpoint_sample_voxel_indices_xyz": hit_voxel_indices,
        "free_to_occupied_transition_count": hit_count,
        "free_to_occupied_transition_samples_truncated": False,
        "free_to_occupied_transition_hit_samples_world_xyz": hit_samples,
        "free_to_occupied_transition_voxel_indices_xyz": hit_voxel_indices,
        "explicit_free_miss_voxel_count": 3 if clear_count else 0,
        "occupied_to_free_by_explicit_miss_count": clear_count,
        "occupied_to_free_samples_truncated": False,
        "occupied_to_free_by_explicit_miss_samples_world_xyz": clear_samples,
        "occupied_to_free_sample_voxel_indices_xyz": clear_voxel_indices,
        "occupied_to_free_transition_hit_observation_sequences": (
            transition_sequences
        ),
        "occupied_to_free_transition_hit_samples_world_xyz": clear_samples,
        "occupied_to_free_transition_hit_header_stamp_ns": (
            transition_hit_header_stamps_ns
        ),
        "occupied_removed_by_sliding_reset_count": 0,
        "dynamic_obstacle_hit_matches": [],
        "dynamic_obstacle_transition_hit_matches": [],
        "dynamic_obstacle_explicit_miss_clear_matches": [],
    }


def _bspline_diagnostic_report(
    sequence: int,
    trajectory_id: int,
    *,
    maximum_deviation: float,
    trajectory_samples: list[list[float]],
    reference_samples: list[list[float]],
) -> dict[str, object]:
    identity = _controller_identity(trajectory_id)
    header_stamp = identity["bspline_header_stamp_ns"]
    assert isinstance(header_stamp, int)
    return {
        "source": "ros2_scan_bspline_diagnostics",
        "topic": "/planning/bspline_diagnostics",
        "receipt_timestamp": 5.0 + sequence,
        "rx_sequence": sequence,
        "ros_time_offset_s": 0.0,
        "header": {
            "frame_id": "world",
            "stamp": _stamp(header_stamp),
            "stamp_ns": header_stamp,
        },
        "episode_elapsed_time_s": header_stamp * 1.0e-9,
        "diagnostic_sequence": sequence,
        "identity": identity,
        "is_final": False,
        "emergency_stop": False,
        "stationary": False,
        "ordered_reference_checked": True,
        "ordered_reference_safe": True,
        "maximum_trajectory_deviation_m": maximum_deviation,
        "maximum_guide_anchor_deviation_m": maximum_deviation,
        "maximum_guide_progress_lead_m": 0.01,
        "maximum_deviation_limit_m": 0.10,
        "maximum_progress_lead_limit_m": 0.02,
        "trajectory_duration_s": 0.02,
        "maximum_velocity_upper_bound_mps": 1.0,
        "double_cylinder_radius_m": 0.27,
        "double_cylinder_offset_m": 0.16,
        "required_any_yaw_clearance_radius_m": 0.43,
        "trajectory_sample_interval_s": 0.01,
        "sampling_clearance_margin_m": 0.005,
        "trajectory_sample_count_total": len(trajectory_samples),
        "trajectory_samples_truncated": False,
        "trajectory_samples_world_xyz": trajectory_samples,
        "ordered_reference_sample_count_total": len(reference_samples),
        "ordered_reference_samples_truncated": False,
        "ordered_reference_samples_world_xyz": reference_samples,
        "detour_deviation_minimum_m": 0.02,
        "dynamic_obstacle_clearances": [],
        "dynamic_obstacle_relevant": True,
        "ordered_detour_candidate": maximum_deviation >= 0.02,
    }


def _controller_status(
    sequence: int,
    trajectory_id: int,
    *,
    state: int,
    event: int,
    reason: str,
) -> dict[str, object]:
    header_stamp_ns = {
        1: 9_000_000_000,
        2: 12_100_000_000,
        3: 13_000_000_000,
    }.get(sequence, 13_000_000_000 + sequence)
    receipt_timestamp = {
        1: 7.0,
        2: 7.5,
        3: 8.0,
    }.get(sequence, 8.0 + sequence)
    return {
        "source": "ros2_scan_planner_msgs_controller_status",
        "topic": "/planning/controller_status",
        "receipt_timestamp": receipt_timestamp,
        "rx_sequence": sequence,
        "header": {
            "frame_id": "world",
            "stamp": _stamp(header_stamp_ns),
            "stamp_ns": header_stamp_ns,
        },
        "status_sequence": sequence,
        "acceptance_sequence": trajectory_id,
        "event": event,
        "state": state,
        "reason": reason,
        "accepted": True,
        "trajectory_valid": True,
        "is_final": state == 12,
        "emergency_stop": False,
        "identity": _controller_identity(trajectory_id),
        "candidate": None,
    }


def _controller_lifecycle() -> dict[str, object]:
    first = _controller_status(1, 1, state=10, event=1, reason="首条局部轨迹已接受")
    second = _controller_status(2, 2, state=10, event=1, reason="新局部轨迹已接受")
    last = _controller_status(3, 2, state=12, event=4, reason="目标已到达")
    return {
        "schema": "scan_controller_status_lifecycle_v1",
        "sample_count": 3,
        "first_status_sequence": 1,
        "last_status_sequence": 3,
        "maximum_acceptance_sequence": 2,
        "event_counts": {"accepted": 2, "state_changed": 1},
        "state_counts": {"tracking": 2, "goal_reached": 1},
        "reason_counts": {
            "首条局部轨迹已接受": 1,
            "新局部轨迹已接受": 1,
            "目标已到达": 1,
        },
        "accepted_status_count": 3,
        "trajectory_valid_status_count": 3,
        "candidate_rejection_count": 0,
        "goal_latched_same_path_candidate_rejection_count": 0,
        "unexpected_candidate_rejection_count": 0,
        "first_candidate_rejection_status": None,
        "last_candidate_rejection_status": None,
        "emergency_stop_status_count": 0,
        "tracking_status_count": 2,
        "tracking_status_reports": [first, second],
        "dropped_tracking_status_report_count": 0,
        "goal_reached_status_count": 1,
        "distinct_accepted_trajectory_count": 2,
        "trajectory_replacement_count": 1,
        "accepted_trajectory_identities": [
            _controller_identity(1),
            _controller_identity(2),
        ],
        "accepted_status_reports": [first, second, last],
        "dropped_accepted_status_report_count": 0,
        "first_accepted_status": first,
        "last_accepted_status": last,
        "first_tracking_status": first,
        "last_tracking_status": second,
        "first_emergency_stop_status": None,
        "last_emergency_stop_status": None,
        "tracking_after_emergency_stop_observed": False,
        "emergency_stop_recovery_count": 0,
        "emergency_stop_pending_recovery": False,
        "last_status": last,
    }


def _observed_status_evidence(
    sequence: int,
    revision: int,
    *,
    tracking: bool,
    write_sequence: int,
) -> dict[str, object]:
    state = 3 if tracking else 6
    return {
        "write_sequence": write_sequence,
        "timestamp": float(write_sequence),
        "navigation_status_observed_report": {
            "schema": "navigation_status_observed_diagnostics_v1",
            "topic": "/navigation/status",
            "status_error": None,
            "local_pct_goal_stamp_ns": GOAL_STAMP_NS,
            "local_active_path_stamp_ns": PATH_STAMP_NS,
            "local_reference_path_identity_fault": None,
            "status": {
                "receipt_timestamp": float(write_sequence),
                "rx_sequence": sequence,
                "header_stamp_ns": 600 + sequence,
                "status_sequence": sequence,
                "state_revision": revision,
                "goal_id": GOAL_STAMP_NS,
                "state": state,
                "allow_tracking_command": tracking,
                "force_zero_velocity": not tracking,
                "stop_confirmed": not tracking,
                "global_replan_requested": False,
                "global_replan_in_flight": False,
                "global_replan_request_id": 0,
                "pct_plan_id": 1,
                "active_path_stamp_ns": PATH_STAMP_NS,
                "consecutive_scan_failures": 0,
                "stale_inputs": [],
                "reason": "TRACKING" if tracking else "GOAL_REACHED",
                "identity_valid": True,
            },
        },
    }


def _consumed_tracking_evidence(
    sequence: int,
    revision: int,
    *,
    write_sequence: int,
) -> dict[str, object]:
    controller_sequence = 1 if revision == 1 else 2
    controller_snapshot = _controller_status(
        controller_sequence,
        revision,
        state=10,
        event=1,
        reason=(
            "首条局部轨迹已接受"
            if revision == 1
            else "新局部轨迹已接受"
        ),
    )
    write_timestamp = {
        2: 9.2,
        4: 9.4,
        6: 12.2,
        8: 12.4,
    }.get(write_sequence, float(write_sequence))
    return {
        "write_sequence": write_sequence,
        "timestamp": write_timestamp,
        "written_command": [0.2, 0.0, 0.0],
        "navigation_gate_diagnostics": {
            "schema": "navigation_policy_gate_diagnostics_v1",
            "required": True,
            "timeout_s": 0.25,
            "status_fault": None,
            "permit_received": True,
            "permit": {
                "header_stamp_ns": 700 + sequence,
                "received_at": float(write_sequence),
                "status_sequence": sequence,
                "state_revision": revision,
                "goal_id": GOAL_STAMP_NS,
                "active_path_stamp_ns": PATH_STAMP_NS,
                "state": 3,
                "allow_tracking_command": True,
                "force_zero_velocity": False,
                "identity_valid": True,
                "reason": "TRACKING",
            },
            "command_identity": [GOAL_STAMP_NS, PATH_STAMP_NS, revision],
            "command_identity_matches_permit": True,
        },
        "scan_controller_status_snapshot": controller_snapshot,
    }


def _policy_lifecycle(*, tracking: bool) -> dict[str, object]:
    if not tracking:
        first_observed = {
            "write_sequence": 2,
            "timestamp": 1.04,
            "navigation_status_observed_report": {
                "schema": "navigation_status_observed_diagnostics_v1",
                "topic": "/navigation/status",
                "status_error": None,
                "local_pct_goal_stamp_ns": GOAL_STAMP_NS,
                "local_active_path_stamp_ns": PATH_STAMP_NS,
                "local_reference_path_identity_fault": None,
                "status": {
                    "receipt_timestamp": 1.04,
                    "rx_sequence": 1,
                    "header_stamp_ns": 1_040_000_000,
                    "status_sequence": 1,
                    "state_revision": 1,
                    "goal_id": GOAL_STAMP_NS,
                    "state": 2,
                    "allow_tracking_command": False,
                    "force_zero_velocity": True,
                    "stop_confirmed": True,
                    "global_replan_requested": False,
                    "global_replan_in_flight": False,
                    "global_replan_request_id": 0,
                    "pct_plan_id": 1,
                    "active_path_stamp_ns": PATH_STAMP_NS,
                    "consecutive_scan_failures": 0,
                    "stale_inputs": ["point_cloud", "bspline"],
                    "reason": "global_path_available",
                    "identity_valid": True,
                },
            },
        }
        last_observed = _observed_status_evidence(
            3,
            3,
            tracking=False,
            write_sequence=9,
        )
        return {
            "schema": "navigation_policy_gate_lifecycle_v1",
            "policy_write_count": 10,
            "motion_allowed_write_count": 0,
            "identity_verified_tracking_write_count": 0,
            "identity_verified_tracking_snapshot_count": 0,
            "observed_status_sequence_count": 3,
            "identity_valid_observed_status_count": 3,
            "forced_zero_write_count": 10,
            "first_identity_verified_tracking_write": None,
            "last_identity_verified_tracking_write": None,
            "identity_verified_tracking_write_reports": [],
            "dropped_identity_verified_tracking_write_report_count": 0,
            "first_identity_valid_observed_status": first_observed,
            "last_identity_valid_observed_status": last_observed,
            "last_observed_status_sequence": 3,
            "last_observed_state": 6,
            "observed_state_transition_count": 2,
            "observed_state_counts": {
                "local_planning": 1,
                "emergency_stop": 1,
                "goal_reached": 1,
            },
            "observed_reason_counts": {
                "global_path_available": 1,
                "scan_stair_execution_inhibited": 1,
                "GOAL_REACHED": 1,
            },
            "maximum_consecutive_scan_failures": 0,
            "global_replan_requested_status_count": 0,
            "global_replan_in_flight_status_count": 0,
            "distinct_global_replan_request_ids": [],
            "distinct_pct_plan_ids": [1],
            "first_global_replan_status": None,
            "last_global_replan_status": None,
            "global_replan_pending_recovery": False,
            "tracking_after_global_replan_observed": False,
            "global_replan_recovery_count": 0,
            "emergency_stop_observed_status_count": 1,
            "goal_reached_observed_status_count": 1,
            "last_observed_status": last_observed,
            "last_write_sequence": 10,
            "last_stop_reasons": ["scan_stair_terminal_hold"],
            "stop_reason_counts": {
                "scan_stair_freeze": 8,
                "scan_stair_terminal_hold": 2,
            },
        }
    first_observed = _observed_status_evidence(1, 1, tracking=True, write_sequence=2)
    last_observed = _observed_status_evidence(3, 3, tracking=False, write_sequence=9)
    first_tracking_write = _consumed_tracking_evidence(
        1,
        1,
        write_sequence=2,
    )
    last_tracking_write = _consumed_tracking_evidence(
        2,
        2,
        write_sequence=8,
    )
    tracking_writes = (
        [
            first_tracking_write,
            _consumed_tracking_evidence(2, 2, write_sequence=6),
        ]
        if tracking
        else []
    )
    return {
        "schema": "navigation_policy_gate_lifecycle_v1",
        "policy_write_count": 10,
        "motion_allowed_write_count": 4 if tracking else 0,
        "identity_verified_tracking_write_count": 4 if tracking else 0,
        "identity_verified_tracking_snapshot_count": 2 if tracking else 0,
        "observed_status_sequence_count": 3,
        "identity_valid_observed_status_count": 3,
        "forced_zero_write_count": 6 if tracking else 10,
        "first_identity_verified_tracking_write": (
            first_tracking_write
            if tracking
            else None
        ),
        "last_identity_verified_tracking_write": (
            last_tracking_write
            if tracking
            else None
        ),
        "identity_verified_tracking_write_reports": tracking_writes,
        "dropped_identity_verified_tracking_write_report_count": 0,
        "first_identity_valid_observed_status": first_observed,
        "last_identity_valid_observed_status": last_observed,
        "last_observed_status_sequence": 3,
        "last_observed_state": 6,
        "observed_state_transition_count": 1,
        "observed_state_counts": {"tracking": 2, "goal_reached": 1},
        "observed_reason_counts": {"TRACKING": 2, "GOAL_REACHED": 1},
        "maximum_consecutive_scan_failures": 0,
        "global_replan_requested_status_count": 0,
        "global_replan_in_flight_status_count": 0,
        "distinct_global_replan_request_ids": [],
        "distinct_pct_plan_ids": [1],
        "first_global_replan_status": None,
        "last_global_replan_status": None,
        "global_replan_pending_recovery": False,
        "tracking_after_global_replan_observed": False,
        "global_replan_recovery_count": 0,
        "emergency_stop_observed_status_count": 0,
        "goal_reached_observed_status_count": 1,
        "last_observed_status": last_observed,
        "last_write_sequence": 10,
        "last_stop_reasons": ["goal_reached"],
        "stop_reason_counts": {"goal_reached": 2},
    }


def _bridge_report() -> dict[str, object]:
    return {
        "enabled": True,
        "command_topic": "/cmd_vel",
        "controller_status_topic": "/planning/controller_status",
        "grid_map_diagnostics_topic": "/planning/grid_map_observation_diagnostics",
        "bspline_diagnostics_topic": "/planning/bspline_diagnostics",
        "navigation_status_topic": "/navigation/status",
        "navigation_status_qos_profile": json.dumps(
            {
                "history": "keepLast",
                "depth": 1,
                "reliability": "reliable",
                "durability": "transientLocal",
            }
        ),
        "stair_execution_frozen_topic": "/planning/stair_execution_frozen",
        "reference_path_topic": "/pct/global_path",
        "pct_goal_topic": "/pct/goal",
        "command_authority_enabled": True,
        "navigation_status_gate_required": True,
        "navigation_status_timeout_s": 0.25,
        "goal_lifecycle_enabled": True,
        "controller_status_subscription_enabled": True,
        "grid_map_diagnostics_subscription_enabled": True,
        "bspline_diagnostics_subscription_enabled": True,
        "stair_execution_frozen_publisher_enabled": True,
        "reference_path_subscription_enabled": True,
        "pct_goal_publisher_enabled": True,
        "odom_frame_id": "world",
        "base_frame_id": "base_link",
        "point_cloud_frame_id": "world",
        "continuous_time_source": "successful_physics_steps_x_physics_dt",
    }


def _stair_sensor_acquisition_fixture() -> dict[str, object]:
    """构造冻结激活后、root 尚未前进时通过的本代传感器屏障。"""

    activation_timestamp = 1.0
    status_receipt_timestamp = 1.08
    write_timestamp = 1.10
    write_sequence = 3
    status_report: dict[str, object] = {
        "schema": "navigation_status_observed_diagnostics_v1",
        "topic": "/navigation/status",
        "status_error": None,
        "local_pct_goal_stamp_ns": GOAL_STAMP_NS,
        "local_active_path_stamp_ns": PATH_STAMP_NS,
        "local_reference_path_identity_fault": None,
        "status": {
            "receipt_timestamp": status_receipt_timestamp,
            "rx_sequence": 2,
            "header_stamp_ns": 1_080_000_000,
            "status_sequence": 2,
            "state_revision": 2,
            "goal_id": GOAL_STAMP_NS,
            "state": 5,
            "allow_tracking_command": False,
            "force_zero_velocity": True,
            "stop_confirmed": True,
            "global_replan_requested": False,
            "global_replan_in_flight": False,
            "global_replan_request_id": 0,
            "pct_plan_id": 1,
            "active_path_stamp_ns": PATH_STAMP_NS,
            "consecutive_scan_failures": 0,
            "stale_inputs": ["bspline"],
            "reason": "scan_stair_execution_inhibited",
            "identity_valid": True,
        },
    }
    policy_write_report = {
        "write_sequence": write_sequence,
        "timestamp": write_timestamp,
        "owner_id": "scan_cmd_vel",
        "written_command": [0.0, 0.0, 0.0],
        "motion_allowed": False,
        "stop_reasons": ["scan_stair_freeze"],
        "navigation_cmd_vel_inhibited": True,
        "navigation_cmd_vel_inhibit_reason": "scan_stair_freeze",
        "navigation_status_observed_report": status_report,
    }
    return {
        "required": True,
        "passed": True,
        "pending": False,
        "path_stamp_ns": PATH_STAMP_NS,
        "activation_timestamp": activation_timestamp,
        "timeout_s": 8.0,
        "status_freshness_timeout_s": 0.25,
        "write_sequence": write_sequence,
        "write_timestamp": write_timestamp,
        "progress_m_at_pass": 0.0,
        "local_sensors_fresh": True,
        "supervisor_sensors_fresh": True,
        "last_write_sequence": write_sequence,
        "last_write_timestamp": write_timestamp,
        "pending_reasons": [],
        "navigation_status_observed_report": status_report,
        "last_navigation_status_observed_report": status_report,
        "policy_write_report": policy_write_report,
    }


def _executor(*, stair: bool) -> dict[str, object]:
    stair_report: dict[str, object]
    if stair:
        stair_report = {
            "enabled": True,
            "applicable": True,
            "phase": "terminal_hold",
            "path_stamp_ns": PATH_STAMP_NS,
            "path_points_sha256": PATH_HASH,
            "terminal_component": True,
            "terminal_hold": True,
            "terminal_goal_bound": True,
            "active": True,
            "finish_ready": True,
            "certified_progress_seen": True,
            "progress_m": 8.0,
            "total_length_m": 8.0,
            "progress_ratio": 1.0,
            "emergency_hold_latched": False,
            "invalid_controller_status_count": 0,
            "controller_status_sequence_reset_count": 0,
            "terminal_supervisor_goal_acknowledged": True,
            "terminal_supervisor_goal_pending_started_timestamp": None,
            "sensor_safety_fault_reasons": [],
            "sensor_safety_fault_write_sequence": None,
            "sensor_safety_fault_timestamp": None,
            "policy_freeze_write_fault_reasons": [],
            "policy_freeze_write_fault_sequence": None,
            "policy_freeze_write_fault_timestamp": None,
            "sensor_acquisition_required": True,
            "sensor_acquisition_pending": False,
            "sensor_acquisition_complete": True,
            "sensor_acquisition_started_timestamp": 1.0,
            "sensor_acquisition_completed_timestamp": 1.10,
            "sensor_acquisition_timeout_s": 8.0,
            "sensor_acquisition_write_sequence_floor": 2,
            "sensor_acquisition_write_sequence": 3,
            "sensor_acquisition_last_write_sequence": 3,
            "sensor_acquisition_last_write_timestamp": 1.10,
            "sensor_acquisition_pending_reasons": [],
            "sensor_acquisition_barrier": (
                _stair_sensor_acquisition_fixture()
            ),
            "non_physical_root_lock_workaround": True,
        }
    else:
        stair_report = {
            "enabled": True,
            "applicable": False,
            "phase": "not_applicable",
            "active": False,
            "non_physical_root_lock_workaround": False,
        }
    return {
        "backend": "scan_ros2_goal_event",
        "phase": "completed",
        "generation": 1,
        "tick_index": 20,
        "done": True,
        "success": True,
        "failed": False,
        "failure_reason": "",
        "fresh_false_seen": True,
        "policy_activity_seen": not stair,
        "certified_root_lock_progress_seen": stair,
        "execution_activity_seen": True,
        "goal_rising_edge_seen": True,
        "goal_false_sequence": 1,
        "goal_true_sequence": 2,
        "goal_false_receipt_timestamp": 8.0,
        "goal_true_receipt_timestamp": 9.0,
        "zero_write_streak": 5,
        "required_zero_write_ticks": 5,
        "last_policy_write_sequence": 10,
        "last_policy_write_timestamp": 10.0,
        "last_requested_command": [0.0, 0.0, 0.0],
        "last_written_command": [0.0, 0.0, 0.0],
        "last_stop_reasons": ["scan_stair_terminal_hold"] if stair else ["goal_reached"],
        "last_navigation_cmd_vel_inhibited": stair,
        "last_navigation_cmd_vel_inhibit_reason": "scan_stair_terminal_hold" if stair else None,
        "invalid_goal_sample_count": 0,
        "premature_true_count": 0,
        "goal_true_waiting_for_supervisor_ack_count": 0,
        "post_goal_nonzero_write_count": 0,
        "goal_sequence_reset_count": 0,
        "policy_write_sequence_reset_count": 0,
        "invalid_policy_write_count": 0,
        "invalid_progress_pose_count": 0,
        "scan_controller_goal_reached_verified": True,
        "policy_zero_hold_verified": True,
        "stair_freeze": stair_report,
        "stair_freeze_finish_ready": True,
        "live_reference_path_required": True,
        "live_reference_path_verified": True,
        "live_reference_path_sequence": 2,
        "live_reference_path_stamp_ns": PATH_STAMP_NS,
        "live_reference_path_points_sha256": PATH_HASH,
        "live_reference_path_source": "ros2_nav_msgs_path",
        "live_reference_path_generation_count": 1,
        "live_reference_path_goal_bound": True,
        "live_reference_path_goal_xy_error_m": 0.0,
        "live_reference_path_goal_z_error_m": 0.0,
        "live_reference_path_goal_yaw_error_rad": 0.0,
        "invalid_reference_path_report_count": 0,
        "pct_goal_required": True,
        "pct_goal_local_publish_triggered": True,
        "pct_goal_acknowledged": True,
        "pct_goal_transport_acknowledged": True,
        "pct_goal_stamp_ns": GOAL_STAMP_NS,
        "pct_goal_publish_sequence": 1,
        "pct_goal_request_action_count": 1,
        "pct_goal_transport_retry_count": 0,
        "invalid_pct_goal_report_count": 0,
    }


def _base_summary(mode: str) -> dict[str, object]:
    stair = mode == "static_stair"
    crossfloor = mode == "crossfloor_carry"
    task_id = (
        17705
        if mode == "dynamic_replan_f1"
        else (17704 if mode == "dynamic_f1" else 1002)
    )
    task: dict[str, object] = {
        "task_id": task_id,
        "scene_profile": "multi_floor",
        "start": {"floor_id": "F1"},
        "pick": {"base_goal": {"floor_id": "F1"}},
        "place": {
            "enabled": mode not in {"dynamic_f1", "dynamic_replan_f1"},
            "base_goal": {"floor_id": "F2" if crossfloor else "F1"},
        },
    }
    if mode in {"dynamic_f1", "dynamic_replan_f1"}:
        task_name = (
            "nav_smoke_scan_multifloor_dynamic_blocker_replan_f1.json"
            if mode == "dynamic_replan_f1"
            else "nav_smoke_scan_multifloor_dynamic_cart_f1.json"
        )
        task = json.loads(
            (PROJECT_ROOT / "tasks" / task_name).read_text(encoding="utf-8")
        )
    task["scan_stair_freeze_profile_runtime"] = dict(
        SCAN_STAIR_FREEZE_PROFILE_RUNTIME
    )
    simulation: dict[str, object] = {
        "navigation_ros2_bridge_report": _bridge_report(),
        "navigation_policy_gate_lifecycle_report": _policy_lifecycle(tracking=not stair),
        "scan_controller_status_lifecycle_report": _controller_lifecycle(),
    }
    executor = _executor(stair=stair)
    if crossfloor:
        executor["execution_phase"] = "carry_nav_to_place"
        lifecycle = simulation["navigation_policy_gate_lifecycle_report"]
        assert isinstance(lifecycle, dict)
        stop_reason_counts = lifecycle["stop_reason_counts"]
        assert isinstance(stop_reason_counts, dict)
        stop_reason_counts["scan_stair_freeze"] = 8
        stop_reason_counts["scan_stair_freeze_release"] = 2
    summary: dict[str, object] = {
        "task_id": task_id,
        "episode_id": 1,
        "success": True,
        "final_state": "done",
        "failure_reason": "",
        "failure_metadata": {},
        "execution_provenance_verified": True,
        "execution_mode": (
            "stair_locomotion_smoke"
            if stair
            else (
                "navigation_carry_smoke"
                if crossfloor
                else "navigation_smoke"
            )
        ),
        "success_semantics": (
            "scan_stair_root_lock_workaround"
            if stair
            else (
                "physical_nav_to_place_with_arm_gripper_hold_with_"
                "scan_stair_root_lock_workaround"
                if crossfloor
                else "physical_nav_to_pick_only"
            )
        ),
        "state_trace": (
            [
                "build_stage",
                "reset_episode",
                "plan_nav_to_place",
                "exec_nav_to_place",
                "cleanup_episode",
                "done",
            ]
            if crossfloor
            else [
                "build_stage",
                "reset_episode",
                "plan_nav_to_pick",
                "exec_nav_to_pick",
                "cleanup_episode",
                "done",
            ]
        ),
        "task_config": task,
        "latest_executor_status": executor,
        "simulation_report": simulation,
        "navigation_root_lock_workaround_success": stair or crossfloor,
        "physical_navigation_success": not (stair or crossfloor),
        "pure_physics_success": False,
        "used_base_teleport": stair or crossfloor,
        "used_direct_joint_state": stair or crossfloor,
        "used_object_teleport": crossfloor,
        "used_kinematic_object_follow": crossfloor,
        "used_visual_replay": False,
        "used_navigation_base_lock": stair or crossfloor,
        "used_navigation_support_joint_lock": stair or crossfloor,
        "used_navigation_joint_pose_lock": stair or crossfloor,
    }
    if stair:
        simulation["navigation_stair_execution_frozen_last_publish_report"] = {
            "schema": "isaac_stair_execution_frozen_v1",
            "message_type": "scan_planner_msgs/msg/StairExecutionFreeze",
            "source": "isaac_action_metadata",
            "topic": "/planning/stair_execution_frozen",
            "header": {
                "frame_id": "world",
                "stamp": _stamp(10_000_000_000),
            },
            "reference_path_stamp": _stamp(PATH_STAMP_NS),
            "reference_path_stamp_ns": PATH_STAMP_NS,
            "writer_id": "isaac_ros2_ogn_bridge",
            "writer_epoch": "typed-freeze-fixture-epoch",
            "frozen": True,
            "value": True,
            "action_source": "scan_stair_freeze_terminal_hold",
            "action_phase": "terminal_hold",
            "decision_reason": "scan_stair_terminal_hold",
            "sequence": 4,
            "publish_timestamp": 10.0,
        }
    if mode in {"dynamic_f1", "dynamic_replan_f1"}:
        _add_dynamic_reports(summary)
    if mode == "dynamic_replan_f1":
        _add_global_replan_reports(summary)
    return summary


def _add_global_replan_reports(summary: dict[str, object]) -> None:
    """为纯 Python fixture 加入旧 Path→PCT replan→新 Path 的 typed 状态。"""

    simulation = summary["simulation_report"]
    assert isinstance(simulation, dict)
    lifecycle = simulation["navigation_policy_gate_lifecycle_report"]
    assert isinstance(lifecycle, dict)
    grid_lifecycle = simulation["grid_map_observation_lifecycle_report"]
    assert isinstance(grid_lifecycle, dict)
    transition_report = grid_lifecycle["first_transition_hit_report"]
    clear_report = grid_lifecycle["last_explicit_miss_clear_report"]
    assert isinstance(transition_report, dict)
    assert isinstance(clear_report, dict)
    transition_match = transition_report[
        "dynamic_obstacle_transition_hit_matches"
    ][0]
    assert isinstance(transition_match, dict)
    transition_point = transition_match["point_world_xyz"]
    assert isinstance(transition_point, list)
    hit_receipt = float(transition_report["receipt_timestamp"])
    clear_receipt = float(clear_report["receipt_timestamp"])
    replan_receipt = hit_receipt + 0.5
    assert replan_receipt < clear_receipt
    bspline_lifecycle = simulation["bspline_diagnostics_lifecycle_report"]
    assert isinstance(bspline_lifecycle, dict)
    first_bspline = bspline_lifecycle["first_report"]
    assert isinstance(first_bspline, dict)
    first_bspline["receipt_timestamp"] = max(0.0, hit_receipt - 0.25)
    first_identity = first_bspline["identity"]
    assert isinstance(first_identity, dict)
    first_identity["reference_path_stamp"] = _stamp(100)
    first_identity["reference_path_stamp_ns"] = 100
    first_bspline["ordered_reference_samples_world_xyz"] = [
        [transition_point[0] - 0.1, transition_point[1], transition_point[2]],
        list(transition_point),
        [transition_point[0] + 0.1, transition_point[1], transition_point[2]],
    ]
    trajectory_identities = bspline_lifecycle["trajectory_identities"]
    assert isinstance(trajectory_identities, list)
    trajectory_identities[0] = deepcopy(first_identity)
    replan_status = {
        "write_sequence": 5,
        "timestamp": replan_receipt,
        "navigation_status_observed_report": {
            "schema": "navigation_status_observed_diagnostics_v1",
            "topic": "/navigation/status",
            "status_error": None,
            "local_pct_goal_stamp_ns": GOAL_STAMP_NS,
            "local_active_path_stamp_ns": 100,
            "local_reference_path_identity_fault": None,
            "status": {
                "receipt_timestamp": replan_receipt,
                "rx_sequence": 2,
                "header_stamp_ns": 900,
                "status_sequence": 2,
                "state_revision": 2,
                "goal_id": GOAL_STAMP_NS,
                "state": 4,
                "allow_tracking_command": False,
                "force_zero_velocity": True,
                "stop_confirmed": True,
                "global_replan_requested": True,
                "global_replan_in_flight": True,
                "global_replan_request_id": 1,
                "pct_plan_id": 1,
                "active_path_stamp_ns": 100,
                "consecutive_scan_failures": 5,
                "stale_inputs": [],
                "reason": "scan_consecutive_failures",
                "identity_valid": True,
            },
        },
    }
    lifecycle.update(
        {
            "maximum_consecutive_scan_failures": 5,
            "global_replan_requested_status_count": 1,
            "global_replan_in_flight_status_count": 1,
            "distinct_global_replan_request_ids": [1],
            "distinct_pct_plan_ids": [1, 2],
            "first_global_replan_status": deepcopy(replan_status),
            "last_global_replan_status": deepcopy(replan_status),
            "global_replan_pending_recovery": False,
            "tracking_after_global_replan_observed": True,
            "global_replan_recovery_count": 1,
        }
    )
    last_observed = lifecycle["last_observed_status"]
    assert isinstance(last_observed, dict)
    last_report = last_observed["navigation_status_observed_report"]
    assert isinstance(last_report, dict)
    last_status = last_report["status"]
    assert isinstance(last_status, dict)
    last_status["pct_plan_id"] = 2
    last_tracking = lifecycle["last_identity_verified_tracking_write"]
    assert isinstance(last_tracking, dict)
    last_tracking["timestamp"] = clear_receipt + 0.5


def _add_dynamic_reports(summary: dict[str, object]) -> None:
    task = summary["task_config"]
    assert isinstance(task, dict)
    plan = resolve_dynamic_obstacle_plan(task)
    replan_task = task.get("task_id") == 17705
    configuration = plan.to_dict()
    configuration["registered_scene_assets"] = [
        {
            "id": obstacle.obstacle_id,
            "scene_asset_name": obstacle.scene_asset_name,
            "prim_path": obstacle.prim_path,
            "shape": "cuboid",
            "kinematic_enabled": True,
            "collision_enabled": True,
            "visible": True,
        }
        for obstacle in plan.obstacles
    ]
    first_time = 0.0
    last_time = max(
        obstacle.start_delay_s
        + obstacle.total_path_length_m / obstacle.speed_mps
        + 1.0
        for obstacle in plan.obstacles
    )
    first_states = plan.state_at(first_time)
    last_states = plan.state_at(last_time)
    frame_count = 3
    pose_write_count = frame_count * len(plan.obstacles)
    lifecycle_obstacles: dict[str, object] = {}
    for obstacle, first, last in zip(
        plan.obstacles,
        first_states,
        last_states,
        strict=True,
    ):
        span = last.path_distance_m - first.path_distance_m
        lifecycle_obstacles[obstacle.obstacle_id] = {
            "scene_asset_name": obstacle.scene_asset_name,
            "sample_count": frame_count,
            "first_state": first.to_dict(),
            "last_state": last.to_dict(),
            "minimum_path_distance_m": first.path_distance_m,
            "maximum_path_distance_m": last.path_distance_m,
            "path_distance_span_m": span,
            "path_directions_seen": [0, 1],
            "direction_transition_count": 1,
            "waiting_for_start_seen": True,
            "motion_started_seen": True,
            "maximum_displacement_from_first_m": span,
        }
    simulation = summary["simulation_report"]
    assert isinstance(simulation, dict)
    simulation.update(
        {
            "dynamic_obstacle_configuration_report": configuration,
            "dynamic_obstacle_runtime_report": {
                "enabled": True,
                "reason": "before_scene_write_data_to_sim",
                "physics_step_index": 500,
                "elapsed_time_s": last_time,
                "obstacle_count": len(plan.obstacles),
                "pose_write_count": pose_write_count,
                "obstacles": [state.to_dict() for state in last_states],
                "root_lock_state_used": False,
                "time_source": "episode_physics_step_index_x_physics_dt",
                "lifecycle_schema": "dynamic_obstacle_lifecycle_v1",
            },
            "dynamic_obstacle_lifecycle_report": {
                "schema": "dynamic_obstacle_lifecycle_v1",
                "ros_time_offset_s": 0.0,
                "enabled": True,
                "obstacle_count": len(plan.obstacles),
                "pose_write_count": pose_write_count,
                "sample_frame_count": frame_count,
                "first_physics_step_index": 0,
                "last_physics_step_index": 500,
                "first_elapsed_time_s": first_time,
                "last_elapsed_time_s": last_time,
                "all_configured_obstacles_sampled": True,
                "all_configured_obstacles_moved": True,
                "maximum_path_distance_span_m": max(
                    report["path_distance_span_m"]
                    for report in lifecycle_obstacles.values()
                    if isinstance(report, dict)
                ),
                "direction_transition_count": len(plan.obstacles),
                "obstacles": lifecycle_obstacles,
            },
            "dynamic_obstacle_pose_write_count": pose_write_count,
        }
    )
    raw_cloud_obstacles: dict[str, object] = {}
    global_first_frame: dict[str, object] | None = None
    global_last_frame: dict[str, object] | None = None
    for obstacle in plan.obstacles:
        first_detection_time = obstacle.start_delay_s + 0.5
        last_detection_time = last_time
        first_state = obstacle.state_at(first_detection_time)
        last_state = obstacle.state_at(last_detection_time)
        first_detection = {
            "timestamp": first_detection_time,
            "completed_control_step": 425,
            "physics_step_index": 425,
            "elapsed_time_s": first_detection_time,
            "point_count": 8,
            "state": first_state.to_dict(),
        }
        last_detection = {
            "timestamp": last_detection_time,
            "completed_control_step": 500,
            "physics_step_index": 500,
            "elapsed_time_s": last_detection_time,
            "point_count": 10,
            "state": last_state.to_dict(),
        }
        raw_cloud_obstacles[obstacle.obstacle_id] = {
            "scene_asset_name": obstacle.scene_asset_name,
            "sample_frame_count": 3,
            "detected_frame_count": 2,
            "maximum_point_count": 10,
            "first_detection": first_detection,
            "last_detection": last_detection,
            "motion_started_detection_seen": True,
            "path_directions_detected": sorted(
                {first_state.path_direction, last_state.path_direction}
            ),
            "minimum_detected_path_distance_m": first_state.path_distance_m,
            "maximum_detected_path_distance_m": last_state.path_distance_m,
            "detected_path_distance_span_m": (
                last_state.path_distance_m - first_state.path_distance_m
            ),
        }
        first_frame_obstacles = {
            item.obstacle_id: {
                "point_count": 8,
                "state": item.state_at(first_detection_time).to_dict(),
            }
            for item in plan.obstacles
        }
        last_frame_obstacles = {
            item.obstacle_id: {
                "point_count": 10,
                "state": item.state_at(last_detection_time).to_dict(),
            }
            for item in plan.obstacles
        }
        global_first_frame = {
            "schema": "dynamic_obstacle_raw_cloud_frame_v1",
            "source": "isaac_rtx_world_cloud_before_ros_filter",
            "proof_scope": "raw_cloud_visibility_only",
            "timestamp": first_detection_time,
            "completed_control_step": 425,
            "physics_step_index": 425,
            "elapsed_time_s": first_detection_time,
            "raw_point_count": 100,
            "total_obstacle_point_count": 8 * len(plan.obstacles),
            "obstacles": first_frame_obstacles,
        }
        global_last_frame = {
            "schema": "dynamic_obstacle_raw_cloud_frame_v1",
            "source": "isaac_rtx_world_cloud_before_ros_filter",
            "proof_scope": "raw_cloud_visibility_only",
            "timestamp": last_detection_time,
            "completed_control_step": 500,
            "physics_step_index": 500,
            "elapsed_time_s": last_detection_time,
            "raw_point_count": 120,
            "total_obstacle_point_count": 10 * len(plan.obstacles),
            "obstacles": last_frame_obstacles,
        }
    assert global_first_frame is not None
    assert global_last_frame is not None
    simulation["dynamic_obstacle_raw_cloud_lifecycle_report"] = {
        "schema": "dynamic_obstacle_raw_cloud_lifecycle_v1",
        "enabled": True,
        "source": "isaac_rtx_world_cloud_before_ros_filter",
        "proof_scope": "raw_cloud_visibility_only",
        "aabb_tolerance_m": 0.03,
        "sample_frame_count": 3,
        "frames_with_any_obstacle_points": 2,
        "frames_with_motion_started_obstacle_points": 2,
        "maximum_total_obstacle_point_count": 10,
        "first_detection": global_first_frame,
        "last_detection": global_last_frame,
        "all_configured_obstacles_observed": True,
        "obstacles": raw_cloud_obstacles,
    }
    simulation["dynamic_obstacle_raw_cloud_last_report"] = global_last_frame

    obstacle = plan.obstacles[0]
    hit_elapsed = (
        obstacle.start_delay_s + 0.5
        if replan_task
        else max(0.5, obstacle.start_delay_s - 1.0)
    )
    clear_elapsed = (
        obstacle.start_delay_s
        + obstacle.total_path_length_m / obstacle.speed_mps
        + 0.5
    )
    recovery_elapsed = last_time
    detour_elapsed = hit_elapsed + 0.5

    def retimed_identity(trajectory_id: int, elapsed_time_s: float) -> dict[str, object]:
        """让动态 fixture 的轨迹身份落在该任务真实运动时间窗内。"""

        identity = _controller_identity(trajectory_id)
        stamp_ns = int(round(elapsed_time_s * 1_000_000_000))
        identity["bspline_header_stamp"] = _stamp(stamp_ns)
        identity["bspline_header_stamp_ns"] = stamp_ns
        identity["start_time"] = _stamp(stamp_ns + 1)
        identity["start_time_ns"] = stamp_ns + 1
        return identity

    detour_identity = retimed_identity(1, detour_elapsed)
    recovery_identity = retimed_identity(2, recovery_elapsed)
    hit_state = obstacle.state_at(hit_elapsed)
    clear_state = obstacle.state_at(clear_elapsed)
    recovery_state = obstacle.state_at(recovery_elapsed)
    hit_point = [
        hit_state.position_world_xyz[0],
        hit_state.position_world_xyz[1],
        0.4,
    ]
    grid_hit = _grid_map_diagnostic_report(
        1,
        timestamp_s=hit_elapsed,
        hit_samples=[hit_point],
        clear_samples=[],
    )
    grid_clear = _grid_map_diagnostic_report(
        2,
        timestamp_s=clear_elapsed,
        hit_samples=[],
        clear_samples=[hit_point],
    )
    grid_clear[
        "occupied_to_free_transition_hit_observation_sequences"
    ] = [1]
    grid_clear["occupied_to_free_transition_hit_samples_world_xyz"] = [
        hit_point
    ]
    grid_clear["occupied_to_free_transition_hit_header_stamp_ns"] = [
        grid_hit["header"]["stamp_ns"]
    ]
    hit_match = {
        "obstacle_id": obstacle.obstacle_id,
        "point_world_xyz": hit_point,
        "voxel_index_xyz": grid_hit[
            "hit_endpoint_sample_voxel_indices_xyz"
        ][0],
        "map_resolution_m": 0.10,
        "point_to_obstacle_xy_clearance_m": 0.0,
        "association_tolerance_m": 0.05,
        "obstacle_state": hit_state.to_dict(),
    }
    clear_match = {
        "obstacle_id": obstacle.obstacle_id,
        "point_world_xyz": hit_point,
        "matched_hit_observation_sequence": 1,
        "matched_hit_header": grid_hit["header"],
        "matched_hit_point_world_xyz": hit_point,
        "voxel_index_xyz": grid_clear[
            "occupied_to_free_sample_voxel_indices_xyz"
        ][0],
        "map_resolution_m": 0.10,
        "match_distance_m": 0.0,
        "match_tolerance_m": 0.5 * math.sqrt(3.0) * 0.10 + 1.0e-9,
        "voxel_obstacle_separation_tolerance_m": (
            0.5 * math.sqrt(3.0) * 0.10 + 1.0e-9
        ),
        "matched_hit_provenance_verified": True,
        "matched_hit_point_to_obstacle_xy_clearance_m": 0.0,
        "obstacle_state_at_hit": hit_state.to_dict(),
        "obstacle_state_after_clear": clear_state.to_dict(),
        "sliding_reset_used": False,
    }
    grid_hit["dynamic_obstacle_hit_matches"] = [hit_match]
    grid_hit["dynamic_obstacle_transition_hit_matches"] = [hit_match]
    grid_clear["dynamic_obstacle_explicit_miss_clear_matches"] = [clear_match]
    simulation["grid_map_observation_diagnostics_last_report"] = grid_clear
    simulation["grid_map_observation_lifecycle_report"] = {
        "schema": "grid_map_observation_lifecycle_v1",
        "ros_time_offset_s": 0.0,
        "sample_count": 2,
        "first_observation_sequence": 1,
        "last_observation_sequence": 2,
        "sequence_reset_count": 0,
        "first_report": grid_hit,
        "last_report": grid_clear,
        "first_hit_report": grid_hit,
        "last_hit_report": grid_hit,
        "first_transition_hit_report": grid_hit,
        "last_transition_hit_report": grid_hit,
        "first_explicit_miss_clear_report": grid_clear,
        "last_explicit_miss_clear_report": grid_clear,
        "hit_reports": [grid_hit],
        "transition_hit_reports": [grid_hit],
        "diagnostic_reports": [grid_hit, grid_clear],
        "dropped_diagnostic_report_count": 0,
    }

    reference_samples = [
        [-4.28, 3.60, 0.4],
        [-4.10, 3.60, 0.4],
        [-4.08, 3.60, 0.4],
    ]
    detour_samples = [
        [-4.28, 3.52, 0.4],
        [-4.10, 3.52, 0.4],
        [-4.08, 3.52, 0.4],
    ]
    recovery_samples = [
        [-4.18, 3.5, 0.4],
        [-4.08, 3.5, 0.4],
        [-3.98, 3.5, 0.4],
    ]
    detour_report = _bspline_diagnostic_report(
        1,
        1,
        maximum_deviation=0.08,
        trajectory_samples=detour_samples,
        reference_samples=reference_samples,
    )
    recovery_report = _bspline_diagnostic_report(
        2,
        2,
        maximum_deviation=0.02,
        trajectory_samples=recovery_samples,
        reference_samples=reference_samples,
    )
    for report, identity, elapsed in (
        (detour_report, detour_identity, detour_elapsed),
        (recovery_report, recovery_identity, recovery_elapsed),
    ):
        stamp_ns = int(identity["bspline_header_stamp_ns"])
        report["identity"] = deepcopy(identity)
        report["header"] = {
            "frame_id": "world",
            "stamp": _stamp(stamp_ns),
            "stamp_ns": stamp_ns,
        }
        report["episode_elapsed_time_s"] = elapsed
        report["receipt_timestamp"] = elapsed + 0.01
    detour_clearance = {
        "obstacle_id": obstacle.obstacle_id,
        "obstacle_state": obstacle.state_at(detour_elapsed).to_dict(),
        "minimum_trajectory_center_to_obstacle_xy_m": 0.505,
        "trajectory_sample_interval_s": 0.01,
        "maximum_velocity_upper_bound_mps": 1.0,
        "sampling_clearance_margin_m": 0.005,
        "continuous_clearance_lower_bound_m": 0.5,
        "continuous_clearance_verified": True,
        "minimum_ordered_reference_center_to_obstacle_xy_m": 0.425,
        "reference_obstructed": True,
        "reference_blocked_then_trajectory_clear": True,
        "required_clearance_m": 0.43,
        "clearance_verified": True,
        "relevant": True,
        "relevance_distance_m": 2.0,
    }
    detour_report["dynamic_obstacle_clearances"] = [detour_clearance]
    detour_report["dynamic_obstacle_reference_obstructed"] = True
    recovery_report["dynamic_obstacle_clearances"] = [
        {
            "obstacle_id": obstacle.obstacle_id,
            "obstacle_state": recovery_state.to_dict(),
            "minimum_trajectory_center_to_obstacle_xy_m": 0.5636044712384742,
            "trajectory_sample_interval_s": 0.01,
            "maximum_velocity_upper_bound_mps": 1.0,
            "sampling_clearance_margin_m": 0.005,
            "continuous_clearance_lower_bound_m": 0.5586044712384742,
            "continuous_clearance_verified": True,
            "minimum_ordered_reference_center_to_obstacle_xy_m": (
                0.5231156659860228
            ),
            "reference_obstructed": False,
            "reference_blocked_then_trajectory_clear": False,
            "required_clearance_m": 0.43,
            "clearance_verified": True,
            "relevant": True,
            "relevance_distance_m": 2.0,
        }
    ]
    recovery_report["dynamic_obstacle_reference_obstructed"] = False
    simulation["bspline_diagnostics_last_report"] = recovery_report
    simulation["bspline_diagnostics_lifecycle_report"] = {
        "schema": "bspline_diagnostics_lifecycle_v1",
        "ros_time_offset_s": 0.0,
        "sample_count": 2,
        "first_diagnostic_sequence": 1,
        "last_diagnostic_sequence": 2,
        "sequence_reset_count": 0,
        "distinct_trajectory_identity_count": 2,
        "trajectory_identities": [
            deepcopy(detour_identity),
            deepcopy(recovery_identity),
        ],
        "first_report": detour_report,
        "last_report": recovery_report,
        "diagnostic_reports": [detour_report, recovery_report],
        "dropped_diagnostic_report_count": 0,
    }
    controller_lifecycle = simulation["scan_controller_status_lifecycle_report"]
    assert isinstance(controller_lifecycle, dict)
    controller_statuses: dict[int, dict[str, object]] = {}
    for collection_name in ("accepted_status_reports", "tracking_status_reports"):
        collection = controller_lifecycle[collection_name]
        assert isinstance(collection, list)
        for status in collection:
            assert isinstance(status, dict)
            status_identity = status.get("identity")
            assert isinstance(status_identity, dict)
            trajectory_id = int(status_identity["traj_id"])
            elapsed = detour_elapsed if trajectory_id == 1 else recovery_elapsed
            identity = detour_identity if trajectory_id == 1 else recovery_identity
            status_stamp_ns = int(round((elapsed + 0.02) * 1_000_000_000))
            status["identity"] = deepcopy(identity)
            status["receipt_timestamp"] = elapsed + 0.02
            status["header"] = {
                "frame_id": "world",
                "stamp": _stamp(status_stamp_ns),
                "stamp_ns": status_stamp_ns,
            }
            controller_statuses.setdefault(trajectory_id, status)
    controller_lifecycle["accepted_trajectory_identities"] = [
        deepcopy(detour_identity),
        deepcopy(recovery_identity),
    ]
    controller_detour_status = controller_lifecycle[
        "accepted_status_reports"
    ][0]
    controller_detour_tracking_status = controller_lifecycle[
        "tracking_status_reports"
    ][0]
    controller_tracking_status = controller_lifecycle["last_tracking_status"]
    policy_lifecycle = simulation["navigation_policy_gate_lifecycle_report"]
    assert isinstance(policy_lifecycle, dict)
    tracking_write_reports = policy_lifecycle[
        "identity_verified_tracking_write_reports"
    ]
    assert isinstance(tracking_write_reports, list)
    for index, write in enumerate(tracking_write_reports):
        assert isinstance(write, dict)
        trajectory_id = 1 if index == 0 else 2
        elapsed = detour_elapsed if trajectory_id == 1 else recovery_elapsed
        write["timestamp"] = elapsed + 0.10 + 0.05 * (index % 2)
        write["scan_controller_status_snapshot"] = deepcopy(
            controller_statuses[trajectory_id]
        )
    policy_tracking_write = tracking_write_reports[-1]
    detour_policy_tracking_write = policy_lifecycle[
        "identity_verified_tracking_write_reports"
    ][0]
    policy_tracking_write["scan_controller_status_snapshot"] = (  # type: ignore[index]
        controller_tracking_status
    )
    simulation["dynamic_navigation_evidence_report"] = {
        "schema": "dynamic_navigation_evidence_v1",
        "ros_time_offset_s": 0.0,
        "enabled": True,
        "verified": True,
        "obstacle_ids": [obstacle.obstacle_id],
        "post_filter_hit": {
            "verified": True,
            "source": "ros2_scan_grid_map_observation_diagnostics",
            "topic": "/planning/grid_map_observation_diagnostics",
            "header": grid_hit["header"],
            "observation_sequence": 1,
            "hit_endpoint_count": 1,
            "hit_endpoint_samples_world_xyz": [hit_point],
            "dynamic_obstacle_hit_matches": [hit_match],
        },
        "ordered_detour": {
            "verified": True,
            "source": "ros2_scan_bspline_diagnostics",
            "topic": "/planning/bspline_diagnostics",
            "header": detour_report["header"],
            "diagnostic_sequence": 1,
            "identity": deepcopy(detour_identity),
            "maximum_trajectory_deviation_m": 0.08,
            "maximum_deviation_limit_m": 0.10,
            "maximum_guide_progress_lead_m": 0.01,
            "maximum_progress_lead_limit_m": 0.02,
            "trajectory_samples_world_xyz": detour_samples,
            "ordered_reference_samples_world_xyz": reference_samples,
            "dynamic_obstacle_clearances": [detour_clearance],
            "dynamic_obstacle_reference_obstructed": True,
            "controller_identity_accepted": True,
            "controller_accepted_status": controller_detour_status,
            "controller_tracking_status": controller_detour_tracking_status,
            "policy_identity_valid_tracking": True,
            "policy_identity_verified_tracking_write": (
                detour_policy_tracking_write
            ),
            "causal_map_transition_clear_match": clear_match,
        },
        "current_obstacle_clearance": {
            "verified": True,
            "source": "ros2_scan_bspline_diagnostics",
            "topic": "/planning/bspline_diagnostics",
            "header": detour_report["header"],
            "diagnostic_sequence": 1,
            "identity": deepcopy(detour_identity),
            "required_clearance_m": 0.43,
            "obstacle_clearances": [detour_clearance],
            "reason": "all_relevant_obstacles_clear",
        },
        "explicit_miss_ghost_clear": {
            "verified": True,
            "source": "ros2_scan_grid_map_observation_diagnostics",
            "topic": "/planning/grid_map_observation_diagnostics",
            "header": grid_clear["header"],
            "observation_sequence": 2,
            "matched_hit_observation_sequence": 1,
            "explicit_free_miss_voxel_count": 3,
            "occupied_to_free_by_explicit_miss_count": 1,
            "occupied_removed_by_sliding_reset_count": 0,
            "clear_matches": [clear_match],
        },
        "trajectory_recovery": {
            "verified": True,
            "source": "ros2_scan_bspline_diagnostics",
            "topic": "/planning/bspline_diagnostics",
            "before_diagnostic_sequence": 1,
            "before_header": detour_report["header"],
            "before_detour_identity": deepcopy(detour_identity),
            "after_recovery_identity": deepcopy(recovery_identity),
            "after_diagnostic_sequence": 2,
            "after_header": recovery_report["header"],
            "before_maximum_trajectory_deviation_m": 0.08,
            "after_maximum_trajectory_deviation_m": 0.02,
            "recovery_maximum_deviation_m": 0.02,
            "recovery_minimum_improvement_m": 0.01,
            "controller_tracking_status": controller_tracking_status,
            "controller_acceptance_sequence": 2,
            "controller_status_sequence": 2,
            "policy_identity_valid_tracking": True,
            "controller_accepted_status": controller_lifecycle[
                "accepted_status_reports"
            ][1],
            "policy_identity_verified_tracking_write": policy_tracking_write,
            "same_reference_path_generation": True,
        },
    }


def _add_active_sensing_report(summary: dict[str, object]) -> None:
    """为 dynamic_f1 fixture 加入一条完整、同 Path 的主动观测证据。"""

    simulation = summary["simulation_report"]
    assert isinstance(simulation, dict)
    bspline_lifecycle = simulation["bspline_diagnostics_lifecycle_report"]
    assert isinstance(bspline_lifecycle, dict)
    ordinary_reports = bspline_lifecycle["diagnostic_reports"]
    assert isinstance(ordinary_reports, list) and ordinary_reports
    template = ordinary_reports[0]
    assert isinstance(template, dict)

    active_header_ns = 47_500_000_000
    active_identity = {
        "reference_path_stamp": _stamp(PATH_STAMP_NS),
        "reference_path_stamp_ns": PATH_STAMP_NS,
        "bspline_header_stamp": _stamp(active_header_ns),
        "bspline_header_stamp_ns": active_header_ns,
        "start_time": _stamp(active_header_ns),
        "start_time_ns": active_header_ns,
        "traj_id": 99,
    }
    settle_stamp_ns = 48_650_000_000
    base_active = {
        "enabled": True,
        "event": 0,
        "start_yaw": 0.10,
        "target_yaw": 0.30,
        "yaw_offset": 0.20,
        "yaw_rate": 0.20,
        "settle_stamp": _stamp(0),
        "settle_stamp_ns": 0,
        "settle_yaw_error": 0.0,
        "settle_angular_speed": 0.0,
        "stable_duration": 0.0,
        "fusion_baseline": 0,
        "fusion_current": 0,
        "fusion_distinct": 0,
        "fusion_required": 3,
        "completed": False,
        "failed": False,
        "reason": "主动观测生命周期",
    }
    event_specs = (
        ("STARTED", 1, 0, 0, 0, 47.51),
        ("ACCEPTED", 2, 0, 0, 0, 47.56),
        ("YAW_STABLE", 3, 2, 2, 0, 48.66),
        ("FUSION_PROGRESS", 4, 2, 3, 1, 49.06),
        ("FUSION_PROGRESS", 4, 2, 5, 3, 49.16),
        ("COMPLETED", 5, 2, 5, 3, 49.20),
    )
    event_reports: list[dict[str, object]] = []
    for index, (
        event_name,
        event_code,
        baseline,
        current,
        distinct,
        receipt_timestamp,
    ) in enumerate(
        event_specs,
        start=1,
    ):
        report = deepcopy(template)
        report.update(
            {
                "receipt_timestamp": receipt_timestamp,
                "rx_sequence": 20 + index,
                "header": {
                    "frame_id": "world",
                    "stamp": _stamp(active_header_ns),
                    "stamp_ns": active_header_ns,
                },
                "episode_elapsed_time_s": 47.5,
                "diagnostic_sequence": 1 + index,
                "identity": deepcopy(active_identity),
                "is_final": False,
                "emergency_stop": False,
                "stationary": True,
                "ordered_reference_checked": False,
                "ordered_reference_safe": False,
                "maximum_trajectory_deviation_m": 0.0,
                "maximum_guide_anchor_deviation_m": 0.0,
                "maximum_guide_progress_lead_m": 0.0,
                "maximum_velocity_upper_bound_mps": 0.0,
                "trajectory_duration_s": 2.75,
                "trajectory_sample_interval_s": 2.75 / 63.0,
                "trajectory_samples_world_xyz": [
                    [-4.15, 3.50, 0.4] for _ in range(64)
                ],
                "trajectory_sample_count_total": 276,
                "trajectory_samples_truncated": True,
                "sampling_clearance_margin_m": 0.0,
                "ordered_reference_samples_world_xyz": [],
                "ordered_reference_sample_count_total": 0,
                "dynamic_obstacle_clearances": [],
                "dynamic_obstacle_relevant": False,
                "dynamic_obstacle_reference_obstructed": False,
                "ordered_detour_candidate": False,
            }
        )
        active = deepcopy(base_active)
        active.update(
            {
                "event": event_code,
                "fusion_baseline": baseline,
                "fusion_current": current,
                "fusion_distinct": distinct,
                "completed": event_name == "COMPLETED",
                "reason": event_name,
            }
        )
        if event_name in {"YAW_STABLE", "FUSION_PROGRESS", "COMPLETED"}:
            active.update(
                {
                    "settle_stamp": _stamp(settle_stamp_ns),
                    "settle_stamp_ns": settle_stamp_ns,
                    "settle_yaw_error": 0.01,
                    "settle_angular_speed": 0.04,
                    "stable_duration": 0.11,
                }
            )
        report["active_sensing"] = active
        event_reports.append(report)

    controller_first = _controller_status(
        2,
        99,
        state=9,
        event=1,
        reason="主动观测已接受",
    )
    controller_last = _controller_status(
        3,
        99,
        state=9,
        event=4,
        reason="主动观测原地旋转",
    )
    first_controller_command = [0.0] * 6
    first_controller_aggregate = {
        "sample_count": 1,
        "first_command": first_controller_command,
        "max_abs_vx": 0.0,
        "max_abs_vy": 0.0,
        "max_abs_wz": 0.0,
        "violation_count": 0,
    }
    final_controller_aggregate = {
        "sample_count": 3,
        "first_command": first_controller_command,
        "max_abs_vx": 0.0,
        "max_abs_vy": 0.0,
        "max_abs_wz": 0.20,
        "violation_count": 0,
    }
    for status, aggregate in (
        (controller_first, first_controller_aggregate),
        (controller_last, final_controller_aggregate),
    ):
        status["identity"] = deepcopy(active_identity)
        status["active_sensing_yaw_only"] = True
        status["command_aggregate"] = deepcopy(aggregate)
    controller_first["receipt_timestamp"] = 47.55
    controller_first["header"] = {
        "frame_id": "world",
        "stamp": _stamp(47_550_000_000),
        "stamp_ns": 47_550_000_000,
    }
    controller_last["receipt_timestamp"] = 48.40
    controller_last["header"] = {
        "frame_id": "world",
        "stamp": _stamp(48_400_000_000),
        "stamp_ns": 48_400_000_000,
    }

    first_write = _consumed_tracking_evidence(1, 1, write_sequence=3)
    rotation_write = _consumed_tracking_evidence(1, 1, write_sequence=4)
    last_write = _consumed_tracking_evidence(1, 1, write_sequence=5)
    first_write["timestamp"] = 47.57
    rotation_write["timestamp"] = 47.58
    last_write["timestamp"] = 49.19
    first_write["written_command"] = [0.0, 0.0, 0.0]
    rotation_write["written_command"] = [0.0, 0.0, 0.20]
    last_write["written_command"] = [0.0, 0.0, 0.0]
    first_write.update(
        {
            "owner_id": "scan_cmd_vel",
            "requested_command": [0.0, 0.0, 0.10],
            "limited_target": [0.0, 0.0, 0.0],
            "motion_allowed": False,
            "stop_reasons": ["active_sensing_identity_zero_gate"],
            "clipped_axes": [],
            "rate_limited_axes": [],
            "navigation_emergency_stop_latched": False,
            "navigation_cmd_vel_inhibited": False,
            "navigation_cmd_vel_inhibit_reason": None,
            "cmd_vel_sample_received_this_tick": False,
            "cmd_vel_sample_drained_this_tick": True,
            "cmd_vel_source_sequence": 10,
            "cmd_vel_source_receipt_timestamp": 47.56,
            "last_cmd_vel_drain_sequence": 10,
            "last_cmd_vel_drain_receipt_timestamp": 47.56,
        }
    )
    first_write["navigation_gate_diagnostics"]["command_identity"] = None
    first_write["navigation_gate_diagnostics"][
        "command_identity_matches_permit"
    ] = False
    rotation_write.update(
        {
            "owner_id": "scan_cmd_vel",
            "requested_command": [0.0, 0.0, 0.20],
            "limited_target": [0.0, 0.0, 0.20],
            "motion_allowed": True,
            "stop_reasons": [],
            "clipped_axes": [],
            "rate_limited_axes": [],
            "navigation_emergency_stop_latched": False,
            "navigation_cmd_vel_inhibited": False,
            "navigation_cmd_vel_inhibit_reason": None,
            "cmd_vel_sample_received_this_tick": True,
            "cmd_vel_sample_drained_this_tick": False,
            "cmd_vel_source_sequence": 11,
            "cmd_vel_source_receipt_timestamp": 47.57,
            "last_cmd_vel_drain_sequence": 10,
            "last_cmd_vel_drain_receipt_timestamp": 47.56,
        }
    )
    last_write.update(
        {
            "owner_id": "scan_cmd_vel",
            "requested_command": [0.0, 0.0, 0.10],
            "limited_target": [0.0, 0.0, 0.0],
            "motion_allowed": False,
            "stop_reasons": ["cmd_vel_timeout"],
            "clipped_axes": [],
            "rate_limited_axes": [],
            "navigation_emergency_stop_latched": False,
            "navigation_emergency_stop_reason": None,
            "navigation_cmd_vel_inhibited": False,
            "navigation_cmd_vel_inhibit_reason": None,
            "cmd_vel_sample_received_this_tick": False,
            "cmd_vel_sample_drained_this_tick": False,
            "cmd_vel_source_sequence": 11,
            "cmd_vel_source_receipt_timestamp": 48.49,
            "last_cmd_vel_drain_sequence": 10,
            "last_cmd_vel_drain_receipt_timestamp": 47.56,
        }
    )
    for write, snapshot in (
        (first_write, controller_first),
        (rotation_write, controller_first),
        (last_write, controller_last),
    ):
        write["policy_navigation_gate_consumed_report"] = deepcopy(
            write["navigation_gate_diagnostics"]
        )
        write["scan_controller_status_snapshot"] = deepcopy(snapshot)
        observed = _observed_status_evidence(
            1,
            1,
            tracking=True,
            write_sequence=int(write["write_sequence"]),
        )
        write["navigation_status_observed_report"] = deepcopy(
            observed["navigation_status_observed_report"]
        )

    policy_lifecycle = simulation["navigation_policy_gate_lifecycle_report"]
    assert isinstance(policy_lifecycle, dict)
    tracking_write_reports = policy_lifecycle[
        "identity_verified_tracking_write_reports"
    ]
    assert isinstance(tracking_write_reports, list)
    active_tracking_write = {
        "write_sequence": rotation_write["write_sequence"],
        "timestamp": rotation_write["timestamp"],
        "written_command": deepcopy(rotation_write["written_command"]),
        "navigation_gate_diagnostics": deepcopy(
            rotation_write["navigation_gate_diagnostics"]
        ),
        "scan_controller_status_snapshot": deepcopy(
            rotation_write["scan_controller_status_snapshot"]
        ),
    }
    policy_stop_reason_counts = policy_lifecycle["stop_reason_counts"]
    assert isinstance(policy_stop_reason_counts, dict)
    policy_stop_reason_counts["active_sensing_identity_zero_gate"] = 1
    policy_lifecycle["identity_verified_tracking_snapshot_count"] = 3
    tracking_write_reports.insert(1, active_tracking_write)

    dynamic = simulation["dynamic_navigation_evidence_report"]
    assert isinstance(dynamic, dict)
    trajectory_recovery = dynamic["trajectory_recovery"]
    assert isinstance(trajectory_recovery, dict)
    recovery_identity = deepcopy(trajectory_recovery["after_recovery_identity"])
    recovery_report = ordinary_reports[-1]
    assert isinstance(recovery_report, dict)
    recovery_report["diagnostic_sequence"] = 8
    bspline_lifecycle["last_diagnostic_sequence"] = 8
    bspline_lifecycle["active_sensing_diagnostic_count"] = len(event_reports)
    trajectory_recovery["after_diagnostic_sequence"] = 8

    grid_lifecycle = simulation["grid_map_observation_lifecycle_report"]
    assert isinstance(grid_lifecycle, dict)
    grid_reports = grid_lifecycle["diagnostic_reports"]
    assert isinstance(grid_reports, list)
    fused_reports = [
        _grid_map_diagnostic_report(
            sequence,
            timestamp_s=49.05 + 0.05 * (sequence - 3),
            hit_samples=[[-4.15 + 0.01 * sequence, 3.50, 0.4]],
            clear_samples=[],
        )
        for sequence in range(3, 6)
    ]
    grid_reports.extend(fused_reports)
    grid_lifecycle["sample_count"] = 5
    grid_lifecycle["last_observation_sequence"] = 5
    grid_lifecycle["last_report"] = fused_reports[-1]
    grid_lifecycle["last_hit_report"] = fused_reports[-1]
    grid_lifecycle["last_transition_hit_report"] = fused_reports[-1]
    grid_lifecycle["hit_reports"].extend(fused_reports)  # type: ignore[index]
    grid_lifecycle["transition_hit_reports"].extend(fused_reports)  # type: ignore[index]
    simulation["grid_map_observation_diagnostics_last_report"] = fused_reports[-1]
    controller_lifecycle = simulation["scan_controller_status_lifecycle_report"]
    assert isinstance(controller_lifecycle, dict)
    controller_lifecycle["active_sensing_status_count"] = 2

    def retime_recovery_controller_status(value: object) -> None:
        """把普通恢复与到达状态排在 active status 之后。"""

        if isinstance(value, dict):
            identity = value.get("identity")
            if (
                value.get("source")
                == "ros2_scan_planner_msgs_controller_status"
                and isinstance(identity, dict)
                and identity.get("traj_id") == 2
            ):
                state = value.get("state")
                if state == 10:
                    sequence, stamp_ns, receipt = 4, 49_550_000_000, 49.55
                elif state == 12:
                    sequence, stamp_ns, receipt = 5, 50_000_000_000, 50.0
                else:
                    sequence = stamp_ns = receipt = None
                if sequence is not None:
                    value["status_sequence"] = sequence
                    value["rx_sequence"] = sequence
                    value["receipt_timestamp"] = receipt
                    value["header"] = {
                        "frame_id": "world",
                        "stamp": _stamp(stamp_ns),
                        "stamp_ns": stamp_ns,
                    }
            for child in value.values():
                retime_recovery_controller_status(child)
        elif isinstance(value, list):
            for child in value:
                retime_recovery_controller_status(child)

    retime_recovery_controller_status(simulation)
    controller_lifecycle["last_status_sequence"] = 5
    trajectory_recovery["controller_status_sequence"] = 4
    simulation["active_sensing_lifecycle_report"] = {
        "schema": "active_sensing_lifecycle_v1",
        "attempt_count": 1,
        "completed_attempt_count": 1,
        "failed_attempt_count": 0,
        "active_attempt_identity": None,
        "pending_active_controller_statuses": [],
        "pending_recovery_controller_statuses": [],
        "pending_active_policy_writes": [],
        "policy_zero_gate": None,
        "policy_zero_gate_armed_count": 1,
        "policy_zero_gate_consumed_count": 1,
        "policy_zero_gated_identities": [deepcopy(active_identity)],
        "attempts": [
            {
                "identity": active_identity,
                "events": [item[0] for item in event_specs],
                "event_reports": event_reports,
                "started": deepcopy(event_reports[0]),
                "accepted": deepcopy(event_reports[1]),
                "yaw_stable": deepcopy(event_reports[2]),
                "completed": deepcopy(event_reports[-1]),
                "failed": None,
                "planner_fusion": {
                    "baseline": 2,
                    "current": 5,
                    "distinct": 3,
                    "required": 3,
                },
                "post_settle_fused_observations": [
                    {
                        "header_stamp_ns": int(report["header"]["stamp_ns"]),
                        "map_fusion_performed": True,
                        "accepted_endpoint_count": int(
                            report["accepted_endpoint_count"]
                        ),
                        "observation_sequence": int(
                            report["observation_sequence"]
                        ),
                        "header": deepcopy(report["header"]),
                    }
                    for report in fused_reports
                ],
                "controller_command_aggregate": {
                    **final_controller_aggregate,
                    "first_status": controller_first,
                    "last_status": controller_last,
                },
                "policy_command_aggregate": {
                    "sample_count": 3,
                    "first_command": [0.0, 0.0, 0.0],
                    "max_abs_vx": 0.0,
                    "max_abs_vy": 0.0,
                    "max_abs_wz": 0.20,
                    "violation_count": 0,
                    "first_write": first_write,
                    "first_rotation_write": rotation_write,
                    "maximum_abs_wz_write": deepcopy(rotation_write),
                    "last_write": last_write,
                },
                "pct_plan_ids": [1],
                "recovery": {
                    "identity": recovery_identity,
                    "reference_path_stamp_ns": PATH_STAMP_NS,
                    "pct_plan_id": 1,
                    "stationary": False,
                    "controller_state": 10,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "mode",
    [
        "static_stair",
        "flat_policy",
        "crossfloor_carry",
        "dynamic_f1",
        "dynamic_replan_f1",
    ],
)
def test_valid_contracts(mode: str) -> None:
    report = validate_pct_scan_live_summary(_base_summary(mode), mode)  # type: ignore[arg-type]
    assert report["valid"] is True, report["errors"]
    assert report["errors"] == []
    if mode == "dynamic_f1":
        assert report["not_validated_claims"] == []
        assert (
            "controller_tracking_recovery_identity_observed_before_policy_write"
            in report["validated_claims"]
        )
    if mode == "dynamic_replan_f1":
        assert report["not_validated_claims"] == []
        assert (
            "pct_replan_request_inflight_and_new_plan_identity_verified"
            in report["validated_claims"]
        )


def test_dynamic_active_sensing_contract_is_strictly_accepted() -> None:
    summary = _base_summary("dynamic_f1")
    _add_active_sensing_report(summary)

    report = validate_pct_scan_live_summary(
        summary,
        "dynamic_f1",
        require_active_sensing=True,
    )

    assert report["valid"] is True, report["errors"]
    assert report["require_active_sensing"] is True
    assert (
        "scan_active_sensing_typed_lifecycle_completed"
        in report["validated_claims"]
    )


def test_active_zero_gate_accepts_missing_old_permit_and_observed_status() -> None:
    """首拍零门先于 supervisor topic 时，不建立虚假的跨 topic join。"""

    summary = _base_summary("dynamic_f1")
    _add_active_sensing_report(summary)
    first_write = summary["simulation_report"][  # type: ignore[index]
        "active_sensing_lifecycle_report"
    ]["attempts"][0]["policy_command_aggregate"]["first_write"]
    gate = first_write["navigation_gate_diagnostics"]
    gate.update(
        {
            "status_fault": "navigation_status_missing",
            "permit_received": False,
            "permit": None,
        }
    )
    first_write["policy_navigation_gate_consumed_report"] = deepcopy(gate)
    first_write["navigation_status_observed_report"] = None

    report = validate_pct_scan_live_summary(
        summary,
        "dynamic_f1",
        require_active_sensing=True,
    )

    assert report["valid"] is True, report["errors"]


def test_tracking_ring_counts_distinct_controller_snapshots_not_writes() -> None:
    """同一 typed ControllerStatus 下多拍实写只占一个 pinned ring 槽。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    first = lifecycle["identity_verified_tracking_write_reports"][0]
    last = _consumed_tracking_evidence(1, 1, write_sequence=8)
    last["scan_controller_status_snapshot"] = deepcopy(
        first["scan_controller_status_snapshot"]
    )
    lifecycle["identity_verified_tracking_snapshot_count"] = 1
    lifecycle["identity_verified_tracking_write_reports"] = [first]
    lifecycle["last_identity_verified_tracking_write"] = last

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is True, report["errors"]


def test_tracking_ring_rejects_duplicate_controller_snapshot() -> None:
    """不同 write sequence 不得重复占用同一 typed ControllerStatus 槽。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    duplicate = _consumed_tracking_evidence(1, 1, write_sequence=4)
    lifecycle["identity_verified_tracking_snapshot_count"] = 3
    lifecycle["identity_verified_tracking_write_reports"].insert(1, duplicate)

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is False
    assert any(
        issue["code"] == "duplicate_identity" for issue in report["errors"]
    ), report["errors"]


def test_flat_policy_rejects_recovered_trajectory_timeout() -> None:
    """平地最终虽到达，也不能掩盖中途 trajectory timeout。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "scan_controller_status_lifecycle_report"
    ]
    state_counts = lifecycle["state_counts"]
    state_counts["tracking"] = 1
    state_counts["trajectory_timeout"] = 1

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is False
    assert any(
        issue["code"] == "flat_trajectory_timeout"
        for issue in report["errors"]
    ), report["errors"]


def test_flat_policy_allows_goal_latched_same_path_candidate_rejection() -> None:
    """到达后拒绝同一 Path 的迟到候选是完成锁存证据，不是运行失败。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "scan_controller_status_lifecycle_report"
    ]
    lifecycle["candidate_rejection_count"] = 1
    lifecycle["goal_latched_same_path_candidate_rejection_count"] = 1
    lifecycle["unexpected_candidate_rejection_count"] = 0

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is True, report["errors"]


def test_flat_policy_rejects_runtime_candidate_rejection() -> None:
    """运行期任何候选拒绝仍使稳定平地基线失败。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "scan_controller_status_lifecycle_report"
    ]
    lifecycle["candidate_rejection_count"] = 1
    lifecycle["goal_latched_same_path_candidate_rejection_count"] = 0
    lifecycle["unexpected_candidate_rejection_count"] = 1

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is False
    assert any(
        issue["code"] == "flat_controller_rejection"
        for issue in report["errors"]
    ), report["errors"]


def test_flat_policy_rejects_recovered_scan_planning_failure() -> None:
    """平地最终虽到达，也不能掩盖中途 SCAN 规划失败。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["maximum_consecutive_scan_failures"] = 1

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is False
    assert any(
        issue["code"] == "flat_scan_planning_failure"
        for issue in report["errors"]
    ), report["errors"]


def test_flat_policy_allows_only_stair_resume_waiting_emergency_state() -> None:
    """组合 launch 的启动等待状态不是运行中急停。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["observed_state_counts"] = {
        "emergency_stop": 1,
        "tracking": 1,
        "goal_reached": 1,
    }
    lifecycle["observed_reason_counts"] = {
        "scan_stair_resume_waiting": 1,
        "TRACKING": 1,
        "GOAL_REACHED": 1,
    }
    lifecycle["emergency_stop_observed_status_count"] = 1

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is True, report["errors"]


def test_flat_policy_rejects_nonstartup_supervisor_emergency() -> None:
    """非启动等待原因的 supervisor emergency 不能被最终到达掩盖。"""

    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["observed_state_counts"] = {
        "emergency_stop": 1,
        "tracking": 1,
        "goal_reached": 1,
    }
    lifecycle["observed_reason_counts"] = {
        "scan_planning_failed": 1,
        "TRACKING": 1,
        "GOAL_REACHED": 1,
    }
    lifecycle["emergency_stop_observed_status_count"] = 1

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert report["valid"] is False
    assert any(
        issue["code"] == "flat_supervisor_emergency_stop"
        for issue in report["errors"]
    ), report["errors"]


@pytest.mark.parametrize(
    "corruption",
    [
        "event_order",
        "pre_settle_fusion",
        "empty_fusion",
        "controller_first_nonzero",
        "controller_yaw_over_cap",
        "controller_raw_violation",
        "policy_first_nonzero",
        "policy_translation",
        "policy_last_write_translation",
        "pct_plan_changed",
        "recovery_reuses_active_identity",
        "recovery_not_tracking",
        "active_counted_as_motion",
        "grid_report_not_joined",
        "controller_sample_count_regressed",
        "active_global_count_missing",
        "trajectory_too_short",
        "stationary_geometry_moves",
        "failed_attempt",
    ],
)
def test_dynamic_active_sensing_contract_fails_closed(
    corruption: str,
) -> None:
    summary = _base_summary("dynamic_f1")
    _add_active_sensing_report(summary)
    simulation = summary["simulation_report"]
    assert isinstance(simulation, dict)
    lifecycle = simulation["active_sensing_lifecycle_report"]
    assert isinstance(lifecycle, dict)
    attempts = lifecycle["attempts"]
    assert isinstance(attempts, list) and len(attempts) == 1
    attempt = attempts[0]
    assert isinstance(attempt, dict)
    if corruption == "event_order":
        attempt["events"][1:3] = ["YAW_STABLE", "ACCEPTED"]  # type: ignore[index]
    elif corruption == "pre_settle_fusion":
        settle = attempt["yaw_stable"]["active_sensing"]["settle_stamp_ns"]  # type: ignore[index]
        attempt["post_settle_fused_observations"][0][  # type: ignore[index]
            "header_stamp_ns"
        ] = settle
    elif corruption == "empty_fusion":
        attempt["post_settle_fused_observations"][0][  # type: ignore[index]
            "accepted_endpoint_count"
        ] = 0
    elif corruption == "controller_first_nonzero":
        attempt["controller_command_aggregate"]["first_command"][0] = 0.01  # type: ignore[index]
    elif corruption == "controller_yaw_over_cap":
        attempt["controller_command_aggregate"]["max_abs_wz"] = 0.201  # type: ignore[index]
    elif corruption == "controller_raw_violation":
        attempt["controller_command_aggregate"]["violation_count"] = 1  # type: ignore[index]
    elif corruption == "policy_first_nonzero":
        attempt["policy_command_aggregate"]["first_command"][2] = 0.01  # type: ignore[index]
    elif corruption == "policy_translation":
        attempt["policy_command_aggregate"]["max_abs_vx"] = 0.01  # type: ignore[index]
    elif corruption == "policy_last_write_translation":
        attempt["policy_command_aggregate"]["last_write"][  # type: ignore[index]
            "written_command"
        ] = [0.5, 0.0, 0.0]
    elif corruption == "pct_plan_changed":
        attempt["pct_plan_ids"] = [1, 2]
    elif corruption == "recovery_reuses_active_identity":
        attempt["recovery"]["identity"] = deepcopy(attempt["identity"])  # type: ignore[index]
    elif corruption == "recovery_not_tracking":
        attempt["recovery"]["controller_state"] = 9  # type: ignore[index]
    elif corruption == "active_counted_as_motion":
        simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
            "trajectory_identities"
        ].append(deepcopy(attempt["identity"]))
    elif corruption == "grid_report_not_joined":
        attempt["post_settle_fused_observations"][0][  # type: ignore[index]
            "observation_sequence"
        ] = 99
    elif corruption == "controller_sample_count_regressed":
        attempt["controller_command_aggregate"]["first_status"][  # type: ignore[index]
            "command_aggregate"
        ]["sample_count"] = 99
    elif corruption == "active_global_count_missing":
        simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
            "active_sensing_diagnostic_count"
        ] = 0
    elif corruption == "trajectory_too_short":
        for report in attempt["event_reports"]:  # type: ignore[index]
            report["trajectory_duration_s"] = 0.5
            report["trajectory_sample_count_total"] = 51
            report["trajectory_samples_world_xyz"] = [  # type: ignore[index]
                [-4.15, 3.50, 0.4] for _ in range(51)
            ]
            report["trajectory_samples_truncated"] = False
            report["trajectory_sample_interval_s"] = 0.01
        for key in ("started", "accepted", "yaw_stable", "completed"):
            attempt[key] = deepcopy(
                next(
                    report
                    for report, event in zip(
                        attempt["event_reports"],  # type: ignore[index]
                        attempt["events"],  # type: ignore[index]
                        strict=True,
                    )
                    if event
                    == {
                        "started": "STARTED",
                        "accepted": "ACCEPTED",
                        "yaw_stable": "YAW_STABLE",
                        "completed": "COMPLETED",
                    }[key]
                )
            )
    elif corruption == "stationary_geometry_moves":
        attempt["event_reports"][0]["trajectory_samples_world_xyz"][1][  # type: ignore[index]
            0
        ] += 0.01
        attempt["started"] = deepcopy(attempt["event_reports"][0])  # type: ignore[index]
    elif corruption == "failed_attempt":
        lifecycle["failed_attempt_count"] = 1
    else:  # pragma: no cover - 参数列表与分支必须同步。
        raise AssertionError(corruption)

    report = validate_pct_scan_live_summary(
        summary,
        "dynamic_f1",
        require_active_sensing=True,
    )

    assert report["valid"] is False
    assert report["errors"]


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("event_receipt_regressed", "invalid_active_sensing_event_order"),
        ("event_rx_regressed", "invalid_active_sensing_event_order"),
        ("event_time_offset_forged", "inconsistent_episode_time_offset"),
        ("settle_snapshot_changed", "inconsistent_active_sensing_snapshot"),
        ("fusion_progress_stalled", "invalid_active_sensing_fusion_sequence"),
        (
            "completion_advances_fusion",
            "inconsistent_active_sensing_fusion_evidence",
        ),
        ("controller_final", "invalid_active_sensing_controller_state"),
        (
            "controller_first_envelope_nonzero",
            "invalid_active_sensing_first_controller_sample",
        ),
        (
            "controller_acceptance_changed",
            "invalid_active_sensing_controller_sequence",
        ),
        ("policy_gate_schema_forged", "unexpected_value"),
        ("observed_status_schema_forged", "unexpected_value"),
        ("observed_status_idle", "invalid_status_contract"),
        (
            "policy_snapshot_outside_active_window",
            "invalid_active_sensing_policy_sequence",
        ),
        (
            "policy_amplifies_controller",
            "active_sensing_policy_amplified_command",
        ),
        (
            "fusion_history_truncated",
            "active_sensing_fusion_history_truncated",
        ),
        (
            "trajectory_payload_changed",
            "inconsistent_active_sensing_trajectory",
        ),
        (
            "trajectory_expires_before_fusion",
            "active_sensing_trajectory_expired",
        ),
        (
            "pending_controller_evidence",
            "unterminated_active_sensing_evidence",
        ),
        ("zero_gate_still_armed", "unterminated_active_sensing_evidence"),
        ("zero_gate_count_mismatch", "invalid_active_sensing_zero_gate_count"),
        ("zero_gate_wrong_identity", "wrong_active_sensing_policy_identity"),
        (
            "zero_gate_global_count_mismatch",
            "invalid_active_sensing_zero_gate_count",
        ),
        (
            "future_grid_source",
            "post_completion_active_sensing_fusion",
        ),
        ("rotation_write_zero", "missing_active_sensing_rotation_command"),
        (
            "rotation_after_settle",
            "active_sensing_rotation_after_settle",
        ),
        (
            "maximum_write_mismatch",
            "inconsistent_active_sensing_command_aggregate",
        ),
        ("recovery_policy_not_after_active", "active_sensing_not_recovered"),
    ],
)
def test_dynamic_active_sensing_new_typed_joins_fail_closed(
    corruption: str,
    expected_code: str,
) -> None:
    """逐项证明主动观测新增时序、许可与容量合同不是空校验。"""

    summary = _base_summary("dynamic_f1")
    _add_active_sensing_report(summary)
    simulation = summary["simulation_report"]
    active_lifecycle = simulation["active_sensing_lifecycle_report"]  # type: ignore[index]
    attempt = active_lifecycle["attempts"][0]
    reports = attempt["event_reports"]  # type: ignore[index]
    controller = attempt["controller_command_aggregate"]  # type: ignore[index]
    policy = attempt["policy_command_aggregate"]  # type: ignore[index]

    if corruption == "event_receipt_regressed":
        reports[1]["receipt_timestamp"] = reports[0]["receipt_timestamp"]
    elif corruption == "event_rx_regressed":
        reports[1]["rx_sequence"] = reports[0]["rx_sequence"]
    elif corruption == "event_time_offset_forged":
        reports[0]["ros_time_offset_s"] = 1.0
    elif corruption == "settle_snapshot_changed":
        reports[3]["active_sensing"]["stable_duration"] = 0.12
    elif corruption == "fusion_progress_stalled":
        reports[4]["active_sensing"]["fusion_current"] = 3
        reports[4]["active_sensing"]["fusion_distinct"] = 1
    elif corruption == "completion_advances_fusion":
        reports[-1]["active_sensing"]["fusion_current"] = 6
        reports[-1]["active_sensing"]["fusion_distinct"] = 4
        attempt["completed"] = deepcopy(reports[-1])
    elif corruption == "controller_final":
        controller["first_status"]["is_final"] = True
    elif corruption == "controller_first_envelope_nonzero":
        controller["first_status"]["command_aggregate"]["max_abs_wz"] = 0.01
    elif corruption == "controller_acceptance_changed":
        controller["last_status"]["acceptance_sequence"] += 1
    elif corruption == "policy_gate_schema_forged":
        policy["first_write"]["navigation_gate_diagnostics"]["schema"] = "forged"
    elif corruption == "observed_status_schema_forged":
        policy["first_write"]["navigation_status_observed_report"][
            "schema"
        ] = "forged"
    elif corruption == "observed_status_idle":
        policy["first_write"]["navigation_status_observed_report"]["status"][
            "state"
        ] = 0
    elif corruption == "policy_snapshot_outside_active_window":
        policy["first_write"]["scan_controller_status_snapshot"][
            "status_sequence"
        ] = 1
    elif corruption == "policy_amplifies_controller":
        policy["max_abs_wz"] = 0.201
    elif corruption == "fusion_history_truncated":
        for report in reports[4:]:
            report["active_sensing"]["fusion_current"] = 70
            report["active_sensing"]["fusion_distinct"] = 3
        attempt["completed"] = deepcopy(reports[-1])
        attempt["planner_fusion"]["current"] = 70
    elif corruption == "trajectory_payload_changed":
        reports[3]["trajectory_duration_s"] = 2.80
        reports[3]["trajectory_sample_interval_s"] = 2.80 / 63.0
    elif corruption == "trajectory_expires_before_fusion":
        for report in reports:
            report["trajectory_duration_s"] = 1.20
            report["trajectory_sample_interval_s"] = 1.20 / 63.0
            report["trajectory_sample_count_total"] = 121
        for key, event_name in (
            ("started", "STARTED"),
            ("accepted", "ACCEPTED"),
            ("yaw_stable", "YAW_STABLE"),
            ("completed", "COMPLETED"),
        ):
            attempt[key] = deepcopy(
                next(
                    report
                    for report, event in zip(
                        reports,
                        attempt["events"],
                        strict=True,
                    )
                    if event == event_name
                )
            )
    elif corruption == "pending_controller_evidence":
        active_lifecycle["pending_active_controller_statuses"] = [  # type: ignore[index]
            deepcopy(controller["last_status"])
        ]
    elif corruption == "zero_gate_still_armed":
        active_lifecycle["policy_zero_gate"] = {  # type: ignore[index]
            "identity": deepcopy(attempt["identity"])
        }
    elif corruption == "zero_gate_count_mismatch":
        active_lifecycle["policy_zero_gate_consumed_count"] = 0  # type: ignore[index]
    elif corruption == "zero_gate_wrong_identity":
        active_lifecycle["policy_zero_gated_identities"][0][  # type: ignore[index]
            "traj_id"
        ] += 1
    elif corruption == "zero_gate_global_count_mismatch":
        simulation["navigation_policy_gate_lifecycle_report"][  # type: ignore[index]
            "stop_reason_counts"
        ]["active_sensing_identity_zero_gate"] = 2
    elif corruption == "future_grid_source":
        observation = attempt["post_settle_fused_observations"][-1]
        observation_sequence = observation["observation_sequence"]
        future_stamp_ns = 49_210_000_000
        observation["header_stamp_ns"] = future_stamp_ns
        observation["header"]["stamp_ns"] = future_stamp_ns
        observation["header"]["stamp"] = _stamp(future_stamp_ns)
        matching_grid = next(
            item
            for item in simulation["grid_map_observation_lifecycle_report"][  # type: ignore[index]
                "diagnostic_reports"
            ]
            if item["observation_sequence"] == observation_sequence
        )
        matching_grid["header"]["stamp_ns"] = future_stamp_ns
        matching_grid["header"]["stamp"] = _stamp(future_stamp_ns)
    elif corruption == "rotation_write_zero":
        for key in ("first_rotation_write", "maximum_abs_wz_write"):
            policy[key]["written_command"] = [0.0, 0.0, 0.0]
            policy[key]["limited_target"] = [0.0, 0.0, 0.0]
        rotation_sequence = policy["first_rotation_write"]["write_sequence"]
        pinned = next(
            item
            for item in simulation["navigation_policy_gate_lifecycle_report"][  # type: ignore[index]
                "identity_verified_tracking_write_reports"
            ]
            if item["write_sequence"] == rotation_sequence
        )
        pinned["written_command"] = [0.0, 0.0, 0.0]
    elif corruption == "rotation_after_settle":
        for key in ("first_rotation_write", "maximum_abs_wz_write"):
            policy[key]["timestamp"] = 48.70
        rotation_sequence = policy["first_rotation_write"]["write_sequence"]
        pinned = next(
            item
            for item in simulation["navigation_policy_gate_lifecycle_report"][  # type: ignore[index]
                "identity_verified_tracking_write_reports"
            ]
            if item["write_sequence"] == rotation_sequence
        )
        pinned["timestamp"] = 48.70
    elif corruption == "maximum_write_mismatch":
        policy["maximum_abs_wz_write"]["written_command"] = [0.0, 0.0, 0.10]
    elif corruption == "recovery_policy_not_after_active":
        simulation["dynamic_navigation_evidence_report"]["trajectory_recovery"][
            "policy_identity_verified_tracking_write"
        ]["write_sequence"] = policy["last_write"]["write_sequence"]
    else:  # pragma: no cover - 参数列表与分支必须同步。
        raise AssertionError(corruption)

    validation = validate_pct_scan_live_summary(
        summary,
        "dynamic_f1",
        require_active_sensing=True,
    )

    assert validation["valid"] is False
    assert any(
        issue["code"] == expected_code for issue in validation["errors"]
    ), validation["errors"]


def test_active_sensing_flag_is_restricted_to_dynamic_f1() -> None:
    with pytest.raises(SummaryInputError, match="只允许用于 dynamic_f1"):
        validate_pct_scan_live_summary(
            _base_summary("flat_policy"),
            "flat_policy",
            require_active_sensing=True,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "static_stair",
        "flat_policy",
        "crossfloor_carry",
        "dynamic_f1",
        "dynamic_replan_f1",
    ],
)
def test_all_modes_reject_invalid_policy_write_count(mode: str) -> None:
    summary = _base_summary(mode)
    summary["latest_executor_status"]["invalid_policy_write_count"] = 1  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, mode)  # type: ignore[arg-type]

    assert any(
        issue["code"] == "protocol_error_count"
        and issue["path"]
        == "$.latest_executor_status.invalid_policy_write_count"
        for issue in report["errors"]
    )


def test_static_stair_accepts_bounded_supervisor_ack_wait_count() -> None:
    summary = _base_summary("static_stair")
    summary["latest_executor_status"][  # type: ignore[index]
        "goal_true_waiting_for_supervisor_ack_count"
    ] = 1

    report = validate_pct_scan_live_summary(  # type: ignore[arg-type]
        summary,
        "static_stair",
    )

    assert report["valid"] is True, report["errors"]


@pytest.mark.parametrize(
    ("field", "drifted_value", "expected_code"),
    (
        ("invalid_controller_status_count", 1, "protocol_error_count"),
        ("controller_status_sequence_reset_count", 1, "protocol_error_count"),
        ("terminal_supervisor_goal_acknowledged", False, "unexpected_value"),
        (
            "terminal_supervisor_goal_pending_started_timestamp",
            1.2,
            "unexpected_value",
        ),
        (
            "policy_freeze_write_fault_reasons",
            ["requested_command_not_zero"],
            "unexpected_policy_freeze_write_fault",
        ),
        ("policy_freeze_write_fault_sequence", 9, "unexpected_value"),
        ("policy_freeze_write_fault_timestamp", 1.2, "unexpected_value"),
    ),
)
def test_static_stair_rejects_terminal_protocol_faults(
    field: str,
    drifted_value: object,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    stair = summary["latest_executor_status"]["stair_freeze"]  # type: ignore[index]
    stair[field] = drifted_value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(
        issue["code"] == expected_code
        and issue["path"] == f"$.latest_executor_status.stair_freeze.{field}"
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "drifted_value", "expected_code"),
    (
        ("required", False, "unexpected_value"),
        ("passed", False, "unexpected_value"),
        ("pending", True, "unexpected_value"),
        ("path_stamp_ns", PATH_STAMP_NS + 1, "wrong_identity"),
        ("timeout_s", 7.0, "unexpected_value"),
        ("status_freshness_timeout_s", 0.5, "unexpected_value"),
        ("progress_m_at_pass", 0.01, "unexpected_value"),
        ("local_sensors_fresh", False, "unexpected_value"),
        ("supervisor_sensors_fresh", False, "unexpected_value"),
        ("pending_reasons", ["supervisor_point_cloud_stale"], "sensor_acquisition_not_complete"),
    ),
)
def test_static_stair_rejects_sensor_acquisition_barrier_drift(
    field: str,
    drifted_value: object,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    barrier[field] = drifted_value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize(
    ("field", "drifted_value", "expected_code"),
    (
        ("state", 2, "invalid_stair_freeze_acknowledgement"),
        ("state", 4, "invalid_stair_freeze_acknowledgement"),
        ("state", 255, "invalid_stair_freeze_acknowledgement"),
        (
            "reason",
            "scan_status_invariant:Path 绑定的 SCAN 事件缺少 reference stamp",
            "unexpected_value",
        ),
        ("reason", "scan_emergency_stop", "unexpected_value"),
        ("stop_confirmed", False, "unexpected_value"),
        ("global_replan_requested", True, "unexpected_value"),
        ("global_replan_in_flight", True, "unexpected_value"),
        ("global_replan_request_id", 7, "invalid_stair_freeze_acknowledgement"),
        ("pct_plan_id", 0, "invalid_number"),
        ("consecutive_scan_failures", 5, "invalid_stair_freeze_acknowledgement"),
    ),
)
def test_static_stair_rejects_invalid_supervisor_freeze_ack(
    field: str,
    drifted_value: object,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    status = barrier["navigation_status_observed_report"]["status"]  # type: ignore[index]
    status[field] = drifted_value  # type: ignore[index]
    barrier["last_navigation_status_observed_report"] = deepcopy(  # type: ignore[index]
        barrier["navigation_status_observed_report"]  # type: ignore[index]
    )
    barrier["policy_write_report"][  # type: ignore[index]
        "navigation_status_observed_report"
    ] = deepcopy(barrier["navigation_status_observed_report"])  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize("stale_inputs", ([], ["bspline", "bogus"]))
def test_static_stair_rejects_noncanonical_freeze_ack_stale_inputs(
    stale_inputs: list[str],
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    status = barrier["navigation_status_observed_report"]["status"]  # type: ignore[index]
    status["stale_inputs"] = stale_inputs  # type: ignore[index]
    barrier["last_navigation_status_observed_report"] = deepcopy(  # type: ignore[index]
        barrier["navigation_status_observed_report"]  # type: ignore[index]
    )
    barrier["policy_write_report"][  # type: ignore[index]
        "navigation_status_observed_report"
    ] = deepcopy(barrier["navigation_status_observed_report"])  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(
        issue["code"] == "stale_sensor_acquisition_input"
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "drifted_value", "expected_code"),
    (
        ("last_write_sequence", 999, "wrong_identity"),
        ("last_write_timestamp", 999.0, "unexpected_value"),
        (
            "last_navigation_status_observed_report",
            {"forged": True},
            "wrong_typed_evidence_reference",
        ),
    ),
)
def test_static_stair_rejects_forged_sensor_barrier_last_evidence(
    field: str,
    drifted_value: object,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    barrier[field] = drifted_value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


def test_static_stair_rejects_barrier_sequence_beyond_policy_lifecycle() -> None:
    """同时伪造屏障全部序号副本也不能逃过 lifecycle 上界。"""

    summary = _base_summary("static_stair")
    stair = summary["latest_executor_status"]["stair_freeze"]  # type: ignore[index]
    barrier = stair["sensor_acquisition_barrier"]  # type: ignore[index]
    barrier["write_sequence"] = 999  # type: ignore[index]
    barrier["last_write_sequence"] = 999  # type: ignore[index]
    barrier["policy_write_report"]["write_sequence"] = 999  # type: ignore[index]
    stair["sensor_acquisition_write_sequence"] = 999  # type: ignore[index]
    stair["sensor_acquisition_last_write_sequence"] = 999  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(
        issue["code"] == "invalid_sensor_acquisition_order"
        and issue["path"].endswith("sensor_acquisition_barrier.write_sequence")
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "drifted_value", "expected_code"),
    (
        ("status_sequence", 999, "invalid_status_sequence"),
        ("rx_sequence", 999, "invalid_status_sequence"),
        ("state_revision", 999, "invalid_status_sequence"),
        ("header_stamp_ns", 1, "invalid_sensor_acquisition_order"),
        ("pct_plan_id", 999, "wrong_identity"),
    ),
)
def test_static_stair_rejects_ack_outside_lifecycle_bounds(
    field: str,
    drifted_value: object,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    status = barrier["navigation_status_observed_report"]["status"]  # type: ignore[index]
    status[field] = drifted_value  # type: ignore[index]
    barrier["last_navigation_status_observed_report"] = deepcopy(  # type: ignore[index]
        barrier["navigation_status_observed_report"]  # type: ignore[index]
    )
    barrier["policy_write_report"][  # type: ignore[index]
        "navigation_status_observed_report"
    ] = deepcopy(barrier["navigation_status_observed_report"])  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("zero_emergency_count", "invalid_number"),
        ("missing_emergency_state_count", "missing_field"),
        ("missing_inhibit_reason_count", "missing_field"),
        ("missing_freeze_stop_count", "missing_field"),
        ("zero_forced_write_count", "invalid_number"),
    ),
)
def test_static_stair_requires_ack_lifecycle_counts(
    mutation: str,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    if mutation == "zero_emergency_count":
        lifecycle["emergency_stop_observed_status_count"] = 0  # type: ignore[index]
    elif mutation == "missing_emergency_state_count":
        del lifecycle["observed_state_counts"]["emergency_stop"]  # type: ignore[index]
    elif mutation == "missing_inhibit_reason_count":
        del lifecycle["observed_reason_counts"][  # type: ignore[index]
            "scan_stair_execution_inhibited"
        ]
    elif mutation == "missing_freeze_stop_count":
        del lifecycle["stop_reason_counts"]["scan_stair_freeze"]  # type: ignore[index]
    elif mutation == "zero_forced_write_count":
        lifecycle["forced_zero_write_count"] = 0  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize("stale_input", ["odometry", "point_cloud"])
def test_static_stair_sensor_barrier_rejects_stale_required_input(
    stale_input: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    status = barrier["navigation_status_observed_report"]["status"]  # type: ignore[index]
    status["stale_inputs"] = [stale_input, "bspline"]  # type: ignore[index]
    barrier["policy_write_report"][  # type: ignore[index]
        "navigation_status_observed_report"
    ] = deepcopy(barrier["navigation_status_observed_report"])  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(
        issue["code"] == "stale_sensor_acquisition_input"
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("write_not_after_activation", "invalid_sensor_acquisition_order"),
        ("status_not_after_activation", "invalid_sensor_acquisition_order"),
        ("status_from_future", "sensor_acquisition_status_from_future"),
        ("status_timed_out", "sensor_acquisition_status_timeout"),
        ("sequence_not_after_floor", "invalid_sensor_acquisition_order"),
    ),
)
def test_static_stair_rejects_sensor_acquisition_time_or_sequence_forgery(
    mutation: str,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    stair = summary["latest_executor_status"]["stair_freeze"]  # type: ignore[index]
    barrier = stair["sensor_acquisition_barrier"]  # type: ignore[index]
    status = barrier["navigation_status_observed_report"]["status"]  # type: ignore[index]
    if mutation == "write_not_after_activation":
        barrier["activation_timestamp"] = barrier["write_timestamp"]  # type: ignore[index]
        stair["sensor_acquisition_started_timestamp"] = barrier[  # type: ignore[index]
            "activation_timestamp"
        ]
    elif mutation == "status_not_after_activation":
        status["receipt_timestamp"] = barrier["activation_timestamp"]  # type: ignore[index]
    elif mutation == "status_from_future":
        status["receipt_timestamp"] = float(barrier["write_timestamp"]) + 0.01  # type: ignore[arg-type,index]
    elif mutation == "status_timed_out":
        barrier["activation_timestamp"] = 0.5  # type: ignore[index]
        stair["sensor_acquisition_started_timestamp"] = 0.5  # type: ignore[index]
        status["receipt_timestamp"] = 0.80  # type: ignore[index]
        barrier["write_timestamp"] = 1.10  # type: ignore[index]
        stair["sensor_acquisition_completed_timestamp"] = 1.10  # type: ignore[index]
        stair["sensor_acquisition_last_write_timestamp"] = 1.10  # type: ignore[index]
        barrier["policy_write_report"]["timestamp"] = 1.10  # type: ignore[index]
    elif mutation == "sequence_not_after_floor":
        stair["sensor_acquisition_write_sequence_floor"] = barrier[  # type: ignore[index]
            "write_sequence"
        ]
    barrier["policy_write_report"][  # type: ignore[index]
        "navigation_status_observed_report"
    ] = deepcopy(barrier["navigation_status_observed_report"])  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("sequence", "wrong_identity"),
        ("timestamp", "unexpected_value"),
        ("owner", "unexpected_value"),
        ("nonzero", "nonzero_sensor_acquisition_write"),
        ("sensor_stop", "invalid_sensor_acquisition_stop_context"),
        ("status_reference", "wrong_typed_evidence_reference"),
    ),
)
def test_static_stair_rejects_forged_sensor_acquisition_policy_write(
    mutation: str,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    barrier = summary["latest_executor_status"]["stair_freeze"][  # type: ignore[index]
        "sensor_acquisition_barrier"
    ]
    policy = barrier["policy_write_report"]  # type: ignore[index]
    if mutation == "sequence":
        policy["write_sequence"] = int(barrier["write_sequence"]) + 1  # type: ignore[arg-type,index]
    elif mutation == "timestamp":
        policy["timestamp"] = float(barrier["write_timestamp"]) + 0.01  # type: ignore[arg-type,index]
    elif mutation == "owner":
        policy["owner_id"] = "second_writer"  # type: ignore[index]
    elif mutation == "nonzero":
        policy["written_command"] = [0.01, 0.0, 0.0]  # type: ignore[index]
    elif mutation == "sensor_stop":
        policy["stop_reasons"] = [  # type: ignore[index]
            "scan_stair_freeze",
            "point_cloud_timeout",
        ]
    elif mutation == "status_reference":
        policy["navigation_status_observed_report"] = deepcopy(  # type: ignore[index]
            barrier["navigation_status_observed_report"]
        )
        policy["navigation_status_observed_report"]["status"][  # type: ignore[index]
            "status_sequence"
        ] = 999

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(issue["code"] == expected_code for issue in report["errors"])


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (
        ("sensor_safety_fault_reasons", ["supervisor_point_cloud_stale"]),
        ("sensor_safety_fault_write_sequence", 4),
        ("sensor_safety_fault_timestamp", 1.2),
    ),
)
def test_static_stair_success_rejects_latched_sensor_fault(
    field: str,
    drifted_value: object,
) -> None:
    summary = _base_summary("static_stair")
    stair = summary["latest_executor_status"]["stair_freeze"]  # type: ignore[index]
    stair[field] = drifted_value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")  # type: ignore[arg-type]

    assert any(
        issue["code"] in {"unexpected_sensor_safety_fault", "unexpected_value"}
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "static_stair",
        "flat_policy",
        "crossfloor_carry",
        "dynamic_f1",
        "dynamic_replan_f1",
    ],
)
def test_all_modes_reject_scan_stair_freeze_profile_audit_drift(
    mode: str,
) -> None:
    summary = _base_summary(mode)
    runtime = summary["task_config"][  # type: ignore[index]
        "scan_stair_freeze_profile_runtime"
    ]
    runtime["profile_id"] = "drifted-profile"  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, mode)  # type: ignore[arg-type]

    assert any(
        issue["code"] == "scan_stair_freeze_profile_audit_drift"
        and issue["path"].endswith(".profile_id")
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (
        ("profile_id", "drifted-profile"),
        ("scene", "flat_only"),
        ("robot", "go2"),
        ("controller", "legacy_controller"),
        ("contract_sha256", "0" * 64),
        ("source_sha256", "0" * 64),
        ("source_path", "/tmp/drifted-profile.json"),
        ("source_branch", "pct-scan"),
        ("baseline_behavior", "physics_tracking"),
        ("pct_topology_profile_reused", 0),
        ("non_physical_root_lock_workaround", 1),
    ),
)
def test_rejects_each_scan_stair_freeze_profile_audit_field_drift(
    field: str,
    drifted_value: object,
) -> None:
    summary = _base_summary("flat_policy")
    runtime = summary["task_config"][  # type: ignore[index]
        "scan_stair_freeze_profile_runtime"
    ]
    runtime[field] = drifted_value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert any(
        issue["code"] == "scan_stair_freeze_profile_audit_drift"
        and issue["path"].endswith(f".{field}")
        for issue in report["errors"]
    )


@pytest.mark.parametrize("schema_drift", ["missing_key", "unexpected_key"])
def test_rejects_scan_stair_freeze_profile_audit_schema_drift(
    schema_drift: str,
) -> None:
    summary = _base_summary("flat_policy")
    runtime = summary["task_config"][  # type: ignore[index]
        "scan_stair_freeze_profile_runtime"
    ]
    if schema_drift == "missing_key":
        del runtime["source_branch"]  # type: ignore[index]
    else:
        runtime["legacy_profile_alias"] = "not-allowed"  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert any(
        issue["code"]
        == "invalid_scan_stair_freeze_profile_audit_schema"
        for issue in report["errors"]
    )


def test_rejects_missing_scan_stair_freeze_profile_runtime_audit() -> None:
    summary = _base_summary("flat_policy")
    del summary["task_config"][  # type: ignore[index]
        "scan_stair_freeze_profile_runtime"
    ]

    report = validate_pct_scan_live_summary(summary, "flat_policy")

    assert any(
        issue["code"] == "missing_field"
        and issue["path"]
        == "$.task_config.scan_stair_freeze_profile_runtime"
        for issue in report["errors"]
    )


def test_dynamic_replan_mode_fails_when_replan_never_triggered() -> None:
    summary = _base_summary("dynamic_replan_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle.update(  # type: ignore[union-attr]
        {
            "maximum_consecutive_scan_failures": 0,
            "global_replan_requested_status_count": 0,
            "global_replan_in_flight_status_count": 0,
            "distinct_global_replan_request_ids": [],
            "distinct_pct_plan_ids": [1],
            "first_global_replan_status": None,
            "last_global_replan_status": None,
            "global_replan_pending_recovery": False,
            "tracking_after_global_replan_observed": False,
            "global_replan_recovery_count": 0,
        }
    )

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_required_global_replan" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_requires_obstacle_transition_before_replan() -> None:
    summary = _base_summary("dynamic_replan_f1")
    grid_lifecycle = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_lifecycle_report"
    ]
    transition = grid_lifecycle["first_transition_hit_report"]  # type: ignore[index]
    transition_receipt = float(transition["receipt_timestamp"])
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    first = lifecycle["first_global_replan_status"]  # type: ignore[index]
    last = lifecycle["last_global_replan_status"]  # type: ignore[index]
    for evidence in (first, last):
            evidence["navigation_status_observed_report"]["status"][  # type: ignore[index]
                "receipt_timestamp"
            ] = transition_receipt - 0.25

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_dynamic_replan_obstacle_transition" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_requires_clear_after_replan() -> None:
    summary = _base_summary("dynamic_replan_f1")
    simulation = summary["simulation_report"]
    grid = simulation["grid_map_observation_lifecycle_report"]  # type: ignore[index]
    clear = grid["last_report"]  # type: ignore[index]
    clear["dynamic_obstacle_explicit_miss_clear_matches"] = []  # type: ignore[index]
    grid["first_explicit_miss_clear_report"] = clear  # type: ignore[index]
    grid["last_explicit_miss_clear_report"] = clear  # type: ignore[index]
    simulation["grid_map_observation_diagnostics_last_report"] = clear  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_post_replan_explicit_clear" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_requires_policy_recovery_after_clear() -> None:
    summary = _base_summary("dynamic_replan_f1")
    grid_lifecycle = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_lifecycle_report"
    ]
    clear = grid_lifecycle["last_explicit_miss_clear_report"]  # type: ignore[index]
    clear_receipt = float(clear["receipt_timestamp"])
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["last_identity_verified_tracking_write"][  # type: ignore[index]
        "timestamp"
    ] = clear_receipt - 0.1

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "tracking_recovered_before_obstacle_clear" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_accepts_typed_chinese_scan_failure_reason() -> None:
    summary = _base_summary("dynamic_replan_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    for key in ("first_global_replan_status", "last_global_replan_status"):
        lifecycle[key]["navigation_status_observed_report"]["status"][  # type: ignore[index]
            "reason"
        ] = "预测碰撞后的局部重规划失败:5"

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is True, report["errors"]


def test_dynamic_replan_mode_requires_threshold_in_first_replan_snapshot() -> None:
    summary = _base_summary("dynamic_replan_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["first_global_replan_status"][  # type: ignore[index]
        "navigation_status_observed_report"
    ]["status"]["consecutive_scan_failures"] = 4

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_scan_failure_replan_trigger" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_requires_old_reference_to_be_blocked() -> None:
    summary = _base_summary("dynamic_replan_f1")
    bspline = summary["simulation_report"][  # type: ignore[index]
        "bspline_diagnostics_lifecycle_report"
    ]["first_report"]
    bspline["ordered_reference_samples_world_xyz"] = [  # type: ignore[index]
        [-8.0, -8.0, 0.4],
        [-7.9, -8.0, 0.4],
    ]

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_replan_reference_obstruction" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_replan_mode_requires_nonzero_policy_motion_after_clear() -> None:
    summary = _base_summary("dynamic_replan_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "navigation_policy_gate_lifecycle_report"
    ]
    lifecycle["last_identity_verified_tracking_write"][  # type: ignore[index]
        "written_command"
    ] = [0.0, 0.0, 0.0]

    report = validate_pct_scan_live_summary(summary, "dynamic_replan_f1")

    assert report["valid"] is False
    assert "missing_post_replan_policy_motion" in {
        issue["code"] for issue in report["errors"]
    }


def test_dynamic_final_mode_is_fail_closed_without_perception_geometry() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    for key in (
        "grid_map_observation_diagnostics_last_report",
        "grid_map_observation_lifecycle_report",
        "bspline_diagnostics_last_report",
        "bspline_diagnostics_lifecycle_report",
        "dynamic_navigation_evidence_report",
    ):
        del simulation[key]  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert report["valid"] is False
    codes = {issue["code"] for issue in report["errors"]}
    assert {
        "missing_live_point_cloud_obstacle_evidence",
        "missing_detour_geometry_evidence",
        "missing_free_ray_ghost_clear_evidence",
    }.issubset(codes)
    assert report["validated_claims"] == []


def test_rejects_missing_status_sequence_and_missing_state_trace() -> None:
    summary = _base_summary("flat_policy")
    summary["state_trace"] = []
    lifecycle = summary["simulation_report"]["navigation_policy_gate_lifecycle_report"]  # type: ignore[index]
    lifecycle["observed_status_sequence_count"] = 0  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "flat_policy")
    codes = {issue["code"] for issue in report["errors"]}
    assert "missing_state_sequence" in codes
    assert "missing_status_sequence" in codes


def test_rejects_wrong_observed_and_consumed_identity() -> None:
    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"]["navigation_policy_gate_lifecycle_report"]  # type: ignore[index]
    observed = lifecycle["last_identity_valid_observed_status"]  # type: ignore[index]
    observed["navigation_status_observed_report"]["status"]["goal_id"] = 999  # type: ignore[index]
    consumed = lifecycle["last_identity_verified_tracking_write"]  # type: ignore[index]
    consumed["navigation_gate_diagnostics"]["command_identity"] = [999, PATH_STAMP_NS, 2]  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "flat_policy")
    assert sum(issue["code"] == "wrong_identity" for issue in report["errors"]) >= 2


@pytest.mark.parametrize("field", ["last_requested_command", "last_written_command"])
def test_rejects_nonzero_terminal_command(field: str) -> None:
    summary = _base_summary("flat_policy")
    summary["latest_executor_status"][field] = [0.1, 0.0, 0.0]  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "flat_policy")
    assert any(issue["code"] == "post_goal_motion" for issue in report["errors"])


def test_rejects_post_goal_nonzero_count() -> None:
    summary = _base_summary("flat_policy")
    summary["latest_executor_status"]["post_goal_nonzero_write_count"] = 1  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "flat_policy")
    assert any(issue["code"] == "post_goal_motion" for issue in report["errors"])


def test_static_allows_no_consumed_tracking_but_requires_complete_progress() -> None:
    valid = validate_pct_scan_live_summary(_base_summary("static_stair"), "static_stair")
    assert valid["valid"] is True
    incomplete = _base_summary("static_stair")
    incomplete["latest_executor_status"]["stair_freeze"]["progress_m"] = 7.0  # type: ignore[index]
    report = validate_pct_scan_live_summary(incomplete, "static_stair")
    assert any(issue["code"] == "incomplete_stair_progress" for issue in report["errors"])


def test_static_rejects_legacy_untyped_stair_freeze_report() -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    del publish["message_type"]  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")

    assert any(
        issue["code"] == "missing_field"
        and issue["path"].endswith(".message_type")
        for issue in report["errors"]
    )


def test_static_rejects_stair_freeze_wrong_path_identity() -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    publish["reference_path_stamp_ns"] = PATH_STAMP_NS + 1  # type: ignore[index]
    publish["reference_path_stamp"] = _stamp(PATH_STAMP_NS + 1)  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")

    assert any(
        issue["code"] == "wrong_identity"
        and issue["path"].endswith(".reference_path_stamp_ns")
        for issue in report["errors"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("writer_id", "", "unexpected_value"),
        ("writer_epoch", " epoch-with-spaces ", "invalid_writer_identity"),
    ),
)
def test_static_rejects_invalid_stair_freeze_writer_identity(
    field: str,
    value: str,
    expected_code: str,
) -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    publish[field] = value  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")

    assert any(
        issue["code"] == expected_code
        and issue["path"].endswith(f".{field}")
        for issue in report["errors"]
    )


def test_static_rejects_stair_freeze_header_time_or_frame_mismatch() -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    publish["header"] = {  # type: ignore[index]
        "frame_id": "map",
        "stamp": _stamp(9_000_000_000),
    }

    report = validate_pct_scan_live_summary(summary, "static_stair")

    errors = report["errors"]
    assert any(
        issue["code"] == "unexpected_value"
        and issue["path"].endswith(".header.frame_id")
        for issue in errors
    )
    assert any(
        issue["code"] == "invalid_timestamp"
        and issue["path"].endswith(".header.stamp")
        for issue in errors
    )


def test_static_rejects_inconsistent_typed_stair_freeze_state() -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    publish["frozen"] = False  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")

    assert any(
        issue["code"] == "inconsistent_freeze_state"
        and issue["path"].endswith(".frozen")
        for issue in report["errors"]
    )


def test_static_rejects_nonpositive_stair_freeze_sequence() -> None:
    summary = _base_summary("static_stair")
    publish = summary["simulation_report"][  # type: ignore[index]
        "navigation_stair_execution_frozen_last_publish_report"
    ]
    publish["sequence"] = 0  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "static_stair")

    assert any(
        issue["code"] == "invalid_number"
        and issue["path"].endswith(".sequence")
        for issue in report["errors"]
    )


def test_flat_requires_identity_verified_tracking_write() -> None:
    summary = _base_summary("flat_policy")
    lifecycle = summary["simulation_report"]["navigation_policy_gate_lifecycle_report"]  # type: ignore[index]
    lifecycle["motion_allowed_write_count"] = 0  # type: ignore[index]
    lifecycle["identity_verified_tracking_write_count"] = 0  # type: ignore[index]
    lifecycle["first_identity_verified_tracking_write"] = None  # type: ignore[index]
    lifecycle["last_identity_verified_tracking_write"] = None  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "flat_policy")
    assert any(issue["code"] == "missing_tracking_write" for issue in report["errors"])


def test_dynamic_requires_lifecycle_motion_and_multiple_trajectory_evidence() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    del simulation["dynamic_obstacle_lifecycle_report"]  # type: ignore[index]
    controller = simulation["scan_controller_status_lifecycle_report"]  # type: ignore[index]
    controller["distinct_accepted_trajectory_count"] = 1  # type: ignore[index]
    controller["trajectory_replacement_count"] = 0  # type: ignore[index]
    controller["accepted_trajectory_identities"] = [_controller_identity(1)]  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "missing_field" in codes
    assert "missing_dynamic_replan_evidence" in codes


def test_dynamic_rejects_wrong_deterministic_state_and_discloses_scope() -> None:
    summary = _base_summary("dynamic_f1")
    runtime = summary["simulation_report"]["dynamic_obstacle_runtime_report"]  # type: ignore[index]
    runtime["obstacles"][0]["position_world_xyz"][0] += 0.1  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(issue["code"] == "invalid_dynamic_state" for issue in report["errors"])
    assert (
        "scan_ordered_local_detour_and_current_obstacle_clearance_verified"
        in report["not_validated_claims"]
    )
    assert not any("detour" in claim for claim in report["validated_claims"])


def test_dynamic_requires_moving_obstacle_hits_in_raw_rtx_cloud() -> None:
    summary = _base_summary("dynamic_f1")
    raw = summary["simulation_report"]["dynamic_obstacle_raw_cloud_lifecycle_report"]  # type: ignore[index]
    obstacle = raw["obstacles"]["crossing_cart"]  # type: ignore[index]
    obstacle["detected_path_distance_span_m"] = 0.0  # type: ignore[index]
    obstacle["maximum_detected_path_distance_m"] = obstacle["minimum_detected_path_distance_m"]  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_raw_cloud_motion_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_post_filter_hit_outside_deterministic_cart() -> None:
    summary = _base_summary("dynamic_f1")
    match = summary["simulation_report"]["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "post_filter_hit"
    ]["dynamic_obstacle_hit_matches"][0]
    match["obstacle_state"]["position_world_xyz"][0] += 0.2  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(issue["code"] == "invalid_dynamic_state" for issue in report["errors"])


def test_dynamic_rejects_ros_header_without_episode_local_time_binding() -> None:
    summary = _base_summary("dynamic_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_lifecycle_report"
    ]
    hit_report = lifecycle["first_hit_report"]  # type: ignore[index]
    hit_report["episode_elapsed_time_s"] = 7.5  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_episode_time_binding"
        for issue in report["errors"]
    )


def test_dynamic_rejects_inconsistent_typed_episode_time_origins() -> None:
    summary = _base_summary("dynamic_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "bspline_diagnostics_lifecycle_report"
    ]
    lifecycle["ros_time_offset_s"] = 1.0  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "inconsistent_episode_time_offset"
        for issue in report["errors"]
    )


def test_dynamic_accepts_bounded_grid_and_bspline_diagnostic_rings() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]

    grid_lifecycle = simulation["grid_map_observation_lifecycle_report"]  # type: ignore[index]
    grid_hit = grid_lifecycle["first_report"]  # type: ignore[index]
    grid_clear = grid_lifecycle["last_report"]  # type: ignore[index]
    grid_clear["observation_sequence"] = 129  # type: ignore[index]
    grid_clear["rx_sequence"] = 129  # type: ignore[index]
    grid_lifecycle["sample_count"] = 129  # type: ignore[index]
    grid_lifecycle["last_observation_sequence"] = 129  # type: ignore[index]
    grid_lifecycle["dropped_diagnostic_report_count"] = 1  # type: ignore[index]
    intermediate_grid_reports = [
        _grid_map_diagnostic_report(
            sequence,
            timestamp_s=8.50 + 0.01 * (sequence - 1),
            hit_samples=[],
            clear_samples=[],
        )
        for sequence in range(2, 129)
    ]
    grid_lifecycle["diagnostic_reports"] = [  # type: ignore[index]
        *intermediate_grid_reports,
        grid_clear,
    ]
    ghost = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "explicit_miss_ghost_clear"
    ]
    ghost["observation_sequence"] = 129  # type: ignore[index]
    assert grid_hit["observation_sequence"] == 1  # type: ignore[index]

    bspline_lifecycle = simulation["bspline_diagnostics_lifecycle_report"]  # type: ignore[index]
    recovery_report = bspline_lifecycle["last_report"]  # type: ignore[index]
    recovery_report["diagnostic_sequence"] = 129  # type: ignore[index]
    bspline_lifecycle["sample_count"] = 129  # type: ignore[index]
    bspline_lifecycle["last_diagnostic_sequence"] = 129  # type: ignore[index]
    bspline_lifecycle["dropped_diagnostic_report_count"] = 1  # type: ignore[index]
    intermediate_bspline_reports = []
    for sequence in range(2, 129):
        report = deepcopy(recovery_report)
        report["diagnostic_sequence"] = sequence
        intermediate_bspline_reports.append(report)
    bspline_lifecycle["diagnostic_reports"] = [  # type: ignore[index]
        *intermediate_bspline_reports,
        recovery_report,
    ]
    recovery = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "trajectory_recovery"
    ]
    recovery["after_diagnostic_sequence"] = 129  # type: ignore[index]

    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert report["valid"] is True, report["errors"]


def test_dynamic_rejects_detour_identity_not_accepted_by_controller() -> None:
    summary = _base_summary("dynamic_f1")
    identity = summary["simulation_report"]["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "ordered_detour"
    ]["identity"]
    identity["traj_id"] = 99  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "wrong_identity" in codes


def test_dynamic_rejects_detour_controller_snapshot_identity_mismatch() -> None:
    summary = _base_summary("dynamic_f1")
    detour = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["ordered_detour"]
    snapshot = deepcopy(detour["controller_accepted_status"])
    snapshot["identity"]["traj_id"] = 99
    detour["controller_accepted_status"] = snapshot
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "controller_identity_not_accepted" in codes
    assert "wrong_controller_acceptance_reference" in codes


def test_dynamic_rejects_detour_controller_acceptance_after_clear() -> None:
    summary = _base_summary("dynamic_f1")
    detour = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["ordered_detour"]
    snapshot = detour["controller_accepted_status"]
    late_stamp_ns = 10_500_000_000
    snapshot["header"]["stamp_ns"] = late_stamp_ns
    snapshot["header"]["stamp"] = _stamp(late_stamp_ns)
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        for issue in report["errors"]
    )


def test_dynamic_recomputes_current_obstacle_clearance() -> None:
    summary = _base_summary("dynamic_f1")
    clearance = summary["simulation_report"]["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "current_obstacle_clearance"
    ]
    clearance["required_clearance_m"] = 0.60  # type: ignore[index]
    clearance["obstacle_clearances"][0]["required_clearance_m"] = 0.60  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "insufficient_dynamic_obstacle_clearance"
        for issue in report["errors"]
    )


def test_dynamic_requires_cart_to_obstruct_original_reference() -> None:
    summary = _base_summary("dynamic_f1")
    clearance = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["current_obstacle_clearance"]
    clearance["obstacle_clearances"][0]["reference_obstructed"] = False  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "invalid_reference_obstruction_evidence" in codes
    assert "missing_reference_obstruction_evidence" not in codes


def test_dynamic_rejects_deviation_when_reference_was_already_clear() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    detour_report = simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
        "first_report"
    ]
    for point in detour_report["ordered_reference_samples_world_xyz"]:  # type: ignore[index]
        point[1] = 3.5
    clearance = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "current_obstacle_clearance"
    ]["obstacle_clearances"][0]
    clearance["minimum_ordered_reference_center_to_obstacle_xy_m"] = 0.525  # type: ignore[index]
    clearance["reference_obstructed"] = False  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_reference_obstruction_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_sliding_reset_as_explicit_miss_ghost_clear() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    clear_report = simulation["grid_map_observation_diagnostics_last_report"]  # type: ignore[index]
    clear_report["occupied_removed_by_sliding_reset_count"] = 1  # type: ignore[index]
    ghost = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "explicit_miss_ghost_clear"
    ]
    ghost["occupied_removed_by_sliding_reset_count"] = 1  # type: ignore[index]
    ghost["clear_matches"][0]["sliding_reset_used"] = True  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "sliding_reset_not_explicit_miss"
        for issue in report["errors"]
    )


def test_dynamic_allows_unrelated_sliding_reset_with_sample_provenance() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    clear_report = simulation["grid_map_observation_diagnostics_last_report"]  # type: ignore[index]
    clear_report["occupied_removed_by_sliding_reset_count"] = 1  # type: ignore[index]
    ghost = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "explicit_miss_ghost_clear"
    ]
    ghost["occupied_removed_by_sliding_reset_count"] = 1  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert report["valid"] is True, report["errors"]


def test_dynamic_rejects_neighbor_voxel_as_clear_provenance() -> None:
    summary = _base_summary("dynamic_f1")
    match = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["explicit_miss_ghost_clear"]["clear_matches"][0]
    wrong_voxel = list(match["voxel_index_xyz"])
    wrong_voxel[0] += 1
    match["voxel_index_xyz"] = wrong_voxel
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "wrong_geometry_reference" in codes


def test_dynamic_accepts_direct_clear_provenance_without_transition_ring() -> None:
    summary = _base_summary("dynamic_f1")
    lifecycle = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_lifecycle_report"
    ]
    lifecycle["transition_hit_reports"] = []  # type: ignore[index]
    lifecycle["first_transition_hit_report"] = None  # type: ignore[index]
    lifecycle["last_transition_hit_report"] = None  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert report["valid"] is True, report["errors"]


def test_dynamic_rejects_direct_provenance_header_mismatch() -> None:
    summary = _base_summary("dynamic_f1")
    clear_report = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_diagnostics_last_report"
    ]
    clear_report["occupied_to_free_transition_hit_header_stamp_ns"][0] += 1  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "wrong_typed_evidence_reference"
        for issue in report["errors"]
    )


def test_dynamic_rejects_unverified_direct_provenance() -> None:
    summary = _base_summary("dynamic_f1")
    match = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["explicit_miss_ghost_clear"]["clear_matches"][0]
    match["matched_hit_provenance_verified"] = False  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["path"].endswith("matched_hit_provenance_verified")
        for issue in report["errors"]
    )


def test_dynamic_rejects_sparse_grid_samples_claimed_as_truncated() -> None:
    summary = _base_summary("dynamic_f1")
    hit_report = summary["simulation_report"][  # type: ignore[index]
        "grid_map_observation_lifecycle_report"
    ]["first_report"]
    extra_point = list(hit_report["hit_endpoint_samples_world_xyz"][0])  # type: ignore[index]
    extra_point[0] += 0.01
    hit_report["input_point_count"] = 60  # type: ignore[index]
    hit_report["accepted_endpoint_count"] = 60  # type: ignore[index]
    hit_report["hit_endpoint_count"] = 60  # type: ignore[index]
    hit_report["hit_endpoint_samples_truncated"] = True  # type: ignore[index]
    hit_report["hit_endpoint_samples_world_xyz"].append(extra_point)  # type: ignore[index]
    hit_report["hit_endpoint_sample_voxel_indices_xyz"].append(  # type: ignore[index]
        list(hit_report["hit_endpoint_sample_voxel_indices_xyz"][0])  # type: ignore[index]
    )
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_sample_contract"
        for issue in report["errors"]
    )


def test_dynamic_rejects_single_point_bspline_contract() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    detour_report = simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
        "first_report"
    ]
    detour_report["trajectory_sample_count_total"] = 1  # type: ignore[index]
    detour_report["trajectory_samples_world_xyz"] = [  # type: ignore[index]
        detour_report["trajectory_samples_world_xyz"][0]  # type: ignore[index]
    ]
    detour = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "ordered_detour"
    ]
    detour["trajectory_samples_world_xyz"] = detour_report[  # type: ignore[index]
        "trajectory_samples_world_xyz"
    ]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_sample_contract"
        for issue in report["errors"]
    )


def test_dynamic_rejects_sparse_bspline_samples_claimed_as_truncated() -> None:
    summary = _base_summary("dynamic_f1")
    detour_report = summary["simulation_report"][  # type: ignore[index]
        "bspline_diagnostics_lifecycle_report"
    ]["first_report"]
    detour_report["trajectory_sample_count_total"] = 60  # type: ignore[index]
    detour_report["trajectory_samples_truncated"] = True  # type: ignore[index]
    detour_report["trajectory_samples_world_xyz"] = detour_report[  # type: ignore[index]
        "trajectory_samples_world_xyz"
    ][:2]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_sample_contract"
        for issue in report["errors"]
    )


def test_dynamic_rejects_tampered_sampling_clearance_margin() -> None:
    summary = _base_summary("dynamic_f1")
    detour_report = summary["simulation_report"][  # type: ignore[index]
        "bspline_diagnostics_lifecycle_report"
    ]["first_report"]
    detour_report["sampling_clearance_margin_m"] = 0.001  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["path"].endswith("sampling_clearance_margin_m")
        for issue in report["errors"]
    )


def test_dynamic_rejects_tampered_continuous_clearance_lower_bound() -> None:
    summary = _base_summary("dynamic_f1")
    clearance = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["current_obstacle_clearance"]["obstacle_clearances"][0]
    clearance["continuous_clearance_lower_bound_m"] += 0.1
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["path"].endswith("continuous_clearance_lower_bound_m")
        for issue in report["errors"]
    )


def test_dynamic_rejects_detour_tracking_after_clear() -> None:
    summary = _base_summary("dynamic_f1")
    evidence = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]
    detour = evidence["ordered_detour"]
    late_stamp_ns = (
        evidence["explicit_miss_ghost_clear"]["header"]["stamp_ns"]
        + 500_000_000
    )
    tracking = detour["controller_tracking_status"]
    tracking["header"]["stamp_ns"] = late_stamp_ns
    tracking["header"]["stamp"] = _stamp(late_stamp_ns)
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        and "controller_tracking_status" in issue["path"]
        for issue in report["errors"]
    )


def test_dynamic_rejects_detour_policy_write_at_or_after_clear_poll() -> None:
    summary = _base_summary("dynamic_f1")
    evidence = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]
    detour = evidence["ordered_detour"]
    detour["policy_identity_verified_tracking_write"]["timestamp"] = (
        summary["simulation_report"][  # type: ignore[index]
            "grid_map_observation_diagnostics_last_report"
        ]["receipt_timestamp"]
    )
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        and "policy_identity_verified_tracking_write" in issue["path"]
        for issue in report["errors"]
    )


def test_dynamic_rejects_detour_before_true_map_transition() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    lifecycle = simulation["grid_map_observation_lifecycle_report"]  # type: ignore[index]
    lifecycle["transition_hit_reports"] = []  # type: ignore[index]
    lifecycle["first_transition_hit_report"] = None  # type: ignore[index]
    lifecycle["last_transition_hit_report"] = None  # type: ignore[index]
    clear_report = simulation["grid_map_observation_diagnostics_last_report"]  # type: ignore[index]
    detour = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "ordered_detour"
    ]
    transition_stamp_ns = detour["header"]["stamp_ns"] + 100_000_000
    clear_report["occupied_to_free_transition_hit_header_stamp_ns"] = [  # type: ignore[index]
        transition_stamp_ns
    ]
    match = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "explicit_miss_ghost_clear"
    ]["clear_matches"][0]
    match["matched_hit_header"] = {  # type: ignore[index]
        "frame_id": "world",
        "stamp": _stamp(transition_stamp_ns),
        "stamp_ns": transition_stamp_ns,
    }
    plan = resolve_dynamic_obstacle_plan(summary["task_config"])
    match["obstacle_state_at_hit"] = plan.obstacles[0].state_at(  # type: ignore[index]
        transition_stamp_ns / 1e9
    ).to_dict()
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        and "causal_map_transition_clear_match" in issue["path"]
        for issue in report["errors"]
    )


def test_dynamic_allows_zero_provenance_on_unselected_clear_sample() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    lifecycle = simulation["grid_map_observation_lifecycle_report"]  # type: ignore[index]
    hit = lifecycle["first_report"]  # type: ignore[index]
    selected_clear = lifecycle["last_report"]  # type: ignore[index]
    selected_clear["observation_sequence"] = 3  # type: ignore[index]
    selected_clear["rx_sequence"] = 3  # type: ignore[index]
    unselected_clear = _grid_map_diagnostic_report(
        2,
        timestamp_s=9.5,
        hit_samples=[],
        clear_samples=[[-4.9, 4.2, 0.4]],
    )
    unselected_clear[
        "occupied_to_free_transition_hit_observation_sequences"
    ] = [0]
    unselected_clear[
        "occupied_to_free_transition_hit_header_stamp_ns"
    ] = [0]
    lifecycle["sample_count"] = 3  # type: ignore[index]
    lifecycle["last_observation_sequence"] = 3  # type: ignore[index]
    lifecycle["diagnostic_reports"] = [  # type: ignore[index]
        hit,
        unselected_clear,
        selected_clear,
    ]
    ghost = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "explicit_miss_ghost_clear"
    ]
    ghost["observation_sequence"] = 3  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert report["valid"] is True, report["errors"]


def test_dynamic_rejects_recovery_identity_reuse() -> None:
    summary = _base_summary("dynamic_f1")
    recovery = summary["simulation_report"]["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "trajectory_recovery"
    ]
    recovery["after_recovery_identity"] = deepcopy(  # type: ignore[index]
        recovery["before_detour_identity"]  # type: ignore[index]
    )
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_dynamic_recovery_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_recovery_above_deviation_limit() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    recovery_report = simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
        "last_report"
    ]
    recovery_report["maximum_trajectory_deviation_m"] = 0.03  # type: ignore[index]
    recovery = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "trajectory_recovery"
    ]
    recovery["after_maximum_trajectory_deviation_m"] = 0.03  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_dynamic_recovery_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_recovery_without_minimum_improvement() -> None:
    summary = _base_summary("dynamic_f1")
    simulation = summary["simulation_report"]
    detour_report = simulation["bspline_diagnostics_lifecycle_report"][  # type: ignore[index]
        "first_report"
    ]
    detour_report["maximum_trajectory_deviation_m"] = 0.025  # type: ignore[index]
    detour = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "ordered_detour"
    ]
    detour["maximum_trajectory_deviation_m"] = 0.025  # type: ignore[index]
    recovery = simulation["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "trajectory_recovery"
    ]
    recovery["before_maximum_trajectory_deviation_m"] = 0.025  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_dynamic_recovery_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_recovery_across_reference_generation() -> None:
    summary = _base_summary("dynamic_f1")
    recovery = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["trajectory_recovery"]
    recovery["same_reference_path_generation"] = False  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "missing_dynamic_recovery_evidence"
        for issue in report["errors"]
    )


def test_dynamic_rejects_policy_tracking_write_before_recovery_receipt() -> None:
    summary = _base_summary("dynamic_f1")
    recovery = summary["simulation_report"]["dynamic_navigation_evidence_report"][  # type: ignore[index]
        "trajectory_recovery"
    ]
    recovery["policy_identity_verified_tracking_write"]["timestamp"] = 6.0  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        for issue in report["errors"]
    )


def test_dynamic_rejects_policy_write_before_controller_tracking_receipt() -> None:
    summary = _base_summary("dynamic_f1")
    recovery = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["trajectory_recovery"]
    recovery["controller_tracking_status"]["receipt_timestamp"] = 13.0  # type: ignore[index]
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(
        issue["code"] == "invalid_dynamic_evidence_window"
        for issue in report["errors"]
    )


def test_dynamic_rejects_policy_write_controller_snapshot_mismatch() -> None:
    summary = _base_summary("dynamic_f1")
    policy_write = summary["simulation_report"][  # type: ignore[index]
        "dynamic_navigation_evidence_report"
    ]["trajectory_recovery"]["policy_identity_verified_tracking_write"]
    snapshot = deepcopy(policy_write["scan_controller_status_snapshot"])
    snapshot["identity"]["traj_id"] = 99
    policy_write["scan_controller_status_snapshot"] = snapshot
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    codes = {issue["code"] for issue in report["errors"]}
    assert "wrong_controller_acceptance_reference" in codes
    assert "missing_dynamic_recovery_evidence" in codes


def test_dynamic_global_replan_requires_new_plan_and_tracking_recovery() -> None:
    summary = _base_summary("dynamic_f1")
    lifecycle = summary["simulation_report"]["navigation_policy_gate_lifecycle_report"]  # type: ignore[index]
    first_replan = _observed_status_evidence(2, 2, tracking=False, write_sequence=4)
    first_status = first_replan["navigation_status_observed_report"]["status"]  # type: ignore[index]
    first_status.update(  # type: ignore[union-attr]
        {
            "state": 4,
            "global_replan_requested": True,
            "global_replan_request_id": 1,
            "reason": "trajectory_collision_ahead",
        }
    )
    last_replan = _observed_status_evidence(3, 3, tracking=False, write_sequence=5)
    last_status = last_replan["navigation_status_observed_report"]["status"]  # type: ignore[index]
    last_status.update(  # type: ignore[union-attr]
        {
            "state": 1,
            "global_replan_in_flight": True,
            "global_replan_request_id": 1,
            "pct_plan_id": 2,
            "reason": "pct_replan_in_flight",
        }
    )
    final_observed = _observed_status_evidence(5, 5, tracking=False, write_sequence=9)
    lifecycle.update(  # type: ignore[union-attr]
        {
            "observed_status_sequence_count": 5,
            "identity_valid_observed_status_count": 5,
            "last_identity_valid_observed_status": final_observed,
            "last_observed_status_sequence": 5,
            "last_observed_state": 6,
            "observed_state_transition_count": 3,
            "observed_state_counts": {
                "tracking": 2,
                "global_replan": 1,
                "global_planning": 1,
                "goal_reached": 1,
            },
            "observed_reason_counts": {
                "TRACKING": 2,
                "trajectory_collision_ahead": 1,
                "pct_replan_in_flight": 1,
                "GOAL_REACHED": 1,
            },
            "global_replan_requested_status_count": 1,
            "global_replan_in_flight_status_count": 1,
            "distinct_global_replan_request_ids": [1],
            "distinct_pct_plan_ids": [1, 2],
            "first_global_replan_status": first_replan,
            "last_global_replan_status": last_replan,
            "global_replan_pending_recovery": False,
            "tracking_after_global_replan_observed": True,
            "global_replan_recovery_count": 1,
            "last_observed_status": final_observed,
        }
    )
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert not any(issue["code"] == "incomplete_global_replan" for issue in report["errors"])

    lifecycle["distinct_pct_plan_ids"] = [1]  # type: ignore[index]
    lifecycle["tracking_after_global_replan_observed"] = False  # type: ignore[index]
    lifecycle["global_replan_recovery_count"] = 0  # type: ignore[index]
    lifecycle["global_replan_pending_recovery"] = True  # type: ignore[index]
    rejected = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(issue["code"] == "incomplete_global_replan" for issue in rejected["errors"])


def test_dynamic_emergency_requires_controller_tracking_recovery() -> None:
    summary = _base_summary("dynamic_f1")
    controller = summary["simulation_report"]["scan_controller_status_lifecycle_report"]  # type: ignore[index]
    emergency = _controller_status(
        2,
        1,
        state=8,
        event=4,
        reason="预测碰撞急停",
    )
    emergency["emergency_stop"] = True
    controller.update(  # type: ignore[union-attr]
        {
            "event_counts": {"accepted": 1, "state_changed": 2},
            "state_counts": {"tracking": 1, "emergency_stop": 1, "goal_reached": 1},
            "reason_counts": {
                "首条局部轨迹已接受": 1,
                "预测碰撞急停": 1,
                "目标已到达": 1,
            },
            "emergency_stop_status_count": 1,
            "first_emergency_stop_status": emergency,
            "last_emergency_stop_status": emergency,
            "tracking_after_emergency_stop_observed": True,
            "emergency_stop_recovery_count": 1,
            "emergency_stop_pending_recovery": False,
        }
    )
    report = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert not any(issue["code"] == "unrecovered_emergency_stop" for issue in report["errors"])
    controller["tracking_after_emergency_stop_observed"] = False  # type: ignore[index]
    controller["emergency_stop_recovery_count"] = 0  # type: ignore[index]
    controller["emergency_stop_pending_recovery"] = True  # type: ignore[index]
    rejected = validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert any(issue["code"] == "unrecovered_emergency_stop" for issue in rejected["errors"])


def test_strict_loader_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"success": true, "success": false}', encoding="utf-8")
    with pytest.raises(SummaryInputError, match="重复键"):
        load_summary(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(SummaryInputError, match="非有限"):
        load_summary(nonfinite)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    valid_path = tmp_path / "summary.json"
    valid_path.write_text(json.dumps(_base_summary("flat_policy")), encoding="utf-8")
    assert main([str(valid_path), "--mode", "flat_policy"]) == 0
    assert "PASS [flat_policy]" in capsys.readouterr().out
    invalid = _base_summary("flat_policy")
    invalid["success"] = False
    valid_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert main([str(valid_path), "--mode", "flat_policy"]) == 1
    assert "FAIL [flat_policy]" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.json"), "--mode", "flat_policy"]) == 2


def test_cli_requires_active_sensing_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = _base_summary("dynamic_f1")
    _add_active_sensing_report(summary)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    assert main(
        [
            str(summary_path),
            "--mode",
            "dynamic_f1",
            "--require-active-sensing",
        ]
    ) == 0
    assert "PASS [dynamic_f1]" in capsys.readouterr().out

    summary_path.write_text(
        json.dumps(_base_summary("dynamic_f1")),
        encoding="utf-8",
    )
    assert main(
        [
            str(summary_path),
            "--mode",
            "dynamic_f1",
            "--require-active-sensing",
        ]
    ) == 1
    assert "FAIL [dynamic_f1]" in capsys.readouterr().err


def test_input_mapping_is_not_mutated() -> None:
    summary = _base_summary("dynamic_f1")
    before = deepcopy(summary)
    validate_pct_scan_live_summary(summary, "dynamic_f1")
    assert summary == before
