from __future__ import annotations

import hashlib
import json
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

import source.navigation.isaac_ros2_ogn_bridge as bridge_module
from source.navigation.cmd_vel_to_policy import PolicyCommandInput
from source.navigation.isaac_ros2_ogn_bridge import (
    CLOCK_QOS_PROFILE,
    CMD_VEL_QOS_PROFILE,
    CONTROLLER_STATUS_QOS_PROFILE,
    PLANNING_DIAGNOSTICS_QOS_PROFILE,
    NAVIGATION_STATUS_QOS_PROFILE,
    GOAL_REACHED_QOS_PROFILE,
    PCT_GOAL_QOS_PROFILE,
    REFERENCE_PATH_QOS_PROFILE,
    SENSOR_DATA_QOS_PROFILE,
    STAIR_EXECUTION_FROZEN_QOS_PROFILE,
    IsaacRos2OgnBridge,
    IsaacRos2OgnBridgeConfig,
    OgnStairExecutionFreezePublicationReport,
    build_graph_spec,
    enable_ros2_bridge_extension,
    parse_controller_status_outputs,
    parse_bspline_diagnostics_outputs,
    parse_grid_map_observation_diagnostics_outputs,
    parse_navigation_status_outputs,
    parse_reference_path_outputs,
    prepare_odometry_sample,
    validate_point_cloud,
)


def _path_outputs(
    points: list[tuple[float, float, float]],
    *,
    frame_id: str = "world",
    sec: int = 12,
    nanosec: int = 345,
    terminal_yaw: float = 0.0,
) -> tuple[int, int, str, list[str]]:
    poses = [
        json.dumps(
            {
                "header": {
                    "stamp": {"sec": sec, "nanosec": nanosec},
                    "frame_id": frame_id,
                },
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": math.sin(
                            0.5 * terminal_yaw
                            if index == len(points) - 1
                            else 0.0
                        ),
                        "w": math.cos(
                            0.5 * terminal_yaw
                            if index == len(points) - 1
                            else 0.0
                        ),
                    },
                },
            }
        )
        for index, (x, y, z) in enumerate(points)
    ]
    return sec, nanosec, frame_id, poses


def _controller_status_outputs(
    **overrides: object,
) -> dict[str, object]:
    outputs: dict[str, object] = {
        "header_stamp_sec": 20,
        "header_stamp_nanosec": 1,
        "header_frame_id": "world",
        "status_sequence": 7,
        "acceptance_sequence": 3,
        "event": 1,
        "reference_path_stamp_sec": 18,
        "reference_path_stamp_nanosec": 999_999_999,
        "bspline_header_stamp_sec": 19,
        "bspline_header_stamp_nanosec": 2,
        "start_time_sec": 19,
        "start_time_nanosec": 3,
        "traj_id": 42,
        "accepted": True,
        "trajectory_valid": True,
        "is_final": False,
        "emergency_stop": False,
        "state": 10,
        "reason": "B-spline 已接受",
        "candidate_present": False,
        "candidate_reference_path_stamp_sec": 0,
        "candidate_reference_path_stamp_nanosec": 0,
        "candidate_bspline_header_stamp_sec": 0,
        "candidate_bspline_header_stamp_nanosec": 0,
        "candidate_start_time_sec": 0,
        "candidate_start_time_nanosec": 0,
        "candidate_traj_id": 0,
        "active_sensing_yaw_only": False,
        "command_sample_count": 0,
        "first_command_linear_x": 0.0,
        "first_command_linear_y": 0.0,
        "first_command_linear_z": 0.0,
        "first_command_angular_x": 0.0,
        "first_command_angular_y": 0.0,
        "first_command_angular_z": 0.0,
        "max_abs_vx": 0.0,
        "max_abs_vy": 0.0,
        "max_abs_wz": 0.0,
        "command_violation_count": 0,
    }
    outputs.update(overrides)
    return outputs


def _point_json(x: float, y: float, z: float) -> str:
    return json.dumps({"x": x, "y": y, "z": z})


def _grid_map_diagnostics_outputs(**overrides: object) -> dict[str, object]:
    outputs: dict[str, object] = {
        "header_stamp_sec": 30,
        "header_stamp_nanosec": 5,
        "header_frame_id": "world",
        "observation_sequence": 9,
        "sensor_pose_stamp_sec": 30,
        "sensor_pose_stamp_nanosec": 4,
        "sensor_origin_x": -4.0,
        "sensor_origin_y": 3.0,
        "sensor_origin_z": 0.4,
        "canonical_empty": False,
        "map_fusion_performed": True,
        "map_resolution": 0.05,
        "input_point_count": 2,
        "accepted_endpoint_count": 2,
        "hit_endpoint_count": 1,
        "explicit_free_endpoint_count": 1,
        "hit_endpoint_samples_truncated": False,
        "hit_endpoint_samples": [_point_json(-4.0, 4.2, 0.5)],
        "hit_endpoint_sample_voxel_indices_xyz": [-80, 84, 10],
        "free_to_occupied_transition_count": 1,
        "free_to_occupied_transition_samples_truncated": False,
        "free_to_occupied_transition_hit_samples": [
            _point_json(-4.0, 4.2, 0.5)
        ],
        "free_to_occupied_transition_voxel_indices_xyz": [-80, 84, 10],
        "explicit_free_miss_voxel_count": 3,
        "occupied_to_free_by_explicit_miss_count": 1,
        "occupied_to_free_samples_truncated": False,
        "occupied_to_free_by_explicit_miss_samples": [
            _point_json(-4.4, 4.2, 0.5)
        ],
        "occupied_to_free_sample_voxel_indices_xyz": [-88, 84, 10],
        "occupied_to_free_transition_hit_observation_sequences": [8],
        "occupied_to_free_transition_hit_samples": [
            _point_json(-4.4, 4.2, 0.5)
        ],
        "occupied_to_free_transition_hit_header_stamp_ns": [
            29_000_000_005
        ],
        "occupied_removed_by_sliding_reset_count": 0,
    }
    outputs.update(overrides)
    return outputs


def _bspline_diagnostics_outputs(**overrides: object) -> dict[str, object]:
    trajectory = [
        _point_json(-4.2, 3.8, 0.4),
        _point_json(-4.1, 4.2, 0.4),
        _point_json(-3.8, 4.5, 0.4),
    ]
    reference = [
        _point_json(-4.2, 3.8, 0.1),
        _point_json(-4.0, 4.2, 0.1),
        _point_json(-3.8, 4.5, 0.1),
    ]
    outputs: dict[str, object] = {
        "header_stamp_sec": 31,
        "header_stamp_nanosec": 2,
        "header_frame_id": "world",
        "diagnostic_sequence": 4,
        "start_time_sec": 31,
        "start_time_nanosec": 3,
        "reference_path_stamp_sec": 30,
        "reference_path_stamp_nanosec": 1,
        "traj_id": 7,
        "is_final": False,
        "emergency_stop": False,
        "stationary": False,
        "ordered_reference_checked": True,
        "ordered_reference_safe": True,
        "maximum_trajectory_deviation": 0.08,
        "maximum_guide_anchor_deviation": 0.07,
        "maximum_guide_progress_lead": 0.01,
        "maximum_deviation_limit": 0.10,
        "maximum_progress_lead_limit": 0.02,
        "trajectory_duration": 0.02,
        "maximum_velocity_upper_bound": 1.0,
        "double_cylinder_radius": 0.27,
        "double_cylinder_offset": 0.16,
        "trajectory_sample_count_total": len(trajectory),
        "trajectory_samples_truncated": False,
        "trajectory_samples": trajectory,
        "ordered_reference_sample_count_total": len(reference),
        "ordered_reference_samples_truncated": False,
        "ordered_reference_samples": reference,
        "active_sensing": False,
        "active_sensing_event": 0,
        "active_sensing_start_yaw": 0.0,
        "active_sensing_target_yaw": 0.0,
        "active_sensing_yaw_offset": 0.0,
        "active_sensing_yaw_rate": 0.0,
        "active_sensing_settle_stamp_sec": 0,
        "active_sensing_settle_stamp_nanosec": 0,
        "active_sensing_settle_yaw_error": 0.0,
        "active_sensing_settle_angular_speed": 0.0,
        "active_sensing_stable_duration": 0.0,
        "active_sensing_fusion_baseline": 0,
        "active_sensing_fusion_current": 0,
        "active_sensing_fusion_distinct": 0,
        "active_sensing_fusion_required": 0,
        "active_sensing_completed": False,
        "active_sensing_failed": False,
        "active_sensing_reason": "",
    }
    outputs.update(overrides)
    return outputs


def _navigation_status_outputs(
    **overrides: object,
) -> dict[str, object]:
    outputs: dict[str, object] = {
        "header_stamp_sec": 12,
        "header_stamp_nanosec": 345,
        "header_frame_id": "world",
        "status_sequence": 7,
        "state_revision": 3,
        "goal_id": 10_000_000_001,
        "state": 3,
        "allow_tracking_command": True,
        "force_zero_velocity": False,
        "stop_confirmed": False,
        "global_replan_requested": False,
        "global_replan_in_flight": False,
        "global_replan_request_id": 2,
        "pct_plan_id": 4,
        "active_path_stamp_sec": 12,
        "active_path_stamp_nanosec": 345,
        "consecutive_scan_failures": 0,
        "stale_inputs": [],
        "reason": "跟踪输入有效",
    }
    outputs.update(overrides)
    return outputs


def _connections(spec: bridge_module.OgnGraphSpec) -> set[tuple[str, str]]:
    return set(spec.connections)


def test_pure_graph_spec_does_not_import_isaac_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_import(name: str) -> object:
        raise AssertionError(f"不应导入 {name}")

    monkeypatch.setattr(bridge_module.importlib, "import_module", _unexpected_import)
    assert IsaacRos2OgnBridgeConfig().odometry_source == "direct"
    assert "ROS2PublishOdometry" in build_graph_spec().node_types()


def test_default_direct_graph_uses_independent_impulses_and_raw_topics() -> None:
    config = IsaacRos2OgnBridgeConfig()
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()

    assert nodes["ROS2Context"] == "isaacsim.ros2.bridge.ROS2Context"
    assert nodes["StateTx"] == "omni.graph.action.OnImpulseEvent"
    assert nodes["CloudTx"] == "omni.graph.action.OnImpulseEvent"
    assert nodes["IsaacReadSimulationTime"] == "isaacsim.core.nodes.IsaacReadSimulationTime"
    assert nodes["ROS2PublishClock"] == "isaacsim.ros2.bridge.ROS2PublishClock"
    assert nodes["ROS2PublishOdometry"] == "isaacsim.ros2.bridge.ROS2PublishOdometry"
    assert nodes["ROS2PublishPointCloud"] == "isaacsim.ros2.bridge.ROS2PublishPointCloud"
    assert "IsaacComputeOdometry" not in nodes
    assert "ROS2QoSProfile" not in nodes

    assert values["ROS2PublishClock.inputs:topicName"] == "/clock"
    assert values["ROS2PublishOdometry.inputs:topicName"] == "/isaac/body_pose_raw"
    assert (
        values["ROS2PublishPointCloud.inputs:topicName"]
        == "/isaac/cloud_registered_raw"
    )
    assert values["ROS2PublishOdometry.inputs:odomFrameId"] == "world"
    assert values["ROS2PublishOdometry.inputs:chassisFrameId"] == "base_link"
    assert values["ROS2PublishPointCloud.inputs:frameId"] == "world"
    assert values["ROS2Context.inputs:useDomainIDEnvVar"] is True
    assert values["ROS2PublishOdometry.inputs:publishRawVelocities"] is True

    connections = _connections(spec)
    assert (
        "StateTx.outputs:execOut",
        "ROS2PublishOdometry.inputs:execIn",
    ) in connections
    assert (
        "CloudTx.outputs:execOut",
        "ROS2PublishPointCloud.inputs:execIn",
    ) in connections
    assert (
        "IsaacReadSimulationTime.outputs:simulationTime",
        "ROS2PublishPointCloud.inputs:timeStamp",
    ) not in connections
    assert not any(
        source.startswith("IsaacComputeOdometry.outputs:")
        for source, _ in connections
    )


def test_sensor_data_qos_json_contains_all_isaac_51_fields() -> None:
    qos = json.loads(SENSOR_DATA_QOS_PROFILE)

    assert qos == {
        "history": "keepLast",
        "depth": 5,
        "reliability": "bestEffort",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
    values = build_graph_spec().values()
    assert values["ROS2PublishOdometry.inputs:qosProfile"] == SENSOR_DATA_QOS_PROFILE
    assert values["ROS2PublishPointCloud.inputs:qosProfile"] == SENSOR_DATA_QOS_PROFILE


def test_clock_uses_best_effort_depth_one_qos() -> None:
    qos = json.loads(CLOCK_QOS_PROFILE)
    values = build_graph_spec().values()

    assert qos["reliability"] == "bestEffort"
    assert qos["durability"] == "volatile"
    assert qos["depth"] == 1
    assert values["ROS2PublishClock.inputs:qosProfile"] == CLOCK_QOS_PROFILE
    assert values["ROS2PublishClock.inputs:queueSize"] == 1


def test_optional_cmd_vel_subscriber_uses_reliable_qos_and_counter() -> None:
    config = IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["CommandRxTick"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2SubscribeTwist"]
        == "isaacsim.ros2.bridge.ROS2SubscribeTwist"
    )
    assert nodes["CommandRxCounter"] == "omni.graph.action.Counter"
    assert values["ROS2SubscribeTwist.inputs:topicName"] == "/cmd_vel"
    assert (
        values["ROS2SubscribeTwist.inputs:qosProfile"]
        == CMD_VEL_QOS_PROFILE
    )
    assert json.loads(CMD_VEL_QOS_PROFILE)["reliability"] == "reliable"
    assert (
        "CommandRxTick.outputs:execOut",
        "ROS2SubscribeTwist.inputs:execIn",
    ) in connections
    assert (
        "ROS2SubscribeTwist.outputs:execOut",
        "CommandRxCounter.inputs:execIn",
    ) in connections
    assert nodes["NavigationStatusRxTick"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2SubscribeNavigationStatus"]
        == "isaacsim.ros2.bridge.ROS2Subscriber"
    )
    assert nodes["NavigationStatusRxCounter"] == "omni.graph.action.Counter"
    assert (
        values["ROS2SubscribeNavigationStatus.inputs:messageName"]
        == "NavigationStatus"
    )
    assert (
        values["ROS2SubscribeNavigationStatus.inputs:topicName"]
        == "/navigation/status"
    )
    assert (
        values["ROS2SubscribeNavigationStatus.inputs:qosProfile"]
        == NAVIGATION_STATUS_QOS_PROFILE
    )
    assert json.loads(NAVIGATION_STATUS_QOS_PROFILE)["durability"] == (
        "transientLocal"
    )
    assert (
        "ROS2SubscribeNavigationStatus.outputs:execOut",
        "NavigationStatusRxCounter.inputs:execIn",
    ) in connections


def test_optional_goal_reached_subscriber_uses_generic_bool_and_counter() -> None:
    config = IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["GoalReachedRxTick"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2SubscribeGoalReached"]
        == "isaacsim.ros2.bridge.ROS2Subscriber"
    )
    assert nodes["GoalReachedRxCounter"] == "omni.graph.action.Counter"
    assert values["ROS2SubscribeGoalReached.inputs:messagePackage"] == "std_msgs"
    assert values["ROS2SubscribeGoalReached.inputs:messageSubfolder"] == "msg"
    assert values["ROS2SubscribeGoalReached.inputs:messageName"] == "Bool"
    assert (
        values["ROS2SubscribeGoalReached.inputs:topicName"]
        == "/planning/goal_reached"
    )
    assert (
        values["ROS2SubscribeGoalReached.inputs:qosProfile"]
        == GOAL_REACHED_QOS_PROFILE
    )
    assert values["ROS2SubscribeGoalReached.inputs:queueSize"] == 1
    assert json.loads(GOAL_REACHED_QOS_PROFILE) == {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
    assert (
        "ROS2Context.outputs:context",
        "ROS2SubscribeGoalReached.inputs:context",
    ) in connections
    assert (
        "GoalReachedRxTick.outputs:execOut",
        "ROS2SubscribeGoalReached.inputs:execIn",
    ) in connections
    assert (
        "ROS2SubscribeGoalReached.outputs:execOut",
        "GoalReachedRxCounter.inputs:execIn",
    ) in connections


def test_optional_controller_status_uses_typed_transient_snapshot() -> None:
    config = IsaacRos2OgnBridgeConfig(
        enable_controller_status_subscription=True,
    )
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["ControllerStatusRxTick"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2SubscribeControllerStatus"]
        == "isaacsim.ros2.bridge.ROS2Subscriber"
    )
    assert nodes["ControllerStatusRxCounter"] == "omni.graph.action.Counter"
    assert (
        values["ROS2SubscribeControllerStatus.inputs:messagePackage"]
        == "scan_planner_msgs"
    )
    assert values["ROS2SubscribeControllerStatus.inputs:messageSubfolder"] == "msg"
    assert (
        values["ROS2SubscribeControllerStatus.inputs:messageName"]
        == "ControllerStatus"
    )
    assert (
        values["ROS2SubscribeControllerStatus.inputs:topicName"]
        == "/planning/controller_status"
    )
    assert (
        values["ROS2SubscribeControllerStatus.inputs:qosProfile"]
        == CONTROLLER_STATUS_QOS_PROFILE
    )
    assert values["ROS2SubscribeControllerStatus.inputs:queueSize"] == 64
    assert json.loads(CONTROLLER_STATUS_QOS_PROFILE) == {
        "history": "keepLast",
        "depth": 64,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
    assert (
        "ROS2Context.outputs:context",
        "ROS2SubscribeControllerStatus.inputs:context",
    ) in connections
    assert (
        "ControllerStatusRxTick.outputs:execOut",
        "ROS2SubscribeControllerStatus.inputs:execIn",
    ) in connections
    assert (
        "ROS2SubscribeControllerStatus.outputs:execOut",
        "ControllerStatusRxCounter.inputs:execIn",
    ) in connections


def test_optional_planning_diagnostics_use_bounded_typed_snapshot_queues() -> None:
    config = IsaacRos2OgnBridgeConfig(
        enable_grid_map_diagnostics_subscription=True,
        enable_bspline_diagnostics_subscription=True,
    )
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    diagnostics_qos = json.loads(PLANNING_DIAGNOSTICS_QOS_PROFILE)
    assert diagnostics_qos["reliability"] == "reliable"
    assert diagnostics_qos["durability"] == "transientLocal"
    assert diagnostics_qos["history"] == "keepLast"
    assert diagnostics_qos["depth"] == 64
    for label, message_name, topic in (
        (
            "GridMapDiagnostics",
            "GridMapObservationDiagnostics",
            "/planning/grid_map_observation_diagnostics",
        ),
        (
            "BsplineDiagnostics",
            "BsplineDiagnostics",
            "/planning/bspline_diagnostics",
        ),
    ):
        subscriber = f"ROS2Subscribe{label}"
        tick = f"{label}RxTick"
        counter = f"{label}RxCounter"
        assert nodes[tick] == "omni.graph.action.OnImpulseEvent"
        assert nodes[subscriber] == "isaacsim.ros2.bridge.ROS2Subscriber"
        assert nodes[counter] == "omni.graph.action.Counter"
        assert values[f"{subscriber}.inputs:messagePackage"] == (
            "scan_planner_msgs"
        )
        assert values[f"{subscriber}.inputs:messageName"] == message_name
        assert values[f"{subscriber}.inputs:topicName"] == topic
        assert values[f"{subscriber}.inputs:qosProfile"] == (
            PLANNING_DIAGNOSTICS_QOS_PROFILE
        )
        assert values[f"{subscriber}.inputs:queueSize"] == 64
        assert (
            "ROS2Context.outputs:context",
            f"{subscriber}.inputs:context",
        ) in connections
        assert (
            f"{tick}.outputs:execOut",
            f"{subscriber}.inputs:execIn",
        ) in connections
        assert (
            f"{subscriber}.outputs:execOut",
            f"{counter}.inputs:execIn",
        ) in connections


def test_default_reference_path_subscriber_uses_transient_reliable_qos() -> None:
    spec = build_graph_spec()
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["ReferencePathRxTick"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2SubscribeReferencePath"]
        == "isaacsim.ros2.bridge.ROS2Subscriber"
    )
    assert nodes["ReferencePathRxCounter"] == "omni.graph.action.Counter"
    assert values["ROS2SubscribeReferencePath.inputs:messagePackage"] == "nav_msgs"
    assert values["ROS2SubscribeReferencePath.inputs:messageSubfolder"] == "msg"
    assert values["ROS2SubscribeReferencePath.inputs:messageName"] == "Path"
    assert values["ROS2SubscribeReferencePath.inputs:topicName"] == "/initial_path"
    assert (
        values["ROS2SubscribeReferencePath.inputs:qosProfile"]
        == REFERENCE_PATH_QOS_PROFILE
    )
    assert json.loads(REFERENCE_PATH_QOS_PROFILE)["reliability"] == "reliable"
    assert json.loads(REFERENCE_PATH_QOS_PROFILE)["durability"] == "transientLocal"
    assert (
        "ReferencePathRxTick.outputs:execOut",
        "ROS2SubscribeReferencePath.inputs:execIn",
    ) in connections
    assert (
        "ROS2SubscribeReferencePath.outputs:execOut",
        "ReferencePathRxCounter.inputs:execIn",
    ) in connections


def test_optional_pct_goal_publisher_uses_pose_stamped_and_volatile_qos() -> None:
    config = IsaacRos2OgnBridgeConfig(enable_pct_goal_publisher=True)
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["PCTGoalTx"] == "omni.graph.action.OnImpulseEvent"
    assert (
        nodes["ROS2PublishPCTGoal"]
        == "isaacsim.ros2.bridge.ROS2Publisher"
    )
    assert values["ROS2PublishPCTGoal.inputs:messagePackage"] == "geometry_msgs"
    assert values["ROS2PublishPCTGoal.inputs:messageSubfolder"] == "msg"
    assert values["ROS2PublishPCTGoal.inputs:messageName"] == "PoseStamped"
    assert values["ROS2PublishPCTGoal.inputs:topicName"] == "/pct/goal"
    assert values["ROS2PublishPCTGoal.inputs:qosProfile"] == PCT_GOAL_QOS_PROFILE
    assert json.loads(PCT_GOAL_QOS_PROFILE)["reliability"] == "reliable"
    assert json.loads(PCT_GOAL_QOS_PROFILE)["durability"] == "volatile"
    assert (
        "ROS2Context.outputs:context",
        "ROS2PublishPCTGoal.inputs:context",
    ) in connections
    assert (
        "PCTGoalTx.outputs:execOut",
        "ROS2PublishPCTGoal.inputs:execIn",
    ) in connections


def test_optional_stair_frozen_publisher_is_typed_identity_snapshot() -> None:
    config = IsaacRos2OgnBridgeConfig(
        enable_stair_execution_frozen_publisher=True,
    )
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["StairExecutionFrozenTx"] == (
        "omni.graph.action.OnImpulseEvent"
    )
    assert nodes["ROS2PublishStairExecutionFrozen"] == (
        "isaacsim.ros2.bridge.ROS2Publisher"
    )
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:messagePackage"
    ] == "scan_planner_msgs"
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:messageSubfolder"
    ] == "msg"
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:messageName"
    ] == "StairExecutionFreeze"
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:topicName"
    ] == "/planning/stair_execution_frozen"
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:qosProfile"
    ] == STAIR_EXECUTION_FROZEN_QOS_PROFILE
    assert json.loads(STAIR_EXECUTION_FROZEN_QOS_PROFILE) == {
        "history": "keepLast",
        "depth": 1,
        "reliability": "reliable",
        "durability": "transientLocal",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
    assert (
        "ROS2Context.outputs:context",
        "ROS2PublishStairExecutionFrozen.inputs:context",
    ) in connections
    assert (
        "StairExecutionFrozenTx.outputs:execOut",
        "ROS2PublishStairExecutionFrozen.inputs:execIn",
    ) in connections
    # controller 的冻结状态是另一个 ROS writer，不能被本发布器复用。
    assert values[
        "ROS2PublishStairExecutionFrozen.inputs:topicName"
    ] != "/planning/go2_execution_frozen"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durability", "volatile"),
        ("reliability", "bestEffort"),
        ("history", "keepAll"),
        ("depth", 63),
    ],
)
def test_stair_frozen_qos_must_match_snapshot_contract(
    field: str,
    value: object,
) -> None:
    qos = json.loads(STAIR_EXECUTION_FROZEN_QOS_PROFILE)
    qos[field] = value

    with pytest.raises(ValueError, match="reliable.*transientLocal.*keepLast"):
        IsaacRos2OgnBridgeConfig(
            stair_execution_frozen_qos_profile=json.dumps(qos),
        )


@pytest.mark.parametrize("writer_id", ("", " writer", "writer "))
def test_stair_frozen_writer_id_must_be_nonempty_and_trimmed(
    writer_id: str,
) -> None:
    with pytest.raises(ValueError, match="writer_id.*非空"):
        IsaacRos2OgnBridgeConfig(
            stair_execution_frozen_writer_id=writer_id,
        )


def test_reference_path_subscription_can_only_be_disabled_explicitly() -> None:
    spec = build_graph_spec(
        IsaacRos2OgnBridgeConfig(enable_reference_path_subscription=False)
    )

    assert "ROS2SubscribeReferencePath" not in spec.node_types()
    assert not any(
        "ReferencePath" in endpoint
        for connection in spec.connections
        for endpoint in connection
    )


def test_reference_path_qos_cannot_drop_reliability_or_durability() -> None:
    qos = json.loads(REFERENCE_PATH_QOS_PROFILE)
    qos["durability"] = "volatile"

    with pytest.raises(ValueError, match="reliable.*transientLocal"):
        IsaacRos2OgnBridgeConfig(
            reference_path_qos_profile=json.dumps(qos),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durability", "volatile"),
        ("reliability", "bestEffort"),
        ("history", "keepAll"),
        ("depth", 2),
    ],
)
def test_controller_status_qos_must_match_snapshot_contract(
    field: str,
    value: object,
) -> None:
    qos = json.loads(CONTROLLER_STATUS_QOS_PROFILE)
    qos[field] = value

    with pytest.raises(ValueError, match="reliable.*transientLocal.*keepLast"):
        IsaacRos2OgnBridgeConfig(
            controller_status_qos_profile=json.dumps(qos),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durability", "volatile"),
        ("reliability", "bestEffort"),
        ("history", "keepAll"),
        ("depth", 63),
    ],
)
def test_planning_diagnostics_qos_must_preserve_bounded_history(
    field: str,
    value: object,
) -> None:
    qos = json.loads(PLANNING_DIAGNOSTICS_QOS_PROFILE)
    qos[field] = value

    with pytest.raises(ValueError, match="reliable.*transientLocal.*keepLast"):
        IsaacRos2OgnBridgeConfig(
            planning_diagnostics_qos_profile=json.dumps(qos),
        )


def test_default_graph_does_not_create_cmd_vel_subscriber() -> None:
    spec = build_graph_spec()

    assert "ROS2SubscribeTwist" not in spec.node_types()
    assert not any(
        "ROS2SubscribeTwist" in endpoint
        for connection in spec.connections
        for endpoint in connection
    )


def test_default_graph_does_not_create_goal_reached_subscriber() -> None:
    spec = build_graph_spec()

    assert "ROS2SubscribeGoalReached" not in spec.node_types()
    assert not any(
        "GoalReached" in endpoint
        for connection in spec.connections
        for endpoint in connection
    )


def test_default_graph_does_not_create_controller_status_subscriber() -> None:
    spec = build_graph_spec()

    assert "ROS2SubscribeControllerStatus" not in spec.node_types()
    assert not any(
        "ControllerStatus" in endpoint
        for connection in spec.connections
        for endpoint in connection
    )


def test_default_graph_does_not_create_planning_diagnostics_subscribers() -> None:
    spec = build_graph_spec()

    assert "ROS2SubscribeGridMapDiagnostics" not in spec.node_types()
    assert "ROS2SubscribeBsplineDiagnostics" not in spec.node_types()
    assert not any(
        "Diagnostics" in endpoint
        for connection in spec.connections
        for endpoint in connection
    )


def test_compute_mode_adds_target_and_connects_pose_velocity_outputs() -> None:
    config = IsaacRos2OgnBridgeConfig(
        robot_prim_path="/World/envs/env_3/Robot",
        odometry_source="compute",
    )
    spec = build_graph_spec(config)
    nodes = spec.node_types()
    values = spec.values()
    connections = _connections(spec)

    assert nodes["IsaacComputeOdometry"] == "isaacsim.core.nodes.IsaacComputeOdometry"
    assert (
        values["IsaacComputeOdometry.inputs:chassisPrim"]
        == "/World/envs/env_3/Robot"
    )
    assert (
        "StateTx.outputs:execOut",
        "IsaacComputeOdometry.inputs:execIn",
    ) in connections
    assert (
        "IsaacComputeOdometry.outputs:position",
        "ROS2PublishOdometry.inputs:position",
    ) in connections
    assert (
        "IsaacComputeOdometry.outputs:orientation",
        "ROS2PublishOdometry.inputs:orientation",
    ) in connections
    assert (
        "IsaacComputeOdometry.outputs:linearVelocity",
        "ROS2PublishOdometry.inputs:linearVelocity",
    ) in connections
    assert (
        "IsaacComputeOdometry.outputs:angularVelocity",
        "ROS2PublishOdometry.inputs:angularVelocity",
    ) in connections
    assert (
        "StateTx.outputs:execOut",
        "ROS2PublishOdometry.inputs:execIn",
    ) not in connections
    assert (
        "IsaacReadSimulationTime.outputs:simulationTime",
        "ROS2PublishPointCloud.inputs:timeStamp",
    ) in connections
    assert values["ROS2PublishOdometry.inputs:publishRawVelocities"] is False


def test_explicit_domain_disables_environment_lookup() -> None:
    values = build_graph_spec(IsaacRos2OgnBridgeConfig(domain_id=17)).values()

    assert values["ROS2Context.inputs:domain_id"] == 17
    assert values["ROS2Context.inputs:useDomainIDEnvVar"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clock_topic", "clock"),
        ("odometry_topic", "//body_pose"),
        ("point_cloud_topic", "/cloud/"),
        ("command_topic", "cmd_vel"),
        ("goal_reached_topic", "planning/goal_reached"),
        ("controller_status_topic", "planning/controller_status"),
        ("grid_map_diagnostics_topic", "planning/grid_map_diagnostics"),
        ("bspline_diagnostics_topic", "planning/bspline_diagnostics"),
        ("reference_path_topic", "initial_path"),
        ("odom_frame_id", "/world"),
        ("base_frame_id", ""),
        ("point_cloud_frame_id", "bad frame"),
        ("point_cloud_frame_id", "map"),
        ("graph_path", "World/Graph"),
        ("robot_prim_path", "/World/bad-name"),
    ],
)
def test_config_rejects_invalid_topics_frames_and_prim_paths(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        IsaacRos2OgnBridgeConfig(**{field: value})


@pytest.mark.parametrize("domain_id", [-1, 233])
def test_config_rejects_out_of_range_domain_id(domain_id: int) -> None:
    with pytest.raises(ValueError, match="domain_id"):
        IsaacRos2OgnBridgeConfig(domain_id=domain_id)


@pytest.mark.parametrize(
    "field",
    [
        "graph_backed_by_usd",
        "enable_command_subscription",
        "enable_goal_reached_subscription",
        "enable_controller_status_subscription",
        "enable_grid_map_diagnostics_subscription",
        "enable_bspline_diagnostics_subscription",
        "enable_reference_path_subscription",
    ],
)
def test_config_rejects_non_bool_subscription_flags(field: str) -> None:
    with pytest.raises(TypeError, match="布尔值"):
        IsaacRos2OgnBridgeConfig(**{field: 1})


def test_point_cloud_validation_returns_contiguous_float32_copy() -> None:
    source = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    result = validate_point_cloud(source)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert not np.shares_memory(source, result)
    assert result == pytest.approx(source)


def test_point_cloud_validation_rejects_empty_raw_observation() -> None:
    source = np.empty((0, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="原始点云不能为空"):
        validate_point_cloud(source)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 3, 1), dtype=np.float32),
    ],
)
def test_point_cloud_validation_rejects_non_nx3_shapes(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        validate_point_cloud(points)


def test_point_cloud_validation_rejects_non_float_and_non_finite_data() -> None:
    with pytest.raises(TypeError, match="浮点"):
        validate_point_cloud(np.zeros((2, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="NaN"):
        validate_point_cloud(np.array([[0.0, np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        validate_point_cloud(np.array([[0.0, np.inf, 1.0]], dtype=np.float32))


def test_prepare_odometry_reorders_wxyz_to_ogn_ijkr_without_axis_changes() -> None:
    sample = prepare_odometry_sample(
        position=(1.0, 2.0, 3.0),
        orientation_wxyz=(0.5, 0.1, 0.2, 0.3),
        linear_velocity=(4.0, 5.0, 6.0),
        angular_velocity=(0.4, 0.5, 0.6),
        timestamp=12.25,
    )

    assert sample.position == (1.0, 2.0, 3.0)
    norm = math.sqrt(0.5**2 + 0.1**2 + 0.2**2 + 0.3**2)
    assert sample.orientation_ijkr == pytest.approx(
        (0.1 / norm, 0.2 / norm, 0.3 / norm, 0.5 / norm)
    )
    assert sample.linear_velocity == (4.0, 5.0, 6.0)
    assert sample.angular_velocity == (0.4, 0.5, 0.6)
    assert sample.timestamp == 12.25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": (1.0, 2.0)},
        {"orientation_wxyz": (0.0, 0.0, 0.0, 0.0)},
        {"linear_velocity": (0.0, float("nan"), 0.0)},
        {"angular_velocity": (0.0, 0.0, float("inf"))},
        {"timestamp": -0.01},
        {"timestamp": 0.0},
    ],
)
def test_prepare_odometry_rejects_invalid_samples(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "position": (1.0, 2.0, 3.0),
        "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "linear_velocity": (0.0, 0.0, 0.0),
        "angular_velocity": (0.0, 0.0, 0.0),
        "timestamp": 1.0,
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        prepare_odometry_sample(**arguments)


def test_parse_reference_path_preserves_ground_z_and_hashes_geometry() -> None:
    points = [(0.0, -0.0, 0.1), (1.25, 2.0, 0.35)]
    sec, nanosec, frame_id, poses = _path_outputs(
        points,
        sec=7,
        nanosec=900,
    )

    parsed = parse_reference_path_outputs(sec, nanosec, frame_id, poses)

    parsed_points, frame_id, sec, nanosec, points_sha256, terminal_yaw = parsed
    assert parsed_points == tuple(points)
    assert frame_id == "world"
    assert (sec, nanosec) == (7, 900)
    assert terminal_yaw == 0.0
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("!ddd", *point))
    assert points_sha256 == digest.hexdigest()


def test_parse_reference_path_normalizes_and_wraps_terminal_yaw() -> None:
    requested_yaw = math.pi + 0.25
    sec, nanosec, frame_id, poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1)],
        terminal_yaw=requested_yaw,
    )
    terminal_pose = json.loads(poses[-1])
    orientation = terminal_pose["pose"]["orientation"]
    terminal_pose["pose"]["orientation"] = {
        key: 3.0 * float(value) for key, value in orientation.items()
    }
    poses[-1] = json.dumps(terminal_pose)

    *_, terminal_yaw = parse_reference_path_outputs(
        sec,
        nanosec,
        frame_id,
        poses,
    )

    assert terminal_yaw == pytest.approx(-math.pi + 0.25)


@pytest.mark.parametrize(
    ("orientation", "error_match"),
    [
        (None, "orientation"),
        ({"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0}, "零四元数"),
        (
            {
                "x": 0.0,
                "y": math.sqrt(0.5),
                "z": 0.0,
                "w": math.sqrt(0.5),
            },
            "无法确定",
        ),
        ({"x": 0.0, "y": 0.0, "z": float("nan"), "w": 1.0}, "NaN"),
    ],
)
def test_parse_reference_path_rejects_invalid_terminal_orientation(
    orientation: dict[str, float] | None,
    error_match: str,
) -> None:
    sec, nanosec, frame_id, poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1)]
    )
    terminal_pose = json.loads(poses[-1])
    terminal_pose["pose"]["orientation"] = orientation
    poses[-1] = json.dumps(terminal_pose)

    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_reference_path_outputs(sec, nanosec, frame_id, poses)


@pytest.mark.parametrize(
    ("header_mutation", "poses_mutation", "error_match"),
    [
        ({"frame_id": "map"}, None, "frame_id"),
        ({"sec": -1, "nanosec": 0}, None, "不能为负"),
        ({"sec": 1, "nanosec": 1_000_000_000}, None, "nanosec"),
        (None, ["not-json", "{}"], "合法 JSON"),
        (None, [json.dumps({"pose": {"position": {"x": 0, "y": 0, "z": 0}}})], "一个"),
        (
            None,
            [
                json.dumps(
                    {"pose": {"position": {"x": float("nan"), "y": 0, "z": 0}}}
                ),
                json.dumps(
                    {"pose": {"position": {"x": 1, "y": 0, "z": 0}}}
                ),
            ],
            "NaN",
        ),
    ],
)
def test_parse_reference_path_rejects_invalid_world_path(
    header_mutation: dict[str, object] | None,
    poses_mutation: list[str] | None,
    error_match: str,
) -> None:
    sec, nanosec, frame_id, poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1)]
    )
    if header_mutation is not None:
        sec = header_mutation.get("sec", sec)
        nanosec = header_mutation.get("nanosec", nanosec)
        frame_id = header_mutation.get("frame_id", frame_id)
    if poses_mutation is not None:
        poses = poses_mutation

    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_reference_path_outputs(sec, nanosec, frame_id, poses)


def test_parse_reference_path_accepts_empty_generation_tombstone() -> None:
    points, frame_id, sec, nanosec, digest, terminal_yaw = (
        parse_reference_path_outputs(
            7,
            9,
            "world",
            [],
        )
    )

    assert points == ()
    assert frame_id == "world"
    assert (sec, nanosec) == (7, 9)
    assert digest == hashlib.sha256(b"").hexdigest()
    assert terminal_yaw is None


def test_parse_controller_status_preserves_exact_identity_nanoseconds() -> None:
    sample = parse_controller_status_outputs(
        _controller_status_outputs(),
        source_topic="/planning/controller_status",
        receipt_timestamp=20.02,
        rx_sequence=11,
    )

    assert sample.source_topic == "/planning/controller_status"
    assert sample.receipt_timestamp == 20.02
    assert sample.rx_sequence == 11
    assert sample.header_stamp == {"sec": 20, "nanosec": 1}
    assert sample.header_stamp_ns == 20_000_000_001
    assert sample.status_sequence == 7
    assert sample.acceptance_sequence == 3
    assert sample.event == 1
    assert sample.reference_path_stamp == {"sec": 18, "nanosec": 999_999_999}
    assert sample.reference_path_stamp_ns == 18_999_999_999
    assert sample.bspline_header_stamp_ns == 19_000_000_002
    assert sample.start_time_ns == 19_000_000_003
    assert sample.traj_id == 42
    assert sample.accepted is True
    assert sample.trajectory_valid is True
    assert sample.state == 10
    assert sample.candidate_present is False


def test_parse_controller_status_preserves_active_yaw_only_command_aggregate() -> None:
    sample = parse_controller_status_outputs(
        _controller_status_outputs(
            active_sensing_yaw_only=True,
            command_sample_count=4,
            max_abs_wz=0.18,
        ),
        source_topic="/planning/controller_status",
        receipt_timestamp=20.03,
        rx_sequence=12,
    )

    assert sample.active_sensing_yaw_only is True
    assert sample.command_sample_count == 4
    assert sample.first_command == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert sample.max_abs_vx == 0.0
    assert sample.max_abs_vy == 0.0
    assert sample.max_abs_wz == pytest.approx(0.18)
    assert sample.command_violation_count == 0


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        (
            {"command_sample_count": 0, "max_abs_wz": 0.1},
            "聚合全部为默认值",
        ),
        (
            {
                "active_sensing_yaw_only": True,
                "command_sample_count": 1,
                "first_command_angular_z": 0.1,
                "max_abs_wz": 0.1,
            },
            "first_command 必须严格为零",
        ),
        (
            {
                "active_sensing_yaw_only": True,
                "command_sample_count": 2,
                "max_abs_vx": 0.01,
            },
            "vx/vy 必须严格为零",
        ),
        (
            {
                "active_sensing_yaw_only": True,
                "command_sample_count": 2,
                "max_abs_wz": 0.201,
            },
            "不得超过 0.20",
        ),
        (
            {"command_sample_count": 1, "command_violation_count": 2},
            "不能超过 sample_count",
        ),
        (
            {
                "command_sample_count": 1,
                "first_command_angular_x": 0.01,
            },
            "非平面轴必须严格为零",
        ),
    ],
)
def test_parse_controller_status_rejects_invalid_command_aggregate(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_controller_status_outputs(
            _controller_status_outputs(**overrides),
            source_topic="/planning/controller_status",
            receipt_timestamp=20.03,
            rx_sequence=12,
        )


def test_parse_controller_status_keeps_rejected_candidate_separate() -> None:
    sample = parse_controller_status_outputs(
        _controller_status_outputs(
            status_sequence=8,
            event=2,
            reason="候选 B-spline frame_id 非法",
            candidate_present=True,
            candidate_reference_path_stamp_sec=20,
            candidate_reference_path_stamp_nanosec=10,
            candidate_bspline_header_stamp_sec=20,
            candidate_bspline_header_stamp_nanosec=11,
            candidate_start_time_sec=20,
            candidate_start_time_nanosec=12,
            candidate_traj_id=43,
        ),
        source_topic="/planning/controller_status",
        receipt_timestamp=20.04,
        rx_sequence=12,
    )

    assert sample.traj_id == 42
    assert sample.reference_path_stamp_ns == 18_999_999_999
    assert sample.candidate_present is True
    assert sample.candidate_traj_id == 43
    assert sample.candidate_reference_path_stamp_ns == 20_000_000_010
    assert sample.candidate_bspline_header_stamp_ns == 20_000_000_011
    assert sample.candidate_start_time_ns == 20_000_000_012


def test_parse_controller_status_accepts_initial_empty_snapshot() -> None:
    sample = parse_controller_status_outputs(
        _controller_status_outputs(
            status_sequence=1,
            acceptance_sequence=0,
            event=0,
            reference_path_stamp_sec=0,
            reference_path_stamp_nanosec=0,
            bspline_header_stamp_sec=0,
            bspline_header_stamp_nanosec=0,
            start_time_sec=0,
            start_time_nanosec=0,
            traj_id=0,
            accepted=False,
            trajectory_valid=False,
            is_final=False,
            emergency_stop=False,
            state=0,
            reason="控制状态变化：WAITING_FOR_TRAJECTORY",
        ),
        source_topic="/planning/controller_status",
        receipt_timestamp=1.02,
        rx_sequence=1,
    )

    assert sample.status_sequence == 1
    assert sample.acceptance_sequence == 0
    assert sample.accepted is False
    assert sample.reference_path_stamp_ns == 0
    assert sample.traj_id == 0


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"header_frame_id": "map"}, "frame_id"),
        ({"header_stamp_sec": 0, "header_stamp_nanosec": 0}, "非零"),
        ({"header_stamp_nanosec": 1_000_000_000}, "nanosec"),
        ({"status_sequence": 0}, "status_sequence"),
        ({"status_sequence": 2, "acceptance_sequence": 3}, "不能大于"),
        ({"event": 6}, "已定义事件"),
        ({"state": 13}, "已定义状态"),
        ({"accepted": 1}, "布尔值"),
        ({"reason": "  "}, "非空字符串"),
        (
            {
                "accepted": False,
                "trajectory_valid": False,
                "acceptance_sequence": 0,
            },
            "accepted=false",
        ),
        ({"candidate_traj_id": 9}, "candidate_present=false"),
        ({"event": 2}, "EVENT_REJECTED"),
        ({"event": 3}, "EVENT_INVALIDATED"),
        ({"event": 5, "trajectory_valid": False}, "ACCEPTED/DUPLICATE"),
    ],
)
def test_parse_controller_status_rejects_inconsistent_snapshot(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_controller_status_outputs(
            _controller_status_outputs(**overrides),
            source_topic="/planning/controller_status",
            receipt_timestamp=20.02,
            rx_sequence=11,
        )


def test_parse_controller_status_requires_exact_output_field_set() -> None:
    outputs = _controller_status_outputs()
    outputs.pop("reason")

    with pytest.raises(ValueError, match="missing=.*reason"):
        parse_controller_status_outputs(
            outputs,
            source_topic="/planning/controller_status",
            receipt_timestamp=20.02,
            rx_sequence=11,
        )


def test_parse_grid_map_diagnostics_preserves_exact_typed_evidence() -> None:
    sample = parse_grid_map_observation_diagnostics_outputs(
        _grid_map_diagnostics_outputs(),
        source_topic="/planning/grid_map_observation_diagnostics",
        receipt_timestamp=30.1,
        rx_sequence=12,
    )

    assert sample.source_topic == (
        "/planning/grid_map_observation_diagnostics"
    )
    assert sample.receipt_timestamp == 30.1
    assert sample.rx_sequence == 12
    assert sample.header_stamp_ns == 30_000_000_005
    assert sample.sensor_pose_stamp_ns == 30_000_000_004
    assert sample.observation_sequence == 9
    assert sample.sensor_origin == (-4.0, 3.0, 0.4)
    assert sample.input_point_count == 2
    assert sample.accepted_endpoint_count == 2
    assert sample.hit_endpoint_count == 1
    assert sample.explicit_free_endpoint_count == 1
    assert sample.hit_endpoint_samples == ((-4.0, 4.2, 0.5),)
    assert sample.hit_endpoint_sample_voxel_indices == ((-80, 84, 10),)
    assert sample.free_to_occupied_transition_count == 1
    assert sample.free_to_occupied_transition_hit_samples == (
        (-4.0, 4.2, 0.5),
    )
    assert sample.explicit_free_miss_voxel_count == 3
    assert sample.occupied_to_free_by_explicit_miss_count == 1
    assert sample.occupied_to_free_by_explicit_miss_samples == (
        (-4.4, 4.2, 0.5),
    )
    assert sample.occupied_removed_by_sliding_reset_count == 0


def test_parse_grid_map_diagnostics_normalizes_ogn_numpy_integer_scalars() -> None:
    """Isaac 动态 int64/uint64 输出必须规范化为可序列化 Python int。"""

    outputs = _grid_map_diagnostics_outputs()
    for key in (
        "hit_endpoint_sample_voxel_indices_xyz",
        "free_to_occupied_transition_voxel_indices_xyz",
        "occupied_to_free_sample_voxel_indices_xyz",
    ):
        outputs[key] = np.asarray(outputs[key], dtype=np.int64)
    for key in (
        "occupied_to_free_transition_hit_observation_sequences",
        "occupied_to_free_transition_hit_header_stamp_ns",
    ):
        outputs[key] = np.asarray(outputs[key], dtype=np.uint64)
    outputs["observation_sequence"] = np.uint64(9)
    outputs["input_point_count"] = np.uint32(2)

    sample = parse_grid_map_observation_diagnostics_outputs(
        outputs,
        source_topic="/planning/grid_map_observation_diagnostics",
        receipt_timestamp=30.1,
        rx_sequence=12,
    )

    assert sample.hit_endpoint_sample_voxel_indices == ((-80, 84, 10),)
    assert sample.occupied_to_free_sample_voxel_indices == ((-88, 84, 10),)
    assert type(sample.observation_sequence) is int
    assert type(sample.input_point_count) is int
    assert all(
        type(value) is int
        for triple in sample.hit_endpoint_sample_voxel_indices
        for value in triple
    )


@pytest.mark.parametrize("invalid_value", [True, -80.0, np.float64(-80.0)])
def test_parse_grid_map_diagnostics_rejects_noninteger_voxel_scalars(
    invalid_value: object,
) -> None:
    outputs = _grid_map_diagnostics_outputs()
    indices = list(outputs["hit_endpoint_sample_voxel_indices_xyz"])
    indices[0] = invalid_value
    outputs["hit_endpoint_sample_voxel_indices_xyz"] = indices

    with pytest.raises(TypeError, match="必须是整数"):
        parse_grid_map_observation_diagnostics_outputs(
            outputs,
            source_topic="/planning/grid_map_observation_diagnostics",
            receipt_timestamp=30.1,
            rx_sequence=12,
        )


def test_parse_grid_map_diagnostics_keeps_sliding_reset_separate() -> None:
    sample = parse_grid_map_observation_diagnostics_outputs(
        _grid_map_diagnostics_outputs(
            input_point_count=1,
            accepted_endpoint_count=1,
            hit_endpoint_count=1,
            explicit_free_endpoint_count=0,
            explicit_free_miss_voxel_count=0,
            occupied_to_free_by_explicit_miss_count=0,
            occupied_to_free_by_explicit_miss_samples=[],
            occupied_to_free_sample_voxel_indices_xyz=[],
            occupied_to_free_transition_hit_observation_sequences=[],
            occupied_to_free_transition_hit_samples=[],
            occupied_to_free_transition_hit_header_stamp_ns=[],
            occupied_removed_by_sliding_reset_count=4,
        ),
        source_topic="/planning/grid_map_observation_diagnostics",
        receipt_timestamp=30.2,
        rx_sequence=13,
    )

    assert sample.occupied_removed_by_sliding_reset_count == 4
    assert sample.occupied_to_free_by_explicit_miss_count == 0
    assert sample.occupied_to_free_by_explicit_miss_samples == ()


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"header_frame_id": "map"}, "frame"),
        ({"observation_sequence": 0}, "observation_sequence"),
        ({"accepted_endpoint_count": 3}, "接纳端点"),
        ({"explicit_free_endpoint_count": 0}, "hit/free"),
        ({"hit_endpoint_count": 2}, "必须保留"),
        ({"hit_endpoint_samples_truncated": True}, "截断标志"),
        (
            {
                "input_point_count": 61,
                "accepted_endpoint_count": 61,
                "hit_endpoint_count": 60,
                "hit_endpoint_samples_truncated": True,
                "hit_endpoint_samples": [
                    _point_json(-4.0, 4.2, 0.5),
                    _point_json(-3.9, 4.2, 0.5),
                ],
                "hit_endpoint_sample_voxel_indices_xyz": [
                    -80, 84, 10, -78, 84, 10
                ],
            },
            "必须保留",
        ),
        (
            {
                "occupied_to_free_by_explicit_miss_count": 4,
                "occupied_to_free_by_explicit_miss_samples": [
                    _point_json(-4.4 + 0.01 * index, 4.2, 0.5)
                    for index in range(4)
                ],
                "occupied_to_free_sample_voxel_indices_xyz": [
                    value
                    for index in range(4)
                    for value in (-88 + index, 84, 10)
                ],
                "occupied_to_free_transition_hit_observation_sequences": [
                    8, 8, 8, 8
                ],
                "occupied_to_free_transition_hit_samples": [
                    _point_json(-4.4 + 0.01 * index, 4.2, 0.5)
                    for index in range(4)
                ],
                "occupied_to_free_transition_hit_header_stamp_ns": [
                    29_000_000_005
                ] * 4,
            },
            "超过 explicit-free",
        ),
        ({"map_fusion_performed": False}, "map_fusion_performed"),
        ({"canonical_empty": True}, "canonical empty"),
        ({"hit_endpoint_samples": [_point_json(0.0, 0.0, math.nan)]}, "NaN"),
    ],
)
def test_parse_grid_map_diagnostics_rejects_inconsistent_evidence(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_grid_map_observation_diagnostics_outputs(
            _grid_map_diagnostics_outputs(**overrides),
            source_topic="/planning/grid_map_observation_diagnostics",
            receipt_timestamp=30.1,
            rx_sequence=12,
        )


def test_parse_grid_map_diagnostics_accepts_canonical_empty_observation() -> None:
    sample = parse_grid_map_observation_diagnostics_outputs(
        _grid_map_diagnostics_outputs(
            canonical_empty=True,
            map_fusion_performed=False,
            input_point_count=0,
            accepted_endpoint_count=0,
            hit_endpoint_count=0,
            explicit_free_endpoint_count=0,
            hit_endpoint_samples=[],
            hit_endpoint_sample_voxel_indices_xyz=[],
            free_to_occupied_transition_count=0,
            free_to_occupied_transition_hit_samples=[],
            free_to_occupied_transition_voxel_indices_xyz=[],
            explicit_free_miss_voxel_count=0,
            occupied_to_free_by_explicit_miss_count=0,
            occupied_to_free_by_explicit_miss_samples=[],
            occupied_to_free_sample_voxel_indices_xyz=[],
            occupied_to_free_transition_hit_observation_sequences=[],
            occupied_to_free_transition_hit_samples=[],
            occupied_to_free_transition_hit_header_stamp_ns=[],
        ),
        source_topic="/planning/grid_map_observation_diagnostics",
        receipt_timestamp=30.3,
        rx_sequence=14,
    )

    assert sample.canonical_empty is True
    assert sample.map_fusion_performed is False
    assert sample.input_point_count == 0


def test_parse_bspline_diagnostics_preserves_identity_and_geometry() -> None:
    sample = parse_bspline_diagnostics_outputs(
        _bspline_diagnostics_outputs(),
        source_topic="/planning/bspline_diagnostics",
        receipt_timestamp=31.1,
        rx_sequence=5,
    )

    assert sample.source_topic == "/planning/bspline_diagnostics"
    assert sample.header_stamp_ns == 31_000_000_002
    assert sample.start_time_ns == 31_000_000_003
    assert sample.reference_path_stamp_ns == 30_000_000_001
    assert sample.diagnostic_sequence == 4
    assert sample.traj_id == 7
    assert sample.ordered_reference_checked is True
    assert sample.ordered_reference_safe is True
    assert sample.maximum_trajectory_deviation == pytest.approx(0.08)
    assert sample.maximum_guide_anchor_deviation == pytest.approx(0.07)
    assert sample.maximum_guide_progress_lead == pytest.approx(0.01)
    assert sample.trajectory_duration == pytest.approx(0.02)
    assert sample.maximum_velocity_upper_bound == pytest.approx(1.0)
    assert sample.double_cylinder_radius == pytest.approx(0.27)
    assert sample.double_cylinder_offset == pytest.approx(0.16)
    assert sample.trajectory_sample_count_total == 3
    assert sample.trajectory_samples[1] == pytest.approx((-4.1, 4.2, 0.4))
    assert sample.ordered_reference_sample_count_total == 3
    assert sample.ordered_reference_samples[-1] == pytest.approx(
        (-3.8, 4.5, 0.1)
    )


def test_parse_bspline_diagnostics_preserves_active_sensing_completion() -> None:
    sample = parse_bspline_diagnostics_outputs(
        _bspline_diagnostics_outputs(
            stationary=True,
            ordered_reference_checked=False,
            ordered_reference_safe=False,
            active_sensing=True,
            active_sensing_event=5,
            active_sensing_start_yaw=0.1,
            active_sensing_target_yaw=0.3,
            active_sensing_yaw_offset=0.2,
            active_sensing_yaw_rate=0.2,
            active_sensing_settle_stamp_sec=32,
            active_sensing_settle_stamp_nanosec=5,
            active_sensing_settle_yaw_error=0.01,
            active_sensing_settle_angular_speed=0.04,
            active_sensing_stable_duration=0.1,
            active_sensing_fusion_baseline=10,
            active_sensing_fusion_current=13,
            active_sensing_fusion_distinct=3,
            active_sensing_fusion_required=3,
            active_sensing_completed=True,
            active_sensing_reason="主动感知融合完成",
        ),
        source_topic="/planning/bspline_diagnostics",
        receipt_timestamp=32.2,
        rx_sequence=9,
    )

    assert sample.active_sensing is True
    assert sample.active_sensing_event == 5
    assert sample.active_sensing_settle_stamp_ns == 32_000_000_005
    assert sample.active_sensing_fusion_baseline == 10
    assert sample.active_sensing_fusion_current == 13
    assert sample.active_sensing_fusion_distinct == 3
    assert sample.active_sensing_completed is True
    assert sample.active_sensing_failed is False


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"active_sensing_event": 1}, "必须全部为默认值"),
        (
            {
                "stationary": True,
                "ordered_reference_checked": False,
                "ordered_reference_safe": False,
                "active_sensing": True,
                "active_sensing_event": 1,
                "active_sensing_start_yaw": 0.1,
                "active_sensing_target_yaw": 0.4,
                "active_sensing_yaw_offset": 0.2,
                "active_sensing_yaw_rate": 0.2,
                "active_sensing_fusion_required": 3,
                "active_sensing_reason": "开始主动感知",
            },
            "target_yaw",
        ),
        (
            {
                "stationary": True,
                "ordered_reference_checked": False,
                "ordered_reference_safe": False,
                "active_sensing": True,
                "active_sensing_event": 5,
                "active_sensing_start_yaw": 0.1,
                "active_sensing_target_yaw": 0.3,
                "active_sensing_yaw_offset": 0.2,
                "active_sensing_yaw_rate": 0.2,
                "active_sensing_settle_stamp_sec": 32,
                "active_sensing_settle_yaw_error": 0.01,
                "active_sensing_settle_angular_speed": 0.04,
                "active_sensing_stable_duration": 0.1,
                "active_sensing_fusion_baseline": 10,
                "active_sensing_fusion_current": 12,
                "active_sensing_fusion_distinct": 2,
                "active_sensing_fusion_required": 3,
                "active_sensing_completed": True,
                "active_sensing_reason": "伪完成",
            },
            "三帧融合",
        ),
    ],
)
def test_parse_bspline_diagnostics_rejects_inconsistent_active_sensing(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_bspline_diagnostics_outputs(
            _bspline_diagnostics_outputs(**overrides),
            source_topic="/planning/bspline_diagnostics",
            receipt_timestamp=32.2,
            rx_sequence=9,
        )


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"header_frame_id": "map"}, "frame"),
        ({"diagnostic_sequence": 0}, "sequence"),
        ({"ordered_reference_safe": False}, "不得发布"),
        (
            {
                "ordered_reference_checked": False,
                "ordered_reference_safe": False,
            },
            "普通运动",
        ),
        ({"maximum_trajectory_deviation": 0.11}, "超过门限"),
        ({"maximum_guide_anchor_deviation": 0.11}, "超过门限"),
        ({"maximum_guide_progress_lead": 0.021}, "超过门限"),
        ({"trajectory_sample_count_total": 4}, "必须保留"),
        ({"trajectory_samples_truncated": True}, "截断标志"),
        ({"ordered_reference_sample_count_total": 1}, "必须保留"),
        (
            {
                "trajectory_duration": 0.59,
                "trajectory_sample_count_total": 60,
                "trajectory_samples_truncated": True,
                "trajectory_samples": [
                    _point_json(0.0, 0.0, 0.0),
                    _point_json(1.0, 0.0, 0.0),
                ],
            },
            "必须保留",
        ),
        (
            {
                "trajectory_sample_count_total": 1,
                "trajectory_samples": [_point_json(0.0, 0.0, 0.0)],
            },
            "trajectory_sample_count_total",
        ),
        ({"trajectory_duration": 0.03}, "0.01 s 全时域合同"),
        ({"double_cylinder_radius": 0.0}, "必须为正数"),
        ({"maximum_velocity_upper_bound": 0.0}, "速度上界"),
        ({"trajectory_samples": [_point_json(0.0, math.inf, 0.0)]}, "NaN"),
    ],
)
def test_parse_bspline_diagnostics_rejects_unsafe_or_incomplete_evidence(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_bspline_diagnostics_outputs(
            _bspline_diagnostics_outputs(**overrides),
            source_topic="/planning/bspline_diagnostics",
            receipt_timestamp=31.1,
            rx_sequence=5,
        )


def test_parse_navigation_status_preserves_identity_and_safety_bits() -> None:
    sample = parse_navigation_status_outputs(
        _navigation_status_outputs(),
        source_topic="/navigation/status",
        receipt_timestamp=12.4,
        rx_sequence=9,
    )

    assert sample.status_sequence == 7
    assert sample.state_revision == 3
    assert sample.goal_id == 10_000_000_001
    assert sample.header_stamp_ns == 12_000_000_345
    assert sample.active_path_stamp_ns == 12_000_000_345
    assert sample.allow_tracking_command is True
    assert sample.force_zero_velocity is False
    permit = sample.to_safety_permit(identity_valid=True)
    assert permit.command_identity == (10_000_000_001, 12_000_000_345, 3)


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"header_frame_id": "map"}, "frame_id"),
        ({"status_sequence": 0}, "status_sequence"),
        ({"allow_tracking_command": True, "force_zero_velocity": True}, "互反"),
        ({"state": 5}, "拒绝态语义"),
        ({"stale_inputs": ["point_cloud"]}, "拒绝态语义"),
        (
            {
                "global_replan_requested": False,
                "global_replan_in_flight": True,
            },
            "拒绝态语义",
        ),
        ({"allow_tracking_command": 1}, "布尔"),
        ({"stale_inputs": ["odom", "odom"]}, "重复"),
    ],
)
def test_parse_navigation_status_rejects_unsafe_snapshot(
    overrides: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        parse_navigation_status_outputs(
            _navigation_status_outputs(**overrides),
            source_topic="/navigation/status",
            receipt_timestamp=12.4,
            rx_sequence=9,
        )


def test_parse_navigation_status_requires_exact_output_fields() -> None:
    outputs = _navigation_status_outputs()
    outputs.pop("force_zero_velocity")

    with pytest.raises(ValueError, match="missing=.*force_zero_velocity"):
        parse_navigation_status_outputs(
            outputs,
            source_topic="/navigation/status",
            receipt_timestamp=12.4,
            rx_sequence=9,
        )


class _FakeAttributeValueHelper:
    values: dict[str, object] = {}

    def __init__(self, attribute: str) -> None:
        self.attribute = attribute

    def set(self, value: object) -> None:
        self.values[self.attribute] = value

    def get(self) -> object:
        return self.values.get(self.attribute)


class _FakeController:
    evaluations: list[object] = []
    attribute_lookups: list[tuple[str, object | None]] = []

    @staticmethod
    def attribute(path: str, node: object | None = None) -> str:
        _FakeController.attribute_lookups.append((path, node))
        if node is None:
            return path
        return f"{node}.{path}"

    @staticmethod
    def node(path: str) -> str:
        return path

    @staticmethod
    def graph(path: str) -> str | None:
        del path
        return "fake_graph"

    @classmethod
    def evaluate_sync(cls, *, graph_id: object) -> None:
        cls.evaluations.append(graph_id)


def _fake_og() -> SimpleNamespace:
    _FakeAttributeValueHelper.values = {}
    _FakeController.evaluations = []
    _FakeController.attribute_lookups = []
    return SimpleNamespace(
        AttributeValueHelper=_FakeAttributeValueHelper,
        Controller=_FakeController,
    )


def _set_fake_controller_status(
    bridge: IsaacRos2OgnBridge,
    *,
    rx_sequence: int,
    **overrides: object,
) -> None:
    prefix = f"{bridge.config.graph_path}/"
    values = _controller_status_outputs(**overrides)
    _FakeAttributeValueHelper.values[
        prefix + "ControllerStatusRxCounter.outputs:count"
    ] = rx_sequence
    for field_name, output_path in (
        bridge_module._CONTROLLER_STATUS_OUTPUT_PORTS.items()
    ):
        _FakeAttributeValueHelper.values[
            prefix + "ROS2SubscribeControllerStatus." + output_path
        ] = values[field_name]


def _set_fake_navigation_status(
    bridge: IsaacRos2OgnBridge,
    *,
    rx_sequence: int,
    **overrides: object,
) -> None:
    prefix = f"{bridge.config.graph_path}/"
    values = _navigation_status_outputs(**overrides)
    _FakeAttributeValueHelper.values[
        prefix + "NavigationStatusRxCounter.outputs:count"
    ] = rx_sequence
    for field_name, output_path in (
        bridge_module._NAVIGATION_STATUS_OUTPUT_PORTS.items()
    ):
        _FakeAttributeValueHelper.values[
            prefix + "ROS2SubscribeNavigationStatus." + output_path
        ] = values[field_name]


def _set_fake_grid_map_diagnostics(
    bridge: IsaacRos2OgnBridge,
    *,
    rx_sequence: int,
    **overrides: object,
) -> None:
    prefix = f"{bridge.config.graph_path}/"
    values = _grid_map_diagnostics_outputs(**overrides)
    _FakeAttributeValueHelper.values[
        prefix + "GridMapDiagnosticsRxCounter.outputs:count"
    ] = rx_sequence
    for field_name, output_path in (
        bridge_module._GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS.items()
    ):
        _FakeAttributeValueHelper.values[
            prefix + "ROS2SubscribeGridMapDiagnostics." + output_path
        ] = values[field_name]


def _set_fake_bspline_diagnostics(
    bridge: IsaacRos2OgnBridge,
    *,
    rx_sequence: int,
    **overrides: object,
) -> None:
    prefix = f"{bridge.config.graph_path}/"
    values = _bspline_diagnostics_outputs(**overrides)
    _FakeAttributeValueHelper.values[
        prefix + "BsplineDiagnosticsRxCounter.outputs:count"
    ] = rx_sequence
    for field_name, output_path in (
        bridge_module._BSPLINE_DIAGNOSTICS_OUTPUT_PORTS.items()
    ):
        _FakeAttributeValueHelper.values[
            prefix + "ROS2SubscribeBsplineDiagnostics." + output_path
        ] = values[field_name]


def test_configure_controller_status_generates_every_dynamic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    update_calls: list[bool] = []
    monkeypatch.setattr(
        bridge_module,
        "_update_kit_once",
        lambda: update_calls.append(True),
    )
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"

    bridge._configure_controller_status_dynamic_outputs(fake_og)

    assert update_calls == [True, True]
    node = f"{bridge.config.graph_path}/ROS2SubscribeControllerStatus"
    assert _FakeAttributeValueHelper.values[
        f"{bridge.config.graph_path}/"
        "ROS2SubscribeControllerStatus.inputs:messageName"
    ] == "ControllerStatus"
    for output_path in bridge_module._CONTROLLER_STATUS_OUTPUT_PORTS.values():
        relative_path = f"ROS2SubscribeControllerStatus.{output_path}"
        assert relative_path in bridge._attribute_helpers
        assert (output_path, node) in _FakeController.attribute_lookups


def test_configure_navigation_status_generates_every_dynamic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    update_calls: list[bool] = []
    monkeypatch.setattr(
        bridge_module,
        "_update_kit_once",
        lambda: update_calls.append(True),
    )
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"

    bridge._configure_navigation_status_dynamic_outputs(fake_og)

    assert update_calls == [True, True]
    node = f"{bridge.config.graph_path}/ROS2SubscribeNavigationStatus"
    assert _FakeAttributeValueHelper.values[
        f"{bridge.config.graph_path}/"
        "ROS2SubscribeNavigationStatus.inputs:messageName"
    ] == "NavigationStatus"
    for output_path in bridge_module._NAVIGATION_STATUS_OUTPUT_PORTS.values():
        relative_path = f"ROS2SubscribeNavigationStatus.{output_path}"
        assert relative_path in bridge._attribute_helpers
        assert (output_path, node) in _FakeController.attribute_lookups


@pytest.mark.parametrize(
    ("node_name", "message_name", "output_ports"),
    [
        (
            "ROS2SubscribeGridMapDiagnostics",
            "GridMapObservationDiagnostics",
            bridge_module._GRID_MAP_DIAGNOSTICS_OUTPUT_PORTS,
        ),
        (
            "ROS2SubscribeBsplineDiagnostics",
            "BsplineDiagnostics",
            bridge_module._BSPLINE_DIAGNOSTICS_OUTPUT_PORTS,
        ),
    ],
)
def test_configure_planning_diagnostics_generates_every_dynamic_output(
    monkeypatch: pytest.MonkeyPatch,
    node_name: str,
    message_name: str,
    output_ports: dict[str, str],
) -> None:
    fake_og = _fake_og()
    update_calls: list[bool] = []
    monkeypatch.setattr(
        bridge_module,
        "_update_kit_once",
        lambda: update_calls.append(True),
    )
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"

    bridge._configure_planning_diagnostics_dynamic_outputs(
        fake_og,
        node_name=node_name,
        message_name=message_name,
        output_ports=output_ports,
    )

    assert update_calls == [True, True]
    node = f"{bridge.config.graph_path}/{node_name}"
    assert _FakeAttributeValueHelper.values[
        f"{node}.inputs:messageName"
    ] == message_name
    for output_path in output_ports.values():
        relative_path = f"{node_name}.{output_path}"
        assert relative_path in bridge._attribute_helpers
        assert (output_path, node) in _FakeController.attribute_lookups


def test_configure_stair_frozen_generates_typed_identity_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    update_calls: list[bool] = []
    monkeypatch.setattr(
        bridge_module,
        "_update_kit_once",
        lambda: update_calls.append(True),
    )
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_stair_execution_frozen_publisher=True,
        )
    )
    bridge._graph = "fake_graph"

    bridge._configure_stair_execution_frozen_dynamic_input(fake_og)

    assert update_calls == [True, True]
    node = f"{bridge.config.graph_path}/ROS2PublishStairExecutionFrozen"
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:frozen"] is True
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:sequence"] == 0
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:writer_id"] == (
        "isaac_ros2_ogn_bridge"
    )
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:writer_epoch"]
    for input_name in (
        "header:stamp:sec",
        "header:stamp:nanosec",
        "header:frame_id",
        "reference_path_stamp:sec",
        "reference_path_stamp:nanosec",
        "writer_id",
        "writer_epoch",
        "sequence",
        "frozen",
    ):
        assert (
            f"ROS2PublishStairExecutionFrozen.inputs:{input_name}"
            in bridge._attribute_helpers
        )
        assert (f"inputs:{input_name}", node) in (
            _FakeController.attribute_lookups
        )


def test_update_odometry_writes_xyzw_then_triggers_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"

    bridge.update_odometry(
        position=(1.0, 2.0, 3.0),
        orientation_wxyz=(0.7, 0.1, 0.2, 0.3),
        linear_velocity=(4.0, 5.0, 6.0),
        angular_velocity=(0.4, 0.5, 0.6),
        timestamp=8.5,
    )

    prefix = f"{bridge.config.graph_path}/"
    norm = math.sqrt(0.7**2 + 0.1**2 + 0.2**2 + 0.3**2)
    assert _FakeAttributeValueHelper.values[
        prefix + "ROS2PublishOdometry.inputs:orientation"
    ] == pytest.approx(
        (0.1 / norm, 0.2 / norm, 0.3 / norm, 0.7 / norm)
    )
    assert _FakeAttributeValueHelper.values[
        prefix + "ROS2PublishOdometry.inputs:timeStamp"
    ] == 8.5
    assert _FakeAttributeValueHelper.values[
        prefix + "ROS2PublishClock.inputs:timeStamp"
    ] == 8.5
    assert _FakeAttributeValueHelper.values[
        prefix + "StateTx.state:enableImpulse"
    ] is True
    assert _FakeController.evaluations == ["fake_graph"]


def test_update_point_cloud_uses_separate_cloud_impulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"

    bridge.update_point_cloud(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        timestamp=8.5,
    )

    prefix = f"{bridge.config.graph_path}/"
    stored = _FakeAttributeValueHelper.values[
        prefix + "ROS2PublishPointCloud.inputs:data"
    ]
    assert isinstance(stored, np.ndarray)
    assert stored.dtype == np.float32
    np.testing.assert_allclose(stored, [[1.0, 2.0, 3.0]])
    assert _FakeAttributeValueHelper.values[
        prefix + "ROS2PublishPointCloud.inputs:timeStamp"
    ] == 8.5
    assert _FakeAttributeValueHelper.values[
        prefix + "CloudTx.state:enableImpulse"
    ] is True
    assert prefix + "StateTx.state:enableImpulse" not in _FakeAttributeValueHelper.values
    assert _FakeController.evaluations == ["fake_graph"]


def test_publish_pct_goal_writes_base_pose_and_exact_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_pct_goal_publisher=True)
    )
    bridge._graph = "fake_graph"

    first = bridge.publish_pct_goal(
        (1.0, -2.0, 3.5),
        math.pi / 2.0,
        stamp_ns=2_000_000_123,
    )
    second = bridge.publish_pct_goal(
        (1.5, -2.5, 3.6),
        -math.pi / 2.0,
        stamp_ns=2_000_000_123,
    )

    node = f"{bridge.config.graph_path}/ROS2PublishPCTGoal"
    values = _FakeAttributeValueHelper.values
    assert values[f"{node}.inputs:header:stamp:sec"] == 2
    assert values[f"{node}.inputs:header:stamp:nanosec"] == 124
    assert values[f"{node}.inputs:header:frame_id"] == "world"
    assert values[f"{node}.inputs:pose:position:x"] == 1.5
    assert values[f"{node}.inputs:pose:position:y"] == -2.5
    assert values[f"{node}.inputs:pose:position:z"] == 3.6
    assert values[f"{node}.inputs:pose:orientation:x"] == 0.0
    assert values[f"{node}.inputs:pose:orientation:y"] == 0.0
    assert values[f"{node}.inputs:pose:orientation:z"] == pytest.approx(
        -math.sqrt(0.5)
    )
    assert values[f"{node}.inputs:pose:orientation:w"] == pytest.approx(
        math.sqrt(0.5)
    )
    prefix = f"{bridge.config.graph_path}/"
    assert values[prefix + "PCTGoalTx.state:enableImpulse"] is True
    assert first.stamp == {"sec": 2, "nanosec": 123}
    assert first.sequence == 1
    assert second.stamp == {"sec": 2, "nanosec": 124}
    assert second.sequence == 2
    assert bridge.pct_goal_transport_attempt_count == 1
    assert _FakeController.evaluations == ["fake_graph", "fake_graph"]


def test_publish_stair_frozen_reports_monotonic_sequence_and_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_stair_execution_frozen_publisher=True,
        )
    )
    bridge._graph = "fake_graph"
    bridge._active_reference_path_stamp_ns = 12_000_000_345

    initial = bridge.publish_stair_execution_frozen(False, timestamp=0.01)
    frozen = bridge.publish_stair_execution_frozen(True, timestamp=0.02)

    node = f"{bridge.config.graph_path}/ROS2PublishStairExecutionFrozen"
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:frozen"] is True
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:sequence"] == 2
    assert _FakeAttributeValueHelper.values[
        f"{node}.inputs:reference_path_stamp:sec"
    ] == 12
    assert _FakeAttributeValueHelper.values[
        f"{node}.inputs:reference_path_stamp:nanosec"
    ] == 345
    assert _FakeAttributeValueHelper.values[f"{node}.inputs:writer_epoch"]
    assert _FakeAttributeValueHelper.values[
        f"{bridge.config.graph_path}/"
        "StairExecutionFrozenTx.state:enableImpulse"
    ] is True
    assert initial.value is False
    assert initial.source_topic == "/planning/stair_execution_frozen"
    assert initial.publish_timestamp == 0.01
    assert initial.sequence == 1
    assert initial.reference_path_stamp_ns == 12_000_000_345
    assert initial.writer_id == "isaac_ros2_ogn_bridge"
    assert initial.writer_epoch == frozen.writer_epoch
    assert frozen.value is True
    assert frozen.publish_timestamp == 0.02
    assert frozen.sequence == 2
    assert bridge.last_stair_execution_frozen_report is frozen
    assert _FakeController.evaluations == ["fake_graph", "fake_graph"]

    with pytest.raises(ValueError, match="不能回退"):
        bridge.publish_stair_execution_frozen(False, timestamp=0.019)
    assert bridge.last_stair_execution_frozen_report is frozen
    assert _FakeController.evaluations == ["fake_graph", "fake_graph"]

    bridge.invalidate_after_stage_reload()
    assert bridge.last_stair_execution_frozen_report is None
    assert bridge._stair_execution_frozen_publish_sequence == 0
    assert bridge._last_stair_execution_frozen_publish_timestamp is None


def test_publish_stair_frozen_rejects_disabled_or_non_boolean_value() -> None:
    disabled = IsaacRos2OgnBridge()
    with pytest.raises(RuntimeError, match="未启用"):
        disabled.publish_stair_execution_frozen(False, timestamp=0.01)

    enabled = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_stair_execution_frozen_publisher=True,
        )
    )
    with pytest.raises(TypeError, match="布尔值"):
        enabled.publish_stair_execution_frozen(1, timestamp=0.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="必须为正"):
        enabled.publish_stair_execution_frozen(False, timestamp=-0.01)


def test_publish_stair_frozen_requires_exact_current_path_identity() -> None:
    enabled = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_stair_execution_frozen_publisher=True,
        )
    )
    enabled._graph = "fake_graph"
    with pytest.raises(RuntimeError, match="current reference Path identity"):
        enabled.publish_stair_execution_frozen(False, timestamp=0.01)

    enabled._active_reference_path_stamp_ns = 10
    enabled._reference_path_identity_fault = "conflict"
    with pytest.raises(RuntimeError, match="identity 无效"):
        enabled.publish_stair_execution_frozen(False, timestamp=0.01)


def test_republish_last_pct_goal_reuses_exact_sample_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_pct_goal_publisher=True)
    )
    bridge._graph = "fake_graph"

    first = bridge.publish_pct_goal(
        (1.0, -2.0, 3.5),
        math.pi / 2.0,
        stamp_ns=2_000_000_123,
    )
    retried = bridge.republish_last_pct_goal()

    assert retried is first
    assert retried.stamp == {"sec": 2, "nanosec": 123}
    assert retried.sequence == 1
    assert bridge.pct_goal_transport_attempt_count == 2
    assert _FakeController.evaluations == ["fake_graph", "fake_graph"]

    next_generation = bridge.publish_pct_goal(
        (1.5, -2.5, 3.6),
        -math.pi / 2.0,
        stamp_ns=2_000_000_123,
    )
    assert next_generation.stamp == {"sec": 2, "nanosec": 124}
    assert next_generation.sequence == 2
    assert bridge.pct_goal_transport_attempt_count == 1


def test_republish_last_pct_goal_rejects_missing_or_tampered_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_pct_goal_publisher=True)
    )
    bridge._graph = "fake_graph"

    with pytest.raises(RuntimeError, match="尚无可重试"):
        bridge.republish_last_pct_goal()
    assert bridge.pct_goal_transport_attempt_count == 0
    assert _FakeController.evaluations == []

    first = bridge.publish_pct_goal(
        (1.0, -2.0, 3.5),
        math.pi / 2.0,
        stamp_ns=2_000_000_123,
    )
    node = f"{bridge.config.graph_path}/ROS2PublishPCTGoal"
    _FakeAttributeValueHelper.values[
        f"{node}.inputs:pose:position:z"
    ] = 999.0

    with pytest.raises(RuntimeError, match="已被篡改"):
        bridge.republish_last_pct_goal()
    assert first.sequence == 1
    assert first.stamp == {"sec": 2, "nanosec": 123}
    assert bridge.pct_goal_transport_attempt_count == 1
    assert _FakeController.evaluations == ["fake_graph"]


def test_publish_pct_goal_rejects_disabled_or_invalid_contract() -> None:
    disabled = IsaacRos2OgnBridge()
    with pytest.raises(RuntimeError, match="未启用"):
        disabled.publish_pct_goal((0.0, 0.0, 0.3), 0.0, stamp_ns=1)

    enabled = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_pct_goal_publisher=True)
    )
    with pytest.raises(ValueError, match="frame"):
        enabled.publish_pct_goal(
            (0.0, 0.0, 0.3),
            0.0,
            stamp_ns=1,
            frame_id="map",
        )
    with pytest.raises(ValueError, match="正数"):
        enabled.publish_pct_goal((0.0, 0.0, 0.3), 0.0, stamp_ns=0)


def test_poll_twist_detects_repeated_value_by_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "CommandRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeTwist.outputs:linearVelocity": (
                0.2,
                -0.1,
                9.0,
            ),
            prefix + "ROS2SubscribeTwist.outputs:angularVelocity": (
                8.0,
                7.0,
                0.3,
            ),
        }
    )

    first = bridge.poll_twist(receipt_timestamp=2.0)
    assert first is not None
    assert first.sequence == 1
    first_input = first.planar_command
    assert isinstance(first_input, PolicyCommandInput)
    assert first_input.command == pytest.approx((0.2, -0.1, 0.3))
    assert first_input.navigation_status_error == "missing_navigation_status"
    assert bridge.poll_twist(receipt_timestamp=2.02) is None

    _FakeAttributeValueHelper.values[
        prefix + "CommandRxCounter.outputs:count"
    ] = 2
    repeated = bridge.poll_twist(receipt_timestamp=2.04)
    assert repeated is not None
    assert repeated.sequence == 2
    repeated_input = repeated.planar_command
    assert isinstance(repeated_input, PolicyCommandInput)
    assert repeated_input.command == pytest.approx((0.2, -0.1, 0.3))
    assert _FakeAttributeValueHelper.values[
        prefix + "CommandRxTick.state:enableImpulse"
    ] is True


def test_poll_twist_treats_uninitialized_counter_as_no_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"

    assert bridge.poll_twist(receipt_timestamp=1.0) is None


def test_poll_twist_primes_counter_after_timeline_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"
    bridge._last_command_sequence = None
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "CommandRxCounter.outputs:count": 7,
            prefix + "ROS2SubscribeTwist.outputs:linearVelocity": (
                0.4,
                0.0,
                0.0,
            ),
            prefix + "ROS2SubscribeTwist.outputs:angularVelocity": (
                0.0,
                0.0,
                0.0,
            ),
        }
    )

    assert bridge.poll_twist(receipt_timestamp=3.0) is None
    _FakeAttributeValueHelper.values[
        prefix + "CommandRxCounter.outputs:count"
    ] = 8
    assert bridge.poll_twist(receipt_timestamp=3.02) is not None


def test_poll_twist_carries_matching_status_and_emits_status_only_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_command_subscription=True,
            enable_pct_goal_publisher=True,
        )
    )
    bridge._graph = "fake_graph"
    bridge.publish_pct_goal(
        (1.0, 2.0, 0.3),
        0.0,
        stamp_ns=10_000_000_001,
    )
    prefix = f"{bridge.config.graph_path}/"
    sec, nanosec, frame_id, poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.1)],
        sec=12,
        nanosec=345,
    )
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": sec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": nanosec,
            prefix + "ROS2SubscribeReferencePath.outputs:header:frame_id": frame_id,
            prefix + "ROS2SubscribeReferencePath.outputs:poses": poses,
            prefix + "CommandRxCounter.outputs:count": 0,
        }
    )
    assert bridge.poll_reference_path() is not None
    _set_fake_navigation_status(bridge, rx_sequence=1)

    status_edge = bridge.poll_twist(receipt_timestamp=12.4)
    assert status_edge is not None
    assert status_edge.command_present is False
    status_input = status_edge.planar_command
    assert isinstance(status_input, PolicyCommandInput)
    assert status_input.command is None
    assert status_input.navigation_permit is not None
    assert status_input.navigation_permit.identity_valid is True
    observed = bridge.navigation_status_observed_diagnostics()
    assert observed["schema"] == (
        "navigation_status_observed_diagnostics_v1"
    )
    assert observed["status_error"] is None
    assert observed["local_pct_goal_stamp_ns"] == 10_000_000_001
    assert observed["local_active_path_stamp_ns"] == 12_000_000_345
    assert observed["status"]["status_sequence"] == 7
    assert observed["status"]["goal_id"] == 10_000_000_001
    assert observed["status"]["active_path_stamp_ns"] == 12_000_000_345
    assert observed["status"]["identity_valid"] is True

    _FakeAttributeValueHelper.values.update(
        {
            prefix + "CommandRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeTwist.outputs:linearVelocity": (0.2, 0.0, 0.0),
            prefix + "ROS2SubscribeTwist.outputs:angularVelocity": (0.0, 0.0, 0.1),
        }
    )
    command = bridge.poll_twist(receipt_timestamp=12.41)
    assert command is not None
    command_input = command.planar_command
    assert isinstance(command_input, PolicyCommandInput)
    assert command_input.command == pytest.approx((0.2, 0.0, 0.1))
    assert command_input.navigation_permit is not None
    assert command_input.navigation_permit.identity_valid is True

    bridge.publish_pct_goal(
        (2.0, 2.0, 0.3),
        0.0,
        stamp_ns=10_000_000_002,
    )
    new_goal_edge = bridge.poll_twist(receipt_timestamp=12.415)
    assert new_goal_edge is not None
    new_goal_input = new_goal_edge.planar_command
    assert isinstance(new_goal_input, PolicyCommandInput)
    assert new_goal_input.navigation_permit is not None
    assert new_goal_input.navigation_permit.identity_valid is False

    _set_fake_navigation_status(
        bridge,
        rx_sequence=2,
        header_stamp_nanosec=400_000_000,
        status_sequence=8,
        state_revision=4,
        goal_id=10_000_000_002,
    )
    new_goal_status = bridge.poll_twist(receipt_timestamp=12.42)
    assert new_goal_status is not None
    new_goal_permit_input = new_goal_status.planar_command
    assert isinstance(new_goal_permit_input, PolicyCommandInput)
    assert new_goal_permit_input.navigation_permit is not None
    assert new_goal_permit_input.navigation_permit.identity_valid is True

    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 2,
            prefix + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": 13,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": 0,
            prefix + "ROS2SubscribeReferencePath.outputs:header:frame_id": "world",
            prefix + "ROS2SubscribeReferencePath.outputs:poses": [],
        }
    )
    tombstone = bridge.poll_reference_path()
    assert tombstone is not None
    assert tombstone.points_ground_xyz == ()
    tombstone_edge = bridge.poll_twist(receipt_timestamp=12.43)
    assert tombstone_edge is not None
    tombstone_input = tombstone_edge.planar_command
    assert isinstance(tombstone_input, PolicyCommandInput)
    assert tombstone_input.navigation_permit is not None
    assert tombstone_input.navigation_permit.identity_valid is False

    _set_fake_navigation_status(
        bridge,
        rx_sequence=3,
        header_stamp_sec=13,
        header_stamp_nanosec=1,
        status_sequence=9,
        state_revision=5,
        goal_id=10_000_000_002,
        state=5,
        allow_tracking_command=False,
        force_zero_velocity=True,
        reason="预测碰撞，强制停车",
    )
    force_zero = bridge.poll_twist(receipt_timestamp=13.01)
    assert force_zero is not None
    force_input = force_zero.planar_command
    assert isinstance(force_input, PolicyCommandInput)
    assert force_input.command is None
    assert force_input.navigation_permit is not None
    assert force_input.navigation_permit.force_zero_velocity is True


def test_poll_twist_turns_malformed_status_into_fail_closed_writer_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values[
        prefix + "CommandRxCounter.outputs:count"
    ] = 0
    _set_fake_navigation_status(
        bridge,
        rx_sequence=1,
        allow_tracking_command=True,
        force_zero_velocity=True,
    )

    sample = bridge.poll_twist(receipt_timestamp=12.4)

    assert sample is not None
    policy_input = sample.planar_command
    assert isinstance(policy_input, PolicyCommandInput)
    assert policy_input.command is None
    assert policy_input.navigation_permit is None
    assert policy_input.navigation_status_error is not None
    assert "互反" in policy_input.navigation_status_error


def test_navigation_status_semantic_regression_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_command_subscription=True)
    )
    bridge._graph = "fake_graph"
    _set_fake_navigation_status(bridge, rx_sequence=1)
    assert bridge.poll_navigation_status(receipt_timestamp=12.4) is not None

    _set_fake_navigation_status(bridge, rx_sequence=2)
    with pytest.raises(RuntimeError, match="status_sequence 未严格递增"):
        bridge.poll_navigation_status(receipt_timestamp=12.41)


def test_poll_goal_reached_detects_repeated_bool_by_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    )
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "GoalReachedRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeGoalReached.outputs:data": False,
        }
    )

    first = bridge.poll_goal_reached(receipt_timestamp=2.0)
    assert first is not None
    assert first.value is False
    assert first.sequence == 1
    assert first.receipt_timestamp == 2.0
    assert bridge.poll_goal_reached(receipt_timestamp=2.02) is None

    _FakeAttributeValueHelper.values[
        prefix + "GoalReachedRxCounter.outputs:count"
    ] = 2
    repeated = bridge.poll_goal_reached(receipt_timestamp=2.04)
    assert repeated is not None
    assert repeated.value is False
    assert repeated.sequence == 2
    assert _FakeAttributeValueHelper.values[
        prefix + "GoalReachedRxTick.state:enableImpulse"
    ] is True

    _FakeAttributeValueHelper.values[
        prefix + "GoalReachedRxCounter.outputs:count"
    ] = 3
    _FakeAttributeValueHelper.values[
        prefix + "ROS2SubscribeGoalReached.outputs:data"
    ] = True
    reached = bridge.poll_goal_reached(receipt_timestamp=2.06)
    assert reached is not None
    assert reached.value is True
    assert reached.sequence == 3


def test_poll_goal_reached_primes_counter_after_timeline_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    )
    bridge._graph = "fake_graph"
    bridge._last_goal_reached_sequence = None
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "GoalReachedRxCounter.outputs:count": 7,
            prefix + "ROS2SubscribeGoalReached.outputs:data": True,
        }
    )

    assert bridge.poll_goal_reached(receipt_timestamp=3.0) is None
    _FakeAttributeValueHelper.values[
        prefix + "GoalReachedRxCounter.outputs:count"
    ] = 8
    _FakeAttributeValueHelper.values[
        prefix + "ROS2SubscribeGoalReached.outputs:data"
    ] = False
    fresh = bridge.poll_goal_reached(receipt_timestamp=3.02)
    assert fresh is not None
    assert fresh.value is False
    assert fresh.sequence == 8


def test_poll_goal_reached_rebases_counter_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    )
    bridge._graph = "fake_graph"
    bridge._last_goal_reached_sequence = 9
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "GoalReachedRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeGoalReached.outputs:data": True,
        }
    )

    assert bridge.poll_goal_reached(receipt_timestamp=4.0) is None
    assert bridge._last_goal_reached_sequence == 1
    _FakeAttributeValueHelper.values[
        prefix + "GoalReachedRxCounter.outputs:count"
    ] = 2
    fresh = bridge.poll_goal_reached(receipt_timestamp=4.02)
    assert fresh is not None
    assert fresh.sequence == 2


def test_poll_goal_reached_rejects_non_bool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    )
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "GoalReachedRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeGoalReached.outputs:data": 1,
        }
    )

    with pytest.raises(RuntimeError, match="不是布尔值"):
        bridge.poll_goal_reached(receipt_timestamp=1.0)


def test_poll_goal_reached_rejects_disabled_subscription() -> None:
    bridge = IsaacRos2OgnBridge()

    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_goal_reached(receipt_timestamp=1.0)


def test_poll_controller_status_uses_counter_and_typed_dynamic_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_controller_status(bridge, rx_sequence=1)

    first = bridge.poll_controller_status(receipt_timestamp=20.02)

    assert first is not None
    assert first.rx_sequence == 1
    assert first.status_sequence == 7
    assert first.acceptance_sequence == 3
    assert first.reference_path_stamp_ns == 18_999_999_999
    assert bridge.poll_controller_status(receipt_timestamp=20.04) is None

    _set_fake_controller_status(
        bridge,
        rx_sequence=2,
        status_sequence=8,
        event=4,
        reason="控制状态变化：TRACKING",
    )
    second = bridge.poll_controller_status(receipt_timestamp=20.06)
    assert second is not None
    assert second.rx_sequence == 2
    assert second.status_sequence == 8
    assert second.acceptance_sequence == 3
    assert _FakeAttributeValueHelper.values[
        f"{bridge.config.graph_path}/ControllerStatusRxTick.state:enableImpulse"
    ] is True
    assert (
        "outputs:reference_path_stamp:nanosec",
        f"{bridge.config.graph_path}/ROS2SubscribeControllerStatus",
    ) in _FakeController.attribute_lookups


def test_poll_controller_status_drains_active_terminal_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模拟同回调 burst，确认旧 active 终态不会被新接受态覆盖。"""

    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_controller_status(
        bridge,
        rx_sequence=1,
        event=3,
        trajectory_valid=False,
        reason="主动感知轨迹已完成并失效",
        active_sensing_yaw_only=True,
        command_sample_count=5,
        max_abs_wz=0.18,
    )
    terminal = bridge.poll_controller_status(receipt_timestamp=20.02)

    _set_fake_controller_status(
        bridge,
        rx_sequence=2,
        status_sequence=8,
        acceptance_sequence=4,
        reference_path_stamp_sec=18,
        reference_path_stamp_nanosec=999_999_999,
        bspline_header_stamp_sec=20,
        bspline_header_stamp_nanosec=2,
        start_time_sec=20,
        start_time_nanosec=3,
        traj_id=43,
        active_sensing_yaw_only=False,
        command_sample_count=1,
        max_abs_wz=0.0,
        event=1,
        trajectory_valid=True,
        reason="恢复运动轨迹已接受",
    )
    replacement = bridge.poll_controller_status(receipt_timestamp=20.04)

    assert terminal is not None
    assert terminal.active_sensing_yaw_only is True
    assert terminal.event == 3
    assert terminal.trajectory_valid is False
    assert terminal.command_sample_count == 5
    assert replacement is not None
    assert replacement.active_sensing_yaw_only is False
    assert replacement.event == 1
    assert replacement.traj_id == 43
    assert replacement.status_sequence == terminal.status_sequence + 1


def test_poll_controller_status_rejects_ogn_counter_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_controller_status(bridge, rx_sequence=2)

    with pytest.raises(RuntimeError, match="接收序列出现缺口"):
        bridge.poll_controller_status(receipt_timestamp=20.02)


def test_poll_controller_status_rejects_message_sequence_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_controller_status(bridge, rx_sequence=1)
    assert bridge.poll_controller_status(receipt_timestamp=20.02) is not None

    _set_fake_controller_status(bridge, rx_sequence=2)
    with pytest.raises(RuntimeError, match="status_sequence 未严格递增"):
        bridge.poll_controller_status(receipt_timestamp=20.04)

    _set_fake_controller_status(
        bridge,
        rx_sequence=2,
        status_sequence=8,
        acceptance_sequence=2,
        event=4,
        reason="旧 controller epoch",
    )
    with pytest.raises(RuntimeError, match="acceptance_sequence 发生回退"):
        bridge.poll_controller_status(receipt_timestamp=20.06)


def test_poll_controller_status_primes_counter_after_timeline_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_controller_status_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    bridge._last_controller_status_rx_sequence = None
    _set_fake_controller_status(bridge, rx_sequence=7)

    assert bridge.poll_controller_status(receipt_timestamp=20.02) is None

    _set_fake_controller_status(
        bridge,
        rx_sequence=8,
        status_sequence=8,
        event=4,
        reason="控制状态变化：TRACKING",
    )
    fresh = bridge.poll_controller_status(receipt_timestamp=20.04)
    assert fresh is not None
    assert fresh.rx_sequence == 8
    assert fresh.status_sequence == 8


def test_poll_controller_status_rejects_disabled_subscription() -> None:
    bridge = IsaacRos2OgnBridge()

    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_controller_status(receipt_timestamp=1.0)


def test_poll_grid_map_diagnostics_uses_typed_sequence_and_bounded_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_grid_map_diagnostics_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_grid_map_diagnostics(bridge, rx_sequence=1)

    first = bridge.poll_grid_map_observation_diagnostics(
        receipt_timestamp=30.1
    )

    assert first is not None
    assert first.rx_sequence == 1
    assert first.observation_sequence == 9
    assert first.hit_endpoint_samples == ((-4.0, 4.2, 0.5),)
    assert (
        bridge.poll_grid_map_observation_diagnostics(receipt_timestamp=30.2)
        is None
    )

    _set_fake_grid_map_diagnostics(
        bridge,
        rx_sequence=2,
        header_stamp_nanosec=6,
        sensor_pose_stamp_nanosec=5,
        observation_sequence=10,
    )
    second = bridge.poll_grid_map_observation_diagnostics(
        receipt_timestamp=30.3
    )
    assert second is not None
    assert second.observation_sequence == 10
    assert _FakeAttributeValueHelper.values[
        f"{bridge.config.graph_path}/"
        "GridMapDiagnosticsRxTick.state:enableImpulse"
    ] is True

    _set_fake_grid_map_diagnostics(
        bridge,
        rx_sequence=3,
        header_stamp_nanosec=7,
        sensor_pose_stamp_nanosec=6,
        observation_sequence=10,
    )
    with pytest.raises(RuntimeError, match="observation_sequence 未严格递增"):
        bridge.poll_grid_map_observation_diagnostics(receipt_timestamp=30.4)


def test_poll_bspline_diagnostics_uses_typed_identity_and_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_bspline_diagnostics_subscription=True,
        )
    )
    bridge._graph = "fake_graph"
    _set_fake_bspline_diagnostics(bridge, rx_sequence=1)

    first = bridge.poll_bspline_diagnostics(receipt_timestamp=31.1)

    assert first is not None
    assert first.rx_sequence == 1
    assert first.diagnostic_sequence == 4
    assert first.reference_path_stamp_ns == 30_000_000_001
    assert first.traj_id == 7
    assert bridge.poll_bspline_diagnostics(receipt_timestamp=31.2) is None

    _set_fake_bspline_diagnostics(
        bridge,
        rx_sequence=2,
        header_stamp_nanosec=4,
        start_time_nanosec=5,
        diagnostic_sequence=5,
        traj_id=8,
    )
    second = bridge.poll_bspline_diagnostics(receipt_timestamp=31.3)
    assert second is not None
    assert second.diagnostic_sequence == 5
    assert second.traj_id == 8

    _set_fake_bspline_diagnostics(
        bridge,
        rx_sequence=3,
        header_stamp_nanosec=6,
        start_time_nanosec=7,
        diagnostic_sequence=5,
        traj_id=9,
    )
    with pytest.raises(RuntimeError, match="diagnostic_sequence 未严格递增"):
        bridge.poll_bspline_diagnostics(receipt_timestamp=31.4)


def test_poll_planning_diagnostics_reject_disabled_subscriptions() -> None:
    bridge = IsaacRos2OgnBridge()

    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_grid_map_observation_diagnostics(receipt_timestamp=1.0)
    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_bspline_diagnostics(receipt_timestamp=1.0)


def test_poll_reference_path_uses_counter_and_strict_dynamic_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    sec, nanosec, frame_id, poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.1, 0.2)],
        terminal_yaw=-1.2,
    )
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": sec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": nanosec,
            prefix + "ROS2SubscribeReferencePath.outputs:header:frame_id": frame_id,
            prefix + "ROS2SubscribeReferencePath.outputs:poses": poses,
        }
    )

    first = bridge.poll_reference_path()

    assert first is not None
    assert first.points_ground_xyz == ((0.0, 0.0, 0.0), (1.0, 0.1, 0.2))
    assert first.source_topic == "/initial_path"
    assert first.frame_id == "world"
    assert first.terminal_yaw == pytest.approx(-1.2)
    assert first.stamp == {"sec": 12, "nanosec": 345}
    assert first.sequence == 1
    assert len(first.points_sha256) == 64
    assert bridge.active_reference_path_stamp_ns == 12_000_000_345
    assert bridge.poll_reference_path() is None

    bridge._active_reference_path_stamp_ns = 0
    bridge._reference_path_identity_fault = "timeline_reset_requires_fresh_path"
    _FakeAttributeValueHelper.values[
        prefix + "ReferencePathRxCounter.outputs:count"
    ] = 2
    repeated_geometry = bridge.poll_reference_path()
    assert repeated_geometry is not None
    assert repeated_geometry.sequence == 2
    assert repeated_geometry.terminal_yaw == pytest.approx(-1.2)
    assert repeated_geometry.points_sha256 == first.points_sha256
    assert bridge.active_reference_path_stamp_ns == 12_000_000_345
    assert _FakeAttributeValueHelper.values[
        prefix + "ReferencePathRxTick.state:enableImpulse"
    ] is True
    assert (
        "outputs:header:stamp:sec",
        prefix.rstrip("/") + "/ROS2SubscribeReferencePath",
    ) in _FakeController.attribute_lookups
    assert (
        prefix + "ROS2SubscribeReferencePath.outputs:header:stamp:sec",
        None,
    ) not in _FakeController.attribute_lookups


def test_poll_reference_path_primes_timeline_reset_then_accepts_next_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"
    bridge._last_reference_path_sequence = None
    prefix = f"{bridge.config.graph_path}/"
    old_sec, old_nanosec, old_frame_id, old_poses = _path_outputs(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    )
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 7,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": old_sec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": old_nanosec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:frame_id": old_frame_id,
            prefix + "ROS2SubscribeReferencePath.outputs:poses": old_poses,
        }
    )

    assert bridge.poll_reference_path() is None

    new_sec, new_nanosec, new_frame_id, new_poses = _path_outputs(
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.4)],
        sec=13,
    )
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 8,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": new_sec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": new_nanosec,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:frame_id": new_frame_id,
            prefix + "ROS2SubscribeReferencePath.outputs:poses": new_poses,
        }
    )
    fresh = bridge.poll_reference_path()
    assert fresh is not None
    assert fresh.sequence == 8
    assert fresh.points_ground_xyz[-1] == (2.0, 0.0, 0.4)
    assert fresh.stamp_sec == 13


def test_poll_reference_path_rejects_malformed_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"
    prefix = f"{bridge.config.graph_path}/"
    _FakeAttributeValueHelper.values.update(
        {
            prefix + "ReferencePathRxCounter.outputs:count": 1,
            prefix + "ROS2SubscribeReferencePath.outputs:header:stamp:sec": 1,
            prefix
            + "ROS2SubscribeReferencePath.outputs:header:stamp:nanosec": 0,
            prefix + "ROS2SubscribeReferencePath.outputs:header:frame_id": "map",
            prefix + "ROS2SubscribeReferencePath.outputs:poses": [],
        }
    )

    with pytest.raises(ValueError, match="frame_id"):
        bridge.poll_reference_path()


def test_poll_reference_path_rejects_disabled_subscription() -> None:
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_reference_path_subscription=False)
    )

    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_reference_path()


def test_poll_twist_rejects_disabled_subscription() -> None:
    bridge = IsaacRos2OgnBridge()

    with pytest.raises(RuntimeError, match="未启用"):
        bridge.poll_twist(receipt_timestamp=1.0)


def test_update_methods_require_setup_before_importing_omni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def _unexpected_import() -> object:
        nonlocal imported
        imported = True
        raise AssertionError("不应导入 OmniGraph")

    monkeypatch.setattr(bridge_module, "_import_omni_graph", _unexpected_import)
    bridge = IsaacRos2OgnBridge()

    with pytest.raises(RuntimeError, match=r"setup\(\)"):
        bridge.update_point_cloud(
            np.zeros((1, 3), dtype=np.float32),
            timestamp=1.0,
        )

    assert imported is False


def test_setup_can_create_non_usd_runtime_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingController:
        Keys = SimpleNamespace(
            CREATE_NODES="create_nodes",
            SET_VALUES="set_values",
            CONNECT="connect",
        )
        constructor_kwargs: list[dict[str, object]] = []
        edits: list[tuple[dict[str, object], dict[str, object]]] = []

        def __init__(self, **kwargs: object) -> None:
            self.constructor_kwargs.append(dict(kwargs))

        @staticmethod
        def graph(_path: str) -> None:
            return None

        def edit(
            self,
            graph_info: dict[str, object],
            edits: dict[str, object],
        ) -> tuple[object, object, object, object]:
            self.edits.append((graph_info, edits))
            return "runtime_graph", (), (), {}

    fake_og = SimpleNamespace(Controller=RecordingController)
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(
        bridge_module,
        "_ros2_bridge_extension_is_enabled",
        lambda: True,
    )
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            graph_backed_by_usd=False,
            enable_reference_path_subscription=False,
        )
    )

    assert bridge.setup() == "runtime_graph"
    assert RecordingController.constructor_kwargs == [
        {"update_usd": False, "undoable": False}
    ]
    assert RecordingController.edits[0][0] == {
        "graph_path": bridge.config.graph_path,
        "evaluator_name": "execution",
    }


def test_setup_failure_blocks_publish_and_can_retry_dynamic_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(
        bridge_module,
        "_ros2_bridge_extension_is_enabled",
        lambda: True,
    )
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            enable_stair_execution_frozen_publisher=True,
        )
    )
    bridge._graph = "partial_graph"
    bridge._setup_failed = True
    bridge._active_reference_path_stamp_ns = 1
    attempts: list[object] = []

    def configure_dynamic_interfaces(og: object) -> None:
        attempts.append(og)
        if len(attempts) == 1:
            raise RuntimeError("injected_dynamic_port_failure")

    monkeypatch.setattr(
        bridge,
        "_configure_dynamic_interfaces",
        configure_dynamic_interfaces,
    )

    with pytest.raises(RuntimeError, match="injected_dynamic_port_failure"):
        bridge.setup()
    assert bridge.is_setup is False
    with pytest.raises(RuntimeError, match=r"重试 setup\(\)"):
        bridge.publish_stair_execution_frozen(False, timestamp=0.01)

    assert bridge.setup() == "fake_graph"
    assert bridge.is_setup is True
    assert attempts == [fake_og, fake_og]


def test_direct_cloud_requires_timestamp_and_timestamps_cannot_go_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"
    points = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="必须提供"):
        bridge.update_point_cloud(points)

    bridge.update_point_cloud(points, timestamp=2.0)
    with pytest.raises(ValueError, match="不能回退"):
        bridge.update_point_cloud(points, timestamp=1.0)


def test_runtime_rejects_stopped_timeline_and_missing_stage_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: False)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "fake_graph"

    with pytest.raises(RuntimeError, match="timeline"):
        bridge.update_point_cloud(
            np.zeros((1, 3), dtype=np.float32),
            timestamp=1.0,
        )

    monkeypatch.setattr(_FakeController, "graph", lambda path: None)
    monkeypatch.setattr(bridge_module, "_timeline_is_playing", lambda: True)
    with pytest.raises(RuntimeError, match="stage"):
        bridge.update_point_cloud(
            np.zeros((1, 3), dtype=np.float32),
            timestamp=1.0,
        )
    assert bridge.is_setup is False


def test_timeline_reset_refreshes_helpers_without_resetting_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_og = _fake_og()
    monkeypatch.setattr(bridge_module, "_import_omni_graph", lambda: fake_og)
    bridge = IsaacRos2OgnBridge()
    bridge._graph = "old_graph"
    bridge._attribute_helpers["cached"] = object()
    bridge._last_state_timestamp = 3.5
    bridge._last_cloud_timestamp = 3.0
    bridge._last_command_sequence = 10
    bridge._last_goal_reached_sequence = 11
    bridge._last_controller_status_rx_sequence = 12
    bridge._last_controller_status_status_sequence = 17
    bridge._last_controller_status_acceptance_sequence = 5
    bridge._last_reference_path_sequence = 12
    bridge._stair_execution_frozen_publish_sequence = 13
    bridge._last_stair_execution_frozen_publish_timestamp = 3.5
    frozen_report = OgnStairExecutionFreezePublicationReport(
        frozen=True,
        source_topic="/planning/stair_execution_frozen",
        publish_timestamp=3.5,
        header_stamp_sec=3,
        header_stamp_nanosec=500_000_000,
        reference_path_stamp_sec=2,
        reference_path_stamp_nanosec=1,
        writer_id="isaac_ros2_ogn_bridge",
        writer_epoch="epoch",
        sequence=13,
    )
    bridge._last_stair_execution_frozen_report = frozen_report

    bridge.refresh_after_timeline_reset()

    assert bridge._graph == "fake_graph"
    assert bridge._attribute_helpers == {}
    assert bridge._last_state_timestamp == 3.5
    assert bridge._last_cloud_timestamp == 3.0
    assert bridge._last_command_sequence is None
    assert bridge._last_goal_reached_sequence is None
    assert bridge._last_controller_status_rx_sequence is None
    assert bridge._last_controller_status_status_sequence == 17
    assert bridge._last_controller_status_acceptance_sequence == 5
    assert bridge._last_reference_path_sequence is None
    assert bridge._stair_execution_frozen_publish_sequence == 13
    assert bridge._last_stair_execution_frozen_publish_timestamp == 3.5
    assert bridge._last_stair_execution_frozen_report is frozen_report


def test_stage_invalidation_clears_goal_reached_sequence() -> None:
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(enable_goal_reached_subscription=True)
    )
    bridge._graph = "old_graph"
    bridge._last_goal_reached_sequence = 12
    bridge._last_controller_status_rx_sequence = 14
    bridge._last_controller_status_status_sequence = 18
    bridge._last_controller_status_acceptance_sequence = 6
    bridge._last_reference_path_sequence = 13

    bridge.invalidate_after_stage_reload()

    assert bridge.is_setup is False
    assert bridge._last_goal_reached_sequence == 0
    assert bridge._last_controller_status_rx_sequence == 0
    assert bridge._last_controller_status_status_sequence is None
    assert bridge._last_controller_status_acceptance_sequence is None
    assert bridge._last_reference_path_sequence == 0


def test_enable_ros2_bridge_extension_enables_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.enabled: set[str] = set()
            self.requests: list[tuple[str, bool]] = []

        def is_extension_enabled(self, name: str) -> bool:
            assert name == "isaacsim.ros2.bridge"
            return name in self.enabled

        def set_extension_enabled_immediate(self, name: str, enabled: bool) -> None:
            self.requests.append((name, enabled))
            if enabled:
                self.enabled.add(name)
            else:
                self.enabled.discard(name)

    manager = FakeManager()
    fake_module = SimpleNamespace(
        get_app=lambda: SimpleNamespace(
            get_extension_manager=lambda: manager,
        )
    )
    monkeypatch.setattr(
        bridge_module.importlib,
        "import_module",
        lambda name: fake_module if name == "omni.kit.app" else None,
    )

    first = enable_ros2_bridge_extension()
    second = enable_ros2_bridge_extension()

    assert first["enabled_before"] is False
    assert first["enabled"] is True
    assert second["enabled_before"] is True
    assert manager.requests == [("isaacsim.ros2.bridge", True)]


def test_enable_ros2_bridge_extension_preserves_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        def is_extension_enabled(self, _name: str) -> bool:
            return False

        def set_extension_enabled_immediate(
            self,
            _name: str,
            _enabled: bool,
        ) -> None:
            raise RuntimeError("Used null prim")

    fake_module = SimpleNamespace(
        get_app=lambda: SimpleNamespace(
            get_extension_manager=lambda: FakeManager(),
        )
    )
    monkeypatch.setattr(
        bridge_module.importlib,
        "import_module",
        lambda name: fake_module if name == "omni.kit.app" else None,
    )

    with pytest.raises(
        RuntimeError,
        match="原始错误：RuntimeError: Used null prim",
    ):
        enable_ros2_bridge_extension()
