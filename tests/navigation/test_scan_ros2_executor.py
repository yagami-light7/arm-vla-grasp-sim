from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from source.interfaces import NavGoal, NavPlan, RobotAction, SimulationState
from source.navigation.scan_ros2_executor import (
    ScanRos2LifecyclePlanner,
    ScanRos2NavExecutor,
    ScanRos2NavExecutorConfig,
)
from source.navigation.scan_stair_freeze import (
    ScanReferencePath,
    ScanStairFreezeConfig,
    hash_ground_path_points,
)


def _plan(*, execution_phase: str = "nav_to_pick") -> NavPlan:
    return NavPlan(
        goal=NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
        waypoints=((0.0, 0.0, 0.0), (1.0, 2.0, 0.3)),
        metadata={"execution_phase": execution_phase},
    )


def _goal(value: bool, sequence: int, *, timestamp: float | None = None) -> dict:
    return {
        "value": value,
        "sequence": sequence,
        "receipt_timestamp": (
            float(sequence) * 0.02 if timestamp is None else timestamp
        ),
    }


def _stamp(stamp_ns: int) -> dict[str, int]:
    return {
        "sec": stamp_ns // 1_000_000_000,
        "nanosec": stamp_ns % 1_000_000_000,
    }


def _controller_status(
    *,
    status_sequence: int,
    acceptance_sequence: int,
    receipt_timestamp: float,
    header_stamp_ns: int,
    reference_path_stamp_ns: int,
    bspline_header_stamp_ns: int,
    start_time_ns: int,
    traj_id: int,
    event: int = 1,
    state: int = 10,
    accepted: bool = True,
    trajectory_valid: bool = True,
    is_final: bool = False,
    emergency_stop: bool = False,
    reason: str = "unit-test controller status",
    candidate: dict | None = None,
) -> dict:
    identity = {
        "reference_path_stamp": _stamp(reference_path_stamp_ns),
        "reference_path_stamp_ns": reference_path_stamp_ns,
        "bspline_header_stamp": _stamp(bspline_header_stamp_ns),
        "bspline_header_stamp_ns": bspline_header_stamp_ns,
        "start_time": _stamp(start_time_ns),
        "start_time_ns": start_time_ns,
        "traj_id": traj_id,
    }
    return {
        "source": "ros2_scan_planner_msgs_controller_status",
        "topic": "/planning/controller_status",
        "receipt_timestamp": receipt_timestamp,
        "rx_sequence": status_sequence,
        "header": {
            "frame_id": "world",
            "stamp": _stamp(header_stamp_ns),
            "stamp_ns": header_stamp_ns,
        },
        "status_sequence": status_sequence,
        "acceptance_sequence": acceptance_sequence,
        "event": event,
        "state": state,
        "reason": reason,
        "accepted": accepted,
        "trajectory_valid": trajectory_valid,
        "is_final": is_final,
        "emergency_stop": emergency_stop,
        "active_sensing_yaw_only": False,
        "command_aggregate": {
            "sample_count": 0,
            "first_command": [0.0] * 6,
            "max_abs_vx": 0.0,
            "max_abs_vy": 0.0,
            "max_abs_wz": 0.0,
            "violation_count": 0,
        },
        "identity": identity,
        "candidate": candidate,
    }


def _write(
    sequence: int | None,
    command: tuple[float, float, float],
    *,
    timestamp: float | None = None,
    requested: tuple[float, float, float] | None = None,
    stop_reasons: tuple[str, ...] = (),
    motion_allowed: bool = True,
    owner_id: str = "scan_cmd_vel",
    navigation_cmd_vel_inhibited: bool | None = None,
    navigation_cmd_vel_inhibit_reason: str | None = None,
    cmd_vel_source_sequence: int | None = None,
    cmd_vel_source_receipt_timestamp: float | None = None,
    cmd_vel_sample_received_this_tick: bool | None = None,
    cmd_vel_sample_drained_this_tick: bool | None = None,
    last_cmd_vel_drain_sequence: int | None = None,
    last_cmd_vel_drain_receipt_timestamp: float | None = None,
    navigation_status_observed_report: dict | None = None,
    policy_navigation_gate_consumed_report: dict | None = None,
    navigation_emergency_stop_latched: bool = False,
    navigation_emergency_stop_reason: str | None = None,
) -> dict:
    inhibited = (
        bool(not motion_allowed or stop_reasons)
        if navigation_cmd_vel_inhibited is None
        else navigation_cmd_vel_inhibited
    )
    effective_timestamp = (
        float(sequence or 1) * 0.02 if timestamp is None else timestamp
    )
    source_sequence = (
        sequence
        if cmd_vel_source_sequence is None
        else cmd_vel_source_sequence
    )
    source_timestamp = (
        effective_timestamp
        if cmd_vel_source_receipt_timestamp is None
        else cmd_vel_source_receipt_timestamp
    )
    received_this_tick = (
        bool(not inhibited and source_sequence is not None)
        if cmd_vel_sample_received_this_tick is None
        else cmd_vel_sample_received_this_tick
    )
    drained_this_tick = (
        bool(inhibited and source_sequence is not None)
        if cmd_vel_sample_drained_this_tick is None
        else cmd_vel_sample_drained_this_tick
    )
    drain_sequence = last_cmd_vel_drain_sequence
    drain_timestamp = last_cmd_vel_drain_receipt_timestamp
    if inhibited and source_sequence is not None and drain_sequence is None:
        drain_sequence = source_sequence
        drain_timestamp = source_timestamp
    report = {
        "timestamp": effective_timestamp,
        "owner_id": owner_id,
        "requested_command": list(command if requested is None else requested),
        "written_command": list(command),
        "motion_allowed": motion_allowed,
        "stop_reasons": list(stop_reasons),
        "navigation_cmd_vel_inhibited": inhibited,
        "navigation_cmd_vel_inhibit_reason": (
            navigation_cmd_vel_inhibit_reason
        ),
        "cmd_vel_source_sequence": source_sequence,
        "cmd_vel_source_receipt_timestamp": source_timestamp,
        "cmd_vel_sample_received_this_tick": received_this_tick,
        "cmd_vel_sample_drained_this_tick": drained_this_tick,
        "last_cmd_vel_drain_sequence": drain_sequence,
        "last_cmd_vel_drain_receipt_timestamp": drain_timestamp,
        "navigation_status_observed_report": (
            navigation_status_observed_report
        ),
        "policy_navigation_gate_consumed_report": (
            policy_navigation_gate_consumed_report
        ),
        "navigation_emergency_stop_latched": (
            navigation_emergency_stop_latched
        ),
        "navigation_emergency_stop_reason": (
            navigation_emergency_stop_reason
        ),
    }
    if sequence is not None:
        report["write_sequence"] = sequence
    return report


def _state(
    step: int,
    *,
    goal: dict | None = None,
    write: dict | None = None,
    pose_xy: tuple[float, float] = (0.0, 0.0),
    root_z: float = 0.3,
    timestamp: float | None = None,
    extra_metadata: dict | None = None,
) -> SimulationState:
    metadata = {}
    if goal is not None:
        metadata["scan_goal_reached_last_sample"] = goal
    if write is not None:
        metadata["scan_cmd_vel_last_write_report"] = write
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return SimulationState(
        step_index=step,
        timestamp=(float(step) * 0.02 if timestamp is None else timestamp),
        robot_root_pose=(
            float(pose_xy[0]),
            float(pose_xy[1]),
            float(root_z),
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        robot_root_velocity=(0.0,) * 6,
        metadata=metadata,
    )


def _observed_navigation_status(
    *,
    receipt_timestamp: float,
    path_stamp_ns: int,
    stale_inputs: tuple[str, ...] = (),
) -> dict:
    """构造 executor 可追溯到 supervisor 接收侧的状态快照。"""

    return {
        "schema": "navigation_status_observed_diagnostics_v1",
        "topic": "/navigation/status",
        "status_error": None,
        "local_pct_goal_stamp_ns": 1,
        "local_active_path_stamp_ns": path_stamp_ns,
        "local_reference_path_identity_fault": None,
        "status": {
            "receipt_timestamp": receipt_timestamp,
            "state": 5,
            "reason": "scan_stair_execution_inhibited",
            "allow_tracking_command": False,
            "force_zero_velocity": True,
            "stop_confirmed": True,
            "global_replan_requested": False,
            "global_replan_in_flight": False,
            "global_replan_request_id": 0,
            "pct_plan_id": 1,
            "consecutive_scan_failures": 0,
            "active_path_stamp_ns": path_stamp_ns,
            "identity_valid": True,
            "stale_inputs": list(stale_inputs),
        },
    }


def _goal_reached_navigation_status(
    *,
    receipt_timestamp: float,
    path_stamp_ns: int,
    goal_stamp_ns: int,
    status_sequence: int,
    state_revision: int = 12,
    global_replan_request_id: int = 0,
) -> dict:
    """构造绑定同代 goal/Path 的 supervisor 完成快照。"""

    header_stamp_ns = int(round(receipt_timestamp * 1_000_000_000))
    return {
        "schema": "navigation_status_observed_diagnostics_v1",
        "topic": "/navigation/status",
        "status_error": None,
        "local_pct_goal_stamp_ns": goal_stamp_ns,
        "local_active_path_stamp_ns": path_stamp_ns,
        "local_reference_path_identity_fault": None,
        "status": {
            "receipt_timestamp": receipt_timestamp,
            "rx_sequence": status_sequence - 1,
            "header_stamp_ns": header_stamp_ns,
            "status_sequence": status_sequence,
            "state_revision": state_revision,
            "goal_id": goal_stamp_ns,
            "state": 6,
            "allow_tracking_command": False,
            "force_zero_velocity": True,
            "stop_confirmed": True,
            "global_replan_requested": False,
            "global_replan_in_flight": False,
            "global_replan_request_id": global_replan_request_id,
            "pct_plan_id": 3,
            "active_path_stamp_ns": path_stamp_ns,
            "consecutive_scan_failures": 0,
            "stale_inputs": ["bspline"],
            "reason": "goal_reached",
            "identity_valid": True,
        },
    }


def _goal_reached_gate_report(
    navigation_status_report: dict,
) -> dict:
    """构造 policy writer 实际消费的同一 supervisor permit。"""

    status = navigation_status_report["status"]
    return {
        "schema": "navigation_policy_gate_diagnostics_v1",
        "required": True,
        "timeout_s": 0.25,
        "status_fault": None,
        "permit_received": True,
        "permit": {
            "header_stamp_ns": status["header_stamp_ns"],
            "received_at": status["receipt_timestamp"],
            "status_sequence": status["status_sequence"],
            "state_revision": status["state_revision"],
            "goal_id": status["goal_id"],
            "active_path_stamp_ns": status["active_path_stamp_ns"],
            "state": status["state"],
            "allow_tracking_command": status["allow_tracking_command"],
            "force_zero_velocity": status["force_zero_velocity"],
            "identity_valid": status["identity_valid"],
            "reason": status["reason"],
        },
        "command_identity": None,
        "command_identity_matches_permit": False,
    }


def _certified_goal_reached_zero_write(
    *,
    sequence: int,
    timestamp: float,
    path_stamp_ns: int,
    goal_stamp_ns: int,
    status_sequence: int,
    global_replan_request_id: int = 0,
) -> dict:
    """构造 supervisor 完成锁存后的唯一 owner 严格零写。"""

    navigation_status = _goal_reached_navigation_status(
        receipt_timestamp=timestamp - 0.01,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
        status_sequence=status_sequence,
        global_replan_request_id=global_replan_request_id,
    )
    return _write(
        sequence,
        (0.0, 0.0, 0.0),
        timestamp=timestamp,
        stop_reasons=(
            "navigation_status_force_zero",
            "navigation_tracking_not_allowed",
        ),
        motion_allowed=False,
        navigation_cmd_vel_inhibited=False,
        navigation_status_observed_report=navigation_status,
        policy_navigation_gate_consumed_report=(
            _goal_reached_gate_report(navigation_status)
        ),
    )


def _executor(*, zero_ticks: int = 3) -> ScanRos2NavExecutor:
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(required_zero_write_ticks=zero_ticks),
    )
    executor.reset(_plan())
    return executor


def _flat_executor_waiting_for_goal_zero(
    *,
    zero_ticks: int = 2,
    path_stamp_ns: int = 2_460_000_000,
    goal_stamp_ns: int = 1_760_000_000,
) -> tuple[ScanRos2NavExecutor, dict]:
    """推进到 Bool true 后，等待 supervisor 完成态零写。"""

    executor = _executor(zero_ticks=zero_ticks)
    executor._live_reference_path_stamp_ns = path_stamp_ns
    executor._pct_goal_stamp_ns = goal_stamp_ns
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=1.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=1.0),
            timestamp=1.0,
        )
    )
    controller = _controller_status(
        status_sequence=128,
        acceptance_sequence=27,
        receipt_timestamp=2.0,
        header_stamp_ns=2_000_000_000,
        reference_path_stamp_ns=path_stamp_ns,
        bspline_header_stamp_ns=1_800_000_000,
        start_time_ns=1_800_000_000,
        traj_id=27,
        event=4,
        state=12,
        is_final=True,
        reason="控制器已完成同代 final 轨迹",
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2, timestamp=2.0),
            write=_write(2, (0.0, 0.0, 0.0), timestamp=2.0),
            timestamp=2.0,
            extra_metadata={
                "scan_controller_status_last_report": controller,
            },
        )
    )
    return executor, controller


def _terminal_executor_at_hold(
    *,
    zero_ticks: int = 2,
) -> tuple[ScanRos2NavExecutor, tuple[float, float, float, float], int]:
    """把严格绑定的短楼梯执行器推进到 terminal_hold。"""

    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(required_zero_write_ticks=zero_ticks),
        stair_freeze_config=ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            certified_progress_m=0.01,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        ),
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.6, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "reference_path_3d_ground": points,
                "reference_path_terminal_yaw": 0.0,
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    action = executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
            pose_xy=(0.2, 0.0),
        )
    )
    target = action.metadata["navigation_base_pose_lock_xyzyaw"]
    for step in range(2, 12):
        action = executor.compute_action(
            _state(
                step,
                goal=_goal(False, step),
                write=_write(
                    step,
                    (0.0, 0.0, 0.0),
                    stop_reasons=("scan_stair_freeze",),
                    motion_allowed=False,
                    navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                ),
                pose_xy=(float(target[0]), float(target[1])),
                root_z=float(target[2]),
            )
        )
        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        if action.source == "scan_stair_freeze_terminal_hold":
            return executor, target, step
    raise AssertionError("执行器未进入 terminal_hold")


def _path_report(
    points: tuple[tuple[float, float, float], ...],
    *,
    sequence: int = 1,
    points_sha256: str | None = None,
    stamp_ns: int | None = None,
    terminal_yaw: float | None = None,
) -> dict:
    effective_stamp_ns = 1_000_000_000 + sequence if stamp_ns is None else stamp_ns
    if points and terminal_yaw is None:
        end = points[-1]
        previous = next(
            point
            for point in reversed(points[:-1])
            if math.hypot(end[0] - point[0], end[1] - point[1]) > 1.0e-9
        )
        terminal_yaw = math.atan2(end[1] - previous[1], end[0] - previous[0])
    return {
        "points_ground_xyz": [list(point) for point in points],
        "source": "ros2_nav_msgs_path",
        "topic": "/initial_path",
        "frame_id": "world",
        "stamp": {
            "sec": effective_stamp_ns // 1_000_000_000,
            "nanosec": effective_stamp_ns % 1_000_000_000,
        },
        "sequence": sequence,
        "points_sha256": (
            hash_ground_path_points(points)
            if points_sha256 is None
            else points_sha256
        ),
        "cleared": len(points) == 0,
        "terminal_yaw": terminal_yaw,
    }


def _pct_goal_report(
    *,
    generation: int,
    sequence: int = 1,
    stamp_ns: int = 20_000_000,
    position_base_xyz: tuple[float, float, float] = (1.0, 2.0, 0.3),
    yaw: float = 0.4,
    effective_goal_provenance: dict | None = None,
) -> dict:
    report = {
        "published": True,
        "source": "isaac_ros2_ogn_pose_stamped",
        "topic": "/pct/goal",
        "frame_id": "world",
        "stamp": {
            "sec": stamp_ns // 1_000_000_000,
            "nanosec": stamp_ns % 1_000_000_000,
        },
        "sequence": sequence,
        "generation": generation,
        "position_base_xyz": list(position_base_xyz),
        "yaw": yaw,
        "height_semantics": "base",
    }
    if effective_goal_provenance is not None:
        report["effective_goal_provenance_required"] = True
        report["effective_goal_provenance"] = effective_goal_provenance
    return report


def _assert_stair_emergency_hold_remains_latched(
    executor: ScanRos2NavExecutor,
    action: RobotAction,
    *,
    target: tuple[float, float, float, float],
    failure_reason: str,
    origin_phase: str,
    next_step: int,
    next_timestamp: float,
) -> None:
    """断言楼梯急停在后续 tick 仍保持同一底盘、关节与携物锁。"""

    def assert_action(held_action: RobotAction) -> None:
        assert held_action.source == "scan_stair_emergency_hold"
        assert held_action.base_velocity == (0.0, 0.0, 0.0)
        assert held_action.metadata["navigation_emergency_stop"] is True
        assert held_action.metadata["navigation_emergency_stop_reason"] == (
            failure_reason
        )
        assert held_action.metadata["navigation_global_replan_requested"] is True
        assert held_action.metadata["navigation_cmd_vel_inhibit"] is True
        assert held_action.metadata["navigation_cmd_vel_inhibit_reason"] == (
            "scan_stair_emergency_hold"
        )
        assert held_action.metadata["navigation_base_pose_lock"] is True
        assert held_action.metadata["navigation_base_pose_lock_xyzyaw"] == target
        assert held_action.metadata["navigation_support_joint_lock"] is True
        assert held_action.metadata["navigation_full_body_joint_lock"] is True
        assert held_action.metadata["navigation_carry_object_follow"] is True
        assert held_action.metadata["navigation_stair_emergency_hold"] is True
        assert held_action.metadata[
            "navigation_stair_emergency_hold_origin_phase"
        ] == origin_phase

    assert_action(action)
    status = executor.status()
    assert status["failed"] is True
    assert status["failure_reason"] == failure_reason
    assert status["stair_freeze"]["emergency_hold_latched"] is True
    assert status["stair_freeze"]["emergency_hold_origin_phase"] == origin_phase
    assert status["stair_freeze"]["emergency_hold_full_body_lock"] is True

    repeated = executor.compute_action(
        _state(
            next_step,
            timestamp=next_timestamp,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
        )
    )
    assert_action(repeated)


def test_lifecycle_planner_only_carries_external_path_goal() -> None:
    planner = ScanRos2LifecyclePlanner()
    goal = NavGoal(x=1.0, y=2.0, z=0.48, yaw=0.4)

    plan = planner.plan(_state(0), goal)

    assert plan.goal is goal
    assert plan.waypoints == (
        (0.0, 0.0, 0.3),
        (1.0, 2.0, 0.48),
    )
    assert plan.metadata == {
        "planner": "external_ros2_path_lifecycle",
        "path_source": "/initial_path",
        "path_consumed_by": "scan_planner",
        "pipeline_waypoints_are_control_inputs": False,
    }


def test_lifecycle_planner_injects_same_ground_reference_path() -> None:
    reference = ScanReferencePath(
        points_ground_xyz=((0.0, 0.0, -0.1), (1.0, 0.0, 0.2)),
        source_path="/tmp/manual_path.yaml",
        sha256="a" * 64,
        points_sha256="b" * 64,
        topic="/initial_path",
        frame_id="world",
        use_sim_time=True,
        min_point_distance_m=0.02,
        stair_segment_indices=((0, 1),),
    )
    planner = ScanRos2LifecyclePlanner(reference_path=reference)

    plan = planner.plan(_state(0), NavGoal(x=1.0, y=0.0, z=0.5, yaw=0.0))

    assert plan.waypoints == reference.points_ground_xyz
    assert plan.metadata["reference_path_3d_ground"] == (
        reference.points_ground_xyz
    )
    assert plan.metadata["reference_path_height_semantics"] == "ground"
    assert plan.metadata["reference_path_sha256"] == "a" * 64
    assert plan.metadata["reference_path_points_sha256"] == "b" * 64
    assert plan.metadata["reference_path_stair_segment_indices"] == ((0, 1),)


def test_lifecycle_planner_exports_dynamic_pct_goal_with_base_height() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    goal = NavGoal(x=1.0, y=2.0, z=0.48, yaw=0.4)

    plan = planner.plan(_state(0), goal)

    assert plan.metadata["path_source"] == "/pct/global_path"
    assert plan.metadata["pct_goal_request"] == {
        "frame_id": "world",
        "position_base_xyz": (1.0, 2.0, 0.48),
        "yaw": 0.4,
        "height_semantics": "base",
    }
    with pytest.raises(ValueError, match="base 高度"):
        planner.plan(_state(0), NavGoal(x=1.0, y=2.0, z=None, yaw=0.0))
    with pytest.raises(ValueError, match="不能同时"):
        ScanRos2LifecyclePlanner(
            reference_path=ScanReferencePath(
                points_ground_xyz=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                source_path="/tmp/path.yaml",
                sha256="a" * 64,
                points_sha256="b" * 64,
                topic="/initial_path",
                frame_id="world",
                use_sim_time=True,
                min_point_distance_m=0.02,
                stair_segment_indices=(),
            ),
            publish_pct_goal=True,
        )


def test_dynamic_pct_goal_requires_exact_effective_height_provenance_ack() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    base_plan = planner.plan(
        _state(0),
        NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
    )
    provenance = {
        "schema": "pct_effective_goal_height_v1",
        "collision_ply_sha256": "a" * 64,
    }
    request = {
        **base_plan.metadata["pct_goal_request"],
        "effective_goal_provenance_required": True,
        "effective_goal_provenance": provenance,
    }
    plan = NavPlan(
        goal=base_plan.goal,
        waypoints=base_plan.waypoints,
        metadata={**base_plan.metadata, "pct_goal_request": request},
    )
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True)
    )
    executor.reset(plan)

    publish = executor.compute_action(_state(1))
    assert publish.metadata["navigation_pct_goal_request"][
        "effective_goal_provenance"
    ] == provenance

    executor.compute_action(
        _state(
            2,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(
                    generation=1,
                    effective_goal_provenance={**provenance, "tampered": True},
                )
            },
        )
    )
    assert executor.status()["failed"] is True
    assert executor.status()["failure_reason"] == (
        "pct_goal_publish_report_mismatch"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"required_zero_write_ticks": 0}, "正整数"),
        ({"required_zero_write_ticks": True}, "正整数"),
        ({"zero_epsilon": -1.0}, "有限非负数"),
        ({"zero_epsilon": math.inf}, "有限非负数"),
        ({"policy_owner_id": ""}, "非空字符串"),
        ({"progress_watchdog_timeout_s": 0.0}, "有限正数"),
        ({"progress_watchdog_timeout_s": math.inf}, "有限正数"),
        ({"progress_watchdog_min_displacement_m": -0.1}, "有限正数"),
        ({"progress_watchdog_min_forward_command_mps": True}, "有限正数"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ScanRos2NavExecutorConfig(**kwargs)


def test_executor_requires_full_ordered_evidence_chain() -> None:
    executor = _executor(zero_ticks=3)

    # transient-local 遗留 true 与 false 前的活动都不能成为本轮证据。
    stale = _state(
        1,
        goal=_goal(True, 1),
        write=_write(1, (0.2, 0.0, 0.0)),
    )
    assert executor.is_done(stale) is False
    assert executor.status()["fresh_false_seen"] is False
    assert executor.status()["policy_activity_seen"] is False

    executor.compute_action(
        _state(
            2,
            goal=_goal(False, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    assert executor.status()["phase"] == "waiting_for_policy_activity"

    executor.compute_action(
        _state(
            3,
            goal=_goal(False, 3),
            write=_write(3, (0.2, -0.1, 0.1)),
        )
    )
    assert executor.status()["policy_activity_seen"] is True
    assert executor.status()["phase"] == "tracking_waiting_for_goal"

    first_zero = _state(
        4,
        goal=_goal(True, 4),
        write=_write(4, (0.0, 0.0, 0.0)),
    )
    assert executor.is_done(first_zero) is False
    assert executor.status()["zero_write_streak"] == 0

    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(5, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            6,
            goal=_goal(True, 6),
            write=_write(6, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            7,
            goal=_goal(True, 7),
            write=_write(7, (0.0, 0.0, 0.0)),
        )
    ) is True

    status = executor.status()
    assert status["success"] is True
    assert status["scan_controller_goal_reached_verified"] is True
    assert status["policy_zero_hold_verified"] is True
    assert status["activity_write_sequence"] == 3
    assert status["goal_true_sequence"] == 4
    assert status["zero_write_streak"] == 3


def test_same_observation_and_sequences_are_idempotent() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    first_zero = _state(
        2,
        goal=_goal(True, 2),
        write=_write(2, (0.0, 0.0, 0.0)),
    )

    assert executor.is_done(first_zero) is False
    assert executor.compute_action(first_zero).base_velocity == (0.0, 0.0, 0.0)
    assert executor.compute_action(first_zero).source == "scan_ros2_navigation"
    assert executor.is_done(first_zero) is False
    assert executor.status()["zero_write_streak"] == 0

    duplicate_sequences = _state(
        3,
        goal=_goal(True, 2, timestamp=0.06),
        write=_write(2, (0.0, 0.0, 0.0), timestamp=0.06),
    )
    assert executor.is_done(duplicate_sequences) is False
    assert executor.status()["zero_write_streak"] == 0

    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 3),
            write=_write(3, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0)),
        )
    ) is True


@pytest.mark.parametrize(
    "inhibit_reason",
    [
        "body_height_preflight",
        "pct_goal_waiting_for_publish",
        "pct_goal_waiting_for_transport_ack",
        "pct_goal_waiting_for_path",
        "scan_stair_freeze",
        "scan_stair_freeze_release",
        "scan_stair_terminal_hold",
    ],
)
def test_explicit_temporary_inhibit_without_upstream_twist_is_valid_zero(
    inhibit_reason: str,
) -> None:
    executor = _executor(zero_ticks=2)
    report = _write(
        1,
        (0.0, 0.0, 0.0),
        stop_reasons=(inhibit_reason,),
        motion_allowed=False,
        navigation_cmd_vel_inhibited=True,
        navigation_cmd_vel_inhibit_reason=inhibit_reason,
    )
    report["requested_command"] = None

    state = _state(1, goal=_goal(False, 1), write=report)
    executor.compute_action(state)
    executor.is_done(state)

    status = executor.status()
    assert status["invalid_policy_write_count"] == 0
    assert status["last_requested_command"] == (0.0, 0.0, 0.0)
    assert status["last_written_command"] == (0.0, 0.0, 0.0)


def test_temporary_inhibit_accepts_only_declared_sensor_co_reasons() -> None:
    executor = _executor(zero_ticks=2)
    report = _write(
        1,
        (0.0, 0.0, 0.0),
        stop_reasons=(
            "body_height_preflight",
            "missing_odometry",
            "missing_point_cloud",
        ),
        motion_allowed=False,
        navigation_cmd_vel_inhibited=True,
        navigation_cmd_vel_inhibit_reason="body_height_preflight",
    )
    report["requested_command"] = None

    executor.compute_action(
        _state(1, goal=_goal(False, 1), write=report)
    )

    assert executor.status()["invalid_policy_write_count"] == 0


def test_missing_command_bootstrap_zero_is_valid_fail_closed_write() -> None:
    executor = _executor(zero_ticks=2)
    report = _write(
        1,
        (0.0, 0.0, 0.0),
        stop_reasons=(
            "missing_cmd_vel",
            "missing_odometry",
            "missing_point_cloud",
            "missing_navigation_status",
        ),
        motion_allowed=False,
        navigation_cmd_vel_inhibited=False,
        navigation_cmd_vel_inhibit_reason=None,
    )
    report["requested_command"] = None

    executor.compute_action(
        _state(1, goal=_goal(False, 1), write=report)
    )

    status = executor.status()
    assert status["invalid_policy_write_count"] == 0
    assert status["policy_activity_seen"] is False


@pytest.mark.parametrize(
    ("stop_reasons", "motion_allowed", "inhibited", "inhibit_reason"),
    [
        (("unexpected_inhibit",), False, True, "unexpected_inhibit"),
        (
            ("pct_goal_waiting_for_path",),
            True,
            True,
            "pct_goal_waiting_for_path",
        ),
        (
            ("pct_goal_waiting_for_path",),
            False,
            False,
            "pct_goal_waiting_for_path",
        ),
        (("pct_goal_waiting_for_path",), False, True, "scan_stair_freeze"),
        (
            ("pct_goal_waiting_for_path", "scan_stair_freeze"),
            False,
            True,
            "pct_goal_waiting_for_path",
        ),
        (
            ("scan_stair_freeze", "scan_stair_freeze"),
            False,
            True,
            "scan_stair_freeze",
        ),
    ],
)
def test_null_requested_command_requires_exact_safe_inhibit_contract(
    stop_reasons: tuple[str, ...],
    motion_allowed: bool,
    inhibited: bool,
    inhibit_reason: str,
) -> None:
    executor = _executor(zero_ticks=2)
    report = _write(
        1,
        (0.0, 0.0, 0.0),
        stop_reasons=stop_reasons,
        motion_allowed=motion_allowed,
        navigation_cmd_vel_inhibited=inhibited,
        navigation_cmd_vel_inhibit_reason=inhibit_reason,
    )
    report["requested_command"] = None
    state = _state(1, goal=_goal(False, 1), write=report)

    executor.compute_action(state)
    executor.is_done(state)

    assert executor.status()["invalid_policy_write_count"] == 1


def test_distinct_invalid_policy_writes_are_counted_once_each() -> None:
    executor = _executor(zero_ticks=2)
    first = _write(1, (0.0, 0.0, 0.0))
    first["requested_command"] = None
    first_state = _state(1, goal=_goal(False, 1), write=first)
    executor.compute_action(first_state)
    executor.is_done(first_state)

    second = _write(2, (0.0, 0.0, 0.0))
    second["requested_command"] = None
    second_state = _state(2, goal=_goal(False, 2), write=second)
    executor.compute_action(second_state)
    executor.is_done(second_state)

    assert executor.status()["invalid_policy_write_count"] == 2


def test_terminal_true_waiting_for_supervisor_ack_is_not_premature() -> None:
    executor = _executor(zero_ticks=2)
    executor._fresh_false_seen = True
    executor._root_lock_progress_seen = True
    stair = SimpleNamespace(
        finish_ready=False,
        terminal_supervisor_transition_pending=True,
        status=lambda: {"applicable": True},
    )
    executor._stair_freeze = stair  # type: ignore[assignment]

    executor._consume_goal_sample(  # type: ignore[attr-defined]
        {"scan_goal_reached_last_sample": _goal(True, 1)}
    )
    status = executor.status()
    assert status["premature_true_count"] == 0
    assert status["goal_true_waiting_for_supervisor_ack_count"] == 1
    assert status["goal_rising_edge_seen"] is False
    assert status["phase"] == "goal_reached_waiting_for_supervisor_ack"

    stair.finish_ready = True
    stair.terminal_supervisor_transition_pending = False
    executor._consume_goal_sample(  # type: ignore[attr-defined]
        {"scan_goal_reached_last_sample": _goal(True, 2)}
    )
    status = executor.status()
    assert status["goal_rising_edge_seen"] is True
    assert status["goal_true_sequence"] == 2


@pytest.mark.parametrize(
    "write",
    [
        _write(2, (0.2, 0.0, 0.0), stop_reasons=("cloud_timeout",)),
        _write(2, (0.2, 0.0, 0.0), motion_allowed=False),
        _write(2, (0.2, 0.0, 0.0), owner_id="other_writer"),
        _write(
            2,
            (0.2, 0.0, 0.0),
            requested=(0.0, 0.0, 0.0),
        ),
    ],
)
def test_policy_activity_must_be_a_valid_requested_and_written_nonzero(
    write: dict,
) -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
        )
    )
    executor.compute_action(_state(2, goal=_goal(False, 2), write=write))

    assert executor.status()["policy_activity_seen"] is False
    assert executor.status()["phase"] == "waiting_for_policy_activity"


def test_v23_point_cloud_timeout_zero_counts_after_certified_goal() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(
                2,
                (0.0, 0.0, 0.0),
                stop_reasons=("point_cloud_timeout",),
                motion_allowed=False,
            ),
        )
    )
    assert executor.status()["zero_write_streak"] == 0

    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                stop_reasons=("point_cloud_timeout",),
                motion_allowed=False,
            ),
        )
    ) is False
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(
                4,
                (0.0, 0.0, 0.0),
                stop_reasons=("point_cloud_timeout",),
                motion_allowed=False,
            ),
        )
    ) is True
    assert executor.status()["last_stop_reasons"] == (
        "point_cloud_timeout",
    )


def test_supervisor_goal_reached_force_zero_counts_with_exact_identity() -> None:
    path_stamp_ns = 2_460_000_000
    goal_stamp_ns = 1_760_000_000
    executor, controller = _flat_executor_waiting_for_goal_zero(
        zero_ticks=2,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
    )

    first_zero = _certified_goal_reached_zero_write(
        sequence=3,
        timestamp=2.02,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
        status_sequence=20,
    )
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3, timestamp=2.02),
            write=first_zero,
            timestamp=2.02,
            extra_metadata={
                "scan_controller_status_last_report": controller,
            },
        )
    ) is False
    second_zero = _certified_goal_reached_zero_write(
        sequence=4,
        timestamp=2.04,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
        status_sequence=21,
        # 已完成全局重规划后 request id 可以保持非零。
        global_replan_request_id=7,
    )
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4, timestamp=2.04),
            write=second_zero,
            timestamp=2.04,
            extra_metadata={
                "scan_controller_status_last_report": controller,
            },
        )
    ) is True
    status = executor.status()
    assert status["zero_write_streak"] == 2
    assert status["supervisor_goal_reached_zero_count"] == 2
    assert status["last_supervisor_goal_reached_zero_verified"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_status",
        "emergency_state",
        "goal_identity_mismatch",
        "path_identity_mismatch",
        "stop_not_confirmed",
        "replan_active",
        "permit_mismatch",
        "extra_stop_reason",
        "controller_missing",
        "controller_path_mismatch",
        "emergency_latched",
    ],
)
def test_supervisor_goal_reached_force_zero_fails_closed(
    mutation: str,
) -> None:
    path_stamp_ns = 2_460_000_000
    goal_stamp_ns = 1_760_000_000
    executor, controller = _flat_executor_waiting_for_goal_zero(
        zero_ticks=1,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
    )
    write = _certified_goal_reached_zero_write(
        sequence=3,
        timestamp=2.02,
        path_stamp_ns=path_stamp_ns,
        goal_stamp_ns=goal_stamp_ns,
        status_sequence=20,
    )
    controller_report: dict | None = copy.deepcopy(controller)
    navigation_status = write["navigation_status_observed_report"]
    permit = write["policy_navigation_gate_consumed_report"]["permit"]
    if mutation == "missing_status":
        write["navigation_status_observed_report"] = None
    elif mutation == "emergency_state":
        navigation_status["status"]["state"] = 5
        permit["state"] = 5
    elif mutation == "goal_identity_mismatch":
        navigation_status["local_pct_goal_stamp_ns"] += 1
        navigation_status["status"]["goal_id"] += 1
        permit["goal_id"] += 1
    elif mutation == "path_identity_mismatch":
        navigation_status["local_active_path_stamp_ns"] += 1
        navigation_status["status"]["active_path_stamp_ns"] += 1
        permit["active_path_stamp_ns"] += 1
        controller_report["identity"]["reference_path_stamp_ns"] += 1
    elif mutation == "stop_not_confirmed":
        navigation_status["status"]["stop_confirmed"] = False
    elif mutation == "replan_active":
        navigation_status["status"]["global_replan_in_flight"] = True
    elif mutation == "permit_mismatch":
        permit["status_sequence"] += 1
    elif mutation == "extra_stop_reason":
        write["stop_reasons"].append("predicted_collision")
    elif mutation == "controller_missing":
        controller_report = None
    elif mutation == "controller_path_mismatch":
        controller_report["identity"]["reference_path_stamp_ns"] += 1
    elif mutation == "emergency_latched":
        write["navigation_emergency_stop_latched"] = True
        write["navigation_emergency_stop_reason"] = "predicted_collision"
    else:  # pragma: no cover - 参数表必须穷尽。
        raise AssertionError(mutation)

    extra_metadata = {}
    if controller_report is not None:
        extra_metadata[
            "scan_controller_status_last_report"
        ] = controller_report
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3, timestamp=2.02),
            write=write,
            timestamp=2.02,
            extra_metadata=extra_metadata,
        )
    ) is False
    status = executor.status()
    assert status["zero_write_streak"] == 0
    assert status["supervisor_goal_reached_zero_count"] == 0
    assert status["last_supervisor_goal_reached_zero_verified"] is False


@pytest.mark.parametrize(
    "stop_reasons",
    [
        ("environment_terminated",),
        ("predicted_collision",),
        ("clock_rewind",),
        ("point_cloud_timeout", "environment_terminated"),
    ],
)
def test_fatal_safety_zero_cannot_count_after_certified_goal(
    stop_reasons: tuple[str, ...],
) -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )

    executor.compute_action(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                stop_reasons=stop_reasons,
                motion_allowed=False,
            ),
        )
    )

    assert executor.status()["zero_write_streak"] == 0
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(
                4,
                (0.0, 0.0, 0.0),
                stop_reasons=("point_cloud_timeout",),
                motion_allowed=False,
            ),
        )
    ) is False
    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(
                5,
                (0.0, 0.0, 0.0),
                stop_reasons=("point_cloud_timeout",),
                motion_allowed=False,
            ),
        )
    ) is True


def test_safety_stopped_zero_without_true_cannot_complete() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )

    for sequence in range(2, 5):
        executor.compute_action(
            _state(
                sequence,
                goal=_goal(False, sequence),
                write=_write(
                    sequence,
                    (0.0, 0.0, 0.0),
                    stop_reasons=("point_cloud_timeout",),
                    motion_allowed=False,
                ),
            )
        )

    assert executor.status()["goal_rising_edge_seen"] is False
    assert executor.status()["zero_write_streak"] == 0
    assert executor.is_done(_state(5)) is False


@pytest.mark.parametrize(
    "invalid_zero",
    [
        _write(3, (0.0, 0.0, 0.0), owner_id="other_writer"),
        _write(
            3,
            (0.0, 0.0, 0.0),
            requested=(0.1, 0.0, 0.0),
        ),
    ],
)
def test_wrong_owner_or_unrequested_zero_cannot_complete(
    invalid_zero: dict,
) -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    assert executor.status()["zero_write_streak"] == 0

    assert executor.is_done(
        _state(3, goal=_goal(True, 3), write=invalid_zero)
    ) is False
    assert executor.status()["zero_write_streak"] == 0

    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(5, (0.0, 0.0, 0.0)),
        )
    ) is True


def test_nonzero_after_true_resets_zero_streak() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(3, (0.05, 0.0, 0.0)),
        )
    )

    assert executor.status()["zero_write_streak"] == 0
    assert executor.status()["post_goal_nonzero_write_count"] == 1


def test_write_sequence_gap_restarts_consecutive_zero_count() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(10, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(11, (0.0, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(13, (0.0, 0.0, 0.0)),
        )
    )

    assert executor.status()["zero_write_streak"] == 1
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(14, (0.0, 0.0, 0.0)),
        )
    ) is True


def test_missing_state_step_restarts_consecutive_zero_count() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(3, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 1

    # write_sequence 虽连续，但中间缺失了一个控制 tick。
    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(4, (0.0, 0.0, 0.0), timestamp=0.10),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 1
    assert executor.is_done(
        _state(
            6,
            goal=_goal(True, 6),
            write=_write(5, (0.0, 0.0, 0.0), timestamp=0.12),
        )
    ) is True


def test_timestamp_rollback_restarts_consecutive_zero_count() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(3, (0.0, 0.0, 0.0), timestamp=0.06),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 1

    # 序号和 state.step 连续也不能掩盖 policy 写入时钟回退。
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0), timestamp=0.05),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 1
    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(5, (0.0, 0.0, 0.0), timestamp=0.07),
        )
    ) is True


def test_zero_not_later_than_goal_true_receipt_cannot_count() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2, timestamp=0.04),
            write=_write(2, (0.0, 0.0, 0.0), timestamp=0.04),
        )
    )
    assert executor.status()["zero_write_streak"] == 0

    # 即使来自下一个 observation，与 true 同时的零速仍非到达后证据。
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(3, (0.0, 0.0, 0.0), timestamp=0.04),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 0
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0), timestamp=0.06),
        )
    ) is True


def test_timestamp_and_state_step_are_fallback_before_write_sequence_exists() -> None:
    executor = _executor(zero_ticks=2)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(None, (0.2, 0.0, 0.0), timestamp=0.02),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(None, (0.0, 0.0, 0.0), timestamp=0.04),
        )
    )
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(None, (0.0, 0.0, 0.0), timestamp=0.06),
        )
    ) is False
    assert executor.is_done(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(None, (0.0, 0.0, 0.0), timestamp=0.08),
        )
    ) is True


def test_false_after_true_requires_new_policy_activity() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.1, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            3,
            goal=_goal(False, 3),
            write=_write(3, (0.0, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0)),
        )
    )

    assert executor.is_done(
        _state(
            5,
            goal=_goal(True, 5),
            write=_write(5, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.status()["policy_activity_seen"] is False

    executor.compute_action(
        _state(
            6,
            goal=_goal(False, 6),
            write=_write(6, (0.2, 0.0, 0.0)),
        )
    )
    assert executor.is_done(
        _state(
            7,
            goal=_goal(True, 7),
            write=_write(7, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            8,
            goal=_goal(True, 8),
            write=_write(8, (0.0, 0.0, 0.0)),
        )
    ) is True


def test_reset_drops_previous_navigation_generation_evidence() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    assert executor.is_done(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    ) is False
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(3, (0.0, 0.0, 0.0)),
        )
    ) is True
    first_generation = executor.status()["generation"]

    executor.reset(_plan(execution_phase="carry_nav_to_place"))
    executor.compute_action(
        _state(
            4,
            goal=_goal(True, 4),
            write=_write(4, (0.0, 0.0, 0.0)),
        )
    )

    status = executor.status()
    assert status["generation"] == first_generation + 1
    assert status["execution_phase"] == "carry_nav_to_place"
    assert status["fresh_false_seen"] is False
    assert status["policy_activity_seen"] is False
    assert status["goal_rising_edge_seen"] is False
    assert status["zero_write_streak"] == 0
    assert status["done"] is False


def test_status_exposes_goal_and_acceptance_contract() -> None:
    executor = _executor()

    status = executor.status()

    assert status["backend"] == "scan_ros2_goal_event"
    assert status["goal"] == (1.0, 2.0, 0.4)
    assert status["goal_z"] == 0.3
    assert status["execution_phase"] == "nav_to_pick"
    assert status["acceptance_mode"] == (
        "fresh_false_execution_activity_true_stair_fresh_bspline_"
        "twist_or_terminal_hold_policy_zero_hold"
    )
    assert status["failed"] is False
    assert status["failure_reason"] == ""


def test_stair_freeze_blocks_completion_until_release_and_fresh_write() -> None:
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(required_zero_write_ticks=1),
        stair_freeze_config=ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            root_release_settle_time_s=0.0,
            post_release_stable_time_s=0.02,
            certified_progress_m=0.01,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        ),
    )
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
        (0.8, 0.0, 0.30),
    )
    path_stamp_ns = 100_000_001
    plan = NavPlan(
        goal=NavGoal(x=0.8, y=0.0, z=0.6, yaw=0.0),
        waypoints=((0.0, 0.0, 0.0), (0.8, 0.0, 0.3)),
        metadata={
            "execution_phase": "nav_to_pick",
            "reference_path_3d_ground": path,
            "reference_path_config": "unit-test",
            "reference_path_sha256": "b" * 64,
            "reference_path_points_sha256": hash_ground_path_points(path),
            "reference_path_topic": "/initial_path",
            "reference_path_stair_segment_indices": ((1, 3),),
        },
    )
    executor.reset(plan)

    activated = executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
            pose_xy=(0.2, 0.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    path,
                    sequence=1,
                    stamp_ns=path_stamp_ns,
                ),
                "scan_controller_status_last_report": _controller_status(
                    status_sequence=1,
                    acceptance_sequence=1,
                    receipt_timestamp=0.13,
                    header_stamp_ns=110_000_000,
                    reference_path_stamp_ns=path_stamp_ns,
                    bspline_header_stamp_ns=120_000_000,
                    start_time_ns=125_000_000,
                    traj_id=1,
                ),
            },
        )
    )
    assert activated.source == "scan_stair_freeze_activated"
    target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    release = None
    for step in range(2, 12):
        goal = _goal(True, step) if step >= 3 else _goal(False, step)
        state = _state(
            step,
            goal=goal,
            write=_write(
                step,
                (0.0, 0.0, 0.0),
                stop_reasons=("scan_stair_freeze",),
                motion_allowed=False,
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
            ),
            pose_xy=(float(target[0]), float(target[1])),
        )
        assert executor.is_done(state) is False
        action = executor.compute_action(state)
        if action.source == "scan_stair_freeze_released":
            release = action
            break
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]

    assert release is not None
    status = executor.status()
    assert status["certified_root_lock_progress_seen"] is True
    # 冻结/释放尚未交接完成时的 true 必须被丢弃，不能跨阶段存活。
    assert status["goal_rising_edge_seen"] is False
    assert status["premature_true_count"] >= 1
    assert status["zero_write_streak"] == 0
    assert status["stair_freeze_finish_ready"] is False

    release_sequence = status["stair_freeze"]["release_write_sequence"]
    stable_state = _state(
        release_sequence + 1,
        goal=_goal(True, release_sequence + 1),
        write=_write(
            release_sequence + 1,
            (0.0, 0.0, 0.0),
            stop_reasons=("scan_stair_freeze_release",),
            motion_allowed=False,
        ),
        pose_xy=(float(target[0]), float(target[1])),
        root_z=float(target[2]),
    )
    assert executor.is_done(stable_state) is False
    stable_action = executor.compute_action(stable_state)
    assert stable_action.source == "scan_stair_freeze_post_release_stabilizing"
    stair_status = executor.status()["stair_freeze"]
    release_sequence = stair_status["release_write_sequence"]
    release_timestamp = stair_status["release_write_timestamp"]
    assert isinstance(release_timestamp, float)
    handoff_step = release_sequence + 2
    fresh_receipt = release_timestamp + 0.04
    fresh_stamp_ns = int(round(fresh_receipt * 1.0e9))
    fresh_accepted = _controller_status(
        status_sequence=2,
        acceptance_sequence=2,
        receipt_timestamp=fresh_receipt,
        header_stamp_ns=fresh_stamp_ns,
        reference_path_stamp_ns=path_stamp_ns,
        bspline_header_stamp_ns=fresh_stamp_ns,
        start_time_ns=fresh_stamp_ns + 1,
        traj_id=2,
        event=1,
        state=2,
    )
    assert executor.is_done(
        _state(
            handoff_step,
            goal=_goal(True, handoff_step, timestamp=fresh_receipt),
            write=_write(
                handoff_step,
                (0.0, 0.0, 0.0),
                timestamp=fresh_receipt,
                stop_reasons=(),
                motion_allowed=True,
            ),
            pose_xy=(float(target[0]), float(target[1])),
            root_z=float(target[2]),
            timestamp=fresh_receipt,
            extra_metadata={
                "scan_controller_status_last_report": fresh_accepted,
            },
        )
    ) is False
    tracking_receipt = fresh_receipt + 0.02
    tracking_status = _controller_status(
        status_sequence=3,
        acceptance_sequence=2,
        receipt_timestamp=tracking_receipt,
        header_stamp_ns=fresh_stamp_ns + 2,
        reference_path_stamp_ns=path_stamp_ns,
        bspline_header_stamp_ns=fresh_stamp_ns,
        start_time_ns=fresh_stamp_ns + 1,
        traj_id=2,
        event=4,
        state=10,
    )
    assert executor.is_done(
        _state(
            handoff_step + 1,
            goal=_goal(True, handoff_step + 1, timestamp=tracking_receipt),
            write=_write(
                handoff_step + 1,
                (0.0, 0.0, 0.0),
                timestamp=tracking_receipt,
            ),
            pose_xy=(float(target[0]), float(target[1])),
            root_z=float(target[2]),
            timestamp=tracking_receipt,
            extra_metadata={
                "scan_controller_status_last_report": tracking_status,
            },
        )
    ) is False
    # ControllerStatus 与 Twist 同 tick 仍不算“接受之后”的新命令；下一 tick
    # 才完成 SCAN 交接，随后还需要一个独立的到达后零速写入。
    resume_receipt = tracking_receipt + 0.02
    assert executor.is_done(
        _state(
            handoff_step + 2,
            goal=_goal(True, handoff_step + 2, timestamp=resume_receipt),
            write=_write(
                handoff_step + 2,
                (0.0, 0.0, 0.0),
                timestamp=resume_receipt,
            ),
            pose_xy=(float(target[0]), float(target[1])),
            root_z=float(target[2]),
            timestamp=resume_receipt,
            extra_metadata={
                "scan_controller_status_last_report": tracking_status,
            },
        )
    ) is False
    final_receipt = resume_receipt + 0.02
    assert executor.is_done(
        _state(
            handoff_step + 3,
            goal=_goal(True, handoff_step + 3, timestamp=final_receipt),
            write=_write(
                handoff_step + 3,
                (0.0, 0.0, 0.0),
                timestamp=final_receipt,
            ),
            pose_xy=(float(target[0]), float(target[1])),
            root_z=float(target[2]),
            timestamp=final_receipt,
            extra_metadata={
                "scan_controller_status_last_report": tracking_status,
            },
        )
    ) is True


def test_terminal_stair_hold_completes_with_goal_and_frozen_zero_writes() -> None:
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(required_zero_write_ticks=2),
        stair_freeze_config=ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            root_release_settle_time_s=0.0,
            certified_progress_m=0.01,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        ),
    )
    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.6, y=0.0, z=0.6, yaw=0.0),
            waypoints=((0.0, 0.0, 0.0), (0.6, 0.0, 0.3)),
            metadata={
                "execution_phase": "nav_to_pick",
                "reference_path_3d_ground": points,
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )

    activated = executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
            pose_xy=(0.2, 0.0),
        )
    )
    assert activated.source == "scan_stair_freeze_activated"
    target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    terminal_action = None
    for step in range(2, 12):
        action = executor.compute_action(
            _state(
                step,
                goal=_goal(False, step),
                write=_write(
                    step,
                    (0.0, 0.0, 0.0),
                    stop_reasons=("scan_stair_freeze",),
                    motion_allowed=False,
                    navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                ),
                pose_xy=(float(target[0]), float(target[1])),
                root_z=float(target[2]),
            )
        )
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        if action.source == "scan_stair_freeze_terminal_hold":
            terminal_action = action
            break

    assert terminal_action is not None
    status = executor.status()
    assert status["stair_freeze_finish_ready"] is True
    assert status["stair_freeze"]["terminal_hold"] is True
    assert status["stair_freeze"]["completed_component_count"] == 1

    goal_step = step + 1
    goal_state = _state(
        goal_step,
        goal=_goal(True, goal_step),
        write=_write(
            goal_step,
            (0.0, 0.0, 0.0),
            stop_reasons=("scan_stair_terminal_hold",),
            motion_allowed=False,
            navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
        ),
        pose_xy=(float(target[0]), float(target[1])),
        root_z=float(target[2]),
    )
    assert executor.is_done(goal_state) is False
    assert executor.compute_action(goal_state).source == (
        "scan_stair_freeze_terminal_hold"
    )

    for offset in range(1, 3):
        zero_step = goal_step + offset
        done = executor.is_done(
            _state(
                zero_step,
                goal=_goal(True, zero_step),
                write=_write(
                    zero_step,
                    (0.0, 0.0, 0.0),
                    stop_reasons=("scan_stair_terminal_hold",),
                    motion_allowed=False,
                    navigation_cmd_vel_inhibit_reason=(
                        "scan_stair_terminal_hold"
                    ),
                ),
                pose_xy=(float(target[0]), float(target[1])),
                root_z=float(target[2]),
            )
        )
        assert done is (offset == 2)

    status = executor.status()
    assert status["scan_controller_goal_reached_verified"] is True
    assert status["policy_zero_hold_verified"] is True
    assert status["zero_write_streak"] == 2


@pytest.mark.parametrize(
    "invalid_write",
    [
        _write(
            100,
            (0.0, 0.0, 0.0),
            stop_reasons=("scan_stair_terminal_hold",),
            motion_allowed=False,
            navigation_cmd_vel_inhibited=False,
            navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
        ),
        _write(
            100,
            (0.0, 0.0, 0.0),
            stop_reasons=("scan_stair_terminal_hold",),
            motion_allowed=False,
            navigation_cmd_vel_inhibited=True,
            navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
        ),
        _write(
            100,
            (0.0, 0.0, 0.0),
            stop_reasons=(
                "scan_stair_terminal_hold",
                "point_cloud_timeout",
            ),
            motion_allowed=False,
            navigation_cmd_vel_inhibited=True,
            navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
        ),
        _write(
            100,
            (0.0, 0.0, 0.0),
            stop_reasons=("scan_stair_terminal_hold",),
            motion_allowed=False,
            owner_id="other_writer",
            navigation_cmd_vel_inhibited=True,
            navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
        ),
    ],
)
def test_terminal_zero_requires_exact_phase_and_inhibit_provenance(
    invalid_write: dict,
) -> None:
    executor, target, terminal_step = _terminal_executor_at_hold(zero_ticks=1)
    goal_step = terminal_step + 1
    executor.compute_action(
        _state(
            goal_step,
            goal=_goal(True, goal_step),
            write=_write(
                goal_step,
                (0.0, 0.0, 0.0),
                stop_reasons=("scan_stair_terminal_hold",),
                motion_allowed=False,
                navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
            ),
            pose_xy=(target[0], target[1]),
            root_z=target[2],
        )
    )
    invalid_step = goal_step + 1
    invalid_write = {
        **invalid_write,
        "write_sequence": invalid_step,
        "timestamp": invalid_step * 0.02,
    }
    assert executor.is_done(
        _state(
            invalid_step,
            goal=_goal(True, invalid_step),
            write=invalid_write,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 0


def test_terminal_stop_reason_outside_terminal_phase_cannot_count() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1),
            write=_write(1, (0.2, 0.0, 0.0)),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2),
            write=_write(2, (0.0, 0.0, 0.0)),
        )
    )
    assert executor.is_done(
        _state(
            3,
            goal=_goal(True, 3),
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                stop_reasons=("scan_stair_terminal_hold",),
                motion_allowed=False,
                navigation_cmd_vel_inhibited=True,
                navigation_cmd_vel_inhibit_reason=(
                    "scan_stair_terminal_hold"
                ),
            ),
        )
    ) is False
    assert executor.status()["zero_write_streak"] == 0


def test_live_reference_path_hash_binds_ros_path_before_freeze() -> None:
    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    points_sha256 = hash_ground_path_points(points)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(exit_distance_m=0.0),
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.6, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": points_sha256,
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )

    action = executor.compute_action(
        _state(
            1,
            pose_xy=(0.2, 0.0),
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(points)
            },
        )
    )

    assert action.source == "scan_stair_freeze_activated"
    status = executor.status()
    assert status["live_reference_path_verified"] is True
    assert status["live_reference_path_points_sha256"] == points_sha256
    assert status["live_reference_path_terminal_yaw"] == pytest.approx(0.0)
    assert status["live_reference_path_goal_bound"] is True
    assert status["live_reference_path_goal_xy_error_m"] == pytest.approx(0.0)
    assert status["live_reference_path_goal_z_error_m"] == pytest.approx(0.0)
    assert status["live_reference_path_goal_yaw_error_rad"] == pytest.approx(0.0)
    assert status["stair_freeze"]["component_source"] == (
        "explicit_hash_bound_indices"
    )


def test_supervisor_sensor_stale_keeps_root_lock_and_requests_replan() -> None:
    """代际屏障通过后丢失点云必须立即急停并请求重规划。"""

    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
        (0.8, 0.0, 0.30),
    )
    path_stamp_ns = 1_000_000_001
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
        ),
        allow_carry_object_follow=True,
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.8, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "execution_phase": "carry_nav_to_place",
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": hash_ground_path_points(
                    points
                ),
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            timestamp=1.0,
            pose_xy=(0.2, 0.0),
            write=_write(1, (0.1, 0.0, 0.0), timestamp=1.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=path_stamp_ns,
                )
            },
        )
    )
    assert activated.source == "scan_stair_freeze_activated"
    target = tuple(
        activated.metadata["navigation_base_pose_lock_xyzyaw"]
    )

    waiting = executor.compute_action(
        _state(
            2,
            timestamp=1.02,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                2,
                (0.0, 0.0, 0.0),
                timestamp=1.02,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.02,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("point_cloud",),
                    )
                ),
            ),
        )
    )
    assert waiting.source == "scan_stair_sensor_acquisition_wait"
    assert waiting.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        target
    )
    status = executor.status()
    assert status["failed"] is False
    assert status["stair_sensor_acquisition_pending"] is True
    assert status["stair_freeze"]["progress_m"] == pytest.approx(0.0)

    acquired = executor.compute_action(
        _state(
            3,
            timestamp=1.04,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                timestamp=1.04,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.04,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("bspline",),
                    )
                ),
            ),
        )
    )
    assert acquired.source == "scan_stair_freeze_active"
    target = tuple(acquired.metadata["navigation_base_pose_lock_xyzyaw"])
    barrier = executor.status()["stair_freeze"][
        "sensor_acquisition_barrier"
    ]
    assert barrier["passed"] is True
    assert barrier["activation_timestamp"] == pytest.approx(1.0)
    assert barrier["write_sequence"] == 3
    assert barrier["write_timestamp"] == pytest.approx(1.04)
    assert barrier["progress_m_at_pass"] == pytest.approx(0.0)
    assert barrier["navigation_status_observed_report"]["status"][
        "stale_inputs"
    ] == ["bspline"]

    failed = executor.compute_action(
        _state(
            4,
            timestamp=1.06,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                4,
                (0.0, 0.0, 0.0),
                timestamp=1.06,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.06,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("point_cloud",),
                    )
                ),
            ),
        )
    )

    _assert_stair_emergency_hold_remains_latched(
        executor,
        failed,
        target=target,
        failure_reason="stair_sensor_freshness_fault",
        origin_phase="active",
        next_step=5,
        next_timestamp=1.08,
    )
    status = executor.status()
    assert status["stair_freeze"]["progress_m"] > 0.0
    assert status["stair_freeze"]["sensor_safety_fault_reasons"] == [
        "supervisor_point_cloud_stale"
    ]


def test_invalid_freeze_policy_write_keeps_root_lock_and_requests_replan() -> None:
    """屏障后的伪造 stop reason 必须由 executor 映射成可重规划急停。"""

    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
        (0.8, 0.0, 0.30),
    )
    path_stamp_ns = 1_000_000_011
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
        ),
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.8, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": hash_ground_path_points(points),
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            timestamp=1.0,
            pose_xy=(0.2, 0.0),
            write=_write(1, (0.1, 0.0, 0.0), timestamp=1.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=path_stamp_ns,
                )
            },
        )
    )
    target = tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"])
    acquired = executor.compute_action(
        _state(
            2,
            timestamp=1.02,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                2,
                (0.0, 0.0, 0.0),
                timestamp=1.02,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.02,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("bspline",),
                    )
                ),
            ),
        )
    )
    target = tuple(acquired.metadata["navigation_base_pose_lock_xyzyaw"])
    failed = executor.compute_action(
        _state(
            3,
            timestamp=1.04,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                timestamp=1.04,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze", "environment_closed"),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.04,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("bspline",),
                    )
                ),
            ),
        )
    )

    assert failed.source == "scan_stair_emergency_hold"
    assert failed.metadata["navigation_global_replan_requested"] is True
    assert failed.metadata["navigation_emergency_stop_reason"] == (
        "stair_policy_freeze_write_fault"
    )
    status = executor.status()
    assert status["failed"] is True
    assert status["failure_reason"] == "stair_policy_freeze_write_fault"
    assert status["stair_freeze"]["policy_freeze_write_fault_reasons"] == [
        "invalid_stair_freeze_policy_write"
    ]


def test_new_path_sensor_acquisition_timeout_requests_replan() -> None:
    """新 Path 传感器屏障超时后由 executor 保持 root 并请求 PCT 重规划。"""

    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
        (0.8, 0.0, 0.30),
    )
    path_stamp_ns = 1_000_000_002
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            activation_timeout_s=0.03,
            supervisor_sensor_status_timeout_s=0.25,
        ),
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.8, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": hash_ground_path_points(points),
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            timestamp=1.0,
            pose_xy=(0.2, 0.0),
            write=_write(1, (0.1, 0.0, 0.0), timestamp=1.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=path_stamp_ns,
                )
            },
        )
    )
    target = tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"])
    waiting = executor.compute_action(
        _state(
            2,
            timestamp=1.02,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                2,
                (0.0, 0.0, 0.0),
                timestamp=1.02,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.02,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("point_cloud",),
                    )
                ),
            ),
        )
    )
    assert waiting.source == "scan_stair_sensor_acquisition_wait"

    failed = executor.compute_action(
        _state(
            3,
            timestamp=1.04,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            write=_write(
                3,
                (0.0, 0.0, 0.0),
                timestamp=1.04,
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
                navigation_cmd_vel_inhibit_reason="scan_stair_freeze",
                navigation_status_observed_report=(
                    _observed_navigation_status(
                        receipt_timestamp=1.04,
                        path_stamp_ns=path_stamp_ns,
                        stale_inputs=("point_cloud",),
                    )
                ),
            ),
        )
    )
    assert failed.source == "scan_stair_emergency_hold"
    assert failed.metadata["navigation_global_replan_requested"] is True
    assert failed.metadata["navigation_emergency_stop_reason"] == (
        "stair_sensor_freshness_fault"
    )
    status = executor.status()
    assert status["failed"] is True
    assert status["failure_reason"] == "stair_sensor_freshness_fault"
    assert status["stair_freeze"]["progress_m"] == pytest.approx(0.0)
    assert status["stair_freeze"]["sensor_acquisition_barrier"][
        "passed"
    ] is False


@pytest.mark.parametrize(
    ("points", "terminal_yaw", "failure_reason"),
    [
        (
            ((0.0, 0.0, 0.0), (0.500002, 0.0, 0.0)),
            0.0,
            "scan_reference_path_goal_xy_mismatch",
        ),
        (
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0002)),
            0.0,
            "scan_reference_path_goal_z_mismatch",
        ),
        (
            ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
            2.0e-5,
            "scan_reference_path_goal_yaw_mismatch",
        ),
    ],
)
def test_live_reference_path_must_bind_exact_terminal_nav_goal(
    points: tuple[tuple[float, float, float], ...],
    terminal_yaw: float,
    failure_reason: str,
) -> None:
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True)
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.5, y=0.0, z=0.3, yaw=0.0),
            waypoints=points,
        )
    )

    action = executor.compute_action(
        _state(
            1,
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    terminal_yaw=terminal_yaw,
                )
            },
        )
    )

    assert action.source == "scan_ros2_navigation_failed"
    assert executor.status()["failure_reason"] == failure_reason
    assert executor.status()["live_reference_path_goal_bound"] is False


def test_dynamic_path_tombstone_during_active_stair_keeps_all_locks() -> None:
    # 不提供 expected sha，确保覆盖生产动态 Path 的楼梯中途 tombstone 分支。
    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.21, 0.0, 0.10),
        (0.4, 0.0, 0.10),
        (0.41, 0.0, 0.20),
        (0.6, 0.0, 0.20),
        (0.61, 0.0, 0.30),
    )
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(exit_distance_m=0.0),
        allow_carry_object_follow=True,
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.61, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "execution_phase": "carry_nav_to_place",
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            pose_xy=(0.2, 0.0),
            goal=_goal(False, 1),
            write=_write(1, (0.0, 0.0, 0.0)),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=1_000_000_001,
                )
            },
            timestamp=1.0,
        )
    )
    assert activated.source == "scan_stair_freeze_activated"
    assert executor.status()["stair_freeze"]["component_source"] == (
        "geometry_heuristic"
    )
    target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    failed = executor.compute_action(
        _state(
            2,
            pose_xy=(float(target[0]), float(target[1])),
            root_z=float(target[2]),
            goal=_goal(False, 2),
            write=_write(
                2,
                (0.0, 0.0, 0.0),
                motion_allowed=False,
                stop_reasons=("scan_stair_freeze",),
            ),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    (),
                    sequence=2,
                    stamp_ns=1_000_000_002,
                )
            },
            timestamp=1.02,
        )
    )

    _assert_stair_emergency_hold_remains_latched(
        executor,
        failed,
        target=target,
        failure_reason="scan_reference_path_cleared_during_stair_freeze",
        origin_phase="active",
        next_step=3,
        next_timestamp=1.04,
    )


def test_same_stamp_path_conflict_in_full_lock_settle_keeps_all_locks() -> None:
    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    initial_stamp_ns = 1_000_000_001
    points_sha256 = hash_ground_path_points(points)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(
            speed_mps=100.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=1.0,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        ),
        allow_carry_object_follow=True,
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.6, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "execution_phase": "carry_nav_to_place",
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": points_sha256,
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            timestamp=1.0,
            pose_xy=(0.2, 0.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=initial_stamp_ns,
                )
            },
        )
    )
    assert activated.source == "scan_stair_freeze_activated"

    full_lock = executor.compute_action(
        _state(
            2,
            timestamp=1.02,
            pose_xy=(0.2, 0.0),
            root_z=0.3,
        )
    )
    assert full_lock.source == "scan_stair_freeze_full_lock_settle"
    assert executor.status()["stair_freeze"]["phase"] == "full_lock_settle"
    target = full_lock.metadata["navigation_base_pose_lock_xyzyaw"]

    conflicting_points = ((0.0, 0.0, 0.0), (0.7, 0.0, 0.31))
    failed = executor.compute_action(
        _state(
            3,
            timestamp=1.04,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    conflicting_points,
                    sequence=2,
                    stamp_ns=initial_stamp_ns,
                )
            },
        )
    )

    _assert_stair_emergency_hold_remains_latched(
        executor,
        failed,
        target=target,
        failure_reason="scan_reference_path_stamp_conflict",
        origin_phase="full_lock_settle",
        next_step=4,
        next_timestamp=1.06,
    )


def test_stair_controller_timestamp_rollback_keeps_all_locks() -> None:
    points = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    points_sha256 = hash_ground_path_points(points)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True),
        stair_freeze_config=ScanStairFreezeConfig(exit_distance_m=0.0),
        allow_carry_object_follow=True,
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.6, y=0.0, z=0.6, yaw=0.0),
            waypoints=points,
            metadata={
                "execution_phase": "carry_nav_to_place",
                "reference_path_3d_ground": points,
                "reference_path_points_sha256": points_sha256,
                "reference_path_stair_segment_indices": ((1, 3),),
            },
        )
    )
    activated = executor.compute_action(
        _state(
            1,
            timestamp=1.0,
            pose_xy=(0.2, 0.0),
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    points,
                    stamp_ns=1_000_000_001,
                )
            },
        )
    )
    assert activated.source == "scan_stair_freeze_activated"
    target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    # state timestamp 回退会直接命中 stair controller 的 `_control_dt`，
    # 与 policy report 中名为 clock_rewind 的停车原因不是同一条路径。
    failed = executor.compute_action(
        _state(
            2,
            timestamp=0.5,
            pose_xy=(target[0], target[1]),
            root_z=target[2],
        )
    )

    assert failed.metadata["navigation_stair_freeze_error"] == (
        "楼梯冻结期间仿真时钟发生回退。"
    )
    _assert_stair_emergency_hold_remains_latched(
        executor,
        failed,
        target=target,
        failure_reason="scan_stair_freeze_failed",
        origin_phase="active",
        next_step=3,
        next_timestamp=0.6,
    )


def test_live_reference_path_mismatch_stops_and_requests_replan() -> None:
    points = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True)
    )
    executor.reset(
        NavPlan(
            goal=NavGoal(x=0.5, y=0.0, z=0.3, yaw=0.0),
            waypoints=points,
            metadata={
                "reference_path_points_sha256": "f" * 64,
            },
        )
    )

    action = executor.compute_action(
        _state(
            1,
            extra_metadata={
                "scan_reference_path_last_report": _path_report(points)
            },
        )
    )

    assert action.source == "scan_ros2_navigation_failed"
    assert action.metadata["navigation_emergency_stop"] is True
    assert action.metadata["navigation_global_replan_requested"] is True
    assert executor.status()["failure_reason"] == "scan_reference_path_mismatch"


def test_dynamic_pct_goal_ack_tombstone_and_new_path_are_ordered() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    plan = planner.plan(
        _state(0),
        NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
    )
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True)
    )
    executor.reset(plan)
    old_points = ((-1.0, 0.0, 0.0), (-0.5, 0.0, 0.0))
    new_points = ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0))

    publish = executor.compute_action(
        _state(
            1,
            extra_metadata={
                "scan_reference_path_last_report": _path_report(
                    old_points,
                    stamp_ns=10_000_000,
                )
            },
        )
    )
    assert publish.source == "scan_pct_goal_publish"
    assert publish.metadata["navigation_cmd_vel_inhibit"] is True
    request = publish.metadata["navigation_pct_goal_request"]
    assert request["generation"] == 1
    assert request["position_base_xyz"] == (1.0, 2.0, 0.3)
    assert executor.status()["live_reference_path_verified"] is False

    waiting = executor.compute_action(
        _state(
            2,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": _path_report(
                    old_points,
                    sequence=2,
                    stamp_ns=10_000_000,
                ),
            },
        )
    )
    assert waiting.source == "scan_pct_goal_waiting_for_transport_ack"
    assert executor.status()["pct_goal_acknowledged"] is True
    assert executor.status()["pct_goal_transport_acknowledged"] is False
    assert executor.status()["live_reference_path_verified"] is False

    empty = ()
    waiting = executor.compute_action(
        _state(
            3,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": _path_report(
                    empty,
                    sequence=3,
                    stamp_ns=21_000_000,
                ),
            },
        )
    )
    assert waiting.source == "scan_pct_goal_waiting_for_path"
    assert executor.status()["reference_path_tombstone_stamp_ns"] == 21_000_000
    assert executor.status()["pct_goal_transport_acknowledged"] is True

    running = executor.compute_action(
        _state(
            4,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": _path_report(
                    new_points,
                    sequence=4,
                    stamp_ns=22_000_000,
                    terminal_yaw=0.4,
                ),
            },
        )
    )
    assert running.source == "scan_ros2_navigation"
    status = executor.status()
    assert status["live_reference_path_verified"] is True
    assert status["live_reference_path_stamp_ns"] == 22_000_000


def test_dynamic_pct_goal_retries_same_generation_until_remote_path_ack() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(
            require_live_reference_path=True,
            pct_goal_transport_retry_interval_s=0.10,
        )
    )
    executor.reset(
        planner.plan(
            _state(0),
            NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
        )
    )

    first = executor.compute_action(_state(1, timestamp=0.02))
    assert first.source == "scan_pct_goal_publish"
    assert first.metadata["navigation_pct_goal_request"][
        "transport_retry"
    ] is False
    executor.compute_action(
        _state(
            2,
            timestamp=0.04,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1)
            },
        )
    )

    retry = executor.compute_action(_state(8, timestamp=0.16))
    assert retry.source == "scan_pct_goal_transport_retry"
    request = retry.metadata["navigation_pct_goal_request"]
    assert request["generation"] == 1
    assert request["transport_retry"] is True
    same_tick = executor.compute_action(_state(8, timestamp=0.16))
    assert same_tick.source == "scan_pct_goal_waiting_for_transport_ack"
    assert "navigation_pct_goal_request" not in same_tick.metadata

    tombstone = executor.compute_action(
        _state(
            9,
            timestamp=0.18,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": _path_report(
                    (),
                    sequence=2,
                    stamp_ns=21_000_000,
                ),
            },
        )
    )
    assert tombstone.source == "scan_pct_goal_waiting_for_path"
    assert "navigation_pct_goal_request" not in tombstone.metadata
    status = executor.status()
    assert status["pct_goal_transport_acknowledged"] is True
    assert status["pct_goal_request_action_count"] == 2
    assert status["pct_goal_transport_retry_count"] == 1


def test_dynamic_pct_goal_rejects_duplicate_publish_ack_and_stale_next_phase_path() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(require_live_reference_path=True)
    )
    first_plan = planner.plan(
        _state(0),
        NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
    )
    executor.reset(first_plan)
    executor.compute_action(_state(1))
    executor.compute_action(
        _state(
            2,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
            },
        )
    )
    duplicate = _pct_goal_report(
        generation=1,
        sequence=2,
        stamp_ns=21_000_000,
    )
    failed = executor.compute_action(
        _state(3, extra_metadata={"scan_pct_goal_last_report": duplicate})
    )
    assert failed.source == "scan_ros2_navigation_failed"
    assert executor.status()["failure_reason"] == "pct_goal_publish_not_exactly_once"

    second_plan = planner.plan(
        _state(4),
        NavGoal(x=3.0, y=4.0, z=3.6, yaw=-0.2),
    )
    executor.reset(second_plan)
    stale_path = _path_report(
        ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0)),
        sequence=5,
        stamp_ns=22_000_000,
    )
    action = executor.compute_action(
        _state(
            5,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": stale_path,
            },
        )
    )
    assert action.source == "scan_pct_goal_publish"
    assert action.metadata["navigation_pct_goal_request"]["generation"] == 2
    assert executor.status()["live_reference_path_verified"] is False


def test_required_live_reference_path_has_finite_wait_timeout() -> None:
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(
            require_live_reference_path=True,
            live_reference_path_timeout_s=0.05,
        )
    )
    executor.reset(_plan())

    assert executor.compute_action(_state(0, timestamp=0.0)).source == (
        "scan_ros2_navigation"
    )
    failed = executor.compute_action(_state(3, timestamp=0.06))
    assert failed.source == "scan_ros2_navigation_failed"
    assert executor.status()["failure_reason"] == "scan_reference_path_timeout"


def test_dynamic_pct_goal_has_separate_transport_ack_timeout() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(
            require_live_reference_path=True,
            pct_goal_transport_ack_timeout_s=0.05,
            pct_goal_transport_retry_interval_s=0.01,
        )
    )
    executor.reset(
        planner.plan(
            _state(0),
            NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
        )
    )
    executor.compute_action(_state(1, timestamp=0.01))
    executor.compute_action(
        _state(
            2,
            timestamp=0.02,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1)
            },
        )
    )

    failed = executor.compute_action(_state(8, timestamp=0.08))

    assert failed.source == "scan_ros2_navigation_failed"
    assert executor.status()["failure_reason"] == "pct_goal_transport_ack_timeout"


def test_dynamic_pct_goal_path_timeout_starts_after_transport_ack() -> None:
    planner = ScanRos2LifecyclePlanner(publish_pct_goal=True)
    executor = ScanRos2NavExecutor(
        ScanRos2NavExecutorConfig(
            require_live_reference_path=True,
            live_reference_path_timeout_s=0.05,
        )
    )
    executor.reset(
        planner.plan(
            _state(0),
            NavGoal(x=1.0, y=2.0, z=0.3, yaw=0.4),
        )
    )
    executor.compute_action(_state(1, timestamp=0.01))
    executor.compute_action(
        _state(
            2,
            timestamp=0.02,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1)
            },
        )
    )
    executor.compute_action(
        _state(
            4,
            timestamp=0.04,
            extra_metadata={
                "scan_pct_goal_last_report": _pct_goal_report(generation=1),
                "scan_reference_path_last_report": _path_report(
                    (),
                    sequence=2,
                    stamp_ns=21_000_000,
                ),
            },
        )
    )

    waiting = executor.compute_action(_state(8, timestamp=0.08))
    assert waiting.source == "scan_pct_goal_waiting_for_path"
    failed = executor.compute_action(_state(10, timestamp=0.10))
    assert failed.source == "scan_ros2_navigation_failed"
    assert executor.status()["failure_reason"] == "scan_reference_path_timeout"


def test_progress_watchdog_fails_only_at_complete_fixed_window() -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )

    before_boundary = _state(
        2,
        goal=_goal(False, 2, timestamp=3.98),
        write=_write(2, (0.2, 0.0, 0.0), timestamp=3.98),
        timestamp=0.02,
    )
    assert executor.is_done(before_boundary) is False
    assert executor.status()["failed"] is False
    assert executor.status()[
        "progress_watchdog_elapsed_without_progress_s"
    ] == pytest.approx(3.98)

    boundary = _state(
        3,
        goal=_goal(False, 3, timestamp=4.0),
        write=_write(3, (0.2, 0.0, 0.0), timestamp=4.0),
        timestamp=0.04,
    )
    assert executor.is_done(boundary) is False
    action = executor.compute_action(boundary)
    status = executor.status()
    assert status["failed"] is True
    assert status["failure_reason"] == "locomotion_stall"
    assert status["phase"] == "failed"
    assert status["progress_watchdog_trigger_count"] == 1
    assert status["progress_watchdog_failure_timestamp"] == 4.0
    assert status["progress_watchdog_failure_pose_xy"] == (0.0, 0.0)
    assert action.metadata == {
        "navigation_emergency_stop": True,
        "navigation_emergency_stop_reason": "locomotion_stall",
        "navigation_global_replan_requested": True,
    }


@pytest.mark.parametrize(
    ("displacement", "failed"),
    [
        (0.029, True),
        (0.030, False),
        (0.031, False),
    ],
)
def test_progress_watchdog_net_displacement_boundary(
    displacement: float,
    failed: bool,
) -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(False, 2, timestamp=4.0),
            write=_write(2, (0.2, 0.0, 0.0), timestamp=4.0),
            pose_xy=(displacement, 0.0),
        )
    )

    status = executor.status()
    assert status["failed"] is failed
    if failed:
        assert status["progress_watchdog_last_displacement_m"] == pytest.approx(
            displacement
        )
    else:
        assert status["progress_watchdog_progress_event_count"] == 1
        assert status["progress_watchdog_window_start_pose_xy"] == (
            displacement,
            0.0,
        )
        assert status["progress_watchdog_last_displacement_m"] == 0.0


def test_progress_watchdog_fixed_window_rejects_collision_oscillation() -> None:
    executor = _executor()
    samples = (
        (1, 0.0, (0.0, 0.0)),
        (2, 2.0, (0.05, 0.0)),
        (3, 4.0, (0.0, 0.0)),
    )
    for sequence, timestamp, pose_xy in samples:
        executor.compute_action(
            _state(
                sequence,
                goal=_goal(False, sequence, timestamp=timestamp),
                write=_write(
                    sequence,
                    (0.2, 0.0, 0.0),
                    timestamp=timestamp,
                ),
                pose_xy=pose_xy,
            )
        )

    status = executor.status()
    assert status["failed"] is True
    assert status["progress_watchdog_last_displacement_m"] == 0.0


@pytest.mark.parametrize(
    "write",
    [
        _write(2, (0.0, 0.0, 0.4), timestamp=8.0),
        _write(2, (-0.2, 0.0, 0.0), timestamp=8.0),
        _write(2, (0.0, 0.2, 0.0), timestamp=8.0),
        _write(
            2,
            (0.2, 0.0, 0.0),
            timestamp=8.0,
            stop_reasons=("point_cloud_timeout",),
            motion_allowed=False,
        ),
        _write(
            2,
            (0.04, 0.0, 0.0),
            timestamp=8.0,
            requested=(0.2, 0.0, 0.0),
        ),
    ],
)
def test_ineligible_motion_clears_progress_watchdog_without_failure(
    write: dict,
) -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )
    executor.compute_action(_state(2, goal=_goal(False, 2), write=write))

    status = executor.status()
    assert status["failed"] is False
    assert status["progress_watchdog_active"] is False
    assert status["progress_watchdog_reset_count"] == 1
    assert status["progress_watchdog_last_pause_reason"] == (
        "ineligible_policy_write"
    )


def test_policy_write_gap_restarts_progress_watchdog_window() -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(False, 2, timestamp=3.9),
            write=_write(3, (0.2, 0.0, 0.0), timestamp=3.9),
        )
    )
    executor.compute_action(
        _state(
            3,
            goal=_goal(False, 3, timestamp=4.1),
            write=_write(4, (0.2, 0.0, 0.0), timestamp=4.1),
        )
    )

    status = executor.status()
    assert status["failed"] is False
    assert status["progress_watchdog_active"] is True
    assert status["progress_watchdog_window_start_timestamp"] == 3.9
    assert status["progress_watchdog_elapsed_without_progress_s"] == pytest.approx(
        0.2
    )


def test_goal_true_at_watchdog_boundary_takes_precedence() -> None:
    executor = _executor(zero_ticks=1)
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(True, 2, timestamp=4.0),
            write=_write(2, (0.0, 0.0, 0.0), timestamp=4.0),
        )
    )

    status = executor.status()
    assert status["failed"] is False
    assert status["goal_rising_edge_seen"] is True
    assert status["progress_watchdog_active"] is False
    assert status["phase"] == "goal_reached_waiting_for_zero_hold"


def test_goal_distance_worsening_does_not_fail_when_net_xy_moves() -> None:
    executor = _executor()
    executor.compute_action(
        _state(
            1,
            goal=_goal(False, 1, timestamp=0.0),
            write=_write(1, (0.2, 0.0, 0.0), timestamp=0.0),
        )
    )
    executor.compute_action(
        _state(
            2,
            goal=_goal(False, 2, timestamp=4.0),
            write=_write(2, (0.2, 0.0, 0.0), timestamp=4.0),
            pose_xy=(-0.04, 0.0),
        )
    )

    status = executor.status()
    assert status["failed"] is False
    assert status["progress_watchdog_progress_event_count"] == 1
    assert status["progress_watchdog_source"] == "fixed_window_net_xy"
