from __future__ import annotations

import math
from pathlib import Path

import pytest

from source.interfaces import SimulationState
from source.navigation.scan_stair_freeze import (
    ScanStairFreezeConfig,
    ScanStairFreezeController,
    extract_stair_components,
    load_scan_reference_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _stamp(stamp_ns: int) -> dict[str, int]:
    return {
        "sec": stamp_ns // 1_000_000_000,
        "nanosec": stamp_ns % 1_000_000_000,
    }


def _controller_status_report(
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
    candidate: dict[str, object] | None = None,
) -> dict[str, object]:
    """构造与 Isaac runtime 完全同构的 typed controller 快照。"""

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
        "reason": "unit-test controller status",
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


def _terminal_goal(
    path: tuple[tuple[float, float, float], ...],
    *,
    body_height_m: float = 0.30,
    yaw: float | None = None,
) -> tuple[float, float, float, float]:
    """构造与 Path 地面末点严格一致的 base 高度目标。"""

    end = path[-1]
    if yaw is None:
        previous = next(
            point
            for point in reversed(path[:-1])
            if math.hypot(end[0] - point[0], end[1] - point[1]) > 1.0e-9
        )
        yaw = math.atan2(end[1] - previous[1], end[0] - previous[0])
    return (end[0], end[1], end[2] + body_height_m, yaw)


def _policy_report(
    sequence: int,
    *,
    timestamp: float,
    owner_id: str = "scan_cmd_vel",
    motion_allowed: bool = False,
    stop_reasons: tuple[str, ...] = ("scan_stair_freeze",),
    requested_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    written_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    navigation_cmd_vel_inhibited: bool | None = None,
    navigation_cmd_vel_inhibit_reason: str | None = None,
    cmd_vel_source_sequence: int | None = None,
    cmd_vel_source_receipt_timestamp: float | None = None,
    cmd_vel_sample_received_this_tick: bool | None = None,
    cmd_vel_sample_drained_this_tick: bool | None = None,
    last_cmd_vel_drain_sequence: int | None = None,
    last_cmd_vel_drain_receipt_timestamp: float | None = None,
) -> dict[str, object]:
    """构造与 Isaac runtime 字段一致的 policy 实写报告。"""

    inhibited = (
        bool(not motion_allowed or stop_reasons)
        if navigation_cmd_vel_inhibited is None
        else navigation_cmd_vel_inhibited
    )
    source_sequence = (
        sequence
        if cmd_vel_source_sequence is None
        else cmd_vel_source_sequence
    )
    source_timestamp = (
        timestamp
        if cmd_vel_source_receipt_timestamp is None
        else cmd_vel_source_receipt_timestamp
    )
    received_this_tick = (
        not inhibited
        if cmd_vel_sample_received_this_tick is None
        else cmd_vel_sample_received_this_tick
    )
    drained_this_tick = (
        inhibited
        if cmd_vel_sample_drained_this_tick is None
        else cmd_vel_sample_drained_this_tick
    )
    drain_sequence = last_cmd_vel_drain_sequence
    drain_timestamp = last_cmd_vel_drain_receipt_timestamp
    if inhibited and drain_sequence is None and drain_timestamp is None:
        drain_sequence = source_sequence
        drain_timestamp = source_timestamp
    return {
        "write_sequence": sequence,
        "timestamp": timestamp,
        "owner_id": owner_id,
        "requested_command": list(requested_command),
        "written_command": list(written_command),
        "motion_allowed": motion_allowed,
        "stop_reasons": list(stop_reasons),
        "navigation_cmd_vel_inhibited": inhibited,
        "navigation_cmd_vel_inhibit_reason": (
            (
                "scan_stair_freeze"
                if navigation_cmd_vel_inhibit_reason is None
                else navigation_cmd_vel_inhibit_reason
            )
            if inhibited
            else None
        ),
        "cmd_vel_source_sequence": source_sequence,
        "cmd_vel_source_receipt_timestamp": source_timestamp,
        "cmd_vel_sample_received_this_tick": received_this_tick,
        "cmd_vel_sample_drained_this_tick": drained_this_tick,
        "last_cmd_vel_drain_sequence": drain_sequence,
        "last_cmd_vel_drain_receipt_timestamp": drain_timestamp,
    }


def _navigation_status_observed_report(
    *,
    receipt_timestamp: float,
    path_stamp_ns: int,
    stale_inputs: tuple[str, ...] = (),
    consecutive_scan_failures: int = 0,
    identity_valid: bool = True,
    state: int = 5,
    reason: str = "scan_stair_execution_inhibited",
    allow_tracking_command: bool = False,
    force_zero_velocity: bool = True,
    stop_confirmed: bool = True,
    global_replan_requested: bool = False,
    global_replan_in_flight: bool = False,
) -> dict[str, object]:
    """构造楼梯冻结只读校验所需的 supervisor 接收侧快照。"""

    return {
        "schema": "navigation_status_observed_diagnostics_v1",
        "topic": "/navigation/status",
        "status_error": None,
        "local_pct_goal_stamp_ns": 1,
        "local_active_path_stamp_ns": path_stamp_ns,
        "local_reference_path_identity_fault": None,
        "status": {
            "receipt_timestamp": receipt_timestamp,
            "state": state,
            "reason": reason,
            "allow_tracking_command": allow_tracking_command,
            "force_zero_velocity": force_zero_velocity,
            "stop_confirmed": stop_confirmed,
            "global_replan_requested": global_replan_requested,
            "global_replan_in_flight": global_replan_in_flight,
            "global_replan_request_id": 0,
            "pct_plan_id": 1,
            "consecutive_scan_failures": consecutive_scan_failures,
            "active_path_stamp_ns": path_stamp_ns,
            "identity_valid": identity_valid,
            "stale_inputs": list(stale_inputs),
        },
    }


def _state(
    step: int,
    *,
    timestamp: float,
    xyz: tuple[float, float, float],
    yaw: float = 0.0,
    write_sequence: int | None = None,
    motion_allowed: bool = False,
    stop_reasons: tuple[str, ...] = ("scan_stair_freeze",),
    requested_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    written_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float, float, float, float] = (0.0,) * 6,
) -> SimulationState:
    metadata = {}
    if write_sequence is not None:
        metadata["scan_cmd_vel_last_write_report"] = _policy_report(
            write_sequence,
            timestamp=timestamp,
            motion_allowed=motion_allowed,
            stop_reasons=stop_reasons,
            requested_command=requested_command,
            written_command=written_command,
        )
    return SimulationState(
        step_index=step,
        timestamp=timestamp,
        robot_root_pose=(
            *xyz,
            math.cos(yaw * 0.5),
            0.0,
            0.0,
            math.sin(yaw * 0.5),
        ),
        robot_root_velocity=velocity,
        metadata=metadata,
    )


def _activated_sensor_barrier_controller(
    *,
    path_stamp_ns: int,
    terminal: bool = False,
    speed_mps: float = 1.0,
) -> tuple[ScanStairFreezeController, tuple[float, float, float, float]]:
    """构造已激活、尚未通过生产传感器屏障的楼梯控制器。"""

    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
        *((() if terminal else ((0.8, 0.0, 0.30),))),
    )
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=speed_mps,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            require_supervisor_sensor_status=True,
            activation_timeout_s=0.20,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        path,
        path_stamp_ns=path_stamp_ns,
        terminal_goal_base_xyzyaw=(
            _terminal_goal(path) if terminal else None
        ),
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    return (
        controller,
        tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"]),
    )


def _advance_to_release(
    controller: ScanStairFreezeController,
) -> tuple[int, tuple[float, float, float]]:
    """把短楼梯控制器推进到刚解除全部锁的动作。"""

    target = (0.0, 0.0, 0.30)
    for step in range(20):
        action = controller.compute_action(
            _state(
                step,
                timestamp=step * 0.02,
                xyz=target,
                write_sequence=step,
            )
        )
        assert action is not None
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if action.source == "scan_stair_freeze_released":
            return step, target
    raise AssertionError("控制器未在预期步数内进入 release 动作")


def _controller_waiting_for_fresh_handoff() -> tuple[
    ScanStairFreezeController,
    int,
    float,
    int,
]:
    """构造一个已稳定解锁、但尚未认证新 SCAN 轨迹的控制器。"""

    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.20,
            min_component_z_delta_m=0.20,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            root_release_settle_time_s=0.0,
            post_release_stable_time_s=0.02,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    path_stamp_ns = 10_000_001
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.15),
            (0.4, 0.0, 0.30),
            (0.6, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((0, 2),),
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=0.02,
            header_stamp_ns=20_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=30_000_000,
            start_time_ns=35_000_000,
            traj_id=1,
        )
    )
    target = (0.0, 0.0, 0.30)
    for step in range(1, 20):
        timestamp = step * 0.02
        action = controller.compute_action(
            _state(
                step,
                timestamp=timestamp,
                xyz=target,
                write_sequence=step,
            )
        )
        assert action is not None
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if controller.status()["phase"] == "resume_wait_fresh_cmd":
            break
    status = controller.status()
    assert status["phase"] == "resume_wait_fresh_cmd"
    release_sequence = status["release_write_sequence"]
    release_timestamp = status["release_write_timestamp"]
    assert isinstance(release_sequence, int)
    assert isinstance(release_timestamp, float)
    return controller, release_sequence, release_timestamp, path_stamp_ns


def _controller_with_fresh_tracking_handoff() -> tuple[
    ScanStairFreezeController,
    int,
    float,
    float,
]:
    """构造已收到 release 后新 B-spline 且进入 TRACKING 的控制器。"""

    controller, release_sequence, release_timestamp, path_stamp_ns = (
        _controller_waiting_for_fresh_handoff()
    )
    fresh_receipt = release_timestamp + 0.02
    fresh_stamp_ns = int(round(fresh_receipt * 1.0e9))
    controller.observe_controller_status(
        _controller_status_report(
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
    )
    tracking_receipt = fresh_receipt + 0.02
    controller.observe_controller_status(
        _controller_status_report(
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
    )
    return controller, release_sequence, release_timestamp, tracking_receipt


def test_shared_yaml_loader_preserves_ground_height_and_hash() -> None:
    path = (
        PROJECT_ROOT
        / "ros2_ws/src/scan_navigation_tools/config/"
        "multifloor_stair_two_step_path.yaml"
    )

    reference = load_scan_reference_path(path)

    assert reference.topic == "/initial_path"
    assert reference.frame_id == "world"
    assert reference.use_sim_time is True
    assert len(reference.sha256) == 64
    assert len(reference.points_sha256) == 64
    assert reference.stair_segment_indices == ((2, 5),)
    assert len(reference.points_ground_xyz) == 6
    assert reference.points_ground_xyz[0][2] == pytest.approx(
        -0.12757488791649585
    )


def test_step_grade_detects_stair_but_rejects_continuous_ramp() -> None:
    config = ScanStairFreezeConfig(approach_distance_m=0.0)
    stair = load_scan_reference_path(
        PROJECT_ROOT
        / "ros2_ws/src/scan_navigation_tools/config/"
        "multifloor_stair_two_step_path.yaml"
    )
    ramp = load_scan_reference_path(
        PROJECT_ROOT
        / "ros2_ws/src/scan_navigation_tools/config/validation_ramp_path.yaml"
    )

    stair_components = extract_stair_components(
        stair.points_ground_xyz,
        config,
    )
    ramp_components = extract_stair_components(
        ramp.points_ground_xyz,
        config,
    )

    assert len(stair_components) == 1
    assert stair_components[0][0] == stair.points_ground_xyz[2]
    assert stair_components[0][-1] == stair.points_ground_xyz[-1]
    assert ramp_components == ()


def test_geometry_fallback_rejects_uniform_steep_continuous_ramp() -> None:
    ramp = tuple((0.2 * index, 0.0, 0.08 * index) for index in range(5))

    assert extract_stair_components(ramp, ScanStairFreezeConfig()) == ()


def test_same_direction_flights_keep_one_lock_across_long_landing() -> None:
    path = (
        (0.0, 0.0, 0.00),
        (0.1, 0.0, 0.10),
        (0.3, 0.0, 0.10),
        (0.4, 0.0, 0.22),
        (1.1, 0.0, 0.22),
        (1.2, 0.0, 0.32),
        (1.4, 0.0, 0.32),
        (1.5, 0.0, 0.44),
    )

    components = extract_stair_components(
        path,
        ScanStairFreezeConfig(
            approach_distance_m=0.0,
            exit_distance_m=0.0,
        ),
    )

    assert components == (path,)


@pytest.mark.parametrize(
    "bad_value",
    [math.nan, math.inf, -math.inf],
)
def test_reference_path_rejects_nonfinite_coordinate(bad_value: float) -> None:
    with pytest.raises(ValueError, match="有限数值"):
        extract_stair_components(
            ((0.0, 0.0, 0.0), (0.2, 0.0, bad_value)),
            ScanStairFreezeConfig(),
        )


def test_freeze_activates_without_jump_and_releases_in_two_stages() -> None:
    config = ScanStairFreezeConfig(
        speed_mps=1.0,
        activation_radius_m=0.15,
        min_component_z_delta_m=0.20,
        exit_distance_m=0.0,
        full_lock_settle_time_s=0.04,
        root_release_settle_time_s=0.04,
        post_release_stable_time_s=0.04,
        certified_progress_m=0.01,
        default_control_dt_s=0.02,
        max_control_dt_s=0.02,
    )
    controller = ScanStairFreezeController(config)
    path_stamp_ns = 100_000_001
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            # 保留一段楼梯后的平地，验证非终点组件仍按 pct_scene
            # 语义分阶段释放并交回正常局部控制。
            (0.8, 0.0, 0.30),
        ),
        path_source="unit-test",
        path_sha256="abc",
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=0.11,
            header_stamp_ns=110_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=120_000_000,
            start_time_ns=130_000_000,
            traj_id=1,
        )
    )

    state = _state(0, timestamp=0.0, xyz=(0.2, 0.0, 0.30))
    activated = controller.compute_action(state)
    assert activated is not None
    assert activated.source == "scan_stair_freeze_activated"
    assert activated.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        (0.2, 0.0, 0.30, 0.0)
    )
    assert activated.metadata["navigation_full_body_joint_lock"] is True
    assert activated.metadata["navigation_cmd_vel_inhibit"] is True
    assert "navigation_carry_object_follow" not in activated.metadata
    assert controller.compute_action(state) is activated

    last_target = activated.metadata["navigation_base_pose_lock_xyzyaw"]
    saw_root_only = False
    release_action = None
    for step in range(1, 80):
        state = _state(
            step,
            timestamp=step * 0.02,
            xyz=tuple(last_target[:3]),
            write_sequence=step,
        )
        action = controller.compute_action(state)
        assert action is not None
        if action.source == "scan_stair_freeze_released":
            release_action = action
            break
        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        # 每 tick 目标位移由 speed*dt 限制，不在末端按完成半径跳跃。
        assert math.dist(last_target[:3], target[:3]) <= 0.020001
        last_target = target
        if action.source == "scan_stair_freeze_root_release_settle":
            saw_root_only = True
            assert action.metadata["navigation_base_pose_lock"] is True
            assert "navigation_full_body_joint_lock" not in action.metadata

    assert saw_root_only is True
    assert release_action is not None
    assert "navigation_base_pose_lock" not in release_action.metadata
    assert "navigation_support_joint_lock" not in release_action.metadata
    assert release_action.metadata["navigation_cmd_vel_inhibit"] is True
    assert controller.status()["phase"] == "post_release_stabilizing"
    assert controller.finish_ready is False
    assert controller.certified_progress_seen is True

    # 全部锁已经释放，但唯一 owner 继续零速；只有实测速度、姿态和 z
    # 连续稳定满窗口后才允许等待一条解锁后的新鲜正常写。
    for offset in range(1, 4):
        stable_state = _state(
            step + offset,
            timestamp=(step + offset) * 0.02,
            xyz=tuple(last_target[:3]),
            write_sequence=step + offset,
        )
        stable_action = controller.compute_action(stable_state)
        assert stable_action is not None
        assert stable_action.source == "scan_stair_freeze_post_release_stabilizing"
        assert "navigation_base_pose_lock" not in stable_action.metadata
        assert stable_action.metadata["navigation_cmd_vel_inhibit"] is True
        if controller.status()["phase"] == "resume_wait_fresh_cmd":
            break
    assert controller.status()["phase"] == "resume_wait_fresh_cmd"

    release_marker = controller.status()["release_write_sequence"]
    release_marker_timestamp = controller.status()["release_write_timestamp"]
    assert isinstance(release_marker, int)
    assert isinstance(release_marker_timestamp, float)
    controller.observe_policy_write(
        _policy_report(
            release_marker + 1,
            timestamp=release_marker_timestamp + 0.02,
            motion_allowed=False,
            stop_reasons=("scan_stair_freeze_release",),
            navigation_cmd_vel_inhibited=True,
        )
    )
    assert controller.finish_ready is False
    fresh_bspline_stamp_ns = int(
        round((release_marker_timestamp + 0.02) * 1.0e9)
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=2,
            receipt_timestamp=release_marker_timestamp + 0.02,
            header_stamp_ns=fresh_bspline_stamp_ns,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=fresh_bspline_stamp_ns,
            start_time_ns=fresh_bspline_stamp_ns + 1,
            traj_id=2,
            event=1,
            state=2,
        )
    )
    assert controller.finish_ready is False
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=3,
            acceptance_sequence=2,
            receipt_timestamp=release_marker_timestamp + 0.03,
            header_stamp_ns=fresh_bspline_stamp_ns + 2,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=fresh_bspline_stamp_ns,
            start_time_ns=fresh_bspline_stamp_ns + 1,
            traj_id=2,
            event=4,
            state=10,
        )
    )
    controller.observe_policy_write(
        _policy_report(
            release_marker + 4,
            timestamp=release_marker_timestamp + 0.04,
            motion_allowed=True,
            stop_reasons=(),
            # 平地交接时新鲜正常写入也可能是合法零速，不能强制非零。
            requested_command=(0.0, 0.0, 0.0),
            written_command=(0.0, 0.0, 0.0),
            navigation_cmd_vel_inhibited=False,
        )
    )
    assert controller.finish_ready is True
    assert controller.status()["completed_component_count"] == 1


def test_sensor_timeout_latches_last_root_target_instead_of_blind_progress() -> None:
    """冻结期间传感器失鲜必须保持最后目标并失败关闭。"""

    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    progressed = controller.compute_action(
        _state(
            1,
            timestamp=0.02,
            xyz=tuple(
                activated.metadata["navigation_base_pose_lock_xyzyaw"][:3]
            ),
        )
    )
    assert progressed is not None
    last_target = tuple(
        progressed.metadata["navigation_base_pose_lock_xyzyaw"]
    )
    progress_before_fault = controller.status()["progress_m"]

    controller.observe_policy_write(
        _policy_report(
            2,
            timestamp=0.60,
            stop_reasons=("scan_stair_freeze", "point_cloud_timeout"),
        )
    )
    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(2, timestamp=0.60, xyz=tuple(last_target[:3]))
        )

    status = controller.status()
    assert status["phase"] == "failed"
    assert status["reason"] == "stair_sensor_freshness_fault"
    assert status["progress_m"] == pytest.approx(progress_before_fault)
    assert status["sensor_safety_fault_reasons"] == ["point_cloud_timeout"]
    assert status["sensor_safety_fault_write_sequence"] == 2
    assert status["emergency_hold_latched"] is True
    assert status["emergency_hold_origin_phase"] == "active"
    assert status["emergency_hold_full_body_lock"] is True

    held = controller.emergency_hold_action(
        _state(3, timestamp=0.62, xyz=tuple(last_target[:3])),
        reason="stair_sensor_freshness_fault",
    )
    assert held is not None
    assert held.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        last_target
    )
    assert held.metadata["navigation_full_body_joint_lock"] is True


def test_new_path_sensor_barrier_waits_for_fresh_cloud_before_root_progress() -> None:
    """新 Path 的首帧点云竞态只能冻结等待，不能立即锁存永久故障。"""

    path_stamp_ns = 89_000_001
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            activation_timeout_s=0.20,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    initial_target = tuple(
        activated.metadata["navigation_base_pose_lock_xyzyaw"]
    )
    assert (
        activated.metadata[
            "navigation_scan_stair_sensor_acquisition_pending"
        ]
        is True
    )

    stale_report = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze", "missing_point_cloud"),
    )
    stale_report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("point_cloud", "bspline"),
        )
    )
    controller.observe_policy_write(stale_report)
    waiting = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(initial_target[:3]))
    )
    assert waiting is not None
    assert waiting.source == "scan_stair_sensor_acquisition_wait"
    assert waiting.base_velocity == pytest.approx((0.0, 0.0, 0.0))
    assert waiting.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        initial_target
    )
    status = controller.status()
    assert status["progress_m"] == pytest.approx(0.0)
    assert status["sensor_safety_fault_reasons"] == []
    assert status["sensor_acquisition_pending"] is True
    assert status["sensor_acquisition_pending_reasons"] == [
        "missing_point_cloud",
        "supervisor_point_cloud_stale"
    ]

    fresh_report = _policy_report(
        2,
        timestamp=1.04,
        stop_reasons=("scan_stair_freeze",),
    )
    fresh_status_report = _navigation_status_observed_report(
        receipt_timestamp=1.04,
        path_stamp_ns=path_stamp_ns,
        stale_inputs=("bspline",),
    )
    fresh_report["navigation_status_observed_report"] = fresh_status_report
    controller.observe_policy_write(fresh_report)
    advanced = controller.compute_action(
        _state(2, timestamp=1.04, xyz=tuple(initial_target[:3]))
    )
    assert advanced is not None
    assert advanced.source == "scan_stair_freeze_active"
    status = controller.status()
    barrier = status["sensor_acquisition_barrier"]
    assert status["progress_m"] == pytest.approx(0.02)
    assert barrier["required"] is True
    assert barrier["passed"] is True
    assert barrier["pending"] is False
    assert barrier["path_stamp_ns"] == path_stamp_ns
    assert barrier["write_sequence"] == 2
    assert barrier["write_timestamp"] == pytest.approx(1.04)
    assert barrier["progress_m_at_pass"] == pytest.approx(0.0)
    assert barrier["local_sensors_fresh"] is True
    assert barrier["supervisor_sensors_fresh"] is True
    assert barrier["navigation_status_observed_report"] == fresh_status_report
    assert barrier["last_navigation_status_observed_report"] == (
        fresh_status_report
    )
    assert barrier["policy_write_report"]["write_sequence"] == 2


def test_sensor_barrier_allows_nonzero_scan_failure_history() -> None:
    """楼梯 ACK 已生效时，入口前的非零 SCAN 历史计数不能阻塞上楼。"""

    path_stamp_ns = 89_000_002
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acknowledged = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    acknowledged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
            consecutive_scan_failures=1,
        )
    )

    controller.observe_policy_write(acknowledged)
    advanced = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )

    assert advanced is not None
    assert advanced.source == "scan_stair_freeze_active"
    status = controller.status()
    assert status["progress_m"] == pytest.approx(0.02)
    assert status["sensor_acquisition_complete"] is True
    assert status["sensor_acquisition_pending_reasons"] == []


@pytest.mark.parametrize(
    "invalid_case",
    [
        "sequence_zero",
        "wrong_owner",
        "motion_allowed",
        "nonzero_write",
        "environment_closed",
        "emergency_reason",
        "terminal_reason",
        "release_reason",
    ],
)
def test_sensor_barrier_rejects_forged_policy_freeze_writes(
    invalid_case: str,
) -> None:
    """代际屏障只能由本 owner 的 active 阶段精确写零报告放行。"""

    path_stamp_ns = 89_000_010
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    sequence = 0 if invalid_case == "sequence_zero" else 1
    kwargs: dict[str, object] = {}
    if invalid_case == "wrong_owner":
        kwargs["owner_id"] = "forged_owner"
    elif invalid_case == "motion_allowed":
        kwargs["motion_allowed"] = True
    elif invalid_case == "nonzero_write":
        kwargs["written_command"] = (0.01, 0.0, 0.0)
    elif invalid_case == "environment_closed":
        kwargs["stop_reasons"] = (
            "scan_stair_freeze",
            "environment_closed",
        )
    elif invalid_case.endswith("_reason"):
        inhibit_reason = {
            "emergency_reason": "scan_stair_emergency_hold",
            "terminal_reason": "scan_stair_terminal_hold",
            "release_reason": "scan_stair_freeze_release",
        }[invalid_case]
        kwargs["stop_reasons"] = (inhibit_reason,)
        kwargs["navigation_cmd_vel_inhibit_reason"] = inhibit_reason

    report = _policy_report(
        sequence,
        timestamp=1.02,
        **kwargs,  # type: ignore[arg-type]
    )
    report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(report)
    waiting = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )

    assert waiting is not None
    assert waiting.source == "scan_stair_sensor_acquisition_wait"
    status = controller.status()
    assert status["sensor_acquisition_complete"] is False
    assert status["progress_m"] == pytest.approx(0.0)
    assert status["sensor_acquisition_pending_reasons"] in (
        ["awaiting_new_active_generation_policy_write"],
        ["awaiting_active_generation_freeze_write"],
    )


@pytest.mark.parametrize(
    "invalid_case",
    [
        "wrong_owner",
        "motion_allowed",
        "nonzero_write",
        "extra_stop_reason",
        "wrong_inhibit_reason",
        "sequence_regressed",
    ],
)
def test_root_lock_latches_invalid_policy_write_after_barrier(
    invalid_case: str,
) -> None:
    """屏障后任何新的非法冻结写入都必须在 root 前进前锁存故障。"""

    path_stamp_ns = 89_000_011
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acknowledged = _policy_report(1, timestamp=1.02)
    acknowledged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acknowledged)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    progress_before_fault = controller.status()["progress_m"]

    sequence = 0 if invalid_case == "sequence_regressed" else 2
    kwargs: dict[str, object] = {}
    if invalid_case == "wrong_owner":
        kwargs["owner_id"] = "forged_owner"
    elif invalid_case == "motion_allowed":
        kwargs["motion_allowed"] = True
    elif invalid_case == "nonzero_write":
        kwargs["written_command"] = (0.01, 0.0, 0.0)
    elif invalid_case == "extra_stop_reason":
        kwargs["stop_reasons"] = (
            "scan_stair_freeze",
            "environment_closed",
        )
    elif invalid_case == "wrong_inhibit_reason":
        kwargs["stop_reasons"] = ("scan_stair_terminal_hold",)
        kwargs["navigation_cmd_vel_inhibit_reason"] = (
            "scan_stair_terminal_hold"
        )
    forged = _policy_report(
        sequence,
        timestamp=1.04,
        **kwargs,  # type: ignore[arg-type]
    )
    forged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(forged)

    with pytest.raises(RuntimeError, match="policy 写零协议非法"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.04,
                xyz=tuple(
                    progressed.metadata["navigation_base_pose_lock_xyzyaw"][:3]
                ),
            )
        )
    status = controller.status()
    assert status["phase"] == "failed"
    assert status["reason"] == "stair_policy_freeze_write_fault"
    assert status["progress_m"] == pytest.approx(progress_before_fault)
    assert status["policy_freeze_write_fault_reasons"]
    assert status["emergency_hold_latched"] is True


def test_sensor_barrier_rejects_protocol_poisoned_emergency_ack() -> None:
    """传感器虽新鲜，协议异常急停也不能解锁 direct-root 进度。"""

    path_stamp_ns = 89_000_003
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            activation_timeout_s=0.20,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    target = tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"])

    poisoned = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    poisoned_status = _navigation_status_observed_report(
        receipt_timestamp=1.02,
        path_stamp_ns=path_stamp_ns,
        stale_inputs=("bspline",),
    )
    poisoned_status["status"]["reason"] = (  # type: ignore[index]
        "scan_status_invariant:Path 绑定的 SCAN 事件缺少 reference stamp"
    )
    poisoned["navigation_status_observed_report"] = poisoned_status
    controller.observe_policy_write(poisoned)

    waiting = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert waiting is not None
    assert waiting.source == "scan_stair_sensor_acquisition_wait"
    status = controller.status()
    assert status["progress_m"] == pytest.approx(0.0)
    assert status["sensor_acquisition_complete"] is False
    assert status["sensor_acquisition_pending_reasons"] == [
        "supervisor_stair_freeze_not_acknowledged"
    ]

    acknowledged = _policy_report(
        2,
        timestamp=1.04,
        stop_reasons=("scan_stair_freeze",),
    )
    acknowledged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acknowledged)
    advanced = controller.compute_action(
        _state(2, timestamp=1.04, xyz=tuple(target[:3]))
    )
    assert advanced is not None
    assert controller.status()["progress_m"] == pytest.approx(0.02)


def test_protocol_poison_after_sensor_barrier_fails_closed() -> None:
    """屏障通过后出现非预期 emergency 原因必须立即锁停。"""

    path_stamp_ns = 89_000_004
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    target = tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"])
    acknowledged = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    acknowledged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acknowledged)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None

    poisoned = _policy_report(
        2,
        timestamp=1.04,
        stop_reasons=("scan_stair_freeze",),
    )
    poisoned_status = _navigation_status_observed_report(
        receipt_timestamp=1.04,
        path_stamp_ns=path_stamp_ns,
        stale_inputs=("bspline",),
    )
    poisoned_status["status"]["reason"] = "scan_emergency_stop"  # type: ignore[index]
    poisoned["navigation_status_observed_report"] = poisoned_status
    controller.observe_policy_write(poisoned)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.04,
                xyz=tuple(
                    progressed.metadata[
                        "navigation_base_pose_lock_xyzyaw"
                    ][:3]
                ),
            )
        )
    status = controller.status()
    assert status["sensor_safety_fault_reasons"] == [
        "supervisor_stair_freeze_not_acknowledged"
    ]
    assert status["emergency_hold_latched"] is True


@pytest.mark.parametrize(
    (
        "controller_state",
        "navigation_state",
        "navigation_reason",
        "allow_tracking",
        "force_zero",
        "stop_confirmed",
    ),
    [
        (9, 3, "tracking_inputs_ready", True, False, False),
        (10, 3, "tracking_inputs_ready", True, False, False),
        (12, 6, "goal_reached", False, True, True),
    ],
)
def test_final_controller_status_allows_terminal_supervisor_transition(
    controller_state: int,
    navigation_state: int,
    navigation_reason: str,
    allow_tracking: bool,
    force_zero: bool,
    stop_confirmed: bool,
) -> None:
    """v5 的 final SCAN 状态切换不能被误判为冻结 ACK 丢失。"""

    path_stamp_ns = 89_000_012
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    progress_before_transition = controller.status()["progress_m"]

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=controller_state,
            is_final=True,
        )
    )
    transition = _policy_report(2, timestamp=1.04)
    transition["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            state=navigation_state,
            reason=navigation_reason,
            allow_tracking_command=allow_tracking,
            force_zero_velocity=force_zero,
            stop_confirmed=stop_confirmed,
        )
    )
    controller.observe_policy_write(transition)
    action = controller.compute_action(
        _state(
            2,
            timestamp=1.04,
            xyz=tuple(
                progressed.metadata["navigation_base_pose_lock_xyzyaw"][:3]
            ),
        )
    )

    assert action is not None
    status = controller.status()
    assert status["phase"] == "active"
    assert status["progress_m"] > progress_before_transition
    assert status["sensor_safety_fault_reasons"] == []
    assert status["last_controller_is_final"] is True


def test_newer_goal_status_keeps_same_trajectory_causal_anchor() -> None:
    """v7 中晚到一拍的同轨迹 GOAL_REACHED 不得抹掉已有 TRACKING 锚点。"""

    path_stamp_ns = 89_000_016
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=1.08,
            header_stamp_ns=1_080_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=12,
            is_final=True,
        )
    )
    interleaved = _policy_report(2, timestamp=1.06)
    interleaved["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.05,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(interleaved)
    action = controller.compute_action(
        _state(
            2,
            timestamp=1.06,
            xyz=tuple(
                progressed.metadata["navigation_base_pose_lock_xyzyaw"][:3]
            ),
        )
    )

    assert action is not None
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert controller.status()["terminal_supervisor_goal_acknowledged"] is False

    acknowledged = _policy_report(3, timestamp=1.10)
    acknowledged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.09,
            path_stamp_ns=path_stamp_ns,
            state=6,
            reason="goal_reached",
            allow_tracking_command=False,
            force_zero_velocity=True,
            stop_confirmed=True,
        )
    )
    controller.observe_policy_write(acknowledged)
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert controller.status()["terminal_supervisor_goal_acknowledged"] is True


def test_supervisor_goal_waits_for_cross_topic_controller_receipt_order() -> None:
    """nav GOAL 先被轮询时应等下一 heartbeat，不能当拍锁存故障。"""

    path_stamp_ns = 89_000_019
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    early_goal = _policy_report(2, timestamp=1.06)
    early_goal["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.05,
            path_stamp_ns=path_stamp_ns,
            state=6,
            reason="goal_reached",
            allow_tracking_command=False,
            force_zero_velocity=True,
            stop_confirmed=True,
        )
    )
    controller.observe_policy_write(early_goal)
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert (
        controller.status()[
            "terminal_supervisor_goal_pending_started_timestamp"
        ]
        == pytest.approx(1.06)
    )

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=1.08,
            header_stamp_ns=1_080_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=12,
            is_final=True,
        )
    )
    inverted = _policy_report(3, timestamp=1.10)
    inverted["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.07,
            path_stamp_ns=path_stamp_ns,
            state=6,
            reason="goal_reached",
            allow_tracking_command=False,
            force_zero_velocity=True,
            stop_confirmed=True,
        )
    )
    controller.observe_policy_write(inverted)
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert controller.status()["terminal_supervisor_goal_acknowledged"] is False

    heartbeat = _policy_report(4, timestamp=1.12)
    heartbeat["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.11,
            path_stamp_ns=path_stamp_ns,
            state=6,
            reason="goal_reached",
            allow_tracking_command=False,
            force_zero_velocity=True,
            stop_confirmed=True,
        )
    )
    controller.observe_policy_write(heartbeat)
    status = controller.status()
    assert status["sensor_safety_fault_reasons"] == []
    assert status["terminal_supervisor_goal_acknowledged"] is True
    assert status["terminal_supervisor_goal_pending_started_timestamp"] is None


def test_supervisor_tracking_waits_for_cross_topic_controller_receipt_order() -> None:
    """final 接受先于 controller TRACKING 时应有限等待下一拍状态。"""

    path_stamp_ns = 89_000_021
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.05,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=1,
            state=0,
            is_final=True,
        )
    )
    early_tracking = _policy_report(2, timestamp=1.06)
    early_tracking["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.05,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(early_tracking)
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert (
        controller.status()[
            "terminal_supervisor_goal_pending_started_timestamp"
        ]
        == pytest.approx(1.06)
    )

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=1.08,
            header_stamp_ns=1_080_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    heartbeat = _policy_report(3, timestamp=1.10)
    heartbeat["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.09,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(heartbeat)
    status = controller.status()
    assert status["sensor_safety_fault_reasons"] == []
    assert status["invalid_controller_status_count"] == 0
    assert status["controller_status_observation_count"] == 2
    assert status["terminal_supervisor_goal_pending_started_timestamp"] is None


def test_supervisor_tracking_controller_evidence_wait_is_bounded() -> None:
    """final 接受后 controller 长时间不进入 TRACKING 仍必须停车。"""

    path_stamp_ns = 89_000_022
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.05,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=1,
            state=0,
            is_final=True,
        )
    )

    for sequence, timestamp in ((2, 1.06), (3, 1.32)):
        report = _policy_report(sequence, timestamp=timestamp)
        report["navigation_status_observed_report"] = (
            _navigation_status_observed_report(
                receipt_timestamp=timestamp - 0.01,
                path_stamp_ns=path_stamp_ns,
                state=3,
                reason="tracking_inputs_ready",
                allow_tracking_command=True,
                force_zero_velocity=False,
                stop_confirmed=False,
            )
        )
        controller.observe_policy_write(report)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.32,
                xyz=tuple(
                    progressed.metadata[
                        "navigation_base_pose_lock_xyzyaw"
                    ][:3]
                ),
            )
        )
    assert controller.status()["sensor_safety_fault_reasons"] == [
        "supervisor_terminal_controller_evidence_timeout"
    ]


def test_controller_status_command_aggregate_is_strictly_validated() -> None:
    """运行时新增命令聚合字段必须完整且内部一致。"""

    controller = ScanStairFreezeController()
    report = _controller_status_report(
        status_sequence=1,
        acceptance_sequence=0,
        receipt_timestamp=1.0,
        header_stamp_ns=1_000_000_000,
        reference_path_stamp_ns=0,
        bspline_header_stamp_ns=0,
        start_time_ns=0,
        traj_id=0,
        event=0,
        state=0,
        accepted=False,
        trajectory_valid=False,
    )
    report["command_aggregate"]["max_abs_vx"] = 0.1  # type: ignore[index]
    controller.observe_controller_status(report)
    status = controller.status()
    assert status["controller_status_observation_count"] == 0
    assert status["invalid_controller_status_count"] == 1


def test_supervisor_goal_controller_evidence_wait_is_bounded() -> None:
    """supervisor GOAL 缺少本地 typed controller 证据时只能有限等待。"""

    path_stamp_ns = 89_000_020
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None

    for sequence, timestamp, receipt_timestamp in (
        (2, 1.06, 1.05),
        (3, 1.32, 1.31),
    ):
        report = _policy_report(sequence, timestamp=timestamp)
        report["navigation_status_observed_report"] = (
            _navigation_status_observed_report(
                receipt_timestamp=receipt_timestamp,
                path_stamp_ns=path_stamp_ns,
                state=6,
                reason="goal_reached",
                allow_tracking_command=False,
                force_zero_velocity=True,
                stop_confirmed=True,
            )
        )
        controller.observe_policy_write(report)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.32,
                xyz=tuple(
                    progressed.metadata[
                        "navigation_base_pose_lock_xyzyaw"
                    ][:3]
                ),
            )
        )
    assert controller.status()["sensor_safety_fault_reasons"] == [
        "supervisor_terminal_controller_evidence_timeout"
    ]


def test_controller_goal_requires_bounded_supervisor_goal_transition() -> None:
    """controller 已到达后，supervisor 不能无限停留在 TRACKING。"""

    path_stamp_ns = 89_000_018
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=1.08,
            header_stamp_ns=1_080_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=12,
            is_final=True,
        )
    )
    stale_transition = _policy_report(2, timestamp=1.34)
    stale_transition["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.34,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(stale_transition)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.34,
                xyz=tuple(
                    progressed.metadata[
                        "navigation_base_pose_lock_xyzyaw"
                    ][:3]
                ),
            )
        )
    assert controller.status()["sensor_safety_fault_reasons"] == [
        "supervisor_terminal_transition_timeout"
    ]


@pytest.mark.parametrize("newer_case", ["invalidated", "emergency", "replacement"])
def test_old_terminal_anchor_cannot_mask_newer_unsafe_status(
    newer_case: str,
) -> None:
    """旧 final 锚点不得掩盖失效、急停或轨迹 identity 替换。"""

    path_stamp_ns = 89_000_017
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=1.08,
            header_stamp_ns=1_080_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=(
                1_070_000_000
                if newer_case == "replacement"
                else 1_030_000_000
            ),
            start_time_ns=(
                1_070_000_000
                if newer_case == "replacement"
                else 1_030_000_000
            ),
            traj_id=2 if newer_case == "replacement" else 1,
            event=3 if newer_case == "invalidated" else 4,
            state=12 if newer_case != "emergency" else 10,
            trajectory_valid=newer_case != "invalidated",
            is_final=True,
            emergency_stop=newer_case == "emergency",
        )
    )
    interleaved = _policy_report(2, timestamp=1.06)
    interleaved["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.05,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(interleaved)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.06,
                xyz=tuple(
                    progressed.metadata[
                        "navigation_base_pose_lock_xyzyaw"
                    ][:3]
                ),
            )
        )
    assert controller.status()["sensor_safety_fault_reasons"] == [
        "supervisor_stair_freeze_not_acknowledged"
    ]


def test_acquisition_rejects_tracking_even_with_final_controller_status() -> None:
    """final 轨迹例外只能在代际屏障已经通过后使用。"""

    path_stamp_ns = 89_000_013
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.01,
            header_stamp_ns=1_010_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_005_000_000,
            start_time_ns=1_005_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    forged = _policy_report(1, timestamp=1.02)
    forged["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(forged)
    action = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )

    assert action is not None
    assert action.source == "scan_stair_sensor_acquisition_wait"
    status = controller.status()
    assert status["sensor_acquisition_complete"] is False
    assert status["sensor_acquisition_pending_reasons"] == [
        "supervisor_stair_freeze_not_acknowledged"
    ]


@pytest.mark.parametrize(
    "invalid_case",
    ["not_final", "wrong_path", "emergency", "replan"],
)
def test_terminal_supervisor_transition_rejects_invalid_typed_provenance(
    invalid_case: str,
) -> None:
    """final 例外不得掩盖伪造 identity、急停或全局重规划。"""

    path_stamp_ns = 89_000_014
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    progress_before_fault = controller.status()["progress_m"]
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.04,
            header_stamp_ns=1_040_000_000,
            reference_path_stamp_ns=(
                path_stamp_ns + 1 if invalid_case == "wrong_path" else path_stamp_ns
            ),
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=invalid_case != "not_final",
            emergency_stop=invalid_case == "emergency",
        )
    )
    transition = _policy_report(2, timestamp=1.04)
    transition["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
            global_replan_requested=invalid_case == "replan",
        )
    )
    controller.observe_policy_write(transition)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(
                2,
                timestamp=1.04,
                xyz=tuple(
                    progressed.metadata["navigation_base_pose_lock_xyzyaw"][:3]
                ),
            )
        )
    status = controller.status()
    assert status["reason"] == "stair_sensor_freshness_fault"
    assert status["progress_m"] == pytest.approx(progress_before_fault)
    assert status["sensor_safety_fault_reasons"] == [
        "supervisor_stair_freeze_not_acknowledged"
    ]


def test_terminal_supervisor_transition_allows_one_tick_controller_lead() -> None:
    """ROS 跨 topic 轮询可让同轨迹 controller 快照领先一个控制周期。"""

    path_stamp_ns = 89_000_023
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns
    )
    acquired = _policy_report(1, timestamp=1.02)
    acquired["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(acquired)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert progressed is not None
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=1.06,
            header_stamp_ns=1_060_000_000,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=1_030_000_000,
            start_time_ns=1_030_000_000,
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    inverted = _policy_report(2, timestamp=1.04)
    inverted["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(inverted)
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert (
        controller.status()[
            "terminal_supervisor_goal_pending_started_timestamp"
        ]
        == pytest.approx(1.04)
    )

    heartbeat = _policy_report(3, timestamp=1.08)
    heartbeat["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.08,
            path_stamp_ns=path_stamp_ns,
            state=3,
            reason="tracking_inputs_ready",
            allow_tracking_command=True,
            force_zero_velocity=False,
            stop_confirmed=False,
        )
    )
    controller.observe_policy_write(heartbeat)
    status = controller.status()
    assert status["sensor_safety_fault_reasons"] == []
    assert status["terminal_supervisor_goal_pending_started_timestamp"] is None


def test_terminal_phase_transition_consumes_each_policy_write_once() -> None:
    """active→terminal_hold 时重复观察上一动作报告不能触发误急停。"""

    path_stamp_ns = 89_000_015
    controller, target = _activated_sensor_barrier_controller(
        path_stamp_ns=path_stamp_ns,
        terminal=True,
        speed_mps=10.0,
    )
    for sequence in range(1, 8):
        timestamp = 1.0 + sequence * 0.02
        expected_reason = (
            "scan_stair_terminal_hold"
            if controller.status()["phase"] == "terminal_hold"
            else "scan_stair_freeze"
        )
        report = _policy_report(
            sequence,
            timestamp=timestamp,
            stop_reasons=(expected_reason,),
            navigation_cmd_vel_inhibit_reason=expected_reason,
        )
        report["navigation_status_observed_report"] = (
            _navigation_status_observed_report(
                receipt_timestamp=timestamp,
                path_stamp_ns=path_stamp_ns,
                stale_inputs=("bspline",),
            )
        )
        controller.observe_policy_write(report)
        action = controller.compute_action(
            _state(
                sequence,
                timestamp=timestamp,
                xyz=tuple(target[:3]),
            )
        )
        assert action is not None
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"])
        # 模拟 executor 在同一 observation 中由 is_done 再观察一次 metadata。
        controller.observe_policy_write(report)
        if (
            expected_reason == "scan_stair_terminal_hold"
            and controller.status()["phase"] == "terminal_hold"
        ):
            break

    assert controller.status()["phase"] == "terminal_hold"
    assert controller.status()["policy_freeze_write_fault_reasons"] == []
    assert controller.finish_ready is False

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=1,
            acceptance_sequence=1,
            receipt_timestamp=timestamp + 0.01,
            header_stamp_ns=int(round((timestamp + 0.01) * 1.0e9)),
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=int(round(timestamp * 1.0e9)),
            start_time_ns=int(round(timestamp * 1.0e9)),
            traj_id=1,
            event=4,
            state=10,
            is_final=True,
        )
    )
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=1,
            receipt_timestamp=timestamp + 0.02,
            header_stamp_ns=int(round((timestamp + 0.02) * 1.0e9)),
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=int(round(timestamp * 1.0e9)),
            start_time_ns=int(round(timestamp * 1.0e9)),
            traj_id=1,
            event=4,
            state=12,
            is_final=True,
        )
    )
    goal_ack = _policy_report(
        sequence + 1,
        timestamp=timestamp + 0.04,
        stop_reasons=("scan_stair_terminal_hold",),
        navigation_cmd_vel_inhibit_reason="scan_stair_terminal_hold",
    )
    goal_ack["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=timestamp + 0.03,
            path_stamp_ns=path_stamp_ns,
            state=6,
            reason="goal_reached",
            allow_tracking_command=False,
            force_zero_velocity=True,
            stop_confirmed=True,
        )
    )
    controller.observe_policy_write(goal_ack)

    assert controller.status()["terminal_supervisor_goal_acknowledged"] is True
    assert controller.finish_ready is True


def test_new_path_sensor_barrier_times_out_fail_closed() -> None:
    """首帧传感器始终未就绪时，有限等待后必须升级为可重规划故障。"""

    path_stamp_ns = 89_000_002
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            activation_timeout_s=0.05,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    target = tuple(activated.metadata["navigation_base_pose_lock_xyzyaw"])
    stale_report = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    stale_report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("point_cloud",),
        )
    )
    controller.observe_policy_write(stale_report)
    waiting = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(target[:3]))
    )
    assert waiting is not None
    assert controller.status()["progress_m"] == pytest.approx(0.0)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(2, timestamp=1.06, xyz=tuple(target[:3]))
        )
    status = controller.status()
    assert status["phase"] == "failed"
    assert status["reason"] == "stair_sensor_freshness_fault"
    assert status["sensor_safety_fault_reasons"] == [
        "supervisor_point_cloud_stale"
    ]
    assert status["sensor_safety_fault_write_sequence"] == 1
    assert status["sensor_safety_fault_timestamp"] == pytest.approx(1.06)
    assert status["sensor_acquisition_barrier"]["passed"] is False
    assert status["sensor_acquisition_barrier"][
        "last_navigation_status_observed_report"
    ]["status"]["stale_inputs"] == ["point_cloud"]
    assert status["progress_m"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("stale_input", "expected_fault"),
    [
        ("odometry", "supervisor_odometry_stale"),
        ("point_cloud", "supervisor_point_cloud_stale"),
    ],
)
def test_supervisor_received_sensor_stale_fails_closed_during_root_lock(
    stale_input: str,
    expected_fault: str,
) -> None:
    """屏障通过后的 DDS 传感器失鲜必须立即停止 direct-root。"""

    path_stamp_ns = 90_000_001
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            supervisor_sensor_status_timeout_s=0.25,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    last_target = tuple(
        activated.metadata["navigation_base_pose_lock_xyzyaw"]
    )
    fresh_report = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    fresh_report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
        )
    )
    controller.observe_policy_write(fresh_report)
    progressed = controller.compute_action(
        _state(1, timestamp=1.02, xyz=tuple(last_target[:3]))
    )
    assert progressed is not None
    last_target = tuple(
        progressed.metadata["navigation_base_pose_lock_xyzyaw"]
    )

    stale_report = _policy_report(
        2,
        timestamp=1.04,
        stop_reasons=("scan_stair_freeze",),
    )
    stale_report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.04,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=(stale_input,),
        )
    )
    controller.observe_policy_write(stale_report)

    with pytest.raises(RuntimeError, match="传感器失鲜"):
        controller.compute_action(
            _state(2, timestamp=1.04, xyz=tuple(last_target[:3]))
        )

    status = controller.status()
    assert status["sensor_safety_fault_reasons"] == [expected_fault]
    assert status["progress_m"] == pytest.approx(0.02)
    assert status["emergency_hold_latched"] is True


def test_supervisor_bspline_stale_is_expected_while_root_lock_is_active() -> None:
    """冻结会主动暂停 SCAN 轨迹时钟，单独 Bspline stale 不能误报传感器故障。"""

    path_stamp_ns = 91_000_001
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
            require_supervisor_sensor_status=True,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.4, 0.0, 0.15),
            (0.6, 0.0, 0.30),
            (0.8, 0.0, 0.30),
        ),
        path_stamp_ns=path_stamp_ns,
        stair_segment_indices=((1, 3),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    report = _policy_report(
        1,
        timestamp=1.02,
        stop_reasons=("scan_stair_freeze",),
    )
    report["navigation_status_observed_report"] = (
        _navigation_status_observed_report(
            receipt_timestamp=1.02,
            path_stamp_ns=path_stamp_ns,
            stale_inputs=("bspline",),
        )
    )
    controller.observe_policy_write(report)

    advanced = controller.compute_action(
        _state(
            1,
            timestamp=1.02,
            xyz=tuple(
                activated.metadata["navigation_base_pose_lock_xyzyaw"][:3]
            ),
        )
    )

    assert advanced is not None
    assert controller.status()["phase"] == "active"
    assert controller.status()["sensor_safety_fault_reasons"] == []
    assert controller.status()["progress_m"] == pytest.approx(0.02)


def test_resume_requires_current_path_acceptance_and_tracking_state() -> None:
    controller, _, release_timestamp, path_stamp_ns = (
        _controller_waiting_for_fresh_handoff()
    )
    fresh_receipt = release_timestamp + 0.02
    fresh_stamp_ns = int(round(fresh_receipt * 1.0e9))

    # 晚到的 transient 快照即使 acceptance_sequence 更新，只要 B-spline
    # 本身早于 release，就不能成为新鲜交接证据。
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=2,
            acceptance_sequence=2,
            receipt_timestamp=fresh_receipt,
            header_stamp_ns=fresh_stamp_ns,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=int(round(release_timestamp * 1.0e9)),
            start_time_ns=int(round(release_timestamp * 1.0e9)),
            traj_id=2,
        )
    )
    assert controller.status()["fresh_controller_pending_identity"] is None

    # 另一代 Path 上的新轨迹同样不能接管当前楼梯后的平地。
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=3,
            acceptance_sequence=3,
            receipt_timestamp=fresh_receipt + 0.01,
            header_stamp_ns=fresh_stamp_ns + 1,
            reference_path_stamp_ns=path_stamp_ns + 1,
            bspline_header_stamp_ns=fresh_stamp_ns + 1,
            start_time_ns=fresh_stamp_ns + 2,
            traj_id=3,
        )
    )
    assert controller.status()["fresh_controller_pending_identity"] is None

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=4,
            acceptance_sequence=4,
            receipt_timestamp=fresh_receipt + 0.02,
            header_stamp_ns=fresh_stamp_ns + 3,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=fresh_stamp_ns + 3,
            start_time_ns=fresh_stamp_ns + 4,
            traj_id=4,
            event=1,
            state=2,
        )
    )
    pending = controller.status()["fresh_controller_pending_identity"]
    assert pending is not None
    assert controller.status()["fresh_controller_execution_identity"] is None

    # DUPLICATE 只能确认幂等重发，不能把 WAITING 状态升级为执行证明。
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=5,
            acceptance_sequence=4,
            receipt_timestamp=fresh_receipt + 0.03,
            header_stamp_ns=fresh_stamp_ns + 5,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=fresh_stamp_ns + 3,
            start_time_ns=fresh_stamp_ns + 4,
            traj_id=4,
            event=5,
            state=10,
        )
    )
    assert controller.status()["fresh_controller_execution_identity"] is None

    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=6,
            acceptance_sequence=4,
            receipt_timestamp=fresh_receipt + 0.04,
            header_stamp_ns=fresh_stamp_ns + 6,
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=fresh_stamp_ns + 3,
            start_time_ns=fresh_stamp_ns + 4,
            traj_id=4,
            event=4,
            state=10,
        )
    )
    assert controller.status()["fresh_controller_execution_identity"] == pending


@pytest.mark.parametrize(
    "invalid_case",
    (
        "received_false",
        "drained_true",
        "old_source_sequence",
        "old_source_receipt",
        "same_status_receipt",
        "drain_not_crossed",
        "wrong_owner",
        "inhibited",
        "stop_reason",
    ),
)
def test_resume_rejects_unproven_cmd_vel_source(invalid_case: str) -> None:
    controller, release_sequence, release_timestamp, tracking_receipt = (
        _controller_with_fresh_tracking_handoff()
    )
    status = controller.status()
    source_sequence = release_sequence + 20
    source_receipt = tracking_receipt + 0.02
    report = _policy_report(
        release_sequence + 20,
        timestamp=source_receipt,
        motion_allowed=True,
        stop_reasons=(),
        navigation_cmd_vel_inhibited=False,
        cmd_vel_source_sequence=source_sequence,
        cmd_vel_source_receipt_timestamp=source_receipt,
        last_cmd_vel_drain_sequence=status[
            "release_cmd_vel_drain_sequence"
        ],
        last_cmd_vel_drain_receipt_timestamp=status[
            "release_cmd_vel_drain_receipt_timestamp"
        ],
    )
    if invalid_case == "received_false":
        report["cmd_vel_sample_received_this_tick"] = False
    elif invalid_case == "drained_true":
        report["cmd_vel_sample_drained_this_tick"] = True
    elif invalid_case == "old_source_sequence":
        report["cmd_vel_source_sequence"] = status[
            "release_cmd_vel_source_sequence"
        ]
    elif invalid_case == "old_source_receipt":
        report["cmd_vel_source_receipt_timestamp"] = release_timestamp
    elif invalid_case == "same_status_receipt":
        report["cmd_vel_source_receipt_timestamp"] = tracking_receipt
    elif invalid_case == "drain_not_crossed":
        report["last_cmd_vel_drain_sequence"] = source_sequence
        report["last_cmd_vel_drain_receipt_timestamp"] = source_receipt - 0.01
    elif invalid_case == "wrong_owner":
        report["owner_id"] = "other_writer"
    elif invalid_case == "inhibited":
        report["navigation_cmd_vel_inhibited"] = True
        report["navigation_cmd_vel_inhibit_reason"] = "unexpected_inhibit"
    elif invalid_case == "stop_reason":
        report["stop_reasons"] = ["point_cloud_timeout"]
    else:  # pragma: no cover - 参数集合由测试本身固定
        raise AssertionError(invalid_case)

    controller.observe_policy_write(report)
    assert controller.finish_ready is False

    # 被拒报告不能污染下界；随后一条真正晚于 controller 状态和 drain 的
    # 新 Twist 仍可完成交接。
    valid_receipt = tracking_receipt + 0.10
    controller.observe_policy_write(
        _policy_report(
            release_sequence + 50,
            timestamp=valid_receipt,
            motion_allowed=True,
            stop_reasons=(),
            navigation_cmd_vel_inhibited=False,
            cmd_vel_source_sequence=release_sequence + 50,
            cmd_vel_source_receipt_timestamp=valid_receipt,
            last_cmd_vel_drain_sequence=status[
                "release_cmd_vel_drain_sequence"
            ],
            last_cmd_vel_drain_receipt_timestamp=status[
                "release_cmd_vel_drain_receipt_timestamp"
            ],
        )
    )
    assert controller.finish_ready is True


def test_fresh_controller_identity_invalidation_revokes_handoff() -> None:
    controller, release_sequence, _, tracking_receipt = (
        _controller_with_fresh_tracking_handoff()
    )
    status = controller.status()
    identity = status["fresh_controller_execution_identity"]
    assert isinstance(identity, list)
    path_stamp_ns, bspline_stamp_ns, start_time_ns, traj_id = identity
    controller.observe_controller_status(
        _controller_status_report(
            status_sequence=4,
            acceptance_sequence=2,
            receipt_timestamp=tracking_receipt + 0.02,
            header_stamp_ns=int(round((tracking_receipt + 0.02) * 1.0e9)),
            reference_path_stamp_ns=path_stamp_ns,
            bspline_header_stamp_ns=bspline_stamp_ns,
            start_time_ns=start_time_ns,
            traj_id=traj_id,
            event=3,
            state=4,
            trajectory_valid=False,
        )
    )
    assert controller.status()["fresh_controller_execution_identity"] is None
    controller.observe_policy_write(
        _policy_report(
            release_sequence + 20,
            timestamp=tracking_receipt + 0.04,
            motion_allowed=True,
            stop_reasons=(),
            navigation_cmd_vel_inhibited=False,
        )
    )
    assert controller.finish_ready is False


def test_terminal_component_keeps_certified_full_body_lock() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
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
        )
    )
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.15),
        (0.4, 0.0, 0.30),
    )
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )

    target = (0.0, 0.0, 0.30)
    terminal_action = None
    for step in range(10):
        action = controller.compute_action(
            _state(
                step,
                timestamp=step * 0.02,
                xyz=target,
                write_sequence=step,
            )
        )
        assert action is not None
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if action.source == "scan_stair_freeze_terminal_hold":
            terminal_action = action
            break

    assert terminal_action is not None
    assert terminal_action.metadata["navigation_base_pose_lock"] is True
    assert terminal_action.metadata["navigation_support_joint_lock"] is True
    assert terminal_action.metadata["navigation_full_body_joint_lock"] is True
    assert terminal_action.metadata["navigation_cmd_vel_inhibit_reason"] == (
        "scan_stair_terminal_hold"
    )
    status = controller.status()
    assert status["phase"] == "terminal_hold"
    assert status["terminal_component"] is True
    assert status["terminal_hold"] is True
    assert status["completed_component_count"] == 1
    assert controller.finish_ready is True

    next_action = controller.compute_action(
        _state(
            step + 1,
            timestamp=(step + 1) * 0.02,
            xyz=target,
            write_sequence=step + 1,
            stop_reasons=("scan_stair_terminal_hold",),
        )
    )
    assert next_action is not None
    assert next_action.source == "scan_stair_freeze_terminal_hold"
    assert next_action.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        terminal_action.metadata["navigation_base_pose_lock_xyzyaw"]
    )


@pytest.mark.parametrize(
    ("goal", "path_terminal_yaw", "error_match"),
    [
        ((0.400002, 0.0, 0.60, 0.0), 0.0, "XY"),
        ((0.40, 0.0, 0.6002, 0.0), 0.0, r"ground\+body_height"),
        ((0.40, 0.0, 0.60, 0.0), 2.0e-5, "terminal yaw"),
    ],
)
def test_terminal_component_rejects_nav_goal_contract_mismatch(
    goal: tuple[float, float, float, float],
    path_terminal_yaw: float,
    error_match: str,
) -> None:
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.15),
        (0.4, 0.0, 0.30),
    )
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            approach_distance_m=0.0,
            exit_distance_m=0.0,
        )
    )

    with pytest.raises(ValueError, match=error_match):
        controller.reset(
            path,
            path_terminal_yaw=path_terminal_yaw,
            terminal_goal_base_xyzyaw=goal,
            stair_segment_indices=((0, 2),),
        )


def test_terminal_hold_uses_goal_yaw_with_shortest_pi_wrap() -> None:
    goal_yaw = -math.pi + 0.02
    path = (
        (0.0, 0.0, 0.0),
        (-0.2, 0.0, 0.15),
        (-0.4, 0.0, 0.30),
    )
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.15,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        path,
        path_terminal_yaw=goal_yaw,
        terminal_goal_base_xyzyaw=_terminal_goal(path, yaw=goal_yaw),
        stair_segment_indices=((0, 2),),
    )

    target = (0.0, 0.0, 0.30)
    action = None
    for step in range(10):
        action = controller.compute_action(
            _state(
                step,
                timestamp=step * 0.02,
                xyz=target,
                yaw=math.pi - 0.02,
            )
        )
        assert action is not None
        target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if action.source == "scan_stair_freeze_terminal_hold":
            break

    assert action is not None
    assert action.source == "scan_stair_freeze_terminal_hold"
    hold_yaw = action.metadata["navigation_base_pose_lock_xyzyaw"][3]
    assert hold_yaw == pytest.approx(goal_yaw)
    assert controller.status()["terminal_goal_yaw_error_rad"] == pytest.approx(0.0)


def test_terminal_hold_timeout_latches_full_body_emergency_hold() -> None:
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.15),
        (0.4, 0.0, 0.30),
    )
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=10.0,
            activation_radius_m=0.15,
            approach_distance_m=0.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            terminal_goal_hold_timeout_s=0.03,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )
    target = (0.0, 0.0, 0.30)
    terminal_step = 0
    for terminal_step in range(10):
        action = controller.compute_action(
            _state(
                terminal_step,
                timestamp=terminal_step * 0.02,
                xyz=target,
            )
        )
        assert action is not None
        target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if action.source == "scan_stair_freeze_terminal_hold":
            break

    timeout_timestamp = terminal_step * 0.02 + 0.04
    with pytest.raises(RuntimeError, match="stair_terminal_goal_hold_timeout"):
        controller.compute_action(
            _state(
                terminal_step + 1,
                timestamp=timeout_timestamp,
                xyz=target,
            )
        )
    emergency = controller.emergency_hold_action(
        _state(
            terminal_step + 2,
            timestamp=timeout_timestamp + 0.02,
            xyz=target,
        ),
        reason="stair_terminal_goal_hold_timeout",
    )
    assert emergency is not None
    assert emergency.metadata["navigation_full_body_joint_lock"] is True
    assert emergency.metadata["navigation_base_pose_lock_xyzyaw"][:3] == pytest.approx(
        target
    )


def test_emergency_hold_latches_last_active_target_and_joint_locks() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            activation_radius_m=0.15,
            min_component_z_delta_m=0.20,
            exit_distance_m=0.0,
        )
    )
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((1, 3),),
        carry_object_follow=True,
    )
    activated = controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.2, 0.0, 0.30))
    )
    assert activated is not None
    target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    hold = controller.emergency_hold_action(
        _state(1, timestamp=0.02, xyz=tuple(target[:3])),
        reason="scan_reference_path_cleared_during_stair_freeze",
    )

    assert hold is not None
    assert hold.source == "scan_stair_emergency_hold"
    assert hold.metadata["navigation_base_pose_lock_xyzyaw"] == target
    assert hold.metadata["navigation_base_pose_lock"] is True
    assert hold.metadata["navigation_support_joint_lock"] is True
    assert hold.metadata["navigation_full_body_joint_lock"] is True
    assert hold.metadata["navigation_carry_object_follow"] is True
    assert hold.metadata["navigation_cmd_vel_inhibit"] is True
    assert hold.metadata["navigation_cmd_vel_inhibit_reason"] == (
        "scan_stair_emergency_hold"
    )
    status = controller.status()
    assert status["emergency_hold_latched"] is True
    assert status["emergency_hold_origin_phase"] == "active"
    assert status["emergency_hold_full_body_lock"] is True

    repeated = controller.emergency_hold_action(
        _state(2, timestamp=0.04, xyz=tuple(target[:3])),
        reason="later_failure_must_not_replace_latched_reason",
    )
    assert repeated is not None
    assert repeated.metadata["navigation_base_pose_lock_xyzyaw"] == target
    assert repeated.metadata["navigation_stair_emergency_hold_reason"] == (
        "scan_reference_path_cleared_during_stair_freeze"
    )


def test_default_approach_buffer_covers_go2_x5_front_envelope() -> None:
    reference = load_scan_reference_path(
        PROJECT_ROOT
        / "ros2_ws/src/scan_navigation_tools/config/"
        "multifloor_stair_two_step_path.yaml"
    )
    controller = ScanStairFreezeController(ScanStairFreezeConfig())
    controller.reset(
        reference.points_ground_xyz,
        terminal_goal_base_xyzyaw=_terminal_goal(
            reference.points_ground_xyz,
        ),
        path_points_sha256=reference.points_sha256,
        stair_segment_indices=reference.stair_segment_indices,
    )

    start = reference.points_ground_xyz[0]
    action = controller.compute_action(
        _state(
            0,
            timestamp=1.0,
            xyz=(start[0], start[1] - 0.04, start[2] + 0.336),
        )
    )

    assert action is not None
    assert action.source == "scan_stair_freeze_activated"
    assert action.metadata["navigation_cmd_vel_inhibit"] is True
    assert action.metadata["navigation_base_pose_lock_xyzyaw"][:2] == pytest.approx(
        (start[0], start[1] - 0.04)
    )


def test_ground_height_receives_exactly_one_body_height_offset() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.20,
            min_component_z_delta_m=0.10,
            exit_distance_m=0.0,
            body_height_m=0.30,
            default_control_dt_s=0.10,
            max_control_dt_s=0.10,
        )
    )
    path = ((0.0, 0.0, 1.0), (0.2, 0.0, 1.15), (0.4, 0.0, 1.30))
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )
    first = controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.0, 0.0, 1.30))
    )
    assert first is not None
    second = controller.compute_action(
        _state(1, timestamp=0.1, xyz=(0.0, 0.0, 1.30))
    )
    assert second is not None
    target = second.metadata["navigation_base_pose_lock_xyzyaw"]
    status = controller.status()
    progress = status["progress_m"]
    segment_length = math.dist((0.0, 0.0, 1.0), (0.2, 0.0, 1.15))
    ratio = progress / segment_length
    expected_ground_z = 1.0 + ratio * 0.15
    assert target[2] == pytest.approx(expected_ground_z + 0.30)
    assert target[2] != pytest.approx(expected_ground_z + 0.60)


def test_measured_body_height_above_config_is_never_pushed_down() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.20,
            min_component_z_delta_m=0.10,
            exit_distance_m=0.0,
            body_height_m=0.30,
            default_control_dt_s=0.10,
            max_control_dt_s=0.10,
        )
    )
    path = ((0.0, 0.0, 1.0), (0.2, 0.0, 1.15), (0.4, 0.0, 1.30))
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )

    activated = controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.0, 0.0, 1.42))
    )
    advanced = controller.compute_action(
        _state(1, timestamp=0.1, xyz=(0.0, 0.0, 1.42))
    )

    assert activated is not None
    assert advanced is not None
    target = advanced.metadata["navigation_base_pose_lock_xyzyaw"]
    status = controller.status()
    progress = status["progress_m"]
    segment_length = math.dist(path[0], path[1])
    ratio = progress / segment_length
    expected_ground_z = path[0][2] + ratio * (path[1][2] - path[0][2])
    assert target[2] == pytest.approx(expected_ground_z + 0.42)
    assert status["measured_body_height_m"] == pytest.approx(0.42)
    assert status["configured_body_height_m"] == pytest.approx(0.30)
    assert status["target_body_height_m"] == pytest.approx(0.42)


def test_activation_rejects_xy_overlap_from_another_floor() -> None:
    """同一 XY 上的另一楼层不能触发当前有序楼梯段。"""

    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            activation_radius_m=0.20,
            min_component_z_delta_m=0.10,
            exit_distance_m=0.0,
        )
    )
    path = ((0.0, 0.0, 1.0), (0.2, 0.0, 1.15), (0.4, 0.0, 1.30))
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )

    action = controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.0, 0.0, 0.30))
    )

    assert action is None
    assert controller.status()["phase"] == "approach"
    assert controller.status()["reason"] == "activation_height_mismatch"


def test_active_progress_does_not_advance_while_sim_clock_is_paused() -> None:
    """已有时间基准后，重复仿真时间戳必须保持 root 目标不动。"""

    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=1.0,
            activation_radius_m=0.20,
            min_component_z_delta_m=0.10,
            exit_distance_m=0.0,
            default_control_dt_s=0.10,
            max_control_dt_s=0.10,
        )
    )
    path = ((0.0, 0.0, 1.0), (0.2, 0.0, 1.15), (0.4, 0.0, 1.30))
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )
    activated = controller.compute_action(
        _state(0, timestamp=1.0, xyz=(0.0, 0.0, 1.30))
    )
    assert activated is not None
    initial_target = activated.metadata["navigation_base_pose_lock_xyzyaw"]

    paused = controller.compute_action(
        _state(1, timestamp=1.0, xyz=tuple(initial_target[:3]))
    )

    assert paused is not None
    assert paused.metadata["navigation_base_pose_lock_xyzyaw"] == pytest.approx(
        initial_target
    )
    assert controller.status()["progress_m"] == pytest.approx(0.0)


def test_missed_stair_activation_fails_instead_of_waiting_forever() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(exit_distance_m=0.0)
    )
    path = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.4, 0.0, 0.15),
        (0.6, 0.0, 0.30),
    )
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((1, 3),),
    )

    with pytest.raises(RuntimeError, match="stair_activation_missed"):
        controller.compute_action(
            _state(1, timestamp=0.1, xyz=(0.60, 0.0, 0.60))
        )
    assert controller.status()["phase"] == "failed"


def test_activation_timeout_starts_only_after_real_policy_activity() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            activation_radius_m=0.05,
            activation_lookahead_m=0.50,
            activation_timeout_s=0.50,
            exit_distance_m=0.0,
        )
    )
    path = ((1.0, 0.0, 0.0), (1.2, 0.0, 0.15), (1.4, 0.0, 0.30))
    controller.reset(
        path,
        terminal_goal_base_xyzyaw=_terminal_goal(path),
        stair_segment_indices=((0, 2),),
    )

    # 已在 lookahead 内，但 planner/controller 尚未产生非零实写；即使等待
    # 超过 activation_timeout，也不能把启动等待误报为楼梯接近失败。
    assert controller.compute_action(
        _state(0, timestamp=0.0, xyz=(0.60, 0.0, 0.30))
    ) is None
    assert controller.compute_action(
        _state(1, timestamp=1.0, xyz=(0.60, 0.0, 0.30))
    ) is None
    waiting_status = controller.status()
    assert waiting_status["approach_window_entered_timestamp"] == 0.0
    assert waiting_status["approach_execution_activity_seen"] is False
    assert waiting_status["approach_started_timestamp"] is None

    active_state = _state(
        2,
        timestamp=1.02,
        xyz=(0.60, 0.0, 0.30),
        write_sequence=2,
        motion_allowed=True,
        stop_reasons=(),
        requested_command=(0.20, 0.0, 0.0),
        written_command=(0.02, 0.0, 0.0),
    )
    controller.observe_policy_write(
        active_state.metadata["scan_cmd_vel_last_write_report"]
    )
    assert controller.compute_action(active_state) is None
    active_status = controller.status()
    assert active_status["approach_execution_activity_seen"] is True
    assert active_status["approach_activity_write_sequence"] == 2
    assert active_status["approach_started_timestamp"] == pytest.approx(1.02)

    with pytest.raises(RuntimeError, match="stair_activation_timeout"):
        controller.compute_action(
            _state(3, timestamp=1.54, xyz=(0.60, 0.0, 0.30))
        )
    assert controller.status()["phase"] == "failed"


def test_post_release_stability_resets_on_measured_motion() -> None:
    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=10.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            root_release_settle_time_s=0.0,
            post_release_stable_time_s=0.04,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.15),
            (0.4, 0.0, 0.30),
            (0.6, 0.0, 0.30),
        ),
        stair_segment_indices=((0, 2),),
    )
    target = (0.0, 0.0, 0.30)
    release_seen = False
    step = 0
    for step in range(20):
        action = controller.compute_action(
            _state(step, timestamp=step * 0.02, xyz=target, write_sequence=step)
        )
        assert action is not None
        if "navigation_base_pose_lock_xyzyaw" in action.metadata:
            target = tuple(action.metadata["navigation_base_pose_lock_xyzyaw"][:3])
        if action.source == "scan_stair_freeze_released":
            release_seen = True
            break
    assert release_seen is True

    moving = controller.compute_action(
        _state(
            step + 1,
            timestamp=(step + 1) * 0.02,
            xyz=target,
            write_sequence=step + 1,
            velocity=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    assert moving is not None
    assert controller.status()["post_release_stable_elapsed_s"] == 0.0
    assert controller.finish_ready is False


def test_post_release_stability_uses_yaw_rate_not_gait_rocking_rate() -> None:
    """站姿 roll/pitch 周期摆动不能冒充 SCAN 航向角速度。"""

    controller = ScanStairFreezeController(
        ScanStairFreezeConfig(
            speed_mps=10.0,
            exit_distance_m=0.0,
            full_lock_settle_time_s=0.0,
            root_release_settle_time_s=0.0,
            post_release_stable_time_s=0.04,
            post_release_max_angular_speed_rps=0.20,
            default_control_dt_s=0.02,
            max_control_dt_s=0.02,
        )
    )
    controller.reset(
        (
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.15),
            (0.4, 0.0, 0.30),
            (0.6, 0.0, 0.30),
        ),
        stair_segment_indices=((0, 2),),
    )
    step, target = _advance_to_release(controller)

    rocking = controller.compute_action(
        _state(
            step + 1,
            timestamp=(step + 1) * 0.02,
            xyz=target,
            write_sequence=step + 1,
            velocity=(0.0, 0.0, 0.0, 0.45, -0.40, 0.10),
        )
    )
    assert rocking is not None
    assert rocking.metadata["navigation_scan_stair_freeze_stable"] is True
    assert controller.status()["post_release_last_angular_speed_rps"] == (
        pytest.approx(0.10)
    )

    yawing = controller.compute_action(
        _state(
            step + 2,
            timestamp=(step + 2) * 0.02,
            xyz=target,
            write_sequence=step + 2,
            velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.25),
        )
    )
    assert yawing is not None
    assert yawing.metadata["navigation_scan_stair_freeze_stable"] is False
    assert controller.status()["post_release_stable_elapsed_s"] == 0.0
