"""测试 PCT + SCAN 导航 supervisor 的安全状态语义。"""

from __future__ import annotations

import json
import math

import pytest

from source.navigation.navigation_supervisor import (
    NavigationState,
    NavigationSupervisor,
    NavigationSupervisorConfig,
    ZERO_BODY_VELOCITY,
)


def _reach_local_planning(
    supervisor: NavigationSupervisor,
    *,
    start_s: float = 1.0,
) -> float:
    supervisor.start_goal(start_s)
    supervisor.report_global_path_available(start_s + 0.1)
    return start_s + 0.1


def _reach_tracking(
    supervisor: NavigationSupervisor,
    *,
    start_s: float = 1.0,
    bspline_duration_s: float = 2.0,
) -> float:
    now = _reach_local_planning(supervisor, start_s=start_s)
    supervisor.observe_odometry(now + 0.1)
    supervisor.observe_point_cloud(now + 0.1)
    decision = supervisor.report_scan_success(
        now + 0.1,
        valid_until_s=now + 0.1 + bspline_duration_s,
    )
    assert decision.state is NavigationState.TRACKING
    return now + 0.1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"odometry_timeout_s": 0.0}, "odometry_timeout_s"),
        ({"point_cloud_timeout_s": math.inf}, "point_cloud_timeout_s"),
        ({"bspline_timeout_s": -1.0}, "bspline_timeout_s"),
        ({"bspline_timeout_s": None}, "bspline_timeout_s"),
        ({"odometry_timeout_s": True}, "odometry_timeout_s"),
        ({"max_consecutive_scan_failures": 0}, "正整数"),
        ({"max_consecutive_scan_failures": True}, "正整数"),
    ),
)
def test_config_rejects_invalid_safety_limits(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NavigationSupervisorConfig(**kwargs)


def test_nominal_state_sequence_only_forwards_command_during_tracking() -> None:
    supervisor = NavigationSupervisor()

    idle = supervisor.tick(1.0)
    global_planning = supervisor.start_goal(1.1)
    local_planning = supervisor.report_global_path_available(1.2)
    supervisor.observe_odometry(1.3)
    supervisor.observe_point_cloud(1.3)
    tracking = supervisor.report_scan_success(1.3, valid_until_s=2.0)

    assert [
        idle.state,
        global_planning.state,
        local_planning.state,
        tracking.state,
    ] == [
        NavigationState.IDLE,
        NavigationState.GLOBAL_PLANNING,
        NavigationState.LOCAL_PLANNING,
        NavigationState.TRACKING,
    ]
    for decision in (idle, global_planning, local_planning):
        assert decision.force_zero_velocity is True
        assert decision.velocity_override == ZERO_BODY_VELOCITY
        assert decision.allow_tracking_command is False
    assert tracking.allow_tracking_command is True
    assert tracking.velocity_override is None


def test_bspline_waits_for_both_odometry_and_point_cloud() -> None:
    supervisor = NavigationSupervisor()
    _reach_local_planning(supervisor)

    supervisor.observe_odometry(1.2)
    waiting = supervisor.report_scan_success(1.2, valid_until_s=2.0)

    assert waiting.state is NavigationState.LOCAL_PLANNING
    assert waiting.stale_inputs == ("point_cloud",)
    assert waiting.force_zero_velocity is True

    tracking = supervisor.observe_point_cloud(1.3)
    assert tracking.state is NavigationState.TRACKING
    assert tracking.allow_tracking_command is True


def test_sensor_freshness_uses_source_stamp_not_callback_receipt() -> None:
    """接近超时的源消息不能再获得第二个完整 receipt 超时窗口。"""

    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=0.5,
            point_cloud_timeout_s=0.5,
            bspline_timeout_s=2.0,
        )
    )
    _reach_local_planning(supervisor)
    supervisor.observe_odometry(2.0, observed_at_s=1.6)
    supervisor.observe_point_cloud(2.0, observed_at_s=1.6)
    supervisor.report_scan_success(2.0, valid_until_s=3.0)

    stopped = supervisor.tick(2.11)

    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert "odometry" in stopped.stale_inputs
    assert "point_cloud" in stopped.stale_inputs


@pytest.mark.parametrize(
    ("missing_update", "expected_stale"),
    (
        ("odometry", "odometry"),
        ("point_cloud", "point_cloud"),
    ),
)
def test_sensor_timeout_enters_emergency_stop_without_guessing_pct_replan(
    missing_update: str,
    expected_stale: str,
) -> None:
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=0.5,
            point_cloud_timeout_s=0.5,
            bspline_timeout_s=2.0,
        )
    )
    _reach_tracking(supervisor, bspline_duration_s=2.0)
    if missing_update != "odometry":
        supervisor.observe_odometry(1.65)
    if missing_update != "point_cloud":
        supervisor.observe_point_cloud(1.65)

    stopped = supervisor.tick(1.71)

    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert stopped.force_zero_velocity is True
    assert expected_stale in stopped.stale_inputs
    assert stopped.reason.startswith("input_timeout:")
    assert stopped.global_replan_requested is False


def test_bspline_expiry_enters_emergency_stop() -> None:
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=2.0,
            point_cloud_timeout_s=2.0,
            bspline_timeout_s=0.5,
        )
    )
    _reach_tracking(supervisor, bspline_duration_s=0.4)

    stopped = supervisor.tick(1.61)

    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert stopped.stale_inputs == ("bspline",)
    assert stopped.force_zero_velocity is True
    assert stopped.global_replan_requested is False


def test_consecutive_scan_failures_stop_and_latch_one_replan_request() -> None:
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(max_consecutive_scan_failures=3)
    )
    _reach_tracking(supervisor)

    first = supervisor.report_scan_failure(1.3)
    second = supervisor.report_scan_failure(1.4)
    stopped = supervisor.report_scan_failure(1.5)
    held = supervisor.tick(1.6)

    assert first.state is NavigationState.TRACKING
    assert first.consecutive_scan_failures == 1
    assert second.state is NavigationState.TRACKING
    assert stopped.state is NavigationState.GLOBAL_REPLAN
    assert stopped.force_zero_velocity is True
    assert stopped.global_replan_requested is True
    assert stopped.global_replan_request_id == 1
    assert held.global_replan_request_id == stopped.global_replan_request_id


def test_scan_success_resets_failure_counter() -> None:
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(max_consecutive_scan_failures=2)
    )
    _reach_tracking(supervisor)
    supervisor.report_scan_failure(1.3)

    recovered = supervisor.report_scan_success(1.4, valid_until_s=2.0)
    next_failure = supervisor.report_scan_failure(1.5)

    assert recovered.consecutive_scan_failures == 0
    assert next_failure.state is NavigationState.TRACKING
    assert next_failure.consecutive_scan_failures == 1


def test_predicted_collision_immediately_stops_and_requests_pct_replan() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)

    stopped = supervisor.report_predicted_collision(
        1.3,
        reason="trajectory_collision_ahead",
    )

    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert stopped.force_zero_velocity is True
    assert stopped.global_replan_requested is True
    assert stopped.global_replan_request_id == 1
    assert stopped.reason == "trajectory_collision_ahead"
    with pytest.raises(RuntimeError, match="旧 SCAN 轨迹"):
        supervisor.report_scan_success(1.4, valid_until_s=2.0)


def test_terminal_replan_transport_failure_stops_without_phantom_request() -> None:
    """有界 transport 失败后不应继续把已终止请求标成 pending。"""

    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)
    collision = supervisor.report_predicted_collision(1.3)

    stopped = supervisor.report_global_replan_transport_failed(
        1.4,
        reason="pct_replan_service_unavailable",
    )

    assert collision.global_replan_requested is True
    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert stopped.force_zero_velocity is True
    assert stopped.global_replan_requested is False
    assert stopped.global_replan_in_flight is False
    assert stopped.global_replan_request_id == collision.global_replan_request_id
    assert stopped.reason == "pct_replan_service_unavailable"


def test_pct_replan_handshake_returns_to_local_planning_then_tracking() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)
    stopped = supervisor.report_predicted_collision(1.3)

    planning = supervisor.report_global_planning_started(1.4)
    local = supervisor.report_global_path_available(1.5)
    supervisor.observe_odometry(1.6)
    supervisor.observe_point_cloud(1.6)
    tracking = supervisor.report_scan_success(1.6, valid_until_s=2.5)

    assert planning.state is NavigationState.GLOBAL_PLANNING
    assert planning.global_replan_requested is False
    assert planning.global_replan_in_flight is True
    assert local.state is NavigationState.LOCAL_PLANNING
    assert local.global_replan_requested is False
    assert local.global_replan_in_flight is False
    assert tracking.state is NavigationState.TRACKING
    assert tracking.global_replan_request_id == stopped.global_replan_request_id


def test_failed_acknowledged_replan_allocates_new_request_id() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)
    first = supervisor.report_predicted_collision(1.3)

    acknowledged = supervisor.report_global_planning_started(1.4)
    second = supervisor.report_global_planning_failed(
        1.5,
        reason="pct_no_path_after_replan",
    )

    assert first.global_replan_request_id == 1
    assert acknowledged.global_replan_requested is False
    assert acknowledged.global_replan_in_flight is True
    assert second.state is NavigationState.GLOBAL_REPLAN
    assert second.global_replan_requested is True
    assert second.global_replan_in_flight is False
    assert second.global_replan_request_id == 2


def test_global_planning_failure_requests_replan_while_holding_zero() -> None:
    supervisor = NavigationSupervisor()
    supervisor.start_goal(1.0)

    failed = supervisor.report_global_planning_failed(
        1.1,
        reason="pct_no_path",
    )

    assert failed.state is NavigationState.GLOBAL_REPLAN
    assert failed.reason == "pct_no_path"
    assert failed.global_replan_requested is True
    assert failed.force_zero_velocity is True


def test_timeout_emergency_requires_fresh_inputs_before_explicit_clear() -> None:
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=0.5,
            point_cloud_timeout_s=0.5,
            bspline_timeout_s=1.0,
        )
    )
    _reach_tracking(supervisor)
    supervisor.tick(1.71)

    with pytest.raises(RuntimeError, match="尚未全部恢复"):
        supervisor.clear_emergency(1.72)

    supervisor.observe_odometry(1.73)
    supervisor.observe_point_cloud(1.73)
    supervisor.report_scan_success(1.73, valid_until_s=2.5)
    recovered = supervisor.clear_emergency(1.74)

    assert recovered.state is NavigationState.TRACKING
    assert recovered.allow_tracking_command is True


def test_repeated_external_emergency_preserves_original_resume_state() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)

    supervisor.report_emergency_stop(1.3, reason="bumper")
    repeated = supervisor.report_emergency_stop(1.4, reason="bumper_still_active")
    supervisor.observe_odometry(1.5)
    supervisor.observe_point_cloud(1.5)
    supervisor.report_scan_success(1.5, valid_until_s=2.5)
    recovered = supervisor.clear_emergency(1.6)

    assert repeated.state is NavigationState.EMERGENCY_STOP
    assert recovered.state is NavigationState.TRACKING


def test_goal_reached_latches_zero_velocity_across_future_ticks() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)

    reached = supervisor.report_goal_reached(1.3)
    repeated = supervisor.report_goal_reached(1.4)
    much_later = supervisor.tick(100.0)

    assert reached.state is NavigationState.GOAL_REACHED
    assert repeated.state is NavigationState.GOAL_REACHED
    assert much_later.state is NavigationState.GOAL_REACHED
    assert much_later.force_zero_velocity is True
    assert much_later.global_replan_requested is False

    next_goal = supervisor.start_goal(100.1)
    assert next_goal.state is NavigationState.GLOBAL_PLANNING
    assert next_goal.force_zero_velocity is True


def test_cancel_returns_to_idle_and_invalidates_tracking() -> None:
    supervisor = NavigationSupervisor()
    _reach_tracking(supervisor)

    cancelled = supervisor.cancel(1.3)

    assert cancelled.state is NavigationState.IDLE
    assert cancelled.force_zero_velocity is True
    assert "bspline" in cancelled.stale_inputs


@pytest.mark.parametrize("invalid_time", (-1.0, math.inf, math.nan, True, "x"))
def test_invalid_ros_clock_values_fail_loud(invalid_time: object) -> None:
    supervisor = NavigationSupervisor()

    with pytest.raises(ValueError, match="now_s"):
        supervisor.tick(invalid_time)  # type: ignore[arg-type]


def test_ros_clock_cannot_move_backwards() -> None:
    supervisor = NavigationSupervisor()
    supervisor.tick(2.0)

    with pytest.raises(ValueError, match="不能倒退"):
        supervisor.tick(1.9)


def test_bspline_validity_must_cover_receive_time() -> None:
    supervisor = NavigationSupervisor()
    _reach_local_planning(supervisor)

    with pytest.raises(ValueError, match="不能早于"):
        supervisor.report_scan_success(1.2, valid_until_s=1.1)


def test_status_payload_is_json_serializable_and_exposes_adapter_gate() -> None:
    supervisor = NavigationSupervisor()
    decision = supervisor.start_goal(1.0)

    payload = decision.to_status_dict()

    assert json.loads(json.dumps(payload))["state"] == "GLOBAL_PLANNING"
    assert payload["force_zero_velocity"] is True
    assert payload["allow_tracking_command"] is False
    assert payload["global_replan_in_flight"] is False
