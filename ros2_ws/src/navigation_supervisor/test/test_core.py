"""测试 navigation supervisor core 的 Path 代际传感器安全门."""

from __future__ import annotations

import pytest

from navigation_core import (
    NavigationState,
    NavigationSupervisor,
    NavigationSupervisorConfig,
)


def _reach_tracking(
    supervisor: NavigationSupervisor,
    *,
    start_s: float,
) -> float:
    supervisor.start_goal(start_s)
    path_s = start_s + 0.1
    supervisor.report_global_path_available(path_s)
    sensor_s = path_s + 0.1
    supervisor.observe_odometry(sensor_s)
    supervisor.observe_point_cloud(sensor_s)
    decision = supervisor.report_scan_success(
        sensor_s,
        valid_until_s=sensor_s + 2.0,
    )
    assert decision.state is NavigationState.TRACKING
    return sensor_s


def test_path_acceptance_requires_post_path_sensor_callbacks() -> None:
    """Path 前的新鲜传感器不能替代当前 Path 代次的首次采集."""
    supervisor = NavigationSupervisor()
    supervisor.start_goal(1.0)
    supervisor.observe_odometry(1.05)
    supervisor.observe_point_cloud(1.05)

    accepted = supervisor.report_global_path_available(1.1)

    assert accepted.state is NavigationState.LOCAL_PLANNING
    assert accepted.stale_inputs == ("odometry", "point_cloud", "bspline")
    waiting_for_sensors = supervisor.report_scan_success(
        1.1,
        valid_until_s=2.0,
    )
    assert waiting_for_sensors.stale_inputs == ("odometry", "point_cloud")
    odometry_only = supervisor.observe_odometry(1.11)
    assert odometry_only.state is NavigationState.LOCAL_PLANNING
    assert odometry_only.stale_inputs == ("point_cloud",)

    ready = supervisor.observe_point_cloud(1.12)

    assert ready.state is NavigationState.TRACKING
    assert ready.stale_inputs == ()
    assert ready.allow_tracking_command is True


@pytest.mark.parametrize(
    ("observer", "message"),
    (
        ("observe_odometry", "Odometry 源时间"),
        ("observe_point_cloud", "PointCloud2 源时间"),
    ),
)
def test_core_still_rejects_future_sensor_observation(
    observer: str,
    message: str,
) -> None:
    """DDS 容差只属于 adapter，core 继续严格拒绝未来时间。"""

    supervisor = NavigationSupervisor()

    with pytest.raises(ValueError, match=message):
        getattr(supervisor, observer)(1.0, observed_at_s=1.001)


def test_replanned_path_cannot_inherit_pre_path_sensor_freshness() -> None:
    """第二代 Path 必须再次等待其接受后的 Odometry 与点云 callback."""
    supervisor = NavigationSupervisor()
    tracking_s = _reach_tracking(supervisor, start_s=1.0)
    supervisor.report_predicted_collision(tracking_s + 0.1)
    supervisor.report_global_planning_started(tracking_s + 0.2)
    supervisor.observe_odometry(tracking_s + 0.25)
    supervisor.observe_point_cloud(tracking_s + 0.25)

    replanned = supervisor.report_global_path_available(tracking_s + 0.3)

    assert replanned.state is NavigationState.LOCAL_PLANNING
    assert replanned.stale_inputs == ("odometry", "point_cloud", "bspline")
    supervisor.report_scan_success(
        tracking_s + 0.3,
        valid_until_s=tracking_s + 2.0,
    )
    after_odometry = supervisor.observe_odometry(tracking_s + 0.31)
    assert after_odometry.stale_inputs == ("point_cloud",)
    recovered = supervisor.observe_point_cloud(tracking_s + 0.32)
    assert recovered.state is NavigationState.TRACKING
    assert recovered.allow_tracking_command is True


def test_nonfinal_trajectory_finish_waits_for_replacement_without_emergency() -> None:
    """滚动局部轨迹自然结束时零速等待，不应误报急停或重规划。"""

    supervisor = NavigationSupervisor()
    tracking_s = _reach_tracking(supervisor, start_s=1.0)

    waiting = supervisor.report_local_trajectory_finished(tracking_s + 0.1)

    assert waiting.state is NavigationState.LOCAL_PLANNING
    assert waiting.reason == "local_trajectory_finished"
    assert waiting.stale_inputs == ("bspline",)
    assert waiting.force_zero_velocity is True
    assert waiting.global_replan_requested is False

    resumed = supervisor.report_scan_success(
        tracking_s + 0.2,
        valid_until_s=tracking_s + 2.0,
    )

    assert resumed.state is NavigationState.TRACKING
    assert resumed.allow_tracking_command is True
    assert resumed.global_replan_requested is False


@pytest.mark.parametrize(
    ("missing_sensor", "expected_stale"),
    (("odometry", "odometry"), ("point_cloud", "point_cloud")),
)
def test_tracking_sensor_timeout_behavior_is_unchanged(
    missing_sensor: str,
    expected_stale: str,
) -> None:
    """完成代际采集后，执行中任一传感器超时仍立即进入急停."""
    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=0.5,
            point_cloud_timeout_s=0.5,
            bspline_timeout_s=2.0,
        )
    )
    tracking_s = _reach_tracking(supervisor, start_s=1.0)
    refresh_s = tracking_s + 0.45
    if missing_sensor != "odometry":
        supervisor.observe_odometry(refresh_s)
    if missing_sensor != "point_cloud":
        supervisor.observe_point_cloud(refresh_s)

    stopped = supervisor.tick(tracking_s + 0.51)

    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert expected_stale in stopped.stale_inputs
    assert stopped.reason.startswith("input_timeout:")
    assert stopped.force_zero_velocity is True
    assert stopped.global_replan_requested is False


def test_scan_validity_extension_is_absolute_monotonic_and_bounded() -> None:
    """同一轨迹的状态心跳只能提升到固定绝对截止，不能滚动续期。"""

    supervisor = NavigationSupervisor(
        NavigationSupervisorConfig(
            odometry_timeout_s=20.0,
            point_cloud_timeout_s=20.0,
            bspline_timeout_s=2.0,
        )
    )
    tracking_s = _reach_tracking(supervisor, start_s=1.0)
    hard_expiry_s = tracking_s + 5.0

    extended = supervisor.extend_scan_trajectory_validity(
        tracking_s + 0.1,
        valid_until_s=hard_expiry_s,
    )
    repeated = supervisor.extend_scan_trajectory_validity(
        tracking_s + 0.2,
        valid_until_s=hard_expiry_s - 1.0,
    )

    assert extended.state is NavigationState.TRACKING
    assert repeated.state is NavigationState.TRACKING
    assert supervisor.tick(hard_expiry_s).state is NavigationState.TRACKING
    stopped = supervisor.tick(hard_expiry_s + 0.001)
    assert stopped.state is NavigationState.EMERGENCY_STOP
    assert stopped.stale_inputs == ("bspline",)
