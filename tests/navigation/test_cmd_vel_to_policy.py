from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from source.navigation.cmd_vel_to_policy import (
    CmdVelToPolicyAdapter,
    CmdVelToPolicyConfig,
    NavigationSafetyPermit,
    PolicyCommandInput,
    PolicyCommandOwnershipError,
    body_velocity_from_input,
)


class _FakeSink:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float, float]] = []

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        self.commands.append((float(vx), float(vy), float(wz)))


@dataclass
class _BuiltinTime:
    sec: int
    nanosec: int


def _permit(
    now: float,
    *,
    status_sequence: int = 1,
    state_revision: int = 1,
    goal_id: int = 10,
    path_stamp_ns: int = 20,
    allow: bool = True,
    identity_valid: bool = True,
) -> NavigationSafetyPermit:
    return NavigationSafetyPermit(
        header_stamp_ns=max(1, round(now * 1_000_000_000)),
        received_at=now,
        status_sequence=status_sequence,
        state_revision=state_revision,
        goal_id=goal_id if allow else max(0, goal_id),
        active_path_stamp_ns=path_stamp_ns if allow else max(0, path_stamp_ns),
        state=3 if allow else 5,
        allow_tracking_command=allow,
        force_zero_velocity=not allow,
        identity_valid=identity_valid,
        reason="允许跟踪" if allow else "安全停车",
    )


def _ready_adapter(
    *,
    config: CmdVelToPolicyConfig | None = None,
    now: float = 0.0,
    owner: str = "scan_controller",
) -> tuple[CmdVelToPolicyAdapter, _FakeSink]:
    sink = _FakeSink()
    adapter = CmdVelToPolicyAdapter(sink, config)
    assert adapter.claim(owner, now) is True
    adapter.accept_navigation_status(_permit(now), owner_id=owner)
    adapter.accept_cmd_vel((0.4, 0.2, 0.3), owner_id=owner, received_at=now)
    adapter.mark_odometry(owner_id=owner, received_at=now)
    adapter.mark_point_cloud(owner_id=owner, received_at=now)
    adapter.renew_control_lease(owner, now)
    return adapter, sink


def test_parses_ros_twist_and_supported_pure_data_inputs() -> None:
    twist = SimpleNamespace(
        linear=SimpleNamespace(x=0.1, y=-0.2),
        angular=SimpleNamespace(z=0.3),
    )
    stamped = SimpleNamespace(twist=twist)
    action = SimpleNamespace(base_velocity=(0.4, 0.5, -0.6))

    assert body_velocity_from_input(twist).as_tuple() == pytest.approx(
        (0.1, -0.2, 0.3)
    )
    assert body_velocity_from_input(stamped).as_tuple() == pytest.approx(
        (0.1, -0.2, 0.3)
    )
    assert body_velocity_from_input(action).as_tuple() == pytest.approx(
        (0.4, 0.5, -0.6)
    )
    assert body_velocity_from_input({"vx": 1, "vy": 2, "wz": 3}).as_tuple() == (
        1.0,
        2.0,
        3.0,
    )
    assert body_velocity_from_input(
        {"linear": {"x": 1, "y": 2}, "angular": {"z": 3}}
    ).as_tuple() == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "value",
    [
        (1.0, 2.0),
        (1.0, float("nan"), 3.0),
        {"linear": {"x": 1.0}, "angular": {"z": 2.0}},
        object(),
    ],
)
def test_rejects_malformed_or_nonfinite_commands(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        body_velocity_from_input(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_vx": 0.0},
        {"max_vy_rate": -1.0},
        {"cmd_vel_timeout_s": float("inf")},
        {"future_tolerance_s": -0.1},
        {"require_odometry": 1},
    ],
)
def test_config_rejects_unsafe_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CmdVelToPolicyConfig(**kwargs)


def test_first_claim_writes_zero_and_ros_time_objects_are_supported() -> None:
    sink = _FakeSink()
    adapter = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            require_odometry=False,
            require_point_cloud=False,
        ),
    )

    assert adapter.claim("controller", _BuiltinTime(4, 500_000_000)) is True
    adapter.accept_navigation_status(_permit(4.5), owner_id="controller")
    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="controller",
        received_at=_BuiltinTime(4, 500_000_000),
    )
    adapter.renew_control_lease("controller", _BuiltinTime(4, 500_000_000))
    report = adapter.write(owner_id="controller", now=_BuiltinTime(4, 600_000_000))

    assert sink.commands[0] == _ZERO
    assert report.written_command.vx == pytest.approx(0.05)


_ZERO = (0.0, 0.0, 0.0)


def test_limits_all_axes_then_applies_per_axis_rate_limits() -> None:
    config = CmdVelToPolicyConfig(
        max_vx=0.5,
        max_vy=0.25,
        max_wz=0.6,
        max_vx_rate=1.0,
        max_vy_rate=0.5,
        max_wz_rate=2.0,
    )
    adapter, sink = _ready_adapter(config=config)
    adapter.accept_cmd_vel((9.0, -9.0, 9.0), owner_id="scan_controller", received_at=0.0)

    report = adapter.write(owner_id="scan_controller", now=0.2)

    assert report.limited_target.as_tuple() == pytest.approx((0.5, -0.25, 0.6))
    assert report.written_command.as_tuple() == pytest.approx((0.2, -0.1, 0.4))
    assert report.clipped_axes == ("vx", "vy", "wz")
    assert report.rate_limited_axes == ("vx", "vy", "wz")
    assert sink.commands[-1] == pytest.approx((0.2, -0.1, 0.4))


def test_valid_command_ramps_from_previous_output() -> None:
    config = CmdVelToPolicyConfig(
        max_vy=0.30,
        max_vx_rate=1.0,
        max_vy_rate=1.0,
        max_wz_rate=1.0,
    )
    adapter, _sink = _ready_adapter(config=config)

    first = adapter.write(owner_id="scan_controller", now=0.1)
    second = adapter.write(owner_id="scan_controller", now=0.2)

    assert first.written_command.as_tuple() == pytest.approx((0.1, 0.1, 0.1))
    assert second.written_command.as_tuple() == pytest.approx((0.2, 0.2, 0.2))


def test_explicit_zero_command_stops_immediately_without_slew() -> None:
    adapter, sink = _ready_adapter()
    moving = adapter.write(owner_id="scan_controller", now=0.1)
    assert moving.written_command.vx > 0.0

    adapter.accept_cmd_vel(
        _ZERO,
        owner_id="scan_controller",
        received_at=0.11,
    )
    stopped = adapter.write(owner_id="scan_controller", now=0.11)

    assert stopped.motion_allowed is True
    assert stopped.stop_reasons == ()
    assert stopped.written_command.as_tuple() == _ZERO
    assert stopped.rate_limited_axes == ()
    assert sink.commands[-1] == _ZERO


def test_v15_explicit_zero_clears_positive_wz_rate_limit_history() -> None:
    """复现 v15 正转、显式零速、反转序列，零速后不得残留正向角速度。"""

    config = CmdVelToPolicyConfig(
        cmd_vel_timeout_s=2.0,
        odometry_timeout_s=2.0,
        point_cloud_timeout_s=2.0,
        navigation_status_timeout_s=2.0,
        control_lease_timeout_s=2.0,
    )
    adapter, sink = _ready_adapter(config=config)
    adapter.accept_cmd_vel(
        (0.0, 0.0, 0.45),
        owner_id="scan_controller",
        received_at=0.0,
    )
    positive = adapter.write(owner_id="scan_controller", now=0.45)
    assert positive.written_command.as_tuple() == pytest.approx((0.0, 0.0, 0.45))

    adapter.accept_cmd_vel(
        _ZERO,
        owner_id="scan_controller",
        received_at=0.46,
    )
    stopped = adapter.write(owner_id="scan_controller", now=0.46)
    assert stopped.written_command.as_tuple() == _ZERO
    assert stopped.rate_limited_axes == ()
    assert adapter.last_output.as_tuple() == _ZERO
    assert sink.commands[-1] == _ZERO

    adapter.accept_cmd_vel(
        (0.0, 0.0, -0.10),
        owner_id="scan_controller",
        received_at=0.47,
    )
    reversing = adapter.write(owner_id="scan_controller", now=0.51)
    assert reversing.written_command.wz == pytest.approx(-0.05)
    assert reversing.written_command.wz < 0.0
    assert reversing.rate_limited_axes == ("wz",)
    assert sink.commands[-1][2] == pytest.approx(-0.05)


def test_rotate_in_place_drops_residual_linear_velocity_immediately() -> None:
    adapter, _sink = _ready_adapter()
    moving = adapter.write(owner_id="scan_controller", now=0.1)
    assert moving.written_command.vx > 0.0
    assert moving.written_command.vy > 0.0

    adapter.accept_cmd_vel(
        (0.0, 0.0, 0.6),
        owner_id="scan_controller",
        received_at=0.11,
    )
    rotating = adapter.write(owner_id="scan_controller", now=0.11)

    assert rotating.written_command.vx == 0.0
    assert rotating.written_command.vy == 0.0
    assert rotating.written_command.wz > 0.0
    assert rotating.written_command.wz < 0.6
    assert rotating.rate_limited_axes == ("wz",)


def test_cmd_vel_timeout_stops_immediately_without_slew() -> None:
    config = CmdVelToPolicyConfig(cmd_vel_timeout_s=0.10)
    adapter, sink = _ready_adapter(config=config)
    moving = adapter.write(owner_id="scan_controller", now=0.05)
    assert moving.written_command.vx > 0.0

    stopped = adapter.write(owner_id="scan_controller", now=0.11)

    assert stopped.motion_allowed is False
    assert "cmd_vel_timeout" in stopped.stop_reasons
    assert stopped.written_command.as_tuple() == _ZERO
    assert sink.commands[-1] == _ZERO


@pytest.mark.parametrize(
    ("field", "timeout", "reason"),
    [
        ("odometry", 0.10, "odometry_timeout"),
        ("point_cloud", 0.10, "point_cloud_timeout"),
    ],
)
def test_stale_navigation_observation_forces_zero(
    field: str,
    timeout: float,
    reason: str,
) -> None:
    config = CmdVelToPolicyConfig(
        odometry_timeout_s=timeout,
        point_cloud_timeout_s=timeout,
        cmd_vel_timeout_s=1.0,
    )
    adapter, _sink = _ready_adapter(config=config)
    if field == "odometry":
        adapter.mark_point_cloud(owner_id="scan_controller", received_at=0.15)
    else:
        adapter.mark_odometry(owner_id="scan_controller", received_at=0.15)
    adapter.accept_cmd_vel(
        (0.4, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=0.15,
    )
    adapter.renew_control_lease("scan_controller", 0.15)

    report = adapter.write(owner_id="scan_controller", now=0.15)

    assert reason in report.stop_reasons
    assert report.written_command.as_tuple() == _ZERO


def test_missing_inputs_force_zero_until_every_required_input_arrives() -> None:
    sink = _FakeSink()
    adapter = CmdVelToPolicyAdapter(sink)
    adapter.claim("controller", 0.0)
    adapter.renew_control_lease("controller", 0.0)

    report = adapter.write(owner_id="controller", now=0.01)

    assert set(report.stop_reasons) == {
        "missing_cmd_vel",
        "missing_odometry",
        "missing_point_cloud",
        "missing_navigation_status",
    }
    assert sink.commands[-1] == _ZERO


def test_control_lease_timeout_forces_zero_but_owner_can_renew() -> None:
    config = CmdVelToPolicyConfig(
        control_lease_timeout_s=0.10,
        cmd_vel_timeout_s=1.0,
        odometry_timeout_s=1.0,
        point_cloud_timeout_s=1.0,
    )
    adapter, _sink = _ready_adapter(config=config)

    expired = adapter.write(owner_id="scan_controller", now=0.11)
    assert "control_lease_expired" in expired.stop_reasons
    assert expired.written_command.as_tuple() == _ZERO

    adapter.renew_control_lease("scan_controller", 0.11)
    resumed = adapter.write(owner_id="scan_controller", now=0.12)
    assert resumed.motion_allowed is True


def test_reclaim_after_own_expired_lease_writes_zero_and_clears_inputs() -> None:
    config = CmdVelToPolicyConfig(
        control_lease_timeout_s=0.10,
        cmd_vel_timeout_s=1.0,
        odometry_timeout_s=1.0,
        point_cloud_timeout_s=1.0,
    )
    adapter, sink = _ready_adapter(config=config)

    assert adapter.claim("scan_controller", 0.11) is True
    report = adapter.tick(0.12, "scan_controller")

    assert sink.commands[-2:] == [_ZERO, _ZERO]
    assert "missing_cmd_vel" in report.stop_reasons


def test_another_adapter_cannot_write_before_lease_expires() -> None:
    sink = _FakeSink()
    resource = "go2_env_0_policy_command"
    first = CmdVelToPolicyAdapter(sink, ownership_resource=resource)
    second = CmdVelToPolicyAdapter(sink, ownership_resource=resource)
    first.claim("same_label", 0.0)

    with pytest.raises(PolicyCommandOwnershipError):
        second.claim("same_label", 0.1)
    with pytest.raises(PolicyCommandOwnershipError):
        second.write(owner_id="same_label", now=0.1)

    assert sink.commands == [_ZERO]
    first.release(owner_id="same_label", now=0.1)


def test_default_resource_detects_two_wrappers_around_same_sink() -> None:
    sink = _FakeSink()
    first = CmdVelToPolicyAdapter(sink)
    second = CmdVelToPolicyAdapter(sink)
    first.claim("first", 0.0)

    with pytest.raises(PolicyCommandOwnershipError):
        second.claim("second", 0.1)

    first.release(owner_id="first", now=0.1)


def test_same_instance_cannot_change_owner_label_without_release() -> None:
    adapter, _sink = _ready_adapter(owner="first")

    with pytest.raises(PolicyCommandOwnershipError):
        adapter.claim("second", 0.1)

    adapter.release(owner_id="first", now=0.1)


def test_expired_owner_can_be_replaced_and_old_instance_loses_write_access() -> None:
    sink = _FakeSink()
    resource = "replaceable_policy_command"
    config = CmdVelToPolicyConfig(control_lease_timeout_s=0.10)
    first = CmdVelToPolicyAdapter(sink, config, ownership_resource=resource)
    second = CmdVelToPolicyAdapter(sink, config, ownership_resource=resource)
    first.claim("first", 0.0)

    assert second.claim("second", 0.11) is True
    with pytest.raises(PolicyCommandOwnershipError):
        first.write(owner_id="first", now=0.12)

    assert sink.commands == [_ZERO, _ZERO]
    second.release(owner_id="second", now=0.12)


def test_release_writes_zero_and_allows_new_owner() -> None:
    sink = _FakeSink()
    resource = "released_policy_command"
    first = CmdVelToPolicyAdapter(sink, ownership_resource=resource)
    second = CmdVelToPolicyAdapter(sink, ownership_resource=resource)
    first.claim("first", 0.0)

    first.release(owner_id="first", now=0.1)
    assert second.claim("second", 0.1) is True

    assert sink.commands == [_ZERO, _ZERO, _ZERO]
    second.release(owner_id="second", now=0.1)


def test_reset_keeps_owner_but_requires_fresh_episode_inputs() -> None:
    adapter, sink = _ready_adapter(now=10.0)
    adapter.tick(10.1, "scan_controller")

    adapter.reset(owner_id="scan_controller", now=0.0)
    blocked = adapter.tick(0.0, "scan_controller")

    assert adapter.owner_id == "scan_controller"
    assert sink.commands[-2:] == [_ZERO, _ZERO]
    assert set(blocked.stop_reasons) == {
        "missing_cmd_vel",
        "missing_odometry",
        "missing_point_cloud",
        "missing_navigation_status",
    }

    adapter.accept_navigation_status(_permit(0.0), owner_id="scan_controller")
    adapter.receive((0.2, 0.0, 0.0), 0.0, "scan_controller")
    adapter.mark_odometry(owner_id="scan_controller", received_at=0.0)
    adapter.mark_point_cloud(owner_id="scan_controller", received_at=0.0)
    resumed = adapter.tick(0.1, "scan_controller")
    assert resumed.motion_allowed is True


def test_clock_rewind_invalidates_inputs_and_requires_fresh_sim_time_data() -> None:
    adapter, sink = _ready_adapter(now=10.0)
    moving = adapter.write(owner_id="scan_controller", now=10.1)
    assert moving.written_command.vx > 0.0

    adapter.renew_control_lease("scan_controller", 0.0)
    rewound = adapter.write(owner_id="scan_controller", now=0.0)

    assert "clock_rewind" in rewound.stop_reasons
    assert "missing_cmd_vel" in rewound.stop_reasons
    assert rewound.written_command.as_tuple() == _ZERO
    assert sink.commands[-1] == _ZERO

    adapter.accept_navigation_status(_permit(0.0), owner_id="scan_controller")
    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=0.0,
    )
    adapter.mark_odometry(owner_id="scan_controller", received_at=0.0)
    adapter.mark_point_cloud(owner_id="scan_controller", received_at=0.0)
    resumed = adapter.write(owner_id="scan_controller", now=0.1)
    assert resumed.motion_allowed is True


def test_tick_before_latest_receipt_is_treated_as_clock_rewind() -> None:
    adapter, _sink = _ready_adapter(now=1.0)
    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=2.0,
    )

    report = adapter.write(owner_id="scan_controller", now=1.0)

    assert "clock_rewind" in report.stop_reasons
    assert "missing_cmd_vel" in report.stop_reasons
    assert report.written_command.as_tuple() == _ZERO


def test_invalid_command_invalidates_previous_command() -> None:
    adapter, _sink = _ready_adapter()

    with pytest.raises(ValueError):
        adapter.accept_cmd_vel(
            (float("nan"), 0.0, 0.0),
            owner_id="scan_controller",
            received_at=0.1,
        )
    report = adapter.write(owner_id="scan_controller", now=0.1)

    assert "invalid_cmd_vel" in report.stop_reasons
    assert report.written_command.as_tuple() == _ZERO


def test_optional_sensor_gates_can_be_disabled_explicitly() -> None:
    config = CmdVelToPolicyConfig(
        require_odometry=False,
        require_point_cloud=False,
    )
    sink = _FakeSink()
    adapter = CmdVelToPolicyAdapter(sink, config)
    adapter.claim("controller", 0.0)
    adapter.accept_navigation_status(_permit(0.0), owner_id="controller")
    adapter.accept_cmd_vel((0.2, 0.0, 0.0), owner_id="controller", received_at=0.0)
    adapter.renew_control_lease("controller", 0.0)

    report = adapter.write(owner_id="controller", now=0.1)

    assert report.motion_allowed is True
    assert report.stop_reasons == ()


def test_emergency_stop_bypasses_rate_limit_and_invalidates_command() -> None:
    adapter, sink = _ready_adapter()
    adapter.write(owner_id="scan_controller", now=0.1)

    stopped = adapter.emergency_stop(
        owner_id="scan_controller",
        now=0.11,
        reason="predicted_collision",
    )
    next_tick = adapter.write(owner_id="scan_controller", now=0.12)

    assert stopped.stop_reasons == ("predicted_collision",)
    assert stopped.written_command.as_tuple() == _ZERO
    assert "missing_cmd_vel" in next_tick.stop_reasons
    assert sink.commands[-2:] == [_ZERO, _ZERO]


def test_policy_envelope_allows_motion_then_force_zero_writes_actual_zero() -> None:
    sink = _FakeSink()
    adapter = CmdVelToPolicyAdapter(sink)
    adapter.claim("controller", 1.0)
    adapter.mark_odometry(owner_id="controller", received_at=1.0)
    adapter.mark_point_cloud(owner_id="controller", received_at=1.0)
    adapter.renew_control_lease("controller", 1.0)

    accepted = adapter.receive(
        PolicyCommandInput(
            command=(0.2, 0.0, 0.0),
            navigation_permit=_permit(1.0),
        ),
        1.0,
        "controller",
    )
    moving = adapter.tick(1.1, "controller")
    assert accepted is not None
    assert moving.motion_allowed is True
    assert moving.written_command.vx > 0.0

    status_only = adapter.receive(
        PolicyCommandInput(
            navigation_permit=_permit(1.11, status_sequence=2, allow=False),
        ),
        1.11,
        "controller",
    )
    stopped = adapter.tick(1.11, "controller")

    assert status_only is None
    assert "navigation_status_force_zero" in stopped.stop_reasons
    assert "navigation_tracking_not_allowed" in stopped.stop_reasons
    assert stopped.written_command.as_tuple() == _ZERO
    assert sink.commands[-1] == _ZERO


def test_identity_change_or_invalid_identity_cannot_revive_old_twist() -> None:
    adapter, sink = _ready_adapter(now=1.0)
    assert adapter.tick(1.1, "scan_controller").motion_allowed is True

    adapter.receive(
        PolicyCommandInput(
            navigation_permit=_permit(
                1.11,
                status_sequence=2,
                state_revision=2,
                goal_id=11,
                path_stamp_ns=21,
            ),
        ),
        1.11,
        "scan_controller",
    )
    mismatched = adapter.tick(1.11, "scan_controller")
    assert "navigation_command_identity_mismatch" in mismatched.stop_reasons
    assert sink.commands[-1] == _ZERO

    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=1.12,
    )
    assert adapter.tick(1.12, "scan_controller").motion_allowed is True

    adapter.receive(
        PolicyCommandInput(
            navigation_permit=_permit(
                1.11,
                status_sequence=2,
                state_revision=2,
                goal_id=11,
                path_stamp_ns=21,
                identity_valid=False,
            ),
        ),
        1.13,
        "scan_controller",
    )
    invalid = adapter.tick(1.13, "scan_controller")
    assert "navigation_status_identity_invalid" in invalid.stop_reasons
    assert invalid.written_command.as_tuple() == _ZERO


def test_navigation_status_freshness_and_sequence_faults_fail_closed() -> None:
    config = CmdVelToPolicyConfig(
        cmd_vel_timeout_s=1.0,
        odometry_timeout_s=1.0,
        point_cloud_timeout_s=1.0,
        control_lease_timeout_s=1.0,
        navigation_status_timeout_s=0.10,
    )
    adapter, sink = _ready_adapter(config=config, now=1.0)

    stale = adapter.tick(1.11, "scan_controller")
    assert "navigation_status_timeout" in stale.stop_reasons
    assert "navigation_status_source_timeout" in stale.stop_reasons
    assert sink.commands[-1] == _ZERO

    adapter.accept_navigation_status(
        _permit(1.12, status_sequence=2),
        owner_id="scan_controller",
    )
    heartbeat_only = adapter.tick(1.12, "scan_controller")
    assert "navigation_command_identity_mismatch" in (
        heartbeat_only.stop_reasons
    )

    adapter.accept_navigation_status(
        _permit(2.0, status_sequence=3),
        owner_id="scan_controller",
    )
    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=1.2,
    )
    future = adapter.tick(1.2, "scan_controller")
    assert "navigation_status_from_future" in future.stop_reasons
    assert "navigation_status_source_from_future" in future.stop_reasons

    adapter.accept_navigation_status(
        _permit(1.21, status_sequence=2),
        owner_id="scan_controller",
    )
    regression = adapter.tick(1.21, "scan_controller")
    assert any(
        reason.startswith("navigation_status_invalid:")
        for reason in regression.stop_reasons
    )
    assert regression.written_command.as_tuple() == _ZERO


def test_temporary_inhibit_preserves_permit_but_requires_post_freeze_twist() -> None:
    adapter, sink = _ready_adapter(now=1.0)
    assert adapter.tick(1.1, "scan_controller").motion_allowed is True

    frozen = adapter.inhibit(
        owner_id="scan_controller",
        now=1.11,
        reason="scan_stair_freeze",
    )
    blocked = adapter.tick(1.12, "scan_controller")
    assert frozen.written_command.as_tuple() == _ZERO
    assert "missing_cmd_vel" in blocked.stop_reasons

    adapter.accept_cmd_vel(
        (0.2, 0.0, 0.0),
        owner_id="scan_controller",
        received_at=1.13,
    )
    resumed = adapter.tick(1.13, "scan_controller")
    assert resumed.motion_allowed is True
    assert sink.commands[-1][0] > 0.0


def test_temporary_inhibit_does_not_mask_sensor_timeouts() -> None:
    """楼梯冻结写零仍必须把独立 Odometry/点云失鲜暴露给 root 锁。"""

    adapter, sink = _ready_adapter(now=1.0)

    frozen = adapter.inhibit(
        owner_id="scan_controller",
        now=1.60,
        reason="scan_stair_freeze",
    )

    assert frozen.motion_allowed is False
    assert frozen.written_command.as_tuple() == _ZERO
    assert frozen.stop_reasons == (
        "scan_stair_freeze",
        "odometry_timeout",
        "point_cloud_timeout",
    )
    assert sink.commands[-1] == _ZERO


def test_navigation_gate_diagnostics_preserves_consumed_identity() -> None:
    adapter, _sink = _ready_adapter(now=1.0)

    report = adapter.navigation_gate_diagnostics()

    assert report["schema"] == "navigation_policy_gate_diagnostics_v1"
    assert report["required"] is True
    assert report["status_fault"] is None
    assert report["permit"] == {
        "header_stamp_ns": 1_000_000_000,
        "received_at": 1.0,
        "status_sequence": 1,
        "state_revision": 1,
        "goal_id": 10,
        "active_path_stamp_ns": 20,
        "state": 3,
        "allow_tracking_command": True,
        "force_zero_velocity": False,
        "identity_valid": True,
        "reason": "允许跟踪",
    }
    assert report["command_identity"] == [10, 20, 1]
    assert report["command_identity_matches_permit"] is True

    adapter.invalidate_navigation_status(
        "invalid_navigation_status",
        owner_id="scan_controller",
    )
    invalid = adapter.navigation_gate_diagnostics()
    assert invalid["permit_received"] is False
    assert invalid["status_fault"] == "invalid_navigation_status"
    assert invalid["command_identity"] is None
