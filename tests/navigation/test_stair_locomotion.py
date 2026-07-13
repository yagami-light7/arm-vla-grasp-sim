from __future__ import annotations

import math

import pytest

from source.interfaces import NavGoal, NavPlan, SimulationState
from source.navigation import (
    StairCenterlinePlanner,
    StairLocomotionExecutor,
    StairLocomotionExecutorConfig,
)


def _state(
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    timestamp: float = 0.0,
) -> SimulationState:
    return SimulationState(
        step_index=int(timestamp * 50.0),
        timestamp=timestamp,
        robot_root_pose=(
            x,
            y,
            z,
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ),
        robot_root_velocity=(0.0,) * 6,
    )


def _plan() -> NavPlan:
    path = ((0.0, 0.0, 0.4), (0.0, 2.0, 1.4), (1.0, 2.0, 2.4))
    return NavPlan(
        goal=NavGoal(x=1.0, y=2.0, z=2.4, yaw=0.0),
        waypoints=tuple((point[0], point[1]) for point in path),
        metadata={"planner": "pct_stair_centerline", "path_3d": path},
    )


def test_stair_centerline_planner_preserves_calibrated_path() -> None:
    path = ((0.0, 0.0, 0.4), (0.0, 2.0, 1.4), (1.0, 2.0, 2.4))
    planner = StairCenterlinePlanner(path)

    plan = planner.plan(
        _state(0.0, 0.0, 0.4, math.pi / 2.0),
        NavGoal(x=1.0, y=2.0, z=2.4, yaw=0.0),
    )

    assert plan.metadata["planner"] == "pct_stair_centerline"
    assert plan.metadata["controller"] == "stair_heading_tracker"
    assert plan.metadata["low_level_policy_isolation"] is True
    assert plan.metadata["path_3d"] == path


def test_stair_executor_commands_forward_motion_when_aligned() -> None:
    executor = StairLocomotionExecutor()
    executor.reset(_plan())

    action = executor.compute_action(_state(0.0, 0.0, 0.4, math.pi / 2.0))

    assert action.source == "stair_locomotion_heading_tracker"
    assert action.base_velocity[0] == pytest.approx(0.25)
    assert action.base_velocity[1] == pytest.approx(0.0, abs=1.0e-9)
    assert action.base_velocity[2] == pytest.approx(0.0, abs=1.0e-9)
    assert action.metadata["navigation_base_pose_lock"] is False
    assert executor.status()["float_enabled"] is False


def test_stair_executor_uses_lateral_velocity_without_changing_stair_heading() -> None:
    executor = StairLocomotionExecutor()
    executor.reset(_plan())

    action = executor.compute_action(_state(0.10, 0.50, 0.65, math.pi / 2.0))

    assert action.base_velocity[0] > 0.0
    assert action.base_velocity[1] > 0.0
    assert action.base_velocity[2] == pytest.approx(0.0, abs=1.0e-9)
    assert action.metadata["stair_heading_error_rad"] == pytest.approx(0.0)
    assert action.metadata["stair_cross_track_error_m"] == pytest.approx(-0.10)


def test_stair_executor_rotates_before_advancing_when_heading_is_wrong() -> None:
    executor = StairLocomotionExecutor()
    executor.reset(_plan())

    action = executor.compute_action(_state(0.0, 0.0, 0.4, 0.0))

    assert action.base_velocity[0] == pytest.approx(0.0, abs=1.0e-9)
    assert action.base_velocity[1] == pytest.approx(0.0, abs=1.0e-9)
    assert action.base_velocity[2] == pytest.approx(0.50)


def test_stair_executor_requires_xy_and_f2_height_before_completion() -> None:
    executor = StairLocomotionExecutor(
        StairLocomotionExecutorConfig(goal_z_tolerance_m=0.35)
    )
    executor.reset(_plan())

    assert not executor.is_done(_state(1.0, 2.0, 1.5, 0.0))
    assert executor.is_done(_state(1.0, 2.0, 2.4, 0.0))
    assert executor.status()["done"] is True


def test_stair_executor_reports_excessive_centerline_deviation() -> None:
    executor = StairLocomotionExecutor(
        StairLocomotionExecutorConfig(max_path_deviation_m=0.30)
    )
    executor.reset(_plan())

    action = executor.compute_action(_state(0.40, 0.50, 0.65, math.pi / 2.0))

    assert action.source == "stair_locomotion_failed"
    assert executor.status()["failed"] is True
    assert executor.status()["failure_reason"] == "stair_locomotion_path_deviation"
