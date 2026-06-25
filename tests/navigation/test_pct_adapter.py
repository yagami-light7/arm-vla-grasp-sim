from __future__ import annotations

import math
from typing import Any

import pytest

from source.interfaces import NavGoal, SimulationState
from source.navigation.pct_adapter import PCTNavPlanner, PCTPlannerConfig


class FakePCTClient:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.requests: list[dict[str, tuple[float, float, float]]] = []

    def plan(self, *, start, end):
        self.requests.append({"start": tuple(start), "end": tuple(end)})
        return self.response


def _state(x: float, y: float, z: float = 0.35) -> SimulationState:
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(x, y, z, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
    )


def test_pct_nav_planner_returns_xy_waypoints_and_path_3d_metadata() -> None:
    client = FakePCTClient(
        {
            "status": "ok",
            "traj": [
                [-1.0, -2.0, 0.35],
                [0.0, 0.0, 1.0],
                [2.0, 3.0, 1.8],
            ],
            "slice_start": 1,
            "slice_end": 4,
            "snap_start_dist": 0.02,
            "snap_end_dist": 0.04,
        }
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=client,
    )

    plan = planner.plan(_state(1.0, 2.0), NavGoal(x=-2.0, y=-3.0, yaw=math.pi, z=1.8))

    assert client.requests == [
        {"start": (-1.0, -2.0, 0.35), "end": (2.0, 3.0, 1.8)}
    ]
    assert plan.waypoints == ((1.0, 2.0), (-0.0, -0.0), (-2.0, -3.0))
    assert plan.metadata["planner"] == "pct"
    expected_path_3d = ((1.0, 2.0, 0.35), (-0.0, -0.0, 1.0), (-2.0, -3.0, 1.8))
    for actual, expected in zip(plan.metadata["path_3d"], expected_path_3d, strict=True):
        assert actual == pytest.approx(expected)
    assert plan.metadata["slice_start"] == 1
    assert plan.metadata["slice_end"] == 4
    assert plan.metadata["snap_start_dist"] == 0.02
    assert plan.metadata["snap_end_dist"] == 0.04


def test_pct_nav_planner_uses_robot_z_when_goal_z_is_missing() -> None:
    client = FakePCTClient(
        {
            "status": "ok",
            "traj": [[-1.0, -2.0, 0.6], [-3.0, -4.0, 0.6]],
        }
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=client,
    )

    plan = planner.plan(_state(1.0, 2.0, z=0.6), NavGoal(x=3.0, y=4.0, yaw=0.0))

    assert client.requests[0]["end"] == (-3.0, -4.0, 0.6)
    assert plan.metadata["goal_z_missing"] is True
    assert plan.metadata["goal_z_source"] == "robot_root_pose"


def test_pct_nav_planner_raises_clear_error_without_fallback() -> None:
    class FailingClient:
        def plan(self, *, start, end):
            del start, end
            raise RuntimeError("server unavailable")

    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=FailingClient(),
    )

    with pytest.raises(RuntimeError, match="PCT global planning failed: server unavailable"):
        planner.plan(_state(0.0, 0.0), NavGoal(x=1.0, y=1.0, yaw=0.0, z=0.35))
