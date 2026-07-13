from __future__ import annotations

import math
import json
from typing import Any

import pytest

from source.interfaces import NavGoal, SimulationState
from source.navigation.pct_adapter import PCTNavPlanner, PCTPlannerClient, PCTPlannerConfig


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
    assert plan.metadata["sim_start"] == pytest.approx((1.0, 2.0, 0.35))
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


def test_pct_client_exports_cross_floor_gateway_in_pct_frame() -> None:
    client = PCTPlannerClient(
        PCTPlannerConfig(
            enabled=True,
            cross_floor_gateway_points=((1.5, 5.7, 0.6),),
            cross_floor_gateway_radius_m=1.2,
        )
    )

    env = client._server_env()

    assert json.loads(env["PCT_CROSS_FLOOR_GATEWAYS_PCT"]) == [[-1.5, -5.7, 0.6]]
    assert json.loads(env["PCT_CROSS_FLOOR_STAIR_EXITS_PCT"]) == [
        [-2.7, -7.05, 3.0]
    ]
    assert json.loads(env["PCT_CROSS_FLOOR_STAIR_MIDPOINTS_PCT"]) == [
        [-1.51822, -6.27683, 0.29486],
        [-2.74512, -9.14634, 1.64666],
        [-1.9202, -9.52807, 1.71919],
        [-2.69841, -7.79872, 2.61031],
    ]
    assert env["PCT_CROSS_FLOOR_GATEWAY_RADIUS_M"] == "1.2"
    assert env["PCT_ROBOT_ROOT_TO_FLOOR_M"] == "0.45"
    assert env["PCT_BODY_OBSTACLE_MIN_HEIGHT_M"] == "0.3"
    assert env["PCT_BODY_OBSTACLE_MAX_HEIGHT_M"] == "1.0"
    assert env["PCT_STAIR_VERTICAL_RADIUS_M"] == "0.6"
    assert env["PCT_STAIR_PROGRESS_TOLERANCE"] == "0.35"
    assert env["PCT_STAIR_PROGRESS_COST_WEIGHT"] == "20.0"


def test_pct_nav_planner_replaces_stair_zigzag_with_calibrated_centerline() -> None:
    raw_traj_sim = (
        (-3.4, 6.6, 0.0),
        (0.12149, 4.65642, 0.0),
        (0.52149, 4.65642, 0.0),
        (0.92149, 5.05642, 0.0),
        (1.32149, 5.05642, 0.0),
        (1.52149, 5.25642, 0.0),
        (1.52, 5.86, 0.0),
        (1.72, 7.66, 1.0),
        (1.92, 9.26, 1.5),
        (2.92, 8.66, 2.0),
        (2.92, 7.66, 2.5),
        (2.32, 7.66, 3.0),
        (2.52, 6.66, 3.0),
        (0.4, -0.1, 3.0),
    )
    client = FakePCTClient(
        {
            "status": "ok",
            "cross_floor": True,
            "traj": [[-x, -y, z] for x, y, z in raw_traj_sim],
            "slice_start": 8,
            "slice_end": 15,
        }
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=client,
    )

    plan = planner.plan(
        _state(-3.4, 6.6),
        NavGoal(x=0.4, y=-0.1, z=3.62628, yaw=0.0),
    )

    report = plan.metadata["stair_centerline_refinement"]
    assert report["applied"] is True
    assert report["reason"] == "calibrated_stair_centerline"
    for actual, expected in zip(
        plan.metadata["pct_raw_path_3d"],
        raw_traj_sim,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    anchors = report["centerline_anchors"]
    assert anchors[2][:2] == pytest.approx((1.9202, 9.52807))
    assert anchors[3][:2] == pytest.approx((2.74512, 9.14634))
    assert report["approach_start"] == pytest.approx((0.52149, 4.65642, 0.0))
    refined_path = plan.metadata["path_3d"]
    approach_start_index = next(
        index
        for index, point in enumerate(refined_path)
        if point == pytest.approx((0.52149, 4.65642, 0.0))
    )
    gateway_index = next(
        index
        for index, point in enumerate(refined_path[approach_start_index:])
        if point == pytest.approx((1.5, 5.7, 0.0))
    ) + approach_start_index
    approach = refined_path[approach_start_index : gateway_index + 1]
    assert len(approach) >= 8
    assert max(float(point[0]) for point in approach) <= 1.500001
    headings = [
        math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
        for start, end in zip(approach, approach[1:])
    ]
    heading_steps = [
        abs(math.atan2(math.sin(end - start), math.cos(end - start)))
        for start, end in zip(headings, headings[1:])
    ]
    assert max(heading_steps) < 0.20
    assert anchors[-2][:2] == pytest.approx((2.69841, 7.79872))
    assert anchors[-1][:2] == pytest.approx((2.70, 7.05))
    upper_flight = [
        point
        for point in plan.metadata["path_3d"]
        if 2.60 <= float(point[2]) <= 3.01
        and 7.0 <= float(point[1]) <= 7.85
    ]
    assert len(upper_flight) >= 4
    assert max(abs(float(point[0]) - 2.70) for point in upper_flight) < 0.01
    assert all(
        float(next_point[1]) <= float(point[1]) + 1.0e-9
        for point, next_point in zip(upper_flight, upper_flight[1:])
    )


def test_pct_stair_refinement_accepts_gateway_as_first_path_point() -> None:
    raw_traj_sim = (
        (1.5, 5.7, 0.0),
        (1.52, 6.3, 0.3),
        (1.92, 9.5, 1.7),
        (2.74, 9.14, 1.7),
        (2.70, 7.8, 2.6),
        (2.70, 7.05, 3.0),
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=FakePCTClient(
            {
                "status": "ok",
                "cross_floor": True,
                "traj": [[-x, -y, z] for x, y, z in raw_traj_sim],
            }
        ),
    )

    plan = planner.plan(
        _state(1.5, 5.7, z=0.36742),
        NavGoal(x=2.70, y=7.05, z=3.62628, yaw=-math.pi / 2.0),
    )

    report = plan.metadata["stair_centerline_refinement"]
    assert report["applied"] is True
    assert report["raw_start_index"] == 0
    assert report["approach_start_index"] == 0
    assert plan.metadata["path_3d"][0] == pytest.approx((1.5, 5.7, 0.0))
    assert report["centerline_anchors"][2][:2] == pytest.approx(
        (1.9202, 9.52807)
    )
    assert report["centerline_anchors"][3][:2] == pytest.approx(
        (2.74512, 9.14634)
    )


def test_pct_stair_short_gateway_approach_does_not_overshoot() -> None:
    snapped_start = (1.5214885711669925, 5.656423950195313, 0.0)
    gateway = (1.5, 5.7, 0.0)
    raw_traj_sim = (
        snapped_start,
        (1.52, 6.3, 0.3),
        (1.92, 9.5, 1.7),
        (2.74, 9.14, 1.7),
        (2.70, 7.8, 2.6),
        (2.70, 7.05, 3.0),
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=FakePCTClient(
            {
                "status": "ok",
                "cross_floor": True,
                "traj": [[-x, -y, z] for x, y, z in raw_traj_sim],
            }
        ),
    )

    plan = planner.plan(
        _state(1.5, 5.7, z=0.36742),
        NavGoal(x=2.70, y=7.05, z=3.62628, yaw=-math.pi / 2.0),
    )

    report = plan.metadata["stair_centerline_refinement"]
    approach = plan.metadata["path_3d"][: report["approach_point_count"]]
    assert approach[0] == pytest.approx(snapped_start)
    assert approach[-1] == pytest.approx(gateway)
    assert all(
        min(snapped_start[0], gateway[0]) - 1.0e-9
        <= float(point[0])
        <= max(snapped_start[0], gateway[0]) + 1.0e-9
        for point in approach
    )
    assert all(
        min(snapped_start[1], gateway[1]) - 1.0e-9
        <= float(point[1])
        <= max(snapped_start[1], gateway[1]) + 1.0e-9
        for point in approach
    )


def test_pct_stair_refinement_preserves_post_exit_extension() -> None:
    stair_exit = (2.70, 7.05, 3.0)
    terminal = (2.702, 6.05, 3.0)
    raw_traj_sim = (
        (1.5, 5.7, 0.0),
        (1.52, 6.3, 0.3),
        (1.92, 9.5, 1.7),
        (2.74, 9.14, 1.7),
        (2.70, 7.8, 2.6),
        stair_exit,
        terminal,
    )
    planner = PCTNavPlanner(
        PCTPlannerConfig(enabled=True, fallback_to_astar=False),
        client=FakePCTClient(
            {
                "status": "ok",
                "cross_floor": True,
                "traj": [[-x, -y, z] for x, y, z in raw_traj_sim],
            }
        ),
    )

    plan = planner.plan(
        _state(1.5, 5.7, z=0.36742),
        NavGoal(x=terminal[0], y=terminal[1], z=3.62628, yaw=-math.pi / 2.0),
    )

    path = plan.metadata["path_3d"]
    exit_index = next(
        index for index, point in enumerate(path) if point == pytest.approx(stair_exit)
    )
    assert exit_index < len(path) - 1
    assert path[-1] == pytest.approx(terminal)
    assert float(path[-1][1]) < float(path[exit_index][1])


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
