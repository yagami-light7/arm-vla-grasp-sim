from __future__ import annotations

import math

import numpy as np
import pytest

from source.interfaces import NavGoal, NavPlan, SimulationState
from source.navigation.executor import DwaNavExecutor
from source.navigation.navlib import DWAConfig, DWAController, OccupancyGridMap
from source.navigation.navlib.path_refinement import (
    LocalPathRefinementError,
    first_blocked_path_segment,
    refine_same_floor_path,
    world_segment_clearance,
)


def _open_grid(
    *,
    resolution: float = 0.2,
    origin: tuple[float, float, float] = (-2.0, -2.0, 0.0),
    shape: tuple[int, int] = (40, 40),
) -> OccupancyGridMap:
    return OccupancyGridMap(
        np.zeros(shape, dtype=bool),
        resolution,
        origin,
    )


def _state(
    x: float,
    y: float,
    yaw: float,
    *,
    step: int = 0,
    velocity: tuple[float, float, float, float, float, float] = (0.0,) * 6,
) -> SimulationState:
    return SimulationState(
        step_index=step,
        timestamp=0.02 * step,
        robot_root_pose=(
            x,
            y,
            0.3,
            math.cos(0.5 * yaw),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
        ),
        robot_root_velocity=velocity,
    )


def _carry_departure_config(
    *,
    support_center: tuple[float, float] = (0.0, 1.0),
) -> dict[str, object]:
    return {
        "enabled": True,
        "source_support_id": "support",
        "source_support_prim_path": "/World/support",
        "source_support_center_xy": support_center,
        "source_support_half_diagonal_m": 0.28,
        "required_center_clearance_m": 0.85,
        "clearance_formula": "test",
        "activation_heading_error_rad": math.pi / 4.0,
        "minimum_reverse_distance_m": 0.20,
        "maximum_reverse_distance_m": 0.36,
        "reverse_speed_mps": 0.25,
        "yaw_hold_kp": 1.2,
        "max_yaw_rate_rps": 0.1,
        "minimum_backward_alignment_cosine": 0.7,
        "completion_distance_tolerance_m": 0.02,
        "settle_linear_velocity_mps": 0.08,
        "settle_angular_velocity_rps": 0.2,
        "settle_required_stable_steps": 5,
        "settle_max_steps": 60,
        "max_steps": 250,
    }


def test_exact_live_start_replaces_map_dependent_pct_cell_center() -> None:
    grid = _open_grid(
        resolution=0.2,
        origin=(-34.41524505615235, -30.475133705139164, 0.0),
        shape=(443, 275),
    )
    live_start = (-0.6187453269958496, 5.3233962059021)
    snapped_start = grid.grid_to_world(*grid.world_to_grid(*live_start))
    exact_goal = (-0.6047087424253801, 6.157497863251304)

    result = refine_same_floor_path(
        grid_map=grid,
        global_path_world=(snapped_start, grid.grid_to_world(*grid.world_to_grid(*exact_goal))),
        live_start_xy=live_start,
        exact_goal_xy=exact_goal,
    )

    assert result.report["mode"] == "pct_same_floor_direct"
    np.testing.assert_allclose(result.path_world, (live_start, exact_goal))
    assert result.report["global_start_offset_m"] > 0.10
    assert result.report["first_segment_heading_error_to_goal_rad"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "origin",
    [
        (-2.0, -2.0, 0.0),
        (-2.17, -2.09, 0.0),
        (-3.1, -1.7, 0.31),
        (-1.4, -3.2, -0.47),
    ],
)
def test_open_space_controller_path_is_invariant_to_grid_origin(
    origin: tuple[float, float, float],
) -> None:
    grid = _open_grid(origin=origin, shape=(80, 80))
    # Select points through the map transform so rotated-map fixtures remain in bounds.
    start_center = grid.grid_to_world(55, 20)
    goal_center = grid.grid_to_world(25, 45)
    live_start = (start_center[0] + 0.071, start_center[1] - 0.043)
    exact_goal = (goal_center[0] - 0.037, goal_center[1] + 0.052)
    snapped_start = grid.grid_to_world(*grid.world_to_grid(*live_start))

    result = refine_same_floor_path(
        grid_map=grid,
        global_path_world=(snapped_start, goal_center),
        live_start_xy=live_start,
        exact_goal_xy=exact_goal,
    )

    np.testing.assert_allclose(result.path_world, (live_start, exact_goal))
    assert result.report["turn_count"] == 0


def test_low_clearance_terminal_approach_is_not_rejected_by_hidden_threshold() -> None:
    occupancy = np.zeros((30, 30), dtype=bool)
    grid = OccupancyGridMap(occupancy, 0.2, (-3.0, -3.0, 0.0))
    start = (0.1, -1.5)
    goal = (0.1, 1.1)
    # Put an obstacle one cell beside the terminal approach.  The route is free
    # with 0.2 m map clearance, which the previous fixed 0.4 m gate rejected.
    goal_row, goal_col = grid.world_to_grid(*goal)
    occupancy[goal_row, goal_col + 1] = True
    grid = OccupancyGridMap(occupancy, 0.2, (-3.0, -3.0, 0.0))

    direct_free, clearance = world_segment_clearance(start, goal, grid)
    result = refine_same_floor_path(
        grid_map=grid,
        global_path_world=(start, goal),
        live_start_xy=start,
        exact_goal_xy=goal,
    )

    assert direct_free is True
    assert clearance == pytest.approx(0.2)
    assert result.report["mode"] == "pct_same_floor_direct"
    assert result.report["hidden_clearance_threshold_m"] is None


def test_blocked_direct_route_uses_exact_endpoint_astar_and_string_pulling() -> None:
    occupancy = np.zeros((40, 40), dtype=bool)
    occupancy[:, 20] = True
    occupancy[24:28, 20] = False
    grid = OccupancyGridMap(occupancy, 0.1, (-2.0, -2.0, 0.0))
    start = (-1.35, 0.55)
    goal = (1.35, 0.55)

    result = refine_same_floor_path(
        grid_map=grid,
        global_path_world=(grid.grid_to_world(*grid.world_to_grid(*start)), goal),
        live_start_xy=start,
        exact_goal_xy=goal,
    )

    assert result.report["mode"] == "pct_same_floor_local_astar"
    assert result.path_world[0] == pytest.approx(start)
    assert result.path_world[-1] == pytest.approx(goal)
    assert result.report["line_of_sight_waypoints"] <= result.report["anchored_waypoints"]
    assert first_blocked_path_segment(result.path_world, result.grid_map) is None


def test_supercover_rejects_diagonal_corner_cutting() -> None:
    occupancy = np.zeros((4, 4), dtype=bool)
    occupancy[3, 1] = True
    occupancy[2, 0] = True
    grid = OccupancyGridMap(occupancy, 1.0, (0.0, 0.0, 0.0))
    start = grid.grid_to_world(3, 0)
    end = grid.grid_to_world(2, 1)

    free, clearance = world_segment_clearance(start, end, grid)

    assert free is False
    assert clearance == 0.0


def test_only_live_robot_cell_is_cleared_for_start_recovery() -> None:
    occupancy = np.zeros((20, 20), dtype=bool)
    grid = OccupancyGridMap(occupancy, 0.2, (-2.0, -2.0, 0.0))
    start = (-0.31, -0.27)
    goal = (1.2, 1.1)
    start_rc = grid.world_to_grid(*start)
    occupancy[start_rc] = True
    grid = OccupancyGridMap(occupancy, 0.2, (-2.0, -2.0, 0.0))

    result = refine_same_floor_path(
        grid_map=grid,
        global_path_world=(start, goal),
        live_start_xy=start,
        exact_goal_xy=goal,
    )

    assert result.report["start_cell_recovered"] is True
    assert result.grid_map.is_occupied(*start_rc) is False
    assert int(np.count_nonzero(grid.occupancy ^ result.grid_map.occupancy)) == 1


def test_occupied_exact_goal_is_rejected_instead_of_silently_snapped() -> None:
    occupancy = np.zeros((20, 20), dtype=bool)
    grid = OccupancyGridMap(occupancy, 0.2, (-2.0, -2.0, 0.0))
    start = (-1.0, -1.0)
    goal = (1.0, 1.0)
    occupancy[grid.world_to_grid(*goal)] = True
    grid = OccupancyGridMap(occupancy, 0.2, (-2.0, -2.0, 0.0))

    with pytest.raises(LocalPathRefinementError) as exc_info:
        refine_same_floor_path(
            grid_map=grid,
            global_path_world=(start, goal),
            live_start_xy=start,
            exact_goal_xy=goal,
        )

    assert exc_info.value.report["reason"] == "exact_goal_occupied"


def test_executor_uses_sim_start_and_drops_pct_snapped_start_detour() -> None:
    grid = _open_grid()
    live_start = (-0.63, -0.21)
    snapped_start = grid.grid_to_world(*grid.world_to_grid(*live_start))
    goal = (1.4, 1.2)
    plan = NavPlan(
        goal=NavGoal(x=goal[0], y=goal[1], yaw=0.0),
        waypoints=(snapped_start, grid.grid_to_world(*grid.world_to_grid(*goal))),
        metadata={"planner": "pct", "sim_start": (*live_start, 0.3)},
    )
    executor = DwaNavExecutor(
        grid_map=grid,
        dwa_config=DWAConfig(control_dt=0.02, lookahead_distance=0.12),
    )

    executor.reset(plan)

    assert executor._controller is not None
    np.testing.assert_allclose(
        executor._controller.reference_path_world,
        [live_start, goal],
    )
    report = executor.status()["local_refinement"]
    assert report["global_start_offset_m"] > 0.0
    assert report["turn_count"] == 0


def test_multifloor_executor_preserves_route_but_anchors_exact_live_start() -> None:
    grid = _open_grid(shape=(80, 80))
    live_start = (-0.63, -0.21)
    snapped_start = grid.grid_to_world(*grid.world_to_grid(*live_start))
    middle = (0.2, 0.6)
    goal = (1.4, 1.2)
    plan = NavPlan(
        goal=NavGoal(x=goal[0], y=goal[1], yaw=0.0),
        waypoints=(snapped_start, middle, goal),
        metadata={
            "planner": "pct",
            "sim_start": (*live_start, 0.3),
            "path_3d": (
                (*snapped_start, 0.3),
                (*middle, 1.2),
                (*goal, 1.2),
            ),
            "slice_start": 1,
            "slice_end": 6,
        },
    )
    executor = DwaNavExecutor(
        grid_map=grid,
        multifloor_grid_map=grid,
        dwa_config=DWAConfig(control_dt=0.02, lookahead_distance=0.12),
    )

    executor.reset(plan)

    assert executor._controller is not None
    np.testing.assert_allclose(
        executor._controller.reference_path_world,
        [live_start, middle, goal],
    )
    report = executor.status()["local_refinement"]
    assert report["mode"] == "pct_multifloor_path_preserved"
    assert report["global_start_offset_m"] > 0.0


def test_dwa_first_live_pose_projects_forward_on_stale_path() -> None:
    grid = _open_grid(resolution=0.1)
    controller = DWAController(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        grid,
        DWAConfig(control_dt=0.05, lookahead_distance=0.3),
    )

    _command, debug = controller.compute_command(
        (1.6, 0.03, 0.0),
        (0.0, 0.0),
    )

    assert debug.path_anchor_applied is True
    assert debug.path_anchor_progress_m == pytest.approx(1.6, abs=0.02)
    assert debug.path_anchor_distance == pytest.approx(0.03, abs=1.0e-6)
    assert debug.target_point[0] > 1.6


@pytest.mark.parametrize("yaw", np.linspace(-math.pi, math.pi, 17))
def test_random_initial_yaw_does_not_change_direct_path_geometry(yaw: float) -> None:
    grid = _open_grid(resolution=0.1)
    start = (-0.4, -0.2)
    goal = (1.6, 0.9)
    controller = DWAController(
        [start, goal],
        grid,
        DWAConfig(control_dt=0.02, lookahead_distance=0.12),
    )

    _command, debug = controller.compute_command((*start, float(yaw)), (0.0, 0.0))
    expected_heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    expected_error = (expected_heading - yaw + math.pi) % (2.0 * math.pi) - math.pi

    assert debug.heading_error == pytest.approx(expected_error)
    assert debug.target_point[0] > start[0]


def test_rotation_gate_uses_hysteresis_and_zero_creep() -> None:
    grid = _open_grid(resolution=0.1)
    controller = DWAController(
        [(0.0, 0.0), (2.0, 0.0)],
        grid,
        DWAConfig(
            control_dt=0.05,
            lookahead_distance=0.2,
            rotate_in_place_angle=0.45,
            rotate_in_place_exit_angle=0.20,
            rotate_in_place_settle_angular_velocity=0.12,
            large_heading_creep_velocity=0.0,
            max_angular_velocity=0.5,
        ),
    )

    command, debug = controller.compute_command((0.0, 0.0, math.pi), (0.0, 0.0))
    assert debug.rotation_gate_active is True
    assert command[0] == pytest.approx(0.0)

    command, debug = controller.compute_command((0.0, 0.0, 0.30), (0.0, 0.0))
    assert debug.rotation_gate_active is True
    assert command[0] == pytest.approx(0.0)

    command, debug = controller.compute_command((0.0, 0.0, 0.15), (0.0, 0.30))
    assert debug.rotation_gate_active is False
    assert debug.rotation_settle_active is True
    assert command == pytest.approx((0.0, 0.0, 0.0))
    assert debug.rotate_in_place_exit_angle_used == pytest.approx(0.20)

    _command, debug = controller.compute_command((0.0, 0.0, 0.15), (0.0, 0.05))
    assert debug.rotation_settle_active is False

    _command, debug = controller.compute_command((0.0, 0.0, 0.30), (0.0, 0.0))
    assert debug.rotation_gate_active is False

    command, debug = controller.compute_command((0.0, 0.0, 0.60), (0.0, 0.0))
    assert debug.rotation_gate_active is True
    assert command[0] == pytest.approx(0.0)


def test_moving_path_tracking_keeps_small_angular_corrections() -> None:
    grid = _open_grid(resolution=0.1)
    controller = DWAController(
        [(0.0, 0.0), (2.0, 0.0)],
        grid,
        DWAConfig(
            control_dt=0.02,
            max_angular_velocity=0.5,
            min_active_angular_velocity=0.3,
            enforce_min_active_angular_velocity=True,
            enforce_min_active_angular_velocity_only_during_rotation=True,
        ),
    )

    samples = controller._sample_velocities(
        current_vx=0.25,
        current_wz=0.0,
        distance_to_goal=1.5,
        heading_error=0.1,
        rotation_gate_active=False,
    )
    angular_values = {abs(wz) for _, wz in samples if abs(wz) > 1.0e-6}

    assert angular_values
    assert min(angular_values) < 0.3


def test_same_floor_carry_reverses_outside_support_then_turns_in_place() -> None:
    grid = _open_grid(
        resolution=0.05,
        origin=(-3.0, -3.0, 0.0),
        shape=(140, 140),
    )
    start = (0.0, 0.4)
    goal = (0.0, -2.0)
    plan = NavPlan(
        goal=NavGoal(x=goal[0], y=goal[1], yaw=-math.pi / 2.0),
        waypoints=(start, goal),
        metadata={
            "planner": "pct",
            "sim_start": (*start, 0.3),
            "execution_phase": "carry_nav_to_place",
            "navigation_execution": {
                "same_floor_alignment": {
                    "rotate_in_place_enter_angle_rad": 0.45,
                    "rotate_in_place_exit_angle_rad": 0.20,
                    "large_heading_creep_velocity_mps": 0.0,
                }
            },
            "carry_departure": _carry_departure_config(),
        },
    )
    executor = DwaNavExecutor(
        grid_map=grid,
        carry_grid_map=grid,
        dwa_config=DWAConfig(
            control_dt=0.02,
            lookahead_distance=0.12,
            use_command_velocity_window=True,
            enforce_min_active_linear_velocity=True,
            min_active_linear_velocity=0.25,
            max_angular_velocity=0.5,
        ),
        carry_max_linear_velocity=0.25,
        carry_max_angular_velocity=0.30,
        carry_initial_alignment_path_deviation_limit=0.40,
    )
    executor.reset(plan)

    first = executor.compute_action(_state(*start, math.pi / 2.0, step=0))
    assert first.source == "navigation_carry_departure"
    assert first.base_velocity == pytest.approx((-0.25, 0.0, 0.0))
    assert executor.status()["carry_departure"]["planned_reverse_distance_m"] \
        == pytest.approx(0.25, abs=0.011)

    departed = (0.0, 0.14)
    stop = executor.compute_action(
        _state(*departed, math.pi / 2.0, step=1)
    )
    assert stop.source == "navigation_carry_departure_settling"
    assert stop.base_velocity == pytest.approx((0.0, 0.0, 0.0))

    for step in range(2, 7):
        settle = executor.compute_action(
            _state(*departed, math.pi / 2.0, step=step)
        )
    assert settle.source == "navigation_carry_departure_complete"
    status = executor.status()
    assert status["carry_departure"]["completed"] is True
    assert status["local_refinement"]["reanchored_after_carry_departure"] is True
    assert status["local_refinement"]["live_start_xy"] == pytest.approx(departed)

    turn = executor.compute_action(
        _state(*departed, math.pi / 2.0, step=7)
    )
    status = executor.status()
    assert turn.source == "navigation_dwa"
    assert turn.base_velocity[0] == pytest.approx(0.0)
    assert status["dwa"]["rotation_gate_active"] is True
    assert status["dwa_limits"]["rotate_in_place_angle"] == pytest.approx(0.45)
    assert status["dwa_limits"]["rotate_in_place_exit_angle"] == pytest.approx(0.20)


def test_carry_departure_fails_before_turn_when_reverse_segment_is_blocked() -> None:
    occupancy = np.zeros((140, 140), dtype=bool)
    grid = OccupancyGridMap(occupancy, 0.05, (-3.0, -3.0, 0.0))
    blocked_row, blocked_col = grid.world_to_grid(0.0, 0.22)
    occupancy[blocked_row, blocked_col] = True
    grid = OccupancyGridMap(occupancy, 0.05, (-3.0, -3.0, 0.0))
    start = (0.0, 0.4)
    goal = (2.0, 0.4)
    plan = NavPlan(
        goal=NavGoal(x=goal[0], y=goal[1], yaw=0.0),
        waypoints=(start, goal),
        metadata={
            "planner": "pct",
            "sim_start": (*start, 0.3),
            "execution_phase": "carry_nav_to_place",
            "carry_departure": _carry_departure_config(),
        },
    )
    executor = DwaNavExecutor(
        grid_map=grid,
        carry_grid_map=grid,
        dwa_config=DWAConfig(control_dt=0.02, lookahead_distance=0.12),
    )
    executor.reset(plan)

    action = executor.compute_action(_state(*start, math.pi / 2.0))
    status = executor.status()

    assert action.base_velocity == pytest.approx((0.0, 0.0, 0.0))
    assert status["failed"] is True
    assert status["failure_reason"] == "carry_departure_reverse_segment_blocked"
