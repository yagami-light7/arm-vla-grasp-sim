"""Pure-Python global-planning and closed-loop DWA smoke tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import SimulationState
from source.navigation import NavPlanner
from source.navigation.executor import (
    DwaNavExecutor,
    PCT_STAIR_FLOAT_DOG_JOINT_NAMES,
    PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS,
    _plan_clearance_optimized_floor_path,
    _world_path_is_clear,
)
from source.navigation.navlib import (
    AStarPlanner,
    DWAConfig,
    DWAController,
    DWADebug,
    OccupancyGridMap,
)


class AStarDwaSmokeTest(unittest.TestCase):
    def test_astar_routes_through_gap(self) -> None:
        occupancy = np.zeros((20, 20), dtype=bool)
        occupancy[:, 10] = True
        occupancy[9:12, 10] = False
        grid = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0))
        plan = AStarPlanner().plan(grid, (0.25, 0.95), (1.75, 0.95))
        self.assertGreater(len(plan.raw_path_world), 2)
        self.assertTrue(any(col == 10 and 9 <= row <= 11 for row, col in plan.raw_path_grid))

    def test_dwa_closed_loop_reaches_goal_and_keeps_vy_zero(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        path = [(0.0, 0.0), (1.2, 0.0)]
        config = DWAConfig(control_dt=0.05, goal_tolerance=0.12)
        controller = DWAController(path, grid, config)
        pose = np.array([0.0, 0.0, 0.0])
        velocity = (0.0, 0.0)
        reached = False
        for _ in range(600):
            command, debug = controller.compute_command(tuple(pose), velocity)
            self.assertEqual(float(command[1]), 0.0)
            velocity = (float(command[0]), float(command[2]))
            pose[0] += velocity[0] * math.cos(pose[2]) * config.control_dt
            pose[1] += velocity[0] * math.sin(pose[2]) * config.control_dt
            pose[2] += velocity[1] * config.control_dt
            if debug.reached_goal:
                reached = True
                break
        self.assertTrue(reached)

    def test_dwa_compensates_measured_lateral_slip_when_enabled(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.10,
            (0.0, 0.0, 0.0),
        )
        controller = DWAController(
            [(0.5, 2.0), (3.0, 2.0)],
            grid,
            DWAConfig(
                control_dt=0.10,
                max_linear_accel=10.0,
                lateral_velocity_compensation_gain=1.0,
                max_lateral_velocity_command=0.12,
            ),
        )

        command, debug = controller.compute_command(
            (0.5, 2.0, 0.0),
            (0.20, -0.10, 0.0),
        )

        self.assertGreater(debug.feasible_candidates, 0)
        self.assertAlmostEqual(float(command[1]), 0.10, places=5)

    def test_dwa_lateral_rollout_respects_acceleration_limited_response(
        self,
    ) -> None:
        grid = OccupancyGridMap(
            np.zeros((20, 20), dtype=bool),
            0.10,
            (0.0, 0.0, 0.0),
        )
        controller = DWAController(
            [(0.5, 1.0), (1.5, 1.0)],
            grid,
            DWAConfig(control_dt=0.10, integration_dt=0.10),
        )

        trajectory = controller._rollout(
            x=0.5,
            y=1.0,
            yaw=0.0,
            linear_velocity=0.20,
            angular_velocity=0.0,
            lateral_velocity=-0.10,
            lateral_velocity_target=0.10,
            max_lateral_accel=0.50,
        )

        self.assertLess(float(trajectory[0][1]), 1.0)
        self.assertGreater(float(trajectory[-1][1]), float(trajectory[0][1]))

    def test_dwa_occupied_start_escape_only_leaves_footprint_inflation(self) -> None:
        occupancy = np.zeros((20, 20), dtype=bool)
        occupancy[:, 4] = True
        raw_map = OccupancyGridMap(occupancy, 0.10, (0.0, 0.0, 0.0))
        footprint_map = raw_map.inflate(0.20)
        start = raw_map.grid_to_world(10, 6)
        goal = raw_map.grid_to_world(10, 14)
        controller = DWAController(
            [start, goal],
            footprint_map,
            DWAConfig(
                control_dt=0.10,
                prediction_horizon=0.80,
                integration_dt=0.10,
                max_linear_accel=10.0,
                allow_inflated_occupied_start_escape=True,
            ),
            raw_grid_map=raw_map,
        )

        command, debug = controller.compute_command(
            (start[0], start[1], 0.0),
            (0.0, 0.0, 0.0),
        )

        self.assertTrue(footprint_map.is_occupied(*footprint_map.world_to_grid(*start)))
        self.assertFalse(raw_map.is_occupied(*raw_map.world_to_grid(*start)))
        self.assertTrue(debug.occupied_start_escape_active)
        self.assertGreater(debug.occupied_start_escape_candidates, 0)
        self.assertGreater(float(command[0]), 0.0)

    def test_dwa_occupied_start_escape_rejects_raw_obstacle(self) -> None:
        occupancy = np.zeros((20, 20), dtype=bool)
        occupancy[:, 4] = True
        raw_map = OccupancyGridMap(occupancy, 0.10, (0.0, 0.0, 0.0))
        footprint_map = raw_map.inflate(0.20)
        start = raw_map.grid_to_world(10, 4)
        goal = raw_map.grid_to_world(10, 14)
        controller = DWAController(
            [start, goal],
            footprint_map,
            DWAConfig(
                control_dt=0.10,
                max_linear_accel=10.0,
                allow_inflated_occupied_start_escape=True,
            ),
            raw_grid_map=raw_map,
        )

        command, debug = controller.compute_command(
            (start[0], start[1], 0.0),
            (0.0, 0.0, 0.0),
        )

        self.assertFalse(debug.occupied_start_escape_active)
        self.assertEqual(debug.feasible_candidates, 0)
        self.assertAlmostEqual(float(command[0]), 0.0)

    def test_dwa_raw_clearance_changes_continuously_inside_one_cell(self) -> None:
        occupancy = np.zeros((20, 20), dtype=bool)
        occupancy[:, 4] = True
        raw_map = OccupancyGridMap(occupancy, 0.10, (0.0, 0.0, 0.0))
        controller = DWAController(
            [(0.65, 1.0), (1.5, 1.0)],
            raw_map.inflate(0.20),
            DWAConfig(control_dt=0.10),
            raw_grid_map=raw_map,
        )

        near_wall = controller._raw_map_clearance_at(0.61, 1.0)
        far_from_wall = controller._raw_map_clearance_at(0.69, 1.0)

        self.assertIsNotNone(near_wall)
        self.assertIsNotNone(far_from_wall)
        self.assertLess(float(near_wall), float(far_from_wall))

    def test_dwa_accepts_approach_command_when_obstacle_is_behind_goal(self) -> None:
        occupancy = np.zeros((80, 80), dtype=bool)
        grid = OccupancyGridMap(occupancy, 0.05, (-1.0, -1.0, 0.0))
        for y in np.arange(-0.5, 0.51, 0.05):
            occupancy[grid.world_to_grid(1.55, float(y))] = True

        config = DWAConfig(control_dt=0.05, goal_tolerance=0.15, prediction_horizon=1.8)
        controller = DWAController([(0.0, 0.0), (1.4, 0.0)], grid, config)
        command, debug = controller.compute_command((0.9, 0.0, 0.0), (0.3, 0.0))

        self.assertGreaterEqual(float(command[0]), 0.30)
        self.assertEqual(debug.collision_rejections, 0)

    def test_dwa_respects_close_goal_speed_limit(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.05,
            goal_tolerance=0.05,
            close_goal_distance=0.45,
            close_goal_speed_limit=0.20,
            max_linear_velocity=0.70,
            max_linear_accel=10.0,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)
        command, debug = controller.compute_command((0.70, 0.0, 0.0), (0.60, 0.0))

        self.assertFalse(debug.reached_goal)
        self.assertLessEqual(float(command[0]), 0.2001)

    def test_dwa_velocity_samples_stay_inside_dynamic_window(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            max_angular_velocity=0.50,
            max_linear_accel=2.5,
            max_angular_accel=3.0,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        samples = controller._sample_velocities(
            current_vx=0.20,
            current_wz=0.10,
            distance_to_goal=1.0,
            heading_error=0.0,
        )

        self.assertTrue(samples)
        self.assertTrue(all(0.15 - 1.0e-6 <= vx <= 0.25 + 1.0e-6 for vx, _ in samples))
        self.assertTrue(all(0.04 - 1.0e-6 <= wz <= 0.16 + 1.0e-6 for _, wz in samples))

    def test_dwa_large_heading_error_keeps_creeping_turn_candidates(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            min_active_linear_velocity=0.25,
            max_linear_accel=2.5,
            rotate_in_place_angle=0.45,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        samples = controller._sample_velocities(
            current_vx=0.0,
            current_wz=0.0,
            distance_to_goal=1.0,
            heading_error=1.2,
        )
        linear_values = {vx for vx, _ in samples}

        self.assertIn(0.0, linear_values)
        self.assertTrue(any(vx > 0.0 for vx in linear_values))

    def test_dwa_large_heading_creep_can_be_disabled(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            min_active_linear_velocity=0.25,
            max_linear_accel=2.5,
            rotate_in_place_angle=0.45,
            large_heading_creep_velocity=0.0,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        samples = controller._sample_velocities(
            current_vx=0.0,
            current_wz=0.0,
            distance_to_goal=1.0,
            heading_error=1.2,
        )
        linear_values = {vx for vx, _ in samples}

        self.assertEqual(linear_values, {0.0})

    def test_dwa_path_recovery_caps_forward_speed(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            min_active_linear_velocity=0.25,
            max_linear_accel=2.5,
            path_recovery_speed_limit=0.12,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)
        controller._path_recovery_active = True

        samples = controller._sample_velocities(
            current_vx=0.25,
            current_wz=0.0,
            distance_to_goal=1.0,
            heading_error=0.0,
        )

        self.assertTrue(samples)
        self.assertTrue(all(vx <= 0.12 + 1.0e-6 for vx, _ in samples))

    def test_dwa_dynamic_window_prevents_instant_angular_sign_flip(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_angular_velocity=0.50,
            max_angular_accel=3.0,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        samples = controller._sample_velocities(
            current_vx=0.20,
            current_wz=0.50,
            distance_to_goal=1.0,
            heading_error=-0.5,
        )

        self.assertGreaterEqual(min(wz for _, wz in samples), 0.44 - 1.0e-6)

    def test_dwa_out_of_range_measured_velocity_saturates_to_legal_commands(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            min_linear_velocity=0.0,
            max_linear_velocity=0.45,
            max_angular_velocity=0.50,
            max_linear_accel=2.5,
            max_angular_accel=3.0,
        )
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        negative_samples = controller._sample_velocities(
            current_vx=-0.30,
            current_wz=-1.10,
            distance_to_goal=1.0,
            heading_error=0.0,
        )
        positive_samples = controller._sample_velocities(
            current_vx=0.90,
            current_wz=1.10,
            distance_to_goal=1.0,
            heading_error=0.0,
        )

        self.assertTrue(all(vx >= 0.0 and wz >= -0.50 for vx, wz in negative_samples))
        self.assertTrue(all(vx <= 0.45 and wz <= 0.50 for vx, wz in positive_samples))
        self.assertEqual({vx for vx, _ in negative_samples}, {0.0})
        self.assertEqual({wz for _, wz in negative_samples}, {-0.5})
        self.assertEqual({vx for vx, _ in positive_samples}, {0.45})
        self.assertEqual({wz for _, wz in positive_samples}, {0.5})

    def test_dwa_command_window_ramps_through_policy_velocity_deadzone(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            min_active_linear_velocity=0.25,
            max_linear_accel=2.5,
            speed_bias=1.0,
            use_command_velocity_window=True,
        )
        controller = DWAController([(0.0, 0.0), (2.0, 0.0)], grid, config)

        commands = [
            float(controller.compute_command((0.0, 0.0, 0.0), (0.0, 0.0))[0][0])
            for _ in range(8)
        ]

        self.assertLessEqual(commands[0], 0.05 + 1.0e-6)
        self.assertGreater(commands[-1], 0.25)
        self.assertTrue(
            all(next_value >= value for value, next_value in zip(commands, commands[1:]))
        )

    def test_dwa_measured_window_default_does_not_accumulate_commands(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_linear_velocity=0.45,
            min_active_linear_velocity=0.25,
            max_linear_accel=2.5,
            speed_bias=1.0,
        )
        controller = DWAController([(0.0, 0.0), (2.0, 0.0)], grid, config)

        commands = [
            float(controller.compute_command((0.0, 0.0, 0.0), (0.0, 0.0))[0][0])
            for _ in range(4)
        ]

        self.assertEqual(commands, [commands[0]] * len(commands))

    def test_dwa_collision_fallback_respects_angular_dynamic_window(self) -> None:
        occupancy = np.ones((20, 20), dtype=bool)
        grid = OccupancyGridMap(occupancy, 0.1, (-1.0, -1.0, 0.0))
        config = DWAConfig(
            control_dt=0.02,
            max_angular_velocity=0.50,
            max_angular_accel=3.0,
        )
        controller = DWAController([(0.0, 0.0), (0.0, -1.0)], grid, config)

        command, debug = controller.compute_command(
            (0.0, 0.0, 0.0),
            (0.20, 0.50),
        )

        self.assertEqual(debug.feasible_candidates, 0)
        self.assertGreaterEqual(float(command[2]), 0.44 - 1.0e-6)
        self.assertLessEqual(float(command[2]), 0.50 + 1.0e-6)

    def test_dwa_rollout_uses_configured_integration_dt(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 80), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        config = DWAConfig(control_dt=0.05, integration_dt=0.10, prediction_horizon=0.20)
        controller = DWAController([(0.0, 0.0), (1.0, 0.0)], grid, config)

        trajectory = controller._rollout(x=0.0, y=0.0, yaw=0.0, linear_velocity=0.5, angular_velocity=0.0)

        self.assertEqual(len(trajectory), 2)
        self.assertAlmostEqual(float(trajectory[0][0]), 0.05)

    def test_path_distance_window_limits_scoring_work(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 200), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        path = [(index * 0.05, 0.0) for index in range(120)]
        config = DWAConfig(control_dt=0.05, path_distance_window=10)
        controller = DWAController(path, grid, config)
        controller.target_index = 40
        target_point = controller.path_world[controller.target_index]

        distances = controller._path_distances(
            np.array([[target_point[0], target_point[1]], [target_point[0] + 0.45, target_point[1]]], dtype=np.float64)
        )

        self.assertEqual(distances.shape, (2,))
        self.assertLess(float(distances[0]), 0.1)

    def test_dwa_target_index_never_moves_backward(self) -> None:
        grid = OccupancyGridMap(np.zeros((80, 120), dtype=bool), 0.05, (-1.0, -1.0, 0.0))
        path = [(index * 0.05, 0.0) for index in range(80)]
        config = DWAConfig(control_dt=0.05, path_distance_window=10)
        controller = DWAController(path, grid, config)
        controller.target_index = 40

        _, debug = controller.compute_command((0.9, 0.18, 0.0), (0.0, 0.0))

        self.assertGreaterEqual(controller.target_index, 40)
        self.assertGreaterEqual(debug.target_index, 40)

    def test_dwa_preserves_sharp_corner_and_turns_before_next_segment(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.50,
                rotate_in_place_angle=0.45,
                preserve_sharp_corners=True,
                corner_angle_threshold=0.35,
                corner_waypoint_tolerance=0.08,
            ),
        )

        controller._advance_target(np.array([0.75, 0.0]))
        self.assertEqual(
            controller.path_world[controller.target_index].tolist(),
            [1.0, 0.0],
        )

        command, debug = controller.compute_command(
            (1.0, 0.0, 0.0),
            (0.0, 0.0),
        )
        self.assertGreater(debug.heading_error, 1.0)
        self.assertEqual(float(command[0]), 0.0)

    def test_dwa_hard_path_limit_rejects_outside_trajectory(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (2.0, 0.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                path_deviation_limit=0.10,
                enforce_path_deviation_limit=True,
            ),
        )

        command, debug = controller.compute_command(
            (0.5, 0.15, 0.0),
            (0.0, 0.0),
        )

        self.assertEqual(debug.feasible_candidates, 0)
        self.assertGreater(debug.path_deviation_rejections, 0)
        self.assertEqual(float(command[0]), 0.0)

    def test_dwa_initial_alignment_allows_rotation_drift_then_restores_strict_limit(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((100, 100), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (2.0, 0.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.12,
                rotate_in_place_angle=0.45,
                path_deviation_limit=0.14,
                enforce_path_deviation_limit=True,
                initial_alignment_path_deviation_limit=0.40,
            ),
        )

        rotation_command, rotation_debug = controller.compute_command(
            (0.2, 0.17, math.pi / 2.0),
            (0.0, 0.0),
        )

        self.assertTrue(rotation_debug.initial_alignment_active)
        self.assertAlmostEqual(rotation_debug.path_deviation_limit_used, 0.40)
        self.assertGreater(rotation_debug.feasible_candidates, 0)
        self.assertEqual(float(rotation_command[0]), 0.0)

        controller.compute_command((0.2, 0.02, 0.0), (0.0, 0.0))
        _, strict_debug = controller.compute_command(
            (0.4, 0.17, 0.0),
            (0.0, 0.0),
        )

        self.assertFalse(strict_debug.initial_alignment_active)
        self.assertAlmostEqual(strict_debug.path_deviation_limit_used, 0.14)
        self.assertEqual(strict_debug.feasible_candidates, 0)
        self.assertGreater(strict_debug.path_deviation_rejections, 0)

    def test_dwa_path_recovery_can_return_from_small_tracking_overshoot(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((100, 100), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (2.0, 0.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.12,
                path_deviation_limit=0.14,
                enforce_path_deviation_limit=True,
                path_recovery_deviation_limit=0.20,
            ),
        )

        _, recovery_debug = controller.compute_command(
            (0.4, 0.15, 0.0),
            (0.0, 0.0),
        )

        self.assertTrue(recovery_debug.path_recovery_active)
        self.assertAlmostEqual(recovery_debug.path_deviation_limit_used, 0.20)
        self.assertGreater(recovery_debug.feasible_candidates, 0)

        controller.compute_command((0.5, 0.10, 0.0), (0.0, 0.0))
        _, strict_debug = controller.compute_command(
            (0.6, 0.10, 0.0),
            (0.0, 0.0),
        )

        self.assertFalse(strict_debug.path_recovery_active)
        self.assertAlmostEqual(strict_debug.path_deviation_limit_used, 0.14)

    def test_dwa_path_recovery_activates_before_zero_velocity_deadlock(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((100, 100), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (2.0, 0.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.12,
                path_deviation_limit=0.14,
                enforce_path_deviation_limit=True,
                path_recovery_deviation_limit=0.35,
                max_linear_velocity=0.20,
                max_linear_accel=1.00,
                use_command_velocity_window=True,
            ),
        )

        command, debug = controller.compute_command(
            (0.4, 0.13, 0.0),
            (0.0, 0.0),
        )

        self.assertTrue(debug.path_recovery_active)
        self.assertAlmostEqual(debug.path_deviation_limit_used, 0.35)
        self.assertGreater(debug.feasible_candidates, 0)
        self.assertGreater(float(command[0]), 0.0)

    def test_dwa_near_goal_allows_final_convergence_off_sparse_path(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((100, 100), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        controller = DWAController(
            [(0.0, 0.0), (2.0, 0.0)],
            grid,
            DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.12,
                goal_tracking_distance=0.80,
                path_deviation_limit=0.14,
                enforce_path_deviation_limit=True,
                path_recovery_deviation_limit=0.35,
                near_goal_path_deviation_limit=0.75,
                near_goal_path_deviation_distance=0.65,
            ),
        )

        command, debug = controller.compute_command(
            (1.60, 0.50, 0.0),
            (0.0, 0.0),
        )

        self.assertTrue(debug.path_recovery_active)
        self.assertAlmostEqual(debug.path_deviation_limit_used, 0.75)
        self.assertGreater(debug.feasible_candidates, 0)
        self.assertLess(debug.path_deviation_rejections, debug.sampled_candidates)

    def test_pct_executor_refines_sparse_path_with_local_map(self) -> None:
        occupancy = np.zeros((40, 40), dtype=bool)
        occupancy[:, 15] = True
        occupancy[25, 15] = False
        grid = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0))
        plan = NavPlan(
            goal=NavGoal(x=3.5, y=0.5, yaw=0.0),
            waypoints=((0.2, 0.5), (3.5, 0.5)),
            metadata={"planner": "pct"},
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
        )

        executor.reset(plan)

        report = executor.status()["local_refinement"]
        self.assertTrue(report["success"])
        self.assertEqual(report["input_waypoints"], 2)
        self.assertGreater(report["output_waypoints"], 2)
        self.assertLess(
            report["collinear_waypoints"],
            report["raw_grid_waypoints"],
        )

    def test_pct_path_simplification_keeps_occupied_corner_detour(self) -> None:
        occupancy = np.zeros((12, 12), dtype=bool)
        occupancy[5, 6] = True
        occupancy[6, 5] = True
        grid = OccupancyGridMap(occupancy, 0.1, (0.0, 0.0, 0.0))
        start = grid.grid_to_world(8, 4)
        goal = grid.grid_to_world(3, 8)
        plan = NavPlan(
            goal=NavGoal(x=goal[0], y=goal[1], yaw=0.0),
            waypoints=(start, goal),
            metadata={"planner": "pct"},
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
        )

        executor.reset(plan)

        controller = executor._controller
        self.assertIsNotNone(controller)
        self.assertGreater(len(controller.reference_path_world), 2)

    def test_pct_multifloor_path_preserves_global_waypoint_order(self) -> None:
        single_floor_grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        multifloor_grid = OccupancyGridMap(
            np.zeros((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.0, 0.5),
            (0.8, 0.2, 1.0),
            (1.0, 0.2, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.2, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 2,
                "slice_end": 4,
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
        )

        executor.reset(plan)

        report = executor.status()["local_refinement"]
        self.assertEqual(report["mode"], "pct_multifloor_path_preserved")
        self.assertEqual(report["output_waypoints"], 4)
        self.assertIs(executor.local_map, multifloor_grid)
        self.assertEqual(
            executor._controller.reference_path_world.tolist(),
            [[0.0, 0.0], [0.5, 0.0], [0.8, 0.2], [1.0, 0.2]],
        )

    def test_pct_no_float_simplifies_clear_flat_approach_before_stair(self) -> None:
        occupancy = np.zeros((80, 80), dtype=bool)
        occupancy[[0, -1], :] = True
        occupancy[:, [0, -1]] = True
        single_floor_grid = OccupancyGridMap(
            occupancy,
            0.1,
            (-2.0, -2.0, 0.0),
        )
        multifloor_grid = OccupancyGridMap(
            occupancy.copy(),
            0.1,
            (-2.0, -2.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (0.3, 0.2, 0.0),
            (0.6, -0.2, 0.0),
            (0.9, 0.2, 0.0),
            (1.2, -0.2, 0.0),
            (1.5, 0.2, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.2, 0.3),
            (2.0, 0.4, 0.6),
            (2.2, 0.4, 0.6),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.2, y=0.4, z=0.6, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 2,
                "slice_end": 4,
                "execution_phase": "carry_nav_to_place",
                "stair_centerline_refinement": {
                    "applied": True,
                    "approach_start": [2.0, 0.0, 0.0],
                },
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            multifloor_no_float_clearance_radius=0.25,
            dwa_config=DWAConfig(control_dt=0.05),
        )

        executor.reset(plan)

        status = executor.status()
        flat_report = status["local_refinement"]["flat_approach"]
        self.assertTrue(flat_report["applied"])
        self.assertEqual(
            flat_report["reason"],
            "planner_path_already_clearance_optimized",
        )
        planner_optimization = flat_report["planner_optimization"]
        self.assertEqual(planner_optimization["original_pre_stair_point_count"], 7)
        self.assertGreaterEqual(planner_optimization["preserve_end_index"], 2)
        self.assertLess(
            planner_optimization["maximum_heading_change_after_rad"],
            planner_optimization["maximum_heading_change_before_rad"],
        )
        self.assertAlmostEqual(
            status["map_selection"]["local_clearance_radius_m"],
            0.25,
        )
        reference_path = executor._controller.reference_path_world.tolist()
        self.assertEqual(reference_path[0], [0.0, 0.0])
        self.assertEqual(reference_path[-1], [2.2, 0.4])
        self.assertGreater(len(reference_path), len(path_3d))

    def test_pct_carry_map_keeps_route_corridor_and_blocks_deviations(self) -> None:
        single_floor_grid = OccupancyGridMap(
            np.zeros((60, 60), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        multifloor_occupancy = np.ones((60, 60), dtype=bool)
        multifloor_grid = OccupancyGridMap(
            multifloor_occupancy,
            0.1,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.5),
            (2.0, 0.0, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.0, y=0.0, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 2,
                "slice_end": 4,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(
                control_dt=0.05,
                max_linear_velocity=0.45,
                max_angular_velocity=0.50,
                max_linear_accel=2.5,
                path_deviation_limit=0.30,
            ),
            multifloor_obstacle_inflate_radius=0.10,
            multifloor_route_corridor_radius=0.10,
            carry_max_linear_velocity=0.20,
            carry_max_angular_velocity=0.30,
            carry_max_linear_accel=1.00,
            carry_path_deviation_limit=0.12,
            carry_initial_alignment_path_deviation_limit=0.40,
            carry_path_recovery_deviation_limit=0.35,
            carry_max_infeasible_recomputes=8,
        )

        executor.reset(plan)

        self.assertFalse(
            executor.local_map.is_occupied(
                *executor.local_map.world_to_grid(1.0, 0.0)
            )
        )
        self.assertTrue(
            executor.local_map.is_occupied(
                *executor.local_map.world_to_grid(1.0, 0.5)
            )
        )
        status = executor.status()
        self.assertGreater(status["map_selection"]["route_cells_cleared"], 0)
        self.assertAlmostEqual(status["dwa_limits"]["max_linear_velocity"], 0.20)
        self.assertAlmostEqual(status["dwa_limits"]["max_angular_velocity"], 0.30)
        self.assertAlmostEqual(status["dwa_limits"]["max_linear_accel"], 1.00)
        self.assertAlmostEqual(status["dwa_limits"]["path_deviation_limit"], 0.12)
        self.assertTrue(status["dwa_limits"]["enforce_path_deviation_limit"])
        self.assertAlmostEqual(
            status["dwa_limits"]["initial_alignment_path_deviation_limit"],
            0.30,
        )
        self.assertAlmostEqual(
            status["dwa_limits"]["path_recovery_deviation_limit"],
            0.30,
        )
        self.assertIsNone(status["dwa_limits"]["path_recovery_speed_limit"])
        self.assertAlmostEqual(
            status["dwa_limits"]["near_goal_path_deviation_limit"],
            0.75,
        )
        self.assertAlmostEqual(
            status["dwa_limits"]["near_goal_path_deviation_distance"],
            0.65,
        )
        self.assertTrue(status["dwa_limits"]["preserve_sharp_corners"])
        self.assertAlmostEqual(
            status["dwa_limits"]["corner_waypoint_tolerance"],
            0.18,
        )
        self.assertAlmostEqual(status["dwa_limits"]["rotate_in_place_angle"], 1.60)
        self.assertIsNone(status["dwa_limits"]["large_heading_creep_velocity"])
        self.assertEqual(status["dwa_limits"]["max_infeasible_recomputes"], 8)

    def test_pct_physical_flat_carry_avoids_policy_standing_velocity_deadzone(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.1,
            (-2.0, -2.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (2.0, 0.0)),
            metadata={
                "execution_phase": "carry_nav_to_place",
            },
        )
        dwa_config = DWAConfig(
            control_dt=0.05,
            max_linear_velocity=0.25,
            max_linear_accel=1.0,
        )
        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )
        no_float = DwaNavExecutor(
            grid_map=grid,
            multifloor_grid_map=grid,
            dwa_config=dwa_config,
            carry_max_linear_velocity=0.25,
            stair_float_enabled=False,
        )
        with_float = DwaNavExecutor(
            grid_map=grid,
            multifloor_grid_map=grid,
            dwa_config=dwa_config,
            carry_max_linear_velocity=0.25,
            stair_float_enabled=True,
            stair_float_activation_radius_m=0.0,
            stair_float_approach_distance_m=0.0,
        )

        no_float.reset(plan)
        with_float.reset(plan)
        debug = DWADebug(
            target_index=1,
            target_point=(1.0, 0.0),
            distance_to_target=1.0,
            distance_to_goal=2.0,
            heading_error=0.0,
            clearance=1.0,
            score=1.0,
            reached_goal=False,
            near_goal_tracking=False,
            sampled_candidates=1,
            feasible_candidates=1,
            collision_rejections=0,
            path_deviation_rejections=0,
            best_linear_velocity=0.05,
            best_angular_velocity=0.0,
            path_distance=0.0,
            path_deviation_limit_used=0.14,
            initial_alignment_active=False,
            path_recovery_active=False,
            window_linear_velocity=0.05,
            window_angular_velocity=0.0,
            velocity_window_source="command",
        )

        class FixedLowSpeedController:
            def compute_command(self, pose, velocity):
                del pose, velocity
                return np.array([0.05, 0.0, 0.0], dtype=np.float32), debug

        no_float._controller = FixedLowSpeedController()
        with_float._controller = FixedLowSpeedController()
        no_float_action = no_float.compute_action(state)
        with_float_action = with_float.compute_action(state)

        self.assertAlmostEqual(no_float_action.base_velocity[0], 0.25)
        self.assertAlmostEqual(with_float_action.base_velocity[0], 0.25)

        with_float._stair_float_started = True
        with_float._stair_float_done = True
        post_float_action = with_float.compute_action(state)
        self.assertAlmostEqual(post_float_action.base_velocity[0], 0.25)

        class FixedTurningController:
            def compute_command(self, pose, velocity):
                del pose, velocity
                return np.array([0.10, 0.0, 0.30], dtype=np.float32), debug

        no_float._controller = FixedTurningController()
        turning_action = no_float.compute_action(state)
        self.assertAlmostEqual(turning_action.base_velocity[0], 0.25)
        self.assertAlmostEqual(turning_action.base_velocity[2], 0.30)
        self.assertIsNone(
            with_float.status()["dwa_limits"]["no_float_carry_gait_activation_vx"]
        )

    def test_pct_carry_stair_float_freezes_base_only_on_stair_segment(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.10,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.30),
            (0.5, 0.0, 0.30),
            (0.7, 0.2, 0.80),
            (1.0, 0.5, 1.35),
            (1.4, 0.6, 1.80),
            (2.0, 0.6, 1.80),
            (2.5, 0.6, 1.80),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.5, y=0.6, z=1.80, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 12,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(
                control_dt=0.10,
                max_linear_velocity=0.45,
                max_angular_velocity=0.50,
            ),
            stair_float_enabled=True,
            stair_float_speed_mps=1.0,
            stair_float_activation_radius_m=0.12,
            stair_float_completion_radius_m=0.05,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
            stair_float_settle_time_s=0.0,
        )
        executor.reset(plan)
        pre_stair_state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.3, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )
        state = SimulationState(
            step_index=1,
            timestamp=0.1,
            robot_root_pose=(0.5, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )

        before_stair = executor.compute_action(pre_stair_state)
        first = executor.compute_action(state)

        self.assertEqual(before_stair.source, "navigation_dwa")
        self.assertNotIn("navigation_base_pose_lock", before_stair.metadata)
        self.assertFalse(executor.status()["stair_float"]["approach_distance_m"])
        self.assertEqual(first.source, "navigation_stair_float")
        self.assertTrue(first.metadata["navigation_base_pose_lock"])
        self.assertTrue(first.metadata["navigation_support_joint_lock"])
        self.assertTrue(first.metadata["navigation_full_body_joint_lock"])
        self.assertTrue(first.metadata["navigation_carry_object_follow"])
        self.assertEqual(
            first.metadata["navigation_dog_joint_names"],
            PCT_STAIR_FLOAT_DOG_JOINT_NAMES,
        )
        self.assertEqual(
            first.metadata["navigation_dog_joint_positions"],
            PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS,
        )
        first_target = first.metadata["navigation_base_pose_lock_xyzyaw"]
        self.assertGreater(first_target[2], 0.30)

        action = first
        for step_index in range(1, 40):
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]
            yaw = float(target[3])
            state = SimulationState(
                step_index=step_index,
                timestamp=0.10 * step_index,
                robot_root_pose=(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                robot_root_velocity=(0.0,) * 6,
            )
            action = executor.compute_action(state)
            if action.source == "navigation_stair_float_completed":
                break

        self.assertEqual(action.source, "navigation_stair_float_completed")
        status = executor.status()
        self.assertTrue(status["stair_float"]["done"])
        self.assertEqual(status["stair_float"]["reason"], "completed")

        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        next_state = SimulationState(
            step_index=50,
            timestamp=5.0,
            robot_root_pose=(
                float(target[0]),
                float(target[1]),
                float(target[2]),
                math.cos(float(target[3]) / 2.0),
                0.0,
                0.0,
                math.sin(float(target[3]) / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        resumed = executor.compute_action(next_state)

        self.assertNotIn("navigation_base_pose_lock", resumed.metadata)
        self.assertIn(
            resumed.source,
            {
                "navigation_dwa",
                "navigation_terminal_pose",
                "navigation_completed",
            },
        )

    def test_pct_carry_stair_float_activates_in_first_segment_corridor(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((120, 120), dtype=bool),
            0.10,
            (-2.0, -2.0, 0.0),
        )
        path_3d = (
            (1.5, 5.7, 0.0),
            (1.5060733333333334, 5.892276666666667, 0.09828666666666666),
            (1.51822, 6.27683, 0.29486),
            (2.7, 7.05, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.7, y=7.05, z=3.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 14,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.10),
            stair_float_enabled=True,
            stair_float_activation_radius_m=0.12,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
        )
        executor.reset(plan)

        # 该位姿来自 gui_float 失败日志：距入口点 0.124m，但仍位于首段走廊内。
        observed_pose = (
            1.6196765899658203,
            5.732725143432617,
            2.0905144677922816,
        )
        self.assertGreater(
            math.hypot(observed_pose[0] - 1.5, observed_pose[1] - 5.7),
            0.12,
        )
        self.assertTrue(executor._stair_float_should_activate(observed_pose))
        status = executor.status()["stair_float"]
        self.assertEqual(status["activation_trigger"], "entry_corridor")
        self.assertLessEqual(status["activation_corridor_distance_m"], 0.12)

    def test_pct_carry_stair_float_corridor_does_not_activate_before_entry(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((120, 120), dtype=bool),
            0.10,
            (-2.0, -2.0, 0.0),
        )
        path_3d = (
            (1.5, 5.7, 0.0),
            (1.5060733333333334, 5.892276666666667, 0.09828666666666666),
            (1.51822, 6.27683, 0.29486),
            (2.7, 7.05, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.7, y=7.05, z=3.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 14,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.10),
            stair_float_enabled=True,
            stair_float_activation_radius_m=0.12,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
        )
        executor.reset(plan)

        self.assertFalse(executor._stair_float_should_activate((1.619, 5.55, 2.0)))
        status = executor.status()["stair_float"]
        self.assertIsNone(status["activation_trigger"])
        self.assertIsNone(status["activation_corridor_distance_m"])

    def test_pct_carry_stair_float_extends_exit_and_settles_before_release(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((100, 100), dtype=bool),
            0.10,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.0, 0.5),
            (1.0, 0.5, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.5, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=1.5, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 12,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.10),
            stair_float_enabled=True,
            stair_float_speed_mps=2.0,
            stair_float_activation_radius_m=0.50,
            stair_float_completion_radius_m=0.01,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.75,
            stair_float_settle_time_s=0.20,
        )
        executor.reset(plan)

        status = executor.status()
        self.assertEqual(status["stair_float"]["end"], [1.0, 1.5, 1.0])
        self.assertEqual(status["stair_float"]["path_point_count"], 5)

        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.5, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )
        action = executor.compute_action(state)
        sources = [action.source]
        for step_index in range(1, 20):
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]
            yaw = float(target[3])
            state = SimulationState(
                step_index=step_index,
                timestamp=0.10 * step_index,
                robot_root_pose=(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                robot_root_velocity=(0.0,) * 6,
            )
            action = executor.compute_action(state)
            sources.append(action.source)
            if action.source == "navigation_stair_float_completed":
                break

        self.assertEqual(action.source, "navigation_stair_float_completed")
        self.assertTrue(action.metadata["navigation_base_pose_lock"])

        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        yaw = float(target[3])
        settle_state = SimulationState(
            step_index=len(sources),
            timestamp=0.10 * len(sources),
            robot_root_pose=(
                float(target[0]),
                float(target[1]),
                float(target[2]),
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        settle_action = executor.compute_action(settle_state)

        self.assertEqual(settle_action.source, "navigation_stair_float_settle")
        self.assertTrue(settle_action.metadata["navigation_base_pose_lock"])
        self.assertTrue(settle_action.metadata["navigation_support_joint_lock"])
        self.assertTrue(settle_action.metadata["navigation_full_body_joint_lock"])
        self.assertFalse(
            settle_action.metadata["navigation_stair_float_root_only_settle"]
        )
        self.assertTrue(
            settle_action.metadata["navigation_carry_object_follow"]
        )

    def test_pct_carry_stair_float_straightens_tail_climb_turn(
        self,
    ) -> None:
        grid = OccupancyGridMap(
            np.zeros((160, 160), dtype=bool),
            0.10,
            (-6.0, -1.0, 0.0),
        )
        path_3d = (
            (-3.478511428833008, 6.456423950195314, 0.0),
            (-3.278511428833008, 6.2564239501953125, 0.0),
            (-3.278511428833008, 5.456423950195314, 0.0),
            (-2.878511428833008, 5.056423950195313, 0.0),
            (-2.6785114288330076, 5.056423950195313, 0.0),
            (-2.478511428833008, 4.856423950195313, 0.0),
            (-1.6785114288330079, 4.856423950195313, 0.0),
            (-1.478511428833008, 4.656423950195313, 0.0),
            (-0.6785114288330079, 4.656423950195313, 0.0),
            (0.12148857116699219, 4.656423950195313, 0.0),
            (0.5214885711669923, 4.656423950195313, 0.0),
            (0.9214885711669922, 5.056423950195313, 0.0),
            (1.3214885711669924, 5.056423950195313, 0.0),
            (1.5214885711669925, 5.2564239501953125, 0.0),
            (1.5214885711669925, 5.856423950195312, 0.0),
            (1.5214885711669925, 6.656423950195313, 0.5),
            (1.5214885711669925, 6.856423950195314, 0.5),
            (1.7214885711669923, 7.656423950195313, 1.0),
            (1.7214885711669923, 8.056423950195313, 1.0),
            (1.9214885711669925, 8.856423950195314, 1.5),
            (1.9214885711669925, 9.256423950195312, 1.5),
            (2.3214885711669924, 9.256423950195312, 1.5),
            (2.9214885711669925, 8.656423950195313, 2.0),
            (2.9214885711669925, 8.456423950195314, 2.0),
            (2.9214885711669925, 7.656423950195313, 2.5),
            (2.3214885711669924, 7.656423950195313, 3.0),
            (2.5214885711669925, 7.456423950195314, 3.0),
            (2.5214885711669925, 6.656423950195313, 3.0),
            (2.5214885711669925, 6.456423950195314, 3.0),
            (2.3214885711669924, 6.2564239501953125, 3.0),
            (2.3214885711669924, 5.2564239501953125, 3.0),
            (2.3214885711669924, 5.056423950195313, 3.0),
            (1.3214885711669924, 5.056423950195313, 3.0),
            (1.1214885711669922, 4.856423950195313, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.4, y=-0.1, z=3.62628, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 15,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            stair_float_enabled=True,
            stair_float_speed_mps=0.5,
            stair_float_activation_radius_m=0.45,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=1.80,
        )

        executor.reset(plan)
        status = executor.status()["stair_float"]

        tail_path = tuple(
            point
            for point in executor._stair_float_path
            if float(point[2]) >= 2.5
        )
        self.assertGreaterEqual(len(tail_path), 3)
        self.assertTrue(
            all(
                math.isclose(float(point[0]), 2.9214885711669925, abs_tol=1.0e-6)
                for point in tail_path
            )
        )
        self.assertTrue(
            all(
                float(next_point[1]) <= float(point[1]) + 1.0e-6
                for point, next_point in zip(tail_path, tail_path[1:])
            )
        )

        self.assertEqual(status["exit_distance_m"], 0.75)
        self.assertEqual(
            status["start"][:2],
            [0.5214885711669923, 4.656423950195313],
        )
        self.assertAlmostEqual(status["end"][0], 2.9214885711669925)
        self.assertAlmostEqual(status["end"][1], 5.973581237720693)
        self.assertAlmostEqual(status["end"][2], 3.0)
        self.assertEqual(status["path_point_count"], 18)
        splice = status["controller_path_splice"]
        self.assertTrue(splice["applied"])
        self.assertEqual(splice["reason"], "stair_release_forward_merge")
        self.assertLess(float(splice["merge_xy"][1]), float(status["end"][1]))
        self.assertLess(
            abs(float(splice["merge_heading_error_rad"])),
            math.radians(75.0),
        )

        release = status["end"]
        executor._sync_controller_after_stair_float(
            (float(release[0]), float(release[1]), float(release[2]))
        )
        executor._stair_float_done = True
        release_state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(
                float(release[0]),
                float(release[1]),
                3.62628,
                math.cos(-math.pi / 4.0),
                0.0,
                0.0,
                math.sin(-math.pi / 4.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        resumed_action = executor.compute_action(release_state)
        resumed_status = executor.status()

        self.assertEqual(resumed_action.source, "navigation_dwa")
        self.assertGreater(resumed_status["dwa"]["feasible_candidates"], 0)
        self.assertEqual(resumed_status["dwa"]["path_deviation_rejections"], 0)

    def test_pct_carry_stair_float_route_clears_projected_release_corridor(
        self,
    ) -> None:
        single_floor_grid = OccupancyGridMap(
            np.zeros((160, 160), dtype=bool),
            0.10,
            (-6.0, -1.0, 0.0),
        )
        multifloor_occupancy = np.zeros((160, 160), dtype=bool)
        multifloor_grid = OccupancyGridMap(
            multifloor_occupancy,
            0.10,
            (-6.0, -1.0, 0.0),
        )
        release_xy = (2.9214885711669925, 5.973581237720693)
        release_row, release_col = multifloor_grid.world_to_grid(*release_xy)
        multifloor_occupancy[
            release_row - 1 : release_row + 2,
            release_col - 1 : release_col + 2,
        ] = True
        path_3d = (
            (-3.478511428833008, 6.456423950195314, 0.0),
            (-3.278511428833008, 6.2564239501953125, 0.0),
            (-3.278511428833008, 5.456423950195314, 0.0),
            (-2.878511428833008, 5.056423950195313, 0.0),
            (-2.6785114288330076, 5.056423950195313, 0.0),
            (-2.478511428833008, 4.856423950195313, 0.0),
            (-1.6785114288330079, 4.856423950195313, 0.0),
            (-1.478511428833008, 4.656423950195313, 0.0),
            (-0.6785114288330079, 4.656423950195313, 0.0),
            (0.12148857116699219, 4.656423950195313, 0.0),
            (0.5214885711669923, 4.656423950195313, 0.0),
            (0.9214885711669922, 5.056423950195313, 0.0),
            (1.3214885711669924, 5.056423950195313, 0.0),
            (1.5214885711669925, 5.2564239501953125, 0.0),
            (1.5214885711669925, 5.856423950195312, 0.0),
            (1.5214885711669925, 6.656423950195313, 0.5),
            (1.5214885711669925, 6.856423950195314, 0.5),
            (1.7214885711669923, 7.656423950195313, 1.0),
            (1.7214885711669923, 8.056423950195313, 1.0),
            (1.9214885711669925, 8.856423950195314, 1.5),
            (1.9214885711669925, 9.256423950195312, 1.5),
            (2.3214885711669924, 9.256423950195312, 1.5),
            (2.9214885711669925, 8.656423950195313, 2.0),
            (2.9214885711669925, 8.456423950195314, 2.0),
            (2.9214885711669925, 7.656423950195313, 2.5),
            (2.3214885711669924, 7.656423950195313, 3.0),
            (2.5214885711669925, 7.456423950195314, 3.0),
            (2.5214885711669925, 6.656423950195313, 3.0),
            (2.5214885711669925, 6.456423950195314, 3.0),
            (2.3214885711669924, 6.2564239501953125, 3.0),
            (2.3214885711669924, 5.2564239501953125, 3.0),
            (2.3214885711669924, 5.056423950195313, 3.0),
            (1.3214885711669924, 5.056423950195313, 3.0),
            (1.1214885711669922, 4.856423950195313, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.4, y=-0.1, z=3.62628, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 15,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            multifloor_route_corridor_radius=0.24,
            multifloor_obstacle_inflate_radius=0.0,
            stair_float_enabled=True,
            stair_float_speed_mps=0.5,
            stair_float_activation_radius_m=0.45,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=1.80,
        )

        self.assertTrue(multifloor_grid.is_occupied(release_row, release_col))
        executor.reset(plan)
        map_status = executor.status()["map_selection"]

        self.assertGreater(map_status["stair_float_route_cells_cleared"], 0)
        self.assertIsNotNone(executor.local_map)
        self.assertFalse(executor.local_map.is_occupied(release_row, release_col))

    def test_pct_carry_replans_on_single_floor_map_after_stair_float(
        self,
    ) -> None:
        floor_occupancy = np.zeros((40, 50), dtype=bool)
        floor_grid = OccupancyGridMap(
            floor_occupancy,
            0.10,
            (0.0, 0.0, 0.0),
        )
        wall_row, wall_col = floor_grid.world_to_grid(2.0, 1.0)
        floor_occupancy[wall_row - 6 : wall_row + 7, wall_col] = True
        floor_grid = OccupancyGridMap(
            floor_occupancy,
            0.10,
            (0.0, 0.0, 0.0),
        )
        multifloor_grid = OccupancyGridMap(
            np.zeros((40, 50), dtype=bool),
            0.10,
            (0.0, 0.0, 0.0),
        )
        start_floor_grid = OccupancyGridMap(
            np.zeros((40, 50), dtype=bool),
            0.10,
            (0.0, 0.0, 0.0),
        )
        path_3d = (
            (0.5, 0.5, 0.0),
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            (3.0, 1.0, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=3.0, y=1.0, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 1,
                "slice_end": 5,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=start_floor_grid,
            multifloor_grid_map=multifloor_grid,
            post_stair_grid_map=floor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            stair_float_enabled=True,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
        )

        executor.reset(plan)
        self.assertIs(executor._raw_map, multifloor_grid)
        executor._sync_controller_after_stair_float((1.0, 1.0, 1.0))

        status = executor.status()
        replan = status["stair_float"]["post_stair_floor_replan"]
        self.assertTrue(replan["applied"])
        self.assertEqual(
            replan["reason"],
            "post_stair_pct_clearance_optimized",
        )
        self.assertTrue(replan["path_optimization"]["applied"])
        self.assertAlmostEqual(replan["max_linear_velocity"], 0.25)
        self.assertAlmostEqual(replan["max_angular_velocity"], 0.50)
        self.assertAlmostEqual(replan["min_active_linear_velocity"], 0.22)
        self.assertAlmostEqual(
            replan["near_goal_min_active_linear_velocity"],
            0.22,
        )
        self.assertAlmostEqual(
            replan["close_goal_speed_limit"],
            0.22,
        )
        self.assertAlmostEqual(replan["path_deviation_limit"], 0.14)
        self.assertAlmostEqual(replan["path_recovery_deviation_limit"], 0.50)
        self.assertAlmostEqual(replan["path_recovery_speed_limit"], 0.20)
        self.assertAlmostEqual(replan["corner_waypoint_tolerance"], 0.18)
        self.assertGreaterEqual(replan["lookahead_distance"], 0.40)
        self.assertAlmostEqual(replan["rotate_in_place_angle"], 0.55)
        self.assertEqual(
            status["map_selection"]["active_map"],
            "post_stair_single_floor",
        )
        self.assertIs(executor._raw_map, floor_grid)
        self.assertIsNotNone(executor._controller)
        self.assertTrue(
            all(
                not floor_grid.is_occupied(
                    *floor_grid.world_to_grid(float(point[0]), float(point[1]))
                )
                for point in executor._controller.path_world
            )
        )
        self.assertTrue(
            any(
                abs(float(point[1]) - 1.0) > 0.20
                for point in executor._controller.path_world
            )
        )

    def test_post_stair_release_bridge_keeps_footprint_inflation(self) -> None:
        occupancy = np.zeros((20, 20), dtype=bool)
        occupancy[:, 4] = True
        raw_map = OccupancyGridMap(occupancy, 0.10, (0.0, 0.0, 0.0))
        start = raw_map.grid_to_world(10, 6)
        reference = raw_map.grid_to_world(10, 7)
        goal = raw_map.grid_to_world(10, 14)
        executor = DwaNavExecutor(
            grid_map=raw_map,
            post_stair_grid_map=raw_map,
            post_stair_clearance_radius=0.20,
            dwa_config=DWAConfig(control_dt=0.05),
        )
        executor.plan = NavPlan(
            goal=NavGoal(x=goal[0], y=goal[1], z=1.0, yaw=0.0),
            waypoints=(start, reference, goal),
            metadata={"planner": "pct", "execution_phase": "carry_nav_to_place"},
        )
        executor._post_stair_reference_path = (reference, goal)
        executor._post_stair_path_optimization_report = {"applied": True}

        applied = executor._replan_controller_on_post_stair_floor(
            (start[0], start[1], 1.0),
        )
        report = executor.status()["stair_float"]["post_stair_floor_replan"]

        self.assertTrue(applied)
        self.assertIsNotNone(executor.local_map)
        self.assertTrue(
            executor.local_map.is_occupied(
                *executor.local_map.world_to_grid(*start)
            )
        )
        bridge = report["path_optimization"]["release_bridge"]
        self.assertEqual(bridge["mode"], "occupied_start_escape")
        self.assertEqual(bridge["reopened_cells"], 0)

    def test_pct_post_stair_path_uses_footprint_clearance_and_rounds_corners(
        self,
    ) -> None:
        occupancy = np.zeros((100, 100), dtype=bool)
        raw_map = OccupancyGridMap(
            occupancy,
            0.10,
            (0.0, 0.0, 0.0),
        )
        for x in np.arange(4.0, 6.01, 0.10):
            for y in np.arange(3.0, 7.01, 0.10):
                occupancy[raw_map.world_to_grid(float(x), float(y))] = True
        raw_map = OccupancyGridMap(
            occupancy,
            0.10,
            (0.0, 0.0, 0.0),
        )

        footprint_map, path_world, report = (
            _plan_clearance_optimized_floor_path(
                raw_map,
                (2.0, 5.0),
                (8.0, 5.0),
                clearance_radius_m=0.20,
            )
        )

        self.assertTrue(report["applied"])
        self.assertGreaterEqual(report["rounded_corner_count"], 1)
        self.assertLess(
            report["maximum_heading_change_after_rad"],
            report["maximum_heading_change_before_rad"],
        )
        self.assertGreaterEqual(
            report["minimum_raw_map_clearance_m"],
            0.20,
        )
        self.assertTrue(_world_path_is_clear(path_world, footprint_map))

    def test_pct_post_stair_path_preserves_single_cell_corridor_centers(
        self,
    ) -> None:
        occupancy = np.ones((20, 20), dtype=bool)
        occupancy[1:19, 7:10] = False
        raw_map = OccupancyGridMap(
            occupancy,
            0.20,
            (0.0, 0.0, 0.0),
        )
        start = raw_map.grid_to_world(16, 8)
        goal = raw_map.grid_to_world(3, 8)

        footprint_map, path_world, report = (
            _plan_clearance_optimized_floor_path(
                raw_map,
                start,
                goal,
                clearance_radius_m=0.20,
            )
        )

        self.assertGreaterEqual(report["narrow_corridor_anchor_count"], 10)
        self.assertAlmostEqual(report["optimization_edge_margin_m"], 0.02)
        for row in range(4, 16):
            center = footprint_map.grid_to_world(row, 8)
            self.assertTrue(
                any(
                    math.hypot(point[0] - center[0], point[1] - center[1])
                    <= 1.0e-8
                    for point in path_world
                ),
                msg=f"缺少单格通道中心锚点 row={row}",
            )

    def test_pct_carry_stair_entry_allows_turn_but_rejects_handrail_drift(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.10,
            (-1.0, 4.0, 0.0),
        )
        path_3d = (
            (1.5214885711669925, 5.856423950195312, 0.0),
            (1.5214885711669925, 6.656423950195313, 0.5),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.5214885711669925, y=6.656423950195313, z=0.5, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 9,
                "slice_end": 10,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(
                control_dt=0.05,
                lookahead_distance=0.12,
                waypoint_tolerance=0.05,
                max_linear_velocity=0.45,
                max_angular_velocity=0.50,
                max_linear_accel=2.5,
                path_deviation_limit=0.30,
                use_command_velocity_window=True,
            ),
            carry_max_linear_velocity=0.20,
            carry_max_angular_velocity=0.30,
            carry_max_linear_accel=1.00,
            carry_path_deviation_limit=0.14,
            carry_initial_alignment_path_deviation_limit=0.40,
            carry_path_recovery_deviation_limit=0.35,
            carry_max_infeasible_recomputes=8,
        )
        executor.reset(plan)

        aligned_state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(
                1.5214885711669925,
                5.90,
                0.30,
                math.cos(math.pi / 4.0),
                0.0,
                0.0,
                math.sin(math.pi / 4.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        executor.compute_action(aligned_state)

        drifted_state = SimulationState(
            step_index=1,
            timestamp=0.05,
            robot_root_pose=(
                1.8174806833267212,
                6.036071300506592,
                0.31,
                math.cos(2.087394762116031 / 2.0),
                0.0,
                0.0,
                math.sin(2.087394762116031 / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        action = executor.compute_action(drifted_state)
        status = executor.status()

        self.assertFalse(status["failed"])
        self.assertLessEqual(abs(action.base_velocity[0]), 0.20 + 1.0e-6)
        self.assertLessEqual(abs(action.base_velocity[2]), 0.30 + 1.0e-6)
        self.assertGreater(status["dwa"]["feasible_candidates"], 0)
        self.assertGreater(float(action.base_velocity[0]), 0.0)
        self.assertGreater(abs(float(action.base_velocity[2])), 0.0)
        self.assertTrue(status["dwa"]["path_recovery_active"])
        self.assertAlmostEqual(status["dwa"]["path_deviation_limit_used"], 0.30)

        handrail_drift_state = SimulationState(
            step_index=2,
            timestamp=0.10,
            robot_root_pose=(
                1.88,
                6.036071300506592,
                0.31,
                math.cos(2.087394762116031 / 2.0),
                0.0,
                0.0,
                math.sin(2.087394762116031 / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        handrail_action = executor.compute_action(handrail_drift_state)
        handrail_status = executor.status()

        self.assertEqual(handrail_status["dwa"]["feasible_candidates"], 0)
        self.assertAlmostEqual(float(handrail_action.base_velocity[0]), 0.0)

        executor._last_dwa_debug = replace(
            executor._last_dwa_debug,
            heading_error=1.50,
            best_linear_velocity=0.0,
            best_angular_velocity=0.30,
        )
        executor._consecutive_infeasible_recomputes = 0
        for _ in range(10):
            executor._update_infeasible_recomputes()
        self.assertFalse(executor.status()["failed"])
        self.assertEqual(executor._consecutive_infeasible_recomputes, 0)

    def test_pct_carry_route_corridor_covers_grid_discretization_margin(self) -> None:
        single_floor_grid = OccupancyGridMap(
            np.zeros((20, 20), dtype=bool),
            0.2,
            (-1.0, -1.0, 0.0),
        )
        multifloor_grid = OccupancyGridMap(
            np.ones((20, 20), dtype=bool),
            0.2,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (-0.5, 0.065, 0.0),
            (0.8, 0.065, 0.5),
        )
        plan = NavPlan(
            goal=NavGoal(x=0.8, y=0.065, z=0.5, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 2,
                "slice_end": 4,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            multifloor_route_corridor_radius=0.24,
            carry_path_deviation_limit=0.14,
        )

        executor.reset(plan)
        query = (0.2, -0.07)

        self.assertLessEqual(abs(query[1] - path_3d[0][1]), 0.14)
        self.assertFalse(
            executor.local_map.is_occupied(
                *executor.local_map.world_to_grid(*query)
            )
        )
        self.assertAlmostEqual(
            executor.status()["map_selection"]["route_corridor_radius_m"],
            0.24,
        )

    def test_pct_carry_stair_float_includes_flat_stairwell_approach(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((140, 140), dtype=bool),
            0.1,
            (-5.0, -1.0, 0.0),
        )
        path_3d = (
            (-3.4785, 6.6564, 0.0),
            (-3.2785, 6.4564, 0.0),
            (-3.2785, 5.6564, 0.0),
            (-2.8785, 5.0564, 0.0),
            (-1.6785, 4.8564, 0.0),
            (-0.6785, 4.6564, 0.0),
            (0.1215, 4.6564, 0.0),
            (0.5215, 4.6564, 0.0),
            (0.9215, 5.0564, 0.0),
            (1.3215, 5.0564, 0.0),
            (1.5215, 5.2564, 0.0),
            (1.5215, 5.8564, 0.0),
            (1.5215, 6.6564, 0.5),
            (1.7215, 7.6564, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.7215, y=7.6564, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 15,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            stair_float_enabled=True,
            stair_float_speed_mps=0.5,
            stair_float_activation_radius_m=0.45,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=1.80,
        )

        executor.reset(plan)
        status = executor.status()

        self.assertEqual(status["stair_float"]["start"][:2], [0.5215, 4.6564])
        self.assertEqual(status["stair_float"]["path_point_count"], 7)

        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(
                0.8625,
                4.6308,
                0.2088,
                math.cos(0.1888 / 2.0),
                0.0,
                0.0,
                math.sin(0.1888 / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )
        action = executor.compute_action(state)
        target = action.metadata["navigation_base_pose_lock_xyzyaw"]

        self.assertEqual(action.source, "navigation_stair_float")
        self.assertTrue(action.metadata["navigation_base_pose_lock"])
        self.assertGreater(target[2], 0.15)
        self.assertAlmostEqual(
            executor.status()["stair_float"]["root_z_offset_m"],
            0.2088,
            delta=0.05,
        )
        self.assertAlmostEqual(
            executor.status()["stair_float"]["measured_root_z_offset_m"],
            0.2088,
            delta=0.05,
        )

    def test_pct_carry_stair_float_starts_from_activation_pose_without_xy_jump(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((50, 50), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (1.0, 0.0, 0.0),
            (1.0, 0.6, 0.5),
            (1.0, 1.2, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=1.2, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 12,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            stair_float_enabled=True,
            stair_float_speed_mps=0.5,
            stair_float_activation_radius_m=0.45,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
            stair_float_settle_time_s=0.0,
        )
        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.65, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )

        executor.reset(plan)
        action = executor.compute_action(state)
        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        status = executor.status()["stair_float"]

        self.assertEqual(action.source, "navigation_stair_float")
        self.assertTrue(status["activation_inserted_current_point"])
        self.assertGreater(status["activation_target_jump_m"], 0.30)
        self.assertLess(abs(target[0] - 0.65), 0.05)
        self.assertLess(abs(target[1]), 0.05)

    def test_pct_carry_stair_float_preserves_measured_activation_height(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.5),
            (0.5, 1.5, 1.5),
            (1.0, 2.0, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=2.0, z=3.62628, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 15,
                "execution_phase": "carry_nav_to_place",
                "robot_root_to_floor_m": 0.45,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.10),
            stair_float_enabled=True,
            stair_float_speed_mps=2.0,
            stair_float_activation_radius_m=0.45,
            stair_float_completion_radius_m=0.01,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
            stair_float_settle_time_s=0.0,
        )
        executor.reset(plan)
        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.0, -0.2, 0.22, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )

        action = executor.compute_action(state)
        first_target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        status = executor.status()["stair_float"]

        self.assertEqual(action.source, "navigation_stair_float")
        self.assertAlmostEqual(status["initial_root_z_offset_m"], 0.22)
        self.assertAlmostEqual(status["target_root_z_offset_m"], 0.36)
        self.assertLess(abs(float(first_target[2]) - 0.22), 0.08)

        for step_index in range(1, 40):
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]
            yaw = float(target[3])
            state = SimulationState(
                step_index=step_index,
                timestamp=0.10 * step_index,
                robot_root_pose=(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                robot_root_velocity=(0.0,) * 6,
            )
            action = executor.compute_action(state)
            if action.source == "navigation_stair_float_completed":
                break

        self.assertEqual(action.source, "navigation_stair_float_completed")
        final_target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        self.assertAlmostEqual(float(final_target[2]), 3.36, places=4)

    def test_pct_carry_stair_float_release_settle_drops_full_body_before_root_release(
        self,
    ) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        path_3d = (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.5),
            (0.5, 1.5, 1.5),
            (1.0, 2.0, 3.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=2.0, z=3.62628, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 8,
                "slice_end": 15,
                "execution_phase": "carry_nav_to_place",
                "robot_root_to_floor_m": 0.45,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.10),
            stair_float_enabled=True,
            stair_float_speed_mps=2.0,
            stair_float_activation_radius_m=0.45,
            stair_float_completion_radius_m=0.01,
            stair_float_min_z_delta_m=0.75,
            stair_float_approach_distance_m=0.0,
            stair_float_exit_distance_m=0.0,
            stair_float_settle_time_s=0.0,
            stair_float_release_settle_time_s=0.20,
        )
        executor.reset(plan)
        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.0, -0.2, 0.22, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )

        action = executor.compute_action(state)
        for step_index in range(1, 40):
            target = action.metadata["navigation_base_pose_lock_xyzyaw"]
            yaw = float(target[3])
            state = SimulationState(
                step_index=step_index,
                timestamp=0.10 * step_index,
                robot_root_pose=(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                robot_root_velocity=(0.0,) * 6,
            )
            action = executor.compute_action(state)
            if action.source == "navigation_stair_float_completed":
                break

        self.assertEqual(action.source, "navigation_stair_float_completed")
        self.assertTrue(action.metadata["navigation_base_pose_lock"])
        self.assertTrue(action.metadata["navigation_full_body_joint_lock"])
        target = action.metadata["navigation_base_pose_lock_xyzyaw"]
        yaw = float(target[3])
        release_state = SimulationState(
            step_index=50,
            timestamp=float(state.timestamp) + 0.10,
            robot_root_pose=(
                float(target[0]),
                float(target[1]),
                float(target[2]),
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            ),
            robot_root_velocity=(0.0,) * 6,
        )

        release_action = executor.compute_action(release_state)

        self.assertEqual(
            release_action.source,
            "navigation_stair_float_release_settle",
        )
        self.assertTrue(release_action.metadata["navigation_base_pose_lock"])
        self.assertEqual(
            release_action.metadata["navigation_base_pose_lock_phase"],
            "pct_stair_float_release_settle",
        )
        self.assertNotIn("navigation_full_body_joint_lock", release_action.metadata)
        self.assertTrue(release_action.metadata["navigation_support_joint_lock"])
        self.assertTrue(release_action.metadata["navigation_carry_object_follow"])
        self.assertEqual(release_action.gripper_command, "hold")

        final_release_state = SimulationState(
            step_index=51,
            timestamp=float(release_state.timestamp) + 0.30,
            robot_root_pose=release_state.robot_root_pose,
            robot_root_velocity=(0.0,) * 6,
        )
        final_release = executor.compute_action(final_release_state)

        self.assertEqual(
            final_release.source,
            "navigation_stair_float_release_settle_completed",
        )
        self.assertNotIn("navigation_base_pose_lock", final_release.metadata)
        self.assertNotIn("navigation_full_body_joint_lock", final_release.metadata)
        self.assertNotIn("navigation_support_joint_lock", final_release.metadata)
        self.assertTrue(final_release.metadata["navigation_stair_float_completed"])

    def test_pct_carry_initial_turn_rotation_recovers_only_once(self) -> None:
        grid = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-1.0, -1.0, 0.0),
        )
        path_3d = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, z=1.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 0,
                "slice_end": 4,
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            stall_window_steps=4,
            stall_min_progress_m=0.05,
        )
        executor.reset(plan)
        executor._last_dwa_debug = DWADebug(
            target_index=1,
            target_point=(1.0, 0.0),
            distance_to_target=0.18,
            distance_to_goal=1.0,
            heading_error=-0.876,
            clearance=0.40,
            score=1.0,
            reached_goal=False,
            near_goal_tracking=False,
            sampled_candidates=21,
            feasible_candidates=14,
            collision_rejections=0,
            path_deviation_rejections=7,
            best_linear_velocity=0.105,
            best_angular_velocity=-0.30,
            path_distance=0.105,
            path_deviation_limit_used=0.12,
            initial_alignment_active=False,
            path_recovery_active=False,
            window_linear_velocity=0.105,
            window_angular_velocity=-0.30,
            velocity_window_source="command",
        )
        poses = (0.0, 0.015, 0.030, 0.049)
        body_velocity = (0.009, 0.0, -0.319)
        command = (0.105, 0.0, -0.30)

        for x in poses:
            executor._update_stall((x, 0.0, 0.0), body_velocity, command)

        first_status = executor.status()
        self.assertFalse(first_status["stall_detected"])
        self.assertEqual(first_status["stall_recovery_count"], 1)
        self.assertEqual(
            first_status["last_stall_recovery_reason"],
            "pct_initial_turn_rotation",
        )
        self.assertTrue(first_status["pct_initial_turn_stall_recovery_used"])

        for x in poses:
            executor._update_stall((x, 0.0, 0.0), body_velocity, command)

        second_status = executor.status()
        self.assertTrue(second_status["stall_detected"])
        self.assertEqual(second_status["failure_reason"], "nav_collision")
        self.assertEqual(second_status["stall_recovery_count"], 1)

    def test_pct_carry_stops_after_repeated_collision_only_windows(self) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            carry_path_deviation_limit=0.14,
            carry_max_infeasible_recomputes=3,
        )
        state = SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )
        executor.reset(plan)

        actions = [executor.compute_action(state) for _ in range(3)]
        status = executor.status()

        self.assertEqual(actions[-1].source, "navigation_stalled")
        self.assertTrue(status["done"])
        self.assertEqual(status["failure_reason"], "nav_collision")
        self.assertEqual(
            status["dwa_limits"]["consecutive_infeasible_recomputes"],
            3,
        )

    def test_pct_carry_stall_window_resets_after_measured_motion_recovers(self) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            stall_window_steps=4,
            stall_min_progress_m=0.05,
        )
        executor.reset(plan)

        for x in (0.0, 0.01, 0.02, 0.04):
            executor._update_stall(
                (x, 0.0, 0.0),
                (0.10, 0.0, 0.0),
                (0.20, 0.0, 0.0),
            )

        status = executor.status()
        self.assertFalse(status["done"])
        self.assertEqual(status["stall_recovery_count"], 1)
        self.assertEqual(status["stall"]["sample_count"], 0)

    def test_post_stair_float_near_threshold_motion_resets_stall_window(
        self,
    ) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            stall_window_steps=4,
            stall_min_progress_m=0.05,
            stair_float_enabled=True,
        )
        executor.reset(plan)
        executor._stair_float_done = True
        executor._last_dwa_debug = DWADebug(
            target_index=1,
            target_point=(1.0, 0.0),
            distance_to_target=1.0,
            distance_to_goal=1.0,
            heading_error=0.0,
            clearance=1.0,
            score=1.0,
            reached_goal=False,
            near_goal_tracking=False,
            sampled_candidates=21,
            feasible_candidates=21,
            collision_rejections=0,
            path_deviation_rejections=0,
            best_linear_velocity=0.20,
            best_angular_velocity=0.0,
            path_distance=0.0,
            path_deviation_limit_used=0.14,
            initial_alignment_active=False,
            path_recovery_active=False,
            window_linear_velocity=0.20,
            window_angular_velocity=0.0,
            velocity_window_source="command",
        )

        for x in (0.0, 0.015, 0.030, 0.046):
            executor._update_stall(
                (x, 0.0, 0.0),
                (0.05, 0.0, 0.0),
                (0.20, 0.0, 0.0),
            )

        status = executor.status()
        self.assertFalse(status["done"])
        self.assertEqual(status["stall_recovery_count"], 1)
        self.assertEqual(
            status["last_stall_recovery_reason"],
            "post_stair_float_near_threshold_motion",
        )
        self.assertEqual(status["stall"]["sample_count"], 0)

    def test_no_float_carry_near_threshold_motion_resets_stall_window(self) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            stall_window_steps=4,
            stall_min_progress_m=0.05,
            stair_float_enabled=False,
        )
        executor.reset(plan)
        executor._last_dwa_debug = DWADebug(
            target_index=1,
            target_point=(1.0, 0.0),
            distance_to_target=1.0,
            distance_to_goal=1.0,
            heading_error=0.0,
            clearance=1.0,
            score=1.0,
            reached_goal=False,
            near_goal_tracking=False,
            sampled_candidates=21,
            feasible_candidates=21,
            collision_rejections=0,
            path_deviation_rejections=0,
            best_linear_velocity=0.20,
            best_angular_velocity=0.0,
            path_distance=0.0,
            path_deviation_limit_used=0.14,
            initial_alignment_active=False,
            path_recovery_active=False,
            window_linear_velocity=0.20,
            window_angular_velocity=0.0,
            velocity_window_source="command",
        )

        for x in (0.0, 0.015, 0.030, 0.046):
            executor._update_stall(
                (x, 0.0, 0.0),
                (0.015, 0.0, 0.0),
                (0.20, 0.0, 0.0),
            )

        status = executor.status()
        self.assertFalse(status["done"])
        self.assertEqual(status["failure_reason"], "")
        self.assertEqual(status["stall_recovery_count"], 1)
        self.assertEqual(
            status["last_stall_recovery_reason"],
            "carry_near_threshold_motion",
        )

    def test_carry_near_threshold_motion_allows_partial_collision_filtering(self) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=1.0, y=0.0, yaw=0.0),
            waypoints=((0.0, 0.0), (1.0, 0.0)),
            metadata={
                "planner": "pct",
                "execution_phase": "carry_nav_to_place",
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            dwa_config=DWAConfig(control_dt=0.05),
            stall_window_steps=4,
            stall_min_progress_m=0.05,
        )
        executor.reset(plan)
        executor._last_dwa_debug = DWADebug(
            target_index=1,
            target_point=(1.0, 0.0),
            distance_to_target=1.0,
            distance_to_goal=1.0,
            heading_error=0.0,
            clearance=0.2,
            score=1.0,
            reached_goal=False,
            near_goal_tracking=False,
            sampled_candidates=21,
            feasible_candidates=14,
            collision_rejections=7,
            path_deviation_rejections=0,
            best_linear_velocity=0.06,
            best_angular_velocity=-0.12,
            path_distance=0.23,
            path_deviation_limit_used=0.35,
            initial_alignment_active=False,
            path_recovery_active=True,
            window_linear_velocity=0.06,
            window_angular_velocity=-0.06,
            velocity_window_source="command",
        )

        for x in (0.0, 0.015, 0.030, 0.046):
            executor._update_stall(
                (x, 0.0, 0.0),
                (0.03, 0.02, 0.0),
                (0.06, 0.0, -0.12),
            )

        status = executor.status()
        self.assertFalse(status["done"])
        self.assertEqual(status["stall_recovery_count"], 1)
        self.assertEqual(
            status["last_stall_recovery_reason"],
            "carry_near_threshold_motion",
        )

    def test_pct_route_corridor_preserves_global_hard_obstacles(self) -> None:
        grid = OccupancyGridMap(
            np.ones((40, 40), dtype=bool),
            0.1,
            (-1.0, -1.0, 0.0),
        )
        protected_occupancy = np.zeros((40, 40), dtype=bool)
        protected_map = OccupancyGridMap(
            protected_occupancy,
            0.1,
            (-1.0, -1.0, 0.0),
        )
        protected_row, protected_col = protected_map.world_to_grid(1.0, 0.0)
        protected_occupancy[protected_row, protected_col] = True
        path_3d = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.5),
            (2.0, 0.0, 1.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.0, y=0.0, z=1.0, yaw=0.0),
            waypoints=tuple((point[0], point[1]) for point in path_3d),
            metadata={
                "planner": "pct",
                "path_3d": path_3d,
                "slice_start": 2,
                "slice_end": 4,
            },
        )
        executor = DwaNavExecutor(
            grid_map=grid,
            multifloor_grid_map=grid,
            multifloor_protected_obstacle_map=protected_map,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
            multifloor_route_corridor_radius=0.10,
        )

        executor.reset(plan)

        self.assertTrue(
            executor.local_map.is_occupied(
                *executor.local_map.world_to_grid(1.0, 0.0)
            )
        )
        self.assertGreater(
            executor.status()["map_selection"]["protected_cells_preserved"],
            0,
        )

    def test_pct_adjacent_slices_with_flat_path_use_direct_same_floor_route(self) -> None:
        single_floor_grid = OccupancyGridMap(
            np.zeros((120, 120), dtype=bool),
            0.1,
            (-2.0, -2.0, 0.0),
        )
        multifloor_grid = OccupancyGridMap(
            np.zeros((120, 120), dtype=bool),
            0.1,
            (-2.0, -2.0, 0.0),
        )
        plan = NavPlan(
            goal=NavGoal(x=2.0, y=5.0, z=0.35, yaw=0.0),
            waypoints=((1.8, 0.2), (1.8, 2.5), (2.0, 5.0)),
            metadata={
                "planner": "pct",
                "path_3d": (
                    (1.8, 0.2, 0.0),
                    (1.8, 2.5, 0.0),
                    (2.0, 5.0, 0.0),
                ),
                "sim_start": (2.0, 0.0, 0.25),
                "slice_start": 9,
                "slice_end": 10,
            },
        )
        executor = DwaNavExecutor(
            grid_map=single_floor_grid,
            multifloor_grid_map=multifloor_grid,
            local_clearance_radius=0.0,
            dwa_config=DWAConfig(control_dt=0.05),
        )

        executor.reset(plan)

        report = executor.status()["local_refinement"]
        self.assertEqual(report["mode"], "pct_same_floor_direct")
        self.assertEqual(report["output_waypoints"], 2)
        self.assertGreaterEqual(report["direct_clearance_m"], 0.4)
        self.assertIs(executor.local_map, single_floor_grid)
        self.assertEqual(
            executor._controller.reference_path_world.tolist(),
            [[2.0, 0.0], [2.0, 5.0]],
        )

    def test_nav_planner_loads_map_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            occupancy = np.zeros((30, 30), dtype=bool)
            np.save(root / "occupancy.npy", occupancy)
            (root / "map.json").write_text(
                json.dumps({"image": "occupancy.npy", "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )
            planner = NavPlanner(str(root / "map.json"), 0.0, DWAConfig(control_dt=0.05))
            path = planner.plan_global_path((0.15, 0.15), (1.15, 0.15))
            command = planner.compute_command((0.15, 0.15, 0.0), (0.0, 0.0), path)
            self.assertEqual(len(command), 3)
            self.assertEqual(command[1], 0.0)

    def test_nav_planner_rejects_blocked_task_goal_instead_of_snapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            occupancy = np.zeros((30, 30), dtype=bool)
            occupancy[18, 15] = True
            np.save(root / "occupancy.npy", occupancy)
            (root / "map.json").write_text(
                json.dumps({"image": "occupancy.npy", "resolution": 0.1, "origin": [0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )
            planner = NavPlanner(str(root / "map.json"), 0.0, DWAConfig(control_dt=0.05))
            blocked_goal = planner.global_map.grid_to_world(18, 15)
            with self.assertRaisesRegex(ValueError, "goal cell .* is occupied"):
                planner.plan_global_path((0.15, 0.15), blocked_goal)


if __name__ == "__main__":
    unittest.main()
