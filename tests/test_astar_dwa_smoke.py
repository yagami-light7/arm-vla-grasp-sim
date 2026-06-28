"""Pure-Python global-planning and closed-loop DWA smoke tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.interfaces.navigation import NavGoal, NavPlan
from source.navigation import NavPlanner
from source.navigation.executor import DwaNavExecutor
from source.navigation.navlib import AStarPlanner, DWAConfig, DWAController, OccupancyGridMap


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
            carry_max_linear_velocity=0.30,
            carry_max_angular_velocity=0.35,
            carry_max_linear_accel=1.50,
            carry_path_deviation_limit=0.14,
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
        self.assertAlmostEqual(status["dwa_limits"]["max_linear_velocity"], 0.30)
        self.assertAlmostEqual(status["dwa_limits"]["max_angular_velocity"], 0.35)
        self.assertAlmostEqual(status["dwa_limits"]["max_linear_accel"], 1.50)
        self.assertAlmostEqual(status["dwa_limits"]["path_deviation_limit"], 0.14)

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
