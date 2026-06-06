"""Pure-Python global-planning and closed-loop DWA smoke tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.navigation import NavPlanner
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
