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


if __name__ == "__main__":
    unittest.main()
