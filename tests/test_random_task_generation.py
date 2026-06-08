"""Tests for randomized pick task base-goal generation."""

from __future__ import annotations

import math
import random
import tempfile
import unittest

import numpy as np

from source.data.random_task import (
    ObjectPose,
    SpawnRegion,
    generate_random_pick_task,
    generate_base_goal_candidates,
    sample_object_pose,
    select_valid_base_goal,
    _path_final_heading,
    _path_length,
)
from source.navigation.navlib import OccupancyGridMap


class RandomTaskGenerationTest(unittest.TestCase):
    def _write_temp_map(self, directory: str) -> str:
        grid = OccupancyGridMap(np.zeros((60, 60), dtype=bool), 0.05, (0.0, 0.0, 0.0))
        map_json = f"{directory}/map.json"
        grid.save_pgm(f"{directory}/occupancy.pgm")
        grid.save_meta_file(map_json)
        return map_json

    def test_spawn_region_uses_table_z_as_default_object_z(self) -> None:
        spawn_region = SpawnRegion(
            x_min=0.86,
            x_max=0.96,
            y_min=0.9,
            y_max=1.6,
            table_z=0.82,
        )

        object_pose = sample_object_pose(random.Random(7), spawn_region)

        self.assertAlmostEqual(spawn_region.object_z, 0.82)
        self.assertAlmostEqual(object_pose.z, 0.82)

    def test_spawn_region_keeps_explicit_object_z_offset(self) -> None:
        spawn_region = SpawnRegion(
            x_min=0.86,
            x_max=0.96,
            y_min=0.9,
            y_max=1.6,
            table_z=0.82,
            object_z_offset=0.04,
        )

        object_pose = sample_object_pose(random.Random(7), spawn_region)

        self.assertAlmostEqual(spawn_region.object_z, 0.86)
        self.assertAlmostEqual(object_pose.z, 0.86)

    def test_fixed_object_pose_overrides_z_and_rpy_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_json = self._write_temp_map(tmpdir)
            base_task = {
                "nav_map": map_json,
                "start": {"x": 0.1, "y": 0.1, "yaw": 0.0},
                "pick": {"object_prim_path": "/World/apple"},
            }
            kwargs = {
                "base_task": base_task,
                "spawn_region": SpawnRegion(
                    x_min=0.86,
                    x_max=0.96,
                    y_min=0.9,
                    y_max=1.6,
                    table_z=0.82,
                    object_z_offset=0.04,
                ),
                "object_fixed_z": 0.81653,
                "object_fixed_rpy": (-2.524, -7.822, -0.181),
                "object_fixed_rpy_unit": "deg",
                "randomize_object_yaw": False,
                "edge_margin": None,
                "clearance_radius": 0.0,
                "min_boundary_clearance": 0.0,
            }

            task_seed_7 = generate_random_pick_task(seed=7, **kwargs)
            task_seed_8 = generate_random_pick_task(seed=8, **kwargs)

        pose_7 = task_seed_7["pick"]["object_pose_world"]
        pose_8 = task_seed_8["pick"]["object_pose_world"]
        self.assertNotEqual((pose_7["x"], pose_7["y"]), (pose_8["x"], pose_8["y"]))
        for pose in (pose_7, pose_8):
            self.assertAlmostEqual(pose["z"], 0.81653)
            self.assertAlmostEqual(pose["roll"], math.radians(-2.524))
            self.assertAlmostEqual(pose["pitch"], math.radians(-7.822))
            self.assertAlmostEqual(pose["yaw"], math.radians(-0.181))

        policy = task_seed_7["randomization"]["object_pose_policy"]
        self.assertEqual(policy["xy"], "random_in_table_region")
        self.assertEqual(policy["z"], "fixed")
        self.assertEqual(policy["rpy"], "fixed")
        self.assertEqual(policy["fixed_z"], 0.81653)
        self.assertEqual(policy["fixed_rpy_input"], [-2.524, -7.822, -0.181])
        self.assertEqual(policy["fixed_rpy_input_unit"], "deg")
        self.assertEqual(policy["fixed_rpy_stored_unit"], "rad")
        self.assertFalse(policy["randomize_object_yaw"])

    def test_path_final_heading_uses_dense_tail_segment(self) -> None:
        path = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.1), (0.3, 0.2)]

        heading = _path_final_heading(path, lookback_points=3, min_segment_length=0.15)

        self.assertIsNotNone(heading)
        self.assertAlmostEqual(float(heading), math.atan2(0.2, 0.2))
        self.assertAlmostEqual(_path_length(path), 0.1 + math.sqrt(0.02) + math.sqrt(0.02))

    def test_path_final_heading_returns_none_for_short_tail_segment(self) -> None:
        path = [(0.0, 0.0), (0.04, 0.0), (0.08, 0.0)]

        heading = _path_final_heading(path, lookback_points=2, min_segment_length=0.10)

        self.assertIsNone(heading)

    def test_base_goal_candidate_rejects_bad_path_final_heading(self) -> None:
        grid = OccupancyGridMap(np.zeros((40, 40), dtype=bool), 0.1, (-1.0, -1.0, 0.0))
        object_pose = ObjectPose(x=1.0, y=0.0, z=0.86, yaw=0.0, edge_side="x_max")

        candidates = generate_base_goal_candidates(
            object_pose,
            standoff_candidates=(0.5,),
            approach_angles_deg=(180.0,),
            grid_map=grid,
            clearance_map=grid,
            clearance_radius=0.0,
            min_boundary_clearance=0.1,
            start_xy=(0.0, 0.0),
            preferred_edge_side=object_pose.edge_side,
            max_path_heading_error=1.0,
            path_heading_lookback_points=5,
            path_heading_min_segment_length=0.10,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertTrue(candidate.path_found)
        self.assertIsNotNone(candidate.path_heading_error)
        self.assertGreater(float(candidate.path_heading_error), 1.0)
        self.assertEqual(candidate.rejection_reason, "path_heading_error_above_1.000rad")
        self.assertIsNone(select_valid_base_goal(candidates))

    def test_base_goal_candidate_accepts_aligned_path_final_heading(self) -> None:
        grid = OccupancyGridMap(np.zeros((40, 40), dtype=bool), 0.1, (-1.0, -1.0, 0.0))
        object_pose = ObjectPose(x=1.0, y=0.0, z=0.86, yaw=0.0, edge_side="x_min")

        candidates = generate_base_goal_candidates(
            object_pose,
            standoff_candidates=(0.5,),
            approach_angles_deg=(0.0,),
            grid_map=grid,
            clearance_map=grid,
            clearance_radius=0.0,
            min_boundary_clearance=0.1,
            start_xy=(0.0, 0.0),
            preferred_edge_side=object_pose.edge_side,
            max_path_heading_error=1.0,
            path_heading_lookback_points=5,
            path_heading_min_segment_length=0.10,
        )

        selected = select_valid_base_goal(candidates)
        self.assertIsNotNone(selected)
        self.assertTrue(selected.path_found)
        self.assertIsNotNone(selected.path_heading_error)
        self.assertLess(float(selected.path_heading_error), 0.25)

    def test_object_offset_base_goal_mode_uses_direct_xy_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_json = self._write_temp_map(tmpdir)
            base_task = {
                "nav_map": map_json,
                "start": {"x": 0.1, "y": 0.1, "yaw": 0.0},
                "pick": {"object_prim_path": "/World/apple"},
            }
            task = generate_random_pick_task(
                base_task,
                seed=3,
                spawn_region=SpawnRegion(
                    x_min=1.0,
                    x_max=1.0,
                    y_min=1.0,
                    y_max=1.0,
                    table_z=0.82,
                ),
                edge_margin=None,
                base_goal_mode="object_offset",
                base_goal_offset_xy=(0.35, 0.0),
                clearance_radius=0.0,
                min_boundary_clearance=0.0,
                max_path_heading_error=0.01,
            )

        object_pose = task["pick"]["object_pose_world"]
        base_goal = task["pick"]["base_goal"]
        generation = task["randomization"]["base_goal_generation"]
        self.assertAlmostEqual(base_goal["x"], object_pose["x"] + 0.35)
        self.assertAlmostEqual(base_goal["y"], object_pose["y"])
        self.assertAlmostEqual(abs(base_goal["yaw"]), math.pi)
        self.assertEqual(generation["mode"], "object_offset")
        self.assertEqual(generation["base_goal_offset_xy"], [0.35, 0.0])
        self.assertFalse(generation["path_heading_filter"]["enabled"])

    def test_object_offset_edge_bias_respects_selected_edge_clearance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_json = self._write_temp_map(tmpdir)
            base_task = {
                "nav_map": map_json,
                "start": {"x": 0.1, "y": 0.1, "yaw": 0.0},
                "pick": {"object_prim_path": "/World/apple"},
            }
            task = generate_random_pick_task(
                base_task,
                seed=7,
                spawn_region=SpawnRegion(
                    x_min=0.86,
                    x_max=0.96,
                    y_min=0.9,
                    y_max=1.6,
                    table_z=0.82,
                ),
                edge_margin=0.12,
                edge_min_clearance=0.03,
                base_goal_mode="object_offset",
                base_goal_offset_xy=(0.28, -0.08),
                clearance_radius=0.0,
                min_boundary_clearance=0.0,
            )

        edge_sampling = task["randomization"]["object_edge_sampling"]
        self.assertGreaterEqual(edge_sampling["selected_edge_distance_m"], 0.03)
        self.assertAlmostEqual(task["pick"]["object_pose_world"]["z"], 0.82)


if __name__ == "__main__":
    unittest.main()
