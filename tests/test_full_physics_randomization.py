"""Full-physics episode 随机化与可视化描述测试。"""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

from scripts.pipeline.run_full_physics_pipeline import _keep_gui_open, _parse_args
from source.diagnostics import randomization_debug_spec
from source.pipeline import (
    BaseGoalRandomizationSettings,
    FullPhysicsConfig,
    RandomizationSettings,
)
from source.tasks import JsonTaskProvider, episode_spec_from_dict, prepare_episode_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"


class FullPhysicsRandomizationTest(unittest.TestCase):
    def test_cli_randomization_and_visualization_defaults(self) -> None:
        args = _parse_args(["--task-json", str(TASK_PATH), "--dry-run"])

        self.assertTrue(args.randomize_task)
        self.assertTrue(args.randomize_base_goal)
        self.assertFalse(args.show_randomization_debug)
        self.assertFalse(args.keep_window_open)

    def test_keep_gui_open_updates_until_window_closes(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.update_count = 0

            def is_running(self) -> bool:
                return self.update_count < 3

            def update(self) -> None:
                self.update_count += 1

        app = FakeApp()
        _keep_gui_open(app)

        self.assertEqual(app.update_count, 3)

    def test_randomized_episode_is_deterministic_and_preserves_goal_offsets(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        settings = RandomizationSettings(
            enabled=True,
            show_debug_region=True,
            base_goal=BaseGoalRandomizationSettings(enabled=False),
        )

        first = prepare_episode_spec(
            base,
            episode_id=7,
            seed=41,
            settings=settings,
        )
        repeated = prepare_episode_spec(
            base,
            episode_id=7,
            seed=41,
            settings=settings,
        )
        second = prepare_episode_spec(
            base,
            episode_id=8,
            seed=42,
            settings=settings,
        )

        self.assertEqual(first.raw_task, repeated.raw_task)
        self.assertNotEqual(first.object_initial_pose[:2], second.object_initial_pose[:2])
        self.assertEqual(first.episode_id, 7)
        self.assertGreaterEqual(first.object_initial_pose[0], settings.pick_x_range[0])
        self.assertLessEqual(first.object_initial_pose[0], settings.pick_x_range[1])
        self.assertGreaterEqual(first.object_initial_pose[1], settings.pick_y_range[0])
        self.assertLessEqual(first.object_initial_pose[1], settings.pick_y_range[1])
        self.assertGreaterEqual(first.place_target_pose[0], settings.place_x_range[0])
        self.assertLessEqual(first.place_target_pose[0], settings.place_x_range[1])
        self.assertGreaterEqual(first.place_target_pose[1], settings.place_y_range[0])
        self.assertLessEqual(first.place_target_pose[1], settings.place_y_range[1])

        base_pick_offset = (
            base.pick_goal.x - base.object_initial_pose[0],
            base.pick_goal.y - base.object_initial_pose[1],
        )
        randomized_pick_offset = (
            first.pick_goal.x - first.object_initial_pose[0],
            first.pick_goal.y - first.object_initial_pose[1],
        )
        self.assertAlmostEqual(randomized_pick_offset[0], base_pick_offset[0])
        self.assertAlmostEqual(randomized_pick_offset[1], base_pick_offset[1])
        base_pick_pose = base.raw_task["pick"]["object_pose_world"]
        randomized_pick_pose = first.raw_task["pick"]["object_pose_world"]
        expected_rpy = tuple(math.radians(value) for value in (-2.524, -7.822, -0.181))
        self.assertAlmostEqual(base_pick_pose["z"], 0.81653)
        self.assertAlmostEqual(base_pick_pose["roll"], expected_rpy[0])
        self.assertAlmostEqual(base_pick_pose["pitch"], expected_rpy[1])
        self.assertAlmostEqual(base_pick_pose["yaw"], expected_rpy[2])
        for key in set(base_pick_pose) - {"x", "y"}:
            self.assertEqual(randomized_pick_pose[key], base_pick_pose[key], key)
        base_pick_goal = base.raw_task["pick"]["base_goal"]
        randomized_pick_goal = first.raw_task["pick"]["base_goal"]
        for key in set(base_pick_goal) - {"x", "y"}:
            self.assertEqual(randomized_pick_goal[key], base_pick_goal[key], key)
        self.assertEqual(
            first.raw_task["randomization"]["object_pose_policy"],
            {
                "mode": "xy_only",
                "randomize_xy": True,
                "randomize_z": False,
                "randomize_roll": False,
                "randomize_pitch": False,
                "randomize_yaw": False,
            },
        )

        base_place_offset = (
            base.place_goal.x - base.place_target_pose[0],
            base.place_goal.y - base.place_target_pose[1],
        )
        randomized_place_offset = (
            first.place_goal.x - first.place_target_pose[0],
            first.place_goal.y - first.place_target_pose[1],
        )
        self.assertAlmostEqual(randomized_place_offset[0], base_place_offset[0])
        self.assertAlmostEqual(randomized_place_offset[1], base_place_offset[1])
        base_place_pose = base.raw_task["place"]["place_pose_world"]
        randomized_place_pose = first.raw_task["place"]["place_pose_world"]
        for key in set(base_place_pose) - {"x", "y"}:
            self.assertEqual(randomized_place_pose[key], base_place_pose[key], key)
        base_place_goal = base.raw_task["place"]["base_goal"]
        randomized_place_goal = first.raw_task["place"]["base_goal"]
        for key in set(base_place_goal) - {"x", "y"}:
            self.assertEqual(randomized_place_goal[key], base_place_goal[key], key)

    def test_debug_spec_uses_task_regions_without_physics(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        settings = RandomizationSettings(
            show_debug_region=True,
            base_goal=BaseGoalRandomizationSettings(enabled=False),
        )
        episode = prepare_episode_spec(
            base,
            episode_id=1,
            seed=54,
            settings=settings,
        )

        spec = randomization_debug_spec(episode.raw_task)

        self.assertFalse(spec["physics_enabled"])
        self.assertFalse(spec["collision_enabled"])
        self.assertEqual(spec["usd_purpose"], "default")
        self.assertEqual(
            spec["pick"]["xy_range"],
            (settings.pick_x_range, settings.pick_y_range),
        )
        self.assertEqual(
            spec["place"]["xy_range"],
            (settings.place_x_range, settings.place_y_range),
        )

    def test_base_goal_randomization_disabled_preserves_fixed_goals(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        episode = prepare_episode_spec(
            base,
            episode_id=3,
            seed=5,
            settings=RandomizationSettings(
                enabled=False,
                base_goal=BaseGoalRandomizationSettings(enabled=False),
            ),
        )

        self.assertEqual(episode.pick_goal, base.pick_goal)
        self.assertEqual(episode.place_goal, base.place_goal)
        self.assertNotIn("base_goal_randomization", episode.raw_task.get("randomization", {}))

    def test_base_goal_randomization_is_seeded_and_records_metadata(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        raw = copy.deepcopy(base.raw_task)
        raw["nav_map"] = "missing_map_for_unit_test.json"
        base = episode_spec_from_dict(raw)
        settings = RandomizationSettings(
            enabled=True,
            base_goal=BaseGoalRandomizationSettings(enabled=True),
        )

        first = prepare_episode_spec(base, episode_id=1, seed=11, settings=settings)
        repeated = prepare_episode_spec(base, episode_id=1, seed=11, settings=settings)
        second = prepare_episode_spec(base, episode_id=2, seed=12, settings=settings)

        self.assertEqual(first.raw_task, repeated.raw_task)
        self.assertNotEqual(first.pick_goal, base.pick_goal)
        self.assertNotEqual(first.place_goal, base.place_goal)
        self.assertNotEqual(first.pick_goal, second.pick_goal)
        self.assertNotEqual(first.place_goal, second.place_goal)

        report = first.raw_task["randomization"]["base_goal_randomization"]
        self.assertTrue(report["enabled"])
        self.assertEqual(report["pick"]["nav_map_check"]["status"], "not_available")
        self.assertEqual(report["config"]["pick_yaw_noise_deg"], 0.0)
        self.assertEqual(report["config"]["place_yaw_noise_deg"], 0.0)
        pick_sample = report["pick"]
        self.assertTrue(pick_sample["valid"])
        self.assertFalse(pick_sample["fallback_used"])
        self.assertGreaterEqual(
            pick_sample["radius_m"],
            settings.base_goal.arm_workspace_min_xy_radius_m,
        )
        self.assertLessEqual(
            pick_sample["radius_m"],
            settings.base_goal.arm_workspace_max_xy_radius_m,
        )
        for sample in (pick_sample, report["place"]):
            self.assertEqual(sample["yaw_policy"], "preserve_nominal_base_goal_yaw")
            self.assertAlmostEqual(sample["yaw_noise_rad"], 0.0)
            self.assertAlmostEqual(
                sample["sampled_base_goal_xyyaw"][2],
                sample["nominal_base_goal_xyyaw"][2],
            )
            self.assertIn("sampled_base_goal_xyyaw", sample)
            self.assertIn("sampled_arm_base_xy", sample)
            self.assertIn("theta_rad", sample)
        place_sample = report["place"]
        self.assertTrue(place_sample["valid"])
        self.assertFalse(place_sample["fallback_used"])
        self.assertEqual(place_sample["mode"], "place_rectangular_offset_xy")
        self.assertIn("robot_base_radius_m", place_sample)
        self.assertLessEqual(
            place_sample["robot_base_radius_m"],
            settings.base_goal.place_robot_base_max_xy_radius_m,
        )
        self.assertGreaterEqual(
            place_sample["offset_xy_m"][0],
            settings.base_goal.place_offset_x_range_m[0],
        )
        self.assertLessEqual(
            place_sample["offset_xy_m"][0],
            settings.base_goal.place_offset_x_range_m[1],
        )
        self.assertGreaterEqual(
            place_sample["offset_xy_m"][1],
            settings.base_goal.place_offset_y_range_m[0],
        )
        self.assertLessEqual(
            place_sample["offset_xy_m"][1],
            settings.base_goal.place_offset_y_range_m[1],
        )
        self.assertFalse(place_sample["fallback_used"])
        self.assertLess(
            place_sample["sampled_base_goal_xyyaw"][1],
            place_sample["target_xy"][1],
        )

    def test_base_goal_randomization_fallback_uses_fixed_goal(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        raw = copy.deepcopy(base.raw_task)
        raw["nav_map"] = "missing_map_for_unit_test.json"
        base = episode_spec_from_dict(raw)
        settings = RandomizationSettings(
            enabled=False,
            base_goal=BaseGoalRandomizationSettings(
                enabled=True,
                pick_radius_min_m=0.10,
                pick_radius_max_m=0.12,
                place_radius_min_m=0.10,
                place_radius_max_m=0.12,
                max_goal_sample_attempts=3,
                fallback_to_fixed_offset=True,
            ),
        )

        episode = prepare_episode_spec(base, episode_id=4, seed=13, settings=settings)

        self.assertEqual(episode.pick_goal, base.pick_goal)
        self.assertEqual(episode.place_goal, base.place_goal)
        report = episode.raw_task["randomization"]["base_goal_randomization"]
        self.assertTrue(report["pick"]["fallback_used"])
        self.assertTrue(report["place"]["fallback_used"])
        self.assertEqual(report["place"]["mode"], "fallback_fixed_goal")

    def test_place_rectangular_base_goal_rejects_arm_workspace_outliers(self) -> None:
        base = JsonTaskProvider().load(TASK_PATH)
        raw = copy.deepcopy(base.raw_task)
        raw["nav_map"] = "missing_map_for_unit_test.json"
        base = episode_spec_from_dict(raw)
        settings = RandomizationSettings(
            enabled=False,
            base_goal=BaseGoalRandomizationSettings(
                enabled=True,
                place_offset_x_range_m=(1.0, 1.1),
                place_offset_y_range_m=(-0.01, 0.0),
                max_goal_sample_attempts=3,
                fallback_to_fixed_offset=False,
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "failed_to_sample_valid_place_base_goal",
        ):
            prepare_episode_spec(base, episode_id=5, seed=14, settings=settings)

    def test_episode_seed_increments_with_episode_index(self) -> None:
        config = FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=PROJECT_ROOT / "outputs/test",
            num_episodes=3,
            seed=100,
            dry_run=True,
        )

        self.assertEqual([config.episode_seed(index) for index in range(3)], [100, 101, 102])


if __name__ == "__main__":
    unittest.main()
