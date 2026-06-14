"""Full-physics episode 随机化与可视化描述测试。"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from scripts.pipeline.run_full_physics_pipeline import _build_parser, _keep_gui_open
from source.diagnostics import randomization_debug_spec
from source.pipeline import FullPhysicsConfig, RandomizationSettings
from source.tasks import JsonTaskProvider, prepare_episode_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"


class FullPhysicsRandomizationTest(unittest.TestCase):
    def test_cli_randomization_and_visualization_defaults(self) -> None:
        args = _build_parser().parse_args(
            ["--task-json", str(TASK_PATH), "--dry-run"]
        )

        self.assertTrue(args.randomize_task)
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
        settings = RandomizationSettings(enabled=True, show_debug_region=True)

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
        settings = RandomizationSettings(show_debug_region=True)
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
        self.assertEqual(spec["pick"]["xy_range"], ((0.83, 0.93), (1.0, 1.5)))
        self.assertEqual(
            spec["place"]["xy_range"],
            ((0.65285, 0.75285), (5.00337, 5.50337)),
        )

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
