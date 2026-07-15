"""Full-physics episode 随机化与可视化描述测试。"""

from __future__ import annotations

import copy
import math
import os
import unittest
from pathlib import Path
from unittest import mock

from scripts.pipeline.run_full_physics_pipeline import _build_parser, _keep_gui_open
from source.diagnostics import randomization_debug_spec
from source.pipeline import (
    BaseGoalRandomizationSettings,
    FullPhysicsConfig,
    RandomizationSettings,
)
from source.pipeline.factory import (
    _manipulation_settings_for_episode,
    _navigation_settings_for_episode,
)
from source.tasks import JsonTaskProvider, episode_spec_from_dict, prepare_episode_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
LIANGZHU_TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json"


class FullPhysicsRandomizationTest(unittest.TestCase):
    def test_liangzhu_task_requires_precise_stable_navigation_handoff(self) -> None:
        episode = JsonTaskProvider().load(LIANGZHU_TASK_PATH)
        settings = _navigation_settings_for_episode(
            FullPhysicsConfig(
                task_json=LIANGZHU_TASK_PATH,
                output_dir=Path("/tmp/test-liangzhu-navigation-settings"),
            ).navigation,
            episode,
        )

        # 10 cm keeps the floor object inside the verified front-arm workspace while
        # avoiding terminal-controller stalls when the locomotion policy cannot realize
        # very small mixed vx/vy corrections.
        self.assertEqual(settings.final_position_tolerance, 0.1)
        self.assertEqual(settings.stall_min_forward_command, 0.03)
        self.assertEqual(settings.final_yaw_tolerance, 0.15)
        self.assertTrue(settings.require_yaw_alignment)
        self.assertTrue(settings.require_stable_base)

        manipulation = _manipulation_settings_for_episode(
            FullPhysicsConfig(
                task_json=LIANGZHU_TASK_PATH,
                output_dir=Path("/tmp/test-liangzhu-navigation-settings"),
            ).manipulation,
            episode,
        )
        self.assertTrue(manipulation.reuse_pick_grasp_orientation_for_place)

    def test_cli_randomization_and_visualization_defaults(self) -> None:
        args = _build_parser().parse_args(
            ["--task-json", str(TASK_PATH), "--dry-run"]
        )

        self.assertTrue(args.randomize_task)
        self.assertTrue(args.randomize_base_goal)
        self.assertFalse(args.show_randomization_debug)
        self.assertFalse(args.keep_window_open)
        self.assertEqual(args.navigation_visual_mode, "full")
        self.assertEqual(args.overview_camera_mode, "fixed")
        self.assertEqual(args.overview_camera_prim_path, "/World/overview")

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

    def test_liangzhu_forward_sector_randomization_is_synchronized(self) -> None:
        """良渚布局必须同步重建起点、物体、地垫、导航点和碰撞代理。"""

        support = {
            "collision_ply": "/tmp/unit-test-collision.ply",
            "collision_ply_env": "LIANGZHU_COLLISION_PLY",
            "collision_ply_expected_sha256": "unit-test",
            "query_ceiling_z": 2.0,
            "cola": {
                "xy": [0.0, 0.0],
                "z": -0.15,
                "face_index": 7,
                "normal_xyz": [0.0, 0.0, 1.0],
            },
            "mat": {
                "probes": [],
                "floor_min_z": -0.15,
                "floor_max_z": -0.15,
                "height_variation_m": 0.0,
                "max_height_variation_m": 0.006,
            },
            "geometry_verified": True,
        }
        base = JsonTaskProvider().load(LIANGZHU_TASK_PATH)
        settings = RandomizationSettings(
            enabled=True,
            collision_ply_path=Path("/tmp/unit-test-collision.ply"),
            base_goal=BaseGoalRandomizationSettings(enabled=True),
        )

        with mock.patch(
            "source.tasks.forward_sector_randomization._apply_support_geometry",
            return_value=support,
        ) as support_probe:
            first = prepare_episode_spec(
                base,
                episode_id=3,
                seed=41,
                settings=settings,
            )
            repeated = prepare_episode_spec(
                base,
                episode_id=3,
                seed=41,
                settings=settings,
            )
            different = prepare_episode_spec(
                base,
                episode_id=4,
                seed=42,
                settings=settings,
            )

        self.assertEqual(
            support_probe.call_args_list[0].kwargs["config"]["collision_ply_path"],
            "/tmp/unit-test-collision.ply",
        )

        self.assertEqual(first.raw_task, repeated.raw_task)
        self.assertNotEqual(first.raw_task, different.raw_task)
        task = first.raw_task
        sample = task["randomization"]["sample"]
        config = task["randomization"]["forward_sector"]
        self.assertEqual(
            [task["start"][key] for key in ("x", "y", "z")],
            config["robot_translate_xyz"],
        )
        self.assertGreaterEqual(
            math.degrees(task["start"]["yaw"]),
            config["robot_yaw_range_deg"][0],
        )
        self.assertLessEqual(
            math.degrees(task["start"]["yaw"]),
            config["robot_yaw_range_deg"][1],
        )
        for target_name in ("cola", "mat"):
            self.assertLessEqual(
                abs(math.degrees(sample[target_name]["relative_angle_rad"])),
                config["sector_half_angle_deg"],
            )
        self.assertFalse(sample["cola_overlaps_mat_footprint"])
        self.assertGreaterEqual(
            sample["cola_mat_footprint_clearance_m"],
            sample["required_cola_mat_clearance_m"],
        )
        self.assertTrue(all(task["randomization"]["synchronization"].values()))
        self.assertEqual(
            task["pick"]["base_goal"]["x"],
            sample["pick_base_goal"]["x"],
        )
        self.assertEqual(
            task["place"]["base_goal"]["x"],
            sample["place_base_goal"]["x"],
        )
        for phase, target_name in (("pick", "cola"), ("place", "mat")):
            goal = task[phase]["base_goal"]
            target = sample[target_name]
            target_bearing = math.atan2(
                target["y"] - goal["y"],
                target["x"] - goal["x"],
            )
            target_bearing_base = math.atan2(
                math.sin(target_bearing - goal["yaw"]),
                math.cos(target_bearing - goal["yaw"]),
            )
            self.assertAlmostEqual(target_bearing_base, 0.0)
            self.assertEqual(goal["target_region_in_base"], "front")
            self.assertEqual(goal["final_alignment_mode"], "face_target")
            self.assertAlmostEqual(
                sample[f"{phase}_base_goal"]["target_bearing_base_rad"],
                0.0,
            )
        self.assertEqual(sample["place_base_goal"]["approach_origin"], "pick_base_goal")
        pick_to_mat_bearing = math.atan2(
            sample["mat"]["y"] - sample["pick_base_goal"]["y"],
            sample["mat"]["x"] - sample["pick_base_goal"]["x"],
        )
        self.assertAlmostEqual(
            sample["place_base_goal"]["approach_bearing_world_rad"],
            pick_to_mat_bearing,
        )
        self.assertEqual(config["base_approach_angle_noise_deg"], 0.0)
        self.assertEqual(
            task["place"]["receptacle_pose_world"],
            sample["mat_root_pose_world"],
        )
        self.assertEqual(
            task["place"]["curobo_world_collision"]["cuboids_world"][0][
                "center_xyz"
            ][:2],
            [sample["mat"]["x"], sample["mat"]["y"]],
        )
        mat_proxy_source = task["place"]["curobo_world_collision"][
            "cuboids_world"
        ][0]["source"]
        self.assertEqual(len(mat_proxy_source["world_bbox_min_xyz"]), 3)
        self.assertEqual(len(mat_proxy_source["world_bbox_max_xyz"]), 3)
        self.assertAlmostEqual(
            mat_proxy_source["world_bbox_max_xyz"][2],
            task["place"]["placement_region"]["z_surface"],
        )
        debug_spec = randomization_debug_spec(task)
        self.assertEqual(
            debug_spec["forward_sector"]["origin_xyz"],
            tuple(config["robot_translate_xyz"]),
        )
        self.assertEqual(
            debug_spec["forward_sector"]["robot_yaw_rad"],
            sample["robot_yaw_rad"],
        )

    def test_liangzhu_randomization_requires_collision_ply_configuration(self) -> None:
        """缺少碰撞 PLY 是配置错误，不能通过重复采样静默掩盖。"""

        base = JsonTaskProvider().load(LIANGZHU_TASK_PATH)
        with mock.patch.dict(os.environ, {"LIANGZHU_COLLISION_PLY": ""}):
            with self.assertRaisesRegex(ValueError, "LIANGZHU_COLLISION_PLY"):
                prepare_episode_spec(
                    base,
                    episode_id=3,
                    seed=41,
                    settings=RandomizationSettings(enabled=True),
                )

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
