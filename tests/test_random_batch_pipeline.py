"""Tests for randomized batch pipeline command assembly."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from source.data.random_task import RandomTaskGenerationError
from scripts.pipeline.run_random_nav_pick_batch import _parse_args, _pipeline_command, _validate_apple_spawn_stability
from scripts.pipeline import run_random_nav_pick_place_batch as pick_place_batch


class RandomBatchPipelineTest(unittest.TestCase):
    def test_batch_defaults_match_apple_table_region(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_batch.py", "--num-episodes", "1"]):
            args = _parse_args()

        self.assertEqual(tuple(args.table_x_range), (0.88, 0.93))
        self.assertEqual(tuple(args.table_y_range), (0.9, 1.6))
        self.assertAlmostEqual(args.table_z, 0.82)
        self.assertAlmostEqual(args.object_z_offset, 0.0)
        self.assertAlmostEqual(args.object_fixed_z, 0.81653)
        self.assertAlmostEqual(args.object_fixed_roll, -2.524)
        self.assertAlmostEqual(args.object_fixed_pitch, -7.822)
        self.assertAlmostEqual(args.object_fixed_yaw, -0.181)
        self.assertEqual(args.object_fixed_rpy_unit, "deg")
        self.assertFalse(args.randomize_object_yaw)
        self.assertAlmostEqual(args.edge_min_clearance, 0.03)
        self.assertAlmostEqual(args.object_support_clearance, 0.0)
        self.assertEqual(tuple(args.standoff_candidates), (0.50, 0.55, 0.60))
        self.assertEqual(args.base_goal_mode, "object-offset")
        self.assertEqual(tuple(args.base_goal_offset_xy), (0.28, -0.08))
        self.assertAlmostEqual(args.clearance_radius, 0.20)
        self.assertTrue(args.ignore_goal_yaw)
        self.assertFalse(args.precompute_nav_first)

    def test_batch_rejects_unstable_apple_xy_support_when_overridden(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_random_nav_pick_batch.py",
                "--num-episodes",
                "1",
                "--edge-min-clearance",
                "0.02",
            ],
        ):
            args = _parse_args()

        with self.assertRaisesRegex(ValueError, "too small"):
            _validate_apple_spawn_stability(args)

        with patch(
            "sys.argv",
            [
                "run_random_nav_pick_batch.py",
                "--num-episodes",
                "1",
                "--table-x-range",
                "0.90",
                "0.94",
                "--object-support-clearance",
                "0.03",
            ],
        ):
            args = _parse_args()

        with self.assertRaisesRegex(ValueError, "too narrow"):
            _validate_apple_spawn_stability(args)

    def test_batch_command_forwards_video_replay_and_planner_options(self) -> None:
        args = Namespace(
            pipeline_python="/python",
            task="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
            checkpoint="checkpoints/go2_x5/flat/model_8500.pt",
            isaaclab_launcher="/isaaclab.sh",
            isaac_python="/isaac-python",
            max_nav_steps=3000,
            goal_yaw_tolerance=0.15,
            terminal_yaw_tolerance=0.08,
            final_yaw_tolerance_margin=0.07,
            yaw_align_start_distance=0.5,
            yaw_align_min_vy=0.18,
            yaw_align_lateral_kp=0.9,
            yaw_align_min_wz=0.4,
            yaw_align_max_wz=0.6,
            terminal_yaw_slowdown_max_wz=0.42,
            terminal_recovery_steps=90,
            terminal_recovery_yaw_max_wz=0.32,
            terminal_yaw_polish_vx=0.08,
            terminal_yaw_polish_min_wz=0.45,
            terminal_yaw_polish_max_wz=0.55,
            yaw_settle_max_wz=0.55,
            base_stable_linear_tolerance=0.06,
            base_stable_angular_tolerance=0.20,
            nav_map=None,
            handoff_clearance_radius=0.20,
            clearance_radius=0.20,
            base_goal_mode="object-offset",
            nav_headless=True,
            nav_only=False,
            handoff_smoke_only=False,
            use_planner_server=True,
            auto_start_planner_server=True,
            restart_planner_server=True,
            planner_server_log="/tmp/planner.log",
            planner_server_start_timeout_s=120.0,
            replay_nav_before_grasp=True,
            replay_nav_real_time=True,
            replay_nav_speed=1.25,
            demo_visuals=True,
            follow_camera_mode="stage",
            viewport_camera_prim="/World/Camera_main",
            keep_window_open=False,
            show_grasp_trajectory=False,
            allow_retreat_success=False,
            legacy_side_retreat=False,
            side_retreat_only=True,
            side_grasp_fallback_retreat=False,
        )

        command = _pipeline_command(
            args,
            task_json=Path("/tmp/task.json"),
            dataset_dir=Path("/tmp/dataset"),
            nav_result=Path("/tmp/nav_result.json"),
            handoff_report=Path("/tmp/handoff.json"),
        )

        self.assertIn("--nav-headless", command)
        self.assertIn("--replay-nav-before-grasp", command)
        self.assertIn("--replay-nav-real-time", command)
        self.assertIn("--demo-visuals", command)
        self.assertIn("--follow-camera-mode", command)
        self.assertIn("stage", command)
        self.assertIn("--viewport-camera-prim", command)
        self.assertIn("/World/Camera_main", command)
        self.assertIn("--auto-start-planner-server", command)
        self.assertIn("--restart-planner-server", command)
        self.assertIn("--no-keep-window-open", command)
        self.assertIn("--side-retreat-only", command)
        yaw_settle_index = command.index("--yaw-settle-max-wz")
        self.assertEqual(command[yaw_settle_index + 1], "0.55")
        self.assertEqual(command[command.index("--inflate-radius") + 1], "0.2")
        self.assertEqual(command[command.index("--local-clearance-radius") + 1], "0.2")

    def test_batch_command_can_build_grasp_only_replay_phase(self) -> None:
        args = Namespace(
            pipeline_python="/python",
            task="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
            checkpoint="checkpoints/go2_x5/flat/model_8500.pt",
            isaaclab_launcher="/isaaclab.sh",
            isaac_python="/isaac-python",
            max_nav_steps=3000,
            goal_yaw_tolerance=0.15,
            terminal_yaw_tolerance=0.08,
            final_yaw_tolerance_margin=0.07,
            yaw_align_start_distance=0.5,
            yaw_align_min_vy=0.18,
            yaw_align_lateral_kp=0.9,
            yaw_align_min_wz=0.4,
            yaw_align_max_wz=0.6,
            terminal_yaw_slowdown_max_wz=0.42,
            terminal_recovery_steps=90,
            terminal_recovery_yaw_max_wz=0.32,
            terminal_yaw_polish_vx=0.08,
            terminal_yaw_polish_min_wz=0.45,
            terminal_yaw_polish_max_wz=0.55,
            yaw_settle_max_wz=0.55,
            base_stable_linear_tolerance=0.06,
            base_stable_angular_tolerance=0.20,
            nav_map=None,
            handoff_clearance_radius=0.20,
            clearance_radius=0.20,
            base_goal_mode="object-offset",
            nav_headless=True,
            nav_only=True,
            handoff_smoke_only=False,
            use_planner_server=False,
            auto_start_planner_server=False,
            restart_planner_server=False,
            planner_server_log="/tmp/planner.log",
            planner_server_start_timeout_s=120.0,
            replay_nav_before_grasp=True,
            replay_nav_real_time=False,
            replay_nav_speed=1.0,
            demo_visuals=True,
            follow_camera_mode="stage",
            viewport_camera_prim="/World/Camera_main",
            keep_window_open=False,
            show_grasp_trajectory=False,
            allow_retreat_success=False,
            legacy_side_retreat=False,
            side_retreat_only=True,
            side_grasp_fallback_retreat=False,
        )

        command = _pipeline_command(
            args,
            task_json=Path("/tmp/task.json"),
            dataset_dir=Path("/tmp/dataset"),
            nav_result=Path("/tmp/nav_result.json"),
            handoff_report=Path("/tmp/handoff.json"),
            nav_only=False,
            grasp_only=True,
        )

        self.assertIn("--grasp-only", command)
        self.assertNotIn("--nav-only", command)
        self.assertIn("--replay-nav-before-grasp", command)
        self.assertEqual(command[command.index("--inflate-radius") + 1], "0.2")

    def test_pick_place_batch_defaults_match_stable_pick_batch_base_goal(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py"]):
            args = pick_place_batch._parse_args()

        self.assertEqual(args.base_goal_mode, "object-offset")
        self.assertEqual(tuple(args.base_goal_offset_xy), (0.28, -0.08))
        self.assertAlmostEqual(args.clearance_radius, 0.20)
        self.assertTrue(args.ignore_goal_yaw)
        self.assertAlmostEqual(args.terminal_yaw_polish_min_wz, 0.45)
        self.assertAlmostEqual(args.terminal_yaw_polish_max_wz, 0.55)
        self.assertEqual(args.manipulation_backend, "single-stage-07")
        self.assertEqual(args.single_stage_put_mode, "arm-place")
        self.assertEqual(args.single_stage_carry_mode, "logical")

    def test_pick_place_batch_single_stage_07_command_forwards_options(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_random_nav_pick_place_batch.py",
                "--demo-visuals",
                "--viewport-camera-prim",
                "/World/Camera_main",
                "--no-keep-window-open",
                "--use-planner-server",
                "--side-retreat-only",
                "--allow-retreat-success",
                "--legacy-side-retreat",
                "--side-grasp-fallback-retreat",
                "--show-grasp-trajectory",
            ],
        ):
            args = pick_place_batch._parse_args()

        command = pick_place_batch._single_stage_07_command(
            args,
            task_nav_to_pick=Path("/tmp/task_nav_to_pick.json"),
            nav_pick_result=Path("/tmp/nav_pick_result.json"),
            task_nav_to_place=Path("/tmp/task_nav_to_place.json"),
            nav_place_result=Path("/tmp/nav_place_result.json"),
            episode_dir=Path("/tmp/episode_0000"),
        )

        self.assertIn("07_run_pick_put_demo_from_nav_results.py", command[1])
        self.assertEqual(command[command.index("--task-nav-to-pick") + 1], "/tmp/task_nav_to_pick.json")
        self.assertEqual(command[command.index("--nav-pick-result") + 1], "/tmp/nav_pick_result.json")
        self.assertEqual(command[command.index("--task-nav-to-place") + 1], "/tmp/task_nav_to_place.json")
        self.assertEqual(command[command.index("--nav-place-result") + 1], "/tmp/nav_place_result.json")
        self.assertEqual(command[command.index("--put-mode") + 1], "arm-place")
        self.assertEqual(command[command.index("--carry-mode") + 1], "logical")
        self.assertIn("--demo-visuals", command)
        self.assertIn("--viewport-camera-prim", command)
        self.assertIn("--no-keep-window-open", command)
        self.assertIn("--use-planner-server", command)
        self.assertIn("--side-retreat-only", command)
        self.assertIn("--allow-retreat-success", command)
        self.assertIn("--legacy-side-retreat", command)
        self.assertIn("--side-grasp-fallback-retreat", command)
        self.assertIn("--show-grasp-trajectory", command)

    def test_pick_place_batch_injects_place_template_for_pick_only_task(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py"]):
            args = pick_place_batch._parse_args()

        task = {
            "pick": {"base_goal": {"x": 1.0, "y": 1.0, "yaw": 0.0}},
            "place": {"enabled": False, "base_goal": None, "place_pose_world": None},
            "randomization": {},
        }

        patched = pick_place_batch._ensure_place_goal(task, args)

        self.assertTrue(patched["place"]["enabled"])
        self.assertIsNotNone(patched["place"]["base_goal"])
        self.assertIsNotNone(patched["place"]["place_pose_world"])
        self.assertTrue(patched["randomization"]["place_template"]["enabled"])

    def test_pick_place_batch_command_forwards_stable_terminal_yaw_options(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py"]):
            args = pick_place_batch._parse_args()

        command = pick_place_batch._pipeline_command(
            args,
            task_json=Path("/tmp/task.json"),
            dataset_dir=Path("/tmp/dataset"),
            nav_result=Path("/tmp/nav_result.json"),
            handoff_report=Path("/tmp/handoff.json"),
            nav_only=True,
        )

        self.assertIn("--nav-only", command)
        self.assertIn("--yaw-align-vx", command)
        self.assertEqual(command[command.index("--yaw-align-vx") + 1], "0.35")
        self.assertIn("--yaw-align-lateral-deadband", command)
        self.assertIn("--terminal-yaw-polish-min-wz", command)
        self.assertEqual(command[command.index("--terminal-yaw-polish-min-wz") + 1], "0.45")
        self.assertIn("--yaw-settle-max-wz", command)
        self.assertEqual(command[command.index("--inflate-radius") + 1], "0.2")
        self.assertEqual(command[command.index("--local-clearance-radius") + 1], "0.2")

    def test_pick_place_batch_place_reconstruction_is_explicit_opt_in(self) -> None:
        task = {"scene_usd": "source/scene/839920_go2_x5.usd"}
        with tempfile.TemporaryDirectory() as tmpdir:
            task_json = Path(tmpdir) / "task.json"
            task_json.write_text(json.dumps(task), encoding="utf-8")
            with patch("sys.argv", ["run_random_nav_pick_place_batch.py"]):
                args = pick_place_batch._parse_args()

            command = pick_place_batch._place_command(
                args,
                task_json=task_json,
                dataset_dir=Path(tmpdir) / "place",
                nav_result=Path(tmpdir) / "nav_place_result.json",
                place_result=Path(tmpdir) / "place_result.json",
                handoff_report=Path(tmpdir) / "place_handoff_report.json",
            )
            self.assertNotIn("--mvp-reconstruct-place", command)

            with patch("sys.argv", ["run_random_nav_pick_place_batch.py", "--mvp-reconstruct-place"]):
                args = pick_place_batch._parse_args()
            command = pick_place_batch._place_command(
                args,
                task_json=task_json,
                dataset_dir=Path(tmpdir) / "place",
                nav_result=Path(tmpdir) / "nav_place_result.json",
                place_result=Path(tmpdir) / "place_result.json",
                handoff_report=Path(tmpdir) / "place_handoff_report.json",
            )
            self.assertIn("--mvp-reconstruct-place", command)

    def test_pick_place_alignment_report_flags_far_final_base(self) -> None:
        task = {
            "pick": {
                "object_pose_world": {"x": 1.0, "y": 1.0, "z": 0.81653},
                "base_goal": {"x": 1.28, "y": 0.92, "yaw": 0.0},
            }
        }
        nav_result = {
            "success": True,
            "final_base_pose_world": {"x": 0.2, "y": 0.2, "yaw": 0.0},
            "final_goal_distance": 0.0,
            "yaw_error": 0.0,
            "final_position_reached": True,
            "final_yaw_aligned": True,
            "base_stable": True,
        }

        report = pick_place_batch._nav_pick_alignment_report(task, nav_result)

        self.assertTrue(report["available"])
        self.assertGreater(report["final_base_to_object_distance_m"], 0.45)
        self.assertEqual(report["warning"], "final_base_far_from_object_for_arm_planning")

    def test_pick_place_generation_retries_object_offset_with_handoff_clearance(self) -> None:
        with patch(
            "sys.argv",
            [
                "run_random_nav_pick_place_batch.py",
                "--clearance-radius",
                "0.25",
                "--handoff-clearance-radius",
                "0.20",
            ],
        ):
            args = pick_place_batch._parse_args()

        returned_task = {
            "pick": {
                "object_pose_world": {"x": 0.9, "y": 1.1, "z": 0.81653},
                "base_goal": {"x": 1.18, "y": 1.02, "yaw": 0.0},
            },
            "randomization": {"base_goal_generation": {"clearance_radius": 0.20}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "task.json"
            with patch.object(
                pick_place_batch,
                "write_random_pick_task",
                side_effect=[RandomTaskGenerationError("clearance_below_0.250m"), returned_task],
            ) as write_mock:
                task = pick_place_batch._write_random_pick_task_with_clearance_fallback(
                    args,
                    generated_task_json=output_path,
                    episode_seed=7,
                    spawn_region=pick_place_batch.SpawnRegion(0.88, 0.93, 0.9, 1.6, 0.82),
                )

        self.assertEqual(write_mock.call_count, 2)
        first_call = write_mock.call_args_list[0].kwargs
        second_call = write_mock.call_args_list[1].kwargs
        self.assertAlmostEqual(first_call["clearance_radius"], 0.25)
        self.assertAlmostEqual(second_call["clearance_radius"], 0.20)
        fallback = task["randomization"]["generation_clearance_fallback"]
        self.assertTrue(fallback["enabled"])
        self.assertAlmostEqual(fallback["requested_clearance_radius"], 0.25)
        self.assertAlmostEqual(fallback["fallback_clearance_radius"], 0.20)


if __name__ == "__main__":
    unittest.main()
