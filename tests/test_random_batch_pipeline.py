"""Tests for randomized batch pipeline command assembly."""

from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline.run_random_nav_pick_batch import _parse_args, _pipeline_command


class RandomBatchPipelineTest(unittest.TestCase):
    def test_batch_defaults_match_apple_table_region(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_batch.py", "--num-episodes", "1"]):
            args = _parse_args()

        self.assertEqual(tuple(args.table_x_range), (0.90, 0.96))
        self.assertEqual(tuple(args.table_y_range), (0.9, 1.6))
        self.assertAlmostEqual(args.table_z, 0.82)
        self.assertAlmostEqual(args.object_z_offset, 0.0)
        self.assertEqual(tuple(args.standoff_candidates), (0.50, 0.55, 0.60))
        self.assertEqual(args.base_goal_mode, "object-offset")
        self.assertEqual(tuple(args.base_goal_offset_xy), (0.35, 0.0))
        self.assertTrue(args.ignore_goal_yaw)
        self.assertFalse(args.precompute_nav_first)

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


if __name__ == "__main__":
    unittest.main()
