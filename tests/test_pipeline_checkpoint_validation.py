"""Tests for navigation coordinator checkpoint preflight validation."""

from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.pipeline.run_nav_then_pick import (
    _apply_derived_defaults,
    _apply_preset_defaults,
    _default_checkpoint_path,
    _nav_command,
    _pick_script_editor_command,
    _standalone_pick_command,
    _validate_checkpoint,
)


class PipelineCheckpointValidationTest(unittest.TestCase):
    def test_existing_local_checkpoint_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model_8500.pt"
            checkpoint.touch()
            self.assertEqual(_validate_checkpoint(str(checkpoint)), str(checkpoint.resolve()))

    def test_missing_checkpoint_fails_before_isaaclab_launch(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Do not use the documentation placeholder"):
            _validate_checkpoint("/你的实际路径/model_8500.pt")

    def test_env_checkpoint_is_default(self) -> None:
        with mock.patch.dict(os.environ, {"GO2_X5_CHECKPOINT": "/tmp/custom_go2_x5.pt"}):
            self.assertEqual(_default_checkpoint_path(), "/tmp/custom_go2_x5.pt")

    def test_apple_fast_preset_applies_defaults_without_overriding_explicit_values(self) -> None:
        args = Namespace(
            preset="apple-fast",
            task_json="custom_task.json",
            dataset_dir=None,
            nav_result="/tmp/go2_x5_nav_result.json",
            max_nav_steps=5000,
            lookahead_distance=0.35,
            prediction_horizon=0.90,
            goal_tolerance=0.15,
            goal_yaw_tolerance=0.15,
            terminal_position_tolerance=0.10,
            terminal_yaw_tolerance=0.10,
            final_goal_tolerance_margin=0.01,
            final_yaw_tolerance_margin=0.01,
            yaw_align_start_distance=0.65,
            yaw_align_vx=0.16,
            yaw_align_max_vx=0.35,
            yaw_align_position_kp=0.8,
            yaw_align_max_vy=0.18,
            yaw_align_min_vy=0.0,
            yaw_align_lateral_kp=0.8,
            yaw_align_lateral_deadband=0.03,
            yaw_align_min_wz=0.75,
            yaw_align_max_wz=1.00,
            terminal_yaw_polish_vx=0.04,
            terminal_yaw_polish_min_wz=0.40,
            terminal_yaw_polish_max_wz=0.45,
            settle_steps=120,
            yaw_settle_stable_steps=20,
            yaw_settle_max_wz=0.35,
            brisk_nav=False,
            fast_dwa=False,
        )
        with mock.patch("scripts.pipeline.run_nav_then_pick.RAW_CLI_ARGS", ["--task-json", "custom_task.json"]):
            _apply_preset_defaults(args)

        self.assertEqual(args.task_json, "custom_task.json")
        self.assertEqual(args.dataset_dir, "/tmp/nav_pick_apple_fast")
        self.assertEqual(args.nav_result, "/tmp/go2_x5_nav_pick_apple_fast_result.json")
        self.assertEqual(args.max_nav_steps, 3000)
        self.assertEqual(args.goal_tolerance, 0.15)
        self.assertEqual(args.terminal_position_tolerance, 0.08)
        self.assertEqual(args.final_goal_tolerance_margin, 0.03)
        self.assertEqual(args.final_yaw_tolerance_margin, 0.07)
        self.assertEqual(args.yaw_align_vx, 0.35)
        self.assertEqual(args.yaw_align_max_vx, 0.60)
        self.assertEqual(args.yaw_align_lateral_kp, 0.9)
        self.assertEqual(args.yaw_align_lateral_deadband, 0.015)
        self.assertEqual(args.yaw_align_max_wz, 0.60)
        self.assertEqual(args.terminal_yaw_polish_vx, 0.08)
        self.assertEqual(args.terminal_yaw_polish_min_wz, 0.45)
        self.assertEqual(args.terminal_yaw_polish_max_wz, 0.55)
        self.assertEqual(args.yaw_settle_stable_steps, 15)
        self.assertEqual(args.yaw_settle_max_wz, 0.25)
        self.assertTrue(args.brisk_nav)
        self.assertTrue(args.fast_dwa)

    def test_video_replay_mode_defaults_to_headless_nav(self) -> None:
        args = Namespace(demo_visuals=True, replay_nav_before_grasp=True, nav_headless=False, keep_window_open=None)

        with mock.patch("scripts.pipeline.run_nav_then_pick.RAW_CLI_ARGS", ["--demo-visuals", "--replay-nav-before-grasp"]):
            _apply_derived_defaults(args)

        self.assertTrue(args.nav_headless)
        self.assertTrue(args.keep_window_open)

    def test_side_retreat_only_enables_existing_side_retreat_policy(self) -> None:
        args = Namespace(
            demo_visuals=False,
            replay_nav_before_grasp=False,
            nav_headless=False,
            keep_window_open=None,
            side_retreat_only=True,
            legacy_side_retreat=False,
            allow_retreat_success=False,
        )

        with mock.patch("scripts.pipeline.run_nav_then_pick.RAW_CLI_ARGS", ["--side-retreat-only"]):
            _apply_derived_defaults(args)

        self.assertTrue(args.legacy_side_retreat)
        self.assertTrue(args.allow_retreat_success)
        self.assertFalse(args.keep_window_open)

    def test_nav_command_forwards_visual_and_speed_options(self) -> None:
        args = Namespace(
            isaaclab_launcher="/opt/IsaacLab/isaaclab.sh",
            isaaclab_python="/unused/python",
            task_json="tasks/nav_pick_example.json",
            task="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
            checkpoint="/tmp/model_8500.pt",
            terrain_prim_path="/World/scene_collision",
            ground_height=0.0,
            nav_result="/tmp/go2_x5_nav_result.json",
            handoff_report="/tmp/go2_x5_handoff_report.json",
            nav_headless=False,
            max_nav_steps=3000,
            settle_steps=120,
            stall_window_steps=240,
            stall_min_progress=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.25,
            goal_tolerance=0.15,
            goal_yaw_tolerance=0.15,
            terminal_position_tolerance=0.08,
            terminal_yaw_tolerance=0.08,
            final_goal_tolerance_margin=0.03,
            final_yaw_tolerance_margin=0.03,
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            lookahead_distance=0.35,
            prediction_horizon=0.90,
            max_lin_vel=0.50,
            max_ang_vel=1.00,
            brisk_nav=True,
            fast_dwa=True,
            dwa_linear_samples=3,
            dwa_angular_samples=7,
            dwa_integration_dt=0.05,
            dwa_path_sample_spacing=0.08,
            dwa_path_distance_window=80,
            min_active_lin_vel=0.30,
            near_goal_min_active_lin_vel=0.22,
            close_goal_speed_limit=0.22,
            speed_bias=0.35,
            max_linear_accel=2.5,
            yaw_align_kp=2.0,
            yaw_align_min_wz=0.75,
            yaw_align_max_wz=1.00,
            yaw_align_vx=0.16,
            yaw_align_max_vx=0.35,
            yaw_align_position_kp=0.8,
            yaw_align_max_vy=0.18,
            yaw_align_min_vy=0.0,
            yaw_align_lateral_kp=0.8,
            yaw_align_lateral_deadband=0.03,
            yaw_align_start_distance=0.65,
            yaw_align_activation_yaw_error=0.0,
            yaw_align_allow_reverse=False,
            yaw_align_stall_window_steps=240,
            yaw_align_min_progress=0.08,
            yaw_settle_stable_steps=20,
            yaw_settle_kp=0.8,
            yaw_settle_min_wz=0.0,
            yaw_settle_max_wz=0.35,
            yaw_settle_realign_margin=0.08,
            base_stable_linear_tolerance=0.06,
            base_stable_angular_tolerance=0.20,
            terminal_allow_reverse=True,
            terminal_yaw_slowdown_error=0.65,
            terminal_yaw_slowdown_min_wz=0.20,
            terminal_yaw_slowdown_max_wz=0.45,
            terminal_large_yaw_error=1.0,
            terminal_large_yaw_position_scale=0.45,
            terminal_gait_vx=0.04,
            terminal_recovery_steps=90,
            terminal_recovery_yaw_max_wz=0.35,
            terminal_recovery_gait_vx=0.08,
            terminal_yaw_polish_vx=0.08,
            terminal_yaw_polish_min_wz=0.45,
            terminal_yaw_polish_max_wz=0.55,
            debug_print_every=30,
            scene_usd=None,
            add_nav_ground=False,
            nav_map=None,
            dataset_dir=None,
            no_record=True,
            head_camera=False,
            demo_visuals=False,
            load_visual_scene=True,
            visual_load_mode="sublayer",
            visual_prim_path="/World/gauss",
            follow_camera=True,
            follow_camera_mode="chase",
            viewport_camera_prim="/World/camera_main",
            follow_camera_distance=2.0,
            follow_camera_height=0.7,
            follow_camera_side=0.0,
            fixed_camera_preset="start",
            fixed_camera_close_distance=2.2,
            fixed_camera_close_height=1.35,
            fixed_camera_close_side=-0.75,
            fixed_camera_eye=None,
            fixed_camera_lookat=None,
            hide_nav_collision_visual=True,
            save_replay_trajectory=True,
            replay_sample_every=2,
            replay_output="/tmp/replay.jsonl",
            replay_trajectory_name="trajectory.jsonl",
            replay_nav_before_grasp=False,
            replay_nav_real_time=False,
            replay_nav_speed=1.0,
            replay_include_initial_settle=True,
            profile_dwa=True,
            profile_print_every=60,
            flat_terrain=False,
            disable_sky_light=False,
            debug_command=None,
        )
        command = _nav_command(args, task=None)
        self.assertIn("--load-visual-scene", command)
        self.assertIn("--visual-load-mode", command)
        self.assertIn("sublayer", command)
        self.assertIn("/World/gauss", command)
        self.assertIn("--max-lin-vel", command)
        self.assertIn("0.5", command)
        self.assertIn("--brisk-nav", command)
        self.assertIn("--fast-dwa", command)
        self.assertIn("--terminal-position-tolerance", command)
        self.assertIn("0.08", command)
        self.assertIn("--terminal-yaw-tolerance", command)
        self.assertIn("--final-goal-tolerance-margin", command)
        self.assertIn("--final-yaw-tolerance-margin", command)
        self.assertIn("--dwa-linear-samples", command)
        self.assertIn("3", command)
        self.assertIn("--dwa-angular-samples", command)
        self.assertIn("7", command)
        self.assertIn("--dwa-integration-dt", command)
        self.assertIn("0.05", command)
        self.assertIn("--dwa-path-sample-spacing", command)
        self.assertIn("0.08", command)
        self.assertIn("--dwa-path-distance-window", command)
        self.assertIn("80", command)
        self.assertIn("--min-active-lin-vel", command)
        self.assertIn("0.3", command)
        self.assertIn("--yaw-align-max-wz", command)
        self.assertIn("1.0", command)
        self.assertIn("--yaw-align-min-wz", command)
        self.assertIn("0.75", command)
        self.assertIn("--yaw-align-vx", command)
        self.assertIn("0.16", command)
        self.assertIn("--yaw-align-max-vx", command)
        self.assertIn("0.35", command)
        self.assertIn("--yaw-align-position-kp", command)
        self.assertIn("0.8", command)
        self.assertIn("--yaw-align-max-vy", command)
        self.assertIn("0.18", command)
        self.assertIn("--yaw-align-lateral-kp", command)
        self.assertIn("--yaw-align-lateral-deadband", command)
        self.assertIn("--yaw-align-start-distance", command)
        self.assertIn("0.65", command)
        self.assertIn("--yaw-settle-max-wz", command)
        self.assertIn("0.35", command)
        self.assertIn("--yaw-settle-realign-margin", command)
        self.assertIn("--debug-print-every", command)
        self.assertIn("30", command)
        self.assertIn("--viewport-camera-prim", command)
        self.assertIn("/World/camera_main", command)
        self.assertIn("--hide-nav-collision-visual", command)
        self.assertIn("--save-replay-trajectory", command)
        self.assertIn("--replay-sample-every", command)
        self.assertIn("2", command)
        self.assertIn("--replay-output", command)
        self.assertIn("/tmp/replay.jsonl", command)
        self.assertIn("--replay-include-initial-settle", command)
        self.assertIn("--profile-dwa", command)
        self.assertIn("--profile-print-every", command)

    def test_pick_script_editor_smoke_command_sets_env_flag(self) -> None:
        smoke_command = _pick_script_editor_command(smoke_only=True)
        full_command = _pick_script_editor_command(smoke_only=False)

        self.assertIn('os.environ["GO2_X5_HANDOFF_FORCE_RECORD"] = "1"', smoke_command)
        self.assertIn('os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "1"', smoke_command)
        self.assertIn('os.environ["GO2_X5_HANDOFF_FORCE_RECORD"] = "1"', full_command)
        self.assertIn('os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "0"', full_command)

    def test_standalone_pick_command_is_headless_and_strict_by_default(self) -> None:
        args = Namespace(
            isaac_python="/opt/isaac/python",
            task_json="tasks/nav_pick_example.json",
            scene_usd=None,
            nav_map=None,
            nav_result="/tmp/go2_x5_nav_result.json",
            handoff_report="/tmp/go2_x5_handoff_report.json",
            replay_nav_before_grasp=False,
            replay_nav_real_time=False,
            replay_nav_speed=1.0,
            follow_camera_mode="chase",
            viewport_camera_prim="/World/Camera_main",
            terrain_prim_path="/World/scene_collision",
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            settle_steps=120,
            dataset_dir=None,
            use_planner_server=False,
            handoff_smoke_only=False,
            no_record=False,
            allow_retreat_success=False,
            legacy_side_retreat=False,
            side_retreat_only=False,
            side_grasp_fallback_retreat=False,
            keep_window_open=False,
            show_grasp_trajectory=False,
            grasp_headless=True,
            demo_visuals=False,
        )
        task = SimpleNamespace(scene_usd="source/scene/839920_go2_x5.usd", nav_map="source/scene/nav_maps/839920/map.json")
        command = _standalone_pick_command(args, task)

        self.assertEqual(command[0], "/opt/isaac/python")
        self.assertIn("--headless", command)
        self.assertIn("--require-lift-success", command)
        self.assertIn("--scene-usd", command)
        self.assertIn("--nav-map", command)
        self.assertIn("--handoff-report", command)
        self.assertIn("/tmp/go2_x5_handoff_report.json", command)
        self.assertNotIn("--show-grasp-trajectory", command)

    def test_standalone_pick_command_forwards_replay_and_stage_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            nav_result = Path(tmp_dir) / "nav_result.json"
            replay = Path(tmp_dir) / "trajectory.jsonl"
            replay.touch()
            nav_result.write_text(
                '{"success": true, "replay_trajectory_path": ' + repr(str(replay)).replace("'", '"') + "}",
                encoding="utf-8",
            )
            args = Namespace(
                isaac_python="/opt/isaac/python",
                task_json="tasks/nav_pick_example.json",
                scene_usd=None,
                nav_map=None,
                nav_result=str(nav_result),
                handoff_report="/tmp/go2_x5_handoff_report.json",
                replay_nav_before_grasp=True,
                replay_nav_real_time=True,
                replay_nav_speed=1.25,
                follow_camera_mode="stage",
                viewport_camera_prim="/World/Camera_main",
                terrain_prim_path="/World/scene_collision",
                inflate_radius=0.25,
                local_clearance_radius=0.20,
                settle_steps=120,
                dataset_dir=None,
                use_planner_server=False,
                handoff_smoke_only=False,
                no_record=False,
                allow_retreat_success=False,
                legacy_side_retreat=False,
                side_retreat_only=False,
                side_grasp_fallback_retreat=False,
                keep_window_open=True,
                show_grasp_trajectory=False,
                grasp_headless=True,
                demo_visuals=True,
            )
            task = SimpleNamespace(
                scene_usd="source/scene/839920_go2_x5.usd",
                nav_map="source/scene/nav_maps/839920/map.json",
            )

            command = _standalone_pick_command(args, task)

        self.assertNotIn("--headless", command)
        self.assertIn("--replay-trajectory", command)
        self.assertIn(str(replay.resolve()), command)
        self.assertIn("--replay-real-time", command)
        self.assertIn("--replay-speed", command)
        self.assertIn("1.25", command)
        self.assertIn("--set-viewport-camera", command)
        self.assertIn("--viewport-camera-prim", command)
        self.assertIn("/World/Camera_main", command)
        self.assertIn("--keep-window-open", command)
        self.assertNotIn("--show-grasp-trajectory", command)

    def test_standalone_pick_command_forwards_side_retreat_and_optional_debug_trajectory(self) -> None:
        args = Namespace(
            isaac_python="/opt/isaac/python",
            task_json="tasks/nav_pick_example.json",
            scene_usd=None,
            nav_map=None,
            nav_result="/tmp/go2_x5_nav_result.json",
            handoff_report="/tmp/go2_x5_handoff_report.json",
            replay_nav_before_grasp=False,
            replay_nav_real_time=False,
            replay_nav_speed=1.0,
            follow_camera_mode="chase",
            viewport_camera_prim="/World/Camera_main",
            terrain_prim_path="/World/scene_collision",
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            settle_steps=120,
            dataset_dir=None,
            use_planner_server=True,
            handoff_smoke_only=False,
            no_record=False,
            allow_retreat_success=True,
            legacy_side_retreat=True,
            side_retreat_only=True,
            side_grasp_fallback_retreat=False,
            keep_window_open=False,
            show_grasp_trajectory=True,
            grasp_headless=True,
            demo_visuals=False,
        )
        task = SimpleNamespace(
            scene_usd="source/scene/839920_go2_x5.usd",
            nav_map="source/scene/nav_maps/839920/map.json",
        )

        command = _standalone_pick_command(args, task)

        self.assertIn("--use-planner-server", command)
        self.assertIn("--allow-retreat-success", command)
        self.assertIn("--side-retreat-only", command)
        self.assertIn("--legacy-side-retreat", command)
        self.assertNotIn("--require-lift-success", command)
        self.assertIn("--show-grasp-trajectory", command)

    def test_demo_visuals_uses_fixed_camera_and_visible_grasp(self) -> None:
        args = Namespace(
            isaaclab_launcher="/opt/IsaacLab/isaaclab.sh",
            isaaclab_python="/unused/python",
            task_json="tasks/nav_pick_example.json",
            task="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0",
            checkpoint="/tmp/model_8500.pt",
            terrain_prim_path="/World/scene_collision",
            ground_height=0.0,
            nav_result="/tmp/go2_x5_nav_result.json",
            handoff_report="/tmp/go2_x5_handoff_report.json",
            nav_headless=False,
            max_nav_steps=5000,
            settle_steps=120,
            stall_window_steps=240,
            stall_min_progress=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.25,
            goal_tolerance=0.15,
            goal_yaw_tolerance=0.15,
            terminal_position_tolerance=0.08,
            terminal_yaw_tolerance=0.08,
            final_goal_tolerance_margin=0.03,
            final_yaw_tolerance_margin=0.03,
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            lookahead_distance=0.35,
            prediction_horizon=0.90,
            max_lin_vel=0.50,
            max_ang_vel=1.00,
            brisk_nav=False,
            fast_dwa=False,
            dwa_linear_samples=None,
            dwa_angular_samples=None,
            dwa_integration_dt=None,
            dwa_path_sample_spacing=None,
            dwa_path_distance_window=None,
            min_active_lin_vel=0.30,
            near_goal_min_active_lin_vel=0.22,
            close_goal_speed_limit=0.22,
            speed_bias=0.35,
            max_linear_accel=2.5,
            yaw_align_kp=2.0,
            yaw_align_min_wz=0.75,
            yaw_align_max_wz=1.00,
            yaw_align_vx=0.16,
            yaw_align_max_vx=0.35,
            yaw_align_position_kp=0.8,
            yaw_align_max_vy=0.18,
            yaw_align_min_vy=0.0,
            yaw_align_lateral_kp=0.8,
            yaw_align_lateral_deadband=0.03,
            yaw_align_start_distance=0.65,
            yaw_align_activation_yaw_error=0.0,
            yaw_align_allow_reverse=False,
            yaw_align_stall_window_steps=240,
            yaw_align_min_progress=0.08,
            yaw_settle_stable_steps=20,
            yaw_settle_kp=0.8,
            yaw_settle_min_wz=0.0,
            yaw_settle_max_wz=0.35,
            yaw_settle_realign_margin=0.08,
            base_stable_linear_tolerance=0.06,
            base_stable_angular_tolerance=0.20,
            terminal_allow_reverse=True,
            terminal_yaw_slowdown_error=0.65,
            terminal_yaw_slowdown_min_wz=0.20,
            terminal_yaw_slowdown_max_wz=0.45,
            terminal_large_yaw_error=1.0,
            terminal_large_yaw_position_scale=0.45,
            terminal_gait_vx=0.04,
            terminal_recovery_steps=90,
            terminal_recovery_yaw_max_wz=0.35,
            terminal_recovery_gait_vx=0.08,
            terminal_yaw_polish_vx=0.08,
            terminal_yaw_polish_min_wz=0.45,
            terminal_yaw_polish_max_wz=0.55,
            debug_print_every=30,
            scene_usd=None,
            add_nav_ground=False,
            nav_map=None,
            dataset_dir=None,
            no_record=False,
            head_camera=False,
            demo_visuals=True,
            load_visual_scene=False,
            visual_load_mode="reference",
            visual_prim_path="/World/gauss",
            follow_camera=True,
            follow_camera_mode="chase",
            viewport_camera_prim="/World/camera_main",
            follow_camera_distance=2.4,
            follow_camera_height=0.8,
            follow_camera_side=0.0,
            fixed_camera_preset="start",
            fixed_camera_close_distance=2.2,
            fixed_camera_close_height=1.35,
            fixed_camera_close_side=-0.75,
            fixed_camera_eye=[-4.0, -3.5, 5.0],
            fixed_camera_lookat=[-1.8, 0.5, 0.35],
            hide_nav_collision_visual=None,
            save_replay_trajectory=False,
            replay_sample_every=1,
            replay_output=None,
            replay_trajectory_name="trajectory.jsonl",
            replay_nav_before_grasp=False,
            replay_nav_real_time=False,
            replay_nav_speed=1.0,
            replay_include_initial_settle=False,
            profile_dwa=False,
            profile_print_every=60,
            flat_terrain=False,
            disable_sky_light=False,
            debug_command=None,
            use_planner_server=False,
            handoff_smoke_only=False,
            allow_retreat_success=False,
            legacy_side_retreat=False,
            side_retreat_only=False,
            side_grasp_fallback_retreat=False,
            keep_window_open=True,
            show_grasp_trajectory=False,
            grasp_headless=True,
            isaac_python="/opt/isaac/python",
        )
        task = SimpleNamespace(scene_usd="source/scene/839920_go2_x5.usd", nav_map="source/scene/nav_maps/839920/map.json")

        nav_command = _nav_command(args, task)
        pick_command = _standalone_pick_command(args, task)

        self.assertIn("--load-visual-scene", nav_command)
        self.assertIn("--visual-load-mode", nav_command)
        self.assertIn("reference", nav_command)
        self.assertIn("--follow-camera-mode", nav_command)
        self.assertIn("fixed", nav_command)
        self.assertIn("--viewport-camera-prim", nav_command)
        self.assertIn("/World/camera_main", nav_command)
        self.assertIn("--fixed-camera-preset", nav_command)
        self.assertIn("start", nav_command)
        self.assertIn("--fixed-camera-close-distance", nav_command)
        self.assertIn("2.2", nav_command)
        self.assertIn("--fixed-camera-eye", nav_command)
        self.assertIn("-4.0", nav_command)
        self.assertIn("--fixed-camera-lookat", nav_command)
        self.assertIn("0.35", nav_command)
        self.assertNotIn("--headless", pick_command)
        self.assertIn("--keep-window-open", pick_command)


if __name__ == "__main__":
    unittest.main()
