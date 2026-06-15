"""Tests for randomized batch pipeline command assembly."""

from __future__ import annotations

import json
import os
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

        self.assertEqual(tuple(args.table_x_range), (0.90, 0.95))
        self.assertEqual(tuple(args.table_y_range), (0.75, 1.5))
        self.assertAlmostEqual(args.table_z, 0.81653)
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
        self.assertEqual(tuple(args.base_goal_offset_xy), (0.28, -0.12))
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

        self.assertEqual(args.base_task, "tasks/nav_pick_apple_fast.json")
        self.assertEqual(args.place_template_task, "tasks/nav_pick_place_apple_contact.json")
        self.assertEqual(args.output_task_dir, "outputs/random_tasks/apple_pick_place_07_far_manual")
        self.assertEqual(args.dataset_root, "outputs/random_pick_place_dataset/apple_pick_place_07_far_manual")
        self.assertEqual(tuple(args.place_x_range), (0.65285, 0.75285))
        self.assertEqual(tuple(args.place_y_range), (5.00337, 5.50337))
        self.assertEqual(args.base_goal_mode, "object-offset")
        self.assertEqual(tuple(args.base_goal_offset_xy), (0.35, -0.08))
        self.assertAlmostEqual(args.clearance_radius, 0.20)
        self.assertAlmostEqual(args.handoff_clearance_radius, 0.20)
        self.assertTrue(args.brisk_nav)
        self.assertTrue(args.fast_dwa)
        self.assertEqual(args.max_nav_steps, 3000)
        self.assertTrue(args.ignore_goal_yaw)
        self.assertTrue(args.edge_biased)
        self.assertAlmostEqual(args.edge_min_clearance, 0.03)
        self.assertAlmostEqual(args.object_support_clearance, 0.0)
        self.assertFalse(args.randomize_object_yaw)
        self.assertAlmostEqual(args.terminal_yaw_polish_min_wz, 0.45)
        self.assertAlmostEqual(args.terminal_yaw_polish_max_wz, 0.55)
        self.assertEqual(args.manipulation_backend, "single-stage-07")
        self.assertEqual(args.single_stage_put_mode, "arm-place")
        self.assertEqual(args.single_stage_carry_mode, "logical")
        self.assertTrue(args.single_stage_replay_nav)
        self.assertTrue(args.single_stage_replay_nav_real_time)
        self.assertAlmostEqual(args.single_stage_replay_nav_speed, 1.0)
        self.assertFalse(args.restore_nav_place_for_arm_place)
        self.assertTrue(args.demo_visuals)
        self.assertEqual(args.follow_camera_mode, "stage")
        self.assertEqual(args.viewport_camera_prim, "/World/Camera1")
        self.assertFalse(args.keep_window_open)
        self.assertFalse(args.show_randomization_debug)
        self.assertTrue(args.side_retreat_only)
        self.assertTrue(args.use_planner_server)
        self.assertTrue(args.restart_planner_server)
        self.assertTrue(args.single_stage_stable_defaults)

    def test_pick_place_batch_single_stage_stable_env_defaults(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py", "--no-demo-visuals"]):
            args = pick_place_batch._parse_args()

        with patch.dict(os.environ, {}, clear=True):
            env = pick_place_batch._single_stage_07_env(args)

        self.assertIsNotNone(env)
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_BACKEND"], "visual_root_only")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_JOINTS"], "0")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_FULL_ROOT_POSE"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_HOLD_SUPPORT_JOINTS"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_DIRECT_JOINT_STATE"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_SETTLE_TO_START_SKIP_ERROR_TOL"], "0.005")
        self.assertEqual(env["GO2_X5_ARM_PLACE_EXEC_TIME_SCALE"], "1.50")
        self.assertEqual(env["GO2_X5_RETURN_HOME_AFTER_LIFT"], "0")
        self.assertEqual(env["GO2_X5_HOLD_GRIPPER_AFTER_CLOSE"], "1")
        self.assertEqual(env["GO2_X5_CARRY_HOLD_GRIPPER_CLOSE_TARGET"], "0")
        self.assertEqual(env["GO2_X5_REQUIRE_SIDE_RETREAT_HEIGHT_OK"], "0")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_POSE_SOURCE"], "usd_root")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_ORIENTATION_MODE"], "world_locked")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_TRANSLATION_ONLY"], "1")
        self.assertEqual(env["GO2_X5_CARRY_RESET_OBJECT_XFORM_STACK"], "0")
        self.assertEqual(env["GO2_X5_CARRY_COMPENSATE_PRESERVED_XFORM_STACK"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REAPPLY_OBJECT_POSE_BEFORE_REPLAY"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_CALLBACK_SAFE_LEG_DIRECT_SET"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_DIRECT_STATE_ONLY"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_LOCK_BASE"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_SKIP_WORLD_STEP"], "1")
        self.assertNotIn("GO2_X5_ARM_PLACE_OBJECT_FOLLOW_MODE", env)
        self.assertNotIn("GO2_X5_ARM_PLACE_REQUIRE_OBJECT_NEAR_RELEASE_BEFORE_OPEN", env)

        with patch("sys.argv", ["run_random_nav_pick_place_batch.py", "--no-single-stage-stable-defaults"]):
            args = pick_place_batch._parse_args()
        self.assertIsNone(pick_place_batch._single_stage_07_env(args))

    def test_pick_place_batch_video_baseline_env_defaults_keep_nav_gait(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py"]):
            args = pick_place_batch._parse_args()

        with patch.dict(os.environ, {}, clear=True):
            env = pick_place_batch._single_stage_07_env(args)

        self.assertIsNotNone(env)
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_MODE"], "1")
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_SOURCE"], "--demo-visuals")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_BACKEND"], "live_planar_articulation")
        self.assertEqual(env["GO2_X5_CARRY_VISUAL_ROOT_XFORM_SYNC"], "0")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_FULL_ROOT_POSE"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_ROOT_VELOCITY"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_JOINTS"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_LEG_ONLY"], "0")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_JOINT_ACTION"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_JOINT_VELOCITY"], "0")
        self.assertEqual(env["GO2_X5_CARRY_ZERO_ROOT_VELOCITY_WHEN_SKIPPED"], "0")
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_RENDER_WORLD_STEP"], "1")
        self.assertEqual(env["GO2_X5_CARRY_TCP_SOURCE"], "root_delta")
        self.assertEqual(env["GO2_X5_RETURN_HOME_AFTER_LIFT"], "0")
        self.assertEqual(env["GO2_X5_HOLD_GRIPPER_AFTER_CLOSE"], "1")
        self.assertEqual(env["GO2_X5_CARRY_HOLD_GRIPPER_CLOSE_TARGET"], "0")
        self.assertEqual(env["GO2_X5_REQUIRE_SIDE_RETREAT_HEIGHT_OK"], "0")
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_AFTER_PICK_HOLD_S"], "0.0")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_POSE_SOURCE"], "usd_root")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_ORIENTATION_MODE"], "world_locked")
        self.assertEqual(env["GO2_X5_CARRY_OBJECT_TRANSLATION_ONLY"], "1")
        self.assertEqual(env["GO2_X5_CARRY_RESET_OBJECT_XFORM_STACK"], "0")
        self.assertEqual(env["GO2_X5_CARRY_COMPENSATE_PRESERVED_XFORM_STACK"], "1")
        self.assertEqual(env["GO2_X5_CARRY_REAPPLY_OBJECT_POSE_BEFORE_REPLAY"], "0")
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_LEG_POLICY"], "frozen_safe_pose")
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_SAFE_LEG_SETTLE_SKIP_ERROR_TOL"], "0.01")
        self.assertEqual(env["GO2_X5_VIDEO_HOLD_ARM_GRIPPER_MODE"], "action")
        self.assertEqual(env["GO2_X5_ARM_PLACE_PRE_SETTLE_S"], "2.50")
        self.assertEqual(env["GO2_X5_ARM_PLACE_HOLD_SUPPORT_JOINTS"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_BEFORE_WORLD_STEP"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_PLANNING_HOLD_ARM_GRIPPER"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_PLANNING_HOLD_MODE"], "action")
        self.assertEqual(env["GO2_X5_ARM_PLACE_CALLBACK_SAFE_LEG_DIRECT_SET"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_DIRECT_JOINT_STATE"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_DIRECT_STATE_ONLY"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_LOCK_BASE"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_SKIP_WORLD_STEP"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_OBJECT_FOLLOW_MODE"], "tcp_kinematic_clamp")
        self.assertEqual(env["GO2_X5_ARM_PLACE_REQUIRE_OBJECT_NEAR_RELEASE_BEFORE_OPEN"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_SETTLE_TO_START_SKIP_ERROR_TOL"], "0.005")
        self.assertEqual(env["GO2_X5_ARM_PLACE_EXEC_TIME_SCALE"], "1.00")
        self.assertEqual(env["GO2_X5_ARM_PLACE_MOVE_TO_PRE_PLACE_TIME_SCALE"], "1.00")
        self.assertEqual(env["GO2_X5_ARM_PLACE_APPROACH_TO_PLACE_TIME_SCALE"], "1.00")
        self.assertEqual(env["GO2_X5_ARM_PLACE_RETREAT_PLACE_TIME_SCALE"], "1.00")
        self.assertEqual(env["GO2_X5_ARM_PLACE_DISABLE_KINEMATIC_OBJECT_COLLISION"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_RETURN_HOME_AFTER_RELEASE"], "1")
        self.assertEqual(env["GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_DURING_RETURN_HOME"], "0")
        self.assertEqual(env["GO2_X5_ARM_PRE_PLACE_CLEARANCE_M"], "0.06")
        self.assertEqual(env["GO2_X5_ARM_PLACE_RELEASE_CLEARANCE_M"], "0.005")
        self.assertEqual(env["GO2_X5_ARM_PLACE_RETREAT_CLEARANCE_M"], "0.08")

        with patch.dict(os.environ, {"GO2_X5_VIDEO_BASELINE_MODE": "0"}, clear=True):
            env = pick_place_batch._single_stage_07_env(args)
        self.assertIsNotNone(env)
        self.assertEqual(env["GO2_X5_CARRY_REPLAY_JOINTS"], "0")
        self.assertEqual(env["GO2_X5_ARM_PLACE_SKIP_WORLD_STEP"], "1")

        with patch.dict(
            os.environ,
            {
                "GO2_X5_VIDEO_BASELINE_MODE": "1",
                "GO2_X5_CARRY_REPLAY_BACKEND": "custom_backend",
            },
            clear=True,
        ):
            env = pick_place_batch._single_stage_07_env(args)
        self.assertIsNotNone(env)
        self.assertNotIn("GO2_X5_VIDEO_BASELINE_MODE", env)
        self.assertEqual(env["GO2_X5_VIDEO_BASELINE_SOURCE"], "GO2_X5_VIDEO_BASELINE_MODE")
        self.assertNotIn("GO2_X5_CARRY_REPLAY_BACKEND", env)

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
                "--show-randomization-debug",
                "--single-stage-replay-nav",
                "--single-stage-replay-nav-real-time",
                "--single-stage-replay-nav-speed",
                "1.5",
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
        self.assertIn("--show-randomization-debug", command)
        self.assertIn("--replay-nav-to-pick", command)
        self.assertIn("--replay-nav-to-place", command)
        self.assertIn("--replay-nav-real-time", command)
        self.assertEqual(command[command.index("--replay-nav-speed") + 1], "1.5")

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

    def test_pick_place_batch_randomizes_place_xy_and_keeps_base_offset(self) -> None:
        task = {
            "place": {
                "enabled": True,
                "place_pose_world": {
                    "x": 0.65,
                    "y": 5.00,
                    "z": 0.72664,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
                "base_goal": {"x": 1.10, "y": 4.85, "yaw": 1.57},
            },
            "randomization": {},
        }

        randomized = pick_place_batch._apply_default_place_xy_randomization(
            task,
            episode_seed=7,
            place_x_range=(0.55, 0.75),
            place_y_range=(4.90, 5.10),
        )
        report = randomized["randomization"]["place_xy_randomization"]
        before_pose = report["place_pose_world_before"]
        after_pose = report["place_pose_world_after"]
        before_goal = report["base_goal_before"]
        after_goal = report["base_goal_after"]
        dx, dy = report["delta_xy_m"]

        self.assertTrue(report["enabled"])
        self.assertEqual(report["mode"], "sample_xy_within_cli_range_translate_base_goal")
        self.assertEqual(report["x_range_m"], [0.55, 0.75])
        self.assertEqual(report["y_range_m"], [4.90, 5.10])
        self.assertGreaterEqual(report["sampled_xy"]["x"], 0.55)
        self.assertLessEqual(report["sampled_xy"]["x"], 0.75)
        self.assertGreaterEqual(report["sampled_xy"]["y"], 4.90)
        self.assertLessEqual(report["sampled_xy"]["y"], 5.10)
        self.assertAlmostEqual(after_pose["x"] - before_pose["x"], dx)
        self.assertAlmostEqual(after_pose["y"] - before_pose["y"], dy)
        self.assertAlmostEqual(after_goal["x"] - before_goal["x"], dx)
        self.assertAlmostEqual(after_goal["y"] - before_goal["y"], dy)
        self.assertAlmostEqual(
            after_goal["x"] - after_pose["x"],
            before_goal["x"] - before_pose["x"],
        )
        self.assertAlmostEqual(
            after_goal["y"] - after_pose["y"],
            before_goal["y"] - before_pose["y"],
        )
        self.assertEqual(after_pose["z"], before_pose["z"])
        self.assertEqual(after_pose["roll"], before_pose["roll"])
        self.assertEqual(after_pose["pitch"], before_pose["pitch"])
        self.assertEqual(after_pose["yaw"], before_pose["yaw"])
        self.assertFalse(report["object_mesh_randomization"]["enabled"])

    def test_pick_place_batch_command_forwards_stable_terminal_yaw_options(self) -> None:
        with patch("sys.argv", ["run_random_nav_pick_place_batch.py", "--demo-visuals"]):
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
        self.assertIn("--nav-headless", command)
        self.assertNotIn("--demo-visuals", command)
        self.assertNotIn("--follow-camera-mode", command)
        self.assertNotIn("--viewport-camera-prim", command)
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
