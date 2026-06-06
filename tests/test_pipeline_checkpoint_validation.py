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
            max_nav_steps=3000,
            settle_steps=120,
            stall_window_steps=240,
            stall_min_progress=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.25,
            goal_tolerance=0.15,
            goal_yaw_tolerance=0.15,
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            lookahead_distance=0.35,
            prediction_horizon=0.90,
            max_lin_vel=0.50,
            max_ang_vel=1.00,
            brisk_nav=True,
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
            replay_include_initial_settle=True,
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
            side_grasp_fallback_retreat=False,
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
            max_nav_steps=5000,
            settle_steps=120,
            stall_window_steps=240,
            stall_min_progress=0.05,
            stall_min_forward_command=0.05,
            stall_min_forward_ratio=0.25,
            goal_tolerance=0.15,
            goal_yaw_tolerance=0.15,
            inflate_radius=0.25,
            local_clearance_radius=0.20,
            lookahead_distance=0.35,
            prediction_horizon=0.90,
            max_lin_vel=0.50,
            max_ang_vel=1.00,
            brisk_nav=False,
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
            replay_include_initial_settle=False,
            flat_terrain=False,
            disable_sky_light=False,
            debug_command=None,
            use_planner_server=False,
            handoff_smoke_only=False,
            allow_retreat_success=False,
            legacy_side_retreat=False,
            side_grasp_fallback_retreat=False,
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


if __name__ == "__main__":
    unittest.main()
