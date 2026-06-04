"""Tests for navigation coordinator checkpoint preflight validation."""

from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from scripts.pipeline.run_nav_then_pick import _default_checkpoint_path, _nav_command, _validate_checkpoint


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
            yaw_align_kp=2.0,
            yaw_align_min_wz=0.55,
            yaw_align_max_wz=1.00,
            yaw_align_vx=0.08,
            yaw_align_activation_yaw_error=0.0,
            yaw_align_allow_reverse=False,
            yaw_align_stall_window_steps=240,
            yaw_align_min_progress=0.08,
            yaw_settle_stable_steps=20,
            debug_print_every=30,
            scene_usd=None,
            add_nav_ground=False,
            nav_map=None,
            dataset_dir=None,
            no_record=True,
            head_camera=False,
            load_visual_scene=True,
            visual_prim_path="/World/gauss",
            follow_camera=True,
            follow_camera_distance=2.0,
            follow_camera_height=0.7,
            follow_camera_side=0.0,
            flat_terrain=False,
            disable_sky_light=False,
            debug_command=None,
        )
        command = _nav_command(args, task=None)
        self.assertIn("--load-visual-scene", command)
        self.assertIn("/World/gauss", command)
        self.assertIn("--max-lin-vel", command)
        self.assertIn("0.5", command)
        self.assertIn("--yaw-align-max-wz", command)
        self.assertIn("1.0", command)
        self.assertIn("--yaw-align-min-wz", command)
        self.assertIn("0.55", command)
        self.assertIn("--yaw-align-vx", command)
        self.assertIn("0.08", command)
        self.assertIn("--debug-print-every", command)
        self.assertIn("30", command)


if __name__ == "__main__":
    unittest.main()
