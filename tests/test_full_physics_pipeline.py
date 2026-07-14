"""Tests for the first-phase full-physics pipeline skeleton."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.pipeline.run_full_physics_pipeline import (
    _build_parser,
    _locomotion_runtime_kwargs,
    _navigation_smoke_viewport_runtime_kwargs,
    _parse_args,
    main,
)
from source.diagnostics import (
    DryRunEpisodeVerifier,
    FullPhysicsVerifier,
    ManipulationApplySmokeVerifier,
    NavigationEpisodeVerifier,
)
from source.interfaces import ArmPlan, EpisodeSpec, RobotAction, SimulationState, VerificationResult
from source.manipulation import (
    BinaryGripperController,
    SegmentedArmExecutor,
    SegmentedArmExecutorConfig,
    SegmentedSmokeManipulationPlanner,
)
from source.navigation.dry_run import DryRunNavExecutor, DryRunNavPlanner
from source.pipeline import (
    FullPhysicsConfig,
    FullPhysicsPipeline,
    LocomotionPolicySettings,
    ManipulationSettings,
    NavigationSettings,
    PCT_MULTIFLOOR_LOCOMOTION_TASK,
    PipelineState,
    StateLimits,
)
from source.pipeline.dry_run import create_dry_run_pipeline
from source.pipeline.isaac_compat import patch_numpy_for_isaacsim
from source.pipeline.factory import create_full_physics_pipeline
from source.pipeline.full_physics_pipeline import _should_auto_switch_overview_camera
from source.pipeline.manipulation_apply_smoke import create_manipulation_apply_smoke_pipeline
from source.pipeline.manipulation_smoke import create_manipulation_smoke_pipeline
from source.pipeline.navigation_smoke import _build_dwa_config
from source.pipeline.state_machine import FullPhysicsStateMachine
from source.recording import JsonlEpisodeRecorder
from source.simulation import InMemorySimulationRuntime
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FullPhysicsPipelineTest(unittest.TestCase):
    def test_numpy_isaacsim_compat_restores_broadcast_to_alias(self) -> None:
        import numpy as np
        import numpy.lib.stride_tricks as stride_tricks

        original = getattr(stride_tricks, "broadcast_to", None)
        if hasattr(stride_tricks, "broadcast_to"):
            delattr(stride_tricks, "broadcast_to")
        try:
            report = patch_numpy_for_isaacsim()

            self.assertTrue(report["has_broadcast_to"])
            self.assertIs(stride_tricks.broadcast_to, np.broadcast_to)
        finally:
            if original is not None:
                stride_tricks.broadcast_to = original
            elif hasattr(stride_tricks, "broadcast_to"):
                delattr(stride_tricks, "broadcast_to")

    def _run_task(
        self,
        task_name: str,
        output_dir: Path,
        *,
        limits: StateLimits | None = None,
        seed: int = 7,
    ) -> tuple[dict, object]:
        task_path = PROJECT_ROOT / "tasks" / task_name
        config = FullPhysicsConfig(
            task_json=task_path,
            output_dir=output_dir,
            seed=seed,
            dry_run=True,
            limits=limits or StateLimits(),
        )
        spec = JsonTaskProvider().load(task_path)
        pipeline = create_dry_run_pipeline(
            config=config,
            episode_spec=spec,
            episode_seed=seed,
            episode_dir=output_dir,
        )
        summary = pipeline.run_episode()
        return summary, pipeline

    def test_dry_run_completes_full_state_flow_and_writes_artifacts(self) -> None:
        expected_trace = [state.value for state in PipelineState if state != PipelineState.FAILED]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "episode"
            summary, pipeline = self._run_task(
                "nav_pick_place_apple_contact.json",
                output_dir,
            )

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertFalse(summary["execution_provenance_verified"])
            self.assertEqual(summary["success_semantics"], "control_flow_only")
            self.assertEqual(summary["state_trace"], expected_trace)
            self.assertGreater(summary["duration_steps"], len(expected_trace))
            self.assertEqual(pipeline.simulation.apply_calls, summary["duration_steps"])

            for filename in (
                "task.json",
                "events.jsonl",
                "frames.jsonl",
                "lerobot_manifest.json",
                "summary.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)

            manifest = json.loads((output_dir / "lerobot_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["lerobot_exported"])
            event_names = [
                json.loads(line)["name"]
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("gripper_close", event_names)
            self.assertIn("gripper_open", event_names)
            self.assertIn("episode_success", event_names)
            self.assertEqual(event_names.count("gripper_close"), 1)
            self.assertEqual(event_names.count("gripper_open"), 1)

            frames = [
                json.loads(line)
                for line in (output_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(pipeline.simulation.step_calls, summary["duration_steps"])
            nav_place_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_NAV_TO_PLACE.value
            ]
            self.assertGreater(len(nav_place_frames), 0)
            self.assertTrue(
                all(frame["action"]["gripper_command"] == "close" for frame in nav_place_frames)
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("carry_gripper_hold")
                    for frame in nav_place_frames
                )
            )
            place_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
            ]
            self.assertGreater(len(place_frames), 0)
            place_open_frames = [
                frame for frame in place_frames if frame["action"]["gripper_command"] == "open"
            ]
            self.assertEqual(len(place_open_frames), 1)
            saw_place_open = False
            for frame in place_frames:
                action = frame["action"]
                if action["gripper_command"] == "open":
                    saw_place_open = True
                    self.assertFalse(action["metadata"].get("carry_gripper_hold", False))
                    continue
                if saw_place_open:
                    self.assertFalse(action["metadata"].get("carry_gripper_hold", False))
                    continue
                self.assertEqual(action["gripper_command"], "close")
                self.assertTrue(action["metadata"].get("carry_gripper_hold"))

    def test_runtime_exception_writes_operation_and_fresh_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "episode"
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=output_dir,
                dry_run=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=output_dir,
            )

            def fail_step(*, render: bool) -> None:
                del render
                raise RuntimeError("synthetic step failure")

            pipeline.simulation.step = fail_step
            with self.assertRaisesRegex(RuntimeError, "synthetic step failure"):
                pipeline.run_episode()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "pipeline_runtime_exception")
            self.assertEqual(summary["runtime_exception"]["operation"], "simulation_step")
            self.assertEqual(summary["runtime_exception"]["pipeline_state"], "reset_episode")
            self.assertEqual(summary["runtime_exception"]["exception_type"], "RuntimeError")
            self.assertIn("synthetic step failure", summary["runtime_exception"]["traceback"])

            events = [
                json.loads(line)
                for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[-1]["name"], "pipeline_runtime_exception")
            self.assertEqual(events[-1]["metadata"]["operation"], "simulation_step")

    def test_carry_gripper_forces_close_target_like_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                dry_run=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=1,
                episode_dir=Path(tmp_dir) / "episode",
            )
            open_observation = SimulationState(
                step_index=10,
                timestamp=0.5,
                robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
                robot_root_velocity=(0.0,) * 6,
                joint_positions=(0.0,) * 6 + (0.043, 0.043),
                joint_velocities=(0.0,) * 8,
                metadata={
                    "joint_names": tuple(
                        f"arm_joint{index}" for index in range(1, 9)
                    )
                },
            )
            close_action = RobotAction(
                gripper_command="close",
                source="arm_pick",
                metadata={
                    "event_marker": "gripper_close",
                    "gripper_joint_names": ("arm_joint7", "arm_joint8"),
                    "gripper_joint_positions": (0.0, 0.0),
                },
            )
            pipeline.machine._remember_carry_gripper_target(close_action, open_observation)

            target = pipeline.machine._carry_gripper_target
            self.assertIsNotNone(target)
            self.assertEqual(target["gripper_joint_positions"], (0.0, 0.0))
            self.assertEqual(
                target["hold_position_source"],
                "forced_close_target_for_carry",
            )
            self.assertEqual(target["commanded_close_positions"], (0.0, 0.0))

            contact_observation = replace(
                open_observation,
                step_index=30,
                joint_positions=(0.0,) * 6 + (0.035, 0.020),
            )
            pipeline.machine._remember_carry_gripper_target(
                close_action,
                contact_observation,
            )
            self.assertEqual(
                pipeline.machine._carry_gripper_target["gripper_joint_positions"],
                (0.0, 0.0),
            )

            pipeline.machine._capture_verified_carry_gripper_preload(
                contact_observation
            )
            self.assertEqual(
                pipeline.machine._carry_gripper_target["hold_position_source"],
                "verified_contact_preload",
            )
            hold_positions = pipeline.machine._carry_gripper_target[
                "gripper_joint_positions"
            ]
            self.assertAlmostEqual(hold_positions[0], 0.023)
            self.assertAlmostEqual(hold_positions[1], 0.008)
            self.assertEqual(
                pipeline.machine._carry_gripper_target["commanded_close_positions"],
                (0.0, 0.0),
            )

    def test_carry_object_tracking_is_read_only_and_reports_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                dry_run=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=1,
                episode_dir=Path(tmp_dir) / "episode",
            )
            pick_state = SimulationState(
                step_index=10,
                timestamp=0.5,
                robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
                robot_root_velocity=(0.0,) * 6,
                tcp_pose=(1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0),
                object_pose=(1.02, 2.01, 0.78, 1.0, 0.0, 0.0, 0.0),
            )
            dropped_state = replace(
                pick_state,
                step_index=20,
                object_pose=(1.02, 2.01, 0.30, 1.0, 0.0, 0.0, 0.0),
            )
            pipeline.machine._capture_carry_object_tcp_offset(pick_state)

            report = pipeline.machine._verify_carry_object_tracking(dropped_state)

            self.assertFalse(report["success"])
            self.assertEqual(report["failure_reason"], "object_dropped_after_pick")
            self.assertTrue(report["read_only_check"])
            self.assertFalse(report["object_pose_modified"])
            self.assertGreater(report["object_tcp_offset_drift_m"], 0.10)

    def test_pick_only_task_fails_with_structured_place_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary, _pipeline = self._run_task(
                "nav_pick_apple_fast.json",
                Path(tmp_dir) / "episode",
            )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "place_target_unreachable")
            self.assertEqual(summary["final_state"], PipelineState.FAILED.value)
            self.assertIn(PipelineState.PLAN_NAV_TO_PLACE.value, summary["state_trace"])
            self.assertEqual(summary["state_trace"][-1], PipelineState.FAILED.value)

    def test_navigation_timeout_is_reported_without_exception_exit(self) -> None:
        limits = StateLimits(navigation=1, episode=100)
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary, _pipeline = self._run_task(
                "nav_pick_place_apple_contact.json",
                Path(tmp_dir) / "episode",
                limits=limits,
            )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "nav_to_pick_timeout")
            self.assertEqual(
                summary["failure_metadata"]["current_state"],
                PipelineState.EXEC_NAV_TO_PICK.value,
            )

    def test_cli_help_is_chinese_and_full_physics_is_default(self) -> None:
        help_text = _build_parser().format_help()
        self.assertIn("任务 JSON 路径", help_text)
        self.assertIn("--dry-run", help_text)
        self.assertIn("--simulation-smoke", help_text)
        self.assertIn("--navigation-smoke", help_text)
        self.assertIn("--navigation-carry-smoke", help_text)
        self.assertIn("--stair-locomotion-smoke", help_text)
        self.assertIn("--pct-plan-preview", help_text)
        self.assertIn("--pct-cross-floor-gateway", help_text)
        self.assertIn("--pct-stair-float", help_text)
        self.assertIn("--pct-stair-float-exit-distance", help_text)
        self.assertIn("--pct-stair-float-settle-time", help_text)
        self.assertIn("--show-planned-trajectories", help_text)
        self.assertIn("--pick-smoke", help_text)
        self.assertIn("--manipulation-smoke", help_text)
        self.assertIn("--manipulation-apply-smoke", help_text)
        self.assertNotIn("--full-physics", help_text)
        self.assertNotIn("--integrated-apply-smoke", help_text)
        self.assertIn("--pick-plan-json", help_text)
        self.assertIn("--place-plan-json", help_text)
        self.assertNotIn("--viewport-camera-prim", help_text)
        self.assertNotIn("--save-video", help_text)
        self.assertNotIn("--enable-debug-vis", help_text)
        self.assertNotIn("--no-lock-base-during-manipulation", help_text)
        self.assertNotIn("--no-lock-support-joints-during-manipulation", help_text)
        self.assertNotIn("--no-replan-pick-from-current-state", help_text)
        self.assertNotIn("--no-auto-start-curobo-server", help_text)
        self.assertIn("--no-headless", help_text)

    def test_manipulation_stability_defaults_are_fixed_on(self) -> None:
        default_args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--dry-run",
            ]
        )
        self.assertEqual(default_args.mode, "dry_run")
        self.assertTrue(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.lock_base_during_manipulation
        )
        self.assertTrue(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.lock_support_joints_during_manipulation
        )
        self.assertEqual(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.base_lock_settle_steps,
            60,
        )
        self.assertEqual(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.place_base_lock_settle_steps,
            0,
        )
        self.assertFalse(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.settle_object_before_navigation
        )
        self.assertFalse(
            FullPhysicsConfig(
                task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                output_dir=PROJECT_ROOT / "outputs/test",
            ).manipulation.settle_base_before_navigation
        )

    def test_navigation_defaults_use_stable_brisk_fast_profile(self) -> None:
        config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs/test",
        )
        dwa_config = _build_dwa_config(config.navigation)

        self.assertTrue(config.navigation.brisk_nav)
        self.assertTrue(config.navigation.fast_dwa)
        self.assertEqual(config.navigation.dwa_replan_interval_steps, 2)
        self.assertAlmostEqual(dwa_config.max_linear_velocity, 0.80)
        self.assertAlmostEqual(dwa_config.min_active_linear_velocity, 0.55)
        self.assertAlmostEqual(dwa_config.near_goal_min_active_linear_velocity, 0.38)
        self.assertAlmostEqual(dwa_config.close_goal_speed_limit, 0.35)
        self.assertAlmostEqual(dwa_config.speed_bias, 1.10)
        self.assertAlmostEqual(dwa_config.max_linear_accel, 4.5)
        self.assertAlmostEqual(dwa_config.prediction_horizon, 0.45)
        self.assertAlmostEqual(dwa_config.lookahead_distance, 0.30)
        self.assertEqual(dwa_config.linear_samples, 3)
        self.assertEqual(dwa_config.angular_samples, 7)
        self.assertEqual(dwa_config.path_distance_window, 80)
        self.assertFalse(dwa_config.use_command_velocity_window)

        pct_dwa_config = _build_dwa_config(
            config.navigation,
            policy_profile="pct_multifloor",
        )
        self.assertTrue(pct_dwa_config.use_command_velocity_window)

    def test_cli_defaults_to_full_physics_mode(self) -> None:
        args = _parse_args(
            ["--task-json", "tasks/nav_pick_place_apple_contact.json"]
        )
        self.assertEqual(args.mode, "full_physics")
        self.assertFalse(args.show_planned_trajectories)

    def test_pct_multifloor_stable_preset_resolves_runtime_defaults(self) -> None:
        args = _parse_args(["--pct-multifloor"])

        self.assertEqual(args.runtime_preset, "pct_multifloor_stable")
        self.assertEqual(
            args.task_json,
            "tasks/nav_pick_place_apple_multifloor_pct.json",
        )
        self.assertEqual(args.global_planner, "pct")
        self.assertEqual(args.pct_server_script, "scripts/navigation/pct_grid_server.py")
        self.assertEqual(
            args.pct_tomogram_path,
            "source/scene/multifloor/mutifloor.pickle",
        )
        self.assertEqual(
            args.pct_walkable_path,
            "source/scene/multifloor/mutifloor_ply_walkable.npy",
        )
        self.assertEqual(
            args.pct_collision_ply_path,
            "source/scene/multifloor/ply/3dgs_collision.ply",
        )
        self.assertTrue(args.pct_no_fallback)
        self.assertEqual(args.policy_profile, "pct_multifloor")
        self.assertEqual(args.locomotion_task, PCT_MULTIFLOOR_LOCOMOTION_TASK)
        self.assertEqual(
            args.locomotion_checkpoint,
            "checkpoints/go2_x5/pct_multifloor/model_26000.pt",
        )
        self.assertFalse(args.randomize_task)
        self.assertFalse(args.randomize_base_goal)
        self.assertFalse(args.show_planned_trajectories)
        self.assertFalse(args.headless)
        self.assertFalse(args.keep_window_open)
        self.assertEqual(args.output_dir, "outputs/full_physics_pipeline")
        self.assertEqual(args.navigation_visual_mode, "collision")
        self.assertTrue(args.record_video)
        self.assertEqual(args.video_mode, "all")
        self.assertEqual(
            args.overview_camera_schedule,
            "configs/recording/multifloor_overview_camera_schedule.json",
        )
        self.assertTrue(args.pct_stair_float)

    def test_pct_multifloor_stable_preset_preserves_explicit_overrides(self) -> None:
        args = _parse_args(
            [
                "--pct-multifloor",
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--pct-fallback-to-astar",
                "--randomize-task",
                "--randomize-base-goal",
                "--show-planned-trajectories",
                "--no-record-video",
                "--video-mode",
                "overview",
                "--navigation-visual-mode",
                "full",
            ]
        )

        self.assertEqual(args.task_json, "tasks/nav_pick_place_apple_contact.json")
        self.assertFalse(args.pct_no_fallback)
        self.assertTrue(args.randomize_task)
        self.assertTrue(args.randomize_base_goal)
        self.assertTrue(args.show_planned_trajectories)
        self.assertFalse(args.record_video)
        self.assertEqual(args.video_mode, "overview")
        self.assertEqual(args.navigation_visual_mode, "full")

    def test_pct_multifloor_full_pipeline_can_explicitly_disable_float(self) -> None:
        args = _parse_args(["--pct-multifloor", "--no-pct-stair-float"])

        self.assertEqual(args.mode, "full_physics")
        self.assertFalse(args.pct_stair_float)
        self.assertFalse(args.show_planned_trajectories)

    def test_stair_locomotion_smoke_selects_pct_and_disables_float(self) -> None:
        args = _parse_args(["--stair-locomotion-smoke"])

        self.assertEqual(args.mode, "stair_locomotion_smoke")
        self.assertEqual(args.global_planner, "pct")
        self.assertEqual(args.policy_profile, "pct_multifloor")
        self.assertFalse(args.pct_stair_float)
        self.assertTrue(args.show_planned_trajectories)
        self.assertFalse(args.headless)
        self.assertTrue(args.keep_window_open)
        self.assertTrue(args.record_video)
        self.assertTrue(args.record_dataset)
        self.assertEqual(args.video_mode, "overview")
        self.assertEqual(args.output_dir, "outputs/stair_locomotion_smoke")
        self.assertEqual(
            args.overview_camera_schedule,
            "configs/recording/stair_locomotion_camera_schedule.json",
        )
        self.assertEqual(
            args.task_json,
            "tasks/nav_pick_place_apple_multifloor_pct.json",
        )

    def test_stair_locomotion_smoke_preserves_explicit_runtime_overrides(self) -> None:
        args = _parse_args(
            [
                "--stair-locomotion-smoke",
                "--headless",
                "--no-record-video",
                "--no-record-dataset",
                "--output-dir",
                "/tmp/stair_override",
            ]
        )

        self.assertTrue(args.headless)
        self.assertFalse(args.keep_window_open)
        self.assertFalse(args.record_video)
        self.assertFalse(args.record_dataset)
        self.assertEqual(args.output_dir, "/tmp/stair_override")

    def test_stair_locomotion_smoke_manages_gui_camera3(self) -> None:
        stair = _navigation_smoke_viewport_runtime_kwargs(
            headless=False,
            stair_locomotion_smoke=True,
        )
        regular_gui = _navigation_smoke_viewport_runtime_kwargs(
            headless=False,
            stair_locomotion_smoke=False,
        )

        self.assertEqual(stair["viewport_camera_prim_path"], "/World/Camera3")
        self.assertTrue(stair["auto_manage_viewport_camera"])
        self.assertFalse(regular_gui["auto_manage_viewport_camera"])

    def test_stair_locomotion_video_schedule_auto_switches_in_gui(self) -> None:
        stair_config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
            output_dir=PROJECT_ROOT / "outputs/test",
            headless=False,
            stair_locomotion_smoke=True,
        )
        regular_gui_config = replace(stair_config, stair_locomotion_smoke=False)

        self.assertTrue(_should_auto_switch_overview_camera(stair_config))
        self.assertFalse(_should_auto_switch_overview_camera(regular_gui_config))

    def test_stair_locomotion_smoke_rejects_explicit_float(self) -> None:
        with self.assertRaisesRegex(SystemExit, "固定禁用 Float"):
            main(["--stair-locomotion-smoke", "--pct-stair-float"])

    def test_astar_without_task_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "A\\* 模式需要显式提供 --task-json"):
            _parse_args([])

    def test_cli_can_enable_planned_trajectory_visualization(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--show-planned-trajectories",
            ]
        )

        self.assertTrue(args.show_planned_trajectories)

    def test_pct_plan_preview_mode_does_not_require_locomotion_checkpoint(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--pct-plan-preview",
                "--policy-profile",
                "pct_multifloor",
            ]
        )

        self.assertEqual(args.mode, "pct_plan_preview")
        FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs/test",
            pct_plan_preview=True,
            locomotion=LocomotionPolicySettings(policy_profile="pct_multifloor"),
        )

    def test_cli_pct_global_hard_obstacle_default_matches_config(self) -> None:
        args = _build_parser().parse_args(
            ["--task-json", "tasks/nav_pick_place_apple_contact.json"]
        )

        self.assertEqual(
            args.pct_global_vertical_obstacle_min_slices,
            NavigationSettings().pct_global_vertical_obstacle_min_slices,
        )
        self.assertEqual(
            args.pct_cross_floor_vertical_obstacle_min_slices,
            NavigationSettings().pct_cross_floor_vertical_obstacle_min_slices,
        )
        self.assertIsNone(args.pct_cross_floor_gateway)
        self.assertEqual(
            args.pct_cross_floor_gateway_radius,
            NavigationSettings().pct_cross_floor_gateway_radius_m,
        )
        self.assertEqual(
            args.pct_multifloor_route_corridor_radius,
            NavigationSettings().pct_multifloor_route_corridor_radius,
        )
        self.assertEqual(
            args.pct_carry_max_linear_velocity,
            NavigationSettings().pct_carry_max_linear_velocity,
        )
        self.assertEqual(
            args.pct_carry_max_angular_velocity,
            NavigationSettings().pct_carry_max_angular_velocity,
        )
        self.assertIsNone(args.pct_stair_float)
        self.assertEqual(
            args.pct_stair_float_speed,
            NavigationSettings().pct_stair_float_speed_mps,
        )
        self.assertEqual(
            args.pct_stair_float_activation_radius,
            NavigationSettings().pct_stair_float_activation_radius_m,
        )
        self.assertEqual(
            args.pct_stair_float_completion_radius,
            NavigationSettings().pct_stair_float_completion_radius_m,
        )
        self.assertEqual(
            args.pct_stair_float_approach_distance,
            NavigationSettings().pct_stair_float_approach_distance_m,
        )
        self.assertEqual(
            args.pct_stair_float_exit_distance,
            NavigationSettings().pct_stair_float_exit_distance_m,
        )
        self.assertEqual(
            args.pct_stair_float_settle_time,
            NavigationSettings().pct_stair_float_settle_time_s,
        )
        self.assertEqual(
            args.pct_stair_float_min_root_z_offset,
            NavigationSettings().pct_stair_float_min_root_z_offset_m,
        )
        self.assertEqual(
            args.pct_stair_float_release_root_z_offset,
            NavigationSettings().pct_stair_float_release_root_z_offset_m,
        )

    def test_cli_accepts_pct_stair_float_overrides(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_multifloor_pct.json",
                "--pct-stair-float",
                "--pct-stair-float-speed",
                "0.22",
                "--pct-stair-float-activation-radius",
                "0.55",
                "--pct-stair-float-completion-radius",
                "0.12",
                "--pct-stair-float-approach-distance",
                "2.1",
                "--pct-stair-float-exit-distance",
                "1.4",
                "--pct-stair-float-settle-time",
                "0.6",
                "--pct-stair-float-min-root-z-offset",
                "0.42",
                "--pct-stair-float-release-root-z-offset",
                "0.38",
            ]
        )

        self.assertTrue(args.pct_stair_float)
        self.assertEqual(args.pct_stair_float_speed, 0.22)
        self.assertEqual(args.pct_stair_float_activation_radius, 0.55)
        self.assertEqual(args.pct_stair_float_completion_radius, 0.12)
        self.assertEqual(args.pct_stair_float_approach_distance, 2.1)
        self.assertEqual(args.pct_stair_float_exit_distance, 1.4)
        self.assertEqual(args.pct_stair_float_settle_time, 0.6)
        self.assertEqual(args.pct_stair_float_min_root_z_offset, 0.42)
        self.assertEqual(args.pct_stair_float_release_root_z_offset, 0.38)

    def test_cli_dry_run_writes_startup_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "full_physics"
            output_dir.mkdir(parents=True)
            (output_dir / "startup_failure.json").write_text("stale", encoding="utf-8")

            result = main(
                [
                    "--task-json",
                    "tasks/nav_pick_place_apple_contact.json",
                    "--dry-run",
                    "--no-randomize-task",
                    "--no-randomize-base-goal",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(result, 0)
            status = json.loads(
                (output_dir / "startup_status.json").read_text(encoding="utf-8")
            )
            phases = [event["phase"] for event in status["phases"]]
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["mode"], "dry_run")
            self.assertIn("config_ready", phases)
            self.assertIn("pipeline_created", phases)
            self.assertIn("episode_finished", phases)
            self.assertIn("completed", phases)
            self.assertFalse((output_dir / "startup_failure.json").exists())

    def test_cli_pick_smoke_mode_is_available(self) -> None:
        args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--pick-smoke",
            ]
        )
        self.assertEqual(args.mode, "pick_smoke")

    def test_cli_rejects_missing_external_plan_paths_before_isaac_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_pick = Path(tmp_dir) / "missing_pick_plan.json"
            missing_place = Path(tmp_dir) / "missing_place_plan.json"

            with self.assertRaisesRegex(SystemExit, "外部 cuRobo plan JSON 不存在"):
                main(
                    [
                        "--task-json",
                        "tasks/nav_pick_place_apple_contact.json",
                        "--manipulation-apply-smoke",
                        "--pick-plan-json",
                        str(missing_pick),
                        "--place-plan-json",
                        str(missing_place),
                    ]
                )

    def test_removed_integrated_apply_smoke_flag_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            _build_parser().parse_args(
                [
                    "--task-json",
                    "tasks/nav_pick_place_apple_contact.json",
                    "--integrated-apply-smoke",
                ]
            )

    def test_full_physics_rejects_offline_plan_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick.json"
            place_path = root / "place.json"
            pick_path.write_text("{}", encoding="utf-8")
            place_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "禁止使用离线 plan JSON"):
                main(
                    [
                        "--task-json",
                        "tasks/nav_pick_place_apple_contact.json",
                        "--pick-plan-json",
                        str(pick_path),
                        "--place-plan-json",
                        str(place_path),
                    ]
                )

    def test_pick_smoke_rejects_offline_plan_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick.json"
            place_path = root / "place.json"
            pick_path.write_text("{}", encoding="utf-8")
            place_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "禁止使用离线 plan JSON"):
                main(
                    [
                        "--task-json",
                        "tasks/nav_pick_place_apple_contact.json",
                        "--pick-smoke",
                        "--pick-plan-json",
                        str(pick_path),
                        "--place-plan-json",
                        str(place_path),
                    ]
                )
            with self.assertRaisesRegex(ValueError, "plan JSON fallback is disabled"):
                FullPhysicsConfig(
                    task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
                    output_dir=root,
                    full_physics=True,
                    pick_plan_json=pick_path,
                    place_plan_json=place_path,
                )

    def test_full_physics_factory_uses_online_planner_without_offline_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
            )
            spec = JsonTaskProvider().load(task_path)

            pipeline = create_full_physics_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=root / "episode",
                simulation=InMemorySimulationRuntime(),  # type: ignore[arg-type]
            )

            self.assertIsInstance(pipeline.machine.arm_executor, SegmentedArmExecutor)
            self.assertFalse(
                pipeline.machine.manipulation_planner._config.side_grasp_plan_vertical_lift
            )
            self.assertFalse(
                pipeline.machine.manipulation_planner._config.side_grasp_fallback_retreat
            )
            self.assertFalse(
                pipeline.machine.manipulation_planner._config.side_grasp_retreat_to_pregrasp
            )
            self.assertTrue(
                pipeline.machine.manipulation_planner._config.split_pregrasp_motion
            )
            self.assertFalse(
                pipeline.machine.manipulation_planner._config.reuse_pick_grasp_orientation_for_place
            )
            self.assertFalse(config.manipulation.return_home_after_pick)
            self.assertTrue(pipeline.machine.config.manipulation.return_home_after_pick)
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.post_motion_hold_duration,
                0.75,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.post_motion_joint_error_tolerance,
                0.030,
            )
            self.assertTrue(
                pipeline.machine.arm_executor.config.require_close_progress_for_motion
            )
            self.assertTrue(
                pipeline.machine.arm_executor.config.fail_on_strict_post_motion_state_unavailable
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.settle_to_segment_start_skip_error_tolerance,
                0.005,
            )
            self.assertIn(
                "return_home_after_retreat",
                pipeline.machine.arm_executor.config.strict_post_motion_hold_segments,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.motion_time_scale,
                config.manipulation.arm_motion_time_scale,
            )
            self.assertAlmostEqual(config.manipulation.arm_motion_time_scale, 0.50)
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.pick_approach_motion_time_scale,
                config.manipulation.pick_approach_motion_time_scale,
            )
            self.assertAlmostEqual(
                config.manipulation.pick_approach_motion_time_scale,
                0.50,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_move_to_pre_place_motion_time_scale,
                1.00,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_approach_motion_time_scale,
                1.50,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_retreat_motion_time_scale,
                1.00,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_release_joint_error_tolerance,
                0.025,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_release_joint_velocity_tolerance,
                0.03,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.place_release_stability_window_duration,
                0.30,
            )
            self.assertIn(
                "approach_to_place",
                pipeline.machine.arm_executor.config.unskippable_post_motion_hold_segments,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.arm_command_dt,
                0.02,
            )
            self.assertIsInstance(pipeline.machine.verifier, FullPhysicsVerifier)

    def test_pct_multifloor_factory_relaxes_arm_post_motion_convergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                locomotion=LocomotionPolicySettings(
                    policy_profile="pct_multifloor",
                    locomotion_task=PCT_MULTIFLOOR_LOCOMOTION_TASK,
                    locomotion_checkpoint=checkpoint,
                ),
            )
            spec = JsonTaskProvider().load(task_path)

            pipeline = create_full_physics_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=root / "episode",
                simulation=InMemorySimulationRuntime(),  # type: ignore[arg-type]
            )

            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.post_motion_hold_duration,
                1.00,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.post_motion_joint_error_tolerance,
                0.065,
            )
            self.assertTrue(
                pipeline.machine.config.manipulation.settle_object_before_navigation
            )
            self.assertTrue(
                pipeline.machine.config.manipulation.settle_base_before_navigation
            )
            self.assertEqual(
                pipeline.machine.config.limits.navigation,
                12000,
            )
            self.assertEqual(
                pipeline.machine.config.limits.episode,
                24000,
            )
            self.assertEqual(
                _locomotion_runtime_kwargs(config)["standing_command_threshold"],
                0.08,
            )
            self.assertEqual(
                _locomotion_runtime_kwargs(config)["policy_action_warmup_steps"],
                50,
            )

    def test_full_physics_can_settle_object_before_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                manipulation=ManipulationSettings(
                    settle_object_before_navigation=True,
                    object_settle_max_steps=5,
                    object_settle_required_stable_steps=2,
                ),
            )
            spec = JsonTaskProvider().load(task_path)
            simulation = InMemorySimulationRuntime()
            pipeline = create_full_physics_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=root / "episode",
                simulation=simulation,  # type: ignore[arg-type]
            )

            build = pipeline.machine.tick(simulation.read())
            simulation.apply(build.action)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)

            reset = pipeline.machine.tick(simulation.read())
            simulation.apply(reset.action)
            simulation.step(render=False)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertIn("object_settle_started", {event.name for event in reset.events})

            settling = pipeline.machine.tick(simulation.read())
            simulation.apply(settling.action)
            simulation.step(render=False)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)

            stabilized = pipeline.machine.tick(simulation.read())
            self.assertEqual(pipeline.machine.state, PipelineState.PLAN_NAV_TO_PICK)
            self.assertIn(
                "object_initial_pose_stabilized",
                {event.name for event in stabilized.events},
            )
            self.assertTrue(
                simulation.read().metadata["object_settle_final_report"]["applied"]
            )

    def test_full_physics_waits_for_base_stability_before_pct_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                manipulation=ManipulationSettings(
                    settle_object_before_navigation=True,
                    settle_base_before_navigation=True,
                    object_settle_max_steps=8,
                    object_settle_required_stable_steps=2,
                ),
            )
            spec = JsonTaskProvider().load(task_path)
            simulation = InMemorySimulationRuntime()
            pipeline = create_full_physics_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=root / "episode",
                simulation=simulation,  # type: ignore[arg-type]
            )

            build = pipeline.machine.tick(simulation.read())
            simulation.apply(build.action)
            reset = pipeline.machine.tick(simulation.read())
            simulation.apply(reset.action)
            simulation.step(render=False)

            moving_base = replace(
                simulation.read(),
                robot_root_velocity=(0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            settling = pipeline.machine.tick(moving_base)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            report = settling.action.metadata["object_settle_report"]
            self.assertFalse(report["base_stable"])
            self.assertEqual(report["stable_steps"], 0)

            first_stable = pipeline.machine.tick(simulation.read())
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertTrue(
                first_stable.action.metadata["object_settle_report"]["base_stable"]
            )

            stabilized = pipeline.machine.tick(simulation.read())
            self.assertEqual(pipeline.machine.state, PipelineState.PLAN_NAV_TO_PICK)
            self.assertIn(
                "object_initial_pose_stabilized",
                {event.name for event in stabilized.events},
            )
            stabilized_event = next(
                event
                for event in stabilized.events
                if event.name == "object_initial_pose_stabilized"
            )
            stability = stabilized_event.metadata["stability"]
            self.assertTrue(stability["base_settle_enabled"])
            self.assertTrue(stability["base_stable"])

    def test_full_physics_verifier_checks_navigation_gripper_and_object_tcp_distance(self) -> None:
        spec = JsonTaskProvider().load(PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json")
        verifier = FullPhysicsVerifier(NavigationEpisodeVerifier())
        pick_state = SimulationState(
            step_index=1,
            timestamp=0.0,
            robot_root_pose=(
                spec.pick_goal.x,
                spec.pick_goal.y,
                0.35,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            robot_root_velocity=(0.0,) * 6,
            tcp_pose=(0.4, 0.0, 0.84, 1.0, 0.0, 0.0, 0.0),
            object_pose=(0.45, 0.0, 0.87, 1.0, 0.0, 0.0, 0.0),
            object_velocity=(0.0,) * 6,
            metadata={
                "body_velocity": (0.0, 0.0, 0.0),
                "gripper_close_apply_count": 2,
            },
        )

        self.assertTrue(verifier.verify_pick_reachable(pick_state, spec).success)
        pick_result = verifier.verify_pick_success(pick_state, spec)
        self.assertTrue(pick_result.success)
        self.assertEqual(pick_result.failure_reason, "")
        self.assertEqual(
            pick_result.metadata["validation_mode"],
            "main_pick_lift_contact_and_stability",
        )
        self.assertGreaterEqual(pick_result.metadata["object_lift_height_m"], 0.04)

        not_lifted_state = replace(
            pick_state,
            object_pose=(
                0.45,
                0.0,
                spec.object_initial_pose[2] + 0.01,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        not_lifted = verifier.verify_pick_success(not_lifted_state, spec)
        self.assertFalse(not_lifted.success)
        self.assertEqual(not_lifted.failure_reason, "object_not_lifted")

        returned_home_state = replace(
            not_lifted_state,
            tcp_pose=(
                0.45,
                0.0,
                spec.object_initial_pose[2] + 0.02,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            metadata={
                **not_lifted_state.metadata,
                "pick_peak_object_lift_height_m": 0.10,
                "pick_peak_object_pose": (
                    0.45,
                    0.0,
                    spec.object_initial_pose[2] + 0.10,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                "pick_peak_step_index": 123,
            },
        )
        returned_home = verifier.verify_pick_success(returned_home_state, spec)
        self.assertTrue(returned_home.success)
        self.assertAlmostEqual(
            returned_home.metadata["verified_object_lift_height_m"],
            0.10,
        )

        side_retreat_state = replace(
            not_lifted_state,
            object_pose=(
                spec.object_initial_pose[0] + 0.06,
                spec.object_initial_pose[1],
                spec.object_initial_pose[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            tcp_pose=(
                spec.object_initial_pose[0] + 0.06,
                spec.object_initial_pose[1],
                spec.object_initial_pose[2] + 0.01,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            metadata={
                **not_lifted_state.metadata,
                "pick_has_lift_segment": False,
                "pick_has_retreat_segment": True,
            },
        )
        side_retreat = verifier.verify_pick_success(side_retreat_state, spec)
        self.assertTrue(side_retreat.success)
        self.assertEqual(
            side_retreat.metadata["validation_mode"],
            "side_retreat_contact_and_stability",
        )

        no_retreat_state = replace(
            side_retreat_state,
            object_pose=(
                spec.object_initial_pose[0] + 0.01,
                spec.object_initial_pose[1],
                spec.object_initial_pose[2],
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        no_retreat = verifier.verify_pick_success(no_retreat_state, spec)
        self.assertFalse(no_retreat.success)
        self.assertEqual(no_retreat.failure_reason, "object_not_retreated")

    def test_full_physics_verifier_rejects_place_release_ejection(self) -> None:
        spec = JsonTaskProvider().load(
            PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
        )
        verifier = FullPhysicsVerifier(NavigationEpisodeVerifier())
        target_xyz = tuple(float(value) for value in spec.place_target_pose[:3])
        stable_state = SimulationState(
            step_index=200,
            timestamp=4.0,
            robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
            object_pose=(*target_xyz, 1.0, 0.0, 0.0, 0.0),
            object_velocity=(0.0,) * 6,
            metadata={
                "gripper_open_apply_count": 40,
                "place_open_apply_count_delta": 35,
                "place_release_observed": True,
                "place_release_object_pose": (*target_xyz, 1.0, 0.0, 0.0, 0.0),
                "place_expected_release_object_center": target_xyz,
                "place_release_velocity_sample_count": 80,
                "place_peak_object_linear_speed_mps": 0.04,
                "place_peak_object_horizontal_speed_mps": 0.02,
                "place_peak_object_angular_speed_rps": 0.40,
                "place_max_horizontal_displacement_m": 0.01,
            },
        )

        stable = verifier.verify_place_success(stable_state, spec)
        ejected = verifier.verify_place_success(
            replace(
                stable_state,
                metadata={
                    **stable_state.metadata,
                    "place_peak_object_linear_speed_mps": 0.62,
                    "place_peak_object_horizontal_speed_mps": 0.47,
                    "place_peak_object_angular_speed_rps": 40.8,
                    "place_max_horizontal_displacement_m": 0.063,
                },
            ),
            spec,
        )
        released_too_high = verifier.verify_place_success(
            replace(
                stable_state,
                metadata={
                    **stable_state.metadata,
                    "place_release_object_pose": (
                        target_xyz[0],
                        target_xyz[1],
                        target_xyz[2] + 0.045,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                },
            ),
            spec,
        )

        self.assertTrue(stable.success)
        self.assertEqual(
            stable.metadata["validation_mode"],
            "contact_release_pose_dynamics_and_final_stability",
        )
        self.assertFalse(ejected.success)
        self.assertEqual(ejected.failure_reason, "place_release_ejected")
        self.assertFalse(released_too_high.success)
        self.assertEqual(released_too_high.failure_reason, "place_release_pose_error")

        non_finite_pose = verifier.verify_place_success(
            replace(stable_state, object_pose=(float("nan"), *stable_state.object_pose[1:])),
            spec,
        )
        non_finite_velocity = verifier.verify_place_success(
            replace(stable_state, object_velocity=(float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0)),
            spec,
        )
        self.assertFalse(non_finite_pose.success)
        self.assertEqual(non_finite_pose.failure_reason, "object_out_of_place")
        self.assertFalse(non_finite_velocity.success)
        self.assertEqual(non_finite_velocity.failure_reason, "place_dynamics_unavailable")

    def test_full_physics_mode_reports_stable_success_with_lock_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                keep_window_open=True,
                manipulation=ManipulationSettings(return_home_after_pick=True),
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=12,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(
                    gripper,
                    config=SegmentedArmExecutorConfig(
                        sim_dt=0.05,
                        gripper_move_duration=0.10,
                        gripper_hold_duration=0.05,
                    ),
                ),
                gripper=gripper,
                verifier=DryRunEpisodeVerifier(),
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["execution_mode"], "full_physics")
            self.assertEqual(
                summary["success_semantics"],
                "stable_physical_execution_with_base_support_lock",
            )
            self.assertTrue(summary["stable_physics_success"])
            self.assertTrue(summary["carry_control_success"])
            self.assertTrue(summary["manipulation_apply_success"])
            self.assertTrue(summary["manipulation_base_lock_requested"])
            self.assertTrue(summary["manipulation_support_joint_lock_requested"])
            self.assertTrue(summary["object_carry_verified"])
            self.assertTrue(summary["physical_manipulation_success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(
                summary["simulation_report"]["object_prepare_for_pick_report"]["wake_policy"],
                "physx_contact",
            )
            self.assertTrue(summary["simulation_report"]["terminal_hold_report"]["paused"])
            self.assertFalse(pipeline.simulation.closed)
            event_names = [
                json.loads(line)["name"]
                for line in (root / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("pick_success", event_names)
            self.assertIn("object_prepared_for_pick", event_names)
            self.assertIn("carry_control_success", event_names)
            self.assertIn("place_success", event_names)
            self.assertIn("pick_base_settle_start", event_names)
            self.assertIn("pick_base_settle_complete", event_names)

    def test_carry_holds_last_pick_pose_when_return_home_was_not_inserted(self) -> None:
        machine = object.__new__(FullPhysicsStateMachine)
        machine.config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs",
            manipulation=ManipulationSettings(return_home_after_pick=True),
        )
        machine._carry_arm_home_target = None
        joint_names = tuple(f"arm_joint{index}" for index in range(1, 7))
        final_q = (0.11, -0.22, 0.33, -0.44, 0.55, -0.66)
        plan = ArmPlan(
            operation="pick",
            joint_trajectory=((0.0,) * 6, final_q),
            metadata={
                "joint_names": joint_names,
                "segments": [
                    {
                        "name": "retreat_object",
                        "type": "motion",
                        "trajectory": {"q": [(0.0,) * 6, final_q]},
                    }
                ],
            },
        )

        machine._configure_carry_arm_home_target(
            plan,
            {
                "inserted": False,
                "reason": "planned_retreat_or_reverse_return_present",
                "joint_names": joint_names,
                "home_positions": (0.0,) * 6,
            },
        )

        self.assertEqual(
            machine._carry_arm_home_target["source"],
            "pick_final_arm_pose",
        )
        self.assertEqual(
            machine._carry_arm_home_target["arm_joint_positions"],
            final_q,
        )
        self.assertFalse(
            machine._carry_arm_home_target["return_home_inserted"]
        )

    def test_place_carry_handoff_keeps_navigation_locks_for_pct_stair_float(self) -> None:
        machine = object.__new__(FullPhysicsStateMachine)
        machine.config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs",
            navigation=NavigationSettings(pct_stair_float_enabled=True),
        )
        state = SimulationState(
            step_index=9,
            timestamp=0.45,
            robot_root_pose=(0.24, -0.05, 3.28, 0.9238795, 0.0, 0.0, -0.3826834),
            robot_root_velocity=(0.0,) * 6,
        )

        action = machine._place_carry_handoff_action(state)

        self.assertEqual(action.source, "place_carry_handoff")
        self.assertTrue(action.metadata["place_carry_handoff_hold"])
        self.assertTrue(action.metadata["navigation_base_pose_lock"])
        self.assertTrue(action.metadata["navigation_full_body_joint_lock"])
        self.assertTrue(action.metadata["navigation_carry_object_follow"])
        self.assertEqual(
            action.metadata["navigation_base_pose_lock_phase"],
            "place_carry_handoff",
        )
        self.assertEqual(
            action.metadata["navigation_base_pose_lock_xyzyaw"][:3],
            (0.24, -0.05, 3.28),
        )

    def test_pick_smoke_stops_after_physical_pick_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                pick_smoke=True,
                keep_window_open=True,
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=13,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(
                    gripper,
                    config=SegmentedArmExecutorConfig(
                        sim_dt=0.05,
                        gripper_move_duration=0.10,
                        gripper_hold_duration=0.05,
                    ),
                ),
                gripper=gripper,
                verifier=DryRunEpisodeVerifier(),
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["execution_mode"], "pick_smoke")
            self.assertEqual(
                summary["success_semantics"],
                "physical_nav_to_pick_and_pick_only",
            )
            self.assertTrue(summary["physical_navigation_success"])
            self.assertTrue(summary["physical_manipulation_success"])
            self.assertTrue(summary["manipulation_apply_success"])
            self.assertFalse(summary["carry_control_success"])
            self.assertFalse(summary["object_carry_verified"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.PLAN_NAV_TO_PICK.value,
                    PipelineState.EXEC_NAV_TO_PICK.value,
                    PipelineState.VERIFY_PICK_REACHABLE.value,
                    PipelineState.PLAN_PICK.value,
                    PipelineState.EXEC_PICK.value,
                    PipelineState.VERIFY_PICK_SUCCESS.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            event_names = [
                json.loads(line)["name"]
                for line in (root / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("pick_success", event_names)
            self.assertIn("pick_smoke_success", event_names)
            self.assertNotIn("nav_to_place_start", event_names)
            self.assertNotIn("place_base_settle_start", event_names)
            self.assertNotIn("place_base_settle_complete", event_names)
            self.assertNotIn("object_lift_success", event_names)
            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(frames[0]["pipeline_state"], PipelineState.BUILD_STAGE.value)
            self.assertTrue(frames[0]["action"]["metadata"]["skip_physics_step"])
            self.assertEqual(
                frames[0]["action"]["metadata"]["skip_reason"],
                "await_object_pose_reset_before_first_physics_step",
            )
            pick_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PICK.value
            ]
            place_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
            ]
            nav_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"]
                in {
                    PipelineState.EXEC_NAV_TO_PICK.value,
                    PipelineState.EXEC_NAV_TO_PLACE.value,
                }
            ]
            carry_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_NAV_TO_PLACE.value
            ]
            manipulation_boundary_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"]
                in {
                    PipelineState.PLAN_PICK.value,
                    PipelineState.EXEC_PICK.value,
                    PipelineState.VERIFY_PICK_SUCCESS.value,
                    PipelineState.PLAN_PLACE.value,
                    PipelineState.EXEC_PLACE.value,
                    PipelineState.VERIFY_PLACE_SUCCESS.value,
                }
            ]
            self.assertTrue(
                all(frame["action"]["metadata"]["manipulation_base_lock"] for frame in pick_frames)
            )
            self.assertTrue(
                all(frame["action"]["metadata"]["manipulation_base_lock"] for frame in place_frames)
            )
            self.assertTrue(
                all(
                    not frame["action"]["metadata"]["manipulation_base_lock"]
                    for frame in nav_frames
                )
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"]["manipulation_base_lock"]
                    for frame in manipulation_boundary_frames
                )
            )
            carry_arm_target = list(
                summary["carry_arm_home_target"]["arm_joint_positions"]
            )
            self.assertFalse(place_frames)
            self.assertFalse(carry_frames)
            self.assertTrue(
                all(
                    frame["action"]["arm_joint_positions"] == carry_arm_target
                    for frame in carry_frames
                )
            )
            self.assertEqual(
                summary["carry_arm_home_target"]["source"],
                "pick_final_arm_pose",
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("carry_arm_home_phase")
                    == PipelineState.EXEC_NAV_TO_PLACE.value
                    for frame in carry_frames
                )
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get(
                        "carry_gripper_hold_position_source"
                    )
                    == "forced_close_target_for_carry"
                    for frame in carry_frames
                )
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("gripper_joint_positions")
                    == [0.0, 0.0]
                    for frame in carry_frames
                )
            )
            pick_settle_frames = [
                frame
                for frame in frames
                if frame["action"]["source"] == "pick_base_settle"
            ]
            self.assertEqual(
                len(pick_settle_frames),
                config.manipulation.base_lock_settle_steps,
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("manipulation_base_lock")
                    and frame["action"]["metadata"].get("manipulation_support_joint_lock")
                    for frame in pick_settle_frames
                )
            )
            self.assertTrue(pick_frames)
            final_action = frames[-1]["action"]
            self.assertTrue(final_action["metadata"]["terminal_hold"])
            self.assertTrue(final_action["metadata"]["manipulation_base_lock"])
            self.assertTrue(final_action["metadata"]["manipulation_support_joint_lock"])
            pipeline.simulation.close()

    def test_full_physics_failure_terminal_frame_keeps_robot_locked_and_paused(self) -> None:
        class PlaceFailureVerifier(DryRunEpisodeVerifier):
            def verify_place_success(self, state, episode_spec):
                del state, episode_spec
                return VerificationResult(False, "object_out_of_place")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                keep_window_open=True,
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            simulation = InMemorySimulationRuntime()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=13,
                simulation=simulation,
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=PlaceFailureVerifier(),
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()
            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "object_out_of_place")
            self.assertFalse(summary["lerobot_training_eligible"])
            self.assertTrue(summary["lerobot_export_skipped"])
            self.assertFalse((root / "episode" / "lerobot_manifest.json").exists())
            self.assertFalse((root / "episode" / "lerobot_dataset").exists())
            self.assertTrue(summary["simulation_report"]["terminal_hold_report"]["paused"])
            final_action = frames[-1]["action"]
            self.assertTrue(final_action["metadata"]["terminal_hold"])
            self.assertTrue(final_action["metadata"]["manipulation_base_lock"])
            self.assertTrue(final_action["metadata"]["manipulation_support_joint_lock"])
            self.assertFalse(simulation.closed)
            simulation.close()

    def test_simulation_smoke_stops_after_real_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                simulation_smoke=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=0,
                episode_dir=Path(tmp_dir) / "episode",
            )
            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(summary["execution_mode"], "simulation_smoke")
            self.assertEqual(summary["success_semantics"], "stage_build_and_reset_only")
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            event_names = [
                json.loads(line)["name"]
                for line in (Path(tmp_dir) / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn("simulation_smoke_success", event_names)

    def test_simulation_smoke_keep_window_skips_pose_inspection_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                simulation_smoke=True,
                keep_window_open=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=0,
                episode_dir=Path(tmp_dir) / "episode",
            )
            summary = pipeline.run_episode()
            frames = [
                json.loads(line)
                for line in (Path(tmp_dir) / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertTrue(summary["success"])
            self.assertEqual(pipeline.simulation.step_calls, 1)
            skipped = [
                frame
                for frame in frames
                if frame["action"]["metadata"].get("skip_physics_step")
            ]
            self.assertEqual(
                [frame["pipeline_state"] for frame in skipped],
                [PipelineState.RESET_EPISODE.value, PipelineState.CLEANUP_EPISODE.value],
            )

    def test_manipulation_smoke_uses_segmented_executor_without_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            episode_dir = Path(tmp_dir) / "episode"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                manipulation_smoke=True,
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_manipulation_smoke_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=3,
                episode_dir=episode_dir,
            )
            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertFalse(summary["physical_manipulation_success"])
            self.assertEqual(summary["execution_mode"], "manipulation_smoke")
            self.assertEqual(
                summary["success_semantics"],
                "segmented_manipulation_contract_only",
            )
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.PLAN_PICK.value,
                    PipelineState.EXEC_PICK.value,
                    PipelineState.VERIFY_PICK_SUCCESS.value,
                    PipelineState.PLAN_PLACE.value,
                    PipelineState.EXEC_PLACE.value,
                    PipelineState.VERIFY_PLACE_SUCCESS.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            self.assertFalse((episode_dir / "lerobot_manifest.json").exists())

            event_names = [
                json.loads(line)["name"]
                for line in (episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("manipulation_smoke_start", event_names)
            self.assertIn("manipulation_smoke_pick_success", event_names)
            self.assertIn("manipulation_smoke_success", event_names)
            self.assertEqual(event_names.count("gripper_close"), 1)
            self.assertEqual(event_names.count("gripper_open"), 1)

            frames = [
                json.loads(line)
                for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertFalse(
                any(
                    frame["pipeline_state"]
                    in {
                        PipelineState.PLAN_NAV_TO_PICK.value,
                        PipelineState.EXEC_NAV_TO_PICK.value,
                        PipelineState.PLAN_NAV_TO_PLACE.value,
                        PipelineState.EXEC_NAV_TO_PLACE.value,
                    }
                    for frame in frames
                )
            )
            self.assertTrue(
                any(
                    frame["action"]["source"] == "arm_pick"
                    and frame["action"]["metadata"].get("segment_name") == "lift_object"
                    for frame in frames
                )
            )
            place_approach_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
                and frame["action"]["metadata"].get("segment_name") == "approach_to_place"
            ]
            self.assertGreater(len(place_approach_frames), 0)
            self.assertTrue(
                all(frame["action"]["gripper_command"] == "close" for frame in place_approach_frames)
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("carry_gripper_hold")
                    for frame in place_approach_frames
                )
            )
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("gripper_joint_positions") == [0.0, 0.0]
                    for frame in place_approach_frames
                )
            )

    def test_manipulation_apply_smoke_verifies_joint_apply_contract_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            episode_dir = Path(tmp_dir) / "episode"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                manipulation_apply_smoke=True,
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            simulation = _ApplySmokeSpyRuntime()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=4,
                simulation=simulation,
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(
                    gripper,
                    config=SegmentedArmExecutorConfig(
                        sim_dt=0.05,
                        gripper_move_duration=0.10,
                        gripper_hold_duration=0.05,
                    ),
                ),
                gripper=gripper,
                verifier=ManipulationApplySmokeVerifier(),
                recorder=JsonlEpisodeRecorder(episode_dir),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertFalse(summary["physical_manipulation_success"])
            self.assertTrue(summary["manipulation_apply_success"])
            self.assertEqual(summary["execution_mode"], "manipulation_apply_smoke")
            self.assertEqual(summary["success_semantics"], "isaac_joint_action_apply_only")
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.PLAN_PICK.value,
                    PipelineState.EXEC_PICK.value,
                    PipelineState.VERIFY_PICK_SUCCESS.value,
                    PipelineState.PLAN_PLACE.value,
                    PipelineState.EXEC_PLACE.value,
                    PipelineState.VERIFY_PLACE_SUCCESS.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            report = summary["simulation_report"]
            self.assertGreater(report["joint_action_apply_count"], 0)
            self.assertGreater(report["arm_joint_action_apply_count"], 0)
            self.assertGreater(report["gripper_close_apply_count"], 0)
            self.assertGreater(report["gripper_open_apply_count"], 0)
            self.assertEqual(simulation.step_calls, summary["duration_steps"])

            event_names = [
                json.loads(line)["name"]
                for line in (episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("manipulation_apply_smoke_start", event_names)
            self.assertIn("manipulation_apply_smoke_pick_apply_success", event_names)
            self.assertIn("manipulation_apply_smoke_pick_success", event_names)
            self.assertIn("manipulation_apply_smoke_place_apply_success", event_names)
            self.assertIn("manipulation_apply_smoke_success", event_names)
            self.assertNotIn("manipulation_smoke_success", event_names)
            self.assertNotIn("object_lift_success", event_names)
            self.assertNotIn("place_success", event_names)

    def test_manipulation_apply_smoke_can_use_external_curobo_plan_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick_plan.json"
            place_path = root / "place_plan.json"
            pick_path.write_text(json.dumps(_external_pick_payload()), encoding="utf-8")
            place_path.write_text(json.dumps(_external_place_payload()), encoding="utf-8")

            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            episode_dir = root / "episode"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                manipulation_apply_smoke=True,
                pick_plan_json=pick_path,
                place_plan_json=place_path,
                manipulation=ManipulationSettings(return_home_after_pick=True),
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_manipulation_apply_smoke_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=5,
                episode_dir=episode_dir,
                simulation=_ApplySmokeSpyRuntime(),  # type: ignore[arg-type]
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(summary["success_semantics"], "isaac_joint_action_apply_only")
            self.assertEqual(summary["latest_planner_result"]["phase"], "place")
            self.assertEqual(summary["latest_planner_result"]["source_plan_json"], str(place_path))

            frames = [
                json.loads(line)
                for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            segment_names = {
                frame["action"]["metadata"].get("segment_name")
                for frame in frames
                if frame["action"]["source"] in {"arm_pick", "arm_place"}
            }
            self.assertIn("json_pick_motion", segment_names)
            self.assertIn("return_home_reverse_json_pick_motion", segment_names)
            self.assertNotIn("hold_home_before_carry", segment_names)
            self.assertIn("json_place_motion", segment_names)
            self.assertIn("return_home_after_place", segment_names)
            self.assertTrue(
                summary["latest_planner_result"]["place_return_home"]["inserted"]
            )
            self.assertFalse(summary["latest_planner_result"]["start_state_check"]["mismatch_detected"])
            self.assertFalse(summary["latest_planner_result"]["start_state_transition"]["inserted"])
            carry_home_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"]
                in {
                    PipelineState.VERIFY_PICK_SUCCESS.value,
                    PipelineState.PLAN_PLACE.value,
                }
            ]
            self.assertTrue(
                any(
                    frame["action"]["metadata"].get("carry_arm_home_hold")
                    and frame["action"]["arm_joint_positions"] == [0.0] * 6
                    for frame in carry_home_frames
                )
            )

    def test_explicit_pick_return_home_uses_reverse_executed_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick_plan.json"
            place_path = root / "place_plan.json"
            pick_payload = _external_pick_payload()
            pick_payload["segments"][0]["trajectory"]["q"][-1][0] = 1.0
            pick_payload["segments"].append(
                _external_motion_segment(
                    "lift_object",
                    (
                        (1.00, 0.04, 0.03, 0.02, 0.01, 0.00),
                        (1.10, 0.08, 0.06, 0.04, 0.02, 0.01),
                    ),
                )
            )
            place_payload = _external_place_payload()
            place_payload["segments"][0]["trajectory"]["q"][0] = [0.0] * 6
            pick_path.write_text(json.dumps(pick_payload), encoding="utf-8")
            place_path.write_text(json.dumps(place_payload), encoding="utf-8")

            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            episode_dir = root / "episode"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                manipulation_apply_smoke=True,
                pick_plan_json=pick_path,
                place_plan_json=place_path,
                manipulation=ManipulationSettings(return_home_after_pick=True),
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_manipulation_apply_smoke_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=6,
                episode_dir=episode_dir,
                simulation=_ApplySmokeSpyRuntime(),  # type: ignore[arg-type]
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            check = summary["latest_planner_result"]["start_state_check"]
            self.assertFalse(check["mismatch_detected"])
            self.assertAlmostEqual(check["peak_joint"]["actual_position"], 0.0)
            self.assertAlmostEqual(check["peak_joint"]["target_position"], 0.0)
            self.assertFalse(summary["latest_planner_result"]["start_state_transition"]["inserted"])
            event_names = [
                json.loads(line)["name"]
                for line in (episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("pick_return_home_appended", event_names)
            self.assertNotIn("place_plan_start_state_mismatch", event_names)
            frames = [
                json.loads(line)
                for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                any(
                    frame["action"]["metadata"].get("segment_name")
                    == "return_home_reverse_json_pick_motion"
                    and frame["action"]["arm_joint_positions"] == [0.0] * 6
                    for frame in frames
                    if frame["pipeline_state"] == PipelineState.EXEC_PICK.value
                )
            )
            self.assertTrue(
                any(
                    frame["action"]["metadata"].get("segment_name")
                    == "return_home_reverse_lift_object"
                    for frame in frames
                    if frame["pipeline_state"] == PipelineState.EXEC_PICK.value
                )
            )
            first_place_frame = next(
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
            )
            self.assertEqual(
                first_place_frame["action"]["metadata"]["segment_name"],
                "json_place_motion_settle_to_start",
            )
            self.assertEqual(
                first_place_frame["action"]["metadata"]["parent_segment_name"],
                "json_place_motion",
            )
            first_place_motion_frame = next(
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
                and frame["action"]["metadata"].get("segment_type") == "motion"
            )
            self.assertEqual(
                first_place_motion_frame["action"]["metadata"]["segment_name"],
                "json_place_motion",
            )
            self.assertAlmostEqual(first_place_frame["action"]["arm_joint_positions"][0], 0.0)

    def test_place_plan_start_state_mismatch_is_recorded_when_return_home_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick_plan.json"
            place_path = root / "place_plan.json"
            pick_payload = _external_pick_payload()
            pick_payload["segments"][0]["trajectory"]["q"][-1][0] = 1.0
            place_payload = _external_place_payload()
            place_payload["segments"][0]["trajectory"]["q"][0] = [0.0] * 6
            pick_path.write_text(json.dumps(pick_payload), encoding="utf-8")
            place_path.write_text(json.dumps(place_payload), encoding="utf-8")

            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            episode_dir = root / "episode"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                manipulation_apply_smoke=True,
                pick_plan_json=pick_path,
                place_plan_json=place_path,
                manipulation=ManipulationSettings(return_home_after_pick=False),
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_manipulation_apply_smoke_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=6,
                episode_dir=episode_dir,
                simulation=_ApplySmokeSpyRuntime(),  # type: ignore[arg-type]
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            check = summary["latest_planner_result"]["start_state_check"]
            self.assertTrue(check["mismatch_detected"])
            self.assertEqual(check["recommended_failure_reason"], "place_plan_start_state_mismatch")
            self.assertEqual(check["peak_joint"]["joint_name"], "arm_joint1")
            self.assertAlmostEqual(check["peak_joint"]["actual_position"], 1.0)
            self.assertAlmostEqual(check["peak_joint"]["target_position"], 0.0)
            transition = summary["latest_planner_result"]["start_state_transition"]
            self.assertTrue(transition["inserted"])
            self.assertEqual(transition["segment_name"], "start_state_transition_to_place_plan")
            event_names = [
                json.loads(line)["name"]
                for line in (episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("place_plan_start_state_mismatch", event_names)
            self.assertIn("place_plan_start_state_transition_inserted", event_names)
            frames = [
                json.loads(line)
                for line in (episode_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            first_place_frame = next(
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
            )
            self.assertEqual(
                first_place_frame["action"]["metadata"]["segment_name"],
                "start_state_transition_to_place_plan_settle_to_start",
            )
            self.assertEqual(
                first_place_frame["action"]["metadata"]["parent_segment_name"],
                "start_state_transition_to_place_plan",
            )
            first_place_motion_frame = next(
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_PLACE.value
                and frame["action"]["metadata"].get("segment_type") == "motion"
            )
            self.assertEqual(
                first_place_motion_frame["action"]["metadata"]["segment_name"],
                "start_state_transition_to_place_plan",
            )
            self.assertAlmostEqual(first_place_frame["action"]["arm_joint_positions"][0], 1.0)

    def test_place_plan_start_state_mismatch_can_fail_with_config_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pick_path = root / "pick_plan.json"
            place_path = root / "place_plan.json"
            pick_payload = _external_pick_payload()
            pick_payload["segments"][0]["trajectory"]["q"][-1][0] = 1.0
            place_payload = _external_place_payload()
            place_payload["segments"][0]["trajectory"]["q"][0] = [0.0] * 6
            pick_path.write_text(json.dumps(pick_payload), encoding="utf-8")
            place_path.write_text(json.dumps(place_payload), encoding="utf-8")

            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                manipulation_apply_smoke=True,
                pick_plan_json=pick_path,
                place_plan_json=place_path,
                manipulation=ManipulationSettings(
                    return_home_after_pick=False,
                    fail_on_place_plan_start_state_mismatch=True,
                ),
            )
            spec = JsonTaskProvider().load(task_path)
            pipeline = create_manipulation_apply_smoke_pipeline(
                config=config,
                episode_spec=spec,
                episode_seed=7,
                episode_dir=root / "episode",
                simulation=_ApplySmokeSpyRuntime(),  # type: ignore[arg-type]
            )

            summary = pipeline.run_episode()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "place_plan_start_state_mismatch")
            self.assertEqual(summary["failure_metadata"]["peak_joint"]["joint_name"], "arm_joint1")
            self.assertNotIn(PipelineState.EXEC_PLACE.value, summary["state_trace"])

    def test_navigation_verifier_defaults_to_xy_only_handoff(self) -> None:
        spec = JsonTaskProvider().load(PROJECT_ROOT / "tasks/nav_smoke_example.json")
        verifier = NavigationEpisodeVerifier()
        stable_state = SimulationState(
            step_index=1,
            timestamp=0.0,
            robot_root_pose=(-1.70, 0.10, 0.35, 0.0, 0.0, 0.0, 1.0),
            robot_root_velocity=(0.0,) * 6,
            metadata={"body_velocity": (0.0, 0.0, 0.0)},
        )
        moving_state = SimulationState(
            step_index=1,
            timestamp=0.0,
            robot_root_pose=stable_state.robot_root_pose,
            robot_root_velocity=(0.0,) * 6,
            metadata={"body_velocity": (0.2, 0.0, 0.0)},
        )

        self.assertTrue(verifier.verify_pick_reachable(stable_state, spec).success)
        moving_result = verifier.verify_pick_reachable(moving_state, spec)
        self.assertTrue(moving_result.success)
        self.assertEqual(moving_result.metadata["acceptance_mode"], "xy_only")
        self.assertFalse(moving_result.metadata["base_stable"])

    def test_navigation_verifier_can_run_strict_diagnostics(self) -> None:
        spec = JsonTaskProvider().load(PROJECT_ROOT / "tasks/nav_smoke_example.json")
        verifier = NavigationEpisodeVerifier(
            require_yaw_alignment=True,
            require_stable_base=True,
        )
        state = SimulationState(
            step_index=1,
            timestamp=0.0,
            robot_root_pose=(-1.70, 0.10, 0.35, 0.0, 0.0, 0.0, 1.0),
            robot_root_velocity=(0.0,) * 6,
            metadata={"body_velocity": (0.2, 0.0, 0.0)},
        )

        result = verifier.verify_pick_reachable(state, spec)

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["acceptance_mode"], "xy_yaw_stable")
        self.assertFalse(result.metadata["yaw_aligned"])
        self.assertFalse(result.metadata["base_stable"])

    def test_navigation_carry_smoke_holds_home_arm_and_closed_gripper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                navigation_carry_smoke=True,
            )
            base_spec = JsonTaskProvider().load(task_path)
            spec = replace(base_spec, start=base_spec.pick_goal)
            gripper = BinaryGripperController()
            simulation = InMemorySimulationRuntime()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=11,
                simulation=simulation,
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=NavigationEpisodeVerifier(),
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["execution_mode"], "navigation_carry_smoke")
            self.assertEqual(
                summary["success_semantics"],
                "physical_nav_to_place_with_arm_gripper_hold",
            )
            self.assertTrue(summary["carry_control_success"])
            self.assertFalse(summary["object_carry_verified"])
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.PLAN_NAV_TO_PLACE.value,
                    PipelineState.EXEC_NAV_TO_PLACE.value,
                    PipelineState.VERIFY_PLACE_REACHABLE.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            tracking = summary["simulation_report"]["arm_tracking_report"]
            self.assertGreater(tracking["sample_count"], 0)
            self.assertEqual(tracking["max_abs_error"], 0.0)

            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            carry_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] == PipelineState.EXEC_NAV_TO_PLACE.value
            ]
            self.assertGreater(len(carry_frames), 0)
            self.assertTrue(
                all(frame["action"]["arm_joint_positions"] == [0.0] * 6 for frame in carry_frames)
            )
            self.assertTrue(
                all(frame["action"]["gripper_command"] == "close" for frame in carry_frames)
            )

    def test_stair_locomotion_smoke_stops_after_navigation_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_smoke_example.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                stair_locomotion_smoke=True,
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=5,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=DryRunNavExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=NavigationEpisodeVerifier(),
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["execution_mode"], "stair_locomotion_smoke")
            self.assertEqual(
                summary["success_semantics"],
                "pure_physics_stair_locomotion_without_dwa_or_float",
            )
            self.assertFalse(summary["pure_physics_success"])
            self.assertEqual(
                summary["state_trace"],
                [
                    PipelineState.BUILD_STAGE.value,
                    PipelineState.RESET_EPISODE.value,
                    PipelineState.PLAN_NAV_TO_PICK.value,
                    PipelineState.EXEC_NAV_TO_PICK.value,
                    PipelineState.VERIFY_PICK_REACHABLE.value,
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )

    def test_component_exception_becomes_structured_failure(self) -> None:
        class RaisingPlanner:
            def plan(self, state, goal):
                del state, goal
                raise RuntimeError("planned failure")

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=Path(tmp_dir),
                dry_run=True,
            )
            pipeline = create_dry_run_pipeline(
                config=config,
                episode_spec=JsonTaskProvider().load(task_path),
                episode_seed=0,
                episode_dir=Path(tmp_dir) / "episode",
            )
            pipeline.machine.nav_planner = RaisingPlanner()
            summary = pipeline.run_episode()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure_reason"], "nav_to_pick_plan_failed")
            self.assertEqual(
                summary["failure_metadata"]["exception_type"],
                "RuntimeError",
            )

    def test_task_provider_preserves_existing_schema_values(self) -> None:
        spec = JsonTaskProvider().load(PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json")
        self.assertEqual(spec.object_prim_path, "/World/apple")
        self.assertIsNotNone(spec.place_goal)
        self.assertIsNotNone(spec.place_target_pose)
        self.assertEqual(spec.raw_task["carry"]["mode"], "contact")


def _external_arm_joint_names() -> tuple[str, ...]:
    return (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )


def _external_pick_payload() -> dict:
    return {
        "schema_version": 1,
        "planner": "curobo.MotionPlanner.plan_pose json pick",
        "joint_names": list(_external_arm_joint_names()),
        "tool_frame": "grasp_tcp_link",
        "object_prim_path": "/World/apple",
        "grasp_mode": "side",
        "segments": [
            _external_motion_segment(
                "json_pick_motion",
                (
                    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
                    (0.05, 0.04, 0.03, 0.02, 0.01, 0.00),
                ),
            ),
            {
                "name": "close_gripper",
                "type": "gripper",
                "joint_names": ["arm_joint7", "arm_joint8"],
                "target_position": [0.0, 0.0],
            },
        ],
    }


def _external_place_payload() -> dict:
    return {
        "schema_version": 1,
        "planner": "curobo.MotionPlanner.plan_pose json place",
        "joint_names": list(_external_arm_joint_names()),
        "tool_frame": "grasp_tcp_link",
        "object_prim_path": "/World/apple",
        "place_mode": "arm_place",
        "segments": [
            _external_motion_segment(
                "json_place_motion",
                (
                    (0.05, 0.04, 0.03, 0.02, 0.01, 0.00),
                    (0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
                ),
            ),
            {
                "name": "open_gripper",
                "type": "gripper",
                "joint_names": ["arm_joint7", "arm_joint8"],
                "target_position": [0.04, 0.04],
            },
        ],
    }


def _external_motion_segment(name: str, q_rows: tuple[tuple[float, ...], ...]) -> dict:
    return {
        "name": name,
        "type": "motion",
        "trajectory": {
            "time_from_start": [0.05 * index for index in range(len(q_rows))],
            "q": [list(row) for row in q_rows],
        },
    }


class _ApplySmokeSpyRuntime:
    def __init__(self) -> None:
        self.step_calls = 0
        self.apply_calls = 0
        self.closed = False
        self._episode_spec: EpisodeSpec | None = None
        self._last_action = RobotAction.idle()
        self._joint_positions = (0.0,) * 6
        self._metadata = self._base_metadata()

    def build(self, episode_spec: EpisodeSpec) -> None:
        self._episode_spec = episode_spec
        self._metadata["simulation_ready"] = True

    def reset(self, episode_spec: EpisodeSpec, *, seed: int) -> None:
        self._episode_spec = episode_spec
        self._metadata = {
            **self._base_metadata(),
            "simulation_ready": True,
            "seed": int(seed),
            "episode_reset_complete": True,
            # apply smoke 只验证 action 下发，不声明纯物理 provenance。
            "execution_provenance_verified": False,
        }
        self._last_action = RobotAction.idle(source="episode_reset")

    def read(self) -> SimulationState:
        return SimulationState(
            step_index=self.step_calls,
            timestamp=float(self.step_calls) * 0.05,
            robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
            joint_positions=self._joint_positions,
            joint_velocities=(0.0,) * len(self._joint_positions),
            object_pose=(0.0, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0),
            object_velocity=(0.0,) * 6,
            metadata={
                **self._metadata,
                "last_action_source": self._last_action.source,
            },
        )

    def apply(self, action: RobotAction) -> None:
        self.apply_calls += 1
        self._last_action = action
        if action.arm_joint_positions is not None:
            self._joint_positions = tuple(action.arm_joint_positions)
        if action.arm_joint_positions is None and action.gripper_command is None:
            return

        gripper_targeted = action.gripper_command in {"close", "open"} or (
            "gripper_joint_positions" in action.metadata
        )
        report = {
            "applied": True,
            "source": action.source,
            "arm_targeted": action.arm_joint_positions is not None,
            "gripper_targeted": gripper_targeted,
            "uses_direct_joint_state": False,
            "world_step_owned_by_pipeline": True,
        }
        self._metadata["last_joint_action_report"] = report
        self._metadata["joint_action_apply_count"] += 1
        if report["arm_targeted"]:
            self._metadata["arm_joint_action_apply_count"] += 1
        if report["gripper_targeted"]:
            self._metadata["gripper_joint_action_apply_count"] += 1
        if action.gripper_command == "close" and report["gripper_targeted"]:
            self._metadata["gripper_close_apply_count"] += 1
        if action.gripper_command == "open" and report["gripper_targeted"]:
            self._metadata["gripper_open_apply_count"] += 1

    def step(self, *, render: bool) -> None:
        del render
        self.step_calls += 1

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _base_metadata() -> dict:
        return {
            "execution_provenance_verified": False,
            "used_base_teleport": False,
            "used_direct_joint_state": False,
            "used_object_teleport": False,
            "used_kinematic_object_follow": False,
            "used_visual_replay": False,
            "last_joint_action_report": None,
            "joint_action_apply_count": 0,
            "arm_joint_action_apply_count": 0,
            "gripper_joint_action_apply_count": 0,
            "gripper_close_apply_count": 0,
            "gripper_open_apply_count": 0,
        }


if __name__ == "__main__":
    unittest.main()
