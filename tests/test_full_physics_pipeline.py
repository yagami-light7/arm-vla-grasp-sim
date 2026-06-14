"""Tests for the first-phase full-physics pipeline skeleton."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.pipeline.run_full_physics_pipeline import _build_parser, main
from source.diagnostics import (
    DryRunEpisodeVerifier,
    FullPhysicsVerifier,
    ManipulationApplySmokeVerifier,
    NavigationEpisodeVerifier,
)
from source.interfaces import EpisodeSpec, RobotAction, SimulationState, VerificationResult
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
    ManipulationSettings,
    PipelineState,
    StateLimits,
)
from source.pipeline.dry_run import create_dry_run_pipeline
from source.pipeline.integrated_apply_smoke import create_full_physics_pipeline
from source.pipeline.manipulation_apply_smoke import create_manipulation_apply_smoke_pipeline
from source.pipeline.manipulation_smoke import create_manipulation_smoke_pipeline
from source.pipeline.navigation_smoke import _build_dwa_config
from source.recording import JsonlEpisodeRecorder
from source.simulation import InMemorySimulationRuntime
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FullPhysicsPipelineTest(unittest.TestCase):
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

    def test_cli_help_is_chinese_and_modes_are_explicit(self) -> None:
        help_text = _build_parser().format_help()
        self.assertIn("任务 JSON 路径", help_text)
        self.assertIn("--dry-run", help_text)
        self.assertIn("--simulation-smoke", help_text)
        self.assertIn("--navigation-smoke", help_text)
        self.assertIn("--navigation-carry-smoke", help_text)
        self.assertIn("--manipulation-smoke", help_text)
        self.assertIn("--manipulation-apply-smoke", help_text)
        self.assertIn("--full-physics", help_text)
        self.assertIn("--integrated-apply-smoke", help_text)
        self.assertIn("--pick-plan-json", help_text)
        self.assertIn("--place-plan-json", help_text)
        self.assertIn("--viewport-camera-prim", help_text)
        self.assertIn("--no-lock-base-during-manipulation", help_text)
        self.assertIn("--no-lock-support-joints-during-manipulation", help_text)
        self.assertIn("--no-replan-pick-from-current-state", help_text)
        self.assertIn("--no-auto-start-curobo-server", help_text)
        self.assertIn("--no-headless", help_text)

    def test_manipulation_base_lock_defaults_on_and_can_be_disabled(self) -> None:
        default_args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--dry-run",
            ]
        )
        disabled_args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--dry-run",
                "--no-lock-base-during-manipulation",
                "--no-lock-support-joints-during-manipulation",
                "--no-auto-start-curobo-server",
            ]
        )

        self.assertTrue(default_args.lock_base_during_manipulation)
        self.assertTrue(default_args.lock_support_joints_during_manipulation)
        self.assertTrue(default_args.auto_start_curobo_server)
        self.assertFalse(disabled_args.lock_base_during_manipulation)
        self.assertFalse(disabled_args.lock_support_joints_during_manipulation)
        self.assertFalse(disabled_args.auto_start_curobo_server)
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
            10,
        )

    def test_navigation_defaults_restore_local_stable_brisk_fast_profile(self) -> None:
        config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs/test",
        )
        dwa_config = _build_dwa_config(config.navigation)

        self.assertTrue(config.navigation.brisk_nav)
        self.assertTrue(config.navigation.fast_dwa)
        self.assertEqual(config.navigation.dwa_replan_interval_steps, 1)
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

    def test_cli_requires_an_explicit_execution_mode(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--task-json", "tasks/nav_pick_place_apple_contact.json"])

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

    def test_integrated_apply_smoke_is_cancelled_before_isaac_startup(self) -> None:
        with self.assertRaisesRegex(SystemExit, "已取消，请改用 --full-physics"):
            main(
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
                        "--full-physics",
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
            self.assertTrue(
                pipeline.machine.manipulation_planner._config.side_grasp_retreat_to_pregrasp
            )
            self.assertTrue(pipeline.machine.config.manipulation.return_home_after_pick)
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.motion_time_scale,
                config.manipulation.arm_motion_time_scale,
            )
            self.assertIsInstance(pipeline.machine.verifier, FullPhysicsVerifier)

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
            tcp_pose=(0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
            object_pose=(0.45, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
            metadata={
                "body_velocity": (0.0, 0.0, 0.0),
                "gripper_close_apply_count": 2,
            },
        )

        self.assertTrue(verifier.verify_pick_reachable(pick_state, spec).success)
        pick_result = verifier.verify_pick_success(pick_state, spec)
        self.assertTrue(pick_result.success)
        self.assertEqual(pick_result.failure_reason, "")
        self.assertEqual(pick_result.metadata["validation_mode"], "object_tcp_contact_window")

    def test_full_physics_mode_reports_stable_success_with_lock_provenance(self) -> None:
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
            self.assertIn("place_base_settle_start", event_names)
            self.assertIn("place_base_settle_complete", event_names)
            self.assertNotIn("object_lift_success", event_names)
            self.assertNotIn("integrated_apply_smoke_place_apply_success", event_names)
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
            self.assertTrue(
                all(frame["action"]["arm_joint_positions"] == [0.0] * 6 for frame in carry_frames)
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
            self.assertEqual(len(pick_settle_frames), 10)
            self.assertTrue(
                all(
                    frame["action"]["metadata"].get("manipulation_base_lock")
                    and frame["action"]["metadata"].get("manipulation_support_joint_lock")
                    for frame in pick_settle_frames
                )
            )
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
            self.assertIn("return_home_after_pick", segment_names)
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

    def test_pick_return_home_aligns_place_plan_start_by_default(self) -> None:
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
                    frame["action"]["metadata"].get("segment_name") == "return_home_after_pick"
                    and frame["action"]["arm_joint_positions"] == [0.0] * 6
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
