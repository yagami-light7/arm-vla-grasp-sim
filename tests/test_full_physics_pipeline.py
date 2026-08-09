"""Tests for the first-phase full-physics pipeline skeleton."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline.run_full_physics_pipeline import (
    _apply_scan_manual_path_goal_override,
    _build_parser,
    _camera_sensor_runtime_kwargs,
    _locomotion_runtime_kwargs,
    _load_body_height_preflight_settings,
    _load_navigation_point_cloud_settings,
    _navigation_ros2_runtime_kwargs,
    _navigation_visual_runtime_kwargs,
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
from source.interfaces import (
    ArmPlan,
    EpisodeSpec,
    NavGoal,
    RobotAction,
    SimulationState,
    VerificationResult,
)
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
    RecordingSettings,
    StateLimits,
    VideoRecordingSettings,
)
from source.pipeline.dry_run import create_dry_run_pipeline
from source.pipeline.isaac_compat import patch_numpy_for_isaacsim
from source.pipeline.factory import create_full_physics_pipeline
from source.pipeline.full_physics_pipeline import _should_auto_switch_overview_camera
from source.pipeline.manipulation_apply_smoke import create_manipulation_apply_smoke_pipeline
from source.pipeline.manipulation_smoke import create_manipulation_smoke_pipeline
from source.pipeline.navigation_smoke import (
    _build_dwa_config,
    _scan_stair_freeze_config,
)
from source.pipeline.state_machine import (
    FullPhysicsStateMachine,
    _accept_successful_navigation_handoff_drift,
    _navigation_handoff_requires_zero_command_settle,
)
from source.recording import JsonlEpisodeRecorder
from source.recording.jsonl_recorder import _compact_simulation_metadata
from source.simulation import InMemorySimulationRuntime
from source.simulation.lighting import resolve_scene_light_mode
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _liangzhu_pct_navigation_settings() -> NavigationSettings:
    """Return a real in-repo planner fixture without relying on removed A* assets."""

    return NavigationSettings(
        global_planner="pct",
        pct_enabled=True,
        pct_server_script=PROJECT_ROOT / "scripts/navigation/pct_grid_server.py",
        pct_tomogram_path=(
            PROJECT_ROOT
            / "source/scene/liangzhu/pct/liangzhu_single_floor.pickle"
        ),
        pct_walkable_path=(
            PROJECT_ROOT
            / "source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy"
        ),
        pct_collision_ply_path=(
            PROJECT_ROOT / "source/scene/liangzhu/ply/liangzhu_collision.ply"
        ),
        pct_fallback_to_astar=False,
        pct_coord_mode="identity",
    )


_STRICT_STAIR_HOLD_XYZYAW = (1.25, -0.5, 0.82, 0.35)


class _StrictScanCompletionExecutor:
    """构造可审计的 SCAN 成功终态及楼梯冻结动作。"""

    def __init__(
        self,
        terminal_fields: dict[str, object] | None = None,
        *,
        invalidate_hold_on_call: int | None = None,
    ) -> None:
        self._terminal_fields = (
            {
                "success": True,
                "failed": False,
                "failure_reason": "",
            }
            if terminal_fields is None
            else dict(terminal_fields)
        )
        self._has_plan = False
        self._completed = False
        self._invalidate_hold_on_call = invalidate_hold_on_call
        self.compute_action_calls = 0

    def reset(self, plan: object) -> None:
        del plan
        self._has_plan = True
        self._completed = False
        self.compute_action_calls = 0

    def compute_action(self, state: SimulationState) -> RobotAction:
        del state
        if not self._has_plan:
            raise RuntimeError("SCAN 测试 executor 尚未接收路径。")
        self.compute_action_calls += 1
        self._completed = True
        metadata = {
            "navigation_base_pose_lock": True,
            "navigation_base_pose_lock_phase": "terminal_hold",
            "navigation_base_pose_lock_xyzyaw": _STRICT_STAIR_HOLD_XYZYAW,
            "navigation_support_joint_lock": True,
            "navigation_full_body_joint_lock": True,
            "navigation_scan_stair_freeze": True,
            "navigation_scan_stair_freeze_phase": "terminal_hold",
            "navigation_cmd_vel_inhibit": True,
            "navigation_cmd_vel_inhibit_reason": "scan_stair_terminal_hold",
        }
        if self.compute_action_calls == self._invalidate_hold_on_call:
            metadata.pop("navigation_full_body_joint_lock")
        return RobotAction(
            base_velocity=(0.2, -0.1, 0.3),
            source="strict_scan_terminal_hold_fixture",
            metadata=metadata,
        )

    def is_done(self, state: SimulationState) -> bool:
        del state
        return self._completed and self.status().get("done") is True

    def status(self) -> dict[str, object]:
        if not self._completed:
            return {
                "backend": "scan_ros2_goal_event",
                "phase": "tracking",
                "done": False,
                "success": False,
                "failed": False,
                "failure_reason": "",
            }
        return {
            "backend": "scan_ros2_goal_event",
            "phase": "completed",
            "done": True,
            "stair_freeze": {
                "phase": "terminal_hold",
                "finish_ready": True,
                "terminal_goal_bound": True,
                "terminal_supervisor_goal_acknowledged": True,
                "hold_xyzyaw": _STRICT_STAIR_HOLD_XYZYAW,
            },
            **self._terminal_fields,
        }


class _PickReachabilitySpyVerifier(DryRunEpisodeVerifier):
    """记录导航交接校验次数，必要时令意外调用立即暴露。"""

    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.raise_on_call = bool(raise_on_call)
        self.pick_reachable_calls = 0

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del state, episode_spec
        self.pick_reachable_calls += 1
        if self.raise_on_call:
            raise AssertionError("strict 楼梯完成后不应调用抓取可达性校验器")
        return VerificationResult(
            success=True,
            metadata={"verifier": "pick_reachability_spy"},
        )


class FullPhysicsPipelineTest(unittest.TestCase):
    def test_dynamic_obstacle_lifecycle_evidence_is_preserved_in_outputs(self) -> None:
        summary_source = inspect.getsource(FullPhysicsPipeline._build_summary)
        for key in (
            "dynamic_obstacle_configuration_report",
            "dynamic_obstacle_runtime_report",
            "dynamic_obstacle_lifecycle_report",
            "dynamic_obstacle_raw_cloud_last_report",
            "dynamic_obstacle_raw_cloud_lifecycle_report",
            "dynamic_obstacle_pose_write_count",
            "scan_controller_status_lifecycle_report",
            "grid_map_observation_diagnostics_last_report",
            "grid_map_observation_lifecycle_report",
            "bspline_diagnostics_last_report",
            "bspline_diagnostics_lifecycle_report",
            "active_sensing_lifecycle_report",
            "dynamic_navigation_evidence_report",
        ):
            self.assertIn(f'"{key}"', summary_source)

        lifecycle = {
            "schema": "dynamic_obstacle_lifecycle_v1",
            "all_configured_obstacles_moved": True,
        }
        grid_map_last = {
            "observation_sequence": 17,
            "accepted_endpoint_count": 23,
        }
        grid_map_lifecycle = {
            "schema": "grid_map_observation_lifecycle_v1",
            "observation_count": 17,
        }
        bspline_last = {
            "diagnostic_sequence": 9,
            "trajectory_id": 4,
        }
        bspline_lifecycle = {
            "schema": "bspline_diagnostics_lifecycle_v1",
            "diagnostic_count": 9,
        }
        dynamic_navigation_evidence = {
            "schema": "dynamic_navigation_evidence_v1",
            "verified": True,
        }
        active_sensing_lifecycle = {
            "schema": "active_sensing_lifecycle_v1",
            "attempt_count": 1,
            "attempts": [{"identity": {"traj_id": 7}}],
        }
        compact = _compact_simulation_metadata(
            {
                "dynamic_obstacle_runtime_report": {"enabled": True},
                "dynamic_obstacle_lifecycle_report": lifecycle,
                "dynamic_obstacle_raw_cloud_last_report": {
                    "total_obstacle_point_count": 5,
                },
                "dynamic_obstacle_raw_cloud_lifecycle_report": {
                    "schema": "dynamic_obstacle_raw_cloud_lifecycle_v1",
                },
                "scan_controller_status_lifecycle_report": {
                    "schema": "scan_controller_status_lifecycle_v1",
                },
                "grid_map_observation_diagnostics_last_report": grid_map_last,
                "grid_map_observation_lifecycle_report": grid_map_lifecycle,
                "bspline_diagnostics_last_report": bspline_last,
                "bspline_diagnostics_lifecycle_report": bspline_lifecycle,
                "active_sensing_lifecycle_report": active_sensing_lifecycle,
                "dynamic_navigation_evidence_report": (
                    dynamic_navigation_evidence
                ),
            }
        )
        self.assertEqual(
            compact["dynamic_obstacle_lifecycle_report"],
            lifecycle,
        )
        self.assertEqual(
            compact["dynamic_obstacle_raw_cloud_lifecycle_report"]["schema"],
            "dynamic_obstacle_raw_cloud_lifecycle_v1",
        )
        self.assertEqual(
            compact["scan_controller_status_lifecycle_report"]["schema"],
            "scan_controller_status_lifecycle_v1",
        )
        self.assertEqual(
            compact["grid_map_observation_diagnostics_last_report"],
            grid_map_last,
        )
        self.assertEqual(
            compact["grid_map_observation_lifecycle_report"],
            grid_map_lifecycle,
        )
        self.assertEqual(
            compact["grid_map_observation_lifecycle_report"]["schema"],
            "grid_map_observation_lifecycle_v1",
        )
        self.assertEqual(
            compact["bspline_diagnostics_last_report"],
            bspline_last,
        )
        self.assertEqual(
            compact["bspline_diagnostics_lifecycle_report"],
            bspline_lifecycle,
        )
        self.assertEqual(
            compact["bspline_diagnostics_lifecycle_report"]["schema"],
            "bspline_diagnostics_lifecycle_v1",
        )
        self.assertEqual(
            compact["active_sensing_lifecycle_report"],
            {
                "schema": "active_sensing_lifecycle_v1",
                "attempt_count": 1,
            },
        )
        self.assertEqual(
            compact["dynamic_navigation_evidence_report"],
            dynamic_navigation_evidence,
        )
        self.assertEqual(
            compact["dynamic_navigation_evidence_report"]["schema"],
            "dynamic_navigation_evidence_v1",
        )

    def test_navigation_failure_decision_requests_latched_policy_stop(self) -> None:
        machine = object.__new__(FullPhysicsStateMachine)
        machine.state = PipelineState.FAILED
        machine.failure_reason = "locomotion_stall"

        decision = machine._decision(PipelineState.EXEC_NAV_TO_PICK, [])

        self.assertTrue(decision.terminal)
        self.assertEqual(decision.action.base_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(
            decision.action.metadata,
            {
                "navigation_emergency_stop": True,
                "navigation_emergency_stop_reason": "locomotion_stall",
            },
        )

    def test_successful_navigation_handoff_accepts_only_bounded_frame_drift(self) -> None:
        base_metadata = {
            "goal_distance": 0.100017,
            "goal_z_error": 0.10,
            "yaw_error": 0.145,
            "linear_speed": 0.041,
            "angular_speed": 0.2072,
            "position_tolerance": 0.10,
            "goal_z_tolerance": 0.35,
            "yaw_tolerance": 0.15,
            "linear_velocity_tolerance": 0.06,
            "angular_velocity_tolerance": 0.20,
            "z_check_enabled": True,
            "yaw_alignment_required": True,
            "base_stability_required": True,
        }
        failed = VerificationResult(
            success=False,
            failure_reason="pick_target_unreachable",
            metadata=base_metadata,
        )

        accepted = _accept_successful_navigation_handoff_drift(
            failed,
            {
                "success": True,
                "failed": False,
                "distance_to_goal": 0.09998,
                "yaw_error": 0.146,
            },
            phase="pick",
        )

        self.assertTrue(accepted.success)
        self.assertEqual(
            accepted.metadata["navigation_verifier_override"],
            "successful_executor_one_frame_drift_margin",
        )
        self.assertTrue(
            all(accepted.metadata["navigation_handoff_margin_checks"].values())
        )

        # Regression from the full-yaw seed-7000 place handoff: the executor
        # had already succeeded, then the adjacent verifier frame observed a
        # small residual locomotion-policy pulse.
        residual_velocity = VerificationResult(
            success=False,
            failure_reason="place_target_unreachable",
            metadata={
                **base_metadata,
                "goal_distance": 0.089,
                "yaw_error": 0.1366,
                "linear_speed": 0.070280,
                "angular_speed": 0.23605,
            },
        )
        accepted_residual_velocity = _accept_successful_navigation_handoff_drift(
            residual_velocity,
            {"success": True, "failed": False},
            phase="place",
        )
        self.assertTrue(accepted_residual_velocity.success)

        excessive_residual_velocity = VerificationResult(
            success=False,
            failure_reason="place_target_unreachable",
            metadata={
                **base_metadata,
                "linear_speed": 0.076,
                "angular_speed": 0.241,
            },
        )
        rejected_residual_velocity = _accept_successful_navigation_handoff_drift(
            excessive_residual_velocity,
            {"success": True, "failed": False},
            phase="place",
        )
        self.assertFalse(rejected_residual_velocity.success)

        settle_only_residual = VerificationResult(
            success=False,
            failure_reason="place_target_unreachable",
            metadata={
                **base_metadata,
                "goal_distance": 0.0857,
                "yaw_error": 0.1382,
                "linear_speed": 0.0553,
                "angular_speed": 0.2676,
                "position_reached": True,
                "z_reached": True,
                "yaw_aligned": True,
                "base_stable": False,
            },
        )
        self.assertTrue(
            _navigation_handoff_requires_zero_command_settle(
                settle_only_residual,
                {"success": True, "failed": False},
            )
        )
        self.assertFalse(
            _navigation_handoff_requires_zero_command_settle(
                VerificationResult(
                    success=False,
                    failure_reason="place_target_unreachable",
                    metadata={
                        **settle_only_residual.metadata,
                        "position_reached": False,
                    },
                ),
                {"success": True, "failed": False},
            )
        )
        self.assertFalse(
            _navigation_handoff_requires_zero_command_settle(
                VerificationResult(
                    success=False,
                    failure_reason="place_target_unreachable",
                    metadata={
                        **settle_only_residual.metadata,
                        "angular_speed": 0.401,
                    },
                ),
                {"success": True, "failed": False},
            )
        )

        too_far = VerificationResult(
            success=False,
            failure_reason="pick_target_unreachable",
            metadata={**base_metadata, "goal_distance": 0.106},
        )
        rejected = _accept_successful_navigation_handoff_drift(
            too_far,
            {"success": True, "failed": False},
            phase="pick",
        )
        self.assertFalse(rejected.success)

        no_executor_success = _accept_successful_navigation_handoff_drift(
            failed,
            {"success": False, "failed": False},
            phase="pick",
        )
        self.assertFalse(no_executor_success.success)

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
            self.assertFalse(summary["lerobot_training_eligible"])
            self.assertFalse(summary["execution_provenance_verified"])
            self.assertEqual(summary["success_semantics"], "control_flow_only")
            self.assertEqual(summary["state_trace"], expected_trace)
            self.assertTrue(summary["place_verification_result"]["success"])
            self.assertGreater(summary["duration_steps"], len(expected_trace))
            self.assertEqual(pipeline.simulation.apply_calls, summary["duration_steps"])
            performance = summary["performance_report"]
            self.assertEqual(performance["schema_version"], "wall_time_profile_v1")
            self.assertEqual(performance["seed"], 7)
            self.assertEqual(
                performance["operations"]["pipeline.state_machine_tick"]["count"],
                summary["duration_steps"],
            )
            self.assertIn("frames.jsonl", performance["artifact_sizes_bytes"])

            for filename in (
                "task.json",
                "events.jsonl",
                "frames.jsonl",
                "lerobot_manifest.json",
                "pipeline_startup_status.json",
                "pipeline_startup_traceback.log",
                "summary.json",
            ):
                self.assertTrue((output_dir / filename).exists(), filename)

            startup_status = json.loads(
                (output_dir / "pipeline_startup_status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(startup_status["status"], "completed")
            self.assertEqual(startup_status["pid"], os.getpid())
            self.assertEqual(
                [phase["phase"] for phase in startup_status["phases"]],
                [
                    "run_episode_entered",
                    "task_saved",
                    "initial_simulation_read_started",
                    "initial_simulation_read_finished",
                    "initial_state_machine_tick_started",
                    "initial_state_machine_tick_finished",
                ],
            )
            self.assertEqual(
                (output_dir / "pipeline_startup_traceback.log").read_text(
                    encoding="utf-8"
                ),
                "",
            )

            manifest = json.loads((output_dir / "lerobot_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["lerobot_exported"])
            self.assertFalse(manifest["training_eligible"])
            self.assertFalse(manifest["episode_success_verified"])
            self.assertEqual(
                manifest["reason"],
                "execution_mode_is_not_full_physics",
            )
            self.assertEqual(manifest["control_action_dimension"], 11)
            self.assertEqual(
                manifest["vla_training_action_schema"],
                "base_xyyaw_tcp_base_rpy_gripper_v1",
            )
            self.assertEqual(manifest["vla_training_action_dimension"], 10)
            self.assertFalse(manifest["vla_training_action_available"])
            self.assertFalse(manifest["vla_training_eligible"])
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
        self.assertIn("--pct-coord-mode", help_text)
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
        self.assertTrue(pct_dwa_config.enforce_min_active_linear_velocity)
        self.assertTrue(pct_dwa_config.enforce_min_active_angular_velocity)
        self.assertAlmostEqual(pct_dwa_config.min_active_angular_velocity, 0.30)

    def test_cli_defaults_to_full_physics_mode(self) -> None:
        args = _parse_args(
            ["--task-json", "tasks/nav_pick_place_apple_contact.json"]
        )
        self.assertEqual(args.mode, "full_physics")
        self.assertFalse(args.show_planned_trajectories)

    def test_cli_defaults_to_liangzhu_scene_profile(self) -> None:
        args = _parse_args([])

        self.assertEqual(args.scene_profile, "liangzhu")
        self.assertEqual(args.runtime_preset, "scene_profile:liangzhu")
        self.assertEqual(
            args.task_json,
            "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json",
        )
        self.assertEqual(args.global_planner, "pct")
        self.assertTrue(args.enable_navigation_ros2_bridge)
        self.assertEqual(args.pct_coord_mode, "identity")
        self.assertTrue(args.pct_no_fallback)
        self.assertEqual(
            args.pct_server_script,
            "scripts/navigation/pct_grid_server.py",
        )
        self.assertEqual(
            args.pct_tomogram_path,
            "source/scene/liangzhu/pct/liangzhu_single_floor.pickle",
        )
        self.assertEqual(
            args.pct_walkable_path,
            "source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy",
        )
        self.assertEqual(
            args.pct_collision_ply_path,
            "source/scene/liangzhu/ply/liangzhu_collision.ply",
        )
        self.assertEqual(args.pct_cross_floor_gateway, [])
        self.assertEqual(args.pct_cross_floor_stair_exit, [])
        self.assertEqual(args.pct_cross_floor_stair_midpoint, [])
        self.assertEqual(args.policy_profile, "pct_multifloor")
        self.assertTrue(args.randomize_task)
        self.assertTrue(args.randomize_base_goal)
        self.assertEqual(args.navigation_visual_mode, "collision")
        self.assertEqual(args.scene_light_mode, "auto")
        self.assertEqual(args.overview_camera_mode, "fixed")
        self.assertEqual(args.overview_camera_prim_path, "/World/overview")
        self.assertFalse(args.pct_stair_float)
        self.assertTrue(args.record_video)

    def test_pct_multifloor_stable_preset_resolves_runtime_defaults(self) -> None:
        args = _parse_args(["--pct-multifloor"])

        self.assertEqual(args.scene_profile, "multi_floor")
        self.assertEqual(args.runtime_preset, "scene_profile:multi_floor")
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
        self.assertTrue(args.require_locomotion_checkpoint)
        self.assertEqual(args.navigation_visual_mode, "collision")
        self.assertFalse(args.randomize_task)
        self.assertFalse(args.randomize_base_goal)
        self.assertFalse(args.show_planned_trajectories)
        self.assertFalse(args.headless)
        self.assertFalse(args.keep_window_open)
        self.assertEqual(args.output_dir, "outputs/multi_floor")
        self.assertEqual(args.navigation_visual_mode, "collision")
        self.assertEqual(args.scene_light_mode, "auto")
        self.assertTrue(args.record_video)
        self.assertEqual(args.video_mode, "composite")
        self.assertEqual(
            args.overview_camera_schedule,
            "configs/recording/multifloor_overview_camera_schedule.json",
        )
        self.assertFalse(args.pct_stair_float)

    def test_full_visual_auto_uses_stage_lights_for_both_scene_profiles(self) -> None:
        for argv in (
            ["--scene-profile", "liangzhu", "--navigation-visual-mode", "full"],
            ["--pct-multifloor", "--navigation-visual-mode", "full"],
        ):
            with self.subTest(argv=argv):
                args = _parse_args(argv)
                visual_runtime = _navigation_visual_runtime_kwargs(
                    args.policy_profile,
                    args.navigation_visual_mode,
                )

                self.assertTrue(visual_runtime["enable_scene_visual"])
                self.assertEqual(
                    resolve_scene_light_mode(
                        args.scene_light_mode,
                        scene_visual_enabled=bool(
                            visual_runtime["enable_scene_visual"]
                        ),
                    ),
                    "stage",
                )

    def test_scene_light_auto_and_explicit_override_modes(self) -> None:
        self.assertEqual(
            resolve_scene_light_mode("auto", scene_visual_enabled=False),
            "camera",
        )
        self.assertEqual(
            resolve_scene_light_mode("camera", scene_visual_enabled=True),
            "camera",
        )
        self.assertEqual(
            resolve_scene_light_mode("stage", scene_visual_enabled=False),
            "stage",
        )

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
        args = _parse_args(
            ["--scene-profile", "multi_floor", "--stair-locomotion-smoke"]
        )

        self.assertEqual(args.mode, "stair_locomotion_smoke")
        self.assertEqual(args.global_planner, "pct")
        self.assertEqual(args.policy_profile, "pct_multifloor")
        self.assertFalse(args.pct_stair_float)
        self.assertTrue(args.show_planned_trajectories)
        self.assertFalse(args.headless)
        self.assertTrue(args.keep_window_open)
        self.assertTrue(args.record_video)
        self.assertTrue(args.record_dataset)
        self.assertEqual(
            args.dataset_camera_keys,
            ["front", "wrist", "overview"],
        )
        self.assertEqual(args.video_mode, "composite")
        self.assertEqual(
            args.output_dir,
            "outputs/multi_floor_stair_locomotion_smoke",
        )
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
                "--scene-profile",
                "multi_floor",
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

    def test_stair_fixed_command_probe_cli_preserves_ab_parameters(self) -> None:
        args = _parse_args(
            [
                "--scene-profile",
                "multi_floor",
                "--stair-locomotion-smoke",
                "--task-json",
                "tasks/nav_smoke_scan_multifloor_stair_two_step.json",
                "--stair-fixed-command-probe",
                "--stair-probe-vx",
                "0.30",
                "--stair-probe-duration",
                "3.20",
                "--headless",
                "--no-record-video",
                "--no-record-dataset",
            ]
        )

        self.assertTrue(args.stair_fixed_command_probe)
        self.assertEqual(args.stair_probe_vx, 0.30)
        self.assertEqual(args.stair_probe_duration, 3.20)
        self.assertFalse(args.enable_navigation_ros2_bridge)
        self.assertFalse(args.pct_stair_float)

    def test_stair_locomotion_smoke_manages_gui_camera3(self) -> None:
        stair = _navigation_smoke_viewport_runtime_kwargs(
            headless=False,
            stair_locomotion_smoke=True,
            overview_camera_mode="auto",
            overview_camera_prim_path="/World/Camera3",
        )
        regular_gui = _navigation_smoke_viewport_runtime_kwargs(
            headless=False,
            stair_locomotion_smoke=False,
            overview_camera_mode="auto",
            overview_camera_prim_path="/World/Camera0",
        )

        self.assertEqual(stair["viewport_camera_prim_path"], "/World/Camera3")
        self.assertTrue(stair["auto_manage_viewport_camera"])
        self.assertEqual(regular_gui["viewport_camera_prim_path"], "/World/Camera0")
        self.assertFalse(regular_gui["auto_manage_viewport_camera"])

    def test_stair_locomotion_video_schedule_auto_switches_in_gui(self) -> None:
        stair_config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
            output_dir=PROJECT_ROOT / "outputs/test",
            headless=False,
            stair_locomotion_smoke=True,
            video=VideoRecordingSettings(
                enabled=True,
                mode="composite",
                overview_camera_mode="auto",
            ),
        )
        regular_gui_config = replace(stair_config, stair_locomotion_smoke=False)

        self.assertTrue(_should_auto_switch_overview_camera(stair_config))
        self.assertTrue(_should_auto_switch_overview_camera(regular_gui_config))

    def test_composite_video_enables_three_runtime_cameras_without_dataset(self) -> None:
        config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json",
            output_dir=PROJECT_ROOT / "outputs/test",
            headless=True,
            navigation_smoke=True,
            recording=RecordingSettings(enabled=False),
            video=VideoRecordingSettings(
                enabled=True,
                mode="composite",
                overview_camera_mode="auto",
            ),
        )

        runtime_kwargs = _camera_sensor_runtime_kwargs(config)

        self.assertTrue(runtime_kwargs["enable_front_camera"])
        self.assertTrue(runtime_kwargs["enable_wrist_camera"])
        self.assertTrue(runtime_kwargs["enable_overview_camera"])
        self.assertEqual(runtime_kwargs["front_camera_width"], 640)
        self.assertEqual(runtime_kwargs["front_camera_height"], 480)
        self.assertEqual(runtime_kwargs["camera_render_interval_control_steps"], 1)

    def test_navigation_ros2_cli_builds_paired_runtime_config(self) -> None:
        disabled = _navigation_ros2_runtime_kwargs(_parse_args(["--dry-run"]))
        default_mainline = _navigation_ros2_runtime_kwargs(_parse_args([]))
        args = _parse_args(
            [
                "--enable-navigation-ros2-bridge",
                "--ros2-domain-id",
                "17",
                "--ros2-point-cloud-stride",
                "8",
                "--ros2-point-cloud-interval",
                "4",
                "--ros2-point-cloud-max-depth",
                "6.5",
                "--ros2-cmd-vel-topic",
                "/scan/cmd_vel",
                "--ros2-goal-reached-topic",
                "/scan/goal_reached",
                "--ros2-grid-map-diagnostics-topic",
                "/scan/grid_map_diagnostics",
                "--ros2-bspline-diagnostics-topic",
                "/scan/bspline_diagnostics",
                "--ros2-navigation-status-topic",
                "/scan/navigation_status",
                "--ros2-reference-path-topic",
                "/scan/initial_path",
                "--ros2-pct-goal-topic",
                "/scan/pct_goal",
                "--ros2-stair-execution-frozen-topic",
                "/scan/stair_execution_frozen",
                "--ros2-world-frame",
                "map",
                "--ros2-base-frame",
                "robot/base_link",
            ]
        )

        enabled = _navigation_ros2_runtime_kwargs(args)

        self.assertEqual(disabled, {})
        self.assertTrue(
            default_mainline["ros2_ogn_bridge_config"].enable_pct_goal_publisher
        )
        self.assertTrue(
            default_mainline[
                "ros2_ogn_bridge_config"
            ].enable_stair_execution_frozen_publisher
        )
        default_cloud = default_mainline["depth_point_cloud_config"]
        self.assertEqual(default_cloud.pixel_stride, 8)
        self.assertEqual(default_cloud.publish_interval_control_steps, 10)
        self.assertEqual(default_cloud.max_points, 12000)
        bridge = enabled["ros2_ogn_bridge_config"]
        cloud = enabled["depth_point_cloud_config"]
        command_gate = enabled["cmd_vel_to_policy_config"]
        self.assertEqual(bridge.domain_id, 17)
        self.assertEqual(bridge.odometry_topic, "/isaac/body_pose_raw")
        self.assertEqual(bridge.odom_frame_id, "map")
        self.assertEqual(bridge.point_cloud_frame_id, "map")
        self.assertEqual(bridge.base_frame_id, "robot/base_link")
        self.assertTrue(bridge.enable_command_subscription)
        self.assertTrue(bridge.enable_goal_reached_subscription)
        self.assertEqual(bridge.command_topic, "/scan/cmd_vel")
        self.assertEqual(bridge.goal_reached_topic, "/scan/goal_reached")
        self.assertTrue(bridge.enable_grid_map_diagnostics_subscription)
        self.assertEqual(
            bridge.grid_map_diagnostics_topic,
            "/scan/grid_map_diagnostics",
        )
        self.assertTrue(bridge.enable_bspline_diagnostics_subscription)
        self.assertEqual(
            bridge.bspline_diagnostics_topic,
            "/scan/bspline_diagnostics",
        )
        self.assertEqual(
            bridge.navigation_status_topic,
            "/scan/navigation_status",
        )
        self.assertEqual(bridge.reference_path_topic, "/scan/initial_path")
        self.assertTrue(bridge.enable_reference_path_subscription)
        self.assertEqual(bridge.pct_goal_topic, "/scan/pct_goal")
        self.assertTrue(bridge.enable_pct_goal_publisher)
        self.assertEqual(
            bridge.stair_execution_frozen_topic,
            "/scan/stair_execution_frozen",
        )
        self.assertTrue(bridge.enable_stair_execution_frozen_publisher)
        self.assertEqual(cloud.sensor_name, "head_camera")
        self.assertEqual(cloud.pixel_stride, 8)
        self.assertEqual(cloud.publish_interval_control_steps, 4)
        self.assertEqual(cloud.max_depth_m, 6.5)
        self.assertEqual(cloud.minimum_valid_points, 64)
        self.assertEqual(command_gate.max_vx, 0.65)
        self.assertEqual(command_gate.max_vy, 0.15)
        self.assertEqual(command_gate.max_wz, 0.60)
        self.assertEqual(command_gate.max_vx_rate, 1.20)
        self.assertEqual(command_gate.max_vy_rate, 0.40)
        self.assertEqual(command_gate.max_wz_rate, 1.50)

    def test_navigation_point_cloud_tuning_rejects_invalid_resource_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid_tuning.yaml"
            config_path.write_text(
                """
isaac_navigation_runtime:
  ros__parameters:
    point_cloud.pixel_stride: 6
    point_cloud.publish_interval_control_steps: 10
    point_cloud.min_depth_m: 0.15
    point_cloud.max_depth_m: 8.0
    point_cloud.max_points: 32
    point_cloud.minimum_valid_points: 64
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "minimum_valid_points 不能大于 max_points",
            ):
                _load_navigation_point_cloud_settings(config_path)

    def test_body_height_preflight_tuning_loads_strict_quick_and_full_windows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "preflight_tuning.yaml"
            config_path.write_text(
                """
navigation_contract:
  ros__parameters:
    body_height_preflight.quick.enabled: true
    body_height_preflight.quick.minimum_samples: 26
    body_height_preflight.quick.minimum_duration_s: 0.50
    body_height_preflight.quick.maximum_mad_m: 0.0025
    body_height_preflight.quick.maximum_p95_p05_m: 0.0075
    body_height_preflight.quick.maximum_configured_height_error_m: 0.020
    body_height_preflight.full.minimum_samples: 50
    body_height_preflight.full.minimum_duration_s: 1.00
    body_height_preflight.full.maximum_mad_m: 0.010
    body_height_preflight.full.maximum_p95_p05_m: 0.030
    body_height_preflight.full.maximum_configured_height_error_m: 0.080
    body_height_preflight.timeout_s: 5.00
""".lstrip(),
                encoding="utf-8",
            )

            settings = _load_body_height_preflight_settings(config_path)

            self.assertTrue(
                settings["body_height_calibration_quick_enabled"]
            )
            self.assertEqual(
                settings["body_height_calibration_quick_min_samples"],
                26,
            )
            self.assertEqual(
                settings["body_height_calibration_min_samples"],
                50,
            )
            self.assertEqual(
                settings["body_height_calibration_quick_min_duration_s"],
                0.5,
            )
            self.assertEqual(
                settings["body_height_calibration_min_duration_s"],
                1.0,
            )
            self.assertEqual(
                settings[
                    "body_height_calibration_quick_contract_tolerance_m"
                ],
                0.02,
            )

    def test_body_height_preflight_tuning_rejects_non_boolean_quick_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "invalid_preflight.yaml"
            config_path.write_text(
                """
navigation_contract:
  ros__parameters:
    body_height_preflight.quick.enabled: 1
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "必须是布尔值"):
                _load_body_height_preflight_settings(config_path)

    def test_navigation_ros2_environment_fails_before_isaac_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "navigation_environment"
            with patch(
                "scripts.pipeline.run_full_physics_pipeline."
                "validate_isaac_ros2_custom_message_environment",
                side_effect=RuntimeError("custom_message_environment_sentinel"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "custom_message_environment_sentinel",
                ):
                    main(
                        [
                            "--scene-profile",
                            "multi_floor",
                            "--stair-locomotion-smoke",
                            "--headless",
                            "--no-record-video",
                            "--no-record-dataset",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            status = json.loads(
                (output_dir / "startup_status.json").read_text(
                    encoding="utf-8"
                )
            )
            phases = [entry["phase"] for entry in status["phases"]]
            self.assertEqual(status["status"], "failed")
            self.assertEqual(
                status["exception"]["type"],
                "RuntimeError",
            )
            self.assertEqual(
                status["exception"]["message"],
                "custom_message_environment_sentinel",
            )
            self.assertIn("config_ready", phases)
            self.assertNotIn("isaac_app_starting", phases)
            self.assertNotIn("curobo_server_starting", phases)

    def test_scan_manual_path_goal_override_preserves_height_and_provenance(
        self,
    ) -> None:
        original_goal = NavGoal(
            x=-0.55,
            y=6.08,
            z=0.31,
            yaw=1.57,
            floor_id="liangzhu_F1",
            slice_id=1,
        )
        episode_spec = EpisodeSpec(
            task_id=1,
            episode_id=2,
            instruction="测试手工转弯路径。",
            scene_usd="scene.usda",
            nav_map="",
            start=NavGoal(x=-0.55, y=5.05, z=0.28, yaw=0.0),
            pick_goal=original_goal,
            place_goal=None,
            object_prim_path=None,
            object_initial_pose=None,
            place_target_pose=None,
            raw_task={"runtime_override": {"existing": True}},
        )

        result = _apply_scan_manual_path_goal_override(
            episode_spec,
            (0.05315879305802, 5.652347877290755, 1.5707963267948966),
        )

        self.assertAlmostEqual(result.pick_goal.x, 0.05315879305802)
        self.assertAlmostEqual(result.pick_goal.y, 5.652347877290755)
        self.assertAlmostEqual(result.pick_goal.yaw, 1.5707963267948966)
        self.assertEqual(result.pick_goal.z, original_goal.z)
        self.assertEqual(result.pick_goal.floor_id, original_goal.floor_id)
        self.assertEqual(result.pick_goal.slice_id, original_goal.slice_id)
        runtime_override = result.raw_task["runtime_override"]
        self.assertTrue(runtime_override["existing"])
        self.assertEqual(
            runtime_override["scan_manual_path_goal"]["path_source"],
            "external_ros2_nav_msgs_path",
        )
        self.assertEqual(
            runtime_override["scan_manual_path_goal"][
                "original_pick_goal_xyyaw"
            ],
            [original_goal.x, original_goal.y, original_goal.yaw],
        )
        self.assertIsNone(
            _apply_scan_manual_path_goal_override(episode_spec, None)
            .raw_task.get("scan_manual_path_goal")
        )
        with self.assertRaisesRegex(ValueError, "三个有限"):
            _apply_scan_manual_path_goal_override(
                episode_spec,
                (0.0, float("nan"), 0.0),
            )

    def test_scan_manual_path_goal_override_requires_ros2_navigation_smoke(
        self,
    ) -> None:
        args = _parse_args(
            [
                "--navigation-smoke",
                "--enable-navigation-ros2-bridge",
                "--scan-manual-path-goal-xyyaw",
                "0.05",
                "5.65",
                "1.57",
            ]
        )
        self.assertEqual(
            args.scan_manual_path_goal_xyyaw,
            [0.05, 5.65, 1.57],
        )
        with self.assertRaisesRegex(
            SystemExit,
            "只允许用于.*navigation-smoke",
        ):
            main(
                [
                    "--dry-run",
                    "--scan-manual-path-goal-xyyaw",
                    "0.05",
                    "5.65",
                    "1.57",
                ]
            )

    def test_navigation_ros2_runtime_disables_post_gate_standing_deadzone(
        self,
    ) -> None:
        config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=Path("/tmp/pct_scan_deadzone_test"),
            full_physics=True,
            pct_plan_preview=True,
            locomotion=LocomotionPolicySettings(
                policy_profile="pct_multifloor",
            ),
        )

        self.assertEqual(
            _locomotion_runtime_kwargs(config)["standing_command_threshold"],
            0.08,
        )
        self.assertEqual(
            _locomotion_runtime_kwargs(
                config,
                navigation_ros2_bridge_enabled=True,
            )["standing_command_threshold"],
            0.0,
        )

    def test_navigation_ros2_cli_rejects_mode_without_navigation_runtime(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "只支持会创建 IsaacLabNavigationRuntime",
        ):
            main(["--dry-run", "--enable-navigation-ros2-bridge"])

    def test_pct_scan_navigation_cli_rejects_legacy_planner_switches(self) -> None:
        cases = (
            (
                ["--navigation-smoke", "--no-enable-navigation-ros2-bridge"],
                "只允许 PCT→SCAN ROS 2 链",
            ),
            (
                ["--navigation-smoke", "--global-planner", "astar"],
                "只允许 PCT 全局规划器",
            ),
            (
                ["--navigation-smoke", "--pct-allow-fallback"],
                r"禁止 PCT→A\* fallback",
            ),
        )
        for argv, pattern in cases:
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(SystemExit, pattern):
                    main(argv)

    def test_navigation_ros2_cli_rejects_unsafe_multi_episode_lifecycle(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "epoch/reset/ack",
        ):
            main(
                [
                    "--navigation-smoke",
                    "--enable-navigation-ros2-bridge",
                    "--num-episodes",
                    "2",
                ]
            )

    def test_navigation_ros2_cli_requires_collision_visual_mode(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "navigation-visual-mode collision",
        ):
            main(
                [
                    "--navigation-smoke",
                    "--enable-navigation-ros2-bridge",
                    "--navigation-visual-mode",
                    "full",
                ]
            )

    def test_multifloor_gui_defaults_to_auto_overview_composite(self) -> None:
        args = _parse_args(
            [
                "--scene-profile",
                "multi_floor",
                "--seed",
                "0",
                "--output-dir",
                "/tmp/multi_floor_gui",
                "--no-headless",
                "--keep-window-open",
            ]
        )

        self.assertFalse(args.headless)
        self.assertTrue(args.keep_window_open)
        self.assertEqual(args.overview_camera_mode, "auto")
        self.assertEqual(args.video_mode, "composite")

    def test_stair_locomotion_smoke_rejects_explicit_float(self) -> None:
        with self.assertRaisesRegex(SystemExit, "固定禁用 Float"):
            main(
                [
                    "--scene-profile",
                    "multi_floor",
                    "--stair-locomotion-smoke",
                    "--pct-stair-float",
                ]
            )

    def test_stair_locomotion_requires_profile_capability(self) -> None:
        with self.assertRaisesRegex(SystemExit, "stair_locomotion_smoke 能力"):
            _parse_args(["--stair-locomotion-smoke"])

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
        args = _parse_args(
            ["--task-json", "tasks/nav_pick_place_apple_contact.json"]
        )

        self.assertEqual(
            args.pct_global_vertical_obstacle_min_slices,
            NavigationSettings().pct_global_vertical_obstacle_min_slices,
        )
        self.assertEqual(args.pct_coord_mode, "identity")
        self.assertEqual(
            args.pct_cross_floor_vertical_obstacle_min_slices,
            NavigationSettings().pct_cross_floor_vertical_obstacle_min_slices,
        )
        self.assertEqual(args.pct_cross_floor_gateway, [])
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
        self.assertFalse(args.pct_stair_float)
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
        self.assertEqual(
            args.navigation_body_height_m,
            NavigationSettings().navigation_body_height_m,
        )

    def test_cli_and_host_freeze_share_unique_navigation_body_height(self) -> None:
        args = _build_parser().parse_args(
            ["--navigation-body-height-m", "0.34"]
        )
        navigation = NavigationSettings(
            navigation_body_height_m=args.navigation_body_height_m
        )

        self.assertEqual(args.navigation_body_height_m, 0.34)
        self.assertEqual(
            _scan_stair_freeze_config(navigation).body_height_m,
            0.34,
        )

    def test_scan_stair_freeze_runtime_receives_max_control_dt(self) -> None:
        navigation = NavigationSettings(
            control_dt=0.02,
            scan_stair_freeze_max_control_dt_s=0.07,
        )

        runtime_config = _scan_stair_freeze_config(navigation)

        self.assertEqual(runtime_config.default_control_dt_s, 0.02)
        self.assertEqual(runtime_config.max_control_dt_s, 0.07)

    def test_scan_stair_freeze_max_control_dt_is_validated(self) -> None:
        common = {
            "task_json": (
                PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            ),
            "output_dir": PROJECT_ROOT / "outputs/test",
        }
        for value in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "scan_stair_freeze_max_control_dt_s",
                ):
                    FullPhysicsConfig(
                        **common,
                        navigation=NavigationSettings(
                            scan_stair_freeze_max_control_dt_s=value
                        ),
                    )
        with self.assertRaisesRegex(ValueError, "smaller than control_dt"):
            FullPhysicsConfig(
                **common,
                navigation=NavigationSettings(
                    control_dt=0.02,
                    scan_stair_freeze_max_control_dt_s=0.01,
                ),
            )

    def test_legacy_scan_freeze_body_height_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "scan_stair_freeze_body_height_m"):
            NavigationSettings(scan_stair_freeze_body_height_m=0.34)

    def test_navigation_body_height_must_be_finite_and_positive(self) -> None:
        for value in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "navigation_body_height_m must be finite and positive",
                ):
                    FullPhysicsConfig(
                        task_json=(
                            PROJECT_ROOT
                            / "tasks/nav_pick_place_apple_contact.json"
                        ),
                        output_dir=PROJECT_ROOT / "outputs/test",
                        navigation=NavigationSettings(
                            navigation_body_height_m=value
                        ),
                    )
        with self.assertRaisesRegex(ValueError, "measured body height maximum"):
            FullPhysicsConfig(
                task_json=(
                    PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
                ),
                output_dir=PROJECT_ROOT / "outputs/test",
                navigation=NavigationSettings(navigation_body_height_m=0.61),
            )

    def test_cli_accepts_liangzhu_identity_pct_frame(self) -> None:
        """良渚 PLY 与 Isaac 同坐标时必须能显式关闭旧场景的 X/Y 取反。"""

        args = _build_parser().parse_args(
            [
                "--task-json",
                "tasks/nav_pick_place_apple_contact.json",
                "--global-planner",
                "pct",
                "--pct-coord-mode",
                "identity",
            ]
        )

        self.assertEqual(args.global_planner, "pct")
        self.assertEqual(args.pct_coord_mode, "identity")

    def test_cli_accepts_complete_pct_coordinate_transform(self) -> None:
        args = _build_parser().parse_args(
            [
                "--pct-offset-x",
                "1.1",
                "--pct-offset-y",
                "-2.2",
                "--pct-offset-z",
                "3.3",
                "--pct-scale-x",
                "0.7",
                "--pct-scale-y",
                "-1.2",
                "--pct-scale-z",
                "1.4",
                "--pct-rotation-x-rad",
                "0.1",
                "--pct-rotation-y-rad",
                "-0.2",
                "--pct-rotation-z-rad",
                "0.3",
            ]
        )

        self.assertEqual(args.pct_offset_x, 1.1)
        self.assertEqual(args.pct_offset_y, -2.2)
        self.assertEqual(args.pct_offset_z, 3.3)
        self.assertEqual(args.pct_scale_x, 0.7)
        self.assertEqual(args.pct_scale_y, -1.2)
        self.assertEqual(args.pct_scale_z, 1.4)
        self.assertEqual(args.pct_rotation_x_rad, 0.1)
        self.assertEqual(args.pct_rotation_y_rad, -0.2)
        self.assertEqual(args.pct_rotation_z_rad, 0.3)

    def test_pipeline_config_rejects_invalid_pct_coordinate_transform(self) -> None:
        invalid_cases = (
            ({"pct_scale_z": 0.0}, "scales must be non-zero"),
            ({"pct_rotation_x_rad": float("nan")}, "must be finite"),
            ({"pct_coord_mode": "unsupported"}, "coordinate mode"),
        )
        for overrides, pattern in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, pattern):
                    FullPhysicsConfig(
                        task_json=(
                            PROJECT_ROOT
                            / "tasks/nav_pick_place_apple_contact.json"
                        ),
                        output_dir=PROJECT_ROOT / "outputs/test",
                        navigation=NavigationSettings(**overrides),
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
                diagnostic_frame_stride=7,
                navigation=_liangzhu_pct_navigation_settings(),
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
            self.assertEqual(pipeline.recorder._diagnostic_frame_stride, 7)
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
                navigation=NavigationSettings(
                    global_planner="pct",
                    pct_enabled=True,
                    pct_server_script=(
                        PROJECT_ROOT / "scripts/navigation/pct_grid_server.py"
                    ),
                    pct_tomogram_path=(
                        PROJECT_ROOT
                        / "source/scene/liangzhu/pct/liangzhu_single_floor.pickle"
                    ),
                    pct_walkable_path=(
                        PROJECT_ROOT
                        / "source/scene/liangzhu/pct/liangzhu_single_floor_walkable.npy"
                    ),
                    pct_collision_ply_path=(
                        PROJECT_ROOT
                        / "source/scene/liangzhu/ply/liangzhu_collision.ply"
                    ),
                    pct_fallback_to_astar=False,
                    pct_coord_mode="identity",
                ),
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
                1.50,
            )
            self.assertAlmostEqual(
                pipeline.machine.arm_executor.config.post_motion_joint_error_tolerance,
                0.070,
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
                navigation=_liangzhu_pct_navigation_settings(),
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
            self.assertTrue(
                reset.action.metadata["manipulation_support_joint_lock"]
            )
            self.assertEqual(
                reset.action.metadata["manipulation_support_joint_lock_phase"],
                "episode_initialization_settle",
            )
            simulation.apply(reset.action)
            simulation.step(render=False)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertIn("object_settle_started", {event.name for event in reset.events})

            settling = pipeline.machine.tick(simulation.read())
            self.assertTrue(
                settling.action.metadata["manipulation_support_joint_lock"]
            )
            self.assertEqual(
                settling.action.metadata["manipulation_support_joint_lock_phase"],
                "episode_initialization_settle",
            )
            simulation.apply(settling.action)
            simulation.step(render=False)
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)

            stabilized = pipeline.machine.tick(simulation.read())
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertTrue(
                stabilized.action.metadata["body_height_calibration_active"]
            )
            self.assertIn(
                "body_height_preflight_started",
                {event.name for event in stabilized.events},
            )
            self.assertIn(
                "object_initial_pose_stabilized",
                {event.name for event in stabilized.events},
            )
            self.assertTrue(
                simulation.read().metadata["object_settle_final_report"]["applied"]
            )

    def test_full_physics_rejects_historical_seed7_toppled_cola_during_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = (
                PROJECT_ROOT
                / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
            )
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                navigation=_liangzhu_pct_navigation_settings(),
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
                episode_seed=7,
                episode_dir=root / "episode",
                simulation=simulation,  # type: ignore[arg-type]
            )

            build = pipeline.machine.tick(simulation.read())
            simulation.apply(build.action)
            reset = pipeline.machine.tick(simulation.read())
            simulation.apply(reset.action)
            simulation.step(render=False)

            state = simulation.read()
            toppled = replace(
                state,
                object_pose=(
                    -0.513323962688446,
                    6.43435525894165,
                    0.167711079120636,
                    0.43538790941238403,
                    0.4164320230484009,
                    0.22225421667099,
                    -0.7665668725967407,
                ),
                object_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                metadata={
                    **state.metadata,
                    "object_pose_setup_report": {
                        "authored_world_quaternion_wxyz": [
                            0.6917987507794513,
                            0.6917987507794512,
                            -0.14633690040448005,
                            -0.1463369004044801,
                        ]
                    },
                },
            )
            rejected = pipeline.machine.tick(toppled)

            self.assertEqual(pipeline.machine.state, PipelineState.FAILED)
            self.assertEqual(
                pipeline.machine.failure_reason,
                "object_initialization_pose_invalid",
            )
            report = next(
                event.metadata
                for event in rejected.events
                if event.name == "episode_failed"
            )
            self.assertEqual(
                report["failure_reason"],
                "object_initialization_pose_invalid",
            )

    def test_full_physics_waits_for_base_stability_before_pct_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                navigation=_liangzhu_pct_navigation_settings(),
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
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertTrue(
                stabilized.action.metadata["body_height_calibration_active"]
            )
            self.assertIn(
                "body_height_preflight_started",
                {event.name for event in stabilized.events},
            )
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

    def test_full_physics_revalidates_base_after_initialization_lock_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                full_physics=True,
                navigation=_liangzhu_pct_navigation_settings(),
                manipulation=ManipulationSettings(
                    settle_object_before_navigation=True,
                    settle_base_before_navigation=True,
                    initialization_base_lock_steps=2,
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
            self.assertTrue(reset.action.metadata["manipulation_base_lock"])
            simulation.apply(reset.action)
            simulation.step(render=False)

            for expected_elapsed in (1, 2):
                locked = pipeline.machine.tick(simulation.read())
                report = locked.action.metadata["object_settle_report"]
                self.assertEqual(report["elapsed_steps"], expected_elapsed)
                self.assertTrue(report["initialization_base_lock_active"])
                self.assertEqual(report["stable_steps"], 0)
                self.assertTrue(locked.action.metadata["manipulation_base_lock"])
                simulation.apply(locked.action)
                simulation.step(render=False)

            released = pipeline.machine.tick(simulation.read())
            release_report = released.action.metadata["object_settle_report"]
            self.assertFalse(release_report["initialization_base_lock_active"])
            self.assertTrue(release_report["initialization_base_lock_released"])
            self.assertEqual(release_report["stable_steps"], 1)
            self.assertFalse(released.action.metadata["manipulation_base_lock"])
            self.assertFalse(
                released.action.metadata["manipulation_support_joint_lock"]
            )
            self.assertFalse(
                release_report["initialization_support_joint_lock_active"]
            )
            simulation.apply(released.action)
            simulation.step(render=False)

            stabilized = pipeline.machine.tick(simulation.read())
            self.assertEqual(pipeline.machine.state, PipelineState.RESET_EPISODE)
            self.assertTrue(
                stabilized.action.metadata["body_height_calibration_active"]
            )
            self.assertIn(
                "body_height_preflight_started",
                {event.name for event in stabilized.events},
            )
            self.assertIn(
                "object_initial_pose_stabilized",
                {event.name for event in stabilized.events},
            )

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

    def test_place_verifier_uses_task_tolerance_and_safe_region(self) -> None:
        """小型垫子任务不能继续使用宽松的全局默认容差。"""

        base_spec = JsonTaskProvider().load(
            PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"
        )
        target_pose = (1.0, 2.0, 0.10, 0.0, 0.0, 0.0)
        raw_task = {
            **base_spec.raw_task,
            "place": {
                **dict(base_spec.raw_task.get("place") or {}),
                "place_xy_tolerance": 0.025,
                "place_z_tolerance": 0.02,
                "placement_region": {
                    "frame": "world",
                    "x_min": 0.96,
                    "x_max": 1.04,
                    "y_min": 1.96,
                    "y_max": 2.04,
                },
            },
        }
        spec = replace(
            base_spec,
            place_target_pose=target_pose,
            raw_task=raw_task,
        )
        verifier = FullPhysicsVerifier(NavigationEpisodeVerifier())
        base_state = SimulationState(
            step_index=1,
            timestamp=0.0,
            robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
            object_pose=(1.0, 2.0, 0.10, 1.0, 0.0, 0.0, 0.0),
            object_velocity=(0.0,) * 6,
            metadata={
                "gripper_open_apply_count": 1,
                "place_open_apply_count_delta": 1,
                "place_release_observed": True,
                "place_release_object_pose": (1.0, 2.0, 0.10, 1.0, 0.0, 0.0, 0.0),
                "place_expected_release_object_center": (1.0, 2.0, 0.10),
                "place_release_velocity_sample_count": 1,
                "place_peak_object_linear_speed_mps": 0.0,
                "place_peak_object_horizontal_speed_mps": 0.0,
                "place_peak_object_angular_speed_rps": 0.0,
                "place_max_horizontal_displacement_m": 0.0,
            },
        )

        accepted = verifier.verify_place_success(base_state, spec)
        self.assertTrue(accepted.success)
        self.assertEqual(
            accepted.metadata["validation_mode"],
            "contact_release_pose_dynamics_region_and_final_stability",
        )
        self.assertEqual(accepted.metadata["place_xy_tolerance_m"], 0.025)
        self.assertTrue(
            accepted.metadata["placement_region_contains_object_center"]
        )

        runtime_target_state = replace(
            base_state,
            object_pose=(1.0, 2.0, 0.07, 1.0, 0.0, 0.0, 0.0),
            metadata={
                **base_state.metadata,
                "place_release_object_pose": (1.0, 2.0, 0.07, 1.0, 0.0, 0.0, 0.0),
                "place_expected_release_object_center": (1.0, 2.0, 0.07),
                "last_current_state_curobo_place_export": {
                    "mesh_truth_place_target_report": {
                        "verified": True,
                        "xyz_source": "runtime_mesh_truth",
                        "derived_place_pose_world": {
                            "x": 1.0,
                            "y": 2.0,
                            "z": 0.07,
                        },
                    }
                },
            },
        )
        runtime_target = verifier.verify_place_success(runtime_target_state, spec)
        self.assertTrue(runtime_target.success)
        self.assertEqual(
            runtime_target.metadata["place_target_pose_source"],
            "runtime_mesh_truth",
        )
        self.assertEqual(runtime_target.metadata["place_target_pose"][:3], (1.0, 2.0, 0.07))
        self.assertEqual(runtime_target.metadata["configured_place_target_pose"], target_pose)

        outside_task_tolerance = verifier.verify_place_success(
            replace(base_state, object_pose=(1.05, 2.0, 0.10, 1.0, 0.0, 0.0, 0.0)),
            spec,
        )
        self.assertFalse(outside_task_tolerance.success)

        region_limited_spec = replace(
            spec,
            raw_task={
                **raw_task,
                "place": {
                    **raw_task["place"],
                    "place_xy_tolerance": 0.10,
                    "placement_region": {
                        "frame": "world",
                        "x_min": 0.98,
                        "x_max": 1.02,
                        "y_min": 1.98,
                        "y_max": 2.02,
                    },
                },
            },
        )
        outside_region = verifier.verify_place_success(
            replace(base_state, object_pose=(1.05, 2.0, 0.10, 1.0, 0.0, 0.0, 0.0)),
            region_limited_spec,
        )
        self.assertFalse(outside_region.success)
        self.assertFalse(
            outside_region.metadata["placement_region_contains_object_center"]
        )

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
                "place_peak_object_upward_speed_mps": 0.01,
                "place_peak_object_downward_speed_mps": 0.04,
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
                    "place_peak_object_upward_speed_mps": 0.62,
                    "place_peak_object_downward_speed_mps": 0.10,
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

        downward_settle_state = replace(
            stable_state,
            metadata={
                **stable_state.metadata,
                "place_peak_object_linear_speed_mps": 0.409,
                "place_peak_object_horizontal_speed_mps": 0.066,
                "place_peak_object_upward_speed_mps": 0.025,
                "place_peak_object_downward_speed_mps": 0.409,
                "place_peak_object_angular_speed_rps": 2.50,
                "place_max_horizontal_displacement_m": 0.0021,
            },
        )
        downward_settle_spec = replace(
            spec,
            raw_task={
                **spec.raw_task,
                "place": {
                    **dict(spec.raw_task.get("place") or {}),
                    "place_release_peak_downward_speed_tolerance_mps": 0.55,
                },
            },
        )
        strict_downward = verifier.verify_place_success(
            downward_settle_state,
            spec,
        )
        accepted_downward = verifier.verify_place_success(
            downward_settle_state,
            downward_settle_spec,
        )
        upward_ejection = verifier.verify_place_success(
            replace(
                downward_settle_state,
                metadata={
                    **downward_settle_state.metadata,
                    "place_peak_object_upward_speed_mps": 0.409,
                    "place_peak_object_downward_speed_mps": 0.025,
                },
            ),
            downward_settle_spec,
        )

        self.assertFalse(strict_downward.success)
        self.assertEqual(strict_downward.failure_reason, "place_release_ejected")
        self.assertTrue(accepted_downward.success)
        self.assertEqual(
            accepted_downward.metadata[
                "place_release_peak_downward_speed_tolerance_mps"
            ],
            0.55,
        )
        self.assertEqual(
            accepted_downward.metadata["place_directional_speed_source"],
            "signed_vertical_velocity_peaks",
        )
        self.assertFalse(upward_ejection.success)
        self.assertEqual(upward_ejection.failure_reason, "place_release_ejected")

        invalid_directional_threshold_spec = replace(
            spec,
            raw_task={
                **spec.raw_task,
                "place": {
                    **dict(spec.raw_task.get("place") or {}),
                    "place_release_peak_downward_speed_tolerance_mps": 0.0,
                },
            },
        )
        invalid_directional_threshold = verifier.verify_place_success(
            stable_state,
            invalid_directional_threshold_spec,
        )
        self.assertFalse(invalid_directional_threshold.success)
        self.assertEqual(
            invalid_directional_threshold.failure_reason,
            "place_validation_config_invalid",
        )

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

    def test_liangzhu_can_accepts_supported_release_and_contact_jitter(self) -> None:
        spec = JsonTaskProvider().load(
            PROJECT_ROOT
            / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
        )
        verifier = FullPhysicsVerifier(NavigationEpisodeVerifier())
        target_xyz = tuple(float(value) for value in spec.place_target_pose[:3])
        release_xyz = (
            target_xyz[0] + 0.001,
            target_xyz[1] - 0.009,
            target_xyz[2] - 0.007,
        )
        state = SimulationState(
            step_index=2136,
            timestamp=42.72,
            robot_root_pose=(0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
            object_pose=(*release_xyz, 1.0, 0.0, 0.0, 0.0),
            object_velocity=(0.054, 0.0, 0.0, 0.994, 0.0, 0.0),
            metadata={
                "gripper_open_apply_count": 220,
                "place_open_apply_count_delta": 170,
                "place_release_observed": True,
                "place_release_object_pose": (
                    *release_xyz,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                "place_expected_release_object_center": (
                    target_xyz[0],
                    target_xyz[1],
                    target_xyz[2] + 0.01,
                ),
                "place_release_velocity_sample_count": 299,
                "place_peak_object_linear_speed_mps": 0.0844,
                "place_peak_object_horizontal_speed_mps": 0.0839,
                "place_peak_object_upward_speed_mps": 0.0201,
                "place_peak_object_downward_speed_mps": 0.0296,
                "place_peak_object_angular_speed_rps": 1.676,
                "place_max_horizontal_displacement_m": 0.0027,
            },
        )

        result = verifier.verify_place_success(state, spec)

        self.assertTrue(result.success)
        self.assertEqual(
            result.metadata["place_release_z_reference"],
            "final_supported_target",
        )
        self.assertAlmostEqual(
            result.metadata["place_linear_velocity_tolerance_mps"],
            0.1,
        )
        self.assertAlmostEqual(
            result.metadata["place_angular_velocity_tolerance_rps"],
            2.0,
        )

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

    def test_navigation_to_pick_holds_pct_checkpoint_stow_pose(self) -> None:
        machine = object.__new__(FullPhysicsStateMachine)
        machine.state = PipelineState.EXEC_NAV_TO_PICK
        machine._navigation_arm_stow_target = {
            "arm_joint_names": tuple(
                f"arm_joint{index}" for index in range(1, 7)
            ),
            "arm_joint_positions": (0.0, 0.3, 0.5, 0.0, 0.0, 0.0),
            "source": "pct_multifloor_checkpoint_default",
            "fixed_during_navigation": True,
        }

        held = machine._with_navigation_arm_stow_hold(
            RobotAction(
                base_velocity=(0.2, 0.0, 0.1),
                source="scan_ros2_navigation",
            ),
            PipelineState.EXEC_NAV_TO_PICK,
        )

        self.assertEqual(
            held.arm_joint_positions,
            (0.0, 0.3, 0.5, 0.0, 0.0, 0.0),
        )
        self.assertTrue(held.metadata["navigation_arm_stow_hold"])
        self.assertEqual(
            held.metadata["navigation_arm_stow_phase"],
            PipelineState.EXEC_NAV_TO_PICK.value,
        )
        self.assertTrue(held.metadata["arm_velocity_hold"])
        self.assertEqual(held.base_velocity, (0.2, 0.0, 0.1))

    def test_place_carry_handoff_uses_actual_scan_stair_freeze_provenance(self) -> None:
        machine = object.__new__(FullPhysicsStateMachine)
        machine.config = FullPhysicsConfig(
            task_json=PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json",
            output_dir=PROJECT_ROOT / "outputs",
            navigation=NavigationSettings(
                pct_stair_float_enabled=False,
                scan_stair_freeze_enabled=True,
            ),
        )
        state = SimulationState(
            step_index=9,
            timestamp=0.45,
            robot_root_pose=(0.24, -0.05, 3.28, 0.9238795, 0.0, 0.0, -0.3826834),
            robot_root_velocity=(0.0,) * 6,
            metadata={"used_navigation_base_lock": True},
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

        no_stair_action = machine._place_carry_handoff_action(
            replace(state, metadata={"used_navigation_base_lock": False})
        )
        self.assertEqual(no_stair_action.source, "verify_place_reachable")
        self.assertNotIn("navigation_base_pose_lock", no_stair_action.metadata)

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

    def test_navigation_carry_terminal_uses_latest_exec_arm_sample(self) -> None:
        """导航前收拢峰值不得否决已经稳定到点的 carry 姿态。"""

        class _PreNavigationArmPeakRuntime(InMemorySimulationRuntime):
            def read(self) -> SimulationState:
                state = super().read()
                latest = dict(
                    state.metadata.get("last_arm_tracking_report") or {}
                )
                if latest.get("available") is not True:
                    return state
                aggregate = dict(state.metadata.get("arm_tracking_report") or {})
                aggregate.update(
                    {
                        "max_abs_error": 0.3588,
                        "peak_report": {
                            "pipeline_state": PipelineState.PLAN_NAV_TO_PLACE.value,
                            "max_abs_error": 0.3588,
                        },
                        "latest_report": latest,
                    }
                )
                return replace(
                    state,
                    metadata={
                        **state.metadata,
                        "arm_tracking_report": aggregate,
                    },
                )

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
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=11,
                simulation=_PreNavigationArmPeakRuntime(),
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
            self.assertTrue(summary["carry_control_success"])
            success_event = next(
                json.loads(line)
                for line in (root / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(line)["name"] == "navigation_carry_smoke_success"
            )
            self.assertEqual(
                success_event["metadata"]["tracking_scope"],
                "latest_carry_sample",
            )
            self.assertLess(
                success_event["metadata"]["arm_tracking_report"]["max_abs_error"],
                config.manipulation.carry_home_tracking_tolerance,
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
            nav_executor = _StrictScanCompletionExecutor()
            verifier = _PickReachabilitySpyVerifier(raise_on_call=True)
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=5,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=nav_executor,
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=verifier,
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
                    PipelineState.CLEANUP_EPISODE.value,
                    PipelineState.DONE.value,
                ],
            )
            self.assertEqual(verifier.pick_reachable_calls, 0)

            events = [
                json.loads(line)
                for line in (root / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            event_names = {event["name"] for event in events}
            self.assertIn("nav_to_pick_success", event_names)
            self.assertIn("stair_locomotion_smoke_success", event_names)

            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            terminal_hold_frames = [
                frame
                for frame in frames
                if frame["action"]["source"]
                in {"stair_navigation_complete", "cleanup_episode"}
            ]
            self.assertEqual(
                [frame["action"]["source"] for frame in terminal_hold_frames],
                ["stair_navigation_complete", "cleanup_episode"],
            )
            for frame in terminal_hold_frames:
                action = frame["action"]
                metadata = action["metadata"]
                self.assertEqual(action["base_velocity"], [0.0, 0.0, 0.0])
                self.assertTrue(metadata["navigation_base_pose_lock"])
                self.assertEqual(
                    metadata["navigation_base_pose_lock_phase"],
                    "terminal_hold",
                )
                self.assertEqual(
                    metadata["navigation_base_pose_lock_xyzyaw"],
                    list(_STRICT_STAIR_HOLD_XYZYAW),
                )
                self.assertTrue(metadata["navigation_support_joint_lock"])
                self.assertTrue(metadata["navigation_full_body_joint_lock"])
                self.assertTrue(metadata["navigation_scan_stair_freeze"])
                self.assertEqual(
                    metadata["navigation_scan_stair_freeze_phase"],
                    "terminal_hold",
                )
                self.assertTrue(metadata["navigation_cmd_vel_inhibit"])
                self.assertEqual(
                    metadata["navigation_cmd_vel_inhibit_reason"],
                    "scan_stair_terminal_hold",
                )
                self.assertTrue(metadata["stair_navigation_strict_completion"])
                self.assertTrue(
                    metadata["stair_navigation_terminal_hold_preserved"]
                )

            final_state = pipeline.simulation.read()
            workaround_state = replace(
                final_state,
                metadata={
                    **final_state.metadata,
                    "execution_provenance_verified": True,
                    "used_base_teleport": True,
                    "used_direct_joint_state": True,
                    "used_navigation_base_lock": True,
                    "used_navigation_support_joint_lock": True,
                    "used_navigation_joint_pose_lock": True,
                    "navigation_ros2_bridge_report": {"publish_count": 8},
                    "navigation_stair_execution_frozen_last_publish_report": {
                        "sequence": 9,
                        "value": True,
                    },
                    "scan_controller_status_last_report": {
                        "state_name": "GOAL_REACHED",
                    },
                },
            )
            workaround = pipeline._build_summary(
                started_at=0.0,
                duration_steps=1,
                final_state=workaround_state,
                last_action={},
            )
            self.assertEqual(
                workaround["success_semantics"],
                "scan_stair_root_lock_workaround",
            )
            self.assertFalse(workaround["physical_navigation_success"])
            self.assertFalse(workaround["low_level_stair_locomotion_success"])
            self.assertTrue(
                workaround["navigation_root_lock_workaround_success"]
            )
            self.assertEqual(
                workaround["simulation_report"]
                ["navigation_ros2_bridge_report"]["publish_count"],
                8,
            )
            self.assertTrue(
                workaround["simulation_report"]
                ["navigation_stair_execution_frozen_last_publish_report"]
                ["value"]
            )
            self.assertEqual(
                workaround["simulation_report"]
                ["scan_controller_status_last_report"]["state_name"],
                "GOAL_REACHED",
            )

            # 同一 provenance 出现在完整 pipeline 时也必须明确标成非物理
            # 楼梯 root-lock，不能只在 stair-smoke 下关闭 physical 成功位。
            pipeline.config = replace(
                pipeline.config,
                stair_locomotion_smoke=False,
                full_physics=True,
            )
            full_pipeline_workaround = pipeline._build_summary(
                started_at=0.0,
                duration_steps=1,
                final_state=workaround_state,
                last_action={},
            )
            self.assertEqual(
                full_pipeline_workaround["execution_mode"],
                "full_physics",
            )
            self.assertIn(
                "scan_stair_root_lock_workaround",
                full_pipeline_workaround["success_semantics"],
            )
            self.assertFalse(
                full_pipeline_workaround["physical_navigation_success"]
            )
            self.assertTrue(
                full_pipeline_workaround[
                    "navigation_root_lock_workaround_success"
                ]
            )

    def test_strict_stair_completion_status_is_fail_closed(self) -> None:
        cases = (
            (
                "missing_success",
                {"failed": False, "failure_reason": ""},
                "stair_navigation_executor_completion_invalid",
            ),
            (
                "executor_failed",
                {
                    "success": True,
                    "failed": True,
                    "failure_reason": "scan_controller_failed",
                },
                "scan_controller_failed",
            ),
        )
        for case_name, terminal_fields, expected_reason in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                task_path = PROJECT_ROOT / "tasks/nav_smoke_example.json"
                config = FullPhysicsConfig(
                    task_json=task_path,
                    output_dir=root,
                    stair_locomotion_smoke=True,
                )
                spec = JsonTaskProvider().load(task_path)
                gripper = BinaryGripperController()
                verifier = _PickReachabilitySpyVerifier(raise_on_call=True)
                pipeline = FullPhysicsPipeline(
                    config=config,
                    episode_spec=spec,
                    episode_seed=5,
                    simulation=InMemorySimulationRuntime(),
                    nav_planner=DryRunNavPlanner(),
                    nav_executor=_StrictScanCompletionExecutor(terminal_fields),
                    manipulation_planner=SegmentedSmokeManipulationPlanner(),
                    arm_executor=SegmentedArmExecutor(gripper),
                    gripper=gripper,
                    verifier=verifier,
                    recorder=JsonlEpisodeRecorder(root / "episode"),
                )

                summary = pipeline.run_episode()

                self.assertFalse(summary["success"])
                self.assertEqual(summary["failure_reason"], expected_reason)
                self.assertEqual(verifier.pick_reachable_calls, 0)
                self.assertNotIn(
                    PipelineState.VERIFY_PICK_REACHABLE.value,
                    summary["state_trace"],
                )
                self.assertEqual(
                    summary["state_trace"][-2:],
                    [
                        PipelineState.EXEC_NAV_TO_PICK.value,
                        PipelineState.FAILED.value,
                    ],
                )
                events = [
                    json.loads(line)
                    for line in (root / "episode" / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                event_names = {event["name"] for event in events}
                self.assertNotIn("nav_to_pick_success", event_names)
                self.assertNotIn("stair_locomotion_smoke_success", event_names)

    def test_strict_stair_terminal_hold_contract_is_fail_closed(self) -> None:
        cases = (
            (
                "missing_terminal_supervisor_ack",
                _StrictScanCompletionExecutor(
                    {
                        "success": True,
                        "failed": False,
                        "failure_reason": "",
                        "stair_freeze": {
                            "phase": "terminal_hold",
                            "finish_ready": True,
                            "terminal_goal_bound": True,
                            "hold_xyzyaw": _STRICT_STAIR_HOLD_XYZYAW,
                        },
                    }
                ),
                False,
            ),
            (
                "cleanup_action_lost_full_body_lock",
                _StrictScanCompletionExecutor(invalidate_hold_on_call=3),
                True,
            ),
        )
        for case_name, nav_executor, reached_cleanup in cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                task_path = PROJECT_ROOT / "tasks/nav_smoke_example.json"
                config = FullPhysicsConfig(
                    task_json=task_path,
                    output_dir=root,
                    stair_locomotion_smoke=True,
                )
                spec = JsonTaskProvider().load(task_path)
                gripper = BinaryGripperController()
                verifier = _PickReachabilitySpyVerifier(raise_on_call=True)
                pipeline = FullPhysicsPipeline(
                    config=config,
                    episode_spec=spec,
                    episode_seed=5,
                    simulation=InMemorySimulationRuntime(),
                    nav_planner=DryRunNavPlanner(),
                    nav_executor=nav_executor,
                    manipulation_planner=SegmentedSmokeManipulationPlanner(),
                    arm_executor=SegmentedArmExecutor(gripper),
                    gripper=gripper,
                    verifier=verifier,
                    recorder=JsonlEpisodeRecorder(root / "episode"),
                )

                summary = pipeline.run_episode()

                self.assertFalse(summary["success"])
                self.assertEqual(
                    summary["failure_reason"],
                    "stair_navigation_terminal_hold_invalid",
                )
                self.assertEqual(verifier.pick_reachable_calls, 0)
                self.assertNotIn(PipelineState.DONE.value, summary["state_trace"])
                self.assertEqual(
                    PipelineState.CLEANUP_EPISODE.value in summary["state_trace"],
                    reached_cleanup,
                )
                self.assertEqual(
                    summary["state_trace"][-1],
                    PipelineState.FAILED.value,
                )

    def test_navigation_smoke_still_uses_pick_reachability_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_smoke_example.json"
            config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                navigation_smoke=True,
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            verifier = _PickReachabilitySpyVerifier()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=5,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=_StrictScanCompletionExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=verifier,
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(verifier.pick_reachable_calls, 1)
            self.assertIn(
                PipelineState.VERIFY_PICK_REACHABLE.value,
                summary["state_trace"],
            )
            events = [
                json.loads(line)
                for line in (root / "episode" / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIn(
                "navigation_smoke_success",
                {event["name"] for event in events},
            )
            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            handoff_states = (
                PipelineState.EXEC_NAV_TO_PICK.value,
                PipelineState.VERIFY_PICK_REACHABLE.value,
                PipelineState.CLEANUP_EPISODE.value,
            )
            handoff_frames = [
                frame
                for frame in frames
                if frame["pipeline_state"] in handoff_states
            ]
            self.assertEqual(
                [frame["pipeline_state"] for frame in handoff_frames],
                list(handoff_states),
            )
            for frame in handoff_frames:
                action = frame["action"]
                metadata = action["metadata"]
                self.assertEqual(action["base_velocity"], [0.0, 0.0, 0.0])
                self.assertTrue(metadata["navigation_base_pose_lock"])
                self.assertEqual(
                    metadata["navigation_base_pose_lock_xyzyaw"],
                    list(_STRICT_STAIR_HOLD_XYZYAW),
                )
                self.assertTrue(metadata["navigation_support_joint_lock"])
                self.assertTrue(metadata["navigation_full_body_joint_lock"])
                self.assertTrue(metadata["navigation_cmd_vel_inhibit"])

    def test_stair_fixed_command_probe_keeps_legacy_verifier_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_path = PROJECT_ROOT / "tasks/nav_smoke_example.json"
            base_config = FullPhysicsConfig(
                task_json=task_path,
                output_dir=root,
                stair_locomotion_smoke=True,
            )
            config = replace(
                base_config,
                navigation=replace(
                    base_config.navigation,
                    stair_fixed_command_probe=True,
                ),
            )
            spec = JsonTaskProvider().load(task_path)
            gripper = BinaryGripperController()
            verifier = _PickReachabilitySpyVerifier()
            pipeline = FullPhysicsPipeline(
                config=config,
                episode_spec=spec,
                episode_seed=5,
                simulation=InMemorySimulationRuntime(),
                nav_planner=DryRunNavPlanner(),
                nav_executor=_StrictScanCompletionExecutor(),
                manipulation_planner=SegmentedSmokeManipulationPlanner(),
                arm_executor=SegmentedArmExecutor(gripper),
                gripper=gripper,
                verifier=verifier,
                recorder=JsonlEpisodeRecorder(root / "episode"),
            )

            summary = pipeline.run_episode()

            self.assertTrue(summary["success"])
            self.assertEqual(verifier.pick_reachable_calls, 1)
            self.assertIn(
                PipelineState.VERIFY_PICK_REACHABLE.value,
                summary["state_trace"],
            )
            frames = [
                json.loads(line)
                for line in (root / "episode" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(
                any(
                    frame["action"]["source"] == "stair_navigation_complete"
                    for frame in frames
                )
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
