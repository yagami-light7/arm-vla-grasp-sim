"""Single-owner simulation loop for one full-physics episode."""

from __future__ import annotations

import faulthandler
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from source.diagnostics.performance import WallTimeProfiler
from source.interfaces import (
    ArmExecutor,
    EpisodeRecorder,
    EpisodeSpec,
    EpisodeVerifier,
    GripperController,
    ManipulationPlanner,
    NavExecutor,
    NavPlanner,
    SimulationRuntime,
    StepRecord,
)
from source.recording.overview_video_recorder import OverviewVideoRecorder

from .config import FullPhysicsConfig
from .state_machine import FullPhysicsStateMachine


_INITIAL_TICK_WATCHDOG_SECONDS = 90.0


def _should_auto_switch_overview_camera(config: FullPhysicsConfig) -> bool:
    """让录像器按配置管理 overview，相机调度不受 GUI/headless 限制。"""

    return config.video.overview_camera_mode in {"auto", "fixed"}


class FullPhysicsPipeline:
    """Own the only simulation step loop and coordinate one episode."""

    def __init__(
        self,
        *,
        config: FullPhysicsConfig,
        episode_spec: EpisodeSpec,
        episode_seed: int,
        simulation: SimulationRuntime,
        nav_planner: NavPlanner,
        nav_executor: NavExecutor,
        manipulation_planner: ManipulationPlanner,
        arm_executor: ArmExecutor,
        gripper: GripperController,
        verifier: EpisodeVerifier,
        recorder: EpisodeRecorder,
        close_simulation_on_exit: bool = True,
    ):
        self.config = config
        self.episode_spec = episode_spec
        self.episode_seed = episode_seed
        self.simulation = simulation
        self.nav_planner = nav_planner
        self.recorder = recorder
        self._close_simulation_on_exit = bool(close_simulation_on_exit)
        self._profiler: WallTimeProfiler | None = None
        self.machine = FullPhysicsStateMachine(
            config=config,
            episode_spec=episode_spec,
            episode_seed=episode_seed,
            simulation=simulation,
            nav_planner=nav_planner,
            nav_executor=nav_executor,
            manipulation_planner=manipulation_planner,
            arm_executor=arm_executor,
            gripper=gripper,
            verifier=verifier,
            recorder=recorder,
        )

    def run_episode(self) -> dict[str, Any]:
        started_at = time.time()
        self._profiler = WallTimeProfiler()
        for component in (self.simulation, self.recorder):
            set_profiler = getattr(component, "set_performance_profiler", None)
            if callable(set_profiler):
                set_profiler(self._profiler)
        duration_steps = 0
        last_action: dict[str, Any] = {}
        video_recorder = (
            OverviewVideoRecorder(
                settings=self.config.video,
                episode_dir=self.recorder.output_dir,
                episode_id=self.episode_spec.episode_id,
                auto_switch_camera=_should_auto_switch_overview_camera(self.config),
                save_overview_images=bool(
                    self.config.recording.enabled
                    and self.config.recording.save_raw_images
                ),
                overview_image_fps=float(self.config.recording.dataset_fps),
                overview_jpeg_quality=int(self.config.recording.jpeg_quality),
            )
            if self.config.video.enabled
            else None
        )
        video_closed = False
        current_operation = "pipeline_start"
        startup_status_path = (
            self.recorder.output_dir / "pipeline_startup_status.json"
        )
        startup_trace_path = (
            self.recorder.output_dir / "pipeline_startup_traceback.log"
        )
        startup_phases: list[dict[str, Any]] = []
        startup_watchdog_armed = False
        startup_trace_stream: Any | None = None

        def _record_startup_phase(
            phase: str,
            *,
            status: str = "starting",
            **details: Any,
        ) -> None:
            """原子记录首次状态机 tick 的细粒度进度，避免启动卡死时没有证据。"""

            startup_phases.append(
                {
                    "phase": str(phase),
                    "wall_time": time.time(),
                    **details,
                }
            )
            payload = {
                "status": str(status),
                "pid": os.getpid(),
                "watchdog_timeout_s": _INITIAL_TICK_WATCHDOG_SECONDS,
                "traceback_path": str(startup_trace_path),
                "phases": startup_phases,
            }
            temporary_path = startup_status_path.with_suffix(".json.tmp")
            try:
                temporary_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(startup_status_path)
            except OSError as exc:
                print(
                    "[full-physics] 写入启动诊断失败："
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        def _arm_startup_watchdog() -> None:
            """只监控首个状态机 tick；正常进入回合后立即取消。"""

            nonlocal startup_trace_stream, startup_watchdog_armed
            try:
                startup_trace_stream = startup_trace_path.open(
                    "w",
                    encoding="utf-8",
                )
                faulthandler.dump_traceback_later(
                    _INITIAL_TICK_WATCHDOG_SECONDS,
                    repeat=False,
                    file=startup_trace_stream,
                )
                startup_watchdog_armed = True
            except (OSError, RuntimeError, ValueError) as exc:
                if startup_trace_stream is not None:
                    startup_trace_stream.close()
                    startup_trace_stream = None
                _record_startup_phase(
                    "initial_tick_watchdog_unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )

        def _cancel_startup_watchdog() -> None:
            nonlocal startup_trace_stream, startup_watchdog_armed
            if startup_watchdog_armed:
                faulthandler.cancel_dump_traceback_later()
                startup_watchdog_armed = False
            if startup_trace_stream is not None:
                startup_trace_stream.close()
                startup_trace_stream = None

        def _close_video(status: str) -> dict[str, Any] | None:
            nonlocal video_closed
            if video_recorder is None or video_closed:
                return None
            video_closed = True
            assert self._profiler is not None
            with self._profiler.measure("pipeline.video_close"):
                return video_recorder.close(status=status)

        _record_startup_phase("run_episode_entered")
        with self._profiler.measure("pipeline.recorder_save_task"):
            self.recorder.save_task(self.episode_spec)
        _record_startup_phase("task_saved")
        self.recorder.mark_training_eligible(
            False,
            reason="episode_not_verified_yet",
        )
        _arm_startup_watchdog()
        try:
            if video_recorder is not None:
                current_operation = "video_start_episode"
                with self._profiler.measure("pipeline.video_start"):
                    video_recorder.start_episode()
            while True:
                current_operation = "simulation_read_before_tick"
                if duration_steps == 0:
                    _record_startup_phase("initial_simulation_read_started")
                with self._profiler.measure(
                    "pipeline.simulation_read_before_tick"
                ):
                    observation = self.simulation.read()
                if duration_steps == 0:
                    _record_startup_phase(
                        "initial_simulation_read_finished",
                        simulation_step_index=int(observation.step_index),
                    )
                current_operation = "state_machine_tick"
                state_before_tick = self.machine.state.value
                if duration_steps == 0:
                    _record_startup_phase(
                        "initial_state_machine_tick_started",
                        pipeline_state=state_before_tick,
                    )
                with self._profiler.measure("pipeline.state_machine_tick"):
                    with self._profiler.measure(
                        f"pipeline.state.{state_before_tick}.tick"
                    ):
                        decision = self.machine.tick(observation)
                if duration_steps == 0:
                    _cancel_startup_watchdog()
                    _record_startup_phase(
                        "initial_state_machine_tick_finished",
                        status="completed",
                        pipeline_state=state_before_tick,
                        next_pipeline_state=decision.state.value,
                    )
                current_operation = "simulation_apply"
                with self._profiler.measure("pipeline.simulation_apply"):
                    self.simulation.apply(decision.action)
                current_operation = "record_pipeline_events"
                with self._profiler.measure("pipeline.record_events"):
                    for event in decision.events:
                        self.recorder.record_event(event.to_dict())

                skip_physics_step = bool(decision.action.metadata.get("skip_physics_step"))
                if not skip_physics_step:
                    current_operation = "simulation_step"
                    with self._profiler.measure("pipeline.simulation_step"):
                        self.simulation.step(render=self.config.render)
                current_operation = "simulation_read_after_step"
                with self._profiler.measure("pipeline.simulation_read_after_step"):
                    post_step = self.simulation.read()
                if video_recorder is not None and not skip_physics_step:
                    current_operation = "video_add_frame"
                    with self._profiler.measure("pipeline.video_add_frame"):
                        video_recorder.add_frame(
                            state=decision.state.value,
                            timestamp=post_step.timestamp,
                            step_index=duration_steps,
                            camera_images=post_step.camera_images,
                            robot_root_pose=post_step.robot_root_pose,
                        )
                current_operation = "record_step"
                with self._profiler.measure("pipeline.recorder_record_step"):
                    self.recorder.record_step(
                        StepRecord(
                            step_index=duration_steps,
                            timestamp=observation.timestamp,
                            pipeline_state=decision.state.value,
                            observation=observation,
                            action=decision.action,
                            post_step_observation=post_step,
                            metadata=decision.metadata,
                        )
                    )
                duration_steps += 1
                last_action = {
                    "source": decision.action.source,
                    "base_velocity": decision.action.base_velocity,
                    "arm_joint_positions": decision.action.arm_joint_positions,
                    "gripper_command": decision.action.gripper_command,
                    "metadata": decision.action.metadata,
                    "physics_step_skipped": skip_physics_step,
                }
                if decision.terminal:
                    break

            if self.config.keep_window_open:
                current_operation = "simulation_pause"
                with self._profiler.measure("pipeline.simulation_pause"):
                    self.simulation.pause()
                if hasattr(self.simulation, "refresh_viewport"):
                    current_operation = "simulation_refresh_viewport"
                    with self._profiler.measure("pipeline.simulation_refresh_viewport"):
                        self.simulation.refresh_viewport(reason="keep_window_open")
            current_operation = "simulation_read_final"
            with self._profiler.measure("pipeline.simulation_read_final"):
                final_state = self.simulation.read()
            summary = self._build_summary(
                started_at=started_at,
                duration_steps=duration_steps,
                final_state=final_state,
                last_action=last_action,
            )
            video_summary = _close_video("success" if summary["success"] else "failed")
            if video_summary is not None:
                summary["overview_video"] = video_summary
            summary["performance_report"] = self._performance_report(
                duration_steps=duration_steps,
                final_state=final_state,
            )
            with self._profiler.measure("pipeline.recorder_close"):
                summary_path = self.recorder.close(summary)
            return self._finalize_performance_report(
                summary_path,
                duration_steps=duration_steps,
                final_state=final_state,
            )
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            if not any(
                phase.get("phase") == "initial_state_machine_tick_finished"
                for phase in startup_phases
            ):
                _record_startup_phase(
                    "startup_failed",
                    status="interrupted" if interrupted else "failed",
                    operation=current_operation,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            video_summary = _close_video("interrupted" if interrupted else "failed")
            failure_reason = "pipeline_interrupted" if interrupted else "pipeline_runtime_exception"
            exception_report = {
                "operation": current_operation,
                "pipeline_state": self.machine.state.value,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
            try:
                self.recorder.record_event(
                    {
                        "name": failure_reason,
                        "pipeline_state": self.machine.state.value,
                        "step_index": duration_steps,
                        "timestamp": time.time(),
                        "metadata": exception_report,
                    }
                )
                failure_summary = self._build_runtime_failure_summary(
                    started_at=started_at,
                    duration_steps=duration_steps,
                    failure_reason=failure_reason,
                    exception_report=exception_report,
                    video_summary=video_summary,
                )
                failure_summary["performance_report"] = self._performance_report(
                    duration_steps=duration_steps,
                    final_state=None,
                )
                with self._profiler.measure("pipeline.recorder_close_failure"):
                    failure_path = self.recorder.close(failure_summary)
                self._finalize_performance_report(
                    failure_path,
                    duration_steps=duration_steps,
                    final_state=None,
                )
            except Exception as recorder_exc:
                print(
                    "[full-physics] 写入运行时失败报告时再次失败："
                    f"{type(recorder_exc).__name__}: {recorder_exc}",
                    flush=True,
                )
            print(
                "[full-physics] pipeline 未完成："
                f"state={self.machine.state.value} operation={current_operation} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        finally:
            _cancel_startup_watchdog()
            _close_video("closed_without_summary")
            close_nav_planner = getattr(self.nav_planner, "close", None)
            if callable(close_nav_planner):
                if self._profiler is None:
                    close_nav_planner()
                else:
                    with self._profiler.measure("pipeline.nav_planner_close"):
                        close_nav_planner()
            if not self.config.keep_window_open and self._close_simulation_on_exit:
                if self._profiler is None:
                    self.simulation.close()
                else:
                    with self._profiler.measure("pipeline.simulation_close"):
                        self.simulation.close()

    def _performance_report(
        self,
        *,
        duration_steps: int,
        final_state: Any | None,
    ) -> dict[str, Any]:
        assert self._profiler is not None
        metadata = getattr(final_state, "metadata", {}) if final_state is not None else {}
        control_dt = metadata.get("control_dt") if isinstance(metadata, dict) else None
        physics_dt = metadata.get("physics_dt") if isinstance(metadata, dict) else None
        decimation = metadata.get("decimation") if isinstance(metadata, dict) else None
        simulation_steps = (
            int(getattr(final_state, "step_index", 0)) if final_state is not None else None
        )
        simulation_seconds = (
            float(getattr(final_state, "timestamp", 0.0)) if final_state is not None else None
        )
        wall_seconds = self._profiler.elapsed_seconds()
        return self._profiler.report(
            episode_id=self.episode_spec.episode_id,
            seed=self.episode_seed,
            pipeline_ticks=int(duration_steps),
            simulation_control_steps=simulation_steps,
            simulation_seconds=simulation_seconds,
            real_time_factor=(
                simulation_seconds / wall_seconds
                if isinstance(simulation_seconds, (int, float)) and wall_seconds > 0.0
                else None
            ),
            timing_invariants={
                "physics_dt": physics_dt,
                "control_dt": control_dt,
                "decimation": decimation,
            },
        )

    def _finalize_performance_report(
        self,
        summary_path: Path,
        *,
        duration_steps: int,
        final_state: Any | None,
    ) -> dict[str, Any]:
        """Persist timings that become known only after recorder finalization."""

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report = self._performance_report(
            duration_steps=duration_steps,
            final_state=final_state,
        )
        report["artifact_sizes_bytes"] = {
            name: path.stat().st_size
            for name in ("events.jsonl", "frames.jsonl", "samples.jsonl", "data.csv")
            if (path := self.recorder.output_dir / name).is_file()
        }
        summary["performance_report"] = report
        temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(summary_path)
        return summary

    def _build_runtime_failure_summary(
        self,
        *,
        started_at: float,
        duration_steps: int,
        failure_reason: str,
        exception_report: dict[str, Any],
        video_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """构造异常退出摘要，避免真实仿真只留下过期的 summary。"""

        machine_fields = self.machine.summary_fields()
        if self.config.dry_run:
            execution_mode = "dry_run"
        elif self.config.simulation_smoke:
            execution_mode = "simulation_smoke"
        elif self.config.stair_locomotion_smoke:
            execution_mode = "stair_locomotion_smoke"
        elif self.config.navigation_smoke:
            execution_mode = "navigation_smoke"
        elif self.config.navigation_carry_smoke:
            execution_mode = "navigation_carry_smoke"
        elif self.config.pick_smoke:
            execution_mode = "pick_smoke"
        elif self.config.manipulation_smoke:
            execution_mode = "manipulation_smoke"
        elif self.config.manipulation_apply_smoke:
            execution_mode = "manipulation_apply_smoke"
        else:
            execution_mode = "full_physics"
        summary = {
            "episode_id": self.episode_spec.episode_id,
            "task_id": self.episode_spec.task_id,
            "seed": self.episode_seed,
            "success": False,
            "pure_physics_success": False,
            "stable_physics_success": False,
            "physical_navigation_success": False,
            "physical_manipulation_success": False,
            "execution_mode": execution_mode,
            "failure_reason": failure_reason,
            "duration_steps": duration_steps,
            "duration_seconds": time.time() - started_at,
            "runtime_exception": exception_report,
            **machine_fields,
        }
        # 运行时异常优先于状态机尚未设置的 failure_reason。
        summary["success"] = False
        summary["failure_reason"] = failure_reason
        if video_summary is not None:
            summary["overview_video"] = video_summary
        return summary

    def _build_summary(
        self,
        *,
        started_at: float,
        duration_steps: int,
        final_state: Any,
        last_action: dict[str, Any],
    ) -> dict[str, Any]:
        machine_fields = self.machine.summary_fields()
        success = bool(machine_fields["success"])
        dry_run = bool(self.config.dry_run)
        simulation_smoke = bool(self.config.simulation_smoke)
        navigation_smoke = bool(self.config.navigation_smoke)
        navigation_carry_smoke = bool(self.config.navigation_carry_smoke)
        stair_locomotion_smoke = bool(self.config.stair_locomotion_smoke)
        pick_smoke = bool(self.config.pick_smoke)
        manipulation_smoke = bool(self.config.manipulation_smoke)
        manipulation_apply_smoke = bool(self.config.manipulation_apply_smoke)
        full_physics = bool(self.config.full_physics)
        provenance = {
            "used_base_teleport": bool(final_state.metadata.get("used_base_teleport", False)),
            "used_direct_joint_state": bool(final_state.metadata.get("used_direct_joint_state", False)),
            "used_object_teleport": bool(final_state.metadata.get("used_object_teleport", False)),
            "used_kinematic_object_follow": bool(
                final_state.metadata.get("used_kinematic_object_follow", False)
            ),
            "used_visual_replay": bool(final_state.metadata.get("used_visual_replay", False)),
            "used_manipulation_base_lock": bool(
                final_state.metadata.get("used_manipulation_base_lock", False)
            ),
            "used_manipulation_support_joint_lock": bool(
                final_state.metadata.get("used_manipulation_support_joint_lock", False)
            ),
            "used_navigation_base_lock": bool(
                final_state.metadata.get("used_navigation_base_lock", False)
            ),
            "used_navigation_support_joint_lock": bool(
                final_state.metadata.get(
                    "used_navigation_support_joint_lock",
                    False,
                )
            ),
            "used_navigation_joint_pose_lock": bool(
                final_state.metadata.get(
                    "used_navigation_joint_pose_lock",
                    False,
                )
            ),
        }
        provenance_verified = bool(
            (not dry_run) and final_state.metadata.get("execution_provenance_verified", False)
        )
        pure_physics_success = (
            success
            and not dry_run
            and not simulation_smoke
            and not navigation_smoke
            and not navigation_carry_smoke
            and not stair_locomotion_smoke
            and not pick_smoke
            and not manipulation_smoke
            and not manipulation_apply_smoke
            and provenance_verified
            and not any(provenance.values())
        )
        # 楼梯 root lock 的非物理语义由实际 provenance 决定，不能只在
        # stair-smoke 模式识别；完整 carry/place pipeline 同样会经过该动作。
        navigation_root_lock_workaround = bool(
            provenance["used_navigation_base_lock"]
        )
        stable_physics_success = bool(success and full_physics and provenance_verified)
        vla_training_action_requested = bool(
            isinstance(self.episode_spec.raw_task.get("training_action"), dict)
            and self.episode_spec.raw_task["training_action"].get("enabled", False)
        )
        visual_scene_report = final_state.metadata.get("visual_scene_report")
        camera_capture_report = final_state.metadata.get("camera_capture_report")
        overview_camera_report = final_state.metadata.get("overview_camera_report")
        required_camera_keys = (
            set(self.config.recording.camera_keys)
            if self.config.recording.enabled
            else set()
        )
        available_camera_keys = set(
            camera_capture_report.get("available_camera_keys", ())
            if isinstance(camera_capture_report, dict)
            else ()
        )
        camera_capture_complete = bool(
            isinstance(camera_capture_report, dict)
            and required_camera_keys.issubset(available_camera_keys)
        )
        overview_camera_verified = bool(
            "overview" not in required_camera_keys
            or (
                isinstance(overview_camera_report, dict)
                and overview_camera_report.get("enabled") is True
                and overview_camera_report.get("prim_path")
                == self.config.recording.overview_camera_prim_path
            )
        )
        training_visual_source_verified = bool(
            not vla_training_action_requested
            or (
                isinstance(visual_scene_report, dict)
                and visual_scene_report.get("loaded") is True
                and camera_capture_complete
                and overview_camera_verified
            )
        )
        training_visual_policy = {
            "gaussian_scene_required": False,
            "gaussian_scene_enabled": bool(
                isinstance(visual_scene_report, dict)
                and visual_scene_report.get("scene_visual_enabled") is True
            ),
            "required_camera_keys": sorted(required_camera_keys),
            "available_camera_keys": sorted(available_camera_keys),
            "camera_capture_complete": camera_capture_complete,
            "overview_camera_verified": overview_camera_verified,
            "overview_camera_prim_path": (
                self.config.recording.overview_camera_prim_path
            ),
        }
        simulation_report = {
            key: final_state.metadata.get(key)
            for key in (
                "simulation_ready",
                "world_count",
                "opened_stage_count",
                "stage_build_count",
                "stage_reuse_count",
                "stage_reuse_report",
                "articulation_prim_path",
                "object_root_prim_path",
                "object_state_prim_path",
                "tcp_prim_path",
                "camera_prim_path",
                "front_camera_report",
                "wrist_camera_report",
                "wrist_camera_object_clearance_report",
                "camera_runtime_intrinsics_report",
                "d436_lens_distortion_schema_report",
                "overview_camera_report",
                "camera_capture_report",
                "camera_render_schedule",
                "camera_render_interval_control_steps",
                "camera_render_hz",
                "gripper_collision_patch_report",
                "apple_collision_patch_report",
                "stage_report",
                "visual_scene_report",
                "task_receptacle_support_source_report",
                "task_receptacle_support_runtime_stage_report",
                "task_receptacle_pose_report",
                "last_current_state_curobo_pick_export",
                "last_current_state_curobo_place_export",
                "last_mesh_truth_pick_target_report",
                "last_mesh_truth_place_target_report",
                "viewport_report",
                "object_pose_setup_report",
                "object_pose_setup_before_physics_report",
                "object_pose_setup_after_physics_report",
                "object_pose_setup_after_reset_report",
                "object_pose_debug_after_reset",
                "object_pose_debug_latest",
                "episode_reset_complete",
                "used_episode_reset_pose",
                "reset_robot_root_pose",
                "last_arm_action_report",
                "last_joint_action_report",
                "last_arm_tracking_report",
                "arm_tracking_peak_report",
                "arm_tracking_report",
                "arm_tracking_sample_count",
                "arm_tracking_max_abs_error",
                "last_gripper_action_report",
                "joint_action_apply_count",
                "arm_joint_action_apply_count",
                "gripper_joint_action_apply_count",
                "gripper_close_apply_count",
                "gripper_open_apply_count",
                "arm_joint_position_target_apply_count",
                "last_arm_joint_position_target_report",
                "gripper_joint_position_target_apply_count",
                "last_gripper_joint_position_target_report",
                "used_manipulation_base_lock",
                "used_manipulation_support_joint_lock",
                "manipulation_base_lock_active",
                "manipulation_base_lock_apply_count",
                "last_manipulation_base_lock_report",
                "manipulation_support_joint_lock_active",
                "manipulation_support_joint_lock_apply_count",
                "last_manipulation_support_joint_lock_report",
                "used_navigation_base_lock",
                "used_navigation_support_joint_lock",
                "used_navigation_joint_pose_lock",
                "last_navigation_base_lock_report",
                "last_navigation_support_joint_lock_report",
                "navigation_joint_pose_lock_active",
                "navigation_joint_pose_lock_apply_count",
                "last_navigation_joint_pose_lock_report",
                "navigation_ros2_bridge_report",
                "navigation_policy_gate_lifecycle_report",
                "grid_map_observation_diagnostics_last_report",
                "grid_map_observation_lifecycle_report",
                "bspline_diagnostics_last_report",
                "bspline_diagnostics_lifecycle_report",
                "active_sensing_lifecycle_report",
                "dynamic_navigation_evidence_report",
                "dynamic_obstacle_configuration_report",
                "dynamic_obstacle_runtime_report",
                "dynamic_obstacle_lifecycle_report",
                "dynamic_obstacle_raw_cloud_last_report",
                "dynamic_obstacle_raw_cloud_lifecycle_report",
                "dynamic_obstacle_pose_write_count",
                "navigation_stair_execution_frozen_last_publish_report",
                "scan_controller_status_last_report",
                "scan_controller_status_lifecycle_report",
                "object_reset_for_navigation_report",
                "object_settle_begin_report",
                "object_settle_final_report",
                "object_prepare_for_pick_report",
                "terminal_hold_report",
            )
            if key in final_state.metadata
        }
        if dry_run:
            execution_mode = "dry_run"
            success_semantics = "control_flow_only"
        elif simulation_smoke:
            execution_mode = "simulation_smoke"
            success_semantics = "stage_build_and_reset_only"
        elif stair_locomotion_smoke:
            execution_mode = "stair_locomotion_smoke"
            success_semantics = (
                "scan_stair_root_lock_workaround"
                if navigation_root_lock_workaround
                else "pure_physics_stair_locomotion_without_dwa_or_float"
            )
        elif navigation_smoke:
            execution_mode = "navigation_smoke"
            success_semantics = "physical_nav_to_pick_only"
        elif navigation_carry_smoke:
            execution_mode = "navigation_carry_smoke"
            success_semantics = "physical_nav_to_place_with_arm_gripper_hold"
        elif pick_smoke:
            execution_mode = "pick_smoke"
            success_semantics = "physical_nav_to_pick_and_pick_only"
        elif manipulation_smoke:
            execution_mode = "manipulation_smoke"
            success_semantics = "segmented_manipulation_contract_only"
        elif manipulation_apply_smoke:
            execution_mode = "manipulation_apply_smoke"
            success_semantics = "isaac_joint_action_apply_only"
        elif full_physics:
            execution_mode = "full_physics"
            success_semantics = (
                "stable_physical_execution_with_base_support_lock"
                if (
                    self.config.manipulation.lock_base_during_manipulation
                    or self.config.manipulation.lock_support_joints_during_manipulation
                )
                else "strict_physical_execution"
            )
        else:
            execution_mode = "full_physics"
            success_semantics = "physical_execution"
        if navigation_root_lock_workaround and not stair_locomotion_smoke:
            success_semantics = (
                f"{success_semantics}_with_scan_stair_root_lock_workaround"
            )
        navigation_acceptance = None
        if (
            navigation_smoke
            or navigation_carry_smoke
            or stair_locomotion_smoke
            or pick_smoke
            or full_physics
        ):
            navigation_acceptance = {
                "global_planner": self.config.navigation.global_planner,
                "mode": (
                    "xy_yaw_stable"
                    if self.config.navigation.require_yaw_alignment
                    and self.config.navigation.require_stable_base
                    else "xy_only"
                ),
                "position_tolerance": self.config.navigation.final_position_tolerance,
                "place_position_tolerance": (
                    self.config.navigation.place_position_tolerance
                    if self.config.navigation.place_position_tolerance is not None
                    else self.config.navigation.final_position_tolerance
                ),
                "goal_z_tolerance": self.config.navigation.goal_z_tolerance,
                "yaw_alignment_required": self.config.navigation.require_yaw_alignment,
                "base_stability_required": self.config.navigation.require_stable_base,
                "yaw_tolerance": self.config.navigation.final_yaw_tolerance,
                "linear_velocity_tolerance": self.config.navigation.stable_linear_velocity,
                "angular_velocity_tolerance": self.config.navigation.stable_angular_velocity,
            }
        randomization = self.episode_spec.raw_task.get("randomization")
        randomization = randomization if isinstance(randomization, dict) else {}
        base_goal_randomization = randomization.get("base_goal_randomization")
        base_goal_randomization = (
            base_goal_randomization
            if isinstance(base_goal_randomization, dict)
            else {}
        )
        pick_base_goal_sample = base_goal_randomization.get("pick")
        pick_base_goal_sample = (
            pick_base_goal_sample if isinstance(pick_base_goal_sample, dict) else {}
        )
        place_base_goal_sample = base_goal_randomization.get("place")
        place_base_goal_sample = (
            place_base_goal_sample if isinstance(place_base_goal_sample, dict) else {}
        )
        forward_sector_sample = randomization.get("sample")
        forward_sector_sample = (
            forward_sector_sample
            if isinstance(forward_sector_sample, dict)
            else {}
        )

        def _forward_sector_goal_sample(stage_name: str) -> list[float] | None:
            sample = forward_sector_sample.get(f"{stage_name}_base_goal")
            if not isinstance(sample, dict):
                return None
            try:
                return [
                    float(sample["x"]),
                    float(sample["y"]),
                    float(sample["yaw"]),
                ]
            except (KeyError, TypeError, ValueError):
                return None

        pick_base_goal_sampled = pick_base_goal_sample.get(
            "sampled_base_goal_xyyaw",
        ) or _forward_sector_goal_sample("pick")
        place_base_goal_sampled = place_base_goal_sample.get(
            "sampled_base_goal_xyyaw",
        ) or _forward_sector_goal_sample("place")
        base_goal_randomization_enabled = bool(
            base_goal_randomization.get("enabled", False)
            or forward_sector_sample.get("base_goal_randomized", False)
        )
        return {
            "episode_id": self.episode_spec.episode_id,
            "task_id": self.episode_spec.task_id,
            "seed": self.episode_seed,
            "task_config": self.episode_spec.raw_task,
            "object_initial_pose": self.episode_spec.object_initial_pose,
            "pick_target": {
                "base_goal": (
                    self.episode_spec.pick_goal.x,
                    self.episode_spec.pick_goal.y,
                    self.episode_spec.pick_goal.yaw,
                ),
                "object_prim_path": self.episode_spec.object_prim_path,
            },
            "place_target": {
                "base_goal": None
                if self.episode_spec.place_goal is None
                else (
                    self.episode_spec.place_goal.x,
                    self.episode_spec.place_goal.y,
                    self.episode_spec.place_goal.yaw,
                ),
                "object_pose": self.episode_spec.place_target_pose,
            },
            "final_object_pose": final_state.object_pose,
            "final_robot_pose": final_state.robot_root_pose,
            "last_action": last_action,
            "duration_steps": duration_steps,
            "duration_seconds": time.time() - started_at,
            "data_output_path": str(self.recorder.output_dir),
            "lerobot_training_eligible": bool(success),
            "lerobot_export_skipped": not bool(success),
            "lerobot_export_skip_reason": (
                machine_fields.get("failure_reason") or "episode_failed"
                if not success
                else None
            ),
            "execution_mode": execution_mode,
            "success_semantics": success_semantics,
            "pure_physics_success": pure_physics_success,
            "stable_physics_success": stable_physics_success,
            "physical_navigation_success": bool(
                success
                and (
                    navigation_smoke
                    or navigation_carry_smoke
                    or stair_locomotion_smoke
                    or pick_smoke
                    or full_physics
                )
                and provenance_verified
                and not navigation_root_lock_workaround
            ),
            "low_level_stair_locomotion_success": bool(
                success
                and stair_locomotion_smoke
                and provenance_verified
                and not navigation_root_lock_workaround
            ),
            "navigation_root_lock_workaround_success": bool(
                success
                and navigation_root_lock_workaround
                and provenance_verified
            ),
            "carry_control_success": bool(
                success and (navigation_carry_smoke or full_physics)
            ),
            "object_carry_verified": bool(success and full_physics),
            "physical_manipulation_success": bool(success and (pick_smoke or full_physics)),
            "manipulation_apply_success": bool(
                success and (manipulation_apply_smoke or pick_smoke or full_physics)
            ),
            "manipulation_base_lock_requested": bool(
                self.config.manipulation.lock_base_during_manipulation
            ),
            "manipulation_support_joint_lock_requested": bool(
                self.config.manipulation.lock_support_joints_during_manipulation
            ),
            "execution_provenance_verified": provenance_verified,
            "training_visual_source_verified": training_visual_source_verified,
            "training_visual_policy": training_visual_policy,
            "simulation_report": simulation_report,
            "navigation_acceptance": navigation_acceptance,
            "task_randomization_mode": randomization.get("mode"),
            "base_goal_randomization_enabled": base_goal_randomization_enabled,
            "pick_base_goal_sampled": pick_base_goal_sampled,
            "place_base_goal_sampled": place_base_goal_sampled,
            "pick_base_goal_fallback_used": bool(
                pick_base_goal_sample.get("fallback_used", False)
            ),
            "place_base_goal_fallback_used": bool(
                place_base_goal_sample.get("fallback_used", False)
            ),
            **provenance,
            **machine_fields,
        }
