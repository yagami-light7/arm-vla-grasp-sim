"""Single-owner simulation loop for one full-physics episode."""

from __future__ import annotations

import time
from typing import Any

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

from .config import FullPhysicsConfig
from .state_machine import FullPhysicsStateMachine


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
    ):
        self.config = config
        self.episode_spec = episode_spec
        self.episode_seed = episode_seed
        self.simulation = simulation
        self.recorder = recorder
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
        duration_steps = 0
        last_action: dict[str, Any] = {}
        self.recorder.save_task(self.episode_spec)
        try:
            while True:
                observation = self.simulation.read()
                decision = self.machine.tick(observation)
                self.simulation.apply(decision.action)
                for event in decision.events:
                    self.recorder.record_event(event.to_dict())

                skip_physics_step = bool(decision.action.metadata.get("skip_physics_step"))
                if not skip_physics_step:
                    self.simulation.step(render=self.config.render)
                post_step = self.simulation.read()
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
                self.simulation.pause()
                if hasattr(self.simulation, "refresh_viewport"):
                    self.simulation.refresh_viewport(reason="keep_window_open")
            final_state = self.simulation.read()
            summary = self._build_summary(
                started_at=started_at,
                duration_steps=duration_steps,
                final_state=final_state,
                last_action=last_action,
            )
            self.recorder.close(summary)
            return summary
        finally:
            if not self.config.keep_window_open:
                self.simulation.close()

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
        manipulation_smoke = bool(self.config.manipulation_smoke)
        manipulation_apply_smoke = bool(self.config.manipulation_apply_smoke)
        integrated_apply_smoke = bool(self.config.integrated_apply_smoke)
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
            and not manipulation_smoke
            and not manipulation_apply_smoke
            and not integrated_apply_smoke
            and provenance_verified
            and not any(provenance.values())
        )
        stable_physics_success = bool(success and full_physics and provenance_verified)
        simulation_report = {
            key: final_state.metadata.get(key)
            for key in (
                "simulation_ready",
                "world_count",
                "opened_stage_count",
                "articulation_prim_path",
                "object_root_prim_path",
                "object_state_prim_path",
                "tcp_prim_path",
                "camera_prim_path",
                "front_camera_report",
                "wrist_camera_report",
                "camera_capture_report",
                "stage_report",
                "visual_scene_report",
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
                "object_reset_for_navigation_report",
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
        elif navigation_smoke:
            execution_mode = "navigation_smoke"
            success_semantics = "physical_nav_to_pick_only"
        elif navigation_carry_smoke:
            execution_mode = "navigation_carry_smoke"
            success_semantics = "physical_nav_to_place_with_arm_gripper_hold"
        elif manipulation_smoke:
            execution_mode = "manipulation_smoke"
            success_semantics = "segmented_manipulation_contract_only"
        elif manipulation_apply_smoke:
            execution_mode = "manipulation_apply_smoke"
            success_semantics = "isaac_joint_action_apply_only"
        elif integrated_apply_smoke:
            execution_mode = "integrated_apply_smoke"
            success_semantics = "single_stage_nav_manipulation_action_apply_only"
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
        navigation_acceptance = None
        if navigation_smoke or navigation_carry_smoke or integrated_apply_smoke or full_physics:
            navigation_acceptance = {
                "mode": (
                    "xy_yaw_stable"
                    if self.config.navigation.require_yaw_alignment
                    and self.config.navigation.require_stable_base
                    else "xy_only"
                ),
                "position_tolerance": self.config.navigation.final_position_tolerance,
                "yaw_alignment_required": self.config.navigation.require_yaw_alignment,
                "base_stability_required": self.config.navigation.require_stable_base,
                "yaw_tolerance": self.config.navigation.final_yaw_tolerance,
                "linear_velocity_tolerance": self.config.navigation.stable_linear_velocity,
                "angular_velocity_tolerance": self.config.navigation.stable_angular_velocity,
            }
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
            "execution_mode": execution_mode,
            "success_semantics": success_semantics,
            "pure_physics_success": pure_physics_success,
            "stable_physics_success": stable_physics_success,
            "physical_navigation_success": bool(
                success
                and (navigation_smoke or navigation_carry_smoke or integrated_apply_smoke or full_physics)
                and provenance_verified
            ),
            "carry_control_success": bool(
                success and (navigation_carry_smoke or integrated_apply_smoke or full_physics)
            ),
            "object_carry_verified": bool(success and full_physics),
            "physical_manipulation_success": bool(success and full_physics),
            "manipulation_apply_success": bool(
                success and (manipulation_apply_smoke or integrated_apply_smoke or full_physics)
            ),
            "integrated_control_success": bool(success and integrated_apply_smoke),
            "manipulation_base_lock_requested": bool(
                self.config.manipulation.lock_base_during_manipulation
            ),
            "manipulation_support_joint_lock_requested": bool(
                self.config.manipulation.lock_support_joints_during_manipulation
            ),
            "execution_provenance_verified": provenance_verified,
            "simulation_report": simulation_report,
            "navigation_acceptance": navigation_acceptance,
            **provenance,
            **machine_fields,
        }
