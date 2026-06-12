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
                }
                if decision.terminal:
                    break

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
        provenance = {
            "used_base_teleport": bool(final_state.metadata.get("used_base_teleport", False)),
            "used_direct_joint_state": bool(final_state.metadata.get("used_direct_joint_state", False)),
            "used_object_teleport": bool(final_state.metadata.get("used_object_teleport", False)),
            "used_kinematic_object_follow": bool(
                final_state.metadata.get("used_kinematic_object_follow", False)
            ),
            "used_visual_replay": bool(final_state.metadata.get("used_visual_replay", False)),
        }
        provenance_verified = bool(
            final_state.metadata.get("execution_provenance_verified", False)
        )
        pure_physics_success = (
            success
            and not dry_run
            and provenance_verified
            and not any(provenance.values())
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
            "execution_mode": "dry_run" if dry_run else "full_physics",
            "success_semantics": "control_flow_only" if dry_run else "physical_execution",
            "pure_physics_success": pure_physics_success,
            "execution_provenance_verified": provenance_verified,
            "debug_visualization_enabled": self.config.enable_debug_vis,
            "video_requested": self.config.save_video,
            **provenance,
            **machine_fields,
        }
