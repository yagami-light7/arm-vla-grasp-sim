"""Deterministic manipulation components for control-flow dry runs."""

from __future__ import annotations

from typing import Any

from source.interfaces import ArmPlan, EpisodeSpec, RobotAction, SimulationState


class DryRunGripperController:
    def command_open(self) -> str:
        return "open"

    def command_close(self) -> str:
        return "close"

    def command_hold(self) -> str:
        return "hold"


class DryRunManipulationPlanner:
    def plan_pick(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state, episode_spec
        return self._plan("pick")

    def plan_place(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state
        if episode_spec.place_target_pose is None:
            raise RuntimeError("place target pose is missing")
        plan = self._plan("place")
        return ArmPlan(
            operation=plan.operation,
            joint_trajectory=plan.joint_trajectory,
            metadata={
                **plan.metadata,
                "target_pose": episode_spec.place_target_pose,
            },
        )

    @staticmethod
    def _plan(operation: str) -> ArmPlan:
        trajectory = tuple(
            tuple(0.05 * point * (joint + 1) for joint in range(8))
            for point in range(4)
        )
        return ArmPlan(
            operation=operation,
            joint_trajectory=trajectory,
            metadata={"planner": "dry_run_joint_trajectory"},
        )


class DryRunArmExecutor:
    def __init__(self, gripper: DryRunGripperController):
        self.gripper = gripper
        self.plan: ArmPlan | None = None
        self.tick_index = 0

    def reset(self, plan: ArmPlan) -> None:
        self.plan = plan
        self.tick_index = 0

    def compute_action(self, state: SimulationState) -> RobotAction:
        del state
        if self.plan is None:
            raise RuntimeError("arm executor has no plan")
        trajectory_index = min(self.tick_index, len(self.plan.joint_trajectory) - 1)
        target = self.plan.joint_trajectory[trajectory_index]
        self.tick_index += 1
        final_tick = self.tick_index >= len(self.plan.joint_trajectory)
        metadata: dict[str, Any] = {
            "operation": self.plan.operation,
            "progress": self.tick_index / len(self.plan.joint_trajectory),
        }
        gripper_command = self.gripper.command_hold()
        if self.plan.operation == "pick" and self.tick_index == len(self.plan.joint_trajectory) - 1:
            gripper_command = self.gripper.command_close()
            metadata["event_marker"] = "gripper_close"
        if self.plan.operation == "pick" and final_tick:
            metadata["dry_run_effect"] = "pick_lifted"
            gripper_command = self.gripper.command_close()
        if self.plan.operation == "place" and final_tick:
            metadata["dry_run_effect"] = "place_completed"
            metadata["event_marker"] = "gripper_open"
            metadata["target_pose"] = self.plan.metadata.get("target_pose")
            gripper_command = self.gripper.command_open()
        return RobotAction(
            arm_joint_positions=target,
            gripper_command=gripper_command,
            source=f"dry_run_{self.plan.operation}",
            metadata=metadata,
        )

    def is_done(self, state: SimulationState) -> bool:
        del state
        return self.plan is not None and self.tick_index >= len(self.plan.joint_trajectory)

    def status(self) -> dict[str, Any]:
        return {
            "backend": "dry_run",
            "operation": self.plan.operation if self.plan else None,
            "tick_index": self.tick_index,
            "trajectory_points": len(self.plan.joint_trajectory) if self.plan else 0,
            "done": self.plan is not None and self.tick_index >= len(self.plan.joint_trajectory),
        }
