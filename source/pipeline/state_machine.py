"""Tick-driven state machine for full-physics nav-pick-place episodes."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from source.interfaces import (
    ArmExecutor,
    EpisodeRecorder,
    EpisodeSpec,
    EpisodeVerifier,
    GripperController,
    ManipulationPlanner,
    NavExecutor,
    NavPlanner,
    RobotAction,
    SimulationRuntime,
    SimulationState,
)

from .config import FullPhysicsConfig
from .events import PipelineEvent
from .states import PipelineState


@dataclass(frozen=True)
class TickDecision:
    state: PipelineState
    action: RobotAction
    events: tuple[PipelineEvent, ...] = ()
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FullPhysicsStateMachine:
    """Advance pipeline control flow without ever advancing simulation time."""

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
        self.nav_planner = nav_planner
        self.nav_executor = nav_executor
        self.manipulation_planner = manipulation_planner
        self.arm_executor = arm_executor
        self.gripper = gripper
        self.verifier = verifier
        self.recorder = recorder

        self.state = PipelineState.BUILD_STAGE
        self.state_ticks = 0
        self.total_ticks = 0
        self.failure_reason = ""
        self.failure_metadata: dict[str, Any] = {}
        self.state_trace = [self.state.value]
        self.latest_planner_result: dict[str, Any] = {}
        self.latest_executor_status: dict[str, Any] = {}
        self.export_result: dict[str, Any] = {}
        self._pending_events = [
            self._event("state_entered", 0),
            self._event("episode_start", 0, {"seed": episode_seed}),
        ]

    def tick(self, observation: SimulationState) -> TickDecision:
        state_before = self.state
        events = self._take_pending_events()
        if self.state.terminal:
            return TickDecision(
                state=state_before,
                action=RobotAction.idle(source=self.state.value),
                events=tuple(events),
                terminal=True,
            )
        if self.total_ticks >= self.config.limits.episode:
            events.extend(self._fail("episode_timeout", observation))
            return self._decision(state_before, events)
        if self.state_ticks >= self._state_limit(self.state):
            events.extend(self._fail(self._timeout_reason(self.state), observation))
            return self._decision(state_before, events)

        self.total_ticks += 1
        self.state_ticks += 1
        action = RobotAction.idle(source=self.state.value)
        try:
            action, emitted = self._handle_state(observation)
            events.extend(emitted)
        except Exception as exc:
            events.extend(
                self._fail(
                    self._exception_reason(self.state),
                    observation,
                    {
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        return TickDecision(
            state=state_before,
            action=action,
            events=tuple(events),
            terminal=self.state.terminal,
            metadata={"next_state": self.state.value},
        )

    def summary_fields(self) -> dict[str, Any]:
        return {
            "success": self.state == PipelineState.DONE,
            "failure_reason": self.failure_reason,
            "failure_metadata": self.failure_metadata,
            "final_state": self.state.value,
            "state_trace": list(self.state_trace),
            "latest_planner_result": dict(self.latest_planner_result),
            "latest_executor_status": dict(self.latest_executor_status),
            "lerobot_export": dict(self.export_result),
        }

    def _handle_state(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        handlers: dict[
            PipelineState,
            Callable[[SimulationState], tuple[RobotAction, list[PipelineEvent]]],
        ] = {
            PipelineState.BUILD_STAGE: self._build_stage,
            PipelineState.RESET_EPISODE: self._reset_episode,
            PipelineState.PLAN_NAV_TO_PICK: self._plan_nav_to_pick,
            PipelineState.EXEC_NAV_TO_PICK: self._exec_nav_to_pick,
            PipelineState.VERIFY_PICK_REACHABLE: self._verify_pick_reachable,
            PipelineState.PLAN_PICK: self._plan_pick,
            PipelineState.EXEC_PICK: self._exec_pick,
            PipelineState.VERIFY_PICK_SUCCESS: self._verify_pick_success,
            PipelineState.PLAN_NAV_TO_PLACE: self._plan_nav_to_place,
            PipelineState.EXEC_NAV_TO_PLACE: self._exec_nav_to_place,
            PipelineState.VERIFY_PLACE_REACHABLE: self._verify_place_reachable,
            PipelineState.PLAN_PLACE: self._plan_place,
            PipelineState.EXEC_PLACE: self._exec_place,
            PipelineState.VERIFY_PLACE_SUCCESS: self._verify_place_success,
            PipelineState.EXPORT_LEROBOT: self._export_lerobot,
            PipelineState.CLEANUP_EPISODE: self._cleanup_episode,
        }
        return handlers[self.state](observation)

    def _build_stage(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        self.simulation.build(self.episode_spec)
        events = [self._event("stage_built", observation.step_index)]
        events.extend(self._transition(PipelineState.RESET_EPISODE, observation.step_index))
        return RobotAction.idle(source="stage_build"), events

    def _reset_episode(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        self.simulation.reset(self.episode_spec, seed=self.episode_seed)
        events = self._transition(PipelineState.PLAN_NAV_TO_PICK, observation.step_index)
        return RobotAction.idle(source="episode_reset"), events

    def _plan_nav_to_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events = [self._event("nav_to_pick_start", observation.step_index)]
        plan = self.nav_planner.plan(observation, self.episode_spec.pick_goal)
        self.nav_executor.reset(plan)
        self.latest_planner_result = {
            "type": "navigation",
            "phase": "pick",
            "waypoint_count": len(plan.waypoints),
            **plan.metadata,
        }
        events.extend(self._transition(PipelineState.EXEC_NAV_TO_PICK, observation.step_index))
        return RobotAction.idle(source="nav_plan_pick"), events

    def _exec_nav_to_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        return self._execute_nav(observation, PipelineState.VERIFY_PICK_REACHABLE)

    def _verify_pick_reachable(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_pick_reachable(observation, self.episode_spec)
        if not result.success:
            return RobotAction.idle(source="verify_pick_reachable"), self._fail(
                result.failure_reason or "pick_target_unreachable",
                observation,
                result.metadata,
            )
        events = [self._event("nav_to_pick_success", observation.step_index, result.metadata)]
        events.extend(self._transition(PipelineState.PLAN_PICK, observation.step_index))
        return RobotAction.idle(source="verify_pick_reachable"), events

    def _plan_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events = [self._event("pick_plan_start", observation.step_index)]
        plan = self.manipulation_planner.plan_pick(observation, self.episode_spec)
        self.arm_executor.reset(plan)
        self.latest_planner_result = {
            "type": "manipulation",
            "phase": "pick",
            "trajectory_points": len(plan.joint_trajectory),
            **plan.metadata,
        }
        events.extend(self._transition(PipelineState.EXEC_PICK, observation.step_index))
        events.append(self._event("pick_execute_start", observation.step_index))
        return RobotAction.idle(source="pick_plan"), events

    def _exec_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        return self._execute_arm(observation, PipelineState.VERIFY_PICK_SUCCESS)

    def _verify_pick_success(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_pick_success(observation, self.episode_spec)
        if not result.success:
            return RobotAction.idle(source="verify_pick_success"), self._fail(
                result.failure_reason or "grasp_failed",
                observation,
                result.metadata,
            )
        events = [self._event("object_lift_success", observation.step_index, result.metadata)]
        events.extend(self._transition(PipelineState.PLAN_NAV_TO_PLACE, observation.step_index))
        return RobotAction.idle(source="verify_pick_success"), events

    def _plan_nav_to_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.episode_spec.place_goal is None:
            return RobotAction.idle(source="nav_plan_place"), self._fail(
                "place_target_unreachable",
                observation,
                {"detail": "task does not define an enabled place base goal"},
            )
        events = [self._event("nav_to_place_start", observation.step_index)]
        plan = self.nav_planner.plan(observation, self.episode_spec.place_goal)
        self.nav_executor.reset(plan)
        self.latest_planner_result = {
            "type": "navigation",
            "phase": "place",
            "waypoint_count": len(plan.waypoints),
            **plan.metadata,
        }
        events.extend(self._transition(PipelineState.EXEC_NAV_TO_PLACE, observation.step_index))
        return RobotAction.idle(source="nav_plan_place"), events

    def _exec_nav_to_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        return self._execute_nav(observation, PipelineState.VERIFY_PLACE_REACHABLE)

    def _verify_place_reachable(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_place_reachable(observation, self.episode_spec)
        if not result.success:
            return RobotAction.idle(source="verify_place_reachable"), self._fail(
                result.failure_reason or "place_target_unreachable",
                observation,
                result.metadata,
            )
        events = [self._event("nav_to_place_success", observation.step_index, result.metadata)]
        events.extend(self._transition(PipelineState.PLAN_PLACE, observation.step_index))
        return RobotAction.idle(source="verify_place_reachable"), events

    def _plan_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events = [self._event("place_plan_start", observation.step_index)]
        plan = self.manipulation_planner.plan_place(observation, self.episode_spec)
        self.arm_executor.reset(plan)
        self.latest_planner_result = {
            "type": "manipulation",
            "phase": "place",
            "trajectory_points": len(plan.joint_trajectory),
            **plan.metadata,
        }
        events.extend(self._transition(PipelineState.EXEC_PLACE, observation.step_index))
        events.append(self._event("place_execute_start", observation.step_index))
        return RobotAction.idle(source="place_plan"), events

    def _exec_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        return self._execute_arm(observation, PipelineState.VERIFY_PLACE_SUCCESS)

    def _verify_place_success(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_place_success(observation, self.episode_spec)
        if not result.success:
            return RobotAction.idle(source="verify_place_success"), self._fail(
                result.failure_reason or "object_out_of_place",
                observation,
                result.metadata,
            )
        events = [self._event("place_success", observation.step_index, result.metadata)]
        events.extend(self._transition(PipelineState.EXPORT_LEROBOT, observation.step_index))
        return RobotAction.idle(source="verify_place_success"), events

    def _export_lerobot(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        self.export_result = self.recorder.prepare_lerobot_export()
        events = [
            self._event(
                "episode_exported",
                observation.step_index,
                self.export_result,
            )
        ]
        events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
        return RobotAction.idle(source="export_lerobot"), events

    def _cleanup_episode(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events = [self._event("episode_success", observation.step_index)]
        events.extend(self._transition(PipelineState.DONE, observation.step_index))
        return RobotAction.idle(source="cleanup_episode"), events

    def _execute_nav(
        self,
        observation: SimulationState,
        next_state: PipelineState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.nav_executor.is_done(observation):
            self.latest_executor_status = self.nav_executor.status()
            return RobotAction.idle(source=self.state.value), self._transition(
                next_state,
                observation.step_index,
            )
        action = self.nav_executor.compute_action(observation)
        self.latest_executor_status = self.nav_executor.status()
        events: list[PipelineEvent] = []
        if self.nav_executor.is_done(observation):
            events.extend(self._transition(next_state, observation.step_index))
        return action, events

    def _execute_arm(
        self,
        observation: SimulationState,
        next_state: PipelineState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.arm_executor.is_done(observation):
            self.latest_executor_status = self.arm_executor.status()
            return RobotAction.idle(source=self.state.value), self._transition(
                next_state,
                observation.step_index,
            )
        action = self.arm_executor.compute_action(observation)
        self.latest_executor_status = self.arm_executor.status()
        events: list[PipelineEvent] = []
        marker = action.metadata.get("event_marker")
        if marker:
            events.append(self._event(str(marker), observation.step_index))
        if self.arm_executor.is_done(observation):
            events.extend(self._transition(next_state, observation.step_index))
        return action, events

    def _transition(self, next_state: PipelineState, step_index: int) -> list[PipelineEvent]:
        previous = self.state
        events = [
            PipelineEvent(
                name="state_completed",
                pipeline_state=previous.value,
                step_index=step_index,
                metadata={"next_state": next_state.value},
            )
        ]
        self.state = next_state
        self.state_ticks = 0
        self.state_trace.append(next_state.value)
        events.append(self._event("state_entered", step_index, {"previous_state": previous.value}))
        return events

    def _fail(
        self,
        reason: str,
        observation: SimulationState,
        metadata: dict[str, Any] | None = None,
    ) -> list[PipelineEvent]:
        failed_state = self.state
        self.failure_reason = reason
        self.failure_metadata = {
            "current_state": failed_state.value,
            "robot_pose": observation.robot_root_pose,
            "object_pose": observation.object_pose,
            "latest_planner_result": dict(self.latest_planner_result),
            "latest_executor_status": dict(self.latest_executor_status),
            **dict(metadata or {}),
        }
        self.state = PipelineState.FAILED
        self.state_ticks = 0
        self.state_trace.append(self.state.value)
        return [
            PipelineEvent(
                name="state_failed",
                pipeline_state=failed_state.value,
                step_index=observation.step_index,
                metadata={"failure_reason": reason, **self.failure_metadata},
            ),
            self._event("state_entered", observation.step_index, {"previous_state": failed_state.value}),
            self._event("episode_failed", observation.step_index, {"failure_reason": reason}),
        ]

    def _state_limit(self, state: PipelineState) -> int:
        limits = self.config.limits
        if state == PipelineState.BUILD_STAGE:
            return limits.build_stage
        if state == PipelineState.RESET_EPISODE:
            return limits.reset_episode
        if state in {
            PipelineState.PLAN_NAV_TO_PICK,
            PipelineState.PLAN_NAV_TO_PLACE,
            PipelineState.PLAN_PICK,
            PipelineState.PLAN_PLACE,
        }:
            return limits.planning
        if state in {PipelineState.EXEC_NAV_TO_PICK, PipelineState.EXEC_NAV_TO_PLACE}:
            return limits.navigation
        if state in {PipelineState.EXEC_PICK, PipelineState.EXEC_PLACE}:
            return limits.manipulation
        if state in {
            PipelineState.VERIFY_PICK_REACHABLE,
            PipelineState.VERIFY_PICK_SUCCESS,
            PipelineState.VERIFY_PLACE_REACHABLE,
            PipelineState.VERIFY_PLACE_SUCCESS,
        }:
            return limits.verification
        if state == PipelineState.EXPORT_LEROBOT:
            return limits.export
        return limits.cleanup

    @staticmethod
    def _timeout_reason(state: PipelineState) -> str:
        return {
            PipelineState.EXEC_NAV_TO_PICK: "nav_to_pick_timeout",
            PipelineState.EXEC_PICK: "pick_tracking_failed",
            PipelineState.EXEC_NAV_TO_PLACE: "nav_to_place_timeout",
            PipelineState.EXEC_PLACE: "place_tracking_failed",
        }.get(state, "episode_timeout")

    @staticmethod
    def _exception_reason(state: PipelineState) -> str:
        return {
            PipelineState.BUILD_STAGE: "stage_build_failed",
            PipelineState.PLAN_NAV_TO_PICK: "nav_to_pick_plan_failed",
            PipelineState.PLAN_PICK: "pick_plan_failed",
            PipelineState.PLAN_NAV_TO_PLACE: "nav_to_place_plan_failed",
            PipelineState.PLAN_PLACE: "place_plan_failed",
            PipelineState.EXPORT_LEROBOT: "episode_export_failed",
        }.get(state, f"{state.value}_failed")

    def _event(
        self,
        name: str,
        step_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineEvent:
        return PipelineEvent(
            name=name,
            pipeline_state=self.state.value,
            step_index=step_index,
            metadata=dict(metadata or {}),
        )

    def _take_pending_events(self) -> list[PipelineEvent]:
        events = self._pending_events
        self._pending_events = []
        return events

    def _decision(
        self,
        state_before: PipelineState,
        events: list[PipelineEvent],
    ) -> TickDecision:
        return TickDecision(
            state=state_before,
            action=RobotAction.idle(source=state_before.value),
            events=tuple(events),
            terminal=self.state.terminal,
            metadata={"next_state": self.state.value},
        )
