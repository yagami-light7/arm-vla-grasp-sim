"""Tick-driven state machine for full-physics nav-pick-place episodes."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable

from source.interfaces import (
    ArmExecutor,
    ArmPlan,
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


_CARRY_GRIPPER_HOLD_STATES = frozenset(
    {
        PipelineState.VERIFY_PICK_SUCCESS,
        PipelineState.PLAN_NAV_TO_PLACE,
        PipelineState.EXEC_NAV_TO_PLACE,
        PipelineState.VERIFY_PLACE_REACHABLE,
        PipelineState.PLAN_PLACE,
        PipelineState.EXEC_PLACE,
    }
)
_CARRY_ARM_HOME_HOLD_STATES = frozenset(
    {
        PipelineState.VERIFY_PICK_SUCCESS,
        PipelineState.PLAN_NAV_TO_PLACE,
        PipelineState.EXEC_NAV_TO_PLACE,
        PipelineState.VERIFY_PLACE_REACHABLE,
        PipelineState.PLAN_PLACE,
    }
)
_MANIPULATION_BASE_LOCK_STATES = frozenset(
    {
        PipelineState.PLAN_PICK,
        PipelineState.EXEC_PICK,
        PipelineState.VERIFY_PICK_SUCCESS,
        PipelineState.PLAN_PLACE,
        PipelineState.EXEC_PLACE,
        PipelineState.VERIFY_PLACE_SUCCESS,
    }
)

_TERMINAL_HOLD_STATES = frozenset(
    {
        PipelineState.CLEANUP_EPISODE,
        PipelineState.DONE,
    }
)


def _first_motion_target(plan: Any) -> dict[str, Any] | None:
    segments = plan.metadata.get("segments") if hasattr(plan, "metadata") else None
    if isinstance(segments, list | tuple):
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, dict) or segment.get("type") != "motion":
                continue
            trajectory = segment.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            q_rows = trajectory.get("q")
            if not isinstance(q_rows, list | tuple) or not q_rows:
                continue
            return {
                "q": tuple(float(value) for value in q_rows[0]),
                "metadata": {
                    "start_segment_index": segment_index,
                    "start_segment_name": str(segment.get("name") or "motion"),
                    "start_segment_type": "motion",
                    "start_waypoint_index": 0,
                    "start_state_source": "segments",
                },
            }
    if getattr(plan, "joint_trajectory", None):
        return {
            "q": tuple(float(value) for value in plan.joint_trajectory[0]),
            "metadata": {
                "start_segment_name": "flat_trajectory",
                "start_segment_type": "motion",
                "start_waypoint_index": 0,
                "start_state_source": "joint_trajectory",
            },
        }
    return None


def _last_motion_target(plan: Any) -> dict[str, Any] | None:
    segments = plan.metadata.get("segments") if hasattr(plan, "metadata") else None
    if isinstance(segments, list | tuple):
        for segment_index in range(len(segments) - 1, -1, -1):
            segment = segments[segment_index]
            if not isinstance(segment, dict) or segment.get("type") != "motion":
                continue
            trajectory = segment.get("trajectory")
            if not isinstance(trajectory, dict):
                continue
            q_rows = trajectory.get("q")
            if not isinstance(q_rows, list | tuple) or not q_rows:
                continue
            return {
                "q": tuple(float(value) for value in q_rows[-1]),
                "metadata": {
                    "end_segment_index": segment_index,
                    "end_segment_name": str(segment.get("name") or "motion"),
                    "end_segment_type": "motion",
                    "end_waypoint_index": len(q_rows) - 1,
                    "end_state_source": "segments",
                },
            }
    if getattr(plan, "joint_trajectory", None):
        return {
            "q": tuple(float(value) for value in plan.joint_trajectory[-1]),
            "metadata": {
                "end_segment_name": "flat_trajectory",
                "end_segment_type": "motion",
                "end_waypoint_index": len(plan.joint_trajectory) - 1,
                "end_state_source": "joint_trajectory",
            },
        }
    return None


def _max_abs_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max((abs(left - right) for left, right in zip(a, b)), default=0.0)


def _append_pick_return_home(
    plan: ArmPlan,
    *,
    duration_s: float,
    hold_duration_s: float,
    skip_tolerance: float,
) -> tuple[ArmPlan, dict[str, Any]]:
    joint_names = tuple(str(name) for name in plan.metadata.get("joint_names") or ())
    last_target = _last_motion_target(plan)
    if last_target is None or not joint_names:
        return (
            plan,
            {
                "inserted": False,
                "reason": "plan_missing_motion_or_joint_names",
            },
        )

    start_positions = tuple(float(value) for value in last_target["q"])
    home_positions = tuple(0.0 for _ in joint_names)
    max_abs_start_error = _max_abs_delta(start_positions, home_positions)
    if max_abs_start_error <= skip_tolerance:
        return (
            plan,
            {
                "inserted": False,
                "reason": "already_at_home",
                "joint_names": joint_names,
                "start_positions": start_positions,
                "home_positions": home_positions,
                "max_abs_start_error": max_abs_start_error,
                **last_target["metadata"],
            },
        )

    return_home_segment = {
        "name": "return_home_after_pick",
        "type": "motion",
        "target_name": "arm_home",
        "timing": {
            "duration_s": float(duration_s),
            "num_waypoints": 2,
            "generated_by": "full_physics_pipeline",
        },
        "final_error": {},
        "plan_info": {
            "generated_return_home": True,
            "reason": "baseline_pick_retreat_return_home_before_carry",
            "max_abs_start_error": max_abs_start_error,
        },
        "trajectory": {
            "time_from_start": (0.0, float(duration_s)),
            "q": (start_positions, home_positions),
        },
    }
    home_hold_segment = {
        "name": "hold_home_before_carry",
        "type": "motion",
        "target_name": "arm_home",
        "timing": {
            "duration_s": float(hold_duration_s),
            "num_waypoints": 2,
            "generated_by": "full_physics_pipeline",
        },
        "final_error": {},
        "plan_info": {
            "generated_home_hold": True,
            "reason": "baseline_carry_keeps_arm_at_home",
        },
        "trajectory": {
            "time_from_start": (0.0, float(hold_duration_s)),
            "q": (home_positions, home_positions),
        },
    }
    generated_segments = (
        (return_home_segment, home_hold_segment)
        if hold_duration_s > 0.0
        else (return_home_segment,)
    )
    raw_segments = plan.metadata.get("segments")
    if isinstance(raw_segments, list | tuple) and raw_segments:
        segments = (*tuple(raw_segments), *generated_segments)
    else:
        segments = (
            {
                "name": "original_flat_trajectory",
                "type": "motion",
                "trajectory": {
                    "time_from_start": tuple(
                        0.05 * index for index in range(len(plan.joint_trajectory))
                    ),
                    "q": tuple(plan.joint_trajectory),
                },
            },
            *generated_segments,
        )

    joint_trajectory = (
        *tuple(tuple(float(value) for value in row) for row in plan.joint_trajectory),
        start_positions,
        home_positions,
        *((home_positions,) if hold_duration_s > 0.0 else ()),
    )
    report = {
        "inserted": True,
        "segment_name": return_home_segment["name"],
        "duration_s": float(duration_s),
        "home_hold_inserted": hold_duration_s > 0.0,
        "home_hold_segment_name": home_hold_segment["name"] if hold_duration_s > 0.0 else None,
        "home_hold_duration_s": float(hold_duration_s),
        "joint_names": joint_names,
        "start_positions": start_positions,
        "home_positions": home_positions,
        "max_abs_start_error": max_abs_start_error,
        **last_target["metadata"],
    }
    return (
        ArmPlan(
            operation=plan.operation,
            joint_trajectory=joint_trajectory,
            metadata={
                **plan.metadata,
                "segments": segments,
                "joint_names": joint_names,
                "pick_return_home": report,
                "original_trajectory_points": len(plan.joint_trajectory),
            },
        ),
        report,
    )


def _append_place_return_home(
    plan: ArmPlan,
    *,
    duration_s: float,
    skip_tolerance: float,
) -> tuple[ArmPlan, dict[str, Any]]:
    """在 release/retreat 后追加回零轨迹，保持与 baseline 收尾一致。"""

    joint_names = tuple(str(name) for name in plan.metadata.get("joint_names") or ())
    last_target = _last_motion_target(plan)
    if last_target is None or not joint_names:
        return plan, {"inserted": False, "reason": "plan_missing_motion_or_joint_names"}

    start_positions = tuple(float(value) for value in last_target["q"])
    home_positions = tuple(0.0 for _ in joint_names)
    max_abs_start_error = _max_abs_delta(start_positions, home_positions)
    if max_abs_start_error <= skip_tolerance:
        return plan, {
            "inserted": False,
            "reason": "already_at_home",
            "joint_names": joint_names,
            "start_positions": start_positions,
            "home_positions": home_positions,
            "max_abs_start_error": max_abs_start_error,
            **last_target["metadata"],
        }

    return_home_segment = {
        "name": "return_home_after_place",
        "type": "motion",
        "target_name": "arm_home",
        "timing": {
            "duration_s": float(duration_s),
            "num_waypoints": 2,
            "generated_by": "full_physics_pipeline",
        },
        "final_error": {},
        "plan_info": {
            "generated_return_home": True,
            "reason": "baseline_place_return_home_after_release",
            "max_abs_start_error": max_abs_start_error,
        },
        "trajectory": {
            "time_from_start": (0.0, float(duration_s)),
            "q": (start_positions, home_positions),
        },
    }
    raw_segments = plan.metadata.get("segments")
    segments = (*tuple(raw_segments), return_home_segment) if raw_segments else (return_home_segment,)
    report = {
        "inserted": True,
        "segment_name": return_home_segment["name"],
        "duration_s": float(duration_s),
        "joint_names": joint_names,
        "start_positions": start_positions,
        "home_positions": home_positions,
        "max_abs_start_error": max_abs_start_error,
        **last_target["metadata"],
    }
    return (
        ArmPlan(
            operation=plan.operation,
            joint_trajectory=(
                *tuple(tuple(float(value) for value in row) for row in plan.joint_trajectory),
                start_positions,
                home_positions,
            ),
            metadata={
                **plan.metadata,
                "segments": segments,
                "joint_names": joint_names,
                "place_return_home": report,
                "original_trajectory_points": len(plan.joint_trajectory),
            },
        ),
        report,
    )


def _actual_arm_positions_for_plan(
    observation: SimulationState,
    joint_names: tuple[str, ...],
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    current_positions = tuple(float(value) for value in observation.joint_positions)
    runtime_joint_names = tuple(str(name) for name in observation.metadata.get("joint_names") or ())
    if runtime_joint_names:
        index_by_name = {name: index for index, name in enumerate(runtime_joint_names)}
        missing_names = [name for name in joint_names if name not in index_by_name]
        if missing_names:
            return None, {
                "mapping": "runtime_joint_names",
                "missing_joint_names": missing_names,
                "runtime_joint_names": runtime_joint_names,
            }
        indices = tuple(index_by_name[name] for name in joint_names)
        out_of_range = [index for index in indices if index < 0 or index >= len(current_positions)]
        if out_of_range:
            return None, {
                "mapping": "runtime_joint_names",
                "joint_indices": indices,
                "out_of_range_joint_indices": out_of_range,
                "current_joint_count": len(current_positions),
            }
        return tuple(current_positions[index] for index in indices), {
            "mapping": "runtime_joint_names",
            "joint_indices": indices,
        }

    if len(current_positions) == len(joint_names):
        return current_positions, {
            "mapping": "same_length_without_joint_names",
            "joint_indices": tuple(range(len(joint_names))),
        }
    return None, {
        "mapping": "unavailable",
        "current_joint_count": len(current_positions),
        "plan_joint_count": len(joint_names),
    }


def _prepend_place_start_transition(
    plan: ArmPlan,
    start_state_check: dict[str, Any],
    *,
    duration_s: float,
) -> tuple[ArmPlan, dict[str, Any]]:
    joint_names = tuple(str(name) for name in start_state_check["joint_names"])
    actual_positions = tuple(float(value) for value in start_state_check["actual_positions"])
    target_positions = tuple(float(value) for value in start_state_check["target_positions"])
    transition_segment = {
        "name": "start_state_transition_to_place_plan",
        "type": "motion",
        "target_name": "place_plan_start",
        "timing": {
            "duration_s": float(duration_s),
            "num_waypoints": 2,
            "generated_by": "full_physics_pipeline",
        },
        "final_error": {},
        "plan_info": {
            "generated_transition": True,
            "reason": "place_plan_start_state_mismatch",
            "max_abs_start_error": start_state_check.get("max_abs_error"),
            "peak_joint": dict(start_state_check.get("peak_joint") or {}),
        },
        "trajectory": {
            "time_from_start": (0.0, float(duration_s)),
            "q": (actual_positions, target_positions),
        },
    }
    raw_segments = plan.metadata.get("segments")
    if isinstance(raw_segments, list | tuple) and raw_segments:
        original_segments = tuple(raw_segments)
    else:
        original_segments = (
            {
                "name": "original_flat_trajectory",
                "type": "motion",
                "trajectory": {
                    "time_from_start": tuple(
                        0.05 * index for index in range(len(plan.joint_trajectory))
                    ),
                    "q": tuple(plan.joint_trajectory),
                },
            },
        )
    segments = (transition_segment, *original_segments)
    joint_trajectory = (
        actual_positions,
        target_positions,
        *tuple(tuple(float(value) for value in row) for row in plan.joint_trajectory),
    )
    report = {
        "inserted": True,
        "segment_name": transition_segment["name"],
        "duration_s": float(duration_s),
        "joint_names": joint_names,
        "start_positions": actual_positions,
        "target_positions": target_positions,
        "max_abs_start_error": start_state_check.get("max_abs_error"),
        "peak_joint": dict(start_state_check.get("peak_joint") or {}),
    }
    return (
        ArmPlan(
            operation=plan.operation,
            joint_trajectory=joint_trajectory,
            metadata={
                **plan.metadata,
                "segments": segments,
                "joint_names": joint_names,
                "place_start_transition": report,
                "original_trajectory_points": len(plan.joint_trajectory),
            },
        ),
        report,
    )


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
        self._carry_gripper_target: dict[str, Any] | None = None
        self._carry_arm_home_target: dict[str, Any] | None = None
        self._carry_object_tcp_offset: tuple[float, float, float] | None = None
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
        action = self._with_carry_gripper_hold(action, state_before)
        action = self._with_carry_arm_home_hold(action, state_before)
        action = self._with_manipulation_base_lock(action, state_before)
        if self.state.terminal and self.config.full_physics:
            # 终止帧也必须继续施加稳定目标；状态切到 FAILED/DONE 后撤锁一帧
            # 就足以让机器狗在 GUI 保留窗口中失稳。
            action = self._with_terminal_hold(action)
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
            "carry_gripper_target": dict(self._carry_gripper_target or {}),
            "carry_arm_home_target": dict(self._carry_arm_home_target or {}),
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
        metadata = {}
        if self.config.full_physics:
            # Stage 创建后必须先执行 episode reset，再允许首个物理步。
            # 否则动态苹果会在 reset/sleep 之前因重力和接触产生角速度。
            metadata = {
                "skip_physics_step": True,
                "skip_reason": "await_object_pose_reset_before_first_physics_step",
            }
        return RobotAction(source="stage_build", metadata=metadata), events

    def _reset_episode(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        self.simulation.reset(self.episode_spec, seed=self.episode_seed)
        events = [self._event("episode_reset", observation.step_index)]
        if self.config.full_physics:
            reset_state = self.simulation.read()
            pose_check = dict(reset_state.metadata.get("object_pose_debug_after_reset") or {})
            if pose_check.get("available") is not True or pose_check.get("within_tolerance") is not True:
                return RobotAction.idle(source="episode_reset"), self._fail(
                    "object_initial_pose_mismatch",
                    reset_state,
                    pose_check,
                )
        if self.config.simulation_smoke:
            events.append(self._event("simulation_smoke_success", observation.step_index))
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            metadata = {}
            if self.config.keep_window_open:
                # GUI 检查随机化时停在刚 reset/apply pose 的画面，不再让物理推进改变苹果姿态。
                metadata["skip_physics_step"] = True
                metadata["skip_reason"] = "simulation_smoke_gui_pose_inspection"
            return RobotAction(
                source="episode_reset",
                metadata=metadata,
            ), events
        if self._manipulation_only_smoke_enabled():
            events.append(self._event(self._manipulation_smoke_event("start"), observation.step_index))
            events.extend(self._transition(PipelineState.PLAN_PICK, observation.step_index))
            return RobotAction.idle(source="episode_reset"), events
        if self.config.navigation_carry_smoke:
            if self.episode_spec.place_goal is None:
                return RobotAction.idle(source="episode_reset"), self._fail(
                    "place_target_unreachable",
                    observation,
                    {"detail": "navigation carry smoke requires place base goal"},
                )
            self._initialize_navigation_carry_smoke_targets()
            events.append(self._event("navigation_carry_smoke_start", observation.step_index))
            events.extend(self._transition(PipelineState.PLAN_NAV_TO_PLACE, observation.step_index))
            return RobotAction.idle(source="episode_reset"), events
        events.extend(self._transition(PipelineState.PLAN_NAV_TO_PICK, observation.step_index))
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
        if self.config.navigation_smoke:
            events.append(self._event("navigation_smoke_success", observation.step_index))
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            return RobotAction.idle(source="verify_pick_reachable"), events
        events.extend(self._transition(PipelineState.PLAN_PICK, observation.step_index))
        return RobotAction.idle(source="verify_pick_reachable"), events

    def _plan_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        settle_result = self._settle_locked_base_before_plan(observation, phase="pick")
        if settle_result is not None:
            return settle_result

        events = [self._event("pick_plan_start", observation.step_index)]
        if self.config.full_physics:
            prepare_report = self.simulation.prepare_object_for_pick(self.episode_spec)
            events.append(
                self._event(
                    "object_prepared_for_pick",
                    observation.step_index,
                    prepare_report,
                )
            )
            if prepare_report.get("applied") is not True:
                return RobotAction.idle(source="pick_plan"), self._fail(
                    "object_initial_pose_mismatch",
                    observation,
                    prepare_report,
                )
            observation = self.simulation.read()
        try:
            plan = self.manipulation_planner.plan_pick(observation, self.episode_spec)
        except Exception as exc:
            return RobotAction.idle(source="pick_plan"), self._fail(
                "pick_plan_failed",
                observation,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        return_home_report = {"inserted": False, "reason": "disabled"}
        if self.config.manipulation.return_home_after_pick:
            plan, return_home_report = _append_pick_return_home(
                plan,
                duration_s=self.config.manipulation.pick_return_home_duration_s,
                hold_duration_s=self.config.manipulation.pick_home_hold_duration_s,
                skip_tolerance=self.config.manipulation.pick_return_home_skip_tolerance,
            )
        self._configure_carry_arm_home_target(plan, return_home_report)
        self.arm_executor.reset(plan)
        self.latest_planner_result = {
            "type": "manipulation",
            "phase": "pick",
            "trajectory_points": len(plan.joint_trajectory),
            "pick_return_home": return_home_report,
            **plan.metadata,
        }
        if return_home_report.get("inserted"):
            events.append(
                self._event(
                    "pick_return_home_appended",
                    observation.step_index,
                    return_home_report,
                )
            )
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
        pick_success_event = (
            "pick_success"
            if self.config.full_physics
            else (
                "integrated_apply_smoke_pick_apply_success"
                if self.config.integrated_apply_smoke
                else (
                    "manipulation_apply_smoke_pick_apply_success"
                    if self.config.manipulation_apply_smoke
                    else "object_lift_success"
                )
            )
        )
        events = [self._event(pick_success_event, observation.step_index, result.metadata)]
        self._capture_carry_object_tcp_offset(observation)
        if self._manipulation_only_smoke_enabled():
            if self.episode_spec.place_target_pose is None:
                return RobotAction.idle(source="verify_pick_success"), self._fail(
                    "place_target_unreachable",
                    observation,
                    {"detail": "manipulation smoke requires place target pose"},
                )
            events.append(
                self._event(
                    self._manipulation_smoke_event("pick_success"),
                    observation.step_index,
                )
            )
            events.extend(self._transition(PipelineState.PLAN_PLACE, observation.step_index))
            return RobotAction.idle(source="verify_pick_success"), events
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
        if self.config.navigation_carry_smoke or self.config.integrated_apply_smoke or self.config.full_physics:
            if self.config.integrated_apply_smoke or self.config.full_physics:
                object_carry_check = self._verify_carry_object_tracking(observation)
                if not object_carry_check["success"]:
                    return RobotAction.idle(source="verify_place_reachable"), self._fail(
                        str(object_carry_check["failure_reason"]),
                        observation,
                        object_carry_check,
                    )
            carry_check = self._verify_navigation_carry_targets(
                observation,
                latest_sample_only=self.config.integrated_apply_smoke or self.config.full_physics,
            )
            if not carry_check["success"]:
                return RobotAction.idle(source="verify_place_reachable"), self._fail(
                    str(carry_check["failure_reason"]),
                    observation,
                    carry_check,
                )
            if self.config.integrated_apply_smoke or self.config.full_physics:
                events.append(
                    self._event(
                        "carry_control_success" if self.config.full_physics else "integrated_apply_smoke_carry_control_success",
                        observation.step_index,
                        carry_check,
                    )
                )
                events.extend(self._transition(PipelineState.PLAN_PLACE, observation.step_index))
                return RobotAction.idle(source="verify_place_reachable"), events
            events.append(
                self._event(
                    "navigation_carry_smoke_success",
                    observation.step_index,
                    carry_check,
                )
            )
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            return RobotAction.idle(source="verify_place_reachable"), events
        events.extend(self._transition(PipelineState.PLAN_PLACE, observation.step_index))
        return RobotAction.idle(source="verify_place_reachable"), events

    def _plan_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.config.integrated_apply_smoke or self.config.full_physics:
            object_carry_check = self._verify_carry_object_tracking(observation)
            if not object_carry_check["success"]:
                return RobotAction.idle(source="plan_place"), self._fail(
                    str(object_carry_check["failure_reason"]),
                    observation,
                    object_carry_check,
                )
        settle_result = self._settle_locked_base_before_plan(observation, phase="place")
        if settle_result is not None:
            return settle_result

        events = [self._event("place_plan_start", observation.step_index)]
        try:
            plan = self.manipulation_planner.plan_place(observation, self.episode_spec)
        except Exception as exc:
            return RobotAction.idle(source="place_plan"), self._fail(
                "place_plan_failed",
                observation,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        start_state_check = self._diagnose_arm_plan_start_state(
            observation,
            plan,
            phase="place",
        )
        self.latest_planner_result = {
            "type": "manipulation",
            "phase": "place",
            "trajectory_points": len(plan.joint_trajectory),
            "start_state_check": start_state_check,
            **plan.metadata,
        }
        if start_state_check.get("mismatch_detected"):
            events.append(
                self._event(
                    "place_plan_start_state_mismatch",
                    observation.step_index,
                    start_state_check,
                )
            )
            if self.config.manipulation.fail_on_place_plan_start_state_mismatch:
                return RobotAction.idle(source="place_plan"), self._fail(
                    "place_plan_start_state_mismatch",
                    observation,
                    start_state_check,
                )
        transition_report = {"inserted": False}
        if (
            start_state_check.get("warning_detected")
            and self.config.manipulation.insert_place_plan_start_transition
        ):
            plan, transition_report = _prepend_place_start_transition(
                plan,
                start_state_check,
                duration_s=self.config.manipulation.place_plan_start_transition_duration_s,
            )
            self.latest_planner_result = {
                "type": "manipulation",
                "phase": "place",
                "trajectory_points": len(plan.joint_trajectory),
                "start_state_check": start_state_check,
                "start_state_transition": transition_report,
                **plan.metadata,
            }
            events.append(
                self._event(
                    "place_plan_start_state_transition_inserted",
                    observation.step_index,
                    transition_report,
                )
            )
        else:
            self.latest_planner_result = {
                "type": "manipulation",
                "phase": "place",
                "trajectory_points": len(plan.joint_trajectory),
                "start_state_check": start_state_check,
                "start_state_transition": transition_report,
                **plan.metadata,
            }
        place_return_home_report = {"inserted": False, "reason": "disabled"}
        if self.config.manipulation.return_home_after_place:
            plan, place_return_home_report = _append_place_return_home(
                plan,
                duration_s=self.config.manipulation.place_return_home_duration_s,
                skip_tolerance=self.config.manipulation.place_return_home_skip_tolerance,
            )
            self.latest_planner_result = {
                **self.latest_planner_result,
                "trajectory_points": len(plan.joint_trajectory),
                "place_return_home": place_return_home_report,
                **plan.metadata,
            }
            if place_return_home_report.get("inserted"):
                events.append(
                    self._event(
                        "place_return_home_appended",
                        observation.step_index,
                        place_return_home_report,
                    )
                )
        self.arm_executor.reset(plan)
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
        place_success_event = (
            "place_success"
            if self.config.full_physics
            else (
                "integrated_apply_smoke_place_apply_success"
                if self.config.integrated_apply_smoke
                else (
                    "manipulation_apply_smoke_place_apply_success"
                    if self.config.manipulation_apply_smoke
                    else "place_success"
                )
            )
        )
        events = [self._event(place_success_event, observation.step_index, result.metadata)]
        if self._manipulation_only_smoke_enabled():
            events.append(
                self._event(self._manipulation_smoke_event("success"), observation.step_index)
            )
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            return RobotAction.idle(source="verify_place_success"), events
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
        metadata = {}
        if self.config.simulation_smoke and self.config.keep_window_open:
            metadata["skip_physics_step"] = True
            metadata["skip_reason"] = "simulation_smoke_gui_pose_inspection"
        if self.config.full_physics:
            metadata.update(
                {
                    "terminal_hold": True,
                    "terminal_hold_phase": "cleanup_episode",
                    "terminal_hold_reason": "keep_robot_stable_after_episode_success",
                }
            )
        action = RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source="cleanup_episode",
            metadata=metadata,
        )
        if self.config.full_physics:
            action = self._with_terminal_hold(action)
        return action, events

    def _execute_nav(
        self,
        observation: SimulationState,
        next_state: PipelineState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        if observation.metadata.get("environment_terminated"):
            return RobotAction.idle(source=self.state.value), self._fail(
                "navigation_environment_terminated",
                observation,
                {"runtime_metadata": dict(observation.metadata)},
            )
        executor_status = self.nav_executor.status()
        if executor_status.get("failed"):
            return RobotAction.idle(source=self.state.value), self._fail(
                str(executor_status.get("failure_reason") or "nav_tracking_failed"),
                observation,
                executor_status,
            )
        if self.nav_executor.is_done(observation):
            self.latest_executor_status = self.nav_executor.status()
            return RobotAction.idle(source=self.state.value), self._transition(
                next_state,
                observation.step_index,
            )
        action = self.nav_executor.compute_action(observation)
        self.latest_executor_status = self.nav_executor.status()
        if self.latest_executor_status.get("failed"):
            return RobotAction.idle(source=self.state.value), self._fail(
                str(
                    self.latest_executor_status.get("failure_reason")
                    or "nav_tracking_failed"
                ),
                observation,
                self.latest_executor_status,
            )
        events: list[PipelineEvent] = []
        if self.nav_executor.is_done(observation):
            events.extend(self._transition(next_state, observation.step_index))
        return action, events

    def _execute_arm(
        self,
        observation: SimulationState,
        next_state: PipelineState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        executor_status = self.arm_executor.status()
        if executor_status.get("failed"):
            return RobotAction.idle(source=self.state.value), self._fail(
                str(executor_status.get("failure_reason") or "arm_execution_failed"),
                observation,
                executor_status,
            )
        if self.arm_executor.is_done(observation):
            self.latest_executor_status = executor_status
            return RobotAction.idle(source=self.state.value), self._transition(
                next_state,
                observation.step_index,
            )
        action = self.arm_executor.compute_action(observation)
        self._remember_carry_gripper_target(action, observation)
        self.latest_executor_status = self.arm_executor.status()
        if self.latest_executor_status.get("failed"):
            return RobotAction.idle(source=self.state.value), self._fail(
                str(
                    self.latest_executor_status.get("failure_reason")
                    or "arm_execution_failed"
                ),
                observation,
                self.latest_executor_status,
            )
        events: list[PipelineEvent] = []
        marker = action.metadata.get("event_marker")
        if marker:
            events.append(self._event(str(marker), observation.step_index))
        if self.arm_executor.is_done(observation):
            events.extend(self._transition(next_state, observation.step_index))
        return action, events

    def _remember_carry_gripper_target(
        self,
        action: RobotAction,
        observation: SimulationState,
    ) -> None:
        del observation
        command = action.gripper_command
        marker = str(action.metadata.get("event_marker") or "")
        if command == self.gripper.command_open() or marker == "gripper_open":
            self._carry_gripper_target = None
            # pick 开头会先打开夹爪，不能因此清除已由 pick plan 建立的 carry home 目标。
            # 机械臂目标是否生效由 carry 状态集合控制，与夹爪 release 生命周期解耦。
            return
        if command != self.gripper.command_close() and marker != "gripper_close":
            return

        target: dict[str, Any] = {
            "gripper_command": self.gripper.command_close(),
            "source": action.source,
            "captured_state": self.state.value,
        }
        joint_names = _metadata_tuple(action.metadata, "gripper_joint_names")
        joint_positions = _metadata_tuple(action.metadata, "gripper_joint_positions")
        if joint_names is not None:
            target["gripper_joint_names"] = joint_names
        if joint_positions is not None:
            target["gripper_joint_positions"] = joint_positions
            # baseline 的导航 carry 会强制保持 close target，而不是被动冻结实际开度。
            # 实际开度的 position error 接近 0，无法持续提供夹紧力，起步时容易松开。
            target["hold_position_source"] = "forced_close_target_for_carry"
            target["commanded_close_positions"] = joint_positions
        if "operation" in action.metadata:
            target["operation"] = action.metadata["operation"]
        if "segment_name" in action.metadata:
            target["segment_name"] = action.metadata["segment_name"]
        self._carry_gripper_target = target

    def _capture_carry_object_tcp_offset(self, observation: SimulationState) -> None:
        """记录 pick 完成时的物体-TCP 相对位置，仅用于后续只读掉落检测。"""

        if observation.object_pose is None or observation.tcp_pose is None:
            self._carry_object_tcp_offset = None
            return
        self._carry_object_tcp_offset = tuple(
            float(object_value) - float(tcp_value)
            for object_value, tcp_value in zip(
                observation.object_pose[:3],
                observation.tcp_pose[:3],
            )
        )

    def _verify_carry_object_tracking(
        self,
        observation: SimulationState,
    ) -> dict[str, Any]:
        """检查真实物理 carry 是否滑移；本函数不施加任何物体控制。"""

        reference = self._carry_object_tcp_offset
        if reference is None:
            return {
                "success": False,
                "failure_reason": "object_dropped_after_pick",
                "detail": "pick success 时缺少 object/TCP pose，无法验证物理 carry。",
            }
        if observation.object_pose is None or observation.tcp_pose is None:
            return {
                "success": False,
                "failure_reason": "object_dropped_after_pick",
                "detail": "carry 阶段缺少 object/TCP pose。",
            }
        current = tuple(
            float(object_value) - float(tcp_value)
            for object_value, tcp_value in zip(
                observation.object_pose[:3],
                observation.tcp_pose[:3],
            )
        )
        drift = math.sqrt(
            sum(
                (current_value - reference_value) ** 2
                for current_value, reference_value in zip(current, reference)
            )
        )
        tolerance = self.config.manipulation.carry_object_tcp_slip_tolerance
        return {
            "success": drift <= tolerance,
            "failure_reason": "" if drift <= tolerance else "object_dropped_after_pick",
            "reference_object_tcp_offset_xyz": reference,
            "current_object_tcp_offset_xyz": current,
            "object_tcp_offset_drift_m": drift,
            "carry_object_tcp_slip_tolerance_m": tolerance,
            "object_pose": observation.object_pose,
            "tcp_pose": observation.tcp_pose,
            "read_only_check": True,
            "object_pose_modified": False,
        }

    def _initialize_navigation_carry_smoke_targets(self) -> None:
        """模拟 baseline carry 控制目标，不声明物体已被真实抓取。"""

        arm_joint_names = (
            "arm_joint1",
            "arm_joint2",
            "arm_joint3",
            "arm_joint4",
            "arm_joint5",
            "arm_joint6",
        )
        self._carry_arm_home_target = {
            "arm_joint_names": arm_joint_names,
            "arm_joint_positions": (0.0,) * len(arm_joint_names),
            "source": "navigation_carry_smoke",
            "return_home_inserted": False,
            "return_home_reason": "carry_smoke_starts_from_pick_goal_home_posture",
        }
        self._carry_gripper_target = {
            "gripper_command": self.gripper.command_close(),
            "gripper_joint_names": ("arm_joint7", "arm_joint8"),
            "gripper_joint_positions": (0.0, 0.0),
            "source": "navigation_carry_smoke",
            "captured_state": self.state.value,
        }

    def _verify_navigation_carry_targets(
        self,
        observation: SimulationState,
        *,
        latest_sample_only: bool = False,
    ) -> dict[str, Any]:
        if latest_sample_only:
            tracking_report = dict(
                observation.metadata.get("last_arm_tracking_report") or {}
            )
            sample_count = 1 if tracking_report.get("available") is True else 0
        else:
            tracking_report = dict(observation.metadata.get("arm_tracking_report") or {})
            sample_count = int(tracking_report.get("sample_count") or 0)
        max_abs_error = tracking_report.get("max_abs_error")
        tolerance = self.config.manipulation.carry_home_tracking_tolerance
        expected_phase = PipelineState.EXEC_NAV_TO_PLACE.value
        if latest_sample_only and tracking_report.get("pipeline_state") != expected_phase:
            return {
                "success": False,
                "failure_reason": "carry_arm_tracking_unavailable",
                "arm_tracking_report": tracking_report,
                "expected_pipeline_state": expected_phase,
                "carry_home_tracking_tolerance": tolerance,
            }
        if sample_count <= 0 or max_abs_error is None:
            return {
                "success": False,
                "failure_reason": "carry_arm_tracking_unavailable",
                "arm_tracking_report": tracking_report,
                "carry_home_tracking_tolerance": tolerance,
            }
        if float(max_abs_error) > tolerance:
            return {
                "success": False,
                "failure_reason": "carry_arm_home_tracking_failed",
                "arm_tracking_report": tracking_report,
                "carry_home_tracking_tolerance": tolerance,
            }

        gripper_report = dict(observation.metadata.get("last_gripper_action_report") or {})
        gripper_positions = tuple(
            float(value) for value in gripper_report.get("gripper_joint_positions") or ()
        )
        expected_gripper_positions = tuple(
            float(value)
            for value in (self._carry_gripper_target or {}).get(
                "gripper_joint_positions",
                (),
            )
        )
        gripper_held = (
            gripper_report.get("target_staged") is True
            and gripper_report.get("gripper_command") == self.gripper.command_close()
            and bool(expected_gripper_positions)
            and gripper_positions == expected_gripper_positions
        )
        if not gripper_held:
            return {
                "success": False,
                "failure_reason": "carry_gripper_hold_failed",
                "arm_tracking_report": tracking_report,
                "last_gripper_action_report": gripper_report,
                "expected_gripper_joint_positions": expected_gripper_positions,
                "carry_home_tracking_tolerance": tolerance,
            }
        return {
            "success": True,
            "failure_reason": "",
            "arm_tracking_report": tracking_report,
            "last_gripper_action_report": gripper_report,
            "expected_gripper_joint_positions": expected_gripper_positions,
            "carry_home_tracking_tolerance": tolerance,
            "tracking_scope": "latest_carry_sample" if latest_sample_only else "episode_aggregate",
        }

    def _configure_carry_arm_home_target(
        self,
        plan: ArmPlan,
        return_home_report: dict[str, Any],
    ) -> None:
        if not self.config.manipulation.hold_arm_home_during_carry:
            self._carry_arm_home_target = None
            return

        joint_names = tuple(str(name) for name in return_home_report.get("joint_names") or ())
        if not joint_names:
            joint_names = tuple(str(name) for name in plan.metadata.get("joint_names") or ())
        if not joint_names:
            self._carry_arm_home_target = None
            return

        if not self.config.manipulation.return_home_after_pick:
            last_target = _last_motion_target(plan)
            if last_target is None:
                self._carry_arm_home_target = None
                return
            carry_positions = tuple(float(value) for value in last_target["q"])
            if len(carry_positions) != len(joint_names):
                self._carry_arm_home_target = None
                return
            self._carry_arm_home_target = {
                "arm_joint_names": joint_names,
                "arm_joint_positions": carry_positions,
                "source": "pick_final_arm_pose",
                "return_home_inserted": False,
                "return_home_reason": return_home_report.get("reason"),
                # full_physics 需要真实 carry，因此保持撤离后的抓取姿态，
                # 不把机械臂拉回 home 破坏苹果与夹爪的接触。
                "carry_hold_policy": "hold_side_retreat_final_pose",
                **last_target["metadata"],
            }
            return

        home_positions = tuple(float(value) for value in return_home_report.get("home_positions") or ())
        if not home_positions:
            home_positions = tuple(0.0 for _ in joint_names)
        if len(home_positions) != len(joint_names):
            self._carry_arm_home_target = None
            return

        self._carry_arm_home_target = {
            "arm_joint_names": joint_names,
            "arm_joint_positions": home_positions,
            "source": "pick_return_home",
            "return_home_inserted": bool(return_home_report.get("inserted")),
            "return_home_reason": return_home_report.get("reason"),
        }

    def _with_carry_gripper_hold(
        self,
        action: RobotAction,
        state_before: PipelineState,
    ) -> RobotAction:
        if self.state == PipelineState.FAILED:
            return action
        if state_before not in _CARRY_GRIPPER_HOLD_STATES:
            return action
        if self._carry_gripper_target is None:
            return action
        if action.gripper_command == self.gripper.command_open():
            return action
        if action.metadata.get("event_marker") == "gripper_open":
            return action

        metadata = dict(action.metadata)
        metadata.update(
            {
                "carry_gripper_hold": True,
                "carry_gripper_phase": state_before.value,
                "carry_gripper_source": self._carry_gripper_target.get("source"),
                "carry_gripper_hold_position_source": self._carry_gripper_target.get(
                    "hold_position_source"
                ),
                "carry_gripper_commanded_close_positions": self._carry_gripper_target.get(
                    "commanded_close_positions"
                ),
            }
        )
        for key in ("gripper_joint_names", "gripper_joint_positions"):
            if key in self._carry_gripper_target and key not in metadata:
                metadata[key] = self._carry_gripper_target[key]
        if "gripper_joint_positions" not in metadata:
            # dry-run 没有真实夹爪 joint target；这里仍显式传 close，避免导航阶段语义丢失。
            metadata["carry_gripper_fallback"] = "close_command_without_explicit_joint_target"
        return replace(
            action,
            gripper_command=str(
                self._carry_gripper_target.get(
                    "gripper_command",
                    self.gripper.command_close(),
                )
            ),
            metadata=metadata,
        )

    def _with_carry_arm_home_hold(
        self,
        action: RobotAction,
        state_before: PipelineState,
    ) -> RobotAction:
        if self.state == PipelineState.FAILED:
            return action
        if state_before not in _CARRY_ARM_HOME_HOLD_STATES:
            return action
        if self._carry_arm_home_target is None:
            return action
        if action.arm_joint_positions is not None:
            return action
        if action.gripper_command == self.gripper.command_open():
            return action
        if action.metadata.get("event_marker") == "gripper_open":
            return action

        metadata = dict(action.metadata)
        metadata.update(
            {
                "carry_arm_home_hold": True,
                "carry_arm_home_phase": state_before.value,
                "carry_arm_home_source": self._carry_arm_home_target.get("source"),
                "arm_joint_names": self._carry_arm_home_target["arm_joint_names"],
            }
        )
        # baseline carry 阶段维持 0 位 home；这里仍只发 position target，不直接写 joint state。
        return replace(
            action,
            arm_joint_positions=tuple(self._carry_arm_home_target["arm_joint_positions"]),
            metadata=metadata,
        )

    def _with_manipulation_base_lock(
        self,
        action: RobotAction,
        state_before: PipelineState,
    ) -> RobotAction:
        """覆盖完整 manipulation 交接阶段，导航状态必须立即释放 root lock。"""

        lock_phase = (
            state_before
            if state_before in _MANIPULATION_BASE_LOCK_STATES
            else (
                self.state
                if self.state in _MANIPULATION_BASE_LOCK_STATES
                else None
            )
        )
        if (
            lock_phase is None
            and self.config.full_physics
            and action.metadata.get("terminal_hold") is True
        ):
            lock_phase = PipelineState.CLEANUP_EPISODE
        requested = bool(
            self.config.manipulation.lock_base_during_manipulation
            and lock_phase is not None
            and self.state != PipelineState.FAILED
        )
        support_requested = bool(
            self.config.manipulation.lock_support_joints_during_manipulation
            and lock_phase is not None
            and self.state != PipelineState.FAILED
        )
        metadata = dict(action.metadata)
        metadata.update(
            {
                "manipulation_base_lock": requested,
                "manipulation_base_lock_phase": lock_phase.value if requested else None,
                "manipulation_support_joint_lock": support_requested,
                "manipulation_support_joint_lock_phase": (
                    lock_phase.value if support_requested else None
                ),
            }
        )
        hold_base = requested or support_requested
        return replace(
            action,
            base_velocity=(0.0, 0.0, 0.0) if hold_base else action.base_velocity,
            metadata=metadata,
        )

    def _with_terminal_hold(self, action: RobotAction) -> RobotAction:
        """episode 成功后保持最后控制目标，避免 GUI 收尾阶段释放控制导致倒地。"""

        metadata = dict(action.metadata)
        metadata.update(
            {
                "manipulation_base_lock": bool(
                    self.config.manipulation.lock_base_during_manipulation
                ),
                "manipulation_base_lock_phase": "terminal_hold",
                "manipulation_support_joint_lock": bool(
                    self.config.manipulation.lock_support_joints_during_manipulation
                ),
                "manipulation_support_joint_lock_phase": "terminal_hold",
                "terminal_hold": True,
            }
        )
        return replace(
            action,
            gripper_command=self.gripper.command_hold(),
            metadata=metadata,
        )

    def _settle_locked_base_before_plan(
        self,
        observation: SimulationState,
        *,
        phase: str,
    ) -> tuple[RobotAction, list[PipelineEvent]] | None:
        """在规划和机械臂动作前留出明确停驻窗口，避免与导航视觉上连成一段。"""

        settle_steps = (
            self.config.manipulation.base_lock_settle_steps
            if self.config.manipulation.lock_base_during_manipulation
            else 0
        )
        if settle_steps <= 0 or self.state_ticks > settle_steps:
            return None

        events: list[PipelineEvent] = []
        if self.state_ticks == 1:
            events.append(
                self._event(
                    f"{phase}_base_settle_start",
                    observation.step_index,
                    {"settle_steps": settle_steps},
                )
            )
        if self.state_ticks == settle_steps:
            events.append(
                self._event(
                    f"{phase}_base_settle_complete",
                    observation.step_index,
                    {"settle_steps": settle_steps},
                )
            )
        return (
            RobotAction(
                source=f"{phase}_base_settle",
                metadata={
                    "manipulation_base_settle": True,
                    "manipulation_base_settle_phase": phase,
                    "manipulation_base_settle_step": self.state_ticks,
                    "manipulation_base_settle_steps": settle_steps,
                },
            ),
            events,
        )

    def _diagnose_arm_plan_start_state(
        self,
        observation: SimulationState,
        plan: Any,
        *,
        phase: str,
    ) -> dict[str, Any]:
        start_target = _first_motion_target(plan)
        if start_target is None:
            return {
                "available": False,
                "reason": "plan_has_no_motion_target",
                "phase": phase,
            }
        joint_names = tuple(str(name) for name in plan.metadata.get("joint_names") or ())
        if not joint_names:
            return {
                "available": False,
                "reason": "plan_missing_joint_names",
                "phase": phase,
                **start_target["metadata"],
            }
        actual_positions, mapping_report = _actual_arm_positions_for_plan(
            observation,
            joint_names,
        )
        if actual_positions is None:
            return {
                "available": False,
                "reason": "current_joint_state_unavailable",
                "phase": phase,
                "joint_names": joint_names,
                **mapping_report,
                **start_target["metadata"],
            }

        targets = tuple(float(value) for value in start_target["q"])
        errors = tuple(actual - target for actual, target in zip(actual_positions, targets))
        abs_errors = tuple(abs(value) for value in errors)
        max_abs_error = max(abs_errors) if abs_errors else 0.0
        mean_abs_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        peak_index = max(range(len(abs_errors)), key=lambda index: abs_errors[index]) if abs_errors else 0
        warning_threshold = self.config.manipulation.plan_start_state_warning_threshold
        failure_threshold = self.config.manipulation.plan_start_state_failure_threshold
        # 这里只做计划起点连续性诊断；是否失败由配置 gate 决定，默认不改变 smoke 语义。
        return {
            "available": True,
            "phase": phase,
            "mismatch_detected": max_abs_error > failure_threshold,
            "warning_detected": max_abs_error > warning_threshold,
            "recommended_failure_reason": (
                "place_plan_start_state_mismatch"
                if phase == "place" and max_abs_error > failure_threshold
                else None
            ),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "l2_error": math.sqrt(sum(value * value for value in errors)),
            "warning_threshold": warning_threshold,
            "failure_threshold": failure_threshold,
            "joint_names": joint_names,
            "target_positions": targets,
            "actual_positions": actual_positions,
            "position_errors": errors,
            "peak_joint": {
                "joint_name": joint_names[peak_index] if peak_index < len(joint_names) else None,
                "joint_order_index": peak_index,
                "target_position": targets[peak_index] if peak_index < len(targets) else None,
                "actual_position": (
                    actual_positions[peak_index] if peak_index < len(actual_positions) else None
                ),
                "position_error": errors[peak_index] if peak_index < len(errors) else None,
                "abs_error": abs_errors[peak_index] if peak_index < len(abs_errors) else None,
            },
            **mapping_report,
            **start_target["metadata"],
        }

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

    def _manipulation_only_smoke_enabled(self) -> bool:
        return bool(self.config.manipulation_smoke or self.config.manipulation_apply_smoke)

    def _manipulation_smoke_event(self, suffix: str) -> str:
        prefix = (
            "manipulation_apply_smoke"
            if self.config.manipulation_apply_smoke
            else "manipulation_smoke"
        )
        return f"{prefix}_{suffix}"

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


def _metadata_tuple(metadata: dict[str, Any], key: str) -> tuple[Any, ...] | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
