"""State machine for continuous contact-only nav-pick-carry-place episodes."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from source.data.task_schema import NavPickTask, Pose2D
from source.data.vla_episode_recorder import VLAEpisodeRecorder
from source.manipulation.contact_grasp_monitor import ContactGraspMonitor


class ContactPickPlacePhase(str, Enum):
    RESET = "reset"
    NAV_TO_PICK = "nav_to_pick"
    PICK_PREPARE = "pick_prepare"
    PICK_APPROACH = "pick_approach"
    GRIPPER_CLOSE = "gripper_close"
    LIFT = "lift"
    VERIFY_GRASP = "verify_grasp"
    MOVE_TO_CARRY_POSTURE = "move_to_carry_posture"
    CARRY_NAV_TO_PLACE = "carry_nav_to_place"
    PLACE_APPROACH = "place_approach"
    GRIPPER_OPEN = "gripper_open"
    PLACE_RETREAT = "place_retreat"
    VERIFY_PLACE = "verify_place"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class ContactPickPlaceLimits:
    """Loop limits and acceptance thresholds for one contact-only episode."""

    max_nav_to_pick_steps: int = 3000
    max_carry_nav_steps: int = 3000
    goal_tolerance: float = 0.15
    goal_yaw_tolerance: float = 0.10
    terminal_position_tolerance: float = 0.08
    terminal_yaw_tolerance: float = 0.08
    final_goal_tolerance_margin: float = 0.03
    final_yaw_tolerance_margin: float = 0.08
    place_xy_tolerance: float = 0.10
    place_z_tolerance: float = 0.08
    verify_grasp_steps: int = 60
    verify_carry_every_steps: int = 10

    @property
    def position_acceptance_tolerance(self) -> float:
        return self.goal_tolerance + max(0.0, self.final_goal_tolerance_margin)

    @property
    def yaw_acceptance_tolerance(self) -> float:
        return self.goal_yaw_tolerance + max(0.0, self.final_yaw_tolerance_margin)


@dataclass(frozen=True)
class RuntimeStepResult:
    """One runtime step to be recorded in VLA JSONL."""

    observation: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    images: dict[str, bytes | str | None] = field(default_factory=dict)
    reached_goal: bool = False


@dataclass(frozen=True)
class PhaseResult:
    """Result returned by runtime phase executors."""

    success: bool
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ContactPickPlaceRuntime(Protocol):
    """Runtime adapter implemented by Isaac Lab or smoke-test backends."""

    def reset_episode(self, task: NavPickTask) -> PhaseResult:
        ...

    def step_navigation(self, goal: Pose2D, phase: ContactPickPlacePhase) -> RuntimeStepResult:
        ...

    def stop_base(self) -> None:
        ...

    def settle_base(self, phase: ContactPickPlacePhase) -> RuntimeStepResult:
        ...

    def execute_pick(self) -> PhaseResult:
        ...

    def move_to_carry_posture(self, posture_name: str) -> PhaseResult:
        ...

    def hold_carry_posture(self, phase: ContactPickPlacePhase) -> RuntimeStepResult:
        ...

    def execute_place(self) -> PhaseResult:
        ...

    def verify_place(self) -> PhaseResult:
        ...


class ContactPickPlaceStateMachine:
    """Run one continuous contact-only nav-pick-carry-place episode."""

    def __init__(
        self,
        *,
        task: NavPickTask,
        runtime: ContactPickPlaceRuntime,
        monitor: ContactGraspMonitor,
        recorder: VLAEpisodeRecorder,
        limits: ContactPickPlaceLimits | None = None,
        carry_posture_name: str | None = None,
    ):
        self.task = task
        self.runtime = runtime
        self.monitor = monitor
        self.recorder = recorder
        self.limits = limits or ContactPickPlaceLimits(
            verify_grasp_steps=task.carry.verify_grasp_steps,
            verify_carry_every_steps=task.carry.verify_carry_every_steps,
            place_xy_tolerance=task.place.place_xy_tolerance,
            place_z_tolerance=task.place.place_z_tolerance,
        )
        self.carry_posture_name = carry_posture_name or task.carry.arm_posture
        self.phase = ContactPickPlacePhase.RESET
        self.summary: dict[str, Any] = {
            "success": False,
            "failure_reason": "",
            "carry_mode": "contact",
            "grasp_verified": False,
            "object_lifted": False,
            "object_slipped": False,
            "place_success": False,
            "started_at": time.time(),
        }

    def run(self) -> dict[str, Any]:
        if self.task.carry.mode != "contact":
            return self._fail("unsupported_carry_mode", {"carry_mode": self.task.carry.mode})
        if not self.task.place.enabled:
            return self._fail("place_disabled", {"place": "task.place.enabled is false"})
        if self.task.place.base_goal is None:
            return self._fail("place_base_goal_missing")
        if self.task.place.place_pose_world is None:
            return self._fail("place_pose_world_missing")

        reset = self.runtime.reset_episode(self.task)
        self._record_result(ContactPickPlacePhase.RESET, reset)
        if not reset.success:
            return self._fail(reset.failure_reason or "reset_failed", reset.metadata)

        if not self._navigate(
            goal=self.task.pick.base_goal,
            phase=ContactPickPlacePhase.NAV_TO_PICK,
            max_steps=self.limits.max_nav_to_pick_steps,
            failure_reason="nav_to_pick_failed",
        ):
            return self._finalize()

        self.runtime.stop_base()
        self._record_step(ContactPickPlacePhase.PICK_PREPARE, self.runtime.settle_base(ContactPickPlacePhase.PICK_PREPARE))
        self.monitor.mark_grasp_reference()
        pick = self.runtime.execute_pick()
        self._record_result(ContactPickPlacePhase.LIFT, pick)
        if not pick.success:
            return self._fail(pick.failure_reason or "pick_failed", pick.metadata)

        if not self._verify_grasp():
            return self._finalize()

        carry_posture = self.runtime.move_to_carry_posture(self.carry_posture_name)
        self._record_result(ContactPickPlacePhase.MOVE_TO_CARRY_POSTURE, carry_posture)
        if not carry_posture.success:
            return self._fail(carry_posture.failure_reason or "carry_posture_failed", carry_posture.metadata)
        if self.monitor.check_object_slipped():
            return self._fail("object_dropped_during_carry", self.monitor.get_slip_report())

        if not self._navigate(
            goal=self.task.place.base_goal,
            phase=ContactPickPlacePhase.CARRY_NAV_TO_PLACE,
            max_steps=self.limits.max_carry_nav_steps,
            failure_reason="carry_nav_failed",
            verify_carry=True,
        ):
            return self._finalize()

        self.runtime.stop_base()
        self._record_step(ContactPickPlacePhase.PLACE_APPROACH, self.runtime.settle_base(ContactPickPlacePhase.PLACE_APPROACH))
        place = self.runtime.execute_place()
        self._record_result(ContactPickPlacePhase.GRIPPER_OPEN, place)
        if not place.success:
            return self._fail(place.failure_reason or "place_plan_failed", place.metadata)

        verify_place = self.runtime.verify_place()
        self._record_result(ContactPickPlacePhase.VERIFY_PLACE, verify_place)
        if not verify_place.success:
            return self._fail(verify_place.failure_reason or "object_out_of_place", verify_place.metadata)

        self.phase = ContactPickPlacePhase.DONE
        self.summary.update(
            {
                "success": True,
                "failure_reason": "",
                "place_success": True,
                "finished_at": time.time(),
            }
        )
        return self._finalize()

    def _navigate(
        self,
        *,
        goal: Pose2D,
        phase: ContactPickPlacePhase,
        max_steps: int,
        failure_reason: str,
        verify_carry: bool = False,
    ) -> bool:
        self.phase = phase
        for step in range(max(1, int(max_steps))):
            try:
                result = self.runtime.step_navigation(goal, phase)
            except Exception as exc:
                self._fail(
                    failure_reason,
                    {
                        "step": step,
                        "goal": {"x": goal.x, "y": goal.y, "yaw": goal.yaw},
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                return False
            state = dict(result.state)
            if verify_carry and step % max(1, self.limits.verify_carry_every_steps) == 0:
                slipped = self.monitor.check_object_slipped()
                state.update(
                    {
                        "object_slipped": slipped,
                        "grasp_verified": self.summary["grasp_verified"],
                        "object_lifted": self.summary["object_lifted"],
                        "contact_report": self.monitor.get_contact_report(),
                    }
                )
                if slipped:
                    self._record_step(phase, result, state_override=state)
                    self._fail("object_dropped_during_carry", self.monitor.get_slip_report())
                    return False
            self._record_step(phase, result, state_override=state)
            if result.reached_goal:
                return True
        self._fail(failure_reason, {"max_steps": max_steps})
        return False

    def _verify_grasp(self) -> bool:
        self.phase = ContactPickPlacePhase.VERIFY_GRASP
        lifted = self.monitor.check_object_lifted()
        self.summary["object_lifted"] = lifted
        if not lifted:
            self._fail("grasp_not_lifted", self.monitor.get_lift_report())
            return False

        stable = True
        for _ in range(max(1, self.limits.verify_grasp_steps)):
            result = self.runtime.hold_carry_posture(ContactPickPlacePhase.VERIFY_GRASP)
            lifted = self.monitor.check_object_lifted()
            slipped = self.monitor.check_object_slipped()
            stable = self.monitor.check_grasp_stable() and not slipped
            state = dict(result.state)
            state.update(
                {
                    "object_lifted": lifted,
                    "object_slipped": slipped,
                    "grasp_verified": stable,
                    "contact_report": self.monitor.get_contact_report(),
                }
            )
            self._record_step(ContactPickPlacePhase.VERIFY_GRASP, result, state_override=state)
            if slipped or not stable:
                self._fail("grasp_slipped", self.monitor.get_slip_report())
                return False

        self.summary.update({"grasp_verified": stable, "object_lifted": lifted})
        return stable

    def _record_result(self, phase: ContactPickPlacePhase, result: PhaseResult) -> None:
        self.recorder.record(
            phase=phase.value,
            observation=result.metadata.get("observation", {}),
            action=result.metadata.get("action", {}),
            state={
                "phase_success": result.success,
                "failure_reason": result.failure_reason,
                "object_in_gripper_contact": False,
                "grasp_verified": self.summary["grasp_verified"],
                "object_lifted": self.summary["object_lifted"],
                "object_slipped": self.summary["object_slipped"],
                **result.metadata.get("state", {}),
            },
            force=True,
        )

    def _record_step(
        self,
        phase: ContactPickPlacePhase,
        result: RuntimeStepResult,
        *,
        state_override: dict[str, Any] | None = None,
    ) -> None:
        state = {
            "object_in_gripper_contact": False,
            "grasp_verified": self.summary["grasp_verified"],
            "object_lifted": self.summary["object_lifted"],
            "object_slipped": self.summary["object_slipped"],
            **dict(result.state),
        }
        if state_override is not None:
            state.update(state_override)
        self.recorder.record(
            phase=phase.value,
            observation=result.observation,
            action=result.action,
            state=state,
            timestamp=result.timestamp,
            images=result.images,
        )

    def _fail(self, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self.phase = ContactPickPlacePhase.FAILED
        self.summary.update(
            {
                "success": False,
                "failure_reason": reason,
                "failure_metadata": metadata or {},
                "object_slipped": bool(self.summary.get("object_slipped", False))
                or reason in {"object_dropped_during_carry", "grasp_slipped"},
                "finished_at": time.time(),
            }
        )
        return self._finalize()

    def _finalize(self) -> dict[str, Any]:
        self.summary.update(
            {
                "frame_count": self.recorder.frame_count,
                "final_phase": self.phase.value,
                "attachment_mode": "contact_only",
                "object_attached": False,
            }
        )
        self.recorder.write_summary(self.summary)
        return dict(self.summary)
