"""tick 化机械臂轨迹执行器。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from source.interfaces import ArmPlan, GripperController, RobotAction, SimulationState


@dataclass(frozen=True)
class SegmentedArmExecutorConfig:
    """执行器采样参数；这里只生成命令，不推进仿真。"""

    sim_dt: float = 1.0 / 50.0
    arm_command_dt: float = 0.05
    motion_time_scale: float = 1.0
    place_approach_motion_time_scale: float = 1.0
    settle_to_segment_start_duration: float = 0.10
    settle_to_segment_start_skip_error_tolerance: float = 0.0
    post_motion_hold_duration: float = 1.50
    post_motion_joint_error_tolerance: float = 0.030
    fail_on_strict_post_motion_timeout: bool = True
    strict_post_motion_hold_segments: tuple[str, ...] = (
        "move_to_pregrasp",
        "approach_to_grasp",
        "move_to_pre_place",
        "approach_to_place",
        "retreat_place",
        "return_home_after_place",
    )
    pre_close_arm_hold_duration: float = 0.10
    gripper_move_duration: float = 0.70
    gripper_hold_duration: float = 0.45
    post_open_release_settle_duration: float = 0.50
    hold_arm_during_gripper: bool = True
    hold_gripper_after_close: bool = True

    def __post_init__(self) -> None:
        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")
        if self.arm_command_dt <= 0.0:
            raise ValueError("arm_command_dt must be positive")
        if self.motion_time_scale <= 0.0:
            raise ValueError("motion_time_scale must be positive")
        if self.place_approach_motion_time_scale <= 0.0:
            raise ValueError("place_approach_motion_time_scale must be positive")
        if self.settle_to_segment_start_duration < 0.0:
            raise ValueError("settle_to_segment_start_duration must be non-negative")
        if self.settle_to_segment_start_skip_error_tolerance < 0.0:
            raise ValueError("settle_to_segment_start_skip_error_tolerance must be non-negative")
        if self.post_motion_hold_duration < 0.0:
            raise ValueError("post_motion_hold_duration must be non-negative")
        if self.post_motion_joint_error_tolerance < 0.0:
            raise ValueError("post_motion_joint_error_tolerance must be non-negative")
        if self.pre_close_arm_hold_duration < 0.0:
            raise ValueError("pre_close_arm_hold_duration must be non-negative")
        if self.gripper_move_duration < 0.0:
            raise ValueError("gripper_move_duration must be non-negative")
        if self.gripper_hold_duration < 0.0:
            raise ValueError("gripper_hold_duration must be non-negative")
        if self.post_open_release_settle_duration < 0.0:
            raise ValueError("post_open_release_settle_duration must be non-negative")


@dataclass(frozen=True)
class _ActionStep:
    action: RobotAction
    segment_name: str
    segment_type: str


class SegmentedArmExecutor:
    """把 cuRobo 分段计划拆成 pipeline 主循环可调度的一拍一拍 action。"""

    def __init__(
        self,
        gripper: GripperController,
        *,
        config: SegmentedArmExecutorConfig | None = None,
    ):
        self.gripper = gripper
        self.config = config or SegmentedArmExecutorConfig()
        self.plan: ArmPlan | None = None
        self._steps: tuple[_ActionStep, ...] = ()
        self._tick_index = 0
        self._latest_metadata: dict[str, Any] = {}
        self._settle_context: dict[tuple[Any, ...], tuple[float, ...]] = {}
        self._gripper_context: dict[tuple[Any, ...], tuple[float, ...]] = {}
        self._pending_post_motion_hold_step: _ActionStep | None = None
        self._failed = False
        self._failure_reason = ""
        self._failure_metadata: dict[str, Any] = {}

    def reset(self, plan: ArmPlan) -> None:
        self.plan = plan
        self._steps = tuple(self._build_steps(plan))
        self._tick_index = 0
        self._latest_metadata = {}
        self._settle_context = {}
        self._gripper_context = {}
        self._pending_post_motion_hold_step = None
        self._failed = False
        self._failure_reason = ""
        self._failure_metadata = {}
        if not self._steps:
            raise RuntimeError(f"arm plan has no executable steps: {plan.operation}")

    def compute_action(self, state: SimulationState) -> RobotAction:
        if self.plan is None:
            raise RuntimeError("arm executor has no plan")
        self._finalize_pending_post_motion_hold(state)
        if self._failed:
            return RobotAction.idle(source=f"arm_{self.plan.operation}_failed")
        self._skip_settle_to_start_if_allowed(state)
        self._skip_converged_post_motion_holds(state)
        if self.is_done(state):
            return RobotAction.idle(source=f"arm_{self.plan.operation}_done")
        step = self._steps[self._tick_index]
        action = self._materialize_step_action(step, state)
        self._tick_index += 1
        self._remember_pending_post_motion_hold(step)
        self._latest_metadata = dict(action.metadata)
        return action

    def is_done(self, state: SimulationState) -> bool:
        del state
        return self.plan is not None and self._tick_index >= len(self._steps)

    def status(self) -> dict[str, Any]:
        current = None
        if self._tick_index < len(self._steps):
            step = self._steps[self._tick_index]
            current = {"name": step.segment_name, "type": step.segment_type}
        return {
            "backend": "segmented_arm_executor",
            "operation": self.plan.operation if self.plan is not None else None,
            "tick_index": self._tick_index,
            "total_ticks": len(self._steps),
            "done": self.plan is not None and self._tick_index >= len(self._steps),
            "current_segment": current,
            "latest_action": dict(self._latest_metadata),
            "failed": self._failed,
            "failure_reason": self._failure_reason,
            "failure_metadata": dict(self._failure_metadata),
            "world_step_owned_by_pipeline": True,
        }

    def _materialize_step_action(
        self,
        step: _ActionStep,
        state: SimulationState,
    ) -> RobotAction:
        if step.segment_type == "gripper":
            return self._materialize_gripper_step(step, state)
        if step.segment_type != "settle_to_segment_start":
            return step.action
        metadata = dict(step.action.metadata)
        arm_joint_names = tuple(str(name) for name in metadata.get("arm_joint_names", ()))
        q_target = tuple(float(value) for value in metadata["settle_target_positions"])
        key = (
            metadata.get("operation"),
            metadata.get("segment_index"),
            metadata.get("parent_segment_name"),
        )
        q_initial = self._settle_context.get(key)
        actual_positions, mapping_report = _actual_arm_positions(state, arm_joint_names)
        if q_initial is None:
            q_initial = actual_positions if actual_positions is not None else q_target
            self._settle_context[key] = q_initial
        start_error = _joint_error_norm(q_initial, q_target)
        tick = int(metadata.get("segment_tick") or 0)
        ticks = max(1, int(metadata.get("segment_ticks") or 1))
        u = 1.0 if ticks <= 1 else tick / float(ticks - 1)
        s = _smoothstep5(u)
        q_command = tuple((1.0 - s) * start + s * target for start, target in zip(q_initial, q_target))
        metadata.update(
            {
                "settle_start_positions": q_initial,
                "settle_target_positions": q_target,
                "settle_start_error_norm": start_error,
                "settle_joint_mapping": mapping_report,
                "baseline_settle_to_segment_start": True,
            }
        )
        return RobotAction(
            arm_joint_positions=q_command,
            gripper_command=step.action.gripper_command,
            source=step.action.source,
            metadata=metadata,
        )

    def _materialize_gripper_step(
        self,
        step: _ActionStep,
        state: SimulationState,
    ) -> RobotAction:
        """按 baseline 从实际夹爪开度平滑插值到目标，不在首帧瞬间跳变。"""

        metadata = dict(step.action.metadata)
        joint_names = tuple(str(name) for name in metadata.get("gripper_joint_names", ()))
        q_target = tuple(float(value) for value in metadata["gripper_final_target_positions"])
        key = (
            metadata.get("operation"),
            metadata.get("segment_index"),
            metadata.get("segment_name"),
        )
        q_initial = self._gripper_context.get(key)
        actual_positions, mapping_report = _actual_named_joint_positions(state, joint_names)
        if q_initial is None:
            q_initial = actual_positions if actual_positions is not None else q_target
            self._gripper_context[key] = q_initial

        move_tick = int(metadata.get("gripper_move_tick") or 0)
        move_steps = max(1, int(metadata.get("gripper_move_steps") or 1))
        if move_tick >= move_steps:
            q_command = q_target
            phase = "hold"
        else:
            u = 1.0 if move_steps <= 1 else move_tick / float(move_steps - 1)
            s = _smoothstep5(u)
            q_command = tuple(
                (1.0 - s) * start + s * target
                for start, target in zip(q_initial, q_target)
            )
            phase = "move"
        metadata.update(
            {
                "gripper_joint_positions": q_command,
                "gripper_start_positions": q_initial,
                "gripper_final_target_positions": q_target,
                "gripper_interpolation": "smoothstep5",
                "gripper_phase": phase,
                "gripper_joint_mapping": mapping_report,
                "baseline_gripper_interpolation": True,
            }
        )
        return RobotAction(
            arm_joint_positions=step.action.arm_joint_positions,
            gripper_command=step.action.gripper_command,
            source=step.action.source,
            metadata=metadata,
        )

    def _finalize_pending_post_motion_hold(self, state: SimulationState) -> None:
        step = self._pending_post_motion_hold_step
        if step is None or self._failed:
            return
        self._pending_post_motion_hold_step = None
        error = self._post_motion_hold_error(state, step)
        if error is None:
            return
        metadata = dict(step.action.metadata)
        metadata["post_motion_hold_final_error"] = error
        if error <= self.config.post_motion_joint_error_tolerance:
            metadata["post_motion_hold_converged"] = True
            self._latest_metadata = metadata
            return
        metadata["post_motion_hold_converged"] = False
        metadata["post_motion_hold_timeout"] = True
        self._latest_metadata = metadata
        if self.config.fail_on_strict_post_motion_timeout:
            self._failed = True
            self._failure_reason = "arm_post_motion_convergence_timeout"
            self._failure_metadata = {
                "segment_name": step.segment_name,
                "segment_type": step.segment_type,
                "final_error": error,
                "tolerance": self.config.post_motion_joint_error_tolerance,
                "timeout_s": self.config.post_motion_hold_duration,
                "operation": self.plan.operation if self.plan is not None else None,
            }

    def _remember_pending_post_motion_hold(self, step: _ActionStep) -> None:
        if step.segment_type != "post_motion_hold":
            return
        next_index = self._tick_index
        if next_index < len(self._steps):
            next_step = self._steps[next_index]
            if next_step.segment_type == "post_motion_hold" and next_step.segment_name == step.segment_name:
                return
        self._pending_post_motion_hold_step = step

    def _skip_settle_to_start_if_allowed(self, state: SimulationState) -> None:
        tolerance = self.config.settle_to_segment_start_skip_error_tolerance
        if tolerance <= 0.0:
            return
        while self._tick_index < len(self._steps):
            step = self._steps[self._tick_index]
            if step.segment_type != "settle_to_segment_start":
                return
            metadata = step.action.metadata
            if int(metadata.get("segment_tick") or 0) != 0:
                return
            arm_joint_names = tuple(str(name) for name in metadata.get("arm_joint_names", ()))
            q_target = tuple(float(value) for value in metadata["settle_target_positions"])
            actual_positions, mapping_report = _actual_arm_positions(state, arm_joint_names)
            if actual_positions is None:
                return
            error = _joint_error_norm(actual_positions, q_target)
            if error > tolerance:
                return
            latest = dict(metadata)
            latest.update(
                {
                    "settle_to_start_skipped": True,
                    "settle_start_error_norm": error,
                    "settle_joint_mapping": mapping_report,
                }
            )
            self._latest_metadata = latest
            self._skip_remaining_settle_to_start(step)

    def _skip_remaining_settle_to_start(self, step: _ActionStep) -> None:
        parent_name = step.action.metadata.get("parent_segment_name")
        while self._tick_index < len(self._steps):
            candidate = self._steps[self._tick_index]
            if (
                candidate.segment_type != "settle_to_segment_start"
                or candidate.action.metadata.get("parent_segment_name") != parent_name
            ):
                return
            self._tick_index += 1

    def _skip_converged_post_motion_holds(self, state: SimulationState) -> None:
        while self._tick_index < len(self._steps):
            step = self._steps[self._tick_index]
            error = self._post_motion_hold_error(state, step)
            if error is None or error > self.config.post_motion_joint_error_tolerance:
                return
            metadata = dict(step.action.metadata)
            metadata.update(
                {
                    "post_motion_hold_skipped_on_tolerance": True,
                    "post_motion_hold_error": error,
                }
            )
            self._latest_metadata = metadata
            self._skip_remaining_matching_hold(step)

    def _post_motion_hold_error(
        self,
        state: SimulationState,
        step: _ActionStep,
    ) -> float | None:
        if step.segment_type != "post_motion_hold" or step.action.arm_joint_positions is None:
            return None
        all_joint_names = tuple(str(name) for name in state.metadata.get("joint_names", ()))
        if not all_joint_names or not state.joint_positions:
            return None
        target_joint_names = tuple(str(name) for name in step.action.metadata.get("arm_joint_names", ()))
        if not target_joint_names:
            return None
        try:
            joint_indices = tuple(all_joint_names.index(name) for name in target_joint_names)
        except ValueError:
            return None
        if any(index >= len(state.joint_positions) for index in joint_indices):
            return None
        actual_positions = tuple(float(state.joint_positions[index]) for index in joint_indices)
        # 旧 baseline 用 joint error norm 判断是否到位；这里保持同一语义。
        return _joint_error_norm(
            actual_positions,
            tuple(float(value) for value in step.action.arm_joint_positions),
        )

    def _skip_remaining_matching_hold(self, step: _ActionStep) -> None:
        segment_name = step.segment_name
        while self._tick_index < len(self._steps):
            candidate = self._steps[self._tick_index]
            if (
                candidate.segment_type != "post_motion_hold"
                or candidate.segment_name != segment_name
            ):
                return
            self._tick_index += 1

    def _build_steps(self, plan: ArmPlan) -> list[_ActionStep]:
        metadata_segments = plan.metadata.get("segments")
        if metadata_segments:
            return self._build_segment_steps(plan, metadata_segments)
        return self._build_flat_trajectory_steps(plan)

    def _build_flat_trajectory_steps(self, plan: ArmPlan) -> list[_ActionStep]:
        arm_joint_names = tuple(plan.metadata.get("joint_names") or ())
        steps: list[_ActionStep] = []
        for index, target in enumerate(plan.joint_trajectory):
            steps.append(
                _ActionStep(
                    action=RobotAction(
                        arm_joint_positions=tuple(float(value) for value in target),
                        gripper_command=self.gripper.command_hold(),
                        source=f"arm_{plan.operation}",
                        metadata={
                            "operation": plan.operation,
                            "segment_name": "flat_trajectory",
                            "segment_type": "motion",
                            "segment_tick": index,
                            "progress": (index + 1) / len(plan.joint_trajectory),
                            "arm_joint_names": arm_joint_names,
                        },
                    ),
                    segment_name="flat_trajectory",
                    segment_type="motion",
                )
            )
        return steps

    def _build_segment_steps(
        self,
        plan: ArmPlan,
        segments: Any,
    ) -> list[_ActionStep]:
        arm_joint_names = tuple(plan.metadata.get("joint_names") or ())
        steps: list[_ActionStep] = []
        last_arm_target: tuple[float, ...] | None = None
        closed_gripper_target: tuple[float, ...] | None = None
        closed_gripper_joint_names: tuple[str, ...] = ()

        for segment_index, segment in enumerate(segments):
            segment_type = str(segment.get("type"))
            if segment_type == "motion":
                segment_steps, last_arm_target = self._build_motion_segment_steps(
                    plan=plan,
                    segment=segment,
                    segment_index=segment_index,
                    arm_joint_names=arm_joint_names,
                    closed_gripper_target=closed_gripper_target,
                    closed_gripper_joint_names=closed_gripper_joint_names,
                )
                steps.extend(segment_steps)
            elif segment_type == "gripper":
                segment_steps = self._build_gripper_segment_steps(
                    plan=plan,
                    segment=segment,
                    segment_index=segment_index,
                    arm_joint_names=arm_joint_names,
                    last_arm_target=last_arm_target,
                )
                steps.extend(segment_steps)
                if _is_close_segment(segment):
                    closed_gripper_target = tuple(segment["target_position"])
                    closed_gripper_joint_names = tuple(segment["joint_names"])
                elif _is_open_segment(segment):
                    closed_gripper_target = None
                    closed_gripper_joint_names = ()
            else:
                raise RuntimeError(f"unsupported arm segment type: {segment_type}")

        return steps

    def _build_motion_segment_steps(
        self,
        *,
        plan: ArmPlan,
        segment: dict[str, Any],
        segment_index: int,
        arm_joint_names: tuple[str, ...],
        closed_gripper_target: tuple[float, ...] | None,
        closed_gripper_joint_names: tuple[str, ...],
    ) -> tuple[list[_ActionStep], tuple[float, ...]]:
        name = str(segment.get("name") or "motion")
        trajectory = segment["trajectory"]
        q_rows = tuple(tuple(float(value) for value in row) for row in trajectory["q"])
        time_from_start = tuple(float(value) for value in trajectory["time_from_start"])
        qd_rows = tuple(
            tuple(float(value) for value in row)
            for row in trajectory.get("qd", _zero_rows_like(q_rows))
        )
        motion_time_scale = self.config.motion_time_scale
        if name == "approach_to_place":
            # 下放阶段保留更慢速度，其他大范围 motion 才使用统一加速比例。
            motion_time_scale = max(
                motion_time_scale,
                self.config.place_approach_motion_time_scale,
            )
        duration = max(0.0, time_from_start[-1] * motion_time_scale)
        num_steps = max(1, int(math.ceil(duration / self.config.sim_dt)) + 1)
        command_period_steps = max(1, int(round(self.config.arm_command_dt / self.config.sim_dt)))
        steps: list[_ActionStep] = self._build_settle_to_segment_start_steps(
            plan=plan,
            segment_name=name,
            segment_index=segment_index,
            arm_joint_names=arm_joint_names,
            q_start=q_rows[0],
        )
        q_target = q_rows[0]
        for step_index in range(num_steps):
            t_exec = min(step_index * self.config.sim_dt, duration)
            t_plan = min(t_exec / motion_time_scale, time_from_start[-1])
            if step_index % command_period_steps == 0 or step_index == num_steps - 1:
                q_target = _sample_cubic_hermite(time_from_start, q_rows, qd_rows, t_plan)
            metadata = {
                "operation": plan.operation,
                "segment_index": segment_index,
                "segment_name": name,
                "segment_type": "motion",
                "segment_tick": step_index,
                "segment_ticks": num_steps,
                "t_exec_s": t_exec,
                "t_plan_s": t_plan,
                "arm_command_dt": self.config.arm_command_dt,
                "motion_time_scale": motion_time_scale,
                "command_period_steps": command_period_steps,
                "interpolation": "cubic_hermite",
                "progress": (step_index + 1) / num_steps,
                "arm_joint_names": arm_joint_names,
                # 这里明确声明执行器只产生命令，world.step 由 pipeline 统一推进。
                "world_step_owned_by_pipeline": True,
            }
            gripper_command = self.gripper.command_hold()
            if self.config.hold_gripper_after_close and closed_gripper_target is not None:
                gripper_command = self.gripper.command_close()
                metadata.update(
                    {
                        "gripper_hold_after_close": True,
                        "gripper_joint_names": closed_gripper_joint_names,
                        "gripper_joint_positions": closed_gripper_target,
                    }
                )
            steps.append(
                _ActionStep(
                    action=RobotAction(
                        arm_joint_positions=q_target,
                        gripper_command=gripper_command,
                        source=f"arm_{plan.operation}",
                        metadata=metadata,
                    ),
                    segment_name=name,
                    segment_type="motion",
                )
            )
        final_target = q_rows[-1]
        if name in self.config.strict_post_motion_hold_segments:
            steps.extend(
                self._build_post_motion_hold_steps(
                    plan=plan,
                    segment_name=name,
                    segment_index=segment_index,
                    arm_joint_names=arm_joint_names,
                    arm_target=final_target,
                    closed_gripper_target=closed_gripper_target,
                    closed_gripper_joint_names=closed_gripper_joint_names,
                )
            )
        return steps, final_target

    def _build_settle_to_segment_start_steps(
        self,
        *,
        plan: ArmPlan,
        segment_name: str,
        segment_index: int,
        arm_joint_names: tuple[str, ...],
        q_start: tuple[float, ...],
    ) -> list[_ActionStep]:
        if self.config.settle_to_segment_start_duration <= 0.0:
            return []
        settle_steps = max(
            2,
            int(self.config.settle_to_segment_start_duration / self.config.sim_dt),
        )
        steps: list[_ActionStep] = []
        for step_index in range(settle_steps):
            metadata = {
                "operation": plan.operation,
                "segment_index": segment_index,
                "segment_name": f"{segment_name}_settle_to_start",
                "parent_segment_name": segment_name,
                "segment_type": "settle_to_segment_start",
                "segment_tick": step_index,
                "segment_ticks": settle_steps,
                "arm_joint_names": arm_joint_names,
                "settle_target_positions": q_start,
                "settle_duration_s": self.config.settle_to_segment_start_duration,
                "settle_skip_error_tolerance": (
                    self.config.settle_to_segment_start_skip_error_tolerance
                ),
                # 对齐旧 baseline：每段 motion 前先平滑贴到该段第一帧，避免状态偏差直接冲击轨迹。
                "baseline_settle_to_segment_start": True,
                "world_step_owned_by_pipeline": True,
            }
            steps.append(
                _ActionStep(
                    action=RobotAction(
                        arm_joint_positions=q_start,
                        gripper_command=self.gripper.command_hold(),
                        source=f"arm_{plan.operation}",
                        metadata=metadata,
                    ),
                    segment_name=f"{segment_name}_settle_to_start",
                    segment_type="settle_to_segment_start",
                )
            )
        return steps

    def _build_post_motion_hold_steps(
        self,
        *,
        plan: ArmPlan,
        segment_name: str,
        segment_index: int,
        arm_joint_names: tuple[str, ...],
        arm_target: tuple[float, ...],
        closed_gripper_target: tuple[float, ...] | None,
        closed_gripper_joint_names: tuple[str, ...],
    ) -> list[_ActionStep]:
        hold_steps = max(
            0,
            int(math.ceil(self.config.post_motion_hold_duration / self.config.sim_dt)),
        )
        steps: list[_ActionStep] = []
        for step_index in range(hold_steps):
            metadata = {
                "operation": plan.operation,
                "segment_index": segment_index,
                "segment_name": segment_name,
                "segment_type": "post_motion_hold",
                "segment_tick": step_index,
                "segment_ticks": hold_steps,
                "progress": (step_index + 1) / hold_steps if hold_steps else 1.0,
                "arm_joint_names": arm_joint_names,
                "post_motion_hold_duration_s": self.config.post_motion_hold_duration,
                "post_motion_joint_error_tolerance": self.config.post_motion_joint_error_tolerance,
                # 旧 baseline 会在关键段后等待真实 articulation 追上，避免提前关夹爪或放物体。
                "baseline_convergence_hold": True,
                "world_step_owned_by_pipeline": True,
            }
            gripper_command = self.gripper.command_hold()
            if self.config.hold_gripper_after_close and closed_gripper_target is not None:
                gripper_command = self.gripper.command_close()
                metadata.update(
                    {
                        "gripper_hold_after_close": True,
                        "gripper_joint_names": closed_gripper_joint_names,
                        "gripper_joint_positions": closed_gripper_target,
                    }
                )
            steps.append(
                _ActionStep(
                    action=RobotAction(
                        arm_joint_positions=arm_target,
                        gripper_command=gripper_command,
                        source=f"arm_{plan.operation}",
                        metadata=metadata,
                    ),
                    segment_name=segment_name,
                    segment_type="post_motion_hold",
                )
            )
        return steps

    def _build_gripper_segment_steps(
        self,
        *,
        plan: ArmPlan,
        segment: dict[str, Any],
        segment_index: int,
        arm_joint_names: tuple[str, ...],
        last_arm_target: tuple[float, ...] | None,
    ) -> list[_ActionStep]:
        name = str(segment.get("name") or "gripper")
        target_position = tuple(float(value) for value in segment["target_position"])
        joint_names = tuple(str(name) for name in segment["joint_names"])
        is_close = _is_close_segment(segment)
        is_open = _is_open_segment(segment)
        command = (
            self.gripper.command_close()
            if is_close
            else self.gripper.command_open()
            if is_open
            else self.gripper.command_hold()
        )
        event_marker = "gripper_close" if is_close else "gripper_open" if is_open else None
        move_steps = max(1, int(math.ceil(self.config.gripper_move_duration / self.config.sim_dt)))
        hold_steps = max(0, int(math.ceil(self.config.gripper_hold_duration / self.config.sim_dt)))
        total_steps = move_steps + hold_steps
        steps: list[_ActionStep] = []
        if is_close and self.config.hold_arm_during_gripper and last_arm_target is not None:
            pre_hold_steps = max(
                0,
                int(math.ceil(self.config.pre_close_arm_hold_duration / self.config.sim_dt)),
            )
            for step_index in range(pre_hold_steps):
                steps.append(
                    _ActionStep(
                        action=RobotAction(
                            arm_joint_positions=last_arm_target,
                            gripper_command=self.gripper.command_hold(),
                            source=f"arm_{plan.operation}",
                            metadata={
                                "operation": plan.operation,
                                "segment_index": segment_index,
                                "segment_name": f"{name}_pre_hold",
                                "parent_segment_name": name,
                                "segment_type": "pre_close_arm_hold",
                                "segment_tick": step_index,
                                "segment_ticks": pre_hold_steps,
                                "progress": (step_index + 1) / pre_hold_steps
                                if pre_hold_steps
                                else 1.0,
                                "arm_joint_names": arm_joint_names,
                                "pre_close_arm_hold_duration_s": (
                                    self.config.pre_close_arm_hold_duration
                                ),
                                # close 前再保持一小段时间，对齐旧 baseline 的抓取节奏。
                                "baseline_pre_close_hold": True,
                                "world_step_owned_by_pipeline": True,
                            },
                        ),
                        segment_name=f"{name}_pre_hold",
                        segment_type="pre_close_arm_hold",
                    )
                )
        for step_index in range(total_steps):
            metadata = {
                "operation": plan.operation,
                "segment_index": segment_index,
                "segment_name": name,
                "segment_type": "gripper",
                "segment_tick": step_index,
                "segment_ticks": total_steps,
                "progress": (step_index + 1) / total_steps,
                "gripper_joint_names": joint_names,
                "gripper_joint_positions": target_position,
                "gripper_final_target_positions": target_position,
                "gripper_move_tick": min(step_index, move_steps),
                "gripper_move_steps": move_steps,
                # 夹爪闭合/打开期间持续保持上一段 arm 末端，避免 TCP 漂走。
                "arm_hold_during_gripper": bool(
                    self.config.hold_arm_during_gripper and last_arm_target is not None
                ),
                "world_step_owned_by_pipeline": True,
            }
            if event_marker is not None and step_index == 0:
                metadata["event_marker"] = event_marker
            steps.append(
                _ActionStep(
                    action=RobotAction(
                        arm_joint_positions=(
                            last_arm_target
                            if self.config.hold_arm_during_gripper and last_arm_target is not None
                            else None
                        ),
                        gripper_command=command,
                        source=f"arm_{plan.operation}",
                        metadata=metadata,
                    ),
                    segment_name=name,
                    segment_type="gripper",
                )
            )
        if is_open and last_arm_target is not None:
            settle_steps = max(
                0,
                int(
                    math.ceil(
                        self.config.post_open_release_settle_duration / self.config.sim_dt
                    )
                ),
            )
            for step_index in range(settle_steps):
                steps.append(
                    _ActionStep(
                        action=RobotAction(
                            arm_joint_positions=last_arm_target,
                            gripper_command=self.gripper.command_open(),
                            source=f"arm_{plan.operation}",
                            metadata={
                                "operation": plan.operation,
                                "segment_index": segment_index,
                                "segment_name": f"{name}_release_settle",
                                "parent_segment_name": name,
                                "segment_type": "post_open_release_settle",
                                "segment_tick": step_index,
                                "segment_ticks": settle_steps,
                                "progress": (step_index + 1) / settle_steps,
                                "arm_joint_names": arm_joint_names,
                                "gripper_joint_names": joint_names,
                                "gripper_joint_positions": target_position,
                                "post_open_release_settle_duration_s": (
                                    self.config.post_open_release_settle_duration
                                ),
                                # baseline 在退臂前保持释放位姿，避免尚未脱离的苹果被再次抬起。
                                "baseline_release_settle": True,
                                "world_step_owned_by_pipeline": True,
                            },
                        ),
                        segment_name=f"{name}_release_settle",
                        segment_type="post_open_release_settle",
                    )
                )
        return steps


def _is_close_segment(segment: dict[str, Any]) -> bool:
    return "close" in str(segment.get("name") or "").lower()


def _is_open_segment(segment: dict[str, Any]) -> bool:
    return "open" in str(segment.get("name") or "").lower()


def _actual_named_joint_positions(
    state: SimulationState,
    joint_names: tuple[str, ...],
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    runtime_joint_names = tuple(str(name) for name in state.metadata.get("joint_names", ()))
    if not runtime_joint_names or not state.joint_positions:
        return None, {
            "available": False,
            "reason": "joint_names_or_positions_unavailable",
        }
    index_by_name = {name: index for index, name in enumerate(runtime_joint_names)}
    missing = tuple(name for name in joint_names if name not in index_by_name)
    if missing:
        return None, {
            "available": False,
            "reason": "joint_names_missing",
            "missing_joint_names": missing,
        }
    positions = tuple(
        float(state.joint_positions[index_by_name[name]])
        for name in joint_names
    )
    return positions, {
        "available": True,
        "joint_names": joint_names,
        "joint_indices": tuple(index_by_name[name] for name in joint_names),
    }


def _zero_rows_like(
    q_rows: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(0.0 for _ in row) for row in q_rows)


def _sample_cubic_hermite(
    time_from_start: tuple[float, ...],
    q_rows: tuple[tuple[float, ...], ...],
    qd_rows: tuple[tuple[float, ...], ...],
    t_plan: float,
) -> tuple[float, ...]:
    if len(q_rows) == 1 or t_plan <= time_from_start[0]:
        return q_rows[0]
    if t_plan >= time_from_start[-1]:
        return q_rows[-1]
    index = 0
    for candidate in range(1, len(time_from_start)):
        if t_plan <= time_from_start[candidate]:
            index = candidate - 1
            break
    next_index = min(index + 1, len(q_rows) - 1)
    t0 = float(time_from_start[index])
    t1 = float(time_from_start[next_index])
    h = t1 - t0
    if h <= 1.0e-9:
        return q_rows[index]
    u = (float(t_plan) - t0) / h
    q0 = q_rows[index]
    q1 = q_rows[next_index]
    v0 = qd_rows[index]
    v1 = qd_rows[next_index]
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    # cuRobo JSON 同时保存 q/qd；旧 baseline 使用 cubic Hermite，而不是线性插值。
    return tuple(
        h00 * p0 + h10 * h * d0 + h01 * p1 + h11 * h * d1
        for p0, p1, d0, d1 in zip(q0, q1, v0, v1)
    )


def _smoothstep5(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _actual_arm_positions(
    state: SimulationState,
    joint_names: tuple[str, ...],
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    runtime_joint_names = tuple(str(name) for name in state.metadata.get("joint_names", ()))
    if not runtime_joint_names or not state.joint_positions:
        return None, {
            "available": False,
            "reason": "joint_names_or_positions_unavailable",
        }
    index_by_name = {name: index for index, name in enumerate(runtime_joint_names)}
    missing = [name for name in joint_names if name not in index_by_name]
    if missing:
        return None, {
            "available": False,
            "reason": "joint_names_missing",
            "missing_joint_names": missing,
        }
    indices = tuple(index_by_name[name] for name in joint_names)
    if any(index >= len(state.joint_positions) for index in indices):
        return None, {
            "available": False,
            "reason": "joint_index_out_of_range",
            "joint_indices": indices,
            "joint_position_count": len(state.joint_positions),
        }
    return tuple(float(state.joint_positions[index]) for index in indices), {
        "available": True,
        "joint_indices": indices,
    }


def _joint_error_norm(
    actual: tuple[float, ...],
    target: tuple[float, ...],
) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, target)))


def _sample_linear(
    time_from_start: tuple[float, ...],
    q_rows: tuple[tuple[float, ...], ...],
    t_plan: float,
) -> tuple[float, ...]:
    if len(q_rows) == 1 or t_plan <= time_from_start[0]:
        return q_rows[0]
    for index in range(1, len(time_from_start)):
        t0 = time_from_start[index - 1]
        t1 = time_from_start[index]
        if t_plan <= t1:
            if t1 <= t0:
                return q_rows[index]
            alpha = (t_plan - t0) / (t1 - t0)
            return tuple(
                (1.0 - alpha) * q0 + alpha * q1
                for q0, q1 in zip(q_rows[index - 1], q_rows[index])
            )
    return q_rows[-1]
