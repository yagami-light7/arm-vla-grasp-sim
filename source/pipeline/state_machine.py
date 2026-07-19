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
    VerificationResult,
)
from source.simulation.object_initialization import (
    evaluate_object_initialization_pose,
    resolve_object_initialization_policy,
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

# The navigation executor and verifier observe adjacent physics frames.  A
# successful zero-command handoff can therefore drift by a few millimetres or
# retain a small residual velocity before manipulation locks and settles the
# base.  Keep this allowance deliberately small and require an explicit
# executor success so it cannot turn a genuinely unreachable goal into a pass.
_NAV_HANDOFF_POSITION_MARGIN_M = 0.005
_NAV_HANDOFF_Z_MARGIN_M = 0.010
_NAV_HANDOFF_YAW_MARGIN_RAD = 0.010
_NAV_HANDOFF_LINEAR_SPEED_MARGIN_MPS = 0.015
_NAV_HANDOFF_ANGULAR_SPEED_MARGIN_RADPS = 0.040
_NAV_HANDOFF_SETTLE_MAX_STEPS = 12
_NAV_HANDOFF_SETTLE_LINEAR_SPEED_MARGIN_MPS = 0.060
_NAV_HANDOFF_SETTLE_ANGULAR_SPEED_MARGIN_RADPS = 0.200


def _quat_wxyz_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _navigation_plan_execution_metadata(
    raw_task: dict[str, Any],
    *,
    include_carry_departure: bool,
) -> dict[str, Any]:
    """Resolve task-level navigation manoeuvres into planner metadata.

    PCT and DWA should not know task-schema paths such as
    ``pick.support_geometry``.  The task layer converts that geometry into a
    small, explicit execution contract.  The executor can then apply the same
    departure policy to any source receptacle without hard-coding box names or
    scene coordinates.
    """

    raw_execution = raw_task.get("navigation_execution")
    if raw_execution is None:
        return {}
    if not isinstance(raw_execution, dict):
        raise ValueError("task.navigation_execution 必须是对象")

    metadata: dict[str, Any] = {
        "navigation_execution": dict(raw_execution),
    }
    if not include_carry_departure:
        return metadata

    raw_departure = raw_execution.get("carry_departure")
    if raw_departure is None:
        return metadata
    if not isinstance(raw_departure, dict):
        raise ValueError("task.navigation_execution.carry_departure 必须是对象")
    departure = dict(raw_departure)
    if not bool(departure.get("enabled", False)):
        metadata["carry_departure"] = departure
        return metadata

    pick = raw_task.get("pick")
    support_geometry = pick.get("support_geometry") if isinstance(pick, dict) else None
    if not isinstance(support_geometry, dict):
        raise ValueError(
            "启用 carry_departure 时 task.pick.support_geometry 必须存在"
        )
    bbox_min = support_geometry.get("world_bbox_min_xyz")
    bbox_max = support_geometry.get("world_bbox_max_xyz")
    if not (
        isinstance(bbox_min, (list, tuple))
        and isinstance(bbox_max, (list, tuple))
        and len(bbox_min) >= 2
        and len(bbox_max) >= 2
    ):
        raise ValueError(
            "启用 carry_departure 时 support_geometry 必须提供 world bbox"
        )
    min_x, min_y = float(bbox_min[0]), float(bbox_min[1])
    max_x, max_y = float(bbox_max[0]), float(bbox_max[1])
    values = (min_x, min_y, max_x, max_y)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("carry_departure source support bbox 包含非有限数值")
    support_center = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
    support_half_diagonal = 0.5 * math.hypot(max_x - min_x, max_y - min_y)
    turn_swept_radius = float(departure.get("robot_turn_swept_radius_m", 0.0))
    safety_margin = float(departure.get("safety_margin_m", 0.0))
    if turn_swept_radius <= 0.0 or safety_margin < 0.0:
        raise ValueError(
            "carry_departure robot_turn_swept_radius_m 必须为正且 safety_margin_m 不能为负"
        )
    departure.update(
        {
            "source_support_id": raw_task.get("source_receptacle_id"),
            "source_support_prim_path": pick.get("target_support_prim_path"),
            "source_support_center_xy": support_center,
            "source_support_half_diagonal_m": support_half_diagonal,
            "required_center_clearance_m": (
                support_half_diagonal + turn_swept_radius + safety_margin
            ),
            "clearance_formula": (
                "support_half_diagonal + robot_turn_swept_radius + safety_margin"
            ),
        }
    )
    metadata["carry_departure"] = departure
    return metadata


def _accept_successful_navigation_handoff_drift(
    result: VerificationResult,
    executor_status: dict[str, Any] | None,
    *,
    phase: str,
) -> VerificationResult:
    """Accept only tiny one-frame drift after the executor already succeeded."""

    if result.success or not isinstance(executor_status, dict):
        return result
    if executor_status.get("success") is not True:
        return result
    if executor_status.get("failed") is True:
        return result

    metadata = dict(result.metadata)
    checks = {
        "position": float(metadata.get("goal_distance", math.inf))
        <= float(metadata.get("position_tolerance", 0.0))
        + _NAV_HANDOFF_POSITION_MARGIN_M,
        "z": (
            not bool(metadata.get("z_check_enabled"))
            or float(metadata.get("goal_z_error", math.inf))
            <= float(metadata.get("goal_z_tolerance", 0.0))
            + _NAV_HANDOFF_Z_MARGIN_M
        ),
        "yaw": (
            not bool(metadata.get("yaw_alignment_required"))
            or float(metadata.get("yaw_error", math.inf))
            <= float(metadata.get("yaw_tolerance", 0.0))
            + _NAV_HANDOFF_YAW_MARGIN_RAD
        ),
        "linear_speed": (
            not bool(metadata.get("base_stability_required"))
            or float(metadata.get("linear_speed", math.inf))
            <= float(metadata.get("linear_velocity_tolerance", 0.0))
            + _NAV_HANDOFF_LINEAR_SPEED_MARGIN_MPS
        ),
        "angular_speed": (
            not bool(metadata.get("base_stability_required"))
            or float(metadata.get("angular_speed", math.inf))
            <= float(metadata.get("angular_velocity_tolerance", 0.0))
            + _NAV_HANDOFF_ANGULAR_SPEED_MARGIN_RADPS
        ),
    }
    if not all(checks.values()):
        return result
    return VerificationResult(
        success=True,
        failure_reason="",
        metadata={
            **metadata,
            "navigation_verifier_override": (
                "successful_executor_one_frame_drift_margin"
            ),
            "navigation_handoff_phase": phase,
            "navigation_handoff_margin_checks": checks,
            "navigation_handoff_margins": {
                "position_m": _NAV_HANDOFF_POSITION_MARGIN_M,
                "z_m": _NAV_HANDOFF_Z_MARGIN_M,
                "yaw_rad": _NAV_HANDOFF_YAW_MARGIN_RAD,
                "linear_speed_mps": _NAV_HANDOFF_LINEAR_SPEED_MARGIN_MPS,
                "angular_speed_radps": (
                    _NAV_HANDOFF_ANGULAR_SPEED_MARGIN_RADPS
                ),
            },
            "executor_distance_to_goal": executor_status.get(
                "distance_to_goal"
            ),
            "executor_yaw_error": executor_status.get("yaw_error"),
        },
    )


def _navigation_handoff_requires_zero_command_settle(
    result: VerificationResult,
    executor_status: dict[str, Any] | None,
) -> bool:
    """Return whether only bounded residual base motion blocks verification."""

    if result.success or not isinstance(executor_status, dict):
        return False
    if executor_status.get("success") is not True:
        return False
    if executor_status.get("failed") is True:
        return False

    metadata = result.metadata
    if not bool(metadata.get("base_stability_required")):
        return False
    if bool(metadata.get("base_stable")):
        return False
    if not bool(metadata.get("position_reached")):
        return False
    if bool(metadata.get("z_check_enabled")) and not bool(
        metadata.get("z_reached")
    ):
        return False
    if bool(metadata.get("yaw_alignment_required")) and not bool(
        metadata.get("yaw_aligned")
    ):
        return False

    linear_speed = float(metadata.get("linear_speed", math.inf))
    angular_speed = float(metadata.get("angular_speed", math.inf))
    linear_limit = float(metadata.get("linear_velocity_tolerance", 0.0)) + (
        _NAV_HANDOFF_SETTLE_LINEAR_SPEED_MARGIN_MPS
    )
    angular_limit = float(metadata.get("angular_velocity_tolerance", 0.0)) + (
        _NAV_HANDOFF_SETTLE_ANGULAR_SPEED_MARGIN_RADPS
    )
    return (
        math.isfinite(linear_speed)
        and math.isfinite(angular_speed)
        and linear_speed <= linear_limit
        and angular_speed <= angular_limit
    )


def _vector_norm(values: tuple[float, ...] | list[float], *, limit: int | None = None) -> float:
    subset = values[:limit] if limit is not None else values
    return math.sqrt(sum(float(value) * float(value) for value in subset))


def _max_abs(values: tuple[float, ...] | list[float]) -> float | None:
    if not values:
        return None
    return max(abs(float(value)) for value in values)


def _roll_pitch_from_wxyz(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float]:
    """从 wxyz 四元数计算底盘 roll 和 pitch。"""

    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(pitch_term)


def _yaw_from_wxyz(quaternion: tuple[float, float, float, float]) -> float:
    """从 wxyz 四元数计算底盘 yaw。"""

    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
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


def _append_pick_reverse_return_home(
    plan: ArmPlan,
    *,
    hold_duration_s: float,
) -> tuple[ArmPlan, dict[str, Any]]:
    """按 main 语义反向回放已规划 motion，避免无避障直连 all-zero home。"""

    raw_segments = plan.metadata.get("segments")
    if not isinstance(raw_segments, list | tuple):
        return plan, {"inserted": False, "reason": "plan_segments_unavailable"}

    motion_segments = [
        segment
        for segment in raw_segments
        if str(segment.get("type")) == "motion"
        and isinstance(segment.get("trajectory"), dict)
        and segment["trajectory"].get("q")
    ]
    if not motion_segments:
        return plan, {"inserted": False, "reason": "plan_has_no_motion_segments"}

    reverse_segments: list[dict[str, Any]] = []
    for segment in reversed(motion_segments):
        trajectory = segment["trajectory"]
        q_rows = tuple(
            tuple(float(value) for value in row)
            for row in trajectory["q"]
        )
        raw_times = tuple(float(value) for value in trajectory.get("time_from_start", ()))
        if len(raw_times) != len(q_rows) or not raw_times:
            raw_times = tuple(0.05 * index for index in range(len(q_rows)))
        duration = max(0.0, raw_times[-1] - raw_times[0])
        reverse_times = tuple(duration - (value - raw_times[0]) for value in reversed(raw_times))
        reverse_trajectory: dict[str, Any] = {
            "time_from_start": reverse_times,
            "q": tuple(reversed(q_rows)),
        }
        raw_qd = trajectory.get("qd")
        if isinstance(raw_qd, list | tuple) and len(raw_qd) == len(q_rows):
            reverse_trajectory["qd"] = tuple(
                tuple(-float(value) for value in row)
                for row in reversed(raw_qd)
            )
        original_name = str(segment.get("name") or "motion")
        reverse_segments.append(
            {
                "name": f"return_home_reverse_{original_name}",
                "type": "motion",
                "target_name": "reverse_executed_motion",
                "timing": {
                    "duration_s": duration,
                    "num_waypoints": len(q_rows),
                    "generated_by": "full_physics_pipeline",
                },
                "final_error": {},
                "plan_info": {
                    "generated_return_home": True,
                    "return_home_strategy": "reverse_executed_motion",
                    "source_segment_name": original_name,
                },
                "trajectory": reverse_trajectory,
            }
        )

    final_home = tuple(float(value) for value in reverse_segments[-1]["trajectory"]["q"][-1])
    if hold_duration_s > 0.0:
        reverse_segments.append(
            {
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
                    "reason": "hold_reverse_executed_motion_final_target",
                },
                "trajectory": {
                    "time_from_start": (0.0, float(hold_duration_s)),
                    "q": (final_home, final_home),
                },
            }
        )

    segments = (*tuple(raw_segments), *tuple(reverse_segments))
    return_rows = tuple(
        tuple(float(value) for value in row)
        for segment in reverse_segments
        for row in segment["trajectory"]["q"]
    )
    report = {
        "inserted": True,
        "strategy": "reverse_executed_motion",
        "segment_names": tuple(segment["name"] for segment in reverse_segments),
        "source_segment_names": tuple(
            str(segment.get("name") or "motion") for segment in motion_segments
        ),
        "joint_names": tuple(str(name) for name in plan.metadata.get("joint_names") or ()),
        "home_positions": final_home,
        "home_hold_inserted": hold_duration_s > 0.0,
        "home_hold_duration_s": float(hold_duration_s),
    }
    return (
        ArmPlan(
            operation=plan.operation,
            joint_trajectory=(
                *tuple(tuple(float(value) for value in row) for row in plan.joint_trajectory),
                *return_rows,
            ),
            metadata={
                **plan.metadata,
                "segments": segments,
                "pick_return_home": report,
                "original_trajectory_points": len(plan.joint_trajectory),
            },
        ),
        report,
    )


def _pick_plan_terminal_motion_names(plan: ArmPlan) -> tuple[str, ...]:
    raw_segments = plan.metadata.get("segments")
    if not isinstance(raw_segments, list | tuple):
        return ()
    return tuple(
        str(segment.get("name") or "")
        for segment in raw_segments
        if str(segment.get("type")) == "motion"
    )


def _target_vs_pre_execution_object_drift(
    *,
    plan: ArmPlan,
    simulation: SimulationRuntime,
    observation: SimulationState,
    max_allowed_drift_m: float = 0.012,
) -> dict[str, Any]:
    """对比 target 生成时 bbox 与 EXEC_PICK 前 live bbox。"""

    replan_report = plan.metadata.get("current_state_replan")
    export_report = (
        replan_report.get("export_report")
        if isinstance(replan_report, dict)
        else None
    )
    target_bbox = (
        export_report.get("bbox_world")
        if isinstance(export_report, dict)
        else None
    )
    target_center = (
        target_bbox.get("center_xyz")
        if isinstance(target_bbox, dict)
        else None
    )
    if not isinstance(target_center, list | tuple) or len(target_center) < 3:
        return {
            "available": False,
            "reason": "target_bbox_center_unavailable",
            "max_allowed_drift_m": float(max_allowed_drift_m),
        }

    current_bbox = None
    bbox_reader = getattr(simulation, "read_object_bbox_world", None)
    if callable(bbox_reader):
        try:
            current_bbox = bbox_reader()
        except Exception as exc:
            current_bbox = {"read_error": str(exc)}
    current_center = (
        current_bbox.get("center_xyz")
        if isinstance(current_bbox, dict)
        else None
    )
    current_source = "runtime_live_bbox"
    if not isinstance(current_center, list | tuple) or len(current_center) < 3:
        if observation.object_pose is None:
            return {
                "available": False,
                "reason": "current_object_bbox_center_unavailable",
                "target_bbox_center_xyz": tuple(float(value) for value in target_center[:3]),
                "current_bbox_report": current_bbox,
                "max_allowed_drift_m": float(max_allowed_drift_m),
            }
        current_center = observation.object_pose[:3]
        current_source = "observation_object_pose_fallback"

    target_xyz = tuple(float(value) for value in target_center[:3])
    current_xyz = tuple(float(value) for value in current_center[:3])
    delta = tuple(current - target for current, target in zip(current_xyz, target_xyz))
    drift_m = math.sqrt(sum(value * value for value in delta))
    xy_drift_m = math.hypot(delta[0], delta[1])
    return {
        "available": True,
        "target_bbox_center_xyz": target_xyz,
        "current_bbox_center_xyz": current_xyz,
        "current_bbox_center_source": current_source,
        "delta_xyz": delta,
        "drift_m": drift_m,
        "xy_drift_m": xy_drift_m,
        "max_allowed_drift_m": float(max_allowed_drift_m),
        "within_tolerance": drift_m <= float(max_allowed_drift_m),
        "current_bbox_report": current_bbox,
    }


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
        self.pick_planner_result: dict[str, Any] = {}
        self.pick_executor_status: dict[str, Any] = {}
        self.pick_verification_result: dict[str, Any] = {}
        self.place_verification_result: dict[str, Any] = {}
        self.export_result: dict[str, Any] = {}
        self._carry_gripper_target: dict[str, Any] | None = None
        self._carry_arm_home_target: dict[str, Any] | None = None
        self._carry_object_tcp_offset: tuple[float, float, float] | None = None
        self._pick_peak_object_lift_height_m: float | None = None
        self._pick_peak_object_pose: tuple[float, ...] | None = None
        self._pick_peak_step_index: int | None = None
        self._place_opening_started = False
        self._place_opening_step_index: int | None = None
        self._place_opening_object_pose: tuple[float, ...] | None = None
        self._place_expected_object_tcp_offset: tuple[float, float, float] | None = None
        self._place_pre_release_object_tcp_offset_report: dict[str, Any] = {}
        self._place_release_observed = False
        self._place_release_step_index: int | None = None
        self._place_release_object_pose: tuple[float, ...] | None = None
        self._place_release_gripper_open_progress: float | None = None
        self._place_expected_release_object_center: tuple[float, float, float] | None = None
        self._place_expected_release_center_source = "unavailable"
        self._place_open_apply_count_baseline = 0
        self._place_release_velocity_sample_count = 0
        self._place_peak_object_linear_speed_mps: float | None = None
        self._place_peak_object_horizontal_speed_mps: float | None = None
        self._place_peak_object_upward_speed_mps: float | None = None
        self._place_peak_object_downward_speed_mps: float | None = None
        self._place_peak_object_angular_speed_rps: float | None = None
        self._place_max_horizontal_displacement_m: float | None = None
        self._episode_reset_applied = False
        self._object_settle_elapsed_steps = 0
        self._object_settle_stable_steps = 0
        self._object_settle_completed = False
        self._initialization_base_lock_released = False
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
        if self.state.terminal and self._physical_pick_enabled():
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
            "pick_planner_result": dict(self.pick_planner_result),
            "pick_executor_status": dict(self.pick_executor_status),
            "pick_verification_result": dict(self.pick_verification_result),
            "place_verification_result": dict(self.place_verification_result),
            "lerobot_export": dict(self.export_result),
            "carry_gripper_target": dict(self._carry_gripper_target or {}),
            "carry_arm_home_target": dict(self._carry_arm_home_target or {}),
            "pick_peak_object_lift_height_m": self._pick_peak_object_lift_height_m,
            "pick_peak_object_pose": self._pick_peak_object_pose,
            "pick_peak_step_index": self._pick_peak_step_index,
            "place_opening_started": self._place_opening_started,
            "place_opening_step_index": self._place_opening_step_index,
            "place_pre_release_object_tcp_offset_report": dict(
                self._place_pre_release_object_tcp_offset_report
            ),
            "place_release_observed": self._place_release_observed,
            "place_release_step_index": self._place_release_step_index,
            "place_release_object_pose": self._place_release_object_pose,
            "place_peak_object_linear_speed_mps": self._place_peak_object_linear_speed_mps,
            "place_peak_object_horizontal_speed_mps": (
                self._place_peak_object_horizontal_speed_mps
            ),
            "place_peak_object_upward_speed_mps": (
                self._place_peak_object_upward_speed_mps
            ),
            "place_peak_object_downward_speed_mps": (
                self._place_peak_object_downward_speed_mps
            ),
            "place_peak_object_angular_speed_rps": self._place_peak_object_angular_speed_rps,
            "place_max_horizontal_displacement_m": (
                self._place_max_horizontal_displacement_m
            ),
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
        prepare_episode = getattr(self.simulation, "prepare_episode", None)
        if callable(prepare_episode):
            prepare_report = dict(prepare_episode(self.episode_spec))
            event_name = (
                "stage_reused"
                if prepare_report.get("stage_reused") is True
                else "stage_built"
            )
            events = [
                self._event(event_name, observation.step_index, prepare_report)
            ]
        else:
            self.simulation.build(self.episode_spec)
            events = [self._event("stage_built", observation.step_index)]
        events.extend(self._transition(PipelineState.RESET_EPISODE, observation.step_index))
        metadata = {}
        if self._physical_pick_enabled():
            # Stage 创建后必须先执行 episode reset，再允许首个物理步。
            # 否则动态苹果会在 reset/sleep 之前因重力和接触产生角速度。
            metadata = {
                "skip_physics_step": True,
                "skip_reason": "await_object_pose_reset_before_first_physics_step",
            }
        return RobotAction(source="stage_build", metadata=metadata), events

    def _reset_episode(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events: list[PipelineEvent] = []
        if not self._episode_reset_applied:
            self.simulation.reset(self.episode_spec, seed=self.episode_seed)
            self._episode_reset_applied = True
            events.append(self._event("episode_reset", observation.step_index))
            if self._object_settle_enabled():
                begin_settle = getattr(self.simulation, "begin_object_settle", None)
                if not callable(begin_settle):
                    return RobotAction.idle(source="object_settle"), self._fail(
                        "object_settle_unsupported",
                        self.simulation.read(),
                    )
                begin_report = dict(begin_settle(self.episode_spec))
                if begin_report.get("applied") is not True:
                    return RobotAction.idle(source="object_settle"), self._fail(
                        "object_settle_start_failed",
                        self.simulation.read(),
                        begin_report,
                    )
                events.append(
                    self._event(
                        "object_settle_started",
                        observation.step_index,
                        begin_report,
                    )
                )
                return RobotAction(
                    source="object_settle",
                    metadata={
                        "object_settle_active": True,
                        "manipulation_base_lock": bool(
                            self.config.manipulation.settle_base_before_navigation
                            and self.config.manipulation.initialization_base_lock_steps > 0
                        ),
                        "manipulation_base_lock_phase": (
                            "episode_initialization_settle"
                            if (
                                self.config.manipulation.settle_base_before_navigation
                                and self.config.manipulation.initialization_base_lock_steps
                                > 0
                            )
                            else None
                        ),
                        # reset 后先用 actuator target 保持四足支撑姿态，避免脚尚未
                        # 建立接触时 policy/碰撞冲击把腿折叠并触发 base_settle_timeout。
                        # 该锁只写 position/velocity target，不直接改写关节状态；
                        # initialization_base_lock_steps 结束后会与 root lock 一起解除，
                        # 将支撑腿交还 RL policy，并在真实动力学下重新累计稳定步数。
                        "manipulation_support_joint_lock": True,
                        "manipulation_support_joint_lock_phase": (
                            "episode_initialization_settle"
                        ),
                        # Observe and record the exact post-reset state before the
                        # first task physics step.  This makes reset regressions
                        # distinguishable from failures caused by the first action.
                        "skip_physics_step": True,
                        "skip_reason": "audit_post_reset_state_before_first_physics_step",
                    },
                ), events
        if self._object_settle_enabled() and not self._object_settle_completed:
            settle_action, settle_events, settled = self._advance_object_settle(observation)
            events.extend(settle_events)
            if not settled:
                return settle_action, events
        if self._physical_pick_enabled():
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

    def _advance_object_settle(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent], bool]:
        settings = self.config.manipulation
        self._object_settle_elapsed_steps += 1
        initialization_base_lock_active = bool(
            settings.settle_base_before_navigation
            and settings.initialization_base_lock_steps > 0
            and self._object_settle_elapsed_steps
            <= settings.initialization_base_lock_steps
        )
        # 支撑腿只在 root 被显式固定时保持默认关节目标。root 释放后必须把
        # 12 个腿关节交还 rough-terrain RL policy；否则固定站姿无法补偿局部
        # 地面坡度，某些 yaw 会缓慢侧翻。配置为 0 时保留旧的全程支撑锁语义。
        initialization_support_joint_lock_active = bool(
            initialization_base_lock_active
            or settings.initialization_base_lock_steps <= 0
        )
        if (
            not initialization_base_lock_active
            and settings.initialization_base_lock_steps > 0
            and not self._initialization_base_lock_released
        ):
            # 锁定期间的零速度是人为约束结果，不能作为进入导航的依据。
            # 从本 tick 开始释放 root，并重新累计 RL/真实接触下的稳定步数。
            self._initialization_base_lock_released = True
            self._object_settle_stable_steps = 0
        pose = observation.object_pose
        velocity = observation.object_velocity
        if pose is None or velocity is None or len(velocity) < 6:
            events = self._fail(
                "object_settle_state_unavailable",
                observation,
                {
                    "object_pose_available": pose is not None,
                    "object_velocity_available": velocity is not None,
                },
            )
            return RobotAction.idle(source="object_settle"), events, False

        linear_speed = _vector_norm(tuple(velocity[:3]))
        angular_speed = _vector_norm(tuple(velocity[3:6]))
        expected_position = self.episode_spec.object_initial_pose
        displacement = (
            _vector_norm(
                [
                    float(pose[index]) - float(expected_position[index])
                    for index in range(3)
                ]
            )
            if expected_position is not None
            else 0.0
        )
        initialization_policy = resolve_object_initialization_policy(
            self.episode_spec.raw_task
        )
        initialization_pose_validation: dict[str, Any] | None = None
        requested_position: tuple[float, float, float] | None = None
        requested_quaternion: tuple[float, float, float, float] | None = None
        if initialization_policy.get("enabled"):
            if expected_position is None or len(pose) < 7:
                events = self._fail(
                    "object_initialization_pose_unavailable",
                    observation,
                    {
                        "expected_position_available": expected_position is not None,
                        "actual_pose_length": len(pose),
                        "initialization_policy": initialization_policy,
                    },
                )
                return RobotAction.idle(source="object_settle"), events, False
            requested_position = tuple(
                float(expected_position[index]) for index in range(3)
            )
            pose_setup_report = observation.metadata.get("object_pose_setup_report")
            pose_setup_report = (
                pose_setup_report if isinstance(pose_setup_report, dict) else {}
            )
            authored_quaternion = pose_setup_report.get(
                "authored_world_quaternion_wxyz"
            )
            if isinstance(authored_quaternion, (list, tuple)) and len(
                authored_quaternion
            ) >= 4:
                requested_quaternion = tuple(
                    float(authored_quaternion[index]) for index in range(4)
                )
            else:
                requested_quaternion = _quat_wxyz_from_rpy(
                    float(expected_position[3]),
                    float(expected_position[4]),
                    float(expected_position[5]),
                )
            initialization_pose_validation = evaluate_object_initialization_pose(
                policy=initialization_policy,
                requested_position_xyz=requested_position,
                requested_quaternion_wxyz=requested_quaternion,
                actual_pose_xyz_wxyz=pose,
            )
        stable = bool(
            linear_speed <= settings.object_settle_linear_velocity_mps
            and angular_speed <= settings.object_settle_angular_velocity_rps
        )
        root_velocity = observation.robot_root_velocity
        base_linear_speed = _vector_norm(tuple(root_velocity[:3]))
        base_angular_speed = _vector_norm(tuple(root_velocity[3:6]))
        base_roll, base_pitch = _roll_pitch_from_wxyz(
            tuple(float(value) for value in observation.robot_root_pose[3:7])
        )
        base_stable = bool(
            base_linear_speed <= settings.base_settle_linear_velocity_mps
            and base_angular_speed <= settings.base_settle_angular_velocity_rps
            and abs(base_roll) <= settings.base_settle_max_tilt_rad
            and abs(base_pitch) <= settings.base_settle_max_tilt_rad
        )
        if settings.settle_base_before_navigation:
            stable = stable and base_stable
        if initialization_base_lock_active:
            stable = False
        self._object_settle_stable_steps = (
            self._object_settle_stable_steps + 1 if stable else 0
        )
        report = {
            "elapsed_steps": self._object_settle_elapsed_steps,
            "stable_steps": self._object_settle_stable_steps,
            "required_stable_steps": settings.object_settle_required_stable_steps,
            "linear_speed_mps": linear_speed,
            "angular_speed_rps": angular_speed,
            "displacement_from_task_pose_m": displacement,
            "current_pose": tuple(float(value) for value in pose),
            "base_settle_enabled": settings.settle_base_before_navigation,
            "base_stable": base_stable,
            "base_linear_speed_mps": base_linear_speed,
            "base_angular_speed_rps": base_angular_speed,
            "base_roll_rad": base_roll,
            "base_pitch_rad": base_pitch,
            "initialization_base_lock_steps": (
                settings.initialization_base_lock_steps
            ),
            "initialization_base_lock_active": initialization_base_lock_active,
            "initialization_base_lock_released": (
                self._initialization_base_lock_released
            ),
            "initialization_support_joint_lock_active": (
                initialization_support_joint_lock_active
            ),
            "initialization_pose_validation": initialization_pose_validation,
        }
        if (
            initialization_pose_validation is not None
            and initialization_policy.get("required_for_episode")
            and initialization_pose_validation.get("verified") is not True
        ):
            events = self._fail(
                "object_initialization_pose_invalid",
                observation,
                report,
            )
            return RobotAction.idle(source="object_settle"), events, False
        if displacement > settings.object_settle_max_displacement_m:
            events = self._fail(
                "object_settle_out_of_bounds",
                observation,
                report,
            )
            return RobotAction.idle(source="object_settle"), events, False
        if (
            self._object_settle_stable_steps
            < settings.object_settle_required_stable_steps
        ):
            if self._object_settle_elapsed_steps >= settings.object_settle_max_steps:
                failure_reason = (
                    "base_settle_timeout"
                    if settings.settle_base_before_navigation and not base_stable
                    else "object_settle_timeout"
                )
                events = self._fail(
                    failure_reason,
                    observation,
                    report,
                )
                return RobotAction.idle(source="object_settle"), events, False
            return RobotAction(
                source="object_settle",
                metadata={
                    "object_settle_active": True,
                    "object_settle_report": report,
                    "manipulation_base_lock": initialization_base_lock_active,
                    "manipulation_base_lock_phase": (
                        "episode_initialization_settle"
                        if initialization_base_lock_active
                        else None
                    ),
                    "manipulation_support_joint_lock": (
                        initialization_support_joint_lock_active
                    ),
                    "manipulation_support_joint_lock_phase": (
                        "episode_initialization_settle"
                        if initialization_support_joint_lock_active
                        else None
                    ),
                },
            ), [], False

        finalize_settle = getattr(self.simulation, "finalize_object_settle", None)
        if not callable(finalize_settle):
            events = self._fail(
                "object_settle_unsupported",
                observation,
                report,
            )
            return RobotAction.idle(source="object_settle"), events, False
        final_report = dict(finalize_settle(self.episode_spec))
        final_report["stability"] = report
        if initialization_policy.get("enabled"):
            final_pose = final_report.get("settled_pose")
            final_pose = (
                final_pose
                if isinstance(final_pose, (list, tuple)) and len(final_pose) >= 7
                else pose
            )
            final_validation = evaluate_object_initialization_pose(
                policy=initialization_policy,
                requested_position_xyz=requested_position or (),
                requested_quaternion_wxyz=requested_quaternion or (),
                actual_pose_xyz_wxyz=final_pose,
            )
            final_report["initialization_pose_validation"] = final_validation
            if (
                initialization_policy.get("required_for_episode")
                and final_validation.get("verified") is not True
            ):
                events = self._fail(
                    "object_initialization_pose_invalid",
                    observation,
                    final_report,
                )
                return RobotAction.idle(source="object_settle"), events, False
        if final_report.get("applied") is not True:
            events = self._fail(
                "object_settle_finalize_failed",
                observation,
                final_report,
            )
            return RobotAction.idle(source="object_settle"), events, False
        self._object_settle_completed = True
        return (
            RobotAction.idle(source="object_settle_complete"),
            [
                self._event(
                    "object_initial_pose_stabilized",
                    observation.step_index,
                    final_report,
                )
            ],
            True,
        )

    def _plan_nav_to_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        events = [self._event("nav_to_pick_start", observation.step_index)]
        plan = self.nav_planner.plan(observation, self.episode_spec.pick_goal)
        execution_metadata = _navigation_plan_execution_metadata(
            self.episode_spec.raw_task,
            include_carry_departure=False,
        )
        plan = replace(
            plan,
            metadata={
                **plan.metadata,
                **execution_metadata,
                "execution_phase": "nav_to_pick",
            },
        )
        self.nav_executor.reset(plan)
        self.latest_planner_result = {
            "type": "navigation",
            "phase": "pick",
            "waypoint_count": len(plan.waypoints),
            **plan.metadata,
        }
        visualization_event = self._visualize_planned_trajectory(
            plan,
            trajectory_type="navigation",
            phase="pick",
            step_index=observation.step_index,
        )
        if visualization_event is not None:
            events.append(visualization_event)
        events.extend(self._transition(PipelineState.EXEC_NAV_TO_PICK, observation.step_index))
        return RobotAction.idle(source="nav_plan_pick"), events

    def _exec_nav_to_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        return self._execute_nav(observation, PipelineState.VERIFY_PICK_REACHABLE)

    def _verify_pick_reachable(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_pick_reachable(observation, self.episode_spec)
        result = _accept_successful_navigation_handoff_drift(
            result,
            self.latest_executor_status,
            phase="pick",
        )
        if not result.success:
            settle_result = self._settle_navigation_handoff(
                result,
                observation,
                phase="pick",
            )
            if settle_result is not None:
                return settle_result
            handoff_status = dict(self.latest_executor_status or {})
            if handoff_status.get("near_goal_stall_handoff") is True:
                result = VerificationResult(
                    success=True,
                    failure_reason="",
                    metadata={
                        **result.metadata,
                        "navigation_verifier_override": "near_goal_stall_handoff",
                        "near_goal_stall_handoff": True,
                        "executor_distance_to_goal": handoff_status.get(
                            "distance_to_goal"
                        ),
                        "executor_position_tolerance": handoff_status.get(
                            "position_tolerance"
                        ),
                        "executor_near_goal_stall_handoff_tolerance": (
                            handoff_status.get(
                                "near_goal_stall_handoff_tolerance"
                            )
                        ),
                    },
                )
            else:
                return RobotAction.idle(source="verify_pick_reachable"), self._fail(
                    result.failure_reason or "pick_target_unreachable",
                    observation,
                    result.metadata,
                )
        events = self._navigation_handoff_settle_complete_events(
            observation,
            phase="pick",
            result=result,
        )
        events.append(
            self._event("nav_to_pick_success", observation.step_index, result.metadata)
        )
        if self.config.navigation_smoke or self.config.stair_locomotion_smoke:
            success_event = (
                "stair_locomotion_smoke_success"
                if self.config.stair_locomotion_smoke
                else "navigation_smoke_success"
            )
            events.append(self._event(success_event, observation.step_index))
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            return RobotAction.idle(source="verify_pick_reachable"), events
        events.extend(self._transition(PipelineState.PLAN_PICK, observation.step_index))
        return RobotAction.idle(source="verify_pick_reachable"), events

    def _plan_pick(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        settle_result = self._settle_locked_base_before_plan(observation, phase="pick")
        if settle_result is not None:
            return settle_result

        events = [self._event("pick_plan_start", observation.step_index)]
        if self._physical_pick_enabled():
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
        terminal_motion_names = _pick_plan_terminal_motion_names(plan)
        has_lift_segment = "lift_object" in terminal_motion_names
        has_retreat_segment = "retreat_object" in terminal_motion_names
        protected_terminal_motion_names = tuple(
            name
            for name in terminal_motion_names
            if name == "retreat_object"
            or name == "return_home_after_retreat"
            or name.startswith("return_home_reverse_")
        )
        return_home_report = {
            "inserted": False,
            "reason": "disabled_by_default_main_pick_parity",
            "configured": bool(self.config.manipulation.return_home_after_pick),
            "plan_motion_segment_names": terminal_motion_names,
            "protected_terminal_motion_names": protected_terminal_motion_names,
        }
        if self.config.manipulation.return_home_after_pick:
            if protected_terminal_motion_names:
                return_home_report = {
                    **return_home_report,
                    "reason": "planned_retreat_or_reverse_return_present",
                }
            else:
                plan, return_home_report = _append_pick_reverse_return_home(
                    plan,
                    hold_duration_s=self.config.manipulation.pick_home_hold_duration_s,
                )
        pre_execution_observation = self.simulation.read()
        target_drift_report = _target_vs_pre_execution_object_drift(
            plan=plan,
            simulation=self.simulation,
            observation=pre_execution_observation,
        )
        plan.metadata["target_vs_pre_execution_object_drift"] = target_drift_report
        self._configure_carry_arm_home_target(plan, return_home_report)
        planner_result = {
            "type": "manipulation",
            "phase": "pick",
            "trajectory_points": len(plan.joint_trajectory),
            "pick_return_home": return_home_report,
            "target_vs_pre_execution_object_drift": target_drift_report,
            "pick_motion_segment_names": terminal_motion_names,
            "pick_has_lift_segment": has_lift_segment,
            "pick_has_retreat_segment": has_retreat_segment,
            **plan.metadata,
        }
        self.latest_planner_result = planner_result
        self.pick_planner_result = dict(planner_result)
        if (
            target_drift_report.get("available") is True
            and target_drift_report.get("within_tolerance") is not True
        ):
            return RobotAction.idle(source="pick_plan"), self._fail(
                "object_moved_after_target_generation",
                pre_execution_observation,
                target_drift_report,
            )
        visualization_event = self._visualize_planned_trajectory(
            plan,
            trajectory_type="manipulation",
            phase="pick",
            step_index=observation.step_index,
        )
        if visualization_event is not None:
            events.append(visualization_event)
        self.arm_executor.reset(plan)
        self._pick_peak_object_lift_height_m = None
        self._pick_peak_object_pose = None
        self._pick_peak_step_index = None
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
        self._update_pick_peak_lift(observation)
        return self._execute_arm(observation, PipelineState.VERIFY_PICK_SUCCESS)

    def _verify_pick_success(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        self._update_pick_peak_lift(observation)
        verification_observation = replace(
            observation,
            metadata={
                **observation.metadata,
                "pick_peak_object_lift_height_m": self._pick_peak_object_lift_height_m,
                "pick_peak_object_pose": self._pick_peak_object_pose,
                "pick_peak_step_index": self._pick_peak_step_index,
                "pick_motion_segment_names": tuple(
                    self.pick_planner_result.get("pick_motion_segment_names") or ()
                ),
                "pick_has_lift_segment": bool(
                    self.pick_planner_result.get("pick_has_lift_segment", True)
                ),
                "pick_has_retreat_segment": bool(
                    self.pick_planner_result.get("pick_has_retreat_segment", False)
                ),
            },
        )
        result = self.verifier.verify_pick_success(
            verification_observation,
            self.episode_spec,
        )
        self.pick_verification_result = {
            "success": bool(result.success),
            "failure_reason": result.failure_reason,
            **result.metadata,
        }
        if not result.success:
            return RobotAction.idle(source="verify_pick_success"), self._fail(
                result.failure_reason or "grasp_failed",
                observation,
                result.metadata,
            )
        pick_success_event = (
            "pick_success"
            if self._physical_pick_enabled()
            else (
                "manipulation_apply_smoke_pick_apply_success"
                if self.config.manipulation_apply_smoke
                else "object_lift_success"
            )
        )
        events = [self._event(pick_success_event, observation.step_index, result.metadata)]
        self._capture_verified_carry_gripper_preload(observation)
        self._capture_carry_object_tcp_offset(observation)
        if self.config.pick_smoke:
            events.append(self._event("pick_smoke_success", observation.step_index, result.metadata))
            events.extend(self._transition(PipelineState.CLEANUP_EPISODE, observation.step_index))
            return RobotAction.idle(source="verify_pick_success"), events
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
        execution_metadata = _navigation_plan_execution_metadata(
            self.episode_spec.raw_task,
            include_carry_departure=True,
        )
        plan = replace(
            plan,
            metadata={
                **plan.metadata,
                **execution_metadata,
                "execution_phase": "carry_nav_to_place",
                "require_yaw_alignment": True,
                "yaw_tolerance": self.config.navigation.final_yaw_tolerance,
            },
        )
        self.nav_executor.reset(plan)
        self.latest_planner_result = {
            "type": "navigation",
            "phase": "place",
            "waypoint_count": len(plan.waypoints),
            **plan.metadata,
        }
        visualization_event = self._visualize_planned_trajectory(
            plan,
            trajectory_type="navigation",
            phase="place",
            step_index=observation.step_index,
        )
        if visualization_event is not None:
            events.append(visualization_event)
        events.extend(self._transition(PipelineState.EXEC_NAV_TO_PLACE, observation.step_index))
        return RobotAction.idle(source="nav_plan_place"), events

    def _exec_nav_to_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.config.full_physics:
            raw_carry = self.episode_spec.raw_task.get("carry")
            interval_steps = 10
            if isinstance(raw_carry, dict):
                interval_steps = max(
                    1,
                    int(raw_carry.get("verify_carry_every_steps", interval_steps)),
                )
            if observation.step_index % interval_steps == 0:
                carry_check = self._verify_carry_object_tracking(observation)
                if not carry_check["success"]:
                    return RobotAction.idle(source="exec_nav_to_place"), self._fail(
                        str(carry_check["failure_reason"]),
                        observation,
                        {
                            **carry_check,
                            "carry_verify_interval_steps": interval_steps,
                        },
                    )
        return self._execute_nav(observation, PipelineState.VERIFY_PLACE_REACHABLE)

    def _verify_place_reachable(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        result = self.verifier.verify_place_reachable(observation, self.episode_spec)
        result = _accept_successful_navigation_handoff_drift(
            result,
            self.latest_executor_status,
            phase="place",
        )
        if not result.success:
            settle_result = self._settle_navigation_handoff(
                result,
                observation,
                phase="place",
            )
            if settle_result is not None:
                return settle_result
            return RobotAction.idle(source="verify_place_reachable"), self._fail(
                result.failure_reason or "place_target_unreachable",
                observation,
                result.metadata,
            )
        events = self._navigation_handoff_settle_complete_events(
            observation,
            phase="place",
            result=result,
        )
        events.append(
            self._event("nav_to_place_success", observation.step_index, result.metadata)
        )
        if self.config.navigation_carry_smoke or self.config.full_physics:
            if self.config.full_physics:
                object_carry_check = self._verify_carry_object_tracking(observation)
                if not object_carry_check["success"]:
                    return RobotAction.idle(source="verify_place_reachable"), self._fail(
                        str(object_carry_check["failure_reason"]),
                        observation,
                        object_carry_check,
                    )
            carry_check = self._verify_navigation_carry_targets(
                observation,
                latest_sample_only=self.config.full_physics,
            )
            if not carry_check["success"]:
                return RobotAction.idle(source="verify_place_reachable"), self._fail(
                    str(carry_check["failure_reason"]),
                    observation,
                    carry_check,
                )
            if self.config.full_physics:
                events.append(
                    self._event(
                        "carry_control_success",
                        observation.step_index,
                        carry_check,
                    )
                )
                events.extend(self._transition(PipelineState.PLAN_PLACE, observation.step_index))
                return self._place_carry_handoff_action(observation), events
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

    def _place_carry_handoff_action(self, observation: SimulationState) -> RobotAction:
        """进入 place 规划前继续保持 carry 姿态，避免交接帧释放夹持。"""

        if not self.config.navigation.pct_stair_float_enabled:
            return RobotAction.idle(source="verify_place_reachable")
        root_pose = tuple(float(value) for value in observation.robot_root_pose)
        yaw = _yaw_from_wxyz(root_pose[3:7])  # type: ignore[arg-type]
        root_xyzyaw = (
            root_pose[0],
            root_pose[1],
            root_pose[2],
            yaw,
        )
        metadata = {
            "place_carry_handoff_hold": True,
            "place_carry_handoff_reason": "preserve_tcp_object_before_place_plan",
            "navigation_base_pose_lock": True,
            "navigation_base_pose_lock_phase": "place_carry_handoff",
            "navigation_base_pose_lock_xyzyaw": root_xyzyaw,
            "navigation_support_joint_lock": True,
            "navigation_support_joint_lock_phase": "place_carry_handoff",
            "navigation_full_body_joint_lock": True,
            "navigation_full_body_joint_lock_phase": "place_carry_handoff",
            "navigation_carry_object_follow": True,
        }
        return RobotAction(source="place_carry_handoff", metadata=metadata)

    def _plan_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        if self.config.full_physics:
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
            error_report = {
                "type": "manipulation",
                "phase": "place",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            self.latest_planner_result = error_report
            return RobotAction.idle(source="place_plan"), self._fail(
                "place_plan_failed",
                observation,
                error_report,
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
        plan, stable_release_report = self._configure_stable_place_release(plan)
        self.latest_planner_result = {
            **self.latest_planner_result,
            "trajectory_points": len(plan.joint_trajectory),
            "stable_place_release": stable_release_report,
            **plan.metadata,
        }
        self._reset_place_release_tracking(observation, plan)
        visualization_event = self._visualize_planned_trajectory(
            plan,
            trajectory_type="manipulation",
            phase="place",
            step_index=observation.step_index,
        )
        if visualization_event is not None:
            events.append(visualization_event)
        self.arm_executor.reset(plan)
        events.extend(self._transition(PipelineState.EXEC_PLACE, observation.step_index))
        events.append(self._event("place_execute_start", observation.step_index))
        return RobotAction.idle(source="place_plan"), events

    def _exec_place(self, observation: SimulationState) -> tuple[RobotAction, list[PipelineEvent]]:
        self._update_place_release_tracking(observation)
        action, events = self._execute_arm(
            observation,
            PipelineState.VERIFY_PLACE_SUCCESS,
        )
        if action.metadata.get("event_marker") == "gripper_open":
            offset_report = self._place_object_tcp_offset_report(observation)
            self._place_pre_release_object_tcp_offset_report = offset_report
            if (
                observation.metadata.get("execution_provenance_verified") is True
                and not offset_report.get("within_tolerance", False)
            ):
                events = [event for event in events if event.name != "gripper_open"]
                events.extend(
                    self._fail(
                        "place_object_slipped_before_release",
                        observation,
                        offset_report,
                    )
                )
                return RobotAction.idle(source="place_release_blocked"), events
            self._begin_place_opening_tracking(observation)
        if (
            not self._place_release_observed
            and action.metadata.get("segment_name") == "open_gripper"
            and action.metadata.get("gripper_phase") == "hold"
        ):
            open_progress = _gripper_motion_progress(observation, action)
            if open_progress is not None and open_progress >= 0.80:
                self._mark_place_release_observed(observation, open_progress)
                events.append(
                    self._event(
                        "place_release_observed",
                        observation.step_index,
                        {
                            "gripper_open_progress": open_progress,
                            "observation_timing": "after_open_move_before_hold_action",
                        },
                    )
                )
        return action, events

    def _configure_stable_place_release(
        self,
        plan: ArmPlan,
    ) -> tuple[ArmPlan, dict[str, Any]]:
        """把 pick 验证后的轻夹持目标和任务等待时间写入 place plan。"""

        metadata = dict(plan.metadata)
        raw_place = dict((self.episode_spec.raw_task or {}).get("place") or {})
        settle_steps = max(0, int(raw_place.get("settle_steps", 0) or 0))
        metadata["place_release_settle_steps"] = settle_steps

        carry_target = dict(self._carry_gripper_target or {})
        joint_names = tuple(
            str(value) for value in carry_target.get("gripper_joint_names") or ()
        )
        joint_positions = tuple(
            float(value)
            for value in carry_target.get("gripper_joint_positions") or ()
        )
        inherited = bool(
            joint_names
            and len(joint_names) == len(joint_positions)
            and all(math.isfinite(value) for value in joint_positions)
        )
        hold_source = str(
            carry_target.get("hold_position_source") or "carry_target_unavailable"
        )
        if inherited:
            metadata["place_closed_gripper_hold"] = {
                "gripper_joint_names": joint_names,
                "gripper_joint_positions": joint_positions,
                "source": hold_source,
            }

        report = {
            "carry_preload_inherited": inherited,
            "carry_preload_source": hold_source,
            "gripper_joint_names": joint_names,
            "gripper_joint_positions": joint_positions,
            "release_settle_steps": settle_steps,
            "release_settle_duration_s": settle_steps * 0.02,
            "release_clearance_min_m": (
                self.config.manipulation.place_release_clearance_min_m
            ),
            "release_joint_error_tolerance": (
                self.config.manipulation.place_release_joint_error_tolerance
            ),
            "release_joint_velocity_tolerance": (
                self.config.manipulation.place_release_joint_velocity_tolerance
            ),
            "release_object_tcp_offset_tolerance_m": (
                self.config.manipulation.place_release_object_tcp_offset_tolerance_m
            ),
        }
        return replace(plan, metadata=metadata), report

    def _reset_place_release_tracking(
        self,
        observation: SimulationState,
        plan: ArmPlan,
    ) -> None:
        self.place_verification_result = {}
        self._place_opening_started = False
        self._place_opening_step_index = None
        self._place_opening_object_pose = None
        self._place_expected_object_tcp_offset = _object_tcp_offset(observation)
        self._place_pre_release_object_tcp_offset_report = {}
        self._place_release_observed = False
        self._place_release_step_index = None
        self._place_release_object_pose = None
        self._place_release_gripper_open_progress = None
        self._place_open_apply_count_baseline = int(
            observation.metadata.get("gripper_open_apply_count", 0) or 0
        )
        self._place_release_velocity_sample_count = 0
        self._place_peak_object_linear_speed_mps = None
        self._place_peak_object_horizontal_speed_mps = None
        self._place_peak_object_upward_speed_mps = None
        self._place_peak_object_downward_speed_mps = None
        self._place_peak_object_angular_speed_rps = None
        self._place_max_horizontal_displacement_m = None

        current_state_replan = plan.metadata.get("current_state_replan")
        export_report = (
            current_state_replan.get("export_report")
            if isinstance(current_state_replan, dict)
            else None
        )
        release_center = (
            export_report.get("release_object_center_world")
            if isinstance(export_report, dict)
            else None
        )
        parsed_center = _finite_xyz(release_center)
        if parsed_center is not None:
            self._place_expected_release_object_center = parsed_center
            self._place_expected_release_center_source = "current_state_curobo_export"
            return
        fallback = _finite_xyz(self.episode_spec.place_target_pose)
        self._place_expected_release_object_center = fallback
        self._place_expected_release_center_source = (
            "episode_place_target" if fallback is not None else "unavailable"
        )

    def _place_object_tcp_offset_report(
        self,
        observation: SimulationState,
    ) -> dict[str, Any]:
        """检查 place 下放后物体是否仍保持规划时的 TCP 相对位置。"""

        expected = self._place_expected_object_tcp_offset
        actual = _object_tcp_offset(observation)
        tolerance = float(
            self.config.manipulation.place_release_object_tcp_offset_tolerance_m
        )
        if expected is None or actual is None:
            return {
                "available": False,
                "within_tolerance": False,
                "reason": "object_or_tcp_pose_unavailable",
                "expected_object_tcp_offset_xyz": expected,
                "actual_object_tcp_offset_xyz": actual,
                "tolerance_m": tolerance,
            }
        delta = tuple(
            actual_value - expected_value
            for actual_value, expected_value in zip(actual, expected)
        )
        drift = math.sqrt(sum(value * value for value in delta))
        return {
            "available": True,
            "within_tolerance": drift <= tolerance,
            "expected_object_tcp_offset_xyz": expected,
            "actual_object_tcp_offset_xyz": actual,
            "offset_delta_xyz": delta,
            "offset_drift_m": drift,
            "tolerance_m": tolerance,
        }

    def _begin_place_opening_tracking(self, observation: SimulationState) -> None:
        """从渐进开夹首帧开始统计动力学，但此时尚不宣告物理释放。"""

        if self._place_opening_started:
            return
        self._place_opening_started = True
        self._place_opening_step_index = int(observation.step_index)
        self._place_opening_object_pose = (
            tuple(float(value) for value in observation.object_pose)
            if observation.object_pose is not None
            else None
        )
        self._place_max_horizontal_displacement_m = 0.0
        self._update_place_release_tracking(observation)

    def _mark_place_release_observed(
        self,
        observation: SimulationState,
        gripper_open_progress: float,
    ) -> None:
        """夹爪达到足够开度后记录物理释放位姿。"""

        self._place_release_observed = True
        self._place_release_step_index = int(observation.step_index)
        self._place_release_object_pose = (
            tuple(float(value) for value in observation.object_pose)
            if observation.object_pose is not None
            else None
        )
        self._place_release_gripper_open_progress = float(gripper_open_progress)

    def _update_place_release_tracking(self, observation: SimulationState) -> None:
        if not self._place_opening_started:
            return
        if observation.object_velocity is not None and len(observation.object_velocity) >= 6:
            vx, vy, vz, wx, wy, wz = (
                float(value) for value in observation.object_velocity[:6]
            )
            if all(math.isfinite(value) for value in (vx, vy, vz, wx, wy, wz)):
                linear_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
                horizontal_speed = math.hypot(vx, vy)
                angular_speed = math.sqrt(wx * wx + wy * wy + wz * wz)
                self._place_release_velocity_sample_count += 1
                self._place_peak_object_linear_speed_mps = max(
                    self._place_peak_object_linear_speed_mps or 0.0,
                    linear_speed,
                )
                self._place_peak_object_horizontal_speed_mps = max(
                    self._place_peak_object_horizontal_speed_mps or 0.0,
                    horizontal_speed,
                )
                # 向下落座与向上弹射的物理含义不同，不能再只用三轴速度模长
                # 判断“ejected”。保留总速度用于诊断，并额外记录有方向的 Z 峰值。
                self._place_peak_object_upward_speed_mps = max(
                    self._place_peak_object_upward_speed_mps or 0.0,
                    max(vz, 0.0),
                )
                self._place_peak_object_downward_speed_mps = max(
                    self._place_peak_object_downward_speed_mps or 0.0,
                    max(-vz, 0.0),
                )
                self._place_peak_object_angular_speed_rps = max(
                    self._place_peak_object_angular_speed_rps or 0.0,
                    angular_speed,
                )
        if observation.object_pose is None or self._place_opening_object_pose is None:
            return
        horizontal_displacement = math.hypot(
            float(observation.object_pose[0]) - self._place_opening_object_pose[0],
            float(observation.object_pose[1]) - self._place_opening_object_pose[1],
        )
        self._place_max_horizontal_displacement_m = max(
            self._place_max_horizontal_displacement_m or 0.0,
            horizontal_displacement,
        )

    def _place_release_tracking_metadata(
        self,
        observation: SimulationState,
    ) -> dict[str, Any]:
        open_count = int(observation.metadata.get("gripper_open_apply_count", 0) or 0)
        return {
            "place_opening_started": self._place_opening_started,
            "place_opening_step_index": self._place_opening_step_index,
            "place_release_observed": self._place_release_observed,
            "place_release_step_index": self._place_release_step_index,
            "place_release_object_pose": self._place_release_object_pose,
            "place_release_gripper_open_progress": (
                self._place_release_gripper_open_progress
            ),
            "place_pre_release_object_tcp_offset_report": dict(
                self._place_pre_release_object_tcp_offset_report
            ),
            "place_expected_release_object_center": (
                self._place_expected_release_object_center
            ),
            "place_expected_release_center_source": (
                self._place_expected_release_center_source
            ),
            "place_open_apply_count_baseline": self._place_open_apply_count_baseline,
            "place_open_apply_count_delta": max(
                0,
                open_count - self._place_open_apply_count_baseline,
            ),
            "place_release_velocity_sample_count": (
                self._place_release_velocity_sample_count
            ),
            "place_peak_object_linear_speed_mps": (
                self._place_peak_object_linear_speed_mps
            ),
            "place_peak_object_horizontal_speed_mps": (
                self._place_peak_object_horizontal_speed_mps
            ),
            "place_peak_object_upward_speed_mps": (
                self._place_peak_object_upward_speed_mps
            ),
            "place_peak_object_downward_speed_mps": (
                self._place_peak_object_downward_speed_mps
            ),
            "place_peak_object_angular_speed_rps": (
                self._place_peak_object_angular_speed_rps
            ),
            "place_max_horizontal_displacement_m": (
                self._place_max_horizontal_displacement_m
            ),
        }

    def _visualize_planned_trajectory(
        self,
        plan: Any,
        *,
        trajectory_type: str,
        phase: str,
        step_index: int,
    ) -> PipelineEvent | None:
        """按显式开关向当前 USD stage 写入非物理规划轨迹。"""

        if not self.config.show_planned_trajectories:
            return None
        from source.diagnostics.planned_trajectories import (
            draw_manipulation_plan,
            draw_navigation_plan,
        )

        if trajectory_type == "navigation":
            report = draw_navigation_plan(plan, phase=phase)
        elif trajectory_type == "manipulation":
            report = draw_manipulation_plan(plan, phase=phase)
        else:
            report = {
                "available": False,
                "type": trajectory_type,
                "phase": phase,
                "reason": "unsupported_trajectory_type",
            }
        self.latest_planner_result = {
            **self.latest_planner_result,
            "trajectory_visualization": report,
        }
        return self._event("planned_trajectory_visualized", step_index, report)

    def _verify_place_success(
        self,
        observation: SimulationState,
    ) -> tuple[RobotAction, list[PipelineEvent]]:
        self._update_place_release_tracking(observation)
        verification_observation = replace(
            observation,
            metadata={
                **observation.metadata,
                **self._place_release_tracking_metadata(observation),
            },
        )
        result = self.verifier.verify_place_success(
            verification_observation,
            self.episode_spec,
        )
        self.place_verification_result = {
            "success": bool(result.success),
            "failure_reason": result.failure_reason,
            **result.metadata,
        }
        if not result.success:
            return RobotAction.idle(source="verify_place_success"), self._fail(
                result.failure_reason or "object_out_of_place",
                verification_observation,
                result.metadata,
            )
        place_success_event = (
            "place_success"
            if self.config.full_physics
            else (
                "manipulation_apply_smoke_place_apply_success"
                if self.config.manipulation_apply_smoke
                else "place_success"
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
        training_eligible = bool(self.config.full_physics and not self.config.dry_run)
        self.export_result = self.recorder.prepare_lerobot_export(
            training_eligible=training_eligible,
            training_eligibility_reason=(
                "episode_success_verified"
                if training_eligible
                else "execution_mode_is_not_full_physics"
            ),
        )
        if (
            self.config.full_physics
            and self.export_result.get("recording_enabled")
            and not self.export_result.get("lerobot_exported")
        ):
            return RobotAction.idle(source="export_lerobot"), self._fail(
                "lerobot_export_failed",
                observation,
                {"lerobot_export": dict(self.export_result)},
            )
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
        if self._physical_pick_enabled():
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
        if self._physical_pick_enabled():
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
            if self.state == PipelineState.EXEC_PICK:
                self.pick_executor_status = dict(executor_status)
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
            self.latest_executor_status = self.arm_executor.status()
            if self.state == PipelineState.EXEC_PICK:
                self.pick_executor_status = dict(self.latest_executor_status)
            if self.latest_executor_status.get("failed"):
                return RobotAction.idle(source=self.state.value), self._fail(
                    str(
                        self.latest_executor_status.get("failure_reason")
                        or "arm_execution_failed"
                    ),
                    observation,
                    self.latest_executor_status,
                )
            return RobotAction.idle(source=self.state.value), self._transition(
                next_state,
                observation.step_index,
            )
        action = self.arm_executor.compute_action(observation)
        self._remember_carry_gripper_target(action, observation)
        self.latest_executor_status = self.arm_executor.status()
        if self.state == PipelineState.EXEC_PICK:
            self.pick_executor_status = dict(self.latest_executor_status)
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
        # 最后一拍 action 必须先经过 apply + world.step，下一 tick 再用新 observation
        # 完成 close progress / strict wait 验证和状态迁移。
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
        if self.state == PipelineState.EXEC_PLACE:
            # place 只消费 pick 验证出的低内力 preload，不能把它重记成规划中的零开度。
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

    def _capture_verified_carry_gripper_preload(
        self,
        observation: SimulationState,
    ) -> None:
        """按验证后的真实接触开度生成低内力 carry 夹持目标。"""

        target = self._carry_gripper_target
        if target is None:
            return
        gripper_names = tuple(str(value) for value in target.get("gripper_joint_names") or ())
        close_positions = tuple(
            float(value) for value in target.get("commanded_close_positions") or ()
        )
        joint_names = tuple(
            str(value) for value in observation.metadata.get("joint_names") or ()
        )
        if (
            not gripper_names
            or len(close_positions) != len(gripper_names)
            or len(joint_names) != len(observation.joint_positions)
        ):
            return
        position_by_name = {
            name: float(position)
            for name, position in zip(joint_names, observation.joint_positions)
        }
        if any(name not in position_by_name for name in gripper_names):
            return

        contact_positions = tuple(position_by_name[name] for name in gripper_names)
        preload = float(self.config.manipulation.carry_gripper_preload_m)
        hold_positions = tuple(
            max(close_position, contact_position - preload)
            for close_position, contact_position in zip(
                close_positions,
                contact_positions,
            )
        )
        target.update(
            {
                "gripper_joint_positions": hold_positions,
                "hold_position_source": "verified_contact_preload",
                "verified_contact_positions": contact_positions,
                "carry_gripper_preload_m": preload,
            }
        )

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

        return_home_inserted = bool(return_home_report.get("inserted"))
        if (
            not self.config.manipulation.return_home_after_pick
            or not return_home_inserted
        ):
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
                "return_home_inserted": return_home_inserted,
                "return_home_reason": return_home_report.get("reason"),
                # main-pick parity 默认保持 planned lift/retreat 的安全末端目标，
                # 不追加未经 cuRobo 避障的 all-zero 直连轨迹。
                "carry_hold_policy": "hold_planned_pick_final_motion_target",
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
        # carry 阶段维持 pick 的安全末端目标；这里只发 position target，不直接写 joint state。
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

        action_metadata = dict(action.metadata)
        explicit_base_requested = bool(
            action_metadata.get("manipulation_base_lock") is True
            and self.state != PipelineState.FAILED
        )
        explicit_base_phase = action_metadata.get(
            "manipulation_base_lock_phase"
        )
        explicit_support_requested = bool(
            action_metadata.get("manipulation_support_joint_lock") is True
            and self.state != PipelineState.FAILED
        )
        explicit_support_phase = action_metadata.get(
            "manipulation_support_joint_lock_phase"
        )

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
            and self._physical_pick_enabled()
            and action.metadata.get("terminal_hold") is True
        ):
            lock_phase = PipelineState.CLEANUP_EPISODE
        requested = bool(
            explicit_base_requested
            or (
                self.config.manipulation.lock_base_during_manipulation
                and lock_phase is not None
                and self.state != PipelineState.FAILED
            )
        )
        support_requested = bool(
            explicit_support_requested
            or (
                self.config.manipulation.lock_support_joints_during_manipulation
                and lock_phase is not None
                and self.state != PipelineState.FAILED
            )
        )
        support_phase = (
            explicit_support_phase
            if explicit_support_requested
            else (lock_phase.value if support_requested and lock_phase is not None else None)
        )
        base_phase = (
            explicit_base_phase
            if explicit_base_requested
            else (lock_phase.value if requested and lock_phase is not None else None)
        )
        metadata = action_metadata
        metadata.update(
            {
                "manipulation_base_lock": requested,
                "manipulation_base_lock_phase": base_phase,
                "manipulation_support_joint_lock": support_requested,
                "manipulation_support_joint_lock_phase": support_phase,
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

        if phase == "place":
            configured_settle_steps = (
                self.config.manipulation.place_base_lock_settle_steps
            )
        else:
            configured_settle_steps = self.config.manipulation.base_lock_settle_steps
        settle_steps = (
            configured_settle_steps
            if self.config.manipulation.lock_base_during_manipulation
            else 0
        )
        if settle_steps <= 0 or self.state_ticks > settle_steps:
            return None

        events: list[PipelineEvent] = []
        root_linear_speed_mps = _vector_norm(observation.robot_root_velocity, limit=3)
        root_angular_speed_radps = _vector_norm(observation.robot_root_velocity[3:6])
        joint_velocity_max_abs = _max_abs(observation.joint_velocities)
        diagnostics = {
            "settle_steps": settle_steps,
            "root_linear_speed_mps": root_linear_speed_mps,
            "root_angular_speed_radps": root_angular_speed_radps,
            "joint_velocity_max_abs": joint_velocity_max_abs,
        }
        if self.state_ticks == 1:
            events.append(
                self._event(
                    f"{phase}_base_settle_start",
                    observation.step_index,
                    diagnostics,
                )
            )
        if self.state_ticks == settle_steps:
            events.append(
                self._event(
                    f"{phase}_base_settle_complete",
                    observation.step_index,
                    diagnostics,
                )
            )
        metadata = {
            "manipulation_base_settle": True,
            "manipulation_base_settle_phase": phase,
            "manipulation_base_settle_step": self.state_ticks,
            "manipulation_base_settle_steps": settle_steps,
            "manipulation_base_settle_root_linear_speed_mps": root_linear_speed_mps,
            "manipulation_base_settle_root_angular_speed_radps": root_angular_speed_radps,
            "manipulation_base_settle_joint_velocity_max_abs": joint_velocity_max_abs,
        }
        return (
            RobotAction(
                source=f"{phase}_base_settle",
                metadata=metadata,
            ),
            events,
        )

    def _settle_navigation_handoff(
        self,
        result: VerificationResult,
        observation: SimulationState,
        *,
        phase: str,
    ) -> tuple[RobotAction, list[PipelineEvent]] | None:
        """Wait briefly for bounded RL residual motion after nav success."""

        if not _navigation_handoff_requires_zero_command_settle(
            result,
            self.latest_executor_status,
        ):
            return None
        if self.state_ticks >= _NAV_HANDOFF_SETTLE_MAX_STEPS:
            return None

        metadata = {
            **result.metadata,
            "navigation_handoff_settle": True,
            "navigation_handoff_settle_phase": phase,
            "navigation_handoff_settle_step": self.state_ticks,
            "navigation_handoff_settle_max_steps": (
                _NAV_HANDOFF_SETTLE_MAX_STEPS
            ),
            "navigation_handoff_settle_linear_speed_limit_mps": float(
                result.metadata.get("linear_velocity_tolerance", 0.0)
            )
            + _NAV_HANDOFF_SETTLE_LINEAR_SPEED_MARGIN_MPS,
            "navigation_handoff_settle_angular_speed_limit_radps": float(
                result.metadata.get("angular_velocity_tolerance", 0.0)
            )
            + _NAV_HANDOFF_SETTLE_ANGULAR_SPEED_MARGIN_RADPS,
        }
        events: list[PipelineEvent] = []
        if self.state_ticks == 1:
            events.append(
                self._event(
                    f"nav_to_{phase}_handoff_settle_start",
                    observation.step_index,
                    metadata,
                )
            )
        return (
            RobotAction(
                base_velocity=(0.0, 0.0, 0.0),
                source=f"nav_to_{phase}_handoff_settle",
                metadata=metadata,
            ),
            events,
        )

    def _navigation_handoff_settle_complete_events(
        self,
        observation: SimulationState,
        *,
        phase: str,
        result: VerificationResult,
    ) -> list[PipelineEvent]:
        if self.state_ticks <= 1:
            return []
        return [
            self._event(
                f"nav_to_{phase}_handoff_settle_complete",
                observation.step_index,
                {
                    **result.metadata,
                    "navigation_handoff_settle_steps": self.state_ticks - 1,
                },
            )
        ]

    def _update_pick_peak_lift(self, observation: SimulationState) -> None:
        """记录 primary pick motion 的峰值抬升，回位后仍可验证真实 lift。"""

        if (
            observation.object_pose is None
            or self.episode_spec.object_initial_pose is None
        ):
            return
        lift_height = (
            float(observation.object_pose[2])
            - float(self.episode_spec.object_initial_pose[2])
        )
        if (
            self._pick_peak_object_lift_height_m is not None
            and lift_height <= self._pick_peak_object_lift_height_m
        ):
            return
        self._pick_peak_object_lift_height_m = lift_height
        self._pick_peak_object_pose = tuple(float(value) for value in observation.object_pose)
        self._pick_peak_step_index = int(observation.step_index)

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

    def _physical_pick_enabled(self) -> bool:
        """返回当前模式是否需要真实物理 pick 保护逻辑。"""

        return bool(self.config.full_physics or self.config.pick_smoke)

    def _object_settle_enabled(self) -> bool:
        """仅在真实 pick 模式按配置启用动态物体沉降。"""

        return bool(
            self._physical_pick_enabled()
            and self.config.manipulation.settle_object_before_navigation
            and self.episode_spec.object_initial_pose is not None
        )

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
            "pick_planner_result": dict(self.pick_planner_result),
            "pick_executor_status": dict(self.pick_executor_status),
            "pick_verification_result": dict(self.pick_verification_result),
            "place_verification_result": dict(self.place_verification_result),
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


def _finite_xyz(values: Any) -> tuple[float, float, float] | None:
    if not isinstance(values, (tuple, list)) or len(values) < 3:
        return None
    xyz = tuple(float(value) for value in values[:3])
    if not all(math.isfinite(value) for value in xyz):
        return None
    return (xyz[0], xyz[1], xyz[2])


def _object_tcp_offset(
    observation: SimulationState,
) -> tuple[float, float, float] | None:
    object_xyz = _finite_xyz(observation.object_pose)
    tcp_xyz = _finite_xyz(observation.tcp_pose)
    if object_xyz is None or tcp_xyz is None:
        return None
    return tuple(
        object_value - tcp_value
        for object_value, tcp_value in zip(object_xyz, tcp_xyz)
    )


def _gripper_motion_progress(
    observation: SimulationState,
    action: RobotAction,
) -> float | None:
    """根据真实关节开度计算夹爪从起点到打开目标的最小进度。"""

    metadata = action.metadata
    joint_names = tuple(str(value) for value in metadata.get("gripper_joint_names", ()))
    q_start = tuple(float(value) for value in metadata.get("gripper_start_positions", ()))
    q_target = tuple(
        float(value) for value in metadata.get("gripper_final_target_positions", ())
    )
    all_joint_names = tuple(str(value) for value in observation.metadata.get("joint_names", ()))
    if (
        not joint_names
        or len(q_start) != len(joint_names)
        or len(q_target) != len(joint_names)
        or not all_joint_names
        or not observation.joint_positions
    ):
        return None
    try:
        joint_indices = tuple(all_joint_names.index(name) for name in joint_names)
    except ValueError:
        return None
    if any(index >= len(observation.joint_positions) for index in joint_indices):
        return None

    progress_values: list[float] = []
    for index, start, target in zip(joint_indices, q_start, q_target):
        span = target - start
        if start >= target - 0.002:
            progress_values.append(1.0)
            continue
        if abs(span) <= 1.0e-9:
            continue
        actual = float(observation.joint_positions[index])
        if not math.isfinite(actual):
            return None
        progress_values.append(max(0.0, min(1.0, (actual - start) / span)))
    if not progress_values:
        return None
    return min(progress_values)
