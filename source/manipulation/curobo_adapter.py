"""cuRobo 分段计划到新 pipeline ArmPlan 的适配。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from source.interfaces import ArmPlan, EpisodeSpec, SimulationState


DEFAULT_ARM_JOINT_NAMES = (
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
)


class CuroboPlanFormatError(ValueError):
    """旧 cuRobo JSON 缺少执行所需字段时抛出。"""


def load_curobo_plan_json(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise CuroboPlanFormatError(f"cuRobo plan must be a JSON object: {plan_path}")
    return payload


def arm_plan_from_curobo_payload(
    payload: dict[str, Any],
    *,
    operation: str | None = None,
    source_path: str | Path | None = None,
) -> ArmPlan:
    """保留旧 plan 的 segment 语义，同时给状态机提供统一 ArmPlan。"""

    joint_names = _read_joint_names(payload)
    segments = _read_segments(payload, arm_dof=len(joint_names))
    joint_trajectory = tuple(
        point
        for segment in segments
        if segment["type"] == "motion"
        for point in segment["trajectory"]["q"]
    )
    if not joint_trajectory:
        raise CuroboPlanFormatError("cuRobo plan does not contain any motion trajectory")

    return ArmPlan(
        operation=operation or _infer_operation(payload),
        joint_trajectory=joint_trajectory,
        metadata={
            "plan_format": "curobo_segments_v1",
            "schema_version": payload.get("schema_version"),
            "planner": payload.get("planner"),
            "joint_names": joint_names,
            "tool_frame": payload.get("tool_frame"),
            "object_prim_path": payload.get("object_prim_path"),
            "segments": segments,
            "summary": dict(payload.get("summary") or {}),
            "source_plan_json": str(source_path) if source_path is not None else None,
        },
    )


class CuroboJsonManipulationPlanner:
    """读取预先生成的 cuRobo JSON；真实在线 cuRobo 后续替换这里。"""

    def __init__(
        self,
        *,
        pick_plan_json: str | Path | None = None,
        place_plan_json: str | Path | None = None,
    ):
        self.pick_plan_json = Path(pick_plan_json) if pick_plan_json is not None else None
        self.place_plan_json = Path(place_plan_json) if place_plan_json is not None else None

    def plan_pick(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state, episode_spec
        return self._load(self.pick_plan_json, operation="pick")

    def plan_place(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state, episode_spec
        return self._load(self.place_plan_json, operation="place")

    @staticmethod
    def _load(path: Path | None, *, operation: str) -> ArmPlan:
        if path is None:
            raise RuntimeError(f"{operation} cuRobo plan json is not configured")
        return arm_plan_from_curobo_payload(
            load_curobo_plan_json(path),
            operation=operation,
            source_path=path,
        )


def _read_joint_names(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("joint_names", DEFAULT_ARM_JOINT_NAMES)
    if not isinstance(raw, list | tuple) or not raw:
        raise CuroboPlanFormatError("cuRobo plan joint_names must be a non-empty list")
    joint_names = tuple(str(name) for name in raw)
    if len(set(joint_names)) != len(joint_names):
        raise CuroboPlanFormatError(f"cuRobo plan joint_names contains duplicates: {joint_names}")
    return joint_names


def _read_segments(payload: dict[str, Any], *, arm_dof: int) -> tuple[dict[str, Any], ...]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise CuroboPlanFormatError("cuRobo plan segments must be a non-empty list")
    return tuple(_normalize_segment(segment, arm_dof=arm_dof) for segment in raw_segments)


def _normalize_segment(segment: Any, *, arm_dof: int) -> dict[str, Any]:
    if not isinstance(segment, dict):
        raise CuroboPlanFormatError(f"cuRobo segment must be an object: {segment!r}")
    segment_type = str(segment.get("type") or "")
    if segment_type == "motion":
        return _normalize_motion_segment(segment, arm_dof=arm_dof)
    if segment_type == "gripper":
        return _normalize_gripper_segment(segment)
    raise CuroboPlanFormatError(f"unsupported cuRobo segment type: {segment_type!r}")


def _normalize_motion_segment(segment: dict[str, Any], *, arm_dof: int) -> dict[str, Any]:
    trajectory = segment.get("trajectory")
    if not isinstance(trajectory, dict):
        raise CuroboPlanFormatError(f"{segment.get('name')}: motion segment missing trajectory")
    q_rows = _read_matrix(trajectory.get("q"), width=arm_dof, label=f"{segment.get('name')}:trajectory.q")
    time_from_start = _read_vector(
        trajectory.get("time_from_start"),
        label=f"{segment.get('name')}:trajectory.time_from_start",
    )
    if len(q_rows) != len(time_from_start):
        raise CuroboPlanFormatError(
            f"{segment.get('name')}: trajectory.q and time_from_start length mismatch"
        )

    normalized_trajectory: dict[str, Any] = {
        "time_from_start": time_from_start,
        "q": q_rows,
    }
    if "qd" in trajectory:
        qd_rows = _read_matrix(
            trajectory.get("qd"),
            width=arm_dof,
            label=f"{segment.get('name')}:trajectory.qd",
        )
        _ensure_same_length(qd_rows, q_rows, label=f"{segment.get('name')}:trajectory.qd")
        normalized_trajectory["qd"] = qd_rows
    if "qdd" in trajectory:
        qdd_rows = _read_matrix(
            trajectory.get("qdd"),
            width=arm_dof,
            label=f"{segment.get('name')}:trajectory.qdd",
        )
        _ensure_same_length(qdd_rows, q_rows, label=f"{segment.get('name')}:trajectory.qdd")
        normalized_trajectory["qdd"] = qdd_rows
    for key in ("tcp_position_world", "tcp_quaternion_world", "tcp_position_base", "tcp_quaternion_base"):
        if key in trajectory:
            normalized_trajectory[key] = trajectory[key]

    return {
        "name": str(segment.get("name") or "motion"),
        "type": "motion",
        "target_name": segment.get("target_name"),
        "target_pose_base": segment.get("target_pose_base"),
        "timing": dict(segment.get("timing") or {}),
        "final_error": dict(segment.get("final_error") or {}),
        "plan_info": dict(segment.get("plan_info") or {}),
        "trajectory": normalized_trajectory,
    }


def _normalize_gripper_segment(segment: dict[str, Any]) -> dict[str, Any]:
    joint_names_raw = segment.get("joint_names")
    if not isinstance(joint_names_raw, list | tuple) or not joint_names_raw:
        raise CuroboPlanFormatError(f"{segment.get('name')}: gripper segment missing joint_names")
    target_position = _read_vector(
        segment.get("target_position"),
        label=f"{segment.get('name')}:target_position",
    )
    if len(target_position) != len(joint_names_raw):
        raise CuroboPlanFormatError(
            f"{segment.get('name')}: gripper target length does not match joint_names"
        )
    return {
        "name": str(segment.get("name") or "gripper"),
        "type": "gripper",
        "joint_names": tuple(str(name) for name in joint_names_raw),
        "target_position": target_position,
    }


def _read_matrix(raw: Any, *, width: int, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise CuroboPlanFormatError(f"{label} must be a non-empty matrix")
    rows = tuple(_read_vector(row, label=label) for row in raw)
    bad = [row for row in rows if len(row) != width]
    if bad:
        raise CuroboPlanFormatError(f"{label} row width must be {width}")
    return rows


def _read_vector(raw: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise CuroboPlanFormatError(f"{label} must be a non-empty vector")
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise CuroboPlanFormatError(f"{label} must contain only numbers") from exc


def _ensure_same_length(
    rows: tuple[tuple[float, ...], ...],
    reference_rows: tuple[tuple[float, ...], ...],
    *,
    label: str,
) -> None:
    if len(rows) != len(reference_rows):
        raise CuroboPlanFormatError(f"{label} length must match trajectory.q")


def _infer_operation(payload: dict[str, Any]) -> str:
    planner = str(payload.get("planner") or "").lower()
    if payload.get("place_mode") or "place" in planner:
        return "place"
    if payload.get("grasp_mode") or "grasp" in planner:
        return "pick"
    return "manipulation"
