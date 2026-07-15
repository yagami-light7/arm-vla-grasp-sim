"""统一生成移动操作 episode 的 task_stage 与连续 subtask 片段。"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .training_action import task_requests_vla_training_action


SUBTASK_SCHEMA_VERSION = (
    "nav_straight_turn_stop__arm_approach_contact_retreat_v1"
)
SUBTASK_DIRECTORY_LAYOUT = "episodes_task_episode_subtask_front_wrist_v3"
_LEGACY_SUBTASK_DIRECTORY_LAYOUTS = {
    "episodes_task_episode_segment_v1",
    "episodes_task_episode_segment_front_wrist_v2",
}
TASK_STAGES = ("nav_to_pick", "pick", "nav_to_place", "place")
NAV_SUBTASKS = ("nav_straight", "nav_turn", "nav_stop")
ARM_SUBTASKS = ("arm_approach", "arm_contact", "arm_retreat")
SUBTASK_LABELS = (*NAV_SUBTASKS, *ARM_SUBTASKS)


@dataclass(frozen=True)
class SubtaskSegmentationConfig:
    """保存切分阈值、防抖长度和接触标签来源。"""

    enabled: bool = True
    schema: str = SUBTASK_SCHEMA_VERSION
    directory_export: bool = True
    output_layout: str = SUBTASK_DIRECTORY_LAYOUT
    min_segment_frames: int = 3
    hysteresis_frames: int = 2
    stop_command_linear_max_mps: float = 0.03
    stop_command_angular_max_rps: float = 0.08
    stop_measured_linear_max_mps: float = 0.08
    stop_measured_angular_max_rps: float = 0.20
    turn_command_angular_min_rps: float = 0.12
    turn_measured_angular_min_rps: float = 0.25
    turn_yaw_delta_min_rad: float = 0.03
    arm_contact_distance_m: float = 0.20
    pick_lift_delta_m: float = 0.02
    arm_retreat_distance_delta_m: float = 0.02
    contact_label_source: str = "heuristic_action_and_kinematics"
    config_source: str = "task"

    def metadata(self) -> dict[str, Any]:
        """返回可写入数据版本 metadata 的完整配置。"""

        return asdict(self)


def task_requests_subtask_segmentation(task: dict[str, Any]) -> bool:
    """判断任务是否要求正式子任务切分。"""

    raw = task.get("subtask_segmentation")
    if isinstance(raw, dict):
        return bool(raw.get("enabled", True))
    return bool(task_requests_vla_training_action(task))


def validate_subtask_segmentation_config(
    task: dict[str, Any],
) -> SubtaskSegmentationConfig | None:
    """读取并校验 task 中的子任务配置；旧 10 维任务使用可追溯默认值。"""

    if not task_requests_subtask_segmentation(task):
        return None
    raw = task.get("subtask_segmentation")
    payload = dict(raw) if isinstance(raw, dict) else {}
    navigation = payload.get("navigation")
    navigation = navigation if isinstance(navigation, dict) else {}
    requested_output_layout = str(
        payload.get("output_layout") or SUBTASK_DIRECTORY_LAYOUT
    )
    legacy_layout_migrated = (
        requested_output_layout in _LEGACY_SUBTASK_DIRECTORY_LAYOUTS
    )
    output_layout = (
        SUBTASK_DIRECTORY_LAYOUT
        if legacy_layout_migrated
        else requested_output_layout
    )
    config = SubtaskSegmentationConfig(
        enabled=bool(payload.get("enabled", True)),
        schema=str(payload.get("schema") or SUBTASK_SCHEMA_VERSION),
        directory_export=bool(payload.get("directory_export", True)),
        output_layout=output_layout,
        min_segment_frames=int(payload.get("min_segment_frames", 3)),
        hysteresis_frames=int(payload.get("hysteresis_frames", 2)),
        stop_command_linear_max_mps=float(
            navigation.get("stop_command_linear_max_mps", 0.03)
        ),
        stop_command_angular_max_rps=float(
            navigation.get("stop_command_angular_max_rps", 0.08)
        ),
        stop_measured_linear_max_mps=float(
            navigation.get("stop_measured_linear_max_mps", 0.08)
        ),
        stop_measured_angular_max_rps=float(
            navigation.get("stop_measured_angular_max_rps", 0.20)
        ),
        turn_command_angular_min_rps=float(
            navigation.get("turn_command_angular_min_rps", 0.12)
        ),
        turn_measured_angular_min_rps=float(
            navigation.get("turn_measured_angular_min_rps", 0.25)
        ),
        turn_yaw_delta_min_rad=float(
            navigation.get("turn_yaw_delta_min_rad", 0.03)
        ),
        arm_contact_distance_m=float(
            payload.get("arm_contact_distance_m", 0.20)
        ),
        pick_lift_delta_m=float(payload.get("pick_lift_delta_m", 0.02)),
        arm_retreat_distance_delta_m=float(
            payload.get("arm_retreat_distance_delta_m", 0.02)
        ),
        contact_label_source=str(
            payload.get("contact_label_source")
            or "heuristic_action_and_kinematics"
        ),
        config_source=(
            "task_legacy_layout_migrated"
            if legacy_layout_migrated
            else ("task" if isinstance(raw, dict) else "default_v2")
        ),
    )
    if config.schema != SUBTASK_SCHEMA_VERSION:
        raise ValueError(
            "subtask_segmentation.schema 必须是 "
            f"{SUBTASK_SCHEMA_VERSION!r}"
        )
    if config.output_layout != SUBTASK_DIRECTORY_LAYOUT:
        raise ValueError(
            "subtask_segmentation.output_layout 必须是 "
            f"{SUBTASK_DIRECTORY_LAYOUT!r}"
        )
    if not config.directory_export:
        raise ValueError(
            "subtask_segmentation.directory_export 必须为 true；"
            "当前 schema 要求 parquet 与逐图片目录同步导出"
        )
    if config.min_segment_frames < 1:
        raise ValueError("subtask_segmentation.min_segment_frames 必须至少为 1")
    if config.hysteresis_frames < 1:
        raise ValueError("subtask_segmentation.hysteresis_frames 必须至少为 1")
    positive_values = {
        "stop_command_linear_max_mps": config.stop_command_linear_max_mps,
        "stop_command_angular_max_rps": config.stop_command_angular_max_rps,
        "stop_measured_linear_max_mps": config.stop_measured_linear_max_mps,
        "stop_measured_angular_max_rps": config.stop_measured_angular_max_rps,
        "turn_command_angular_min_rps": config.turn_command_angular_min_rps,
        "turn_measured_angular_min_rps": config.turn_measured_angular_min_rps,
        "turn_yaw_delta_min_rad": config.turn_yaw_delta_min_rad,
        "arm_contact_distance_m": config.arm_contact_distance_m,
        "pick_lift_delta_m": config.pick_lift_delta_m,
        "arm_retreat_distance_delta_m": config.arm_retreat_distance_delta_m,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"subtask_segmentation.{name} 必须是有限正数")
    if not (
        config.contact_label_source.startswith("heuristic")
        or config.contact_label_source == "contact_sensor"
    ):
        raise ValueError(
            "subtask_segmentation.contact_label_source 必须明确标记 heuristic 或 contact_sensor"
        )
    return config


def extract_action_semantics(action: dict[str, Any]) -> dict[str, Any]:
    """从控制 action 中提取切分所需的小型语义信号。"""

    metadata = action.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    signals = {
        "action_source": str(action.get("source") or ""),
        "gripper_command": str(action.get("gripper_command") or ""),
        "operation": str(metadata.get("operation") or ""),
        "segment_name": str(metadata.get("segment_name") or ""),
        "parent_segment_name": str(metadata.get("parent_segment_name") or ""),
        "segment_type": str(metadata.get("segment_type") or ""),
        "event_marker": str(metadata.get("event_marker") or ""),
        "navigation_phase": str(metadata.get("phase") or ""),
        "terminal_control_mode": str(
            metadata.get("terminal_control_mode") or ""
        ),
        "distance_to_goal": _finite_or_none(metadata.get("distance_to_goal")),
        "yaw_error": _finite_or_none(metadata.get("yaw_error")),
        "explicit_task_contact": any(
            metadata.get(key) is True
            for key in (
                "target_contact",
                "object_gripper_contact",
                "object_table_contact",
                "task_contact_verified",
            )
        ),
    }
    return {
        "action_source": signals["action_source"],
        "gripper_command": signals["gripper_command"],
        "action_segment_name": signals["segment_name"],
        "action_segment_type": signals["segment_type"],
        "subtask_signals": signals,
    }


def hydrate_sample_action_semantics(
    episode_dir: str | Path,
    samples: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从 frames.jsonl 为旧 episode 恢复采样帧缺失的动作语义。"""

    source_dir = Path(episode_dir).expanduser().resolve()
    by_step: dict[int, dict[str, Any]] = {}
    frames_path = source_dir / "frames.jsonl"
    if frames_path.is_file():
        with frames_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                step = payload.get("step_index")
                action = payload.get("action")
                if (
                    isinstance(step, int)
                    and not isinstance(step, bool)
                    and isinstance(action, dict)
                ):
                    semantics = extract_action_semantics(action)
                    post = payload.get("post_step_observation")
                    post = post if isinstance(post, dict) else {}
                    observation_metadata = post.get("metadata")
                    if isinstance(observation_metadata, dict):
                        signals = dict(semantics["subtask_signals"])
                        signals["explicit_task_contact"] = bool(
                            signals["explicit_task_contact"]
                            or any(
                                observation_metadata.get(key) is True
                                for key in (
                                    "target_contact",
                                    "object_gripper_contact",
                                    "object_table_contact",
                                    "task_contact_verified",
                                )
                            )
                        )
                        signals["contact_force_max"] = _finite_or_none(
                            observation_metadata.get("contact_force_max")
                        )
                        semantics["subtask_signals"] = signals
                    by_step[int(step)] = semantics

    hydrated: list[dict[str, Any]] = []
    for sample in samples:
        updated = dict(sample)
        step = sample.get("simulation_step")
        recovered = (
            by_step.get(int(step), {})
            if isinstance(step, int) and not isinstance(step, bool)
            else {}
        )
        recovered_signals = recovered.get("subtask_signals")
        recovered_signals = (
            recovered_signals if isinstance(recovered_signals, dict) else {}
        )
        existing_signals = updated.get("subtask_signals")
        existing_signals = (
            existing_signals if isinstance(existing_signals, dict) else {}
        )
        merged_signals = dict(recovered_signals)
        for key, value in existing_signals.items():
            if key == "explicit_task_contact":
                merged_signals[key] = bool(
                    merged_signals.get(key) is True or value is True
                )
            elif value not in (None, ""):
                merged_signals[key] = value
        updated["subtask_signals"] = merged_signals
        for key in (
            "action_source",
            "gripper_command",
            "action_segment_name",
            "action_segment_type",
        ):
            if not updated.get(key) and recovered.get(key):
                updated[key] = recovered[key]
            else:
                updated.setdefault(key, "")
        hydrated.append(updated)
    return hydrated


def segment_episode_samples(
    samples: Sequence[dict[str, Any]],
    task: dict[str, Any] | None = None,
    *,
    config: SubtaskSegmentationConfig,
    fps: float,
) -> dict[str, Any]:
    """生成逐帧标签和连续片段，保证每帧恰好属于一个目录。"""

    if not samples:
        return {
            "frames": [],
            "segments": [],
            "segment_count": 0,
            "frame_counts": {},
            "task_stage_frame_counts": {},
            "contact_label_source": config.contact_label_source,
            "config": config.metadata(),
        }
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("subtask segmentation fps 必须是有限正数")
    raw_task = task if isinstance(task, dict) else {}
    stages = _task_stages(samples, raw_task)
    labels: list[str] = [""] * len(samples)
    sources: list[str] = [""] * len(samples)
    contact_sources: list[str] = ["not_applicable"] * len(samples)

    previous_nav_label: dict[str, str] = {}
    manipulation_phase = {"pick": 0, "place": 0}
    initial_object_z: dict[str, float | None] = {"pick": None, "place": None}
    contact_distance: dict[str, float | None] = {"pick": None, "place": None}
    for frame_index, (sample, stage) in enumerate(zip(samples, stages)):
        if stage in {"nav_to_pick", "nav_to_place"}:
            labels[frame_index] = _navigation_label(
                samples,
                stages,
                frame_index,
                previous_label=previous_nav_label.get(stage),
                config=config,
                fps=float(fps),
            )
            sources[frame_index] = "kinematic_command_hysteresis"
            previous_nav_label[stage] = labels[frame_index]
            continue

        object_xyz = _sample_object_xyz(sample)
        if initial_object_z[stage] is None and object_xyz is not None:
            initial_object_z[stage] = object_xyz[2]
        (
            labels[frame_index],
            manipulation_phase[stage],
            sources[frame_index],
            contact_sources[frame_index],
            current_contact_distance,
        ) = _manipulation_label(
            sample,
            raw_task,
            stage=stage,
            phase=manipulation_phase[stage],
            initial_object_z=initial_object_z[stage],
            contact_distance=contact_distance[stage],
            config=config,
        )
        if current_contact_distance is not None:
            contact_distance[stage] = current_contact_distance

    labels = _apply_hysteresis(
        stages,
        labels,
        frames=config.hysteresis_frames,
    )
    labels, short_reasons = _merge_short_segments(
        stages,
        labels,
        samples,
        min_segment_frames=config.min_segment_frames,
    )
    segments = _segments_from_labels(
        stages,
        labels,
        sources,
        contact_sources,
        short_reasons,
    )
    annotations: list[dict[str, Any]] = [dict() for _ in samples]
    for segment in segments:
        start = int(segment["global_start_frame"])
        end = int(segment["global_end_frame"])
        for local_index, frame_index in enumerate(range(start, end + 1)):
            annotations[frame_index] = {
                "frame_index": frame_index,
                "task_stage": segment["task_stage"],
                "subtask": segment["subtask"],
                "segment_index": segment["segment_index"],
                "subtask_frame_index": local_index,
                "label_source": segment["label_source"],
                "contact_label_source": segment["contact_label_source"],
            }
    return {
        "frames": annotations,
        "segments": segments,
        "segment_count": len(segments),
        "frame_counts": dict(Counter(labels)),
        "task_stage_frame_counts": dict(Counter(stages)),
        "contact_label_source": config.contact_label_source,
        "config": config.metadata(),
    }


def _task_stages(
    samples: Sequence[dict[str, Any]],
    task: dict[str, Any],
) -> list[str]:
    place = task.get("place")
    place_enabled = not isinstance(place, dict) or bool(place.get("enabled", True))
    result: list[str] = []
    previous: str | None = None
    for sample in samples:
        state = str(sample.get("pipeline_state") or "").strip().lower()
        exact = {
            "build_stage": "nav_to_pick",
            "reset_episode": "nav_to_pick",
            "plan_nav_to_pick": "nav_to_pick",
            "exec_nav_to_pick": "nav_to_pick",
            "verify_pick_reachable": "nav_to_pick",
            "plan_pick": "pick",
            "exec_pick": "pick",
            "verify_pick_success": "pick",
            "plan_nav_to_place": "nav_to_place",
            "exec_nav_to_place": "nav_to_place",
            "verify_place_reachable": "nav_to_place",
            "plan_place": "place",
            "exec_place": "place",
            "verify_place_success": "place",
            "export_lerobot": "place",
            "cleanup_episode": "place",
            "done": "place",
        }
        stage = exact.get(state)
        if stage is None:
            if "nav" in state and "place" in state:
                stage = "nav_to_place"
            elif "nav" in state:
                stage = "nav_to_pick"
            elif "place" in state:
                stage = "place"
            elif "pick" in state or "grasp" in state:
                stage = "pick"
            else:
                stage = previous or "nav_to_pick"
        if not place_enabled and stage in {"nav_to_place", "place"}:
            stage = "pick"
        if previous is not None and TASK_STAGES.index(stage) < TASK_STAGES.index(previous):
            stage = previous
        result.append(stage)
        previous = stage
    return result


def _navigation_label(
    samples: Sequence[dict[str, Any]],
    stages: Sequence[str],
    frame_index: int,
    *,
    previous_label: str | None,
    config: SubtaskSegmentationConfig,
    fps: float,
) -> str:
    sample = samples[frame_index]
    measured = _vector(sample.get("base_velocity"), 3)
    control = _vector(sample.get("action"), 11)
    command = control[:3]
    measured_linear = math.hypot(measured[0], measured[1])
    measured_angular = abs(measured[2])
    command_linear = math.hypot(command[0], command[1])
    command_angular = abs(command[2])
    _delta_xy, delta_yaw, _delta_time = _neighbor_pose_delta(
        samples,
        stages,
        frame_index,
        fps=fps,
    )
    stop_scale = 1.20 if previous_label == "nav_stop" else 1.0
    if (
        command_linear <= config.stop_command_linear_max_mps * stop_scale
        and command_angular <= config.stop_command_angular_max_rps * stop_scale
        and measured_linear <= config.stop_measured_linear_max_mps * stop_scale
        and measured_angular <= config.stop_measured_angular_max_rps * stop_scale
    ):
        return "nav_stop"
    turn_scale = 0.80 if previous_label == "nav_turn" else 1.0
    if (
        command_angular >= config.turn_command_angular_min_rps * turn_scale
        or measured_angular >= config.turn_measured_angular_min_rps * turn_scale
        or delta_yaw >= config.turn_yaw_delta_min_rad * turn_scale
    ):
        return "nav_turn"
    return "nav_straight"


def _manipulation_label(
    sample: dict[str, Any],
    task: dict[str, Any],
    *,
    stage: str,
    phase: int,
    initial_object_z: float | None,
    contact_distance: float | None,
    config: SubtaskSegmentationConfig,
) -> tuple[str, int, str, str, float | None]:
    signals = sample.get("subtask_signals")
    signals = signals if isinstance(signals, dict) else {}
    segment_name = " ".join(
        str(signals.get(key) or "").lower()
        for key in ("segment_name", "parent_segment_name", "event_marker")
    )
    gripper_command = str(
        signals.get("gripper_command") or sample.get("gripper_command") or ""
    ).lower()
    explicit_contact = signals.get("explicit_task_contact") is True
    object_xyz = _sample_object_xyz(sample)
    tcp_xyz = _sample_tcp_xyz(sample)
    current_distance = (
        math.dist(object_xyz, tcp_xyz)
        if object_xyz is not None and tcp_xyz is not None
        else None
    )
    gripper = _finite_or_none(sample.get("gripper_position"))

    if stage == "pick":
        close_signal = bool(
            gripper_command == "close"
            or "close_gripper" in segment_name
            or "gripper_close" in segment_name
        )
        near_object = bool(
            current_distance is not None
            and current_distance <= config.arm_contact_distance_m
        )
        contact_trigger = bool(
            explicit_contact
            or (close_signal and near_object)
        )
        lifted = bool(
            object_xyz is not None
            and initial_object_z is not None
            and object_xyz[2] - initial_object_z >= config.pick_lift_delta_m
        )
        retreat_trigger = bool(
            phase >= 1
            and (
                any(
                    token in segment_name
                    for token in ("lift", "retreat", "return_home")
                )
                or (lifted and (gripper is None or gripper < 0.02))
            )
        )
    else:
        open_signal = bool(
            gripper_command == "open"
            or "open_gripper" in segment_name
            or "gripper_open" in segment_name
        )
        place_target = _place_target_xyz(task)
        near_target = bool(
            object_xyz is not None
            and place_target is not None
            and math.hypot(
                object_xyz[0] - place_target[0],
                object_xyz[1] - place_target[1],
            )
            <= config.arm_contact_distance_m
        )
        contact_trigger = bool(
            explicit_contact
            or (open_signal and near_target)
        )
        moved_away = bool(
            current_distance is not None
            and contact_distance is not None
            and current_distance - contact_distance
            >= config.arm_retreat_distance_delta_m
        )
        retreat_trigger = bool(
            phase >= 1
            and (
                any(token in segment_name for token in ("retreat", "return_home"))
                or (moved_away and (gripper is None or gripper > 0.03))
            )
        )

    if phase == 0 and contact_trigger:
        phase = 1
        contact_distance = current_distance
    if phase == 1 and retreat_trigger:
        phase = 2
    label = ("arm_approach", "arm_contact", "arm_retreat")[phase]
    if label == "arm_contact":
        contact_sensor_verified = bool(
            explicit_contact and config.contact_label_source == "contact_sensor"
        )
        contact_source = (
            "contact_sensor"
            if contact_sensor_verified
            else config.contact_label_source
        )
        if contact_sensor_verified:
            label_source = "contact_sensor"
        elif explicit_contact:
            label_source = "heuristic_explicit_task_contact"
        else:
            label_source = "heuristic_gripper_object_pose_executor"
    else:
        contact_source = "not_applicable"
        label_source = "executor_semantics_and_kinematics"
    return label, phase, label_source, contact_source, contact_distance


def _apply_hysteresis(
    stages: Sequence[str],
    labels: Sequence[str],
    *,
    frames: int,
) -> list[str]:
    if frames <= 1 or not labels:
        return list(labels)
    result = list(labels)
    stage_start = 0
    while stage_start < len(labels):
        stage_end = stage_start
        while stage_end + 1 < len(labels) and stages[stage_end + 1] == stages[stage_start]:
            stage_end += 1
        stable = labels[stage_start]
        candidate: str | None = None
        candidate_start = stage_start
        candidate_count = 0
        for index in range(stage_start + 1, stage_end + 1):
            raw = labels[index]
            if raw == stable:
                candidate = None
                candidate_count = 0
                result[index] = stable
                continue
            if raw != candidate:
                candidate = raw
                candidate_start = index
                candidate_count = 1
            else:
                candidate_count += 1
            result[index] = stable
            if candidate_count >= frames:
                for confirmed_index in range(candidate_start, index + 1):
                    result[confirmed_index] = raw
                stable = raw
                candidate = None
                candidate_count = 0
        stage_start = stage_end + 1
    return result


def _merge_short_segments(
    stages: Sequence[str],
    labels: list[str],
    samples: Sequence[dict[str, Any]],
    *,
    min_segment_frames: int,
) -> tuple[list[str], dict[tuple[int, int], str]]:
    labels = list(labels)
    while True:
        runs = _label_runs(stages, labels)
        changed = False
        for run_index, (start, end, stage, label) in enumerate(runs):
            if end - start + 1 >= min_segment_frames:
                continue
            if _protected_short_reason(
                samples,
                start=start,
                end=end,
                label=label,
            ):
                continue
            neighbors = [
                run
                for run in (
                    runs[run_index - 1] if run_index > 0 else None,
                    runs[run_index + 1] if run_index + 1 < len(runs) else None,
                )
                if run is not None and run[2] == stage
            ]
            if not neighbors:
                continue
            if len(neighbors) == 2 and neighbors[0][3] == neighbors[1][3]:
                target_label = neighbors[0][3]
            else:
                target_label = max(
                    neighbors,
                    key=lambda run: run[1] - run[0] + 1,
                )[3]
            for index in range(start, end + 1):
                labels[index] = target_label
            changed = True
            break
        if not changed:
            break
    reasons: dict[tuple[int, int], str] = {}
    for start, end, _stage, label in _label_runs(stages, labels):
        if end - start + 1 >= min_segment_frames:
            continue
        reasons[(start, end)] = _protected_short_reason(
            samples,
            start=start,
            end=end,
            label=label,
        ) or "no_same_stage_neighbor_for_short_segment"
    return labels, reasons


def _protected_short_reason(
    samples: Sequence[dict[str, Any]],
    *,
    start: int,
    end: int,
    label: str,
) -> str | None:
    if label == "arm_contact":
        return "task_contact_event_shorter_than_min_segment_frames"
    if label in {"nav_turn", "nav_stop"}:
        for sample in samples[start : end + 1]:
            signals = sample.get("subtask_signals")
            if not isinstance(signals, dict):
                continue
            if (
                str(signals.get("action_source") or "").startswith(
                    "navigation_terminal"
                )
                or signals.get("terminal_control_mode")
            ):
                return "terminal_alignment_event_shorter_than_min_segment_frames"
    if start == 0 and end == len(samples) - 1:
        return "episode_shorter_than_min_segment_frames"
    return None


def _segments_from_labels(
    stages: Sequence[str],
    labels: Sequence[str],
    sources: Sequence[str],
    contact_sources: Sequence[str],
    short_reasons: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    runs = _label_runs(stages, labels)
    segments: list[dict[str, Any]] = []
    for order, (start, end, stage, label) in enumerate(runs, start=1):
        label_sources = sorted({value for value in sources[start : end + 1] if value})
        contact_values = sorted(
            {
                value
                for value in contact_sources[start : end + 1]
                if value and value != "not_applicable"
            }
        )
        segments.append(
            {
                "segment_index": order,
                "task_stage": stage,
                "subtask": label,
                "global_start_frame": start,
                "global_end_frame": end,
                "frame_count": end - start + 1,
                "previous_segment_index": order - 1 if order > 1 else None,
                "next_segment_index": order + 1 if order < len(runs) else None,
                "label_source": "+".join(label_sources) or "unknown",
                "contact_label_source": (
                    "+".join(contact_values)
                    if label == "arm_contact" and contact_values
                    else "not_applicable"
                ),
                "retained_short_reason": short_reasons.get((start, end)),
            }
        )
    return segments


def _label_runs(
    stages: Sequence[str],
    labels: Sequence[str],
) -> list[tuple[int, int, str, str]]:
    if not labels:
        return []
    runs: list[tuple[int, int, str, str]] = []
    start = 0
    for index in range(1, len(labels)):
        if stages[index] != stages[start] or labels[index] != labels[start]:
            runs.append((start, index - 1, stages[start], labels[start]))
            start = index
    runs.append((start, len(labels) - 1, stages[start], labels[start]))
    return runs


def _neighbor_pose_delta(
    samples: Sequence[dict[str, Any]],
    stages: Sequence[str],
    frame_index: int,
    *,
    fps: float,
) -> tuple[float, float, float]:
    neighbor = frame_index - 1
    if neighbor < 0 or stages[neighbor] != stages[frame_index]:
        neighbor = frame_index + 1
    if neighbor >= len(samples) or stages[neighbor] != stages[frame_index]:
        return 0.0, 0.0, 1.0 / fps
    current = _vector(samples[frame_index].get("base_pose"), 7)
    other = _vector(samples[neighbor].get("base_pose"), 7)
    delta_xy = math.hypot(current[0] - other[0], current[1] - other[1])
    delta_yaw = abs(
        _wrap_angle(_quat_yaw(current[3:7]) - _quat_yaw(other[3:7]))
    )
    current_time = _finite_or_none(samples[frame_index].get("timestamp"))
    other_time = _finite_or_none(samples[neighbor].get("timestamp"))
    delta_time = (
        abs(current_time - other_time)
        if current_time is not None and other_time is not None
        else 1.0 / fps
    )
    if delta_time <= 1.0e-9:
        delta_time = 1.0 / fps
    return delta_xy, delta_yaw, delta_time


def _sample_object_xyz(sample: dict[str, Any]) -> tuple[float, float, float] | None:
    raw = sample.get("object_state")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    values = tuple(float(value) for value in raw[:3])
    return values if all(math.isfinite(value) for value in values) else None


def _sample_tcp_xyz(sample: dict[str, Any]) -> tuple[float, float, float] | None:
    raw = sample.get("tcp_pose")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    values = tuple(float(value) for value in raw[:3])
    return values if all(math.isfinite(value) for value in values) else None


def _place_target_xyz(task: dict[str, Any]) -> tuple[float, float, float] | None:
    place = task.get("place")
    if not isinstance(place, dict):
        return None
    pose = place.get("place_pose_world")
    if isinstance(pose, dict) and all(axis in pose for axis in ("x", "y", "z")):
        values = tuple(float(pose[axis]) for axis in ("x", "y", "z"))
        if all(math.isfinite(value) for value in values):
            return values
    region = place.get("placement_region")
    if isinstance(region, dict):
        center = region.get("center_xyz")
        if isinstance(center, (list, tuple)) and len(center) >= 3:
            values = tuple(float(value) for value in center[:3])
            if all(math.isfinite(value) for value in values):
                return values
    return None


def _vector(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return (0.0,) * length
    result = tuple(float(item) for item in value[:length])
    return result if all(math.isfinite(item) for item in result) else (0.0,) * length


def _quat_yaw(quat: Sequence[float]) -> float:
    if len(quat) != 4:
        return 0.0
    w, x, y, z = (float(value) for value in quat)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


__all__ = [
    "ARM_SUBTASKS",
    "NAV_SUBTASKS",
    "SUBTASK_DIRECTORY_LAYOUT",
    "SUBTASK_LABELS",
    "SUBTASK_SCHEMA_VERSION",
    "TASK_STAGES",
    "SubtaskSegmentationConfig",
    "extract_action_semantics",
    "hydrate_sample_action_semantics",
    "segment_episode_samples",
    "task_requests_subtask_segmentation",
    "validate_subtask_segmentation_config",
]
