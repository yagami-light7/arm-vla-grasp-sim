"""按 task/episode/连续 subtask 目录导出 VLA CSV 与前视、腕部图像。"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .subtask_segmentation import (
    SUBTASK_DIRECTORY_LAYOUT,
    SUBTASK_LABELS,
    SUBTASK_SCHEMA_VERSION,
    TASK_STAGES,
)
from .training_action import (
    VLA_TRAINING_ACTION_NAMES,
    VLA_TRAINING_ACTION_SCHEMA,
    build_vla_observation_poses,
)


SUBTASK_TASK_COLUMNS = (
    "task_id",
    "episode_id",
    "source_task_episode_id",
    "instruction",
    "target_object_id",
    "target_object_class",
    "target_receptacle_id",
    "start_base_x",
    "start_base_y",
    "start_base_yaw",
    "goal_base_x",
    "goal_base_y",
    "goal_base_yaw",
    "start_x_world",
    "start_y_world",
    "start_z_world",
    "start_yaw_world",
    "pick_goal_x_world",
    "pick_goal_y_world",
    "pick_goal_z_world",
    "pick_goal_yaw_world",
    "place_goal_x_world",
    "place_goal_y_world",
    "place_goal_z_world",
    "place_goal_yaw_world",
    "goal_yaw",
    "scene_profile",
    "policy_profile",
    "dataset_schema_version",
    "subtask_schema_version",
    "subtask_directory_layout",
    "action_schema",
    "action_dimension",
    "base_pose_frame",
    "tcp_pose_frame",
    "tcp_euler_order",
    "angle_unit",
    "position_unit",
    "gripper_convention",
    "collection_status",
    "success",
    "failure_reason",
    "training_eligible",
    "source_episode_index",
    "seed",
    "frame_count",
    "subtask_segment_count",
    "subtask_segments_json",
)

_ACTION_COLUMNS = tuple(f"action_{name}" for name in VLA_TRAINING_ACTION_NAMES)
_CONTROL_ACTION_NAMES = (
    "base_cmd_vx",
    "base_cmd_vy",
    "base_cmd_wz",
    "arm_joint1_target",
    "arm_joint2_target",
    "arm_joint3_target",
    "arm_joint4_target",
    "arm_joint5_target",
    "arm_joint6_target",
    "gripper_joint7_target",
    "gripper_joint8_target",
)
_CONTROL_COLUMNS = tuple(f"control_{name}" for name in _CONTROL_ACTION_NAMES)
SUBTASK_DATA_COLUMNS = (
    "global_frame_index",
    "dataset_global_index",
    "episode_frame_index",
    "subtask_frame_index",
    "timestamp",
    "task_stage",
    "subtask",
    "segment_index",
    "segment_frame_count",
    "segment_global_start_frame",
    "segment_global_end_frame",
    "previous_segment_index",
    "next_segment_index",
    "retained_short_reason",
    "instruction",
    "image_front_path",
    "image_wrist_path",
    "pipeline_state",
    "label_source",
    "contact_label_source",
    "action_source",
    *_ACTION_COLUMNS,
    "observation_base_x_world",
    "observation_base_y_world",
    "observation_base_z_world",
    "observation_base_quat_w_world",
    "observation_base_quat_x_world",
    "observation_base_quat_y_world",
    "observation_base_quat_z_world",
    "observation_base_yaw_world",
    "observation_base_vx_body",
    "observation_base_vy_body",
    "observation_base_wz_body",
    "observation_tcp_x_world",
    "observation_tcp_y_world",
    "observation_tcp_z_world",
    "observation_tcp_quat_w_world",
    "observation_tcp_quat_x_world",
    "observation_tcp_quat_y_world",
    "observation_tcp_quat_z_world",
    "observation_tcp_x_base",
    "observation_tcp_y_base",
    "observation_tcp_z_base",
    "observation_tcp_roll_base",
    "observation_tcp_pitch_base",
    "observation_tcp_yaw_base",
    "observation_gripper_joint_mean_m",
    "observation_gripper_normalized",
    *_CONTROL_COLUMNS,
)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SUBTASK_IMAGE_CAMERA_KEYS = ("front", "wrist")


def materialize_subtask_episode(
    *,
    source_episode_dir: str | Path,
    dataset_root: str | Path,
    task: dict[str, Any],
    episode_index: int,
    global_frame_offset: int,
    rows: Sequence[dict[str, str]],
    samples: Sequence[dict[str, Any]],
    training_actions: np.ndarray,
    control_actions: np.ndarray,
    segmentation: dict[str, Any],
    dataset_schema_version: str,
    source_gripper_joint_range_m: tuple[float, float],
) -> dict[str, Any]:
    """写出图片所示的 task.csv 与 episode-subtask 目录结构。"""

    source_dir = Path(source_episode_dir).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    frame_count = len(samples)
    if not (
        len(rows)
        == frame_count
        == int(training_actions.shape[0])
        == int(control_actions.shape[0])
        == len(segmentation.get("frames", []))
    ):
        raise ValueError("subtask export inputs must contain the same frame count")
    if training_actions.shape[1] != len(VLA_TRAINING_ACTION_NAMES):
        raise ValueError("subtask export requires the standard 10D training action")
    if control_actions.shape[1] != len(_CONTROL_ACTION_NAMES):
        raise ValueError("subtask export requires the 11D raw control action")
    observation_poses = build_vla_observation_poses(
        samples,
        source_gripper_joint_range_m=source_gripper_joint_range_m,
    )

    task_id = _safe_identifier(task.get("task_id", 0), name="task_id")
    episode_id = _safe_identifier(
        task.get("episode_id", episode_index + 1),
        name="episode_id",
    )
    episode_root = root / "episodes" / task_id / episode_id
    if episode_root.exists():
        shutil.rmtree(episode_root)
    episode_root.mkdir(parents=True, exist_ok=True)

    frames = list(segmentation["frames"])
    segments = list(segmentation["segments"])
    segment_reports: list[dict[str, Any]] = []
    for segment in segments:
        segment_index = int(segment["segment_index"])
        segment_dir_name = f"{episode_id}-{segment_index}"
        segment_dir = episode_root / segment_dir_name
        image_dirs = {
            camera_key: segment_dir / "images" / camera_key
            for camera_key in _SUBTASK_IMAGE_CAMERA_KEYS
        }
        for image_dir in image_dirs.values():
            image_dir.mkdir(parents=True, exist_ok=True)
        start = int(segment["global_start_frame"])
        end = int(segment["global_end_frame"])
        data_path = segment_dir / "data.csv"
        with data_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBTASK_DATA_COLUMNS)
            writer.writeheader()
            for local_index, frame_index in enumerate(range(start, end + 1)):
                image_name = f"camera0_{local_index:05d}.jpg"
                for camera_key, image_dir in image_dirs.items():
                    source_image = _camera_image_path(
                        source_dir,
                        row=rows[frame_index],
                        sample=samples[frame_index],
                        camera_key=camera_key,
                    )
                    if not source_image.is_file():
                        raise FileNotFoundError(
                            f"missing {camera_key} image for subtask frame "
                            f"{frame_index}: {source_image}"
                        )
                    shutil.copyfile(source_image, image_dir / image_name)
                writer.writerow(
                    _subtask_data_row(
                        frame_index=frame_index,
                        dataset_global_index=global_frame_offset + frame_index,
                        local_index=local_index,
                        instruction=str(task.get("instruction") or ""),
                        row=rows[frame_index],
                        sample=samples[frame_index],
                        annotation=frames[frame_index],
                        segment=segment,
                        training_action=training_actions[frame_index],
                        observation_pose=observation_poses[frame_index],
                        control_action=control_actions[frame_index],
                        image_name=image_name,
                    )
                )
        segment_reports.append(
            {
                **segment,
                "segment_dir_name": segment_dir_name,
                "relative_path": str(segment_dir.relative_to(root)),
                "data_csv": str(data_path.relative_to(root)),
                "image_count": end - start + 1,
                "image_counts": {
                    camera_key: end - start + 1
                    for camera_key in _SUBTASK_IMAGE_CAMERA_KEYS
                },
                "global_start_index": start,
                "global_end_index": end,
                "dataset_global_start_index": global_frame_offset + start,
                "dataset_global_end_index": global_frame_offset + end,
            }
        )

    task_csv_path = _write_task_csv(
        episode_root=episode_root,
        task=task,
        dataset_schema_version=dataset_schema_version,
        episode_index=episode_index,
        frame_count=frame_count,
        segments=segment_reports,
        source_episode_dir=source_dir,
    )
    return {
        "episode_index": episode_index,
        "task_id": task_id,
        "episode_id": episode_id,
        "source_task_episode_id": task.get(
            "_source_task_episode_id", task.get("episode_id")
        ),
        "episode_path": str(episode_root.relative_to(root)),
        "task_csv": str(task_csv_path.relative_to(root)),
        "frame_count": frame_count,
        "global_start_index": 0,
        "global_end_index": frame_count - 1,
        "dataset_global_start_index": global_frame_offset,
        "dataset_global_end_index": global_frame_offset + frame_count - 1,
        "segment_count": len(segment_reports),
        "segments": segment_reports,
        "frame_counts": dict(segmentation.get("frame_counts", {})),
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
        "contact_label_source": segmentation.get("contact_label_source"),
        "segmentation_config": dict(segmentation.get("config", {})),
    }


def write_subtask_task_stub(
    *,
    dataset_root: str | Path,
    task: dict[str, Any],
    dataset_schema_version: str,
    episode_index: int = 0,
) -> Path:
    """为尚未采集的 episode 只创建 task.csv，不伪造任何轨迹目录。"""

    root = Path(dataset_root).expanduser().resolve()
    task_id = _safe_identifier(task.get("task_id", 0), name="task_id")
    episode_id = _safe_identifier(
        task.get("episode_id", episode_index + 1),
        name="episode_id",
    )
    episode_root = root / "episodes" / task_id / episode_id
    if episode_root.exists():
        shutil.rmtree(episode_root)
    episode_root.mkdir(parents=True, exist_ok=True)
    return _write_task_csv(
        episode_root=episode_root,
        task=task,
        dataset_schema_version=dataset_schema_version,
        episode_index=episode_index,
        frame_count=0,
        segments=[],
        source_episode_dir=None,
    )


def update_subtask_task_gate(
    dataset_root: str | Path,
    *,
    eligible: bool,
    reason: str | None,
) -> None:
    """把最终物理质量门禁同步到自定义 task.csv。"""

    root = Path(dataset_root).expanduser().resolve()
    for task_csv in root.glob("episodes/*/*/task.csv"):
        with task_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            continue
        row = {column: rows[0].get(column, "") for column in SUBTASK_TASK_COLUMNS}
        row["training_eligible"] = _csv_bool(eligible)
        if eligible:
            row["collection_status"] = "accepted"
            row["success"] = "true"
            row["failure_reason"] = ""
        else:
            row["collection_status"] = "rejected"
            row["success"] = "false"
            row["failure_reason"] = str(reason or "training_quality_gate_not_verified")
        with task_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBTASK_TASK_COLUMNS)
            writer.writeheader()
            writer.writerow(row)


def validate_subtask_directory_export(
    dataset_root: str | Path,
) -> dict[str, Any]:
    """校验子任务目录的帧覆盖、动作、图像和顺序不变量。"""

    root = Path(dataset_root).expanduser().resolve()
    metadata_path = root / "meta" / "subtasks.jsonl"
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    if not metadata_path.is_file():
        return {
            "valid": False,
            "errors": [
                {
                    "code": "missing_subtask_metadata",
                    "message": f"缺少子任务元数据：{metadata_path}",
                    "path": str(metadata_path),
                }
            ],
            "warnings": warnings,
            "episodes": episode_reports,
        }

    records = []
    for line_number, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "code": "invalid_subtask_metadata",
                    "message": f"第 {line_number} 行 JSON 无效：{exc}",
                    "path": str(metadata_path),
                }
            )
            continue
        if not isinstance(payload, dict):
            errors.append(
                {
                    "code": "invalid_subtask_metadata",
                    "message": f"第 {line_number} 行必须是对象。",
                    "path": str(metadata_path),
                }
            )
            continue
        records.append(payload)

    seen_episode_paths: set[str] = set()
    for record in records:
        episode_path = str(record.get("episode_path") or "")
        if episode_path in seen_episode_paths:
            _validation_error(
                errors,
                "duplicate_subtask_episode_path",
                "subtasks.jsonl 中多个 episode 指向同一目录。",
                root / episode_path,
            )
        seen_episode_paths.add(episode_path)
        episode_reports.append(_validate_subtask_episode(root, record, errors))
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "episodes": episode_reports,
        "episode_count": len(episode_reports),
        "frame_count": sum(
            int(report.get("frame_count", 0)) for report in episode_reports
        ),
        "segment_count": sum(
            int(report.get("segment_count", 0)) for report in episode_reports
        ),
    }


def _write_task_csv(
    *,
    episode_root: Path,
    task: dict[str, Any],
    dataset_schema_version: str,
    episode_index: int,
    frame_count: int,
    segments: Sequence[dict[str, Any]],
    source_episode_dir: Path | None,
) -> Path:
    start = _mapping(task.get("start"))
    pick = _mapping(_mapping(task.get("pick")).get("base_goal"))
    place_config = _mapping(task.get("place"))
    place = _mapping(place_config.get("base_goal"))
    goal = place if place and bool(place_config.get("enabled", True)) else pick
    pick_config = _mapping(task.get("pick"))
    training = _mapping(task.get("training_action"))
    summary = _read_summary(source_episode_dir)
    if summary and summary.get("success") is True:
        training_eligible = bool(
            summary.get("lerobot_training_eligible") is True
        )
        collection_status = "collected"
        success = "true"
        failure_reason = ""
    elif summary:
        training_eligible = False
        collection_status = "rejected"
        success = "false"
        failure_reason = str(summary.get("failure_reason") or "episode_failed")
    elif frame_count:
        training_eligible = False
        collection_status = "collected"
        success = ""
        failure_reason = ""
    else:
        training_eligible = False
        collection_status = "planned"
        success = ""
        failure_reason = ""
    goal_yaw = goal.get("yaw")
    row = {
        "task_id": task.get("task_id", 0),
        "episode_id": task.get("episode_id", episode_index + 1),
        "source_task_episode_id": task.get(
            "_source_task_episode_id", task.get("episode_id", episode_index + 1)
        ),
        "instruction": str(task.get("instruction") or ""),
        "target_object_id": task.get("target_object_id")
        or pick_config.get("target_object_id")
        or "",
        "target_object_class": task.get("target_object_class")
        or pick_config.get("target_object_class")
        or "",
        "target_receptacle_id": task.get("target_receptacle_id")
        or place_config.get("target_receptacle_id")
        or "",
        "start_base_x": _csv_scalar(start.get("x")),
        "start_base_y": _csv_scalar(start.get("y")),
        "start_base_yaw": _csv_scalar(start.get("yaw")),
        "goal_base_x": _csv_scalar(goal.get("x")),
        "goal_base_y": _csv_scalar(goal.get("y")),
        "goal_base_yaw": _csv_scalar(goal.get("yaw")),
        **_pose_columns("start", start),
        **_pose_columns("pick_goal", pick),
        **_pose_columns("place_goal", place),
        "goal_yaw": _csv_scalar(goal_yaw),
        "scene_profile": task.get("scene_profile", ""),
        "policy_profile": task.get("policy_profile", ""),
        "dataset_schema_version": dataset_schema_version,
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_directory_layout": SUBTASK_DIRECTORY_LAYOUT,
        "action_schema": training.get("schema", VLA_TRAINING_ACTION_SCHEMA),
        "action_dimension": training.get("dimension", len(VLA_TRAINING_ACTION_NAMES)),
        "base_pose_frame": training.get("base_pose_frame", "world"),
        "tcp_pose_frame": training.get("tcp_pose_frame", "base_frame"),
        "tcp_euler_order": training.get("tcp_euler_order", "roll_pitch_yaw"),
        "angle_unit": training.get("angle_unit", "rad"),
        "position_unit": training.get("position_unit", "m"),
        "gripper_convention": "0_closed_1_open",
        "collection_status": collection_status,
        "success": success,
        "failure_reason": failure_reason,
        "training_eligible": (
            _csv_bool(training_eligible) if summary or frame_count else ""
        ),
        "source_episode_index": episode_index,
        "seed": summary.get("seed", "") if summary else "",
        "frame_count": frame_count,
        "subtask_segment_count": len(segments),
        "subtask_segments_json": json.dumps(
            list(segments),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    task_csv = episode_root / "task.csv"
    with task_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBTASK_TASK_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SUBTASK_TASK_COLUMNS})
    return task_csv


def _subtask_data_row(
    *,
    frame_index: int,
    dataset_global_index: int,
    local_index: int,
    instruction: str,
    row: dict[str, str],
    sample: dict[str, Any],
    annotation: dict[str, Any],
    segment: dict[str, Any],
    training_action: Sequence[float],
    observation_pose: Sequence[float],
    control_action: Sequence[float],
    image_name: str,
) -> dict[str, Any]:
    base_pose = _finite_vector(sample.get("base_pose"), 7, name="base_pose")
    base_velocity = _finite_vector(
        sample.get("base_velocity"), 3, name="base_velocity"
    )
    tcp_pose = _finite_vector(sample.get("tcp_pose"), 7, name="tcp_pose")
    gripper_position = float(sample.get("gripper_position"))
    observation_values = tuple(float(value) for value in observation_pose)
    if len(observation_values) != len(VLA_TRAINING_ACTION_NAMES) or not all(
        math.isfinite(value) for value in observation_values
    ):
        raise ValueError("subtask observation pose must be a finite 10D vector")
    payload: dict[str, Any] = {
        "global_frame_index": frame_index,
        "dataset_global_index": dataset_global_index,
        "episode_frame_index": frame_index,
        "subtask_frame_index": local_index,
        "timestamp": _float_text(sample.get("timestamp", 0.0)),
        "task_stage": annotation["task_stage"],
        "subtask": annotation["subtask"],
        "segment_index": segment["segment_index"],
        "segment_frame_count": segment["frame_count"],
        "segment_global_start_frame": segment["global_start_frame"],
        "segment_global_end_frame": segment["global_end_frame"],
        "previous_segment_index": segment.get("previous_segment_index") or "",
        "next_segment_index": segment.get("next_segment_index") or "",
        "retained_short_reason": segment.get("retained_short_reason") or "",
        "instruction": instruction,
        "image_front_path": f"images/front/{image_name}",
        "image_wrist_path": f"images/wrist/{image_name}",
        "pipeline_state": sample.get("pipeline_state") or row.get("pipeline_state") or "",
        "label_source": annotation.get("label_source", ""),
        "contact_label_source": annotation.get("contact_label_source", ""),
        "action_source": sample.get("action_source", ""),
        "observation_base_x_world": _float_text(base_pose[0]),
        "observation_base_y_world": _float_text(base_pose[1]),
        "observation_base_z_world": _float_text(base_pose[2]),
        "observation_base_quat_w_world": _float_text(base_pose[3]),
        "observation_base_quat_x_world": _float_text(base_pose[4]),
        "observation_base_quat_y_world": _float_text(base_pose[5]),
        "observation_base_quat_z_world": _float_text(base_pose[6]),
        "observation_base_yaw_world": _float_text(observation_values[2]),
        "observation_base_vx_body": _float_text(base_velocity[0]),
        "observation_base_vy_body": _float_text(base_velocity[1]),
        "observation_base_wz_body": _float_text(base_velocity[2]),
        "observation_tcp_x_world": _float_text(tcp_pose[0]),
        "observation_tcp_y_world": _float_text(tcp_pose[1]),
        "observation_tcp_z_world": _float_text(tcp_pose[2]),
        "observation_tcp_quat_w_world": _float_text(tcp_pose[3]),
        "observation_tcp_quat_x_world": _float_text(tcp_pose[4]),
        "observation_tcp_quat_y_world": _float_text(tcp_pose[5]),
        "observation_tcp_quat_z_world": _float_text(tcp_pose[6]),
        "observation_tcp_x_base": _float_text(observation_values[3]),
        "observation_tcp_y_base": _float_text(observation_values[4]),
        "observation_tcp_z_base": _float_text(observation_values[5]),
        "observation_tcp_roll_base": _float_text(observation_values[6]),
        "observation_tcp_pitch_base": _float_text(observation_values[7]),
        "observation_tcp_yaw_base": _float_text(observation_values[8]),
        "observation_gripper_joint_mean_m": _float_text(gripper_position),
        "observation_gripper_normalized": _float_text(observation_values[9]),
    }
    for name, value in zip(_ACTION_COLUMNS, training_action):
        payload[name] = _float_text(value)
    for name, value in zip(_CONTROL_COLUMNS, control_action):
        payload[name] = _float_text(value)
    return {column: payload.get(column, "") for column in SUBTASK_DATA_COLUMNS}


def _validate_subtask_episode(
    dataset_root: Path,
    record: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_path = dataset_root / str(record.get("episode_path") or "")
    task_csv = episode_path / "task.csv"
    report: dict[str, Any] = {
        "episode_index": record.get("episode_index"),
        "episode_path": str(episode_path),
        "frame_count": 0,
        "segment_count": 0,
        "global_indices": [],
        "dataset_global_indices": [],
        "task_stages": [],
        "subtasks": [],
        "segment_indices": [],
        "image_counts": {camera_key: 0 for camera_key in _SUBTASK_IMAGE_CAMERA_KEYS},
    }
    if not task_csv.is_file():
        _validation_error(
            errors,
            "missing_subtask_task_csv",
            f"缺少 task.csv：{task_csv}",
            task_csv,
        )
        return report
    with task_csv.open("r", encoding="utf-8", newline="") as stream:
        task_rows = list(csv.DictReader(stream))
    if len(task_rows) != 1 or tuple(task_rows[0]) != SUBTASK_TASK_COLUMNS:
        _validation_error(
            errors,
            "invalid_subtask_task_csv",
            "task.csv 必须恰好一行并使用固定列。",
            task_csv,
        )
    elif task_rows[0].get("collection_status") not in {
        "planned",
        "collected",
        "validated",
        "accepted",
        "rejected",
    }:
        _validation_error(
            errors,
            "invalid_subtask_collection_status",
            "task.csv 的 collection_status 不在正式状态枚举中。",
            task_csv,
        )

    segments = record.get("segments")
    if not isinstance(segments, list):
        _validation_error(
            errors,
            "invalid_subtask_segments",
            "subtasks.jsonl 的 segments 必须是列表。",
            task_csv,
        )
        return report
    reconstructed_indices: list[int] = []
    reconstructed_dataset_indices: list[int] = []
    reconstructed_episode_indices: list[int] = []
    reconstructed_stages: list[str] = []
    reconstructed_labels: list[str] = []
    reconstructed_segment_indices: list[int] = []
    episode_id = str(record.get("episode_id") or "")
    expected_segment_dirs: list[str] = []
    previous_pair: tuple[str, str] | None = None
    for expected_order, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            _validation_error(
                errors,
                "invalid_subtask_segment",
                f"segment {expected_order} 必须是对象。",
                task_csv,
            )
            continue
        expected_dir_name = f"{episode_id}-{expected_order}"
        segment_dir_name = str(segment.get("segment_dir_name") or "")
        expected_segment_dirs.append(expected_dir_name)
        if (
            int(segment.get("segment_index", -1)) != expected_order
            or segment_dir_name != expected_dir_name
        ):
            _validation_error(
                errors,
                "subtask_order_not_continuous",
                "segment_index 和目录名必须按 episode-id-1、episode-id-2 连续递增。",
                episode_path / segment_dir_name,
            )
        expected_previous = expected_order - 1 if expected_order > 1 else None
        expected_next = expected_order + 1 if expected_order < len(segments) else None
        if (
            segment.get("previous_segment_index") != expected_previous
            or segment.get("next_segment_index") != expected_next
        ):
            _validation_error(
                errors,
                "invalid_subtask_neighbors",
                "前后子任务编号与时间顺序不一致。",
                episode_path / segment_dir_name,
            )
        expected_stage = str(segment.get("task_stage") or "")
        expected_label = str(segment.get("subtask") or "")
        if not _stage_accepts_subtask(expected_stage, expected_label):
            _validation_error(
                errors,
                "invalid_subtask_stage_pair",
                f"task_stage/subtask 组合非法：{expected_stage}/{expected_label}",
                episode_path / segment_dir_name,
            )
        current_pair = (expected_stage, expected_label)
        if current_pair == previous_pair:
            _validation_error(
                errors,
                "adjacent_duplicate_subtask_segments",
                "相邻且标签相同的目录必须合并为一个连续 subtask。",
                episode_path / segment_dir_name,
            )
        previous_pair = current_pair
        segment_dir = episode_path / segment_dir_name
        data_csv = segment_dir / "data.csv"
        if not data_csv.is_file():
            _validation_error(
                errors,
                "missing_subtask_data_csv",
                f"缺少 data.csv：{data_csv}",
                data_csv,
            )
            continue
        with data_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            data_rows = list(reader)
            if tuple(reader.fieldnames or ()) != SUBTASK_DATA_COLUMNS:
                _validation_error(
                    errors,
                    "invalid_subtask_data_columns",
                    "subtask data.csv 列与 schema 不一致。",
                    data_csv,
                )
        expected_count = int(segment.get("frame_count", -1))
        if len(data_rows) != expected_count or not data_rows:
            _validation_error(
                errors,
                "subtask_row_count_mismatch",
                f"data.csv 行数 {len(data_rows)} 与 metadata {expected_count} 不一致。",
                data_csv,
            )
        local_indices: list[int] = []
        episode_indices: list[int] = []
        segment_pairs: set[tuple[str, str]] = set()
        row_segment_indices: set[int] = set()
        for data_row in data_rows:
            try:
                global_index = int(data_row["global_frame_index"])
                dataset_global_index = int(data_row["dataset_global_index"])
                episode_index = int(data_row["episode_frame_index"])
                local_index = int(data_row["subtask_frame_index"])
                row_segment_index = int(data_row["segment_index"])
            except (KeyError, TypeError, ValueError):
                _validation_error(
                    errors,
                    "invalid_subtask_frame_index",
                    "frame index 必须是整数。",
                    data_csv,
                )
                continue
            local_indices.append(local_index)
            episode_indices.append(episode_index)
            reconstructed_indices.append(global_index)
            reconstructed_dataset_indices.append(dataset_global_index)
            reconstructed_episode_indices.append(episode_index)
            row_segment_indices.add(row_segment_index)
            stage = data_row.get("task_stage", "")
            label = data_row.get("subtask", "")
            segment_pairs.add((stage, label))
            reconstructed_stages.append(stage)
            reconstructed_labels.append(label)
            reconstructed_segment_indices.append(row_segment_index)
            if not _stage_accepts_subtask(stage, label):
                _validation_error(
                    errors,
                    "invalid_subtask_stage_pair",
                    f"非法 task_stage/subtask：{stage}/{label}",
                    data_csv,
                )
            for camera_key in _SUBTASK_IMAGE_CAMERA_KEYS:
                image_column = f"image_{camera_key}_path"
                image_path = segment_dir / data_row.get(image_column, "")
                if not image_path.is_file():
                    _validation_error(
                        errors,
                        f"missing_subtask_{camera_key}_image",
                        f"缺少 {camera_key} 图像：{image_path}",
                        image_path,
                    )
                expected_image_path = (
                    f"images/{camera_key}/camera0_{local_index:05d}.jpg"
                )
                if data_row.get(image_column) != expected_image_path:
                    _validation_error(
                        errors,
                        f"invalid_subtask_{camera_key}_image_name",
                        f"片段内 {camera_key} 图像必须从 "
                        "camera0_00000.jpg 连续编号。",
                        data_csv,
                    )
            for column in _ACTION_COLUMNS:
                if not _finite_csv_float(data_row.get(column)):
                    _validation_error(
                        errors,
                        "invalid_subtask_action",
                        f"{column} 必须是有限数值。",
                        data_csv,
                    )
                    break
            gripper = data_row.get("action_gripper_normalized")
            if _finite_csv_float(gripper) and not 0.0 <= float(gripper) <= 1.0:
                _validation_error(
                    errors,
                    "invalid_subtask_gripper",
                    "action_gripper_normalized 必须位于 [0, 1]。",
                    data_csv,
                )
        if segment_pairs != {(expected_stage, expected_label)}:
            _validation_error(
                errors,
                "mixed_subtask_folder",
                "每个 subtask 文件夹只能包含 metadata 指定的一种 task_stage/subtask。",
                data_csv,
            )
        if row_segment_indices != {expected_order}:
            _validation_error(
                errors,
                "mixed_subtask_segment_index",
                "每个 subtask 文件夹只能包含自身的 segment_index。",
                data_csv,
            )
        if local_indices != list(range(len(data_rows))):
            _validation_error(
                errors,
                "subtask_local_index_not_continuous",
                "subtask_frame_index 必须从 0 连续递增。",
                data_csv,
            )
        expected_episode_start = int(segment.get("global_start_frame", -1))
        expected_episode_end = int(segment.get("global_end_frame", -1))
        if episode_indices != list(
            range(expected_episode_start, expected_episode_end + 1)
        ):
            _validation_error(
                errors,
                "subtask_episode_index_not_continuous",
                "episode_frame_index 与片段全局起止帧不一致。",
                data_csv,
            )
        image_root = segment_dir / "images"
        actual_image_directories = (
            {path.name for path in image_root.iterdir() if path.is_dir()}
            if image_root.is_dir()
            else set()
        )
        if actual_image_directories != set(_SUBTASK_IMAGE_CAMERA_KEYS):
            _validation_error(
                errors,
                "invalid_subtask_image_directories",
                "images 目录必须且只能包含 front 和 wrist 两个子目录。",
                image_root,
            )
        for camera_key in _SUBTASK_IMAGE_CAMERA_KEYS:
            camera_dir = image_root / camera_key
            image_count = len(list(camera_dir.glob("*.jpg")))
            if image_count != len(data_rows):
                _validation_error(
                    errors,
                    f"subtask_{camera_key}_image_count_mismatch",
                    f"{camera_key} 图像数 {image_count} 与 CSV 行数 "
                    f"{len(data_rows)} 不一致。",
                    camera_dir,
                )
            report["image_counts"][camera_key] += image_count
        min_segment_frames = int(
            _mapping(record.get("segmentation_config")).get(
                "min_segment_frames", 3
            )
        )
        if (
            expected_count < min_segment_frames
            and not segment.get("retained_short_reason")
        ):
            _validation_error(
                errors,
                "unexplained_short_subtask_segment",
                f"少于 {min_segment_frames} 帧的片段必须记录 retained_short_reason。",
                data_csv,
            )

    actual_segment_dirs = sorted(
        path.name for path in episode_path.iterdir() if path.is_dir()
    ) if episode_path.is_dir() else []
    if set(actual_segment_dirs) != set(expected_segment_dirs):
        _validation_error(
            errors,
            "unexpected_subtask_directories",
            "episode 下的 subtask 目录与 metadata 不完全一致。",
            episode_path,
        )

    expected_start = int(record.get("global_start_index", 0))
    expected_count = int(record.get("frame_count", 0))
    if reconstructed_indices != list(
        range(expected_start, expected_start + expected_count)
    ):
        _validation_error(
            errors,
            "subtask_reconstruction_failed",
            "按片段顺序重建后存在重复、缺失或乱序帧。",
            episode_path,
        )
    if reconstructed_episode_indices != list(range(expected_count)):
        _validation_error(
            errors,
            "subtask_episode_reconstruction_failed",
            "episode_frame_index 无法无损重建完整 episode。",
            episode_path,
        )
    expected_dataset_start = int(
        record.get("dataset_global_start_index", expected_start)
    )
    if reconstructed_dataset_indices != list(
        range(expected_dataset_start, expected_dataset_start + expected_count)
    ):
        _validation_error(
            errors,
            "subtask_dataset_reconstruction_failed",
            "dataset_global_index 与 LeRobot 合并数据集索引不一致。",
            episode_path,
        )
    report.update(
        {
            "frame_count": len(reconstructed_indices),
            "segment_count": len(segments),
            "global_indices": reconstructed_indices,
            "dataset_global_indices": reconstructed_dataset_indices,
            "task_stages": reconstructed_stages,
            "subtasks": reconstructed_labels,
            "segment_indices": reconstructed_segment_indices,
        }
    )
    return report


def _camera_image_path(
    episode_dir: Path,
    *,
    row: dict[str, str],
    sample: dict[str, Any],
    camera_key: str,
) -> Path:
    if camera_key not in _SUBTASK_IMAGE_CAMERA_KEYS:
        raise ValueError(f"unsupported subtask image camera: {camera_key}")
    camera_frames = sample.get("camera_frames")
    if isinstance(camera_frames, dict):
        camera = camera_frames.get(camera_key)
        if isinstance(camera, dict) and camera.get("raw_image_path"):
            return episode_dir / str(camera["raw_image_path"])
    row_column = {
        "front": "前摄像头图像",
        "wrist": "腕部摄像头图像",
    }[camera_key]
    image_name = row.get(row_column)
    if image_name:
        return episode_dir / "images" / camera_key / image_name
    return episode_dir / "images" / camera_key / "__missing__.jpg"


def _read_summary(episode_dir: Path | None) -> dict[str, Any]:
    if episode_dir is None:
        return {}
    path = episode_dir / "summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _pose_columns(prefix: str, pose: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_x_world": _csv_scalar(pose.get("x")),
        f"{prefix}_y_world": _csv_scalar(pose.get("y")),
        f"{prefix}_z_world": _csv_scalar(pose.get("z")),
        f"{prefix}_yaw_world": _csv_scalar(pose.get("yaw")),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stage_accepts_subtask(stage: str, label: str) -> bool:
    if stage not in TASK_STAGES or label not in SUBTASK_LABELS:
        return False
    if stage in {"nav_to_pick", "nav_to_place"}:
        return label.startswith("nav_")
    return label.startswith("arm_")


def _safe_identifier(value: Any, *, name: str) -> str:
    result = str(value)
    if (
        not result
        or result in {".", ".."}
        or not _SAFE_IDENTIFIER_RE.fullmatch(result)
    ):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")
    return result


def _finite_vector(value: Any, length: int, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _float_text(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("CSV numeric value must be finite")
    return f"{number:.9g}"


def _csv_scalar(value: Any) -> Any:
    return "" if value is None else value


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def _finite_csv_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validation_error(
    errors: list[dict[str, Any]],
    code: str,
    message: str,
    path: Path,
) -> None:
    errors.append({"code": code, "message": message, "path": str(path)})


__all__ = [
    "SUBTASK_DATA_COLUMNS",
    "SUBTASK_TASK_COLUMNS",
    "materialize_subtask_episode",
    "update_subtask_task_gate",
    "validate_subtask_directory_export",
    "write_subtask_task_stub",
]
