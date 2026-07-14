"""LeRobot v2.1 数据集结构与时序一致性校验。"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .training_action import (
    VLA_TRAINING_ACTION_ALIGNMENT,
    VLA_TRAINING_ACTION_DIMENSION,
    VLA_TRAINING_ACTION_NAMES,
    VLA_TRAINING_ACTION_SCHEMA,
)


REQUIRED_PARQUET_COLUMNS = (
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "observation.state",
    "observation.base_velocity",
    "pipeline_state",
    "action",
    "next.done",
)

REQUIRED_INFO_FEATURES = (
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "observation.state",
    "observation.base_velocity",
    "pipeline_state",
    "action",
    "next.done",
)

_EPISODE_PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")


@dataclass
class _ValidationContext:
    dataset_root: Path
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def error(
        self,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        episode_index: int | None = None,
        feature: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"code": code, "message": message}
        if path is not None:
            payload["path"] = str(path)
        if episode_index is not None:
            payload["episode_index"] = int(episode_index)
        if feature is not None:
            payload["feature"] = feature
        self.errors.append(payload)

    def warning(self, code: str, message: str, *, path: Path | None = None) -> None:
        payload: dict[str, Any] = {"code": code, "message": message}
        if path is not None:
            payload["path"] = str(path)
        self.warnings.append(payload)


def _load_json_object(
    path: Path,
    context: _ValidationContext,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            context.error("missing_json", f"缺少 JSON 文件：{path}", path=path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        context.error("invalid_json", f"无法读取 JSON：{exc}", path=path)
        return None
    if not isinstance(payload, dict):
        context.error("invalid_json_object", "JSON 顶层必须是对象。", path=path)
        return None
    return payload


def _load_jsonl(path: Path, context: _ValidationContext) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        context.error("invalid_jsonl", f"无法读取 JSONL：{exc}", path=path)
        return []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            context.error(
                "invalid_jsonl",
                f"第 {line_number} 行不是有效 JSON：{exc}",
                path=path,
            )
            continue
        if not isinstance(payload, dict):
            context.error(
                "invalid_jsonl_record",
                f"第 {line_number} 行顶层必须是对象。",
                path=path,
            )
            continue
        records.append(payload)
    return records


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def _feature_dimension(feature: dict[str, Any]) -> int | None:
    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    dimension = 1
    for item in shape:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return None
        dimension *= item
    return dimension


def _validate_info(
    info: dict[str, Any],
    context: _ValidationContext,
) -> tuple[float | None, dict[str, dict[str, Any]], list[str]]:
    fps_value = info.get("fps")
    fps: float | None = None
    if (
        isinstance(fps_value, bool)
        or not isinstance(fps_value, (int, float))
        or not math.isfinite(float(fps_value))
        or float(fps_value) <= 0.0
    ):
        context.error("invalid_info_fps", "info.json 的 fps 必须是正数。")
    else:
        fps = float(fps_value)

    for key in ("data_path", "video_path"):
        if not isinstance(info.get(key), str) or not str(info[key]).strip():
            context.error(
                "invalid_info_path_template",
                f"info.json 的 {key} 必须是非空路径模板。",
            )

    raw_features = info.get("features")
    if not isinstance(raw_features, dict):
        context.error("invalid_info_features", "info.json 的 features 必须是对象。")
        return fps, {}, []

    features: dict[str, dict[str, Any]] = {}
    for name, raw_feature in raw_features.items():
        if not isinstance(name, str) or not isinstance(raw_feature, dict):
            context.error(
                "invalid_info_feature",
                f"feature {name!r} 的定义必须是对象。",
            )
            continue
        features[name] = raw_feature
        dtype = raw_feature.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            context.error(
                "invalid_feature_dtype",
                f"feature {name} 缺少有效 dtype。",
                feature=name,
            )
        if _feature_dimension(raw_feature) is None:
            context.error(
                "invalid_feature_shape",
                f"feature {name} 的 shape 必须由正整数组成。",
                feature=name,
            )

    for name in REQUIRED_INFO_FEATURES:
        if name not in features:
            context.error(
                "missing_info_feature",
                f"info.json 缺少 feature：{name}",
                feature=name,
            )

    for name in ("observation.state", "action"):
        feature = features.get(name)
        if feature is None:
            continue
        dimension = _feature_dimension(feature)
        names = feature.get("names")
        if not isinstance(names, list):
            context.error(
                "missing_feature_names",
                f"feature {name} 必须提供 names 列表。",
                feature=name,
            )
    metadata_names = {
        "observation.state": "observation_state_names",
        "observation.base_velocity": "base_velocity_names",
        "observation.base_pose": "base_pose_names",
        "observation.object_state": "object_state_names",
        "observation.tcp_pose": "tcp_pose_names",
        "control.action": "control_action_names",
        "action": "action_names",
    }
    for feature_name, metadata_name in metadata_names.items():
        feature = features.get(feature_name)
        if feature is None:
            continue
        feature_names = feature.get("names")
        metadata_value = info.get(metadata_name)
        if feature_name in {
            "observation.base_pose",
            "observation.object_state",
            "observation.tcp_pose",
            "control.action",
        }:
            # 扩展 feature 在旧数据中可缺失；出现时才要求顶层 names metadata。
            if metadata_value is None:
                context.error(
                    "missing_dimension_names_metadata",
                    f"info.json 缺少 {metadata_name} 列表。",
                    feature=feature_name,
                )
                continue
        if not isinstance(metadata_value, list):
            context.error(
                "missing_dimension_names_metadata",
                f"info.json 缺少 {metadata_name} 列表。",
                feature=feature_name,
            )
        elif isinstance(feature_names, list) and metadata_value != feature_names:
            context.error(
                "dimension_names_metadata_mismatch",
                f"{metadata_name} 与 feature {feature_name}.names 不一致。",
                feature=feature_name,
            )
        dimension = _feature_dimension(feature)
        if (
            isinstance(feature_names, list)
            and dimension is not None
            and len(feature_names) != dimension
        ):
            context.error(
                "feature_names_count_mismatch",
                f"feature {feature_name} 的 names 数量 {len(feature_names)} 与维度 {dimension} 不一致。",
                feature=feature_name,
            )

    if info.get("vla_training_action_available") is True:
        vla_requirements = {
            "vla_training_action_schema": VLA_TRAINING_ACTION_SCHEMA,
            "training_action_alignment": VLA_TRAINING_ACTION_ALIGNMENT,
            "training_action_horizon_frames": 1,
            "base_pose_frame": "world",
            "tcp_pose_frame": "world",
            "training_action_base_pose_frame": "world",
            "training_action_tcp_pose_frame": "base_frame",
            "training_action_tcp_euler_order": "roll_pitch_yaw",
            "training_action_position_unit": "m",
            "training_action_angle_unit": "rad",
        }
        for key, expected in vla_requirements.items():
            if info.get(key) != expected:
                context.error(
                    "invalid_vla_training_metadata",
                    f"info.json 的 {key} 必须是 {expected!r}。",
                    feature="action",
                )
        if info.get("training_action_gripper_range") != [0.0, 1.0]:
            context.error(
                "invalid_vla_training_metadata",
                "info.json 的 training_action_gripper_range 必须是 [0.0, 1.0]。",
                feature="action",
            )
        action_feature = features.get("action")
        if (
            action_feature is None
            or _feature_dimension(action_feature) != VLA_TRAINING_ACTION_DIMENSION
            or action_feature.get("names") != list(VLA_TRAINING_ACTION_NAMES)
        ):
            context.error(
                "invalid_vla_training_action_feature",
                "VLA 数据集的 action 必须是已命名的标准 10 维向量。",
                feature="action",
            )
        for required_feature in ("observation.base_pose", "control.action"):
            if required_feature not in features:
                context.error(
                    "missing_vla_support_feature",
                    f"VLA 数据集缺少 {required_feature}。",
                    feature=required_feature,
                )

    image_features = sorted(
        name for name in features if name.startswith("observation.images.")
    )
    if not image_features:
        context.error(
            "missing_image_feature",
            "info.json 至少需要一个 observation.images.* 视频 feature。",
        )
    for name in image_features:
        if str(features[name].get("dtype", "")).lower() != "video":
            context.error(
                "image_feature_not_video",
                f"图像 feature {name} 必须使用 video dtype。",
                feature=name,
            )
        video_info = features[name].get("video_info")
        video_fps = video_info.get("video.fps") if isinstance(video_info, dict) else None
        if fps is not None and video_fps is not None:
            if (
                isinstance(video_fps, bool)
                or not isinstance(video_fps, (int, float))
                or not math.isclose(float(video_fps), fps, rel_tol=1.0e-6, abs_tol=1.0e-6)
            ):
                context.error(
                    "image_feature_fps_mismatch",
                    f"图像 feature {name} 的 video.fps 与 info fps 不一致。",
                    feature=name,
                )
    return fps, features, image_features


def _render_dataset_path(
    dataset_root: Path,
    template: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path | None:
    values = {
        "episode_index": int(episode_index),
        "episode_chunk": int(episode_index) // int(chunks_size),
        "video_key": video_key or "",
    }
    try:
        rendered = template.format(**values)
    except (KeyError, IndexError, ValueError):
        return None
    relative = Path(rendered)
    if relative.is_absolute():
        return None
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError:
        return None
    return resolved


def _discover_parquets(dataset_root: Path) -> dict[int, list[Path]]:
    discovered: dict[int, list[Path]] = {}
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        return discovered
    for path in sorted(data_root.rglob("*.parquet")):
        match = _EPISODE_PARQUET_RE.search(path.name)
        if match is None:
            continue
        discovered.setdefault(int(match.group(1)), []).append(path.resolve())
    return discovered


def _metadata_indices(
    records: list[dict[str, Any]],
    key: str,
    context: _ValidationContext,
    *,
    path: Path,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            context.error(
                "invalid_metadata_index",
                f"{path.name} 中的 {key} 必须是非负整数。",
                path=path,
            )
            continue
        index = int(value)
        if index in indexed:
            context.error(
                "duplicate_metadata_index",
                f"{path.name} 中存在重复 {key}={index}。",
                path=path,
            )
            continue
        indexed[index] = record
    return indexed


def _column_values(table: Any, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _check_integer_sequence(
    values: list[Any],
    expected: list[int],
) -> bool:
    if len(values) != len(expected):
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, int)
        and int(value) == expected_value
        for value, expected_value in zip(values, expected)
    )


def _check_vector_column(
    values: list[Any],
    *,
    expected_dimension: int | None,
    feature: str,
    episode_index: int,
    context: _ValidationContext,
) -> int | None:
    dimensions: set[int] = set()
    for value in values:
        if not isinstance(value, (list, tuple)):
            context.error(
                "invalid_vector_value",
                f"{feature} 包含非向量值。",
                episode_index=episode_index,
                feature=feature,
            )
            return None
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        ):
            context.error(
                "non_finite_vector_value",
                f"{feature} 包含非有限数值。",
                episode_index=episode_index,
                feature=feature,
            )
            return None
        dimensions.add(len(value))
    if len(dimensions) != 1:
        context.error(
            "inconsistent_vector_dimension",
            f"{feature} 的行维度不一致：{sorted(dimensions)}",
            episode_index=episode_index,
            feature=feature,
        )
        return None
    actual = next(iter(dimensions), 0)
    if expected_dimension is not None and actual != expected_dimension:
        context.error(
            "vector_dimension_mismatch",
            f"{feature} 的 Parquet 维度 {actual} 与 info 维度 {expected_dimension} 不一致。",
            episode_index=episode_index,
            feature=feature,
        )
    return actual


def _arrow_value_dtype(field_type: Any) -> Any:
    import pyarrow as pa

    if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        return field_type.value_type
    if pa.types.is_fixed_size_list(field_type):
        return field_type.value_type
    return field_type


def _arrow_dtype_matches(field_type: Any, dtype: str) -> bool:
    import pyarrow as pa

    value_type = _arrow_value_dtype(field_type)
    normalized = dtype.lower()
    checks = {
        "float32": pa.types.is_float32,
        "float64": pa.types.is_float64,
        "int64": pa.types.is_int64,
        "int32": pa.types.is_int32,
        "bool": pa.types.is_boolean,
        "boolean": pa.types.is_boolean,
        "string": lambda item: pa.types.is_string(item) or pa.types.is_large_string(item),
    }
    checker = checks.get(normalized)
    return True if checker is None else bool(checker(value_type))


def _probe_video(video_path: Path) -> dict[str, float | int]:
    """读取视频帧数和帧率；OpenCV 不可用时回退到 ffprobe。"""

    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            if capture.isOpened():
                frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if frame_count > 0 and math.isfinite(fps) and fps > 0.0:
                    return {"frame_count": frame_count, "fps": fps}
        finally:
            capture.release()
    except ImportError:
        pass

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("视频中没有可读取的视频流")
    stream = streams[0]
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    frame_count = int(raw_count)
    numerator, denominator = str(stream["avg_frame_rate"]).split("/", maxsplit=1)
    fps = float(numerator) / float(denominator)
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("视频帧数或帧率无效")
    return {"frame_count": frame_count, "fps": fps}


def _validate_episode_table(
    table: Any,
    *,
    parquet_path: Path,
    episode_index: int,
    fps: float | None,
    features: dict[str, dict[str, Any]],
    image_features: list[str],
    vla_training_action_available: bool,
    context: _ValidationContext,
) -> dict[str, Any]:
    row_count = int(table.num_rows)
    episode_report: dict[str, Any] = {
        "episode_index": episode_index,
        "parquet_path": str(parquet_path),
        "row_count": row_count,
        "videos": [],
    }
    if row_count <= 0:
        context.error(
            "empty_parquet",
            "Parquet 不允许为空。",
            path=parquet_path,
            episode_index=episode_index,
        )
        return episode_report

    column_names = set(table.column_names)
    missing_columns = [
        name for name in REQUIRED_PARQUET_COLUMNS if name not in column_names
    ]
    for name in missing_columns:
        context.error(
            "missing_parquet_column",
            f"Parquet 缺少列：{name}",
            path=parquet_path,
            episode_index=episode_index,
            feature=name,
        )
    for image_feature in image_features:
        if image_feature in column_names:
            context.error(
                "image_feature_stored_in_parquet",
                f"{image_feature} 应只由视频承载，不应写入 Parquet。",
                path=parquet_path,
                episode_index=episode_index,
                feature=image_feature,
            )
    if missing_columns:
        return episode_report

    for name, feature in features.items():
        if name not in column_names or name in image_features:
            continue
        dtype = feature.get("dtype")
        if isinstance(dtype, str):
            field_type = table.schema.field(name).type
            if not _arrow_dtype_matches(field_type, dtype):
                context.error(
                    "parquet_dtype_mismatch",
                    f"{name} 的 Parquet 类型 {field_type} 与 info dtype={dtype} 不一致。",
                    path=parquet_path,
                    episode_index=episode_index,
                    feature=name,
                )

    episode_values = _column_values(table, "episode_index")
    if not _check_integer_sequence(episode_values, [episode_index] * row_count):
        context.error(
            "episode_index_mismatch",
            f"episode_index 列必须全部等于 {episode_index}。",
            path=parquet_path,
            episode_index=episode_index,
        )

    frame_values = _column_values(table, "frame_index")
    if not _check_integer_sequence(frame_values, list(range(row_count))):
        context.error(
            "frame_index_not_continuous",
            "每个 episode 的 frame_index 必须从 0 连续递增。",
            path=parquet_path,
            episode_index=episode_index,
        )

    timestamp_values = _column_values(table, "timestamp")
    numeric_timestamps: list[float] = []
    timestamps_valid = True
    for value in timestamp_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            timestamps_valid = False
            break
        numeric_timestamps.append(float(value))
    if not timestamps_valid:
        context.error(
            "invalid_timestamp",
            "timestamp 必须全部是有限数值。",
            path=parquet_path,
            episode_index=episode_index,
        )
    else:
        if any(
            current <= previous
            for previous, current in zip(numeric_timestamps, numeric_timestamps[1:])
        ):
            context.error(
                "timestamp_not_monotonic",
                "timestamp 必须在 episode 内严格递增。",
                path=parquet_path,
                episode_index=episode_index,
            )
        if fps is not None:
            tolerance = max(1.0e-5, (1.0 / fps) * 1.0e-4)
            if any(
                not math.isclose(
                    timestamp,
                    frame_index / fps,
                    rel_tol=1.0e-5,
                    abs_tol=tolerance,
                )
                for frame_index, timestamp in enumerate(numeric_timestamps)
            ):
                context.error(
                    "timestamp_fps_mismatch",
                    "timestamp 必须与 frame_index / info.fps 对齐。",
                    path=parquet_path,
                    episode_index=episode_index,
                )

    done_values = _column_values(table, "next.done")
    expected_done = [False] * (row_count - 1) + [True]
    if done_values != expected_done:
        context.error(
            "next_done_invalid",
            "next.done 只能在 episode 末帧为 true。",
            path=parquet_path,
            episode_index=episode_index,
        )

    task_values = _column_values(table, "task_index")
    valid_task_values = [
        int(value)
        for value in task_values
        if not isinstance(value, bool) and isinstance(value, int) and value >= 0
    ]
    if len(valid_task_values) != row_count:
        context.error(
            "invalid_task_index",
            "task_index 必须全部是非负整数。",
            path=parquet_path,
            episode_index=episode_index,
        )
    episode_report["task_indices"] = sorted(set(valid_task_values))

    pipeline_states = _column_values(table, "pipeline_state")
    if not all(isinstance(value, str) and bool(value) for value in pipeline_states):
        context.error(
            "invalid_pipeline_state",
            "pipeline_state 必须全部是非空字符串。",
            path=parquet_path,
            episode_index=episode_index,
        )

    for feature_name in (
        "observation.state",
        "observation.base_velocity",
        "observation.base_pose",
        "observation.object_state",
        "observation.tcp_pose",
        "control.action",
        "action",
    ):
        if feature_name not in features or feature_name not in column_names:
            continue
        feature = features.get(feature_name, {})
        _check_vector_column(
            _column_values(table, feature_name),
            expected_dimension=_feature_dimension(feature),
            feature=feature_name,
            episode_index=episode_index,
            context=context,
        )

    if vla_training_action_available:
        action_values = _column_values(table, "action")
        if any(
            len(value) != VLA_TRAINING_ACTION_DIMENSION
            or float(value[-1]) < 0.0
            or float(value[-1]) > 1.0
            for value in action_values
            if isinstance(value, (list, tuple))
        ):
            context.error(
                "invalid_vla_gripper_action",
                "VLA action 的 gripper_normalized 必须位于 [0, 1]。",
                path=parquet_path,
                episode_index=episode_index,
                feature="action",
            )

    episode_report["global_indices"] = _column_values(table, "index")
    return episode_report


def _validate_video(
    *,
    video_path: Path,
    feature_name: str,
    episode_index: int,
    expected_frames: int,
    expected_fps: float | None,
    context: _ValidationContext,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "feature": feature_name,
        "path": str(video_path),
        "exists": video_path.is_file(),
    }
    if not video_path.is_file():
        context.error(
            "missing_video",
            f"缺少视频：{video_path}",
            path=video_path,
            episode_index=episode_index,
            feature=feature_name,
        )
        return report
    try:
        probe = _probe_video(video_path)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        context.error(
            "video_probe_failed",
            f"无法读取视频信息：{exc}",
            path=video_path,
            episode_index=episode_index,
            feature=feature_name,
        )
        return report

    report.update(probe)
    frame_count = int(probe["frame_count"])
    video_fps = float(probe["fps"])
    if frame_count != expected_frames:
        context.error(
            "video_frame_count_mismatch",
            f"视频帧数 {frame_count} 与 Parquet 行数 {expected_frames} 不一致。",
            path=video_path,
            episode_index=episode_index,
            feature=feature_name,
        )
    if expected_fps is not None and not math.isclose(
        video_fps,
        expected_fps,
        rel_tol=1.0e-3,
        abs_tol=1.0e-2,
    ):
        context.error(
            "video_fps_mismatch",
            f"视频 fps={video_fps:.6g} 与 info fps={expected_fps:.6g} 不一致。",
            path=video_path,
            episode_index=episode_index,
            feature=feature_name,
        )
    return report


def _validate_dataset(dataset_root: Path, context: _ValidationContext) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    info = _load_json_object(info_path, context, required=True)
    if info is None:
        return {}

    fps, features, image_features = _validate_info(info, context)
    chunks_size = _positive_int(info.get("chunks_size")) or 1000

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    episodes_meta = _metadata_indices(
        _load_jsonl(episodes_path, context),
        "episode_index",
        context,
        path=episodes_path,
    )
    tasks_meta = _metadata_indices(
        _load_jsonl(tasks_path, context),
        "task_index",
        context,
        path=tasks_path,
    )
    discovered = _discover_parquets(dataset_root)

    for episode_index, paths in discovered.items():
        if len(paths) > 1:
            context.error(
                "duplicate_episode_parquet",
                f"episode {episode_index} 存在多个 Parquet：{[str(path) for path in paths]}",
                episode_index=episode_index,
            )

    total_episodes = _positive_int(info.get("total_episodes"))
    if total_episodes is not None:
        expected_indices = list(range(total_episodes))
    elif episodes_meta:
        expected_indices = sorted(episodes_meta)
    else:
        expected_indices = sorted(discovered)
    if not expected_indices:
        context.error("missing_parquet", "数据集中没有可校验的 episode Parquet。")
        return {
            "info": info,
            "fps": fps,
            "features": sorted(features),
            "image_features": image_features,
        }
    if expected_indices != list(range(len(expected_indices))):
        context.error(
            "episode_indices_not_continuous",
            "episode index 必须从 0 连续递增。",
        )

    data_template = info.get("data_path")
    video_template = info.get("video_path")
    all_global_indices: list[Any] = []
    all_task_indices: set[int] = set()
    validated_episode_indices: list[int] = []

    for episode_index in expected_indices:
        expected_path = (
            _render_dataset_path(
                dataset_root,
                str(data_template),
                episode_index=episode_index,
                chunks_size=chunks_size,
            )
            if isinstance(data_template, str)
            else None
        )
        if expected_path is None:
            context.error(
                "invalid_data_path_template",
                "无法使用 info.data_path 生成安全的 Parquet 路径。",
                episode_index=episode_index,
            )
            paths = discovered.get(episode_index, [])
            parquet_path = paths[0] if paths else None
        elif expected_path.is_file():
            parquet_path = expected_path
        else:
            paths = discovered.get(episode_index, [])
            parquet_path = paths[0] if paths else None
            context.error(
                "missing_parquet",
                f"缺少 episode {episode_index} 的 Parquet。",
                path=expected_path,
                episode_index=episode_index,
            )
            if parquet_path is not None and parquet_path != expected_path:
                context.error(
                    "parquet_path_mismatch",
                    "Parquet 存在，但不符合 info.data_path。",
                    path=parquet_path,
                    episode_index=episode_index,
                )
        if parquet_path is None:
            continue

        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_path)
        except ImportError as exc:
            context.error(
                "missing_pyarrow",
                f"缺少 Parquet 校验依赖：{exc}",
                path=parquet_path,
                episode_index=episode_index,
            )
            break
        except Exception as exc:
            context.error(
                "parquet_read_failed",
                f"无法读取 Parquet：{exc}",
                path=parquet_path,
                episode_index=episode_index,
            )
            continue

        episode_report = _validate_episode_table(
            table,
            parquet_path=parquet_path,
            episode_index=episode_index,
            fps=fps,
            features=features,
            image_features=image_features,
            vla_training_action_available=(
                info.get("vla_training_action_available") is True
            ),
            context=context,
        )
        context.episodes.append(episode_report)
        validated_episode_indices.append(episode_index)
        all_global_indices.extend(episode_report.pop("global_indices", []))
        all_task_indices.update(episode_report.get("task_indices", []))

        meta = episodes_meta.get(episode_index)
        if meta is not None:
            metadata_length = meta.get("length")
            if (
                isinstance(metadata_length, bool)
                or not isinstance(metadata_length, int)
                or metadata_length != table.num_rows
            ):
                context.error(
                    "episode_length_mismatch",
                    f"episodes.jsonl length 与 Parquet 行数不一致："
                    f"{metadata_length!r} != {table.num_rows}",
                    path=episodes_path,
                    episode_index=episode_index,
                )

        if isinstance(video_template, str):
            for image_feature in image_features:
                video_path = _render_dataset_path(
                    dataset_root,
                    video_template,
                    episode_index=episode_index,
                    chunks_size=chunks_size,
                    video_key=image_feature,
                )
                if video_path is None:
                    context.error(
                        "invalid_video_path_template",
                        "无法使用 info.video_path 生成安全的视频路径。",
                        episode_index=episode_index,
                        feature=image_feature,
                    )
                    continue
                episode_report["videos"].append(
                    _validate_video(
                        video_path=video_path,
                        feature_name=image_feature,
                        episode_index=episode_index,
                        expected_frames=int(table.num_rows),
                        expected_fps=fps,
                        context=context,
                    )
                )

    extra_indices = sorted(set(discovered) - set(expected_indices))
    for episode_index in extra_indices:
        context.error(
            "unexpected_episode_parquet",
            f"发现 info/episodes 元数据之外的 episode {episode_index} Parquet。",
            episode_index=episode_index,
        )

    if episodes_meta and sorted(episodes_meta) != expected_indices:
        context.error(
            "episodes_metadata_mismatch",
            "episodes.jsonl 的 episode index 与数据集不一致。",
            path=episodes_path,
        )

    expected_global = list(range(len(all_global_indices)))
    if not _check_integer_sequence(all_global_indices, expected_global):
        context.error(
            "global_index_not_continuous",
            "全局 index 必须按 episode 顺序从 0 连续递增。",
        )

    total_frames = info.get("total_frames")
    if total_frames is not None and (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames != len(all_global_indices)
    ):
        context.error(
            "total_frames_mismatch",
            f"info.total_frames 与 Parquet 总行数不一致："
            f"{total_frames!r} != {len(all_global_indices)}",
        )

    if total_episodes is not None and total_episodes != len(validated_episode_indices):
        context.error(
            "total_episodes_mismatch",
            f"info.total_episodes 与已读取 episode 数量不一致："
            f"{total_episodes} != {len(validated_episode_indices)}",
        )

    if tasks_meta:
        unknown_tasks = sorted(all_task_indices - set(tasks_meta))
        if unknown_tasks:
            context.error(
                "unknown_task_index",
                f"Parquet 引用了 tasks.jsonl 中不存在的 task_index：{unknown_tasks}",
                path=tasks_path,
            )
        for episode_report in context.episodes:
            meta = episodes_meta.get(int(episode_report["episode_index"]))
            if meta is None:
                continue
            task_names = meta.get("tasks")
            if not isinstance(task_names, list):
                continue
            parquet_task_names = {
                tasks_meta[index].get("task")
                for index in episode_report.get("task_indices", [])
                if index in tasks_meta
            }
            if parquet_task_names != set(task_names):
                context.error(
                    "episode_task_mismatch",
                    "episodes.jsonl 的 tasks 与 Parquet task_index 不一致。",
                    path=episodes_path,
                    episode_index=int(episode_report["episode_index"]),
                )

    total_tasks = _positive_int(info.get("total_tasks"))
    if total_tasks is not None:
        expected_tasks = set(range(total_tasks))
        if all_task_indices != expected_tasks:
            context.error(
                "task_indices_mismatch",
                f"数据集 task_index 集合 {sorted(all_task_indices)} "
                f"与 info.total_tasks={total_tasks} 不一致。",
            )
        if tasks_meta and set(tasks_meta) != expected_tasks:
            context.error(
                "tasks_metadata_mismatch",
                "tasks.jsonl 的 task_index 与 info.total_tasks 不一致。",
                path=tasks_path,
            )

    return {
        "info": info,
        "fps": fps,
        "features": sorted(features),
        "image_features": image_features,
        "validated_episode_indices": validated_episode_indices,
        "total_rows": len(all_global_indices),
    }


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_lerobot_dataset(
    dataset_root: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """校验一个 LeRobot 数据集，并写出 validation_report.json。"""

    root = Path(dataset_root).expanduser().resolve()
    context = _ValidationContext(dataset_root=root)
    details: dict[str, Any] = {}
    if not root.is_dir():
        context.error("missing_dataset_root", f"数据集目录不存在：{root}", path=root)
    else:
        try:
            details = _validate_dataset(root, context)
        except Exception as exc:
            # 报告意外错误，避免 CLI 在没有 validation_report.json 的情况下退出。
            context.error(
                "validator_internal_error",
                f"校验器内部错误：{type(exc).__name__}: {exc}",
            )

    target = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else root / "validation_report.json"
    )
    is_valid = not context.errors
    report = {
        "valid": is_valid,
        "success": is_valid,
        "failure_reason": None if is_valid else "lerobot_validation_failed",
        "dataset_root": str(root),
        "report_path": str(target),
        "summary": {
            "episode_count": len(context.episodes),
            "row_count": sum(
                int(episode.get("row_count", 0)) for episode in context.episodes
            ),
            "error_count": len(context.errors),
            "warning_count": len(context.warnings),
        },
        "errors": context.errors,
        "warnings": context.warnings,
        "episodes": context.episodes,
        "details": details,
    }
    _write_report(target, report)
    return report


def resolve_episode_dataset_root(episode_dir: str | Path) -> Path:
    """解析 full-physics episode 目录中的 LeRobot 子数据集。"""

    root = Path(episode_dir).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    nested = root / "lerobot_dataset"
    if (nested / "meta" / "info.json").is_file():
        return nested
    return nested


def validate_lerobot_episode(
    episode_dir: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """校验一个 full-physics episode 下的 LeRobot 数据，并在 episode 目录写报告。"""

    episode_root = Path(episode_dir).expanduser().resolve()
    dataset_root = resolve_episode_dataset_root(episode_root)
    target = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else episode_root / "validation_report.json"
    )
    return validate_lerobot_dataset(dataset_root, report_path=target)


__all__ = [
    "REQUIRED_INFO_FEATURES",
    "REQUIRED_PARQUET_COLUMNS",
    "resolve_episode_dataset_root",
    "validate_lerobot_dataset",
    "validate_lerobot_episode",
]
