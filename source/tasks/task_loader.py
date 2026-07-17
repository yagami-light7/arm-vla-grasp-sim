"""Adapt the existing NavPickTask schema to the new pipeline contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from source.data.task_schema import NavPickTask, ObjectPoseWorld, Pose2D
from source.interfaces import EpisodeSpec, NavGoal


def _pose_tuple(pose: ObjectPoseWorld | None) -> tuple[float, float, float, float, float, float] | None:
    if pose is None:
        return None
    return (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)


def _nav_goal(pose: Pose2D) -> NavGoal:
    return NavGoal(
        x=pose.x,
        y=pose.y,
        yaw=pose.yaw,
        z=pose.z,
        floor_id=pose.floor_id,
        slice_id=pose.slice_id,
    )


class JsonTaskProvider:
    """加载现有任务 JSON，并合并其显式引用的标注配置。"""

    def load(self, path: str | Path) -> EpisodeSpec:
        task_path = Path(path).expanduser().resolve()
        raw_task = json.loads(task_path.read_text(encoding="utf-8"))
        raw_task = _merge_annotation_config(raw_task, task_path=task_path)
        return episode_spec_from_dict(raw_task)


def _deep_merge_mapping(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置对象；列表与标量由标注文件整体覆盖。"""

    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_mapping(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _merge_annotation_config(
    raw_task: dict[str, Any],
    *,
    task_path: Path,
) -> dict[str, Any]:
    """把任务引用的 pick/place 标注文件作为任务专属字段唯一真值。"""

    annotation_raw = raw_task.get("annotation_config")
    if annotation_raw is None:
        return raw_task
    if not isinstance(annotation_raw, str) or not annotation_raw.strip():
        raise ValueError("task.annotation_config 必须是非空路径字符串")
    annotation_path = Path(annotation_raw).expanduser()
    if not annotation_path.is_absolute():
        annotation_path = task_path.parent / annotation_path
    annotation_path = annotation_path.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"task annotation_config 不存在: {annotation_path}")
    raw_bytes = annotation_path.read_bytes()
    try:
        annotation = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"task annotation_config 不是有效 UTF-8 JSON: {annotation_path}") from exc
    if not isinstance(annotation, dict):
        raise ValueError("task annotation_config 顶层必须是对象")
    overrides = annotation.get("task_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("task annotation_config.task_overrides 必须是对象")
    merged = _deep_merge_mapping(raw_task, overrides)
    merged["annotation_config"] = annotation_raw
    merged["annotation_config_report"] = {
        "path": str(annotation_path),
        "schema_version": annotation.get("schema_version"),
        "annotation_id": annotation.get("annotation_id"),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "task_overrides_applied": True,
    }
    return merged


def episode_spec_from_dict(raw_task: dict[str, Any]) -> EpisodeSpec:
    """将内存中的任务字典规范化为 pipeline EpisodeSpec。"""

    raw_task = dict(raw_task)
    task = NavPickTask.from_dict(raw_task)
    place_goal = None
    if task.place.enabled and task.place.base_goal is not None:
        place_goal = _nav_goal(task.place.base_goal)
    return EpisodeSpec(
        task_id=task.task_id,
        episode_id=task.episode_id,
        instruction=task.instruction,
        scene_usd=task.scene_usd,
        nav_map=task.nav_map,
        start=_nav_goal(task.start),
        pick_goal=_nav_goal(task.pick.base_goal),
        place_goal=place_goal,
        object_prim_path=task.pick.object_prim_path,
        object_initial_pose=_pose_tuple(task.pick.object_pose_world),
        place_target_pose=_pose_tuple(task.place.place_pose_world),
        raw_task=raw_task,
    )
