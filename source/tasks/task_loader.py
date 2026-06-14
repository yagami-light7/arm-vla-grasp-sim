"""Adapt the existing NavPickTask schema to the new pipeline contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from source.data.task_schema import NavPickTask, ObjectPoseWorld
from source.interfaces import EpisodeSpec, NavGoal


def _pose_tuple(pose: ObjectPoseWorld | None) -> tuple[float, float, float, float, float, float] | None:
    if pose is None:
        return None
    return (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)


class JsonTaskProvider:
    """Load current task JSON files without introducing a second schema."""

    def load(self, path: str | Path) -> EpisodeSpec:
        task_path = Path(path).expanduser().resolve()
        raw_task = json.loads(task_path.read_text(encoding="utf-8"))
        return episode_spec_from_dict(raw_task)


def episode_spec_from_dict(raw_task: dict[str, Any]) -> EpisodeSpec:
    """将内存中的任务字典规范化为 pipeline EpisodeSpec。"""

    raw_task = dict(raw_task)
    task = NavPickTask.from_dict(raw_task)
    place_goal = None
    if task.place.enabled and task.place.base_goal is not None:
        place_goal = NavGoal(
            x=task.place.base_goal.x,
            y=task.place.base_goal.y,
            yaw=task.place.base_goal.yaw,
        )
    return EpisodeSpec(
        task_id=task.task_id,
        episode_id=task.episode_id,
        instruction=task.instruction,
        scene_usd=task.scene_usd,
        nav_map=task.nav_map,
        start=NavGoal(x=task.start.x, y=task.start.y, yaw=task.start.yaw),
        pick_goal=NavGoal(
            x=task.pick.base_goal.x,
            y=task.pick.base_goal.y,
            yaw=task.pick.base_goal.yaw,
        ),
        place_goal=place_goal,
        object_prim_path=task.pick.object_prim_path,
        object_initial_pose=_pose_tuple(task.pick.object_pose_world),
        place_target_pose=_pose_tuple(task.place.place_pose_world),
        raw_task=raw_task,
    )
