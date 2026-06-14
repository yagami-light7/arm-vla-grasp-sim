"""单进程 full-physics pipeline 的 episode 任务随机化。"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from source.data.random_task import SpawnRegion, sample_object_pose
from source.interfaces import EpisodeSpec

from .task_loader import episode_spec_from_dict

if TYPE_CHECKING:
    from source.pipeline.config import RandomizationSettings


def _randomize_pick_xy(
    task: dict[str, Any],
    *,
    seed: int,
    settings: RandomizationSettings,
) -> None:
    """只平移苹果和 pick base-goal 的 XY，其他字段严格保持模板值。"""

    pick = dict(task.get("pick") or {})
    object_pose = dict(pick.get("object_pose_world") or {})
    base_goal = dict(pick.get("base_goal") or {})
    if not {"x", "y", "z"}.issubset(object_pose):
        raise ValueError("pick.object_pose_world must contain x, y and z")
    if not {"x", "y"}.issubset(base_goal):
        raise ValueError("pick.base_goal must contain x and y")

    rng = random.Random(int(seed))
    sampled_pose = sample_object_pose(
        rng,
        SpawnRegion(
            x_min=settings.pick_x_range[0],
            x_max=settings.pick_x_range[1],
            y_min=settings.pick_y_range[0],
            y_max=settings.pick_y_range[1],
            table_z=float(object_pose["z"]),
        ),
        object_fixed_z=float(object_pose["z"]),
        # 采样器只负责复用 baseline 的 XY edge-biased 分布，返回姿态不会写入任务。
        edge_margin=settings.edge_margin if settings.edge_biased else None,
        edge_min_clearance=settings.edge_min_clearance,
    )
    before_pose = copy.deepcopy(object_pose)
    before_goal = copy.deepcopy(base_goal)
    dx = float(sampled_pose.x) - float(object_pose["x"])
    dy = float(sampled_pose.y) - float(object_pose["y"])
    object_pose["x"] = float(sampled_pose.x)
    object_pose["y"] = float(sampled_pose.y)
    base_goal["x"] = float(base_goal["x"]) + dx
    base_goal["y"] = float(base_goal["y"]) + dy
    pick["object_pose_world"] = object_pose
    pick["base_goal"] = base_goal
    task["pick"] = pick

    randomization = dict(task.get("randomization") or {})
    randomization.update(
        {
            "enabled": True,
            "seed": int(seed),
            "object_xy_randomization": {
                "enabled": True,
                "mode": "sample_xy_translate_base_goal",
                "x_range_m": list(settings.pick_x_range),
                "y_range_m": list(settings.pick_y_range),
                "sampled_xy": {
                    "x": float(sampled_pose.x),
                    "y": float(sampled_pose.y),
                },
                "delta_xy_m": [dx, dy],
                "object_pose_world_before": before_pose,
                "object_pose_world_after": copy.deepcopy(object_pose),
                "base_goal_before": before_goal,
                "base_goal_after": copy.deepcopy(base_goal),
                "selected_edge_side": sampled_pose.edge_side,
                "pose_policy": "严格只修改 x/y；z/roll/pitch/yaw 及其他字段原样保留",
                "nav_goal_rule": "只平移 base_goal x/y；yaw 及其他字段原样保留",
            },
            "object_pose_policy": {
                "mode": "xy_only",
                "randomize_xy": True,
                "randomize_z": False,
                "randomize_roll": False,
                "randomize_pitch": False,
                "randomize_yaw": False,
            },
        }
    )
    task["randomization"] = randomization


def _randomize_place_xy(
    task: dict[str, Any],
    *,
    seed: int,
    settings: RandomizationSettings,
) -> None:
    """随机平移 place 目标，同时保持 baseline 中 base-goal 的相对偏移。"""

    place = dict(task.get("place") or {})
    place_pose = dict(place.get("place_pose_world") or {})
    base_goal = dict(place.get("base_goal") or {})
    if not place.get("enabled") or not {"x", "y"}.issubset(place_pose):
        return

    rng = random.Random(int(seed) + settings.place_seed_offset)
    sampled_x = rng.uniform(*settings.place_x_range)
    sampled_y = rng.uniform(*settings.place_y_range)
    before_pose = copy.deepcopy(place_pose)
    before_goal = copy.deepcopy(base_goal)
    dx = sampled_x - float(place_pose["x"])
    dy = sampled_y - float(place_pose["y"])
    place_pose.update({"x": sampled_x, "y": sampled_y})
    if {"x", "y"}.issubset(base_goal):
        base_goal["x"] = float(base_goal["x"]) + dx
        base_goal["y"] = float(base_goal["y"]) + dy
        place["base_goal"] = base_goal
    place["place_pose_world"] = place_pose
    task["place"] = place

    randomization = dict(task.get("randomization") or {})
    randomization["place_xy_randomization"] = {
        "enabled": True,
        "mode": "sample_xy_translate_base_goal",
        "seed": int(seed) + settings.place_seed_offset,
        "x_range_m": list(settings.place_x_range),
        "y_range_m": list(settings.place_y_range),
        "sampled_xy": {"x": sampled_x, "y": sampled_y},
        "delta_xy_m": [dx, dy],
        "place_pose_world_before": before_pose,
        "place_pose_world_after": copy.deepcopy(place_pose),
        "base_goal_before": before_goal,
        "base_goal_after": copy.deepcopy(base_goal) if base_goal else None,
        "nav_goal_rule": "保持模板 base_goal 相对 place_pose_world 的 XY 偏移",
        "pose_policy": "仅随机 XY，其余位姿分量保持不变",
    }
    task["randomization"] = randomization


def _attach_region_metadata(
    task: dict[str, Any],
    *,
    seed: int,
    settings: RandomizationSettings,
) -> None:
    """即使未启用采样，也记录配置区域供 debug guide 使用。"""

    randomization = dict(task.get("randomization") or {})
    randomization.setdefault(
        "object_xy_randomization",
        {
            "enabled": False,
            "mode": "configured_region_only",
            "x_range_m": list(settings.pick_x_range),
            "y_range_m": list(settings.pick_y_range),
        },
    )
    randomization.setdefault(
        "place_xy_randomization",
        {
            "enabled": False,
            "mode": "configured_region_only",
            "seed": int(seed) + settings.place_seed_offset,
            "x_range_m": list(settings.place_x_range),
            "y_range_m": list(settings.place_y_range),
        },
    )
    task["randomization"] = randomization


def prepare_episode_spec(
    base_spec: EpisodeSpec,
    *,
    episode_id: int,
    seed: int,
    settings: RandomizationSettings,
) -> EpisodeSpec:
    """按 seed 构造 episode；关闭随机化时保持原任务位姿不变。"""

    task = copy.deepcopy(base_spec.raw_task)
    task["episode_id"] = int(episode_id)
    if settings.enabled:
        _randomize_pick_xy(
            task,
            seed=seed,
            settings=settings,
        )
        _randomize_place_xy(task, seed=seed, settings=settings)
    if settings.show_debug_region:
        _attach_region_metadata(task, seed=seed, settings=settings)

    if not settings.enabled and not settings.show_debug_region:
        return replace(
            base_spec,
            episode_id=int(episode_id),
            raw_task=task,
        )
    return episode_spec_from_dict(task)
