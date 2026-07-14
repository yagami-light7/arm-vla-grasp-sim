"""单进程 full-physics pipeline 的 episode 任务随机化。"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from source.data.random_task import SpawnRegion, sample_object_pose
from source.interfaces import EpisodeSpec

from .forward_sector_randomization import (
    apply_forward_sector_randomization,
    uses_forward_sector_randomization,
)
from .task_loader import episode_spec_from_dict

if TYPE_CHECKING:
    from source.pipeline.config import RandomizationSettings


_RANDOMIZATION_NAV_MAP_CACHE: dict[tuple[str, float], tuple[Any, dict[str, Any]]] = {}


def _fixed_curobo_collision_proxy_paths(task: dict[str, Any]) -> list[str]:
    """列出不会随目标采样自动更新的任务级 CuRobo collision proxy。"""

    paths: list[str] = []
    for phase in ("pick", "place"):
        raw_phase = task.get(phase) or {}
        if not isinstance(raw_phase, dict):
            continue
        raw_collision = raw_phase.get("curobo_world_collision")
        if not isinstance(raw_collision, dict) or not raw_collision.get("enabled", True):
            continue
        cuboids = raw_collision.get("cuboids_world") or []
        if not isinstance(cuboids, list):
            continue
        for index, cuboid in enumerate(cuboids):
            if isinstance(cuboid, dict):
                name = str(cuboid.get("name") or index)
                paths.append(
                    f"task.{phase}.curobo_world_collision.cuboids_world[{index}]={name}"
                )
    return paths


def _reject_unsynchronized_collision_proxy_randomization(
    task: dict[str, Any],
) -> None:
    """目标随机化与固定碰撞代理不能使用不同坐标。"""

    proxy_paths = _fixed_curobo_collision_proxy_paths(task)
    if not proxy_paths:
        return
    raise RuntimeError(
        "task_randomization_conflicts_with_fixed_curobo_collision_proxies: "
        f"{proxy_paths}; use --no-randomize-task until proxy regeneration is implemented"
    )


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _rotate_xy(xy: tuple[float, float], yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    x, y = float(xy[0]), float(xy[1])
    return (
        cos_yaw * x - sin_yaw * y,
        sin_yaw * x + cos_yaw * y,
    )


def _finite_values(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _dict_xyyaw(goal: dict[str, Any]) -> tuple[float, float, float]:
    return (float(goal["x"]), float(goal["y"]), float(goal.get("yaw", 0.0)))


def _task_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _load_randomization_nav_map(
    task: dict[str, Any],
    settings: "RandomizationSettings",
) -> tuple[Any | None, dict[str, Any]]:
    """加载轻量 occupancy map；失败时只写诊断，不阻塞 base-goal 采样。"""

    nav_map_path = _task_path(str(task.get("nav_map") or ""))
    if nav_map_path is None:
        return None, {"status": "not_available", "reason": "nav_map_missing"}
    if not nav_map_path.is_file():
        return None, {
            "status": "not_available",
            "reason": "nav_map_file_missing",
            "path": str(nav_map_path),
        }
    cache_key = (str(nav_map_path), float(settings.clearance_radius))
    if cache_key in _RANDOMIZATION_NAV_MAP_CACHE:
        grid_map, status = _RANDOMIZATION_NAV_MAP_CACHE[cache_key]
        return grid_map, {**status, "cache_hit": True}
    try:
        from source.navigation.navlib import OccupancyGridMap

        grid_map = OccupancyGridMap.from_meta_file(nav_map_path).inflate(
            float(settings.clearance_radius),
        )
    except Exception as exc:
        return None, {
            "status": "not_available",
            "reason": "nav_map_load_failed",
            "path": str(nav_map_path),
            "error": str(exc),
        }
    status = {
        "status": "available",
        "path": str(nav_map_path),
        "inflate_radius_m": float(settings.clearance_radius),
    }
    _RANDOMIZATION_NAV_MAP_CACHE[cache_key] = (grid_map, dict(status))
    return grid_map, status


def _map_point_report(grid_map: Any, xy: tuple[float, float]) -> dict[str, Any]:
    row, col = grid_map.world_to_grid(float(xy[0]), float(xy[1]))
    in_bounds = bool(grid_map.in_bounds(row, col))
    occupied = bool(grid_map.is_occupied(row, col))
    clearance = grid_map.distance_to_obstacle(row, col)
    return {
        "xy": [float(xy[0]), float(xy[1])],
        "grid": [int(row), int(col)],
        "in_bounds": in_bounds,
        "occupied": occupied,
        "clearance_m": None if clearance is None else float(clearance),
    }


def _nav_map_reject_reason(
    grid_map: Any | None,
    *,
    robot_base_xy: tuple[float, float],
    arm_base_xy: tuple[float, float],
    min_clearance_m: float,
) -> tuple[str | None, dict[str, Any]]:
    if grid_map is None:
        return None, {"status": "not_available"}
    robot_report = _map_point_report(grid_map, robot_base_xy)
    arm_report = _map_point_report(grid_map, arm_base_xy)
    report = {
        "status": "checked",
        "robot_base": robot_report,
        "arm_base": arm_report,
    }
    if not robot_report["in_bounds"] or robot_report["occupied"]:
        return "robot_base_not_free_in_nav_map", report
    if not arm_report["in_bounds"] or arm_report["occupied"]:
        return "arm_base_not_free_in_nav_map", report
    min_clearance_m = float(min_clearance_m)
    if min_clearance_m > 0.0:
        robot_clearance = robot_report.get("clearance_m")
        arm_clearance = arm_report.get("clearance_m")
        if robot_clearance is not None and float(robot_clearance) < min_clearance_m:
            return "robot_base_clearance_too_low", report
        if arm_clearance is not None and float(arm_clearance) < min_clearance_m:
            return "arm_base_clearance_too_low", report
    return None, report


def sample_handoff_base_goal(
    *,
    target_xy: tuple[float, float],
    nominal_base_goal_xyyaw: tuple[float, float, float],
    rng: random.Random,
    radius_min_m: float,
    radius_max_m: float,
    angle_noise_deg: float,
    yaw_noise_deg: float,
    arm_base_offset_xy: tuple[float, float],
    workspace_min_xy_radius_m: float,
    workspace_max_xy_radius_m: float,
    nav_map_min_clearance_m: float,
    max_attempts: int,
    occupancy_map: Any | None = None,
    nav_map_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """围绕目标采样 handoff base-goal；只生成任务定义，不推进仿真。"""

    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    nominal_x, nominal_y, nominal_yaw = (
        float(nominal_base_goal_xyyaw[0]),
        float(nominal_base_goal_xyyaw[1]),
        float(nominal_base_goal_xyyaw[2]),
    )
    offset_x, offset_y = float(arm_base_offset_xy[0]), float(arm_base_offset_xy[1])
    nominal_offset_world = _rotate_xy((offset_x, offset_y), nominal_yaw)
    nominal_arm_base_xy = (
        nominal_x + nominal_offset_world[0],
        nominal_y + nominal_offset_world[1],
    )
    theta_nominal = math.atan2(
        target_y - nominal_arm_base_xy[1],
        target_x - nominal_arm_base_xy[0],
    )
    angle_noise_rad = math.radians(float(angle_noise_deg))
    attempts: list[dict[str, Any]] = []
    nav_map_status = dict(nav_map_status or {"status": "not_available"})
    # 导航完成判据是 XY-only，yaw 不参与到达判定。这里固定模板 yaw，
    # 避免随机 yaw 让 place 规划落到机械臂实际工作区之外。
    yaw = _normalize_angle(nominal_yaw)

    for attempt in range(1, int(max_attempts) + 1):
        radius_m = rng.uniform(float(radius_min_m), float(radius_max_m))
        theta_rad = theta_nominal + rng.uniform(-angle_noise_rad, angle_noise_rad)
        arm_base_x = target_x - radius_m * math.cos(theta_rad)
        arm_base_y = target_y - radius_m * math.sin(theta_rad)
        yaw_to_target = math.atan2(target_y - arm_base_y, target_x - arm_base_x)
        offset_world = _rotate_xy((offset_x, offset_y), yaw)
        robot_base_x = arm_base_x - offset_world[0]
        robot_base_y = arm_base_y - offset_world[1]
        actual_radius = math.hypot(target_x - arm_base_x, target_y - arm_base_y)
        reject_reason = None
        nav_report: dict[str, Any]
        if not _finite_values(
            robot_base_x,
            robot_base_y,
            arm_base_x,
            arm_base_y,
            yaw,
            actual_radius,
        ):
            reject_reason = "non_finite_sample"
            nav_report = {"status": "not_checked", "reason": reject_reason}
        elif not (
            float(workspace_min_xy_radius_m)
            <= actual_radius
            <= float(workspace_max_xy_radius_m)
        ):
            reject_reason = "arm_workspace_radius_out_of_range"
            nav_report = {"status": "not_checked", "reason": reject_reason}
        else:
            reject_reason, nav_report = _nav_map_reject_reason(
                occupancy_map,
                robot_base_xy=(robot_base_x, robot_base_y),
                arm_base_xy=(arm_base_x, arm_base_y),
                min_clearance_m=float(nav_map_min_clearance_m),
            )

        attempt_report = {
            "target_xy": [target_x, target_y],
            "nominal_base_goal_xyyaw": [
                nominal_x,
                nominal_y,
                nominal_yaw,
            ],
            "sampled_base_goal_xyyaw": [
                robot_base_x,
                robot_base_y,
                yaw,
            ],
            "sampled_arm_base_xy": [arm_base_x, arm_base_y],
            "radius_m": actual_radius,
            "theta_rad": theta_rad,
            "theta_nominal_rad": theta_nominal,
            "yaw_to_target_rad": yaw_to_target,
            "yaw_noise_rad": 0.0,
            "yaw_policy": "preserve_nominal_base_goal_yaw",
            "attempt": attempt,
            "valid": reject_reason is None,
            "reject_reason": reject_reason,
            "fallback_used": False,
            "nav_map_check": nav_map_status,
            "nav_map_sample_check": nav_report,
        }
        attempts.append(attempt_report)
        if reject_reason is None:
            return attempt_report

    last_attempt = attempts[-1] if attempts else None
    raise RuntimeError(
        "failed_to_sample_valid_base_goal: "
        f"attempts={max_attempts} "
        f"last_reason={(last_attempt or {}).get('reject_reason')}"
    )


def _fixed_base_goal_report(
    *,
    target_xy: tuple[float, float],
    nominal_base_goal: dict[str, Any],
    arm_base_offset_xy: tuple[float, float],
    reason: str,
    nav_map_status: dict[str, Any],
    fallback_used: bool = True,
    report_mode: str = "fallback_fixed_goal",
) -> dict[str, Any]:
    nominal_x, nominal_y, nominal_yaw = _dict_xyyaw(nominal_base_goal)
    offset_world = _rotate_xy(arm_base_offset_xy, nominal_yaw)
    arm_base_x = nominal_x + offset_world[0]
    arm_base_y = nominal_y + offset_world[1]
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    yaw_to_target = math.atan2(target_y - arm_base_y, target_x - arm_base_x)
    return {
        "target_xy": [target_x, target_y],
        "nominal_base_goal_xyyaw": [nominal_x, nominal_y, nominal_yaw],
        "sampled_base_goal_xyyaw": [nominal_x, nominal_y, nominal_yaw],
        "sampled_arm_base_xy": [arm_base_x, arm_base_y],
        "radius_m": math.hypot(target_x - arm_base_x, target_y - arm_base_y),
        "robot_base_radius_m": math.hypot(target_x - nominal_x, target_y - nominal_y),
        "theta_rad": math.atan2(target_y - arm_base_y, target_x - arm_base_x),
        "yaw_to_target_rad": yaw_to_target,
        "yaw_noise_rad": 0.0,
        "yaw_policy": "preserve_nominal_base_goal_yaw",
        "attempt": 0,
        "valid": True,
        "fallback_used": bool(fallback_used),
        "fallback_reason": reason if fallback_used else None,
        "mode": report_mode,
        "nav_map_check": dict(nav_map_status),
        "nav_map_sample_check": {"status": "not_checked", "reason": report_mode},
    }


def _sample_rectangular_place_base_goal(
    *,
    target_xy: tuple[float, float],
    nominal_base_goal_xyyaw: tuple[float, float, float],
    rng: random.Random,
    offset_x_range_m: tuple[float, float],
    offset_y_range_m: tuple[float, float],
    arm_base_offset_xy: tuple[float, float],
    workspace_min_xy_radius_m: float,
    workspace_max_xy_radius_m: float,
    robot_base_max_xy_radius_m: float,
    nav_map_min_clearance_m: float,
    max_attempts: int,
    occupancy_map: Any | None = None,
    nav_map_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """为 place 采样矩形 base-goal offset，避免极坐标把狗推到桌群深处。"""

    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    nominal_x, nominal_y, nominal_yaw = (
        float(nominal_base_goal_xyyaw[0]),
        float(nominal_base_goal_xyyaw[1]),
        float(nominal_base_goal_xyyaw[2]),
    )
    yaw = _normalize_angle(nominal_yaw)
    offset_x_min, offset_x_max = float(offset_x_range_m[0]), float(offset_x_range_m[1])
    offset_y_min, offset_y_max = float(offset_y_range_m[0]), float(offset_y_range_m[1])
    arm_offset_x, arm_offset_y = float(arm_base_offset_xy[0]), float(arm_base_offset_xy[1])
    workspace_min_m = float(workspace_min_xy_radius_m)
    workspace_max_m = float(workspace_max_xy_radius_m)
    robot_base_max_m = float(robot_base_max_xy_radius_m)
    nav_map_status = dict(nav_map_status or {"status": "not_available"})
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, int(max_attempts) + 1):
        # 用户指定的 place 策略：base_goal.x 在目标右侧采样，base_goal.y 在目标前侧采样。
        sampled_offset_x = rng.uniform(offset_x_min, offset_x_max)
        sampled_offset_y = rng.uniform(offset_y_min, offset_y_max)
        robot_base_x = target_x + sampled_offset_x
        robot_base_y = target_y + sampled_offset_y
        arm_offset_world = _rotate_xy((arm_offset_x, arm_offset_y), yaw)
        arm_base_x = robot_base_x + arm_offset_world[0]
        arm_base_y = robot_base_y + arm_offset_world[1]
        radius_m = math.hypot(target_x - arm_base_x, target_y - arm_base_y)
        robot_base_radius_m = math.hypot(
            target_x - robot_base_x,
            target_y - robot_base_y,
        )
        theta_rad = math.atan2(target_y - arm_base_y, target_x - arm_base_x)
        yaw_to_target = math.atan2(target_y - arm_base_y, target_x - arm_base_x)

        reject_reason = None
        nav_report: dict[str, Any]
        if not _finite_values(
            robot_base_x,
            robot_base_y,
            arm_base_x,
            arm_base_y,
            yaw,
            radius_m,
            robot_base_radius_m,
        ):
            reject_reason = "non_finite_sample"
            nav_report = {"status": "not_checked", "reason": reject_reason}
        elif not workspace_min_m <= radius_m <= workspace_max_m:
            reject_reason = "arm_workspace_radius_out_of_range"
            nav_report = {"status": "not_checked", "reason": reject_reason}
        elif robot_base_radius_m > robot_base_max_m:
            reject_reason = "robot_base_radius_out_of_range"
            nav_report = {"status": "not_checked", "reason": reject_reason}
        else:
            reject_reason, nav_report = _nav_map_reject_reason(
                occupancy_map,
                robot_base_xy=(robot_base_x, robot_base_y),
                arm_base_xy=(arm_base_x, arm_base_y),
                min_clearance_m=float(nav_map_min_clearance_m),
            )

        attempt_report = {
            "target_xy": [target_x, target_y],
            "nominal_base_goal_xyyaw": [nominal_x, nominal_y, nominal_yaw],
            "sampled_base_goal_xyyaw": [robot_base_x, robot_base_y, yaw],
            "sampled_arm_base_xy": [arm_base_x, arm_base_y],
            "offset_xy_m": [sampled_offset_x, sampled_offset_y],
            "offset_x_range_m": [offset_x_min, offset_x_max],
            "offset_y_range_m": [offset_y_min, offset_y_max],
            "workspace_radius_range_m": [workspace_min_m, workspace_max_m],
            "robot_base_workspace_radius_range_m": [0.0, robot_base_max_m],
            "radius_m": radius_m,
            "robot_base_radius_m": robot_base_radius_m,
            "theta_rad": theta_rad,
            "yaw_to_target_rad": yaw_to_target,
            "yaw_noise_rad": 0.0,
            "yaw_policy": "preserve_nominal_base_goal_yaw",
            "attempt": attempt,
            "valid": reject_reason is None,
            "reject_reason": reject_reason,
            "fallback_used": False,
            "mode": "place_rectangular_offset_xy",
            "nav_map_check": nav_map_status,
            "nav_map_sample_check": nav_report,
        }
        attempts.append(attempt_report)
        if reject_reason is None:
            return attempt_report

    raise RuntimeError(
        "failed_to_sample_valid_place_base_goal: "
        f"attempts={max_attempts} last_reason="
        f"{(attempts[-1] if attempts else {}).get('reject_reason')}"
    )


def _apply_sampled_goal(goal: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    updated = dict(goal)
    x, y, yaw = sample["sampled_base_goal_xyyaw"]
    updated.update({"x": float(x), "y": float(y), "yaw": float(yaw)})
    return updated


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


def _base_goal_sample_for_stage(
    *,
    stage_name: str,
    target_xy: tuple[float, float],
    nominal_base_goal: dict[str, Any],
    rng: random.Random,
    settings: RandomizationSettings,
    occupancy_map: Any | None,
    nav_map_status: dict[str, Any],
) -> dict[str, Any]:
    config = settings.base_goal
    if stage_name == "pick":
        radius_min_m = config.pick_radius_min_m
        radius_max_m = config.pick_radius_max_m
        angle_noise_deg = config.pick_angle_noise_deg
        yaw_noise_deg = config.pick_yaw_noise_deg
    elif stage_name == "place":
        arm_base_offset_xy = (
            float(config.arm_base_offset_x_m),
            float(config.arm_base_offset_y_m),
        )
        try:
            return _sample_rectangular_place_base_goal(
                target_xy=target_xy,
                nominal_base_goal_xyyaw=_dict_xyyaw(nominal_base_goal),
                rng=rng,
                offset_x_range_m=config.place_offset_x_range_m,
                offset_y_range_m=config.place_offset_y_range_m,
                arm_base_offset_xy=arm_base_offset_xy,
                workspace_min_xy_radius_m=config.arm_workspace_min_xy_radius_m,
                workspace_max_xy_radius_m=config.place_workspace_max_xy_radius_m,
                robot_base_max_xy_radius_m=config.place_robot_base_max_xy_radius_m,
                nav_map_min_clearance_m=config.nav_map_min_clearance_m,
                max_attempts=config.max_goal_sample_attempts,
                occupancy_map=occupancy_map,
                nav_map_status=nav_map_status,
            )
        except RuntimeError as exc:
            if not config.fallback_to_fixed_offset:
                raise
            return _fixed_base_goal_report(
                target_xy=target_xy,
                nominal_base_goal=nominal_base_goal,
                arm_base_offset_xy=arm_base_offset_xy,
                reason=str(exc),
                nav_map_status=nav_map_status,
            )
    else:
        raise ValueError(f"unsupported base-goal randomization stage: {stage_name}")

    arm_base_offset_xy = (
        float(config.arm_base_offset_x_m),
        float(config.arm_base_offset_y_m),
    )
    try:
        return sample_handoff_base_goal(
            target_xy=target_xy,
            nominal_base_goal_xyyaw=_dict_xyyaw(nominal_base_goal),
            rng=rng,
            radius_min_m=radius_min_m,
            radius_max_m=radius_max_m,
            angle_noise_deg=angle_noise_deg,
            yaw_noise_deg=yaw_noise_deg,
            arm_base_offset_xy=arm_base_offset_xy,
            workspace_min_xy_radius_m=config.arm_workspace_min_xy_radius_m,
            workspace_max_xy_radius_m=config.arm_workspace_max_xy_radius_m,
            nav_map_min_clearance_m=config.nav_map_min_clearance_m,
            max_attempts=config.max_goal_sample_attempts,
            occupancy_map=occupancy_map,
            nav_map_status=nav_map_status,
        )
    except RuntimeError as exc:
        if not config.fallback_to_fixed_offset:
            raise
        return _fixed_base_goal_report(
            target_xy=target_xy,
            nominal_base_goal=nominal_base_goal,
            arm_base_offset_xy=arm_base_offset_xy,
            reason=str(exc),
            nav_map_status=nav_map_status,
        )


def _randomize_base_goals(
    task: dict[str, Any],
    *,
    seed: int,
    settings: RandomizationSettings,
) -> None:
    """按极坐标随机 pick/place handoff base-goal，不修改目标物体或执行器。"""

    config = settings.base_goal
    if not config.enabled:
        return

    pick = dict(task.get("pick") or {})
    place = dict(task.get("place") or {})
    object_pose = dict(pick.get("object_pose_world") or {})
    pick_base_goal = dict(pick.get("base_goal") or {})
    if not {"x", "y"}.issubset(object_pose):
        raise ValueError("pick.object_pose_world must contain x and y")
    if not {"x", "y"}.issubset(pick_base_goal):
        raise ValueError("pick.base_goal must contain x and y")

    occupancy_map, nav_map_status = _load_randomization_nav_map(task, settings)
    rng = random.Random(int(seed) + int(config.seed_offset))
    pick_sample = _base_goal_sample_for_stage(
        stage_name="pick",
        target_xy=(float(object_pose["x"]), float(object_pose["y"])),
        nominal_base_goal=pick_base_goal,
        rng=rng,
        settings=settings,
        occupancy_map=occupancy_map,
        nav_map_status=nav_map_status,
    )
    pick["base_goal"] = _apply_sampled_goal(pick_base_goal, pick_sample)
    task["pick"] = pick

    place_sample: dict[str, Any] | None = None
    if (
        place.get("enabled")
        and isinstance(place.get("place_pose_world"), dict)
        and isinstance(place.get("base_goal"), dict)
    ):
        place_pose = dict(place.get("place_pose_world") or {})
        place_base_goal = dict(place.get("base_goal") or {})
        if {"x", "y"}.issubset(place_pose) and {"x", "y"}.issubset(place_base_goal):
            place_sample = _base_goal_sample_for_stage(
                stage_name="place",
                target_xy=(float(place_pose["x"]), float(place_pose["y"])),
                nominal_base_goal=place_base_goal,
                rng=rng,
                settings=settings,
                occupancy_map=occupancy_map,
                nav_map_status=nav_map_status,
            )
            place["base_goal"] = _apply_sampled_goal(place_base_goal, place_sample)
            task["place"] = place

    randomization = dict(task.get("randomization") or {})
    randomization["base_goal_randomization"] = {
        "enabled": True,
        "seed": int(seed) + int(config.seed_offset),
        "config": {
            "pick_radius_range_m": [
                float(config.pick_radius_min_m),
                float(config.pick_radius_max_m),
            ],
            "pick_angle_noise_deg": float(config.pick_angle_noise_deg),
            "pick_yaw_noise_deg": float(config.pick_yaw_noise_deg),
            "place_offset_x_range_m": [
                float(config.place_offset_x_range_m[0]),
                float(config.place_offset_x_range_m[1]),
            ],
            "place_offset_y_range_m": [
                float(config.place_offset_y_range_m[0]),
                float(config.place_offset_y_range_m[1]),
            ],
            "place_radius_range_m": [
                float(config.place_radius_min_m),
                float(config.place_radius_max_m),
            ],
            "place_angle_noise_deg": float(config.place_angle_noise_deg),
            "place_yaw_noise_deg": float(config.place_yaw_noise_deg),
            "arm_base_offset_xy_m": [
                float(config.arm_base_offset_x_m),
                float(config.arm_base_offset_y_m),
            ],
            "workspace_radius_range_m": [
                float(config.arm_workspace_min_xy_radius_m),
                float(config.arm_workspace_max_xy_radius_m),
            ],
            "place_workspace_radius_range_m": [
                float(config.arm_workspace_min_xy_radius_m),
                float(config.place_workspace_max_xy_radius_m),
            ],
            "place_robot_base_max_xy_radius_m": float(
                config.place_robot_base_max_xy_radius_m,
            ),
            "nav_map_min_clearance_m": float(config.nav_map_min_clearance_m),
            "max_goal_sample_attempts": int(config.max_goal_sample_attempts),
            "max_place_target_sample_attempts": int(
                config.max_place_target_sample_attempts
            ),
            "fallback_to_fixed_offset": bool(config.fallback_to_fixed_offset),
            "validate_with_curobo": bool(config.validate_with_curobo),
        },
        "pick": pick_sample,
        "place": place_sample
        if place_sample is not None
        else {
            "valid": False,
            "fallback_used": False,
            "reason": "place_disabled_or_missing_goal",
        },
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


def _randomize_place_xy_and_base_goals(
    task: dict[str, Any],
    *,
    seed: int,
    settings: RandomizationSettings,
) -> None:
    """联合采样 place 目标和 base_goal，过滤确定无法导航或规划的组合。"""

    config = settings.base_goal
    max_attempts = int(config.max_place_target_sample_attempts)
    rejected: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        candidate = copy.deepcopy(task)
        # 每次只改变 place XY 的随机流；pick XY 和 pick base_goal 保持同一 seed 可复现。
        place_seed = int(seed) + (attempt - 1) * 1009
        _randomize_place_xy(candidate, seed=place_seed, settings=settings)
        try:
            _randomize_base_goals(candidate, seed=seed, settings=settings)
        except RuntimeError as exc:
            error_text = str(exc)
            if "failed_to_sample_valid_place_base_goal" not in error_text:
                raise
            place_randomization = (
                candidate.get("randomization", {}).get("place_xy_randomization", {})
            )
            rejected.append(
                {
                    "attempt": attempt,
                    "seed": int(place_seed) + int(settings.place_seed_offset),
                    "sampled_xy": place_randomization.get("sampled_xy"),
                    "reason": error_text,
                }
            )
            continue
        place_randomization = candidate.setdefault("randomization", {}).setdefault(
            "place_xy_randomization",
            {},
        )
        place_randomization["target_resample_attempt"] = attempt
        place_randomization["rejected_target_samples"] = rejected
        place_randomization["target_base_goal_joint_filter"] = (
            "place XY accepted only after nav-map and arm-workspace base_goal checks"
        )
        task.clear()
        task.update(candidate)
        return
    raise RuntimeError(
        "failed_to_sample_valid_place_target_and_base_goal: "
        f"attempts={max_attempts} last_reason="
        f"{(rejected[-1] if rejected else {}).get('reason')}"
    )


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
    forward_sector_mode = uses_forward_sector_randomization(task)
    if settings.enabled and forward_sector_mode:
        apply_forward_sector_randomization(
            task,
            seed=seed,
            randomize_base_goal=settings.base_goal.enabled,
            collision_ply_path=settings.collision_ply_path,
        )
    elif settings.enabled:
        _reject_unsynchronized_collision_proxy_randomization(task)
        _randomize_pick_xy(
            task,
            seed=seed,
            settings=settings,
        )
        if settings.base_goal.enabled:
            _randomize_place_xy_and_base_goals(task, seed=seed, settings=settings)
        else:
            _randomize_place_xy(task, seed=seed, settings=settings)
    if not forward_sector_mode and not (
        settings.enabled and settings.base_goal.enabled
    ):
        _randomize_base_goals(task, seed=seed, settings=settings)
    if settings.show_debug_region and not forward_sector_mode:
        _attach_region_metadata(task, seed=seed, settings=settings)

    if (
        not settings.enabled
        and not settings.show_debug_region
        and not settings.base_goal.enabled
    ):
        return replace(
            base_spec,
            episode_id=int(episode_id),
            raw_task=task,
        )
    return episode_spec_from_dict(task)
