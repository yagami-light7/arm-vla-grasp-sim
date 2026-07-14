"""良渚可乐到地垫任务的前向扇形联合随机化。"""

from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Any

from source.scene.placement_support import (
    inspect_placement_support,
    load_binary_triangle_ply,
)


FORWARD_SECTOR_MODE = "robot_forward_sector_v1"


def uses_forward_sector_randomization(raw_task: dict[str, Any]) -> bool:
    """判断任务是否声明了良渚前向扇形联合采样模式。"""

    randomization = raw_task.get("randomization")
    return bool(
        isinstance(randomization, dict)
        and randomization.get("mode") == FORWARD_SECTOR_MODE
    )


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数值")
    return result


def _vector(
    value: Any,
    *,
    field_name: str,
    length: int,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} 必须包含 {length} 个数值")
    return tuple(
        _finite_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _range(value: Any, *, field_name: str) -> tuple[float, float]:
    lower, upper = _vector(value, field_name=field_name, length=2)
    if lower > upper:
        raise ValueError(f"{field_name} 的下界不能大于上界")
    return lower, upper


def _positive_range(value: Any, *, field_name: str) -> tuple[float, float]:
    lower, upper = _range(value, field_name=field_name)
    if lower <= 0.0:
        raise ValueError(f"{field_name} 的下界必须大于零")
    return lower, upper


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _rotate_xy(xy: tuple[float, float], yaw_rad: float) -> tuple[float, float]:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    x, y = float(xy[0]), float(xy[1])
    return cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y


def _sample_area_uniform_radius(
    rng: random.Random,
    radius_range_m: tuple[float, float],
) -> float:
    """按扇形面积均匀采样半径，避免样本过度聚集在机器人附近。"""

    lower, upper = radius_range_m
    return math.sqrt(rng.uniform(lower * lower, upper * upper))


def _sample_polar_point(
    *,
    rng: random.Random,
    origin_xy: tuple[float, float],
    robot_yaw_rad: float,
    radius_range_m: tuple[float, float],
    sector_half_angle_rad: float,
) -> dict[str, Any]:
    radius_m = _sample_area_uniform_radius(rng, radius_range_m)
    relative_angle_rad = rng.uniform(-sector_half_angle_rad, sector_half_angle_rad)
    bearing_world_rad = _normalize_angle(robot_yaw_rad + relative_angle_rad)
    return {
        "x": float(origin_xy[0]) + radius_m * math.cos(bearing_world_rad),
        "y": float(origin_xy[1]) + radius_m * math.sin(bearing_world_rad),
        "radius_m": radius_m,
        "relative_angle_rad": relative_angle_rad,
        "bearing_world_rad": bearing_world_rad,
    }


def _point_in_expanded_oriented_mat(
    *,
    point_xy: tuple[float, float],
    mat_center_xy: tuple[float, float],
    mat_yaw_rad: float,
    mat_dims_xy_m: tuple[float, float],
    expansion_m: float,
) -> tuple[bool, tuple[float, float]]:
    delta_world = (
        float(point_xy[0]) - float(mat_center_xy[0]),
        float(point_xy[1]) - float(mat_center_xy[1]),
    )
    local_x, local_y = _rotate_xy(delta_world, -float(mat_yaw_rad))
    half_x = 0.5 * float(mat_dims_xy_m[0]) + float(expansion_m)
    half_y = 0.5 * float(mat_dims_xy_m[1]) + float(expansion_m)
    return abs(local_x) <= half_x and abs(local_y) <= half_y, (local_x, local_y)


def _point_to_oriented_mat_distance(
    *,
    point_xy: tuple[float, float],
    mat_center_xy: tuple[float, float],
    mat_yaw_rad: float,
    mat_dims_xy_m: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    """计算世界点到旋转垫子矩形足迹的最短平面距离。"""

    delta_world = (
        float(point_xy[0]) - float(mat_center_xy[0]),
        float(point_xy[1]) - float(mat_center_xy[1]),
    )
    local_x, local_y = _rotate_xy(delta_world, -float(mat_yaw_rad))
    outside_x = max(abs(local_x) - 0.5 * float(mat_dims_xy_m[0]), 0.0)
    outside_y = max(abs(local_y) - 0.5 * float(mat_dims_xy_m[1]), 0.0)
    return math.hypot(outside_x, outside_y), (local_x, local_y)


def _goal_facing_target(
    *,
    rng: random.Random,
    robot_xy: tuple[float, float],
    target_xy: tuple[float, float],
    standoff_range_m: tuple[float, float],
    angle_noise_rad: float,
    randomize: bool,
) -> dict[str, Any]:
    """生成目标位于机器人正前方的底盘终态。"""

    target_bearing = math.atan2(
        float(target_xy[1]) - float(robot_xy[1]),
        float(target_xy[0]) - float(robot_xy[0]),
    )
    if randomize:
        standoff_m = rng.uniform(*standoff_range_m)
        approach_bearing = target_bearing + rng.uniform(
            -angle_noise_rad,
            angle_noise_rad,
        )
    else:
        standoff_m = 0.5 * (standoff_range_m[0] + standoff_range_m[1])
        approach_bearing = target_bearing
    goal_x = float(target_xy[0]) - standoff_m * math.cos(approach_bearing)
    goal_y = float(target_xy[1]) - standoff_m * math.sin(approach_bearing)
    target_bearing_from_goal = math.atan2(
        float(target_xy[1]) - goal_y,
        float(target_xy[0]) - goal_x,
    )
    yaw = _normalize_angle(target_bearing_from_goal)
    target_bearing_base = _normalize_angle(target_bearing_from_goal - yaw)
    return {
        "x": goal_x,
        "y": goal_y,
        "yaw": yaw,
        "standoff_m": standoff_m,
        "approach_bearing_world_rad": _normalize_angle(approach_bearing),
        "target_bearing_from_goal_world_rad": _normalize_angle(
            target_bearing_from_goal
        ),
        "target_bearing_base_rad": target_bearing_base,
        "target_region_in_base": "front",
        "final_alignment_mode": "face_target",
    }


def _resolve_collision_ply(config: dict[str, Any]) -> tuple[Path, str | None]:
    path_value = config.get("collision_ply_path")
    env_name = str(config.get("collision_ply_env") or "").strip() or None
    if path_value is None and env_name is not None:
        path_value = os.environ.get(env_name)
    if not path_value:
        suffix = f"；请设置环境变量 {env_name}" if env_name else ""
        # 配置缺失不会因重新采样而恢复，必须立即停止，避免重复相同错误。
        raise ValueError(f"前向扇形随机化缺少 collision PLY{suffix}")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"前向扇形随机化 collision PLY 不存在: {path}")
    return path, env_name


def _surface_probe(
    collision_ply: Path,
    *,
    xy: tuple[float, float],
    query_ceiling_z: float,
) -> dict[str, Any]:
    result = inspect_placement_support(
        collision_ply,
        (float(xy[0]), float(xy[1]), float(query_ceiling_z)),
        minimum_clearance_m=1.0e-4,
        maximum_clearance_m=max(1.0, abs(float(query_ceiling_z)) + 5.0),
    )
    if result.support is None:
        raise RuntimeError(f"collision PLY 在 XY={xy} 下方没有支撑三角面")
    return {
        "xy": [float(xy[0]), float(xy[1])],
        "z": float(result.support.z),
        "face_index": int(result.support.face_index),
    }


def _support_normal(
    collision_ply: Path,
    *,
    face_index: int,
) -> tuple[float, float, float]:
    vertices, face_indices = load_binary_triangle_ply(collision_ply)
    indices = face_indices[int(face_index)]
    first = tuple(float(value) for value in vertices[int(indices[0])])
    second = tuple(float(value) for value in vertices[int(indices[1])])
    third = tuple(float(value) for value in vertices[int(indices[2])])
    first_edge = tuple(second[index] - first[index] for index in range(3))
    second_edge = tuple(third[index] - first[index] for index in range(3))
    normal = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    norm = math.sqrt(sum(value * value for value in normal))
    if norm <= 1.0e-12:
        raise RuntimeError(f"collision PLY face {face_index} 是退化三角面")
    normalized = tuple(value / norm for value in normal)
    if normalized[2] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized


def _quat_align_z_to_normal(
    normal_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """生成把局部 +Z 对齐到支撑法向的 wxyz 四元数。"""

    nx, ny, nz = normal_xyz
    if nz <= -1.0 + 1.0e-9:
        return 0.0, 1.0, 0.0, 0.0
    scale = math.sqrt(max(1.0e-12, 2.0 * (1.0 + nz)))
    quaternion = (0.5 * scale, -ny / scale, nx / scale, 0.0)
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def _yaw_quaternion(yaw_rad: float) -> list[float]:
    return [math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)]


def _mat_probe_points(
    *,
    center_xy: tuple[float, float],
    yaw_rad: float,
    dims_xy_m: tuple[float, float],
    inset_m: float,
) -> tuple[tuple[float, float], ...]:
    half_x = max(0.0, 0.5 * dims_xy_m[0] - inset_m)
    half_y = max(0.0, 0.5 * dims_xy_m[1] - inset_m)
    output = [center_xy]
    for local_x, local_y in (
        (-half_x, -half_y),
        (-half_x, half_y),
        (half_x, -half_y),
        (half_x, half_y),
    ):
        dx, dy = _rotate_xy((local_x, local_y), yaw_rad)
        output.append((center_xy[0] + dx, center_xy[1] + dy))
    return tuple(output)


def _updated_goal(
    template: Any,
    *,
    sampled: dict[str, Any],
    default_z: float,
) -> dict[str, Any]:
    goal = dict(template) if isinstance(template, dict) else {}
    goal.update(
        {
            "x": float(sampled["x"]),
            "y": float(sampled["y"]),
            "yaw": float(sampled["yaw"]),
            "target_region_in_base": str(sampled["target_region_in_base"]),
            "final_alignment_mode": str(sampled["final_alignment_mode"]),
        }
    )
    goal.pop("target_side_in_base", None)
    goal.pop("lateral_yaw_offset_rad", None)
    goal.setdefault("z", float(default_z))
    return goal


def _layout_config(raw_task: dict[str, Any]) -> dict[str, Any]:
    randomization = raw_task.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("task.randomization 必须是对象")
    config = randomization.get("forward_sector")
    if not isinstance(config, dict):
        raise ValueError("前向扇形随机化要求 task.randomization.forward_sector")
    return config


def _sample_layout_geometry(
    *,
    rng: random.Random,
    config: dict[str, Any],
    randomize_base_goal: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    robot_xyz = _vector(
        config.get("robot_translate_xyz"),
        field_name="forward_sector.robot_translate_xyz",
        length=3,
    )
    robot_yaw_deg = _range(
        config.get("robot_yaw_range_deg"),
        field_name="forward_sector.robot_yaw_range_deg",
    )
    sector_half_angle_deg = _finite_float(
        config.get("sector_half_angle_deg"),
        field_name="forward_sector.sector_half_angle_deg",
    )
    if not 0.0 < sector_half_angle_deg < 90.0:
        raise ValueError("forward_sector.sector_half_angle_deg 必须位于 (0, 90)")
    cola_radius_range = _positive_range(
        config.get("cola_radius_range_m"),
        field_name="forward_sector.cola_radius_range_m",
    )
    mat_radius_range = _positive_range(
        config.get("mat_radius_range_m"),
        field_name="forward_sector.mat_radius_range_m",
    )
    cola_yaw_deg = _range(
        config.get("cola_yaw_range_deg", [-180.0, 180.0]),
        field_name="forward_sector.cola_yaw_range_deg",
    )
    mat_yaw_deg = _range(
        config.get("mat_yaw_range_deg", [-180.0, 180.0]),
        field_name="forward_sector.mat_yaw_range_deg",
    )
    place_yaw_deg = _range(
        config.get("place_object_yaw_range_deg", [-180.0, 180.0]),
        field_name="forward_sector.place_object_yaw_range_deg",
    )
    mat_dims = _vector(
        config.get("mat_support_dims_xyz"),
        field_name="forward_sector.mat_support_dims_xyz",
        length=3,
    )
    if any(value <= 0.0 for value in mat_dims):
        raise ValueError("forward_sector.mat_support_dims_xyz 必须全部大于零")
    cola_footprint_radius = _finite_float(
        config.get("cola_footprint_radius_m"),
        field_name="forward_sector.cola_footprint_radius_m",
    )
    separation_margin = _finite_float(
        config.get("cola_mat_separation_margin_m"),
        field_name="forward_sector.cola_mat_separation_margin_m",
    )
    min_center_distance = _finite_float(
        config.get("min_cola_mat_center_distance_m"),
        field_name="forward_sector.min_cola_mat_center_distance_m",
    )
    pick_standoff = _positive_range(
        config.get("pick_base_standoff_range_m"),
        field_name="forward_sector.pick_base_standoff_range_m",
    )
    place_standoff = _positive_range(
        config.get("place_base_standoff_range_m"),
        field_name="forward_sector.place_base_standoff_range_m",
    )
    base_angle_noise_rad = math.radians(
        _finite_float(
            config.get("base_approach_angle_noise_deg", 0.0),
            field_name="forward_sector.base_approach_angle_noise_deg",
        )
    )
    base_other_clearance = _finite_float(
        config.get("base_other_target_clearance_m", 0.30),
        field_name="forward_sector.base_other_target_clearance_m",
    )
    max_attempts = int(config.get("max_attempts", 200))
    if max_attempts < 1:
        raise ValueError("forward_sector.max_attempts 必须至少为 1")
    if min(cola_footprint_radius, separation_margin, min_center_distance) < 0.0:
        raise ValueError("前向扇形随机化的间距参数不能为负数")

    rejected: list[dict[str, Any]] = []
    robot_xy = (robot_xyz[0], robot_xyz[1])
    sector_half_angle_rad = math.radians(sector_half_angle_deg)
    for attempt in range(1, max_attempts + 1):
        robot_yaw_rad = math.radians(rng.uniform(*robot_yaw_deg))
        cola = _sample_polar_point(
            rng=rng,
            origin_xy=robot_xy,
            robot_yaw_rad=robot_yaw_rad,
            radius_range_m=cola_radius_range,
            sector_half_angle_rad=sector_half_angle_rad,
        )
        mat = _sample_polar_point(
            rng=rng,
            origin_xy=robot_xy,
            robot_yaw_rad=robot_yaw_rad,
            radius_range_m=mat_radius_range,
            sector_half_angle_rad=sector_half_angle_rad,
        )
        cola_yaw_rad = math.radians(rng.uniform(*cola_yaw_deg))
        mat_yaw_rad = math.radians(rng.uniform(*mat_yaw_deg))
        place_yaw_rad = math.radians(rng.uniform(*place_yaw_deg))
        cola_xy = (cola["x"], cola["y"])
        mat_xy = (mat["x"], mat["y"])
        center_distance = math.hypot(cola_xy[0] - mat_xy[0], cola_xy[1] - mat_xy[1])
        cola_to_mat_footprint, cola_in_mat_xy = _point_to_oriented_mat_distance(
            point_xy=cola_xy,
            mat_center_xy=mat_xy,
            mat_yaw_rad=mat_yaw_rad,
            mat_dims_xy_m=(mat_dims[0], mat_dims[1]),
        )
        cola_mat_clearance = cola_to_mat_footprint - cola_footprint_radius
        reject_reason = None
        if center_distance < min_center_distance:
            reject_reason = "cola_mat_center_distance_too_small"
        elif cola_mat_clearance < separation_margin:
            reject_reason = "cola_mat_footprint_clearance_too_small"

        pick_goal = _goal_facing_target(
            rng=rng,
            robot_xy=robot_xy,
            target_xy=cola_xy,
            standoff_range_m=pick_standoff,
            angle_noise_rad=base_angle_noise_rad,
            randomize=randomize_base_goal,
        )
        place_goal = _goal_facing_target(
            rng=rng,
            robot_xy=(float(pick_goal["x"]), float(pick_goal["y"])),
            target_xy=mat_xy,
            standoff_range_m=place_standoff,
            angle_noise_rad=base_angle_noise_rad,
            randomize=randomize_base_goal,
        )
        place_goal["approach_origin"] = "pick_base_goal"
        pick_goal_overlaps_mat, _ = _point_in_expanded_oriented_mat(
            point_xy=(pick_goal["x"], pick_goal["y"]),
            mat_center_xy=mat_xy,
            mat_yaw_rad=mat_yaw_rad,
            mat_dims_xy_m=(mat_dims[0], mat_dims[1]),
            expansion_m=base_other_clearance,
        )
        place_goal_to_cola = math.hypot(
            place_goal["x"] - cola_xy[0],
            place_goal["y"] - cola_xy[1],
        )
        if reject_reason is None and pick_goal_overlaps_mat:
            reject_reason = "pick_base_goal_too_close_to_mat"
        elif reject_reason is None and place_goal_to_cola < base_other_clearance:
            reject_reason = "place_base_goal_too_close_to_cola"

        if reject_reason is not None:
            rejected.append(
                {
                    "attempt": attempt,
                    "reason": reject_reason,
                    "cola_mat_center_distance_m": center_distance,
                }
            )
            continue
        return (
            {
                "attempt": attempt,
                "robot_xyz": list(robot_xyz),
                "robot_yaw_rad": _normalize_angle(robot_yaw_rad),
                "cola": {
                    **cola,
                    "yaw_rad": _normalize_angle(cola_yaw_rad),
                },
                "mat": {
                    **mat,
                    "yaw_rad": _normalize_angle(mat_yaw_rad),
                },
                "place_object_yaw_rad": _normalize_angle(place_yaw_rad),
                "cola_mat_center_distance_m": center_distance,
                "cola_center_in_mat_frame_xy_m": list(cola_in_mat_xy),
                "cola_to_mat_footprint_distance_m": cola_to_mat_footprint,
                "cola_mat_footprint_clearance_m": cola_mat_clearance,
                "required_cola_mat_clearance_m": separation_margin,
                "cola_overlaps_mat_footprint": False,
                "pick_base_goal": pick_goal,
                "place_base_goal": place_goal,
                "base_goal_randomized": bool(randomize_base_goal),
            },
            rejected,
        )
    raise RuntimeError(
        "failed_to_sample_forward_sector_layout: "
        f"attempts={max_attempts} last_reason="
        f"{(rejected[-1] if rejected else {}).get('reason')}"
    )


def _apply_support_geometry(
    *,
    layout: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    collision_ply, collision_ply_env = _resolve_collision_ply(config)
    query_ceiling_z = _finite_float(
        config.get("ground_query_ceiling_z", 2.0),
        field_name="forward_sector.ground_query_ceiling_z",
    )
    mat_dims = _vector(
        config.get("mat_support_dims_xyz"),
        field_name="forward_sector.mat_support_dims_xyz",
        length=3,
    )
    mat_probe_inset = _finite_float(
        config.get("mat_ground_probe_inset_m", 0.01),
        field_name="forward_sector.mat_ground_probe_inset_m",
    )
    max_mat_variation = _finite_float(
        config.get("max_mat_ground_height_variation_m", 0.03),
        field_name="forward_sector.max_mat_ground_height_variation_m",
    )
    cola_xy = (float(layout["cola"]["x"]), float(layout["cola"]["y"]))
    mat_xy = (float(layout["mat"]["x"]), float(layout["mat"]["y"]))
    mat_yaw = float(layout["mat"]["yaw_rad"])
    cola_probe = _surface_probe(
        collision_ply,
        xy=cola_xy,
        query_ceiling_z=query_ceiling_z,
    )
    mat_probes = [
        _surface_probe(
            collision_ply,
            xy=point,
            query_ceiling_z=query_ceiling_z,
        )
        for point in _mat_probe_points(
            center_xy=mat_xy,
            yaw_rad=mat_yaw,
            dims_xy_m=(mat_dims[0], mat_dims[1]),
            inset_m=mat_probe_inset,
        )
    ]
    mat_floor_values = [float(probe["z"]) for probe in mat_probes]
    mat_floor_min = min(mat_floor_values)
    mat_floor_max = max(mat_floor_values)
    mat_floor_variation = mat_floor_max - mat_floor_min
    if mat_floor_variation > max_mat_variation:
        raise RuntimeError(
            "sampled_mat_ground_height_variation_too_large: "
            f"variation={mat_floor_variation:.6f} limit={max_mat_variation:.6f}"
        )
    normal = _support_normal(
        collision_ply,
        face_index=int(cola_probe["face_index"]),
    )
    return {
        "collision_ply": str(collision_ply),
        "collision_ply_env": collision_ply_env,
        "collision_ply_expected_sha256": config.get("collision_ply_sha256"),
        "query_ceiling_z": query_ceiling_z,
        "cola": {**cola_probe, "normal_xyz": list(normal)},
        "mat": {
            "probes": mat_probes,
            "floor_min_z": mat_floor_min,
            "floor_max_z": mat_floor_max,
            "height_variation_m": mat_floor_variation,
            "max_height_variation_m": max_mat_variation,
        },
        "geometry_verified": True,
    }


def _sync_task_from_layout(
    task: dict[str, Any],
    *,
    layout: dict[str, Any],
    support: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    rejected: list[dict[str, Any]],
) -> None:
    robot_xyz = tuple(float(value) for value in layout["robot_xyz"])
    cola = layout["cola"]
    mat = layout["mat"]
    cola_xy = (float(cola["x"]), float(cola["y"]))
    mat_xy = (float(mat["x"]), float(mat["y"]))
    robot_yaw = float(layout["robot_yaw_rad"])
    mat_yaw = float(mat["yaw_rad"])
    mat_dims = _vector(
        config.get("mat_support_dims_xyz"),
        field_name="forward_sector.mat_support_dims_xyz",
        length=3,
    )
    mat_root_to_center = _vector(
        config.get("mat_root_to_support_center_xyz"),
        field_name="forward_sector.mat_root_to_support_center_xyz",
        length=3,
    )
    mat_root_to_top = _finite_float(
        config.get("mat_root_to_support_top_z_m"),
        field_name="forward_sector.mat_root_to_support_top_z_m",
    )
    mat_top_above_ground = _finite_float(
        config.get("mat_top_above_ground_m"),
        field_name="forward_sector.mat_top_above_ground_m",
    )
    cola_center_to_ground = _finite_float(
        config.get("cola_center_to_ground_m"),
        field_name="forward_sector.cola_center_to_ground_m",
    )
    place_object_half_height = _finite_float(
        config.get("place_object_bbox_center_to_min_z_m"),
        field_name="forward_sector.place_object_bbox_center_to_min_z_m",
    )
    placement_half_extent = _vector(
        config.get("placement_region_half_extent_xy_m"),
        field_name="forward_sector.placement_region_half_extent_xy_m",
        length=2,
    )
    if any(value <= 0.0 for value in placement_half_extent):
        raise ValueError(
            "forward_sector.placement_region_half_extent_xy_m 必须全部大于零"
        )

    cola_floor_z = float(support["cola"]["z"])
    mat_floor_z = float(support["mat"]["floor_max_z"])
    mat_top_z = mat_floor_z + mat_top_above_ground
    cola_center_z = cola_floor_z + cola_center_to_ground
    place_center_z = mat_top_z + place_object_half_height
    rotated_center_offset = _rotate_xy(
        (mat_root_to_center[0], mat_root_to_center[1]),
        mat_yaw,
    )
    mat_root_xyz = (
        mat_xy[0] - rotated_center_offset[0],
        mat_xy[1] - rotated_center_offset[1],
        mat_top_z - mat_root_to_top,
    )

    start = dict(task.get("start") or {})
    start.update(
        {
            "x": robot_xyz[0],
            "y": robot_xyz[1],
            "z": robot_xyz[2],
            "yaw": robot_yaw,
        }
    )
    task["start"] = start

    pick = dict(task.get("pick") or {})
    pick_pose = dict(pick.get("object_pose_world") or {})
    pick_pose.update(
        {
            "x": cola_xy[0],
            "y": cola_xy[1],
            "z": cola_center_z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(cola["yaw_rad"]),
        }
    )
    pick["object_pose_world"] = pick_pose
    pick["base_goal"] = _updated_goal(
        pick.get("base_goal"),
        sampled=layout["pick_base_goal"],
        default_z=robot_xyz[2],
    )
    pick["support_geometry"] = {
        "collision_support_z": cola_floor_z,
        "center_to_support_m": cola_center_to_ground,
        "collision_face_index": int(support["cola"]["face_index"]),
        "collision_support_normal_xyz": list(support["cola"]["normal_xyz"]),
        "collision_ply_sha256": config.get("collision_ply_sha256"),
        "source": "episode_collision_ply_vertical_probe",
    }
    pick_collision = dict(pick.get("curobo_world_collision") or {})
    pick_collision["enabled"] = True
    pick_collision["required"] = True
    pick_proxy_dims = _vector(
        config.get("pick_floor_proxy_dims_xyz", [0.45, 0.30, 0.04]),
        field_name="forward_sector.pick_floor_proxy_dims_xyz",
        length=3,
    )
    pick_normal = tuple(float(value) for value in support["cola"]["normal_xyz"])
    pick_proxy_center = [
        cola_xy[0] - pick_normal[0] * pick_proxy_dims[2] * 0.5,
        cola_xy[1] - pick_normal[1] * pick_proxy_dims[2] * 0.5,
        cola_floor_z - pick_normal[2] * pick_proxy_dims[2] * 0.5,
    ]
    pick_collision["cuboids_world"] = [
        {
            "name": "liangzhu_pick_floor_support_episode",
            "frame": "world",
            "semantic_role": "floor_support",
            "source_prim_path": (
                "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
            ),
            "center_xyz": pick_proxy_center,
            "quaternion_wxyz": list(_quat_align_z_to_normal(pick_normal)),
            "dims_xyz": list(pick_proxy_dims),
            "padding_mode": "preserve_top",
            "source": {
                "type": "episode_collision_ply_support_face_proxy",
                "support_point_xyz": [cola_xy[0], cola_xy[1], cola_floor_z],
                "support_normal_xyz": list(pick_normal),
                "collision_face_index": int(support["cola"]["face_index"]),
                "collision_ply_sha256": config.get("collision_ply_sha256"),
                "episode_seed": int(seed),
            },
        }
    ]
    pick["curobo_world_collision"] = pick_collision
    task["pick"] = pick

    place = dict(task.get("place") or {})
    place["base_goal"] = _updated_goal(
        place.get("base_goal"),
        sampled=layout["place_base_goal"],
        default_z=robot_xyz[2],
    )
    place["receptacle_pose_world"] = {
        "x": mat_root_xyz[0],
        "y": mat_root_xyz[1],
        "z": mat_root_xyz[2],
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": mat_yaw,
        "prim_path": str(place.get("target_receptacle_prim_path") or "/World/carpet"),
    }
    place_pose = dict(place.get("place_pose_world") or {})
    place_pose.update(
        {
            "x": mat_xy[0],
            "y": mat_xy[1],
            "z": place_center_z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(layout["place_object_yaw_rad"]),
        }
    )
    place["place_pose_world"] = place_pose
    place["placement_region"] = {
        "mode": "axis_aligned_safe_center_episode_randomized",
        "frame": "world",
        "center_xyz": [mat_xy[0], mat_xy[1], mat_top_z],
        "x_min": mat_xy[0] - placement_half_extent[0],
        "x_max": mat_xy[0] + placement_half_extent[0],
        "y_min": mat_xy[1] - placement_half_extent[1],
        "y_max": mat_xy[1] + placement_half_extent[1],
        "z_surface": mat_top_z,
        "object_footprint_radius_m": _finite_float(
            config.get("cola_footprint_radius_m"),
            field_name="forward_sector.cola_footprint_radius_m",
        ),
        "edge_clearance_m": _finite_float(
            config.get("placement_edge_clearance_m", 0.01),
            field_name="forward_sector.placement_edge_clearance_m",
        ),
    }
    mat_half_x_world = (
        abs(math.cos(mat_yaw)) * 0.5 * mat_dims[0]
        + abs(math.sin(mat_yaw)) * 0.5 * mat_dims[1]
    )
    mat_half_y_world = (
        abs(math.sin(mat_yaw)) * 0.5 * mat_dims[0]
        + abs(math.cos(mat_yaw)) * 0.5 * mat_dims[1]
    )
    mat_bbox_min_world = [
        mat_xy[0] - mat_half_x_world,
        mat_xy[1] - mat_half_y_world,
        mat_top_z - mat_dims[2],
    ]
    mat_bbox_max_world = [
        mat_xy[0] + mat_half_x_world,
        mat_xy[1] + mat_half_y_world,
        mat_top_z,
    ]
    place_collision = dict(place.get("curobo_world_collision") or {})
    place_collision["enabled"] = True
    place_collision["required"] = True
    place_collision["cuboids_world"] = [
        {
            "name": "liangzhu_mat_support_episode",
            "frame": "world",
            "semantic_role": "mat_support",
            "source_prim_path": str(place.get("target_support_prim_path")),
            "center_xyz": [
                mat_xy[0],
                mat_xy[1],
                mat_top_z - 0.5 * mat_dims[2],
            ],
            "quaternion_wxyz": _yaw_quaternion(mat_yaw),
            "dims_xyz": list(mat_dims),
            "padding_mode": "preserve_top",
            "source": {
                "type": "episode_randomized_receptacle_local_bbox_proxy",
                "asset_path": config.get("mat_asset_path"),
                "asset_sha256": config.get("mat_asset_sha256"),
                "world_bbox_min_xyz": mat_bbox_min_world,
                "world_bbox_max_xyz": mat_bbox_max_world,
                "support_surface_z": mat_top_z,
                "receptacle_pose_world": copy.deepcopy(
                    place["receptacle_pose_world"]
                ),
                "episode_seed": int(seed),
            },
        }
    ]
    place["curobo_world_collision"] = place_collision
    task["place"] = place

    task.pop("phase0_spatial_preconditions", None)
    task["spatial_preconditions"] = {
        "mode": FORWARD_SECTOR_MODE,
        "robot_fixed_translate_xyz": list(robot_xyz),
        "robot_yaw_randomized": True,
        "cola_in_robot_forward_sector": True,
        "mat_in_robot_forward_sector": True,
        "cola_not_on_mat": True,
        "cola_mat_center_distance_m": float(layout["cola_mat_center_distance_m"]),
        "cola_on_floor_geometry_verified": True,
        "mat_on_floor_geometry_verified": True,
        "mat_ground_height_variation_m": float(support["mat"]["height_variation_m"]),
    }

    randomization = dict(task.get("randomization") or {})
    randomization.update(
        {
            "enabled": True,
            "seed": int(seed),
            "sample": {
                **copy.deepcopy(layout),
                "rejected_layout_samples": copy.deepcopy(rejected),
                "ground_support": copy.deepcopy(support),
                "mat_root_pose_world": copy.deepcopy(
                    place["receptacle_pose_world"]
                ),
                "cola_pose_world": copy.deepcopy(pick_pose),
                "place_pose_world": copy.deepcopy(place_pose),
            },
            "synchronization": {
                "robot_start": True,
                "object_pose": True,
                "receptacle_stage_pose": True,
                "placement_region": True,
                "pick_floor_collision_proxy": True,
                "place_mat_collision_proxy": True,
                "pick_base_goal": True,
                "place_base_goal": True,
                "mesh_truth_targets": True,
            },
            "object_pose_policy": {
                "mode": "ground_supported_xyz_plus_random_yaw",
                "randomize_xy": True,
                "randomize_z": True,
                "randomize_roll": False,
                "randomize_pitch": False,
                "randomize_yaw": True,
            },
        }
    )
    task["randomization"] = randomization
    notes = dict(task.get("notes") or {})
    notes["randomization_guard"] = (
        "前向扇形 episode 会同步重建 stage receptacle pose、placement region、"
        "pick/place CuRobo 支撑代理与导航交接点。"
    )
    notes["phase"] = "phase1_forward_sector_randomization"
    task["notes"] = notes


def apply_forward_sector_randomization(
    task: dict[str, Any],
    *,
    seed: int,
    randomize_base_goal: bool,
    collision_ply_path: str | Path | None = None,
) -> None:
    """联合采样并同步写回一个可复现的良渚地面任务 episode。"""

    config = copy.deepcopy(_layout_config(task))
    if collision_ply_path is not None:
        config["collision_ply_path"] = str(
            Path(collision_ply_path).expanduser().resolve()
        )
    rng = random.Random(int(seed))
    rejected: list[dict[str, Any]] = []
    max_support_resamples = int(config.get("max_support_resamples", 20))
    if max_support_resamples < 1:
        raise ValueError("forward_sector.max_support_resamples 必须至少为 1")
    last_error: Exception | None = None
    for _support_attempt in range(1, max_support_resamples + 1):
        layout, geometry_rejected = _sample_layout_geometry(
            rng=rng,
            config=config,
            randomize_base_goal=randomize_base_goal,
        )
        rejected.extend(geometry_rejected)
        try:
            support = _apply_support_geometry(
                layout=layout,
                config=config,
            )
        except RuntimeError as exc:
            last_error = exc
            rejected.append(
                {
                    "attempt": int(layout["attempt"]),
                    "reason": str(exc),
                    "stage": "ground_support",
                }
            )
            continue
        _sync_task_from_layout(
            task,
            layout=layout,
            support=support,
            config=config,
            seed=int(seed),
            rejected=rejected,
        )
        return
    raise RuntimeError(
        "failed_to_sample_ground_supported_forward_sector_layout: "
        f"attempts={max_support_resamples} last_error={last_error}"
    )
