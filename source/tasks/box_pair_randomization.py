"""良渚 box1 取可乐并放到 box2 的联合随机化。"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

from .forward_sector_randomization import (
    _finite_float,
    _goal_facing_target,
    _mat_probe_points,
    _point_to_oriented_mat_distance,
    _positive_range,
    _range,
    _resolve_collision_ply,
    _surface_probe,
    _updated_goal,
    _vector,
)


BOX_PAIR_MODE = "liangzhu_box_pair_xy_v1"


def uses_box_pair_randomization(raw_task: dict[str, Any]) -> bool:
    """判断任务是否声明了双箱桌面搬运随机化。"""

    randomization = raw_task.get("randomization")
    return bool(
        isinstance(randomization, dict)
        and randomization.get("mode") == BOX_PAIR_MODE
    )


def _layout_config(raw_task: dict[str, Any]) -> dict[str, Any]:
    randomization = raw_task.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("task.randomization 必须是对象")
    config = randomization.get("box_pair")
    if not isinstance(config, dict):
        raise ValueError("双箱随机化要求 task.randomization.box_pair")
    return copy.deepcopy(config)


def _table_config(config: dict[str, Any], table_name: str) -> dict[str, Any]:
    """读取一张箱桌的固定姿态、组合几何与 XY 采样范围。"""

    raw = config.get(table_name)
    field = f"box_pair.{table_name}"
    if not isinstance(raw, dict):
        raise ValueError(f"{field} 必须是对象")
    root_xyz = _vector(
        raw.get("root_translate_xyz"),
        field_name=f"{field}.root_translate_xyz",
        length=3,
    )
    center_xy = _vector(
        raw.get("support_center_xy"),
        field_name=f"{field}.support_center_xy",
        length=2,
    )
    dims = _vector(
        raw.get("support_dims_xyz"),
        field_name=f"{field}.support_dims_xyz",
        length=3,
    )
    if any(value <= 0.0 for value in dims):
        raise ValueError(f"{field}.support_dims_xyz 必须全部大于零")
    x_offset = _range(
        raw.get("center_x_offset_range_m"),
        field_name=f"{field}.center_x_offset_range_m",
    )
    y_offset = _range(
        raw.get("center_y_offset_range_m"),
        field_name=f"{field}.center_y_offset_range_m",
    )
    root_prim_path = str(raw.get("root_prim_path") or "").strip()
    support_prim_path = str(raw.get("support_prim_path") or "").strip()
    if not root_prim_path.startswith("/") or root_prim_path == "/":
        raise ValueError(f"{field}.root_prim_path 必须是绝对 USD prim path")
    if not support_prim_path.startswith(root_prim_path.rstrip("/") + "/"):
        raise ValueError(f"{field}.support_prim_path 必须位于 root_prim_path 下")
    support_top_z = _finite_float(
        raw.get("support_top_z"),
        field_name=f"{field}.support_top_z",
    )
    fixed_world_yaw = _finite_float(
        raw.get("fixed_world_yaw_rad", 0.0),
        field_name=f"{field}.fixed_world_yaw_rad",
    )
    return {
        "name": table_name,
        "root_prim_path": root_prim_path.rstrip("/"),
        "support_prim_path": support_prim_path.rstrip("/"),
        "root_translate_xyz": root_xyz,
        "support_center_xy": center_xy,
        "support_dims_xyz": dims,
        "support_top_z": support_top_z,
        "fixed_world_yaw_rad": fixed_world_yaw,
        "center_x_offset_range_m": x_offset,
        "center_y_offset_range_m": y_offset,
        "asset_path": str(raw.get("asset_path") or ""),
        "asset_sha256": str(raw.get("asset_sha256") or ""),
    }


def _sample_table(
    rng: random.Random,
    table: dict[str, Any],
) -> dict[str, Any]:
    """仅平移桌子 XY；Z 和所有 USD 姿态 op 保持模板值。"""

    dx = rng.uniform(*table["center_x_offset_range_m"])
    dy = rng.uniform(*table["center_y_offset_range_m"])
    nominal_center = table["support_center_xy"]
    nominal_root = table["root_translate_xyz"]
    center_xy = (nominal_center[0] + dx, nominal_center[1] + dy)
    root_xyz = (nominal_root[0] + dx, nominal_root[1] + dy, nominal_root[2])
    dims = table["support_dims_xyz"]
    top_z = float(table["support_top_z"])
    return {
        **table,
        "offset_xy_m": [dx, dy],
        "support_center_xy_sampled": list(center_xy),
        "root_translate_xyz_sampled": list(root_xyz),
        "support_bbox_min_xyz": [
            center_xy[0] - 0.5 * dims[0],
            center_xy[1] - 0.5 * dims[1],
            top_z - dims[2],
        ],
        "support_bbox_max_xyz": [
            center_xy[0] + 0.5 * dims[0],
            center_xy[1] + 0.5 * dims[1],
            top_z,
        ],
    }


def _table_ground_report(
    *,
    collision_ply: Path,
    table: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """检查固定 Z 的桌子在采样 XY 处不会悬空或严重穿地。"""

    center_xy = tuple(float(value) for value in table["support_center_xy_sampled"])
    dims = tuple(float(value) for value in table["support_dims_xyz"])
    inset = _finite_float(
        config.get("table_ground_probe_inset_m", 0.02),
        field_name="box_pair.table_ground_probe_inset_m",
    )
    ceiling = _finite_float(
        config.get("ground_query_ceiling_z", 2.0),
        field_name="box_pair.ground_query_ceiling_z",
    )
    probes = [
        _surface_probe(collision_ply, xy=point, query_ceiling_z=ceiling)
        for point in _mat_probe_points(
            center_xy=center_xy,
            yaw_rad=float(table["fixed_world_yaw_rad"]),
            dims_xy_m=(dims[0], dims[1]),
            inset_m=inset,
        )
    ]
    floor_values = [float(probe["z"]) for probe in probes]
    floor_min = min(floor_values)
    floor_max = max(floor_values)
    variation = floor_max - floor_min
    max_variation = _finite_float(
        config.get("max_table_ground_height_variation_m", 0.035),
        field_name="box_pair.max_table_ground_height_variation_m",
    )
    if variation > max_variation:
        raise RuntimeError(
            f"{table['name']}_ground_height_variation_too_large: "
            f"variation={variation:.6f} limit={max_variation:.6f}"
        )
    bottom_z = float(table["support_bbox_min_xyz"][2])
    bottom_to_highest_ground = bottom_z - floor_max
    clearance_range = _range(
        config.get("table_bottom_to_ground_range_m", [-0.04, 0.01]),
        field_name="box_pair.table_bottom_to_ground_range_m",
    )
    if not clearance_range[0] <= bottom_to_highest_ground <= clearance_range[1]:
        raise RuntimeError(
            f"{table['name']}_bottom_ground_clearance_out_of_range: "
            f"clearance={bottom_to_highest_ground:.6f} range={clearance_range}"
        )
    return {
        "probes": probes,
        "floor_min_z": floor_min,
        "floor_max_z": floor_max,
        "height_variation_m": variation,
        "max_height_variation_m": max_variation,
        "table_bottom_z": bottom_z,
        "bottom_to_highest_ground_m": bottom_to_highest_ground,
        "allowed_bottom_to_ground_range_m": list(clearance_range),
        "geometry_verified": True,
    }


def _point_table_clearance(
    point_xy: tuple[float, float],
    table: dict[str, Any],
) -> float:
    distance, _local = _point_to_oriented_mat_distance(
        point_xy=point_xy,
        mat_center_xy=tuple(table["support_center_xy_sampled"]),
        mat_yaw_rad=float(table["fixed_world_yaw_rad"]),
        mat_dims_xy_m=tuple(table["support_dims_xyz"][:2]),
    )
    return float(distance)


def _sample_robot_between_tables(
    *,
    rng: random.Random,
    box1: dict[str, Any],
    box2: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """在两张桌子的连线中段采样机器人 XY 和全范围 yaw。"""

    alpha_range = _range(
        config.get("robot_segment_fraction_range", [0.40, 0.60]),
        field_name="box_pair.robot_segment_fraction_range",
    )
    if alpha_range[0] < 0.0 or alpha_range[1] > 1.0:
        raise ValueError("box_pair.robot_segment_fraction_range 必须位于 [0, 1]")
    lateral_range = _range(
        config.get("robot_lateral_offset_range_m", [-0.18, 0.18]),
        field_name="box_pair.robot_lateral_offset_range_m",
    )
    yaw_range_deg = _range(
        config.get("robot_yaw_range_deg", [-180.0, 180.0]),
        field_name="box_pair.robot_yaw_range_deg",
    )
    box2_xy = tuple(float(value) for value in box2["support_center_xy_sampled"])
    box1_xy = tuple(float(value) for value in box1["support_center_xy_sampled"])
    segment = (box1_xy[0] - box2_xy[0], box1_xy[1] - box2_xy[1])
    segment_length = math.hypot(*segment)
    if segment_length <= 1.0e-6:
        raise RuntimeError("box_pair_table_centers_are_coincident")
    perpendicular = (-segment[1] / segment_length, segment[0] / segment_length)
    alpha = rng.uniform(*alpha_range)
    lateral = rng.uniform(*lateral_range)
    xy = (
        box2_xy[0] + alpha * segment[0] + lateral * perpendicular[0],
        box2_xy[1] + alpha * segment[1] + lateral * perpendicular[1],
    )
    minimum_clearance = _finite_float(
        config.get("robot_table_min_clearance_m", 0.55),
        field_name="box_pair.robot_table_min_clearance_m",
    )
    clearances = {
        "box1": _point_table_clearance(xy, box1),
        "box2": _point_table_clearance(xy, box2),
    }
    if min(clearances.values()) < minimum_clearance:
        raise RuntimeError(
            "sampled_robot_too_close_to_table: "
            f"clearances={clearances} required={minimum_clearance}"
        )
    yaw = math.radians(rng.uniform(*yaw_range_deg))
    return {
        "xy": list(xy),
        "yaw_rad": math.atan2(math.sin(yaw), math.cos(yaw)),
        "segment_fraction_box2_to_box1": alpha,
        "lateral_offset_m": lateral,
        "table_clearance_m": clearances,
        "minimum_table_clearance_m": minimum_clearance,
    }


def _table_keepout(table: dict[str, Any], *, margin_m: float) -> dict[str, Any]:
    dims = tuple(float(value) for value in table["support_dims_xyz"])
    radius = math.hypot(0.5 * dims[0], 0.5 * dims[1]) + float(margin_m)
    return {
        "id": str(table["name"]),
        "center_xy": list(table["support_center_xy_sampled"]),
        "radius_m": radius,
        "phases": ["nav_to_pick", "nav_to_place"],
        "source_prim_path": str(table["support_prim_path"]),
        "source": "episode_randomized_table_footprint",
    }


def _support_pose(table: dict[str, Any]) -> dict[str, Any]:
    root_xyz = table["root_translate_xyz_sampled"]
    return {
        "prim_path": str(table["root_prim_path"]),
        "x": float(root_xyz[0]),
        "y": float(root_xyz[1]),
        "z": float(root_xyz[2]),
        "translation_only": True,
        "collision_prim_path": str(table["support_prim_path"]),
        "ensure_static_mesh_collision": True,
        "expected_support_bbox_dims_xyz": list(table["support_dims_xyz"]),
        "support_bbox_tolerance_m": 5.0e-5,
    }


def _curobo_table_proxy(
    table: dict[str, Any],
    *,
    name: str,
    seed: int,
) -> dict[str, Any]:
    bbox_min = tuple(float(value) for value in table["support_bbox_min_xyz"])
    bbox_max = tuple(float(value) for value in table["support_bbox_max_xyz"])
    return {
        "name": name,
        "frame": "world",
        "semantic_role": "table_support",
        "source_prim_path": str(table["support_prim_path"]),
        "center_xyz": [
            (bbox_min[index] + bbox_max[index]) * 0.5 for index in range(3)
        ],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "dims_xyz": list(table["support_dims_xyz"]),
        "padding_mode": "preserve_top",
        "source": {
            "type": "episode_randomized_static_table_bbox_proxy",
            "asset_path": str(table["asset_path"]),
            "asset_sha256": str(table["asset_sha256"]),
            "world_bbox_min_xyz": list(bbox_min),
            "world_bbox_max_xyz": list(bbox_max),
            "support_surface_z": float(bbox_max[2]),
            "root_translate_xyz": list(table["root_translate_xyz_sampled"]),
            "orientation_policy": "preserve_authored_xform_ops",
            "episode_seed": int(seed),
        },
    }


def _sample_layout(
    *,
    rng: random.Random,
    config: dict[str, Any],
    collision_ply: Path,
    randomize_base_goal: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """联合采样并拒绝没有地面支撑或操作净空的布局。"""

    box1_config = _table_config(config, "box1")
    box2_config = _table_config(config, "box2")
    max_attempts = int(config.get("max_attempts", 300))
    if max_attempts < 1:
        raise ValueError("box_pair.max_attempts 必须至少为 1")
    minimum_center_distance = _finite_float(
        config.get("min_table_center_distance_m", 2.4),
        field_name="box_pair.min_table_center_distance_m",
    )
    cola_half_extent = _vector(
        config.get("cola_center_region_half_extent_xy_m", [0.08, 0.07]),
        field_name="box_pair.cola_center_region_half_extent_xy_m",
        length=2,
    )
    if any(value < 0.0 for value in cola_half_extent):
        raise ValueError(
            "box_pair.cola_center_region_half_extent_xy_m 必须全部非负"
        )
    cola_yaw_deg = _range(
        config.get("cola_yaw_range_deg", [-180.0, 180.0]),
        field_name="box_pair.cola_yaw_range_deg",
    )
    cola_half_height = _finite_float(
        config.get("cola_bbox_center_to_min_z_m"),
        field_name="box_pair.cola_bbox_center_to_min_z_m",
    )
    cola_radius = _finite_float(
        config.get("cola_footprint_radius_m", 0.03),
        field_name="box_pair.cola_footprint_radius_m",
    )
    cola_edge_clearance = _finite_float(
        config.get("cola_table_edge_clearance_m", 0.04),
        field_name="box_pair.cola_table_edge_clearance_m",
    )
    box1_safe_half = (
        0.5 * box1_config["support_dims_xyz"][0] - cola_radius - cola_edge_clearance,
        0.5 * box1_config["support_dims_xyz"][1] - cola_radius - cola_edge_clearance,
    )
    if any(value <= 0.0 for value in box1_safe_half):
        raise ValueError("box1 桌面不足以容纳可乐安全区")
    if any(
        requested > safe + 1.0e-9
        for requested, safe in zip(cola_half_extent, box1_safe_half)
    ):
        raise ValueError(
            "box_pair.cola_center_region_half_extent_xy_m 超出 box1 安全区"
        )

    pick_standoff = _positive_range(
        config.get("pick_base_standoff_range_m", [0.50, 0.54]),
        field_name="box_pair.pick_base_standoff_range_m",
    )
    place_standoff = _positive_range(
        config.get("place_base_standoff_range_m", [0.50, 0.54]),
        field_name="box_pair.place_base_standoff_range_m",
    )
    angle_noise = math.radians(
        _finite_float(
            config.get("base_approach_angle_noise_deg", 0.0),
            field_name="box_pair.base_approach_angle_noise_deg",
        )
    )
    base_table_clearance = _finite_float(
        config.get("base_goal_table_min_clearance_m", 0.20),
        field_name="box_pair.base_goal_table_min_clearance_m",
    )
    root_to_ground = _finite_float(
        config.get("robot_root_to_ground_m"),
        field_name="box_pair.robot_root_to_ground_m",
    )
    query_ceiling = _finite_float(
        config.get("ground_query_ceiling_z", 2.0),
        field_name="box_pair.ground_query_ceiling_z",
    )

    rejected: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            box1 = _sample_table(rng, box1_config)
            box2 = _sample_table(rng, box2_config)
            center_distance = math.dist(
                box1["support_center_xy_sampled"],
                box2["support_center_xy_sampled"],
            )
            if center_distance < minimum_center_distance:
                raise RuntimeError(
                    "table_center_distance_too_small: "
                    f"distance={center_distance:.6f} required={minimum_center_distance:.6f}"
                )
            box1_ground = _table_ground_report(
                collision_ply=collision_ply,
                table=box1,
                config=config,
            )
            box2_ground = _table_ground_report(
                collision_ply=collision_ply,
                table=box2,
                config=config,
            )
            robot = _sample_robot_between_tables(
                rng=rng,
                box1=box1,
                box2=box2,
                config=config,
            )

            cola_local_xy = (
                rng.uniform(-cola_half_extent[0], cola_half_extent[0]),
                rng.uniform(-cola_half_extent[1], cola_half_extent[1]),
            )
            box1_center = tuple(box1["support_center_xy_sampled"])
            cola_xy = (
                box1_center[0] + cola_local_xy[0],
                box1_center[1] + cola_local_xy[1],
            )
            cola_yaw = math.radians(rng.uniform(*cola_yaw_deg))
            cola_yaw = math.atan2(math.sin(cola_yaw), math.cos(cola_yaw))
            cola_xyz = (
                cola_xy[0],
                cola_xy[1],
                float(box1["support_top_z"]) + cola_half_height,
            )
            box2_center = tuple(box2["support_center_xy_sampled"])

            pick_goal = _goal_facing_target(
                rng=rng,
                robot_xy=tuple(robot["xy"]),
                target_xy=cola_xy,
                standoff_range_m=pick_standoff,
                angle_noise_rad=angle_noise,
                randomize=randomize_base_goal,
            )
            place_goal = _goal_facing_target(
                rng=rng,
                robot_xy=(float(pick_goal["x"]), float(pick_goal["y"])),
                target_xy=box2_center,
                standoff_range_m=place_standoff,
                angle_noise_rad=angle_noise,
                randomize=randomize_base_goal,
            )
            place_goal["approach_origin"] = "pick_base_goal"
            pick_clearance = _point_table_clearance(
                (float(pick_goal["x"]), float(pick_goal["y"])), box1
            )
            place_clearance = _point_table_clearance(
                (float(place_goal["x"]), float(place_goal["y"])), box2
            )
            if min(pick_clearance, place_clearance) < base_table_clearance:
                raise RuntimeError(
                    "base_goal_table_clearance_too_small: "
                    f"pick={pick_clearance:.6f} place={place_clearance:.6f} "
                    f"required={base_table_clearance:.6f}"
                )

            point_probes = {
                "robot": _surface_probe(
                    collision_ply,
                    xy=tuple(robot["xy"]),
                    query_ceiling_z=query_ceiling,
                ),
                "pick_base_goal": _surface_probe(
                    collision_ply,
                    xy=(float(pick_goal["x"]), float(pick_goal["y"])),
                    query_ceiling_z=query_ceiling,
                ),
                "place_base_goal": _surface_probe(
                    collision_ply,
                    xy=(float(place_goal["x"]), float(place_goal["y"])),
                    query_ceiling_z=query_ceiling,
                ),
            }
            robot_xyz = [
                float(robot["xy"][0]),
                float(robot["xy"][1]),
                float(point_probes["robot"]["z"]) + root_to_ground,
            ]
            pick_goal["z"] = (
                float(point_probes["pick_base_goal"]["z"]) + root_to_ground
            )
            place_goal["z"] = (
                float(point_probes["place_base_goal"]["z"]) + root_to_ground
            )
            return (
                {
                    "attempt": attempt,
                    "box1": box1,
                    "box2": box2,
                    "box1_ground": box1_ground,
                    "box2_ground": box2_ground,
                    "table_center_distance_m": center_distance,
                    "robot": {**robot, "xyz": robot_xyz},
                    "cola": {
                        "xyz": list(cola_xyz),
                        "yaw_rad": cola_yaw,
                        "local_xy_from_box1_center_m": list(cola_local_xy),
                        "center_region_half_extent_xy_m": list(cola_half_extent),
                        "footprint_radius_m": cola_radius,
                        "edge_clearance_m": cola_edge_clearance,
                    },
                    "pick_base_goal": pick_goal,
                    "place_base_goal": place_goal,
                    "base_goal_randomized": bool(randomize_base_goal),
                    "ground_probes": point_probes,
                },
                rejected,
            )
        except RuntimeError as exc:
            rejected.append(
                {
                    "attempt": attempt,
                    "reason": str(exc),
                }
            )
    raise RuntimeError(
        "failed_to_sample_box_pair_layout: "
        f"attempts={max_attempts} last_reason="
        f"{(rejected[-1] if rejected else {}).get('reason')}"
    )


def _sync_task_from_layout(
    task: dict[str, Any],
    *,
    layout: dict[str, Any],
    config: dict[str, Any],
    collision_ply: Path,
    collision_ply_env: str | None,
    seed: int,
    rejected: list[dict[str, Any]],
) -> None:
    """把同一布局同步写入仿真、导航、CuRobo、验证与数据 metadata。"""

    box1 = layout["box1"]
    box2 = layout["box2"]
    robot = layout["robot"]
    cola = layout["cola"]
    cola_xyz = tuple(float(value) for value in cola["xyz"])
    box2_center = tuple(float(value) for value in box2["support_center_xy_sampled"])
    cola_half_height = _finite_float(
        config.get("cola_bbox_center_to_min_z_m"),
        field_name="box_pair.cola_bbox_center_to_min_z_m",
    )
    placement_half_extent = _vector(
        config.get("placement_region_half_extent_xy_m", [0.10, 0.05]),
        field_name="box_pair.placement_region_half_extent_xy_m",
        length=2,
    )
    if any(value <= 0.0 for value in placement_half_extent):
        raise ValueError(
            "box_pair.placement_region_half_extent_xy_m 必须全部大于零"
        )
    box2_safe_half = (
        0.5 * float(box2["support_dims_xyz"][0])
        - _finite_float(
            config.get("cola_footprint_radius_m", 0.03),
            field_name="box_pair.cola_footprint_radius_m",
        )
        - _finite_float(
            config.get("placement_edge_clearance_m", 0.04),
            field_name="box_pair.placement_edge_clearance_m",
        ),
        0.5 * float(box2["support_dims_xyz"][1])
        - _finite_float(
            config.get("cola_footprint_radius_m", 0.03),
            field_name="box_pair.cola_footprint_radius_m",
        )
        - _finite_float(
            config.get("placement_edge_clearance_m", 0.04),
            field_name="box_pair.placement_edge_clearance_m",
        ),
    )
    if any(
        requested > safe + 1.0e-9
        for requested, safe in zip(placement_half_extent, box2_safe_half)
    ):
        raise ValueError("box_pair.placement_region_half_extent_xy_m 超出 box2 安全区")

    start = dict(task.get("start") or {})
    start.update(
        {
            "x": float(robot["xyz"][0]),
            "y": float(robot["xyz"][1]),
            "z": float(robot["xyz"][2]),
            "yaw": float(robot["yaw_rad"]),
        }
    )
    task["start"] = start

    pick = dict(task.get("pick") or {})
    pick_pose = dict(pick.get("object_pose_world") or {})
    pick_pose.update(
        {
            "x": cola_xyz[0],
            "y": cola_xyz[1],
            "z": cola_xyz[2],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(cola["yaw_rad"]),
        }
    )
    pick["object_pose_world"] = pick_pose
    pick["base_goal"] = _updated_goal(
        pick.get("base_goal"),
        sampled=layout["pick_base_goal"],
        default_z=float(robot["xyz"][2]),
    )
    pick["base_goal"]["z"] = float(layout["pick_base_goal"]["z"])
    pick["target_support_root_prim_path"] = str(box1["root_prim_path"])
    pick["target_support_prim_path"] = str(box1["support_prim_path"])
    pick["support_pose_world"] = _support_pose(box1)
    pick["support_geometry"] = {
        "support_surface_z": float(box1["support_top_z"]),
        "object_center_to_support_m": cola_half_height,
        "world_bbox_min_xyz": list(box1["support_bbox_min_xyz"]),
        "world_bbox_max_xyz": list(box1["support_bbox_max_xyz"]),
        "world_bbox_dims_xyz": list(box1["support_dims_xyz"]),
        "ground_report": copy.deepcopy(layout["box1_ground"]),
        "source": "episode_randomized_box1_composed_bbox",
    }
    pick_collision = dict(pick.get("curobo_world_collision") or {})
    pick_collision.update(
        {
            "enabled": True,
            "required": True,
            "cuboids_world": [
                _curobo_table_proxy(
                    box1,
                    name="liangzhu_box1_pick_support_episode",
                    seed=seed,
                )
            ],
        }
    )
    pick["curobo_world_collision"] = pick_collision
    task["pick"] = pick

    place = dict(task.get("place") or {})
    place["base_goal"] = _updated_goal(
        place.get("base_goal"),
        sampled=layout["place_base_goal"],
        default_z=float(robot["xyz"][2]),
    )
    place["base_goal"]["z"] = float(layout["place_base_goal"]["z"])
    place["receptacle_pose_world"] = _support_pose(box2)
    place_center_z = float(box2["support_top_z"]) + cola_half_height
    place_pose = dict(place.get("place_pose_world") or {})
    place_pose.update(
        {
            "x": box2_center[0],
            "y": box2_center[1],
            "z": place_center_z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(cola["yaw_rad"]),
        }
    )
    place["place_pose_world"] = place_pose
    place["placement_region"] = {
        "mode": "axis_aligned_box2_safe_center_episode_randomized",
        "frame": "world",
        "center_xyz": [box2_center[0], box2_center[1], float(box2["support_top_z"])],
        "x_min": box2_center[0] - placement_half_extent[0],
        "x_max": box2_center[0] + placement_half_extent[0],
        "y_min": box2_center[1] - placement_half_extent[1],
        "y_max": box2_center[1] + placement_half_extent[1],
        "z_surface": float(box2["support_top_z"]),
        "object_footprint_radius_m": _finite_float(
            config.get("cola_footprint_radius_m", 0.03),
            field_name="box_pair.cola_footprint_radius_m",
        ),
        "edge_clearance_m": _finite_float(
            config.get("placement_edge_clearance_m", 0.04),
            field_name="box_pair.placement_edge_clearance_m",
        ),
    }
    place_collision = dict(place.get("curobo_world_collision") or {})
    place_collision.update(
        {
            "enabled": True,
            "required": True,
            "cuboids_world": [
                _curobo_table_proxy(
                    box2,
                    name="liangzhu_box2_place_support_episode",
                    seed=seed,
                )
            ],
        }
    )
    place["curobo_world_collision"] = place_collision
    task["place"] = place

    keepout_margin = _finite_float(
        config.get("navigation_keepout_margin_m", 0.04),
        field_name="box_pair.navigation_keepout_margin_m",
    )
    task["navigation_dynamic_keepouts"] = [
        _table_keepout(box1, margin_m=keepout_margin),
        _table_keepout(box2, margin_m=keepout_margin),
    ]
    task.pop("phase0_spatial_preconditions", None)
    task["spatial_preconditions"] = {
        "mode": BOX_PAIR_MODE,
        "box1_xy_randomized_only": True,
        "box2_xy_randomized_only": True,
        "box_orientation_and_z_preserved": True,
        "cola_supported_by_box1": True,
        "cola_xy_and_yaw_randomized_near_box1_center": True,
        "robot_between_tables": True,
        "robot_yaw_randomized": True,
        "table_center_distance_m": float(layout["table_center_distance_m"]),
        "box1_ground_geometry_verified": True,
        "box2_ground_geometry_verified": True,
    }

    randomization = dict(task.get("randomization") or {})
    box1_nominal = box1["support_center_xy"]
    box2_nominal = box2["support_center_xy"]
    cola_region = cola["center_region_half_extent_xy_m"]
    box1_x_offset = box1["center_x_offset_range_m"]
    box1_y_offset = box1["center_y_offset_range_m"]
    box2_x_offset = box2["center_x_offset_range_m"]
    box2_y_offset = box2["center_y_offset_range_m"]
    randomization.update(
        {
            "enabled": True,
            "seed": int(seed),
            "sample": {
                **copy.deepcopy(layout),
                "rejected_layout_samples": copy.deepcopy(rejected),
                "collision_ply": str(collision_ply),
                "collision_ply_env": collision_ply_env,
                "collision_ply_expected_sha256": config.get(
                    "collision_ply_sha256"
                ),
            },
            "object_xy_randomization": {
                "enabled": True,
                "mode": "box1_center_offset_plus_local_safe_region",
                "sampled_xy": [cola_xyz[0], cola_xyz[1]],
                "x_range_m": [
                    box1_nominal[0] + box1_x_offset[0] - cola_region[0],
                    box1_nominal[0] + box1_x_offset[1] + cola_region[0],
                ],
                "y_range_m": [
                    box1_nominal[1] + box1_y_offset[0] - cola_region[1],
                    box1_nominal[1] + box1_y_offset[1] + cola_region[1],
                ],
            },
            "place_xy_randomization": {
                "enabled": True,
                "mode": "box2_center_tracks_randomized_table",
                "sampled_xy": list(box2_center),
                "x_range_m": [
                    box2_nominal[0] + box2_x_offset[0],
                    box2_nominal[0] + box2_x_offset[1],
                ],
                "y_range_m": [
                    box2_nominal[1] + box2_y_offset[0],
                    box2_nominal[1] + box2_y_offset[1],
                ],
            },
            "synchronization": {
                "robot_start": True,
                "box1_stage_pose": True,
                "box2_stage_pose": True,
                "box1_static_collision": True,
                "box2_static_collision": True,
                "cola_pose": True,
                "placement_region": True,
                "pick_table_collision_proxy": True,
                "place_table_collision_proxy": True,
                "navigation_table_keepouts": True,
                "pick_base_goal": True,
                "place_base_goal": True,
                "mesh_truth_targets": True,
            },
            "object_pose_policy": {
                "mode": "box1_safe_center_xy_plus_random_yaw",
                "randomize_xy": True,
                "randomize_z": False,
                "randomize_roll": False,
                "randomize_pitch": False,
                "randomize_yaw": True,
            },
        }
    )
    task["randomization"] = randomization
    notes = dict(task.get("notes") or {})
    notes["randomization_guard"] = (
        "box_pair episode 同步更新双桌根 XY、box2 静态碰撞、box1 桌面可乐、"
        "桌间机器人、导航 keepout、抓放 base goal、placement region 与 CuRobo 代理；"
        "双桌 Z、scale、orient 和 unitsResolve 均保持场景 authored 值。"
    )
    notes["phase"] = "box1_cola_to_box2_randomization"
    task["notes"] = notes


def apply_box_pair_randomization(
    task: dict[str, Any],
    *,
    seed: int,
    randomize_base_goal: bool,
    collision_ply_path: str | Path | None = None,
) -> None:
    """按 seed 生成一个可复现的 box1 到 box2 episode。"""

    config = _layout_config(task)
    if collision_ply_path is not None:
        config["collision_ply_path"] = str(
            Path(collision_ply_path).expanduser().resolve()
        )
    collision_ply, collision_ply_env = _resolve_collision_ply(config)
    layout, rejected = _sample_layout(
        rng=random.Random(int(seed)),
        config=config,
        collision_ply=collision_ply,
        randomize_base_goal=randomize_base_goal,
    )
    _sync_task_from_layout(
        task,
        layout=layout,
        config=config,
        collision_ply=collision_ply,
        collision_ply_env=collision_ply_env,
        seed=int(seed),
        rejected=rejected,
    )


__all__ = [
    "BOX_PAIR_MODE",
    "apply_box_pair_randomization",
    "uses_box_pair_randomization",
]
