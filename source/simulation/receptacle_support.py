"""任务级 receptacle 支撑体的 USD 组合与运行时检查。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _absolute_prim_path(value: Any, *, field_name: str) -> str:
    """把任务字段规范为绝对 prim path。"""

    path = str(value or "").strip()
    if not path.startswith("/") or path == "/" or "//" in path:
        raise ValueError(f"{field_name} 必须是绝对 USD prim path: {value!r}")
    return path.rstrip("/")


def resolve_task_receptacle_support_settings(
    raw_task: dict[str, Any],
) -> dict[str, Any]:
    """解析 place receptacle/support 约定；未配置的旧任务保持兼容。"""

    raw_place = raw_task.get("place")
    if raw_place is None:
        raw_place = {}
    if not isinstance(raw_place, dict):
        raise ValueError("task.place 必须是对象")

    enabled = bool(raw_place.get("enabled", False))
    validation_required = bool(
        raw_place.get("support_runtime_validation_required", False)
    )
    receptacle_raw = raw_place.get("target_receptacle_prim_path") or raw_task.get(
        "target_receptacle_prim_path"
    )
    support_raw = raw_place.get("target_support_prim_path") or raw_place.get(
        "collision_support_prim_path"
    )
    any_path_configured = bool(receptacle_raw or support_raw)

    if not enabled or not any_path_configured:
        if validation_required:
            raise ValueError(
                "place.support_runtime_validation_required=true 时必须启用 place，"
                "并配置 target_receptacle_prim_path 与 target_support_prim_path"
            )
        return {
            "configured": False,
            "place_enabled": enabled,
            "runtime_validation_required": False,
            "reason": "place_disabled" if not enabled else "support_paths_not_configured",
        }

    if not receptacle_raw or not support_raw:
        raise ValueError(
            "任务级 receptacle 支撑检查要求同时配置 "
            "place.target_receptacle_prim_path 与 place.target_support_prim_path"
        )
    receptacle_path = _absolute_prim_path(
        receptacle_raw,
        field_name="task.place.target_receptacle_prim_path",
    )
    support_path = _absolute_prim_path(
        support_raw,
        field_name="task.place.target_support_prim_path",
    )
    if not support_path.startswith(receptacle_path + "/"):
        raise ValueError(
            "task.place.target_support_prim_path 必须位于 target receptacle 下: "
            f"support={support_path!r}, receptacle={receptacle_path!r}"
        )

    placement_region = raw_place.get("placement_region")
    if placement_region is not None and not isinstance(placement_region, dict):
        raise ValueError("task.place.placement_region 必须是对象")
    place_pose = raw_place.get("place_pose_world")
    if place_pose is not None and not isinstance(place_pose, dict):
        raise ValueError("task.place.place_pose_world 必须是对象")

    return {
        "configured": True,
        "place_enabled": True,
        "runtime_validation_required": validation_required,
        "target_receptacle_id": (
            raw_place.get("target_receptacle_id")
            or raw_task.get("target_receptacle_id")
        ),
        "target_receptacle_prim_path": receptacle_path,
        "target_support_prim_path": support_path,
        "support_expected_static": bool(
            raw_place.get("support_expected_static", False)
        ),
        "placement_region": placement_region,
        "place_pose_world": place_pose,
    }


def inspect_task_receptacle_support_stage(
    stage: Any,
    raw_task: dict[str, Any],
    *,
    source: str,
    tolerance_m: float = 5.0e-6,
) -> dict[str, Any]:
    """检查组合 stage 中的目标支撑 Mesh、CollisionAPI、包围盒和任务代理。

    USD 组合后的单精度 xform/bbox 计算会产生数微米量级的舍入误差。这里仍以
    5 µm 严格拒绝真实几何漂移，但不会把等价代理误判为过期配置。
    """

    settings = resolve_task_receptacle_support_settings(raw_task)
    if not settings["configured"]:
        return {
            **settings,
            "source": source,
            "geometry_verified": None,
        }

    from pxr import Usd, UsdGeom, UsdPhysics

    receptacle_path = str(settings["target_receptacle_prim_path"])
    support_path = str(settings["target_support_prim_path"])
    receptacle_prim = stage.GetPrimAtPath(receptacle_path)
    support_prim = stage.GetPrimAtPath(support_path)
    if not receptacle_prim.IsValid() or not receptacle_prim.IsActive():
        raise RuntimeError(
            f"task receptacle prim is unavailable or inactive: {receptacle_path}"
        )
    if not support_prim.IsValid() or not support_prim.IsActive():
        raise RuntimeError(
            f"task receptacle support prim is unavailable or inactive: {support_path}"
        )

    mesh_count = 0
    collision_api_count = 0
    collision_enabled_count = 0
    collision_prim_paths: list[str] = []
    for prim in Usd.PrimRange(support_prim):
        mesh_count += int(prim.IsA(UsdGeom.Mesh))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
            collision_prim_paths.append(str(prim.GetPath()))
            enabled_value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            collision_enabled_count += int(
                True if enabled_value is None else bool(enabled_value)
            )
    from .task_scene_pose import inspect_episode_static_support_body_mode

    body_mode_report = inspect_episode_static_support_body_mode(receptacle_prim)

    if mesh_count == 0:
        raise RuntimeError(f"task receptacle support has no Mesh: {support_path}")
    if collision_api_count == 0 or collision_enabled_count == 0:
        raise RuntimeError(
            f"task receptacle support has no enabled CollisionAPI: {support_path}"
        )
    if (
        settings["support_expected_static"]
        and body_mode_report["episode_static_support_verified"] is not True
    ):
        raise RuntimeError(
            "task receptacle support was expected to be episode-static, but has "
            f"an unsupported rigid-body layout: {body_mode_report}"
        )

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    aligned_box = bbox_cache.ComputeWorldBound(support_prim).ComputeAlignedBox()
    bbox_min = tuple(float(value) for value in aligned_box.GetMin())
    bbox_max = tuple(float(value) for value in aligned_box.GetMax())
    bbox_center = tuple(
        (bbox_min[index] + bbox_max[index]) * 0.5 for index in range(3)
    )
    bbox_dims = tuple(bbox_max[index] - bbox_min[index] for index in range(3))
    if not all(math.isfinite(value) for value in (*bbox_min, *bbox_max)):
        raise RuntimeError(f"task receptacle support bbox is not finite: {support_path}")
    if any(value <= 0.0 for value in bbox_dims):
        raise RuntimeError(
            f"task receptacle support bbox has non-positive dimensions: {bbox_dims}"
        )

    region_report = _validate_placement_region(
        settings.get("placement_region"),
        settings.get("place_pose_world"),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        tolerance_m=float(tolerance_m),
    )
    proxy_report = _validate_task_support_proxy(
        raw_task,
        support_path=support_path,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        tolerance_m=float(tolerance_m),
    )
    root_layer = stage.GetRootLayer() if hasattr(stage, "GetRootLayer") else None
    return {
        **settings,
        "source": source,
        "stage_root_layer": (
            str(getattr(root_layer, "identifier", "")) if root_layer is not None else None
        ),
        "receptacle_active": True,
        "support_active": True,
        "support_loaded": bool(support_prim.IsLoaded()),
        "mesh_count": mesh_count,
        "collision_api_count": collision_api_count,
        "collision_enabled_count": collision_enabled_count,
        "collision_prim_paths": collision_prim_paths,
        **body_mode_report,
        "world_bbox_min_xyz": list(bbox_min),
        "world_bbox_max_xyz": list(bbox_max),
        "world_bbox_center_xyz": list(bbox_center),
        "world_bbox_dims_xyz": list(bbox_dims),
        "support_surface_z": bbox_max[2],
        "placement_region_report": region_report,
        "task_support_proxy_report": proxy_report,
        "geometry_verified": True,
    }


def inspect_task_receptacle_support_usd(
    scene_usd: str | Path,
    raw_task: dict[str, Any],
    *,
    source: str = "source_scene_usd",
) -> dict[str, Any]:
    """打开源场景并执行同一套 receptacle 支撑检查。"""

    from pxr import Usd

    scene_path = Path(scene_usd).expanduser().resolve()
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"failed to open scene USD: {scene_path}")
    from .task_scene_pose import apply_task_receptacle_pose

    scene_pose_report = apply_task_receptacle_pose(stage, raw_task)
    report = inspect_task_receptacle_support_stage(
        stage,
        raw_task,
        source=source,
    )
    report["task_scene_pose_report"] = scene_pose_report
    return report


def _validate_placement_region(
    placement_region: Any,
    place_pose_world: Any,
    *,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    tolerance_m: float,
) -> dict[str, Any]:
    """确保 world-frame 安全区域和目标中心位于实际支撑包围盒内。"""

    if placement_region is None:
        return {"configured": False, "verified": None}
    if placement_region.get("frame") != "world":
        raise RuntimeError("task.place.placement_region.frame 必须是 world")
    try:
        x_min = float(placement_region["x_min"])
        x_max = float(placement_region["x_max"])
        y_min = float(placement_region["y_min"])
        y_max = float(placement_region["y_max"])
        z_surface = float(placement_region["z_surface"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("task.place.placement_region 缺少有限 world bounds") from exc
    values = (x_min, x_max, y_min, y_max, z_surface)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("task.place.placement_region 包含非有限值")
    if not x_min < x_max or not y_min < y_max:
        raise RuntimeError("task.place.placement_region 边界顺序无效")
    region_inside_bbox = bool(
        x_min >= bbox_min[0] - tolerance_m
        and x_max <= bbox_max[0] + tolerance_m
        and y_min >= bbox_min[1] - tolerance_m
        and y_max <= bbox_max[1] + tolerance_m
    )
    surface_matches_bbox = abs(z_surface - bbox_max[2]) <= tolerance_m
    if not region_inside_bbox or not surface_matches_bbox:
        raise RuntimeError(
            "task placement region does not match the composed support bbox: "
            f"inside={region_inside_bbox}, surface_matches={surface_matches_bbox}"
        )

    pose_inside_region = None
    pose_above_support = None
    if place_pose_world is not None:
        try:
            pose_x = float(place_pose_world["x"])
            pose_y = float(place_pose_world["y"])
            pose_z = float(place_pose_world["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("task.place.place_pose_world 缺少有限 xyz") from exc
        if not all(math.isfinite(value) for value in (pose_x, pose_y, pose_z)):
            raise RuntimeError("task.place.place_pose_world 包含非有限 xyz")
        pose_inside_region = bool(
            x_min <= pose_x <= x_max and y_min <= pose_y <= y_max
        )
        pose_above_support = pose_z > z_surface
        if not pose_inside_region or not pose_above_support:
            raise RuntimeError(
                "task place pose is not inside/above the configured support region: "
                f"inside={pose_inside_region}, above={pose_above_support}"
            )
    return {
        "configured": True,
        "frame": "world",
        "region_inside_support_bbox": region_inside_bbox,
        "surface_matches_support_bbox": surface_matches_bbox,
        "place_pose_inside_region": pose_inside_region,
        "place_pose_above_support": pose_above_support,
        "verified": True,
    }


def _validate_task_support_proxy(
    raw_task: dict[str, Any],
    *,
    support_path: str,
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    tolerance_m: float,
) -> dict[str, Any]:
    """核对任务给 CuRobo 的 USD bbox proxy 没有相对真实支撑体漂移。"""

    raw_place = raw_task.get("place")
    raw_place = raw_place if isinstance(raw_place, dict) else {}
    collision = raw_place.get("curobo_world_collision")
    if not isinstance(collision, dict):
        return {"configured": False, "verified": None}
    required = bool(collision.get("required", False))
    cuboids = collision.get("cuboids_world")
    if not isinstance(cuboids, list):
        cuboids = []
    matches = [
        item
        for item in cuboids
        if isinstance(item, dict) and item.get("source_prim_path") == support_path
    ]
    if required and not matches:
        raise RuntimeError(
            f"required CuRobo support proxy is missing for {support_path}"
        )
    if not matches:
        return {"configured": False, "required": required, "verified": None}

    proxy = matches[0]
    source = proxy.get("source")
    source = source if isinstance(source, dict) else {}
    expected_min = source.get("world_bbox_min_xyz")
    expected_max = source.get("world_bbox_max_xyz")
    expected_surface = source.get("support_surface_z")
    geometry_declared = bool(expected_min is not None and expected_max is not None)
    geometry_matches = None
    max_abs_error_m = None
    if geometry_declared:
        try:
            expected_values = tuple(float(value) for value in (*expected_min, *expected_max))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("CuRobo support proxy bbox source must contain numeric values") from exc
        if len(expected_values) != 6 or not all(
            math.isfinite(value) for value in expected_values
        ):
            raise RuntimeError("CuRobo support proxy bbox source must contain six finite values")
        actual_values = (*bbox_min, *bbox_max)
        max_abs_error_m = max(
            abs(actual - expected)
            for actual, expected in zip(actual_values, expected_values)
        )
        surface_error = (
            0.0
            if expected_surface is None
            else abs(float(expected_surface) - bbox_max[2])
        )
        max_abs_error_m = max(max_abs_error_m, surface_error)
        geometry_matches = max_abs_error_m <= tolerance_m
        if not geometry_matches:
            raise RuntimeError(
                "CuRobo support proxy geometry drifted from composed USD support: "
                f"max_abs_error_m={max_abs_error_m}"
            )
    return {
        "configured": True,
        "required": required,
        "matching_proxy_count": len(matches),
        "proxy_name": proxy.get("name"),
        "source_type": source.get("type"),
        "geometry_declared": geometry_declared,
        "geometry_matches_composed_support": geometry_matches,
        "max_abs_error_m": max_abs_error_m,
        "verified": bool(not required or matches) and geometry_matches is not False,
    }
