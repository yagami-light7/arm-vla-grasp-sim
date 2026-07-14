"""随机化区域的非物理 USD guide 可视化。"""

from __future__ import annotations

import math
from typing import Any


def _xy_range(
    randomization: dict[str, Any],
    key: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    report = randomization.get(key)
    if not isinstance(report, dict):
        return None
    x_range = report.get("x_range_m")
    y_range = report.get("y_range_m")
    if not isinstance(x_range, (list, tuple)) or len(x_range) != 2:
        return None
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        return None
    try:
        values = tuple(float(value) for value in (*x_range, *y_range))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x_min, x_max, y_min, y_max = values
    if x_min > x_max or y_min > y_max:
        return None
    return (x_min, x_max), (y_min, y_max)


def _forward_sector_spec(
    randomization: dict[str, Any],
) -> dict[str, Any] | None:
    """解析 episode 实际使用的机器人前向扇区。"""

    if randomization.get("mode") != "robot_forward_sector_v1":
        return None
    config = randomization.get("forward_sector")
    sample = randomization.get("sample")
    if not isinstance(config, dict) or not isinstance(sample, dict):
        return None
    try:
        robot_xyz = tuple(float(value) for value in config["robot_translate_xyz"])
        robot_yaw = float(sample["robot_yaw_rad"])
        half_angle = math.radians(float(config["sector_half_angle_deg"]))
        cola_radius = tuple(float(value) for value in config["cola_radius_range_m"])
        mat_radius = tuple(float(value) for value in config["mat_radius_range_m"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(robot_xyz) != 3
        or len(cola_radius) != 2
        or len(mat_radius) != 2
        or not all(
            math.isfinite(value)
            for value in (*robot_xyz, robot_yaw, half_angle, *cola_radius, *mat_radius)
        )
    ):
        return None
    return {
        "origin_xyz": robot_xyz,
        "robot_yaw_rad": robot_yaw,
        "sector_half_angle_rad": half_angle,
        "cola_radius_range_m": cola_radius,
        "mat_radius_range_m": mat_radius,
    }


def randomization_debug_spec(raw_task: dict[str, Any]) -> dict[str, Any]:
    """提取与 episode 采样结果一致的可视化描述。"""

    randomization = dict(raw_task.get("randomization") or {})
    pick_pose = dict((raw_task.get("pick") or {}).get("object_pose_world") or {})
    place_pose = dict((raw_task.get("place") or {}).get("place_pose_world") or {})
    return {
        "enabled": True,
        "root_prim_path": "/World/RandomizationDebug",
        # baseline 使用默认 purpose；guide purpose 可能被 viewport 默认过滤。
        "usd_purpose": "default",
        "physics_enabled": False,
        "collision_enabled": False,
        "forward_sector": _forward_sector_spec(randomization),
        "pick": {
            "xy_range": _xy_range(randomization, "object_xy_randomization"),
            "pose_world": pick_pose,
            "color_rgb": (0.1, 1.0, 0.2),
        },
        "place": {
            "xy_range": _xy_range(randomization, "place_xy_randomization"),
            "pose_world": place_pose,
            "color_rgb": (0.1, 0.4, 1.0),
        },
    }


def _sector_outline_points(
    *,
    origin_xy: tuple[float, float],
    yaw_rad: float,
    half_angle_rad: float,
    radius_range_m: tuple[float, float],
    z: float,
) -> list[tuple[float, float, float]]:
    """生成闭合扇环轮廓，用于显示实际可采样区域。"""

    inner_radius, outer_radius = radius_range_m
    angles = [
        -half_angle_rad + 2.0 * half_angle_rad * index / 24.0
        for index in range(25)
    ]

    def _point(radius: float, angle: float) -> tuple[float, float, float]:
        world_angle = yaw_rad + angle
        return (
            origin_xy[0] + radius * math.cos(world_angle),
            origin_xy[1] + radius * math.sin(world_angle),
            z,
        )

    inner = [_point(inner_radius, angle) for angle in angles]
    outer = [_point(outer_radius, angle) for angle in reversed(angles)]
    return [*inner, *outer, inner[0]]


def _create_forward_sector_guides(
    stage: Any,
    spec: dict[str, Any] | None,
    *,
    z: float,
) -> dict[str, Any]:
    """创建可乐与地垫的前向扇环 guide。"""

    if spec is None:
        return {"created": False, "reason": "forward_sector_not_configured"}
    from pxr import UsdGeom

    root_path = "/World/RandomizationDebug/ForwardSector"
    UsdGeom.Xform.Define(stage, root_path)
    origin = tuple(float(value) for value in spec["origin_xyz"][:2])
    yaw = float(spec["robot_yaw_rad"])
    half_angle = float(spec["sector_half_angle_rad"])
    paths: dict[str, str] = {}
    for name, radius_key, color in (
        ("Cola", "cola_radius_range_m", (0.1, 1.0, 0.2)),
        ("Mat", "mat_radius_range_m", (0.1, 0.4, 1.0)),
    ):
        path = f"{root_path}/{name}Region"
        _create_curve(
            stage,
            path,
            _sector_outline_points(
                origin_xy=origin,
                yaw_rad=yaw,
                half_angle_rad=half_angle,
                radius_range_m=tuple(spec[radius_key]),
                z=z,
            ),
            color,
            width=0.016,
        )
        paths[name.lower()] = path
    return {
        "created": True,
        "root_prim_path": root_path,
        "region_prim_paths": paths,
        **spec,
        "physics_enabled": False,
        "collision_enabled": False,
    }


def _create_curve(
    stage: Any,
    prim_path: str,
    points: list[tuple[float, float, float]],
    color: tuple[float, float, float],
    *,
    width: float,
) -> None:
    from pxr import Gf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr([float(width)] * len(points))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _create_group(stage: Any, *, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    root_path = f"/World/RandomizationDebug/{name}"
    UsdGeom.Xform.Define(stage, root_path)
    pose = spec["pose_world"]
    try:
        x, y, z = (float(pose[key]) for key in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError) as exc:
        return {"created": False, "root_prim_path": root_path, "failure_reason": str(exc)}

    color = spec["color_rgb"]
    marker_height = z + 0.12
    sphere_path = f"{root_path}/SampledPosition"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr(0.035)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, marker_height))
    _create_curve(
        stage,
        f"{root_path}/PositionStem",
        [(x, y, z + 0.01), (x, y, marker_height)],
        color,
        width=0.012,
    )

    range_path = None
    xy_range = spec["xy_range"]
    if xy_range is not None:
        (x_min, x_max), (y_min, y_max) = xy_range
        range_path = f"{root_path}/XYRange"
        range_z = z + 0.02
        _create_curve(
            stage,
            range_path,
            [
                (x_min, y_min, range_z),
                (x_max, y_min, range_z),
                (x_max, y_max, range_z),
                (x_min, y_max, range_z),
                (x_min, y_min, range_z),
            ],
            color,
            width=0.016,
        )
    return {
        "created": True,
        "root_prim_path": root_path,
        "sampled_position_prim_path": sphere_path,
        "xy_range_prim_path": range_path,
        "pose_world": {"x": x, "y": y, "z": z},
        "xy_range": xy_range,
        "color_rgb": color,
        "physics_enabled": False,
        "collision_enabled": False,
    }


def create_randomization_debug(stage: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
    """创建默认可见的调试 prim；不附加碰撞、刚体或质量 API。"""

    from pxr import UsdGeom

    spec = randomization_debug_spec(raw_task)
    root_path = spec["root_prim_path"]
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    guide_z = min(
        float(spec["pick"]["pose_world"].get("z", 0.0)),
        float(spec["place"]["pose_world"].get("z", 0.0)),
    ) + 0.02
    report = {
        **spec,
        "forward_sector": _create_forward_sector_guides(
            stage,
            spec["forward_sector"],
            z=guide_z,
        ),
        "pick": _create_group(stage, name="Pick", spec=spec["pick"]),
        "place": _create_group(stage, name="Place", spec=spec["place"]),
    }
    print(
        "[randomization-debug] "
        f"pick_range={report['pick'].get('xy_range')} "
        f"pick={report['pick'].get('pose_world')} "
        f"place_range={report['place'].get('xy_range')} "
        f"place={report['place'].get('pose_world')} "
        "legend=green:pick,blue:place",
        flush=True,
    )
    return report
