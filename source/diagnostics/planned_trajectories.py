"""在当前 Isaac USD stage 中绘制导航和机械臂规划轨迹。"""

from __future__ import annotations

import math
import re
import traceback
from typing import Any


_ROOT_PRIM_PATH = "/World/PlannedTrajectories"
_DEBUG_DRAW_EXTENSION_NAME = "isaacsim.util.debug_draw"
_VELOCITY_DEBUG_DRAW_INTERFACE: Any | None = None
_VELOCITY_DEBUG_DRAW_EXTENSION_ENABLED_BY_US = False


def navigation_path_points(plan: Any) -> tuple[tuple[float, float, float], ...]:
    """从 NavPlan 中提取优先使用三维信息的世界坐标路径。"""

    metadata = getattr(plan, "metadata", {})
    raw_path = None
    if isinstance(metadata, dict):
        raw_path = metadata.get("visualization_path_3d") or metadata.get("path_3d")
    points = _xyz_points(raw_path)
    if len(points) >= 2:
        return points

    goal = getattr(plan, "goal", None)
    goal_z = float(getattr(goal, "z", 0.0) or 0.0)
    return tuple(
        (float(point[0]), float(point[1]), goal_z)
        for point in getattr(plan, "waypoints", ())
        if isinstance(point, (list, tuple)) and len(point) >= 2
    )


def manipulation_tcp_segments(plan: Any) -> tuple[dict[str, Any], ...]:
    """从 cuRobo 分段计划中提取已有 FK 计算的 TCP 世界坐标轨迹。"""

    metadata = getattr(plan, "metadata", {})
    raw_segments = metadata.get("segments") if isinstance(metadata, dict) else None
    if not isinstance(raw_segments, (list, tuple)):
        return ()

    reports: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, dict) or segment.get("type") != "motion":
            continue
        trajectory = segment.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        points = _xyz_points(trajectory.get("tcp_position_world"))
        if len(points) < 2:
            continue
        reports.append(
            {
                "index": index,
                "name": str(segment.get("name") or f"motion_{index:02d}"),
                "points": points,
            }
        )
    return tuple(reports)


def draw_navigation_plan(plan: Any, *, phase: str) -> dict[str, Any]:
    """绘制一段 PCT/导航世界坐标路径，不修改任何物理属性。"""

    points = navigation_path_points(plan)
    if len(points) < 2:
        return {
            "available": False,
            "type": "navigation",
            "phase": phase,
            "reason": "navigation_path_unavailable",
        }
    metadata = getattr(plan, "metadata", {})
    stair_centerline = bool(
        isinstance(metadata, dict)
        and metadata.get("planner") == "pct_stair_centerline"
    )
    colors = {
        "pick": (0.1, 0.8, 1.0),
        "place": (1.0, 0.75, 0.1),
    }
    color = (0.10, 0.80, 1.0) if stair_centerline else colors.get(
        phase,
        (0.8, 0.8, 0.8),
    )
    lifted = tuple(
        (x, y, z + (0.08 if stair_centerline else 0.12))
        for x, y, z in points
    )
    group_name = (
        "stair_locomotion_centerline"
        if stair_centerline
        else _safe_name(phase)
    )
    return _draw_curve_group(
        group_path=f"{_ROOT_PRIM_PATH}/navigation/{group_name}",
        curves=(("centerline" if stair_centerline else "path", lifted),),
        color=color,
        width=0.050 if stair_centerline else 0.035,
        marker_radius=0.040 if stair_centerline else 0.055,
        marker_limit=72 if stair_centerline else 48,
        report_type="navigation",
        phase="stair_locomotion" if stair_centerline else phase,
    )


def draw_manipulation_plan(plan: Any, *, phase: str) -> dict[str, Any]:
    """绘制 cuRobo 已计算好的 TCP 世界坐标轨迹，不在运行时重复做 FK。"""

    segments = manipulation_tcp_segments(plan)
    if not segments:
        return {
            "available": False,
            "type": "manipulation",
            "phase": phase,
            "reason": "tcp_world_trajectory_unavailable",
        }
    colors = {
        "pick": (1.0, 0.2, 0.8),
        "place": (0.2, 1.0, 0.35),
    }
    curves = tuple(
        (f"{segment['index']:02d}_{_safe_name(segment['name'])}", segment["points"])
        for segment in segments
    )
    return _draw_curve_group(
        group_path=f"{_ROOT_PRIM_PATH}/manipulation/{_safe_name(phase)}",
        curves=curves,
        color=colors.get(phase, (0.9, 0.3, 0.9)),
        width=0.018,
        marker_radius=0.025,
        marker_limit=32,
        report_type="manipulation",
        phase=phase,
    )


def velocity_command_guide_geometry(
    robot_root_pose: Any,
    base_velocity: Any,
) -> dict[str, Any]:
    """把机体系速度命令转换为机器人上方的世界坐标箭头。"""

    if not isinstance(robot_root_pose, (list, tuple)) or len(robot_root_pose) < 7:
        raise ValueError("速度 guide 需要 xyz+wxyz root pose。")
    if not isinstance(base_velocity, (list, tuple)) or len(base_velocity) < 3:
        raise ValueError("速度 guide 需要 vx/vy/wz 三维命令。")
    x, y, z = (float(value) for value in robot_root_pose[:3])
    qw, qx, qy, qz = (float(value) for value in robot_root_pose[3:7])
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    vx, vy, wz = (float(value) for value in base_velocity[:3])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_vx = cos_yaw * vx - sin_yaw * vy
    world_vy = sin_yaw * vx + cos_yaw * vy
    origin = (x, y, z + 0.48)
    linear_scale = 1.60
    linear_end = (
        origin[0] + world_vx * linear_scale,
        origin[1] + world_vy * linear_scale,
        origin[2],
    )
    linear_visible = math.hypot(vx, vy) > 1.0e-4
    linear_curves = _arrow_curve_segments(origin, linear_end)

    angular_radius = 0.32
    angular_span = min(max(wz * 1.40, -1.10), 1.10)
    angular_visible = abs(wz) > 1.0e-4
    angular_points: tuple[tuple[float, float, float], ...]
    if angular_visible:
        sample_count = max(5, int(math.ceil(abs(angular_span) / 0.08)) + 1)
        angular_points = tuple(
            (
                origin[0]
                + angular_radius
                * math.cos(yaw + angular_span * index / (sample_count - 1)),
                origin[1]
                + angular_radius
                * math.sin(yaw + angular_span * index / (sample_count - 1)),
                origin[2] + 0.04,
            )
            for index in range(sample_count)
        )
        end_angle = yaw + angular_span
        direction_sign = 1.0 if angular_span > 0.0 else -1.0
        tangent = (
            -math.sin(end_angle) * direction_sign,
            math.cos(end_angle) * direction_sign,
            0.0,
        )
        angular_head = _arrow_head_segments(
            angular_points[-1],
            tangent,
            head_length=0.09,
        )
    else:
        angular_points = (origin, origin)
        angular_head = ((origin, origin), (origin, origin))

    return {
        "origin": origin,
        "command_body": (vx, vy, wz),
        "linear_velocity_world": (world_vx, world_vy),
        "linear_visible": linear_visible,
        "linear_curves": linear_curves,
        "angular_visible": angular_visible,
        "angular_curves": (angular_points, *angular_head),
    }


def draw_velocity_command(
    *,
    robot_root_pose: Any,
    base_velocity: Any,
    source: str,
) -> dict[str, Any]:
    """优先用 Isaac debug-draw 逐 tick 更新速度箭头，失败时回退 USD。"""

    draw_step = "build_geometry"
    try:
        geometry = velocity_command_guide_geometry(robot_root_pose, base_velocity)
    except Exception as exc:
        return {
            "available": False,
            "type": "stair_velocity_command",
            "reason": "geometry_build_failed",
            "draw_step": draw_step,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    try:
        return _draw_velocity_command_with_debug_draw(
            geometry=geometry,
            source=source,
        )
    except Exception as debug_draw_exc:
        debug_draw_traceback = traceback.format_exc()
        fallback = _draw_velocity_command_with_usd(
            geometry=geometry,
            source=source,
        )
        return {
            **fallback,
            "renderer": "usd_basis_curves_fallback",
            "fallback_used": True,
            "debug_draw_error": (
                f"{type(debug_draw_exc).__name__}: {debug_draw_exc}"
            ),
            "debug_draw_traceback": debug_draw_traceback,
        }


def _acquire_velocity_debug_draw_interface() -> Any:
    """按需启用官方扩展，并缓存本进程唯一的 debug-draw 接口。"""

    global _VELOCITY_DEBUG_DRAW_INTERFACE
    global _VELOCITY_DEBUG_DRAW_EXTENSION_ENABLED_BY_US
    if _VELOCITY_DEBUG_DRAW_INTERFACE is not None:
        return _VELOCITY_DEBUG_DRAW_INTERFACE

    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    was_enabled = extension_manager.is_extension_enabled(
        _DEBUG_DRAW_EXTENSION_NAME
    )
    if not was_enabled:
        extension_manager.set_extension_enabled_immediate(
            _DEBUG_DRAW_EXTENSION_NAME,
            True,
        )
    if not extension_manager.is_extension_enabled(_DEBUG_DRAW_EXTENSION_NAME):
        raise RuntimeError(
            f"无法启用 Isaac debug-draw 扩展: {_DEBUG_DRAW_EXTENSION_NAME}"
        )

    from isaacsim.util.debug_draw import _debug_draw

    interface = _debug_draw.acquire_debug_draw_interface()
    if interface is None:
        raise RuntimeError("Isaac debug-draw interface unavailable")
    _VELOCITY_DEBUG_DRAW_INTERFACE = interface
    _VELOCITY_DEBUG_DRAW_EXTENSION_ENABLED_BY_US = not was_enabled
    return interface


def _velocity_debug_draw_lines(
    geometry: dict[str, Any],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float, float], ...],
    tuple[float, ...],
]:
    """把箭头折线展开为官方 draw_lines 所需的逐线段数组。"""

    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    widths: list[float] = []

    def append_curves(
        curves: Any,
        *,
        color: tuple[float, float, float, float],
        width: float,
    ) -> None:
        for curve in curves:
            for start, end in zip(curve, curve[1:]):
                starts.append(tuple(float(value) for value in start[:3]))
                ends.append(tuple(float(value) for value in end[:3]))
                colors.append(color)
                widths.append(float(width))

    if geometry["linear_visible"]:
        append_curves(
            geometry["linear_curves"],
            color=(0.20, 1.0, 0.25, 1.0),
            width=5.0,
        )
    if geometry["angular_visible"]:
        append_curves(
            geometry["angular_curves"],
            color=(1.0, 0.35, 0.05, 1.0),
            width=4.0,
        )
    return tuple(starts), tuple(ends), tuple(colors), tuple(widths)


def _draw_velocity_command_with_debug_draw(
    *,
    geometry: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """使用 Isaac 官方临时线缓冲区绘制一帧速度命令。"""

    draw = _acquire_velocity_debug_draw_interface()
    starts, ends, colors, widths = _velocity_debug_draw_lines(geometry)
    # 该模式独占临时 debug line 缓冲区；静态 PCT 路径使用 USD，不受影响。
    draw.clear_lines()
    if starts:
        draw.draw_lines(list(starts), list(ends), list(colors), list(widths))
    command = geometry["command_body"]
    return {
        "available": True,
        "type": "stair_velocity_command",
        "renderer": _DEBUG_DRAW_EXTENSION_NAME,
        "fallback_used": False,
        "extension_enabled_by_pipeline": (
            _VELOCITY_DEBUG_DRAW_EXTENSION_ENABLED_BY_US
        ),
        "debug_draw_line_count": len(starts),
        "debug_draw_reported_line_count": int(draw.get_num_lines()),
        "debug_draw_global_line_buffer_owned": True,
        "command_source": str(source),
        "command_body_vx_mps": float(command[0]),
        "command_body_vy_mps": float(command[1]),
        "command_body_wz_rps": float(command[2]),
        "linear_color_rgba": (0.20, 1.0, 0.25, 1.0),
        "angular_color_rgba": (1.0, 0.35, 0.05, 1.0),
        "updated_per_control_tick": True,
        "physics_unchanged": True,
    }


def _draw_velocity_command_with_usd(
    *,
    geometry: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """在官方 debug-draw 不可用时保留原有 USD 曲线降级路径。"""

    draw_step = "import_usd"
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        draw_step = "get_stage"
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        group_path = f"{_ROOT_PRIM_PATH}/navigation/stair_locomotion_command"
        stage.DefinePrim(Sdf.Path(_ROOT_PRIM_PATH), "Xform")
        group = stage.DefinePrim(Sdf.Path(group_path), "Xform")
        command = geometry["command_body"]
        group.SetCustomDataByKey("commandBodyVxMps", float(command[0]))
        group.SetCustomDataByKey("commandBodyVyMps", float(command[1]))
        group.SetCustomDataByKey("commandBodyWzRps", float(command[2]))
        group.SetCustomDataByKey("commandSource", str(source))

        draw_step = "update_linear_arrow"
        _update_dynamic_curves(
            stage=stage,
            group_path=f"{group_path}/linear_velocity",
            curve_names=("shaft", "head_left", "head_right"),
            curves=geometry["linear_curves"],
            color=(0.20, 1.0, 0.25),
            width=0.030,
            visible=bool(geometry["linear_visible"]),
            gf=Gf,
            sdf=Sdf,
            usd_geom=UsdGeom,
        )
        draw_step = "update_angular_arrow"
        _update_dynamic_curves(
            stage=stage,
            group_path=f"{group_path}/angular_velocity",
            curve_names=("arc", "head_left", "head_right"),
            curves=geometry["angular_curves"],
            color=(1.0, 0.35, 0.05),
            width=0.024,
            visible=bool(geometry["angular_visible"]),
            gf=Gf,
            sdf=Sdf,
            usd_geom=UsdGeom,
        )
        return {
            "available": True,
            "type": "stair_velocity_command",
            "root_prim_path": group_path,
            "command_source": str(source),
            "command_body_vx_mps": float(command[0]),
            "command_body_vy_mps": float(command[1]),
            "command_body_wz_rps": float(command[2]),
            "linear_color_rgb": (0.20, 1.0, 0.25),
            "angular_color_rgb": (1.0, 0.35, 0.05),
            "updated_per_control_tick": True,
            "physics_unchanged": True,
        }
    except Exception as exc:
        return {
            "available": False,
            "type": "stair_velocity_command",
            "reason": "usd_draw_failed",
            "draw_step": draw_step,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _arrow_curve_segments(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dz = float(end[2]) - float(start[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1.0e-8:
        return ((start, start), (start, start), (start, start))
    direction = (dx / length, dy / length, dz / length)
    head = _arrow_head_segments(
        end,
        direction,
        head_length=min(0.13, max(0.06, length * 0.32)),
    )
    return ((start, end), *head)


def _arrow_head_segments(
    end: tuple[float, float, float],
    direction: tuple[float, float, float],
    *,
    head_length: float,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float]],
]:
    dx, dy, dz = (float(value) for value in direction)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1.0e-8:
        return ((end, end), (end, end))
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    side_x, side_y = -dy, dx
    wing = head_length * 0.55
    base = (
        float(end[0]) - dx * head_length,
        float(end[1]) - dy * head_length,
        float(end[2]) - dz * head_length,
    )
    left = (base[0] + side_x * wing, base[1] + side_y * wing, base[2])
    right = (base[0] - side_x * wing, base[1] - side_y * wing, base[2])
    return ((end, left), (end, right))


def _update_dynamic_curves(
    *,
    stage: Any,
    group_path: str,
    curve_names: tuple[str, ...],
    curves: tuple[tuple[tuple[float, float, float], ...], ...],
    color: tuple[float, float, float],
    width: float,
    visible: bool,
    gf: Any,
    sdf: Any,
    usd_geom: Any,
) -> None:
    stage.DefinePrim(sdf.Path(group_path), "Xform")
    for name, points in zip(curve_names, curves):
        curve_path = f"{group_path}/{_safe_name(name)}"
        curve = usd_geom.BasisCurves.Define(stage, sdf.Path(curve_path))
        curve.CreateTypeAttr(usd_geom.Tokens.linear)
        curve.CreateWrapAttr(usd_geom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr().Set([len(points)])
        curve.CreatePointsAttr().Set([gf.Vec3f(*point) for point in points])
        curve.CreateWidthsAttr().Set([float(width)])
        curve.SetWidthsInterpolation(usd_geom.Tokens.constant)
        curve.CreateDisplayColorAttr().Set([gf.Vec3f(*color)])
        imageable = usd_geom.Imageable(curve.GetPrim())
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _draw_curve_group(
    *,
    group_path: str,
    curves: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...],
    color: tuple[float, float, float],
    width: float,
    marker_radius: float,
    marker_limit: int,
    report_type: str,
    phase: str,
) -> dict[str, Any]:
    draw_step = "import_usd"
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        draw_step = "get_stage"
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        draw_step = "replace_group"
        if stage.GetPrimAtPath(group_path).IsValid():
            stage.RemovePrim(group_path)
        stage.DefinePrim(Sdf.Path(_ROOT_PRIM_PATH), "Xform")
        stage.DefinePrim(Sdf.Path(group_path), "Xform")

        curve_reports: list[dict[str, Any]] = []
        for curve_index, (name, points) in enumerate(curves):
            draw_step = f"define_curve_root:{curve_index}"
            curve_root = f"{group_path}/curve_{curve_index:02d}_{_safe_name(name)}"
            stage.DefinePrim(Sdf.Path(curve_root), "Xform")
            curve_path = f"{curve_root}/curve"
            curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path(curve_path))
            draw_step = f"configure_curve:{curve_index}"
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            curve.CreateWidthsAttr([width] * len(points))
            curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])

            marker_stride = max(1, len(points) // max(1, marker_limit))
            marker_indices = list(range(0, len(points), marker_stride))
            if marker_indices[-1] != len(points) - 1:
                marker_indices.append(len(points) - 1)
            for marker_index in marker_indices:
                draw_step = f"define_marker:{curve_index}:{marker_index}"
                point = points[marker_index]
                marker_path = f"{curve_root}/wp_{marker_index:04d}"
                marker = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_path))
                marker.CreateRadiusAttr(marker_radius)
                marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(*point)
                )
            curve_reports.append(
                {
                    "name": name,
                    "curve_prim_path": curve_path,
                    "point_count": len(points),
                    "marker_count": len(marker_indices),
                }
            )
        return {
            "available": True,
            "type": report_type,
            "phase": phase,
            "root_prim_path": group_path,
            "color_rgb": color,
            "curves": curve_reports,
            "physics_unchanged": True,
        }
    except Exception as exc:
        return {
            "available": False,
            "type": report_type,
            "phase": phase,
            "reason": "usd_draw_failed",
            "draw_step": draw_step,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _xyz_points(raw_points: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw_points, (list, tuple)):
        return ()
    return tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in raw_points
        if isinstance(point, (list, tuple)) and len(point) >= 3
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    return cleaned or "trajectory"
