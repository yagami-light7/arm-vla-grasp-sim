"""在当前 Isaac USD stage 中绘制导航和机械臂规划轨迹。"""

from __future__ import annotations

import re
from typing import Any


_ROOT_PRIM_PATH = "/World/PlannedTrajectories"


def navigation_path_points(plan: Any) -> tuple[tuple[float, float, float], ...]:
    """从 NavPlan 中提取优先使用三维信息的世界坐标路径。"""

    metadata = getattr(plan, "metadata", {})
    raw_path = metadata.get("path_3d") if isinstance(metadata, dict) else None
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
    colors = {
        "pick": (0.1, 0.8, 1.0),
        "place": (1.0, 0.75, 0.1),
    }
    lifted = tuple((x, y, z + 0.12) for x, y, z in points)
    return _draw_curve_group(
        group_path=f"{_ROOT_PRIM_PATH}/navigation/{_safe_name(phase)}",
        curves=(("path", lifted),),
        color=colors.get(phase, (0.8, 0.8, 0.8)),
        width=0.035,
        marker_radius=0.055,
        marker_limit=48,
        report_type="navigation",
        phase=phase,
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
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("stage_unavailable")
        if stage.GetPrimAtPath(group_path).IsValid():
            stage.RemovePrim(group_path)
        UsdGeom.Xform.Define(stage, _ROOT_PRIM_PATH)
        parent_path = group_path.rsplit("/", 1)[0]
        UsdGeom.Xform.Define(stage, parent_path)
        UsdGeom.Xform.Define(stage, group_path)

        curve_reports: list[dict[str, Any]] = []
        for curve_index, (name, points) in enumerate(curves):
            curve_root = f"{group_path}/{curve_index:02d}_{_safe_name(name)}"
            UsdGeom.Xform.Define(stage, curve_root)
            curve_path = f"{curve_root}/curve"
            curve = UsdGeom.BasisCurves.Define(stage, curve_path)
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
                point = points[marker_index]
                marker_path = f"{curve_root}/wp_{marker_index:04d}"
                marker = UsdGeom.Sphere.Define(stage, marker_path)
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
            "error": f"{type(exc).__name__}: {exc}",
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
