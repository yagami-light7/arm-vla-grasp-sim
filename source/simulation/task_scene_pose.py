"""在 PhysX 初始化前应用任务级静态场景物体位姿。"""

from __future__ import annotations

import math
from typing import Any


def resolve_task_receptacle_pose(raw_task: dict[str, Any]) -> dict[str, Any]:
    """解析可选的 receptacle 根位姿；旧任务保持无覆盖语义。"""

    raw_place = raw_task.get("place")
    raw_place = raw_place if isinstance(raw_place, dict) else {}
    raw_pose = raw_place.get("receptacle_pose_world")
    if raw_pose is None:
        return {"configured": False, "reason": "receptacle_pose_world_missing"}
    if not isinstance(raw_pose, dict):
        raise ValueError("task.place.receptacle_pose_world 必须是对象")
    prim_path = str(
        raw_pose.get("prim_path")
        or raw_place.get("target_receptacle_prim_path")
        or raw_task.get("target_receptacle_prim_path")
        or ""
    ).strip()
    if not prim_path.startswith("/") or prim_path == "/" or "//" in prim_path:
        raise ValueError(
            "task.place.receptacle_pose_world.prim_path 必须是绝对 USD prim path"
        )
    missing_position = [key for key in ("x", "y", "z") if key not in raw_pose]
    if missing_position:
        raise ValueError(
            "task.place.receptacle_pose_world 缺少位置字段: "
            f"{missing_position}"
        )
    values: dict[str, float] = {}
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        try:
            value = float(raw_pose.get(key, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"task.place.receptacle_pose_world.{key} 必须是数值"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"task.place.receptacle_pose_world.{key} 必须是有限数值"
            )
        values[key] = value
    return {
        "configured": True,
        "prim_path": prim_path.rstrip("/"),
        "pose_world": values,
    }


def _get_or_add_xform_op(xformable: Any, op_type: Any) -> Any:
    """复用已有 xform op，确保 scale 和 unitsResolve 顺序不被破坏。"""

    from pxr import UsdGeom

    attr_name = {
        UsdGeom.XformOp.TypeTranslate: "xformOp:translate",
        UsdGeom.XformOp.TypeOrient: "xformOp:orient",
    }[op_type]
    attr = xformable.GetPrim().GetAttribute(attr_name)
    if attr.IsValid():
        return UsdGeom.XformOp(attr)
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    return xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)


def _quaternion_wxyz_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def apply_task_receptacle_pose(stage: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
    """把 episode 采样的静态垫子位姿写入当前组合 stage。"""

    settings = resolve_task_receptacle_pose(raw_task)
    if not settings["configured"]:
        return settings

    from pxr import Gf, UsdGeom

    prim_path = str(settings["prim_path"])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"task receptacle prim 不是有效 Xformable: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    op_order_before = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
    pose = settings["pose_world"]
    quaternion = _quaternion_wxyz_from_rpy(
        float(pose["roll"]),
        float(pose["pitch"]),
        float(pose["yaw"]),
    )
    translate_op = _get_or_add_xform_op(
        xformable,
        UsdGeom.XformOp.TypeTranslate,
    )
    orient_op = _get_or_add_xform_op(
        xformable,
        UsdGeom.XformOp.TypeOrient,
    )
    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        translate_op.Set(Gf.Vec3d(float(pose["x"]), float(pose["y"]), float(pose["z"])))
    else:
        translate_op.Set(Gf.Vec3f(float(pose["x"]), float(pose["y"]), float(pose["z"])))
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient_op.Set(
            Gf.Quatd(
                quaternion[0],
                Gf.Vec3d(quaternion[1], quaternion[2], quaternion[3]),
            )
        )
    else:
        orient_op.Set(
            Gf.Quatf(
                quaternion[0],
                Gf.Vec3f(quaternion[1], quaternion[2], quaternion[3]),
            )
        )
    op_order_after = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
    return {
        **settings,
        "applied": True,
        "quaternion_wxyz": list(quaternion),
        "xform_op_order_before": op_order_before,
        "xform_op_order_after": op_order_after,
        "reset_xform_stack": False,
        "units_resolve_preserved": any(
            "unitsResolve" in name for name in op_order_after
        ),
    }
