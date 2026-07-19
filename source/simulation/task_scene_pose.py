"""在 PhysX 初始化前应用任务级静态场景物体位姿。"""

from __future__ import annotations

import math
from typing import Any


def _absolute_prim_path(value: Any, *, field_name: str) -> str:
    """校验并规范绝对 USD prim path。"""

    path = str(value or "").strip()
    if not path.startswith("/") or path == "/" or "//" in path:
        raise ValueError(f"{field_name} 必须是绝对 USD prim path")
    return path.rstrip("/")


def _resolve_scene_pose(
    raw_pose: Any,
    *,
    field_name: str,
    fallback_prim_path: Any = None,
    missing_reason: str,
) -> dict[str, Any]:
    """解析一个静态场景物体根位姿及其可选碰撞约束。"""

    if raw_pose is None:
        return {"configured": False, "reason": missing_reason}
    if not isinstance(raw_pose, dict):
        raise ValueError(f"{field_name} 必须是对象")
    prim_path = _absolute_prim_path(
        raw_pose.get("prim_path") or fallback_prim_path,
        field_name=f"{field_name}.prim_path",
    )
    missing_position = [key for key in ("x", "y", "z") if key not in raw_pose]
    if missing_position:
        raise ValueError(f"{field_name} 缺少位置字段: {missing_position}")
    values: dict[str, float] = {}
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        try:
            value = float(raw_pose.get(key, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}.{key} 必须是数值") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name}.{key} 必须是有限数值")
        values[key] = value

    translation_only = bool(raw_pose.get("translation_only", False))
    collision_prim_raw = raw_pose.get("collision_prim_path")
    collision_prim_path = None
    if collision_prim_raw is not None:
        collision_prim_path = _absolute_prim_path(
            collision_prim_raw,
            field_name=f"{field_name}.collision_prim_path",
        )
        if not collision_prim_path.startswith(prim_path + "/"):
            raise ValueError(
                f"{field_name}.collision_prim_path 必须位于场景物体根 prim 下"
            )
    ensure_collision = bool(raw_pose.get("ensure_static_mesh_collision", False))
    if ensure_collision and collision_prim_path is None:
        raise ValueError(
            f"{field_name}.ensure_static_mesh_collision=true 时必须配置 collision_prim_path"
        )

    expected_dims_raw = raw_pose.get("expected_support_bbox_dims_xyz")
    expected_dims = None
    if expected_dims_raw is not None:
        if not isinstance(expected_dims_raw, (list, tuple)) or len(expected_dims_raw) != 3:
            raise ValueError(
                f"{field_name}.expected_support_bbox_dims_xyz 必须包含 3 个数值"
            )
        expected_dims = []
        for index, raw_value in enumerate(expected_dims_raw):
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name}.expected_support_bbox_dims_xyz[{index}] 必须是数值"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{field_name}.expected_support_bbox_dims_xyz[{index}] 必须是正有限数"
                )
            expected_dims.append(value)
    try:
        bbox_tolerance = float(raw_pose.get("support_bbox_tolerance_m", 5.0e-5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}.support_bbox_tolerance_m 必须是数值") from exc
    if not math.isfinite(bbox_tolerance) or bbox_tolerance < 0.0:
        raise ValueError(f"{field_name}.support_bbox_tolerance_m 必须是非负有限数")

    return {
        "configured": True,
        "field_name": field_name,
        "prim_path": prim_path,
        "pose_world": values,
        "translation_only": translation_only,
        "collision_prim_path": collision_prim_path,
        "ensure_static_mesh_collision": ensure_collision,
        "expected_support_bbox_dims_xyz": expected_dims,
        "support_bbox_tolerance_m": bbox_tolerance,
    }


def resolve_task_pick_support_pose(raw_task: dict[str, Any]) -> dict[str, Any]:
    """解析可选的抓取支撑体根位姿。"""

    raw_pick = raw_task.get("pick")
    raw_pick = raw_pick if isinstance(raw_pick, dict) else {}
    return _resolve_scene_pose(
        raw_pick.get("support_pose_world"),
        field_name="task.pick.support_pose_world",
        fallback_prim_path=raw_pick.get("target_support_root_prim_path"),
        missing_reason="pick_support_pose_world_missing",
    )


def resolve_task_receptacle_pose(raw_task: dict[str, Any]) -> dict[str, Any]:
    """解析可选的 receptacle 根位姿；旧任务保持无覆盖语义。"""

    raw_place = raw_task.get("place")
    raw_place = raw_place if isinstance(raw_place, dict) else {}
    return _resolve_scene_pose(
        raw_place.get("receptacle_pose_world"),
        field_name="task.place.receptacle_pose_world",
        fallback_prim_path=(
            raw_place.get("target_receptacle_prim_path")
            or raw_task.get("target_receptacle_prim_path")
        ),
        missing_reason="receptacle_pose_world_missing",
    )


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


def inspect_episode_static_support_body_mode(root_prim: Any) -> dict[str, Any]:
    """Classify a support as USD-static or episode-static kinematic.

    A support randomized between episodes cannot remain a USD-static collider when
    one live PhysX stage is reused: PhysX does not guarantee hot propagation of a
    static actor's authored transform.  A kinematic rigid body at the support root
    is equivalent to a static obstacle *within* an episode, while still exposing a
    tensor pose that can be changed transactionally during reset.
    """

    from pxr import Usd, UsdPhysics

    root_path = str(root_prim.GetPath())
    rigid_body_prims = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rigid_body_paths = [str(prim.GetPath()) for prim in rigid_body_prims]
    root_kinematic = False
    root_rigid_enabled = False
    if len(rigid_body_prims) == 1 and rigid_body_paths == [root_path]:
        rigid_api = UsdPhysics.RigidBodyAPI(rigid_body_prims[0])
        kinematic_value = rigid_api.GetKinematicEnabledAttr().Get()
        enabled_value = rigid_api.GetRigidBodyEnabledAttr().Get()
        root_kinematic = bool(kinematic_value)
        root_rigid_enabled = True if enabled_value is None else bool(enabled_value)

    usd_static = not rigid_body_prims
    kinematic_episode_static = bool(root_kinematic and root_rigid_enabled)
    episode_static = bool(usd_static or kinematic_episode_static)
    return {
        "support_body_mode": (
            "usd_static"
            if usd_static
            else (
                "kinematic_episode_static"
                if kinematic_episode_static
                else "unsupported_rigid_body_layout"
            )
        ),
        "rigid_body_api_count": len(rigid_body_paths),
        "rigid_body_prim_paths": rigid_body_paths,
        "root_kinematic_enabled": root_kinematic,
        "root_rigid_body_enabled": root_rigid_enabled,
        "usd_static_support_verified": usd_static,
        "episode_static_support_verified": episode_static,
        # Existing quality gates use this field to mean that contacts cannot move
        # the receptacle during an episode.  Kinematic supports satisfy that
        # contract even though their reset pose is mutable between episodes.
        "static_support_verified": episode_static,
    }


def configure_task_supports_for_stage_reuse(
    stage: Any,
    raw_task: dict[str, Any],
) -> dict[str, Any]:
    """Make randomized support roots kinematic before PhysX parses the stage."""

    from pxr import Usd, UsdPhysics

    settings_by_role = {
        "pick": resolve_task_pick_support_pose(raw_task),
        "place": resolve_task_receptacle_pose(raw_task),
    }
    reports: dict[str, Any] = {}
    configured_count = 0
    for role, settings in settings_by_role.items():
        if settings.get("configured") is not True:
            reports[role] = {
                "configured": False,
                "reason": settings.get("reason"),
            }
            continue
        root_path = str(settings["prim_path"])
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid() or not root_prim.IsActive():
            raise RuntimeError(
                f"stage-reuse support root is unavailable: {root_path}"
            )
        collision_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root_prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not collision_paths:
            reports[role] = {
                "configured": False,
                "reason": "support_root_has_no_collision_api",
                "prim_path": root_path,
            }
            continue

        body_report_before = inspect_episode_static_support_body_mode(root_prim)
        if body_report_before["rigid_body_api_count"]:
            if body_report_before["episode_static_support_verified"] is not True:
                raise RuntimeError(
                    "stage-reuse support already has a non-kinematic or nested "
                    f"RigidBodyAPI: {body_report_before}"
                )
            rigid_api = UsdPhysics.RigidBodyAPI(root_prim)
            api_existed = True
        else:
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
            api_existed = False
        rigid_api.CreateRigidBodyEnabledAttr(True).Set(True)
        rigid_api.CreateKinematicEnabledAttr(True).Set(True)
        body_report_after = inspect_episode_static_support_body_mode(root_prim)
        if body_report_after["support_body_mode"] != "kinematic_episode_static":
            raise RuntimeError(
                "failed to configure episode-static kinematic support: "
                f"{body_report_after}"
            )
        configured_count += 1
        reports[role] = {
            "configured": True,
            "prim_path": root_path,
            "collision_prim_paths": collision_paths,
            "rigid_body_api_existed": api_existed,
            "pose_mutation_scope": "between_episodes_only",
            **body_report_after,
        }
    return {
        "enabled": True,
        "configured_count": configured_count,
        "supports": reports,
        "contact_semantics": "immovable_kinematic_support_within_episode",
    }


def _support_collision_report(stage: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """按任务要求补齐静态 Mesh 碰撞并核对组合后支撑包围盒。"""

    collision_prim_path = settings.get("collision_prim_path")
    if collision_prim_path is None:
        return {
            "configured": False,
            "ensured": False,
            "reason": "collision_prim_path_missing",
        }

    from pxr import Usd, UsdGeom, UsdPhysics

    collision_prim = stage.GetPrimAtPath(str(collision_prim_path))
    if not collision_prim.IsValid() or not collision_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            f"task scene support collision prim 不是有效 Mesh: {collision_prim_path}"
        )
    root_prim = stage.GetPrimAtPath(str(settings["prim_path"]))
    body_mode_report = inspect_episode_static_support_body_mode(root_prim)
    if body_mode_report["episode_static_support_verified"] is not True:
        raise RuntimeError(
            "task scene support requires an episode-static collision body, but "
            f"found: {body_mode_report}"
        )

    collision_existed = collision_prim.HasAPI(UsdPhysics.CollisionAPI)
    mesh_collision_existed = collision_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    if settings.get("ensure_static_mesh_collision"):
        collision_api = (
            UsdPhysics.CollisionAPI(collision_prim)
            if collision_existed
            else UsdPhysics.CollisionAPI.Apply(collision_prim)
        )
        collision_api.CreateCollisionEnabledAttr(True).Set(True)
        mesh_collision_api = (
            UsdPhysics.MeshCollisionAPI(collision_prim)
            if mesh_collision_existed
            else UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        )
        mesh_collision_api.CreateApproximationAttr("none").Set("none")

    collision_enabled = None
    if collision_prim.HasAPI(UsdPhysics.CollisionAPI):
        collision_enabled = UsdPhysics.CollisionAPI(
            collision_prim
        ).GetCollisionEnabledAttr().Get()
        if collision_enabled is None:
            collision_enabled = True
    if settings.get("ensure_static_mesh_collision") and collision_enabled is not True:
        raise RuntimeError(
            f"task scene support CollisionAPI 未启用: {collision_prim_path}"
        )

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    aligned_box = bbox_cache.ComputeWorldBound(collision_prim).ComputeAlignedBox()
    bbox_min = tuple(float(value) for value in aligned_box.GetMin())
    bbox_max = tuple(float(value) for value in aligned_box.GetMax())
    bbox_dims = tuple(bbox_max[index] - bbox_min[index] for index in range(3))
    if not all(math.isfinite(value) for value in (*bbox_min, *bbox_max)):
        raise RuntimeError(f"task scene support bbox 不是有限值: {collision_prim_path}")
    if any(value <= 0.0 for value in bbox_dims):
        raise RuntimeError(
            f"task scene support bbox 尺寸无效: {collision_prim_path} {bbox_dims}"
        )

    expected_dims = settings.get("expected_support_bbox_dims_xyz")
    dims_error = None
    if expected_dims is not None:
        dims_error = [
            abs(float(actual) - float(expected))
            for actual, expected in zip(bbox_dims, expected_dims)
        ]
        tolerance = float(settings["support_bbox_tolerance_m"])
        if any(error > tolerance for error in dims_error):
            raise RuntimeError(
                "task scene support bbox 与任务配置不一致: "
                f"actual={bbox_dims} expected={expected_dims} error={dims_error}"
            )

    return {
        "configured": True,
        "ensured": bool(settings.get("ensure_static_mesh_collision")),
        "collision_prim_path": str(collision_prim_path),
        "collision_api_existed": collision_existed,
        "mesh_collision_api_existed": mesh_collision_existed,
        "collision_api_present": collision_prim.HasAPI(UsdPhysics.CollisionAPI),
        "mesh_collision_api_present": collision_prim.HasAPI(
            UsdPhysics.MeshCollisionAPI
        ),
        "collision_enabled": collision_enabled,
        **body_mode_report,
        "world_bbox_min_xyz": list(bbox_min),
        "world_bbox_max_xyz": list(bbox_max),
        "world_bbox_center_xyz": [
            (bbox_min[index] + bbox_max[index]) * 0.5 for index in range(3)
        ],
        "world_bbox_dims_xyz": list(bbox_dims),
        "support_surface_z": bbox_max[2],
        "expected_bbox_dims_xyz": expected_dims,
        "bbox_dims_error_m": dims_error,
        "bbox_tolerance_m": float(settings["support_bbox_tolerance_m"]),
        "geometry_verified": True,
    }


def _apply_scene_pose(stage: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """把一个任务位姿写入组合 stage，且不破坏资产导入变换。"""

    from pxr import Gf, UsdGeom

    prim_path = str(settings["prim_path"])
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"task scene prim 不是有效 Xformable: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    op_order_before = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
    pose = settings["pose_world"]
    translate_op = _get_or_add_xform_op(
        xformable,
        UsdGeom.XformOp.TypeTranslate,
    )
    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        translate_op.Set(
            Gf.Vec3d(float(pose["x"]), float(pose["y"]), float(pose["z"]))
        )
    else:
        translate_op.Set(
            Gf.Vec3f(float(pose["x"]), float(pose["y"]), float(pose["z"]))
        )

    quaternion = None
    if not settings.get("translation_only"):
        quaternion = _quaternion_wxyz_from_rpy(
            float(pose["roll"]),
            float(pose["pitch"]),
            float(pose["yaw"]),
        )
        orient_op = _get_or_add_xform_op(
            xformable,
            UsdGeom.XformOp.TypeOrient,
        )
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
    collision_report = _support_collision_report(stage, settings)
    return {
        **settings,
        "applied": True,
        "quaternion_wxyz": None if quaternion is None else list(quaternion),
        "orientation_preserved": bool(settings.get("translation_only")),
        "xform_op_order_before": op_order_before,
        "xform_op_order_after": op_order_after,
        "reset_xform_stack": False,
        "units_resolve_preserved": any(
            "unitsResolve" in name for name in op_order_after
        ),
        "support_collision_report": collision_report,
    }


def apply_task_receptacle_pose(stage: Any, raw_task: dict[str, Any]) -> dict[str, Any]:
    """同步抓取支撑体与放置 receptacle，并保留旧报告入口。"""

    pick_settings = resolve_task_pick_support_pose(raw_task)
    receptacle_settings = resolve_task_receptacle_pose(raw_task)
    if (
        pick_settings.get("configured")
        and receptacle_settings.get("configured")
        and pick_settings["prim_path"] == receptacle_settings["prim_path"]
    ):
        raise ValueError("pick support 与 place receptacle 不能重复覆盖同一 prim")

    pick_report = (
        _apply_scene_pose(stage, pick_settings)
        if pick_settings.get("configured")
        else pick_settings
    )
    receptacle_report = (
        _apply_scene_pose(stage, receptacle_settings)
        if receptacle_settings.get("configured")
        else receptacle_settings
    )
    return {
        **receptacle_report,
        "any_scene_pose_configured": bool(
            pick_settings.get("configured") or receptacle_settings.get("configured")
        ),
        "pick_support_pose_report": pick_report,
        "receptacle_pose_report": receptacle_report,
    }
