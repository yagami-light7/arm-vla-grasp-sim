"""USD 视觉层可见性修正工具。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _emit(logger: Callable[[str], None], message: str) -> None:
    logger(f"[visual-visibility] {message}")


def _usd_modules() -> tuple[Any, Any]:
    """延迟导入 Kit 内 USD 模块，避免普通 Python import 依赖 Isaac App。"""

    from pxr import Usd, UsdGeom

    return Usd, UsdGeom


def _current_stage() -> Any:
    import omni.usd

    return omni.usd.get_context().get_stage()


def _keywords_tuple(keywords: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(keyword) for keyword in keywords if str(keyword))


def _path_matches_keywords(path: str, keywords: tuple[str, ...]) -> bool:
    lowered = path.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _resolve_root(stage: Any, root_path: str) -> Any | None:
    try:
        root = stage.GetPrimAtPath(root_path)
    except Exception:
        return None
    return root if root and root.IsValid() else None


def _attribute_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        return None
    try:
        return attr.Get()
    except Exception:
        return None


def _descendant_matches_keywords(
    prim: Any,
    *,
    keywords: tuple[str, ...],
    Usd: Any,
) -> bool:
    if not keywords:
        return False
    root_path = str(prim.GetPath())
    for child in Usd.PrimRange(prim):
        child_path = str(child.GetPath())
        if child_path == root_path:
            continue
        if _path_matches_keywords(child_path, keywords):
            return True
    return False


def _is_renderable_leaf_candidate(prim: Any, *, UsdGeom: Any) -> bool:
    try:
        return bool(prim.IsA(UsdGeom.Mesh))
    except Exception:
        return str(prim.GetTypeName()) == "Mesh"


def _is_collision_mesh_candidate(prim: Any, *, UsdGeom: Any) -> bool:
    """Return whether a renderable mesh is authored as collision geometry.

    Matching only the full prim path is insufficient for assets where a visual
    mesh lives below a collision-named parent (for example
    Apple_M_Apple_0/visual). In that layout both meshes match the parent
    keyword, so hiding every matching Mesh also removes the real visual and
    leaves the object with an empty visible bbox.
    """

    if not _is_renderable_leaf_candidate(prim, UsdGeom=UsdGeom):
        return False
    try:
        schemas = tuple(str(schema) for schema in prim.GetAppliedSchemas())
    except Exception:
        schemas = ()
    if any(
        schema == "PhysicsCollisionAPI"
        or schema.startswith("PhysicsCollisionAPI:")
        for schema in schemas
    ):
        return True
    # Keep compatibility with flattened or partially composed collision assets
    # that expose the collision property without retaining applied-schema data.
    collision_enabled = _attribute_value(prim, "physics:collisionEnabled")
    return collision_enabled is not None


def hide_visual_prims_by_keywords(
    root_path: str = "/World",
    hide_keywords: tuple[str, ...] = ("Apple_M_Apple",),
    keep_keywords: tuple[str, ...] = ("visual_video",),
    *,
    stage: Any | None = None,
    logger: Callable[[str], None] = print,
) -> dict[str, Any]:
    """按关键词隐藏碰撞 mesh 的渲染，同时保留其物理碰撞属性。

    Apple_M_Apple 的父 Xform 下通常还包含 visual_video 补偿层。这里不隐藏包含
    keep keyword 子节点的父 Xform，并要求叶子 Mesh 具有碰撞 schema/属性，只隐藏
    真正的碰撞/占位 Mesh，避免同一父节点下的 visual 或 visual_video 被误隐藏。
    """

    Usd, UsdGeom = _usd_modules()
    stage = stage or _current_stage()
    hide_tuple = _keywords_tuple(hide_keywords)
    keep_tuple = _keywords_tuple(keep_keywords)
    if stage is None:
        report = {
            "applied": False,
            "reason": "usd_stage_unavailable",
            "root_path": root_path,
            "hide_keywords": list(hide_tuple),
            "keep_keywords": list(keep_tuple),
            "hidden_count": 0,
        }
        _emit(logger, "warning: current USD stage is unavailable")
        return report

    root = _resolve_root(stage, root_path)
    if root is None:
        report = {
            "applied": False,
            "reason": "root_prim_not_found",
            "root_path": root_path,
            "hide_keywords": list(hide_tuple),
            "keep_keywords": list(keep_tuple),
            "hidden_count": 0,
        }
        _emit(logger, f"warning: root prim is unavailable: {root_path}")
        return report

    hidden: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    matched_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if not _path_matches_keywords(path, hide_tuple):
            continue
        matched_paths.append(path)
        if _path_matches_keywords(path, keep_tuple):
            skipped.append({"prim_path": path, "reason": "path_matches_keep_keyword"})
            continue
        if _descendant_matches_keywords(prim, keywords=keep_tuple, Usd=Usd):
            skipped.append(
                {
                    "prim_path": path,
                    "reason": "descendant_matches_keep_keyword",
                }
            )
            continue
        if not prim.IsA(UsdGeom.Imageable):
            skipped.append({"prim_path": path, "reason": "not_imageable"})
            continue
        if not _is_renderable_leaf_candidate(prim, UsdGeom=UsdGeom):
            skipped.append({"prim_path": path, "reason": "not_mesh_leaf"})
            continue
        if not _is_collision_mesh_candidate(prim, UsdGeom=UsdGeom):
            skipped.append(
                {
                    "prim_path": path,
                    "reason": "mesh_without_collision_schema_or_attribute",
                }
            )
            continue

        imageable = UsdGeom.Imageable(prim)
        before = {
            "visibility": _attribute_value(prim, "visibility"),
            "computed_visibility": str(imageable.ComputeVisibility()),
        }
        imageable.MakeInvisible()
        after = {
            "visibility": _attribute_value(prim, "visibility"),
            "computed_visibility": str(imageable.ComputeVisibility()),
        }
        item = {
            "prim_path": path,
            "type": str(prim.GetTypeName()),
            "before": before,
            "after": after,
        }
        hidden.append(item)
        _emit(
            logger,
            (
                f"hidden {path} visibility={before['visibility']}"
                f"->{after['visibility']} computed={before['computed_visibility']}"
                f"->{after['computed_visibility']}"
            ),
        )

    if not hidden:
        _emit(
            logger,
            (
                "warning: no visual prim hidden; "
                f"root={root_path} hide_keywords={list(hide_tuple)}"
            ),
        )
    _emit(logger, f"hide complete: count={len(hidden)}")
    return {
        "applied": bool(hidden),
        "reason": None if hidden else "visual_prim_not_found",
        "root_path": root_path,
        "hide_keywords": list(hide_tuple),
        "keep_keywords": list(keep_tuple),
        "matched_paths": matched_paths,
        "collision_mesh_required": True,
        "skipped_paths": skipped,
        "hidden_count": len(hidden),
        "hidden_prims": hidden,
    }
