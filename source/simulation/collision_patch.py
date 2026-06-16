"""Go2-X5 夹爪碰撞几何修正工具。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


_PATCH_REPORT_ATTR = "_go2_x5_gripper_collision_patch_report"
_KEYWORD_PATCH_REPORT_ATTR = "_keyword_collision_patch_report"
_PATCH_INSTALLED_ATTR = "_go2_x5_gripper_collision_patch_installed"


def _emit(logger: Callable[[str], None], message: str) -> None:
    logger(f"[gripper-collision] {message}")


def _emit_keyword(logger: Callable[[str], None], message: str) -> None:
    logger(f"[keyword-collision] {message}")


def _usd_modules() -> tuple[Any, Any, Any, Any]:
    """延迟导入 Kit 内的 USD 模块，避免普通单元测试依赖 Isaac App。"""

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    return Usd, UsdPhysics, PhysxSchema, Sdf


def _current_stage() -> Any:
    import omni.usd

    return omni.usd.get_context().get_stage()


def _iter_prims(root: Any, Usd: Any, *, include_instance_proxies: bool) -> list[Any]:
    if not root or not root.IsValid():
        return []
    if include_instance_proxies:
        try:
            return list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        except (AttributeError, TypeError):
            pass
    return list(Usd.PrimRange(root))


def _path_contains_gripper_link(path: str, gripper_links: tuple[str, ...]) -> bool:
    segments = tuple(segment for segment in path.split("/") if segment)
    return any(link in segments for link in gripper_links)


def _path_matches_keywords(path: str, keywords: tuple[str, ...]) -> bool:
    lowered = path.lower()
    return any(str(keyword).lower() in lowered for keyword in keywords if str(keyword))


def _root_path_from_gripper_path(path: str, gripper_links: tuple[str, ...]) -> str | None:
    for link in gripper_links:
        marker = f"/{link}"
        if marker in path:
            return path.split(marker, 1)[0]
    return None


def _resolve_robot_roots(
    stage: Any,
    *,
    robot_root: str,
    gripper_links: tuple[str, ...],
    Usd: Any,
) -> list[Any]:
    """保留显式 root，同时兼容 IsaacLab 的 /World/envs/env_0/Robot。"""

    roots: dict[str, Any] = {}
    try:
        requested = stage.GetPrimAtPath(robot_root)
    except Exception:
        requested = None
    if requested and requested.IsValid():
        roots[str(requested.GetPath())] = requested

    pseudo_root = stage.GetPseudoRoot()
    for prim in _iter_prims(pseudo_root, Usd, include_instance_proxies=True):
        path = str(prim.GetPath())
        if not _path_contains_gripper_link(path, gripper_links):
            continue
        root_path = _root_path_from_gripper_path(path, gripper_links)
        if not root_path:
            continue
        candidate = stage.GetPrimAtPath(root_path)
        if candidate and candidate.IsValid():
            roots[root_path] = candidate
    return [roots[path] for path in sorted(roots)]


def _instance_root_for_prim(prim: Any) -> Any | None:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.IsInstance() or current.IsInstanceable():
            return current
        current = current.GetParent()
    return None


def _deinstance_roots(
    robot_roots: list[Any],
    *,
    Usd: Any,
    logger: Callable[[str], None],
) -> tuple[list[str], list[str]]:
    """逐层取消实例化，确保引用资产中的碰撞 mesh 可在当前 stage 编辑。"""

    found: list[str] = []
    disabled: list[str] = []
    for _pass_index in range(8):
        candidates: dict[str, Any] = {}
        for robot_root in robot_roots:
            for prim in _iter_prims(robot_root, Usd, include_instance_proxies=True):
                instance_root = _instance_root_for_prim(prim)
                if instance_root is None:
                    continue
                path = str(instance_root.GetPath())
                candidates[path] = instance_root
                if path not in found:
                    found.append(path)
                    _emit(logger, f"found instance root: {path}")
        pending = [
            candidates[path]
            for path in sorted(candidates, key=lambda item: (item.count("/"), item))
            if path not in disabled
        ]
        if not pending:
            break
        progress = False
        for prim in pending:
            path = str(prim.GetPath())
            try:
                prim.SetInstanceable(False)
            except Exception as exc:
                _emit(logger, f"warning: failed to disable instanceable {path}: {exc}")
                continue
            disabled.append(path)
            progress = True
            _emit(logger, f"disabled instanceable prim: {path}")
        if not progress:
            break
    return found, disabled


def _applied_schema_names(prim: Any) -> tuple[str, ...]:
    try:
        return tuple(str(name) for name in prim.GetAppliedSchemas())
    except Exception:
        return ()


def _has_api(prim: Any, api_type: Any) -> bool:
    try:
        return bool(prim.HasAPI(api_type))
    except Exception:
        return False


def _attribute_names(prim: Any) -> tuple[str, ...]:
    try:
        return tuple(str(attribute.GetName()) for attribute in prim.GetAttributes())
    except Exception:
        return ()


def _has_collision_attribute(prim: Any) -> bool:
    return any("collision" in name.lower() for name in _attribute_names(prim))


def _is_mesh_collision_target(prim: Any, *, UsdPhysics: Any) -> bool:
    schemas = set(_applied_schema_names(prim))
    return bool(
        str(prim.GetTypeName()) == "Mesh"
        or any("PhysicsMeshCollisionAPI" in schema for schema in schemas)
        or _has_api(prim, UsdPhysics.MeshCollisionAPI)
        or prim.GetAttribute("physics:approximation").IsValid()
    )


def _is_collision_prim(
    prim: Any,
    *,
    UsdPhysics: Any,
    PhysxSchema: Any,
) -> bool:
    schemas = set(_applied_schema_names(prim))
    collision_schemas = {
        "PhysicsCollisionAPI",
        "PhysxCollisionAPI",
    }
    collision_like = any(
        compatible in schema
        for schema in schemas
        for compatible in collision_schemas
    ) or bool(
        _has_api(prim, UsdPhysics.CollisionAPI)
        or _has_api(prim, PhysxSchema.PhysxCollisionAPI)
        or _has_collision_attribute(prim)
    )
    mesh_like = _is_mesh_collision_target(prim, UsdPhysics=UsdPhysics)
    return bool(collision_like and mesh_like)


def _find_gripper_collision_prims(
    robot_roots: Iterable[Any],
    *,
    gripper_links: tuple[str, ...],
    Usd: Any,
    UsdPhysics: Any,
    PhysxSchema: Any,
) -> list[Any]:
    collisions: dict[str, Any] = {}
    for robot_root in robot_roots:
        for prim in _iter_prims(robot_root, Usd, include_instance_proxies=False):
            path = str(prim.GetPath())
            if not _path_contains_gripper_link(path, gripper_links):
                continue
            if _is_collision_prim(
                prim,
                UsdPhysics=UsdPhysics,
                PhysxSchema=PhysxSchema,
            ):
                collisions[path] = prim
    return [collisions[path] for path in sorted(collisions)]


def _resolve_root(stage: Any, root_path: str) -> Any | None:
    try:
        root = stage.GetPrimAtPath(root_path)
    except Exception:
        return None
    return root if root and root.IsValid() else None


def _keyword_instance_roots(
    root: Any,
    *,
    keywords: tuple[str, ...],
    Usd: Any,
) -> list[Any]:
    """仅取消关键词相关资产的实例化，避免误改 /World 下其他资产。"""

    roots: dict[str, Any] = {}
    for prim in _iter_prims(root, Usd, include_instance_proxies=True):
        if not _path_matches_keywords(str(prim.GetPath()), keywords):
            continue
        instance_root = _instance_root_for_prim(prim)
        if instance_root is not None:
            roots[str(instance_root.GetPath())] = instance_root
    return [roots[path] for path in sorted(roots)]


def _deinstance_keyword_roots(
    instance_roots: list[Any],
    *,
    logger: Callable[[str], None],
) -> tuple[list[str], list[str]]:
    found: list[str] = []
    disabled: list[str] = []
    for prim in instance_roots:
        path = str(prim.GetPath())
        found.append(path)
        _emit_keyword(logger, f"found instance root: {path}")
        try:
            prim.SetInstanceable(False)
        except Exception as exc:
            _emit_keyword(logger, f"warning: failed to disable instanceable {path}: {exc}")
            continue
        disabled.append(path)
        _emit_keyword(logger, f"disabled instanceable prim: {path}")
    return found, disabled


def _find_collision_prims_by_keywords(
    root: Any,
    *,
    keywords: tuple[str, ...],
    Usd: Any,
    UsdPhysics: Any,
    PhysxSchema: Any,
) -> list[Any]:
    collisions: dict[str, Any] = {}
    for prim in _iter_prims(root, Usd, include_instance_proxies=False):
        path = str(prim.GetPath())
        if not _path_matches_keywords(path, keywords):
            continue
        if _is_collision_prim(
            prim,
            UsdPhysics=UsdPhysics,
            PhysxSchema=PhysxSchema,
        ):
            collisions[path] = prim
    return [collisions[path] for path in sorted(collisions)]


def _attribute_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        return None
    try:
        return attr.Get()
    except Exception:
        return None


def _set_attribute(prim: Any, name: str, type_name: Any, value: Any) -> None:
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, type_name, custom=False)
    attr.Set(value)


def _collision_info(prim: Any) -> dict[str, Any]:
    return {
        "prim_path": str(prim.GetPath()),
        "type": str(prim.GetTypeName()),
        "applied_schemas": list(_applied_schema_names(prim)),
        "physics:collisionEnabled": _attribute_value(prim, "physics:collisionEnabled"),
        "physics:approximation": _attribute_value(prim, "physics:approximation"),
        "physxCollision:contactOffset": _attribute_value(
            prim,
            "physxCollision:contactOffset",
        ),
        "physxCollision:restOffset": _attribute_value(
            prim,
            "physxCollision:restOffset",
        ),
        "physics:restOffset": _attribute_value(prim, "physics:restOffset"),
    }


def print_gripper_collision_info(
    robot_root: str = "/World/go2_x5",
    gripper_links: tuple[str, ...] = ("arm_link7", "arm_link8"),
    *,
    stage: Any | None = None,
    logger: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """打印夹爪碰撞 prim 的 schema、近接触参数和 mesh approximation。"""

    Usd, UsdPhysics, PhysxSchema, _Sdf = _usd_modules()
    stage = stage or _current_stage()
    if stage is None:
        _emit(logger, "warning: current USD stage is unavailable")
        return []
    roots = _resolve_robot_roots(
        stage,
        robot_root=robot_root,
        gripper_links=gripper_links,
        Usd=Usd,
    )
    collisions = _find_gripper_collision_prims(
        roots,
        gripper_links=gripper_links,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
    )
    infos = [_collision_info(prim) for prim in collisions]
    if not infos:
        _emit(logger, f"warning: no gripper collision prim found below {robot_root}")
        return []
    for info in infos:
        _emit(
            logger,
            "info "
            f"path={info['prim_path']} type={info['type']} "
            f"schemas={info['applied_schemas']} "
            f"collisionEnabled={info['physics:collisionEnabled']} "
            f"approximation={info['physics:approximation']} "
            f"contactOffset={info['physxCollision:contactOffset']} "
            f"physxRestOffset={info['physxCollision:restOffset']} "
            f"physicsRestOffset={info['physics:restOffset']}",
        )
    return infos


def print_collision_info_by_keywords(
    root_path: str = "/World",
    keywords: tuple[str, ...] = ("apple", "Apple"),
    *,
    stage: Any | None = None,
    logger: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """按路径关键词打印碰撞 prim 信息，用于确认苹果等资产的 patch 是否生效。"""

    Usd, UsdPhysics, PhysxSchema, _Sdf = _usd_modules()
    stage = stage or _current_stage()
    if stage is None:
        _emit_keyword(logger, "warning: current USD stage is unavailable")
        return []
    keyword_tuple = tuple(str(keyword) for keyword in keywords)
    root = _resolve_root(stage, root_path)
    if root is None:
        _emit_keyword(logger, f"warning: root prim is unavailable: {root_path}")
        return []
    collisions = _find_collision_prims_by_keywords(
        root,
        keywords=keyword_tuple,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
    )
    infos = [_collision_info(prim) for prim in collisions]
    if not infos:
        _emit_keyword(
            logger,
            (
                "warning: no keyword collision prim found "
                f"root={root_path} keywords={list(keyword_tuple)}"
            ),
        )
        return []
    for info in infos:
        _emit_keyword(
            logger,
            "info "
            f"path={info['prim_path']} type={info['type']} "
            f"schemas={info['applied_schemas']} "
            f"collisionEnabled={info['physics:collisionEnabled']} "
            f"approximation={info['physics:approximation']} "
            f"contactOffset={info['physxCollision:contactOffset']} "
            f"physxRestOffset={info['physxCollision:restOffset']} "
            f"physicsRestOffset={info['physics:restOffset']}",
        )
    return infos


def patch_go2_x5_gripper_collision(
    robot_root: str = "/World/go2_x5",
    gripper_links: tuple[str, ...] = ("arm_link7", "arm_link8"),
    approximation: str = "convexDecomposition",
    contact_offset: float = 0.0002,
    rest_offset: float = 0.0,
    *,
    stage: Any | None = None,
    logger: Callable[[str], None] = print,
) -> dict[str, Any]:
    """取消夹爪实例化并收紧真实碰撞边界；只应在首次 physics reset 前调用。"""

    Usd, UsdPhysics, PhysxSchema, Sdf = _usd_modules()
    stage = stage or _current_stage()
    if stage is None:
        report = {
            "applied": False,
            "reason": "usd_stage_unavailable",
            "patch_count": 0,
        }
        _emit(logger, "warning: current USD stage is unavailable")
        return report

    links = tuple(str(link) for link in gripper_links)
    roots = _resolve_robot_roots(
        stage,
        robot_root=robot_root,
        gripper_links=links,
        Usd=Usd,
    )
    resolved_root_paths = [str(root.GetPath()) for root in roots]
    _emit(
        logger,
        f"requested robot root={robot_root}, resolved roots={resolved_root_paths}",
    )
    instance_roots, deinstanced = _deinstance_roots(
        roots,
        Usd=Usd,
        logger=logger,
    )
    _emit(logger, f"instance roots found={instance_roots}")
    _emit(logger, f"instanceable prims disabled={deinstanced}")
    # 取消 instanceable 后重新解析 root，确保 traversal 使用新的 composed prim。
    roots = [
        stage.GetPrimAtPath(path)
        for path in resolved_root_paths
        if stage.GetPrimAtPath(path).IsValid()
    ]
    collisions = _find_gripper_collision_prims(
        roots,
        gripper_links=links,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
    )

    patched: list[dict[str, Any]] = []
    for prim in collisions:
        before = _collision_info(prim)
        try:
            if not _has_api(prim, UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not _has_api(prim, UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            if not _has_api(prim, PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            _set_attribute(
                prim,
                "physics:approximation",
                Sdf.ValueTypeNames.Token,
                str(approximation),
            )
            _set_attribute(
                prim,
                "physxCollision:contactOffset",
                Sdf.ValueTypeNames.Float,
                float(contact_offset),
            )
            _set_attribute(
                prim,
                "physxCollision:restOffset",
                Sdf.ValueTypeNames.Float,
                float(rest_offset),
            )
            _set_attribute(
                prim,
                "physics:restOffset",
                Sdf.ValueTypeNames.Float,
                float(rest_offset),
            )
        except Exception as exc:
            _emit(logger, f"warning: failed to patch {prim.GetPath()}: {exc}")
            continue
        after = _collision_info(prim)
        patched.append({"prim_path": str(prim.GetPath()), "before": before, "after": after})
        _emit(
            logger,
            f"patched {prim.GetPath()} "
            f"approximation={before['physics:approximation']}->{after['physics:approximation']} "
            f"contactOffset={before['physxCollision:contactOffset']}"
            f"->{after['physxCollision:contactOffset']} "
            f"restOffset={before['physxCollision:restOffset']}"
            f"->{after['physxCollision:restOffset']}",
        )

    if not patched:
        _emit(
            logger,
            f"warning: no gripper collision prim patched below resolved roots {resolved_root_paths}",
        )
    _emit(logger, f"patch complete: count={len(patched)}")
    return {
        "applied": bool(patched),
        "reason": None if patched else "gripper_collision_prim_not_found",
        "requested_robot_root": robot_root,
        "resolved_robot_roots": resolved_root_paths,
        "gripper_links": list(links),
        "approximation": str(approximation),
        "contact_offset": float(contact_offset),
        "rest_offset": float(rest_offset),
        "instance_roots_found": instance_roots,
        "deinstanced_prim_paths": deinstanced,
        "patch_count": len(patched),
        "patched_prims": patched,
    }


def patch_collision_prims_by_keywords(
    root_path: str = "/World",
    keywords: tuple[str, ...] = ("apple", "Apple"),
    approximation: str = "convexDecomposition",
    contact_offset: float = 0.001,
    rest_offset: float = 0.0,
    *,
    stage: Any | None = None,
    logger: Callable[[str], None] = print,
) -> dict[str, Any]:
    """按关键词修正资产碰撞几何；默认用于苹果，必须在首次 physics reset 前调用。"""

    Usd, UsdPhysics, PhysxSchema, Sdf = _usd_modules()
    stage = stage or _current_stage()
    keyword_tuple = tuple(str(keyword) for keyword in keywords)
    if stage is None:
        report = {
            "applied": False,
            "reason": "usd_stage_unavailable",
            "patch_count": 0,
            "root_path": root_path,
            "keywords": list(keyword_tuple),
        }
        _emit_keyword(logger, "warning: current USD stage is unavailable")
        return report

    root = _resolve_root(stage, root_path)
    if root is None:
        report = {
            "applied": False,
            "reason": "root_prim_not_found",
            "patch_count": 0,
            "root_path": root_path,
            "keywords": list(keyword_tuple),
        }
        _emit_keyword(logger, f"warning: root prim is unavailable: {root_path}")
        return report

    instance_roots = _keyword_instance_roots(
        root,
        keywords=keyword_tuple,
        Usd=Usd,
    )
    instance_roots_found, deinstanced = _deinstance_keyword_roots(
        instance_roots,
        logger=logger,
    )
    # 取消 instanceable 后重新读取 root，避免使用失效的 composed prim 句柄。
    root = _resolve_root(stage, root_path)
    if root is None:
        report = {
            "applied": False,
            "reason": "root_prim_not_found_after_deinstance",
            "patch_count": 0,
            "root_path": root_path,
            "keywords": list(keyword_tuple),
            "instance_roots_found": instance_roots_found,
            "deinstanced_prim_paths": deinstanced,
        }
        _emit_keyword(logger, f"warning: root prim disappeared after deinstance: {root_path}")
        return report

    collisions = _find_collision_prims_by_keywords(
        root,
        keywords=keyword_tuple,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
    )
    patched: list[dict[str, Any]] = []
    for prim in collisions:
        before = _collision_info(prim)
        try:
            if not _has_api(prim, UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if (
                _is_mesh_collision_target(prim, UsdPhysics=UsdPhysics)
                and not _has_api(prim, UsdPhysics.MeshCollisionAPI)
            ):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            if not _has_api(prim, PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            _set_attribute(
                prim,
                "physics:collisionEnabled",
                Sdf.ValueTypeNames.Bool,
                True,
            )
            _set_attribute(
                prim,
                "physics:approximation",
                Sdf.ValueTypeNames.Token,
                str(approximation),
            )
            _set_attribute(
                prim,
                "physxCollision:contactOffset",
                Sdf.ValueTypeNames.Float,
                float(contact_offset),
            )
            _set_attribute(
                prim,
                "physxCollision:restOffset",
                Sdf.ValueTypeNames.Float,
                float(rest_offset),
            )
            _set_attribute(
                prim,
                "physics:restOffset",
                Sdf.ValueTypeNames.Float,
                float(rest_offset),
            )
        except Exception as exc:
            _emit_keyword(logger, f"warning: failed to patch {prim.GetPath()}: {exc}")
            continue
        after = _collision_info(prim)
        patched.append({"prim_path": str(prim.GetPath()), "before": before, "after": after})
        _emit_keyword(
            logger,
            f"patched {prim.GetPath()} "
            f"approximation={before['physics:approximation']}->{after['physics:approximation']} "
            f"contactOffset={before['physxCollision:contactOffset']}"
            f"->{after['physxCollision:contactOffset']} "
            f"physxRestOffset={before['physxCollision:restOffset']}"
            f"->{after['physxCollision:restOffset']} "
            f"physicsRestOffset={before['physics:restOffset']}"
            f"->{after['physics:restOffset']}",
        )

    if not patched:
        _emit_keyword(
            logger,
            (
                "warning: no keyword collision prim patched; "
                f"check root={root_path} keywords={list(keyword_tuple)}"
            ),
        )
    _emit_keyword(logger, f"patch complete: count={len(patched)}")
    return {
        "applied": bool(patched),
        "reason": None if patched else "keyword_collision_prim_not_found",
        "root_path": root_path,
        "keywords": list(keyword_tuple),
        "approximation": str(approximation),
        "contact_offset": float(contact_offset),
        "rest_offset": float(rest_offset),
        "instance_roots_found": instance_roots_found,
        "deinstanced_prim_paths": deinstanced,
        "patch_count": len(patched),
        "patched_prims": patched,
    }


def install_gripper_collision_patch_on_spawn(
    spawn_cfg: Any,
    *,
    enable_gripper_patch: bool = True,
    enable_keyword_patch: bool = True,
    robot_root: str = "/World/go2_x5",
    gripper_links: tuple[str, ...] = ("arm_link7", "arm_link8"),
    approximation: str = "convexDecomposition",
    contact_offset: float = 0.0002,
    rest_offset: float = 0.0,
    keyword_root_path: str = "/World",
    keywords: tuple[str, ...] = ("apple", "Apple"),
    keyword_approximation: str = "convexDecomposition",
    keyword_contact_offset: float = 0.001,
    keyword_rest_offset: float = 0.0,
    stage_getter: Callable[[], Any] | None = None,
) -> None:
    """包装 IsaacLab spawner，在机器人 prim 创建后、首次 reset 前执行一次 patch。"""

    if getattr(spawn_cfg, _PATCH_INSTALLED_ATTR, False):
        return
    original_spawn = spawn_cfg.func

    def _spawn_then_patch(
        prim_path: str,
        cfg: Any,
        translation: Any = None,
        orientation: Any = None,
        **kwargs: Any,
    ) -> Any:
        spawned = original_spawn(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        if stage_getter is None:
            from isaaclab.sim.utils.stage import get_current_stage

            current_stage = get_current_stage()
        else:
            current_stage = stage_getter()
        if enable_gripper_patch:
            report = patch_go2_x5_gripper_collision(
                robot_root=robot_root,
                gripper_links=gripper_links,
                approximation=approximation,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
                stage=current_stage,
            )
            setattr(cfg, _PATCH_REPORT_ATTR, report)
        if enable_keyword_patch:
            keyword_report = patch_collision_prims_by_keywords(
                root_path=keyword_root_path,
                keywords=keywords,
                approximation=keyword_approximation,
                contact_offset=keyword_contact_offset,
                rest_offset=keyword_rest_offset,
                stage=current_stage,
            )
            setattr(cfg, _KEYWORD_PATCH_REPORT_ATTR, keyword_report)
        # patch 后立即输出实际 authored 值，便于区分 cooking 不支持与属性未写入。
        if enable_gripper_patch:
            print_gripper_collision_info(
                robot_root=robot_root,
                gripper_links=gripper_links,
                stage=current_stage,
            )
        if enable_keyword_patch:
            print_collision_info_by_keywords(
                root_path=keyword_root_path,
                keywords=keywords,
                stage=current_stage,
            )
        return spawned

    spawn_cfg.func = _spawn_then_patch
    setattr(spawn_cfg, _PATCH_INSTALLED_ATTR, True)


def gripper_collision_patch_report(spawn_cfg: Any) -> dict[str, Any] | None:
    """读取 spawner 执行时保存的结构化报告。"""

    report = getattr(spawn_cfg, _PATCH_REPORT_ATTR, None)
    return dict(report) if isinstance(report, dict) else None


def keyword_collision_patch_report(spawn_cfg: Any) -> dict[str, Any] | None:
    """读取关键词碰撞修正的结构化报告；默认用于苹果资产。"""

    report = getattr(spawn_cfg, _KEYWORD_PATCH_REPORT_ATTR, None)
    return dict(report) if isinstance(report, dict) else None
