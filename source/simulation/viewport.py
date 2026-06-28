"""Isaac GUI viewport 的非物理显示配置。"""

from __future__ import annotations

from typing import Any


def candidate_stage_camera_paths(camera_prim_path: str) -> tuple[str, ...]:
    """返回 sublayer 与 reference 两种视觉场景中的相机候选路径。"""

    candidates = [str(camera_prim_path)]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path == "/World/Camera_main":
        candidates.append("/World/camera_main")
    # 当前多楼层 USD 使用 Camera0-8；旧场景 camera1/2/3 继续作为 fallback。
    candidates.extend(
        (
            "/World/camera0",
            "/World/camera1",
            "/World/camera2",
            "/World/camera3",
            "/World/Camera0",
            "/World/Camera1",
            "/World/Camera2",
            "/World/Camera3",
            "/World/Camera_main",
            "/World/camera_main",
            "/World/Camera_font",
            "/World/camera_font",
        )
    )
    if camera_prim_path.startswith("/World/"):
        suffix = camera_prim_path.removeprefix("/World/")
        candidates.extend(
            (
                f"/World/nav_visual_scene/{suffix}",
                f"/World/contact_visual_scene/{suffix}",
            )
        )
    for camera_name in (
        "Camera0",
        "Camera1",
        "Camera2",
        "Camera3",
        "camera0",
        "camera1",
        "camera2",
        "camera3",
        "Camera_main",
        "camera_main",
        "Camera_font",
        "camera_font",
    ):
        candidates.extend(
            (
                f"/World/gauss/{camera_name}",
                f"/World/nav_visual_scene/{camera_name}",
                f"/World/contact_visual_scene/{camera_name}",
            )
        )
    return tuple(dict.fromkeys(candidates))


def _discover_camera_paths(stage: Any, requested_path: str) -> tuple[str, ...]:
    """从已加载 stage 里补充相机候选，兼容 USD 内部不同命名层级。"""

    try:
        from pxr import UsdGeom
    except ImportError:
        return ()

    requested_name = requested_path.rsplit("/", 1)[-1].lower()
    preferred_tokens = (
        requested_name,
        "camera0",
        "camera_0",
        "camera 0",
        "camera1",
        "camera_1",
        "camera 1",
        "camera2",
        "camera_2",
        "camera 2",
        "camera3",
        "camera_3",
        "camera 3",
        "camera_font",
        "camerafont",
        "camera_front",
        "camerafront",
        "camera_main",
        "cameramain",
        "camera",
    )
    scored: list[tuple[int, str]] = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        name = path.rsplit("/", 1)[-1].lower()
        score = 100
        for index, token in enumerate(preferred_tokens):
            if token and token in name:
                score = index
                break
        scored.append((score, path))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(path for _, path in scored)


def _camera_prim_reports(stage: Any, paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """记录候选 camera 的 stage 状态，避免只凭 GUI 现象猜测加载结果。"""

    try:
        from pxr import UsdGeom
    except ImportError as exc:
        return (
            {
                "available": False,
                "reason": "usdgeom_unavailable",
                "error": str(exc),
            },
        )

    reports: list[dict[str, Any]] = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        report: dict[str, Any] = {
            "prim_path": path,
            "valid": bool(prim.IsValid()),
        }
        if prim.IsValid():
            report.update(
                {
                    "type_name": prim.GetTypeName(),
                    "is_camera": bool(prim.IsA(UsdGeom.Camera)),
                    "active": bool(prim.IsActive()),
                }
            )
            if prim.IsA(UsdGeom.Imageable):
                try:
                    report["visibility"] = UsdGeom.Imageable(prim).ComputeVisibility()
                except Exception as exc:
                    report["visibility_error"] = str(exc)
        reports.append(report)
    return tuple(reports)


def _try_apply_viewport_camera(viewport: Any, selected_path: str, sdf_path: Any) -> tuple[bool, tuple[dict[str, str], ...]]:
    """兼容不同 Isaac/Kit 版本的 viewport 相机切换 API。"""

    attempts: list[dict[str, str]] = []

    def _record(method: str, fn: Any) -> None:
        try:
            fn()
        except Exception as exc:
            attempts.append({"method": method, "status": "failed", "error": str(exc)})
        else:
            attempts.append({"method": method, "status": "ok"})

    _record("viewport.camera_path:sdf_path", lambda: setattr(viewport, "camera_path", sdf_path))
    _record("viewport.camera_path:string", lambda: setattr(viewport, "camera_path", selected_path))

    if hasattr(viewport, "set_active_camera"):
        _record("viewport.set_active_camera:sdf_path", lambda: viewport.set_active_camera(sdf_path))
        _record("viewport.set_active_camera:string", lambda: viewport.set_active_camera(selected_path))

    viewport_api = getattr(viewport, "viewport_api", None)
    if viewport_api is not None:
        _record(
            "viewport.viewport_api.camera_path:sdf_path",
            lambda: setattr(viewport_api, "camera_path", sdf_path),
        )
        _record(
            "viewport.viewport_api.camera_path:string",
            lambda: setattr(viewport_api, "camera_path", selected_path),
        )

    return any(item["status"] == "ok" for item in attempts), tuple(attempts)


def _stage_camera_eye_target(camera: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """按 baseline 方式把 USD camera 转成 Perspective eye/target。"""

    import math
    from pxr import Gf, Usd, UsdGeom

    prim = camera.GetPrim()
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    eye_vec = matrix.ExtractTranslation()
    forward_vec = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    if forward_vec.GetLength() <= 1.0e-9:
        forward_vec = Gf.Vec3d(1.0, 0.0, 0.0)
    forward_vec.Normalize()
    focus_distance = camera.GetFocusDistanceAttr().Get()
    try:
        focus_distance = float(focus_distance)
    except (TypeError, ValueError):
        focus_distance = 0.0
    if not math.isfinite(focus_distance) or focus_distance <= 0.1:
        focus_distance = 3.0
    target_vec = eye_vec + forward_vec * focus_distance
    return (
        (float(eye_vec[0]), float(eye_vec[1]), float(eye_vec[2])),
        (float(target_vec[0]), float(target_vec[1]), float(target_vec[2])),
    )


def _copy_stage_camera_to_perspective(
    *,
    camera: Any,
    selected_path: str,
    viewports: tuple[tuple[str, Any], ...],
    sdf_path_factory: Any,
) -> dict[str, Any]:
    """把 stage camera 视角复制到 Perspective，规避 GUI 不切 stage camera 的问题。"""

    report: dict[str, Any] = {
        "requested": True,
        "source_camera_prim_path": selected_path,
    }
    try:
        eye, target = _stage_camera_eye_target(camera)
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp")
        perspective_path = sdf_path_factory("/OmniverseKit_Persp")
        apply_attempts: list[dict[str, str]] = []
        for viewport_source, viewport in viewports:
            applied, attempts = _try_apply_viewport_camera(
                viewport,
                "/OmniverseKit_Persp",
                perspective_path,
            )
            for attempt in attempts:
                apply_attempts.append(
                    {
                        "viewport_source": viewport_source,
                        "applied": str(bool(applied)),
                        **attempt,
                    }
                )
        report.update(
            {
                "applied": True,
                "eye": eye,
                "target": target,
                "camera_prim_path": "/OmniverseKit_Persp",
                "apply_attempts": tuple(apply_attempts),
            }
        )
    except Exception as exc:
        report.update({"applied": False, "error": str(exc)})
    return report


def _candidate_viewports() -> tuple[tuple[str, Any], ...]:
    """按优先级返回可能的 GUI viewport，兼容 IsaacLab 创建窗口较晚的情况。"""

    candidates: list[tuple[str, Any]] = []
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            candidates.append(("active_viewport", viewport))
    except Exception:
        pass
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        for window_name in ("Viewport", "Isaac Sim Viewport"):
            viewport = get_viewport_from_window_name(window_name)
            if viewport is not None:
                candidates.append((f"window:{window_name}", viewport))
    except Exception:
        pass

    deduped: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for source, viewport in candidates:
        key = id(viewport)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((source, viewport))
    return tuple(deduped)


def configure_navigation_viewport(
    *,
    camera_prim_path: str = "/World/Camera0",
    hide_collision_visual: bool = True,
) -> dict[str, Any]:
    """配置导航 GUI；只修改可见性与 viewport，不改动物理属性。"""

    try:
        import omni.usd
        from pxr import Sdf, UsdGeom
    except ImportError as exc:
        return {
            "available": False,
            "reason": "viewport_dependencies_unavailable",
            "error": str(exc),
            "requested_camera_prim_path": camera_prim_path,
        }

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {
            "available": False,
            "reason": "stage_unavailable",
            "requested_camera_prim_path": camera_prim_path,
        }

    hidden_paths: list[str] = []
    hide_errors: list[dict[str, str]] = []
    if hide_collision_visual:
        for prim_path in ("/World/nav_collision/terrain", "/World/nav_collision"):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Imageable):
                continue
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden_paths.append(prim_path)
            except Exception as exc:
                hide_errors.append({"prim_path": prim_path, "error": str(exc)})

    discovered_camera_paths = _discover_camera_paths(stage, camera_prim_path)
    candidates = tuple(
        dict.fromkeys(
            (
                *candidate_stage_camera_paths(camera_prim_path),
                *discovered_camera_paths,
            )
        )
    )
    camera_report_paths = tuple(
        dict.fromkeys(
            (
                camera_prim_path,
                "/World/camera0",
                "/World/camera1",
                "/World/camera2",
                "/World/camera3",
                "/World/Camera0",
                "/World/Camera1",
                "/World/Camera2",
                "/World/Camera3",
                "/World/Camera_main",
                "/World/camera_main",
                "/World/Camera_font",
                "/World/camera_font",
                *candidates,
            )
        )
    )
    selected_path = None
    for candidate in candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid() and prim.IsA(UsdGeom.Camera):
            selected_path = candidate
            break

    report: dict[str, Any] = {
        "available": True,
        "requested_camera_prim_path": camera_prim_path,
        "camera_candidates": candidates,
        "discovered_camera_paths": discovered_camera_paths,
        "camera_prim_reports": _camera_prim_reports(stage, camera_report_paths),
        "selected_camera_prim_path": selected_path,
        "collision_visual_hidden_paths": tuple(hidden_paths),
        "collision_visual_hide_errors": tuple(hide_errors),
        "display_only": True,
        "physics_unchanged": True,
    }
    if selected_path is None:
        report.update(
            {
                "camera_applied": False,
                "reason": "camera_prim_unavailable",
            }
        )
        return report

    try:
        viewports = _candidate_viewports()
        if not viewports:
            report.update(
                {
                    "camera_applied": False,
                    "reason": "active_viewport_unavailable",
                }
            )
            return report
        sdf_path = Sdf.Path(selected_path)
        all_attempts: list[dict[str, str]] = []
        camera_applied = False
        applied_viewport_source = None
        readbacks: list[dict[str, str]] = []
        for viewport_source, viewport in viewports:
            applied, apply_attempts = _try_apply_viewport_camera(
                viewport,
                selected_path,
                sdf_path,
            )
            for attempt in apply_attempts:
                all_attempts.append({"viewport_source": viewport_source, **attempt})
            readbacks.append(
                {
                    "viewport_source": viewport_source,
                    "camera_path": str(getattr(viewport, "camera_path", "")),
                }
            )
            if applied and not camera_applied:
                camera_applied = True
                applied_viewport_source = viewport_source
        report["camera_apply_attempts"] = tuple(all_attempts)
        report["viewport_camera_path_readbacks"] = tuple(readbacks)
        report["applied_viewport_source"] = applied_viewport_source
        selected_camera = UsdGeom.Camera(stage.GetPrimAtPath(selected_path))
        perspective_sync = _copy_stage_camera_to_perspective(
            camera=selected_camera,
            selected_path=selected_path,
            viewports=viewports,
            sdf_path_factory=Sdf.Path,
        )
        report["perspective_camera_sync"] = perspective_sync
        camera_applied = camera_applied or bool(perspective_sync.get("applied"))
        try:
            import omni.kit.app

            # 切换 stage camera 后推进一次 Kit UI update，避免 GUI 仍显示旧 viewport。
            omni.kit.app.get_app().update()
            report["viewport_update_called"] = True
        except Exception as exc:
            report["viewport_update_called"] = False
            report["viewport_update_error"] = str(exc)
        report["camera_applied"] = camera_applied
        if not camera_applied:
            report["reason"] = "camera_apply_failed"
        return report
    except Exception as exc:
        report.update(
            {
                "camera_applied": False,
                "reason": "camera_apply_failed",
                "error": str(exc),
            }
        )
        return report


def set_active_camera(camera_prim_path: str) -> dict[str, Any]:
    """Switch the active viewport to an existing USD camera prim for display capture."""

    try:
        import omni.usd
        from pxr import Sdf, UsdGeom
    except ImportError as exc:
        return {
            "applied": False,
            "reason": "viewport_dependencies_unavailable",
            "requested_camera_prim_path": camera_prim_path,
            "error": str(exc),
        }

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {
            "applied": False,
            "reason": "stage_unavailable",
            "requested_camera_prim_path": camera_prim_path,
        }
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        return {
            "applied": False,
            "reason": "camera_prim_unavailable",
            "requested_camera_prim_path": camera_prim_path,
        }

    viewports = _candidate_viewports()
    if not viewports:
        return {
            "applied": False,
            "reason": "active_viewport_unavailable",
            "requested_camera_prim_path": camera_prim_path,
        }

    sdf_path = Sdf.Path(camera_prim_path)
    all_attempts: list[dict[str, str]] = []
    applied = False
    applied_viewport_source = None
    readbacks: list[dict[str, str]] = []
    for viewport_source, viewport in viewports:
        viewport_applied, attempts = _try_apply_viewport_camera(
            viewport,
            camera_prim_path,
            sdf_path,
        )
        for attempt in attempts:
            all_attempts.append({"viewport_source": viewport_source, **attempt})
        readbacks.append(
            {
                "viewport_source": viewport_source,
                "camera_path": str(getattr(viewport, "camera_path", "")),
            }
        )
        if viewport_applied and not applied:
            applied = True
            applied_viewport_source = viewport_source

    perspective_sync = _copy_stage_camera_to_perspective(
        camera=UsdGeom.Camera(prim),
        selected_path=camera_prim_path,
        viewports=viewports,
        sdf_path_factory=Sdf.Path,
    )
    applied = applied or bool(perspective_sync.get("applied"))
    try:
        import omni.kit.app

        omni.kit.app.get_app().update()
        viewport_update_called = True
        viewport_update_error = None
    except Exception as exc:
        viewport_update_called = False
        viewport_update_error = str(exc)
    report: dict[str, Any] = {
        "applied": applied,
        "requested_camera_prim_path": camera_prim_path,
        "selected_camera_prim_path": camera_prim_path,
        "camera_apply_attempts": tuple(all_attempts),
        "viewport_camera_path_readbacks": tuple(readbacks),
        "applied_viewport_source": applied_viewport_source,
        "perspective_camera_sync": perspective_sync,
        "viewport_update_called": viewport_update_called,
    }
    if viewport_update_error is not None:
        report["viewport_update_error"] = viewport_update_error
    if not applied:
        report["reason"] = "camera_apply_failed"
    return report


__all__ = [
    "candidate_stage_camera_paths",
    "configure_navigation_viewport",
    "set_active_camera",
]
