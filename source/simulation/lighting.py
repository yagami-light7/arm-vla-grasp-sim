"""Isaac stage 灯光模式切换工具。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_LIGHT_TYPE_NAMES = {
    "CylinderLight",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "GeometryLight",
    "PortalLight",
    "RectLight",
    "SphereLight",
}


def resolve_scene_light_mode(
    requested_mode: str,
    *,
    scene_visual_enabled: bool,
) -> str:
    """Resolve the CLI-facing lighting mode to a concrete runtime mode.

    ``auto`` follows the visual payload: the complete authored scene uses its
    stage lights, while collision-only navigation keeps the camera-mounted
    fill lights that make diagnostic images readable.
    """

    normalized_mode = str(requested_mode).lower()
    if normalized_mode == "auto":
        return "stage" if scene_visual_enabled else "camera"
    if normalized_mode not in {"camera", "stage"}:
        raise ValueError("场景灯光模式必须是 auto、camera 或 stage。")
    return normalized_mode


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(f"[scene-lighting] {message}")


def _light_visibility(prim: Any, UsdGeom: Any) -> str | None:
    try:
        return str(UsdGeom.Imageable(prim).ComputeVisibility())
    except Exception:
        return None


def _set_visibility(prim: Any, UsdGeom: Any, *, visible: bool) -> dict[str, Any]:
    """设置 prim 可见性，并返回修改前后的只读诊断。"""

    before = _light_visibility(prim, UsdGeom)
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    after = _light_visibility(prim, UsdGeom)
    return {"before": before, "after": after}


def _is_light_prim(prim: Any) -> bool:
    return str(prim.GetTypeName()) in _LIGHT_TYPE_NAMES


def _is_camera_light_path(path: str, camera_light_name: str) -> bool:
    suffix = "/" + camera_light_name
    return path.endswith(suffix) or (suffix + "/") in path


def _iter_world_cameras(stage: Any, Usd: Any, UsdGeom: Any) -> tuple[Any, ...]:
    """只枚举用户 stage 中的摄像机，跳过 Kit 内置视角 prim。"""

    root = stage.GetPrimAtPath("/World")
    if not root or not root.IsValid():
        return ()
    cameras: list[Any] = []
    for prim in Usd.PrimRange(root):
        try:
            if prim.IsA(UsdGeom.Camera):
                cameras.append(prim)
        except Exception:
            if str(prim.GetTypeName()) == "Camera":
                cameras.append(prim)
    return tuple(cameras)


def configure_scene_lighting(
    *,
    stage: Any,
    mode: str = "camera",
    camera_light_name: str = "camera_light",
    camera_light_intensity: float = 3500.0,
    camera_light_radius: float = 2.0,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """在当前 USD stage 上配置 camera light 或恢复 stage light。"""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"camera", "stage"}:
        raise ValueError("场景灯光模式必须是 camera 或 stage。")
    if stage is None:
        return {
            "applied": False,
            "reason": "usd_stage_unavailable",
            "mode": normalized_mode,
        }

    camera_paths: list[str] = []
    camera_light_paths: list[str] = []
    camera_light_updates: list[dict[str, Any]] = []
    if normalized_mode == "camera":
        for camera_prim in _iter_world_cameras(stage, Usd, UsdGeom):
            camera_path = str(camera_prim.GetPath())
            camera_paths.append(camera_path)
            light_path = Sdf.Path(camera_path).AppendChild(camera_light_name)
            light = UsdLux.SphereLight.Define(stage, light_path)
            light.CreateIntensityAttr().Set(float(camera_light_intensity))
            light.CreateRadiusAttr().Set(float(camera_light_radius))
            light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            visibility = _set_visibility(light.GetPrim(), UsdGeom, visible=True)
            path_text = str(light_path)
            camera_light_paths.append(path_text)
            camera_light_updates.append(
                {
                    "camera_path": camera_path,
                    "light_path": path_text,
                    "visibility": visibility,
                }
            )
    else:
        camera_paths = [str(prim.GetPath()) for prim in _iter_world_cameras(stage, Usd, UsdGeom)]

    stage_light_updates: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if not _is_light_prim(prim):
            continue
        path = str(prim.GetPath())
        is_camera_light = _is_camera_light_path(path, camera_light_name)
        visible = is_camera_light if normalized_mode == "camera" else not is_camera_light
        visibility = _set_visibility(prim, UsdGeom, visible=visible)
        stage_light_updates.append(
            {
                "prim_path": path,
                "type": str(prim.GetTypeName()),
                "camera_light": is_camera_light,
                "visibility": visibility,
            }
        )

    stage_light_count = sum(1 for item in stage_light_updates if not item["camera_light"])
    camera_light_count = sum(1 for item in stage_light_updates if item["camera_light"])
    if normalized_mode == "camera" and not camera_light_paths:
        _emit(logger, "警告：未找到可挂载 camera light 的 /World 相机 prim")
    _emit(
        logger,
        (
            f"mode={normalized_mode} cameras={len(camera_paths)} "
            f"camera_lights={camera_light_count} stage_lights={stage_light_count}"
        ),
    )
    return {
        "applied": True,
        "reason": None,
        "mode": normalized_mode,
        "camera_paths": camera_paths,
        "camera_light_name": camera_light_name,
        "camera_light_intensity": float(camera_light_intensity),
        "camera_light_radius": float(camera_light_radius),
        "camera_light_paths": camera_light_paths,
        "camera_light_count": camera_light_count,
        "stage_light_count": stage_light_count,
        "camera_light_updates": camera_light_updates,
        "light_visibility_updates": stage_light_updates,
    }


__all__ = ["configure_scene_lighting", "resolve_scene_light_mode"]
