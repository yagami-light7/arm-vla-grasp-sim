"""从现有 task raw config 解析场景运行时 prim 约定。"""

from __future__ import annotations

from typing import Any


_SUPPORTED_COLLISION_FLOOR_PROXY_PROFILES = {None, "yinluyuan_f2"}


def _absolute_prim_path(value: Any, *, field_name: str) -> str:
    """严格要求绝对 USD prim path，避免 wrapper 引用错误子树。"""

    path = str(value or "").strip()
    if not path.startswith("/") or path == "/" or "//" in path:
        raise ValueError(f"{field_name} 必须是绝对 USD prim path: {value!r}")
    return path.rstrip("/")


def resolve_scene_runtime_settings(
    raw_task: dict[str, Any],
    *,
    default_collision_prim_path: str,
    default_visual_prim_path: str,
    default_collision_floor_proxy_profile: str | None,
) -> dict[str, Any]:
    """解析 task.scene_runtime；未配置时完整保留旧 runtime 默认值。"""

    raw_scene_runtime = raw_task.get("scene_runtime")
    if raw_scene_runtime is None:
        raw_scene_runtime = {}
    if not isinstance(raw_scene_runtime, dict):
        raise ValueError("task.scene_runtime 必须是对象")

    collision_prim_path = _absolute_prim_path(
        raw_scene_runtime.get(
            "collision_prim_path",
            default_collision_prim_path,
        ),
        field_name="task.scene_runtime.collision_prim_path",
    )
    visual_prim_path = _absolute_prim_path(
        raw_scene_runtime.get(
            "visual_prim_path",
            default_visual_prim_path,
        ),
        field_name="task.scene_runtime.visual_prim_path",
    )
    if "collision_floor_proxy_profile" in raw_scene_runtime:
        floor_proxy_profile = raw_scene_runtime["collision_floor_proxy_profile"]
    else:
        floor_proxy_profile = default_collision_floor_proxy_profile
    if floor_proxy_profile not in _SUPPORTED_COLLISION_FLOOR_PROXY_PROFILES:
        raise ValueError(
            "task.scene_runtime.collision_floor_proxy_profile 不受支持: "
            f"{floor_proxy_profile!r}"
        )

    return {
        "collision_prim_path": collision_prim_path,
        "visual_prim_path": visual_prim_path,
        "collision_floor_proxy_profile": floor_proxy_profile,
        "source": "task.scene_runtime" if raw_scene_runtime else "runtime_defaults",
        "task_override_present": bool(raw_scene_runtime),
    }
