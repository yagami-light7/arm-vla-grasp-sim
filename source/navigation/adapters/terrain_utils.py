"""Helpers for loading navigation scene geometry into Isaac Lab."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


def _top_level_child_name(prim_path: str) -> str | None:
    """Return the first child below /World for a prim path."""

    parts = [part for part in prim_path.strip("/").split("/") if part]
    if len(parts) < 2 or parts[0] != "World":
        return None
    return parts[1]


def _yinluyuan_f2_floor_proxy_lines() -> list[str]:
    """生成 Yinluyuan 二楼导航走廊的不可见平滑碰撞体。"""

    route = (
        ((2.921488571, 5.490738525), (1.921488571, 4.456363678)),
        ((1.921488571, 4.456363678), (0.321488571, 4.456363678)),
        ((0.321488571, 4.456363678), (0.321488571, -0.143636322)),
        ((0.321488571, -0.143636322), (0.40, -0.10)),
    )
    width_m = 0.65
    endpoint_padding_m = 0.40
    thickness_m = 0.04
    # 高于原 mesh 约 17--19 mm，覆盖 PhysX contact offset 对裂缝边缘的提前接触。
    top_z_m = 3.05
    lines: list[str] = []
    for index, (start, end) in enumerate(route):
        # scene_collision 根节点带 Z 轴 180 度旋转，子节点需先写为 PLY 局部坐标。
        local_start = (-float(start[0]), -float(start[1]))
        local_end = (-float(end[0]), -float(end[1]))
        dx = local_end[0] - local_start[0]
        dy = local_end[1] - local_start[1]
        length = math.hypot(dx, dy) + 2.0 * endpoint_padding_m
        yaw = math.atan2(dy, dx)
        half_yaw = yaw * 0.5
        center_x = (local_start[0] + local_end[0]) * 0.5
        center_y = (local_start[1] + local_end[1]) * 0.5
        center_z = top_z_m - thickness_m * 0.5
        lines.extend(
            [
                f'    def Cube "f2_floor_proxy_{index:02d}" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
                "    )",
                "    {",
                "        bool physics:collisionEnabled = 1",
                "        float physxCollision:contactOffset = 0.005",
                "        float physxCollision:restOffset = 0",
                "        double size = 2",
                '        token visibility = "invisible"',
                (
                    "        quatf xformOp:orient = "
                    f"({math.cos(half_yaw):.9f}, 0, 0, {math.sin(half_yaw):.9f})"
                ),
                (
                    "        float3 xformOp:scale = "
                    f"({length * 0.5:.9f}, {width_m * 0.5:.9f}, "
                    f"{thickness_m * 0.5:.9f})"
                ),
                (
                    "        double3 xformOp:translate = "
                    f"({center_x:.9f}, {center_y:.9f}, {center_z:.9f})"
                ),
                (
                    '        uniform token[] xformOpOrder = '
                    '["xformOp:translate", "xformOp:orient", "xformOp:scale"]'
                ),
                "    }",
                "",
            ]
        )
    return lines


def write_collision_terrain_wrapper(
    scene_usd: str | Path,
    prim_path: str = "/World/scene_collision",
    *,
    floor_proxy_profile: str | None = None,
) -> Path:
    """写入只引用碰撞子树、并可选附加平滑通行面的 USD layer。"""

    scene_usd = Path(scene_usd).expanduser().resolve()
    supported_profiles = {None, "yinluyuan_f2"}
    if floor_proxy_profile not in supported_profiles:
        raise ValueError(f"未知碰撞地面代理 profile: {floor_proxy_profile}")
    digest = hashlib.sha256(
        f"{scene_usd}:{prim_path}:{floor_proxy_profile}".encode("utf-8")
    ).hexdigest()[:12]
    wrapper = Path("/tmp") / f"go2_x5_collision_terrain_{digest}.usda"
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "scene_collision"',
        ")",
        "",
        'def Xform "scene_collision" (',
        f"    prepend references = @{scene_usd}@<{prim_path}>",
        ")",
        "{",
    ]
    if floor_proxy_profile == "yinluyuan_f2":
        lines.extend(_yinluyuan_f2_floor_proxy_lines())
    lines.extend(["}", ""])
    wrapper.write_text("\n".join(lines), encoding="utf-8")
    return wrapper


def write_visual_prim_wrapper(
    scene_usd: str | Path,
    prim_path: str = "/World/gauss",
    *,
    excluded_prim_paths: list[str] | tuple[str, ...] = (),
) -> Path:
    """Write a USD layer that references a scene for visual debugging.

    Navigation uses a collision-only terrain by default. For 3DGS scenes, the
    visual and collision roots can each carry transforms authored in the parent
    scene layer. Referencing the complete scene default prim preserves that
    composition path, while local stronger opinions deactivate collision and
    robot/object prims that should not be duplicated in the navigation runtime.
    """

    scene_usd = Path(scene_usd).expanduser().resolve()
    excluded_children = {
        _top_level_child_name(path)
        for path in (
            prim_path,
            *excluded_prim_paths,
        )
    }
    visual_child = _top_level_child_name(prim_path)
    excluded_children.discard(None)
    excluded_children.discard(visual_child)
    digest = hashlib.sha256(
        f"visual-scene:{scene_usd}:{prim_path}:{sorted(excluded_children)}".encode("utf-8")
    ).hexdigest()[:12]
    wrapper = Path("/tmp") / f"go2_x5_visual_scene_{digest}.usda"
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "visual_scene"',
        ")",
        "",
        'def Xform "visual_scene" (',
        f"    prepend references = @{scene_usd}@",
        ")",
        "{",
    ]
    for child_name in sorted(excluded_children):
        lines.extend(
            [
                f'    over "{child_name}" (',
                "        active = false",
                "    )",
                "    {",
                "    }",
                "",
            ]
        )
    lines.extend(["}", ""])
    wrapper.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return wrapper


def write_visual_sublayer_wrapper(
    scene_usd: str | Path,
    prim_path: str = "/World/gauss",
    *,
    excluded_prim_paths: list[str] | tuple[str, ...] = (),
    include_visual_prim: bool = True,
) -> Path:
    """Write a stronger layer that sublayers the complete scene for display.

    Referencing a USD file as an asset composes only its default prim. SAGE-3D
    scenes can also carry render settings outside the default prim, so this
    wrapper is intended to be added as a stage sublayer. It preserves the same
    root-level scene composition as opening the scene directly in Isaac Sim and
    deactivates selected top-level scene prims to avoid duplicate physics.
    """

    scene_usd = Path(scene_usd).expanduser().resolve()
    excluded_children = {
        _top_level_child_name(path)
        for path in (
            prim_path,
            *excluded_prim_paths,
        )
    }
    visual_child = _top_level_child_name(prim_path)
    excluded_children.discard(None)
    if include_visual_prim:
        excluded_children.discard(visual_child)
    digest = hashlib.sha256(
        (
            f"visual-sublayer:{scene_usd}:{prim_path}:{include_visual_prim}:"
            f"{sorted(excluded_children)}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    wrapper = Path("/tmp") / f"go2_x5_visual_sublayer_{digest}.usda"
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    subLayers = [",
        f"        @{scene_usd}@",
        "    ]",
        ")",
        "",
        'over "World"',
        "{",
    ]
    for child_name in sorted(excluded_children):
        lines.extend(
            [
                f'    over "{child_name}" (',
                "        active = false",
                "    )",
                "    {",
                "    }",
                "",
            ]
        )
    lines.extend(["}", ""])
    wrapper.write_text("\n".join(lines), encoding="utf-8")
    return wrapper
