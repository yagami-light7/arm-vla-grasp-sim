"""Helpers for loading navigation scene geometry into Isaac Lab."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _top_level_child_name(prim_path: str) -> str | None:
    """Return the first child below /World for a prim path."""

    parts = [part for part in prim_path.strip("/").split("/") if part]
    if len(parts) < 2 or parts[0] != "World":
        return None
    return parts[1]


def write_collision_terrain_wrapper(scene_usd: str | Path, prim_path: str = "/World/scene_collision") -> Path:
    """Write a USD layer that references only one collision subtree from a scene."""

    scene_usd = Path(scene_usd).expanduser().resolve()
    digest = hashlib.sha256(f"{scene_usd}:{prim_path}".encode("utf-8")).hexdigest()[:12]
    wrapper = Path("/tmp") / f"go2_x5_collision_terrain_{digest}.usda"
    wrapper.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    defaultPrim = "scene_collision"',
                ")",
                "",
                'def Xform "scene_collision" (',
                f"    prepend references = @{scene_usd}@<{prim_path}>",
                ")",
                "{",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
