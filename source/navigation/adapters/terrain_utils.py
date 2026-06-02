"""Helpers for loading navigation collision geometry into Isaac Lab."""

from __future__ import annotations

import hashlib
from pathlib import Path


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
