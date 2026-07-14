#!/usr/bin/env python3

"""生成只包含 multifloor collision 的诊断 USDA。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLISION_USD = PROJECT_ROOT / "source/scene/multifloor/usd/multifloor_collision.usd"
DEFAULT_OUTPUT_USDA = PROJECT_ROOT / "source/scene/multifloor/usda/multifloor_collision_only.usda"


def main() -> int:
    args = _parse_args()
    collision_usd = _project_path(args.collision_usd)
    output_usda = _project_path(args.output_usda)
    if not collision_usd.is_file():
        raise FileNotFoundError(f"collision USD 不存在: {collision_usd}")
    if output_usda.exists() and not args.force:
        raise FileExistsError(f"输出已存在，覆盖请传 --force: {output_usda}")

    rel_collision = os.path.relpath(collision_usd, start=output_usda.parent).replace(os.sep, "/")
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    output_usda.write_text(_stage_text(rel_collision), encoding="utf-8")
    print(f"[OK] collision-only USDA 已生成: {output_usda}")
    return 0


def _stage_text(collision_ref: str) -> str:
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def DomeLight "DomeLight"
    {{
        float inputs:intensity = 700
        token visibility = "inherited"
    }}

    def Camera "MySensorCamera"
    {{
        float focalLength = 12
        float verticalAperture = 20.955
        token visibility = "inherited"
        quatd xformOp:orient = (0.36294232640931545, 0.19249779216725593, 0.4271882865093521, 0.8054363013798979)
        double3 xformOp:translate = (2.064822018987709, 3.0484421384955835, 2.4044287216166205)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}

    def "scene_collision" (
        prepend payload = @{collision_ref}@
    )
    {{
        token visibility = "inherited"
        quatf xformOp:orient = (6.123234e-17, 0, 0, 1)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
}}
"""


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 multifloor collision-only 诊断 USDA。")
    parser.add_argument("--collision-usd", default=os.fspath(DEFAULT_COLLISION_USD), help="collision USD 路径。")
    parser.add_argument("--output-usda", default=os.fspath(DEFAULT_OUTPUT_USDA), help="输出 USDA 路径。")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
