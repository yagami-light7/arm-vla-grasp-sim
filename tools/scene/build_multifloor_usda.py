#!/usr/bin/env python3

"""生成 multifloor Isaac Sim 主 USDA。"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = Path("~/workspace/SAGE-3D_Official/Data/template.usda").expanduser()
DEFAULT_USDZ = PROJECT_ROOT / "source/scene/multifloor/usdz/multifloor.usdz"
DEFAULT_COLLISION_USD = PROJECT_ROOT / "source/scene/multifloor/usd/multifloor_collision.usd"
DEFAULT_OUTPUT_USDA = PROJECT_ROOT / "source/scene/multifloor/usda/multifloor.usda"
DEFAULT_UNPACK_DIR = PROJECT_ROOT / "source/scene/multifloor/usdz/multifloor_unpack"
APPLE_REFERENCE = "../../objects/apple/MesaTask-10K/MesaTask_Assets/apple/0176be079c2449e7aaebfb652910a854/usd/0176be079c2449e7aaebfb652910a854.usd"
GAUSS_ALIGNED_ROTATE_XYZ = "        double3 xformOp:rotateXYZ = (0, 0, 180)"


def main() -> int:
    args = _parse_args()
    template_path = _project_path(args.template)
    usdz_path = _project_path(args.usdz)
    collision_usd = _project_path(args.collision_usd)
    output_usda = _project_path(args.output_usda)
    unpack_dir = _project_path(args.unpack_dir)

    _require_regular_file(collision_usd, "collision USD")
    if args.visual_mode == "nurec":
        _require_regular_file(usdz_path, "USDZ")
        _require_regular_file(template_path, "SAGE template")
    if output_usda.exists() and not args.force:
        raise FileExistsError(f"输出已存在，覆盖请传 --force: {output_usda}")

    if args.visual_mode == "collision_proxy":
        _write_collision_proxy_stage(output_usda, collision_usd)
        return 0

    _write_nurec_stage(
        template_path=template_path,
        usdz_path=usdz_path,
        collision_usd=collision_usd,
        output_usda=output_usda,
        unpack_dir=unpack_dir,
        force=args.force,
        force_unpack=args.force_unpack,
    )
    return 0


def _write_collision_proxy_stage(output_usda: Path, collision_usd: Path) -> None:
    rel_collision = os.path.relpath(collision_usd, start=output_usda.parent).replace(os.sep, "/")
    content = _collision_proxy_stage_text(rel_collision)
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    output_usda.write_text(content, encoding="utf-8")
    print(f"[OK] collision-proxy USDA 已生成: {output_usda}")
    print(f"[OK] /World/gauss proxy payload: {rel_collision}")
    print(f"[OK] /World/scene_collision payload: {rel_collision}")


def _write_nurec_stage(
    *,
    template_path: Path,
    usdz_path: Path,
    collision_usd: Path,
    output_usda: Path,
    unpack_dir: Path,
    force: bool,
    force_unpack: bool,
) -> None:
    rel_usdz = os.path.relpath(usdz_path, start=output_usda.parent).replace(os.sep, "/")
    rel_collision = os.path.relpath(collision_usd, start=output_usda.parent).replace(os.sep, "/")
    visual_ref = f"{rel_usdz}[gauss.usda]"
    if force_unpack or not _can_open_usdz_member(usdz_path):
        _unpack_usdz(usdz_path, unpack_dir, force=force)
        visual_ref = os.path.relpath(unpack_dir / "gauss.usda", start=output_usda.parent).replace(os.sep, "/")

    content = template_path.read_text(encoding="utf-8")
    content = content.replace("839920", "multifloor")
    content = content.replace("@usdz_root[gauss.usda]@", f"@{visual_ref}@")
    content = content.replace("@collision_root@", f"@{rel_collision}@")
    content = content.replace('string authoring_layer = "./839920.usda"', f'string authoring_layer = "./{output_usda.name}"')
    content = _align_gauss_transform(content)
    content = _insert_task_apple(content)

    output_usda.parent.mkdir(parents=True, exist_ok=True)
    output_usda.write_text(content, encoding="utf-8")
    print(f"[OK] NuRec USDA 已生成: {output_usda}")
    print(f"[OK] gauss reference: {visual_ref}")
    print("[OK] collision payload: ../usd/multifloor_collision.usd")


def _align_gauss_transform(content: str) -> str:
    """把 NuRec visual 层对齐到 PCT/Isaac 约定的 Z 轴 180 度旋转。"""
    old_line = "        double3 xformOp:rotateXYZ = (-90, 0, 0)"
    if old_line not in content:
        raise RuntimeError("template 中未找到 /World/gauss 的 rotateXYZ，无法对齐 visual transform")
    return content.replace(old_line, GAUSS_ALIGNED_ROTATE_XYZ, 1)


def _insert_task_apple(content: str) -> str:
    apple_block = _apple_block()
    marker = '\n}\n\ndef "Render"'
    if marker not in content:
        raise RuntimeError("template 中未找到 World 结束位置，无法插入 apple 引用")
    return content.replace(marker, f"{apple_block}{marker}", 1)


def _apple_block() -> str:
    return f"""

    def Xform "apple" (
        prepend references = @{APPLE_REFERENCE}@
    )
    {{
        token visibility = "inherited"
        quatf xformOp:orient = (0.99743015, -0.021866165, -0.06822734, -0.00007213679)
        float3 xformOp:scale = (0.08, 0.08, 0.08)
        double3 xformOp:translate = (1.3, 2, 0.81653)
        float xformOp:rotateX:unitsResolve = 90
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOp:rotateX:unitsResolve"]
    }}
"""


def _collision_proxy_stage_text(rel_collision: str) -> str:
    apple_block = _apple_block()
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    token visibility = "inherited"

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

    def "gauss" (
        prepend payload = @{rel_collision}@
    )
    {{
        token visibility = "inherited"
        quatf xformOp:orient = (6.123234e-17, 0, 0, 1)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def "scene_collision" (
        prepend payload = @{rel_collision}@
    )
    {{
        token visibility = "invisible"
        quatf xformOp:orient = (6.123234e-17, 0, 0, 1)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
{apple_block}}}
"""


def _can_open_usdz_member(usdz_path: Path) -> bool:
    try:
        from pxr import Sdf
    except Exception:
        return False
    layer = Sdf.Layer.FindOrOpen(f"{os.fspath(usdz_path)}[gauss.usda]")
    return layer is not None


def _unpack_usdz(usdz_path: Path, unpack_dir: Path, *, force: bool) -> None:
    if unpack_dir.exists():
        if not force:
            raise FileExistsError(f"解包目录已存在，覆盖请传 --force: {unpack_dir}")
        shutil.rmtree(unpack_dir)
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(usdz_path, "r") as archive:
        archive.extractall(unpack_dir)
    gauss = unpack_dir / "gauss.usda"
    if not gauss.is_file():
        raise RuntimeError(f"USDZ 解包后缺少 gauss.usda: {unpack_dir}")
    print(f"[INFO] USDZ resolver 不支持内部引用，已解包到: {unpack_dir}")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} 不能是软链接: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 multifloor Isaac Sim USDA 主场景。")
    parser.add_argument("--template", default=os.fspath(DEFAULT_TEMPLATE), help="SAGE-3D template.usda。")
    parser.add_argument("--usdz", default=os.fspath(DEFAULT_USDZ), help="NuRec USDZ。")
    parser.add_argument("--collision-usd", default=os.fspath(DEFAULT_COLLISION_USD), help="collision USD。")
    parser.add_argument("--output-usda", default=os.fspath(DEFAULT_OUTPUT_USDA), help="输出主 USDA。")
    parser.add_argument("--unpack-dir", default=os.fspath(DEFAULT_UNPACK_DIR), help="USDZ fallback 解包目录。")
    parser.add_argument(
        "--visual-mode",
        choices=("collision_proxy", "nurec"),
        default="nurec",
        help="nurec 生成最终高质量视觉入口；collision_proxy 仅用于诊断渲染问题。",
    )
    parser.add_argument("--force-unpack", action="store_true", help="强制使用解包 gauss.usda 引用。")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
