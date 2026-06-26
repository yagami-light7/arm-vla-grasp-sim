#!/usr/bin/env python3

"""验证 multifloor USDZ / collision USD / 主 USDA composition。"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USDZ = PROJECT_ROOT / "source/scene/multifloor/usdz/multifloor.usdz"
DEFAULT_COLLISION_USD = PROJECT_ROOT / "source/scene/multifloor/usd/multifloor_collision.usd"
DEFAULT_USDA = PROJECT_ROOT / "source/scene/multifloor/usda/multifloor.usda"


def main() -> int:
    args = _parse_args()
    usdz_path = _project_path(args.usdz)
    collision_usd = _project_path(args.collision_usd)
    usda_path = _project_path(args.usda)
    _validate_regular_file(collision_usd, "collision USD")
    _validate_regular_file(usda_path, "主 USDA")
    if args.visual_mode == "nurec" or args.require_usdz:
        _validate_regular_file(usdz_path, "USDZ")
        _validate_usdz(usdz_path)
    elif usdz_path.exists():
        _validate_regular_file(usdz_path, "USDZ")
        _validate_usdz(usdz_path)
    else:
        print(f"[INFO] collision-proxy 模式不强制检查 USDZ: {usdz_path}")
    _validate_stage(usda_path, visual_mode=args.visual_mode)
    print(f"[OK] Isaac Sim 应打开: {usda_path}")
    return 0


def _validate_regular_file(path: Path, label: str) -> None:
    print(f"[CHECK] {label}: {path}")
    if path.is_symlink():
        raise RuntimeError(f"{label} 不能是软链接: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    print(f"[OK] {label} exists size={path.stat().st_size} bytes")


def _validate_usdz(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
    print("[INFO] USDZ entries:")
    for name in names:
        print(f"  {name}")
    if "gauss.usda" not in names:
        raise RuntimeError("USDZ 缺少 gauss.usda")
    if not any(name.endswith(".nurec") for name in names):
        raise RuntimeError("USDZ 缺少 .nurec 文件")


def _validate_stage(usda_path: Path, *, visual_mode: str) -> None:
    try:
        from pxr import Usd
    except Exception as exc:
        raise RuntimeError("无法导入 pxr。请先激活 conda activate /data/conda_envs/sage") from exc

    stage = Usd.Stage.Open(os.fspath(usda_path))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open 失败: {usda_path}")
    for prim_path in ("/World", "/World/gauss", "/World/scene_collision", "/World/apple"):
        prim = stage.GetPrimAtPath(prim_path)
        print(f"[CHECK] prim {prim_path}: valid={prim.IsValid()} type={prim.GetTypeName() if prim.IsValid() else ''}")
        if not prim.IsValid():
            raise RuntimeError(f"Stage 缺少 prim: {prim_path}")

    nurec_prims = []
    mesh_prims_under_gauss = []
    print("[INFO] 包含 gauss/nurec/collision 的 prim:")
    for prim in stage.Traverse():
        text = str(prim.GetPath()).lower()
        if "gauss" in text or "nurec" in text or "collision" in text:
            print(f"  {prim.GetPath()} type={prim.GetTypeName()}")
        if prim.GetTypeName() == "OmniNuRecFieldAsset":
            nurec_prims.append(prim)
        if str(prim.GetPath()).startswith("/World/gauss") and prim.GetTypeName() == "Mesh":
            mesh_prims_under_gauss.append(prim)

    print("[INFO] Volume prim:")
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Volume":
            print(f"  {prim.GetPath()} type={prim.GetTypeName()}")

    if visual_mode == "collision_proxy":
        if nurec_prims:
            raise RuntimeError("collision-proxy 主场景不应包含 OmniNuRecFieldAsset")
        if not mesh_prims_under_gauss:
            raise RuntimeError("collision-proxy 主场景的 /World/gauss 下缺少 Mesh 代理")
        print(f"[OK] collision-proxy mesh count under /World/gauss: {len(mesh_prims_under_gauss)}")
    elif visual_mode == "nurec":
        if not nurec_prims:
            raise RuntimeError("NuRec 主场景缺少 OmniNuRecFieldAsset")
        _validate_nurec_gauss_transform(stage)
        print(f"[OK] NuRec field asset count: {len(nurec_prims)}")


def _validate_nurec_gauss_transform(stage: object) -> None:
    """确认 visual 层和 PCT/Isaac 场景约定使用同一个 Z 轴 180 度旋转。"""
    gauss = stage.GetPrimAtPath("/World/gauss")
    rotate_attr = gauss.GetAttribute("xformOp:rotateXYZ")
    if not rotate_attr.IsValid():
        raise RuntimeError("NuRec /World/gauss 缺少 xformOp:rotateXYZ")
    rotate = rotate_attr.Get()
    values = tuple(round(float(value), 6) for value in rotate)
    if values != (0.0, 0.0, 180.0):
        raise RuntimeError(f"NuRec /World/gauss rotateXYZ 应为 (0, 0, 180)，实际为 {values}")
    print(f"[OK] NuRec /World/gauss rotateXYZ={values}")


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 multifloor USD 资产。")
    parser.add_argument("--usdz", default=os.fspath(DEFAULT_USDZ), help="USDZ 路径。")
    parser.add_argument("--collision-usd", default=os.fspath(DEFAULT_COLLISION_USD), help="collision USD 路径。")
    parser.add_argument("--usda", default=os.fspath(DEFAULT_USDA), help="主 USDA 路径。")
    parser.add_argument(
        "--visual-mode",
        choices=("collision_proxy", "nurec"),
        default="nurec",
        help="按最终 NuRec stage 或显式 collision proxy 诊断 stage 校验。",
    )
    parser.add_argument("--require-usdz", action="store_true", help="即使是 collision_proxy 也强制校验 USDZ。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
