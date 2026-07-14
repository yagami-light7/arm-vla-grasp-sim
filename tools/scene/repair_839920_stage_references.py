#!/usr/bin/env python3

"""修复 839920 单楼层 stage 的本仓库相对引用。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE = PROJECT_ROOT / "source/scene/839920/839920_go2_x5.usd"

APPLE_USD = "../objects/apple/MesaTask-10K/MesaTask_Assets/apple/0176be079c2449e7aaebfb652910a854/usd/0176be079c2449e7aaebfb652910a854.usd"
ORANGE_USD = "../objects/orange/MesaTask-10K/MesaTask_Assets/orange/0896dc31d5154c97aa3f24e8ec1277aa/usd/0896dc31d5154c97aa3f24e8ec1277aa.usd"
BOTTLE_USD = "../objects/bottle/MesaTask-10K/MesaTask_Assets/body-care_products/ec67d7141333464ca1061320452f06a2/usd/ec67d7141333464ca1061320452f06a2.usd"


def main() -> int:
    args = _parse_args()
    stage_path = _project_path(args.stage)
    if not stage_path.is_file():
        raise FileNotFoundError(f"stage 不存在: {stage_path}")

    from pxr import Usd

    stage = Usd.Stage.Open(os.fspath(stage_path))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open 失败: {stage_path}")
    stage.SetEditTarget(stage.GetRootLayer())

    _set_reference(stage, "/World/gauss", "assets/839920.usdz[gauss.usda]")
    _set_payload(stage, "/World/scene_collision", "collision/839920_collision.usd")
    _set_reference(stage, "/World/go2_x5", "../../robot/go2_x5/urdf/go2_x5/go2_x5.usd")
    for prim_path in ("/World/apple", "/World/apple_01", "/World/apple_02", "/World/apple_03"):
        _set_reference(stage, prim_path, APPLE_USD)
    for prim_path in ("/World/orange", "/World/orange_01"):
        _set_reference(stage, prim_path, ORANGE_USD)
    _set_reference(stage, "/World/bottle", BOTTLE_USD)

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"保存 stage 失败: {stage_path}")
    print(f"[OK] 已修复 839920 stage 引用: {stage_path}")
    return 0


def _set_reference(stage, prim_path: str, asset_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"stage 缺少 prim: {prim_path}")
    references = prim.GetReferences()
    references.ClearReferences()
    references.AddReference(asset_path)
    print(f"[REF] {prim_path} -> {asset_path}")


def _set_payload(stage, prim_path: str, asset_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"stage 缺少 prim: {prim_path}")
    payloads = prim.GetPayloads()
    payloads.ClearPayloads()
    payloads.AddPayload(asset_path)
    print(f"[PAYLOAD] {prim_path} -> {asset_path}")


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复 839920 单楼层 USD stage 引用。")
    parser.add_argument("--stage", default=os.fspath(DEFAULT_STAGE), help="839920_go2_x5.usd 路径。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
