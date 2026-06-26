#!/usr/bin/env python3

"""检查 multifloor 场景资产，并按需复制为 PCT README 兼容命名。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from source.scene.gaussian_splat_ply import describe_ply, is_gaussian_splat_ply, parse_ply_header


DEFAULT_SCENE_DIR = REPO_ROOT / "source/scene/multifloor"
DEFAULT_PCT_ROOT = REPO_ROOT / "external/PCT"
DEFAULT_REPORT = REPO_ROOT / "outputs/pct_mutifloor_asset_report.json"


def main() -> int:
    args = _parse_args()
    scene_dir = _project_path(args.scene_dir)
    pct_root = _project_path(args.pct_root)
    report_path = _project_path(args.report_json)

    collision_ply = _project_path(args.collision_ply)
    visual_ply = _project_path(args.visual_ply)
    collision_usd = _project_path(args.collision_usd)
    stage_usda = _project_path(args.stage_usda)
    usdz = _project_path(args.usdz)

    _validate_scene_assets(
        collision_ply=collision_ply,
        visual_ply=visual_ply,
        collision_usd=collision_usd,
        stage_usda=stage_usda,
        usdz=usdz,
    )
    copied: list[dict[str, str]] = []
    if args.copy_assets:
        _copy_pct_readme_assets(
            pct_root=pct_root,
            collision_ply=collision_ply,
            visual_ply=visual_ply,
            collision_usd=collision_usd,
            stage_usda=stage_usda,
            force=args.force,
            copied=copied,
        )

    report = {
        "scene_dir": str(scene_dir),
        "pct_root": str(pct_root),
        "copy_assets": bool(args.copy_assets),
        "copied": copied,
        "assets": {
            "collision_ply": describe_ply(collision_ply),
            "visual_ply": describe_ply(visual_ply),
            "collision_usd": _file_status(collision_usd),
            "stage_usda": _file_status(stage_usda),
            "usdz": _file_status(usdz),
        },
        "pct_readme_mapping": {
            "mutifloor/3dgs_collision_cropped.ply": "source/scene/multifloor/ply/3dgs_collision.ply",
            "mutifloor/3dgs_cropped.ply": "source/scene/multifloor/ply/3dgs_visual.ply",
            "mutifloor/mutifloor_collision_cropped.usd": "source/scene/multifloor/usd/multifloor_collision.usd",
            "mutifloor/mutifloor_cropped.usda": "source/scene/multifloor/usda/multifloor.usda",
        },
        "note": "默认只检查当前布局；如需外部 PCT README 兼容目录，显式传 --copy-assets。脚本不创建软链接。",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 multifloor 资产布局，并可复制出 PCT README 兼容命名。",
    )
    parser.add_argument("--scene-dir", default=os.fspath(DEFAULT_SCENE_DIR))
    parser.add_argument("--pct-root", default=os.fspath(DEFAULT_PCT_ROOT))
    parser.add_argument("--collision-ply", default=os.fspath(DEFAULT_SCENE_DIR / "ply/3dgs_collision.ply"))
    parser.add_argument("--visual-ply", default=os.fspath(DEFAULT_SCENE_DIR / "ply/3dgs_visual.ply"))
    parser.add_argument("--collision-usd", default=os.fspath(DEFAULT_SCENE_DIR / "usd/multifloor_collision.usd"))
    parser.add_argument("--stage-usda", default=os.fspath(DEFAULT_SCENE_DIR / "usda/multifloor.usda"))
    parser.add_argument("--usdz", default=os.fspath(DEFAULT_SCENE_DIR / "usdz/multifloor.usdz"))
    parser.add_argument("--report-json", default=os.fspath(DEFAULT_REPORT))
    parser.add_argument("--copy-assets", action="store_true", help="复制资产到 external/PCT/mutifloor 兼容命名。")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的复制目标。")
    return parser.parse_args()


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _validate_scene_assets(
    *,
    collision_ply: Path,
    visual_ply: Path,
    collision_usd: Path,
    stage_usda: Path,
    usdz: Path,
) -> None:
    _require_regular_file(collision_ply, "collision PLY")
    _require_regular_file(visual_ply, "visual PLY")
    _require_regular_file(collision_usd, "collision USD")
    _require_regular_file(stage_usda, "主 USDA")
    _require_regular_file(usdz, "NuRec USDZ")

    collision_header = parse_ply_header(collision_ply)
    visual_header = parse_ply_header(visual_ply)
    if not collision_header.has_faces:
        raise RuntimeError(f"collision PLY 必须包含 face: {collision_ply}")
    if not is_gaussian_splat_ply(visual_header):
        raise RuntimeError(f"visual PLY 必须包含 3DGS scale/rot/opacity 属性: {visual_ply}")


def _copy_pct_readme_assets(
    *,
    pct_root: Path,
    collision_ply: Path,
    visual_ply: Path,
    collision_usd: Path,
    stage_usda: Path,
    force: bool,
    copied: list[dict[str, str]],
) -> None:
    mutifloor_dir = pct_root / "mutifloor"
    jobs = [
        (collision_ply, mutifloor_dir / "3dgs_collision_cropped.ply"),
        (visual_ply, mutifloor_dir / "3dgs_cropped.ply"),
        (collision_usd, mutifloor_dir / "mutifloor_collision_cropped.usd"),
        (stage_usda, mutifloor_dir / "mutifloor_cropped.usda"),
    ]
    for source, target in jobs:
        _copy_file(source=source, target=target, force=force, copied=copied)


def _copy_file(*, source: Path, target: Path, force: bool, copied: list[dict[str, str]]) -> None:
    if target.exists():
        if not force:
            raise FileExistsError(f"复制目标已存在，覆盖请传 --force: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"复制目标是目录: {target}")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append({"type": "copy", "source": str(source), "target": str(target)})


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} 不能是软链接: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")


def _file_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.exists():
        status["size_bytes"] = path.stat().st_size
    return status


if __name__ == "__main__":
    raise SystemExit(main())
