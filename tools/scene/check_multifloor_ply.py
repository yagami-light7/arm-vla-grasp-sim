#!/usr/bin/env python3

"""检查 multifloor PLY，并按实际内容选择视觉层和碰撞层输入。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from source.scene.gaussian_splat_ply import describe_ply


DEFAULT_PLY_DIR = PROJECT_ROOT / "source/scene/multifloor/ply"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs/multifloor_ply_check.json"
EXPECTED_NAMES = ("3dgs_collision.ply", "3dgs_visual.ply")
LEGACY_NAMES = ("3dgs_collision_cropped.ply", "3dgs_cropped.ply")


def main() -> int:
    args = _parse_args()
    ply_dir = _project_path(args.ply_dir)
    output_json = _project_path(args.output_json)
    report = build_report(ply_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["problems"] else 2


def build_report(ply_dir: Path) -> dict[str, Any]:
    if not ply_dir.is_dir():
        return {
            "ply_dir": str(ply_dir),
            "files": [],
            "selected_3dgs_visual_ply": None,
            "selected_collision_ply": None,
            "problems": [f"PLY 目录不存在: {ply_dir}"],
        }

    candidates = _candidate_paths(ply_dir)
    files: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in candidates:
        try:
            info = describe_ply(path)
        except Exception as exc:
            problems.append(f"读取 PLY 失败: {path}: {exc}")
            continue
        info["size_bytes"] = path.stat().st_size
        info["size_mb"] = round(path.stat().st_size / 1024 / 1024, 3)
        info["has_3dgs_fields"] = bool(info["is_gaussian_splat_ply"])
        info["role_hint"] = _role_hint(info)
        files.append(info)

    visual = _select_visual(files)
    collision = _select_collision(files)
    if visual is None:
        problems.append("未找到包含 f_dc/opacity/scale/rot 的 3DGS visual PLY")
    if collision is None:
        problems.append("未找到包含 face 的 collision mesh PLY")

    return {
        "ply_dir": str(ply_dir),
        "expected_names": list(EXPECTED_NAMES),
        "files": files,
        "selected_3dgs_visual_ply": visual["path"] if visual else None,
        "selected_collision_ply": collision["path"] if collision else None,
        "recommendation": {
            "3dgs_visual": visual["path"] if visual else None,
            "collision": collision["path"] if collision else None,
            "reason": "按 PLY header 内容判断，不按历史文件名判断。",
        },
        "problems": problems,
    }


def _candidate_paths(ply_dir: Path) -> list[Path]:
    preferred = [ply_dir / name for name in (*EXPECTED_NAMES, *LEGACY_NAMES)]
    existing: list[Path] = []
    seen: set[Path] = set()
    for path in preferred:
        if path.is_file() and path not in seen:
            existing.append(path)
            seen.add(path)
    for path in sorted(ply_dir.glob("*.ply")):
        if path not in seen:
            existing.append(path)
            seen.add(path)
    return existing


def _role_hint(info: dict[str, Any]) -> str:
    if info["is_gaussian_splat_ply"]:
        return "3dgs_visual"
    if info["has_faces"]:
        return "collision_mesh"
    return "unknown"


def _select_visual(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    visual_candidates = [info for info in files if info["is_gaussian_splat_ply"]]
    if not visual_candidates:
        return None
    return max(visual_candidates, key=lambda info: int(info["vertex_count"]))


def _select_collision(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    collision_candidates = [info for info in files if info["has_faces"] and not info["is_gaussian_splat_ply"]]
    if not collision_candidates:
        return None
    return min(collision_candidates, key=lambda info: int(info["size_bytes"]))


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 multifloor PLY 文件结构。")
    parser.add_argument("--ply-dir", default=os.fspath(DEFAULT_PLY_DIR), help="PLY 输入目录。")
    parser.add_argument("--output-json", default=os.fspath(DEFAULT_OUTPUT_JSON), help="检查报告输出路径。")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
