#!/usr/bin/env python3
"""验证手动放置目标正下方是否存在 collision PLY 支撑面。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.scene.placement_support import inspect_placement_support  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    target_path = _project_path(args.target_json)
    collision_ply = _project_path(args.collision_ply)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    position_xyz = target["placement_pose_world"]["position_xyz"]
    result = inspect_placement_support(
        collision_ply,
        position_xyz,
        minimum_clearance_m=float(args.minimum_clearance),
        maximum_clearance_m=float(args.maximum_clearance),
    )
    report = {
        "schema_version": 1,
        "target_json": str(target_path),
        "target_object_id": target.get("target_object_id"),
        "target_object_prim_path": target.get("target_object_prim_path"),
        **result.to_dict(include_sha256=bool(args.include_sha256)),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output_path = _project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"[placement-support] report={output_path}")
    print(payload, end="")
    return 0 if result.geometry_verified else 2


def _project_path(raw_path: str | Path) -> Path:
    """解析显式资产路径；相对路径以当前 worktree 为根。"""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读检查固定 place target 的碰撞几何支撑。")
    parser.add_argument(
        "--target-json",
        default="tasks/liangzhu_placement_target.json",
        help="包含 placement_pose_world.position_xyz 的目标标注 JSON。",
    )
    parser.add_argument(
        "--collision-ply",
        default=os.environ.get("LIANGZHU_COLLISION_PLY"),
        required=os.environ.get("LIANGZHU_COLLISION_PLY") is None,
        help="碰撞三角网格 PLY；也可通过 LIANGZHU_COLLISION_PLY 提供。",
    )
    parser.add_argument("--minimum-clearance", type=float, default=0.01)
    parser.add_argument("--maximum-clearance", type=float, default=0.20)
    parser.add_argument("--include-sha256", action="store_true")
    parser.add_argument("--output-json", help="可选 JSON 报告路径。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
