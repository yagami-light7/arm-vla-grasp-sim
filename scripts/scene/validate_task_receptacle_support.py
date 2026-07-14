#!/usr/bin/env python3
"""只读验证 task 指定 receptacle 在组合 USD 中的碰撞支撑。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.simulation.receptacle_support import (  # noqa: E402
    inspect_task_receptacle_support_usd,
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    task_path = _project_path(args.task_json)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    scene_raw = args.scene_usd or task.get("scene_usd")
    if not scene_raw:
        raise ValueError("task.scene_usd 或 --scene-usd 必须提供")
    scene_path = _project_path(scene_raw)
    report = inspect_task_receptacle_support_usd(
        scene_path,
        task,
        source="offline_composed_scene_validation",
    )
    payload = {
        "schema_version": 1,
        "task_json": str(task_path),
        "scene_usd": str(scene_path),
        **report,
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output_path = _project_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"[receptacle-support] report={output_path}")
    print(output, end="")
    return 0 if report.get("geometry_verified") is True else 2


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 OpenUSD 检查 task receptacle/support 的组合碰撞几何。"
    )
    parser.add_argument(
        "--task-json",
        default="tasks/nav_pick_place_cola_liangzhu_pct.json",
    )
    parser.add_argument("--scene-usd", help="可选场景覆盖；默认读取 task.scene_usd。")
    parser.add_argument("--output-json", help="可选 JSON 报告路径。")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
