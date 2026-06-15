"""Validation-first conversion entrypoint for future LeRobot export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from source.recording import discover_recorded_episodes, materialize_lerobot_dataset

from .episode_recorder import EPISODE_COLUMNS, PHASES


def validate_episode(episode_dir: str | Path) -> dict[str, Any]:
    """Validate the stable multi-phase schema and return an export manifest."""

    episode_path = Path(episode_dir).expanduser().resolve()
    task_path = episode_path / "task.json"
    summary_path = episode_path / "summary.json"
    if not task_path.exists():
        raise FileNotFoundError(f"missing task.json: {task_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")

    manifest: dict[str, Any] = {
        "episode_dir": str(episode_path),
        "task": json.loads(task_path.read_text(encoding="utf-8")),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "phases": {},
    }
    expected = list(EPISODE_COLUMNS)
    for phase in PHASES:
        csv_path = episode_path / phase / "data.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != expected:
                raise ValueError(f"{csv_path} has unexpected columns")
            rows = list(reader)
        manifest["phases"][phase] = {"csv": str(csv_path), "rows": len(rows)}
    if not manifest["phases"]:
        raise ValueError(f"episode has no phase data: {episode_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="校验旧多阶段 episode，或将 full-physics 原始数据转换为 LeRobot v2.1。",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--episode-dir", help="校验一个旧多阶段 episode。")
    source_group.add_argument(
        "--episodes-root",
        help="扫描成功的 full-physics episode 并执行 DWA-compatible LeRobot 转换。",
    )
    parser.add_argument("--manifest", default=None, help="Optional output path for the validated manifest.")
    parser.add_argument(
        "--output-root",
        help="LeRobot 输出目录；默认写入 <episodes-root>/lerobot_dataset。",
    )
    args = parser.parse_args()
    if args.episodes_root:
        episodes_root = Path(args.episodes_root).expanduser().resolve()
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else episodes_root / "lerobot_dataset"
        )
        episode_dirs = discover_recorded_episodes(episodes_root, require_success=True)
        manifest = materialize_lerobot_dataset(episode_dirs, output_root)
    else:
        manifest = validate_episode(args.episode_dir)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.episodes_root:
        if not manifest.get("lerobot_exported"):
            raise SystemExit(1)
        print("[INFO] LeRobot v2.1 conversion completed.")
    else:
        print("[INFO] Schema validation passed.")


if __name__ == "__main__":
    main()
