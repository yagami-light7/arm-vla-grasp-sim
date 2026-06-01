"""Validation-first conversion entrypoint for future LeRobot export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="Validate an episode before LeRobot conversion.")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--manifest", default=None, help="Optional output path for the validated manifest.")
    args = parser.parse_args()
    manifest = validate_episode(args.episode_dir)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[INFO] Schema validation passed. LeRobot tensor/video materialization is reserved for the next dataset pass.")


if __name__ == "__main__":
    main()
