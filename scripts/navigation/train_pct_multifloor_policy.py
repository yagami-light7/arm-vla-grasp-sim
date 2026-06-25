#!/usr/bin/env python3
"""Scaffold for Go2-X5 PCT multi-floor locomotion policy training.

This script intentionally does not train a checkpoint by itself. It is the
stable repository entrypoint for wiring the future Isaac Lab/RSL-RL task,
config, and output locations without committing generated checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/navigation/pct_multifloor_policy.yaml"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/go2_x5/pct_multifloor"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the Isaac Lab/RSL-RL PCT multi-floor training run.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Training design/config YAML path.",
    )
    parser.add_argument(
        "--task",
        default="RobotLab-Isaac-Velocity-PCT-Multifloor-Go2-X5-v0",
        help="Future Isaac Lab task id for PCT multi-floor locomotion.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory where RSL-RL should write model_*.pt checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned training invocation without launching Isaac Lab.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"training config does not exist: {config_path}")

    payload = {
        "status": "scaffold",
        "task": str(args.task),
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "expected_checkpoint_pattern": str(checkpoint_dir / "model_*.pt"),
        "next_step": (
            "Register the PCT multi-floor Isaac Lab task, then invoke the "
            "project's Isaac Lab RSL-RL training command with this task id."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.dry_run:
        raise SystemExit(
            "PCT multi-floor training environment is not registered yet; "
            "use --dry-run for planning or implement the Isaac Lab task first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
