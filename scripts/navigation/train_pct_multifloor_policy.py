#!/usr/bin/env python3
"""Go2-X5 PCT 多楼层 locomotion policy 训练脚手架。

本脚本当前不直接训练 checkpoint，只固定后续 Isaac Lab/RSL-RL
任务、配置和输出目录的入口，避免把生成的大型 checkpoint 提交进仓库。
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
        description="准备 Isaac Lab/RSL-RL PCT 多楼层训练运行参数。",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="训练设计和配置 YAML 路径。",
    )
    parser.add_argument(
        "--task",
        default="RobotLab-Isaac-Velocity-PCT-Multifloor-Go2-X5-v0",
        help="后续注册的 PCT 多楼层 Isaac Lab task id。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="RSL-RL 写入 model_*.pt checkpoint 的目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印训练入口信息，不启动 Isaac Lab。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"训练配置不存在：{config_path}")

    payload = {
        "status": "scaffold",
        "task": str(args.task),
        "config": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "expected_checkpoint_pattern": str(checkpoint_dir / "model_*.pt"),
        "next_step": (
            "先注册 PCT 多楼层 Isaac Lab task，再用该 task id 调用项目的 "
            "Isaac Lab RSL-RL 训练命令。"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.dry_run:
        raise SystemExit(
            "PCT 多楼层训练环境尚未注册；请使用 --dry-run 查看计划，"
            "或先实现 Isaac Lab task。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
