#!/usr/bin/env python3
"""Run the new full-physics pipeline or its dependency-free dry run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.pipeline import FullPhysicsConfig  # noqa: E402
from source.pipeline.dry_run import create_dry_run_pipeline  # noqa: E402
from source.tasks import JsonTaskProvider  # noqa: E402


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行单进程、单 World 的纯物理 nav-pick-place pipeline。",
    )
    parser.add_argument("--task-json", required=True, help="任务 JSON 路径。")
    parser.add_argument(
        "--output-dir",
        default="outputs/full_physics_dry_run",
        help="episode、事件、帧和 summary 的输出目录。",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行的 episode 数量。")
    parser.add_argument("--seed", type=int, default=0, help="首个 episode 的随机种子。")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否以无界面模式运行；使用 --no-headless 开启渲染。",
    )
    parser.add_argument(
        "--enable-debug-vis",
        action="store_true",
        help="开启调试可视化；dry-run 只记录该请求，不创建物理可视化。",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="请求保存视频；dry-run 只记录该请求，不生成视频。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="使用无 Isaac 依赖的内存后端验证完整状态流。",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run:
        raise SystemExit("真实纯物理后端尚未接入；第一阶段请显式传入 --dry-run。")

    config = FullPhysicsConfig(
        task_json=_project_path(args.task_json),
        output_dir=_project_path(args.output_dir),
        num_episodes=args.num_episodes,
        seed=args.seed,
        headless=args.headless,
        enable_debug_vis=args.enable_debug_vis,
        save_video=args.save_video,
        dry_run=True,
    )
    base_spec = JsonTaskProvider().load(config.task_json)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = config.output_dir / "batch_summary.jsonl"
    batch_summary_path.write_text("", encoding="utf-8")

    all_success = True
    for episode_index in range(config.num_episodes):
        episode_seed = config.episode_seed(episode_index)
        raw_task = {
            **base_spec.raw_task,
            "episode_id": base_spec.episode_id + episode_index,
        }
        episode_spec = replace(
            base_spec,
            episode_id=base_spec.episode_id + episode_index,
            raw_task=raw_task,
        )
        episode_dir = config.output_dir / f"episode_{episode_index:06d}"
        pipeline = create_dry_run_pipeline(
            config=config,
            episode_spec=episode_spec,
            episode_seed=episode_seed,
            episode_dir=episode_dir,
        )
        summary = pipeline.run_episode()
        all_success = all_success and bool(summary["success"])
        with batch_summary_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        print(
            f"[full-physics] episode={episode_index} seed={episode_seed} "
            f"success={summary['success']} pure_physics_success={summary['pure_physics_success']}"
        )
        print("[full-physics] states:", " -> ".join(summary["state_trace"]))

    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
