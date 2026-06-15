#!/usr/bin/env python3
"""校验 full-physics 生成的 LeRobot episode 或合并数据集。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.recording.lerobot_validator import (
    validate_lerobot_dataset,
    validate_lerobot_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校验 LeRobot Parquet、索引、时间戳、feature 定义和视频一致性。",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--episode-dir",
        help="full-physics episode 目录；校验其中的 lerobot_dataset 子目录。",
    )
    source_group.add_argument(
        "--dataset-root",
        help="LeRobot 数据集根目录，目录下应包含 meta/info.json。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episode_dir:
        report = validate_lerobot_episode(args.episode_dir)
    else:
        report = validate_lerobot_dataset(args.dataset_root)

    summary = report["summary"]
    status = "通过" if report["valid"] else "失败"
    details = report.get("details") or {}
    episodes = report.get("episodes") or []
    video_frames = sum(
        int(video.get("frame_count", 0))
        for episode in episodes
        for video in episode.get("videos", [])
    )
    info = details.get("info") if isinstance(details, dict) else {}
    info = info if isinstance(info, dict) else {}
    features = details.get("features") if isinstance(details, dict) else []
    features = features if isinstance(features, list) else []
    image_features = details.get("image_features") if isinstance(details, dict) else []
    image_features = image_features if isinstance(image_features, list) else []
    action = info.get("features", {}).get("action", {}) if info else {}
    state = info.get("features", {}).get("observation.state", {}) if info else {}
    print(
        f"LeRobot 校验{status}：episode={summary['episode_count']}，"
        f"rows={summary['row_count']}，errors={summary['error_count']}"
    )
    print(
        f"parquet rows={summary['row_count']} video frames={video_frames} "
        f"fps={details.get('fps')} camera keys={image_features}"
    )
    print(
        f"feature keys={features} action dim={action.get('shape')} "
        f"observation.state dim={state.get('shape')}"
    )
    print("PASS" if report["valid"] else "FAIL")
    print(f"校验报告：{report['report_path']}")
    if not report["valid"]:
        for issue in report["errors"]:
            print(f"[错误] {issue['code']}: {issue['message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
