#!/usr/bin/env python3
"""Validate one full-physics episode, its LeRobot export, and composite video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path, *, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"缺少 JSON：{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"无法读取 JSON {path}：{type(exc).__name__}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"JSON 顶层不是对象：{path}")
        return {}
    return payload


def _composite_video_path(
    episode_dir: Path,
    summary: dict[str, Any],
) -> Path | None:
    video_summary = summary.get("overview_video")
    if not isinstance(video_summary, dict):
        return None
    video_paths = video_summary.get("video_paths")
    raw_path = (
        video_paths.get("composite")
        if isinstance(video_paths, dict)
        else None
    )
    if raw_path is None:
        videos = video_summary.get("videos")
        composite = videos.get("composite") if isinstance(videos, dict) else None
        raw_path = (
            composite.get("video_path")
            if isinstance(composite, dict)
            else None
        )
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = episode_dir / path
    return path.resolve()


def _probe_video(path: Path, *, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"缺少 composite MP4：{path}")
        return {}
    if path.stat().st_size <= 0:
        errors.append(f"composite MP4 为空文件：{path}")
        return {}
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            errors.append(f"OpenCV 无法打开 composite MP4：{path}")
            return {}
        return {
            "path": str(path),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "size_bytes": int(path.stat().st_size),
        }
    finally:
        capture.release()


def validate_full_pipeline_episode(
    episode_dir: str | Path,
    *,
    expected_width: int = 1280,
    expected_height: int = 720,
    expected_fps: float = 25.0,
    fps_tolerance: float = 0.1,
) -> dict[str, Any]:
    """Return a structured, non-mutating acceptance report."""

    root = Path(episode_dir).expanduser().resolve()
    errors: list[str] = []
    summary_path = root / "summary.json"
    lerobot_report_path = root / "lerobot_dataset" / "validation_report.json"
    summary = _read_json(summary_path, errors=errors)
    lerobot_report = _read_json(lerobot_report_path, errors=errors)

    required_true_fields = (
        "success",
        "physical_navigation_success",
        "physical_manipulation_success",
        "stable_physics_success",
        "training_quality_gate_passed",
        "lerobot_training_eligible",
    )
    for field in required_true_fields:
        if summary.get(field) is not True:
            errors.append(f"summary.{field} 必须为 true，当前为 {summary.get(field)!r}")
    if summary.get("final_state") != "done":
        errors.append(
            f"summary.final_state 必须为 'done'，当前为 {summary.get('final_state')!r}"
        )

    validation_summary = lerobot_report.get("summary")
    if not isinstance(validation_summary, dict):
        errors.append("LeRobot validation_report.summary 缺失或不是对象")
        validation_summary = {}
    row_count = int(validation_summary.get("row_count") or 0)
    error_count = int(validation_summary.get("error_count") or 0)
    warning_count = int(validation_summary.get("warning_count") or 0)
    if row_count <= 0:
        errors.append(f"LeRobot row_count 必须大于 0，当前为 {row_count}")
    if error_count != 0:
        errors.append(f"LeRobot error_count 必须为 0，当前为 {error_count}")
    if warning_count != 0:
        errors.append(f"LeRobot warning_count 必须为 0，当前为 {warning_count}")

    video_summary = summary.get("overview_video")
    if not isinstance(video_summary, dict):
        errors.append("summary.overview_video 缺失或不是对象")
        video_summary = {}
    if video_summary.get("mode") != "composite":
        errors.append(
            "summary.overview_video.mode 必须为 'composite'，当前为 "
            f"{video_summary.get('mode')!r}"
        )
    if video_summary.get("success") is not True:
        errors.append("summary.overview_video.success 必须为 true")
    streams = video_summary.get("streams")
    if streams != ["composite"]:
        errors.append(f"composite streams 必须为 ['composite']，当前为 {streams!r}")
    layout = video_summary.get("composite_layout")
    if not isinstance(layout, dict):
        errors.append("summary.overview_video.composite_layout 缺失")
    else:
        if layout.get("view_keys") != ["overview", "front", "wrist"]:
            errors.append("composite_layout.view_keys 必须为 overview/front/wrist")
        if layout.get("synchronization") != "same_simulation_step":
            errors.append("composite_layout 必须声明 same_simulation_step 同步")

    video_path = _composite_video_path(root, summary)
    if video_path is None:
        errors.append("summary 中缺少 composite video path")
        video_probe: dict[str, Any] = {}
    else:
        video_probe = _probe_video(video_path, errors=errors)
    if video_probe:
        if video_probe["width"] != int(expected_width):
            errors.append(
                f"composite width={video_probe['width']}，期望 {expected_width}"
            )
        if video_probe["height"] != int(expected_height):
            errors.append(
                f"composite height={video_probe['height']}，期望 {expected_height}"
            )
        if video_probe["frame_count"] <= 0:
            errors.append("composite frame_count 必须大于 0")
        if not math.isclose(
            video_probe["fps"],
            float(expected_fps),
            rel_tol=0.0,
            abs_tol=float(fps_tolerance),
        ):
            errors.append(
                f"composite fps={video_probe['fps']:.6g}，期望 {expected_fps}"
            )
        videos = video_summary.get("videos")
        composite_summary = (
            videos.get("composite") if isinstance(videos, dict) else None
        )
        expected_frames = (
            int(composite_summary.get("frame_count") or 0)
            if isinstance(composite_summary, dict)
            else 0
        )
        if expected_frames <= 0:
            errors.append("summary 中 composite frame_count 必须大于 0")
        elif video_probe["frame_count"] != expected_frames:
            errors.append(
                "composite 编码帧数与 summary 不一致："
                f"video={video_probe['frame_count']} summary={expected_frames}"
            )

    return {
        "valid": not errors,
        "episode_dir": str(root),
        "summary_path": str(summary_path),
        "lerobot_validation_report": str(lerobot_report_path),
        "pipeline": {
            "success": summary.get("success"),
            "final_state": summary.get("final_state"),
            "stable_physics_success": summary.get("stable_physics_success"),
            "training_eligible": summary.get("lerobot_training_eligible"),
        },
        "lerobot": {
            "row_count": row_count,
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "composite_video": video_probe,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验收 full-physics 状态机、LeRobot 数据和三视角 composite MP4。"
    )
    parser.add_argument("--episode-dir", required=True, help="待验收的 episode 目录。")
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument("--expected-height", type=int, default=720)
    parser.add_argument("--expected-fps", type=float, default=25.0)
    parser.add_argument("--fps-tolerance", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_full_pipeline_episode(
        args.episode_dir,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        expected_fps=args.expected_fps,
        fps_tolerance=args.fps_tolerance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["valid"]:
        print("完整 pipeline 验收通过：状态机、LeRobot 与三视角视频均有效。")
        return 0
    print("完整 pipeline 验收失败。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
