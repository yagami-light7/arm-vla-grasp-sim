"""Full pipeline + LeRobot + composite video acceptance tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.pipeline.validate_full_pipeline_acceptance import (
    validate_full_pipeline_episode,
)
from source.recording.overview_video_recorder import compose_multiview_frame


def _write_video(
    path: Path,
    *,
    width: int,
    height: int,
    frames: int,
    labels: bool = True,
    blue_overview: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            if labels:
                overview_color = (
                    (88, 118, 154)
                    if blue_overview
                    else (220, 20 + index, 20)
                )
                image_rgb = compose_multiview_frame(
                    {
                        "overview": np.full(
                            (90, 160, 3),
                            overview_color,
                            dtype=np.uint8,
                        ),
                        "front": np.full(
                            (90, 160, 3),
                            (20, 220, 20 + index),
                            dtype=np.uint8,
                        ),
                        "wrist": np.full(
                            (90, 160, 3),
                            (220, 220, 20 + index),
                            dtype=np.uint8,
                        ),
                    },
                    width=width,
                    height=height,
                )
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            else:
                image = np.full(
                    (height, width, 3),
                    80 + index * 10,
                    dtype=np.uint8,
                )
            writer.write(image)
    finally:
        writer.release()


def _write_valid_episode(root: Path) -> Path:
    episode_dir = root / "episode_000000"
    video_path = episode_dir / "overview_videos/episode_000001_composite.mp4"
    _write_video(video_path, width=320, height=180, frames=3)
    summary = {
        "success": True,
        "final_state": "done",
        "physical_navigation_success": True,
        "physical_manipulation_success": True,
        "stable_physics_success": True,
        "training_quality_gate_passed": True,
        "lerobot_training_eligible": True,
        "overview_video": {
            "success": True,
            "mode": "composite",
            "streams": ["composite"],
            "video_paths": {"composite": str(video_path)},
            "videos": {
                "composite": {
                    "video_path": str(video_path),
                    "success": True,
                    "frame_count": 3,
                }
            },
            "composite_layout": {
                "view_keys": ["overview", "front", "wrist"],
                "synchronization": "same_simulation_step",
            },
        },
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    validation_path = episode_dir / "lerobot_dataset/validation_report.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(
            {
                "summary": {
                    "episode_count": 1,
                    "row_count": 30,
                    "error_count": 0,
                    "warning_count": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    return episode_dir


def test_acceptance_passes_for_complete_pipeline_and_composite_video() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        episode_dir = _write_valid_episode(Path(tmp_dir))

        report = validate_full_pipeline_episode(
            episode_dir,
            expected_width=320,
            expected_height=180,
        )

        assert report["valid"] is True
        assert report["errors"] == []
        assert report["composite_video"]["frame_count"] == 3


def test_acceptance_rejects_pipeline_failure_and_missing_composite() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        episode_dir = _write_valid_episode(Path(tmp_dir))
        summary_path = episode_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["success"] = False
        summary["overview_video"]["video_paths"].pop("composite")
        summary["overview_video"]["videos"].pop("composite")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        report = validate_full_pipeline_episode(
            episode_dir,
            expected_width=320,
            expected_height=180,
        )

        assert report["valid"] is False
        assert any("summary.success" in error for error in report["errors"])
        assert any("composite video path" in error for error in report["errors"])


def test_acceptance_rejects_blue_placeholder_overview_frames() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        episode_dir = _write_valid_episode(Path(tmp_dir))
        video_path = (
            episode_dir
            / "overview_videos"
            / "episode_000001_composite.mp4"
        )
        _write_video(
            video_path,
            width=320,
            height=180,
            frames=3,
            blue_overview=True,
        )

        report = validate_full_pipeline_episode(
            episode_dir,
            expected_width=320,
            expected_height=180,
        )

        assert report["valid"] is False
        assert any("overview 存在无效/占位帧" in error for error in report["errors"])
        assert (
            report["composite_video"]["quality"]["invalid_overview_frame_count"]
            == 3
        )


def test_acceptance_rejects_missing_composite_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        episode_dir = _write_valid_episode(Path(tmp_dir))
        video_path = (
            episode_dir
            / "overview_videos"
            / "episode_000001_composite.mp4"
        )
        _write_video(
            video_path,
            width=320,
            height=180,
            frames=3,
            labels=False,
        )

        report = validate_full_pipeline_episode(
            episode_dir,
            expected_width=320,
            expected_height=180,
        )

        assert report["valid"] is False
        assert any("标签缺失帧" in error for error in report["errors"])
        assert report["composite_video"]["quality"]["missing_label_frame_count"] == 3
