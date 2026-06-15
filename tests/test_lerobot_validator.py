"""LeRobot 数据集校验器测试。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.pipeline.validate_lerobot_episode import build_parser, main
from source.recording.lerobot_validator import (
    validate_lerobot_dataset,
    validate_lerobot_episode,
)


FPS = 5
IMAGE_FEATURE = "observation.images.front"


def _feature(dtype: str, shape: list[int], names: list[str] | None) -> dict:
    return {"dtype": dtype, "shape": shape, "names": names}


def _write_video(path: Path, *, frame_count: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (24, 16),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建测试视频：{path}")
    try:
        for frame_index in range(frame_count):
            frame = np.full((16, 24, 3), frame_index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _episode_table(
    episode_index: int,
    length: int,
    global_start: int,
    *,
    overrides: dict[str, list] | None = None,
    fixed_size_vectors: bool = True,
) -> pa.Table:
    values: dict[str, list] = {
        "index": list(range(global_start, global_start + length)),
        "episode_index": [episode_index] * length,
        "frame_index": list(range(length)),
        "timestamp": [frame_index / FPS for frame_index in range(length)],
        "task_index": [0] * length,
        "observation.state": [
            [float(frame_index), 0.1, 0.2] for frame_index in range(length)
        ],
        "observation.base_velocity": [[0.1, 0.0, 0.0] for _ in range(length)],
        "pipeline_state": ["exec_nav_to_pick"] * length,
        "action": [[0.2, -0.1] for _ in range(length)],
        "next.done": [False] * (length - 1) + [True],
    }
    values.update(overrides or {})
    state_type = (
        pa.list_(pa.float32(), 3)
        if fixed_size_vectors
        else pa.list_(pa.float32())
    )
    base_velocity_type = (
        pa.list_(pa.float32(), 3)
        if fixed_size_vectors
        else pa.list_(pa.float32())
    )
    action_type = (
        pa.list_(pa.float32(), 2)
        if fixed_size_vectors
        else pa.list_(pa.float32())
    )
    schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("task_index", pa.int64()),
            pa.field("observation.state", state_type),
            pa.field("observation.base_velocity", base_velocity_type),
            pa.field("pipeline_state", pa.string()),
            pa.field("action", action_type),
            pa.field("next.done", pa.bool_()),
        ]
    )
    arrays = {
        field.name: pa.array(values[field.name], type=field.type) for field in schema
    }
    return pa.table(arrays, schema=schema)


def _write_dataset(root: Path, *, lengths: tuple[int, ...] = (3, 2)) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.state": _feature(
            "float32",
            [3],
            ["base_x", "base_y", "base_z"],
        ),
        "observation.base_velocity": _feature(
            "float32",
            [3],
            ["vel_x", "vel_y", "vel_z"],
        ),
        "pipeline_state": _feature("string", [1], None),
        "action": _feature("float32", [2], ["cmd_x", "cmd_y"]),
        "next.done": _feature("bool", [1], None),
        "timestamp": _feature("float32", [1], None),
        "frame_index": _feature("int64", [1], None),
        "episode_index": _feature("int64", [1], None),
        "index": _feature("int64", [1], None),
        "task_index": _feature("int64", [1], None),
        IMAGE_FEATURE: {
            **_feature("video", [16, 24, 3], ["height", "width", "channels"]),
            "video_info": {"video.fps": FPS},
        },
    }
    info = {
        "codebase_version": "v2.1",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "features": features,
        "observation_state_names": features["observation.state"]["names"],
        "base_velocity_names": features["observation.base_velocity"]["names"],
        "action_names": features["action"]["names"],
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "抓取并放置苹果"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    global_start = 0
    episode_lines: list[str] = []
    for episode_index, length in enumerate(lengths):
        parquet_path = (
            root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _episode_table(episode_index, length, global_start),
            parquet_path,
        )
        _write_video(
            root
            / "videos"
            / "chunk-000"
            / IMAGE_FEATURE
            / f"episode_{episode_index:06d}.mp4",
            frame_count=length,
            fps=FPS,
        )
        episode_lines.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": ["抓取并放置苹果"],
                    "length": length,
                },
                ensure_ascii=False,
            )
        )
        global_start += length
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(episode_lines) + "\n",
        encoding="utf-8",
    )


def _error_codes(report: dict) -> set[str]:
    return {issue["code"] for issue in report["errors"]}


class LeRobotValidatorTest(unittest.TestCase):
    def test_validates_dataset_and_nested_episode_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_root)

            report = validate_lerobot_dataset(dataset_root)

            self.assertTrue(report["valid"], report["errors"])
            self.assertTrue(report["success"])
            self.assertIsNone(report["failure_reason"])
            self.assertEqual(report["summary"]["episode_count"], 2)
            self.assertEqual(report["summary"]["row_count"], 5)
            report_path = dataset_root / "validation_report.json"
            self.assertTrue(report_path.is_file())
            self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["valid"])

            episode_dir = Path(tmp_dir) / "episode_000123"
            nested_dataset = episode_dir / "lerobot_dataset"
            _write_dataset(nested_dataset, lengths=(2,))
            nested_report = validate_lerobot_episode(episode_dir)
            self.assertTrue(nested_report["valid"], nested_report["errors"])
            self.assertEqual(nested_report["dataset_root"], str(nested_dataset.resolve()))
            self.assertTrue((episode_dir / "validation_report.json").is_file())

    def test_detects_index_timestamp_done_task_and_dimension_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_root, lengths=(3, 2))
            parquet_path = (
                dataset_root / "data/chunk-000/episode_000001.parquet"
            )
            invalid_table = _episode_table(
                1,
                2,
                3,
                overrides={
                    "index": [8, 9],
                    "episode_index": [1, 7],
                    "frame_index": [0, 2],
                    "timestamp": [0.2, 0.1],
                    "task_index": [9, 9],
                    "observation.state": [[1.0, 2.0], [3.0, 4.0]],
                    "pipeline_state": ["exec_pick", ""],
                    "action": [[0.2, -0.1], [0.2]],
                    "next.done": [True, True],
                },
                fixed_size_vectors=False,
            ).append_column(
                IMAGE_FEATURE,
                pa.array(["video-only", "video-only"], type=pa.string()),
            )
            pq.write_table(invalid_table, parquet_path)

            report = validate_lerobot_dataset(dataset_root)
            codes = _error_codes(report)

            self.assertFalse(report["valid"])
            self.assertFalse(report["success"])
            self.assertEqual(report["failure_reason"], "lerobot_validation_failed")
            self.assertIn("global_index_not_continuous", codes)
            self.assertIn("episode_index_mismatch", codes)
            self.assertIn("frame_index_not_continuous", codes)
            self.assertIn("timestamp_not_monotonic", codes)
            self.assertIn("timestamp_fps_mismatch", codes)
            self.assertIn("next_done_invalid", codes)
            self.assertIn("unknown_task_index", codes)
            self.assertIn("task_indices_mismatch", codes)
            self.assertIn("invalid_pipeline_state", codes)
            self.assertIn("inconsistent_vector_dimension", codes)
            self.assertIn("vector_dimension_mismatch", codes)
            self.assertIn("image_feature_stored_in_parquet", codes)

    def test_detects_missing_parquet_row_count_features_and_video_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_root, lengths=(3, 2))

            info_path = dataset_root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"]["action"]["names"] = ["cmd_x"]
            info_path.write_text(
                json.dumps(info, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (dataset_root / "data/chunk-000/episode_000001.parquet").unlink()
            _write_video(
                dataset_root
                / "videos"
                / "chunk-000"
                / IMAGE_FEATURE
                / "episode_000000.mp4",
                frame_count=2,
                fps=10,
            )

            report = validate_lerobot_dataset(dataset_root)
            codes = _error_codes(report)

            self.assertFalse(report["valid"])
            self.assertIn("feature_names_count_mismatch", codes)
            self.assertIn("missing_parquet", codes)
            self.assertIn("total_frames_mismatch", codes)
            self.assertIn("total_episodes_mismatch", codes)
            self.assertIn("video_frame_count_mismatch", codes)
            self.assertIn("video_fps_mismatch", codes)

    def test_cli_returns_nonzero_and_help_is_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            _write_dataset(dataset_root, lengths=(2,))
            (
                dataset_root
                / "videos"
                / "chunk-000"
                / IMAGE_FEATURE
                / "episode_000000.mp4"
            ).unlink()

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(["--dataset-root", str(dataset_root)])

            self.assertEqual(exit_code, 1)
            self.assertIn(
                "missing_video",
                _error_codes(
                    json.loads(
                        (dataset_root / "validation_report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                ),
            )
            help_text = build_parser().format_help()
            self.assertIn("--episode-dir", help_text)
            self.assertIn("--dataset-root", help_text)
            self.assertIn("校验", help_text)


if __name__ == "__main__":
    unittest.main()
