"""Tests for fixed six-label subtask directory export."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from source.recording.subtask_export import (
    SUBTASK_DATA_COLUMNS,
    materialize_subtask_episode,
    validate_subtask_directory_export,
)
from source.recording.subtask_segmentation import (
    SubtaskSegmentationConfig,
    segment_episode_samples,
)


def _quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]


def _sample(
    frame_index: int,
    *,
    x: float,
    yaw: float,
    vx: float,
    wz: float,
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "timestamp": frame_index * 0.2,
        "simulation_step": frame_index * 10,
        "pipeline_state": "exec_nav_to_pick",
        "base_velocity": [vx, 0.0, wz],
        "action": [vx, 0.0, wz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04],
        "base_pose": [x, 0.0, 0.35, *_quat_from_yaw(yaw)],
        "object_state": [1.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tcp_pose": [x + 0.3, 0.0, 0.6, *_quat_from_yaw(yaw)],
        "tcp_pose_valid": True,
        "gripper_position": 0.04,
        "camera_frames": {
            "front": {
                "raw_image_path": f"images/front/camera0_{frame_index:05d}.jpg"
            },
            "wrist": {
                "raw_image_path": f"images/wrist/camera0_{frame_index:05d}.jpg"
            }
        },
        "action_source": "navigation_test",
        "subtask_signals": {"action_source": "navigation_test"},
    }


def _write_source_episode(root: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    front_image_dir = root / "images" / "front"
    wrist_image_dir = root / "images" / "wrist"
    front_image_dir.mkdir(parents=True)
    wrist_image_dir.mkdir(parents=True)
    motion = [
        (0.0, 0.0, 0.30, 0.0),
        (0.1, 0.0, 0.30, 0.0),
        (0.2, 0.0, 0.30, 0.0),
        (0.2, 0.1, 0.00, 0.5),
        (0.2, 0.2, 0.00, 0.5),
        (0.2, 0.3, 0.00, 0.5),
        (0.2, 0.3, 0.30, 0.0),
        (0.3, 0.3, 0.30, 0.0),
        (0.4, 0.3, 0.30, 0.0),
    ]
    samples: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []
    for frame_index, (x, yaw, vx, wz) in enumerate(motion):
        image_name = f"camera0_{frame_index:05d}.jpg"
        Image.new("RGB", (8, 6), color=(frame_index, 0, 0)).save(
            front_image_dir / image_name
        )
        Image.new("RGB", (8, 6), color=(0, frame_index, 0)).save(
            wrist_image_dir / image_name
        )
        samples.append(
            _sample(frame_index, x=x, yaw=yaw, vx=vx, wz=wz)
        )
        rows.append(
            {
                "pipeline_state": "exec_nav_to_pick",
                "偏航角": str(yaw),
                "前摄像头图像": image_name,
                "腕部摄像头图像": image_name,
            }
        )
    return rows, samples


def _task() -> dict[str, object]:
    return {
        "task_id": 4,
        "episode_id": 7,
        "instruction": "Navigate to the target.",
        "start": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "pick": {
            "base_goal": {"x": 1.0, "y": 0.0, "yaw": 0.0},
            "target_object_id": "cola_01",
            "target_object_class": "cola_can",
        },
        "place": {"enabled": False},
        "training_action": {
            "schema": "base_xyyaw_tcp_base_rpy_gripper_v1",
            "dimension": 10,
            "base_pose_frame": "world",
            "tcp_pose_frame": "base_frame",
            "tcp_euler_order": "roll_pitch_yaw",
            "angle_unit": "rad",
            "position_unit": "m",
        },
    }


def _export(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "source_episode"
    rows, samples = _write_source_episode(source)
    config = SubtaskSegmentationConfig(
        min_segment_frames=1,
        hysteresis_frames=1,
        config_source="test",
    )
    segmentation = segment_episode_samples(
        samples,
        _task(),
        config=config,
        fps=5.0,
    )
    output = tmp_path / "dataset"
    report = materialize_subtask_episode(
        source_episode_dir=source,
        dataset_root=output,
        task=_task(),
        episode_index=0,
        global_frame_offset=20,
        rows=rows,
        samples=samples,
        training_actions=np.zeros((len(samples), 10), dtype=np.float32),
        control_actions=np.zeros((len(samples), 11), dtype=np.float32),
        segmentation=segmentation,
        dataset_schema_version="test_schema",
        source_gripper_joint_range_m=(0.0, 0.04),
    )
    meta = output / "meta"
    meta.mkdir()
    (meta / "subtasks.jsonl").write_text(
        json.dumps(report, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, report


def test_repeated_subtask_is_merged_into_one_of_six_fixed_directories(
    tmp_path: Path,
) -> None:
    output, report = _export(tmp_path)

    assert [subtask["subtask"] for subtask in report["subtasks"]] == [
        "nav_straight",
        "nav_turn",
        "nav_stop",
        "arm_approach",
        "arm_contact",
        "arm_retreat",
    ]
    assert [subtask["segment_dir_name"] for subtask in report["subtasks"]] == [
        "7-1",
        "7-2",
        "7-3",
        "7-4",
        "7-5",
        "7-6",
    ]
    assert report["subtask_directory_count"] == 6
    assert report["subtasks"][0]["source_segment_indices"] == [1, 3]
    assert report["subtasks"][0]["frame_count"] == 6
    assert report["subtasks"][1]["source_segment_indices"] == [2]
    assert report["subtasks"][1]["frame_count"] == 3
    assert all(
        subtask["frame_count"] == 0 for subtask in report["subtasks"][2:]
    )
    episode_root = output / "episodes" / "4" / "7"
    assert {path.name for path in episode_root.iterdir() if path.is_dir()} == {
        "7-1",
        "7-2",
        "7-3",
        "7-4",
        "7-5",
        "7-6",
    }
    with (episode_root / "task.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        task_row = next(csv.DictReader(stream))
    assert task_row["collection_status"] == "collected"
    assert task_row["start_base_x"] == "0.0"
    assert task_row["goal_base_x"] == "1.0"
    assert task_row["goal_base_yaw"] == "0.0"
    assert task_row["subtask_directory_count"] == "6"
    assert task_row["source_subtask_segment_count"] == "3"
    assert len(json.loads(task_row["subtask_directories_json"])) == 6
    reconstructed: list[int] = []
    dataset_indices: list[int] = []
    for subtask in report["subtasks"]:
        segment_dir = episode_root / str(subtask["segment_dir_name"])
        with (segment_dir / "data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert {row["subtask"] for row in rows} in (
            set(),
            {subtask["subtask"]},
        )
        assert [int(row["subtask_frame_index"]) for row in rows] == list(
            range(len(rows))
        )
        assert len(list((segment_dir / "images" / "front").glob("*.jpg"))) == len(
            rows
        )
        assert len(list((segment_dir / "images" / "wrist").glob("*.jpg"))) == len(
            rows
        )
        assert {path.name for path in (segment_dir / "images").iterdir()} == {
            "front",
            "wrist",
        }
        assert [row["image_front_path"] for row in rows] == [
            f"images/front/camera0_{index:05d}.jpg"
            for index in range(len(rows))
        ]
        assert [row["image_wrist_path"] for row in rows] == [
            f"images/wrist/camera0_{index:05d}.jpg"
            for index in range(len(rows))
        ]
        reconstructed.extend(int(row["global_frame_index"]) for row in rows)
        dataset_indices.extend(int(row["dataset_global_index"]) for row in rows)
        if rows:
            for field in (
                "observation_tcp_x_base",
                "observation_tcp_y_base",
                "observation_tcp_z_base",
                "observation_tcp_roll_base",
                "observation_tcp_pitch_base",
                "observation_tcp_yaw_base",
            ):
                assert field in rows[0]
                assert math.isfinite(float(rows[0][field]))
    assert sorted(reconstructed) == list(range(9))
    assert sorted(dataset_indices) == list(range(20, 29))
    assert validate_subtask_directory_export(output)["valid"] is True


def test_validator_rejects_missing_wrist_image(tmp_path: Path) -> None:
    output, report = _export(tmp_path)
    segment_dir = (
        output
        / "episodes"
        / "4"
        / "7"
        / str(report["subtasks"][0]["segment_dir_name"])
    )
    next((segment_dir / "images" / "wrist").glob("*.jpg")).unlink()

    validation = validate_subtask_directory_export(output)

    assert validation["valid"] is False
    error_codes = {error["code"] for error in validation["errors"]}
    assert "missing_subtask_wrist_image" in error_codes
    assert "subtask_wrist_image_count_mismatch" in error_codes


def test_validator_rejects_mixed_labels_inside_one_subtask_folder(
    tmp_path: Path,
) -> None:
    output, report = _export(tmp_path)
    data_csv = (
        output
        / "episodes"
        / "4"
        / "7"
        / str(report["subtasks"][0]["segment_dir_name"])
        / "data.csv"
    )
    with data_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["subtask"] = "nav_turn"
    with data_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBTASK_DATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    validation = validate_subtask_directory_export(output)

    assert validation["valid"] is False
    assert "mixed_subtask_folder" in {
        error["code"] for error in validation["errors"]
    }


def test_validator_rejects_duplicate_episode_directory_records(
    tmp_path: Path,
) -> None:
    output, report = _export(tmp_path)
    metadata_path = output / "meta" / "subtasks.jsonl"
    with metadata_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")

    validation = validate_subtask_directory_export(output)

    assert validation["valid"] is False
    assert "duplicate_subtask_episode_path" in {
        error["code"] for error in validation["errors"]
    }


def test_pick_contact_boundary_belongs_to_the_new_subtask() -> None:
    samples: list[dict[str, object]] = []
    segment_names = [
        "pregrasp",
        "approach",
        "approach",
        "close_gripper",
        "close_gripper",
        "close_gripper",
        "lift",
        "retreat",
        "return_home",
    ]
    for frame_index, segment_name in enumerate(segment_names):
        sample = _sample(
            frame_index,
            x=0.0,
            yaw=0.0,
            vx=0.0,
            wz=0.0,
        )
        sample["pipeline_state"] = "exec_pick"
        object_z = 0.09 if frame_index >= 6 else 0.05
        sample["object_state"] = [
            1.0,
            0.0,
            object_z,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        sample["tcp_pose"] = [1.0, 0.0, object_z, 1.0, 0.0, 0.0, 0.0]
        sample["gripper_position"] = 0.01 if frame_index >= 3 else 0.04
        sample["gripper_command"] = "close" if 3 <= frame_index <= 5 else ""
        sample["subtask_signals"] = {
            "segment_name": segment_name,
            "gripper_command": sample["gripper_command"],
            "explicit_task_contact": False,
        }
        samples.append(sample)

    segmentation = segment_episode_samples(
        samples,
        _task(),
        config=SubtaskSegmentationConfig(
            min_segment_frames=1,
            hysteresis_frames=1,
            config_source="test",
        ),
        fps=5.0,
    )

    assert [segment["subtask"] for segment in segmentation["segments"]] == [
        "arm_approach",
        "arm_contact",
        "arm_retreat",
    ]
    assert [
        (segment["global_start_frame"], segment["global_end_frame"])
        for segment in segmentation["segments"]
    ] == [(0, 2), (3, 5), (6, 8)]
    assert segmentation["frames"][3]["subtask"] == "arm_contact"
    assert segmentation["segments"][1]["contact_label_source"] == (
        "heuristic_action_and_kinematics"
    )


def test_curobo_segment_name_alone_does_not_create_contact_label() -> None:
    samples: list[dict[str, object]] = []
    for frame_index in range(3):
        sample = _sample(
            frame_index,
            x=0.0,
            yaw=0.0,
            vx=0.0,
            wz=0.0,
        )
        sample["pipeline_state"] = "exec_pick"
        sample["tcp_pose"] = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
        sample["gripper_command"] = "close"
        sample["subtask_signals"] = {
            "segment_name": "close_gripper",
            "gripper_command": "close",
            "explicit_task_contact": False,
        }
        samples.append(sample)

    segmentation = segment_episode_samples(
        samples,
        _task(),
        config=SubtaskSegmentationConfig(
            min_segment_frames=1,
            hysteresis_frames=1,
            config_source="test",
        ),
        fps=5.0,
    )

    assert [segment["subtask"] for segment in segmentation["segments"]] == [
        "arm_approach"
    ]
