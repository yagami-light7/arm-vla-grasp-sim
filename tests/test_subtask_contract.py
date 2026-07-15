"""子任务标签全集与未采集目录语义的 contract 测试。"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from source.recording.lerobot_dataset import _resolve_subtask_episode_ids
from source.recording.subtask_export import write_subtask_task_stub
from source.recording.subtask_segmentation import (
    SUBTASK_DIRECTORY_LAYOUT,
    SubtaskSegmentationConfig,
    hydrate_sample_action_semantics,
    segment_episode_samples,
    validate_subtask_segmentation_config,
)


def _sample(
    frame_index: int,
    *,
    pipeline_state: str,
    command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    object_z: float = 0.10,
    gripper: float = 0.04,
    segment_name: str = "",
    explicit_contact: bool = False,
) -> dict:
    return {
        "timestamp": frame_index / 5.0,
        "pipeline_state": pipeline_state,
        "base_pose": [0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0],
        "base_velocity": [*command],
        "tcp_pose": [0.05, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0],
        "tcp_pose_valid": True,
        "object_state": [0.05, 0.0, object_z, 1.0, 0.0, 0.0, 0.0],
        "gripper_position": gripper,
        "action": [*command, *([0.0] * 8)],
        "subtask_signals": {
            "action_source": "contract_test",
            "segment_name": segment_name,
            "parent_segment_name": "",
            "event_marker": "",
            "gripper_command": "",
            "explicit_task_contact": explicit_contact,
        },
    }


def _append(samples: list[dict], count: int, **kwargs) -> None:
    for _ in range(count):
        samples.append(_sample(len(samples), **kwargs))


def _task(*, episode_id: int = 1) -> dict:
    return {
        "task_id": 4,
        "episode_id": episode_id,
        "instruction": "Pick up the coke and place it on the mat.",
        "start": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "pick": {"base_goal": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
        "place": {
            "enabled": True,
            "base_goal": {"x": 2.0, "y": 0.0, "yaw": 0.0},
        },
    }


def test_legacy_front_only_layout_config_migrates_to_front_wrist_v2() -> None:
    task = _task()
    task["subtask_segmentation"] = {
        "enabled": True,
        "directory_export": True,
        "output_layout": "episodes_task_episode_segment_v1",
    }

    config = validate_subtask_segmentation_config(task)

    assert config is not None
    assert config.output_layout == SUBTASK_DIRECTORY_LAYOUT
    assert config.config_source == "task_legacy_layout_migrated"


def test_all_six_labels_preserve_stage_order_and_frame_coverage() -> None:
    samples: list[dict] = []
    _append(
        samples,
        3,
        pipeline_state="exec_nav_to_pick",
        command=(0.20, 0.0, 0.0),
    )
    _append(
        samples,
        3,
        pipeline_state="exec_nav_to_pick",
        command=(0.0, 0.0, 0.30),
    )
    _append(samples, 3, pipeline_state="exec_pick")
    _append(
        samples,
        3,
        pipeline_state="exec_pick",
        gripper=0.0,
        segment_name="close_gripper",
        explicit_contact=True,
    )
    _append(
        samples,
        3,
        pipeline_state="exec_pick",
        object_z=0.20,
        gripper=0.0,
        segment_name="lift_object",
    )
    _append(samples, 3, pipeline_state="exec_nav_to_place")
    _append(samples, 3, pipeline_state="exec_place", gripper=0.0)
    _append(
        samples,
        3,
        pipeline_state="exec_place",
        segment_name="open_gripper",
        explicit_contact=True,
    )
    _append(
        samples,
        3,
        pipeline_state="exec_place",
        segment_name="retreat",
    )
    result = segment_episode_samples(
        samples,
        _task(),
        config=SubtaskSegmentationConfig(
            min_segment_frames=1,
            hysteresis_frames=1,
            config_source="contract_test",
        ),
        fps=5.0,
    )

    labels = [segment["subtask"] for segment in result["segments"]]
    assert set(labels) == {
        "nav_straight",
        "nav_turn",
        "nav_stop",
        "arm_approach",
        "arm_contact",
        "arm_retreat",
    }
    assert [segment["segment_index"] for segment in result["segments"]] == list(
        range(1, len(result["segments"]) + 1)
    )
    covered = [
        frame_index
        for segment in result["segments"]
        for frame_index in range(
            segment["global_start_frame"],
            segment["global_end_frame"] + 1,
        )
    ]
    assert covered == list(range(len(samples)))
    assert {
        segment["contact_label_source"]
        for segment in result["segments"]
        if segment["subtask"] == "arm_contact"
    } == {"heuristic_action_and_kinematics"}


def test_uncollected_episode_has_only_task_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_root = Path(tmp_dir) / "dataset"
        task_csv = write_subtask_task_stub(
            dataset_root=dataset_root,
            task=_task(episode_id=3),
            dataset_schema_version="test_v1",
        )
        episode_root = dataset_root / "episodes" / "4" / "3"

        assert task_csv == episode_root / "task.csv"
        assert sorted(path.name for path in episode_root.iterdir()) == ["task.csv"]

        stale_segment = episode_root / "3-1"
        stale_segment.mkdir()
        write_subtask_task_stub(
            dataset_root=dataset_root,
            task=_task(episode_id=3),
            dataset_schema_version="test_v1",
        )
        assert sorted(path.name for path in episode_root.iterdir()) == ["task.csv"]
        with task_csv.open("r", encoding="utf-8", newline="") as stream:
            task_row = next(csv.DictReader(stream))
        assert task_row["collection_status"] == "planned"
        assert task_row["training_eligible"] == ""


def test_duplicate_batch_episode_ids_are_renumbered_per_task() -> None:
    tasks = [
        {"task_id": 4, "episode_id": 1},
        {"task_id": 4, "episode_id": 1},
        {"task_id": 9, "episode_id": 7},
        {"task_id": 9, "episode_id": 8},
    ]

    assert _resolve_subtask_episode_ids(tasks) == [1, 2, 7, 8]


def test_hydration_merges_recovered_semantics_into_empty_legacy_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        episode_dir = Path(tmp_dir)
        frame = {
            "step_index": 10,
            "action": {
                "source": "navigation_terminal",
                "gripper_command": "close",
                "metadata": {
                    "segment_name": "close_gripper",
                    "target_contact": True,
                },
            },
        }
        (episode_dir / "frames.jsonl").write_text(
            json.dumps(frame) + "\n",
            encoding="utf-8",
        )

        hydrated = hydrate_sample_action_semantics(
            episode_dir,
            [
                {
                    "simulation_step": 10,
                    "action_source": "",
                    "gripper_command": "",
                    "subtask_signals": {},
                }
            ],
        )[0]

        assert hydrated["action_source"] == "navigation_terminal"
        assert hydrated["gripper_command"] == "close"
        assert hydrated["subtask_signals"]["segment_name"] == "close_gripper"
        assert hydrated["subtask_signals"]["explicit_task_contact"] is True
