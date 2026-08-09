from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from source.interfaces import RobotAction, SimulationState, StepRecord
from source.recording.jsonl_recorder import JsonlEpisodeRecorder


def _points_sha256(points: list[list[float]]) -> str:
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("!ddd", *point))
    return digest.hexdigest()


def _path_report(
    *,
    sequence: int,
    stamp_sec: int,
    points: list[list[float]],
    cleared: bool = False,
) -> dict[str, object]:
    return {
        "points_ground_xyz": points,
        "terminal_yaw": 0.4,
        "source": "ros2_nav_msgs_path",
        "topic": "/pct/global_path",
        "frame_id": "world",
        "stamp": {"sec": stamp_sec, "nanosec": 0},
        "sequence": sequence,
        "points_sha256": _points_sha256(points),
        "cleared": cleared,
    }


def _state(
    step_index: int,
    report: dict[str, object] | None = None,
) -> SimulationState:
    metadata: dict[str, object] = {}
    if report is not None:
        metadata["scan_reference_path_last_report"] = report
    return SimulationState(
        step_index=step_index,
        timestamp=step_index * 0.02,
        robot_root_pose=(0.0, 0.0, 0.338, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0,) * 6,
        metadata=metadata,
    )


def _record(
    recorder: JsonlEpisodeRecorder,
    *,
    step_index: int,
    report: dict[str, object] | None,
) -> None:
    observation = _state(step_index, report)
    recorder.record_step(
        StepRecord(
            step_index=step_index,
            timestamp=observation.timestamp,
            pipeline_state="exec_nav_to_place",
            observation=observation,
            action=RobotAction(source="scan_cmd_vel"),
            post_step_observation=_state(step_index + 1, report),
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_path_snapshot_is_independent_from_frame_stride_and_deduplicated(
    tmp_path: Path,
) -> None:
    """Path 在非诊断帧到达时也必须完整落盘，且同一报告只写一次。"""

    recorder = JsonlEpisodeRecorder(
        tmp_path / "episode",
        diagnostic_frame_stride=100,
    )
    points = [[0.0, 0.0, 0.0], [1.0, 0.2, 1.5], [2.0, 0.0, 3.0]]
    report = _path_report(sequence=1, stamp_sec=2, points=points)

    _record(recorder, step_index=1, report=None)
    _record(recorder, step_index=2, report=report)
    _record(recorder, step_index=3, report=report)

    snapshots = _read_jsonl(recorder.navigation_path_snapshots_path)
    frames = _read_jsonl(recorder.frames_path)
    assert recorder.navigation_path_snapshot_count == 1
    assert len(snapshots) == 1
    assert snapshots[0]["schema"] == "navigation_path_snapshot_v1"
    assert snapshots[0]["snapshot_index"] == 1
    assert snapshots[0]["report"]["points_ground_xyz"] == points
    assert len(frames) == 1
    assert "scan_reference_path_last_report" not in frames[0]["observation"][
        "metadata"
    ]

    summary_path = recorder.close(
        {"success": False, "failure_reason": "unit_test_stop"}
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["navigation_path_snapshot_count"] == 1
    assert summary["navigation_path_snapshots_path"] == str(
        recorder.navigation_path_snapshots_path
    )


def test_new_path_generation_and_clear_tombstone_are_preserved_once(
    tmp_path: Path,
) -> None:
    """相同几何的新代际和清空墓碑都要保留，供重规划审计。"""

    recorder = JsonlEpisodeRecorder(tmp_path / "episode")
    points = [[0.0, 0.0, 0.0], [2.0, 0.0, 3.0]]
    first = _path_report(sequence=1, stamp_sec=2, points=points)
    second = _path_report(sequence=2, stamp_sec=3, points=points)
    tombstone = _path_report(
        sequence=3,
        stamp_sec=4,
        points=[],
        cleared=True,
    )

    _record(recorder, step_index=1, report=first)
    _record(recorder, step_index=2, report=second)
    _record(recorder, step_index=3, report=tombstone)
    _record(recorder, step_index=4, report=tombstone)

    snapshots = _read_jsonl(recorder.navigation_path_snapshots_path)
    assert recorder.navigation_path_snapshot_count == 3
    assert [snapshot["report"]["sequence"] for snapshot in snapshots] == [
        1,
        2,
        3,
    ]
    assert snapshots[-1]["report"]["cleared"] is True
    assert snapshots[-1]["report"]["points_ground_xyz"] == []
