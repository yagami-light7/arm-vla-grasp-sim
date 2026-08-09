from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from scripts.navigation import export_planner_comparison_contract as comparison


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture_run(tmp_path: Path) -> Path:
    """构造包含完整跨层Path、终止门和原checkpoint身份的最小fresh run。"""

    run_dir = tmp_path / "run"
    episode_dir = run_dir / "episode_000000"
    episode_dir.mkdir(parents=True)
    checkpoint = tmp_path / "model_26000.pt"
    checkpoint.write_bytes(b"original-go2-x5-checkpoint")
    task_path = tmp_path / "task.json"
    _write_json(task_path, {"task_id": 1002})
    scene_profile = tmp_path / "multi_floor.json"
    _write_json(scene_profile, {"profile": "multi_floor"})

    tuning = {
        "navigation_contract": {
            "ros__parameters": {"body_height_m": 0.338}
        },
        "scan_controller": {
            "ros__parameters": {
                **comparison.EXPECTED_FINISH_PARAMETERS,
                "finish.min_approach_speed": 0.22,
            }
        },
    }
    tuning_path = run_dir / "pct_scan_tuning_snapshot.yaml"
    tuning_path.write_text(
        yaml.safe_dump(tuning, sort_keys=False),
        encoding="utf-8",
    )
    tuning_raw = tuning_path.read_bytes()
    tuning_sha = hashlib.sha256(tuning_raw).hexdigest()

    source_files = [
        {
            "path": "scripts/navigation/example.py",
            "sha256": hashlib.sha256(b"example-source").hexdigest(),
            "byte_count": len(b"example-source"),
        }
    ]
    source_roots = ["scripts"]
    source_digest = comparison._canonical_payload_sha256(
        {
            "schema": "pct_scan_source_bundle_digest_v1",
            "source_roots": source_roots,
            "files": source_files,
        }
    )
    source_snapshot_path = run_dir / "pct_scan_source_bundle_snapshot.json"
    _write_json(
        source_snapshot_path,
        {
            "schema": "pct_scan_source_bundle_snapshot_v1",
            "source_roots": source_roots,
            "sha256": source_digest,
            "file_count": len(source_files),
            "total_bytes": len(b"example-source"),
            "files": source_files,
        },
    )
    source_snapshot_file_sha = hashlib.sha256(
        source_snapshot_path.read_bytes()
    ).hexdigest()

    stair = {
        "profile_id": "go2_x5_multifloor_scan_stair_freeze_v1",
        "contract_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "source_branch": "pct-scene",
        "baseline_behavior": "chassis_root_lock",
        "non_physical_root_lock_workaround": True,
    }
    _write_json(
        run_dir / "pct_scan_live_acceptance.json",
        {
            "status": "passed",
            "result": {"valid": True},
            "mode": "crossfloor_carry",
            "seed": 0,
            "run_id": "unit-run",
            "navigation_body_height_m": 0.338,
            "tuning_config_file": str(tuning_path),
            "tuning_config_snapshot": {
                "snapshot_path": str(tuning_path),
                "sha256": tuning_sha,
            },
            "source_bundle_snapshot": {
                "schema": "pct_scan_source_bundle_identity_v1",
                "snapshot_path": str(source_snapshot_path),
                "snapshot_file_sha256": source_snapshot_file_sha,
                "sha256": source_digest,
                "file_count": len(source_files),
                "total_bytes": len(b"example-source"),
                "source_roots": source_roots,
            },
            "source_bundle_verification": {
                "schema": "pct_scan_source_bundle_verification_v1",
                "expected_sha256": source_digest,
                "current_sha256": source_digest,
                "snapshot_file_sha256": source_snapshot_file_sha,
                "verified": True,
                "error": None,
            },
            "scan_stair_freeze_profile": stair,
        },
    )
    _write_json(
        run_dir / "startup_status.json",
        {
            "status": "completed",
            "exit_code": 0,
            "scene_profile_config_path": str(scene_profile),
            "task_json": str(task_path),
            "locomotion_task": comparison.EXPECTED_POLICY_TASK,
            "locomotion_checkpoint": str(checkpoint),
            "scene_profile_defaults_applied": {
                "locomotion_task": comparison.EXPECTED_POLICY_TASK,
            },
            "scan_stair_freeze_profile_runtime": stair,
        },
    )

    path_points = [
        [0.0, 0.0, -0.10],
        [1.0, 0.2, 1.40],
        [2.0, 0.0, 3.00],
    ]
    path_sha = comparison._path_points_sha256(path_points)
    terminal_yaw = 0.5
    yaw_start = 0.1
    path_report = {
        "points_ground_xyz": path_points,
        "terminal_yaw": terminal_yaw,
        "source": "ros2_nav_msgs_path",
        "topic": "/pct/global_path",
        "frame_id": "world",
        "stamp": {"sec": 2, "nanosec": 0},
        "sequence": 1,
        "points_sha256": path_sha,
        "cleared": False,
    }
    frame = {
        "timestamp": 1.6,
        "pipeline_state": "exec_nav_to_place",
        "observation": {
            "robot_root_pose": [
                0.0,
                0.0,
                0.238,
                math.cos(yaw_start / 2.0),
                0.0,
                0.0,
                math.sin(yaw_start / 2.0),
            ],
            "metadata": {
                "scan_reference_path_last_report": {
                    key: value
                    for key, value in path_report.items()
                    if key != "points_ground_xyz"
                }
            },
        },
    }
    (episode_dir / "frames.jsonl").write_text(
        json.dumps(frame) + "\n",
        encoding="utf-8",
    )
    snapshot = {
        "schema": "navigation_path_snapshot_v1",
        "snapshot_index": 1,
        "captured_from": "observation",
        "step_index": 80,
        "timestamp": 1.6,
        "pipeline_state": "exec_nav_to_place",
        "state_step_index": 79,
        "state_timestamp": 1.58,
        "report_payload_sha256": comparison._canonical_payload_sha256(
            path_report
        ),
        "report": path_report,
    }
    (episode_dir / "navigation_path_snapshots.jsonl").write_text(
        json.dumps(snapshot) + "\n",
        encoding="utf-8",
    )
    _write_json(
        episode_dir / "summary.json",
        {
            "seed": 0,
            "task_id": 1002,
            "success": True,
            "final_state": "done",
            "latest_planner_result": {
                "pct_goal_request": {
                    "position_base_xyz": [2.0, 0.0, 3.338],
                    "yaw": terminal_yaw,
                    "height_semantics": "base",
                    "effective_goal_provenance": {
                        "calibration": {
                            "collision_ply": "/runtime/collision.ply",
                            "collision_ply_sha256": "c" * 64,
                        }
                    },
                },
                "navigation_execution": dict(
                    comparison.EXPECTED_TERMINATION
                ),
            },
            "latest_executor_status": {
                "done": True,
                "success": True,
                "scan_controller_goal_reached_verified": True,
                "policy_zero_hold_verified": True,
                "post_goal_nonzero_write_count": 0,
                "required_zero_write_ticks": 5,
            },
        },
    )
    return run_dir


def _load_frame(run_dir: Path) -> dict[str, object]:
    return json.loads(
        (run_dir / "episode_000000/frames.jsonl").read_text(encoding="utf-8")
    )


def _load_path_snapshots(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (
            run_dir / "episode_000000/navigation_path_snapshots.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_path_snapshots(
    run_dir: Path,
    snapshots: list[dict[str, object]],
) -> None:
    path = run_dir / "episode_000000/navigation_path_snapshots.jsonl"
    path.write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )


def test_builds_eligible_contract_with_exact_serialized_path(tmp_path: Path) -> None:
    contract = comparison.build_comparison_contract(_fixture_run(tmp_path))

    assert contract["eligible_as_crossplanner_navigation_source"] is True
    assert contract["ineligibility_reasons"] == []
    assert contract["source_run"]["planner"] == "scan"
    assert contract["global_path"]["point_count"] == 3
    assert contract["global_path"]["height_semantics"] == "ground"
    assert contract["global_path"]["evidence"]["kind"] == (
        "dedicated_navigation_path_snapshots_jsonl"
    )
    assert contract["global_path"]["source_sequence_values"] == [1]
    assert contract["global_path"]["points_sha256"] == (
        comparison._path_points_sha256(
            contract["global_path"]["points_ground_xyz"]
        )
    )
    assert contract["goal"]["serialized_path_endpoint_base_xyz"] == pytest.approx(
        [2.0, 0.0, 3.338]
    )
    assert contract["termination_contract"]["final_position_tolerance"] == 0.08
    assert contract["termination_contract"]["stable_dwell_s"] == 0.50
    assert contract["shared_runtime_contract"]["policy_task"] == (
        comparison.EXPECTED_POLICY_TASK
    )
    assert contract["shared_runtime_contract"]["tuning"][
        "immutable_run_snapshot"
    ] is True
    assert contract["shared_runtime_contract"]["source_revision"][
        "immutable_run_snapshot"
    ] is True
    assert comparison.verify_contract_payload(contract) is True


def test_rejects_path_payload_hash_mismatch(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    snapshots = _load_path_snapshots(run_dir)
    report = snapshots[0]["report"]
    report["points_sha256"] = "f" * 64
    snapshots[0]["report_payload_sha256"] = (
        comparison._canonical_payload_sha256(report)
    )
    _write_path_snapshots(run_dir, snapshots)

    with pytest.raises(comparison.ComparisonContractError, match="哈希不匹配"):
        comparison.build_comparison_contract(run_dir)


def test_rejects_tampered_snapshot_report_digest(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    snapshots = _load_path_snapshots(run_dir)
    snapshots[0]["report"]["sequence"] = 9
    _write_path_snapshots(run_dir, snapshots)

    with pytest.raises(
        comparison.ComparisonContractError,
        match="报告摘要不匹配",
    ):
        comparison.build_comparison_contract(run_dir)


def test_rejects_multiple_distinct_path_generations(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    snapshots = _load_path_snapshots(run_dir)
    second = json.loads(json.dumps(snapshots[0]))
    second["snapshot_index"] = 2
    second["timestamp"] = 2.0
    report = second["report"]
    report["points_ground_xyz"][1][1] = 0.4
    report["points_sha256"] = comparison._path_points_sha256(
        report["points_ground_xyz"]
    )
    report["stamp"] = {"sec": 3, "nanosec": 0}
    report["sequence"] = 2
    second["report_payload_sha256"] = comparison._canonical_payload_sha256(
        report
    )
    _write_path_snapshots(run_dir, [snapshots[0], second])

    with pytest.raises(comparison.ComparisonContractError, match="唯一Path几何"):
        comparison.build_comparison_contract(run_dir)


def test_old_compact_frames_without_snapshot_fail_with_actionable_reason(
    tmp_path: Path,
) -> None:
    run_dir = _fixture_run(tmp_path)
    (run_dir / "episode_000000/navigation_path_snapshots.jsonl").unlink()

    with pytest.raises(
        comparison.ComparisonContractError,
        match="只保留了Path哈希摘要",
    ):
        comparison.build_comparison_contract(run_dir)


def test_legacy_full_frame_path_remains_readable(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    snapshots = _load_path_snapshots(run_dir)
    report = snapshots[0]["report"]
    frame = _load_frame(run_dir)
    frame["observation"]["metadata"]["scan_reference_path_last_report"] = report
    (run_dir / "episode_000000/frames.jsonl").write_text(
        json.dumps(frame) + "\n",
        encoding="utf-8",
    )
    (run_dir / "episode_000000/navigation_path_snapshots.jsonl").unlink()

    contract = comparison.build_comparison_contract(run_dir)

    assert contract["global_path"]["evidence"]["kind"] == (
        "legacy_full_diagnostic_frame"
    )


def test_incomplete_source_is_diagnostic_only(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    manifest_path = run_dir / "pct_scan_live_acceptance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest["result"] = None
    _write_json(manifest_path, manifest)
    summary_path = run_dir / "episode_000000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["success"] = False
    summary["final_state"] = "exec_nav_to_place"
    summary["latest_executor_status"]["done"] = False
    summary["latest_executor_status"]["success"] = False
    _write_json(summary_path, summary)

    with pytest.raises(comparison.ComparisonContractError, match="不具备"):
        comparison.build_comparison_contract(run_dir)

    diagnostic = comparison.build_comparison_contract(
        run_dir,
        allow_incomplete_source=True,
    )
    assert diagnostic["eligible_as_crossplanner_navigation_source"] is False
    assert "acceptance_manifest_not_passed" in diagnostic[
        "ineligibility_reasons"
    ]
    assert "episode_not_successful" in diagnostic["ineligibility_reasons"]


def test_rejects_modified_tuning_snapshot(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    with (run_dir / "pct_scan_tuning_snapshot.yaml").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("# changed after run\n")

    with pytest.raises(comparison.ComparisonContractError, match="快照哈希"):
        comparison.build_comparison_contract(run_dir)


def test_rejects_modified_source_bundle_snapshot(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    with (run_dir / "pct_scan_source_bundle_snapshot.json").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")

    with pytest.raises(comparison.ComparisonContractError, match="源码bundle快照文件哈希"):
        comparison.build_comparison_contract(run_dir)


def test_rejects_relaxed_terminal_contract(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    summary_path = run_dir / "episode_000000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["latest_planner_result"]["navigation_execution"][
        "final_position_tolerance"
    ] = 0.18
    _write_json(summary_path, summary)

    with pytest.raises(comparison.ComparisonContractError, match="公平基准"):
        comparison.build_comparison_contract(run_dir)


def test_writer_refuses_overwrite_and_preserves_contract_hash(tmp_path: Path) -> None:
    contract = comparison.build_comparison_contract(_fixture_run(tmp_path))
    output = tmp_path / "comparison_contract.json"

    written = comparison.write_comparison_contract(contract, output)

    assert written == output.resolve()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert comparison.verify_contract_payload(loaded) is True
    with pytest.raises(comparison.ComparisonContractError, match="原本不存在"):
        comparison.write_comparison_contract(contract, output)
