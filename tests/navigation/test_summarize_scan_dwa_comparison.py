from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.navigation import summarize_scan_dwa_comparison as comparison


def _trial(
    planner: str,
    seed: int,
    *,
    success: bool,
    completion_s: float | None,
    failure_s: float | None = None,
    tuning_sha: str | None = None,
    source_sha: str | None = None,
) -> comparison.PlannerTrial:
    return comparison.PlannerTrial(
        planner=planner,
        seed=seed,
        run_dir=f"/{planner}/{seed}",
        success=success,
        failure_reason="" if success else "nav_collision",
        contract_payload_sha256=(f"{seed + 1:064x}"),
        path_points_sha256="a" * 64,
        path_point_count=146,
        termination_contract_json='{"final_position_tolerance":0.08}',
        checkpoint_sha256="b" * 64,
        collision_ply_sha256="c" * 64,
        task_sha256="d" * 64,
        stair_contract_sha256="e" * 64,
        tuning_sha256=tuning_sha,
        source_revision_sha256=source_sha,
        planner_controlled_completion_s=completion_s,
        planner_controlled_time_to_failure_s=failure_s,
        full_navigation_stage_s=(completion_s + 60.0 if completion_s else None),
        common_stair_freeze_s=60.0,
        terminal_capture_s=(5.0 if success else None),
        zero_command_s=(1.0 if success else None),
        planner_compute_wall_s=1.0,
        planner_attempt_count=10,
        failure_signature_json=(
            None
            if success
            else '{"failure_reason":"nav_collision","feasible_candidates":0}'
        ),
    )


def _three_scan_successes(
    *,
    tuning_sha: str | None = "f" * 64,
    source_sha: str | None = "1" * 64,
) -> list[comparison.PlannerTrial]:
    return [
        _trial(
            "scan",
            seed,
            success=True,
            completion_s=40.0 + seed,
            tuning_sha=tuning_sha,
            source_sha=source_sha,
        )
        for seed in range(3)
    ]


def _three_dwa_failures() -> list[comparison.PlannerTrial]:
    return [
        _trial(
            "dwa",
            seed,
            success=False,
            completion_s=None,
            failure_s=39.26,
        )
        for seed in range(3)
    ]


def test_success_rate_superiority_wins_without_faking_completion_time() -> None:
    report = comparison.summarize_trials(
        _three_scan_successes(),
        _three_dwa_failures(),
    )

    assert report["scan"]["success_count"] == 3
    assert report["dwa"]["success_count"] == 0
    assert report["dwa"]["planner_controlled_completion_mean_s"] is None
    assert report["dwa"]["planner_controlled_time_to_failure_mean_s"] == (
        pytest.approx(39.26)
    )
    assert report["verdict"]["navigation_success_superiority_verified"] is True
    assert report["verdict"]["paired_completion_time_comparable"] is False
    assert report["verdict"]["completion_speed_superiority_verified"] is False
    assert report["verdict"]["overall_navigation_performance_exceeds_dwa"] is True
    assert report["verdict"]["verdict_basis"] == "higher_navigation_success_rate"
    assert report["dwa"]["deterministic_failure_signature"] is True


def test_equal_success_rate_uses_paired_completion_time() -> None:
    scan = _three_scan_successes()
    dwa = [
        _trial("dwa", seed, success=True, completion_s=50.0 + seed)
        for seed in range(3)
    ]

    report = comparison.summarize_trials(scan, dwa)

    assert report["verdict"]["navigation_success_superiority_verified"] is False
    assert report["verdict"]["paired_completion_time_comparable"] is True
    assert report["verdict"]["completion_speed_superiority_verified"] is True
    assert report["verdict"]["overall_navigation_performance_exceeds_dwa"] is True
    assert report["verdict"]["verdict_basis"] == (
        "equal_success_rate_and_lower_completion_time"
    )


def test_rejects_unpaired_seed_sets() -> None:
    dwa = _three_dwa_failures()[:-1]

    with pytest.raises(comparison.ComparisonSummaryError, match="seed集合不同"):
        comparison.summarize_trials(_three_scan_successes(), dwa)


def test_rejects_path_contract_mismatch() -> None:
    dwa = _three_dwa_failures()
    dwa[2] = replace(dwa[2], path_points_sha256="9" * 64)

    with pytest.raises(comparison.ComparisonSummaryError, match="共享合同字段不一致"):
        comparison.summarize_trials(_three_scan_successes(), dwa)


def test_rejects_different_contract_for_same_seed() -> None:
    dwa = _three_dwa_failures()
    dwa[1] = replace(dwa[1], contract_payload_sha256="9" * 64)

    with pytest.raises(comparison.ComparisonSummaryError, match="同一份比较合同"):
        comparison.summarize_trials(_three_scan_successes(), dwa)


def test_reports_missing_exact_revision_evidence_without_invalidating_history() -> None:
    scan = _three_scan_successes(source_sha=None)

    report = comparison.summarize_trials(scan, _three_dwa_failures())

    assert report["scan"]["tuning_snapshot_uniform"] is True
    assert report["scan"]["source_revision_uniform"] is False
    assert report["verdict"]["current_exact_revision_three_seed_verified"] is False
    assert any("源码bundle" in caveat for caveat in report["caveats"])


def test_reports_nonuniform_scan_tuning() -> None:
    scan = _three_scan_successes()
    scan[0] = replace(scan[0], tuning_sha256="2" * 64)

    report = comparison.summarize_trials(scan, _three_dwa_failures())

    assert report["scan"]["tuning_snapshot_uniform"] is False
    assert report["verdict"]["current_exact_revision_three_seed_verified"] is False
    assert any("YAML调参快照" in caveat for caveat in report["caveats"])
