#!/usr/bin/env python3
"""汇总同一 PCT Path 与终止合同下的 SCAN/DWA 多 seed 对照。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.navigation.analyze_pct_scan_live_timing import (
    TimingInputError,
    analyze_episode,
)
from scripts.navigation.export_planner_comparison_contract import (
    verify_contract_payload,
)


EXPECTED_CONTRACT_SCHEMA = "pct_local_planner_comparison_contract_v1"
EXPECTED_DWA_SCHEMA = "legacy_dwa_comparison_run_v1"
SUMMARY_SCHEMA = "pct_scan_dwa_comparison_summary_v1"


class ComparisonSummaryError(ValueError):
    """表示输入产物不足以形成公平、可复核的多 seed 结论。"""


@dataclass(frozen=True, slots=True)
class PlannerTrial:
    """保留跨规划器配对与汇总所需的不可变字段。"""

    planner: str
    seed: int
    run_dir: str
    success: bool
    failure_reason: str
    contract_payload_sha256: str
    path_points_sha256: str
    path_point_count: int
    termination_contract_json: str
    checkpoint_sha256: str
    collision_ply_sha256: str
    task_sha256: str
    stair_contract_sha256: str
    tuning_sha256: str | None
    source_revision_sha256: str | None
    planner_controlled_completion_s: float | None
    planner_controlled_time_to_failure_s: float | None
    full_navigation_stage_s: float | None
    common_stair_freeze_s: float
    terminal_capture_s: float | None
    zero_command_s: float | None
    planner_compute_wall_s: float
    planner_attempt_count: int
    failure_signature_json: str | None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonSummaryError(f"{label} 必须是 JSON 对象。")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonSummaryError(
            f"无法读取{label} {path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ComparisonSummaryError(f"{label}顶层必须是 JSON 对象：{path}")
    return payload


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonSummaryError(f"{label} 必须是有限数。")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonSummaryError(f"{label} 必须是有限数。")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    number = _finite(value, label)
    if not number.is_integer() or number < minimum:
        raise ComparisonSummaryError(f"{label} 必须是 >= {minimum} 的整数。")
    return int(number)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ComparisonSummaryError(f"{label} 必须是小写 SHA256。")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _load_json(path, "比较合同")
    if contract.get("schema") != EXPECTED_CONTRACT_SCHEMA:
        raise ComparisonSummaryError(f"比较合同 schema 不受支持：{path}")
    if not verify_contract_payload(contract):
        raise ComparisonSummaryError(f"比较合同 payload SHA256 无效：{path}")
    if contract.get("eligible_as_crossplanner_navigation_source") is not True:
        raise ComparisonSummaryError(f"比较合同没有通过公平输入门：{path}")
    if contract.get("ineligibility_reasons") != []:
        raise ComparisonSummaryError(f"比较合同仍包含不合格原因：{path}")
    return contract


def _contract_fields(contract: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(contract.get("source_run"), "contract.source_run")
    path = _mapping(contract.get("global_path"), "contract.global_path")
    shared = _mapping(
        contract.get("shared_runtime_contract"),
        "contract.shared_runtime_contract",
    )
    checkpoint = _mapping(
        shared.get("policy_checkpoint"),
        "contract.shared_runtime_contract.policy_checkpoint",
    )
    collision = _mapping(
        shared.get("collision_ply"),
        "contract.shared_runtime_contract.collision_ply",
    )
    task = _mapping(
        shared.get("task_json"),
        "contract.shared_runtime_contract.task_json",
    )
    stair = _mapping(
        shared.get("stair_freeze"),
        "contract.shared_runtime_contract.stair_freeze",
    )
    tuning = shared.get("tuning")
    tuning_sha = None
    if isinstance(tuning, Mapping):
        tuning_sha = _sha256(
            tuning.get("sha256"),
            "contract.shared_runtime_contract.tuning.sha256",
        )
    source_revision = shared.get("source_revision")
    source_revision_sha = None
    if isinstance(source_revision, Mapping):
        source_revision_sha = _sha256(
            source_revision.get("sha256"),
            "contract.shared_runtime_contract.source_revision.sha256",
        )
    return {
        "seed": _integer(source.get("seed"), "contract.source_run.seed"),
        "contract_payload_sha256": _sha256(
            contract.get("contract_payload_sha256"),
            "contract.contract_payload_sha256",
        ),
        "path_points_sha256": _sha256(
            path.get("points_sha256"),
            "contract.global_path.points_sha256",
        ),
        "path_point_count": _integer(
            path.get("point_count"),
            "contract.global_path.point_count",
            minimum=2,
        ),
        "termination_contract_json": _canonical_json(
            _mapping(
                contract.get("termination_contract"),
                "contract.termination_contract",
            )
        ),
        "checkpoint_sha256": _sha256(
            checkpoint.get("sha256"),
            "contract.policy_checkpoint.sha256",
        ),
        "collision_ply_sha256": _sha256(
            collision.get("sha256"),
            "contract.collision_ply.sha256",
        ),
        "task_sha256": _sha256(
            task.get("sha256"),
            "contract.task_json.sha256",
        ),
        "stair_contract_sha256": _sha256(
            stair.get("contract_sha256"),
            "contract.stair_freeze.contract_sha256",
        ),
        "tuning_sha256": tuning_sha,
        "source_revision_sha256": source_revision_sha,
    }


def load_scan_trial(run_value: str | Path) -> PlannerTrial:
    """读取一次已通过的 SCAN 跨层验收及其不可变比较合同。"""

    run_dir = Path(run_value).expanduser().resolve()
    contract = _load_contract(run_dir / "planner_comparison_contract.json")
    fields = _contract_fields(contract)
    acceptance = _load_json(run_dir / "pct_scan_live_acceptance.json", "SCAN验收")
    if acceptance.get("status") != "passed":
        raise ComparisonSummaryError(f"SCAN验收未通过：{run_dir}")
    if _integer(acceptance.get("seed"), "acceptance.seed") != fields["seed"]:
        raise ComparisonSummaryError(f"SCAN验收 seed 与合同不一致：{run_dir}")
    try:
        timing = analyze_episode(run_dir)
    except TimingInputError as exc:
        raise ComparisonSummaryError(f"SCAN计时分析失败 {run_dir}：{exc}") from exc
    if timing.get("success") is not True or timing.get("final_state") != "done":
        raise ComparisonSummaryError(f"SCAN episode 未成功完成：{run_dir}")
    if _integer(timing.get("seed"), "timing.seed") != fields["seed"]:
        raise ComparisonSummaryError(f"SCAN计时 seed 与合同不一致：{run_dir}")
    crossfloor = _mapping(timing.get("crossfloor"), "timing.crossfloor")
    planner = _mapping(
        timing.get("planner_crossfloor"),
        "timing.planner_crossfloor",
    )
    primary = _finite(
        crossfloor.get("planner_controlled_navigation_sim_time_s"),
        "crossfloor.planner_controlled_navigation_sim_time_s",
    )
    return PlannerTrial(
        planner="scan",
        seed=fields["seed"],
        run_dir=str(run_dir),
        success=True,
        failure_reason="",
        contract_payload_sha256=fields["contract_payload_sha256"],
        path_points_sha256=fields["path_points_sha256"],
        path_point_count=fields["path_point_count"],
        termination_contract_json=fields["termination_contract_json"],
        checkpoint_sha256=fields["checkpoint_sha256"],
        collision_ply_sha256=fields["collision_ply_sha256"],
        task_sha256=fields["task_sha256"],
        stair_contract_sha256=fields["stair_contract_sha256"],
        tuning_sha256=fields["tuning_sha256"],
        source_revision_sha256=fields["source_revision_sha256"],
        planner_controlled_completion_s=primary,
        planner_controlled_time_to_failure_s=None,
        full_navigation_stage_s=_finite(
            crossfloor.get("full_navigation_stage_sim_time_s"),
            "crossfloor.full_navigation_stage_sim_time_s",
        ),
        common_stair_freeze_s=_finite(
            crossfloor.get("common_stair_freeze_sim_time_s"),
            "crossfloor.common_stair_freeze_sim_time_s",
        ),
        terminal_capture_s=_finite(
            crossfloor.get("terminal_capture_sim_time_s"),
            "crossfloor.terminal_capture_sim_time_s",
        ),
        zero_command_s=_finite(
            crossfloor.get("zero_command_sim_time_s"),
            "crossfloor.zero_command_sim_time_s",
        ),
        planner_compute_wall_s=_finite(
            planner.get("successful_total_wall_s"),
            "planner_crossfloor.successful_total_wall_s",
        ),
        planner_attempt_count=_integer(
            planner.get("attempt_count"),
            "planner_crossfloor.attempt_count",
        ),
        failure_signature_json=None,
    )


def _resolve_dwa_contract_path(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> Path:
    descriptor = _mapping(
        manifest.get("comparison_contract"),
        "legacy_dwa.comparison_contract",
    )
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ComparisonSummaryError(
            f"DWA manifest 缺少比较合同路径：{manifest_path}"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_dwa_trial(run_value: str | Path) -> PlannerTrial:
    """读取一次隔离旧 worktree 的有效 DWA 成功或算法失败试验。"""

    run_dir = Path(run_value).expanduser().resolve()
    manifest_path = run_dir / "legacy_dwa_comparison.json"
    manifest = _load_json(manifest_path, "DWA对照manifest")
    if manifest.get("schema") != EXPECTED_DWA_SCHEMA:
        raise ComparisonSummaryError(f"DWA manifest schema 不受支持：{run_dir}")
    if manifest.get("production_runtime_unchanged") is not True:
        raise ComparisonSummaryError(f"DWA试验修改了生产运行链：{run_dir}")
    if manifest.get("dwa_imported_into_pct_scan_runtime") is not False:
        raise ComparisonSummaryError(f"DWA被导入pct-scan生产运行时：{run_dir}")
    if manifest.get("legacy_branch") != "pct_scene":
        raise ComparisonSummaryError(f"DWA没有运行在pct-scene基线分支：{run_dir}")

    contract = _load_contract(_resolve_dwa_contract_path(manifest, manifest_path))
    fields = _contract_fields(contract)
    descriptor = _mapping(
        manifest.get("comparison_contract"),
        "legacy_dwa.comparison_contract",
    )
    if descriptor.get("contract_payload_sha256") != fields["contract_payload_sha256"]:
        raise ComparisonSummaryError(f"DWA manifest 合同摘要不匹配：{run_dir}")
    raw_analysis = manifest.get("navigation_analysis")
    analysis_from_replay = False
    replay_path = run_dir / "legacy_dwa_analysis.json"
    if replay_path.is_file():
        analysis = _load_json(replay_path, "DWA流式重算分析")
        analysis_from_replay = True
    elif isinstance(raw_analysis, Mapping):
        analysis = raw_analysis
    else:
        raise ComparisonSummaryError(
            f"DWA manifest缺少流式导航分析，且没有重算产物：{run_dir}"
        )
    if analysis.get("schema") != "legacy_dwa_navigation_analysis_v2":
        raise ComparisonSummaryError(
            f"DWA分析缺少二楼交接兼容性审计，请先重新分析：{run_dir}"
        )
    handoff = _mapping(analysis.get("handoff"), "navigation_analysis.handoff")
    handoff_audit = _mapping(
        handoff.get("contract_audit"),
        "navigation_analysis.handoff.contract_audit",
    )
    if handoff_audit.get("contract_compatible") is not True:
        raise ComparisonSummaryError(
            f"DWA试验的楼梯后交接点不是双方共同自由空间：{run_dir}"
        )
    raw_validation = analysis.get("summary_validation")
    if isinstance(raw_validation, Mapping):
        validation = raw_validation
    else:
        validation = _mapping(
            manifest.get("summary_validation"),
            "legacy_dwa.summary_validation",
        )
    if (
        analysis.get("comparison_trial_valid") is not True
        or validation.get("comparison_trial_valid") is not True
    ):
        raise ComparisonSummaryError(f"DWA试验合同无效：{run_dir}")
    seed = _integer(analysis.get("seed"), "navigation_analysis.seed")
    if seed != fields["seed"]:
        raise ComparisonSummaryError(f"DWA seed 与合同不一致：{run_dir}")
    if analysis.get("global_path_points_sha256") != fields["path_points_sha256"]:
        raise ComparisonSummaryError(f"DWA实际消费的Path摘要不匹配：{run_dir}")
    if _integer(
        analysis.get("global_path_point_count"),
        "navigation_analysis.global_path_point_count",
        minimum=2,
    ) != fields["path_point_count"]:
        raise ComparisonSummaryError(f"DWA实际消费的Path点数不匹配：{run_dir}")

    success = analysis.get("navigation_success") is True
    failure_reason = analysis.get("failure_reason", "")
    if not isinstance(failure_reason, str):
        raise ComparisonSummaryError(f"DWA failure_reason 必须是字符串：{run_dir}")
    expected_status = "passed" if success else "completed_navigation_failure"
    if manifest.get("status") != expected_status:
        # phase282 v3 生成于“有效算法失败”状态语义收口前，旧父进程把
        # child=0 的 nav_collision 写成 failed。只有同目录新版流式重算、
        # child正常退出且没有wrapper异常三项同时成立时才兼容这份历史证据。
        legacy_failure_reclassified = bool(
            analysis_from_replay
            and not success
            and manifest.get("status") == "failed"
            and manifest.get("child_returncode") == 0
            and manifest.get("error") is None
        )
        if not legacy_failure_reclassified:
            raise ComparisonSummaryError(
                f"DWA manifest状态与导航结果不一致：{run_dir}，"
                f"expected={expected_status!r}, actual={manifest.get('status')!r}"
            )
    timing = _mapping(analysis.get("timing"), "navigation_analysis.timing")
    planner_controlled = _finite(
        timing.get("planner_controlled_to_outcome_sim_time_s"),
        "navigation_analysis.timing.planner_controlled_to_outcome_sim_time_s",
    )
    failed_time_flag = timing.get("failed_run_time_is_time_to_failure_not_completion")
    if failed_time_flag is not (not success):
        raise ComparisonSummaryError(f"DWA完成/失败时间语义标志错误：{run_dir}")
    compute = _mapping(analysis.get("dwa_compute"), "navigation_analysis.dwa_compute")
    diagnostics = _mapping(
        analysis.get("outcome_diagnostics"),
        "navigation_analysis.outcome_diagnostics",
    )
    failure_signature = None
    if not success:
        release_bridge = diagnostics.get("release_bridge")
        failure_signature = _canonical_json(
            {
                "failure_reason": failure_reason,
                "active_map": diagnostics.get("active_map"),
                "post_stair_replan_reason": diagnostics.get(
                    "post_stair_replan_reason"
                ),
                "clearance_m": diagnostics.get("clearance_m"),
                "sampled_candidates": diagnostics.get("sampled_candidates"),
                "feasible_candidates": diagnostics.get("feasible_candidates"),
                "collision_rejections": diagnostics.get("collision_rejections"),
                "occupied_start_escape_candidates": diagnostics.get(
                    "occupied_start_escape_candidates"
                ),
                "release_bridge": release_bridge,
            }
        )
    return PlannerTrial(
        planner="dwa",
        seed=seed,
        run_dir=str(run_dir),
        success=success,
        failure_reason=failure_reason,
        contract_payload_sha256=fields["contract_payload_sha256"],
        path_points_sha256=fields["path_points_sha256"],
        path_point_count=fields["path_point_count"],
        termination_contract_json=fields["termination_contract_json"],
        checkpoint_sha256=fields["checkpoint_sha256"],
        collision_ply_sha256=fields["collision_ply_sha256"],
        task_sha256=fields["task_sha256"],
        stair_contract_sha256=fields["stair_contract_sha256"],
        tuning_sha256=None,
        source_revision_sha256=None,
        planner_controlled_completion_s=planner_controlled if success else None,
        planner_controlled_time_to_failure_s=(
            None if success else planner_controlled
        ),
        full_navigation_stage_s=None,
        common_stair_freeze_s=_finite(
            timing.get("common_stair_freeze_sim_time_s"),
            "navigation_analysis.timing.common_stair_freeze_sim_time_s",
        ),
        terminal_capture_s=None,
        zero_command_s=None,
        planner_compute_wall_s=_finite(
            compute.get("successful_compute_wall_time_total_s"),
            "navigation_analysis.dwa_compute.successful_compute_wall_time_total_s",
        ),
        planner_attempt_count=_integer(
            compute.get("unique_recompute_count"),
            "navigation_analysis.dwa_compute.unique_recompute_count",
        ),
        failure_signature_json=failure_signature,
    )


def _unique_by_seed(
    trials: Sequence[PlannerTrial],
    planner: str,
) -> dict[int, PlannerTrial]:
    result: dict[int, PlannerTrial] = {}
    for trial in trials:
        if trial.planner != planner:
            raise ComparisonSummaryError(
                f"{planner}输入包含planner={trial.planner!r}的试验。"
            )
        if trial.seed in result:
            raise ComparisonSummaryError(f"{planner} seed={trial.seed}重复。")
        result[trial.seed] = trial
    return result


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_trials(
    scan_trials: Sequence[PlannerTrial],
    dwa_trials: Sequence[PlannerTrial],
) -> dict[str, Any]:
    """验证配对合同并按“成功率优先、成功耗时其次”形成结论。"""

    scan_by_seed = _unique_by_seed(scan_trials, "scan")
    dwa_by_seed = _unique_by_seed(dwa_trials, "dwa")
    if set(scan_by_seed) != set(dwa_by_seed):
        raise ComparisonSummaryError(
            f"SCAN/DWA seed集合不同：scan={sorted(scan_by_seed)}, "
            f"dwa={sorted(dwa_by_seed)}"
        )
    seeds = sorted(scan_by_seed)
    if not seeds:
        raise ComparisonSummaryError("至少需要一个配对seed。")

    invariant_fields = (
        "path_points_sha256",
        "path_point_count",
        "termination_contract_json",
        "checkpoint_sha256",
        "collision_ply_sha256",
        "task_sha256",
        "stair_contract_sha256",
    )
    all_trials = [*scan_trials, *dwa_trials]
    for field in invariant_fields:
        values = {getattr(trial, field) for trial in all_trials}
        if len(values) != 1:
            raise ComparisonSummaryError(
                f"跨试验共享合同字段不一致：{field}={sorted(values, key=str)}"
            )
    for seed in seeds:
        if (
            scan_by_seed[seed].contract_payload_sha256
            != dwa_by_seed[seed].contract_payload_sha256
        ):
            raise ComparisonSummaryError(
                f"seed={seed}的SCAN/DWA没有消费同一份比较合同。"
            )

    minimum_seed_count = 3
    minimum_seed_count_met = len(seeds) >= minimum_seed_count
    scan_success_count = sum(trial.success for trial in scan_trials)
    dwa_success_count = sum(trial.success for trial in dwa_trials)
    scan_success_rate = scan_success_count / len(scan_trials)
    dwa_success_rate = dwa_success_count / len(dwa_trials)
    scan_completion_times = [
        trial.planner_controlled_completion_s
        for trial in scan_trials
        if trial.success and trial.planner_controlled_completion_s is not None
    ]
    dwa_completion_times = [
        trial.planner_controlled_completion_s
        for trial in dwa_trials
        if trial.success and trial.planner_controlled_completion_s is not None
    ]
    paired_success_seeds = [
        seed
        for seed in seeds
        if scan_by_seed[seed].success and dwa_by_seed[seed].success
    ]
    paired_scan_times = [
        float(scan_by_seed[seed].planner_controlled_completion_s)
        for seed in paired_success_seeds
    ]
    paired_dwa_times = [
        float(dwa_by_seed[seed].planner_controlled_completion_s)
        for seed in paired_success_seeds
    ]
    enough_paired_completions = (
        len(paired_success_seeds) >= minimum_seed_count
    )
    completion_speed_superiority = bool(
        enough_paired_completions
        and statistics.fmean(paired_scan_times)
        < statistics.fmean(paired_dwa_times)
    )
    navigation_success_superiority = bool(
        minimum_seed_count_met and scan_success_rate > dwa_success_rate
    )
    equal_success_rate = math.isclose(
        scan_success_rate,
        dwa_success_rate,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    overall_exceeds = navigation_success_superiority or bool(
        minimum_seed_count_met
        and equal_success_rate
        and completion_speed_superiority
    )
    if navigation_success_superiority:
        verdict_basis = "higher_navigation_success_rate"
    elif equal_success_rate and completion_speed_superiority:
        verdict_basis = "equal_success_rate_and_lower_completion_time"
    elif not minimum_seed_count_met:
        verdict_basis = "insufficient_seed_count"
    elif not enough_paired_completions:
        verdict_basis = "completion_time_not_pairwise_evaluable"
    else:
        verdict_basis = "scan_did_not_exceed_dwa"

    tuning_shas = sorted(
        {
            trial.tuning_sha256
            for trial in scan_trials
            if trial.tuning_sha256 is not None
        }
    )
    source_shas = sorted(
        {
            trial.source_revision_sha256
            for trial in scan_trials
            if trial.source_revision_sha256 is not None
        }
    )
    scan_tuning_uniform = (
        len(tuning_shas) == 1
        and all(trial.tuning_sha256 is not None for trial in scan_trials)
    )
    scan_source_revision_uniform = (
        len(source_shas) == 1
        and all(
            trial.source_revision_sha256 is not None
            for trial in scan_trials
        )
    )
    exact_current_revision_three_seed_verified = bool(
        minimum_seed_count_met
        and scan_tuning_uniform
        and scan_source_revision_uniform
    )
    failure_signatures = {
        trial.failure_signature_json
        for trial in dwa_trials
        if not trial.success and trial.failure_signature_json is not None
    }
    dwa_time_to_failure = [
        trial.planner_controlled_time_to_failure_s
        for trial in dwa_trials
        if trial.planner_controlled_time_to_failure_s is not None
    ]
    shared = scan_trials[0]
    paired = []
    for seed in seeds:
        scan = scan_by_seed[seed]
        dwa = dwa_by_seed[seed]
        paired.append(
            {
                "seed": seed,
                "contract_payload_sha256": scan.contract_payload_sha256,
                "scan": {
                    "success": scan.success,
                    "planner_controlled_completion_s": (
                        scan.planner_controlled_completion_s
                    ),
                    "terminal_capture_s": scan.terminal_capture_s,
                    "zero_command_s": scan.zero_command_s,
                    "planner_compute_wall_s": scan.planner_compute_wall_s,
                    "run_dir": scan.run_dir,
                },
                "dwa": {
                    "success": dwa.success,
                    "failure_reason": dwa.failure_reason,
                    "planner_controlled_completion_s": (
                        dwa.planner_controlled_completion_s
                    ),
                    "planner_controlled_time_to_failure_s": (
                        dwa.planner_controlled_time_to_failure_s
                    ),
                    "planner_compute_wall_s": dwa.planner_compute_wall_s,
                    "run_dir": dwa.run_dir,
                },
            }
        )
    caveats = []
    if not enough_paired_completions:
        caveats.append(
            "DWA没有足够成功episode；其失败前时间不能冒充完成时间，"
            "因此当前不能声明SCAN的完成耗时更低。"
        )
    if not scan_tuning_uniform:
        caveats.append("SCAN输入不是同一份不可变YAML调参快照。")
    if not scan_source_revision_uniform:
        caveats.append(
            "历史SCAN合同没有统一的导航源码bundle摘要；"
            "不能把这组三seed冒充为当前精确代码的回归。"
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "input_valid": True,
        "seed_count": len(seeds),
        "seeds": seeds,
        "shared_contract": {
            "path_points_sha256": shared.path_points_sha256,
            "path_point_count": shared.path_point_count,
            "termination_contract": json.loads(
                shared.termination_contract_json
            ),
            "checkpoint_sha256": shared.checkpoint_sha256,
            "collision_ply_sha256": shared.collision_ply_sha256,
            "task_sha256": shared.task_sha256,
            "stair_contract_sha256": shared.stair_contract_sha256,
            "primary_metric_excludes_common_stair_freeze": True,
        },
        "scan": {
            "success_count": scan_success_count,
            "trial_count": len(scan_trials),
            "success_rate": scan_success_rate,
            "planner_controlled_completion_mean_s": _mean(
                [float(value) for value in scan_completion_times]
            ),
            "planner_controlled_completion_median_s": _median(
                [float(value) for value in scan_completion_times]
            ),
            "planner_controlled_completion_max_s": (
                max(scan_completion_times) if scan_completion_times else None
            ),
            "planner_compute_wall_total_s": sum(
                trial.planner_compute_wall_s for trial in scan_trials
            ),
            "terminal_capture_mean_s": _mean(
                [
                    float(trial.terminal_capture_s)
                    for trial in scan_trials
                    if trial.terminal_capture_s is not None
                ]
            ),
            "tuning_snapshot_sha256_values": tuning_shas,
            "tuning_snapshot_uniform": scan_tuning_uniform,
            "source_revision_sha256_values": source_shas,
            "source_revision_uniform": scan_source_revision_uniform,
        },
        "dwa": {
            "success_count": dwa_success_count,
            "trial_count": len(dwa_trials),
            "success_rate": dwa_success_rate,
            "planner_controlled_completion_mean_s": _mean(
                [float(value) for value in dwa_completion_times]
            ),
            "planner_controlled_time_to_failure_mean_s": _mean(
                [float(value) for value in dwa_time_to_failure]
            ),
            "planner_compute_wall_total_s": sum(
                trial.planner_compute_wall_s for trial in dwa_trials
            ),
            "failure_reasons": sorted(
                {trial.failure_reason for trial in dwa_trials if not trial.success}
            ),
            "deterministic_failure_signature": (
                len(failure_signatures) == 1
                and dwa_success_count == 0
                and len(dwa_trials) > 0
            ),
            "failure_signature": (
                json.loads(next(iter(failure_signatures)))
                if len(failure_signatures) == 1
                else None
            ),
        },
        "paired_trials": paired,
        "verdict": {
            "minimum_seed_count": minimum_seed_count,
            "minimum_seed_count_met": minimum_seed_count_met,
            "evaluation_order": "success_rate_then_successful_completion_time",
            "navigation_success_superiority_verified": (
                navigation_success_superiority
            ),
            "paired_completion_seed_count": len(paired_success_seeds),
            "paired_completion_time_comparable": enough_paired_completions,
            "completion_speed_superiority_verified": (
                completion_speed_superiority
            ),
            "overall_navigation_performance_exceeds_dwa": overall_exceeds,
            "verdict_basis": verdict_basis,
            "legacy_contract_time_only_rule_satisfied": (
                completion_speed_superiority
            ),
            "current_exact_revision_three_seed_verified": (
                exact_current_revision_three_seed_verified
            ),
        },
        "caveats": caveats,
    }


def write_summary(summary: Mapping[str, Any], output_value: str | Path) -> Path:
    """独占写入小型JSON证据，避免覆盖先前实验结论。"""

    output = Path(output_value).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ComparisonSummaryError(f"拒绝覆盖已有汇总：{output}") from exc
    except OSError as exc:
        raise ComparisonSummaryError(f"无法写入汇总 {output}：{exc}") from exc
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", nargs="+", required=True, type=Path)
    parser.add_argument("--dwa", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        scan_trials = [load_scan_trial(path) for path in arguments.scan]
        dwa_trials = [load_dwa_trial(path) for path in arguments.dwa]
        summary = summarize_trials(scan_trials, dwa_trials)
        output = None
        if arguments.output is not None:
            output = write_summary(summary, arguments.output)
    except ComparisonSummaryError as exc:
        print(f"FAIL：{exc}", file=sys.stderr)
        return 1
    payload = {"summary": summary, "output": str(output) if output else None}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
