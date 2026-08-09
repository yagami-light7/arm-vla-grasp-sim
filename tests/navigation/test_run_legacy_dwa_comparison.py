from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.navigation import run_legacy_dwa_comparison as comparison


def _contract() -> dict[str, object]:
    points = [
        [-3.48, 6.52, -0.174],
        [1.50, 5.70, -0.128],
        [2.70, 7.05, 2.529],
        [0.40, -0.02, 3.001914814802456],
    ]
    payload: dict[str, object] = {
        "schema": comparison.EXPECTED_SCHEMA,
        "eligible_as_crossplanner_navigation_source": True,
        "ineligibility_reasons": [],
        "source_run": {"seed": 2},
        "global_path": {
            "points_ground_xyz": points,
            "points_sha256": comparison._path_points_sha256(points),
            "point_count": len(points),
            "height_semantics": "ground",
        },
        "initial_condition": {
            "nav_stage_first_diagnostic_base_xyzyaw": [
                -3.48,
                6.52,
                0.164,
                1.665,
            ]
        },
        "goal": {
            "position_base_xyz": [0.40, -0.02, 3.339914814802456],
            "yaw_rad": -math.pi / 2.0,
        },
        "termination_contract": {
            "final_position_tolerance": 0.08,
            "place_position_tolerance": 0.08,
            "final_yaw_tolerance": 0.20,
            "stable_linear_velocity": 0.05,
            "stable_angular_velocity": 0.10,
            "require_yaw_alignment": True,
            "require_stable_base": True,
            "finish_distance_z": 0.12,
            "stable_dwell_s": 0.50,
            "post_goal_zero_write_ticks": 5,
        },
        "shared_runtime_contract": {
            "body_height_m": 0.338,
            "policy_task": comparison.EXPECTED_POLICY_TASK,
            "policy_checkpoint": {"sha256": "a" * 64},
            "collision_ply": {"sha256": "b" * 64},
            "task_json": {"sha256": "c" * 64},
            "stair_freeze": {"contract_sha256": "d" * 64},
        },
    }
    payload["contract_payload_sha256"] = comparison._canonical_payload_sha256(
        payload
    )
    return payload


def _write_contract(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state(
    *,
    timestamp: float,
    step_index: int,
    x: float = 0.40,
    y: float = -0.02,
    z: float = 3.339914814802456,
    yaw: float = -math.pi / 2.0,
    velocity: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=timestamp,
        step_index=step_index,
        robot_root_pose=(
            x,
            y,
            z,
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ),
        robot_root_velocity=velocity,
    )


@dataclass(frozen=True)
class _Action:
    base_velocity: tuple[float, float, float]
    source: str
    metadata: dict[str, object]


class _InnerExecutor:
    def __init__(self) -> None:
        self._done = True
        self._success = True
        self._failure_reason = ""
        self._phase = "completed"
        self.reset_count = 0

    def reset(self, plan: object) -> None:
        del plan
        self.reset_count += 1

    def is_done(self, state: object) -> bool:
        del state
        return self._done

    def compute_action(self, state: object) -> _Action:
        del state
        return _Action(
            base_velocity=(0.0, 0.0, 0.0),
            source="inner_zero",
            metadata={"carry_hold": True},
        )

    def status(self) -> dict[str, object]:
        return {
            "done": self._done,
            "success": self._success,
            "failed": bool(self._failure_reason),
            "failure_reason": self._failure_reason,
            "phase": self._phase,
        }


@dataclass(frozen=True)
class _NavigationConfig:
    final_position_tolerance: float = 0.18
    place_position_tolerance: float | None = None
    final_yaw_tolerance: float = math.pi
    stable_linear_velocity: float = 0.06
    stable_angular_velocity: float = 0.20
    require_yaw_alignment: bool = False
    require_stable_base: bool = False
    goal_z_tolerance: float = 0.35


@dataclass(frozen=True)
class _PipelineConfig:
    navigation: _NavigationConfig = _NavigationConfig()


class _ConfiguredExecutor:
    def __init__(self, config: _NavigationConfig) -> None:
        self.position_tolerance = config.final_position_tolerance
        self.carry_position_tolerance = config.place_position_tolerance
        self.yaw_tolerance = config.final_yaw_tolerance
        self.completion_linear_velocity_tolerance = (
            config.stable_linear_velocity if config.require_stable_base else None
        )
        self.completion_angular_velocity_tolerance = (
            config.stable_angular_velocity if config.require_stable_base else None
        )
        self.require_yaw_alignment = config.require_yaw_alignment
        self.terminal_pose_config = SimpleNamespace(
            position_acceptance_tolerance=config.final_position_tolerance,
            yaw_tolerance=config.final_yaw_tolerance,
        )


def _comparison_audit(
    payload: dict[str, object],
    *,
    complete: bool,
) -> dict[str, object]:
    configured = comparison.strict_termination_config(_PipelineConfig(), payload)
    executor_audit = comparison.verify_legacy_executor_termination(
        _ConfiguredExecutor(configured.navigation),
        payload,
    )
    return {
        "contract_payload_sha256": payload["contract_payload_sha256"],
        "global_path_points_sha256": payload["global_path"]["points_sha256"],
        "stair_freeze_contract_sha256": payload["shared_runtime_contract"][
            "stair_freeze"
        ]["contract_sha256"],
        "stable_dwell_verified": complete,
        "complete": complete,
        "post_goal_zero_write_ticks_required": 5,
        "post_goal_zero_write_count": 5 if complete else 0,
        "post_goal_nonzero_write_count": 0,
        "legacy_executor_config_audit": executor_audit,
    }


def test_load_comparison_contract_checks_both_payload_and_path_hash(
    tmp_path: Path,
) -> None:
    payload = _contract()
    path = tmp_path / "contract.json"
    _write_contract(path, payload)
    loaded = comparison.load_comparison_contract(path)
    assert loaded["contract_payload_sha256"] == payload["contract_payload_sha256"]

    payload["global_path"]["points_ground_xyz"][1][0] += 0.01  # type: ignore[index]
    payload["contract_payload_sha256"] = comparison._canonical_payload_sha256(
        {key: value for key, value in payload.items() if key != "contract_payload_sha256"}
    )
    _write_contract(path, payload)
    with pytest.raises(comparison.LegacyDwaComparisonError, match="Path点列SHA256"):
        comparison.load_comparison_contract(path)


def test_contract_path_planner_injects_exact_points_and_hash() -> None:
    payload = _contract()
    planner = comparison.ContractPathPlanner(
        payload,
        nav_plan_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    initial = payload["initial_condition"][
        "nav_stage_first_diagnostic_base_xyzyaw"
    ]
    state = _state(
        timestamp=1.58,
        step_index=10,
        x=initial[0],
        y=initial[1],
        z=initial[2],
        yaw=initial[3],
    )
    goal_xyz = payload["goal"]["position_base_xyz"]
    goal = SimpleNamespace(
        x=goal_xyz[0],
        y=goal_xyz[1],
        z=goal_xyz[2],
        yaw=payload["goal"]["yaw_rad"],
    )
    plan = planner.plan(state, goal)
    assert plan.metadata["path_3d"] == planner.points
    assert plan.metadata["comparison_global_path_points_sha256"] == payload[
        "global_path"
    ]["points_sha256"]
    assert plan.waypoints[-1] == pytest.approx((0.40, -0.02))


def test_strict_config_is_applied_before_legacy_executor_is_built() -> None:
    payload = _contract()
    original = _PipelineConfig()
    configured = comparison.strict_termination_config(original, payload)

    assert original.navigation.final_position_tolerance == pytest.approx(0.18)
    assert configured.navigation.final_position_tolerance == pytest.approx(0.08)
    assert configured.navigation.place_position_tolerance == pytest.approx(0.08)
    assert configured.navigation.final_yaw_tolerance == pytest.approx(0.20)
    assert configured.navigation.stable_linear_velocity == pytest.approx(0.05)
    assert configured.navigation.stable_angular_velocity == pytest.approx(0.10)
    assert configured.navigation.require_yaw_alignment is True
    assert configured.navigation.require_stable_base is True
    assert configured.navigation.goal_z_tolerance == pytest.approx(0.12)

    audit = comparison.verify_legacy_executor_termination(
        _ConfiguredExecutor(configured.navigation),
        payload,
    )
    assert audit["verified"] is True
    assert audit["position_tolerance"] == pytest.approx(0.08)
    assert audit["carry_position_tolerance"] == pytest.approx(0.08)
    assert audit["require_yaw_alignment"] is True
    assert audit["require_stable_base"] is True


def test_legacy_executor_config_audit_rejects_old_loose_defaults() -> None:
    with pytest.raises(
        comparison.LegacyDwaComparisonError,
        match="终点配置不一致|严格字段",
    ):
        comparison.verify_legacy_executor_termination(
            _ConfiguredExecutor(_NavigationConfig()),
            _contract(),
        )


def test_strict_executor_requires_dwell_then_five_applied_zero_ticks() -> None:
    payload = _contract()
    inner = _InnerExecutor()
    executor = comparison.StrictTerminationExecutor(inner, payload)
    executor.reset(SimpleNamespace(goal=SimpleNamespace()))

    assert executor.is_done(_state(timestamp=10.0, step_index=100)) is False
    assert executor.is_done(_state(timestamp=10.5, step_index=125)) is False
    for index in range(5):
        state = _state(timestamp=10.52 + 0.02 * index, step_index=126 + index)
        assert executor.is_done(state) is False
        action = executor.compute_action(state)
        assert action.base_velocity == (0.0, 0.0, 0.0)
        assert action.metadata["carry_hold"] is True
        done_after_action = executor.is_done(state)
        assert done_after_action is (index == 4)

    status = executor.status()
    comparison_status = status["comparison_contract"]
    assert status["done"] is True
    assert status["success"] is True
    assert comparison_status["stable_dwell_verified"] is True
    assert comparison_status["post_goal_zero_write_count"] == 5
    assert comparison_status["post_goal_nonzero_write_count"] == 0


def test_strict_executor_reactivates_terminal_control_after_drift() -> None:
    payload = _contract()
    inner = _InnerExecutor()
    executor = comparison.StrictTerminationExecutor(inner, payload)
    executor.reset(SimpleNamespace(goal=SimpleNamespace()))
    assert executor.is_done(_state(timestamp=1.0, step_index=1)) is False

    drifted = _state(timestamp=1.2, step_index=2, x=0.60)
    assert executor.is_done(drifted) is False
    assert inner._done is False
    assert inner._success is False
    assert inner._phase == "terminal_pose"
    assert executor.status()["comparison_contract"]["terminal_recovery_count"] == 1


def test_legacy_argv_locks_common_stair_and_termination_parameters(
    tmp_path: Path,
) -> None:
    args = comparison.build_legacy_argv(
        contract=_contract(),
        task_json=tmp_path / "task.json",
        output_dir=tmp_path / "run",
    )
    assert args[args.index("--seed") + 1] == "2"
    assert args[args.index("--pct-stair-float-speed") + 1] == "0.18"
    assert args[args.index("--pct-stair-float-activation-radius") + 1] == "0.35"
    assert args[args.index("--pct-stair-float-approach-distance") + 1] == "1.5"
    assert args[args.index("--pct-stair-float-exit-distance") + 1] == "0.4"
    assert args[args.index("--goal-z-tolerance") + 1] == "0.12"
    assert "--navigation-carry-smoke" in args
    assert "--no-record-video" in args


def test_parent_summary_validation_requires_strict_executor_audit(
    tmp_path: Path,
) -> None:
    payload = _contract()
    audit = _comparison_audit(payload, complete=True)
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    (episode_dir / "summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "final_state": "done",
                "failure_reason": None,
                "duration_steps": 123,
                "latest_executor_status": {"comparison_contract": audit},
            }
        ),
        encoding="utf-8",
    )

    result = comparison._read_child_summary(tmp_path, payload)
    assert result["valid"] is True
    audit["legacy_executor_config_audit"] = {}
    (episode_dir / "summary.json").write_text(
        json.dumps(
            {
                "success": True,
                "final_state": "done",
                "failure_reason": None,
                "duration_steps": 123,
                "latest_executor_status": {"comparison_contract": audit},
            }
        ),
        encoding="utf-8",
    )
    result = comparison._read_child_summary(tmp_path, payload)
    assert result["valid"] is False
    assert "legacy_executor_config_not_verified" in result["errors"]


def test_navigation_failure_is_a_valid_comparison_trial(tmp_path: Path) -> None:
    payload = _contract()
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    (episode_dir / "summary.json").write_text(
        json.dumps(
            {
                "success": False,
                "final_state": "failed",
                "failure_reason": "nav_collision",
                "duration_steps": 5000,
                "latest_executor_status": {
                    "failed": True,
                    "failure_reason": "nav_collision",
                    "comparison_contract": _comparison_audit(
                        payload,
                        complete=False,
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    result = comparison._read_child_summary(tmp_path, payload)
    assert result["valid"] is False
    assert result["comparison_trial_valid"] is True
    assert result["navigation_success"] is False
    assert result["navigation_failure_is_valid_outcome"] is True
    assert result["errors"] == []


def test_analyzer_excludes_stair_interval_and_keeps_failure_diagnostics(
    tmp_path: Path,
) -> None:
    payload = _contract()
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    audit = _comparison_audit(payload, complete=False)
    summary = {
        "success": False,
        "final_state": "failed",
        "failure_reason": "nav_collision",
        "duration_steps": 5,
        "latest_executor_status": {
            "phase": "stalled",
            "failed": True,
            "failure_reason": "nav_collision",
            "comparison_contract": audit,
            "dwa_compute": {
                "recomputed_this_tick": True,
                "compute_count": 3,
                "last_duration_s": 0.003,
            },
            "dwa": {
                "sampled_candidates": 6,
                "feasible_candidates": 0,
                "collision_rejections": 6,
                "clearance": 0.0,
                "occupied_start_escape_active": True,
                "occupied_start_escape_candidates": 0,
            },
            "map_selection": {
                "post_stair_floor_replan": {
                    "path_optimization": {
                        "release_bridge": {
                            "applied": True,
                            "mode": "collision_checked_direct",
                            "bridge_is_clear": True,
                        }
                    }
                }
            },
        },
    }
    (episode_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    def frame(
        step: int,
        timestamp: float,
        source: str,
        pose: list[float],
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "step_index": step,
            "timestamp": timestamp,
            "pipeline_state": "exec_nav_to_place",
            "action": {"source": source, "metadata": metadata or {}},
            "post_step_observation": {"robot_root_pose": pose},
        }

    first_metadata = {
        "stair_float": {"end": [2.70, 5.76, 3.00]},
        "dwa_compute": {
            "recomputed_this_tick": True,
            "compute_count": 1,
            "last_duration_s": 0.001,
        },
    }
    second_metadata = {
        "dwa_compute": {
            "recomputed_this_tick": True,
            "compute_count": 2,
            "last_duration_s": 0.002,
        }
    }
    frames = [
        frame(0, 0.0, "navigation_dwa", [-3.48, 6.52, 0.17], first_metadata),
        frame(1, 1.0, "navigation_stair_float", [0.50, 4.82, 0.21]),
        frame(
            2,
            3.0,
            "navigation_stair_float_release_settle_completed",
            [2.70, 5.76, 3.34],
        ),
        frame(3, 3.1, "navigation_dwa", [2.70, 5.76, 3.338], second_metadata),
        frame(4, 4.0, "exec_nav_to_place", [2.40, 5.20, 3.40]),
    ]
    (episode_dir / "frames.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in frames),
        encoding="utf-8",
    )

    result = comparison.analyze_legacy_dwa_result(tmp_path, payload)
    assert result["comparison_trial_valid"] is True
    assert result["navigation_success"] is False
    assert result["timing"]["f1_planner_controlled_sim_time_s"] == pytest.approx(
        1.0
    )
    assert result["timing"]["common_stair_freeze_sim_time_s"] == pytest.approx(
        2.1
    )
    assert result["timing"]["f2_to_outcome_sim_time_s"] == pytest.approx(0.9)
    assert result["timing"][
        "planner_controlled_to_outcome_sim_time_s"
    ] == pytest.approx(1.9)
    assert result["dwa_compute"]["unique_recompute_count"] == 3
    assert result["dwa_compute"][
        "successful_compute_wall_time_total_s"
    ] == pytest.approx(0.006)
    assert result["outcome_diagnostics"]["collision_rejections"] == 6
    assert result["handoff"]["contract_audit"]["contract_compatible"] is True


def test_analyzer_rejects_post_stair_handoff_inside_inflated_obstacle(
    tmp_path: Path,
) -> None:
    payload = _contract()
    episode_dir = tmp_path / "episode_000000"
    episode_dir.mkdir()
    audit = _comparison_audit(payload, complete=False)
    summary = {
        "success": False,
        "final_state": "failed",
        "failure_reason": "nav_collision",
        "duration_steps": 5,
        "latest_executor_status": {
            "failed": True,
            "failure_reason": "nav_collision",
            "comparison_contract": audit,
            "map_selection": {
                "post_stair_floor_replan": {
                    "path_optimization": {
                        "release_bridge": {
                            "applied": True,
                            "mode": "occupied_start_escape",
                            "bridge_is_clear": False,
                        }
                    }
                }
            },
        },
    }
    (episode_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    def frame(
        step: int,
        timestamp: float,
        source: str,
        pose: list[float],
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "step_index": step,
            "timestamp": timestamp,
            "pipeline_state": "exec_nav_to_place",
            "action": {"source": source, "metadata": metadata or {}},
            "post_step_observation": {"robot_root_pose": pose},
        }

    frames = [
        frame(
            0,
            0.0,
            "navigation_dwa",
            [-3.48, 6.52, 0.17],
            {"stair_float": {"end": [2.70, 5.76, 3.00]}},
        ),
        frame(1, 1.0, "navigation_stair_float", [0.50, 4.82, 0.21]),
        frame(2, 3.0, "navigation_stair_float", [2.70, 5.76, 3.338]),
        frame(3, 3.1, "navigation_dwa", [2.70, 5.76, 3.338]),
        frame(4, 4.0, "exec_nav_to_place", [2.40, 5.20, 3.40]),
    ]
    (episode_dir / "frames.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in frames),
        encoding="utf-8",
    )

    result = comparison.analyze_legacy_dwa_result(tmp_path, payload)
    assert result["comparison_trial_valid"] is False
    assert result["handoff"]["contract_audit"]["contract_compatible"] is False
    assert "post_stair_release_not_in_common_free_space" in result[
        "comparison_invalidation_reasons"
    ]
