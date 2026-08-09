#!/usr/bin/env python3
"""在隔离旧 worktree 中回放不可变 PCT 路径并运行 DWA 公平基线。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCHEMA = "pct_local_planner_comparison_contract_v1"
EXPECTED_LEGACY_BRANCH = "pct_scene"
EXPECTED_POLICY_TASK = "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0"
LEGACY_CHECKPOINT_RELATIVE = Path(
    "checkpoints/go2_x5/pct_multifloor/model_26000.pt"
)
LEGACY_COLLISION_PLY_RELATIVE = Path(
    "source/scene/multifloor/ply/3dgs_collision.ply"
)


class LegacyDwaComparisonError(ValueError):
    """表示旧 DWA 运行不满足不可变公平对照合同。"""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise LegacyDwaComparisonError(f"无法读取文件 {path}：{exc}") from exc


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LegacyDwaComparisonError(f"{label}必须是有限数。")
    result = float(value)
    if not math.isfinite(result):
        raise LegacyDwaComparisonError(f"{label}必须是有限数。")
    return result


def _finite_vector(
    value: object,
    size: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise LegacyDwaComparisonError(f"{label}必须包含{size}个有限数。")
    return tuple(
        _finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyDwaComparisonError(f"{label}必须是JSON对象。")
    return value


def _path_points_sha256(points: Sequence[Sequence[float]]) -> str:
    """复用 ROS Path identity 的网络字节序双精度哈希。"""

    digest = hashlib.sha256()
    for index, point in enumerate(points):
        x, y, z = _finite_vector(point, 3, f"global_path[{index}]")
        digest.update(struct.pack("!ddd", x, y, z))
    return digest.hexdigest()


def load_comparison_contract(path_value: str | Path) -> dict[str, Any]:
    """读取并完整验证 SCAN 导出的不可变比较合同。"""

    path = Path(path_value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyDwaComparisonError(f"无法读取比较合同 {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise LegacyDwaComparisonError("比较合同顶层必须是JSON对象。")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise LegacyDwaComparisonError(
            f"比较合同schema不匹配：{payload.get('schema')!r}"
        )
    if payload.get("eligible_as_crossplanner_navigation_source") is not True:
        raise LegacyDwaComparisonError(
            "比较合同来源没有通过cross-planner资格门。"
        )
    declared_payload_sha = payload.get("contract_payload_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("contract_payload_sha256", None)
    actual_payload_sha = _canonical_payload_sha256(unsigned_payload)
    if declared_payload_sha != actual_payload_sha:
        raise LegacyDwaComparisonError(
            "比较合同payload SHA256不匹配："
            f"declared={declared_payload_sha!r}, actual={actual_payload_sha}"
        )

    global_path = _mapping(payload.get("global_path"), "global_path")
    points_value = global_path.get("points_ground_xyz")
    if not isinstance(points_value, list) or len(points_value) < 2:
        raise LegacyDwaComparisonError("global_path必须至少包含两个完整三维点。")
    points = [
        list(_finite_vector(point, 3, f"global_path[{index}]"))
        for index, point in enumerate(points_value)
    ]
    actual_points_sha = _path_points_sha256(points)
    if global_path.get("points_sha256") != actual_points_sha:
        raise LegacyDwaComparisonError(
            "PCT Path点列SHA256不匹配："
            f"declared={global_path.get('points_sha256')!r}, "
            f"actual={actual_points_sha}"
        )
    if int(global_path.get("point_count", -1)) != len(points):
        raise LegacyDwaComparisonError("PCT Path point_count与实际点数不一致。")
    if global_path.get("height_semantics") != "ground":
        raise LegacyDwaComparisonError("PCT Path必须使用ground高度语义。")

    termination = _mapping(
        payload.get("termination_contract"),
        "termination_contract",
    )
    expected_termination: dict[str, object] = {
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
    }
    for key, expected in expected_termination.items():
        actual = termination.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, int):
            matches = (
                not isinstance(actual, bool)
                and isinstance(actual, (int, float))
                and float(actual).is_integer()
                and int(actual) == expected
            )
        else:
            matches = math.isclose(
                _finite_float(actual, f"termination_contract.{key}"),
                float(expected),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        if not matches:
            raise LegacyDwaComparisonError(
                f"termination_contract.{key}不符合公平门："
                f"expected={expected!r}, actual={actual!r}"
            )
    return payload


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_wxyz(values: Sequence[float]) -> float:
    w, x, y, z = _finite_vector(values, 4, "robot_root_quaternion_wxyz")
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def strict_termination_config(
    config: Any,
    contract: Mapping[str, Any],
) -> Any:
    """把公平合同的终点门写入旧 navigation-smoke 配置副本。"""

    termination = _mapping(
        contract.get("termination_contract"),
        "termination_contract",
    )
    navigation = getattr(config, "navigation", None)
    if navigation is None:
        raise LegacyDwaComparisonError("旧 pipeline config 缺少 navigation。")
    return replace(
        config,
        navigation=replace(
            navigation,
            final_position_tolerance=_finite_float(
                termination.get("final_position_tolerance"),
                "termination.final_position_tolerance",
            ),
            place_position_tolerance=_finite_float(
                termination.get("place_position_tolerance"),
                "termination.place_position_tolerance",
            ),
            final_yaw_tolerance=_finite_float(
                termination.get("final_yaw_tolerance"),
                "termination.final_yaw_tolerance",
            ),
            stable_linear_velocity=_finite_float(
                termination.get("stable_linear_velocity"),
                "termination.stable_linear_velocity",
            ),
            stable_angular_velocity=_finite_float(
                termination.get("stable_angular_velocity"),
                "termination.stable_angular_velocity",
            ),
            require_yaw_alignment=(
                termination.get("require_yaw_alignment") is True
            ),
            require_stable_base=(termination.get("require_stable_base") is True),
            goal_z_tolerance=_finite_float(
                termination.get("finish_distance_z"),
                "termination.finish_distance_z",
            ),
        ),
    )


def verify_legacy_executor_termination(
    executor: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """确认旧 DWA 本身也使用严格终点门，避免只依赖外层补丁。"""

    termination = _mapping(
        contract.get("termination_contract"),
        "termination_contract",
    )
    expected = {
        "position_tolerance": termination["final_position_tolerance"],
        "carry_position_tolerance": termination["place_position_tolerance"],
        "yaw_tolerance": termination["final_yaw_tolerance"],
        "completion_linear_velocity_tolerance": termination[
            "stable_linear_velocity"
        ],
        "completion_angular_velocity_tolerance": termination[
            "stable_angular_velocity"
        ],
    }
    actual: dict[str, Any] = {}
    for name, raw_expected in expected.items():
        expected_value = _finite_float(raw_expected, f"termination.{name}")
        raw_actual = getattr(executor, name, None)
        if raw_actual is None:
            raise LegacyDwaComparisonError(
                f"旧 DWA executor 没有应用严格字段{name}。"
            )
        actual_value = _finite_float(raw_actual, f"legacy_executor.{name}")
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise LegacyDwaComparisonError(
                f"旧 DWA executor终点配置不一致：{name} "
                f"expected={expected_value}, actual={actual_value}"
            )
        actual[name] = actual_value
    require_yaw = getattr(executor, "require_yaw_alignment", None)
    if require_yaw is not (termination.get("require_yaw_alignment") is True):
        raise LegacyDwaComparisonError("旧 DWA executor没有应用严格yaw门。")
    terminal_pose_config = getattr(executor, "terminal_pose_config", None)
    if terminal_pose_config is None:
        raise LegacyDwaComparisonError("旧 DWA executor缺少terminal_pose_config。")
    terminal_acceptance = _finite_float(
        getattr(terminal_pose_config, "position_acceptance_tolerance", None),
        "legacy_executor.terminal_pose.position_acceptance_tolerance",
    )
    terminal_yaw = _finite_float(
        getattr(terminal_pose_config, "yaw_tolerance", None),
        "legacy_executor.terminal_pose.yaw_tolerance",
    )
    if not math.isclose(
        terminal_acceptance,
        float(termination["final_position_tolerance"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        terminal_yaw,
        float(termination["final_yaw_tolerance"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise LegacyDwaComparisonError("旧 DWA terminal pose配置与合同不一致。")
    return {
        "verified": True,
        **actual,
        "require_yaw_alignment": bool(require_yaw),
        "require_stable_base": (
            actual["completion_linear_velocity_tolerance"] is not None
            and actual["completion_angular_velocity_tolerance"] is not None
        ),
        "terminal_position_acceptance_tolerance": terminal_acceptance,
        "terminal_yaw_tolerance": terminal_yaw,
        "goal_z_tolerance": _finite_float(
            termination.get("finish_distance_z"),
            "termination.finish_distance_z",
        ),
    }


class ContractPathPlanner:
    """向旧 DWA 注入合同中逐点完全相同的 PCT 全局路径。"""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        nav_plan_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.contract = contract
        self._nav_plan_factory = nav_plan_factory
        global_path = _mapping(contract.get("global_path"), "global_path")
        self.points = tuple(
            _finite_vector(point, 3, f"global_path[{index}]")
            for index, point in enumerate(global_path["points_ground_xyz"])
        )
        self.points_sha256 = str(global_path["points_sha256"])
        self.contract_sha256 = str(contract["contract_payload_sha256"])
        self.goal_contract = _mapping(contract.get("goal"), "goal")
        self.initial_contract = _mapping(
            contract.get("initial_condition"),
            "initial_condition",
        )

    def plan(self, state: Any, goal: Any) -> Any:
        expected_goal = _finite_vector(
            self.goal_contract.get("position_base_xyz"),
            3,
            "goal.position_base_xyz",
        )
        expected_yaw = _finite_float(
            self.goal_contract.get("yaw_rad"),
            "goal.yaw_rad",
        )
        actual_goal = (
            float(goal.x),
            float(goal.y),
            float(expected_goal[2] if goal.z is None else goal.z),
        )
        if math.dist(actual_goal, expected_goal) > 1.0e-6:
            raise LegacyDwaComparisonError(
                "旧 DWA 任务目标与合同不一致："
                f"expected={expected_goal}, actual={actual_goal}"
            )
        if abs(_wrap_angle(float(goal.yaw) - expected_yaw)) > 1.0e-6:
            raise LegacyDwaComparisonError("旧 DWA 任务目标yaw与合同不一致。")

        initial = _finite_vector(
            self.initial_contract.get("nav_stage_first_diagnostic_base_xyzyaw"),
            4,
            "initial_condition.nav_stage_first_diagnostic_base_xyzyaw",
        )
        root_pose = _finite_vector(state.robot_root_pose, 7, "robot_root_pose")
        actual_yaw = _yaw_from_wxyz(root_pose[3:7])
        xy_error = math.dist(root_pose[:2], initial[:2])
        z_error = abs(root_pose[2] - initial[2])
        yaw_error = abs(_wrap_angle(actual_yaw - initial[3]))
        if xy_error > 0.08 or z_error > 0.06 or yaw_error > 0.15:
            raise LegacyDwaComparisonError(
                "旧 DWA 初始状态没有复现合同："
                f"xy_error={xy_error:.6f}, z_error={z_error:.6f}, "
                f"yaw_error={yaw_error:.6f}"
            )

        factory = self._nav_plan_factory
        if factory is None:
            from source.interfaces.navigation import NavPlan

            factory = NavPlan
        metadata = {
            "planner": "pct",
            "path_3d": self.points,
            "pct_raw_path_3d": self.points,
            "sim_start": tuple(float(value) for value in root_pose[:3]),
            "cross_floor": True,
            "comparison_contract_sha256": self.contract_sha256,
            "comparison_global_path_points_sha256": self.points_sha256,
            "comparison_global_path_point_count": len(self.points),
            "comparison_input_is_immutable": True,
        }
        return factory(
            goal=goal,
            waypoints=tuple((point[0], point[1]) for point in self.points),
            metadata=metadata,
        )

    def close(self) -> None:
        """固定路径 planner 不持有子进程。"""


class StrictTerminationExecutor:
    """为旧 DWA 补齐连续停稳与目标后零速写入合同。"""

    def __init__(
        self,
        inner: Any,
        contract: Mapping[str, Any],
        *,
        legacy_executor_config_audit: Mapping[str, Any] | None = None,
    ) -> None:
        self.inner = inner
        self.contract = contract
        self.termination = _mapping(
            contract.get("termination_contract"),
            "termination_contract",
        )
        self.goal_contract = _mapping(contract.get("goal"), "goal")
        self.path_contract = _mapping(contract.get("global_path"), "global_path")
        self.stair_contract = _mapping(
            _mapping(
                contract.get("shared_runtime_contract"),
                "shared_runtime_contract",
            ).get("stair_freeze"),
            "shared_runtime_contract.stair_freeze",
        )
        self.legacy_executor_config_audit = dict(
            legacy_executor_config_audit or {}
        )
        self._goal: Any | None = None
        self._stable_started_at: float | None = None
        self._stable_elapsed_s = 0.0
        self._dwell_verified = False
        self._zero_write_count = 0
        self._zero_write_steps: set[int] = set()
        self._complete = False
        self._last_gate: dict[str, Any] = {}
        self._recovery_count = 0

    def reset(self, plan: Any) -> None:
        self.inner.reset(plan)
        self._goal = plan.goal
        self._stable_started_at = None
        self._stable_elapsed_s = 0.0
        self._dwell_verified = False
        self._zero_write_count = 0
        self._zero_write_steps.clear()
        self._complete = False
        self._last_gate = {}
        self._recovery_count = 0

    def _strict_gate(self, state: Any) -> dict[str, Any]:
        if self._goal is None:
            return {"satisfied": False, "reason": "goal_unavailable"}
        root_pose = _finite_vector(state.robot_root_pose, 7, "robot_root_pose")
        root_velocity = _finite_vector(
            state.robot_root_velocity,
            6,
            "robot_root_velocity",
        )
        expected_goal = _finite_vector(
            self.goal_contract.get("position_base_xyz"),
            3,
            "goal.position_base_xyz",
        )
        expected_yaw = _finite_float(
            self.goal_contract.get("yaw_rad"),
            "goal.yaw_rad",
        )
        xy_error = math.dist(root_pose[:2], expected_goal[:2])
        z_error = abs(root_pose[2] - expected_goal[2])
        yaw_error = abs(_wrap_angle(_yaw_from_wxyz(root_pose[3:7]) - expected_yaw))
        linear_speed = math.sqrt(sum(value * value for value in root_velocity[:3]))
        angular_speed = math.sqrt(sum(value * value for value in root_velocity[3:6]))
        xy_tolerance = _finite_float(
            self.termination.get("final_position_tolerance"),
            "termination.final_position_tolerance",
        )
        z_tolerance = _finite_float(
            self.termination.get("finish_distance_z"),
            "termination.finish_distance_z",
        )
        yaw_tolerance = _finite_float(
            self.termination.get("final_yaw_tolerance"),
            "termination.final_yaw_tolerance",
        )
        linear_tolerance = _finite_float(
            self.termination.get("stable_linear_velocity"),
            "termination.stable_linear_velocity",
        )
        angular_tolerance = _finite_float(
            self.termination.get("stable_angular_velocity"),
            "termination.stable_angular_velocity",
        )
        checks = {
            "xy": xy_error <= xy_tolerance,
            "z": z_error <= z_tolerance,
            "yaw": yaw_error <= yaw_tolerance,
            "linear_speed": linear_speed <= linear_tolerance,
            "angular_speed": angular_speed <= angular_tolerance,
        }
        return {
            "satisfied": all(checks.values()),
            "checks": checks,
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "yaw_error_rad": yaw_error,
            "linear_speed_mps": linear_speed,
            "angular_speed_rps": angular_speed,
            "xy_tolerance_m": xy_tolerance,
            "z_tolerance_m": z_tolerance,
            "yaw_tolerance_rad": yaw_tolerance,
            "linear_speed_tolerance_mps": linear_tolerance,
            "angular_speed_tolerance_rps": angular_tolerance,
        }

    def _resume_inner_after_drift(self) -> None:
        """旧执行器完成后若漂出严格门，恢复其末端控制。"""

        changed = False
        for name, value in (
            ("_done", False),
            ("_success", False),
            ("_failure_reason", ""),
            ("_phase", "terminal_pose"),
        ):
            if hasattr(self.inner, name):
                setattr(self.inner, name, value)
                changed = True
        if changed:
            self._recovery_count += 1

    def is_done(self, state: Any) -> bool:
        if self._complete:
            return True
        inner_done = bool(self.inner.is_done(state))
        inner_status = dict(self.inner.status())
        if inner_status.get("failed"):
            return inner_done
        if not inner_done or inner_status.get("success") is not True:
            self._stable_started_at = None
            self._stable_elapsed_s = 0.0
            self._dwell_verified = False
            self._zero_write_count = 0
            self._zero_write_steps.clear()
            return False

        gate = self._strict_gate(state)
        self._last_gate = gate
        if gate.get("satisfied") is not True:
            self._stable_started_at = None
            self._stable_elapsed_s = 0.0
            self._dwell_verified = False
            self._zero_write_count = 0
            self._zero_write_steps.clear()
            self._resume_inner_after_drift()
            return False

        timestamp = _finite_float(state.timestamp, "state.timestamp")
        if self._stable_started_at is None:
            self._stable_started_at = timestamp
        self._stable_elapsed_s = max(0.0, timestamp - self._stable_started_at)
        required_dwell = _finite_float(
            self.termination.get("stable_dwell_s"),
            "termination.stable_dwell_s",
        )
        self._dwell_verified = self._stable_elapsed_s + 1.0e-12 >= required_dwell
        required_zero_writes = int(self.termination["post_goal_zero_write_ticks"])
        self._complete = (
            self._dwell_verified
            and self._zero_write_count >= required_zero_writes
        )
        return self._complete

    def compute_action(self, state: Any) -> Any:
        action = self.inner.compute_action(state)
        if not self._dwell_verified or self._last_gate.get("satisfied") is not True:
            return action
        step_index = int(state.step_index)
        if step_index not in self._zero_write_steps:
            self._zero_write_steps.add(step_index)
            self._zero_write_count += 1
        metadata = dict(getattr(action, "metadata", {}) or {})
        metadata["legacy_dwa_comparison_terminal_hold"] = True
        metadata["comparison_contract_sha256"] = self.contract[
            "contract_payload_sha256"
        ]
        metadata["comparison_post_goal_zero_write_count"] = self._zero_write_count
        return replace(
            action,
            base_velocity=(0.0, 0.0, 0.0),
            source="legacy_dwa_comparison_terminal_hold",
            metadata=metadata,
        )

    def status(self) -> dict[str, Any]:
        status = dict(self.inner.status())
        inner_failed = bool(status.get("failed"))
        if not inner_failed:
            status["done"] = self._complete
            status["success"] = self._complete
            if not self._complete and (
                self._stable_started_at is not None or self._dwell_verified
            ):
                status["phase"] = "comparison_terminal_dwell"
        status["comparison_contract"] = {
            "schema": EXPECTED_SCHEMA,
            "contract_payload_sha256": self.contract[
                "contract_payload_sha256"
            ],
            "global_path_points_sha256": self.path_contract["points_sha256"],
            "global_path_point_count": self.path_contract["point_count"],
            "stair_freeze_contract_sha256": self.stair_contract[
                "contract_sha256"
            ],
            "strict_gate": dict(self._last_gate),
            "stable_elapsed_s": self._stable_elapsed_s,
            "stable_dwell_required_s": self.termination["stable_dwell_s"],
            "stable_dwell_verified": self._dwell_verified,
            "post_goal_zero_write_count": self._zero_write_count,
            "post_goal_zero_write_ticks_required": self.termination[
                "post_goal_zero_write_ticks"
            ],
            "post_goal_nonzero_write_count": 0,
            "terminal_recovery_count": self._recovery_count,
            "legacy_executor_config_audit": dict(
                self.legacy_executor_config_audit
            ),
            "complete": self._complete,
        }
        return status


def patch_navigation_smoke_module(
    module: ModuleType,
    contract: Mapping[str, Any],
) -> None:
    """在 Isaac App 已启动后替换旧 carry smoke 的 planner/executor。"""

    original = module.create_navigation_carry_smoke_pipeline
    if getattr(original, "_legacy_dwa_comparison_patched", False):
        return

    def patched_factory(*args: Any, **kwargs: Any) -> Any:
        if args or "episode_spec" not in kwargs:
            raise LegacyDwaComparisonError(
                "旧 carry smoke factory 必须以关键字传入 episode_spec。"
            )
        episode_spec = kwargs["episode_spec"]
        if episode_spec.place_goal is None:
            raise LegacyDwaComparisonError("旧 carry smoke 缺少 place_goal。")
        effective_goal_xyz = _finite_vector(
            _mapping(contract.get("goal"), "goal").get("position_base_xyz"),
            3,
            "goal.position_base_xyz",
        )
        effective_goal_yaw = _finite_float(
            _mapping(contract.get("goal"), "goal").get("yaw_rad"),
            "goal.yaw_rad",
        )
        effective_place_goal = replace(
            episode_spec.place_goal,
            x=effective_goal_xyz[0],
            y=effective_goal_xyz[1],
            z=effective_goal_xyz[2],
            yaw=effective_goal_yaw,
        )
        # 当前 PCT adapter 会先把任务声明高度标定为真实楼面 base 高度；旧
        # pipeline 没有这一步，因此只在隔离回放对象中应用同一个有效目标。
        raw_task = json.loads(json.dumps(episode_spec.raw_task))
        raw_task.setdefault("runtime_override", {}).update(
            {
                "planner_comparison": "legacy_dwa",
                "effective_place_goal_base_xyz": list(effective_goal_xyz),
                "effective_place_goal_yaw_rad": effective_goal_yaw,
                "comparison_contract_sha256": contract[
                    "contract_payload_sha256"
                ],
            }
        )
        effective_start = _finite_vector(
            _mapping(
                contract.get("initial_condition"),
                "initial_condition",
            ).get("nav_stage_first_diagnostic_base_xyzyaw"),
            4,
            "initial_condition.nav_stage_first_diagnostic_base_xyzyaw",
        )
        raw_carry = raw_task.get("carry")
        if not isinstance(raw_carry, dict) or not isinstance(
            raw_carry.get("smoke_start"), dict
        ):
            raise LegacyDwaComparisonError(
                "旧任务缺少 carry.smoke_start，无法绑定同一初始状态。"
            )
        raw_carry["smoke_start"].update(
            {
                "x": effective_start[0],
                "y": effective_start[1],
                "z": effective_start[2],
                "yaw": effective_start[3],
            }
        )
        raw_place = raw_task.get("place")
        if isinstance(raw_place, dict) and isinstance(
            raw_place.get("base_goal"), dict
        ):
            raw_place["base_goal"].update(
                {
                    "x": effective_goal_xyz[0],
                    "y": effective_goal_xyz[1],
                    "z": effective_goal_xyz[2],
                    "yaw": effective_goal_yaw,
                }
            )
        effective_spec = replace(
            episode_spec,
            place_goal=effective_place_goal,
            raw_task=raw_task,
        )
        kwargs["episode_spec"] = effective_spec
        kwargs["config"] = strict_termination_config(kwargs["config"], contract)
        pipeline = original(*args, **kwargs)
        old_planner = pipeline.nav_planner
        close = getattr(old_planner, "close", None)
        if callable(close):
            close()
        fixed_planner = ContractPathPlanner(contract)
        pipeline.nav_planner = fixed_planner
        pipeline.machine.nav_planner = fixed_planner
        legacy_executor = pipeline.machine.nav_executor
        legacy_executor_config_audit = verify_legacy_executor_termination(
            legacy_executor,
            contract,
        )
        pipeline.machine.nav_executor = StrictTerminationExecutor(
            legacy_executor,
            contract,
            legacy_executor_config_audit=legacy_executor_config_audit,
        )
        return pipeline

    patched_factory._legacy_dwa_comparison_patched = True  # type: ignore[attr-defined]
    module.create_navigation_carry_smoke_pipeline = patched_factory


class _PatchingLoader(importlib.abc.Loader):
    """先执行旧模块，再在其 factory 上安装公平合同。"""

    def __init__(
        self,
        delegate: importlib.abc.Loader,
        patch: Callable[[ModuleType], None],
    ) -> None:
        self.delegate = delegate
        self.patch = patch

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self.delegate, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self.delegate, "exec_module", None)
        if execute is None:
            raise ImportError("旧 navigation_smoke loader 不支持 exec_module。")
        execute(module)
        self.patch(module)


class NavigationSmokePatchFinder(importlib.abc.MetaPathFinder):
    """延迟到 Isaac 初始化后的真实 navigation_smoke import 再打补丁。"""

    TARGET = "source.pipeline.navigation_smoke"

    def __init__(self, patch: Callable[[ModuleType], None]) -> None:
        self.patch = patch
        self._resolving = False

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> Any:
        del target
        if fullname != self.TARGET or self._resolving:
            return None
        self._resolving = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self._resolving = False
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchingLoader(spec.loader, self.patch)
        return spec


def _legacy_branch(legacy_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(legacy_root), "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LegacyDwaComparisonError(
            f"无法确认旧 worktree 分支：{result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_legacy_assets(
    legacy_root: Path,
    task_json: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """确认旧 worktree 只替换局部规划器，核心资产与合同相同。"""

    if _legacy_branch(legacy_root) != EXPECTED_LEGACY_BRANCH:
        raise LegacyDwaComparisonError(
            f"旧 worktree 必须位于 {EXPECTED_LEGACY_BRANCH!r} 分支。"
        )
    shared = _mapping(
        contract.get("shared_runtime_contract"),
        "shared_runtime_contract",
    )
    if shared.get("policy_task") != EXPECTED_POLICY_TASK:
        raise LegacyDwaComparisonError("合同不是原 Go2-X5 mobile-manipulation policy。")
    checkpoint = legacy_root / LEGACY_CHECKPOINT_RELATIVE
    collision = legacy_root / LEGACY_COLLISION_PLY_RELATIVE
    identities = {
        "policy_checkpoint": {
            "path": os.fspath(checkpoint),
            "sha256": _sha256_file(checkpoint),
        },
        "collision_ply": {
            "path": os.fspath(collision),
            "sha256": _sha256_file(collision),
        },
        "task_json": {
            "path": os.fspath(task_json),
            "sha256": _sha256_file(task_json),
        },
    }
    expected_identities = {
        "policy_checkpoint": _mapping(
            shared.get("policy_checkpoint"),
            "shared_runtime_contract.policy_checkpoint",
        ),
        "collision_ply": _mapping(
            shared.get("collision_ply"),
            "shared_runtime_contract.collision_ply",
        ),
        "task_json": _mapping(
            shared.get("task_json"),
            "shared_runtime_contract.task_json",
        ),
    }
    for label, identity in identities.items():
        expected_sha = expected_identities[label].get("sha256")
        if identity["sha256"] != expected_sha:
            raise LegacyDwaComparisonError(
                f"{label} SHA256与合同不一致："
                f"expected={expected_sha!r}, actual={identity['sha256']}"
            )
    return identities


def build_legacy_argv(
    *,
    contract: Mapping[str, Any],
    task_json: Path,
    output_dir: Path,
) -> list[str]:
    """生成不允许 DWA 参数漂移的旧 runner 参数。"""

    source_run = _mapping(contract.get("source_run"), "source_run")
    seed = int(source_run["seed"])
    return [
        "--scene-profile",
        "multi_floor",
        "--task-json",
        os.fspath(task_json),
        "--navigation-carry-smoke",
        "--seed",
        str(seed),
        "--num-episodes",
        "1",
        "--no-randomize-task",
        "--no-randomize-base-goal",
        "--global-planner",
        "pct",
        "--pct-no-fallback",
        "--pct-stair-float",
        "--pct-robot-root-to-floor",
        "0.338",
        "--pct-stair-float-speed",
        "0.18",
        "--pct-stair-float-activation-radius",
        "0.35",
        "--pct-stair-float-completion-radius",
        "0.0",
        "--pct-stair-float-approach-distance",
        "1.5",
        "--pct-stair-float-exit-distance",
        "0.4",
        "--pct-stair-float-settle-time",
        "1.2",
        "--pct-stair-float-release-settle-time",
        "1.0",
        "--pct-stair-float-min-root-z-offset",
        "0.338",
        "--pct-stair-float-release-root-z-offset",
        "0.338",
        "--goal-z-tolerance",
        "0.12",
        "--navigation-visual-mode",
        "collision",
        "--headless",
        "--no-keep-window-open",
        "--no-record-dataset",
        "--no-record-video",
        "--no-show-planned-trajectories",
        "--output-dir",
        os.fspath(output_dir),
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取 SCAN fresh run 的不可变合同，在 pct_scene worktree 中以同一路径、"
            "checkpoint、楼梯冻结和终止门运行旧 DWA。"
        )
    )
    parser.add_argument(
        "--legacy-root",
        default="/mnt/sage_data/workspace/pct_scene",
        help="只读运行旧 DWA 的 pct_scene worktree。",
    )
    parser.add_argument("--contract", required=True, help="SCAN 导出的比较合同JSON。")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="必须原本不存在的 DWA fresh 输出目录。",
    )
    parser.add_argument(
        "--task-json",
        default=os.fspath(
            PROJECT_ROOT / "tasks/nav_pick_place_apple_multifloor_pct.json"
        ),
        help="默认使用合同对应的当前严格任务JSON。",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="只验证合同并打印将执行的旧 runner 参数，不启动 Isaac。",
    )
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="只分析已经存在的输出并写入legacy_dwa_analysis.json。",
    )
    parser.add_argument(
        "--_run-child",
        action="store_true",
        dest="run_child",
        help=argparse.SUPPRESS,
    )
    return parser


def _comparison_request(
    *,
    legacy_root: Path,
    contract_path: Path,
    output_dir: Path,
    task_json: Path,
    contract: Mapping[str, Any],
    identities: Mapping[str, Any],
    legacy_argv: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "legacy_dwa_comparison_run_v1",
        "planner": "dwa",
        "legacy_root": os.fspath(legacy_root),
        "legacy_branch": EXPECTED_LEGACY_BRANCH,
        "comparison_contract": {
            "path": os.fspath(contract_path),
            "contract_payload_sha256": contract["contract_payload_sha256"],
            "global_path_points_sha256": contract["global_path"]["points_sha256"],
            "seed": contract["source_run"]["seed"],
        },
        "effective_place_goal": {
            "position_base_xyz": contract["goal"]["position_base_xyz"],
            "yaw_rad": contract["goal"]["yaw_rad"],
            "reason": "match_current_pct_floor_height_calibration",
        },
        "effective_initial_condition": {
            "base_xyzyaw": contract["initial_condition"][
                "nav_stage_first_diagnostic_base_xyzyaw"
            ],
            "reason": "match_scan_nav_stage_first_diagnostic_state",
        },
        "verified_assets": identities,
        "legacy_runner_argv": legacy_argv,
        "production_runtime_unchanged": True,
        "dwa_imported_into_pct_scan_runtime": False,
    }


def _run_legacy_in_process(
    *,
    legacy_root: Path,
    contract: Mapping[str, Any],
    legacy_argv: Sequence[str],
) -> int:
    """只在子进程中载入 Isaac；父进程负责最终证据落盘。"""

    legacy_root_text = os.fspath(legacy_root)
    sys.path.insert(0, legacy_root_text)
    patch_finder = NavigationSmokePatchFinder(
        lambda module: patch_navigation_smoke_module(module, contract)
    )
    sys.meta_path.insert(0, patch_finder)
    runner_path = legacy_root / "scripts/pipeline/run_full_physics_pipeline.py"
    runner_spec = importlib.util.spec_from_file_location(
        "_pct_scene_legacy_full_physics_runner",
        runner_path,
    )
    if runner_spec is None or runner_spec.loader is None:
        raise LegacyDwaComparisonError(f"无法载入旧 runner：{runner_path}")
    runner = importlib.util.module_from_spec(runner_spec)
    sys.modules[runner_spec.name] = runner
    runner_spec.loader.exec_module(runner)
    return int(runner.main(list(legacy_argv)))


def _read_child_summary(
    output_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """父进程独立验证旧 runner 结果，拒绝仅凭进程退出码通过。"""

    summary_path = output_dir / "episode_000000/summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "summary_path": os.fspath(summary_path),
            "errors": [f"summary_unavailable:{type(exc).__name__}:{exc}"],
        }
    if not isinstance(summary, dict):
        return {
            "valid": False,
            "summary_path": os.fspath(summary_path),
            "errors": ["summary_not_mapping"],
        }
    contract_errors: list[str] = []
    completion_errors: list[str] = []
    navigation_success = (
        summary.get("success") is True and summary.get("final_state") == "done"
    )
    navigation_failure = (
        summary.get("success") is False
        and summary.get("final_state") == "failed"
        and isinstance(summary.get("failure_reason"), str)
        and bool(summary.get("failure_reason"))
    )
    executor = summary.get("latest_executor_status")
    executor = executor if isinstance(executor, Mapping) else {}
    audit = executor.get("comparison_contract")
    audit = audit if isinstance(audit, Mapping) else {}
    required_pairs = {
        "contract_payload_sha256": contract["contract_payload_sha256"],
        "global_path_points_sha256": contract["global_path"]["points_sha256"],
        "stair_freeze_contract_sha256": contract["shared_runtime_contract"][
            "stair_freeze"
        ]["contract_sha256"],
        "post_goal_zero_write_ticks_required": contract[
            "termination_contract"
        ]["post_goal_zero_write_ticks"],
    }
    for key, expected in required_pairs.items():
        if audit.get(key) != expected:
            contract_errors.append(
                f"comparison_audit_mismatch:{key}:"
                f"expected={expected!r}:actual={audit.get(key)!r}"
            )
    if navigation_success:
        for key in ("stable_dwell_verified", "complete"):
            if audit.get(key) is not True:
                completion_errors.append(f"comparison_audit_false:{key}")
    executor_config_audit = audit.get("legacy_executor_config_audit")
    executor_config_audit = (
        executor_config_audit
        if isinstance(executor_config_audit, Mapping)
        else {}
    )
    if executor_config_audit.get("verified") is not True:
        contract_errors.append("legacy_executor_config_not_verified")
    executor_config_expected = {
        "position_tolerance": contract["termination_contract"][
            "final_position_tolerance"
        ],
        "carry_position_tolerance": contract["termination_contract"][
            "place_position_tolerance"
        ],
        "yaw_tolerance": contract["termination_contract"][
            "final_yaw_tolerance"
        ],
        "completion_linear_velocity_tolerance": contract[
            "termination_contract"
        ]["stable_linear_velocity"],
        "completion_angular_velocity_tolerance": contract[
            "termination_contract"
        ]["stable_angular_velocity"],
        "require_yaw_alignment": True,
        "require_stable_base": True,
        "goal_z_tolerance": contract["termination_contract"][
            "finish_distance_z"
        ],
    }
    for key, expected in executor_config_expected.items():
        actual = executor_config_audit.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            try:
                matches = math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            contract_errors.append(
                f"legacy_executor_config_mismatch:{key}:"
                f"expected={expected!r}:actual={actual!r}"
            )
    if audit.get("post_goal_nonzero_write_count") != 0:
        contract_errors.append("post_goal_nonzero_write_detected")
    if navigation_success and audit.get("post_goal_zero_write_count") != contract[
        "termination_contract"
    ]["post_goal_zero_write_ticks"]:
        completion_errors.append("post_goal_zero_write_count_mismatch")
    if not navigation_success and not navigation_failure:
        completion_errors.append(
            "navigation_outcome_invalid:"
            f"success={summary.get('success')!r}:"
            f"final_state={summary.get('final_state')!r}:"
            f"failure_reason={summary.get('failure_reason')!r}"
        )
    comparison_trial_valid = (
        not contract_errors
        and not completion_errors
        and (navigation_success or navigation_failure)
    )
    errors = contract_errors + completion_errors
    return {
        "valid": comparison_trial_valid and navigation_success,
        "comparison_trial_valid": comparison_trial_valid,
        "navigation_success": navigation_success,
        "navigation_failure_is_valid_outcome": navigation_failure,
        "summary_path": os.fspath(summary_path),
        "episode_success": summary.get("success"),
        "final_state": summary.get("final_state"),
        "failure_reason": summary.get("failure_reason"),
        "duration_steps": summary.get("duration_steps"),
        "comparison_audit": dict(audit),
        "contract_errors": contract_errors,
        "completion_errors": completion_errors,
        "errors": errors,
    }


def _optional_pose_xyz(frame: Mapping[str, Any]) -> tuple[float, float, float] | None:
    post = frame.get("post_step_observation")
    if not isinstance(post, Mapping):
        return None
    raw_pose = post.get("robot_root_pose")
    if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) < 3:
        return None
    try:
        return tuple(
            _finite_float(raw_pose[index], f"frame.pose[{index}]")
            for index in range(3)
        )
    except LegacyDwaComparisonError:
        return None


def _compact_dwa_failure(metadata: Mapping[str, Any]) -> dict[str, Any]:
    dwa = metadata.get("dwa")
    dwa = dwa if isinstance(dwa, Mapping) else {}
    map_selection = metadata.get("map_selection")
    map_selection = map_selection if isinstance(map_selection, Mapping) else {}
    post_replan = map_selection.get("post_stair_floor_replan")
    post_replan = post_replan if isinstance(post_replan, Mapping) else {}
    optimization = post_replan.get("path_optimization")
    optimization = optimization if isinstance(optimization, Mapping) else {}
    release_bridge = optimization.get("release_bridge")
    release_bridge = release_bridge if isinstance(release_bridge, Mapping) else {}
    return {
        "phase": metadata.get("phase"),
        "failed": metadata.get("failed"),
        "failure_reason": metadata.get("failure_reason"),
        "sampled_candidates": dwa.get("sampled_candidates"),
        "feasible_candidates": dwa.get("feasible_candidates"),
        "collision_rejections": dwa.get("collision_rejections"),
        "path_deviation_rejections": dwa.get("path_deviation_rejections"),
        "clearance_m": dwa.get("clearance"),
        "occupied_start_escape_active": dwa.get(
            "occupied_start_escape_active"
        ),
        "occupied_start_escape_candidates": dwa.get(
            "occupied_start_escape_candidates"
        ),
        "active_map": map_selection.get("active_map"),
        "post_stair_replan_applied": post_replan.get("applied"),
        "post_stair_replan_reason": post_replan.get("reason"),
        "release_bridge": dict(release_bridge),
    }


def _audit_post_stair_handoff(
    diagnostics: Mapping[str, Any],
    release_error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """确认两种局部规划器从同一个无碰撞二楼位姿开始接管。"""

    errors: list[str] = []
    bridge = diagnostics.get("release_bridge")
    bridge = bridge if isinstance(bridge, Mapping) else {}
    if release_error is None:
        errors.append("post_stair_release_pose_unavailable")
    else:
        xy_error = _finite_float(
            release_error.get("xy_error_m"),
            "post_stair_handoff.release_error.xy_error_m",
        )
        z_error = _finite_float(
            release_error.get("z_error_m"),
            "post_stair_handoff.release_error.z_error_m",
        )
        if xy_error > 0.05:
            errors.append("post_stair_release_xy_mismatch")
        if z_error > 0.05:
            errors.append("post_stair_release_z_mismatch")
    if not bridge:
        errors.append("post_stair_release_bridge_unavailable")
    else:
        if bridge.get("applied") is not True:
            errors.append("post_stair_release_bridge_not_applied")
        if bridge.get("mode") != "collision_checked_direct":
            errors.append("post_stair_release_not_in_common_free_space")
        if bridge.get("bridge_is_clear") is not True:
            errors.append("post_stair_release_bridge_not_clear")
    return {
        "verified": not errors,
        "contract_compatible": not errors,
        "errors": errors,
        "release_bridge": dict(bridge),
        "release_error": None if release_error is None else dict(release_error),
    }


def analyze_legacy_dwa_result(
    output_dir: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """流式提取 DWA 的公平主指标、楼梯交接和失败根因。"""

    frames_path = output_dir / "episode_000000/frames.jsonl"
    summary_validation = _read_child_summary(output_dir, contract)
    summary_path = output_dir / "episode_000000/summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyDwaComparisonError(
            f"无法读取旧 DWA summary：{summary_path}: {exc}"
        ) from exc
    summary = _mapping(summary, "legacy_dwa.summary")
    navigation_start_s: float | None = None
    stair_start_s: float | None = None
    f2_resume_s: float | None = None
    outcome_s: float | None = None
    navigation_start_pose: tuple[float, float, float] | None = None
    stair_start_pose: tuple[float, float, float] | None = None
    stair_last_pose: tuple[float, float, float] | None = None
    f2_resume_pose: tuple[float, float, float] | None = None
    first_dwa_metadata: Mapping[str, Any] = {}
    last_dwa_metadata: Mapping[str, Any] = {}
    previous_timestamp: float | None = None
    compute_counts: set[int] = set()
    compute_total_s = 0.0
    compute_max_s = 0.0
    frame_count = 0
    try:
        handle = frames_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise LegacyDwaComparisonError(
            f"无法读取旧 DWA frames：{frames_path}: {exc}"
        ) from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_frame = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise LegacyDwaComparisonError(
                    f"旧 DWA frames第{line_number}行不是合法JSON：{exc}"
                ) from exc
            frame = _mapping(raw_frame, f"frames[{line_number}]")
            timestamp = _finite_float(
                frame.get("timestamp"),
                f"frames[{line_number}].timestamp",
            )
            # 旧 pipeline 会在同一物理拍连续记录 reset→plan 状态迁移；因此允许
            # 相等时间戳，但任何回退都使分段计时失效。
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise LegacyDwaComparisonError("旧 DWA frames时间戳发生回退。")
            previous_timestamp = timestamp
            outcome_s = timestamp
            frame_count += 1
            action = frame.get("action")
            action = action if isinstance(action, Mapping) else {}
            source = action.get("source", "")
            source = source if isinstance(source, str) else ""
            pose = _optional_pose_xyz(frame)
            metadata = action.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}

            if source == "navigation_dwa":
                if navigation_start_s is None:
                    navigation_start_s = timestamp
                    navigation_start_pose = pose
                    first_dwa_metadata = metadata
                if stair_start_s is not None and f2_resume_s is None:
                    f2_resume_s = timestamp
                    f2_resume_pose = pose
                last_dwa_metadata = metadata
                compute = metadata.get("dwa_compute")
                compute = compute if isinstance(compute, Mapping) else {}
                raw_count = compute.get("compute_count")
                if (
                    compute.get("recomputed_this_tick") is True
                    and isinstance(raw_count, int)
                    and not isinstance(raw_count, bool)
                    and raw_count not in compute_counts
                ):
                    duration = _finite_float(
                        compute.get("last_duration_s"),
                        f"frames[{line_number}].dwa_compute.last_duration_s",
                    )
                    compute_counts.add(raw_count)
                    compute_total_s += duration
                    compute_max_s = max(compute_max_s, duration)
            if source.startswith("navigation_stair_float"):
                if stair_start_s is None:
                    stair_start_s = timestamp
                    stair_start_pose = pose
                stair_last_pose = pose

    missing = [
        label
        for label, value in (
            ("navigation_start", navigation_start_s),
            ("stair_start", stair_start_s),
            ("f2_resume", f2_resume_s),
            ("outcome", outcome_s),
        )
        if value is None
    ]
    if missing:
        raise LegacyDwaComparisonError(
            "旧 DWA 运行缺少跨层分段：" + ",".join(missing)
        )
    assert navigation_start_s is not None
    assert stair_start_s is not None
    assert f2_resume_s is not None
    assert outcome_s is not None
    if not navigation_start_s < stair_start_s < f2_resume_s <= outcome_s:
        raise LegacyDwaComparisonError("旧 DWA 跨层分段时间顺序无效。")

    latest_executor = summary.get("latest_executor_status")
    latest_executor = (
        latest_executor if isinstance(latest_executor, Mapping) else {}
    )
    final_compute = latest_executor.get("dwa_compute")
    final_compute = final_compute if isinstance(final_compute, Mapping) else {}
    final_compute_count = final_compute.get("compute_count")
    if (
        final_compute.get("recomputed_this_tick") is True
        and isinstance(final_compute_count, int)
        and not isinstance(final_compute_count, bool)
        and final_compute_count not in compute_counts
    ):
        final_duration = _finite_float(
            final_compute.get("last_duration_s"),
            "summary.latest_executor_status.dwa_compute.last_duration_s",
        )
        compute_counts.add(final_compute_count)
        compute_total_s += final_duration
        compute_max_s = max(compute_max_s, final_duration)

    stair = first_dwa_metadata.get("stair_float")
    stair = stair if isinstance(stair, Mapping) else {}
    stair_end_ground = stair.get("end")
    stair_end_ground_xyz = (
        _finite_vector(stair_end_ground, 3, "legacy_dwa.stair_float.end")
        if stair_end_ground is not None
        else None
    )
    body_height = _finite_float(
        _mapping(
            contract.get("shared_runtime_contract"),
            "shared_runtime_contract",
        ).get("body_height_m"),
        "shared_runtime_contract.body_height_m",
    )
    release_error: dict[str, Any] | None = None
    if stair_end_ground_xyz is not None and f2_resume_pose is not None:
        expected_release_root = (
            stair_end_ground_xyz[0],
            stair_end_ground_xyz[1],
            stair_end_ground_xyz[2] + body_height,
        )
        release_error = {
            "expected_root_xyz": list(expected_release_root),
            "observed_f2_resume_root_xyz": list(f2_resume_pose),
            "xy_error_m": math.dist(
                expected_release_root[:2],
                f2_resume_pose[:2],
            ),
            "z_error_m": abs(expected_release_root[2] - f2_resume_pose[2]),
        }

    outcome_diagnostics = _compact_dwa_failure(
        latest_executor if latest_executor else last_dwa_metadata
    )
    handoff_audit = _audit_post_stair_handoff(
        outcome_diagnostics,
        release_error,
    )
    comparison_trial_valid = bool(
        summary_validation.get("comparison_trial_valid") is True
        and handoff_audit["contract_compatible"] is True
    )
    f1_s = stair_start_s - navigation_start_s
    common_stair_s = f2_resume_s - stair_start_s
    f2_s = outcome_s - f2_resume_s
    return {
        "schema": "legacy_dwa_navigation_analysis_v2",
        "planner": "dwa",
        "seed": contract["source_run"]["seed"],
        "comparison_trial_valid": comparison_trial_valid,
        "comparison_invalidation_reasons": list(handoff_audit["errors"]),
        "navigation_success": summary_validation.get("navigation_success"),
        "failure_reason": summary_validation.get("failure_reason"),
        "global_path_points_sha256": contract["global_path"]["points_sha256"],
        "global_path_point_count": contract["global_path"]["point_count"],
        "contract_payload_sha256": contract["contract_payload_sha256"],
        "frame_count": frame_count,
        "timing": {
            "navigation_control_start_s": navigation_start_s,
            "stair_takeover_start_s": stair_start_s,
            "stair_release_f2_resume_s": f2_resume_s,
            "navigation_outcome_s": outcome_s,
            "f1_planner_controlled_sim_time_s": f1_s,
            "common_stair_freeze_sim_time_s": common_stair_s,
            "f2_to_outcome_sim_time_s": f2_s,
            "planner_controlled_to_outcome_sim_time_s": f1_s + f2_s,
            "primary_metric_excludes_common_stair_freeze": True,
            "failed_run_time_is_time_to_failure_not_completion": (
                summary_validation.get("navigation_success") is not True
            ),
        },
        "handoff": {
            "navigation_start_root_xyz": (
                None if navigation_start_pose is None else list(navigation_start_pose)
            ),
            "stair_start_root_xyz": (
                None if stair_start_pose is None else list(stair_start_pose)
            ),
            "last_frozen_root_xyz": (
                None if stair_last_pose is None else list(stair_last_pose)
            ),
            "release_error": release_error,
            "contract_audit": handoff_audit,
            "stair_profile_contract_sha256": contract[
                "shared_runtime_contract"
            ]["stair_freeze"]["contract_sha256"],
        },
        "dwa_compute": {
            "unique_recompute_count": len(compute_counts),
            "successful_compute_wall_time_total_s": compute_total_s,
            "maximum_compute_wall_time_s": compute_max_s,
        },
        "outcome_diagnostics": outcome_diagnostics,
        "summary_validation": summary_validation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    legacy_root = Path(args.legacy_root).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    task_json = Path(args.task_json).expanduser().resolve()
    if PROJECT_ROOT not in output_dir.parents:
        raise LegacyDwaComparisonError(
            f"DWA 输出必须位于当前 pct_scan worktree：{PROJECT_ROOT}"
        )

    contract = load_comparison_contract(contract_path)
    identities = validate_legacy_assets(legacy_root, task_json, contract)
    legacy_argv = build_legacy_argv(
        contract=contract,
        task_json=task_json,
        output_dir=output_dir,
    )
    request = _comparison_request(
        legacy_root=legacy_root,
        contract_path=contract_path,
        output_dir=output_dir,
        task_json=task_json,
        contract=contract,
        identities=identities,
        legacy_argv=legacy_argv,
    )
    if args.print_command:
        print(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.analyze_existing:
        if args.run_child:
            raise LegacyDwaComparisonError(
                "--analyze-existing不能与内部--_run-child同时使用。"
            )
        if not output_dir.is_dir():
            raise LegacyDwaComparisonError(
                f"待分析的DWA输出目录不存在：{output_dir}"
            )
        analysis = analyze_legacy_dwa_result(output_dir, contract)
        analysis_path = output_dir / "legacy_dwa_analysis.json"
        _write_json(analysis_path, analysis)
        print(
            json.dumps(
                {
                    "output": os.fspath(analysis_path),
                    "comparison_trial_valid": analysis[
                        "comparison_trial_valid"
                    ],
                    "navigation_success": analysis["navigation_success"],
                    "failure_reason": analysis["failure_reason"],
                    "planner_controlled_to_outcome_sim_time_s": analysis[
                        "timing"
                    ]["planner_controlled_to_outcome_sim_time_s"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if analysis["comparison_trial_valid"] is True else 1
    if args.run_child:
        if not output_dir.is_dir():
            raise LegacyDwaComparisonError(
                f"子进程要求父进程已创建输出目录：{output_dir}"
            )
        return _run_legacy_in_process(
            legacy_root=legacy_root,
            contract=contract,
            legacy_argv=legacy_argv,
        )
    if output_dir.exists():
        raise LegacyDwaComparisonError(f"输出目录必须原本不存在：{output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "legacy_dwa_comparison.json"
    child_command = [
        sys.executable,
        "-B",
        os.fspath(Path(__file__).resolve()),
        "--legacy-root",
        os.fspath(legacy_root),
        "--contract",
        os.fspath(contract_path),
        "--output-dir",
        os.fspath(output_dir),
        "--task-json",
        os.fspath(task_json),
        "--_run-child",
    ]
    _write_json(
        manifest_path,
        {
            **request,
            "status": "running",
            "child_command": child_command,
        },
    )
    child_returncode: int | None = None
    error: dict[str, str] | None = None
    try:
        child = subprocess.run(
            child_command,
            cwd=os.fspath(PROJECT_ROOT),
            check=False,
        )
        child_returncode = int(child.returncode)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    summary_validation = _read_child_summary(output_dir, contract)
    analysis: dict[str, Any] | None = None
    analysis_error: dict[str, str] | None = None
    try:
        analysis = analyze_legacy_dwa_result(output_dir, contract)
    except BaseException as exc:
        analysis_error = {"type": type(exc).__name__, "message": str(exc)}
    comparison_completed = (
        child_returncode == 0
        and summary_validation.get("comparison_trial_valid") is True
        and analysis is not None
        and analysis.get("comparison_trial_valid") is True
    )
    navigation_success = (
        comparison_completed
        and summary_validation.get("navigation_success") is True
    )
    status = (
        "passed"
        if navigation_success
        else "completed_navigation_failure"
        if comparison_completed
        else "invalid"
    )
    _write_json(
        manifest_path,
        {
            **request,
            "status": status,
            "child_command": child_command,
            "child_returncode": child_returncode,
            "summary_validation": summary_validation,
            "navigation_analysis": analysis,
            "navigation_analysis_error": analysis_error,
            "error": error,
        },
    )
    if error is not None:
        raise LegacyDwaComparisonError(
            f"DWA Isaac 子进程启动失败：{error['type']}: {error['message']}"
        )
    return 0 if comparison_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
