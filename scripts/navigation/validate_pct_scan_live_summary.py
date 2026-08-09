#!/usr/bin/env python3
"""离线校验 PCT→SCAN 五类真实运行摘要。

校验器只读取 ``summary.json``，不导入 ROS 2、Isaac Sim 或 GPU 运行时。
四种模式使用不同的成功合同：楼梯冻结只要求 supervisor 状态被可靠观察，
非冻结平地和动态 F1 则必须证明许可已被唯一 policy writer 消费。动态模式
仅认证 summary 已经结构化记录的推车配置、PhysX 位姿写入、原始 RTX 点云命中
和运动生命周期。``dynamic_f1`` 校验局部绕障恢复：过滤后点云命中、B-spline 有序绕行
与当前障碍净距、explicit-miss 清 ghost 以及 controller 实际接受的恢复轨迹，必须
由 typed ROS 2 诊断和同一时间窗/identity 共同证明；任一引用缺失或错配都会失败。
``dynamic_replan_f1`` 独立校验阻断型推车触发的 PCT request/in-flight/new plan，
并要求同一 occupied epoch 在重规划后被显式 miss 清除、随后恢复有效跟踪。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.navigation.scan_stair_freeze_profile import (  # noqa: E402
    ScanStairFreezeProfileError,
    load_scan_stair_freeze_profile,
)
from source.simulation.dynamic_obstacles import (  # noqa: E402
    DynamicObstaclePlan,
    DynamicObstacleSpec,
    DynamicObstacleState,
    resolve_dynamic_obstacle_plan,
)


ValidationMode = Literal[
    "static_stair",
    "flat_policy",
    "crossfloor_carry",
    "dynamic_f1",
    "dynamic_replan_f1",
]
VALIDATION_MODES: tuple[ValidationMode, ...] = (
    "static_stair",
    "flat_policy",
    "crossfloor_carry",
    "dynamic_f1",
    "dynamic_replan_f1",
)

_MISSING = object()
_ZERO_COMMAND = (0.0, 0.0, 0.0)
_IDENTITY_STATE_TRACKING = 3
_NAVIGATION_STATE_EMERGENCY_STOP = 5
_NAVIGATION_STAIR_INHIBIT_REASON = "scan_stair_execution_inhibited"
_EXPECTED_GATE_TIMEOUT_S = 0.25
_DYNAMIC_POSITION_TOLERANCE_M = 1.0e-6
_DYNAMIC_GEOMETRY_TOLERANCE_M = 1.0e-6
_MAX_POST_FILTER_HIT_TOLERANCE_M = 0.05
_DIAGNOSTIC_RING_CAPACITY = 128
_PROOF_RING_CAPACITY = 64
_DYNAMIC_RECOVERY_MAX_DEVIATION_M = 0.02
_DYNAMIC_RECOVERY_MIN_IMPROVEMENT_M = 0.01
_GO2_X5_DOUBLE_CYLINDER_RADIUS_M = 0.27
_GO2_X5_DOUBLE_CYLINDER_OFFSET_M = 0.16
_EXPECTED_SCAN_REPLAN_FAILURE_COUNT = 5
_SCAN_STAIR_FREEZE_PROFILE_PATH = (
    PROJECT_ROOT
    / "configs/navigation/scan_stair_freeze_go2_x5_multifloor_v1.json"
)

_COMMON_VALIDATED_CLAIMS = (
    "episode_success_and_goal_reached",
    "pct_goal_and_live_path_identity_bound",
    "supervisor_status_sequence_observed",
    "post_goal_policy_zero_hold",
)
_DYNAMIC_FINAL_CLAIMS = (
    "dynamic_obstacle_preserved_in_post_filter_cloud_registered",
    "scan_ordered_local_detour_and_current_obstacle_clearance_verified",
    "explicit_free_ray_live_ghost_clear_verified",
    "controller_tracking_recovery_identity_observed_before_policy_write",
)
_DYNAMIC_REPLAN_CLAIMS = (
    "dynamic_obstacle_free_to_occupied_precedes_scan_replan",
    "pct_replan_request_inflight_and_new_plan_identity_verified",
    "same_obstacle_explicit_miss_clear_after_replan_verified",
    "identity_valid_tracking_and_policy_motion_recovered_after_clear",
)
_ACTIVE_SENSING_CLAIMS = (
    "scan_active_sensing_typed_lifecycle_completed",
    "active_sensing_controller_and_policy_commands_bounded",
    "same_pct_path_recovered_after_three_post_settle_fusions",
)

_ACTIVE_SENSING_EVENT_CODES = {
    "STARTED": 1,
    "ACCEPTED": 2,
    "YAW_STABLE": 3,
    "FUSION_PROGRESS": 4,
    "COMPLETED": 5,
    "FAILED": 6,
}
_ACTIVE_SENSING_MAX_YAW_OFFSET_RAD = 0.22
_ACTIVE_SENSING_MAX_YAW_RATE_RAD_S = 0.20
_ACTIVE_SENSING_MAX_SETTLE_YAW_ERROR_RAD = 0.02
_ACTIVE_SENSING_MAX_SETTLE_ANGULAR_SPEED_RAD_S = 0.05
_ACTIVE_SENSING_MIN_STABLE_DURATION_S = 0.10
_ACTIVE_SENSING_CONTROLLER_START_YAW_TOLERANCE_RAD = 0.02
_ACTIVE_SENSING_REQUIRED_FUSIONS = 3
_ACTIVE_SENSING_FUSION_HISTORY_CAPACITY = 64
_CONTROLLER_STATE_ALIGNING_YAW = 9
_CONTROLLER_STATE_TRACKING = 10


class SummaryInputError(ValueError):
    """表示 summary 文件本身无法作为严格 JSON 对象读取。"""


@dataclass(slots=True)
class _ValidationContext:
    """收集 fail-closed 合同错误，并提供严格类型检查。"""

    mode: ValidationMode
    errors: list[dict[str, Any]] = field(default_factory=list)

    def reject(
        self,
        code: str,
        path: str,
        message: str,
        *,
        actual: object = _MISSING,
    ) -> None:
        issue: dict[str, Any] = {
            "code": code,
            "path": path,
            "message": message,
        }
        if actual is not _MISSING:
            issue["actual"] = actual
        self.errors.append(issue)

    def field(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
    ) -> object:
        if key not in parent:
            self.reject("missing_field", path, "缺少必需字段。")
            return _MISSING
        return parent[key]

    def mapping(self, value: object, path: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self.reject("invalid_type", path, "必须是 JSON 对象。", actual=value)
            return {}
        return value

    def required_mapping(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
    ) -> Mapping[str, Any]:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return {}
        return self.mapping(value, path)

    def boolean(self, value: object, path: str) -> bool | None:
        if not isinstance(value, bool):
            self.reject("invalid_type", path, "必须是布尔值。", actual=value)
            return None
        return value

    def required_boolean(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
    ) -> bool | None:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return None
        return self.boolean(value, path)

    def integer(
        self,
        value: object,
        path: str,
        *,
        minimum: int | None = None,
    ) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            self.reject("invalid_type", path, "必须是整数。", actual=value)
            return None
        if minimum is not None and value < minimum:
            self.reject(
                "invalid_number",
                path,
                f"必须不小于 {minimum}。",
                actual=value,
            )
            return None
        return value

    def required_integer(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
        *,
        minimum: int | None = None,
    ) -> int | None:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return None
        return self.integer(value, path, minimum=minimum)

    def number(
        self,
        value: object,
        path: str,
        *,
        minimum: float | None = None,
    ) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.reject("invalid_type", path, "必须是有限实数。", actual=value)
            return None
        normalized = float(value)
        if not math.isfinite(normalized):
            self.reject("invalid_number", path, "必须是有限实数。", actual=value)
            return None
        if minimum is not None and normalized < minimum:
            self.reject(
                "invalid_number",
                path,
                f"必须不小于 {minimum}。",
                actual=value,
            )
            return None
        return normalized

    def required_number(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
        *,
        minimum: float | None = None,
    ) -> float | None:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return None
        return self.number(value, path, minimum=minimum)

    def string(
        self,
        value: object,
        path: str,
        *,
        nonempty: bool = False,
    ) -> str | None:
        if not isinstance(value, str):
            self.reject("invalid_type", path, "必须是字符串。", actual=value)
            return None
        if nonempty and not value.strip():
            self.reject("unexpected_value", path, "必须是非空字符串。", actual=value)
            return None
        return value

    def required_string(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
        *,
        nonempty: bool = False,
    ) -> str | None:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return None
        return self.string(value, path, nonempty=nonempty)

    def sequence(self, value: object, path: str) -> Sequence[object]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value,
            Sequence,
        ):
            self.reject("invalid_type", path, "必须是 JSON 数组。", actual=value)
            return ()
        return value

    def required_sequence(
        self,
        parent: Mapping[str, Any],
        key: str,
        path: str,
    ) -> Sequence[object]:
        value = self.field(parent, key, path)
        if value is _MISSING:
            return ()
        return self.sequence(value, path)

    def expect(self, actual: object, expected: object, path: str) -> None:
        if actual is _MISSING:
            return
        if actual != expected or type(actual) is not type(expected):
            self.reject(
                "unexpected_value",
                path,
                f"必须严格等于 {expected!r}。",
                actual=actual,
            )

    def expect_true(self, actual: object, path: str) -> None:
        value = self.boolean(actual, path)
        if value is not None and value is not True:
            self.reject("unexpected_value", path, "必须为 true。", actual=value)

    def expect_false(self, actual: object, path: str) -> None:
        value = self.boolean(actual, path)
        if value is not None and value is not False:
            self.reject("unexpected_value", path, "必须为 false。", actual=value)


@dataclass(frozen=True, slots=True)
class _CommonEvidence:
    """四种 live 模式共同使用的身份与生命周期字段。"""

    task_config: Mapping[str, Any]
    executor: Mapping[str, Any]
    simulation: Mapping[str, Any]
    lifecycle: Mapping[str, Any]
    pct_goal_stamp_ns: int | None
    path_stamp_ns: int | None
    policy_write_count: int | None
    first_observed_sequence: int | None
    last_observed_sequence: int | None


@dataclass(frozen=True, slots=True)
class _GridMapDiagnosticEvidence:
    """已严格解析的一帧 GridMap typed 观测证据。"""

    report: Mapping[str, Any]
    receipt_timestamp: float | None
    rx_sequence: int | None
    ros_time_offset_s: float | None
    header_stamp_ns: int | None
    episode_elapsed_time_s: float | None
    observation_sequence: int | None
    map_resolution_m: float | None
    hit_samples: tuple[tuple[float, float, float], ...]
    hit_voxel_indices: tuple[tuple[int, int, int], ...]
    transition_hit_samples: tuple[tuple[float, float, float], ...]
    transition_voxel_indices: tuple[tuple[int, int, int], ...]
    clear_samples: tuple[tuple[float, float, float], ...]
    clear_voxel_indices: tuple[tuple[int, int, int], ...]
    clear_transition_hit_sequences: tuple[int, ...]
    clear_transition_hit_samples: tuple[tuple[float, float, float], ...]
    clear_transition_hit_header_stamps_ns: tuple[int, ...]
    hit_endpoint_count: int | None
    free_to_occupied_transition_count: int | None
    explicit_free_miss_voxel_count: int | None
    occupied_to_free_count: int | None
    sliding_reset_count: int | None


@dataclass(frozen=True, slots=True)
class _BsplineDiagnosticEvidence:
    """已严格解析的一条 B-spline typed 几何与 identity 证据。"""

    report: Mapping[str, Any]
    receipt_timestamp: float | None
    rx_sequence: int | None
    ros_time_offset_s: float | None
    header_stamp_ns: int | None
    episode_elapsed_time_s: float | None
    diagnostic_sequence: int | None
    identity: tuple[int, int, int, int] | None
    stationary: bool | None
    emergency_stop: bool | None
    ordered_reference_checked: bool | None
    ordered_reference_safe: bool | None
    maximum_trajectory_deviation: float | None
    trajectory_duration_s: float | None
    maximum_velocity_upper_bound_mps: float | None
    required_any_yaw_clearance_radius_m: float | None
    trajectory_sample_interval_s: float | None
    sampling_clearance_margin_m: float | None
    trajectory_samples: tuple[tuple[float, float, float], ...]
    reference_samples: tuple[tuple[float, float, float], ...]


def _strict_json_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SummaryInputError(f"JSON 含重复键：{key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise SummaryInputError(f"JSON 含非有限常量：{value}")


def load_summary(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """严格读取 episode 目录或 ``summary.json`` 文件。"""

    summary_path = Path(path).expanduser().resolve()
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    if not summary_path.is_file():
        raise SummaryInputError(f"summary 文件不存在：{summary_path}")
    try:
        raw_text = summary_path.read_text(encoding="utf-8")
        payload = json.loads(
            raw_text,
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except SummaryInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryInputError(
            f"无法读取严格 JSON {summary_path}：{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SummaryInputError("summary JSON 顶层必须是对象。")
    return summary_path, payload


def _field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> object:
    return ctx.field(parent, key, path)


def _expect_exact_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    expected: object,
    path: str,
) -> None:
    ctx.expect(_field(ctx, parent, key, path), expected, path)


def _expect_true_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> None:
    value = _field(ctx, parent, key, path)
    if value is not _MISSING:
        ctx.expect_true(value, path)


def _expect_false_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> None:
    value = _field(ctx, parent, key, path)
    if value is not _MISSING:
        ctx.expect_false(value, path)


def _validate_string_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[str, ...]:
    raw = ctx.sequence(value, path)
    result: list[str] = []
    for index, item in enumerate(raw):
        normalized = ctx.string(item, f"{path}[{index}]", nonempty=True)
        if normalized is not None:
            result.append(normalized)
    return tuple(result)


def _validate_vector(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    length: int,
) -> tuple[float, ...] | None:
    raw = ctx.sequence(value, path)
    if len(raw) != length:
        ctx.reject(
            "invalid_length",
            path,
            f"必须包含 {length} 个元素。",
            actual=len(raw),
        )
        return None
    parsed: list[float] = []
    for index, item in enumerate(raw):
        number = ctx.number(item, f"{path}[{index}]")
        if number is None:
            return None
        parsed.append(number)
    return tuple(parsed)


def _validate_integer_vector(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    length: int,
    minimum: int | None = None,
) -> tuple[int, ...] | None:
    raw = ctx.sequence(value, path)
    if len(raw) != length:
        ctx.reject(
            "invalid_length",
            path,
            f"必须包含 {length} 个整数。",
            actual=len(raw),
        )
        return None
    parsed: list[int] = []
    for index, item in enumerate(raw):
        integer = ctx.integer(item, f"{path}[{index}]", minimum=minimum)
        if integer is None:
            return None
        parsed.append(integer)
    return tuple(parsed)


def _validate_point_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[tuple[float, float, float], ...]:
    """严格解析有界世界系三维点数组。"""

    raw = ctx.sequence(value, path)
    if len(raw) > 64:
        ctx.reject(
            "invalid_length",
            path,
            "typed 诊断最多保留 64 个三维样本。",
            actual=len(raw),
        )
    points: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw):
        parsed = _validate_vector(ctx, item, f"{path}[{index}]", length=3)
        if parsed is not None:
            points.append(parsed)
    return tuple(points)


def _validate_voxel_index_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[tuple[int, int, int], ...]:
    """严格解析与有界点样本逐项对齐的 canonical voxel index。"""

    raw = ctx.sequence(value, path)
    if len(raw) > _PROOF_RING_CAPACITY:
        ctx.reject(
            "invalid_length",
            path,
            "typed 诊断最多保留 64 个 voxel index。",
            actual=len(raw),
        )
    indices: list[tuple[int, int, int]] = []
    for index, item in enumerate(raw):
        parsed = _validate_integer_vector(
            ctx,
            item,
            f"{path}[{index}]",
            length=3,
        )
        if parsed is not None:
            indices.append(parsed)
    return tuple(indices)


def _validate_observation_sequence_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[int, ...]:
    """解析与 clear 样本等长的 free→occupied 来源序号。"""

    raw = ctx.sequence(value, path)
    if len(raw) > _PROOF_RING_CAPACITY:
        ctx.reject(
            "invalid_length",
            path,
            "typed 诊断最多保留 64 个 provenance sequence。",
            actual=len(raw),
        )
    sequences: list[int] = []
    for index, item in enumerate(raw):
        parsed = ctx.integer(item, f"{path}[{index}]", minimum=0)
        if parsed is not None:
            sequences.append(parsed)
    return tuple(sequences)


def _validate_header_stamp_ns_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[int, ...]:
    """解析与 clear 样本等长的来源 ROS header 纳秒时间戳。"""

    raw = ctx.sequence(value, path)
    if len(raw) > _PROOF_RING_CAPACITY:
        ctx.reject(
            "invalid_length",
            path,
            "typed 诊断最多保留 64 个 provenance header stamp。",
            actual=len(raw),
        )
    stamps: list[int] = []
    for index, item in enumerate(raw):
        parsed = ctx.integer(item, f"{path}[{index}]", minimum=0)
        if parsed is not None:
            stamps.append(parsed)
    return tuple(stamps)


def _validate_ros_stamp(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    *,
    stamp_key: str,
    stamp_ns_key: str,
    path: str,
) -> int | None:
    """校验 ROS ``sec/nanosec`` 与扁平纳秒字段严格一致。"""

    stamp_ns = ctx.required_integer(
        parent,
        stamp_ns_key,
        f"{path}.{stamp_ns_key}",
        minimum=1,
    )
    stamp = ctx.required_mapping(parent, stamp_key, f"{path}.{stamp_key}")
    seconds = ctx.required_integer(
        stamp,
        "sec",
        f"{path}.{stamp_key}.sec",
        minimum=0,
    )
    nanoseconds = ctx.required_integer(
        stamp,
        "nanosec",
        f"{path}.{stamp_key}.nanosec",
        minimum=0,
    )
    if nanoseconds is not None and nanoseconds >= 1_000_000_000:
        ctx.reject(
            "invalid_timestamp",
            f"{path}.{stamp_key}.nanosec",
            "nanosec 必须小于 1e9。",
            actual=nanoseconds,
        )
    if seconds is not None and nanoseconds is not None and stamp_ns is not None:
        reconstructed = seconds * 1_000_000_000 + nanoseconds
        if reconstructed != stamp_ns:
            ctx.reject(
                "invalid_timestamp",
                f"{path}.{stamp_key}",
                "sec/nanosec 与对应 *_ns 字段不一致。",
                actual=dict(stamp),
            )
    return stamp_ns


def _validate_stair_execution_freeze_report(
    ctx: _ValidationContext,
    publish: Mapping[str, Any],
    path: str,
    *,
    active_path_stamp_ns: int | None,
) -> None:
    """严格校验 Isaac 发布的 typed 楼梯冻结终态快照。"""

    _expect_exact_field(
        ctx,
        publish,
        "schema",
        "isaac_stair_execution_frozen_v1",
        f"{path}.schema",
    )
    _expect_exact_field(
        ctx,
        publish,
        "message_type",
        "scan_planner_msgs/msg/StairExecutionFreeze",
        f"{path}.message_type",
    )
    _expect_exact_field(
        ctx,
        publish,
        "source",
        "isaac_action_metadata",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        publish,
        "topic",
        "/planning/stair_execution_frozen",
        f"{path}.topic",
    )

    publish_timestamp = ctx.required_number(
        publish,
        "publish_timestamp",
        f"{path}.publish_timestamp",
        minimum=0.0,
    )
    if publish_timestamp is not None and publish_timestamp <= 0.0:
        ctx.reject(
            "invalid_timestamp",
            f"{path}.publish_timestamp",
            "typed 楼梯冻结发布时间必须为正数。",
            actual=publish_timestamp,
        )

    header = ctx.required_mapping(publish, "header", f"{path}.header")
    _expect_exact_field(
        ctx,
        header,
        "frame_id",
        "world",
        f"{path}.header.frame_id",
    )
    header_stamp = ctx.required_mapping(
        header,
        "stamp",
        f"{path}.header.stamp",
    )
    header_sec = ctx.required_integer(
        header_stamp,
        "sec",
        f"{path}.header.stamp.sec",
        minimum=0,
    )
    header_nanosec = ctx.required_integer(
        header_stamp,
        "nanosec",
        f"{path}.header.stamp.nanosec",
        minimum=0,
    )
    if header_nanosec is not None and header_nanosec >= 1_000_000_000:
        ctx.reject(
            "invalid_timestamp",
            f"{path}.header.stamp.nanosec",
            "nanosec 必须小于 1e9。",
            actual=header_nanosec,
        )
    if (
        header_sec is not None
        and header_nanosec is not None
        and header_nanosec < 1_000_000_000
    ):
        header_stamp_ns = header_sec * 1_000_000_000 + header_nanosec
        if header_stamp_ns <= 0:
            ctx.reject(
                "invalid_timestamp",
                f"{path}.header.stamp",
                "typed 楼梯冻结 Header 时间戳必须非零。",
                actual=dict(header_stamp),
            )
        if publish_timestamp is not None:
            expected_header_stamp_ns = int(
                publish_timestamp * 1_000_000_000
            )
            if header_stamp_ns != expected_header_stamp_ns:
                ctx.reject(
                    "invalid_timestamp",
                    f"{path}.header.stamp",
                    "Header 与 publish_timestamp 不属于同一连续仿真时刻。",
                    actual=dict(header_stamp),
                )

    reference_path_stamp_ns = _validate_ros_stamp(
        ctx,
        publish,
        stamp_key="reference_path_stamp",
        stamp_ns_key="reference_path_stamp_ns",
        path=path,
    )
    if (
        active_path_stamp_ns is not None
        and reference_path_stamp_ns != active_path_stamp_ns
    ):
        ctx.reject(
            "wrong_identity",
            f"{path}.reference_path_stamp_ns",
            "typed 楼梯冻结快照未绑定当前精确 Path identity。",
            actual=reference_path_stamp_ns,
        )

    for key in ("writer_id", "writer_epoch"):
        writer = ctx.required_string(
            publish,
            key,
            f"{path}.{key}",
            nonempty=True,
        )
        if writer is not None and writer != writer.strip():
            ctx.reject(
                "invalid_writer_identity",
                f"{path}.{key}",
                "writer identity 不能包含首尾空白。",
                actual=writer,
            )

    _expect_positive_integer_field(
        ctx,
        publish,
        "sequence",
        f"{path}.sequence",
    )
    frozen = ctx.required_boolean(publish, "frozen", f"{path}.frozen")
    value = ctx.required_boolean(publish, "value", f"{path}.value")
    if frozen is not None and value is not None and frozen != value:
        ctx.reject(
            "inconsistent_freeze_state",
            f"{path}.frozen",
            "typed frozen 与兼容 metadata value 必须严格一致。",
            actual=frozen,
        )
    if frozen is not None and frozen is not True:
        ctx.reject(
            "unexpected_value",
            f"{path}.frozen",
            "静态楼梯终态 typed frozen 必须为 true。",
            actual=frozen,
        )
    if value is not None and value is not True:
        ctx.reject(
            "unexpected_value",
            f"{path}.value",
            "静态楼梯终态兼容 value 必须为 true。",
            actual=value,
        )
    _expect_exact_field(
        ctx,
        publish,
        "action_phase",
        "terminal_hold",
        f"{path}.action_phase",
    )
    for key in ("action_source", "decision_reason"):
        ctx.required_string(
            publish,
            key,
            f"{path}.{key}",
            nonempty=True,
        )


def _mapping_identity_tuple(value: object) -> tuple[int, int, int, int] | None:
    """无副作用提取完整 trajectory identity，供跨报告集合匹配。"""

    if not isinstance(value, Mapping):
        return None
    keys = (
        "reference_path_stamp_ns",
        "bspline_header_stamp_ns",
        "start_time_ns",
        "traj_id",
    )
    values = tuple(value.get(key) for key in keys)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        return None
    return values  # type: ignore[return-value]


def _validate_full_trajectory_identity(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    path_stamp_ns: int | None,
) -> tuple[int, int, int, int] | None:
    """复用 controller identity 合同并保留 start_time 组成完整四元组。"""

    _validate_controller_identity(
        ctx,
        value,
        path,
        path_stamp_ns=path_stamp_ns,
    )
    identity = _mapping_identity_tuple(value)
    if identity is None:
        return None
    return identity


def _validate_zero_command(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> None:
    command = _validate_vector(ctx, value, path, length=3)
    if command is not None and any(abs(component) > 1.0e-12 for component in command):
        ctx.reject(
            "post_goal_motion",
            path,
            "目标完成后的最终 policy 写入必须为精确零速。",
            actual=list(command),
        )


def _validate_sha256(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> str | None:
    text = ctx.string(value, path, nonempty=True)
    if text is None:
        return None
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        ctx.reject(
            "invalid_identity",
            path,
            "必须是 64 位小写十六进制 SHA256。",
            actual=text,
        )
        return None
    return text


def _expect_close(
    ctx: _ValidationContext,
    actual: float | None,
    expected: float,
    path: str,
    *,
    tolerance: float = 1.0e-9,
) -> None:
    if actual is not None and not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        ctx.reject(
            "unexpected_value",
            path,
            f"必须在绝对误差 {tolerance:g} 内等于 {expected!r}。",
            actual=actual,
        )


def _expect_positive_integer_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> int | None:
    value = ctx.required_integer(parent, key, path, minimum=1)
    return value


def _expect_nonnegative_integer_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> int | None:
    return ctx.required_integer(parent, key, path, minimum=0)


def _expect_empty_string_field(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    key: str,
    path: str,
) -> None:
    value = ctx.required_string(parent, key, path)
    if value is not None and value != "":
        ctx.reject(
            "unexpected_value",
            path,
            "成功摘要中的失败原因必须为空字符串。",
            actual=value,
        )


def _validate_navigation_qos(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> None:
    raw = ctx.string(value, path, nonempty=True)
    if raw is None:
        return
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (SummaryInputError, json.JSONDecodeError) as exc:
        ctx.reject("invalid_qos", path, f"QoS 必须是严格 JSON：{exc}")
        return
    qos = ctx.mapping(parsed, path)
    expected = {
        "reliability": "reliable",
        "durability": "transientLocal",
        "history": "keepLast",
        "depth": 1,
    }
    for key, expected_value in expected.items():
        ctx.expect(
            ctx.field(qos, key, f"{path}.{key}"),
            expected_value,
            f"{path}.{key}",
        )


def _validate_bridge_report(
    ctx: _ValidationContext,
    simulation: Mapping[str, Any],
) -> None:
    path = "$.simulation_report.navigation_ros2_bridge_report"
    report = ctx.required_mapping(simulation, "navigation_ros2_bridge_report", path)
    _expect_true_field(ctx, report, "enabled", f"{path}.enabled")
    exact_fields = {
        "command_topic": "/cmd_vel",
        "controller_status_topic": "/planning/controller_status",
        "navigation_status_topic": "/navigation/status",
        "stair_execution_frozen_topic": "/planning/stair_execution_frozen",
        "reference_path_topic": "/pct/global_path",
        "pct_goal_topic": "/pct/goal",
        "odom_frame_id": "world",
        "base_frame_id": "base_link",
        "point_cloud_frame_id": "world",
        "continuous_time_source": "successful_physics_steps_x_physics_dt",
    }
    for key, expected in exact_fields.items():
        _expect_exact_field(ctx, report, key, expected, f"{path}.{key}")
    for key in (
        "command_authority_enabled",
        "navigation_status_gate_required",
        "goal_lifecycle_enabled",
        "controller_status_subscription_enabled",
        "stair_execution_frozen_publisher_enabled",
        "reference_path_subscription_enabled",
        "pct_goal_publisher_enabled",
    ):
        _expect_true_field(ctx, report, key, f"{path}.{key}")
    timeout = ctx.required_number(
        report,
        "navigation_status_timeout_s",
        f"{path}.navigation_status_timeout_s",
        minimum=0.0,
    )
    _expect_close(
        ctx,
        timeout,
        _EXPECTED_GATE_TIMEOUT_S,
        f"{path}.navigation_status_timeout_s",
    )
    qos_value = ctx.field(
        report,
        "navigation_status_qos_profile",
        f"{path}.navigation_status_qos_profile",
    )
    if qos_value is not _MISSING:
        _validate_navigation_qos(
            ctx,
            qos_value,
            f"{path}.navigation_status_qos_profile",
        )


def _validate_observed_status_evidence(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    pct_goal_stamp_ns: int | None,
    path_stamp_ns: int | None,
) -> tuple[int | None, int | None, int | None]:
    evidence = ctx.mapping(value, path)
    write_sequence = ctx.required_integer(
        evidence,
        "write_sequence",
        f"{path}.write_sequence",
        minimum=1,
    )
    ctx.required_number(evidence, "timestamp", f"{path}.timestamp", minimum=0.0)
    report_path = f"{path}.navigation_status_observed_report"
    report = ctx.required_mapping(
        evidence,
        "navigation_status_observed_report",
        report_path,
    )
    _expect_exact_field(
        ctx,
        report,
        "schema",
        "navigation_status_observed_diagnostics_v1",
        f"{report_path}.schema",
    )
    _expect_exact_field(
        ctx,
        report,
        "topic",
        "/navigation/status",
        f"{report_path}.topic",
    )
    _expect_exact_field(ctx, report, "status_error", None, f"{report_path}.status_error")
    local_goal = ctx.required_integer(
        report,
        "local_pct_goal_stamp_ns",
        f"{report_path}.local_pct_goal_stamp_ns",
        minimum=1,
    )
    local_path = ctx.required_integer(
        report,
        "local_active_path_stamp_ns",
        f"{report_path}.local_active_path_stamp_ns",
        minimum=1,
    )
    _expect_exact_field(
        ctx,
        report,
        "local_reference_path_identity_fault",
        None,
        f"{report_path}.local_reference_path_identity_fault",
    )
    if pct_goal_stamp_ns is not None and local_goal != pct_goal_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{report_path}.local_pct_goal_stamp_ns",
            "观测状态的本地 PCT goal identity 与 executor 不一致。",
            actual=local_goal,
        )
    if path_stamp_ns is not None and local_path != path_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{report_path}.local_active_path_stamp_ns",
            "观测状态的本地 Path identity 与 executor 不一致。",
            actual=local_path,
        )

    status_path = f"{report_path}.status"
    status = ctx.required_mapping(report, "status", status_path)
    status_sequence = ctx.required_integer(
        status,
        "status_sequence",
        f"{status_path}.status_sequence",
        minimum=1,
    )
    revision = ctx.required_integer(
        status,
        "state_revision",
        f"{status_path}.state_revision",
        minimum=1,
    )
    ctx.required_integer(status, "rx_sequence", f"{status_path}.rx_sequence", minimum=1)
    ctx.required_integer(
        status,
        "header_stamp_ns",
        f"{status_path}.header_stamp_ns",
        minimum=1,
    )
    ctx.required_number(
        status,
        "receipt_timestamp",
        f"{status_path}.receipt_timestamp",
        minimum=0.0,
    )
    goal_id = ctx.required_integer(status, "goal_id", f"{status_path}.goal_id", minimum=1)
    active_path = ctx.required_integer(
        status,
        "active_path_stamp_ns",
        f"{status_path}.active_path_stamp_ns",
        minimum=1,
    )
    if local_goal is not None and goal_id != local_goal:
        ctx.reject(
            "wrong_identity",
            f"{status_path}.goal_id",
            "supervisor goal_id 与本地 PCT goal stamp 不一致。",
            actual=goal_id,
        )
    if local_path is not None and active_path != local_path:
        ctx.reject(
            "wrong_identity",
            f"{status_path}.active_path_stamp_ns",
            "supervisor active Path stamp 与本地 Path 不一致。",
            actual=active_path,
        )
    _expect_true_field(ctx, status, "identity_valid", f"{status_path}.identity_valid")
    state = ctx.required_integer(status, "state", f"{status_path}.state", minimum=0)
    if state is not None and state not in {*range(7), 255}:
        ctx.reject(
            "unexpected_value",
            f"{status_path}.state",
            "NavigationStatus state 不在协议枚举中。",
            actual=state,
        )
    allow = ctx.required_boolean(
        status,
        "allow_tracking_command",
        f"{status_path}.allow_tracking_command",
    )
    force_zero = ctx.required_boolean(
        status,
        "force_zero_velocity",
        f"{status_path}.force_zero_velocity",
    )
    if state is not None and allow is not None and allow != (state == 3):
        ctx.reject(
            "invalid_status_contract",
            f"{status_path}.allow_tracking_command",
            "仅 TRACKING 状态可许可运动。",
            actual=allow,
        )
    if allow is not None and force_zero is not None and force_zero == allow:
        ctx.reject(
            "invalid_status_contract",
            f"{status_path}.force_zero_velocity",
            "force_zero_velocity 必须与运动许可互斥。",
            actual=force_zero,
        )
    stale_inputs = _validate_string_array(
        ctx,
        ctx.field(status, "stale_inputs", f"{status_path}.stale_inputs"),
        f"{status_path}.stale_inputs",
    )
    if state == 3 and stale_inputs:
        ctx.reject(
            "stale_tracking_input",
            f"{status_path}.stale_inputs",
            "TRACKING 许可不能携带 stale input。",
            actual=list(stale_inputs),
        )
    ctx.required_string(status, "reason", f"{status_path}.reason", nonempty=True)
    for key in (
        "stop_confirmed",
        "global_replan_requested",
        "global_replan_in_flight",
    ):
        ctx.required_boolean(status, key, f"{status_path}.{key}")
    for key in (
        "global_replan_request_id",
        "pct_plan_id",
        "consecutive_scan_failures",
    ):
        ctx.required_integer(status, key, f"{status_path}.{key}", minimum=0)
    return write_sequence, status_sequence, revision


def _validate_consumed_tracking_evidence(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    pct_goal_stamp_ns: int | None,
    path_stamp_ns: int | None,
) -> tuple[int | None, int | None, int | None]:
    evidence = ctx.mapping(value, path)
    write_sequence = ctx.required_integer(
        evidence,
        "write_sequence",
        f"{path}.write_sequence",
        minimum=1,
    )
    ctx.required_number(evidence, "timestamp", f"{path}.timestamp", minimum=0.0)
    _validate_vector(
        ctx,
        ctx.field(evidence, "written_command", f"{path}.written_command"),
        f"{path}.written_command",
        length=3,
    )
    gate_path = f"{path}.navigation_gate_diagnostics"
    gate = ctx.required_mapping(evidence, "navigation_gate_diagnostics", gate_path)
    _expect_exact_field(
        ctx,
        gate,
        "schema",
        "navigation_policy_gate_diagnostics_v1",
        f"{gate_path}.schema",
    )
    _expect_true_field(ctx, gate, "required", f"{gate_path}.required")
    timeout = ctx.required_number(gate, "timeout_s", f"{gate_path}.timeout_s", minimum=0.0)
    _expect_close(ctx, timeout, _EXPECTED_GATE_TIMEOUT_S, f"{gate_path}.timeout_s")
    _expect_exact_field(ctx, gate, "status_fault", None, f"{gate_path}.status_fault")
    _expect_true_field(ctx, gate, "permit_received", f"{gate_path}.permit_received")
    _expect_true_field(
        ctx,
        gate,
        "command_identity_matches_permit",
        f"{gate_path}.command_identity_matches_permit",
    )
    permit_path = f"{gate_path}.permit"
    permit = ctx.required_mapping(gate, "permit", permit_path)
    _expect_exact_field(ctx, permit, "state", 3, f"{permit_path}.state")
    _expect_true_field(
        ctx,
        permit,
        "allow_tracking_command",
        f"{permit_path}.allow_tracking_command",
    )
    _expect_false_field(
        ctx,
        permit,
        "force_zero_velocity",
        f"{permit_path}.force_zero_velocity",
    )
    _expect_true_field(ctx, permit, "identity_valid", f"{permit_path}.identity_valid")
    for key in ("header_stamp_ns", "status_sequence", "state_revision"):
        ctx.required_integer(permit, key, f"{permit_path}.{key}", minimum=1)
    ctx.required_number(permit, "received_at", f"{permit_path}.received_at", minimum=0.0)
    ctx.required_string(permit, "reason", f"{permit_path}.reason", nonempty=True)
    goal_id = ctx.required_integer(permit, "goal_id", f"{permit_path}.goal_id", minimum=1)
    active_path = ctx.required_integer(
        permit,
        "active_path_stamp_ns",
        f"{permit_path}.active_path_stamp_ns",
        minimum=1,
    )
    revision = permit.get("state_revision") if isinstance(permit.get("state_revision"), int) else None
    if pct_goal_stamp_ns is not None and goal_id != pct_goal_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{permit_path}.goal_id",
            "policy 消费许可的 goal_id 与 executor 不一致。",
            actual=goal_id,
        )
    if path_stamp_ns is not None and active_path != path_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{permit_path}.active_path_stamp_ns",
            "policy 消费许可的 Path stamp 与 executor 不一致。",
            actual=active_path,
        )
    command_identity = _validate_integer_vector(
        ctx,
        ctx.field(gate, "command_identity", f"{gate_path}.command_identity"),
        f"{gate_path}.command_identity",
        length=3,
        minimum=1,
    )
    expected_identity = (
        goal_id,
        active_path,
        revision,
    )
    if command_identity is not None and None not in expected_identity:
        if command_identity != expected_identity:
            ctx.reject(
                "wrong_identity",
                f"{gate_path}.command_identity",
                "policy command identity 必须严格绑定 goal/path/state revision。",
                actual=list(command_identity),
            )
    status_sequence = permit.get("status_sequence")
    return (
        write_sequence,
        status_sequence if isinstance(status_sequence, int) else None,
        revision,
    )


def _validate_executor_common(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int | None, int | None]:
    path = "$.latest_executor_status"
    executor = ctx.required_mapping(summary, "latest_executor_status", path)
    exact_fields = {
        "backend": "scan_ros2_goal_event",
        "phase": "completed",
        "failure_reason": "",
        "live_reference_path_source": "ros2_nav_msgs_path",
    }
    for key, expected in exact_fields.items():
        _expect_exact_field(ctx, executor, key, expected, f"{path}.{key}")
    for key in (
        "done",
        "success",
        "fresh_false_seen",
        "execution_activity_seen",
        "goal_rising_edge_seen",
        "scan_controller_goal_reached_verified",
        "policy_zero_hold_verified",
        "live_reference_path_required",
        "live_reference_path_verified",
        "live_reference_path_goal_bound",
        "pct_goal_required",
        "pct_goal_local_publish_triggered",
        "pct_goal_acknowledged",
        "pct_goal_transport_acknowledged",
    ):
        _expect_true_field(ctx, executor, key, f"{path}.{key}")
    _expect_false_field(ctx, executor, "failed", f"{path}.failed")
    for key in ("generation", "tick_index"):
        _expect_positive_integer_field(ctx, executor, key, f"{path}.{key}")

    false_sequence = _expect_positive_integer_field(
        ctx,
        executor,
        "goal_false_sequence",
        f"{path}.goal_false_sequence",
    )
    true_sequence = _expect_positive_integer_field(
        ctx,
        executor,
        "goal_true_sequence",
        f"{path}.goal_true_sequence",
    )
    false_timestamp = ctx.required_number(
        executor,
        "goal_false_receipt_timestamp",
        f"{path}.goal_false_receipt_timestamp",
        minimum=0.0,
    )
    true_timestamp = ctx.required_number(
        executor,
        "goal_true_receipt_timestamp",
        f"{path}.goal_true_receipt_timestamp",
        minimum=0.0,
    )
    if (
        false_sequence is not None
        and true_sequence is not None
        and true_sequence <= false_sequence
    ):
        ctx.reject(
            "invalid_goal_sequence",
            f"{path}.goal_true_sequence",
            "goal true 必须晚于本代 fresh false。",
            actual=true_sequence,
        )
    if (
        false_timestamp is not None
        and true_timestamp is not None
        and true_timestamp <= false_timestamp
    ):
        ctx.reject(
            "invalid_goal_sequence",
            f"{path}.goal_true_receipt_timestamp",
            "goal true 接收时间必须晚于 fresh false。",
            actual=true_timestamp,
        )

    zero_streak = _expect_positive_integer_field(
        ctx,
        executor,
        "zero_write_streak",
        f"{path}.zero_write_streak",
    )
    required_zero = _expect_positive_integer_field(
        ctx,
        executor,
        "required_zero_write_ticks",
        f"{path}.required_zero_write_ticks",
    )
    if (
        zero_streak is not None
        and required_zero is not None
        and zero_streak < required_zero
    ):
        ctx.reject(
            "insufficient_zero_hold",
            f"{path}.zero_write_streak",
            "目标完成后的连续零写次数不足。",
            actual=zero_streak,
        )
    last_write_sequence = _expect_positive_integer_field(
        ctx,
        executor,
        "last_policy_write_sequence",
        f"{path}.last_policy_write_sequence",
    )
    last_write_timestamp = ctx.required_number(
        executor,
        "last_policy_write_timestamp",
        f"{path}.last_policy_write_timestamp",
        minimum=0.0,
    )
    if (
        last_write_timestamp is not None
        and true_timestamp is not None
        and last_write_timestamp < true_timestamp
    ):
        ctx.reject(
            "invalid_goal_sequence",
            f"{path}.last_policy_write_timestamp",
            "最终零速写必须发生在 goal true 之后。",
            actual=last_write_timestamp,
        )
    _validate_zero_command(
        ctx,
        ctx.field(executor, "last_requested_command", f"{path}.last_requested_command"),
        f"{path}.last_requested_command",
    )
    _validate_zero_command(
        ctx,
        ctx.field(executor, "last_written_command", f"{path}.last_written_command"),
        f"{path}.last_written_command",
    )
    post_goal_nonzero = _expect_nonnegative_integer_field(
        ctx,
        executor,
        "post_goal_nonzero_write_count",
        f"{path}.post_goal_nonzero_write_count",
    )
    if post_goal_nonzero not in (None, 0):
        ctx.reject(
            "post_goal_motion",
            f"{path}.post_goal_nonzero_write_count",
            "目标完成后出现非零 policy 写入。",
            actual=post_goal_nonzero,
        )
    for key in (
        "invalid_goal_sample_count",
        "premature_true_count",
        "goal_sequence_reset_count",
        "policy_write_sequence_reset_count",
        "invalid_policy_write_count",
        "invalid_progress_pose_count",
        "invalid_reference_path_report_count",
        "invalid_pct_goal_report_count",
    ):
        count = _expect_nonnegative_integer_field(ctx, executor, key, f"{path}.{key}")
        if count not in (None, 0):
            ctx.reject(
                "protocol_error_count",
                f"{path}.{key}",
                "成功摘要不允许该协议错误计数非零。",
                actual=count,
            )
    _expect_nonnegative_integer_field(
        ctx,
        executor,
        "goal_true_waiting_for_supervisor_ack_count",
        f"{path}.goal_true_waiting_for_supervisor_ack_count",
    )

    path_sequence = _expect_positive_integer_field(
        ctx,
        executor,
        "live_reference_path_sequence",
        f"{path}.live_reference_path_sequence",
    )
    path_stamp = _expect_positive_integer_field(
        ctx,
        executor,
        "live_reference_path_stamp_ns",
        f"{path}.live_reference_path_stamp_ns",
    )
    _validate_sha256(
        ctx,
        ctx.field(
            executor,
            "live_reference_path_points_sha256",
            f"{path}.live_reference_path_points_sha256",
        ),
        f"{path}.live_reference_path_points_sha256",
    )
    generation_count = _expect_positive_integer_field(
        ctx,
        executor,
        "live_reference_path_generation_count",
        f"{path}.live_reference_path_generation_count",
    )
    for key in (
        "live_reference_path_goal_xy_error_m",
        "live_reference_path_goal_z_error_m",
        "live_reference_path_goal_yaw_error_rad",
    ):
        error = ctx.required_number(executor, key, f"{path}.{key}", minimum=0.0)
        _expect_close(ctx, error, 0.0, f"{path}.{key}", tolerance=1.0e-6)
    pct_goal_stamp = _expect_positive_integer_field(
        ctx,
        executor,
        "pct_goal_stamp_ns",
        f"{path}.pct_goal_stamp_ns",
    )
    _expect_positive_integer_field(
        ctx,
        executor,
        "pct_goal_publish_sequence",
        f"{path}.pct_goal_publish_sequence",
    )
    _expect_positive_integer_field(
        ctx,
        executor,
        "pct_goal_request_action_count",
        f"{path}.pct_goal_request_action_count",
    )
    _expect_nonnegative_integer_field(
        ctx,
        executor,
        "pct_goal_transport_retry_count",
        f"{path}.pct_goal_transport_retry_count",
    )
    if (
        pct_goal_stamp is not None
        and path_stamp is not None
        and path_stamp <= pct_goal_stamp
    ):
        ctx.reject(
            "wrong_identity",
            f"{path}.live_reference_path_stamp_ns",
            "live Path 必须是当前 PCT goal 发布后的新一代 Path。",
            actual=path_stamp,
        )
    if path_sequence is not None and generation_count is not None and generation_count > path_sequence:
        ctx.reject(
            "invalid_generation",
            f"{path}.live_reference_path_generation_count",
            "Path generation count 不能大于接收 sequence。",
            actual=generation_count,
        )
    if last_write_sequence is None:
        # 保持变量参与严格读取；错误已由 required helper 记录。
        pass
    return executor, pct_goal_stamp, path_stamp


def _validate_policy_lifecycle(
    ctx: _ValidationContext,
    simulation: Mapping[str, Any],
    *,
    pct_goal_stamp_ns: int | None,
    path_stamp_ns: int | None,
) -> tuple[Mapping[str, Any], int | None, int | None, int | None]:
    path = "$.simulation_report.navigation_policy_gate_lifecycle_report"
    lifecycle = ctx.required_mapping(
        simulation,
        "navigation_policy_gate_lifecycle_report",
        path,
    )
    _expect_exact_field(
        ctx,
        lifecycle,
        "schema",
        "navigation_policy_gate_lifecycle_v1",
        f"{path}.schema",
    )
    policy_count = _expect_positive_integer_field(
        ctx,
        lifecycle,
        "policy_write_count",
        f"{path}.policy_write_count",
    )
    motion_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "motion_allowed_write_count",
        f"{path}.motion_allowed_write_count",
    )
    verified_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "identity_verified_tracking_write_count",
        f"{path}.identity_verified_tracking_write_count",
    )
    observed_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "observed_status_sequence_count",
        f"{path}.observed_status_sequence_count",
    )
    identity_observed_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "identity_valid_observed_status_count",
        f"{path}.identity_valid_observed_status_count",
    )
    forced_zero_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "forced_zero_write_count",
        f"{path}.forced_zero_write_count",
    )
    if observed_count == 0:
        ctx.reject(
            "missing_status_sequence",
            f"{path}.observed_status_sequence_count",
            "至少必须观测一条本代 supervisor 状态序列。",
            actual=observed_count,
        )
    if identity_observed_count == 0:
        ctx.reject(
            "missing_identity_evidence",
            f"{path}.identity_valid_observed_status_count",
            "至少必须观测一条 goal/path identity 有效的状态。",
            actual=identity_observed_count,
        )
    if policy_count is not None:
        for count, count_path in (
            (motion_count, f"{path}.motion_allowed_write_count"),
            (verified_count, f"{path}.identity_verified_tracking_write_count"),
            (forced_zero_count, f"{path}.forced_zero_write_count"),
        ):
            if count is not None and count > policy_count:
                ctx.reject(
                    "invalid_count",
                    count_path,
                    "子计数不能大于 policy_write_count。",
                    actual=count,
                )
    if motion_count is not None and verified_count is not None and verified_count > motion_count:
        ctx.reject(
            "invalid_count",
            f"{path}.identity_verified_tracking_write_count",
            "identity verified tracking 计数不能大于 motion allowed 计数。",
            actual=verified_count,
        )
    if observed_count is not None and identity_observed_count is not None and identity_observed_count > observed_count:
        ctx.reject(
            "invalid_count",
            f"{path}.identity_valid_observed_status_count",
            "identity-valid 状态数不能大于观测状态序列数。",
            actual=identity_observed_count,
        )
    last_write = _expect_positive_integer_field(
        ctx,
        lifecycle,
        "last_write_sequence",
        f"{path}.last_write_sequence",
    )
    if policy_count is not None and last_write is not None and policy_count != last_write:
        ctx.reject(
            "invalid_count",
            f"{path}.last_write_sequence",
            "单 episode policy write sequence 必须等于累计写次数。",
            actual=last_write,
        )
    last_observed = _expect_positive_integer_field(
        ctx,
        lifecycle,
        "last_observed_status_sequence",
        f"{path}.last_observed_status_sequence",
    )
    _validate_string_array(
        ctx,
        ctx.field(lifecycle, "last_stop_reasons", f"{path}.last_stop_reasons"),
        f"{path}.last_stop_reasons",
    )
    reason_counts = ctx.required_mapping(lifecycle, "stop_reason_counts", f"{path}.stop_reason_counts")
    for reason, count in reason_counts.items():
        if not isinstance(reason, str) or not reason:
            ctx.reject("invalid_type", f"{path}.stop_reason_counts", "stop reason 键必须非空。")
        ctx.integer(count, f"{path}.stop_reason_counts.{reason}", minimum=1)

    first_observed = _validate_observed_status_evidence(
        ctx,
        ctx.field(
            lifecycle,
            "first_identity_valid_observed_status",
            f"{path}.first_identity_valid_observed_status",
        ),
        f"{path}.first_identity_valid_observed_status",
        pct_goal_stamp_ns=pct_goal_stamp_ns,
        # 全局重规划前的第一条有效状态可以绑定旧 Path；内部 identity 仍会
        # 严格自洽，只有最终证据必须绑定 executor 的最终 Path。
        path_stamp_ns=None,
    )
    last_observed_evidence = _validate_observed_status_evidence(
        ctx,
        ctx.field(
            lifecycle,
            "last_identity_valid_observed_status",
            f"{path}.last_identity_valid_observed_status",
        ),
        f"{path}.last_identity_valid_observed_status",
        pct_goal_stamp_ns=pct_goal_stamp_ns,
        path_stamp_ns=path_stamp_ns,
    )
    first_write, first_sequence, first_revision = first_observed
    last_evidence_write, last_sequence, last_revision = last_observed_evidence
    if first_write is not None and last_evidence_write is not None and last_evidence_write < first_write:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_identity_valid_observed_status.write_sequence",
            "最后身份状态证据不能早于第一条。",
            actual=last_evidence_write,
        )
    if first_sequence is not None and last_sequence is not None and last_sequence < first_sequence:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_identity_valid_observed_status",
            "status sequence 发生回退。",
            actual=last_sequence,
        )
    if first_revision is not None and last_revision is not None and last_revision < first_revision:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_identity_valid_observed_status",
            "state revision 发生回退。",
            actual=last_revision,
        )
    if last_observed is not None and last_sequence is not None and last_observed != last_sequence:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_observed_status_sequence",
            "生命周期末序号与最后身份有效状态证据不一致。",
            actual=last_observed,
        )
    _validate_policy_replan_lifecycle(
        ctx,
        lifecycle,
        path=path,
        observed_count=observed_count,
        pct_goal_stamp_ns=pct_goal_stamp_ns,
        path_stamp_ns=path_stamp_ns,
    )
    return lifecycle, policy_count, motion_count, verified_count


def _validate_positive_integer_array(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> tuple[int, ...]:
    raw = ctx.sequence(value, path)
    parsed: list[int] = []
    for index, item in enumerate(raw):
        integer = ctx.integer(item, f"{path}[{index}]", minimum=1)
        if integer is not None:
            parsed.append(integer)
    if len(parsed) != len(set(parsed)):
        ctx.reject("duplicate_identity", path, "identity 数组不允许重复。", actual=parsed)
    return tuple(parsed)


def _validate_global_replan_evidence(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    pct_goal_stamp_ns: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """校验 replan 状态自身，不把旧 Path 误绑到最终 Path。"""

    evidence = ctx.mapping(value, path)
    write_sequence = ctx.required_integer(
        evidence,
        "write_sequence",
        f"{path}.write_sequence",
        minimum=1,
    )
    ctx.required_number(evidence, "timestamp", f"{path}.timestamp", minimum=0.0)
    report_path = f"{path}.navigation_status_observed_report"
    report = ctx.required_mapping(
        evidence,
        "navigation_status_observed_report",
        report_path,
    )
    _expect_exact_field(
        ctx,
        report,
        "schema",
        "navigation_status_observed_diagnostics_v1",
        f"{report_path}.schema",
    )
    _expect_exact_field(ctx, report, "topic", "/navigation/status", f"{report_path}.topic")
    _expect_exact_field(ctx, report, "status_error", None, f"{report_path}.status_error")
    local_goal = ctx.required_integer(
        report,
        "local_pct_goal_stamp_ns",
        f"{report_path}.local_pct_goal_stamp_ns",
        minimum=1,
    )
    ctx.required_integer(
        report,
        "local_active_path_stamp_ns",
        f"{report_path}.local_active_path_stamp_ns",
        minimum=0,
    )
    status_path = f"{report_path}.status"
    status = ctx.required_mapping(report, "status", status_path)
    sequence = ctx.required_integer(status, "status_sequence", f"{status_path}.status_sequence", minimum=1)
    ctx.required_integer(status, "state_revision", f"{status_path}.state_revision", minimum=1)
    goal_id = ctx.required_integer(status, "goal_id", f"{status_path}.goal_id", minimum=1)
    state = ctx.required_integer(status, "state", f"{status_path}.state", minimum=0)
    requested = ctx.required_boolean(
        status,
        "global_replan_requested",
        f"{status_path}.global_replan_requested",
    )
    in_flight = ctx.required_boolean(
        status,
        "global_replan_in_flight",
        f"{status_path}.global_replan_in_flight",
    )
    request_id = ctx.required_integer(
        status,
        "global_replan_request_id",
        f"{status_path}.global_replan_request_id",
        minimum=1,
    )
    plan_id = ctx.required_integer(status, "pct_plan_id", f"{status_path}.pct_plan_id", minimum=0)
    if state != 4 and requested is not True and in_flight is not True:
        ctx.reject(
            "invalid_global_replan_evidence",
            status_path,
            "global replan lifecycle evidence 必须处于 GLOBAL_REPLAN 或显式请求/in-flight。",
        )
    if local_goal is not None and goal_id != local_goal:
        ctx.reject("wrong_identity", f"{status_path}.goal_id", "replan status goal_id 与本地 goal 不一致。", actual=goal_id)
    if pct_goal_stamp_ns is not None and local_goal != pct_goal_stamp_ns:
        ctx.reject("wrong_identity", f"{report_path}.local_pct_goal_stamp_ns", "replan status 不属于本代 PCT goal。", actual=local_goal)
    ctx.required_boolean(status, "identity_valid", f"{status_path}.identity_valid")
    ctx.required_string(status, "reason", f"{status_path}.reason", nonempty=True)
    return write_sequence, sequence, request_id, plan_id


def _validate_policy_replan_lifecycle(
    ctx: _ValidationContext,
    lifecycle: Mapping[str, Any],
    *,
    path: str,
    observed_count: int | None,
    pct_goal_stamp_ns: int | None,
    path_stamp_ns: int | None,
) -> None:
    last_state = ctx.required_integer(
        lifecycle,
        "last_observed_state",
        f"{path}.last_observed_state",
        minimum=0,
    )
    if last_state is not None and last_state not in {*range(7), 255}:
        ctx.reject("unexpected_value", f"{path}.last_observed_state", "NavigationStatus state 非法。", actual=last_state)
    transition_count = ctx.required_integer(
        lifecycle,
        "observed_state_transition_count",
        f"{path}.observed_state_transition_count",
        minimum=0,
    )
    if observed_count is not None and transition_count is not None and transition_count > max(0, observed_count - 1):
        ctx.reject("invalid_count", f"{path}.observed_state_transition_count", "状态迁移数不能超过观测数减一。", actual=transition_count)
    state_total = _sum_count_mapping(
        ctx,
        ctx.field(lifecycle, "observed_state_counts", f"{path}.observed_state_counts"),
        f"{path}.observed_state_counts",
    )
    reason_total = _sum_count_mapping(
        ctx,
        ctx.field(lifecycle, "observed_reason_counts", f"{path}.observed_reason_counts"),
        f"{path}.observed_reason_counts",
    )
    if observed_count is not None:
        for total, count_path in (
            (state_total, f"{path}.observed_state_counts"),
            (reason_total, f"{path}.observed_reason_counts"),
        ):
            if total != observed_count:
                ctx.reject("invalid_count", count_path, "观测分类计数总和必须等于 observed status count。", actual=total)
    _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "maximum_consecutive_scan_failures",
        f"{path}.maximum_consecutive_scan_failures",
    )
    requested_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "global_replan_requested_status_count",
        f"{path}.global_replan_requested_status_count",
    )
    in_flight_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "global_replan_in_flight_status_count",
        f"{path}.global_replan_in_flight_status_count",
    )
    emergency_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "emergency_stop_observed_status_count",
        f"{path}.emergency_stop_observed_status_count",
    )
    goal_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "goal_reached_observed_status_count",
        f"{path}.goal_reached_observed_status_count",
    )
    if goal_count == 0:
        ctx.reject(
            "missing_status_sequence",
            f"{path}.goal_reached_observed_status_count",
            "成功摘要必须观测 supervisor GOAL_REACHED 状态。",
            actual=0,
        )
    if observed_count is not None:
        for count, count_path in (
            (requested_count, f"{path}.global_replan_requested_status_count"),
            (in_flight_count, f"{path}.global_replan_in_flight_status_count"),
            (emergency_count, f"{path}.emergency_stop_observed_status_count"),
            (goal_count, f"{path}.goal_reached_observed_status_count"),
        ):
            if count is not None and count > observed_count:
                ctx.reject("invalid_count", count_path, "状态子计数不能超过 observed status count。", actual=count)
    request_ids = _validate_positive_integer_array(
        ctx,
        ctx.field(lifecycle, "distinct_global_replan_request_ids", f"{path}.distinct_global_replan_request_ids"),
        f"{path}.distinct_global_replan_request_ids",
    )
    pct_plan_ids = _validate_positive_integer_array(
        ctx,
        ctx.field(lifecycle, "distinct_pct_plan_ids", f"{path}.distinct_pct_plan_ids"),
        f"{path}.distinct_pct_plan_ids",
    )
    pending = ctx.required_boolean(
        lifecycle,
        "global_replan_pending_recovery",
        f"{path}.global_replan_pending_recovery",
    )
    tracking_after = ctx.required_boolean(
        lifecycle,
        "tracking_after_global_replan_observed",
        f"{path}.tracking_after_global_replan_observed",
    )
    recovery_count = _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "global_replan_recovery_count",
        f"{path}.global_replan_recovery_count",
    )
    triggered = bool(
        requested_count
        or in_flight_count
        or request_ids
        or lifecycle.get("first_global_replan_status") is not None
        or lifecycle.get("last_global_replan_status") is not None
    )
    if triggered:
        first = _validate_global_replan_evidence(
            ctx,
            ctx.field(lifecycle, "first_global_replan_status", f"{path}.first_global_replan_status"),
            f"{path}.first_global_replan_status",
            pct_goal_stamp_ns=pct_goal_stamp_ns,
        )
        last = _validate_global_replan_evidence(
            ctx,
            ctx.field(lifecycle, "last_global_replan_status", f"{path}.last_global_replan_status"),
            f"{path}.last_global_replan_status",
            pct_goal_stamp_ns=pct_goal_stamp_ns,
        )
        if first[0] is not None and last[0] is not None and last[0] < first[0]:
            ctx.reject("invalid_status_sequence", f"{path}.last_global_replan_status", "replan write sequence 发生回退。")
    else:
        _expect_exact_field(ctx, lifecycle, "first_global_replan_status", None, f"{path}.first_global_replan_status")
        _expect_exact_field(ctx, lifecycle, "last_global_replan_status", None, f"{path}.last_global_replan_status")
        if request_ids:
            ctx.reject("invalid_global_replan_evidence", f"{path}.distinct_global_replan_request_ids", "无 replan 状态时不得记录 request ID。")
        if pending is not False or tracking_after is not False or recovery_count not in (None, 0):
            ctx.reject("invalid_global_replan_evidence", path, "无 replan 时不得伪造恢复证据。")
    last_observed = _validate_observed_status_evidence(
        ctx,
        ctx.field(lifecycle, "last_observed_status", f"{path}.last_observed_status"),
        f"{path}.last_observed_status",
        pct_goal_stamp_ns=pct_goal_stamp_ns,
        path_stamp_ns=path_stamp_ns,
    )
    nested_last_state = _extract_nested_integer(
        lifecycle,
        "last_observed_status",
        "navigation_status_observed_report",
        "status",
        "state",
    )
    if last_state is not None and nested_last_state != last_state:
        ctx.reject("invalid_status_sequence", f"{path}.last_observed_state", "last_observed_state 与最后状态证据不一致。", actual=last_state)
    if last_observed[1] is not None and last_observed[1] != lifecycle.get("last_observed_status_sequence"):
        ctx.reject("invalid_status_sequence", f"{path}.last_observed_status", "最后状态证据序号与 lifecycle 不一致。")
    # pct_plan_ids 在初始成功规划时就应至少包含一项；全局重规划的新增 plan
    # 数量由 dynamic_f1 的条件合同进一步校验。
    if not pct_plan_ids:
        ctx.reject("missing_identity_evidence", f"{path}.distinct_pct_plan_ids", "缺少本代 PCT plan identity。")


def _expected_scan_stair_freeze_profile_audit() -> dict[str, Any]:
    """用生产 profile 的严格 loader 重建当前运行审计合同。"""

    profile = load_scan_stair_freeze_profile(
        _SCAN_STAIR_FREEZE_PROFILE_PATH,
        expected_scene="multi_floor",
        expected_robot="go2_x5",
    )
    return profile.audit_report()


def _validate_scan_stair_freeze_profile_runtime(
    ctx: _ValidationContext,
    task_config: Mapping[str, Any],
) -> None:
    """要求 summary 携带且严格匹配当前生产冻结 profile 审计。"""

    path = "$.task_config.scan_stair_freeze_profile_runtime"
    raw_runtime = ctx.field(task_config, "scan_stair_freeze_profile_runtime", path)
    if raw_runtime is _MISSING:
        return
    if not isinstance(raw_runtime, Mapping):
        ctx.mapping(raw_runtime, path)
        return
    runtime = raw_runtime
    try:
        expected = _expected_scan_stair_freeze_profile_audit()
    except (OSError, ScanStairFreezeProfileError) as exc:
        ctx.reject(
            "invalid_scan_stair_freeze_production_profile",
            path,
            f"无法加载当前生产楼梯冻结 profile：{exc}",
        )
        return

    expected_keys = set(expected)
    runtime_keys = set(runtime)
    if runtime_keys != expected_keys:
        ctx.reject(
            "invalid_scan_stair_freeze_profile_audit_schema",
            path,
            "运行审计字段必须与当前生产 profile.audit_report() 完全一致。",
            actual={
                "missing_keys": sorted(
                    str(key) for key in expected_keys - runtime_keys
                ),
                "unexpected_keys": sorted(
                    str(key) for key in runtime_keys - expected_keys
                ),
            },
        )

    for key, expected_value in expected.items():
        if key not in runtime:
            continue
        actual_value = runtime[key]
        if (
            actual_value != expected_value
            or type(actual_value) is not type(expected_value)
        ):
            ctx.reject(
                "scan_stair_freeze_profile_audit_drift",
                f"{path}.{key}",
                "运行审计值必须严格等于当前生产楼梯冻结 profile。",
                actual=actual_value,
            )


def _validate_common(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
) -> _CommonEvidence:
    _expect_true_field(ctx, summary, "success", "$.success")
    _expect_exact_field(ctx, summary, "final_state", "done", "$.final_state")
    _expect_empty_string_field(ctx, summary, "failure_reason", "$.failure_reason")
    _expect_true_field(
        ctx,
        summary,
        "execution_provenance_verified",
        "$.execution_provenance_verified",
    )
    failure_metadata = ctx.required_mapping(summary, "failure_metadata", "$.failure_metadata")
    if failure_metadata:
        ctx.reject(
            "unexpected_failure_metadata",
            "$.failure_metadata",
            "成功摘要不允许携带失败元数据。",
            actual=dict(failure_metadata),
        )
    state_trace = _validate_string_array(
        ctx,
        ctx.field(summary, "state_trace", "$.state_trace"),
        "$.state_trace",
    )
    if not state_trace:
        ctx.reject("missing_state_sequence", "$.state_trace", "状态序列不能为空。")
    else:
        if state_trace[-1] != "done":
            ctx.reject(
                "missing_terminal_state",
                "$.state_trace",
                "状态序列必须以 done 结束。",
                actual=list(state_trace),
            )
        if "failed" in state_trace:
            ctx.reject(
                "unexpected_failure_state",
                "$.state_trace",
                "成功摘要的状态序列不能包含 failed。",
                actual=list(state_trace),
            )
        required_states = (
            ("plan_nav_to_place", "exec_nav_to_place")
            if ctx.mode == "crossfloor_carry"
            else ("plan_nav_to_pick", "exec_nav_to_pick")
        )
        for required_state in required_states:
            if required_state not in state_trace:
                ctx.reject(
                    "missing_state_sequence",
                    "$.state_trace",
                    f"状态序列缺少 {required_state}。",
                    actual=list(state_trace),
                )

    task_config = ctx.required_mapping(summary, "task_config", "$.task_config")
    _validate_scan_stair_freeze_profile_runtime(ctx, task_config)
    executor, pct_goal_stamp, path_stamp = _validate_executor_common(ctx, summary)
    simulation = ctx.required_mapping(summary, "simulation_report", "$.simulation_report")
    _validate_bridge_report(ctx, simulation)
    lifecycle, policy_count, _motion_count, _verified_count = _validate_policy_lifecycle(
        ctx,
        simulation,
        pct_goal_stamp_ns=pct_goal_stamp,
        path_stamp_ns=path_stamp,
    )
    return _CommonEvidence(
        task_config=task_config,
        executor=executor,
        simulation=simulation,
        lifecycle=lifecycle,
        pct_goal_stamp_ns=pct_goal_stamp,
        path_stamp_ns=path_stamp,
        policy_write_count=policy_count,
        first_observed_sequence=(
            _extract_nested_integer(
                lifecycle,
                "first_identity_valid_observed_status",
                "navigation_status_observed_report",
                "status",
                "status_sequence",
            )
        ),
        last_observed_sequence=(
            _extract_nested_integer(
                lifecycle,
                "last_identity_valid_observed_status",
                "navigation_status_observed_report",
                "status",
                "status_sequence",
            )
        ),
    )


def _extract_nested_integer(parent: Mapping[str, Any], *keys: str) -> int | None:
    value: object = parent
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_controller_identity(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    path_stamp_ns: int | None,
) -> tuple[int | None, int | None, int | None]:
    identity = ctx.mapping(value, path)
    reference_stamp = ctx.required_integer(
        identity,
        "reference_path_stamp_ns",
        f"{path}.reference_path_stamp_ns",
        minimum=1,
    )
    bspline_stamp = ctx.required_integer(
        identity,
        "bspline_header_stamp_ns",
        f"{path}.bspline_header_stamp_ns",
        minimum=1,
    )
    start_stamp = ctx.required_integer(
        identity,
        "start_time_ns",
        f"{path}.start_time_ns",
        minimum=1,
    )
    trajectory_id = ctx.required_integer(
        identity,
        "traj_id",
        f"{path}.traj_id",
        minimum=0,
    )
    if path_stamp_ns is not None and reference_stamp != path_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{path}.reference_path_stamp_ns",
            "controller trajectory 必须绑定本代 live Path stamp。",
            actual=reference_stamp,
        )
    for key in ("reference_path_stamp", "bspline_header_stamp", "start_time"):
        stamp = ctx.required_mapping(identity, key, f"{path}.{key}")
        seconds = ctx.required_integer(stamp, "sec", f"{path}.{key}.sec", minimum=0)
        nanoseconds = ctx.required_integer(
            stamp,
            "nanosec",
            f"{path}.{key}.nanosec",
            minimum=0,
        )
        if nanoseconds is not None and nanoseconds >= 1_000_000_000:
            ctx.reject(
                "invalid_timestamp",
                f"{path}.{key}.nanosec",
                "nanosec 必须小于 1e9。",
                actual=nanoseconds,
            )
        expected_ns = {
            "reference_path_stamp": reference_stamp,
            "bspline_header_stamp": bspline_stamp,
            "start_time": start_stamp,
        }[key]
        if seconds is not None and nanoseconds is not None and expected_ns is not None:
            reconstructed = seconds * 1_000_000_000 + nanoseconds
            if reconstructed != expected_ns:
                ctx.reject(
                    "invalid_timestamp",
                    f"{path}.{key}",
                    "sec/nanosec 与对应 *_ns 字段不一致。",
                    actual=dict(stamp),
                )
    return reference_stamp, bspline_stamp, trajectory_id


def _validate_controller_status(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    path_stamp_ns: int | None,
) -> tuple[int | None, int | None]:
    status = ctx.mapping(value, path)
    _expect_exact_field(
        ctx,
        status,
        "source",
        "ros2_scan_planner_msgs_controller_status",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        status,
        "topic",
        "/planning/controller_status",
        f"{path}.topic",
    )
    ctx.required_number(status, "receipt_timestamp", f"{path}.receipt_timestamp", minimum=0.0)
    ctx.required_integer(status, "rx_sequence", f"{path}.rx_sequence", minimum=1)
    sequence = ctx.required_integer(
        status,
        "status_sequence",
        f"{path}.status_sequence",
        minimum=1,
    )
    acceptance_sequence = ctx.required_integer(
        status,
        "acceptance_sequence",
        f"{path}.acceptance_sequence",
        minimum=0,
    )
    event = ctx.required_integer(status, "event", f"{path}.event", minimum=0)
    state = ctx.required_integer(status, "state", f"{path}.state", minimum=0)
    if event is not None and event not in range(6):
        ctx.reject("unexpected_value", f"{path}.event", "ControllerStatus event 非法。", actual=event)
    if state is not None and state not in {*range(13), 255}:
        ctx.reject("unexpected_value", f"{path}.state", "ControllerStatus state 非法。", actual=state)
    ctx.required_string(status, "reason", f"{path}.reason", nonempty=True)
    for key in ("accepted", "trajectory_valid", "is_final", "emergency_stop"):
        ctx.required_boolean(status, key, f"{path}.{key}")
    header = ctx.required_mapping(status, "header", f"{path}.header")
    _expect_exact_field(ctx, header, "frame_id", "world", f"{path}.header.frame_id")
    header_ns = ctx.required_integer(header, "stamp_ns", f"{path}.header.stamp_ns", minimum=1)
    header_stamp = ctx.required_mapping(header, "stamp", f"{path}.header.stamp")
    header_sec = ctx.required_integer(header_stamp, "sec", f"{path}.header.stamp.sec", minimum=0)
    header_nanosec = ctx.required_integer(
        header_stamp,
        "nanosec",
        f"{path}.header.stamp.nanosec",
        minimum=0,
    )
    if (
        header_ns is not None
        and header_sec is not None
        and header_nanosec is not None
        and header_sec * 1_000_000_000 + header_nanosec != header_ns
    ):
        ctx.reject("invalid_timestamp", f"{path}.header", "controller header 时间字段不一致。")
    _validate_controller_identity(
        ctx,
        ctx.field(status, "identity", f"{path}.identity"),
        f"{path}.identity",
        path_stamp_ns=path_stamp_ns,
    )
    candidate = ctx.field(status, "candidate", f"{path}.candidate")
    if candidate is not _MISSING and candidate is not None:
        _validate_controller_identity(
            ctx,
            candidate,
            f"{path}.candidate",
            # rejected candidate 可以携带旧/新 Path identity，不能冒充 active。
            path_stamp_ns=None,
        )
    return sequence, acceptance_sequence


def _sum_count_mapping(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> int:
    mapping = ctx.mapping(value, path)
    total = 0
    for key, raw_count in mapping.items():
        if not isinstance(key, str) or not key:
            ctx.reject("invalid_type", path, "计数键必须是非空字符串。")
        count = ctx.integer(raw_count, f"{path}.{key}", minimum=1)
        if count is not None:
            total += count
    return total


def _validate_controller_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    *,
    require_multiple_trajectories: bool,
) -> None:
    path = "$.simulation_report.scan_controller_status_lifecycle_report"
    lifecycle = ctx.required_mapping(
        evidence.simulation,
        "scan_controller_status_lifecycle_report",
        path,
    )
    _expect_exact_field(
        ctx,
        lifecycle,
        "schema",
        "scan_controller_status_lifecycle_v1",
        f"{path}.schema",
    )
    sample_count = _expect_positive_integer_field(ctx, lifecycle, "sample_count", f"{path}.sample_count")
    first_sequence = _expect_positive_integer_field(
        ctx,
        lifecycle,
        "first_status_sequence",
        f"{path}.first_status_sequence",
    )
    last_sequence = _expect_positive_integer_field(
        ctx,
        lifecycle,
        "last_status_sequence",
        f"{path}.last_status_sequence",
    )
    if first_sequence is not None and last_sequence is not None and last_sequence < first_sequence:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_status_sequence",
            "SCAN controller status sequence 发生回退。",
            actual=last_sequence,
        )
    _expect_nonnegative_integer_field(
        ctx,
        lifecycle,
        "maximum_acceptance_sequence",
        f"{path}.maximum_acceptance_sequence",
    )
    event_total = _sum_count_mapping(
        ctx,
        ctx.field(lifecycle, "event_counts", f"{path}.event_counts"),
        f"{path}.event_counts",
    )
    state_total = _sum_count_mapping(
        ctx,
        ctx.field(lifecycle, "state_counts", f"{path}.state_counts"),
        f"{path}.state_counts",
    )
    reason_total = _sum_count_mapping(
        ctx,
        ctx.field(lifecycle, "reason_counts", f"{path}.reason_counts"),
        f"{path}.reason_counts",
    )
    if sample_count is not None:
        for total, total_path in (
            (event_total, f"{path}.event_counts"),
            (state_total, f"{path}.state_counts"),
            (reason_total, f"{path}.reason_counts"),
        ):
            if total != sample_count:
                ctx.reject(
                    "invalid_count",
                    total_path,
                    "生命周期分类计数总和必须等于 sample_count。",
                    actual=total,
                )
    counts: dict[str, int | None] = {}
    for key in (
        "accepted_status_count",
        "trajectory_valid_status_count",
        "candidate_rejection_count",
        "emergency_stop_status_count",
        "tracking_status_count",
        "goal_reached_status_count",
        "distinct_accepted_trajectory_count",
        "trajectory_replacement_count",
        "emergency_stop_recovery_count",
    ):
        counts[key] = _expect_nonnegative_integer_field(ctx, lifecycle, key, f"{path}.{key}")
        if sample_count is not None and counts[key] is not None and key not in {
            "distinct_accepted_trajectory_count",
            "trajectory_replacement_count",
        } and counts[key] > sample_count:
            ctx.reject("invalid_count", f"{path}.{key}", "状态子计数不能超过 sample_count。", actual=counts[key])
    for required_key in (
        "accepted_status_count",
        "trajectory_valid_status_count",
        "goal_reached_status_count",
    ):
        if counts[required_key] == 0:
            ctx.reject(
                "missing_controller_evidence",
                f"{path}.{required_key}",
                "成功 live 摘要缺少 controller 接受/有效/到达证据。",
                actual=0,
            )
    distinct = counts["distinct_accepted_trajectory_count"]
    replacements = counts["trajectory_replacement_count"]
    if distinct is not None and replacements is not None and replacements != max(0, distinct - 1):
        ctx.reject(
            "invalid_count",
            f"{path}.trajectory_replacement_count",
            "轨迹替换数必须等于 distinct accepted trajectory 数减一。",
            actual=replacements,
        )
    identities = ctx.required_sequence(
        lifecycle,
        "accepted_trajectory_identities",
        f"{path}.accepted_trajectory_identities",
    )
    normalized_identities: set[tuple[int | None, int | None, int | None]] = set()
    full_accepted_identities: set[tuple[int, int, int, int]] = set()
    for index, identity in enumerate(identities):
        normalized_identities.add(
            _validate_controller_identity(
                ctx,
                identity,
                f"{path}.accepted_trajectory_identities[{index}]",
                # 数组可能跨越 global replan；最终 active status 另行绑定。
                path_stamp_ns=None,
            )
        )
        full_identity = _mapping_identity_tuple(identity)
        if full_identity is not None:
            full_accepted_identities.add(full_identity)
    if distinct is not None and len(identities) != distinct:
        ctx.reject(
            "invalid_count",
            f"{path}.accepted_trajectory_identities",
            "identity 数组长度必须等于 distinct accepted trajectory count。",
            actual=len(identities),
        )
    if len(normalized_identities) != len(identities):
        ctx.reject(
            "duplicate_identity",
            f"{path}.accepted_trajectory_identities",
            "accepted trajectory identity 必须唯一。",
        )
    accepted_status_reports = ctx.required_sequence(
        lifecycle,
        "accepted_status_reports",
        f"{path}.accepted_status_reports",
    )
    dropped_accepted_reports = ctx.required_integer(
        lifecycle,
        "dropped_accepted_status_report_count",
        f"{path}.dropped_accepted_status_report_count",
        minimum=0,
    )
    accepted_status_count = counts["accepted_status_count"]
    if accepted_status_count is not None:
        expected_retained = min(
            accepted_status_count,
            _DIAGNOSTIC_RING_CAPACITY,
        )
        expected_dropped = max(
            accepted_status_count - _DIAGNOSTIC_RING_CAPACITY,
            0,
        )
        if len(accepted_status_reports) != expected_retained:
            ctx.reject(
                "invalid_count",
                f"{path}.accepted_status_reports",
                "controller accepted status ring 必须保留最近至多 128 条。",
                actual=len(accepted_status_reports),
            )
        if dropped_accepted_reports != expected_dropped:
            ctx.reject(
                "invalid_count",
                f"{path}.dropped_accepted_status_report_count",
                "accepted status ring 丢弃计数与累计接受数不一致。",
                actual=dropped_accepted_reports,
            )
    previous_accepted_status_sequence: int | None = None
    for index, raw_status in enumerate(accepted_status_reports):
        status_path = f"{path}.accepted_status_reports[{index}]"
        status_sequence, _ = _validate_controller_status(
            ctx,
            raw_status,
            status_path,
            path_stamp_ns=None,
        )
        status = ctx.mapping(raw_status, status_path)
        if status.get("accepted") is not True:
            ctx.reject(
                "invalid_controller_acceptance",
                f"{status_path}.accepted",
                "accepted status ring 只能保存 accepted=true 的 ControllerStatus。",
                actual=status.get("accepted"),
            )
        if (
            status_sequence is not None
            and previous_accepted_status_sequence is not None
            and status_sequence <= previous_accepted_status_sequence
        ):
            ctx.reject(
                "invalid_status_sequence",
                f"{status_path}.status_sequence",
                "accepted status ring 必须按 status sequence 严格递增。",
                actual=status_sequence,
            )
        previous_accepted_status_sequence = status_sequence
        identity = _mapping_identity_tuple(status.get("identity"))
        if identity is not None and identity not in full_accepted_identities:
            ctx.reject(
                "wrong_identity",
                f"{status_path}.identity",
                "accepted status identity 未收录于 lifecycle identity 集。",
            )
    tracking_status_reports = ctx.required_sequence(
        lifecycle,
        "tracking_status_reports",
        f"{path}.tracking_status_reports",
    )
    dropped_tracking_reports = ctx.required_integer(
        lifecycle,
        "dropped_tracking_status_report_count",
        f"{path}.dropped_tracking_status_report_count",
        minimum=0,
    )
    tracking_status_count = counts["tracking_status_count"]
    if tracking_status_count is not None:
        expected_retained = min(
            tracking_status_count,
            _DIAGNOSTIC_RING_CAPACITY,
        )
        expected_dropped = max(
            tracking_status_count - _DIAGNOSTIC_RING_CAPACITY,
            0,
        )
        if len(tracking_status_reports) != expected_retained:
            ctx.reject(
                "invalid_count",
                f"{path}.tracking_status_reports",
                "controller TRACKING ring 必须保留最近至多 128 条。",
                actual=len(tracking_status_reports),
            )
        if dropped_tracking_reports != expected_dropped:
            ctx.reject(
                "invalid_count",
                f"{path}.dropped_tracking_status_report_count",
                "TRACKING ring 丢弃计数与累计 TRACKING 数不一致。",
                actual=dropped_tracking_reports,
            )
    previous_tracking_status_sequence: int | None = None
    for index, raw_status in enumerate(tracking_status_reports):
        status_path = f"{path}.tracking_status_reports[{index}]"
        status_sequence, _ = _validate_controller_status(
            ctx,
            raw_status,
            status_path,
            path_stamp_ns=None,
        )
        status = ctx.mapping(raw_status, status_path)
        if (
            status.get("state") != 10
            or status.get("trajectory_valid") is not True
        ):
            ctx.reject(
                "invalid_controller_tracking",
                status_path,
                "TRACKING ring 只能保存 state=TRACKING 且 trajectory_valid=true 的状态。",
            )
        if (
            status_sequence is not None
            and previous_tracking_status_sequence is not None
            and status_sequence <= previous_tracking_status_sequence
        ):
            ctx.reject(
                "invalid_status_sequence",
                f"{status_path}.status_sequence",
                "TRACKING ring 必须按 status sequence 严格递增。",
                actual=status_sequence,
            )
        previous_tracking_status_sequence = status_sequence
        identity = _mapping_identity_tuple(status.get("identity"))
        if identity is not None and identity not in full_accepted_identities:
            ctx.reject(
                "wrong_identity",
                f"{status_path}.identity",
                "TRACKING status identity 未收录于 accepted identity 集。",
            )
    first_accepted_sequence, _ = _validate_controller_status(
        ctx,
        ctx.field(lifecycle, "first_accepted_status", f"{path}.first_accepted_status"),
        f"{path}.first_accepted_status",
        path_stamp_ns=None,
    )
    last_accepted_sequence, _ = _validate_controller_status(
        ctx,
        ctx.field(lifecycle, "last_accepted_status", f"{path}.last_accepted_status"),
        f"{path}.last_accepted_status",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    last_status_sequence, _ = _validate_controller_status(
        ctx,
        ctx.field(lifecycle, "last_status", f"{path}.last_status"),
        f"{path}.last_status",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    if accepted_status_reports:
        if accepted_status_reports[-1] != lifecycle.get("last_accepted_status"):
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.last_accepted_status",
                "last_accepted_status 必须等于 accepted status ring 末事件。",
            )
        if (
            dropped_accepted_reports == 0
            and accepted_status_reports[0]
            != lifecycle.get("first_accepted_status")
        ):
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.first_accepted_status",
                "未发生 ring 淘汰时 first_accepted_status 必须等于 ring 首事件。",
            )
    if tracking_status_reports:
        if tracking_status_reports[-1] != lifecycle.get("last_tracking_status"):
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.last_tracking_status",
                "last_tracking_status 必须等于 TRACKING ring 末事件。",
            )
        if (
            dropped_tracking_reports == 0
            and tracking_status_reports[0]
            != lifecycle.get("first_tracking_status")
        ):
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.first_tracking_status",
                "未发生 ring 淘汰时 first_tracking_status 必须等于 ring 首事件。",
            )
    if (
        first_accepted_sequence is not None
        and last_accepted_sequence is not None
        and last_accepted_sequence < first_accepted_sequence
    ):
        ctx.reject("invalid_status_sequence", f"{path}.last_accepted_status", "accepted 状态序列发生回退。")
    if last_sequence is not None and last_status_sequence != last_sequence:
        ctx.reject(
            "invalid_status_sequence",
            f"{path}.last_status.status_sequence",
            "last_status 必须对应 lifecycle 最后序号。",
            actual=last_status_sequence,
        )
    tracking_after = ctx.required_boolean(
        lifecycle,
        "tracking_after_emergency_stop_observed",
        f"{path}.tracking_after_emergency_stop_observed",
    )
    pending = ctx.required_boolean(
        lifecycle,
        "emergency_stop_pending_recovery",
        f"{path}.emergency_stop_pending_recovery",
    )
    emergency_count = counts["emergency_stop_status_count"]
    recovery_count = counts["emergency_stop_recovery_count"]
    if emergency_count:
        if tracking_after is not True or not recovery_count or pending is not False:
            ctx.reject(
                "unrecovered_emergency_stop",
                path,
                "出现 emergency stop 时必须有后续 TRACKING 恢复且不能留下 pending recovery。",
            )
        _validate_controller_status(
            ctx,
            ctx.field(lifecycle, "first_emergency_stop_status", f"{path}.first_emergency_stop_status"),
            f"{path}.first_emergency_stop_status",
            path_stamp_ns=evidence.path_stamp_ns,
        )
        _validate_controller_status(
            ctx,
            ctx.field(lifecycle, "last_emergency_stop_status", f"{path}.last_emergency_stop_status"),
            f"{path}.last_emergency_stop_status",
            path_stamp_ns=evidence.path_stamp_ns,
        )
    else:
        _expect_exact_field(ctx, lifecycle, "first_emergency_stop_status", None, f"{path}.first_emergency_stop_status")
        _expect_exact_field(ctx, lifecycle, "last_emergency_stop_status", None, f"{path}.last_emergency_stop_status")
        if tracking_after is not False or recovery_count not in (None, 0) or pending is not False:
            ctx.reject("invalid_recovery_evidence", path, "无 emergency stop 时不得伪造恢复证据。")
    if require_multiple_trajectories and (distinct is None or distinct < 2):
        ctx.reject(
            "missing_dynamic_replan_evidence",
            f"{path}.distinct_accepted_trajectory_count",
            "dynamic_f1 必须结构化证明 SCAN 接受至少两代不同局部轨迹。",
            actual=distinct,
        )
    if require_multiple_trajectories:
        first_identity = (
            lifecycle.get("first_accepted_status", {}).get("identity")
            if isinstance(lifecycle.get("first_accepted_status"), Mapping)
            else None
        )
        last_accepted = lifecycle.get("last_accepted_status")
        last_identity = (
            last_accepted.get("identity")
            if isinstance(last_accepted, Mapping)
            else None
        )
        last_accepted_state = (
            last_accepted.get("state")
            if isinstance(last_accepted, Mapping)
            else None
        )
        if first_identity == last_identity:
            ctx.reject(
                "missing_dynamic_replan_evidence",
                f"{path}.last_accepted_status.identity",
                "dynamic_f1 最后一条 accepted trajectory identity 必须区别于第一条。",
            )
        if last_accepted_state not in (10, 12):
            ctx.reject(
                "missing_dynamic_recovery_evidence",
                f"{path}.last_accepted_status.state",
                "替换轨迹必须进入 TRACKING 或 GOAL_REACHED，不能只停留在接受事件。",
                actual=last_accepted_state,
            )


def _validate_no_dynamic_obstacles(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
) -> None:
    raw_dynamic = evidence.task_config.get("dynamic_obstacles", ())
    raw_sequence = ctx.sequence(raw_dynamic, "$.task_config.dynamic_obstacles")
    if raw_sequence:
        ctx.reject(
            "wrong_mode",
            "$.task_config.dynamic_obstacles",
            f"{ctx.mode} 模式不允许启用动态障碍。",
            actual=list(raw_sequence),
        )
    for key in (
        "dynamic_obstacle_configuration_report",
        "dynamic_obstacle_runtime_report",
        "dynamic_obstacle_lifecycle_report",
        "dynamic_obstacle_raw_cloud_lifecycle_report",
    ):
        if key not in evidence.simulation:
            continue
        report_path = f"$.simulation_report.{key}"
        report = ctx.mapping(evidence.simulation[key], report_path)
        _expect_false_field(ctx, report, "enabled", f"{report_path}.enabled")
    if (
        "dynamic_obstacle_raw_cloud_last_report" in evidence.simulation
        and evidence.simulation["dynamic_obstacle_raw_cloud_last_report"] is not None
    ):
        ctx.reject(
            "wrong_mode",
            "$.simulation_report.dynamic_obstacle_raw_cloud_last_report",
            "无动态障碍模式不得保留动态障碍点云命中报告。",
        )
    if "dynamic_obstacle_pose_write_count" in evidence.simulation:
        write_count = ctx.integer(
            evidence.simulation["dynamic_obstacle_pose_write_count"],
            "$.simulation_report.dynamic_obstacle_pose_write_count",
            minimum=0,
        )
        if write_count not in (None, 0):
            ctx.reject(
                "wrong_mode",
                "$.simulation_report.dynamic_obstacle_pose_write_count",
                "无动态障碍模式不得写入动态障碍位姿。",
                actual=write_count,
            )


def _validate_task_identity(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
    *,
    expected_task_id: int,
) -> None:
    summary_task_id = ctx.required_integer(summary, "task_id", "$.task_id", minimum=1)
    task_task_id = ctx.required_integer(
        evidence.task_config,
        "task_id",
        "$.task_config.task_id",
        minimum=1,
    )
    if summary_task_id != expected_task_id:
        ctx.reject(
            "wrong_mode",
            "$.task_id",
            f"{ctx.mode} 必须使用 task_id={expected_task_id} 的验收任务。",
            actual=summary_task_id,
        )
    if task_task_id != summary_task_id:
        ctx.reject(
            "wrong_identity",
            "$.task_config.task_id",
            "顶层与 task_config 的 task_id 不一致。",
            actual=task_task_id,
        )
    _expect_exact_field(
        ctx,
        evidence.task_config,
        "scene_profile",
        "multi_floor",
        "$.task_config.scene_profile",
    )


def _validate_top_provenance(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    expected: Mapping[str, bool],
) -> None:
    for key, expected_value in expected.items():
        if expected_value:
            _expect_true_field(ctx, summary, key, f"$.{key}")
        else:
            _expect_false_field(ctx, summary, key, f"$.{key}")


def _validate_tracking_writes(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    *,
    required: bool,
) -> None:
    path = "$.simulation_report.navigation_policy_gate_lifecycle_report"
    motion_count = ctx.required_integer(
        evidence.lifecycle,
        "motion_allowed_write_count",
        f"{path}.motion_allowed_write_count",
        minimum=0,
    )
    verified_count = ctx.required_integer(
        evidence.lifecycle,
        "identity_verified_tracking_write_count",
        f"{path}.identity_verified_tracking_write_count",
        minimum=0,
    )
    snapshot_count = ctx.required_integer(
        evidence.lifecycle,
        "identity_verified_tracking_snapshot_count",
        f"{path}.identity_verified_tracking_snapshot_count",
        minimum=0,
    )
    if required and (motion_count is None or motion_count < 1):
        ctx.reject(
            "missing_tracking_write",
            f"{path}.motion_allowed_write_count",
            "非冻结模式必须至少一次实际许可运动写入。",
            actual=motion_count,
        )
    if required and (verified_count is None or verified_count < 1):
        ctx.reject(
            "missing_identity_evidence",
            f"{path}.identity_verified_tracking_write_count",
            "非冻结模式必须至少一次消费本代 goal/path TRACKING 许可。",
            actual=verified_count,
        )
    tracking_write_reports = ctx.required_sequence(
        evidence.lifecycle,
        "identity_verified_tracking_write_reports",
        f"{path}.identity_verified_tracking_write_reports",
    )
    dropped_tracking_writes = ctx.required_integer(
        evidence.lifecycle,
        "dropped_identity_verified_tracking_write_report_count",
        (
            f"{path}."
            "dropped_identity_verified_tracking_write_report_count"
        ),
        minimum=0,
    )
    if (
        verified_count is not None
        and snapshot_count is not None
        and snapshot_count > verified_count
    ):
        ctx.reject(
            "invalid_count",
            f"{path}.identity_verified_tracking_snapshot_count",
            "不同 typed controller snapshot 数不能超过 identity-valid 写入总数。",
            actual=snapshot_count,
        )
    if snapshot_count is not None:
        expected_retained = min(
            snapshot_count,
            _DIAGNOSTIC_RING_CAPACITY,
        )
        expected_dropped = max(
            snapshot_count - _DIAGNOSTIC_RING_CAPACITY,
            0,
        )
        if len(tracking_write_reports) != expected_retained:
            ctx.reject(
                "invalid_count",
                f"{path}.identity_verified_tracking_write_reports",
                "policy identity-valid TRACKING ring 必须按不同 typed snapshot 保留最近至多 128 条。",
                actual=len(tracking_write_reports),
            )
        if dropped_tracking_writes != expected_dropped:
            ctx.reject(
                "invalid_count",
                (
                    f"{path}."
                    "dropped_identity_verified_tracking_write_report_count"
                ),
                "policy TRACKING write ring 丢弃计数与不同 typed snapshot 数不一致。",
                actual=dropped_tracking_writes,
            )
    if required and verified_count and snapshot_count == 0:
        ctx.reject(
            "missing_identity_evidence",
            f"{path}.identity_verified_tracking_snapshot_count",
            "已声明 identity-valid 写入时必须至少钉住一个 typed controller snapshot。",
            actual=snapshot_count,
        )
    previous_write_sequence: int | None = None
    retained_write_sequences: list[int] = []
    retained_snapshot_keys: set[tuple[int, int, int, int, int, int]] = set()
    for index, raw_write in enumerate(tracking_write_reports):
        write_path = f"{path}.identity_verified_tracking_write_reports[{index}]"
        parsed_write_sequence, _, _ = _validate_consumed_tracking_evidence(
            ctx,
            raw_write,
            write_path,
            pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
            # ring 可跨 global replan；最终命名快照另行绑定 active Path。
            path_stamp_ns=None,
        )
        if parsed_write_sequence is not None:
            retained_write_sequences.append(parsed_write_sequence)
        if (
            parsed_write_sequence is not None
            and previous_write_sequence is not None
            and parsed_write_sequence <= previous_write_sequence
        ):
            ctx.reject(
                "invalid_status_sequence",
                (
                    f"{path}.identity_verified_tracking_write_reports"
                    f"[{index}].write_sequence"
                ),
                "policy TRACKING write ring 必须按 write sequence 严格递增。",
                actual=parsed_write_sequence,
            )
        previous_write_sequence = parsed_write_sequence
        write = ctx.mapping(raw_write, write_path)
        snapshot = ctx.required_mapping(
            write,
            "scan_controller_status_snapshot",
            f"{write_path}.scan_controller_status_snapshot",
        )
        snapshot_status_sequence, _ = _validate_controller_status(
            ctx,
            snapshot,
            f"{write_path}.scan_controller_status_snapshot",
            path_stamp_ns=None,
        )
        snapshot_identity = _mapping_identity_tuple(snapshot.get("identity"))
        snapshot_state = snapshot.get("state")
        snapshot_key = (
            (*snapshot_identity, snapshot_status_sequence, snapshot_state)
            if snapshot_identity is not None
            and snapshot_status_sequence is not None
            and isinstance(snapshot_state, int)
            and not isinstance(snapshot_state, bool)
            else None
        )
        if snapshot_key is not None:
            if snapshot_key in retained_snapshot_keys:
                ctx.reject(
                    "duplicate_identity",
                    f"{write_path}.scan_controller_status_snapshot",
                    "policy TRACKING ring 每个 typed controller snapshot 只能钉住一次。",
                )
            retained_snapshot_keys.add(snapshot_key)
    first_key = "first_identity_verified_tracking_write"
    last_key = "last_identity_verified_tracking_write"
    if not verified_count:
        _expect_exact_field(ctx, evidence.lifecycle, first_key, None, f"{path}.{first_key}")
        _expect_exact_field(ctx, evidence.lifecycle, last_key, None, f"{path}.{last_key}")
        return
    first = _validate_consumed_tracking_evidence(
        ctx,
        ctx.field(evidence.lifecycle, first_key, f"{path}.{first_key}"),
        f"{path}.{first_key}",
        pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
        # 第一条 tracking 许可允许来自 global replan 前一代 Path。
        path_stamp_ns=None,
    )
    last = _validate_consumed_tracking_evidence(
        ctx,
        ctx.field(evidence.lifecycle, last_key, f"{path}.{last_key}"),
        f"{path}.{last_key}",
        pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
        path_stamp_ns=evidence.path_stamp_ns,
    )
    for first_value, last_value, label in zip(
        first,
        last,
        ("write_sequence", "status_sequence", "state_revision"),
        strict=True,
    ):
        if first_value is not None and last_value is not None and last_value < first_value:
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.{last_key}.{label}",
                f"最后 tracking evidence 的 {label} 发生回退。",
                actual=last_value,
            )
    first_write_sequence = first[0]
    last_write_sequence = last[0]
    if (
        first_write_sequence is not None
        and last_write_sequence is not None
        and any(
            sequence < first_write_sequence or sequence > last_write_sequence
            for sequence in retained_write_sequences
        )
    ):
            ctx.reject(
                "invalid_status_sequence",
                f"{path}.identity_verified_tracking_write_reports",
                "typed snapshot 子集的 write sequence 必须位于全量 first/last tracking 边界内。",
                actual=retained_write_sequences,
            )


def _validate_stair_barrier_lifecycle_binding(
    ctx: _ValidationContext,
    *,
    lifecycle: Mapping[str, Any],
    barrier_path: str,
    report: Mapping[str, Any],
    status: Mapping[str, Any],
    write_sequence: int | None,
    write_timestamp: float | None,
    activation_timestamp: float | None,
    control_dt_s: float,
) -> None:
    """把屏障 ACK 绑定到同一次 policy lifecycle，而非孤立 JSON。"""

    lifecycle_path = "$.simulation_report.navigation_policy_gate_lifecycle_report"
    policy_count = ctx.required_integer(
        lifecycle,
        "policy_write_count",
        f"{lifecycle_path}.policy_write_count",
        minimum=1,
    )
    last_write = ctx.required_integer(
        lifecycle,
        "last_write_sequence",
        f"{lifecycle_path}.last_write_sequence",
        minimum=1,
    )
    if write_sequence is not None:
        for upper_bound, upper_path in (
            (policy_count, f"{lifecycle_path}.policy_write_count"),
            (last_write, f"{lifecycle_path}.last_write_sequence"),
        ):
            if upper_bound is not None and write_sequence > upper_bound:
                ctx.reject(
                    "invalid_sensor_acquisition_order",
                    f"{barrier_path}.write_sequence",
                    "屏障写入序号不能超出 lifecycle 已实际记录的 policy 写入。",
                    actual=write_sequence,
                )
    forced_zero_count = ctx.required_integer(
        lifecycle,
        "forced_zero_write_count",
        f"{lifecycle_path}.forced_zero_write_count",
        minimum=1,
    )
    if forced_zero_count == 0:
        ctx.reject(
            "missing_stair_freeze_evidence",
            f"{lifecycle_path}.forced_zero_write_count",
            "楼梯屏障必须出现在 policy 强制零写生命周期中。",
            actual=0,
        )
    stop_counts = ctx.required_mapping(
        lifecycle,
        "stop_reason_counts",
        f"{lifecycle_path}.stop_reason_counts",
    )
    freeze_stop_count = ctx.required_integer(
        stop_counts,
        "scan_stair_freeze",
        f"{lifecycle_path}.stop_reason_counts.scan_stair_freeze",
        minimum=1,
    )
    if freeze_stop_count == 0:
        ctx.reject(
            "missing_stair_freeze_evidence",
            f"{lifecycle_path}.stop_reason_counts.scan_stair_freeze",
            "lifecycle 未记录屏障所依赖的 scan_stair_freeze 零写。",
            actual=0,
        )

    emergency_count = ctx.required_integer(
        lifecycle,
        "emergency_stop_observed_status_count",
        f"{lifecycle_path}.emergency_stop_observed_status_count",
        minimum=1,
    )
    if emergency_count == 0:
        ctx.reject(
            "missing_stair_freeze_evidence",
            f"{lifecycle_path}.emergency_stop_observed_status_count",
            "lifecycle 必须实际消费至少一条合法楼梯 stop advisory。",
            actual=0,
        )
    state_counts = ctx.required_mapping(
        lifecycle,
        "observed_state_counts",
        f"{lifecycle_path}.observed_state_counts",
    )
    ctx.required_integer(
        state_counts,
        "emergency_stop",
        f"{lifecycle_path}.observed_state_counts.emergency_stop",
        minimum=1,
    )
    reason_counts = ctx.required_mapping(
        lifecycle,
        "observed_reason_counts",
        f"{lifecycle_path}.observed_reason_counts",
    )
    ctx.required_integer(
        reason_counts,
        _NAVIGATION_STAIR_INHIBIT_REASON,
        f"{lifecycle_path}.observed_reason_counts.{_NAVIGATION_STAIR_INHIBIT_REASON}",
        minimum=1,
    )

    plan_ids = _validate_positive_integer_array(
        ctx,
        ctx.field(
            lifecycle,
            "distinct_pct_plan_ids",
            f"{lifecycle_path}.distinct_pct_plan_ids",
        ),
        f"{lifecycle_path}.distinct_pct_plan_ids",
    )
    barrier_plan_id = ctx.required_integer(
        status,
        "pct_plan_id",
        f"{barrier_path}.navigation_status_observed_report.status.pct_plan_id",
        minimum=1,
    )
    if barrier_plan_id is not None and barrier_plan_id not in plan_ids:
        ctx.reject(
            "wrong_identity",
            f"{barrier_path}.navigation_status_observed_report.status.pct_plan_id",
            "屏障 PCT plan identity 未出现在 policy lifecycle 中。",
            actual=barrier_plan_id,
        )

    first_evidence = ctx.required_mapping(
        lifecycle,
        "first_identity_valid_observed_status",
        f"{lifecycle_path}.first_identity_valid_observed_status",
    )
    last_evidence = ctx.required_mapping(
        lifecycle,
        "last_identity_valid_observed_status",
        f"{lifecycle_path}.last_identity_valid_observed_status",
    )
    boundary_statuses: list[
        tuple[str, Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for label, evidence_value in (
        ("first_identity_valid_observed_status", first_evidence),
        ("last_identity_valid_observed_status", last_evidence),
    ):
        boundary_report = ctx.required_mapping(
            evidence_value,
            "navigation_status_observed_report",
            f"{lifecycle_path}.{label}.navigation_status_observed_report",
        )
        boundary_status = ctx.required_mapping(
            boundary_report,
            "status",
            f"{lifecycle_path}.{label}.navigation_status_observed_report.status",
        )
        boundary_statuses.append((label, evidence_value, boundary_status))

    barrier_status_sequence = ctx.required_integer(
        status,
        "status_sequence",
        f"{barrier_path}.navigation_status_observed_report.status.status_sequence",
        minimum=1,
    )
    barrier_revision = ctx.required_integer(
        status,
        "state_revision",
        f"{barrier_path}.navigation_status_observed_report.status.state_revision",
        minimum=0,
    )
    barrier_rx_sequence = ctx.required_integer(
        status,
        "rx_sequence",
        f"{barrier_path}.navigation_status_observed_report.status.rx_sequence",
        minimum=1,
    )
    for field_name, barrier_value in (
        ("status_sequence", barrier_status_sequence),
        ("state_revision", barrier_revision),
        ("rx_sequence", barrier_rx_sequence),
    ):
        boundary_values = [
            ctx.required_integer(
                boundary_status,
                field_name,
                f"{lifecycle_path}.{label}.navigation_status_observed_report.status.{field_name}",
                minimum=0 if field_name == "state_revision" else 1,
            )
            for label, _, boundary_status in boundary_statuses
        ]
        if (
            barrier_value is not None
            and all(value is not None for value in boundary_values)
            and not boundary_values[0] <= barrier_value <= boundary_values[1]
        ):
            ctx.reject(
                "invalid_status_sequence",
                f"{barrier_path}.navigation_status_observed_report.status.{field_name}",
                "屏障状态身份必须落在 lifecycle 首尾身份有效证据之间。",
                actual=barrier_value,
            )

    for label, boundary_evidence, boundary_status in boundary_statuses:
        boundary_sequence = boundary_status.get("status_sequence")
        if barrier_status_sequence == boundary_sequence:
            expected_boundary = {
                "write_sequence": write_sequence,
                "timestamp": write_timestamp,
                "navigation_status_observed_report": report,
            }
            if boundary_evidence != expected_boundary:
                ctx.reject(
                    "wrong_typed_evidence_reference",
                    f"{lifecycle_path}.{label}",
                    "相同 status sequence 的 lifecycle 边界必须与屏障 ACK 完全一致。",
                )

    header_stamp_ns = ctx.required_integer(
        status,
        "header_stamp_ns",
        f"{barrier_path}.navigation_status_observed_report.status.header_stamp_ns",
        minimum=1,
    )
    if header_stamp_ns is not None and activation_timestamp is not None:
        minimum_header_ns = int(
            math.floor(
                max(0.0, activation_timestamp - control_dt_s)
                * 1_000_000_000
            )
        )
        if header_stamp_ns < minimum_header_ns:
            ctx.reject(
                "invalid_sensor_acquisition_order",
                f"{barrier_path}.navigation_status_observed_report.status.header_stamp_ns",
                "屏障 ACK Header 不能早于冻结激活窗口。",
                actual=header_stamp_ns,
            )
    if header_stamp_ns is not None and write_timestamp is not None:
        maximum_header_ns = int(
            math.floor((write_timestamp + 1.0e-9) * 1_000_000_000)
        )
        if header_stamp_ns > maximum_header_ns:
            ctx.reject(
                "sensor_acquisition_status_from_future",
                f"{barrier_path}.navigation_status_observed_report.status.header_stamp_ns",
                "屏障 ACK Header 不能晚于消费它的 policy 写入。",
                actual=header_stamp_ns,
            )


def _validate_stair_sensor_acquisition_barrier(
    ctx: _ValidationContext,
    stair: Mapping[str, Any],
    *,
    stair_path: str,
    evidence: _CommonEvidence,
) -> None:
    """证明本代 Path 的传感器屏障先通过，随后才允许 root 前进。"""

    barrier_path = f"{stair_path}.sensor_acquisition_barrier"
    barrier = ctx.required_mapping(
        stair,
        "sensor_acquisition_barrier",
        barrier_path,
    )
    for key in ("required", "passed", "local_sensors_fresh", "supervisor_sensors_fresh"):
        _expect_true_field(ctx, barrier, key, f"{barrier_path}.{key}")
    _expect_false_field(ctx, barrier, "pending", f"{barrier_path}.pending")

    path_stamp_ns = ctx.required_integer(
        barrier,
        "path_stamp_ns",
        f"{barrier_path}.path_stamp_ns",
        minimum=1,
    )
    if evidence.path_stamp_ns is not None and path_stamp_ns != evidence.path_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{barrier_path}.path_stamp_ns",
            "传感器屏障必须绑定 executor 当前 live Path。",
            actual=path_stamp_ns,
        )

    try:
        production_config = load_scan_stair_freeze_profile(
            _SCAN_STAIR_FREEZE_PROFILE_PATH,
            expected_scene="multi_floor",
            expected_robot="go2_x5",
        ).config
    except (OSError, ScanStairFreezeProfileError) as exc:
        ctx.reject(
            "invalid_scan_stair_freeze_production_profile",
            barrier_path,
            f"无法读取传感器屏障生产参数：{exc}",
        )
        production_config = None

    activation_timestamp = ctx.required_number(
        barrier,
        "activation_timestamp",
        f"{barrier_path}.activation_timestamp",
        minimum=0.0,
    )
    write_sequence = ctx.required_integer(
        barrier,
        "write_sequence",
        f"{barrier_path}.write_sequence",
        minimum=1,
    )
    write_timestamp = ctx.required_number(
        barrier,
        "write_timestamp",
        f"{barrier_path}.write_timestamp",
        minimum=0.0,
    )
    timeout_s = ctx.required_number(
        barrier,
        "timeout_s",
        f"{barrier_path}.timeout_s",
        minimum=0.0,
    )
    status_timeout_s = ctx.required_number(
        barrier,
        "status_freshness_timeout_s",
        f"{barrier_path}.status_freshness_timeout_s",
        minimum=0.0,
    )
    if production_config is not None:
        _expect_close(
            ctx,
            timeout_s,
            production_config.activation_timeout_s,
            f"{barrier_path}.timeout_s",
        )
        _expect_close(
            ctx,
            status_timeout_s,
            production_config.supervisor_sensor_status_timeout_s,
            f"{barrier_path}.status_freshness_timeout_s",
        )
    if (
        activation_timestamp is not None
        and write_timestamp is not None
        and write_timestamp <= activation_timestamp
    ):
        ctx.reject(
            "invalid_sensor_acquisition_order",
            f"{barrier_path}.write_timestamp",
            "屏障通过写入必须严格晚于楼梯冻结激活。",
            actual=write_timestamp,
        )

    progress_at_pass = ctx.required_number(
        barrier,
        "progress_m_at_pass",
        f"{barrier_path}.progress_m_at_pass",
        minimum=0.0,
    )
    _expect_close(
        ctx,
        progress_at_pass,
        0.0,
        f"{barrier_path}.progress_m_at_pass",
        tolerance=1.0e-12,
    )
    pending_reasons = _validate_string_array(
        ctx,
        ctx.field(barrier, "pending_reasons", f"{barrier_path}.pending_reasons"),
        f"{barrier_path}.pending_reasons",
    )
    if pending_reasons:
        ctx.reject(
            "sensor_acquisition_not_complete",
            f"{barrier_path}.pending_reasons",
            "已通过的传感器屏障不能保留 pending 原因。",
            actual=list(pending_reasons),
        )

    report = ctx.required_mapping(
        barrier,
        "navigation_status_observed_report",
        f"{barrier_path}.navigation_status_observed_report",
    )
    _validate_observed_status_evidence(
        ctx,
        {
            "write_sequence": write_sequence,
            "timestamp": write_timestamp,
            "navigation_status_observed_report": report,
        },
        barrier_path,
        pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
        path_stamp_ns=evidence.path_stamp_ns,
    )
    status = report.get("status") if isinstance(report, Mapping) else None
    receipt_timestamp: float | None = None
    if isinstance(status, Mapping):
        state = ctx.required_integer(
            status,
            "state",
            f"{barrier_path}.navigation_status_observed_report.status.state",
            minimum=0,
        )
        if state != _NAVIGATION_STATE_EMERGENCY_STOP:
            ctx.reject(
                "invalid_stair_freeze_acknowledgement",
                f"{barrier_path}.navigation_status_observed_report.status.state",
                "屏障只能消费 supervisor 对合法楼梯冻结的专用 stop advisory。",
                actual=state,
            )
        _expect_exact_field(
            ctx,
            status,
            "reason",
            _NAVIGATION_STAIR_INHIBIT_REASON,
            f"{barrier_path}.navigation_status_observed_report.status.reason",
        )
        _expect_false_field(
            ctx,
            status,
            "allow_tracking_command",
            f"{barrier_path}.navigation_status_observed_report.status.allow_tracking_command",
        )
        for key in ("force_zero_velocity", "stop_confirmed"):
            _expect_true_field(
                ctx,
                status,
                key,
                f"{barrier_path}.navigation_status_observed_report.status.{key}",
            )
        for key in ("global_replan_requested", "global_replan_in_flight"):
            _expect_false_field(
                ctx,
                status,
                key,
                f"{barrier_path}.navigation_status_observed_report.status.{key}",
            )
        replan_request_id = ctx.required_integer(
            status,
            "global_replan_request_id",
            f"{barrier_path}.navigation_status_observed_report.status.global_replan_request_id",
            minimum=0,
        )
        if replan_request_id != 0:
            ctx.reject(
                "invalid_stair_freeze_acknowledgement",
                f"{barrier_path}.navigation_status_observed_report.status.global_replan_request_id",
                "静态楼梯首代 Path 的冻结 ACK 不能携带重规划 request identity。",
                actual=replan_request_id,
            )
        pct_plan_id = ctx.required_integer(
            status,
            "pct_plan_id",
            f"{barrier_path}.navigation_status_observed_report.status.pct_plan_id",
            minimum=1,
        )
        if pct_plan_id is not None and pct_plan_id < 1:
            ctx.reject(
                "invalid_stair_freeze_acknowledgement",
                f"{barrier_path}.navigation_status_observed_report.status.pct_plan_id",
                "冻结 ACK 必须绑定已接受的非零 PCT plan identity。",
                actual=pct_plan_id,
            )
        scan_failures = ctx.required_integer(
            status,
            "consecutive_scan_failures",
            f"{barrier_path}.navigation_status_observed_report.status.consecutive_scan_failures",
            minimum=0,
        )
        if scan_failures != 0:
            ctx.reject(
                "invalid_stair_freeze_acknowledgement",
                f"{barrier_path}.navigation_status_observed_report.status.consecutive_scan_failures",
                "合法楼梯冻结 ACK 不能夹带 SCAN 连续失败。",
                actual=scan_failures,
            )
        stale_inputs = _validate_string_array(
            ctx,
            ctx.field(
                status,
                "stale_inputs",
                f"{barrier_path}.navigation_status_observed_report.status.stale_inputs",
            ),
            f"{barrier_path}.navigation_status_observed_report.status.stale_inputs",
        )
        if stale_inputs != ("bspline",):
            ctx.reject(
                "stale_sensor_acquisition_input",
                f"{barrier_path}.navigation_status_observed_report.status.stale_inputs",
                "静态楼梯冻结 ACK 必须仅保留因规划抑制而失鲜的 bspline。",
                actual=list(stale_inputs),
            )
        receipt_timestamp = ctx.required_number(
            status,
            "receipt_timestamp",
            f"{barrier_path}.navigation_status_observed_report.status.receipt_timestamp",
            minimum=0.0,
        )
    if (
        activation_timestamp is not None
        and receipt_timestamp is not None
        and receipt_timestamp <= activation_timestamp
    ):
        ctx.reject(
            "invalid_sensor_acquisition_order",
            f"{barrier_path}.navigation_status_observed_report.status.receipt_timestamp",
            "屏障必须消费冻结激活后收到的 supervisor 状态。",
            actual=receipt_timestamp,
        )
    if write_timestamp is not None and receipt_timestamp is not None:
        status_age = write_timestamp - receipt_timestamp
        if status_age < -1.0e-9:
            ctx.reject(
                "sensor_acquisition_status_from_future",
                f"{barrier_path}.navigation_status_observed_report.status.receipt_timestamp",
                "supervisor 状态接收时间不能晚于 policy 写入时间。",
                actual=receipt_timestamp,
            )
        elif status_timeout_s is not None and status_age > status_timeout_s + 1.0e-9:
            ctx.reject(
                "sensor_acquisition_status_timeout",
                f"{barrier_path}.navigation_status_observed_report.status.receipt_timestamp",
                "屏障使用的 supervisor 状态已超过生产新鲜度时限。",
                actual=status_age,
            )

    policy_path = f"{barrier_path}.policy_write_report"
    policy = ctx.required_mapping(barrier, "policy_write_report", policy_path)
    policy_sequence = ctx.required_integer(
        policy,
        "write_sequence",
        f"{policy_path}.write_sequence",
        minimum=1,
    )
    policy_timestamp = ctx.required_number(
        policy,
        "timestamp",
        f"{policy_path}.timestamp",
        minimum=0.0,
    )
    if policy_sequence != write_sequence:
        ctx.reject(
            "wrong_identity",
            f"{policy_path}.write_sequence",
            "屏障 policy 写入序号与通过证据不一致。",
            actual=policy_sequence,
        )
    if write_timestamp is not None:
        _expect_close(
            ctx,
            policy_timestamp,
            write_timestamp,
            f"{policy_path}.timestamp",
            tolerance=1.0e-12,
        )
    _expect_exact_field(ctx, policy, "owner_id", "scan_cmd_vel", f"{policy_path}.owner_id")
    _expect_false_field(ctx, policy, "motion_allowed", f"{policy_path}.motion_allowed")
    _expect_true_field(
        ctx,
        policy,
        "navigation_cmd_vel_inhibited",
        f"{policy_path}.navigation_cmd_vel_inhibited",
    )
    _expect_exact_field(
        ctx,
        policy,
        "navigation_cmd_vel_inhibit_reason",
        "scan_stair_freeze",
        f"{policy_path}.navigation_cmd_vel_inhibit_reason",
    )
    written = _validate_vector(
        ctx,
        ctx.field(policy, "written_command", f"{policy_path}.written_command"),
        f"{policy_path}.written_command",
        length=3,
    )
    if written is not None and written != _ZERO_COMMAND:
        ctx.reject(
            "nonzero_sensor_acquisition_write",
            f"{policy_path}.written_command",
            "传感器屏障期间唯一 policy owner 必须精确写零。",
            actual=list(written),
        )
    stop_reasons = _validate_string_array(
        ctx,
        ctx.field(policy, "stop_reasons", f"{policy_path}.stop_reasons"),
        f"{policy_path}.stop_reasons",
    )
    if stop_reasons != ("scan_stair_freeze",):
        ctx.reject(
            "invalid_sensor_acquisition_stop_context",
            f"{policy_path}.stop_reasons",
            "屏障通过写入只能保留 scan_stair_freeze，不能夹带传感器失鲜。",
            actual=list(stop_reasons),
        )
    if policy.get("navigation_status_observed_report") != report:
        ctx.reject(
            "wrong_typed_evidence_reference",
            f"{policy_path}.navigation_status_observed_report",
            "屏障必须保存同一次 policy 写入实际观察到的 supervisor 快照。",
        )
    last_sequence = ctx.required_integer(
        barrier,
        "last_write_sequence",
        f"{barrier_path}.last_write_sequence",
        minimum=1,
    )
    if last_sequence != write_sequence:
        ctx.reject(
            "wrong_identity",
            f"{barrier_path}.last_write_sequence",
            "屏障最后写入序号副本与通过证据不一致。",
            actual=last_sequence,
        )
    last_timestamp = ctx.required_number(
        barrier,
        "last_write_timestamp",
        f"{barrier_path}.last_write_timestamp",
        minimum=0.0,
    )
    if write_timestamp is not None:
        _expect_close(
            ctx,
            last_timestamp,
            write_timestamp,
            f"{barrier_path}.last_write_timestamp",
            tolerance=1.0e-12,
        )
    if barrier.get("last_navigation_status_observed_report") != report:
        ctx.reject(
            "wrong_typed_evidence_reference",
            f"{barrier_path}.last_navigation_status_observed_report",
            "屏障最后观察状态必须等于实际放行所消费的 supervisor ACK。",
        )
    if isinstance(status, Mapping):
        _validate_stair_barrier_lifecycle_binding(
            ctx,
            lifecycle=evidence.lifecycle,
            barrier_path=barrier_path,
            report=report,
            status=status,
            write_sequence=write_sequence,
            write_timestamp=write_timestamp,
            activation_timestamp=activation_timestamp,
            control_dt_s=(
                0.02
                if production_config is None
                else production_config.default_control_dt_s
            ),
        )

    _expect_true_field(
        ctx,
        stair,
        "sensor_acquisition_required",
        f"{stair_path}.sensor_acquisition_required",
    )
    _expect_true_field(
        ctx,
        stair,
        "sensor_acquisition_complete",
        f"{stair_path}.sensor_acquisition_complete",
    )
    _expect_false_field(
        ctx,
        stair,
        "sensor_acquisition_pending",
        f"{stair_path}.sensor_acquisition_pending",
    )
    started_timestamp_copy = ctx.required_number(
        stair,
        "sensor_acquisition_started_timestamp",
        f"{stair_path}.sensor_acquisition_started_timestamp",
        minimum=0.0,
    )
    if activation_timestamp is not None:
        _expect_close(
            ctx,
            started_timestamp_copy,
            activation_timestamp,
            f"{stair_path}.sensor_acquisition_started_timestamp",
            tolerance=1.0e-12,
        )
    completed_timestamp_copy = ctx.required_number(
        stair,
        "sensor_acquisition_completed_timestamp",
        f"{stair_path}.sensor_acquisition_completed_timestamp",
        minimum=0.0,
    )
    if write_timestamp is not None:
        _expect_close(
            ctx,
            completed_timestamp_copy,
            write_timestamp,
            f"{stair_path}.sensor_acquisition_completed_timestamp",
            tolerance=1.0e-12,
        )
    timeout_copy = ctx.required_number(
        stair,
        "sensor_acquisition_timeout_s",
        f"{stair_path}.sensor_acquisition_timeout_s",
        minimum=0.0,
    )
    if timeout_s is not None:
        _expect_close(
            ctx,
            timeout_copy,
            timeout_s,
            f"{stair_path}.sensor_acquisition_timeout_s",
        )
    sequence_floor = ctx.required_integer(
        stair,
        "sensor_acquisition_write_sequence_floor",
        f"{stair_path}.sensor_acquisition_write_sequence_floor",
        minimum=0,
    )
    if sequence_floor is not None and write_sequence is not None and write_sequence <= sequence_floor:
        ctx.reject(
            "invalid_sensor_acquisition_order",
            f"{stair_path}.sensor_acquisition_write_sequence",
            "屏障通过写入必须晚于冻结激活前的 policy 序号下界。",
            actual=write_sequence,
        )
    for key in ("sensor_acquisition_write_sequence", "sensor_acquisition_last_write_sequence"):
        value = ctx.required_integer(stair, key, f"{stair_path}.{key}", minimum=1)
        if value != write_sequence:
            ctx.reject(
                "wrong_identity",
                f"{stair_path}.{key}",
                "楼梯状态中的屏障写入序号副本不一致。",
                actual=value,
            )
    last_write_timestamp_copy = ctx.required_number(
        stair,
        "sensor_acquisition_last_write_timestamp",
        f"{stair_path}.sensor_acquisition_last_write_timestamp",
        minimum=0.0,
    )
    if write_timestamp is not None:
        _expect_close(
            ctx,
            last_write_timestamp_copy,
            write_timestamp,
            f"{stair_path}.sensor_acquisition_last_write_timestamp",
            tolerance=1.0e-12,
        )
    duplicate_pending = _validate_string_array(
        ctx,
        ctx.field(
            stair,
            "sensor_acquisition_pending_reasons",
            f"{stair_path}.sensor_acquisition_pending_reasons",
        ),
        f"{stair_path}.sensor_acquisition_pending_reasons",
    )
    if duplicate_pending:
        ctx.reject(
            "sensor_acquisition_not_complete",
            f"{stair_path}.sensor_acquisition_pending_reasons",
            "完成状态不能保留屏障等待原因。",
            actual=list(duplicate_pending),
        )
    sensor_faults = _validate_string_array(
        ctx,
        ctx.field(stair, "sensor_safety_fault_reasons", f"{stair_path}.sensor_safety_fault_reasons"),
        f"{stair_path}.sensor_safety_fault_reasons",
    )
    if sensor_faults:
        ctx.reject(
            "unexpected_sensor_safety_fault",
            f"{stair_path}.sensor_safety_fault_reasons",
            "成功楼梯验收不能携带传感器故障。",
            actual=list(sensor_faults),
        )
    _expect_exact_field(
        ctx,
        stair,
        "sensor_safety_fault_write_sequence",
        None,
        f"{stair_path}.sensor_safety_fault_write_sequence",
    )
    _expect_exact_field(
        ctx,
        stair,
        "sensor_safety_fault_timestamp",
        None,
        f"{stair_path}.sensor_safety_fault_timestamp",
    )
    for key in (
        "invalid_controller_status_count",
        "controller_status_sequence_reset_count",
    ):
        count = _expect_nonnegative_integer_field(
            ctx,
            stair,
            key,
            f"{stair_path}.{key}",
        )
        if count not in (None, 0):
            ctx.reject(
                "protocol_error_count",
                f"{stair_path}.{key}",
                "成功楼梯摘要不允许该控制器状态协议错误计数非零。",
                actual=count,
            )
    _expect_true_field(
        ctx,
        stair,
        "terminal_supervisor_goal_acknowledged",
        f"{stair_path}.terminal_supervisor_goal_acknowledged",
    )
    _expect_exact_field(
        ctx,
        stair,
        "terminal_supervisor_goal_pending_started_timestamp",
        None,
        f"{stair_path}.terminal_supervisor_goal_pending_started_timestamp",
    )
    policy_freeze_faults = _validate_string_array(
        ctx,
        ctx.field(
            stair,
            "policy_freeze_write_fault_reasons",
            f"{stair_path}.policy_freeze_write_fault_reasons",
        ),
        f"{stair_path}.policy_freeze_write_fault_reasons",
    )
    if policy_freeze_faults:
        ctx.reject(
            "unexpected_policy_freeze_write_fault",
            f"{stair_path}.policy_freeze_write_fault_reasons",
            "成功楼梯验收不能携带 policy 冻结写入协议故障。",
            actual=list(policy_freeze_faults),
        )
    for key in (
        "policy_freeze_write_fault_sequence",
        "policy_freeze_write_fault_timestamp",
    ):
        _expect_exact_field(ctx, stair, key, None, f"{stair_path}.{key}")


def _validate_stair_mode(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
) -> None:
    _validate_task_identity(ctx, summary, evidence, expected_task_id=1002)
    _expect_exact_field(ctx, summary, "execution_mode", "stair_locomotion_smoke", "$.execution_mode")
    _expect_exact_field(
        ctx,
        summary,
        "success_semantics",
        "scan_stair_root_lock_workaround",
        "$.success_semantics",
    )
    _validate_top_provenance(
        ctx,
        summary,
        {
            "navigation_root_lock_workaround_success": True,
            "physical_navigation_success": False,
            "pure_physics_success": False,
            "used_base_teleport": True,
            "used_direct_joint_state": True,
            "used_object_teleport": False,
            "used_kinematic_object_follow": False,
            "used_visual_replay": False,
            "used_navigation_base_lock": True,
            "used_navigation_support_joint_lock": True,
            "used_navigation_joint_pose_lock": True,
        },
    )
    executor = evidence.executor
    _expect_false_field(ctx, executor, "policy_activity_seen", "$.latest_executor_status.policy_activity_seen")
    _expect_true_field(
        ctx,
        executor,
        "certified_root_lock_progress_seen",
        "$.latest_executor_status.certified_root_lock_progress_seen",
    )
    _expect_true_field(
        ctx,
        executor,
        "stair_freeze_finish_ready",
        "$.latest_executor_status.stair_freeze_finish_ready",
    )
    _expect_true_field(
        ctx,
        executor,
        "last_navigation_cmd_vel_inhibited",
        "$.latest_executor_status.last_navigation_cmd_vel_inhibited",
    )
    inhibit_reason = ctx.required_string(
        executor,
        "last_navigation_cmd_vel_inhibit_reason",
        "$.latest_executor_status.last_navigation_cmd_vel_inhibit_reason",
        nonempty=True,
    )
    if inhibit_reason is not None and not inhibit_reason.startswith("scan_stair_"):
        ctx.reject(
            "wrong_mode",
            "$.latest_executor_status.last_navigation_cmd_vel_inhibit_reason",
            "静态楼梯终态必须由 scan_stair 冻结原因抑制速度。",
            actual=inhibit_reason,
        )
    stop_reasons = _validate_string_array(
        ctx,
        ctx.field(executor, "last_stop_reasons", "$.latest_executor_status.last_stop_reasons"),
        "$.latest_executor_status.last_stop_reasons",
    )
    if not any(reason.startswith("scan_stair_") for reason in stop_reasons):
        ctx.reject(
            "missing_stair_freeze_evidence",
            "$.latest_executor_status.last_stop_reasons",
            "静态楼梯终态缺少 scan_stair 停车原因。",
            actual=list(stop_reasons),
        )
    stair_path = "$.latest_executor_status.stair_freeze"
    stair = ctx.required_mapping(executor, "stair_freeze", stair_path)
    for key in (
        "enabled",
        "applicable",
        "terminal_component",
        "terminal_hold",
        "terminal_goal_bound",
        "active",
        "finish_ready",
        "certified_progress_seen",
        "non_physical_root_lock_workaround",
    ):
        _expect_true_field(ctx, stair, key, f"{stair_path}.{key}")
    _expect_false_field(ctx, stair, "emergency_hold_latched", f"{stair_path}.emergency_hold_latched")
    _expect_exact_field(ctx, stair, "phase", "terminal_hold", f"{stair_path}.phase")
    path_stamp = ctx.required_integer(stair, "path_stamp_ns", f"{stair_path}.path_stamp_ns", minimum=1)
    if evidence.path_stamp_ns is not None and path_stamp != evidence.path_stamp_ns:
        ctx.reject("wrong_identity", f"{stair_path}.path_stamp_ns", "楼梯冻结绑定的 Path stamp 错误。", actual=path_stamp)
    stair_hash = _validate_sha256(
        ctx,
        ctx.field(stair, "path_points_sha256", f"{stair_path}.path_points_sha256"),
        f"{stair_path}.path_points_sha256",
    )
    executor_hash = executor.get("live_reference_path_points_sha256")
    if stair_hash is not None and stair_hash != executor_hash:
        ctx.reject("wrong_identity", f"{stair_path}.path_points_sha256", "楼梯冻结几何哈希与 live Path 不一致。", actual=stair_hash)
    _validate_stair_sensor_acquisition_barrier(
        ctx,
        stair,
        stair_path=stair_path,
        evidence=evidence,
    )
    progress = ctx.required_number(stair, "progress_m", f"{stair_path}.progress_m", minimum=0.0)
    total = ctx.required_number(stair, "total_length_m", f"{stair_path}.total_length_m", minimum=0.0)
    ratio = ctx.required_number(stair, "progress_ratio", f"{stair_path}.progress_ratio", minimum=0.0)
    if total is not None and total <= 0.0:
        ctx.reject("invalid_progress", f"{stair_path}.total_length_m", "楼梯冻结路径总长必须为正。", actual=total)
    if progress is not None and total is not None and not math.isclose(progress, total, rel_tol=0.0, abs_tol=1.0e-6):
        ctx.reject("incomplete_stair_progress", f"{stair_path}.progress_m", "楼梯冻结未完成全部路径进度。", actual=progress)
    _expect_close(ctx, ratio, 1.0, f"{stair_path}.progress_ratio", tolerance=1.0e-6)

    publish_path = "$.simulation_report.navigation_stair_execution_frozen_last_publish_report"
    publish = ctx.required_mapping(
        evidence.simulation,
        "navigation_stair_execution_frozen_last_publish_report",
        publish_path,
    )
    _validate_stair_execution_freeze_report(
        ctx,
        publish,
        publish_path,
        active_path_stamp_ns=evidence.path_stamp_ns,
    )
    _validate_tracking_writes(ctx, evidence, required=False)
    _validate_controller_lifecycle(ctx, evidence, require_multiple_trajectories=False)
    _validate_no_dynamic_obstacles(ctx, evidence)


def _validate_nonfreezing_policy_mode(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
    *,
    expected_task_id: int,
) -> None:
    _validate_task_identity(ctx, summary, evidence, expected_task_id=expected_task_id)
    _expect_exact_field(ctx, summary, "execution_mode", "navigation_smoke", "$.execution_mode")
    _expect_exact_field(ctx, summary, "success_semantics", "physical_nav_to_pick_only", "$.success_semantics")
    _validate_top_provenance(
        ctx,
        summary,
        {
            "navigation_root_lock_workaround_success": False,
            "physical_navigation_success": True,
            "used_base_teleport": False,
            "used_direct_joint_state": False,
            "used_object_teleport": False,
            "used_kinematic_object_follow": False,
            "used_visual_replay": False,
            "used_navigation_base_lock": False,
            "used_navigation_support_joint_lock": False,
            "used_navigation_joint_pose_lock": False,
        },
    )
    _expect_true_field(ctx, evidence.executor, "policy_activity_seen", "$.latest_executor_status.policy_activity_seen")
    _expect_false_field(
        ctx,
        evidence.executor,
        "certified_root_lock_progress_seen",
        "$.latest_executor_status.certified_root_lock_progress_seen",
    )
    stair_path = "$.latest_executor_status.stair_freeze"
    stair = ctx.required_mapping(evidence.executor, "stair_freeze", stair_path)
    _expect_false_field(ctx, stair, "applicable", f"{stair_path}.applicable")
    _expect_false_field(ctx, stair, "active", f"{stair_path}.active")
    _expect_false_field(
        ctx,
        stair,
        "non_physical_root_lock_workaround",
        f"{stair_path}.non_physical_root_lock_workaround",
    )
    _expect_exact_field(ctx, stair, "phase", "not_applicable", f"{stair_path}.phase")
    _validate_tracking_writes(ctx, evidence, required=True)


def _floor_id(
    ctx: _ValidationContext,
    parent: Mapping[str, Any],
    path: str,
) -> None:
    _expect_exact_field(ctx, parent, "floor_id", "F1", f"{path}.floor_id")


def _validate_flat_task_floors(
    ctx: _ValidationContext,
    task: Mapping[str, Any],
    *,
    require_place_disabled: bool,
) -> None:
    start = ctx.required_mapping(task, "start", "$.task_config.start")
    pick = ctx.required_mapping(task, "pick", "$.task_config.pick")
    base_goal = ctx.required_mapping(pick, "base_goal", "$.task_config.pick.base_goal")
    _floor_id(ctx, start, "$.task_config.start")
    _floor_id(ctx, base_goal, "$.task_config.pick.base_goal")
    if require_place_disabled:
        place = ctx.required_mapping(task, "place", "$.task_config.place")
        _expect_false_field(ctx, place, "enabled", "$.task_config.place.enabled")


def _validate_dynamic_state(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    expected: Mapping[str, Any],
) -> None:
    state = ctx.mapping(value, path)
    for key in ("id", "scene_asset_name"):
        _expect_exact_field(ctx, state, key, expected[key], f"{path}.{key}")
    for key in ("elapsed_time_s", "path_distance_m"):
        actual = ctx.required_number(state, key, f"{path}.{key}", minimum=0.0)
        _expect_close(
            ctx,
            actual,
            float(expected[key]),
            f"{path}.{key}",
            tolerance=_DYNAMIC_POSITION_TOLERANCE_M,
        )
    direction = ctx.required_integer(state, "path_direction", f"{path}.path_direction")
    if direction != expected["path_direction"]:
        ctx.reject(
            "invalid_dynamic_state",
            f"{path}.path_direction",
            "动态障碍方向与确定性运动学不一致。",
            actual=direction,
        )
    waiting = ctx.required_boolean(state, "waiting_for_start", f"{path}.waiting_for_start")
    if waiting != expected["waiting_for_start"]:
        ctx.reject(
            "invalid_dynamic_state",
            f"{path}.waiting_for_start",
            "动态障碍等待状态与确定性运动学不一致。",
            actual=waiting,
        )
    for key, length in (
        ("position_world_xyz", 3),
        ("orientation_world_wxyz", 4),
    ):
        actual_vector = _validate_vector(
            ctx,
            ctx.field(state, key, f"{path}.{key}"),
            f"{path}.{key}",
            length=length,
        )
        expected_vector = tuple(float(item) for item in expected[key])
        if actual_vector is not None and any(
            not math.isclose(
                actual,
                wanted,
                rel_tol=0.0,
                abs_tol=_DYNAMIC_POSITION_TOLERANCE_M,
            )
            for actual, wanted in zip(actual_vector, expected_vector, strict=True)
        ):
            ctx.reject(
                "invalid_dynamic_state",
                f"{path}.{key}",
                "动态障碍位姿与确定性运动学不一致。",
                actual=list(actual_vector),
            )


def _validate_dynamic_configuration(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    plan: DynamicObstaclePlan,
) -> None:
    path = "$.simulation_report.dynamic_obstacle_configuration_report"
    report = ctx.required_mapping(
        evidence.simulation,
        "dynamic_obstacle_configuration_report",
        path,
    )
    expected_report = plan.to_dict()
    for key, expected in expected_report.items():
        ctx.expect(ctx.field(report, key, f"{path}.{key}"), expected, f"{path}.{key}")
    assets = ctx.required_sequence(report, "registered_scene_assets", f"{path}.registered_scene_assets")
    if len(assets) != len(plan.obstacles):
        ctx.reject(
            "invalid_count",
            f"{path}.registered_scene_assets",
            "注册 scene asset 数必须等于动态障碍配置数。",
            actual=len(assets),
        )
    for index, obstacle in enumerate(plan.obstacles):
        if index >= len(assets):
            break
        asset_path = f"{path}.registered_scene_assets[{index}]"
        asset = ctx.mapping(assets[index], asset_path)
        exact = {
            "id": obstacle.obstacle_id,
            "scene_asset_name": obstacle.scene_asset_name,
            "prim_path": obstacle.prim_path,
            "shape": "cuboid",
            "kinematic_enabled": True,
            "collision_enabled": True,
            "visible": True,
        }
        for key, expected in exact.items():
            ctx.expect(ctx.field(asset, key, f"{asset_path}.{key}"), expected, f"{asset_path}.{key}")


def _validate_dynamic_runtime_and_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    plan: DynamicObstaclePlan,
) -> None:
    runtime_path = "$.simulation_report.dynamic_obstacle_runtime_report"
    runtime = ctx.required_mapping(
        evidence.simulation,
        "dynamic_obstacle_runtime_report",
        runtime_path,
    )
    _expect_true_field(ctx, runtime, "enabled", f"{runtime_path}.enabled")
    _expect_exact_field(
        ctx,
        runtime,
        "reason",
        "before_scene_write_data_to_sim",
        f"{runtime_path}.reason",
    )
    _expect_exact_field(
        ctx,
        runtime,
        "time_source",
        "episode_physics_step_index_x_physics_dt",
        f"{runtime_path}.time_source",
    )
    _expect_exact_field(
        ctx,
        runtime,
        "lifecycle_schema",
        "dynamic_obstacle_lifecycle_v1",
        f"{runtime_path}.lifecycle_schema",
    )
    _expect_false_field(ctx, runtime, "root_lock_state_used", f"{runtime_path}.root_lock_state_used")
    runtime_step = _expect_positive_integer_field(
        ctx,
        runtime,
        "physics_step_index",
        f"{runtime_path}.physics_step_index",
    )
    runtime_elapsed = ctx.required_number(
        runtime,
        "elapsed_time_s",
        f"{runtime_path}.elapsed_time_s",
        minimum=0.0,
    )
    runtime_count = ctx.required_integer(
        runtime,
        "obstacle_count",
        f"{runtime_path}.obstacle_count",
        minimum=1,
    )
    runtime_write_count = _expect_positive_integer_field(
        ctx,
        runtime,
        "pose_write_count",
        f"{runtime_path}.pose_write_count",
    )
    if runtime_count != len(plan.obstacles):
        ctx.reject("invalid_count", f"{runtime_path}.obstacle_count", "runtime 障碍数与任务配置不一致。", actual=runtime_count)
    runtime_states = ctx.required_sequence(runtime, "obstacles", f"{runtime_path}.obstacles")
    if len(runtime_states) != len(plan.obstacles):
        ctx.reject("invalid_count", f"{runtime_path}.obstacles", "runtime state 数与任务配置不一致。", actual=len(runtime_states))
    expected_runtime_states = (
        plan.state_at(runtime_elapsed)
        if runtime_elapsed is not None
        else ()
    )
    for index, expected_state in enumerate(expected_runtime_states):
        if index >= len(runtime_states):
            break
        _validate_dynamic_state(
            ctx,
            runtime_states[index],
            f"{runtime_path}.obstacles[{index}]",
            expected=expected_state.to_dict(),
        )

    lifecycle_path = "$.simulation_report.dynamic_obstacle_lifecycle_report"
    lifecycle = ctx.required_mapping(
        evidence.simulation,
        "dynamic_obstacle_lifecycle_report",
        lifecycle_path,
    )
    _expect_exact_field(ctx, lifecycle, "schema", "dynamic_obstacle_lifecycle_v1", f"{lifecycle_path}.schema")
    ctx.required_number(
        lifecycle,
        "ros_time_offset_s",
        f"{lifecycle_path}.ros_time_offset_s",
        minimum=0.0,
    )
    _expect_true_field(ctx, lifecycle, "enabled", f"{lifecycle_path}.enabled")
    _expect_true_field(
        ctx,
        lifecycle,
        "all_configured_obstacles_sampled",
        f"{lifecycle_path}.all_configured_obstacles_sampled",
    )
    _expect_true_field(
        ctx,
        lifecycle,
        "all_configured_obstacles_moved",
        f"{lifecycle_path}.all_configured_obstacles_moved",
    )
    obstacle_count = ctx.required_integer(lifecycle, "obstacle_count", f"{lifecycle_path}.obstacle_count", minimum=1)
    pose_write_count = _expect_positive_integer_field(ctx, lifecycle, "pose_write_count", f"{lifecycle_path}.pose_write_count")
    frame_count = ctx.required_integer(lifecycle, "sample_frame_count", f"{lifecycle_path}.sample_frame_count", minimum=2)
    first_step = ctx.required_integer(lifecycle, "first_physics_step_index", f"{lifecycle_path}.first_physics_step_index", minimum=0)
    last_step = _expect_positive_integer_field(ctx, lifecycle, "last_physics_step_index", f"{lifecycle_path}.last_physics_step_index")
    first_elapsed = ctx.required_number(lifecycle, "first_elapsed_time_s", f"{lifecycle_path}.first_elapsed_time_s", minimum=0.0)
    last_elapsed = ctx.required_number(lifecycle, "last_elapsed_time_s", f"{lifecycle_path}.last_elapsed_time_s", minimum=0.0)
    maximum_span = ctx.required_number(
        lifecycle,
        "maximum_path_distance_span_m",
        f"{lifecycle_path}.maximum_path_distance_span_m",
        minimum=0.0,
    )
    _expect_nonnegative_integer_field(ctx, lifecycle, "direction_transition_count", f"{lifecycle_path}.direction_transition_count")
    if obstacle_count != len(plan.obstacles):
        ctx.reject("invalid_count", f"{lifecycle_path}.obstacle_count", "lifecycle 障碍数与任务配置不一致。", actual=obstacle_count)
    if (
        runtime_write_count is not None
        and pose_write_count is not None
        and runtime_write_count != pose_write_count
    ):
        ctx.reject("invalid_count", f"{lifecycle_path}.pose_write_count", "runtime 与 lifecycle pose write count 不一致。", actual=pose_write_count)
    summary_write_count = ctx.required_integer(
        evidence.simulation,
        "dynamic_obstacle_pose_write_count",
        "$.simulation_report.dynamic_obstacle_pose_write_count",
        minimum=1,
    )
    if summary_write_count != pose_write_count:
        ctx.reject("invalid_count", "$.simulation_report.dynamic_obstacle_pose_write_count", "summary 与 lifecycle pose write count 不一致。", actual=summary_write_count)
    if (
        frame_count is not None
        and obstacle_count is not None
        and pose_write_count is not None
        and pose_write_count != frame_count * obstacle_count
    ):
        ctx.reject("invalid_count", f"{lifecycle_path}.pose_write_count", "每个 sample frame 必须写入全部配置障碍。", actual=pose_write_count)
    if first_step is not None and last_step is not None and last_step <= first_step:
        ctx.reject("invalid_dynamic_lifecycle", f"{lifecycle_path}.last_physics_step_index", "动态生命周期必须跨越多个递增 physics step。", actual=last_step)
    if runtime_step is not None and last_step is not None and runtime_step != last_step:
        ctx.reject("invalid_dynamic_lifecycle", f"{runtime_path}.physics_step_index", "runtime 末 step 与 lifecycle 不一致。", actual=runtime_step)
    if first_elapsed is not None and last_elapsed is not None and last_elapsed <= first_elapsed:
        ctx.reject("invalid_dynamic_lifecycle", f"{lifecycle_path}.last_elapsed_time_s", "动态生命周期 elapsed time 必须递增。", actual=last_elapsed)
    if runtime_elapsed is not None and last_elapsed is not None and not math.isclose(runtime_elapsed, last_elapsed, rel_tol=0.0, abs_tol=1.0e-9):
        ctx.reject("invalid_dynamic_lifecycle", f"{runtime_path}.elapsed_time_s", "runtime 末时间与 lifecycle 不一致。", actual=runtime_elapsed)
    if maximum_span is not None and maximum_span <= 1.0e-6:
        ctx.reject("dynamic_obstacle_not_moved", f"{lifecycle_path}.maximum_path_distance_span_m", "动态障碍必须有可测的路径距离跨度。", actual=maximum_span)

    obstacle_reports = ctx.required_mapping(lifecycle, "obstacles", f"{lifecycle_path}.obstacles")
    expected_ids = {obstacle.obstacle_id for obstacle in plan.obstacles}
    if set(obstacle_reports) != expected_ids:
        ctx.reject(
            "wrong_identity",
            f"{lifecycle_path}.obstacles",
            "lifecycle 障碍 ID 集合与任务配置不一致。",
            actual=sorted(obstacle_reports),
        )
    for obstacle in plan.obstacles:
        obstacle_path = f"{lifecycle_path}.obstacles.{obstacle.obstacle_id}"
        report = ctx.mapping(obstacle_reports.get(obstacle.obstacle_id), obstacle_path)
        _expect_exact_field(ctx, report, "scene_asset_name", obstacle.scene_asset_name, f"{obstacle_path}.scene_asset_name")
        samples = ctx.required_integer(report, "sample_count", f"{obstacle_path}.sample_count", minimum=2)
        if samples != frame_count:
            ctx.reject("invalid_count", f"{obstacle_path}.sample_count", "每个障碍 sample_count 必须等于 frame count。", actual=samples)
        minimum_distance = ctx.required_number(report, "minimum_path_distance_m", f"{obstacle_path}.minimum_path_distance_m", minimum=0.0)
        maximum_distance = ctx.required_number(report, "maximum_path_distance_m", f"{obstacle_path}.maximum_path_distance_m", minimum=0.0)
        span = ctx.required_number(report, "path_distance_span_m", f"{obstacle_path}.path_distance_span_m", minimum=0.0)
        displacement = ctx.required_number(
            report,
            "maximum_displacement_from_first_m",
            f"{obstacle_path}.maximum_displacement_from_first_m",
            minimum=0.0,
        )
        if minimum_distance is not None and maximum_distance is not None and span is not None:
            _expect_close(
                ctx,
                span,
                maximum_distance - minimum_distance,
                f"{obstacle_path}.path_distance_span_m",
                tolerance=1.0e-9,
            )
        if span is not None and span <= 1.0e-6:
            ctx.reject("dynamic_obstacle_not_moved", f"{obstacle_path}.path_distance_span_m", "配置障碍没有发生有效运动。", actual=span)
        if displacement is not None and displacement <= 1.0e-6:
            ctx.reject("dynamic_obstacle_not_moved", f"{obstacle_path}.maximum_displacement_from_first_m", "配置障碍没有世界系位移。", actual=displacement)
        _expect_true_field(ctx, report, "waiting_for_start_seen", f"{obstacle_path}.waiting_for_start_seen")
        _expect_true_field(ctx, report, "motion_started_seen", f"{obstacle_path}.motion_started_seen")
        directions = ctx.required_sequence(report, "path_directions_seen", f"{obstacle_path}.path_directions_seen")
        for index, direction in enumerate(directions):
            parsed_direction = ctx.integer(direction, f"{obstacle_path}.path_directions_seen[{index}]")
            if parsed_direction not in (-1, 0, 1):
                ctx.reject("invalid_dynamic_state", f"{obstacle_path}.path_directions_seen[{index}]", "动态障碍方向只能为 -1/0/1。", actual=parsed_direction)
        _expect_nonnegative_integer_field(ctx, report, "direction_transition_count", f"{obstacle_path}.direction_transition_count")
        if first_elapsed is not None:
            _validate_dynamic_state(
                ctx,
                ctx.field(report, "first_state", f"{obstacle_path}.first_state"),
                f"{obstacle_path}.first_state",
                expected=obstacle.state_at(first_elapsed).to_dict(),
            )
        if last_elapsed is not None:
            _validate_dynamic_state(
                ctx,
                ctx.field(report, "last_state", f"{obstacle_path}.last_state"),
                f"{obstacle_path}.last_state",
                expected=obstacle.state_at(last_elapsed).to_dict(),
            )


def _validate_raw_cloud_detection(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    obstacle: Any,
) -> tuple[int | None, float | None]:
    detection = ctx.mapping(value, path)
    ctx.required_number(detection, "timestamp", f"{path}.timestamp", minimum=0.0)
    ctx.required_integer(
        detection,
        "completed_control_step",
        f"{path}.completed_control_step",
        minimum=1,
    )
    ctx.required_integer(
        detection,
        "physics_step_index",
        f"{path}.physics_step_index",
        minimum=1,
    )
    elapsed = ctx.required_number(
        detection,
        "elapsed_time_s",
        f"{path}.elapsed_time_s",
        minimum=0.0,
    )
    point_count = ctx.required_integer(
        detection,
        "point_count",
        f"{path}.point_count",
        minimum=1,
    )
    if elapsed is not None:
        _validate_dynamic_state(
            ctx,
            ctx.field(detection, "state", f"{path}.state"),
            f"{path}.state",
            expected=obstacle.state_at(elapsed).to_dict(),
        )
    return point_count, elapsed


def _validate_raw_cloud_frame(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    plan: DynamicObstaclePlan,
    require_detection: bool,
) -> tuple[int | None, float | None]:
    frame = ctx.mapping(value, path)
    _expect_exact_field(ctx, frame, "schema", "dynamic_obstacle_raw_cloud_frame_v1", f"{path}.schema")
    _expect_exact_field(ctx, frame, "source", "isaac_rtx_world_cloud_before_ros_filter", f"{path}.source")
    _expect_exact_field(ctx, frame, "proof_scope", "raw_cloud_visibility_only", f"{path}.proof_scope")
    ctx.required_number(frame, "timestamp", f"{path}.timestamp", minimum=0.0)
    ctx.required_integer(frame, "completed_control_step", f"{path}.completed_control_step", minimum=1)
    ctx.required_integer(frame, "physics_step_index", f"{path}.physics_step_index", minimum=1)
    elapsed = ctx.required_number(frame, "elapsed_time_s", f"{path}.elapsed_time_s", minimum=0.0)
    ctx.required_integer(frame, "raw_point_count", f"{path}.raw_point_count", minimum=1)
    total = ctx.required_integer(
        frame,
        "total_obstacle_point_count",
        f"{path}.total_obstacle_point_count",
        minimum=1 if require_detection else 0,
    )
    obstacles = ctx.required_mapping(frame, "obstacles", f"{path}.obstacles")
    expected_ids = {obstacle.obstacle_id for obstacle in plan.obstacles}
    if set(obstacles) != expected_ids:
        ctx.reject("wrong_identity", f"{path}.obstacles", "点云帧障碍 ID 集合与任务配置不一致。", actual=sorted(obstacles))
    expected_states = {
        state.obstacle_id: state.to_dict()
        for state in plan.state_at(elapsed)
    } if elapsed is not None else {}
    summed_points = 0
    for obstacle in plan.obstacles:
        obstacle_path = f"{path}.obstacles.{obstacle.obstacle_id}"
        item = ctx.mapping(obstacles.get(obstacle.obstacle_id), obstacle_path)
        point_count = ctx.required_integer(item, "point_count", f"{obstacle_path}.point_count", minimum=0)
        if point_count is not None:
            summed_points += point_count
        if obstacle.obstacle_id in expected_states:
            _validate_dynamic_state(
                ctx,
                ctx.field(item, "state", f"{obstacle_path}.state"),
                f"{obstacle_path}.state",
                expected=expected_states[obstacle.obstacle_id],
            )
    if total is not None and summed_points != total:
        ctx.reject("invalid_count", f"{path}.total_obstacle_point_count", "点云帧总障碍点数与逐障碍求和不一致。", actual=total)
    return total, elapsed


def _validate_dynamic_raw_cloud_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    plan: DynamicObstaclePlan,
) -> None:
    """认证 RTX 原始点云命中；明确不外推到过滤后的 cloud_registered。"""

    path = "$.simulation_report.dynamic_obstacle_raw_cloud_lifecycle_report"
    report = ctx.required_mapping(
        evidence.simulation,
        "dynamic_obstacle_raw_cloud_lifecycle_report",
        path,
    )
    _expect_exact_field(ctx, report, "schema", "dynamic_obstacle_raw_cloud_lifecycle_v1", f"{path}.schema")
    _expect_true_field(ctx, report, "enabled", f"{path}.enabled")
    _expect_exact_field(ctx, report, "source", "isaac_rtx_world_cloud_before_ros_filter", f"{path}.source")
    _expect_exact_field(ctx, report, "proof_scope", "raw_cloud_visibility_only", f"{path}.proof_scope")
    tolerance = ctx.required_number(report, "aabb_tolerance_m", f"{path}.aabb_tolerance_m", minimum=0.0)
    if tolerance is not None and tolerance > 0.05:
        ctx.reject("invalid_cloud_evidence", f"{path}.aabb_tolerance_m", "原始点云 AABB 命中容差不能超过 5 cm。", actual=tolerance)
    sample_frames = _expect_positive_integer_field(ctx, report, "sample_frame_count", f"{path}.sample_frame_count")
    frames_detected = _expect_positive_integer_field(
        ctx,
        report,
        "frames_with_any_obstacle_points",
        f"{path}.frames_with_any_obstacle_points",
    )
    motion_frames = _expect_positive_integer_field(
        ctx,
        report,
        "frames_with_motion_started_obstacle_points",
        f"{path}.frames_with_motion_started_obstacle_points",
    )
    maximum_points = _expect_positive_integer_field(
        ctx,
        report,
        "maximum_total_obstacle_point_count",
        f"{path}.maximum_total_obstacle_point_count",
    )
    _expect_true_field(
        ctx,
        report,
        "all_configured_obstacles_observed",
        f"{path}.all_configured_obstacles_observed",
    )
    _first_total, first_frame_elapsed = _validate_raw_cloud_frame(
        ctx,
        ctx.field(report, "first_detection", f"{path}.first_detection"),
        f"{path}.first_detection",
        plan=plan,
        require_detection=True,
    )
    _last_total, last_frame_elapsed = _validate_raw_cloud_frame(
        ctx,
        ctx.field(report, "last_detection", f"{path}.last_detection"),
        f"{path}.last_detection",
        plan=plan,
        require_detection=True,
    )
    if (
        first_frame_elapsed is not None
        and last_frame_elapsed is not None
        and last_frame_elapsed <= first_frame_elapsed
    ):
        ctx.reject("invalid_cloud_evidence", f"{path}.last_detection.elapsed_time_s", "全局最后命中帧必须晚于首次命中帧。", actual=last_frame_elapsed)
    _validate_raw_cloud_frame(
        ctx,
        ctx.field(
            evidence.simulation,
            "dynamic_obstacle_raw_cloud_last_report",
            "$.simulation_report.dynamic_obstacle_raw_cloud_last_report",
        ),
        "$.simulation_report.dynamic_obstacle_raw_cloud_last_report",
        plan=plan,
        require_detection=False,
    )
    if sample_frames is not None:
        for count, count_path in (
            (frames_detected, f"{path}.frames_with_any_obstacle_points"),
            (motion_frames, f"{path}.frames_with_motion_started_obstacle_points"),
        ):
            if count is not None and count > sample_frames:
                ctx.reject("invalid_count", count_path, "点云命中帧数不能大于采样帧数。", actual=count)
    obstacles = ctx.required_mapping(report, "obstacles", f"{path}.obstacles")
    expected_ids = {obstacle.obstacle_id for obstacle in plan.obstacles}
    if set(obstacles) != expected_ids:
        ctx.reject("wrong_identity", f"{path}.obstacles", "原始点云障碍 ID 集合与任务配置不一致。", actual=sorted(obstacles))
    for obstacle in plan.obstacles:
        obstacle_path = f"{path}.obstacles.{obstacle.obstacle_id}"
        obstacle_report = ctx.mapping(obstacles.get(obstacle.obstacle_id), obstacle_path)
        _expect_exact_field(ctx, obstacle_report, "scene_asset_name", obstacle.scene_asset_name, f"{obstacle_path}.scene_asset_name")
        obstacle_samples = ctx.required_integer(
            obstacle_report,
            "sample_frame_count",
            f"{obstacle_path}.sample_frame_count",
            minimum=1,
        )
        detected = _expect_positive_integer_field(
            ctx,
            obstacle_report,
            "detected_frame_count",
            f"{obstacle_path}.detected_frame_count",
        )
        _expect_positive_integer_field(ctx, obstacle_report, "maximum_point_count", f"{obstacle_path}.maximum_point_count")
        if sample_frames is not None and obstacle_samples != sample_frames:
            ctx.reject("invalid_count", f"{obstacle_path}.sample_frame_count", "每个障碍的原始点云采样帧数必须等于全局帧数。", actual=obstacle_samples)
        if obstacle_samples is not None and detected is not None and detected > obstacle_samples:
            ctx.reject("invalid_count", f"{obstacle_path}.detected_frame_count", "命中帧数不能大于采样帧数。", actual=detected)
        _expect_true_field(
            ctx,
            obstacle_report,
            "motion_started_detection_seen",
            f"{obstacle_path}.motion_started_detection_seen",
        )
        minimum_distance = ctx.required_number(
            obstacle_report,
            "minimum_detected_path_distance_m",
            f"{obstacle_path}.minimum_detected_path_distance_m",
            minimum=0.0,
        )
        maximum_distance = ctx.required_number(
            obstacle_report,
            "maximum_detected_path_distance_m",
            f"{obstacle_path}.maximum_detected_path_distance_m",
            minimum=0.0,
        )
        span = ctx.required_number(
            obstacle_report,
            "detected_path_distance_span_m",
            f"{obstacle_path}.detected_path_distance_span_m",
            minimum=0.0,
        )
        if minimum_distance is not None and maximum_distance is not None and span is not None:
            _expect_close(ctx, span, maximum_distance - minimum_distance, f"{obstacle_path}.detected_path_distance_span_m", tolerance=1.0e-9)
        if span is not None and span <= 1.0e-6:
            ctx.reject("missing_raw_cloud_motion_evidence", f"{obstacle_path}.detected_path_distance_span_m", "原始点云必须在两个不同运动位置命中障碍。", actual=span)
        directions = ctx.required_sequence(
            obstacle_report,
            "path_directions_detected",
            f"{obstacle_path}.path_directions_detected",
        )
        if not directions:
            ctx.reject("missing_raw_cloud_motion_evidence", f"{obstacle_path}.path_directions_detected", "缺少动态障碍命中方向。")
        for index, direction in enumerate(directions):
            parsed = ctx.integer(direction, f"{obstacle_path}.path_directions_detected[{index}]")
            if parsed not in (-1, 0, 1):
                ctx.reject("invalid_dynamic_state", f"{obstacle_path}.path_directions_detected[{index}]", "命中方向只能为 -1/0/1。", actual=parsed)
        first_count, first_elapsed = _validate_raw_cloud_detection(
            ctx,
            ctx.field(obstacle_report, "first_detection", f"{obstacle_path}.first_detection"),
            f"{obstacle_path}.first_detection",
            obstacle=obstacle,
        )
        last_count, last_elapsed = _validate_raw_cloud_detection(
            ctx,
            ctx.field(obstacle_report, "last_detection", f"{obstacle_path}.last_detection"),
            f"{obstacle_path}.last_detection",
            obstacle=obstacle,
        )
        if first_elapsed is not None and last_elapsed is not None and last_elapsed <= first_elapsed:
            ctx.reject("invalid_cloud_evidence", f"{obstacle_path}.last_detection.elapsed_time_s", "最后点云命中必须晚于首次命中。", actual=last_elapsed)
        if maximum_points is not None and first_count is not None and first_count > maximum_points:
            ctx.reject("invalid_count", f"{obstacle_path}.first_detection.point_count", "单障碍点数不能超过全局 maximum total。", actual=first_count)
        if maximum_points is not None and last_count is not None and last_count > maximum_points:
            ctx.reject("invalid_count", f"{obstacle_path}.last_detection.point_count", "单障碍点数不能超过全局 maximum total。", actual=last_count)


def _validate_grid_map_diagnostic_report(
    ctx: _ValidationContext,
    value: object,
    path: str,
) -> _GridMapDiagnosticEvidence:
    """校验 ``GridMapObservationDiagnostics`` 的 summary 序列化。"""

    report = ctx.mapping(value, path)
    _expect_exact_field(
        ctx,
        report,
        "source",
        "ros2_scan_grid_map_observation_diagnostics",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        report,
        "topic",
        "/planning/grid_map_observation_diagnostics",
        f"{path}.topic",
    )
    receipt_timestamp = ctx.required_number(
        report,
        "receipt_timestamp",
        f"{path}.receipt_timestamp",
        minimum=0.0,
    )
    rx_sequence = ctx.required_integer(
        report,
        "rx_sequence",
        f"{path}.rx_sequence",
        minimum=1,
    )
    ros_time_offset_s = ctx.required_number(
        report,
        "ros_time_offset_s",
        f"{path}.ros_time_offset_s",
        minimum=0.0,
    )
    header = ctx.required_mapping(report, "header", f"{path}.header")
    _expect_exact_field(ctx, header, "frame_id", "world", f"{path}.header.frame_id")
    header_stamp_ns = _validate_ros_stamp(
        ctx,
        header,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{path}.header",
    )
    episode_elapsed_time_s = ctx.required_number(
        report,
        "episode_elapsed_time_s",
        f"{path}.episode_elapsed_time_s",
        minimum=0.0,
    )
    observation_sequence = ctx.required_integer(
        report,
        "observation_sequence",
        f"{path}.observation_sequence",
        minimum=1,
    )
    sensor_pose_stamp = ctx.required_mapping(
        report,
        "sensor_pose_stamp",
        f"{path}.sensor_pose_stamp",
    )
    sensor_pose_stamp_ns = ctx.required_integer(
        sensor_pose_stamp,
        "stamp_ns",
        f"{path}.sensor_pose_stamp.stamp_ns",
        minimum=1,
    )
    sensor_pose_sec = ctx.required_integer(
        sensor_pose_stamp,
        "sec",
        f"{path}.sensor_pose_stamp.sec",
        minimum=0,
    )
    sensor_pose_nanosec = ctx.required_integer(
        sensor_pose_stamp,
        "nanosec",
        f"{path}.sensor_pose_stamp.nanosec",
        minimum=0,
    )
    if sensor_pose_nanosec is not None and sensor_pose_nanosec >= 1_000_000_000:
        ctx.reject(
            "invalid_timestamp",
            f"{path}.sensor_pose_stamp.nanosec",
            "nanosec 必须小于 1e9。",
            actual=sensor_pose_nanosec,
        )
    if (
        sensor_pose_stamp_ns is not None
        and sensor_pose_sec is not None
        and sensor_pose_nanosec is not None
        and sensor_pose_sec * 1_000_000_000 + sensor_pose_nanosec
        != sensor_pose_stamp_ns
    ):
        ctx.reject(
            "invalid_timestamp",
            f"{path}.sensor_pose_stamp",
            "sensor pose sec/nanosec 与 stamp_ns 不一致。",
        )
    _validate_vector(
        ctx,
        ctx.field(report, "sensor_origin_world_xyz", f"{path}.sensor_origin_world_xyz"),
        f"{path}.sensor_origin_world_xyz",
        length=3,
    )
    canonical_empty = ctx.required_boolean(
        report,
        "canonical_empty",
        f"{path}.canonical_empty",
    )
    fusion = ctx.required_boolean(
        report,
        "map_fusion_performed",
        f"{path}.map_fusion_performed",
    )
    map_resolution_m = ctx.required_number(
        report,
        "map_resolution_m",
        f"{path}.map_resolution_m",
        minimum=1.0e-9,
    )
    count_keys = (
        "input_point_count",
        "accepted_endpoint_count",
        "hit_endpoint_count",
        "explicit_free_endpoint_count",
        "free_to_occupied_transition_count",
        "explicit_free_miss_voxel_count",
        "occupied_to_free_by_explicit_miss_count",
        "occupied_removed_by_sliding_reset_count",
    )
    counts = {
        key: ctx.required_integer(report, key, f"{path}.{key}", minimum=0)
        for key in count_keys
    }
    hit_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "hit_endpoint_samples_world_xyz",
            f"{path}.hit_endpoint_samples_world_xyz",
        ),
        f"{path}.hit_endpoint_samples_world_xyz",
    )
    hit_truncated = ctx.required_boolean(
        report,
        "hit_endpoint_samples_truncated",
        f"{path}.hit_endpoint_samples_truncated",
    )
    hit_voxel_indices = _validate_voxel_index_array(
        ctx,
        ctx.field(
            report,
            "hit_endpoint_sample_voxel_indices_xyz",
            f"{path}.hit_endpoint_sample_voxel_indices_xyz",
        ),
        f"{path}.hit_endpoint_sample_voxel_indices_xyz",
    )
    transition_hit_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "free_to_occupied_transition_hit_samples_world_xyz",
            f"{path}.free_to_occupied_transition_hit_samples_world_xyz",
        ),
        f"{path}.free_to_occupied_transition_hit_samples_world_xyz",
    )
    transition_truncated = ctx.required_boolean(
        report,
        "free_to_occupied_transition_samples_truncated",
        f"{path}.free_to_occupied_transition_samples_truncated",
    )
    transition_voxel_indices = _validate_voxel_index_array(
        ctx,
        ctx.field(
            report,
            "free_to_occupied_transition_voxel_indices_xyz",
            f"{path}.free_to_occupied_transition_voxel_indices_xyz",
        ),
        f"{path}.free_to_occupied_transition_voxel_indices_xyz",
    )
    clear_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "occupied_to_free_by_explicit_miss_samples_world_xyz",
            f"{path}.occupied_to_free_by_explicit_miss_samples_world_xyz",
        ),
        f"{path}.occupied_to_free_by_explicit_miss_samples_world_xyz",
    )
    clear_truncated = ctx.required_boolean(
        report,
        "occupied_to_free_samples_truncated",
        f"{path}.occupied_to_free_samples_truncated",
    )
    clear_voxel_indices = _validate_voxel_index_array(
        ctx,
        ctx.field(
            report,
            "occupied_to_free_sample_voxel_indices_xyz",
            f"{path}.occupied_to_free_sample_voxel_indices_xyz",
        ),
        f"{path}.occupied_to_free_sample_voxel_indices_xyz",
    )
    clear_transition_hit_sequences = _validate_observation_sequence_array(
        ctx,
        ctx.field(
            report,
            "occupied_to_free_transition_hit_observation_sequences",
            (
                f"{path}."
                "occupied_to_free_transition_hit_observation_sequences"
            ),
        ),
        (
            f"{path}."
            "occupied_to_free_transition_hit_observation_sequences"
        ),
    )
    clear_transition_hit_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "occupied_to_free_transition_hit_samples_world_xyz",
            f"{path}.occupied_to_free_transition_hit_samples_world_xyz",
        ),
        f"{path}.occupied_to_free_transition_hit_samples_world_xyz",
    )
    clear_transition_hit_header_stamps_ns = _validate_header_stamp_ns_array(
        ctx,
        ctx.field(
            report,
            "occupied_to_free_transition_hit_header_stamp_ns",
            f"{path}.occupied_to_free_transition_hit_header_stamp_ns",
        ),
        f"{path}.occupied_to_free_transition_hit_header_stamp_ns",
    )
    hit_count = counts["hit_endpoint_count"]
    transition_count = counts["free_to_occupied_transition_count"]
    clear_count = counts["occupied_to_free_by_explicit_miss_count"]
    for total, samples, truncated, sample_path in (
        (
            hit_count,
            hit_samples,
            hit_truncated,
            f"{path}.hit_endpoint_samples_world_xyz",
        ),
        (
            transition_count,
            transition_hit_samples,
            transition_truncated,
            f"{path}.free_to_occupied_transition_hit_samples_world_xyz",
        ),
        (
            clear_count,
            clear_samples,
            clear_truncated,
            f"{path}.occupied_to_free_by_explicit_miss_samples_world_xyz",
        ),
    ):
        if total is None or truncated is None:
            continue
        expected_sample_count = min(total, _PROOF_RING_CAPACITY)
        expected_truncated = total > _PROOF_RING_CAPACITY
        if truncated != expected_truncated:
            ctx.reject(
                "invalid_sample_contract",
                sample_path,
                "typed 样本截断标志必须且仅能表示总数超过 64。",
            )
        if len(samples) != expected_sample_count:
            ctx.reject(
                "invalid_sample_contract",
                sample_path,
                "typed 样本必须严格保留 min(total_count, 64) 个元素。",
                actual=len(samples),
            )
    for samples, indices, indices_path in (
        (
            hit_samples,
            hit_voxel_indices,
            f"{path}.hit_endpoint_sample_voxel_indices_xyz",
        ),
        (
            transition_hit_samples,
            transition_voxel_indices,
            f"{path}.free_to_occupied_transition_voxel_indices_xyz",
        ),
        (
            clear_samples,
            clear_voxel_indices,
            f"{path}.occupied_to_free_sample_voxel_indices_xyz",
        ),
    ):
        if len(samples) != len(indices):
            ctx.reject(
                "invalid_sample_contract",
                indices_path,
                "voxel index 数组必须与对应点样本逐项等长。",
                actual=len(indices),
            )
    for provenance_values, provenance_path in (
        (
            clear_transition_hit_sequences,
            (
                f"{path}."
                "occupied_to_free_transition_hit_observation_sequences"
            ),
        ),
        (
            clear_transition_hit_samples,
            f"{path}.occupied_to_free_transition_hit_samples_world_xyz",
        ),
        (
            clear_transition_hit_header_stamps_ns,
            f"{path}.occupied_to_free_transition_hit_header_stamp_ns",
        ),
    ):
        if len(clear_samples) != len(provenance_values):
            ctx.reject(
                "invalid_sample_contract",
                provenance_path,
                "clear provenance 数组必须与 clear 样本逐项等长。",
                actual=len(provenance_values),
            )
    if observation_sequence is not None:
        for index, (sequence, hit_stamp_ns) in enumerate(
            zip(
                clear_transition_hit_sequences,
                clear_transition_hit_header_stamps_ns,
            )
        ):
            if sequence >= observation_sequence:
                ctx.reject(
                    "invalid_ghost_clear_sequence",
                    (
                        f"{path}."
                        "occupied_to_free_transition_hit_observation_sequences"
                        f"[{index}]"
                    ),
                    "clear provenance 必须引用更早 observation sequence。",
                    actual=sequence,
                )
            if sequence == 0 and hit_stamp_ns != 0:
                ctx.reject(
                    "invalid_ghost_clear_provenance",
                    (
                        f"{path}."
                        "occupied_to_free_transition_hit_header_stamp_ns"
                        f"[{index}]"
                    ),
                    "无来源的 sequence=0 必须同时使用 header stamp=0。",
                    actual=hit_stamp_ns,
                )
            if sequence > 0 and hit_stamp_ns <= 0:
                ctx.reject(
                    "invalid_ghost_clear_provenance",
                    (
                        f"{path}."
                        "occupied_to_free_transition_hit_header_stamp_ns"
                        f"[{index}]"
                    ),
                    "可认证 transition sequence 必须携带正的来源 header stamp。",
                    actual=hit_stamp_ns,
                )
            if (
                sequence > 0
                and header_stamp_ns is not None
                and hit_stamp_ns >= header_stamp_ns
            ):
                ctx.reject(
                    "invalid_ghost_clear_sequence",
                    (
                        f"{path}."
                        "occupied_to_free_transition_hit_header_stamp_ns"
                        f"[{index}]"
                    ),
                    "transition hit header 必须早于 clear report header。",
                    actual=hit_stamp_ns,
                )
    input_count = counts["input_point_count"]
    accepted_count = counts["accepted_endpoint_count"]
    explicit_free_count = counts["explicit_free_endpoint_count"]
    explicit_miss_count = counts["explicit_free_miss_voxel_count"]
    if (
        input_count is not None
        and accepted_count is not None
        and accepted_count > input_count
    ):
        ctx.reject(
            "invalid_count",
            f"{path}.accepted_endpoint_count",
            "接纳端点数不能超过输入点数。",
            actual=accepted_count,
        )
    if (
        accepted_count is not None
        and hit_count is not None
        and explicit_free_count is not None
        and hit_count + explicit_free_count != accepted_count
    ):
        ctx.reject(
            "invalid_count",
            f"{path}.accepted_endpoint_count",
            "hit 与 explicit-free 端点数之和必须等于接纳端点数。",
            actual=accepted_count,
        )
    if clear_count is not None and explicit_miss_count is not None and clear_count > explicit_miss_count:
        ctx.reject(
            "invalid_count",
            f"{path}.occupied_to_free_by_explicit_miss_count",
            "occupied→free 数不能超过 explicit-miss 更新体素数。",
            actual=clear_count,
        )
    if accepted_count is not None and fusion is not None and fusion != (accepted_count > 0):
        ctx.reject(
            "invalid_grid_map_observation",
            f"{path}.map_fusion_performed",
            "融合标志必须与接纳端点数一致。",
            actual=fusion,
        )
    if canonical_empty is True and any(
        count not in (None, 0) for count in counts.values()
    ):
        ctx.reject(
            "invalid_grid_map_observation",
            path,
            "canonical empty 不得携带端点、融合或清障计数。",
        )
    return _GridMapDiagnosticEvidence(
        report=report,
        receipt_timestamp=receipt_timestamp,
        rx_sequence=rx_sequence,
        ros_time_offset_s=ros_time_offset_s,
        header_stamp_ns=header_stamp_ns,
        episode_elapsed_time_s=episode_elapsed_time_s,
        observation_sequence=observation_sequence,
        map_resolution_m=map_resolution_m,
        hit_samples=hit_samples,
        hit_voxel_indices=hit_voxel_indices,
        transition_hit_samples=transition_hit_samples,
        transition_voxel_indices=transition_voxel_indices,
        clear_samples=clear_samples,
        clear_voxel_indices=clear_voxel_indices,
        clear_transition_hit_sequences=clear_transition_hit_sequences,
        clear_transition_hit_samples=clear_transition_hit_samples,
        clear_transition_hit_header_stamps_ns=(
            clear_transition_hit_header_stamps_ns
        ),
        hit_endpoint_count=hit_count,
        free_to_occupied_transition_count=transition_count,
        explicit_free_miss_voxel_count=explicit_miss_count,
        occupied_to_free_count=clear_count,
        sliding_reset_count=counts["occupied_removed_by_sliding_reset_count"],
    )


def _validate_bspline_diagnostic_report(
    ctx: _ValidationContext,
    value: object,
    path: str,
    *,
    path_stamp_ns: int | None,
) -> _BsplineDiagnosticEvidence:
    """校验 ``BsplineDiagnostics`` 的 summary 序列化。"""

    report = ctx.mapping(value, path)
    _expect_exact_field(
        ctx,
        report,
        "source",
        "ros2_scan_bspline_diagnostics",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        report,
        "topic",
        "/planning/bspline_diagnostics",
        f"{path}.topic",
    )
    receipt_timestamp = ctx.required_number(
        report,
        "receipt_timestamp",
        f"{path}.receipt_timestamp",
        minimum=0.0,
    )
    rx_sequence = ctx.required_integer(
        report,
        "rx_sequence",
        f"{path}.rx_sequence",
        minimum=1,
    )
    ros_time_offset_s = ctx.required_number(
        report,
        "ros_time_offset_s",
        f"{path}.ros_time_offset_s",
        minimum=0.0,
    )
    header = ctx.required_mapping(report, "header", f"{path}.header")
    _expect_exact_field(ctx, header, "frame_id", "world", f"{path}.header.frame_id")
    header_stamp_ns = _validate_ros_stamp(
        ctx,
        header,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{path}.header",
    )
    episode_elapsed_time_s = ctx.required_number(
        report,
        "episode_elapsed_time_s",
        f"{path}.episode_elapsed_time_s",
        minimum=0.0,
    )
    diagnostic_sequence = ctx.required_integer(
        report,
        "diagnostic_sequence",
        f"{path}.diagnostic_sequence",
        minimum=1,
    )
    identity = _validate_full_trajectory_identity(
        ctx,
        ctx.field(report, "identity", f"{path}.identity"),
        f"{path}.identity",
        path_stamp_ns=path_stamp_ns,
    )
    if identity is not None and header_stamp_ns is not None and identity[1] != header_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{path}.identity.bspline_header_stamp_ns",
            "B-spline identity 的 header stamp 必须等于 typed report header。",
            actual=identity[1],
        )
    is_final = ctx.required_boolean(report, "is_final", f"{path}.is_final")
    emergency_stop = ctx.required_boolean(
        report,
        "emergency_stop",
        f"{path}.emergency_stop",
    )
    stationary = ctx.required_boolean(report, "stationary", f"{path}.stationary")
    checked = ctx.required_boolean(
        report,
        "ordered_reference_checked",
        f"{path}.ordered_reference_checked",
    )
    safe = ctx.required_boolean(
        report,
        "ordered_reference_safe",
        f"{path}.ordered_reference_safe",
    )
    metric_keys = (
        "maximum_trajectory_deviation_m",
        "maximum_guide_anchor_deviation_m",
        "maximum_guide_progress_lead_m",
        "maximum_deviation_limit_m",
        "maximum_progress_lead_limit_m",
    )
    metrics = {
        key: ctx.required_number(report, key, f"{path}.{key}", minimum=0.0)
        for key in metric_keys
    }
    trajectory_duration_s = ctx.required_number(
        report,
        "trajectory_duration_s",
        f"{path}.trajectory_duration_s",
        minimum=1.0e-12,
    )
    maximum_velocity_upper_bound_mps = ctx.required_number(
        report,
        "maximum_velocity_upper_bound_mps",
        f"{path}.maximum_velocity_upper_bound_mps",
        minimum=0.0,
    )
    double_cylinder_radius_m = ctx.required_number(
        report,
        "double_cylinder_radius_m",
        f"{path}.double_cylinder_radius_m",
        minimum=1.0e-12,
    )
    double_cylinder_offset_m = ctx.required_number(
        report,
        "double_cylinder_offset_m",
        f"{path}.double_cylinder_offset_m",
        minimum=0.0,
    )
    required_any_yaw_clearance_radius_m = ctx.required_number(
        report,
        "required_any_yaw_clearance_radius_m",
        f"{path}.required_any_yaw_clearance_radius_m",
        minimum=1.0e-12,
    )
    trajectory_sample_interval_s = ctx.required_number(
        report,
        "trajectory_sample_interval_s",
        f"{path}.trajectory_sample_interval_s",
        minimum=1.0e-12,
    )
    sampling_clearance_margin_m = ctx.required_number(
        report,
        "sampling_clearance_margin_m",
        f"{path}.sampling_clearance_margin_m",
        minimum=0.0,
    )
    _expect_close(
        ctx,
        double_cylinder_radius_m,
        _GO2_X5_DOUBLE_CYLINDER_RADIUS_M,
        f"{path}.double_cylinder_radius_m",
        tolerance=1.0e-12,
    )
    _expect_close(
        ctx,
        double_cylinder_offset_m,
        _GO2_X5_DOUBLE_CYLINDER_OFFSET_M,
        f"{path}.double_cylinder_offset_m",
        tolerance=1.0e-12,
    )
    if (
        double_cylinder_radius_m is not None
        and double_cylinder_offset_m is not None
    ):
        _expect_close(
            ctx,
            required_any_yaw_clearance_radius_m,
            double_cylinder_radius_m + double_cylinder_offset_m,
            f"{path}.required_any_yaw_clearance_radius_m",
            tolerance=1.0e-12,
        )
    trajectory_total = ctx.required_integer(
        report,
        "trajectory_sample_count_total",
        f"{path}.trajectory_sample_count_total",
        minimum=2,
    )
    trajectory_truncated = ctx.required_boolean(
        report,
        "trajectory_samples_truncated",
        f"{path}.trajectory_samples_truncated",
    )
    trajectory_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "trajectory_samples_world_xyz",
            f"{path}.trajectory_samples_world_xyz",
        ),
        f"{path}.trajectory_samples_world_xyz",
    )
    if len(trajectory_samples) < 2:
        ctx.reject(
            "invalid_sample_contract",
            f"{path}.trajectory_samples_world_xyz",
            "连续 B-spline 净距证明至少需要两个全时域样本。",
            actual=len(trajectory_samples),
        )
    reference_total = ctx.required_integer(
        report,
        "ordered_reference_sample_count_total",
        f"{path}.ordered_reference_sample_count_total",
        minimum=0,
    )
    reference_truncated = ctx.required_boolean(
        report,
        "ordered_reference_samples_truncated",
        f"{path}.ordered_reference_samples_truncated",
    )
    reference_samples = _validate_point_array(
        ctx,
        ctx.field(
            report,
            "ordered_reference_samples_world_xyz",
            f"{path}.ordered_reference_samples_world_xyz",
        ),
        f"{path}.ordered_reference_samples_world_xyz",
    )
    for total, samples, truncated, sample_path in (
        (
            trajectory_total,
            trajectory_samples,
            trajectory_truncated,
            f"{path}.trajectory_samples_world_xyz",
        ),
        (
            reference_total,
            reference_samples,
            reference_truncated,
            f"{path}.ordered_reference_samples_world_xyz",
        ),
    ):
        if total is None or truncated is None:
            continue
        expected_sample_count = min(total, _PROOF_RING_CAPACITY)
        expected_truncated = total > _PROOF_RING_CAPACITY
        if truncated != expected_truncated:
            ctx.reject(
                "invalid_sample_contract",
                sample_path,
                "typed B-spline 截断标志必须且仅能表示总数超过 64。",
            )
        if len(samples) != expected_sample_count:
            ctx.reject(
                "invalid_sample_contract",
                sample_path,
                "typed B-spline 必须严格保留 min(total_count, 64) 个样本。",
                actual=len(samples),
            )
    if trajectory_duration_s is not None and trajectory_total is not None:
        expected_trajectory_total = (
            math.ceil(trajectory_duration_s / 0.01) + 1
        )
        if trajectory_total != expected_trajectory_total:
            ctx.reject(
                "invalid_sample_contract",
                f"{path}.trajectory_sample_count_total",
                "trajectory total 必须覆盖 0.01 s 全时域稠密合同。",
                actual=trajectory_total,
            )
    if len(trajectory_samples) >= 2 and trajectory_duration_s is not None:
        expected_interval = trajectory_duration_s / float(
            len(trajectory_samples) - 1
        )
        _expect_close(
            ctx,
            trajectory_sample_interval_s,
            expected_interval,
            f"{path}.trajectory_sample_interval_s",
            tolerance=1.0e-12,
        )
        if maximum_velocity_upper_bound_mps is not None:
            _expect_close(
                ctx,
                sampling_clearance_margin_m,
                (
                    maximum_velocity_upper_bound_mps
                    * expected_interval
                    * 0.5
                ),
                f"{path}.sampling_clearance_margin_m",
                tolerance=1.0e-12,
            )
    if (
        stationary is False
        and maximum_velocity_upper_bound_mps is not None
        and maximum_velocity_upper_bound_mps <= 0.0
    ):
        ctx.reject(
            "invalid_continuous_clearance_contract",
            f"{path}.maximum_velocity_upper_bound_mps",
            "运动 B-spline 的连续速度上界必须为正数。",
            actual=maximum_velocity_upper_bound_mps,
        )
    if safe is True and checked is not True:
        ctx.reject(
            "invalid_ordered_reference_contract",
            f"{path}.ordered_reference_safe",
            "ordered_reference_safe=true 要求已执行有序检查。",
        )
    if checked is True:
        if safe is not True or reference_total is not None and reference_total < 2:
            ctx.reject(
                "invalid_ordered_reference_contract",
                path,
                "已检查的有序参考必须安全且至少含两个参考样本。",
            )
        deviation_limit = metrics["maximum_deviation_limit_m"]
        lead_limit = metrics["maximum_progress_lead_limit_m"]
        for key in (
            "maximum_trajectory_deviation_m",
            "maximum_guide_anchor_deviation_m",
        ):
            value_number = metrics[key]
            if value_number is not None and deviation_limit is not None and value_number > deviation_limit + 1.0e-9:
                ctx.reject(
                    "invalid_ordered_reference_contract",
                    f"{path}.{key}",
                    "已发布轨迹的 deviation 超过有序 corridor 门限。",
                    actual=value_number,
                )
        lead = metrics["maximum_guide_progress_lead_m"]
        if lead is not None and lead_limit is not None and lead > lead_limit + 1.0e-9:
            ctx.reject(
                "invalid_ordered_reference_contract",
                f"{path}.maximum_guide_progress_lead_m",
                "已发布轨迹的 progress lead 超过门限。",
                actual=lead,
            )
    elif stationary is False and emergency_stop is False:
        ctx.reject(
            "invalid_ordered_reference_contract",
            path,
            "普通运动 B-spline 必须执行有序参考检查。",
        )
    if is_final is None:
        # 字段已由 required_boolean 报错；保留局部变量供静态检查器识别为已消费。
        pass
    return _BsplineDiagnosticEvidence(
        report=report,
        receipt_timestamp=receipt_timestamp,
        rx_sequence=rx_sequence,
        ros_time_offset_s=ros_time_offset_s,
        header_stamp_ns=header_stamp_ns,
        episode_elapsed_time_s=episode_elapsed_time_s,
        diagnostic_sequence=diagnostic_sequence,
        identity=identity,
        stationary=stationary,
        emergency_stop=emergency_stop,
        ordered_reference_checked=checked,
        ordered_reference_safe=safe,
        maximum_trajectory_deviation=metrics["maximum_trajectory_deviation_m"],
        trajectory_duration_s=trajectory_duration_s,
        maximum_velocity_upper_bound_mps=(
            maximum_velocity_upper_bound_mps
        ),
        required_any_yaw_clearance_radius_m=(
            required_any_yaw_clearance_radius_m
        ),
        trajectory_sample_interval_s=trajectory_sample_interval_s,
        sampling_clearance_margin_m=sampling_clearance_margin_m,
        trajectory_samples=trajectory_samples,
        reference_samples=reference_samples,
    )


def _validate_episode_time_binding(
    ctx: _ValidationContext,
    *,
    header_stamp_ns: int | None,
    episode_elapsed_time_s: float | None,
    report_ros_time_offset_s: float | None,
    ros_time_offset_s: float | None,
    path: str,
) -> None:
    """证明连续 ROS 时间已显式换算为本 episode 动障碍时间。"""

    if (
        header_stamp_ns is None
        or episode_elapsed_time_s is None
        or report_ros_time_offset_s is None
        or ros_time_offset_s is None
    ):
        return
    if not math.isclose(
        report_ros_time_offset_s,
        ros_time_offset_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        ctx.reject(
            "inconsistent_episode_time_offset",
            f"{path}.ros_time_offset_s",
            "typed report 的 ROS 时间原点必须等于所属 lifecycle 原点。",
            actual=report_ros_time_offset_s,
        )
    header_time_s = float(header_stamp_ns) * 1.0e-9
    expected_header_time_s = (
        report_ros_time_offset_s + episode_elapsed_time_s
    )
    if not math.isclose(
        header_time_s,
        expected_header_time_s,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        ctx.reject(
            "invalid_episode_time_binding",
            f"{path}.episode_elapsed_time_s",
            "typed report 必须证明 header ROS 时间等于 episode offset 加本地时间。",
            actual=episode_elapsed_time_s,
        )


def _validate_grid_map_observation_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
) -> tuple[
    dict[int, _GridMapDiagnosticEvidence],
    frozenset[int],
]:
    """校验过滤后点云诊断生命周期，并返回可引用事件索引。"""

    path = "$.simulation_report.grid_map_observation_lifecycle_report"
    lifecycle = ctx.required_mapping(
        evidence.simulation,
        "grid_map_observation_lifecycle_report",
        path,
    )
    _expect_exact_field(
        ctx,
        lifecycle,
        "schema",
        "grid_map_observation_lifecycle_v1",
        f"{path}.schema",
    )
    sample_count = ctx.required_integer(
        lifecycle,
        "sample_count",
        f"{path}.sample_count",
        minimum=2,
    )
    ros_time_offset_s = ctx.required_number(
        lifecycle,
        "ros_time_offset_s",
        f"{path}.ros_time_offset_s",
        minimum=0.0,
    )
    first_sequence = ctx.required_integer(
        lifecycle,
        "first_observation_sequence",
        f"{path}.first_observation_sequence",
        minimum=1,
    )
    last_sequence = ctx.required_integer(
        lifecycle,
        "last_observation_sequence",
        f"{path}.last_observation_sequence",
        minimum=1,
    )
    reset_count = ctx.required_integer(
        lifecycle,
        "sequence_reset_count",
        f"{path}.sequence_reset_count",
        minimum=0,
    )
    if reset_count not in (None, 0):
        ctx.reject(
            "diagnostic_sequence_reset",
            f"{path}.sequence_reset_count",
            "单 episode GridMap observation sequence 不允许回拨。",
            actual=reset_count,
        )
    dropped_count = ctx.required_integer(
        lifecycle,
        "dropped_diagnostic_report_count",
        f"{path}.dropped_diagnostic_report_count",
        minimum=0,
    )
    raw_reports = ctx.required_sequence(
        lifecycle,
        "diagnostic_reports",
        f"{path}.diagnostic_reports",
    )
    reports: dict[int, _GridMapDiagnosticEvidence] = {}
    previous_sequence: int | None = None
    for index, raw_report in enumerate(raw_reports):
        report_path = f"{path}.diagnostic_reports[{index}]"
        parsed = _validate_grid_map_diagnostic_report(
            ctx,
            raw_report,
            report_path,
        )
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=ros_time_offset_s,
            path=report_path,
        )
        sequence = parsed.observation_sequence
        if sequence is None:
            continue
        if previous_sequence is not None and sequence <= previous_sequence:
            ctx.reject(
                "invalid_diagnostic_lifecycle",
                f"{report_path}.observation_sequence",
                "diagnostic_reports 必须按 observation sequence 严格递增。",
                actual=sequence,
            )
        previous_sequence = sequence
        if sequence in reports:
            ctx.reject(
                "duplicate_identity",
                f"{report_path}.observation_sequence",
                "GridMap observation sequence 必须唯一。",
                actual=sequence,
            )
        reports[sequence] = parsed
    if sample_count is not None:
        expected_retained = min(
            sample_count,
            _DIAGNOSTIC_RING_CAPACITY,
        )
        expected_dropped = max(
            sample_count - _DIAGNOSTIC_RING_CAPACITY,
            0,
        )
        if len(raw_reports) != expected_retained:
            ctx.reject(
                "invalid_count",
                f"{path}.diagnostic_reports",
                "GridMap diagnostic ring 必须保留最近至多 128 条事件。",
                actual=len(raw_reports),
            )
        if dropped_count != expected_dropped:
            ctx.reject(
                "invalid_count",
                f"{path}.dropped_diagnostic_report_count",
                "GridMap diagnostic ring 丢弃计数与累计样本数不一致。",
                actual=dropped_count,
            )

    raw_transition_reports = ctx.required_sequence(
        lifecycle,
        "transition_hit_reports",
        f"{path}.transition_hit_reports",
    )
    if len(raw_transition_reports) > _PROOF_RING_CAPACITY:
        ctx.reject(
            "invalid_count",
            f"{path}.transition_hit_reports",
            "transition proof ring 不能超过 64 条。",
            actual=len(raw_transition_reports),
        )
    transition_sequences: set[int] = set()
    previous_transition_sequence: int | None = None
    for index, raw_report in enumerate(raw_transition_reports):
        report_path = f"{path}.transition_hit_reports[{index}]"
        parsed = _validate_grid_map_diagnostic_report(
            ctx,
            raw_report,
            report_path,
        )
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=ros_time_offset_s,
            path=report_path,
        )
        sequence = parsed.observation_sequence
        if sequence is None:
            continue
        if (
            previous_transition_sequence is not None
            and sequence <= previous_transition_sequence
        ):
            ctx.reject(
                "invalid_diagnostic_lifecycle",
                f"{report_path}.observation_sequence",
                "transition_hit_reports 必须按 sequence 严格递增。",
                actual=sequence,
            )
        previous_transition_sequence = sequence
        if sequence in transition_sequences:
            ctx.reject(
                "duplicate_identity",
                f"{report_path}.observation_sequence",
                "transition proof observation sequence 必须唯一。",
                actual=sequence,
            )
        transition_sequences.add(sequence)
        existing = reports.get(sequence)
        if existing is not None and existing.report != parsed.report:
            ctx.reject(
                "conflicting_diagnostic_identity",
                report_path,
                "proof ring 与 diagnostic ring 的同序号 payload 不一致。",
                actual=sequence,
            )
        reports[sequence] = parsed

    named_reports: dict[str, _GridMapDiagnosticEvidence] = {}
    for key in (
        "first_report",
        "last_report",
        "first_hit_report",
        "last_hit_report",
        "first_explicit_miss_clear_report",
        "last_explicit_miss_clear_report",
    ):
        parsed = _validate_grid_map_diagnostic_report(
            ctx,
            ctx.field(lifecycle, key, f"{path}.{key}"),
            f"{path}.{key}",
        )
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=ros_time_offset_s,
            path=f"{path}.{key}",
        )
        named_reports[key] = parsed
        if parsed.observation_sequence is not None:
            existing = reports.get(parsed.observation_sequence)
            if existing is not None and existing.report != parsed.report:
                ctx.reject(
                    "conflicting_diagnostic_identity",
                    f"{path}.{key}",
                    "同一 observation_sequence 对应了不同 typed payload。",
                    actual=parsed.observation_sequence,
                )
            reports[parsed.observation_sequence] = parsed

    # transition proof ring 只用于交叉检查；direct clear provenance 已经在
    # clear report 中逐项保存，因此旧 transition report 可以合法淘汰。
    for key in (
        "first_transition_hit_report",
        "last_transition_hit_report",
    ):
        raw_report = lifecycle.get(key)
        if raw_report is None:
            continue
        parsed = _validate_grid_map_diagnostic_report(
            ctx,
            raw_report,
            f"{path}.{key}",
        )
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=ros_time_offset_s,
            path=f"{path}.{key}",
        )
        named_reports[key] = parsed
        if parsed.observation_sequence is not None:
            existing = reports.get(parsed.observation_sequence)
            if existing is not None and existing.report != parsed.report:
                ctx.reject(
                    "conflicting_diagnostic_identity",
                    f"{path}.{key}",
                    "同一 observation_sequence 对应了不同 typed payload。",
                    actual=parsed.observation_sequence,
                )
            reports[parsed.observation_sequence] = parsed
    first_report = named_reports["first_report"]
    last_report = named_reports["last_report"]
    if first_report.observation_sequence != first_sequence:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.first_report.observation_sequence",
            "first_report 必须匹配 lifecycle 首序号。",
            actual=first_report.observation_sequence,
        )
    if last_report.observation_sequence != last_sequence:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.last_report.observation_sequence",
            "last_report 必须匹配 lifecycle 末序号。",
            actual=last_report.observation_sequence,
        )
    if (
        first_sequence is not None
        and last_sequence is not None
        and last_sequence <= first_sequence
    ):
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.last_observation_sequence",
            "动态验收的 GridMap observation sequence 必须严格推进。",
            actual=last_sequence,
        )
    if (
        sample_count is not None
        and first_sequence is not None
        and last_sequence is not None
        and last_sequence - first_sequence + 1 < sample_count
    ):
        ctx.reject(
            "invalid_count",
            f"{path}.sample_count",
            "sample_count 不能超过 observation sequence 跨度。",
            actual=sample_count,
        )
    for key in ("first_hit_report", "last_hit_report"):
        report = named_reports[key]
        if not report.hit_endpoint_count or not report.hit_samples:
            ctx.reject(
                "missing_live_point_cloud_obstacle_evidence",
                f"{path}.{key}",
                "hit 生命周期事件必须保留过滤后 hit 端点样本。",
            )
    for key in (
        "first_explicit_miss_clear_report",
        "last_explicit_miss_clear_report",
    ):
        report = named_reports[key]
        if not report.occupied_to_free_count or not report.clear_samples:
            ctx.reject(
                "missing_free_ray_ghost_clear_evidence",
                f"{path}.{key}",
                "clear 生命周期事件必须保留 explicit-miss occupied→free 样本。",
            )
    for key in (
        "first_transition_hit_report",
        "last_transition_hit_report",
    ):
        report = named_reports.get(key)
        if report is None:
            continue
        if (
            not report.free_to_occupied_transition_count
            or not report.transition_hit_samples
            or not report.transition_voxel_indices
        ):
            ctx.reject(
                "missing_free_ray_ghost_clear_evidence",
                f"{path}.{key}",
                "transition snapshot 必须保留 free→occupied point+voxel 样本。",
            )
        if (
            key == "last_transition_hit_report"
            and transition_sequences
            and report.observation_sequence not in transition_sequences
        ):
            ctx.reject(
                "missing_typed_evidence_reference",
                f"{path}.{key}.observation_sequence",
                "存在 transition proof ring 时，最后 snapshot 必须仍在 ring 中。",
                actual=report.observation_sequence,
            )
    last_summary_path = "$.simulation_report.grid_map_observation_diagnostics_last_report"
    last_summary = _validate_grid_map_diagnostic_report(
        ctx,
        ctx.field(
            evidence.simulation,
            "grid_map_observation_diagnostics_last_report",
            last_summary_path,
        ),
        last_summary_path,
    )
    _validate_episode_time_binding(
        ctx,
        header_stamp_ns=last_summary.header_stamp_ns,
        episode_elapsed_time_s=last_summary.episode_elapsed_time_s,
        report_ros_time_offset_s=last_summary.ros_time_offset_s,
        ros_time_offset_s=ros_time_offset_s,
        path=last_summary_path,
    )
    if last_summary.report != last_report.report:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            last_summary_path,
            "summary last GridMap report 必须与 lifecycle.last_report 完全一致。",
        )
    return reports, frozenset(transition_sequences)


def _validate_bspline_diagnostics_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
) -> dict[int, _BsplineDiagnosticEvidence]:
    """校验 B-spline typed 生命周期和有界完整 identity 集。"""

    path = "$.simulation_report.bspline_diagnostics_lifecycle_report"
    lifecycle = ctx.required_mapping(
        evidence.simulation,
        "bspline_diagnostics_lifecycle_report",
        path,
    )
    _expect_exact_field(
        ctx,
        lifecycle,
        "schema",
        "bspline_diagnostics_lifecycle_v1",
        f"{path}.schema",
    )
    sample_count = ctx.required_integer(
        lifecycle,
        "sample_count",
        f"{path}.sample_count",
        minimum=2,
    )
    ros_time_offset_s = ctx.required_number(
        lifecycle,
        "ros_time_offset_s",
        f"{path}.ros_time_offset_s",
        minimum=0.0,
    )
    first_sequence = ctx.required_integer(
        lifecycle,
        "first_diagnostic_sequence",
        f"{path}.first_diagnostic_sequence",
        minimum=1,
    )
    last_sequence = ctx.required_integer(
        lifecycle,
        "last_diagnostic_sequence",
        f"{path}.last_diagnostic_sequence",
        minimum=1,
    )
    reset_count = ctx.required_integer(
        lifecycle,
        "sequence_reset_count",
        f"{path}.sequence_reset_count",
        minimum=0,
    )
    if reset_count not in (None, 0):
        ctx.reject(
            "diagnostic_sequence_reset",
            f"{path}.sequence_reset_count",
            "单 episode B-spline diagnostic sequence 不允许回拨。",
            actual=reset_count,
        )
    dropped_count = ctx.required_integer(
        lifecycle,
        "dropped_diagnostic_report_count",
        f"{path}.dropped_diagnostic_report_count",
        minimum=0,
    )
    distinct_count = ctx.required_integer(
        lifecycle,
        "distinct_trajectory_identity_count",
        f"{path}.distinct_trajectory_identity_count",
        minimum=2,
    )
    raw_identities = ctx.required_sequence(
        lifecycle,
        "trajectory_identities",
        f"{path}.trajectory_identities",
    )
    lifecycle_identities: set[tuple[int, int, int, int]] = set()
    for index, raw_identity in enumerate(raw_identities):
        identity = _validate_full_trajectory_identity(
            ctx,
            raw_identity,
            f"{path}.trajectory_identities[{index}]",
            path_stamp_ns=None,
        )
        if identity is not None:
            lifecycle_identities.add(identity)
    if distinct_count is not None and len(raw_identities) != distinct_count:
        ctx.reject(
            "invalid_count",
            f"{path}.trajectory_identities",
            "trajectory identity 数组长度必须等于 distinct count。",
            actual=len(raw_identities),
        )
    if len(lifecycle_identities) != len(raw_identities):
        ctx.reject(
            "duplicate_identity",
            f"{path}.trajectory_identities",
            "B-spline lifecycle trajectory identity 必须唯一。",
        )
    raw_reports = ctx.required_sequence(
        lifecycle,
        "diagnostic_reports",
        f"{path}.diagnostic_reports",
    )
    reports: dict[int, _BsplineDiagnosticEvidence] = {}
    previous_sequence: int | None = None
    last_retained_report: _BsplineDiagnosticEvidence | None = None
    for index, raw_report in enumerate(raw_reports):
        parsed = _validate_bspline_diagnostic_report(
            ctx,
            raw_report,
            f"{path}.diagnostic_reports[{index}]",
            # 生命周期允许包含 global-replan 前代；aggregate/final report 再绑定。
            path_stamp_ns=None,
        )
        last_retained_report = parsed
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=ros_time_offset_s,
            path=f"{path}.diagnostic_reports[{index}]",
        )
        sequence = parsed.diagnostic_sequence
        if sequence is not None:
            if previous_sequence is not None and sequence <= previous_sequence:
                ctx.reject(
                    "invalid_diagnostic_lifecycle",
                    f"{path}.diagnostic_reports[{index}].diagnostic_sequence",
                    "diagnostic_reports 必须按 sequence 严格递增。",
                    actual=sequence,
                )
            previous_sequence = sequence
            if sequence in reports:
                ctx.reject(
                    "duplicate_identity",
                    f"{path}.diagnostic_reports[{index}].diagnostic_sequence",
                    "B-spline diagnostic sequence 必须唯一。",
                    actual=sequence,
                )
            reports[sequence] = parsed
        if (
            parsed.identity is not None
            and parsed.identity not in lifecycle_identities
        ):
            ctx.reject(
                "wrong_identity",
                f"{path}.diagnostic_reports[{index}].identity",
                "typed B-spline report identity 未收录于 lifecycle identity 集。",
            )
    if sample_count is not None:
        expected_retained = min(
            sample_count,
            _DIAGNOSTIC_RING_CAPACITY,
        )
        expected_dropped = max(
            sample_count - _DIAGNOSTIC_RING_CAPACITY,
            0,
        )
        if len(raw_reports) != expected_retained:
            ctx.reject(
                "invalid_count",
                f"{path}.diagnostic_reports",
                "B-spline diagnostic ring 必须保留最近至多 128 条事件。",
                actual=len(raw_reports),
            )
        if dropped_count != expected_dropped:
            ctx.reject(
                "invalid_count",
                f"{path}.dropped_diagnostic_report_count",
                "B-spline diagnostic ring 丢弃计数与累计样本数不一致。",
                actual=dropped_count,
            )
    first_report = _validate_bspline_diagnostic_report(
        ctx,
        ctx.field(lifecycle, "first_report", f"{path}.first_report"),
        f"{path}.first_report",
        path_stamp_ns=None,
    )
    _validate_episode_time_binding(
        ctx,
        header_stamp_ns=first_report.header_stamp_ns,
        episode_elapsed_time_s=first_report.episode_elapsed_time_s,
        report_ros_time_offset_s=first_report.ros_time_offset_s,
        ros_time_offset_s=ros_time_offset_s,
        path=f"{path}.first_report",
    )
    last_report = _validate_bspline_diagnostic_report(
        ctx,
        ctx.field(lifecycle, "last_report", f"{path}.last_report"),
        f"{path}.last_report",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    _validate_episode_time_binding(
        ctx,
        header_stamp_ns=last_report.header_stamp_ns,
        episode_elapsed_time_s=last_report.episode_elapsed_time_s,
        report_ros_time_offset_s=last_report.ros_time_offset_s,
        ros_time_offset_s=ros_time_offset_s,
        path=f"{path}.last_report",
    )
    for label, parsed in (
        ("first_report", first_report),
        ("last_report", last_report),
    ):
        if (
            parsed.identity is not None
            and parsed.identity not in lifecycle_identities
        ):
            ctx.reject(
                "wrong_identity",
                f"{path}.{label}.identity",
                "命名 B-spline report identity 未收录于 lifecycle identity 集。",
            )
        sequence = parsed.diagnostic_sequence
        if sequence is None:
            continue
        existing = reports.get(sequence)
        if existing is not None and existing.report != parsed.report:
            ctx.reject(
                "conflicting_diagnostic_identity",
                f"{path}.{label}",
                "命名 snapshot 与 diagnostic ring 的同序号 payload 不一致。",
                actual=sequence,
            )
        reports[sequence] = parsed
    if first_report.diagnostic_sequence != first_sequence:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.first_report.diagnostic_sequence",
            "first_report 必须匹配 lifecycle 首序号。",
            actual=first_report.diagnostic_sequence,
        )
    if last_report.diagnostic_sequence != last_sequence:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.last_report.diagnostic_sequence",
            "last_report 必须匹配 lifecycle 末序号。",
            actual=last_report.diagnostic_sequence,
        )
    if (
        last_sequence is not None
        and (
            last_retained_report is None
            or last_retained_report.report != last_report.report
        )
    ):
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.last_report",
            "last_report 必须等于 diagnostic ring 末事件。",
        )
    if first_sequence is not None and last_sequence is not None and last_sequence <= first_sequence:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            f"{path}.last_diagnostic_sequence",
            "动态验收必须包含严格递增的多条 B-spline diagnostics。",
            actual=last_sequence,
        )
    last_summary_path = "$.simulation_report.bspline_diagnostics_last_report"
    last_summary = _validate_bspline_diagnostic_report(
        ctx,
        ctx.field(
            evidence.simulation,
            "bspline_diagnostics_last_report",
            last_summary_path,
        ),
        last_summary_path,
        path_stamp_ns=evidence.path_stamp_ns,
    )
    _validate_episode_time_binding(
        ctx,
        header_stamp_ns=last_summary.header_stamp_ns,
        episode_elapsed_time_s=last_summary.episode_elapsed_time_s,
        report_ros_time_offset_s=last_summary.ros_time_offset_s,
        ros_time_offset_s=ros_time_offset_s,
        path=last_summary_path,
    )
    if last_summary.report != last_report.report:
        ctx.reject(
            "invalid_diagnostic_lifecycle",
            last_summary_path,
            "summary last B-spline report 必须与 lifecycle.last_report 完全一致。",
        )
    return reports


def _dynamic_obstacle_for_id(
    plan: DynamicObstaclePlan,
    obstacle_id: str | None,
) -> DynamicObstacleSpec | None:
    if obstacle_id is None:
        return None
    for obstacle in plan.obstacles:
        if obstacle.obstacle_id == obstacle_id:
            return obstacle
    return None


def _point_to_oriented_cuboid_xy_clearance(
    point: tuple[float, float, float],
    obstacle: DynamicObstacleSpec,
    state: DynamicObstacleState,
) -> float:
    """复算 runtime 使用的轨迹中心到旋转 cuboid 的 XY 净距。"""

    delta_x = point[0] - state.position_world_xyz[0]
    delta_y = point[1] - state.position_world_xyz[1]
    cosine = math.cos(obstacle.yaw_rad)
    sine = math.sin(obstacle.yaw_rad)
    local_x = cosine * delta_x + sine * delta_y
    local_y = -sine * delta_x + cosine * delta_y
    outside_x = max(abs(local_x) - 0.5 * obstacle.size_xyz_m[0], 0.0)
    outside_y = max(abs(local_y) - 0.5 * obstacle.size_xyz_m[1], 0.0)
    return math.hypot(outside_x, outside_y)


def _point_inside_oriented_cuboid(
    point: tuple[float, float, float],
    obstacle: DynamicObstacleSpec,
    state: DynamicObstacleState,
    *,
    tolerance_m: float,
) -> bool:
    """按闭包判断点是否落在当前障碍 cuboid（允许有界测量容差）。"""

    delta_x = point[0] - state.position_world_xyz[0]
    delta_y = point[1] - state.position_world_xyz[1]
    cosine = math.cos(obstacle.yaw_rad)
    sine = math.sin(obstacle.yaw_rad)
    local = (
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
        point[2] - state.position_world_xyz[2],
    )
    return all(
        abs(local[axis]) <= 0.5 * obstacle.size_xyz_m[axis] + tolerance_m
        for axis in range(3)
    )


def _validate_grid_leaf_reference(
    ctx: _ValidationContext,
    leaf: Mapping[str, Any],
    path: str,
    reports: Mapping[int, _GridMapDiagnosticEvidence],
    *,
    sequence_key: str = "observation_sequence",
    header_key: str = "header",
) -> _GridMapDiagnosticEvidence | None:
    """把 aggregate grid 叶子严格 join 到 typed lifecycle payload。"""

    _expect_exact_field(
        ctx,
        leaf,
        "source",
        "ros2_scan_grid_map_observation_diagnostics",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        leaf,
        "topic",
        "/planning/grid_map_observation_diagnostics",
        f"{path}.topic",
    )
    sequence = ctx.required_integer(
        leaf,
        sequence_key,
        f"{path}.{sequence_key}",
        minimum=1,
    )
    header = ctx.required_mapping(leaf, header_key, f"{path}.{header_key}")
    _expect_exact_field(ctx, header, "frame_id", "world", f"{path}.{header_key}.frame_id")
    header_stamp = _validate_ros_stamp(
        ctx,
        header,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{path}.{header_key}",
    )
    report = reports.get(sequence) if sequence is not None else None
    if report is None and sequence is not None:
        ctx.reject(
            "missing_typed_evidence_reference",
            f"{path}.{sequence_key}",
            "aggregate observation sequence 未在 typed lifecycle 事件中保留。",
            actual=sequence,
        )
        return None
    if report is not None and header_stamp != report.header_stamp_ns:
        ctx.reject(
            "wrong_identity",
            f"{path}.{header_key}",
            "aggregate header stamp 与 typed GridMap report 不一致。",
            actual=header_stamp,
        )
    return report


def _validate_bspline_leaf_reference(
    ctx: _ValidationContext,
    leaf: Mapping[str, Any],
    path: str,
    reports: Mapping[int, _BsplineDiagnosticEvidence],
    *,
    path_stamp_ns: int | None,
    sequence_key: str = "diagnostic_sequence",
    header_key: str = "header",
    identity_key: str = "identity",
) -> _BsplineDiagnosticEvidence | None:
    """把 aggregate B-spline 叶子 join 到 typed diagnostics 与四元 identity。"""

    _expect_exact_field(
        ctx,
        leaf,
        "source",
        "ros2_scan_bspline_diagnostics",
        f"{path}.source",
    )
    _expect_exact_field(
        ctx,
        leaf,
        "topic",
        "/planning/bspline_diagnostics",
        f"{path}.topic",
    )
    sequence = ctx.required_integer(
        leaf,
        sequence_key,
        f"{path}.{sequence_key}",
        minimum=1,
    )
    header = ctx.required_mapping(leaf, header_key, f"{path}.{header_key}")
    _expect_exact_field(ctx, header, "frame_id", "world", f"{path}.{header_key}.frame_id")
    header_stamp = _validate_ros_stamp(
        ctx,
        header,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{path}.{header_key}",
    )
    identity = _validate_full_trajectory_identity(
        ctx,
        ctx.field(leaf, identity_key, f"{path}.{identity_key}"),
        f"{path}.{identity_key}",
        path_stamp_ns=path_stamp_ns,
    )
    report = reports.get(sequence) if sequence is not None else None
    if report is None and sequence is not None:
        ctx.reject(
            "missing_typed_evidence_reference",
            f"{path}.{sequence_key}",
            "aggregate diagnostic sequence 未在 typed B-spline lifecycle 中保留。",
            actual=sequence,
        )
        return None
    if report is not None:
        if header_stamp != report.header_stamp_ns:
            ctx.reject(
                "wrong_identity",
                f"{path}.{header_key}",
                "aggregate header stamp 与 typed B-spline report 不一致。",
                actual=header_stamp,
            )
        if identity != report.identity:
            ctx.reject(
                "wrong_identity",
                f"{path}.{identity_key}",
                "aggregate trajectory identity 与 typed B-spline report 不一致。",
                actual=list(identity) if identity is not None else None,
            )
    return report


def _controller_accepted_identity_set(
    evidence: _CommonEvidence,
) -> set[tuple[int, int, int, int]]:
    lifecycle = evidence.simulation.get("scan_controller_status_lifecycle_report")
    if not isinstance(lifecycle, Mapping):
        return set()
    raw_identities = lifecycle.get("accepted_trajectory_identities")
    if not isinstance(raw_identities, Sequence) or isinstance(
        raw_identities,
        (str, bytes, bytearray),
    ):
        return set()
    return {
        identity
        for raw_identity in raw_identities
        if (identity := _mapping_identity_tuple(raw_identity)) is not None
    }


def _validate_dynamic_navigation_evidence(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    plan: DynamicObstaclePlan,
) -> None:
    """验证动态闭环五类聚合证据，而不是信任聚合布尔值。"""

    required_groups = (
        (
            (
                "grid_map_observation_diagnostics_last_report",
                "grid_map_observation_lifecycle_report",
                "dynamic_navigation_evidence_report",
            ),
            "missing_live_point_cloud_obstacle_evidence",
            "缺少过滤后 GridMap hit 的 typed 生命周期或动态聚合证据。",
        ),
        (
            (
                "bspline_diagnostics_last_report",
                "bspline_diagnostics_lifecycle_report",
                "dynamic_navigation_evidence_report",
            ),
            "missing_detour_geometry_evidence",
            "缺少 accepted B-spline 绕行/净距的 typed 生命周期或动态聚合证据。",
        ),
        (
            (
                "grid_map_observation_lifecycle_report",
                "dynamic_navigation_evidence_report",
            ),
            "missing_free_ray_ghost_clear_evidence",
            "缺少 explicit-miss occupied→free 的 typed 生命周期或动态聚合证据。",
        ),
    )
    for keys, code, message in required_groups:
        if any(key not in evidence.simulation for key in keys):
            ctx.reject(code, "$.simulation_report", message)

    grid_reports, transition_hit_sequences = (
        _validate_grid_map_observation_lifecycle(ctx, evidence)
    )
    bspline_reports = _validate_bspline_diagnostics_lifecycle(ctx, evidence)
    bridge_path = "$.simulation_report.navigation_ros2_bridge_report"
    bridge = ctx.required_mapping(
        evidence.simulation,
        "navigation_ros2_bridge_report",
        bridge_path,
    )
    bridge_exact = {
        "grid_map_diagnostics_topic": "/planning/grid_map_observation_diagnostics",
        "bspline_diagnostics_topic": "/planning/bspline_diagnostics",
        "grid_map_diagnostics_subscription_enabled": True,
        "bspline_diagnostics_subscription_enabled": True,
    }
    for key, expected in bridge_exact.items():
        ctx.expect(ctx.field(bridge, key, f"{bridge_path}.{key}"), expected, f"{bridge_path}.{key}")

    path = "$.simulation_report.dynamic_navigation_evidence_report"
    aggregate = ctx.required_mapping(
        evidence.simulation,
        "dynamic_navigation_evidence_report",
        path,
    )
    _expect_exact_field(
        ctx,
        aggregate,
        "schema",
        "dynamic_navigation_evidence_v1",
        f"{path}.schema",
    )
    aggregate_ros_time_offset_s = ctx.required_number(
        aggregate,
        "ros_time_offset_s",
        f"{path}.ros_time_offset_s",
        minimum=0.0,
    )
    _expect_true_field(ctx, aggregate, "enabled", f"{path}.enabled")
    _expect_true_field(ctx, aggregate, "verified", f"{path}.verified")
    obstacle_ids = _validate_string_array(
        ctx,
        ctx.field(aggregate, "obstacle_ids", f"{path}.obstacle_ids"),
        f"{path}.obstacle_ids",
    )
    expected_ids = tuple(obstacle.obstacle_id for obstacle in plan.obstacles)
    if obstacle_ids != expected_ids:
        ctx.reject(
            "wrong_identity",
            f"{path}.obstacle_ids",
            "dynamic aggregate 障碍 ID 必须按任务定义完整列出。",
            actual=list(obstacle_ids),
        )

    lifecycle = evidence.simulation.get("dynamic_obstacle_lifecycle_report")
    first_elapsed_s = (
        float(lifecycle["first_elapsed_time_s"])
        if isinstance(lifecycle, Mapping)
        and isinstance(lifecycle.get("first_elapsed_time_s"), (int, float))
        and not isinstance(lifecycle.get("first_elapsed_time_s"), bool)
        else None
    )
    last_elapsed_s = (
        float(lifecycle["last_elapsed_time_s"])
        if isinstance(lifecycle, Mapping)
        and isinstance(lifecycle.get("last_elapsed_time_s"), (int, float))
        and not isinstance(lifecycle.get("last_elapsed_time_s"), bool)
        else None
    )
    grid_lifecycle = evidence.simulation.get(
        "grid_map_observation_lifecycle_report"
    )
    bspline_lifecycle = evidence.simulation.get(
        "bspline_diagnostics_lifecycle_report"
    )
    raw_grid_offset = (
        grid_lifecycle.get("ros_time_offset_s")
        if isinstance(grid_lifecycle, Mapping)
        else None
    )
    raw_dynamic_offset = (
        lifecycle.get("ros_time_offset_s")
        if isinstance(lifecycle, Mapping)
        else None
    )
    raw_bspline_offset = (
        bspline_lifecycle.get("ros_time_offset_s")
        if isinstance(bspline_lifecycle, Mapping)
        else None
    )
    grid_offset = (
        float(raw_grid_offset)
        if isinstance(raw_grid_offset, (int, float))
        and not isinstance(raw_grid_offset, bool)
        and math.isfinite(float(raw_grid_offset))
        else None
    )
    bspline_offset = (
        float(raw_bspline_offset)
        if isinstance(raw_bspline_offset, (int, float))
        and not isinstance(raw_bspline_offset, bool)
        and math.isfinite(float(raw_bspline_offset))
        else None
    )
    dynamic_offset = (
        float(raw_dynamic_offset)
        if isinstance(raw_dynamic_offset, (int, float))
        and not isinstance(raw_dynamic_offset, bool)
        and math.isfinite(float(raw_dynamic_offset))
        else None
    )
    if (
        grid_offset is not None
        and bspline_offset is not None
        and not math.isclose(
            grid_offset,
            bspline_offset,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        ctx.reject(
            "inconsistent_episode_time_offset",
            "$.simulation_report.bspline_diagnostics_lifecycle_report.ros_time_offset_s",
            "GridMap 与 B-spline typed 生命周期必须共享同一 episode ROS 时间原点。",
            actual=bspline_offset,
        )
    for offset, offset_path in (
        (
            dynamic_offset,
            "$.simulation_report.dynamic_obstacle_lifecycle_report.ros_time_offset_s",
        ),
        (
            grid_offset,
            "$.simulation_report.grid_map_observation_lifecycle_report.ros_time_offset_s",
        ),
        (
            bspline_offset,
            "$.simulation_report.bspline_diagnostics_lifecycle_report.ros_time_offset_s",
        ),
    ):
        if (
            aggregate_ros_time_offset_s is not None
            and offset is not None
            and not math.isclose(
                aggregate_ros_time_offset_s,
                offset,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            ctx.reject(
                "inconsistent_episode_time_offset",
                offset_path,
                "typed lifecycle 与动态 aggregate 必须共享同一 episode ROS 时间原点。",
                actual=offset,
            )
    accepted_identities = _controller_accepted_identity_set(evidence)

    # 1) 过滤后的 /cloud_registered 必须在 typed GridMap report 中留下
    #    与同一 ROS 仿真时间动态 cuboid 对齐的 hit。
    post_path = f"{path}.post_filter_hit"
    post = ctx.required_mapping(aggregate, "post_filter_hit", post_path)
    _expect_true_field(ctx, post, "verified", f"{post_path}.verified")
    post_report = _validate_grid_leaf_reference(
        ctx,
        post,
        post_path,
        grid_reports,
    )
    post_elapsed = (
        post_report.episode_elapsed_time_s
        if post_report is not None
        else None
    )
    if post_report is not None:
        claimed_hit_count = ctx.required_integer(
            post,
            "hit_endpoint_count",
            f"{post_path}.hit_endpoint_count",
            minimum=1,
        )
        if claimed_hit_count != post_report.hit_endpoint_count:
            ctx.reject(
                "invalid_count",
                f"{post_path}.hit_endpoint_count",
                "aggregate hit count 与 typed GridMap report 不一致。",
                actual=claimed_hit_count,
            )
        claimed_samples = _validate_point_array(
            ctx,
            ctx.field(
                post,
                "hit_endpoint_samples_world_xyz",
                f"{post_path}.hit_endpoint_samples_world_xyz",
            ),
            f"{post_path}.hit_endpoint_samples_world_xyz",
        )
        if claimed_samples != post_report.hit_samples:
            ctx.reject(
                "wrong_geometry_reference",
                f"{post_path}.hit_endpoint_samples_world_xyz",
                "aggregate hit 样本必须逐点复制 typed GridMap report。",
            )
        typed_matches = post_report.report.get(
            "dynamic_obstacle_hit_matches"
        )
        aggregate_matches_raw = ctx.required_sequence(
            post,
            "dynamic_obstacle_hit_matches",
            f"{post_path}.dynamic_obstacle_hit_matches",
        )
        if aggregate_matches_raw != typed_matches:
            ctx.reject(
                "wrong_typed_evidence_reference",
                f"{post_path}.dynamic_obstacle_hit_matches",
                "aggregate hit matches 必须来自同一 typed GridMap report。",
            )
        if not aggregate_matches_raw:
            ctx.reject(
                "missing_live_point_cloud_obstacle_evidence",
                f"{post_path}.dynamic_obstacle_hit_matches",
                "过滤后点云没有动态障碍 hit match。",
            )
        for index, raw_match in enumerate(aggregate_matches_raw):
            match_path = f"{post_path}.dynamic_obstacle_hit_matches[{index}]"
            match = ctx.mapping(raw_match, match_path)
            obstacle_id = ctx.required_string(
                match,
                "obstacle_id",
                f"{match_path}.obstacle_id",
                nonempty=True,
            )
            obstacle = _dynamic_obstacle_for_id(plan, obstacle_id)
            if obstacle is None:
                ctx.reject(
                    "wrong_identity",
                    f"{match_path}.obstacle_id",
                    "post-filter hit 引用了任务外障碍。",
                    actual=obstacle_id,
                )
                continue
            point = _validate_vector(
                ctx,
                ctx.field(match, "point_world_xyz", f"{match_path}.point_world_xyz"),
                f"{match_path}.point_world_xyz",
                length=3,
            )
            voxel_index = _validate_integer_vector(
                ctx,
                ctx.field(
                    match,
                    "voxel_index_xyz",
                    f"{match_path}.voxel_index_xyz",
                ),
                f"{match_path}.voxel_index_xyz",
                length=3,
            )
            match_resolution = ctx.required_number(
                match,
                "map_resolution_m",
                f"{match_path}.map_resolution_m",
                minimum=1.0e-9,
            )
            _expect_close(
                ctx,
                match_resolution,
                post_report.map_resolution_m,
                f"{match_path}.map_resolution_m",
                tolerance=1.0e-12,
            )
            tolerance = ctx.required_number(
                match,
                "association_tolerance_m",
                f"{match_path}.association_tolerance_m",
                minimum=0.0,
            )
            ctx.required_number(
                match,
                "point_to_obstacle_xy_clearance_m",
                f"{match_path}.point_to_obstacle_xy_clearance_m",
                minimum=0.0,
            )
            if tolerance is not None and tolerance > _MAX_POST_FILTER_HIT_TOLERANCE_M:
                ctx.reject(
                    "invalid_cloud_evidence",
                    f"{match_path}.association_tolerance_m",
                    "过滤后点云命中关联容差不能超过运行时固定的 5 cm。",
                    actual=tolerance,
                )
            state = (
                obstacle.state_at(post_elapsed)
                if post_elapsed is not None
                else None
            )
            if state is not None:
                _validate_dynamic_state(
                    ctx,
                    ctx.field(match, "obstacle_state", f"{match_path}.obstacle_state"),
                    f"{match_path}.obstacle_state",
                    expected=state.to_dict(),
                )
            if (
                point is not None
                and voxel_index is not None
                and (point, voxel_index)
                not in tuple(
                    zip(
                        post_report.hit_samples,
                        post_report.hit_voxel_indices,
                    )
                )
            ):
                ctx.reject(
                    "wrong_geometry_reference",
                    f"{match_path}.voxel_index_xyz",
                    "动态障碍 match 必须引用同一 typed hit 的 point+voxel 对。",
                    actual=list(voxel_index),
                )
            if (
                point is not None
                and state is not None
                and tolerance is not None
                and not _point_inside_oriented_cuboid(
                    point,
                    obstacle,
                    state,
                    tolerance_m=tolerance,
                )
            ):
                ctx.reject(
                    "missing_live_point_cloud_obstacle_evidence",
                    f"{match_path}.point_world_xyz",
                    "过滤后 hit 样本不在该时刻推车 OBB 内。",
                    actual=list(point),
                )

    # 2) typed B-spline 必须保持 ordered corridor、达到运行时固定绕行门限，
    #    且其完整 identity 确实被 controller 接受。
    detour_path = f"{path}.ordered_detour"
    detour = ctx.required_mapping(aggregate, "ordered_detour", detour_path)
    _expect_true_field(ctx, detour, "verified", f"{detour_path}.verified")
    detour_report = _validate_bspline_leaf_reference(
        ctx,
        detour,
        detour_path,
        bspline_reports,
        path_stamp_ns=None,
    )
    detour_controller_header_stamp_ns: int | None = None
    detour_controller_receipt_timestamp: float | None = None
    detour_tracking_header_stamp_ns: int | None = None
    detour_tracking_receipt_timestamp: float | None = None
    detour_policy_write_timestamp: float | None = None
    detour_causal_clear_match: Mapping[str, Any] | None = None
    detour_elapsed = (
        detour_report.episode_elapsed_time_s
        if detour_report is not None
        else None
    )
    if detour_report is not None:
        if (
            detour_report.stationary is not False
            or detour_report.emergency_stop is not False
            or detour_report.ordered_reference_checked is not True
            or detour_report.ordered_reference_safe is not True
        ):
            ctx.reject(
                "missing_detour_geometry_evidence",
                detour_path,
                "detour 必须来自普通运动且通过 ordered reference 门的 B-spline。",
            )
        if detour_report.identity not in accepted_identities:
            ctx.reject(
                "controller_identity_not_accepted",
                f"{detour_path}.identity",
                "detour B-spline identity 未被 controller 接受。",
                actual=list(detour_report.identity) if detour_report.identity is not None else None,
            )
        typed = detour_report.report
        _expect_true_field(
            ctx,
            typed,
            "dynamic_obstacle_relevant",
            f"{detour_path}.typed.dynamic_obstacle_relevant",
        )
        _expect_true_field(
            ctx,
            typed,
            "ordered_detour_candidate",
            f"{detour_path}.typed.ordered_detour_candidate",
        )
        _expect_true_field(
            ctx,
            typed,
            "dynamic_obstacle_reference_obstructed",
            (
                f"{detour_path}.typed."
                "dynamic_obstacle_reference_obstructed"
            ),
        )
        copied_fields = (
            "maximum_trajectory_deviation_m",
            "maximum_deviation_limit_m",
            "maximum_guide_progress_lead_m",
            "maximum_progress_lead_limit_m",
            "trajectory_samples_world_xyz",
            "ordered_reference_samples_world_xyz",
            "dynamic_obstacle_clearances",
            "dynamic_obstacle_reference_obstructed",
        )
        for key in copied_fields:
            if ctx.field(detour, key, f"{detour_path}.{key}") != typed.get(key):
                ctx.reject(
                    "wrong_typed_evidence_reference",
                    f"{detour_path}.{key}",
                    "ordered_detour 字段必须来自同一 typed B-spline report。",
                )
        controller_accepted = ctx.required_boolean(
            detour,
            "controller_identity_accepted",
            f"{detour_path}.controller_identity_accepted",
        )
        if controller_accepted is not True:
            ctx.reject(
                "controller_identity_not_accepted",
                f"{detour_path}.controller_identity_accepted",
                "detour identity 必须被 controller 接受。",
            )
        controller_status_path = f"{detour_path}.controller_accepted_status"
        controller_status = ctx.required_mapping(
            detour,
            "controller_accepted_status",
            controller_status_path,
        )
        _validate_controller_status(
            ctx,
            controller_status,
            controller_status_path,
            path_stamp_ns=None,
        )
        detour_controller_receipt_timestamp = ctx.required_number(
            controller_status,
            "receipt_timestamp",
            f"{controller_status_path}.receipt_timestamp",
            minimum=0.0,
        )
        controller_header = ctx.required_mapping(
            controller_status,
            "header",
            f"{controller_status_path}.header",
        )
        detour_controller_header_stamp_ns = ctx.required_integer(
            controller_header,
            "stamp_ns",
            f"{controller_status_path}.header.stamp_ns",
            minimum=1,
        )
        controller_identity = _mapping_identity_tuple(
            controller_status.get("identity")
        )
        if (
            controller_status.get("accepted") is not True
            or controller_identity != detour_report.identity
        ):
            ctx.reject(
                "controller_identity_not_accepted",
                controller_status_path,
                "detour 必须绑定 exact identity 且 accepted=true 的 ControllerStatus。",
            )
        controller_lifecycle = evidence.simulation.get(
            "scan_controller_status_lifecycle_report"
        )
        accepted_status_reports = (
            controller_lifecycle.get("accepted_status_reports")
            if isinstance(controller_lifecycle, Mapping)
            else None
        )
        if (
            not isinstance(accepted_status_reports, Sequence)
            or isinstance(
                accepted_status_reports,
                (str, bytes, bytearray),
            )
            or not any(
                controller_status == retained
                for retained in accepted_status_reports
            )
        ):
            ctx.reject(
                "wrong_controller_acceptance_reference",
                controller_status_path,
                "detour ControllerStatus snapshot 必须来自 bounded accepted status ring。",
            )
        if (
            detour_controller_header_stamp_ns is not None
            and detour_report.header_stamp_ns is not None
            and detour_controller_header_stamp_ns
            < detour_report.header_stamp_ns
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{controller_status_path}.header.stamp_ns",
                "controller 接受 header 不能早于 detour diagnostics header。",
                actual=detour_controller_header_stamp_ns,
            )
        if (
            detour_controller_receipt_timestamp is not None
            and detour_report.receipt_timestamp is not None
            and detour_controller_receipt_timestamp
            < detour_report.receipt_timestamp - 1.0e-9
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{controller_status_path}.receipt_timestamp",
                "controller 接收时间不能早于 detour diagnostics 接收时间。",
                actual=detour_controller_receipt_timestamp,
            )
        tracking_status_path = f"{detour_path}.controller_tracking_status"
        tracking_status = ctx.required_mapping(
            detour,
            "controller_tracking_status",
            tracking_status_path,
        )
        _validate_controller_status(
            ctx,
            tracking_status,
            tracking_status_path,
            path_stamp_ns=None,
        )
        detour_tracking_receipt_timestamp = ctx.required_number(
            tracking_status,
            "receipt_timestamp",
            f"{tracking_status_path}.receipt_timestamp",
            minimum=0.0,
        )
        tracking_header = ctx.required_mapping(
            tracking_status,
            "header",
            f"{tracking_status_path}.header",
        )
        detour_tracking_header_stamp_ns = ctx.required_integer(
            tracking_header,
            "stamp_ns",
            f"{tracking_status_path}.header.stamp_ns",
            minimum=1,
        )
        if (
            tracking_status.get("state") != 10
            or tracking_status.get("trajectory_valid") is not True
            or _mapping_identity_tuple(tracking_status.get("identity"))
            != detour_report.identity
        ):
            ctx.reject(
                "missing_detour_execution_evidence",
                tracking_status_path,
                "detour identity 必须进入 controller identity-valid TRACKING。",
            )
        tracking_status_reports = (
            controller_lifecycle.get("tracking_status_reports")
            if isinstance(controller_lifecycle, Mapping)
            else None
        )
        if (
            not isinstance(tracking_status_reports, Sequence)
            or isinstance(
                tracking_status_reports,
                (str, bytes, bytearray),
            )
            or not any(
                tracking_status == retained
                for retained in tracking_status_reports
            )
        ):
            ctx.reject(
                "wrong_controller_acceptance_reference",
                tracking_status_path,
                "detour TRACKING snapshot 必须来自 bounded controller ring。",
            )
        if (
            detour_tracking_header_stamp_ns is not None
            and detour_controller_header_stamp_ns is not None
            and detour_tracking_header_stamp_ns
            < detour_controller_header_stamp_ns
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{tracking_status_path}.header.stamp_ns",
                "detour TRACKING header 不能早于接受 header。",
                actual=detour_tracking_header_stamp_ns,
            )
        if (
            detour_tracking_receipt_timestamp is not None
            and detour_controller_receipt_timestamp is not None
            and detour_tracking_receipt_timestamp
            < detour_controller_receipt_timestamp - 1.0e-9
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{tracking_status_path}.receipt_timestamp",
                "detour TRACKING 接收时间不能早于接受状态。",
                actual=detour_tracking_receipt_timestamp,
            )
        _expect_true_field(
            ctx,
            detour,
            "policy_identity_valid_tracking",
            f"{detour_path}.policy_identity_valid_tracking",
        )
        policy_write_path = (
            f"{detour_path}.policy_identity_verified_tracking_write"
        )
        policy_write = ctx.required_mapping(
            detour,
            "policy_identity_verified_tracking_write",
            policy_write_path,
        )
        _validate_consumed_tracking_evidence(
            ctx,
            policy_write,
            policy_write_path,
            pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
            path_stamp_ns=(
                detour_report.identity[0]
                if detour_report.identity is not None
                else None
            ),
        )
        detour_policy_write_timestamp = ctx.required_number(
            policy_write,
            "timestamp",
            f"{policy_write_path}.timestamp",
            minimum=0.0,
        )
        policy_snapshot = ctx.required_mapping(
            policy_write,
            "scan_controller_status_snapshot",
            f"{policy_write_path}.scan_controller_status_snapshot",
        )
        if policy_snapshot != tracking_status:
            ctx.reject(
                "wrong_controller_acceptance_reference",
                f"{policy_write_path}.scan_controller_status_snapshot",
                "detour policy write 必须快照同一 exact TRACKING ControllerStatus。",
            )
        policy_lifecycle = evidence.simulation.get(
            "navigation_policy_gate_lifecycle_report"
        )
        policy_write_reports = (
            policy_lifecycle.get(
                "identity_verified_tracking_write_reports"
            )
            if isinstance(policy_lifecycle, Mapping)
            else None
        )
        if (
            not isinstance(policy_write_reports, Sequence)
            or isinstance(policy_write_reports, (str, bytes, bytearray))
            or not any(
                policy_write == retained
                for retained in policy_write_reports
            )
        ):
            ctx.reject(
                "wrong_policy_write_reference",
                policy_write_path,
                "detour policy write 必须来自 bounded identity-valid TRACKING ring。",
            )
        policy_time_lower_bounds = tuple(
            value
            for value in (
                detour_report.receipt_timestamp,
                detour_tracking_receipt_timestamp,
            )
            if value is not None
        )
        minimum_policy_timestamp = (
            max(policy_time_lower_bounds)
            if policy_time_lower_bounds
            else None
        )
        if (
            detour_policy_write_timestamp is not None
            and minimum_policy_timestamp is not None
            and detour_policy_write_timestamp
            < minimum_policy_timestamp - 1.0e-9
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{policy_write_path}.timestamp",
                "detour policy write 必须晚于 diagnostics 与 TRACKING 接收。",
                actual=detour_policy_write_timestamp,
            )
        detour_causal_clear_match = ctx.required_mapping(
            detour,
            "causal_map_transition_clear_match",
            f"{detour_path}.causal_map_transition_clear_match",
        )
        minimum_detour = ctx.required_number(
            typed,
            "detour_deviation_minimum_m",
            f"{detour_path}.typed.detour_deviation_minimum_m",
            minimum=1.0e-6,
        )
        if (
            minimum_detour is not None
            and detour_report.maximum_trajectory_deviation is not None
            and detour_report.maximum_trajectory_deviation
            < minimum_detour - 1.0e-9
        ):
            ctx.reject(
                "missing_detour_geometry_evidence",
                f"{detour_path}.maximum_trajectory_deviation_m",
                "B-spline 未达到运行时固定 detour deviation 门限。",
                actual=detour_report.maximum_trajectory_deviation,
            )

    # 3) 对同一 accepted detour identity，按该时刻 OBB 重算所有保留轨迹样本净距。
    clearance_path = f"{path}.current_obstacle_clearance"
    clearance = ctx.required_mapping(
        aggregate,
        "current_obstacle_clearance",
        clearance_path,
    )
    _expect_true_field(ctx, clearance, "verified", f"{clearance_path}.verified")
    clearance_report = _validate_bspline_leaf_reference(
        ctx,
        clearance,
        clearance_path,
        bspline_reports,
        path_stamp_ns=None,
    )
    clearance_elapsed = (
        clearance_report.episode_elapsed_time_s
        if clearance_report is not None
        else None
    )
    required_clearance = ctx.required_number(
        clearance,
        "required_clearance_m",
        f"{clearance_path}.required_clearance_m",
        minimum=1.0e-6,
    )
    if (
        clearance_report is not None
        and clearance_report.required_any_yaw_clearance_radius_m is not None
    ):
        _expect_close(
            ctx,
            required_clearance,
            clearance_report.required_any_yaw_clearance_radius_m,
            f"{clearance_path}.required_clearance_m",
            tolerance=1.0e-12,
        )
    if detour_report is not None and clearance_report is not None and clearance_report.identity != detour_report.identity:
        ctx.reject(
            "wrong_identity",
            f"{clearance_path}.identity",
            "净距必须绑定 ordered_detour 的同一 accepted trajectory identity。",
        )
    if clearance_report is not None and clearance_report.identity not in accepted_identities:
        ctx.reject(
            "controller_identity_not_accepted",
            f"{clearance_path}.identity",
            "净距所用 B-spline identity 未被 controller 接受。",
        )
    raw_clearances = ctx.required_sequence(
        clearance,
        "obstacle_clearances",
        f"{clearance_path}.obstacle_clearances",
    )
    if clearance_report is not None:
        typed_clearances = clearance_report.report.get(
            "dynamic_obstacle_clearances"
        )
        expected_relevant = (
            [
                item
                for item in typed_clearances
                if isinstance(item, Mapping) and item.get("relevant") is True
            ]
            if isinstance(typed_clearances, Sequence)
            and not isinstance(typed_clearances, (str, bytes, bytearray))
            else []
        )
        if raw_clearances != expected_relevant:
            ctx.reject(
                "wrong_typed_evidence_reference",
                f"{clearance_path}.obstacle_clearances",
                "净距数组必须等于 typed report 中的 relevant obstacles。",
            )
    if not raw_clearances:
        ctx.reject(
            "missing_current_obstacle_clearance_evidence",
            f"{clearance_path}.obstacle_clearances",
            "detour 时刻没有相关动态障碍净距记录。",
        )
    reference_obstruction_witness = False
    for index, raw_clearance in enumerate(raw_clearances):
        item_path = f"{clearance_path}.obstacle_clearances[{index}]"
        item = ctx.mapping(raw_clearance, item_path)
        obstacle_id = ctx.required_string(
            item,
            "obstacle_id",
            f"{item_path}.obstacle_id",
            nonempty=True,
        )
        obstacle = _dynamic_obstacle_for_id(plan, obstacle_id)
        if obstacle is None:
            ctx.reject(
                "wrong_identity",
                f"{item_path}.obstacle_id",
                "净距记录引用了任务外障碍。",
                actual=obstacle_id,
            )
            continue
        state = (
            obstacle.state_at(clearance_elapsed)
            if clearance_elapsed is not None
            else None
        )
        if state is not None:
            _validate_dynamic_state(
                ctx,
                ctx.field(item, "obstacle_state", f"{item_path}.obstacle_state"),
                f"{item_path}.obstacle_state",
                expected=state.to_dict(),
            )
        claimed_clearance = ctx.required_number(
            item,
            "minimum_trajectory_center_to_obstacle_xy_m",
            f"{item_path}.minimum_trajectory_center_to_obstacle_xy_m",
            minimum=0.0,
        )
        claimed_sample_interval = ctx.required_number(
            item,
            "trajectory_sample_interval_s",
            f"{item_path}.trajectory_sample_interval_s",
            minimum=1.0e-12,
        )
        claimed_velocity_upper_bound = ctx.required_number(
            item,
            "maximum_velocity_upper_bound_mps",
            f"{item_path}.maximum_velocity_upper_bound_mps",
            minimum=0.0,
        )
        claimed_sampling_margin = ctx.required_number(
            item,
            "sampling_clearance_margin_m",
            f"{item_path}.sampling_clearance_margin_m",
            minimum=0.0,
        )
        claimed_continuous_lower_bound = ctx.required_number(
            item,
            "continuous_clearance_lower_bound_m",
            f"{item_path}.continuous_clearance_lower_bound_m",
        )
        continuous_clearance_verified = ctx.required_boolean(
            item,
            "continuous_clearance_verified",
            f"{item_path}.continuous_clearance_verified",
        )
        if clearance_report is not None:
            _expect_close(
                ctx,
                claimed_sample_interval,
                clearance_report.trajectory_sample_interval_s,
                f"{item_path}.trajectory_sample_interval_s",
                tolerance=1.0e-12,
            )
            _expect_close(
                ctx,
                claimed_velocity_upper_bound,
                clearance_report.maximum_velocity_upper_bound_mps,
                f"{item_path}.maximum_velocity_upper_bound_mps",
                tolerance=1.0e-12,
            )
            _expect_close(
                ctx,
                claimed_sampling_margin,
                clearance_report.sampling_clearance_margin_m,
                f"{item_path}.sampling_clearance_margin_m",
                tolerance=1.0e-12,
            )
        claimed_reference_clearance = ctx.required_number(
            item,
            "minimum_ordered_reference_center_to_obstacle_xy_m",
            (
                f"{item_path}."
                "minimum_ordered_reference_center_to_obstacle_xy_m"
            ),
            minimum=0.0,
        )
        reference_obstructed = ctx.required_boolean(
            item,
            "reference_obstructed",
            f"{item_path}.reference_obstructed",
        )
        blocked_then_clear = ctx.required_boolean(
            item,
            "reference_blocked_then_trajectory_clear",
            f"{item_path}.reference_blocked_then_trajectory_clear",
        )
        item_required = ctx.required_number(
            item,
            "required_clearance_m",
            f"{item_path}.required_clearance_m",
            minimum=1.0e-6,
        )
        _expect_close(
            ctx,
            item_required,
            required_clearance,
            f"{item_path}.required_clearance_m",
        )
        clearance_verified = ctx.required_boolean(
            item,
            "clearance_verified",
            f"{item_path}.clearance_verified",
        )
        if clearance_verified is not True:
            ctx.reject(
                "insufficient_dynamic_obstacle_clearance",
                f"{item_path}.clearance_verified",
                "aggregate 中的相关动态障碍都必须有安全轨迹净距。",
                actual=clearance_verified,
            )
        if continuous_clearance_verified is not True:
            ctx.reject(
                "insufficient_dynamic_obstacle_clearance",
                f"{item_path}.continuous_clearance_verified",
                "动态验收必须证明连续曲线净距下界安全。",
                actual=continuous_clearance_verified,
            )
        if clearance_verified != continuous_clearance_verified:
            ctx.reject(
                "invalid_continuous_clearance_contract",
                f"{item_path}.clearance_verified",
                "clearance_verified 必须等于连续净距验证结果。",
                actual=clearance_verified,
            )
        if blocked_then_clear != bool(
            reference_obstructed is True and clearance_verified is True
        ):
            ctx.reject(
                "invalid_reference_obstruction_evidence",
                f"{item_path}.reference_blocked_then_trajectory_clear",
                "阻断后绕行布尔值必须由 reference_obstructed 与 clearance_verified 推导。",
                actual=blocked_then_clear,
            )
        _expect_true_field(ctx, item, "relevant", f"{item_path}.relevant")
        ctx.required_number(
            item,
            "relevance_distance_m",
            f"{item_path}.relevance_distance_m",
            minimum=0.0,
        )
        if (
            clearance_report is not None
            and state is not None
            and clearance_report.trajectory_samples
        ):
            sampled_clearance = min(
                _point_to_oriented_cuboid_xy_clearance(
                    point,
                    obstacle,
                    state,
                )
                for point in clearance_report.trajectory_samples
            )
            _expect_close(
                ctx,
                claimed_clearance,
                sampled_clearance,
                f"{item_path}.minimum_trajectory_center_to_obstacle_xy_m",
                tolerance=_DYNAMIC_GEOMETRY_TOLERANCE_M,
            )
            recomputed_margin = (
                claimed_velocity_upper_bound
                * claimed_sample_interval
                * 0.5
                if claimed_velocity_upper_bound is not None
                and claimed_sample_interval is not None
                else None
            )
            if recomputed_margin is not None:
                _expect_close(
                    ctx,
                    claimed_sampling_margin,
                    recomputed_margin,
                    f"{item_path}.sampling_clearance_margin_m",
                    tolerance=1.0e-12,
                )
                recomputed_continuous_lower_bound = (
                    sampled_clearance - recomputed_margin
                )
                _expect_close(
                    ctx,
                    claimed_continuous_lower_bound,
                    recomputed_continuous_lower_bound,
                    f"{item_path}.continuous_clearance_lower_bound_m",
                    tolerance=_DYNAMIC_GEOMETRY_TOLERANCE_M,
                )
                recomputed_continuous_verified = bool(
                    required_clearance is not None
                    and recomputed_continuous_lower_bound + 1.0e-9
                    >= required_clearance
                )
                if (
                    continuous_clearance_verified
                    != recomputed_continuous_verified
                ):
                    ctx.reject(
                        "invalid_continuous_clearance_contract",
                        f"{item_path}.continuous_clearance_verified",
                        "连续净距布尔值与重算 lower bound 不一致。",
                        actual=continuous_clearance_verified,
                    )
            if (
                required_clearance is not None
                and claimed_continuous_lower_bound is not None
                and claimed_continuous_lower_bound
                < required_clearance - 1.0e-9
            ):
                ctx.reject(
                    "insufficient_dynamic_obstacle_clearance",
                    f"{item_path}.continuous_clearance_lower_bound_m",
                    "accepted detour 对当前推车的连续曲线净距下界不足。",
                    actual=claimed_continuous_lower_bound,
                )
        if (
            clearance_report is not None
            and state is not None
            and clearance_report.trajectory_samples
            and clearance_report.reference_samples
        ):
            sampled_reference_clearance = min(
                _point_to_oriented_cuboid_xy_clearance(
                    point,
                    obstacle,
                    state,
                )
                for point in clearance_report.reference_samples
            )
            _expect_close(
                ctx,
                claimed_reference_clearance,
                sampled_reference_clearance,
                (
                    f"{item_path}."
                    "minimum_ordered_reference_center_to_obstacle_xy_m"
                ),
                tolerance=_DYNAMIC_GEOMETRY_TOLERANCE_M,
            )
            sampled_reference_obstructed = bool(
                item_required is not None
                and sampled_reference_clearance
                < item_required - 1.0e-9
            )
            if reference_obstructed != sampled_reference_obstructed:
                ctx.reject(
                    "invalid_reference_obstruction_evidence",
                    f"{item_path}.reference_obstructed",
                    "reference_obstructed 必须与有界有序参考样本的 OBB 净距一致。",
                    actual=reference_obstructed,
                )
            if (
                sampled_reference_obstructed
                and required_clearance is not None
                and claimed_continuous_lower_bound is not None
                and claimed_continuous_lower_bound
                >= required_clearance - 1.0e-9
            ):
                reference_obstruction_witness = True
    if not reference_obstruction_witness:
        ctx.reject(
            "missing_reference_obstruction_evidence",
            f"{clearance_path}.obstacle_clearances",
            "绕行必须由同一推车阻断原有序参考、而 accepted 轨迹恢复安全净距来证明。",
        )
    _expect_exact_field(
        ctx,
        clearance,
        "reason",
        "all_relevant_obstacles_clear",
        f"{clearance_path}.reason",
    )

    # 4) 同一旧占据点必须由更晚 observation 的 explicit p_miss 清除，不能靠滑窗。
    ghost_path = f"{path}.explicit_miss_ghost_clear"
    ghost = ctx.required_mapping(
        aggregate,
        "explicit_miss_ghost_clear",
        ghost_path,
    )
    _expect_true_field(ctx, ghost, "verified", f"{ghost_path}.verified")
    clear_report = _validate_grid_leaf_reference(
        ctx,
        ghost,
        ghost_path,
        grid_reports,
    )
    hit_sequence = ctx.required_integer(
        ghost,
        "matched_hit_observation_sequence",
        f"{ghost_path}.matched_hit_observation_sequence",
        minimum=1,
    )
    hit_report = (
        grid_reports.get(hit_sequence)
        if hit_sequence is not None
        and hit_sequence in transition_hit_sequences
        else None
    )
    clear_elapsed = (
        clear_report.episode_elapsed_time_s
        if clear_report is not None
        else None
    )
    if (
        clear_report is not None
        and detour_controller_header_stamp_ns is not None
        and clear_report.header_stamp_ns is not None
        and detour_controller_header_stamp_ns > clear_report.header_stamp_ns
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{detour_path}.controller_accepted_status.header.stamp_ns",
            "detour controller 接受 header 不能晚于 explicit-miss clear。",
            actual=detour_controller_header_stamp_ns,
        )
    if (
        clear_report is not None
        and detour_controller_receipt_timestamp is not None
        and clear_report.receipt_timestamp is not None
        and detour_controller_receipt_timestamp
        > clear_report.receipt_timestamp + 1.0e-9
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            (
                f"{detour_path}.controller_accepted_status."
                "receipt_timestamp"
            ),
            "detour controller 接收时间不能晚于 explicit-miss clear 接收时间。",
            actual=detour_controller_receipt_timestamp,
        )
    if (
        clear_report is not None
        and detour_tracking_header_stamp_ns is not None
        and clear_report.header_stamp_ns is not None
        and detour_tracking_header_stamp_ns > clear_report.header_stamp_ns
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{detour_path}.controller_tracking_status.header.stamp_ns",
            "detour TRACKING header 不能晚于 explicit-miss clear。",
            actual=detour_tracking_header_stamp_ns,
        )
    if (
        clear_report is not None
        and detour_tracking_receipt_timestamp is not None
        and clear_report.receipt_timestamp is not None
        and detour_tracking_receipt_timestamp
        > clear_report.receipt_timestamp + 1.0e-9
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{detour_path}.controller_tracking_status.receipt_timestamp",
            "detour TRACKING 接收时间不能晚于 explicit-miss clear。",
            actual=detour_tracking_receipt_timestamp,
        )
    if (
        clear_report is not None
        and detour_policy_write_timestamp is not None
        and clear_report.receipt_timestamp is not None
        and detour_policy_write_timestamp + 1.0e-9
        >= clear_report.receipt_timestamp
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            (
                f"{detour_path}.policy_identity_verified_tracking_write."
                "timestamp"
            ),
            "detour policy write 必须严格早于 clear poll 的接收时间。",
            actual=detour_policy_write_timestamp,
        )
    if (
        clear_report is not None
        and clear_report.observation_sequence is not None
        and hit_sequence is not None
        and clear_report.observation_sequence <= hit_sequence
    ):
        ctx.reject(
            "invalid_ghost_clear_sequence",
            f"{ghost_path}.observation_sequence",
            "clear observation sequence 必须晚于 hit sequence。",
            actual=clear_report.observation_sequence,
        )
    explicit_miss_voxel_count = ctx.required_integer(
        ghost,
        "explicit_free_miss_voxel_count",
        f"{ghost_path}.explicit_free_miss_voxel_count",
        minimum=1,
    )
    explicit_miss_count = ctx.required_integer(
        ghost,
        "occupied_to_free_by_explicit_miss_count",
        f"{ghost_path}.occupied_to_free_by_explicit_miss_count",
        minimum=1,
    )
    sliding_count = ctx.required_integer(
        ghost,
        "occupied_removed_by_sliding_reset_count",
        f"{ghost_path}.occupied_removed_by_sliding_reset_count",
        minimum=0,
    )
    if clear_report is not None:
        if explicit_miss_voxel_count != clear_report.explicit_free_miss_voxel_count:
            ctx.reject(
                "invalid_count",
                f"{ghost_path}.explicit_free_miss_voxel_count",
                "aggregate explicit-miss voxel count 与 typed report 不一致。",
                actual=explicit_miss_voxel_count,
            )
        if explicit_miss_count != clear_report.occupied_to_free_count:
            ctx.reject(
                "invalid_count",
                f"{ghost_path}.occupied_to_free_by_explicit_miss_count",
                "aggregate clear count 与 typed report 不一致。",
                actual=explicit_miss_count,
            )
        if sliding_count != clear_report.sliding_reset_count:
            ctx.reject(
                "invalid_count",
                f"{ghost_path}.occupied_removed_by_sliding_reset_count",
                "aggregate sliding reset count 与 typed clear report 不一致。",
                actual=sliding_count,
            )
    clear_matches = ctx.required_sequence(
        ghost,
        "clear_matches",
        f"{ghost_path}.clear_matches",
    )
    typed_clear_matches = (
        clear_report.report.get("dynamic_obstacle_explicit_miss_clear_matches")
        if clear_report is not None
        else None
    )
    if clear_matches != typed_clear_matches:
        ctx.reject(
            "wrong_typed_evidence_reference",
            f"{ghost_path}.clear_matches",
            "aggregate clear matches 必须来自同一 typed GridMap report。",
        )
    if not clear_matches:
        ctx.reject(
            "missing_free_ray_ghost_clear_evidence",
            f"{ghost_path}.clear_matches",
            "缺少 explicit-miss clear 与旧 hit 的几何匹配。",
        )
    for index, raw_match in enumerate(clear_matches):
        match_path = f"{ghost_path}.clear_matches[{index}]"
        match = ctx.mapping(raw_match, match_path)
        obstacle_id = ctx.required_string(
            match,
            "obstacle_id",
            f"{match_path}.obstacle_id",
            nonempty=True,
        )
        obstacle = _dynamic_obstacle_for_id(plan, obstacle_id)
        if obstacle is None:
            ctx.reject(
                "wrong_identity",
                f"{match_path}.obstacle_id",
                "ghost clear match 引用了任务外障碍。",
                actual=obstacle_id,
            )
            continue
        clear_point = _validate_vector(
            ctx,
            ctx.field(match, "point_world_xyz", f"{match_path}.point_world_xyz"),
            f"{match_path}.point_world_xyz",
            length=3,
        )
        matched_hit_sequence = ctx.required_integer(
            match,
            "matched_hit_observation_sequence",
            f"{match_path}.matched_hit_observation_sequence",
            minimum=1,
        )
        if matched_hit_sequence != hit_sequence:
            ctx.reject(
                "wrong_identity",
                f"{match_path}.matched_hit_observation_sequence",
                "clear match 与 aggregate 旧 hit sequence 不一致。",
                actual=matched_hit_sequence,
            )
        matched_hit_header = ctx.required_mapping(
            match,
            "matched_hit_header",
            f"{match_path}.matched_hit_header",
        )
        _expect_exact_field(
            ctx,
            matched_hit_header,
            "frame_id",
            "world",
            f"{match_path}.matched_hit_header.frame_id",
        )
        matched_hit_header_stamp_ns = _validate_ros_stamp(
            ctx,
            matched_hit_header,
            stamp_key="stamp",
            stamp_ns_key="stamp_ns",
            path=f"{match_path}.matched_hit_header",
        )
        hit_point = _validate_vector(
            ctx,
            ctx.field(
                match,
                "matched_hit_point_world_xyz",
                f"{match_path}.matched_hit_point_world_xyz",
            ),
            f"{match_path}.matched_hit_point_world_xyz",
            length=3,
        )
        voxel_index = _validate_integer_vector(
            ctx,
            ctx.field(
                match,
                "voxel_index_xyz",
                f"{match_path}.voxel_index_xyz",
            ),
            f"{match_path}.voxel_index_xyz",
            length=3,
        )
        match_resolution = ctx.required_number(
            match,
            "map_resolution_m",
            f"{match_path}.map_resolution_m",
            minimum=1.0e-9,
        )
        match_distance = ctx.required_number(
            match,
            "match_distance_m",
            f"{match_path}.match_distance_m",
            minimum=0.0,
        )
        voxel_separation_tolerance = ctx.required_number(
            match,
            "voxel_obstacle_separation_tolerance_m",
            f"{match_path}.voxel_obstacle_separation_tolerance_m",
            minimum=0.0,
        )
        tolerance = ctx.required_number(
            match,
            "match_tolerance_m",
            f"{match_path}.match_tolerance_m",
            minimum=0.0,
        )
        _expect_true_field(
            ctx,
            match,
            "matched_hit_provenance_verified",
            f"{match_path}.matched_hit_provenance_verified",
        )
        sliding_reset_used = ctx.required_boolean(
            match,
            "sliding_reset_used",
            f"{match_path}.sliding_reset_used",
        )
        if sliding_reset_used is not False:
            ctx.reject(
                "sliding_reset_not_explicit_miss",
                f"{match_path}.sliding_reset_used",
                "该 clear provenance 不能由 sliding reset 产生。",
                actual=sliding_reset_used,
            )
        expected_match_tolerance = (
            0.5 * math.sqrt(3.0) * match_resolution + 1.0e-9
            if match_resolution is not None
            else None
        )
        if expected_match_tolerance is not None:
            _expect_close(
                ctx,
                tolerance,
                expected_match_tolerance,
                f"{match_path}.match_tolerance_m",
                tolerance=1.0e-12,
            )
            _expect_close(
                ctx,
                voxel_separation_tolerance,
                expected_match_tolerance,
                f"{match_path}.voxel_obstacle_separation_tolerance_m",
                tolerance=1.0e-12,
            )
        direct_provenance: tuple[
            tuple[float, float, float],
            tuple[int, int, int],
            int,
            tuple[float, float, float],
            int,
        ] | None = None
        if (
            clear_report is not None
            and clear_point is not None
            and voxel_index is not None
            and matched_hit_sequence is not None
            and hit_point is not None
        ):
            for candidate in zip(
                clear_report.clear_samples,
                clear_report.clear_voxel_indices,
                clear_report.clear_transition_hit_sequences,
                clear_report.clear_transition_hit_samples,
                clear_report.clear_transition_hit_header_stamps_ns,
            ):
                if candidate[:4] == (
                    clear_point,
                    voxel_index,
                    matched_hit_sequence,
                    hit_point,
                ):
                    direct_provenance = candidate
                    break
        if direct_provenance is None:
            ctx.reject(
                "wrong_geometry_reference",
                f"{match_path}.matched_hit_point_world_xyz",
                "clear match 必须逐项引用 clear report 直接携带的 point+voxel+transition 来源。",
            )
        direct_hit_stamp_ns = (
            direct_provenance[4] if direct_provenance is not None else None
        )
        if (
            direct_hit_stamp_ns is not None
            and matched_hit_header_stamp_ns != direct_hit_stamp_ns
        ):
            ctx.reject(
                "wrong_typed_evidence_reference",
                f"{match_path}.matched_hit_header",
                "matched hit header 必须等于 clear report 对齐保存的来源 stamp。",
                actual=matched_hit_header_stamp_ns,
            )
        if (
            hit_report is not None
            and direct_hit_stamp_ns is not None
            and hit_report.header_stamp_ns != direct_hit_stamp_ns
        ):
            ctx.reject(
                "conflicting_diagnostic_identity",
                f"{match_path}.matched_hit_header",
                "仍在 proof ring 的旧 transition report 与 direct provenance stamp 冲突。",
            )
        if (
            hit_report is not None
            and hit_report.free_to_occupied_transition_count is not None
            and hit_report.free_to_occupied_transition_count <= 64
            and hit_point is not None
            and voxel_index is not None
            and (hit_point, voxel_index)
            not in tuple(
                zip(
                    hit_report.transition_hit_samples,
                    hit_report.transition_voxel_indices,
                )
            )
        ):
            ctx.reject(
                "conflicting_diagnostic_identity",
                f"{match_path}.matched_hit_point_world_xyz",
                "未截断的旧 transition report 与 direct provenance point+voxel 冲突。",
            )
        if clear_report is not None and match_resolution is not None:
            if clear_report.map_resolution_m is not None:
                _expect_close(
                    ctx,
                    match_resolution,
                    clear_report.map_resolution_m,
                    f"{match_path}.map_resolution_m",
                    tolerance=1.0e-12,
                )
        if hit_report is not None and match_resolution is not None:
            if hit_report.map_resolution_m is not None:
                _expect_close(
                    ctx,
                    match_resolution,
                    hit_report.map_resolution_m,
                    f"{match_path}.map_resolution_m",
                    tolerance=1.0e-12,
                )
        hit_elapsed: float | None = None
        if (
            direct_hit_stamp_ns is not None
            and clear_report is not None
            and clear_report.ros_time_offset_s is not None
        ):
            hit_elapsed = (
                float(direct_hit_stamp_ns) * 1.0e-9
                - clear_report.ros_time_offset_s
            )
            if hit_elapsed < 0.0:
                ctx.reject(
                    "invalid_episode_time_binding",
                    f"{match_path}.matched_hit_header",
                    "direct provenance hit header 早于本 episode ROS 时间原点。",
                    actual=hit_elapsed,
                )
            if clear_elapsed is not None and clear_elapsed <= hit_elapsed:
                ctx.reject(
                    "invalid_ghost_clear_sequence",
                    f"{match_path}.matched_hit_header",
                    "explicit-miss clear 必须晚于 direct provenance hit。",
                    actual=hit_elapsed,
                )
        if (
            clear_point is not None
            and hit_point is not None
            and match_distance is not None
        ):
            recomputed_distance = math.dist(clear_point, hit_point)
            _expect_close(
                ctx,
                match_distance,
                recomputed_distance,
                f"{match_path}.match_distance_m",
                tolerance=1.0e-12,
            )
            if (
                tolerance is not None
                and recomputed_distance > tolerance + 1.0e-9
            ):
                ctx.reject(
                    "missing_free_ray_ghost_clear_evidence",
                    f"{match_path}.point_world_xyz",
                    "同一 canonical voxel 的 hit/clear 几何距离越界。",
                )
        clear_state = (
            obstacle.state_at(clear_elapsed)
            if clear_elapsed is not None
            else None
        )
        if clear_state is not None:
            _validate_dynamic_state(
                ctx,
                ctx.field(
                    match,
                    "obstacle_state_after_clear",
                    f"{match_path}.obstacle_state_after_clear",
                ),
                f"{match_path}.obstacle_state_after_clear",
                expected=clear_state.to_dict(),
            )
        hit_state = (
            obstacle.state_at(hit_elapsed)
            if hit_elapsed is not None
            else None
        )
        if hit_state is not None:
            _validate_dynamic_state(
                ctx,
                ctx.field(
                    match,
                    "obstacle_state_at_hit",
                    f"{match_path}.obstacle_state_at_hit",
                ),
                f"{match_path}.obstacle_state_at_hit",
                expected=hit_state.to_dict(),
            )
        reported_hit_clearance = ctx.required_number(
            match,
            "matched_hit_point_to_obstacle_xy_clearance_m",
            (
                f"{match_path}."
                "matched_hit_point_to_obstacle_xy_clearance_m"
            ),
            minimum=0.0,
        )
        if hit_point is not None and hit_state is not None:
            recomputed_hit_clearance = (
                _point_to_oriented_cuboid_xy_clearance(
                    hit_point,
                    obstacle,
                    hit_state,
                )
            )
            _expect_close(
                ctx,
                reported_hit_clearance,
                recomputed_hit_clearance,
                (
                    f"{match_path}."
                    "matched_hit_point_to_obstacle_xy_clearance_m"
                ),
                tolerance=_DYNAMIC_GEOMETRY_TOLERANCE_M,
            )
        if (
            hit_point is not None
            and hit_state is not None
            and not _point_inside_oriented_cuboid(
                hit_point,
                obstacle,
                hit_state,
                tolerance_m=_MAX_POST_FILTER_HIT_TOLERANCE_M,
            )
        ):
            ctx.reject(
                "missing_free_ray_ghost_clear_evidence",
                f"{match_path}.matched_hit_point_world_xyz",
                "ghost clear 的起始 hit 不属于当时推车。",
            )
        if (
            clear_point is not None
            and clear_state is not None
            and _point_inside_oriented_cuboid(
                clear_point,
                obstacle,
                clear_state,
                tolerance_m=(
                    voxel_separation_tolerance
                    if voxel_separation_tolerance is not None
                    else 0.0
                ),
            )
        ):
            ctx.reject(
                "missing_free_ray_ghost_clear_evidence",
                f"{match_path}.point_world_xyz",
                "clear 时推车仍占据该体素，不能认证 ghost 已离开。",
            )

    blocked_obstacle_ids = {
        str(item.get("obstacle_id"))
        for item in raw_clearances
        if isinstance(item, Mapping)
        and item.get("reference_blocked_then_trajectory_clear") is True
    }
    if detour_causal_clear_match is not None:
        if not any(
            detour_causal_clear_match == clear_match
            for clear_match in clear_matches
        ):
            ctx.reject(
                "wrong_typed_evidence_reference",
                f"{detour_path}.causal_map_transition_clear_match",
                "detour map 因果快照必须来自同一 typed direct clear matches。",
            )
        causal_obstacle_id = ctx.required_string(
            detour_causal_clear_match,
            "obstacle_id",
            (
                f"{detour_path}.causal_map_transition_clear_match."
                "obstacle_id"
            ),
            nonempty=True,
        )
        if causal_obstacle_id not in blocked_obstacle_ids:
            ctx.reject(
                "missing_reference_obstruction_evidence",
                (
                    f"{detour_path}.causal_map_transition_clear_match."
                    "obstacle_id"
                ),
                "map transition 与阻断 ordered reference 的必须是同一障碍。",
                actual=causal_obstacle_id,
            )
        causal_header = ctx.required_mapping(
            detour_causal_clear_match,
            "matched_hit_header",
            (
                f"{detour_path}.causal_map_transition_clear_match."
                "matched_hit_header"
            ),
        )
        causal_hit_stamp_ns = _validate_ros_stamp(
            ctx,
            causal_header,
            stamp_key="stamp",
            stamp_ns_key="stamp_ns",
            path=(
                f"{detour_path}.causal_map_transition_clear_match."
                "matched_hit_header"
            ),
        )
        if (
            causal_hit_stamp_ns is not None
            and detour_report is not None
            and detour_report.header_stamp_ns is not None
            and causal_hit_stamp_ns > detour_report.header_stamp_ns
        ):
            ctx.reject(
                "invalid_dynamic_evidence_window",
                f"{detour_path}.causal_map_transition_clear_match",
                "detour diagnostics 不能早于同一障碍真正的 free→occupied transition。",
                actual=causal_hit_stamp_ns,
            )

    # 5) 恢复必须是另一条 typed B-spline，且 before/after 都被 controller 接受。
    recovery_path = f"{path}.trajectory_recovery"
    recovery = ctx.required_mapping(
        aggregate,
        "trajectory_recovery",
        recovery_path,
    )
    _expect_true_field(ctx, recovery, "verified", f"{recovery_path}.verified")
    _expect_exact_field(
        ctx,
        recovery,
        "source",
        "ros2_scan_bspline_diagnostics",
        f"{recovery_path}.source",
    )
    _expect_exact_field(
        ctx,
        recovery,
        "topic",
        "/planning/bspline_diagnostics",
        f"{recovery_path}.topic",
    )
    before_sequence = ctx.required_integer(
        recovery,
        "before_diagnostic_sequence",
        f"{recovery_path}.before_diagnostic_sequence",
        minimum=1,
    )
    after_sequence = ctx.required_integer(
        recovery,
        "after_diagnostic_sequence",
        f"{recovery_path}.after_diagnostic_sequence",
        minimum=1,
    )
    before_header_mapping = ctx.required_mapping(
        recovery,
        "before_header",
        f"{recovery_path}.before_header",
    )
    _expect_exact_field(
        ctx,
        before_header_mapping,
        "frame_id",
        "world",
        f"{recovery_path}.before_header.frame_id",
    )
    before_header = _validate_ros_stamp(
        ctx,
        before_header_mapping,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{recovery_path}.before_header",
    )
    after_header_mapping = ctx.required_mapping(
        recovery,
        "after_header",
        f"{recovery_path}.after_header",
    )
    _expect_exact_field(
        ctx,
        after_header_mapping,
        "frame_id",
        "world",
        f"{recovery_path}.after_header.frame_id",
    )
    after_header = _validate_ros_stamp(
        ctx,
        after_header_mapping,
        stamp_key="stamp",
        stamp_ns_key="stamp_ns",
        path=f"{recovery_path}.after_header",
    )
    before_identity = _validate_full_trajectory_identity(
        ctx,
        ctx.field(
            recovery,
            "before_detour_identity",
            f"{recovery_path}.before_detour_identity",
        ),
        f"{recovery_path}.before_detour_identity",
        path_stamp_ns=None,
    )
    after_identity = _validate_full_trajectory_identity(
        ctx,
        ctx.field(
            recovery,
            "after_recovery_identity",
            f"{recovery_path}.after_recovery_identity",
        ),
        f"{recovery_path}.after_recovery_identity",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    before_report = bspline_reports.get(before_sequence) if before_sequence is not None else None
    after_report = bspline_reports.get(after_sequence) if after_sequence is not None else None
    before_deviation = ctx.required_number(
        recovery,
        "before_maximum_trajectory_deviation_m",
        f"{recovery_path}.before_maximum_trajectory_deviation_m",
        minimum=0.0,
    )
    after_deviation = ctx.required_number(
        recovery,
        "after_maximum_trajectory_deviation_m",
        f"{recovery_path}.after_maximum_trajectory_deviation_m",
        minimum=0.0,
    )
    recovery_maximum_deviation = ctx.required_number(
        recovery,
        "recovery_maximum_deviation_m",
        f"{recovery_path}.recovery_maximum_deviation_m",
        minimum=0.0,
    )
    recovery_minimum_improvement = ctx.required_number(
        recovery,
        "recovery_minimum_improvement_m",
        f"{recovery_path}.recovery_minimum_improvement_m",
        minimum=0.0,
    )
    _expect_close(
        ctx,
        recovery_maximum_deviation,
        _DYNAMIC_RECOVERY_MAX_DEVIATION_M,
        f"{recovery_path}.recovery_maximum_deviation_m",
    )
    _expect_close(
        ctx,
        recovery_minimum_improvement,
        _DYNAMIC_RECOVERY_MIN_IMPROVEMENT_M,
        f"{recovery_path}.recovery_minimum_improvement_m",
    )
    for label, report, identity, header, sequence in (
        ("before", before_report, before_identity, before_header, before_sequence),
        ("after", after_report, after_identity, after_header, after_sequence),
    ):
        if report is None:
            ctx.reject(
                "missing_typed_evidence_reference",
                f"{recovery_path}.{label}_diagnostic_sequence",
                "恢复 identity 未找到对应 typed B-spline diagnostics。",
                actual=sequence,
            )
            continue
        if report.identity != identity or report.header_stamp_ns != header:
            ctx.reject(
                "wrong_identity",
                f"{recovery_path}.{label}_identity",
                "恢复 aggregate 的 header/identity 与 typed diagnostics 不一致。",
            )
        if identity not in accepted_identities:
            ctx.reject(
                "controller_identity_not_accepted",
                f"{recovery_path}.{label}_identity",
                "恢复前后 B-spline identity 都必须被 controller 接受。",
            )
    if before_sequence is not None and after_sequence is not None and after_sequence <= before_sequence:
        ctx.reject(
            "invalid_recovery_sequence",
            f"{recovery_path}.after_diagnostic_sequence",
            "恢复轨迹 diagnostics 必须晚于 detour 轨迹。",
            actual=after_sequence,
        )
    if (
        before_report is not None
        and before_report.maximum_trajectory_deviation is not None
    ):
        _expect_close(
            ctx,
            before_deviation,
            before_report.maximum_trajectory_deviation,
            f"{recovery_path}.before_maximum_trajectory_deviation_m",
        )
    if (
        after_report is not None
        and after_report.maximum_trajectory_deviation is not None
    ):
        _expect_close(
            ctx,
            after_deviation,
            after_report.maximum_trajectory_deviation,
            f"{recovery_path}.after_maximum_trajectory_deviation_m",
        )
    if (
        after_deviation is not None
        and recovery_maximum_deviation is not None
        and after_deviation
        > recovery_maximum_deviation + 1.0e-9
    ):
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            f"{recovery_path}.after_maximum_trajectory_deviation_m",
            "恢复轨迹 deviation 必须回落到 recovery 上限以内。",
            actual=after_deviation,
        )
    if (
        before_deviation is not None
        and after_deviation is not None
        and recovery_minimum_improvement is not None
        and before_deviation - after_deviation
        < recovery_minimum_improvement - 1.0e-9
    ):
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            f"{recovery_path}.after_maximum_trajectory_deviation_m",
            "恢复轨迹必须相对绕行轨迹达到最小 deviation 改善量。",
            actual=before_deviation - after_deviation,
        )
    if before_identity is not None and after_identity is not None and before_identity == after_identity:
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            f"{recovery_path}.after_recovery_identity",
            "恢复必须由不同的 accepted trajectory identity 证明。",
        )
    if detour_report is not None and before_identity != detour_report.identity:
        ctx.reject(
            "wrong_identity",
            f"{recovery_path}.before_detour_identity",
            "trajectory_recovery.before 必须引用 ordered_detour identity。",
        )
    recovery_elapsed = (
        after_report.episode_elapsed_time_s
        if after_report is not None
        else None
    )
    if after_report is not None:
        if (
            after_report.stationary is not False
            or after_report.emergency_stop is not False
            or after_report.ordered_reference_checked is not True
            or after_report.ordered_reference_safe is not True
        ):
            ctx.reject(
                "missing_dynamic_recovery_evidence",
                recovery_path,
                "恢复轨迹必须是通过 ordered reference 门的普通运动 B-spline。",
            )

    controller_lifecycle = evidence.simulation.get(
        "scan_controller_status_lifecycle_report"
    )
    recovery_accepted_path = f"{recovery_path}.controller_accepted_status"
    recovery_accepted_status = ctx.required_mapping(
        recovery,
        "controller_accepted_status",
        recovery_accepted_path,
    )
    _, recovery_accepted_sequence = _validate_controller_status(
        ctx,
        recovery_accepted_status,
        recovery_accepted_path,
        path_stamp_ns=evidence.path_stamp_ns,
    )
    recovery_acceptance_header = ctx.required_mapping(
        recovery_accepted_status,
        "header",
        f"{recovery_accepted_path}.header",
    )
    recovery_acceptance_header_ns = ctx.required_integer(
        recovery_acceptance_header,
        "stamp_ns",
        f"{recovery_accepted_path}.header.stamp_ns",
        minimum=1,
    )
    recovery_acceptance_receipt = ctx.required_number(
        recovery_accepted_status,
        "receipt_timestamp",
        f"{recovery_accepted_path}.receipt_timestamp",
        minimum=0.0,
    )
    if (
        recovery_accepted_status.get("accepted") is not True
        or _mapping_identity_tuple(
            recovery_accepted_status.get("identity")
        )
        != after_identity
    ):
        ctx.reject(
            "controller_identity_not_accepted",
            recovery_accepted_path,
            "recovery 必须绑定 exact after identity 且 accepted=true 的状态。",
        )
    accepted_status_reports = (
        controller_lifecycle.get("accepted_status_reports")
        if isinstance(controller_lifecycle, Mapping)
        else None
    )
    if (
        not isinstance(accepted_status_reports, Sequence)
        or isinstance(accepted_status_reports, (str, bytes, bytearray))
        or not any(
            recovery_accepted_status == retained
            for retained in accepted_status_reports
        )
    ):
        ctx.reject(
            "wrong_controller_acceptance_reference",
            recovery_accepted_path,
            "recovery acceptance snapshot 必须来自 bounded accepted ring。",
        )
    if (
        after_report is not None
        and recovery_acceptance_header_ns is not None
        and after_report.header_stamp_ns is not None
        and recovery_acceptance_header_ns < after_report.header_stamp_ns
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{recovery_accepted_path}.header.stamp_ns",
            "recovery acceptance header 不能早于 after diagnostics。",
            actual=recovery_acceptance_header_ns,
        )
    if (
        after_report is not None
        and recovery_acceptance_receipt is not None
        and after_report.receipt_timestamp is not None
        and recovery_acceptance_receipt
        < after_report.receipt_timestamp - 1.0e-9
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{recovery_accepted_path}.receipt_timestamp",
            "recovery acceptance 接收时间不能早于 after diagnostics。",
            actual=recovery_acceptance_receipt,
        )
    recovery_tracking_path = f"{recovery_path}.controller_tracking_status"
    controller_tracking_status = ctx.required_mapping(
        recovery,
        "controller_tracking_status",
        recovery_tracking_path,
    )
    controller_status_sequence, controller_acceptance_sequence = (
        _validate_controller_status(
            ctx,
            controller_tracking_status,
            recovery_tracking_path,
            path_stamp_ns=evidence.path_stamp_ns,
        )
    )
    copied_status_sequence = ctx.required_integer(
        recovery,
        "controller_status_sequence",
        f"{recovery_path}.controller_status_sequence",
        minimum=1,
    )
    copied_acceptance_sequence = ctx.required_integer(
        recovery,
        "controller_acceptance_sequence",
        f"{recovery_path}.controller_acceptance_sequence",
        minimum=1,
    )
    if (
        copied_status_sequence != controller_status_sequence
        or copied_acceptance_sequence != controller_acceptance_sequence
        or controller_acceptance_sequence != recovery_accepted_sequence
    ):
        ctx.reject(
            "wrong_controller_acceptance_reference",
            recovery_path,
            "recovery 的 controller 序号副本与 typed tracking status 不一致。",
        )
    if (
        controller_tracking_status.get("state") != 10
        or controller_tracking_status.get("trajectory_valid") is not True
        or _mapping_identity_tuple(controller_tracking_status.get("identity"))
        != after_identity
    ):
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            f"{recovery_path}.controller_tracking_status",
            "恢复 identity 必须由 controller 的 identity-valid TRACKING 状态证明。",
        )
    tracking_status_reports = (
        controller_lifecycle.get("tracking_status_reports")
        if isinstance(controller_lifecycle, Mapping)
        else None
    )
    if (
        not isinstance(tracking_status_reports, Sequence)
        or isinstance(tracking_status_reports, (str, bytes, bytearray))
        or not any(
            controller_tracking_status == retained
            for retained in tracking_status_reports
        )
    ):
        ctx.reject(
            "wrong_controller_acceptance_reference",
            recovery_tracking_path,
            "recovery TRACKING snapshot 必须来自 bounded controller ring。",
        )
    recovery_tracking_header = ctx.required_mapping(
        controller_tracking_status,
        "header",
        f"{recovery_tracking_path}.header",
    )
    recovery_tracking_header_ns = ctx.required_integer(
        recovery_tracking_header,
        "stamp_ns",
        f"{recovery_tracking_path}.header.stamp_ns",
        minimum=1,
    )
    controller_tracking_receipt = ctx.required_number(
        controller_tracking_status,
        "receipt_timestamp",
        f"{recovery_tracking_path}.receipt_timestamp",
        minimum=0.0,
    )
    if (
        recovery_tracking_header_ns is not None
        and recovery_acceptance_header_ns is not None
        and recovery_tracking_header_ns < recovery_acceptance_header_ns
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{recovery_tracking_path}.header.stamp_ns",
            "recovery TRACKING header 不能早于 acceptance header。",
            actual=recovery_tracking_header_ns,
        )
    if (
        controller_tracking_receipt is not None
        and recovery_acceptance_receipt is not None
        and controller_tracking_receipt
        < recovery_acceptance_receipt - 1.0e-9
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{recovery_tracking_path}.receipt_timestamp",
            "recovery TRACKING 接收时间不能早于 acceptance。",
            actual=controller_tracking_receipt,
        )
    _expect_true_field(
        ctx,
        recovery,
        "policy_identity_valid_tracking",
        f"{recovery_path}.policy_identity_valid_tracking",
    )
    policy_evidence = ctx.field(
        recovery,
        "policy_identity_verified_tracking_write",
        f"{recovery_path}.policy_identity_verified_tracking_write",
    )
    _validate_consumed_tracking_evidence(
        ctx,
        policy_evidence,
        f"{recovery_path}.policy_identity_verified_tracking_write",
        pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
        path_stamp_ns=evidence.path_stamp_ns,
    )
    policy_controller_snapshot = (
        ctx.required_mapping(
            policy_evidence,
            "scan_controller_status_snapshot",
            (
                f"{recovery_path}."
                "policy_identity_verified_tracking_write."
                "scan_controller_status_snapshot"
            ),
        )
        if isinstance(policy_evidence, Mapping)
        else {}
    )
    _validate_controller_status(
        ctx,
        policy_controller_snapshot,
        (
            f"{recovery_path}."
            "policy_identity_verified_tracking_write."
            "scan_controller_status_snapshot"
        ),
        path_stamp_ns=evidence.path_stamp_ns,
    )
    if policy_controller_snapshot != controller_tracking_status:
        ctx.reject(
            "wrong_controller_acceptance_reference",
            (
                f"{recovery_path}."
                "policy_identity_verified_tracking_write."
                "scan_controller_status_snapshot"
            ),
            "policy write 必须保存与 recovery exact TRACKING 完全相同的 controller snapshot。",
        )
    if (
        _mapping_identity_tuple(policy_controller_snapshot.get("identity"))
        != after_identity
    ):
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            (
                f"{recovery_path}."
                "policy_identity_verified_tracking_write."
                "scan_controller_status_snapshot.identity"
            ),
            "policy write 前观测的 controller snapshot 必须跟踪 recovery identity。",
        )
    policy_timestamp = (
        float(policy_evidence.get("timestamp"))
        if isinstance(policy_evidence, Mapping)
        and isinstance(policy_evidence.get("timestamp"), (int, float))
        and not isinstance(policy_evidence.get("timestamp"), bool)
        else None
    )
    policy_lifecycle = evidence.simulation.get(
        "navigation_policy_gate_lifecycle_report"
    )
    policy_write_reports = (
        policy_lifecycle.get("identity_verified_tracking_write_reports")
        if isinstance(policy_lifecycle, Mapping)
        else None
    )
    if (
        not isinstance(policy_write_reports, Sequence)
        or isinstance(policy_write_reports, (str, bytes, bytearray))
        or not any(
            policy_evidence == retained
            for retained in policy_write_reports
        )
    ):
        ctx.reject(
            "wrong_policy_write_reference",
            f"{recovery_path}.policy_identity_verified_tracking_write",
            "recovery policy write 必须来自 bounded identity-valid TRACKING ring。",
        )
    recovery_ready_timestamps = [
        timestamp
        for timestamp in (
            (
                after_report.receipt_timestamp
                if after_report is not None
                else None
            ),
            controller_tracking_receipt,
        )
        if timestamp is not None
    ]
    if (
        policy_timestamp is not None
        and recovery_ready_timestamps
        and policy_timestamp + 1.0e-9 < max(recovery_ready_timestamps)
    ):
        ctx.reject(
            "invalid_dynamic_evidence_window",
            f"{recovery_path}.policy_identity_verified_tracking_write.timestamp",
            "policy identity-valid tracking write 必须晚于 recovery diagnostics 与 controller TRACKING 接收。",
            actual=policy_timestamp,
        )
    same_generation = ctx.required_boolean(
        recovery,
        "same_reference_path_generation",
        f"{recovery_path}.same_reference_path_generation",
    )
    expected_same_generation = bool(
        before_identity is not None
        and after_identity is not None
        and before_identity[0] == after_identity[0]
    )
    if same_generation is not True or not expected_same_generation:
        ctx.reject(
            "missing_dynamic_recovery_evidence",
            f"{recovery_path}.same_reference_path_generation",
            "恢复前后必须属于同一 reference Path generation。",
            actual=same_generation,
        )

    # 不同 topic 没有 DDS 全局顺序；只使用各叶子记录的 physics elapsed time 建立窗口。
    ordered_times = (
        (post_elapsed, f"{post_path}.elapsed_time_s"),
        (detour_elapsed, f"{detour_path}.elapsed_time_s"),
        (clearance_elapsed, f"{clearance_path}.elapsed_time_s"),
        (clear_elapsed, f"{ghost_path}.clear_elapsed_time_s"),
        (recovery_elapsed, f"{recovery_path}.elapsed_time_s"),
    )
    previous_time: float | None = None
    for current_time, time_path in ordered_times:
        if current_time is None:
            continue
        if first_elapsed_s is not None and current_time < first_elapsed_s - 1.0e-9:
            ctx.reject(
                "outside_dynamic_time_window",
                time_path,
                "typed 动态证据早于 PhysX 障碍生命周期。",
                actual=current_time,
            )
        if last_elapsed_s is not None and current_time > last_elapsed_s + 1.0e-9:
            ctx.reject(
                "outside_dynamic_time_window",
                time_path,
                "typed 动态证据晚于 PhysX 障碍生命周期。",
                actual=current_time,
            )
        if previous_time is not None and current_time < previous_time - 1.0e-9:
            ctx.reject(
                "invalid_dynamic_evidence_window",
                time_path,
                "动态证据 physics 时间窗顺序必须为 hit→detour/clearance→clear→recovery。",
                actual=current_time,
            )
        previous_time = current_time


def _validate_flat_policy(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
) -> None:
    _validate_nonfreezing_policy_mode(ctx, summary, evidence, expected_task_id=1002)
    _validate_flat_task_floors(ctx, evidence.task_config, require_place_disabled=False)
    _validate_no_dynamic_obstacles(ctx, evidence)
    _validate_controller_lifecycle(ctx, evidence, require_multiple_trajectories=False)
    _validate_clean_flat_baseline(ctx, evidence)


def _validate_crossfloor_carry(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
) -> None:
    """校验原 Go2-X5 checkpoint 的完整跨楼层搬运导航。"""

    _validate_task_identity(ctx, summary, evidence, expected_task_id=1002)
    _expect_exact_field(
        ctx,
        summary,
        "execution_mode",
        "navigation_carry_smoke",
        "$.execution_mode",
    )
    _expect_exact_field(
        ctx,
        summary,
        "success_semantics",
        (
            "physical_nav_to_place_with_arm_gripper_hold_with_"
            "scan_stair_root_lock_workaround"
        ),
        "$.success_semantics",
    )
    _validate_top_provenance(
        ctx,
        summary,
        {
            "navigation_root_lock_workaround_success": True,
            "physical_navigation_success": False,
            "pure_physics_success": False,
            "used_base_teleport": True,
            "used_direct_joint_state": True,
            "used_object_teleport": True,
            "used_kinematic_object_follow": True,
            "used_visual_replay": False,
            "used_navigation_base_lock": True,
            "used_navigation_support_joint_lock": True,
            "used_navigation_joint_pose_lock": True,
        },
    )
    _expect_exact_field(
        ctx,
        evidence.executor,
        "execution_phase",
        "carry_nav_to_place",
        "$.latest_executor_status.execution_phase",
    )
    _expect_true_field(
        ctx,
        evidence.executor,
        "policy_activity_seen",
        "$.latest_executor_status.policy_activity_seen",
    )

    start = ctx.required_mapping(
        evidence.task_config,
        "start",
        "$.task_config.start",
    )
    _expect_exact_field(
        ctx,
        start,
        "floor_id",
        "F1",
        "$.task_config.start.floor_id",
    )
    place = ctx.required_mapping(
        evidence.task_config,
        "place",
        "$.task_config.place",
    )
    _expect_true_field(ctx, place, "enabled", "$.task_config.place.enabled")
    place_goal = ctx.required_mapping(
        place,
        "base_goal",
        "$.task_config.place.base_goal",
    )
    _expect_exact_field(
        ctx,
        place_goal,
        "floor_id",
        "F2",
        "$.task_config.place.base_goal.floor_id",
    )

    stop_counts_path = (
        "$.simulation_report.navigation_policy_gate_lifecycle_report."
        "stop_reason_counts"
    )
    stop_counts = ctx.required_mapping(
        evidence.lifecycle,
        "stop_reason_counts",
        stop_counts_path,
    )
    for reason in ("scan_stair_freeze", "scan_stair_freeze_release"):
        count = ctx.required_integer(
            stop_counts,
            reason,
            f"{stop_counts_path}.{reason}",
            minimum=1,
        )
        if count == 0:
            ctx.reject(
                "missing_stair_freeze_evidence",
                f"{stop_counts_path}.{reason}",
                "跨楼层成功必须实际执行并释放楼梯底盘冻结。",
                actual=count,
            )

    _validate_tracking_writes(ctx, evidence, required=True)
    _validate_controller_lifecycle(
        ctx,
        evidence,
        require_multiple_trajectories=True,
    )
    _validate_no_dynamic_obstacles(ctx, evidence)


def _validate_clean_flat_baseline(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
) -> None:
    """要求静态平地基线全程无规划失败、轨迹超时和安全停车。"""

    controller_path = "$.simulation_report.scan_controller_status_lifecycle_report"
    controller = ctx.required_mapping(
        evidence.simulation,
        "scan_controller_status_lifecycle_report",
        controller_path,
    )
    state_counts = ctx.required_mapping(
        controller,
        "state_counts",
        f"{controller_path}.state_counts",
    )
    trajectory_timeout_count = ctx.integer(
        state_counts.get("trajectory_timeout", 0),
        f"{controller_path}.state_counts.trajectory_timeout",
        minimum=0,
    )
    if trajectory_timeout_count not in (None, 0):
        ctx.reject(
            "flat_trajectory_timeout",
            f"{controller_path}.state_counts.trajectory_timeout",
            "静态平地稳定基线不允许出现轨迹超时后再恢复。",
            actual=trajectory_timeout_count,
        )
    candidate_rejection_count = ctx.required_integer(
        controller,
        "candidate_rejection_count",
        f"{controller_path}.candidate_rejection_count",
        minimum=0,
    )
    goal_latched_rejection_count = ctx.required_integer(
        controller,
        "goal_latched_same_path_candidate_rejection_count",
        (
            f"{controller_path}."
            "goal_latched_same_path_candidate_rejection_count"
        ),
        minimum=0,
    )
    unexpected_rejection_count = ctx.required_integer(
        controller,
        "unexpected_candidate_rejection_count",
        f"{controller_path}.unexpected_candidate_rejection_count",
        minimum=0,
    )
    if (
        candidate_rejection_count is not None
        and goal_latched_rejection_count is not None
        and unexpected_rejection_count is not None
        and candidate_rejection_count
        != goal_latched_rejection_count + unexpected_rejection_count
    ):
        ctx.reject(
            "invalid_controller_rejection_accounting",
            f"{controller_path}.candidate_rejection_count",
            "controller 候选拒绝总数必须等于到达锁存拒绝与运行期拒绝之和。",
            actual=candidate_rejection_count,
        )
    if unexpected_rejection_count not in (None, 0):
        ctx.reject(
            "flat_controller_rejection",
            f"{controller_path}.unexpected_candidate_rejection_count",
            "静态平地稳定基线不允许 controller 在运行期拒绝候选轨迹。",
            actual=unexpected_rejection_count,
        )

    emergency_stop_count = ctx.required_integer(
        controller,
        "emergency_stop_status_count",
        f"{controller_path}.emergency_stop_status_count",
        minimum=0,
    )
    if emergency_stop_count not in (None, 0):
        ctx.reject(
            "flat_controller_emergency_stop",
            f"{controller_path}.emergency_stop_status_count",
            "静态平地稳定基线不允许出现 controller 急停后再恢复。",
            actual=emergency_stop_count,
        )

    policy_path = "$.simulation_report.navigation_policy_gate_lifecycle_report"
    policy = evidence.lifecycle
    for key, code, message in (
        (
            "maximum_consecutive_scan_failures",
            "flat_scan_planning_failure",
            "静态平地稳定基线不允许出现 SCAN 局部规划失败。",
        ),
        (
            "global_replan_requested_status_count",
            "flat_unexpected_global_replan",
            "静态平地稳定基线不应请求 PCT 全局重规划。",
        ),
        (
            "global_replan_in_flight_status_count",
            "flat_unexpected_global_replan",
            "静态平地稳定基线不应进入 PCT 全局重规划。",
        ),
    ):
        count = ctx.required_integer(
            policy,
            key,
            f"{policy_path}.{key}",
            minimum=0,
        )
        if count not in (None, 0):
            ctx.reject(code, f"{policy_path}.{key}", message, actual=count)

    # 组合 launch 在首帧 Path/点云完成同代绑定前会短暂发布
    # scan_stair_resume_waiting；其 NavigationStatus state 复用了 emergency_stop，
    # 但 controller 始终零速且没有安全故障。只豁免这一种启动原因，任何
    # 其他 supervisor emergency 仍会令稳定基线失败。
    emergency_count = ctx.required_integer(
        policy,
        "emergency_stop_observed_status_count",
        f"{policy_path}.emergency_stop_observed_status_count",
        minimum=0,
    )
    reason_counts = ctx.required_mapping(
        policy,
        "observed_reason_counts",
        f"{policy_path}.observed_reason_counts",
    )
    stair_resume_wait_count = ctx.integer(
        reason_counts.get("scan_stair_resume_waiting", 0),
        f"{policy_path}.observed_reason_counts.scan_stair_resume_waiting",
        minimum=0,
    )
    if (
        emergency_count is not None
        and stair_resume_wait_count is not None
        and emergency_count != stair_resume_wait_count
    ):
        ctx.reject(
            "flat_supervisor_emergency_stop",
            f"{policy_path}.emergency_stop_observed_status_count",
            "静态平地只允许启动阶段的 scan_stair_resume_waiting 零速状态，不允许真实 supervisor 急停。",
            actual=emergency_count,
        )


def _validate_active_sensing_command_aggregate(
    ctx: _ValidationContext,
    aggregate: Mapping[str, Any],
    path: str,
    *,
    command_length: int,
) -> tuple[float, ...] | None:
    """校验主动观测窗口内 controller/policy 的实际命令包络。"""

    _expect_positive_integer_field(
        ctx,
        aggregate,
        "sample_count",
        f"{path}.sample_count",
    )
    first_command = _validate_vector(
        ctx,
        ctx.field(aggregate, "first_command", f"{path}.first_command"),
        f"{path}.first_command",
        length=command_length,
    )
    if first_command is not None and any(
        abs(component) > 1.0e-12 for component in first_command
    ):
        ctx.reject(
            "active_sensing_nonzero_first_command",
            f"{path}.first_command",
            "主动观测 identity 的第一拍实际命令必须严格全零。",
            actual=list(first_command),
        )
    maximum_vx = ctx.required_number(
        aggregate,
        "max_abs_vx",
        f"{path}.max_abs_vx",
        minimum=0.0,
    )
    maximum_vy = ctx.required_number(
        aggregate,
        "max_abs_vy",
        f"{path}.max_abs_vy",
        minimum=0.0,
    )
    maximum_wz = ctx.required_number(
        aggregate,
        "max_abs_wz",
        f"{path}.max_abs_wz",
        minimum=0.0,
    )
    for maximum, maximum_path in (
        (maximum_vx, f"{path}.max_abs_vx"),
        (maximum_vy, f"{path}.max_abs_vy"),
    ):
        if maximum is not None and maximum > 1.0e-12:
            ctx.reject(
                "active_sensing_translation_command",
                maximum_path,
                "主动观测期间不允许任何平移命令。",
                actual=maximum,
            )
    if (
        maximum_wz is not None
        and maximum_wz > _ACTIVE_SENSING_MAX_YAW_RATE_RAD_S + 1.0e-12
    ):
        ctx.reject(
            "active_sensing_yaw_rate_exceeded",
            f"{path}.max_abs_wz",
            "主动观测实际角速度不得超过代码级 0.20 rad/s 上限。",
            actual=maximum_wz,
        )
    violation_count = ctx.required_integer(
        aggregate,
        "violation_count",
        f"{path}.violation_count",
        minimum=0,
    )
    if violation_count is not None and violation_count != 0:
        ctx.reject(
            "active_sensing_command_violation",
            f"{path}.violation_count",
            "主动观测的限幅前命令不能出现任何合同违规。",
            actual=violation_count,
        )
    return first_command


def _validate_active_sensing_zero_policy_write(
    ctx: _ValidationContext,
    write: Mapping[str, Any],
    path: str,
) -> tuple[int | None, int | None]:
    """校验 active identity 首拍由唯一 owner 真实强制清零。"""

    _expect_exact_field(ctx, write, "owner_id", "scan_cmd_vel", f"{path}.owner_id")
    _expect_false_field(ctx, write, "motion_allowed", f"{path}.motion_allowed")
    _expect_false_field(
        ctx,
        write,
        "navigation_emergency_stop_latched",
        f"{path}.navigation_emergency_stop_latched",
    )
    _expect_false_field(
        ctx,
        write,
        "navigation_cmd_vel_inhibited",
        f"{path}.navigation_cmd_vel_inhibited",
    )
    _validate_zero_command(
        ctx,
        ctx.field(write, "limited_target", f"{path}.limited_target"),
        f"{path}.limited_target",
    )
    _validate_zero_command(
        ctx,
        ctx.field(write, "written_command", f"{path}.written_command"),
        f"{path}.written_command",
    )
    stop_reasons = _validate_string_array(
        ctx,
        ctx.field(write, "stop_reasons", f"{path}.stop_reasons"),
        f"{path}.stop_reasons",
    )
    if stop_reasons != ("active_sensing_identity_zero_gate",):
        ctx.reject(
            "invalid_active_sensing_zero_gate",
            f"{path}.stop_reasons",
            "active 首拍必须且只能由 identity zero gate 强制清零。",
            actual=list(stop_reasons),
        )
    _expect_false_field(
        ctx,
        write,
        "cmd_vel_sample_received_this_tick",
        f"{path}.cmd_vel_sample_received_this_tick",
    )
    drained = ctx.required_boolean(
        write,
        "cmd_vel_sample_drained_this_tick",
        f"{path}.cmd_vel_sample_drained_this_tick",
    )
    if drained is True:
        source_sequence = ctx.required_integer(
            write,
            "cmd_vel_source_sequence",
            f"{path}.cmd_vel_source_sequence",
            minimum=1,
        )
        drain_sequence = ctx.required_integer(
            write,
            "last_cmd_vel_drain_sequence",
            f"{path}.last_cmd_vel_drain_sequence",
            minimum=1,
        )
        if source_sequence != drain_sequence:
            ctx.reject(
                "invalid_active_sensing_zero_gate",
                f"{path}.last_cmd_vel_drain_sequence",
                "zero gate 声明 drain 时必须绑定当拍最新 Twist sequence。",
                actual=drain_sequence,
            )

    gate_path = f"{path}.navigation_gate_diagnostics"
    gate = ctx.required_mapping(
        write,
        "navigation_gate_diagnostics",
        gate_path,
    )
    _expect_exact_field(
        ctx,
        gate,
        "schema",
        "navigation_policy_gate_diagnostics_v1",
        f"{gate_path}.schema",
    )
    _expect_true_field(ctx, gate, "required", f"{gate_path}.required")
    timeout = ctx.required_number(
        gate,
        "timeout_s",
        f"{gate_path}.timeout_s",
        minimum=0.0,
    )
    _expect_close(ctx, timeout, _EXPECTED_GATE_TIMEOUT_S, f"{gate_path}.timeout_s")
    status_fault = ctx.field(gate, "status_fault", f"{gate_path}.status_fault")
    if status_fault is not None and status_fault is not _MISSING:
        ctx.string(status_fault, f"{gate_path}.status_fault", nonempty=True)
    permit_received = ctx.required_boolean(
        gate,
        "permit_received",
        f"{gate_path}.permit_received",
    )
    _expect_exact_field(
        ctx,
        gate,
        "command_identity",
        None,
        f"{gate_path}.command_identity",
    )
    _expect_false_field(
        ctx,
        gate,
        "command_identity_matches_permit",
        f"{gate_path}.command_identity_matches_permit",
    )
    permit_path = f"{gate_path}.permit"
    permit_value = ctx.field(gate, "permit", permit_path)
    if permit_received is False:
        if permit_value is not None and permit_value is not _MISSING:
            ctx.reject(
                "invalid_active_sensing_zero_gate",
                permit_path,
                "permit_received=false 时 permit 必须为 null。",
                actual=permit_value,
            )
    elif permit_received is True:
        permit = ctx.mapping(permit_value, permit_path)
        for field in ("header_stamp_ns", "status_sequence", "state_revision"):
            ctx.required_integer(
                permit,
                field,
                f"{permit_path}.{field}",
                minimum=1,
            )
        ctx.required_integer(permit, "state", f"{permit_path}.state", minimum=0)
        for field in (
            "allow_tracking_command",
            "force_zero_velocity",
            "identity_valid",
        ):
            ctx.required_boolean(permit, field, f"{permit_path}.{field}")
    return None, None


def _validate_active_sensing_safety_zero_policy_write(
    ctx: _ValidationContext,
    write: Mapping[str, Any],
    path: str,
) -> None:
    """校验 active 窗口尾拍由安全门自洽地写入零速。"""

    _expect_exact_field(ctx, write, "owner_id", "scan_cmd_vel", f"{path}.owner_id")
    _expect_false_field(ctx, write, "motion_allowed", f"{path}.motion_allowed")
    _validate_zero_command(
        ctx,
        ctx.field(write, "limited_target", f"{path}.limited_target"),
        f"{path}.limited_target",
    )
    _validate_zero_command(
        ctx,
        ctx.field(write, "written_command", f"{path}.written_command"),
        f"{path}.written_command",
    )
    stop_reasons = _validate_string_array(
        ctx,
        ctx.field(write, "stop_reasons", f"{path}.stop_reasons"),
        f"{path}.stop_reasons",
    )
    if not stop_reasons:
        ctx.reject(
            "invalid_active_sensing_policy_permit",
            f"{path}.stop_reasons",
            "未放行的 active policy 尾拍必须记录至少一个安全停车原因。",
        )
    emergency_latched = ctx.required_boolean(
        write,
        "navigation_emergency_stop_latched",
        f"{path}.navigation_emergency_stop_latched",
    )
    emergency_reason = ctx.field(
        write,
        "navigation_emergency_stop_reason",
        f"{path}.navigation_emergency_stop_reason",
    )
    if emergency_latched is True:
        ctx.string(
            emergency_reason,
            f"{path}.navigation_emergency_stop_reason",
            nonempty=True,
        )
    elif emergency_reason not in (None, _MISSING):
        ctx.reject(
            "invalid_active_sensing_policy_permit",
            f"{path}.navigation_emergency_stop_reason",
            "未锁存 emergency stop 时原因必须为 null。",
            actual=emergency_reason,
        )
    inhibited = ctx.required_boolean(
        write,
        "navigation_cmd_vel_inhibited",
        f"{path}.navigation_cmd_vel_inhibited",
    )
    inhibit_reason = ctx.field(
        write,
        "navigation_cmd_vel_inhibit_reason",
        f"{path}.navigation_cmd_vel_inhibit_reason",
    )
    if inhibited is True:
        ctx.string(
            inhibit_reason,
            f"{path}.navigation_cmd_vel_inhibit_reason",
            nonempty=True,
        )
    elif inhibit_reason not in (None, _MISSING):
        ctx.reject(
            "invalid_active_sensing_policy_permit",
            f"{path}.navigation_cmd_vel_inhibit_reason",
            "未 inhibit cmd_vel 时原因必须为 null。",
            actual=inhibit_reason,
        )

    gate_path = f"{path}.navigation_gate_diagnostics"
    gate = ctx.required_mapping(write, "navigation_gate_diagnostics", gate_path)
    _expect_exact_field(
        ctx,
        gate,
        "schema",
        "navigation_policy_gate_diagnostics_v1",
        f"{gate_path}.schema",
    )
    _expect_true_field(ctx, gate, "required", f"{gate_path}.required")
    timeout = ctx.required_number(
        gate,
        "timeout_s",
        f"{gate_path}.timeout_s",
        minimum=0.0,
    )
    _expect_close(ctx, timeout, _EXPECTED_GATE_TIMEOUT_S, f"{gate_path}.timeout_s")
    permit_received = ctx.required_boolean(
        gate,
        "permit_received",
        f"{gate_path}.permit_received",
    )
    ctx.required_boolean(
        gate,
        "command_identity_matches_permit",
        f"{gate_path}.command_identity_matches_permit",
    )
    command_identity = gate.get("command_identity")
    if command_identity is not None:
        _validate_integer_vector(
            ctx,
            command_identity,
            f"{gate_path}.command_identity",
            length=3,
            minimum=1,
        )
    permit_path = f"{gate_path}.permit"
    permit_value = ctx.field(gate, "permit", permit_path)
    if permit_received is False:
        if permit_value not in (None, _MISSING):
            ctx.reject(
                "invalid_active_sensing_policy_permit",
                permit_path,
                "permit_received=false 时 permit 必须为 null。",
                actual=permit_value,
            )
    elif permit_received is True:
        permit = ctx.mapping(permit_value, permit_path)
        for field in (
            "header_stamp_ns",
            "status_sequence",
            "state_revision",
            "goal_id",
            "active_path_stamp_ns",
        ):
            ctx.required_integer(permit, field, f"{permit_path}.{field}", minimum=1)
        ctx.required_integer(permit, "state", f"{permit_path}.state", minimum=0)
        for field in (
            "allow_tracking_command",
            "force_zero_velocity",
            "identity_valid",
        ):
            ctx.required_boolean(permit, field, f"{permit_path}.{field}")


def _validate_active_sensing_lifecycle(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
) -> None:
    """校验一次完整主动复观测及同 Path 恢复的 typed 生命周期。"""

    path = "$.simulation_report.active_sensing_lifecycle_report"
    lifecycle = ctx.required_mapping(
        evidence.simulation,
        "active_sensing_lifecycle_report",
        path,
    )
    _expect_exact_field(
        ctx,
        lifecycle,
        "schema",
        "active_sensing_lifecycle_v1",
        f"{path}.schema",
    )
    attempt_count = ctx.required_integer(
        lifecycle,
        "attempt_count",
        f"{path}.attempt_count",
        minimum=0,
    )
    completed_count = ctx.required_integer(
        lifecycle,
        "completed_attempt_count",
        f"{path}.completed_attempt_count",
        minimum=0,
    )
    failed_count = ctx.required_integer(
        lifecycle,
        "failed_attempt_count",
        f"{path}.failed_attempt_count",
        minimum=0,
    )
    attempts = ctx.required_sequence(lifecycle, "attempts", f"{path}.attempts")
    if attempt_count is not None and len(attempts) != attempt_count:
        ctx.reject(
            "invalid_count",
            f"{path}.attempts",
            "attempts 长度必须等于 attempt_count。",
            actual=len(attempts),
        )
    if attempt_count != 1 or completed_count != 1 or failed_count != 0:
        ctx.reject(
            "missing_active_sensing_completion",
            path,
            "本次 dynamic_f1 必须且只能有一次成功主动观测，不能含失败尝试。",
            actual={
                "attempt_count": attempt_count,
                "completed_attempt_count": completed_count,
                "failed_attempt_count": failed_count,
            },
        )
    active_attempt_identity = ctx.field(
        lifecycle,
        "active_attempt_identity",
        f"{path}.active_attempt_identity",
    )
    if active_attempt_identity is not _MISSING and active_attempt_identity is not None:
        ctx.reject(
            "unterminated_active_sensing",
            f"{path}.active_attempt_identity",
            "成功摘要结束时不能仍有未终止的主动观测 identity。",
            actual=active_attempt_identity,
        )
    for pending_key in (
        "pending_active_controller_statuses",
        "pending_recovery_controller_statuses",
        "pending_active_policy_writes",
    ):
        pending_path = f"{path}.{pending_key}"
        pending = ctx.required_sequence(lifecycle, pending_key, pending_path)
        if pending:
            ctx.reject(
                "unterminated_active_sensing_evidence",
                pending_path,
                "成功摘要不能遗留尚未归属的跨 topic 主动观测证据。",
                actual=len(pending),
            )
    policy_zero_gate = ctx.field(
        lifecycle,
        "policy_zero_gate",
        f"{path}.policy_zero_gate",
    )
    if policy_zero_gate is not _MISSING and policy_zero_gate is not None:
        ctx.reject(
            "unterminated_active_sensing_evidence",
            f"{path}.policy_zero_gate",
            "成功摘要结束时 active identity zero gate 必须已经消费并清空。",
            actual=policy_zero_gate,
        )
    zero_gate_armed_count = ctx.required_integer(
        lifecycle,
        "policy_zero_gate_armed_count",
        f"{path}.policy_zero_gate_armed_count",
        minimum=0,
    )
    zero_gate_consumed_count = ctx.required_integer(
        lifecycle,
        "policy_zero_gate_consumed_count",
        f"{path}.policy_zero_gate_consumed_count",
        minimum=0,
    )
    if zero_gate_armed_count != 1 or zero_gate_consumed_count != 1:
        ctx.reject(
            "invalid_active_sensing_zero_gate_count",
            f"{path}.policy_zero_gate_consumed_count",
            "唯一成功主动观测必须恰好布署并消费一次 policy identity zero gate。",
            actual={
                "armed": zero_gate_armed_count,
                "consumed": zero_gate_consumed_count,
            },
        )
    gated_identities = ctx.required_sequence(
        lifecycle,
        "policy_zero_gated_identities",
        f"{path}.policy_zero_gated_identities",
    )
    if len(attempts) != 1:
        return

    attempt_path = f"{path}.attempts[0]"
    attempt = ctx.mapping(attempts[0], attempt_path)
    identity_value = ctx.field(attempt, "identity", f"{attempt_path}.identity")
    active_identity = _validate_full_trajectory_identity(
        ctx,
        identity_value,
        f"{attempt_path}.identity",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    parsed_gated_identities = [
        _validate_full_trajectory_identity(
            ctx,
            value,
            f"{path}.policy_zero_gated_identities[{index}]",
            path_stamp_ns=evidence.path_stamp_ns,
        )
        for index, value in enumerate(gated_identities)
    ]
    if len(parsed_gated_identities) != 1 or parsed_gated_identities[0] != (
        active_identity
    ):
        ctx.reject(
            "wrong_active_sensing_policy_identity",
            f"{path}.policy_zero_gated_identities",
            "zero gate 必须且只能记录本次完整主动观测 identity。",
            actual=parsed_gated_identities,
        )
    stop_reason_counts = ctx.required_mapping(
        evidence.lifecycle,
        "stop_reason_counts",
        (
            "$.simulation_report.navigation_policy_gate_lifecycle_report."
            "stop_reason_counts"
        ),
    )
    zero_gate_stop_count = ctx.required_integer(
        stop_reason_counts,
        "active_sensing_identity_zero_gate",
        (
            "$.simulation_report.navigation_policy_gate_lifecycle_report."
            "stop_reason_counts.active_sensing_identity_zero_gate"
        ),
        minimum=0,
    )
    if zero_gate_stop_count != 1:
        ctx.reject(
            "invalid_active_sensing_zero_gate_count",
            (
                "$.simulation_report.navigation_policy_gate_lifecycle_report."
                "stop_reason_counts.active_sensing_identity_zero_gate"
            ),
            "全局 policy 生命周期必须精确记录一次 active identity zero gate 实写。",
            actual=zero_gate_stop_count,
        )
    events = _validate_string_array(
        ctx,
        ctx.field(attempt, "events", f"{attempt_path}.events"),
        f"{attempt_path}.events",
    )
    event_reports = ctx.required_sequence(
        attempt,
        "event_reports",
        f"{attempt_path}.event_reports",
    )
    if len(event_reports) != len(events):
        ctx.reject(
            "invalid_count",
            f"{attempt_path}.event_reports",
            "主动观测 event_reports 必须与 events 一一对应。",
            actual=len(event_reports),
        )
    sequence_valid = bool(
        len(events) >= 5
        and events[:3] == ("STARTED", "ACCEPTED", "YAW_STABLE")
        and events[-1] == "COMPLETED"
        and all(event == "FUSION_PROGRESS" for event in events[3:-1])
        and len(events[3:-1]) >= 1
    )
    if not sequence_valid:
        ctx.reject(
            "invalid_active_sensing_event_order",
            f"{attempt_path}.events",
            "事件必须严格为 STARTED→ACCEPTED→YAW_STABLE→至少一次 FUSION_PROGRESS→COMPLETED。",
            actual=list(events),
        )

    bspline_lifecycle_path = (
        "$.simulation_report.bspline_diagnostics_lifecycle_report"
    )
    bspline_lifecycle = ctx.required_mapping(
        evidence.simulation,
        "bspline_diagnostics_lifecycle_report",
        bspline_lifecycle_path,
    )
    bspline_ros_time_offset_s = ctx.required_number(
        bspline_lifecycle,
        "ros_time_offset_s",
        f"{bspline_lifecycle_path}.ros_time_offset_s",
    )

    parsed_reports: dict[str, Mapping[str, Any]] = {}
    previous_diagnostic_sequence: int | None = None
    previous_event_receipt_timestamp: float | None = None
    previous_event_rx_sequence: int | None = None
    canonical_yaw: tuple[float, float, float, float] | None = None
    canonical_settle_values: tuple[float, float, float] | None = None
    canonical_trajectory_payload: Mapping[str, Any] | None = None
    active_trajectory_duration_s: float | None = None
    settle_stamp_ns: int | None = None
    settled_fusion_baseline: int | None = None
    previous_fusion_current: int | None = None
    previous_fusion_distinct: int | None = None
    stationary_world_position: tuple[float, float, float] | None = None
    for index, raw_report in enumerate(event_reports):
        report_path = f"{attempt_path}.event_reports[{index}]"
        report = ctx.mapping(raw_report, report_path)
        parsed = _validate_bspline_diagnostic_report(
            ctx,
            report,
            report_path,
            path_stamp_ns=evidence.path_stamp_ns,
        )
        _validate_episode_time_binding(
            ctx,
            header_stamp_ns=parsed.header_stamp_ns,
            episode_elapsed_time_s=parsed.episode_elapsed_time_s,
            report_ros_time_offset_s=parsed.ros_time_offset_s,
            ros_time_offset_s=bspline_ros_time_offset_s,
            path=report_path,
        )
        if (
            parsed.receipt_timestamp is not None
            and previous_event_receipt_timestamp is not None
            and parsed.receipt_timestamp <= previous_event_receipt_timestamp
        ):
            ctx.reject(
                "invalid_active_sensing_event_order",
                f"{report_path}.receipt_timestamp",
                "主动观测事件的接收时间必须严格递增。",
                actual=parsed.receipt_timestamp,
            )
        if parsed.receipt_timestamp is not None:
            previous_event_receipt_timestamp = parsed.receipt_timestamp
        if (
            parsed.rx_sequence is not None
            and previous_event_rx_sequence is not None
            and parsed.rx_sequence <= previous_event_rx_sequence
        ):
            ctx.reject(
                "invalid_active_sensing_event_order",
                f"{report_path}.rx_sequence",
                "主动观测事件的 DDS 接收序号必须严格递增。",
                actual=parsed.rx_sequence,
            )
        if parsed.rx_sequence is not None:
            previous_event_rx_sequence = parsed.rx_sequence
        event_name = events[index] if index < len(events) else ""
        trajectory_payload = {
            key: value
            for key, value in report.items()
            if key
            not in {
                "receipt_timestamp",
                "rx_sequence",
                "diagnostic_sequence",
                "active_sensing",
            }
        }
        if canonical_trajectory_payload is None:
            canonical_trajectory_payload = trajectory_payload
        elif trajectory_payload != canonical_trajectory_payload:
            ctx.reject(
                "inconsistent_active_sensing_trajectory",
                report_path,
                "同一主动观测 identity 的非事件 B-spline 几何与时域 payload 必须完全冻结。",
            )
        if active_trajectory_duration_s is None:
            active_trajectory_duration_s = parsed.trajectory_duration_s
        if parsed.identity is not None and active_identity is not None and (
            parsed.identity != active_identity
        ):
            ctx.reject(
                "wrong_active_sensing_identity",
                f"{report_path}.identity",
                "所有主动观测事件必须绑定同一完整 B-spline identity。",
                actual=list(parsed.identity),
            )
        if report.get("stationary") is not True:
            ctx.reject(
                "active_sensing_not_stationary",
                f"{report_path}.stationary",
                "主动观测 B-spline 必须严格同点。",
                actual=report.get("stationary"),
            )
        if parsed.trajectory_samples:
            report_stationary_position = parsed.trajectory_samples[0]
            if any(
                any(
                    not math.isclose(
                        component,
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for component, expected in zip(
                        sample,
                        report_stationary_position,
                        strict=True,
                    )
                )
                for sample in parsed.trajectory_samples[1:]
            ):
                ctx.reject(
                    "active_sensing_not_stationary",
                    f"{report_path}.trajectory_samples_world_xyz",
                    "stationary=true 必须由全时域严格同点样本证明。",
                )
            if stationary_world_position is None:
                stationary_world_position = report_stationary_position
            elif any(
                not math.isclose(
                    component,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for component, expected in zip(
                    report_stationary_position,
                    stationary_world_position,
                    strict=True,
                )
            ):
                ctx.reject(
                    "active_sensing_position_changed",
                    f"{report_path}.trajectory_samples_world_xyz",
                    "同一主动观测 identity 的所有事件必须保持相同世界位置。",
                )
        for key in ("is_final", "emergency_stop"):
            if report.get(key) is not False:
                ctx.reject(
                    "invalid_active_sensing_trajectory",
                    f"{report_path}.{key}",
                    "主动观测 B-spline 必须为 non-final、non-emergency。",
                    actual=report.get(key),
                )
        diagnostic_sequence = report.get("diagnostic_sequence")
        if isinstance(diagnostic_sequence, int) and not isinstance(
            diagnostic_sequence,
            bool,
        ):
            if (
                previous_diagnostic_sequence is not None
                and diagnostic_sequence <= previous_diagnostic_sequence
            ):
                ctx.reject(
                    "invalid_diagnostic_lifecycle",
                    f"{report_path}.diagnostic_sequence",
                    "主动观测 typed 诊断序号必须严格递增。",
                    actual=diagnostic_sequence,
                )
            previous_diagnostic_sequence = diagnostic_sequence

        active_path = f"{report_path}.active_sensing"
        active = ctx.required_mapping(report, "active_sensing", active_path)
        _expect_true_field(ctx, active, "enabled", f"{active_path}.enabled")
        expected_event_code = _ACTIVE_SENSING_EVENT_CODES.get(event_name)
        event_code = ctx.required_integer(
            active,
            "event",
            f"{active_path}.event",
            minimum=1,
        )
        if expected_event_code is not None and event_code != expected_event_code:
            ctx.reject(
                "wrong_active_sensing_event",
                f"{active_path}.event",
                "事件字符串与 typed event code 不一致。",
                actual=event_code,
            )
        start_yaw = ctx.required_number(
            active,
            "start_yaw",
            f"{active_path}.start_yaw",
        )
        target_yaw = ctx.required_number(
            active,
            "target_yaw",
            f"{active_path}.target_yaw",
        )
        yaw_offset = ctx.required_number(
            active,
            "yaw_offset",
            f"{active_path}.yaw_offset",
        )
        yaw_rate = ctx.required_number(
            active,
            "yaw_rate",
            f"{active_path}.yaw_rate",
            minimum=0.0,
        )
        if (
            yaw_offset is not None
            and (
                abs(yaw_offset) <= 1.0e-12
                or abs(yaw_offset)
                > _ACTIVE_SENSING_MAX_YAW_OFFSET_RAD + 1.0e-12
            )
        ):
            ctx.reject(
                "active_sensing_yaw_excursion_exceeded",
                f"{active_path}.yaw_offset",
                "主动观测 yaw offset 必须非零且不超过 0.22 rad。",
                actual=yaw_offset,
            )
        if (
            yaw_rate is not None
            and (
                yaw_rate <= 0.0
                or yaw_rate
                > _ACTIVE_SENSING_MAX_YAW_RATE_RAD_S + 1.0e-12
            )
        ):
            ctx.reject(
                "active_sensing_yaw_rate_exceeded",
                f"{active_path}.yaw_rate",
                "主动观测规划角速度必须在 (0, 0.20] rad/s。",
                actual=yaw_rate,
            )
        if (
            parsed.trajectory_duration_s is not None
            and yaw_offset is not None
            and yaw_rate is not None
            and yaw_rate > 0.0
            and parsed.trajectory_duration_s + 1.0e-12
            < abs(yaw_offset) / yaw_rate
        ):
            ctx.reject(
                "active_sensing_trajectory_too_short",
                f"{report_path}.trajectory_duration_s",
                "yaw-only B-spline 时长必须覆盖完整 signed yaw offset。",
                actual=parsed.trajectory_duration_s,
            )
        if (
            parsed.trajectory_duration_s is not None
            and yaw_offset is not None
            and yaw_rate is not None
            and yaw_rate > 0.0
            and parsed.trajectory_duration_s + 1.0e-12
            < abs(yaw_offset) / yaw_rate + 0.20
        ):
            ctx.reject(
                "active_sensing_trajectory_timeout_too_short",
                f"{report_path}.trajectory_duration_s",
                "主动观测总时限必须覆盖 yaw 及代码允许的最小接管、观测和安全余量。",
                actual=parsed.trajectory_duration_s,
            )
        if (
            start_yaw is not None
            and target_yaw is not None
            and yaw_offset is not None
        ):
            expected_target = math.atan2(
                math.sin(start_yaw + yaw_offset),
                math.cos(start_yaw + yaw_offset),
            )
            target_error = abs(
                math.atan2(
                    math.sin(target_yaw - expected_target),
                    math.cos(target_yaw - expected_target),
                )
            )
            if target_error > 1.0e-9:
                ctx.reject(
                    "invalid_active_sensing_target_yaw",
                    f"{active_path}.target_yaw",
                    "target_yaw 必须等于 start_yaw 加 signed yaw_offset。",
                    actual=target_yaw,
                )
        current_yaw = (
            start_yaw,
            target_yaw,
            yaw_offset,
            yaw_rate,
        )
        if all(value is not None for value in current_yaw):
            normalized_yaw = tuple(float(value) for value in current_yaw)
            if canonical_yaw is None:
                canonical_yaw = normalized_yaw  # type: ignore[assignment]
            elif any(
                not math.isclose(current, expected, rel_tol=0.0, abs_tol=1.0e-12)
                for current, expected in zip(
                    normalized_yaw,
                    canonical_yaw,
                    strict=True,
                )
            ):
                ctx.reject(
                    "inconsistent_active_sensing_snapshot",
                    active_path,
                    "同一 identity 的 yaw 参数在生命周期中不能变化。",
                )

        completed = ctx.required_boolean(
            active,
            "completed",
            f"{active_path}.completed",
        )
        failed = ctx.required_boolean(
            active,
            "failed",
            f"{active_path}.failed",
        )
        if completed is not None and completed != (event_name == "COMPLETED"):
            ctx.reject(
                "inconsistent_active_sensing_terminal_flag",
                f"{active_path}.completed",
                "completed 只能出现在 COMPLETED 事件。",
                actual=completed,
            )
        if failed is not False:
            ctx.reject(
                "active_sensing_failed",
                f"{active_path}.failed",
                "成功尝试不能携带 FAILED 终态。",
                actual=failed,
            )
        ctx.required_string(
            active,
            "reason",
            f"{active_path}.reason",
            nonempty=True,
        )
        report_fusion_baseline = ctx.required_integer(
            active,
            "fusion_baseline",
            f"{active_path}.fusion_baseline",
            minimum=0,
        )
        report_fusion_current = ctx.required_integer(
            active,
            "fusion_current",
            f"{active_path}.fusion_current",
            minimum=0,
        )
        report_fusion_distinct = ctx.required_integer(
            active,
            "fusion_distinct",
            f"{active_path}.fusion_distinct",
            minimum=0,
        )
        report_fusion_required = ctx.required_integer(
            active,
            "fusion_required",
            f"{active_path}.fusion_required",
            minimum=1,
        )
        if report_fusion_required != _ACTIVE_SENSING_REQUIRED_FUSIONS:
            ctx.reject(
                "wrong_active_sensing_fusion_requirement",
                f"{active_path}.fusion_required",
                "每条主动观测快照都必须保留 required=3。",
                actual=report_fusion_required,
            )
        if (
            report_fusion_baseline is not None
            and report_fusion_current is not None
            and report_fusion_current < report_fusion_baseline
        ):
            ctx.reject(
                "invalid_active_sensing_fusion_sequence",
                f"{active_path}.fusion_current",
                "fusion current 不能小于 baseline。",
                actual=report_fusion_current,
            )
        if (
            report_fusion_baseline is not None
            and report_fusion_current is not None
            and report_fusion_distinct is not None
            and report_fusion_distinct
            > report_fusion_current - report_fusion_baseline
        ):
            ctx.reject(
                "invalid_active_sensing_fusion_sequence",
                f"{active_path}.fusion_distinct",
                "不同采集时间戳计数不能超过 sequence 增量。",
                actual=report_fusion_distinct,
            )
        if event_name in {"YAW_STABLE", "FUSION_PROGRESS", "COMPLETED"}:
            current_settle_stamp = _validate_ros_stamp(
                ctx,
                active,
                stamp_key="settle_stamp",
                stamp_ns_key="settle_stamp_ns",
                path=active_path,
            )
            if settle_stamp_ns is None:
                settle_stamp_ns = current_settle_stamp
            elif current_settle_stamp != settle_stamp_ns:
                ctx.reject(
                    "inconsistent_active_sensing_snapshot",
                    f"{active_path}.settle_stamp_ns",
                    "稳定后的 settle stamp 必须保持不变。",
                    actual=current_settle_stamp,
                )
            if settled_fusion_baseline is None:
                settled_fusion_baseline = report_fusion_baseline
            elif report_fusion_baseline != settled_fusion_baseline:
                ctx.reject(
                    "inconsistent_active_sensing_fusion_evidence",
                    f"{active_path}.fusion_baseline",
                    "YAW_STABLE 后的 fusion baseline 不能改变。",
                    actual=report_fusion_baseline,
                )
            if event_name == "YAW_STABLE" and (
                report_fusion_current != report_fusion_baseline
                or report_fusion_distinct != 0
            ):
                ctx.reject(
                    "premature_active_sensing_fusion_evidence",
                    active_path,
                    "YAW_STABLE 建立基线时不能已经计入稳定后融合。",
                )
            if event_name in {"FUSION_PROGRESS", "COMPLETED"}:
                if (
                    previous_fusion_current is not None
                    and report_fusion_current is not None
                    and report_fusion_current < previous_fusion_current
                ):
                    ctx.reject(
                        "invalid_active_sensing_fusion_sequence",
                        f"{active_path}.fusion_current",
                        "主动观测 fusion current 不能回退。",
                        actual=report_fusion_current,
                    )
                if (
                    previous_fusion_distinct is not None
                    and report_fusion_distinct is not None
                    and report_fusion_distinct < previous_fusion_distinct
                ):
                    ctx.reject(
                        "invalid_active_sensing_fusion_sequence",
                        f"{active_path}.fusion_distinct",
                        "主动观测 distinct fusion 计数不能回退。",
                        actual=report_fusion_distinct,
                    )
                previous_fusion_pair = (
                    previous_fusion_current,
                    previous_fusion_distinct,
                )
                current_fusion_pair = (
                    report_fusion_current,
                    report_fusion_distinct,
                )
                if (
                    event_name == "FUSION_PROGRESS"
                    and None not in previous_fusion_pair
                    and current_fusion_pair == previous_fusion_pair
                ):
                    ctx.reject(
                        "invalid_active_sensing_fusion_sequence",
                        active_path,
                        "每个 FUSION_PROGRESS 事件都必须证明融合进度严格前进。",
                        actual=list(current_fusion_pair),
                    )
                if (
                    event_name == "COMPLETED"
                    and None not in previous_fusion_pair
                    and current_fusion_pair != previous_fusion_pair
                ):
                    ctx.reject(
                        "inconsistent_active_sensing_fusion_evidence",
                        active_path,
                        "COMPLETED 必须冻结并复述最后一条 FUSION_PROGRESS 计数。",
                        actual=list(current_fusion_pair),
                    )
            previous_fusion_current = report_fusion_current
            previous_fusion_distinct = report_fusion_distinct
            settle_yaw_error = ctx.required_number(
                active,
                "settle_yaw_error",
                f"{active_path}.settle_yaw_error",
                minimum=0.0,
            )
            settle_angular_speed = ctx.required_number(
                active,
                "settle_angular_speed",
                f"{active_path}.settle_angular_speed",
                minimum=0.0,
            )
            stable_duration = ctx.required_number(
                active,
                "stable_duration",
                f"{active_path}.stable_duration",
                minimum=0.0,
            )
            settle_values = (
                settle_yaw_error,
                settle_angular_speed,
                stable_duration,
            )
            if all(value is not None for value in settle_values):
                normalized_settle_values = tuple(
                    float(value) for value in settle_values
                )
                if canonical_settle_values is None:
                    canonical_settle_values = normalized_settle_values  # type: ignore[assignment]
                elif any(
                    not math.isclose(
                        current,
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for current, expected in zip(
                        normalized_settle_values,
                        canonical_settle_values,
                        strict=True,
                    )
                ):
                    ctx.reject(
                        "inconsistent_active_sensing_snapshot",
                        active_path,
                        "YAW_STABLE 后的稳定窗口测量值必须保持不变。",
                    )
            if (
                settle_yaw_error is not None
                and settle_yaw_error
                > _ACTIVE_SENSING_MAX_SETTLE_YAW_ERROR_RAD + 1.0e-12
            ):
                ctx.reject(
                    "active_sensing_not_settled",
                    f"{active_path}.settle_yaw_error",
                    "稳定窗口 yaw 误差不得超过 0.02 rad。",
                    actual=settle_yaw_error,
                )
            if (
                settle_angular_speed is not None
                and settle_angular_speed
                > _ACTIVE_SENSING_MAX_SETTLE_ANGULAR_SPEED_RAD_S + 1.0e-12
            ):
                ctx.reject(
                    "active_sensing_not_settled",
                    f"{active_path}.settle_angular_speed",
                    "稳定窗口角速度不得超过 0.05 rad/s。",
                    actual=settle_angular_speed,
                )
            if (
                stable_duration is not None
                and stable_duration + 1.0e-12
                < _ACTIVE_SENSING_MIN_STABLE_DURATION_S
            ):
                ctx.reject(
                    "active_sensing_not_settled",
                    f"{active_path}.stable_duration",
                    "yaw 与角速度必须连续稳定至少 0.10 秒。",
                    actual=stable_duration,
                )
            if (
                settle_stamp_ns is not None
                and active_identity is not None
                and active_trajectory_duration_s is not None
                and settle_stamp_ns
                > active_identity[2]
                + int(round(active_trajectory_duration_s * 1.0e9))
            ):
                ctx.reject(
                    "active_sensing_trajectory_expired",
                    f"{active_path}.settle_stamp_ns",
                    "yaw settle 必须发生在主动观测 B-spline 总时限内。",
                    actual=settle_stamp_ns,
                )
        else:
            early_settle_ns = ctx.required_integer(
                active,
                "settle_stamp_ns",
                f"{active_path}.settle_stamp_ns",
                minimum=0,
            )
            if early_settle_ns not in (None, 0):
                ctx.reject(
                    "premature_active_sensing_settle_evidence",
                    f"{active_path}.settle_stamp_ns",
                    "STARTED/ACCEPTED 不能预填稳定窗口证据。",
                    actual=early_settle_ns,
                )
            early_settle = ctx.required_mapping(
                active,
                "settle_stamp",
                f"{active_path}.settle_stamp",
            )
            for key in ("sec", "nanosec"):
                value = ctx.required_integer(
                    early_settle,
                    key,
                    f"{active_path}.settle_stamp.{key}",
                    minimum=0,
                )
                if value not in (None, 0):
                    ctx.reject(
                        "premature_active_sensing_settle_evidence",
                        f"{active_path}.settle_stamp.{key}",
                        "STARTED/ACCEPTED 的 settle stamp 必须严格为零。",
                        actual=value,
                    )
            for key in (
                "settle_yaw_error",
                "settle_angular_speed",
                "stable_duration",
            ):
                value = ctx.required_number(
                    active,
                    key,
                    f"{active_path}.{key}",
                    minimum=0.0,
                )
                if value is not None and value != 0.0:
                    ctx.reject(
                        "premature_active_sensing_settle_evidence",
                        f"{active_path}.{key}",
                        "STARTED/ACCEPTED 不能预填稳定值。",
                        actual=value,
                    )
            if any(
                value not in (None, 0)
                for value in (
                    report_fusion_baseline,
                    report_fusion_current,
                    report_fusion_distinct,
                )
            ):
                ctx.reject(
                    "premature_active_sensing_fusion_evidence",
                    active_path,
                    "STARTED/ACCEPTED 不能预填 fusion 进度。",
                )
        if event_name and event_name not in parsed_reports:
            parsed_reports[event_name] = report

    for event_name, snapshot_key in (
        ("STARTED", "started"),
        ("ACCEPTED", "accepted"),
        ("YAW_STABLE", "yaw_stable"),
        ("COMPLETED", "completed"),
    ):
        snapshot_path = f"{attempt_path}.{snapshot_key}"
        snapshot = ctx.mapping(
            ctx.field(attempt, snapshot_key, snapshot_path),
            snapshot_path,
        )
        expected_snapshot = parsed_reports.get(event_name)
        if expected_snapshot is not None and snapshot != expected_snapshot:
            ctx.reject(
                "wrong_active_sensing_snapshot_reference",
                snapshot_path,
                f"{snapshot_key} 必须引用同一事件的完整 typed report。",
            )
    failed_snapshot = ctx.field(attempt, "failed", f"{attempt_path}.failed")
    if failed_snapshot is not _MISSING and failed_snapshot is not None:
        ctx.reject(
            "active_sensing_failed",
            f"{attempt_path}.failed",
            "成功尝试的 failed snapshot 必须为 null。",
            actual=failed_snapshot,
        )

    completed_report = parsed_reports.get("COMPLETED", {})
    completed_active = (
        completed_report.get("active_sensing")
        if isinstance(completed_report, Mapping)
        else None
    )
    completed_active = ctx.mapping(
        completed_active,
        f"{attempt_path}.completed.active_sensing",
    )
    completed_receipt_timestamp = (
        float(completed_report["receipt_timestamp"])
        if isinstance(completed_report.get("receipt_timestamp"), (int, float))
        and not isinstance(completed_report.get("receipt_timestamp"), bool)
        else None
    )
    completed_source_cutoff_ns = (
        int(round(completed_receipt_timestamp * 1.0e9))
        if completed_receipt_timestamp is not None
        else None
    )
    planner_fusion_path = f"{attempt_path}.planner_fusion"
    planner_fusion = ctx.required_mapping(
        attempt,
        "planner_fusion",
        planner_fusion_path,
    )
    fusion_baseline = ctx.required_integer(
        planner_fusion,
        "baseline",
        f"{planner_fusion_path}.baseline",
        minimum=0,
    )
    fusion_current = ctx.required_integer(
        planner_fusion,
        "current",
        f"{planner_fusion_path}.current",
        minimum=0,
    )
    fusion_distinct = ctx.required_integer(
        planner_fusion,
        "distinct",
        f"{planner_fusion_path}.distinct",
        minimum=0,
    )
    fusion_required = ctx.required_integer(
        planner_fusion,
        "required",
        f"{planner_fusion_path}.required",
        minimum=1,
    )
    if fusion_required != _ACTIVE_SENSING_REQUIRED_FUSIONS:
        ctx.reject(
            "wrong_active_sensing_fusion_requirement",
            f"{planner_fusion_path}.required",
            "主动观测必须要求恰好 3 个稳定后真实融合。",
            actual=fusion_required,
        )
    if (
        fusion_baseline is not None
        and fusion_current is not None
        and fusion_current < fusion_baseline
    ):
        ctx.reject(
            "invalid_active_sensing_fusion_sequence",
            f"{planner_fusion_path}.current",
            "fusion current 不能小于 baseline。",
            actual=fusion_current,
        )
    if (
        fusion_baseline is not None
        and fusion_current is not None
        and fusion_current - fusion_baseline
        > _ACTIVE_SENSING_FUSION_HISTORY_CAPACITY
    ):
        ctx.reject(
            "active_sensing_fusion_history_truncated",
            f"{planner_fusion_path}.current",
            "成功主动观测的 fused evidence 区间不能超过代码级 64 条历史容量。",
            actual=fusion_current - fusion_baseline,
        )
    if fusion_distinct is not None and fusion_distinct < _ACTIVE_SENSING_REQUIRED_FUSIONS:
        ctx.reject(
            "insufficient_active_sensing_fusions",
            f"{planner_fusion_path}.distinct",
            "COMPLETED 必须证明至少 3 个不同采集时间戳的真实融合。",
            actual=fusion_distinct,
        )
    for key in ("baseline", "current", "distinct", "required"):
        if completed_active.get(f"fusion_{key}") != planner_fusion.get(key):
            ctx.reject(
                "inconsistent_active_sensing_fusion_evidence",
                f"{planner_fusion_path}.{key}",
                "planner_fusion 必须与 COMPLETED typed 快照严格一致。",
                actual=planner_fusion.get(key),
            )

    observation_path = f"{attempt_path}.post_settle_fused_observations"
    observations = ctx.required_sequence(
        attempt,
        "post_settle_fused_observations",
        observation_path,
    )
    if len(observations) != _ACTIVE_SENSING_REQUIRED_FUSIONS:
        ctx.reject(
            "insufficient_active_sensing_fusions",
            observation_path,
            "必须保留恰好 3 条稳定后真实非空 GridMap 融合证据。",
            actual=len(observations),
        )
    seen_header_stamps: set[int] = set()
    previous_observation_sequence: int | None = None
    grid_lifecycle_path = (
        "$.simulation_report.grid_map_observation_lifecycle_report"
    )
    grid_lifecycle = ctx.required_mapping(
        evidence.simulation,
        "grid_map_observation_lifecycle_report",
        grid_lifecycle_path,
    )
    grid_reports = ctx.required_sequence(
        grid_lifecycle,
        "diagnostic_reports",
        f"{grid_lifecycle_path}.diagnostic_reports",
    )
    first_grid_sequence = ctx.required_integer(
        grid_lifecycle,
        "first_observation_sequence",
        f"{grid_lifecycle_path}.first_observation_sequence",
        minimum=1,
    )
    last_grid_sequence = ctx.required_integer(
        grid_lifecycle,
        "last_observation_sequence",
        f"{grid_lifecycle_path}.last_observation_sequence",
        minimum=1,
    )
    for index, raw_observation in enumerate(observations):
        item_path = f"{observation_path}[{index}]"
        observation = ctx.mapping(raw_observation, item_path)
        header_stamp = ctx.required_integer(
            observation,
            "header_stamp_ns",
            f"{item_path}.header_stamp_ns",
            minimum=1,
        )
        _expect_true_field(
            ctx,
            observation,
            "map_fusion_performed",
            f"{item_path}.map_fusion_performed",
        )
        accepted_endpoint_count = ctx.required_integer(
            observation,
            "accepted_endpoint_count",
            f"{item_path}.accepted_endpoint_count",
            minimum=1,
        )
        observation_sequence = ctx.required_integer(
            observation,
            "observation_sequence",
            f"{item_path}.observation_sequence",
            minimum=1,
        )
        header = ctx.required_mapping(observation, "header", f"{item_path}.header")
        nested_header_stamp = _validate_ros_stamp(
            ctx,
            header,
            stamp_key="stamp",
            stamp_ns_key="stamp_ns",
            path=f"{item_path}.header",
        )
        if header_stamp is not None and nested_header_stamp != header_stamp:
            ctx.reject(
                "invalid_timestamp",
                f"{item_path}.header_stamp_ns",
                "GridMap 证据的扁平与 header stamp 必须一致。",
                actual=header_stamp,
            )
        if settle_stamp_ns is not None and header_stamp is not None and (
            header_stamp <= settle_stamp_ns
        ):
            ctx.reject(
                "pre_settle_fusion_evidence",
                f"{item_path}.header_stamp_ns",
                "真实融合的采集时间戳必须严格晚于 yaw settle。",
                actual=header_stamp,
            )
        if (
            header_stamp is not None
            and completed_source_cutoff_ns is not None
            and header_stamp > completed_source_cutoff_ns + 1
        ):
            ctx.reject(
                "post_completion_active_sensing_fusion",
                f"{item_path}.header_stamp_ns",
                "晚到消息可以回填，但 GridMap source frame 必须在 COMPLETED 接收时刻前采集。",
                actual=header_stamp,
            )
        if (
            header_stamp is not None
            and active_identity is not None
            and active_trajectory_duration_s is not None
            and header_stamp
            > active_identity[2]
            + int(round(active_trajectory_duration_s * 1.0e9))
        ):
            ctx.reject(
                "active_sensing_trajectory_expired",
                f"{item_path}.header_stamp_ns",
                "稳定后 GridMap 采集必须发生在主动观测 B-spline 总时限内。",
                actual=header_stamp,
            )
        if header_stamp is not None:
            if header_stamp in seen_header_stamps:
                ctx.reject(
                    "duplicate_active_sensing_fusion_stamp",
                    f"{item_path}.header_stamp_ns",
                    "三条真实融合必须来自不同采集时间戳。",
                    actual=header_stamp,
                )
            seen_header_stamps.add(header_stamp)
        if (
            observation_sequence is not None
            and previous_observation_sequence is not None
            and observation_sequence <= previous_observation_sequence
        ):
            ctx.reject(
                "invalid_active_sensing_fusion_sequence",
                f"{item_path}.observation_sequence",
                "稳定后 GridMap observation sequence 必须严格递增。",
                actual=observation_sequence,
            )
        if observation_sequence is not None:
            previous_observation_sequence = observation_sequence
        if (
            observation_sequence is not None
            and first_grid_sequence is not None
            and last_grid_sequence is not None
            and not first_grid_sequence <= observation_sequence <= last_grid_sequence
        ):
            ctx.reject(
                "active_sensing_fusion_outside_grid_lifecycle",
                f"{item_path}.observation_sequence",
                "主动观测融合必须位于全局 GridMap lifecycle 序号边界内。",
                actual=observation_sequence,
            )
        matching_grid_reports = [
            report
            for report in grid_reports
            if isinstance(report, Mapping)
            and report.get("observation_sequence") == observation_sequence
            and isinstance(report.get("header"), Mapping)
            and report["header"].get("stamp_ns") == header_stamp
        ]
        if len(matching_grid_reports) != 1:
            ctx.reject(
                "missing_active_sensing_grid_map_join",
                item_path,
                "每条主动观测融合必须精确引用全局 GridMap diagnostic ring 的唯一报告。",
                actual=len(matching_grid_reports),
            )
        else:
            grid_report = matching_grid_reports[0]
            if (
                grid_report.get("map_fusion_performed") is not True
                or grid_report.get("accepted_endpoint_count")
                != accepted_endpoint_count
                or grid_report.get("header") != header
            ):
                ctx.reject(
                    "inconsistent_active_sensing_grid_map_join",
                    item_path,
                    "主动观测融合摘要必须与全局 GridMap typed report 严格一致。",
                )
        if accepted_endpoint_count == 0:
            # minimum=1 已给出类型化错误；此分支保留更具体的合同语义。
            ctx.reject(
                "empty_active_sensing_fusion",
                f"{item_path}.accepted_endpoint_count",
                "canonical empty 不能推进主动观测融合门。",
            )
    controller_path = f"{attempt_path}.controller_command_aggregate"
    controller_aggregate = ctx.required_mapping(
        attempt,
        "controller_command_aggregate",
        controller_path,
    )
    _validate_active_sensing_command_aggregate(
        ctx,
        controller_aggregate,
        controller_path,
        command_length=6,
    )
    controller_statuses: list[Mapping[str, Any]] = []
    controller_status_sample_counts: list[int] = []
    controller_status_sequences: list[int] = []
    controller_acceptance_sequences: list[int] = []
    controller_header_stamps_ns: list[int] = []
    controller_receipt_timestamps: list[float] = []
    for key in ("first_status", "last_status"):
        status_path = f"{controller_path}.{key}"
        status = ctx.required_mapping(controller_aggregate, key, status_path)
        status_sequence, acceptance_sequence = _validate_controller_status(
            ctx,
            status,
            status_path,
            path_stamp_ns=evidence.path_stamp_ns,
        )
        if status_sequence is not None:
            controller_status_sequences.append(status_sequence)
        if acceptance_sequence is not None:
            controller_acceptance_sequences.append(acceptance_sequence)
        status_header = status.get("header")
        status_header_stamp_ns = (
            status_header.get("stamp_ns")
            if isinstance(status_header, Mapping)
            else None
        )
        if isinstance(status_header_stamp_ns, int) and not isinstance(
            status_header_stamp_ns,
            bool,
        ):
            controller_header_stamps_ns.append(status_header_stamp_ns)
        status_receipt_timestamp = status.get("receipt_timestamp")
        if isinstance(status_receipt_timestamp, (int, float)) and not isinstance(
            status_receipt_timestamp,
            bool,
        ):
            controller_receipt_timestamps.append(float(status_receipt_timestamp))
        controller_statuses.append(status)
        status_aggregate = ctx.required_mapping(
            status,
            "command_aggregate",
            f"{status_path}.command_aggregate",
        )
        _validate_active_sensing_command_aggregate(
            ctx,
            status_aggregate,
            f"{status_path}.command_aggregate",
            command_length=6,
        )
        status_sample_count = status_aggregate.get("sample_count")
        if isinstance(status_sample_count, int) and not isinstance(
            status_sample_count,
            bool,
        ):
            controller_status_sample_counts.append(status_sample_count)
        if status.get("active_sensing_yaw_only") is not True:
            ctx.reject(
                "wrong_active_sensing_controller_identity",
                f"{status_path}.active_sensing_yaw_only",
                "controller 聚合必须由结构化 yaw-only identity 产生。",
                actual=status.get("active_sensing_yaw_only"),
            )
        if _mapping_identity_tuple(status.get("identity")) != active_identity:
            ctx.reject(
                "wrong_active_sensing_controller_identity",
                f"{status_path}.identity",
                "controller 聚合必须精确匹配 planner 主动观测 identity。",
            )
        if status.get("accepted") is not True or status.get(
            "trajectory_valid"
        ) is not True:
            ctx.reject(
                "missing_active_sensing_controller_acceptance",
                status_path,
                "主动观测 controller 状态必须为 accepted 且 trajectory_valid。",
            )
        if status.get("is_final") is not False or status.get(
            "emergency_stop"
        ) is not False:
            ctx.reject(
                "invalid_active_sensing_controller_state",
                status_path,
                "主动观测 controller 状态必须为 non-final、non-emergency。",
            )
        if key == "first_status" and status.get("event") != 1:
            ctx.reject(
                "missing_active_sensing_controller_acceptance",
                f"{status_path}.event",
                "第一条主动观测 controller 快照必须是 EVENT_ACCEPTED。",
                actual=status.get("event"),
            )
        if status.get("state") not in {
            _CONTROLLER_STATE_ALIGNING_YAW,
            _CONTROLLER_STATE_TRACKING,
        }:
            ctx.reject(
                "wrong_active_sensing_controller_state",
                f"{status_path}.state",
                "主动观测只能处于 ALIGNING_YAW 或 TRACKING。",
                actual=status.get("state"),
            )
        if key == "first_status":
            if status_sample_count != 1:
                ctx.reject(
                    "invalid_active_sensing_first_controller_sample",
                    f"{status_path}.command_aggregate.sample_count",
                    "第一条 controller 快照必须只包含首个同步零速样本。",
                    actual=status_sample_count,
                )
            for maximum_key in ("max_abs_vx", "max_abs_vy", "max_abs_wz"):
                if status_aggregate.get(maximum_key) != 0.0:
                    ctx.reject(
                        "invalid_active_sensing_first_controller_sample",
                        f"{status_path}.command_aggregate.{maximum_key}",
                        "第一条 controller 快照的命令包络必须严格为零。",
                        actual=status_aggregate.get(maximum_key),
                    )
    if (
        len(controller_status_sequences) == 2
        and controller_status_sequences[1] <= controller_status_sequences[0]
    ):
        ctx.reject(
            "invalid_active_sensing_controller_sequence",
            f"{controller_path}.last_status.status_sequence",
            "active first/last controller status sequence 必须严格递增。",
            actual=controller_status_sequences,
        )
    if (
        len(controller_acceptance_sequences) == 2
        and (
            controller_acceptance_sequences[0] <= 0
            or controller_acceptance_sequences[1]
            != controller_acceptance_sequences[0]
        )
    ):
        ctx.reject(
            "invalid_active_sensing_controller_sequence",
            f"{controller_path}.last_status.acceptance_sequence",
            "active first/last 必须绑定同一正 acceptance sequence。",
            actual=controller_acceptance_sequences,
        )
    if (
        len(controller_header_stamps_ns) == 2
        and controller_header_stamps_ns[1] <= controller_header_stamps_ns[0]
    ):
        ctx.reject(
            "invalid_active_sensing_controller_sequence",
            f"{controller_path}.last_status.header.stamp_ns",
            "active first/last controller header 时间必须严格递增。",
            actual=controller_header_stamps_ns,
        )
    if (
        controller_header_stamps_ns
        and active_identity is not None
        and active_trajectory_duration_s is not None
        and controller_header_stamps_ns[-1]
        > active_identity[2]
        + int(round(active_trajectory_duration_s * 1.0e9))
    ):
        ctx.reject(
            "active_sensing_trajectory_expired",
            f"{controller_path}.last_status.header.stamp_ns",
            "active controller 状态必须位于主动观测 B-spline 总时限内。",
            actual=controller_header_stamps_ns[-1],
        )
    if (
        controller_header_stamps_ns
        and settle_stamp_ns is not None
        and canonical_yaw is not None
        and canonical_settle_values is not None
        and canonical_yaw[3] > 0.0
    ):
        minimum_rotation_angle = max(
            0.0,
            abs(canonical_yaw[2])
            - _ACTIVE_SENSING_CONTROLLER_START_YAW_TOLERANCE_RAD
            - canonical_settle_values[0],
        )
        minimum_settle_elapsed_s = (
            minimum_rotation_angle / canonical_yaw[3]
            + canonical_settle_values[2]
        )
        actual_settle_elapsed_s = (
            settle_stamp_ns - controller_header_stamps_ns[0]
        ) * 1.0e-9
        if actual_settle_elapsed_s + 1.0e-9 < minimum_settle_elapsed_s:
            ctx.reject(
                "active_sensing_not_settled",
                f"{attempt_path}.yaw_stable.active_sensing.settle_stamp_ns",
                "settle 必须覆盖 controller 接受后的保守剩余旋转量与稳定窗口。",
                actual=actual_settle_elapsed_s,
            )
    if (
        len(controller_receipt_timestamps) == 2
        and controller_receipt_timestamps[1] <= controller_receipt_timestamps[0]
    ):
        ctx.reject(
            "invalid_active_sensing_controller_sequence",
            f"{controller_path}.last_status.receipt_timestamp",
            "active first/last controller 接收时间必须严格递增。",
            actual=controller_receipt_timestamps,
        )
    if (
        controller_header_stamps_ns
        and active_identity is not None
        and controller_header_stamps_ns[0] < active_identity[2]
    ):
        ctx.reject(
            "invalid_active_sensing_controller_sequence",
            f"{controller_path}.first_status.header.stamp_ns",
            "controller 接受快照不能早于主动观测轨迹 start_time。",
            actual=controller_header_stamps_ns[0],
        )
    if (
        len(controller_status_sample_counts) == 2
        and controller_status_sample_counts[0] > controller_status_sample_counts[1]
    ):
        ctx.reject(
            "active_sensing_command_aggregate_regressed",
            f"{controller_path}.last_status.command_aggregate.sample_count",
            "controller active command sample_count 不能从 first_status 回退。",
            actual=controller_status_sample_counts,
        )
    last_controller_aggregate = controller_statuses[-1].get("command_aggregate")
    if isinstance(last_controller_aggregate, Mapping):
        for key in (
            "sample_count",
            "first_command",
            "max_abs_vx",
            "max_abs_vy",
            "max_abs_wz",
            "violation_count",
        ):
            if last_controller_aggregate.get(key) != controller_aggregate.get(key):
                ctx.reject(
                    "inconsistent_active_sensing_command_aggregate",
                    f"{controller_path}.{key}",
                    "外层 controller 聚合必须等于最后一条 typed status 快照。",
                    actual=controller_aggregate.get(key),
                )
    else:
        ctx.reject(
            "missing_active_sensing_command_aggregate",
            f"{controller_path}.last_status.command_aggregate",
            "最后 controller status 必须内嵌同 identity 命令聚合。",
        )
    controller_max_abs_wz = controller_aggregate.get("max_abs_wz")
    if (
        not isinstance(controller_max_abs_wz, (int, float))
        or isinstance(controller_max_abs_wz, bool)
        or float(controller_max_abs_wz) <= 0.0
    ):
        ctx.reject(
            "missing_active_sensing_rotation_command",
            f"{controller_path}.max_abs_wz",
            "成功主动观测必须证明 controller 实际输出过非零原地旋转命令。",
            actual=controller_max_abs_wz,
        )

    policy_path = f"{attempt_path}.policy_command_aggregate"
    policy_aggregate = ctx.required_mapping(
        attempt,
        "policy_command_aggregate",
        policy_path,
    )
    policy_first_command = _validate_active_sensing_command_aggregate(
        ctx,
        policy_aggregate,
        policy_path,
        command_length=3,
    )
    policy_write_plan_ids: set[int] = set()
    policy_write_sequences: dict[str, int] = {}
    policy_write_timestamps: dict[str, float] = {}
    policy_snapshot_status_sequences: list[int] = []
    for key in (
        "first_write",
        "first_rotation_write",
        "maximum_abs_wz_write",
        "last_write",
    ):
        write_path = f"{policy_path}.{key}"
        write = ctx.required_mapping(policy_aggregate, key, write_path)
        write_sequence = ctx.required_integer(
            write,
            "write_sequence",
            f"{write_path}.write_sequence",
            minimum=1,
        )
        if write_sequence is not None:
            policy_write_sequences[key] = write_sequence
        write_timestamp = ctx.required_number(
            write,
            "timestamp",
            f"{write_path}.timestamp",
            minimum=0.0,
        )
        if write_timestamp is not None:
            policy_write_timestamps[key] = write_timestamp
        raw_consumed_gate = ctx.required_mapping(
            write,
            "policy_navigation_gate_consumed_report",
            f"{write_path}.policy_navigation_gate_consumed_report",
        )
        canonical_consumed_gate = ctx.required_mapping(
            write,
            "navigation_gate_diagnostics",
            f"{write_path}.navigation_gate_diagnostics",
        )
        if raw_consumed_gate != canonical_consumed_gate:
            ctx.reject(
                "inconsistent_active_sensing_policy_permit",
                f"{write_path}.navigation_gate_diagnostics",
                "active policy 的规范 gate 别名必须与 runtime raw 消费报告完全一致。",
            )
        _expect_exact_field(
            ctx,
            write,
            "owner_id",
            "scan_cmd_vel",
            f"{write_path}.owner_id",
        )
        motion_allowed = ctx.required_boolean(
            write,
            "motion_allowed",
            f"{write_path}.motion_allowed",
        )
        strict_tracking_write = bool(
            key in {"first_rotation_write", "maximum_abs_wz_write"}
            or (key == "last_write" and motion_allowed is True)
        )
        if strict_tracking_write:
            _expect_true_field(
                ctx,
                write,
                "motion_allowed",
                f"{write_path}.motion_allowed",
            )
            _expect_false_field(
                ctx,
                write,
                "navigation_emergency_stop_latched",
                f"{write_path}.navigation_emergency_stop_latched",
            )
            _expect_false_field(
                ctx,
                write,
                "navigation_cmd_vel_inhibited",
                f"{write_path}.navigation_cmd_vel_inhibited",
            )
            tracking_stop_reasons = _validate_string_array(
                ctx,
                ctx.field(
                    write,
                    "stop_reasons",
                    f"{write_path}.stop_reasons",
                ),
                f"{write_path}.stop_reasons",
            )
            if tracking_stop_reasons:
                ctx.reject(
                    "invalid_active_sensing_policy_permit",
                    f"{write_path}.stop_reasons",
                    "active 旋转实写必须由无停车原因的 TRACKING permit 放行。",
                    actual=list(tracking_stop_reasons),
                )
        if key == "first_write":
            (
                permit_status_sequence,
                permit_state_revision,
            ) = _validate_active_sensing_zero_policy_write(
                ctx,
                write,
                write_path,
            )
            consumed_write_sequence = write_sequence
        elif strict_tracking_write:
            (
                consumed_write_sequence,
                permit_status_sequence,
                permit_state_revision,
            ) = _validate_consumed_tracking_evidence(
                ctx,
                write,
                write_path,
                pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
                path_stamp_ns=evidence.path_stamp_ns,
            )
        else:
            _validate_active_sensing_safety_zero_policy_write(
                ctx,
                write,
                write_path,
            )
            consumed_write_sequence = write_sequence
            permit_status_sequence = None
            permit_state_revision = None
        limited_target = _validate_vector(
            ctx,
            ctx.field(write, "limited_target", f"{write_path}.limited_target"),
            f"{write_path}.limited_target",
            length=3,
        )
        if strict_tracking_write and limited_target is not None:
            limited_vx, limited_vy, limited_wz = limited_target
            if abs(limited_vx) > 1.0e-12 or abs(limited_vy) > 1.0e-12:
                ctx.reject(
                    "active_sensing_translation_command",
                    f"{write_path}.limited_target",
                    "主动观测的 policy 限幅目标必须严格零平移。",
                    actual=list(limited_target),
                )
            if (
                abs(limited_wz)
                > _ACTIVE_SENSING_MAX_YAW_RATE_RAD_S + 1.0e-12
            ):
                ctx.reject(
                    "active_sensing_yaw_rate_exceeded",
                    f"{write_path}.limited_target",
                    "主动观测的 policy 限幅目标不得超过 0.20 rad/s。",
                    actual=list(limited_target),
                )
        raw_observed_report = write.get("navigation_status_observed_report")
        if raw_observed_report is None and not strict_tracking_write:
            observed_write_sequence = write_sequence
            observed_status_sequence = None
            observed_state_revision = None
        else:
            (
                observed_write_sequence,
                observed_status_sequence,
                observed_state_revision,
            ) = _validate_observed_status_evidence(
                ctx,
                write,
                write_path,
                pct_goal_stamp_ns=evidence.pct_goal_stamp_ns,
                path_stamp_ns=evidence.path_stamp_ns,
            )
        if consumed_write_sequence != write_sequence or (
            observed_write_sequence != write_sequence
        ):
            ctx.reject(
                "invalid_active_sensing_policy_sequence",
                f"{write_path}.write_sequence",
                "active policy 的 gate、observed status 与实际写入必须共享同一 write sequence。",
                actual={
                    "write": write_sequence,
                    "gate": consumed_write_sequence,
                    "observed": observed_write_sequence,
                },
            )
        if strict_tracking_write and (
            permit_status_sequence != observed_status_sequence
            or permit_state_revision != observed_state_revision
        ):
            ctx.reject(
                "invalid_active_sensing_policy_permit",
                write_path,
                "policy gate permit 必须与同拍观测的 supervisor status/revision 一致。",
                actual={
                    "permit_status_sequence": permit_status_sequence,
                    "observed_status_sequence": observed_status_sequence,
                    "permit_state_revision": permit_state_revision,
                    "observed_state_revision": observed_state_revision,
                },
            )
        command = _validate_vector(
            ctx,
            ctx.field(write, "written_command", f"{write_path}.written_command"),
            f"{write_path}.written_command",
            length=3,
        )
        if command is not None:
            vx, vy, wz = command
            if abs(vx) > 1.0e-12 or abs(vy) > 1.0e-12:
                ctx.reject(
                    "active_sensing_translation_command",
                    f"{write_path}.written_command",
                    "主动观测的每条保留 policy 实写都必须严格零平移。",
                    actual=list(command),
                )
            if abs(wz) > _ACTIVE_SENSING_MAX_YAW_RATE_RAD_S + 1.0e-12:
                ctx.reject(
                    "active_sensing_yaw_rate_exceeded",
                    f"{write_path}.written_command",
                    "主动观测的每条保留 policy 实写都不得超过 0.20 rad/s。",
                    actual=list(command),
                )
            for component, maximum_key in (
                (vx, "max_abs_vx"),
                (vy, "max_abs_vy"),
                (wz, "max_abs_wz"),
            ):
                maximum = policy_aggregate.get(maximum_key)
                if (
                    isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and abs(component) > float(maximum) + 1.0e-12
                ):
                    ctx.reject(
                        "inconsistent_active_sensing_command_aggregate",
                        f"{write_path}.written_command",
                        "first/last policy 实写不能超过声明的聚合包络。",
                        actual=list(command),
                    )
            if key == "first_rotation_write" and abs(wz) <= 1.0e-12:
                ctx.reject(
                    "missing_active_sensing_rotation_command",
                    f"{write_path}.written_command",
                    "first_rotation_write 必须证明一次真实非零原地旋转。",
                    actual=list(command),
                )
            if key == "maximum_abs_wz_write":
                policy_maximum = policy_aggregate.get("max_abs_wz")
                if (
                    not isinstance(policy_maximum, (int, float))
                    or isinstance(policy_maximum, bool)
                    or not math.isclose(
                        abs(wz),
                        float(policy_maximum),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    ctx.reject(
                        "inconsistent_active_sensing_command_aggregate",
                        f"{write_path}.written_command",
                        "maximum_abs_wz_write 必须精确证明 policy 聚合的最大角速度。",
                        actual=list(command),
                    )
        snapshot = ctx.required_mapping(
            write,
            "scan_controller_status_snapshot",
            f"{write_path}.scan_controller_status_snapshot",
        )
        snapshot_status_sequence, snapshot_acceptance_sequence = (
            _validate_controller_status(
                ctx,
                snapshot,
                f"{write_path}.scan_controller_status_snapshot",
                path_stamp_ns=evidence.path_stamp_ns,
            )
        )
        if snapshot_status_sequence is not None:
            policy_snapshot_status_sequences.append(snapshot_status_sequence)
        if _mapping_identity_tuple(snapshot.get("identity")) != active_identity:
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                f"{write_path}.scan_controller_status_snapshot.identity",
                "policy 实际写入必须由同一主动观测 controller identity 许可。",
            )
        if snapshot.get("active_sensing_yaw_only") is not True:
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                f"{write_path}.scan_controller_status_snapshot.active_sensing_yaw_only",
                "policy 写入快照必须显式标记 active_sensing_yaw_only。",
                actual=snapshot.get("active_sensing_yaw_only"),
            )
        if snapshot.get("accepted") is not True or snapshot.get(
            "trajectory_valid"
        ) is not True:
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                f"{write_path}.scan_controller_status_snapshot",
                "policy 写入必须由 accepted 且 trajectory_valid 的主动观测快照许可。",
            )
        if snapshot.get("is_final") is not False or snapshot.get(
            "emergency_stop"
        ) is not False:
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                f"{write_path}.scan_controller_status_snapshot",
                "policy 写入快照必须为 non-final、non-emergency。",
            )
        if snapshot.get("state") not in {
            _CONTROLLER_STATE_ALIGNING_YAW,
            _CONTROLLER_STATE_TRACKING,
        }:
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                f"{write_path}.scan_controller_status_snapshot.state",
                "policy 主动观测写入只能由 ALIGNING_YAW 或 TRACKING 状态许可。",
                actual=snapshot.get("state"),
            )
        if (
            controller_acceptance_sequences
            and snapshot_acceptance_sequence
            != controller_acceptance_sequences[0]
        ):
            ctx.reject(
                "wrong_active_sensing_policy_identity",
                (
                    f"{write_path}.scan_controller_status_snapshot."
                    "acceptance_sequence"
                ),
                "policy 写入必须绑定本次主动观测的 controller acceptance sequence。",
                actual=snapshot_acceptance_sequence,
            )
        snapshot_receipt_timestamp = snapshot.get("receipt_timestamp")
        if (
            write_timestamp is not None
            and isinstance(snapshot_receipt_timestamp, (int, float))
            and not isinstance(snapshot_receipt_timestamp, bool)
            and write_timestamp < float(snapshot_receipt_timestamp)
        ):
            ctx.reject(
                "invalid_active_sensing_policy_sequence",
                f"{write_path}.timestamp",
                "policy 实写时间不能早于所消费的 controller 快照。",
                actual=write_timestamp,
            )
        if raw_observed_report is not None:
            observed = ctx.mapping(
                raw_observed_report,
                f"{write_path}.navigation_status_observed_report",
            )
            navigation_status = ctx.required_mapping(
                observed,
                "status",
                f"{write_path}.navigation_status_observed_report.status",
            )
            observed_path_stamp = ctx.required_integer(
                navigation_status,
                "active_path_stamp_ns",
                (
                    f"{write_path}.navigation_status_observed_report.status."
                    "active_path_stamp_ns"
                ),
                minimum=1,
            )
            if observed_path_stamp != evidence.path_stamp_ns:
                ctx.reject(
                    "wrong_active_sensing_policy_identity",
                    (
                        f"{write_path}.navigation_status_observed_report.status."
                        "active_path_stamp_ns"
                    ),
                    "主动观测 policy 写入必须继续绑定同一 Path。",
                    actual=observed_path_stamp,
                )
            observed_plan_id = ctx.required_integer(
                navigation_status,
                "pct_plan_id",
                (
                    f"{write_path}.navigation_status_observed_report.status."
                    "pct_plan_id"
                ),
                minimum=1,
            )
            if observed_plan_id is not None:
                policy_write_plan_ids.add(observed_plan_id)
        if key == "first_write" and (
            command is not None
            and policy_first_command is not None
            and command != policy_first_command
        ):
            ctx.reject(
                "inconsistent_active_sensing_command_aggregate",
                f"{write_path}.written_command",
                "first_write 必须等于 policy 聚合的 first_command。",
                actual=list(command),
            )

    ordered_policy_keys = (
        "first_write",
        "first_rotation_write",
        "maximum_abs_wz_write",
        "last_write",
    )
    if all(key in policy_write_sequences for key in ordered_policy_keys):
        ordered_sequences = [
            policy_write_sequences[key] for key in ordered_policy_keys
        ]
        if (
            ordered_sequences[1] <= ordered_sequences[0]
            or any(
                current < previous
                for previous, current in zip(
                    ordered_sequences[1:-1],
                    ordered_sequences[2:],
                    strict=True,
                )
            )
        ):
            ctx.reject(
                "invalid_active_sensing_policy_sequence",
                f"{policy_path}.last_write.write_sequence",
                "policy 写入必须满足 first < first_rotation <= maximum <= last。",
                actual={
                    key: policy_write_sequences[key]
                    for key in ordered_policy_keys
                },
            )
        for previous_key, current_key in zip(
            ordered_policy_keys[1:-1],
            ordered_policy_keys[2:],
            strict=True,
        ):
            if (
                policy_write_sequences[previous_key]
                == policy_write_sequences[current_key]
                and policy_aggregate.get(previous_key)
                != policy_aggregate.get(current_key)
            ):
                ctx.reject(
                    "inconsistent_active_sensing_policy_sequence",
                    f"{policy_path}.{current_key}",
                    "共享 write sequence 的主动观测证据必须是同一完整实写。",
                )
    policy_sample_count = policy_aggregate.get("sample_count")
    if (
        "first_write" in policy_write_sequences
        and "last_write" in policy_write_sequences
        and isinstance(policy_sample_count, int)
        and not isinstance(policy_sample_count, bool)
        and policy_sample_count
        != policy_write_sequences["last_write"]
        - policy_write_sequences["first_write"]
        + 1
    ):
        ctx.reject(
            "invalid_active_sensing_policy_sequence",
            f"{policy_path}.sample_count",
            "active policy 聚合必须覆盖 first 到 last 间每个连续全局 write sequence。",
            actual=policy_sample_count,
        )
    if all(key in policy_write_timestamps for key in ordered_policy_keys):
        ordered_timestamps = [
            policy_write_timestamps[key] for key in ordered_policy_keys
        ]
        if (
            ordered_timestamps[1] <= ordered_timestamps[0]
            or any(
                current < previous
                for previous, current in zip(
                    ordered_timestamps[1:-1],
                    ordered_timestamps[2:],
                    strict=True,
                )
            )
        ):
            ctx.reject(
                "invalid_active_sensing_policy_sequence",
                f"{policy_path}.last_write.timestamp",
                "policy 时间必须满足 first < first_rotation <= maximum <= last。",
                actual={
                    key: policy_write_timestamps[key]
                    for key in ordered_policy_keys
                },
            )
    first_rotation_timestamp = policy_write_timestamps.get(
        "first_rotation_write"
    )
    if (
        first_rotation_timestamp is not None
        and settle_stamp_ns is not None
        and first_rotation_timestamp
        > settle_stamp_ns * 1.0e-9 + 1.0e-9
    ):
        ctx.reject(
            "active_sensing_rotation_after_settle",
            f"{policy_path}.first_rotation_write.timestamp",
            "首条真实非零 policy 旋转必须不晚于 yaw settle。",
            actual=first_rotation_timestamp,
        )
    last_policy_timestamp = policy_write_timestamps.get("last_write")
    if (
        last_policy_timestamp is not None
        and completed_receipt_timestamp is not None
        and last_policy_timestamp > completed_receipt_timestamp + 1.0e-9
    ):
        ctx.reject(
            "active_sensing_policy_write_after_completion",
            f"{policy_path}.last_write.timestamp",
            "active policy 窗口尾拍不能晚于 COMPLETED 接收时刻。",
            actual=last_policy_timestamp,
        )
    if (
        len(policy_snapshot_status_sequences) >= 2
        and any(
            current < previous
            for previous, current in zip(
                policy_snapshot_status_sequences[:-1],
                policy_snapshot_status_sequences[1:],
                strict=True,
            )
        )
    ):
        ctx.reject(
            "invalid_active_sensing_policy_sequence",
            f"{policy_path}.last_write.scan_controller_status_snapshot.status_sequence",
            "policy 消费的 controller status sequence 不能回退。",
            actual=policy_snapshot_status_sequences,
        )
    if (
        len(controller_status_sequences) == 2
        and any(
            sequence < controller_status_sequences[0]
            or sequence > controller_status_sequences[-1]
            for sequence in policy_snapshot_status_sequences
        )
    ):
        ctx.reject(
            "invalid_active_sensing_policy_sequence",
            f"{policy_path}.first_write.scan_controller_status_snapshot.status_sequence",
            "policy 快照 status sequence 必须落在 active controller first/last 窗口内。",
            actual=policy_snapshot_status_sequences,
        )
    if (
        policy_write_sequences
        and evidence.policy_write_count is not None
        and max(policy_write_sequences.values()) > evidence.policy_write_count
    ):
        ctx.reject(
            "active_sensing_policy_write_outside_lifecycle",
            f"{policy_path}.last_write.write_sequence",
            "主动观测 policy 实写必须位于全局 policy lifecycle 序号边界内。",
            actual=max(policy_write_sequences.values()),
        )
    policy_max_abs_wz = policy_aggregate.get("max_abs_wz")
    if (
        not isinstance(policy_max_abs_wz, (int, float))
        or isinstance(policy_max_abs_wz, bool)
        or float(policy_max_abs_wz) <= 0.0
    ):
        ctx.reject(
            "missing_active_sensing_rotation_command",
            f"{policy_path}.max_abs_wz",
            "成功主动观测必须证明 policy 实际写入过非零原地旋转命令。",
            actual=policy_max_abs_wz,
        )
    if (
        isinstance(policy_max_abs_wz, (int, float))
        and not isinstance(policy_max_abs_wz, bool)
        and isinstance(controller_max_abs_wz, (int, float))
        and not isinstance(controller_max_abs_wz, bool)
        and float(policy_max_abs_wz)
        > float(controller_max_abs_wz) + 1.0e-12
    ):
        ctx.reject(
            "active_sensing_policy_amplified_command",
            f"{policy_path}.max_abs_wz",
            "policy 实写角速度包络不能放大 controller 输出包络。",
            actual=policy_max_abs_wz,
        )

    global_tracking_ring_path = (
        "$.simulation_report.navigation_policy_gate_lifecycle_report."
        "identity_verified_tracking_write_reports"
    )
    global_tracking_ring = ctx.required_sequence(
        evidence.lifecycle,
        "identity_verified_tracking_write_reports",
        global_tracking_ring_path,
    )
    dropped_global_tracking_writes = ctx.required_integer(
        evidence.lifecycle,
        "dropped_identity_verified_tracking_write_report_count",
        (
            "$.simulation_report.navigation_policy_gate_lifecycle_report."
            "dropped_identity_verified_tracking_write_report_count"
        ),
        minimum=0,
    )
    first_rotation_sequence = policy_write_sequences.get(
        "first_rotation_write"
    )
    retained_rotation_writes = [
        report
        for report in global_tracking_ring
        if isinstance(report, Mapping)
        and first_rotation_sequence is not None
        and report.get("write_sequence") == first_rotation_sequence
    ]
    if not retained_rotation_writes and dropped_global_tracking_writes == 0:
        ctx.reject(
            "missing_active_sensing_global_policy_join",
            global_tracking_ring_path,
            "未发生 ring 淘汰时，active 首条非零旋转必须进入全局 tracking ring。",
        )
    elif retained_rotation_writes:
        if len(retained_rotation_writes) != 1:
            ctx.reject(
                "duplicate_identity",
                global_tracking_ring_path,
                "全局 tracking ring 的 active 首条旋转 write sequence 必须唯一。",
                actual=len(retained_rotation_writes),
            )
        retained_active_write = retained_rotation_writes[0]
        first_rotation_write = policy_aggregate.get("first_rotation_write")
        if isinstance(first_rotation_write, Mapping):
            for field in (
                "write_sequence",
                "timestamp",
                "written_command",
                "navigation_gate_diagnostics",
                "scan_controller_status_snapshot",
            ):
                if (
                    retained_active_write.get(field)
                    != first_rotation_write.get(field)
                ):
                    ctx.reject(
                        "inconsistent_active_sensing_global_policy_join",
                        f"{global_tracking_ring_path}.{field}",
                        "全局 tracking ring 必须与 active 首条旋转实写完全一致。",
                    )

    pct_plan_ids_path = f"{attempt_path}.pct_plan_ids"
    pct_plan_ids_raw = ctx.required_sequence(
        attempt,
        "pct_plan_ids",
        pct_plan_ids_path,
    )
    pct_plan_ids: list[int] = []
    for index, raw_plan_id in enumerate(pct_plan_ids_raw):
        plan_id = ctx.integer(
            raw_plan_id,
            f"{pct_plan_ids_path}[{index}]",
            minimum=1,
        )
        if plan_id is not None:
            pct_plan_ids.append(plan_id)
    if len(pct_plan_ids) != 1 or len(set(pct_plan_ids)) != 1:
        ctx.reject(
            "wrong_active_sensing_pct_plan_generation",
            pct_plan_ids_path,
            "主动观测与恢复必须只绑定同一个正 PCT plan ID。",
            actual=pct_plan_ids,
        )
    if set(pct_plan_ids) != policy_write_plan_ids:
        ctx.reject(
            "wrong_active_sensing_pct_plan_generation",
            pct_plan_ids_path,
            "pct_plan_ids 必须由主动观测窗口的实际 policy 写入逐条证明。",
            actual={
                "reported": pct_plan_ids,
                "write_evidence": sorted(policy_write_plan_ids),
            },
        )

    recovery_path = f"{attempt_path}.recovery"
    recovery = ctx.required_mapping(attempt, "recovery", recovery_path)
    recovery_identity = _validate_full_trajectory_identity(
        ctx,
        ctx.field(recovery, "identity", f"{recovery_path}.identity"),
        f"{recovery_path}.identity",
        path_stamp_ns=evidence.path_stamp_ns,
    )
    if recovery_identity is not None and recovery_identity == active_identity:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.identity",
            "恢复轨迹必须是区别于主动观测的正常运动 identity。",
        )
    recovery_path_stamp = ctx.required_integer(
        recovery,
        "reference_path_stamp_ns",
        f"{recovery_path}.reference_path_stamp_ns",
        minimum=1,
    )
    if recovery_path_stamp != evidence.path_stamp_ns:
        ctx.reject(
            "wrong_active_sensing_recovery_path",
            f"{recovery_path}.reference_path_stamp_ns",
            "恢复轨迹必须继续使用同一 PCT reference Path。",
            actual=recovery_path_stamp,
        )
    recovery_plan_id = ctx.required_integer(
        recovery,
        "pct_plan_id",
        f"{recovery_path}.pct_plan_id",
        minimum=1,
    )
    if len(pct_plan_ids) == 1 and recovery_plan_id != pct_plan_ids[0]:
        ctx.reject(
            "wrong_active_sensing_pct_plan_generation",
            f"{recovery_path}.pct_plan_id",
            "恢复轨迹必须继续绑定主动观测前的同一 PCT plan ID。",
            actual=recovery_plan_id,
        )
    _expect_false_field(ctx, recovery, "stationary", f"{recovery_path}.stationary")
    recovery_controller_state = ctx.required_integer(
        recovery,
        "controller_state",
        f"{recovery_path}.controller_state",
        minimum=0,
    )
    if recovery_controller_state != _CONTROLLER_STATE_TRACKING:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.controller_state",
            "恢复 identity 必须被 controller 观测为 TRACKING。",
            actual=recovery_controller_state,
        )

    active_diagnostic_count = ctx.required_integer(
        bspline_lifecycle,
        "active_sensing_diagnostic_count",
        (
            "$.simulation_report.bspline_diagnostics_lifecycle_report."
            "active_sensing_diagnostic_count"
        ),
        minimum=0,
    )
    if active_diagnostic_count != len(event_reports):
        ctx.reject(
            "invalid_active_sensing_global_count",
            (
                "$.simulation_report.bspline_diagnostics_lifecycle_report."
                "active_sensing_diagnostic_count"
            ),
            "全局 B-spline lifecycle 必须精确计入全部主动观测 typed 事件。",
            actual=active_diagnostic_count,
        )
    first_global_diagnostic_sequence = ctx.required_integer(
        bspline_lifecycle,
        "first_diagnostic_sequence",
        f"{bspline_lifecycle_path}.first_diagnostic_sequence",
        minimum=1,
    )
    last_global_diagnostic_sequence = ctx.required_integer(
        bspline_lifecycle,
        "last_diagnostic_sequence",
        f"{bspline_lifecycle_path}.last_diagnostic_sequence",
        minimum=1,
    )
    active_diagnostic_sequences = [
        report.get("diagnostic_sequence")
        for report in event_reports
        if isinstance(report, Mapping)
        and isinstance(report.get("diagnostic_sequence"), int)
        and not isinstance(report.get("diagnostic_sequence"), bool)
    ]
    if (
        active_diagnostic_sequences
        and first_global_diagnostic_sequence is not None
        and last_global_diagnostic_sequence is not None
        and (
            active_diagnostic_sequences[0] < first_global_diagnostic_sequence
            or active_diagnostic_sequences[-1] > last_global_diagnostic_sequence
        )
    ):
        ctx.reject(
            "active_sensing_diagnostic_outside_lifecycle",
            f"{attempt_path}.event_reports",
            "主动观测 diagnostic sequence 必须位于全局 B-spline 生命周期边界内。",
            actual=active_diagnostic_sequences,
        )
    ordinary_bspline_identities = {
        identity
        for identity in (
            _mapping_identity_tuple(value)
            for value in ctx.required_sequence(
                bspline_lifecycle,
                "trajectory_identities",
                "$.simulation_report.bspline_diagnostics_lifecycle_report.trajectory_identities",
            )
        )
        if identity is not None
    }
    if active_identity in ordinary_bspline_identities:
        ctx.reject(
            "active_sensing_polluted_motion_lifecycle",
            "$.simulation_report.bspline_diagnostics_lifecycle_report.trajectory_identities",
            "主动观测 identity 不能计入普通运动 B-spline 数量。",
        )
    if recovery_identity is not None and recovery_identity not in ordinary_bspline_identities:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.identity",
            "恢复 identity 必须出现在普通运动 B-spline 生命周期中。",
        )
    ordinary_bspline_reports = ctx.required_sequence(
        bspline_lifecycle,
        "diagnostic_reports",
        f"{bspline_lifecycle_path}.diagnostic_reports",
    )
    recovery_bspline_reports = [
        report
        for report in ordinary_bspline_reports
        if isinstance(report, Mapping)
        and _mapping_identity_tuple(report.get("identity")) == recovery_identity
    ]
    if not recovery_bspline_reports:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.identity",
            "恢复 identity 必须有普通 B-spline typed report 证明。",
        )
    else:
        recovery_bspline_report = recovery_bspline_reports[-1]
        recovery_diagnostic_sequence = recovery_bspline_report.get(
            "diagnostic_sequence"
        )
        if (
            active_diagnostic_sequences
            and isinstance(recovery_diagnostic_sequence, int)
            and not isinstance(recovery_diagnostic_sequence, bool)
            and recovery_diagnostic_sequence
            <= active_diagnostic_sequences[-1]
        ):
            ctx.reject(
                "active_sensing_not_recovered",
                f"{recovery_path}.identity",
                "普通恢复 B-spline 必须严格晚于主动观测 COMPLETED。",
                actual=recovery_diagnostic_sequence,
            )

    controller_lifecycle = ctx.required_mapping(
        evidence.simulation,
        "scan_controller_status_lifecycle_report",
        "$.simulation_report.scan_controller_status_lifecycle_report",
    )
    active_status_count = ctx.required_integer(
        controller_lifecycle,
        "active_sensing_status_count",
        (
            "$.simulation_report.scan_controller_status_lifecycle_report."
            "active_sensing_status_count"
        ),
        minimum=0,
    )
    if active_status_count is not None and active_status_count < 2:
        ctx.reject(
            "invalid_active_sensing_global_count",
            (
                "$.simulation_report.scan_controller_status_lifecycle_report."
                "active_sensing_status_count"
            ),
            "全局 controller lifecycle 至少应计入 active first/last 两条快照。",
            actual=active_status_count,
        )
    first_global_status_sequence = ctx.required_integer(
        controller_lifecycle,
        "first_status_sequence",
        "$.simulation_report.scan_controller_status_lifecycle_report.first_status_sequence",
        minimum=1,
    )
    last_global_status_sequence = ctx.required_integer(
        controller_lifecycle,
        "last_status_sequence",
        "$.simulation_report.scan_controller_status_lifecycle_report.last_status_sequence",
        minimum=1,
    )
    if (
        controller_status_sequences
        and first_global_status_sequence is not None
        and last_global_status_sequence is not None
        and (
            controller_status_sequences[0] < first_global_status_sequence
            or controller_status_sequences[-1] > last_global_status_sequence
        )
    ):
        ctx.reject(
            "active_sensing_controller_outside_lifecycle",
            controller_path,
            "active controller status sequence 必须位于全局 lifecycle 边界内。",
            actual=controller_status_sequences,
        )
    ordinary_controller_identities = {
        identity
        for identity in (
            _mapping_identity_tuple(value)
            for value in ctx.required_sequence(
                controller_lifecycle,
                "accepted_trajectory_identities",
                "$.simulation_report.scan_controller_status_lifecycle_report.accepted_trajectory_identities",
            )
        )
        if identity is not None
    }
    if active_identity in ordinary_controller_identities:
        ctx.reject(
            "active_sensing_polluted_motion_lifecycle",
            "$.simulation_report.scan_controller_status_lifecycle_report.accepted_trajectory_identities",
            "主动观测 identity 不能计入普通 accepted 轨迹数量。",
        )
    if recovery_identity is not None and recovery_identity not in ordinary_controller_identities:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.identity",
            "恢复 identity 必须被普通 controller 生命周期接受。",
        )
    recovery_tracking_statuses = [
        status
        for status in ctx.required_sequence(
            controller_lifecycle,
            "tracking_status_reports",
            "$.simulation_report.scan_controller_status_lifecycle_report.tracking_status_reports",
        )
        if isinstance(status, Mapping)
        and _mapping_identity_tuple(status.get("identity")) == recovery_identity
    ]
    if not recovery_tracking_statuses:
        ctx.reject(
            "active_sensing_not_recovered",
            f"{recovery_path}.identity",
            "恢复 identity 必须有 TRACKING ControllerStatus typed report 证明。",
        )
    else:
        recovery_tracking_status = recovery_tracking_statuses[-1]
        recovery_status_sequence = recovery_tracking_status.get("status_sequence")
        if (
            controller_status_sequences
            and isinstance(recovery_status_sequence, int)
            and not isinstance(recovery_status_sequence, bool)
            and recovery_status_sequence <= controller_status_sequences[-1]
        ):
            ctx.reject(
                "active_sensing_not_recovered",
                f"{recovery_path}.identity",
                "恢复 TRACKING ControllerStatus 必须晚于 active controller 状态。",
                actual=recovery_status_sequence,
            )

    dynamic = ctx.required_mapping(
        evidence.simulation,
        "dynamic_navigation_evidence_report",
        "$.simulation_report.dynamic_navigation_evidence_report",
    )
    ordered_detour = ctx.required_mapping(
        dynamic,
        "ordered_detour",
        "$.simulation_report.dynamic_navigation_evidence_report.ordered_detour",
    )
    detour_identity = _mapping_identity_tuple(ordered_detour.get("identity"))
    trajectory_recovery = ctx.required_mapping(
        dynamic,
        "trajectory_recovery",
        "$.simulation_report.dynamic_navigation_evidence_report.trajectory_recovery",
    )
    recovery_policy_write = ctx.required_mapping(
        trajectory_recovery,
        "policy_identity_verified_tracking_write",
        (
            "$.simulation_report.dynamic_navigation_evidence_report."
            "trajectory_recovery.policy_identity_verified_tracking_write"
        ),
    )
    recovery_policy_write_sequence = ctx.required_integer(
        recovery_policy_write,
        "write_sequence",
        (
            "$.simulation_report.dynamic_navigation_evidence_report."
            "trajectory_recovery.policy_identity_verified_tracking_write."
            "write_sequence"
        ),
        minimum=1,
    )
    recovery_policy_write_timestamp = ctx.required_number(
        recovery_policy_write,
        "timestamp",
        (
            "$.simulation_report.dynamic_navigation_evidence_report."
            "trajectory_recovery.policy_identity_verified_tracking_write.timestamp"
        ),
        minimum=0.0,
    )
    if (
        "last_write" in policy_write_sequences
        and recovery_policy_write_sequence is not None
        and recovery_policy_write_sequence
        <= policy_write_sequences["last_write"]
    ):
        ctx.reject(
            "active_sensing_not_recovered",
            (
                "$.simulation_report.dynamic_navigation_evidence_report."
                "trajectory_recovery.policy_identity_verified_tracking_write."
                "write_sequence"
            ),
            "恢复运动的 policy 实写必须严格晚于主动观测 policy 窗口。",
            actual=recovery_policy_write_sequence,
        )
    if (
        recovery_policy_write_timestamp is not None
        and "last_write" in policy_write_timestamps
        and recovery_policy_write_timestamp
        < policy_write_timestamps["last_write"]
    ):
        ctx.reject(
            "active_sensing_not_recovered",
            (
                "$.simulation_report.dynamic_navigation_evidence_report."
                "trajectory_recovery.policy_identity_verified_tracking_write.timestamp"
            ),
            "恢复运动的 policy 实写时间不能早于主动观测最后一次实写。",
            actual=recovery_policy_write_timestamp,
        )
    dynamic_recovery_identity = _mapping_identity_tuple(
        trajectory_recovery.get("after_recovery_identity")
    )
    if active_identity is not None and active_identity in {
        detour_identity,
        dynamic_recovery_identity,
    }:
        ctx.reject(
            "active_sensing_polluted_motion_evidence",
            "$.simulation_report.dynamic_navigation_evidence_report",
            "主动观测 identity 不能冒充动态绕障或恢复运动轨迹。",
        )
    if (
        recovery_identity is not None
        and dynamic_recovery_identity is not None
        and dynamic_recovery_identity != recovery_identity
    ):
        ctx.reject(
            "wrong_active_sensing_recovery_identity",
            "$.simulation_report.dynamic_navigation_evidence_report.trajectory_recovery.after_recovery_identity",
            "主动观测恢复必须与动态验收使用同一正常运动 identity。",
        )


def _validate_dynamic_f1(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
) -> None:
    _validate_nonfreezing_policy_mode(ctx, summary, evidence, expected_task_id=17704)
    _validate_flat_task_floors(ctx, evidence.task_config, require_place_disabled=True)
    try:
        plan = resolve_dynamic_obstacle_plan(evidence.task_config)
    except (TypeError, ValueError) as exc:
        ctx.reject(
            "invalid_dynamic_configuration",
            "$.task_config.dynamic_obstacles",
            f"动态障碍任务合同非法：{exc}",
        )
        return
    if not plan.enabled:
        ctx.reject(
            "missing_dynamic_configuration",
            "$.task_config.dynamic_obstacles",
            "dynamic_f1 必须启用至少一个任务定义的动态障碍。",
        )
        return
    for index, obstacle in enumerate(plan.obstacles):
        if obstacle.floor_id != "F1" or obstacle.surface_class != "flat":
            ctx.reject(
                "wrong_mode",
                f"$.task_config.dynamic_obstacles[{index}]",
                "dynamic_f1 只认证 F1 平地动态障碍。",
            )
    _validate_dynamic_configuration(ctx, evidence, plan)
    _validate_dynamic_runtime_and_lifecycle(ctx, evidence, plan)
    _validate_dynamic_raw_cloud_lifecycle(ctx, evidence, plan)
    _validate_controller_lifecycle(ctx, evidence, require_multiple_trajectories=True)
    _validate_dynamic_global_replan_contract(ctx, evidence)
    _validate_dynamic_navigation_evidence(ctx, evidence, plan)


def _validate_dynamic_replan_f1(
    ctx: _ValidationContext,
    summary: Mapping[str, Any],
    evidence: _CommonEvidence,
) -> None:
    """校验阻断推车触发的独立 PCT 全局重规划与清障后恢复。"""

    _validate_nonfreezing_policy_mode(
        ctx,
        summary,
        evidence,
        expected_task_id=17705,
    )
    _validate_flat_task_floors(
        ctx,
        evidence.task_config,
        require_place_disabled=True,
    )
    try:
        plan = resolve_dynamic_obstacle_plan(evidence.task_config)
    except (TypeError, ValueError) as exc:
        ctx.reject(
            "invalid_dynamic_configuration",
            "$.task_config.dynamic_obstacles",
            f"动态重规划任务合同非法：{exc}",
        )
        return
    if not plan.enabled:
        ctx.reject(
            "missing_dynamic_configuration",
            "$.task_config.dynamic_obstacles",
            "dynamic_replan_f1 必须启用阻断型动态障碍。",
        )
        return
    for index, obstacle in enumerate(plan.obstacles):
        obstacle_path = f"$.task_config.dynamic_obstacles[{index}]"
        if obstacle.floor_id != "F1" or obstacle.surface_class != "flat":
            ctx.reject(
                "wrong_mode",
                obstacle_path,
                "dynamic_replan_f1 只认证 F1 平地动态障碍。",
            )
        if obstacle.motion != "one_shot":
            ctx.reject(
                "wrong_mode",
                f"{obstacle_path}.motion",
                "阻断重规划验收要求 one_shot 推车离开后保持净空。",
                actual=obstacle.motion,
            )
    _validate_dynamic_configuration(ctx, evidence, plan)
    _validate_dynamic_runtime_and_lifecycle(ctx, evidence, plan)
    _validate_dynamic_raw_cloud_lifecycle(ctx, evidence, plan)
    _validate_controller_lifecycle(
        ctx,
        evidence,
        require_multiple_trajectories=True,
    )
    grid_reports, _ = _validate_grid_map_observation_lifecycle(ctx, evidence)
    bspline_reports = _validate_bspline_diagnostics_lifecycle(ctx, evidence)
    _validate_required_dynamic_replan_causality(
        ctx,
        evidence,
        plan,
        grid_reports,
        bspline_reports,
    )


def _validate_dynamic_global_replan_contract(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    *,
    required: bool = False,
) -> bool:
    """校验 request→new plan→TRACKING；专用模式可强制必须触发。"""

    path = "$.simulation_report.navigation_policy_gate_lifecycle_report"
    lifecycle = evidence.lifecycle
    requested = lifecycle.get("global_replan_requested_status_count")
    in_flight = lifecycle.get("global_replan_in_flight_status_count")
    request_ids = lifecycle.get("distinct_global_replan_request_ids")
    plan_ids = lifecycle.get("distinct_pct_plan_ids")
    triggered = bool(
        isinstance(requested, int) and requested > 0
        or isinstance(in_flight, int) and in_flight > 0
        or isinstance(request_ids, Sequence)
        and not isinstance(request_ids, (str, bytes, bytearray))
        and len(request_ids) > 0
    )
    if required and not triggered:
        ctx.reject(
            "missing_required_global_replan",
            path,
            "dynamic_replan_f1 必须真实触发 PCT 全局重规划，不能用局部绕障替代。",
        )
    if triggered:
        if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
            ctx.reject("incomplete_global_replan", f"{path}.global_replan_requested_status_count", "已触发 replan 时必须观测 request 状态。", actual=requested)
        if not isinstance(in_flight, int) or isinstance(in_flight, bool) or in_flight < 1:
            ctx.reject("incomplete_global_replan", f"{path}.global_replan_in_flight_status_count", "已触发 replan 时必须观测 in-flight 状态。", actual=in_flight)
        if not isinstance(request_ids, Sequence) or isinstance(request_ids, (str, bytes, bytearray)) or not request_ids:
            ctx.reject("incomplete_global_replan", f"{path}.distinct_global_replan_request_ids", "已触发 replan 时必须记录 request ID。")
        if not isinstance(plan_ids, Sequence) or isinstance(plan_ids, (str, bytes, bytearray)) or len(plan_ids) < 2:
            ctx.reject("incomplete_global_replan", f"{path}.distinct_pct_plan_ids", "全局重规划必须观测初始与新 PCT plan ID。", actual=plan_ids)
        _expect_true_field(
            ctx,
            lifecycle,
            "tracking_after_global_replan_observed",
            f"{path}.tracking_after_global_replan_observed",
        )
        recovery = lifecycle.get("global_replan_recovery_count")
        if not isinstance(recovery, int) or isinstance(recovery, bool) or recovery < 1:
            ctx.reject("incomplete_global_replan", f"{path}.global_replan_recovery_count", "新 PCT plan 后必须恢复 identity-valid TRACKING。", actual=recovery)
        _expect_false_field(
            ctx,
            lifecycle,
            "global_replan_pending_recovery",
            f"{path}.global_replan_pending_recovery",
        )
    gate_emergency = lifecycle.get("emergency_stop_observed_status_count")
    controller = evidence.simulation.get("scan_controller_status_lifecycle_report")
    controller_emergency = (
        controller.get("emergency_stop_status_count")
        if isinstance(controller, Mapping)
        else None
    )
    if isinstance(gate_emergency, int) and gate_emergency > 0 and not triggered:
        if not isinstance(controller_emergency, int) or controller_emergency < 1:
            ctx.reject(
                "missing_dynamic_recovery_evidence",
                f"{path}.emergency_stop_observed_status_count",
                "supervisor emergency stop 必须对应 controller 新轨迹恢复或完整 global replan。",
                actual=gate_emergency,
            )
    return triggered


def _validate_required_dynamic_replan_causality(
    ctx: _ValidationContext,
    evidence: _CommonEvidence,
    plan: DynamicObstaclePlan,
    grid_reports: Mapping[int, _GridMapDiagnosticEvidence],
    bspline_reports: Mapping[int, _BsplineDiagnosticEvidence],
) -> None:
    """把阻断推车、SCAN failure、PCT 新代与清障后恢复连接成同一因果链。"""

    lifecycle_path = (
        "$.simulation_report.navigation_policy_gate_lifecycle_report"
    )
    lifecycle = evidence.lifecycle
    if not _validate_dynamic_global_replan_contract(
        ctx,
        evidence,
        required=True,
    ):
        return

    maximum_failures = ctx.required_integer(
        lifecycle,
        "maximum_consecutive_scan_failures",
        f"{lifecycle_path}.maximum_consecutive_scan_failures",
        minimum=0,
    )
    if (
        maximum_failures is not None
        and maximum_failures < _EXPECTED_SCAN_REPLAN_FAILURE_COUNT
    ):
        ctx.reject(
            "missing_scan_failure_replan_trigger",
            f"{lifecycle_path}.maximum_consecutive_scan_failures",
            "阻断验收必须达到 SCAN 连续规划失败阈值后再请求 PCT 重规划。",
            actual=maximum_failures,
        )

    first_replan_path = f"{lifecycle_path}.first_global_replan_status"
    first_replan = ctx.required_mapping(
        lifecycle,
        "first_global_replan_status",
        first_replan_path,
    )
    first_report = ctx.required_mapping(
        first_replan,
        "navigation_status_observed_report",
        f"{first_replan_path}.navigation_status_observed_report",
    )
    first_status = ctx.required_mapping(
        first_report,
        "status",
        f"{first_replan_path}.navigation_status_observed_report.status",
    )
    replan_receipt = ctx.required_number(
        first_status,
        "receipt_timestamp",
        (
            f"{first_replan_path}.navigation_status_observed_report."
            "status.receipt_timestamp"
        ),
        minimum=0.0,
    )
    old_path_stamp_ns = ctx.required_integer(
        first_status,
        "active_path_stamp_ns",
        (
            f"{first_replan_path}.navigation_status_observed_report."
            "status.active_path_stamp_ns"
        ),
        minimum=1,
    )
    ctx.required_string(
        first_status,
        "reason",
        f"{first_replan_path}.navigation_status_observed_report.status.reason",
        nonempty=True,
    )
    first_replan_failures = ctx.required_integer(
        first_status,
        "consecutive_scan_failures",
        (
            f"{first_replan_path}.navigation_status_observed_report."
            "status.consecutive_scan_failures"
        ),
        minimum=0,
    )
    if (
        first_replan_failures is not None
        and first_replan_failures < _EXPECTED_SCAN_REPLAN_FAILURE_COUNT
    ):
        ctx.reject(
            "missing_scan_failure_replan_trigger",
            (
                f"{first_replan_path}.navigation_status_observed_report."
                "status.consecutive_scan_failures"
            ),
            "首次 PCT replan 快照本身必须已经达到 SCAN 连续失败阈值。",
            actual=first_replan_failures,
        )
    if (
        old_path_stamp_ns is not None
        and evidence.path_stamp_ns is not None
        and old_path_stamp_ns >= evidence.path_stamp_ns
    ):
        ctx.reject(
            "missing_new_replan_path_generation",
            (
                f"{first_replan_path}.navigation_status_observed_report."
                "status.active_path_stamp_ns"
            ),
            "PCT 重规划后的最终 Path stamp 必须严格晚于触发时的旧 Path。",
            actual=old_path_stamp_ns,
        )

    obstacle_by_id = {
        obstacle.obstacle_id: obstacle for obstacle in plan.obstacles
    }
    transition_candidates: list[
        tuple[float, _GridMapDiagnosticEvidence, str, Mapping[str, Any]]
    ] = []
    for report in grid_reports.values():
        if (
            not report.free_to_occupied_transition_count
            or report.receipt_timestamp is None
            or replan_receipt is None
            or report.receipt_timestamp >= replan_receipt
            or report.episode_elapsed_time_s is None
        ):
            continue
        raw_matches = report.report.get(
            "dynamic_obstacle_transition_hit_matches"
        )
        if (
            not isinstance(raw_matches, Sequence)
            or isinstance(raw_matches, (str, bytes, bytearray))
        ):
            continue
        for index, raw_match in enumerate(raw_matches):
            match_path = (
                "$.simulation_report.grid_map_observation_lifecycle_report."
                f"transition_match[{report.observation_sequence}][{index}]"
            )
            match = ctx.mapping(raw_match, match_path)
            obstacle_id = ctx.required_string(
                match,
                "obstacle_id",
                f"{match_path}.obstacle_id",
                nonempty=True,
            )
            obstacle = obstacle_by_id.get(obstacle_id or "")
            if obstacle is None:
                ctx.reject(
                    "wrong_identity",
                    f"{match_path}.obstacle_id",
                    "free→occupied transition 引用了任务外障碍。",
                    actual=obstacle_id,
                )
                continue
            point = _validate_vector(
                ctx,
                ctx.field(match, "point_world_xyz", f"{match_path}.point_world_xyz"),
                f"{match_path}.point_world_xyz",
                length=3,
            )
            tolerance = ctx.required_number(
                match,
                "association_tolerance_m",
                f"{match_path}.association_tolerance_m",
                minimum=0.0,
            )
            expected_state = obstacle.state_at(report.episode_elapsed_time_s)
            _validate_dynamic_state(
                ctx,
                ctx.field(match, "obstacle_state", f"{match_path}.obstacle_state"),
                f"{match_path}.obstacle_state",
                expected=expected_state.to_dict(),
            )
            if (
                point is not None
                and tolerance is not None
                and tolerance <= _MAX_POST_FILTER_HIT_TOLERANCE_M
                and _point_inside_oriented_cuboid(
                    point,
                    obstacle,
                    expected_state,
                    tolerance_m=tolerance,
                )
            ):
                transition_candidates.append(
                    (report.receipt_timestamp, report, obstacle.obstacle_id, match)
                )
            else:
                ctx.reject(
                    "missing_dynamic_replan_obstacle_transition",
                    match_path,
                    "重规划前 transition hit 必须位于同一时刻阻断推车的 OBB 内。",
                )

    if not transition_candidates:
        ctx.reject(
            "missing_dynamic_replan_obstacle_transition",
            "$.simulation_report.grid_map_observation_lifecycle_report",
            "PCT replan 前缺少同一动态推车的真实 free→occupied typed transition。",
        )
        return
    _, transition, obstacle_id, transition_match = max(
        transition_candidates,
        key=lambda item: item[0],
    )
    transition_sequence = transition.observation_sequence
    transition_point = _validate_vector(
        ctx,
        transition_match.get("point_world_xyz"),
        "$.dynamic_replan.transition.point_world_xyz",
        length=3,
    )
    old_reference_obstructed = False
    if transition_point is not None and old_path_stamp_ns is not None:
        for report in bspline_reports.values():
            if (
                report.identity is None
                or report.identity[0] != old_path_stamp_ns
                or not report.reference_samples
                or report.receipt_timestamp is None
                or replan_receipt is None
                or report.receipt_timestamp >= replan_receipt
            ):
                continue
            minimum_reference_distance = min(
                math.hypot(
                    sample[0] - transition_point[0],
                    sample[1] - transition_point[1],
                )
                for sample in report.reference_samples
            )
            if minimum_reference_distance < (
                _GO2_X5_DOUBLE_CYLINDER_RADIUS_M
                + _GO2_X5_DOUBLE_CYLINDER_OFFSET_M
                - 1.0e-9
            ):
                old_reference_obstructed = True
                break
    if not old_reference_obstructed:
        ctx.reject(
            "missing_replan_reference_obstruction",
            "$.simulation_report.bspline_diagnostics_lifecycle_report",
            "触发 replan 的推车 transition 必须实际阻断旧 active Path 的有序 reference。",
        )

    clear_candidates: list[
        tuple[float, _GridMapDiagnosticEvidence, Mapping[str, Any]]
    ] = []
    for report in grid_reports.values():
        if (
            not report.occupied_to_free_count
            or report.receipt_timestamp is None
            or replan_receipt is None
            or report.receipt_timestamp <= replan_receipt
            or transition_sequence is None
            or transition_sequence not in report.clear_transition_hit_sequences
        ):
            continue
        raw_matches = report.report.get(
            "dynamic_obstacle_explicit_miss_clear_matches"
        )
        if (
            not isinstance(raw_matches, Sequence)
            or isinstance(raw_matches, (str, bytes, bytearray))
        ):
            continue
        for raw_match in raw_matches:
            if not isinstance(raw_match, Mapping):
                continue
            if (
                raw_match.get("obstacle_id") == obstacle_id
                and raw_match.get("matched_hit_observation_sequence")
                == transition_sequence
                and raw_match.get("matched_hit_provenance_verified") is True
                and raw_match.get("sliding_reset_used") is False
            ):
                clear_candidates.append(
                    (report.receipt_timestamp, report, raw_match)
                )
    if not clear_candidates:
        ctx.reject(
            "missing_post_replan_explicit_clear",
            "$.simulation_report.grid_map_observation_lifecycle_report",
            "PCT replan 后缺少同一 occupied epoch 的 explicit-miss clear。",
        )
        return
    clear_receipt, clear_report, clear_match = min(
        clear_candidates,
        key=lambda item: item[0],
    )
    if clear_report.sliding_reset_count not in (None, 0):
        ctx.reject(
            "sliding_reset_cannot_certify_replan_clear",
            "$.dynamic_replan.clear.occupied_removed_by_sliding_reset_count",
            "阻断推车清除不能由 sliding-map reset 冒充。",
            actual=clear_report.sliding_reset_count,
        )
    matched_hit_point = _validate_vector(
        ctx,
        clear_match.get("matched_hit_point_world_xyz"),
        "$.dynamic_replan.clear.matched_hit_point_world_xyz",
        length=3,
    )
    if (
        transition_point is not None
        and matched_hit_point is not None
        and any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9)
            for left, right in zip(
                transition_point,
                matched_hit_point,
                strict=True,
            )
        )
    ):
        ctx.reject(
            "wrong_dynamic_replan_clear_provenance",
            "$.dynamic_replan.clear.matched_hit_point_world_xyz",
            "clear 必须直接引用触发 replan 前的同一 transition hit。",
        )

    last_tracking_path = (
        f"{lifecycle_path}.last_identity_verified_tracking_write"
    )
    last_tracking = ctx.required_mapping(
        lifecycle,
        "last_identity_verified_tracking_write",
        last_tracking_path,
    )
    tracking_timestamp = ctx.required_number(
        last_tracking,
        "timestamp",
        f"{last_tracking_path}.timestamp",
        minimum=0.0,
    )
    gate = ctx.required_mapping(
        last_tracking,
        "navigation_gate_diagnostics",
        f"{last_tracking_path}.navigation_gate_diagnostics",
    )
    permit = ctx.required_mapping(
        gate,
        "permit",
        f"{last_tracking_path}.navigation_gate_diagnostics.permit",
    )
    recovered_path_stamp_ns = ctx.required_integer(
        permit,
        "active_path_stamp_ns",
        (
            f"{last_tracking_path}.navigation_gate_diagnostics."
            "permit.active_path_stamp_ns"
        ),
        minimum=1,
    )
    written_command = _validate_vector(
        ctx,
        ctx.field(
            last_tracking,
            "written_command",
            f"{last_tracking_path}.written_command",
        ),
        f"{last_tracking_path}.written_command",
        length=3,
    )
    if written_command is not None and not any(
        abs(component) > 1.0e-12 for component in written_command
    ):
        ctx.reject(
            "missing_post_replan_policy_motion",
            f"{last_tracking_path}.written_command",
            "clear 后恢复必须包含最终 PCT Path 上的非零 policy 运动写入。",
            actual=list(written_command),
        )
    if tracking_timestamp is not None and tracking_timestamp <= clear_receipt:
        ctx.reject(
            "tracking_recovered_before_obstacle_clear",
            f"{last_tracking_path}.timestamp",
            "阻断验收的有效 policy TRACKING 必须发生在同障碍 explicit clear 之后。",
            actual=tracking_timestamp,
        )
    if (
        recovered_path_stamp_ns is not None
        and evidence.path_stamp_ns is not None
        and recovered_path_stamp_ns != evidence.path_stamp_ns
    ):
        ctx.reject(
            "wrong_replan_recovery_path_identity",
            (
                f"{last_tracking_path}.navigation_gate_diagnostics."
                "permit.active_path_stamp_ns"
            ),
            "clear 后 policy 恢复必须绑定最终 PCT replan Path。",
            actual=recovered_path_stamp_ns,
        )


def _claims_for_mode(
    mode: ValidationMode,
    *,
    require_active_sensing: bool = False,
) -> tuple[str, ...]:
    mode_claims: dict[ValidationMode, tuple[str, ...]] = {
        "static_stair": (
            "static_stair_root_lock_freeze_reached_terminal_hold",
            "stair_progress_and_path_identity_certified",
        ),
        "flat_policy": (
            "nonfreezing_flat_policy_tracking_write_consumed",
            "scan_controller_accepted_valid_trajectory_and_goal",
        ),
        "crossfloor_carry": (
            "crossfloor_f1_to_f2_goal_reached_with_original_go2_x5_policy",
            "stair_root_lock_freeze_and_release_observed",
            "post_stair_scan_tracking_and_goal_zero_hold_verified",
        ),
        "dynamic_f1": (
            "nonfreezing_flat_policy_tracking_write_consumed",
            "dynamic_obstacle_visible_colliding_kinematic_asset_configured",
            "dynamic_obstacle_physx_pose_writes_and_motion_span_recorded",
            "scan_multiple_accepted_trajectory_identities_recorded",
            *_DYNAMIC_FINAL_CLAIMS,
        ),
        "dynamic_replan_f1": (
            "nonfreezing_flat_policy_tracking_write_consumed",
            "dynamic_blocker_visible_colliding_one_shot_asset_configured",
            "scan_controller_accepted_valid_trajectory_and_goal",
            *_DYNAMIC_REPLAN_CLAIMS,
        ),
    }
    active_claims = _ACTIVE_SENSING_CLAIMS if require_active_sensing else ()
    return (*_COMMON_VALIDATED_CLAIMS, *mode_claims[mode], *active_claims)


def validate_pct_scan_live_summary(
    summary: Mapping[str, Any],
    mode: ValidationMode,
    *,
    source: str | None = None,
    require_active_sensing: bool = False,
) -> dict[str, Any]:
    """按指定 live 模式校验已解析的 summary，并返回机器可读报告。"""

    if mode not in VALIDATION_MODES:
        raise SummaryInputError(
            f"未知验收模式 {mode!r}；允许值：{', '.join(VALIDATION_MODES)}"
        )
    if not isinstance(summary, Mapping):
        raise SummaryInputError("summary 顶层必须是 JSON 对象。")
    if require_active_sensing and mode != "dynamic_f1":
        raise SummaryInputError(
            "--require-active-sensing 只允许用于 dynamic_f1 验收。"
        )
    ctx = _ValidationContext(mode=mode)
    evidence = _validate_common(ctx, summary)
    if mode == "static_stair":
        _validate_stair_mode(ctx, summary, evidence)
    elif mode == "flat_policy":
        _validate_flat_policy(ctx, summary, evidence)
    elif mode == "crossfloor_carry":
        _validate_crossfloor_carry(ctx, summary, evidence)
    elif mode == "dynamic_f1":
        _validate_dynamic_f1(ctx, summary, evidence)
        if require_active_sensing:
            _validate_active_sensing_lifecycle(ctx, evidence)
    else:
        _validate_dynamic_replan_f1(ctx, summary, evidence)

    valid = not ctx.errors
    identity = {
        "task_id": summary.get("task_id"),
        "episode_id": summary.get("episode_id"),
        "pct_goal_stamp_ns": evidence.pct_goal_stamp_ns,
        "live_reference_path_stamp_ns": evidence.path_stamp_ns,
        "first_observed_status_sequence": evidence.first_observed_sequence,
        "last_observed_status_sequence": evidence.last_observed_sequence,
        "policy_write_count": evidence.policy_write_count,
    }
    return {
        "schema": "pct_scan_live_summary_validation_v1",
        "mode": mode,
        "source": source,
        "valid": valid,
        "error_count": len(ctx.errors),
        "errors": ctx.errors,
        "identity": identity,
        "require_active_sensing": bool(require_active_sensing),
        "validated_claims": (
            list(
                _claims_for_mode(
                    mode,
                    require_active_sensing=require_active_sensing,
                )
            )
            if valid
            else []
        ),
        "not_validated_claims": (
            list(
                (
                    *_DYNAMIC_FINAL_CLAIMS,
                    *(
                        _ACTIVE_SENSING_CLAIMS
                        if require_active_sensing
                        else ()
                    ),
                )
            )
            if mode == "dynamic_f1" and not valid
            else (
                list(_DYNAMIC_REPLAN_CLAIMS)
                if mode == "dynamic_replan_f1" and not valid
                else []
            )
        ),
    }


def validate_summary_path(
    path: str | Path,
    mode: ValidationMode,
    *,
    require_active_sensing: bool = False,
) -> dict[str, Any]:
    """严格读取一个 episode summary 并执行对应模式验收。"""

    summary_path, summary = load_summary(path)
    return validate_pct_scan_live_summary(
        summary,
        mode,
        source=str(summary_path),
        require_active_sensing=require_active_sensing,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线 fail-closed 校验 phase184 PCT→SCAN live summary。",
        epilog=(
            "示例：python3 scripts/navigation/validate_pct_scan_live_summary.py "
            "outputs/pct_scan/<run>/episode_000000 --mode dynamic_replan_f1"
        ),
    )
    parser.add_argument(
        "summary",
        help="summary.json 文件，或包含 summary.json 的 episode 目录。",
    )
    parser.add_argument(
        "--mode",
        choices=VALIDATION_MODES,
        required=True,
        help=(
            "static_stair、flat_policy、crossfloor_carry、dynamic_f1 或 "
            "dynamic_replan_f1。"
        ),
    )
    parser.add_argument(
        "--require-active-sensing",
        action="store_true",
        help=(
            "仅 dynamic_f1：强制校验一次完整 yaw-only 主动观测、三帧真实融合、"
            "controller/policy 命令包络和同 Path 恢复。"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整机器可读 JSON 报告。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：合同失败返回 1，输入文件错误返回 2。"""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = validate_summary_path(
            arguments.summary,
            arguments.mode,
            require_active_sensing=arguments.require_active_sensing,
        )
    except SummaryInputError as exc:
        if arguments.json:
            print(
                json.dumps(
                    {
                        "schema": "pct_scan_live_summary_validation_v1",
                        "mode": arguments.mode,
                        "source": str(arguments.summary),
                        "valid": False,
                        "input_error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"INPUT ERROR [{arguments.mode}]: {exc}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif report["valid"]:
        identity = report["identity"]
        print(
            f"PASS [{arguments.mode}] task={identity['task_id']} "
            f"goal={identity['pct_goal_stamp_ns']} "
            f"path={identity['live_reference_path_stamp_ns']}"
        )
        if report["not_validated_claims"]:
            print(
                "范围外（未认证）："
                + ", ".join(report["not_validated_claims"])
            )
    else:
        print(
            f"FAIL [{arguments.mode}] {report['error_count']} 个合同错误：",
            file=sys.stderr,
        )
        for issue in report["errors"]:
            print(
                f"- {issue['code']} {issue['path']}: {issue['message']}",
                file=sys.stderr,
            )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
