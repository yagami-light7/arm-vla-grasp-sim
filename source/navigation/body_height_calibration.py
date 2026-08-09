"""基于实时稳定状态标定机器人 base 到碰撞支撑面的高度。

本模块不依赖 ROS 2、Isaac Sim 或 pipeline 状态机。调用方只需把每个控制
周期已经观测到的位姿、速度、锁状态和 policy 实写结果整理成
``BodyHeightCalibrationSample``。校准器不会接收 task goal z，也不会把离线
任务位姿当作物理高度证据。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from source.scene.placement_support import load_binary_triangle_ply

from .pct_adapter import pct_to_sim_xyz, sim_to_pct_xyz


@dataclass(frozen=True, slots=True)
class BodyHeightCalibrationConfig:
    """定义一次 live body-height 稳态校准的严格门限。"""

    collision_ply: str | Path
    configured_body_height_hint_m: float
    arm_stow_joint_positions: tuple[float, ...]
    minimum_consecutive_samples: int = 50
    minimum_stable_duration_s: float = 1.0
    maximum_ground_hint_error_m: float = 0.12
    maximum_linear_speed_mps: float = 0.02
    maximum_angular_speed_rps: float = 0.05
    maximum_tilt_rad: float = 0.08
    maximum_arm_stow_error_rad: float = 0.05
    maximum_body_height_mad_m: float = 0.005
    maximum_body_height_p95_p05_m: float = 0.015
    quick_minimum_consecutive_samples: int | None = None
    quick_minimum_stable_duration_s: float | None = None
    quick_maximum_body_height_mad_m: float | None = None
    quick_maximum_body_height_p95_p05_m: float | None = None
    quick_maximum_configured_height_error_m: float | None = None
    maximum_quaternion_norm_error: float = 1.0e-3
    coord_mode: str = "sim_to_pct_180deg"
    pct_offset_x: float = 0.0
    pct_offset_y: float = 0.0
    pct_offset_z: float = 0.0
    pct_scale_x: float = 1.0
    pct_scale_y: float = 1.0
    pct_scale_z: float = 1.0
    pct_rotation_x_rad: float = 0.0
    pct_rotation_y_rad: float = 0.0
    pct_rotation_z_rad: float = 0.0

    def __post_init__(self) -> None:
        path = Path(self.collision_ply).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"collision PLY 不存在: {path}")
        object.__setattr__(self, "collision_ply", path)

        if (
            isinstance(self.minimum_consecutive_samples, bool)
            or not isinstance(self.minimum_consecutive_samples, int)
            or self.minimum_consecutive_samples < 2
        ):
            raise ValueError("minimum_consecutive_samples 必须是不小于 2 的整数。")

        positive_fields = (
            "configured_body_height_hint_m",
            "maximum_ground_hint_error_m",
            "maximum_linear_speed_mps",
            "maximum_angular_speed_rps",
            "maximum_tilt_rad",
            "maximum_arm_stow_error_rad",
            "maximum_body_height_mad_m",
            "maximum_body_height_p95_p05_m",
            "maximum_quaternion_norm_error",
        )
        nonnegative_fields = ("minimum_stable_duration_s",)
        for field_name in (*positive_fields, *nonnegative_fields):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} 必须是实数。")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} 必须是有限值。")
            object.__setattr__(self, field_name, normalized)
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} 必须大于零。")
        if self.minimum_stable_duration_s < 0.0:
            raise ValueError("minimum_stable_duration_s 不能为负数。")

        arm_target = _finite_tuple(
            self.arm_stow_joint_positions,
            field_name="arm_stow_joint_positions",
            minimum_length=1,
        )
        object.__setattr__(self, "arm_stow_joint_positions", arm_target)

        quick_fields = (
            "quick_minimum_consecutive_samples",
            "quick_minimum_stable_duration_s",
            "quick_maximum_body_height_mad_m",
            "quick_maximum_body_height_p95_p05_m",
            "quick_maximum_configured_height_error_m",
        )
        quick_values = tuple(getattr(self, name) for name in quick_fields)
        if any(value is not None for value in quick_values):
            if any(value is None for value in quick_values):
                raise ValueError("快速校准参数必须同时提供或同时省略。")
            quick_samples = self.quick_minimum_consecutive_samples
            if (
                isinstance(quick_samples, bool)
                or not isinstance(quick_samples, int)
                or quick_samples < 2
                or quick_samples > self.minimum_consecutive_samples
            ):
                raise ValueError(
                    "quick_minimum_consecutive_samples 必须位于 "
                    "[2, minimum_consecutive_samples]。"
                )
            quick_numeric_fields = quick_fields[1:]
            for field_name in quick_numeric_fields:
                raw_value = getattr(self, field_name)
                if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                    raise TypeError(f"{field_name} 必须是实数。")
                normalized = float(raw_value)
                if not math.isfinite(normalized) or normalized <= 0.0:
                    raise ValueError(f"{field_name} 必须是有限正数。")
                object.__setattr__(self, field_name, normalized)
            if (
                self.quick_minimum_stable_duration_s
                > self.minimum_stable_duration_s
            ):
                raise ValueError(
                    "quick_minimum_stable_duration_s 不能超过完整窗口。"
                )
            if (
                self.quick_maximum_body_height_mad_m
                > self.maximum_body_height_mad_m
                or self.quick_maximum_body_height_p95_p05_m
                > self.maximum_body_height_p95_p05_m
            ):
                raise ValueError("快速校准离散度门不能宽于完整窗口。")
            if (
                self.quick_maximum_body_height_p95_p05_m
                < self.quick_maximum_body_height_mad_m
            ):
                raise ValueError("快速校准 spread 不能小于 MAD。")

        # 复用生产 PCT adapter 的统一转换及其配置校验，禁止本模块手写坐标负号。
        sim_to_pct_xyz(
            (0.0, 0.0, 0.0),
            coord_mode=self.coord_mode,
            pct_offset_x=self.pct_offset_x,
            pct_offset_y=self.pct_offset_y,
            pct_offset_z=self.pct_offset_z,
            pct_scale_x=self.pct_scale_x,
            pct_scale_y=self.pct_scale_y,
            pct_scale_z=self.pct_scale_z,
            pct_rotation_x_rad=self.pct_rotation_x_rad,
            pct_rotation_y_rad=self.pct_rotation_y_rad,
            pct_rotation_z_rad=self.pct_rotation_z_rad,
        )


@dataclass(frozen=True, slots=True)
class BodyHeightCalibrationSample:
    """一个实时控制周期的传输无关校准观测。

    四个锁字段必须表示当前实际状态，而不是历史上是否曾经启用过。四元数采用
    pipeline ``SimulationState`` 一致的 ``(w, x, y, z)`` 顺序。
    """

    step_index: int
    timestamp_s: float
    policy_write_sequence: int
    written_command: Sequence[float]
    root_position_sim_xyz: Sequence[float]
    root_orientation_wxyz: Sequence[float]
    root_linear_velocity_xyz: Sequence[float]
    root_angular_velocity_xyz: Sequence[float]
    arm_joint_positions: Sequence[float]
    base_lock_active: bool
    support_joint_lock_active: bool
    full_body_joint_lock_active: bool
    object_follow_active: bool


@dataclass(frozen=True, slots=True)
class BodyHeightCalibrationResult:
    """达到连续样本数、持续时间和离散度门后的校准证据。"""

    sample_count: int
    sample_duration_s: float
    body_height_median_m: float
    body_height_mad_m: float
    body_height_p05_m: float
    body_height_p95_m: float
    body_height_p95_p05_m: float
    ground_surface_z_median_m: float
    ground_face_index: int
    ground_face_indices: tuple[int, ...]
    collision_ply: Path
    collision_ply_sha256: str
    first_step_index: int
    last_step_index: int
    first_timestamp_s: float
    last_timestamp_s: float
    first_policy_write_sequence: int
    last_policy_write_sequence: int
    configured_body_height_hint_m: float
    configured_body_height_error_m: float
    certification_mode: str
    certification_minimum_samples: int
    certification_minimum_duration_s: float
    certification_maximum_body_height_mad_m: float
    certification_maximum_body_height_p95_p05_m: float
    quick_fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接写入运行 summary 的审计字典。"""

        return {
            "sample_count": self.sample_count,
            "sample_duration_s": self.sample_duration_s,
            "body_height_median_m": self.body_height_median_m,
            "body_height_mad_m": self.body_height_mad_m,
            "body_height_p05_m": self.body_height_p05_m,
            "body_height_p95_m": self.body_height_p95_m,
            "body_height_p95_p05_m": self.body_height_p95_p05_m,
            "ground_surface_z_median_m": self.ground_surface_z_median_m,
            "ground_face_index": self.ground_face_index,
            "ground_face_indices": list(self.ground_face_indices),
            "collision_ply": str(self.collision_ply),
            "collision_ply_sha256": self.collision_ply_sha256,
            "first_step_index": self.first_step_index,
            "last_step_index": self.last_step_index,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "first_policy_write_sequence": self.first_policy_write_sequence,
            "last_policy_write_sequence": self.last_policy_write_sequence,
            "configured_body_height_hint_m": self.configured_body_height_hint_m,
            "configured_body_height_error_m": (
                self.configured_body_height_error_m
            ),
            "certification_mode": self.certification_mode,
            "certification_minimum_samples": (
                self.certification_minimum_samples
            ),
            "certification_minimum_duration_s": (
                self.certification_minimum_duration_s
            ),
            "certification_maximum_body_height_mad_m": (
                self.certification_maximum_body_height_mad_m
            ),
            "certification_maximum_body_height_p95_p05_m": (
                self.certification_maximum_body_height_p95_p05_m
            ),
            "quick_fallback_reason": self.quick_fallback_reason,
            "height_semantics": "live_root_z_minus_collision_support_z",
            "raw_task_z_used": False,
        }


@dataclass(frozen=True, slots=True)
class GroundSurfaceProjection:
    """一次只读碰撞地面投影及其楼层提示误差。"""

    query_sim_xyz: tuple[float, float, float]
    query_pct_xyz: tuple[float, float, float]
    ground_surface_sim_xyz: tuple[float, float, float]
    ground_surface_pct_xyz: tuple[float, float, float]
    ground_z_m: float
    ground_face_index: int
    hint_error_m: float
    projected_base_sim_xyz: tuple[float, float, float]
    configured_body_height_hint_m: float
    collision_ply: Path
    collision_ply_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """转换为可审计的楼层支撑面投影报告。"""

        return {
            "query_sim_xyz": list(self.query_sim_xyz),
            "query_pct_xyz": list(self.query_pct_xyz),
            "ground_surface_sim_xyz": list(self.ground_surface_sim_xyz),
            "ground_surface_pct_xyz": list(self.ground_surface_pct_xyz),
            "ground_z_m": self.ground_z_m,
            "ground_face_index": self.ground_face_index,
            "hint_error_m": self.hint_error_m,
            "projected_base_sim_xyz": list(self.projected_base_sim_xyz),
            "configured_body_height_hint_m": self.configured_body_height_hint_m,
            "collision_ply": str(self.collision_ply),
            "collision_ply_sha256": self.collision_ply_sha256,
            "z_hint_semantics": "floor_disambiguation_only",
            "raw_task_z_used_as_height_evidence": False,
        }


class GroundSurfaceProjectionError(ValueError):
    """表示楼层提示无法唯一、安全地投影到碰撞支撑面。"""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class BodyHeightCalibrationUpdate:
    """描述当前样本是否进入窗口以及窗口是否已经完成。"""

    accepted: bool
    reason: str
    window_reset: bool
    consecutive_sample_count: int
    stable_duration_s: float
    selected_ground_z_m: float | None = None
    selected_ground_face_index: int | None = None
    body_height_sample_m: float | None = None
    result: BodyHeightCalibrationResult | None = None
    quick_candidate_evaluated: bool = False
    quick_rejection_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _AcceptedSample:
    """校准窗口内部保存的已认证样本。"""

    step_index: int
    timestamp_s: float
    policy_write_sequence: int
    body_height_m: float
    ground_z_m: float
    ground_face_index: int


@dataclass(frozen=True, slots=True)
class _CollisionTriangleCache:
    """一个 collision PLY 的只读三角面、XY 包围盒和资产指纹。"""

    collision_ply: Path
    triangles: np.ndarray
    lower_xy: np.ndarray
    upper_xy: np.ndarray
    sha256: str

    def nearest_vertical_surface(
        self,
        *,
        x: float,
        y: float,
        z_hint: float,
    ) -> tuple[float, int] | None:
        """返回固定 XY 处最接近 z 提示的三角面交点。"""

        candidate_indices = np.flatnonzero(
            (self.lower_xy[:, 0] <= x)
            & (x <= self.upper_xy[:, 0])
            & (self.lower_xy[:, 1] <= y)
            & (y <= self.upper_xy[:, 1])
        )
        if candidate_indices.size == 0:
            return None

        triangles = self.triangles[candidate_indices]
        first = triangles[:, 0, :2]
        second = triangles[:, 1, :2]
        third = triangles[:, 2, :2]
        denominator = (
            (second[:, 1] - third[:, 1]) * (first[:, 0] - third[:, 0])
            + (third[:, 0] - second[:, 0]) * (first[:, 1] - third[:, 1])
        )
        valid = np.abs(denominator) > 1.0e-12
        safe_denominator = np.where(valid, denominator, 1.0)
        weight_first = (
            (second[:, 1] - third[:, 1]) * (x - third[:, 0])
            + (third[:, 0] - second[:, 0]) * (y - third[:, 1])
        ) / safe_denominator
        weight_second = (
            (third[:, 1] - first[:, 1]) * (x - third[:, 0])
            + (first[:, 0] - third[:, 0]) * (y - third[:, 1])
        ) / safe_denominator
        weight_third = 1.0 - weight_first - weight_second
        valid &= (
            np.minimum(np.minimum(weight_first, weight_second), weight_third)
            >= -1.0e-7
        )
        valid_offsets = np.flatnonzero(valid)
        if valid_offsets.size == 0:
            return None

        valid_triangles = triangles[valid_offsets]
        valid_first = weight_first[valid_offsets]
        valid_second = weight_second[valid_offsets]
        valid_third = weight_third[valid_offsets]
        intersections = (
            valid_first * valid_triangles[:, 0, 2]
            + valid_second * valid_triangles[:, 1, 2]
            + valid_third * valid_triangles[:, 2, 2]
        )
        face_indices = candidate_indices[valid_offsets]
        best_offset = min(
            range(len(intersections)),
            key=lambda index: (
                abs(float(intersections[index]) - z_hint),
                int(face_indices[index]),
            ),
        )
        return float(intersections[best_offset]), int(face_indices[best_offset])


class LiveBodyHeightCalibrator:
    """收集连续 live 稳态样本并生成 fail-closed body-height 证据。"""

    def __init__(self, config: BodyHeightCalibrationConfig):
        if not isinstance(config, BodyHeightCalibrationConfig):
            raise TypeError("config 必须是 BodyHeightCalibrationConfig。")
        self.config = config
        path = Path(config.collision_ply)
        stat = path.stat()
        self._mesh = _load_collision_triangle_cache(
            str(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        self._samples: list[_AcceptedSample] = []
        self._result: BodyHeightCalibrationResult | None = None
        self._quick_fallback_reason: str | None = None

    @property
    def result(self) -> BodyHeightCalibrationResult | None:
        """返回已经锁存的完整结果；窗口未完成时返回 ``None``。"""

        return self._result

    @property
    def consecutive_sample_count(self) -> int:
        """返回当前连续认证窗口的样本数。"""

        return len(self._samples)

    def reset(self) -> None:
        """显式开始一轮新校准，并丢弃已锁存结果。"""

        self._samples.clear()
        self._result = None
        self._quick_fallback_reason = None

    def observe(
        self,
        sample: BodyHeightCalibrationSample,
    ) -> BodyHeightCalibrationUpdate:
        """消费一个控制周期；任何非法或不稳定观测都会清空当前窗口。"""

        if not isinstance(sample, BodyHeightCalibrationSample):
            return self._reject("sample_type_invalid")

        parsed, reason = self._validate_sample(sample)
        if parsed is None:
            return self._reject(reason)

        if self._samples:
            previous = self._samples[-1]
            if parsed.step_index <= previous.step_index:
                return self._reject("step_index_not_strictly_increasing")
            if parsed.timestamp_s <= previous.timestamp_s:
                return self._reject("timestamp_not_strictly_increasing")
            if parsed.policy_write_sequence <= previous.policy_write_sequence:
                return self._reject("write_sequence_not_strictly_increasing")

        self._samples.append(parsed)
        duration = self._stable_duration()
        quick_candidate_evaluated = False
        if (
            self.config.quick_minimum_consecutive_samples is not None
            and self._quick_fallback_reason is None
        ):
            quick_candidate_evaluated = bool(
                len(self._samples)
                >= self.config.quick_minimum_consecutive_samples
                and duration
                >= float(self.config.quick_minimum_stable_duration_s)
            )
            if quick_candidate_evaluated:
                quick_result = self._build_result(
                    certification_mode="quick_window",
                    certification_minimum_samples=(
                        self.config.quick_minimum_consecutive_samples
                    ),
                    certification_minimum_duration_s=float(
                        self.config.quick_minimum_stable_duration_s
                    ),
                    certification_maximum_body_height_mad_m=float(
                        self.config.quick_maximum_body_height_mad_m
                    ),
                    certification_maximum_body_height_p95_p05_m=float(
                        self.config.quick_maximum_body_height_p95_p05_m
                    ),
                    quick_fallback_reason=None,
                )
                quick_rejection = self._quick_rejection_reason(quick_result)
                if quick_rejection is None:
                    self._result = quick_result
                    return self._update(
                        accepted=True,
                        reason="quick_calibration_complete",
                        selected=parsed,
                        result=quick_result,
                        quick_candidate_evaluated=True,
                    )
                self._quick_fallback_reason = quick_rejection

        enough_samples = (
            len(self._samples) >= self.config.minimum_consecutive_samples
        )
        enough_duration = duration >= self.config.minimum_stable_duration_s
        if not (enough_samples and enough_duration):
            return self._update(
                accepted=True,
                reason=(
                    "collecting_full_window"
                    if self._quick_fallback_reason is not None
                    else "collecting"
                ),
                selected=parsed,
                quick_candidate_evaluated=quick_candidate_evaluated,
                quick_rejection_reason=self._quick_fallback_reason,
            )

        result = self._build_result(
            certification_mode="full_window",
            certification_minimum_samples=(
                self.config.minimum_consecutive_samples
            ),
            certification_minimum_duration_s=(
                self.config.minimum_stable_duration_s
            ),
            certification_maximum_body_height_mad_m=(
                self.config.maximum_body_height_mad_m
            ),
            certification_maximum_body_height_p95_p05_m=(
                self.config.maximum_body_height_p95_p05_m
            ),
            quick_fallback_reason=self._quick_fallback_reason,
        )
        if (
            result.body_height_mad_m
            > self.config.maximum_body_height_mad_m
            or result.body_height_p95_p05_m
            > self.config.maximum_body_height_p95_p05_m
        ):
            return self._reject("body_height_dispersion_exceeded")
        self._result = result
        return self._update(
            accepted=True,
            reason=(
                "full_calibration_complete"
                if self.config.quick_minimum_consecutive_samples is not None
                else "calibration_complete"
            ),
            selected=parsed,
            result=result,
            quick_candidate_evaluated=quick_candidate_evaluated,
            quick_rejection_reason=self._quick_fallback_reason,
        )

    def project_ground_surface(
        self,
        sim_xyz_hint: Sequence[float],
    ) -> GroundSurfaceProjection:
        """用 Sim XY 和地面 z 提示只读投影 collision PLY 支撑面。

        ``sim_xyz_hint[2]`` 只负责在多个楼层间消歧。调用方可以传入历史
        goal base z 减去已配置 body height 得到该提示，但该值绝不会进入
        body-height 统计，也不会成为修改配置高度的证据。
        """

        query_sim = _try_finite_tuple(sim_xyz_hint, expected_length=3)
        if query_sim is None:
            raise GroundSurfaceProjectionError(
                "query_invalid",
                "sim_xyz_hint 必须包含三个有限实数。",
            )
        query_pct = sim_to_pct_xyz(
            query_sim,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_offset_z=self.config.pct_offset_z,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
            pct_scale_z=self.config.pct_scale_z,
            pct_rotation_x_rad=self.config.pct_rotation_x_rad,
            pct_rotation_y_rad=self.config.pct_rotation_y_rad,
            pct_rotation_z_rad=self.config.pct_rotation_z_rad,
        )
        surface = self._mesh.nearest_vertical_surface(
            x=query_pct[0],
            y=query_pct[1],
            z_hint=query_pct[2],
        )
        if surface is None:
            raise GroundSurfaceProjectionError(
                "ground_surface_not_found",
                "collision PLY 在目标 XY 没有支撑面。",
            )
        ground_z, face_index = surface
        hint_error = abs(float(ground_z - query_pct[2]))
        if hint_error > self.config.maximum_ground_hint_error_m:
            raise GroundSurfaceProjectionError(
                "ground_surface_hint_error_exceeded",
                "collision PLY 支撑面与楼层 z 提示偏差超过安全门。"
            )
        ground_pct = (query_pct[0], query_pct[1], float(ground_z))
        ground_sim = pct_to_sim_xyz(
            ground_pct,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_offset_z=self.config.pct_offset_z,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
            pct_scale_z=self.config.pct_scale_z,
            pct_rotation_x_rad=self.config.pct_rotation_x_rad,
            pct_rotation_y_rad=self.config.pct_rotation_y_rad,
            pct_rotation_z_rad=self.config.pct_rotation_z_rad,
        )
        projected_base_sim = (
            ground_sim[0],
            ground_sim[1],
            ground_sim[2] + self.config.configured_body_height_hint_m,
        )
        return GroundSurfaceProjection(
            query_sim_xyz=query_sim,
            query_pct_xyz=query_pct,
            ground_surface_sim_xyz=ground_sim,
            ground_surface_pct_xyz=ground_pct,
            ground_z_m=float(ground_z),
            ground_face_index=face_index,
            hint_error_m=hint_error,
            projected_base_sim_xyz=projected_base_sim,
            configured_body_height_hint_m=(
                self.config.configured_body_height_hint_m
            ),
            collision_ply=self._mesh.collision_ply,
            collision_ply_sha256=self._mesh.sha256,
        )

    def _validate_sample(
        self,
        sample: BodyHeightCalibrationSample,
    ) -> tuple[_AcceptedSample | None, str]:
        step_index = _nonnegative_int(sample.step_index)
        if step_index is None:
            return None, "step_index_invalid"
        timestamp = _nonnegative_finite(sample.timestamp_s)
        if timestamp is None:
            return None, "timestamp_invalid"
        write_sequence = _nonnegative_int(sample.policy_write_sequence)
        if write_sequence is None:
            return None, "write_sequence_invalid"

        written_command = _try_finite_tuple(
            sample.written_command,
            expected_length=3,
        )
        if written_command is None:
            return None, "written_command_invalid"
        if any(value != 0.0 for value in written_command):
            return None, "written_command_not_exact_zero"

        lock_fields = (
            ("base_lock_active", sample.base_lock_active),
            ("support_joint_lock_active", sample.support_joint_lock_active),
            ("full_body_joint_lock_active", sample.full_body_joint_lock_active),
            ("object_follow_active", sample.object_follow_active),
        )
        for field_name, active in lock_fields:
            if not isinstance(active, bool):
                return None, f"{field_name}_invalid"
            if active:
                return None, f"{field_name}_must_be_false"

        linear_velocity = _try_finite_tuple(
            sample.root_linear_velocity_xyz,
            expected_length=3,
        )
        if linear_velocity is None:
            return None, "root_linear_velocity_invalid"
        if math.sqrt(sum(value * value for value in linear_velocity)) > (
            self.config.maximum_linear_speed_mps
        ):
            return None, "linear_speed_exceeded"

        angular_velocity = _try_finite_tuple(
            sample.root_angular_velocity_xyz,
            expected_length=3,
        )
        if angular_velocity is None:
            return None, "root_angular_velocity_invalid"
        if math.sqrt(sum(value * value for value in angular_velocity)) > (
            self.config.maximum_angular_speed_rps
        ):
            return None, "angular_speed_exceeded"

        orientation = _try_finite_tuple(
            sample.root_orientation_wxyz,
            expected_length=4,
        )
        if orientation is None:
            return None, "root_orientation_invalid"
        tilt = _tilt_from_wxyz(
            orientation,
            maximum_norm_error=self.config.maximum_quaternion_norm_error,
        )
        if tilt is None:
            return None, "root_orientation_invalid"
        if tilt > self.config.maximum_tilt_rad:
            return None, "tilt_exceeded"

        arm_positions = _try_finite_tuple(
            sample.arm_joint_positions,
            expected_length=len(self.config.arm_stow_joint_positions),
        )
        if arm_positions is None:
            return None, "arm_joint_positions_invalid"
        arm_error = max(
            abs(actual - target)
            for actual, target in zip(
                arm_positions,
                self.config.arm_stow_joint_positions,
                strict=True,
            )
        )
        if arm_error > self.config.maximum_arm_stow_error_rad:
            return None, "arm_not_stowed"

        root_position = _try_finite_tuple(
            sample.root_position_sim_xyz,
            expected_length=3,
        )
        if root_position is None:
            return None, "root_position_invalid"
        ground_hint_sim = (
            root_position[0],
            root_position[1],
            root_position[2] - self.config.configured_body_height_hint_m,
        )
        try:
            projection = self.project_ground_surface(ground_hint_sim)
        except GroundSurfaceProjectionError as error:
            return None, error.reason
        body_height = float(root_position[2] - projection.ground_z_m)
        if not math.isfinite(body_height) or body_height <= 0.0:
            return None, "body_height_not_positive"
        return (
            _AcceptedSample(
                step_index=step_index,
                timestamp_s=timestamp,
                policy_write_sequence=write_sequence,
                body_height_m=body_height,
                ground_z_m=projection.ground_z_m,
                ground_face_index=projection.ground_face_index,
            ),
            "accepted",
        )

    def _reject(self, reason: str) -> BodyHeightCalibrationUpdate:
        self._samples.clear()
        self._result = None
        self._quick_fallback_reason = None
        return BodyHeightCalibrationUpdate(
            accepted=False,
            reason=reason,
            window_reset=True,
            consecutive_sample_count=0,
            stable_duration_s=0.0,
        )

    def _update(
        self,
        *,
        accepted: bool,
        reason: str,
        selected: _AcceptedSample | None = None,
        result: BodyHeightCalibrationResult | None = None,
        quick_candidate_evaluated: bool = False,
        quick_rejection_reason: str | None = None,
    ) -> BodyHeightCalibrationUpdate:
        return BodyHeightCalibrationUpdate(
            accepted=accepted,
            reason=reason,
            window_reset=False,
            consecutive_sample_count=len(self._samples),
            stable_duration_s=self._stable_duration(),
            selected_ground_z_m=(
                None if selected is None else selected.ground_z_m
            ),
            selected_ground_face_index=(
                None if selected is None else selected.ground_face_index
            ),
            body_height_sample_m=(
                None if selected is None else selected.body_height_m
            ),
            result=result,
            quick_candidate_evaluated=quick_candidate_evaluated,
            quick_rejection_reason=quick_rejection_reason,
        )

    def _stable_duration(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return float(
            self._samples[-1].timestamp_s - self._samples[0].timestamp_s
        )

    def _build_result(
        self,
        *,
        certification_mode: str,
        certification_minimum_samples: int,
        certification_minimum_duration_s: float,
        certification_maximum_body_height_mad_m: float,
        certification_maximum_body_height_p95_p05_m: float,
        quick_fallback_reason: str | None,
    ) -> BodyHeightCalibrationResult:
        first = self._samples[0]
        last = self._samples[-1]
        heights = np.asarray(
            [sample.body_height_m for sample in self._samples],
            dtype=np.float64,
        )
        ground_z = np.asarray(
            [sample.ground_z_m for sample in self._samples],
            dtype=np.float64,
        )
        median = float(np.median(heights))
        mad = float(np.median(np.abs(heights - median)))
        p05, p95 = (
            float(value) for value in np.quantile(heights, (0.05, 0.95))
        )
        face_counts = Counter(sample.ground_face_index for sample in self._samples)
        primary_face = min(
            face_counts,
            key=lambda face_index: (-face_counts[face_index], face_index),
        )
        return BodyHeightCalibrationResult(
            sample_count=len(self._samples),
            sample_duration_s=self._stable_duration(),
            body_height_median_m=median,
            body_height_mad_m=mad,
            body_height_p05_m=p05,
            body_height_p95_m=p95,
            body_height_p95_p05_m=float(p95 - p05),
            ground_surface_z_median_m=float(np.median(ground_z)),
            ground_face_index=primary_face,
            ground_face_indices=tuple(sorted(face_counts)),
            collision_ply=self._mesh.collision_ply,
            collision_ply_sha256=self._mesh.sha256,
            first_step_index=first.step_index,
            last_step_index=last.step_index,
            first_timestamp_s=first.timestamp_s,
            last_timestamp_s=last.timestamp_s,
            first_policy_write_sequence=first.policy_write_sequence,
            last_policy_write_sequence=last.policy_write_sequence,
            configured_body_height_hint_m=(
                self.config.configured_body_height_hint_m
            ),
            configured_body_height_error_m=abs(
                median - self.config.configured_body_height_hint_m
            ),
            certification_mode=certification_mode,
            certification_minimum_samples=certification_minimum_samples,
            certification_minimum_duration_s=(
                certification_minimum_duration_s
            ),
            certification_maximum_body_height_mad_m=(
                certification_maximum_body_height_mad_m
            ),
            certification_maximum_body_height_p95_p05_m=(
                certification_maximum_body_height_p95_p05_m
            ),
            quick_fallback_reason=quick_fallback_reason,
        )

    def _quick_rejection_reason(
        self,
        result: BodyHeightCalibrationResult,
    ) -> str | None:
        """返回快速窗口拒绝原因；拒绝只回退，不清空完整窗口。"""

        if (
            result.body_height_mad_m
            > float(self.config.quick_maximum_body_height_mad_m)
            or result.body_height_p95_p05_m
            > float(self.config.quick_maximum_body_height_p95_p05_m)
        ):
            return "quick_body_height_dispersion_exceeded"
        if result.configured_body_height_error_m > float(
            self.config.quick_maximum_configured_height_error_m
        ):
            return "quick_configured_height_error_exceeded"
        return None


@lru_cache(maxsize=4)
def _load_collision_triangle_cache(
    resolved_path: str,
    file_size: int,
    file_mtime_ns: int,
) -> _CollisionTriangleCache:
    """按规范路径和文件指纹缓存三角面，资产变化后自动生成新条目。"""

    del file_size, file_mtime_ns
    path = Path(resolved_path)
    vertices, face_indices = load_binary_triangle_ply(path)
    triangles = np.asarray(vertices[face_indices], dtype=np.float64)
    lower_xy = np.min(triangles[:, :, :2], axis=1)
    upper_xy = np.max(triangles[:, :, :2], axis=1)
    for array in (triangles, lower_xy, upper_xy):
        array.setflags(write=False)
    return _CollisionTriangleCache(
        collision_ply=path,
        triangles=triangles,
        lower_xy=lower_xy,
        upper_xy=upper_xy,
        sha256=_sha256(path),
    )


def _sha256(path: Path) -> str:
    """流式计算 collision PLY 的 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _nonnegative_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        return None
    return normalized


def _finite_tuple(
    values: Sequence[float],
    *,
    field_name: str,
    minimum_length: int,
) -> tuple[float, ...]:
    parsed = _try_finite_tuple(values, minimum_length=minimum_length)
    if parsed is None:
        raise ValueError(f"{field_name} 必须包含至少 {minimum_length} 个有限实数。")
    return parsed


def _try_finite_tuple(
    values: Any,
    *,
    expected_length: int | None = None,
    minimum_length: int | None = None,
) -> tuple[float, ...] | None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return None
    if expected_length is not None and len(values) != expected_length:
        return None
    if minimum_length is not None and len(values) < minimum_length:
        return None
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        normalized = float(value)
        if not math.isfinite(normalized):
            return None
        output.append(normalized)
    return tuple(output)


def _tilt_from_wxyz(
    orientation: tuple[float, ...],
    *,
    maximum_norm_error: float,
) -> float | None:
    """从 wxyz 四元数计算 max(abs(roll), abs(pitch))。"""

    norm = math.sqrt(sum(value * value for value in orientation))
    if norm <= 1.0e-12 or abs(norm - 1.0) > maximum_norm_error:
        return None
    w, x, y, z = (value / norm for value in orientation)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return max(abs(roll), abs(pitch))


__all__ = [
    "BodyHeightCalibrationConfig",
    "BodyHeightCalibrationResult",
    "BodyHeightCalibrationSample",
    "BodyHeightCalibrationUpdate",
    "GroundSurfaceProjection",
    "GroundSurfaceProjectionError",
    "LiveBodyHeightCalibrator",
]
