"""Task-level validation for dynamically settled manipulation objects."""

from __future__ import annotations

import math
from typing import Any, Sequence


SUPPORTED_UPRIGHT_MODE = "supported_upright_v1"


def _finite_non_negative(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有限非负数") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} 必须是有限非负数")
    return result


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是正整数")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是正整数") from exc
    if result <= 0 or not math.isfinite(numeric) or numeric != float(result):
        raise ValueError(f"{field_name} 必须是正整数")
    return result


def resolve_object_initialization_policy(raw_task: dict[str, Any]) -> dict[str, Any]:
    """Resolve an opt-in supported-upright initialization policy."""

    raw = raw_task.get("object_initialization") if isinstance(raw_task, dict) else None
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return {
            "enabled": False,
            "mode": None,
            "source": "task.object_initialization_disabled",
        }

    mode = str(raw.get("mode") or "").strip()
    if mode != SUPPORTED_UPRIGHT_MODE:
        raise ValueError(
            "object_initialization.mode 当前只支持 "
            f"{SUPPORTED_UPRIGHT_MODE!r}"
        )
    booleans: dict[str, bool] = {}
    for name, default in (
        ("restore_pose_after_runtime_reset", True),
        ("stabilize_xy_and_orientation_during_settle", True),
        ("required_for_episode", True),
    ):
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"object_initialization.{name} 必须是 bool")
        booleans[name] = value

    return {
        "enabled": True,
        "mode": mode,
        **booleans,
        "max_horizontal_displacement_m": _finite_non_negative(
            raw.get("max_horizontal_displacement_m", 0.02),
            field_name="object_initialization.max_horizontal_displacement_m",
        ),
        "max_vertical_displacement_m": _finite_non_negative(
            raw.get("max_vertical_displacement_m", 0.02),
            field_name="object_initialization.max_vertical_displacement_m",
        ),
        "max_orientation_error_rad": _finite_non_negative(
            raw.get("max_orientation_error_rad", 0.10),
            field_name="object_initialization.max_orientation_error_rad",
        ),
        "dynamic_settle_steps_before_sleep": _positive_int(
            raw.get("dynamic_settle_steps_before_sleep", 8),
            field_name="object_initialization.dynamic_settle_steps_before_sleep",
        ),
        "source": "task.object_initialization",
    }


def _finite_tuple(
    values: Sequence[float],
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(values) < length:
        raise ValueError(f"{field_name} 至少需要 {length} 个数值")
    result = tuple(float(values[index]) for index in range(length))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} 必须包含有限数值")
    return result


def quaternion_angle_error_rad(
    actual_wxyz: Sequence[float],
    expected_wxyz: Sequence[float],
) -> float:
    """Return the shortest angular distance between scalar-first quaternions."""

    actual = _finite_tuple(actual_wxyz, length=4, field_name="actual_quaternion")
    expected = _finite_tuple(
        expected_wxyz,
        length=4,
        field_name="expected_quaternion",
    )
    actual_norm = math.sqrt(sum(value * value for value in actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    if actual_norm <= 1.0e-12 or expected_norm <= 1.0e-12:
        raise ValueError("object initialization quaternion norm must be positive")
    dot = abs(
        sum(
            actual[index] * expected[index]
            for index in range(4)
        )
        / (actual_norm * expected_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def evaluate_object_initialization_pose(
    *,
    policy: dict[str, Any],
    requested_position_xyz: Sequence[float],
    requested_quaternion_wxyz: Sequence[float],
    actual_pose_xyz_wxyz: Sequence[float],
) -> dict[str, Any]:
    """Evaluate whether a settled object preserved its requested support pose."""

    requested_position = _finite_tuple(
        requested_position_xyz,
        length=3,
        field_name="requested_position_xyz",
    )
    requested_quaternion = _finite_tuple(
        requested_quaternion_wxyz,
        length=4,
        field_name="requested_quaternion_wxyz",
    )
    actual_pose = _finite_tuple(
        actual_pose_xyz_wxyz,
        length=7,
        field_name="actual_pose_xyz_wxyz",
    )
    horizontal_displacement = math.hypot(
        actual_pose[0] - requested_position[0],
        actual_pose[1] - requested_position[1],
    )
    vertical_displacement = abs(actual_pose[2] - requested_position[2])
    orientation_error = quaternion_angle_error_rad(
        actual_pose[3:7],
        requested_quaternion,
    )
    max_horizontal = float(policy["max_horizontal_displacement_m"])
    max_vertical = float(policy["max_vertical_displacement_m"])
    max_orientation = float(policy["max_orientation_error_rad"])
    violations: list[str] = []
    if horizontal_displacement > max_horizontal:
        violations.append("horizontal_displacement")
    if vertical_displacement > max_vertical:
        violations.append("vertical_displacement")
    if orientation_error > max_orientation:
        violations.append("orientation_error")
    return {
        "enabled": bool(policy.get("enabled")),
        "mode": policy.get("mode"),
        "required_for_episode": bool(policy.get("required_for_episode", False)),
        "requested_position_xyz": list(requested_position),
        "requested_quaternion_wxyz": list(requested_quaternion),
        "actual_pose_xyz_wxyz": list(actual_pose),
        "horizontal_displacement_m": horizontal_displacement,
        "vertical_displacement_m": vertical_displacement,
        "orientation_error_rad": orientation_error,
        "orientation_error_deg": math.degrees(orientation_error),
        "max_horizontal_displacement_m": max_horizontal,
        "max_vertical_displacement_m": max_vertical,
        "max_orientation_error_rad": max_orientation,
        "violations": violations,
        "verified": not violations,
    }
