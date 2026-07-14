"""将同步仿真状态转换为可追溯的 VLA 训练动作。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


VLA_TRAINING_ACTION_SCHEMA = "base_xyyaw_tcp_base_rpy_gripper_v1"
VLA_TRAINING_ACTION_DIMENSION = 10
VLA_TRAINING_ACTION_NAMES = (
    "base_x_world",
    "base_y_world",
    "base_yaw_world",
    "tcp_x_base",
    "tcp_y_base",
    "tcp_z_base",
    "tcp_roll_base",
    "tcp_pitch_base",
    "tcp_yaw_base",
    "gripper_normalized",
)
VLA_TRAINING_ACTION_ALIGNMENT = "next_sample_executed_pose"
VLA_TRAINING_TERMINAL_ACTION = "hold_current_pose"

BASE_POSE_NAMES = (
    "base_x_world",
    "base_y_world",
    "base_z_world",
    "base_quat_w_world",
    "base_quat_x_world",
    "base_quat_y_world",
    "base_quat_z_world",
)

PHYSICAL_TRAINING_SUCCESS_SEMANTICS = frozenset(
    {
        "physical_execution",
        "strict_physical_execution",
        "stable_physical_execution_with_base_support_lock",
    }
)


def physical_execution_success_verified(summary: dict[str, Any]) -> bool:
    """统一判定完整物理执行与来源证据，排除 dry-run 和所有 smoke。"""

    return bool(
        summary.get("success") is True
        and not summary.get("failure_reason")
        and summary.get("success_semantics")
        in PHYSICAL_TRAINING_SUCCESS_SEMANTICS
        and summary.get("execution_provenance_verified") is True
    )


def training_visual_source_verified(summary: dict[str, Any]) -> bool:
    """VLA 任务要求同步相机帧来源完整；旧非 VLA summary 保持兼容。"""

    raw_task = summary.get("task_config")
    if not isinstance(raw_task, dict) or not task_requests_vla_training_action(raw_task):
        return True
    return summary.get("training_visual_source_verified") is True


def task_requires_receptacle_support_validation(task: dict[str, Any]) -> bool:
    """返回任务是否要求在真实 runtime stage 验证目标支撑体。"""

    raw_place = task.get("place")
    return bool(
        isinstance(raw_place, dict)
        and raw_place.get("support_runtime_validation_required", False)
    )


def training_receptacle_support_verified(summary: dict[str, Any]) -> bool:
    """要求垫子/桌面等目标支撑体在真实 stage 中具备一致碰撞几何。"""

    raw_task = summary.get("task_config")
    if not isinstance(raw_task, dict) or not task_requires_receptacle_support_validation(
        raw_task
    ):
        return True
    simulation_report = summary.get("simulation_report")
    if not isinstance(simulation_report, dict):
        return False
    report = simulation_report.get(
        "task_receptacle_support_runtime_stage_report"
    )
    if not isinstance(report, dict):
        return False
    raw_place = raw_task.get("place")
    raw_place = raw_place if isinstance(raw_place, dict) else {}
    region = report.get("placement_region_report")
    proxy = report.get("task_support_proxy_report")
    mesh_count = report.get("mesh_count")
    collision_enabled_count = report.get("collision_enabled_count")
    proxy_config = raw_place.get("curobo_world_collision")
    proxy_required = bool(
        isinstance(proxy_config, dict) and proxy_config.get("required", False)
    )
    return bool(
        report.get("configured") is True
        and report.get("geometry_verified") is True
        and isinstance(mesh_count, int)
        and not isinstance(mesh_count, bool)
        and mesh_count > 0
        and isinstance(collision_enabled_count, int)
        and not isinstance(collision_enabled_count, bool)
        and collision_enabled_count > 0
        and (
            not raw_place.get("support_expected_static", False)
            or report.get("static_support_verified") is True
        )
        and (
            not raw_place.get("placement_region")
            or (isinstance(region, dict) and region.get("verified") is True)
        )
        and (
            not proxy_required
            or (isinstance(proxy, dict) and proxy.get("verified") is True)
        )
    )


def task_requires_mesh_truth_manipulation_targets(task: dict[str, Any]) -> bool:
    """返回任务是否要求抓取/放置目标来自运行时 Mesh/PhysX 真值。"""

    config = task.get("mesh_truth_manipulation_targets")
    return bool(isinstance(config, dict) and config.get("required", False))


def _resolved_task_grasp_mode(task: dict[str, Any]) -> str | None:
    """返回运行时应执行的抓取模式；auto 与旧任务一致解析为 side。"""

    raw_pick = task.get("pick")
    raw_pick = raw_pick if isinstance(raw_pick, dict) else {}
    requested = str(raw_pick.get("grasp_mode") or "auto").strip().lower()
    requested = requested.replace("-", "_")
    if requested == "auto":
        return "side"
    if requested in {"side", "top_down"}:
        return requested
    return None


def training_mesh_truth_manipulation_targets_verified(
    summary: dict[str, Any],
) -> bool:
    """验证 CuRobo 实际导出使用了运行时 Mesh 真值，而非静态/视觉替代。"""

    raw_task = summary.get("task_config")
    if not isinstance(raw_task, dict) or not task_requires_mesh_truth_manipulation_targets(
        raw_task
    ):
        return True
    contract = raw_task.get("mesh_truth_manipulation_targets")
    if not isinstance(contract, dict):
        return False
    if contract.get("visual_localization_required") is not False:
        return False
    if contract.get("pick_tcp_source") != "runtime_live_object_bbox":
        return False
    if contract.get("place_tcp_source") != (
        "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
    ):
        return False
    expected_grasp_mode = _resolved_task_grasp_mode(raw_task)
    if expected_grasp_mode is None:
        return False
    expected_pick_source_type = f"sim_object_bbox_{expected_grasp_mode}"

    simulation_report = summary.get("simulation_report")
    if not isinstance(simulation_report, dict):
        return False
    pick_export = simulation_report.get("last_current_state_curobo_pick_export")
    place_export = simulation_report.get("last_current_state_curobo_place_export")
    if not isinstance(pick_export, dict) or not isinstance(place_export, dict):
        return False
    pick_report = pick_export.get("mesh_truth_pick_target_report")
    place_report = place_export.get("mesh_truth_place_target_report")
    if not isinstance(pick_report, dict) or not isinstance(place_report, dict):
        return False

    pick_verified = bool(
        pick_report.get("required") is True
        and pick_report.get("verified") is True
        and pick_report.get("visual_localization_used") is False
        and pick_report.get("pick_tcp_source") == "runtime_live_object_bbox"
        and pick_report.get("resolved_grasp_mode") == expected_grasp_mode
        and pick_report.get("target_source_type") == expected_pick_source_type
        and pick_report.get("bbox_center_source") == "live_physx_object_pose"
    )
    place_verified = bool(
        place_report.get("required") is True
        and place_report.get("verified") is True
        and place_report.get("visual_localization_used") is False
        and place_report.get("xyz_source") == "runtime_mesh_truth"
        and place_report.get("support_geometry_verified") is True
        and place_report.get("object_extent_consistency_verified") is True
        and place_report.get("configured_pose_consistency_verified") is True
        and place_report.get("current_object_center_live_verified") is True
        and place_report.get("place_tcp_source")
        == "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
        and place_report.get("current_tcp_offset_source")
        == "runtime_current_tcp_and_live_object_center"
    )
    if not pick_verified or not place_verified:
        return False

    derived_pose = place_report.get("derived_place_pose_world")
    desired_center = place_export.get("desired_final_object_center_world")
    if not isinstance(derived_pose, dict) or not isinstance(
        desired_center, (list, tuple)
    ):
        return False
    if len(desired_center) != 3:
        return False
    try:
        derived_xyz = tuple(float(derived_pose[axis]) for axis in ("x", "y", "z"))
        desired_xyz = tuple(float(value) for value in desired_center)
        tolerance_m = float(
            place_report.get("configured_pose_consistency_tolerance_m", 1.0e-6)
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        tolerance_m >= 0.0
        and all(math.isfinite(value) for value in (*derived_xyz, *desired_xyz))
        and max(
            abs(derived_xyz[index] - desired_xyz[index]) for index in range(3)
        )
        <= tolerance_m
    )


def training_quality_success_verified(summary: dict[str, Any]) -> bool:
    """物理来源、RGB、支撑体和 Mesh-truth 操作目标必须同时通过。"""

    return bool(
        physical_execution_success_verified(summary)
        and training_visual_source_verified(summary)
        and training_receptacle_support_verified(summary)
        and training_mesh_truth_manipulation_targets_verified(summary)
    )


def task_requests_vla_training_action(task: dict[str, Any]) -> bool:
    """返回任务是否显式要求标准 10 维 VLA action。"""

    config = task.get("training_action")
    return bool(isinstance(config, dict) and config.get("enabled", False))


def validate_vla_training_action_config(task: dict[str, Any]) -> dict[str, Any] | None:
    """严格校验任务中的动作 schema，避免转换时静默改变字段语义。"""

    if not task_requests_vla_training_action(task):
        return None
    config = dict(task["training_action"])
    expected_scalars = {
        "schema": VLA_TRAINING_ACTION_SCHEMA,
        "dimension": VLA_TRAINING_ACTION_DIMENSION,
        "base_pose_frame": "world",
        "tcp_pose_frame": "base_frame",
        "tcp_euler_order": "roll_pitch_yaw",
        "angle_unit": "rad",
        "position_unit": "m",
        "action_alignment": VLA_TRAINING_ACTION_ALIGNMENT,
        "terminal_action": VLA_TRAINING_TERMINAL_ACTION,
        "gripper_closed_value": 0.0,
        "gripper_open_value": 1.0,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise ValueError(
                f"training_action.{key} must be {expected!r}, got {config.get(key)!r}"
            )
    if config.get("gripper_range") != [0.0, 1.0]:
        raise ValueError("training_action.gripper_range must be [0.0, 1.0]")
    source_range = config.get("source_gripper_joint_range_m")
    if (
        not isinstance(source_range, list)
        or len(source_range) != 2
        or isinstance(source_range[0], bool)
        or isinstance(source_range[1], bool)
        or not isinstance(source_range[0], (int, float))
        or not isinstance(source_range[1], (int, float))
        or not math.isfinite(float(source_range[0]))
        or not math.isfinite(float(source_range[1]))
        or float(source_range[0]) >= float(source_range[1])
    ):
        raise ValueError(
            "training_action.source_gripper_joint_range_m must be two ordered finite values"
        )
    return config


def build_vla_training_actions(
    samples: Sequence[dict[str, Any]],
    *,
    source_gripper_joint_range_m: tuple[float, float],
) -> np.ndarray:
    """用下一采样帧的实际位姿生成动作，末帧显式导出保持动作。"""

    if not samples:
        return np.empty((0, VLA_TRAINING_ACTION_DIMENSION), dtype=np.float32)
    closed, opened = (float(value) for value in source_gripper_joint_range_m)
    if not math.isfinite(closed) or not math.isfinite(opened) or closed >= opened:
        raise ValueError("source gripper joint range must be finite and ordered")

    actions: list[list[float]] = []
    for frame_index in range(len(samples)):
        target_index = min(frame_index + 1, len(samples) - 1)
        target = samples[target_index]
        base_pose = _finite_vector(target.get("base_pose"), 7, name="base_pose")
        tcp_pose = _finite_vector(target.get("tcp_pose"), 7, name="tcp_pose")
        if target.get("tcp_pose_valid") is not True:
            raise ValueError(
                f"sample {target_index} has no valid TCP pose for VLA action export"
            )
        gripper_position = target.get("gripper_position")
        if (
            isinstance(gripper_position, bool)
            or not isinstance(gripper_position, (int, float))
            or not math.isfinite(float(gripper_position))
        ):
            raise ValueError(
                f"sample {target_index} has no finite gripper_position for VLA action export"
            )

        base_quat = _normalize_quat_wxyz(base_pose[3:7], name="base_pose quaternion")
        tcp_quat = _normalize_quat_wxyz(tcp_pose[3:7], name="tcp_pose quaternion")
        base_yaw = _quat_wxyz_to_rpy(base_quat)[2]
        tcp_position_base = _rotate_vector_by_quat_wxyz(
            _quat_conjugate(base_quat),
            (
                tcp_pose[0] - base_pose[0],
                tcp_pose[1] - base_pose[1],
                tcp_pose[2] - base_pose[2],
            ),
        )
        tcp_quat_base = _normalize_quat_wxyz(
            _quat_multiply(_quat_conjugate(base_quat), tcp_quat),
            name="base-frame TCP quaternion",
        )
        tcp_rpy_base = _quat_wxyz_to_rpy(tcp_quat_base)
        gripper_normalized = min(
            1.0,
            max(0.0, (float(gripper_position) - closed) / (opened - closed)),
        )
        action = [
            base_pose[0],
            base_pose[1],
            base_yaw,
            *tcp_position_base,
            *tcp_rpy_base,
            gripper_normalized,
        ]
        if len(action) != VLA_TRAINING_ACTION_DIMENSION or not all(
            math.isfinite(value) for value in action
        ):
            raise ValueError(f"sample {target_index} produced an invalid VLA action")
        actions.append(action)
    return np.asarray(actions, dtype=np.float32)


def _finite_vector(value: Any, length: int, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalize_quat_wxyz(
    quat: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quat)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain four finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} norm is zero")
    normalized = tuple(value / norm for value in values)
    # q 与 -q 表示同一旋转；统一半球可减少离线动作的不必要跳变。
    if normalized[0] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def _quat_conjugate(
    quat: Sequence[float],
) -> tuple[float, float, float, float]:
    w, x, y, z = (float(value) for value in quat)
    return w, -x, -y, -z


def _quat_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector_by_quat_wxyz(
    quat: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    pure = (0.0, *(float(value) for value in vector))
    rotated = _quat_multiply(_quat_multiply(quat, pure), _quat_conjugate(quat))
    return float(rotated[1]), float(rotated[2]), float(rotated[3])


def _quat_wxyz_to_rpy(quat: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw
