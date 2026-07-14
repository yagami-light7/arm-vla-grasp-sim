"""基于当前仿真状态生成 cuRobo pick/place 输入并调用规划器。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from source.interfaces import ArmPlan, EpisodeSpec, SimulationState

from .curobo_adapter import arm_plan_from_curobo_payload, load_curobo_plan_json
from .grasp_pipeline import GraspPipeline, GraspPipelineConfig, GraspTask


ACTIVE_ARM_JOINT_NAMES = (
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
)
GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")

TIP_TCP_INSERTION_BEYOND_GRASP_CENTER_M = 0.005
SIDE_GRASP_CENTER_Z_OFFSET_M = 0.0075
# 张开夹爪最低碰撞球约位于 TCP 下方 36 mm；真实跟踪还需留出安全余量。
OPEN_GRIPPER_TCP_BELOW_EXTENT_M = 0.036
# baseline 不通过抬高 grasp center 规避桌面；桌面碰撞应由路径/碰撞模型处理。
SIDE_GRASP_SUPPORT_CLEARANCE_M = 0.0
# 对齐 baseline 的 GO2_X5_GRIPPER_CLOSE_M=0.0。真实接触会阻止关节完全到零，
# 但持续保持零目标能在 carry 阶段提供夹紧力，避免苹果在导航起步时滑落。
SIDE_GRASP_SAFE_CLOSE_M = 0.0
SIDE_PREGRASP_OFFSET_M = 0.10
TOP_GRASP_DEPTH_BELOW_TOP_M = 0.035
TOP_PREGRASP_OFFSET_M = 0.10
LIFT_OFFSET_M = 0.10
ARM_PLACE_RELEASE_CLEARANCE_M = 0.013
ARM_PLACE_PRE_PLACE_CLEARANCE_M = 0.06
ARM_PLACE_RETREAT_CLEARANCE_M = 0.08

WORKSPACE_WARN_XY_RADIUS_M = 0.55
WORKSPACE_WARN_GRASP_Z_M = 0.35
WORKSPACE_WARN_PREGRASP_Z_M = 0.45
WORKSPACE_WARN_RADIUS_3D_M = 0.65

# grasp_tcp_link 局部 +X 是夹爪伸出方向；该姿态把 +X 对齐到 base -Z。
TCP_TOP_DOWN_QUAT_BASE_WXYZ = np.array(
    [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
    dtype=float,
)


def normalize_quat_wxyz(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        raise RuntimeError(f"四元数范数异常: {quaternion}")
    return q / norm


def rotmat_to_quat_wxyz(rotation: Any) -> np.ndarray:
    """把 3x3 rotation matrix 转成 wxyz 四元数。"""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=float,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=float,
            )
        elif diagonal_index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=float,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=float,
            )
    return normalize_quat_wxyz(quat)


def pose_to_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    """使用 wxyz 四元数构造标准 SE(3) 矩阵。"""

    x, y, z = np.asarray(position, dtype=float)
    w, qx, qy, qz = normalize_quat_wxyz(quaternion_wxyz)
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * w),
                2.0 * (qx * qz + qy * w),
            ],
            [
                2.0 * (qx * qy + qz * w),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * w),
            ],
            [
                2.0 * (qx * qz - qy * w),
                2.0 * (qy * qz + qx * w),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (x, y, z)
    return matrix


def matrix_to_pose(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    se3 = np.asarray(matrix, dtype=float)
    return se3[:3, 3].copy(), rotmat_to_quat_wxyz(se3[:3, :3])


def pose_dict_from_matrix(matrix: Any) -> dict[str, Any]:
    position, quaternion = matrix_to_pose(matrix)
    return {
        "matrix_4x4": np.asarray(matrix, dtype=float).tolist(),
        "position_xyz": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def build_curobo_state_payload(
    *,
    q_arm: Any,
    dq_arm: Any,
    T_world_base: Any,
    T_world_tcp: Any | None = None,
    q_full: Any | None = None,
    dq_full: Any | None = None,
    dof_names: tuple[str, ...] | list[str] = (),
    q_gripper: Any | None = None,
    dq_gripper: Any | None = None,
    robot_root_path: str,
    articulation_root_path: str,
    base_frame_path: str,
    tcp_frame_path: str,
    tcp_mode: str,
    world_collision_cuboids: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    world_collision_metadata: dict[str, Any] | None = None,
    source: str = "IsaacLabNavigationRuntime",
) -> dict[str, Any]:
    """组装 `03_plan_grasp_trajectory.py` 所需的最小状态 JSON。"""

    q_arm_array = np.asarray(q_arm, dtype=float)
    dq_arm_array = np.asarray(dq_arm, dtype=float)
    if q_arm_array.shape != (len(ACTIVE_ARM_JOINT_NAMES),):
        raise RuntimeError(f"q_arm 维度必须为 6，当前 {q_arm_array.shape}")
    if dq_arm_array.shape != q_arm_array.shape:
        raise RuntimeError(f"dq_arm 维度必须和 q_arm 一致，当前 {dq_arm_array.shape}")

    T_world_base = np.asarray(T_world_base, dtype=float)
    if T_world_base.shape != (4, 4):
        raise RuntimeError(f"T_world_base 必须是 4x4，当前 {T_world_base.shape}")
    if T_world_tcp is None:
        T_world_tcp = T_world_base.copy()
    T_world_tcp = np.asarray(T_world_tcp, dtype=float)
    T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp

    dof_names = tuple(str(name) for name in dof_names)
    q_full_array = np.asarray(q_full if q_full is not None else [], dtype=float)
    dq_full_array = np.asarray(dq_full if dq_full is not None else [], dtype=float)
    arm_index_map = {
        name: dof_names.index(name)
        for name in ACTIVE_ARM_JOINT_NAMES
        if name in dof_names
    }
    gripper_names = tuple(name for name in GRIPPER_JOINT_NAMES if name in dof_names)
    gripper_index_map = {
        name: dof_names.index(name)
        for name in gripper_names
        if name in dof_names
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "robot_name": "go2_x5",
        "source": source,
        "paths": {
            "robot_root_path": robot_root_path,
            "articulation_root_path": articulation_root_path,
            "base_frame_path": base_frame_path,
            "tcp_frame_path": tcp_frame_path,
            "tcp_mode": tcp_mode,
        },
        "planner_convention": {
            "base_link": "arm_base_link",
            "tool_frame": "grasp_tcp_link",
            "active_joint_names": list(ACTIVE_ARM_JOINT_NAMES),
            "active_joint_order_note": "q_arm follows active_joint_names and cuRobo joint_names order.",
        },
        "isaac_state": {
            "dof_count": len(dof_names),
            "dof_names": list(dof_names),
            "q_full": q_full_array.tolist(),
            "dq_full": dq_full_array.tolist(),
            "arm_joint_indices": arm_index_map,
            "q_arm": q_arm_array.tolist(),
            "dq_arm": dq_arm_array.tolist(),
            "gripper_joint_names": list(gripper_names),
            "gripper_joint_indices": gripper_index_map,
            "q_gripper": np.asarray(q_gripper if q_gripper is not None else [], dtype=float).tolist(),
            "dq_gripper": np.asarray(dq_gripper if dq_gripper is not None else [], dtype=float).tolist(),
        },
        "poses": {
            "world_base": pose_dict_from_matrix(T_world_base),
            "world_tcp": pose_dict_from_matrix(T_world_tcp),
            "base_tcp": pose_dict_from_matrix(T_base_tcp),
        },
        "world_collision": {
            "enabled": bool(world_collision_cuboids),
            "representation": (
                "IsaacLab current CollisionAPI world AABB exported as cuRobo cuboids "
                "in arm_base_link frame"
            ),
            **dict(world_collision_metadata or {}),
            "cuboids_base": list(world_collision_cuboids),
        },
    }


def build_side_grasp_target_payload(
    *,
    object_prim_path: str,
    T_world_base: Any,
    bbox_min: Any,
    bbox_max: Any,
    bbox_center: Any,
    bbox_size: Any,
) -> dict[str, Any]:
    """按 baseline side-grasp 几何规则生成当前物体的 pick target JSON。"""

    T_world_base = np.asarray(T_world_base, dtype=float)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    bbox_center = np.asarray(bbox_center, dtype=float)
    bbox_size = np.asarray(bbox_size, dtype=float)

    T_base_grasp_contact = _make_side_grasp_contact_pose_base(
        T_world_base,
        bbox_center,
        bbox_min,
    )
    T_base_grasp = _offset_pose_along_local_x(
        T_base_grasp_contact,
        TIP_TCP_INSERTION_BEYOND_GRASP_CENTER_M,
    )
    T_base_pregrasp = _offset_pose_along_local_x(
        T_base_grasp,
        -SIDE_PREGRASP_OFFSET_M,
    )
    T_base_lift = _offset_pose_along_base_z(T_base_grasp, LIFT_OFFSET_M)

    T_world_pregrasp = T_world_base @ T_base_pregrasp
    T_world_grasp = T_world_base @ T_base_grasp
    T_world_grasp_contact = T_world_base @ T_base_grasp_contact
    T_world_lift = T_world_base @ T_base_lift

    diagnostics = _make_target_workspace_diagnostics(
        T_base_pregrasp,
        T_base_grasp,
        T_base_lift,
    )
    grasp_pos_base, grasp_quat_base = matrix_to_pose(T_base_grasp)
    grasp_pos_world, grasp_quat_world = matrix_to_pose(T_world_grasp)
    grasp_contact_pos_world, grasp_contact_quat_world = matrix_to_pose(
        T_world_grasp_contact
    )
    pregrasp_pos_world, pregrasp_quat_world = matrix_to_pose(T_world_pregrasp)
    lift_pos_world, lift_quat_world = matrix_to_pose(T_world_lift)

    return {
        "schema_version": 1,
        "frame": "arm_base_link",
        "default_target_name": "grasp",
        "position_xyz": grasp_pos_base.tolist(),
        "quaternion_wxyz": grasp_quat_base.tolist(),
        "sequence": ["pregrasp", "grasp", "close_gripper", "lift"],
        "poses": {
            "pregrasp": _make_named_pose_entry(T_base_pregrasp, T_world_pregrasp),
            "grasp": _make_named_pose_entry(T_base_grasp, T_world_grasp),
            "lift": _make_named_pose_entry(T_base_lift, T_world_lift),
        },
        "gripper": {
            "open_m": 0.043,
            "close_m": SIDE_GRASP_SAFE_CLOSE_M,
            "joint_names": list(GRIPPER_JOINT_NAMES),
        },
        "source": {
            "type": "sim_object_bbox_side",
            "grasp_mode": "side",
            "object_prim_path": object_prim_path,
            "world_grasp_pose": {
                "position_xyz": grasp_pos_world.tolist(),
                "quaternion_wxyz": grasp_quat_world.tolist(),
            },
            "world_grasp_contact_pose": {
                "position_xyz": grasp_contact_pos_world.tolist(),
                "quaternion_wxyz": grasp_contact_quat_world.tolist(),
            },
            "grasp_contact_pose": _make_named_pose_entry(
                T_base_grasp_contact,
                T_world_grasp_contact,
            ),
            "world_pregrasp_pose": {
                "position_xyz": pregrasp_pos_world.tolist(),
                "quaternion_wxyz": pregrasp_quat_world.tolist(),
            },
            "world_lift_pose": {
                "position_xyz": lift_pos_world.tolist(),
                "quaternion_wxyz": lift_quat_world.tolist(),
            },
            "bbox_world": {
                "min_xyz": bbox_min.tolist(),
                "max_xyz": bbox_max.tolist(),
                "center_xyz": bbox_center.tolist(),
                "size_xyz": bbox_size.tolist(),
            },
            "side_grasp_center_z_offset_m": SIDE_GRASP_CENTER_Z_OFFSET_M,
            "open_gripper_tcp_below_extent_m": OPEN_GRIPPER_TCP_BELOW_EXTENT_M,
            "side_grasp_support_clearance_m": SIDE_GRASP_SUPPORT_CLEARANCE_M,
            "side_grasp_safe_close_m": SIDE_GRASP_SAFE_CLOSE_M,
            "applied_grasp_center_z_offset_m": float(
                grasp_contact_pos_world[2] - bbox_center[2]
            ),
            "estimated_open_gripper_bottom_clearance_m": float(
                grasp_contact_pos_world[2]
                - OPEN_GRIPPER_TCP_BELOW_EXTENT_M
                - bbox_min[2]
            ),
            "tip_tcp_insertion_beyond_grasp_center_m": (
                TIP_TCP_INSERTION_BEYOND_GRASP_CENTER_M
            ),
            "side_pregrasp_offset_m": SIDE_PREGRASP_OFFSET_M,
            "lift_offset_m": LIFT_OFFSET_M,
            "tcp_orientation_rule": (
                "side: grasp_tcp_link local +X points horizontally from "
                "arm_base_link to object"
            ),
        },
        "diagnostics": {
            "target_workspace_base": diagnostics,
        },
    }


def build_top_down_grasp_target_payload(
    *,
    object_prim_path: str,
    T_world_base: Any,
    bbox_min: Any,
    bbox_max: Any,
    bbox_center: Any,
    bbox_size: Any,
) -> dict[str, Any]:
    """从物体上方竖直下探，生成当前物体的 pick target JSON。"""

    T_world_base = np.asarray(T_world_base, dtype=float)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    bbox_center = np.asarray(bbox_center, dtype=float)
    bbox_size = np.asarray(bbox_size, dtype=float)

    grasp_position_world = bbox_center.copy()
    grasp_position_world[2] = bbox_max[2] - TOP_GRASP_DEPTH_BELOW_TOP_M
    grasp_position_base = _world_point_to_base_position(
        T_world_base,
        grasp_position_world,
    )
    T_base_grasp = pose_to_matrix(
        grasp_position_base,
        TCP_TOP_DOWN_QUAT_BASE_WXYZ,
    )
    T_base_pregrasp = _offset_pose_along_base_z(
        T_base_grasp,
        TOP_PREGRASP_OFFSET_M,
    )
    T_base_lift = _offset_pose_along_base_z(T_base_grasp, LIFT_OFFSET_M)

    T_world_pregrasp = T_world_base @ T_base_pregrasp
    T_world_grasp = T_world_base @ T_base_grasp
    T_world_lift = T_world_base @ T_base_lift
    diagnostics = _make_target_workspace_diagnostics(
        T_base_pregrasp,
        T_base_grasp,
        T_base_lift,
    )
    grasp_pos_base, grasp_quat_base = matrix_to_pose(T_base_grasp)
    grasp_pos_world, grasp_quat_world = matrix_to_pose(T_world_grasp)
    pregrasp_pos_world, pregrasp_quat_world = matrix_to_pose(T_world_pregrasp)
    lift_pos_world, lift_quat_world = matrix_to_pose(T_world_lift)

    return {
        "schema_version": 1,
        "frame": "arm_base_link",
        "default_target_name": "grasp",
        "position_xyz": grasp_pos_base.tolist(),
        "quaternion_wxyz": grasp_quat_base.tolist(),
        "sequence": ["pregrasp", "grasp", "close_gripper", "lift"],
        "poses": {
            "pregrasp": _make_named_pose_entry(T_base_pregrasp, T_world_pregrasp),
            "grasp": _make_named_pose_entry(T_base_grasp, T_world_grasp),
            "lift": _make_named_pose_entry(T_base_lift, T_world_lift),
        },
        "gripper": {
            "open_m": 0.043,
            "close_m": SIDE_GRASP_SAFE_CLOSE_M,
            "joint_names": list(GRIPPER_JOINT_NAMES),
        },
        "source": {
            "type": "sim_object_bbox_top_down",
            "grasp_mode": "top_down",
            "object_prim_path": object_prim_path,
            "world_grasp_pose": {
                "position_xyz": grasp_pos_world.tolist(),
                "quaternion_wxyz": grasp_quat_world.tolist(),
            },
            "world_pregrasp_pose": {
                "position_xyz": pregrasp_pos_world.tolist(),
                "quaternion_wxyz": pregrasp_quat_world.tolist(),
            },
            "world_lift_pose": {
                "position_xyz": lift_pos_world.tolist(),
                "quaternion_wxyz": lift_quat_world.tolist(),
            },
            "bbox_world": {
                "min_xyz": bbox_min.tolist(),
                "max_xyz": bbox_max.tolist(),
                "center_xyz": bbox_center.tolist(),
                "size_xyz": bbox_size.tolist(),
            },
            "grasp_depth_below_top_m": TOP_GRASP_DEPTH_BELOW_TOP_M,
            "pregrasp_offset_m": TOP_PREGRASP_OFFSET_M,
            "lift_offset_m": LIFT_OFFSET_M,
            "tcp_orientation_rule": (
                "base_top_down: grasp_tcp_link local +X points to arm_base_link -Z"
            ),
        },
        "diagnostics": {
            "target_workspace_base": diagnostics,
        },
    }


def build_grasp_target_payload(
    *,
    grasp_mode: str,
    object_prim_path: str,
    T_world_base: Any,
    bbox_min: Any,
    bbox_max: Any,
    bbox_center: Any,
    bbox_size: Any,
) -> dict[str, Any]:
    """按显式抓取模式分派目标生成；auto 保持旧 side 行为。"""

    normalized_mode = str(grasp_mode).strip().lower().replace("-", "_")
    if normalized_mode == "auto":
        normalized_mode = "side"
    builders = {
        "side": build_side_grasp_target_payload,
        "top_down": build_top_down_grasp_target_payload,
    }
    builder = builders.get(normalized_mode)
    if builder is None:
        raise ValueError(
            "grasp_mode 必须是 auto、side 或 top_down，"
            f"当前为 {grasp_mode!r}"
        )
    return builder(
        object_prim_path=object_prim_path,
        T_world_base=T_world_base,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_center=bbox_center,
        bbox_size=bbox_size,
    )


def build_arm_place_target_payload(
    *,
    object_prim_path: str,
    T_world_base: Any,
    T_world_tcp: Any,
    bbox_min: Any,
    bbox_max: Any,
    bbox_center: Any,
    bbox_size: Any,
    place_pose_world: dict[str, Any],
    pick_grasp_quaternion_base: Any | None = None,
) -> dict[str, Any]:
    """按 baseline arm-place 规则生成 place target。

    place_pose_world 描述的是最终希望物体中心到达的位置，不是 TCP 位置。
    因此必须保留当前 TCP 到物体中心的偏移，再反推出每个放置阶段的 TCP 目标。
    """

    T_world_base = np.asarray(T_world_base, dtype=float)
    T_world_tcp = np.asarray(T_world_tcp, dtype=float)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    bbox_center = np.asarray(bbox_center, dtype=float)
    bbox_size = np.asarray(bbox_size, dtype=float)
    for key in ("x", "y", "z"):
        if key not in place_pose_world:
            raise RuntimeError(f"place_pose_world 缺少 {key}，无法生成 arm-place target。")

    place_center_world = np.asarray(
        [
            float(place_pose_world["x"]),
            float(place_pose_world["y"]),
            float(place_pose_world["z"]),
        ],
        dtype=float,
    )
    release_clearance = max(
        0.0,
        float(place_pose_world.get("release_clearance", ARM_PLACE_RELEASE_CLEARANCE_M)),
    )
    pre_place_clearance = max(
        release_clearance,
        float(place_pose_world.get("pre_place_clearance", ARM_PLACE_PRE_PLACE_CLEARANCE_M)),
    )
    retreat_clearance = max(
        release_clearance,
        float(place_pose_world.get("retreat_clearance", ARM_PLACE_RETREAT_CLEARANCE_M)),
    )

    tcp_to_object_center_world = bbox_center - T_world_tcp[:3, 3]
    target_object_centers = {
        "pre_place": place_center_world + np.array([0.0, 0.0, pre_place_clearance], dtype=float),
        "place": place_center_world + np.array([0.0, 0.0, release_clearance], dtype=float),
        "retreat": place_center_world + np.array([0.0, 0.0, retreat_clearance], dtype=float),
    }

    pick_quat = (
        normalize_quat_wxyz(pick_grasp_quaternion_base)
        if pick_grasp_quaternion_base is not None
        else None
    )
    T_base_world = np.linalg.inv(T_world_base)
    target_poses: dict[str, Any] = {}
    target_world_matrices: dict[str, Any] = {}
    for name, object_center_target in target_object_centers.items():
        target_tcp_world_pos = object_center_target - tcp_to_object_center_world
        if pick_quat is not None:
            target_tcp_base_pos = (
                T_base_world
                @ np.array(
                    [target_tcp_world_pos[0], target_tcp_world_pos[1], target_tcp_world_pos[2], 1.0],
                    dtype=float,
                )
            )[:3]
            T_base_target_tcp = pose_to_matrix(target_tcp_base_pos, pick_quat)
            T_world_target_tcp = T_world_base @ T_base_target_tcp
        else:
            # 没有 pick target 方向时保持当前 TCP 朝向，仅平移到 baseline 计算的位置。
            T_world_target_tcp = T_world_tcp.copy()
            T_world_target_tcp[:3, 3] = target_tcp_world_pos
            T_base_target_tcp = T_base_world @ T_world_target_tcp
        target_poses[name] = _make_named_pose_entry(T_base_target_tcp, T_world_target_tcp)
        target_world_matrices[name] = np.asarray(T_world_target_tcp, dtype=float).tolist()

    orientation_rule = (
        "reuse_pick_target_grasp_orientation"
        if pick_quat is not None
        else "preserve_current_tcp_orientation_after_pick"
    )
    return {
        "schema_version": 1,
        "frame": "arm_base_link",
        "default_target_name": "place",
        "sequence": ["pre_place", "place", "open_gripper", "retreat"],
        "poses": target_poses,
        "gripper": {
            "open_m": 0.043,
            "close_m": 0.0,
            "joint_names": list(GRIPPER_JOINT_NAMES),
        },
        "source": {
            "type": "full_physics_arm_place_target",
            "mode": "arm_place",
            "object_prim_path": object_prim_path,
            "place_pose_world": {
                "x": float(place_pose_world["x"]),
                "y": float(place_pose_world["y"]),
                "z": float(place_pose_world["z"]),
                "roll": float(place_pose_world.get("roll", 0.0)),
                "pitch": float(place_pose_world.get("pitch", 0.0)),
                "yaw": float(place_pose_world.get("yaw", 0.0)),
            },
            "current_object_bbox_world": {
                "min_xyz": bbox_min.tolist(),
                "max_xyz": bbox_max.tolist(),
                "center_xyz": bbox_center.tolist(),
                "size_xyz": bbox_size.tolist(),
            },
            "current_tcp_world": pose_dict_from_matrix(T_world_tcp),
            "tcp_to_object_center_world_xyz": tcp_to_object_center_world.tolist(),
            "target_object_centers_world": {
                name: center.tolist() for name, center in target_object_centers.items()
            },
            "target_tcp_matrices_world": target_world_matrices,
            "desired_final_object_center_world": place_center_world.tolist(),
            "release_object_center_world": target_object_centers["place"].tolist(),
            "pre_place_object_center_world": target_object_centers["pre_place"].tolist(),
            "retreat_object_center_world": target_object_centers["retreat"].tolist(),
            "place_strategy": "vertical_clearance_place",
            "place_clearances_m": {
                "pre_place": pre_place_clearance,
                "release": release_clearance,
                "retreat": retreat_clearance,
            },
            "orientation_rule": orientation_rule,
            "pick_grasp_quaternion_base": None if pick_quat is None else pick_quat.tolist(),
        },
        "diagnostics": {
            "target_workspace_base": _place_target_workspace_diagnostics(target_poses),
            "current_tcp_base": pose_dict_from_matrix(T_base_world @ T_world_tcp),
            "current_tcp_world": pose_dict_from_matrix(T_world_tcp),
            "orientation_rule": orientation_rule,
        },
    }


def _world_point_to_base_position(T_world_base: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    point_world_h = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=float)
    return (np.linalg.inv(T_world_base) @ point_world_h)[:3]


def _make_side_grasp_contact_pose_base(
    T_world_base: np.ndarray,
    bbox_center: np.ndarray,
    bbox_min: np.ndarray,
) -> np.ndarray:
    del bbox_min
    grasp_position_world = np.asarray(bbox_center, dtype=float).copy()
    grasp_position_world[2] += SIDE_GRASP_CENTER_Z_OFFSET_M
    grasp_position_base = _world_point_to_base_position(T_world_base, grasp_position_world)

    approach_xy = grasp_position_base[:2].copy()
    approach_norm = float(np.linalg.norm(approach_xy))
    if approach_norm < 1.0e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_axis = np.array(
            [approach_xy[0] / approach_norm, approach_xy[1] / approach_norm, 0.0],
            dtype=float,
        )

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = grasp_position_base
    return matrix


def _offset_pose_along_local_x(T_base_pose: np.ndarray, offset_x_m: float) -> np.ndarray:
    shifted = np.asarray(T_base_pose, dtype=float).copy()
    shifted[:3, 3] += shifted[:3, 0] * float(offset_x_m)
    return shifted


def _offset_pose_along_base_z(T_base_pose: np.ndarray, offset_z_m: float) -> np.ndarray:
    shifted = np.asarray(T_base_pose, dtype=float).copy()
    shifted[2, 3] += float(offset_z_m)
    return shifted


def _make_named_pose_entry(T_base_pose: np.ndarray, T_world_pose: np.ndarray) -> dict[str, Any]:
    base_position, base_quaternion = matrix_to_pose(T_base_pose)
    world_position, world_quaternion = matrix_to_pose(T_world_pose)
    return {
        "frame": "arm_base_link",
        "position_xyz": base_position.tolist(),
        "quaternion_wxyz": base_quaternion.tolist(),
        "world": {
            "frame": "world",
            "position_xyz": world_position.tolist(),
            "quaternion_wxyz": world_quaternion.tolist(),
        },
    }


def _make_target_workspace_diagnostics(
    T_base_pregrasp: np.ndarray,
    T_base_grasp: np.ndarray,
    T_base_lift: np.ndarray,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []

    for name, matrix in (
        ("pregrasp", T_base_pregrasp),
        ("grasp", T_base_grasp),
        ("lift", T_base_lift),
    ):
        position, quaternion = matrix_to_pose(matrix)
        xy_radius = float(np.linalg.norm(position[:2]))
        radius_3d = float(np.linalg.norm(position))
        diagnostics[name] = {
            "position_xyz": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "xy_radius_m": xy_radius,
            "radius_3d_m": radius_3d,
            "z_m": float(position[2]),
        }

    grasp = diagnostics["grasp"]
    pregrasp = diagnostics["pregrasp"]
    if grasp["xy_radius_m"] > WORKSPACE_WARN_XY_RADIUS_M:
        warnings.append("grasp 的水平距离偏大，可能接近机械臂可达边界。")
    if grasp["z_m"] > WORKSPACE_WARN_GRASP_Z_M:
        warnings.append("grasp 在 arm_base_link 上方较高，可能难以规划。")
    if pregrasp["z_m"] > WORKSPACE_WARN_PREGRASP_Z_M:
        warnings.append("pregrasp 比 arm_base_link 高很多，move_to_pregrasp 可能失败。")
    if pregrasp["radius_3d_m"] > WORKSPACE_WARN_RADIUS_3D_M:
        warnings.append("pregrasp 的三维距离偏大，建议调整导航 handoff。")
    diagnostics["warnings"] = warnings
    return diagnostics


def _place_target_workspace_diagnostics(target_poses: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    for name, entry in target_poses.items():
        position = np.asarray(entry.get("position_xyz"), dtype=float)
        quaternion = normalize_quat_wxyz(entry.get("quaternion_wxyz"))
        xy_radius = float(np.linalg.norm(position[:2]))
        radius_3d = float(np.linalg.norm(position))
        diagnostics[name] = {
            "position_xyz": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "xy_radius_m": xy_radius,
            "radius_3d_m": radius_3d,
            "z_m": float(position[2]),
        }
        if radius_3d > WORKSPACE_WARN_RADIUS_3D_M:
            warnings.append(f"{name} 的三维距离偏大，place 可能接近机械臂可达边界。")
    diagnostics["warnings"] = warnings
    return diagnostics


def _pick_grasp_quaternion_from_target_json(path: str | Path | None) -> list[float] | None:
    if path is None:
        return None
    target_path = Path(path)
    if not target_path.exists():
        return None
    try:
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        ((payload.get("poses") or {}).get("grasp") or {}).get("quaternion_wxyz"),
        payload.get("quaternion_wxyz"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return normalize_quat_wxyz(candidate).tolist()
        except RuntimeError:
            continue
    return None


@dataclass(frozen=True)
class CurrentStateCuroboPlannerConfig:
    """当前状态重规划的文件与外部 cuRobo 设置。"""

    output_dir: Path
    project_root: Path
    place_plan_json: Path | None = None
    curobo_python: str = os.environ.get(
        "GO2_X5_CUROBO_PYTHON",
        "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python",
    )
    use_planner_server: bool = True
    # full-physics 对齐 random nav-pick-place baseline：侧抓后沿 approach 原路撤回，避免 lift/down 剐蹭桌面。
    side_grasp_plan_vertical_lift: bool = False
    side_grasp_fallback_retreat: bool = False
    side_grasp_retreat_to_pregrasp: bool = False
    # full-physics 里必须显式等待 pregrasp 到位，再进入 grasp；回撤同理先离开桌沿再回 home。
    split_pregrasp_motion: bool = True
    # pick 后回 home/carry 时 TCP 姿态已经不同于 pick grasp；place 默认保持当前 TCP 姿态。
    reuse_pick_grasp_orientation_for_place: bool = False


class CurrentStateCuroboPlanner:
    """pick/place 都使用当前 IsaacLab 状态重规划。"""

    def __init__(
        self,
        *,
        simulation: Any,
        config: CurrentStateCuroboPlannerConfig,
        plan_runner: Callable[[GraspTask], dict[str, Any]] | None = None,
    ):
        self._simulation = simulation
        self._config = config
        self._last_pick_grasp_quaternion_base: list[float] | None = None
        if plan_runner is None:
            grasp_pipeline = GraspPipeline(
                GraspPipelineConfig(
                    workspace=config.project_root,
                    curobo_python=config.curobo_python,
                    side_grasp_plan_vertical_lift=config.side_grasp_plan_vertical_lift,
                    side_grasp_fallback_retreat=config.side_grasp_fallback_retreat,
                    side_grasp_retreat_to_pregrasp=config.side_grasp_retreat_to_pregrasp,
                    split_pregrasp_motion=config.split_pregrasp_motion,
                )
            )
            plan_runner = grasp_pipeline.plan
        self._plan_runner = plan_runner

    def plan_pick(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        exporter = getattr(self._simulation, "export_current_curobo_pick_inputs", None)
        if not callable(exporter):
            raise RuntimeError(
                "当前 simulation backend 不支持 current-state cuRobo pick export，"
                "full-physics 模式禁止回退到离线 pick plan。"
            )

        replan_dir = self._config.output_dir / f"pick_replan_step_{state.step_index:06d}"
        replan_dir.mkdir(parents=True, exist_ok=True)
        export_report = exporter(
            output_dir=replan_dir,
            episode_spec=episode_spec,
            state=state,
        )
        self._last_pick_grasp_quaternion_base = _pick_grasp_quaternion_from_target_json(
            export_report.get("target_json")
        )
        plan_json = replan_dir / "pick_plan.json"
        task = GraspTask(
            object_prim_path=str(export_report.get("object_prim_path") or ""),
            use_planner_server=self._config.use_planner_server,
            state_json=str(export_report["state_json"]),
            target_json=str(export_report["target_json"]),
            plan_json=str(plan_json),
            result_json=str(replan_dir / "pick_execution_result.json"),
        )

        started_at = time.time()
        payload = self._plan_runner(task)
        elapsed_s = time.time() - started_at
        if not isinstance(payload, dict):
            payload = load_curobo_plan_json(plan_json)
        segment_names = _motion_segment_names(payload)
        pick_target = export_report.get("pick_target")
        pick_target = pick_target if isinstance(pick_target, dict) else {}
        pick_source = pick_target.get("source")
        pick_source = pick_source if isinstance(pick_source, dict) else {}
        grasp_mode = str(pick_source.get("grasp_mode") or "side")
        expected_side_retreat = bool(
            grasp_mode == "side"
            and not self._config.side_grasp_plan_vertical_lift
        )
        if expected_side_retreat and (
            "lift_object" in segment_names or "retreat_object" not in segment_names
        ):
            raise RuntimeError(
                "full-physics pick 必须使用 baseline side-retreat 轨迹："
                "需要 retreat_object，且禁止 lift_object。"
                f"当前 cuRobo 返回 motion segments={segment_names}。"
                "如果正在复用 planner server，请重启 server 后再试。"
            )
        if grasp_mode == "top_down" and "lift_object" not in segment_names:
            raise RuntimeError(
                "top-down pick 必须在闭合夹爪后执行 lift_object；"
                f"当前 cuRobo 返回 motion segments={segment_names}。"
            )
        plan = arm_plan_from_curobo_payload(
            payload,
            operation="pick",
            source_path=plan_json,
        )
        plan.metadata["current_state_replan"] = {
            "enabled": True,
            "operation": "pick",
            "state_json": str(task.state_json),
            "target_json": str(task.target_json),
            "plan_json": str(task.plan_json),
            "elapsed_wall_time_s": elapsed_s,
            "export_report": _json_safe(export_report),
        }
        plan.metadata["current_state_pick_strategy"] = {
            "expected": (
                "baseline_side_retreat"
                if expected_side_retreat
                else (
                    "vertical_lift_after_top_down_grasp"
                    if grasp_mode == "top_down"
                    else "vertical_lift_after_side_grasp"
                )
            ),
            "grasp_mode": grasp_mode,
            "motion_segment_names": segment_names,
            "side_grasp_plan_vertical_lift": self._config.side_grasp_plan_vertical_lift,
            "side_grasp_fallback_retreat": self._config.side_grasp_fallback_retreat,
            "side_grasp_retreat_to_pregrasp": self._config.side_grasp_retreat_to_pregrasp,
            "split_pregrasp_motion": self._config.split_pregrasp_motion,
        }
        return plan

    def plan_place(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        exporter = getattr(self._simulation, "export_current_curobo_place_inputs", None)
        if not callable(exporter):
            raise RuntimeError(
                "当前 simulation backend 不支持 current-state cuRobo place export，"
                "full-physics 模式禁止回退到离线 place plan。"
            )

        replan_dir = self._config.output_dir / f"place_replan_step_{state.step_index:06d}"
        replan_dir.mkdir(parents=True, exist_ok=True)
        place_pick_quat = (
            self._last_pick_grasp_quaternion_base
            if self._config.reuse_pick_grasp_orientation_for_place
            else None
        )
        export_report = exporter(
            output_dir=replan_dir,
            episode_spec=episode_spec,
            state=state,
            pick_grasp_quaternion_base=place_pick_quat,
        )
        plan_json = replan_dir / "place_plan.json"
        task = GraspTask(
            object_prim_path=str(export_report.get("object_prim_path") or ""),
            curobo_task_mode="place",
            use_planner_server=self._config.use_planner_server,
            state_json=str(export_report["state_json"]),
            target_json=str(export_report["target_json"]),
            plan_json=str(plan_json),
            result_json=str(replan_dir / "place_execution_result.json"),
        )

        started_at = time.time()
        payload = self._plan_runner(task)
        elapsed_s = time.time() - started_at
        if not isinstance(payload, dict):
            payload = load_curobo_plan_json(plan_json)
        plan = arm_plan_from_curobo_payload(
            payload,
            operation="place",
            source_path=plan_json,
        )
        plan.metadata["current_state_replan"] = {
            "enabled": True,
            "operation": "place",
            "state_json": str(task.state_json),
            "target_json": str(task.target_json),
            "plan_json": str(task.plan_json),
            "elapsed_wall_time_s": elapsed_s,
            "export_report": _json_safe(export_report),
        }
        plan.metadata["current_state_place_orientation"] = {
            "reuse_pick_grasp_orientation_for_place": (
                self._config.reuse_pick_grasp_orientation_for_place
            ),
            "stored_pick_grasp_quaternion_base": self._last_pick_grasp_quaternion_base,
            "exported_pick_grasp_quaternion_base": (
                None if place_pick_quat is None else list(place_pick_quat)
            ),
        }
        return plan


def run_curobo_plan_subprocess(
    task: GraspTask,
    *,
    workspace: Path,
    curobo_python: str,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """测试可替换的轻量 subprocess runner。默认路径使用 GraspPipeline。"""

    script_plan = workspace / "scripts/curobo/03_plan_grasp_trajectory.py"
    Path(task.plan_json).unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GO2_X5_WORKSPACE": str(workspace),
            "GO2_X5_CUROBO_TASK_MODE": task.curobo_task_mode or "grasp",
            "GO2_X5_STATE_JSON": task.state_json,
            "GO2_X5_TARGET_JSON": task.target_json,
            "GO2_X5_PLAN_JSON": task.plan_json,
            "GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT": os.environ.get(
                "GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT",
                "0",
            ),
            "GO2_X5_SIDE_GRASP_FALLBACK_RETREAT": os.environ.get(
                "GO2_X5_SIDE_GRASP_FALLBACK_RETREAT",
                "0",
            ),
            "GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP": os.environ.get(
                "GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP",
                "0",
            ),
            "GO2_X5_SPLIT_PREGRASP_MOTION": os.environ.get(
                "GO2_X5_SPLIT_PREGRASP_MOTION",
                "1",
            ),
        }
    )
    result = subprocess.run(
        [curobo_python, str(script_plan)],
        cwd=str(workspace),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"cuRobo pick replan failed with return code {result.returncode}")
    return load_curobo_plan_json(task.plan_json)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _motion_segment_names(payload: dict[str, Any]) -> tuple[str, ...]:
    """从 cuRobo payload 中读取 motion 段名，用于保护 full-physics pick 语义。"""

    segments = payload.get("segments")
    if not isinstance(segments, list | tuple):
        return ()
    return tuple(
        str(segment.get("name") or "")
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("type")) == "motion"
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


__all__ = [
    "ACTIVE_ARM_JOINT_NAMES",
    "CurrentStateCuroboPlanner",
    "CurrentStateCuroboPlannerConfig",
    "CurrentStateCuroboPickPlanner",
    "CurrentStateCuroboPickPlannerConfig",
    "build_arm_place_target_payload",
    "build_curobo_state_payload",
    "build_grasp_target_payload",
    "build_side_grasp_target_payload",
    "build_top_down_grasp_target_payload",
    "matrix_to_pose",
    "pose_dict_from_matrix",
    "pose_to_matrix",
    "write_json",
]

# 兼容旧测试和过渡引用；新 full-physics 路径使用无 Pick 后缀命名。
CurrentStateCuroboPickPlanner = CurrentStateCuroboPlanner
CurrentStateCuroboPickPlannerConfig = CurrentStateCuroboPlannerConfig
