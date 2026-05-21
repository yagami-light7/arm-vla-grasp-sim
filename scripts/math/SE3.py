"""
SE(3) 位姿工具函数。

本项目统一使用下面两个约定：
    1. 四元数顺序是 wxyz。
    2. 4x4 齐次变换矩阵使用标准列向量约定，平移在最后一列：

    T = [[R00, R01, R02, tx],
            [R10, R11, R12, ty],
            [R20, R21, R22, tz],
            [0,   0,   0,   1 ]]

这些函数只依赖 numpy，可以同时被普通 Python 脚本和 Isaac Sim Script Editor 使用。
"""

from __future__ import annotations

import math

import numpy as np


def normalize_quat_wxyz(quat) -> np.ndarray:
    """归一化 wxyz 四元数。"""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError(f"四元数范数太小: {quat}")
    return quat / norm


def normalize_quat(quat) -> np.ndarray:
    """normalize_quat_wxyz 的短别名，便于旧代码迁移。"""
    return normalize_quat_wxyz(quat)


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    """wxyz 四元数转 3x3 旋转矩阵。"""
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(rotation) -> np.ndarray:
    """3x3 旋转矩阵转 wxyz 四元数。"""
    rotation = np.asarray(rotation, dtype=float)
    trace = np.trace(rotation)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag = np.diag(rotation)
        index = int(np.argmax(diag))

        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    return normalize_quat_wxyz([w, x, y, z])


def rpy_to_rotmat(rpy_xyz) -> np.ndarray:
    """URDF rpy 转 3x3 旋转矩阵，旋转顺序为 Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = [float(value) for value in rpy_xyz]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return rot_z @ rot_y @ rot_x


def pose_to_matrix(position_xyz, quat_wxyz) -> np.ndarray:
    """position + wxyz quaternion 转标准 4x4 SE(3) 矩阵。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    transform[:3, 3] = np.asarray(position_xyz, dtype=float)
    return transform


def xyz_rpy_to_matrix(xyz, rpy) -> np.ndarray:
    """xyz + URDF rpy 转标准 4x4 SE(3) 矩阵。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_rotmat(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def matrix_to_pose(transform) -> tuple[np.ndarray, np.ndarray]:
    """标准 4x4 SE(3) 矩阵转 position + wxyz quaternion。"""
    transform = np.asarray(transform, dtype=float)
    position = transform[:3, 3].copy()
    quaternion = rotmat_to_quat_wxyz(transform[:3, :3])
    return position, quaternion


def pose_dict_from_matrix(transform) -> dict:
    """把标准 4x4 SE(3) 矩阵转成 JSON 友好字段。"""
    position, quaternion = matrix_to_pose(transform)
    return {
        "position_xyz": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "matrix_4x4": np.asarray(transform, dtype=float).tolist(),
        "matrix_convention": "standard_SE3_translation_last_column",
    }


def quat_angle_error_deg(q_a, q_b) -> float:
    """
    计算两个 wxyz 四元数的旋转夹角误差，单位 degree。

    q 和 -q 表示同一个旋转，所以使用 abs(dot)。
    """
    q_a = normalize_quat_wxyz(q_a)
    q_b = normalize_quat_wxyz(q_b)
    dot = abs(float(np.dot(q_a, q_b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def quat_error_deg(q_a, q_b) -> float:
    """quat_angle_error_deg 的短别名。"""
    return quat_angle_error_deg(q_a, q_b)
