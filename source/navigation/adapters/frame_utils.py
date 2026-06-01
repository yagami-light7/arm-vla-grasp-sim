"""Coordinate-frame helpers shared by navigation and grasp handoff code.

Navigation maps use ROS-style metadata: ``origin`` is the world pose of the
bottom-left map corner and image row zero is the top row. Quaternions use the
project-wide ``wxyz`` convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wrap_yaw(yaw: float) -> float:
    """Wrap a yaw angle to ``[-pi, pi)``."""

    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def map_local_to_world_xy(
    local_xy: Sequence[float],
    origin_xyyaw: Sequence[float],
) -> tuple[float, float]:
    """Transform an XY point from the map-local frame into the world frame."""

    local_x, local_y = (float(value) for value in local_xy)
    origin_x, origin_y, origin_yaw = (float(value) for value in origin_xyyaw)
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    return (
        origin_x + cos_yaw * local_x - sin_yaw * local_y,
        origin_y + sin_yaw * local_x + cos_yaw * local_y,
    )


def world_to_map_local_xy(
    world_xy: Sequence[float],
    origin_xyyaw: Sequence[float],
) -> tuple[float, float]:
    """Transform an XY point from the world frame into the map-local frame."""

    world_x, world_y = (float(value) for value in world_xy)
    origin_x, origin_y, origin_yaw = (float(value) for value in origin_xyyaw)
    delta_x = world_x - origin_x
    delta_y = world_y - origin_y
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    return (
        cos_yaw * delta_x + sin_yaw * delta_y,
        -sin_yaw * delta_x + cos_yaw * delta_y,
    )


def body_velocity_to_world(
    vx_body: float,
    vy_body: float,
    base_yaw: float,
) -> tuple[float, float]:
    """Rotate a planar body-frame velocity into the world frame."""

    return map_local_to_world_xy((vx_body, vy_body), (0.0, 0.0, base_yaw))


def world_velocity_to_body(
    vx_world: float,
    vy_world: float,
    base_yaw: float,
) -> tuple[float, float]:
    """Rotate a planar world-frame velocity into the body frame."""

    return world_to_map_local_xy((vx_world, vy_world), (0.0, 0.0, base_yaw))


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Convert a planar yaw to a ``wxyz`` quaternion."""

    half_yaw = 0.5 * float(yaw)
    return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


def pose_xyyaw_to_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """Build a standard 4x4 world transform from a planar base pose."""

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0, float(x)],
            [sin_yaw, cos_yaw, 0.0, float(y)],
            [0.0, 0.0, 1.0, float(z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def transform_point(transform: np.ndarray, point_xyz: Sequence[float]) -> np.ndarray:
    """Apply a standard 4x4 transform to a 3D point."""

    point = np.asarray([*point_xyz, 1.0], dtype=float)
    return (np.asarray(transform, dtype=float) @ point)[:3]


def world_point_to_base(transform_world_base: np.ndarray, point_world: Sequence[float]) -> np.ndarray:
    """Transform a world-frame point into a base frame."""

    return transform_point(np.linalg.inv(np.asarray(transform_world_base, dtype=float)), point_world)
