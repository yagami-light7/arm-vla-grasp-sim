"""PCT 地面路径的纯几何校验与朝向生成。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class GroundPathPoint:
    """一个世界系地面点及其平面 yaw。"""

    x: float
    y: float
    z: float
    yaw: float


def normalize_frame_id(value: object, *, field_name: str) -> str:
    """规范化 ROS frame，并拒绝空白和复合非法名称。"""

    normalized = str(value).strip().lstrip("/")
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError(f"{field_name} 必须是非空且不含空白的 frame_id")
    components = normalized.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{field_name} 不能包含空层级或相对路径")
    return normalized


def quaternion_xyzw_to_yaw(values: Sequence[float]) -> float:
    """校验并单位化 xyzw 四元数，返回 yaw。"""

    if len(values) != 4:
        raise ValueError("四元数必须包含 4 个分量")
    x, y, z, w = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise ValueError("四元数不能包含 NaN 或 Inf")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("四元数不能为零")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def prepare_ground_path(
    points_xyz: Sequence[Sequence[float]],
    *,
    terminal_yaw: float,
    minimum_point_spacing_m: float,
) -> tuple[GroundPathPoint, ...]:
    """移除连续近重复点，并让最后一个 Pose 精确使用目标 yaw。

    输入和输出的 ``z`` 都表示 collision PLY 的真实地面高度；本函数不会增加
    ``body_height``。
    """

    spacing = float(minimum_point_spacing_m)
    final_yaw = float(terminal_yaw)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("minimum_point_spacing_m 必须是有限正数")
    if not math.isfinite(final_yaw):
        raise ValueError("terminal_yaw 必须是有限数值")
    raw_coordinates: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(points_xyz):
        if len(raw_point) != 3:
            raise ValueError(f"points_xyz[{index}] 必须包含 3 个坐标")
        point = tuple(float(value) for value in raw_point)
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"points_xyz[{index}] 不能包含 NaN 或 Inf")
        raw_coordinates.append(point)
    if len(raw_coordinates) < 2:
        raise ValueError("PCT Path 必须包含至少 2 个点")
    if math.dist(raw_coordinates[0], raw_coordinates[-1]) <= spacing:
        raise ValueError(
            "PCT 起终点距离不大于 minimum_point_spacing_m，"
            "不能发布会被 SCAN 去重为单点的成功 Path"
        )

    coordinates: list[tuple[float, float, float]] = [raw_coordinates[0]]
    for point in raw_coordinates[1:-1]:
        if math.dist(coordinates[-1], point) <= spacing:
            continue
        coordinates.append(point)
    terminal = raw_coordinates[-1]
    if math.dist(coordinates[-1], terminal) <= spacing:
        if len(coordinates) > 1:
            # 去重不得丢掉 backend 认证的精确请求终点。
            coordinates[-1] = terminal
        elif math.dist(coordinates[0], terminal) > 1.0e-12:
            coordinates.append(terminal)
    else:
        coordinates.append(terminal)
    if len(coordinates) < 2:
        raise ValueError("PCT Path 去重后必须仍包含至少 2 个点")

    points: list[GroundPathPoint] = []
    previous_yaw = final_yaw
    for index, (x, y, z) in enumerate(coordinates):
        yaw = (
            final_yaw
            if index == len(coordinates) - 1
            else _forward_segment_yaw(coordinates, index, previous_yaw)
        )
        points.append(GroundPathPoint(x=x, y=y, z=z, yaw=yaw))
        previous_yaw = yaw
    return tuple(points)


def _forward_segment_yaw(
    points: Sequence[tuple[float, float, float]],
    index: int,
    fallback_yaw: float,
) -> float:
    current_x, current_y, _ = points[index]
    for next_index in range(index + 1, len(points)):
        next_x, next_y, _ = points[next_index]
        delta_x = next_x - current_x
        delta_y = next_y - current_y
        if math.hypot(delta_x, delta_y) > 1.0e-9:
            return math.atan2(delta_y, delta_x)
    return fallback_yaw
