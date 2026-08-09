"""手工三维路径的纯数据校验与几何预处理。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence


_NAME_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLANAR_DIRECTION_EPSILON_M = 1.0e-9


@dataclass(frozen=True)
class PathPoint:
    """一个经过校验的地面路径点及其平面朝向。"""

    x: float
    y: float
    z: float
    yaw: float


def _finite_float(value: object, *, field_name: str) -> float:
    """把输入转为有限浮点数，同时拒绝布尔值和文本。"""

    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{field_name} 必须是有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有限数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数值")
    return result


def _validate_name_tokens(
    value: object,
    *,
    field_name: str,
    allow_leading_slash: bool,
) -> str:
    """校验本工具支持的简单 ROS topic 或 frame 名称。"""

    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError(f"{field_name} 不能为空且不能包含首尾空白")
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} 不能包含空白字符")
    if value.startswith("/"):
        if not allow_leading_slash:
            raise ValueError(f"{field_name} 不能以 / 开头")
        value_without_root = value[1:]
    else:
        value_without_root = value
    tokens = value_without_root.split("/")
    if not tokens or any(not _NAME_TOKEN.fullmatch(token) for token in tokens):
        raise ValueError(
            f"{field_name} 只能包含字母、数字、下划线和单层 / 分隔符，"
            "且每段必须以字母或下划线开头"
        )
    return value


def validate_topic_name(value: object) -> str:
    """返回经过校验的绝对或相对 ROS topic 名称。"""

    return _validate_name_tokens(
        value,
        field_name="topic",
        allow_leading_slash=True,
    )


def validate_frame_id(value: object) -> str:
    """返回经过校验且不含前导斜杠的 ROS frame 名称。"""

    return _validate_name_tokens(
        value,
        field_name="frame_id",
        allow_leading_slash=False,
    )


def _point_yaw(
    coordinates: Sequence[tuple[float, float, float]],
    index: int,
    fallback_yaw: float,
) -> float:
    """从最近的有效平面路径段计算当前点的 yaw。"""

    current_x, current_y, _ = coordinates[index]
    for candidate_index in range(index + 1, len(coordinates)):
        next_x, next_y, _ = coordinates[candidate_index]
        delta_x = next_x - current_x
        delta_y = next_y - current_y
        if math.hypot(delta_x, delta_y) > _PLANAR_DIRECTION_EPSILON_M:
            return math.atan2(delta_y, delta_x)
    for candidate_index in range(index - 1, -1, -1):
        previous_x, previous_y, _ = coordinates[candidate_index]
        delta_x = current_x - previous_x
        delta_y = current_y - previous_y
        if math.hypot(delta_x, delta_y) > _PLANAR_DIRECTION_EPSILON_M:
            return math.atan2(delta_y, delta_x)
    return fallback_yaw


def prepare_path_points(
    points_xyz: Sequence[object],
    *,
    min_point_distance_m: object,
) -> tuple[PathPoint, ...]:
    """校验展平点列、移除相邻近重复点并计算平面朝向。

    ``z`` 原样保留为地面高度；本函数不会增加机器人 ``body_height``。
    """

    if isinstance(points_xyz, (str, bytes)):
        raise ValueError("points_xyz 必须是展平的数值数组")
    try:
        raw_values = tuple(points_xyz)
    except TypeError as exc:
        raise ValueError("points_xyz 必须是展平的数值数组") from exc
    if len(raw_values) < 6 or len(raw_values) % 3 != 0:
        raise ValueError("points_xyz 必须包含至少 2 个完整的 xyz 点")

    minimum_distance = _finite_float(
        min_point_distance_m,
        field_name="min_point_distance_m",
    )
    if minimum_distance < 0.0:
        raise ValueError("min_point_distance_m 不能为负数")

    coordinates: list[tuple[float, float, float]] = []
    for point_index in range(len(raw_values) // 3):
        start = point_index * 3
        point = (
            _finite_float(
                raw_values[start],
                field_name=f"points_xyz[{start}]",
            ),
            _finite_float(
                raw_values[start + 1],
                field_name=f"points_xyz[{start + 1}]",
            ),
            _finite_float(
                raw_values[start + 2],
                field_name=f"points_xyz[{start + 2}]",
            ),
        )
        if coordinates and math.dist(coordinates[-1], point) <= minimum_distance:
            continue
        coordinates.append(point)

    if len(coordinates) < 2:
        raise ValueError("去除相邻近重复点后，路径必须仍包含至少 2 个点")

    points: list[PathPoint] = []
    previous_yaw = 0.0
    for index, (x, y, z) in enumerate(coordinates):
        yaw = _point_yaw(coordinates, index, previous_yaw)
        points.append(PathPoint(x=x, y=y, z=z, yaw=yaw))
        previous_yaw = yaw
    return tuple(points)
