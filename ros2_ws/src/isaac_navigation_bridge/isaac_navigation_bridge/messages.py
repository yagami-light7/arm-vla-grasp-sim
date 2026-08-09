"""Odometry 与 PointCloud2 的纯消息转换工具。"""

from __future__ import annotations

import copy
import math
from typing import Iterable

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry, Path
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from .geometry import (
    LocalGroundPathSegments,
    OrderedGroundPath,
    PointCloudFilterConfig,
    classify_ray_endpoints_xyz,
    finite_xyz,
)


RAY_ENDPOINT_TYPE_FIELD = "ray_endpoint_type"


def normalize_frame_id(value: str, *, field_name: str) -> str:
    """去掉 ROS 1 风格前导斜杠并校验 frame 名称。"""

    normalized = str(value).strip().lstrip("/")
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError(f"{field_name} 必须是非空且不含空白的 frame_id")
    return normalized


def matching_frame_id(
    actual: str,
    expected: str,
    *,
    field_name: str,
) -> str:
    """校验输入 frame 与配置合同一致，并返回规范化名称。"""

    actual_normalized = normalize_frame_id(actual, field_name=field_name)
    expected_normalized = normalize_frame_id(
        expected,
        field_name=f"预期 {field_name}",
    )
    if actual_normalized != expected_normalized:
        raise ValueError(
            f"{field_name} 不匹配：收到 {actual_normalized!r}，"
            f"预期 {expected_normalized!r}"
        )
    return actual_normalized


def stamp_is_valid(stamp: Time) -> bool:
    """判断时间戳是否为合法的非零 ROS 时间。"""

    return (
        int(stamp.sec) >= 0
        and 0 <= int(stamp.nanosec) < 1_000_000_000
        and (int(stamp.sec) > 0 or int(stamp.nanosec) > 0)
    )


def normalized_stamp(input_stamp: Time, fallback_stamp: Time) -> Time:
    """优先保留输入时间戳，否则使用节点时钟的非零时间戳。"""

    if stamp_is_valid(input_stamp):
        return copy.deepcopy(input_stamp)
    if stamp_is_valid(fallback_stamp):
        return copy.deepcopy(fallback_stamp)
    raise ValueError("输入时间戳和节点时钟均为零或非法")


def stamp_to_nanoseconds(stamp: Time) -> int:
    """把合法 ROS 时间戳转换为纳秒整数。"""

    if not stamp_is_valid(stamp):
        raise ValueError("时间戳必须是合法的非零 ROS 时间")
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _finite_or_zero(value: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def _sanitize_covariance(values: Iterable[float]) -> list[float]:
    return [_finite_or_zero(value) for value in values]


def _normalize_quaternion_xyzw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Odometry 四元数包含 NaN 或 Inf")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise ValueError("Odometry 四元数不能为零")
    return tuple(value / norm for value in values)


def normalize_odometry(
    message: Odometry,
    *,
    fallback_stamp: Time,
    frame_id: str,
    child_frame_id: str,
) -> Odometry:
    """复制并规范化里程计；非法位置会被拒绝而不是静默归零。"""

    normalized_frame = matching_frame_id(
        message.header.frame_id,
        frame_id,
        field_name="Odometry header.frame_id",
    )
    normalized_child_frame = matching_frame_id(
        message.child_frame_id,
        child_frame_id,
        field_name="Odometry child_frame_id",
    )
    result = copy.deepcopy(message)
    position = result.pose.pose.position
    position_values = (
        float(position.x),
        float(position.y),
        float(position.z),
    )
    if not all(math.isfinite(value) for value in position_values):
        raise ValueError("Odometry 位置包含 NaN 或 Inf")

    orientation = result.pose.pose.orientation
    (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ) = _normalize_quaternion_xyzw(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )

    linear = result.twist.twist.linear
    angular = result.twist.twist.angular
    linear.x, linear.y, linear.z = (
        _finite_or_zero(linear.x),
        _finite_or_zero(linear.y),
        _finite_or_zero(linear.z),
    )
    angular.x, angular.y, angular.z = (
        _finite_or_zero(angular.x),
        _finite_or_zero(angular.y),
        _finite_or_zero(angular.z),
    )
    result.pose.covariance = _sanitize_covariance(result.pose.covariance)
    result.twist.covariance = _sanitize_covariance(result.twist.covariance)
    result.header.stamp = normalized_stamp(message.header.stamp, fallback_stamp)
    result.header.frame_id = normalized_frame
    result.child_frame_id = normalized_child_frame
    return result


def base_pose_from_odometry(
    message: Odometry,
) -> tuple[tuple[float, float, float], float]:
    """从已规范化 Odometry 提取世界位置和底盘 yaw。"""

    position, orientation = base_transform_from_odometry(message)
    x, y, z, w = orientation
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return position, float(yaw)


def base_transform_from_odometry(
    message: Odometry,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """从已规范化 Odometry 提取世界位置和完整 xyzw 姿态。"""

    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    normalized_orientation = _normalize_quaternion_xyzw(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    return (
        (float(position.x), float(position.y), float(position.z)),
        normalized_orientation,
    )


def ordered_ground_path_from_message(
    message: Path,
    *,
    frame_id: str,
    min_point_spacing_m: float,
) -> OrderedGroundPath:
    """按 SCAN 输入合同校验 Path，并移除连续近重复点。

    顶层和每个 Pose 都必须具有非零时间戳；Pose frame 可按 ROS 惯例留空，
    非空时必须匹配世界系。姿态只参与合法性检查，不影响地面高度插值，但仍
    会执行有限性、非零范数与单位化检查。
    """

    matching_frame_id(
        message.header.frame_id,
        frame_id,
        field_name="Path header.frame_id",
    )
    if not stamp_is_valid(message.header.stamp):
        raise ValueError("Path header.stamp 必须是合法非零时间")
    spacing = float(min_point_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Path min_point_spacing_m 必须是有限正数")
    if len(message.poses) < 2:
        raise ValueError("Path 至少需要 2 个 Pose")

    accepted: list[tuple[float, float, float]] = []
    for index, pose_stamped in enumerate(message.poses):
        if pose_stamped.header.frame_id:
            matching_frame_id(
                pose_stamped.header.frame_id,
                frame_id,
                field_name=f"Path poses[{index}].header.frame_id",
            )
        if not stamp_is_valid(pose_stamped.header.stamp):
            raise ValueError(
                f"Path poses[{index}].header.stamp 必须是合法非零时间"
            )

        position = pose_stamped.pose.position
        point = (
            float(position.x),
            float(position.y),
            float(position.z),
        )
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Path poses[{index}] 位置包含 NaN 或 Inf")

        orientation = pose_stamped.pose.orientation
        orientation_values = (
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        orientation_norm = math.sqrt(
            sum(value * value for value in orientation_values)
        )
        if not all(math.isfinite(value) for value in orientation_values):
            raise ValueError(f"Path poses[{index}] 四元数包含 NaN 或 Inf")
        if orientation_norm <= 1.0e-6:
            raise ValueError(f"Path poses[{index}] 四元数范数无效")
        # 调用单位化工具，确保 Path 与 Odometry 使用同一姿态规则。
        _normalize_quaternion_xyzw(*orientation_values)

        if accepted:
            distance = math.sqrt(
                sum(
                    (current - previous) ** 2
                    for current, previous in zip(point, accepted[-1])
                )
            )
            if distance < spacing:
                continue
        accepted.append(point)

    if len(accepted) < 2:
        raise ValueError("Path 按最小点间距去重后少于 2 个点")
    return OrderedGroundPath(accepted)


def point_cloud_xyz_array(message: PointCloud2) -> np.ndarray:
    """读取任意字段布局的 PointCloud2，并只返回 xyz。"""

    available_fields = {field.name for field in message.fields}
    if not {"x", "y", "z"}.issubset(available_fields):
        raise ValueError("PointCloud2 必须包含 x、y、z 字段")
    try:
        structured = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z"],
            skip_nans=False,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"无法解析 PointCloud2：{exc}") from exc
    return np.column_stack(
        (structured["x"], structured["y"], structured["z"])
    )


def convert_point_cloud(
    message: PointCloud2,
    *,
    fallback_stamp: Time,
    frame_id: str,
    base_position_world_xyz: Iterable[float] | None,
    base_yaw_rad: float | None,
    filter_config: PointCloudFilterConfig,
    base_orientation_world_xyzw: Iterable[float] | None = None,
    local_ground_path_segments: LocalGroundPathSegments | None = None,
    minimum_valid_input_points: int = 1,
) -> PointCloud2:
    """输出只含 float32 xyz 的非组织点云。

    当 base pose 为 ``None`` 时，仅移除非有限点；该模式只供显式关闭
    ``drop_cloud_without_odom`` 的诊断场景使用。
    """

    normalized_frame = matching_frame_id(
        message.header.frame_id,
        frame_id,
        field_name="PointCloud2 header.frame_id",
    )
    if (
        isinstance(minimum_valid_input_points, bool)
        or not isinstance(minimum_valid_input_points, int)
        or minimum_valid_input_points < 1
    ):
        raise ValueError("minimum_valid_input_points 必须是正整数")
    points = finite_xyz(point_cloud_xyz_array(message))
    if points.shape[0] < minimum_valid_input_points:
        raise ValueError(
            "原始 PointCloud2 的有限 xyz 点数不足："
            f"{points.shape[0]} < {minimum_valid_input_points}"
        )
    classified_endpoint_types: np.ndarray | None = None
    if base_position_world_xyz is None or (
        base_yaw_rad is None and base_orientation_world_xyzw is None
    ):
        filtered = points
    else:
        classified = classify_ray_endpoints_xyz(
            points,
            base_position_world_xyz=base_position_world_xyz,
            base_yaw_rad=base_yaw_rad,
            config=filter_config,
            base_orientation_world_xyzw=base_orientation_world_xyzw,
            local_ground_path_segments=local_ground_path_segments,
        )
        filtered = classified.points_xyz
        classified_endpoint_types = classified.endpoint_types

    header = Header()
    header.stamp = normalized_stamp(message.header.stamp, fallback_stamp)
    header.frame_id = normalized_frame
    if classified_endpoint_types is None or filtered.shape[0] == 0:
        # 零点帧必须继续使用 GridMap/controller 认证的
        # canonical xyz32 布局，不得附加字段。
        output = point_cloud2.create_cloud_xyz32(header, filtered)
    else:
        fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name=RAY_ENDPOINT_TYPE_FIELD,
                offset=12,
                datatype=PointField.UINT8,
                count=1,
            ),
        ]
        records = (
            (
                float(point[0]),
                float(point[1]),
                float(point[2]),
                int(endpoint_type),
            )
            for point, endpoint_type in zip(
                filtered,
                classified_endpoint_types,
                strict=True,
            )
        )
        output = point_cloud2.create_cloud(header, fields, records)
    output.is_dense = True
    return output
