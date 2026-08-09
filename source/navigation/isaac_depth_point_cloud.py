"""把 Isaac Lab 前视深度相机数据转换为世界系导航点云。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DepthPointCloudConfig:
    """定义深度图抽样、量程和发布节拍。"""

    sensor_name: str = "head_camera"
    depth_key: str = "distance_to_image_plane"
    environment_index: int = 0
    pixel_stride: int = 4
    min_depth_m: float = 0.15
    max_depth_m: float = 8.0
    max_points: int = 30_000
    minimum_valid_points: int = 64
    publish_interval_control_steps: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_name, str) or not self.sensor_name:
            raise ValueError("sensor_name 必须是非空字符串。")
        if not isinstance(self.depth_key, str) or not self.depth_key:
            raise ValueError("depth_key 必须是非空字符串。")
        _validate_nonnegative_integer(self.environment_index, "environment_index")
        _validate_positive_integer(self.pixel_stride, "pixel_stride")
        _validate_positive_integer(self.max_points, "max_points")
        _validate_positive_integer(
            self.minimum_valid_points,
            "minimum_valid_points",
        )
        if self.minimum_valid_points > self.max_points:
            raise ValueError("minimum_valid_points 不能大于 max_points。")
        _validate_positive_integer(
            self.publish_interval_control_steps,
            "publish_interval_control_steps",
        )
        min_depth = _finite_float(self.min_depth_m, "min_depth_m")
        max_depth = _finite_float(self.max_depth_m, "max_depth_m")
        if min_depth < 0.0:
            raise ValueError("min_depth_m 不能小于零。")
        if max_depth <= min_depth:
            raise ValueError("max_depth_m 必须大于 min_depth_m。")


def depth_image_to_world_points(
    depth_image: Any,
    intrinsic_matrix: Any,
    camera_position_world: Any,
    camera_orientation_world_wxyz: Any,
    config: DepthPointCloudConfig | None = None,
) -> np.ndarray:
    """把 ROS 光学坐标约定的平面深度反投影到世界坐标。

    Isaac Lab 的 ``distance_to_image_plane`` 沿相机光学 ``+Z`` 轴测距，
    ``CameraData.quat_w_ros`` 则给出同一光学坐标系到世界系的 WXYZ 旋转。
    本函数只保留配置量程内的有限点，并在反投影前按原始像素坐标抽样。
    """

    cfg = config or DepthPointCloudConfig()
    depth = _depth_array(depth_image)
    intrinsics = _matrix_array(intrinsic_matrix, (3, 3), "intrinsic_matrix")
    position = _vector_array(camera_position_world, 3, "camera_position_world")
    orientation = _normalized_quaternion_wxyz(
        camera_orientation_world_wxyz,
        "camera_orientation_world_wxyz",
    )

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsic_matrix 的 fx、fy 必须为正数。")

    stride = cfg.pixel_stride
    sampled_depth = depth[::stride, ::stride]
    rows = np.arange(0, depth.shape[0], stride, dtype=np.float64)
    columns = np.arange(0, depth.shape[1], stride, dtype=np.float64)
    pixel_u, pixel_v = np.meshgrid(columns, rows, indexing="xy")
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= cfg.min_depth_m)
        & (sampled_depth <= cfg.max_depth_m)
    )
    if not bool(np.any(valid)):
        return np.empty((0, 3), dtype=np.float32)

    distance = sampled_depth[valid]
    camera_points = np.column_stack(
        (
            (pixel_u[valid] - cx) * distance / fx,
            (pixel_v[valid] - cy) * distance / fy,
            distance,
        )
    )
    world_points = _rotate_points_wxyz(camera_points, orientation)
    world_points += position
    world_points = world_points[np.isfinite(world_points).all(axis=1)]
    if world_points.shape[0] > cfg.max_points:
        # 等距选点保持确定性，避免每帧随机采样给局部占据图引入闪烁。
        indices = np.linspace(
            0,
            world_points.shape[0] - 1,
            cfg.max_points,
            dtype=np.int64,
        )
        world_points = world_points[indices]
    return np.array(world_points, dtype=np.float32, order="C", copy=True)


def camera_sensor_to_world_points(
    sensor: Any,
    config: DepthPointCloudConfig | None = None,
) -> np.ndarray:
    """从 Isaac Lab ``Camera`` 的指定环境读取一帧世界系点云。"""

    cfg = config or DepthPointCloudConfig()
    try:
        data = sensor.data
        output = data.output
        depth_batch = output[cfg.depth_key]
        intrinsic_batch = data.intrinsic_matrices
        position_batch = data.pos_w
        orientation_batch = data.quat_w_ros
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"相机 {cfg.sensor_name!r} 缺少深度、内参或最新世界位姿。"
        ) from exc

    index = cfg.environment_index
    depth = _batch_item(depth_batch, index, cfg.depth_key)
    intrinsics = _batch_item(intrinsic_batch, index, "intrinsic_matrices")
    position = _batch_item(position_batch, index, "pos_w")
    orientation = _batch_item(orientation_batch, index, "quat_w_ros")
    conversion_config = cfg
    if cfg.pixel_stride > 1:
        # 先在 CUDA 张量上抽样，再复制约 1/stride² 的数据到 CPU。抽样后的
        # 像素坐标缩小同样倍数，因此 K 的前两行必须同步缩放。
        depth = depth[:: cfg.pixel_stride, :: cfg.pixel_stride, ...]
        intrinsics = np.array(_to_numpy(intrinsics), dtype=np.float64, copy=True)
        intrinsics[0, :] /= cfg.pixel_stride
        intrinsics[1, :] /= cfg.pixel_stride
        conversion_config = replace(cfg, pixel_stride=1)
    return depth_image_to_world_points(
        depth,
        intrinsics,
        position,
        orientation,
        conversion_config,
    )


def _batch_item(values: Any, index: int, field_name: str) -> Any:
    """读取批量传感器张量中的一个环境，并提供明确越界错误。"""

    try:
        return values[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{field_name} 不包含 environment_index={index}。"
        ) from exc


def _to_numpy(value: Any) -> np.ndarray:
    """把 NumPy 或 Torch 风格张量复制到 CPU NumPy。"""

    converted = value
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    numpy = getattr(converted, "numpy", None)
    if callable(numpy):
        converted = numpy()
    return np.asarray(converted)


def _depth_array(value: Any) -> np.ndarray:
    """接受 H×W 或 H×W×1 深度，并转换为 float64 计算数组。"""

    depth = _to_numpy(value)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or 0 in depth.shape:
        raise ValueError("depth_image 必须是非空 H×W 或 H×W×1 数组。")
    if not np.issubdtype(depth.dtype, np.number):
        raise TypeError("depth_image 必须使用数值 dtype。")
    return np.asarray(depth, dtype=np.float64)


def _matrix_array(value: Any, shape: tuple[int, int], field_name: str) -> np.ndarray:
    """校验有限矩阵。"""

    matrix = _to_numpy(value)
    if matrix.shape != shape:
        raise ValueError(f"{field_name} 必须是形状 {shape} 的矩阵。")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{field_name} 必须使用数值 dtype。")
    result = np.asarray(matrix, dtype=np.float64)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{field_name} 不能包含 NaN 或无穷值。")
    return result


def _vector_array(value: Any, size: int, field_name: str) -> np.ndarray:
    """校验一维有限向量。"""

    vector = _to_numpy(value)
    if vector.shape != (size,):
        raise ValueError(f"{field_name} 必须包含 {size} 个元素。")
    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError(f"{field_name} 必须使用数值 dtype。")
    result = np.asarray(vector, dtype=np.float64)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{field_name} 不能包含 NaN 或无穷值。")
    return result


def _normalized_quaternion_wxyz(value: Any, field_name: str) -> np.ndarray:
    """校验并规范化 WXYZ 四元数。"""

    quaternion = _vector_array(value, 4, field_name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError(f"{field_name} 不能是零四元数。")
    return quaternion / norm


def _rotate_points_wxyz(points: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """使用 WXYZ 四元数批量旋转 N×3 点。"""

    w = float(quaternion[0])
    vector = quaternion[1:]
    cross = 2.0 * np.cross(np.broadcast_to(vector, points.shape), points)
    return points + w * cross + np.cross(
        np.broadcast_to(vector, points.shape),
        cross,
    )


def _finite_float(value: Any, field_name: str) -> float:
    """转换单个有限浮点配置。"""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是数值。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} 必须是数值。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限值。")
    return result


def _validate_positive_integer(value: Any, field_name: str) -> None:
    """校验正整数配置。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} 必须是正整数。")


def _validate_nonnegative_integer(value: Any, field_name: str) -> None:
    """校验非负整数配置。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数。")


__all__ = [
    "DepthPointCloudConfig",
    "camera_sensor_to_world_points",
    "depth_image_to_world_points",
]
