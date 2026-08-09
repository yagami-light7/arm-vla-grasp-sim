"""不依赖 ROS executor 的点云几何过滤。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _finite_triplet(name: str, values: Iterable[float]) -> tuple[float, float, float]:
    """把输入规范化为有限三元组。"""

    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 必须是 3 个有限数值")
    return result


def _normalized_quaternion_xyzw(
    values: Iterable[float],
) -> tuple[float, float, float, float]:
    """把输入规范化为有限单位四元数。"""

    result = tuple(float(value) for value in values)
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError("base_orientation_world_xyzw 必须是 4 个有限数值")
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-12:
        raise ValueError("base_orientation_world_xyzw 不能是零四元数")
    return tuple(value / norm for value in result)


def _body_to_world_rotation_xyzw(
    orientation_xyzw: Iterable[float],
) -> np.ndarray:
    """返回 ROS xyzw 四元数对应的机体系到世界系旋转矩阵。"""

    x, y, z, w = _normalized_quaternion_xyzw(orientation_xyzw)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PointCloudFilterConfig:
    """点云裁剪、地面过滤和双圆柱自点过滤参数。"""

    range_min_m: float = 0.0
    range_max_m: float = 8.0
    crop_min_xyz_m: tuple[float, float, float] = (-5.0, -5.0, -0.60)
    crop_max_xyz_m: tuple[float, float, float] = (5.0, 5.0, 2.00)
    filter_ground: bool = True
    body_height_m: float = 0.30
    ground_clearance_m: float = 0.03
    ground_band_down_m: float = 0.03
    filter_path_ground: bool = True
    path_ground_corridor_radius_m: float = 0.70
    path_ground_clearance_m: float = 0.05
    path_ground_stair_minimum_slope: float = 0.20
    path_ground_stair_clearance_m: float = 0.09
    path_ground_band_down_m: float = 0.05
    path_ground_stair_band_down_m: float = 0.09
    path_min_point_spacing_m: float = 0.05
    path_ground_backward_arc_m: float = 1.0
    path_ground_forward_arc_m: float = 3.0
    filter_self: bool = True
    double_cylinder_radius_m: float = 0.27
    double_cylinder_offset_m: float = 0.16
    self_z_min_m: float = -0.40
    self_z_max_m: float = 0.50

    def __post_init__(self) -> None:
        crop_min = _finite_triplet("crop_min_xyz_m", self.crop_min_xyz_m)
        crop_max = _finite_triplet("crop_max_xyz_m", self.crop_max_xyz_m)
        object.__setattr__(self, "crop_min_xyz_m", crop_min)
        object.__setattr__(self, "crop_max_xyz_m", crop_max)

        scalar_names = (
            "range_min_m",
            "range_max_m",
            "body_height_m",
            "ground_clearance_m",
            "ground_band_down_m",
            "path_ground_corridor_radius_m",
            "path_ground_clearance_m",
            "path_ground_stair_minimum_slope",
            "path_ground_stair_clearance_m",
            "path_ground_band_down_m",
            "path_ground_stair_band_down_m",
            "path_min_point_spacing_m",
            "path_ground_backward_arc_m",
            "path_ground_forward_arc_m",
            "double_cylinder_radius_m",
            "double_cylinder_offset_m",
            "self_z_min_m",
            "self_z_max_m",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} 必须是有限数值")
            object.__setattr__(self, name, value)

        if self.range_min_m < 0.0 or self.range_max_m <= self.range_min_m:
            raise ValueError("点云距离范围必须满足 0 <= min < max")
        if any(lower >= upper for lower, upper in zip(crop_min, crop_max)):
            raise ValueError("点云轴向裁剪下界必须小于上界")
        if self.body_height_m <= 0.0:
            raise ValueError("body_height_m 必须为正数")
        if self.ground_clearance_m < 0.0:
            raise ValueError("ground_clearance_m 不能为负数")
        if self.ground_band_down_m < 0.0:
            raise ValueError("ground_band_down_m 不能为负数")
        if self.path_ground_corridor_radius_m < 0.0:
            raise ValueError("path_ground_corridor_radius_m 不能为负数")
        if self.path_ground_clearance_m < 0.0:
            raise ValueError("path_ground_clearance_m 不能为负数")
        if self.path_ground_stair_minimum_slope <= 0.0:
            raise ValueError("path_ground_stair_minimum_slope 必须为正数")
        if self.path_ground_stair_clearance_m < self.path_ground_clearance_m:
            raise ValueError(
                "path_ground_stair_clearance_m 不得小于常规 Path 间隙"
            )
        if self.path_ground_band_down_m < 0.0:
            raise ValueError("path_ground_band_down_m 不能为负数")
        if self.path_ground_stair_band_down_m < self.path_ground_band_down_m:
            raise ValueError(
                "path_ground_stair_band_down_m 不得小于常规 Path 下方窄带"
            )
        if self.path_min_point_spacing_m <= 0.0:
            raise ValueError("path_min_point_spacing_m 必须为正数")
        if self.path_ground_backward_arc_m < 0.0:
            raise ValueError("path_ground_backward_arc_m 不能为负数")
        if self.path_ground_forward_arc_m <= 0.0:
            raise ValueError("path_ground_forward_arc_m 必须为正数")
        if self.double_cylinder_radius_m < 0.0:
            raise ValueError("double_cylinder_radius_m 不能为负数")
        if self.double_cylinder_offset_m < 0.0:
            raise ValueError("double_cylinder_offset_m 不能为负数")
        if self.self_z_min_m >= self.self_z_max_m:
            raise ValueError("自点过滤竖直下界必须小于上界")


RAY_ENDPOINT_FREE = 0
RAY_ENDPOINT_OCCUPIED = 1


@dataclass(frozen=True)
class RayEndpointClassification:
    """保留顺序的占据端点与显式自由端点。"""

    points_xyz: np.ndarray
    endpoint_types: np.ndarray


def finite_xyz(points_xyz: object) -> np.ndarray:
    """返回有限、连续的 N×3 float32 点数组。"""

    points = np.asarray(points_xyz)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz 必须是 N×3 数组")
    try:
        numeric = points.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("points_xyz 必须只包含数值") from exc
    numeric = numeric[np.isfinite(numeric).all(axis=1)]
    return np.ascontiguousarray(numeric, dtype=np.float32)


@dataclass(frozen=True)
class LocalGroundPathSegments:
    """参考路径局部弧长窗口内预计算的 XY 线段。"""

    start_xy: np.ndarray
    delta_xy: np.ndarray
    inverse_length_squared_xy: np.ndarray
    start_z: np.ndarray
    delta_z: np.ndarray
    progress_start_m: float
    progress_end_m: float

    @property
    def segment_count(self) -> int:
        """返回可用于支撑面插值的非竖直线段数。"""

        return int(self.start_z.shape[0])


class OrderedGroundPath:
    """按输入顺序保存地面 Path，并预计算三维弧长与线段。"""

    def __init__(self, points_world_xyz: object) -> None:
        points = np.asarray(points_world_xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("有序地面 Path 必须是 N×3 数组")
        try:
            points = points.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("有序地面 Path 必须只包含数值") from exc
        if not np.isfinite(points).all():
            raise ValueError("有序地面 Path 不能包含 NaN 或 Inf")
        if points.shape[0] < 2:
            raise ValueError("有序地面 Path 至少需要 2 个有限点")
        self._points = np.ascontiguousarray(points, dtype=np.float64)
        self._segment_delta_xyz = np.diff(self._points, axis=0)
        self._segment_length_m = np.linalg.norm(
            self._segment_delta_xyz,
            axis=1,
        )
        self._segment_start_progress_m = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(self._segment_length_m[:-1]),
            )
        )
        self._segment_end_progress_m = (
            self._segment_start_progress_m + self._segment_length_m
        )
        self._total_length_m = float(self._segment_length_m.sum())
        if self._total_length_m <= 1.0e-12:
            raise ValueError("有序地面 Path 不能全部由退化线段组成")
        for array in (
            self._points,
            self._segment_delta_xyz,
            self._segment_length_m,
            self._segment_start_progress_m,
            self._segment_end_progress_m,
        ):
            array.setflags(write=False)

    @property
    def points_world_xyz(self) -> np.ndarray:
        """返回只读的世界系地面点。"""

        return self._points

    @property
    def point_count(self) -> int:
        """返回路径点数。"""

        return int(self._points.shape[0])

    @property
    def total_length_m(self) -> float:
        """返回三维累计弧长。"""

        return self._total_length_m

    def _segment_indices_in_arc_window(
        self,
        start_progress_m: float,
        end_progress_m: float,
    ) -> np.ndarray:
        """返回与闭弧长窗口相交的非退化线段索引。"""

        lower = max(0.0, float(start_progress_m))
        upper = min(self._total_length_m, float(end_progress_m))
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Path 弧长窗口必须是有限数值")
        if upper < lower:
            raise ValueError("Path 弧长窗口上界不能小于下界")
        return np.flatnonzero(
            (self._segment_length_m > 1.0e-12)
            & (self._segment_end_progress_m >= lower)
            & (self._segment_start_progress_m <= upper)
        )

    def project_progress(
        self,
        query_world_xyz: Iterable[float],
        *,
        previous_progress_m: float | None,
        backward_arc_m: float,
        forward_arc_m: float,
    ) -> float:
        """把机器人 XYZ 投影到有序 Path，并返回不回退的累计弧长。

        首次投影检查整条路径，以三维距离区分 XY 重叠的不同楼层。后续投影
        只检查上一进度附近的弧长窗口；距离相同的自交点优先选择离上一进度
        最近的分支，最后再执行单调夹紧。
        """

        query = np.asarray(
            _finite_triplet("query_world_xyz", query_world_xyz),
            dtype=np.float64,
        )
        backward = float(backward_arc_m)
        forward = float(forward_arc_m)
        if (
            not math.isfinite(backward)
            or not math.isfinite(forward)
            or backward < 0.0
            or forward <= 0.0
        ):
            raise ValueError("Path 进度搜索窗口必须满足 backward>=0、forward>0")

        previous: float | None
        if previous_progress_m is None:
            previous = None
            search_lower = 0.0
            search_upper = self._total_length_m
            indices = self._segment_indices_in_arc_window(
                search_lower,
                search_upper,
            )
        else:
            previous = float(previous_progress_m)
            if not math.isfinite(previous):
                raise ValueError("previous_progress_m 必须是有限数值")
            previous = min(max(previous, 0.0), self._total_length_m)
            search_lower = max(0.0, previous - backward)
            search_upper = min(
                self._total_length_m,
                previous + forward,
            )
            indices = self._segment_indices_in_arc_window(
                search_lower,
                search_upper,
            )
        if indices.size == 0:
            return 0.0 if previous is None else previous

        starts = self._points[indices]
        deltas = self._segment_delta_xyz[indices]
        length_squared = self._segment_length_m[indices] ** 2
        ratio_lower = np.clip(
            (
                search_lower
                - self._segment_start_progress_m[indices]
            )
            / self._segment_length_m[indices],
            0.0,
            1.0,
        )
        ratio_upper = np.clip(
            (
                search_upper
                - self._segment_start_progress_m[indices]
            )
            / self._segment_length_m[indices],
            0.0,
            1.0,
        )
        ratios = np.minimum(
            np.maximum(
                ((query - starts) * deltas).sum(axis=1)
                / length_squared,
                ratio_lower,
            ),
            ratio_upper,
        )
        projections = starts + ratios[:, None] * deltas
        distance_squared = ((projections - query) ** 2).sum(axis=1)
        candidate_progress = (
            self._segment_start_progress_m[indices]
            + ratios * self._segment_length_m[indices]
        )

        minimum_distance = float(distance_squared.min())
        tied = np.flatnonzero(
            np.isclose(
                distance_squared,
                minimum_distance,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        )
        if previous is None:
            selected = int(tied[np.argmin(candidate_progress[tied])])
            return float(candidate_progress[selected])

        selected = int(
            tied[
                np.argmin(
                    np.abs(candidate_progress[tied] - previous)
                )
            ]
        )
        return float(max(previous, candidate_progress[selected]))

    def local_segments(
        self,
        progress_m: float,
        *,
        backward_arc_m: float,
        forward_arc_m: float,
    ) -> LocalGroundPathSegments:
        """裁剪并预计算当前进度附近的支撑面插值线段。"""

        progress = float(progress_m)
        backward = float(backward_arc_m)
        forward = float(forward_arc_m)
        if not all(math.isfinite(value) for value in (progress, backward, forward)):
            raise ValueError("Path 局部窗口参数必须是有限数值")
        if backward < 0.0 or forward <= 0.0:
            raise ValueError("Path 局部窗口必须满足 backward>=0、forward>0")
        progress = min(max(progress, 0.0), self._total_length_m)
        lower = max(0.0, progress - backward)
        upper = min(self._total_length_m, progress + forward)
        indices = self._segment_indices_in_arc_window(lower, upper)

        clipped_starts: list[np.ndarray] = []
        clipped_ends: list[np.ndarray] = []
        for index in indices:
            length = float(self._segment_length_m[index])
            start_progress = float(self._segment_start_progress_m[index])
            ratio_start = min(
                1.0,
                max(0.0, (lower - start_progress) / length),
            )
            ratio_end = min(
                1.0,
                max(0.0, (upper - start_progress) / length),
            )
            if ratio_end - ratio_start <= 1.0e-12:
                continue
            start = (
                self._points[index]
                + ratio_start * self._segment_delta_xyz[index]
            )
            end = (
                self._points[index]
                + ratio_end * self._segment_delta_xyz[index]
            )
            # 纯竖直或 XY 退化段不是可插值的地面支撑线，安全地跳过。
            if float(np.dot(end[:2] - start[:2], end[:2] - start[:2])) <= 1.0e-12:
                continue
            clipped_starts.append(start)
            clipped_ends.append(end)

        if clipped_starts:
            starts = np.ascontiguousarray(clipped_starts, dtype=np.float64)
            ends = np.ascontiguousarray(clipped_ends, dtype=np.float64)
            delta_xy = np.ascontiguousarray(
                ends[:, :2] - starts[:, :2],
                dtype=np.float64,
            )
            length_squared_xy = np.einsum(
                "ij,ij->i",
                delta_xy,
                delta_xy,
            )
            inverse_length_squared_xy = np.ascontiguousarray(
                1.0 / length_squared_xy,
                dtype=np.float64,
            )
            start_xy = np.ascontiguousarray(starts[:, :2], dtype=np.float64)
            start_z = np.ascontiguousarray(starts[:, 2], dtype=np.float64)
            delta_z = np.ascontiguousarray(
                ends[:, 2] - starts[:, 2],
                dtype=np.float64,
            )
        else:
            start_xy = np.empty((0, 2), dtype=np.float64)
            delta_xy = np.empty((0, 2), dtype=np.float64)
            inverse_length_squared_xy = np.empty((0,), dtype=np.float64)
            start_z = np.empty((0,), dtype=np.float64)
            delta_z = np.empty((0,), dtype=np.float64)

        for array in (
            start_xy,
            delta_xy,
            inverse_length_squared_xy,
            start_z,
            delta_z,
        ):
            array.setflags(write=False)
        return LocalGroundPathSegments(
            start_xy=start_xy,
            delta_xy=delta_xy,
            inverse_length_squared_xy=inverse_length_squared_xy,
            start_z=start_z,
            delta_z=delta_z,
            progress_start_m=lower,
            progress_end_m=upper,
        )


def path_ground_band_mask(
    points_world_xyz: object,
    local_path_segments: LocalGroundPathSegments,
    *,
    corridor_radius_m: float,
    clearance_m: float,
    band_down_m: float,
    stair_minimum_slope: float | None = None,
    stair_clearance_m: float | None = None,
    stair_band_down_m: float | None = None,
) -> np.ndarray:
    """标记参考 Path 地面廊道内的支撑面窄带点。

    Path 的 z 必须是世界系地面高度。调用方先按机器人单调进度裁剪局部弧长
    窗口，本函数仅在预计算线段上分块投影。候选线段以三维残差选取，可区分
    XY 重叠的楼层；最终仍只过滤有限宽度、有限高度的窄带。

    楼梯 Path 往往比实际踏面稀疏，折线内插高度会落在相邻两级踏面
    之间。只对 ``|dz| / dxy`` 不小于阈值的线段采用楼梯上下窄带；
    平地和缓坡仍使用常规窄带，防止将低矮障碍误判为支撑面。
    """

    points = finite_xyz(points_world_xyz).astype(np.float64, copy=False)
    if not isinstance(local_path_segments, LocalGroundPathSegments):
        raise ValueError("local_path_segments 类型无效")

    corridor_radius = float(corridor_radius_m)
    clearance = float(clearance_m)
    band_down = float(band_down_m)
    if (stair_minimum_slope is None) != (stair_clearance_m is None):
        raise ValueError("楼梯斜率阈值与楼梯间隙必须同时提供")
    stair_slope = (
        math.inf
        if stair_minimum_slope is None
        else float(stair_minimum_slope)
    )
    stair_clearance = (
        clearance
        if stair_clearance_m is None
        else float(stair_clearance_m)
    )
    stair_band_down = (
        band_down
        if stair_band_down_m is None
        else float(stair_band_down_m)
    )
    finite_values = [
        corridor_radius,
        clearance,
        band_down,
        stair_clearance,
        stair_band_down,
    ]
    if stair_minimum_slope is not None:
        finite_values.append(stair_slope)
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("Path 地面廊道参数必须是有限数值")
    if (
        corridor_radius < 0.0
        or clearance < 0.0
        or band_down < 0.0
        or stair_slope <= 0.0
        or stair_clearance < clearance
        or stair_band_down < band_down
    ):
        raise ValueError(
            "Path 地面廊道要求非负边界、正楼梯斜率，且楼梯上下窄带"
            "均不得小于常规窄带"
        )
    if (
        points.shape[0] == 0
        or corridor_radius == 0.0
        or local_path_segments.segment_count == 0
    ):
        return np.zeros(points.shape[0], dtype=bool)

    result = np.zeros(points.shape[0], dtype=bool)
    starts = local_path_segments.start_xy
    deltas = local_path_segments.delta_xy
    inverse_length_squared = (
        local_path_segments.inverse_length_squared_xy
    )
    start_z = local_path_segments.start_z
    delta_z = local_path_segments.delta_z
    segment_slope = np.abs(delta_z) * np.sqrt(inverse_length_squared)
    segment_clearance = np.where(
        segment_slope >= stair_slope,
        stair_clearance,
        clearance,
    )
    segment_band_down = np.where(
        segment_slope >= stair_slope,
        stair_band_down,
        band_down,
    )

    # 固定点块上限，避免 30k 点与局部线段广播产生不可控峰值内存。
    chunk_size = 4096
    corridor_squared = corridor_radius * corridor_radius
    for begin in range(0, points.shape[0], chunk_size):
        end = min(begin + chunk_size, points.shape[0])
        chunk = points[begin:end]
        relative_x = chunk[:, None, 0] - starts[None, :, 0]
        relative_y = chunk[:, None, 1] - starts[None, :, 1]
        ratio = np.clip(
            (
                relative_x * deltas[None, :, 0]
                + relative_y * deltas[None, :, 1]
            )
            * inverse_length_squared[None, :],
            0.0,
            1.0,
        )
        offset_x = relative_x - ratio * deltas[None, :, 0]
        offset_y = relative_y - ratio * deltas[None, :, 1]
        distance_squared_xy = offset_x * offset_x + offset_y * offset_y
        expected_z = start_z[None, :] + ratio * delta_z[None, :]
        residual_z = chunk[:, None, 2] - expected_z
        # 使用三维残差选定候选，可避免低层与高层 XY 重叠时取错高度。
        candidate_score = distance_squared_xy + residual_z * residual_z
        best_index = np.argmin(candidate_score, axis=1)
        row_index = np.arange(end - begin)
        best_distance_squared_xy = distance_squared_xy[
            row_index,
            best_index,
        ]
        best_expected_z = expected_z[row_index, best_index]
        best_clearance = segment_clearance[best_index]
        best_band_down = segment_band_down[best_index]
        result[begin:end] = (
            (best_distance_squared_xy <= corridor_squared)
            & (chunk[:, 2] >= best_expected_z - best_band_down)
            & (chunk[:, 2] <= best_expected_z + best_clearance)
        )
    return result


def classify_ray_endpoints_xyz(
    points_world_xyz: object,
    *,
    base_position_world_xyz: Iterable[float],
    base_yaw_rad: float | None,
    config: PointCloudFilterConfig,
    base_orientation_world_xyzw: Iterable[float] | None = None,
    local_ground_path_segments: LocalGroundPathSegments | None = None,
) -> RayEndpointClassification:
    """分类世界系射线端点，并保持输入点顺序。

    优先使用 Odometry 的完整四元数把世界点转换到 ``base_link``；兼容调用
    可只提供 yaw。局部地面中心为机体系 ``z=-body_height``，因此机器人在
    坡面上产生的 roll/pitch 不会把前方正常支撑面误写成障碍。只剔除该平面
    上下的窄带，不删除更低楼梯或台阶；双圆柱中心位于机体 x 轴正负
    ``double_cylinder_offset`` 处，竖直边界相对 ``base_link`` 定义。

    地面和 Path 支撑面是真实测得的射线端点，但不应成为障碍；将它们
    标记为显式自由端点，使 GridMap 只沿有观测证据的射线清除旧占据。
    机器人自点会遮挡后方空间，因此仍然完全丢弃，不得用于清图。
    """

    points = finite_xyz(points_world_xyz)
    if points.shape[0] == 0:
        return RayEndpointClassification(
            points_xyz=points,
            endpoint_types=np.empty((0,), dtype=np.uint8),
        )

    base_position = np.asarray(
        _finite_triplet("base_position_world_xyz", base_position_world_xyz),
        dtype=np.float64,
    )

    delta = points.astype(np.float64, copy=False) - base_position
    if base_orientation_world_xyzw is None:
        if base_yaw_rad is None:
            raise ValueError("必须提供底盘完整四元数或 yaw")
        yaw = float(base_yaw_rad)
        if not math.isfinite(yaw):
            raise ValueError("base_yaw_rad 必须是有限数值")
        body_to_world = _body_to_world_rotation_xyzw(
            (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        )
    else:
        body_to_world = _body_to_world_rotation_xyzw(
            base_orientation_world_xyzw
        )
    # 对行向量右乘 R，等价于对列向量应用 R^T，把世界系转换回机体系。
    body_points = delta @ body_to_world

    distance_squared = np.einsum("ij,ij->i", body_points, body_points)
    spatial_keep = np.logical_and(
        distance_squared >= config.range_min_m * config.range_min_m,
        distance_squared <= config.range_max_m * config.range_max_m,
    )

    crop_min = np.asarray(config.crop_min_xyz_m, dtype=np.float64)
    crop_max = np.asarray(config.crop_max_xyz_m, dtype=np.float64)
    spatial_keep &= np.logical_and(
        body_points >= crop_min,
        body_points <= crop_max,
    ).all(axis=1)

    support_surface = np.zeros(points.shape[0], dtype=bool)
    if config.filter_ground:
        ground_center_relative_base = -config.body_height_m
        inside_ground_band = np.logical_and(
            body_points[:, 2]
            >= ground_center_relative_base - config.ground_band_down_m,
            body_points[:, 2]
            <= ground_center_relative_base + config.ground_clearance_m,
        )
        support_surface |= inside_ground_band

    if (
        config.filter_path_ground
        and config.path_ground_corridor_radius_m > 0.0
        and local_ground_path_segments is not None
    ):
        inside_path_ground_band = path_ground_band_mask(
            points,
            local_ground_path_segments,
            corridor_radius_m=config.path_ground_corridor_radius_m,
            clearance_m=config.path_ground_clearance_m,
            band_down_m=config.path_ground_band_down_m,
            stair_minimum_slope=config.path_ground_stair_minimum_slope,
            stair_clearance_m=config.path_ground_stair_clearance_m,
            stair_band_down_m=config.path_ground_stair_band_down_m,
        )
        support_surface |= inside_path_ground_band

    inside_self = np.zeros(points.shape[0], dtype=bool)
    if config.filter_self and config.double_cylinder_radius_m > 0.0:
        front_distance_squared = (
            body_points[:, 0] - config.double_cylinder_offset_m
        ) ** 2 + body_points[:, 1] ** 2
        rear_distance_squared = (
            body_points[:, 0] + config.double_cylinder_offset_m
        ) ** 2 + body_points[:, 1] ** 2
        inside_xy = np.logical_or(
            front_distance_squared <= config.double_cylinder_radius_m**2,
            rear_distance_squared <= config.double_cylinder_radius_m**2,
        )
        inside_z = np.logical_and(
            body_points[:, 2] >= config.self_z_min_m,
            body_points[:, 2] <= config.self_z_max_m,
        )
        inside_self = np.logical_and(inside_xy, inside_z)

    keep = spatial_keep & ~inside_self
    endpoint_types = np.where(
        support_surface[keep],
        RAY_ENDPOINT_FREE,
        RAY_ENDPOINT_OCCUPIED,
    ).astype(np.uint8, copy=False)
    return RayEndpointClassification(
        points_xyz=np.ascontiguousarray(points[keep], dtype=np.float32),
        endpoint_types=np.ascontiguousarray(endpoint_types, dtype=np.uint8),
    )


def filter_points_xyz(
    points_world_xyz: object,
    *,
    base_position_world_xyz: Iterable[float],
    base_yaw_rad: float | None,
    config: PointCloudFilterConfig,
    base_orientation_world_xyzw: Iterable[float] | None = None,
    local_ground_path_segments: LocalGroundPathSegments | None = None,
) -> np.ndarray:
    """返回只含障碍端点的兼容视图。"""

    classified = classify_ray_endpoints_xyz(
        points_world_xyz,
        base_position_world_xyz=base_position_world_xyz,
        base_yaw_rad=base_yaw_rad,
        config=config,
        base_orientation_world_xyzw=base_orientation_world_xyzw,
        local_ground_path_segments=local_ground_path_segments,
    )
    occupied = (
        classified.endpoint_types == RAY_ENDPOINT_OCCUPIED
    )
    return np.ascontiguousarray(
        classified.points_xyz[occupied],
        dtype=np.float32,
    )
