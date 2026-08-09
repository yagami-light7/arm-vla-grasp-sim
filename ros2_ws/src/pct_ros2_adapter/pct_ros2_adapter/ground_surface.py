"""把 PCT 粗切片高度投影到 collision PLY 的真实支撑面。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GroundProjection:
    """单个 XY 点的支撑面投影结果。"""

    z: float
    face_index: int
    hint_error_m: float


class TriangleGroundProjector:
    """缓存三角面包围盒，并按粗 slice 高度选择正确楼层支撑面。"""

    def __init__(
        self,
        vertices: np.ndarray,
        face_indices: np.ndarray,
        *,
        maximum_hint_error_m: float,
    ) -> None:
        maximum_error = float(maximum_hint_error_m)
        if not math.isfinite(maximum_error) or maximum_error <= 0.0:
            raise ValueError("maximum_hint_error_m 必须是有限正数")
        vertices_array = np.asarray(vertices, dtype=np.float64)
        faces_array = np.asarray(face_indices)
        if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
            raise ValueError("collision PLY vertices 必须是 N×3")
        if faces_array.ndim != 2 or faces_array.shape[1] != 3:
            raise ValueError("collision PLY faces 必须是 M×3")
        if vertices_array.shape[0] == 0:
            raise ValueError("collision PLY 不包含顶点")
        if faces_array.size == 0:
            raise ValueError("collision PLY 不包含三角面")
        if not np.issubdtype(faces_array.dtype, np.integer):
            raise ValueError("collision PLY face index 必须是整数")
        if not np.isfinite(vertices_array).all():
            raise ValueError("collision PLY vertices 不能包含 NaN 或 Inf")
        if int(faces_array.min()) < 0 or int(faces_array.max()) >= len(
            vertices_array
        ):
            raise ValueError("collision PLY face index 超出 vertex 范围")

        all_triangles = vertices_array[faces_array]
        all_xy = all_triangles[:, :, :2]
        all_denominators = (
            (all_xy[:, 1, 1] - all_xy[:, 2, 1])
            * (all_xy[:, 0, 0] - all_xy[:, 2, 0])
            + (all_xy[:, 2, 0] - all_xy[:, 1, 0])
            * (all_xy[:, 0, 1] - all_xy[:, 2, 1])
        )
        projectable = np.abs(all_denominators) > 1.0e-12
        if not np.any(projectable):
            raise ValueError("collision PLY 不包含可做垂直投影的支撑面")

        # 垂直墙面在 XY 上退化，不能当作地面；初始化时一次性
        # 剔除并缓存包围盒与重心坐标分母，规划时不再重复扫面准备。
        self._triangles = all_triangles[projectable]
        self._face_indices = np.flatnonzero(projectable)
        self._denominators = all_denominators[projectable]
        xy = self._triangles[:, :, :2]
        self._lower_xy = xy.min(axis=1)
        self._upper_xy = xy.max(axis=1)
        self._first_xy = xy[:, 0]
        self._second_xy = xy[:, 1]
        self._third_xy = xy[:, 2]
        self._maximum_hint_error_m = maximum_error

    def project(self, *, x: float, y: float, z_hint: float) -> GroundProjection:
        """返回固定 XY 上最接近 PCT 粗 slice 高度的真实三角面交点。"""

        query_x = float(x)
        query_y = float(y)
        hint = float(z_hint)
        if not all(math.isfinite(value) for value in (query_x, query_y, hint)):
            raise ValueError("支撑面投影输入不能包含 NaN 或 Inf")
        candidate_indices = np.flatnonzero(
            (self._lower_xy[:, 0] <= query_x)
            & (query_x <= self._upper_xy[:, 0])
            & (self._lower_xy[:, 1] <= query_y)
            & (query_y <= self._upper_xy[:, 1])
        )
        if candidate_indices.size == 0:
            raise ValueError(
                f"collision PLY 在 XY=({query_x:.6f},{query_y:.6f}) 没有支撑面"
            )
        triangles = self._triangles[candidate_indices]
        face_ids = self._face_indices[candidate_indices]
        first = self._first_xy[candidate_indices]
        second = self._second_xy[candidate_indices]
        third = self._third_xy[candidate_indices]
        denominator = self._denominators[candidate_indices]
        weight_first = (
            (second[:, 1] - third[:, 1]) * (query_x - third[:, 0])
            + (third[:, 0] - second[:, 0]) * (query_y - third[:, 1])
        ) / denominator
        weight_second = (
            (third[:, 1] - first[:, 1]) * (query_x - third[:, 0])
            + (first[:, 0] - third[:, 0]) * (query_y - third[:, 1])
        ) / denominator
        weight_third = 1.0 - weight_first - weight_second
        inside = (
            (weight_first >= -1.0e-7)
            & (weight_second >= -1.0e-7)
            & (weight_third >= -1.0e-7)
        )
        if not np.any(inside):
            raise ValueError(
                f"collision PLY 在 XY=({query_x:.6f},{query_y:.6f}) 没有有效交点"
            )
        triangles = triangles[inside]
        face_ids = face_ids[inside]
        weight_first = weight_first[inside]
        weight_second = weight_second[inside]
        weight_third = weight_third[inside]
        intersections = (
            weight_first * triangles[:, 0, 2]
            + weight_second * triangles[:, 1, 2]
            + weight_third * triangles[:, 2, 2]
        )
        errors = np.abs(intersections - hint)
        selected = int(np.argmin(errors))
        error = float(errors[selected])
        if error > self._maximum_hint_error_m:
            ranked = np.argsort(errors)[: min(3, len(errors))]
            candidate_summary = ",".join(
                f"{float(intersections[index]):.3f}"
                for index in ranked
            )
            raise ValueError(
                "collision PLY 最近支撑面与 PCT slice 高度相差 "
                f"{error:.3f} m，超过 {self._maximum_hint_error_m:.3f} m；"
                f"最近候选 z={candidate_summary}"
            )
        return GroundProjection(
            z=float(intersections[selected]),
            face_index=int(face_ids[selected]),
            hint_error_m=error,
        )

    def project_path(
        self,
        points_pct_xyz: Sequence[Sequence[float]],
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[GroundProjection, ...],
    ]:
        """投影完整 PCT 坐标路径，并保留每点诊断。"""

        points: list[tuple[float, float, float]] = []
        reports: list[GroundProjection] = []
        for index, raw_point in enumerate(points_pct_xyz):
            if len(raw_point) != 3:
                raise ValueError(f"points_pct_xyz[{index}] 必须包含 3 个坐标")
            x, y, z_hint = (float(value) for value in raw_point)
            try:
                report = self.project(x=x, y=y, z_hint=z_hint)
            except ValueError as exc:
                raise ValueError(
                    f"points_pct_xyz[{index}]=({x:.6f},{y:.6f},{z_hint:.6f}) "
                    f"投影失败：{exc}"
                ) from exc
            points.append((x, y, report.z))
            reports.append(report)
        return tuple(points), tuple(reports)
