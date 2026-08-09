"""官方 PCT TomogramPlanner 的进程内 ROS 2 backend。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import sysconfig
import threading
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterator, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from .backend import (
    PCTBackendConfig,
    PCTBackendError,
    PCTBackendPlan,
    PCTNoPathError,
    _finite_xyz,
    _load_project_coordinate_module,
    _sample_polyline,
    _validated_config,
)
from .ground_surface import TriangleGroundProjector


UPSTREAM_PCT_REPOSITORY = "https://github.com/byangw/PCT_planner"
UPSTREAM_PCT_COMMIT = "35cd73fd82bcd51bc538429294af7646b2a09815"
UPSTREAM_PCT_ARCHIVE_SHA256 = (
    "daf5f90b29c76cfa5fc6bf10d6dcfd200c1077778b22671c98aa51f9adb06d64"
)
UPSTREAM_PCT_LICENSE = "GPL-2.0-or-later"
UPSTREAM_PCT_PATCH_ID = (
    "pct-scan-native-astar-cancel-cost-aware-no-corner-cut-gateway-v3"
)
UPSTREAM_PCT_PATCH_SHA256 = (
    "fa6f9364b7bdf07a9d698e40083617646829608e64727bf5c68df391589b4e1b"
)

# 运行身份是固定 upstream commit 加受跟踪的 pct-scan patch；不能把补丁态
# 工作副本误报为未经修改的官方提交。完整 pre/postimage 在 manifest v2 中。
_PINNED_FILE_SHA256 = {
    "LICENSE": "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643",
    "NOTICE": "65812c5a3752aaa175ae35eacbcbad415a443cbe6f4ae015fa8834d4b886e7b7",
    "planner/scripts/planner_wrapper.py": (
        "815b5265d6c5fea97e63c1dded547989d032b0878db60ccbcfe942e46a9a19d8"
    ),
    "planner/lib/CMakeLists.txt": (
        "865efc6f76026a38a090a76a7c9895a313c9999b26814df81e80a7278dfe69d7"
    ),
    "planner/lib/src/a_star/a_star_search.h": (
        "264d2325d9d64be71d3ee6eab89179fd90037d6cb009196f73c058fec5e1bc0b"
    ),
    "planner/lib/src/a_star/a_star_search.cc": (
        "728bd418b4e8483decf026300f29c7be4d0581fbc06ba5b75cf9012346964d21"
    ),
    "planner/lib/src/a_star/python_interface.cc": (
        "58ba72d3449df9593d1ab9409f30df6ec7b2b8fc4b13a115a248ff522097f302"
    ),
    "planner/lib/src/ele_planner/offline_ele_planner.h": (
        "4bcd6637a6b7ee228b2dbb3c7a0fa17d8fc15562531ceb17ce2eb0c117f0a94e"
    ),
    "planner/lib/src/ele_planner/python_interface.cc": (
        "8c677ee8e3de0b4c1a8557d832b5fa9bc4733ec0843162b87f95688ff1ba0462"
    ),
    "planner/lib/src/ele_planner/offline_ele_planner.cc": (
        "7ed8bcd7d4c0248935f71c08d4847974662c07c4d3664e9ae6358731ba9831a8"
    ),
}
_UPSTREAM_IMPORT_LOCK = threading.RLock()
_UPSTREAM_ASTAR_COST_THRESHOLD = 20.0


@dataclass(frozen=True)
class _EndpointLayerMatch:
    """端点 XY 单元内按真实地面高度选出的官方逻辑层。"""

    layer: int
    grid_x: int
    grid_y: int
    ground_z: float
    cost: float
    height_error_m: float


@dataclass(frozen=True)
class _StairApproachClearanceContract:
    """一楼楼梯接近段针对真实 collision PLY 的包络净空合同。"""

    double_cylinder_radius_m: float
    double_cylinder_offset_m: float
    obstacle_minimum_z_m: float
    obstacle_maximum_z_m: float
    sample_spacing_m: float
    maximum_yaw_step_rad: float
    minimum_surface_clearance_m: float


@dataclass(frozen=True)
class _StairCenterlineProfile:
    """与 upstream tomogram 同源的仿真世界系楼梯中心线。"""

    path: Path
    sha256: str
    asset_kind: str
    anchors_sim_ground_xyz: tuple[tuple[float, float, float], ...]
    lower_floor_approach_anchors_sim_ground_xyz: tuple[
        tuple[float, float, float], ...
    ] = ()
    lower_floor_approach_clearance: (
        _StairApproachClearanceContract | None
    ) = None
    sampling_spacing_m: float = 0.10
    corridor_radius_cells: int = 0
    logical_layer_first: int = 0
    logical_layer_last: int = 0
    ground_z_origin_m: float = 0.0
    height_step_m: float = 0.50


@dataclass(frozen=True)
class _BodyClearanceOverlay:
    """写入原生 A* 的机身软净空代价与诊断。"""

    traversability: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    report: dict[str, object]


class _UpstreamTomogramIndex:
    """保存官方简化层的真实地面与 gateway 查询信息。"""

    def __init__(self, tomogram_path: Path) -> None:
        try:
            with tomogram_path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise PCTBackendError(
                "官方 PCT tomogram 反序列化失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise PCTBackendError("官方 PCT tomogram 顶层必须是 dict")
        missing = {
            "data",
            "resolution",
            "center",
            "slice_h0",
            "slice_dh",
        } - payload.keys()
        if missing:
            raise PCTBackendError(
                f"官方 PCT tomogram 缺少字段：{sorted(missing)}"
            )
        data = np.asarray(payload["data"], dtype=np.float32)
        if data.ndim != 4 or data.shape[0] != 5 or data.shape[1] < 1:
            raise PCTBackendError(
                "官方 PCT tomogram data 必须是 5×L×X×Y，"
                f"收到 {data.shape}"
            )
        resolution = float(payload["resolution"])
        center = np.asarray(payload["center"], dtype=np.float64)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise PCTBackendError("官方 PCT tomogram resolution 必须为正数")
        if center.shape != (2,) or not np.isfinite(center).all():
            raise PCTBackendError("官方 PCT tomogram center 必须是有限 XY")
        if not np.isfinite(data[0]).all():
            raise PCTBackendError("官方 PCT traversability 含 NaN 或 Inf")

        self.path = tomogram_path
        self.traversability = data[0]
        self.gradient_x = data[1]
        self.gradient_y = data[2]
        self.ground = data[3]
        self.ceiling = data[4]
        self.resolution = resolution
        self.center = center
        self.layer_count = int(data.shape[1])
        self.grid_dim_x = int(data.shape[2])
        self.grid_dim_y = int(data.shape[3])
        self.asset_kind = str(payload.get("pct_scan_asset_kind", "unknown"))
        self.stair_profile_sha256 = str(
            payload.get("stair_profile_sha256", "")
        ).strip()
        converted_ground = np.nan_to_num(self.ground, nan=-100.0)
        diff_t = self.traversability[1:] - self.traversability[:-1]
        diff_g = np.abs(converted_ground[1:] - converted_ground[:-1])
        self.gateway_up_count = int(
            np.count_nonzero((diff_t < -8.0) & (diff_g < 0.1))
        )
        self.gateway_down_count = int(
            np.count_nonzero((diff_t > 8.0) & (diff_g < 0.1))
        )
        self._blocked_cell_centers_xy: dict[int, np.ndarray] = {}
        self._shortcut_blocked_cell_source = "native_traversability"

    def configure_shortcut_body_obstacles(
        self,
        overlay_traversability: np.ndarray,
        *,
        obstacle_surface_cost: float,
    ) -> None:
        """让视线压缩同时避开机身高度带里的真实障碍表面。"""

        overlay = np.asarray(overlay_traversability, dtype=np.float32)
        surface_cost = float(obstacle_surface_cost)
        if overlay.shape != self.traversability.shape:
            raise ValueError("视线压缩净空层与 tomogram 尺寸不一致")
        if not math.isfinite(surface_cost) or surface_cost <= 0.0:
            raise ValueError("视线压缩障碍表面代价必须是有限正数")

        self._blocked_cell_centers_xy.clear()
        for layer in range(self.layer_count):
            blocked = (
                ~np.isfinite(self.traversability[layer])
                | (
                    self.traversability[layer]
                    > _UPSTREAM_ASTAR_COST_THRESHOLD
                )
                | (overlay[layer] >= surface_cost - 1.0e-6)
            )
            blocked_indices = np.argwhere(blocked)
            self._blocked_cell_centers_xy[layer] = np.column_stack(
                (
                    (
                        blocked_indices[:, 0] - self.grid_dim_x // 2
                    )
                    * self.resolution
                    + self.center[0],
                    (
                        blocked_indices[:, 1] - self.grid_dim_y // 2
                    )
                    * self.resolution
                    + self.center[1],
                )
            ).astype(np.float64, copy=False)
        self._shortcut_blocked_cell_source = (
            "native_traversability_plus_body_obstacle_surface"
        )

    def endpoint_layer(
        self,
        *,
        point_xyz: Sequence[float],
        maximum_height_error_m: float,
        label: str,
    ) -> _EndpointLayerMatch:
        """按端点 XY 的 ground/cost 选层，禁止用简化前 slice 公式。"""

        point = _finite_xyz(point_xyz, field_name=f"{label}_ground_pct")
        grid_x = int(
            np.rint((point[0] - self.center[0]) / self.resolution)
        ) + self.grid_dim_x // 2
        grid_y = int(
            np.rint((point[1] - self.center[1]) / self.resolution)
        ) + self.grid_dim_y // 2
        if not 0 <= grid_x < self.grid_dim_x or not 0 <= grid_y < self.grid_dim_y:
            raise PCTNoPathError(
                f"官方 PCT {label} XY 映射到 tomogram 范围外："
                f"grid=({grid_x}, {grid_y}), "
                f"shape=({self.grid_dim_x}, {self.grid_dim_y})"
            )

        candidates: list[tuple[float, float, int, float]] = []
        for layer in range(self.layer_count):
            ground_z = float(self.ground[layer, grid_x, grid_y])
            cost = float(self.traversability[layer, grid_x, grid_y])
            if (
                math.isfinite(ground_z)
                and math.isfinite(cost)
                and cost <= _UPSTREAM_ASTAR_COST_THRESHOLD
            ):
                candidates.append(
                    (abs(ground_z - point[2]), cost, layer, ground_z)
                )
        if not candidates:
            raise PCTNoPathError(
                f"官方 PCT {label} XY 在所有逻辑层均不可通行："
                f"grid=({grid_x}, {grid_y})"
            )
        height_error, cost, layer, ground_z = min(candidates)
        if height_error > float(maximum_height_error_m):
            raise PCTNoPathError(
                f"官方 PCT {label} 找不到匹配地面层："
                f"query_z={point[2]:.3f}, nearest_z={ground_z:.3f}, "
                f"error={height_error:.3f} m > "
                f"{maximum_height_error_m:.3f} m"
            )
        return _EndpointLayerMatch(
            layer=layer,
            grid_x=grid_x,
            grid_y=grid_y,
            ground_z=ground_z,
            cost=cost,
            height_error_m=height_error,
        )

    def validate_cross_layer_gateway(
        self,
        start: _EndpointLayerMatch,
        goal: _EndpointLayerMatch,
    ) -> None:
        """跨逻辑层前确认官方 wrapper 实际能生成对应方向 gateway。"""

        if start.layer < goal.layer and self.gateway_up_count == 0:
            raise PCTNoPathError(
                "官方 PCT 多楼层 tomogram 没有向上 gateway；"
                "当前资产不是可供 upstream A* 跨层的 tomography 输出"
            )
        if start.layer > goal.layer and self.gateway_down_count == 0:
            raise PCTNoPathError(
                "官方 PCT 多楼层 tomogram 没有向下 gateway；"
                "当前资产不是可供 upstream A* 跨层的 tomography 输出"
            )

    def native_astar_ground_points(
        self,
        path_matrix: object,
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[int, ...],
    ]:
        """把官方 A* 的 layer/grid matrix 转为 PCT 地面坐标。"""

        try:
            matrix = np.asarray(path_matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PCTBackendError("官方 PCT A* path matrix 不可解析") from exc
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise PCTBackendError(
                "官方 PCT A* path matrix 必须是 N×3，"
                f"收到 {matrix.shape}"
            )
        if len(matrix) < 1 or not np.isfinite(matrix).all():
            raise PCTNoPathError("官方 PCT A* path matrix 为空或含非法值")
        output: list[tuple[float, float, float]] = []
        logical_layers: list[int] = []
        for index, row in enumerate(matrix):
            rounded = np.rint(row)
            if not np.allclose(row, rounded, rtol=0.0, atol=1.0e-6):
                raise PCTBackendError(
                    f"官方 PCT A* path matrix[{index}] 不是整数索引"
                )
            layer, grid_x, grid_y = (int(value) for value in rounded)
            if layer < 0 or layer >= self.layer_count:
                raise PCTBackendError(
                    f"官方 PCT A* layer[{index}] 越界：{layer}"
                )
            if (
                grid_x < 0
                or grid_x >= self.grid_dim_x
                or grid_y < 0
                or grid_y >= self.grid_dim_y
            ):
                raise PCTBackendError(
                    f"官方 PCT A* path matrix[{index}] XY 越界"
                )
            ground_z = float(self.ground[layer, grid_x, grid_y])
            if not math.isfinite(ground_z):
                raise PCTBackendError(
                    "官方 PCT A* 节点的逻辑层没有有效地面："
                    f"index={index}, layer={layer}, "
                    f"grid=({grid_x}, {grid_y})"
                )
            point_x = (
                (grid_x - self.grid_dim_x // 2) * self.resolution
                + float(self.center[0])
            )
            point_y = (
                (grid_y - self.grid_dim_y // 2) * self.resolution
                + float(self.center[1])
            )
            output.append((point_x, point_y, ground_z))
            logical_layers.append(layer)
        return tuple(output), tuple(logical_layers)

    def shortcut_same_layer_path(
        self,
        points_xyz: Sequence[Sequence[float]],
        *,
        layer: int,
        clearance_m: float,
        maximum_segment_m: float,
    ) -> tuple[tuple[tuple[float, float, float], ...], dict[str, object]]:
        """用带机身净空的视线段压缩同层 A* 栅格折线。"""

        points = tuple(
            _finite_xyz(point, field_name="upstream_same_layer_shortcut")
            for point in points_xyz
        )
        clearance = float(clearance_m)
        maximum_segment = float(maximum_segment_m)
        if not 0 <= int(layer) < self.layer_count:
            raise ValueError("同层捷径的逻辑层越界")
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("同层捷径净空必须是有限非负数")
        if not math.isfinite(maximum_segment) or maximum_segment <= 0.0:
            raise ValueError("同层捷径最大线段长度必须是有限正数")

        report: dict[str, object] = {
            "applied": False,
            "reason": "path_has_at_most_two_points",
            "raw_point_count": len(points),
            "shortcut_point_count": len(points),
            "logical_layer": int(layer),
            "clearance_m": clearance,
            "grid_cell_cover_margin_m": self.resolution / math.sqrt(2.0),
            "maximum_segment_m": maximum_segment,
            "minimum_blocked_cell_center_distance_m": None,
            "minimum_verified_segment_clearance_m": None,
            "minimum_preserved_adjacent_segment_clearance_m": None,
            "preserved_adjacent_segment_count": 0,
            "blocked_cell_source": self._shortcut_blocked_cell_source,
        }
        if len(points) <= 2:
            return points, report

        shortcut: list[tuple[float, float, float]] = [points[0]]
        selected_clearances: list[float] = []
        verified_clearances: list[float] = []
        preserved_clearances: list[float] = []
        preserved_adjacent_segment_count = 0
        anchor = 0
        while anchor < len(points) - 1:
            selected = anchor + 1
            selected_safe, selected_clearance = (
                self._same_layer_segment_is_clear(
                    points[anchor],
                    points[selected],
                    layer=int(layer),
                    clearance_m=clearance,
                )
            )
            for candidate in range(len(points) - 1, anchor + 1, -1):
                if (
                    math.dist(points[anchor][:2], points[candidate][:2])
                    > maximum_segment + 1.0e-9
                ):
                    continue
                safe, blocked_clearance = self._same_layer_segment_is_clear(
                    points[anchor],
                    points[candidate],
                    layer=int(layer),
                    clearance_m=clearance,
                )
                if safe:
                    selected = candidate
                    selected_safe = True
                    selected_clearance = blocked_clearance
                    break
            shortcut.append(points[selected])
            if not selected_safe:
                preserved_adjacent_segment_count += 1
            if math.isfinite(selected_clearance):
                selected_clearances.append(selected_clearance)
                if selected_safe:
                    verified_clearances.append(selected_clearance)
                else:
                    preserved_clearances.append(selected_clearance)
            anchor = selected

        result = tuple(shortcut)
        report.update(
            {
                "applied": len(result) < len(points),
                "reason": (
                    "clearance_verified_line_of_sight"
                    if len(result) < len(points)
                    else "no_safe_reduction"
                ),
                "shortcut_point_count": len(result),
                "minimum_blocked_cell_center_distance_m": (
                    min(selected_clearances) if selected_clearances else None
                ),
                "minimum_verified_segment_clearance_m": (
                    min(verified_clearances) if verified_clearances else None
                ),
                "minimum_preserved_adjacent_segment_clearance_m": (
                    min(preserved_clearances)
                    if preserved_clearances
                    else None
                ),
                "preserved_adjacent_segment_count": (
                    preserved_adjacent_segment_count
                ),
            }
        )
        return result, report

    def shortcut_cross_layer_floor_segments(
        self,
        points_xyz: Sequence[Sequence[float]],
        *,
        profile_start_index: int,
        profile_end_index: int,
        start_layer: int,
        goal_layer: int,
        clearance_m: float,
        maximum_segment_m: float,
    ) -> tuple[tuple[tuple[float, float, float], ...], dict[str, object]]:
        """压缩跨层路径的两端平地折线，标定楼梯中心线保持不变。"""

        points = tuple(
            _finite_xyz(point, field_name="upstream_cross_layer_shortcut")
            for point in points_xyz
        )
        profile_first = int(profile_start_index)
        profile_last = int(profile_end_index)
        if not (
            0 <= profile_first < profile_last < len(points)
        ):
            raise ValueError("跨层路径的楼梯 profile 索引非法")

        prefix = points[: profile_first + 1]
        stair = points[profile_first: profile_last + 1]
        suffix = points[profile_last:]
        prefix_shortcut, prefix_report = self.shortcut_same_layer_path(
            prefix,
            layer=int(start_layer),
            clearance_m=clearance_m,
            maximum_segment_m=maximum_segment_m,
        )
        suffix_shortcut, suffix_report = self.shortcut_same_layer_path(
            suffix,
            layer=int(goal_layer),
            clearance_m=clearance_m,
            maximum_segment_m=maximum_segment_m,
        )
        combined = (
            *prefix_shortcut[:-1],
            *stair,
            *suffix_shortcut[1:],
        )
        result = tuple(combined)
        output_profile_first = len(prefix_shortcut) - 1
        output_profile_last = output_profile_first + len(stair) - 1
        return result, {
            "applied": len(result) < len(points),
            "reason": (
                "cross_layer_floor_segments_clearance_shortcut"
                if len(result) < len(points)
                else "cross_layer_floor_segments_no_safe_reduction"
            ),
            "raw_point_count": len(points),
            "shortcut_point_count": len(result),
            "input_profile_start_index": profile_first,
            "input_profile_end_index": profile_last,
            "profile_start_index": output_profile_first,
            "profile_end_index": output_profile_last,
            "profile_point_count": len(stair),
            "clearance_m": float(clearance_m),
            "grid_cell_cover_margin_m": self.resolution / math.sqrt(2.0),
            "maximum_segment_m": float(maximum_segment_m),
            "blocked_cell_source": self._shortcut_blocked_cell_source,
            "start_floor_report": prefix_report,
            "goal_floor_report": suffix_report,
        }

    def _same_layer_segment_is_clear(
        self,
        start_xyz: Sequence[float],
        end_xyz: Sequence[float],
        *,
        layer: int,
        clearance_m: float,
    ) -> tuple[bool, float]:
        """保守检查圆形包络；半栅格对角线覆盖离散单元面积。"""

        start = np.asarray(start_xyz[:2], dtype=np.float64)
        end = np.asarray(end_xyz[:2], dtype=np.float64)
        map_min = np.asarray(
            (
                (0 - self.grid_dim_x // 2) * self.resolution
                + self.center[0]
                - self.resolution / 2.0,
                (0 - self.grid_dim_y // 2) * self.resolution
                + self.center[1]
                - self.resolution / 2.0,
            ),
            dtype=np.float64,
        )
        map_max = np.asarray(
            (
                (self.grid_dim_x - 1 - self.grid_dim_x // 2)
                * self.resolution
                + self.center[0]
                + self.resolution / 2.0,
                (self.grid_dim_y - 1 - self.grid_dim_y // 2)
                * self.resolution
                + self.center[1]
                + self.resolution / 2.0,
            ),
            dtype=np.float64,
        )
        if (
            np.any(np.minimum(start, end) - clearance_m < map_min)
            or np.any(np.maximum(start, end) + clearance_m > map_max)
        ):
            return False, 0.0

        blocked_clearance = self._segment_blocked_cell_clearance_m(
            start_xyz,
            end_xyz,
            layer=layer,
        )
        cell_cover_margin = self.resolution / math.sqrt(2.0)
        required_center_distance = clearance_m + cell_cover_margin
        return (
            blocked_clearance + 1.0e-9 >= required_center_distance,
            blocked_clearance,
        )

    def _segment_blocked_cell_clearance_m(
        self,
        start_xyz: Sequence[float],
        end_xyz: Sequence[float],
        *,
        layer: int,
    ) -> float:
        """返回线段到最近不可通行栅格中心的二维距离。"""

        blocked_xy = self._blocked_cell_centers_xy.get(layer)
        if blocked_xy is None:
            blocked = np.argwhere(
                ~np.isfinite(self.traversability[layer])
                | (
                    self.traversability[layer]
                    > _UPSTREAM_ASTAR_COST_THRESHOLD
                )
            )
            blocked_xy = np.column_stack(
                (
                    (
                        blocked[:, 0] - self.grid_dim_x // 2
                    )
                    * self.resolution
                    + self.center[0],
                    (
                        blocked[:, 1] - self.grid_dim_y // 2
                    )
                    * self.resolution
                    + self.center[1],
                )
            ).astype(np.float64, copy=False)
            self._blocked_cell_centers_xy[layer] = blocked_xy
        if len(blocked_xy) == 0:
            return math.inf

        start = np.asarray(start_xyz[:2], dtype=np.float64)
        end = np.asarray(end_xyz[:2], dtype=np.float64)
        segment = end - start
        denominator = float(segment @ segment)
        if denominator <= 1.0e-18:
            return float(np.min(np.linalg.norm(blocked_xy - start, axis=1)))
        alpha = np.clip(((blocked_xy - start) @ segment) / denominator, 0.0, 1.0)
        nearest = start + alpha[:, None] * segment
        return float(np.min(np.linalg.norm(blocked_xy - nearest, axis=1)))


def _sha256_file(path: Path) -> str:
    """流式计算资产哈希，避免把大型 collision PLY 整体复制到内存。"""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stair_centerline_profile(
    path: Path,
    *,
    config: PCTBackendConfig,
    tomogram_index: _UpstreamTomogramIndex,
) -> _StairCenterlineProfile:
    """加载并核对楼梯 profile、tomogram、PLY 与坐标合同。"""

    try:
        profile_bytes = path.read_bytes()
        payload = json.loads(profile_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PCTBackendError(f"楼梯中心线 profile 无法读取：{path}") from exc
    if not isinstance(payload, dict):
        raise PCTBackendError("楼梯中心线 profile 顶层必须是对象")
    if int(payload.get("schema_version", -1)) != 1:
        raise PCTBackendError("楼梯中心线 profile schema_version 必须为 1")

    profile_hash = sha256(profile_bytes).hexdigest()
    if not tomogram_index.stair_profile_sha256:
        raise PCTBackendError("upstream tomogram 缺少 stair_profile_sha256")
    if profile_hash != tomogram_index.stair_profile_sha256:
        raise PCTBackendError(
            "楼梯中心线 profile 与 upstream tomogram 的哈希不一致"
        )

    asset_kind = str(payload.get("asset_kind", "")).strip()
    if not asset_kind or asset_kind != tomogram_index.asset_kind:
        raise PCTBackendError(
            "楼梯中心线 profile 与 upstream tomogram 的 asset_kind 不一致"
        )
    expected_ply_hash = str(payload.get("collision_ply_sha256", "")).strip()
    if not expected_ply_hash:
        raise PCTBackendError("楼梯中心线 profile 缺少 collision_ply_sha256")
    collision_path = config.collision_ply_path
    if (
        collision_path is None
        or _sha256_file(collision_path) != expected_ply_hash
    ):
        raise PCTBackendError(
            "楼梯中心线 profile 与当前 collision PLY 的哈希不一致"
        )

    transform = payload.get("coordinate_transform")
    if not isinstance(transform, dict):
        raise PCTBackendError("楼梯中心线 profile 缺少 coordinate_transform")
    for key, expected in _coordinate_arguments(config).items():
        actual = transform.get(key)
        if isinstance(expected, str):
            matches = str(actual) == expected
        else:
            try:
                matches = math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise PCTBackendError(
                f"楼梯中心线 profile 坐标参数 {key} 与运行配置不一致"
            )

    raw_anchors = payload.get("anchors_sim_ground_xyz")
    if not isinstance(raw_anchors, list) or len(raw_anchors) < 2:
        raise PCTBackendError("楼梯中心线 profile 至少需要两个标定锚点")
    try:
        anchors = tuple(
            _finite_xyz(point, field_name="anchors_sim_ground_xyz")
            for point in raw_anchors
        )
    except (TypeError, ValueError) as exc:
        raise PCTBackendError("楼梯中心线 profile 锚点非法") from exc
    if any(
        anchors[index + 1][2] + 1.0e-9 < anchors[index][2]
        for index in range(len(anchors) - 1)
    ):
        raise PCTBackendError("楼梯中心线 profile 高度必须沿锚点顺序单调不减")

    approach_anchors: tuple[tuple[float, float, float], ...] = ()
    approach_clearance: _StairApproachClearanceContract | None = None
    approach = payload.get("lower_floor_approach")
    if approach is not None:
        if not isinstance(approach, dict):
            raise PCTBackendError("lower_floor_approach 必须是对象")
        raw_approach_anchors = approach.get("anchors_sim_ground_xyz")
        if (
            not isinstance(raw_approach_anchors, list)
            or len(raw_approach_anchors) < 3
        ):
            raise PCTBackendError("一楼接近段至少需要三个标定锚点")
        try:
            approach_anchors = tuple(
                _finite_xyz(
                    point,
                    field_name=(
                        "lower_floor_approach.anchors_sim_ground_xyz"
                    ),
                )
                for point in raw_approach_anchors
            )
        except (TypeError, ValueError) as exc:
            raise PCTBackendError("一楼接近段锚点非法") from exc
        if any(
            math.dist(first, second) <= 1.0e-9
            for first, second in zip(
                approach_anchors,
                approach_anchors[1:],
            )
        ):
            raise PCTBackendError("一楼接近段不能包含连续重复锚点")
        if math.dist(approach_anchors[-1], anchors[0]) > 1.0e-6:
            raise PCTBackendError(
                "一楼接近段末锚点必须与楼梯中心线入口完全相同"
            )
        if any(
            abs(point[2] - anchors[0][2]) > 1.0e-6
            for point in approach_anchors
        ):
            raise PCTBackendError(
                "一楼接近段必须保持楼梯入口所在的同一地面高度"
            )

        clearance = approach.get("clearance_audit")
        if not isinstance(clearance, dict):
            raise PCTBackendError("一楼接近段缺少 clearance_audit")
        try:
            approach_clearance = _StairApproachClearanceContract(
                double_cylinder_radius_m=float(
                    clearance["double_cylinder_radius_m"]
                ),
                double_cylinder_offset_m=float(
                    clearance["double_cylinder_offset_m"]
                ),
                obstacle_minimum_z_m=float(
                    clearance["obstacle_minimum_z_m"]
                ),
                obstacle_maximum_z_m=float(
                    clearance["obstacle_maximum_z_m"]
                ),
                sample_spacing_m=float(clearance["sample_spacing_m"]),
                maximum_yaw_step_rad=float(
                    clearance["maximum_yaw_step_rad"]
                ),
                minimum_surface_clearance_m=float(
                    clearance["minimum_surface_clearance_m"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PCTBackendError("一楼接近段 clearance_audit 非法") from exc
        finite_positive = (
            approach_clearance.double_cylinder_radius_m,
            approach_clearance.sample_spacing_m,
            approach_clearance.maximum_yaw_step_rad,
            approach_clearance.minimum_surface_clearance_m,
        )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in finite_positive
        ):
            raise PCTBackendError("一楼接近段净空合同中的正数参数非法")
        if (
            not math.isfinite(
                approach_clearance.double_cylinder_offset_m
            )
            or approach_clearance.double_cylinder_offset_m < 0.0
        ):
            raise PCTBackendError("一楼接近段双圆柱偏置不能为负数")
        if not (
            math.isfinite(approach_clearance.obstacle_minimum_z_m)
            and math.isfinite(approach_clearance.obstacle_maximum_z_m)
            and approach_clearance.obstacle_minimum_z_m
            < approach_clearance.obstacle_maximum_z_m
        ):
            raise PCTBackendError("一楼接近段障碍高度审计区间非法")
        if (
            approach_clearance.minimum_surface_clearance_m + 1.0e-9
            < approach_clearance.double_cylinder_radius_m
        ):
            raise PCTBackendError(
                "一楼接近段最小表面净空不得小于双圆柱半径"
            )

    logical_layer_band = payload.get("logical_layer_band")
    if not isinstance(logical_layer_band, dict):
        raise PCTBackendError("楼梯中心线 profile 缺少 logical_layer_band")
    try:
        sampling_spacing_m = float(payload["sampling_spacing_m"])
        corridor_radius_cells = int(payload["corridor_radius_cells"])
        logical_layer_first = int(logical_layer_band["first_layer"])
        logical_layer_last = int(logical_layer_band["last_layer"])
        ground_z_origin_m = float(logical_layer_band["ground_z_origin_m"])
        height_step_m = float(logical_layer_band["height_step_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PCTBackendError("楼梯中心线 profile 的栅格走廊参数非法") from exc
    if not math.isfinite(sampling_spacing_m) or sampling_spacing_m <= 0.0:
        raise PCTBackendError("楼梯中心线 sampling_spacing_m 必须为有限正数")
    if corridor_radius_cells < 0:
        raise PCTBackendError("楼梯中心线 corridor_radius_cells 不能为负数")
    if not (
        0
        <= logical_layer_first
        <= logical_layer_last
        < tomogram_index.layer_count
    ):
        raise PCTBackendError("楼梯中心线 logical_layer_band 超出 tomogram")
    if not math.isfinite(ground_z_origin_m):
        raise PCTBackendError("楼梯中心线 ground_z_origin_m 必须为有限数值")
    if not math.isfinite(height_step_m) or height_step_m <= 0.0:
        raise PCTBackendError("楼梯中心线 height_step_m 必须为有限正数")
    return _StairCenterlineProfile(
        path=path,
        sha256=profile_hash,
        asset_kind=asset_kind,
        anchors_sim_ground_xyz=anchors,
        lower_floor_approach_anchors_sim_ground_xyz=approach_anchors,
        lower_floor_approach_clearance=approach_clearance,
        sampling_spacing_m=sampling_spacing_m,
        corridor_radius_cells=corridor_radius_cells,
        logical_layer_first=logical_layer_first,
        logical_layer_last=logical_layer_last,
        ground_z_origin_m=ground_z_origin_m,
        height_step_m=height_step_m,
    )


def _runtime_stair_profile_anchors(
    profile: _StairCenterlineProfile,
) -> tuple[tuple[float, float, float], ...]:
    """
    @brief 组合一楼接近段与楼梯拓扑段，供跨层 Path 运行时拼接
    @param profile 已完成资产身份校验的楼梯中心线 profile
    @return 去掉公共入口重复点后的完整运行时锚点
    """

    approach = profile.lower_floor_approach_anchors_sim_ground_xyz
    if not approach:
        return profile.anchors_sim_ground_xyz
    return (*approach[:-1], *profile.anchors_sim_ground_xyz)


def _minimum_point_to_triangle_projection_distance(
    point_xy: np.ndarray,
    triangles_xy: np.ndarray,
) -> tuple[float, int]:
    """
    @brief 计算一个平面点到一组三角形 XY 投影的精确最小距离
    @param point_xy 待查询的二维点
    @param triangles_xy 形状为 M×3×2 的三角形投影
    @return 最小距离及其在输入三角形数组中的索引
    """

    point = np.asarray(point_xy, dtype=np.float64)
    triangles = np.asarray(triangles_xy, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError("point_xy 必须是二维点")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("triangles_xy 必须是 M×3×2")
    if len(triangles) == 0:
        raise ValueError("triangles_xy 不能为空")

    edge_start = triangles
    edge_end = np.roll(triangles, shift=-1, axis=1)
    edge_delta = edge_end - edge_start
    denominator = np.sum(edge_delta * edge_delta, axis=2)
    numerator = np.sum((point - edge_start) * edge_delta, axis=2)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1.0e-18,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    nearest = edge_start + ratio[:, :, None] * edge_delta
    edge_distance = np.linalg.norm(nearest - point, axis=2).min(axis=1)

    signed_cross = (
        edge_delta[:, :, 0] * (point[1] - edge_start[:, :, 1])
        - edge_delta[:, :, 1] * (point[0] - edge_start[:, :, 0])
    )
    doubled_area = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    nondegenerate = np.abs(doubled_area) > 1.0e-12
    inside = nondegenerate & (
        np.all(signed_cross >= -1.0e-10, axis=1)
        | np.all(signed_cross <= 1.0e-10, axis=1)
    )
    distances = np.where(inside, 0.0, edge_distance)
    triangle_index = int(np.argmin(distances))
    return float(distances[triangle_index]), triangle_index


def _audit_stair_approach_clearance(
    *,
    approach_anchors: Sequence[Sequence[float]],
    collision_vertices_sim: np.ndarray,
    collision_faces: np.ndarray,
    contract: _StairApproachClearanceContract,
) -> dict[str, object]:
    """
    @brief 用连续位置和转角 yaw 扫掠审计接近段双圆柱净空
    @param approach_anchors 仿真世界系的一楼接近段地面锚点
    @param collision_vertices_sim 仿真世界系 collision PLY 顶点
    @param collision_faces collision PLY 三角面顶点索引
    @param contract 双圆柱尺寸、障碍高度带与最小净空合同
    @return 可序列化的最小净空、最危险位姿及通过状态
    """

    points = np.asarray(
        [
            _finite_xyz(point, field_name="lower_floor_approach")
            for point in approach_anchors
        ],
        dtype=np.float64,
    )
    vertices = np.asarray(collision_vertices_sim, dtype=np.float64)
    faces = np.asarray(collision_faces, dtype=np.int64)
    if len(points) < 3:
        raise ValueError("接近段净空审计至少需要三个锚点")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("collision_vertices_sim 必须是 N×3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("collision_faces 必须是 M×3")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("collision_faces 索引超出顶点范围")

    triangles = vertices[faces]
    audit_padding = (
        contract.double_cylinder_radius_m
        + contract.double_cylinder_offset_m
        + 0.05
    )
    lower_xy = points[:, :2].min(axis=0) - audit_padding
    upper_xy = points[:, :2].max(axis=0) + audit_padding
    relevant = (
        (triangles[:, :, 2].max(axis=1)
         >= contract.obstacle_minimum_z_m)
        & (triangles[:, :, 2].min(axis=1)
           <= contract.obstacle_maximum_z_m)
        & (triangles[:, :, 0].max(axis=1) >= lower_xy[0])
        & (triangles[:, :, 0].min(axis=1) <= upper_xy[0])
        & (triangles[:, :, 1].max(axis=1) >= lower_xy[1])
        & (triangles[:, :, 1].min(axis=1) <= upper_xy[1])
    )
    relevant_face_indices = np.flatnonzero(relevant)
    relevant_triangles = triangles[relevant]
    if len(relevant_triangles) == 0:
        raise ValueError("接近段净空审计范围内没有 collision 三角面")

    segment_yaws: list[float] = []
    for start, end in zip(points, points[1:]):
        delta = end[:2] - start[:2]
        if float(delta @ delta) <= 1.0e-18:
            raise ValueError("接近段包含退化的平面线段")
        segment_yaws.append(math.atan2(float(delta[1]), float(delta[0])))

    query_positions: list[np.ndarray] = []
    query_yaws: list[float] = []
    for segment_index, (start, end) in enumerate(
        zip(points, points[1:])
    ):
        segment_length = math.dist(start[:2], end[:2])
        divisions = max(
            1,
            int(math.ceil(segment_length / contract.sample_spacing_m)),
        )
        for sample_index in range(divisions):
            ratio = sample_index / divisions
            query_positions.append(start + ratio * (end - start))
            query_yaws.append(segment_yaws[segment_index])
    query_positions.append(points[-1])
    query_yaws.append(segment_yaws[-1])

    # 折线锚点只是平滑曲线的离散标定；在每个锚点补查前后切向之间的
    # 最短角扫掠，防止只检查两条边却漏掉 B-spline 的中间朝向。
    for point_index in range(1, len(points) - 1):
        previous_yaw = segment_yaws[point_index - 1]
        next_yaw = segment_yaws[point_index]
        yaw_delta = math.atan2(
            math.sin(next_yaw - previous_yaw),
            math.cos(next_yaw - previous_yaw),
        )
        divisions = max(
            1,
            int(
                math.ceil(
                    abs(yaw_delta) / contract.maximum_yaw_step_rad
                )
            ),
        )
        for sample_index in range(divisions + 1):
            query_positions.append(points[point_index])
            query_yaws.append(
                previous_yaw + yaw_delta * sample_index / divisions
            )

    minimum_distance = math.inf
    minimum_query_index = -1
    minimum_face_index = -1
    minimum_side = ""
    minimum_center = np.zeros(2, dtype=np.float64)
    for query_index, (position, yaw) in enumerate(
        zip(query_positions, query_yaws)
    ):
        heading = np.asarray((math.cos(yaw), math.sin(yaw)))
        for side, sign in (("front", 1.0), ("rear", -1.0)):
            cylinder_center = (
                position[:2]
                + sign * contract.double_cylinder_offset_m * heading
            )
            distance, local_face_index = (
                _minimum_point_to_triangle_projection_distance(
                    cylinder_center,
                    relevant_triangles[:, :, :2],
                )
            )
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_query_index = query_index
                minimum_face_index = int(
                    relevant_face_indices[local_face_index]
                )
                minimum_side = side
                minimum_center = cylinder_center

    minimum_position = query_positions[minimum_query_index]
    minimum_yaw = query_yaws[minimum_query_index]
    collision_free = (
        minimum_distance + 1.0e-9
        >= contract.minimum_surface_clearance_m
    )
    return {
        "enabled": True,
        "collision_free": collision_free,
        "approach_anchor_count": len(points),
        "query_pose_count": len(query_positions),
        "relevant_collision_face_count": len(relevant_triangles),
        "double_cylinder_radius_m": (
            contract.double_cylinder_radius_m
        ),
        "double_cylinder_offset_m": (
            contract.double_cylinder_offset_m
        ),
        "minimum_required_surface_clearance_m": (
            contract.minimum_surface_clearance_m
        ),
        "minimum_surface_clearance_m": minimum_distance,
        "clearance_margin_over_radius_m": (
            minimum_distance - contract.double_cylinder_radius_m
        ),
        "minimum_base_xyz": tuple(float(value) for value in minimum_position),
        "minimum_yaw_rad": float(minimum_yaw),
        "minimum_cylinder_side": minimum_side,
        "minimum_cylinder_center_xy": tuple(
            float(value) for value in minimum_center
        ),
        "minimum_collision_face_index": minimum_face_index,
        "obstacle_z_band_m": (
            contract.obstacle_minimum_z_m,
            contract.obstacle_maximum_z_m,
        ),
    }


def _sample_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """用顶点、边中点和面中心覆盖薄墙，避免只看顶点漏栅格。"""

    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise PCTBackendError("collision PLY 顶点必须是 N×3")
    if faces_array.size == 0:
        return vertices_array
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise PCTBackendError("collision PLY 三角面必须是 M×3")
    triangles = vertices_array[faces_array]
    return np.concatenate(
        (
            vertices_array,
            triangles.mean(axis=1),
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ),
        axis=0,
    )


def _stair_profile_protection_mask(
    *,
    profile: _StairCenterlineProfile | None,
    tomogram_index: _UpstreamTomogramIndex,
    sim_to_pct: Callable[[Sequence[float]], tuple[float, float, float]],
) -> np.ndarray:
    """保护标定楼梯走廊，软净空不得改变其跨层通行代价。"""

    shape = tomogram_index.traversability.shape
    protected = np.zeros(shape, dtype=bool)
    if profile is None:
        return protected

    sampled: list[tuple[float, float, float]] = []
    anchors = profile.anchors_sim_ground_xyz
    for first, second in zip(anchors, anchors[1:]):
        length = math.dist(first, second)
        divisions = max(1, int(math.ceil(length / profile.sampling_spacing_m)))
        sampled.extend(
            tuple(
                first[axis]
                + (second[axis] - first[axis]) * (index / divisions)
                for axis in range(3)
            )
            for index in range(divisions)
        )
    sampled.append(anchors[-1])

    radius = profile.corridor_radius_cells
    for point_sim in sampled:
        point_pct = sim_to_pct(point_sim)
        if point_pct[2] <= profile.ground_z_origin_m:
            layer = profile.logical_layer_first
        else:
            layer = profile.logical_layer_first + int(
                math.ceil(
                    (point_pct[2] - profile.ground_z_origin_m)
                    / profile.height_step_m
                )
            )
        layer = min(
            max(layer, profile.logical_layer_first),
            profile.logical_layer_last,
        )
        grid_x = int(
            np.rint(
                (point_pct[0] - float(tomogram_index.center[0]))
                / tomogram_index.resolution
            )
        ) + tomogram_index.grid_dim_x // 2
        grid_y = int(
            np.rint(
                (point_pct[1] - float(tomogram_index.center[1]))
                / tomogram_index.resolution
            )
        ) + tomogram_index.grid_dim_y // 2
        x_first = max(0, grid_x - radius)
        x_last = min(tomogram_index.grid_dim_x, grid_x + radius + 1)
        y_first = max(0, grid_y - radius)
        y_last = min(tomogram_index.grid_dim_y, grid_y + radius + 1)
        if x_first < x_last and y_first < y_last:
            protected[layer, x_first:x_last, y_first:y_last] = True
    return protected


def _build_body_clearance_overlay(
    *,
    tomogram_index: _UpstreamTomogramIndex,
    collision_vertices: np.ndarray,
    collision_faces: np.ndarray,
    profile: _StairCenterlineProfile | None,
    sim_to_pct: Callable[[Sequence[float]], tuple[float, float, float]],
    minimum_height_m: float,
    maximum_height_m: float,
    radius_m: float,
    maximum_cost: float,
    power: float,
) -> _BodyClearanceOverlay:
    """从 PLY 机身高度带生成 A* 软净空代价，不改原始 gateway。"""

    samples = _sample_triangle_mesh(collision_vertices, collision_faces)
    resolution = tomogram_index.resolution
    dim_x = tomogram_index.grid_dim_x
    dim_y = tomogram_index.grid_dim_y
    sample_x = np.rint(
        (samples[:, 0] - float(tomogram_index.center[0])) / resolution
    ).astype(np.int64) + dim_x // 2
    sample_y = np.rint(
        (samples[:, 1] - float(tomogram_index.center[1])) / resolution
    ).astype(np.int64) + dim_y // 2
    valid = (
        np.isfinite(samples).all(axis=1)
        & (sample_x >= 0)
        & (sample_x < dim_x)
        & (sample_y >= 0)
        & (sample_y < dim_y)
    )
    sample_x = sample_x[valid]
    sample_y = sample_y[valid]
    sample_z = samples[valid, 2]
    protected = _stair_profile_protection_mask(
        profile=profile,
        tomogram_index=tomogram_index,
        sim_to_pct=sim_to_pct,
    )

    traversability = np.asarray(
        tomogram_index.traversability,
        dtype=np.float32,
    ).copy()
    changed_cell_count = 0
    obstacle_cell_count = 0
    changed_layer_count = 0
    maximum_applied_cost = 0.0
    for layer in range(tomogram_index.layer_count):
        base_cost = traversability[layer]
        ground_layer = tomogram_index.ground[layer]
        ground_seed = (
            np.isfinite(ground_layer)
            & (ground_layer > -50.0)
            & np.isfinite(base_cost)
            & (base_cost <= _UPSTREAM_ASTAR_COST_THRESHOLD)
        )
        if not np.any(ground_seed) or len(sample_z) == 0:
            continue
        _, nearest_seed = distance_transform_edt(
            ~ground_seed,
            return_indices=True,
        )
        nearest_ground = ground_layer[tuple(nearest_seed)]
        relative_height = sample_z - nearest_ground[sample_x, sample_y]
        in_body_band = (
            (relative_height >= minimum_height_m)
            & (relative_height <= maximum_height_m)
        )
        obstacle = np.zeros((dim_x, dim_y), dtype=bool)
        obstacle[sample_x[in_body_band], sample_y[in_body_band]] = True
        layer_obstacle_count = int(np.count_nonzero(obstacle))
        obstacle_cell_count += layer_obstacle_count
        if layer_obstacle_count == 0:
            continue

        clearance = distance_transform_edt(~obstacle) * resolution
        normalized_deficit = np.clip(
            (radius_m - clearance) / radius_m,
            0.0,
            1.0,
        )
        penalty = maximum_cost * np.power(normalized_deficit, power)
        editable = (
            np.isfinite(base_cost)
            & (base_cost <= _UPSTREAM_ASTAR_COST_THRESHOLD)
            & ~protected[layer]
        )
        before = base_cost.copy()
        base_cost[editable] = np.maximum(
            base_cost[editable],
            penalty[editable],
        )
        changed = editable & (base_cost > before + 1.0e-6)
        layer_changed_count = int(np.count_nonzero(changed))
        if layer_changed_count > 0:
            changed_layer_count += 1
            changed_cell_count += layer_changed_count
            maximum_applied_cost = max(
                maximum_applied_cost,
                float(np.max(base_cost[changed])),
            )

    gradient_x = np.zeros_like(traversability, dtype=np.float32)
    gradient_y = np.zeros_like(traversability, dtype=np.float32)
    gradient_x[:, 1:-1, :] = (
        traversability[:, 2:, :] - traversability[:, :-2, :]
    )
    gradient_y[:, :, 1:-1] = (
        traversability[:, :, 2:] - traversability[:, :, :-2]
    )
    return _BodyClearanceOverlay(
        traversability=np.ascontiguousarray(traversability),
        gradient_x=np.ascontiguousarray(gradient_x),
        gradient_y=np.ascontiguousarray(gradient_y),
        report={
            "enabled": True,
            "minimum_height_m": float(minimum_height_m),
            "maximum_height_m": float(maximum_height_m),
            "radius_m": float(radius_m),
            "maximum_cost": float(maximum_cost),
            "power": float(power),
            "sample_count": int(len(sample_z)),
            "obstacle_cell_count": obstacle_cell_count,
            "changed_cell_count": changed_cell_count,
            "changed_layer_count": changed_layer_count,
            "protected_stair_cell_count": int(np.count_nonzero(protected)),
            "maximum_applied_cost": maximum_applied_cost,
            "gateway_source": "original_traversability",
        },
    )


def _splice_stair_centerline(
    path_sim: Sequence[Sequence[float]],
    *,
    profile: _StairCenterlineProfile,
    match_tolerance_m: float,
) -> tuple[tuple[tuple[float, float, float], ...], dict[str, object]]:
    """只替换已被原生 A* 穿过的楼梯段，保留两层平地选路。"""

    points = tuple(
        _finite_xyz(point, field_name="upstream_cross_layer_path_sim")
        for point in path_sim
    )
    report: dict[str, object] = {
        "applied": False,
        "reason": "path_has_fewer_than_two_points",
        "raw_point_count": len(points),
        "profile_path": str(profile.path),
        "profile_sha256": profile.sha256,
        "profile_asset_kind": profile.asset_kind,
        "match_tolerance_m": float(match_tolerance_m),
    }
    if len(points) < 2:
        return points, report

    ascending = points[-1][2] >= points[0][2]
    runtime_anchors = _runtime_stair_profile_anchors(profile)
    anchors = (
        runtime_anchors
        if ascending
        else tuple(reversed(runtime_anchors))
    )
    start_index = min(
        range(len(points) - 1),
        key=lambda index: math.dist(points[index], anchors[0]),
    )
    end_index = min(
        range(start_index + 1, len(points)),
        key=lambda index: math.dist(points[index], anchors[-1]),
    )
    start_error = math.dist(points[start_index], anchors[0])
    end_error = math.dist(points[end_index], anchors[-1])
    report.update(
        {
            "reason": "profile_endpoints_not_matched",
            "ascending": ascending,
            "raw_start_index": start_index,
            "raw_end_index": end_index,
            "start_match_error_m": start_error,
            "end_match_error_m": end_error,
        }
    )
    if start_error > match_tolerance_m or end_error > match_tolerance_m:
        return points, report

    combined = (*points[:start_index], *anchors, *points[end_index + 1:])
    refined: list[tuple[float, float, float]] = []
    for point in combined:
        if not refined or math.dist(refined[-1], point) > 1.0e-9:
            refined.append(point)
    # profile 只能约束中间楼梯几何，不能改变请求的精确起终点。楼梯 smoke
    # 的目标会在标定出口外再延伸约 2 cm；若末点恰好也是最佳 profile 匹配点，
    # 仍必须把原始终点接回去，交由后续重采样和 PLY 投影保持消息合同。
    if math.dist(refined[0], points[0]) <= 1.0e-9:
        refined[0] = points[0]
    else:
        refined.insert(0, points[0])
    if math.dist(refined[-1], points[-1]) <= 1.0e-9:
        refined[-1] = points[-1]
    else:
        refined.append(points[-1])
    profile_start_index: int | None = None
    for candidate in range(len(refined) - len(anchors) + 1):
        if all(
            math.dist(refined[candidate + offset], anchor) <= 1.0e-9
            for offset, anchor in enumerate(anchors)
        ):
            profile_start_index = candidate
            break
    if profile_start_index is None:
        raise PCTBackendError("楼梯中心线替换后无法定位完整 profile")
    profile_end_index = profile_start_index + len(anchors) - 1
    report.update(
        {
            "applied": True,
            "reason": "calibrated_stair_centerline",
            "profile_anchor_count": len(anchors),
            "stair_anchor_count": len(profile.anchors_sim_ground_xyz),
            "lower_floor_approach_anchor_count": len(
                profile.lower_floor_approach_anchors_sim_ground_xyz
            ),
            "refined_point_count": len(refined),
            "refined_profile_start_index": profile_start_index,
            "refined_profile_end_index": profile_end_index,
        }
    )
    return tuple(refined), report


class UpstreamTomogramBackend:
    """直接调用固定提交的官方 TomogramPlanner，不提供 compatible 回退。"""

    def __init__(
        self,
        config: PCTBackendConfig,
        *,
        planner_module: ModuleType | None = None,
        coordinate_module: ModuleType | None = None,
    ) -> None:
        self.config = _validated_config(config)
        if self.config.backend_kind != "upstream":
            raise ValueError(
                "UpstreamTomogramBackend 要求 planner.backend_kind=upstream"
            )
        if self.config.collision_ply_path is None:
            raise ValueError(
                "官方 PCT Path 必须配置 collision PLY，"
                "用于统一输出真实地面高度"
            )
        if not math.isclose(
            self.config.slice_query_root_to_floor_m,
            self.config.goal_base_to_ground_m,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "upstream backend 要求 slice_query_root_to_floor_m 与 "
                "goal_base_to_ground_m 相同；base→ground 只能转换一次"
            )
        source_root = self.config.upstream_source_root
        if source_root is None:
            raise ValueError("planner.upstream_source_root 不能为空")
        self._source_root = _validate_pinned_source(source_root)
        self._coordinate_module = coordinate_module or (
            _load_project_coordinate_module(self.config.project_root)
        )
        with _UPSTREAM_IMPORT_LOCK, _temporary_numpy_pickle_aliases():
            self._tomogram_index = _UpstreamTomogramIndex(
                self.config.tomogram_path
            )
            self._planner_module = (
                planner_module
                or _load_upstream_planner_module(self._source_root)
            )

        support_module = __import__(
            "source.scene.placement_support",
            fromlist=("load_binary_triangle_ply",),
        )
        vertices, faces = support_module.load_binary_triangle_ply(
            self.config.collision_ply_path
        )
        self._ground_projector = TriangleGroundProjector(
            vertices,
            faces,
            maximum_hint_error_m=self.config.ground_projection_max_z_error_m,
        )
        self._stair_profile = (
            None
            if self.config.upstream_stair_profile_path is None
            else _load_stair_centerline_profile(
                self.config.upstream_stair_profile_path,
                config=self.config,
                tomogram_index=self._tomogram_index,
            )
        )
        self._stair_approach_clearance_report: dict[str, object] = {
            "enabled": False,
            "collision_free": True,
            "reason": "lower_floor_approach_disabled",
        }
        if (
            self._stair_profile is not None
            and self._stair_profile
            .lower_floor_approach_anchors_sim_ground_xyz
        ):
            clearance_contract = (
                self._stair_profile.lower_floor_approach_clearance
            )
            if clearance_contract is None:
                raise PCTBackendError("一楼接近段缺少双圆柱净空合同")
            vertices_sim = np.asarray(
                [self._pct_to_sim(vertex) for vertex in vertices],
                dtype=np.float64,
            )
            try:
                self._stair_approach_clearance_report = (
                    _audit_stair_approach_clearance(
                        approach_anchors=(
                            self._stair_profile
                            .lower_floor_approach_anchors_sim_ground_xyz
                        ),
                        collision_vertices_sim=vertices_sim,
                        collision_faces=faces,
                        contract=clearance_contract,
                    )
                )
            except ValueError as exc:
                raise PCTBackendError(
                    f"一楼楼梯接近段净空审计失败：{exc}"
                ) from exc
            if not bool(
                self._stair_approach_clearance_report["collision_free"]
            ):
                raise PCTBackendError(
                    "一楼楼梯接近段不满足双圆柱净空合同："
                    f"{self._stair_approach_clearance_report}"
                )
        self._body_clearance_overlay = (
            _build_body_clearance_overlay(
                tomogram_index=self._tomogram_index,
                collision_vertices=vertices,
                collision_faces=faces,
                profile=self._stair_profile,
                sim_to_pct=self._sim_to_pct,
                minimum_height_m=self.config.body_obstacle_min_height_m,
                maximum_height_m=self.config.body_obstacle_max_height_m,
                radius_m=self.config.upstream_body_clearance_radius_m,
                maximum_cost=(
                    self.config.upstream_body_clearance_maximum_cost
                ),
                power=self.config.upstream_body_clearance_power,
            )
            if self.config.upstream_body_clearance_enabled
            else None
        )
        self._body_clearance_report = (
            self._body_clearance_overlay.report
            if self._body_clearance_overlay is not None
            else {
                "enabled": False,
                "gateway_source": "original_traversability",
            }
        )
        if self._body_clearance_overlay is not None:
            self._tomogram_index.configure_shortcut_body_obstacles(
                self._body_clearance_overlay.traversability,
                obstacle_surface_cost=(
                    self.config.upstream_body_clearance_maximum_cost
                ),
            )
        self._planner_requires_rebuild = False
        self._planner = self._create_loaded_planner()
        self._cancel_event = threading.Event()
        self._plan_prepared = False

    def prepare_plan(
        self,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """为一个 worker job 建立不可被起始 reset 吞掉的取消代际。"""

        self._plan_prepared = False
        self._cancel_event.clear()
        self._rebuild_planner_after_native_failure()
        self._reset_native_cancellation()
        if cancel_event is not None and cancel_event.is_set():
            # CANCEL 可能在地图重建或 native reset 期间到达；reset 可能已经
            # 清掉第一次原子请求，因此必须在同一个 job event 上补发一次。
            self.cancel_current_plan()
            raise PCTBackendError("官方 PCT 规划在准备 native core 时已取消")
        self._plan_prepared = True

    def plan(
        self,
        *,
        start_base_xyz: Sequence[float],
        goal_base_xyz: Sequence[float],
        goal_yaw: float,
    ) -> PCTBackendPlan:
        """以 base 位姿调用官方核心，并返回仿真世界系地面路径。"""

        prepared = self._plan_prepared
        self._plan_prepared = False
        if not prepared:
            # 保留 backend 的直接调用能力；ROS worker 路径会先显式握手。
            self.prepare_plan()
            self._plan_prepared = False
        start = _finite_xyz(start_base_xyz, field_name="start_base_xyz")
        goal = _finite_xyz(goal_base_xyz, field_name="goal_base_xyz")
        yaw = float(goal_yaw)
        if not math.isfinite(yaw):
            raise ValueError("goal_yaw 必须是有限数值")

        start_hint_pct = self._sim_to_pct(
            (start[0], start[1], start[2] - self.config.goal_base_to_ground_m)
        )
        goal_hint_pct = self._sim_to_pct(
            (goal[0], goal[1], goal[2] - self.config.goal_base_to_ground_m)
        )
        start_ground_pct, start_error, start_face = self._project_endpoint(
            start_hint_pct,
            maximum_error_m=(
                self.config.start_ground_projection_max_z_error_m
            ),
            label="start",
        )
        goal_ground_pct, goal_error, goal_face = self._project_endpoint(
            goal_hint_pct,
            maximum_error_m=(
                self.config.terminal_ground_projection_max_z_error_m
            ),
            label="goal",
        )
        start_match = self._tomogram_index.endpoint_layer(
            point_xyz=start_ground_pct,
            maximum_height_error_m=(
                self.config.upstream_endpoint_layer_max_z_error_m
            ),
            label="start",
        )
        goal_match = self._tomogram_index.endpoint_layer(
            point_xyz=goal_ground_pct,
            maximum_height_error_m=(
                self.config.upstream_endpoint_layer_max_z_error_m
            ),
            label="goal",
        )
        self._tomogram_index.validate_cross_layer_gateway(
            start_match,
            goal_match,
        )
        self._planner.start_idx[0] = start_match.layer
        self._planner.end_idx[0] = goal_match.layer

        if self._cancel_event.is_set():
            raise PCTBackendError("官方 PCT 规划在执行前已取消")
        grid_delta = max(
            abs(start_match.grid_x - goal_match.grid_x),
            abs(start_match.grid_y - goal_match.grid_y),
        )
        terminal_connector = (
            start_match.layer == goal_match.layer and grid_delta <= 1
        )
        if terminal_connector:
            if math.dist(start_ground_pct, goal_ground_pct) <= 1.0e-9:
                raise PCTNoPathError("官方 PCT 起终点已位于同一地面网格点")
            raw_points = (start_ground_pct, goal_ground_pct)
            trajectory_layers: tuple[int, ...] = ()
            astar_path_matrix: tuple[tuple[int, int, int], ...] = ()
        else:
            try:
                astar_result = _run_native_astar(
                    self._planner,
                    start_match,
                    goal_match,
                )
                native_cancelled = bool(
                    self._planner.planner.was_cancelled()
                )
            except Exception as exc:
                self._planner_requires_rebuild = True
                raise PCTBackendError(
                    "官方 PCT TomogramPlanner 执行失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if self._cancel_event.is_set() or native_cancelled:
                raise PCTBackendError("官方 PCT 规划结果已被新代际取消")
            if astar_result is None:
                raise PCTNoPathError("官方 PCT 未找到路径")
            try:
                raw_points, trajectory_layers = (
                    self._tomogram_index.native_astar_ground_points(
                        astar_result
                    )
                )
                astar_path_matrix = tuple(
                    tuple(int(value) for value in row)
                    for row in np.rint(
                        np.asarray(astar_result, dtype=np.float64)
                    ).astype(np.int64)
                )
            except Exception:
                # native core 已经修改过内部搜索状态；即使只是输出损坏，
                # 下一请求也不能继续复用这份对象。
                self._planner_requires_rebuild = True
                raise

        start_prepended = math.dist(
            raw_points[0][:2], start_ground_pct[:2]
        ) > 1.0e-9
        if start_prepended:
            raw_points = (start_ground_pct, *raw_points)
        else:
            raw_points = (start_ground_pct, *raw_points[1:])
        goal_appended = math.dist(
            raw_points[-1][:2], goal_ground_pct[:2]
        ) > 1.0e-9
        if goal_appended:
            raw_points = (*raw_points, goal_ground_pct)
        else:
            raw_points = (*raw_points[:-1], goal_ground_pct)

        shortcut_points = raw_points
        shortcut_report: dict[str, object] = {
            "applied": False,
            "reason": "terminal_connector",
            "raw_point_count": len(raw_points),
            "shortcut_point_count": len(raw_points),
            "logical_layer": start_match.layer,
            "clearance_m": (
                self.config.upstream_same_layer_shortcut_clearance_m
            ),
            "grid_cell_cover_margin_m": (
                self._tomogram_index.resolution / math.sqrt(2.0)
            ),
            "maximum_segment_m": (
                self.config.upstream_same_layer_shortcut_max_segment_m
            ),
            "minimum_blocked_cell_center_distance_m": None,
        }
        stair_centerline_report: dict[str, object] = {
            "applied": False,
            "reason": "same_layer_plan",
            "raw_point_count": len(raw_points),
        }
        if not terminal_connector:
            if start_match.layer != goal_match.layer:
                if self._stair_profile is None:
                    shortcut_report["reason"] = "cross_layer_path_preserved"
                    stair_centerline_report["reason"] = "profile_disabled"
                else:
                    refined_sim, stair_centerline_report = (
                        _splice_stair_centerline(
                            tuple(
                                self._pct_to_sim(point) for point in raw_points
                            ),
                            profile=self._stair_profile,
                            match_tolerance_m=(
                                self.config
                                .upstream_stair_profile_match_tolerance_m
                            ),
                        )
                    )
                    if not bool(stair_centerline_report["applied"]):
                        raise PCTNoPathError(
                            "upstream PCT 跨层路径没有同时经过标定楼梯入口和出口："
                            f"{stair_centerline_report}"
                        )
                    refined_pct = tuple(
                        self._sim_to_pct(point) for point in refined_sim
                    )
                    shortcut_points, shortcut_report = (
                        self._tomogram_index
                        .shortcut_cross_layer_floor_segments(
                            refined_pct,
                            profile_start_index=int(
                                stair_centerline_report[
                                    "refined_profile_start_index"
                                ]
                            ),
                            profile_end_index=int(
                                stair_centerline_report[
                                    "refined_profile_end_index"
                                ]
                            ),
                            start_layer=start_match.layer,
                            goal_layer=goal_match.layer,
                            clearance_m=(
                                self.config
                                .upstream_same_layer_shortcut_clearance_m
                            ),
                            maximum_segment_m=(
                                self.config
                                .upstream_same_layer_shortcut_max_segment_m
                            ),
                        )
                    )
            elif any(
                layer != start_match.layer for layer in trajectory_layers
            ):
                shortcut_report["reason"] = (
                    "logical_layer_transition_preserved"
                )
            else:
                shortcut_points, shortcut_report = (
                    self._tomogram_index.shortcut_same_layer_path(
                        raw_points,
                        layer=start_match.layer,
                        clearance_m=(
                            self.config.upstream_same_layer_shortcut_clearance_m
                        ),
                        maximum_segment_m=(
                            self.config.upstream_same_layer_shortcut_max_segment_m
                        ),
                    )
                )

        sampled_pct = _sample_polyline(
            shortcut_points,
            spacing_m=self.config.path_sample_spacing_m,
        )
        try:
            projected_pct, projection_reports = (
                self._ground_projector.project_path(sampled_pct)
            )
        except ValueError as exc:
            raise PCTBackendError(
                f"官方 PCT 路径的 collision PLY 地面投影失败：{exc}"
            ) from exc
        points_sim = tuple(self._pct_to_sim(point) for point in projected_pct)
        maximum_index = max(
            range(len(projection_reports)),
            key=lambda index: projection_reports[index].hint_error_m,
        )
        maximum_projection = projection_reports[maximum_index]
        metadata = {
            "backend_kind": "upstream",
            "upstream_repository": UPSTREAM_PCT_REPOSITORY,
            "upstream_commit": UPSTREAM_PCT_COMMIT,
            "upstream_archive_sha256": UPSTREAM_PCT_ARCHIVE_SHA256,
            "upstream_license": UPSTREAM_PCT_LICENSE,
            "upstream_patch_id": UPSTREAM_PCT_PATCH_ID,
            "upstream_patch_sha256": UPSTREAM_PCT_PATCH_SHA256,
            "upstream_native_cancel_supported": True,
            "upstream_native_gil_released": True,
            "upstream_native_astar_path_matrix": astar_path_matrix,
            "upstream_raw_path_3d_pct": raw_points,
            "upstream_shortcut_path_3d_pct": shortcut_points,
            "upstream_same_layer_shortcut_report": shortcut_report,
            "upstream_stair_centerline_report": stair_centerline_report,
            "upstream_stair_profile_sha256": (
                None
                if self._stair_profile is None
                else self._stair_profile.sha256
            ),
            "upstream_stair_approach_clearance_report": dict(
                self._stair_approach_clearance_report
            ),
            "upstream_trajectory_logical_layers": trajectory_layers,
            "upstream_core_mode": "offline_ele_planner_native_astar_ground",
            "upstream_astar_step_cost_weight": (
                self.config.upstream_astar_step_cost_weight
            ),
            "upstream_body_clearance_overlay": dict(
                self._body_clearance_report
            ),
            "upstream_gateway_cost_source": "original_traversability",
            "path_3d": points_sim,
            "height_semantics": "ground_height",
            "transport": "direct_in_process_ros2",
            "slice_query_root_to_floor_m": (
                self.config.slice_query_root_to_floor_m
            ),
            "goal_base_to_ground_m": self.config.goal_base_to_ground_m,
            "upstream_endpoint_layer_max_z_error_m": (
                self.config.upstream_endpoint_layer_max_z_error_m
            ),
            "upstream_tomogram_asset_kind": self._tomogram_index.asset_kind,
            "upstream_gateway_up_count": (
                self._tomogram_index.gateway_up_count
            ),
            "upstream_gateway_down_count": (
                self._tomogram_index.gateway_down_count
            ),
            "upstream_output_ground_normalization": (
                "terminal_connector_ground"
                if terminal_connector
                else "native_astar_tomogram_logical_layer_ground"
            ),
            "upstream_core_invoked": not terminal_connector,
            "start_layer": start_match.layer,
            "start_layer_ground_z": start_match.ground_z,
            "start_layer_cost": start_match.cost,
            "start_layer_height_error_m": start_match.height_error_m,
            "start_layer_grid_xy": (start_match.grid_x, start_match.grid_y),
            "goal_layer": goal_match.layer,
            "goal_layer_ground_z": goal_match.ground_z,
            "goal_layer_cost": goal_match.cost,
            "goal_layer_height_error_m": goal_match.height_error_m,
            "goal_layer_grid_xy": (goal_match.grid_x, goal_match.grid_y),
            "requested_start_prepended": start_prepended,
            "requested_goal_appended": goal_appended,
            "requested_goal_yaw": yaw,
            "start_ground_projection_error_m": start_error,
            "start_ground_projection_face_index": start_face,
            "terminal_ground_projection_error_m": goal_error,
            "terminal_ground_projection_face_index": goal_face,
            "ground_projection_max_hint_error_m": (
                maximum_projection.hint_error_m
            ),
            "ground_projection_max_hint_error_index": maximum_index,
            "ground_projection_max_hint_face_index": (
                maximum_projection.face_index
            ),
            "ground_projection_point_count": len(points_sim),
        }
        return PCTBackendPlan(points_xyz=points_sim, metadata=metadata)

    def cancel_current_plan(self) -> None:
        """同时取消 Python 结果代际与正在执行的 native A*。"""

        self._cancel_event.set()
        request_cancel = getattr(self._planner.planner, "request_cancel", None)
        if not callable(request_cancel):
            raise PCTBackendError(
                "官方 PCT native planner 缺少 request_cancel 取消接口"
            )
        request_cancel()

    def _reset_native_cancellation(self) -> None:
        """在每次新请求进入 native core 前建立独立取消代际。"""

        reset_cancellation = getattr(
            self._planner.planner,
            "reset_cancellation",
            None,
        )
        if not callable(reset_cancellation):
            raise PCTBackendError(
                "官方 PCT native planner 缺少 reset_cancellation 接口"
            )
        try:
            reset_cancellation()
        except Exception as exc:
            self._planner_requires_rebuild = True
            raise PCTBackendError(
                "官方 PCT native planner 重置取消代际失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _create_loaded_planner(self) -> object:
        """构造全新的官方 planner，并重新加载当前 tomogram。"""

        planner_config = _make_upstream_config(
            source_root=self._source_root,
            tomogram_path=self.config.tomogram_path,
            use_quintic=self.config.upstream_use_quintic,
            max_heading_rate=self.config.upstream_max_heading_rate,
            astar_step_cost_weight=(
                self.config.upstream_astar_step_cost_weight
            ),
        )
        with _UPSTREAM_IMPORT_LOCK, _temporary_numpy_pickle_aliases():
            try:
                planner = self._planner_module.TomogramPlanner(planner_config)
                planner.loadTomogram(self.config.tomogram_path.stem)
                overlay = self._body_clearance_overlay
                if overlay is not None:
                    init_planner = getattr(planner, "initPlanner", None)
                    if not callable(init_planner):
                        raise AttributeError(
                            "TomogramPlanner 缺少 initPlanner 净空接口"
                        )
                    init_planner(
                        overlay.traversability,
                        overlay.gradient_x,
                        overlay.gradient_y,
                        np.nan_to_num(
                            self._tomogram_index.ground,
                            nan=-100.0,
                        ),
                        np.nan_to_num(
                            self._tomogram_index.ceiling,
                            nan=1.0e6,
                        ),
                        gateway_trav=(
                            self._tomogram_index.traversability
                        ),
                    )
            except Exception as exc:
                raise PCTBackendError(
                    "官方 PCT TomogramPlanner 初始化或地图加载失败："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        _validate_loaded_planner_contract(planner)
        if int(planner.n_slice) != self._tomogram_index.layer_count:
            raise PCTBackendError(
                "官方 wrapper 与 adapter 读取的 tomogram layer 数不一致："
                f"{planner.n_slice} != "
                f"{self._tomogram_index.layer_count}"
            )
        return planner

    def _rebuild_planner_after_native_failure(self) -> None:
        """失败后的下一次规划必须从全新 core/map 状态开始。"""

        if not self._planner_requires_rebuild:
            return
        planner = self._create_loaded_planner()
        self._planner = planner
        self._planner_requires_rebuild = False

    def _project_endpoint(
        self,
        point_pct: tuple[float, float, float],
        *,
        maximum_error_m: float,
        label: str,
    ) -> tuple[tuple[float, float, float], float, int]:
        try:
            report = self._ground_projector.project(
                x=point_pct[0],
                y=point_pct[1],
                z_hint=point_pct[2],
            )
        except ValueError as exc:
            raise PCTBackendError(
                f"官方 PCT {label} 端点地面投影失败：{exc}"
            ) from exc
        if report.hint_error_m > maximum_error_m:
            raise PCTBackendError(
                f"官方 PCT {label} z 与 collision PLY 支撑面不一致："
                f"误差 {report.hint_error_m:.3f} m，超过 "
                f"{maximum_error_m:.3f} m"
            )
        return (
            (point_pct[0], point_pct[1], report.z),
            report.hint_error_m,
            report.face_index,
        )

    def _sim_to_pct(
        self,
        point: Sequence[float],
    ) -> tuple[float, float, float]:
        return tuple(
            float(value)
            for value in self._coordinate_module.sim_to_pct_xyz(
                point,
                **_coordinate_arguments(self.config),
            )
        )

    def _pct_to_sim(
        self,
        point: Sequence[float],
    ) -> tuple[float, float, float]:
        return tuple(
            float(value)
            for value in self._coordinate_module.pct_to_sim_xyz(
                point,
                **_coordinate_arguments(self.config),
            )
        )


def _coordinate_arguments(config: PCTBackendConfig) -> dict[str, float | str]:
    return {
        "coord_mode": config.coord_mode,
        "pct_offset_x": config.pct_offset_x,
        "pct_offset_y": config.pct_offset_y,
        "pct_offset_z": config.pct_offset_z,
        "pct_scale_x": config.pct_scale_x,
        "pct_scale_y": config.pct_scale_y,
        "pct_scale_z": config.pct_scale_z,
        "pct_rotation_x_rad": config.pct_rotation_x_rad,
        "pct_rotation_y_rad": config.pct_rotation_y_rad,
        "pct_rotation_z_rad": config.pct_rotation_z_rad,
    }


def _validate_pinned_source(source_root: Path) -> Path:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"planner.upstream_source_root 不存在或不是目录: {root}"
        )
    for relative_path, expected_hash in _PINNED_FILE_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"固定 PCT 源码缺少必需文件: {path}"
            )
        actual_hash = sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise PCTBackendError(
                "固定 PCT 源码内容与 commit+patch pin 不一致："
                f"{relative_path} sha256={actual_hash}"
            )
    return root


def _load_upstream_planner_module(source_root: Path) -> ModuleType:
    planner_root = source_root / "planner"
    scripts_root = planner_root / "scripts"
    library_root = planner_root / "lib"
    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    missing_extensions = [
        module_name
        for module_name in ("a_star", "traj_opt", "ele_planner", "py_map_manager")
        if not (library_root / f"{module_name}{extension_suffix}").is_file()
    ]
    if missing_extensions:
        raise PCTBackendError(
            "官方 PCT Python 扩展尚未为当前解释器构建："
            f"缺少 {', '.join(missing_extensions)}{extension_suffix}；"
            "禁止自动切回 compatible backend"
        )

    wrapper_path = scripts_root / "planner_wrapper.py"
    module_name = "_pct_scan_pinned_upstream_planner_wrapper"
    with _UPSTREAM_IMPORT_LOCK:
        _reject_conflicting_import("utils", scripts_root)
        _reject_conflicting_import("lib", planner_root)
        previous_path = list(sys.path)
        try:
            sys.path[:0] = [str(scripts_root), str(planner_root)]
            spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法创建官方 PCT module spec: {wrapper_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise PCTBackendError(
                "官方 PCT 扩展加载失败；请检查 Python ABI、GTSAM/OSQP "
                f"共享库搜索路径：{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            sys.path[:] = previous_path
    module_file = Path(str(module.__file__)).resolve()
    if source_root not in module_file.parents:
        raise PCTBackendError(
            f"官方 PCT wrapper 来自错误目录: {module_file}"
        )
    if not hasattr(module, "TomogramPlanner"):
        raise PCTBackendError("官方 PCT wrapper 缺少 TomogramPlanner")
    return module


def _reject_conflicting_import(module_name: str, expected_parent: Path) -> None:
    existing = sys.modules.get(module_name)
    if existing is None:
        return
    module_file = getattr(existing, "__file__", None)
    module_paths = tuple(getattr(existing, "__path__", ()))
    candidates = []
    if module_file:
        candidates.append(Path(str(module_file)).resolve())
    candidates.extend(Path(str(path)).resolve() for path in module_paths)
    if not any(
        candidate == expected_parent or expected_parent in candidate.parents
        for candidate in candidates
    ):
        raise PCTBackendError(
            f"Python 模块名 {module_name!r} 已被其他目录占用，"
            "拒绝加载来源不确定的官方 PCT 扩展"
        )


def _make_upstream_config(
    *,
    source_root: Path,
    tomogram_path: Path,
    use_quintic: bool,
    max_heading_rate: float,
    astar_step_cost_weight: float,
) -> SimpleNamespace:
    if tomogram_path.suffix != ".pickle":
        raise ValueError("官方 PCT tomogram 文件必须使用 .pickle 后缀")
    relative_parent = os.path.relpath(tomogram_path.parent, source_root)
    # 官方 wrapper 通过 rsg_root + tomo_dir 拼接路径，因此保留前导斜杠。
    tomo_dir = f"/{relative_parent.replace(os.sep, '/')}/"
    return SimpleNamespace(
        planner=SimpleNamespace(
            use_quintic=bool(use_quintic),
            max_heading_rate=float(max_heading_rate),
            astar_step_cost_weight=float(astar_step_cost_weight),
        ),
        wrapper=SimpleNamespace(tomo_dir=tomo_dir),
    )


def _validate_loaded_planner_contract(planner: object) -> None:
    for attribute in (
        "plan",
        "start_idx",
        "end_idx",
        "n_slice",
        "slice_h0",
        "slice_dh",
    ):
        if not hasattr(planner, attribute):
            raise PCTBackendError(
                f"官方 PCT TomogramPlanner 缺少接口属性 {attribute}"
            )
    n_slice = int(planner.n_slice)
    slice_h0 = float(planner.slice_h0)
    slice_dh = float(planner.slice_dh)
    if n_slice < 1 or not math.isfinite(slice_h0):
        raise PCTBackendError("官方 PCT tomogram slice 元数据非法")
    if not math.isfinite(slice_dh) or slice_dh <= 0.0:
        raise PCTBackendError("官方 PCT tomogram slice_dh 必须为有限正数")
    for label, value in (("start_idx", planner.start_idx), ("end_idx", planner.end_idx)):
        if np.asarray(value).shape != (3,):
            raise PCTBackendError(f"官方 PCT {label} 必须是长度 3 的索引")
    native_planner = getattr(planner, "planner", None)
    for method_name in (
        "plan",
        "get_path_finder",
        "request_cancel",
        "reset_cancellation",
        "was_cancelled",
        "get_last_search_status",
        "get_expanded_node_count",
    ):
        if not callable(getattr(native_planner, method_name, None)):
            raise PCTBackendError(
                "官方 PCT OfflineElePlanner 缺少接口方法 "
                f"{method_name}"
            )


def _run_native_astar(
    planner: object,
    start: _EndpointLayerMatch,
    goal: _EndpointLayerMatch,
) -> object | None:
    """只运行官方 OfflineElePlanner A*；连续局部优化由 SCAN 负责。"""

    # 官方 Search 的输入顺序是 [layer, col, row]，而结果矩阵顺序是
    # [layer, row, col]。直接使用已做边界检查的 endpoint match，避免
    # 再次浮点换算后把越界索引送入没有边界保护的 C++ Search。
    planner.start_idx[:] = np.asarray(
        (start.layer, start.grid_y, start.grid_x),
        dtype=np.int32,
    )
    planner.end_idx[:] = np.asarray(
        (goal.layer, goal.grid_y, goal.grid_x),
        dtype=np.int32,
    )
    native_planner = getattr(planner, "planner", None)
    native_plan = getattr(native_planner, "plan", None)
    get_path_finder = getattr(native_planner, "get_path_finder", None)
    if not callable(native_plan) or not callable(get_path_finder):
        raise PCTBackendError("官方 PCT OfflineElePlanner 接口不完整")
    if not bool(native_plan(planner.start_idx, planner.end_idx, False)):
        return None
    path_finder = get_path_finder()
    get_result_matrix = getattr(path_finder, "get_result_matrix", None)
    if not callable(get_result_matrix):
        raise PCTBackendError("官方 PCT A* 缺少 get_result_matrix")
    return get_result_matrix()


@contextmanager
def _temporary_numpy_pickle_aliases() -> Iterator[None]:
    """局部兼容 NumPy 2 写入、ROS NumPy 1 读取的 pickle 路径。"""

    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric

    aliases = (
        ("numpy._core", numpy_core),
        ("numpy._core.numeric", numpy_core_numeric),
    )
    inserted: list[str] = []
    for name, module in aliases:
        if name not in sys.modules:
            sys.modules[name] = module
            inserted.append(name)
    try:
        yield
    finally:
        for name in reversed(inserted):
            sys.modules.pop(name, None)
