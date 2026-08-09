#!/usr/bin/env python3

"""本仓库迁移版 PCT stdin/stdout JSON 栅格规划 server。"""

from __future__ import annotations

import heapq
import itertools
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
BARRIER = 49.0

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.navigation.pct_local_map import pct_robot_body_obstacle_volume_from_ply


def main() -> int:
    sys.stdout.write("LOADING\n")
    sys.stdout.flush()
    state = _load_state()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        response = _handle_request(state, line)
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


class _State:
    """保存 PCT grid server 的运行时地图。"""

    def __init__(
        self,
        *,
        tomogram: dict[str, Any],
        walkable: np.ndarray,
        hard_obstacle_mask: np.ndarray | None = None,
        hard_obstacle_min_slices: int | None = None,
        cross_floor_hard_obstacle_mask: np.ndarray | None = None,
        cross_floor_hard_obstacle_min_slices: int | None = None,
        cross_floor_gateways: tuple[tuple[float, float, float], ...] = (),
        cross_floor_stair_exits: tuple[tuple[float, float, float], ...] = (),
        cross_floor_stair_midpoints: tuple[tuple[float, float, float], ...] = (),
        cross_floor_gateway_radius_m: float = 0.0,
        robot_root_to_floor_m: float = 0.45,
        stair_min_horizontal_per_slice_m: float = 0.40,
        stair_max_horizontal_per_slice_m: float = 0.90,
        stair_vertical_radius_m: float = 0.60,
        stair_progress_tolerance: float = 0.35,
        stair_progress_cost_weight: float = 20.0,
        obstacle_clearance_radius_m: float = 0.60,
        obstacle_clearance_cost_weight: float = 2.0,
        grid_max_expansions: int = 1_500_000,
        grid_compress_max_segment_m: float = 0.80,
        grid_timeout_sec: float = 10.0,
    ) -> None:
        self.tomogram = tomogram
        self.traversability = np.asarray(tomogram["data"][0], dtype=np.float32)
        self.walkable = walkable.astype(bool)
        self.resolution = float(tomogram["resolution"])
        self.center = np.asarray(tomogram["center"], dtype=np.float64)
        self.slice_h0 = float(tomogram["slice_h0"])
        self.slice_dh = float(tomogram["slice_dh"])
        self.n_slice, self.dimx, self.dimy = self.traversability.shape
        self.offset = np.array([self.dimx // 2, self.dimy // 2], dtype=np.int32)
        self.walkable = self.walkable[: self.n_slice, : self.dimx, : self.dimy]
        if hard_obstacle_mask is None:
            hard_obstacle_mask = np.zeros(
                (self.n_slice, self.dimx, self.dimy),
                dtype=bool,
            )
        self._validate_obstacle_mask(hard_obstacle_mask, label="PCT 硬障碍")
        if cross_floor_hard_obstacle_mask is None:
            cross_floor_hard_obstacle_mask = hard_obstacle_mask
        self._validate_obstacle_mask(
            cross_floor_hard_obstacle_mask,
            label="PCT 跨楼层硬障碍",
        )
        self.hard_obstacle_mask = hard_obstacle_mask.astype(bool, copy=False)
        self.hard_obstacle_min_slices = hard_obstacle_min_slices
        self.cross_floor_hard_obstacle_mask = (
            cross_floor_hard_obstacle_mask.astype(bool, copy=False)
        )
        self.cross_floor_hard_obstacle_min_slices = (
            cross_floor_hard_obstacle_min_slices
            if cross_floor_hard_obstacle_min_slices is not None
            else hard_obstacle_min_slices
        )
        self.cross_floor_gateways = tuple(cross_floor_gateways)
        self.cross_floor_stair_exits = tuple(cross_floor_stair_exits)
        self.cross_floor_stair_midpoints = tuple(cross_floor_stair_midpoints)
        self.cross_floor_gateway_radius_m = float(cross_floor_gateway_radius_m)
        if self.cross_floor_gateway_radius_m < 0.0:
            raise ValueError("PCT 跨楼层 gateway 半径不能为负数。")
        self.robot_root_to_floor_m = float(robot_root_to_floor_m)
        if self.robot_root_to_floor_m < 0.0:
            raise ValueError("PCT 机器人 root 到地面高度不能为负数。")
        self.stair_min_horizontal_per_slice_m = float(
            stair_min_horizontal_per_slice_m
        )
        self.stair_max_horizontal_per_slice_m = float(
            stair_max_horizontal_per_slice_m
        )
        if self.stair_min_horizontal_per_slice_m <= 0.0:
            raise ValueError("PCT 楼梯每 slice 最小水平行程必须为正数。")
        if (
            self.stair_max_horizontal_per_slice_m
            < self.stair_min_horizontal_per_slice_m
        ):
            raise ValueError("PCT 楼梯每 slice 最大水平行程不能小于最小值。")
        self.stair_vertical_radius_m = float(stair_vertical_radius_m)
        if self.stair_vertical_radius_m <= 0.0:
            raise ValueError("PCT 楼梯换层中心带半径必须为正数。")
        self.stair_progress_tolerance = float(stair_progress_tolerance)
        if not 0.0 <= self.stair_progress_tolerance <= 1.0:
            raise ValueError("PCT 楼梯进度容差必须位于 [0, 1]。")
        self.stair_progress_cost_weight = float(stair_progress_cost_weight)
        if self.stair_progress_cost_weight < 0.0:
            raise ValueError("PCT 楼梯进度软代价权重不能为负数。")
        self.obstacle_clearance_radius_m = float(obstacle_clearance_radius_m)
        self.obstacle_clearance_cost_weight = float(
            obstacle_clearance_cost_weight
        )
        if self.obstacle_clearance_radius_m < 0.0:
            raise ValueError("PCT 障碍净空半径不能为负数。")
        if self.obstacle_clearance_cost_weight < 0.0:
            raise ValueError("PCT 障碍净空代价权重不能为负数。")
        self.grid_max_expansions = int(grid_max_expansions)
        if self.grid_max_expansions < 1:
            raise ValueError("PCT grid 最大扩展数必须为正数。")
        self.grid_compress_max_segment_m = float(
            grid_compress_max_segment_m
        )
        if (
            not np.isfinite(self.grid_compress_max_segment_m)
            or self.grid_compress_max_segment_m <= 0.0
        ):
            raise ValueError("PCT grid 最大压缩段长必须为正数。")
        self.grid_timeout_sec = float(grid_timeout_sec)
        if not np.isfinite(self.grid_timeout_sec) or self.grid_timeout_sec <= 0.0:
            raise ValueError("PCT grid 规划超时必须为有限正数。")
        self.hard_obstacle_clearance_m = _approximate_obstacle_clearance(
            self.hard_obstacle_mask,
            resolution=self.resolution,
            max_distance_m=self.obstacle_clearance_radius_m,
        )
        if self.cross_floor_hard_obstacle_mask is self.hard_obstacle_mask:
            self.cross_floor_hard_obstacle_clearance_m = (
                self.hard_obstacle_clearance_m
            )
        else:
            self.cross_floor_hard_obstacle_clearance_m = (
                _approximate_obstacle_clearance(
                    self.cross_floor_hard_obstacle_mask,
                    resolution=self.resolution,
                    max_distance_m=self.obstacle_clearance_radius_m,
                )
            )
        self.cross_floor_gateway_mask = _build_gateway_mask(
            self,
            self.cross_floor_gateways,
            self.cross_floor_stair_exits,
            self.cross_floor_stair_midpoints,
            radius_m=self.cross_floor_gateway_radius_m,
        )
        if self.cross_floor_stair_exits:
            vertical_radius = min(
                self.cross_floor_gateway_radius_m,
                self.stair_vertical_radius_m,
            )
        else:
            vertical_radius = self.cross_floor_gateway_radius_m
        self.cross_floor_stair_surface_distance_m = (
            _build_stair_surface_distance(
                self,
                self.cross_floor_gateways,
                self.cross_floor_stair_exits,
                self.cross_floor_stair_midpoints,
            )
        )
        self.cross_floor_stair_vertical_mask = (
            None
            if self.cross_floor_stair_surface_distance_m is None
            else (
                self.cross_floor_stair_surface_distance_m
                <= vertical_radius + 1.0e-9
            )
        )
        self.cross_floor_gateway_progress = _build_gateway_progress(
            self,
            self.cross_floor_gateways,
            self.cross_floor_stair_exits,
            self.cross_floor_stair_midpoints,
            radius_m=self.cross_floor_gateway_radius_m,
        )
        self.cross_floor_gateway_expected_progress_by_slice = (
            _build_gateway_expected_progress_by_slice(
                self,
                self.cross_floor_gateways,
                self.cross_floor_stair_exits,
                self.cross_floor_stair_midpoints,
            )
        )

    def _validate_obstacle_mask(self, mask: np.ndarray, *, label: str) -> None:
        """允许兼容旧二维 mask，并优先使用逐 slice 三维 mask。"""

        valid_shapes = {
            (self.dimx, self.dimy),
            (self.n_slice, self.dimx, self.dimy),
        }
        if mask.shape not in valid_shapes:
            raise ValueError(f"{label} mask 与 tomogram 尺寸不一致。")

    def z_to_slice(self, z: float) -> int:
        si = int(round((float(z) - self.slice_h0) / self.slice_dh))
        return int(np.clip(si, 0, self.n_slice - 1))

    def pct_xy_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        xi = int(round((float(xy[0]) - self.center[0]) / self.resolution)) + int(self.offset[0])
        yj = int(round((float(xy[1]) - self.center[1]) / self.resolution)) + int(self.offset[1])
        return int(np.clip(xi, 0, self.dimx - 1)), int(np.clip(yj, 0, self.dimy - 1))

    def grid_to_pct_xyz(self, node: tuple[int, int, int]) -> list[float]:
        si, xi, yj = node
        x = (xi - self.offset[0]) * self.resolution + self.center[0]
        y = (yj - self.offset[1]) * self.resolution + self.center[1]
        z = self.slice_h0 + si * self.slice_dh
        return [float(x), float(y), float(z)]

    def is_walkable(
        self,
        node: tuple[int, int, int],
        *,
        hard_obstacle_mask: np.ndarray | None = None,
    ) -> bool:
        si, xi, yj = node
        if not (0 <= si < self.n_slice and 0 <= xi < self.dimx and 0 <= yj < self.dimy):
            return False
        trav_ok = 0.0 < float(self.traversability[si, xi, yj]) < BARRIER
        ply_ok = bool(self.walkable[si, xi, yj])
        obstacle_mask = (
            self.hard_obstacle_mask
            if hard_obstacle_mask is None
            else hard_obstacle_mask
        )
        blocked = (
            bool(obstacle_mask[si, xi, yj])
            if obstacle_mask.ndim == 3
            else bool(obstacle_mask[xi, yj])
        )
        return bool((trav_ok or ply_ok) and not blocked)

    def obstacle_clearance_m(
        self,
        node: tuple[int, int, int],
        *,
        hard_obstacle_mask: np.ndarray | None,
    ) -> float:
        """返回节点到当前障碍体的近似 XY 净空。"""

        si, xi, yj = node
        if hard_obstacle_mask is self.cross_floor_hard_obstacle_mask:
            clearance = self.cross_floor_hard_obstacle_clearance_m
        elif hard_obstacle_mask is self.hard_obstacle_mask:
            clearance = self.hard_obstacle_clearance_m
        else:
            return self.obstacle_clearance_radius_m
        if clearance.ndim == 3:
            return float(clearance[si, xi, yj])
        return float(clearance[xi, yj])


def _load_state() -> _State:
    tomogram_path = Path(os.environ.get("PCT_TOMOGRAM_PATH", os.fspath(DEFAULT_TOMOGRAM))).expanduser()
    walkable_path = Path(os.environ.get("PCT_WALKABLE_PATH", os.fspath(DEFAULT_WALKABLE))).expanduser()
    raw_collision_ply_path = os.environ.get("PCT_COLLISION_PLY_PATH")
    collision_ply_path = (
        Path(raw_collision_ply_path).expanduser()
        if raw_collision_ply_path
        else None
    )
    hard_obstacle_min_slices = int(
        os.environ.get("PCT_GLOBAL_VERTICAL_OBSTACLE_MIN_SLICES", "7")
    )
    if hard_obstacle_min_slices < 1:
        raise ValueError("PCT_GLOBAL_VERTICAL_OBSTACLE_MIN_SLICES 必须为正数。")
    cross_floor_hard_obstacle_min_slices = int(
        os.environ.get(
            "PCT_CROSS_FLOOR_VERTICAL_OBSTACLE_MIN_SLICES",
            "9",
        )
    )
    if cross_floor_hard_obstacle_min_slices < 1:
        raise ValueError("PCT_CROSS_FLOOR_VERTICAL_OBSTACLE_MIN_SLICES 必须为正数。")
    cross_floor_gateways = _parse_cross_floor_gateways(
        os.environ.get("PCT_CROSS_FLOOR_GATEWAYS_PCT", "")
    )
    cross_floor_stair_exits = _parse_cross_floor_gateways(
        os.environ.get("PCT_CROSS_FLOOR_STAIR_EXITS_PCT", "")
    )
    cross_floor_stair_midpoints = _parse_cross_floor_gateways(
        os.environ.get("PCT_CROSS_FLOOR_STAIR_MIDPOINTS_PCT", "")
    )
    cross_floor_gateway_radius_m = float(
        os.environ.get("PCT_CROSS_FLOOR_GATEWAY_RADIUS_M", "0.0")
    )
    robot_root_to_floor_m = float(
        os.environ.get("PCT_ROBOT_ROOT_TO_FLOOR_M", "0.45")
    )
    body_obstacle_min_height_m = float(
        os.environ.get("PCT_BODY_OBSTACLE_MIN_HEIGHT_M", "0.30")
    )
    body_obstacle_max_height_m = float(
        os.environ.get("PCT_BODY_OBSTACLE_MAX_HEIGHT_M", "1.0")
    )
    stair_min_horizontal_per_slice_m = float(
        os.environ.get("PCT_STAIR_MIN_HORIZONTAL_PER_SLICE_M", "0.40")
    )
    stair_max_horizontal_per_slice_m = float(
        os.environ.get("PCT_STAIR_MAX_HORIZONTAL_PER_SLICE_M", "0.90")
    )
    stair_vertical_radius_m = float(
        os.environ.get("PCT_STAIR_VERTICAL_RADIUS_M", "0.60")
    )
    stair_progress_tolerance = float(
        os.environ.get("PCT_STAIR_PROGRESS_TOLERANCE", "0.35")
    )
    stair_progress_cost_weight = float(
        os.environ.get("PCT_STAIR_PROGRESS_COST_WEIGHT", "20.0")
    )
    obstacle_clearance_radius_m = float(
        os.environ.get("PCT_OBSTACLE_CLEARANCE_RADIUS_M", "0.60")
    )
    obstacle_clearance_cost_weight = float(
        os.environ.get("PCT_OBSTACLE_CLEARANCE_COST_WEIGHT", "2.0")
    )
    grid_max_expansions = int(
        os.environ.get("PCT_GRID_MAX_EXPANSIONS", "1500000")
    )
    grid_compress_max_segment_m = float(
        os.environ.get("PCT_GRID_COMPRESS_MAX_SEGMENT_M", "0.8")
    )
    grid_timeout_sec = float(os.environ.get("PCT_GRID_TIMEOUT_SEC", "10.0"))
    inserted_aliases = _install_numpy_pickle_aliases()
    try:
        with tomogram_path.open("rb") as stream:
            tomogram = pickle.load(stream)
    finally:
        _remove_numpy_pickle_aliases(inserted_aliases)
    walkable = np.load(walkable_path)
    hard_obstacle_mask = None
    cross_floor_hard_obstacle_mask = None
    if collision_ply_path is not None and collision_ply_path.is_file():
        hard_obstacle_mask = pct_robot_body_obstacle_volume_from_ply(
            collision_ply_path=collision_ply_path,
            tomogram=tomogram,
            min_height_m=body_obstacle_min_height_m,
            max_height_m=body_obstacle_max_height_m,
        )
        cross_floor_hard_obstacle_mask = hard_obstacle_mask
    elif collision_ply_path is not None:
        raise FileNotFoundError(f"PCT collision PLY 不存在: {collision_ply_path}")
    return _State(
        tomogram=tomogram,
        walkable=walkable,
        hard_obstacle_mask=hard_obstacle_mask,
        hard_obstacle_min_slices=(
            hard_obstacle_min_slices
            if hard_obstacle_mask is not None
            else None
        ),
        cross_floor_hard_obstacle_mask=cross_floor_hard_obstacle_mask,
        cross_floor_hard_obstacle_min_slices=(
            cross_floor_hard_obstacle_min_slices
            if cross_floor_hard_obstacle_mask is not None
            else None
        ),
        cross_floor_gateways=cross_floor_gateways,
        cross_floor_stair_exits=cross_floor_stair_exits,
        cross_floor_stair_midpoints=cross_floor_stair_midpoints,
        cross_floor_gateway_radius_m=cross_floor_gateway_radius_m,
        robot_root_to_floor_m=robot_root_to_floor_m,
        stair_min_horizontal_per_slice_m=stair_min_horizontal_per_slice_m,
        stair_max_horizontal_per_slice_m=stair_max_horizontal_per_slice_m,
        stair_vertical_radius_m=stair_vertical_radius_m,
        stair_progress_tolerance=stair_progress_tolerance,
        stair_progress_cost_weight=stair_progress_cost_weight,
        obstacle_clearance_radius_m=obstacle_clearance_radius_m,
        obstacle_clearance_cost_weight=obstacle_clearance_cost_weight,
        grid_max_expansions=grid_max_expansions,
        grid_compress_max_segment_m=grid_compress_max_segment_m,
        grid_timeout_sec=grid_timeout_sec,
    )


def load_state_from_environment() -> _State:
    """按环境参数加载一次 PCT 地图，供 ROS 2 进程内 backend 复用。"""

    return _load_state()


def _install_numpy_pickle_aliases() -> tuple[str, ...]:
    """兼容 numpy 2 保存、numpy 1 读取的 pickle 模块路径。"""

    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric

    aliases: list[tuple[str, object]] = [
        ("numpy._core", numpy_core),
        ("numpy._core.numeric", numpy_core_numeric),
    ]
    inserted: list[str] = []
    for name, module in aliases:
        if name in sys.modules:
            continue
        sys.modules[name] = module
        inserted.append(name)
    return tuple(inserted)


def _remove_numpy_pickle_aliases(inserted_aliases: tuple[str, ...]) -> None:
    """撤销本次 pickle 读取临时加入的 numpy 模块别名。"""

    for name in inserted_aliases:
        sys.modules.pop(name, None)


def _parse_cross_floor_gateways(
    raw_value: str,
) -> tuple[tuple[float, float, float], ...]:
    """解析 PCT 坐标系下的跨层 gateway 列表。"""

    text = raw_value.strip()
    if not text:
        return ()
    if text.startswith("["):
        payload = json.loads(text)
    else:
        payload = [
            [float(part) for part in item.split(",")]
            for item in text.split(";")
            if item.strip()
        ]
    gateways: list[tuple[float, float, float]] = []
    for raw_gateway in payload:
        if not isinstance(raw_gateway, (list, tuple)) or len(raw_gateway) < 2:
            raise ValueError(f"PCT 跨层 gateway 格式错误: {raw_gateway!r}")
        z_value = float(raw_gateway[2]) if len(raw_gateway) >= 3 else 0.0
        gateways.append((float(raw_gateway[0]), float(raw_gateway[1]), z_value))
    return tuple(gateways)


def _build_gateway_mask(
    state: _State,
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...] = (),
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
    *,
    radius_m: float,
) -> np.ndarray | None:
    """把楼梯入口、拐角和出口转换为折线 XY corridor。"""

    if not gateways:
        return None
    mask = np.zeros((state.dimx, state.dimy), dtype=bool)
    grid_x, grid_y = np.indices((state.dimx, state.dimy))
    for anchors in _stair_anchor_groups(gateways, stair_exits, stair_midpoints):
        for start, end in _stair_anchor_grid_segments(state, anchors):
            distance, _alpha = _distance_to_grid_segment(
                grid_x,
                grid_y,
                start,
                end,
                resolution=state.resolution,
            )
            mask |= distance <= float(radius_m) + 1.0e-9
    return mask


def _build_gateway_progress(
    state: _State,
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...],
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
    *,
    radius_m: float,
) -> np.ndarray | None:
    """记录楼梯折线 corridor 从入口到出口的归一化进度。"""

    if not gateways or not stair_exits:
        return None
    progress = np.full((state.dimx, state.dimy), np.nan, dtype=np.float32)
    best_distance = np.full((state.dimx, state.dimy), np.inf, dtype=np.float64)
    grid_x, grid_y = np.indices((state.dimx, state.dimy))
    for anchors in _stair_anchor_groups(gateways, stair_exits, stair_midpoints):
        grid_anchors = _stair_anchor_grid_points(state, anchors)
        segment_lengths = [
            float(np.linalg.norm(end - start))
            for start, end in zip(grid_anchors, grid_anchors[1:])
        ]
        total_length = float(sum(segment_lengths))
        if total_length <= 1.0e-12:
            continue
        accumulated = 0.0
        for segment_index, (start, end) in enumerate(
            zip(grid_anchors, grid_anchors[1:])
        ):
            segment_length = segment_lengths[segment_index]
            if segment_length <= 1.0e-12:
                continue
            distance, alpha = _distance_to_grid_segment(
                grid_x,
                grid_y,
                start,
                end,
                resolution=state.resolution,
            )
            segment_progress = (accumulated + alpha * segment_length) / total_length
            inside = (
                (distance <= float(radius_m) + 1.0e-9)
                & (distance < best_distance)
            )
            progress[inside] = segment_progress[inside].astype(np.float32)
            best_distance[inside] = distance[inside]
            accumulated += segment_length
    return progress


def _build_stair_surface_mask(
    state: _State,
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...],
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
    *,
    radius_m: float,
) -> np.ndarray | None:
    """按实测楼梯表面高度构造逐 slice 三维通行走廊。"""

    distance = _build_stair_surface_distance(
        state,
        gateways,
        stair_exits,
        stair_midpoints,
    )
    if distance is None:
        return None
    return distance <= float(radius_m) + 1.0e-9


def _build_stair_surface_distance(
    state: _State,
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...],
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray | None:
    """计算每个 slice 栅格到三维楼梯中心面的水平距离。"""

    if not gateways or not stair_exits:
        return None
    output = np.full(
        (state.n_slice, state.dimx, state.dimy),
        np.inf,
        dtype=np.float32,
    )
    grid_x, grid_y = np.indices((state.dimx, state.dimy))
    half_slice_height = 0.5 * float(state.slice_dh) + 1.0e-9
    for raw_anchors in _stair_anchor_groups(
        gateways,
        stair_exits,
        stair_midpoints,
    ):
        anchors = _normalized_stair_surface_anchors(raw_anchors)
        if len(anchors) < 2:
            continue
        grid_anchors = _stair_anchor_grid_points(state, anchors)
        for slice_index in range(state.n_slice):
            slice_z = state.slice_h0 + slice_index * state.slice_dh
            if slice_z < float(anchors[0][2]):
                center = grid_anchors[0]
                distance = (
                    np.hypot(grid_x - center[0], grid_y - center[1])
                    * float(state.resolution)
                )
                output[slice_index] = np.minimum(
                    output[slice_index],
                    distance,
                )
            elif slice_z > float(anchors[-1][2]):
                center = grid_anchors[-1]
                distance = (
                    np.hypot(grid_x - center[0], grid_y - center[1])
                    * float(state.resolution)
                )
                output[slice_index] = np.minimum(
                    output[slice_index],
                    distance,
                )
            for anchor_index, (start, end) in enumerate(
                zip(anchors, anchors[1:])
            ):
                start_z = float(start[2])
                end_z = float(end[2])
                z_low = min(start_z, end_z)
                z_high = max(start_z, end_z)
                is_first = anchor_index == 0
                is_last = anchor_index == len(anchors) - 2
                if (
                    slice_z < z_low - half_slice_height
                    and not (is_first and slice_z < z_low)
                ):
                    continue
                if (
                    slice_z > z_high + half_slice_height
                    and not (is_last and slice_z > z_high)
                ):
                    continue
                start_grid = grid_anchors[anchor_index]
                end_grid = grid_anchors[anchor_index + 1]
                delta_z = end_z - start_z
                if abs(delta_z) <= 1.0e-6:
                    if abs(slice_z - 0.5 * (start_z + end_z)) > half_slice_height:
                        continue
                    distance, _alpha = _distance_to_grid_segment(
                        grid_x,
                        grid_y,
                        start_grid,
                        end_grid,
                        resolution=state.resolution,
                    )
                else:
                    alpha = float(np.clip((slice_z - start_z) / delta_z, 0.0, 1.0))
                    center = start_grid + alpha * (end_grid - start_grid)
                    distance = (
                        np.hypot(grid_x - center[0], grid_y - center[1])
                        * float(state.resolution)
                    )
                output[slice_index] = np.minimum(
                    output[slice_index],
                    distance,
                )
    return output


def _stair_anchor_groups(
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...],
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
) -> list[tuple[tuple[float, float, float], ...]]:
    """按入口、用户实测台阶点、出口构造每条楼梯的折线控制点。"""

    groups: list[tuple[tuple[float, float, float], ...]] = []
    for gateway_index, gateway in enumerate(gateways):
        anchors: list[tuple[float, float, float]] = [gateway]
        if stair_exits:
            # 台阶点允许无序输入；同高度平台按连接代价自动确定顺序。
            stair_exit = stair_exits[min(gateway_index, len(stair_exits) - 1)]
            anchors.extend(
                _ordered_stair_midpoints(
                    gateway,
                    stair_exit,
                    stair_midpoints,
                )
            )
            anchors.append(stair_exit)
        groups.append(tuple(anchors))
    return groups


def _normalized_stair_surface_anchors(
    anchors: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """修正入口/出口高度语义，并保持楼梯表面高度单调。"""

    if len(anchors) < 2:
        return anchors
    points = [list(map(float, point)) for point in anchors]
    ascending = points[-1][2] >= points[0][2]
    if ascending:
        points[0][2] = min(points[0][2], points[1][2])
        points[-1][2] = max(points[-1][2], points[-2][2])
        for index in range(1, len(points)):
            points[index][2] = max(points[index][2], points[index - 1][2])
    else:
        points[0][2] = max(points[0][2], points[1][2])
        points[-1][2] = min(points[-1][2], points[-2][2])
        for index in range(1, len(points)):
            points[index][2] = min(points[index][2], points[index - 1][2])
    return tuple(tuple(point) for point in points)


def _ordered_stair_midpoints(
    gateway: tuple[float, float, float],
    stair_exit: tuple[float, float, float],
    stair_midpoints: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """将无序楼梯采样点按上楼或下楼方向排序。"""

    if not stair_midpoints:
        return ()
    ascending = float(stair_exit[2]) >= float(gateway[2])
    sorted_points = sorted(
        stair_midpoints,
        key=lambda point: float(point[2]),
        reverse=not ascending,
    )
    clusters: list[list[tuple[float, float, float]]] = []
    plateau_height_tolerance = 0.25
    for point in sorted_points:
        if (
            not clusters
            or abs(float(point[2]) - float(clusters[-1][-1][2]))
            > plateau_height_tolerance
        ):
            clusters.append([point])
        else:
            clusters[-1].append(point)
    ordered: list[tuple[float, float, float]] = []
    previous = gateway
    for cluster_index, cluster in enumerate(clusters):
        next_anchor = (
            clusters[cluster_index + 1][0]
            if cluster_index + 1 < len(clusters)
            else stair_exit
        )
        plateau = _order_stair_plateau_group(
            cluster,
            previous=previous,
            next_anchor=next_anchor,
        )
        ordered.extend(plateau)
        previous = plateau[-1]
    return tuple(ordered)


def _order_stair_plateau_group(
    points: list[tuple[float, float, float]],
    *,
    previous: tuple[float, float, float],
    next_anchor: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """按连接前后楼梯段的最短 XY 路径排列同高度平台采样。"""

    if len(points) <= 1:
        return tuple(points)
    if len(points) > 8:
        remaining = list(points)
        ordered: list[tuple[float, float, float]] = []
        current = previous
        while remaining:
            chosen = min(remaining, key=lambda point: _xy_distance(current, point))
            ordered.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return tuple(ordered)
    return min(
        itertools.permutations(points),
        key=lambda candidate: (
            _xy_distance(previous, candidate[0])
            + sum(
                _xy_distance(start, end)
                for start, end in zip(candidate, candidate[1:])
            )
            + _xy_distance(candidate[-1], next_anchor)
        ),
    )


def _xy_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """返回两个楼梯采样点的 XY 距离。"""

    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


def _stair_anchor_grid_points(
    state: _State,
    anchors: tuple[tuple[float, float, float], ...],
) -> list[np.ndarray]:
    """把 PCT 坐标系控制点转换为 grid xy 点。"""

    return [
        np.asarray(state.pct_xy_to_grid(np.asarray(anchor[:2])), dtype=np.float64)
        for anchor in anchors
    ]


def _stair_anchor_grid_segments(
    state: _State,
    anchors: tuple[tuple[float, float, float], ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回折线相邻控制点的 grid segment。"""

    grid_points = _stair_anchor_grid_points(state, anchors)
    if len(grid_points) == 1:
        return [(grid_points[0], grid_points[0])]
    return list(zip(grid_points, grid_points[1:]))


def _distance_to_grid_segment(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算所有 grid cell 到线段的距离和线段内进度。"""

    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 1.0e-12:
        alpha = np.zeros(grid_x.shape, dtype=np.float64)
    else:
        alpha = np.clip(
            (
                (grid_x - start[0]) * segment[0]
                + (grid_y - start[1]) * segment[1]
            )
            / denominator,
            0.0,
            1.0,
        )
    nearest_x = start[0] + alpha * segment[0]
    nearest_y = start[1] + alpha * segment[1]
    distance = np.hypot(grid_x - nearest_x, grid_y - nearest_y) * float(resolution)
    return distance, alpha


def _build_gateway_expected_progress_by_slice(
    state: _State,
    gateways: tuple[tuple[float, float, float], ...],
    stair_exits: tuple[tuple[float, float, float], ...],
    stair_midpoints: tuple[tuple[float, float, float], ...] = (),
) -> np.ndarray | None:
    """根据楼梯控制点的 z 值估计每个 slice 应处于的折线进度。"""

    if not gateways or not stair_exits:
        return None
    anchors = _normalized_stair_surface_anchors(
        _stair_anchor_groups(gateways, stair_exits, stair_midpoints)[0]
    )
    if len(anchors) < 2:
        return None
    grid_points = _stair_anchor_grid_points(state, anchors)
    segment_lengths = [
        float(np.linalg.norm(end - start))
        for start, end in zip(grid_points, grid_points[1:])
    ]
    total_length = float(sum(segment_lengths))
    if total_length <= 1.0e-12:
        return None
    anchor_progress = [0.0]
    accumulated = 0.0
    for segment_length in segment_lengths:
        accumulated += segment_length
        anchor_progress.append(accumulated / total_length)
    anchor_z = np.asarray([anchor[2] for anchor in anchors], dtype=np.float64)
    ordered_z, ordered_progress = _ordered_anchor_z_progress(
        anchor_z,
        np.asarray(anchor_progress, dtype=np.float64),
    )
    expected = np.full(state.n_slice, np.nan, dtype=np.float32)
    if ordered_z is None or float(ordered_z[-1] - ordered_z[0]) <= 1.0e-9:
        return expected
    for slice_index in range(state.n_slice):
        z_value = state.slice_h0 + slice_index * state.slice_dh
        expected[slice_index] = float(
            np.interp(
                z_value,
                ordered_z,
                ordered_progress,
                left=ordered_progress[0],
                right=ordered_progress[-1],
            )
        )
    return expected


def _ordered_anchor_z_progress(
    anchor_z: np.ndarray,
    anchor_progress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """按楼梯行走顺序建立 z 到折线进度的单调映射。"""

    if len(anchor_z) != len(anchor_progress) or len(anchor_z) < 2:
        return None, None
    z_values = np.asarray(anchor_z, dtype=np.float64).copy()
    progress_values = np.asarray(anchor_progress, dtype=np.float64).copy()
    if float(z_values[-1]) >= float(z_values[0]):
        z_values = np.maximum.accumulate(z_values)
        for index in range(1, len(z_values)):
            if z_values[index] <= z_values[index - 1]:
                z_values[index] = z_values[index - 1] + 1.0e-6
        return z_values, progress_values
    z_values = np.minimum.accumulate(z_values)
    for index in range(1, len(z_values)):
        if z_values[index] >= z_values[index - 1]:
            z_values[index] = z_values[index - 1] - 1.0e-6
    return z_values[::-1], progress_values[::-1]


def _handle_request(state: _State, line: str) -> dict[str, Any]:
    """兼容历史 stdin/stdout 入口，并把实际规划转交给类型化函数。"""

    try:
        request = json.loads(line)
        start = request["start"]
        end = request["end"]
    except Exception as exc:
        return _error_response(exc)
    return plan_request(state, start=start, end=end)


def plan_request(
    state: _State,
    *,
    start: Any,
    end: Any,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """直接规划一个 PCT xyz 请求，不经过 JSON、文件或子进程通信。"""

    try:
        deadline_monotonic = time.monotonic() + state.grid_timeout_sec
        _check_planning_interrupt(
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("PCT start/end 必须各包含 3 个坐标")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError("PCT start/end 不能包含 NaN 或 Inf")
        start_floor_z = float(start[2]) - state.robot_root_to_floor_m
        end_floor_z = float(end[2]) - state.robot_root_to_floor_m
        start_slice = state.z_to_slice(start_floor_z)
        end_slice = state.z_to_slice(end_floor_z)
        cross_floor = abs(end_slice - start_slice) >= 2
        hard_obstacle_mask = (
            state.cross_floor_hard_obstacle_mask
            if cross_floor
            else state.hard_obstacle_mask
        )
        hard_obstacle_min_slices = (
            state.cross_floor_hard_obstacle_min_slices
            if cross_floor
            else state.hard_obstacle_min_slices
        )
        start_node, start_dist = _snap_to_walkable(
            state,
            start[:2],
            start_slice,
            hard_obstacle_mask=hard_obstacle_mask,
            preferred_xy=end[:2],
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        end_node, end_dist = _snap_to_walkable(
            state,
            end[:2],
            end_slice,
            hard_obstacle_mask=hard_obstacle_mask,
            preferred_xy=start[:2],
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        if start_node[0] != start_slice or end_node[0] != end_slice:
            raise RuntimeError("PCT grid snap 不得跨越请求 slice。")
        snapped_start_xyz = state.grid_to_pct_xyz(start_node)
        snapped_end_xyz = state.grid_to_pct_xyz(end_node)
        snap_start_distance_m = float(
            np.linalg.norm(np.asarray(snapped_start_xyz[:2]) - start[:2])
        )
        snap_end_distance_m = float(
            np.linalg.norm(np.asarray(snapped_end_xyz[:2]) - end[:2])
        )
        _check_planning_interrupt(
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        gateway_mode: str | None = None
        if cross_floor and state.cross_floor_gateways:
            path, path_mode, gateway_mode = _plan_via_cross_floor_gateway(
                state,
                start_node,
                end_node,
                hard_obstacle_mask=hard_obstacle_mask,
                cancel_check=cancel_check,
                deadline_monotonic=deadline_monotonic,
            )
        else:
            path = _same_floor_direct_path(
                state,
                start_node,
                end_node,
                hard_obstacle_mask=hard_obstacle_mask,
            )
            _check_planning_interrupt(
                cancel_check=cancel_check,
                deadline_monotonic=deadline_monotonic,
            )
            path_mode = "same_floor_direct"
            if path is None:
                path = _astar(
                    state,
                    start_node,
                    end_node,
                    hard_obstacle_mask=hard_obstacle_mask,
                    vertical_gateway_mask=(
                        state.cross_floor_gateway_mask if cross_floor else None
                    ),
                    vertical_direction=(
                        _slice_direction(start_node, end_node) if cross_floor else 0
                    ),
                    cancel_check=cancel_check,
                    deadline_monotonic=deadline_monotonic,
                )
                path_mode = "astar_3d"
        _check_planning_interrupt(
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        if path is None:
            return {
                "status": "no_path",
                "slice_start": start_slice,
                "slice_end": end_slice,
                "snap_start_dist": start_dist,
                "snap_end_dist": end_dist,
                "snap_start_distance_m": snap_start_distance_m,
                "snap_end_distance_m": snap_end_distance_m,
                "snapped_start_xyz": snapped_start_xyz,
                "snapped_end_xyz": snapped_end_xyz,
                "snapped_start_slice": int(start_node[0]),
                "snapped_end_slice": int(end_node[0]),
                "snap_start_slice_delta": int(start_node[0] - start_slice),
                "snap_end_slice_delta": int(end_node[0] - end_slice),
                "planner": "pct_grid",
                "cross_floor": cross_floor,
                "hard_obstacle_cells": int(hard_obstacle_mask.sum()),
                "hard_obstacle_mode": _obstacle_mask_mode(hard_obstacle_mask),
                "hard_obstacle_min_slices": hard_obstacle_min_slices,
                "default_hard_obstacle_min_slices": state.hard_obstacle_min_slices,
                "cross_floor_hard_obstacle_min_slices": (
                    state.cross_floor_hard_obstacle_min_slices
                ),
                "cross_floor_gateway_count": len(state.cross_floor_gateways),
                "cross_floor_stair_exit_count": len(
                    state.cross_floor_stair_exits
                ),
                "cross_floor_stair_midpoint_count": len(
                    state.cross_floor_stair_midpoints
                ),
                "cross_floor_gateway_radius_m": state.cross_floor_gateway_radius_m,
                "cross_floor_gateway_cells": int(
                    state.cross_floor_gateway_mask.sum()
                    if state.cross_floor_gateway_mask is not None
                    else 0
                ),
                "cross_floor_stair_vertical_cells": int(
                    state.cross_floor_stair_vertical_mask.sum()
                    if state.cross_floor_stair_vertical_mask is not None
                    else 0
                ),
                "cross_floor_gateway_mode": gateway_mode,
                "robot_root_to_floor_m": state.robot_root_to_floor_m,
                "planning_start_z": start_floor_z,
                "planning_end_z": end_floor_z,
                "stair_vertical_radius_m": state.stair_vertical_radius_m,
                "stair_constraint_mode": _stair_constraint_mode(state),
                "stair_progress_tolerance": state.stair_progress_tolerance,
                "stair_progress_cost_weight": state.stair_progress_cost_weight,
            }
        return {
            "status": "ok",
            "traj": _compress_path(
                [state.grid_to_pct_xyz(node) for node in path],
                max_segment_length_m=state.grid_compress_max_segment_m,
            ),
            "slice_start": start_slice,
            "slice_end": end_slice,
            "snap_start_dist": start_dist,
            "snap_end_dist": end_dist,
            "snap_start_distance_m": snap_start_distance_m,
            "snap_end_distance_m": snap_end_distance_m,
            "snapped_start_xyz": snapped_start_xyz,
            "snapped_end_xyz": snapped_end_xyz,
            "snapped_start_slice": int(start_node[0]),
            "snapped_end_slice": int(end_node[0]),
            "snap_start_slice_delta": int(start_node[0] - start_slice),
            "snap_end_slice_delta": int(end_node[0] - end_slice),
            "planner": "pct_grid",
            "path_mode": path_mode,
            "cross_floor": cross_floor,
            "hard_obstacle_cells": int(hard_obstacle_mask.sum()),
            "hard_obstacle_mode": _obstacle_mask_mode(hard_obstacle_mask),
            "hard_obstacle_min_slices": hard_obstacle_min_slices,
            "default_hard_obstacle_min_slices": state.hard_obstacle_min_slices,
            "cross_floor_hard_obstacle_min_slices": (
                state.cross_floor_hard_obstacle_min_slices
            ),
            "cross_floor_gateway_count": len(state.cross_floor_gateways),
            "cross_floor_stair_exit_count": len(state.cross_floor_stair_exits),
            "cross_floor_stair_midpoint_count": len(
                state.cross_floor_stair_midpoints
            ),
            "cross_floor_gateway_radius_m": state.cross_floor_gateway_radius_m,
            "cross_floor_gateway_cells": int(
                state.cross_floor_gateway_mask.sum()
                if state.cross_floor_gateway_mask is not None
                else 0
            ),
            "cross_floor_stair_vertical_cells": int(
                state.cross_floor_stair_vertical_mask.sum()
                if state.cross_floor_stair_vertical_mask is not None
                else 0
            ),
            "cross_floor_gateway_mode": gateway_mode,
            "robot_root_to_floor_m": state.robot_root_to_floor_m,
            "planning_start_z": start_floor_z,
            "planning_end_z": end_floor_z,
            "stair_vertical_radius_m": state.stair_vertical_radius_m,
            "stair_constraint_mode": _stair_constraint_mode(state),
            "stair_progress_tolerance": state.stair_progress_tolerance,
            "stair_progress_cost_weight": state.stair_progress_cost_weight,
        }
    except Exception as exc:
        return _error_response(exc)


def _error_response(exc: Exception) -> dict[str, Any]:
    """把 grid core 异常转换为旧 server 与 ROS adapter 共用的诊断对象。"""

    import traceback

    return {
        "status": "error",
        "msg": str(exc),
        "traceback": traceback.format_exc(),
    }


def _plan_via_cross_floor_gateway(
    state: _State,
    start_node: tuple[int, int, int],
    end_node: tuple[int, int, int],
    *,
    hard_obstacle_mask: np.ndarray | None = None,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[list[tuple[int, int, int]] | None, str, str]:
    """跨楼层路径先强制经过楼梯/坡道入口，再继续做三维搜索。"""

    candidates: list[tuple[float, list[tuple[int, int, int]], str, str]] = []
    for gateway_index, gateway_xyz in enumerate(state.cross_floor_gateways):
        gateway_node, _dist = _snap_to_walkable(
            state,
            np.asarray(gateway_xyz[:2], dtype=np.float64),
            start_node[0],
            hard_obstacle_mask=hard_obstacle_mask,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        approach = _astar(
            state,
            start_node,
            gateway_node,
            hard_obstacle_mask=hard_obstacle_mask,
            allow_vertical_transitions=False,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        if approach is None:
            continue
        stair_targets = _stair_target_nodes_for_gateway(
            state,
            gateway_index=gateway_index,
            end_node=end_node,
            hard_obstacle_mask=hard_obstacle_mask,
        )
        tail = approach
        current_node = gateway_node
        tail_ok = True
        vertical_mask = (
            state.cross_floor_stair_vertical_mask
            if state.cross_floor_stair_vertical_mask is not None
            else state.cross_floor_gateway_mask
        )
        stair_slice_range = (gateway_node[0], end_node[0])
        for target_node in stair_targets:
            segment = _astar(
                state,
                current_node,
                target_node,
                hard_obstacle_mask=hard_obstacle_mask,
                vertical_gateway_mask=vertical_mask,
                vertical_direction=_slice_direction(current_node, target_node),
                stair_slice_range=stair_slice_range,
                cancel_check=cancel_check,
                deadline_monotonic=deadline_monotonic,
            )
            if segment is None:
                tail_ok = False
                break
            tail = _join_paths(tail, segment)
            current_node = target_node
        if tail_ok:
            full_path = tail
            candidates.append(
                (
                    _path_cost(state, full_path),
                    full_path,
                    "astar_3d_via_gateway_strict_monotonic",
                    "strict_monotonic",
                )
            )
    if not candidates:
        return None, "astar_3d_via_gateway", "no_gateway_path"
    _cost, path, path_mode, gateway_mode = min(candidates, key=lambda item: item[0])
    return path, path_mode, gateway_mode


def _stair_constraint_mode(state: _State) -> str:
    """返回当前跨楼层搜索使用的楼梯空间约束模式。"""

    mask = state.cross_floor_stair_vertical_mask
    if mask is not None and mask.ndim == 3:
        return "surface_3d"
    if mask is not None:
        return "corridor_2d"
    if state.cross_floor_gateway_mask is not None:
        return "corridor_2d"
    return "unconstrained"


def _stair_target_nodes_for_gateway(
    state: _State,
    *,
    gateway_index: int,
    end_node: tuple[int, int, int],
    hard_obstacle_mask: np.ndarray | None,
) -> list[tuple[int, int, int]]:
    """返回跨层搜索目标，楼梯控制点只约束地图通道而不是任务 waypoint。"""

    del state, gateway_index, hard_obstacle_mask
    return [end_node]


def _join_paths(
    first: list[tuple[int, int, int]],
    second: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """拼接两段 A* 路径，避免重复中间 gateway 节点。"""

    if not first:
        return second
    if not second:
        return first
    if first[-1] == second[0]:
        return [*first, *second[1:]]
    return [*first, *second]


def _path_cost(state: _State, path: list[tuple[int, int, int]]) -> float:
    """按当前 step cost 计算完整路径代价。"""

    if len(path) <= 1:
        return 0.0
    total = 0.0
    for current, nxt in zip(path, path[1:]):
        total += _step_cost(
            state,
            nxt[0] - current[0],
            nxt[1] - current[1],
            nxt[2] - current[2],
        )
    return float(total)


def _snap_to_walkable(
    state: _State,
    pct_xy: np.ndarray,
    si: int,
    *,
    hard_obstacle_mask: np.ndarray | None = None,
    preferred_xy: np.ndarray | None = None,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[tuple[int, int, int], int]:
    """只在请求 slice 内选择最近可走格，并消除数组遍历方向偏置。

    相邻楼层可能在同一 XY 上同时存在可走格。snap 若沿 slice 方向搜索，会把
    被阻塞的一层请求静默吸附到另一层，因此这里明确把 BFS 限定在 ``si``。
    同一 BFS 半径内再按请求点距离和路线目标距离稳定排序。
    """

    xi, yj = state.pct_xy_to_grid(pct_xy)
    start = (si, xi, yj)
    _check_planning_interrupt(
        cancel_check=cancel_check,
        deadline_monotonic=deadline_monotonic,
    )
    if state.is_walkable(start, hard_obstacle_mask=hard_obstacle_mask):
        return start, 0
    visited = {(xi, yj)}
    queue = deque([(xi, yj, 0)])
    while queue:
        _check_planning_interrupt(
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        layer_distance = int(queue[0][2])
        candidates: list[tuple[int, int, int]] = []
        while queue and int(queue[0][2]) == layer_distance:
            cxi, cyj, dist = queue.popleft()
            node = (si, cxi, cyj)
            if state.is_walkable(node, hard_obstacle_mask=hard_obstacle_mask):
                candidates.append(node)
                continue
            for dx, dy in _xy_neighbor_offsets(include_center=False):
                nxt = (cxi + dx, cyj + dy)
                if nxt in visited:
                    continue
                if 0 <= nxt[0] < state.dimx and 0 <= nxt[1] < state.dimy:
                    visited.add(nxt)
                    queue.append((nxt[0], nxt[1], dist + 1))
        if candidates:
            return min(
                candidates,
                key=lambda node: _snap_candidate_score(
                    state,
                    node,
                    query_xy=pct_xy,
                    preferred_xy=preferred_xy,
                    requested_slice=si,
                ),
            ), layer_distance
    raise RuntimeError(f"slice {si} 附近找不到可走格")


def _snap_candidate_score(
    state: _State,
    node: tuple[int, int, int],
    *,
    query_xy: np.ndarray,
    preferred_xy: np.ndarray | None,
    requested_slice: int,
) -> tuple[int, float, float, int, int]:
    point_xy = np.asarray(state.grid_to_pct_xyz(node)[:2], dtype=np.float64)
    query_distance = float(np.linalg.norm(point_xy - np.asarray(query_xy)))
    preferred_distance = (
        0.0
        if preferred_xy is None
        else float(np.linalg.norm(point_xy - np.asarray(preferred_xy)))
    )
    return (
        abs(int(node[0]) - int(requested_slice)),
        query_distance,
        preferred_distance,
        int(node[1]),
        int(node[2]),
    )


def _check_planning_interrupt(
    *,
    cancel_check: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    """在搜索边界统一处理取消与单调时钟截止时间。"""

    if cancel_check is not None and cancel_check():
        raise InterruptedError("PCT grid 规划已取消。")
    if (
        deadline_monotonic is not None
        and time.monotonic() >= float(deadline_monotonic)
    ):
        raise TimeoutError("PCT grid 规划超过截止时间。")


def _astar(
    state: _State,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    *,
    hard_obstacle_mask: np.ndarray | None = None,
    vertical_gateway_mask: np.ndarray | None = None,
    allow_vertical_transitions: bool = True,
    vertical_direction: int = 0,
    stair_slice_range: tuple[int, int] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> list[tuple[int, int, int]] | None:
    """执行可取消且受截止时间约束的三维 A* 搜索。"""

    frontier: list[tuple[float, int, tuple[int, int, int]]] = []
    serial = 0
    heapq.heappush(frontier, (0.0, serial, start))
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start: 0.0}
    max_expansions = state.grid_max_expansions
    expansions = 0
    while frontier:
        _check_planning_interrupt(
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )
        _, _, current = heapq.heappop(frontier)
        expansions += 1
        if current == goal:
            return _reconstruct(came_from, current)
        if expansions > max_expansions:
            raise TimeoutError(f"PCT grid A* 超过最大扩展数: {max_expansions}")
        for nxt, step_cost in _neighbors(
            state,
            current,
            hard_obstacle_mask=hard_obstacle_mask,
            vertical_gateway_mask=vertical_gateway_mask,
            allow_vertical_transitions=allow_vertical_transitions,
            vertical_direction=vertical_direction,
            stair_slice_range=stair_slice_range,
        ):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                serial += 1
                heapq.heappush(frontier, (new_cost + _heuristic(state, nxt, goal), serial, nxt))
                came_from[nxt] = current
    return None


def _neighbors(
    state: _State,
    node: tuple[int, int, int],
    *,
    hard_obstacle_mask: np.ndarray | None = None,
    vertical_gateway_mask: np.ndarray | None = None,
    allow_vertical_transitions: bool = True,
    vertical_direction: int = 0,
    stair_slice_range: tuple[int, int] | None = None,
) -> list[tuple[tuple[int, int, int], float]]:
    si, xi, yj = node
    output: list[tuple[tuple[int, int, int], float]] = []
    for dsi in (-1, 0, 1):
        nsi = si + dsi
        if nsi < 0 or nsi >= state.n_slice:
            continue
        if dsi == 0:
            offsets = _xy_neighbor_offsets(include_center=False)
        else:
            offsets = _vertical_xy_neighbor_offsets(state)
        for dx, dy in offsets:
            nxt = (nsi, xi + dx, yj + dy)
            if not state.is_walkable(nxt, hard_obstacle_mask=hard_obstacle_mask):
                continue
            if dsi != 0 and not allow_vertical_transitions:
                continue
            if dsi != 0 and vertical_direction != 0 and dsi != vertical_direction:
                continue
            if dsi != 0 and not _vertical_transition_allowed(
                node,
                nxt,
                vertical_gateway_mask=vertical_gateway_mask,
            ):
                continue
            if dsi != 0 and not _stair_progress_allowed(
                state,
                node,
                nxt,
                vertical_direction=vertical_direction,
                stair_slice_range=stair_slice_range,
            ):
                continue
            if dsi == 0 and not _stair_same_slice_progress_allowed(
                state,
                node,
                nxt,
                vertical_direction=vertical_direction,
                stair_slice_range=stair_slice_range,
            ):
                continue
            if dsi != 0:
                if not _vertical_transition_path_is_walkable(
                    state,
                    node,
                    nxt,
                    hard_obstacle_mask=hard_obstacle_mask,
                ):
                    continue
                output.append(
                    (
                        nxt,
                        _step_cost(state, dsi, dx, dy)
                        + _obstacle_clearance_penalty(
                            state,
                            nxt,
                            hard_obstacle_mask=hard_obstacle_mask,
                        )
                        + _stair_progress_penalty(
                            state,
                            nxt,
                            vertical_direction=vertical_direction,
                            stair_slice_range=stair_slice_range,
                        ),
                    )
                )
                continue
            # 同层对角移动必须通过两个正交邻格，防止中心线从墙角穿过。
            if (
                dx != 0
                and dy != 0
                and (
                    not state.is_walkable(
                        (nsi, xi + dx, yj),
                        hard_obstacle_mask=hard_obstacle_mask,
                    )
                    or not state.is_walkable(
                        (nsi, xi, yj + dy),
                        hard_obstacle_mask=hard_obstacle_mask,
                    )
                )
            ):
                continue
            output.append(
                (
                    nxt,
                    _step_cost(state, dsi, dx, dy)
                    + _obstacle_clearance_penalty(
                        state,
                        nxt,
                        hard_obstacle_mask=hard_obstacle_mask,
                    )
                    + _stair_progress_penalty(
                        state,
                        nxt,
                        vertical_direction=vertical_direction,
                        stair_slice_range=stair_slice_range,
                    ),
                )
            )
    return output


def _vertical_xy_neighbor_offsets(state: _State) -> list[tuple[int, int]]:
    """按楼梯坡度生成相邻 slice 的水平位移候选。"""

    resolution = float(state.resolution)
    min_horizontal = float(
        getattr(state, "stair_min_horizontal_per_slice_m", 0.5 * resolution)
    )
    max_horizontal = float(
        getattr(
            state,
            "stair_max_horizontal_per_slice_m",
            np.sqrt(2.0) * resolution,
        )
    )
    radius_cells = max(1, int(np.ceil(max_horizontal / resolution)))
    offsets: list[tuple[int, int]] = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            horizontal = float(np.hypot(dx, dy) * resolution)
            if (
                horizontal + 1.0e-9 < min_horizontal
                or horizontal > max_horizontal + 1.0e-9
            ):
                continue
            offsets.append((dx, dy))
    return offsets


def _vertical_transition_path_is_walkable(
    state: _State,
    current: tuple[int, int, int],
    nxt: tuple[int, int, int],
    *,
    hard_obstacle_mask: np.ndarray | None,
) -> bool:
    """检查楼梯斜向换层边在目标 slice 上没有穿过墙体或家具。"""

    _si, xi, yj = current
    nsi, nxi, nyj = nxt
    dx = nxi - xi
    dy = nyj - yj
    steps = max(abs(dx), abs(dy))
    if steps < 1:
        return False
    for step in range(1, steps + 1):
        alpha = step / steps
        sample_x = int(round(xi + alpha * dx))
        sample_y = int(round(yj + alpha * dy))
        lower_or_current_walkable = state.is_walkable(
            (_si, sample_x, sample_y),
            hard_obstacle_mask=hard_obstacle_mask,
        )
        upper_or_next_walkable = state.is_walkable(
            (nsi, sample_x, sample_y),
            hard_obstacle_mask=hard_obstacle_mask,
        )
        if not (lower_or_current_walkable or upper_or_next_walkable):
            return False
    return True


def _stair_progress_allowed(
    state: _State,
    current: tuple[int, int, int],
    nxt: tuple[int, int, int],
    *,
    vertical_direction: int,
    stair_slice_range: tuple[int, int] | None = None,
) -> bool:
    """换层点必须沿楼梯入口到出口方向前进，禁止在楼梯内折返升层。"""

    stair_surface_mask = getattr(
        state,
        "cross_floor_stair_vertical_mask",
        None,
    )
    if stair_surface_mask is not None and stair_surface_mask.ndim == 3:
        return True
    progress = getattr(state, "cross_floor_gateway_progress", None)
    if progress is None or vertical_direction == 0:
        return True
    si, xi, yj = current
    nsi, nxi, nyj = nxt
    current_progress = float(progress[xi, yj])
    next_progress = float(progress[nxi, nyj])
    if not np.isfinite(current_progress) or not np.isfinite(next_progress):
        return False
    if vertical_direction > 0:
        monotonic = next_progress + 1.0e-6 >= current_progress
    else:
        monotonic = next_progress <= current_progress + 1.0e-6
    if not monotonic:
        return False
    if stair_slice_range is None:
        return True
    low_slice = min(stair_slice_range)
    high_slice = max(stair_slice_range)
    if high_slice <= low_slice:
        return True
    _current_slice, _xi, _yj = current
    next_slice, _nxi, _nyj = nxt
    expected_by_slice = getattr(
        state,
        "cross_floor_gateway_expected_progress_by_slice",
        None,
    )
    if (
        expected_by_slice is not None
        and 0 <= next_slice < len(expected_by_slice)
        and np.isfinite(float(expected_by_slice[next_slice]))
    ):
        expected_next = float(expected_by_slice[next_slice])
    else:
        expected_next = (float(next_slice) - float(low_slice)) / float(
            high_slice - low_slice
        )
    expected_next = float(np.clip(expected_next, 0.0, 1.0))
    tolerance = float(getattr(state, "stair_progress_tolerance", 1.0))
    return abs(next_progress - expected_next) <= tolerance + 1.0e-6


def _stair_same_slice_progress_allowed(
    state: _State,
    current: tuple[int, int, int],
    nxt: tuple[int, int, int],
    *,
    vertical_direction: int,
    stair_slice_range: tuple[int, int] | None = None,
) -> bool:
    """楼梯中间层内禁止沿入口到出口进度倒退。"""

    progress = getattr(state, "cross_floor_gateway_progress", None)
    if progress is None or vertical_direction == 0 or stair_slice_range is None:
        return True
    current_slice, xi, yj = current
    _next_slice, nxi, nyj = nxt
    low_slice = min(stair_slice_range)
    high_slice = max(stair_slice_range)
    if high_slice <= low_slice:
        return True
    if vertical_direction > 0 and not (low_slice <= current_slice < high_slice):
        return True
    if vertical_direction < 0 and not (low_slice < current_slice <= high_slice):
        return True
    stair_surface_mask = getattr(
        state,
        "cross_floor_stair_vertical_mask",
        None,
    )
    if (
        stair_surface_mask is not None
        and stair_surface_mask.ndim == 3
        and not bool(stair_surface_mask[_next_slice, nxi, nyj])
    ):
        return False
    if stair_surface_mask is not None and stair_surface_mask.ndim == 3:
        return True
    current_progress = float(progress[xi, yj])
    next_progress = float(progress[nxi, nyj])
    if not np.isfinite(current_progress) or not np.isfinite(next_progress):
        return True
    tolerance = float(getattr(state, "stair_progress_tolerance", 0.0))
    expected_by_slice = getattr(
        state,
        "cross_floor_gateway_expected_progress_by_slice",
        None,
    )
    if (
        expected_by_slice is not None
        and 0 <= current_slice < len(expected_by_slice)
        and np.isfinite(float(expected_by_slice[current_slice]))
    ):
        expected_current = float(expected_by_slice[current_slice])
        if vertical_direction > 0:
            if next_progress > expected_current + tolerance + 1.0e-6:
                return False
        elif next_progress < expected_current - tolerance - 1.0e-6:
            return False
    if vertical_direction > 0:
        return next_progress + tolerance + 1.0e-6 >= current_progress
    return next_progress <= current_progress + tolerance + 1.0e-6


def _stair_progress_penalty(
    state: _State,
    node: tuple[int, int, int],
    *,
    vertical_direction: int,
    stair_slice_range: tuple[int, int] | None,
) -> float:
    """对楼梯内高度 slice 与折线进度不匹配的节点增加软代价。"""

    if vertical_direction == 0 or stair_slice_range is None:
        return 0.0
    stair_surface_mask = getattr(
        state,
        "cross_floor_stair_vertical_mask",
        None,
    )
    if stair_surface_mask is not None and stair_surface_mask.ndim == 3:
        distance = getattr(
            state,
            "cross_floor_stair_surface_distance_m",
            None,
        )
        if distance is None:
            return 0.0
        slice_index, xi, yj = node
        center_distance = float(distance[slice_index, xi, yj])
        if not np.isfinite(center_distance):
            return 0.0
        weight = float(getattr(state, "stair_progress_cost_weight", 0.0))
        return float(weight * center_distance * center_distance)
    weight = float(getattr(state, "stair_progress_cost_weight", 0.0))
    if weight <= 0.0:
        return 0.0
    progress = getattr(state, "cross_floor_gateway_progress", None)
    expected_by_slice = getattr(
        state,
        "cross_floor_gateway_expected_progress_by_slice",
        None,
    )
    if progress is None or expected_by_slice is None:
        return 0.0
    slice_index, xi, yj = node
    low_slice = min(stair_slice_range)
    high_slice = max(stair_slice_range)
    if not (low_slice <= slice_index <= high_slice):
        return 0.0
    if not (0 <= slice_index < len(expected_by_slice)):
        return 0.0
    current_progress = float(progress[xi, yj])
    expected_progress = float(expected_by_slice[slice_index])
    if not (
        np.isfinite(current_progress)
        and np.isfinite(expected_progress)
    ):
        return 0.0
    delta = current_progress - expected_progress
    return float(weight * delta * delta)


def _obstacle_clearance_penalty(
    state: _State,
    node: tuple[int, int, int],
    *,
    hard_obstacle_mask: np.ndarray | None,
) -> float:
    """对靠近墙体和家具的节点增加软代价，避免硬膨胀封死门洞。"""

    radius = float(getattr(state, "obstacle_clearance_radius_m", 0.0))
    weight = float(getattr(state, "obstacle_clearance_cost_weight", 0.0))
    clearance_fn = getattr(state, "obstacle_clearance_m", None)
    if radius <= 0.0 or weight <= 0.0 or clearance_fn is None:
        return 0.0
    clearance = float(
        clearance_fn(node, hard_obstacle_mask=hard_obstacle_mask)
    )
    deficit = max(0.0, radius - clearance)
    return float(weight * deficit * deficit)


def _vertical_transition_allowed(
    current: tuple[int, int, int],
    nxt: tuple[int, int, int],
    *,
    vertical_gateway_mask: np.ndarray | None,
) -> bool:
    """如果配置了楼梯/坡道 gateway，只允许 gateway 内发生跨 slice 邻接。"""

    if vertical_gateway_mask is None:
        return True
    si, xi, yj = current
    nsi, nxi, nyj = nxt
    if vertical_gateway_mask.ndim == 3:
        return bool(
            0 <= si < vertical_gateway_mask.shape[0]
            and 0 <= nsi < vertical_gateway_mask.shape[0]
            and 0 <= xi < vertical_gateway_mask.shape[1]
            and 0 <= yj < vertical_gateway_mask.shape[2]
            and 0 <= nxi < vertical_gateway_mask.shape[1]
            and 0 <= nyj < vertical_gateway_mask.shape[2]
            and vertical_gateway_mask[si, xi, yj]
            and vertical_gateway_mask[nsi, nxi, nyj]
        )
    if not (
        0 <= xi < vertical_gateway_mask.shape[0]
        and 0 <= yj < vertical_gateway_mask.shape[1]
        and 0 <= nxi < vertical_gateway_mask.shape[0]
        and 0 <= nyj < vertical_gateway_mask.shape[1]
    ):
        return False
    return bool(vertical_gateway_mask[xi, yj] and vertical_gateway_mask[nxi, nyj])


def _xy_neighbor_offsets(*, include_center: bool) -> list[tuple[int, int]]:
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if include_center:
        return [(0, 0), *offsets]
    return offsets


def _step_cost(state: _State, dsi: int, dx: int, dy: int) -> float:
    horizontal = np.hypot(dx, dy) * state.resolution
    vertical = abs(dsi) * state.slice_dh
    return float(np.hypot(horizontal, vertical))


def _slice_direction(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> int:
    """返回跨层路径允许的单调 slice 方向。"""

    return int(np.sign(end[0] - start[0]))


def _obstacle_mask_mode(mask: np.ndarray) -> str:
    """记录响应所使用的障碍 mask 维度，便于定位跨楼层误封堵。"""

    return "body_clearance_volume" if mask.ndim == 3 else "xy_projection"


def _approximate_obstacle_clearance(
    obstacle_mask: np.ndarray,
    *,
    resolution: float,
    max_distance_m: float,
) -> np.ndarray:
    """用有限步 XY 膨胀近似障碍距离，避免依赖 scipy。"""

    max_distance = float(max_distance_m)
    if max_distance <= 0.0:
        return np.zeros_like(obstacle_mask, dtype=np.float32)
    distance = np.full(
        obstacle_mask.shape,
        max_distance + float(resolution),
        dtype=np.float32,
    )
    distance[obstacle_mask] = 0.0
    reached = obstacle_mask.astype(bool, copy=True)
    step_count = max(1, int(np.ceil(max_distance / float(resolution))))
    for step in range(1, step_count + 1):
        expanded = _expand_obstacle_xy(reached)
        new_cells = expanded & ~reached
        distance[new_cells] = min(step * float(resolution), max_distance)
        reached = expanded
    return distance


def _expand_obstacle_xy(mask: np.ndarray) -> np.ndarray:
    """把二维或逐 slice 障碍沿 XY 扩张一个八邻域。"""

    output = mask.copy()
    x_axis = mask.ndim - 2
    y_axis = mask.ndim - 1
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            source = [slice(None)] * mask.ndim
            target = [slice(None)] * mask.ndim
            if dx > 0:
                source[x_axis] = slice(0, -dx)
                target[x_axis] = slice(dx, None)
            elif dx < 0:
                source[x_axis] = slice(-dx, None)
                target[x_axis] = slice(0, dx)
            if dy > 0:
                source[y_axis] = slice(0, -dy)
                target[y_axis] = slice(dy, None)
            elif dy < 0:
                source[y_axis] = slice(-dy, None)
                target[y_axis] = slice(0, dy)
            output[tuple(target)] |= mask[tuple(source)]
    return output


def _same_floor_direct_path(
    state: _State,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    *,
    corridor_radius_cells: int = 1,
    hard_obstacle_mask: np.ndarray | None = None,
) -> list[tuple[int, int, int]] | None:
    """同层直线及机器人宽度邻域均可走时返回两点捷径。"""

    if abs(end[0] - start[0]) > 1:
        return None
    radius = max(0, int(corridor_radius_cells))
    sample_count = max(
        1,
        2 * max(abs(end[1] - start[1]), abs(end[2] - start[2])),
    )
    for sample_index in range(sample_count + 1):
        alpha = sample_index / sample_count
        center_slice = int(round(start[0] + alpha * (end[0] - start[0])))
        center_x = int(round(start[1] + alpha * (end[1] - start[1])))
        center_y = int(round(start[2] + alpha * (end[2] - start[2])))
        slice_min = max(0, center_slice - 1)
        slice_max = min(state.n_slice, center_slice + 2)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if not any(
                    state.is_walkable(
                        (slice_index, center_x + dx, center_y + dy),
                        hard_obstacle_mask=hard_obstacle_mask,
                    )
                    for slice_index in range(slice_min, slice_max)
                ):
                    return None
    return [start, end]


def _heuristic(state: _State, node: tuple[int, int, int], goal: tuple[int, int, int]) -> float:
    dsi = abs(node[0] - goal[0]) * state.slice_dh
    dxy = np.hypot(node[1] - goal[1], node[2] - goal[2]) * state.resolution
    return float(np.hypot(dxy, dsi))


def _reconstruct(
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None],
    current: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _compress_path(
    path: list[list[float]],
    *,
    max_segment_length_m: float = 0.80,
) -> list[list[float]]:
    if len(path) <= 2:
        return path
    compressed = [path[0]]
    last_direction: tuple[int, int, int] | None = None
    distance_since_keep = 0.0
    max_segment_length = float(max_segment_length_m)
    if not np.isfinite(max_segment_length) or max_segment_length <= 0.0:
        raise ValueError("PCT grid 最大压缩段长必须为有限正数。")
    prev = np.asarray(path[0], dtype=np.float64)
    for raw in path[1:]:
        current = np.asarray(raw, dtype=np.float64)
        delta = current - prev
        distance_since_keep += float(np.linalg.norm(delta))
        direction = tuple(int(np.sign(value)) for value in delta)
        if abs(float(delta[2])) > 1.0e-9:
            _append_compressed_point(compressed, prev.tolist())
            _append_compressed_point(compressed, current.tolist())
            distance_since_keep = 0.0
            last_direction = None
            prev = current
            continue
        if last_direction is not None and direction != last_direction:
            _append_compressed_point(compressed, prev.tolist())
            distance_since_keep = float(np.linalg.norm(current - prev))
        elif (
            max_segment_length > 0.0
            and distance_since_keep >= max_segment_length
        ):
            _append_compressed_point(compressed, current.tolist())
            distance_since_keep = 0.0
        last_direction = direction
        prev = current
    _append_compressed_point(compressed, path[-1])
    return compressed


def _append_compressed_point(path: list[list[float]], point: list[float]) -> None:
    """追加压缩点时去掉连续重复 waypoint。"""

    if path and all(abs(a - b) <= 1.0e-9 for a, b in zip(path[-1], point)):
        return
    path.append(point)


if __name__ == "__main__":
    raise SystemExit(main())
