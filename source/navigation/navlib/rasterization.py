"""Conservative XY rasterization helpers for USD collision triangles."""

from __future__ import annotations

import math

import numpy as np

from .grid_map import OccupancyGridMap


def rasterize_triangles_xy(
    triangles: list[np.ndarray],
    *,
    resolution: float,
    bounds: tuple[float, float, float, float],
) -> OccupancyGridMap:
    """Project collision triangles to a 2D occupancy grid.

    Triangle edges are rasterized even when their XY projection has zero area.
    This preserves vertical walls that remain physically collidable in Isaac
    Lab but would otherwise disappear from a top-down occupancy map.
    """

    min_x, max_x, min_y, max_y = bounds
    width = max(1, int(math.ceil((max_x - min_x) / resolution)))
    height = max(1, int(math.ceil((max_y - min_y) / resolution)))
    occupancy = np.zeros((height, width), dtype=bool)

    for triangle in triangles:
        tri_xy = np.asarray(triangle, dtype=np.float64)[:, :2]
        _rasterize_triangle_edges(occupancy, tri_xy, resolution=resolution, bounds=bounds)
        if _triangle_area_xy(tri_xy) < 1.0e-9:
            continue

        tri_min_x = float(np.min(tri_xy[:, 0]))
        tri_max_x = float(np.max(tri_xy[:, 0]))
        tri_min_y = float(np.min(tri_xy[:, 1]))
        tri_max_y = float(np.max(tri_xy[:, 1]))

        col_min = max(0, int(math.floor((tri_min_x - min_x) / resolution)))
        col_max = min(width - 1, int(math.floor((tri_max_x - min_x) / resolution)))
        row_top = max(0, int(math.floor((max_y - tri_max_y) / resolution)))
        row_bottom = min(height - 1, int(math.floor((max_y - tri_min_y) / resolution)))
        if row_top > row_bottom or col_min > col_max:
            continue

        cols = np.arange(col_min, col_max + 1, dtype=np.int32)
        rows = np.arange(row_top, row_bottom + 1, dtype=np.int32)
        xs = min_x + (cols.astype(np.float64) + 0.5) * resolution
        ys = max_y - (rows.astype(np.float64) + 0.5) * resolution
        sample_x, sample_y = np.meshgrid(xs, ys)
        inside = _points_in_triangle(sample_x, sample_y, tri_xy)
        occupancy[row_top : row_bottom + 1, col_min : col_max + 1] |= inside

    return OccupancyGridMap(
        occupancy=occupancy,
        resolution=resolution,
        origin=(min_x, min_y, 0.0),
    )


def _rasterize_triangle_edges(
    occupancy: np.ndarray,
    triangle_xy: np.ndarray,
    *,
    resolution: float,
    bounds: tuple[float, float, float, float],
) -> None:
    for index in range(3):
        _rasterize_segment(
            occupancy,
            triangle_xy[index],
            triangle_xy[(index + 1) % 3],
            resolution=resolution,
            bounds=bounds,
        )


def _rasterize_segment(
    occupancy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    resolution: float,
    bounds: tuple[float, float, float, float],
) -> None:
    length = float(np.linalg.norm(end_xy - start_xy))
    steps = max(1, int(math.ceil(length / max(0.25 * resolution, 1.0e-9))))
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        point = start_xy + alpha * (end_xy - start_xy)
        row, col = _world_to_grid(float(point[0]), float(point[1]), resolution=resolution, bounds=bounds)
        if 0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]:
            occupancy[row, col] = True


def _world_to_grid(
    x: float,
    y: float,
    *,
    resolution: float,
    bounds: tuple[float, float, float, float],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    width = max(1, int(math.ceil((max_x - min_x) / resolution)))
    height = max(1, int(math.ceil((max_y - min_y) / resolution)))
    col = min(width - 1, int(math.floor((x - min_x) / resolution)))
    row_from_bottom = int(math.floor((y - min_y) / resolution))
    row = height - 1 - row_from_bottom
    return row, col


def _triangle_area_xy(triangle_xy: np.ndarray) -> float:
    a = triangle_xy[1] - triangle_xy[0]
    b = triangle_xy[2] - triangle_xy[0]
    return abs(a[0] * b[1] - a[1] * b[0]) * 0.5


def _points_in_triangle(sample_x: np.ndarray, sample_y: np.ndarray, triangle_xy: np.ndarray) -> np.ndarray:
    x1, y1 = triangle_xy[0]
    x2, y2 = triangle_xy[1]
    x3, y3 = triangle_xy[2]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1.0e-12:
        return np.zeros_like(sample_x, dtype=bool)
    a = ((y2 - y3) * (sample_x - x3) + (x3 - x2) * (sample_y - y3)) / denom
    b = ((y3 - y1) * (sample_x - x3) + (x1 - x3) * (sample_y - y3)) / denom
    c = 1.0 - a - b
    eps = 1.0e-9
    return (a >= -eps) & (b >= -eps) & (c >= -eps)
