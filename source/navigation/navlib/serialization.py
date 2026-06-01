from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .astar import AStarPlanResult
from .grid_map import OccupancyGridMap


def save_path_bundle(
    output_path: str | Path,
    *,
    grid_map: OccupancyGridMap,
    plan: AStarPlanResult,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    inflation_radius_m: float,
):
    output_path = Path(output_path)
    data = {
        "map_meta": str(grid_map.meta_path) if grid_map.meta_path is not None else None,
        "map_image": str(grid_map.image_path) if grid_map.image_path is not None else None,
        "resolution": grid_map.resolution,
        "origin": list(grid_map.origin),
        "start_world": list(start_xy),
        "goal_world": list(goal_xy),
        "start_grid": list(plan.start_grid),
        "goal_grid": list(plan.goal_grid),
        "inflation_radius_m": inflation_radius_m,
        "cost": plan.cost,
        "expanded_nodes": plan.expanded_nodes,
        "raw_path_world": [[float(x), float(y)] for x, y in plan.raw_path_world],
        "raw_path_grid": [[int(r), int(c)] for r, c in plan.raw_path_grid],
        "path_world": [[float(x), float(y)] for x, y in plan.path_world],
        "path_grid": [[int(r), int(c)] for r, c in plan.path_grid],
    }
    output_path.write_text(json.dumps(data, indent=2))


def load_path_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def render_plan_preview(
    grid_map: OccupancyGridMap,
    *,
    path_grid: list[tuple[int, int]] | None = None,
    start_grid: tuple[int, int] | None = None,
    goal_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    rgb = np.where(grid_map.occupancy[..., None], 0, 255).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=2)

    if path_grid is not None:
        _draw_path(rgb, grid_map, path_grid, np.array([220, 30, 30], dtype=np.uint8))
    if start_grid is not None and grid_map.in_bounds(*start_grid):
        _draw_disc(rgb, start_grid, radius=3, color=np.array([40, 180, 40], dtype=np.uint8))
    if goal_grid is not None and grid_map.in_bounds(*goal_grid):
        _draw_disc(rgb, goal_grid, radius=3, color=np.array([40, 80, 220], dtype=np.uint8))
    return rgb


def write_ppm(path: str | Path, rgb: np.ndarray):
    path = Path(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PPM writer expects an HxWx3 RGB array.")
    header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(rgb.astype(np.uint8).tobytes())


def _draw_disc(rgb: np.ndarray, center: tuple[int, int], radius: int, color: np.ndarray):
    row_center, col_center = center
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            row = row_center + dr
            col = col_center + dc
            if 0 <= row < rgb.shape[0] and 0 <= col < rgb.shape[1]:
                rgb[row, col] = color


def _draw_path(rgb: np.ndarray, grid_map: OccupancyGridMap, path_grid: list[tuple[int, int]], color: np.ndarray):
    if not path_grid:
        return
    for index, point in enumerate(path_grid):
        row, col = point
        if grid_map.in_bounds(row, col):
            rgb[row, col] = color
        if index == 0:
            continue
        prev_row, prev_col = path_grid[index - 1]
        for draw_row, draw_col in _bresenham(prev_row, prev_col, row, col):
            if grid_map.in_bounds(draw_row, draw_col):
                rgb[draw_row, draw_col] = color


def _bresenham(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1
    row = row0
    col = col0

    if d_col > d_row:
        error = d_col // 2
        while col != col1:
            cells.append((row, col))
            error -= d_row
            if error < 0:
                row += step_row
                error += d_col
            col += step_col
    else:
        error = d_row // 2
        while row != row1:
            cells.append((row, col))
            error -= d_col
            if error < 0:
                col += step_col
                error += d_row
            row += step_row
    cells.append((row1, col1))
    return cells
