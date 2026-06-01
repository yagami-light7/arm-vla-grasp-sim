from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .grid_map import OccupancyGridMap


@dataclass(frozen=True)
class AStarPlanResult:
    raw_path_grid: list[tuple[int, int]]
    raw_path_world: list[tuple[float, float]]
    path_grid: list[tuple[int, int]]
    path_world: list[tuple[float, float]]
    cost: float
    expanded_nodes: int
    start_grid: tuple[int, int]
    goal_grid: tuple[int, int]


class AStarPlanner:
    """Classic 8-connected A* over an occupancy grid."""

    def __init__(self, allow_diagonal: bool = True, heuristic_weight: float = 1.0):
        self.allow_diagonal = allow_diagonal
        self.heuristic_weight = heuristic_weight

    def plan(
        self,
        grid_map: OccupancyGridMap,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        snap_to_free: bool = True,
        max_snap_distance_m: float = 0.5,
    ) -> AStarPlanResult:
        start_rc = grid_map.world_to_grid(*start_xy)
        goal_rc = grid_map.world_to_grid(*goal_xy)
        max_snap_cells = max(0, int(math.ceil(max_snap_distance_m / grid_map.resolution)))

        if snap_to_free:
            snapped_start = grid_map.nearest_free_cell(start_rc, max_snap_cells)
            snapped_goal = grid_map.nearest_free_cell(goal_rc, max_snap_cells)
            if snapped_start is None:
                raise ValueError(f"start point {start_xy} is not in free space and could not be snapped.")
            if snapped_goal is None:
                raise ValueError(f"goal point {goal_xy} is not in free space and could not be snapped.")
            start_rc, goal_rc = snapped_start, snapped_goal

        if grid_map.is_occupied(*start_rc):
            raise ValueError(f"start cell {start_rc} is occupied.")
        if grid_map.is_occupied(*goal_rc):
            raise ValueError(f"goal cell {goal_rc} is occupied.")

        raw_path_grid, cost, expanded = self._plan_grid(grid_map, start_rc, goal_rc)
        raw_path_world = [grid_map.grid_to_world(row, col) for row, col in raw_path_grid]
        path_grid = _prune_collinear(raw_path_grid)
        path_world = [grid_map.grid_to_world(row, col) for row, col in path_grid]
        return AStarPlanResult(
            raw_path_grid=raw_path_grid,
            raw_path_world=raw_path_world,
            path_grid=path_grid,
            path_world=path_world,
            cost=cost,
            expanded_nodes=expanded,
            start_grid=start_rc,
            goal_grid=goal_rc,
        )

    def _plan_grid(
        self,
        grid_map: OccupancyGridMap,
        start_rc: tuple[int, int],
        goal_rc: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], float, int]:
        height, width = grid_map.shape
        g_cost = np.full((height, width), np.inf, dtype=np.float64)
        closed = np.zeros((height, width), dtype=bool)
        parents = np.full((height, width, 2), -1, dtype=np.int32)
        neighbors = _neighbor_table(self.allow_diagonal)

        g_cost[start_rc] = 0.0
        open_heap: list[tuple[float, float, int, int]] = [
            (self._heuristic(start_rc, goal_rc), 0.0, start_rc[0], start_rc[1])
        ]
        expanded_nodes = 0

        while open_heap:
            _, current_g, row, col = heapq.heappop(open_heap)
            if closed[row, col]:
                continue
            closed[row, col] = True
            expanded_nodes += 1
            current = (row, col)

            if current == goal_rc:
                return _reconstruct_path(parents, start_rc, goal_rc), current_g, expanded_nodes

            for dr, dc, step_cost in neighbors:
                next_row = row + dr
                next_col = col + dc
                if not grid_map.in_bounds(next_row, next_col):
                    continue
                if grid_map.is_occupied(next_row, next_col):
                    continue
                if dr != 0 and dc != 0:
                    if grid_map.is_occupied(row + dr, col) or grid_map.is_occupied(row, col + dc):
                        continue
                candidate_g = current_g + step_cost
                if candidate_g >= g_cost[next_row, next_col]:
                    continue
                g_cost[next_row, next_col] = candidate_g
                parents[next_row, next_col] = (row, col)
                heapq.heappush(
                    open_heap,
                    (
                        candidate_g + self.heuristic_weight * self._heuristic((next_row, next_col), goal_rc),
                        candidate_g,
                        next_row,
                        next_col,
                    ),
                )

        raise RuntimeError(f"A* failed to find a path from {start_rc} to {goal_rc}.")

    @staticmethod
    def _heuristic(current: tuple[int, int], goal: tuple[int, int]) -> float:
        return math.hypot(goal[0] - current[0], goal[1] - current[1])


def _neighbor_table(allow_diagonal: bool) -> list[tuple[int, int, float]]:
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allow_diagonal:
        diag = math.sqrt(2.0)
        neighbors.extend([(-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)])
    return neighbors


def _reconstruct_path(
    parents: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [goal_rc]
    current = goal_rc
    while current != start_rc:
        parent = tuple(int(v) for v in parents[current])  # type: ignore[arg-type]
        if parent == (-1, -1):
            raise RuntimeError("encountered a broken parent chain while reconstructing the A* path.")
        current = parent
        path.append(current)
    path.reverse()
    return path


def _prune_collinear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return list(path)

    pruned = [path[0]]
    last_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for index in range(1, len(path) - 1):
        next_dir = (path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1])
        if next_dir != last_dir:
            pruned.append(path[index])
        last_dir = next_dir
    pruned.append(path[-1])
    return pruned
