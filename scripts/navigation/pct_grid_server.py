#!/usr/bin/env python3

"""本仓库迁移版 PCT stdin/stdout JSON 栅格规划 server。"""

from __future__ import annotations

import heapq
import json
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOMOGRAM = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_WALKABLE = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
BARRIER = 49.0


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

    def __init__(self, *, tomogram: dict[str, Any], walkable: np.ndarray) -> None:
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

    def is_walkable(self, node: tuple[int, int, int]) -> bool:
        si, xi, yj = node
        if not (0 <= si < self.n_slice and 0 <= xi < self.dimx and 0 <= yj < self.dimy):
            return False
        trav_ok = 0.0 < float(self.traversability[si, xi, yj]) < BARRIER
        ply_ok = bool(self.walkable[si, xi, yj])
        return trav_ok or ply_ok


def _load_state() -> _State:
    tomogram_path = Path(os.environ.get("PCT_TOMOGRAM_PATH", os.fspath(DEFAULT_TOMOGRAM))).expanduser()
    walkable_path = Path(os.environ.get("PCT_WALKABLE_PATH", os.fspath(DEFAULT_WALKABLE))).expanduser()
    with tomogram_path.open("rb") as stream:
        tomogram = pickle.load(stream)
    walkable = np.load(walkable_path)
    return _State(tomogram=tomogram, walkable=walkable)


def _handle_request(state: _State, line: str) -> dict[str, Any]:
    try:
        request = json.loads(line)
        start = np.asarray(request["start"], dtype=np.float64)
        end = np.asarray(request["end"], dtype=np.float64)
        start_slice = state.z_to_slice(float(start[2]))
        end_slice = state.z_to_slice(float(end[2]))
        start_node, start_dist = _snap_to_walkable(state, start[:2], start_slice)
        end_node, end_dist = _snap_to_walkable(state, end[:2], end_slice)
        path = _astar(state, start_node, end_node)
        if path is None:
            return {
                "status": "no_path",
                "slice_start": start_slice,
                "slice_end": end_slice,
                "snap_start_dist": start_dist,
                "snap_end_dist": end_dist,
                "planner": "pct_grid",
            }
        return {
            "status": "ok",
            "traj": _compress_path([state.grid_to_pct_xyz(node) for node in path]),
            "slice_start": start_slice,
            "slice_end": end_slice,
            "snap_start_dist": start_dist,
            "snap_end_dist": end_dist,
            "planner": "pct_grid",
        }
    except Exception as exc:
        import traceback

        return {"status": "error", "msg": str(exc), "traceback": traceback.format_exc()}


def _snap_to_walkable(state: _State, pct_xy: np.ndarray, si: int) -> tuple[tuple[int, int, int], int]:
    xi, yj = state.pct_xy_to_grid(pct_xy)
    start = (si, xi, yj)
    if state.is_walkable(start):
        return start, 0
    visited = {(si, xi, yj)}
    queue = deque([(si, xi, yj, 0)])
    while queue:
        csi, cxi, cyj, dist = queue.popleft()
        node = (csi, cxi, cyj)
        if state.is_walkable(node):
            return node, dist
        for nsi in range(max(0, csi - 1), min(state.n_slice, csi + 2)):
            for dx, dy in _xy_neighbor_offsets(include_center=True):
                nxt = (nsi, cxi + dx, cyj + dy)
                if nxt in visited:
                    continue
                if 0 <= nxt[1] < state.dimx and 0 <= nxt[2] < state.dimy:
                    visited.add(nxt)
                    queue.append((nxt[0], nxt[1], nxt[2], dist + 1))
    raise RuntimeError(f"slice {si} 附近找不到可走格")


def _astar(state: _State, start: tuple[int, int, int], goal: tuple[int, int, int]) -> list[tuple[int, int, int]] | None:
    frontier: list[tuple[float, int, tuple[int, int, int]]] = []
    serial = 0
    heapq.heappush(frontier, (0.0, serial, start))
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int, int], float] = {start: 0.0}
    max_expansions = int(os.environ.get("PCT_GRID_MAX_EXPANSIONS", "1500000"))
    expansions = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        expansions += 1
        if current == goal:
            return _reconstruct(came_from, current)
        if expansions > max_expansions:
            raise TimeoutError(f"PCT grid A* 超过最大扩展数: {max_expansions}")
        for nxt, step_cost in _neighbors(state, current):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                serial += 1
                heapq.heappush(frontier, (new_cost + _heuristic(state, nxt, goal), serial, nxt))
                came_from[nxt] = current
    return None


def _neighbors(state: _State, node: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], float]]:
    si, xi, yj = node
    output: list[tuple[tuple[int, int, int], float]] = []
    for dsi in (-1, 0, 1):
        nsi = si + dsi
        if nsi < 0 or nsi >= state.n_slice:
            continue
        for dx, dy in _xy_neighbor_offsets(include_center=dsi != 0):
            if dx == 0 and dy == 0 and dsi == 0:
                continue
            nxt = (nsi, xi + dx, yj + dy)
            if state.is_walkable(nxt):
                output.append((nxt, _step_cost(state, dsi, dx, dy)))
    return output


def _xy_neighbor_offsets(*, include_center: bool) -> list[tuple[int, int]]:
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if include_center:
        return [(0, 0), *offsets]
    return offsets


def _step_cost(state: _State, dsi: int, dx: int, dy: int) -> float:
    horizontal = np.hypot(dx, dy) * state.resolution
    vertical = abs(dsi) * state.slice_dh
    return float(np.hypot(horizontal, vertical))


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


def _compress_path(path: list[list[float]]) -> list[list[float]]:
    if len(path) <= 2:
        return path
    compressed = [path[0]]
    last_direction: tuple[int, int, int] | None = None
    prev = np.asarray(path[0], dtype=np.float64)
    for raw in path[1:]:
        current = np.asarray(raw, dtype=np.float64)
        delta = current - prev
        direction = tuple(int(np.sign(value)) for value in delta)
        if last_direction is not None and direction != last_direction:
            compressed.append(prev.tolist())
        last_direction = direction
        prev = current
    compressed.append(path[-1])
    return compressed


if __name__ == "__main__":
    raise SystemExit(main())
