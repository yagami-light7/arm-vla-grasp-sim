#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source.navigation.navlib import OccupancyGridMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a navigation map and list candidate free-space points.")
    parser.add_argument("--map", required=True, help="Path to a nav-map metadata file (.json/.yaml).")
    parser.add_argument(
        "--world",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Check a world-frame point and report its grid cell and occupancy state.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROW", "COL"),
        help="Check a grid cell and report its world-frame center and occupancy state.",
    )
    parser.add_argument(
        "--list-free",
        type=int,
        default=12,
        help="List up to this many candidate free-space world points. Set to 0 to disable.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.25,
        help="Safety clearance in meters used when selecting candidate free-space points.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.75,
        help="Minimum spacing in meters between listed candidate points.",
    )
    return parser.parse_args()


def map_bounds(grid_map: OccupancyGridMap) -> tuple[float, float, float, float]:
    corners = [
        grid_map.grid_to_world(row, col)
        for row, col in [
            (0, 0),
            (0, grid_map.width - 1),
            (grid_map.height - 1, 0),
            (grid_map.height - 1, grid_map.width - 1),
        ]
    ]
    return (
        min(point[0] for point in corners),
        max(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[1] for point in corners),
    )


def describe_world_point(grid_map: OccupancyGridMap, clearance_map: OccupancyGridMap, x: float, y: float):
    row, col = grid_map.world_to_grid(x, y)
    in_bounds = grid_map.in_bounds(row, col)
    raw_state = "occupied" if grid_map.is_occupied(row, col) else "free"
    clearance_state = "occupied" if clearance_map.is_occupied(row, col) else "free"
    print(f"[WORLD] ({x:.3f}, {y:.3f}) -> grid=({row}, {col}) in_bounds={in_bounds}")
    print(f"[WORLD] raw_map={raw_state} clearance_map={clearance_state}")
    if in_bounds:
        cx, cy = grid_map.grid_to_world(row, col)
        print(f"[WORLD] cell_center=({cx:.3f}, {cy:.3f})")


def describe_grid_cell(grid_map: OccupancyGridMap, clearance_map: OccupancyGridMap, row: int, col: int):
    in_bounds = grid_map.in_bounds(row, col)
    raw_state = "occupied" if grid_map.is_occupied(row, col) else "free"
    clearance_state = "occupied" if clearance_map.is_occupied(row, col) else "free"
    print(f"[GRID] ({row}, {col}) in_bounds={in_bounds}")
    print(f"[GRID] raw_map={raw_state} clearance_map={clearance_state}")
    if in_bounds:
        x, y = grid_map.grid_to_world(row, col)
        print(f"[GRID] world_center=({x:.3f}, {y:.3f})")


def list_candidate_points(grid_map: OccupancyGridMap, clearance_map: OccupancyGridMap, limit: int, spacing_m: float):
    if limit <= 0:
        return

    spacing_cells = max(1, int(math.ceil(spacing_m / grid_map.resolution)))
    row_offset = spacing_cells // 2
    col_offset = spacing_cells // 2
    candidates: list[tuple[float, float, int, int]] = []

    for row in range(row_offset, clearance_map.height, spacing_cells):
        for col in range(col_offset, clearance_map.width, spacing_cells):
            if clearance_map.is_occupied(row, col):
                continue
            x, y = clearance_map.grid_to_world(row, col)
            candidates.append((x, y, row, col))
            if len(candidates) >= limit:
                print("[FREE] Candidate world points with requested clearance:")
                for index, (px, py, prow, pcol) in enumerate(candidates, start=1):
                    print(f"[FREE] {index:02d}: world=({px:.3f}, {py:.3f}) grid=({prow}, {pcol})")
                return

    print("[FREE] Candidate world points with requested clearance:")
    if not candidates:
        print("[FREE] none found; try lowering --clearance or --spacing.")
        return
    for index, (px, py, prow, pcol) in enumerate(candidates, start=1):
        print(f"[FREE] {index:02d}: world=({px:.3f}, {py:.3f}) grid=({prow}, {pcol})")


def main():
    args = parse_args()
    grid_map = OccupancyGridMap.from_meta_file(args.map)
    clearance_map = grid_map.inflate(args.clearance)

    x_min, x_max, y_min, y_max = map_bounds(grid_map)
    print(f"[INFO] map={Path(args.map).expanduser().resolve()}")
    print(f"[INFO] size={grid_map.width}x{grid_map.height} resolution={grid_map.resolution:.3f} m/cell")
    print(f"[INFO] world_bounds x=[{x_min:.3f}, {x_max:.3f}] y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"[INFO] origin=({grid_map.origin[0]:.3f}, {grid_map.origin[1]:.3f}, {grid_map.origin[2]:.3f})")
    print(f"[INFO] clearance={args.clearance:.3f} m spacing={args.spacing:.3f} m")

    if args.world is not None:
        describe_world_point(grid_map, clearance_map, float(args.world[0]), float(args.world[1]))
    if args.grid is not None:
        describe_grid_cell(grid_map, clearance_map, int(args.grid[0]), int(args.grid[1]))

    list_candidate_points(grid_map, clearance_map, args.list_free, args.spacing)


if __name__ == "__main__":
    main()
