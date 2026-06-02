#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source.navigation.navlib import AStarPlanner, DWAConfig, DWAController, OccupancyGridMap, write_ppm


FREE_COLOR = np.array([255, 255, 255], dtype=np.uint8)
OCCUPIED_COLOR = np.array([20, 20, 20], dtype=np.uint8)
ASTAR_RAW_PATH_COLOR = np.array([220, 45, 45], dtype=np.uint8)
ASTAR_PRUNED_PATH_COLOR = np.array([255, 170, 0], dtype=np.uint8)
DWA_POINT_COLOR = np.array([40, 180, 220], dtype=np.uint8)
DWA_ROLLOUT_COLOR = np.array([175, 60, 210], dtype=np.uint8)
START_COLOR = np.array([40, 180, 40], dtype=np.uint8)
GOAL_COLOR = np.array([40, 80, 220], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the A* path and the densified DWA tracking points for a given nav map."
    )
    parser.add_argument("--map", required=True, help="Path to a nav-map metadata file (.json/.yaml).")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="Start point in world coordinates.")
    parser.add_argument("--start-yaw", type=float, default=0.0, help="Initial base yaw in radians for the DWA rollout.")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"), help="Goal point in world coordinates.")
    parser.add_argument(
        "--inflate-radius",
        type=float,
        default=0.40,
        help="Obstacle inflation radius in meters used for the A* planning map.",
    )
    parser.add_argument(
        "--local-clearance-radius",
        type=float,
        default=0.35,
        help="Obstacle inflation radius in meters used by the local DWA controller.",
    )
    parser.add_argument(
        "--lookahead-distance",
        type=float,
        default=0.6,
        help="DWA lookahead distance used when densifying/tracking the A* path.",
    )
    parser.add_argument("--prediction-horizon", type=float, default=1.8, help="DWA candidate rollout horizon in seconds.")
    parser.add_argument("--goal-tolerance", type=float, default=0.15, help="Goal distance tolerance used by DWA.")
    parser.add_argument(
        "--snap-to-free",
        action="store_true",
        help="Allow diagnostic-only snapping of blocked endpoints to nearby free cells.",
    )
    parser.add_argument(
        "--path-sample-spacing",
        type=float,
        default=0.05,
        help="Requested spacing in meters for the DWA densified tracking points.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write preview images and summary data.",
    )
    parser.add_argument("--max-rollout-steps", type=int, default=1200, help="Maximum closed-loop DWA rollout steps.")
    return parser.parse_args()


def _radius_slug(radius_m: float) -> str:
    return f"{radius_m:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _default_output_dir(map_path: Path, start_xy: tuple[float, float], goal_xy: tuple[float, float], inflate_radius: float) -> Path:
    start_tag = f"{start_xy[0]:.2f}_{start_xy[1]:.2f}".replace("-", "m").replace(".", "p")
    goal_tag = f"{goal_xy[0]:.2f}_{goal_xy[1]:.2f}".replace("-", "m").replace(".", "p")
    radius_tag = _radius_slug(inflate_radius)
    return map_path.parent / "visualizations" / f"{start_tag}__{goal_tag}__inflate_{radius_tag}"


def _base_preview(grid_map: OccupancyGridMap) -> np.ndarray:
    rgb = np.empty((grid_map.height, grid_map.width, 3), dtype=np.uint8)
    rgb[...] = FREE_COLOR
    rgb[grid_map.occupancy] = OCCUPIED_COLOR
    return rgb


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


def _draw_world_points(rgb: np.ndarray, grid_map: OccupancyGridMap, points_world: np.ndarray, color: np.ndarray, radius: int):
    seen: set[tuple[int, int]] = set()
    for x, y in points_world:
        row_col = grid_map.world_to_grid(float(x), float(y))
        if row_col in seen:
            continue
        seen.add(row_col)
        if grid_map.in_bounds(*row_col):
            _draw_disc(rgb, row_col, radius=radius, color=color)


def main():
    args = parse_args()
    map_path = Path(args.map).expanduser().resolve()
    start_xy = (float(args.start[0]), float(args.start[1]))
    goal_xy = (float(args.goal[0]), float(args.goal[1]))

    raw_map = OccupancyGridMap.from_meta_file(map_path)
    inflated_map = raw_map.inflate(args.inflate_radius)
    local_map = raw_map.inflate(args.local_clearance_radius)

    planner = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0)
    plan = planner.plan(
        inflated_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        snap_to_free=args.snap_to_free,
        max_snap_distance_m=max(0.5, args.inflate_radius + inflated_map.resolution),
    )

    dwa_cfg = DWAConfig(
        control_dt=0.05,
        lookahead_distance=args.lookahead_distance,
        prediction_horizon=args.prediction_horizon,
        goal_tolerance=args.goal_tolerance,
        path_sample_spacing=args.path_sample_spacing,
    )
    controller = DWAController(path_world=plan.path_world, grid_map=local_map, config=dwa_cfg)
    rollout_pose = np.array([start_xy[0], start_xy[1], float(args.start_yaw)], dtype=np.float64)
    rollout_velocity = (0.0, 0.0)
    rollout_points = [rollout_pose[:2].copy()]
    rollout_reached_goal = False
    rollout_debug = None
    for _step in range(args.max_rollout_steps):
        command, rollout_debug = controller.compute_command(tuple(rollout_pose), rollout_velocity)
        rollout_velocity = (float(command[0]), float(command[2]))
        rollout_pose[0] += float(command[0]) * math.cos(float(rollout_pose[2])) * dwa_cfg.control_dt
        rollout_pose[1] += float(command[0]) * math.sin(float(rollout_pose[2])) * dwa_cfg.control_dt
        rollout_pose[2] = (float(rollout_pose[2]) + float(command[2]) * dwa_cfg.control_dt + math.pi) % (2.0 * math.pi) - math.pi
        rollout_points.append(rollout_pose[:2].copy())
        if rollout_debug.reached_goal:
            rollout_reached_goal = True
            break

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(map_path, start_xy, goal_xy, args.inflate_radius)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    astar_preview = _base_preview(inflated_map)
    _draw_path(astar_preview, inflated_map, plan.raw_path_grid, ASTAR_RAW_PATH_COLOR)
    _draw_path(astar_preview, inflated_map, plan.path_grid, ASTAR_PRUNED_PATH_COLOR)
    _draw_disc(astar_preview, plan.start_grid, radius=3, color=START_COLOR)
    _draw_disc(astar_preview, plan.goal_grid, radius=3, color=GOAL_COLOR)

    dwa_preview = astar_preview.copy()
    _draw_world_points(dwa_preview, inflated_map, controller.path_world, DWA_POINT_COLOR, radius=1)
    _draw_world_points(dwa_preview, inflated_map, controller.reference_path_world, ASTAR_PRUNED_PATH_COLOR, radius=2)
    _draw_world_points(dwa_preview, inflated_map, np.asarray(rollout_points), DWA_ROLLOUT_COLOR, radius=1)

    astar_preview_path = output_dir / "astar_preview.ppm"
    dwa_preview_path = output_dir / "dwa_points_preview.ppm"
    write_ppm(astar_preview_path, astar_preview)
    write_ppm(dwa_preview_path, dwa_preview)

    summary = {
        "map": str(map_path),
        "inflate_radius_m": args.inflate_radius,
        "local_clearance_radius_m": args.local_clearance_radius,
        "start_yaw": args.start_yaw,
        "goal_tolerance_m": args.goal_tolerance,
        "lookahead_distance_m": args.lookahead_distance,
        "prediction_horizon_s": args.prediction_horizon,
        "path_sample_spacing_m": args.path_sample_spacing,
        "start_world": list(start_xy),
        "goal_world": list(goal_xy),
        "start_grid": list(plan.start_grid),
        "goal_grid": list(plan.goal_grid),
        "snap_to_free": args.snap_to_free,
        "planned_start_world": list(inflated_map.grid_to_world(*plan.start_grid)),
        "planned_goal_world": list(inflated_map.grid_to_world(*plan.goal_grid)),
        "astar": {
            "cost": plan.cost,
            "expanded_nodes": plan.expanded_nodes,
            "raw_path_points": len(plan.raw_path_world),
            "pruned_waypoints": len(plan.path_world),
        },
        "dwa": {
            "reference_waypoints": len(controller.reference_path_world),
            "tracking_points": int(len(controller.path_world)),
            "rollout_points": len(rollout_points),
            "rollout_reached_goal": rollout_reached_goal,
            "rollout_final_distance_to_goal_m": (
                float(rollout_debug.distance_to_goal) if rollout_debug is not None else None
            ),
        },
        "outputs": {
            "astar_preview": astar_preview_path.name,
            "dwa_points_preview": dwa_preview_path.name,
        },
        "legend": {
            "dark": "inflated obstacles",
            "red": "raw A* grid path",
            "orange": "A* pruned waypoint path",
            "cyan": "densified DWA tracking points",
            "purple": "closed-loop DWA rollout",
            "green": "start",
            "blue": "goal",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[INFO] map={map_path}")
    print(
        f"[INFO] A* raw points={len(plan.raw_path_world)} pruned_waypoints={len(plan.path_world)} "
        f"cost={plan.cost:.3f} expanded_nodes={plan.expanded_nodes}"
    )
    print(
        f"[INFO] DWA reference_waypoints={len(controller.reference_path_world)} "
        f"tracking_points={len(controller.path_world)}"
    )
    print(f"[INFO] wrote: {astar_preview_path}")
    print(f"[INFO] wrote: {dwa_preview_path}")
    print(f"[INFO] wrote: {summary_path}")


if __name__ == "__main__":
    main()
