#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source.navigation.navlib import AStarPlanner, OccupancyGridMap, render_plan_preview, save_path_bundle, write_ppm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an A* path over an occupancy grid map.")
    parser.add_argument("--map", required=True, help="Path to a nav-map metadata file (.json/.yaml).")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"), help="Start point in world coordinates.")
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"), help="Goal point in world coordinates.")
    parser.add_argument(
        "--inflate-radius",
        type=float,
        default=0.25,
        help="Obstacle inflation radius in meters to approximate the robot footprint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write planning artifacts. Defaults to <map_dir>/plans/<start>_<goal>/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grid_map = OccupancyGridMap.from_meta_file(args.map)
    inflated_map = grid_map.inflate(args.inflate_radius)
    planner = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0)
    plan = planner.plan(
        inflated_map,
        start_xy=(float(args.start[0]), float(args.start[1])),
        goal_xy=(float(args.goal[0]), float(args.goal[1])),
        snap_to_free=True,
        max_snap_distance_m=max(0.5, args.inflate_radius + grid_map.resolution),
    )

    if args.output_dir is None:
        map_stem = Path(args.map).stem
        start_tag = f"{args.start[0]:.2f}_{args.start[1]:.2f}".replace("-", "m").replace(".", "p")
        goal_tag = f"{args.goal[0]:.2f}_{args.goal[1]:.2f}".replace("-", "m").replace(".", "p")
        output_dir = Path(args.map).resolve().parent / "plans" / map_stem / f"{start_tag}__{goal_tag}"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path_bundle(
        output_dir / "path.json",
        grid_map=inflated_map,
        plan=plan,
        start_xy=(float(args.start[0]), float(args.start[1])),
        goal_xy=(float(args.goal[0]), float(args.goal[1])),
        inflation_radius_m=args.inflate_radius,
    )
    preview = render_plan_preview(
        inflated_map,
        path_grid=plan.path_grid,
        start_grid=plan.start_grid,
        goal_grid=plan.goal_grid,
    )
    write_ppm(output_dir / "preview.ppm", preview)

    print(f"[INFO] Planned {len(plan.path_world)} waypoints with cost {plan.cost:.3f}.")
    print(f"[INFO] Expanded nodes: {plan.expanded_nodes}")
    print(f"[INFO] Artifacts written to: {os.fspath(output_dir)}")


if __name__ == "__main__":
    main()
