#!/usr/bin/env python3
"""Generate one randomized apple pick task JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data.random_task import (
    DEFAULT_APPROACH_ANGLES_DEG,
    DEFAULT_OBJECT_OFFSET_BASE_GOAL_XY_M,
    DEFAULT_STANDOFF_CANDIDATES_M,
    SpawnRegion,
    write_random_pick_task,
)


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-task", default="tasks/nav_pick_apple_fast.json")
    parser.add_argument("--output", required=True, help="Output task JSON path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object-prim-path", default=None)
    parser.add_argument("--table-prim-path", default="/World/table")
    parser.add_argument("--table-x-range", type=float, nargs=2, default=(0.88, 0.93), metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--table-y-range", type=float, nargs=2, default=(0.9, 1.6), metavar=("Y_MIN", "Y_MAX"))
    parser.add_argument("--table-z", type=float, default=0.82, help="World z written directly to pick.object_pose_world.z.")
    parser.add_argument("--object-z-offset", type=float, default=0.0, help="Optional explicit offset added to --table-z.")
    parser.add_argument("--object-fixed-z", type=float, default=0.81653)
    parser.add_argument("--object-fixed-roll", type=float, default=-2.524)
    parser.add_argument("--object-fixed-pitch", type=float, default=-7.822)
    parser.add_argument("--object-fixed-yaw", type=float, default=-0.181)
    parser.add_argument("--object-fixed-rpy-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--randomize-object-yaw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--yaw-range", type=float, nargs=2, default=(0.0, 360.0), metavar=("DEG_MIN", "DEG_MAX"))
    parser.add_argument("--standoff-candidates", type=float, nargs="+", default=DEFAULT_STANDOFF_CANDIDATES_M)
    parser.add_argument("--approach-angles-deg", type=float, nargs="+", default=DEFAULT_APPROACH_ANGLES_DEG)
    parser.add_argument(
        "--base-goal-mode",
        choices=("radial", "object-offset"),
        default="radial",
        help="Use radial candidates or a fixed object-frame XY offset for pick base generation.",
    )
    parser.add_argument(
        "--base-goal-offset-xy",
        type=float,
        nargs=2,
        default=DEFAULT_OBJECT_OFFSET_BASE_GOAL_XY_M,
        metavar=("DX", "DY"),
        help="World XY offset added to the sampled object position when --base-goal-mode object-offset is used.",
    )
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--clearance-radius", type=float, default=0.20)
    parser.add_argument("--min-boundary-clearance", type=float, default=0.25)
    parser.add_argument(
        "--edge-biased",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bias object samples toward reachable table edges.",
    )
    parser.add_argument(
        "--edge-sides",
        nargs="+",
        default=None,
        help="Optional edge sides such as x_max y_max x_max_y_max. Defaults derive from approach angles.",
    )
    parser.add_argument("--edge-margin", type=float, default=0.12, help="Width of the near-edge sampling band in meters.")
    parser.add_argument("--edge-min-clearance", type=float, default=0.02, help="Minimum object-center clearance from the selected edge.")
    parser.add_argument(
        "--max-path-heading-error",
        type=float,
        default=1.0,
        help=(
            "Reject base_goal candidates whose A* final heading differs from base_yaw by more than this many radians. "
            "Use a negative value to disable the hard reject."
        ),
    )
    parser.add_argument("--path-heading-weight", type=float, default=1.5)
    parser.add_argument("--path-length-weight", type=float, default=0.03)
    parser.add_argument("--path-heading-lookback-points", type=int, default=5)
    parser.add_argument("--path-heading-min-segment-length", type=float, default=0.10)
    parser.add_argument("--max-sample-attempts", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spawn_region = SpawnRegion(
        x_min=float(args.table_x_range[0]),
        x_max=float(args.table_x_range[1]),
        y_min=float(args.table_y_range[0]),
        y_max=float(args.table_y_range[1]),
        table_z=float(args.table_z),
        object_z_offset=float(args.object_z_offset),
    )
    task = write_random_pick_task(
        base_task_path=_project_path(args.base_task),
        output_path=args.output,
        seed=args.seed,
        nav_map_path=args.nav_map,
        object_prim_path=args.object_prim_path,
        table_prim_path=args.table_prim_path,
        spawn_region=spawn_region,
        yaw_range_deg=(float(args.yaw_range[0]), float(args.yaw_range[1])),
        object_fixed_z=args.object_fixed_z,
        object_fixed_rpy=(args.object_fixed_roll, args.object_fixed_pitch, args.object_fixed_yaw),
        object_fixed_rpy_unit=args.object_fixed_rpy_unit,
        randomize_object_yaw=args.randomize_object_yaw,
        standoff_candidates=args.standoff_candidates,
        approach_angles_deg=args.approach_angles_deg,
        base_goal_mode=args.base_goal_mode.replace("-", "_"),
        base_goal_offset_xy=(float(args.base_goal_offset_xy[0]), float(args.base_goal_offset_xy[1])),
        clearance_radius=args.clearance_radius,
        min_boundary_clearance=args.min_boundary_clearance,
        edge_sides=args.edge_sides,
        edge_margin=args.edge_margin if args.edge_biased else None,
        edge_min_clearance=args.edge_min_clearance,
        max_path_heading_error=args.max_path_heading_error if args.max_path_heading_error >= 0.0 else None,
        path_heading_weight=args.path_heading_weight,
        path_length_weight=args.path_length_weight,
        path_heading_lookback_points=args.path_heading_lookback_points,
        path_heading_min_segment_length=args.path_heading_min_segment_length,
        max_sample_attempts=args.max_sample_attempts,
    )
    selected_base_goal = task.get("randomization", {}).get("selected_base_goal_candidate", {})
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "seed": int(args.seed),
                "object_pose_world": task["pick"].get("object_pose_world"),
                "base_goal": task["pick"].get("base_goal"),
                "base_goal_mode": task.get("randomization", {}).get("base_goal_generation", {}).get("mode"),
                "base_goal_offset_xy": task.get("randomization", {}).get("base_goal_generation", {}).get("base_goal_offset_xy"),
                "path_final_heading": selected_base_goal.get("path_final_heading"),
                "path_heading_error": selected_base_goal.get("path_heading_error"),
                "path_length_m": selected_base_goal.get("path_length_m"),
                "object_edge_sampling": task.get("randomization", {}).get("object_edge_sampling"),
                "attempts_used": task.get("randomization", {}).get("attempts_used"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
