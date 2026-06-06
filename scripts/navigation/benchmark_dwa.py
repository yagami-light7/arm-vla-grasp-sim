#!/usr/bin/env python3
"""Benchmark pure-Python DWA compute time without Isaac Lab or PhysX."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.navigation.navlib import AStarPlanner, DWAConfig, DWAController, OccupancyGridMap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, dest="map_json")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--inflate-radius", type=float, default=0.25)
    parser.add_argument("--local-clearance-radius", type=float, default=0.20)
    parser.add_argument("--max-snap-distance", type=float, default=2.0)
    parser.add_argument("--prediction-horizon", type=float, default=0.90)
    parser.add_argument("--lookahead-distance", type=float, default=0.35)
    parser.add_argument("--dwa-linear-samples", type=int, default=None)
    parser.add_argument("--dwa-angular-samples", type=int, default=None)
    parser.add_argument("--dwa-integration-dt", type=float, default=None)
    parser.add_argument("--dwa-path-sample-spacing", type=float, default=None)
    parser.add_argument("--dwa-path-distance-window", type=int, default=None)
    parser.add_argument("--iters", type=int, default=300)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.dwa_linear_samples is not None and args.dwa_linear_samples < 2:
        raise ValueError("--dwa-linear-samples must be >= 2.")
    if args.dwa_angular_samples is not None and args.dwa_angular_samples < 3:
        raise ValueError("--dwa-angular-samples must be >= 3.")
    if args.dwa_integration_dt is not None and args.dwa_integration_dt <= 0.0:
        raise ValueError("--dwa-integration-dt must be > 0.")
    if args.dwa_path_sample_spacing is not None and args.dwa_path_sample_spacing <= 0.0:
        raise ValueError("--dwa-path-sample-spacing must be > 0.")
    if args.dwa_path_distance_window is not None and args.dwa_path_distance_window < 1:
        raise ValueError("--dwa-path-distance-window must be >= 1.")


def _dwa_config(args: argparse.Namespace) -> DWAConfig:
    kwargs = {}
    if args.dwa_linear_samples is not None:
        kwargs["linear_samples"] = args.dwa_linear_samples
    if args.dwa_angular_samples is not None:
        kwargs["angular_samples"] = args.dwa_angular_samples
    if args.dwa_integration_dt is not None:
        kwargs["integration_dt"] = args.dwa_integration_dt
    if args.dwa_path_sample_spacing is not None:
        kwargs["path_sample_spacing"] = args.dwa_path_sample_spacing
    if args.dwa_path_distance_window is not None:
        kwargs["path_distance_window"] = args.dwa_path_distance_window
    return DWAConfig(
        control_dt=0.05,
        lookahead_distance=args.lookahead_distance,
        prediction_horizon=args.prediction_horizon,
        goal_tolerance=0.20,
        max_linear_velocity=0.8,
        min_active_linear_velocity=0.55,
        near_goal_min_active_linear_velocity=0.38,
        close_goal_speed_limit=0.35,
        speed_bias=1.10,
        max_linear_accel=4.5,
        **kwargs,
    )


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    raw_map = OccupancyGridMap.from_meta_file(args.map_json)
    global_map = raw_map.inflate(args.inflate_radius)
    local_map = raw_map.inflate(args.local_clearance_radius)
    plan = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0).plan(
        global_map,
        start_xy=(args.start[0], args.start[1]),
        goal_xy=(args.goal[0], args.goal[1]),
        snap_to_free=True,
        max_snap_distance_m=args.max_snap_distance,
    )
    path_world = list(plan.path_world)
    if math.hypot(path_world[-1][0] - args.goal[0], path_world[-1][1] - args.goal[1]) > 0.01:
        path_world.append((args.goal[0], args.goal[1]))

    config = _dwa_config(args)
    controller = DWAController(path_world, local_map, config)
    pose = np.array([args.start[0], args.start[1], args.start[2]], dtype=np.float64)
    speed = (0.0, 0.0)
    compute_ms: list[float] = []
    sampled_candidates: list[int] = []
    feasible_candidates: list[int] = []
    collision_rejections: list[int] = []

    for _ in range(args.iters):
        started = time.perf_counter()
        command, debug = controller.compute_command(tuple(pose), speed)
        compute_ms.append((time.perf_counter() - started) * 1000.0)
        sampled_candidates.append(debug.sampled_candidates)
        feasible_candidates.append(debug.feasible_candidates)
        collision_rejections.append(debug.collision_rejections)

        speed = (float(command[0]), float(command[2]))
        pose[0] += speed[0] * math.cos(pose[2]) * config.control_dt
        pose[1] += speed[0] * math.sin(pose[2]) * config.control_dt
        pose[2] = (pose[2] + speed[1] * config.control_dt + math.pi) % (2.0 * math.pi) - math.pi
        if debug.reached_goal or math.hypot(pose[0] - args.goal[0], pose[1] - args.goal[1]) <= config.goal_tolerance:
            pose[:] = (args.start[0], args.start[1], args.start[2])
            speed = (0.0, 0.0)
            controller = DWAController(path_world, local_map, config)

    print(f"[bench] avg_compute_ms={statistics.fmean(compute_ms):.3f}")
    print(f"[bench] p95_compute_ms={float(np.percentile(compute_ms, 95)):.3f}")
    print(f"[bench] sampled_candidates={statistics.fmean(sampled_candidates):.1f}")
    print(f"[bench] feasible_candidates={statistics.fmean(feasible_candidates):.1f}")
    print(f"[bench] collision_rejections={statistics.fmean(collision_rejections):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
