#!/usr/bin/env python3
"""Generate and run a batch of randomized nav-to-pick episodes."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data.random_task import (
    DEFAULT_APPROACH_ANGLES_DEG,
    DEFAULT_OBJECT_OFFSET_BASE_GOAL_XY_M,
    DEFAULT_STANDOFF_CANDIDATES_M,
    RandomTaskGenerationError,
    SpawnRegion,
    write_random_pick_task,
)


DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
DEFAULT_CHECKPOINT = "checkpoints/go2_x5/flat/model_8500.pt"
DEFAULT_ISAACLAB_LAUNCHER = "/home/light/workspace/IsaacLab/isaaclab.sh"
DEFAULT_APPLE_OBJECT_Z_OFFSET_M = 0.0
DEFAULT_APPLE_EDGE_MIN_CLEARANCE_M = 0.03
DEFAULT_APPLE_SUPPORT_CLEARANCE_M = 0.0
DEFAULT_APPLE_FIXED_Z_M = 0.81653
DEFAULT_APPLE_FIXED_RPY_DEG = (-2.524, -7.822, -0.181)


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _command_path(raw_path: str | Path) -> str:
    raw_text = str(raw_path)
    if "/" not in raw_text:
        return raw_text
    return str(_project_path(raw_path))


def _project_path_or_uri(raw_path: str | Path) -> str:
    raw_text = str(raw_path)
    if "://" in raw_text:
        return raw_text
    return str(_project_path(raw_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-task", default="tasks/nav_pick_apple_fast.json")
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-task-dir", default="outputs/random_tasks")
    parser.add_argument("--dataset-root", default="/tmp/random_pick_dataset")
    parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--isaaclab-launcher", default=DEFAULT_ISAACLAB_LAUNCHER)
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--pipeline-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument(
        "--nav-headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward explicit navigation headless mode to run_nav_then_pick.py.",
    )
    parser.add_argument("--nav-only", action="store_true")
    parser.add_argument(
        "--precompute-nav-first",
        action="store_true",
        help="Run all episodes as headless nav-only first, then replay successful nav trajectories and grasp in GUI.",
    )
    parser.add_argument(
        "--single-window-replay",
        action="store_true",
        help=(
            "With --precompute-nav-first, launch one standalone Isaac Sim process "
            "that replays all successful nav trajectories and grasp attempts in sequence."
        ),
    )
    parser.add_argument("--handoff-smoke-only", action="store_true")
    parser.add_argument("--replay-nav-before-grasp", action="store_true")
    parser.add_argument("--replay-nav-real-time", action="store_true")
    parser.add_argument("--replay-nav-speed", type=float, default=1.0)
    parser.add_argument("--demo-visuals", action="store_true")
    parser.add_argument("--follow-camera-mode", choices=("chase", "front", "overhead", "fixed", "stage"), default="stage")
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument("--keep-window-open", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-grasp-trajectory", action="store_true")
    parser.add_argument("--use-planner-server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--auto-start-planner-server", action="store_true")
    parser.add_argument("--restart-planner-server", action="store_true")
    parser.add_argument("--planner-server-log", default="/tmp/go2_x5_curobo_planner_server.log")
    parser.add_argument("--planner-server-start-timeout-s", type=float, default=180.0)
    parser.add_argument("--allow-retreat-success", action="store_true")
    parser.add_argument("--legacy-side-retreat", action="store_true")
    parser.add_argument(
        "--side-retreat-only",
        action="store_true",
        help="For side grasps, skip vertical lift and count the planned reverse retreat as pick success.",
    )
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument(
        "--fail-on-object-reset-drift",
        action="store_true",
        help="Fail grasp handoff if the object drifts after target-pose apply and velocity reset.",
    )
    parser.add_argument("--skip-grasp-on-nav-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table-x-range", type=float, nargs=2, default=(0.83, 0.93), metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--table-y-range", type=float, nargs=2, default=(1.0, 1.5), metavar=("Y_MIN", "Y_MAX"))
    parser.add_argument("--table-z", type=float, default=0.81653, help="World z written directly to generated pick.object_pose_world.z.")
    parser.add_argument(
        "--object-z-offset",
        type=float,
        default=DEFAULT_APPLE_OBJECT_Z_OFFSET_M,
        help="Optional explicit offset added to --table-z. For the current apple asset, 0.0 keeps the stable center height at z=0.82.",
    )
    parser.add_argument("--object-fixed-z", type=float, default=DEFAULT_APPLE_FIXED_Z_M)
    parser.add_argument("--object-fixed-roll", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[0])
    parser.add_argument("--object-fixed-pitch", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[1])
    parser.add_argument("--object-fixed-yaw", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[2])
    parser.add_argument("--object-fixed-rpy-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--randomize-object-yaw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--object-prim-path", default=None)
    parser.add_argument("--table-prim-path", default="/World/table")
    parser.add_argument("--yaw-range", type=float, nargs=2, default=(0.0, 360.0), metavar=("DEG_MIN", "DEG_MAX"))
    parser.add_argument("--standoff-candidates", type=float, nargs="+", default=DEFAULT_STANDOFF_CANDIDATES_M)
    parser.add_argument("--approach-angles-deg", type=float, nargs="+", default=DEFAULT_APPROACH_ANGLES_DEG)
    parser.add_argument(
        "--base-goal-mode",
        choices=("radial", "object-offset"),
        default="object-offset",
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
    parser.add_argument(
        "--handoff-clearance-radius",
        type=float,
        default=0.20,
        help=(
            "Clearance radius used by the grasp handoff map check. The default matches "
            "the nav local clearance used by the apple demo."
        ),
    )
    parser.add_argument("--min-boundary-clearance", type=float, default=0.25)
    parser.add_argument(
        "--max-path-heading-error",
        type=float,
        default=1.0,
        help=(
            "Reject generated base goals whose A* final heading differs from base_yaw by more than this many radians. "
            "Use a negative value to disable the hard reject."
        ),
    )
    parser.add_argument("--path-heading-weight", type=float, default=1.5)
    parser.add_argument("--path-length-weight", type=float, default=0.03)
    parser.add_argument("--path-heading-lookback-points", type=int, default=5)
    parser.add_argument("--path-heading-min-segment-length", type=float, default=0.10)
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
    parser.add_argument(
        "--edge-min-clearance",
        type=float,
        default=DEFAULT_APPLE_EDGE_MIN_CLEARANCE_M,
        help="Minimum apple-center clearance from the selected table edge. The default avoids reset samples at ~2cm from the edge.",
    )
    parser.add_argument(
        "--object-support-clearance",
        type=float,
        default=DEFAULT_APPLE_SUPPORT_CLEARANCE_M,
        help="Optional minimum apple-center clearance from every sampled table-region boundary.",
    )
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.20)
    parser.add_argument("--terminal-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--final-yaw-tolerance-margin", type=float, default=0.20)
    parser.add_argument(
        "--ignore-goal-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not require final base yaw alignment; useful when arm yaw can cover the object.",
    )
    parser.add_argument("--yaw-align-start-distance", type=float, default=0.5)
    parser.add_argument("--yaw-align-min-wz", type=float, default=0.4)
    parser.add_argument("--yaw-align-max-wz", type=float, default=0.6)
    parser.add_argument("--yaw-settle-max-wz", type=float, default=0.25)
    parser.add_argument("--yaw-align-lateral-kp", type=float, default=0.9)
    parser.add_argument(
        "--yaw-align-min-vy",
        type=float,
        default=0.18,
        help="Minimum terminal lateral velocity used to avoid tiny-stride deadlocks near the goal.",
    )
    parser.add_argument("--terminal-yaw-slowdown-max-wz", type=float, default=0.42)
    parser.add_argument("--terminal-recovery-steps", type=int, default=90)
    parser.add_argument("--terminal-recovery-yaw-max-wz", type=float, default=0.32)
    parser.add_argument("--terminal-yaw-polish-vx", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-polish-min-wz", type=float, default=0.45)
    parser.add_argument("--terminal-yaw-polish-max-wz", type=float, default=0.55)
    parser.add_argument("--base-stable-linear-tolerance", type=float, default=0.06)
    parser.add_argument("--base-stable-angular-tolerance", type=float, default=0.20)
    parser.add_argument("--max-sample-attempts", type=int, default=200)
    parser.add_argument("--max-nav-steps", type=int, default=3000)
    return parser.parse_args()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_summary_path(nav_result: dict[str, Any] | None) -> Path | None:
    if not nav_result:
        return None
    episode_dir = nav_result.get("episode_dir")
    if not episode_dir:
        return None
    return Path(episode_dir).expanduser().resolve() / "summary.json"


def _nav_clearance_radius(args: argparse.Namespace) -> float:
    """Return the navigation clearance paired with generated pick base goals."""

    clearance = float(args.clearance_radius)
    if str(args.base_goal_mode).replace("-", "_") == "object_offset":
        return min(clearance, float(args.handoff_clearance_radius))
    return clearance


def _pipeline_command(
    args: argparse.Namespace,
    *,
    task_json: Path,
    dataset_dir: Path,
    nav_result: Path,
    handoff_report: Path,
    nav_only: bool | None = None,
    grasp_only: bool = False,
) -> list[str]:
    command = [
        _command_path(args.pipeline_python),
        str(PROJECT_ROOT / "scripts/pipeline/run_nav_then_pick.py"),
        "--task-json",
        str(task_json),
        "--task",
        args.task,
        "--checkpoint",
        _project_path_or_uri(args.checkpoint),
        "--isaaclab-launcher",
        _command_path(args.isaaclab_launcher),
        "--isaac-python",
        _command_path(args.isaac_python),
        "--dataset-dir",
        str(dataset_dir),
        "--nav-result",
        str(nav_result),
        "--handoff-report",
        str(handoff_report),
        "--brisk-nav",
        "--fast-dwa",
        "--handoff-clearance-radius",
        str(args.handoff_clearance_radius),
        "--inflate-radius",
        str(_nav_clearance_radius(args)),
        "--local-clearance-radius",
        str(_nav_clearance_radius(args)),
        "--max-nav-steps",
        str(args.max_nav_steps),
        "--goal-tolerance",
        "0.15",
        "--goal-yaw-tolerance",
        str(args.goal_yaw_tolerance),
        "--terminal-position-tolerance",
        "0.08",
        "--terminal-yaw-tolerance",
        str(args.terminal_yaw_tolerance),
        "--final-goal-tolerance-margin",
        "0.03",
        "--final-yaw-tolerance-margin",
        str(args.final_yaw_tolerance_margin),
        "--yaw-align-start-distance",
        str(args.yaw_align_start_distance),
        "--yaw-align-vx",
        "0.35",
        "--yaw-align-max-vx",
        "0.6",
        "--yaw-align-position-kp",
        "0.8",
        "--yaw-align-max-vy",
        "0.35",
        "--yaw-align-min-vy",
        str(args.yaw_align_min_vy),
        "--yaw-align-lateral-kp",
        str(args.yaw_align_lateral_kp),
        "--yaw-align-lateral-deadband",
        "0.015",
        "--yaw-align-min-wz",
        str(args.yaw_align_min_wz),
        "--yaw-align-max-wz",
        str(args.yaw_align_max_wz),
        "--terminal-yaw-slowdown-max-wz",
        str(args.terminal_yaw_slowdown_max_wz),
        "--terminal-recovery-steps",
        str(args.terminal_recovery_steps),
        "--terminal-recovery-yaw-max-wz",
        str(args.terminal_recovery_yaw_max_wz),
        "--terminal-yaw-polish-vx",
        str(args.terminal_yaw_polish_vx),
        "--terminal-yaw-polish-min-wz",
        str(args.terminal_yaw_polish_min_wz),
        "--terminal-yaw-polish-max-wz",
        str(args.terminal_yaw_polish_max_wz),
        "--base-stable-linear-tolerance",
        str(args.base_stable_linear_tolerance),
        "--base-stable-angular-tolerance",
        str(args.base_stable_angular_tolerance),
        "--settle-steps",
        "120",
        "--yaw-settle-stable-steps",
        "15",
        "--yaw-settle-max-wz",
        str(args.yaw_settle_max_wz),
        "--save-replay-trajectory",
    ]
    if args.nav_map:
        command.extend(["--nav-map", args.nav_map])
    if args.nav_headless is not None:
        command.append("--nav-headless" if args.nav_headless else "--no-nav-headless")
    nav_only_active = bool(args.nav_only if nav_only is None else nav_only)
    if grasp_only:
        command.append("--grasp-only")
    elif nav_only_active:
        command.append("--nav-only")
    if args.handoff_smoke_only:
        command.append("--handoff-smoke-only")
    if args.use_planner_server:
        command.append("--use-planner-server")
    if args.auto_start_planner_server:
        command.append("--auto-start-planner-server")
    if args.restart_planner_server:
        command.append("--restart-planner-server")
    command.extend(["--planner-server-log", args.planner_server_log])
    command.extend(["--planner-server-start-timeout-s", str(args.planner_server_start_timeout_s)])
    if args.replay_nav_before_grasp:
        command.append("--replay-nav-before-grasp")
        if args.replay_nav_real_time:
            command.append("--replay-nav-real-time")
        command.extend(["--replay-nav-speed", str(args.replay_nav_speed)])
    if args.demo_visuals:
        command.append("--demo-visuals")
    command.extend(["--follow-camera-mode", args.follow_camera_mode])
    command.extend(["--viewport-camera-prim", args.viewport_camera_prim])
    if args.keep_window_open is not None:
        command.append("--keep-window-open" if args.keep_window_open else "--no-keep-window-open")
    if args.show_grasp_trajectory:
        command.append("--show-grasp-trajectory")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_retreat_only:
        command.append("--side-retreat-only")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if getattr(args, "fail_on_object_reset_drift", False):
        command.append("--fail-on-object-reset-drift")
    return command


def _standalone_batch_command(args: argparse.Namespace, *, manifest_path: Path) -> list[str]:
    command = [
        _command_path(args.isaac_python),
        str(PROJECT_ROOT / "scripts/isaac/run_pick_from_nav_result_standalone.py"),
        "--batch-manifest",
        str(manifest_path),
        "--handoff-clearance-radius",
        str(args.handoff_clearance_radius),
        "--settle-steps",
        "120",
        "--timeout-s",
        "900.0",
        "--set-viewport-camera",
        "--viewport-camera-prim",
        args.viewport_camera_prim,
        "--replay-speed",
        str(args.replay_nav_speed),
    ]
    if args.replay_nav_real_time:
        command.append("--replay-real-time")
    if args.use_planner_server:
        command.append("--use-planner-server")
    if args.handoff_smoke_only:
        command.append("--handoff-smoke-only")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_retreat_only:
        command.append("--side-retreat-only")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if getattr(args, "fail_on_object_reset_drift", False):
        command.append("--fail-on-object-reset-drift")
    if args.show_grasp_trajectory:
        command.append("--show-grasp-trajectory")
    if args.keep_window_open:
        command.append("--keep-window-open")
    return command


def _write_single_window_manifest(args: argparse.Namespace, rows: list[dict[str, Any]], manifest_path: Path) -> None:
    episodes: list[dict[str, Any]] = []
    for row in rows:
        replay_trajectory = row.get("replay_trajectory_path")
        nav_result = _read_json_if_exists(Path(str(row["nav_result"])))
        if not replay_trajectory:
            replay_trajectory = (nav_result or {}).get("replay_trajectory_path")
        if not replay_trajectory:
            raise RuntimeError(
                f"episode {row['episode_index']} has no replay trajectory. "
                "Re-run with --precompute-nav-first so nav results include replay_trajectory_path."
            )
        episodes.append(
            {
                "episode_index": row["episode_index"],
                "seed": row["seed"],
                "task_json": row["task_json"],
                "dataset_dir": row["dataset_dir"],
                "nav_result": row["nav_result"],
                "handoff_report": row["handoff_report"],
                "replay_trajectory": replay_trajectory,
            }
        )
    manifest = {
        "schema_version": 1,
        "mode": "single_window_replay",
        "created_at": time.time(),
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _start_planner_server_for_single_window(args: argparse.Namespace) -> subprocess.Popen | None:
    if not args.auto_start_planner_server:
        return None
    from scripts.pipeline import run_nav_then_pick

    if args.side_retreat_only:
        args.legacy_side_retreat = True
        args.allow_retreat_success = True
    args.use_planner_server = True
    process = run_nav_then_pick._start_planner_server_if_requested(args)
    run_nav_then_pick._wait_for_planner_server_if_started(args, process)
    return process


def _write_summary_line(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def _success_and_failure(
    *,
    args: argparse.Namespace,
    returncode: int,
    nav_result: dict[str, Any] | None,
    handoff_report: dict[str, Any] | None,
    episode_summary: dict[str, Any] | None,
) -> tuple[bool, str]:
    if args.nav_only:
        success = bool(nav_result and nav_result.get("success", False)) and returncode == 0
        failure_reason = "" if success else str((nav_result or {}).get("failure_reason") or f"returncode_{returncode}")
        return success, failure_reason
    if handoff_report is not None:
        success = bool(handoff_report.get("success", False)) and returncode == 0
        failure_reason = "" if success else str(handoff_report.get("failure_reason") or f"returncode_{returncode}")
        return success, failure_reason
    if episode_summary is not None:
        success = bool(episode_summary.get("success", False)) and returncode == 0
        failure_reason = "" if success else str(episode_summary.get("failure_reason") or f"returncode_{returncode}")
        return success, failure_reason
    if nav_result is not None and not nav_result.get("success", False):
        return False, str(nav_result.get("failure_reason") or f"returncode_{returncode}")
    return returncode == 0, "" if returncode == 0 else f"returncode_{returncode}"


def _format_seed(seed: int) -> str:
    return f"{seed:04d}" if seed >= 0 else f"neg{abs(seed):04d}"


def _validate_apple_spawn_stability(args: argparse.Namespace) -> None:
    """Reject spawn settings that are likely to make the apple fall at reset."""

    edge_min_clearance = float(args.edge_min_clearance)
    if edge_min_clearance < 0.0:
        raise ValueError("--edge-min-clearance must be non-negative.")
    if bool(args.edge_biased) and edge_min_clearance < DEFAULT_APPLE_EDGE_MIN_CLEARANCE_M:
        raise ValueError(
            "--edge-min-clearance is too small for apple reset stability. "
            f"got={edge_min_clearance:.3f}, required_min={DEFAULT_APPLE_EDGE_MIN_CLEARANCE_M:.3f}."
        )
    support_clearance = float(args.object_support_clearance)
    if support_clearance < 0.0:
        raise ValueError("--object-support-clearance must be non-negative.")
    if bool(args.edge_biased) and edge_min_clearance < support_clearance:
        raise ValueError(
            "--edge-min-clearance must be >= --object-support-clearance so edge-biased samples remain supported. "
            f"edge_min_clearance={edge_min_clearance:.3f}, "
            f"object_support_clearance={support_clearance:.3f}."
        )
    x_width = float(args.table_x_range[1]) - float(args.table_x_range[0])
    y_width = float(args.table_y_range[1]) - float(args.table_y_range[0])
    required_width = 2.0 * support_clearance
    if x_width < required_width or y_width < required_width:
        raise ValueError(
            "table spawn range is too narrow for --object-support-clearance. "
            f"x_width={x_width:.3f}, y_width={y_width:.3f}, required_each_axis={required_width:.3f}."
        )


def main() -> int:
    args = _parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    _validate_apple_spawn_stability(args)
    if args.ignore_goal_yaw:
        args.goal_yaw_tolerance = math.pi
        args.terminal_yaw_tolerance = math.pi
        args.final_yaw_tolerance_margin = 0.0

    output_task_dir = Path(args.output_task_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    summary_jsonl = dataset_root / "batch_summary.jsonl"
    if summary_jsonl.exists():
        summary_jsonl.unlink()

    spawn_region = SpawnRegion(
        x_min=float(args.table_x_range[0]),
        x_max=float(args.table_x_range[1]),
        y_min=float(args.table_y_range[0]),
        y_max=float(args.table_y_range[1]),
        table_z=float(args.table_z),
        object_z_offset=float(args.object_z_offset),
    )
    if args.precompute_nav_first:
        if args.nav_headless is None:
            args.nav_headless = True
        if not args.nav_only:
            args.replay_nav_before_grasp = True
            if args.keep_window_open is None:
                args.keep_window_open = bool(args.single_window_replay and args.demo_visuals)

    overall_success = True
    precomputed_rows: list[dict[str, Any]] = []
    for episode_index in range(args.num_episodes):
        episode_seed = int(args.seed) + episode_index
        seed_label = _format_seed(episode_seed)
        task_json = output_task_dir / f"apple_seed_{seed_label}.json"
        dataset_dir = dataset_root / f"episode_{episode_index:04d}"
        nav_result_path = dataset_dir / "nav_result.json"
        handoff_report_path = dataset_dir / "handoff_report.json"
        started_at = time.time()
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "seed": episode_seed,
            "task_json": str(task_json),
            "dataset_dir": str(dataset_dir),
            "nav_result": str(nav_result_path),
            "handoff_report": str(handoff_report_path),
            "success": False,
            "failure_reason": "",
        }

        try:
            task = write_random_pick_task(
                base_task_path=_project_path(args.base_task),
                output_path=task_json,
                seed=episode_seed,
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
                object_support_clearance=args.object_support_clearance,
                max_path_heading_error=args.max_path_heading_error if args.max_path_heading_error >= 0.0 else None,
                path_heading_weight=args.path_heading_weight,
                path_length_weight=args.path_length_weight,
                path_heading_lookback_points=args.path_heading_lookback_points,
                path_heading_min_segment_length=args.path_heading_min_segment_length,
                max_sample_attempts=args.max_sample_attempts,
            )
            selected_base_goal = task.get("randomization", {}).get("selected_base_goal_candidate", {})
            row["object_pose_world"] = task["pick"].get("object_pose_world")
            row["object_pose_policy"] = task.get("randomization", {}).get("object_pose_policy")
            row["base_goal"] = task["pick"].get("base_goal")
            row["object_edge_sampling"] = task.get("randomization", {}).get("object_edge_sampling")
            row["base_goal_generation"] = task.get("randomization", {}).get("base_goal_generation")
            row["selected_base_goal_candidate"] = selected_base_goal
            row["path_heading_error"] = selected_base_goal.get("path_heading_error")
            row["path_final_heading"] = selected_base_goal.get("path_final_heading")
            row["path_length_m"] = selected_base_goal.get("path_length_m")
            row["path_heading_filter"] = (
                task.get("randomization", {})
                .get("base_goal_generation", {})
                .get("path_heading_filter")
            )
        except (RandomTaskGenerationError, ValueError, FileNotFoundError) as exc:
            row["failure_reason"] = "task_generation_failed"
            row["failure_detail"] = str(exc)
            row["elapsed_wall_time_s"] = time.time() - started_at
            overall_success = False
            _write_summary_line(summary_jsonl, row)
            print(f"[batch] episode={episode_index} seed={episode_seed} task generation failed: {exc}")
            if not args.continue_on_failure:
                break
            continue

        if args.precompute_nav_first:
            command = _pipeline_command(
                args,
                task_json=task_json,
                dataset_dir=dataset_dir,
                nav_result=nav_result_path,
                handoff_report=handoff_report_path,
                nav_only=True,
            )
            print(f"[batch] episode={episode_index} seed={episode_seed} precomputing navigation")
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)

            nav_result = _read_json_if_exists(nav_result_path)
            nav_success = bool(nav_result and nav_result.get("success", False)) and completed.returncode == 0
            failure_reason = "" if nav_success else str((nav_result or {}).get("failure_reason") or f"returncode_{completed.returncode}")
            episode_summary_path = _episode_summary_path(nav_result)
            row.update(
                {
                    "mode": "precompute_nav_first",
                    "nav_returncode": completed.returncode,
                    "success": nav_success if args.nav_only else False,
                    "failure_reason": failure_reason,
                    "nav_result_payload": nav_result,
                    "episode_summary": str(episode_summary_path) if episode_summary_path is not None else None,
                    "replay_trajectory_path": (nav_result or {}).get("replay_trajectory_path"),
                    "elapsed_wall_time_s": time.time() - started_at,
                }
            )
            print(f"[batch] episode={episode_index} nav_success={nav_success} failure_reason={failure_reason}")
            if not nav_success:
                overall_success = False
                _write_summary_line(summary_jsonl, row)
                if not args.continue_on_failure:
                    break
                continue
            if args.nav_only:
                _write_summary_line(summary_jsonl, row)
            else:
                precomputed_rows.append(row)
            continue

        command = _pipeline_command(
            args,
            task_json=task_json,
            dataset_dir=dataset_dir,
            nav_result=nav_result_path,
            handoff_report=handoff_report_path,
        )
        print(f"[batch] episode={episode_index} seed={episode_seed} launching pipeline")
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)

        nav_result = _read_json_if_exists(nav_result_path)
        handoff_report = _read_json_if_exists(handoff_report_path)
        episode_summary_path = _episode_summary_path(nav_result)
        episode_summary = _read_json_if_exists(episode_summary_path) if episode_summary_path is not None else None
        success, failure_reason = _success_and_failure(
            args=args,
            returncode=completed.returncode,
            nav_result=nav_result,
            handoff_report=handoff_report,
            episode_summary=episode_summary,
        )
        row.update(
            {
                "returncode": completed.returncode,
                "success": success,
                "failure_reason": failure_reason,
                "nav_result_payload": nav_result,
                "episode_summary": str(episode_summary_path) if episode_summary_path is not None else None,
                "replay_trajectory_path": (nav_result or {}).get("replay_trajectory_path"),
                "elapsed_wall_time_s": time.time() - started_at,
            }
        )
        _write_summary_line(summary_jsonl, row)
        print(f"[batch] episode={episode_index} success={success} failure_reason={failure_reason}")
        if not success:
            overall_success = False
            if not args.continue_on_failure:
                break

    if args.precompute_nav_first and not args.nav_only:
        if args.single_window_replay:
            if not precomputed_rows:
                overall_success = False
                print("[batch] no successful navigation episodes to replay.")
            else:
                manifest_path = dataset_root / "single_window_replay_manifest.json"
                _write_single_window_manifest(args, precomputed_rows, manifest_path)
                command = _standalone_batch_command(args, manifest_path=manifest_path)
                print(
                    f"[batch] single-window replaying {len(precomputed_rows)} episodes "
                    f"with manifest={manifest_path}"
                )
                _start_planner_server_for_single_window(args)
                started_at = time.time()
                completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)

                for row in precomputed_rows:
                    nav_result_path = Path(str(row["nav_result"]))
                    handoff_report_path = Path(str(row["handoff_report"]))
                    nav_result = _read_json_if_exists(nav_result_path)
                    handoff_report = _read_json_if_exists(handoff_report_path)
                    episode_summary_path = _episode_summary_path(nav_result)
                    episode_summary = _read_json_if_exists(episode_summary_path) if episode_summary_path is not None else None
                    row_returncode = 0 if handoff_report and handoff_report.get("success", False) else completed.returncode
                    success, failure_reason = _success_and_failure(
                        args=args,
                        returncode=row_returncode,
                        nav_result=nav_result,
                        handoff_report=handoff_report,
                        episode_summary=episode_summary,
                    )
                    row.update(
                        {
                            "mode": "single_window_replay",
                            "batch_manifest": str(manifest_path),
                            "grasp_returncode": completed.returncode,
                            "success": success,
                            "failure_reason": failure_reason,
                            "handoff_report_payload": handoff_report,
                            "episode_summary": str(episode_summary_path) if episode_summary_path is not None else None,
                            "elapsed_replay_grasp_wall_time_s": time.time() - started_at,
                        }
                    )
                    _write_summary_line(summary_jsonl, row)
                    print(f"[batch] episode={row['episode_index']} success={success} failure_reason={failure_reason}")
                    if not success:
                        overall_success = False
                if completed.returncode != 0:
                    overall_success = False
        else:
            for row in precomputed_rows:
                task_json = Path(str(row["task_json"]))
                dataset_dir = Path(str(row["dataset_dir"]))
                nav_result_path = Path(str(row["nav_result"]))
                handoff_report_path = Path(str(row["handoff_report"]))
                started_at = time.time()
                command = _pipeline_command(
                    args,
                    task_json=task_json,
                    dataset_dir=dataset_dir,
                    nav_result=nav_result_path,
                    handoff_report=handoff_report_path,
                    nav_only=False,
                    grasp_only=True,
                )
                print(f"[batch] episode={row['episode_index']} seed={row['seed']} replaying navigation and grasping")
                completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)

                nav_result = _read_json_if_exists(nav_result_path)
                handoff_report = _read_json_if_exists(handoff_report_path)
                episode_summary_path = _episode_summary_path(nav_result)
                episode_summary = _read_json_if_exists(episode_summary_path) if episode_summary_path is not None else None
                success, failure_reason = _success_and_failure(
                    args=args,
                    returncode=completed.returncode,
                    nav_result=nav_result,
                    handoff_report=handoff_report,
                    episode_summary=episode_summary,
                )
                row.update(
                    {
                        "grasp_returncode": completed.returncode,
                        "success": success,
                        "failure_reason": failure_reason,
                        "handoff_report_payload": handoff_report,
                        "episode_summary": str(episode_summary_path) if episode_summary_path is not None else None,
                        "elapsed_replay_grasp_wall_time_s": time.time() - started_at,
                    }
                )
                _write_summary_line(summary_jsonl, row)
                print(f"[batch] episode={row['episode_index']} success={success} failure_reason={failure_reason}")
                if not success:
                    overall_success = False
                    if not args.continue_on_failure:
                        break

    print(f"[batch] summary: {summary_jsonl}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
