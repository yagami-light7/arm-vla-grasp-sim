#!/usr/bin/env python3
"""Generate and run a batch of randomized nav-to-pick episodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data.random_task import RandomTaskGenerationError, SpawnRegion, write_random_pick_task


DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
DEFAULT_CHECKPOINT = "checkpoints/go2_x5/flat/model_8500.pt"
DEFAULT_ISAACLAB_LAUNCHER = "/home/light/workspace/IsaacLab/isaaclab.sh"


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
    parser.add_argument("--output-task-dir", default="/tmp/random_tasks")
    parser.add_argument("--dataset-root", default="/tmp/random_pick_dataset")
    parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--isaaclab-launcher", default=DEFAULT_ISAACLAB_LAUNCHER)
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--pipeline-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--nav-only", action="store_true")
    parser.add_argument("--handoff-smoke-only", action="store_true")
    parser.add_argument("--replay-nav-before-grasp", action="store_true")
    parser.add_argument("--replay-nav-real-time", action="store_true")
    parser.add_argument("--replay-nav-speed", type=float, default=1.0)
    parser.add_argument("--demo-visuals", action="store_true")
    parser.add_argument("--allow-retreat-success", action="store_true")
    parser.add_argument("--legacy-side-retreat", action="store_true")
    parser.add_argument(
        "--side-retreat-only",
        action="store_true",
        help="For side grasps, skip vertical lift and count the planned reverse retreat as pick success.",
    )
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument("--skip-grasp-on-nav-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table-x-range", type=float, nargs=2, default=(0.9, 1.6), metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--table-y-range", type=float, nargs=2, default=(1.0, 1.8), metavar=("Y_MIN", "Y_MAX"))
    parser.add_argument("--table-z", type=float, default=0.78)
    parser.add_argument("--object-z-offset", type=float, default=0.04)
    parser.add_argument("--object-prim-path", default=None)
    parser.add_argument("--table-prim-path", default="/World/table")
    parser.add_argument("--yaw-range", type=float, nargs=2, default=(0.0, 360.0), metavar=("DEG_MIN", "DEG_MAX"))
    parser.add_argument("--standoff-candidates", type=float, nargs="+", default=(0.75, 0.90, 1.05))
    parser.add_argument("--approach-angles-deg", type=float, nargs="+", default=(180.0, 210.0, 240.0))
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--clearance-radius", type=float, default=0.25)
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
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
    parser.add_argument("--terminal-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--final-yaw-tolerance-margin", type=float, default=0.07)
    parser.add_argument("--yaw-align-start-distance", type=float, default=0.5)
    parser.add_argument("--yaw-align-min-wz", type=float, default=0.4)
    parser.add_argument("--yaw-align-max-wz", type=float, default=0.6)
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


def _pipeline_command(
    args: argparse.Namespace,
    *,
    task_json: Path,
    dataset_dir: Path,
    nav_result: Path,
    handoff_report: Path,
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
        "0.25",
        "--save-replay-trajectory",
    ]
    if args.nav_map:
        command.extend(["--nav-map", args.nav_map])
    if args.nav_only:
        command.append("--nav-only")
    if args.handoff_smoke_only:
        command.append("--handoff-smoke-only")
    if args.replay_nav_before_grasp:
        command.append("--replay-nav-before-grasp")
        if args.replay_nav_real_time:
            command.append("--replay-nav-real-time")
        command.extend(["--replay-nav-speed", str(args.replay_nav_speed)])
    if args.demo_visuals:
        command.append("--demo-visuals")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_retreat_only:
        command.append("--side-retreat-only")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    return command


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


def main() -> int:
    args = _parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

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

    overall_success = True
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
                standoff_candidates=args.standoff_candidates,
                approach_angles_deg=args.approach_angles_deg,
                clearance_radius=args.clearance_radius,
                min_boundary_clearance=args.min_boundary_clearance,
                edge_sides=args.edge_sides,
                edge_margin=args.edge_margin if args.edge_biased else None,
                edge_min_clearance=args.edge_min_clearance,
                max_sample_attempts=args.max_sample_attempts,
            )
            row["object_pose_world"] = task["pick"].get("object_pose_world")
            row["base_goal"] = task["pick"].get("base_goal")
            row["object_edge_sampling"] = task.get("randomization", {}).get("object_edge_sampling")
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

    print(f"[batch] summary: {summary_jsonl}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
