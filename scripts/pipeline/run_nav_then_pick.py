#!/usr/bin/env python3
"""Coordinate the two-process Go2-X5 navigation-to-pick demo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data import load_task
from source.navigation import NavPlanner
from source.navigation.navlib import DWAConfig


PIPELINE_CONTEXT_JSON = Path("/tmp/go2_x5_pipeline_context.json")
DEFAULT_NAV_RESULT_JSON = Path("/tmp/go2_x5_nav_result.json")


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--map", dest="scene_usd", default=None)
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--isaaclab-python", default=sys.executable)
    parser.add_argument("--isaaclab-launcher", default=None, help="Optional Isaac Lab isaaclab.sh launcher; adds '-p'.")
    parser.add_argument("--nav-result", default=str(DEFAULT_NAV_RESULT_JSON))
    parser.add_argument("--nav-only", action="store_true")
    parser.add_argument("--grasp-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--use-planner-server", action="store_true")
    parser.add_argument("--head-camera", action="store_true")
    parser.add_argument("--debug-command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--max-nav-steps", type=int, default=3000)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--goal-tolerance", type=float, default=0.35)
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
    parser.add_argument("--inflate-radius", type=float, default=0.30)
    parser.add_argument("--local-clearance-radius", type=float, default=0.25)
    return parser.parse_args()


def _pick_script_editor_command() -> str:
    script = PROJECT_ROOT / "scripts/isaac/run_pick_from_nav_result.py"
    return f'exec(open("{script}", "r", encoding="utf-8").read())'


def _write_context(args: argparse.Namespace, task_json: Path) -> None:
    PIPELINE_CONTEXT_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_json": str(task_json),
                "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                "use_planner_server": args.use_planner_server,
                "settle_steps": args.settle_steps,
                "dataset_dir": args.dataset_dir,
                "no_record": args.no_record,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _nav_command(args: argparse.Namespace, task) -> list[str]:
    script = PROJECT_ROOT / "scripts/navigation/run_nav_only.py"
    prefix = [args.isaaclab_launcher, "-p"] if args.isaaclab_launcher else [args.isaaclab_python]
    command = [
        *prefix,
        str(script),
        "--task-json",
        str(_project_path(args.task_json)),
        "--checkpoint",
        str(args.checkpoint),
        "--nav-result",
        str(Path(args.nav_result).expanduser().resolve()),
        "--max-nav-steps",
        str(args.max_nav_steps),
        "--settle-steps",
        str(args.settle_steps),
        "--goal-tolerance",
        str(args.goal_tolerance),
        "--goal-yaw-tolerance",
        str(args.goal_yaw_tolerance),
        "--inflate-radius",
        str(args.inflate_radius),
        "--local-clearance-radius",
        str(args.local_clearance_radius),
    ]
    if args.scene_usd:
        command.extend(["--map", args.scene_usd])
    if args.nav_map:
        command.extend(["--nav-map", args.nav_map])
    if args.dataset_dir:
        command.extend(["--dataset-dir", args.dataset_dir])
    if args.no_record:
        command.append("--no-record")
    if args.head_camera:
        command.append("--head-camera")
    if args.debug_command:
        command.extend(["--debug-command", *(str(value) for value in args.debug_command)])
    return command


def _dry_run(args: argparse.Namespace, task) -> None:
    nav_map = _project_path(args.nav_map or task.nav_map)
    plan_summary: dict[str, object] = {
        "state_machine": [
            "INIT",
            "LOAD_NAV_TASK",
            "NAV_TO_PICK_BASE",
            "YAW_ALIGN",
            "SETTLE_BASE",
            "EXPORT_GRASP_STATE",
            "GENERATE_GRASP_TARGET",
            "PLAN_GRASP",
            "EXECUTE_GRASP",
            "CHECK_PICK_SUCCESS",
            "SAVE_EPISODE",
        ],
        "task": task,
        "nav_map": str(nav_map),
    }
    if nav_map.exists():
        planner = NavPlanner(str(nav_map), args.inflate_radius, DWAConfig(control_dt=0.05))
        path_world = planner.plan_global_path((task.start.x, task.start.y), (task.pick.base_goal.x, task.pick.base_goal.y))
        plan_summary["global_path_world"] = path_world
    else:
        plan_summary["warning"] = f"nav map does not exist yet: {nav_map}"
    print(json.dumps(plan_summary, indent=2, ensure_ascii=False, default=lambda value: value.__dict__))


def main() -> int:
    args = _parse_args()
    task_json = _project_path(args.task_json)
    task = load_task(task_json)
    _write_context(args, task_json)
    if args.dry_run:
        _dry_run(args, task)
        return 0
    if args.grasp_only:
        print("[handoff] Run this command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command())
        return 0
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --dry-run or --grasp-only is used")
    command = _nav_command(args, task)
    print("[pipeline] launching navigation:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    nav_result = json.loads(Path(args.nav_result).expanduser().resolve().read_text(encoding="utf-8"))
    if not nav_result.get("success", False):
        print("[pipeline] navigation failed:", nav_result.get("failure_reason"))
        return 1
    if args.nav_only:
        print("[pipeline] navigation complete:", args.nav_result)
        return 0
    print("[pipeline] navigation complete. Run this command in Isaac Sim Script Editor:")
    print(_pick_script_editor_command())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
