#!/usr/bin/env python3
"""Standalone Isaac Sim runner for the navigation-to-pick handoff.

This entrypoint launches Isaac Sim with ``AppLauncher``, opens the task USD,
then reuses ``scripts/isaac/run_pick_from_nav_result.py``. It is intended for
one-command pipeline runs and avoids Script Editor UI tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT_JSON = Path("/tmp/go2_x5_pipeline_context.json")
DEFAULT_NAV_RESULT_JSON = Path("/tmp/go2_x5_nav_result.json")


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--nav-result", default=str(DEFAULT_NAV_RESULT_JSON))
    parser.add_argument("--pipeline-context", default=str(DEFAULT_CONTEXT_JSON))
    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--handoff-clearance-radius", type=float, default=0.25)
    parser.add_argument("--use-planner-server", action="store_true")
    parser.add_argument("--handoff-smoke-only", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--require-lift-success", action="store_true", default=False)
    parser.add_argument("--allow-retreat-success", action="store_true")
    parser.add_argument("--legacy-side-retreat", action="store_true")
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument("--stage-load-updates", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_raw_task(task_json: Path) -> dict:
    return json.loads(task_json.read_text(encoding="utf-8"))


def _write_context(args: argparse.Namespace, task_json: Path, raw_task: dict) -> Path:
    nav_map = args.nav_map or raw_task["nav_map"]
    scene_usd = args.scene_usd or raw_task["scene_usd"]
    context_path = Path(args.pipeline_context).expanduser().resolve()
    context = {
        "schema_version": 1,
        "task_json": str(task_json),
        "scene_usd": str(_project_path(scene_usd)),
        "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
        "nav_map": str(_project_path(nav_map)),
        "terrain_prim_path": args.terrain_prim_path,
        "handoff_clearance_radius": args.handoff_clearance_radius,
        "use_planner_server": args.use_planner_server,
        "handoff_smoke_only": args.handoff_smoke_only,
        "require_object_lift_success": not args.allow_retreat_success,
        "legacy_side_retreat": args.legacy_side_retreat,
        "side_grasp_fallback_retreat": args.side_grasp_fallback_retreat,
        "settle_steps": args.settle_steps,
        "dataset_dir": args.dataset_dir,
        "no_record": args.no_record,
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    return context_path


def _open_stage(scene_usd: Path, load_updates: int) -> None:
    import omni.usd

    if not scene_usd.exists():
        raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
    usd_context = omni.usd.get_context()
    open_result = usd_context.open_stage(str(scene_usd))
    if open_result is False:
        raise RuntimeError(f"failed to open stage: {scene_usd}")
    for _ in range(max(1, load_updates)):
        simulation_app.update()
    if usd_context.get_stage() is None:
        raise RuntimeError(f"stage did not load: {scene_usd}")
    print(f"[standalone] opened stage: {scene_usd}")


def _run_handoff_task(timeout_s: float) -> None:
    from scripts.isaac import run_pick_from_nav_result

    loop = asyncio.get_event_loop()
    task = asyncio.ensure_future(run_pick_from_nav_result.guarded_main())
    started_at = time.time()
    while not task.done():
        simulation_app.update()
        if timeout_s > 0.0 and time.time() - started_at > timeout_s:
            task.cancel()
            raise TimeoutError(f"standalone pick timed out after {timeout_s:.1f}s")
    exception = task.exception()
    if exception is not None:
        raise exception


def main() -> int:
    task_json = _project_path(args_cli.task_json)
    raw_task = _load_raw_task(task_json)
    scene_usd = _project_path(args_cli.scene_usd or raw_task["scene_usd"])
    context_json = _write_context(args_cli, task_json, raw_task)

    os.environ["GO2_X5_WORKSPACE"] = str(PROJECT_ROOT)
    os.environ["GO2_X5_PIPELINE_CONTEXT"] = str(context_json)
    os.environ["GO2_X5_NAV_RESULT"] = str(Path(args_cli.nav_result).expanduser().resolve())
    os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "1" if args_cli.handoff_smoke_only else "0"
    os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "0" if args_cli.allow_retreat_success else "1"
    os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0" if args_cli.legacy_side_retreat else "1"
    os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "1" if args_cli.side_grasp_fallback_retreat else "0"

    _open_stage(scene_usd, args_cli.stage_load_updates)
    _run_handoff_task(args_cli.timeout_s)
    print("[standalone] pick handoff complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
