#!/usr/bin/env python3
"""Standalone MVP runner for multi-process JSON place handoff."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLACE_RESULT_JSON = Path("/tmp/go2_x5_place_result.json")
DEFAULT_HANDOFF_REPORT_JSON = Path("/tmp/go2_x5_place_handoff_report.json")
HANDOFF_MODE = "multi_process_json"
PHYSICAL_CONTINUITY = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--nav-result", required=True)
    parser.add_argument("--place-result", default=str(DEFAULT_PLACE_RESULT_JSON))
    parser.add_argument("--handoff-report", default=str(DEFAULT_HANDOFF_REPORT_JSON))
    parser.add_argument("--pipeline-context", default="/tmp/go2_x5_place_pipeline_context.json")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    parser.add_argument("--handoff-clearance-radius", type=float, default=0.20)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--stage-load-updates", type=int, default=30)
    parser.add_argument("--place-xy-tolerance", type=float, default=None)
    parser.add_argument("--place-z-tolerance", type=float, default=None)
    parser.add_argument(
        "--mvp-reconstruct-place",
        action="store_true",
        help="Legacy/MVP mode: teleport the object to place_pose_world after nav_to_place instead of running an arm place.",
    )
    parser.add_argument("--demo-visuals", action="store_true")
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument("--keep-window-open", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _result_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return Path(args.place_result).expanduser().resolve(), Path(args.handoff_report).expanduser().resolve()


def _write_reports(args: argparse.Namespace, *, place_result: dict[str, Any], handoff_report: dict[str, Any]) -> None:
    place_path, handoff_path = _result_paths(args)
    now = time.time()
    place_payload = {
        "schema_version": 1,
        "handoff_mode": HANDOFF_MODE,
        "physical_continuity": PHYSICAL_CONTINUITY,
        "updated_at": now,
        **place_result,
    }
    handoff_payload = {
        "schema_version": 1,
        "handoff_mode": HANDOFF_MODE,
        "physical_continuity": PHYSICAL_CONTINUITY,
        "updated_at": now,
        **handoff_report,
    }
    _write_json(place_path, place_payload)
    _write_json(handoff_path, handoff_payload)


def _write_failure(args: argparse.Namespace, reason: str, detail: str, report: dict[str, Any] | None = None) -> None:
    diagnostic = dict(report or {})
    diagnostic.setdefault("failure_detail", detail)
    mode = "mvp_reconstructed_object" if getattr(args, "mvp_reconstruct_place", False) else "not_implemented"
    initialization = "reconstructed" if getattr(args, "mvp_reconstruct_place", False) else "none"
    _write_reports(
        args,
        place_result={
            "success": False,
            "failure_reason": reason,
            "failure_detail": detail,
            "place_execution_mode": mode,
            "place_object_initialization": initialization,
        },
        handoff_report={
            "success": False,
            "failure_reason": reason,
            "failure_detail": detail,
            "diagnostic": diagnostic,
        },
    )


def _write_context(args: argparse.Namespace, task_json: Path, raw_task: dict[str, Any]) -> Path:
    context_path = Path(args.pipeline_context).expanduser().resolve()
    context = {
        "schema_version": 1,
        "task_json": str(task_json),
        "scene_usd": str(_project_path(args.scene_usd or raw_task["scene_usd"])),
        "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
        "place_result_json": str(Path(args.place_result).expanduser().resolve()),
        "handoff_report_json": str(Path(args.handoff_report).expanduser().resolve()),
        "terrain_prim_path": args.terrain_prim_path,
        "handoff_clearance_radius": float(args.handoff_clearance_radius),
        "settle_steps": max(0, int(args.settle_steps)),
        "dataset_dir": args.dataset_dir,
        "handoff_mode": HANDOFF_MODE,
        "physical_continuity": PHYSICAL_CONTINUITY,
        "mvp_reconstruct_place": bool(args.mvp_reconstruct_place),
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
    for _ in range(max(1, int(load_updates))):
        simulation_app.update()
    if usd_context.get_stage() is None:
        raise RuntimeError(f"stage did not load: {scene_usd}")
    print(f"[place] opened stage: {scene_usd}")


def _candidate_stage_camera_paths(camera_prim_path: str) -> list[str]:
    candidates = [camera_prim_path]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path == "/World/Camera_main":
        candidates.append("/World/camera_main")
    return list(dict.fromkeys(candidates))


def _set_viewport_stage_camera(camera_prim_path: str) -> bool:
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        selected_path = None
        for candidate in _candidate_stage_camera_paths(camera_prim_path):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                selected_path = candidate
                break
        if selected_path is None:
            return False
        viewport = get_active_viewport()
        if viewport is None:
            return False
        sdf_path = Sdf.Path(selected_path)
        try:
            viewport.camera_path = sdf_path
        except Exception:
            if hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(sdf_path)
            else:
                raise
        print(f"[place] viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set place viewport camera {camera_prim_path}: {exc}")
        return False


def _schedule_kit_coroutine(coro):
    try:
        from omni.kit.async_engine import run_coroutine

        return run_coroutine(coro)
    except Exception:
        loop = asyncio.get_event_loop()
        return loop.create_task(coro)


def _drive_future_to_completion(future, *, timeout_s: float, label: str):
    started_at = time.time()
    while not future.done():
        simulation_app.update()
        if timeout_s > 0.0 and time.time() - started_at > timeout_s:
            if hasattr(future, "cancel"):
                future.cancel()
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    return future.result()


def _place_config(raw_task: dict[str, Any]) -> dict[str, Any]:
    place = dict(raw_task.get("place") or {})
    if not place.get("enabled", False):
        raise ValueError("place_disabled")
    if not place.get("place_pose_world"):
        raise ValueError("missing_place_pose_world")
    return place


def _apply_reconstructed_place_object(raw_task: dict[str, Any], place: dict[str, Any]) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    object_prim_path = (raw_task.get("pick") or {}).get("object_prim_path")
    if not object_prim_path:
        raise ValueError("pick.object_prim_path is required for reconstructed place object.")
    pose = dict(place["place_pose_world"])
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"object prim does not exist: {object_prim_path}")
    if not prim.IsA(UsdGeom.Xformable):
        raise RuntimeError(f"object prim is not xformable: {object_prim_path}")

    roll = float(pose.get("roll", 0.0))
    pitch = float(pose.get("pitch", 0.0))
    yaw = float(pose.get("yaw", 0.0))
    quat = pick_handoff._rpy_to_quat_wxyz(roll, pitch, yaw)
    xformable = UsdGeom.Xformable(prim)
    translate_op = pick_handoff._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
    orient_op = pick_handoff._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
    pick_handoff._set_translate_op(translate_op, (float(pose["x"]), float(pose["y"]), float(pose["z"])))
    pick_handoff._set_orient_op(orient_op, quat)
    zeroed_count = pick_handoff._zero_rigid_body_velocities(prim)
    report = {
        "applied": True,
        "place_object_initialization": "reconstructed",
        "object_prim_path": object_prim_path,
        "pose_world": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        },
        "quaternion_wxyz": [float(value) for value in quat],
        "rigid_body_velocity_zeroed_count": zeroed_count,
    }
    print("[place] reconstructed object at place pose:", report["pose_world"])
    return report


def _prepare_place_task_stage(task_json: Path) -> dict[str, Any]:
    from source.data import load_task
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    task = load_task(task_json)
    report = {
        "object_visibility": pick_handoff._show_only_task_object(task),
    }
    print("[place] prepared task stage:", report)
    return report


def _verify_place(reconstruction: dict[str, Any], place: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target = dict(place["place_pose_world"])
    actual = dict(reconstruction["pose_world"])
    xy_error = ((float(actual["x"]) - float(target["x"])) ** 2 + (float(actual["y"]) - float(target["y"])) ** 2) ** 0.5
    z_error = abs(float(actual["z"]) - float(target["z"]))
    xy_tolerance = float(args.place_xy_tolerance if args.place_xy_tolerance is not None else place.get("place_xy_tolerance", 0.10))
    z_tolerance = float(args.place_z_tolerance if args.place_z_tolerance is not None else place.get("place_z_tolerance", 0.08))
    return {
        "success": bool(xy_error <= xy_tolerance and z_error <= z_tolerance),
        "place_xy_error": float(xy_error),
        "place_z_error": float(z_error),
        "place_xy_tolerance": xy_tolerance,
        "place_z_tolerance": z_tolerance,
    }


def _restore_base_from_nav_result(nav_result: dict[str, Any], timeout_s: float) -> tuple[Any, dict[str, Any]]:
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    future = _schedule_kit_coroutine(pick_handoff._initialize_robot())
    world, robot = _drive_future_to_completion(future, timeout_s=timeout_s, label="place restore initialization")
    restore_future = _schedule_kit_coroutine(pick_handoff._restore_and_settle(world, robot, nav_result))
    restore_report = _drive_future_to_completion(restore_future, timeout_s=timeout_s, label="place base restore")
    return world, restore_report


def _settle_world(world, steps: int) -> None:
    for _ in range(max(0, int(steps))):
        world.step(render=True)
        simulation_app.update()


def _keep_window_open_until_closed() -> None:
    print("[place] keep-window-open enabled; close the Isaac Sim window to end this process.")
    while simulation_app.is_running():
        simulation_app.update()
        time.sleep(1.0 / 60.0)


def _run() -> int:
    args = args_cli
    task_json = _project_path(args.task_json)
    raw_task = _read_json(task_json)
    scene_usd = _project_path(args.scene_usd or raw_task["scene_usd"])
    nav_result = _read_json(args.nav_result)
    context_path = _write_context(args, task_json, raw_task)

    os.environ["GO2_X5_WORKSPACE"] = str(PROJECT_ROOT)
    os.environ["GO2_X5_PIPELINE_CONTEXT"] = str(context_path)
    os.environ["GO2_X5_NAV_RESULT"] = str(Path(args.nav_result).expanduser().resolve())
    os.environ["GO2_X5_HANDOFF_REPORT"] = str(Path(args.handoff_report).expanduser().resolve())
    os.environ["GO2_X5_PICK_SETTLE_STEPS"] = str(max(0, int(args.settle_steps)))

    _write_reports(
        args,
        place_result={
            "success": False,
            "failure_reason": "place_not_completed",
            "place_execution_mode": "mvp_reconstructed_object" if args.mvp_reconstruct_place else "not_implemented",
            "place_object_initialization": "reconstructed" if args.mvp_reconstruct_place else "none",
        },
        handoff_report={
            "success": False,
            "failure_reason": "place_not_completed",
            "task_json": str(task_json),
            "scene_usd": str(scene_usd),
            "nav_result": str(Path(args.nav_result).expanduser().resolve()),
        },
    )

    place = _place_config(raw_task)
    _open_stage(scene_usd, args.stage_load_updates)
    stage_prepare_report = _prepare_place_task_stage(task_json)
    if args.demo_visuals:
        _set_viewport_stage_camera(args.viewport_camera_prim)
    world, restore_report = _restore_base_from_nav_result(nav_result, args.timeout_s)
    if not args.mvp_reconstruct_place:
        detail = (
            "Full arm place/put execution is not implemented in this multi-process JSON runner. "
            "The previous MVP object teleport is disabled by default; pass --mvp-reconstruct-place "
            "only when you explicitly want that reconstructed-object smoke test."
        )
        place_result = {
            "success": False,
            "failure_reason": "place_arm_execution_not_implemented",
            "failure_detail": detail,
            "place_execution_mode": "not_implemented",
            "place_object_initialization": "none",
            "task_json": str(task_json),
            "nav_result": str(Path(args.nav_result).expanduser().resolve()),
            "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()) if args.dataset_dir else None,
            "restore": restore_report,
            "stage_prepare": stage_prepare_report,
            "notes": [
                "nav_to_place completed, but no arm place/put planner has run.",
                "The object was not teleported to place_pose_world in default mode.",
            ],
        }
        handoff_report = {
            "success": False,
            "failure_reason": "place_arm_execution_not_implemented",
            "failure_detail": detail,
            "task_json": str(task_json),
            "scene_usd": str(scene_usd),
            "nav_result": str(Path(args.nav_result).expanduser().resolve()),
            "stage": {
                "opened": True,
                "scene_usd": str(scene_usd),
                "terrain_prim_path": args.terrain_prim_path,
            },
            "restore": restore_report,
            "stage_prepare": stage_prepare_report,
            "place": {
                "execution_backend": "not_implemented",
                "object_reconstruction": {"applied": False},
            },
        }
        _write_reports(args, place_result=place_result, handoff_report=handoff_report)
        if args.keep_window_open:
            _keep_window_open_until_closed()
        print("[place] success=False failure_reason=place_arm_execution_not_implemented")
        return 1

    reconstruction_report = _apply_reconstructed_place_object(raw_task, place)
    _settle_world(world, args.settle_steps)
    verification = _verify_place(reconstruction_report, place, args)
    success = bool(verification["success"])
    failure_reason = "" if success else "object_out_of_place"

    place_result = {
        "success": success,
        "failure_reason": failure_reason,
        "place_execution_mode": "mvp_reconstructed_object",
        "place_object_initialization": "reconstructed",
        "task_json": str(task_json),
        "nav_result": str(Path(args.nav_result).expanduser().resolve()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()) if args.dataset_dir else None,
        "restore": restore_report,
        "stage_prepare": stage_prepare_report,
        "reconstructed_object": reconstruction_report,
        "verification": verification,
        "notes": [
            "This MVP reconstructs the object at place_pose_world after a multi-process JSON handoff.",
            "It is not a continuous physical carry or full cuRobo place plan.",
        ],
    }
    handoff_report = {
        "success": success,
        "failure_reason": failure_reason,
        "task_json": str(task_json),
        "scene_usd": str(scene_usd),
        "nav_result": str(Path(args.nav_result).expanduser().resolve()),
        "stage": {
            "opened": True,
            "scene_usd": str(scene_usd),
            "terrain_prim_path": args.terrain_prim_path,
        },
        "restore": restore_report,
        "stage_prepare": stage_prepare_report,
        "place": {
            "execution_backend": "mvp_reconstructed_object",
            "object_reconstruction": reconstruction_report,
            "verification": verification,
        },
    }
    _write_reports(args, place_result=place_result, handoff_report=handoff_report)
    if args.keep_window_open:
        _keep_window_open_until_closed()
    print(f"[place] success={success} failure_reason={failure_reason}")
    return 0 if success else 1


def main() -> int:
    try:
        return _run()
    except Exception as exc:
        reason = getattr(exc, "failure_reason", "place_exception")
        report = getattr(exc, "report", None)
        detail = str(exc)
        print(f"[place] failed: {reason}: {detail}")
        traceback.print_exc()
        _write_failure(args_cli, str(reason), detail, report if isinstance(report, dict) else None)
        return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
