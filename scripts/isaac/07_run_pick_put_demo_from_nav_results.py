#!/usr/bin/env python3
"""Single-stage pick + put demo from precomputed navigation results.

This runner intentionally does not run navigation control. It opens one Isaac
Sim stage, restores the base from the pick navigation result, runs the existing
pick pipeline, records a logical carry state, then either performs the legacy
MVP object reconstruction or a real arm-place sequence in the same stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

RAW_CLI_ARGS = sys.argv[1:].copy()

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODE = "single_stage_pick_put_demo"
STAGE_PROCESS = "single_isaac_sim_app"
PHYSICAL_NAV_CONTINUITY = False
DEFAULT_CONTEXT_JSON = Path("/tmp/go2_x5_single_stage_pick_put_context.json")
DEFAULT_CUROBO_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
ARM_PLACE_MAX_TARGET_XY_RADIUS_M = float(os.environ.get("GO2_X5_ARM_PLACE_MAX_TARGET_XY_RADIUS_M", "0.85"))
ARM_PLACE_MAX_TARGET_RADIUS_3D_M = float(os.environ.get("GO2_X5_ARM_PLACE_MAX_TARGET_RADIUS_3D_M", "1.05"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DemoFailure(RuntimeError):
    """Failure with a stable reason and optional stage report."""

    def __init__(self, reason: str, detail: str, report: dict[str, Any] | None = None):
        super().__init__(detail)
        self.reason = reason
        self.report = report if isinstance(report, dict) else {}


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-nav-to-pick", default=None)
    parser.add_argument("--nav-pick-result", default=None)
    parser.add_argument("--task-nav-to-place", default=None)
    parser.add_argument("--nav-place-result", default=None)
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--pick-handoff-report", default=None)
    parser.add_argument("--put-result", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--object-pose-debug-only", action="store_true")
    parser.add_argument("--object-pose-debug-report", default="object_pose_debug_report.json")
    parser.add_argument(
        "--reset-object-xform-stack",
        action="store_true",
        help="Clear object xformOpOrder when applying task poses; by default preserve existing asset ops such as unitsResolve.",
    )
    parser.add_argument("--pipeline-context", default=str(DEFAULT_CONTEXT_JSON))
    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    parser.add_argument("--handoff-clearance-radius", type=float, default=0.20)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--stage-load-updates", type=int, default=30)
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument("--demo-visuals", action="store_true")
    parser.add_argument("--keep-window-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-planner-server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--side-retreat-only", action="store_true")
    parser.add_argument("--allow-retreat-success", action="store_true")
    parser.add_argument("--legacy-side-retreat", action="store_true")
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument("--show-grasp-trajectory", action="store_true")
    parser.add_argument("--carry-mode", choices=("none", "logical", "fixed-joint", "kinematic"), default="logical")
    parser.add_argument("--put-mode", choices=("mvp-reconstruct", "release-only", "arm-place"), default="mvp-reconstruct")
    parser.add_argument("--replay-nav-to-pick", action="store_true")
    parser.add_argument("--replay-nav-to-place", action="store_true")
    parser.add_argument("--replay-nav-real-time", action="store_true")
    parser.add_argument("--replay-nav-speed", type=float, default=1.0)
    parser.add_argument("--replay-before-pick-object-prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-nav-place-with-carried-object", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--arm-place-plan-timeout-s",
        type=float,
        default=300.0,
        help="Maximum wall time for the external cuRobo arm-place planner subprocess.",
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    AppLauncher.add_app_launcher_args(parser)
    if any(arg in {"-h", "--help"} for arg in RAW_CLI_ARGS):
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _read_json(path: str | Path, *, missing_reason: str) -> dict[str, Any]:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise DemoFailure(missing_reason, f"missing JSON file: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _required_path_text(value: str | None, label: str) -> str:
    if not value:
        return f"<missing:{label}>"
    return str(Path(value).expanduser().resolve())


def _put_mode_result_value(mode: str) -> str:
    return "arm_place" if mode == "arm-place" else str(mode).replace("-", "_")


def _put_mode_teleports_object(mode: str) -> bool:
    return mode == "mvp-reconstruct"


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    normalized_put_mode = _put_mode_result_value(args.put_mode)
    return {
        "schema_version": 1,
        "success": False,
        "failure_reason": "not_completed",
        "failure_detail": "",
        "mode": MODE,
        "stage_process": STAGE_PROCESS,
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "carry_mode": args.carry_mode,
        "put_mode": normalized_put_mode,
        "arm_place_executed": False,
        "object_teleported": _put_mode_teleports_object(args.put_mode),
        "physical_place_continuity": args.put_mode == "arm-place",
        "task_nav_to_pick": _required_path_text(args.task_nav_to_pick, "task-nav-to-pick"),
        "task_nav_to_place": _required_path_text(args.task_nav_to_place, "task-nav-to-place"),
        "nav_pick_result": _required_path_text(args.nav_pick_result, "nav-pick-result"),
        "nav_place_result": _required_path_text(args.nav_place_result, "nav-place-result"),
        "dataset_dir": _required_path_text(args.dataset_dir, "dataset-dir"),
        "metadata": {
            "complete_physical_nav_carry": False,
            "nav_execution_in_this_process": False,
            "base_transfer_mode": "restore_from_nav_result",
            "replay_nav_to_pick_requested": bool(args.replay_nav_to_pick),
            "replay_nav_to_place_requested": bool(args.replay_nav_to_place),
            "replay_nav_to_pick_executed": False,
            "replay_nav_to_place_executed": False,
            "notes": [
                "This demo opens one Isaac Sim app/stage for pick and put.",
                "Navigation is precomputed; pick/place base poses are restored from nav results or task place.base_goal.",
                "Logical carry is metadata only and is not a real physical carry constraint.",
                "MVP put reconstructs the object at place_pose_world; arm-place physically moves the held object with the arm.",
            ],
        },
        "stages": {
            "open_stage": {},
            "replay_nav_to_pick": {},
            "restore_pick_base": {},
            "prepare_object": {},
            "pick": {},
            "carry_state": {},
            "replay_nav_to_place": {},
            "restore_place_base": {},
            "put": {},
        },
    }


def _write_failure_result(
    args: argparse.Namespace,
    result: dict[str, Any],
    reason: str,
    detail: str,
    *,
    stage_name: str | None = None,
    report: dict[str, Any] | None = None,
) -> None:
    result["success"] = False
    result["failure_reason"] = reason
    result["failure_detail"] = detail
    result["updated_at"] = time.time()
    if stage_name:
        stage_report = result.setdefault("stages", {}).setdefault(stage_name, {})
        stage_report.update(report or {})
        stage_report["success"] = False
        stage_report["failure_reason"] = reason
        stage_report["failure_detail"] = detail
    _write_json(args.result_json, result)


def _validate_full_demo_args(args: argparse.Namespace) -> None:
    required = {
        "--task-nav-to-pick": args.task_nav_to_pick,
        "--nav-pick-result": args.nav_pick_result,
        "--task-nav-to-place": args.task_nav_to_place,
        "--dataset-dir": args.dataset_dir,
        "--pick-handoff-report": args.pick_handoff_report,
        "--put-result": args.put_result,
        "--result-json": args.result_json,
    }
    if args.put_mode == "mvp-reconstruct":
        required["--nav-place-result"] = args.nav_place_result
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise DemoFailure("missing_required_args", f"missing required args for full demo: {missing}")


def _validate_object_pose_debug_args(args: argparse.Namespace) -> None:
    if not args.task_nav_to_pick:
        raise DemoFailure("missing_required_args", "--task-nav-to-pick is required for --object-pose-debug-only")


def _write_context(args: argparse.Namespace, task_pick: dict[str, Any], scene_usd: Path) -> Path:
    context_path = Path(args.pipeline_context).expanduser().resolve()
    nav_place_result_json = (
        str(Path(args.nav_place_result).expanduser().resolve())
        if args.nav_place_result
        else None
    )
    context = {
        "schema_version": 1,
        "mode": MODE,
        "task_json": str(Path(args.task_nav_to_pick).expanduser().resolve()),
        "task_nav_to_pick": str(Path(args.task_nav_to_pick).expanduser().resolve()),
        "task_nav_to_place": str(Path(args.task_nav_to_place).expanduser().resolve()),
        "scene_usd": str(scene_usd),
        "nav_result_json": str(Path(args.nav_pick_result).expanduser().resolve()),
        "nav_pick_result_json": str(Path(args.nav_pick_result).expanduser().resolve()),
        "nav_place_result_json": nav_place_result_json,
        "handoff_report_json": str(Path(args.pick_handoff_report).expanduser().resolve()),
        "terrain_prim_path": args.terrain_prim_path,
        "handoff_clearance_radius": float(args.handoff_clearance_radius),
        "settle_steps": max(0, int(args.settle_steps)),
        "object_reset_stability_steps": 3,
        "dataset_dir": args.dataset_dir,
        "use_planner_server": bool(args.use_planner_server),
        "require_object_lift_success": not bool(args.allow_retreat_success),
        "legacy_side_retreat": bool(args.legacy_side_retreat),
        "side_retreat_only": bool(args.side_retreat_only),
        "side_grasp_fallback_retreat": bool(args.side_grasp_fallback_retreat),
        "show_grasp_trajectory": bool(args.show_grasp_trajectory),
        "object_pose_policy": (task_pick.get("randomization") or {}).get("object_pose_policy", {}),
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "base_transfer_mode": "restore_from_nav_result",
        "put_mode": _put_mode_result_value(args.put_mode),
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    return context_path


def _open_stage(scene_usd: Path, load_updates: int) -> dict[str, Any]:
    import omni.usd

    if not scene_usd.exists():
        raise DemoFailure("open_stage_failed", f"scene USD does not exist: {scene_usd}")
    usd_context = omni.usd.get_context()
    open_result = usd_context.open_stage(str(scene_usd))
    if open_result is False:
        raise DemoFailure("open_stage_failed", f"failed to open stage: {scene_usd}")
    for _ in range(max(1, int(load_updates))):
        simulation_app.update()
    stage = usd_context.get_stage()
    if stage is None:
        raise DemoFailure("open_stage_failed", f"stage did not load: {scene_usd}")
    return {
        "success": True,
        "scene_usd": str(scene_usd),
        "stage_process": STAGE_PROCESS,
        "opened_stage_count": 1,
        "load_updates": max(1, int(load_updates)),
    }


def _ancestor_paths(path: str) -> list[str]:
    parts = [part for part in str(path).strip("/").split("/") if part]
    paths: list[str] = []
    for index in range(1, len(parts) + 1):
        paths.append("/" + "/".join(parts[:index]))
    return paths


def _ensure_required_prims_active(raw_task_pick: dict[str, Any]) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise DemoFailure("open_stage_failed", "No USD stage is open while activating required prims.")
    object_prim_path = str((raw_task_pick.get("pick") or {}).get("object_prim_path") or "/World/apple")
    required_paths = ["/World/go2_x5", object_prim_path]
    reports: list[dict[str, Any]] = []
    for path in required_paths:
        for prim_path in _ancestor_paths(path):
            prim = stage.GetPrimAtPath(prim_path)
            report = {
                "prim_path": prim_path,
                "valid": bool(prim.IsValid()),
            }
            if not prim.IsValid():
                reports.append(report)
                continue
            was_active = bool(prim.IsActive())
            if not was_active:
                prim.SetActive(True)
            report["was_active"] = was_active
            report["is_active"] = bool(prim.IsActive())
            if prim_path == path and prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                try:
                    imageable.MakeVisible()
                    report["made_visible"] = True
                except Exception as exc:
                    report["make_visible_error"] = str(exc)
            reports.append(report)
    return {
        "applied": True,
        "required_paths": required_paths,
        "prim_reports": reports,
        "activated_paths": [
            item["prim_path"]
            for item in reports
            if item.get("valid") and item.get("was_active") is False and item.get("is_active")
        ],
    }


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
        print(f"[single-stage] viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set viewport camera {camera_prim_path}: {exc}")
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


def _validate_nav_result(nav_result: dict[str, Any], *, reason: str) -> None:
    if not nav_result.get("success", False):
        raise DemoFailure(reason, f"navigation result did not succeed: {nav_result.get('failure_reason')}")
    if not isinstance(nav_result.get("final_base_pose_world"), dict):
        raise DemoFailure(reason, "navigation result is missing final_base_pose_world")


def _nav_place_result_from_task_base_goal(raw_task_place: dict[str, Any]) -> dict[str, Any]:
    place = dict(raw_task_place.get("place") or {})
    base_goal = place.get("base_goal")
    if not isinstance(base_goal, dict):
        raise DemoFailure(
            "missing_place_base_goal",
            "--nav-place-result was not supplied and task_nav_to_place.place.base_goal is missing.",
        )
    missing = [key for key in ("x", "y", "yaw") if key not in base_goal]
    if missing:
        raise DemoFailure("missing_place_base_goal", f"place.base_goal is missing keys: {missing}")
    return {
        "schema_version": 1,
        "success": True,
        "failure_reason": "",
        "synthetic_from_task_place_base_goal": True,
        "final_base_pose_world": {
            "x": float(base_goal["x"]),
            "y": float(base_goal["y"]),
            "z": 0.0,
            "yaw": float(base_goal["yaw"]),
        },
    }


def _apply_grasp_policy_env(args: argparse.Namespace) -> None:
    if args.side_retreat_only:
        args.legacy_side_retreat = True
        args.allow_retreat_success = True
    os.environ["GO2_X5_WORKSPACE"] = str(PROJECT_ROOT)
    os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "0"
    os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "0" if args.allow_retreat_success else "1"
    os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0" if args.legacy_side_retreat else "1"
    os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "1" if args.side_grasp_fallback_retreat else "0"
    os.environ["GO2_X5_SHOW_GRASP_TRAJECTORY"] = "1" if args.show_grasp_trajectory else "0"
    os.environ["GO2_X5_PICK_SETTLE_STEPS"] = str(max(0, int(args.settle_steps)))


def _object_pose_from_bbox(bbox: dict[str, Any] | None) -> dict[str, Any]:
    center = list((bbox or {}).get("center_xyz") or [])
    report: dict[str, Any] = {
        "source": "world_bbox_center",
        "bbox_world": bbox,
    }
    if len(center) == 3:
        report.update({"x": float(center[0]), "y": float(center[1]), "z": float(center[2])})
    return report


def _matrix_to_list(matrix) -> list[list[float]] | None:
    if matrix is None:
        return None
    if isinstance(matrix, tuple):
        matrix = matrix[0]
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(0.5 * float(roll))
    sr = math.sin(0.5 * float(roll))
    cp = math.cos(0.5 * float(pitch))
    sp = math.sin(0.5 * float(pitch))
    cy = math.cos(0.5 * float(yaw))
    sy = math.sin(0.5 * float(yaw))
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _rotation_matrix_to_rpy_deg(matrix) -> dict[str, Any] | None:
    matrix_list = _matrix_to_list(matrix)
    if matrix_list is None:
        return None
    rotation = [[float(matrix_list[row][col]) for col in range(3)] for row in range(3)]
    column_norms = []
    for col in range(3):
        norm = math.sqrt(sum(rotation[row][col] * rotation[row][col] for row in range(3)))
        column_norms.append(norm)
        if norm > 1.0e-9:
            for row in range(3):
                rotation[row][col] /= norm
    sy = max(-1.0, min(1.0, -rotation[2][0]))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) > 1.0e-6:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return {
        "convention": "roll_x_pitch_y_yaw_z_degrees_from_matrix",
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "scale_removed_from_columns": column_norms,
    }


def _compute_bbox(stage, prim_path: str) -> dict[str, Any] | None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    bound = bbox_cache.ComputeWorldBound(prim)
    aligned = bound.ComputeAlignedBox()
    bbox_min = [float(value) for value in aligned.GetMin()]
    bbox_max = [float(value) for value in aligned.GetMax()]
    center = [0.5 * (bbox_min[index] + bbox_max[index]) for index in range(3)]
    return {
        "min_xyz": bbox_min,
        "max_xyz": bbox_max,
        "center_xyz": center,
        "top_z": float(bbox_max[2]),
        "center_z": float(center[2]),
    }


def _usd_value_to_json(value) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return {
            "real": float(value.GetReal()),
            "imaginary": [float(component) for component in imaginary],
        }
    try:
        return [float(component) for component in value]
    except Exception:
        return str(value)


def _selected_physics_attrs(prim) -> dict[str, Any]:
    if not prim.IsValid():
        return {}
    selected: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type_name": str(prim.GetTypeName()),
        "applied_schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
        "attrs": {},
    }
    keywords = (
        "velocity",
        "angular",
        "kinematic",
        "rigidbody",
        "gravity",
        "sleep",
        "mass",
        "density",
        "collision",
        "contactoffset",
        "restoffset",
        "approximation",
        "physx",
    )
    for attr in prim.GetAttributes():
        name = attr.GetName()
        lowered = name.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        try:
            selected["attrs"][name] = _usd_value_to_json(attr.Get())
        except Exception as exc:
            selected["attrs"][name] = f"<read_error:{exc}>"
    return selected


def _bbox_xy_overlap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not a or not b:
        return False
    amin, amax = a.get("min_xyz"), a.get("max_xyz")
    bmin, bmax = b.get("min_xyz"), b.get("max_xyz")
    if not amin or not amax or not bmin or not bmax:
        return False
    return bool(float(amin[0]) <= float(bmax[0]) and float(amax[0]) >= float(bmin[0]) and float(amin[1]) <= float(bmax[1]) and float(amax[1]) >= float(bmin[1]))


def _bbox_center_xy_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    ac, bc = a.get("center_xyz"), b.get("center_xyz")
    if not ac or not bc:
        return None
    return float(math.hypot(float(ac[0]) - float(bc[0]), float(ac[1]) - float(bc[1])))


def _support_candidates_from_task(raw_task: dict[str, Any] | None) -> list[str]:
    randomization = (raw_task or {}).get("randomization") or {}
    candidates = [
        randomization.get("table_prim_path"),
        "/World/table",
        "/World/scene_collision",
    ]
    return [str(path) for path in dict.fromkeys(path for path in candidates if path)]


def _nearby_collision_analysis(
    stage,
    object_prim_path: str,
    object_bbox: dict[str, Any] | None,
    *,
    max_items: int = 30,
) -> list[dict[str, Any]]:
    from pxr import Usd, UsdPhysics

    if not object_bbox:
        return []
    object_prefix = object_prim_path.rstrip("/") + "/"
    object_bottom = float(object_bbox["min_xyz"][2])
    records: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if path == object_prim_path or path.startswith(object_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bbox = _compute_bbox(stage, path)
        if not bbox:
            continue
        center_xy_distance = _bbox_center_xy_distance(object_bbox, bbox)
        xy_overlap = _bbox_xy_overlap(object_bbox, bbox)
        z_gap_to_object_bottom = float(object_bottom - float(bbox["max_xyz"][2]))
        if not xy_overlap and (center_xy_distance is None or center_xy_distance > 0.50) and abs(z_gap_to_object_bottom) > 0.25:
            continue
        records.append(
            {
                "path": path,
                "type_name": str(prim.GetTypeName()),
                "applied_schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
                "bbox": bbox,
                "xy_overlap_with_object": xy_overlap,
                "center_xy_distance_m": center_xy_distance,
                "z_gap_to_object_bottom_m": z_gap_to_object_bottom,
                "physics_attrs": _selected_physics_attrs(prim),
            }
        )
    records.sort(
        key=lambda item: (
            not bool(item.get("xy_overlap_with_object")),
            abs(float(item.get("z_gap_to_object_bottom_m", 999.0))),
            float(item.get("center_xy_distance_m") if item.get("center_xy_distance_m") is not None else 999.0),
        )
    )
    return records[:max_items]


def _support_analysis(
    stage,
    object_prim_path: str,
    object_bbox: dict[str, Any] | None,
    raw_task: dict[str, Any] | None,
) -> dict[str, Any]:
    if not object_bbox:
        return {"available": False, "reason": "object_bbox_unavailable"}
    object_bottom = float(object_bbox["min_xyz"][2])
    candidates = []
    for path in _support_candidates_from_task(raw_task):
        prim = stage.GetPrimAtPath(path)
        bbox = _compute_bbox(stage, path) if prim.IsValid() else None
        candidate = {
            "path": path,
            "valid": bool(prim.IsValid()),
            "bbox": bbox,
            "xy_overlap_with_object": _bbox_xy_overlap(object_bbox, bbox),
            "center_xy_distance_m": _bbox_center_xy_distance(object_bbox, bbox),
            "z_gap_to_object_bottom_m": None,
        }
        if bbox:
            candidate["z_gap_to_object_bottom_m"] = float(object_bottom - float(bbox["max_xyz"][2]))
        candidates.append(candidate)
    return {
        "available": True,
        "object_bottom_z": object_bottom,
        "support_candidates": candidates,
        "nearby_collision_prims": _nearby_collision_analysis(stage, object_prim_path, object_bbox),
    }


def _nearest_xy_support_collision(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    support = snapshot.get("support_analysis") or {}
    nearby = support.get("nearby_collision_prims") or []
    xy_overlapping = [item for item in nearby if item.get("xy_overlap_with_object")]
    if not xy_overlapping:
        return None
    return min(
        xy_overlapping,
        key=lambda item: abs(float(item.get("z_gap_to_object_bottom_m", 999.0))),
    )


def _diagnose_object_pose_debug(report: dict[str, Any]) -> dict[str, Any]:
    after_apply = report.get("after_apply") or {}
    stability = report.get("stability") or {}
    support = after_apply.get("support_analysis") or {}
    support_candidates = support.get("support_candidates") or []
    configured_support_invalid = [
        candidate.get("path")
        for candidate in support_candidates
        if candidate.get("path") and candidate.get("valid") is False
    ]
    nearest_support = _nearest_xy_support_collision(after_apply)
    issues: list[str] = []
    notes: list[str] = []
    if configured_support_invalid:
        issues.append("configured_support_prim_invalid")
    if nearest_support is not None:
        gap = float(nearest_support.get("z_gap_to_object_bottom_m", 0.0))
        if gap > 0.01:
            issues.append("object_bottom_above_nearest_support_collision")
        elif gap < -0.002:
            issues.append("object_bbox_intersects_nearest_support_collision_bbox")
        else:
            notes.append("object_bottom_close_to_nearest_support_collision_bbox")
    else:
        issues.append("no_xy_overlapping_support_collision_found")

    stage_movements = {
        "after_physics_init_before_play_m": stability.get("center_displacement_after_physics_init_before_play_m"),
        "after_play_update_before_explicit_step_m": stability.get(
            "center_displacement_after_play_update_before_explicit_step_m"
        ),
        "after_1_step_m": stability.get("center_displacement_after_1_step_m"),
        "after_120_steps_m": stability.get("center_displacement_after_120_steps_m"),
    }
    first_motion_stage = None
    for stage_name, value in stage_movements.items():
        if value is not None and float(value) > 0.001:
            first_motion_stage = stage_name
            break

    play_state = (report.get("after_play_update_before_explicit_step") or {}).get("live_rigid_body_state") or {}
    live_body = (play_state.get("bodies") or [{}])[0]
    if live_body.get("linear_velocity_norm", 0.0) and float(live_body["linear_velocity_norm"]) > 0.01:
        issues.append("object_has_runtime_linear_velocity_after_play")
    if live_body.get("angular_velocity_norm", 0.0) and float(live_body["angular_velocity_norm"]) > 0.10:
        issues.append("object_has_runtime_angular_velocity_after_play")

    likely_cause = "unknown"
    if "object_bottom_above_nearest_support_collision" in issues:
        likely_cause = "task_object_z_is_above_actual_support_collision_height"
    elif "object_bbox_intersects_nearest_support_collision_bbox" in issues:
        likely_cause = "task_object_pose_interpenetrates_support_collision_bbox"
    elif first_motion_stage:
        likely_cause = "physics_runtime_changes_object_after_play"

    return {
        "likely_cause": likely_cause,
        "issues": issues,
        "notes": notes,
        "configured_support_invalid_paths": configured_support_invalid,
        "nearest_xy_support_collision_after_apply": nearest_support,
        "stage_movements": stage_movements,
        "first_motion_stage_over_1mm": first_motion_stage,
        "runtime_velocity_after_play": live_body,
    }


def _live_rigid_body_state(rigid_body_paths: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"available": False, "bodies": []}
    try:
        from isaacsim.core.prims import SingleRigidPrim
    except Exception as exc:
        report["reason"] = f"SingleRigidPrim_unavailable:{exc}"
        return report
    for path in rigid_body_paths:
        body_report: dict[str, Any] = {"path": path}
        try:
            rigid_prim = SingleRigidPrim(prim_path=path, name=f"debug_rigid_{abs(hash(path))}")
            rigid_prim.initialize()
            linear = [float(value) for value in rigid_prim.get_linear_velocity()]
            angular = [float(value) for value in rigid_prim.get_angular_velocity()]
            body_report.update(
                {
                    "linear_velocity": linear,
                    "angular_velocity": angular,
                    "linear_velocity_norm": float(math.sqrt(sum(value * value for value in linear))),
                    "angular_velocity_norm": float(math.sqrt(sum(value * value for value in angular))),
                }
            )
        except Exception as exc:
            body_report["error"] = str(exc)
        report["bodies"].append(body_report)
    report["available"] = any("error" not in item for item in report["bodies"])
    return report


def _xform_op_order_names(xformable) -> list[str]:
    return [op.GetOpName() for op in xformable.GetOrderedXformOps()]


def _get_or_add_xform_op(xformable, op_type):
    from pxr import UsdGeom

    prim = xformable.GetPrim()
    attr_name_by_type = {
        UsdGeom.XformOp.TypeTranslate: "xformOp:translate",
        UsdGeom.XformOp.TypeOrient: "xformOp:orient",
        UsdGeom.XformOp.TypeScale: "xformOp:scale",
    }
    attr = prim.GetAttribute(attr_name_by_type[op_type])
    if attr.IsValid():
        return UsdGeom.XformOp(attr)
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
    raise ValueError(f"unsupported xform op type: {op_type}")


def _set_translate_op(op, xyz: tuple[float, float, float]) -> None:
    from pxr import Gf

    if op.GetPrecision() == op.PrecisionFloat:
        op.Set(Gf.Vec3f(*xyz))
    else:
        op.Set(Gf.Vec3d(*xyz))


def _set_orient_op(op, quat_wxyz: tuple[float, float, float, float]) -> None:
    from pxr import Gf

    w, x, y, z = (float(value) for value in quat_wxyz)
    if op.GetPrecision() == op.PrecisionDouble:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _saved_scale_from_xform_stack(xformable) -> tuple[float, float, float] | None:
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() != UsdGeom.XformOp.TypeScale:
            continue
        value = op.Get()
        if value is None:
            continue
        values = tuple(float(component) for component in value)
        if len(values) == 3:
            return values
    return None


def _remove_authored_xform_ops(prim) -> None:
    for attr in list(prim.GetAttributes()):
        name = attr.GetName()
        if name.startswith("xformOp:") or name == "xformOpOrder":
            prim.RemoveProperty(name)


def _apply_debug_object_pose(prim, pose: dict[str, Any], *, reset_xform_stack: bool) -> dict[str, Any]:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    order_before = _xform_op_order_names(xformable)
    quat = _rpy_to_quat_wxyz(float(pose.get("roll", 0.0)), float(pose.get("pitch", 0.0)), float(pose.get("yaw", 0.0)))
    if reset_xform_stack:
        saved_scale = _saved_scale_from_xform_stack(xformable)
        xformable.ClearXformOpOrder()
        _remove_authored_xform_ops(prim)
        xformable = UsdGeom.Xformable(prim)
        translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
        _set_translate_op(translate_op, (float(pose["x"]), float(pose["y"]), float(pose["z"])))
        _set_orient_op(orient_op, quat)
        ordered_ops = [translate_op, orient_op]
        if saved_scale is not None:
            scale_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeScale)
            scale_op.Set(Gf.Vec3f(*saved_scale))
            ordered_ops.append(scale_op)
        xformable.SetXformOpOrder(ordered_ops)
    else:
        translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
        _set_translate_op(translate_op, (float(pose["x"]), float(pose["y"]), float(pose["z"])))
        _set_orient_op(orient_op, quat)
    xformable = UsdGeom.Xformable(prim)
    return {
        "task_pose_rad": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
            "roll": float(pose.get("roll", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
        },
        "task_pose_deg": {
            "roll": math.degrees(float(pose.get("roll", 0.0))),
            "pitch": math.degrees(float(pose.get("pitch", 0.0))),
            "yaw": math.degrees(float(pose.get("yaw", 0.0))),
        },
        "quaternion_wxyz": [float(value) for value in quat],
        "reset_object_xform_stack": bool(reset_xform_stack),
        "xformOpOrder_before_apply": order_before,
        "xformOpOrder_after_apply": _xform_op_order_names(xformable),
    }


def _prim_debug_snapshot(
    stage,
    object_prim_path: str,
    label: str,
    *,
    raw_task: dict[str, Any] | None = None,
    include_live_physics: bool = False,
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(object_prim_path)
    report: dict[str, Any] = {
        "label": label,
        "object_prim_path": object_prim_path,
        "object_prim_valid": bool(prim.IsValid()),
    }
    if not prim.IsValid():
        return report
    rigid_paths = [str(child.GetPath()) for child in Usd.PrimRange(prim) if child.HasAPI(UsdPhysics.RigidBodyAPI)]
    collider_paths = [str(child.GetPath()) for child in Usd.PrimRange(prim) if child.HasAPI(UsdPhysics.CollisionAPI)]
    report["rigid_body_prim_path"] = rigid_paths[0] if rigid_paths else None
    report["rigid_body_prim_paths"] = rigid_paths
    report["collider_prim_paths"] = collider_paths
    report["object_physics_attrs"] = _selected_physics_attrs(prim)
    report["rigid_body_physics_attrs"] = [
        _selected_physics_attrs(stage.GetPrimAtPath(path)) for path in rigid_paths
    ]
    report["collider_physics_attrs"] = [
        _selected_physics_attrs(stage.GetPrimAtPath(path)) for path in collider_paths
    ]
    if include_live_physics:
        report["live_rigid_body_state"] = _live_rigid_body_state(rigid_paths)
    if prim.IsA(UsdGeom.Xformable):
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        report.update(
            {
                "xformOpOrder": _xform_op_order_names(xformable),
                "local_transform": _matrix_to_list(local_transform),
                "world_transform": _matrix_to_list(world_transform),
                "ui_equivalent_euler_xyz_deg": _rotation_matrix_to_rpy_deg(local_transform),
                "world_euler_xyz_deg": _rotation_matrix_to_rpy_deg(world_transform),
            }
        )
    report["bbox"] = _compute_bbox(stage, object_prim_path)
    report["support_analysis"] = _support_analysis(stage, object_prim_path, report.get("bbox"), raw_task)
    return report


def _bbox_center_distance(before: dict[str, Any] | None, after: dict[str, Any] | None) -> float | None:
    before_center = (before or {}).get("center_xyz")
    after_center = (after or {}).get("center_xyz")
    if not before_center or not after_center:
        return None
    return float(math.sqrt(sum((float(after_center[index]) - float(before_center[index])) ** 2 for index in range(3))))


async def _initialize_debug_world(*, play: bool):
    import omni.kit.app
    from isaacsim.core.api.world import World

    world = World.instance()
    if world is None:
        world = World()
    if world.get_physics_context() is None:
        await world.initialize_simulation_context_async()
    if play:
        await world.play_async()
        await omni.kit.app.get_app().next_update_async()
    return world


async def _run_object_pose_debug_async(args: argparse.Namespace) -> dict[str, Any]:
    import omni.kit.app
    import omni.usd

    _validate_object_pose_debug_args(args)
    task_path = _project_path(args.task_nav_to_pick)
    raw_task = _read_json(task_path, missing_reason="missing_task_nav_to_pick")
    scene_usd = _project_path(args.scene_usd or raw_task["scene_usd"])
    open_stage_report = _open_stage(scene_usd, args.stage_load_updates)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise DemoFailure("open_stage_failed", f"stage did not load: {scene_usd}")
    pick = dict(raw_task.get("pick") or {})
    object_prim_path = str(pick.get("object_prim_path") or "")
    if not object_prim_path:
        raise DemoFailure("object_pose_debug_failed", "task.pick.object_prim_path is missing.")
    object_pose = pick.get("object_pose_world")
    if not isinstance(object_pose, dict):
        raise DemoFailure("object_pose_debug_failed", "task.pick.object_pose_world is missing.")

    before = _prim_debug_snapshot(stage, object_prim_path, "before_apply", raw_task=raw_task)
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise DemoFailure("object_pose_debug_failed", f"object prim is not valid: {object_prim_path}", before)
    apply_report = _apply_debug_object_pose(prim, object_pose, reset_xform_stack=bool(args.reset_object_xform_stack))
    after_apply = _prim_debug_snapshot(stage, object_prim_path, "after_apply", raw_task=raw_task)

    world = await _initialize_debug_world(play=False)
    after_physics_init_before_play = _prim_debug_snapshot(
        stage,
        object_prim_path,
        "after_physics_init_before_play",
        raw_task=raw_task,
        include_live_physics=True,
    )
    await world.play_async()
    await omni.kit.app.get_app().next_update_async()
    after_play_update_before_explicit_step = _prim_debug_snapshot(
        stage,
        object_prim_path,
        "after_play_update_before_explicit_step",
        raw_task=raw_task,
        include_live_physics=True,
    )
    world.step(render=True)
    await omni.kit.app.get_app().next_update_async()
    after_step_1 = _prim_debug_snapshot(
        stage,
        object_prim_path,
        "after_step_1",
        raw_task=raw_task,
        include_live_physics=True,
    )
    for _ in range(119):
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()
    after_step_120 = _prim_debug_snapshot(
        stage,
        object_prim_path,
        "after_step_120",
        raw_task=raw_task,
        include_live_physics=True,
    )

    center_displacement_1 = _bbox_center_distance(after_apply.get("bbox"), after_step_1.get("bbox"))
    center_displacement_120 = _bbox_center_distance(after_apply.get("bbox"), after_step_120.get("bbox"))
    center_displacement_after_physics_init = _bbox_center_distance(
        after_apply.get("bbox"),
        after_physics_init_before_play.get("bbox"),
    )
    center_displacement_after_play_update = _bbox_center_distance(
        after_apply.get("bbox"),
        after_play_update_before_explicit_step.get("bbox"),
    )
    stable_after_120 = center_displacement_120 is not None and center_displacement_120 <= 0.01
    report = {
        "schema_version": 1,
        "success": True,
        "failure_reason": "",
        "mode": "object_pose_debug_only",
        "scene_usd": str(scene_usd),
        "task_nav_to_pick": str(task_path),
        "object_prim_path": object_prim_path,
        "reset_object_xform_stack": bool(args.reset_object_xform_stack),
        "open_stage": open_stage_report,
        "before_apply": before,
        "apply": apply_report,
        "after_apply": after_apply,
        "after_physics_init_before_play": after_physics_init_before_play,
        "after_play_update_before_explicit_step": after_play_update_before_explicit_step,
        "after_step_1": after_step_1,
        "after_step_120": after_step_120,
        "stability": {
            "center_displacement_after_physics_init_before_play_m": center_displacement_after_physics_init,
            "center_displacement_after_play_update_before_explicit_step_m": center_displacement_after_play_update,
            "center_displacement_after_1_step_m": center_displacement_1,
            "center_displacement_after_120_steps_m": center_displacement_120,
            "stable_after_120_steps": stable_after_120,
            "interpretation": (
                "single_object_pose_stable"
                if stable_after_120
                else "single_object_pose_unstable_or_bbox_unavailable"
            ),
        },
        "notes": [
            "Debug-only mode does not restore the robot.",
            "Debug-only mode does not clear object velocity.",
            "Debug-only mode does not make the object kinematic.",
            "Debug-only mode does not attach, pick, or place the object.",
        ],
        "updated_at": time.time(),
    }
    report["diagnosis"] = _diagnose_object_pose_debug(report)
    print(
        "[object-pose-debug]",
        {
            "object_prim_path": object_prim_path,
            "valid": before.get("object_prim_valid"),
            "xform_before": before.get("xformOpOrder"),
            "xform_after": after_apply.get("xformOpOrder"),
            "center_displacement_after_120_steps_m": center_displacement_120,
            "stable_after_120_steps": stable_after_120,
            "likely_cause": report["diagnosis"].get("likely_cause"),
            "issues": report["diagnosis"].get("issues"),
        },
    )
    return report


async def _prepare_object_for_pick(
    world,
    task_pick,
    context: dict[str, Any],
    *,
    reset_xform_stack: bool,
) -> dict[str, Any]:
    import omni.usd
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    stage = omni.usd.get_context().get_stage()
    object_prim_path = task_pick.pick.object_prim_path
    bbox_before = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    visibility = pick_handoff._show_only_task_object(task_pick)
    pose_report = pick_handoff._apply_object_pose_from_task(
        task_pick,
        reset_xform_stack=bool(reset_xform_stack),
    )
    stability = await pick_handoff._stabilize_object_before_target(
        world,
        object_prim_path,
        bbox_after_pose_apply=pose_report.get("world_bbox_after"),
        context=context,
    )
    velocity_reset = stability.get("object_velocity_reset_before_target") or stability.get("velocity_reset_after_short_step")
    velocity_zeroed_count = 0
    if isinstance(velocity_reset, dict):
        velocity_zeroed_count = int(velocity_reset.get("dynamic_rigid_body_count", 0))
    report = {
        "success": not bool(stability.get("failure_reason")),
        "object_prim_path": object_prim_path,
        "object_pose_expected": pose_report.get("pose_world"),
        "object_bbox_before": bbox_before,
        "object_pose_apply": pose_report,
        "object_visibility": visibility,
        "object_stability": stability,
        "object_bbox_after": stability.get("bbox_after_final_reset") or stability.get("bbox_after_short_step"),
        "center_displacement_m": stability.get("center_displacement_m"),
        "velocity_zeroed_count": velocity_zeroed_count,
        "linear_velocity_norm": stability.get("linear_velocity_norm"),
        "angular_velocity_norm": stability.get("angular_velocity_norm"),
    }
    if stability.get("failure_reason"):
        raise DemoFailure(
            "object_unstable_before_pick",
            f"object unstable before pick: {stability.get('failure_reason')}",
            report,
        )
    return report


async def _run_pick(task_pick, nav_pick_result: dict[str, Any], args: argparse.Namespace, recorder) -> dict[str, Any]:
    from source.manipulation import GraspPipeline, GraspTask
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    pick_dir = Path(args.dataset_dir).expanduser().resolve()
    pick_dir.mkdir(parents=True, exist_ok=True)
    task_spec = GraspTask(
        object_prim_path=task_pick.pick.object_prim_path,
        grasp_mode=task_pick.pick.grasp_mode,
        use_planner_server=bool(args.use_planner_server),
        state_json=str(pick_dir / "pick_state.json"),
        target_json=str(pick_dir / "pick_target.json"),
        plan_json=str(pick_dir / "pick_plan.json"),
        result_json=str(pick_dir / "pick_execution_result.json"),
    )
    started_at = time.time()
    pipeline = GraspPipeline(recorder=recorder)
    state = await pipeline.export_state(task_spec)
    target = await pipeline.generate_target(task_spec)
    plan = pipeline.plan(task_spec)
    execution = await pipeline.execute(task_spec)
    execution_summary = execution.get("summary", {})
    success = bool(execution_summary.get("task_success", False))
    failure_reason = "" if success else pick_handoff._execution_failure_reason(execution_summary)
    return {
        "success": success,
        "failure_reason": failure_reason,
        "elapsed_wall_time_s": time.time() - started_at,
        "task_spec": {
            "object_prim_path": task_spec.object_prim_path,
            "grasp_mode": task_spec.grasp_mode,
            "state_json": task_spec.state_json,
            "target_json": task_spec.target_json,
            "plan_json": task_spec.plan_json,
            "result_json": task_spec.result_json,
            "use_planner_server": task_spec.use_planner_server,
        },
        "state_export": pick_handoff._state_export_report(state, nav_pick_result),
        "target": pick_handoff._summarize_target(target),
        "plan_summary": plan.get("summary", plan),
        "execution_summary": execution_summary,
        "execution": execution,
    }


def _carry_state(args: argparse.Namespace, object_prim_path: str) -> dict[str, Any]:
    from scripts.isaac import run_pick_from_nav_result as pick_handoff
    import omni.usd

    if args.carry_mode in {"fixed-joint", "kinematic"}:
        raise DemoFailure(
            "carry_mode_not_implemented",
            f"carry_mode={args.carry_mode!r} is not implemented in phase 1; use logical.",
        )
    stage = omni.usd.get_context().get_stage()
    bbox = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    return {
        "success": True,
        "object_prim_path": object_prim_path,
        "carry_mode": args.carry_mode,
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "object_pose_world_after_pick": _object_pose_from_bbox(bbox),
        "gripper_state": "closed",
        "source": "pick_result",
        "logical_only": args.carry_mode == "logical",
        "notes": [
            "Logical carry is metadata only.",
            "No fixed joint or true continuous nav carry is active in phase 1.",
            "arm-place base restore may temporarily freeze and TCP-clamp the object as a handoff stabilization step.",
        ],
    }


def _verify_mvp_place(
    place_pose: dict[str, Any],
    bbox_after: dict[str, Any] | None,
    *,
    xy_tolerance: float,
    z_tolerance: float,
) -> dict[str, Any]:
    center = list((bbox_after or {}).get("center_xyz") or [])
    if len(center) != 3:
        return {
            "success": False,
            "failure_reason": "object_bbox_missing",
            "place_xy_tolerance": xy_tolerance,
            "place_z_tolerance": z_tolerance,
        }
    xy_error = ((float(center[0]) - float(place_pose["x"])) ** 2 + (float(center[1]) - float(place_pose["y"])) ** 2) ** 0.5
    z_error = abs(float(center[2]) - float(place_pose["z"]))
    return {
        "success": bool(xy_error <= xy_tolerance and z_error <= z_tolerance),
        "place_xy_error": float(xy_error),
        "place_z_error": float(z_error),
        "place_xy_tolerance": float(xy_tolerance),
        "place_z_tolerance": float(z_tolerance),
        "object_center_xyz": [float(value) for value in center],
    }


def _apply_reconstructed_object_to_pose(raw_task_place: dict[str, Any], place: dict[str, Any]) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    object_prim_path = (raw_task_place.get("pick") or {}).get("object_prim_path")
    if not object_prim_path:
        raise DemoFailure("missing_place_pose_world", "pick.object_prim_path is required for MVP put.")
    pose = dict(place["place_pose_world"])
    missing = [key for key in ("x", "y", "z") if key not in pose]
    if missing:
        raise DemoFailure("missing_place_pose_world", f"place_pose_world is missing keys: {missing}")
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise DemoFailure("put_failed", "No USD stage is open for MVP put.")
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise DemoFailure("put_failed", f"object prim does not exist: {object_prim_path}")
    if not prim.IsA(UsdGeom.Xformable):
        raise DemoFailure("put_failed", f"object prim is not xformable: {object_prim_path}")

    roll = float(pose.get("roll", 0.0))
    pitch = float(pose.get("pitch", 0.0))
    yaw = float(pose.get("yaw", 0.0))
    quat = pick_handoff._rpy_to_quat_wxyz(roll, pitch, yaw)
    xformable = UsdGeom.Xformable(prim)
    xform_op_order_before = pick_handoff._xform_op_order_names(xformable)
    pick_handoff._reset_pose_xform_stack(
        xformable,
        (float(pose["x"]), float(pose["y"]), float(pose["z"])),
        quat,
    )
    xformable = UsdGeom.Xformable(prim)
    velocity_reset = pick_handoff.reset_object_physics_state(
        object_prim_path,
        zero_linear_velocity=True,
        zero_angular_velocity=True,
        wake=True,
    )
    return {
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
        "reset_xform_stack": True,
        "xform_op_order_before": xform_op_order_before,
        "xform_op_order_after": pick_handoff._xform_op_order_names(xformable),
        "local_transform_after": pick_handoff._matrix_to_nested_list(xformable.GetLocalTransformation()),
        "world_bbox_after": pick_handoff._compute_world_bbox(stage, object_prim_path),
        "object_velocity_reset_after_reconstruction": velocity_reset,
        "velocity_zeroed_count": int(velocity_reset.get("dynamic_rigid_body_count", 0)),
        "velocity_before": velocity_reset.get("velocity_before", {}),
        "velocity_after": velocity_reset.get("velocity_after", {}),
        "linear_velocity_norm": float(velocity_reset.get("velocity_after", {}).get("linear_velocity_norm_max", 0.0)),
        "angular_velocity_norm": float(velocity_reset.get("velocity_after", {}).get("angular_velocity_norm_max", 0.0)),
    }


def _settle_world(world, steps: int) -> None:
    for _ in range(max(0, int(steps))):
        world.step(render=True)
        simulation_app.update()


def _run_mvp_put(world, raw_task_place: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from scripts.isaac import run_pick_from_nav_result as pick_handoff
    import omni.usd

    if args.put_mode != "mvp-reconstruct":
        raise DemoFailure("put_mode_not_implemented", f"put_mode={args.put_mode!r} is not implemented in phase 1.")
    place = dict(raw_task_place.get("place") or {})
    place_pose = place.get("place_pose_world")
    if not place.get("enabled", False) or not place_pose:
        raise DemoFailure("missing_place_pose_world", "task_nav_to_place.place.place_pose_world is missing or disabled.")

    reconstruction = _apply_reconstructed_object_to_pose(raw_task_place, place)
    _settle_world(world, args.settle_steps)
    object_prim_path = reconstruction["object_prim_path"]
    final_reset = pick_handoff.reset_object_physics_state(
        object_prim_path,
        zero_linear_velocity=True,
        zero_angular_velocity=True,
        wake=True,
    )
    stage = omni.usd.get_context().get_stage()
    bbox_after = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    verification = _verify_mvp_place(
        dict(place_pose),
        bbox_after,
        xy_tolerance=float(place.get("place_xy_tolerance", 0.10)),
        z_tolerance=float(place.get("place_z_tolerance", 0.08)),
    )
    report = {
        "success": bool(verification["success"]),
        "failure_reason": "" if verification["success"] else "put_failed",
        "put_mode": args.put_mode,
        "execution_backend": "mvp_reconstruct_object",
        "physical_put_execution": False,
        "object_reconstruction": reconstruction,
        "velocity_reset_after_settle": final_reset,
        "bbox_after_settle": bbox_after,
        "verification": verification,
        "notes": [
            "MVP put reconstructs the object at place_pose_world in the same stage.",
            "This is not a full arm put/release plan.",
        ],
    }
    if not verification["success"]:
        raise DemoFailure("put_failed", "MVP reconstructed object did not verify at place pose.", report)
    return report


def _load_script_module(module_prefix: str, path: Path):
    import types

    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"{module_prefix}_{time.time_ns()}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _matrix_from_pose_dict(pose_dict: dict[str, Any]):
    import numpy as np

    if "matrix_4x4" in pose_dict:
        return np.asarray(pose_dict["matrix_4x4"], dtype=float)
    from scripts.math.SE3 import pose_to_matrix

    return pose_to_matrix(pose_dict["position_xyz"], pose_dict["quaternion_wxyz"])


def _planar_pose_matrix(x: float, y: float, z: float, yaw: float):
    import numpy as np

    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    matrix[:3, 3] = [float(x), float(y), float(z)]
    return matrix


def _usd_prim_world_matrix(stage, prim_path: str):
    import numpy as np
    from pxr import Usd, UsdGeom
    from scripts.math.SE3 import normalize_quat_wxyz, pose_to_matrix

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise DemoFailure("stage_not_ready", f"object prim does not exist: {prim_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = xform_cache.GetLocalToWorldTransform(prim)
    translation = world_matrix.ExtractTranslation()
    rotation = world_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return pose_to_matrix(
        np.asarray([translation[0], translation[1], translation[2]], dtype=float),
        normalize_quat_wxyz([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]]),
    )


def _rigid_body_paths_under(stage, prim_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return []
    return [
        str(body_prim.GetPath())
        for body_prim in Usd.PrimRange(prim)
        if body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def _preferred_rigid_body_path(stage, object_prim_path: str) -> str | None:
    rigid_body_paths = _rigid_body_paths_under(stage, object_prim_path)
    if object_prim_path in rigid_body_paths:
        return object_prim_path
    return rigid_body_paths[0] if rigid_body_paths else None


def _dynamic_object_world_matrix(stage, prim_path: str):
    from scripts.math.SE3 import pose_to_matrix

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise DemoFailure("stage_not_ready", f"object prim does not exist: {prim_path}")
    rigid_body_paths = _rigid_body_paths_under(stage, prim_path)
    live_body_path = _preferred_rigid_body_path(stage, prim_path)
    report: dict[str, Any] = {
        "object_prim_path": prim_path,
        "rigid_body_paths": rigid_body_paths,
        "live_rigid_body_prim_path": live_body_path,
        "source": "usd_xform_fallback",
    }
    if live_body_path:
        try:
            from isaacsim.core.prims import SingleRigidPrim

            wrapper_name = "go2_x5_object_pose_read_" + live_body_path.strip("/").replace("/", "_")
            rigid_prim = SingleRigidPrim(
                prim_path=live_body_path,
                name=wrapper_name,
                reset_xform_properties=False,
            )
            try:
                rigid_prim.initialize()
                report["live_initialized"] = True
            except Exception as exc:
                report["live_initialized"] = False
                report["live_initialize_error"] = str(exc)
            position, orientation = rigid_prim.get_world_pose()
            report.update(
                {
                    "source": "live_rigid_body",
                    "live_rigid_body_prim_path": live_body_path,
                    "position_xyz": [float(value) for value in position],
                    "quaternion_wxyz": [float(value) for value in orientation],
                }
            )
            return pose_to_matrix(position, orientation), report
        except Exception as exc:
            report["live_read_error"] = str(exc)

    matrix = _usd_prim_world_matrix(stage, prim_path)
    from scripts.math.SE3 import matrix_to_pose

    position, orientation = matrix_to_pose(matrix)
    report.update(
        {
            "position_xyz": [float(value) for value in position],
            "quaternion_wxyz": [float(value) for value in orientation],
        }
    )
    return matrix, report


def _transform_bbox_world(bbox: dict[str, Any] | None, transform_world) -> dict[str, Any] | None:
    import itertools
    import numpy as np

    if not bbox:
        return None
    min_xyz = bbox.get("min_xyz")
    max_xyz = bbox.get("max_xyz")
    if not min_xyz or not max_xyz:
        return None
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x, y, z in itertools.product(
                [float(min_xyz[0]), float(max_xyz[0])],
                [float(min_xyz[1]), float(max_xyz[1])],
                [float(min_xyz[2]), float(max_xyz[2])],
            )
        ],
        dtype=float,
    )
    transformed = (np.asarray(transform_world, dtype=float) @ corners.T).T[:, :3]
    new_min = transformed.min(axis=0)
    new_max = transformed.max(axis=0)
    center = 0.5 * (new_min + new_max)
    return {
        "min_xyz": [float(value) for value in new_min],
        "max_xyz": [float(value) for value in new_max],
        "center_xyz": [float(value) for value in center],
        "size_xyz": [float(value) for value in (new_max - new_min)],
        "top_z": float(new_max[2]),
        "center_z": float(center[2]),
        "source": "bbox_before_base_restore_transformed_by_carry_delta",
    }


def _set_prim_world_matrix(stage, prim_path: str, target_world_matrix, *, reset_xform_stack: bool = True) -> dict[str, Any]:
    import numpy as np
    from pxr import Usd, UsdGeom
    from scripts.math.SE3 import matrix_to_pose
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise DemoFailure("stage_not_ready", f"object prim does not exist: {prim_path}")
    if not prim.IsA(UsdGeom.Xformable):
        raise DemoFailure("stage_not_ready", f"object prim is not xformable: {prim_path}")

    parent = prim.GetParent()
    if parent and parent.IsValid():
        parent_world = _usd_prim_world_matrix(stage, str(parent.GetPath()))
    else:
        parent_world = np.eye(4, dtype=float)
    target_local_matrix = np.linalg.inv(parent_world) @ np.asarray(target_world_matrix, dtype=float)
    local_position, local_quat = matrix_to_pose(target_local_matrix)
    world_position, world_quat = matrix_to_pose(target_world_matrix)

    xformable = UsdGeom.Xformable(prim)
    order_before = pick_handoff._xform_op_order_names(xformable)
    if reset_xform_stack:
        pick_handoff._reset_pose_xform_stack(xformable, local_position, local_quat)
    else:
        translate_op = pick_handoff._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
        orient_op = pick_handoff._get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
        pick_handoff._set_translate_op(translate_op, local_position)
        pick_handoff._set_orient_op(orient_op, local_quat)
    xformable = UsdGeom.Xformable(prim)
    return {
        "object_prim_path": prim_path,
        "reset_xform_stack": bool(reset_xform_stack),
        "parent_prim_path": str(parent.GetPath()) if parent and parent.IsValid() else None,
        "xform_op_order_before": order_before,
        "xform_op_order_after": pick_handoff._xform_op_order_names(xformable),
        "target_world_position_xyz": [float(value) for value in world_position],
        "target_world_quaternion_wxyz": [float(value) for value in world_quat],
        "target_local_position_xyz": [float(value) for value in local_position],
        "target_local_quaternion_wxyz": [float(value) for value in local_quat],
    }


def _set_dynamic_object_world_matrix(
    stage,
    prim_path: str,
    target_world_matrix,
    *,
    reset_xform_stack: bool = True,
) -> dict[str, Any]:
    import numpy as np
    from pxr import UsdGeom
    from scripts.math.SE3 import matrix_to_pose
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise DemoFailure("stage_not_ready", f"object prim does not exist: {prim_path}")

    xformable = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xformable) else None
    order_before = pick_handoff._xform_op_order_names(xformable) if xformable else []
    target_position, target_quaternion = matrix_to_pose(target_world_matrix)
    rigid_body_paths = _rigid_body_paths_under(stage, prim_path)
    live_body_path = _preferred_rigid_body_path(stage, prim_path)
    live_apply: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "backend": "isaacsim.core.prims.SingleRigidPrim",
        "object_prim_path": prim_path,
        "rigid_body_paths": rigid_body_paths,
        "live_rigid_body_prim_path": live_body_path,
    }
    if live_body_path:
        live_apply["attempted"] = True
        try:
            from isaacsim.core.prims import SingleRigidPrim

            wrapper_name = "go2_x5_object_carry_" + live_body_path.strip("/").replace("/", "_")
            rigid_prim = SingleRigidPrim(
                prim_path=live_body_path,
                name=wrapper_name,
                reset_xform_properties=False,
            )
            try:
                rigid_prim.initialize()
                live_apply["initialized"] = True
            except Exception as exc:
                live_apply["initialized"] = False
                live_apply["initialize_error"] = str(exc)

            try:
                before_position, before_orientation = rigid_prim.get_world_pose()
                live_apply["live_pose_before"] = {
                    "position_xyz": [float(value) for value in before_position],
                    "quaternion_wxyz": [float(value) for value in before_orientation],
                }
            except Exception as exc:
                live_apply["live_pose_before_error"] = str(exc)

            rigid_prim.set_world_pose(
                position=np.asarray(target_position, dtype=np.float32),
                orientation=np.asarray(target_quaternion, dtype=np.float32),
            )
            try:
                rigid_prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
                rigid_prim.set_angular_velocity(np.zeros(3, dtype=np.float32))
                live_apply["velocity_zeroed"] = True
            except Exception as exc:
                live_apply["velocity_zero_error"] = str(exc)
            try:
                after_position, after_orientation = rigid_prim.get_world_pose()
                live_apply["live_pose_after"] = {
                    "position_xyz": [float(value) for value in after_position],
                    "quaternion_wxyz": [float(value) for value in after_orientation],
                }
            except Exception as exc:
                live_apply["live_pose_after_error"] = str(exc)
            live_apply["success"] = True
        except Exception as exc:
            live_apply["error"] = str(exc)
    else:
        live_apply["reason"] = "no_rigid_body_api_under_object_root"

    usd_apply = None
    warning = None
    if not live_apply.get("success", False):
        usd_apply = _set_prim_world_matrix(
            stage,
            prim_path,
            target_world_matrix,
            reset_xform_stack=False,
        )
        warning = "live_rigid_body_pose_not_updated"

    xformable_after = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xformable) else None
    return {
        "object_prim_path": prim_path,
        "target_world_position_xyz": [float(value) for value in target_position],
        "target_world_quaternion_wxyz": [float(value) for value in target_quaternion],
        "reset_xform_stack_requested": bool(reset_xform_stack),
        "xform_op_order_before": order_before,
        "xform_op_order_after": (
            pick_handoff._xform_op_order_names(xformable_after) if xformable_after else []
        ),
        "live_rigid_pose_apply": live_apply,
        "usd_fallback_pose_apply": {
            "used": usd_apply is not None,
            "report": usd_apply,
        },
        "warning": warning,
    }


def _nav_root_report_from_pose(nav_pose: dict[str, Any], root_z: float, *, read_error: str | None = None) -> dict[str, Any]:
    from source.navigation.adapters.frame_utils import yaw_to_quat_wxyz

    yaw = float(nav_pose["yaw"])
    report = {
        "position_xyz": [float(nav_pose["x"]), float(nav_pose["y"]), float(root_z)],
        "quaternion_wxyz": [float(value) for value in yaw_to_quat_wxyz(yaw)],
        "yaw": yaw,
        "linear_velocity_xyz": [0.0, 0.0, 0.0],
        "angular_velocity_xyz": [0.0, 0.0, 0.0],
        "linear_speed_xy": 0.0,
        "angular_speed_z": 0.0,
        "source": "nav_place_result_target_pose_fallback",
    }
    if read_error:
        report["read_error"] = read_error
    return report


def _matrix_pose_report(matrix, *, frame: str = "world") -> dict[str, Any]:
    import numpy as np
    from scripts.math.SE3 import matrix_to_pose

    pose_matrix = np.asarray(matrix, dtype=float)
    position, quaternion = matrix_to_pose(pose_matrix)
    return {
        "frame": frame,
        "position_xyz": [float(value) for value in position],
        "quaternion_wxyz": [float(value) for value in quaternion],
        "matrix_4x4": pose_matrix.tolist(),
    }


def _find_first_prim_by_name(stage, name: str, *, under_path: str | None = None) -> str | None:
    root = stage.GetPrimAtPath(under_path) if under_path else stage.GetPseudoRoot()
    if root is None or not root.IsValid():
        return None
    from pxr import Usd

    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def _resolve_tcp_prim_path(stage) -> str:
    candidates = (
        "/World/go2_x5/arm_link6/grasp_tcp_link",
        "/World/go2_x5/grasp_tcp_link",
    )
    for prim_path in candidates:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return prim_path
    resolved = _find_first_prim_by_name(stage, "grasp_tcp_link", under_path="/World/go2_x5")
    if resolved:
        return resolved
    resolved = _find_first_prim_by_name(stage, "grasp_tcp_link")
    if resolved:
        return resolved
    raise DemoFailure("stage_not_ready", "grasp_tcp_link prim could not be resolved for carried base transfer.")


def _tcp_world_matrix(stage) -> tuple[Any, dict[str, Any]]:
    tcp_path = _resolve_tcp_prim_path(stage)
    matrix = _usd_prim_world_matrix(stage, tcp_path)
    report = _matrix_pose_report(matrix, frame="world")
    report.update({"tcp_prim_path": tcp_path, "source": "usd_xform"})
    return matrix, report


def _read_bool_attr_value(attr, default: bool | None = None) -> bool | None:
    try:
        if attr is not None and attr.IsValid():
            value = attr.Get()
            if value is not None:
                return bool(value)
    except Exception:
        pass
    return default


def _rigid_body_attr_or_create(api, get_name: str, create_name: str):
    attr = getattr(api, get_name)()
    if attr is not None and attr.IsValid():
        return attr, False
    return getattr(api, create_name)(), True


def _set_object_kinematic_enabled(stage, object_prim_path: str, enabled: bool) -> dict[str, Any]:
    from pxr import UsdPhysics

    body_paths = _rigid_body_paths_under(stage, object_prim_path)
    report: dict[str, Any] = {
        "object_prim_path": object_prim_path,
        "requested_kinematic_enabled": bool(enabled),
        "rigid_body_paths": body_paths,
        "rigid_bodies": [],
        "applied": False,
    }
    for body_path in body_paths:
        body_prim = stage.GetPrimAtPath(body_path)
        body_report: dict[str, Any] = {"prim_path": body_path}
        try:
            rigid_body = UsdPhysics.RigidBodyAPI(body_prim)
            kinematic_attr = rigid_body.GetKinematicEnabledAttr()
            body_report["kinematic_attr_was_valid"] = bool(kinematic_attr and kinematic_attr.IsValid())
            if not body_report["kinematic_attr_was_valid"]:
                kinematic_attr = rigid_body.CreateKinematicEnabledAttr()
            before = _read_bool_attr_value(kinematic_attr, False)
            body_report["kinematic_enabled_before"] = before
            kinematic_attr.Set(bool(enabled))
            body_report["kinematic_enabled_after"] = _read_bool_attr_value(kinematic_attr, bool(enabled))
            disable_gravity_attr = rigid_body.GetDisableGravityAttr()
            body_report["disable_gravity_before"] = _read_bool_attr_value(disable_gravity_attr, None)
            body_report["success"] = True
            report["applied"] = True
        except Exception as exc:
            body_report["success"] = False
            body_report["error"] = str(exc)
        report["rigid_bodies"].append(body_report)
    if not body_paths:
        report["warning"] = "no_rigid_body_api_under_object_root"
    return report


def _restore_object_kinematic_enabled(stage, freeze_report: dict[str, Any]) -> dict[str, Any]:
    from pxr import UsdPhysics

    report: dict[str, Any] = {
        "object_prim_path": freeze_report.get("object_prim_path"),
        "rigid_bodies": [],
        "applied": False,
    }
    for body in freeze_report.get("rigid_bodies", []):
        body_path = str(body.get("prim_path") or "")
        body_report: dict[str, Any] = {"prim_path": body_path}
        if not body_path:
            body_report["success"] = False
            body_report["error"] = "missing_prim_path"
            report["rigid_bodies"].append(body_report)
            continue
        try:
            body_prim = stage.GetPrimAtPath(body_path)
            rigid_body = UsdPhysics.RigidBodyAPI(body_prim)
            kinematic_attr = rigid_body.GetKinematicEnabledAttr()
            if not kinematic_attr or not kinematic_attr.IsValid():
                kinematic_attr = rigid_body.CreateKinematicEnabledAttr()
            restore_value = bool(body.get("kinematic_enabled_before", False))
            kinematic_attr.Set(restore_value)
            body_report["kinematic_enabled_restored_to"] = restore_value
            body_report["kinematic_enabled_after_restore"] = _read_bool_attr_value(kinematic_attr, restore_value)
            body_report["success"] = True
            report["applied"] = True
        except Exception as exc:
            body_report["success"] = False
            body_report["error"] = str(exc)
        report["rigid_bodies"].append(body_report)
    return report


def _zero_object_velocity_best_effort(stage, object_prim_path: str) -> dict[str, Any]:
    import numpy as np
    from pxr import UsdPhysics

    body_paths = _rigid_body_paths_under(stage, object_prim_path)
    report: dict[str, Any] = {
        "object_prim_path": object_prim_path,
        "rigid_body_paths": body_paths,
        "rigid_bodies": [],
        "success": False,
    }
    for body_path in body_paths:
        body_report: dict[str, Any] = {"prim_path": body_path}
        body_prim = stage.GetPrimAtPath(body_path)
        try:
            rigid_body = UsdPhysics.RigidBodyAPI(body_prim)
            velocity_attr, _ = _rigid_body_attr_or_create(
                rigid_body,
                "GetVelocityAttr",
                "CreateVelocityAttr",
            )
            angular_velocity_attr, _ = _rigid_body_attr_or_create(
                rigid_body,
                "GetAngularVelocityAttr",
                "CreateAngularVelocityAttr",
            )
            try:
                body_report["usd_velocity_before"] = list(velocity_attr.Get() or [])
                body_report["usd_angular_velocity_before"] = list(angular_velocity_attr.Get() or [])
            except Exception:
                pass
            velocity_attr.Set((0.0, 0.0, 0.0))
            angular_velocity_attr.Set((0.0, 0.0, 0.0))
            body_report["usd_velocity_zeroed"] = True
        except Exception as exc:
            body_report["usd_velocity_zero_error"] = str(exc)
        try:
            from isaacsim.core.prims import SingleRigidPrim

            rigid_prim = SingleRigidPrim(
                prim_path=body_path,
                name="go2_x5_object_velocity_zero_" + body_path.strip("/").replace("/", "_"),
                reset_xform_properties=False,
            )
            try:
                rigid_prim.initialize()
                body_report["live_initialized"] = True
            except Exception as exc:
                body_report["live_initialized"] = False
                body_report["live_initialize_error"] = str(exc)
            rigid_prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rigid_prim.set_angular_velocity(np.zeros(3, dtype=np.float32))
            body_report["live_velocity_zeroed"] = True
        except Exception as exc:
            body_report["live_velocity_zero_error"] = str(exc)
        body_report["success"] = bool(
            body_report.get("usd_velocity_zeroed") or body_report.get("live_velocity_zeroed")
        )
        report["success"] = bool(report["success"] or body_report["success"])
        report["rigid_bodies"].append(body_report)
    if not body_paths:
        report["warning"] = "no_rigid_body_api_under_object_root"
    return report


def _matrix_translation_error_m(a, b) -> float:
    import numpy as np

    return float(np.linalg.norm(np.asarray(a, dtype=float)[:3, 3] - np.asarray(b, dtype=float)[:3, 3]))


def _named_pose_entry(T_base_pose, T_world_pose) -> dict[str, Any]:
    from scripts.math.SE3 import matrix_to_pose

    base_position, base_quaternion = matrix_to_pose(T_base_pose)
    world_position, world_quaternion = matrix_to_pose(T_world_pose)
    return {
        "frame": "arm_base_link",
        "position_xyz": [float(value) for value in base_position],
        "quaternion_wxyz": [float(value) for value in base_quaternion],
        "world": {
            "frame": "world",
            "position_xyz": [float(value) for value in world_position],
            "quaternion_wxyz": [float(value) for value in world_quaternion],
        },
    }


def _place_target_workspace_diagnostics(target_poses: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    for name, entry in target_poses.items():
        position = [float(value) for value in entry["position_xyz"]]
        xy_radius = float(math.hypot(position[0], position[1]))
        radius_3d = float(math.sqrt(sum(value * value for value in position)))
        diagnostics[name] = {
            "position_xyz": position,
            "quaternion_wxyz": [float(value) for value in entry["quaternion_wxyz"]],
            "xy_radius_m": xy_radius,
            "radius_3d_m": radius_3d,
            "z_m": float(position[2]),
        }
        if xy_radius > 0.75:
            warnings.append(f"{name} horizontal radius is large for current base: {xy_radius:.3f}m")
        if radius_3d > 0.95:
            warnings.append(f"{name} 3D radius is large for current base: {radius_3d:.3f}m")
    diagnostics["warnings"] = warnings
    return diagnostics


def _validate_arm_place_target_workspace(target: dict[str, Any], *, task_nav_to_place: str | None) -> None:
    workspace = (target.get("diagnostics") or {}).get("target_workspace_base") or {}
    violations: list[dict[str, Any]] = []
    for name in ("pre_place", "place", "retreat"):
        item = workspace.get(name) or {}
        xy_radius = item.get("xy_radius_m")
        radius_3d = item.get("radius_3d_m")
        if xy_radius is None or radius_3d is None:
            continue
        if float(xy_radius) > ARM_PLACE_MAX_TARGET_XY_RADIUS_M or float(radius_3d) > ARM_PLACE_MAX_TARGET_RADIUS_3D_M:
            violations.append(
                {
                    "target": name,
                    "position_xyz_base": item.get("position_xyz"),
                    "xy_radius_m": float(xy_radius),
                    "radius_3d_m": float(radius_3d),
                }
            )
    if not violations:
        return
    raise DemoFailure(
        "place_target_unreachable_from_current_base",
        (
            "task_nav_to_place.place.place_pose_world is outside the current arm workspace; "
            "arm-place mode does not run or restore nav_to_place."
        ),
        {
            "task_nav_to_place": _required_path_text(task_nav_to_place, "task-nav-to-place"),
            "place_pose_world": (target.get("source") or {}).get("place_pose_world"),
            "target_workspace_base": workspace,
            "workspace_limits": {
                "max_xy_radius_m": ARM_PLACE_MAX_TARGET_XY_RADIUS_M,
                "max_radius_3d_m": ARM_PLACE_MAX_TARGET_RADIUS_3D_M,
            },
            "violations": violations,
            "object_teleported": False,
            "arm_place_executed": False,
            "physical_place_continuity": True,
            "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        },
    )


def _gripper_config_from_pick_target(pick_report: dict[str, Any]) -> dict[str, Any]:
    raw_target_path = (pick_report.get("task_spec") or {}).get("target_json")
    target_path = Path(raw_target_path).expanduser() if raw_target_path else None
    if target_path is not None and target_path.exists() and target_path.is_file():
        try:
            target = json.loads(target_path.read_text(encoding="utf-8"))
            gripper = target.get("gripper") or {}
            return {
                "open_m": float(gripper.get("open_m", 0.043)),
                "close_m": float(gripper.get("close_m", 0.0)),
                "joint_names": list(gripper.get("joint_names", ["arm_joint7", "arm_joint8"])),
                "source": str(target_path.resolve()),
            }
        except Exception as exc:
            return {
                "open_m": 0.043,
                "close_m": 0.0,
                "joint_names": ["arm_joint7", "arm_joint8"],
                "source": str(target_path),
                "warning": f"failed_to_read_pick_target_gripper:{exc}",
            }
    return {
        "open_m": float(os.environ.get("GO2_X5_GRIPPER_OPEN_M", "0.043")),
        "close_m": float(os.environ.get("GO2_X5_GRIPPER_CLOSE_M", "0.0")),
        "joint_names": ["arm_joint7", "arm_joint8"],
        "source": "defaults",
    }


def _make_arm_place_target(
    state: dict[str, Any],
    raw_task_place: dict[str, Any],
    pick_report: dict[str, Any],
    object_prim_path: str,
    object_bbox_before_place: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    place = dict(raw_task_place.get("place") or {})
    place_pose = place.get("place_pose_world")
    if not place.get("enabled", False) or not isinstance(place_pose, dict):
        raise DemoFailure("missing_place_pose_world", "task_nav_to_place.place.place_pose_world is missing or disabled.")
    missing = [key for key in ("x", "y", "z") if key not in place_pose]
    if missing:
        raise DemoFailure("missing_place_pose_world", f"place_pose_world is missing keys: {missing}")

    center = list((object_bbox_before_place or {}).get("center_xyz") or [])
    if len(center) != 3:
        raise DemoFailure("arm_place_failed", "object bbox is unavailable before arm-place target generation.")

    T_world_base = _matrix_from_pose_dict(state["poses"]["world_base"])
    T_world_tcp = _matrix_from_pose_dict(state["poses"]["world_tcp"])
    tcp_position_world = T_world_tcp[:3, 3].copy()
    object_center_world = np.asarray(center, dtype=float)
    tcp_to_object_center_world = object_center_world - tcp_position_world

    place_center_world = np.asarray(
        [float(place_pose["x"]), float(place_pose["y"]), float(place_pose["z"])],
        dtype=float,
    )
    release_height = float(place.get("release_height", 0.04))
    retreat_height = float(place.get("retreat_height", 0.12))

    target_object_centers = {
        "pre_place": place_center_world + np.array([0.0, 0.0, release_height], dtype=float),
        "place": place_center_world,
        "retreat": place_center_world + np.array([0.0, 0.0, retreat_height], dtype=float),
    }

    target_poses: dict[str, Any] = {}
    target_world_matrices: dict[str, Any] = {}
    for name, object_center_target in target_object_centers.items():
        T_world_target_tcp = np.asarray(T_world_tcp, dtype=float).copy()
        T_world_target_tcp[:3, 3] = object_center_target - tcp_to_object_center_world
        T_base_target_tcp = np.linalg.inv(T_world_base) @ T_world_target_tcp
        target_world_matrices[name] = T_world_target_tcp.tolist()
        target_poses[name] = _named_pose_entry(T_base_target_tcp, T_world_target_tcp)

    gripper = _gripper_config_from_pick_target(pick_report)
    payload = {
        "schema_version": 1,
        "frame": "arm_base_link",
        "default_target_name": "place",
        "sequence": ["pre_place", "place", "open_gripper", "retreat"],
        "poses": target_poses,
        "gripper": {
            "open_m": float(gripper["open_m"]),
            "close_m": float(gripper["close_m"]),
            "joint_names": list(gripper["joint_names"]),
            "source": gripper.get("source"),
        },
        "source": {
            "type": "single_stage_arm_place_target",
            "mode": "arm_place",
            "object_prim_path": object_prim_path,
            "place_pose_world": {
                "x": float(place_pose["x"]),
                "y": float(place_pose["y"]),
                "z": float(place_pose["z"]),
                "roll": float(place_pose.get("roll", 0.0)),
                "pitch": float(place_pose.get("pitch", 0.0)),
                "yaw": float(place_pose.get("yaw", 0.0)),
            },
            "current_object_bbox_world": object_bbox_before_place,
            "current_tcp_world": state["poses"]["world_tcp"],
            "tcp_to_object_center_world_xyz": [float(value) for value in tcp_to_object_center_world],
            "target_object_centers_world": {
                name: [float(value) for value in center_xyz]
                for name, center_xyz in target_object_centers.items()
            },
            "target_tcp_matrices_world": target_world_matrices,
            "release_height_m": release_height,
            "retreat_height_m": retreat_height,
            "orientation_rule": "preserve_current_tcp_orientation_after_pick",
        },
        "diagnostics": {
            "target_workspace_base": _place_target_workspace_diagnostics(target_poses),
        },
    }
    return payload


async def _export_arm_place_state(state_json: Path, object_prim_path: str | None = None) -> dict[str, Any]:
    from source.manipulation import GraspPipeline, GraspTask

    if object_prim_path:
        try:
            import omni.usd

            selection = omni.usd.get_context().get_selection()
            selection.set_selected_prim_paths([object_prim_path], True)
        except Exception as exc:
            print(f"[arm-place] warning: failed to select object for collision export exclusion: {exc}")

    pipeline = GraspPipeline()
    task = GraspTask(object_prim_path=None, state_json=str(state_json))
    return await pipeline.export_state(task)


def _plan_arm_place_external(
    state_json: Path,
    target_json: Path,
    plan_json: Path,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    import subprocess

    script_plan = PROJECT_ROOT / "scripts/curobo/03_plan_grasp_trajectory.py"
    plan_json.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GO2_X5_WORKSPACE": str(PROJECT_ROOT),
            "GO2_X5_CUROBO_SOURCE_ROOT": os.environ.get("GO2_X5_CUROBO_SOURCE_ROOT", "/home/light/workspace/curobo"),
            "GO2_X5_CUROBO_TASK_MODE": "place",
            "GO2_X5_STATE_JSON": str(state_json),
            "GO2_X5_TARGET_JSON": str(target_json),
            "GO2_X5_PLAN_JSON": str(plan_json),
        }
    )
    curobo_python = os.environ.get("GO2_X5_CUROBO_PYTHON", DEFAULT_CUROBO_PYTHON)
    print(
        "[arm-place] planning with cuRobo:",
        {
            "state_json": str(state_json),
            "target_json": str(target_json),
            "plan_json": str(plan_json),
            "timeout_s": float(timeout_s),
        },
        flush=True,
    )
    try:
        result = subprocess.run(
            [curobo_python, str(script_plan)],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_s)) if timeout_s > 0.0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise DemoFailure(
            "place_target_unreachable_from_current_base",
            f"arm-place cuRobo planning timed out after {float(timeout_s):.1f}s",
            {
                "state_json": str(state_json),
                "target_json": str(target_json),
                "plan_json": str(plan_json),
                "planner_timeout_s": float(timeout_s),
                "stdout_tail": str(stdout)[-4000:],
                "stderr_tail": str(stderr)[-4000:],
            },
        ) from exc
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0 or not plan_json.exists():
        raise DemoFailure(
            "place_target_unreachable_from_current_base",
            f"arm-place cuRobo planning failed with return code {result.returncode}",
            {
                "state_json": str(state_json),
                "target_json": str(target_json),
                "plan_json": str(plan_json),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
        )
    return _read_json(plan_json, missing_reason="place_plan_missing")


def _arm_place_failure_reason(summary: dict[str, Any]) -> str:
    if summary.get("planning_failure_reason"):
        return "place_target_unreachable_from_current_base"
    if summary.get("tracking_failure_reason"):
        return "arm_place_execution_failed"
    if not summary.get("place_success", False):
        return "object_out_of_place"
    return ""


async def _execute_arm_place_plan(
    world,
    robot,
    plan: dict[str, Any],
    *,
    state_json: Path,
    target_json: Path,
    plan_json: Path,
    execution_json: Path,
    object_prim_path: str,
    place: dict[str, Any],
    settle_steps: int,
) -> dict[str, Any]:
    import omni.usd
    import numpy as np
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    exec_module = _load_script_module(
        "go2_x5_execute_arm_place",
        PROJECT_ROOT / "scripts/isaac/04_execute_grasp_sequence.py",
    )
    exec_module.STATE_JSON = state_json
    exec_module.TARGET_JSON = target_json
    exec_module.GRASP_PLAN_JSON = plan_json
    exec_module.OUTPUT_JSON = execution_json
    exec_module.STRICT_POST_MOTION_WAIT_SEGMENTS = set(exec_module.STRICT_POST_MOTION_WAIT_SEGMENTS) | {
        "move_to_pre_place",
        "approach_to_place",
        "retreat_place",
    }

    stage = omni.usd.get_context().get_stage()
    object_bbox_before = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    dof_names = exec_module.get_dof_names(robot)
    arm_indices = exec_module.get_joint_indices(dof_names, plan["joint_names"])
    gripper_joint_names: list[str] = []
    for segment in plan.get("segments", []):
        if segment.get("type") == "gripper":
            gripper_joint_names = list(segment.get("joint_names", []))
            break
    if not gripper_joint_names:
        raise DemoFailure("arm_place_failed", "arm-place plan has no gripper open segment.")
    gripper_indices = exec_module.get_joint_indices(dof_names, gripper_joint_names)

    logs: list[dict[str, Any]] = []
    last_motion_q_final = None
    object_bbox_at_place_pose = None
    object_bbox_after_open = None
    tracking_failure_reason = ""
    opened_gripper = False

    for segment in plan.get("segments", []):
        if segment.get("type") == "motion":
            motion_log = await exec_module.execute_motion_segment(world, robot, arm_indices, segment)
            logs.append(motion_log)
            last_motion_q_final = np.asarray(segment["trajectory"]["q"][-1], dtype=float)
            if segment.get("name") == "approach_to_place":
                object_bbox_at_place_pose = (
                    pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
                )
            if not motion_log.get("motion_converged", True):
                tracking_failure_reason = f"{segment.get('name')} did not converge before continuing."
                break
        elif segment.get("type") == "gripper":
            if segment.get("name") != "open_gripper":
                raise DemoFailure("arm_place_failed", f"unexpected arm-place gripper segment: {segment.get('name')}")
            gripper_log = await exec_module.execute_gripper_segment(
                world,
                robot,
                gripper_indices,
                segment,
                arm_indices=arm_indices if last_motion_q_final is not None else None,
                q_arm_hold=last_motion_q_final,
            )
            logs.append(gripper_log)
            opened_gripper = True
            object_bbox_after_open = (
                pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
            )
        else:
            raise DemoFailure("arm_place_failed", f"unknown arm-place segment type: {segment.get('type')}")

    if not tracking_failure_reason:
        await exec_module.hold_final(world, robot, plan.get("segments", []), arm_indices)

    _settle_world(world, settle_steps)
    object_bbox_after = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    verification = _verify_mvp_place(
        dict(place["place_pose_world"]),
        object_bbox_after,
        xy_tolerance=float(place.get("place_xy_tolerance", 0.10)),
        z_tolerance=float(place.get("place_z_tolerance", 0.08)),
    )
    summary = exec_module.summarize_logs(logs)
    summary.update(
        {
            "arm_place_executed": opened_gripper and not bool(tracking_failure_reason),
            "opened_gripper": opened_gripper,
            "tracking_failure_reason": tracking_failure_reason,
            "place_success": bool(verification.get("success", False)) and not bool(tracking_failure_reason),
            "object_teleported": False,
            "physical_place_continuity": True,
            "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        }
    )
    result = {
        "schema_version": 1,
        "script": "scripts/isaac/07_run_pick_put_demo_from_nav_results.py",
        "source_plan": str(plan_json),
        "source_target": str(target_json),
        "source_state": str(state_json),
        "object_prim_path": object_prim_path,
        "arm_joint_names": plan["joint_names"],
        "gripper_joint_names": gripper_joint_names,
        "arm_joint_indices": dict(zip(plan["joint_names"], arm_indices)),
        "gripper_joint_indices": dict(zip(gripper_joint_names, gripper_indices)),
        "object_bbox_before": object_bbox_before,
        "object_bbox_at_place_pose": object_bbox_at_place_pose,
        "object_bbox_after_open_gripper": object_bbox_after_open,
        "object_bbox_after": object_bbox_after,
        "verification": verification,
        "execution_logs": logs,
        "summary": summary,
    }
    _write_json(execution_json, result)
    if tracking_failure_reason:
        raise DemoFailure("arm_place_execution_failed", tracking_failure_reason, result)
    if not verification.get("success", False):
        raise DemoFailure("object_out_of_place", "arm-place object did not verify near place_pose_world.", result)
    return result


async def _restore_place_base_with_kinematic_object_carry(
    world,
    robot,
    nav_place_result: dict[str, Any],
    object_prim_path: str,
    *,
    source_nav_result_path: str | None,
    settle_steps: int,
) -> dict[str, Any]:
    import numpy as np
    import omni.kit.app
    import omni.usd
    from scripts.isaac import run_pick_from_nav_result as pick_handoff
    from source.navigation.adapters.frame_utils import yaw_to_quat_wxyz

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise DemoFailure("stage_not_ready", "No USD stage is open for arm-place base restore.")
    if not nav_place_result:
        raise DemoFailure("missing_nav_place_result", "--nav-place-result is required for arm-place base handoff.")
    nav_pose = nav_place_result["final_base_pose_world"]

    root_before = pick_handoff._robot_root_report(robot)
    root_z = float(root_before["position_xyz"][2])
    T_tcp_before, tcp_before_report = _tcp_world_matrix(stage)
    T_object_before, object_before_report = _dynamic_object_world_matrix(stage, object_prim_path)
    T_tcp_object = np.linalg.inv(T_tcp_before) @ T_object_before
    object_bbox_before = pick_handoff._compute_world_bbox(stage, object_prim_path)

    old_root_matrix = _planar_pose_matrix(
        float(root_before["position_xyz"][0]),
        float(root_before["position_xyz"][1]),
        root_z,
        float(root_before["yaw"]),
    )
    new_root_matrix = _planar_pose_matrix(
        float(nav_pose["x"]),
        float(nav_pose["y"]),
        root_z,
        float(nav_pose["yaw"]),
    )
    root_delta_world = new_root_matrix @ np.linalg.inv(old_root_matrix)
    fallback_tcp_after_root_restore = root_delta_world @ T_tcp_before

    freeze_report = _set_object_kinematic_enabled(stage, object_prim_path, True)
    velocity_zero_before_restore = _zero_object_velocity_best_effort(stage, object_prim_path)

    robot.set_world_pose(
        position=np.asarray([float(nav_pose["x"]), float(nav_pose["y"]), root_z], dtype=float),
        orientation=np.asarray(yaw_to_quat_wxyz(float(nav_pose["yaw"])), dtype=float),
    )
    robot.set_linear_velocity(np.zeros(3, dtype=float))
    robot.set_angular_velocity(np.zeros(3, dtype=float))

    def _current_tcp_or_fallback(label: str) -> tuple[Any, dict[str, Any]]:
        try:
            matrix, tcp_report = _tcp_world_matrix(stage)
            fallback_delta_m = _matrix_translation_error_m(matrix, fallback_tcp_after_root_restore)
            if fallback_delta_m > 0.25:
                fallback_report = _matrix_pose_report(fallback_tcp_after_root_restore, frame="world")
                fallback_report.update(
                    {
                        "tcp_prim_path": tcp_report.get("tcp_prim_path"),
                        "source": "root_delta_fallback",
                        "fallback_used": True,
                        "fallback_reason": "usd_tcp_pose_far_from_root_delta_prediction",
                        "usd_tcp_pose": tcp_report,
                        "usd_vs_root_delta_translation_error_m": fallback_delta_m,
                        "label": label,
                    }
                )
                return fallback_tcp_after_root_restore, fallback_report
            tcp_report["fallback_used"] = False
            tcp_report["usd_vs_root_delta_translation_error_m"] = fallback_delta_m
            tcp_report["label"] = label
            return matrix, tcp_report
        except Exception as exc:
            tcp_report = _matrix_pose_report(fallback_tcp_after_root_restore, frame="world")
            tcp_report.update(
                {
                    "tcp_prim_path": None,
                    "source": "root_delta_fallback",
                    "fallback_used": True,
                    "label": label,
                    "read_error": str(exc),
                }
            )
            return fallback_tcp_after_root_restore, tcp_report

    T_tcp_initial, tcp_initial_report = _current_tcp_or_fallback("after_robot_root_set_before_settle")
    T_object_initial_target = T_tcp_initial @ T_tcp_object
    initial_object_apply = _set_dynamic_object_world_matrix(
        stage,
        object_prim_path,
        T_object_initial_target,
        reset_xform_stack=False,
    )
    initial_velocity_zero = _zero_object_velocity_best_effort(stage, object_prim_path)

    clamp_samples: list[dict[str, Any]] = []
    tcp_sample_reports: list[dict[str, Any]] = [tcp_initial_report]
    max_pre_clamp_error_m = 0.0
    pre_clamp_error_count = 0
    max_live_apply_failed = False
    requested_settle_steps = max(0, int(settle_steps))
    sample_steps = {0, 1, 2, max(0, requested_settle_steps - 1)}
    app = omni.kit.app.get_app()
    for step_index in range(requested_settle_steps):
        T_tcp_current, tcp_current_report = _current_tcp_or_fallback(f"settle_step_{step_index}")
        T_object_target = T_tcp_current @ T_tcp_object
        try:
            T_object_pre_clamp, object_pre_report = _dynamic_object_world_matrix(stage, object_prim_path)
            pre_clamp_error_m = _matrix_translation_error_m(T_object_pre_clamp, T_object_target)
            max_pre_clamp_error_m = max(max_pre_clamp_error_m, pre_clamp_error_m)
            pre_clamp_error_count += 1
        except Exception as exc:
            object_pre_report = {"error": str(exc)}
            pre_clamp_error_m = None
        object_apply = _set_dynamic_object_world_matrix(
            stage,
            object_prim_path,
            T_object_target,
            reset_xform_stack=False,
        )
        max_live_apply_failed = bool(
            max_live_apply_failed
            or not (object_apply.get("live_rigid_pose_apply") or {}).get("success", False)
        )
        velocity_zero = _zero_object_velocity_best_effort(stage, object_prim_path)
        if step_index in sample_steps:
            clamp_samples.append(
                {
                    "step_index": step_index,
                    "tcp": tcp_current_report,
                    "object_pre_clamp": object_pre_report,
                    "pre_clamp_error_m": pre_clamp_error_m,
                    "object_pose_apply": object_apply,
                    "velocity_zero": velocity_zero,
                }
            )
        elif tcp_current_report.get("fallback_used"):
            tcp_sample_reports.append(tcp_current_report)
        world.step(render=True)
        await app.next_update_async()

    T_tcp_final, tcp_final_report = _current_tcp_or_fallback("after_settle_before_final_clamp")
    T_object_final_target = T_tcp_final @ T_tcp_object
    try:
        T_object_pre_final, object_pre_final_report = _dynamic_object_world_matrix(stage, object_prim_path)
        pre_final_error_m = _matrix_translation_error_m(T_object_pre_final, T_object_final_target)
        max_pre_clamp_error_m = max(max_pre_clamp_error_m, pre_final_error_m)
        pre_clamp_error_count += 1
    except Exception as exc:
        object_pre_final_report = {"error": str(exc)}
        pre_final_error_m = None
    final_object_apply = _set_dynamic_object_world_matrix(
        stage,
        object_prim_path,
        T_object_final_target,
        reset_xform_stack=False,
    )
    final_velocity_zero = _zero_object_velocity_best_effort(stage, object_prim_path)

    try:
        T_object_after_clamp, object_after_clamp_report = _dynamic_object_world_matrix(stage, object_prim_path)
        T_tcp_object_after = np.linalg.inv(T_tcp_final) @ T_object_after_clamp
        final_relative_error_m = _matrix_translation_error_m(T_tcp_object_after, T_tcp_object)
    except Exception as exc:
        T_object_after_clamp = T_object_final_target
        object_after_clamp_report = {"error": str(exc), "source": "final_target_fallback"}
        final_relative_error_m = None

    object_bbox_after_clamp_usd = pick_handoff._compute_world_bbox(stage, object_prim_path)
    expected_bbox_after_clamp = _transform_bbox_world(
        object_bbox_before,
        np.asarray(T_object_after_clamp, dtype=float) @ np.linalg.inv(T_object_before),
    )

    restore_kinematic_report = _restore_object_kinematic_enabled(stage, freeze_report)
    object_dynamic_restored = all(
        not bool(item.get("kinematic_enabled_after_restore", item.get("kinematic_enabled_restored_to", True)))
        for item in restore_kinematic_report.get("rigid_bodies", [])
    )
    try:
        velocity_reset_after_dynamic_restore = pick_handoff.reset_object_physics_state(
            object_prim_path,
            zero_linear_velocity=True,
            zero_angular_velocity=True,
            wake=True,
        )
    except Exception as exc:
        velocity_reset_after_dynamic_restore = {
            "applied": False,
            "error": str(exc),
            "warning": "reset_object_physics_state_failed_after_dynamic_restore",
        }

    root_after_report_fallback = False
    try:
        root_after = pick_handoff._robot_root_report(robot)
    except Exception as exc:
        root_after_report_fallback = True
        root_after = _nav_root_report_from_pose(nav_pose, root_z, read_error=str(exc))

    drop_threshold_m = 0.04
    transform_preserve_threshold_m = 0.02
    object_dropped = bool(
        max_pre_clamp_error_m > drop_threshold_m
        or (final_relative_error_m is not None and final_relative_error_m > drop_threshold_m)
    )
    tcp_transform_preserved = bool(
        final_relative_error_m is not None and final_relative_error_m <= transform_preserve_threshold_m
    )
    live_success_final = bool((final_object_apply.get("live_rigid_pose_apply") or {}).get("success", False))
    report = {
        "success": True,
        "base_transfer_mode": "restore_nav_place_result_with_kinematic_tcp_relative_object_carry",
        "base_restore_source": (
            "nav_place_result.final_base_pose_world"
            if source_nav_result_path
            else "task_nav_to_place.place.base_goal"
        ),
        "synthetic_nav_place_result_from_task_base_goal": bool(
            nav_place_result.get("synthetic_from_task_place_base_goal", False)
        ),
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "physical_carry_nav": False,
        "stable_carry_implemented": False,
        "fixed_joint_carry_implemented": False,
        "object_freeze_before_base_restore": True,
        "object_freeze_backend": "UsdPhysics.RigidBodyAPI.kinematicEnabled+pose_clamp_each_step",
        "object_pose_synced_during_base_restore": True,
        "object_carried_relative_to": "tcp",
        "object_dropped_during_base_restore": object_dropped,
        "carried_object_clamped_after_base_restore": True,
        "object_dynamic_restored_before_arm_place": bool(object_dynamic_restored),
        "object_kinematic_until_place": bool(not object_dynamic_restored),
        "object_teleported_to_place_pose": False,
        "object_carried_with_base_teleport": True,
        "source_nav_result": _required_path_text(source_nav_result_path, "nav-place-result"),
        "requested_settle_steps": requested_settle_steps,
        "root_before": root_before,
        "root_after": root_after,
        "root_after_report_fallback": root_after_report_fallback,
        "object_prim_path": object_prim_path,
        "object_bbox_before_base_restore": object_bbox_before,
        "object_bbox_after_base_restore": expected_bbox_after_clamp or object_bbox_after_clamp_usd,
        "object_bbox_after_base_restore_usd": object_bbox_after_clamp_usd,
        "object_bbox_after_carry_clamp": expected_bbox_after_clamp or object_bbox_after_clamp_usd,
        "object_bbox_after_carry_clamp_usd": object_bbox_after_clamp_usd,
        "object_bbox_after_base_restore_source": (
            "live_body_transform_of_pre_restore_bbox" if expected_bbox_after_clamp else "usd_bbox"
        ),
        "tcp_pose_before_base_restore": tcp_before_report,
        "tcp_pose_after_base_restore": tcp_final_report,
        "tcp_pose_initial_after_root_set": tcp_initial_report,
        "tcp_pose_fallback_reports": tcp_sample_reports,
        "tcp_to_object_transform_before": _matrix_pose_report(T_tcp_object, frame="tcp"),
        "tcp_to_object_transform_preserved": tcp_transform_preserved,
        "max_object_tcp_relative_error_m": (
            float(max_pre_clamp_error_m) if pre_clamp_error_count > 0 else None
        ),
        "final_object_tcp_relative_error_m": final_relative_error_m,
        "pre_final_clamp_error_m": pre_final_error_m,
        "object_pose_apply_initial_base_restore": initial_object_apply,
        "object_pose_apply_after_base_restore": final_object_apply,
        "object_matrix_before_base_restore": object_before_report,
        "object_matrix_after_carry_clamp": object_after_clamp_report,
        "object_freeze_report": freeze_report,
        "object_kinematic_restore_report": restore_kinematic_report,
        "object_velocity_zero_before_base_restore": velocity_zero_before_restore,
        "object_velocity_zero_initial_after_root_set": initial_velocity_zero,
        "object_velocity_zero_after_carry_clamp": final_velocity_zero,
        "object_velocity_reset_after_dynamic_restore": velocity_reset_after_dynamic_restore,
        "pose_clamp_samples": clamp_samples,
        "live_rigid_body_pose_apply_success": live_success_final and not max_live_apply_failed,
        "root_carry_transform": {
            "old_root_xyyaw": [
                float(root_before["position_xyz"][0]),
                float(root_before["position_xyz"][1]),
                float(root_before["yaw"]),
            ],
            "new_root_xyyaw": [
                float(nav_pose["x"]),
                float(nav_pose["y"]),
                float(nav_pose["yaw"]),
            ],
            "delta_matrix_world": np.asarray(root_delta_world, dtype=float).tolist(),
        },
        "replay_nav_to_place_executed": False,
        "replay_nav_to_place_with_carried_object": False,
    }
    print(
        "[arm-place] restored place base with TCP-relative object carry:",
        {
            "object": object_prim_path,
            "base_transfer_mode": report["base_transfer_mode"],
            "object_dropped": report["object_dropped_during_base_restore"],
            "max_tcp_error_m": report["max_object_tcp_relative_error_m"],
            "dynamic_restored": report["object_dynamic_restored_before_arm_place"],
        },
        flush=True,
    )
    return report


async def _restore_place_base_with_object_carry(
    world,
    robot,
    nav_place_result: dict[str, Any],
    object_prim_path: str,
    *,
    source_nav_result_path: str | None,
) -> dict[str, Any]:
    import omni.kit.app
    import omni.usd
    import numpy as np
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise DemoFailure("stage_not_ready", "No USD stage is open for arm-place base restore.")
    if not nav_place_result:
        raise DemoFailure("missing_nav_place_result", "--nav-place-result is required for arm-place base handoff.")

    root_before = pick_handoff._robot_root_report(robot)
    object_bbox_before = pick_handoff._compute_world_bbox(stage, object_prim_path)
    object_matrix_before, object_matrix_before_report = _dynamic_object_world_matrix(stage, object_prim_path)
    old_root_matrix = _planar_pose_matrix(
        float(root_before["position_xyz"][0]),
        float(root_before["position_xyz"][1]),
        float(root_before["position_xyz"][2]),
        float(root_before["yaw"]),
    )
    nav_pose = nav_place_result["final_base_pose_world"]
    new_root_matrix = _planar_pose_matrix(
        float(nav_pose["x"]),
        float(nav_pose["y"]),
        float(root_before["position_xyz"][2]),
        float(nav_pose["yaw"]),
    )
    carry_delta_world = new_root_matrix @ np.linalg.inv(old_root_matrix)
    target_object_matrix = carry_delta_world @ object_matrix_before
    expected_object_bbox_after = _transform_bbox_world(object_bbox_before, carry_delta_world)

    restore_report = await pick_handoff._restore_and_settle(world, robot, nav_place_result)
    object_apply = _set_dynamic_object_world_matrix(
        stage,
        object_prim_path,
        target_object_matrix,
        reset_xform_stack=False,
    )
    live_apply = object_apply.get("live_rigid_pose_apply", {})
    if live_apply.get("attempted") and not live_apply.get("success", False):
        raise DemoFailure(
            "object_carry_failed",
            f"failed to apply live rigid body pose for carried object: {live_apply.get('error')}",
            object_apply,
        )
    velocity_reset = pick_handoff.reset_object_physics_state(
        object_prim_path,
        zero_linear_velocity=True,
        zero_angular_velocity=True,
        wake=True,
    )
    for _ in range(3):
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()
    velocity_reset_after_short_step = pick_handoff.reset_object_physics_state(
        object_prim_path,
        zero_linear_velocity=True,
        zero_angular_velocity=True,
        wake=True,
    )
    object_bbox_after_usd = pick_handoff._compute_world_bbox(stage, object_prim_path)
    object_bbox_after = expected_object_bbox_after or object_bbox_after_usd
    root_after_report_fallback = False
    try:
        root_after = pick_handoff._robot_root_report(robot)
    except Exception as exc:
        root_after_report_fallback = True
        root_after = _nav_root_report_from_pose(
            nav_pose,
            float(root_before["position_xyz"][2]),
            read_error=str(exc),
        )
    report = {
        "success": True,
        "base_transfer_mode": "restore_nav_place_result_with_object_carry_teleport",
        "base_restore_source": (
            "nav_place_result.final_base_pose_world"
            if source_nav_result_path
            else "task_nav_to_place.place.base_goal"
        ),
        "synthetic_nav_place_result_from_task_base_goal": bool(
            nav_place_result.get("synthetic_from_task_place_base_goal", False)
        ),
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "physical_carry_nav": False,
        "object_carried_with_base_teleport": True,
        "object_teleported_to_place_pose": False,
        "source_nav_result": _required_path_text(source_nav_result_path, "nav-place-result"),
        "restore": restore_report,
        "root_before": root_before,
        "root_after": root_after,
        "root_after_report_fallback": root_after_report_fallback,
        "object_prim_path": object_prim_path,
        "object_bbox_before_base_restore": object_bbox_before,
        "object_bbox_after_base_restore": object_bbox_after,
        "object_bbox_after_base_restore_usd": object_bbox_after_usd,
        "object_bbox_after_base_restore_source": (
            "expected_carry_delta" if expected_object_bbox_after else "usd_bbox"
        ),
        "object_matrix_before_base_restore": object_matrix_before_report,
        "object_carry_transform": {
            "old_root_xyyaw": [
                float(root_before["position_xyz"][0]),
                float(root_before["position_xyz"][1]),
                float(root_before["yaw"]),
            ],
            "new_root_xyyaw": [
                float(nav_pose["x"]),
                float(nav_pose["y"]),
                float(nav_pose["yaw"]),
            ],
            "delta_matrix_world": np.asarray(carry_delta_world, dtype=float).tolist(),
        },
        "object_pose_apply_after_base_restore": object_apply,
        "object_velocity_reset_after_base_restore": velocity_reset,
        "object_velocity_reset_after_short_step": velocity_reset_after_short_step,
    }
    print(
        "[arm-place] restored nav_place base and carried object:",
        {
            "object": object_prim_path,
            "old_root": report["object_carry_transform"]["old_root_xyyaw"],
            "new_root": report["object_carry_transform"]["new_root_xyyaw"],
            "object_bbox_after": object_bbox_after,
        },
        flush=True,
    )
    return report


async def _run_arm_place_put(
    world,
    robot,
    raw_task_place: dict[str, Any],
    pick_report: dict[str, Any],
    args: argparse.Namespace,
    restore_place_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import omni.usd
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    place = dict(raw_task_place.get("place") or {})
    place_pose = place.get("place_pose_world")
    if not place.get("enabled", False) or not place_pose:
        raise DemoFailure("missing_place_pose_world", "task_nav_to_place.place.place_pose_world is missing or disabled.")

    object_prim_path = str((raw_task_place.get("pick") or {}).get("object_prim_path") or "")
    if not object_prim_path:
        object_prim_path = str((pick_report.get("task_spec") or {}).get("object_prim_path") or "")
    if not object_prim_path:
        raise DemoFailure("missing_place_pose_world", "object_prim_path is required for arm-place.")

    started_at = time.time()
    output_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_json = output_dir / "arm_place_state.json"
    target_json = output_dir / "arm_place_target.json"
    plan_json = output_dir / "arm_place_plan.json"
    execution_json = output_dir / "arm_place_execution_result.json"

    stage = omni.usd.get_context().get_stage()
    object_bbox_before_place_usd = pick_handoff._compute_world_bbox(stage, object_prim_path) if stage is not None else None
    object_bbox_before_place = (
        (restore_place_report or {}).get("object_bbox_after_base_restore")
        or object_bbox_before_place_usd
    )
    state = await _export_arm_place_state(state_json, object_prim_path)
    target = _make_arm_place_target(
        state,
        raw_task_place,
        pick_report,
        object_prim_path,
        object_bbox_before_place,
    )
    _write_json(target_json, target)
    _validate_arm_place_target_workspace(target, task_nav_to_place=args.task_nav_to_place)
    try:
        plan = _plan_arm_place_external(
            state_json,
            target_json,
            plan_json,
            timeout_s=float(args.arm_place_plan_timeout_s),
        )
    except DemoFailure:
        raise
    except Exception as exc:
        raise DemoFailure(
            "place_target_unreachable_from_current_base",
            str(exc),
            {
                "state_json": str(state_json),
                "target_json": str(target_json),
                "plan_json": str(plan_json),
                "target": target,
            },
        ) from exc

    settle_steps = int(place.get("settle_steps", args.settle_steps))
    execution = await _execute_arm_place_plan(
        world,
        robot,
        plan,
        state_json=state_json,
        target_json=target_json,
        plan_json=plan_json,
        execution_json=execution_json,
        object_prim_path=object_prim_path,
        place=place,
        settle_steps=settle_steps,
    )
    verification = execution.get("verification", {})
    summary = execution.get("summary", {})
    report = {
        "success": bool(summary.get("place_success", False)),
        "failure_reason": _arm_place_failure_reason(summary),
        "put_mode": "arm_place",
        "arm_place_executed": bool(summary.get("arm_place_executed", False)),
        "object_teleported": False,
        "physical_place_continuity": True,
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "execution_backend": "curobo_arm_place_same_stage",
        "physical_put_execution": True,
        "place_target": target,
        "plan_summary": plan.get("summary", plan),
        "execution_summary": summary,
        "execution_result_json": str(execution_json),
        "state_json": str(state_json),
        "target_json": str(target_json),
        "plan_json": str(plan_json),
        "object_bbox_before_place": object_bbox_before_place,
        "object_bbox_before_place_usd": object_bbox_before_place_usd,
        "object_bbox_before_place_source": (
            "restore_place_base.object_bbox_after_base_restore"
            if (restore_place_report or {}).get("object_bbox_after_base_restore")
            else "usd_bbox"
        ),
        "object_bbox_after": execution.get("object_bbox_after"),
        "verification": verification,
        "elapsed_wall_time_s": time.time() - started_at,
    }
    if not report["success"]:
        raise DemoFailure(report["failure_reason"] or "put_failed", "arm-place did not complete successfully.", report)
    return report


async def _run_demo_async(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    from source.data import EpisodeRecorder, load_task

    task_pick_path = _project_path(args.task_nav_to_pick)
    task_place_path = _project_path(args.task_nav_to_place)
    raw_task_pick = _read_json(task_pick_path, missing_reason="missing_task_nav_to_pick")
    raw_task_place = _read_json(task_place_path, missing_reason="missing_task_nav_to_place")
    nav_pick_result = _read_json(args.nav_pick_result, missing_reason="missing_nav_pick_result")
    _validate_nav_result(nav_pick_result, reason="missing_nav_pick_result")
    nav_place_result = None
    if args.put_mode == "mvp-reconstruct" or args.nav_place_result:
        nav_place_result = _read_json(args.nav_place_result, missing_reason="missing_nav_place_result")
        _validate_nav_result(nav_place_result, reason="missing_nav_place_result")
    elif args.put_mode == "arm-place":
        nav_place_result = _nav_place_result_from_task_base_goal(raw_task_place)

    scene_usd = _project_path(args.scene_usd or raw_task_pick["scene_usd"])
    if raw_task_place.get("scene_usd") and _project_path(raw_task_place["scene_usd"]) != scene_usd:
        result.setdefault("warnings", []).append(
            {
                "warning": "task_scene_usd_mismatch",
                "task_nav_to_pick_scene_usd": raw_task_pick.get("scene_usd"),
                "task_nav_to_place_scene_usd": raw_task_place.get("scene_usd"),
                "used_scene_usd": str(scene_usd),
            }
        )
    _apply_grasp_policy_env(args)
    context_path = _write_context(args, raw_task_pick, scene_usd)
    os.environ["GO2_X5_PIPELINE_CONTEXT"] = str(context_path)
    os.environ["GO2_X5_NAV_RESULT"] = str(Path(args.nav_pick_result).expanduser().resolve())
    os.environ["GO2_X5_HANDOFF_REPORT"] = str(Path(args.pick_handoff_report).expanduser().resolve())
    from scripts.isaac import run_pick_from_nav_result as pick_handoff

    result["stages"]["open_stage"] = _open_stage(scene_usd, args.stage_load_updates)
    result["stages"]["open_stage"]["required_prims_active"] = _ensure_required_prims_active(raw_task_pick)
    if args.demo_visuals:
        result["stages"]["open_stage"]["viewport_camera_set"] = _set_viewport_stage_camera(args.viewport_camera_prim)

    task_pick = load_task(task_pick_path)
    world, robot = await pick_handoff._initialize_robot()

    result["stages"]["replay_nav_to_pick"] = {
        "requested": bool(args.replay_nav_to_pick),
        "executed": False,
        "replay_trajectory_path": nav_pick_result.get("replay_trajectory_path"),
        "reason": (
            "not_implemented_first_version_uses_restore_only"
            if args.replay_nav_to_pick
            else "disabled"
        ),
        "replay_before_pick_object_prepare": bool(args.replay_before_pick_object_prepare),
    }
    if args.replay_nav_to_pick:
        result.setdefault("warnings", []).append(
            {
                "warning": "replay_nav_to_pick_not_executed",
                "detail": "First stable arm-place version keeps nav replay disabled and restores the final pick base pose.",
            }
        )

    try:
        result["stages"]["restore_pick_base"] = await pick_handoff._restore_and_settle(world, robot, nav_pick_result)
    except Exception as exc:
        raise DemoFailure("restore_pick_base_failed", str(exc), getattr(exc, "report", None)) from exc
    result["stages"]["restore_pick_base"].update(
        {
            "success": True,
            "base_transfer_mode": "restore_from_nav_result",
            "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        }
    )
    result["stages"]["prepare_object"] = await _prepare_object_for_pick(
        world,
        task_pick,
        {"object_reset_stability_steps": 3, "fail_on_object_reset_drift": False},
        reset_xform_stack=bool(args.reset_object_xform_stack),
    )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    recorder = EpisodeRecorder(dataset_dir, task_pick.task_id, task_pick.episode_id, enabled=True)
    recorder.save_task(raw_task_pick)
    try:
        pick_report = await _run_pick(task_pick, nav_pick_result, args, recorder)
    except Exception as exc:
        pick_report = {
            "success": False,
            "failure_reason": "pick_failed",
            "failure_detail": str(exc),
        }
        _write_json(args.pick_handoff_report, {"schema_version": 1, "mode": MODE, "success": False, "pick": pick_report})
        raise DemoFailure("pick_failed", str(exc), pick_report) from exc
    result["stages"]["pick"] = pick_report
    _write_json(args.pick_handoff_report, {"schema_version": 1, "mode": MODE, "success": pick_report["success"], "pick": pick_report})
    if not pick_report["success"]:
        raise DemoFailure("pick_failed", f"pick failed: {pick_report.get('failure_reason')}", pick_report)

    carry_state = _carry_state(args, task_pick.pick.object_prim_path)
    result["stages"]["carry_state"] = carry_state

    if args.put_mode == "mvp-reconstruct":
        result["stages"]["replay_nav_to_place"] = {
            "requested": bool(args.replay_nav_to_place),
            "executed": False,
            "replay_trajectory_path": (nav_place_result or {}).get("replay_trajectory_path"),
            "reason": "mvp_reconstruct_uses_final_base_restore_only",
        }
        try:
            result["stages"]["restore_place_base"] = await pick_handoff._restore_and_settle(world, robot, nav_place_result)
        except Exception as exc:
            raise DemoFailure("restore_place_base_failed", str(exc), getattr(exc, "report", None)) from exc
        result["stages"]["restore_place_base"].update(
            {
                "success": True,
                "base_transfer_mode": "restore_from_nav_result",
                "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
                "source_nav_result": str(Path(args.nav_place_result).expanduser().resolve()),
            }
        )
    elif args.put_mode == "arm-place":
        result["stages"]["replay_nav_to_place"] = {
            "requested": bool(args.replay_nav_to_place),
            "executed": False,
            "replay_trajectory_path": (nav_place_result or {}).get("replay_trajectory_path"),
            "reason": (
                "not_implemented_first_version_uses_tcp_relative_carried_restore_only"
                if args.replay_nav_to_place
                else "disabled"
            ),
            "replay_nav_place_with_carried_object": bool(args.replay_nav_place_with_carried_object),
        }
        if args.replay_nav_to_place:
            result.setdefault("warnings", []).append(
                {
                    "warning": "replay_nav_to_place_not_executed",
                    "detail": "First stable arm-place version uses TCP-relative carried restore instead of replaying nav_to_place.",
                }
            )
        try:
            result["stages"]["restore_place_base"] = await _restore_place_base_with_kinematic_object_carry(
                world,
                robot,
                nav_place_result,
                task_pick.pick.object_prim_path,
                source_nav_result_path=args.nav_place_result,
                settle_steps=int(args.settle_steps),
            )
        except Exception as exc:
            raise DemoFailure("restore_place_base_failed", str(exc), getattr(exc, "report", None)) from exc
    else:
        result["stages"]["restore_place_base"] = {
            "success": True,
            "skipped": True,
            "reason": f"put_mode={args.put_mode} has no place-base restore implementation",
            "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        }

    try:
        if args.put_mode == "mvp-reconstruct":
            put_report = _run_mvp_put(world, raw_task_place, args)
        elif args.put_mode == "arm-place":
            put_report = await _run_arm_place_put(
                world,
                robot,
                raw_task_place,
                pick_report,
                args,
                restore_place_report=result["stages"].get("restore_place_base"),
            )
        else:
            raise DemoFailure("put_mode_not_implemented", f"put_mode={args.put_mode!r} is not implemented.")
    except DemoFailure as exc:
        _write_json(
            args.put_result,
            {
                "schema_version": 1,
                "mode": MODE,
                "success": False,
                "failure_reason": exc.reason,
                "failure_detail": str(exc),
                **exc.report,
            },
        )
        raise
    result["stages"]["put"] = put_report
    _write_json(args.put_result, {"schema_version": 1, "mode": MODE, **put_report})
    restore_place_stage = result["stages"].get("restore_place_base") or {}
    result.update(
        {
            "success": True,
            "failure_reason": "",
            "failure_detail": "",
            "put_mode": _put_mode_result_value(args.put_mode),
            "arm_place_executed": bool(put_report.get("arm_place_executed", False)),
            "object_teleported": bool(put_report.get("object_teleported", _put_mode_teleports_object(args.put_mode))),
            "physical_put_execution": bool(put_report.get("physical_put_execution", False)),
            "physical_place_continuity": bool(put_report.get("physical_place_continuity", args.put_mode == "arm-place")),
            "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
            "physical_carry_nav": bool(restore_place_stage.get("physical_carry_nav", False)),
            "base_transfer_mode": restore_place_stage.get("base_transfer_mode"),
            "object_freeze_before_base_restore": restore_place_stage.get("object_freeze_before_base_restore"),
            "object_pose_synced_during_base_restore": restore_place_stage.get("object_pose_synced_during_base_restore"),
            "object_carried_relative_to": restore_place_stage.get("object_carried_relative_to"),
            "object_dropped_during_base_restore": restore_place_stage.get("object_dropped_during_base_restore"),
            "carried_object_clamped_after_base_restore": restore_place_stage.get("carried_object_clamped_after_base_restore"),
            "object_dynamic_restored_before_arm_place": restore_place_stage.get("object_dynamic_restored_before_arm_place"),
            "stable_carry_implemented": bool(restore_place_stage.get("stable_carry_implemented", False)),
            "fixed_joint_carry_implemented": bool(restore_place_stage.get("fixed_joint_carry_implemented", False)),
            "updated_at": time.time(),
            "scene_usd": str(scene_usd),
            "pipeline_context": str(context_path),
        }
    )
    return result


def _keep_window_open_until_closed() -> None:
    print("[single-stage] keep-window-open enabled; close the Isaac Sim window to end this process.")
    while simulation_app.is_running():
        simulation_app.update()
        time.sleep(1.0 / 60.0)


def _run() -> int:
    args = args_cli
    if args.object_pose_debug_only:
        try:
            future = _schedule_kit_coroutine(_run_object_pose_debug_async(args))
            report = _drive_future_to_completion(future, timeout_s=args.timeout_s, label="object_pose_debug_only")
            _write_json(args.object_pose_debug_report, report)
            print(f"[object-pose-debug] report={Path(args.object_pose_debug_report).expanduser().resolve()}")
            if args.keep_window_open:
                _keep_window_open_until_closed()
            return 0
        except Exception as exc:
            reason = exc.reason if isinstance(exc, DemoFailure) else "object_pose_debug_failed"
            detail = str(exc)
            report = {
                "schema_version": 1,
                "success": False,
                "failure_reason": str(reason),
                "failure_detail": detail,
                "mode": "object_pose_debug_only",
                "task_nav_to_pick": _required_path_text(args.task_nav_to_pick, "task-nav-to-pick"),
                "scene_usd": args.scene_usd,
                "reset_object_xform_stack": bool(args.reset_object_xform_stack),
                "diagnostic": exc.report if isinstance(exc, DemoFailure) else {},
                "updated_at": time.time(),
            }
            print(f"[object-pose-debug] failed: {reason}: {detail}")
            traceback.print_exc()
            _write_json(args.object_pose_debug_report, report)
            if args.keep_window_open:
                _keep_window_open_until_closed()
            return 1

    _validate_full_demo_args(args)
    result = _base_result(args)
    _write_json(args.result_json, result)
    try:
        future = _schedule_kit_coroutine(_run_demo_async(args, result))
        result = _drive_future_to_completion(future, timeout_s=args.timeout_s, label=MODE)
        _write_json(args.result_json, result)
        print(f"[single-stage] success=True result={Path(args.result_json).expanduser().resolve()}")
        if args.keep_window_open:
            _keep_window_open_until_closed()
        return 0
    except Exception as exc:
        reason = exc.reason if isinstance(exc, DemoFailure) else "single_stage_pick_put_failed"
        report = exc.report if isinstance(exc, DemoFailure) else {}
        detail = str(exc)
        print(f"[single-stage] failed: {reason}: {detail}")
        traceback.print_exc()
        stage_name = None
        if reason in {
            "open_stage_failed",
            "restore_pick_base_failed",
            "object_unstable_before_pick",
            "pick_failed",
            "carry_mode_not_implemented",
            "restore_place_base_failed",
            "missing_place_pose_world",
            "put_failed",
            "put_mode_not_implemented",
            "place_target_unreachable_from_current_base",
            "arm_place_failed",
            "arm_place_execution_failed",
            "object_out_of_place",
        }:
            stage_name_by_reason = {
                "open_stage_failed": "open_stage",
                "restore_pick_base_failed": "restore_pick_base",
                "object_unstable_before_pick": "prepare_object",
                "pick_failed": "pick",
                "carry_mode_not_implemented": "carry_state",
                "restore_place_base_failed": "restore_place_base",
                "missing_place_pose_world": "put",
                "put_failed": "put",
                "put_mode_not_implemented": "put",
                "place_target_unreachable_from_current_base": "put",
                "arm_place_failed": "put",
                "arm_place_execution_failed": "put",
                "object_out_of_place": "put",
            }
            stage_name = stage_name_by_reason.get(reason)
        _write_failure_result(args, result, str(reason), detail, stage_name=stage_name, report=report)
        if args.keep_window_open:
            _keep_window_open_until_closed()
        return 1
    finally:
        simulation_app.close()


def main() -> int:
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
