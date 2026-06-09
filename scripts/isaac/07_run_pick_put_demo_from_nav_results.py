#!/usr/bin/env python3
"""Single-stage pick + MVP put demo from precomputed navigation results.

This runner intentionally does not run navigation control. It opens one Isaac
Sim stage, restores the base from nav result JSON files, runs the existing
pick pipeline, records a logical carry state, then performs an MVP object
reconstruction at the place pose in the same stage.
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
    parser.add_argument("--put-mode", choices=("mvp-reconstruct", "release-only"), default="mvp-reconstruct")
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


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "success": False,
        "failure_reason": "not_completed",
        "failure_detail": "",
        "mode": MODE,
        "stage_process": STAGE_PROCESS,
        "physical_nav_continuity": PHYSICAL_NAV_CONTINUITY,
        "carry_mode": args.carry_mode,
        "put_mode": args.put_mode,
        "task_nav_to_pick": _required_path_text(args.task_nav_to_pick, "task-nav-to-pick"),
        "task_nav_to_place": _required_path_text(args.task_nav_to_place, "task-nav-to-place"),
        "nav_pick_result": _required_path_text(args.nav_pick_result, "nav-pick-result"),
        "nav_place_result": _required_path_text(args.nav_place_result, "nav-place-result"),
        "dataset_dir": _required_path_text(args.dataset_dir, "dataset-dir"),
        "metadata": {
            "complete_physical_nav_carry": False,
            "nav_execution_in_this_process": False,
            "base_transfer_mode": "restore_from_nav_result",
            "notes": [
                "This demo opens one Isaac Sim app/stage for pick and MVP put.",
                "Navigation is precomputed and only used as base pose JSON handoff.",
                "Logical carry is metadata only and is not a real physical carry constraint.",
                "MVP put reconstructs the object at place_pose_world; it is not a full arm put plan.",
            ],
        },
        "stages": {
            "open_stage": {},
            "restore_pick_base": {},
            "prepare_object": {},
            "pick": {},
            "carry_state": {},
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
        "--nav-place-result": args.nav_place_result,
        "--dataset-dir": args.dataset_dir,
        "--pick-handoff-report": args.pick_handoff_report,
        "--put-result": args.put_result,
        "--result-json": args.result_json,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise DemoFailure("missing_required_args", f"missing required args for full demo: {missing}")


def _validate_object_pose_debug_args(args: argparse.Namespace) -> None:
    if not args.task_nav_to_pick:
        raise DemoFailure("missing_required_args", "--task-nav-to-pick is required for --object-pose-debug-only")


def _write_context(args: argparse.Namespace, task_pick: dict[str, Any], scene_usd: Path) -> Path:
    context_path = Path(args.pipeline_context).expanduser().resolve()
    context = {
        "schema_version": 1,
        "mode": MODE,
        "task_json": str(Path(args.task_nav_to_pick).expanduser().resolve()),
        "task_nav_to_pick": str(Path(args.task_nav_to_pick).expanduser().resolve()),
        "task_nav_to_place": str(Path(args.task_nav_to_place).expanduser().resolve()),
        "scene_usd": str(scene_usd),
        "nav_result_json": str(Path(args.nav_pick_result).expanduser().resolve()),
        "nav_pick_result_json": str(Path(args.nav_pick_result).expanduser().resolve()),
        "nav_place_result_json": str(Path(args.nav_place_result).expanduser().resolve()),
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
            "No fixed joint, kinematic hold, or true continuous nav carry is active in phase 1.",
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


async def _run_demo_async(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    from source.data import EpisodeRecorder, load_task

    task_pick_path = _project_path(args.task_nav_to_pick)
    task_place_path = _project_path(args.task_nav_to_place)
    raw_task_pick = _read_json(task_pick_path, missing_reason="missing_task_nav_to_pick")
    raw_task_place = _read_json(task_place_path, missing_reason="missing_task_nav_to_place")
    nav_pick_result = _read_json(args.nav_pick_result, missing_reason="missing_nav_pick_result")
    nav_place_result = _read_json(args.nav_place_result, missing_reason="missing_nav_place_result")
    _validate_nav_result(nav_pick_result, reason="missing_nav_pick_result")
    _validate_nav_result(nav_place_result, reason="missing_nav_place_result")

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
    if args.demo_visuals:
        result["stages"]["open_stage"]["viewport_camera_set"] = _set_viewport_stage_camera(args.viewport_camera_prim)

    task_pick = load_task(task_pick_path)
    world, robot = await pick_handoff._initialize_robot()

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

    try:
        put_report = _run_mvp_put(world, raw_task_place, args)
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
    result.update(
        {
            "success": True,
            "failure_reason": "",
            "failure_detail": "",
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
