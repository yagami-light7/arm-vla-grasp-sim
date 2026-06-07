"""Restore a navigation handoff pose and run the existing grasp flow.

Run from Isaac Sim Script Editor after ``scripts/navigation/run_nav_only.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, Usd, UsdGeom, UsdPhysics


SCRIPT_FILE = globals().get("__file__")
DEFAULT_PROJECT_ROOT = Path(SCRIPT_FILE).resolve().parents[2] if SCRIPT_FILE else Path.cwd()
PROJECT_ROOT = Path(os.environ.get("GO2_X5_WORKSPACE", DEFAULT_PROJECT_ROOT)).expanduser().resolve()
if not (PROJECT_ROOT / "source/data/__init__.py").exists():
    raise RuntimeError(
        f"Invalid GO2_X5_WORKSPACE: {PROJECT_ROOT}. "
        "Set GO2_X5_WORKSPACE to the arm_vla repository root before running the handoff script."
    )
sys.path.insert(0, str(PROJECT_ROOT))

# Isaac Sim extensions may import an unrelated top-level ``source`` package
# before Script Editor runs this file. Remove that cached package so imports
# resolve against this repository after the workspace path is inserted.
loaded_source = sys.modules.get("source")
expected_source_dir = (PROJECT_ROOT / "source").resolve()
loaded_source_locations: list[Path] = []
if loaded_source is not None:
    loaded_source_file = getattr(loaded_source, "__file__", None)
    if loaded_source_file:
        loaded_source_locations.append(Path(loaded_source_file).resolve())
    loaded_source_locations.extend(Path(path).resolve() for path in getattr(loaded_source, "__path__", []))
source_matches_workspace = any(
    location == expected_source_dir or expected_source_dir in location.parents or location in expected_source_dir.parents
    for location in loaded_source_locations
)
if loaded_source is not None and not source_matches_workspace:
    for module_name in [name for name in sys.modules if name == "source" or name.startswith("source.")]:
        del sys.modules[module_name]

from source.data import EpisodeRecorder, load_task
from source.manipulation import GraspPipeline, GraspTask
from source.navigation.adapters.frame_utils import world_to_map_local_xy, yaw_to_quat_wxyz
from source.navigation.navlib import OccupancyGridMap


PIPELINE_CONTEXT_JSON = Path(os.environ.get("GO2_X5_PIPELINE_CONTEXT", "/tmp/go2_x5_pipeline_context.json"))
NAV_RESULT_JSON = Path(os.environ.get("GO2_X5_NAV_RESULT", "/tmp/go2_x5_nav_result.json"))
DEFAULT_TASK_JSON = PROJECT_ROOT / "tasks/nav_pick_example.json"
SETTLE_STEPS = int(os.environ.get("GO2_X5_PICK_SETTLE_STEPS", "120"))
LINEAR_STABLE_TOLERANCE = 0.05
ANGULAR_STABLE_TOLERANCE = 0.10
HANDOFF_CLEARANCE_M = float(os.environ.get("GO2_X5_HANDOFF_CLEARANCE_M", "0.30"))
HANDOFF_REPORT_JSON = Path(os.environ.get("GO2_X5_HANDOFF_REPORT", "/tmp/go2_x5_handoff_report.json"))
DEFAULT_TERRAIN_PRIM_PATH = "/World/scene_collision"


class HandoffFailure(RuntimeError):
    """Failure with a stable reason and optional report payload."""

    def __init__(self, failure_reason: str, detail: str, report: dict | None = None):
        super().__init__(f"{failure_reason}: {detail}")
        self.failure_reason = failure_reason
        self.report = report or {}


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("updated_at", time.time())
    HANDOFF_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _grasp_failure_reason(detail: str) -> str:
    lowered = detail.lower()
    if "curobo" in lowered or "planner" in lowered or "plan" in lowered:
        return "curobo_plan_failed"
    if "target" in lowered or "reachable" in lowered:
        return "grasp_target_unreachable"
    if "gripper" in lowered or "close" in lowered:
        return "gripper_failed"
    if "motion" in lowered or "tracking" in lowered or "joint" in lowered:
        return "arm_tracking_failed"
    return "object_not_lifted"


def _execution_failure_reason(execution_summary: dict) -> str:
    """Classify grasp execution failures without confusing side-retreat demos with lift failures."""

    abort_reason = execution_summary.get("abort_reason")
    if abort_reason:
        return _grasp_failure_reason(str(abort_reason))

    grasp_mode = str(execution_summary.get("grasp_mode", ""))
    side_retreat_only = (
        grasp_mode == "side"
        and bool(execution_summary.get("has_planned_retreat", False))
        and not bool(execution_summary.get("has_lift_segment", False))
        and not bool(execution_summary.get("require_object_lift_success", True))
    )
    if side_retreat_only and not bool(execution_summary.get("object_retreat_success", False)):
        return "object_not_grasped"
    if not bool(execution_summary.get("object_lift_success", False)):
        return "object_not_lifted"
    return "object_not_lifted"


def _task_json_path() -> Path:
    if PIPELINE_CONTEXT_JSON.exists():
        context = _read_json(PIPELINE_CONTEXT_JSON)
        return Path(context["task_json"]).expanduser().resolve()
    return Path(os.environ.get("GO2_X5_TASK_JSON", DEFAULT_TASK_JSON)).expanduser().resolve()


def _pipeline_context() -> dict:
    return _read_json(PIPELINE_CONTEXT_JSON) if PIPELINE_CONTEXT_JSON.exists() else {}


def _handoff_smoke_only(context: dict) -> bool:
    env_value = _env_bool("GO2_X5_HANDOFF_SMOKE_ONLY")
    if env_value is not None:
        return env_value
    return bool(context.get("handoff_smoke_only", False))


def _record_handoff_episode(context: dict) -> bool:
    env_value = _env_bool("GO2_X5_HANDOFF_FORCE_RECORD")
    if env_value is not None:
        return env_value
    return not bool(context.get("no_record", False))


def _apply_grasp_policy_env(context: dict) -> None:
    """Forward success-standard and side-grasp planning policy to loaded scripts."""

    require_lift = bool(context.get("require_object_lift_success", True))
    legacy_side_retreat = bool(context.get("legacy_side_retreat", False))
    fallback_retreat = bool(context.get("side_grasp_fallback_retreat", False))
    show_grasp_trajectory = bool(context.get("show_grasp_trajectory", False))
    os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "1" if require_lift else "0"
    os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0" if legacy_side_retreat else "1"
    os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "1" if fallback_retreat else "0"
    os.environ["GO2_X5_SHOW_GRASP_TRAJECTORY"] = "1" if show_grasp_trajectory else "0"


def _nav_result_path() -> Path:
    context = _pipeline_context()
    return Path(context.get("nav_result_json", NAV_RESULT_JSON)).expanduser().resolve()


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _effective_handoff_goal_tolerance(context: dict, nav_result: dict) -> tuple[float, str]:
    """Use the same final-position acceptance that navigation used when available."""

    explicit = _optional_float(context.get("handoff_goal_tolerance"))
    if explicit is not None:
        return explicit, "context.handoff_goal_tolerance"

    candidates: list[tuple[float, str]] = []
    nav_acceptance = _optional_float(nav_result.get("position_acceptance_tolerance"))
    if nav_acceptance is not None:
        candidates.append((nav_acceptance, "nav_result.position_acceptance_tolerance"))

    nav_goal_tolerance = _optional_float(nav_result.get("goal_tolerance"))
    nav_goal_margin = _optional_float(nav_result.get("final_goal_tolerance_margin")) or 0.0
    if nav_goal_tolerance is not None:
        candidates.append((nav_goal_tolerance + max(0.0, nav_goal_margin), "nav_result.goal_tolerance+margin"))

    context_goal_tolerance = _optional_float(context.get("goal_tolerance"))
    context_goal_margin = _optional_float(context.get("final_goal_tolerance_margin")) or 0.0
    if context_goal_tolerance is not None:
        candidates.append((context_goal_tolerance + max(0.0, context_goal_margin), "context.goal_tolerance+margin"))
        candidates.append((context_goal_tolerance, "context.goal_tolerance"))

    if not candidates:
        return 0.15, "default"
    return max(candidates, key=lambda item: item[0])


def _validate_handoff_pose(task, nav_result: dict) -> dict:
    """Reject unsafe base teleport targets before mutating the open stage."""

    pose = nav_result["final_base_pose_world"]
    x = float(pose["x"])
    y = float(pose["y"])
    context = _pipeline_context()
    clearance_m = float(context.get("handoff_clearance_radius", HANDOFF_CLEARANCE_M))
    goal_tolerance, goal_tolerance_source = _effective_handoff_goal_tolerance(context, nav_result)
    map_path = _project_path(str(context.get("nav_map") or task.nav_map))
    grid_map = OccupancyGridMap.from_meta_file(map_path)
    clearance_map = grid_map.inflate(clearance_m)
    row, col = grid_map.world_to_grid(x, y)
    local_x, local_y = world_to_map_local_xy((x, y), grid_map.origin)
    boundary_clearance = min(
        local_x,
        local_y,
        grid_map.width * grid_map.resolution - local_x,
        grid_map.height * grid_map.resolution - local_y,
    )
    errors = []
    expected_goal = task.pick.base_goal
    goal_distance = math.hypot(x - expected_goal.x, y - expected_goal.y)
    reported_goal = nav_result.get("goal_xyyaw")
    if reported_goal is not None and math.hypot(float(reported_goal[0]) - expected_goal.x, float(reported_goal[1]) - expected_goal.y) > 1.0e-3:
        errors.append(f"nav result goal {reported_goal[:2]} does not match task goal {[expected_goal.x, expected_goal.y]}")
    if nav_result.get("final_position_reached") is False:
        errors.append("navigation result reports final_position_reached=false")
    if goal_distance > goal_tolerance:
        errors.append(f"final pose is {goal_distance:.3f} m from task goal, tolerance is {goal_tolerance:.3f} m")
    if grid_map.is_occupied(row, col):
        errors.append("raw map cell is occupied")
    if clearance_map.is_occupied(row, col):
        errors.append(f"cell lacks {clearance_m:.2f} m obstacle clearance")
    if boundary_clearance < clearance_m:
        errors.append(f"map-boundary clearance is only {boundary_clearance:.3f} m")
    print(
        f"[handoff] map check: xy=({x:.3f}, {y:.3f}) grid=({row}, {col}) "
        f"goal_distance={goal_distance:.3f}m tolerance={goal_tolerance:.3f}m "
        f"clearance={clearance_m:.2f}m "
        f"boundary_clearance={boundary_clearance:.3f}m"
    )
    report = {
        "nav_xy": [x, y],
        "grid_index": [int(row), int(col)],
        "goal_distance_m": goal_distance,
        "goal_tolerance_m": goal_tolerance,
        "goal_tolerance_source": goal_tolerance_source,
        "nav_final_position_reached": nav_result.get("final_position_reached"),
        "clearance_radius_m": clearance_m,
        "boundary_clearance_m": boundary_clearance,
        "raw_cell_occupied": bool(grid_map.is_occupied(row, col)),
        "clearance_cell_occupied": bool(clearance_map.is_occupied(row, col)),
        "map_json": str(map_path),
    }
    if errors:
        raise HandoffFailure(
            "nav_collision",
            f"unsafe handoff pose ({x:.3f}, {y:.3f}): {'; '.join(errors)}",
            report,
        )
    return report


def _count_meshes_under(stage, prim_path: str) -> int:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0
    return sum(1 for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh))


def _validate_open_stage(task, context: dict) -> dict:
    """Verify that the currently open Isaac Sim stage can run the handoff."""

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise HandoffFailure("stage_not_ready", "No USD stage is open in Isaac Sim.")

    terrain_prim_path = str(context.get("terrain_prim_path", DEFAULT_TERRAIN_PRIM_PATH))
    object_prim_path = task.pick.object_prim_path
    checks = {
        "terrain_prim_path": terrain_prim_path,
        "terrain_prim_exists": stage.GetPrimAtPath(terrain_prim_path).IsValid(),
        "terrain_mesh_count": _count_meshes_under(stage, terrain_prim_path),
        "object_prim_path": object_prim_path,
        "object_prim_exists": bool(object_prim_path and stage.GetPrimAtPath(object_prim_path).IsValid()),
    }
    errors = []
    if not checks["terrain_prim_exists"]:
        errors.append(f"terrain prim does not exist: {terrain_prim_path}")
    elif checks["terrain_mesh_count"] <= 0:
        errors.append(
            f"terrain prim has no meshes: {terrain_prim_path}. "
            "Check that external scene payloads such as /mnt/sage_data are mounted."
        )
    if object_prim_path and not checks["object_prim_exists"]:
        errors.append(f"object prim does not exist: {object_prim_path}")
    print(
        "[handoff] stage check:",
        {
            "terrain": terrain_prim_path,
            "terrain_meshes": checks["terrain_mesh_count"],
            "object": object_prim_path,
            "object_exists": checks["object_prim_exists"],
        },
    )
    if errors:
        raise HandoffFailure("stage_not_ready", "; ".join(errors), checks)
    return checks


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw radians to a wxyz quaternion."""

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


def _zero_rigid_body_velocities(prim) -> int:
    """Clear authored rigid-body velocities on prim and children when present."""

    zeroed = 0
    for child in Usd.PrimRange(prim):
        if not child.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body = UsdPhysics.RigidBodyAPI(child)
        try:
            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            zeroed += 1
        except Exception as exc:
            print(f"[WARN] Failed to clear rigid body velocity for {child.GetPath()}: {exc}")
    return zeroed


def _get_or_add_xform_op(xformable: UsdGeom.Xformable, op_type) -> UsdGeom.XformOp:
    """Return the canonical xform op, reusing existing precision when authored."""

    prim = xformable.GetPrim()
    attr_name_by_type = {
        UsdGeom.XformOp.TypeTranslate: "xformOp:translate",
        UsdGeom.XformOp.TypeOrient: "xformOp:orient",
    }
    attr_name = attr_name_by_type[op_type]
    attr = prim.GetAttribute(attr_name)
    if attr.IsValid():
        return UsdGeom.XformOp(attr)
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    raise ValueError(f"unsupported xform op type: {op_type}")


def _set_translate_op(op: UsdGeom.XformOp, xyz: tuple[float, float, float]) -> None:
    precision = op.GetPrecision()
    if precision == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*xyz))
    else:
        op.Set(Gf.Vec3f(*xyz))


def _set_orient_op(op: UsdGeom.XformOp, quat_wxyz: tuple[float, float, float, float]) -> None:
    w, x, y, z = quat_wxyz
    precision = op.GetPrecision()
    if precision == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _show_only_task_object(task) -> dict:
    """Hide apple/orange/bottle distractors while keeping the task object visible."""

    object_prim_path = task.pick.object_prim_path
    if not object_prim_path:
        return {"applied": False, "reason": "object_prim_path_missing"}
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise HandoffFailure("stage_not_ready", "No USD stage is open in Isaac Sim.")

    object_prefix = object_prim_path.rstrip("/") + "/"
    hidden_paths: list[str] = []
    shown_paths: list[str] = []
    keywords = ("apple", "orange", "bottle")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        lower_path = prim_path.lower()
        if not any(keyword in lower_path for keyword in keywords):
            continue
        if prim_path == object_prim_path or prim_path.startswith(object_prefix):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()
                shown_paths.append(prim_path)
            continue
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_paths.append(prim_path)

    print(
        "[randomize] object visibility:",
        {
            "keep": object_prim_path,
            "shown": len(shown_paths),
            "hidden": len(hidden_paths),
        },
    )
    return {
        "applied": True,
        "kept_object_prim_path": object_prim_path,
        "shown_paths": shown_paths,
        "hidden_paths": hidden_paths,
    }


def _apply_object_pose_from_task(task) -> dict:
    """Apply task.pick.object_pose_world to the open stage when provided."""

    pose = getattr(task.pick, "object_pose_world", None)
    if pose is None:
        return {"applied": False, "reason": "object_pose_world_missing"}
    object_prim_path = task.pick.object_prim_path
    if not object_prim_path:
        raise HandoffFailure("stage_not_ready", "pick.object_prim_path is required when object_pose_world is set.")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise HandoffFailure("stage_not_ready", "No USD stage is open in Isaac Sim.")
    prim = stage.GetPrimAtPath(object_prim_path)
    if not prim.IsValid():
        raise HandoffFailure("stage_not_ready", f"object prim does not exist: {object_prim_path}")
    if not prim.IsA(UsdGeom.Xformable):
        raise HandoffFailure("stage_not_ready", f"object prim is not xformable: {object_prim_path}")

    quat = _rpy_to_quat_wxyz(pose.roll, pose.pitch, pose.yaw)
    xformable = UsdGeom.Xformable(prim)
    translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
    orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
    _set_translate_op(translate_op, (pose.x, pose.y, pose.z))
    _set_orient_op(orient_op, quat)
    # Preserve authored scale/rotate/transform ops from the scene USD. Randomized
    # tasks should only update object pose, not discard asset sizing or physics
    # alignment encoded in the original xform stack.
    zeroed_count = _zero_rigid_body_velocities(prim)
    report = {
        "applied": True,
        "object_prim_path": object_prim_path,
        "pose_world": pose.to_dict(),
        "quaternion_wxyz": [float(value) for value in quat],
        "xform_op_order": [op.GetOpName() for op in xformable.GetOrderedXformOps()],
        "rigid_body_velocity_zeroed_count": zeroed_count,
    }
    print(
        "[randomize] applied object pose:",
        object_prim_path,
        {
            "x": round(pose.x, 4),
            "y": round(pose.y, 4),
            "z": round(pose.z, 4),
            "roll": round(pose.roll, 4),
            "pitch": round(pose.pitch, 4),
            "yaw": round(pose.yaw, 4),
        },
    )
    return report


def _resolve_articulation_root(stage) -> str:
    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    go2_roots = [path for path in roots if "go2_x5" in path.lower()]
    if len(go2_roots) == 1:
        return go2_roots[0]
    if len(roots) == 1:
        return roots[0]
    raise RuntimeError(f"unable to choose Go2-X5 articulation root from: {roots}")


async def _initialize_robot() -> tuple[World, SingleArticulation]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open in Isaac Sim.")
    world = World.instance()
    if world is None:
        world = World()
    if world.get_physics_context() is None:
        await world.initialize_simulation_context_async()
    await world.play_async()
    await omni.kit.app.get_app().next_update_async()
    articulation_path = _resolve_articulation_root(stage)
    robot = SingleArticulation(prim_path=articulation_path, name="go2_x5_nav_handoff_robot")
    robot.initialize()
    if not robot.is_valid():
        raise RuntimeError(f"invalid articulation: {articulation_path}")
    return world, robot


def _quat_wxyz_to_yaw(quat_wxyz) -> float:
    quat = np.asarray(quat_wxyz, dtype=float)
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _robot_root_report(robot: SingleArticulation) -> dict:
    position, orientation = robot.get_world_pose()
    linear = np.asarray(robot.get_linear_velocity(), dtype=float)
    angular = np.asarray(robot.get_angular_velocity(), dtype=float)
    return {
        "position_xyz": np.asarray(position, dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(orientation, dtype=float).tolist(),
        "yaw": _quat_wxyz_to_yaw(orientation),
        "linear_velocity_xyz": linear.tolist(),
        "angular_velocity_xyz": angular.tolist(),
        "linear_speed_xy": float(math.hypot(float(linear[0]), float(linear[1]))),
        "angular_speed_z": abs(float(angular[2])),
    }


async def _restore_and_settle(world: World, robot: SingleArticulation, nav_result: dict) -> dict:
    pose = nav_result["final_base_pose_world"]
    current_position, _ = robot.get_world_pose()
    root_z = float(os.environ.get("GO2_X5_HANDOFF_ROOT_Z", current_position[2]))
    upright_quaternion = yaw_to_quat_wxyz(float(pose["yaw"]))
    print(
        "[handoff] restoring planar root pose:",
        {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": root_z,
            "yaw": float(pose["yaw"]),
        },
    )
    robot.set_world_pose(
        position=np.asarray([pose["x"], pose["y"], root_z], dtype=float),
        orientation=np.asarray(upright_quaternion, dtype=float),
    )
    robot.set_linear_velocity(np.zeros(3, dtype=float))
    robot.set_angular_velocity(np.zeros(3, dtype=float))
    settle_steps = int(_pipeline_context().get("settle_steps", SETTLE_STEPS))
    for _ in range(settle_steps):
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()
    linear = np.asarray(robot.get_linear_velocity(), dtype=float)
    angular = np.asarray(robot.get_angular_velocity(), dtype=float)
    root_report = _robot_root_report(robot)
    root_position = root_report["position_xyz"]
    root_yaw = float(root_report["yaw"])
    nav_yaw = float(pose["yaw"])
    report = {
        "requested_root_pose": {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": root_z,
            "yaw": nav_yaw,
        },
        "settled_root": root_report,
        "settle_steps": settle_steps,
        "xy_error_m": float(math.hypot(root_position[0] - float(pose["x"]), root_position[1] - float(pose["y"]))),
        "yaw_error_rad": float(abs((root_yaw - nav_yaw + math.pi) % (2.0 * math.pi) - math.pi)),
        "stable": bool(
            math.hypot(float(linear[0]), float(linear[1])) <= LINEAR_STABLE_TOLERANCE
            and abs(float(angular[2])) <= ANGULAR_STABLE_TOLERANCE
        ),
    }
    if not report["stable"]:
        raise HandoffFailure("base_not_stable", f"linear={linear.tolist()} angular={angular.tolist()}", report)
    print(
        "[handoff] settled root:",
        {
            "xy_error_m": round(report["xy_error_m"], 4),
            "yaw_error_rad": round(report["yaw_error_rad"], 4),
            "linear_speed_xy": round(root_report["linear_speed_xy"], 4),
            "angular_speed_z": round(root_report["angular_speed_z"], 4),
        },
    )
    return report


def _state_export_report(state: dict, nav_result: dict) -> dict:
    """Summarize exported arm-base state after handoff smoke."""

    world_base = state.get("poses", {}).get("world_base", {})
    matrix = np.asarray(world_base.get("matrix_4x4"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise HandoffFailure("state_export_failed", "exported poses.world_base.matrix_4x4 is missing or invalid")
    position = matrix[:3, 3]
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    nav_pose = nav_result["final_base_pose_world"]
    return {
        "state_json": "/tmp/go2_x5_isaac_state.json",
        "base_frame_path": state.get("paths", {}).get("base_frame_path"),
        "arm_base_position_xyz": position.tolist(),
        "arm_base_yaw": yaw,
        "arm_base_to_nav_root_xy_m": float(
            math.hypot(position[0] - float(nav_pose["x"]), position[1] - float(nav_pose["y"]))
        ),
        "arm_base_to_nav_root_yaw_error_rad": float(
            abs((yaw - float(nav_pose["yaw"]) + math.pi) % (2.0 * math.pi) - math.pi)
        ),
        "world_collision_cuboids": len(state.get("world_collision", {}).get("cuboids_base", [])),
    }


def _summarize_target(target: dict) -> dict:
    source = target.get("source", {})
    diagnostics = target.get("diagnostics", {})
    workspace = diagnostics.get("target_workspace_base", {})
    return {
        "target_json": "/tmp/go2_x5_target_tcp_pose.json",
        "object_prim_path": target.get("object_prim_path")
        or source.get("object_prim_path")
        or diagnostics.get("object_prim_path"),
        "grasp_mode": target.get("grasp_mode") or source.get("grasp_mode") or diagnostics.get("grasp_mode"),
        "target_workspace_base": workspace,
    }


async def main() -> None:
    print("========== Go2-X5 Pick From Navigation Result ==========")
    nav_result = _read_json(_nav_result_path())
    task_path = _task_json_path()
    task = load_task(task_path)
    raw_task = _read_json(task_path)
    context = _pipeline_context()
    _apply_grasp_policy_env(context)
    dataset_dir = Path(context.get("dataset_dir") or task.recording.dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    recorder = EpisodeRecorder(dataset_dir, task.task_id, task.episode_id, enabled=_record_handoff_episode(context))
    recorder.save_task(raw_task)
    summary: dict = {"navigation": nav_result, "task_json": str(task_path)}
    handoff_report: dict = {}
    try:
        if not nav_result.get("success", False):
            raise HandoffFailure(
                str(nav_result.get("failure_reason") or "nav_timeout"),
                f"navigation did not succeed: {nav_result.get('failure_reason')}",
            )
        handoff_report["map_check"] = _validate_handoff_pose(task, nav_result)
        handoff_report["stage_check"] = _validate_open_stage(task, context)
        handoff_report["object_visibility"] = _show_only_task_object(task)
        handoff_report["object_pose"] = _apply_object_pose_from_task(task)
        world, robot = await _initialize_robot()
        handoff_report["restore"] = await _restore_and_settle(world, robot, nav_result)
        pipeline = GraspPipeline(recorder=recorder)
        task_spec = GraspTask(
            object_prim_path=task.pick.object_prim_path,
            grasp_mode=task.pick.grasp_mode,
            use_planner_server=bool(context.get("use_planner_server", True)),
        )
        if _handoff_smoke_only(context):
            state = await pipeline.export_state(task_spec)
            handoff_report["state_export"] = _state_export_report(state, nav_result)
            summary.update(
                {
                    "success": True,
                    "failure_reason": "",
                    "mode": "handoff_smoke_only",
                    "handoff": handoff_report,
                }
            )
            recorder.write_summary(summary)
            _write_report(summary)
            print("[handoff] smoke success:", HANDOFF_REPORT_JSON)
            print("[handoff] state JSON:", task_spec.state_json)
            return

        state = await pipeline.export_state(task_spec)
        handoff_report["state_export"] = _state_export_report(state, nav_result)
        target = await pipeline.generate_target(task_spec)
        handoff_report["target"] = _summarize_target(target)
        plan = pipeline.plan(task_spec)
        execution = await pipeline.execute(task_spec)
        result = {
            "success": bool(execution.get("summary", {}).get("task_success", False)),
            "state": state,
            "target": target,
            "plan": plan,
            "execution": execution,
        }
    except Exception as exc:
        failure_reason = exc.failure_reason if isinstance(exc, HandoffFailure) else _grasp_failure_reason(str(exc))
        if isinstance(exc, HandoffFailure) and exc.report:
            handoff_report.setdefault("failure_report", exc.report)
        recorder.write_summary(
            {
                "success": False,
                "failure_reason": failure_reason,
                "failure_detail": str(exc),
                "navigation": nav_result,
                "handoff": handoff_report,
                "task_json": str(task_path),
            }
        )
        _write_report(
            {
                "success": False,
                "failure_reason": failure_reason,
                "failure_detail": str(exc),
                "navigation": nav_result,
                "handoff": handoff_report,
                "task_json": str(task_path),
            }
        )
        raise
    execution_summary = result["execution"].get("summary", {})
    success = bool(result["success"])
    failure_reason = "" if success else _execution_failure_reason(execution_summary)
    summary.update(
        {
            "success": success,
            "failure_reason": failure_reason,
            "mode": "full_pick",
            "handoff": handoff_report,
            "grasp": {
                "success": success,
                "plan_summary": result["plan"].get("summary", {}),
                "target": handoff_report.get("target", {}),
                "execution_summary": execution_summary,
            },
        }
    )
    recorder.write_summary(summary)
    _write_report(summary)
    print("[pick] success:", success)
    print("[pick] episode:", recorder.episode_dir)
    print("[pick] handoff report:", HANDOFF_REPORT_JSON)


async def guarded_main() -> None:
    try:
        await main()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.ensure_future(guarded_main())
