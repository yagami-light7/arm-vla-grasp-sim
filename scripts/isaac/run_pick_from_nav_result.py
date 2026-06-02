"""Restore a navigation handoff pose and run the existing grasp flow.

Run from Isaac Sim Script Editor after ``scripts/navigation/run_nav_only.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from pxr import UsdPhysics


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


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def _task_json_path() -> Path:
    if PIPELINE_CONTEXT_JSON.exists():
        context = _read_json(PIPELINE_CONTEXT_JSON)
        return Path(context["task_json"]).expanduser().resolve()
    return Path(os.environ.get("GO2_X5_TASK_JSON", DEFAULT_TASK_JSON)).expanduser().resolve()


def _pipeline_context() -> dict:
    return _read_json(PIPELINE_CONTEXT_JSON) if PIPELINE_CONTEXT_JSON.exists() else {}


def _nav_result_path() -> Path:
    context = _pipeline_context()
    return Path(context.get("nav_result_json", NAV_RESULT_JSON)).expanduser().resolve()


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_handoff_pose(task, nav_result: dict) -> None:
    """Reject unsafe base teleport targets before mutating the open stage."""

    pose = nav_result["final_base_pose_world"]
    x = float(pose["x"])
    y = float(pose["y"])
    context = _pipeline_context()
    clearance_m = float(context.get("handoff_clearance_radius", HANDOFF_CLEARANCE_M))
    goal_tolerance = float(context.get("goal_tolerance", 0.15))
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
        f"goal_distance={goal_distance:.3f}m clearance={clearance_m:.2f}m "
        f"boundary_clearance={boundary_clearance:.3f}m"
    )
    if errors:
        raise RuntimeError(f"nav_collision: unsafe handoff pose ({x:.3f}, {y:.3f}): {'; '.join(errors)}")


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


async def _restore_and_settle(world: World, robot: SingleArticulation, nav_result: dict) -> None:
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
    if math.hypot(float(linear[0]), float(linear[1])) > LINEAR_STABLE_TOLERANCE or abs(float(angular[2])) > ANGULAR_STABLE_TOLERANCE:
        raise RuntimeError(f"base_not_stable: linear={linear.tolist()} angular={angular.tolist()}")


async def main() -> None:
    print("========== Go2-X5 Pick From Navigation Result ==========")
    nav_result = _read_json(_nav_result_path())
    if not nav_result.get("success", False):
        raise RuntimeError(f"navigation did not succeed: {nav_result.get('failure_reason')}")
    task_path = _task_json_path()
    task = load_task(task_path)
    _validate_handoff_pose(task, nav_result)
    context = _pipeline_context()
    dataset_dir = Path(context.get("dataset_dir") or task.recording.dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    recorder = EpisodeRecorder(dataset_dir, task.task_id, task.episode_id, enabled=not bool(context.get("no_record", False)))
    world, robot = await _initialize_robot()
    await _restore_and_settle(world, robot, nav_result)

    try:
        result = await GraspPipeline(recorder=recorder).run(
            GraspTask(
                object_prim_path=task.pick.object_prim_path,
                grasp_mode=task.pick.grasp_mode,
                use_planner_server=bool(context.get("use_planner_server", True)),
            )
        )
    except Exception as exc:
        failure_reason = _grasp_failure_reason(str(exc))
        recorder.write_summary(
            {
                "success": False,
                "failure_reason": failure_reason,
                "failure_detail": str(exc),
                "navigation": nav_result,
            }
        )
        raise
    execution_summary = result["execution"].get("summary", {})
    success = bool(result["success"])
    failure_reason = "" if success else _grasp_failure_reason(str(execution_summary.get("abort_reason", "")))
    recorder.write_summary(
        {
            "success": success,
            "failure_reason": failure_reason,
            "navigation": nav_result,
            "grasp": {
                "success": success,
                "execution_summary": execution_summary,
            },
        }
    )
    print("[pick] success:", success)
    print("[pick] episode:", recorder.episode_dir)


async def guarded_main() -> None:
    try:
        await main()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.ensure_future(guarded_main())
