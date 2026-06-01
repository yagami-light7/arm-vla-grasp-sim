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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data import EpisodeRecorder, load_task
from source.manipulation import GraspPipeline, GraspTask


PIPELINE_CONTEXT_JSON = Path(os.environ.get("GO2_X5_PIPELINE_CONTEXT", "/tmp/go2_x5_pipeline_context.json"))
NAV_RESULT_JSON = Path(os.environ.get("GO2_X5_NAV_RESULT", "/tmp/go2_x5_nav_result.json"))
DEFAULT_TASK_JSON = PROJECT_ROOT / "tasks/nav_pick_example.json"
SETTLE_STEPS = int(os.environ.get("GO2_X5_PICK_SETTLE_STEPS", "120"))
LINEAR_STABLE_TOLERANCE = 0.05
ANGULAR_STABLE_TOLERANCE = 0.10


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
    world = World.instance() or World()
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
    robot.set_world_pose(
        position=np.asarray([pose["x"], pose["y"], pose["z"]], dtype=float),
        orientation=np.asarray(pose["quat_wxyz"], dtype=float),
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
