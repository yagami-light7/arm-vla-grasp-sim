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
DEFAULT_HANDOFF_REPORT_JSON = Path("/tmp/go2_x5_handoff_report.json")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", default=None)
    parser.add_argument(
        "--batch-manifest",
        default=None,
        help=(
            "Optional JSON manifest with an 'episodes' list. Each episode can "
            "override task_json, nav_result, handoff_report, dataset_dir, scene_usd, "
            "nav_map, and replay_trajectory while reusing one Isaac Sim window."
        ),
    )
    parser.add_argument("--batch-continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-usd", default=None)
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--nav-result", default=str(DEFAULT_NAV_RESULT_JSON))
    parser.add_argument("--handoff-report", default=str(DEFAULT_HANDOFF_REPORT_JSON))
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
    parser.add_argument(
        "--side-retreat-only",
        action="store_true",
        help="For side grasps, skip vertical lift and count the planned reverse retreat as pick success.",
    )
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument(
        "--fail-on-object-reset-drift",
        action="store_true",
        help="Fail pick handoff if the object center drifts after target-pose apply and velocity reset.",
    )
    parser.add_argument(
        "--keep-window-open",
        action="store_true",
        help="After the handoff run finishes, keep the Isaac Sim window alive until the user closes it.",
    )
    parser.add_argument(
        "--show-grasp-trajectory",
        action="store_true",
        help="Draw planned TCP path markers during grasp execution.",
    )
    parser.add_argument("--stage-load-updates", type=int, default=30)
    parser.add_argument("--replay-trajectory", default=None, help="Optional navigation trajectory JSONL to replay before grasp.")
    parser.add_argument("--replay-real-time", action="store_true", help="Replay the navigation trajectory using recorded timestamps.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed multiplier for --replay-real-time.")
    parser.add_argument("--set-viewport-camera", action="store_true", help="Switch the active viewport to --viewport-camera-prim.")
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _load_raw_task(task_json: Path) -> dict:
    return json.loads(task_json.read_text(encoding="utf-8"))


def _load_batch_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"batch manifest has no episodes: {manifest_path}")
    normalized = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"batch manifest episode {index} must be an object.")
        normalized.append(dict(episode))
    return normalized


def _episode_args(base_args: argparse.Namespace, episode: dict) -> argparse.Namespace:
    values = vars(base_args).copy()
    key_map = {
        "task_json": "task_json",
        "scene_usd": "scene_usd",
        "nav_map": "nav_map",
        "nav_result": "nav_result",
        "handoff_report": "handoff_report",
        "dataset_dir": "dataset_dir",
        "replay_trajectory": "replay_trajectory",
    }
    for source_key, arg_key in key_map.items():
        if source_key in episode and episode[source_key] is not None:
            values[arg_key] = episode[source_key]
    return argparse.Namespace(**values)


def _handoff_report_path(args: argparse.Namespace) -> Path:
    return Path(args.handoff_report).expanduser().resolve()


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_standalone_report(args: argparse.Namespace, payload: dict) -> None:
    """Write a diagnostic report before the handoff coroutine owns the file."""

    report_path = _handoff_report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "success": False,
        "failure_reason": "standalone_not_completed",
        "failure_detail": "",
        "standalone": {
            "status": "starting",
            "task_json": str(_project_path(args.task_json)),
            "scene_usd": None,
            "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
            "handoff_report_json": str(report_path),
            "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve()) if args.replay_trajectory else None,
            "updated_at": time.time(),
        },
    }
    report.update(payload)
    standalone = dict(report.get("standalone", {}))
    standalone.setdefault("updated_at", time.time())
    report["standalone"] = standalone
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_standalone_failure_if_needed(args: argparse.Namespace, exc: BaseException) -> None:
    """Preserve handoff-authored reports, otherwise write standalone failure detail."""

    report_path = _handoff_report_path(args)
    existing = _read_json_if_exists(report_path)
    if existing is not None and existing.get("handoff"):
        return
    _write_standalone_report(
        args,
        {
            "success": False,
            "failure_reason": "standalone_exception",
            "failure_detail": str(exc),
            "standalone": {
                "status": "exception",
                "task_json": str(_project_path(args.task_json)),
                "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                "handoff_report_json": str(report_path),
                "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve()) if args.replay_trajectory else None,
                "updated_at": time.time(),
            },
        },
    )


def _write_context(args: argparse.Namespace, task_json: Path, raw_task: dict) -> Path:
    nav_map = args.nav_map or raw_task["nav_map"]
    scene_usd = args.scene_usd or raw_task["scene_usd"]
    context_path = Path(args.pipeline_context).expanduser().resolve()
    context = _read_json_if_exists(context_path) or {}
    context = {
        **context,
        "schema_version": 1,
        "task_json": str(task_json),
        "scene_usd": str(_project_path(scene_usd)),
        "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
        "handoff_report_json": str(Path(args.handoff_report).expanduser().resolve()),
        "nav_map": str(_project_path(nav_map)),
        "terrain_prim_path": args.terrain_prim_path,
        "handoff_clearance_radius": args.handoff_clearance_radius,
        "use_planner_server": args.use_planner_server,
        "handoff_smoke_only": args.handoff_smoke_only,
        "require_object_lift_success": not args.allow_retreat_success,
        "legacy_side_retreat": args.legacy_side_retreat,
        "side_retreat_only": args.side_retreat_only,
        "side_grasp_fallback_retreat": args.side_grasp_fallback_retreat,
        "keep_window_open": args.keep_window_open,
        "show_grasp_trajectory": args.show_grasp_trajectory,
        "fail_on_object_reset_drift": args.fail_on_object_reset_drift,
        "settle_steps": args.settle_steps,
        "object_pose_policy": (raw_task.get("randomization") or {}).get("object_pose_policy", {}),
        "dataset_dir": args.dataset_dir,
        "no_record": args.no_record,
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    return context_path


def _merge_context(context_path: Path, updates: dict) -> None:
    context = _read_json_if_exists(context_path) or {}
    context.update(updates)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _prepare_randomized_task_stage(task_json: Path) -> dict:
    """Apply randomized object visibility and pose before replay/grasp."""

    from source.data import load_task
    from scripts.isaac.run_pick_from_nav_result import (
        RANDOMIZED_OBJECT_STAGE_PREPARED_ENV,
        _apply_object_pose_from_task,
        _show_only_task_object,
    )

    task = load_task(task_json)
    report = {
        "object_visibility": _show_only_task_object(task),
        "object_pose": _apply_object_pose_from_task(task),
    }
    os.environ[RANDOMIZED_OBJECT_STAGE_PREPARED_ENV] = "1"
    print("[standalone] prepared randomized task stage:", report)
    return report


def _candidate_stage_camera_paths(camera_prim_path: str) -> list[str]:
    """Return common camera path candidates for authored SAGE scenes."""

    candidates = [camera_prim_path]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path == "/World/Camera_main":
        candidates.append("/World/camera_main")
    return list(dict.fromkeys(candidates))


def _set_viewport_stage_camera(camera_prim_path: str) -> bool:
    """Switch the visible Isaac Sim viewport to an authored camera prim."""

    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Cannot set stage camera: no active USD stage.")
            return False

        selected_path = None
        for candidate in _candidate_stage_camera_paths(camera_prim_path):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                selected_path = candidate
                break
        if selected_path is None:
            print(
                "[WARN] Cannot set stage camera: no valid Camera prim found. "
                f"tried={_candidate_stage_camera_paths(camera_prim_path)}"
            )
            return False

        viewport = get_active_viewport()
        if viewport is None:
            print("[WARN] Cannot set stage camera: no active viewport.")
            return False
        sdf_path = Sdf.Path(selected_path)
        try:
            viewport.camera_path = sdf_path
        except Exception:
            if hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(sdf_path)
            else:
                raise
        print(f"[standalone] viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set stage camera {camera_prim_path}: {exc}")
        return False


def _read_replay_trajectory(path: Path) -> list[dict]:
    """Load a navigation replay trajectory written by run_nav_only.py."""

    if not path.exists():
        raise FileNotFoundError(f"replay trajectory does not exist: {path}")
    frames: list[dict] = []
    required_keys = {
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "joint_pos",
    }
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            missing = sorted(required_keys - set(frame))
            if missing:
                raise RuntimeError(f"replay frame {line_number} is missing keys: {missing}")
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"replay trajectory is empty: {path}")
    return frames


def _robot_dof_names(robot) -> list[str]:
    """Return Isaac Sim articulation DOF names using the local API variant."""

    dof_names = getattr(robot, "dof_names", None)
    if dof_names is not None:
        return list(dof_names)
    joint_names = getattr(robot, "joint_names", None)
    if joint_names is not None:
        return list(joint_names)
    return []


def _map_replay_joint_positions(robot, frame: dict) -> list[float] | None:
    """Map recorded joint positions onto the open-stage articulation order."""

    current_joint_positions = robot.get_joint_positions()
    if current_joint_positions is None:
        return None

    current = [float(value) for value in current_joint_positions]
    recorded = [float(value) for value in frame["joint_pos"]]
    dof_names = _robot_dof_names(robot)
    recorded_names = list(frame.get("joint_names") or [])

    if dof_names and recorded_names:
        recorded_by_name = {name: idx for idx, name in enumerate(recorded_names)}
        common = 0
        target = current[:]
        for idx, name in enumerate(dof_names):
            recorded_idx = recorded_by_name.get(name)
            if recorded_idx is None or recorded_idx >= len(recorded):
                continue
            target[idx] = recorded[recorded_idx]
            common += 1
        if common == 0:
            print("[WARN] Replay joint name mapping found no common joints; root-only replay will continue.")
            return None
        if common != len(dof_names):
            print(f"[WARN] Replay mapped {common}/{len(dof_names)} open-stage joints; unmapped joints keep current pose.")
        return target

    if len(recorded) == len(current):
        return recorded

    print(
        "[WARN] Replay joint count mismatch without joint names; "
        f"recorded={len(recorded)} current={len(current)}. Root-only replay will continue."
    )
    return None


def _schedule_kit_coroutine(coro):
    """Schedule a coroutine on Kit's async engine when available."""

    try:
        from omni.kit.async_engine import run_coroutine

        return run_coroutine(coro)
    except Exception:
        loop = asyncio.get_event_loop()
        return loop.create_task(coro)


def _drive_future_to_completion(future, *, timeout_s: float, label: str):
    """Advance Isaac Sim until a Kit/asyncio future completes."""

    started_at = time.time()
    while not future.done():
        simulation_app.update()
        if timeout_s > 0.0 and time.time() - started_at > timeout_s:
            if hasattr(future, "cancel"):
                future.cancel()
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    return future.result()


async def _initialize_replay_robot():
    """Initialize the open-stage articulation for visual replay."""

    import omni.kit.app
    import omni.usd
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleArticulation
    from scripts.isaac.run_pick_from_nav_result import _resolve_articulation_root

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open for replay.")
    world = World.instance()
    if world is None:
        world = World()
    if world.get_physics_context() is None:
        await world.initialize_simulation_context_async()
    await world.play_async()
    await omni.kit.app.get_app().next_update_async()
    articulation_path = _resolve_articulation_root(stage)
    robot = SingleArticulation(prim_path=articulation_path, name="go2_x5_nav_video_replay_robot")
    robot.initialize()
    if not robot.is_valid():
        raise RuntimeError(f"invalid articulation for replay: {articulation_path}")
    return world, robot


async def _replay_navigation_trajectory_async(path: Path, *, real_time: bool, speed: float) -> None:
    """Replay navigation in the open Isaac Sim grasp stage before handoff."""

    import numpy as np
    import omni.kit.app
    from isaacsim.core.utils.types import ArticulationAction

    frames = _read_replay_trajectory(path)
    world, robot = await _initialize_replay_robot()
    print(f"[standalone] replaying navigation trajectory: {path} frames={len(frames)} real_time={real_time} speed={speed}")

    previous_frame = None
    playback_speed = max(float(speed), 1.0e-6)
    for frame in frames:
        position = np.asarray(frame["root_pos_w"], dtype=float)
        orientation = np.asarray(frame["root_quat_w"], dtype=float)
        robot.set_world_pose(position=position, orientation=orientation)
        robot.set_linear_velocity(np.asarray(frame["root_lin_vel_w"], dtype=float))
        robot.set_angular_velocity(np.asarray(frame["root_ang_vel_w"], dtype=float))

        joint_positions = _map_replay_joint_positions(robot, frame)
        if joint_positions is not None:
            robot.apply_action(ArticulationAction(joint_positions=np.asarray(joint_positions, dtype=float)))

        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        if real_time:
            if previous_frame is None:
                delay = 0.0
            else:
                delay = max(
                    0.0,
                    float(frame.get("timestamp", 0.0)) - float(previous_frame.get("timestamp", 0.0)),
                ) / playback_speed
            if delay > 0.0:
                time.sleep(delay)
        previous_frame = frame
    print("[standalone] navigation replay complete")


def _run_replay_task(replay_trajectory: Path, *, real_time: bool, speed: float, timeout_s: float) -> None:
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction

    frames = _read_replay_trajectory(replay_trajectory)
    future = _schedule_kit_coroutine(_initialize_replay_robot())
    world, robot = _drive_future_to_completion(future, timeout_s=timeout_s, label="navigation replay initialization")
    print(
        f"[standalone] replaying navigation trajectory: {replay_trajectory} "
        f"frames={len(frames)} real_time={real_time} speed={speed}"
    )

    previous_frame = None
    playback_speed = max(float(speed), 1.0e-6)
    for frame in frames:
        position = np.asarray(frame["root_pos_w"], dtype=float)
        orientation = np.asarray(frame["root_quat_w"], dtype=float)
        robot.set_world_pose(position=position, orientation=orientation)
        robot.set_linear_velocity(np.asarray(frame["root_lin_vel_w"], dtype=float))
        robot.set_angular_velocity(np.asarray(frame["root_ang_vel_w"], dtype=float))

        joint_positions = _map_replay_joint_positions(robot, frame)
        if joint_positions is not None:
            robot.apply_action(ArticulationAction(joint_positions=np.asarray(joint_positions, dtype=float)))

        world.step(render=True)
        simulation_app.update()

        if real_time:
            if previous_frame is None:
                delay = 0.0
            else:
                delay = max(
                    0.0,
                    float(frame.get("timestamp", 0.0)) - float(previous_frame.get("timestamp", 0.0)),
                ) / playback_speed
            if delay > 0.0:
                time.sleep(delay)
        previous_frame = frame
    print("[standalone] navigation replay complete")


def _run_handoff_task(timeout_s: float) -> None:
    from scripts.isaac import run_pick_from_nav_result

    future = _schedule_kit_coroutine(run_pick_from_nav_result.guarded_main())
    _drive_future_to_completion(future, timeout_s=timeout_s, label="standalone pick")


def _keep_window_open_until_closed() -> None:
    """Keep the Kit app responsive after the run for video capture."""

    print("[standalone] keep-window-open enabled; close the Isaac Sim window to end this process.")
    while simulation_app.is_running():
        simulation_app.update()
        time.sleep(1.0 / 60.0)


def _run_single_episode(args: argparse.Namespace) -> None:
    if args.side_retreat_only:
        args.legacy_side_retreat = True
        args.allow_retreat_success = True
    if not args.task_json:
        raise ValueError("--task-json is required unless --batch-manifest is used.")

    task_json = _project_path(args.task_json)
    raw_task = _load_raw_task(task_json)
    scene_usd = _project_path(args.scene_usd or raw_task["scene_usd"])
    context_json = _write_context(args, task_json, raw_task)

    os.environ["GO2_X5_WORKSPACE"] = str(PROJECT_ROOT)
    os.environ["GO2_X5_PIPELINE_CONTEXT"] = str(context_json)
    os.environ["GO2_X5_NAV_RESULT"] = str(Path(args.nav_result).expanduser().resolve())
    os.environ["GO2_X5_HANDOFF_REPORT"] = str(Path(args.handoff_report).expanduser().resolve())
    os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "1" if args.handoff_smoke_only else "0"
    os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "0" if args.allow_retreat_success else "1"
    os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0" if args.legacy_side_retreat else "1"
    os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "1" if args.side_grasp_fallback_retreat else "0"
    os.environ["GO2_X5_SHOW_GRASP_TRAJECTORY"] = "1" if args.show_grasp_trajectory else "0"

    _write_standalone_report(
        args,
        {
            "standalone": {
                "status": "starting",
                "task_json": str(task_json),
                "scene_usd": str(scene_usd),
                "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                "handoff_report_json": str(_handoff_report_path(args)),
                "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve())
                if args.replay_trajectory
                else None,
                "updated_at": time.time(),
            },
        },
    )
    try:
        _open_stage(scene_usd, args.stage_load_updates)
        stage_prepare_report = _prepare_randomized_task_stage(task_json)
        _merge_context(
            context_json,
            {
                "randomized_object_stage_prepared": True,
                "randomized_object_stage_prepare_report": stage_prepare_report,
            },
        )
        _write_standalone_report(
            args,
            {
                "standalone": {
                    "status": "stage_opened",
                    "task_json": str(task_json),
                    "scene_usd": str(scene_usd),
                    "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                    "handoff_report_json": str(_handoff_report_path(args)),
                    "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve())
                    if args.replay_trajectory
                    else None,
                    "updated_at": time.time(),
                },
                "stage_prepare": stage_prepare_report,
            },
        )
        if args.set_viewport_camera:
            _set_viewport_stage_camera(args.viewport_camera_prim)
        if args.replay_trajectory:
            _run_replay_task(
                Path(args.replay_trajectory).expanduser().resolve(),
                real_time=args.replay_real_time,
                speed=args.replay_speed,
                timeout_s=args.timeout_s,
            )
            _write_standalone_report(
                args,
                {
                    "standalone": {
                        "status": "replay_complete",
                        "task_json": str(task_json),
                        "scene_usd": str(scene_usd),
                        "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                        "handoff_report_json": str(_handoff_report_path(args)),
                        "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve()),
                        "updated_at": time.time(),
                    },
                },
            )
        _write_standalone_report(
            args,
            {
                "standalone": {
                    "status": "handoff_running",
                    "task_json": str(task_json),
                    "scene_usd": str(scene_usd),
                    "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                    "handoff_report_json": str(_handoff_report_path(args)),
                    "replay_trajectory": str(Path(args.replay_trajectory).expanduser().resolve())
                    if args.replay_trajectory
                    else None,
                    "updated_at": time.time(),
                },
            },
        )
        _run_handoff_task(args.timeout_s)
    except Exception as exc:
        _write_standalone_failure_if_needed(args, exc)
        raise

    handoff_report = _handoff_report_path(args)
    if not handoff_report.exists():
        raise RuntimeError(
            f"handoff task finished without writing report: {handoff_report}. "
            "The grasp coroutine may not have run to completion."
        )
    report = _read_json_if_exists(handoff_report) or {}
    if not bool(report.get("success", False)):
        raise RuntimeError(
            f"handoff task finished with failure report: {handoff_report} "
            f"reason={report.get('failure_reason')} detail={report.get('failure_detail')}"
        )
    print("[standalone] pick handoff complete")


def main() -> int:
    if args_cli.batch_manifest:
        episodes = _load_batch_manifest(args_cli.batch_manifest)
        failures = 0
        for index, episode in enumerate(episodes):
            episode_args = _episode_args(args_cli, episode)
            print(
                f"[standalone batch] episode {index + 1}/{len(episodes)} "
                f"task={episode_args.task_json}"
            )
            try:
                _run_single_episode(episode_args)
            except Exception as exc:
                failures += 1
                print(f"[standalone batch] episode {index + 1} failed: {exc}")
                if not args_cli.batch_continue_on_failure:
                    raise
        if failures:
            print(f"[standalone batch] completed with failures={failures}/{len(episodes)}")
            return 1
        print(f"[standalone batch] completed episodes={len(episodes)}")
        return 0

    _run_single_episode(args_cli)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if args_cli.keep_window_open:
            _keep_window_open_until_closed()
        else:
            simulation_app.close()
