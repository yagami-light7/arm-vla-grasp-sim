#!/usr/bin/env python3
"""Run Go2-X5 A* + DWA navigation in Isaac Lab and write a grasp handoff JSON."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_LAB_SOURCE = PROJECT_ROOT / "source/robot_lab"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROBOT_LAB_SOURCE))

from scripts.navigation import isaaclab_cli_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-json", required=True)
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0")
parser.add_argument("--map", dest="scene_usd", default=None, help="Override task scene USD.")
parser.add_argument("--nav-map", default=None, help="Override task navigation map metadata.")
parser.add_argument("--dataset-dir", default=None)
parser.add_argument("--nav-result", default="/tmp/go2_x5_nav_result.json")
parser.add_argument("--inflate-radius", type=float, default=0.30)
parser.add_argument("--local-clearance-radius", type=float, default=0.25)
parser.add_argument("--goal-tolerance", type=float, default=0.35)
parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
parser.add_argument("--max-nav-steps", type=int, default=3000)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--lookahead-distance", type=float, default=0.60)
parser.add_argument("--max-lin-vel", type=float, default=0.50)
parser.add_argument("--max-ang-vel", type=float, default=1.00)
parser.add_argument("--head-camera", action="store_true")
parser.add_argument("--head-camera-height", type=int, default=480)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--no-record", action="store_true")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--debug-command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
parser.add_argument("--debug-print-every", type=int, default=60)
isaaclab_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.head_camera:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
from source.data import EpisodeRecorder, load_task
from source.navigation.adapters.frame_utils import wrap_yaw
from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter
from source.navigation.navlib import DWAConfig
from source.navigation import NavPlanner


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _path_length(path_world: list[tuple[float, float]]) -> float:
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path_world, path_world[1:]))


def _write_nav_result(nav_result_path: Path, recorder: EpisodeRecorder, result: dict) -> None:
    nav_result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    recorder.write_summary(
        {
            "success": result["success"],
            "failure_reason": result["failure_reason"],
            "navigation": result,
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _jpeg_bytes(rgb_tensor) -> bytes | None:
    if rgb_tensor is None:
        return None
    stream = io.BytesIO()
    Image.fromarray(rgb_tensor.cpu().numpy()).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def _disable_event(env_cfg, name: str) -> None:
    if hasattr(env_cfg.events, name):
        setattr(env_cfg.events, name, None)


def _configure_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    scene_usd: Path,
    start_pose: tuple[float, float, float],
) -> None:
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/scene_collision",
        terrain_type="usd",
        usd_path=str(scene_usd),
        debug_vis=False,
    )
    env_cfg.scene.sky_light = None
    start_x, start_y, start_yaw = start_pose
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (start_x, start_x),
            "y": (start_y, start_y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (start_yaw, start_yaw),
        },
        "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
    }
    env_cfg.observations.policy.enable_corruption = False
    _disable_event(env_cfg, "randomize_apply_external_force_torque")
    _disable_event(env_cfg, "push_robot")
    _disable_event(env_cfg, "randomize_push_robot")
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.curriculum.terrain_levels = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.illegal_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None

    if args_cli.head_camera:
        env_cfg.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/head_cam",
            update_period=0.0,
            height=args_cli.head_camera_height,
            width=args_cli.head_camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.28, 0.0, 0.07),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    task_path = _project_path(args_cli.task_json)
    task = load_task(task_path)
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    scene_usd = _project_path(args_cli.scene_usd or task.scene_usd)
    nav_map = _project_path(args_cli.nav_map or task.nav_map)
    dataset_dir = _project_path(args_cli.dataset_dir or task.recording.dataset_dir)
    nav_result_path = Path(args_cli.nav_result).expanduser().resolve()

    _configure_env(env_cfg, scene_usd=scene_usd, start_pose=(task.start.x, task.start.y, task.start.yaw))
    agent_cfg = isaaclab_cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    adapter = Go2LocomotionAdapter(env, policy, env.get_observations())
    recorder = EpisodeRecorder(dataset_dir, task.task_id, task.episode_id, enabled=not args_cli.no_record)
    recorder.save_task(raw_task)

    dt = float(env.unwrapped.step_dt)
    adapter.settle(args_cli.settle_steps)
    settled_pose = adapter.get_base_pose()
    dwa_cfg = DWAConfig(
        control_dt=dt,
        lookahead_distance=args_cli.lookahead_distance,
        goal_tolerance=args_cli.goal_tolerance,
        max_linear_velocity=args_cli.max_lin_vel,
        max_angular_velocity=args_cli.max_ang_vel,
    )
    planner = NavPlanner(
        str(nav_map),
        args_cli.inflate_radius,
        dwa_cfg,
        local_clearance_radius=args_cli.local_clearance_radius,
    )
    goal = task.pick.base_goal
    try:
        path_world = planner.plan_global_path(settled_pose[:2], (goal.x, goal.y))
    except (RuntimeError, ValueError) as exc:
        result = {
            "schema_version": 1,
            "success": False,
            "failure_reason": "nav_collision",
            "failure_detail": str(exc),
            "final_base_pose_world": adapter.get_base_pose_full(),
            "goal_xyyaw": [goal.x, goal.y, goal.yaw],
            "yaw_error": wrap_yaw(goal.yaw - settled_pose[2]),
            "path_length": 0.0,
            "timeout": False,
            "episode_dir": str(recorder.episode_dir),
            "elapsed_wall_time_s": 0.0,
        }
        _write_nav_result(nav_result_path, recorder, result)
        env.close()
        return
    if math.hypot(path_world[-1][0] - goal.x, path_world[-1][1] - goal.y) > 0.01:
        path_world.append((goal.x, goal.y))

    started_at = time.time()
    success = False
    failure_reason = ""
    yaw_error = wrap_yaw(goal.yaw - settled_pose[2])
    final_phase = "nav"
    for step in range(args_cli.max_nav_steps):
        pose = adapter.get_base_pose()
        speed = adapter.get_base_velocity()
        distance = math.hypot(pose[0] - goal.x, pose[1] - goal.y)
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        if distance <= args_cli.goal_tolerance:
            final_phase = "yaw_align"
            if abs(yaw_error) <= args_cli.goal_yaw_tolerance:
                success = True
                break
            command = (0.0, 0.0, float(np.clip(1.5 * yaw_error, -0.35, 0.35)))
        elif args_cli.debug_command is not None:
            command = tuple(float(value) for value in args_cli.debug_command)
        else:
            command = planner.compute_command(pose, speed, path_world)
        adapter.apply_base_command(*command)
        adapter.step()
        if step % task.recording.save_every_n_steps == 0:
            recorder.record(final_phase, adapter.snapshot(timestamp=step * dt, phase=final_phase), front_image=_jpeg_bytes(adapter.get_front_rgb()))
        if args_cli.debug_print_every > 0 and step % args_cli.debug_print_every == 0:
            print(
                f"[nav] step={step} phase={final_phase} pose=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) "
                f"goal_dist={distance:.3f} yaw_error={yaw_error:.3f} cmd={command}"
            )
        if args_cli.real_time:
            time.sleep(max(0.0, dt))
    else:
        failure_reason = "yaw_align_failed" if final_phase == "yaw_align" else "nav_timeout"

    adapter.settle(args_cli.settle_steps)
    stable = adapter.is_stable()
    if success and not stable:
        success = False
        failure_reason = "base_not_stable"
    elif not success and not failure_reason:
        failure_reason = "yaw_align_failed" if final_phase == "yaw_align" else "nav_timeout"

    final_pose = adapter.get_base_pose_full()
    yaw_error = wrap_yaw(goal.yaw - final_pose["yaw"])
    result = {
        "schema_version": 1,
        "success": success,
        "failure_reason": failure_reason,
        "final_base_pose_world": final_pose,
        "goal_xyyaw": [goal.x, goal.y, goal.yaw],
        "yaw_error": yaw_error,
        "path_length": _path_length(path_world),
        "timeout": failure_reason == "nav_timeout",
        "episode_dir": str(recorder.episode_dir),
        "elapsed_wall_time_s": time.time() - started_at,
    }
    _write_nav_result(nav_result_path, recorder, result)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
