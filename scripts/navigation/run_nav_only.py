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
parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
parser.add_argument("--map", dest="scene_usd", default=None, help="Override task scene USD.")
parser.add_argument("--terrain-prim-path", default="/World/scene_collision", help="Collision prim referenced by navigation.")
parser.add_argument("--ground-height", type=float, default=0.0, help="World Z height for the navigation ground plane.")
parser.add_argument(
    "--add-nav-ground",
    action="store_true",
    help="Add a separate ground plane. Leave disabled when scene_collision already contains the floor.",
)
parser.add_argument("--nav-map", default=None, help="Override task navigation map metadata.")
parser.add_argument("--dataset-dir", default=None)
parser.add_argument("--nav-result", default="/tmp/go2_x5_nav_result.json")
parser.add_argument("--inflate-radius", type=float, default=0.40)
parser.add_argument("--local-clearance-radius", type=float, default=0.35)
parser.add_argument("--goal-tolerance", type=float, default=0.15)
parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
parser.add_argument("--max-nav-steps", type=int, default=3000)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--stall-window-steps", type=int, default=240)
parser.add_argument("--stall-min-progress", type=float, default=0.05)
parser.add_argument("--stall-min-forward-command", type=float, default=0.05)
parser.add_argument("--stall-min-forward-ratio", type=float, default=0.25)
parser.add_argument("--lookahead-distance", type=float, default=0.35)
parser.add_argument("--prediction-horizon", type=float, default=1.80)
parser.add_argument("--max-lin-vel", type=float, default=0.50)
parser.add_argument("--max-ang-vel", type=float, default=1.00)
parser.add_argument("--yaw-align-kp", type=float, default=2.0)
parser.add_argument("--yaw-align-min-wz", type=float, default=0.55)
parser.add_argument("--yaw-align-max-wz", type=float, default=1.00)
parser.add_argument("--yaw-align-vx", type=float, default=0.08)
parser.add_argument("--yaw-align-activation-yaw-error", type=float, default=0.0)
parser.add_argument("--yaw-align-allow-reverse", action="store_true")
parser.add_argument("--yaw-align-stall-window-steps", type=int, default=240)
parser.add_argument("--yaw-align-min-progress", type=float, default=0.08)
parser.add_argument("--yaw-settle-stable-steps", type=int, default=20)
parser.add_argument("--head-camera", action="store_true")
parser.add_argument("--head-camera-height", type=int, default=480)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument(
    "--load-visual-scene",
    action="store_true",
    help="Load a visual-only scene prim into the Isaac Lab viewport for debugging/demo.",
)
parser.add_argument("--visual-prim-path", default="/World/gauss", help="Visual prim referenced when --load-visual-scene is set.")
parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--follow-camera-distance", type=float, default=2.4)
parser.add_argument("--follow-camera-height", type=float, default=0.8)
parser.add_argument("--follow-camera-side", type=float, default=0.0)
parser.add_argument("--no-record", action="store_true")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--flat-terrain", action="store_true", help="Keep the task's flat terrain for locomotion debugging.")
parser.add_argument("--disable-sky-light", action="store_true", help="Disable the default Isaac Lab sky light.")
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
from PIL import Image
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from pxr import Usd, UsdGeom, UsdPhysics

import robot_lab.tasks  # noqa: F401
from source.data import EpisodeRecorder, load_task
from source.navigation.adapters.frame_utils import wrap_yaw
from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter
from source.navigation.adapters.stall_detector import NavigationStallDetector
from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper, write_visual_prim_wrapper
from source.navigation.adapters.yaw_align import (
    YawAlignConfig,
    YawAlignStallDetector,
    body_goal_forward_projection,
    compute_yaw_align_command,
)
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


def _update_follow_camera(env) -> None:
    """Keep the Isaac Lab viewport camera near the robot for GUI debugging."""

    if not args_cli.follow_camera:
        return
    try:
        import torch
        import isaaclab.utils.math as math_utils

        runtime = env.unwrapped
        controller = getattr(runtime, "viewport_camera_controller", None)
        if controller is None:
            return
        robot = runtime.scene["robot"]
        robot_pos = robot.data.root_pos_w[0]
        robot_quat = robot.data.root_quat_w[0]
        offset = torch.tensor(
            [
                -float(args_cli.follow_camera_distance),
                float(args_cli.follow_camera_side),
                float(args_cli.follow_camera_height),
            ],
            dtype=torch.float32,
            device=runtime.device,
        )
        eye = math_utils.transform_points(
            offset.unsqueeze(0),
            pos=robot_pos.unsqueeze(0),
            quat=robot_quat.unsqueeze(0),
        ).squeeze(0)
        if not hasattr(_update_follow_camera, "_smooth_eye"):
            _update_follow_camera._smooth_eye = eye
        else:
            _update_follow_camera._smooth_eye = 0.85 * _update_follow_camera._smooth_eye + 0.15 * eye
        lookat = robot_pos.clone()
        lookat[2] = lookat[2] + 0.20
        controller.set_view_env_index(env_index=0)
        controller.update_view_location(
            eye=_update_follow_camera._smooth_eye.detach().cpu().numpy(),
            lookat=lookat.detach().cpu().numpy(),
        )
    except (AttributeError, KeyError, RuntimeError, TypeError):
        return


def _disable_event(env_cfg, name: str) -> None:
    if hasattr(env_cfg.events, name):
        setattr(env_cfg.events, name, None)


def _set_material_ranges(env_cfg) -> None:
    event = getattr(env_cfg.events, "randomize_rigid_body_material", None)
    if event is None:
        return
    event.params["static_friction_range"] = (1.0, 1.0)
    event.params["dynamic_friction_range"] = (1.0, 1.0)
    event.params["restitution_range"] = (0.0, 0.0)


def _validate_scene_collision(scene_usd: Path, prim_path: str) -> None:
    """Fail fast if the scene collision payload is missing or empty."""

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open scene USD: {scene_usd}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene collision prim does not exist: {prim_path} in {scene_usd}")

    mesh_count = 0
    collision_api_count = 0
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if child.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
    print(
        f"[INFO] Scene collision preflight: prim={prim_path} "
        f"meshes={mesh_count} collision_api={collision_api_count}"
    )
    if mesh_count == 0:
        raise RuntimeError(
            f"Scene collision prim {prim_path} has no mesh geometry. "
            "Check that external USD payloads are mounted, especially "
            "/mnt/sage_data/sage3d/single_scene/839920/collision/839920/839920_collision.usd."
        )


def _validate_scene_prim(scene_usd: Path, prim_path: str, label: str) -> None:
    """Fail fast if an optional scene prim path is not present."""

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open scene USD: {scene_usd}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene {label} prim does not exist: {prim_path} in {scene_usd}")
    print(f"[INFO] Scene {label} preflight: prim={prim_path} type={prim.GetTypeName() or 'typeless'}")


def _configure_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    scene_usd: Path,
    start_pose: tuple[float, float, float],
) -> None:
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if not args_cli.flat_terrain:
        terrain_usd = write_collision_terrain_wrapper(scene_usd, args_cli.terrain_prim_path)
        print(f"[INFO] Navigation terrain wrapper: {terrain_usd} -> {scene_usd}<{args_cli.terrain_prim_path}>")
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/scene_collision",
            terrain_type="usd",
            usd_path=str(terrain_usd),
            debug_vis=False,
        )
        if args_cli.add_nav_ground:
            env_cfg.scene.nav_ground = AssetBaseCfg(
                prim_path="/World/nav_ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.ground_height)),
                spawn=sim_utils.GroundPlaneCfg(),
            )
    if args_cli.load_visual_scene:
        visual_usd = write_visual_prim_wrapper(scene_usd, args_cli.visual_prim_path)
        print(f"[INFO] Navigation visual wrapper: {visual_usd} -> {scene_usd}<{args_cli.visual_prim_path}>")
        env_cfg.scene.visual_scene = AssetBaseCfg(
            prim_path="/World/nav_visual_scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(visual_usd)),
        )
    if args_cli.disable_sky_light:
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
    _set_material_ranges(env_cfg)
    for event_name in (
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "push_robot",
        "randomize_push_robot",
        "randomize_actuator_gains",
    ):
        _disable_event(env_cfg, event_name)
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    for curriculum_name in ("terrain_levels", "command_levels_lin_vel", "command_levels_ang_vel"):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)
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


def _yaw_align_config() -> YawAlignConfig:
    """Build terminal yaw-alignment config from CLI args."""

    return YawAlignConfig(
        kp=args_cli.yaw_align_kp,
        min_wz=args_cli.yaw_align_min_wz,
        max_wz=args_cli.yaw_align_max_wz,
        activation_vx=args_cli.yaw_align_vx,
        activation_yaw_error=args_cli.yaw_align_activation_yaw_error,
        allow_reverse=args_cli.yaw_align_allow_reverse,
    )


def _yaw_align_command(
    pose: tuple[float, float, float],
    goal,
    config: YawAlignConfig,
) -> tuple[float, float, float]:
    body_goal_x = body_goal_forward_projection(pose, (goal.x, goal.y))
    return compute_yaw_align_command(
        yaw_error=wrap_yaw(goal.yaw - pose[2]),
        yaw_tolerance=args_cli.goal_yaw_tolerance,
        body_goal_x=body_goal_x,
        config=config,
    )


def _settle_with_yaw_hold(
    *,
    adapter: Go2LocomotionAdapter,
    recorder: EpisodeRecorder,
    task,
    goal,
    dt: float,
    start_step: int,
    config: YawAlignConfig,
) -> tuple[bool, int]:
    """Settle the base while keeping the terminal yaw inside tolerance."""

    stable_count = 0
    steps = max(0, args_cli.settle_steps)
    required_stable_steps = max(1, args_cli.yaw_settle_stable_steps)
    for settle_step in range(steps):
        pose = adapter.get_base_pose()
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        if abs(yaw_error) <= args_cli.goal_yaw_tolerance:
            command = (0.0, 0.0, 0.0)
        else:
            command = _yaw_align_command(pose, goal, config)
        adapter.apply_base_command(*command)
        adapter.step()
        _update_follow_camera(adapter.env)

        pose_after = adapter.get_base_pose()
        yaw_error_after = wrap_yaw(goal.yaw - pose_after[2])
        stable = adapter.is_stable()
        if stable and abs(yaw_error_after) <= args_cli.goal_yaw_tolerance:
            stable_count += 1
            if stable_count >= required_stable_steps:
                return True, settle_step + 1
        else:
            stable_count = 0

        if settle_step % task.recording.save_every_n_steps == 0:
            recorder.record(
                "yaw_align",
                adapter.snapshot(timestamp=(start_step + settle_step) * dt, phase="yaw_align"),
                front_image=_jpeg_bytes(adapter.get_front_rgb()),
            )
    return False, steps


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    task_path = _project_path(args_cli.task_json)
    task = load_task(task_path)
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    scene_usd = _project_path(args_cli.scene_usd or task.scene_usd)
    nav_map = _project_path(args_cli.nav_map or task.nav_map)
    dataset_dir = _project_path(args_cli.dataset_dir or task.recording.dataset_dir)
    nav_result_path = Path(args_cli.nav_result).expanduser().resolve()

    if not args_cli.flat_terrain:
        _validate_scene_collision(scene_usd, args_cli.terrain_prim_path)
    if args_cli.load_visual_scene:
        _validate_scene_prim(scene_usd, args_cli.visual_prim_path, "visual")
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
    _update_follow_camera(env)
    adapter.settle(args_cli.settle_steps)
    _update_follow_camera(env)
    settled_pose = adapter.get_base_pose()
    dwa_cfg = DWAConfig(
        control_dt=dt,
        lookahead_distance=args_cli.lookahead_distance,
        prediction_horizon=args_cli.prediction_horizon,
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
    if len(path_world) >= 2:
        first_heading = math.atan2(path_world[1][1] - path_world[0][1], path_world[1][0] - path_world[0][0])
        start_heading_error = wrap_yaw(first_heading - task.start.yaw)
        print(
            f"[INFO] Start heading check: start_yaw={task.start.yaw:.3f} "
            f"first_path_heading={first_heading:.3f} error={start_heading_error:.3f}"
        )
        if abs(start_heading_error) > 0.75:
            print(
                "[WARN] Start yaw is far from the first path segment. "
                "The locomotion policy may drift during the initial turn in narrow passages."
            )

    started_at = time.time()
    success = False
    failure_reason = ""
    yaw_error = wrap_yaw(goal.yaw - settled_pose[2])
    final_phase = "nav"
    yaw_align_config = _yaw_align_config()
    yaw_stall_detector = YawAlignStallDetector(
        window_steps=args_cli.yaw_align_stall_window_steps,
        min_progress_rad=args_cli.yaw_align_min_progress,
    )
    yaw_stall_diagnostics = yaw_stall_detector.diagnostics()
    yaw_align_stall_detected = False
    stall_detector = NavigationStallDetector(
        window_steps=max(2, args_cli.stall_window_steps),
        min_progress_m=args_cli.stall_min_progress,
        min_forward_command=args_cli.stall_min_forward_command,
        min_forward_ratio=args_cli.stall_min_forward_ratio,
    )
    stall_diagnostics = stall_detector.diagnostics()
    last_nav_step = 0
    for step in range(args_cli.max_nav_steps):
        last_nav_step = step
        pose = adapter.get_base_pose()
        speed = adapter.get_base_velocity()
        distance = math.hypot(pose[0] - goal.x, pose[1] - goal.y)
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        planner_debug = None
        if distance <= args_cli.goal_tolerance:
            if final_phase != "yaw_align":
                yaw_stall_detector.reset()
            final_phase = "yaw_align"
            if abs(yaw_error) <= args_cli.goal_yaw_tolerance:
                success = True
                break
            command = _yaw_align_command(pose, goal, yaw_align_config)
            yaw_stalled, yaw_stall_diagnostics = yaw_stall_detector.update(abs(yaw_error))
            if yaw_stalled:
                failure_reason = "yaw_align_failed"
                yaw_align_stall_detected = True
                print(
                    f"[nav] yaw align stalled: error_reduction={yaw_stall_diagnostics.error_reduction:.3f}rad "
                    f"within {yaw_stall_diagnostics.sample_count} steps; "
                    f"current_abs_error={yaw_stall_diagnostics.current_abs_error:.3f}rad"
                )
                break
        elif args_cli.debug_command is not None:
            yaw_stall_detector.reset()
            command = tuple(float(value) for value in args_cli.debug_command)
        else:
            yaw_stall_detector.reset()
            vx, vy, wz, planner_debug = planner.compute_command_with_debug(pose, speed, path_world)
            command = (vx, vy, wz)
        if final_phase == "nav":
            stalled, stall_diagnostics = stall_detector.update(pose[0], pose[1], float(command[0]))
            if stalled:
                failure_reason = "nav_collision"
                print(
                    f"[nav] stalled: max_displacement={stall_diagnostics.max_displacement_m:.3f}m "
                    f"within {stall_diagnostics.sample_count} steps; "
                    f"forward_command_ratio={stall_diagnostics.forward_command_ratio:.3f}"
                )
                break
        else:
            stall_detector.reset()
        adapter.apply_base_command(*command)
        adapter.step()
        _update_follow_camera(env)
        if step % task.recording.save_every_n_steps == 0:
            recorder.record(final_phase, adapter.snapshot(timestamp=step * dt, phase=final_phase), front_image=_jpeg_bytes(adapter.get_front_rgb()))
        if args_cli.debug_print_every > 0 and step % args_cli.debug_print_every == 0:
            diagnostics = adapter.diagnostics()
            planner_detail = ""
            if planner_debug is not None:
                planner_detail = (
                    f" dwa=(clearance={planner_debug.clearance:.3f}, "
                    f"feasible={planner_debug.feasible_candidates}, "
                    f"collision_rej={planner_debug.collision_rejections}, "
                    f"target=({planner_debug.target_point[0]:.3f}, {planner_debug.target_point[1]:.3f}))"
                )
            print(
                f"[nav] step={step} phase={final_phase} pose=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) "
                f"goal_dist={distance:.3f} yaw_error={yaw_error:.3f} cmd={command} "
                f"measured_vx={diagnostics['measured_vx']:.3f} measured_wz={diagnostics['measured_wz']:.3f} "
                f"base_z={diagnostics['base_z']:.3f} "
                f"roll_pitch=({diagnostics['base_roll']:.3f}, {diagnostics['base_pitch']:.3f}) "
                f"contact=(foot={diagnostics.get('foot_contact_force_max', 0.0):.1f}, "
                f"nonfoot={diagnostics.get('nonfoot_contact_force_max', 0.0):.1f}) "
                f"action_abs_max={diagnostics.get('action_abs_max', 0.0):.3f} "
                f"dog_action_abs_mean={diagnostics.get('dog_action_abs_mean', 0.0):.3f} "
                f"command_seen=({diagnostics['command_seen_vx']:.3f}, {diagnostics['command_seen_wz']:.3f}) "
                f"stall_window=({stall_diagnostics.sample_count}, "
                f"{stall_diagnostics.max_displacement_m:.3f}, {stall_diagnostics.forward_command_ratio:.3f}) "
                f"yaw_window=({yaw_stall_diagnostics.sample_count}, "
                f"{yaw_stall_diagnostics.error_reduction:.3f}, "
                f"{yaw_stall_diagnostics.current_abs_error:.3f})"
                f"{planner_detail}"
            )
        if args_cli.real_time:
            time.sleep(max(0.0, dt))
    else:
        failure_reason = "yaw_align_failed" if final_phase == "yaw_align" else "nav_timeout"

    yaw_settle_success = False
    yaw_settle_steps = 0
    if success:
        yaw_settle_success, yaw_settle_steps = _settle_with_yaw_hold(
            adapter=adapter,
            recorder=recorder,
            task=task,
            goal=goal,
            dt=dt,
            start_step=last_nav_step + 1,
            config=yaw_align_config,
        )
    else:
        adapter.settle(args_cli.settle_steps)
        yaw_settle_steps = args_cli.settle_steps
    stable = adapter.is_stable()
    final_vx, final_vy, final_wz = adapter.get_base_velocity_full()
    final_pose = adapter.get_base_pose_full()
    final_distance = math.hypot(final_pose["x"] - goal.x, final_pose["y"] - goal.y)
    yaw_error = wrap_yaw(goal.yaw - final_pose["yaw"])
    final_position_reached = final_distance <= args_cli.goal_tolerance
    final_yaw_aligned = abs(yaw_error) <= args_cli.goal_yaw_tolerance
    if stable and final_yaw_aligned:
        yaw_settle_success = True
    if success and final_distance > args_cli.goal_tolerance:
        success = False
        failure_reason = "nav_timeout"
    elif success and abs(yaw_error) > args_cli.goal_yaw_tolerance:
        success = False
        failure_reason = "yaw_align_failed"
    elif success and not stable:
        success = False
        failure_reason = "base_not_stable"
    elif not success and not failure_reason:
        failure_reason = "yaw_align_failed" if final_phase == "yaw_align" else "nav_timeout"

    result = {
        "schema_version": 1,
        "success": success,
        "failure_reason": failure_reason,
        "final_base_pose_world": final_pose,
        "goal_xyyaw": [goal.x, goal.y, goal.yaw],
        "final_goal_distance": final_distance,
        "yaw_error": yaw_error,
        "final_position_reached": final_position_reached,
        "final_yaw_aligned": final_yaw_aligned,
        "base_stable": stable,
        "final_body_velocity": {
            "vx": final_vx,
            "vy": final_vy,
            "wz": final_wz,
        },
        "path_length": _path_length(path_world),
        "timeout": failure_reason == "nav_timeout",
        "stall_detected": failure_reason == "nav_collision",
        "yaw_align_stall_detected": yaw_align_stall_detected,
        "yaw_settle_success": yaw_settle_success,
        "yaw_settle_steps": yaw_settle_steps,
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
