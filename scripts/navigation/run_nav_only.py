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
parser.add_argument("--inflate-radius", type=float, default=0.25)
parser.add_argument("--local-clearance-radius", type=float, default=0.20)
parser.add_argument("--goal-tolerance", type=float, default=0.15)
parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
parser.add_argument("--max-nav-steps", type=int, default=5000)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--stall-window-steps", type=int, default=240)
parser.add_argument("--stall-min-progress", type=float, default=0.05)
parser.add_argument("--stall-min-forward-command", type=float, default=0.05)
parser.add_argument("--stall-min-forward-ratio", type=float, default=0.25)
parser.add_argument("--lookahead-distance", type=float, default=0.35)
parser.add_argument("--prediction-horizon", type=float, default=0.90)
parser.add_argument("--max-lin-vel", type=float, default=0.50)
parser.add_argument("--max-ang-vel", type=float, default=1.00)
parser.add_argument("--brisk-nav", action="store_true", help="Use a more aggressive DWA speed profile for open routes.")
parser.add_argument("--min-active-lin-vel", type=float, default=0.30)
parser.add_argument("--near-goal-min-active-lin-vel", type=float, default=0.22)
parser.add_argument("--close-goal-speed-limit", type=float, default=0.22)
parser.add_argument("--speed-bias", type=float, default=0.35)
parser.add_argument("--max-linear-accel", type=float, default=2.5)
parser.add_argument("--yaw-align-kp", type=float, default=2.0)
parser.add_argument("--yaw-align-min-wz", type=float, default=0.75)
parser.add_argument("--yaw-align-max-wz", type=float, default=1.00)
parser.add_argument("--yaw-align-vx", type=float, default=0.16)
parser.add_argument("--yaw-align-max-vx", type=float, default=0.35)
parser.add_argument("--yaw-align-position-kp", type=float, default=0.8)
parser.add_argument("--yaw-align-max-vy", type=float, default=0.18)
parser.add_argument("--yaw-align-lateral-kp", type=float, default=0.8)
parser.add_argument("--yaw-align-lateral-deadband", type=float, default=0.03)
parser.add_argument(
    "--yaw-align-start-distance",
    type=float,
    default=0.65,
    help="Start terminal pose alignment before reaching the position tolerance.",
)
parser.add_argument("--yaw-align-activation-yaw-error", type=float, default=0.0)
parser.add_argument("--yaw-align-allow-reverse", action="store_true")
parser.add_argument("--yaw-align-stall-window-steps", type=int, default=240)
parser.add_argument("--yaw-align-min-progress", type=float, default=0.08)
parser.add_argument("--yaw-settle-stable-steps", type=int, default=20)
parser.add_argument("--yaw-settle-kp", type=float, default=0.8)
parser.add_argument("--yaw-settle-min-wz", type=float, default=0.0)
parser.add_argument("--yaw-settle-max-wz", type=float, default=0.35)
parser.add_argument("--yaw-settle-realign-margin", type=float, default=0.08)
parser.add_argument("--head-camera", action="store_true")
parser.add_argument("--head-camera-height", type=int, default=480)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--save-replay-trajectory", action="store_true", help="Save root and joint states for offline replay.")
parser.add_argument("--replay-sample-every", type=int, default=1, help="Save one replay frame every N simulation steps.")
parser.add_argument("--replay-output", default=None, help="Explicit replay JSONL output path.")
parser.add_argument(
    "--replay-include-initial-settle",
    action="store_true",
    help="Include the pre-navigation zero-command settle segment in the replay trajectory.",
)
parser.add_argument(
    "--replay-trajectory-name",
    default="trajectory.jsonl",
    help="Replay file name under <episode_dir>/replay when --replay-output is omitted.",
)
parser.add_argument(
    "--load-visual-scene",
    action="store_true",
    help="Load a visual-only scene prim into the Isaac Lab viewport for debugging/demo.",
)
parser.add_argument(
    "--hide-nav-collision-visual",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Hide /World/nav_collision/terrain in the viewport while keeping the "
        "navigation collision prim active for physics. Defaults to enabled when "
        "--load-visual-scene is used."
    ),
)
parser.add_argument(
    "--visual-load-mode",
    choices=("sublayer", "reference"),
    default="reference",
    help=(
        "Use the stable referenced visual asset by default, or preload a full-scene "
        "sublayer for SAGE visuals before Isaac Lab creates PhysX tensor views."
    ),
)
parser.add_argument("--visual-prim-path", default="/World/gauss", help="Visual prim referenced when --load-visual-scene is set.")
parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--follow-camera-mode", choices=("chase", "front", "overhead", "fixed", "stage"), default="chase")
parser.add_argument(
    "--viewport-camera-prim",
    default="/World/Camera_main",
    help="USD Camera prim used when --follow-camera-mode stage is selected.",
)
parser.add_argument("--follow-camera-distance", type=float, default=2.4)
parser.add_argument("--follow-camera-height", type=float, default=0.8)
parser.add_argument("--follow-camera-side", type=float, default=0.0)
parser.add_argument(
    "--fixed-camera-preset",
    choices=("start", "goal", "route"),
    default="start",
    help="Automatic fixed-camera placement when explicit eye/lookat are not provided.",
)
parser.add_argument("--fixed-camera-close-distance", type=float, default=2.2)
parser.add_argument("--fixed-camera-close-height", type=float, default=1.35)
parser.add_argument("--fixed-camera-close-side", type=float, default=-0.75)
parser.add_argument(
    "--fixed-camera-eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="World-space viewport camera eye used once when --follow-camera-mode fixed is selected.",
)
parser.add_argument(
    "--fixed-camera-lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="World-space viewport camera target used once when --follow-camera-mode fixed is selected.",
)
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
from pxr import Tf, Usd, UsdGeom, UsdPhysics

import robot_lab.tasks  # noqa: F401
from source.data import EpisodeRecorder, load_task
from source.navigation.adapters.frame_utils import wrap_yaw
from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter
from source.navigation.adapters.stall_detector import NavigationStallDetector
from source.navigation.adapters.terrain_utils import (
    write_collision_terrain_wrapper,
    write_visual_prim_wrapper,
    write_visual_sublayer_wrapper,
)
from source.navigation.adapters.yaw_align import (
    YawAlignConfig,
    YawAlignStallDetector,
    body_goal_components,
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


def _tensor_list(value) -> list:
    """Convert a scalar/tensor-like value to a JSON-serializable list."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _capture_replay_frame(env, *, timestamp: float, step: int, phase: str, command: tuple[float, float, float]) -> dict:
    """Capture one complete robot state frame for offline replay."""

    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    return {
        "schema_version": 1,
        "timestamp": float(timestamp),
        "step": int(step),
        "phase": str(phase),
        "command": [float(command[0]), float(command[1]), float(command[2])],
        "root_pos_w": _tensor_list(robot.data.root_pos_w[0]),
        "root_quat_w": _tensor_list(robot.data.root_quat_w[0]),
        "root_lin_vel_w": _tensor_list(robot.data.root_lin_vel_w[0]),
        "root_ang_vel_w": _tensor_list(robot.data.root_ang_vel_w[0]),
        "joint_names": list(robot.joint_names),
        "joint_pos": _tensor_list(robot.data.joint_pos[0]),
        "joint_vel": _tensor_list(robot.data.joint_vel[0]),
    }


class ReplayTrajectoryRecorder:
    """Buffer navigation states and write them as JSONL for offline replay."""

    def __init__(self, *, enabled: bool, output_path: Path | None, sample_every: int):
        self.enabled = bool(enabled)
        self.output_path = output_path
        self.sample_every = max(1, int(sample_every))
        self.frames: list[dict] = []

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def capture(self, env, *, timestamp: float, step: int, phase: str, command: tuple[float, float, float]) -> None:
        if not self.enabled or step % self.sample_every != 0:
            return
        self.frames.append(_capture_replay_frame(env, timestamp=timestamp, step=step, phase=phase, command=command))

    def write(self) -> None:
        if not self.enabled:
            return
        if self.output_path is None:
            raise RuntimeError("Replay trajectory recorder is enabled but output_path is None.")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as stream:
            for frame in self.frames:
                stream.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        print(f"[INFO] Wrote replay trajectory: {self.output_path} frames={self.frame_count}")


def _dwa_config_from_args(control_dt: float) -> DWAConfig:
    """Build DWA config, optionally applying the brisk navigation profile."""

    if args_cli.brisk_nav:
        max_linear_velocity = max(args_cli.max_lin_vel, 0.80)
        min_active_linear_velocity = max(args_cli.min_active_lin_vel, 0.55)
        near_goal_min_active_linear_velocity = max(args_cli.near_goal_min_active_lin_vel, 0.38)
        close_goal_speed_limit = max(args_cli.close_goal_speed_limit, 0.35)
        speed_bias = max(args_cli.speed_bias, 1.10)
        max_linear_accel = max(args_cli.max_linear_accel, 4.5)
    else:
        max_linear_velocity = args_cli.max_lin_vel
        min_active_linear_velocity = args_cli.min_active_lin_vel
        near_goal_min_active_linear_velocity = args_cli.near_goal_min_active_lin_vel
        close_goal_speed_limit = args_cli.close_goal_speed_limit
        speed_bias = args_cli.speed_bias
        max_linear_accel = args_cli.max_linear_accel
    return DWAConfig(
        control_dt=control_dt,
        lookahead_distance=args_cli.lookahead_distance,
        prediction_horizon=args_cli.prediction_horizon,
        goal_tolerance=args_cli.goal_tolerance,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=args_cli.max_ang_vel,
        min_active_linear_velocity=min_active_linear_velocity,
        near_goal_min_active_linear_velocity=near_goal_min_active_linear_velocity,
        close_goal_speed_limit=close_goal_speed_limit,
        speed_bias=speed_bias,
        max_linear_accel=max_linear_accel,
    )


def _load_visual_scene_sublayer(scene_usd: Path, visual_exclude_prim_paths: list[str] | tuple[str, ...]) -> bool:
    """Add the complete SAGE scene as a display-only sublayer before env creation.

    This must run before Isaac Lab creates the environment. Appending a sublayer
    after PhysX tensor views exist invalidates the simulation view and breaks the
    first reset when joint positions are written.
    """

    try:
        import omni.usd
    except ImportError as exc:
        raise RuntimeError("Cannot load full-scene visual sublayer because omni.usd is unavailable.") from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Cannot load full-scene visual sublayer before an Isaac stage exists.")
        return False
    wrapper = write_visual_sublayer_wrapper(
        scene_usd,
        args_cli.visual_prim_path,
        excluded_prim_paths=visual_exclude_prim_paths,
    )
    root_layer = stage.GetRootLayer()
    wrapper_path = str(wrapper)
    if wrapper_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(wrapper_path)
    print(
        f"[INFO] Navigation visual sublayer: {wrapper} -> {scene_usd} "
        f"with visible prim {args_cli.visual_prim_path}"
    )
    return True


def _jpeg_bytes(rgb_tensor) -> bytes | None:
    if rgb_tensor is None:
        return None
    stream = io.BytesIO()
    Image.fromarray(rgb_tensor.cpu().numpy()).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def _configure_default_fixed_camera(task) -> None:
    """Choose a fixed camera from the task when explicit coordinates are unset."""

    if hasattr(_update_follow_camera, "_fixed_camera_applied"):
        delattr(_update_follow_camera, "_fixed_camera_applied")
    if args_cli.fixed_camera_eye is not None and args_cli.fixed_camera_lookat is not None:
        return
    start_x = float(task.start.x)
    start_y = float(task.start.y)
    goal_x = float(task.pick.base_goal.x)
    goal_y = float(task.pick.base_goal.y)
    preset = args_cli.fixed_camera_preset
    if preset == "route":
        center_x = 0.5 * (start_x + goal_x)
        center_y = 0.5 * (start_y + goal_y)
        span = max(math.hypot(goal_x - start_x, goal_y - start_y), 1.0)
        height = min(max(0.60 * span + 2.5, 3.5), 7.0)
        eye = [
            center_x - max(0.35 * span, 1.8),
            center_y - max(0.65 * span, 3.0),
            height,
        ]
        lookat = [center_x, center_y, 0.35]
    else:
        if preset == "goal":
            focus_x = goal_x
            focus_y = goal_y
            yaw = float(task.pick.base_goal.yaw)
        else:
            focus_x = start_x
            focus_y = start_y
            yaw = float(task.start.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        left_x = -math.sin(yaw)
        left_y = math.cos(yaw)
        distance = float(args_cli.fixed_camera_close_distance)
        side = float(args_cli.fixed_camera_close_side)
        eye = [
            focus_x - forward_x * distance + left_x * side,
            focus_y - forward_y * distance + left_y * side,
            float(args_cli.fixed_camera_close_height),
        ]
        lookat = [focus_x + 0.35 * forward_x, focus_y + 0.35 * forward_y, 0.45]
    if args_cli.fixed_camera_eye is None:
        args_cli.fixed_camera_eye = eye
    if args_cli.fixed_camera_lookat is None:
        args_cli.fixed_camera_lookat = lookat


def _update_follow_camera(env) -> None:
    """Update the Isaac Lab viewport camera for GUI debugging."""

    if not args_cli.follow_camera:
        return
    try:
        import torch
        import isaaclab.utils.math as math_utils

        runtime = env.unwrapped
        controller = getattr(runtime, "viewport_camera_controller", None)
        if controller is None:
            return
        mode = args_cli.follow_camera_mode
        if mode == "stage":
            if getattr(_update_follow_camera, "_stage_camera_applied", False):
                return
            if _set_viewport_stage_camera(args_cli.viewport_camera_prim):
                _update_follow_camera._stage_camera_applied = True
            return
        if mode == "fixed":
            if getattr(_update_follow_camera, "_fixed_camera_applied", False):
                return
            if args_cli.fixed_camera_eye is None or args_cli.fixed_camera_lookat is None:
                return
            eye = torch.tensor(args_cli.fixed_camera_eye, dtype=torch.float32, device=runtime.device)
            lookat = torch.tensor(args_cli.fixed_camera_lookat, dtype=torch.float32, device=runtime.device)
            controller.set_view_env_index(env_index=0)
            controller.update_view_location(
                eye=eye.detach().cpu().numpy(),
                lookat=lookat.detach().cpu().numpy(),
            )
            _update_follow_camera._fixed_camera_applied = True
            return

        robot = runtime.scene["robot"]
        robot_pos = robot.data.root_pos_w[0]
        robot_quat = robot.data.root_quat_w[0]
        if mode == "overhead":
            eye = robot_pos + torch.tensor(
                [
                    -0.35 * float(args_cli.follow_camera_distance),
                    float(args_cli.follow_camera_side),
                    float(args_cli.follow_camera_height),
                ],
                dtype=torch.float32,
                device=runtime.device,
            )
        else:
            direction = 1.0 if mode == "front" else -1.0
            offset = torch.tensor(
                [
                    direction * float(args_cli.follow_camera_distance),
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


def _candidate_stage_camera_paths(camera_prim_path: str) -> list[str]:
    """Return camera prim path candidates for sublayer and referenced visual modes."""

    candidates = [camera_prim_path]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path.startswith("/World/"):
        for candidate in tuple(candidates):
            candidates.append("/World/nav_visual_scene/" + candidate.removeprefix("/World/"))
    return list(dict.fromkeys(candidates))


def _set_viewport_stage_camera(camera_prim_path: str) -> bool:
    """Switch the active viewport to a USD camera prim authored in the stage."""

    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

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
        print(f"[INFO] Viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set stage camera {camera_prim_path}: {exc}")
        return False


def _hide_nav_collision_visual() -> None:
    """Hide navigation collision geometry in the viewport while keeping physics active."""

    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Cannot hide navigation collision visual: no active USD stage.")
            return

        hidden_any = False
        for prim_path in ("/World/nav_collision/terrain", "/World/nav_collision"):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"[WARN] Navigation collision visual prim not found: {prim_path}")
                continue
            if not prim.IsA(UsdGeom.Imageable):
                print(f"[WARN] Navigation collision visual prim is not imageable: {prim_path}")
                continue

            UsdGeom.Imageable(prim).MakeInvisible()
            print(f"[INFO] Hid navigation collision visual: {prim_path}")
            hidden_any = True

        if not hidden_any:
            print("[WARN] No navigation collision visual prim found to hide.")
    except Exception as exc:
        print(f"[WARN] Failed to hide navigation collision visual: {exc}")


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


def _open_scene_stage(scene_usd: Path) -> Usd.Stage:
    """Open a scene USD with actionable diagnostics for missing/truncated files."""

    if not scene_usd.exists():
        raise RuntimeError(f"Scene USD does not exist: {scene_usd}")
    size_bytes = scene_usd.stat().st_size
    if size_bytes <= 0:
        raise RuntimeError(
            f"Scene USD is empty: {scene_usd}. "
            "Regenerate or restore the scene before running navigation."
        )
    try:
        stage = Usd.Stage.Open(str(scene_usd))
    except Tf.ErrorException as exc:
        raise RuntimeError(
            f"Failed to open scene USD: {scene_usd} ({size_bytes} bytes). "
            "Check that the file is a valid USD and that external SAGE assets are mounted."
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"Failed to open scene USD: {scene_usd} ({size_bytes} bytes). "
            "Check that the file is a valid USD and that external SAGE assets are mounted."
        )
    return stage


def _validate_scene_collision(scene_usd: Path, prim_path: str) -> None:
    """Fail fast if the scene collision payload is missing or empty."""

    stage = _open_scene_stage(scene_usd)
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

    stage = _open_scene_stage(scene_usd)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Scene {label} prim does not exist: {prim_path} in {scene_usd}")
    print(f"[INFO] Scene {label} preflight: prim={prim_path} type={prim.GetTypeName() or 'typeless'}")


def _configure_env(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    scene_usd: Path,
    start_pose: tuple[float, float, float],
    visual_exclude_prim_paths: list[str] | tuple[str, ...] = (),
) -> None:
    env_cfg.scene.num_envs = 1
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if not args_cli.flat_terrain:
        terrain_usd = write_collision_terrain_wrapper(scene_usd, args_cli.terrain_prim_path)
        print(f"[INFO] Navigation terrain wrapper: {terrain_usd} -> {scene_usd}<{args_cli.terrain_prim_path}>")
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/nav_collision",
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
    if args_cli.load_visual_scene and args_cli.visual_load_mode == "reference":
        visual_usd = write_visual_prim_wrapper(
            scene_usd,
            args_cli.visual_prim_path,
            excluded_prim_paths=visual_exclude_prim_paths,
        )
        print(
            f"[INFO] Navigation visual wrapper: {visual_usd} -> {scene_usd} "
            f"with visible prim {args_cli.visual_prim_path}"
        )
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


def _yaw_settle_config() -> YawAlignConfig:
    """Build a low-gain yaw correction config for terminal base settling."""

    return YawAlignConfig(
        kp=args_cli.yaw_settle_kp,
        min_wz=args_cli.yaw_settle_min_wz,
        max_wz=args_cli.yaw_settle_max_wz,
        activation_vx=0.0,
        activation_yaw_error=0.0,
        allow_reverse=False,
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


def _terminal_pose_command(
    pose: tuple[float, float, float],
    goal,
    config: YawAlignConfig,
) -> tuple[float, float, float]:
    """Drive the base toward the final pose without handing control back to DWA."""

    body_goal_x, body_goal_y = body_goal_components(pose, (goal.x, goal.y))
    yaw_error = wrap_yaw(goal.yaw - pose[2])
    distance = math.hypot(goal.x - pose[0], goal.y - pose[1])

    if distance <= args_cli.goal_tolerance or abs(body_goal_x) <= args_cli.yaw_align_lateral_deadband:
        vx = 0.0
    elif body_goal_x < -1.0e-3:
        vx = -args_cli.yaw_align_vx if args_cli.yaw_align_allow_reverse else 0.0
    else:
        position_vx = args_cli.yaw_align_position_kp * body_goal_x
        vx = min(args_cli.yaw_align_max_vx, max(0.0, position_vx))
        if abs(body_goal_x) >= abs(body_goal_y) and 0.0 < vx < args_cli.yaw_align_vx:
            vx = min(args_cli.yaw_align_max_vx, args_cli.yaw_align_vx)

    if distance <= args_cli.goal_tolerance or abs(body_goal_y) <= args_cli.yaw_align_lateral_deadband:
        vy = 0.0
    else:
        position_vy = args_cli.yaw_align_lateral_kp * body_goal_y
        vy = max(-args_cli.yaw_align_max_vy, min(args_cli.yaw_align_max_vy, position_vy))

    if abs(yaw_error) <= args_cli.goal_yaw_tolerance:
        wz = 0.0
    else:
        wz_abs = min(config.max_wz, max(config.kp * abs(yaw_error), config.min_wz))
        wz = math.copysign(wz_abs, yaw_error)
    return vx, vy, wz


def _settle_with_yaw_hold(
    *,
    adapter: Go2LocomotionAdapter,
    recorder: EpisodeRecorder,
    replay_recorder: ReplayTrajectoryRecorder,
    task,
    goal,
    dt: float,
    start_step: int,
    replay_step_start: int,
) -> tuple[bool, int, int]:
    """Settle the base while keeping the terminal yaw inside tolerance."""

    stable_count = 0
    steps = max(0, args_cli.settle_steps)
    required_stable_steps = max(1, args_cli.yaw_settle_stable_steps)
    settle_config = _yaw_settle_config()
    realign_tolerance = args_cli.goal_yaw_tolerance + max(0.0, args_cli.yaw_settle_realign_margin)
    replay_step = replay_step_start
    for settle_step in range(steps):
        pose = adapter.get_base_pose()
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        abs_yaw_error = abs(yaw_error)
        braking_near_goal = not adapter.is_stable() and abs_yaw_error <= realign_tolerance
        if abs_yaw_error <= args_cli.goal_yaw_tolerance or braking_near_goal:
            command = (0.0, 0.0, 0.0)
        else:
            command = _yaw_align_command(pose, goal, settle_config)
        adapter.apply_base_command(*command)
        adapter.step()
        _update_follow_camera(adapter.env)
        replay_recorder.capture(
            adapter.env,
            timestamp=replay_step * dt,
            step=replay_step,
            phase="settle",
            command=command,
        )
        replay_step += 1

        pose_after = adapter.get_base_pose()
        yaw_error_after = wrap_yaw(goal.yaw - pose_after[2])
        stable = adapter.is_stable()
        if stable and abs(yaw_error_after) <= args_cli.goal_yaw_tolerance:
            stable_count += 1
            if stable_count >= required_stable_steps:
                return True, settle_step + 1, replay_step
        else:
            stable_count = 0

        if settle_step % task.recording.save_every_n_steps == 0:
            recorder.record(
                "yaw_align",
                adapter.snapshot(timestamp=(start_step + settle_step) * dt, phase="yaw_align"),
                front_image=_jpeg_bytes(adapter.get_front_rgb()),
            )
    return False, steps, replay_step


def _settle_zero_command(
    *,
    adapter: Go2LocomotionAdapter,
    replay_recorder: ReplayTrajectoryRecorder,
    dt: float,
    replay_step_start: int,
    steps: int,
    phase: str,
    capture: bool = True,
) -> int:
    """Hold zero command and optionally capture replay frames."""

    replay_step = replay_step_start
    command = (0.0, 0.0, 0.0)
    adapter.apply_base_command(*command)
    for _ in range(max(0, steps)):
        adapter.step()
        _update_follow_camera(adapter.env)
        if capture:
            replay_recorder.capture(
                adapter.env,
                timestamp=replay_step * dt,
                step=replay_step,
                phase=phase,
                command=command,
            )
        replay_step += 1
    return replay_step


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    task_path = _project_path(args_cli.task_json)
    task = load_task(task_path)
    _configure_default_fixed_camera(task)
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    scene_usd = _project_path(args_cli.scene_usd or task.scene_usd)
    nav_map = _project_path(args_cli.nav_map or task.nav_map)
    dataset_dir = _project_path(args_cli.dataset_dir or task.recording.dataset_dir)
    nav_result_path = Path(args_cli.nav_result).expanduser().resolve()

    if not args_cli.flat_terrain:
        _validate_scene_collision(scene_usd, args_cli.terrain_prim_path)
    if args_cli.load_visual_scene:
        _validate_scene_prim(scene_usd, args_cli.visual_prim_path, "visual")
    visual_exclude_prim_paths = [
        args_cli.terrain_prim_path,
        "/World/go2_x5",
        "/World/mec_arm_6dof",
    ]
    object_prim_path = getattr(task.pick, "object_prim_path", None)
    if object_prim_path:
        visual_exclude_prim_paths.append(object_prim_path)
    if args_cli.load_visual_scene and args_cli.visual_load_mode == "sublayer":
        if not _load_visual_scene_sublayer(scene_usd, visual_exclude_prim_paths):
            print("[WARN] Falling back to stable visual reference mode.")
            args_cli.visual_load_mode = "reference"
    _configure_env(
        env_cfg,
        scene_usd=scene_usd,
        start_pose=(task.start.x, task.start.y, task.start.yaw),
        visual_exclude_prim_paths=visual_exclude_prim_paths,
    )
    agent_cfg = isaaclab_cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)

    hide_nav_collision_visual_arg = getattr(args_cli, "hide_nav_collision_visual", None)
    hide_nav_collision_visual = (
        hide_nav_collision_visual_arg
        if hide_nav_collision_visual_arg is not None
        else args_cli.load_visual_scene
    )

    if hide_nav_collision_visual:
        _hide_nav_collision_visual()

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
    replay_output_path = (
        Path(args_cli.replay_output).expanduser().resolve()
        if args_cli.replay_output
        else recorder.episode_dir / "replay" / args_cli.replay_trajectory_name
    )
    replay_recorder = ReplayTrajectoryRecorder(
        enabled=args_cli.save_replay_trajectory,
        output_path=replay_output_path,
        sample_every=args_cli.replay_sample_every,
    )

    dt = float(env.unwrapped.step_dt)
    replay_step = 0
    _update_follow_camera(env)
    replay_step = _settle_zero_command(
        adapter=adapter,
        replay_recorder=replay_recorder,
        dt=dt,
        replay_step_start=replay_step,
        steps=args_cli.settle_steps,
        phase="settle",
        capture=args_cli.replay_include_initial_settle,
    )
    _update_follow_camera(env)
    settled_pose = adapter.get_base_pose()
    dwa_cfg = _dwa_config_from_args(control_dt=dt)
    print(
        "[INFO] DWA speed config: "
        f"brisk={args_cli.brisk_nav} max_vx={dwa_cfg.max_linear_velocity:.2f} "
        f"min_active_vx={dwa_cfg.min_active_linear_velocity:.2f} "
        f"close_goal_vx={dwa_cfg.close_goal_speed_limit:.2f} "
        f"speed_bias={dwa_cfg.speed_bias:.2f} max_accel={dwa_cfg.max_linear_accel:.2f}"
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
        if replay_recorder.enabled:
            replay_recorder.write()
            result["replay_trajectory_path"] = str(replay_recorder.output_path)
            result["replay_frame_count"] = replay_recorder.frame_count
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
        terminal_yaw_distance = max(args_cli.goal_tolerance, args_cli.yaw_align_start_distance)
        needs_terminal_yaw = abs(yaw_error) > args_cli.goal_yaw_tolerance
        terminal_pose_active = distance <= terminal_yaw_distance
        if distance <= args_cli.goal_tolerance and not needs_terminal_yaw:
            success = True
            break
        if terminal_pose_active:
            if final_phase != "yaw_align":
                yaw_stall_detector.reset()
            final_phase = "yaw_align"
            command = _terminal_pose_command(pose, goal, yaw_align_config)
            if needs_terminal_yaw:
                yaw_stalled, yaw_stall_diagnostics = yaw_stall_detector.update(abs(yaw_error))
                if yaw_stalled and distance <= args_cli.goal_tolerance:
                    failure_reason = "yaw_align_failed"
                    yaw_align_stall_detected = True
                    print(
                        f"[nav] yaw align stalled: error_reduction={yaw_stall_diagnostics.error_reduction:.3f}rad "
                        f"within {yaw_stall_diagnostics.sample_count} steps; "
                        f"current_abs_error={yaw_stall_diagnostics.current_abs_error:.3f}rad"
                    )
                    break
            else:
                yaw_stall_detector.reset()
        elif args_cli.debug_command is not None:
            yaw_stall_detector.reset()
            final_phase = "nav"
            command = tuple(float(value) for value in args_cli.debug_command)
        else:
            yaw_stall_detector.reset()
            final_phase = "nav"
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
        replay_recorder.capture(
            env,
            timestamp=replay_step * dt,
            step=replay_step,
            phase=final_phase,
            command=command,
        )
        replay_step += 1
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
                f"measured_v=({diagnostics['measured_vx']:.3f}, {diagnostics['measured_vy']:.3f}) "
                f"measured_wz={diagnostics['measured_wz']:.3f} "
                f"base_z={diagnostics['base_z']:.3f} "
                f"roll_pitch=({diagnostics['base_roll']:.3f}, {diagnostics['base_pitch']:.3f}) "
                f"contact=(foot={diagnostics.get('foot_contact_force_max', 0.0):.1f}, "
                f"nonfoot={diagnostics.get('nonfoot_contact_force_max', 0.0):.1f}) "
                f"action_abs_max={diagnostics.get('action_abs_max', 0.0):.3f} "
                f"dog_action_abs_mean={diagnostics.get('dog_action_abs_mean', 0.0):.3f} "
                f"command_seen=({diagnostics['command_seen_vx']:.3f}, "
                f"{diagnostics['command_seen_vy']:.3f}, {diagnostics['command_seen_wz']:.3f}) "
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
        yaw_settle_success, yaw_settle_steps, replay_step = _settle_with_yaw_hold(
            adapter=adapter,
            recorder=recorder,
            replay_recorder=replay_recorder,
            task=task,
            goal=goal,
            dt=dt,
            start_step=last_nav_step + 1,
            replay_step_start=replay_step,
        )
    else:
        replay_step = _settle_zero_command(
            adapter=adapter,
            replay_recorder=replay_recorder,
            dt=dt,
            replay_step_start=replay_step,
            steps=args_cli.settle_steps,
            phase="settle",
        )
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
    if replay_recorder.enabled:
        replay_recorder.write()
        result["replay_trajectory_path"] = str(replay_recorder.output_path)
        result["replay_frame_count"] = replay_recorder.frame_count
    _write_nav_result(nav_result_path, recorder, result)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
