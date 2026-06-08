#!/usr/bin/env python3
"""Run one continuous contact-only nav-pick-carry-place dataset episode."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_LAB_SOURCE = PROJECT_ROOT / "source/robot_lab"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROBOT_LAB_SOURCE))

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/go2_x5/flat/model_8500.pt"
DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
GRIPPER_OPEN_TARGET = (0.043, 0.043)
GRIPPER_CLOSE_TARGET = (0.0, 0.0)
CARRY_POSTURES = {
    "stow": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "carry_front": (0.0, -0.35, 0.65, 0.0, 0.70, 0.0),
    "carry_high": (0.0, -0.55, 0.95, 0.0, 0.85, 0.0),
}
CONTACT_ARM_COMMAND_DT = 0.02
CONTACT_SETTLE_TO_SEGMENT_START_DURATION = 0.35
CONTACT_GRIPPER_MOVE_DURATION = 0.70
CONTACT_GRIPPER_HOLD_DURATION = 0.45
CONTACT_PRE_CLOSE_HOLD_DURATION = 0.10
CONTACT_POST_MOTION_CONVERGENCE_TIMEOUT = 3.00
CONTACT_POST_MOTION_JOINT_ERROR_TOL = 0.050
CONTACT_STRICT_POST_MOTION_WAIT_SEGMENTS = {"move_to_pregrasp", "approach_to_grasp"}
CONTACT_GRIPPER_MIN_CLOSE_PROGRESS = 0.05
CONTACT_OBJECT_RETREAT_SUCCESS_THRESHOLD_M = 0.03


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_checkpoint() -> str:
    env_checkpoint = os.environ.get("GO2_X5_CHECKPOINT")
    if env_checkpoint:
        return env_checkpoint
    return str(DEFAULT_CHECKPOINT)


def _parse_args() -> tuple[argparse.Namespace, list[str], Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", default=None)
    parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
    parser.add_argument("--checkpoint", default=_default_checkpoint())
    parser.add_argument("--dataset-dir", default="/tmp/contact_pick_place_once")
    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--add-nav-ground", action="store_true")
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--load-visual-scene", action="store_true")
    parser.add_argument(
        "--demo-visuals",
        action="store_true",
        help="Enable the visual scene and hide navigation collision geometry for GUI/video runs.",
    )
    parser.add_argument(
        "--hide-nav-collision-visual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Hide /World/nav_collision/terrain in the viewport while keeping it active for physics. "
            "Defaults to enabled when --demo-visuals or --load-visual-scene is used."
        ),
    )
    parser.add_argument("--visual-load-mode", choices=("sublayer", "reference"), default="sublayer")
    parser.add_argument("--visual-prim-path", default="/World/gauss")
    parser.add_argument(
        "--hide-distractor-objects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide other apple/orange/bottle prims while keeping pick.object_prim_path visible.",
    )
    parser.add_argument(
        "--set-viewport-camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set the GUI viewport to --viewport-camera-prim. Defaults to enabled for non-headless runs.",
    )
    parser.add_argument(
        "--viewport-camera-mode",
        choices=("perspective", "stage"),
        default="stage",
        help=(
            "stage switches the active viewport to --viewport-camera-prim; perspective activates "
            "/OmniverseKit_Persp after restoring it from the source scene when available."
        ),
    )
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument(
        "--sync-perspective-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the saved source-scene Perspective camera into /OmniverseKit_Persp before selecting the viewport camera.",
    )
    parser.add_argument("--perspective-camera-prim", default="/OmniverseKit_Persp")
    parser.add_argument("--flat-terrain", action="store_true")
    parser.add_argument("--disable-sky-light", action="store_true")
    parser.add_argument("--max-nav-to-pick-steps", type=int, default=3000)
    parser.add_argument("--max-carry-nav-steps", type=int, default=3000)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.10)
    parser.add_argument("--terminal-position-tolerance", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--final-goal-tolerance-margin", type=float, default=0.03)
    parser.add_argument("--final-yaw-tolerance-margin", type=float, default=0.08)
    parser.add_argument(
        "--ignore-goal-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not require final base yaw alignment; the arm base can rotate to cover the object.",
    )
    parser.add_argument("--carry-mode", choices=("contact",), default="contact")
    parser.add_argument("--carry-posture-name", default="carry_high")
    parser.add_argument(
        "--manip-base-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hold the floating base x/y/z/yaw during arm manipulation phases. "
            "Navigation remains free; the lock is released before carry navigation."
        ),
    )
    parser.add_argument(
        "--manip-dog-joint-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the 12 quadruped support joints during manipulation phases. "
            "This is enabled by default with --manip-base-lock to avoid standing-policy foot jitter."
        ),
    )
    parser.add_argument(
        "--contact-arm-speed-scale",
        type=float,
        default=0.35,
        help=(
            "Scale contact-mode cuRobo arm trajectory time. Values below 1.0 execute slower; "
            "0.35 makes a 1.2 s plan take about 3.4 s."
        ),
    )
    parser.add_argument(
        "--contact-arm-settle-duration",
        type=float,
        default=CONTACT_SETTLE_TO_SEGMENT_START_DURATION,
        help="Seconds used to smoothly move from current arm joints to the first planned waypoint.",
    )
    parser.add_argument(
        "--contact-post-motion-convergence-timeout",
        type=float,
        default=CONTACT_POST_MOTION_CONVERGENCE_TIMEOUT,
        help="Seconds to keep holding strict motion targets before allowing gripper close.",
    )
    parser.add_argument(
        "--contact-post-motion-joint-error-tol",
        type=float,
        default=CONTACT_POST_MOTION_JOINT_ERROR_TOL,
        help="Arm joint error tolerance for strict contact motion convergence.",
    )
    parser.add_argument(
        "--side-pregrasp-offset",
        type=float,
        default=None,
        help="Optional override for GO2_X5_SIDE_PREGRASP_OFFSET_M during side-grasp target generation.",
    )
    parser.add_argument(
        "--tip-tcp-insertion",
        type=float,
        default=None,
        help="Optional override for GO2_X5_TIP_TCP_INSERTION_BEYOND_GRASP_CENTER_M during target generation.",
    )
    parser.add_argument("--verify-grasp-steps", type=int, default=60)
    parser.add_argument("--min-lift-height", type=float, default=0.05)
    parser.add_argument("--max-slip-distance", type=float, default=0.08)
    parser.add_argument("--place-xy-tolerance", type=float, default=0.10)
    parser.add_argument("--place-z-tolerance", type=float, default=0.08)
    parser.add_argument("--record-every-n-steps", type=int, default=1)
    parser.add_argument("--front-camera", action="store_true")
    parser.add_argument("--wrist-camera", action="store_true")
    parser.add_argument("--third-camera", action="store_true")
    parser.add_argument("--head-camera-height", type=int, default=480)
    parser.add_argument("--head-camera-width", type=int, default=640)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--place-settle-steps", type=int, default=None)
    parser.add_argument("--inflate-radius", type=float, default=0.25)
    parser.add_argument("--local-clearance-radius", type=float, default=0.20)
    parser.add_argument("--lookahead-distance", type=float, default=0.35)
    parser.add_argument("--prediction-horizon", type=float, default=0.90)
    parser.add_argument("--max-lin-vel", type=float, default=0.50)
    parser.add_argument("--max-ang-vel", type=float, default=1.00)
    parser.add_argument("--brisk-nav", action="store_true", help="Use a more aggressive DWA speed profile for open routes.")
    parser.add_argument("--fast-dwa", action="store_true", help="Use a lower-cost DWA compute preset.")
    parser.add_argument("--dwa-linear-samples", type=int, default=None)
    parser.add_argument("--dwa-angular-samples", type=int, default=None)
    parser.add_argument("--dwa-integration-dt", type=float, default=None)
    parser.add_argument("--dwa-path-sample-spacing", type=float, default=None)
    parser.add_argument("--dwa-path-distance-window", type=int, default=None)
    parser.add_argument("--min-active-lin-vel", type=float, default=0.30)
    parser.add_argument("--near-goal-min-active-lin-vel", type=float, default=0.22)
    parser.add_argument("--close-goal-speed-limit", type=float, default=0.22)
    parser.add_argument("--speed-bias", type=float, default=0.35)
    parser.add_argument("--max-linear-accel", type=float, default=2.5)
    parser.add_argument("--terminal-yaw-start-distance", type=float, default=0.50)
    parser.add_argument("--terminal-position-kp", type=float, default=0.8)
    parser.add_argument("--terminal-lateral-kp", type=float, default=0.8)
    parser.add_argument("--terminal-lateral-deadband", type=float, default=0.03)
    parser.add_argument("--terminal-max-vx", type=float, default=0.35)
    parser.add_argument("--terminal-min-vx", type=float, default=0.16)
    parser.add_argument("--terminal-max-vy", type=float, default=0.18)
    parser.add_argument("--terminal-min-vy", type=float, default=0.0)
    parser.add_argument("--terminal-yaw-kp", type=float, default=1.2)
    parser.add_argument("--terminal-yaw-min-wz", type=float, default=0.20)
    parser.add_argument("--terminal-yaw-max-wz", type=float, default=0.55)
    parser.add_argument("--terminal-allow-reverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-planner-server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-print-every", type=int, default=120)
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Gym registry key for the runner config.")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.ignore_goal_yaw:
        args.goal_yaw_tolerance = math.pi
        args.terminal_yaw_tolerance = math.pi
        args.final_yaw_tolerance_margin = 0.0
    if args.front_camera or args.wrist_camera or args.third_camera:
        args.enable_cameras = True
    if args.demo_visuals:
        args.load_visual_scene = True
    if args.contact_arm_speed_scale <= 0.0:
        raise ValueError("--contact-arm-speed-scale must be > 0.")
    if args.contact_arm_settle_duration < 0.0:
        raise ValueError("--contact-arm-settle-duration must be >= 0.")
    if args.contact_post_motion_convergence_timeout <= 0.0:
        raise ValueError("--contact-post-motion-convergence-timeout must be > 0.")
    if args.contact_post_motion_joint_error_tol <= 0.0:
        raise ValueError("--contact-post-motion-joint-error-tol must be > 0.")
    if args.side_pregrasp_offset is not None and args.side_pregrasp_offset <= 0.0:
        raise ValueError("--side-pregrasp-offset must be > 0.")
    if args.tip_tcp_insertion is not None and args.tip_tcp_insertion < 0.0:
        raise ValueError("--tip-tcp-insertion must be >= 0.")
    return args, hydra_args, AppLauncher


def _validate_checkpoint(raw_path: str) -> str:
    if "://" in raw_path:
        return raw_path
    checkpoint = _project_path(raw_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Go2-X5 checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def _cli_arg_supplied(flag: str) -> bool:
    return flag in sys.argv


def _optional_min_int(name: str, value: int | None, minimum: int) -> int | None:
    if value is None:
        return None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}.")
    return int(value)


def _optional_positive_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0; got {value}.")
    return float(value)


def _dwa_config(args: argparse.Namespace, control_dt: float):
    from source.navigation.navlib import DWAConfig

    if args.brisk_nav:
        max_linear_velocity = max(args.max_lin_vel, 0.80)
        min_active_linear_velocity = max(args.min_active_lin_vel, 0.55)
        near_goal_min_active_linear_velocity = max(args.near_goal_min_active_lin_vel, 0.38)
        close_goal_speed_limit = max(args.close_goal_speed_limit, 0.35)
        speed_bias = max(args.speed_bias, 1.10)
        max_linear_accel = max(args.max_linear_accel, 4.5)
    else:
        max_linear_velocity = args.max_lin_vel
        min_active_linear_velocity = args.min_active_lin_vel
        near_goal_min_active_linear_velocity = args.near_goal_min_active_lin_vel
        close_goal_speed_limit = args.close_goal_speed_limit
        speed_bias = args.speed_bias
        max_linear_accel = args.max_linear_accel

    prediction_horizon = args.prediction_horizon
    lookahead_distance = args.lookahead_distance
    linear_samples = _optional_min_int("--dwa-linear-samples", args.dwa_linear_samples, 2)
    angular_samples = _optional_min_int("--dwa-angular-samples", args.dwa_angular_samples, 3)
    integration_dt = _optional_positive_float("--dwa-integration-dt", args.dwa_integration_dt)
    path_sample_spacing = _optional_positive_float("--dwa-path-sample-spacing", args.dwa_path_sample_spacing)
    path_distance_window = _optional_min_int("--dwa-path-distance-window", args.dwa_path_distance_window, 1)

    if args.fast_dwa:
        if not _cli_arg_supplied("--prediction-horizon"):
            prediction_horizon = 0.45
        if not _cli_arg_supplied("--lookahead-distance"):
            lookahead_distance = 0.30
        if linear_samples is None:
            linear_samples = 3
        if angular_samples is None:
            angular_samples = 7
        if integration_dt is None:
            integration_dt = 0.05
        if path_sample_spacing is None:
            path_sample_spacing = 0.08
        if path_distance_window is None:
            path_distance_window = 80

    config_kwargs = {}
    if linear_samples is not None:
        config_kwargs["linear_samples"] = linear_samples
    if angular_samples is not None:
        config_kwargs["angular_samples"] = angular_samples
    if integration_dt is not None:
        config_kwargs["integration_dt"] = integration_dt
    if path_sample_spacing is not None:
        config_kwargs["path_sample_spacing"] = path_sample_spacing
    if path_distance_window is not None:
        config_kwargs["path_distance_window"] = path_distance_window

    return DWAConfig(
        control_dt=control_dt,
        lookahead_distance=lookahead_distance,
        prediction_horizon=prediction_horizon,
        goal_tolerance=args.goal_tolerance,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=args.max_ang_vel,
        min_active_linear_velocity=min_active_linear_velocity,
        near_goal_min_active_linear_velocity=near_goal_min_active_linear_velocity,
        close_goal_speed_limit=close_goal_speed_limit,
        speed_bias=speed_bias,
        max_linear_accel=max_linear_accel,
        **config_kwargs,
    )


def _yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return math.cos(half), 0.0, 0.0, math.sin(half)


def _configure_env(args: argparse.Namespace, env_cfg: Any, scene_usd: Path, task: Any, sim_utils: Any) -> None:
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.terrains import TerrainImporterCfg
    from source.navigation.adapters.terrain_utils import write_collision_terrain_wrapper, write_visual_prim_wrapper

    env_cfg.scene.num_envs = 1
    if args.device is not None:
        env_cfg.sim.device = args.device
    if not args.flat_terrain:
        terrain_usd = write_collision_terrain_wrapper(scene_usd, args.terrain_prim_path)
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/nav_collision",
            terrain_type="usd",
            usd_path=str(terrain_usd),
            debug_vis=False,
        )
        if args.add_nav_ground:
            env_cfg.scene.nav_ground = AssetBaseCfg(
                prim_path="/World/nav_ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, args.ground_height)),
                spawn=sim_utils.GroundPlaneCfg(),
            )
    load_scene_reference = args.load_visual_scene or bool(task.pick.object_prim_path)
    if load_scene_reference and args.visual_load_mode == "reference":
        visual_usd = write_visual_prim_wrapper(
            scene_usd,
            args.visual_prim_path,
            excluded_prim_paths=(args.terrain_prim_path, "/World/go2_x5", "/World/mec_arm_6dof"),
        )
        env_cfg.scene.contact_visual_scene = AssetBaseCfg(
            prim_path="/World/contact_visual_scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(visual_usd)),
        )
    if args.disable_sky_light:
        env_cfg.scene.sky_light = None
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (task.start.x, task.start.x),
            "y": (task.start.y, task.start.y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (task.start.yaw, task.start.yaw),
        },
        "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
    }
    env_cfg.observations.policy.enable_corruption = False
    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "push_robot",
        "randomize_push_robot",
        "randomize_actuator_gains",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
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
    if args.front_camera:
        env_cfg.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/head_cam",
            update_period=0.0,
            height=args.head_camera_height,
            width=args.head_camera_width,
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


def _load_scene_sublayer(args: argparse.Namespace, scene_usd: Path) -> None:
    if not (args.load_visual_scene or args.visual_load_mode == "sublayer"):
        return
    import omni.usd
    from source.navigation.adapters.terrain_utils import write_visual_sublayer_wrapper

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    wrapper = write_visual_sublayer_wrapper(
        scene_usd,
        args.visual_prim_path,
        excluded_prim_paths=(args.terrain_prim_path, "/World/go2_x5", "/World/mec_arm_6dof"),
    )
    root_layer = stage.GetRootLayer()
    wrapper_path = str(wrapper)
    if wrapper_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(wrapper_path)
    print(f"[contact-pipeline] scene sublayer for object physics: {wrapper}")


def _hide_nav_collision_visual() -> None:
    """Hide navigation collision geometry in the viewport while keeping physics active."""

    try:
        import omni.usd
        from pxr import UsdGeom

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
            print(f"[contact-pipeline] hid navigation collision visual: {prim_path}")
            hidden_any = True

        if not hidden_any:
            print("[WARN] No navigation collision visual prim found to hide.")
    except Exception as exc:
        print(f"[WARN] Failed to hide navigation collision visual: {exc}")


def _candidate_stage_camera_paths(camera_prim_path: str) -> list[str]:
    """Return camera path candidates for directly sublayered and referenced scenes."""

    candidates = [camera_prim_path]
    if camera_prim_path == "/World/camera_main":
        candidates.append("/World/Camera_main")
    if camera_prim_path == "/World/Camera_main":
        candidates.append("/World/camera_main")
    if camera_prim_path.startswith("/World/"):
        suffix = camera_prim_path.removeprefix("/World/")
        candidates.append(f"/World/contact_visual_scene/{suffix}")
        candidates.append(f"/World/nav_visual_scene/{suffix}")
    return list(dict.fromkeys(candidates))


def _resolve_stage_camera(stage: Any, camera_prim_path: str):
    from pxr import UsdGeom

    for candidate in _candidate_stage_camera_paths(camera_prim_path):
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid() and prim.IsA(UsdGeom.Camera):
            return candidate, UsdGeom.Camera(prim)
    return None, None


def _stage_camera_eye_target(camera) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import math
    from pxr import Gf, Usd, UsdGeom

    prim = camera.GetPrim()
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    eye_vec = matrix.ExtractTranslation()
    forward_vec = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    if forward_vec.GetLength() <= 1.0e-9:
        forward_vec = Gf.Vec3d(1.0, 0.0, 0.0)
    forward_vec.Normalize()
    focus_distance = camera.GetFocusDistanceAttr().Get()
    try:
        focus_distance = float(focus_distance)
    except (TypeError, ValueError):
        focus_distance = 0.0
    if not math.isfinite(focus_distance) or focus_distance <= 0.1:
        focus_distance = 3.0
    target_vec = eye_vec + forward_vec * focus_distance
    eye = (float(eye_vec[0]), float(eye_vec[1]), float(eye_vec[2]))
    target = (float(target_vec[0]), float(target_vec[1]), float(target_vec[2]))
    return eye, target


def _set_perspective_from_stage_camera(camera_prim_path: str, sim: Any | None = None) -> bool:
    """Copy an authored camera pose into the freely movable Perspective camera."""

    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Cannot set Perspective camera: no active USD stage.")
            return False

        selected_path, camera = _resolve_stage_camera(stage, camera_prim_path)
        if camera is None:
            print(
                "[WARN] Cannot set Perspective camera: no valid Camera prim found. "
                f"tried={_candidate_stage_camera_paths(camera_prim_path)}"
            )
            return False

        eye, target = _stage_camera_eye_target(camera)
        if sim is not None and hasattr(sim, "set_camera_view"):
            sim.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
        else:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp")

        viewport = get_active_viewport()
        if viewport is not None:
            try:
                viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
            except Exception:
                pass
        print(
            "[contact-pipeline] Perspective camera copied from stage camera: "
            f"{selected_path} eye={tuple(round(value, 3) for value in eye)} "
            f"target={tuple(round(value, 3) for value in target)}"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Failed to copy stage camera {camera_prim_path} to Perspective: {exc}")
        return False


def _set_active_viewport_camera_path(camera_prim_path: str) -> bool:
    """Switch the active viewport to an existing camera prim path."""

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        viewport = get_active_viewport()
        if viewport is None:
            print("[WARN] Cannot set active viewport camera: no active viewport.")
            return False
        sdf_path = Sdf.Path(camera_prim_path)
        try:
            viewport.camera_path = sdf_path
        except Exception:
            if hasattr(viewport, "set_active_camera"):
                viewport.set_active_camera(sdf_path)
            else:
                raise
        print(f"[contact-pipeline] active viewport camera set to: {camera_prim_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set active viewport camera {camera_prim_path}: {exc}")
        return False


def _copy_camera_intrinsics(source_camera: Any, target_camera: Any) -> None:
    """Copy common camera optical attributes when both camera prims exist."""

    attr_names = (
        "projection",
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "clippingRange",
        "clippingPlanes",
        "fStop",
        "focusDistance",
    )
    source_prim = source_camera.GetPrim()
    target_prim = target_camera.GetPrim()
    for attr_name in attr_names:
        source_attr = source_prim.GetAttribute(attr_name)
        target_attr = target_prim.GetAttribute(attr_name)
        if not source_attr.IsValid() or not target_attr.IsValid():
            continue
        value = source_attr.Get()
        if value is not None:
            target_attr.Set(value)


def _sync_perspective_from_source_scene(
    scene_usd: Path,
    camera_prim_path: str,
    sim: Any | None = None,
) -> bool:
    """Restore Perspective from the authored source scene USD."""

    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        source_stage = Usd.Stage.Open(str(scene_usd))
        if source_stage is None:
            print(f"[WARN] Cannot sync Perspective camera: failed to open source scene {scene_usd}")
            return False
        candidate_paths = list(dict.fromkeys([camera_prim_path, "/OmniverseKit_Persp", "/World/OmniverseKit_Persp"]))
        selected_path = None
        source_camera = None
        for candidate in candidate_paths:
            prim = source_stage.GetPrimAtPath(candidate)
            if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                selected_path = candidate
                source_camera = UsdGeom.Camera(prim)
                break
        if source_camera is None:
            print(f"[WARN] Source scene has no saved Perspective camera. tried={candidate_paths}")
            return False

        eye, target = _stage_camera_eye_target(source_camera)
        if sim is not None and hasattr(sim, "set_camera_view"):
            sim.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
        else:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp")

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            target_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
            if target_prim.IsValid() and target_prim.IsA(UsdGeom.Camera):
                _copy_camera_intrinsics(source_camera, UsdGeom.Camera(target_prim))
        print(
            "[contact-pipeline] Perspective camera restored from source scene: "
            f"{selected_path} eye={tuple(round(value, 3) for value in eye)} "
            f"target={tuple(round(value, 3) for value in target)}"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Failed to sync Perspective camera from source scene {scene_usd}: {exc}")
        return False


def _set_viewport_stage_camera(camera_prim_path: str) -> bool:
    """Switch the GUI viewport to an authored stage camera."""

    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Cannot set stage camera: no active USD stage.")
            return False

        selected_path, _camera = _resolve_stage_camera(stage, camera_prim_path)
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
        print(f"[contact-pipeline] viewport camera set to stage camera: {selected_path}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to set stage camera {camera_prim_path}: {exc}")
        return False


def _hide_distractor_objects(task: Any) -> dict[str, Any]:
    """Hide orange/bottle and non-task apple prims while keeping the selected object visible."""

    try:
        import omni.usd
        from pxr import UsdGeom

        object_prim_path = task.pick.object_prim_path
        if not object_prim_path:
            return {"applied": False, "reason": "pick.object_prim_path missing"}
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return {"applied": False, "reason": "stage missing"}

        object_prefix = object_prim_path.rstrip("/") + "/"
        hidden_paths: list[str] = []
        shown_paths: list[str] = []
        keywords = ("apple", "orange", "bottle")
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not any(keyword in prim_path.lower() for keyword in keywords):
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue
            imageable = UsdGeom.Imageable(prim)
            if prim_path == object_prim_path or prim_path.startswith(object_prefix):
                imageable.MakeVisible()
                shown_paths.append(prim_path)
            else:
                imageable.MakeInvisible()
                hidden_paths.append(prim_path)
        report = {
            "applied": True,
            "kept_object_prim_path": object_prim_path,
            "shown_count": len(shown_paths),
            "hidden_count": len(hidden_paths),
        }
        print("[contact-pipeline] object visibility:", report)
        return {**report, "shown_paths": shown_paths, "hidden_paths": hidden_paths}
    except Exception as exc:
        print(f"[WARN] Failed to hide distractor objects: {exc}")
        return {"applied": False, "reason": str(exc)}


def _world_root_path(prim_path: str) -> str:
    """Return the top-level /World child for a prim path when available."""

    parts = prim_path.split("/")
    if len(parts) >= 3 and parts[1] == "World":
        return f"/World/{parts[2]}"
    return prim_path


def _grasp_collision_exclusion_paths(stage: Any, task: Any, *, exclude_distractors: bool) -> list[str]:
    """Paths excluded from cuRobo world collision for the pick planner."""

    object_prim_path = task.pick.object_prim_path
    paths: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in paths:
            paths.append(path)

    add(object_prim_path)
    if not exclude_distractors:
        return paths

    object_prefix = object_prim_path.rstrip("/") + "/" if object_prim_path else ""
    keywords = ("apple", "orange", "bottle")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_path_lower = prim_path.lower()
        if not any(keyword in prim_path_lower for keyword in keywords):
            continue
        if object_prim_path and (prim_path == object_prim_path or prim_path.startswith(object_prefix)):
            continue
        add(_world_root_path(prim_path))
    return paths


def _get_or_add_xform_op(xformable, op_type):
    from pxr import UsdGeom

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


def _set_translate_op(op, xyz: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*xyz))
    else:
        op.Set(Gf.Vec3f(*xyz))


def _set_orient_op(op, quat_wxyz: tuple[float, float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    w, x, y, z = quat_wxyz
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _zero_rigid_body_velocities(prim) -> int:
    """Clear authored rigid-body velocities without creating new rigid bodies."""

    from pxr import Gf, Usd, UsdPhysics

    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if timeline is not None and timeline.is_playing():
            print("[contact-pipeline] skipped rigid body velocity clear while timeline is playing.")
            return 0
    except Exception:
        pass

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


def _apply_object_pose_from_task(task: Any) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdGeom

    object_path = task.pick.object_prim_path
    pose = task.pick.object_pose_world
    if not object_path:
        return {"applied": False, "reason": "pick.object_prim_path missing"}
    if pose is None:
        return {"applied": False, "reason": "pick.object_pose_world missing"}
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {"applied": False, "reason": "stage missing"}
    prim = stage.GetPrimAtPath(object_path)
    if not prim.IsValid():
        return {"applied": False, "reason": f"object prim not found: {object_path}"}
    if not prim.IsA(UsdGeom.Xformable):
        return {"applied": False, "reason": f"object prim is not xformable: {object_path}"}

    xformable = UsdGeom.Xformable(prim)
    translate_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
    orient_op = _get_or_add_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
    quat = _yaw_to_quat_wxyz(pose.yaw)
    _set_translate_op(translate_op, (pose.x, pose.y, pose.z))
    _set_orient_op(orient_op, quat)
    zeroed_count = _zero_rigid_body_velocities(prim)
    return {
        "applied": True,
        "object_prim_path": object_path,
        "object_pose_world": pose.to_dict(),
        "rigid_body_velocity_zeroed_count": zeroed_count,
        "xform_op_order": [op.GetOpName() for op in xformable.GetOrderedXformOps()],
    }


def _tensor_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _jpeg_bytes(rgb_tensor: Any) -> bytes | None:
    if rgb_tensor is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    stream = io.BytesIO()
    Image.fromarray(rgb_tensor.cpu().numpy()).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def _run_kit_coroutine(coro: Any, simulation_app: Any, *, timeout_s: float, label: str) -> Any:
    try:
        from omni.kit.async_engine import run_coroutine

        future = run_coroutine(coro)
    except Exception:
        future = asyncio.get_event_loop().create_task(coro)
    started_at = time.time()
    while not future.done():
        simulation_app.update()
        if timeout_s > 0.0 and time.time() - started_at > timeout_s:
            future.cancel()
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
    return future.result()


def _live_body_matrix(adapter: Any, body_name: str):
    """Return a live PhysX body transform matrix from Isaac Lab tensors."""

    import numpy as np
    from scripts.math.SE3 import pose_to_matrix

    try:
        try:
            body_ids, _ = adapter.robot.find_bodies([body_name], preserve_order=True)
        except TypeError:
            body_ids, _ = adapter.robot.find_bodies([body_name])
    except ValueError:
        return None
    if not body_ids:
        return None
    body_id = int(body_ids[0])
    position = [float(value) for value in _tensor_list(adapter.robot.data.body_pos_w[0, body_id])[:3]]
    quat_wxyz = [float(value) for value in _tensor_list(adapter.robot.data.body_quat_w[0, body_id])[:4]]
    matrix = pose_to_matrix(np.asarray(position, dtype=float), np.asarray(quat_wxyz, dtype=float))
    return matrix


def _robot_body_names(adapter: Any) -> list[str]:
    """Return Isaac Lab articulation body names for diagnostics."""

    try:
        names = getattr(adapter.robot, "body_names", [])
        if callable(names):
            names = names()
        return [str(name) for name in names]
    except Exception as exc:
        return [f"<body_names unavailable: {exc}>"]


def _find_prim_by_name_under(stage: Any, root_path: str, prim_name: str) -> str | None:
    """Find the first USD prim with a given name under likely robot roots."""

    from pxr import Usd

    root_candidates = [root_path, "/World/go2_x5", "/World"]
    for candidate in dict.fromkeys(path for path in root_candidates if path):
        root_prim = stage.GetPrimAtPath(candidate)
        if not root_prim.IsValid():
            continue
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == prim_name:
                return str(prim.GetPath())
    return None


def _usd_world_matrix(stage: Any, prim_path: str):
    """Read a USD prim world transform using the same convention as the state exporter."""

    import numpy as np
    from pxr import Usd, UsdGeom
    from scripts.math.SE3 import normalize_quat_wxyz, pose_to_matrix

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    usd_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    position = np.asarray([translation[0], translation[1], translation[2]], dtype=float)
    quaternion = normalize_quat_wxyz([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
    return pose_to_matrix(position, quaternion)


def _usd_relative_matrix_by_name(stage: Any, root_path: str, parent_name: str, child_name: str):
    """Return the authored fixed USD transform parent_name -> child_name."""

    import numpy as np

    parent_path = _find_prim_by_name_under(stage, root_path, parent_name)
    child_path = _find_prim_by_name_under(stage, root_path, child_name)
    if parent_path is None or child_path is None:
        return None, {"parent_path": parent_path, "child_path": child_path}
    T_world_parent = _usd_world_matrix(stage, parent_path)
    T_world_child = _usd_world_matrix(stage, child_path)
    if T_world_parent is None or T_world_child is None:
        return None, {"parent_path": parent_path, "child_path": child_path}
    return np.linalg.inv(T_world_parent) @ T_world_child, {
        "parent_path": parent_path,
        "child_path": child_path,
    }


def _live_frame_matrix(
    adapter: Any,
    frame_name: str,
    *,
    stage: Any,
    robot_root_path: str,
    frame_report: dict[str, Any],
):
    """Return a live matrix for planner frames that may not be articulation bodies."""

    from scripts.math.SE3 import xyz_rpy_to_matrix

    T_world_body = _live_body_matrix(adapter, frame_name)
    if T_world_body is not None:
        frame_report[frame_name] = {"source": "live_articulation_body", "body_name": frame_name}
        return T_world_body

    if frame_name == "arm_base_link":
        T_world_base_body = _live_body_matrix(adapter, "base")
        if T_world_base_body is None:
            frame_report[frame_name] = {
                "source": "missing",
                "reason": "articulation body base not found",
            }
            return None
        T_base_body_arm_base, usd_paths = _usd_relative_matrix_by_name(
            stage,
            robot_root_path,
            "base",
            "arm_base_link",
        )
        if T_base_body_arm_base is None:
            frame_report[frame_name] = {
                "source": "missing",
                "reason": "USD relative transform base -> arm_base_link not found",
                **usd_paths,
            }
            return None
        frame_report[frame_name] = {
            "source": "live_body_plus_usd_fixed_frame",
            "body_name": "base",
            "relative_transform": "base->arm_base_link",
            **usd_paths,
        }
        return T_world_base_body @ T_base_body_arm_base

    if frame_name == "grasp_tcp_link":
        T_world_arm_link6 = _live_body_matrix(adapter, "arm_link6")
        if T_world_arm_link6 is None:
            frame_report[frame_name] = {
                "source": "missing",
                "reason": "articulation body arm_link6 not found",
            }
            return None
        T_arm_link6_tcp, usd_paths = _usd_relative_matrix_by_name(
            stage,
            robot_root_path,
            "arm_link6",
            "grasp_tcp_link",
        )
        if T_arm_link6_tcp is not None:
            frame_report[frame_name] = {
                "source": "live_body_plus_usd_fixed_frame",
                "body_name": "arm_link6",
                "relative_transform": "arm_link6->grasp_tcp_link",
                **usd_paths,
            }
            return T_world_arm_link6 @ T_arm_link6_tcp
        T_arm_link6_tcp = xyz_rpy_to_matrix((0.15757, 0.0, 0.0), (0.0, 0.0, 0.0))
        frame_report[frame_name] = {
            "source": "live_body_plus_configured_tcp_offset",
            "body_name": "arm_link6",
            "relative_transform": "arm_link6->grasp_tcp_link",
            "fallback_offset_xyz": [0.15757, 0.0, 0.0],
            **usd_paths,
        }
        return T_world_arm_link6 @ T_arm_link6_tcp

    frame_report[frame_name] = {
        "source": "missing",
        "reason": "not an articulation body and no special frame rule",
    }
    return None


def _patch_grasp_state_from_live_robot(
    *,
    state: dict[str, Any],
    adapter: Any,
    task: Any,
    pipeline: Any,
    exclude_distractor_collision: bool = True,
) -> dict[str, Any]:
    """Patch exported grasp state with live Isaac Lab body transforms.

    The generic Isaac Sim exporter reads USD xforms. In Isaac Lab, PhysX tensor
    state is authoritative during a running RL environment, while USD xforms can
    remain at the reset/authored transform. Target generation must therefore use
    the live arm_base_link pose after navigation.
    """

    import numpy as np
    import omni.usd
    from scripts.math.SE3 import pose_dict_from_matrix

    report: dict[str, Any] = {
        "applied": False,
        "robot_body_names": _robot_body_names(adapter),
        "frame_patch": {},
    }
    print("[contact-pipeline] Isaac Lab robot body_names:", report["robot_body_names"])
    stage = omni.usd.get_context().get_stage()
    robot_root_path = state.get("paths", {}).get("robot_root_path") or "/World/go2_x5"

    T_world_base = _live_frame_matrix(
        adapter,
        "arm_base_link",
        stage=stage,
        robot_root_path=robot_root_path,
        frame_report=report["frame_patch"],
    )
    if T_world_base is None:
        report["reason"] = "arm_base_link frame not resolved from live robot body and USD fixed frame"
        print("[contact-pipeline] frame patch report:", json.dumps(report["frame_patch"], ensure_ascii=False))
        return report

    T_world_tcp = _live_frame_matrix(
        adapter,
        "grasp_tcp_link",
        stage=stage,
        robot_root_path=robot_root_path,
        frame_report=report["frame_patch"],
    )
    if T_world_tcp is None:
        old_base_tcp = state.get("poses", {}).get("base_tcp", {})
        old_matrix = old_base_tcp.get("matrix_4x4")
        if old_matrix is None:
            report["reason"] = "grasp_tcp_link frame not resolved and exported base_tcp missing"
            print("[contact-pipeline] frame patch report:", json.dumps(report["frame_patch"], ensure_ascii=False))
            return report
        T_world_tcp = T_world_base @ np.asarray(old_matrix, dtype=float)
        report["frame_patch"]["grasp_tcp_link"] = {
            "source": "exported_base_tcp_recomposed_with_live_arm_base_link",
        }

    print("[contact-pipeline] frame patch report:", json.dumps(report["frame_patch"], ensure_ascii=False))
    T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp
    state.setdefault("poses", {})["world_base"] = pose_dict_from_matrix(T_world_base)
    state["poses"]["world_tcp"] = pose_dict_from_matrix(T_world_tcp)
    state["poses"]["base_tcp"] = pose_dict_from_matrix(T_base_tcp)
    state["source"] = "Isaac Lab contact runtime with live PhysX tensor pose patch"

    try:
        export_module = pipeline._load_module("go2_x5_export_state_live_patch", pipeline.script_export)
        selected_paths = _grasp_collision_exclusion_paths(
            stage,
            task,
            exclude_distractors=exclude_distractor_collision,
        )
        print("[contact-pipeline] grasp collision excluded paths:", selected_paths)
        cuboids = export_module.compute_world_collision_cuboids(
            stage=stage,
            robot_root_path=robot_root_path,
            T_world_base=T_world_base,
            selected_paths=selected_paths,
        )
        world_collision = state.setdefault("world_collision", {})
        world_collision["cuboids_base"] = cuboids
        world_collision["live_pose_recomputed"] = True
        world_collision["excluded_selected_prim_paths"] = selected_paths
        world_collision["exclude_distractor_collision"] = bool(exclude_distractor_collision)
        report["world_collision_recomputed"] = True
        report["world_collision_cuboids"] = len(cuboids)
        report["world_collision_excluded_paths"] = selected_paths
    except Exception as exc:
        report["world_collision_recomputed"] = False
        report["world_collision_recompute_error"] = str(exc)

    report["applied"] = True
    report["world_base_position_xyz"] = state["poses"]["world_base"]["position_xyz"]
    report["world_base_quaternion_wxyz"] = state["poses"]["world_base"]["quaternion_wxyz"]
    state["isaac_lab_live_pose_patch"] = report
    return report


class IsaacLabContactRuntime:
    """Isaac Lab runtime adapter used by the contact-only state machine."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        task: Any,
        adapter: Any,
        nav_map: Path,
        recorder: Any,
        monitor: Any,
        simulation_app: Any,
        object_pose_report: dict[str, Any] | None = None,
    ):
        self.args = args
        self.task = task
        self.adapter = adapter
        self.nav_map = nav_map
        self.recorder = recorder
        self.monitor = monitor
        self.simulation_app = simulation_app
        self.object_pose_report = object_pose_report
        self.dt = float(adapter.env.unwrapped.step_dt)
        self.planner = None
        self.current_goal_key: tuple[float, float, float] | None = None
        self.path_world: list[tuple[float, float]] = []
        self.last_command = (0.0, 0.0, 0.0)
        self._contact_step_count = 0
        self.arm_target = CARRY_POSTURES["stow"]
        self.gripper_target = GRIPPER_OPEN_TARGET
        self.arm_action_override_report = {}
        if hasattr(self.adapter, "set_direct_arm_action_override"):
            self.arm_action_override_report = self.adapter.set_direct_arm_action_override(True)
            print("[contact-pipeline] arm action override:", self.arm_action_override_report)
        self.manip_base_lock_report = {"enabled": False, "pose_xyzyaw": None, "pose_xyyaw": None}
        self.manip_dog_joint_lock_report = {"enabled": False, "joint_names": [], "joint_ids": [], "action_indices": []}
        self._manip_base_lock_enabled = False
        self.runtime_notes = {
            "attachment_mode": "contact_only",
            "object_pose_write_policy": "reset_only",
            "fixed_joint_default": False,
            "kinematic_follow": False,
            "place_controller": "mvp_release_at_current_carry_posture",
            "pick_success_policy": "side_grasp_retreat",
            "arm_control": "direct_policy_action_override",
            "arm_action_override": self.arm_action_override_report,
            "manip_base_lock": self.manip_base_lock_report,
            "manip_dog_joint_lock": self.manip_dog_joint_lock_report,
            "contact_motion": {
                "arm_speed_scale": float(self.args.contact_arm_speed_scale),
                "arm_command_dt": CONTACT_ARM_COMMAND_DT,
                "arm_settle_duration": float(self.args.contact_arm_settle_duration),
                "pre_close_hold_duration": CONTACT_PRE_CLOSE_HOLD_DURATION,
                "post_motion_convergence_timeout": float(self.args.contact_post_motion_convergence_timeout),
                "post_motion_joint_error_tol": float(self.args.contact_post_motion_joint_error_tol),
            },
            "target_generation_overrides": {
                "side_pregrasp_offset": self.args.side_pregrasp_offset,
                "tip_tcp_insertion": self.args.tip_tcp_insertion,
            },
        }

    def reset_episode(self, task: Any):
        from source.pipeline import PhaseResult

        self._set_manip_base_lock(False, reason="reset_episode")
        self.adapter.reset_to_pose(task.start.x, task.start.y, task.start.yaw)
        self.arm_target = CARRY_POSTURES["stow"]
        self.gripper_target = GRIPPER_OPEN_TARGET
        object_pose_report = self.object_pose_report
        if object_pose_report is None or not object_pose_report.get("applied", False):
            object_pose_report = _apply_object_pose_from_task(task)
        for _ in range(max(0, int(self.args.settle_steps))):
            self._step_policy(controller="pd", phase="reset")
        return PhaseResult(
            success=bool(object_pose_report.get("applied", False)),
            failure_reason="" if object_pose_report.get("applied", False) else "object_reset_failed",
            metadata={
                "state": {
                    "object_pose_reset": object_pose_report,
                    "runtime_notes": self.runtime_notes,
                },
                "observation": self._observation(),
                "action": self._action(controller="pd"),
            },
        )

    def step_navigation(self, goal: Any, phase: Any):
        from source.navigation.adapters.frame_utils import wrap_yaw
        from source.pipeline import RuntimeStepResult

        if phase.value == "carry_nav_to_place":
            self._set_manip_base_lock(False, reason="carry_navigation")
        self._ensure_planner(goal)
        pose = self.adapter.get_base_pose()
        distance = math.hypot(goal.x - pose[0], goal.y - pose[1])
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        position_acceptance_tolerance = self.args.goal_tolerance + max(0.0, self.args.final_goal_tolerance_margin)
        yaw_acceptance_tolerance = self.args.goal_yaw_tolerance + max(0.0, self.args.final_yaw_tolerance_margin)
        reached_before = (
            distance <= position_acceptance_tolerance
            and abs(yaw_error) <= yaw_acceptance_tolerance
        )
        terminal_start_distance = max(self.args.goal_tolerance, self.args.terminal_yaw_start_distance)
        use_terminal_controller = distance <= terminal_start_distance
        if self.args.ignore_goal_yaw and distance > position_acceptance_tolerance:
            use_terminal_controller = False
        controller = "terminal" if use_terminal_controller else "dwa"
        command = (0.0, 0.0, 0.0) if reached_before else self._nav_command(goal, controller)
        if phase.value == "carry_nav_to_place":
            self.arm_target = CARRY_POSTURES.get(self.args.carry_posture_name, CARRY_POSTURES["carry_high"])
            self.gripper_target = GRIPPER_CLOSE_TARGET
        else:
            self.arm_target = CARRY_POSTURES["stow"]
            self.gripper_target = GRIPPER_OPEN_TARGET
        self.last_command = command
        self.adapter.apply_base_command(*command)
        self._step_policy(controller=controller, phase=phase.value)
        pose_after = self.adapter.get_base_pose()
        yaw_error_after = wrap_yaw(goal.yaw - pose_after[2])
        reached_after = (
            math.hypot(goal.x - pose_after[0], goal.y - pose_after[1])
            <= position_acceptance_tolerance
            and abs(yaw_error_after) <= yaw_acceptance_tolerance
        )
        return RuntimeStepResult(
            observation=self._observation(),
            action=self._action(controller=controller),
            state={
                "goal_xyyaw": [goal.x, goal.y, goal.yaw],
                "goal_distance": math.hypot(goal.x - pose_after[0], goal.y - pose_after[1]),
                "yaw_error": yaw_error_after,
                "position_acceptance_tolerance": position_acceptance_tolerance,
                "yaw_acceptance_tolerance": yaw_acceptance_tolerance,
                "terminal_controller_enabled": use_terminal_controller,
                "runtime_notes": self.runtime_notes,
            },
            timestamp=time.time(),
            images=self._images(),
            reached_goal=reached_after,
        )

    def stop_base(self) -> None:
        self.last_command = (0.0, 0.0, 0.0)
        self.adapter.apply_base_command(0.0, 0.0, 0.0)

    def _set_manip_base_lock(self, enabled: bool, *, reason: str) -> None:
        if not self.args.manip_base_lock or not hasattr(self.adapter, "set_base_pose_lock"):
            return
        if bool(enabled) == self._manip_base_lock_enabled:
            return
        self.manip_base_lock_report = self.adapter.set_base_pose_lock(enabled)
        if self.args.manip_dog_joint_lock and hasattr(self.adapter, "set_support_joint_lock"):
            self.manip_dog_joint_lock_report = self.adapter.set_support_joint_lock(enabled)
        else:
            self.manip_dog_joint_lock_report = {"enabled": False, "joint_names": [], "joint_ids": [], "action_indices": []}
        self._manip_base_lock_enabled = bool(self.manip_base_lock_report.get("enabled", False))
        self.runtime_notes["manip_base_lock"] = {
            **self.manip_base_lock_report,
            "reason": reason,
        }
        self.runtime_notes["manip_dog_joint_lock"] = {
            **self.manip_dog_joint_lock_report,
            "reason": reason,
        }
        print("[contact-pipeline] manip base lock:", self.runtime_notes["manip_base_lock"])
        print("[contact-pipeline] manip dog joint lock:", self.runtime_notes["manip_dog_joint_lock"])

    def settle_base(self, phase: Any):
        from source.pipeline import RuntimeStepResult

        self.stop_base()
        if phase.value in {"pick_prepare", "place_approach"}:
            self._set_manip_base_lock(True, reason=phase.value)
        self._step_policy(controller="pd", phase=phase.value)
        return RuntimeStepResult(
            observation=self._observation(),
            action=self._action(controller="pd"),
            state={"runtime_notes": self.runtime_notes},
            timestamp=time.time(),
            images=self._images(),
            reached_goal=False,
        )

    def _arm_positions(self):
        import numpy as np

        if len(self.adapter.arm_joint_ids) != 6:
            return np.asarray([], dtype=float)
        return np.asarray(_tensor_list(self.adapter.robot.data.joint_pos[0, self.adapter.arm_joint_ids]), dtype=float)

    def _gripper_positions(self):
        import numpy as np

        if len(self.adapter.gripper_joint_ids) != 2:
            return np.asarray([], dtype=float)
        return np.asarray(_tensor_list(self.adapter.robot.data.joint_pos[0, self.adapter.gripper_joint_ids]), dtype=float)

    @staticmethod
    def _smoothstep5(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5

    @staticmethod
    def _sample_joint_trajectory(time_from_start: Any, q_traj: Any, t: float):
        import numpy as np

        times = np.asarray(time_from_start, dtype=float)
        q = np.asarray(q_traj, dtype=float)
        if q.ndim != 2 or times.ndim != 1 or q.shape[0] != times.shape[0]:
            raise RuntimeError("invalid cuRobo trajectory: q and time_from_start shape mismatch")
        t = float(max(times[0], min(times[-1], t)))
        return np.asarray([np.interp(t, times, q[:, index]) for index in range(q.shape[1])], dtype=float)

    @staticmethod
    def _sample_cubic_hermite(time_from_start: Any, q_traj: Any, qd_traj: Any, t: float):
        import numpy as np

        times = np.asarray(time_from_start, dtype=float)
        q = np.asarray(q_traj, dtype=float)
        qd = np.asarray(qd_traj, dtype=float)
        if q.ndim != 2 or times.ndim != 1 or q.shape[0] != times.shape[0]:
            raise RuntimeError("invalid cuRobo trajectory: q and time_from_start shape mismatch")
        if qd.shape != q.shape:
            return IsaacLabContactRuntime._sample_joint_trajectory(times, q, t)
        t = float(np.clip(t, times[0], times[-1]))
        index = int(np.searchsorted(times, t, side="right") - 1)
        index = max(0, min(index, len(times) - 2))
        t0 = float(times[index])
        t1 = float(times[index + 1])
        h = t1 - t0
        if h <= 1.0e-9:
            return q[index].copy()
        u = (t - t0) / h
        q0 = q[index]
        q1 = q[index + 1]
        v0 = qd[index]
        v1 = qd[index + 1]
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        return h00 * q0 + h10 * h * v0 + h01 * q1 + h11 * h * v1

    @staticmethod
    def _close_progress(q_start: Any, q_final: Any, q_target: Any) -> float:
        import numpy as np

        q_start = np.asarray(q_start, dtype=float)
        q_final = np.asarray(q_final, dtype=float)
        q_target = np.asarray(q_target, dtype=float)
        total = float(np.linalg.norm(q_start - q_target))
        if total < 1.0e-9:
            return 1.0
        actual = float(np.linalg.norm(q_start - q_final))
        return float(np.clip(actual / total, 0.0, 1.0))

    def _step_contact_targets(self, *, arm_target: Any | None, gripper_target: Any | None, phase: str) -> None:
        self.adapter.apply_base_command(0.0, 0.0, 0.0)
        if arm_target is not None:
            self.adapter.set_arm_joint_target(arm_target)
        if gripper_target is not None:
            self.adapter.set_gripper_joint_target(gripper_target)
        self.adapter.step()
        self._contact_step_count += 1
        if self.args.debug_print_every > 0 and self._contact_step_count % self.args.debug_print_every == 0:
            pose = self.adapter.get_base_pose()
            print(f"[contact-pipeline] phase={phase} pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) controller=contact_grasp")

    def _settle_arm_to_plan_start(self, q_start: Any, label: str) -> dict[str, Any]:
        import numpy as np

        q_initial = self._arm_positions()
        q_start = np.asarray(q_start, dtype=float)
        start_error = float(np.linalg.norm(q_initial - q_start))
        print(f"[contact-settle:{label}] start_error={start_error:.6f}")
        settle_duration = max(0.0, float(self.args.contact_arm_settle_duration))
        steps = max(2, int(round(settle_duration / max(self.dt, 1.0e-6))))
        log = {
            "name": f"settle_{label}",
            "type": "settle",
            "duration": settle_duration,
            "start_error": start_error,
            "target_q_arm": [],
            "actual_q_arm": [],
            "joint_error_norm": [],
        }
        gripper_hold = self._gripper_positions()
        for step in range(steps):
            u = step / float(max(1, steps - 1))
            s = self._smoothstep5(u)
            q_target = (1.0 - s) * q_initial + s * q_start
            self._step_contact_targets(arm_target=q_target, gripper_target=gripper_hold, phase=f"settle_{label}")
            q_actual = self._arm_positions()
            log["target_q_arm"].append(q_target.tolist())
            log["actual_q_arm"].append(q_actual.tolist())
            log["joint_error_norm"].append(float(np.linalg.norm(q_actual - q_target)))
        return log

    def _wait_until_arm_reaches(self, q_target: Any, label: str) -> dict[str, Any]:
        import numpy as np

        q_target = np.asarray(q_target, dtype=float)
        timeout_s = max(1.0e-6, float(self.args.contact_post_motion_convergence_timeout))
        error_tol = max(1.0e-6, float(self.args.contact_post_motion_joint_error_tol))
        steps = max(1, int(round(timeout_s / max(self.dt, 1.0e-6))))
        gripper_hold = self._gripper_positions()
        log = {
            "name": f"wait_{label}",
            "type": "wait",
            "converged": False,
            "timeout_s": timeout_s,
            "joint_error_tol": error_tol,
            "joint_error_norm": [],
            "target_q_arm": [],
            "actual_q_arm": [],
        }
        for step in range(steps):
            self._step_contact_targets(arm_target=q_target, gripper_target=gripper_hold, phase=f"wait_{label}")
            q_actual = self._arm_positions()
            error = float(np.linalg.norm(q_actual - q_target))
            log["joint_error_norm"].append(error)
            log["target_q_arm"].append(q_target.tolist())
            log["actual_q_arm"].append(q_actual.tolist())
            if step % 20 == 0:
                print(f"[contact-wait:{label}] step={step:03d}, joint_error={error:.6f}")
            if error <= error_tol:
                log["converged"] = True
                print(f"[contact-wait:{label}] converged, joint_error={error:.6f}")
                break
        if not log["converged"]:
            final_error = log["joint_error_norm"][-1] if log["joint_error_norm"] else None
            print(f"[contact-wait:{label}] timeout, final_joint_error={final_error}")
        return log

    def _execute_contact_motion_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        name = str(segment.get("name", "motion"))
        trajectory = segment.get("trajectory", {})
        times = np.asarray(trajectory.get("time_from_start", []), dtype=float)
        q_traj = np.asarray(trajectory.get("q", []), dtype=float)
        qd_traj = np.asarray(trajectory.get("qd", np.zeros_like(q_traj)), dtype=float)
        if times.size == 0 or q_traj.size == 0:
            raise RuntimeError(f"{name}: missing q trajectory")
        settle_log = self._settle_arm_to_plan_start(q_traj[0], name)
        duration = float(times[-1])
        speed_scale = max(1.0e-6, float(self.args.contact_arm_speed_scale))
        effective_duration = duration / speed_scale
        steps = int(math.ceil(effective_duration / max(self.dt, 1.0e-6))) + 1
        command_period_steps = max(1, int(round(CONTACT_ARM_COMMAND_DT / max(self.dt, 1.0e-6))))
        q_target = q_traj[0].copy()
        gripper_hold = self._gripper_positions()
        log = {
            "name": name,
            "type": "motion",
            "sim_dt": self.dt,
            "command_dt": CONTACT_ARM_COMMAND_DT,
            "arm_speed_scale": speed_scale,
            "planned_duration": duration,
            "effective_duration": effective_duration,
            "time": [],
            "plan_time": [],
            "target_q_arm": [],
            "actual_q_arm": [],
            "joint_error_norm": [],
            "settle_to_start": settle_log,
        }
        print(
            f"[contact-motion:{name}] duration={duration:.3f}s, "
            f"effective_duration={effective_duration:.3f}s, speed_scale={speed_scale:.3f}, sim_steps={steps}"
        )
        for step in range(steps):
            t_wall = min(step * self.dt, effective_duration)
            t = min(t_wall * speed_scale, duration)
            if step % command_period_steps == 0 or step == steps - 1:
                q_target = self._sample_cubic_hermite(times, q_traj, qd_traj, t)
            self._step_contact_targets(arm_target=q_target, gripper_target=gripper_hold, phase=name)
            q_actual = self._arm_positions()
            error = float(np.linalg.norm(q_actual - q_target))
            log["time"].append(float(t_wall))
            log["plan_time"].append(float(t))
            log["target_q_arm"].append(q_target.tolist())
            log["actual_q_arm"].append(q_actual.tolist())
            log["joint_error_norm"].append(error)
            if step % 20 == 0 or step == steps - 1:
                print(
                    f"[contact-motion:{name}] t={t:.3f}/{duration:.3f} "
                    f"(wall={t_wall:.3f}/{effective_duration:.3f}), joint_error={error:.6f}"
                )
        if name in CONTACT_STRICT_POST_MOTION_WAIT_SEGMENTS:
            wait_log = self._wait_until_arm_reaches(q_traj[-1], name)
        else:
            wait_log = {"converged": True, "reason": "strict wait not required"}
        log["post_motion_wait"] = wait_log
        log["motion_converged"] = bool(wait_log.get("converged", True))
        return log

    def _execute_contact_gripper_segment(
        self,
        segment: dict[str, Any],
        *,
        arm_hold: Any | None,
    ) -> dict[str, Any]:
        import numpy as np

        name = str(segment.get("name", "gripper"))
        q_target = np.asarray(segment.get("target_position", []), dtype=float)
        if q_target.size != 2:
            raise RuntimeError(f"{name}: invalid gripper target {q_target}")
        q_start = self._gripper_positions()
        if arm_hold is None:
            arm_hold = self._arm_positions()
        arm_hold = np.asarray(arm_hold, dtype=float)
        if name == "close_gripper":
            hold_steps = max(1, int(round(CONTACT_PRE_CLOSE_HOLD_DURATION / max(self.dt, 1.0e-6))))
            print("[contact-gripper:close_gripper] hold arm at grasp pose before closing")
            for step in range(hold_steps):
                self._step_contact_targets(arm_target=arm_hold, gripper_target=q_start, phase="pre_close_hold")
                if step == 0 or step == hold_steps - 1:
                    q_actual = self._arm_positions()
                    error = float(np.linalg.norm(q_actual - arm_hold))
                    print(f"[contact-hold:pre_close] step={step:03d}, joint_error={error:.6f}")
        move_steps = max(2, int(round(CONTACT_GRIPPER_MOVE_DURATION / max(self.dt, 1.0e-6))))
        hold_steps = max(1, int(round(CONTACT_GRIPPER_HOLD_DURATION / max(self.dt, 1.0e-6))))
        log = {
            "name": name,
            "type": "gripper",
            "target_position": q_target.tolist(),
            "sim_dt": self.dt,
            "time": [],
            "actual_q_gripper": [],
        }
        print(f"[contact-gripper:{name}] start={q_start}, target={q_target}")
        for step in range(move_steps):
            u = step / float(max(1, move_steps - 1))
            s = self._smoothstep5(u)
            q_cmd = (1.0 - s) * q_start + s * q_target
            self._step_contact_targets(arm_target=arm_hold, gripper_target=q_cmd, phase=name)
            q_actual = self._gripper_positions()
            log["time"].append(step * self.dt)
            log["actual_q_gripper"].append(q_actual.tolist())
            if step % 20 == 0 or step == move_steps - 1:
                error = float(np.linalg.norm(q_actual - q_target))
                print(f"[contact-gripper:{name}] step={step:03d}, q_actual={q_actual}, error={error:.6f}")
        for _ in range(hold_steps):
            self._step_contact_targets(arm_target=arm_hold, gripper_target=q_target, phase=f"{name}_hold")
            q_actual = self._gripper_positions()
            log["time"].append(len(log["time"]) * self.dt)
            log["actual_q_gripper"].append(q_actual.tolist())
        q_final = self._gripper_positions()
        log["final_position"] = q_final.tolist()
        log["final_error"] = float(np.linalg.norm(q_final - q_target))
        if name == "close_gripper":
            close_progress = self._close_progress(q_start, q_final, q_target)
            log["close_progress"] = close_progress
            log["min_close_progress"] = CONTACT_GRIPPER_MIN_CLOSE_PROGRESS
            log["close_success"] = close_progress >= CONTACT_GRIPPER_MIN_CLOSE_PROGRESS
            if not log["close_success"]:
                log["abort_reason"] = (
                    "close_gripper did not make enough progress: "
                    f"close_progress={close_progress:.3f}"
                )
        return log

    def _execute_contact_grasp_plan(self, plan: dict[str, Any], task: Any) -> dict[str, Any]:
        import numpy as np

        segments = list(plan.get("segments", []))
        if not segments:
            raise RuntimeError("grasp plan has no segments")
        plan_summary = plan.get("summary", {})
        if not plan_summary.get("all_motion_segments_success", False):
            raise RuntimeError("grasp plan contains failed motion segments")

        logs: list[dict[str, Any]] = []
        executed_segments: list[dict[str, Any]] = []
        last_motion_q_final = None
        abort_reason = None
        for segment in segments:
            segment_type = segment.get("type")
            if segment_type == "motion":
                motion_log = self._execute_contact_motion_segment(segment)
                logs.append(motion_log)
                executed_segments.append(segment)
                q = np.asarray(segment.get("trajectory", {}).get("q", []), dtype=float)
                if q.size:
                    last_motion_q_final = q[-1].copy()
                if (
                    segment.get("name") in CONTACT_STRICT_POST_MOTION_WAIT_SEGMENTS
                    and not motion_log.get("motion_converged", False)
                ):
                    final_error = None
                    wait_log = motion_log.get("post_motion_wait", {})
                    if wait_log.get("joint_error_norm"):
                        final_error = wait_log["joint_error_norm"][-1]
                    abort_reason = (
                        f"{segment.get('name')} did not converge before gripper close; "
                        f"final_joint_error={final_error}"
                    )
                    print("[contact-grasp abort]", abort_reason)
                    break
            elif segment_type == "gripper":
                gripper_log = self._execute_contact_gripper_segment(segment, arm_hold=last_motion_q_final)
                logs.append(gripper_log)
                executed_segments.append(segment)
                if segment.get("name") == "close_gripper" and not gripper_log.get("close_success", True):
                    abort_reason = gripper_log.get("abort_reason", "close_gripper failed")
                    print("[contact-grasp abort]", abort_reason)
                    break
            else:
                raise RuntimeError(f"unknown grasp segment type: {segment_type}")

        for _ in range(max(1, int(round(0.20 / max(self.dt, 1.0e-6))))):
            self._step_contact_targets(
                arm_target=last_motion_q_final if last_motion_q_final is not None else self._arm_positions(),
                gripper_target=self._gripper_positions(),
                phase="contact_grasp_hold_final",
            )

        current_object = self.monitor.get_object_state()
        before_object = self.monitor.object_state_before_grasp
        object_retreat_success = False
        object_retreat_delta = None
        if current_object is not None and before_object is not None:
            object_retreat_delta = float(np.linalg.norm(np.asarray(current_object.position) - np.asarray(before_object.position)))
            object_retreat_success = object_retreat_delta >= CONTACT_OBJECT_RETREAT_SUCCESS_THRESHOLD_M
        lift_report = self.monitor.get_lift_report()
        has_lift_segment = any(
            segment.get("type") == "motion" and segment.get("name") == "lift_object"
            for segment in segments
        )
        has_planned_retreat = any(
            segment.get("type") == "motion" and segment.get("name") == "retreat_object"
            for segment in segments
        )
        grasp_mode = plan.get("grasp_mode") or plan_summary.get("grasp_mode")
        if abort_reason is not None:
            task_success = False
        elif grasp_mode == "side" and has_planned_retreat and not has_lift_segment:
            task_success = object_retreat_success
        elif has_lift_segment:
            task_success = bool(lift_report.get("object_lifted", False))
        else:
            task_success = object_retreat_success

        summary = {
            "task_success": bool(task_success),
            "abort_reason": abort_reason,
            "grasp_mode": grasp_mode,
            "has_lift_segment": has_lift_segment,
            "has_planned_retreat": has_planned_retreat,
            "object_lift_success": bool(lift_report.get("object_lifted", False)),
            "object_lift_report": lift_report,
            "object_retreat_success": object_retreat_success,
            "object_retreat_delta_m": object_retreat_delta,
            "object_retreat_success_threshold_m": CONTACT_OBJECT_RETREAT_SUCCESS_THRESHOLD_M,
            "execution_backend": "isaac_lab_contact_runtime",
            "executed_segment_names": [segment.get("name") for segment in executed_segments],
            "runtime_notes": self.runtime_notes,
        }
        return {
            "schema_version": 1,
            "success": bool(task_success),
            "object_prim_path": task.object_prim_path,
            "execution_logs": logs,
            "summary": summary,
        }

    def execute_pick(self):
        from source.manipulation import GraspPipeline, GraspPipelineConfig, GraspTask
        from source.pipeline import PhaseResult

        self.stop_base()
        self._set_manip_base_lock(True, reason="execute_pick")
        os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "0"
        os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0"
        os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "0"
        if self.args.side_pregrasp_offset is not None:
            os.environ["GO2_X5_SIDE_PREGRASP_OFFSET_M"] = str(float(self.args.side_pregrasp_offset))
        if self.args.tip_tcp_insertion is not None:
            os.environ["GO2_X5_TIP_TCP_INSERTION_BEYOND_GRASP_CENTER_M"] = str(float(self.args.tip_tcp_insertion))
        pipeline = GraspPipeline(
            GraspPipelineConfig(
                workspace=PROJECT_ROOT,
                curobo_python=os.environ.get("GO2_X5_CUROBO_PYTHON", DEFAULT_ISAAC_PYTHON),
            )
        )
        grasp_task = GraspTask(
            object_prim_path=self.task.pick.object_prim_path,
            grasp_mode=self.task.pick.grasp_mode,
            use_planner_server=bool(self.args.use_planner_server),
        )
        try:
            state = _run_kit_coroutine(
                pipeline.export_state(grasp_task),
                self.simulation_app,
                timeout_s=600.0,
                label="contact pick state export",
            )
            live_patch_report = _patch_grasp_state_from_live_robot(
                state=state,
                adapter=self.adapter,
                task=self.task,
                pipeline=pipeline,
                exclude_distractor_collision=bool(self.args.hide_distractor_objects),
            )
            Path(grasp_task.state_json).write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("[contact-pipeline] grasp state live pose patch:", live_patch_report)
            target = _run_kit_coroutine(
                pipeline.generate_target(grasp_task),
                self.simulation_app,
                timeout_s=600.0,
                label="contact pick target generation",
            )
            plan = pipeline.plan(grasp_task)
            execution = self._execute_contact_grasp_plan(plan, grasp_task)
            Path(grasp_task.result_json).write_text(
                json.dumps(execution, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summary = execution.get("summary", {})
            result = {
                "success": bool(summary.get("task_success", False)),
                "state": state,
                "target": target,
                "plan": plan,
                "execution": execution,
                "live_pose_patch": live_patch_report,
            }
        except Exception as exc:
            return PhaseResult(
                success=False,
                failure_reason="pick_execution_failed",
                metadata={
                    "state": {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "runtime_notes": self.runtime_notes,
                    }
                },
            )
        self._record_grasp_logs(result.get("execution", {}))
        plan_summary = result.get("plan", {}).get("summary", {})
        summary = result.get("execution", {}).get("summary", {})
        success = bool(result.get("success", False))
        return PhaseResult(
            success=success,
            failure_reason="" if success else str(summary.get("abort_reason") or "grasp_not_completed"),
            metadata={
                "state": {
                    "plan_summary": plan_summary,
                    "grasp_summary": summary,
                    "live_pose_patch": result.get("live_pose_patch", {}),
                    "runtime_notes": self.runtime_notes,
                },
                "observation": self._observation(),
                "action": self._action(controller="curobo"),
            },
        )

    def move_to_carry_posture(self, posture_name: str):
        from source.pipeline import PhaseResult

        self.stop_base()
        self._set_manip_base_lock(True, reason="move_to_carry_posture")
        self.arm_target = CARRY_POSTURES.get(posture_name, CARRY_POSTURES["carry_high"])
        self.gripper_target = GRIPPER_CLOSE_TARGET
        for _ in range(90):
            self.stop_base()
            self._step_policy(controller="pd", phase="move_to_carry_posture")
        return PhaseResult(
            success=True,
            metadata={
                "state": {
                    "carry_posture_name": posture_name,
                    "carry_posture_target": list(self.arm_target),
                    "runtime_notes": self.runtime_notes,
                },
                "observation": self._observation(),
                "action": self._action(controller="pd"),
            },
        )

    def hold_carry_posture(self, phase: Any):
        from source.pipeline import RuntimeStepResult

        if phase.value == "carry_nav_to_place":
            self._set_manip_base_lock(False, reason="carry_navigation")
        self.arm_target = CARRY_POSTURES.get(self.args.carry_posture_name, CARRY_POSTURES["carry_high"])
        self.gripper_target = GRIPPER_CLOSE_TARGET
        self.stop_base()
        self._step_policy(controller="pd", phase=phase.value)
        return RuntimeStepResult(
            observation=self._observation(),
            action=self._action(controller="pd"),
            state={"runtime_notes": self.runtime_notes},
            timestamp=time.time(),
            images=self._images(),
        )

    def execute_place(self):
        from source.pipeline import PhaseResult

        self.stop_base()
        self._set_manip_base_lock(True, reason="execute_place")
        self.gripper_target = GRIPPER_OPEN_TARGET
        settle_steps = self.args.place_settle_steps
        if settle_steps is None:
            settle_steps = self.task.place.settle_steps
        for _ in range(max(1, int(settle_steps))):
            self._step_policy(controller="pd", phase="gripper_open")
            self.recorder.record(
                phase="gripper_open",
                observation=self._observation(),
                action=self._action(controller="pd", gripper_cmd="open"),
                state={"runtime_notes": self.runtime_notes},
                timestamp=time.time(),
                images=self._images(),
            )
        return PhaseResult(
            success=True,
            metadata={
                "state": {"place_release": "gripper_open_contact_only", "runtime_notes": self.runtime_notes},
                "observation": self._observation(),
                "action": self._action(controller="pd", gripper_cmd="open"),
            },
        )

    def verify_place(self):
        from source.pipeline import PhaseResult

        object_state = self.monitor.get_object_state()
        target = self.task.place.place_pose_world
        if object_state is None or target is None:
            return PhaseResult(False, "object_state_unavailable", {"state": {"runtime_notes": self.runtime_notes}})
        xy_error = math.hypot(object_state.position[0] - target.x, object_state.position[1] - target.y)
        z_error = abs(object_state.position[2] - target.z)
        success = xy_error <= self.args.place_xy_tolerance and z_error <= self.args.place_z_tolerance
        return PhaseResult(
            success=success,
            failure_reason="" if success else "object_out_of_place",
            metadata={
                "state": {
                    "place_xy_error": xy_error,
                    "place_z_error": z_error,
                    "place_xy_tolerance": self.args.place_xy_tolerance,
                    "place_z_tolerance": self.args.place_z_tolerance,
                    "runtime_notes": self.runtime_notes,
                },
                "observation": self._observation(),
                "action": self._action(controller="pd", gripper_cmd="open"),
            },
        )

    def _ensure_planner(self, goal: Any) -> None:
        from source.navigation import NavPlanner

        key = (float(goal.x), float(goal.y), float(goal.yaw))
        if self.planner is not None and self.current_goal_key == key:
            return
        self.planner = NavPlanner(
            str(self.nav_map),
            self.args.inflate_radius,
            _dwa_config(self.args, self.dt),
            local_clearance_radius=self.args.local_clearance_radius,
        )
        pose = self.adapter.get_base_pose()
        self.path_world = self.planner.plan_global_path(pose[:2], (goal.x, goal.y))
        if self.path_world and math.hypot(self.path_world[-1][0] - goal.x, self.path_world[-1][1] - goal.y) > 0.01:
            self.path_world.append((goal.x, goal.y))
        self.current_goal_key = key

    def _nav_command(self, goal: Any, controller: str) -> tuple[float, float, float]:
        if controller == "terminal":
            return self._terminal_command(goal)
        pose = self.adapter.get_base_pose()
        speed = self.adapter.get_base_velocity()
        vx, vy, wz, _debug = self.planner.compute_command_with_debug(pose, speed, self.path_world)
        return vx, vy, wz

    def _terminal_command(self, goal: Any) -> tuple[float, float, float]:
        from source.navigation.adapters.frame_utils import wrap_yaw
        from source.navigation.adapters.yaw_align import TerminalPoseConfig, body_goal_components, compute_terminal_pose_command

        pose = self.adapter.get_base_pose()
        body_goal_x, body_goal_y = body_goal_components(pose, (goal.x, goal.y))
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        distance = math.hypot(goal.x - pose[0], goal.y - pose[1])
        config = TerminalPoseConfig(
            position_tolerance=self.args.terminal_position_tolerance,
            position_acceptance_tolerance=self.args.goal_tolerance + max(0.0, self.args.final_goal_tolerance_margin),
            yaw_tolerance=self.args.terminal_yaw_tolerance,
            position_kp=self.args.terminal_position_kp,
            max_vx=self.args.terminal_max_vx,
            min_vx=self.args.terminal_min_vx,
            allow_reverse=self.args.terminal_allow_reverse,
            lateral_kp=self.args.terminal_lateral_kp,
            lateral_deadband=self.args.terminal_lateral_deadband,
            max_vy=self.args.terminal_max_vy,
            min_vy=self.args.terminal_min_vy,
            yaw_kp=self.args.terminal_yaw_kp,
            yaw_min_wz=self.args.terminal_yaw_min_wz,
            yaw_max_wz=self.args.terminal_yaw_max_wz,
        )
        return compute_terminal_pose_command(
            body_goal_x=body_goal_x,
            body_goal_y=body_goal_y,
            yaw_error=yaw_error,
            distance_to_goal=distance,
            config=config,
        )

    def _step_policy(self, *, controller: str, phase: str) -> None:
        self.adapter.set_arm_joint_target(self.arm_target)
        self.adapter.set_gripper_joint_target(self.gripper_target)
        self.adapter.step()
        if self.args.debug_print_every > 0 and self.recorder.frame_count % self.args.debug_print_every == 0:
            pose = self.adapter.get_base_pose()
            print(f"[contact-pipeline] phase={phase} pose=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) controller={controller}")

    def _observation(self) -> dict[str, Any]:
        robot = self.adapter.robot
        base_pose = self.adapter.get_base_pose_full()
        base_velocity_body = self.adapter.get_base_velocity_full()
        arm_joint_positions = []
        arm_joint_velocities = []
        gripper_positions = []
        if len(self.adapter.arm_joint_ids) == 6:
            arm_joint_positions = _tensor_list(robot.data.joint_pos[0, self.adapter.arm_joint_ids])
            arm_joint_velocities = _tensor_list(robot.data.joint_vel[0, self.adapter.arm_joint_ids])
        if len(self.adapter.gripper_joint_ids) == 2:
            gripper_positions = _tensor_list(robot.data.joint_pos[0, self.adapter.gripper_joint_ids])
        return {
            "base_pose_world": base_pose,
            "base_velocity_body": {
                "vx": base_velocity_body[0],
                "vy": base_velocity_body[1],
                "wz": base_velocity_body[2],
            },
            "arm_joint_positions": arm_joint_positions,
            "arm_joint_velocities": arm_joint_velocities,
            "gripper_state": {
                "joint_positions": gripper_positions,
                "command": "close" if tuple(self.gripper_target) == GRIPPER_CLOSE_TARGET else "open",
            },
            "ee_pose_world": self.monitor.get_ee_pose_world(),
            "object_pose_world": self.monitor.get_object_pose_world(),
            "object_velocity_world": self.monitor.get_object_velocity(),
        }

    def _action(self, *, controller: str, gripper_cmd: str | None = None) -> dict[str, Any]:
        return {
            "base_velocity_cmd": [float(self.last_command[0]), float(self.last_command[1]), float(self.last_command[2])],
            "arm_joint_cmd": list(self.arm_target),
            "gripper_cmd": gripper_cmd or ("close" if tuple(self.gripper_target) == GRIPPER_CLOSE_TARGET else "open"),
            "controller": controller,
        }

    def _images(self) -> dict[str, bytes | None]:
        images: dict[str, bytes | None] = {}
        if self.args.front_camera:
            images["front"] = _jpeg_bytes(self.adapter.get_front_rgb())
        return images

    def _record_grasp_logs(self, execution: dict[str, Any]) -> None:
        for log in execution.get("execution_logs", []):
            if log.get("type") == "motion":
                phase = "lift" if log.get("name") == "lift_object" else "pick_approach"
                for target, actual in zip(log.get("target_q_arm", []), log.get("actual_q_arm", [])):
                    observation = self._observation()
                    observation["arm_joint_positions"] = actual
                    self.recorder.record(
                        phase=phase,
                        observation=observation,
                        action={
                            "base_velocity_cmd": [0.0, 0.0, 0.0],
                            "arm_joint_cmd": target,
                            "gripper_cmd": "close",
                            "controller": "curobo",
                        },
                        state={"runtime_notes": self.runtime_notes},
                        timestamp=time.time(),
                    )
            elif log.get("type") == "gripper":
                phase = "gripper_close" if log.get("name") == "close_gripper" else "pick_prepare"
                target = log.get("target_position", [])
                for actual in log.get("actual_q_gripper", []):
                    observation = self._observation()
                    observation["gripper_state"] = {"joint_positions": actual}
                    self.recorder.record(
                        phase=phase,
                        observation=observation,
                        action={
                            "base_velocity_cmd": [0.0, 0.0, 0.0],
                            "arm_joint_cmd": [],
                            "gripper_cmd": "close" if phase == "gripper_close" else "open",
                            "gripper_joint_cmd": target,
                            "controller": "pd",
                        },
                        state={"runtime_notes": self.runtime_notes},
                        timestamp=time.time(),
                    )


def _run_with_app(args: argparse.Namespace, simulation_app: Any) -> None:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import robot_lab.tasks  # noqa: F401
    from scripts.navigation import isaaclab_cli_args
    from source.data import VLAEpisodeRecorder, load_task
    from source.manipulation import ContactGraspMonitor, ContactGraspMonitorConfig, RigidBodyState
    from source.navigation.adapters.isaaclab_go2_adapter import Go2LocomotionAdapter
    from source.pipeline import ContactPickPlaceLimits, ContactPickPlaceStateMachine

    task_path = _project_path(args.task_json)
    task = load_task(task_path)
    raw_task = json.loads(task_path.read_text(encoding="utf-8"))
    scene_usd = _project_path(task.scene_usd)
    nav_map = _project_path(args.nav_map or task.nav_map)
    checkpoint = retrieve_file_path(_validate_checkpoint(args.checkpoint))

    if args.carry_mode != "contact" or task.carry.mode != "contact":
        raise ValueError("run_contact_pick_place_once only supports contact carry mode")

    @hydra_task_config(args.task, args.agent)
    def _main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
        _load_scene_sublayer(args, scene_usd)
        object_pose_report = _apply_object_pose_from_task(task)
        _configure_env(args, env_cfg, scene_usd, task, sim_utils)
        agent_cfg = isaaclab_cli_args.update_rsl_rl_cfg(agent_cfg, args)
        env_cfg.seed = agent_cfg.seed
        env = gym.make(args.task, cfg=env_cfg)
        hide_nav_collision_visual_arg = getattr(args, "hide_nav_collision_visual", None)
        hide_nav_collision_visual = (
            hide_nav_collision_visual_arg
            if hide_nav_collision_visual_arg is not None
            else bool(args.demo_visuals or args.load_visual_scene)
        )
        if hide_nav_collision_visual:
            _hide_nav_collision_visual()
        if args.hide_distractor_objects:
            _hide_distractor_objects(task)
        has_gui_viewport = not bool(getattr(args, "headless", False))
        perspective_synced = False
        if has_gui_viewport and args.sync_perspective_camera:
            perspective_synced = _sync_perspective_from_source_scene(
                scene_usd,
                args.perspective_camera_prim,
                env.unwrapped.sim,
            )
        set_viewport_camera = (
            args.set_viewport_camera
            if args.set_viewport_camera is not None
            else has_gui_viewport
        )
        if set_viewport_camera:
            if args.viewport_camera_mode == "stage":
                _set_viewport_stage_camera(args.viewport_camera_prim)
            else:
                if not perspective_synced:
                    _set_perspective_from_stage_camera(args.viewport_camera_prim, env.unwrapped.sim)
                _set_active_viewport_camera_path("/OmniverseKit_Persp")
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

        def ee_reader() -> RigidBodyState | None:
            if not adapter.ee_body_ids:
                return None
            body_id = adapter.ee_body_ids[0]
            robot = adapter.robot
            return RigidBodyState(
                position=tuple(float(value) for value in _tensor_list(robot.data.body_pos_w[0, body_id]))[:3],
                quat_wxyz=tuple(float(value) for value in _tensor_list(robot.data.body_quat_w[0, body_id]))[:4],
                linear_velocity=None,
                angular_velocity=None,
            )

        monitor = ContactGraspMonitor(
            object_prim_path=task.pick.object_prim_path or "/World/apple",
            ee_state_reader=ee_reader,
            config=ContactGraspMonitorConfig(
                min_lift_height=args.min_lift_height,
                max_slip_distance=args.max_slip_distance,
                object_drop_height_threshold=task.carry.object_drop_height_threshold,
            ),
        )
        recorder = VLAEpisodeRecorder(
            args.dataset_dir,
            task_id=task.task_id,
            episode_id=task.episode_id,
            record_every_n_steps=args.record_every_n_steps,
        )
        recorder.save_task(raw_task)
        runtime = IsaacLabContactRuntime(
            args=args,
            task=task,
            adapter=adapter,
            nav_map=nav_map,
            recorder=recorder,
            monitor=monitor,
            simulation_app=simulation_app,
            object_pose_report=object_pose_report,
        )
        limits = ContactPickPlaceLimits(
            max_nav_to_pick_steps=args.max_nav_to_pick_steps,
            max_carry_nav_steps=args.max_carry_nav_steps,
            goal_tolerance=args.goal_tolerance,
            goal_yaw_tolerance=args.goal_yaw_tolerance,
            terminal_position_tolerance=args.terminal_position_tolerance,
            terminal_yaw_tolerance=args.terminal_yaw_tolerance,
            final_goal_tolerance_margin=args.final_goal_tolerance_margin,
            final_yaw_tolerance_margin=args.final_yaw_tolerance_margin,
            place_xy_tolerance=args.place_xy_tolerance,
            place_z_tolerance=args.place_z_tolerance,
            verify_grasp_steps=args.verify_grasp_steps,
            verify_carry_every_steps=task.carry.verify_carry_every_steps,
        )
        machine = ContactPickPlaceStateMachine(
            task=task,
            runtime=runtime,
            monitor=monitor,
            recorder=recorder,
            limits=limits,
            carry_posture_name=args.carry_posture_name,
        )
        machine.summary["task_json"] = str(task_path)
        summary = machine.run()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        env.close()

    _main()


def main() -> int:
    args, hydra_args, app_launcher_cls = _parse_args()
    if not args.task_json:
        raise SystemExit("--task-json is required")
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    try:
        _run_with_app(args, simulation_app)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
