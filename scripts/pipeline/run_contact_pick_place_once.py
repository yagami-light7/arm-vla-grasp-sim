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
    parser.add_argument("--carry-mode", choices=("contact",), default="contact")
    parser.add_argument("--carry-posture-name", default="carry_high")
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
    parser.add_argument("--terminal-yaw-start-distance", type=float, default=0.65)
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
    if args.front_camera or args.wrist_camera or args.third_camera:
        args.enable_cameras = True
    if args.demo_visuals:
        args.load_visual_scene = True
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
        body_ids, _ = adapter.robot.find_bodies([body_name], preserve_order=True)
    except TypeError:
        body_ids, _ = adapter.robot.find_bodies([body_name])
    if not body_ids:
        return None
    body_id = int(body_ids[0])
    position = [float(value) for value in _tensor_list(adapter.robot.data.body_pos_w[0, body_id])[:3]]
    quat_wxyz = [float(value) for value in _tensor_list(adapter.robot.data.body_quat_w[0, body_id])[:4]]
    matrix = pose_to_matrix(np.asarray(position, dtype=float), np.asarray(quat_wxyz, dtype=float))
    return matrix


def _patch_grasp_state_from_live_robot(
    *,
    state: dict[str, Any],
    adapter: Any,
    task: Any,
    pipeline: Any,
) -> dict[str, Any]:
    """Patch exported grasp state with live Isaac Lab body transforms.

    The generic Isaac Sim exporter reads USD xforms. In Isaac Lab, PhysX tensor
    state is authoritative during a running RL environment, while USD xforms can
    remain at the reset/authored transform. Target generation must therefore use
    the live arm_base_link pose after navigation.
    """

    import numpy as np
    import omni.usd
    from scripts.math.SE3 import pose_dict_from_matrix, xyz_rpy_to_matrix

    report: dict[str, Any] = {"applied": False}
    T_world_base = _live_body_matrix(adapter, "arm_base_link")
    if T_world_base is None:
        report["reason"] = "arm_base_link body not found in Isaac Lab robot"
        return report

    T_world_tcp = _live_body_matrix(adapter, "grasp_tcp_link")
    if T_world_tcp is None:
        T_world_arm_link6 = _live_body_matrix(adapter, "arm_link6")
        if T_world_arm_link6 is not None:
            T_arm_link6_tcp = xyz_rpy_to_matrix((0.15757, 0.0, 0.0), (0.0, 0.0, 0.0))
            T_world_tcp = T_world_arm_link6 @ T_arm_link6_tcp
            report["tcp_source"] = "live_arm_link6_plus_fixed_offset"
        else:
            old_base_tcp = state.get("poses", {}).get("base_tcp", {})
            old_position = old_base_tcp.get("position_xyz")
            old_quat = old_base_tcp.get("quaternion_wxyz")
            if old_position is None or old_quat is None:
                report["reason"] = "tcp body not found and exported base_tcp missing"
                return report
            from scripts.math.SE3 import pose_to_matrix

            T_base_tcp_old = pose_to_matrix(np.asarray(old_position, dtype=float), np.asarray(old_quat, dtype=float))
            T_world_tcp = T_world_base @ T_base_tcp_old
            report["tcp_source"] = "exported_base_tcp_recomposed_with_live_base"
    else:
        report["tcp_source"] = "live_grasp_tcp_link_body"

    T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp
    state.setdefault("poses", {})["world_base"] = pose_dict_from_matrix(T_world_base)
    state["poses"]["world_tcp"] = pose_dict_from_matrix(T_world_tcp)
    state["poses"]["base_tcp"] = pose_dict_from_matrix(T_base_tcp)
    state["source"] = "Isaac Lab contact runtime with live PhysX tensor pose patch"

    try:
        export_module = pipeline._load_module("go2_x5_export_state_live_patch", pipeline.script_export)
        stage = omni.usd.get_context().get_stage()
        selected_paths = [task.pick.object_prim_path] if task.pick.object_prim_path else []
        cuboids = export_module.compute_world_collision_cuboids(
            stage=stage,
            robot_root_path=state.get("paths", {}).get("robot_root_path", ""),
            T_world_base=T_world_base,
            selected_paths=selected_paths,
        )
        world_collision = state.setdefault("world_collision", {})
        world_collision["cuboids_base"] = cuboids
        world_collision["live_pose_recomputed"] = True
        world_collision["excluded_selected_prim_paths"] = selected_paths
        report["world_collision_recomputed"] = True
        report["world_collision_cuboids"] = len(cuboids)
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
        self.arm_target = CARRY_POSTURES["stow"]
        self.gripper_target = GRIPPER_OPEN_TARGET
        self.runtime_notes = {
            "attachment_mode": "contact_only",
            "object_pose_write_policy": "reset_only",
            "fixed_joint_default": False,
            "kinematic_follow": False,
            "place_controller": "mvp_release_at_current_carry_posture",
            "pick_success_policy": "side_grasp_retreat",
        }

    def reset_episode(self, task: Any):
        from source.pipeline import PhaseResult

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

        self._ensure_planner(goal)
        pose = self.adapter.get_base_pose()
        distance = math.hypot(goal.x - pose[0], goal.y - pose[1])
        yaw_error = wrap_yaw(goal.yaw - pose[2])
        reached_before = (
            distance <= self.args.goal_tolerance + max(0.0, self.args.final_goal_tolerance_margin)
            and abs(yaw_error) <= self.args.goal_yaw_tolerance + max(0.0, self.args.final_yaw_tolerance_margin)
        )
        controller = "terminal" if distance <= max(self.args.goal_tolerance, self.args.terminal_yaw_start_distance) else "dwa"
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
        reached_after = (
            math.hypot(goal.x - pose_after[0], goal.y - pose_after[1])
            <= self.args.goal_tolerance + max(0.0, self.args.final_goal_tolerance_margin)
            and abs(wrap_yaw(goal.yaw - pose_after[2]))
            <= self.args.goal_yaw_tolerance + max(0.0, self.args.final_yaw_tolerance_margin)
        )
        return RuntimeStepResult(
            observation=self._observation(),
            action=self._action(controller=controller),
            state={
                "goal_xyyaw": [goal.x, goal.y, goal.yaw],
                "goal_distance": math.hypot(goal.x - pose_after[0], goal.y - pose_after[1]),
                "yaw_error": wrap_yaw(goal.yaw - pose_after[2]),
                "runtime_notes": self.runtime_notes,
            },
            timestamp=time.time(),
            images=self._images(),
            reached_goal=reached_after,
        )

    def stop_base(self) -> None:
        self.last_command = (0.0, 0.0, 0.0)
        self.adapter.apply_base_command(0.0, 0.0, 0.0)

    def settle_base(self, phase: Any):
        from source.pipeline import RuntimeStepResult

        self.stop_base()
        self._step_policy(controller="pd", phase=phase.value)
        return RuntimeStepResult(
            observation=self._observation(),
            action=self._action(controller="pd"),
            state={"runtime_notes": self.runtime_notes},
            timestamp=time.time(),
            images=self._images(),
            reached_goal=False,
        )

    def execute_pick(self):
        from source.manipulation import GraspPipeline, GraspPipelineConfig, GraspTask
        from source.pipeline import PhaseResult

        os.environ["GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS"] = "0"
        os.environ["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0"
        os.environ["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "0"
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
            execution = _run_kit_coroutine(
                pipeline.execute(grasp_task),
                self.simulation_app,
                timeout_s=600.0,
                label="contact pick execution",
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
