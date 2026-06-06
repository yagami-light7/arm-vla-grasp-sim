#!/usr/bin/env python3
"""Coordinate the two-process Go2-X5 navigation-to-pick demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data import load_task
from source.navigation import NavPlanner
from source.navigation.navlib import DWAConfig


PIPELINE_CONTEXT_JSON = Path("/tmp/go2_x5_pipeline_context.json")
DEFAULT_NAV_RESULT_JSON = Path("/tmp/go2_x5_nav_result.json")
DEFAULT_LOCAL_CHECKPOINT = PROJECT_ROOT / "checkpoints/go2_x5/flat/model_8500.pt"
LEGACY_TMP_CHECKPOINT = Path("/tmp/DWA-reference/flat/model_8500.pt")
DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_checkpoint_path() -> str | None:
    """Return the best available local Go2-X5 checkpoint path."""

    env_checkpoint = os.environ.get("GO2_X5_CHECKPOINT")
    if env_checkpoint:
        return env_checkpoint
    for candidate in (DEFAULT_LOCAL_CHECKPOINT, LEGACY_TMP_CHECKPOINT):
        if candidate.is_file():
            return str(candidate)
    return None


def _validate_checkpoint(raw_path: str | None) -> str:
    """Return an existing local checkpoint path before starting Isaac Lab."""

    if not raw_path:
        raise ValueError(
            "--checkpoint is required unless --dry-run or --grasp-only is used. "
            f"Set GO2_X5_CHECKPOINT or place model_8500.pt at {DEFAULT_LOCAL_CHECKPOINT}."
        )
    if "://" in raw_path:
        return raw_path
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Go2-X5 locomotion checkpoint does not exist: {checkpoint}\n"
            "Pass the real RSL-RL Go2-X5 checkpoint path. "
            "Do not use the documentation placeholder '/你的实际路径/model_8500.pt'. "
            f"Default local path: {DEFAULT_LOCAL_CHECKPOINT}"
        )
    return str(checkpoint)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", default="tasks/nav_pick_example.json")
    parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
    parser.add_argument(
        "--checkpoint",
        default=_default_checkpoint_path(),
        help=(
            "Go2-X5 RSL-RL checkpoint. Defaults to GO2_X5_CHECKPOINT, then "
            f"{DEFAULT_LOCAL_CHECKPOINT}, then {LEGACY_TMP_CHECKPOINT}."
        ),
    )
    parser.add_argument("--map", dest="scene_usd", default=None)
    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--add-nav-ground", action="store_true")
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--isaaclab-python", default=sys.executable)
    parser.add_argument("--isaaclab-launcher", default=None, help="Optional Isaac Lab isaaclab.sh launcher; adds '-p'.")
    parser.add_argument(
        "--isaac-python",
        default=os.environ.get("GO2_X5_ISAAC_PYTHON", DEFAULT_ISAAC_PYTHON),
        help="Python executable used for the standalone Isaac Sim grasp runner.",
    )
    parser.add_argument("--nav-result", default=str(DEFAULT_NAV_RESULT_JSON))
    parser.add_argument("--nav-only", action="store_true")
    parser.add_argument("--grasp-only", action="store_true")
    parser.add_argument("--manual-grasp", action="store_true", help="Do not launch standalone grasp; print Script Editor commands.")
    parser.add_argument(
        "--handoff-smoke-only",
        action="store_true",
        help="In the Isaac Sim handoff script, restore the base pose and export state without planning or executing grasp.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--use-planner-server", action="store_true")
    parser.add_argument(
        "--demo-visuals",
        action="store_true",
        help="Enable visual scene loading, a fixed overview camera, and non-headless grasp for interactive demos.",
    )
    parser.add_argument("--grasp-headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-retreat-success", action="store_true", help="Legacy debug mode: side retreat may count as pick success.")
    parser.add_argument("--legacy-side-retreat", action="store_true", help="Plan side grasp retreat instead of vertical lift.")
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true", help="Fallback to side retreat if vertical lift planning fails.")
    parser.add_argument("--head-camera", action="store_true")
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
        default="sublayer",
        help=(
            "Use the stable referenced visual asset by default, or preload a full-scene "
            "sublayer for SAGE visuals before Isaac Lab creates PhysX tensor views."
        ),
    )
    parser.add_argument("--visual-prim-path", default="/World/gauss")
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-camera-mode", choices=("chase", "front", "overhead", "fixed", "stage"), default="chase")
    parser.add_argument("--viewport-camera-prim", default="/World/camera_main")
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
        help="World-space camera eye used once when --follow-camera-mode fixed is selected.",
    )
    parser.add_argument(
        "--fixed-camera-lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-space camera target used once when --follow-camera-mode fixed is selected.",
    )
    parser.add_argument("--flat-terrain", action="store_true", help="Use the locomotion task's flat terrain for debugging.")
    parser.add_argument("--disable-sky-light", action="store_true", help="Disable the default Isaac Lab sky light.")
    parser.add_argument("--debug-command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--max-nav-steps", type=int, default=5000)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--stall-window-steps", type=int, default=240)
    parser.add_argument("--stall-min-progress", type=float, default=0.05)
    parser.add_argument("--stall-min-forward-command", type=float, default=0.05)
    parser.add_argument("--stall-min-forward-ratio", type=float, default=0.25)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15)
    parser.add_argument("--inflate-radius", type=float, default=0.25)
    parser.add_argument("--local-clearance-radius", type=float, default=0.20)
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
    parser.add_argument("--yaw-align-start-distance", type=float, default=0.65)
    parser.add_argument("--yaw-align-activation-yaw-error", type=float, default=0.0)
    parser.add_argument("--yaw-align-allow-reverse", action="store_true")
    parser.add_argument("--yaw-align-stall-window-steps", type=int, default=240)
    parser.add_argument("--yaw-align-min-progress", type=float, default=0.08)
    parser.add_argument("--yaw-settle-stable-steps", type=int, default=20)
    parser.add_argument("--yaw-settle-kp", type=float, default=0.8)
    parser.add_argument("--yaw-settle-min-wz", type=float, default=0.0)
    parser.add_argument("--yaw-settle-max-wz", type=float, default=0.35)
    parser.add_argument("--yaw-settle-realign-margin", type=float, default=0.08)
    parser.add_argument("--save-replay-trajectory", action="store_true")
    parser.add_argument("--replay-sample-every", type=int, default=1)
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--replay-trajectory-name", default="trajectory.jsonl")
    parser.add_argument(
        "--replay-include-initial-settle",
        action="store_true",
        help="Include the pre-navigation zero-command settle segment in the replay trajectory.",
    )
    parser.add_argument("--debug-print-every", type=int, default=60)
    return parser.parse_args()


def _pick_script_editor_command(*, smoke_only: bool = False) -> str:
    script = PROJECT_ROOT / "scripts/isaac/run_pick_from_nav_result.py"
    env_line = (
        'os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "1"\n'
        if smoke_only
        else 'os.environ["GO2_X5_HANDOFF_SMOKE_ONLY"] = "0"\n'
    )
    return (
        "import os\n"
        'os.environ["GO2_X5_HANDOFF_FORCE_RECORD"] = "1"\n'
        f"{env_line}"
        f'_script = "{script}"\n'
        'exec(compile(open(_script, "r", encoding="utf-8").read(), _script, "exec"), '
        '{"__file__": _script, "__name__": "__main__"})'
    )


def _write_context(args: argparse.Namespace, task_json: Path, task) -> None:
    PIPELINE_CONTEXT_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_json": str(task_json),
                "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                "nav_map": str(_project_path(args.nav_map or task.nav_map)),
                "terrain_prim_path": args.terrain_prim_path,
                "add_nav_ground": args.add_nav_ground,
                "ground_height": args.ground_height,
                "handoff_clearance_radius": max(args.inflate_radius, args.local_clearance_radius),
                "goal_tolerance": args.goal_tolerance,
                "lookahead_distance": args.lookahead_distance,
                "prediction_horizon": args.prediction_horizon,
                "max_lin_vel": args.max_lin_vel,
                "max_ang_vel": args.max_ang_vel,
                "brisk_nav": args.brisk_nav,
                "min_active_lin_vel": args.min_active_lin_vel,
                "near_goal_min_active_lin_vel": args.near_goal_min_active_lin_vel,
                "close_goal_speed_limit": args.close_goal_speed_limit,
                "speed_bias": args.speed_bias,
                "max_linear_accel": args.max_linear_accel,
                "yaw_align_kp": args.yaw_align_kp,
                "yaw_align_min_wz": args.yaw_align_min_wz,
                "yaw_align_max_wz": args.yaw_align_max_wz,
                "yaw_align_vx": args.yaw_align_vx,
                "yaw_align_max_vx": args.yaw_align_max_vx,
                "yaw_align_position_kp": args.yaw_align_position_kp,
                "yaw_align_max_vy": args.yaw_align_max_vy,
                "yaw_align_lateral_kp": args.yaw_align_lateral_kp,
                "yaw_align_lateral_deadband": args.yaw_align_lateral_deadband,
                "yaw_align_start_distance": args.yaw_align_start_distance,
                "yaw_align_activation_yaw_error": args.yaw_align_activation_yaw_error,
                "yaw_align_allow_reverse": args.yaw_align_allow_reverse,
                "yaw_align_stall_window_steps": args.yaw_align_stall_window_steps,
                "yaw_align_min_progress": args.yaw_align_min_progress,
                "yaw_settle_stable_steps": args.yaw_settle_stable_steps,
                "yaw_settle_kp": args.yaw_settle_kp,
                "yaw_settle_min_wz": args.yaw_settle_min_wz,
                "yaw_settle_max_wz": args.yaw_settle_max_wz,
                "yaw_settle_realign_margin": args.yaw_settle_realign_margin,
                "save_replay_trajectory": args.save_replay_trajectory,
                "replay_sample_every": args.replay_sample_every,
                "replay_output": args.replay_output,
                "replay_trajectory_name": args.replay_trajectory_name,
                "replay_include_initial_settle": args.replay_include_initial_settle,
                "use_planner_server": args.use_planner_server,
                "handoff_smoke_only": args.handoff_smoke_only,
                "require_object_lift_success": not args.allow_retreat_success,
                "legacy_side_retreat": args.legacy_side_retreat,
                "side_grasp_fallback_retreat": args.side_grasp_fallback_retreat,
                "settle_steps": args.settle_steps,
                "dataset_dir": args.dataset_dir,
                "load_visual_scene": args.load_visual_scene,
                "visual_load_mode": args.visual_load_mode,
                "visual_prim_path": args.visual_prim_path,
                "follow_camera": args.follow_camera,
                "follow_camera_mode": args.follow_camera_mode,
                "viewport_camera_prim": args.viewport_camera_prim,
                "follow_camera_distance": args.follow_camera_distance,
                "follow_camera_height": args.follow_camera_height,
                "follow_camera_side": args.follow_camera_side,
                "fixed_camera_preset": args.fixed_camera_preset,
                "fixed_camera_close_distance": args.fixed_camera_close_distance,
                "fixed_camera_close_height": args.fixed_camera_close_height,
                "fixed_camera_close_side": args.fixed_camera_close_side,
                "fixed_camera_eye": args.fixed_camera_eye,
                "fixed_camera_lookat": args.fixed_camera_lookat,
                "no_record": args.no_record,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _nav_command(args: argparse.Namespace, task) -> list[str]:
    script = PROJECT_ROOT / "scripts/navigation/run_nav_only.py"
    prefix = [args.isaaclab_launcher, "-p"] if args.isaaclab_launcher else [args.isaaclab_python]
    command = [
        *prefix,
        str(script),
        "--task-json",
        str(_project_path(args.task_json)),
        "--task",
        args.task,
        "--checkpoint",
        str(args.checkpoint),
        "--terrain-prim-path",
        args.terrain_prim_path,
        "--ground-height",
        str(args.ground_height),
        "--nav-result",
        str(Path(args.nav_result).expanduser().resolve()),
        "--max-nav-steps",
        str(args.max_nav_steps),
        "--settle-steps",
        str(args.settle_steps),
        "--stall-window-steps",
        str(args.stall_window_steps),
        "--stall-min-progress",
        str(args.stall_min_progress),
        "--stall-min-forward-command",
        str(args.stall_min_forward_command),
        "--stall-min-forward-ratio",
        str(args.stall_min_forward_ratio),
        "--goal-tolerance",
        str(args.goal_tolerance),
        "--goal-yaw-tolerance",
        str(args.goal_yaw_tolerance),
        "--inflate-radius",
        str(args.inflate_radius),
        "--local-clearance-radius",
        str(args.local_clearance_radius),
        "--lookahead-distance",
        str(args.lookahead_distance),
        "--prediction-horizon",
        str(args.prediction_horizon),
        "--max-lin-vel",
        str(args.max_lin_vel),
        "--max-ang-vel",
        str(args.max_ang_vel),
        "--min-active-lin-vel",
        str(args.min_active_lin_vel),
        "--near-goal-min-active-lin-vel",
        str(args.near_goal_min_active_lin_vel),
        "--close-goal-speed-limit",
        str(args.close_goal_speed_limit),
        "--speed-bias",
        str(args.speed_bias),
        "--max-linear-accel",
        str(args.max_linear_accel),
        "--yaw-align-kp",
        str(args.yaw_align_kp),
        "--yaw-align-min-wz",
        str(args.yaw_align_min_wz),
        "--yaw-align-max-wz",
        str(args.yaw_align_max_wz),
        "--yaw-align-vx",
        str(args.yaw_align_vx),
        "--yaw-align-max-vx",
        str(args.yaw_align_max_vx),
        "--yaw-align-position-kp",
        str(args.yaw_align_position_kp),
        "--yaw-align-max-vy",
        str(args.yaw_align_max_vy),
        "--yaw-align-lateral-kp",
        str(args.yaw_align_lateral_kp),
        "--yaw-align-lateral-deadband",
        str(args.yaw_align_lateral_deadband),
        "--yaw-align-start-distance",
        str(args.yaw_align_start_distance),
        "--yaw-align-activation-yaw-error",
        str(args.yaw_align_activation_yaw_error),
        "--yaw-align-stall-window-steps",
        str(args.yaw_align_stall_window_steps),
        "--yaw-align-min-progress",
        str(args.yaw_align_min_progress),
        "--yaw-settle-stable-steps",
        str(args.yaw_settle_stable_steps),
        "--yaw-settle-kp",
        str(args.yaw_settle_kp),
        "--yaw-settle-min-wz",
        str(args.yaw_settle_min_wz),
        "--yaw-settle-max-wz",
        str(args.yaw_settle_max_wz),
        "--yaw-settle-realign-margin",
        str(args.yaw_settle_realign_margin),
        "--debug-print-every",
        str(args.debug_print_every),
    ]
    if args.scene_usd:
        command.extend(["--map", args.scene_usd])
    if args.add_nav_ground:
        command.append("--add-nav-ground")
    if args.nav_map:
        command.extend(["--nav-map", args.nav_map])
    if args.dataset_dir:
        command.extend(["--dataset-dir", args.dataset_dir])
    if args.no_record:
        command.append("--no-record")
    if args.head_camera:
        command.append("--head-camera")
    if args.save_replay_trajectory:
        command.append("--save-replay-trajectory")
        command.extend(["--replay-sample-every", str(args.replay_sample_every)])
        command.extend(["--replay-trajectory-name", args.replay_trajectory_name])
        if args.replay_output:
            command.extend(["--replay-output", args.replay_output])
        if args.replay_include_initial_settle:
            command.append("--replay-include-initial-settle")
    if args.load_visual_scene or args.demo_visuals:
        command.extend(
            [
                "--load-visual-scene",
                "--visual-load-mode",
                args.visual_load_mode,
                "--visual-prim-path",
                args.visual_prim_path,
            ]
        )
    if args.hide_nav_collision_visual is not None:
        command.append("--hide-nav-collision-visual" if args.hide_nav_collision_visual else "--no-hide-nav-collision-visual")
    if args.brisk_nav:
        command.append("--brisk-nav")
    follow_camera_mode = args.follow_camera_mode
    follow_camera_distance = args.follow_camera_distance
    follow_camera_height = args.follow_camera_height
    if args.demo_visuals and args.follow_camera_mode == "chase":
        follow_camera_mode = "fixed"
    command.extend(
        [
            "--follow-camera" if args.follow_camera else "--no-follow-camera",
            "--follow-camera-mode",
            follow_camera_mode,
            "--viewport-camera-prim",
            args.viewport_camera_prim,
            "--follow-camera-distance",
            str(follow_camera_distance),
            "--follow-camera-height",
            str(follow_camera_height),
            "--follow-camera-side",
            str(args.follow_camera_side),
            "--fixed-camera-preset",
            args.fixed_camera_preset,
            "--fixed-camera-close-distance",
            str(args.fixed_camera_close_distance),
            "--fixed-camera-close-height",
            str(args.fixed_camera_close_height),
            "--fixed-camera-close-side",
            str(args.fixed_camera_close_side),
        ]
    )
    if args.fixed_camera_eye is not None:
        command.extend(["--fixed-camera-eye", *(str(value) for value in args.fixed_camera_eye)])
    if args.fixed_camera_lookat is not None:
        command.extend(["--fixed-camera-lookat", *(str(value) for value in args.fixed_camera_lookat)])
    if args.flat_terrain:
        command.append("--flat-terrain")
    if args.disable_sky_light:
        command.append("--disable-sky-light")
    if args.yaw_align_allow_reverse:
        command.append("--yaw-align-allow-reverse")
    if args.debug_command:
        command.extend(["--debug-command", *(str(value) for value in args.debug_command)])
    return command


def _standalone_pick_command(args: argparse.Namespace, task, *, smoke_only: bool = False) -> list[str]:
    script = PROJECT_ROOT / "scripts/isaac/run_pick_from_nav_result_standalone.py"
    command = [
        args.isaac_python,
        str(script),
        "--task-json",
        str(_project_path(args.task_json)),
        "--scene-usd",
        str(_project_path(args.scene_usd or task.scene_usd)),
        "--nav-map",
        str(_project_path(args.nav_map or task.nav_map)),
        "--nav-result",
        str(Path(args.nav_result).expanduser().resolve()),
        "--terrain-prim-path",
        args.terrain_prim_path,
        "--handoff-clearance-radius",
        str(max(args.inflate_radius, args.local_clearance_radius)),
        "--settle-steps",
        str(args.settle_steps),
    ]
    if args.dataset_dir:
        command.extend(["--dataset-dir", args.dataset_dir])
    if args.use_planner_server:
        command.append("--use-planner-server")
    if smoke_only or args.handoff_smoke_only:
        command.append("--handoff-smoke-only")
    if args.no_record:
        command.append("--no-record")
    if not args.allow_retreat_success:
        command.append("--require-lift-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if args.grasp_headless and not args.demo_visuals:
        command.append("--headless")
    return command


def _dwa_config_from_args(args: argparse.Namespace, control_dt: float) -> DWAConfig:
    """Build the dry-run DWA config from pipeline CLI args."""

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
    return DWAConfig(
        control_dt=control_dt,
        lookahead_distance=args.lookahead_distance,
        prediction_horizon=args.prediction_horizon,
        goal_tolerance=args.goal_tolerance,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=args.max_ang_vel,
        min_active_linear_velocity=min_active_linear_velocity,
        near_goal_min_active_linear_velocity=near_goal_min_active_linear_velocity,
        close_goal_speed_limit=close_goal_speed_limit,
        speed_bias=speed_bias,
        max_linear_accel=max_linear_accel,
    )


def _dry_run(args: argparse.Namespace, task) -> None:
    nav_map = _project_path(args.nav_map or task.nav_map)
    plan_summary: dict[str, object] = {
        "state_machine": [
            "INIT",
            "LOAD_NAV_TASK",
            "NAV_TO_PICK_BASE",
            "YAW_ALIGN",
            "SETTLE_BASE",
            "EXPORT_GRASP_STATE",
            "GENERATE_GRASP_TARGET",
            "PLAN_GRASP",
            "EXECUTE_GRASP",
            "CHECK_PICK_SUCCESS",
            "SAVE_EPISODE",
        ],
        "task": task,
        "nav_map": str(nav_map),
    }
    if nav_map.exists():
        planner = NavPlanner(
            str(nav_map),
            args.inflate_radius,
            _dwa_config_from_args(args, control_dt=0.05),
            local_clearance_radius=args.local_clearance_radius,
        )
        path_world = planner.plan_global_path((task.start.x, task.start.y), (task.pick.base_goal.x, task.pick.base_goal.y))
        plan_summary["global_path_world"] = path_world
    else:
        plan_summary["warning"] = f"nav map does not exist yet: {nav_map}"
    print(json.dumps(plan_summary, indent=2, ensure_ascii=False, default=lambda value: value.__dict__))


def main() -> int:
    # 解析命令行参数
    args = _parse_args()

    # 加载任务
    task_json = _project_path(args.task_json)
    task = load_task(task_json)

    # 写入信息 用于沟通导航阶段和抓取阶段之间的任务上下文
    _write_context(args, task_json, task)

    # 根据参数决定执行流程
    if args.dry_run:
        _dry_run(args, task)
        return 0
    if args.grasp_only:
        if args.manual_grasp:
            if args.handoff_smoke_only:
                print("[handoff] Run this smoke-check command in Isaac Sim Script Editor:")
                print(_pick_script_editor_command(smoke_only=True))
            else:
                print("[handoff] Run this full-pick command in Isaac Sim Script Editor:")
                print(_pick_script_editor_command(smoke_only=False))
            return 0
        command = _standalone_pick_command(args, task, smoke_only=args.handoff_smoke_only)
        print("[pipeline] launching standalone grasp:")
        print(" ".join(command))
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        return 0

    if args.manual_grasp and args.handoff_smoke_only:
        print("[handoff] Run this smoke-check command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=True))
        return 0

    # 验证 checkpoint 路径
    args.checkpoint = _validate_checkpoint(args.checkpoint)

    # 启动导航 后续导航由 run_nav_only.py 完成
    command = _nav_command(args, task)
    print("[pipeline] launching navigation:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)

    # 导航结果写到 nav_result.json
    nav_result = json.loads(Path(args.nav_result).expanduser().resolve().read_text(encoding="utf-8"))

    # 检查导航结果并提示后续步骤
    if not nav_result.get("success", False):
        print("[pipeline] navigation failed:", nav_result.get("failure_reason"))
        return 1
    if args.nav_only:
        print("[pipeline] navigation complete:", args.nav_result)
        print("[handoff] First run this smoke-check command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=True))
        print("[handoff] After smoke passes, run this full-pick command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=False))
        return 0

    if args.manual_grasp:
        print("[pipeline] navigation complete. Run this smoke-check command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=True))
        print("[handoff] After smoke passes, run this full-pick command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=False))
        return 0

    command = _standalone_pick_command(args, task, smoke_only=False)
    print("[pipeline] navigation complete. launching standalone grasp:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
