#!/usr/bin/env python3
"""Coordinate the two-process Go2-X5 navigation-to-pick demo."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data import load_task
from source.navigation import NavPlanner
from source.navigation.navlib import DWAConfig


PIPELINE_CONTEXT_JSON = Path("/tmp/go2_x5_pipeline_context.json")
DEFAULT_NAV_RESULT_JSON = Path("/tmp/go2_x5_nav_result.json")
DEFAULT_HANDOFF_REPORT_JSON = Path("/tmp/go2_x5_handoff_report.json")
DEFAULT_LOCAL_CHECKPOINT = PROJECT_ROOT / "checkpoints/go2_x5/flat/model_8500.pt"
LEGACY_TMP_CHECKPOINT = Path("/tmp/DWA-reference/flat/model_8500.pt")
DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
RAW_CLI_ARGS = sys.argv[1:].copy()
APPLE_FAST_TASK_JSON = "tasks/nav_pick_apple_fast.json"
APPLE_FAST_DATASET_DIR = "/tmp/nav_pick_apple_fast"
APPLE_FAST_NAV_RESULT_JSON = "/tmp/go2_x5_nav_pick_apple_fast_result.json"
DEFAULT_PLANNER_SERVER_HOST = "127.0.0.1"
DEFAULT_PLANNER_SERVER_PORT = 8765
DEFAULT_PLANNER_SERVER_LOG = Path("/tmp/go2_x5_curobo_planner_server.log")


def _project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _cli_arg_supplied(flag: str) -> bool:
    """Return whether a command-line flag was explicitly supplied."""

    return any(arg == flag or arg.startswith(f"{flag}=") for arg in RAW_CLI_ARGS)


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
    parser.add_argument(
        "--preset",
        choices=("apple-fast",),
        default=None,
        help="Apply a stable demo preset. User-supplied CLI values keep precedence.",
    )
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
    parser.add_argument("--handoff-report", default=str(DEFAULT_HANDOFF_REPORT_JSON))
    parser.add_argument(
        "--nav-headless",
        action="store_true",
        help="Run Isaac Lab navigation without a visible window. Useful with --replay-nav-before-grasp for video capture.",
    )
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
    parser.add_argument(
        "--use-planner-server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a running cuRobo grasp_planner_server when available; falls back to one-shot planning.",
    )
    parser.add_argument(
        "--auto-start-planner-server",
        action="store_true",
        help="Start grasp_planner_server before navigation so it can warm up before grasp planning.",
    )
    parser.add_argument(
        "--restart-planner-server",
        action="store_true",
        help="When auto-starting, shut down an existing planner server first so new policy/env flags take effect.",
    )
    parser.add_argument("--planner-server-log", default=str(DEFAULT_PLANNER_SERVER_LOG))
    parser.add_argument("--planner-server-start-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--demo-visuals",
        action="store_true",
        help="Enable visual scene loading, a fixed overview camera, and non-headless grasp for interactive demos.",
    )
    parser.add_argument("--grasp-headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-retreat-success", action="store_true", help="Legacy debug mode: side retreat may count as pick success.")
    parser.add_argument("--legacy-side-retreat", action="store_true", help="Plan side grasp retreat instead of vertical lift.")
    parser.add_argument(
        "--side-retreat-only",
        action="store_true",
        help="For side grasps, skip vertical lift and count the planned reverse retreat as pick success.",
    )
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true", help="Fallback to side retreat if vertical lift planning fails.")
    parser.add_argument(
        "--keep-window-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the standalone Isaac Sim grasp window open after the run; defaults to enabled with --demo-visuals.",
    )
    parser.add_argument(
        "--show-grasp-trajectory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show planned TCP path markers during grasp execution. Disabled by default for clean video capture.",
    )
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
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
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
    parser.add_argument("--terminal-position-tolerance", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--final-goal-tolerance-margin", type=float, default=0.03)
    parser.add_argument("--final-yaw-tolerance-margin", type=float, default=0.03)
    parser.add_argument("--inflate-radius", type=float, default=0.25)
    parser.add_argument("--local-clearance-radius", type=float, default=0.20)
    parser.add_argument(
        "--handoff-clearance-radius",
        type=float,
        default=None,
        help=(
            "Obstacle clearance radius checked before restoring the nav pose in the "
            "Isaac Sim grasp stage. Defaults to max(--inflate-radius, --local-clearance-radius)."
        ),
    )
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
    parser.add_argument("--yaw-align-kp", type=float, default=1.2)
    parser.add_argument("--yaw-align-min-wz", type=float, default=0.40)
    parser.add_argument("--yaw-align-max-wz", type=float, default=1.00)
    parser.add_argument("--yaw-align-vx", type=float, default=0.15)
    parser.add_argument("--yaw-align-max-vx", type=float, default=0.35)
    parser.add_argument("--yaw-align-position-kp", type=float, default=0.1)
    parser.add_argument("--yaw-align-max-vy", type=float, default=0.35)
    parser.add_argument("--yaw-align-min-vy", type=float, default=0.15)
    parser.add_argument("--yaw-align-lateral-kp", type=float, default=0.9)
    parser.add_argument("--yaw-align-lateral-deadband", type=float, default=0.03)
    parser.add_argument("--yaw-align-start-distance", type=float, default=0.70)
    parser.add_argument("--yaw-align-activation-yaw-error", type=float, default=0.0)
    parser.add_argument("--yaw-align-allow-reverse", action="store_true")
    parser.add_argument("--yaw-align-stall-window-steps", type=int, default=120)
    parser.add_argument("--yaw-align-min-progress", type=float, default=0.08)
    parser.add_argument("--yaw-settle-stable-steps", type=int, default=15)
    parser.add_argument("--yaw-settle-kp", type=float, default=0.8)
    parser.add_argument("--yaw-settle-min-wz", type=float, default=0.40)
    parser.add_argument("--yaw-settle-max-wz", type=float, default=0.60)
    parser.add_argument("--yaw-settle-realign-margin", type=float, default=0.08)
    parser.add_argument("--base-stable-linear-tolerance", type=float, default=0.06)
    parser.add_argument("--base-stable-angular-tolerance", type=float, default=0.20)
    parser.add_argument("--terminal-allow-reverse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminal-yaw-slowdown-error", type=float, default=0.65)
    parser.add_argument("--terminal-yaw-slowdown-min-wz", type=float, default=0.20)
    parser.add_argument("--terminal-yaw-slowdown-max-wz", type=float, default=0.45)
    parser.add_argument("--terminal-large-yaw-error", type=float, default=1.00)
    parser.add_argument("--terminal-large-yaw-position-scale", type=float, default=0.45)
    parser.add_argument("--terminal-gait-vx", type=float, default=0.04)
    parser.add_argument("--terminal-recovery-steps", type=int, default=90)
    parser.add_argument("--terminal-recovery-yaw-max-wz", type=float, default=0.35)
    parser.add_argument("--terminal-recovery-gait-vx", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-polish-vx", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-polish-min-wz", type=float, default=0.45)
    parser.add_argument("--terminal-yaw-polish-max-wz", type=float, default=0.55)
    parser.add_argument("--save-replay-trajectory", action="store_true")
    parser.add_argument("--replay-sample-every", type=int, default=1)
    parser.add_argument("--replay-output", default=None)
    parser.add_argument("--replay-trajectory-name", default="trajectory.jsonl")
    parser.add_argument(
        "--replay-nav-before-grasp",
        action="store_true",
        help="Replay the recorded navigation trajectory in the standalone Isaac Sim grasp window before grasp.",
    )
    parser.add_argument("--replay-nav-real-time", action="store_true")
    parser.add_argument("--replay-nav-speed", type=float, default=1.0)
    parser.add_argument(
        "--replay-include-initial-settle",
        action="store_true",
        help="Include the pre-navigation zero-command settle segment in the replay trajectory.",
    )
    parser.add_argument("--debug-print-every", type=int, default=60)
    parser.add_argument("--profile-dwa", action="store_true")
    parser.add_argument("--profile-print-every", type=int, default=60)
    return parser.parse_args()


def _set_if_not_supplied(args: argparse.Namespace, attr: str, value) -> None:
    """Set an arg value only when its matching CLI flag was not supplied."""

    flag = "--" + attr.replace("_", "-")
    if not _cli_arg_supplied(flag):
        setattr(args, attr, value)


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    """Apply stable demo defaults without overriding explicit CLI arguments."""

    if args.preset != "apple-fast":
        return
    _set_if_not_supplied(args, "task_json", APPLE_FAST_TASK_JSON)
    _set_if_not_supplied(args, "dataset_dir", APPLE_FAST_DATASET_DIR)
    _set_if_not_supplied(args, "nav_result", APPLE_FAST_NAV_RESULT_JSON)
    _set_if_not_supplied(args, "max_nav_steps", 3000)
    _set_if_not_supplied(args, "lookahead_distance", 0.30)
    _set_if_not_supplied(args, "prediction_horizon", 0.45)
    _set_if_not_supplied(args, "goal_tolerance", 0.15)
    _set_if_not_supplied(args, "goal_yaw_tolerance", 0.20)
    _set_if_not_supplied(args, "terminal_position_tolerance", 0.08)
    _set_if_not_supplied(args, "terminal_yaw_tolerance", 0.08)
    _set_if_not_supplied(args, "final_goal_tolerance_margin", 0.03)
    _set_if_not_supplied(args, "final_yaw_tolerance_margin", 0.20)
    _set_if_not_supplied(args, "yaw_align_start_distance", 0.50)
    _set_if_not_supplied(args, "yaw_align_vx", 0.35)
    _set_if_not_supplied(args, "yaw_align_max_vx", 0.60)
    _set_if_not_supplied(args, "yaw_align_position_kp", 0.8)
    _set_if_not_supplied(args, "yaw_align_max_vy", 0.35)
    _set_if_not_supplied(args, "yaw_align_lateral_kp", 0.9)
    _set_if_not_supplied(args, "yaw_align_lateral_deadband", 0.015)
    _set_if_not_supplied(args, "yaw_align_kp", 1.2)
    _set_if_not_supplied(args, "yaw_align_min_wz", 0.40)
    _set_if_not_supplied(args, "yaw_align_max_wz", 0.60)
    _set_if_not_supplied(args, "terminal_yaw_slowdown_max_wz", 0.42)
    _set_if_not_supplied(args, "terminal_recovery_yaw_max_wz", 0.32)
    _set_if_not_supplied(args, "terminal_yaw_polish_vx", 0.08)
    _set_if_not_supplied(args, "terminal_yaw_polish_min_wz", 0.45)
    _set_if_not_supplied(args, "terminal_yaw_polish_max_wz", 0.55)
    _set_if_not_supplied(args, "handoff_clearance_radius", 0.20)
    _set_if_not_supplied(args, "settle_steps", 120)
    _set_if_not_supplied(args, "yaw_settle_stable_steps", 15)
    _set_if_not_supplied(args, "yaw_settle_max_wz", 0.25)
    args.brisk_nav = True
    args.fast_dwa = True


def _apply_derived_defaults(args: argparse.Namespace) -> None:
    """Apply cross-option defaults that make demo modes predictable."""

    if args.demo_visuals and args.replay_nav_before_grasp and not _cli_arg_supplied("--nav-headless"):
        args.nav_headless = True
    if getattr(args, "keep_window_open", None) is None:
        args.keep_window_open = bool(args.demo_visuals)
    if getattr(args, "side_retreat_only", False):
        args.legacy_side_retreat = True
        args.allow_retreat_success = True
    if getattr(args, "auto_start_planner_server", False):
        args.use_planner_server = True


def _handoff_clearance_radius(args: argparse.Namespace) -> float:
    """Return the grasp handoff clearance radius for nav-result validation."""

    if args.handoff_clearance_radius is not None:
        return float(args.handoff_clearance_radius)
    return max(float(args.inflate_radius), float(args.local_clearance_radius))


def _pipeline_grasp_resume_command(args: argparse.Namespace, *, smoke_only: bool) -> str:
    """Return a shell command for resuming from an existing nav result."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/pipeline/run_nav_then_pick.py"),
        "--task-json",
        str(_project_path(args.task_json)),
        "--grasp-only",
        "--nav-result",
        str(Path(args.nav_result).expanduser().resolve()),
        "--handoff-report",
        str(Path(args.handoff_report).expanduser().resolve()),
        "--isaac-python",
        args.isaac_python,
    ]
    if args.dataset_dir:
        command.extend(["--dataset-dir", str(args.dataset_dir)])
    if smoke_only:
        command.append("--handoff-smoke-only")
    if args.use_planner_server:
        command.append("--use-planner-server")
    if getattr(args, "auto_start_planner_server", False):
        command.append("--auto-start-planner-server")
    if getattr(args, "restart_planner_server", False):
        command.append("--restart-planner-server")
    if args.no_record:
        command.append("--no-record")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if getattr(args, "side_retreat_only", False):
        command.append("--side-retreat-only")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if getattr(args, "keep_window_open", False):
        command.append("--keep-window-open")
    if getattr(args, "show_grasp_trajectory", False):
        command.append("--show-grasp-trajectory")
    if args.demo_visuals:
        command.append("--demo-visuals")
    if args.replay_nav_before_grasp:
        command.append("--replay-nav-before-grasp")
        if args.replay_nav_real_time:
            command.append("--replay-nav-real-time")
        command.extend(["--replay-nav-speed", str(args.replay_nav_speed)])
    if args.follow_camera_mode == "stage":
        command.extend(["--follow-camera-mode", "stage", "--viewport-camera-prim", args.viewport_camera_prim])
    return " ".join(shlex.quote(str(part)) for part in command)


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
                "preset": args.preset,
                "task_json": str(task_json),
                "nav_result_json": str(Path(args.nav_result).expanduser().resolve()),
                "handoff_report_json": str(Path(args.handoff_report).expanduser().resolve()),
                "nav_headless": args.nav_headless,
                "nav_map": str(_project_path(args.nav_map or task.nav_map)),
                "terrain_prim_path": args.terrain_prim_path,
                "add_nav_ground": args.add_nav_ground,
                "ground_height": args.ground_height,
                "handoff_clearance_radius": _handoff_clearance_radius(args),
                "goal_tolerance": args.goal_tolerance,
                "goal_yaw_tolerance": args.goal_yaw_tolerance,
                "terminal_position_tolerance": args.terminal_position_tolerance,
                "terminal_yaw_tolerance": args.terminal_yaw_tolerance,
                "final_goal_tolerance_margin": args.final_goal_tolerance_margin,
                "final_yaw_tolerance_margin": args.final_yaw_tolerance_margin,
                "lookahead_distance": args.lookahead_distance,
                "prediction_horizon": args.prediction_horizon,
                "max_lin_vel": args.max_lin_vel,
                "max_ang_vel": args.max_ang_vel,
                "brisk_nav": args.brisk_nav,
                "fast_dwa": args.fast_dwa,
                "dwa_linear_samples": args.dwa_linear_samples,
                "dwa_angular_samples": args.dwa_angular_samples,
                "dwa_integration_dt": args.dwa_integration_dt,
                "dwa_path_sample_spacing": args.dwa_path_sample_spacing,
                "dwa_path_distance_window": args.dwa_path_distance_window,
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
                "yaw_align_min_vy": args.yaw_align_min_vy,
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
                "base_stable_linear_tolerance": args.base_stable_linear_tolerance,
                "base_stable_angular_tolerance": args.base_stable_angular_tolerance,
                "terminal_allow_reverse": args.terminal_allow_reverse,
                "terminal_yaw_slowdown_error": args.terminal_yaw_slowdown_error,
                "terminal_yaw_slowdown_min_wz": args.terminal_yaw_slowdown_min_wz,
                "terminal_yaw_slowdown_max_wz": args.terminal_yaw_slowdown_max_wz,
                "terminal_large_yaw_error": args.terminal_large_yaw_error,
                "terminal_large_yaw_position_scale": args.terminal_large_yaw_position_scale,
                "terminal_gait_vx": args.terminal_gait_vx,
                "terminal_recovery_steps": args.terminal_recovery_steps,
                "terminal_recovery_yaw_max_wz": args.terminal_recovery_yaw_max_wz,
                "terminal_recovery_gait_vx": args.terminal_recovery_gait_vx,
                "terminal_yaw_polish_vx": args.terminal_yaw_polish_vx,
                "terminal_yaw_polish_min_wz": args.terminal_yaw_polish_min_wz,
                "terminal_yaw_polish_max_wz": args.terminal_yaw_polish_max_wz,
                "save_replay_trajectory": args.save_replay_trajectory,
                "replay_sample_every": args.replay_sample_every,
                "replay_output": args.replay_output,
                "replay_trajectory_name": args.replay_trajectory_name,
                "replay_nav_before_grasp": args.replay_nav_before_grasp,
                "replay_nav_real_time": args.replay_nav_real_time,
                "replay_nav_speed": args.replay_nav_speed,
                "replay_include_initial_settle": args.replay_include_initial_settle,
                "profile_dwa": args.profile_dwa,
                "profile_print_every": args.profile_print_every,
                "use_planner_server": args.use_planner_server,
                "auto_start_planner_server": getattr(args, "auto_start_planner_server", False),
                "restart_planner_server": getattr(args, "restart_planner_server", False),
                "planner_server_log": str(Path(getattr(args, "planner_server_log", DEFAULT_PLANNER_SERVER_LOG)).expanduser()),
                "planner_server_start_timeout_s": getattr(args, "planner_server_start_timeout_s", 180.0),
                "handoff_smoke_only": args.handoff_smoke_only,
                "require_object_lift_success": not args.allow_retreat_success,
                "legacy_side_retreat": args.legacy_side_retreat,
                "side_retreat_only": getattr(args, "side_retreat_only", False),
                "side_grasp_fallback_retreat": args.side_grasp_fallback_retreat,
                "keep_window_open": getattr(args, "keep_window_open", False),
                "show_grasp_trajectory": getattr(args, "show_grasp_trajectory", False),
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


def _effective_dwa_scalar(args: argparse.Namespace, name: str, fast_value: float) -> float:
    """Apply fast-DWA scalar defaults unless the user explicitly supplied a value."""

    flag = "--" + name.replace("_", "-")
    return fast_value if args.fast_dwa and not _cli_arg_supplied(flag) else getattr(args, name)


def _nav_command(args: argparse.Namespace, task) -> list[str]:
    script = PROJECT_ROOT / "scripts/navigation/run_nav_only.py"
    prefix = [args.isaaclab_launcher, "-p"] if args.isaaclab_launcher else [args.isaaclab_python]
    lookahead_distance = _effective_dwa_scalar(args, "lookahead_distance", 0.30)
    prediction_horizon = _effective_dwa_scalar(args, "prediction_horizon", 0.45)
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
        "--terminal-position-tolerance",
        str(args.terminal_position_tolerance),
        "--terminal-yaw-tolerance",
        str(args.terminal_yaw_tolerance),
        "--final-goal-tolerance-margin",
        str(args.final_goal_tolerance_margin),
        "--final-yaw-tolerance-margin",
        str(args.final_yaw_tolerance_margin),
        "--inflate-radius",
        str(args.inflate_radius),
        "--local-clearance-radius",
        str(args.local_clearance_radius),
        "--lookahead-distance",
        str(lookahead_distance),
        "--prediction-horizon",
        str(prediction_horizon),
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
        "--yaw-align-min-vy",
        str(args.yaw_align_min_vy),
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
        "--base-stable-linear-tolerance",
        str(args.base_stable_linear_tolerance),
        "--base-stable-angular-tolerance",
        str(args.base_stable_angular_tolerance),
        "--terminal-allow-reverse" if args.terminal_allow_reverse else "--no-terminal-allow-reverse",
        "--terminal-yaw-slowdown-error",
        str(args.terminal_yaw_slowdown_error),
        "--terminal-yaw-slowdown-min-wz",
        str(args.terminal_yaw_slowdown_min_wz),
        "--terminal-yaw-slowdown-max-wz",
        str(args.terminal_yaw_slowdown_max_wz),
        "--terminal-large-yaw-error",
        str(args.terminal_large_yaw_error),
        "--terminal-large-yaw-position-scale",
        str(args.terminal_large_yaw_position_scale),
        "--terminal-gait-vx",
        str(args.terminal_gait_vx),
        "--terminal-recovery-steps",
        str(args.terminal_recovery_steps),
        "--terminal-recovery-yaw-max-wz",
        str(args.terminal_recovery_yaw_max_wz),
        "--terminal-recovery-gait-vx",
        str(args.terminal_recovery_gait_vx),
        "--terminal-yaw-polish-vx",
        str(args.terminal_yaw_polish_vx),
        "--terminal-yaw-polish-min-wz",
        str(args.terminal_yaw_polish_min_wz),
        "--terminal-yaw-polish-max-wz",
        str(args.terminal_yaw_polish_max_wz),
        "--debug-print-every",
        str(args.debug_print_every),
    ]
    if args.nav_headless:
        command.append("--headless")
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
    if args.save_replay_trajectory or args.replay_nav_before_grasp:
        command.append("--save-replay-trajectory")
        command.extend(["--replay-sample-every", str(args.replay_sample_every)])
        command.extend(["--replay-trajectory-name", args.replay_trajectory_name])
        if args.replay_output:
            command.extend(["--replay-output", args.replay_output])
        if args.replay_include_initial_settle:
            command.append("--replay-include-initial-settle")
    if (args.load_visual_scene or args.demo_visuals) and not args.nav_headless:
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
    if args.fast_dwa:
        command.append("--fast-dwa")
    if args.dwa_linear_samples is not None:
        command.extend(["--dwa-linear-samples", str(args.dwa_linear_samples)])
    if args.dwa_angular_samples is not None:
        command.extend(["--dwa-angular-samples", str(args.dwa_angular_samples)])
    if args.dwa_integration_dt is not None:
        command.extend(["--dwa-integration-dt", str(args.dwa_integration_dt)])
    if args.dwa_path_sample_spacing is not None:
        command.extend(["--dwa-path-sample-spacing", str(args.dwa_path_sample_spacing)])
    if args.dwa_path_distance_window is not None:
        command.extend(["--dwa-path-distance-window", str(args.dwa_path_distance_window)])
    if args.profile_dwa:
        command.append("--profile-dwa")
        command.extend(["--profile-print-every", str(args.profile_print_every)])
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
        "--handoff-report",
        str(Path(args.handoff_report).expanduser().resolve()),
        "--terrain-prim-path",
        args.terrain_prim_path,
        "--handoff-clearance-radius",
        str(_handoff_clearance_radius(args)),
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
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if getattr(args, "side_retreat_only", False):
        command.append("--side-retreat-only")
    if not args.allow_retreat_success:
        command.append("--require-lift-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if getattr(args, "keep_window_open", False):
        command.append("--keep-window-open")
    if getattr(args, "show_grasp_trajectory", False):
        command.append("--show-grasp-trajectory")
    if args.replay_nav_before_grasp:
        nav_result = _read_json_if_exists(args.nav_result)
        replay_trajectory = (nav_result or {}).get("replay_trajectory_path")
        if not replay_trajectory:
            raise RuntimeError(
                "--replay-nav-before-grasp requires a nav result containing replay_trajectory_path. "
                "Run navigation with --save-replay-trajectory or let the pipeline run navigation first."
            )
        command.extend(["--replay-trajectory", str(Path(replay_trajectory).expanduser().resolve())])
        if args.replay_nav_real_time:
            command.append("--replay-real-time")
        command.extend(["--replay-speed", str(args.replay_nav_speed)])
    if args.demo_visuals and args.follow_camera_mode == "stage":
        command.extend(["--set-viewport-camera", "--viewport-camera-prim", args.viewport_camera_prim])
    if args.grasp_headless and not args.demo_visuals:
        command.append("--headless")
    return command


def _read_json_if_exists(path: str | Path) -> dict | None:
    """Read a JSON file if present, returning None for missing files."""

    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _handoff_report_path(args: argparse.Namespace) -> Path:
    return Path(args.handoff_report).expanduser().resolve()


def _clear_handoff_report(args: argparse.Namespace) -> None:
    """Remove a stale handoff report before launching a new grasp stage."""

    try:
        _handoff_report_path(args).unlink()
    except FileNotFoundError:
        pass


def _handoff_report_created(args: argparse.Namespace) -> bool:
    return _handoff_report_path(args).exists()


def _handoff_report_success(args: argparse.Namespace) -> bool:
    report = _read_json_if_exists(_handoff_report_path(args))
    return bool(report and report.get("success", False))


def _handoff_report_failure_text(args: argparse.Namespace) -> str:
    report = _read_json_if_exists(_handoff_report_path(args))
    if report is None:
        return f"missing handoff report {_handoff_report_path(args)}"
    return (
        f"handoff report success={report.get('success')} "
        f"failure_reason={report.get('failure_reason', '')} "
        f"failure_detail={report.get('failure_detail', '')}"
    )


def _planner_server_ping(timeout_s: float = 1.0) -> bool:
    """Return whether the cuRobo planner server is already accepting requests."""

    try:
        with socket.create_connection((DEFAULT_PLANNER_SERVER_HOST, DEFAULT_PLANNER_SERVER_PORT), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(b'{"command":"ping"}\n')
            response_line = sock.makefile("r", encoding="utf-8").readline()
    except OSError:
        return False
    if not response_line:
        return False
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError:
        return False
    return bool(response.get("ok", False))


def _planner_server_shutdown(timeout_s: float = 2.0) -> bool:
    """Ask an existing cuRobo planner server to shut down."""

    try:
        with socket.create_connection((DEFAULT_PLANNER_SERVER_HOST, DEFAULT_PLANNER_SERVER_PORT), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(b'{"command":"shutdown"}\n')
            response_line = sock.makefile("r", encoding="utf-8").readline()
    except OSError:
        return False
    if not response_line:
        return False
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError:
        return False
    return bool(response.get("ok", False))


def _start_planner_server_if_requested(args: argparse.Namespace) -> subprocess.Popen | None:
    """Start grasp_planner_server early so grasp planning can reuse a warm planner."""

    if not getattr(args, "auto_start_planner_server", False):
        return None
    if _planner_server_ping():
        if getattr(args, "restart_planner_server", False):
            print("[pipeline] shutting down existing cuRobo planner server before restart")
            _planner_server_shutdown()
            deadline = time.time() + 10.0
            while time.time() < deadline and _planner_server_ping(timeout_s=0.5):
                time.sleep(0.5)
        else:
            print("[pipeline] cuRobo planner server already running on 127.0.0.1:8765")
            return None
    if _planner_server_ping():
        print("[pipeline] cuRobo planner server already running on 127.0.0.1:8765")
        return None

    log_path = Path(args.planner_server_log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GO2_X5_WORKSPACE"] = str(PROJECT_ROOT)
    env.setdefault("GO2_X5_CUROBO_SOURCE_ROOT", "/home/light/workspace/curobo")
    env["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "0" if args.legacy_side_retreat else "1"
    env["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "1" if args.side_grasp_fallback_retreat else "0"
    server_script = PROJECT_ROOT / "scripts/curobo/grasp_planner_server.py"
    python_executable = os.environ.get("GO2_X5_CUROBO_PYTHON", args.isaac_python)
    print(f"[pipeline] starting cuRobo planner server in background; log={log_path}")
    with log_path.open("a", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            [python_executable, str(server_script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process


def _wait_for_planner_server_if_started(args: argparse.Namespace, process: subprocess.Popen | None) -> None:
    """Wait for the optional background planner server before launching grasp."""

    if not getattr(args, "use_planner_server", False):
        return
    if _planner_server_ping():
        return
    if process is None and not getattr(args, "auto_start_planner_server", False):
        return

    timeout_s = float(getattr(args, "planner_server_start_timeout_s", 180.0))
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            print(
                "[pipeline] cuRobo planner server exited before becoming ready; "
                f"returncode={process.returncode}, log={Path(args.planner_server_log).expanduser().resolve()}"
            )
            return
        if _planner_server_ping(timeout_s=1.0):
            print("[pipeline] cuRobo planner server is ready")
            return
        time.sleep(1.0)
    print(
        "[pipeline] cuRobo planner server did not become ready before timeout; "
        f"grasp will fall back to one-shot planning if needed. log={Path(args.planner_server_log).expanduser().resolve()}"
    )


def _episode_summary_path(nav_result: dict | None) -> Path | None:
    if not nav_result:
        return None
    episode_dir = nav_result.get("episode_dir")
    if not episode_dir:
        return None
    return Path(episode_dir).expanduser().resolve() / "summary.json"


def _print_pipeline_summary(
    args: argparse.Namespace,
    task,
    *,
    nav_result: dict | None = None,
    include_handoff: bool = False,
) -> None:
    """Print compact nav, handoff, and grasp diagnostics after a pipeline run."""

    if nav_result is None:
        nav_result = _read_json_if_exists(args.nav_result)
    print("========== Go2-X5 Nav+Pick Pipeline Summary ==========")
    print(f"[summary] task_json={_project_path(args.task_json)}")
    print(f"[summary] object_prim={task.pick.object_prim_path} grasp_mode={task.pick.grasp_mode}")
    print(f"[summary] nav_result={Path(args.nav_result).expanduser().resolve()}")
    if nav_result is None:
        print("[summary] navigation: missing nav result")
    else:
        pose = nav_result.get("final_base_pose_world", {})
        print(
            "[summary] navigation: "
            f"success={nav_result.get('success')} failure_reason={nav_result.get('failure_reason', '')} "
            f"position_reached={nav_result.get('final_position_reached')} "
            f"yaw_aligned={nav_result.get('final_yaw_aligned')} "
            f"base_stable={nav_result.get('base_stable')}"
        )
        print(
            "[summary] final_base_pose: "
            f"x={float(pose.get('x', 0.0)):.3f} y={float(pose.get('y', 0.0)):.3f} "
            f"z={float(pose.get('z', 0.0)):.3f} yaw={float(pose.get('yaw', 0.0)):.3f} "
            f"goal_dist={float(nav_result.get('final_goal_distance', 0.0)):.3f} "
            f"yaw_error={float(nav_result.get('yaw_error', 0.0)):.3f} "
            f"pos_accept={nav_result.get('position_acceptance_tolerance')} "
            f"yaw_accept={nav_result.get('yaw_acceptance_tolerance')}"
        )
        if nav_result.get("replay_trajectory_path"):
            print(
                "[summary] replay: "
                f"path={nav_result.get('replay_trajectory_path')} "
                f"frames={nav_result.get('replay_frame_count')}"
            )

    handoff_report_path = _handoff_report_path(args)
    handoff_report = _read_json_if_exists(handoff_report_path) if include_handoff else None
    if include_handoff:
        print(f"[summary] handoff_report={handoff_report_path}")
        if handoff_report is None:
            print("[summary] handoff: missing report")
        else:
            handoff = handoff_report.get("handoff", {})
            map_check = handoff.get("map_check", {})
            target = handoff.get("target", {})
            workspace = target.get("target_workspace_base", {})
            print(
                "[summary] handoff: "
                f"success={handoff_report.get('success')} failure_reason={handoff_report.get('failure_reason', '')} "
                f"goal_distance_m={map_check.get('goal_distance_m')} "
                f"raw_cell_occupied={map_check.get('raw_cell_occupied')} "
                f"clearance_cell_occupied={map_check.get('clearance_cell_occupied')}"
            )
            if workspace:
                print(f"[summary] target_workspace_base={workspace}")

    summary_path = _episode_summary_path(nav_result)
    print(f"[summary] episode_summary={summary_path}")
    episode_summary = _read_json_if_exists(summary_path) if summary_path is not None else None
    if episode_summary is None:
        print("[summary] episode: missing summary")
        return
    grasp = episode_summary.get("grasp", {})
    plan_summary = grasp.get("plan_summary", {})
    execution_summary = grasp.get("execution_summary", {})
    print(
        "[summary] episode: "
        f"success={episode_summary.get('success')} failure_reason={episode_summary.get('failure_reason', '')} "
        f"mode={episode_summary.get('mode', '')}"
    )
    if grasp:
        print(
            "[summary] grasp: "
            f"plan_success={plan_summary.get('all_motion_segments_success')} "
            f"task_success={execution_summary.get('task_success')} "
            f"abort_reason={execution_summary.get('abort_reason')}"
        )


def _dwa_config_from_args(args: argparse.Namespace, control_dt: float) -> DWAConfig:
    """Build the dry-run DWA config from pipeline CLI args."""

    fast_dwa = bool(args.fast_dwa)
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

    prediction_horizon = _effective_dwa_scalar(args, "prediction_horizon", 0.45)
    lookahead_distance = _effective_dwa_scalar(args, "lookahead_distance", 0.30)
    linear_samples = args.dwa_linear_samples
    angular_samples = args.dwa_angular_samples
    integration_dt = args.dwa_integration_dt
    path_sample_spacing = args.dwa_path_sample_spacing
    path_distance_window = args.dwa_path_distance_window
    if fast_dwa:
        linear_samples = 3 if linear_samples is None else linear_samples
        angular_samples = 7 if angular_samples is None else angular_samples
        integration_dt = 0.05 if integration_dt is None else integration_dt
        path_sample_spacing = 0.08 if path_sample_spacing is None else path_sample_spacing
        path_distance_window = 80 if path_distance_window is None else path_distance_window

    if linear_samples is not None and linear_samples < 2:
        raise ValueError("--dwa-linear-samples must be >= 2.")
    if angular_samples is not None and angular_samples < 3:
        raise ValueError("--dwa-angular-samples must be >= 3.")
    if integration_dt is not None and integration_dt <= 0.0:
        raise ValueError("--dwa-integration-dt must be > 0.")
    if path_sample_spacing is not None and path_sample_spacing <= 0.0:
        raise ValueError("--dwa-path-sample-spacing must be > 0.")
    if path_distance_window is not None and path_distance_window < 1:
        raise ValueError("--dwa-path-distance-window must be >= 1.")

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
    _apply_preset_defaults(args)
    _apply_derived_defaults(args)

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
        planner_server_process = _start_planner_server_if_requested(args)
        _wait_for_planner_server_if_started(args, planner_server_process)
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
        _clear_handoff_report(args)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        _print_pipeline_summary(args, task, include_handoff=True)
        print(f"[pipeline] standalone grasp returncode={completed.returncode}")
        if completed.returncode == 0 and not _handoff_report_success(args):
            print(f"[pipeline] grasp failed: {_handoff_report_failure_text(args)}")
            return 1
        return completed.returncode

    if args.manual_grasp and args.handoff_smoke_only:
        print("[handoff] Run this smoke-check command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=True))
        return 0

    # 验证 checkpoint 路径
    args.checkpoint = _validate_checkpoint(args.checkpoint)

    planner_server_process = None
    if not args.nav_only:
        planner_server_process = _start_planner_server_if_requested(args)

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
        _print_pipeline_summary(args, task, nav_result=nav_result, include_handoff=False)
        return 1
    if args.nav_only:
        print("[pipeline] navigation complete:", args.nav_result)
        _print_pipeline_summary(args, task, nav_result=nav_result, include_handoff=False)
        if args.manual_grasp:
            print("[handoff] First run this smoke-check command in Isaac Sim Script Editor:")
            print(_pick_script_editor_command(smoke_only=True))
            print("[handoff] After smoke passes, run this full-pick command in Isaac Sim Script Editor:")
            print(_pick_script_editor_command(smoke_only=False))
        else:
            print("[handoff] First run this smoke-check command:")
            print(_pipeline_grasp_resume_command(args, smoke_only=True))
            print("[handoff] After smoke passes, run this full-pick command:")
            print(_pipeline_grasp_resume_command(args, smoke_only=False))
        return 0

    if args.manual_grasp:
        print("[pipeline] navigation complete. Run this smoke-check command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=True))
        print("[handoff] After smoke passes, run this full-pick command in Isaac Sim Script Editor:")
        print(_pick_script_editor_command(smoke_only=False))
        return 0

    command = _standalone_pick_command(args, task, smoke_only=False)
    _wait_for_planner_server_if_started(args, planner_server_process)
    print("[pipeline] navigation complete. launching standalone grasp:")
    print(" ".join(command))
    _clear_handoff_report(args)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    _print_pipeline_summary(args, task, nav_result=nav_result, include_handoff=True)
    print(f"[pipeline] standalone grasp returncode={completed.returncode}")
    if completed.returncode == 0 and not _handoff_report_success(args):
        print(f"[pipeline] grasp failed: {_handoff_report_failure_text(args)}")
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
