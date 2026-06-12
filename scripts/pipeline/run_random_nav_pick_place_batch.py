#!/usr/bin/env python3
"""Run randomized multi-process JSON nav-pick-place episodes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.data.random_task import (
    DEFAULT_APPROACH_ANGLES_DEG,
    DEFAULT_STANDOFF_CANDIDATES_M,
    RandomTaskGenerationError,
    SpawnRegion,
    write_random_pick_task,
)


DEFAULT_ISAAC_PYTHON = "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python"
DEFAULT_CHECKPOINT = "checkpoints/go2_x5/flat/model_8500.pt"
DEFAULT_ISAACLAB_LAUNCHER = "/home/light/workspace/IsaacLab/isaaclab.sh"
DEFAULT_APPLE_FIXED_Z_M = 0.81653
DEFAULT_APPLE_FIXED_RPY_DEG = (-2.524, -7.822, -0.181)
DEFAULT_PLACE_TEMPLATE_TASK = "tasks/nav_pick_place_apple_contact.json"
DEFAULT_OUTPUT_TASK_DIR = "outputs/random_tasks/apple_pick_place_07_far_manual"
DEFAULT_DATASET_ROOT = "outputs/random_pick_place_dataset/apple_pick_place_07_far_manual"
DEFAULT_PICK_PLACE_BASE_GOAL_OFFSET_XY_M = (0.35, -0.08)
DEFAULT_PICK_X_RANGE_M = (0.83, 0.93)
DEFAULT_PICK_Y_RANGE_M = (1.00, 1.50)
DEFAULT_PLACE_X_RANGE_M = (0.65285, 0.75285)
DEFAULT_PLACE_Y_RANGE_M = (5.00337, 5.50337)
HANDOFF_MODE = "multi_process_json"
PHYSICAL_CONTINUITY = False


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _command_path(raw_path: str | Path) -> str:
    raw_text = str(raw_path)
    if "/" not in raw_text:
        return raw_text
    return str(_project_path(raw_path))


def _project_path_or_uri(raw_path: str | Path) -> str:
    raw_text = str(raw_path)
    if "://" in raw_text:
        return raw_text
    return str(_project_path(raw_path))


def _format_seed(seed: int) -> str:
    return f"{seed:04d}" if seed >= 0 else f"neg{abs(seed):04d}"


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_summary_line(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-task", default="tasks/nav_pick_apple_fast.json")
    parser.add_argument(
        "--place-template-task",
        default=DEFAULT_PLACE_TEMPLATE_TASK,
        help="Task JSON used as a place-goal template when --base-task is pick-only.",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-task-dir", default=DEFAULT_OUTPUT_TASK_DIR)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--task", default="RobotLab-Isaac-Velocity-Flat-Go2-X5-Foundation-v0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--isaaclab-launcher", default=DEFAULT_ISAACLAB_LAUNCHER)
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--pipeline-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precompute-nav-first", action="store_true")

    parser.add_argument("--nav-headless", action="store_true")
    parser.add_argument("--demo-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-camera-mode", choices=("chase", "front", "overhead", "fixed", "stage"), default="stage")
    parser.add_argument("--viewport-camera-prim", default="/World/Camera_main")
    parser.add_argument("--replay-nav-before-grasp", action="store_true")
    parser.add_argument("--replay-nav-real-time", action="store_true")
    parser.add_argument("--replay-nav-speed", type=float, default=1.0)
    parser.add_argument("--keep-window-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-grasp-trajectory", action="store_true")
    parser.add_argument(
        "--show-randomization-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show pick/place XY ranges and sampled episode positions as non-physical USD guide markers.",
    )

    parser.add_argument("--table-x-range", type=float, nargs=2, default=DEFAULT_PICK_X_RANGE_M, metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--table-y-range", type=float, nargs=2, default=DEFAULT_PICK_Y_RANGE_M, metavar=("Y_MIN", "Y_MAX"))
    parser.add_argument("--place-x-range", type=float, nargs=2, default=DEFAULT_PLACE_X_RANGE_M, metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--place-y-range", type=float, nargs=2, default=DEFAULT_PLACE_Y_RANGE_M, metavar=("Y_MIN", "Y_MAX"))
    parser.add_argument("--table-z", type=float, default=0.81653)
    parser.add_argument("--object-z-offset", type=float, default=0.0)
    parser.add_argument("--object-fixed-z", type=float, default=DEFAULT_APPLE_FIXED_Z_M)
    parser.add_argument("--object-fixed-roll", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[0])
    parser.add_argument("--object-fixed-pitch", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[1])
    parser.add_argument("--object-fixed-yaw", type=float, default=DEFAULT_APPLE_FIXED_RPY_DEG[2])
    parser.add_argument("--object-fixed-rpy-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--randomize-object-yaw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--object-prim-path", default=None)
    parser.add_argument("--table-prim-path", default="/World/table")
    parser.add_argument("--object-support-clearance", type=float, default=0.0)
    parser.add_argument("--edge-biased", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-margin", type=float, default=0.12)
    parser.add_argument("--edge-min-clearance", type=float, default=0.03)
    parser.add_argument("--edge-sides", nargs="+", default=None)

    parser.add_argument("--base-goal-mode", choices=("radial", "object-offset"), default="object-offset")
    parser.add_argument("--base-goal-offset-xy", type=float, nargs=2, default=DEFAULT_PICK_PLACE_BASE_GOAL_OFFSET_XY_M)
    parser.add_argument("--standoff-candidates", type=float, nargs="+", default=DEFAULT_STANDOFF_CANDIDATES_M)
    parser.add_argument("--approach-angles-deg", type=float, nargs="+", default=DEFAULT_APPROACH_ANGLES_DEG)
    parser.add_argument("--nav-map", default=None)
    parser.add_argument("--clearance-radius", type=float, default=0.20)
    parser.add_argument("--handoff-clearance-radius", type=float, default=0.20)
    parser.add_argument("--min-boundary-clearance", type=float, default=0.25)
    parser.add_argument("--max-path-heading-error", type=float, default=1.0)
    parser.add_argument("--path-heading-weight", type=float, default=1.5)
    parser.add_argument("--path-length-weight", type=float, default=0.03)
    parser.add_argument("--path-heading-lookback-points", type=int, default=5)
    parser.add_argument("--path-heading-min-segment-length", type=float, default=0.10)
    parser.add_argument("--max-sample-attempts", type=int, default=200)

    parser.add_argument("--brisk-nav", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-dwa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-nav-steps", type=int, default=3000)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-yaw-tolerance", type=float, default=0.20)
    parser.add_argument("--terminal-position-tolerance", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-tolerance", type=float, default=0.08)
    parser.add_argument("--final-goal-tolerance-margin", type=float, default=0.03)
    parser.add_argument("--final-yaw-tolerance-margin", type=float, default=0.20)
    parser.add_argument(
        "--ignore-goal-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the stable random nav-pick batch behavior: do not require final base yaw alignment.",
    )
    parser.add_argument("--yaw-align-start-distance", type=float, default=0.50)
    parser.add_argument("--yaw-align-min-wz", type=float, default=0.40)
    parser.add_argument("--yaw-align-max-wz", type=float, default=0.60)
    parser.add_argument("--yaw-settle-max-wz", type=float, default=0.25)
    parser.add_argument("--yaw-align-lateral-kp", type=float, default=0.90)
    parser.add_argument("--yaw-align-min-vy", type=float, default=0.18)
    parser.add_argument("--terminal-yaw-slowdown-max-wz", type=float, default=0.42)
    parser.add_argument("--terminal-recovery-steps", type=int, default=90)
    parser.add_argument("--terminal-recovery-yaw-max-wz", type=float, default=0.32)
    parser.add_argument("--terminal-yaw-polish-vx", type=float, default=0.08)
    parser.add_argument("--terminal-yaw-polish-min-wz", type=float, default=0.45)
    parser.add_argument("--terminal-yaw-polish-max-wz", type=float, default=0.55)
    parser.add_argument("--base-stable-linear-tolerance", type=float, default=0.06)
    parser.add_argument("--base-stable-angular-tolerance", type=float, default=0.20)

    parser.add_argument("--side-retreat-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-retreat-success", action="store_true")
    parser.add_argument("--legacy-side-retreat", action="store_true")
    parser.add_argument("--side-grasp-fallback-retreat", action="store_true")
    parser.add_argument(
        "--fail-on-object-reset-drift",
        action="store_true",
        help="Fail pick handoff if the object drifts after target-pose apply and velocity reset.",
    )
    parser.add_argument("--use-planner-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-start-planner-server", action="store_true")
    parser.add_argument("--restart-planner-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planner-server-log", default="/tmp/go2_x5_curobo_planner_server.log")
    parser.add_argument("--planner-server-start-timeout-s", type=float, default=180.0)

    parser.add_argument("--place-only", action="store_true", help="Skip nav/pick stages and run place from existing task/nav result paths.")
    parser.add_argument("--place-xy-tolerance", type=float, default=None)
    parser.add_argument("--place-z-tolerance", type=float, default=None)
    parser.add_argument("--place-settle-steps", type=int, default=120)
    parser.add_argument(
        "--mvp-reconstruct-place",
        action="store_true",
        help="Legacy/MVP place mode: teleport the object to place_pose_world after nav_to_place.",
    )
    parser.add_argument(
        "--manipulation-backend",
        choices=("legacy-multiprocess", "single-stage-07"),
        default="single-stage-07",
    )
    parser.add_argument("--single-stage-runner", default="scripts/isaac/07_run_pick_put_demo_from_nav_results.py")
    parser.add_argument("--single-stage-put-mode", choices=("arm-place", "mvp-reconstruct"), default="arm-place")
    parser.add_argument(
        "--single-stage-carry-mode",
        choices=("none", "logical", "fixed-joint", "kinematic"),
        default="logical",
    )
    parser.add_argument("--single-stage-result-name", default="single_stage_result.json")
    parser.add_argument("--single-stage-pick-report-name", default="single_stage_pick_handoff_report.json")
    parser.add_argument("--single-stage-put-result-name", default="single_stage_put_result.json")
    parser.add_argument(
        "--single-stage-stable-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inject stable single-stage-07 env defaults for carry replay and arm-place. "
            "Enabled by default to avoid stale GO2_X5_* debug env causing foot twist or arm jitter."
        ),
    )
    parser.add_argument("--single-stage-replay-nav", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single-stage-replay-nav-real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single-stage-replay-nav-speed", type=float, default=1.0)
    parser.add_argument(
        "--skip-nav-place-for-local-arm-place",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "当使用 single-stage-07 + arm-place 时，是否跳过 nav_to_place。"
            "默认不跳过：nav_to_place 仍 headless 预计算，是否在 07 中恢复到 place base "
            "由 --restore-nav-place-for-arm-place 单独控制。"
        ),
    )
    parser.add_argument(
        "--restore-nav-place-for-arm-place",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "arm-place 阶段是否恢复到 nav_to_place 的 base 位姿。"
            "默认关闭。关闭后，07 会在 pick 后的当前 base 位姿附近做局部放置，"
            "不会把四足恢复/瞬移到 place base，也不会启用携带物体的 base restore。"
        ),
    )

    parser.add_argument("--terrain-prim-path", default="/World/scene_collision")
    return parser.parse_args()


def _nav_clearance_radius(args: argparse.Namespace) -> float:
    """Return the navigation clearance paired with generated pick base goals."""

    clearance = float(args.clearance_radius)
    if str(args.base_goal_mode).replace("-", "_") == "object_offset":
        return min(clearance, float(args.handoff_clearance_radius))
    return clearance


def _pipeline_command(
    args: argparse.Namespace,
    *,
    task_json: Path,
    dataset_dir: Path,
    nav_result: Path,
    handoff_report: Path,
    nav_only: bool = False,
    grasp_only: bool = False,
) -> list[str]:
    nav_clearance_radius = _nav_clearance_radius(args)
    single_stage_nav_only = bool(
        nav_only and getattr(args, "manipulation_backend", "") == "single-stage-07"
    )
    command = [
        _command_path(args.pipeline_python),
        str(PROJECT_ROOT / "scripts/pipeline/run_nav_then_pick.py"),
        "--task-json",
        str(task_json),
        "--task",
        args.task,
        "--checkpoint",
        _project_path_or_uri(args.checkpoint),
        "--isaaclab-launcher",
        _command_path(args.isaaclab_launcher),
        "--isaac-python",
        _command_path(args.isaac_python),
        "--dataset-dir",
        str(dataset_dir),
        "--nav-result",
        str(nav_result),
        "--handoff-report",
        str(handoff_report),
        "--handoff-clearance-radius",
        str(args.handoff_clearance_radius),
        "--inflate-radius",
        str(nav_clearance_radius),
        "--local-clearance-radius",
        str(nav_clearance_radius),
        "--max-nav-steps",
        str(args.max_nav_steps),
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
        "--yaw-align-start-distance",
        str(args.yaw_align_start_distance),
        "--yaw-align-vx",
        "0.35",
        "--yaw-align-max-vx",
        "0.6",
        "--yaw-align-position-kp",
        "0.8",
        "--yaw-align-max-vy",
        "0.35",
        "--yaw-align-min-wz",
        str(args.yaw_align_min_wz),
        "--yaw-align-max-wz",
        str(args.yaw_align_max_wz),
        "--yaw-align-lateral-kp",
        str(args.yaw_align_lateral_kp),
        "--yaw-align-lateral-deadband",
        "0.015",
        "--yaw-align-min-vy",
        str(args.yaw_align_min_vy),
        "--terminal-yaw-slowdown-max-wz",
        str(args.terminal_yaw_slowdown_max_wz),
        "--terminal-recovery-steps",
        str(args.terminal_recovery_steps),
        "--terminal-recovery-yaw-max-wz",
        str(args.terminal_recovery_yaw_max_wz),
        "--terminal-yaw-polish-vx",
        str(args.terminal_yaw_polish_vx),
        "--terminal-yaw-polish-min-wz",
        str(args.terminal_yaw_polish_min_wz),
        "--terminal-yaw-polish-max-wz",
        str(args.terminal_yaw_polish_max_wz),
        "--base-stable-linear-tolerance",
        str(args.base_stable_linear_tolerance),
        "--base-stable-angular-tolerance",
        str(args.base_stable_angular_tolerance),
        "--settle-steps",
        "120",
        "--yaw-settle-stable-steps",
        "15",
        "--yaw-settle-max-wz",
        str(args.yaw_settle_max_wz),
        "--save-replay-trajectory",
    ]
    if args.nav_map:
        command.extend(["--nav-map", args.nav_map])
    if args.nav_headless or (args.precompute_nav_first and nav_only) or single_stage_nav_only:
        command.append("--nav-headless")
    if args.brisk_nav:
        command.append("--brisk-nav")
    if args.fast_dwa:
        command.append("--fast-dwa")
    if nav_only:
        command.append("--nav-only")
    if grasp_only:
        command.append("--grasp-only")
    if args.use_planner_server:
        command.append("--use-planner-server")
    if args.auto_start_planner_server:
        command.append("--auto-start-planner-server")
    if args.restart_planner_server:
        command.append("--restart-planner-server")
    command.extend(["--planner-server-log", args.planner_server_log])
    command.extend(["--planner-server-start-timeout-s", str(args.planner_server_start_timeout_s)])
    if args.replay_nav_before_grasp and grasp_only:
        command.append("--replay-nav-before-grasp")
        if args.replay_nav_real_time:
            command.append("--replay-nav-real-time")
        command.extend(["--replay-nav-speed", str(args.replay_nav_speed)])
    if not single_stage_nav_only:
        if args.demo_visuals:
            command.append("--demo-visuals")
        command.extend(["--follow-camera-mode", args.follow_camera_mode])
        command.extend(["--viewport-camera-prim", args.viewport_camera_prim])
        if args.keep_window_open is not None:
            command.append("--keep-window-open" if args.keep_window_open else "--no-keep-window-open")
        if args.show_grasp_trajectory:
            command.append("--show-grasp-trajectory")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_retreat_only:
        command.append("--side-retreat-only")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if getattr(args, "fail_on_object_reset_drift", False):
        command.append("--fail-on-object-reset-drift")
    return command


def _place_command(
    args: argparse.Namespace,
    *,
    task_json: Path,
    dataset_dir: Path,
    nav_result: Path,
    place_result: Path,
    handoff_report: Path,
) -> list[str]:
    raw_task = json.loads(task_json.read_text(encoding="utf-8"))
    scene_usd = _project_path(raw_task["scene_usd"])
    command = [
        _command_path(args.isaac_python),
        str(PROJECT_ROOT / "scripts/isaac/run_place_from_nav_result_standalone.py"),
        "--task-json",
        str(task_json),
        "--scene-usd",
        str(scene_usd),
        "--nav-result",
        str(nav_result),
        "--place-result",
        str(place_result),
        "--handoff-report",
        str(handoff_report),
        "--dataset-dir",
        str(dataset_dir),
        "--terrain-prim-path",
        args.terrain_prim_path,
        "--handoff-clearance-radius",
        str(args.handoff_clearance_radius),
        "--settle-steps",
        str(args.place_settle_steps),
    ]
    if args.place_xy_tolerance is not None:
        command.extend(["--place-xy-tolerance", str(args.place_xy_tolerance)])
    if args.place_z_tolerance is not None:
        command.extend(["--place-z-tolerance", str(args.place_z_tolerance)])
    if args.mvp_reconstruct_place:
        command.append("--mvp-reconstruct-place")
    if args.demo_visuals:
        command.append("--demo-visuals")
        command.extend(["--viewport-camera-prim", args.viewport_camera_prim])
    if args.keep_window_open:
        command.append("--keep-window-open")
    if not args.demo_visuals:
        command.append("--headless")
    return command


def _single_stage_07_command(
    args: argparse.Namespace,
    *,
    task_nav_to_pick: Path,
    nav_pick_result: Path,
    task_nav_to_place: Path,
    nav_place_result: Path | None,
    episode_dir: Path,
) -> list[str]:
    single_stage_dir = episode_dir / "single_stage_07"
    command = [
        _command_path(args.isaac_python),
        _command_path(args.single_stage_runner),
        "--task-nav-to-pick",
        str(task_nav_to_pick),
        "--nav-pick-result",
        str(nav_pick_result),
        "--task-nav-to-place",
        str(task_nav_to_place),
        "--dataset-dir",
        str(single_stage_dir),
        "--pick-handoff-report",
        str(single_stage_dir / args.single_stage_pick_report_name),
        "--put-result",
        str(single_stage_dir / args.single_stage_put_result_name),
        "--result-json",
        str(single_stage_dir / args.single_stage_result_name),
        "--carry-mode",
        args.single_stage_carry_mode,
        "--put-mode",
        args.single_stage_put_mode,
        "--terrain-prim-path",
        args.terrain_prim_path,
        "--handoff-clearance-radius",
        str(args.handoff_clearance_radius),
        "--settle-steps",
        str(args.place_settle_steps),
    ]
    if nav_place_result is not None:
        command.extend(["--nav-place-result", str(nav_place_result)])
    if args.demo_visuals:
        command.append("--demo-visuals")
        command.extend(["--viewport-camera-prim", args.viewport_camera_prim])
    if args.single_stage_replay_nav:
        command.append("--replay-nav-to-pick")
        command.append("--replay-nav-to-place")
        command.extend(["--replay-nav-speed", str(args.single_stage_replay_nav_speed)])
        if args.single_stage_replay_nav_real_time:
            command.append("--replay-nav-real-time")
    if args.keep_window_open is not None:
        command.append("--keep-window-open" if args.keep_window_open else "--no-keep-window-open")
    if args.use_planner_server:
        command.append("--use-planner-server")
    if args.side_retreat_only:
        command.append("--side-retreat-only")
    if args.allow_retreat_success:
        command.append("--allow-retreat-success")
    if args.legacy_side_retreat:
        command.append("--legacy-side-retreat")
    if args.side_grasp_fallback_retreat:
        command.append("--side-grasp-fallback-retreat")
    if args.show_grasp_trajectory:
        command.append("--show-grasp-trajectory")
    if args.show_randomization_debug:
        command.append("--show-randomization-debug")

    if args.single_stage_put_mode == "arm-place":
        if args.restore_nav_place_for_arm_place:
            command.append("--restore-nav-place-for-arm-place")
            command.append("--no-replay-nav-place-with-carried-object")
        else:
            command.append("--no-restore-nav-place-for-arm-place")
            if args.single_stage_replay_nav:
                # stable carry replay v1.2：
                # 不瞬移恢复 place base，而是在 07 单进程里 replay nav_to_place，
                # 并在 replay 过程中让 object 以 TCP-relative 方式跟随夹爪。
                command.append("--replay-nav-place-with-carried-object")
            else:
                command.append("--no-replay-nav-place-with-carried-object")
    return command


def _single_stage_07_env(args: argparse.Namespace) -> dict[str, str] | None:
    """Return env overrides that keep the 07 carry/place path in the stable profile."""
    if not getattr(args, "single_stage_stable_defaults", True):
        return None
    raw_video_baseline = os.environ.get("GO2_X5_VIDEO_BASELINE_MODE")
    if raw_video_baseline is None:
        video_baseline = bool(getattr(args, "demo_visuals", False))
        video_baseline_source = "--demo-visuals"
    else:
        video_baseline = str(raw_video_baseline).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        video_baseline_source = "GO2_X5_VIDEO_BASELINE_MODE"
    if video_baseline:
        video_defaults = {
            # Video baseline: both nav replay segments should show visible quadruped gait.
            # Keep explicit user GO2_X5_* values when present.
            "GO2_X5_VIDEO_BASELINE_MODE": "1",
            "GO2_X5_VIDEO_BASELINE_SOURCE": video_baseline_source,
            "GO2_X5_CARRY_REPLAY_BACKEND": "live_planar_articulation",
            "GO2_X5_CARRY_VISUAL_ROOT_XFORM_SYNC": "0",
            "GO2_X5_CARRY_REPLAY_FULL_ROOT_POSE": "1",
            "GO2_X5_CARRY_REPLAY_ROOT_VELOCITY": "1",
            "GO2_X5_CARRY_REPLAY_JOINTS": "1",
            "GO2_X5_CARRY_REPLAY_LEG_ONLY": "0",
            "GO2_X5_CARRY_REPLAY_JOINT_ACTION": "1",
            "GO2_X5_CARRY_REPLAY_JOINT_VELOCITY": "0",
            "GO2_X5_CARRY_REPLAY_RENDER_WORLD_STEP": "1",
            "GO2_X5_CARRY_ZERO_ROOT_VELOCITY_WHEN_SKIPPED": "0",
            "GO2_X5_CARRY_TCP_SOURCE": "root_delta",
            "GO2_X5_RETURN_HOME_AFTER_LIFT": "0",
            "GO2_X5_HOLD_GRIPPER_AFTER_CLOSE": "1",
            "GO2_X5_CARRY_HOLD_GRIPPER_CLOSE_TARGET": "0",
            "GO2_X5_REQUIRE_SIDE_RETREAT_HEIGHT_OK": "0",
            "GO2_X5_VIDEO_BASELINE_AFTER_PICK_HOLD_S": "0.0",
            "GO2_X5_VIDEO_BASELINE_LEG_POLICY": "frozen_safe_pose",
            "GO2_X5_VIDEO_BASELINE_HOLD_AFTER_PHASE_S": "0.50",
            "GO2_X5_CARRY_OBJECT_POSE_SOURCE": "usd_root",
            "GO2_X5_CARRY_OBJECT_ORIENTATION_MODE": "world_locked",
            "GO2_X5_CARRY_OBJECT_TRANSLATION_ONLY": "1",
            "GO2_X5_CARRY_RESET_OBJECT_XFORM_STACK": "0",
            "GO2_X5_CARRY_COMPENSATE_PRESERVED_XFORM_STACK": "1",
            "GO2_X5_CARRY_REAPPLY_OBJECT_POSE_BEFORE_REPLAY": "0",
            "GO2_X5_VIDEO_BASELINE_SAFE_LEG_SETTLE_SKIP_ERROR_TOL": "0.01",
            "GO2_X5_VIDEO_HOLD_ARM_GRIPPER_MODE": "action",
            "GO2_X5_ARM_PLACE_PRE_SETTLE_S": "2.50",
            "GO2_X5_ARM_PLACE_PRE_SETTLE_MAX_S": "4.00",
            "GO2_X5_ARM_PLACE_PRE_SETTLE_JOINT_ERROR_TOL": "0.03",
            "GO2_X5_ARM_PLACE_HOLD_SUPPORT_JOINTS": "1",
            "GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_BEFORE_WORLD_STEP": "0",
            "GO2_X5_ARM_PLACE_PLANNING_HOLD_ARM_GRIPPER": "1",
            "GO2_X5_ARM_PLACE_PLANNING_HOLD_MODE": "action",
            "GO2_X5_ARM_PLACE_CALLBACK_SAFE_LEG_DIRECT_SET": "0",
            "GO2_X5_ARM_PLACE_DIRECT_JOINT_STATE": "0",
            "GO2_X5_ARM_PLACE_DIRECT_STATE_ONLY": "0",
            "GO2_X5_ARM_PLACE_LOCK_BASE": "1",
            "GO2_X5_ARM_PLACE_SKIP_WORLD_STEP": "0",
            "GO2_X5_ARM_PLACE_OBJECT_FOLLOW_MODE": "tcp_kinematic_clamp",
            "GO2_X5_ARM_PLACE_REQUIRE_OBJECT_NEAR_RELEASE_BEFORE_OPEN": "1",
            "GO2_X5_ARM_PLACE_SETTLE_TO_START_DURATION": "0.50",
            "GO2_X5_ARM_PLACE_SETTLE_TO_START_SKIP_ERROR_TOL": "0.005",
            "GO2_X5_ARM_PLACE_EXEC_TIME_SCALE": "1.00",
            "GO2_X5_ARM_PLACE_MOVE_TO_PRE_PLACE_TIME_SCALE": "1.00",
            "GO2_X5_ARM_PLACE_APPROACH_TO_PLACE_TIME_SCALE": "1.00",
            "GO2_X5_ARM_PLACE_RETREAT_PLACE_TIME_SCALE": "1.00",
            "GO2_X5_ARM_PLACE_DISABLE_KINEMATIC_OBJECT_COLLISION": "1",
            "GO2_X5_ARM_PLACE_RETURN_HOME_AFTER_RELEASE": "1",
            "GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_DURING_RETURN_HOME": "0",
            "GO2_X5_ARM_PLACE_COMMAND_DT": "0.02",
            "GO2_X5_ARM_PRE_PLACE_CLEARANCE_M": "0.06",
            "GO2_X5_ARM_PLACE_RELEASE_CLEARANCE_M": "0.005",
            "GO2_X5_ARM_PLACE_RETREAT_CLEARANCE_M": "0.08",
            "GO2_X5_WORLD_COLLISION_ACTIVATION_DISTANCE_M": "0.07",
            "GO2_X5_SINGLE_STAGE_STABLE_DEFAULTS": "1",
        }
        return {key: value for key, value in video_defaults.items() if key not in os.environ}

    return {
        # Carry replay: move only the visual root and TCP-clamped object by default.
        # Do not replay leg joints/root velocities; those debug modes easily twist the feet.
        "GO2_X5_CARRY_REPLAY_BACKEND": "visual_root_only",
        "GO2_X5_CARRY_VISUAL_ROOT_XFORM_SYNC": "1",
        "GO2_X5_CARRY_REPLAY_FULL_ROOT_POSE": "0",
        "GO2_X5_CARRY_REPLAY_ROOT_VELOCITY": "0",
        "GO2_X5_CARRY_REPLAY_JOINTS": "0",
        "GO2_X5_CARRY_REPLAY_JOINT_ACTION": "0",
        "GO2_X5_CARRY_REPLAY_JOINT_VELOCITY": "0",
        "GO2_X5_CARRY_REPLAY_RENDER_WORLD_STEP": "0",
        "GO2_X5_CARRY_ZERO_ROOT_VELOCITY_WHEN_SKIPPED": "0",
        "GO2_X5_RETURN_HOME_AFTER_LIFT": "0",
        "GO2_X5_HOLD_GRIPPER_AFTER_CLOSE": "1",
        "GO2_X5_CARRY_HOLD_GRIPPER_CLOSE_TARGET": "0",
        "GO2_X5_REQUIRE_SIDE_RETREAT_HEIGHT_OK": "0",
        "GO2_X5_CARRY_OBJECT_POSE_SOURCE": "usd_root",
        "GO2_X5_CARRY_OBJECT_ORIENTATION_MODE": "world_locked",
        "GO2_X5_CARRY_OBJECT_TRANSLATION_ONLY": "1",
        "GO2_X5_CARRY_RESET_OBJECT_XFORM_STACK": "0",
        "GO2_X5_CARRY_COMPENSATE_PRESERVED_XFORM_STACK": "1",
        "GO2_X5_CARRY_REAPPLY_OBJECT_POSE_BEFORE_REPLAY": "0",
        # Arm place: stable arm replay while leaving support/leg joints alone.
        "GO2_X5_ARM_PLACE_HOLD_SUPPORT_JOINTS": "0",
        "GO2_X5_ARM_PLACE_CALLBACK_SAFE_LEG_DIRECT_SET": "0",
        "GO2_X5_ARM_PLACE_DIRECT_JOINT_STATE": "1",
        "GO2_X5_ARM_PLACE_DIRECT_STATE_ONLY": "1",
        "GO2_X5_ARM_PLACE_LOCK_BASE": "1",
        "GO2_X5_ARM_PLACE_SKIP_WORLD_STEP": "1",
        "GO2_X5_ARM_PLACE_SETTLE_TO_START_DURATION": "0.50",
        "GO2_X5_ARM_PLACE_SETTLE_TO_START_SKIP_ERROR_TOL": "0.005",
        "GO2_X5_ARM_PLACE_EXEC_TIME_SCALE": "1.50",
        "GO2_X5_ARM_PLACE_COMMAND_DT": "0.02",
        # Common place target tuning that was previously supplied on the shell command.
        "GO2_X5_ARM_PRE_PLACE_CLEARANCE_M": "0.08",
        "GO2_X5_ARM_PLACE_RELEASE_CLEARANCE_M": "0.02",
        "GO2_X5_ARM_PLACE_RETREAT_CLEARANCE_M": "0.10",
        "GO2_X5_WORLD_COLLISION_ACTIVATION_DISTANCE_M": "0.07",
        "GO2_X5_SINGLE_STAGE_STABLE_DEFAULTS": "1",
    }


def _stage_excerpt(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    keys = (
        "success",
        "failure_reason",
        "failure_detail",
        "put_mode",
        "arm_place_executed",
        "object_teleported",
        "physical_put_execution",
        "physical_place_continuity",
        "physical_nav_continuity",
        "base_transfer_mode",
        "object_freeze_before_base_restore",
        "object_pose_synced_during_base_restore",
        "object_carried_relative_to",
        "object_dropped_during_base_restore",
        "carried_object_clamped_after_base_restore",
        "object_dynamic_restored_before_arm_place",
        "object_kinematic_until_release",
        "object_dynamic_restored_before_release",
        "physical_carry_nav",
        "video_baseline_mode",
        "video_baseline_leg_policy",
        "gait_replay_executed",
        "joint_replay_enabled",
        "joint_replay_applied_frame_count",
        "root_pose_applied_frame_count",
        "visual_replay_joint_replay_enabled",
        "visual_replay_joint_action_enabled",
        "visual_replay_leg_only_joint_replay",
        "visual_replay_joint_replay_applied_frame_count",
        "visual_replay_root_pose_applied_frame_count",
        "carry_tcp_source",
        "carry_tcp_source_prefers_root_delta",
        "stable_carry_implemented",
        "fixed_joint_carry_implemented",
        "scene_usd",
        "elapsed_wall_time_s",
    )
    return {key: payload[key] for key in keys if key in payload}


def _single_stage_result_success(args: argparse.Namespace, returncode: int, result: dict[str, Any] | None) -> tuple[bool, str]:
    if returncode != 0:
        return False, str((result or {}).get("failure_reason") or f"returncode_{returncode}")
    if not result or not result.get("success", False):
        return False, str((result or {}).get("failure_reason") or "single_stage_result_unsuccessful")
    put_mode = str(result.get("put_mode") or "")
    expected_put_mode = str(args.single_stage_put_mode)
    if put_mode.replace("_", "-") != expected_put_mode:
        return False, f"unexpected_put_mode_{put_mode or 'missing'}"
    if expected_put_mode != "arm-place":
        return True, ""
    object_teleported = bool(result.get("object_teleported", True))
    physical_put_execution = bool(result.get("physical_put_execution", False))
    if object_teleported and not physical_put_execution:
        return False, "single_stage_object_teleported_without_physical_put_execution"
    return True, ""


def _single_stage_stage_summary(
    args: argparse.Namespace,
    *,
    returncode: int,
    result_json: Path,
    pick_handoff_report: Path,
    put_result: Path,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    success, failure_reason = _single_stage_result_success(args, returncode, result)
    result_stages = (result or {}).get("stages") if isinstance(result, dict) else {}
    if not isinstance(result_stages, dict):
        result_stages = {}
    summary = {
        "success": success,
        "returncode": returncode,
        "backend": "single-stage-07",
        "result_json": str(result_json),
        "pick_handoff_report": str(pick_handoff_report),
        "put_result": str(put_result),
        "put_mode": args.single_stage_put_mode,
        "carry_mode": args.single_stage_carry_mode,
        "physical_nav_continuity": False,
        "physical_put_execution": bool((result or {}).get("physical_put_execution", False)),
        "object_teleported": bool((result or {}).get("object_teleported", True)),
        "failure_reason": failure_reason,
    }
    if result is not None:
        summary.update(
            {
                "result_failure_reason": str(result.get("failure_reason", "")),
                "result_failure_detail": str(result.get("failure_detail", "")),
                "result_put_mode": result.get("put_mode"),
                "arm_place_executed": result.get("arm_place_executed"),
                "scene_usd": result.get("scene_usd"),
                "prepare_object": _stage_excerpt(result_stages.get("prepare_object")),
                "pick": _stage_excerpt(result_stages.get("pick")),
                "restore_place_base": _stage_excerpt(result_stages.get("restore_place_base")),
                "put": _stage_excerpt(result_stages.get("put")),
                "base_transfer_mode": result.get("base_transfer_mode")
                or (result_stages.get("restore_place_base") or {}).get("base_transfer_mode"),
                "object_dropped_during_base_restore": result.get("object_dropped_during_base_restore")
                if "object_dropped_during_base_restore" in result
                else (result_stages.get("restore_place_base") or {}).get("object_dropped_during_base_restore"),
                "object_kinematic_until_release": result.get("object_kinematic_until_release")
                if "object_kinematic_until_release" in result
                else (result_stages.get("restore_place_base") or {}).get("object_kinematic_until_release"),
                "object_dynamic_restored_before_release": result.get("object_dynamic_restored_before_release"),
            }
        )
    return summary


def _base_episode_summary(episode_index: int, seed: int, episode_dir: Path, task_json: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episode_index": episode_index,
        "seed": seed,
        "success": False,
        "failure_reason": "",
        "handoff_mode": HANDOFF_MODE,
        "physical_continuity": PHYSICAL_CONTINUITY,
        "physical_nav_continuity": PHYSICAL_CONTINUITY,
        "nav_execution_mode": "headless_precomputed",
        "task_json": str(task_json),
        "episode_dir": str(episode_dir),
        "stages": {
            "nav_to_pick": {"success": False},
            "pick": {"success": False},
            "nav_to_place": {"success": False},
            "place": {"success": False},
            "single_stage_manipulation": {"success": False},
        },
        "object_pose_policy": {},
        "notes": [
            "This is a multi-process JSON handoff pipeline.",
            "Object physical state is not continuous between stages.",
            "legacy-multiprocess place reports not_implemented unless --mvp-reconstruct-place is passed.",
            "Pass --mvp-reconstruct-place only for the legacy reconstructed-object place smoke test.",
            "single-stage-07 uses precomputed nav results and runs pick+arm-place in one Isaac Sim stage.",
        ],
    }


def _finalize_failure(summary: dict[str, Any], reason: str, detail: str | None = None) -> None:
    summary["success"] = False
    summary["failure_reason"] = reason
    if detail:
        summary["failure_detail"] = detail


def _stage_success(returncode: int, payload: dict[str, Any] | None) -> tuple[bool, str]:
    success = bool(payload and payload.get("success", False)) and returncode == 0
    reason = "" if success else str((payload or {}).get("failure_reason") or f"returncode_{returncode}")
    return success, reason


def _handoff_success(returncode: int, payload: dict[str, Any] | None) -> tuple[bool, str]:
    success = bool(payload and payload.get("success", False)) and returncode == 0
    reason = "" if success else str((payload or {}).get("failure_reason") or f"returncode_{returncode}")
    return success, reason


def _make_nav_to_place_task(task_original: dict[str, Any], nav_pick_result: dict[str, Any]) -> dict[str, Any]:
    place = dict(task_original.get("place") or {})
    if not place.get("enabled", False) or not place.get("base_goal"):
        raise ValueError("missing_place_goal")
    final_pose = dict(nav_pick_result.get("final_base_pose_world") or {})
    if not {"x", "y", "yaw"}.issubset(final_pose):
        raise ValueError("missing_nav_pick_final_base_pose")

    task_nav_to_place = copy.deepcopy(task_original)
    task_nav_to_place["instruction"] = "navigate to place base goal"
    task_nav_to_place["start"] = {
        "x": float(final_pose["x"]),
        "y": float(final_pose["y"]),
        "yaw": float(final_pose["yaw"]),
    }
    pick = dict(task_nav_to_place.get("pick") or {})
    pick["base_goal"] = copy.deepcopy(place["base_goal"])
    task_nav_to_place["pick"] = pick
    task_nav_to_place["handoff"] = {
        "source": "nav_pick_result",
        "physical_continuity": PHYSICAL_CONTINUITY,
        "handoff_mode": HANDOFF_MODE,
    }
    return task_nav_to_place


def _has_place_goal(task: dict[str, Any]) -> bool:
    place = task.get("place")
    if not isinstance(place, dict):
        return False
    return bool(place.get("enabled", False) and place.get("base_goal") and place.get("place_pose_world"))


def _load_place_template(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    template_path = _project_path(args.place_template_task)
    template = _read_json_if_exists(template_path)
    if template is None:
        raise FileNotFoundError(f"place template task not found: {template_path}")
    if not _has_place_goal(template):
        raise ValueError(f"place template task has no enabled place goal: {template_path}")
    return copy.deepcopy(template["place"]), template_path


def _ensure_place_goal(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if _has_place_goal(task):
        return task
    place, template_path = _load_place_template(args)
    task["place"] = place
    randomization = dict(task.get("randomization") or {})
    task["randomization"] = randomization
    randomization["place_template"] = {
        "enabled": True,
        "source_task": str(template_path),
        "reason": "base_task_missing_enabled_place_goal",
    }
    return task


def _validated_range_pair(raw_range: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(raw_range) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    lower, upper = (float(raw_range[0]), float(raw_range[1]))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} values must be finite.")
    if lower > upper:
        raise ValueError(f"{name} min must be <= max.")
    return lower, upper


def _apply_default_place_xy_randomization(
    task: dict[str, Any],
    *,
    episode_seed: int,
    place_x_range: Sequence[float] = DEFAULT_PLACE_X_RANGE_M,
    place_y_range: Sequence[float] = DEFAULT_PLACE_Y_RANGE_M,
) -> dict[str, Any]:
    """Sample place x/y inside bounds while preserving the template base-goal offset."""

    place = dict(task.get("place") or {})
    place_pose = dict(place.get("place_pose_world") or {})
    base_goal = dict(place.get("base_goal") or {})
    if not place.get("enabled", False) or not {"x", "y"}.issubset(place_pose):
        return task

    x_min, x_max = _validated_range_pair(place_x_range, name="place_x_range")
    y_min, y_max = _validated_range_pair(place_y_range, name="place_y_range")
    rng = random.Random(int(episode_seed) + 9173)
    sampled_x = rng.uniform(x_min, x_max)
    sampled_y = rng.uniform(y_min, y_max)

    place_pose_before = copy.deepcopy(place_pose)
    base_goal_before = copy.deepcopy(base_goal)
    dx = sampled_x - float(place_pose_before["x"])
    dy = sampled_y - float(place_pose_before["y"])
    place_pose["x"] = sampled_x
    place_pose["y"] = sampled_y
    place["place_pose_world"] = place_pose

    base_goal_offset_before: list[float] | None = None
    if {"x", "y"}.issubset(base_goal):
        base_goal_offset_before = [
            float(base_goal["x"]) - float(place_pose_before["x"]),
            float(base_goal["y"]) - float(place_pose_before["y"]),
        ]
        base_goal["x"] = float(base_goal["x"]) + dx
        base_goal["y"] = float(base_goal["y"]) + dy
        place["base_goal"] = base_goal

    task["place"] = place
    randomization = dict(task.get("randomization") or {})
    randomization["place_xy_randomization"] = {
        "enabled": True,
        "mode": "sample_xy_within_cli_range_translate_base_goal",
        "seed": int(episode_seed) + 9173,
        "x_range_m": [x_min, x_max],
        "y_range_m": [y_min, y_max],
        "sampled_xy": {"x": sampled_x, "y": sampled_y},
        "delta_xy_m": [dx, dy],
        "place_pose_world_before": place_pose_before,
        "place_pose_world_after": copy.deepcopy(place_pose),
        "base_goal_before": base_goal_before,
        "base_goal_after": copy.deepcopy(base_goal) if base_goal else None,
        "base_goal_offset_from_place_xy_m_before": base_goal_offset_before,
        "base_goal_offset_from_place_xy_m_after": [
            float(base_goal["x"]) - float(place_pose["x"]),
            float(base_goal["y"]) - float(place_pose["y"]),
        ]
        if {"x", "y"}.issubset(base_goal)
        else None,
        "nav_goal_rule": "place.base_goal keeps the original template offset from randomized place.place_pose_world",
        "pose_policy": "xy_only; z/roll/pitch/yaw unchanged",
        "object_mesh_randomization": {"enabled": False, "reason": "no mesh or asset swap in batch randomization"},
    }
    task["randomization"] = randomization
    return task


def _wrap_yaw(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def _nav_pick_alignment_report(task: dict[str, Any], nav_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the final nav pose is close enough for arm planning."""

    pick = dict(task.get("pick") or {})
    object_pose = dict(pick.get("object_pose_world") or {})
    base_goal = dict(pick.get("base_goal") or {})
    final_pose = dict(nav_result.get("final_base_pose_world") or {})
    report: dict[str, Any] = {
        "available": False,
        "warning": "",
        "final_goal_distance_m": nav_result.get("final_goal_distance"),
        "yaw_error_rad": nav_result.get("yaw_error"),
        "final_position_reached": nav_result.get("final_position_reached"),
        "final_yaw_aligned": nav_result.get("final_yaw_aligned"),
        "base_stable": nav_result.get("base_stable"),
    }
    if not {"x", "y"}.issubset(final_pose) or not {"x", "y"}.issubset(object_pose):
        report["warning"] = "missing_final_pose_or_object_pose"
        return report

    final_x = float(final_pose["x"])
    final_y = float(final_pose["y"])
    object_x = float(object_pose["x"])
    object_y = float(object_pose["y"])
    object_distance = math.hypot(final_x - object_x, final_y - object_y)
    report.update(
        {
            "available": True,
            "final_base_pose_xyyaw": {
                "x": final_x,
                "y": final_y,
                "yaw": float(final_pose.get("yaw", 0.0)),
            },
            "object_xy": {"x": object_x, "y": object_y},
            "final_base_to_object_distance_m": object_distance,
        }
    )
    if {"x", "y"}.issubset(base_goal):
        base_goal_error = math.hypot(final_x - float(base_goal["x"]), final_y - float(base_goal["y"]))
        report["base_goal_xy_error_m"] = base_goal_error
        if "yaw" in base_goal and "yaw" in final_pose:
            report["base_goal_yaw_error_rad"] = abs(_wrap_yaw(float(final_pose["yaw"]) - float(base_goal["yaw"])))
    if object_distance > 0.45:
        report["warning"] = "final_base_far_from_object_for_arm_planning"
    elif report.get("base_goal_xy_error_m") is not None and float(report["base_goal_xy_error_m"]) > 0.18:
        report["warning"] = "final_base_not_aligned_to_pick_base_goal"
    else:
        report["warning"] = ""
    return report


def _batch_row_from_summary(summary: dict[str, Any], *, started_at: float, task: dict[str, Any]) -> dict[str, Any]:
    randomization = task.get("randomization", {})
    selected = randomization.get("selected_base_goal_candidate", {})
    stages = summary.get("stages", {})
    nav_pick_alignment = stages.get("nav_to_pick", {}).get("alignment", {})
    single_stage = stages.get("single_stage_manipulation", {})
    return {
        "episode_index": summary["episode_index"],
        "seed": summary["seed"],
        "task_json": summary["task_json"],
        "manipulation_backend": summary.get("manipulation_backend"),
        "nav_execution_mode": summary.get("nav_execution_mode"),
        "nav_visualization_in_batch": summary.get("nav_visualization_in_batch"),
        "single_stage_07_visualization": summary.get("single_stage_07_visualization"),
        "single_stage_replay_nav": summary.get("single_stage_replay_nav"),
        "show_randomization_debug": summary.get("show_randomization_debug"),
        "object_pose_world": task.get("pick", {}).get("object_pose_world"),
        "object_pose_policy": randomization.get("object_pose_policy"),
        "object_xy_randomization": randomization.get("object_xy_randomization"),
        "object_mesh_randomization": randomization.get("object_mesh_randomization"),
        "base_goal": task.get("pick", {}).get("base_goal"),
        "place_pose_world": (task.get("place") or {}).get("place_pose_world"),
        "place_base_goal": (task.get("place") or {}).get("base_goal"),
        "place_xy_randomization": randomization.get("place_xy_randomization"),
        "selected_base_goal_candidate": selected,
        "path_heading_error": selected.get("path_heading_error"),
        "nav_pick_base_goal_xy_error_m": nav_pick_alignment.get("base_goal_xy_error_m"),
        "nav_pick_base_goal_yaw_error_rad": nav_pick_alignment.get("base_goal_yaw_error_rad"),
        "nav_pick_final_base_to_object_distance_m": nav_pick_alignment.get("final_base_to_object_distance_m"),
        "nav_pick_alignment_warning": nav_pick_alignment.get("warning"),
        "handoff_mode": HANDOFF_MODE,
        "physical_continuity": PHYSICAL_CONTINUITY,
        "nav_pick_success": bool(stages.get("nav_to_pick", {}).get("success", False)),
        "pick_success": bool(stages.get("pick", {}).get("success", False)),
        "nav_place_success": bool(stages.get("nav_to_place", {}).get("success", False)),
        "place_success": bool(stages.get("place", {}).get("success", False)),
        "single_stage_manipulation_success": bool(single_stage.get("success", False)),
        "single_stage_result_json": summary.get("single_stage_manipulation_result"),
        "single_stage_put_mode": single_stage.get("put_mode"),
        "single_stage_object_teleported": single_stage.get("object_teleported"),
        "single_stage_physical_put_execution": single_stage.get("physical_put_execution"),
        "base_transfer_mode": single_stage.get("base_transfer_mode"),
        "object_dropped_during_base_restore": single_stage.get("object_dropped_during_base_restore"),
        "object_kinematic_until_release": single_stage.get("object_kinematic_until_release"),
        "object_dynamic_restored_before_release": single_stage.get("object_dynamic_restored_before_release"),
        "success": bool(summary.get("success", False)),
        "failure_reason": str(summary.get("failure_reason", "")),
        "elapsed_wall_time_s": time.time() - started_at,
        "summary_json": str(Path(summary["episode_dir"]) / "summary.json"),
    }


def _write_episode_artifacts(summary: dict[str, Any], summary_jsonl: Path, task: dict[str, Any], started_at: float) -> None:
    episode_dir = Path(summary["episode_dir"])
    _write_json(episode_dir / "summary.json", summary)
    _write_summary_line(summary_jsonl, _batch_row_from_summary(summary, started_at=started_at, task=task))


def _random_task_generation_kwargs(
    args: argparse.Namespace,
    *,
    generated_task_json: Path,
    episode_seed: int,
    spawn_region: SpawnRegion,
    clearance_radius: float,
) -> dict[str, Any]:
    return {
        "base_task_path": _project_path(args.base_task),
        "output_path": generated_task_json,
        "seed": episode_seed,
        "nav_map_path": args.nav_map,
        "object_prim_path": args.object_prim_path,
        "table_prim_path": args.table_prim_path,
        "spawn_region": spawn_region,
        "yaw_range_deg": (0.0, 360.0),
        "object_fixed_z": args.object_fixed_z,
        "object_fixed_rpy": (args.object_fixed_roll, args.object_fixed_pitch, args.object_fixed_yaw),
        "object_fixed_rpy_unit": args.object_fixed_rpy_unit,
        "randomize_object_yaw": args.randomize_object_yaw,
        "standoff_candidates": args.standoff_candidates,
        "approach_angles_deg": args.approach_angles_deg,
        "base_goal_mode": args.base_goal_mode.replace("-", "_"),
        "base_goal_offset_xy": (float(args.base_goal_offset_xy[0]), float(args.base_goal_offset_xy[1])),
        "clearance_radius": float(clearance_radius),
        "min_boundary_clearance": args.min_boundary_clearance,
        "edge_sides": args.edge_sides,
        "edge_margin": args.edge_margin if args.edge_biased else None,
        "edge_min_clearance": args.edge_min_clearance,
        "object_support_clearance": args.object_support_clearance,
        "max_path_heading_error": args.max_path_heading_error if args.max_path_heading_error >= 0.0 else None,
        "path_heading_weight": args.path_heading_weight,
        "path_length_weight": args.path_length_weight,
        "path_heading_lookback_points": args.path_heading_lookback_points,
        "path_heading_min_segment_length": args.path_heading_min_segment_length,
        "max_sample_attempts": args.max_sample_attempts,
    }


def _write_random_pick_task_with_clearance_fallback(
    args: argparse.Namespace,
    *,
    generated_task_json: Path,
    episode_seed: int,
    spawn_region: SpawnRegion,
) -> dict[str, Any]:
    requested_clearance = float(args.clearance_radius)
    kwargs = _random_task_generation_kwargs(
        args,
        generated_task_json=generated_task_json,
        episode_seed=episode_seed,
        spawn_region=spawn_region,
        clearance_radius=requested_clearance,
    )
    try:
        return write_random_pick_task(**kwargs)
    except RandomTaskGenerationError as original_exc:
        base_goal_mode = str(kwargs["base_goal_mode"])
        fallback_clearance = min(requested_clearance, float(args.handoff_clearance_radius))
        if base_goal_mode != "object_offset" or fallback_clearance >= requested_clearance:
            raise

        print(
            "[pick-place-batch] object-offset task generation failed at "
            f"clearance_radius={requested_clearance:.3f}; retrying with "
            f"handoff_clearance_radius={fallback_clearance:.3f}"
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["clearance_radius"] = fallback_clearance
        task = write_random_pick_task(**fallback_kwargs)
        randomization = dict(task.get("randomization") or {})
        task["randomization"] = randomization
        randomization["generation_clearance_fallback"] = {
            "enabled": True,
            "reason": "object_offset_requested_clearance_failed",
            "requested_clearance_radius": requested_clearance,
            "fallback_clearance_radius": fallback_clearance,
            "original_failure": str(original_exc),
        }
        _write_json(generated_task_json, task)
        return task


def _run_episode(args: argparse.Namespace, episode_index: int, episode_seed: int, output_task_dir: Path, dataset_root: Path, summary_jsonl: Path) -> bool:
    started_at = time.time()
    seed_label = _format_seed(episode_seed)
    generated_task_json = output_task_dir / f"apple_seed_{seed_label}.json"
    episode_dir = dataset_root / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    task_original_path = episode_dir / "task_original.json"
    task_nav_to_pick_path = episode_dir / "task_nav_to_pick.json"
    task_nav_to_place_path = episode_dir / "task_nav_to_place.json"
    nav_pick_result_path = episode_dir / "nav_pick_result.json"
    pick_handoff_report_path = episode_dir / "pick_handoff_report.json"
    pick_summary_path = episode_dir / "pick_summary.json"
    nav_place_result_path = episode_dir / "nav_place_result.json"
    place_handoff_report_path = episode_dir / "place_handoff_report.json"
    place_result_path = episode_dir / "place_result.json"
    single_stage_dir = episode_dir / "single_stage_07"
    single_stage_result_path = single_stage_dir / args.single_stage_result_name
    single_stage_pick_report_path = single_stage_dir / args.single_stage_pick_report_name
    single_stage_put_result_path = single_stage_dir / args.single_stage_put_result_name

    spawn_region = SpawnRegion(
        x_min=float(args.table_x_range[0]),
        x_max=float(args.table_x_range[1]),
        y_min=float(args.table_y_range[0]),
        y_max=float(args.table_y_range[1]),
        table_z=float(args.table_z),
        object_z_offset=float(args.object_z_offset),
    )

    try:
        task = _write_random_pick_task_with_clearance_fallback(
            args,
            generated_task_json=generated_task_json,
            episode_seed=episode_seed,
            spawn_region=spawn_region,
        )
        # 确保place字段存在
        task = _ensure_place_goal(task, args)
        if not (
            args.manipulation_backend == "single-stage-07"
            and args.single_stage_put_mode == "arm-place"
            and args.skip_nav_place_for_local_arm_place
        ):
            task = _apply_default_place_xy_randomization(
                task,
                episode_seed=episode_seed,
                place_x_range=args.place_x_range,
                place_y_range=args.place_y_range,
            )
        # 是否跳过nav_to_place
        if(
            args.manipulation_backend == "single-stage-07"
            and args.single_stage_put_mode == "arm-place"
            and args.skip_nav_place_for_local_arm_place
        ):
            task = _make_local_arm_place_goal(task)
        _write_json(generated_task_json, task)
    except (RandomTaskGenerationError, ValueError, FileNotFoundError) as exc:
        summary = _base_episode_summary(episode_index, episode_seed, episode_dir, task_nav_to_pick_path)
        _finalize_failure(summary, "task_generation_failed", str(exc))
        _write_episode_artifacts(summary, summary_jsonl, {"pick": {}, "place": {}, "randomization": {}}, started_at)
        print(f"[pick-place-batch] episode={episode_index} task generation failed: {exc}")
        return False

    _write_json(task_original_path, task)
    _write_json(task_nav_to_pick_path, task)
    summary = _base_episode_summary(episode_index, episode_seed, episode_dir, task_nav_to_pick_path)
    summary["generated_task_json"] = str(generated_task_json)
    summary["task_original_json"] = str(task_original_path)
    summary["object_pose_policy"] = task.get("randomization", {}).get("object_pose_policy", {})
    summary["manipulation_backend"] = args.manipulation_backend
    summary["nav_execution_mode"] = "headless_precomputed"
    summary["physical_nav_continuity"] = PHYSICAL_CONTINUITY
    summary["nav_visualization_in_batch"] = bool(args.manipulation_backend != "single-stage-07" and args.demo_visuals)
    summary["single_stage_07_visualization"] = bool(args.manipulation_backend == "single-stage-07" and args.demo_visuals)
    summary["show_randomization_debug"] = bool(
        args.manipulation_backend == "single-stage-07"
        and args.demo_visuals
        and args.show_randomization_debug
    )
    summary["single_stage_replay_nav"] = bool(args.manipulation_backend == "single-stage-07" and args.single_stage_replay_nav)
    summary["single_stage_stable_defaults"] = bool(
        args.manipulation_backend == "single-stage-07"
        and args.single_stage_stable_defaults
    )
    if args.manipulation_backend == "single-stage-07":
        summary["single_stage_manipulation_result"] = str(single_stage_result_path)

    if args.place_only:
        summary["stages"]["nav_to_pick"].update({"success": True, "skipped": True})
        summary["stages"]["pick"].update({"success": True, "skipped": True})
    else:
        nav_pick_command = _pipeline_command(
            args,
            task_json=task_nav_to_pick_path,
            dataset_dir=episode_dir / "nav_pick",
            nav_result=nav_pick_result_path,
            handoff_report=pick_handoff_report_path,
            nav_only=True,
        )
        print(f"[pick-place-batch] episode={episode_index} seed={episode_seed} nav_to_pick")
        nav_pick_completed = subprocess.run(nav_pick_command, cwd=str(PROJECT_ROOT), check=False)
        nav_pick_result = _read_json_if_exists(nav_pick_result_path)
        nav_pick_success, nav_pick_failure = _stage_success(nav_pick_completed.returncode, nav_pick_result)
        summary["stages"]["nav_to_pick"] = {
            "success": nav_pick_success,
            "returncode": nav_pick_completed.returncode,
            "nav_result": str(nav_pick_result_path),
            "failure_reason": nav_pick_failure,
        }
        if nav_pick_result is not None:
            summary["stages"]["nav_to_pick"]["alignment"] = _nav_pick_alignment_report(task, nav_pick_result)
        if not nav_pick_success:
            _finalize_failure(summary, nav_pick_failure or "nav_to_pick_failed")
            _write_episode_artifacts(summary, summary_jsonl, task, started_at)
            print(f"[pick-place-batch] episode={episode_index} nav_to_pick failed: {summary['failure_reason']}")
            return False

        if args.manipulation_backend == "legacy-multiprocess":
            pick_command = _pipeline_command(
                args,
                task_json=task_nav_to_pick_path,
                dataset_dir=episode_dir / "pick",
                nav_result=nav_pick_result_path,
                handoff_report=pick_handoff_report_path,
                grasp_only=True,
            )
            print(f"[pick-place-batch] episode={episode_index} seed={episode_seed} pick")
            pick_completed = subprocess.run(pick_command, cwd=str(PROJECT_ROOT), check=False)
            pick_handoff = _read_json_if_exists(pick_handoff_report_path)
            pick_success, pick_failure = _handoff_success(pick_completed.returncode, pick_handoff)
            if pick_handoff is not None:
                _write_json(pick_summary_path, pick_handoff)
            summary["stages"]["pick"] = {
                "success": pick_success,
                "returncode": pick_completed.returncode,
                "handoff_report": str(pick_handoff_report_path),
                "summary": str(pick_summary_path),
                "failure_reason": pick_failure,
            }
            if not pick_success:
                _finalize_failure(summary, pick_failure or "pick_failed")
                _write_episode_artifacts(summary, summary_jsonl, task, started_at)
                print(f"[pick-place-batch] episode={episode_index} pick failed: {summary['failure_reason']}")
                return False
        else:
            summary["stages"]["pick"] = {
                "success": True,
                "skipped": True,
                "backend": "single-stage-07",
                "reason": "pick runs inside 07 single-stage manipulation",
            }

    nav_pick_result = _read_json_if_exists(nav_pick_result_path)
    if nav_pick_result is None:
        _finalize_failure(summary, "missing_nav_pick_result")
        _write_episode_artifacts(summary, summary_jsonl, task, started_at)
        return False

    try:
        task_nav_to_place = _make_nav_to_place_task(task, nav_pick_result)
    except ValueError as exc:
        _finalize_failure(summary, str(exc))
        summary["stages"]["nav_to_place"] = {"success": False, "failure_reason": str(exc)}
        _write_episode_artifacts(summary, summary_jsonl, task, started_at)
        print(f"[pick-place-batch] episode={episode_index} nav_to_place setup failed: {exc}")
        return False
    _write_json(task_nav_to_place_path, task_nav_to_place)

    skip_nav_place = bool(
        args.manipulation_backend == "single-stage-07"
        and args.skip_nav_place_for_local_arm_place
    )
    nav_place_result_for_07: Path | None = nav_place_result_path
    if skip_nav_place:
        nav_place_result_for_07 = None
        summary["stages"]["nav_to_place"] = {
            "success": True,
            "skipped": True,
            "task_json": str(task_nav_to_place_path),
            "nav_result": None,
            "reason": "single-stage-07 will use task.place.base_goal",
        }
    else:
        nav_place_command = _pipeline_command(
            args,
            task_json=task_nav_to_place_path,
            dataset_dir=episode_dir / "nav_place",
            nav_result=nav_place_result_path,
            handoff_report=place_handoff_report_path,
            nav_only=True,
        )
        print(f"[pick-place-batch] episode={episode_index} seed={episode_seed} nav_to_place")
        nav_place_completed = subprocess.run(nav_place_command, cwd=str(PROJECT_ROOT), check=False)
        nav_place_result = _read_json_if_exists(nav_place_result_path)
        nav_place_success, nav_place_failure = _stage_success(nav_place_completed.returncode, nav_place_result)
        summary["stages"]["nav_to_place"] = {
            "success": nav_place_success,
            "returncode": nav_place_completed.returncode,
            "task_json": str(task_nav_to_place_path),
            "nav_result": str(nav_place_result_path),
            "failure_reason": nav_place_failure,
        }
        if not nav_place_success:
            _finalize_failure(summary, nav_place_failure or "nav_to_place_failed")
            _write_episode_artifacts(summary, summary_jsonl, task, started_at)
            print(f"[pick-place-batch] episode={episode_index} nav_to_place failed: {summary['failure_reason']}")
            return False

    if args.manipulation_backend == "single-stage-07":
        single_stage_command = _single_stage_07_command(
            args,
            task_nav_to_pick=task_nav_to_pick_path,
            nav_pick_result=nav_pick_result_path,
            task_nav_to_place=task_nav_to_place_path,
            nav_place_result=nav_place_result_for_07,
            episode_dir=episode_dir,
        )
        single_stage_env_overrides = _single_stage_07_env(args)
        single_stage_env = None
        if single_stage_env_overrides is not None:
            single_stage_env = {**os.environ, **single_stage_env_overrides}
            print(
                "[pick-place-batch] single_stage_07 env defaults:",
                {
                    "video_baseline": single_stage_env_overrides.get("GO2_X5_VIDEO_BASELINE_MODE"),
                    "video_baseline_source": single_stage_env_overrides.get("GO2_X5_VIDEO_BASELINE_SOURCE"),
                    "carry_backend": single_stage_env_overrides.get("GO2_X5_CARRY_REPLAY_BACKEND"),
                    "carry_joint_replay": single_stage_env_overrides.get("GO2_X5_CARRY_REPLAY_JOINTS"),
                    "arm_direct": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_DIRECT_JOINT_STATE"),
                    "arm_skip_world_step": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_SKIP_WORLD_STEP"),
                    "hold_support": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_HOLD_SUPPORT_JOINTS"),
                    "hold_root_support_before_world_step": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_BEFORE_WORLD_STEP"
                    ),
                    "planning_hold_arm_gripper": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_PLANNING_HOLD_ARM_GRIPPER"
                    ),
                    "planning_hold_mode": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_PLANNING_HOLD_MODE"),
                    "video_hold_arm_gripper_mode": single_stage_env_overrides.get(
                        "GO2_X5_VIDEO_HOLD_ARM_GRIPPER_MODE"
                    ),
                    "object_follow_mode": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_OBJECT_FOLLOW_MODE"),
                    "disable_kinematic_object_collision": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_DISABLE_KINEMATIC_OBJECT_COLLISION"
                    ),
                    "exec_time_scale": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_EXEC_TIME_SCALE"),
                    "move_to_pre_place_time_scale": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_MOVE_TO_PRE_PLACE_TIME_SCALE"
                    ),
                    "approach_to_place_time_scale": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_APPROACH_TO_PLACE_TIME_SCALE"
                    ),
                    "retreat_place_time_scale": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_RETREAT_PLACE_TIME_SCALE"
                    ),
                    "return_home_after_release": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_RETURN_HOME_AFTER_RELEASE"
                    ),
                    "hold_root_support_during_return_home": single_stage_env_overrides.get(
                        "GO2_X5_ARM_PLACE_HOLD_ROOT_SUPPORT_DURING_RETURN_HOME"
                    ),
                    "release_clearance_m": single_stage_env_overrides.get("GO2_X5_ARM_PLACE_RELEASE_CLEARANCE_M"),
                },
                flush=True,
            )
        print(f"[pick-place-batch] episode={episode_index} seed={episode_seed} single_stage_07")
        single_stage_completed = subprocess.run(
            single_stage_command,
            cwd=str(PROJECT_ROOT),
            check=False,
            env=single_stage_env,
        )
        single_stage_result = _read_json_if_exists(single_stage_result_path)
        summary["stages"]["single_stage_manipulation"] = _single_stage_stage_summary(
            args,
            returncode=single_stage_completed.returncode,
            result_json=single_stage_result_path,
            pick_handoff_report=single_stage_pick_report_path,
            put_result=single_stage_put_result_path,
            result=single_stage_result,
        )
        summary["base_transfer_mode"] = summary["stages"]["single_stage_manipulation"].get("base_transfer_mode")
        summary["object_dropped_during_base_restore"] = summary["stages"]["single_stage_manipulation"].get(
            "object_dropped_during_base_restore"
        )
        summary["stages"]["place"] = {
            "success": bool(summary["stages"]["single_stage_manipulation"]["success"]),
            "skipped": True,
            "backend": "single-stage-07",
            "reason": "place runs inside 07 arm-place",
        }
        if not summary["stages"]["single_stage_manipulation"]["success"]:
            result_failure = str((single_stage_result or {}).get("failure_reason") or "")
            result_detail = str((single_stage_result or {}).get("failure_detail") or "")
            detail = result_failure
            if result_detail:
                detail = f"{detail}: {result_detail}" if detail else result_detail
            _finalize_failure(summary, "single_stage_manipulation_failed", detail or None)
            _write_episode_artifacts(summary, summary_jsonl, task, started_at)
            print(
                "[pick-place-batch] episode="
                f"{episode_index} single_stage_07 failed: {summary['failure_reason']}"
            )
            return False
    else:
        place_command = _place_command(
            args,
            task_json=task_nav_to_place_path,
            dataset_dir=episode_dir / "place",
            nav_result=nav_place_result_path,
            place_result=place_result_path,
            handoff_report=place_handoff_report_path,
        )
        print(f"[pick-place-batch] episode={episode_index} seed={episode_seed} place")
        place_completed = subprocess.run(place_command, cwd=str(PROJECT_ROOT), check=False)
        place_result = _read_json_if_exists(place_result_path)
        place_success, place_failure = _stage_success(place_completed.returncode, place_result)
        summary["stages"]["place"] = {
            "success": place_success,
            "returncode": place_completed.returncode,
            "place_result": str(place_result_path),
            "handoff_report": str(place_handoff_report_path),
            "failure_reason": place_failure,
        }
        if not place_success:
            _finalize_failure(summary, place_failure or "place_failed")
            _write_episode_artifacts(summary, summary_jsonl, task, started_at)
            print(f"[pick-place-batch] episode={episode_index} place failed: {summary['failure_reason']}")
            return False

    summary["success"] = True
    summary["failure_reason"] = ""
    summary["elapsed_wall_time_s"] = time.time() - started_at
    _write_episode_artifacts(summary, summary_jsonl, task, started_at)
    print(f"[pick-place-batch] episode={episode_index} success")
    return True

def _make_local_arm_place_goal(task: dict[str, Any]) -> dict[str, Any]:
    task = copy.deepcopy(task)
    pick = dict(task.get("pick") or {})
    place = dict(task.get("place") or {})
    object_pose = dict(pick.get("object_pose_world") or {})
    base_goal = dict(pick.get("base_goal") or {})
    if not {"x", "y", "z"}.issubset(object_pose):
        return task

    place_pose_world_before_update = copy.deepcopy(place.get("place_pose_world"))
    restored_place_pose = copy.deepcopy(object_pose)
    restored_place_pose["x"] = float(restored_place_pose["x"])
    restored_place_pose["y"] = float(restored_place_pose["y"])
    restored_place_pose["z"] = float(restored_place_pose["z"])
    restored_place_pose["roll"] = float(restored_place_pose.get("roll", 0.0))
    restored_place_pose["pitch"] = float(restored_place_pose.get("pitch", 0.0))
    restored_place_pose["yaw"] = float(restored_place_pose.get("yaw", 0.0))

    place["enabled"] = True
    place["place_pose_world"] = restored_place_pose

    place["base_goal"] = {
        "x": float(base_goal.get("x", restored_place_pose["x"])),
        "y": float(base_goal.get("y", restored_place_pose["y"])),
        "yaw": float(base_goal.get("yaw", 0.0)),
    }

    place["release_height"] = float(place.get("release_height", 0.04))
    place["retreat_height"] = float(place.get("retreat_height", 0.12))
    place["place_xy_tolerance"] = float(place.get("place_xy_tolerance", 0.10))
    place["place_z_tolerance"] = float(place.get("place_z_tolerance", 0.08))

    task["place"] = place

    # 在 randomization 里记录这次 place 目标是脚本自动生成的。
    # 这样以后看 summary/result 时能知道当前不是远处 place，而是局部 place。
    randomization = dict(task.get("randomization") or {})
    randomization["local_arm_place_goal"] = {
        "enabled": True,
        "mode": "restore_pick_pose",
        "reason": "place object back to its original pre-pick pose; no x/y offset because +0.15 may fall off table",
        "place_pose_world_before_update": place_pose_world_before_update,
        "pick_object_pose_world": copy.deepcopy(object_pose),
        "place_pose_world_after_update": copy.deepcopy(restored_place_pose),
        "place_offset_from_object_xy_m": [0.0, 0.0],
    }
    task["randomization"] = randomization

    return task

def main() -> int:
    args = _parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if args.side_retreat_only:
        args.legacy_side_retreat = True
        args.allow_retreat_success = True
    if args.ignore_goal_yaw:
        args.goal_yaw_tolerance = math.pi
        args.terminal_yaw_tolerance = math.pi
        args.final_yaw_tolerance_margin = 0.0
    output_task_dir = Path(args.output_task_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    summary_jsonl = dataset_root / "batch_summary.jsonl"
    output_task_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if summary_jsonl.exists():
        summary_jsonl.unlink()

    overall_success = True
    for episode_index in range(args.num_episodes):
        episode_seed = int(args.seed) + episode_index
        success = _run_episode(args, episode_index, episode_seed, output_task_dir, dataset_root, summary_jsonl)
        if not success:
            overall_success = False
            if not args.continue_on_failure:
                break

    print(f"[pick-place-batch] summary: {summary_jsonl}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
