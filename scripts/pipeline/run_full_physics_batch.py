#!/usr/bin/env python3
"""Batch launcher for full-physics pipeline episodes."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shlex
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ENTRY = PROJECT_ROOT / "scripts/pipeline/run_full_physics_pipeline.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.recording.training_action import (  # noqa: E402
    training_quality_success_verified,
)
from source.scene.profiles import SceneProfileError, load_scene_profile  # noqa: E402


_REAL_MODES = {
    "simulation_smoke": "--simulation-smoke",
    "navigation_smoke": "--navigation-smoke",
    "navigation_carry_smoke": "--navigation-carry-smoke",
    "stair_locomotion_smoke": "--stair-locomotion-smoke",
    "manipulation_apply_smoke": "--manipulation-apply-smoke",
}

_NAVIGATION_MODES = {
    "full_physics",
    "navigation_smoke",
    "navigation_carry_smoke",
    "stair_locomotion_smoke",
}


_COLORS = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "white": "\033[37m",
    "reset": "\033[0m",
}


_STATE_COLORS = {
    "launching": "cyan",
    "isaac_startup": "yellow",
    "progress_pending": "yellow",
    "build_stage": "blue",
    "reset_episode": "cyan",
    "plan_nav_to_pick": "magenta",
    "exec_nav_to_pick": "green",
    "verify_pick_reachable": "yellow",
    "plan_pick": "magenta",
    "exec_pick": "green",
    "verify_pick_success": "yellow",
    "plan_nav_to_place": "magenta",
    "exec_nav_to_place": "green",
    "verify_place_reachable": "yellow",
    "plan_place": "magenta",
    "exec_place": "green",
    "verify_place_success": "yellow",
    "export_lerobot": "cyan",
    "cleanup_episode": "cyan",
    "done": "green",
    "failed": "red",
}


@dataclass(frozen=True)
class BatchEpisodeCommand:
    """单个 episode 的子进程命令和预期输出位置。"""

    episode_index: int
    seed: int
    output_dir: Path
    summary_path: Path
    command: list[str]
    num_episodes: int = 1


@dataclass(frozen=True)
class EpisodeProgress:
    """从 episode 输出文件中读取到的轻量状态机进度。"""

    state: str | None = None
    step_index: int | None = None
    source: str = "unavailable"
    episode_index: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class BatchEpisodeResult:
    """batch 结束表格的一行结构化结果。"""

    episode_index: int
    seed: int
    pick_place_xy: str
    base_goal_relative_xy: str
    success: bool
    training_quality_gate_passed: bool
    failed_state: str
    lerobot_path: str
    elapsed_seconds: float
    failure_reason: str


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "批量运行 full-physics episode；默认复用一个 Isaac 进程和 stage，"
            "也可回退到逐 episode 子进程。"
        ),
    )
    parser.add_argument(
        "--scene-profile",
        default="liangzhu",
        help="场景 profile 名称或别名；默认 liangzhu。",
    )
    parser.add_argument(
        "--pct-multifloor",
        action="store_const",
        const="multi_floor",
        dest="scene_profile",
        help="兼容旧命令：等价于 --scene-profile multi_floor。",
    )
    parser.add_argument(
        "--task-json",
        default=None,
        help="任务 JSON 路径；不传时由 scene profile 提供。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="批量运行输出目录；每个 episode 会写入独立子目录。",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行的 episode 数量。")
    parser.add_argument(
        "--reuse-isaac-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "多 episode 时复用同一 Isaac/IsaacLab 进程和 stage；默认开启。"
            "使用 --no-reuse-isaac-process 恢复逐 episode 子进程。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="首个 episode 的随机种子；后续 episode 自动使用 seed+1，重复命令可严格复现。",
    )
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按 episode seed 随机采样任务；不传时由 scene profile 决定。",
    )
    parser.add_argument(
        "--show-randomization-debug",
        action="store_true",
        help="显示 pick/place 随机区域和采样点；默认关闭。",
    )
    parser.add_argument(
        "--randomize-base-goal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="开启 base_goal 随机化；不传时由 scene profile 决定。",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="批量运行默认 headless；需要 GUI 时使用 --no-headless。",
    )
    parser.add_argument(
        "--navigation-visual-mode",
        choices=("auto", "collision", "full"),
        default=None,
        help="转发物理场景视觉模式；不传时由 scene profile 决定。",
    )
    parser.add_argument(
        "--global-planner",
        choices=("astar", "pct"),
        default=None,
        help="转发全局规划器；不传时由 scene profile 决定。",
    )
    parser.add_argument(
        "--pct-server-script",
        default=None,
        help="转发 PCT server 脚本路径。",
    )
    parser.add_argument("--pct-server-python", help="转发 PCT server Python 路径。")
    parser.add_argument(
        "--pct-tomogram-path",
        default=None,
        help="转发 PCT tomogram 路径。",
    )
    parser.add_argument(
        "--pct-walkable-path",
        default=None,
        help="转发 PCT walkable 路径。",
    )
    parser.add_argument(
        "--pct-collision-ply-path",
        default=None,
        help="转发 PCT collision PLY。",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--pct-no-fallback",
        action="store_true",
        dest="pct_no_fallback",
        default=None,
        help="禁止 PCT 失败时回退 A*。",
    )
    fallback_group.add_argument(
        "--pct-allow-fallback",
        "--pct-fallback-to-astar",
        action="store_false",
        dest="pct_no_fallback",
        help="允许 PCT 失败时回退 A*。",
    )
    parser.add_argument(
        "--pct-coord-mode",
        choices=("sim_to_pct_180deg", "identity"),
        default=None,
        help="转发 Isaac 到 PCT 的坐标变换模式。",
    )
    for axis in "xyz":
        parser.add_argument(
            f"--pct-offset-{axis}",
            type=float,
            default=None,
            help=f"转发 PCT 坐标 {axis.upper()} 偏移。",
        )
        parser.add_argument(
            f"--pct-scale-{axis}",
            type=float,
            default=None,
            help=f"转发 PCT 坐标 {axis.upper()} 缩放。",
        )
        parser.add_argument(
            f"--pct-rotation-{axis}-rad",
            type=float,
            default=None,
            help=f"转发 PCT 固定轴 {axis.upper()} 旋转弧度。",
        )
    parser.add_argument(
        "--pct-cross-floor-gateway",
        action="append",
        default=None,
        help="转发跨层 gateway；良渚单层默认 none。",
    )
    parser.add_argument(
        "--pct-cross-floor-stair-exit",
        action="append",
        default=None,
        help="转发跨层楼梯出口；良渚单层默认 none。",
    )
    parser.add_argument(
        "--pct-cross-floor-stair-midpoint",
        action="append",
        default=None,
        help="转发跨层楼梯中点；良渚单层默认 none。",
    )
    parser.add_argument(
        "--policy-profile",
        choices=("flat", "pct_multifloor"),
        default=None,
        help="转发 locomotion policy profile。",
    )
    parser.add_argument("--locomotion-task", default=None, help="转发 locomotion task 名称。")
    parser.add_argument(
        "--locomotion-checkpoint",
        default=None,
        help="转发 locomotion checkpoint。",
    )
    parser.add_argument(
        "--require-locomotion-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求 checkpoint 存在，缺失时立即失败；默认开启。",
    )
    parser.add_argument(
        "--continue-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="单个 episode 失败后是否继续后续 episode；默认继续。",
    )
    parser.add_argument(
        "--pick-plan-json",
        help="预先生成的 pick cuRobo 分段计划 JSON。",
    )
    parser.add_argument(
        "--place-plan-json",
        help="预先生成的 place cuRobo 分段计划 JSON。",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=5.0,
        help="子进程长时间无输出时，batch 心跳进度的打印间隔秒数。",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 ANSI 颜色打印 batch 进度；默认开启，可用 --no-color 关闭。",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "转发视频录制开关；完整 pipeline 默认录制 composite，"
            "可用 --no-record-video 显式关闭。"
        ),
    )
    parser.add_argument(
        "--dataset-camera-keys",
        nargs="+",
        choices=("front", "wrist", "overview"),
        default=None,
        help="转发训练数据相机流；默认由单 episode pipeline 使用 front wrist overview。",
    )
    parser.add_argument(
        "--video-mode",
        choices=("overview", "front", "font", "wrist", "composite", "all"),
        default=None,
        help=(
            "转发给单 episode pipeline：overview 为 third_person 展示视角，front/font 为前视 "
            "observation，wrist 为腕部 observation，composite 输出同步三视角拼接视频，"
            "all 同时导出 overview/front/wrist 三路独立视频。"
        ),
    )
    parser.add_argument(
        "--video-out",
        help="转发给单 episode pipeline：视频输出目录；batch 多 episode 不支持单个 .mp4 文件。",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="转发给单 episode pipeline：overview 捕获宽度，默认 1280。",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="转发给单 episode pipeline：overview 捕获高度，默认 720。",
    )
    parser.add_argument(
        "--overview-camera-mode",
        choices=("fixed", "auto"),
        default=None,
        help="转发给单 episode pipeline：fixed 固定相机，auto 按阶段切换。",
    )
    parser.add_argument(
        "--overview-camera-prim-path",
        default=None,
        help="转发 image/video/GUI 共用的 overview Camera prim。",
    )
    parser.add_argument(
        "--overview-camera-schedule",
        default=None,
        help="转发给单 episode pipeline：headless Camera0-8 切换规则 JSON。",
    )
    parser.add_argument(
        "--overview-capture-backend",
        choices=("viewport", "render_product", "auto"),
        default="viewport",
        help="转发给单 episode pipeline：overview 取帧后端；viewport 最接近 GUI。",
    )
    parser.add_argument(
        "--overview-initial-hold-frames",
        type=int,
        default=160,
        help="转发给单 episode pipeline：third_person1 初始镜头最少保持帧数，默认 160。",
    )
    parser.add_argument(
        "--overview-exposure",
        type=float,
        default=0.0,
        help="转发给单 episode pipeline：overview 曝光补偿 EV stops，默认 0。",
    )
    parser.add_argument(
        "--overview-gamma",
        type=float,
        default=2.2,
        help="转发给单 episode pipeline：overview 线性 RGB 转 sRGB gamma，默认 2.2。",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_const", const="dry_run", dest="mode")
    for mode_name, flag in _REAL_MODES.items():
        mode_group.add_argument(flag, action="store_const", const=mode_name, dest="mode")
    parser.set_defaults(mode="full_physics")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    try:
        args.scene_profile = load_scene_profile(
            args.scene_profile,
            PROJECT_ROOT,
        ).name
    except SceneProfileError as exc:
        raise SystemExit(str(exc)) from exc
    return args


def _bool_flag(enabled: bool, enabled_flag: str, disabled_flag: str) -> str:
    return enabled_flag if enabled else disabled_flag


def _color(text: str, name: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _print_banner(text: str, *, color_enabled: bool) -> None:
    line = "=" * 88
    print(_color(line, "blue", enabled=color_enabled), flush=True)
    print(_color(text, "bold", enabled=color_enabled), flush=True)
    print(_color(line, "blue", enabled=color_enabled), flush=True)


def _state_color(state: str | None) -> str:
    if not state:
        return "dim"
    return _STATE_COLORS.get(state, "bold")


def _format_state(state: str | None, *, color_enabled: bool) -> str:
    label = state or "unknown"
    return _color(label, _state_color(state), enabled=color_enabled)


def _last_json_line(
    path: Path,
    *,
    max_bytes: int = 262_144,
    min_mtime: float | None = None,
) -> dict[str, object] | None:
    """只读取文件尾部来解析最后一条 JSONL，避免 heartbeat 扫描完整 frames。"""

    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    if min_mtime is not None and stat.st_mtime < float(min_mtime):
        return None
    size = stat.st_size
    if size <= 0:
        return None
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, size - max_bytes))
            data = stream.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    for line in reversed(data.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return payload if isinstance(payload, dict) else None
    return None


def _progress_file_is_fresh(path: Path, *, min_mtime: float | None) -> bool:
    """Return whether a progress artifact belongs to the current child run."""

    try:
        if not path.is_file():
            return False
        return min_mtime is None or path.stat().st_mtime >= float(min_mtime)
    except OSError:
        return False


def _episode_progress_paths(
    episode_dir: Path,
    *,
    summary_name: str = "summary.json",
) -> tuple[list[Path], list[Path], list[Path]]:
    """Build direct and legacy-nested progress paths for one episode."""

    roots = (episode_dir, episode_dir / "episode_000000")
    return (
        [root / "frames.jsonl" for root in roots],
        [root / "events.jsonl" for root in roots],
        [root / summary_name for root in roots],
    )


def _progress_from_jsonl(
    path: Path,
    *,
    source: str,
    min_mtime: float | None,
) -> EpisodeProgress | None:
    payload = _last_json_line(path, min_mtime=min_mtime)
    if payload is None:
        return None
    state = payload.get("pipeline_state")
    step_index = payload.get("step_index")
    return EpisodeProgress(
        state=state if isinstance(state, str) and state else None,
        step_index=(
            int(step_index)
            if isinstance(step_index, int) and not isinstance(step_index, bool)
            else None
        ),
        source=source,
    )


def _progress_from_summary(
    path: Path,
    *,
    min_mtime: float | None = None,
) -> EpisodeProgress | None:
    summary = _read_summary(path, min_mtime=min_mtime)
    if summary is None:
        return None
    state = summary.get("final_state")
    if not isinstance(state, str) or not state:
        trace = summary.get("state_trace")
        if isinstance(trace, list) and trace:
            state = str(trace[-1])
    step_index = summary.get("duration_steps")
    return EpisodeProgress(
        state=state if isinstance(state, str) else None,
        step_index=int(step_index) if isinstance(step_index, int) else None,
        source="summary",
    )


def _read_episode_progress(
    episode: BatchEpisodeCommand,
    *,
    min_mtime: float | None = None,
) -> EpisodeProgress:
    active_episode_index = episode.episode_index
    active_episode_dir = episode.summary_path.parent
    active_summary_path = episode.summary_path
    if episode.num_episodes > 1:
        for candidate_index in range(episode.num_episodes - 1, -1, -1):
            candidate_dir = episode.output_dir / f"episode_{candidate_index:06d}"
            candidate_paths = _episode_progress_paths(
                candidate_dir,
                summary_name=episode.summary_path.name,
            )
            if any(
                _progress_file_is_fresh(path, min_mtime=min_mtime)
                for paths in candidate_paths
                for path in paths
            ):
                active_episode_index = candidate_index
                active_episode_dir = candidate_dir
                active_summary_path = candidate_dir / episode.summary_path.name
                break
    frames_candidates, events_candidates, summary_candidates = _episode_progress_paths(
        active_episode_dir,
        summary_name=active_summary_path.name,
    )
    episode_seed = episode.seed + active_episode_index - episode.episode_index
    for path in frames_candidates:
        frame_progress = _progress_from_jsonl(
            path,
            source="frames",
            min_mtime=min_mtime,
        )
        if frame_progress is None:
            continue
        return EpisodeProgress(
            state=frame_progress.state,
            step_index=frame_progress.step_index,
            source=frame_progress.source,
            episode_index=active_episode_index,
            seed=episode_seed,
        )
    for path in events_candidates:
        event_progress = _progress_from_jsonl(
            path,
            source="events",
            min_mtime=min_mtime,
        )
        if event_progress is None:
            continue
        return EpisodeProgress(
            state=event_progress.state,
            step_index=event_progress.step_index,
            source=event_progress.source,
            episode_index=active_episode_index,
            seed=episode_seed,
        )
    for path in summary_candidates:
        summary_progress = _progress_from_summary(path, min_mtime=min_mtime)
        if summary_progress is None:
            continue
        return EpisodeProgress(
            state=summary_progress.state,
            step_index=summary_progress.step_index,
            source=summary_progress.source,
            episode_index=active_episode_index,
            seed=episode_seed,
        )
    has_fresh_progress_file = any(
        _progress_file_is_fresh(path, min_mtime=min_mtime)
        for path in frames_candidates + events_candidates + summary_candidates
    )
    return EpisodeProgress(
        state="progress_pending" if has_fresh_progress_file else "isaac_startup",
        source="batch",
        episode_index=active_episode_index,
        seed=episode_seed,
    )


def _progress_is_low_information(progress: EpisodeProgress) -> bool:
    return progress.source == "unavailable" or (
        progress.source == "batch"
        and progress.state in {None, "isaac_startup", "progress_pending"}
    )


def _should_print_periodic_progress(
    progress: EpisodeProgress,
    *,
    last_printed_progress: EpisodeProgress | None,
    now: float,
    last_low_information_progress_at: float,
    progress_interval_s: float,
) -> bool:
    """Throttle startup/unavailable heartbeats while keeping real states live."""

    if not _progress_is_low_information(progress):
        return True
    if last_printed_progress is None:
        return True
    identity = (
        progress.state,
        progress.source,
        progress.episode_index,
        progress.seed,
    )
    previous_identity = (
        last_printed_progress.state,
        last_printed_progress.source,
        last_printed_progress.episode_index,
        last_printed_progress.seed,
    )
    return identity != previous_identity or (
        now - last_low_information_progress_at
        >= max(30.0, float(progress_interval_s))
    )


def _print_progress_line(
    episode: BatchEpisodeCommand,
    *,
    elapsed_seconds: float,
    progress: EpisodeProgress,
    color_enabled: bool,
) -> None:
    """用独立块打印 batch heartbeat，避免被 Isaac 多行日志夹断。"""

    prefix = _color("[progress] ", "yellow", enabled=color_enabled)
    displayed_episode_index = (
        episode.episode_index
        if progress.episode_index is None
        else progress.episode_index
    )
    displayed_seed = episode.seed if progress.seed is None else progress.seed
    progress_line = (
        prefix
        + (
            f"episode={displayed_episode_index} seed={displayed_seed} "
            f"running elapsed={_format_duration(elapsed_seconds)}"
        )
        + _format_progress_suffix(progress, color_enabled=color_enabled)
    )
    line = _color("-" * 88, "dim", enabled=color_enabled)
    print(line, flush=True)
    print(progress_line, flush=True)
    print(line, flush=True)


def _format_progress_suffix(
    progress: EpisodeProgress,
    *,
    color_enabled: bool,
) -> str:
    step_text = "?" if progress.step_index is None else str(progress.step_index)
    return (
        f" state={_format_state(progress.state, color_enabled=color_enabled)}"
        f" step={_color(step_text, 'cyan', enabled=color_enabled)}"
        f" source={_color(progress.source, 'dim', enabled=color_enabled)}"
    )


def _summary_xy(summary: dict[str, object] | None) -> str:
    """优先读取随机化采样点，关闭随机化时回退任务中的实际目标坐标。"""

    if not summary:
        return "pick=- place=-"
    task = summary.get("task_config")
    task = task if isinstance(task, dict) else {}
    randomization = task.get("randomization")
    randomization = randomization if isinstance(randomization, dict) else {}

    def _sampled_xy(key: str, section: str, pose_key: str) -> tuple[float, float] | None:
        random_section = randomization.get(key)
        if isinstance(random_section, dict):
            sampled = random_section.get("sampled_xy")
            if isinstance(sampled, dict) and "x" in sampled and "y" in sampled:
                return float(sampled["x"]), float(sampled["y"])
        task_section = task.get(section)
        if isinstance(task_section, dict):
            pose = task_section.get(pose_key)
            if isinstance(pose, dict) and "x" in pose and "y" in pose:
                return float(pose["x"]), float(pose["y"])
        return None

    pick_xy = _sampled_xy("object_xy_randomization", "pick", "object_pose_world")
    place_xy = _sampled_xy("place_xy_randomization", "place", "place_pose_world")

    def _format_xy(value: tuple[float, float] | None) -> str:
        return "-" if value is None else f"({value[0]:.4f},{value[1]:.4f})"

    return f"pick={_format_xy(pick_xy)} place={_format_xy(place_xy)}"


def _summary_base_goal_relative_xy(summary: dict[str, object] | None) -> str:
    """汇总随机化 base_goal，并给出 base_goal 相对目标点的世界系 XY 偏移。"""

    if not summary:
        return "pick=- place=-"
    task = summary.get("task_config")
    task = task if isinstance(task, dict) else {}
    randomization = task.get("randomization")
    randomization = randomization if isinstance(randomization, dict) else {}
    base_goal_randomization = randomization.get("base_goal_randomization")
    base_goal_randomization = (
        base_goal_randomization if isinstance(base_goal_randomization, dict) else {}
    )

    def _task_target_xy(section: str, pose_key: str) -> tuple[float, float] | None:
        task_section = task.get(section)
        if not isinstance(task_section, dict):
            return None
        pose = task_section.get(pose_key)
        if isinstance(pose, dict) and "x" in pose and "y" in pose:
            return float(pose["x"]), float(pose["y"])
        return None

    def _task_goal_xyyaw(section: str) -> tuple[float, float, float] | None:
        task_section = task.get(section)
        if not isinstance(task_section, dict):
            return None
        goal = task_section.get("base_goal")
        if isinstance(goal, dict) and "x" in goal and "y" in goal:
            return (
                float(goal["x"]),
                float(goal["y"]),
                float(goal.get("yaw", 0.0)),
            )
        return None

    def _stage_summary(
        stage_name: str,
        section: str,
        pose_key: str,
        top_level_key: str,
    ) -> str:
        sample = base_goal_randomization.get(stage_name)
        sample = sample if isinstance(sample, dict) else {}
        goal = sample.get("sampled_base_goal_xyyaw")
        target = sample.get("target_xy")
        if not (isinstance(goal, (list, tuple)) and len(goal) >= 3):
            goal = summary.get(top_level_key)
        if not (isinstance(goal, (list, tuple)) and len(goal) >= 3):
            goal = _task_goal_xyyaw(section)
        if not (isinstance(target, (list, tuple)) and len(target) >= 2):
            target = _task_target_xy(section, pose_key)
        if not (isinstance(goal, (list, tuple)) and len(goal) >= 3):
            return f"{stage_name}=-"
        goal_x, goal_y, goal_yaw = float(goal[0]), float(goal[1]), float(goal[2])
        if isinstance(target, (list, tuple)) and len(target) >= 2:
            dx = goal_x - float(target[0])
            dy = goal_y - float(target[1])
            delta = f" Δ=({dx:+.4f},{dy:+.4f})"
        else:
            delta = " Δ=-"
        return f"{stage_name}_bg=({goal_x:.4f},{goal_y:.4f},{goal_yaw:.3f}){delta}"

    return " ".join(
        (
            _stage_summary(
                "pick",
                "pick",
                "object_pose_world",
                "pick_base_goal_sampled",
            ),
            _stage_summary(
                "place",
                "place",
                "place_pose_world",
                "place_base_goal_sampled",
            ),
        )
    )


def _failed_state(summary: dict[str, object] | None, *, success: bool) -> str:
    if success:
        return "-"
    if summary:
        failure_metadata = summary.get("failure_metadata")
        if isinstance(failure_metadata, dict):
            current_state = failure_metadata.get("current_state")
            if isinstance(current_state, str) and current_state:
                return current_state
        trace = summary.get("state_trace")
        if isinstance(trace, list) and trace:
            if str(trace[-1]) == "failed" and len(trace) > 1:
                return str(trace[-2])
            return str(trace[-1])
        final_state = summary.get("final_state")
        if isinstance(final_state, str) and final_state:
            return final_state
    return "summary_missing"


def _lerobot_path(
    episode: BatchEpisodeCommand,
    summary: dict[str, object] | None,
) -> str:
    if summary:
        export = summary.get("lerobot_export")
        if isinstance(export, dict):
            for key in (
                "dataset_path",
                "output_path",
                "data_path",
                "manifest_path",
                "source_frames",
            ):
                value = export.get(key)
                if isinstance(value, str) and value:
                    return value
        data_output_path = summary.get("data_output_path")
        if isinstance(data_output_path, str) and data_output_path:
            return data_output_path
    return str(episode.summary_path.parent)


def _build_episode_result(
    *,
    episode: BatchEpisodeCommand,
    summary: dict[str, object] | None,
    success: bool,
    elapsed_seconds: float,
) -> BatchEpisodeResult:
    return BatchEpisodeResult(
        episode_index=episode.episode_index,
        seed=episode.seed,
        pick_place_xy=_summary_xy(summary),
        base_goal_relative_xy=_summary_base_goal_relative_xy(summary),
        success=success,
        training_quality_gate_passed=_training_quality_gate_passed(summary),
        failed_state=_failed_state(summary, success=success),
        lerobot_path=_lerobot_path(episode, summary),
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
        failure_reason=(
            str(summary.get("failure_reason") or "")
            if summary
            else "summary_missing"
        ),
    )


def _training_quality_gate_passed(
    summary: dict[str, object] | None,
) -> bool:
    """批量训练只接收具有最终物理执行来源证据的 episode。"""

    return bool(
        summary
        and summary.get("training_quality_gate_passed") is True
        and training_quality_success_verified(summary)
    )


def _display_width(text: str) -> int:
    """按终端显示宽度计算中英文混排文本长度。"""

    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in text
    )


def _pad_cell(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _format_result_table(
    results: Sequence[BatchEpisodeResult],
    *,
    color_enabled: bool,
) -> str:
    headers = (
        "Episode",
        "随机化 Pick / Place XY",
        "随机化 BaseGoal / 相对目标",
        "Pipeline 成功",
        "训练质量门禁",
        "失败 State",
        "LeRobot 数据路径",
        "Episode 耗时",
    )
    rows = [
        (
            f"{result.episode_index} (seed={result.seed})",
            result.pick_place_xy,
            result.base_goal_relative_xy,
            "成功" if result.success else "失败",
            "通过" if result.training_quality_gate_passed else "拒绝",
            result.failed_state,
            result.lerobot_path,
            _format_duration(result.elapsed_seconds),
        )
        for result in results
    ]
    widths = [
        max(_display_width(headers[index]), *(_display_width(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    column_colors = (
        "cyan",
        "magenta",
        "yellow",
        "green",
        "green",
        "red",
        "blue",
        "white",
    )
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _row(values: Sequence[str], *, header: bool = False) -> str:
        cells: list[str] = []
        for index, value in enumerate(values):
            color_name = column_colors[index]
            if not header and index == 3 and value == "失败":
                color_name = "red"
            if not header and index == 4 and value == "拒绝":
                color_name = "red"
            padded = _pad_cell(value, widths[index])
            cells.append(_color(padded, color_name, enabled=color_enabled))
        return "| " + " | ".join(cells) + " |"

    lines = [separator, _row(headers, header=True), separator]
    lines.extend(_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def _print_result_table(
    results: Sequence[BatchEpisodeResult],
    *,
    color_enabled: bool,
) -> None:
    print(
        _color("[full-physics-batch] Episode 汇总", "bold", enabled=color_enabled),
        flush=True,
    )
    print(_format_result_table(results, color_enabled=color_enabled), flush=True)


def _build_child_command(
    args: argparse.Namespace,
    *,
    episode_index: int,
) -> BatchEpisodeCommand:
    """构造单 episode 子进程命令；真实仿真仍由原入口维护单 World 生命周期。"""

    output_root = _project_path(args.output_dir)
    episode_output_dir = output_root / f"episode_{episode_index:06d}"
    episode_seed = int(args.seed) + int(episode_index)
    command = [
        sys.executable,
        "-B",
        str(PIPELINE_ENTRY),
        "--scene-profile",
        str(args.scene_profile),
        "--output-dir",
        str(episode_output_dir),
        "--num-episodes",
        "1",
        "--seed",
        str(episode_seed),
        _bool_flag(args.headless, "--headless", "--no-headless"),
    ]
    if args.task_json:
        command.extend(["--task-json", str(_project_path(args.task_json))])
    if args.dataset_camera_keys is not None:
        command.append("--dataset-camera-keys")
        command.extend(str(value) for value in args.dataset_camera_keys)
    for value, enabled_flag, disabled_flag in (
        (args.randomize_task, "--randomize-task", "--no-randomize-task"),
        (
            args.randomize_base_goal,
            "--randomize-base-goal",
            "--no-randomize-base-goal",
        ),
        (
            args.require_locomotion_checkpoint,
            "--require-locomotion-checkpoint",
            "--no-require-locomotion-checkpoint",
        ),
        (args.record_video, "--record-video", "--no-record-video"),
    ):
        if value is not None:
            command.append(_bool_flag(bool(value), enabled_flag, disabled_flag))

    for argument_name in (
        "navigation_visual_mode",
        "global_planner",
        "policy_profile",
        "pct_coord_mode",
        "locomotion_task",
        "overview_camera_mode",
        "overview_camera_prim_path",
        "video_mode",
    ):
        value = getattr(args, argument_name)
        if value is not None:
            command.extend([f"--{argument_name.replace('_', '-')}", str(value)])

    for argument_name in (
        "pct_offset_x",
        "pct_offset_y",
        "pct_offset_z",
        "pct_scale_x",
        "pct_scale_y",
        "pct_scale_z",
        "pct_rotation_x_rad",
        "pct_rotation_y_rad",
        "pct_rotation_z_rad",
    ):
        value = getattr(args, argument_name)
        if value is not None:
            command.extend([f"--{argument_name.replace('_', '-')}", str(value)])

    for argument_name in (
        "pct_server_script",
        "pct_server_python",
        "pct_tomogram_path",
        "pct_walkable_path",
        "pct_collision_ply_path",
        "locomotion_checkpoint",
    ):
        value = getattr(args, argument_name)
        if value:
            command.extend(
                [f"--{argument_name.replace('_', '-')}", str(_project_path(value))]
            )

    if args.pct_no_fallback is not None:
        command.append(
            "--pct-no-fallback" if args.pct_no_fallback else "--pct-allow-fallback"
        )
    for argument_name in (
        "pct_cross_floor_gateway",
        "pct_cross_floor_stair_exit",
        "pct_cross_floor_stair_midpoint",
    ):
        values = getattr(args, argument_name)
        if values is None:
            continue
        for value in values:
            command.extend([f"--{argument_name.replace('_', '-')}", str(value)])

    if args.mode == "dry_run":
        command.append("--dry-run")
    elif args.mode != "full_physics":
        command.append(_REAL_MODES[args.mode])
    if args.show_randomization_debug:
        command.append("--show-randomization-debug")
    if args.pick_plan_json and args.mode != "full_physics":
        command.extend(["--pick-plan-json", str(_project_path(args.pick_plan_json))])
    if args.place_plan_json and args.mode != "full_physics":
        command.extend(["--place-plan-json", str(_project_path(args.place_plan_json))])

    command.extend(["--video-width", str(int(args.video_width))])
    command.extend(["--video-height", str(int(args.video_height))])
    command.extend(["--overview-capture-backend", str(args.overview_capture_backend)])
    command.extend(
        [
            "--overview-initial-hold-frames",
            str(int(args.overview_initial_hold_frames)),
        ]
    )
    command.extend(["--overview-exposure", str(float(args.overview_exposure))])
    command.extend(["--overview-gamma", str(float(args.overview_gamma))])
    if args.overview_camera_schedule:
        command.extend(
            [
                "--overview-camera-schedule",
                str(_project_path(args.overview_camera_schedule)),
            ]
        )
    if args.video_out:
        video_output_dir = _project_path(args.video_out) / f"episode_{episode_index:06d}"
        command.extend(["--video-out", str(video_output_dir)])

    return BatchEpisodeCommand(
        episode_index=episode_index,
        seed=episode_seed,
        output_dir=episode_output_dir,
        summary_path=episode_output_dir / "summary.json",
        command=command,
    )


def _replace_command_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise RuntimeError(f"child command is missing required option: {option}") from exc
    if index + 1 >= len(command):
        raise RuntimeError(f"child command option has no value: {option}")
    command[index + 1] = value


def _build_reused_process_command(args: argparse.Namespace) -> BatchEpisodeCommand:
    """Build one child command that runs every episode in one live Isaac stage."""

    base = _build_child_command(args, episode_index=0)
    output_root = _project_path(args.output_dir)
    command = list(base.command)
    _replace_command_option(command, "--output-dir", str(output_root))
    _replace_command_option(command, "--num-episodes", str(int(args.num_episodes)))
    if "--no-reuse-isaac-stage" in command:
        command.remove("--no-reuse-isaac-stage")
    if "--reuse-isaac-stage" not in command:
        command.append("--reuse-isaac-stage")
    if args.video_out and "--video-out" in command:
        _replace_command_option(command, "--video-out", str(_project_path(args.video_out)))
    return BatchEpisodeCommand(
        episode_index=0,
        seed=int(args.seed),
        output_dir=output_root,
        summary_path=output_root / "episode_000000" / "summary.json",
        command=command,
        num_episodes=int(args.num_episodes),
    )


def _read_summary(
    path: Path,
    *,
    min_mtime: float | None = None,
) -> dict[str, object] | None:
    try:
        if not path.is_file():
            legacy_path = path.parent / "episode_000000" / path.name
            if legacy_path.is_file():
                path = legacy_path
        if not path.is_file():
            return None
        if min_mtime is not None and path.stat().st_mtime < float(min_mtime):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_child_process(
    episode: BatchEpisodeCommand,
    *,
    env: dict[str, str],
    progress_interval_s: float,
    color_enabled: bool,
) -> int:
    """实时转发子进程日志，并在无输出时打印 batch 心跳。"""

    def _drain_available_output(timeout_s: float) -> None:
        """尽量排空当前 pipe，避免 heartbeat 插入 Isaac 的多行表格中间。"""

        while True:
            events = selector.select(timeout=timeout_s)
            if not events:
                return
            timeout_s = 0.0
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    return

    started_at = time.monotonic()
    started_at_epoch = time.time()
    progress_interval_s = max(0.1, float(progress_interval_s))
    _print_banner(
        f"[full-physics-batch] start episode={episode.episode_index} seed={episode.seed}",
        color_enabled=color_enabled,
    )
    print(
        _color("[cmd] ", "dim", enabled=color_enabled) + shlex.join(episode.command),
        flush=True,
    )
    print(
        _color("[out] ", "cyan", enabled=color_enabled)
        + "子进程日志开始；长时间无输出时会打印 batch heartbeat。",
        flush=True,
    )

    process = subprocess.Popen(
        episode.command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_progress_at = started_at
    last_printed_progress: EpisodeProgress | None = None
    last_low_information_progress_at = started_at
    select_timeout_s = min(0.5, progress_interval_s * 0.5)
    launching_progress = EpisodeProgress(state="launching", source="batch")
    _print_progress_line(
        episode,
        elapsed_seconds=0.0,
        progress=launching_progress,
        color_enabled=color_enabled,
    )
    last_printed_progress = launching_progress
    try:
        while True:
            _drain_available_output(select_timeout_s)
            returncode = process.poll()
            now = time.monotonic()
            if now - last_progress_at >= progress_interval_s:
                _drain_available_output(0.0)
                progress = _read_episode_progress(
                    episode,
                    min_mtime=started_at_epoch,
                )
                should_print = _should_print_periodic_progress(
                    progress,
                    last_printed_progress=last_printed_progress,
                    now=now,
                    last_low_information_progress_at=(
                        last_low_information_progress_at
                    ),
                    progress_interval_s=progress_interval_s,
                )
                if should_print:
                    _print_progress_line(
                        episode,
                        elapsed_seconds=now - started_at,
                        progress=progress,
                        color_enabled=color_enabled,
                    )
                    last_printed_progress = progress
                    if _progress_is_low_information(progress):
                        last_low_information_progress_at = now
                last_progress_at = now
            if returncode is not None:
                remaining = process.stdout.read()
                if remaining:
                    print(remaining, end="", flush=True)
                return int(returncode)
    except KeyboardInterrupt:
        print(
            _color("[interrupt] ", "red", enabled=color_enabled)
            + f"收到 Ctrl+C，正在终止 episode={episode.episode_index} 子进程。",
            flush=True,
        )
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        raise
    finally:
        selector.close()
        process.stdout.close()


def _write_batch_record(
    stream,
    *,
    episode: BatchEpisodeCommand,
    returncode: int,
    summary: dict[str, object] | None,
    result: BatchEpisodeResult,
) -> None:
    record = {
        "episode_index": episode.episode_index,
        "seed": episode.seed,
        "returncode": returncode,
        "output_dir": str(episode.output_dir),
        "summary_path": str(episode.summary_path),
        "success": bool(summary.get("success")) if summary else False,
        "training_quality_gate_passed": _training_quality_gate_passed(summary),
        "lerobot_training_eligible": (
            summary.get("lerobot_training_eligible") if summary else False
        ),
        "failure_reason": summary.get("failure_reason") if summary else "summary_missing",
        "execution_mode": summary.get("execution_mode") if summary else None,
        "base_goal_randomization_enabled": (
            summary.get("base_goal_randomization_enabled") if summary else None
        ),
        "pick_base_goal_sampled": summary.get("pick_base_goal_sampled") if summary else None,
        "place_base_goal_sampled": summary.get("place_base_goal_sampled") if summary else None,
        "pick_base_goal_fallback_used": (
            summary.get("pick_base_goal_fallback_used") if summary else None
        ),
        "place_base_goal_fallback_used": (
            summary.get("place_base_goal_fallback_used") if summary else None
        ),
        "pick_place_xy": result.pick_place_xy,
        "base_goal_relative_xy": result.base_goal_relative_xy,
        "failed_state": result.failed_state,
        "lerobot_path": result.lerobot_path,
        "elapsed_seconds": result.elapsed_seconds,
    }
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def _materialize_batch_lerobot(
    output_root: Path,
    episode_dirs: Sequence[Path],
) -> dict[str, object]:
    """只合并本次通过物理来源门禁的 episode，避免旧目录和 smoke 污染。"""

    from source.recording import materialize_lerobot_dataset

    report = materialize_lerobot_dataset(
        episode_dirs,
        output_root / "lerobot_dataset",
    )
    manifest_path = output_root / "lerobot_export_manifest.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {**report, "manifest_path": str(manifest_path)}


def _summary_elapsed_seconds(summary: dict[str, object] | None) -> float:
    if summary is None:
        return 0.0
    performance = summary.get("performance_report")
    if isinstance(performance, dict):
        wall_seconds = performance.get("wall_seconds")
        if isinstance(wall_seconds, (int, float)):
            return max(0.0, float(wall_seconds))
    duration = summary.get("duration_seconds")
    return max(0.0, float(duration)) if isinstance(duration, (int, float)) else 0.0


def _validate_pct_scan_batch_contract(args: argparse.Namespace) -> None:
    """在 ROS 2 episode epoch 协议完成前拒绝批量导航与旧 planner 开关。"""

    if args.mode not in _NAVIGATION_MODES:
        return
    if args.num_episodes != 1:
        raise SystemExit(
            "PCT→SCAN ROS 2 导航在 episode epoch/reset/ack 协议完成前"
            "只允许单 episode；batch 多 episode 已失败关闭。"
        )
    if args.global_planner not in {None, "pct"}:
        raise SystemExit("pct-scan batch 导航只允许 PCT 全局规划器。")
    if args.pct_no_fallback is False:
        raise SystemExit("pct-scan batch 导航禁止 PCT→A* fallback。")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be positive.")
    _validate_pct_scan_batch_contract(args)
    if not args.headless and args.num_episodes > 1:
        raise SystemExit("批量 GUI 运行容易阻塞自动化；多 episode 请使用 --headless。")
    if args.progress_interval_s <= 0.0:
        raise SystemExit("--progress-interval-s must be positive.")
    if args.record_video and (args.video_width <= 0 or args.video_height <= 0):
        raise SystemExit("--video-width and --video-height must be positive.")
    if args.record_video and args.overview_initial_hold_frames < 0:
        raise SystemExit("--overview-initial-hold-frames must be non-negative.")
    if args.record_video and args.overview_gamma <= 0:
        raise SystemExit("--overview-gamma must be positive.")
    if args.record_video and args.mode == "dry_run":
        raise SystemExit("--record-video 需要真实 Isaac stage / camera images，不能与 --dry-run 一起使用。")
    if args.record_video and args.video_out and Path(args.video_out).suffix.lower() == ".mp4":
        raise SystemExit("batch --video-out 请传输出目录；单个 .mp4 文件只适用于单 episode pipeline。")
    output_root = _project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = output_root / "batch_summary.jsonl"
    batch_summary_path.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # 子进程 stdout 进入 pipe 时默认可能块缓冲；强制无缓冲便于实时观察仿真进度。
    env["PYTHONUNBUFFERED"] = "1"
    # batch 只在全部子进程结束后生成统一 dataset，避免每个 episode 重复编码一份。
    env["FULL_PHYSICS_DEFER_LEROBOT_EXPORT"] = "1"
    reuse_isaac_process = bool(
        args.reuse_isaac_process
        and args.mode == "full_physics"
        and args.num_episodes > 1
        and args.headless
        and args.continue_on_failure
    )
    if reuse_isaac_process:
        # One child owns output_root/episode_N and one live Isaac stage.
        env.pop("FULL_PHYSICS_FLAT_EPISODE_OUTPUT", None)
        env["FULL_PHYSICS_BATCH_SUMMARY_NAME"] = "pipeline_batch_summary.jsonl"
    else:
        # Isolated children already run inside their final episode directory.
        env["FULL_PHYSICS_FLAT_EPISODE_OUTPUT"] = "1"
        env.pop("FULL_PHYSICS_BATCH_SUMMARY_NAME", None)
    all_success = True
    completed = 0
    batch_started_at = time.monotonic()
    color_enabled = bool(args.color) and "NO_COLOR" not in env
    episode_results: list[BatchEpisodeResult] = []
    training_accepted_episode_dirs: list[Path] = []
    _print_banner(
        (
            f"[full-physics-batch] mode={args.mode} episodes={args.num_episodes} "
            f"seed_range={args.seed}..{args.seed + args.num_episodes - 1} "
            f"reuse_isaac_process={reuse_isaac_process} output={output_root}"
        ),
        color_enabled=color_enabled,
    )

    def _consume_completed_episode(
        summary_stream,
        *,
        episode: BatchEpisodeCommand,
        returncode: int,
        elapsed_seconds: float,
        min_mtime: float,
        require_zero_returncode: bool,
    ) -> bool:
        nonlocal all_success, completed
        summary = _read_summary(episode.summary_path, min_mtime=min_mtime)
        success = bool(
            summary
            and summary.get("success")
            and (returncode == 0 or not require_zero_returncode)
        )
        result = _build_episode_result(
            episode=episode,
            summary=summary,
            success=success,
            elapsed_seconds=elapsed_seconds,
        )
        episode_results.append(result)
        training_quality_passed = _training_quality_gate_passed(summary)
        if args.mode == "full_physics" and success and training_quality_passed:
            training_accepted_episode_dirs.append(episode.output_dir)
        episode_accepted = bool(
            success and (args.mode != "full_physics" or training_quality_passed)
        )
        all_success = all_success and episode_accepted
        completed += 1
        _write_batch_record(
            summary_stream,
            episode=episode,
            returncode=returncode,
            summary=summary,
            result=result,
        )
        status_text = (
            "success"
            if episode_accepted
            else ("quality-rejected" if success else "failed")
        )
        status_color = "green" if episode_accepted else "red"
        final_progress = _progress_from_summary(
            episode.summary_path,
            min_mtime=min_mtime,
        ) or _read_episode_progress(episode, min_mtime=min_mtime)
        print(
            _color(f"[{status_text}] ", status_color, enabled=color_enabled)
            + (
                f"episode={episode.episode_index} seed={episode.seed} "
                f"returncode={returncode} "
                f"elapsed={_format_duration(elapsed_seconds)} "
                f"summary={episode.summary_path}"
            ),
            _format_progress_suffix(final_progress, color_enabled=color_enabled),
            flush=True,
        )
        return episode_accepted

    with batch_summary_path.open("a", encoding="utf-8") as summary_stream:
        if reuse_isaac_process:
            shared_command = _build_reused_process_command(args)
            shared_started_at_epoch = time.time()
            returncode = _run_child_process(
                shared_command,
                env=env,
                progress_interval_s=args.progress_interval_s,
                color_enabled=color_enabled,
            )
            for episode_index in range(args.num_episodes):
                episode_output_dir = output_root / f"episode_{episode_index:06d}"
                episode = BatchEpisodeCommand(
                    episode_index=episode_index,
                    seed=int(args.seed) + episode_index,
                    output_dir=episode_output_dir,
                    summary_path=episode_output_dir / "summary.json",
                    command=shared_command.command,
                )
                summary = _read_summary(
                    episode.summary_path,
                    min_mtime=shared_started_at_epoch,
                )
                _consume_completed_episode(
                    summary_stream,
                    episode=episode,
                    returncode=returncode,
                    elapsed_seconds=_summary_elapsed_seconds(summary),
                    min_mtime=shared_started_at_epoch,
                    require_zero_returncode=False,
                )
        else:
            for episode_index in range(args.num_episodes):
                episode = _build_child_command(args, episode_index=episode_index)
                episode.output_dir.mkdir(parents=True, exist_ok=True)
                episode_started_at = time.monotonic()
                episode_started_at_epoch = time.time()
                returncode = _run_child_process(
                    episode,
                    env=env,
                    progress_interval_s=args.progress_interval_s,
                    color_enabled=color_enabled,
                )
                episode_elapsed_seconds = time.monotonic() - episode_started_at
                episode_accepted = _consume_completed_episode(
                    summary_stream,
                    episode=episode,
                    returncode=returncode,
                    elapsed_seconds=episode_elapsed_seconds,
                    min_mtime=episode_started_at_epoch,
                    require_zero_returncode=True,
                )
                if not episode_accepted and not args.continue_on_failure:
                    break
    lerobot_report: dict[str, object] | None = None
    if args.mode == "full_physics":
        lerobot_report = _materialize_batch_lerobot(
            output_root,
            training_accepted_episode_dirs,
        )
        all_success = bool(all_success and lerobot_report.get("lerobot_exported"))
        export_status = "success" if lerobot_report.get("lerobot_exported") else "pending"
        export_color = "green" if export_status == "success" else "yellow"
        print(
            _color(f"[lerobot-{export_status}] ", export_color, enabled=color_enabled)
            + json.dumps(lerobot_report, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
    total_elapsed = _format_duration(time.monotonic() - batch_started_at)
    final_color = "green" if all_success else "red"
    _print_banner(
        (
            f"[full-physics-batch] done completed={completed}/{args.num_episodes} "
            f"success={all_success} elapsed={total_elapsed} "
            f"batch_summary={batch_summary_path}"
        ),
        color_enabled=color_enabled,
    )
    _print_result_table(episode_results, color_enabled=color_enabled)
    print(_color(f"[full-physics-batch] final={all_success}", final_color, enabled=color_enabled))
    return 0 if all_success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[full-physics-batch] interrupted by user", flush=True)
        raise SystemExit(130)
