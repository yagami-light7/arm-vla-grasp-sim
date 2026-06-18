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


_REAL_MODES = {
    "simulation_smoke": "--simulation-smoke",
    "navigation_smoke": "--navigation-smoke",
    "navigation_carry_smoke": "--navigation-carry-smoke",
    "manipulation_apply_smoke": "--manipulation-apply-smoke",
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


@dataclass(frozen=True)
class EpisodeProgress:
    """从 episode 输出文件中读取到的轻量状态机进度。"""

    state: str | None = None
    step_index: int | None = None
    source: str = "unavailable"


@dataclass(frozen=True)
class BatchEpisodeResult:
    """batch 结束表格的一行结构化结果。"""

    episode_index: int
    seed: int
    pick_place_xy: str
    base_goal_relative_xy: str
    success: bool
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
        description="逐个子进程运行 full-physics episode，适用于真实 Isaac 自动化批量验证。",
    )
    parser.add_argument("--task-json", required=True, help="任务 JSON 路径。")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="批量运行输出目录；每个 episode 会写入独立子目录。",
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="运行的 episode 数量。")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="首个 episode 的随机种子；后续 episode 自动使用 seed+1，重复命令可严格复现。",
    )
    parser.add_argument(
        "--randomize-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按 episode seed 随机采样 pick/place XY；默认开启。",
    )
    parser.add_argument(
        "--show-randomization-debug",
        action="store_true",
        help="显示 pick/place 随机区域和采样点；默认关闭。",
    )
    parser.add_argument(
        "--randomize-base-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转发给单 episode pipeline：开启 pick/place base_goal 极坐标随机化；默认开启。",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="批量运行默认 headless；需要 GUI 时使用 --no-headless。",
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

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_const", const="dry_run", dest="mode")
    for mode_name, flag in _REAL_MODES.items():
        mode_group.add_argument(flag, action="store_const", const=mode_name, dest="mode")
    parser.set_defaults(mode="full_physics")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


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

    if not path.is_file():
        return None
    if min_mtime is not None and path.stat().st_mtime < float(min_mtime):
        return None
    size = path.stat().st_size
    if size <= 0:
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, size - max_bytes))
        data = stream.read().decode("utf-8", errors="ignore")
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
    frames_candidates = [
        episode.summary_path.parent / "frames.jsonl",
        episode.output_dir / "episode_000000" / "frames.jsonl",
    ]
    summary_candidates = [
        episode.summary_path,
        episode.summary_path.parent / "episode_000000" / episode.summary_path.name,
    ]
    any_progress_file_exists = any(path.is_file() for path in frames_candidates + summary_candidates)
    frames_path = next((path for path in frames_candidates if path.is_file()), frames_candidates[0])
    frame = _last_json_line(frames_path, min_mtime=min_mtime)
    if frame is not None:
        state = frame.get("pipeline_state")
        step_index = frame.get("step_index")
        return EpisodeProgress(
            state=state if isinstance(state, str) else None,
            step_index=int(step_index) if isinstance(step_index, int) else None,
            source="frames",
        )
    summary_progress = _progress_from_summary(episode.summary_path, min_mtime=min_mtime)
    if summary_progress is not None:
        return summary_progress
    if any_progress_file_exists:
        return EpisodeProgress()
    return EpisodeProgress(state="isaac_startup", source="batch")


def _print_progress_line(
    episode: BatchEpisodeCommand,
    *,
    elapsed_seconds: float,
    progress: EpisodeProgress,
    color_enabled: bool,
) -> None:
    """用独立块打印 batch heartbeat，避免被 Isaac 多行日志夹断。"""

    prefix = _color("[progress] ", "yellow", enabled=color_enabled)
    progress_line = (
        prefix
        + (
            f"episode={episode.episode_index} seed={episode.seed} "
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
        failed_state=_failed_state(summary, success=success),
        lerobot_path=_lerobot_path(episode, summary),
        elapsed_seconds=max(0.0, float(elapsed_seconds)),
        failure_reason=(
            str(summary.get("failure_reason") or "")
            if summary
            else "summary_missing"
        ),
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
    column_colors = ("cyan", "magenta", "yellow", "green", "red", "blue", "white")
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def _row(values: Sequence[str], *, header: bool = False) -> str:
        cells: list[str] = []
        for index, value in enumerate(values):
            color_name = column_colors[index]
            if not header and index == 3 and value == "失败":
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
        "--task-json",
        str(_project_path(args.task_json)),
        "--output-dir",
        str(episode_output_dir),
        "--num-episodes",
        "1",
        "--seed",
        str(episode_seed),
        _bool_flag(args.randomize_task, "--randomize-task", "--no-randomize-task"),
        _bool_flag(
            args.randomize_base_goal,
            "--randomize-base-goal",
            "--no-randomize-base-goal",
        ),
        _bool_flag(args.headless, "--headless", "--no-headless"),
    ]
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

    return BatchEpisodeCommand(
        episode_index=episode_index,
        seed=episode_seed,
        output_dir=episode_output_dir,
        summary_path=episode_output_dir / "summary.json",
        command=command,
    )


def _read_summary(
    path: Path,
    *,
    min_mtime: float | None = None,
) -> dict[str, object] | None:
    if not path.is_file():
        legacy_path = path.parent / "episode_000000" / path.name
        if legacy_path.is_file():
            path = legacy_path
    if not path.is_file():
        return None
    if min_mtime is not None and path.stat().st_mtime < float(min_mtime):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    last_unknown_progress_at = started_at
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
                startup_waiting = progress.source == "batch" and progress.state == "isaac_startup"
                should_print = (
                    (progress.source != "unavailable" and not startup_waiting)
                    or last_printed_progress is None
                    or progress.state != last_printed_progress.state
                    or now - last_unknown_progress_at >= max(30.0, progress_interval_s)
                )
                if should_print:
                    _print_progress_line(
                        episode,
                        elapsed_seconds=now - started_at,
                        progress=progress,
                        color_enabled=color_enabled,
                    )
                    last_printed_progress = progress
                    if progress.source == "batch" and progress.state == "isaac_startup":
                        last_unknown_progress_at = now
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
    """只把本次 batch 成功的 episode 合并，避免旧目录污染训练数据。"""

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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be positive.")
    if not args.headless and args.num_episodes > 1:
        raise SystemExit("批量 GUI 运行容易阻塞自动化；多 episode 请使用 --headless。")
    if args.progress_interval_s <= 0.0:
        raise SystemExit("--progress-interval-s must be positive.")
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
    # 子进程已经位于 batch episode 目录，不再额外创建 episode_000000。
    env["FULL_PHYSICS_FLAT_EPISODE_OUTPUT"] = "1"
    all_success = True
    completed = 0
    batch_started_at = time.monotonic()
    color_enabled = bool(args.color) and "NO_COLOR" not in env
    episode_results: list[BatchEpisodeResult] = []
    successful_episode_dirs: list[Path] = []
    _print_banner(
        (
            f"[full-physics-batch] mode={args.mode} episodes={args.num_episodes} "
            f"seed_range={args.seed}..{args.seed + args.num_episodes - 1} "
            f"output={output_root}"
        ),
        color_enabled=color_enabled,
    )
    with batch_summary_path.open("a", encoding="utf-8") as summary_stream:
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
            summary = _read_summary(
                episode.summary_path,
                min_mtime=episode_started_at_epoch,
            )
            success = returncode == 0 and bool(summary and summary.get("success"))
            result = _build_episode_result(
                episode=episode,
                summary=summary,
                success=success,
                elapsed_seconds=episode_elapsed_seconds,
            )
            episode_results.append(result)
            if success:
                successful_episode_dirs.append(episode.output_dir)
            all_success = all_success and success
            completed += 1
            _write_batch_record(
                summary_stream,
                episode=episode,
                returncode=returncode,
                summary=summary,
                result=result,
            )
            status_text = "success" if success else "failed"
            status_color = "green" if success else "red"
            final_progress = _progress_from_summary(
                episode.summary_path,
                min_mtime=episode_started_at_epoch,
            ) or _read_episode_progress(
                episode,
                min_mtime=episode_started_at_epoch,
            )
            print(
                _color(f"[{status_text}] ", status_color, enabled=color_enabled)
                + (
                    f"episode={episode_index} seed={episode.seed} "
                    f"returncode={returncode} "
                    f"elapsed={_format_duration(episode_elapsed_seconds)} "
                    f"summary={episode.summary_path}"
                ),
                _format_progress_suffix(final_progress, color_enabled=color_enabled),
                flush=True,
            )
            if not success and not args.continue_on_failure:
                break
    lerobot_report: dict[str, object] | None = None
    if args.mode == "full_physics":
        lerobot_report = _materialize_batch_lerobot(
            output_root,
            successful_episode_dirs,
        )
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
