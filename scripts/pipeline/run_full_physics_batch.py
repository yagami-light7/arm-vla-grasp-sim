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
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ENTRY = PROJECT_ROOT / "scripts/pipeline/run_full_physics_pipeline.py"


_REAL_MODES = {
    "simulation_smoke": "--simulation-smoke",
    "navigation_smoke": "--navigation-smoke",
    "navigation_carry_smoke": "--navigation-carry-smoke",
    "manipulation_apply_smoke": "--manipulation-apply-smoke",
    "full_physics": "--full-physics",
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
    "reset": "\033[0m",
}


_STATE_COLORS = {
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
        "--python",
        default=sys.executable,
        help="用于启动单 episode pipeline 的 Python 解释器路径。",
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
        "--viewport-camera-prim",
        default="/World/Camera_main",
        help=(
            "转发给单 episode pipeline 的 GUI viewpoint camera prim；"
            "当前场景 Camera1/2/3 可传 /World/Camera1。"
        ),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="批量运行默认 headless；需要 GUI 时使用 --no-headless。",
    )
    parser.add_argument(
        "--keep-window-open",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="单 episode GUI 调试时，在子 pipeline 结束后保持窗口；必须配合 --no-headless。",
    )
    parser.add_argument(
        "--continue-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="单个 episode 失败后是否继续后续 episode；默认继续。",
    )
    parser.add_argument(
        "--auto-start-curobo-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转发给单 episode pipeline：自动启动/复用 cuRobo planner server。",
    )
    parser.add_argument(
        "--lock-base-during-manipulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转发给单 episode pipeline：机械臂执行时锁定底盘 root pose。",
    )
    parser.add_argument(
        "--lock-support-joints-during-manipulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转发给单 episode pipeline：机械臂执行时冻结四足支撑关节。",
    )
    parser.add_argument(
        "--replan-pick-from-current-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保留兼容；full-physics 模式始终基于当前状态在线规划 pick/place。",
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
        default=15.0,
        help="子进程长时间无输出时，batch 心跳进度的打印间隔秒数。",
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 ANSI 颜色打印 batch 进度；默认开启，可用 --no-color 关闭。",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_const", const="dry_run", dest="mode")
    for mode_name, flag in _REAL_MODES.items():
        mode_group.add_argument(flag, action="store_const", const=mode_name, dest="mode")
    mode_group.add_argument(
        "--integrated-apply-smoke",
        action="store_const",
        const="integrated_apply_smoke",
        dest="mode",
        help="已取消；请改用 --full-physics。",
    )
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


def _last_json_line(path: Path, *, max_bytes: int = 262_144) -> dict[str, object] | None:
    """只读取文件尾部来解析最后一条 JSONL，避免 heartbeat 扫描完整 frames。"""

    if not path.is_file():
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


def _progress_from_summary(path: Path) -> EpisodeProgress | None:
    summary = _read_summary(path)
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


def _read_episode_progress(episode: BatchEpisodeCommand) -> EpisodeProgress:
    frames_path = episode.summary_path.parent / "frames.jsonl"
    frame = _last_json_line(frames_path)
    if frame is not None:
        state = frame.get("pipeline_state")
        step_index = frame.get("step_index")
        return EpisodeProgress(
            state=state if isinstance(state, str) else None,
            step_index=int(step_index) if isinstance(step_index, int) else None,
            source="frames",
        )
    summary_progress = _progress_from_summary(episode.summary_path)
    if summary_progress is not None:
        return summary_progress
    return EpisodeProgress()


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


def _build_child_command(
    args: argparse.Namespace,
    *,
    episode_index: int,
) -> BatchEpisodeCommand:
    """构造单 episode 子进程命令；真实仿真仍由原入口维护单 World 生命周期。"""

    output_root = _project_path(args.output_dir)
    episode_output_dir = output_root / f"episode_{episode_index:06d}"
    episode_seed = int(args.seed) + int(episode_index)
    if args.mode == "integrated_apply_smoke":
        raise SystemExit("--integrated-apply-smoke 已取消，请改用 --full-physics。")
    mode_flag = "--dry-run" if args.mode == "dry_run" else _REAL_MODES[args.mode]
    command = [
        str(Path(args.python).expanduser()),
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
        mode_flag,
        _bool_flag(args.randomize_task, "--randomize-task", "--no-randomize-task"),
        _bool_flag(args.headless, "--headless", "--no-headless"),
        "--viewport-camera-prim",
        str(args.viewport_camera_prim),
        _bool_flag(args.keep_window_open, "--keep-window-open", "--no-keep-window-open"),
        _bool_flag(
            args.auto_start_curobo_server,
            "--auto-start-curobo-server",
            "--no-auto-start-curobo-server",
        ),
        _bool_flag(
            args.lock_base_during_manipulation,
            "--lock-base-during-manipulation",
            "--no-lock-base-during-manipulation",
        ),
        _bool_flag(
            args.lock_support_joints_during_manipulation,
            "--lock-support-joints-during-manipulation",
            "--no-lock-support-joints-during-manipulation",
        ),
        _bool_flag(
            args.replan_pick_from_current_state,
            "--replan-pick-from-current-state",
            "--no-replan-pick-from-current-state",
        ),
    ]
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
        # 单 episode 入口内部仍按 episode_000000 存 summary。
        summary_path=episode_output_dir / "episode_000000/summary.json",
        command=command,
    )


def _read_summary(path: Path) -> dict[str, object] | None:
    if not path.is_file():
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

    started_at = time.monotonic()
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
    last_child_output_at = started_at
    select_timeout_s = min(0.5, progress_interval_s * 0.5)
    try:
        while True:
            for key, _ in selector.select(timeout=select_timeout_s):
                line = key.fileobj.readline()
                if line:
                    last_child_output_at = time.monotonic()
                    print(line, end="", flush=True)
            returncode = process.poll()
            now = time.monotonic()
            quiet_elapsed = now - max(last_progress_at, last_child_output_at)
            if quiet_elapsed >= progress_interval_s:
                progress = _read_episode_progress(episode)
                print(
                    _color("[progress] ", "yellow", enabled=color_enabled)
                    + (
                        f"episode={episode.episode_index} seed={episode.seed} "
                        f"running elapsed={_format_duration(now - started_at)}"
                    ),
                    _format_progress_suffix(progress, color_enabled=color_enabled),
                    flush=True,
                )
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
    }
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be positive.")
    if not args.headless and args.num_episodes > 1:
        raise SystemExit("批量 GUI 运行容易阻塞自动化；多 episode 请使用 --headless。")
    if args.keep_window_open and args.headless:
        raise SystemExit("--keep-window-open 必须与 --no-headless 一起使用。")
    if args.keep_window_open and args.num_episodes != 1:
        raise SystemExit("--keep-window-open 只支持单 episode GUI 调试。")
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
    all_success = True
    completed = 0
    batch_started_at = time.monotonic()
    color_enabled = bool(args.color) and "NO_COLOR" not in env
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
            returncode = _run_child_process(
                episode,
                env=env,
                progress_interval_s=args.progress_interval_s,
                color_enabled=color_enabled,
            )
            summary = _read_summary(episode.summary_path)
            success = returncode == 0 and bool(summary and summary.get("success"))
            all_success = all_success and success
            completed += 1
            _write_batch_record(
                summary_stream,
                episode=episode,
                returncode=returncode,
                summary=summary,
            )
            status_text = "success" if success else "failed"
            status_color = "green" if success else "red"
            final_progress = _progress_from_summary(episode.summary_path) or _read_episode_progress(
                episode
            )
            print(
                _color(f"[{status_text}] ", status_color, enabled=color_enabled)
                + (
                    f"episode={episode_index} seed={episode.seed} "
                    f"returncode={returncode} "
                    f"elapsed={_format_duration(time.monotonic() - batch_started_at)} "
                    f"summary={episode.summary_path}"
                ),
                _format_progress_suffix(final_progress, color_enabled=color_enabled),
                flush=True,
            )
            if not success and not args.continue_on_failure:
                break
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
    print(_color(f"[full-physics-batch] final={all_success}", final_color, enabled=color_enabled))
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
