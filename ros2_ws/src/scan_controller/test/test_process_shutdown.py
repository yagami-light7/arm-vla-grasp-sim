#!/usr/bin/env python3
"""重复启动控制器进程并验证 SIGINT 后能够干净退出。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time


STARTUP_MARKER = "SCAN 闭环控制器已启动"
SIGNAL_MARKER = "signal_handler(SIGINT/SIGTERM)"


def _wait_for_startup(
    process: subprocess.Popen[str],
    *,
    timeout_sec: float,
) -> str:
    """读取启动日志，确保信号不是在节点尚未初始化时发出。"""

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    lines: list[str] = []
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    lines.append(remainder)
                raise RuntimeError(
                    "scan_controller 在启动完成前退出："
                    f"return_code={process.returncode}, output={''.join(lines)!r}"
                )
            remaining = max(0.0, deadline - time.monotonic())
            for key, _ in selector.select(timeout=min(0.1, remaining)):
                line = key.fileobj.readline()
                if not line:
                    continue
                lines.append(line)
                if STARTUP_MARKER in line:
                    return "".join(lines)
        raise TimeoutError(
            "等待 scan_controller 启动日志超时："
            f"output={''.join(lines)!r}"
        )
    finally:
        selector.close()


def _stop_process(process: subprocess.Popen[str]) -> None:
    """只回收本测试创建的进程，并为失败路径提供有界兜底。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


def run_shutdown_round(
    executable: Path,
    *,
    environment: dict[str, str],
    round_index: int,
) -> None:
    """执行一轮完整初始化、SIGINT 与进程回收。"""

    process = subprocess.Popen(
        (
            os.fspath(executable),
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-r",
            f"__node:=scan_controller_shutdown_round_{round_index}",
        ),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    startup_output = ""
    try:
        startup_output = _wait_for_startup(process, timeout_sec=5.0)
        process.send_signal(signal.SIGINT)
        remaining_output, _ = process.communicate(timeout=8.0)
    except BaseException:
        _stop_process(process)
        raise

    output = startup_output + remaining_output
    if process.returncode != 0:
        raise RuntimeError(
            "scan_controller 未在 SIGINT 后干净退出："
            f"round={round_index}, return_code={process.returncode}, "
            f"output={output!r}"
        )
    if SIGNAL_MARKER not in output:
        raise RuntimeError(
            "scan_controller 没有记录 rclcpp SIGINT 处理证据："
            f"round={round_index}, output={output!r}"
        )


def main() -> None:
    """在一个隔离 ROS domain 中连续执行十二轮关闭回归。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    arguments = parser.parse_args()
    executable = arguments.executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"scan_controller 可执行文件不存在或不可执行：{executable}")

    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(50 + os.getpid() % 150)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    with tempfile.TemporaryDirectory(
        prefix="scan_controller_shutdown_logs_"
    ) as log_directory:
        environment["ROS_LOG_DIR"] = log_directory
        for round_index in range(1, 13):
            run_shutdown_round(
                executable,
                environment=environment,
                round_index=round_index,
            )


if __name__ == "__main__":
    main()
