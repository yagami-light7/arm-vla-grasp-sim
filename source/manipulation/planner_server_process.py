"""cuRobo 常驻规划服务的进程生命周期管理。"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CuroboPlannerServerProcessConfig:
    """自动启动服务所需的固定参数。"""

    project_root: Path
    host: str = "127.0.0.1"
    port: int = 8765
    startup_timeout_s: float = 180.0
    log_path: Path = Path("/tmp/go2_x5_curobo_planner_server.log")
    python_executable: str = os.environ.get("GO2_X5_CUROBO_PYTHON", sys.executable)


def planner_server_ping(*, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 1.0) -> bool:
    """检查 planner server 是否已经能接受请求。"""

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
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


def planner_server_supports_required_features(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout_s: float = 1.0,
) -> bool:
    """确认常驻 server 支持本分支 full_physics 所需的新规划字段。"""

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(b'{"command":"capabilities"}\n')
            response_line = sock.makefile("r", encoding="utf-8").readline()
    except OSError:
        return False
    if not response_line:
        return False
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError:
        return False
    features = response.get("features")
    return bool(
        response.get("ok", False)
        and isinstance(features, dict)
        and features.get("side_grasp_retreat_to_pregrasp") is True
    )


def _request_shutdown(*, host: str, port: int, timeout_s: float = 2.0) -> bool:
    """请求关闭服务；只由拥有该子进程的 manager 调用。"""

    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(b'{"command":"shutdown"}\n')
            response_line = sock.makefile("r", encoding="utf-8").readline()
    except OSError:
        return False
    if not response_line:
        return False
    try:
        return bool(json.loads(response_line).get("ok", False))
    except json.JSONDecodeError:
        return False


class CuroboPlannerServerProcess:
    """复用已有服务，或启动并清理本次 pipeline 创建的服务。"""

    def __init__(self, config: CuroboPlannerServerProcessConfig):
        self.config = config
        self.process: subprocess.Popen[Any] | None = None
        self.reused_existing = False
        self.start_report: dict[str, Any] = {"requested": False}

    def start(self) -> None:
        """尽早启动子进程，让 cuRobo 初始化与 Isaac App 加载并行。"""

        self.start_report = {"requested": True}
        if planner_server_ping(host=self.config.host, port=self.config.port):
            if planner_server_supports_required_features(
                host=self.config.host,
                port=self.config.port,
            ):
                self.reused_existing = True
                self.start_report.update(
                    {
                        "started": False,
                        "reused_existing": True,
                        "ready": True,
                    }
                )
                print(
                    f"[full-physics] 复用已有 cuRobo planner server "
                    f"{self.config.host}:{self.config.port}",
                    flush=True,
                )
                return
            _request_shutdown(host=self.config.host, port=self.config.port)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not planner_server_ping(host=self.config.host, port=self.config.port, timeout_s=0.2):
                    break
                time.sleep(0.2)
            self.start_report["restarted_incompatible_existing"] = True

        log_path = self.config.log_path.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        server_script = self.config.project_root / "scripts/curobo/grasp_planner_server.py"
        env = os.environ.copy()
        env["GO2_X5_WORKSPACE"] = str(self.config.project_root)
        env.setdefault("GO2_X5_CUROBO_SOURCE_ROOT", "/home/light/workspace/curobo")
        # 常驻服务的默认值必须与 baseline 一致；每次请求仍会显式传入策略。
        env["GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT"] = "1"
        env["GO2_X5_SIDE_GRASP_FALLBACK_RETREAT"] = "0"
        env["GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP"] = "0"
        log_stream = log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [self.config.python_executable, "-B", str(server_script)],
                cwd=str(self.config.project_root),
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_stream.close()
        self.start_report.update(
            {
                "started": True,
                "reused_existing": False,
                "ready": False,
                "pid": self.process.pid,
                "log_path": str(log_path),
            }
        )
        print(
            f"[full-physics] 正在后台启动 cuRobo planner server；log={log_path}",
            flush=True,
        )

    def wait_until_ready(self) -> bool:
        """等待服务就绪；失败时允许后续规划器回退 one-shot。"""

        if self.reused_existing:
            return True
        deadline = time.monotonic() + max(0.0, self.config.startup_timeout_s)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.start_report.update(
                    {
                        "ready": False,
                        "failure_reason": "server_exited_before_ready",
                        "returncode": self.process.returncode,
                    }
                )
                print(
                    "[full-physics] cuRobo planner server 提前退出，将回退 one-shot；"
                    f"log={self.config.log_path}",
                    flush=True,
                )
                return False
            if planner_server_ping(host=self.config.host, port=self.config.port):
                self.start_report["ready"] = True
                print("[full-physics] cuRobo planner server 已就绪", flush=True)
                return True
            time.sleep(0.5)
        self.start_report.update(
            {
                "ready": False,
                "failure_reason": "server_start_timeout",
                "startup_timeout_s": self.config.startup_timeout_s,
            }
        )
        print(
            "[full-physics] cuRobo planner server 启动超时，将回退 one-shot；"
            f"log={self.config.log_path}",
            flush=True,
        )
        return False

    def close(self) -> None:
        """只清理本 manager 创建的进程，不影响用户预先启动的服务。"""

        if self.process is None or self.reused_existing:
            return
        if self.process.poll() is not None:
            return
        _request_shutdown(host=self.config.host, port=self.config.port)
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
