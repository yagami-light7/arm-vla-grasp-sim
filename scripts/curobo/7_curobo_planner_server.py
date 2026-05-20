#!/usr/bin/env python3
"""
Go2-X5 cuRobo 常驻规划服务。

用途：
    启动一次外部 Python 进程，常驻持有 cuRobo MotionPlanner。
    Isaac Sim 内的一键任务脚本通过 localhost TCP JSON line 协议发送规划请求，
    server 复用同一个 planner 生成 /tmp/go2_x5_grasp_plan.json。

为什么需要它：
    Isaac Sim 进程内的 omni.warp 与当前 cuRobo 依赖存在冲突，因此 planner
    仍然必须在外部 Python 进程中运行。常驻 server 可以避免每次任务重复：
        - 启动 Python
        - import torch / cuRobo
        - 创建 MotionPlanner
        - 触发第一次 CUDA kernel 初始化

协议：
    TCP 每行一个 JSON 请求，每行一个 JSON 响应。
    默认监听 127.0.0.1:8765。

请求示例：
    {"command": "plan_grasp_segments"}
    {"command": "ping"}
    {"command": "shutdown"}
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import socketserver
import sys
import time
import traceback
from pathlib import Path


WORKSPACE = Path("/home/light/workspace/arm_vla")
PLAN_MODULE_PATH = WORKSPACE / "scripts/curobo/6_plan_grasp_segments.py"

DEFAULT_STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
DEFAULT_TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")
DEFAULT_OUTPUT_JSON = Path("/tmp/go2_x5_grasp_plan.json")
READY_JSON = Path("/tmp/go2_x5_curobo_planner_server.ready.json")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def log(*args) -> None:
    """server 日志写 stderr，stdout 保持为 JSON line 响应通道。"""
    print(*args, file=sys.stderr, flush=True)


def load_plan_module():
    """按路径加载 6_plan_grasp_segments.py，因为文件名以数字开头不能普通 import。"""
    if not PLAN_MODULE_PATH.exists():
        raise RuntimeError(f"找不到 planner module: {PLAN_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "go2_x5_grasp_segments_module",
        PLAN_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 planner module: {PLAN_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["go2_x5_grasp_segments_module"] = module
    spec.loader.exec_module(module)
    return module


class CuroboPlannerServer:
    """持有一个常驻 MotionPlanner，并处理多次规划请求。"""

    def __init__(self):
        self.module = load_plan_module()
        self.planner = None
        self.started_at = time.time()
        self.num_requests = 0

    def start(self) -> None:
        """创建一次 MotionPlanner。"""
        log("========== Go2-X5 cuRobo Planner Server ==========")
        log("[server] workspace:", WORKSPACE)
        log("[server] python:", sys.executable)
        log("[server] loading planner once...")

        self.module.PROFILER = self.module.Profiler()
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            self.planner = self.module.create_planner()
        startup_s = time.perf_counter() - start

        READY_JSON.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ready": True,
                    "pid": os.getpid(),
                    "startup_s": startup_s,
                    "created_at": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log(f"[server] ready, planner_startup_s={startup_s:.3f}")

    def plan_grasp_segments(self, request: dict) -> dict:
        """执行一次抓取分段规划，复用常驻 planner。"""
        self.num_requests += 1

        state_json = Path(request.get("state_json", DEFAULT_STATE_JSON))
        target_json = Path(request.get("target_json", DEFAULT_TARGET_JSON))
        output_json = Path(request.get("output_json", DEFAULT_OUTPUT_JSON))

        module = self.module
        module.STATE_JSON = state_json
        module.TARGET_JSON = target_json
        module.OUTPUT_JSON = output_json
        module.PROFILER = module.Profiler()

        log("")
        log("========== Planner Server Request ==========")
        log("[server] request_id:", self.num_requests)
        log("[server] state_json:", state_json)
        log("[server] target_json:", target_json)
        log("[server] output_json:", output_json)

        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            payload = module.plan_grasp_segments(
                planner=self.planner,
                destroy_planner=False,
            )
        wall_time_s = time.perf_counter() - start

        response = {
            "ok": True,
            "command": "plan_grasp_segments",
            "request_id": self.num_requests,
            "output_json": str(output_json),
            "wall_time_s": wall_time_s,
            "summary": payload.get("summary", {}),
            "profile": module.PROFILER.summary(),
        }

        log(f"[server] request done, wall_time_s={wall_time_s:.3f}")
        return response

    def handle(self, request: dict) -> dict:
        """处理一个 JSON 请求。"""
        command = request.get("command", "plan_grasp_segments")

        if command == "ping":
            return {
                "ok": True,
                "command": "ping",
                "uptime_s": time.time() - self.started_at,
                "num_requests": self.num_requests,
            }

        if command == "plan_grasp_segments":
            return self.plan_grasp_segments(request)

        if command == "shutdown":
            return {
                "ok": True,
                "command": "shutdown",
                "num_requests": self.num_requests,
            }

        raise RuntimeError(f"未知 planner server command: {command}")

    def close(self) -> None:
        """销毁 planner，并删除 ready 标记。"""
        if self.planner is not None:
            log("[server] destroying planner...")
            with contextlib.redirect_stdout(sys.stderr):
                self.planner.destroy()
            self.planner = None

        if READY_JSON.exists():
            READY_JSON.unlink()


class PlannerTCPHandler(socketserver.StreamRequestHandler):
    """处理一个 TCP 连接中的 JSON line 请求。"""

    def handle(self) -> None:
        for raw_line in self.rfile:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.server.planner_server.handle(request)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

            response_line = json.dumps(response, ensure_ascii=False) + "\n"
            self.wfile.write(response_line.encode("utf-8"))
            self.wfile.flush()

            if response.get("ok") and response.get("command") == "shutdown":
                self.server.shutdown_requested = True
                return


class PlannerTCPServer(socketserver.TCPServer):
    """带 planner_server 引用的 TCP server。"""

    allow_reuse_address = True

    def __init__(self, server_address, handler_class, planner_server):
        super().__init__(server_address, handler_class)
        self.planner_server = planner_server
        self.shutdown_requested = False


def parse_args() -> argparse.Namespace:
    """解析 server 启动参数。"""
    parser = argparse.ArgumentParser(description="Go2-X5 cuRobo 常驻规划服务")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    planner_server = CuroboPlannerServer()
    planner_server.start()

    tcp_server = PlannerTCPServer(
        (args.host, args.port),
        PlannerTCPHandler,
        planner_server,
    )

    log(f"[server] listening on {args.host}:{args.port}")

    try:
        while not tcp_server.shutdown_requested:
            tcp_server.handle_request()
    finally:
        tcp_server.server_close()
        planner_server.close()
        log("[server] closed")


if __name__ == "__main__":
    main()
