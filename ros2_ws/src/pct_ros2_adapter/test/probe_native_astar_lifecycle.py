#!/usr/bin/env python3
"""验证固定 PCT 原生 A* 的状态清理、GIL 释放和可取消搜索。"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np


EXPECTED_PYTHON = (3, 10)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description="验证 PCT native A* 可安全复用并能被并发取消",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="pct_scan worktree 根目录",
    )
    parser.add_argument(
        "--cancel-grid-size",
        type=int,
        default=1024,
        help="用于并发取消的正方形网格边长",
    )
    return parser.parse_args()


def _load_native_module(project_root: Path):
    planner_lib = (
        project_root.expanduser().resolve()
        / "external/PCT_planner/planner/lib"
    )
    matches = tuple(planner_lib.glob("a_star.cpython-310-*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "需要且只能存在一个 CPython 3.10 a_star 扩展，"
            f"实际找到 {len(matches)} 个"
        )
    sys.path.insert(0, str(planner_lib))
    module = importlib.import_module("a_star")
    loaded = Path(module.__file__).resolve()
    if loaded != matches[0].resolve():
        raise RuntimeError(f"a_star 从错误位置加载：{loaded}")
    required = (
        "SearchStatus",
        "Astar",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"a_star 缺少生命周期接口：{missing}")
    return module


def _new_planner(a_star, cost: np.ndarray):
    planner = a_star.Astar()
    height = np.zeros_like(cost, dtype=np.float64)
    elevation = np.zeros_like(cost, dtype=np.float64)
    planner.init(20.0, 1, 0.2, 0.2, cost, height, elevation)
    for name in (
        "request_cancel",
        "reset_cancellation",
        "was_cancelled",
        "get_last_search_status",
        "get_expanded_node_count",
    ):
        if not callable(getattr(planner, name, None)):
            raise RuntimeError(f"native A* 缺少方法：{name}")
    return planner


def _index(layer: int, column: int, row: int) -> np.ndarray:
    """生成官方 Search 使用的 ``[layer, column, row]`` 索引。"""

    return np.asarray((layer, column, row), dtype=np.int32)


def _assert_status(planner, expected, label: str) -> None:
    actual = planner.get_last_search_status()
    if actual != expected:
        raise AssertionError(f"{label} 状态错误：{actual!r} != {expected!r}")


def _probe_dirty_reset(a_star) -> dict[str, object]:
    cost = np.zeros((3, 3), dtype=np.float64)
    cost[2, 2] = 100.0
    planner = _new_planner(a_star, cost)

    planner.reset_cancellation()
    blocked = planner.search(_index(0, 0, 0), _index(0, 2, 2))
    if blocked:
        raise AssertionError("障碍终点被错误判为可达")
    _assert_status(planner, a_star.SearchStatus.NO_PATH, "无路径")

    planner.reset_cancellation()
    reachable = planner.search(_index(0, 2, 0), _index(0, 0, 0))
    if not reachable:
        raise AssertionError("无路径搜索后的同实例可达规划仍受旧 g 值污染")
    _assert_status(planner, a_star.SearchStatus.SUCCESS, "复用成功")
    result = np.asarray(planner.get_result_matrix(), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != 3:
        raise AssertionError(f"复用成功路径矩阵非法：{result.shape}")
    return {
        "blocked_status": "NO_PATH",
        "reused_success": True,
        "reused_path_points": int(result.shape[0]),
    }


def _probe_sticky_cancel_and_recovery(a_star) -> dict[str, object]:
    planner = _new_planner(a_star, np.zeros((8, 8), dtype=np.float64))
    planner.request_cancel()
    cancelled = planner.search(_index(0, 0, 0), _index(0, 7, 7))
    if cancelled or not planner.was_cancelled():
        raise AssertionError("搜索前的 sticky cancel 没有阻止规划")
    _assert_status(planner, a_star.SearchStatus.CANCELLED, "预取消")
    cancelled_matrix = np.asarray(planner.get_result_matrix())
    if cancelled_matrix.size != 0:
        raise AssertionError("取消后仍暴露半成品路径")

    planner.reset_cancellation()
    recovered = planner.search(_index(0, 0, 0), _index(0, 7, 7))
    if not recovered:
        raise AssertionError("reset_cancellation 后 planner 未恢复")
    _assert_status(planner, a_star.SearchStatus.SUCCESS, "取消后恢复")
    return {
        "sticky_cancelled": True,
        "cancel_result_empty": True,
        "recovered_after_reset": True,
    }


def _probe_gil_and_running_cancel(a_star, grid_size: int) -> dict[str, object]:
    if grid_size < 256:
        raise ValueError("cancel-grid-size 至少为 256")
    cost = np.zeros((grid_size, grid_size), dtype=np.float64)
    cost[grid_size // 2, :] = 100.0
    planner = _new_planner(a_star, cost)
    start = _index(0, 1, 1)
    goal = _index(0, grid_size - 2, grid_size - 2)
    planner.reset_cancellation()

    started = threading.Event()
    outcome: dict[str, object] = {}

    def _search() -> None:
        started.set()
        try:
            outcome["result"] = bool(planner.search(start, goal))
        except BaseException as exc:  # 探针要把线程异常原样带回主线程。
            outcome["exception"] = exc

    worker = threading.Thread(target=_search, name="pct-native-astar-probe")
    worker.start()
    if not started.wait(timeout=1.0):
        raise AssertionError("native 搜索线程没有启动")

    heartbeat = 0
    deadline = time.monotonic() + 5.0
    expanded_before_cancel = 0
    while time.monotonic() < deadline:
        heartbeat += 1
        expanded_before_cancel = int(planner.get_expanded_node_count())
        if expanded_before_cancel >= 32:
            planner.request_cancel()
            break
        if not worker.is_alive():
            break
        time.sleep(0)
    else:
        planner.request_cancel()
        raise AssertionError("等待 native A* 展开节点超时")

    worker.join(timeout=2.0)
    if worker.is_alive():
        planner.request_cancel()
        raise AssertionError("request_cancel 后 native A* 未在 2 秒内退出")
    if "exception" in outcome:
        raise outcome["exception"]
    if expanded_before_cancel < 32:
        raise AssertionError(
            "主线程未能在搜索期间读取 expansion counter；"
            "pybind 可能仍持有 GIL，或测试网格过小"
        )
    if bool(outcome.get("result")):
        raise AssertionError("被取消的不可达搜索错误返回成功")
    if heartbeat < 2:
        raise AssertionError("native 搜索期间 Python heartbeat 没有推进")
    if not planner.was_cancelled():
        raise AssertionError("运行中取消没有报告 CANCELLED")
    _assert_status(planner, a_star.SearchStatus.CANCELLED, "运行中取消")
    if np.asarray(planner.get_result_matrix()).size != 0:
        raise AssertionError("运行中取消后仍暴露路径")
    return {
        "grid_size": grid_size,
        "heartbeat_count": heartbeat,
        "expanded_before_cancel": expanded_before_cancel,
        "cancelled_within_sec": 2.0,
        "gil_released": True,
    }


def _run(project_root: Path, grid_size: int) -> dict[str, object]:
    a_star = _load_native_module(project_root)
    started = time.perf_counter()
    return {
        "result": "PASS",
        "module": str(Path(a_star.__file__).resolve()),
        "dirty_reset": _probe_dirty_reset(a_star),
        "sticky_cancel": _probe_sticky_cancel_and_recovery(a_star),
        "running_cancel": _probe_gil_and_running_cancel(a_star, grid_size),
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> int:
    """运行原生 A* 生命周期探针并输出机器可读报告。"""

    if sys.version_info[:2] != EXPECTED_PYTHON:
        print(
            json.dumps(
                {
                    "result": "BLOCKED",
                    "reason": "请使用 /usr/bin/python3 的 CPython 3.10 ABI",
                    "actual_python": sys.version,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    args = _parse_args()
    try:
        report = _run(args.project_root, args.cancel_grid_size)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "result": "FAIL",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
