"""PCT 多楼层全局规划器适配器。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import SimulationState

from .planner_adapter import AStarNavPlanner


@dataclass(frozen=True)
class PCTPlannerConfig:
    enabled: bool = False
    planner_root: Path | None = None
    server_script: Path | None = None
    server_python: Path | None = None
    tomogram_name: str = "mutifloor"
    tomogram_path: Path | None = None
    walkable_path: Path | None = None
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 10.0
    coord_mode: str = "sim_to_pct_180deg"
    pct_offset_x: float = 0.0
    pct_offset_y: float = 0.0
    pct_scale_x: float = 1.0
    pct_scale_y: float = 1.0
    fallback_to_astar: bool = True


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def _xyz(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) < 3:
        raise ValueError("expected an xyz sequence with at least three values")
    return (float(values[0]), float(values[1]), float(values[2]))


def _validate_coord_config(
    *,
    coord_mode: str,
    pct_scale_x: float,
    pct_scale_y: float,
) -> None:
    if coord_mode not in {"sim_to_pct_180deg", "identity"}:
        raise ValueError(f"unsupported PCT coord_mode: {coord_mode}")
    if float(pct_scale_x) == 0.0 or float(pct_scale_y) == 0.0:
        raise ValueError("PCT coordinate scales must be non-zero")


def sim_to_pct_xyz(
    xyz: Sequence[float],
    *,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
) -> tuple[float, float, float]:
    """将 Isaac Sim 世界坐标 xyz 转换到 PCT 规划坐标系。"""

    _validate_coord_config(
        coord_mode=coord_mode,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
    )
    sim_x, sim_y, sim_z = _xyz(xyz)
    if coord_mode == "identity":
        return (
            sim_x * float(pct_scale_x) + float(pct_offset_x),
            sim_y * float(pct_scale_y) + float(pct_offset_y),
            sim_z,
        )
    return (
        -sim_x * float(pct_scale_x) + float(pct_offset_x),
        -sim_y * float(pct_scale_y) + float(pct_offset_y),
        sim_z,
    )


def pct_to_sim_xyz(
    xyz: Sequence[float],
    *,
    coord_mode: str = "sim_to_pct_180deg",
    pct_offset_x: float = 0.0,
    pct_offset_y: float = 0.0,
    pct_scale_x: float = 1.0,
    pct_scale_y: float = 1.0,
) -> tuple[float, float, float]:
    """将 PCT 规划坐标 xyz 转回 Isaac Sim 世界坐标系。"""

    _validate_coord_config(
        coord_mode=coord_mode,
        pct_scale_x=pct_scale_x,
        pct_scale_y=pct_scale_y,
    )
    pct_x, pct_y, pct_z = _xyz(xyz)
    if coord_mode == "identity":
        return (
            (pct_x - float(pct_offset_x)) / float(pct_scale_x),
            (pct_y - float(pct_offset_y)) / float(pct_scale_y),
            pct_z,
        )
    return (
        -(pct_x - float(pct_offset_x)) / float(pct_scale_x),
        -(pct_y - float(pct_offset_y)) / float(pct_scale_y),
        pct_z,
    )


class PCTPlannerClient:
    """通过子进程封装 PCT stdin/stdout JSON 规划协议。"""

    def __init__(self, config: PCTPlannerConfig):
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._recent_lines: deque[str] = deque(maxlen=40)

    def plan(
        self,
        *,
        start: Sequence[float],
        end: Sequence[float],
    ) -> dict[str, Any]:
        self.start()
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("PCT server stdin is unavailable")
        request = {"start": list(_xyz(start)), "end": list(_xyz(end))}
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("PCT server pipe closed while sending request") from exc

        deadline = time.monotonic() + float(self.config.request_timeout_s)
        while time.monotonic() < deadline:
            self._raise_if_exited()
            remaining = max(0.01, deadline - time.monotonic())
            try:
                line = self._stdout_queue.get(timeout=min(0.10, remaining))
            except queue.Empty:
                continue
            payload = self._decode_json_line(line)
            if payload is None:
                continue
            if payload.get("status") == "ok" and isinstance(payload.get("traj"), list):
                return payload
            raise RuntimeError(f"PCT planner returned non-ok response: {payload}")
        raise TimeoutError(
            "PCT planner request timed out after "
            f"{float(self.config.request_timeout_s):.1f}s; recent server output: "
            f"{self._recent_output()}"
        )

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        server_python = self._server_python()
        server_script = self._server_script()
        env = self._server_env()
        cwd = self._server_cwd(server_script)
        self._stdout_queue = queue.Queue()
        self._recent_lines.clear()
        self._process = subprocess.Popen(
            [str(server_python), str(server_script)],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process,),
            name="pct-planner-stdout",
            daemon=True,
        )
        self._stdout_thread.start()
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        self._process = None

    def _server_python(self) -> Path:
        path = self.config.server_python or _path_from_env("PCT_SERVER_PYTHON")
        if path is None:
            return Path(sys.executable)
        return Path(path).expanduser()

    def _server_script(self) -> Path:
        if self.config.server_script is not None:
            return Path(self.config.server_script).expanduser()
        root = self.config.planner_root or _path_from_env("PCT_PLANNER_ROOT")
        if root is None:
            raise ValueError(
                "PCT server script is not configured; pass pct_server_script "
                "or use the pipeline factory default scripts/navigation/pct_grid_server.py"
            )
        return Path(root).expanduser() / "scripts/navigation/pct_server.py"

    def _server_env(self) -> dict[str, str]:
        env = dict(os.environ)
        root = self.config.planner_root or _path_from_env("PCT_PLANNER_ROOT")
        tomogram_path = self.config.tomogram_path or _path_from_env("PCT_TOMOGRAM_PATH")
        walkable_path = self.config.walkable_path or _path_from_env("PCT_WALKABLE_PATH")
        if root is not None:
            env["PCT_PLANNER_ROOT"] = str(Path(root).expanduser())
        env["PCT_TOMOGRAM_NAME"] = str(self.config.tomogram_name)
        if tomogram_path is not None:
            env["PCT_TOMOGRAM_PATH"] = str(Path(tomogram_path).expanduser())
        if walkable_path is not None:
            env["PCT_WALKABLE_PATH"] = str(Path(walkable_path).expanduser())
        return env

    def _server_cwd(self, server_script: Path) -> Path | None:
        if self.config.planner_root is not None:
            return Path(self.config.planner_root).expanduser()
        root = _path_from_env("PCT_PLANNER_ROOT")
        if root is not None:
            return root
        return server_script.parent

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + float(self.config.startup_timeout_s)
        while time.monotonic() < deadline:
            self._raise_if_exited()
            try:
                line = self._stdout_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            stripped = line.strip()
            if "READY" in stripped.upper():
                return
            payload = self._decode_json_line(line)
            if payload is not None and payload.get("status") in {"ready", "ok"}:
                return
        raise TimeoutError(
            "PCT server did not report READY within "
            f"{float(self.config.startup_timeout_s):.1f}s; recent server output: "
            f"{self._recent_output()}"
        )

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self._recent_lines.append(line.rstrip("\n"))
            self._stdout_queue.put(line)

    def _decode_json_line(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("PCT server process has not been started")
        self._raise_if_exited()
        return self._process

    def _raise_if_exited(self) -> None:
        process = self._process
        if process is None:
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"PCT server exited with code {returncode}; recent server output: "
                f"{self._recent_output()}"
            )

    def _recent_output(self) -> str:
        if not self._recent_lines:
            return "<none>"
        return " | ".join(self._recent_lines)

    def __enter__(self) -> "PCTPlannerClient":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class PCTNavPlanner:
    """将 PCT server 适配为当前 pipeline 使用的全局导航规划器。"""

    def __init__(
        self,
        config: PCTPlannerConfig,
        *,
        client: PCTPlannerClient | None = None,
        fallback_planner: AStarNavPlanner | None = None,
    ) -> None:
        self.config = config
        self.client = client or PCTPlannerClient(config)
        self.fallback_planner = fallback_planner

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        try:
            return self._plan_with_pct(state, goal)
        except Exception as exc:
            if self.config.fallback_to_astar and self.fallback_planner is not None:
                fallback = self.fallback_planner.plan(state, goal)
                metadata = dict(fallback.metadata)
                metadata["planner"] = "astar_fallback_after_pct_failure"
                metadata["pct_failure_reason"] = str(exc)
                return NavPlan(
                    goal=fallback.goal,
                    waypoints=fallback.waypoints,
                    metadata=metadata,
                )
            raise RuntimeError(f"PCT global planning failed: {exc}") from exc

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _plan_with_pct(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        if not self.config.enabled:
            raise RuntimeError("PCT planner is disabled")
        start_sim = (
            float(state.robot_root_pose[0]),
            float(state.robot_root_pose[1]),
            float(state.robot_root_pose[2]),
        )
        goal_z_missing = goal.z is None
        end_z = start_sim[2] if goal.z is None else float(goal.z)
        end_sim = (float(goal.x), float(goal.y), end_z)
        start_pct = self._sim_to_pct(start_sim)
        end_pct = self._sim_to_pct(end_sim)
        response = self.client.plan(start=start_pct, end=end_pct)
        raw_traj = response.get("traj")
        if not isinstance(raw_traj, list) or len(raw_traj) < 2:
            raise RuntimeError("PCT planner returned fewer than two trajectory points")
        path_3d = tuple(self._pct_to_sim(point) for point in raw_traj)
        waypoints_xy = tuple((float(x), float(y)) for x, y, _z in path_3d)
        metadata = {
            "planner": "pct",
            "path_3d": path_3d,
            "slice_start": _first_present(response, "slice_start", "start_slice", "slice_id_start"),
            "slice_end": _first_present(response, "slice_end", "end_slice", "slice_id_end"),
            "snap_start_dist": _first_present(response, "snap_start_dist", "start_snap_dist"),
            "snap_end_dist": _first_present(response, "snap_end_dist", "end_snap_dist"),
            "goal_z_missing": goal_z_missing,
            "goal_z_source": "robot_root_pose" if goal_z_missing else "goal",
            "coord_mode": self.config.coord_mode,
            "pct_start": start_pct,
            "pct_end": end_pct,
            "pct_status": response.get("status"),
        }
        return NavPlan(goal=goal, waypoints=waypoints_xy, metadata=metadata)

    def _sim_to_pct(self, xyz: Sequence[float]) -> tuple[float, float, float]:
        return sim_to_pct_xyz(
            xyz,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
        )

    def _pct_to_sim(self, xyz: Sequence[float]) -> tuple[float, float, float]:
        return pct_to_sim_xyz(
            xyz,
            coord_mode=self.config.coord_mode,
            pct_offset_x=self.config.pct_offset_x,
            pct_offset_y=self.config.pct_offset_y,
            pct_scale_x=self.config.pct_scale_x,
            pct_scale_y=self.config.pct_scale_y,
        )


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


__all__ = [
    "PCTNavPlanner",
    "PCTPlannerClient",
    "PCTPlannerConfig",
    "pct_to_sim_xyz",
    "sim_to_pct_xyz",
]
