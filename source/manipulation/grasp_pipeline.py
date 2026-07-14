"""cuRobo planner-only wrapper for the full-physics pipeline.

The old video-baseline wrapper used to call Isaac Script Editor export/target/
execute scripts.  The current full-physics path owns IsaacLab stepping itself;
this module only talks to the persistent cuRobo planner server or falls back to
``scripts/curobo/03_plan_grasp_trajectory.py`` for one-shot planning.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_PREFLIGHT_GRASP_XY_RADIUS_M = 0.68
MAX_PREFLIGHT_PREGRASP_RADIUS_3D_M = 0.75


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量；仅用于规划器兼容旧环境开关。"""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    """读取 float 环境变量；非法值按默认值处理。"""

    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class GraspTask:
    """cuRobo planning input/output JSON for one pick or place request."""

    object_prim_path: str | None
    curobo_task_mode: str = "grasp"
    # 保留字段兼容调用方；current-state planner 已直接生成 target_json。
    grasp_mode: str = "auto"
    use_planner_server: bool = True
    state_json: str = "/tmp/go2_x5_isaac_state.json"
    target_json: str = "/tmp/go2_x5_target_tcp_pose.json"
    plan_json: str = "/tmp/go2_x5_grasp_plan.json"
    # 保留字段兼容旧构造代码；planner-only wrapper 不再执行 Isaac 动作。
    result_json: str = "/tmp/go2_x5_grasp_sequence_result.json"


@dataclass(frozen=True)
class GraspPipelineConfig:
    """Runtime paths and cuRobo planner-server settings."""

    workspace: Path = PROJECT_ROOT
    curobo_python: str = os.environ.get(
        "GO2_X5_CUROBO_PYTHON",
        sys.executable,
    )
    curobo_source_root: str = os.environ.get(
        "GO2_X5_CUROBO_SOURCE_ROOT",
        str(PROJECT_ROOT / "external/curobo"),
    )
    planner_host: str = "127.0.0.1"
    planner_port: int = 8765
    planner_timeout_s: float = 30.0
    one_shot_timeout_s: float = _env_float("GO2_X5_CUROBO_PLAN_TIMEOUT_S", 300.0)
    side_grasp_plan_vertical_lift: bool = _env_bool(
        "GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT",
        True,
    )
    side_grasp_fallback_retreat: bool = _env_bool(
        "GO2_X5_SIDE_GRASP_FALLBACK_RETREAT",
        False,
    )
    side_grasp_retreat_to_pregrasp: bool = _env_bool(
        "GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP",
        False,
    )
    split_pregrasp_motion: bool = _env_bool(
        "GO2_X5_SPLIT_PREGRASP_MOTION",
        False,
    )


class GraspPipeline:
    """Plan pick/place arm segments through cuRobo.

    This class deliberately does not export Isaac state, generate targets, or
    execute trajectories. Those responsibilities live in IsaacLab runtime,
    ``CurrentStateCuroboPlanner`` and ``SegmentedArmExecutor``.
    """

    def __init__(
        self,
        config: GraspPipelineConfig | None = None,
        *,
        recorder: Any | None = None,
    ):
        self.config = config or GraspPipelineConfig()
        # recorder is accepted only to avoid breaking stale construction sites.
        self.recorder = recorder
        self.script_plan = self.config.workspace / "scripts/curobo/03_plan_grasp_trajectory.py"

    def plan(self, task: GraspTask) -> dict[str, Any]:
        """Plan arm-only pick/place segments with the external cuRobo runtime."""

        Path(task.plan_json).unlink(missing_ok=True)
        task_mode = (task.curobo_task_mode or "grasp").strip().lower()
        if task_mode == "grasp":
            # 在启动耗时规划前拦截错误的 nav-to-pick 交接位姿。
            self._validate_target_workspace(self._read_json(task.target_json))
        if task.use_planner_server and self._try_server(task):
            return self._read_json(task.plan_json)
        return self._run_one_shot_planner(task, task_mode=task_mode or "grasp")

    def _run_one_shot_planner(self, task: GraspTask, *, task_mode: str) -> dict[str, Any]:
        if not self.script_plan.exists():
            raise FileNotFoundError(self.script_plan)
        env = os.environ.copy()
        env.update(
            {
                "GO2_X5_WORKSPACE": str(self.config.workspace),
                "GO2_X5_CUROBO_SOURCE_ROOT": self.config.curobo_source_root,
                "GO2_X5_CUROBO_TASK_MODE": task_mode,
                "GO2_X5_STATE_JSON": task.state_json,
                "GO2_X5_TARGET_JSON": task.target_json,
                "GO2_X5_PLAN_JSON": task.plan_json,
                "GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS": os.environ.get("GO2_X5_REQUIRE_OBJECT_LIFT_SUCCESS", "1"),
                "GO2_X5_SIDE_GRASP_PLAN_VERTICAL_LIFT": (
                    "1" if self.config.side_grasp_plan_vertical_lift else "0"
                ),
                "GO2_X5_SIDE_GRASP_FALLBACK_RETREAT": (
                    "1" if self.config.side_grasp_fallback_retreat else "0"
                ),
                "GO2_X5_SIDE_GRASP_RETREAT_TO_PREGRASP": (
                    "1" if self.config.side_grasp_retreat_to_pregrasp else "0"
                ),
                "GO2_X5_SPLIT_PREGRASP_MOTION": (
                    "1" if self.config.split_pregrasp_motion else "0"
                ),
            }
        )
        print(
            "[grasp] one-shot cuRobo planner:",
            {
                "state_json": task.state_json,
                "target_json": task.target_json,
                "plan_json": task.plan_json,
                "task_mode": task_mode,
                "timeout_s": float(self.config.one_shot_timeout_s),
            },
            flush=True,
        )
        try:
            result = subprocess.run(
                [self.config.curobo_python, str(self.script_plan)],
                cwd=str(self.config.workspace),
                env=env,
                text=True,
                capture_output=True,
                timeout=(
                    max(1.0, float(self.config.one_shot_timeout_s))
                    if self.config.one_shot_timeout_s > 0.0
                    else None
                ),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                "cuRobo one-shot planner timed out "
                f"after {float(self.config.one_shot_timeout_s):.1f}s; "
                f"stdout_tail={str(stdout)[-2000:]!r}; "
                f"stderr_tail={str(stderr)[-2000:]!r}"
            ) from exc
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"cuRobo one-shot planner failed with return code {result.returncode}")
        return self._read_json(task.plan_json)

    def _try_server(self, task: GraspTask) -> bool:
        request = {
            "command": (
                "plan_place_segments"
                if (task.curobo_task_mode or "grasp").strip().lower() == "place"
                else "plan_grasp_segments"
            ),
            "state_json": task.state_json,
            "target_json": task.target_json,
            "output_json": task.plan_json,
            "side_grasp_plan_vertical_lift": self.config.side_grasp_plan_vertical_lift,
            "side_grasp_fallback_retreat": self.config.side_grasp_fallback_retreat,
            "side_grasp_retreat_to_pregrasp": self.config.side_grasp_retreat_to_pregrasp,
            "split_pregrasp_motion": self.config.split_pregrasp_motion,
        }
        try:
            with socket.create_connection((self.config.planner_host, self.config.planner_port), timeout=1.0) as sock:
                sock.settimeout(self.config.planner_timeout_s)
                sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                response_line = sock.makefile("r", encoding="utf-8").readline()
        except OSError as exc:
            print("[grasp] planner server unavailable, using one-shot planner:", exc)
            return False
        if not response_line:
            return False
        response = json.loads(response_line)
        if not response.get("ok", False):
            print(
                "[grasp] planner server returned planning failure; "
                "running one-shot planner for diagnostics:",
                response.get("error"),
            )
            return False
        return Path(task.plan_json).exists()

    @staticmethod
    def _validate_target_workspace(target: dict[str, Any]) -> None:
        """在调用 cuRobo 前拒绝明显错误的 nav-to-pick 交接位姿。"""

        workspace = target.get("diagnostics", {}).get("target_workspace_base", {})
        grasp = workspace.get("grasp", {})
        pregrasp = workspace.get("pregrasp", {})
        grasp_xy = float(grasp.get("xy_radius_m", 0.0))
        pregrasp_radius = float(pregrasp.get("radius_3d_m", 0.0))
        if grasp_xy > MAX_PREFLIGHT_GRASP_XY_RADIUS_M or pregrasp_radius > MAX_PREFLIGHT_PREGRASP_RADIUS_3D_M:
            raise RuntimeError(
                "grasp_target_unreachable: 导航交接位姿超出机械臂工作空间: "
                f"grasp_xy_radius={grasp_xy:.3f} m "
                f"(limit={MAX_PREFLIGHT_GRASP_XY_RADIUS_M:.3f} m), "
                f"pregrasp_radius_3d={pregrasp_radius:.3f} m "
                f"(limit={MAX_PREFLIGHT_PREGRASP_RADIUS_3D_M:.3f} m)"
            )

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
