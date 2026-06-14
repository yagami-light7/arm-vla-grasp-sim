"""Callable wrapper around the existing Isaac Sim and external cuRobo scripts.

The Isaac-facing methods are async because they must be executed inside the
Isaac Sim application loop. cuRobo remains in an external process or the
persistent TCP planner server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source.data.episode_recorder import EpisodeRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_PREFLIGHT_GRASP_XY_RADIUS_M = 1.00
MAX_PREFLIGHT_PREGRASP_RADIUS_3D_M = 1.20


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment flag."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    """Read a float environment value with a stable fallback."""

    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


@dataclass(frozen=True)
class GraspTask:
    """Input and output files for one pick attempt."""

    object_prim_path: str | None
    curobo_task_mode: str = "grasp"
    grasp_mode: str = "auto"
    use_planner_server: bool = True
    state_json: str = "/tmp/go2_x5_isaac_state.json"
    target_json: str = "/tmp/go2_x5_target_tcp_pose.json"
    plan_json: str = "/tmp/go2_x5_grasp_plan.json"
    result_json: str = "/tmp/go2_x5_grasp_sequence_result.json"


@dataclass(frozen=True)
class GraspPipelineConfig:
    """Runtime paths and planner-server settings."""

    workspace: Path = PROJECT_ROOT
    curobo_python: str = os.environ.get(
        "GO2_X5_CUROBO_PYTHON",
        "/data/conda_envs/isaacsim51_3dgs_grasp/bin/python",
    )
    curobo_source_root: str = os.environ.get("GO2_X5_CUROBO_SOURCE_ROOT", "/home/light/workspace/curobo")
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


class GraspPipeline:
    """Run state export, target generation, planning, and execution in order."""

    def __init__(
        self,
        config: GraspPipelineConfig | None = None,
        *,
        recorder: EpisodeRecorder | None = None,
    ):
        self.config = config or GraspPipelineConfig()
        self.recorder = recorder
        self.script_export = self.config.workspace / "scripts/isaac/01_export_go2_x5_state.py"
        self.script_target = self.config.workspace / "scripts/isaac/02_generate_grasp_target.py"
        self.script_execute = self.config.workspace / "scripts/isaac/04_execute_grasp_sequence.py"
        self.script_plan = self.config.workspace / "scripts/curobo/03_plan_grasp_trajectory.py"

    async def export_state(self, task: GraspTask) -> dict[str, Any]:
        """Export the current full articulation state and post-navigation base pose."""

        module = self._load_module("go2_x5_export_state", self.script_export)
        module.OUTPUT_JSON_PATH = Path(task.state_json)
        await self._call_async_main(module)
        return self._read_json(task.state_json)

    async def generate_target(self, task: GraspTask) -> dict[str, Any]:
        """Generate bbox-based pregrasp, grasp, and retreat/lift targets."""

        module = self._load_module("go2_x5_generate_target", self.script_target)
        module.STATE_JSON = Path(task.state_json)
        module.OUTPUT_TARGET_JSON = Path(task.target_json)
        module.OBJECT_PRIM_PATH_OVERRIDE = task.object_prim_path
        if task.grasp_mode != "auto":
            module.PREFERRED_GRASP_MODE = task.grasp_mode
        await self._call_async_main(module)
        target = self._read_json(task.target_json)
        self._validate_target_workspace(target)
        return target

    def plan(self, task: GraspTask) -> dict[str, Any]:
        """Plan arm-only grasp segments with the external cuRobo runtime."""

        Path(task.plan_json).unlink(missing_ok=True)
        task_mode = (task.curobo_task_mode or "grasp").strip().lower()
        if task.use_planner_server and self._try_server(task):
            return self._read_json(task.plan_json)
        env = os.environ.copy()
        env.update(
            {
                "GO2_X5_WORKSPACE": str(self.config.workspace),
                "GO2_X5_CUROBO_SOURCE_ROOT": self.config.curobo_source_root,
                "GO2_X5_CUROBO_TASK_MODE": task_mode or "grasp",
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
            }
        )
        print(
            "[grasp] one-shot cuRobo planner:",
            {
                "state_json": task.state_json,
                "target_json": task.target_json,
                "plan_json": task.plan_json,
                "task_mode": task_mode or "grasp",
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

    async def execute(self, task: GraspTask) -> dict[str, Any]:
        """Execute the arm and gripper segments on the full Isaac articulation."""

        module = self._load_module("go2_x5_execute_grasp", self.script_execute)
        module.STATE_JSON = Path(task.state_json)
        module.TARGET_JSON = Path(task.target_json)
        module.GRASP_PLAN_JSON = Path(task.plan_json)
        module.OUTPUT_JSON = Path(task.result_json)
        await self._call_async_main(module)
        result = self._read_json(task.result_json)
        self._record_execution(result)
        return result

    async def run(self, task: GraspTask) -> dict[str, Any]:
        """Run the complete grasp flow and return all intermediate payloads."""

        started_at = time.time()
        state = await self.export_state(task)
        target = await self.generate_target(task)
        plan = self.plan(task)
        execution = await self.execute(task)
        summary = execution.get("summary", {})
        return {
            "success": bool(summary.get("task_success", False)),
            "elapsed_wall_time_s": time.time() - started_at,
            "state": state,
            "target": target,
            "plan": plan,
            "execution": execution,
        }

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
            print("[grasp] planner server failed, using one-shot planner:", response.get("error"))
            return False
        return Path(task.plan_json).exists()

    def _record_execution(self, execution: dict[str, Any]) -> None:
        if self.recorder is None:
            return
        elapsed = 0.0
        last_gripper = ""
        for log in execution.get("execution_logs", []):
            if log.get("type") == "motion":
                for sample_time, target, actual in zip(
                    log.get("time", []),
                    log.get("target_q_arm", []),
                    log.get("actual_q_arm", []),
                ):
                    row = {"timestamp": elapsed + float(sample_time), "gripper": last_gripper}
                    row.update({f"arm_joint{index + 1}": value for index, value in enumerate(actual)})
                    row.update({f"arm_action_joint{index + 1}": value for index, value in enumerate(target)})
                    self.recorder.record("grasp", row)
                elapsed += float(log.get("time", [0.0])[-1] if log.get("time") else 0.0)

            elif log.get("type") == "gripper":
                target = log.get("target_position", [])
                gripper_action = sum(target) / len(target) if target else ""
                for sample_time, actual in zip(log.get("time", []), log.get("actual_q_gripper", [])):
                    last_gripper = sum(actual) / len(actual) if actual else ""
                    self.recorder.record(
                        "grasp",
                        {
                            "timestamp": elapsed + float(sample_time),
                            "gripper": last_gripper,
                            "gripper_action": gripper_action,
                        },
                    )
                elapsed += float(log.get("time", [0.0])[-1] if log.get("time") else 0.0)

    @staticmethod
    def _validate_target_workspace(target: dict[str, Any]) -> None:
        """Reject obviously mismatched nav-to-pick handoffs before invoking cuRobo."""

        workspace = target.get("diagnostics", {}).get("target_workspace_base", {})
        grasp = workspace.get("grasp", {})
        pregrasp = workspace.get("pregrasp", {})
        grasp_xy = float(grasp.get("xy_radius_m", 0.0))
        pregrasp_radius = float(pregrasp.get("radius_3d_m", 0.0))
        if grasp_xy > MAX_PREFLIGHT_GRASP_XY_RADIUS_M or pregrasp_radius > MAX_PREFLIGHT_PREGRASP_RADIUS_3D_M:
            raise RuntimeError(
                "grasp_target_unreachable: navigation base pose is not near the selected object: "
                f"grasp_xy_radius={grasp_xy:.3f} m, pregrasp_radius_3d={pregrasp_radius:.3f} m"
            )

    @staticmethod
    async def _call_async_main(module: types.ModuleType) -> None:
        result = module.main()
        if asyncio.iscoroutine(result):
            await result

    @staticmethod
    def _load_module(module_prefix: str, path: Path) -> types.ModuleType:
        if not path.exists():
            raise FileNotFoundError(path)
        module_name = f"{module_prefix}_{time.time_ns()}"
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        sys.modules[module_name] = module
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        return module

    @staticmethod
    def _read_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the callable grasp pipeline inside an Isaac Sim Python runtime.")
    parser.add_argument("--object-prim", default=None)
    parser.add_argument("--grasp-mode", default="auto", choices=("auto", "side", "top_down"))
    parser.add_argument("--no-planner-server", action="store_true")
    args = parser.parse_args()
    try:
        import omni.usd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Run this module from Isaac Sim Python or Script Editor; omni.usd is unavailable.") from exc
    task = GraspTask(
        object_prim_path=args.object_prim,
        grasp_mode=args.grasp_mode,
        use_planner_server=not args.no_planner_server,
    )
    asyncio.ensure_future(GraspPipeline().run(task))


if __name__ == "__main__":
    main()
