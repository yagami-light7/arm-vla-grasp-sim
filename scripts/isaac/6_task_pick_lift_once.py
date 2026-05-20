"""
Go2-X5 一键 pick/lift 任务入口。

用途：
    在 Isaac Sim 中把当前已经验证过的抓取链路串起来：

        1. 导出 Go2-X5 当前状态
        2. 从当前选中的物体生成 grasp target
        3. 调用 cuRobo 规划分段抓取轨迹
        4. 在 Isaac Sim 中执行抓取序列
        5. 汇总任务结果

运行位置：
    Isaac Sim 5.1.0 GUI -> Script Editor

运行前要求：
    1. Isaac stage 中已经有 Go2-X5 robot
    2. 机器狗底盘已经稳定
    3. 已经选中一个要抓取的物体 prim
    4. gripper drive 参数已经设置好
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import omni.kit.app


WORKSPACE = Path("/home/light/workspace/arm_vla")
PYTHON = Path("/data/conda_envs/isaacsim51_3dgs_grasp/bin/python")

STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")
PLAN_JSON = Path("/tmp/go2_x5_grasp_plan.json")
EXEC_RESULT_JSON = Path("/tmp/go2_x5_grasp_sequence_result.json")
TASK_RESULT_JSON = Path("/tmp/go2_x5_task_result.json")

SCRIPT_DUMP_STATE = WORKSPACE / "scripts/isaac/1_dump_go2_x5_state.py"
SCRIPT_GENERATE_TARGET = WORKSPACE / "scripts/isaac/2_generate_sim_grasp_target.py"
SCRIPT_PLAN_SEGMENTS = WORKSPACE / "scripts/curobo/6_plan_grasp_segments.py"
SCRIPT_EXECUTE_GRASP = WORKSPACE / "scripts/isaac/4_demo_grasp_sequence.py"

PLANNER_SERVER_HOST = "127.0.0.1"
PLANNER_SERVER_PORT = 8765
PLANNER_SERVER_TIMEOUT_S = 30.0

# 调试执行抖动/时序时，先禁用常驻 planner，确保每次都使用磁盘上最新的
# scripts/curobo/6_plan_grasp_segments.py 参数。确认稳定后再改回 True。
USE_PLANNER_SERVER = False


def add_workspace_to_sys_path():
    """让一键脚本能找到 workspace 下的模块。"""
    paths = [
        WORKSPACE,
        WORKSPACE / "scripts",
    ]

    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def require_file(path: Path, label: str):
    """检查某一步是否真的生成了预期文件。"""
    if not path.exists():
        raise RuntimeError(f"{label} 不存在: {path}")


def read_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    require_file(path, "JSON 文件")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_target_after_generation():
    """确认本轮确实使用了新版目标生成器。"""
    target = read_json(TARGET_JSON)
    state = read_json(STATE_JSON)

    grasp_mode = target.get("source", {}).get("grasp_mode")
    if grasp_mode != "side":
        raise RuntimeError(
            "target JSON 不是新版侧向抓取格式。"
            f" source.grasp_mode={grasp_mode!r}。"
            "请在 Script Editor 里用 exec(open(...6_task_pick_lift_once.py).read()) "
            "运行磁盘文件，避免继续执行旧的粘贴代码。"
        )

    base_position = np.asarray(
        state.get("poses", {}).get("world_base", {}).get("position_xyz", []),
        dtype=float,
    )
    if base_position.shape != (3,):
        raise RuntimeError("state JSON 缺少 poses.world_base.position_xyz。")
    if np.linalg.norm(base_position) < 1.0e-6:
        raise RuntimeError(
            "state JSON 中 arm_base_link 仍然在 world 原点。"
            "这通常表示没有先正确 dump 当前机器人实例状态。"
        )

    diagnostics = target.get("diagnostics", {}).get("target_workspace_base", {})
    grasp_xy_radius = diagnostics.get("grasp", {}).get("xy_radius_m")
    print("[task] validated target grasp_mode:", grasp_mode)
    print("[task] validated base position:", base_position)
    print("[task] target grasp xy_radius_m:", grasp_xy_radius)


def clear_old_outputs():
    """
    清理上一轮任务的输出，避免某一步失败后误读旧 JSON。
    """
    for path in [
        STATE_JSON,
        TARGET_JSON,
        PLAN_JSON,
        EXEC_RESULT_JSON,
        TASK_RESULT_JSON,
    ]:
        if path.exists():
            path.unlink()


def load_module_from_path(module_name: str, script_path: Path):
    """
    从指定 .py 文件加载模块。

    这里用自定义 module_name，是因为原始文件名以数字开头，
    不能作为普通 Python import 名称。
    这里不用 importlib loader，避免 Isaac 长进程里 .pyc 或 sys.modules
    缓存导致执行到旧脚本。
    """
    if not script_path.exists():
        raise RuntimeError(f"脚本不存在: {script_path}")

    unique_module_name = f"{module_name}_{time.time_ns()}"
    source = script_path.read_text(encoding="utf-8")
    code = compile(source, str(script_path), "exec")
    module = types.ModuleType(unique_module_name)
    module.__file__ = str(script_path)
    module.__package__ = None
    sys.modules[unique_module_name] = module
    exec(code, module.__dict__)
    return module


async def run_isaac_script_main(label: str, module_name: str, script_path: Path):
    """
    加载 Isaac 侧脚本并调用它的 async main()。
    """
    print(f"\n========== Task Step: {label} ==========")
    print("[task] script:", script_path)

    module = load_module_from_path(module_name, script_path)

    if not hasattr(module, "main"):
        raise RuntimeError(f"{script_path} 中没有 main() 函数")

    result = module.main()

    if asyncio.iscoroutine(result):
        await result

    await omni.kit.app.get_app().next_update_async()
    print(f"[task] {label} done")


def run_curobo_planner():
    """
    在外部 Python 进程中运行 cuRobo 规划。

    原因：
        Isaac Sim 内部会加载 omni.warp；
        cuRobo 当前环境需要另一个 warp/cuda 组合。
        所以 cuRobo 规划继续放在独立 Python 子进程里运行。
    """
    print("\n========== Task Step: cuRobo plan grasp segments ==========")

    if USE_PLANNER_SERVER and try_run_curobo_planner_server():
        return

    if not USE_PLANNER_SERVER:
        print("[task] planner server disabled for debugging; use one-shot subprocess.")

    cmd = [
        str(PYTHON),
        str(SCRIPT_PLAN_SEGMENTS),
    ]

    print("[task] command:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print("[cuRobo stdout]")
        print(result.stdout)

    if result.stderr:
        print("[cuRobo stderr]")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"cuRobo planning failed, returncode={result.returncode}")

    require_file(PLAN_JSON, "cuRobo grasp plan")
    print("[task] cuRobo plan done:", PLAN_JSON)


def try_run_curobo_planner_server() -> bool:
    """
    优先使用常驻 cuRobo planner server。

    如果 server 没启动，只返回 False，由 run_curobo_planner() 回退到单次子进程。
    如果 server 已连接但规划失败，则直接抛错，避免误用旧计划。
    """
    request = {
        "command": "plan_grasp_segments",
        "state_json": str(STATE_JSON),
        "target_json": str(TARGET_JSON),
        "output_json": str(PLAN_JSON),
    }

    try:
        with socket.create_connection(
            (PLANNER_SERVER_HOST, PLANNER_SERVER_PORT),
            timeout=1.0,
        ) as sock:
            sock.settimeout(PLANNER_SERVER_TIMEOUT_S)
            sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            with sock.makefile("r", encoding="utf-8") as stream:
                line = stream.readline()

    except OSError as exc:
        print(
            "[task] planner server unavailable, fallback to one-shot subprocess:",
            exc,
        )
        return False

    if not line:
        raise RuntimeError("planner server 没有返回响应。")

    response = json.loads(line)
    if not response.get("ok", False):
        print("[task] planner server 规划失败，回退到单次子进程。")
        print("[task] server error:", response.get("error", "unknown error"))
        traceback_text = response.get("traceback", "")
        if traceback_text:
            print(traceback_text)
        return False

    print("[task] planner server response:")
    print(json.dumps(response, indent=2, ensure_ascii=False))

    require_file(PLAN_JSON, "cuRobo grasp plan")
    print("[task] cuRobo plan done via persistent server:", PLAN_JSON)
    return True


def write_task_summary(start_time_s: float):
    """
    汇总一键任务结果。

    详细轨迹和执行日志仍然在：
        /tmp/go2_x5_grasp_sequence_result.json

    这个 summary 只保存最关键结果，方便后续批量任务读取。
    """
    execution = read_json(EXEC_RESULT_JSON)
    target = read_json(TARGET_JSON)
    plan = read_json(PLAN_JSON)

    execution_summary = execution.get("summary", {})

    summary = {
        "schema_version": 1,
        "task": "pick_lift_once",
        "success": bool(execution_summary.get("object_lift_success", False)),
        "elapsed_wall_time_s": float(time.time() - start_time_s),
        "files": {
            "state_json": str(STATE_JSON),
            "target_json": str(TARGET_JSON),
            "plan_json": str(PLAN_JSON),
            "execution_json": str(EXEC_RESULT_JSON),
        },
        "object": {
            "prim_path": execution.get("object_prim_path"),
            "target_source": target.get("source", {}),
        },
        "planner": {
            "all_motion_segments_success": plan.get("summary", {}).get(
                "all_motion_segments_success"
            ),
            "total_motion_duration_s": plan.get("summary", {}).get(
                "total_motion_duration_s"
            ),
        },
        "execution": {
            "object_lift_success": execution_summary.get("object_lift_success"),
            "aborted": execution_summary.get("aborted"),
            "abort_reason": execution_summary.get("abort_reason"),
            "object_lift_delta_center_z_m": execution_summary.get(
                "object_lift_delta_center_z_m"
            ),
            "object_lift_delta_top_z_m": execution_summary.get(
                "object_lift_delta_top_z_m"
            ),
            "max_motion_joint_error_norm": execution_summary.get(
                "max_motion_joint_error_norm"
            ),
            "mean_motion_joint_error_norm": execution_summary.get(
                "mean_motion_joint_error_norm"
            ),
            "max_gripper_error_norm": execution_summary.get(
                "max_gripper_error_norm"
            ),
        },
    }

    TASK_RESULT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== Task Summary ==========")
    print("[task] output:", TASK_RESULT_JSON)
    print("[task] success:", summary["success"])
    print(
        "[task] lift delta center z:",
        summary["execution"]["object_lift_delta_center_z_m"],
    )


async def main():
    start_time_s = time.time()

    print("========== Go2-X5 One-Click Pick/Lift Task ==========")
    print("[task] workspace:", WORKSPACE)
    print("[task] python:", PYTHON)

    add_workspace_to_sys_path()
    clear_old_outputs()

    # 1. 导出 Isaac 当前机器人状态
    await run_isaac_script_main(
        label="dump Isaac state",
        module_name="task_dump_go2_x5_state",
        script_path=SCRIPT_DUMP_STATE,
    )
    require_file(STATE_JSON, "Isaac state JSON")

    # 2. 从当前选中的物体生成 grasp target
    await run_isaac_script_main(
        label="generate sim grasp target",
        module_name="task_generate_sim_grasp_target",
        script_path=SCRIPT_GENERATE_TARGET,
    )
    require_file(TARGET_JSON, "grasp target JSON")
    validate_target_after_generation()

    # 3. 调用 cuRobo 生成分段抓取轨迹
    run_curobo_planner()
    require_file(PLAN_JSON, "grasp plan JSON")

    # 4. 在 Isaac 中执行抓取序列
    await run_isaac_script_main(
        label="execute grasp sequence",
        module_name="task_execute_grasp_sequence",
        script_path=SCRIPT_EXECUTE_GRASP,
    )
    require_file(EXEC_RESULT_JSON, "grasp execution result JSON")

    # 5. 写一键任务摘要
    write_task_summary(start_time_s)

    print("========== one-click pick/lift complete ==========")


async def guarded_main():
    try:
        await main()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.ensure_future(guarded_main())
