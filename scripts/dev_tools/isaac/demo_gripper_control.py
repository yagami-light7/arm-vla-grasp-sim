"""
Go2-X5 夹爪开合控制 demo。

用途：
    只验证 arm_joint7 / arm_joint8 是否能在 Isaac Sim 中被控制。
    不做 cuRobo 规划，不移动 arm_joint1~6，不做抓取轨迹。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

预期输出：
    1. 打印完整 DOF order
    2. 找到 arm_joint7 / arm_joint8 的 Isaac DOF index
    3. 执行 open -> close -> open
    4. 输出每一步实际 q_gripper 和误差
    5. 保存 /tmp/go2_x5_gripper_control_result.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


ARTICULATION_ROOT_PATH = "/World/go2_x5/root_joint"

GRIPPER_JOINT_NAMES = [
    "arm_joint7",
    "arm_joint8",
]

# URDF limit 是 [0, 0.044]。
# 先不要打满到 0.044，留一点余量。
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0

STEP_HZ = 60.0
MOVE_DURATION = 1.0
HOLD_DURATION = 0.3

OUTPUT_JSON = Path("/tmp/go2_x5_gripper_control_result.json")


def get_dof_names(robot) -> list[str]:
    """读取 Isaac Sim articulation 的 DOF 顺序。"""
    try:
        return list(robot.dof_names)
    except Exception:
        view = getattr(robot, "_articulation_view", None)
        if view is not None:
            return list(view.dof_names)
    raise RuntimeError("无法读取 robot.dof_names")


def get_joint_indices(dof_names: list[str], joint_names: list[str]) -> list[int]:
    """把关节名映射到 Isaac DOF index。"""
    indices = []
    for name in joint_names:
        if name not in dof_names:
            raise RuntimeError(f"Isaac DOF 中找不到 {name}，完整 DOF: {dof_names}")
        indices.append(dof_names.index(name))
    return indices


def make_gripper_action(q_gripper_target, gripper_indices, q_full_current):
    """
    构造只控制夹爪关节的 ArticulationAction。

    如果当前 Isaac 版本不支持 joint_indices，
    fallback 到完整 q_full 目标，其他关节保持当前值。
    """
    q_gripper_target = np.asarray(q_gripper_target, dtype=float)
    gripper_indices_np = np.asarray(gripper_indices, dtype=np.int32)

    try:
        return ArticulationAction(
            joint_positions=q_gripper_target,
            joint_indices=gripper_indices_np,
        )
    except TypeError:
        q_full_target = np.asarray(q_full_current, dtype=float).copy()
        q_full_target[gripper_indices] = q_gripper_target
        return ArticulationAction(joint_positions=q_full_target)


def smoothstep5(u: float) -> float:
    """五次 S 曲线，让开合动作更平滑。"""
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


async def init_robot():
    """进入 play 状态并初始化 SingleArticulation。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 Isaac stage")

    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(
        prim_path=ARTICULATION_ROOT_PATH,
        name="go2_x5_gripper_test_robot",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {ARTICULATION_ROOT_PATH}")

    return world, robot


async def move_gripper(world, robot, gripper_indices, q_target, label: str):
    """
    平滑移动夹爪到 q_target。

    q_target:
        shape = [2]
        顺序对应 [arm_joint7, arm_joint8]
    """
    q_full_start = np.asarray(robot.get_joint_positions(), dtype=float)
    q_gripper_start = q_full_start[gripper_indices].copy()

    q_target = np.asarray(q_target, dtype=float)
    num_steps = max(2, int(MOVE_DURATION * STEP_HZ))

    print(f"[{label}] start:", q_gripper_start)
    print(f"[{label}] target:", q_target)

    log = {
        "label": label,
        "target": q_target.tolist(),
        "time": [],
        "actual": [],
        "error_norm": [],
    }

    for step in range(num_steps):
        u = step / float(num_steps - 1)
        s = smoothstep5(u)

        q_cmd = (1.0 - s) * q_gripper_start + s * q_target
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float)
        action = make_gripper_action(q_cmd, gripper_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        q_full_actual = np.asarray(robot.get_joint_positions(), dtype=float)
        q_actual = q_full_actual[gripper_indices]
        err = float(np.linalg.norm(q_actual - q_cmd))

        log["time"].append(step / STEP_HZ)
        log["actual"].append(q_actual.tolist())
        log["error_norm"].append(err)

        if step % 15 == 0 or step == num_steps - 1:
            print(f"[{label}] step={step:03d}, q_actual={q_actual}, error={err:.6f}")

    hold_steps = int(HOLD_DURATION * STEP_HZ)
    for _ in range(hold_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float)
        action = make_gripper_action(q_target, gripper_indices, q_full_now)
        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

    q_final = np.asarray(robot.get_joint_positions(), dtype=float)[gripper_indices]
    final_error = float(np.linalg.norm(q_final - q_target))

    print(f"[{label}] final:", q_final)
    print(f"[{label}] final_error:", final_error)

    log["final"] = q_final.tolist()
    log["final_error_norm"] = final_error
    return log


async def main():
    print("========== Go2-X5 Gripper Control Demo ==========")

    world, robot = await init_robot()

    dof_names = get_dof_names(robot)
    gripper_indices = get_joint_indices(dof_names, GRIPPER_JOINT_NAMES)

    print("[DOF] count:", len(dof_names))
    print("[DOF] gripper mapping:")
    for name, index in zip(GRIPPER_JOINT_NAMES, gripper_indices):
        print(f"  - {name:12s}: {index}")

    q_full = np.asarray(robot.get_joint_positions(), dtype=float)
    print("[state] initial q_gripper:", q_full[gripper_indices])

    logs = []

    # 先打开，保证夹爪处于明确状态。
    logs.append(
        await move_gripper(
            world,
            robot,
            gripper_indices,
            q_target=[GRIPPER_OPEN, GRIPPER_OPEN],
            label="open_1",
        )
    )

    # 再闭合。
    logs.append(
        await move_gripper(
            world,
            robot,
            gripper_indices,
            q_target=[GRIPPER_CLOSE, GRIPPER_CLOSE],
            label="close",
        )
    )

    # 最后再次打开，确认可逆。
    logs.append(
        await move_gripper(
            world,
            robot,
            gripper_indices,
            q_target=[GRIPPER_OPEN, GRIPPER_OPEN],
            label="open_2",
        )
    )

    result = {
        "schema_version": 1,
        "script": "scripts/dev_tools/isaac/demo_gripper_control.py",
        "articulation_root_path": ARTICULATION_ROOT_PATH,
        "gripper_joint_names": GRIPPER_JOINT_NAMES,
        "gripper_joint_indices": dict(zip(GRIPPER_JOINT_NAMES, gripper_indices)),
        "gripper_open": GRIPPER_OPEN,
        "gripper_close": GRIPPER_CLOSE,
        "logs": logs,
        "summary": {
            "final_q_gripper": logs[-1]["final"],
            "max_final_error_norm": float(max(item["final_error_norm"] for item in logs)),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[output]", OUTPUT_JSON)
    print("[summary]", result["summary"])
    print("========== complete ==========")


asyncio.ensure_future(main())
