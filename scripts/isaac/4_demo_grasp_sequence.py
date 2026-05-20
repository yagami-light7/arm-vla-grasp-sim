"""
Go2-X5 第一版抓取序列执行 demo。

用途：
    在 Isaac Sim 中执行 scripts/curobo/6_plan_grasp_segments.py 生成的抓取计划。

执行流程：
    open_gripper
    move_to_pregrasp
    approach_to_grasp
    close_gripper
    lift_object
    check_success

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

输入：
    /tmp/go2_x5_grasp_plan.json

输出：
    /tmp/go2_x5_grasp_sequence_result.json

说明：
    第一版默认使用真实物理接触夹取物体，不做强制 attach。
    如果物体没有被夹起，优先检查抓取位姿、夹爪闭合量、物体 collision/rigid body 设置。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


GRASP_PLAN_JSON = Path("/tmp/go2_x5_grasp_plan.json")
TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")
STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")
OUTPUT_JSON = Path("/tmp/go2_x5_grasp_sequence_result.json")

ARTICULATION_ROOT_PATH = "/World/go2_x5/root_joint"
DEBUG_ROOT_PATH = "/World/debug_go2_x5_grasp_sequence"

SIM_DT = 1.0 / 50.0
ARM_COMMAND_DT = 0.05
GRIPPER_COMMAND_DT = 0.05
SETTLE_TO_SEGMENT_START_DURATION = 0.10
GRIPPER_MOVE_DURATION = 0.45
GRIPPER_HOLD_DURATION = 0.45
FINAL_HOLD_DURATION = 0.20
PRE_CLOSE_HOLD_DURATION = 0.10
GRIPPER_MIN_CLOSE_PROGRESS_FOR_LIFT = 0.10

TRACK_LOG_EVERY_N_STEPS = 20
DRAW_WAYPOINT_STRIDE = 40

# 每个 motion segment 执行完后，继续保持最终关节目标，等待真实 articulation 追上。
# 这样 close_gripper 不会在 approach_to_grasp 尚未到位时提前执行。
POST_MOTION_CONVERGENCE_TIMEOUT = 1.50
POST_MOTION_JOINT_ERROR_TOL = 0.030
STRICT_POST_MOTION_WAIT_SEGMENTS = {
    "move_to_pregrasp",
    "approach_to_grasp",
}

# 物体中心或 bbox 顶部至少上升这么多，认为第一版 lift 有效果。
OBJECT_LIFT_SUCCESS_THRESHOLD_M = 0.04


def load_grasp_plan() -> dict:
    """读取分段抓取计划。"""
    if not GRASP_PLAN_JSON.exists():
        raise FileNotFoundError(
            f"找不到 {GRASP_PLAN_JSON}。请先在终端运行 scripts/curobo/6_plan_grasp_segments.py"
        )

    data = json.loads(GRASP_PLAN_JSON.read_text(encoding="utf-8"))

    summary = data.get("summary", {})
    if not summary.get("all_motion_segments_success", False):
        raise RuntimeError("grasp plan 中存在未成功的 motion segment，不建议执行。")

    if "segments" not in data:
        raise RuntimeError("grasp plan 缺少 segments 字段。")

    return data


def get_stage():
    """获取当前 Isaac stage。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 Isaac stage。")
    return stage


def get_articulation_root_path() -> str:
    """优先使用 dump state 中解析到的当前机器人 articulation root。"""
    if STATE_JSON.exists():
        data = json.loads(STATE_JSON.read_text(encoding="utf-8"))
        path = data.get("paths", {}).get("articulation_root_path")
        if path:
            return str(path)

    return ARTICULATION_ROOT_PATH


async def init_robot():
    """进入 play 状态并初始化 Go2-X5 articulation。"""
    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    articulation_root_path = get_articulation_root_path()
    print("[robot] articulation root:", articulation_root_path)

    robot = SingleArticulation(
        prim_path=articulation_root_path,
        name="go2_x5_grasp_sequence_robot",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {articulation_root_path}")

    return world, robot


def get_dof_names(robot) -> list[str]:
    """读取 Isaac Sim articulation 的 DOF order。"""
    try:
        return list(robot.dof_names)
    except Exception:
        view = getattr(robot, "_articulation_view", None)
        if view is not None:
            return list(view.dof_names)
    raise RuntimeError("无法读取 robot.dof_names")


def get_joint_indices(dof_names: list[str], joint_names: list[str]) -> list[int]:
    """把 joint names 映射成 Isaac DOF indices。"""
    indices = []
    for name in joint_names:
        if name not in dof_names:
            raise RuntimeError(f"Isaac DOF 中找不到 {name}，完整 DOF: {dof_names}")
        indices.append(dof_names.index(name))
    return indices


def make_partial_action(target_positions, joint_indices, q_full_current):
    """
    构造只控制部分关节的 ArticulationAction。

    当前 Isaac 版本如果支持 joint_indices，就只发送子向量。
    如果不支持，则 fallback 到完整 q_full target，其他关节保持当前值。
    """
    target_positions = np.asarray(target_positions, dtype=float)
    joint_indices_np = np.asarray(joint_indices, dtype=np.int32)

    try:
        return ArticulationAction(
            joint_positions=target_positions,
            joint_indices=joint_indices_np,
        )
    except TypeError:
        q_full_target = np.asarray(q_full_current, dtype=float).copy()
        q_full_target[joint_indices] = target_positions
        return ArticulationAction(joint_positions=q_full_target)


def make_gripper_action_with_optional_arm_hold(
    q_gripper_target,
    gripper_indices,
    q_full_current,
    arm_indices=None,
    q_arm_hold=None,
):
    """
    构造夹爪动作。

    如果提供 q_arm_hold，则夹爪闭合/保持期间同时持续发送 arm_joint1~6
    的 grasp 位姿目标，避免 TCP 在夹爪还没闭合时从目标点漂开。
    """
    return make_partial_action(q_gripper_target, gripper_indices, q_full_current)


def apply_gripper_command_with_arm_hold(
    robot,
    q_gripper_target,
    gripper_indices,
    q_full_current,
    arm_indices=None,
    q_arm_hold=None,
):
    """
    在同一个仿真步中同时更新 arm hold target 和 gripper target。

    注意：
        夹爪单独 joint_indices 命令已经验证可用；完整 q_full 命令在当前
        Isaac stage 中会导致 close_gripper 被忽略。因此这里先发送 arm 的
        partial action，再发送 gripper 的 partial action，避免完整 q_full。
    """
    if arm_indices is not None and q_arm_hold is not None:
        arm_action = make_partial_action(q_arm_hold, arm_indices, q_full_current)
        robot.apply_action(arm_action)

    gripper_action = make_gripper_action_with_optional_arm_hold(
        q_gripper_target,
        gripper_indices,
        q_full_current,
    )
    robot.apply_action(gripper_action)


def smoothstep5(u):
    """五次 S 曲线，起终点速度/加速度为 0。"""
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def compute_close_progress(q_start, q_final, q_target):
    """
    计算夹爪从打开状态向闭合目标实际走过的比例。

    有物体时，夹爪被物体挡住后不会到达 q_target=[0, 0]。这时
    final_error 变大是正常接触结果，不能作为失败条件；更有意义的是
    夹爪是否发生了足够闭合，以及后续物体是否被 lift。
    """
    q_start = np.asarray(q_start, dtype=float)
    q_final = np.asarray(q_final, dtype=float)
    q_target = np.asarray(q_target, dtype=float)

    total_close_distance = float(np.linalg.norm(q_start - q_target))
    actual_close_distance = float(np.linalg.norm(q_start - q_final))

    if total_close_distance < 1.0e-9:
        return 1.0

    return float(np.clip(actual_close_distance / total_close_distance, 0.0, 1.0))


def sample_cubic_hermite(time_from_start, q, qd, t):
    """
    按时间从 q/qd 轨迹中采样目标关节角。

    cuRobo 分段规划脚本输出的是 100 Hz 左右的 q/qd。
    Isaac 执行这里按 60 Hz 仿真步长采样。
    """
    t = float(np.clip(t, time_from_start[0], time_from_start[-1]))

    index = int(np.searchsorted(time_from_start, t, side="right") - 1)
    index = max(0, min(index, len(time_from_start) - 2))

    t0 = float(time_from_start[index])
    t1 = float(time_from_start[index + 1])
    h = t1 - t0

    if h <= 1.0e-9:
        return q[index].copy()

    u = (t - t0) / h

    q0 = q[index]
    q1 = q[index + 1]
    v0 = qd[index]
    v1 = qd[index + 1]

    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    return h00 * q0 + h10 * h * v0 + h01 * q1 + h11 * h * v1


def compute_world_bbox(stage, prim_path: str):
    """读取物体 world bbox，用于抓取前后高度检查。"""
    if not prim_path:
        return None

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[warning] object prim 不存在，无法检查 lift: {prim_path}")
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = bound.ComputeAlignedBox()

    bbox_min = np.array(aligned_box.GetMin(), dtype=float)
    bbox_max = np.array(aligned_box.GetMax(), dtype=float)
    bbox_center = 0.5 * (bbox_min + bbox_max)

    return {
        "min_xyz": bbox_min.tolist(),
        "max_xyz": bbox_max.tolist(),
        "center_xyz": bbox_center.tolist(),
        "top_z": float(bbox_max[2]),
        "center_z": float(bbox_center[2]),
    }


def print_grasp_target_diagnostics(object_bbox):
    """打印计划 TCP 和物体 bbox 的相对关系，辅助判断是不是抓得太高/太偏。"""
    if object_bbox is None:
        return
    if not TARGET_JSON.exists():
        print(f"[diagnostic] target json 不存在，跳过 TCP/bbox 诊断: {TARGET_JSON}")
        return

    target = json.loads(TARGET_JSON.read_text(encoding="utf-8"))
    grasp_pose = target.get("poses", {}).get("grasp", {})
    grasp_world = grasp_pose.get("world", {})
    grasp_pos = grasp_world.get("position_xyz")
    if grasp_pos is None:
        print("[diagnostic] target json 中没有 poses.grasp.world.position_xyz")
        return

    grasp_pos = np.asarray(grasp_pos, dtype=float)
    bbox_center = np.asarray(object_bbox["center_xyz"], dtype=float)
    bbox_top_z = float(object_bbox["top_z"])
    bbox_min = np.asarray(object_bbox["min_xyz"], dtype=float)
    bbox_max = np.asarray(object_bbox["max_xyz"], dtype=float)
    bbox_size = bbox_max - bbox_min

    delta = grasp_pos - bbox_center
    xy_error = float(np.linalg.norm(delta[:2]))
    tcp_depth_below_top = float(bbox_top_z - grasp_pos[2])

    print("[diagnostic] planned grasp TCP world:", grasp_pos)
    print("[diagnostic] object bbox center world:", bbox_center)
    print("[diagnostic] object bbox size:", bbox_size)
    print(
        "[diagnostic] TCP relative to object center: "
        f"delta_xyz={delta}, xy_error={xy_error:.4f}m, "
        f"depth_below_top={tcp_depth_below_top:.4f}m"
    )

    if tcp_depth_below_top < 0.030:
        print("[warning] grasp TCP 偏高，苹果/球体更建议接近 bbox 中心高度。")
    if abs(float(delta[2])) > 0.020:
        print("[warning] grasp TCP 与物体中心高度差超过 2cm，可能夹偏。")
    if xy_error > 0.015:
        print("[warning] grasp TCP 与物体中心 XY 偏差超过 1.5cm，可能夹不到物体。")


def draw_motion_segments(stage, segments):
    """在 Isaac viewport 中画出完整抓取 TCP 路径。"""
    if stage.GetPrimAtPath(DEBUG_ROOT_PATH).IsValid():
        stage.RemovePrim(DEBUG_ROOT_PATH)

    UsdGeom.Xform.Define(stage, DEBUG_ROOT_PATH)

    colors = {
        "move_to_pregrasp": (0.1, 0.7, 1.0),
        "approach_to_grasp": (1.0, 0.7, 0.1),
        "lift_object": (0.1, 1.0, 0.3),
    }

    for segment in segments:
        if segment["type"] != "motion":
            continue

        name = segment["name"]
        tcp_world = np.asarray(segment["trajectory"]["tcp_position_world"], dtype=float)
        color = colors.get(name, (0.8, 0.8, 0.8))

        curve = UsdGeom.BasisCurves.Define(stage, f"{DEBUG_ROOT_PATH}/{name}_path")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(tcp_world)])
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in tcp_world])
        curve.CreateWidthsAttr([0.008])
        UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])

        for index, point in enumerate(tcp_world):
            if index % DRAW_WAYPOINT_STRIDE != 0 and index != len(tcp_world) - 1:
                continue
            sphere = UsdGeom.Sphere.Define(stage, f"{DEBUG_ROOT_PATH}/{name}_wp_{index:04d}")
            sphere.CreateRadiusAttr(0.01)
            UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*point))
            UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])


async def settle_arm_to_start(world, robot, arm_indices, q_start, label: str):
    """执行某段轨迹前，平滑移动到该段起点，避免规划状态和当前仿真状态有小偏差。"""
    q_full_initial = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    q_arm_initial = q_full_initial[arm_indices].copy()
    q_start = np.asarray(q_start, dtype=float)

    start_error = float(np.linalg.norm(q_arm_initial - q_start))
    print(f"[settle:{label}] start_error={start_error:.6f}")

    num_steps = max(2, int(SETTLE_TO_SEGMENT_START_DURATION / SIM_DT))
    for step in range(num_steps):
        u = step / float(num_steps - 1)
        s = smoothstep5(u)
        q_target = (1.0 - s) * q_arm_initial + s * q_start

        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_partial_action(q_target, arm_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()


async def execute_motion_segment(world, robot, arm_indices, segment):
    """执行一个 arm motion segment。"""
    name = segment["name"]
    traj = segment["trajectory"]

    time_from_start = np.asarray(traj["time_from_start"], dtype=float)
    q_traj = np.asarray(traj["q"], dtype=float)
    qd_traj = np.asarray(traj.get("qd", np.zeros_like(q_traj)), dtype=float)

    if q_traj.shape[0] != time_from_start.shape[0]:
        raise RuntimeError(f"{name}: q 和 time_from_start 长度不一致。")

    await settle_arm_to_start(world, robot, arm_indices, q_traj[0], name)

    duration = float(time_from_start[-1])
    num_steps = int(np.ceil(duration / SIM_DT)) + 1
    command_period_steps = max(1, int(round(ARM_COMMAND_DT / SIM_DT)))
    q_target = q_traj[0].copy()

    log = {
        "name": name,
        "type": "motion",
        "sim_dt": SIM_DT,
        "command_dt": ARM_COMMAND_DT,
        "time": [],
        "target_q_arm": [],
        "actual_q_arm": [],
        "joint_error_norm": [],
    }

    print(f"[motion:{name}] duration={duration:.3f}s, sim_steps={num_steps}")

    for step in range(num_steps):
        t = min(step * SIM_DT, duration)
        if step % command_period_steps == 0 or step == num_steps - 1:
            q_target = sample_cubic_hermite(time_from_start, q_traj, qd_traj, t)

        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_partial_action(q_target, arm_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        q_full_actual = np.asarray(robot.get_joint_positions(), dtype=float)
        q_actual = q_full_actual[arm_indices]
        error = float(np.linalg.norm(q_actual - q_target))

        log["time"].append(float(t))
        log["target_q_arm"].append(q_target.tolist())
        log["actual_q_arm"].append(q_actual.tolist())
        log["joint_error_norm"].append(error)

        if step % TRACK_LOG_EVERY_N_STEPS == 0 or step == num_steps - 1:
            print(f"[motion:{name}] t={t:.3f}/{duration:.3f}, joint_error={error:.6f}")

    q_final = q_traj[-1]
    if name in STRICT_POST_MOTION_WAIT_SEGMENTS:
        wait_log = await wait_until_arm_reaches_target(
            world=world,
            robot=robot,
            arm_indices=arm_indices,
            q_target=q_final,
            label=name,
        )
    else:
        wait_log = {
            "skipped": True,
            "reason": "strict post-motion wait is only required before close_gripper",
        }
    log["post_motion_wait"] = wait_log
    log["motion_converged"] = bool(wait_log.get("converged", True))

    return log


async def wait_until_arm_reaches_target(world, robot, arm_indices, q_target, label: str):
    """
    保持某段 motion 的最终关节目标，等待真实关节追上。

    这一步对抓取很重要：
        approach_to_grasp 结束后必须真正到位，才能 close_gripper。
    """
    q_target = np.asarray(q_target, dtype=float)
    max_steps = max(1, int(POST_MOTION_CONVERGENCE_TIMEOUT / SIM_DT))

    log = {
        "timeout_s": POST_MOTION_CONVERGENCE_TIMEOUT,
        "tolerance": POST_MOTION_JOINT_ERROR_TOL,
        "time": [],
        "joint_error_norm": [],
        "converged": False,
    }

    for step in range(max_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_partial_action(q_target, arm_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        q_full_actual = np.asarray(robot.get_joint_positions(), dtype=float)
        q_actual = q_full_actual[arm_indices]
        error = float(np.linalg.norm(q_actual - q_target))

        log["time"].append(step * SIM_DT)
        log["joint_error_norm"].append(error)

        if step % TRACK_LOG_EVERY_N_STEPS == 0:
            print(f"[wait:{label}] step={step:03d}, joint_error={error:.6f}")

        if error <= POST_MOTION_JOINT_ERROR_TOL:
            log["converged"] = True
            print(f"[wait:{label}] converged, joint_error={error:.6f}")
            break

    if not log["converged"]:
        final_error = log["joint_error_norm"][-1] if log["joint_error_norm"] else None
        print(f"[wait:{label}] timeout, final_joint_error={final_error}")

    return log


async def hold_arm_target(world, robot, arm_indices, q_arm_hold, duration_s: float, label: str):
    """在夹爪动作前/中保持机械臂末端位姿。"""
    if arm_indices is None or q_arm_hold is None or duration_s <= 0.0:
        return

    q_arm_hold = np.asarray(q_arm_hold, dtype=float)
    hold_steps = max(1, int(duration_s / SIM_DT))

    for step in range(hold_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_partial_action(q_arm_hold, arm_indices, q_full_now)
        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        if step == 0 or step == hold_steps - 1:
            q_actual = np.asarray(robot.get_joint_positions(), dtype=float)[arm_indices]
            error = float(np.linalg.norm(q_actual - q_arm_hold))
            print(f"[hold_arm:{label}] step={step:03d}, joint_error={error:.6f}")


async def execute_gripper_segment(
    world,
    robot,
    gripper_indices,
    segment,
    arm_indices=None,
    q_arm_hold=None,
):
    """执行一个 gripper segment。"""
    name = segment["name"]
    q_target = np.asarray(segment["target_position"], dtype=float)

    q_full_start = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    q_start = q_full_start[gripper_indices].copy()

    num_steps = max(2, int(GRIPPER_MOVE_DURATION / SIM_DT))
    command_period_steps = max(1, int(round(GRIPPER_COMMAND_DT / SIM_DT)))
    q_cmd = q_start.copy()

    log = {
        "name": name,
        "type": "gripper",
        "target_position": q_target.tolist(),
        "sim_dt": SIM_DT,
        "command_dt": GRIPPER_COMMAND_DT,
        "time": [],
        "actual_q_gripper": [],
        "error_norm": [],
    }

    print(f"[gripper:{name}] start={q_start}, target={q_target}")

    hold_arm_during_gripper = arm_indices is not None and q_arm_hold is not None
    if hold_arm_during_gripper and name == "close_gripper":
        print("[gripper:close_gripper] hold arm at grasp pose before closing")
        await hold_arm_target(
            world,
            robot,
            arm_indices,
            q_arm_hold,
            PRE_CLOSE_HOLD_DURATION,
            "pre_close",
        )

    for step in range(num_steps):
        u = step / float(num_steps - 1)
        if step % command_period_steps == 0 or step == num_steps - 1:
            s = smoothstep5(u)
            q_cmd = (1.0 - s) * q_start + s * q_target

        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        apply_gripper_command_with_arm_hold(
            robot,
            q_cmd,
            gripper_indices,
            q_full_now,
            arm_indices=arm_indices,
            q_arm_hold=q_arm_hold,
        )
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        q_full_actual = np.asarray(robot.get_joint_positions(), dtype=float)
        q_actual = q_full_actual[gripper_indices]
        error = float(np.linalg.norm(q_actual - q_cmd))

        log["time"].append(step * SIM_DT)
        log["actual_q_gripper"].append(q_actual.tolist())
        log["error_norm"].append(error)

        if step % TRACK_LOG_EVERY_N_STEPS == 0 or step == num_steps - 1:
            print(f"[gripper:{name}] step={step:03d}, q_actual={q_actual}, error={error:.6f}")

    hold_steps = int(GRIPPER_HOLD_DURATION / SIM_DT)
    for _ in range(hold_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        apply_gripper_command_with_arm_hold(
            robot,
            q_target,
            gripper_indices,
            q_full_now,
            arm_indices=arm_indices,
            q_arm_hold=q_arm_hold,
        )
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

    q_final = np.asarray(robot.get_joint_positions(), dtype=float)[gripper_indices]
    final_error = float(np.linalg.norm(q_final - q_target))

    log["final_q_gripper"] = q_final.tolist()
    log["final_error_norm"] = final_error
    log["arm_hold_enabled"] = bool(hold_arm_during_gripper)

    if name == "close_gripper":
        close_progress = compute_close_progress(q_start, q_final, q_target)
        log["close_progress"] = close_progress
        log["min_close_progress_for_lift"] = GRIPPER_MIN_CLOSE_PROGRESS_FOR_LIFT
        log["close_success"] = close_progress >= GRIPPER_MIN_CLOSE_PROGRESS_FOR_LIFT
        log["close_success_rule"] = "contact-aware close progress, not final error to zero"
        print(
            "[gripper:close_gripper] close_progress="
            f"{close_progress:.3f}, required={GRIPPER_MIN_CLOSE_PROGRESS_FOR_LIFT:.3f}"
        )

        if not log["close_success"]:
            log["abort_reason"] = (
                "close_gripper 没有发生足够闭合。"
                f" q_start={q_start.tolist()}, q_final={q_final.tolist()}, "
                f"q_target={q_target.tolist()}, close_progress={close_progress:.3f}"
            )
            print("[gripper:close_gripper] close check failed:", log["abort_reason"])

    print(f"[gripper:{name}] final={q_final}, final_error={final_error:.6f}")
    return log


async def hold_final(world, robot, segments, arm_indices):
    """保持最后一个 motion segment 的末端姿态一小段时间。"""
    last_motion = None
    for segment in reversed(segments):
        if segment["type"] == "motion":
            last_motion = segment
            break

    if last_motion is None:
        return

    q_final = np.asarray(last_motion["trajectory"]["q"][-1], dtype=float)
    hold_steps = int(FINAL_HOLD_DURATION / SIM_DT)
    for _ in range(hold_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_partial_action(q_final, arm_indices, q_full_now)
        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()


def summarize_logs(logs):
    """汇总执行误差。"""
    motion_errors = []
    gripper_errors = []

    for item in logs:
        if item["type"] == "motion" and item["joint_error_norm"]:
            motion_errors.extend(item["joint_error_norm"])
        if item["type"] == "gripper" and item["error_norm"]:
            gripper_errors.extend(item["error_norm"])

    summary = {
        "num_executed_segments": len(logs),
        "max_motion_joint_error_norm": float(np.max(motion_errors)) if motion_errors else None,
        "mean_motion_joint_error_norm": float(np.mean(motion_errors)) if motion_errors else None,
        "final_motion_joint_error_norm": float(motion_errors[-1]) if motion_errors else None,
        "max_gripper_error_norm": float(np.max(gripper_errors)) if gripper_errors else None,
        "mean_gripper_error_norm": float(np.mean(gripper_errors)) if gripper_errors else None,
    }
    return summary


async def main():
    print("========== Go2-X5 Grasp Sequence Demo ==========")

    stage = get_stage()
    plan = load_grasp_plan()
    segments = plan["segments"]
    object_path = plan.get("object_prim_path")

    print("[input] grasp plan:", GRASP_PLAN_JSON)
    print("[object]", object_path)
    print("[segments]", [segment["name"] for segment in segments])

    draw_motion_segments(stage, segments)

    object_bbox_before = compute_world_bbox(stage, object_path)
    print("[object] bbox before:", object_bbox_before)
    print_grasp_target_diagnostics(object_bbox_before)

    world, robot = await init_robot()
    await omni.kit.app.get_app().next_update_async()

    dof_names = get_dof_names(robot)
    arm_indices = get_joint_indices(dof_names, plan["joint_names"])

    gripper_joint_names = []
    for segment in segments:
        if segment["type"] == "gripper":
            gripper_joint_names = list(segment["joint_names"])
            break
    if not gripper_joint_names:
        raise RuntimeError("grasp plan 中没有 gripper segment。")
    gripper_indices = get_joint_indices(dof_names, gripper_joint_names)

    print("[mapping] arm:", dict(zip(plan["joint_names"], arm_indices)))
    print("[mapping] gripper:", dict(zip(gripper_joint_names, gripper_indices)))

    logs = []
    executed_segments = []
    last_motion_q_final = None
    abort_reason = None
    for segment in segments:
        if segment["type"] == "motion":
            motion_log = await execute_motion_segment(world, robot, arm_indices, segment)
            logs.append(motion_log)
            executed_segments.append(segment)
            last_motion_q_final = np.asarray(segment["trajectory"]["q"][-1], dtype=float)

            if (
                segment["name"] in STRICT_POST_MOTION_WAIT_SEGMENTS
                and not motion_log.get("motion_converged", False)
            ):
                final_error = None
                wait_log = motion_log.get("post_motion_wait", {})
                if wait_log.get("joint_error_norm"):
                    final_error = wait_log["joint_error_norm"][-1]
                abort_reason = (
                    f"{segment['name']} 未在闭环等待内到位，跳过后续抓取。"
                    f" final_joint_error={final_error}"
                )
                print("[abort]", abort_reason)
                break

        elif segment["type"] == "gripper":
            q_arm_hold = None
            hold_indices = None
            if segment["name"] == "close_gripper" and last_motion_q_final is not None:
                q_arm_hold = last_motion_q_final
                hold_indices = arm_indices

            gripper_log = await execute_gripper_segment(
                world,
                robot,
                gripper_indices,
                segment,
                arm_indices=hold_indices,
                q_arm_hold=q_arm_hold,
            )
            logs.append(gripper_log)
            executed_segments.append(segment)

            if segment["name"] == "close_gripper" and not gripper_log.get("close_success", True):
                abort_reason = gripper_log.get(
                    "abort_reason",
                    "close_gripper 未满足进入 lift_object 的闭合进度条件。",
                )
                print("[abort]", abort_reason)
                break

        else:
            raise RuntimeError(f"未知 segment type: {segment['type']}")

    await hold_final(world, robot, executed_segments, arm_indices)

    object_bbox_after = compute_world_bbox(stage, object_path)
    print("[object] bbox after:", object_bbox_after)

    summary = summarize_logs(logs)

    lift_delta_top_z = None
    lift_delta_center_z = None
    lift_success = None
    if object_bbox_before is not None and object_bbox_after is not None:
        lift_delta_top_z = float(object_bbox_after["top_z"] - object_bbox_before["top_z"])
        lift_delta_center_z = float(object_bbox_after["center_z"] - object_bbox_before["center_z"])
        lift_success = lift_delta_center_z >= OBJECT_LIFT_SUCCESS_THRESHOLD_M

    summary.update(
        {
            "object_lift_delta_top_z_m": lift_delta_top_z,
            "object_lift_delta_center_z_m": lift_delta_center_z,
            "object_lift_success": lift_success,
            "object_lift_success_threshold_m": OBJECT_LIFT_SUCCESS_THRESHOLD_M,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
        }
    )

    result = {
        "schema_version": 1,
        "script": "scripts/isaac/4_demo_grasp_sequence.py",
        "source_plan": str(GRASP_PLAN_JSON),
        "object_prim_path": object_path,
        "arm_joint_names": plan["joint_names"],
        "gripper_joint_names": gripper_joint_names,
        "arm_joint_indices": dict(zip(plan["joint_names"], arm_indices)),
        "gripper_joint_indices": dict(zip(gripper_joint_names, gripper_indices)),
        "object_bbox_before": object_bbox_before,
        "object_bbox_after": object_bbox_after,
        "execution_logs": logs,
        "summary": summary,
        "abort_reason": abort_reason,
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[output]", OUTPUT_JSON)
    print("[summary]", summary)
    print("========== grasp sequence complete ==========")


if __name__ == "__main__":
    asyncio.ensure_future(main())
