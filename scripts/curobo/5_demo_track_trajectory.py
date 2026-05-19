import asyncio
import json
import numpy as np
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

TRAJ_JSON = "/tmp/go2_x5_arm_plan_to_pose.json"
DEBUG_ROOT = "/World/debug_go2_x5_arm_trajectory"

ROBOT_ROOT_PATH = "/World/go2_x5"
ARTICULATION_ROOT_PATH = "/World/go2_x5/root_joint"
TCP_PRIM_PATH = "/World/go2_x5/arm_link6/grasp_tcp_link"

STEPS_PER_WAYPOINT = 4
START_Q_TOL = 0.05

TRACK_RESULT_JSON = "/tmp/go2_x5_arm_track_result.json"

SETTLE_TO_START_DURATION = 1.0
HOLD_FINAL_DURATION = 0.5
TRACK_LOG_EVERY_N_STEPS = 15

# 读取curobo规划的轨迹数据
def load_trajectory():
    with open(TRAJ_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data["plan_info"]["planner_success"]:
        raise RuntimeError("planner_success=False，不建议执行。")

    traj = data["trajectory"]

    q_traj = np.asarray(traj["q"], dtype=float)
    qd_traj = np.asarray(traj.get("qd", np.zeros_like(q_traj)), dtype=float)
    time_from_start = np.asarray(traj["time_from_start"], dtype=float)
    tcp_world = np.asarray(traj["tcp_position_world"], dtype=float)
    joint_names = data["joint_names"]

    if q_traj.shape[0] != time_from_start.shape[0]:
        raise RuntimeError("q_traj 和 time_from_start 长度不一致。")

    return data, joint_names, time_from_start, q_traj, qd_traj, tcp_world


# 绘制TCP轨迹
def draw_tcp_path(stage, tcp_world):
    if stage.GetPrimAtPath(DEBUG_ROOT).IsValid():
        stage.RemovePrim(DEBUG_ROOT)

    UsdGeom.Xform.Define(stage, DEBUG_ROOT)

    curve = UsdGeom.BasisCurves.Define(stage, DEBUG_ROOT + "/tcp_path")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(tcp_world)])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in tcp_world])
    curve.CreateWidthsAttr([0.01])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 1.0)])

    for i, p in enumerate(tcp_world):
        if i % 4 != 0 and i != len(tcp_world) - 1:
            continue
        sphere = UsdGeom.Sphere.Define(stage, f"{DEBUG_ROOT}/wp_{i:03d}")
        sphere.CreateRadiusAttr(0.012)
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*p))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.8, 0.1)])


# 初始化articulation
def init_robot():
    world = World.instance() if World.instance() is not None else World()
    world.play()

    robot = SingleArticulation(
        prim_path=ARTICULATION_ROOT_PATH,
        name="go2_x5_track_robot",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {ARTICULATION_ROOT_PATH}")

    return world, robot


# 建立joint映射
def get_arm_indices(robot, joint_names):
    dof_names = list(robot.dof_names)
    indices = []

    for name in joint_names:
        if name not in dof_names:
            raise RuntimeError(f"Isaac DOF 中找不到 {name}")
        indices.append(dof_names.index(name))

    print("[mapping]", dict(zip(joint_names, indices)))
    return indices

# cubic Hermite样条插值
def sample_cubic_hermite(time_from_start, q, qd, t):
    """
    按时间采样关节目标。

    使用 cubic Hermite：
        q(t_i)
        q(t_{i+1})
        qd(t_i)
        qd(t_{i+1})

    比单纯 waypoint 跳点更平滑。
    """
    t = float(np.clip(t, time_from_start[0], time_from_start[-1]))

    index = int(np.searchsorted(time_from_start, t, side="right") - 1)
    index = max(0, min(index, len(time_from_start) - 2))

    t0 = time_from_start[index]
    t1 = time_from_start[index + 1]
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


# 控制 arm joints 的 action helper
def make_arm_action(q_arm_target, arm_indices, q_full_fallback):
    """
    优先只给 arm_joint1~6 发送目标。

    如果当前 Isaac 版本的 ArticulationAction 不支持 joint_indices，
    就 fallback 到完整 q_full target。
    """
    arm_indices_np = np.asarray(arm_indices, dtype=np.int32)

    try:
        return ArticulationAction(
            joint_positions=np.asarray(q_arm_target, dtype=float),
            joint_indices=arm_indices_np,
        )
    except TypeError:
        q_full_target = q_full_fallback.copy()
        q_full_target[arm_indices] = q_arm_target
        return ArticulationAction(joint_positions=q_full_target)


async def settle_to_start(world, robot, arm_indices, q_start):
    """
    正式执行轨迹前，先平滑移动到轨迹起点。
    """
    q_full_initial = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    q_arm_initial = q_full_initial[arm_indices].copy()

    num_steps = max(2, int(SETTLE_TO_START_DURATION * 60.0))

    print("[settle] q_arm current:", q_arm_initial)
    print("[settle] q_arm start:", q_start)

    for step in range(num_steps):
        u = step / float(num_steps - 1)
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

        q_arm_target = (1.0 - s) * q_arm_initial + s * q_start
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_arm_action(q_arm_target, arm_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

    print("[settle] done")


# 执行轨迹
async def track_trajectory(world, robot, arm_indices, time_from_start, q_traj, qd_traj):
    """
    按仿真时间连续跟踪 q_arm trajectory。
    """
    await settle_to_start(world, robot, arm_indices, q_traj[0])

    duration = float(time_from_start[-1])
    sim_dt = 1.0 / 60.0
    num_steps = int(np.ceil(duration / sim_dt)) + 1

    log = {
        "time": [],
        "target_q_arm": [],
        "actual_q_arm": [],
        "joint_error_norm": [],
    }

    print("[track] duration:", duration)
    print("[track] sim steps:", num_steps)

    for step in range(num_steps):
        t = min(step * sim_dt, duration)
        q_arm_target = sample_cubic_hermite(time_from_start, q_traj, qd_traj, t)

        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_arm_action(q_arm_target, arm_indices, q_full_now)

        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

        q_full_actual = np.asarray(robot.get_joint_positions(), dtype=float)
        q_arm_actual = q_full_actual[arm_indices]
        err = float(np.linalg.norm(q_arm_actual - q_arm_target))

        log["time"].append(t)
        log["target_q_arm"].append(q_arm_target.tolist())
        log["actual_q_arm"].append(q_arm_actual.tolist())
        log["joint_error_norm"].append(err)

        if step % TRACK_LOG_EVERY_N_STEPS == 0 or step == num_steps - 1:
            print(f"[track] t={t:.3f}/{duration:.3f}, joint_error={err:.6f}")

    # 终点保持
    q_final = q_traj[-1]
    hold_steps = int(HOLD_FINAL_DURATION * 60.0)
    for _ in range(hold_steps):
        q_full_now = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        action = make_arm_action(q_final, arm_indices, q_full_now)
        robot.apply_action(action)
        world.step(render=True)
        await omni.kit.app.get_app().next_update_async()

    print("[track] done")
    return log


# 保存跟踪结果到json
def save_tracking_result(log, source_plan):
    result = {
        "schema_version": 1,
        "source_plan": TRAJ_JSON,
        "planner_success": source_plan["plan_info"]["planner_success"],
        "joint_names": source_plan["joint_names"],
        "tracking": log,
        "summary": {
            "num_samples": len(log["time"]),
            "max_joint_error_norm": float(np.max(log["joint_error_norm"])),
            "mean_joint_error_norm": float(np.mean(log["joint_error_norm"])),
            "final_joint_error_norm": float(log["joint_error_norm"][-1]),
        },
    }

    with open(TRACK_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[output] tracking result:", TRACK_RESULT_JSON)
    print("[summary]", result["summary"])


async def main():
    print("========== Go2-X5 arm trajectory visualization + tracking ==========")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 Isaac stage")

    data, joint_names, time_from_start, q_traj, qd_traj, tcp_world = load_trajectory()
    print("[trajectory] q shape:", q_traj.shape)
    print("[trajectory] duration:", float(time_from_start[-1]))
    print("[trajectory] tcp waypoints:", len(tcp_world))

    draw_tcp_path(stage, tcp_world)

    world, robot = init_robot()
    await omni.kit.app.get_app().next_update_async()

    arm_indices = get_arm_indices(robot, joint_names)
    log = await track_trajectory(
        world,
        robot,
        arm_indices,
        time_from_start,
        q_traj,
        qd_traj,
    )

    save_tracking_result(log, data)

    print("========== complete ==========")


asyncio.ensure_future(main())