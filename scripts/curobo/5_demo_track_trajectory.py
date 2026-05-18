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
TCP_PRIM_PATH = "/World/go2_x5/arm_link6/arm_eef_link"

STEPS_PER_WAYPOINT = 4
START_Q_TOL = 0.05


# 读取curobo规划的轨迹数据
def load_trajectory():
    with open(TRAJ_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data["plan_info"]["planner_success"]:
        raise RuntimeError("planner_success=False，不建议执行。")

    q_traj = np.asarray(data["trajectory"]["q"], dtype=float)
    tcp_world = np.asarray(data["trajectory"]["tcp_position_world"], dtype=float)
    joint_names = data["joint_names"]

    return data, joint_names, q_traj, tcp_world


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


# 执行轨迹
async def track_trajectory(world, robot, arm_indices, q_traj):
    q_full_hold = np.asarray(robot.get_joint_positions(), dtype=float).copy()
    q_arm_now = q_full_hold[arm_indices]
    start_error = float(np.linalg.norm(q_arm_now - q_traj[0]))

    print("[start] q_arm current:", q_arm_now)
    print("[start] q_arm traj[0]:", q_traj[0])
    print("[start] error:", start_error)

    if start_error > START_Q_TOL:
        raise RuntimeError("当前 Isaac q_arm 和轨迹起点差太大，请重新 dump state 并重新规划。")

    for i, q_arm in enumerate(q_traj):
        q_full_target = q_full_hold.copy()
        q_full_target[arm_indices] = q_arm

        action = ArticulationAction(joint_positions=q_full_target)

        for _ in range(STEPS_PER_WAYPOINT):
            robot.apply_action(action)
            world.step(render=True)
            await omni.kit.app.get_app().next_update_async()

        if i % 5 == 0 or i == len(q_traj) - 1:
            q_now = np.asarray(robot.get_joint_positions(), dtype=float)
            err = float(np.linalg.norm(q_now[arm_indices] - q_arm))
            print(f"[track] wp={i:03d}, joint_error={err:.6f}")

    print("[track] done")


async def main():
    print("========== Go2-X5 arm trajectory visualization + tracking ==========")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 Isaac stage")

    data, joint_names, q_traj, tcp_world = load_trajectory()
    print("[trajectory] q shape:", q_traj.shape)
    print("[trajectory] tcp waypoints:", len(tcp_world))

    draw_tcp_path(stage, tcp_world)

    world, robot = init_robot()
    await omni.kit.app.get_app().next_update_async()

    arm_indices = get_arm_indices(robot, joint_names)
    await track_trajectory(world, robot, arm_indices, q_traj)

    print("========== complete ==========")


asyncio.ensure_future(main())