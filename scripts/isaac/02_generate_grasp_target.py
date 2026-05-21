"""
从 Isaac Sim 中选中的物体生成一个简单 grasp target。

用途：
    这是 Go2-X5 抓取流程的第一版目标生成器。
    它不使用 AnyGrasp，也不使用相机点云。
    它直接读取仿真中的 object prim pose / bbox，
    用传统几何规则生成一个 grasp pose。
    默认优先侧向抓取，适合桌子较高、top-down 容易碰桌面或腕部姿态难规划的场景。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

输出：
    /tmp/go2_x5_target_tcp_pose.json

后续流程：
    1. 运行本脚本生成 target pose
    2. 单点调试：终端运行 scripts/dev_tools/curobo/demo_plan_to_pose.py
    3. 抓取分段：后续由 scripts/curobo/03_plan_grasp_trajectory.py 读取 poses 字段
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import UsdGeom, Gf

WORKSPACE = Path("/home/light/workspace/arm_vla")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.math.SE3 import matrix_to_pose, pose_to_matrix, rotmat_to_quat_wxyz


OUTPUT_TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")
STATE_JSON = Path("/tmp/go2_x5_isaac_state.json")

ROBOT_ROOT_PATH = "/World/go2_x5"
ARM_BASE_LINK_PATH = "/World/go2_x5/arm_base_link"

DEBUG_ROOT_PATH = "/World/debug_sim_grasp_target"

# grasp_tcp_link 是两指之间的 TCP frame。
PREFERRED_GRASP_MODE = "side"  # 可选: "side", "top_down"

# 对平行夹爪抓立方体，TCP 应该落到物体高度中部附近，而不是停在物体顶部上方。
# 第一版使用 bbox 顶部向下一个深度。
# 对苹果这类近似球体，TCP 停在 bbox 中心高度附近更容易让两指形成对夹；
# 之前 0.025 m 会让 TCP 高于苹果中心约 1 cm，容易夹到上半部分但 lift 不起来。
GRASP_DEPTH_BELOW_TOP_M = 0.035

# 侧向抓取时，TCP 默认放在 bbox 中心高度。正值会让 TCP 稍微高于中心。
SIDE_GRASP_CENTER_Z_OFFSET_M = 0.008

# 侧向抓取时，pregrasp 沿 TCP 局部 -X 退出，即从物体朝机械臂方向退开。
SIDE_PREGRASP_OFFSET_M = 0.10

# pregrasp 比 grasp 再高一点，后续 approach 会从 pregrasp 下降到 grasp。
PREGRASP_OFFSET_M = 0.10

# lift 比 grasp 更高一点，后续 close gripper 后向上抬起物体。
# 当前 Go2-X5 arm 在 top-down 姿态下从低位 grasp 继续抬太高会进入
# cuRobo 难以求解的构型。第一版先回到 pregrasp 高度，保证流程可闭环。
LIFT_OFFSET_M = 0.10

# 经验诊断阈值：只打印 warning，不阻止生成 JSON。
# 这些阈值不是机械臂严格工作空间，而是用来提醒“目标明显比之前成功样例更难”。
WORKSPACE_WARN_XY_RADIUS_M = 0.55
WORKSPACE_WARN_GRASP_Z_M = 0.35
WORKSPACE_WARN_PREGRASP_Z_M = 0.45
WORKSPACE_WARN_RADIUS_3D_M = 0.65

# 当前 grasp_tcp_link 的局部 +X 轴是夹爪伸出方向。
# top-down 抓取时，需要让 TCP +X 指向 arm_base_link -Z。
# 注意这里故意在 base frame 下定义姿态，而不是 world frame：
# 如果机器狗底座在 world 中转了 180 deg yaw，world 固定姿态会给腕部引入
# 一个不必要的 180 deg roll；base 固定姿态更接近机械臂自身的可达构型。
TCP_TOP_DOWN_QUAT_BASE_WXYZ = np.array(
    [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
    dtype=float,
)

# 夹爪控制使用的第一版经验值。这里只写入 JSON，真正控制在 Isaac 执行脚本里完成。
GRIPPER_OPEN_M = 0.043
GRIPPER_CLOSE_M = 0.0


# 读取stage并选中物体
def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage")
    return stage

def get_selected_object_path():
    selected = list(omni.usd.get_context().get_selection().get_selected_prim_paths())

    if not selected:
        raise RuntimeError("请先在 Stage 面板中选中一个要抓取的物体 prim。")

    # 如果选中了多个，默认用第一个。
    return selected[0]


# 读取prime world transform
def get_world_transform(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim 不存在: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    matrix = xformable.ComputeLocalToWorldTransform(0.0)

    T = np.array(matrix, dtype=float).T

    # 注意：pxr Gf.Matrix4d 和 numpy 的行列约定容易混。
    # 我们统一使用标准 SE(3)：平移在最后一列。
    if not np.allclose(T[3, :3], 0.0):
        T = np.array(matrix, dtype=float)

    return T


def get_arm_base_transform(stage):
    """
    获取 arm_base_link 的 world transform。

    一键流程会先运行 01_export_go2_x5_state.py，它会自动解析当前场景中的
    Go2-X5 实例路径，例如 /World/go2_x5 或 /World/go2_x5_01。
    因此这里优先使用 state JSON 中的 world_base，避免重新导入机器人后
    仍然使用硬编码 /World/go2_x5/arm_base_link。
    """
    if STATE_JSON.exists():
        data = json.loads(STATE_JSON.read_text(encoding="utf-8"))
        matrix = data.get("poses", {}).get("world_base", {}).get("matrix_4x4")
        base_path = data.get("paths", {}).get("base_frame_path")
        if matrix is not None:
            return (
                np.asarray(matrix, dtype=float),
                f"{STATE_JSON}::{base_path}",
            )

    return (
        get_world_transform(stage, ARM_BASE_LINK_PATH),
        ARM_BASE_LINK_PATH,
    )


# 读取object bbox
def compute_world_bbox(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim 不存在: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )

    bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = bound.ComputeAlignedBox()

    bbox_min = np.array(aligned_box.GetMin(), dtype=float)
    bbox_max = np.array(aligned_box.GetMax(), dtype=float)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_size = bbox_max - bbox_min

    return bbox_min, bbox_max, bbox_center, bbox_size


def make_grasp_position_world(bbox_min, bbox_max, bbox_center):
    """根据物体 bbox 生成 top-down grasp TCP 的 world 位置。"""
    return np.array([
        bbox_center[0],
        bbox_center[1],
        bbox_max[2] - GRASP_DEPTH_BELOW_TOP_M,
    ], dtype=float)


def make_side_grasp_position_world(bbox_center):
    """根据物体 bbox 生成 side grasp TCP 的 world 位置。"""
    grasp_position = np.array(bbox_center, dtype=float).copy()
    grasp_position[2] += SIDE_GRASP_CENTER_Z_OFFSET_M
    return grasp_position


def world_point_to_base_position(T_world_base, point_world):
    """把 world 中的一个点转换到 arm_base_link 坐标系。"""
    point_world_h = np.array(
        [point_world[0], point_world[1], point_world[2], 1.0],
        dtype=float,
    )
    point_base_h = np.linalg.inv(T_world_base) @ point_world_h
    return point_base_h[:3]


def make_top_down_grasp_pose_base(T_world_base, bbox_min, bbox_max, bbox_center):
    """生成 base frame 下的 top-down grasp pose。"""
    grasp_position_world = make_grasp_position_world(
        bbox_min,
        bbox_max,
        bbox_center,
    )
    grasp_position_base = world_point_to_base_position(
        T_world_base,
        grasp_position_world,
    )
    return pose_to_matrix(grasp_position_base, TCP_TOP_DOWN_QUAT_BASE_WXYZ)


def make_side_grasp_pose_base(T_world_base, bbox_center):
    """
    生成 base frame 下的侧向抓取 pose。

    约定：
        - grasp_tcp_link 局部 +X 是夹爪伸出/接近物体的方向。
        - 侧抓时，让局部 +X 指向 arm_base_link -> object 的水平径向方向。
        - pregrasp 会沿局部 -X 后退，因此机械臂会从靠近自身的一侧接近物体。
        - 局部 +Z 保持朝 base +Z，使夹爪姿态尽量直立。
    """
    grasp_position_world = make_side_grasp_position_world(bbox_center)
    grasp_position_base = world_point_to_base_position(
        T_world_base,
        grasp_position_world,
    )

    approach_xy = grasp_position_base[:2].copy()
    approach_norm = np.linalg.norm(approach_xy)
    if approach_norm < 1.0e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_axis = np.array(
            [approach_xy[0] / approach_norm, approach_xy[1] / approach_norm, 0.0],
            dtype=float,
        )

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = rotmat_to_quat_wxyz(rotation)
    return pose_to_matrix(grasp_position_base, quaternion)


def base_pose_to_world_pose(T_world_base, T_base_pose):
    """arm_base_link frame 下的 pose 转成 world frame 下的 pose。"""
    return T_world_base @ T_base_pose


def offset_pose_along_base_z(T_base_pose, offset_z_m):
    """
    沿 arm_base_link +Z 方向平移一个 pose。

    第一版 top-down grasp 中：
        pregrasp = grasp 上方一点
        lift = grasp 上方更高一点

    注意：
        这里移动的是 base +Z，不是 TCP 局部 Z。
        当前固定底座场景中 base 只绕 world Z 旋转，因此 base +Z 和 world +Z 一致。
    """
    T_offset = np.array(T_base_pose, dtype=float).copy()
    T_offset[2, 3] += float(offset_z_m)
    return T_offset


def offset_pose_along_local_x(T_base_pose, offset_x_m):
    """沿 pose 自身局部 +X 方向平移。负值表示沿局部 -X 后退。"""
    T_offset = np.array(T_base_pose, dtype=float).copy()
    local_x_in_base = T_offset[:3, 0]
    T_offset[:3, 3] += local_x_in_base * float(offset_x_m)
    return T_offset


def world_to_base_pose(T_world_base, T_world_target):
    """world frame 下的 pose 转成 arm_base_link frame 下的 pose。"""
    return np.linalg.inv(T_world_base) @ T_world_target


def make_named_pose_entry(T_base_pose, T_world_pose):
    """把同一个目标 pose 同时保存成 base frame 和 world frame 表达。"""
    base_position, base_quaternion = matrix_to_pose(T_base_pose)
    world_position, world_quaternion = matrix_to_pose(T_world_pose)

    return {
        "frame": "arm_base_link",
        "position_xyz": base_position.tolist(),
        "quaternion_wxyz": base_quaternion.tolist(),
        "world": {
            "frame": "world",
            "position_xyz": world_position.tolist(),
            "quaternion_wxyz": world_quaternion.tolist(),
        },
    }


def make_target_workspace_diagnostics(T_base_pregrasp, T_base_grasp, T_base_lift):
    """生成目标在 arm_base_link 下的简单可达性诊断。"""
    pose_items = {
        "pregrasp": T_base_pregrasp,
        "grasp": T_base_grasp,
        "lift": T_base_lift,
    }

    diagnostics = {}
    warnings = []

    for name, T_base_pose in pose_items.items():
        position, quaternion = matrix_to_pose(T_base_pose)
        xy_radius = float(np.linalg.norm(position[:2]))
        radius_3d = float(np.linalg.norm(position))

        diagnostics[name] = {
            "position_xyz": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "xy_radius_m": xy_radius,
            "radius_3d_m": radius_3d,
            "z_m": float(position[2]),
        }

    grasp = diagnostics["grasp"]
    pregrasp = diagnostics["pregrasp"]

    if grasp["xy_radius_m"] > WORKSPACE_WARN_XY_RADIUS_M:
        warnings.append(
            "grasp 的水平距离偏大，可能接近机械臂可达边界。"
        )
    if grasp["z_m"] > WORKSPACE_WARN_GRASP_Z_M:
        warnings.append(
            "grasp 在 arm_base_link 上方较高，top-down 姿态可能难以规划。"
        )
    if pregrasp["z_m"] > WORKSPACE_WARN_PREGRASP_Z_M:
        warnings.append(
            "pregrasp 比 arm_base_link 高很多，move_to_pregrasp 可能失败。"
        )
    if pregrasp["radius_3d_m"] > WORKSPACE_WARN_RADIUS_3D_M:
        warnings.append(
            "pregrasp 的三维距离偏大，建议降低物体高度或把物体放近机械臂。"
        )

    diagnostics["warnings"] = warnings
    return diagnostics


def print_target_workspace_diagnostics(diagnostics):
    """打印目标在 arm_base_link 下的位置诊断。"""
    print("[diagnostic] target poses in arm_base_link:")
    for name in ["pregrasp", "grasp", "lift"]:
        item = diagnostics[name]
        print(
            f"  - {name:8s} pos={np.array(item['position_xyz'])} "
            f"xy_radius={item['xy_radius_m']:.3f} "
            f"z={item['z_m']:.3f} "
            f"radius_3d={item['radius_3d_m']:.3f}"
        )

    for warning in diagnostics["warnings"]:
        print("[warning]", warning)


def create_marker(stage, path, position, color, radius=0.025):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)

    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return sphere


def draw_debug_markers(stage, grasp_pos_world, pregrasp_pos_world, lift_pos_world, bbox_center):
    if stage.GetPrimAtPath(DEBUG_ROOT_PATH).IsValid():
        stage.RemovePrim(DEBUG_ROOT_PATH)

    UsdGeom.Xform.Define(stage, DEBUG_ROOT_PATH)

    create_marker(
        stage,
        DEBUG_ROOT_PATH + "/object_center",
        bbox_center,
        color=(1.0, 1.0, 1.0),
        radius=0.02,
    )
    create_marker(
        stage,
        DEBUG_ROOT_PATH + "/grasp",
        grasp_pos_world,
        color=(1.0, 0.1, 0.1),
        radius=0.025,
    )
    create_marker(
        stage,
        DEBUG_ROOT_PATH + "/pregrasp",
        pregrasp_pos_world,
        color=(0.1, 0.6, 1.0),
        radius=0.025,
    )
    create_marker(
        stage,
        DEBUG_ROOT_PATH + "/lift",
        lift_pos_world,
        color=(0.1, 1.0, 0.2),
        radius=0.025,
    )

    curve = UsdGeom.BasisCurves.Define(stage, DEBUG_ROOT_PATH + "/sequence_line")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([3])
    curve.CreatePointsAttr([
        Gf.Vec3f(*pregrasp_pos_world),
        Gf.Vec3f(*grasp_pos_world),
        Gf.Vec3f(*lift_pos_world),
    ])
    curve.CreateWidthsAttr([0.008])
    UsdGeom.Gprim(curve.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.9, 0.1)])


def save_target_pose_json(
    object_path,
    T_base_pregrasp,
    T_base_grasp,
    T_base_lift,
    T_world_pregrasp,
    T_world_grasp,
    T_world_lift,
    bbox_min,
    bbox_max,
    bbox_center,
    bbox_size,
    workspace_diagnostics,
):
    pregrasp_entry = make_named_pose_entry(T_base_pregrasp, T_world_pregrasp)
    grasp_entry = make_named_pose_entry(T_base_grasp, T_world_grasp)
    lift_entry = make_named_pose_entry(T_base_lift, T_world_lift)

    # 兼容 dev_tools/curobo/demo_plan_to_pose.py：顶层 pose 仍然代表 grasp。
    grasp_pos_base, grasp_quat_base = matrix_to_pose(T_base_grasp)
    grasp_pos_world, grasp_quat_world = matrix_to_pose(T_world_grasp)
    pregrasp_pos_world, pregrasp_quat_world = matrix_to_pose(T_world_pregrasp)
    lift_pos_world, lift_quat_world = matrix_to_pose(T_world_lift)

    payload = {
        "schema_version": 1,
        "frame": "arm_base_link",
        "default_target_name": "grasp",
        "position_xyz": grasp_pos_base.tolist(),
        "quaternion_wxyz": grasp_quat_base.tolist(),
        "sequence": [
            "pregrasp",
            "grasp",
            "close_gripper",
            "lift",
        ],
        "poses": {
            "pregrasp": pregrasp_entry,
            "grasp": grasp_entry,
            "lift": lift_entry,
        },
        "gripper": {
            "open_m": GRIPPER_OPEN_M,
            "close_m": GRIPPER_CLOSE_M,
            "joint_names": [
                "arm_joint7",
                "arm_joint8",
            ],
        },
        "source": {
            "type": f"sim_object_bbox_{PREFERRED_GRASP_MODE}",
            "grasp_mode": PREFERRED_GRASP_MODE,
            "object_prim_path": object_path,
            "world_grasp_pose": {
                "position_xyz": grasp_pos_world.tolist(),
                "quaternion_wxyz": grasp_quat_world.tolist(),
            },
            "world_pregrasp_pose": {
                "position_xyz": pregrasp_pos_world.tolist(),
                "quaternion_wxyz": pregrasp_quat_world.tolist(),
            },
            "world_lift_pose": {
                "position_xyz": lift_pos_world.tolist(),
                "quaternion_wxyz": lift_quat_world.tolist(),
            },
            "bbox_world": {
                "min_xyz": bbox_min.tolist(),
                "max_xyz": bbox_max.tolist(),
                "center_xyz": bbox_center.tolist(),
                "size_xyz": bbox_size.tolist(),
            },
            "grasp_depth_below_top_m": GRASP_DEPTH_BELOW_TOP_M,
            "side_grasp_center_z_offset_m": SIDE_GRASP_CENTER_Z_OFFSET_M,
            "side_pregrasp_offset_m": SIDE_PREGRASP_OFFSET_M,
            "pregrasp_offset_m": PREGRASP_OFFSET_M,
            "lift_offset_m": LIFT_OFFSET_M,
            "tcp_orientation_rule": (
                "side: grasp_tcp_link local +X points horizontally from arm_base_link to object"
                if PREFERRED_GRASP_MODE == "side"
                else "base_top_down: grasp_tcp_link local +X points to arm_base_link -Z"
            ),
        },
        "diagnostics": {
            "target_workspace_base": workspace_diagnostics,
        },
    }

    with OUTPUT_TARGET_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[output] target pose json:", OUTPUT_TARGET_JSON)
    print("[target default]: grasp")
    print("[grasp base position]:", grasp_pos_base)
    print("[grasp base quat_wxyz]:", grasp_quat_base)
    print("[pregrasp base position]:", pregrasp_entry["position_xyz"])
    print("[lift base position]:", lift_entry["position_xyz"])


async def main():
    print("========== Generate Sim Object Grasp Target ==========")
    print("[config] preferred grasp mode:", PREFERRED_GRASP_MODE)

    stage = get_stage()
    object_path = get_selected_object_path()
    print("[object] selected:", object_path)

    if object_path == ROBOT_ROOT_PATH or object_path.startswith(ROBOT_ROOT_PATH + "/"):
        raise RuntimeError("你选中了机器人自身，请选中要抓取的物体。")

    T_world_base, base_source = get_arm_base_transform(stage)
    base_pos, base_quat = matrix_to_pose(T_world_base)
    print("[base] source:", base_source)
    print("[base] world position:", base_pos)
    print("[base] world quat_wxyz:", base_quat)

    bbox_min, bbox_max, bbox_center, bbox_size = compute_world_bbox(stage, object_path)
    print("[object] bbox min:", bbox_min)
    print("[object] bbox max:", bbox_max)
    print("[object] bbox center:", bbox_center)
    print("[object] bbox size:", bbox_size)

    if PREFERRED_GRASP_MODE == "side":
        T_base_grasp = make_side_grasp_pose_base(T_world_base, bbox_center)
        T_base_pregrasp = offset_pose_along_local_x(
            T_base_grasp,
            -SIDE_PREGRASP_OFFSET_M,
        )
    elif PREFERRED_GRASP_MODE == "top_down":
        T_base_grasp = make_top_down_grasp_pose_base(
            T_world_base,
            bbox_min,
            bbox_max,
            bbox_center,
        )
        T_base_pregrasp = offset_pose_along_base_z(T_base_grasp, PREGRASP_OFFSET_M)
    else:
        raise RuntimeError(f"未知 PREFERRED_GRASP_MODE: {PREFERRED_GRASP_MODE}")

    T_base_lift = offset_pose_along_base_z(T_base_grasp, LIFT_OFFSET_M)

    T_world_pregrasp = base_pose_to_world_pose(T_world_base, T_base_pregrasp)
    T_world_grasp = base_pose_to_world_pose(T_world_base, T_base_grasp)
    T_world_lift = base_pose_to_world_pose(T_world_base, T_base_lift)

    workspace_diagnostics = make_target_workspace_diagnostics(
        T_base_pregrasp,
        T_base_grasp,
        T_base_lift,
    )
    print_target_workspace_diagnostics(workspace_diagnostics)

    grasp_pos_world, _ = matrix_to_pose(T_world_grasp)
    pregrasp_pos_world, _ = matrix_to_pose(T_world_pregrasp)
    lift_pos_world, _ = matrix_to_pose(T_world_lift)

    draw_debug_markers(
        stage,
        grasp_pos_world,
        pregrasp_pos_world,
        lift_pos_world,
        bbox_center,
    )

    save_target_pose_json(
        object_path,
        T_base_pregrasp,
        T_base_grasp,
        T_base_lift,
        T_world_pregrasp,
        T_world_grasp,
        T_world_lift,
        bbox_min,
        bbox_max,
        bbox_center,
        bbox_size,
        workspace_diagnostics,
    )

    await omni.kit.app.get_app().next_update_async()
    print("========== complete ==========")


if __name__ == "__main__":
    asyncio.ensure_future(main())
