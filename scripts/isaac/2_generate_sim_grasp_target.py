"""
从 Isaac Sim 中选中的物体生成一个简单 grasp target。

用途：
    这是 Go2-X5 抓取流程的第一版目标生成器。
    它不使用 AnyGrasp，也不使用相机点云。
    它直接读取仿真中的 object prim pose / bbox，
    用传统几何规则生成一个 top-down grasp pose。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

输出：
    /tmp/go2_x5_target_tcp_pose.json

后续流程：
    1. 运行本脚本生成 target pose
    2. 单点调试：终端运行 scripts/curobo/4_demo_plan_to_pose.py
    3. 抓取分段：后续由 scripts/curobo/6_plan_grasp_segments.py 读取 poses 字段
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
SCRIPTS_DIR = WORKSPACE / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from SE3 import matrix_to_pose, pose_to_matrix


OUTPUT_TARGET_JSON = Path("/tmp/go2_x5_target_tcp_pose.json")

ROBOT_ROOT_PATH = "/World/go2_x5"
ARM_BASE_LINK_PATH = "/World/go2_x5/arm_base_link"

DEBUG_ROOT_PATH = "/World/debug_sim_grasp_target"

# grasp_tcp_link 是两指之间的 TCP frame。
# 对平行夹爪抓立方体，TCP 应该落到物体高度中部附近，而不是停在物体顶部上方。
# 第一版使用 bbox 顶部向下一个深度：5 cm 立方体时，0.025 m 约等于中心高度。
GRASP_DEPTH_BELOW_TOP_M = 0.025

# pregrasp 比 grasp 再高一点，后续 approach 会从 pregrasp 下降到 grasp。
PREGRASP_OFFSET_M = 0.10

# lift 比 grasp 更高一点，后续 close gripper 后向上抬起物体。
# 当前 Go2-X5 arm 在 top-down 姿态下从低位 grasp 继续抬太高会进入
# cuRobo 难以求解的构型。第一版先回到 pregrasp 高度，保证流程可闭环。
LIFT_OFFSET_M = 0.10

# 当前 grasp_tcp_link 的局部 +X 轴是夹爪伸出方向。
# top-down 抓取时，需要让 TCP +X 指向 Isaac world -Z。
# 这等价于绕 world/base Y 轴旋转 +90 deg，四元数顺序为 wxyz。
TCP_TOP_DOWN_QUAT_WXYZ = np.array(
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


# 生成 top-down grasp pose
def make_top_down_grasp_pose_world(bbox_min, bbox_max, bbox_center):
    grasp_position = np.array([
        bbox_center[0],
        bbox_center[1],
        bbox_max[2] - GRASP_DEPTH_BELOW_TOP_M,
    ], dtype=float)

    grasp_quat_wxyz = TCP_TOP_DOWN_QUAT_WXYZ.copy()

    T_world_grasp = pose_to_matrix(grasp_position, grasp_quat_wxyz)
    return T_world_grasp


def offset_pose_along_world_z(T_world_pose, offset_z_m):
    """
    沿 Isaac world +Z 方向平移一个 pose。

    第一版 top-down grasp 中：
        pregrasp = grasp 上方一点
        lift = grasp 上方更高一点

    注意：
        这里移动的是 world +Z，不是 TCP 局部 Z。
        对当前机器狗底座被撑平、base 和 world 朝向一致的调试场景，这是最直观的。
    """
    T_offset = np.array(T_world_pose, dtype=float).copy()
    T_offset[2, 3] += float(offset_z_m)
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
):
    pregrasp_entry = make_named_pose_entry(T_base_pregrasp, T_world_pregrasp)
    grasp_entry = make_named_pose_entry(T_base_grasp, T_world_grasp)
    lift_entry = make_named_pose_entry(T_base_lift, T_world_lift)

    # 兼容旧的 4_demo_plan_to_pose.py：顶层 pose 仍然代表 grasp。
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
            "type": "sim_object_bbox_top_down",
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
            "pregrasp_offset_m": PREGRASP_OFFSET_M,
            "lift_offset_m": LIFT_OFFSET_M,
            "tcp_orientation_rule": "top_down: grasp_tcp_link local +X points to Isaac world -Z",
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

    stage = get_stage()
    object_path = get_selected_object_path()
    print("[object] selected:", object_path)

    if object_path == ROBOT_ROOT_PATH or object_path.startswith(ROBOT_ROOT_PATH + "/"):
        raise RuntimeError("你选中了机器人自身，请选中要抓取的物体。")

    T_world_base = get_world_transform(stage, ARM_BASE_LINK_PATH)
    base_pos, base_quat = matrix_to_pose(T_world_base)
    print("[base] world position:", base_pos)
    print("[base] world quat_wxyz:", base_quat)

    bbox_min, bbox_max, bbox_center, bbox_size = compute_world_bbox(stage, object_path)
    print("[object] bbox min:", bbox_min)
    print("[object] bbox max:", bbox_max)
    print("[object] bbox center:", bbox_center)
    print("[object] bbox size:", bbox_size)

    T_world_grasp = make_top_down_grasp_pose_world(
        bbox_min,
        bbox_max,
        bbox_center,
    )
    T_world_pregrasp = offset_pose_along_world_z(T_world_grasp, PREGRASP_OFFSET_M)
    T_world_lift = offset_pose_along_world_z(T_world_grasp, LIFT_OFFSET_M)

    T_base_pregrasp = world_to_base_pose(T_world_base, T_world_pregrasp)
    T_base_grasp = world_to_base_pose(T_world_base, T_world_grasp)
    T_base_lift = world_to_base_pose(T_world_base, T_world_lift)

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
    )

    await omni.kit.app.get_app().next_update_async()
    print("========== complete ==========")


asyncio.ensure_future(main())
