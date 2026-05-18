"""
导出 Isaac Sim 当前 Go2-X5 机器人状态到 JSON。

用途：
    本脚本是 Go2-X5 + cuRobo 对齐流程的第二步。
    上一步 scripts/isaac/0_inspect_go2_x5_articulation.py 负责人工检查；
    本脚本负责把后续 cuRobo 脚本需要的数据写入 JSON。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

导出内容：
    1. 完整 Isaac DOF order
    2. q_full / dq_full
    3. arm_joint1 ~ arm_joint6 在完整 DOF 中的索引
    4. 按 cuRobo joint_names 顺序排列的 q_arm / dq_arm
    5. gripper joint 的索引和状态
    6. T_world_base, T_world_tcp, T_base_tcp

输出文件：
    /tmp/go2_x5_isaac_state.json

为什么需要这个 JSON：
    普通 Python / cuRobo 脚本不能直接读取 Isaac Sim stage。
    所以后续 FK 对齐脚本会读取这个 JSON：

        Isaac q_arm -> cuRobo FK -> arm_eef_link pose
        Isaac 导出的 T_base_tcp -> 对比误差

注意：
    本脚本只读状态，不控制机器人，不发送动作。
"""

from __future__ import annotations

import asyncio
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation


# 如果自动检测失败，在这里手动指定。
ROBOT_ROOT_PATH = None
ARTICULATION_ROOT_PATH = None

OUTPUT_JSON_PATH = Path("/tmp/go2_x5_isaac_state.json")

ROBOT_NAME = "go2_x5"
PLANNER_BASE_LINK_NAME = "arm_base_link"
PLANNER_TOOL_FRAME_NAME = "arm_eef_link"

ACTIVE_ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

GRIPPER_JOINT_NAMES = [
    "arm_joint7",
    "arm_joint8",
]

TCP_FALLBACK_PARENT_LINK_NAME = "arm_link6"
TCP_FALLBACK_OFFSET_XYZ = (0.08657, 0.0, 0.0)
TCP_FALLBACK_OFFSET_RPY = (0.0, 0.0, 0.0)


def normalize_quat_wxyz(quat) -> np.ndarray:
    """归一化 wxyz 四元数。"""
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError(f"四元数范数太小，无法归一化: {quat}")
    return quat / norm


def quat_wxyz_to_rotmat(quat) -> np.ndarray:
    """wxyz 四元数转 3x3 旋转矩阵。"""
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(rotation) -> np.ndarray:
    """3x3 旋转矩阵转 wxyz 四元数。"""
    rotation = np.asarray(rotation, dtype=float)
    trace = np.trace(rotation)

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag = np.diag(rotation)
        index = int(np.argmax(diag))

        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale

    return normalize_quat_wxyz([w, x, y, z])


def rpy_to_rotmat(rpy_xyz) -> np.ndarray:
    """URDF rpy 转 3x3 旋转矩阵。"""
    roll, pitch, yaw = [float(value) for value in rpy_xyz]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return rot_z @ rot_y @ rot_x


def pose_to_matrix(position_xyz, quat_wxyz) -> np.ndarray:
    """position + wxyz quaternion 转标准 4x4 SE(3) 矩阵，平移在最后一列。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
    transform[:3, 3] = np.asarray(position_xyz, dtype=float)
    return transform


def xyz_rpy_to_matrix(xyz, rpy) -> np.ndarray:
    """xyz + rpy 转标准 4x4 SE(3) 矩阵。"""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_rotmat(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def matrix_to_pose(transform) -> tuple[np.ndarray, np.ndarray]:
    """标准 4x4 SE(3) 矩阵转 position + wxyz quaternion。"""
    transform = np.asarray(transform, dtype=float)
    position = transform[:3, 3].copy()
    quaternion = rotmat_to_quat_wxyz(transform[:3, :3])
    return position, quaternion


def pose_dict_from_matrix(transform) -> dict:
    """把标准 4x4 SE(3) 矩阵转成 JSON 友好字段。"""
    position, quaternion = matrix_to_pose(transform)
    return {
        "position_xyz": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "matrix_4x4": np.asarray(transform, dtype=float).tolist(),
        "matrix_convention": "standard_SE3_translation_last_column",
    }


def get_stage():
    """获取当前 Isaac Sim GUI 已打开的 USD stage。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage。请先在 Isaac Sim GUI 中打开或创建场景。")
    return stage


def selected_prim_paths() -> list[str]:
    """读取 Stage 面板中当前选中的 prim path。"""
    try:
        return list(omni.usd.get_context().get_selection().get_selected_prim_paths())
    except Exception:
        return []


def parent_path(prim_path: str) -> str:
    """返回 prim path 的父路径。"""
    parts = prim_path.rstrip("/").split("/")
    if len(parts) <= 2:
        return "/"
    return "/".join(parts[:-1])


def scan_articulation_roots(stage) -> list[str]:
    """扫描当前 stage 中带 UsdPhysics.ArticulationRootAPI 的 prim。"""
    roots = []
    for prim in stage.TraverseAll():
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
        except Exception:
            pass
    return roots


def roots_under_path(root_paths: list[str], selected_path: str) -> list[str]:
    """找出位于 selected_path 子树下的 articulation roots。"""
    prefix = selected_path.rstrip("/") + "/"
    return [path for path in root_paths if path == selected_path or path.startswith(prefix)]


def resolve_articulation_root(stage) -> str:
    """解析当前要导出的 articulation root path。"""
    if ARTICULATION_ROOT_PATH:
        prim = stage.GetPrimAtPath(ARTICULATION_ROOT_PATH)
        if not prim.IsValid():
            raise RuntimeError(f"手动设置的 ARTICULATION_ROOT_PATH 无效: {ARTICULATION_ROOT_PATH}")
        return ARTICULATION_ROOT_PATH

    roots = scan_articulation_roots(stage)
    print("[扫描] ArticulationRootAPI prim:")
    if roots:
        for path in roots:
            print(f"  - {path}")
    else:
        print("  <none>")

    selected = selected_prim_paths()
    if selected:
        print("[选择] Stage 当前选中 prim:")
        for path in selected:
            print(f"  - {path}")

        exact = [path for path in selected if path in roots]
        if len(exact) == 1:
            return exact[0]

        nested = []
        for path in selected:
            nested.extend(roots_under_path(roots, path))
        nested = sorted(set(nested))
        if len(nested) == 1:
            return nested[0]

    if len(roots) == 1:
        return roots[0]

    raise RuntimeError(
        "无法自动确定唯一 articulation root。"
        "请在 Stage 面板选中 Go2-X5，或设置脚本顶部 ARTICULATION_ROOT_PATH。"
    )


def resolve_robot_root(stage, articulation_root_path: str) -> str:
    """解析 robot root path。"""
    if ROBOT_ROOT_PATH:
        prim = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
        if not prim.IsValid():
            raise RuntimeError(f"手动设置的 ROBOT_ROOT_PATH 无效: {ROBOT_ROOT_PATH}")
        return ROBOT_ROOT_PATH

    candidate = parent_path(articulation_root_path)
    if stage.GetPrimAtPath(candidate).IsValid():
        return candidate

    return articulation_root_path


def find_prims_by_name_under(stage, root_path: str, names: list[str]) -> dict[str, list[str]]:
    """在 robot root 下按 prim name 查找 link/frame/joint prim。"""
    result = {name: [] for name in names}
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"robot root 无效: {root_path}")

    for prim in Usd.PrimRange(root_prim):
        prim_name = prim.GetName()
        if prim_name in result:
            result[prim_name].append(str(prim.GetPath()))

    return result


def usd_world_pose_to_matrix(stage, prim_path: str) -> np.ndarray:
    """读取 USD prim 的 world pose，并转成标准 4x4 SE(3) 矩阵。"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim path 不存在: {prim_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    usd_matrix = xform_cache.GetLocalToWorldTransform(prim)

    translation = usd_matrix.ExtractTranslation()
    rotation = usd_matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()

    position = np.array([translation[0], translation[1], translation[2]], dtype=float)
    quaternion = normalize_quat_wxyz([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
    return pose_to_matrix(position, quaternion)


def resolve_base_and_tcp_matrices(stage, robot_root_path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """解析 base/tcp prim，并返回标准 SE(3) 矩阵。"""
    found = find_prims_by_name_under(
        stage,
        robot_root_path,
        [PLANNER_BASE_LINK_NAME, PLANNER_TOOL_FRAME_NAME, TCP_FALLBACK_PARENT_LINK_NAME],
    )

    base_paths = found[PLANNER_BASE_LINK_NAME]
    if not base_paths:
        raise RuntimeError(f"找不到 base frame prim: {PLANNER_BASE_LINK_NAME}")
    base_path = base_paths[0]

    tcp_mode = "direct_tool_frame_prim"
    tcp_paths = found[PLANNER_TOOL_FRAME_NAME]
    if tcp_paths:
        tcp_path = tcp_paths[0]
        T_world_tcp = usd_world_pose_to_matrix(stage, tcp_path)
    else:
        tcp_mode = "fallback_parent_link_plus_fixed_offset"
        parent_paths = found[TCP_FALLBACK_PARENT_LINK_NAME]
        if not parent_paths:
            raise RuntimeError(f"找不到 TCP fallback parent link: {TCP_FALLBACK_PARENT_LINK_NAME}")
        tcp_path = parent_paths[0]
        T_world_parent = usd_world_pose_to_matrix(stage, tcp_path)
        T_parent_tcp = xyz_rpy_to_matrix(TCP_FALLBACK_OFFSET_XYZ, TCP_FALLBACK_OFFSET_RPY)
        T_world_tcp = T_world_parent @ T_parent_tcp

    T_world_base = usd_world_pose_to_matrix(stage, base_path)
    T_base_tcp = np.linalg.inv(T_world_base) @ T_world_tcp

    resolved_paths = {
        "base_frame_path": base_path,
        "tcp_frame_path": tcp_path,
        "tcp_mode": tcp_mode,
    }
    matrices = {
        "T_world_base": T_world_base,
        "T_world_tcp": T_world_tcp,
        "T_base_tcp": T_base_tcp,
    }
    return resolved_paths, matrices


def safe_numpy(value) -> np.ndarray:
    """把 Isaac 返回的数据转成一维 numpy array。"""
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(float, copy=False)


def get_dof_names(robot) -> list[str]:
    """读取 Isaac Sim 内部 DOF order。"""
    try:
        return list(robot.dof_names)
    except Exception:
        pass

    view = getattr(robot, "_articulation_view", None)
    if view is not None:
        try:
            return list(view.dof_names)
        except Exception:
            pass

    return []


def build_joint_index_map(dof_names: list[str], joint_names: list[str]) -> dict[str, int]:
    """把关节名映射到完整 DOF order 中的索引。"""
    missing = [name for name in joint_names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Isaac DOF order 中缺少关节: {missing}\n完整 DOF order: {dof_names}")
    return {name: dof_names.index(name) for name in joint_names}


def slice_by_index_map(values: np.ndarray, joint_names: list[str], index_map: dict[str, int]) -> np.ndarray:
    """按 joint_names 顺序从完整 q/dq 中切出子向量。"""
    indices = [index_map[name] for name in joint_names]
    return values[indices]


async def create_articulation_handle(articulation_root_path: str):
    """进入 play 状态并初始化 SingleArticulation。"""
    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    robot = SingleArticulation(
        prim_path=articulation_root_path,
        name="go2_x5_dump_state",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {articulation_root_path}")

    return robot


def build_output_json(
    stage,
    robot_root_path: str,
    articulation_root_path: str,
    robot,
) -> dict:
    """读取 Isaac 状态并组装 JSON 数据。"""
    dof_names = get_dof_names(robot)
    if not dof_names:
        raise RuntimeError("无法读取 Isaac DOF order。")

    q_full = safe_numpy(robot.get_joint_positions())
    dq_full = safe_numpy(robot.get_joint_velocities())

    arm_index_map = build_joint_index_map(dof_names, ACTIVE_ARM_JOINT_NAMES)
    q_arm = slice_by_index_map(q_full, ACTIVE_ARM_JOINT_NAMES, arm_index_map)
    dq_arm = slice_by_index_map(dq_full, ACTIVE_ARM_JOINT_NAMES, arm_index_map)

    gripper_joint_names = [name for name in GRIPPER_JOINT_NAMES if name in dof_names]
    gripper_index_map = build_joint_index_map(dof_names, gripper_joint_names) if gripper_joint_names else {}
    q_gripper = slice_by_index_map(q_full, gripper_joint_names, gripper_index_map) if gripper_joint_names else np.array([])
    dq_gripper = slice_by_index_map(dq_full, gripper_joint_names, gripper_index_map) if gripper_joint_names else np.array([])

    frame_paths, matrices = resolve_base_and_tcp_matrices(stage, robot_root_path)

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "robot_name": ROBOT_NAME,
        "source": "Isaac Sim Script Editor",
        "paths": {
            "robot_root_path": robot_root_path,
            "articulation_root_path": articulation_root_path,
            **frame_paths,
        },
        "planner_convention": {
            "base_link": PLANNER_BASE_LINK_NAME,
            "tool_frame": PLANNER_TOOL_FRAME_NAME,
            "active_joint_names": ACTIVE_ARM_JOINT_NAMES,
            "active_joint_order_note": "q_arm follows active_joint_names and cuRobo joint_names order.",
        },
        "isaac_state": {
            "dof_count": len(dof_names),
            "dof_names": dof_names,
            "q_full": q_full.tolist(),
            "dq_full": dq_full.tolist(),
            "arm_joint_indices": arm_index_map,
            "q_arm": q_arm.tolist(),
            "dq_arm": dq_arm.tolist(),
            "gripper_joint_names": gripper_joint_names,
            "gripper_joint_indices": gripper_index_map,
            "q_gripper": q_gripper.tolist(),
            "dq_gripper": dq_gripper.tolist(),
        },
        "poses": {
            "world_base": pose_dict_from_matrix(matrices["T_world_base"]),
            "world_tcp": pose_dict_from_matrix(matrices["T_world_tcp"]),
            "base_tcp": pose_dict_from_matrix(matrices["T_base_tcp"]),
        },
    }


def print_summary(output: dict, output_path: Path) -> None:
    """打印导出摘要。"""
    state = output["isaac_state"]
    poses = output["poses"]

    print("[导出] JSON:", output_path)
    print("[机器人] robot root:", output["paths"]["robot_root_path"])
    print("[机器人] articulation root:", output["paths"]["articulation_root_path"])
    print("[DOF] count:", state["dof_count"])
    print("[DOF] arm_joint_indices:")
    for name in ACTIVE_ARM_JOINT_NAMES:
        print(f"  - {name:12s}: {state['arm_joint_indices'][name]}")
    print("[状态] q_arm:")
    print(np.array2string(np.asarray(state["q_arm"], dtype=float), precision=6, suppress_small=False))
    print("[状态] dq_arm:")
    print(np.array2string(np.asarray(state["dq_arm"], dtype=float), precision=6, suppress_small=False))
    print("[Frame] base frame:", output["paths"]["base_frame_path"])
    print("[Frame] tcp frame:", output["paths"]["tcp_frame_path"])
    print("[Frame] tcp mode:", output["paths"]["tcp_mode"])
    print("[Frame] T_base_tcp position:", poses["base_tcp"]["position_xyz"])
    print("[Frame] T_base_tcp quat_wxyz:", poses["base_tcp"]["quaternion_wxyz"])


async def dump_go2_x5_state():
    """Script Editor 主流程。"""
    print("========== Dump Go2-X5 Isaac State ==========")

    stage = get_stage()
    articulation_root_path = resolve_articulation_root(stage)
    robot_root_path = resolve_robot_root(stage, articulation_root_path)

    robot = await create_articulation_handle(articulation_root_path)
    output = build_output_json(stage, robot_root_path, articulation_root_path, robot)

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary(output, OUTPUT_JSON_PATH)
    print("========== Dump complete ==========")
    print("下一步：普通 Python 脚本读取该 JSON，并用 cuRobo FK 对齐 T_base_tcp。")


async def main():
    try:
        await dump_go2_x5_state()
    except Exception:
        traceback.print_exc()
        raise


asyncio.ensure_future(main())
