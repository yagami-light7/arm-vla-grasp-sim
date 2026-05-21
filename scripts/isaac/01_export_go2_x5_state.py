"""
导出 Isaac Sim 当前 Go2-X5 机器人状态到 JSON。

用途：
    本脚本是最终抓取 demo 的第 01 步。
    如果需要先人工检查 articulation，可运行
    scripts/dev_tools/isaac/inspect_go2_x5_articulation.py；
    本脚本负责把后续抓取和 cuRobo 脚本需要的数据写入 JSON。

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

        Isaac q_arm -> cuRobo FK -> grasp_tcp_link pose
        Isaac 导出的 T_base_tcp -> 对比误差

注意：
    本脚本只读状态，不控制机器人，不发送动作。
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation


WORKSPACE = Path("/home/light/workspace/arm_vla")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.math.SE3 import (
    matrix_to_pose,
    normalize_quat_wxyz,
    pose_dict_from_matrix,
    pose_to_matrix,
    xyz_rpy_to_matrix,
)


# 如果自动检测失败，在这里手动指定。
ROBOT_ROOT_PATH = None
ARTICULATION_ROOT_PATH = None

OUTPUT_JSON_PATH = Path("/tmp/go2_x5_isaac_state.json")

ROBOT_NAME = "go2_x5"
PLANNER_BASE_LINK_NAME = "arm_base_link"
PLANNER_TOOL_FRAME_NAME = "grasp_tcp_link"

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
# 如果 Isaac stage 还没有直接包含 grasp_tcp_link prim，则用这个固定变换
# 从 arm_link6 推算 TCP。该数值必须和 URDF 的 grasp_tcp_fixed_joint 一致。
TCP_FALLBACK_OFFSET_XYZ = (0.1425699970126152, 0.0, 0.0)
TCP_FALLBACK_OFFSET_RPY = (0.0, 0.0, 0.0)

EXPORT_WORLD_COLLISION = True
WORLD_COLLISION_PADDING_M = 0.02
WORLD_COLLISION_MIN_SIZE_M = 0.01
WORLD_COLLISION_MAX_OBSTACLES = 16
WORLD_COLLISION_LOCAL_RADIUS_M = 1.25
WORLD_COLLISION_MAX_EXTENT_M = 2.0
WORLD_COLLISION_MAX_HEIGHT_M = 1.6
WORLD_COLLISION_MAX_VOLUME_M3 = 2.5
WORLD_COLLISION_EXCLUDE_PREFIXES = (
    "/World/debug_",
    "/World/Looks",
    "/World/Render",
)
WORLD_COLLISION_EXCLUDE_NAME_KEYWORDS = (
    "floorplan",
    "wall",
    "door",
    "window",
    "ceiling",
    "wardrobe",
)


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


def path_overlaps(path_a: str, path_b: str) -> bool:
    """判断两个 USD path 是否存在父子或相等关系。"""
    a = path_a.rstrip("/")
    b = path_b.rstrip("/")
    if not a or not b or a == "/" or b == "/":
        return False
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def sanitize_obstacle_name(prim_path: str, index: int) -> str:
    """把 USD path 转成 cuRobo obstacle name。"""
    safe = prim_path.strip("/").replace("/", "_").replace(":", "_")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
    return f"obs_{index:03d}_{safe[-80:]}"


def distance_point_to_aabb_xy(point_xy, bbox_min, bbox_max) -> float:
    """计算 XY 平面中点到 AABB 的距离；点在投影内部时为 0。"""
    point_xy = np.asarray(point_xy, dtype=float)
    min_xy = np.asarray(bbox_min[:2], dtype=float)
    max_xy = np.asarray(bbox_max[:2], dtype=float)
    delta = np.maximum(np.maximum(min_xy - point_xy, point_xy - max_xy), 0.0)
    return float(np.linalg.norm(delta))


def point_inside_aabb(point, bbox_min, bbox_max, margin: float = 0.0) -> bool:
    """判断 point 是否在 AABB 内部。"""
    point = np.asarray(point, dtype=float)
    return bool(
        np.all(point >= np.asarray(bbox_min, dtype=float) - margin)
        and np.all(point <= np.asarray(bbox_max, dtype=float) + margin)
    )


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


def should_skip_collision_prim(prim_path: str, robot_root_path: str, selected_paths: list[str]) -> bool:
    """过滤机器人、自身目标物体和调试 prim，避免把它们当成环境障碍物。"""
    if path_overlaps(prim_path, robot_root_path):
        return True
    if any(prim_path.startswith(prefix) for prefix in WORLD_COLLISION_EXCLUDE_PREFIXES):
        return True
    prim_path_lower = prim_path.lower()
    if any(keyword in prim_path_lower for keyword in WORLD_COLLISION_EXCLUDE_NAME_KEYWORDS):
        return True
    for selected_path in selected_paths:
        if path_overlaps(prim_path, selected_path):
            return True
    return False


def compute_selected_bbox_centers(stage, selected_paths: list[str]) -> list[np.ndarray]:
    """读取当前选中目标的 bbox center，用于筛选操作区域附近障碍物。"""
    centers = []
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )

    for selected_path in selected_paths:
        prim = stage.GetPrimAtPath(selected_path)
        if not prim.IsValid():
            continue
        try:
            bound = bbox_cache.ComputeWorldBound(prim)
            aligned_box = bound.ComputeAlignedBox()
            bbox_min = np.array(aligned_box.GetMin(), dtype=float)
            bbox_max = np.array(aligned_box.GetMax(), dtype=float)
        except Exception:
            continue
        if np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max)):
            centers.append(0.5 * (bbox_min + bbox_max))

    return centers


def compute_world_collision_cuboids(
    stage,
    robot_root_path: str,
    T_world_base: np.ndarray,
    selected_paths: list[str],
) -> list[dict]:
    """
    从 Isaac stage 中导出环境碰撞体。

    第一版使用带 UsdPhysics.CollisionAPI prim 的 world AABB，转成 cuRobo
    可直接使用的 cuboid。这样对桌子、台面、柜体、墙等障碍物足够稳健；
    目标物体和机器人本体会被排除。
    """
    if not EXPORT_WORLD_COLLISION:
        return []

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    T_base_world = np.linalg.inv(T_world_base)
    base_position_world = T_world_base[:3, 3].copy()
    selected_centers = compute_selected_bbox_centers(stage, selected_paths)
    skipped_counts = {
        "name_or_path": 0,
        "too_large": 0,
        "outside_local_workspace": 0,
        "contains_arm_base": 0,
    }
    obstacles = []

    for prim in stage.TraverseAll():
        if not prim.IsValid() or not prim.IsActive():
            continue
        try:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
        except Exception:
            continue

        prim_path = str(prim.GetPath())
        if should_skip_collision_prim(prim_path, robot_root_path, selected_paths):
            skipped_counts["name_or_path"] += 1
            continue

        try:
            bound = bbox_cache.ComputeWorldBound(prim)
            aligned_box = bound.ComputeAlignedBox()
            bbox_min = np.array(aligned_box.GetMin(), dtype=float)
            bbox_max = np.array(aligned_box.GetMax(), dtype=float)
        except Exception:
            continue

        if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
            continue

        size = bbox_max - bbox_min
        if float(np.max(size)) < WORLD_COLLISION_MIN_SIZE_M:
            continue
        if (
            float(np.max(size)) > WORLD_COLLISION_MAX_EXTENT_M
            or float(size[2]) > WORLD_COLLISION_MAX_HEIGHT_M
            or float(np.prod(size)) > WORLD_COLLISION_MAX_VOLUME_M3
        ):
            skipped_counts["too_large"] += 1
            continue

        local_reference_points = selected_centers if selected_centers else [base_position_world]
        nearest_local_distance = min(
            distance_point_to_aabb_xy(point[:2], bbox_min, bbox_max)
            for point in local_reference_points
        )
        if nearest_local_distance > WORLD_COLLISION_LOCAL_RADIUS_M:
            skipped_counts["outside_local_workspace"] += 1
            continue

        if point_inside_aabb(
            base_position_world,
            bbox_min,
            bbox_max,
            margin=WORLD_COLLISION_PADDING_M,
        ):
            skipped_counts["contains_arm_base"] += 1
            continue

        center_world = 0.5 * (bbox_min + bbox_max)
        padded_size = np.maximum(
            size + 2.0 * WORLD_COLLISION_PADDING_M,
            WORLD_COLLISION_MIN_SIZE_M,
        )

        T_world_obstacle = np.eye(4, dtype=float)
        T_world_obstacle[:3, 3] = center_world
        T_base_obstacle = T_base_world @ T_world_obstacle

        obstacles.append(
            {
                "name": sanitize_obstacle_name(prim_path, len(obstacles)),
                "prim_path": prim_path,
                "type": "cuboid_from_world_aabb",
                "dims_xyz": padded_size.tolist(),
                "raw_bbox_world": {
                    "min_xyz": bbox_min.tolist(),
                    "max_xyz": bbox_max.tolist(),
                    "center_xyz": center_world.tolist(),
                    "size_xyz": size.tolist(),
                },
                "pose_world": pose_dict_from_matrix(T_world_obstacle),
                "pose_base": pose_dict_from_matrix(T_base_obstacle),
                "padding_m": WORLD_COLLISION_PADDING_M,
            }
        )

        if len(obstacles) >= WORLD_COLLISION_MAX_OBSTACLES:
            break

    print(
        "[World Collision] skipped: "
        f"name_or_path={skipped_counts['name_or_path']}, "
        f"too_large={skipped_counts['too_large']}, "
        f"outside_local_workspace={skipped_counts['outside_local_workspace']}, "
        f"contains_arm_base={skipped_counts['contains_arm_base']}"
    )
    return obstacles


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
    selected_paths = selected_prim_paths()
    world_collision_cuboids = compute_world_collision_cuboids(
        stage=stage,
        robot_root_path=robot_root_path,
        T_world_base=matrices["T_world_base"],
        selected_paths=selected_paths,
    )

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
        "world_collision": {
            "enabled": EXPORT_WORLD_COLLISION,
            "representation": "UsdPhysics.CollisionAPI world AABB exported as cuRobo cuboids in arm_base_link frame",
            "padding_m": WORLD_COLLISION_PADDING_M,
            "local_radius_m": WORLD_COLLISION_LOCAL_RADIUS_M,
            "max_extent_m": WORLD_COLLISION_MAX_EXTENT_M,
            "max_height_m": WORLD_COLLISION_MAX_HEIGHT_M,
            "max_volume_m3": WORLD_COLLISION_MAX_VOLUME_M3,
            "excluded_selected_prim_paths": selected_paths,
            "cuboids_base": world_collision_cuboids,
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
    world_collision = output.get("world_collision", {})
    cuboids = world_collision.get("cuboids_base", [])
    print("[World Collision] cuboids:", len(cuboids))
    for obstacle in cuboids[:8]:
        print(
            "  - "
            f"{obstacle['name']}: path={obstacle['prim_path']}, "
            f"dims={np.array2string(np.asarray(obstacle['dims_xyz']), precision=3)}"
        )
    if len(cuboids) > 8:
        print(f"  ... {len(cuboids) - 8} more")


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


if __name__ == "__main__":
    asyncio.ensure_future(main())
