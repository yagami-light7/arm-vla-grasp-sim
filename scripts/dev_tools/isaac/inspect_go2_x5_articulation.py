"""
检查 Isaac Sim 当前 stage 中的 Go2-X5 articulation。

用途：
    本脚本用于 Go2-X5 迁移后的第一步 Isaac Sim 验证。
    它只读取和打印信息，不发送控制命令，不移动机器人。

运行位置：
    Isaac Sim 5.1.0 GUI -> Window -> Script Editor

它会检查：
    1. 当前 stage 中有哪些 ArticulationRootAPI prim。
    2. 自动解析 Go2-X5 的 robot root 和 articulation root。
    3. 初始化 SingleArticulation，读取 Isaac 内部 DOF order。
    4. 确认 arm_joint1 ~ arm_joint6 在完整 DOF order 中的位置。
    5. 打印 q_full / dq_full，以及切片后的 q_arm / dq_arm。
    6. 查找 arm_base_link、arm_link6、grasp_tcp_link 等关键 frame/link。
    7. 打印 grasp_tcp_link 在 world 和 arm_base_link 下的位姿。

为什么要做这一步：
    cuRobo 的 Go2-X5 arm planner 只使用：
        arm_joint1 ~ arm_joint6
        base frame = arm_base_link
        tool frame = grasp_tcp_link

    但 Isaac Sim 中加载的是完整 Go2-X5 articulation，里面还包含狗腿关节和夹爪关节。
    所以在做轨迹生成/追踪之前，必须先确认 Isaac 的完整 DOF order，
    并明确 arm_joint1 ~ arm_joint6 对应的索引。

注意：
    如果你的 stage 中机器人路径不是 /World/go2_x5，本脚本会自动扫描。
    如果自动扫描有歧义，可以手动设置下面的 ROBOT_ROOT_PATH 或 ARTICULATION_ROOT_PATH。
"""

from __future__ import annotations

import asyncio
import math
import traceback

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation


# 如果自动检测失败，在这里手动指定。
# 例：
# ROBOT_ROOT_PATH = "/World/go2_x5"
# ARTICULATION_ROOT_PATH = "/World/go2_x5/root_joint"
ROBOT_ROOT_PATH = None
ARTICULATION_ROOT_PATH = None

# cuRobo arm-only yml 中使用的 active joint 顺序。
EXPECTED_ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
]

# 夹爪不参与 cuRobo 主规划，但需要确认 Isaac 是否导入了这些 DOF。
EXPECTED_GRIPPER_JOINT_NAMES = [
    "arm_joint7",
    "arm_joint8",
]

# 关键 link/frame 名称。
BASE_LINK_NAME = "arm_base_link"
TCP_LINK_NAME = "grasp_tcp_link"
TCP_FALLBACK_PARENT_LINK_NAME = "arm_link6"
# 如果 Isaac stage 还没有重新导入 grasp_tcp_link，就用 arm_link6 + 固定偏移
# 临时估计同一个 TCP。这个数值必须和 URDF 中 grasp_tcp_fixed_joint 一致。
TCP_FALLBACK_OFFSET_XYZ = (0.1425699970126152, 0.0, 0.0)
TCP_FALLBACK_OFFSET_RPY = (0.0, 0.0, 0.0)


def get_stage():
    """获取当前 Isaac Sim GUI 已打开的 USD stage。"""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开 USD stage。请先在 Isaac Sim GUI 中打开或创建场景。")
    return stage


def parent_path(prim_path: str) -> str:
    """返回 prim path 的父路径。"""
    parts = prim_path.rstrip("/").split("/")
    if len(parts) <= 2:
        return "/"
    return "/".join(parts[:-1])


def safe_numpy(value) -> np.ndarray:
    """把 Isaac 返回的数据转成一维 numpy array。"""
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(float, copy=False)


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


def selected_prim_paths() -> list[str]:
    """读取 Stage 面板中当前选中的 prim path。"""
    try:
        return list(omni.usd.get_context().get_selection().get_selected_prim_paths())
    except Exception:
        return []


def roots_under_path(root_paths: list[str], selected_path: str) -> list[str]:
    """找出位于 selected_path 子树下的 articulation roots。"""
    prefix = selected_path.rstrip("/") + "/"
    return [
        path for path in root_paths
        if path == selected_path or path.startswith(prefix)
    ]


def resolve_articulation_root(stage) -> str:
    """解析当前要检查的 articulation root path。"""
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

    # 常见 URDF 导入结构是 /World/go2_x5/root_joint，
    # 真正包含所有 link 的 asset root 是它的父路径 /World/go2_x5。
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


def print_key_prims(stage, robot_root_path: str) -> None:
    """打印 Go2-X5 相关关键 prim。"""
    names = [
        "base",
        BASE_LINK_NAME,
        "arm_link1",
        "arm_link2",
        "arm_link3",
        "arm_link4",
        "arm_link5",
        "arm_link6",
        "arm_link7",
        "arm_link8",
        TCP_LINK_NAME,
    ] + EXPECTED_ARM_JOINT_NAMES + EXPECTED_GRIPPER_JOINT_NAMES

    found = find_prims_by_name_under(stage, robot_root_path, names)

    print("[扫描] Go2-X5 关键 prim:")
    for name in names:
        paths = found.get(name, [])
        if not paths:
            print(f"  - {name:20s}: <not found>")
            continue
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            api_flags = []
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api_flags.append("RigidBodyAPI")
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api_flags.append("ArticulationRootAPI")
            api_text = ", ".join(api_flags) if api_flags else "no physics API"
            print(f"  - {name:20s}: {path}  type={prim.GetTypeName()}  api={api_text}")


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


def get_dof_properties(robot):
    """读取 DOF properties。不同 Isaac Sim 小版本里属性入口可能略有差异。"""
    try:
        return robot.dof_properties
    except Exception:
        pass

    view = getattr(robot, "_articulation_view", None)
    if view is not None:
        try:
            return view.dof_properties
        except Exception:
            pass

    return None


def get_dof_property(row, key: str, default=np.nan):
    """从 structured row 中安全读取 DOF 属性。"""
    try:
        return row[key]
    except Exception:
        return default


def print_dof_table(robot) -> list[str]:
    """打印 Isaac 完整 DOF order 和关节属性。"""
    dof_names = get_dof_names(robot)
    if not dof_names:
        raise RuntimeError("无法读取 Isaac DOF order。")

    dof_properties = get_dof_properties(robot)

    print(f"[DOF] 总数量: {len(dof_names)}")
    print("[DOF] Isaac Sim 内部 DOF 顺序:")

    for index, name in enumerate(dof_names):
        lower = upper = stiffness = damping = max_effort = np.nan
        dof_type = "unknown"

        if dof_properties is not None:
            try:
                props = dof_properties[index]
                lower = get_dof_property(props, "lower")
                upper = get_dof_property(props, "upper")
                stiffness = get_dof_property(props, "stiffness")
                damping = get_dof_property(props, "damping")
                max_effort = get_dof_property(props, "maxEffort")
                dof_type = str(get_dof_property(props, "type", "unknown"))
            except Exception:
                pass

        print(
            f"  [{index:02d}] {name:24s} "
            f"type={dof_type:>8s} "
            f"limit=[{float(lower): .5f}, {float(upper): .5f}] "
            f"stiffness={float(stiffness): .3g} "
            f"damping={float(damping): .3g} "
            f"maxEffort={float(max_effort): .3g}"
        )

    return dof_names


def build_joint_index_map(dof_names: list[str], target_joint_names: list[str]) -> dict[str, int]:
    """把关节名映射到 Isaac 完整 DOF order 中的索引。"""
    missing = [name for name in target_joint_names if name not in dof_names]
    if missing:
        raise RuntimeError(f"Isaac DOF order 中缺少关节: {missing}")
    return {name: dof_names.index(name) for name in target_joint_names}


def print_arm_state(robot, dof_names: list[str]) -> None:
    """打印完整关节状态，以及 cuRobo arm planner 需要的 q_arm。"""
    q_full = safe_numpy(robot.get_joint_positions())
    dq_full = safe_numpy(robot.get_joint_velocities())

    arm_index_map = build_joint_index_map(dof_names, EXPECTED_ARM_JOINT_NAMES)
    arm_indices = [arm_index_map[name] for name in EXPECTED_ARM_JOINT_NAMES]

    q_arm = q_full[arm_indices]
    dq_arm = dq_full[arm_indices]

    print("[映射] arm_joint -> Isaac DOF index:")
    for name in EXPECTED_ARM_JOINT_NAMES:
        print(f"  - {name:12s}: {arm_index_map[name]}")

    gripper_found = [name for name in EXPECTED_GRIPPER_JOINT_NAMES if name in dof_names]
    gripper_missing = [name for name in EXPECTED_GRIPPER_JOINT_NAMES if name not in dof_names]
    if gripper_found:
        print("[映射] gripper joints found:")
        for name in gripper_found:
            print(f"  - {name:12s}: {dof_names.index(name)}")
    if gripper_missing:
        print(f"[映射] gripper joints missing: {gripper_missing}")
        print("       如果 URDF 导入时 merge fixed/mimic joints，这是可能的；夹爪后续单独处理。")

    print("[状态] q_full:")
    print(np.array2string(q_full, precision=6, suppress_small=False))
    print("[状态] dq_full:")
    print(np.array2string(dq_full, precision=6, suppress_small=False))
    print("[状态] q_arm 按 cuRobo joint_names 顺序 arm_joint1~6:")
    print(np.array2string(q_arm, precision=6, suppress_small=False))
    print("[状态] dq_arm 按 cuRobo joint_names 顺序 arm_joint1~6:")
    print(np.array2string(dq_arm, precision=6, suppress_small=False))


def rpy_to_matrix(rpy_xyz) -> Gf.Matrix4d:
    """rpy 转 USD Matrix4d，用于 grasp_tcp_link fallback offset。"""
    roll, pitch, yaw = [float(value) for value in rpy_xyz]

    rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(roll))
    rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(pitch))
    rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))

    matrix = Gf.Matrix4d(1.0)
    matrix *= Gf.Matrix4d(rot_x, Gf.Vec3d(0, 0, 0))
    matrix *= Gf.Matrix4d(rot_y, Gf.Vec3d(0, 0, 0))
    matrix *= Gf.Matrix4d(rot_z, Gf.Vec3d(0, 0, 0))
    return matrix


def xyz_rpy_to_matrix(xyz, rpy) -> Gf.Matrix4d:
    """xyz+rpy 转 USD Matrix4d。"""
    matrix = rpy_to_matrix(rpy)
    matrix.SetTranslateOnly(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return matrix


def matrix_to_numpy(matrix: Gf.Matrix4d) -> np.ndarray:
    """USD Matrix4d 转 numpy 4x4。"""
    return np.array([[float(matrix[row][col]) for col in range(4)] for row in range(4)], dtype=float)


def pose_text_from_matrix(matrix: Gf.Matrix4d) -> str:
    """把 USD Matrix4d 格式化为 position + quaternion 文本。"""
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (
        f"position_xyz=({translation[0]: .6f}, {translation[1]: .6f}, {translation[2]: .6f}), "
        f"quat_wxyz=({rotation.GetReal(): .6f}, {imaginary[0]: .6f}, "
        f"{imaginary[1]: .6f}, {imaginary[2]: .6f})"
    )


def print_frame_poses(stage, robot_root_path: str) -> None:
    """打印 arm_base_link 与 grasp_tcp_link 的世界位姿和相对位姿。"""
    found = find_prims_by_name_under(
        stage,
        robot_root_path,
        [BASE_LINK_NAME, TCP_LINK_NAME, TCP_FALLBACK_PARENT_LINK_NAME],
    )

    base_paths = found[BASE_LINK_NAME]
    if not base_paths:
        raise RuntimeError(f"找不到 base frame prim: {BASE_LINK_NAME}")
    base_path = base_paths[0]

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(base_path))

    tcp_mode = "direct grasp_tcp_link prim"
    tcp_paths = found[TCP_LINK_NAME]
    if tcp_paths:
        tcp_path = tcp_paths[0]
        tcp_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(tcp_path))
    else:
        tcp_mode = "fallback arm_link6 + fixed offset"
        parent_paths = found[TCP_FALLBACK_PARENT_LINK_NAME]
        if not parent_paths:
            raise RuntimeError(f"找不到 TCP fallback parent link: {TCP_FALLBACK_PARENT_LINK_NAME}")
        tcp_path = parent_paths[0]
        parent_world = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(tcp_path))
        parent_to_tcp = xyz_rpy_to_matrix(TCP_FALLBACK_OFFSET_XYZ, TCP_FALLBACK_OFFSET_RPY)
        tcp_world = parent_to_tcp * parent_world

    base_to_tcp = tcp_world * base_world.GetInverse()

    print("[Frame] base frame:", base_path)
    print("[Frame] tcp frame source:", tcp_path)
    print("[Frame] tcp mode:", tcp_mode)
    print("[Frame] T_world_base:", pose_text_from_matrix(base_world))
    print("[Frame] T_world_tcp:", pose_text_from_matrix(tcp_world))
    print("[Frame] T_base_tcp:", pose_text_from_matrix(base_to_tcp))
    print("[Frame] T_base_tcp matrix:")
    print(np.array2string(matrix_to_numpy(base_to_tcp), precision=6, suppress_small=False))


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
        name="go2_x5_inspect_articulation",
    )
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 无效: {articulation_root_path}")

    return robot


async def inspect_go2_x5_articulation():
    """Script Editor 主流程。"""
    print("========== Inspect Go2-X5 Articulation ==========")

    stage = get_stage()
    articulation_root_path = resolve_articulation_root(stage)
    robot_root_path = resolve_robot_root(stage, articulation_root_path)

    print("[机器人] robot root:", robot_root_path)
    print("[机器人] articulation root:", articulation_root_path)

    print_key_prims(stage, robot_root_path)

    robot = await create_articulation_handle(articulation_root_path)
    dof_names = print_dof_table(robot)
    print_arm_state(robot, dof_names)
    print_frame_poses(stage, robot_root_path)

    print("========== Inspect complete ==========")
    print("下一步：如果 arm_joint1~6 索引和 T_base_tcp 正常，就可以写状态导出脚本。")


async def main():
    try:
        await inspect_go2_x5_articulation()
    except Exception:
        traceback.print_exc()
        raise


asyncio.ensure_future(main())
