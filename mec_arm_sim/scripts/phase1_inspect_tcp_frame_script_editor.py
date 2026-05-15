"""
阶段 1 后半段：检查 TCP/link/frame，并在视口中可视化 TCP 位置。

运行方式：
1. 在 Isaac Sim GUI 中打开包含机械臂的 stage。
2. 在 Script Editor 中运行本脚本。

本脚本解决的问题：
1. 打印当前 stage 中的 articulation root。
2. 自动推断机器人 asset root，例如 /World/mec_arm_6dof_01。
3. 扫描机器人下面所有 prim，查找 TCP_link、Empty_Link6 等关键 frame/link。
4. 如果找到了 TCP_link，打印它的世界位姿并创建一个小的 RGB marker。
5. 如果没有找到 TCP_link，使用 URDF 中的固定关节 TCP_fixed_joint：
   parent=Empty_Link6，xyz=(0, 0, 0.20)，rpy=(0, 0, -0.8724)
   从 Empty_Link6 推算 TCP 位置，并创建 marker。

注意：
TCP_link 在你的 URDF 中是空 link，没有 visual/collision/inertial。
因此它即使被导入，也通常不会像实体零件一样显示。
"""

import math
import traceback

import omni
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


# 如果自动推断不对，在这里手动写机器人 asset root。
# 例：ROBOT_ASSET_ROOT_PATH = "/World/mec_arm_6dof_01"
ROBOT_ASSET_ROOT_PATH = None

TCP_LINK_NAME = "TCP_link"
FALLBACK_PARENT_LINK_NAME = "Empty_Link6"
FALLBACK_TCP_OFFSET_XYZ = (0.0, 0.0, 0.20)
FALLBACK_TCP_OFFSET_RPY = (0.0, 0.0, -0.8724)

MARKER_ROOT_PATH = "/World/debug_tcp_marker"
MARKER_RADIUS_M = 0.012
AXIS_LENGTH_M = 0.08


def _stage():
    return omni.usd.get_context().get_stage()


def _parent_path(prim_path):
    parts = prim_path.rstrip("/").split("/")
    if len(parts) <= 2:
        return "/"
    return "/".join(parts[:-1])


def _articulation_root_paths(stage):
    paths = []
    for prim in stage.TraverseAll():
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                paths.append(str(prim.GetPath()))
        except Exception:
            pass
    return paths


def _resolve_robot_asset_root(stage):
    if ROBOT_ASSET_ROOT_PATH:
        return ROBOT_ASSET_ROOT_PATH

    roots = _articulation_root_paths(stage)
    print("[扫描] articulation root：")
    for path in roots:
        print(f"  - {path}")

    if len(roots) == 1:
        # 你的导入结果是 /World/mec_arm_6dof_01/root_joint，
        # 真正包含所有 link 的机器人 asset root 是它的父路径。
        return _parent_path(roots[0])

    selected = list(omni.usd.get_context().get_selection().get_selected_prim_paths())
    if len(selected) == 1:
        return selected[0]

    raise RuntimeError("无法自动确定机器人 root。请设置脚本顶部的 ROBOT_ASSET_ROOT_PATH。")


def _find_named_prims_under(root_path, names):
    stage = _stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"机器人 root 无效：{root_path}")

    result = {name: [] for name in names}
    for prim in Usd.PrimRange(root):
        prim_name = prim.GetName()
        if prim_name in result:
            result[prim_name].append(prim)
    return result


def _print_interesting_prims(root_path):
    stage = _stage()
    root = stage.GetPrimAtPath(root_path)

    print(f"[扫描] {root_path} 下的关键 prim：")
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if "TCP" in name or "tcp" in name or "Link" in name or "link" in name or "Joint" in name:
            api_flags = []
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api_flags.append("RigidBodyAPI")
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api_flags.append("ArticulationRootAPI")
            api_text = ", ".join(api_flags) if api_flags else "no physics API"
            print(f"  - {prim.GetPath()}  type={prim.GetTypeName()}  api={api_text}")


def _matrix_to_pose_text(matrix):
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    quat = rotation.GetImaginary()
    return (
        f"position_xyz=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}), "
        f"quat_wxyz=({rotation.GetReal():.6f}, {quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f})"
    )


def _local_tcp_transform_matrix():
    x, y, z = FALLBACK_TCP_OFFSET_XYZ
    roll, pitch, yaw = FALLBACK_TCP_OFFSET_RPY

    transform = Gf.Matrix4d(1.0)
    transform.SetTranslate(Gf.Vec3d(x, y, z))

    rot_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(roll))
    rot_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(pitch))
    rot_z = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))

    # URDF rpy 使用固定轴 roll-pitch-yaw；这里对你的 fallback 主要影响 yaw。
    rotation = Gf.Matrix4d(rot_x, Gf.Vec3d(0, 0, 0))
    rotation *= Gf.Matrix4d(rot_y, Gf.Vec3d(0, 0, 0))
    rotation *= Gf.Matrix4d(rot_z, Gf.Vec3d(0, 0, 0))
    return rotation * transform


def _create_colored_sphere(path, position, color, radius):
    stage = _stage()
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    return sphere.GetPrim()


def _delete_old_marker():
    stage = _stage()
    marker_path = Sdf.Path(MARKER_ROOT_PATH)
    if stage.GetPrimAtPath(marker_path).IsValid():
        stage.RemovePrim(marker_path)


def _make_tcp_marker(tcp_world_matrix):
    _delete_old_marker()
    stage = _stage()
    UsdGeom.Xform.Define(stage, Sdf.Path(MARKER_ROOT_PATH))

    origin = tcp_world_matrix.Transform(Gf.Vec3d(0, 0, 0))
    x_tip = tcp_world_matrix.Transform(Gf.Vec3d(AXIS_LENGTH_M, 0, 0))
    y_tip = tcp_world_matrix.Transform(Gf.Vec3d(0, AXIS_LENGTH_M, 0))
    z_tip = tcp_world_matrix.Transform(Gf.Vec3d(0, 0, AXIS_LENGTH_M))

    _create_colored_sphere(f"{MARKER_ROOT_PATH}/origin_white", origin, (1.0, 1.0, 1.0), MARKER_RADIUS_M * 1.2)
    _create_colored_sphere(f"{MARKER_ROOT_PATH}/x_red", x_tip, (1.0, 0.0, 0.0), MARKER_RADIUS_M)
    _create_colored_sphere(f"{MARKER_ROOT_PATH}/y_green", y_tip, (0.0, 1.0, 0.0), MARKER_RADIUS_M)
    _create_colored_sphere(f"{MARKER_ROOT_PATH}/z_blue", z_tip, (0.0, 0.2, 1.0), MARKER_RADIUS_M)

    print(f"[可视化] 已创建 TCP marker：{MARKER_ROOT_PATH}")
    print("[可视化] 白色=TCP 原点，红色=TCP +X，绿色=TCP +Y，蓝色=TCP +Z")


def inspect_tcp_frame():
    stage = _stage()
    robot_root_path = _resolve_robot_asset_root(stage)
    print(f"[机器人] 使用机器人 asset root：{robot_root_path}")

    _print_interesting_prims(robot_root_path)

    found = _find_named_prims_under(robot_root_path, [TCP_LINK_NAME, FALLBACK_PARENT_LINK_NAME])
    xform_cache = UsdGeom.XformCache()

    tcp_prims = found[TCP_LINK_NAME]
    if tcp_prims:
        tcp_prim = tcp_prims[0]
        tcp_world = xform_cache.GetLocalToWorldTransform(tcp_prim)
        print(f"[TCP] 找到 TCP prim：{tcp_prim.GetPath()}")
        print(f"[TCP] 世界位姿：{_matrix_to_pose_text(tcp_world)}")
        _make_tcp_marker(tcp_world)
        return

    print(f"[TCP] 没有找到名为 {TCP_LINK_NAME} 的 prim。")
    print("[TCP] 这通常表示空 fixed link 在导入 USD 时被省略或合并。")

    parent_prims = found[FALLBACK_PARENT_LINK_NAME]
    if not parent_prims:
        raise RuntimeError(f"也没有找到 fallback parent link：{FALLBACK_PARENT_LINK_NAME}")

    parent_prim = parent_prims[0]
    parent_world = xform_cache.GetLocalToWorldTransform(parent_prim)
    tcp_local = _local_tcp_transform_matrix()
    tcp_world = tcp_local * parent_world

    print(f"[TCP] 使用 fallback parent link：{parent_prim.GetPath()}")
    print(f"[TCP] URDF TCP_fixed_joint xyz={FALLBACK_TCP_OFFSET_XYZ}, rpy={FALLBACK_TCP_OFFSET_RPY}")
    print(f"[TCP] fallback 推算 TCP 世界位姿：{_matrix_to_pose_text(tcp_world)}")
    _make_tcp_marker(tcp_world)


try:
    inspect_tcp_frame()
except Exception:
    traceback.print_exc()
