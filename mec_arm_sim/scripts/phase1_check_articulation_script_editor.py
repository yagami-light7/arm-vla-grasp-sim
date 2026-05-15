"""
阶段 1：Isaac Sim 5.1 Articulation 可控性检查。

运行方式：先在 Isaac Sim GUI 中打开包含机械臂的 USD/stage，
然后在 Script Editor 中运行本脚本。

本脚本做的事情：
1. 扫描当前 stage 中带有 UsdPhysics.ArticulationRootAPI 的 prim。
2. 根据 ROBOT_PRIM_PATH、当前选择项或唯一的 articulation root 确定机器人路径。
3. 初始化 SingleArticulation 封装。
4. 打印 Isaac Sim 内部 DOF 顺序、关节上下限、drive 参数和当前关节状态。
5. 通过 ArticulationAction 发送一个很小的关节位置目标。
6. 打印最终关节状态和目标误差，用于判断物理控制链路是否打通。

如果场景中有多个 articulation root，请先在 Stage 面板中选中你的机器人 root，
或者在下面手动设置 ROBOT_PRIM_PATH。
"""

import asyncio
import math
import traceback

import numpy as np
import omni
from pxr import Usd, UsdPhysics

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


# 如果自动检测有歧义，把这里改成你的 articulation root prim path。
# 示例：ROBOT_PRIM_PATH = "/World/mec_arm/base_link"
ROBOT_PRIM_PATH = None

# 可选：手动指定目标关节角。保持 None 时，脚本会自动生成一个小幅度目标姿态。
# 注意长度必须等于 len(robot.dof_names)，顺序必须用 Isaac Sim 打印出的 DOF 顺序，
# 不要假设它和 URDF 文件中的顺序完全一致。
TARGET_JOINT_POSITIONS = None

AUTO_ROTATION_OFFSET_RAD = math.radians(8.0)
AUTO_TRANSLATION_OFFSET_M = 0.01
SIM_STEPS = 240
PRINT_EVERY_N_STEPS = 60


def _stage():
    return omni.usd.get_context().get_stage()


def _selected_prim_paths():
    try:
        return list(omni.usd.get_context().get_selection().get_selected_prim_paths())
    except Exception:
        return []


def _articulation_root_paths():
    stage = _stage()
    roots = []
    for prim in stage.TraverseAll():
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
        except Exception:
            pass
    return roots


def _optional_articulation_base_paths():
    """仅作为辅助诊断；有些 Isaac Sim 5.1 安装包没有暴露这个函数。"""
    try:
        from isaacsim.core.utils.prims import find_all_articulation_base_paths

        return list(find_all_articulation_base_paths())
    except Exception as exc:
        print(f"[扫描] 可选函数 find_all_articulation_base_paths 不可用：{exc}")
        return []


def _rigid_body_paths_under(root_path):
    stage = _stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []

    links = []
    for prim in Usd.PrimRange(root):
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                links.append(str(prim.GetPath()))
        except Exception:
            pass
    return links


def _parent_path(prim_path):
    parts = prim_path.rstrip("/").split("/")
    if len(parts) <= 2:
        return "/"
    return "/".join(parts[:-1])


def _roots_under_path(root_paths, selected_path):
    prefix = selected_path.rstrip("/") + "/"
    return [path for path in root_paths if path == selected_path or path.startswith(prefix)]


def _resolve_robot_prim_path(root_paths):
    if ROBOT_PRIM_PATH:
        return ROBOT_PRIM_PATH

    selected = _selected_prim_paths()
    if selected:
        print("[选择] 当前选中的 prim path：")
        for path in selected:
            print(f"  - {path}")

        exact = [path for path in selected if path in root_paths]
        if len(exact) == 1:
            return exact[0]

        nested = []
        for path in selected:
            nested.extend(_roots_under_path(root_paths, path))
        nested = sorted(set(nested))
        if len(nested) == 1:
            return nested[0]

    if len(root_paths) == 1:
        return root_paths[0]

    raise RuntimeError("无法自动选择唯一的 articulation root。请在 Stage 面板中选中机器人，或设置脚本顶部的 ROBOT_PRIM_PATH。")


def _safe_numpy(value):
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(float, copy=False)


def _get_dof_property(row, key, default=np.nan):
    try:
        return row[key]
    except Exception:
        return default


def _dof_names(robot):
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


def _dof_properties(robot):
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


def _print_dof_table(robot):
    dof_names = _dof_names(robot)
    dof_count = len(dof_names)
    print(f"[机器人] DOF 数量：{dof_count}")
    print("[机器人] Isaac Sim 内部使用的 DOF 顺序：")

    dof_properties = _dof_properties(robot)

    try:
        limits = _safe_numpy(dof_properties[["lower", "upper"]])
    except Exception:
        limits = None

    for i, name in enumerate(dof_names):
        lower = upper = stiffness = damping = max_effort = np.nan
        dof_type = "unknown"
        try:
            props = dof_properties[i]
            lower = _get_dof_property(props, "lower")
            upper = _get_dof_property(props, "upper")
            stiffness = _get_dof_property(props, "stiffness")
            damping = _get_dof_property(props, "damping")
            max_effort = _get_dof_property(props, "maxEffort")
            dof_type = str(_get_dof_property(props, "type", "unknown"))
        except Exception:
            if limits is not None:
                lower, upper = limits[i]

        print(
            f"  [{i:02d}] {name:30s} type={dof_type:>8s} "
            f"limit=[{float(lower): .5f}, {float(upper): .5f}] "
            f"stiffness={float(stiffness): .3g} damping={float(damping): .3g} "
            f"maxEffort={float(max_effort): .3g}"
        )


def _make_auto_target(robot, q_current):
    dof_names = _dof_names(robot)
    dof_properties = _dof_properties(robot)
    target = q_current.copy()

    for i, _name in enumerate(dof_names):
        lower = -np.inf
        upper = np.inf
        dof_type = ""
        try:
            props = dof_properties[i]
            lower = float(_get_dof_property(props, "lower", -np.inf))
            upper = float(_get_dof_property(props, "upper", np.inf))
            dof_type = str(_get_dof_property(props, "type", ""))
        except Exception:
            pass

        if "Translation" in dof_type or "translation" in dof_type:
            offset = AUTO_TRANSLATION_OFFSET_M
        else:
            offset = AUTO_ROTATION_OFFSET_RAD

        if i % 2:
            offset *= -1.0

        candidate = q_current[i] + offset
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            margin = min(0.05 * (upper - lower), abs(offset))
            candidate = np.clip(candidate, lower + margin, upper - margin)
        target[i] = candidate

    return target


def _dynamic_control_type(prim_path):
    try:
        from omni.isaac.dynamic_control import _dynamic_control

        dc = _dynamic_control.acquire_dynamic_control_interface()
        return str(dc.peek_object_type(prim_path))
    except Exception as exc:
        return f"dynamic_control 不可用或未启用：{exc}"


async def phase1_check_articulation():
    print("\n========== Isaac Sim 阶段 1：Articulation 可控性检查 ==========")

    root_paths = _articulation_root_paths()
    print("[扫描] 带 UsdPhysics.ArticulationRootAPI 的 articulation root prim：")
    if root_paths:
        for path in root_paths:
            print(f"  - {path}")
    else:
        print("  <none>")

    base_paths = _optional_articulation_base_paths()

    print("[扫描] Isaac Sim utils 返回的 articulation base path：")
    if base_paths:
        for path in base_paths:
            print(f"  - {path}")
    else:
        print("  <none>")

    if not root_paths:
        raise RuntimeError(
            "没有找到 ArticulationRootAPI。请重新检查 URDF 导入结果：机器人 root/link prim 上必须有 "
            "Physics > Articulation Root。"
        )

    robot_prim_path = _resolve_robot_prim_path(root_paths)
    print(f"[机器人] 使用的 articulation root prim path：{robot_prim_path}")
    print("[机器人] dynamic_control 是旧接口诊断信息，仅供参考；本脚本以 SingleArticulation 是否可用为准。")

    link_paths = _rigid_body_paths_under(robot_prim_path)
    link_scan_path = robot_prim_path
    if not link_paths:
        parent_path = _parent_path(robot_prim_path)
        parent_link_paths = _rigid_body_paths_under(parent_path)
        if parent_link_paths:
            link_paths = parent_link_paths
            link_scan_path = parent_path

    print(f"[机器人] 在 {link_scan_path} 下找到的 RigidBody/link prim 候选：")
    if link_paths:
        for path in link_paths:
            print(f"  - {path}")
    else:
        print("  <未找到；如果后续 DOF 能正常读取和控制，这条信息不影响阶段 1 结论>")

    world = World.instance()
    if world is None:
        world = World()
        await world.initialize_simulation_context_async()

    await world.reset_async()
    await world.play_async()
    await omni.kit.app.get_app().next_update_async()

    print(f"[机器人] dynamic_control object type，仅供参考：{_dynamic_control_type(robot_prim_path)}")

    robot = SingleArticulation(prim_path=robot_prim_path, name="phase1_robot")
    robot.initialize()

    if not robot.is_valid():
        raise RuntimeError(f"SingleArticulation 对这个 prim path 无效：{robot_prim_path}")

    _print_dof_table(robot)

    q0 = _safe_numpy(robot.get_joint_positions())
    dq0 = _safe_numpy(robot.get_joint_velocities())
    print(f"[状态] q_current 当前关节位置：{np.array2string(q0, precision=5, suppress_small=False)}")
    print(f"[状态] dq_current 当前关节速度：{np.array2string(dq0, precision=5, suppress_small=False)}")

    if TARGET_JOINT_POSITIONS is None:
        q_target = _make_auto_target(robot, q0)
        print("[命令] 使用脚本自动生成的小幅度关节目标。")
    else:
        q_target = np.asarray(TARGET_JOINT_POSITIONS, dtype=float)
        if q_target.shape != q0.shape:
            raise ValueError(
                f"TARGET_JOINT_POSITIONS 的 shape {q_target.shape} 与当前关节 shape {q0.shape} 不一致。"
            )
        print("[命令] 使用手动指定的 TARGET_JOINT_POSITIONS。")

    print(f"[命令] q_target 目标关节位置：{np.array2string(q_target, precision=5, suppress_small=False)}")

    action = ArticulationAction(joint_positions=q_target)
    for step in range(SIM_STEPS):
        robot.apply_action(action)
        await omni.kit.app.get_app().next_update_async()

        if step % PRINT_EVERY_N_STEPS == 0 or step == SIM_STEPS - 1:
            q_now = _safe_numpy(robot.get_joint_positions())
            err = float(np.linalg.norm(q_target - q_now))
            print(
                f"[运行] step={step:04d} |q_target - q|={err:.6f} "
                f"q={np.array2string(q_now, precision=4, suppress_small=False)}"
            )

    q_final = _safe_numpy(robot.get_joint_positions())
    final_err = float(np.linalg.norm(q_target - q_final))
    moved = float(np.linalg.norm(q_final - q0))
    print(f"[结果] q_final 最终关节位置：{np.array2string(q_final, precision=5, suppress_small=False)}")
    print(f"[结果] moved_norm=||q_final - q_initial||：{moved:.6f}")
    print(f"[结果] target_error_norm=||q_target - q_final||：{final_err:.6f}")

    if moved < 1e-4:
        print(
            "[诊断] 机器人几乎没有运动。请检查 prim path 是否正确、仿真是否播放、"
            "position drive 的 stiffness/damping/max force 是否非零。"
        )
    else:
        print("[诊断] 关节位置在仿真中发生变化，物理控制链路已经打通。")

    print("========== 阶段 1 检查完成 ==========\n")


try:
    asyncio.ensure_future(phase1_check_articulation())
except Exception:
    traceback.print_exc()
