"""在 Isaac Lab 中用键盘或 ROS 2 测试 MoE-CTS TorchScript policy。"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


# Isaac Kit 关闭解释器时不会保证刷新普通文件缓冲。行缓冲确保 headless smoke、
# CI 和重定向日志都能保留本脚本自己的 [INFO]/[RESULT] 验收证据。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.navigation.adapters.moe_cts_policy_adapter import (  # noqa: E402
    MoeCtsPolicyAdapter,
    MoeCtsPolicyContract,
)
from source.navigation.adapters.moe_cts_command_sink import (  # noqa: E402
    MoeCtsCommandBufferSink,
)
from source.navigation.isaac_ros2_environment import (  # noqa: E402
    validate_isaac_ros2_custom_message_environment,
)


DEFAULT_ROBOTLAB_REPO = Path("/mnt/sage_data/workspace/go2_rl_robotlab")
DEFAULT_POLICY_RELATIVE_PATH = Path(
    "deploy/pre_train/go2/go2_moe_cts_176k_0.6984.pt"
)
DEFAULT_MULTIFLOOR_SCENE_USD = (
    PROJECT_ROOT / "source/scene/multifloor/usda/multifloor.usda"
)
DEFAULT_MULTIFLOOR_COLLISION_PRIM_PATH = "/World/scene_collision"
# 这些位姿来自当前 collision PLY 的支撑面投影。z 均是 base 高度，不是地面高度。
DEFAULT_MULTIFLOOR_START_XYZ = (
    -3.4748268127441406,
    6.524534225463867,
    0.1636741725990273,
)
DEFAULT_MULTIFLOOR_START_YAW = 1.67247
DEFAULT_MULTIFLOOR_GOAL_XYZ = (
    0.4,
    -0.02,
    3.339914814802456,
)
DEFAULT_MULTIFLOOR_GOAL_YAW = -math.pi / 2.0
POLICY_CONTRACT = MoeCtsPolicyContract()
EXPECTED_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


# 必须先解析 AppLauncher 参数并启动 Isaac Sim，之后才能导入 omni、torch 和
# 外部 robot_lab。外部包的 __init__.py 会导入 omni.ext，顺序错误会报
# ModuleNotFoundError: No module named 'omni'。
parser = argparse.ArgumentParser(
    description=(
        "在确定性楼梯或 PCT multifloor 场景中测试 "
        "go2_rl_robotlab 的 MoE-CTS policy"
    )
)
parser.add_argument(
    "--terrain-mode",
    choices=("deterministic_stairs", "multifloor"),
    default="deterministic_stairs",
    help="Isaac 地形来源；默认保留已经验收的程序生成楼梯",
)
parser.add_argument(
    "--navigation-planning-only",
    action="store_true",
    help=(
        "只在 multifloor 中发布真实观测和 PCT 目标；不创建 /cmd_vel "
        "订阅，也不允许速度写入 policy"
    ),
)
parser.add_argument(
    "--command-source",
    choices=("keyboard", "ros2"),
    default="keyboard",
    help="速度命令来源；ros2 会订阅 geometry_msgs/Twist",
)
parser.add_argument(
    "--cmd-vel-topic",
    type=str,
    default="/cmd_vel",
    help="ROS 2 Twist topic，仅 command-source=ros2 时使用",
)
parser.add_argument(
    "--reference-path-topic",
    type=str,
    default="/initial_path",
    help="SCAN 地面高度参考 Path topic，仅 command-source=ros2 时使用",
)
parser.add_argument(
    "--stair-execution-frozen-topic",
    type=str,
    default="/planning/stair_execution_frozen",
    help="与当前 Path 绑定的楼梯执行冻结状态 topic",
)
parser.add_argument(
    "--ros-domain-id",
    type=int,
    default=None,
    help="ROS_DOMAIN_ID；留空时由 Isaac ROS2Context 读取环境变量",
)
parser.add_argument(
    "--cmd-vel-timeout",
    type=float,
    default=0.25,
    help="没有新 Twist 后的停车超时，单位 s",
)
parser.add_argument(
    "--policy-max-vx",
    type=float,
    default=0.80,
    help="ROS 模式允许的最大 |vx|，单位 m/s",
)
parser.add_argument(
    "--policy-max-vy",
    type=float,
    default=0.50,
    help="ROS 模式允许的最大 |vy|，单位 m/s",
)
parser.add_argument(
    "--policy-max-wz",
    type=float,
    default=0.80,
    help="ROS 模式允许的最大 |wz|，单位 rad/s",
)
parser.add_argument(
    "--policy-max-vx-rate",
    type=float,
    default=2.0,
    help="ROS 模式 vx 最大变化率，单位 m/s^2",
)
parser.add_argument(
    "--policy-max-vy-rate",
    type=float,
    default=1.5,
    help="ROS 模式 vy 最大变化率，单位 m/s^2",
)
parser.add_argument(
    "--policy-max-wz-rate",
    type=float,
    default=2.5,
    help="ROS 模式 wz 最大变化率，单位 rad/s^2",
)
parser.add_argument(
    "--robotlab-repo",
    type=str,
    default=str(DEFAULT_ROBOTLAB_REPO),
    help="go2_rl_robotlab 仓库根目录",
)
parser.add_argument(
    "--policy",
    type=str,
    default="",
    help="导出的 TorchScript policy；留空时使用仓库预训练模型",
)
parser.add_argument(
    "--stair-height",
    type=float,
    default=0.19,
    help="确定性楼梯单级高度，单位 m；0.19 与 stairs_and_slope.xml 一致",
)
parser.add_argument(
    "--stair-width",
    type=float,
    default=0.30,
    help="确定性楼梯单级深度，单位 m",
)
parser.add_argument(
    "--multifloor-scene-usd",
    type=str,
    default=str(DEFAULT_MULTIFLOOR_SCENE_USD),
    help="与 PCT PLY/tomogram 配准的 multifloor 场景 USDA",
)
parser.add_argument(
    "--multifloor-collision-prim-path",
    type=str,
    default=DEFAULT_MULTIFLOOR_COLLISION_PRIM_PATH,
    help="场景中提供真实 PhysX 碰撞几何的 prim 路径",
)
parser.add_argument(
    "--multifloor-floor-proxy-profile",
    choices=("none", "yinluyuan_f2"),
    default="yinluyuan_f2",
    help="二楼平滑碰撞支撑面 profile；none 表示只使用原始 collision mesh",
)
parser.add_argument(
    "--multifloor-ray-origin-offset-z",
    type=float,
    default=1.0,
    help=(
        "multifloor 高度射线原点相对 base 的 z 偏移，单位 m；"
        "禁止沿用单层高度场的 20m 高空原点"
    ),
)
parser.add_argument(
    "--multifloor-start-x",
    type=float,
    default=DEFAULT_MULTIFLOOR_START_XYZ[0],
    help="标准 Go2 初始 base x，单位 m",
)
parser.add_argument(
    "--multifloor-start-y",
    type=float,
    default=DEFAULT_MULTIFLOOR_START_XYZ[1],
    help="标准 Go2 初始 base y，单位 m",
)
parser.add_argument(
    "--multifloor-start-z",
    type=float,
    default=DEFAULT_MULTIFLOOR_START_XYZ[2],
    help="标准 Go2 初始 base z，单位 m；已经包含 0.338m body height",
)
parser.add_argument(
    "--multifloor-start-yaw",
    type=float,
    default=DEFAULT_MULTIFLOOR_START_YAW,
    help="标准 Go2 初始 world yaw，单位 rad",
)
parser.add_argument(
    "--multifloor-goal-x",
    type=float,
    default=DEFAULT_MULTIFLOOR_GOAL_XYZ[0],
    help="PCT 二楼目标 base x，单位 m",
)
parser.add_argument(
    "--multifloor-goal-y",
    type=float,
    default=DEFAULT_MULTIFLOOR_GOAL_XYZ[1],
    help="PCT 二楼目标 base y，单位 m",
)
parser.add_argument(
    "--multifloor-goal-z",
    type=float,
    default=DEFAULT_MULTIFLOOR_GOAL_XYZ[2],
    help="PCT 二楼目标 base z，单位 m；不是地面高度",
)
parser.add_argument(
    "--multifloor-goal-yaw",
    type=float,
    default=DEFAULT_MULTIFLOOR_GOAL_YAW,
    help="PCT 二楼目标 world yaw，单位 rad",
)
parser.add_argument(
    "--pct-goal-delay-steps",
    type=int,
    default=100,
    help="multifloor 启动后等待多少个 policy 控制步再发布目标",
)
parser.add_argument(
    "--pct-goal-retry-interval-steps",
    type=int,
    default=50,
    help="尚未收到 Path 时，按原 stamp 重发 PCT 目标的控制步间隔",
)
parser.add_argument(
    "--pct-goal-max-attempts",
    type=int,
    default=3,
    help="PCT 目标最多传输次数，包含第一次发布",
)
parser.add_argument(
    "--vx",
    type=float,
    default=0.80,
    help="键盘前后速度幅值，单位 m/s",
)
parser.add_argument(
    "--vy",
    type=float,
    default=0.60,
    help="键盘横向速度幅值，单位 m/s",
)
parser.add_argument(
    "--wz",
    type=float,
    default=0.80,
    help="键盘偏航角速度幅值，单位 rad/s",
)
parser.add_argument(
    "--smoke-vx",
    type=float,
    default=0.0,
    help="无界面 smoke 使用的恒定前进速度，单位 m/s",
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=50,
    help="reset 后保持零命令并把 action 从零渐入的控制步数",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="最多运行的 policy 控制步数；0 表示一直运行",
)
parser.add_argument(
    "--status-every",
    type=int,
    default=50,
    help="每隔多少个控制步打印一次位姿和速度；0 表示关闭",
)
parser.add_argument("--seed", type=int, default=0, help="Isaac 环境随机种子")
parser.add_argument(
    "--real-time",
    action="store_true",
    help="尽量按照真实时间运行，便于键盘操作",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not 0.05 <= args_cli.stair_height <= 0.257:
    parser.error("--stair-height 应位于 policy 训练范围 [0.05, 0.257] m")
if args_cli.stair_width <= 0.0:
    parser.error("--stair-width 必须大于 0")
if min(args_cli.vx, args_cli.vy, args_cli.wz) < 0.0:
    parser.error("--vx、--vy 和 --wz 必须为非负数")
if abs(args_cli.smoke_vx) > 1.0:
    parser.error("楼梯上的 --smoke-vx 必须位于 [-1.0, 1.0] m/s")
if args_cli.warmup_steps < 0 or args_cli.max_steps < 0:
    parser.error("--warmup-steps 和 --max-steps 不能为负数")
if args_cli.status_every < 0:
    parser.error("--status-every 不能为负数")
if args_cli.navigation_planning_only:
    if args_cli.terrain_mode != "multifloor":
        parser.error("--navigation-planning-only 只能用于 --terrain-mode multifloor")
    if args_cli.command_source != "ros2":
        parser.error("--navigation-planning-only 必须同时使用 --command-source ros2")
if args_cli.terrain_mode == "multifloor":
    scene_usd = Path(args_cli.multifloor_scene_usd).expanduser().resolve()
    if not scene_usd.is_file():
        parser.error(f"--multifloor-scene-usd 不存在: {scene_usd}")
    if not args_cli.multifloor_collision_prim_path.startswith("/"):
        parser.error("--multifloor-collision-prim-path 必须是绝对 prim 路径")
    multifloor_numeric_values = (
        args_cli.multifloor_start_x,
        args_cli.multifloor_start_y,
        args_cli.multifloor_start_z,
        args_cli.multifloor_start_yaw,
        args_cli.multifloor_goal_x,
        args_cli.multifloor_goal_y,
        args_cli.multifloor_goal_z,
        args_cli.multifloor_goal_yaw,
        args_cli.multifloor_ray_origin_offset_z,
    )
    if not all(math.isfinite(value) for value in multifloor_numeric_values):
        parser.error("multifloor 起点和目标必须全部为有限数")
    if not 0.30 <= args_cli.multifloor_ray_origin_offset_z <= 2.00:
        parser.error(
            "--multifloor-ray-origin-offset-z 必须位于 [0.30, 2.00] m"
        )
if args_cli.pct_goal_delay_steps < 1:
    parser.error("--pct-goal-delay-steps 必须大于 0")
if args_cli.pct_goal_retry_interval_steps < 1:
    parser.error("--pct-goal-retry-interval-steps 必须大于 0")
if args_cli.pct_goal_max_attempts < 1:
    parser.error("--pct-goal-max-attempts 必须大于 0")
if not args_cli.cmd_vel_topic.startswith("/"):
    parser.error("--cmd-vel-topic 必须是以 / 开头的绝对 topic")
if not args_cli.reference_path_topic.startswith("/"):
    parser.error("--reference-path-topic 必须是以 / 开头的绝对 topic")
if not args_cli.stair_execution_frozen_topic.startswith("/"):
    parser.error(
        "--stair-execution-frozen-topic 必须是以 / 开头的绝对 topic"
    )
if args_cli.ros_domain_id is not None and not 0 <= args_cli.ros_domain_id <= 232:
    parser.error("--ros-domain-id 必须位于 [0, 232]")
if args_cli.cmd_vel_timeout <= 0.0:
    parser.error("--cmd-vel-timeout 必须大于 0")
if min(
    args_cli.policy_max_vx,
    args_cli.policy_max_vy,
    args_cli.policy_max_wz,
    args_cli.policy_max_vx_rate,
    args_cli.policy_max_vy_rate,
    args_cli.policy_max_wz_rate,
) <= 0.0:
    parser.error("ROS 速度限幅与变化率必须全部大于 0")
if args_cli.command_source == "ros2" and not args_cli.real_time:
    parser.error("ROS 2 实时命令模式必须同时指定 --real-time")

if args_cli.command_source == "ros2":
    # 现有 OGN bridge 同时创建 NavigationStatus 动态接口。这里虽然只做底层
    # Twist 链验证，Isaac 进程仍必须能发现工作区内的自定义消息共享库。
    validate_isaac_ros2_custom_message_environment()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim 启动后再导入这些模块。
import torch  # noqa: E402  # 必须等待 SimulationApp 启动后再导入

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402

from source.navigation.adapters.terrain_utils import (  # noqa: E402
    write_collision_terrain_wrapper,
)
from source.navigation.cmd_vel_to_policy import (  # noqa: E402
    CmdVelToPolicyAdapter,
    CmdVelToPolicyConfig,
)
from source.navigation.isaac_ros2_ogn_bridge import (  # noqa: E402
    IsaacRos2OgnBridge,
    IsaacRos2OgnBridgeConfig,
    enable_ros2_bridge_extension,
)


ROS_COMMAND_OWNER = "go2_moe_cts_ros_cmd_vel"


def _tensor_vector(
    values,
    *,
    expected_size: int,
    field_name: str,
) -> tuple[float, ...]:
    """
    @brief 将单机器人 GPU 张量转换为固定长度的 CPU 浮点元组
    @param values 待转换的 PyTorch 张量
    @param expected_size 期望的元素数量
    @param field_name 用于错误信息的字段名称
    @return 包含固定数量浮点数的元组
    """

    flattened = values.detach().reshape(-1)
    actual_size = int(flattened.shape[0])

    if actual_size != expected_size:
        raise RuntimeError(
            f"{field_name}维度错误："
            f"期望{expected_size}，实际{actual_size}"
        )

    if not bool(torch.isfinite(flattened).all().item()):
        raise RuntimeError(f"{field_name}包含 NaN 或 Inf")

    return tuple(
        float(value)
        for value in flattened.cpu().tolist()
    )


def _publish_navigation_odometry(
    env,
    bridge: IsaacRos2OgnBridge,
    *,
    timestamp: float,
) -> None:
    """
    @brief 发布当前 Go2 根状态对应的一帧导航里程计
    @param env 当前运行的单机器人 Isaac Lab 环境
    @param bridge 负责向 ROS 2 发布消息的 OGN bridge
    @param timestamp 当前连续仿真时间，单位为秒
    @return 无返回值
    """
    robot = env.scene["robot"]

    bridge.update_odometry(
        position=_tensor_vector(
            robot.data.root_pos_w[0],
            expected_size=3,
            field_name="root_pos_w",
        ),
        orientation_wxyz=_tensor_vector(
            robot.data.root_quat_w[0],
            expected_size=4,
            field_name="root_quat_w",
        ),
        linear_velocity=_tensor_vector(
            robot.data.root_lin_vel_b[0],
            expected_size=3,
            field_name="root_lin_vel_b",
        ),
        angular_velocity=_tensor_vector(
            robot.data.root_ang_vel_b[0],
            expected_size=3,
            field_name="root_ang_vel_b",
        ),
        timestamp=timestamp,
    )


def _publish_navigation_point_cloud(
    env,
    bridge: IsaacRos2OgnBridge,
    *,
    timestamp: float,
) -> int:
    """
    @brief 发布 height_scanner 当前检测到的世界坐标系地形点云
    @param env 当前运行的单机器人 Isaac Lab 环境
    @param bridge 负责向 ROS 2 发布消息的 OGN bridge
    @param timestamp 当前连续仿真时间，单位为秒
    @return 实际发布的点数；没有有效命中点时返回 0
    """

    ray_hits_w = env.scene["height_scanner"].data.ray_hits_w[0].detach()

    if ray_hits_w.ndim != 2 or int(ray_hits_w.shape[1]) != 3:
        raise RuntimeError(
            "height_scanner.ray_hits_w 维度错误："
            f"期望 (N, 3)，实际 {tuple(ray_hits_w.shape)}"
        )

    finite_mask = torch.isfinite(ray_hits_w).all(dim=-1)
    points_w = ray_hits_w[finite_mask]
    point_count = int(points_w.shape[0])

    if point_count == 0:
        return 0

    bridge.update_point_cloud(
        points_w.cpu().numpy(),
        timestamp=timestamp,
    )

    return point_count


def _publish_navigation_unfrozen_state(
    bridge: IsaacRos2OgnBridge,
    *,
    timestamp: float,
) -> int | None:
    """
    @brief 为当前合法 Path 发布标准 Go2 未冻结心跳
    @param bridge 负责 Path 订阅和楼梯执行状态发布的 OGN bridge
    @param timestamp 当前连续仿真时间，单位为秒
    @return 心跳绑定的 Path 整数纳秒时间戳；尚无合法 Path 时返回 None
    """

    active_path_stamp_ns = bridge.active_reference_path_stamp_ns
    if active_path_stamp_ns <= 0:
        return None

    report = bridge.publish_stair_execution_frozen(
        False,
        timestamp=timestamp,
    )
    if report.frozen:
        raise RuntimeError("标准 Go2 导航支线禁止发布 frozen=true")
    if report.reference_path_stamp_ns != active_path_stamp_ns:
        raise RuntimeError("未冻结心跳没有绑定当前 active Path identity")
    return report.reference_path_stamp_ns


def _resolve_external_paths() -> tuple[Path, Path]:
    """解析外部 robot_lab 源码和 TorchScript 路径。"""

    repository = Path(args_cli.robotlab_repo).expanduser().resolve()
    source_directory = repository / "source" / "robot_lab"
    if not (source_directory / "robot_lab" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"找不到 go2_rl_robotlab Python 包：{source_directory}"
        )

    policy_path = (
        Path(args_cli.policy).expanduser().resolve()
        if args_cli.policy
        else repository / DEFAULT_POLICY_RELATIVE_PATH
    )
    if not policy_path.is_file():
        raise FileNotFoundError(f"找不到 TorchScript policy：{policy_path}")
    return source_directory, policy_path


def _import_external_task(source_directory: Path):
    """从用户指定仓库导入 Go2 环境，避免误用同名 robot_lab 包。"""

    sys.path.insert(0, str(source_directory))

    import robot_lab
    from robot_lab.tasks.go2.env.go2_env import Go2Env
    from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, JOINT_NAMES

    imported_path = Path(robot_lab.__file__).resolve()
    expected_package = (source_directory / "robot_lab").resolve()
    if expected_package not in imported_path.parents:
        raise RuntimeError(
            "导入了错误的 robot_lab："
            f"期望位于 {expected_package}，实际为 {imported_path}"
        )
    if tuple(JOINT_NAMES) != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            "Go2 关节顺序与 TorchScript/MuJoCo 合同不一致："
            f"{tuple(JOINT_NAMES)}"
        )
    return Go2EnvCfg, Go2Env, imported_path


def _disable_evaluation_randomization(env_cfg) -> None:
    """关闭训练期随机化，并保留可重复的确定性 reset。"""

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.single_obs.enable_corruption = False

    for event_name in (
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_actuator_gains",
        "randomize_motor_zero_offset",
        "randomize_push_robot",
        "randomize_rigid_body_material",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    # 不能简单删除 reset 事件，否则跌倒后的自动 reset 可能没有明确恢复 root。
    env_cfg.events.reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    for curriculum_name in (
        "terrain_levels",
        "base_linear_velocity",
        "base_height_l2",
    ):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)

    # MuJoCo 验证配置使用 delay_min=delay_max=0；Isaac 比较也固定为零延迟。
    for actuator in env_cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = 0
        if hasattr(actuator, "max_delay"):
            actuator.max_delay = 0


def _inspect_multifloor_collision(
    scene_usd: Path,
    collision_prim_path: str,
) -> dict[str, object]:
    """
    @brief 在创建 Isaac Lab 环境前验证 multifloor 碰撞子树
    @param scene_usd               待加载的场景 USDA/USD 路径
    @param collision_prim_path     场景内碰撞根 prim 的绝对路径
    @return                        碰撞根类型、Mesh 数和 CollisionAPI 数
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"无法打开 multifloor 场景：{scene_usd}")

    collision_root = stage.GetPrimAtPath(collision_prim_path)
    if not collision_root.IsValid():
        raise RuntimeError(
            "multifloor collision prim 不存在："
            f"{collision_prim_path} in {scene_usd}"
        )

    mesh_count = 0
    collision_api_count = 0
    for prim in Usd.PrimRange(collision_root):
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
    if mesh_count == 0:
        raise RuntimeError(
            "multifloor collision payload 没有 Mesh；"
            "请检查 source/scene/multifloor 资产是否完整"
        )

    return {
        "scene_usd": str(scene_usd),
        "collision_prim_path": collision_prim_path,
        "collision_root_type": collision_root.GetTypeName(),
        "collision_root_is_mesh": bool(collision_root.IsA(UsdGeom.Mesh)),
        "mesh_count": mesh_count,
        "collision_api_count": collision_api_count,
    }


def _configure_multifloor_terrain(env_cfg) -> dict[str, object]:
    """
    @brief 用项目真实 multifloor collision USD 替换程序生成楼梯
    @param env_cfg                 外部 RobotLab 的单机器人 Go2 环境配置
    @return                        场景资产、RayCaster 和初始位姿审计信息
    """

    scene_usd = Path(args_cli.multifloor_scene_usd).expanduser().resolve()
    collision_report = _inspect_multifloor_collision(
        scene_usd,
        args_cli.multifloor_collision_prim_path,
    )
    floor_proxy_profile = (
        None
        if args_cli.multifloor_floor_proxy_profile == "none"
        else args_cli.multifloor_floor_proxy_profile
    )
    terrain_usd = write_collision_terrain_wrapper(
        scene_usd,
        args_cli.multifloor_collision_prim_path,
        floor_proxy_profile=floor_proxy_profile,
        source_prim_is_mesh=bool(
            collision_report["collision_root_is_mesh"]
        ),
    )

    original_terrain = env_cfg.scene.terrain
    terrain_prim_path = "/World/multifloor_collision"
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path=terrain_prim_path,
        terrain_type="usd",
        usd_path=str(terrain_usd),
        collision_group=-1,
        physics_material=original_terrain.physics_material,
        debug_vis=False,
    )
    env_cfg.sim.physics_material = original_terrain.physics_material
    env_cfg.scene.env_spacing = 0.0

    # TerrainImporter 把 wrapper default prim 生成为 <prim_path>/terrain。
    # 两个高度扫描器必须跟随新碰撞地形；否则 policy 和 ROS 点云仍会查询
    # 已经不存在的 /World/ground。
    terrain_mesh_prim_path = f"{terrain_prim_path}/terrain"
    updated_scanners: list[str] = []
    for sensor_name in ("height_scanner", "height_scanner_small"):
        sensor_cfg = getattr(env_cfg.scene, sensor_name, None)
        if sensor_cfg is None:
            continue
        sensor_cfg.mesh_prim_paths = [terrain_mesh_prim_path]
        # RobotLab 单层高度场从 base 上方 20m 向下投射。在封闭多楼层
        # mesh 中它会先命中屋顶（当前资产实测 z≈6~7m），而不是机器人
        # 所在楼层。局部原点仍高于楼梯踏面，同时位于上一层楼板之下。
        sensor_cfg.offset.pos = (
            0.0,
            0.0,
            float(args_cli.multifloor_ray_origin_offset_z),
        )
        updated_scanners.append(sensor_name)
    if "height_scanner" not in updated_scanners:
        raise RuntimeError("multifloor 导航要求启用 height_scanner")

    start_yaw = float(args_cli.multifloor_start_yaw)
    half_yaw = 0.5 * start_yaw
    env_cfg.scene.robot.init_state.pos = (
        float(args_cli.multifloor_start_x),
        float(args_cli.multifloor_start_y),
        float(args_cli.multifloor_start_z),
    )
    env_cfg.scene.robot.init_state.rot = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )

    return {
        **collision_report,
        "terrain_wrapper": str(terrain_usd),
        "terrain_prim_path": terrain_prim_path,
        "terrain_mesh_prim_path": terrain_mesh_prim_path,
        "floor_proxy_profile": floor_proxy_profile,
        "updated_scanners": tuple(updated_scanners),
        "ray_origin_offset_z": float(
            args_cli.multifloor_ray_origin_offset_z
        ),
        "start_base_xyz": tuple(env_cfg.scene.robot.init_state.pos),
        "start_yaw": start_yaw,
    }


def _configure_multifloor_command_term(env_cfg) -> None:
    """
    @brief 为 USD 单地形替换 RobotLab 仅支持 generator 的命令项
    @param env_cfg                 外部 RobotLab 的单机器人 Go2 环境配置
    @return                        无返回值
    """

    # RobotLab 的 Go2RLGymCommand 在构造阶段无条件枚举
    # terrain_generator.sub_terrains。USD TerrainImporter 没有 generator，
    # 因此这里只把命令存储换成 Isaac Lab 标准三维速度 buffer。policy 仍从
    # 同名 base_velocity 读取相同顺序的 [vx, vy, wz]，TorchScript 合同不变。
    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        debug_vis=False,
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


def _configure_deterministic_stairs(env_cfg) -> None:
    """只生成一块固定高度的上楼梯地形。"""

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError("Go2 环境没有启用 terrain generator")

    stair_cfg = terrain_generator.sub_terrains["stairs_up"]
    stair_cfg.proportion = 1.0
    stair_cfg.step_height_range = (
        args_cli.stair_height,
        args_cli.stair_height,
    )
    stair_cfg.step_width = args_cli.stair_width
    stair_cfg.platform_width = 3.0

    # IsaacLab 的高度场使用 int(step_width / horizontal_scale) 向下离散。
    # 例如 0.30 / 0.10 在浮点数下略小于 3，实际只得到 2 个栅格；
    # 0.31 才会稳定得到 3 个栅格，即真实 0.30 m 踏面。
    horizontal_scale = float(terrain_generator.horizontal_scale)
    tread_cells = int(args_cli.stair_width / horizontal_scale)
    if tread_cells <= 0:
        raise RuntimeError(
            "楼梯踏面离散后小于一个高度场栅格："
            f"requested={args_cli.stair_width:.6f}m, "
            f"horizontal_scale={horizontal_scale:.6f}m"
        )
    effective_tread_width = tread_cells * horizontal_scale
    print(
        "[INFO] 楼梯踏面离散："
        f"requested={args_cli.stair_width:.3f}m, "
        f"horizontal_scale={horizontal_scale:.3f}m, "
        f"cells={tread_cells}, effective={effective_tread_width:.3f}m"
    )

    terrain_generator.sub_terrains = {"stairs_up": stair_cfg}
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.curriculum = False
    terrain_generator.difficulty_range = (0.0, 0.0)
    env_cfg.scene.terrain.max_init_terrain_level = 0


def _configure_manual_commands(env_cfg) -> None:
    """禁止环境重采样速度，运行时直接覆写 command buffer。"""

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.debug_vis = not args_cli.headless
    if args_cli.terrain_mode == "multifloor":
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
        return

    command_cfg.dynamic_resample_commands = False
    command_cfg.resampling_time = 1.0e9
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.limit_vel_prob = 0.0
    command_cfg.zero_command_curriculum = None
    command_cfg.command_range_curriculum = []
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_yaw = (0.0, 0.0)


def _create_environment(Go2EnvCfg, Go2Env):
    """
    @brief 创建与 MuJoCo 部署合同一致的单机器人 Isaac 环境
    @param Go2EnvCfg               外部 RobotLab 的环境配置类型
    @param Go2Env                  外部 RobotLab 的环境实现类型
    @return                        环境、初始观测和地形审计报告
    """

    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1_000_000.0
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.terminations.time_out = None

    if args_cli.terrain_mode == "multifloor":
        terrain_report = _configure_multifloor_terrain(env_cfg)
        _configure_multifloor_command_term(env_cfg)
    else:
        _configure_deterministic_stairs(env_cfg)
        terrain_report = {
            "terrain_mode": "deterministic_stairs",
            "stair_height_m": float(args_cli.stair_height),
            "requested_stair_width_m": float(args_cli.stair_width),
        }
    _disable_evaluation_randomization(env_cfg)
    _configure_manual_commands(env_cfg)

    env = Go2Env(cfg=env_cfg)
    observations, _ = env.reset()
    return env, observations, terrain_report


def _load_and_verify_policy(policy_path: Path, env, observations):
    """加载 TorchScript，并验证观测、历史和动作接口。"""

    if env.action_manager.total_action_dim != POLICY_CONTRACT.action_dim:
        raise RuntimeError(
            "MoE-CTS action 维度不匹配："
            f"期望 {POLICY_CONTRACT.action_dim}，实际 "
            f"{env.action_manager.total_action_dim}"
        )

    policy_adapter = MoeCtsPolicyAdapter(
        policy_path,
        device=env.device,
        contract=POLICY_CONTRACT,
    )
    verification = policy_adapter.verify_observations(observations)
    print(
        "[INFO] MoE-CTS 接口验证通过："
        f"single_obs={verification.observation_shape[-1]}, "
        f"history={verification.history_shape[-1]}, "
        f"action={verification.action_shape[-1]}"
    )
    return policy_adapter


def _create_ros_command_path(command_buffer):
    """
    @brief 建立 ROS 2 观测、规划接口和可选的 MoE-CTS command buffer 链
    @param command_buffer 标准 Go2 的 base_velocity command buffer
    @return OGN bridge、可选命令 sink 和可选速度安全门
    """

    command_subscription_enabled = not args_cli.navigation_planning_only
    bridge = IsaacRos2OgnBridge(
        IsaacRos2OgnBridgeConfig(
            graph_path="/World/Go2MoeCtsCmdVelBridge",
            # 此图只服务当前进程的速度输入，不需要持久化进 USD。外部 Go2
            # 环境 reset 后的 stage 会拒绝新增 graph prim，非 USD 运行时图
            # 可以避开该无关限制，同时保留相同的 DDS/OGN 数据路径。
            graph_backed_by_usd=False,
            command_topic=args_cli.cmd_vel_topic,
            reference_path_topic=args_cli.reference_path_topic,
            stair_execution_frozen_topic=(
                args_cli.stair_execution_frozen_topic
            ),
            domain_id=args_cli.ros_domain_id,
            enable_command_subscription=command_subscription_enabled,

            # 标准 Go2 + MoE-CTS 不使用 root-lock 跨楼梯，但仍要订阅当前
            # Path，并发布与其代际绑定的 frozen=false 心跳。这样 SCAN
            # 可以区分“明确未冻结”和“冻结状态发布器失活”。
            enable_reference_path_subscription=True,
            enable_stair_execution_frozen_publisher=True,
            # goal_reached 使用 volatile 订阅，只观察当前运行的持续状态；
            # 到达后停止为已经完成的旧 Path 继续发送冻结心跳。
            enable_goal_reached_subscription=command_subscription_enabled,
            enable_pct_goal_publisher=(
                args_cli.terrain_mode == "multifloor"
            ),
        )
    )
    bridge.setup()

    if not command_subscription_enabled:
        return bridge, None, None

    sink = MoeCtsCommandBufferSink(command_buffer)
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx=args_cli.policy_max_vx,
            max_vy=args_cli.policy_max_vy,
            max_wz=args_cli.policy_max_wz,
            max_vx_rate=args_cli.policy_max_vx_rate,
            max_vy_rate=args_cli.policy_max_vy_rate,
            max_wz_rate=args_cli.policy_max_wz_rate,
            cmd_vel_timeout_s=args_cli.cmd_vel_timeout,
            require_odometry=False,
            require_point_cloud=False,
            require_navigation_status=False,
        ),
        ownership_resource=("go2_moe_cts_base_velocity", 0),
    )
    return bridge, sink, gate


def _print_status(
    env,
    step: int,
    command: torch.Tensor,
    reset_count: int,
    initial_position: torch.Tensor,
) -> None:
    """打印单机器人当前位姿和机体系速度。"""

    robot = env.scene["robot"]
    position = robot.data.root_pos_w[0]
    displacement = position - initial_position
    linear_velocity = robot.data.root_lin_vel_b[0]
    angular_velocity = robot.data.root_ang_vel_b[0]
    print(
        "[STATUS] "
        f"step={step} "
        f"cmd=({command[0].item():+.2f},"
        f"{command[1].item():+.2f},{command[2].item():+.2f}) "
        f"pos=({position[0].item():+.2f},"
        f"{position[1].item():+.2f},{position[2].item():+.2f}) "
        f"dpos=({displacement[0].item():+.2f},"
        f"{displacement[1].item():+.2f},{displacement[2].item():+.2f}) "
        f"vel=({linear_velocity[0].item():+.2f},"
        f"{linear_velocity[1].item():+.2f},"
        f"{angular_velocity[2].item():+.2f}) "
        f"resets={reset_count}"
    )


def main() -> None:
    """
    @brief 运行确定性楼梯控制或 multifloor 真实观测只规划循环
    @return 无返回值
    """

    source_directory, policy_path = _resolve_external_paths()
    Go2EnvCfg, Go2Env, imported_path = _import_external_task(source_directory)
    print(f"[INFO] 外部 robot_lab：{imported_path}")
    print(f"[INFO] TorchScript policy：{policy_path}")

    env = None
    command_buffer = None
    command_gate = None
    command_gate_claimed = False
    last_sim_time = 0.0
    try:
        if args_cli.command_source == "ros2":
            extension_report = enable_ros2_bridge_extension()
            print(
                "[INFO] Isaac ROS 2 bridge："
                f"enabled={extension_report['enabled']}"
            )
        env, observations, terrain_report = _create_environment(
            Go2EnvCfg,
            Go2Env,
        )
        print(
            "[INFO] 控制周期："
            f"physics_dt={env.physics_dt:.3f}s, "
            f"decimation={env.cfg.decimation}, step_dt={env.step_dt:.3f}s"
        )
        if args_cli.terrain_mode == "multifloor":
            print(
                "[INFO] multifloor collision 场景："
                f"scene={terrain_report['scene_usd']}, "
                f"meshes={terrain_report['mesh_count']}, "
                f"collision_apis={terrain_report['collision_api_count']}"
            )
            print(
                "[INFO] 标准 Go2 标定位姿："
                f"base_xyz={terrain_report['start_base_xyz']}, "
                f"yaw={terrain_report['start_yaw']:.6f}rad, "
                f"terrain={terrain_report['terrain_mesh_prim_path']}, "
                "ray_origin_offset_z="
                f"{terrain_report['ray_origin_offset_z']:.3f}m"
            )
        else:
            print(
                "[INFO] 确定性楼梯："
                f"height={args_cli.stair_height:.3f}m, "
                f"width={args_cli.stair_width:.3f}m"
            )
        policy_adapter = _load_and_verify_policy(policy_path, env, observations)

        command_buffer = env.command_manager.get_command("base_velocity")
        command_buffer.zero_()
        zero_command = torch.zeros(3, device=env.device)
        headless_command = torch.tensor(
            [args_cli.smoke_vx, 0.0, 0.0],
            dtype=torch.float32,
            device=env.device,
        )

        keyboard = None
        command_bridge = None
        command_sink = None
        if args_cli.command_source == "ros2":
            command_bridge, command_sink, command_gate = (
                _create_ros_command_path(command_buffer)
            )
            # OGN 图创建会调用若干次 Kit update。此时 policy 循环尚未接管，
            # 物理可能已让机器人离开标定起点。图建好后统一 reset，保证首帧
            # ROS Odometry、PCT 起点和 CTS 历史属于同一个 episode。
            observations, _ = env.reset()
            reset_command_buffer = env.command_manager.get_command(
                "base_velocity"
            )
            if reset_command_buffer.data_ptr() != command_buffer.data_ptr():
                raise RuntimeError(
                    "环境 reset 替换了 base_velocity command buffer；"
                    "现有 ROS command sink 将失去唯一写入目标"
                )
            command_buffer = reset_command_buffer
            command_buffer.zero_()
            policy_adapter.reset()
            print("[INFO] OGN 初始化后已恢复 multifloor 标定起点并清空 CTS 历史")
            last_sim_time = env.step_dt
            if command_gate is not None:
                command_gate.claim(ROS_COMMAND_OWNER, last_sim_time)
                command_gate_claimed = True
            domain_text = (
                "ROS_DOMAIN_ID 环境变量"
                if args_cli.ros_domain_id is None
                else str(args_cli.ros_domain_id)
            )
            if args_cli.navigation_planning_only:
                print(
                    "[INFO] multifloor 只规划安全模式已启用："
                    f"domain={domain_text}；不创建 {args_cli.cmd_vel_topic} "
                    "订阅，policy command buffer 每拍保持零。"
                )
                print(
                    "[INFO] 已启用真实 Odometry、地形 RayCaster 点云、"
                    "PCT goal 和 Path 绑定的 frozen=false 发布。"
                )
            else:
                print(
                    "[INFO] ROS 2 速度入口已启用："
                    f"topic={args_cli.cmd_vel_topic}, domain={domain_text}, "
                    f"timeout={args_cli.cmd_vel_timeout:.2f}s"
                )
                print(
                    "[INFO] Twist→policy、Odometry 与原始地形点云发布已启用；"
                    "标准 Go2 将对当前 Path 持续发布 frozen=false；"
                    "supervisor 许可将在接入 SCAN 完整链时启用。"
                )
            print(
                "[INFO] SCAN Path/楼梯状态接口："
                f"path={args_cli.reference_path_topic}, "
                "freeze="
                f"{args_cli.stair_execution_frozen_topic}"
            )
        elif not args_cli.headless:
            keyboard = Se2Keyboard(
                Se2KeyboardCfg(
                    v_x_sensitivity=args_cli.vx,
                    v_y_sensitivity=args_cli.vy,
                    omega_z_sensitivity=args_cli.wz,
                    sim_device=env.device,
                )
            )
            print(keyboard)
            print("[INFO] 点击 Viewport 后使用方向键和 Z/X；按 L 紧急清零。")
            print("[INFO] 初始低平台沿 +X 方向直行即为上楼。")
        else:
            print(
                "[INFO] headless 命令："
                f"vx={args_cli.smoke_vx:.2f}m/s"
            )

        completed_steps = 0
        warmup_step = 0
        reset_count = 0
        initial_position = env.scene["robot"].data.root_pos_w[0].clone()
        max_relative_height = 0.0
        received_twist_count = 0
        last_unfrozen_path_stamp_ns = 0
        last_stop_reasons: tuple[str, ...] | None = None
        pct_goal_transport_attempts = 0
        last_pct_goal_attempt_step = 0
        active_pct_goal_stamp_ns = 0
        pct_path_received = False
        pct_goal_retry_exhausted_reported = False
        goal_reached_armed_path_stamp_ns = 0
        goal_reached_path_stamp_ns = 0
        print(
            f"[INFO] 每次 reset 后先执行 {args_cli.warmup_steps} 步渐入。"
        )

        while simulation_app.is_running():
            step_started = time.time()
            in_warmup = warmup_step < args_cli.warmup_steps
            if args_cli.command_source == "ros2":
                assert command_bridge is not None
                last_sim_time = max(
                    env.step_dt,
                    float(completed_steps + 1) * env.step_dt,
                )
                # 每个控制周期轮询一次 Path。只有收到新代际时才返回消息；
                # 重复轮询不会反复打印，也不会改变已认证的 Path identity。
                path_sample = command_bridge.poll_reference_path()

                if path_sample is not None:
                    # transient-local 可能在本次 goal 发布前先交付上一代缓存
                    # Path 或清空 tombstone。只有本进程已发布 goal 后收到的
                    # 非空 Path 才能终止同 stamp 的有限重试。
                    path_belongs_to_active_goal = (
                        active_pct_goal_stamp_ns > 0
                        and bool(path_sample.points_ground_xyz)
                        and path_sample.stamp_ns >= active_pct_goal_stamp_ns
                    )
                    if path_belongs_to_active_goal:
                        pct_path_received = True
                    if path_sample.points_ground_xyz:
                        # 新 Path 必须先观察到 controller 的 false，才允许随后
                        # 的 true 终止该代冻结心跳，避免上一代 true 跨代复用。
                        goal_reached_armed_path_stamp_ns = 0
                        goal_reached_path_stamp_ns = 0
                    print(
                        "[PATH] 已接收 reference Path："
                        f"points={len(path_sample.points_ground_xyz)}, "
                        f"stamp_ns={path_sample.stamp_ns}, "
                        f"sequence={path_sample.sequence}, "
                        f"active_goal_match={path_belongs_to_active_goal}"
                    )

                if args_cli.navigation_planning_only:
                    # 只规划模式没有 Twist subscriber，也没有 command gate。
                    # 仍然每拍显式清零，避免外部环境的 command manager 重采样
                    # 或 reset 把非零命令带入 policy 观测。
                    command_buffer.zero_()
                else:
                    assert command_gate is not None
                    goal_reached_sample = command_bridge.poll_goal_reached(
                        receipt_timestamp=last_sim_time,
                    )
                    if goal_reached_sample is not None:
                        active_path_stamp_ns = (
                            command_bridge.active_reference_path_stamp_ns
                        )
                        if not goal_reached_sample.value:
                            goal_reached_armed_path_stamp_ns = (
                                active_path_stamp_ns
                            )
                            goal_reached_path_stamp_ns = 0
                        elif (
                            active_path_stamp_ns > 0
                            and goal_reached_armed_path_stamp_ns
                            == active_path_stamp_ns
                        ):
                            if (
                                goal_reached_path_stamp_ns
                                != active_path_stamp_ns
                            ):
                                print(
                                    "[GOAL] controller 已确认当前 Path 到达；"
                                    "停止旧代 frozen=false 心跳："
                                    f"path_stamp_ns={active_path_stamp_ns}"
                                )
                            goal_reached_path_stamp_ns = (
                                active_path_stamp_ns
                            )
                    command_gate.renew_control_lease(
                        ROS_COMMAND_OWNER,
                        last_sim_time,
                    )
                    sample = command_bridge.poll_twist(
                        receipt_timestamp=last_sim_time,
                    )
                    if in_warmup:
                        report = command_gate.inhibit(
                            owner_id=ROS_COMMAND_OWNER,
                            now=last_sim_time,
                            reason="policy_warmup",
                        )
                    else:
                        if sample is not None and sample.command_present:
                            command_gate.accept_cmd_vel(
                                (
                                    sample.linear_velocity[0],
                                    sample.linear_velocity[1],
                                    sample.angular_velocity[2],
                                ),
                                owner_id=ROS_COMMAND_OWNER,
                                received_at=sample.receipt_timestamp,
                            )
                            received_twist_count += 1
                        report = command_gate.write(
                            owner_id=ROS_COMMAND_OWNER,
                            now=last_sim_time,
                        )
                    if report.stop_reasons != last_stop_reasons:
                        state = (
                            "允许运动" if report.motion_allowed else "输出零速度"
                        )
                        reasons = ",".join(report.stop_reasons) or "none"
                        print(
                            f"[CMD_VEL] {state}；原因={reasons}；"
                            f"写入={report.written_command.as_tuple()}"
                        )
                        last_stop_reasons = report.stop_reasons
            else:
                keyboard_command = (
                    headless_command if keyboard is None else keyboard.advance()
                )
                requested_command = zero_command if in_warmup else keyboard_command
                command_buffer[:] = requested_command

            # env.step() 末尾会运行 command manager；每拍重新计算观测，确保
            # TorchScript 读取到刚刚覆写的速度命令。
            observations = env.observation_manager.compute(update_history=False)
            actions = policy_adapter.infer_from_observations(observations)
            if in_warmup and args_cli.warmup_steps > 0:
                warmup_scale = min(
                    1.0,
                    float(warmup_step + 1) / float(args_cli.warmup_steps),
                )
                actions = actions * warmup_scale
            _, _, terminated, truncated, _ = env.step(actions)

            if command_bridge is not None:
                _publish_navigation_odometry(
                    env,
                    command_bridge,
                    timestamp=last_sim_time,
                )

                point_count: int = _publish_navigation_point_cloud(
                    env,
                    command_bridge,
                    timestamp=last_sim_time,
                )

                if args_cli.terrain_mode == "multifloor":
                    current_step = completed_steps + 1
                    first_goal_due = (
                        pct_goal_transport_attempts == 0
                        and current_step >= args_cli.pct_goal_delay_steps
                    )
                    retry_goal_due = (
                        pct_goal_transport_attempts > 0
                        and not pct_path_received
                        and pct_goal_transport_attempts
                        < args_cli.pct_goal_max_attempts
                        and current_step - last_pct_goal_attempt_step
                        >= args_cli.pct_goal_retry_interval_steps
                    )
                    if first_goal_due:
                        goal_sample = command_bridge.publish_pct_goal(
                            (
                                args_cli.multifloor_goal_x,
                                args_cli.multifloor_goal_y,
                                args_cli.multifloor_goal_z,
                            ),
                            args_cli.multifloor_goal_yaw,
                            stamp_ns=int(round(last_sim_time * 1.0e9)),
                            frame_id="world",
                        )
                        pct_goal_transport_attempts = 1
                        last_pct_goal_attempt_step = current_step
                        active_pct_goal_stamp_ns = goal_sample.stamp_ns
                        pct_path_received = False
                        print(
                            "[PCT_GOAL] 已发布 multifloor base 目标："
                            f"xyz={goal_sample.position_base_xyz}, "
                            f"yaw={goal_sample.yaw:.6f}, "
                            f"stamp_ns={goal_sample.stamp_ns}, attempt=1"
                        )
                    elif retry_goal_due:
                        goal_sample = command_bridge.republish_last_pct_goal()
                        pct_goal_transport_attempts += 1
                        last_pct_goal_attempt_step = current_step
                        print(
                            "[PCT_GOAL] 尚未收到 Path，保持原 stamp 重发："
                            f"stamp_ns={goal_sample.stamp_ns}, "
                            f"attempt={pct_goal_transport_attempts}"
                        )
                    elif (
                        not pct_path_received
                        and pct_goal_transport_attempts
                        >= args_cli.pct_goal_max_attempts
                        and not pct_goal_retry_exhausted_reported
                        and current_step - last_pct_goal_attempt_step
                        >= args_cli.pct_goal_retry_interval_steps
                    ):
                        print(
                            "[WARN] PCT 目标传输重试已用尽；"
                            "继续保持零速并等待 Path，请检查 ROS graph。"
                        )
                        pct_goal_retry_exhausted_reported = True

                # 使用已发布完成的上一控制拍时间，避免 /clock 与冻结状态跨
                # DDS writer 乱序时，SCAN 把 frozen=false 误判为来自未来。
                heartbeat_timestamp = max(
                    env.step_dt,
                    last_sim_time - env.step_dt,
                )
                active_path_stamp_ns = (
                    command_bridge.active_reference_path_stamp_ns
                )
                if (
                    active_path_stamp_ns > 0
                    and goal_reached_path_stamp_ns == active_path_stamp_ns
                ):
                    unfrozen_path_stamp_ns = None
                else:
                    unfrozen_path_stamp_ns = (
                        _publish_navigation_unfrozen_state(
                            command_bridge,
                            timestamp=heartbeat_timestamp,
                        )
                    )
                if unfrozen_path_stamp_ns is None:
                    last_unfrozen_path_stamp_ns = 0
                elif unfrozen_path_stamp_ns != last_unfrozen_path_stamp_ns:
                    print(
                        "[STAIR] 标准 Go2 保持非冻结规划："
                        f"path_stamp_ns={unfrozen_path_stamp_ns}"
                    )
                    last_unfrozen_path_stamp_ns = unfrozen_path_stamp_ns

                if completed_steps == 0:
                    first_hits = (
                        env.scene["height_scanner"]
                        .data.ray_hits_w[0]
                        .detach()
                    )
                    finite_hits = first_hits[
                        torch.isfinite(first_hits).all(dim=-1)
                    ]
                    hit_min = finite_hits.amin(dim=0)
                    hit_max = finite_hits.amax(dim=0)
                    base_position = (
                        env.scene["robot"].data.root_pos_w[0]
                    )
                    print(
                        "[INFO] 首帧 height_scanner 点云："
                        f"有效点数={point_count}, "
                        "world_min=("
                        f"{hit_min[0].item():+.3f},"
                        f"{hit_min[1].item():+.3f},"
                        f"{hit_min[2].item():+.3f}), "
                        "world_max=("
                        f"{hit_max[0].item():+.3f},"
                        f"{hit_max[1].item():+.3f},"
                        f"{hit_max[2].item():+.3f}), "
                        "base=("
                        f"{base_position[0].item():+.3f},"
                        f"{base_position[1].item():+.3f},"
                        f"{base_position[2].item():+.3f})"
                    )

            completed_steps += 1
            current_position = env.scene["robot"].data.root_pos_w[0]
            max_relative_height = max(
                max_relative_height,
                float((current_position[2] - initial_position[2]).item()),
            )
            done = bool(torch.any(terminated | truncated).item())
            if done:
                reset_count += 1
                print(
                    "[WARN] 机器人触发自动 reset；"
                    f"累计 reset={reset_count}，重新清空 CTS 历史。"
                )
                policy_adapter.reset()
                if command_gate is not None:
                    command_gate.reset(
                        owner_id=ROS_COMMAND_OWNER,
                        now=last_sim_time,
                    )
                else:
                    command_buffer.zero_()
                warmup_step = 0
                if keyboard is not None:
                    keyboard.reset()
            else:
                warmup_step += 1

            if (
                args_cli.status_every > 0
                and completed_steps % args_cli.status_every == 0
            ):
                _print_status(
                    env,
                    completed_steps,
                    command_buffer[0],
                    reset_count,
                    initial_position,
                )

            if args_cli.max_steps > 0 and completed_steps >= args_cli.max_steps:
                final_displacement = current_position - initial_position
                print(
                    "[INFO] 已完成有限步测试："
                    f"steps={completed_steps}, resets={reset_count}"
                )
                if command_sink is not None:
                    print(
                        "[RESULT] ROS 2 命令链："
                        f"twist_rx={received_twist_count}, "
                        f"buffer_writes={command_sink.write_count}"
                    )
                print(
                    "[RESULT] 相对位移："
                    f"dx={final_displacement[0].item():+.3f}m, "
                    f"dy={final_displacement[1].item():+.3f}m, "
                    f"dz={final_displacement[2].item():+.3f}m, "
                    f"max_dz={max_relative_height:+.3f}m"
                )
                break

            if args_cli.real_time:
                remaining = env.step_dt - (time.time() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if command_gate is not None and command_gate_claimed:
            command_gate.release(
                owner_id=ROS_COMMAND_OWNER,
                now=max(last_sim_time, 0.0),
            )
        if command_buffer is not None:
            command_buffer.zero_()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main_failed = False
    try:
        main()
    except BaseException:
        # SimulationApp.close() 可能在 Kit 关闭阶段结束解释器；必须先打印真实
        # Python 异常并保存非零退出状态，避免 OGN 初始化失败被误报为通过。
        main_failed = True
        traceback.print_exc()
    finally:
        simulation_app.close()
    if main_failed:
        raise SystemExit(1)
