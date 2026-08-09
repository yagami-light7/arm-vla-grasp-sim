"""启动 PCT、Isaac 传感器归一化、SCAN、控制器与 supervisor 主线。"""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_boolean(context, argument_name: str) -> bool:
    """严格解析 launch 布尔参数，避免拼写错误静默改变路径来源。"""

    value = LaunchConfiguration(argument_name).perform(context).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{argument_name} 必须是 true/false、1/0、yes/no 或 on/off"
    )


def _validate_path_source(context) -> list[object]:
    """禁止 PCT 与手工 Path 同时写入同一个全局路径 topic。"""

    if _launch_boolean(context, "start_pct") and _launch_boolean(
        context,
        "start_manual_path",
    ):
        raise RuntimeError(
            "start_pct 与 start_manual_path 不能同时为 true："
            "同一时刻只允许一个全局 Path 发布器"
        )
    return []


def _validate_body_height(context) -> list[object]:
    """要求四个高度消费者共享一个有限正数。"""

    raw_value = LaunchConfiguration("body_height_m").perform(context)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("body_height_m 必须是有限正数") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("body_height_m 必须是有限正数")
    return []


def _validate_stair_freeze_timing(context) -> list[object]:
    """要求楼梯冻结新鲜度和二次确认宽限均为有限正数。"""

    for argument_name in (
        "stair_execution_freeze_timeout_sec",
        "stair_execution_freeze_confirmation_sec",
    ):
        raw_value = LaunchConfiguration(argument_name).perform(context)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{argument_name} 必须是有限正数") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{argument_name} 必须是有限正数")
    return []


def _canonical_frame_argument(context, argument_name: str) -> str:
    """要求组合 launch 使用不含前导斜杠的唯一 frame 拼写。"""

    raw_value = LaunchConfiguration(argument_name).perform(context)
    value = raw_value.strip()
    components = value.split("/")
    if (
        not value
        or value != raw_value
        or value.startswith("/")
        or any(character.isspace() for character in value)
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise RuntimeError(
            f"{argument_name} 必须是无前导斜杠、无空白且层级完整的 frame_id"
        )
    return value


def _validate_frame_arguments(context) -> list[object]:
    """防止各节点因 frame 规范化规则不同而静默分裂。"""

    world_frame = _canonical_frame_argument(context, "world_frame")
    base_frame = _canonical_frame_argument(context, "base_frame")
    if world_frame == base_frame:
        raise RuntimeError("world_frame 与 base_frame 不能相同")
    return []


def generate_launch_description() -> LaunchDescription:
    """返回 PCT、Path 适配、桥接、SCAN、控制器与 supervisor 主线。"""

    default_bridge_config = PathJoinSubstitution(
        [
            FindPackageShare("isaac_navigation_bridge"),
            "config",
            "pct_scan.yaml",
        ]
    )
    default_scan_config = PathJoinSubstitution(
        [FindPackageShare("scan_planner"), "config", "planner.yaml"]
    )
    default_controller_config = PathJoinSubstitution(
        [FindPackageShare("scan_controller"), "config", "controller.yaml"]
    )
    default_tuning_config = PathJoinSubstitution(
        [
            FindPackageShare("isaac_navigation_bridge"),
            "config",
            "pct_scan_tuning.yaml",
        ]
    )
    default_manual_path_config = PathJoinSubstitution(
        [
            FindPackageShare("scan_navigation_tools"),
            "config",
            "manual_path.yaml",
        ]
    )
    default_pct_config = PathJoinSubstitution(
        [
            FindPackageShare("pct_ros2_adapter"),
            "config",
            "pct_ros2_adapter.yaml",
        ]
    )
    default_supervisor_config = PathJoinSubstitution(
        [
            FindPackageShare("navigation_supervisor"),
            "config",
            "navigation_supervisor.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    scan_config_file = LaunchConfiguration("scan_config_file")
    controller_config_file = LaunchConfiguration("controller_config_file")
    tuning_config_file = LaunchConfiguration("tuning_config_file")
    manual_path_config_file = LaunchConfiguration("manual_path_config_file")
    pct_config_file = LaunchConfiguration("pct_config_file")
    supervisor_config_file = LaunchConfiguration("supervisor_config_file")
    start_bridge = LaunchConfiguration("start_bridge")
    start_scan = LaunchConfiguration("start_scan")
    start_controller = LaunchConfiguration("start_controller")
    start_manual_path = LaunchConfiguration("start_manual_path")
    start_pct = LaunchConfiguration("start_pct")
    start_supervisor = LaunchConfiguration("start_supervisor")
    start_odometry_tf = LaunchConfiguration("start_odometry_tf")
    pct_backend_kind = LaunchConfiguration("pct_backend_kind")
    pct_backend_kind_parameter = ParameterValue(
        pct_backend_kind,
        value_type=str,
    )
    body_pose_topic = LaunchConfiguration("body_pose_topic")
    body_height_m = LaunchConfiguration("body_height_m")
    body_height_parameter = ParameterValue(body_height_m, value_type=float)
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    world_frame_parameter = ParameterValue(world_frame, value_type=str)
    base_frame_parameter = ParameterValue(base_frame, value_type=str)
    cloud_topic = LaunchConfiguration("cloud_topic")
    pct_path_topic = LaunchConfiguration("pct_path_topic")
    initial_path_topic = LaunchConfiguration("initial_path_topic")
    pct_goal_topic = LaunchConfiguration("pct_goal_topic")
    pct_status_topic = LaunchConfiguration("pct_status_topic")
    pct_command_service = LaunchConfiguration("pct_command_service")
    bspline_topic = LaunchConfiguration("bspline_topic")
    scan_status_topic = LaunchConfiguration("scan_status_topic")
    controller_status_topic = LaunchConfiguration("controller_status_topic")
    grid_map_observation_diagnostics_topic = LaunchConfiguration(
        "grid_map_observation_diagnostics_topic"
    )
    bspline_diagnostics_topic = LaunchConfiguration(
        "bspline_diagnostics_topic"
    )
    navigation_status_topic = LaunchConfiguration("navigation_status_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    stair_execution_frozen_topic = LaunchConfiguration(
        "stair_execution_frozen_topic"
    )
    stair_execution_freeze_timeout_sec = LaunchConfiguration(
        "stair_execution_freeze_timeout_sec"
    )
    stair_execution_freeze_confirmation_sec = LaunchConfiguration(
        "stair_execution_freeze_confirmation_sec"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_bridge_config,
                description="Isaac/SCAN topic、frame、QoS 与点云过滤配置",
            ),
            DeclareLaunchArgument(
                "scan_config_file",
                default_value=default_scan_config,
                description="SCAN Planner 参数文件",
            ),
            DeclareLaunchArgument(
                "controller_config_file",
                default_value=default_controller_config,
                description="SCAN 闭环控制器参数文件",
            ),
            DeclareLaunchArgument(
                "tuning_config_file",
                default_value=default_tuning_config,
                description=(
                    "PCT、SCAN Planner 与 SCAN Controller 的统一性能调参文件"
                ),
            ),
            DeclareLaunchArgument(
                "manual_path_config_file",
                default_value=default_manual_path_config,
                description="手工三维地面 Path 参数文件",
            ),
            DeclareLaunchArgument(
                "pct_config_file",
                default_value=default_pct_config,
                description="PCT 资产、坐标、高度、QoS 与规划安全参数",
            ),
            DeclareLaunchArgument(
                "supervisor_config_file",
                default_value=default_supervisor_config,
                description="导航 supervisor 状态机、超时与重规划参数",
            ),
            DeclareLaunchArgument(
                "start_bridge",
                default_value="true",
                description=(
                    "是否把 /isaac 原始观测归一化到 /body_pose 与 "
                    "/cloud_registered；直接提供标准输入的探针或 LIO 应设为 false"
                ),
            ),
            DeclareLaunchArgument("start_scan", default_value="true"),
            DeclareLaunchArgument("start_controller", default_value="true"),
            DeclareLaunchArgument("start_manual_path", default_value="false"),
            DeclareLaunchArgument("start_pct", default_value="true"),
            DeclareLaunchArgument("start_supervisor", default_value="true"),
            DeclareLaunchArgument(
                "start_odometry_tf",
                default_value="true",
                description=(
                    "是否根据 /body_pose 广播 world 到 base_link 的动态 TF；"
                    "若外部定位系统已经发布该TF，应设为false"
                )
            ),
            DeclareLaunchArgument(
                "pct_backend_kind",
                default_value="upstream",
                description=(
                    "PCT backend：主线固定 upstream；compatible 仅供隔离回归"
                ),
            ),
            DeclareLaunchArgument(
                "body_pose_topic", default_value="/body_pose"
            ),
            DeclareLaunchArgument(
                "world_frame",
                default_value="world",
                description=(
                    "bridge、PCT、SCAN、controller 与 supervisor "
                    "共享的世界 frame"
                ),
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="base_link",
                description=(
                    "bridge、PCT、SCAN、controller 与 supervisor "
                    "共享的机体 frame"
                ),
            ),
            DeclareLaunchArgument(
                "body_height_m",
                default_value="0.338",
                description=(
                    "base 到地面的统一高度合同；主运行链由统一 tuning YAML "
                    "读取后显式传入，同步覆盖 PCT、bridge、SCAN 与 controller"
                ),
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/cloud_registered"
            ),
            DeclareLaunchArgument(
                "pct_path_topic",
                default_value="/pct/global_path",
                description="PCT adapter 的可审计原始全局 Path",
            ),
            DeclareLaunchArgument(
                "initial_path_topic",
                default_value="/initial_path",
                description="与 PCT Path 同 stamp/payload 的 SCAN 输入 Path",
            ),
            DeclareLaunchArgument(
                "pct_goal_topic", default_value="/pct/goal"
            ),
            DeclareLaunchArgument(
                "pct_status_topic", default_value="/pct/planning_status"
            ),
            DeclareLaunchArgument(
                "pct_command_service", default_value="/pct/planning_command"
            ),
            DeclareLaunchArgument(
                "bspline_topic", default_value="/planning/bspline"
            ),
            DeclareLaunchArgument(
                "scan_status_topic", default_value="/planning/scan_status"
            ),
            DeclareLaunchArgument(
                "controller_status_topic",
                default_value="/planning/controller_status",
            ),
            DeclareLaunchArgument(
                "grid_map_observation_diagnostics_topic",
                default_value="/planning/grid_map_observation_diagnostics",
            ),
            DeclareLaunchArgument(
                "bspline_diagnostics_topic",
                default_value="/planning/bspline_diagnostics",
            ),
            DeclareLaunchArgument(
                "navigation_status_topic",
                default_value="/navigation/status",
            ),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument(
                "stair_execution_frozen_topic",
                default_value="/planning/stair_execution_frozen",
                description="楼梯 root-lock 写入的 typed SCAN 规划暂停快照",
            ),
            DeclareLaunchArgument(
                "stair_execution_freeze_timeout_sec",
                default_value="0.25",
                description="typed 楼梯冻结快照最大新鲜度秒数",
            ),
            DeclareLaunchArgument(
                "stair_execution_freeze_confirmation_sec",
                default_value="0.05",
                description=(
                    "首次快照超时后的调度确认宽限；期间更高合法序列可恢复"
                ),
            ),
            OpaqueFunction(function=_validate_path_source),
            OpaqueFunction(function=_validate_body_height),
            OpaqueFunction(function=_validate_stair_freeze_timing),
            OpaqueFunction(function=_validate_frame_arguments),
            Node(
                package="isaac_navigation_bridge",
                executable="isaac_navigation_bridge",
                name="isaac_navigation_bridge",
                output="screen",
                condition=IfCondition(start_bridge),
                parameters=[
                    config_file,
                    {
                        "use_sim_time": True,
                        "topics.body_pose_output": body_pose_topic,
                        "topics.cloud_output": cloud_topic,
                        "topics.initial_path_input": initial_path_topic,
                        "frames.odom": world_frame_parameter,
                        "frames.cloud": world_frame_parameter,
                        "frames.base": base_frame_parameter,
                        "filters.body_height_m": body_height_parameter,
                    },
                ],
            ),
            Node(
                package="pct_ros2_adapter",
                executable="pct_ros2_adapter",
                name="pct_ros2_adapter",
                output="screen",
                condition=IfCondition(start_pct),
                parameters=[
                    pct_config_file,
                    tuning_config_file,
                    {
                        "use_sim_time": True,
                        "topics.odometry_input": body_pose_topic,
                        "topics.goal_input": pct_goal_topic,
                        "topics.path_output": pct_path_topic,
                        "topics.scan_path_output": initial_path_topic,
                        "topics.status_output": pct_status_topic,
                        "topics.command_service": pct_command_service,
                        "frames.world": world_frame_parameter,
                        "frames.base": base_frame_parameter,
                        "planner.backend_kind": pct_backend_kind_parameter,
                        "planner.goal_base_to_ground_m": (
                            body_height_parameter
                        ),
                        "planner.slice_query_root_to_floor_m": (
                            body_height_parameter
                        ),
                    },
                ],
            ),
            Node(
                package="scan_planner",
                executable="scan_planner_node",
                name="scan_planner_node",
                output="screen",
                condition=IfCondition(start_scan),
                parameters=[
                    scan_config_file,
                    tuning_config_file,
                    {
                        "use_sim_time": True,
                        "grid_map.frame_id": world_frame_parameter,
                        "grid_map.base_frame_id": base_frame_parameter,
                        "grid_map.body_height": body_height_parameter,
                        "topics.stair_execution_frozen": (
                            stair_execution_frozen_topic
                        ),
                        "topics.controller_status": controller_status_topic,
                        "fsm.stair_execution_freeze_timeout_sec": (
                            stair_execution_freeze_timeout_sec
                        ),
                        "fsm.stair_execution_freeze_confirmation_sec": (
                            stair_execution_freeze_confirmation_sec
                        ),
                        "topics.planning_status": scan_status_topic,
                        "topics.grid_map_observation_diagnostics": (
                            grid_map_observation_diagnostics_topic
                        ),
                        "topics.bspline_diagnostics": (
                            bspline_diagnostics_topic
                        ),
                    },
                ],
                remappings=[
                    ("body_pose", body_pose_topic),
                    ("sensor_pose", body_pose_topic),
                    ("cloud", cloud_topic),
                    ("initial_path", initial_path_topic),
                    ("planning/bspline", bspline_topic),
                ],
            ),
            Node(
                package="scan_navigation_tools",
                executable="manual_path_publisher",
                name="manual_path_publisher",
                output="screen",
                condition=IfCondition(start_manual_path),
                parameters=[
                    manual_path_config_file,
                    {
                        "use_sim_time": True,
                        "topic": initial_path_topic,
                        "frame_id": world_frame_parameter,
                    },
                ],
            ),
            Node(
                package="scan_controller",
                executable="scan_controller_node",
                name="scan_controller",
                output="screen",
                condition=IfCondition(start_controller),
                parameters=[
                    controller_config_file,
                    tuning_config_file,
                    {
                        "use_sim_time": True,
                        "topics.bspline": bspline_topic,
                        "topics.initial_path": initial_path_topic,
                        "topics.body_pose": body_pose_topic,
                        "topics.cloud": cloud_topic,
                        "topics.cmd_vel": cmd_vel_topic,
                        "topics.controller_status": controller_status_topic,
                        "frames.world": world_frame_parameter,
                        "frames.base": base_frame_parameter,
                        "reference_path.body_height_m": (
                            body_height_parameter
                        ),
                    },
                ],
            ),
            Node(
                package="navigation_supervisor",
                executable="navigation_supervisor",
                name="navigation_supervisor",
                output="screen",
                condition=IfCondition(start_supervisor),
                parameters=[
                    supervisor_config_file,
                    tuning_config_file,
                    {
                        "use_sim_time": True,
                        "frames.world": world_frame_parameter,
                        "frames.base": base_frame_parameter,
                        "topics.odometry": body_pose_topic,
                        "topics.point_cloud": cloud_topic,
                        "topics.global_path": initial_path_topic,
                        "topics.pct_status": pct_status_topic,
                        "topics.bspline": bspline_topic,
                        "topics.scan_status": scan_status_topic,
                        "topics.controller_status": controller_status_topic,
                        "topics.navigation_status": navigation_status_topic,
                        "topics.pct_command_service": pct_command_service,
                    },
                ],
            ),
            Node(
                package="isaac_navigation_bridge",
                executable="odometry_tf_broadcaster",
                name="odometry_tf_broadcaster",
                output="screen",
                condition=IfCondition(start_odometry_tf),
                parameters=[
                    {
                        "use_sim_time": True,
                        "topics.body_pose": body_pose_topic,
                        "frames.world": world_frame_parameter,
                        "frames.base": base_frame_parameter,
                    }
                ]
            )
        ]
    )
