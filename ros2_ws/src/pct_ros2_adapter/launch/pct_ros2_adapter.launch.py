"""启动使用仿真时钟的 PCT 三维全局规划 ROS 2 adapter。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """声明配置文件与 PCT 在线消息接口。"""

    config_file = LaunchConfiguration("config_file")
    backend_kind = LaunchConfiguration("backend_kind")
    body_height_m = LaunchConfiguration("body_height_m")
    odometry_topic = LaunchConfiguration("odometry_topic")
    goal_topic = LaunchConfiguration("goal_topic")
    command_service = LaunchConfiguration("command_service")
    path_topic = LaunchConfiguration("path_topic")
    scan_path_topic = LaunchConfiguration("scan_path_topic")
    status_topic = LaunchConfiguration("status_topic")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("pct_ros2_adapter"),
                        "config",
                        "pct_ros2_adapter.yaml",
                    ]
                ),
                description="PCT 资产、坐标、高度、QoS 与规划安全参数",
            ),
            DeclareLaunchArgument(
                "odometry_topic",
                default_value="/body_pose",
            ),
            DeclareLaunchArgument(
                "backend_kind",
                default_value="upstream",
                description=(
                    "PCT backend：生产使用 upstream；compatible 仅供隔离回归"
                ),
            ),
            DeclareLaunchArgument(
                "body_height_m",
                default_value="0.30",
                description="目标 base 到最终 Path 地面的统一高度合同",
            ),
            DeclareLaunchArgument("goal_topic", default_value="/pct/goal"),
            DeclareLaunchArgument(
                "command_service",
                default_value="/pct/planning_command",
            ),
            DeclareLaunchArgument(
                "path_topic",
                default_value="/pct/global_path",
            ),
            DeclareLaunchArgument(
                "scan_path_topic",
                default_value="/initial_path",
                description="保持原始 PCT stamp/payload 的 SCAN 输入 Path",
            ),
            DeclareLaunchArgument(
                "status_topic",
                default_value="/pct/planning_status",
            ),
            Node(
                package="pct_ros2_adapter",
                executable="pct_ros2_adapter",
                name="pct_ros2_adapter",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "use_sim_time": True,
                        "topics.odometry_input": odometry_topic,
                        "topics.goal_input": goal_topic,
                        "topics.command_service": command_service,
                        "topics.path_output": path_topic,
                        "topics.scan_path_output": scan_path_topic,
                        "topics.status_output": status_topic,
                        "planner.backend_kind": ParameterValue(
                            backend_kind,
                            value_type=str,
                        ),
                        "planner.goal_base_to_ground_m": ParameterValue(
                            body_height_m,
                            value_type=float,
                        ),
                        "planner.slice_query_root_to_floor_m": ParameterValue(
                            body_height_m,
                            value_type=float,
                        ),
                    },
                ],
            ),
        ]
    )
