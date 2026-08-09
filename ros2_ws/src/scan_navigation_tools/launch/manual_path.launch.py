"""启动 SCAN 手工三维路径发布器。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """返回只包含手工 Path 发布器的 launch。"""

    default_config = PathJoinSubstitution(
        [
            FindPackageShare("scan_navigation_tools"),
            "config",
            "manual_path.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="手工三维 Path 的 topic、frame、点列和发布时机配置",
            ),
            Node(
                package="scan_navigation_tools",
                executable="manual_path_publisher",
                name="manual_path_publisher",
                output="screen",
                parameters=[config_file, {"use_sim_time": True}],
            ),
        ]
    )
