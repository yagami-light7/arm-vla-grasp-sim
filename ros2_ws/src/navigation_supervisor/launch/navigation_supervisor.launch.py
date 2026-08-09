"""启动只负责类型化协调、不发布速度命令的导航 supervisor。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """返回可替换配置文件的独立 supervisor launch。"""

    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("navigation_supervisor"),
                        "config",
                        "navigation_supervisor.yaml",
                    ]
                ),
            ),
            Node(
                package="navigation_supervisor",
                executable="navigation_supervisor",
                name="navigation_supervisor",
                output="screen",
                parameters=[config_file, {"use_sim_time": True}],
            ),
        ]
    )
