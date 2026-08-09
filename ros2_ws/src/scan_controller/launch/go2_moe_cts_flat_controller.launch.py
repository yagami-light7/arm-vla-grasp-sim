"""启动标准 Go2 + MoE-CTS 平地验收使用的 SCAN 闭环控制器。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    @brief 组合通用控制器配置与标准 Go2 的保守平地覆盖层
    @return 只启动一个 scan_controller 节点的 LaunchDescription
    """

    package_share = FindPackageShare("scan_controller")
    base_config_file = LaunchConfiguration("base_config_file")
    profile_config_file = LaunchConfiguration("profile_config_file")
    bspline_topic = LaunchConfiguration("bspline_topic")
    initial_path_topic = LaunchConfiguration("initial_path_topic")
    body_pose_topic = LaunchConfiguration("body_pose_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    controller_status_topic = LaunchConfiguration("controller_status_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "base_config_file",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "controller.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "profile_config_file",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "go2_moe_cts_flat.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "bspline_topic", default_value="/planning/bspline"
            ),
            DeclareLaunchArgument(
                "initial_path_topic", default_value="/initial_path"
            ),
            DeclareLaunchArgument(
                "body_pose_topic", default_value="/body_pose"
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/cloud_registered"
            ),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument(
                "controller_status_topic",
                default_value="/planning/controller_status",
            ),
            Node(
                package="scan_controller",
                executable="scan_controller_node",
                name="scan_controller",
                output="screen",
                parameters=[
                    base_config_file,
                    profile_config_file,
                    {
                        "use_sim_time": True,
                        "topics.bspline": bspline_topic,
                        "topics.initial_path": initial_path_topic,
                        "topics.body_pose": body_pose_topic,
                        "topics.cloud": cloud_topic,
                        "topics.cmd_vel": cmd_vel_topic,
                        "topics.controller_status": controller_status_topic,
                    },
                ],
            ),
        ]
    )
