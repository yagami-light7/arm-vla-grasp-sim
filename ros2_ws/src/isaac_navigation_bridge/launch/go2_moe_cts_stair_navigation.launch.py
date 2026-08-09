"""启动标准 Go2 + MoE-CTS 的手工单跑楼梯闭环验收链。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    @brief 组合 bridge、手工楼梯 Path、SCAN Planner、controller 与动态 TF
    @return 不启动 PCT 和 supervisor 的单跑楼梯 LaunchDescription
    """

    bridge_share = FindPackageShare("isaac_navigation_bridge")
    navigation_tools_share = FindPackageShare("scan_navigation_tools")
    base_launch = PathJoinSubstitution(
        [bridge_share, "launch", "pct_scan_navigation.launch.py"]
    )
    default_stair_path_config = PathJoinSubstitution(
        [
            navigation_tools_share,
            "config",
            "go2_moe_cts_stair_path.yaml",
        ]
    )
    default_stair_tuning_config = PathJoinSubstitution(
        [bridge_share, "config", "go2_moe_cts_stair_tuning.yaml"]
    )
    stair_path_config = LaunchConfiguration("stair_path_config_file")
    stair_tuning_config = LaunchConfiguration("stair_tuning_config_file")

    # 本阶段只替换已验收平地 Path 为确定性楼梯 Path。PCT 与 supervisor
    # 留到单跑楼梯闭环通过后再接入，便于把故障定位在 SCAN 或 locomotion。
    launch_arguments = {
        "start_scan": "true",
        "start_controller": "true",
        "start_manual_path": "true",
        "start_pct": "false",
        "start_supervisor": "false",
        "start_odometry_tf": "true",
        "initial_path_topic": "/initial_path",
        "manual_path_config_file": stair_path_config,
        "tuning_config_file": stair_tuning_config,
        "body_height_m": "0.342",
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stair_path_config_file",
                default_value=default_stair_path_config,
                description=(
                    "标准 Go2 单跑楼梯的手工地面 Path 参数文件"
                ),
            ),
            DeclareLaunchArgument(
                "stair_tuning_config_file",
                default_value=default_stair_tuning_config,
                description=(
                    "标准 Go2 单跑楼梯的 SCAN/controller 调参覆盖层"
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments=launch_arguments.items(),
            )
        ]
    )
