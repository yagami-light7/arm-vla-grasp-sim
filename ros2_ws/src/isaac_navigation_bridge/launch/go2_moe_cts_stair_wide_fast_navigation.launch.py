"""启动真实 0.30 m 踏面的标准 Go2 + MoE-CTS 快速楼梯 A/B。"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    @brief 组合宽踏面 Path、对应地面高度和既有 fast 运动包络
    @return 不启动 PCT 与 supervisor 的单变量楼梯 A/B LaunchDescription
    """

    bridge_share = FindPackageShare("isaac_navigation_bridge")
    navigation_tools_share = FindPackageShare("scan_navigation_tools")
    stair_launch = PathJoinSubstitution(
        [bridge_share, "launch", "go2_moe_cts_stair_navigation.launch.py"]
    )
    wide_path_config = PathJoinSubstitution(
        [
            navigation_tools_share,
            "config",
            "go2_moe_cts_stair_wide_path.yaml",
        ]
    )
    wide_fast_tuning_config = PathJoinSubstitution(
        [
            bridge_share,
            "config",
            "go2_moe_cts_stair_wide_fast_tuning.yaml",
        ]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(stair_launch),
                launch_arguments={
                    "stair_path_config_file": wide_path_config,
                    "stair_tuning_config_file": wide_fast_tuning_config,
                }.items(),
            )
        ]
    )
