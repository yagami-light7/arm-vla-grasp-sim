"""启动标准 Go2 + MoE-CTS 的快速手工楼梯闭环验收链。"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    @brief 在保留楼梯 Path 与安全门的前提下加载快速运动包络
    @return 使用快速调参覆盖层的单跑楼梯 LaunchDescription
    """

    bridge_share = FindPackageShare("isaac_navigation_bridge")
    stair_launch = PathJoinSubstitution(
        [bridge_share, "launch", "go2_moe_cts_stair_navigation.launch.py"]
    )
    fast_tuning_config = PathJoinSubstitution(
        [
            bridge_share,
            "config",
            "go2_moe_cts_stair_fast_tuning.yaml",
        ]
    )

    # 复用保守版本的唯一 Path、bridge、SCAN、controller 与 TF 组合关系，
    # 只替换统一调参文件，确保 A/B 实验不会改变地图或消息链。
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(stair_launch),
                launch_arguments={
                    "stair_tuning_config_file": fast_tuning_config,
                }.items(),
            )
        ]
    )
