"""启动只包含 SCAN 局部规划器的 ROS 2 节点。"""

import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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


def generate_launch_description():
    """声明可配置话题，并保持仿真时间为默认时钟。"""

    config_file = LaunchConfiguration("config_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    body_pose_topic = LaunchConfiguration("body_pose_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    initial_path_topic = LaunchConfiguration("initial_path_topic")
    bspline_topic = LaunchConfiguration("bspline_topic")
    grid_map_observation_diagnostics_topic = LaunchConfiguration(
        "grid_map_observation_diagnostics_topic"
    )
    bspline_diagnostics_topic = LaunchConfiguration(
        "bspline_diagnostics_topic"
    )
    stair_execution_frozen_topic = LaunchConfiguration(
        "stair_execution_frozen_topic"
    )
    controller_status_topic = LaunchConfiguration("controller_status_topic")
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
                default_value=PathJoinSubstitution(
                    [FindPackageShare("scan_planner"), "config", "planner.yaml"]
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("body_pose_topic", default_value="/body_pose"),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/cloud_registered"
            ),
            DeclareLaunchArgument(
                "initial_path_topic", default_value="/initial_path"
            ),
            DeclareLaunchArgument(
                "bspline_topic", default_value="/planning/bspline"
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
                "stair_execution_frozen_topic",
                default_value="/planning/stair_execution_frozen",
            ),
            DeclareLaunchArgument(
                "controller_status_topic",
                default_value="/planning/controller_status",
            ),
            DeclareLaunchArgument(
                "stair_execution_freeze_timeout_sec",
                default_value="0.25",
            ),
            DeclareLaunchArgument(
                "stair_execution_freeze_confirmation_sec",
                default_value="0.05",
            ),
            OpaqueFunction(function=_validate_stair_freeze_timing),
            Node(
                package="scan_planner",
                executable="scan_planner_node",
                name="scan_planner_node",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "use_sim_time": use_sim_time,
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
                    # 世界系点云仍需机体位姿作为射线起点。
                    ("sensor_pose", body_pose_topic),
                    ("cloud", cloud_topic),
                    ("initial_path", initial_path_topic),
                    ("planning/bspline", bspline_topic),
                ],
            ),
        ]
    )
