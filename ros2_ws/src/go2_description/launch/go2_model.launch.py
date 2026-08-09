"""启动 Unitree Go2 的关节状态、机器人 TF 和 RViz 可视化。"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition,UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建 Go2 模型可视化描述"""

    package_share = get_package_share_directory("go2_description")
    urdf_path = os.path.join(
        package_share,
        "urdf",
        "go2_description.urdf",
    )

    # robot_state_publisher 需要 URDF 的完整文本，而不是文件路径
    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_gui = LaunchConfiguration("use_gui")
    start_rviz = LaunchConfiguration("start_rviz")

    # 使用 Unitree RL Lab 中定义的 Go2 初始站姿。
    #
    # 参数名中的 zeros. 前缀由 joint_state_publisher 识别，
    # 后半部分必须与 URDF 中的 joint name 完全一致。
    stand_pose_parameters = {
        "rate": 30,
        "zeros.FR_hip_joint": -0.1,
        "zeros.FR_thigh_joint": 0.8,
        "zeros.FR_calf_joint": -1.5,
        "zeros.FL_hip_joint": 0.1,
        "zeros.FL_thigh_joint": 0.8,
        "zeros.FL_calf_joint": -1.5,
        "zeros.RR_hip_joint": -0.1,
        "zeros.RR_thigh_joint": 1.0,
        "zeros.RR_calf_joint": -1.5,
        "zeros.RL_hip_joint": 0.1,
        "zeros.RL_thigh_joint": 1.0,
        "zeros.RL_calf_joint": -1.5,
        "use_sim_time": use_sim_time,
    }

    # 非GUI
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="go2_joint_state_publisher",
        output = "screen",
        arguments=[urdf_path],
        parameters=[stand_pose_parameters],
        condition=UnlessCondition(use_gui),
    )

    # GUI:与上述非GUI互斥
    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="go2_joint_state_publisher",
        output="screen",
        arguments=[urdf_path],
        parameters=[stand_pose_parameters],
        condition=IfCondition(use_gui),
    )

    # 读取 URDF 和 /joint_states，计算 base 到每个机器人 link 的 TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="go2_robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description":robot_description,
                "use_sim_time":use_sim_time,
            }
        ],
    )

    # 导航框架使用 base_link，但官方 Go2 URDF 的根 link 名为 base。
    # 这里建立一个恒等变换，既保留官方 URDF，又接入导航 TF 命名。
    base_frame_alias = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="go2_base_frame_alias",
        output="screen",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "base",
        ],
        parameters=[
            {
                "use_sim_time":use_sim_time,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="go2_model_rviz",
        output="screen",
        parameters=[
            {
                "use_sim_time":use_sim_time,
            }
        ],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="是否使用仿真时钟",
            ),
            DeclareLaunchArgument(
                "use_gui",
                default_value="false",
                description="是否使用关节角滑块gui",
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="true",
                description="是否启动rviz",
            ),
            joint_state_publisher,
            joint_state_publisher_gui,
            robot_state_publisher,
            base_frame_alias,
            rviz,
        ]
    )