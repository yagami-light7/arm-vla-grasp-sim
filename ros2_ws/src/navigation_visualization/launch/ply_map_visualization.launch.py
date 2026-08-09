"""同时启动PLY地图发布器、静态TF 和 RViz"""

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource

start_go2_model = LaunchConfiguration("start_go2_model")
go2_use_gui = LaunchConfiguration("go2_use_gui")

go2_model_launch = PathJoinSubstitution(
    [
        FindPackageShare("go2_description"),
        "launch",
        "go2_model.launch.py",
    ]
)


def generate_launch_description() -> LaunchDescription:
    """ 创建 PLY 导航地图可视化 launch"""

    # LaunchConfiguration 不是参数值本身，而是启动时才解析的参数占位符。
    ply_path = LaunchConfiguration("ply_path")
    cloud_topic = LaunchConfiguration("cloud_topic")

    # 地图坐标系TF转换参数 占位符
    world_frame = LaunchConfiguration("world_frame")
    ply_frame = LaunchConfiguration("ply_frame")
    map_yaw_rad = LaunchConfiguration("map_yaw_rad")

    voxel_leaf_size_m = LaunchConfiguration("voxel_leaf_size_m")
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz_config = LaunchConfiguration("rviz_config")

    # 是否启用Path发布器
    start_manual_path = LaunchConfiguration("start_manual_path")

    # Path使用的 YAML 配置文件路径
    manual_path_config_file = LaunchConfiguration("manual_path_config_file")

    # Path 发布的 Topic
    path_topic = LaunchConfiguration("path_topic")

    # ParameterValue 用来明确指定参数类型，避免 "0.05" 被当成字符串。
    voxel_leaf_size_value = ParameterValue(voxel_leaf_size_m, value_type=float)
    use_sim_time_value = ParameterValue(use_sim_time, value_type=bool)

    # Rviz 配置文件
    default_rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("navigation_visualization"),
            "rviz",
            "navigation_visualization.rviz",
        ]
    )

    # Path 配置
    default_manual_path_config = PathJoinSubstitution(
        [
            FindPackageShare("scan_navigation_tools"),
            "config",
            "manual_path.yaml",
        ]
    )

    return LaunchDescription(
        [
            # ply_path 不提供默认值，必须在启动时指定
            DeclareLaunchArgument(
                "ply_path",
                description="需要加载的 PLY 地图文件路径",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/map/ply",
                description="静态 PLY 点云发布 Topic",
            ),
            DeclareLaunchArgument(
                "world_frame",
                default_value="world",
                description="Isaac Sim 和导航模块使用的世界坐标系",
            ),
            DeclareLaunchArgument(
                "ply_frame",
                default_value="pct_map",
                description="原始 PCT/PLY 地图坐标系",
            ),
            DeclareLaunchArgument(
                "map_yaw_rad",
                default_value="3.141592653589793",
                description="从原始 PLY 坐标系到 world 的 Z 轴旋转",
            ),
            DeclareLaunchArgument(
                "voxel_leaf_size_m",
                default_value="1.0",
                description="PLY 点云体素化的体素大小（单位：米），0代表关闭降采样",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="是否使用ROS2仿真时间",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="RViz 配置文件路径",
            ),
            DeclareLaunchArgument(
                "start_manual_path",
                default_value="false",
                description=(
                    "是否启动手动三维 Path 发布器；PCT/SCAN 联调必须保持 false"
                ),
            ),
            DeclareLaunchArgument(
                "manual_path_config_file",
                default_value=default_manual_path_config,
                description="手动三维 Path 使用的 YAML 配置文件路径",
            ),
            DeclareLaunchArgument(
                "path_topic",
                default_value="/initial_path",
                description="手动三维 Path 发布 Topic",
            ),
            DeclareLaunchArgument(
                "start_go2_model",
                default_value="true",
                description="是否加载 Go2 RobotModel 和机器人内部 TF",
            ),
            DeclareLaunchArgument(
                "go2_use_gui",
                default_value="false",
                description="是否使用 Go2 关节角 GUI",
            ),
            # package
            #   去哪个 ROS 2 包里查找程序

            # executable
            #   启动哪个真实可执行文件

            # name
            #   程序运行后在 ROS 图中使用什么节点名

            #第一个Node由cmake编译生成可执行文件，后两个Node是ROS2自带的可执行文件

            # 发布静态 PLY 点云
            Node(
                package="navigation_visualization",
                executable="ply_map_publisher",
                name="ply_map_publisher",
                output="screen",
                parameters=[
                    {
                        "ply_path": ply_path,
                        "topic": cloud_topic,
                        "frame_id": ply_frame,
                        "voxel_leaf_size_m": voxel_leaf_size_value,
                        "use_sim_time": use_sim_time_value,
                    }
                ],
            ),

            # 临时建立一个TF子坐标系，使world进入TF树
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_tf_anchor",
                output="screen",
                arguments=[
                    "--x", "0",
                    "--y", "0",
                    "--z", "0",
                    "--yaw", map_yaw_rad,
                    "--pitch", "0",
                    "--roll", "0",
                    "--frame-id", world_frame,
                    "--child-frame-id", ply_frame,
                ],
                parameters=[{"use_sim_time": use_sim_time_value}
                ],
            ),

            # 启动 RViz
            Node(
                package="rviz2",
                executable="rviz2",
                name="navigation_rviz",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[
                    {"use_sim_time": use_sim_time_value}
                ],
            ),

            # 把 PCT Path 转成圆柱/端点 Marker，并记录机器人实际轨迹。
            # 该节点只发布显示数据，不参与规划、控制或安全判定。
            Node(
                package="navigation_visualization",
                executable="navigation_marker_publisher",
                name="navigation_marker_publisher",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time_value}],
            ),

            # 启动手动三维 Path 发布器
            Node(
                package="scan_navigation_tools",
                executable="manual_path_publisher",
                name="manual_path_publisher",
                output="screen",
                condition=IfCondition(start_manual_path),
                parameters=[
                    # 首先加载 YAML 中的点列和其他参数
                    manual_path_config_file,

                    # 后面的字典会覆盖 YAML 中的同名参数，
                    # 保证整个组合 launch 使用统一的 Topic 和坐标系。
                    {
                        "topic": path_topic,
                        "frame_id": world_frame,
                        "use_sim_time": True,
                    },
                ]
            ),

            # 加载 Go2 模型，但不启动第二个 RViz。
            #
            # world → base_link 仍由 odometry_tf_broadcaster 负责；
            # Go2 launch 只负责 base_link → base → 四肢。
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(go2_model_launch),
                condition=IfCondition(start_go2_model),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "use_gui": go2_use_gui,
                    "start_rviz": "false",
                }.items(),
            ),
        ]
    )
