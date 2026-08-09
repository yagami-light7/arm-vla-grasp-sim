cmake_minimum_required(VERSION 3.16)
project(navigation_visualization)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS io filters)

# 添加可执行文件
add_executable(
    ply_map_publisher
    src/ply_map_publisher.cpp
)

add_executable(
    navigation_marker_publisher
    src/navigation_marker_publisher.cpp
)

# 使用 C++17 标准
target_compile_features(
    ply_map_publisher 
    PRIVATE cxx_std_17
)

target_compile_features(
    navigation_marker_publisher
    PRIVATE cxx_std_17
)

# 添加 PCL 头文件路径
target_include_directories(
    ply_map_publisher 
    PRIVATE ${PCL_INCLUDE_DIRS}
)

# 链接 PCL 库
target_link_libraries(
    ply_map_publisher 
    ${PCL_LIBRARIES}
)

# 添加ROS2依赖
ament_target_dependencies(
    ply_map_publisher
    rclcpp
    sensor_msgs
    pcl_conversions
)

ament_target_dependencies(
    navigation_marker_publisher
    geometry_msgs
    nav_msgs
    rclcpp
    visualization_msgs
)

# 安装程序到install目录
install(
    TARGETS
        ply_map_publisher # ros2 run navigation_visualization ply_map_publisher
        navigation_marker_publisher
    DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 和 Rviz 配置，使 ros2 launch 能从 install 空间找到
install(
    DIRECTORY 
        launch 
        rviz
    DESTINATION share/${PROJECT_NAME}
)
ament_package()
