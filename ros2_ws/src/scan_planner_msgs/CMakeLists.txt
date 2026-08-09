cmake_minimum_required(VERSION 3.8)
project(scan_planner_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Bspline.msg"
  "msg/BsplineDiagnostics.msg"
  "msg/ControllerStatus.msg"
  "msg/DataDisp.msg"
  "msg/GridMapObservationDiagnostics.msg"
  "msg/NavigationStatus.msg"
  "msg/PCTPlanningStatus.msg"
  "msg/ScanPlanningStatus.msg"
  "msg/StairExecutionFreeze.msg"
  "srv/PCTPlanningCommand.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

# 安装许可证与上游来源说明，确保二进制工作空间保留归属信息。
install(
  FILES LICENSE NOTICE
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
