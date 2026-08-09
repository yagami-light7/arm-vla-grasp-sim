cmake_minimum_required(VERSION 3.8)
project(scan_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(bspline_opt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(path_searching REQUIRED)
find_package(plan_env REQUIRED)
find_package(rclcpp REQUIRED)
find_package(scan_planner_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(traj_utils REQUIRED)
find_package(visualization_msgs REQUIRED)

set(scan_planner_dependencies
  bspline_opt
  geometry_msgs
  nav_msgs
  path_searching
  plan_env
  rclcpp
  scan_planner_msgs
  std_msgs
  traj_utils
  visualization_msgs
)

add_executable(scan_planner_node
  src/scan_planner_node.cpp
  src/scan_replan_fsm.cpp
  src/planner_manager.cpp
)
target_include_directories(scan_planner_node PRIVATE include)
ament_target_dependencies(scan_planner_node ${scan_planner_dependencies})
target_link_libraries(scan_planner_node Eigen3::Eigen)

install(TARGETS scan_planner_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(FILES LICENSE NOTICE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(
    test_reference_execution
    test/test_reference_execution.cpp
  )
  if(TARGET test_reference_execution)
    target_include_directories(test_reference_execution PRIVATE include)
    target_link_libraries(test_reference_execution Eigen3::Eigen)
  endif()
  ament_add_gtest(
    test_reference_trajectory_progress
    test/test_reference_trajectory_progress.cpp
  )
  if(TARGET test_reference_trajectory_progress)
    target_include_directories(test_reference_trajectory_progress PRIVATE include)
    ament_target_dependencies(test_reference_trajectory_progress bspline_opt)
    target_link_libraries(test_reference_trajectory_progress Eigen3::Eigen)
  endif()
  ament_add_gtest(
    test_reference_velocity
    test/test_reference_velocity.cpp
  )
  if(TARGET test_reference_velocity)
    target_include_directories(test_reference_velocity PRIVATE include)
    target_link_libraries(test_reference_velocity Eigen3::Eigen)
  endif()
  ament_add_gtest(
    test_stair_planning_freeze
    test/test_stair_planning_freeze.cpp
  )
  if(TARGET test_stair_planning_freeze)
    target_include_directories(test_stair_planning_freeze PRIVATE include)
    target_link_libraries(test_stair_planning_freeze Eigen3::Eigen)
  endif()
  ament_add_gtest(
    test_odometry_mailbox
    test/test_odometry_mailbox.cpp
  )
  if(TARGET test_odometry_mailbox)
    target_include_directories(test_odometry_mailbox PRIVATE include)
    ament_target_dependencies(test_odometry_mailbox nav_msgs)
  endif()
  ament_add_gtest(
    test_planning_status
    test/test_planning_status.cpp
  )
  if(TARGET test_planning_status)
    target_include_directories(test_planning_status PRIVATE include)
  endif()
  ament_add_gtest(
    test_local_plan_failure_reason
    test/test_local_plan_failure_reason.cpp
  )
  if(TARGET test_local_plan_failure_reason)
    target_include_directories(test_local_plan_failure_reason PRIVATE include)
    ament_target_dependencies(test_local_plan_failure_reason bspline_opt)
  endif()
  ament_add_gtest(
    test_active_sensing
    test/test_active_sensing.cpp
  )
  if(TARGET test_active_sensing)
    target_include_directories(test_active_sensing PRIVATE include)
    ament_target_dependencies(
      test_active_sensing
      bspline_opt
      geometry_msgs
      scan_planner_msgs
    )
    target_link_libraries(test_active_sensing Eigen3::Eigen)
  endif()
  ament_add_gtest(
    test_controller_status_qos
    test/test_controller_status_qos.cpp
  )
  if(TARGET test_controller_status_qos)
    target_include_directories(test_controller_status_qos PRIVATE include)
    ament_target_dependencies(test_controller_status_qos rclcpp)
  endif()
endif()

ament_package()
