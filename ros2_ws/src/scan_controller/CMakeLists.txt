cmake_minimum_required(VERSION 3.8)
project(scan_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(bspline_opt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(scan_planner_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(scan_controller_core
  src/point_cloud_validation.cpp
  src/trajectory_tracker.cpp
)
target_include_directories(scan_controller_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(scan_controller_core bspline_opt sensor_msgs)
target_link_libraries(scan_controller_core Eigen3::Eigen)

add_library(scan_controller_node_lib
  src/command_evidence.cpp
  src/scan_controller_node.cpp
)
target_include_directories(scan_controller_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(scan_controller_node_lib
  geometry_msgs
  nav_msgs
  rclcpp
  scan_planner_msgs
  sensor_msgs
  std_msgs
)
target_link_libraries(scan_controller_node_lib
  scan_controller_core
  Eigen3::Eigen
)

add_executable(scan_controller_node
  src/scan_controller_main.cpp
)
ament_target_dependencies(scan_controller_node rclcpp)
target_link_libraries(scan_controller_node scan_controller_node_lib)

install(TARGETS
  scan_controller_core
  scan_controller_node_lib
  scan_controller_node
  EXPORT export_scan_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES LICENSE NOTICE README.md DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_scan_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(
  bspline_opt Eigen3 geometry_msgs nav_msgs rclcpp scan_planner_msgs
  sensor_msgs std_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(rosgraph_msgs REQUIRED)
  ament_add_gtest(
    test_trajectory_tracker
    test/test_trajectory_tracker.cpp
  )
  if(TARGET test_trajectory_tracker)
    target_link_libraries(
      test_trajectory_tracker
      scan_controller_core
      Eigen3::Eigen
    )
  endif()
  ament_add_gtest(
    test_point_cloud_validation
    test/test_point_cloud_validation.cpp
  )
  if(TARGET test_point_cloud_validation)
    target_link_libraries(
      test_point_cloud_validation
      scan_controller_core
    )
    ament_target_dependencies(
      test_point_cloud_validation
      sensor_msgs
    )
  endif()
  ament_add_gtest(
    test_command_evidence
    test/test_command_evidence.cpp
  )
  if(TARGET test_command_evidence)
    target_link_libraries(
      test_command_evidence
      scan_controller_node_lib
    )
    ament_target_dependencies(
      test_command_evidence
      geometry_msgs
    )
  endif()
  ament_add_gtest(
    test_controller_status
    test/test_controller_status.cpp
  )
  if(TARGET test_controller_status)
    target_link_libraries(
      test_controller_status
      scan_controller_node_lib
    )
    ament_target_dependencies(
      test_controller_status
      nav_msgs
      rclcpp
      rosgraph_msgs
      scan_planner_msgs
      sensor_msgs
    )
  endif()
  add_test(
    NAME test_scan_controller_process_shutdown
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test/test_process_shutdown.py
      $<TARGET_FILE:scan_controller_node>
  )
  set_tests_properties(
    test_scan_controller_process_shutdown
    PROPERTIES TIMEOUT 120
  )
  add_test(
    NAME test_go2_moe_cts_profile
    COMMAND
      ${Python3_EXECUTABLE}
      -m pytest -q -p no:cacheprovider
      ${CMAKE_CURRENT_SOURCE_DIR}/test/test_go2_moe_cts_profile.py
  )
endif()

ament_package()
