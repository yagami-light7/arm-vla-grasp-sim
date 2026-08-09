#include <cstdio>
#include <exception>

#include <rclcpp/rclcpp.hpp>

#include "scan_controller/scan_controller_node.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(scan_controller::makeScanControllerNode());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "scan_controller 启动失败：%s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
