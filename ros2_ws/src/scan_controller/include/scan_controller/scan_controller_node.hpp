#ifndef SCAN_CONTROLLER__SCAN_CONTROLLER_NODE_HPP_
#define SCAN_CONTROLLER__SCAN_CONTROLLER_NODE_HPP_

#include <memory>

#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

namespace scan_controller
{

// 工厂只暴露标准 Node 接口，便于包级测试通过真实 ROS topic
// 验证状态合同。
std::shared_ptr<rclcpp::Node> makeScanControllerNode(
  const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

}  // 命名空间 scan_controller

#endif  // SCAN_CONTROLLER__SCAN_CONTROLLER_NODE_HPP_ 头文件保护
