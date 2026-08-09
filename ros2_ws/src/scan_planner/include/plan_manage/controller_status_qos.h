#pragma once

#include <cstddef>
#include <stdexcept>

#include <rclcpp/qos.hpp>

namespace scan_planner
{

// controller 会在同一回调中连续发布旧轨迹终态和新轨迹接管证据；固定保留
// 64 条，确保 planner 晚加入或短时调度阻塞后仍能按完整 identity 对账。
inline constexpr int kControllerStatusEvidenceDepth = 64;

inline void validateControllerStatusEvidenceDepth(const int depth)
{
  if (depth != kControllerStatusEvidenceDepth)
    throw std::runtime_error(
        "qos.controller_status_depth 必须固定为 64，"
        "防止连续 typed controller 证据被覆盖");
}

inline rclcpp::QoS makeControllerStatusEvidenceQos()
{
  return rclcpp::QoS(
             rclcpp::KeepLast(
                 static_cast<std::size_t>(kControllerStatusEvidenceDepth)))
      .reliable()
      .transient_local();
}

}  // namespace scan_planner
