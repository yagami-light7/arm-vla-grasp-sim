#pragma once

#include <cstdint>

#include <geometry_msgs/msg/twist.hpp>

#include "scan_controller/trajectory_tracker.hpp"

namespace scan_controller
{

inline constexpr double kActiveSensingYawRateHardCap = 0.20;

struct CommandAggregate
{
  std::uint64_t sample_count{0};
  geometry_msgs::msg::Twist first_command;
  double max_abs_vx{0.0};
  double max_abs_vy{0.0};
  double max_abs_wz{0.0};
  std::uint64_t violation_count{0};
};

// 按 accepted B-spline identity 聚合 controller 实际发布的逐拍命令。
class CommandEvidence
{
public:
  void reset(bool active_sensing_yaw_only);
  void disable();

  bool enabled() const;
  bool activeSensingYawOnly() const;
  const CommandAggregate &aggregate() const;

  // 返回允许发布的 Twist，并在返回前完成本 identity 的证据聚合。
  geometry_msgs::msg::Twist filterAndRecord(const ControlOutput &output);

private:
  void record(const geometry_msgs::msg::Twist &command, bool violation);

  CommandAggregate aggregate_;
  bool enabled_{false};
  bool active_sensing_yaw_only_{false};
};

}  // 命名空间 scan_controller
