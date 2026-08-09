#include "scan_controller/command_evidence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scan_controller
{
namespace
{

constexpr double kYawRateViolationTolerance = 1.0e-12;

}  // 匿名命名空间

void CommandEvidence::reset(bool active_sensing_yaw_only)
{
  const CommandAggregate empty_aggregate;
  aggregate_ = empty_aggregate;
  enabled_ = true;
  active_sensing_yaw_only_ = active_sensing_yaw_only;
}

void CommandEvidence::disable()
{
  enabled_ = false;
}

bool CommandEvidence::enabled() const
{
  return enabled_;
}

bool CommandEvidence::activeSensingYawOnly() const
{
  return active_sensing_yaw_only_;
}

const CommandAggregate &CommandEvidence::aggregate() const
{
  return aggregate_;
}

geometry_msgs::msg::Twist CommandEvidence::filterAndRecord(
  const ControlOutput &output)
{
  geometry_msgs::msg::Twist command;
  bool violation = false;
  if (enabled_ && active_sensing_yaw_only_) {
    // 该硬门不可被 ROS 参数放宽。违规数基于限幅前的 tracker 输出，实际
    // Twist 则始终 fail closed，从而让验收既能看到回归又不会执行危险命令。
    violation =
      !std::isfinite(output.vx) || !std::isfinite(output.vy) ||
      !std::isfinite(output.wz) || output.vx != 0.0 || output.vy != 0.0 ||
      std::abs(output.wz) >
      kActiveSensingYawRateHardCap + kYawRateViolationTolerance;
    command.linear.x = 0.0;
    command.linear.y = 0.0;
    command.angular.z = std::isfinite(output.wz) ?
      std::clamp(
      output.wz, -kActiveSensingYawRateHardCap,
      kActiveSensingYawRateHardCap) : 0.0;
  } else {
    command.linear.x = output.vx;
    command.linear.y = output.vy;
    command.angular.z = output.wz;
  }
  record(command, violation);
  return command;
}

void CommandEvidence::record(
  const geometry_msgs::msg::Twist &command, bool violation)
{
  if (!enabled_) {
    return;
  }
  if (aggregate_.sample_count == 0U) {
    aggregate_.first_command = command;
  }
  if (aggregate_.sample_count < std::numeric_limits<std::uint64_t>::max()) {
    ++aggregate_.sample_count;
  }
  aggregate_.max_abs_vx = std::max(
    aggregate_.max_abs_vx, std::abs(command.linear.x));
  aggregate_.max_abs_vy = std::max(
    aggregate_.max_abs_vy, std::abs(command.linear.y));
  aggregate_.max_abs_wz = std::max(
    aggregate_.max_abs_wz, std::abs(command.angular.z));
  if (
    violation &&
    aggregate_.violation_count < std::numeric_limits<std::uint64_t>::max())
  {
    ++aggregate_.violation_count;
  }
}

}  // 命名空间 scan_controller
