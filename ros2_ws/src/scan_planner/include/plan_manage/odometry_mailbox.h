#ifndef SCAN_PLANNER__ODOMETRY_MAILBOX_H_
#define SCAN_PLANNER__ODOMETRY_MAILBOX_H_

#include <cstddef>
#include <mutex>
#include <utility>

#include <nav_msgs/msg/odometry.hpp>

namespace scan_planner
{

struct OdometryMailboxDrain
{
  nav_msgs::msg::Odometry::ConstSharedPtr message;
  std::size_t coalesced_count{0};
};

// DDS 回调只保存最新一帧 Odometry；FSM 在规划前串行消费，既避免高频
// ROS 定时器饿死位姿订阅，也不让规划状态被第二个 executor 线程并发修改。
class OdometryMailbox
{
public:
  void push(nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    if (!message)
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_)
      ++coalesced_count_;
    latest_ = std::move(message);
  }

  OdometryMailboxDrain drain()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    OdometryMailboxDrain result;
    result.message = std::move(latest_);
    result.coalesced_count = coalesced_count_;
    coalesced_count_ = 0;
    return result;
  }

private:
  std::mutex mutex_;
  nav_msgs::msg::Odometry::ConstSharedPtr latest_;
  std::size_t coalesced_count_{0};
};

}  // namespace scan_planner

#endif  // SCAN_PLANNER__ODOMETRY_MAILBOX_H_
