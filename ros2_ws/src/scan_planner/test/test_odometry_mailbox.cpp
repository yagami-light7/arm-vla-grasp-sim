#include <gtest/gtest.h>

#include <memory>

#include <nav_msgs/msg/odometry.hpp>

#include "plan_manage/odometry_mailbox.h"

namespace
{

nav_msgs::msg::Odometry::ConstSharedPtr makeOdometry(
    const std::int32_t seconds, const double x)
{
  auto message = std::make_shared<nav_msgs::msg::Odometry>();
  message->header.stamp.sec = seconds;
  message->pose.pose.position.x = x;
  return message;
}

}  // namespace

TEST(OdometryMailbox, KeepsOnlyNewestFrameWhilePlannerIsBusy)
{
  scan_planner::OdometryMailbox mailbox;
  mailbox.push(makeOdometry(1, 1.0));
  mailbox.push(makeOdometry(2, 2.0));
  mailbox.push(makeOdometry(3, 3.0));

  const scan_planner::OdometryMailboxDrain drain = mailbox.drain();
  ASSERT_NE(drain.message, nullptr);
  EXPECT_EQ(drain.message->header.stamp.sec, 3);
  EXPECT_DOUBLE_EQ(drain.message->pose.pose.position.x, 3.0);
  EXPECT_EQ(drain.coalesced_count, 2U);
}

TEST(OdometryMailbox, DrainsEachFrameAtMostOnce)
{
  scan_planner::OdometryMailbox mailbox;
  mailbox.push(makeOdometry(7, 4.0));

  const scan_planner::OdometryMailboxDrain first = mailbox.drain();
  ASSERT_NE(first.message, nullptr);
  EXPECT_EQ(first.coalesced_count, 0U);

  const scan_planner::OdometryMailboxDrain second = mailbox.drain();
  EXPECT_EQ(second.message, nullptr);
  EXPECT_EQ(second.coalesced_count, 0U);
}
