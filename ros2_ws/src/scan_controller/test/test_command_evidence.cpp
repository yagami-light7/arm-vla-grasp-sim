#include <limits>

#include <gtest/gtest.h>

#include "scan_controller/command_evidence.hpp"

namespace
{

using scan_controller::CommandEvidence;
using scan_controller::ControlOutput;
using scan_controller::kActiveSensingYawRateHardCap;

TEST(CommandEvidence, ActiveSensingHardGateRecordsRawViolation)
{
  CommandEvidence evidence;
  evidence.reset(true);

  ControlOutput unsafe;
  unsafe.vx = 0.03;
  unsafe.vy = -0.02;
  unsafe.wz = 0.50;
  const auto command = evidence.filterAndRecord(unsafe);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(command.angular.z, kActiveSensingYawRateHardCap);
  const auto &aggregate = evidence.aggregate();
  EXPECT_EQ(aggregate.sample_count, 1U);
  EXPECT_DOUBLE_EQ(aggregate.first_command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(aggregate.first_command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(
    aggregate.first_command.angular.z,
    kActiveSensingYawRateHardCap);
  EXPECT_DOUBLE_EQ(aggregate.max_abs_vx, 0.0);
  EXPECT_DOUBLE_EQ(aggregate.max_abs_vy, 0.0);
  EXPECT_DOUBLE_EQ(
    aggregate.max_abs_wz,
    kActiveSensingYawRateHardCap);
  EXPECT_EQ(aggregate.violation_count, 1U);
}

TEST(CommandEvidence, ActiveSensingNonFiniteYawFailsClosedAndRemainsAuditable)
{
  CommandEvidence evidence;
  evidence.reset(true);

  ControlOutput unsafe;
  unsafe.wz = std::numeric_limits<double>::quiet_NaN();
  const auto command = evidence.filterAndRecord(unsafe);

  EXPECT_DOUBLE_EQ(command.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(command.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(command.angular.z, 0.0);
  EXPECT_EQ(evidence.aggregate().sample_count, 1U);
  EXPECT_EQ(evidence.aggregate().violation_count, 1U);
}

TEST(CommandEvidence, DisabledIdentityDoesNotAccumulateSafetyStop)
{
  CommandEvidence evidence;
  evidence.reset(true);
  const auto first = evidence.filterAndRecord(ControlOutput{});
  EXPECT_DOUBLE_EQ(first.angular.z, 0.0);
  ASSERT_EQ(evidence.aggregate().sample_count, 1U);

  evidence.disable();
  ControlOutput stop_after_invalidation;
  const auto stop = evidence.filterAndRecord(stop_after_invalidation);

  EXPECT_DOUBLE_EQ(stop.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(stop.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(stop.angular.z, 0.0);
  EXPECT_EQ(evidence.aggregate().sample_count, 1U);
  EXPECT_EQ(evidence.aggregate().violation_count, 0U);
}

TEST(CommandEvidence, NormalTrajectoryPassesThroughAndUsesIndependentAggregate)
{
  CommandEvidence evidence;
  evidence.reset(false);
  ControlOutput output;
  output.vx = 0.12;
  output.vy = -0.04;
  output.wz = -0.31;
  const auto command = evidence.filterAndRecord(output);

  EXPECT_DOUBLE_EQ(command.linear.x, output.vx);
  EXPECT_DOUBLE_EQ(command.linear.y, output.vy);
  EXPECT_DOUBLE_EQ(command.angular.z, output.wz);
  EXPECT_EQ(evidence.aggregate().sample_count, 1U);
  EXPECT_DOUBLE_EQ(evidence.aggregate().max_abs_vx, 0.12);
  EXPECT_DOUBLE_EQ(evidence.aggregate().max_abs_vy, 0.04);
  EXPECT_DOUBLE_EQ(evidence.aggregate().max_abs_wz, 0.31);
  EXPECT_EQ(evidence.aggregate().violation_count, 0U);

  evidence.reset(true);
  EXPECT_EQ(evidence.aggregate().sample_count, 0U);
  EXPECT_EQ(evidence.aggregate().violation_count, 0U);
}

}  // 匿名命名空间
