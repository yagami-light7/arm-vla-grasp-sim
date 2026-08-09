#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <plan_manage/active_sensing.h>

namespace
{

builtin_interfaces::msg::Time stamp(const std::int32_t sec)
{
  builtin_interfaces::msg::Time value;
  value.sec = sec;
  return value;
}

scan_planner::ActiveSensingStartContext validStartContext()
{
  scan_planner::ActiveSensingStartContext context;
  context.failure_reason =
      scan_planner::LocalPlanFailureReason::TerminalPointOccupied;
  context.planner_attempted = true;
  context.reference_mode = true;
  context.reference_path_stamp_ns = 100;
  context.inputs_fresh = true;
  context.stair_gate_clear = true;
  context.final_hold_lifecycle_clear = true;
  context.global_replan_gate_clear = true;
  context.body_rotation_envelope_free = true;
  context.planar_speed = 0.01;
  context.maximum_planar_speed = 0.03;
  return context;
}

}  // namespace

TEST(ActiveSensing, OnlyRecoverableTypedFailuresMayStart)
{
  auto context = validStartContext();
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  context.failure_reason =
      scan_planner::LocalPlanFailureReason::CollisionSegmentAstarDegenerate;
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  context.failure_reason =
      scan_planner::LocalPlanFailureReason::ReferenceCorridorRejected;
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  for (const auto reason : {
           scan_planner::LocalPlanFailureReason::None,
           scan_planner::LocalPlanFailureReason::StartRegionOccupied,
           scan_planner::LocalPlanFailureReason::OptimizationFailed,
           scan_planner::LocalPlanFailureReason::DynamicFeasibilityFailed})
  {
    context.failure_reason = reason;
    EXPECT_FALSE(scan_planner::activeSensingMayStart(context));
  }
}

TEST(ActiveSensing, ReferenceCorridorRejectionKeepsOneShotSafetyGates)
{
  auto context = validStartContext();
  context.failure_reason =
      scan_planner::LocalPlanFailureReason::ReferenceCorridorRejected;
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  context.consumed_reference_path_stamp_ns =
      context.reference_path_stamp_ns;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));

  context.consumed_reference_path_stamp_ns = 0;
  context.inputs_fresh = false;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));

  context.inputs_fresh = true;
  context.body_rotation_envelope_free = false;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));

  context.body_rotation_envelope_free = true;
  context.planar_speed = context.maximum_planar_speed + 0.001;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));
}

TEST(ActiveSensing, StartGateIsOncePerPathAndFailsClosedOnEverySafetyGate)
{
  auto context = validStartContext();
  context.consumed_reference_path_stamp_ns = context.reference_path_stamp_ns;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));
  context.reference_path_stamp_ns = 101;
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  const auto expect_blocked = [](auto mutate) {
    auto blocked = validStartContext();
    mutate(blocked);
    EXPECT_FALSE(scan_planner::activeSensingMayStart(blocked));
  };
  expect_blocked([](auto &value) { value.planner_attempted = false; });
  expect_blocked([](auto &value) { value.reference_mode = false; });
  expect_blocked([](auto &value) { value.reference_path_stamp_ns = 0; });
  expect_blocked([](auto &value) { value.inputs_fresh = false; });
  expect_blocked([](auto &value) { value.stair_gate_clear = false; });
  expect_blocked(
      [](auto &value) { value.final_hold_lifecycle_clear = false; });
  expect_blocked(
      [](auto &value) { value.global_replan_gate_clear = false; });
  expect_blocked(
      [](auto &value) { value.body_rotation_envelope_free = false; });
  expect_blocked([](auto &value) { value.planar_speed = 0.031; });
  expect_blocked([](auto &value) {
    value.planar_speed = std::numeric_limits<double>::quiet_NaN();
  });
}

TEST(ActiveSensing, FinalRollingWindowMayStartUntilFinalHoldLifecycleBegins)
{
  auto context = validStartContext();

  // helper 刻意不接收 local_target_is_final_：它只表示当前滚动窗口触及
  // 最终目标，并不等价于机器人已到达或 final-hold 生命周期已经开始。
  EXPECT_TRUE(scan_planner::activeSensingMayStart(context));

  context.final_hold_lifecycle_clear = false;
  EXPECT_FALSE(scan_planner::activeSensingMayStart(context));
}

TEST(ActiveSensing, RuntimeGateFailsClosedOnIdentityInputsAndDrift)
{
  scan_planner::ActiveSensingRuntimeContext context;
  context.fsm_active = true;
  context.expected_reference_path_stamp_ns = 100;
  context.current_reference_path_stamp_ns = 100;
  context.path_available = true;
  context.inputs_fresh = true;
  context.stair_gate_clear = true;
  context.final_hold_lifecycle_clear = true;
  context.global_replan_gate_clear = true;
  context.body_rotation_envelope_free = true;
  context.position_drift = 0.01;
  context.maximum_position_drift = 0.04;
  context.planar_speed = 0.01;
  context.maximum_planar_speed = 0.03;
  context.latest_trajectory_identity_matches = true;
  EXPECT_TRUE(scan_planner::activeSensingMayContinue(context));

  const auto expect_blocked = [&context](auto mutate) {
    auto blocked = context;
    mutate(blocked);
    EXPECT_FALSE(scan_planner::activeSensingMayContinue(blocked));
  };
  expect_blocked([](auto &value) {
    value.current_reference_path_stamp_ns = 101;
  });
  expect_blocked([](auto &value) { value.inputs_fresh = false; });
  expect_blocked([](auto &value) { value.stair_gate_clear = false; });
  expect_blocked([](auto &value) {
    value.global_replan_gate_clear = false;
  });
  expect_blocked([](auto &value) {
    value.body_rotation_envelope_free = false;
  });
  expect_blocked([](auto &value) { value.position_drift = 0.041; });
  expect_blocked([](auto &value) {
    value.latest_trajectory_identity_matches = false;
  });
}

TEST(ActiveSensing, BuildsAuditableYawOnlyNonFinalTrajectory)
{
  const auto trajectory =
      scan_planner::buildActiveSensingTrajectoryMessage(
          "world", stamp(20), stamp(10), 8,
          Eigen::Vector3d(1.0, 2.0, 0.3), 3.10, 0.20, 0.20, 2.75);
  ASSERT_TRUE(trajectory.has_value());
  EXPECT_FALSE(trajectory->is_final);
  EXPECT_FALSE(trajectory->emergency_stop);
  ASSERT_GE(trajectory->pos_pts.size(), 6U);
  for (const auto &point : trajectory->pos_pts)
  {
    EXPECT_DOUBLE_EQ(point.x, 1.0);
    EXPECT_DOUBLE_EQ(point.y, 2.0);
    EXPECT_DOUBLE_EQ(point.z, 0.3);
  }
  ASSERT_EQ(trajectory->yaw_pts.size(), 2U);
  EXPECT_DOUBLE_EQ(trajectory->yaw_pts[0], 3.10);
  EXPECT_DOUBLE_EQ(trajectory->yaw_pts[1], 3.30);
  EXPECT_DOUBLE_EQ(trajectory->yaw_dt, 1.0);
  ASSERT_EQ(trajectory->knots.size(), 10U);
  EXPECT_NEAR(
      trajectory->knots[6] - trajectory->knots[3], 2.75, 1.0e-12);
}

TEST(ActiveSensing, MessageBuilderRejectsUnsafeYawAndInvalidIdentity)
{
  const Eigen::Vector3d position(0.0, 0.0, 0.3);
  EXPECT_FALSE(scan_planner::buildActiveSensingTrajectoryMessage(
      "world", stamp(20), stamp(10), 8, position,
      0.0, 0.23, 0.20, 2.0).has_value());
  EXPECT_FALSE(scan_planner::buildActiveSensingTrajectoryMessage(
      "world", stamp(20), stamp(10), 8, position,
      0.0, 0.20, 0.21, 2.0).has_value());
  EXPECT_FALSE(scan_planner::buildActiveSensingTrajectoryMessage(
      "world", stamp(20), stamp(10), 0, position,
      0.0, 0.20, 0.20, 2.0).has_value());
  EXPECT_FALSE(scan_planner::buildActiveSensingTrajectoryMessage(
      "world", stamp(20), stamp(10), 8, position,
      0.0, 0.20, 0.20, 0.5).has_value());
}

TEST(ActiveSensing, ControllerAcceptanceRequiresExactTrajectoryIdentity)
{
  using Action = scan_planner::ActiveSensingControllerAction;
  using Status = scan_planner_msgs::msg::ControllerStatus;
  const scan_planner::ActiveSensingTrajectoryIdentity expected{
      10000000000LL, 20000000000LL, 20000000000LL, 8};
  Status status;
  status.event = Status::EVENT_ACCEPTED;
  status.reference_path_stamp = stamp(10);
  status.bspline_header_stamp = stamp(20);
  status.start_time = stamp(20);
  status.traj_id = 8;
  status.accepted = true;
  status.trajectory_valid = true;
  status.active_sensing_yaw_only = true;
  status.state = Status::STATE_ALIGNING_YAW;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kAccepted);

  status.active_sensing_yaw_only = false;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kFailClosed);
  status.active_sensing_yaw_only = true;

  status.state = Status::STATE_WAITING_FOR_TRAJECTORY;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kFailClosed);
  status.state = Status::STATE_TRACKING;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kAccepted);

  status.traj_id = 9;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kFailClosed);
  status.traj_id = 8;
  status.event = Status::EVENT_REJECTED;
  status.candidate_present = true;
  status.candidate_reference_path_stamp = stamp(10);
  status.candidate_bspline_header_stamp = stamp(20);
  status.candidate_start_time = stamp(20);
  status.candidate_traj_id = 8;
  EXPECT_EQ(
      scan_planner::decideActiveSensingControllerAction(
          status, expected, false),
      Action::kFailClosed);
}

TEST(ActiveSensing, TypedDiagnosticsCarryBoundedLifecycleEvidence)
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  scan_planner::ActiveSensingDiagnosticsSnapshot snapshot;
  snapshot.event = Diagnostics::ACTIVE_SENSING_EVENT_YAW_STABLE;
  snapshot.start_yaw = 3.10;
  snapshot.yaw_offset = 0.20;
  snapshot.target_yaw = std::atan2(
      std::sin(snapshot.start_yaw + snapshot.yaw_offset),
      std::cos(snapshot.start_yaw + snapshot.yaw_offset));
  snapshot.yaw_rate = 0.20;
  snapshot.settle_stamp_ns = 21000000001LL;
  snapshot.settle_yaw_error = 0.02;
  snapshot.settle_angular_speed = 0.05;
  snapshot.stable_duration = 0.10;
  snapshot.fusion_baseline = 41U;
  snapshot.fusion_current = 41U;
  snapshot.reason = "yaw 已稳定";

  Diagnostics diagnostics;
  ASSERT_TRUE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  EXPECT_TRUE(diagnostics.active_sensing);
  EXPECT_EQ(
      diagnostics.active_sensing_event,
      Diagnostics::ACTIVE_SENSING_EVENT_YAW_STABLE);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_start_yaw, 3.10);
  EXPECT_NEAR(
      diagnostics.active_sensing_target_yaw,
      snapshot.target_yaw, 1.0e-12);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_yaw_offset, 0.20);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_yaw_rate, 0.20);
  EXPECT_EQ(diagnostics.active_sensing_settle_stamp.sec, 21);
  EXPECT_EQ(diagnostics.active_sensing_settle_stamp.nanosec, 1U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_baseline, 41U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_current, 41U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_distinct, 0U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_required, 3U);
  EXPECT_FALSE(diagnostics.active_sensing_completed);
  EXPECT_FALSE(diagnostics.active_sensing_failed);
}

TEST(ActiveSensing, OrdinaryDiagnosticsKeepActiveEvidenceAtDefaults)
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  const Diagnostics diagnostics;
  EXPECT_FALSE(diagnostics.active_sensing);
  EXPECT_EQ(
      diagnostics.active_sensing_event,
      Diagnostics::ACTIVE_SENSING_EVENT_NONE);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_start_yaw, 0.0);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_target_yaw, 0.0);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_yaw_offset, 0.0);
  EXPECT_DOUBLE_EQ(diagnostics.active_sensing_yaw_rate, 0.0);
  EXPECT_EQ(diagnostics.active_sensing_settle_stamp.sec, 0);
  EXPECT_EQ(diagnostics.active_sensing_settle_stamp.nanosec, 0U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_baseline, 0U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_current, 0U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_distinct, 0U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_required, 0U);
  EXPECT_FALSE(diagnostics.active_sensing_completed);
  EXPECT_FALSE(diagnostics.active_sensing_failed);
  EXPECT_TRUE(diagnostics.active_sensing_reason.empty());
}

TEST(ActiveSensing, TypedDiagnosticsRequireThreePostSettleFusionsToComplete)
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  scan_planner::ActiveSensingDiagnosticsSnapshot snapshot;
  snapshot.event = Diagnostics::ACTIVE_SENSING_EVENT_COMPLETED;
  snapshot.start_yaw = 0.5;
  snapshot.yaw_offset = -0.20;
  snapshot.target_yaw = 0.3;
  snapshot.yaw_rate = 0.20;
  snapshot.settle_stamp_ns = 22000000000LL;
  snapshot.settle_yaw_error = 0.01;
  snapshot.settle_angular_speed = 0.04;
  snapshot.stable_duration = 0.11;
  snapshot.fusion_baseline = 7U;
  snapshot.fusion_current = 10U;
  snapshot.fusion_distinct = 3U;
  snapshot.completed = true;
  snapshot.reason = "已完成三个真实融合";

  scan_planner_msgs::msg::BsplineDiagnostics diagnostics;
  EXPECT_TRUE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  EXPECT_TRUE(diagnostics.active_sensing_completed);

  snapshot.fusion_distinct = 2U;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  snapshot.fusion_distinct = 3U;
  snapshot.fusion_current = 9U;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
}

TEST(ActiveSensing, TypedFailureSnapshotRetainsAvailablePhaseEvidence)
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  scan_planner::ActiveSensingDiagnosticsSnapshot snapshot;
  snapshot.event = Diagnostics::ACTIVE_SENSING_EVENT_FAILED;
  snapshot.start_yaw = -0.4;
  snapshot.yaw_offset = 0.20;
  snapshot.target_yaw = -0.2;
  snapshot.yaw_rate = 0.20;
  snapshot.failed = true;
  snapshot.reason = "Path 换代";

  Diagnostics diagnostics;
  ASSERT_TRUE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  EXPECT_TRUE(diagnostics.active_sensing_failed);
  EXPECT_EQ(diagnostics.active_sensing_settle_stamp.sec, 0);
  EXPECT_EQ(diagnostics.active_sensing_fusion_required, 3U);

  snapshot.settle_stamp_ns = 23000000000LL;
  snapshot.settle_yaw_error = 0.01;
  snapshot.settle_angular_speed = 0.03;
  snapshot.stable_duration = 0.12;
  snapshot.fusion_baseline = 90U;
  snapshot.fusion_current = 92U;
  snapshot.fusion_distinct = 2U;
  ASSERT_TRUE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  EXPECT_EQ(diagnostics.active_sensing_fusion_baseline, 90U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_current, 92U);
  EXPECT_EQ(diagnostics.active_sensing_fusion_distinct, 2U);
}

TEST(ActiveSensing, PathReplacementPublishesFailureBeforeResetAndStop)
{
  std::vector<int> order;
  const auto result =
      scan_planner::terminateActiveSensingBeforeTrajectoryReplacement(
          [&order]() {
            order.push_back(1);
            return false;
          },
          [&order]() { order.push_back(2); },
          [&order]() {
            order.push_back(3);
            return true;
          });

  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
  EXPECT_FALSE(result.failure_snapshot_published);
  EXPECT_TRUE(result.stop_published);
}

TEST(ActiveSensing, TypedDiagnosticsRejectUnsafeOrContradictorySnapshots)
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  scan_planner::ActiveSensingDiagnosticsSnapshot snapshot;
  snapshot.event = Diagnostics::ACTIVE_SENSING_EVENT_STARTED;
  snapshot.start_yaw = 0.2;
  snapshot.target_yaw = 0.4;
  snapshot.yaw_offset = 0.2;
  snapshot.yaw_rate = 0.2;
  snapshot.reason = "开始";
  Diagnostics diagnostics;
  ASSERT_TRUE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));

  snapshot.yaw_offset = 0.221;
  snapshot.target_yaw = 0.421;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  snapshot.yaw_offset = 0.2;
  snapshot.target_yaw = 0.4;
  snapshot.completed = true;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  snapshot.completed = false;
  snapshot.settle_stamp_ns = 1;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));

  snapshot.event = Diagnostics::ACTIVE_SENSING_EVENT_YAW_STABLE;
  snapshot.settle_yaw_error = 0.021;
  snapshot.settle_angular_speed = 0.04;
  snapshot.stable_duration = 0.10;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  snapshot.settle_yaw_error = 0.01;
  snapshot.settle_angular_speed = 0.051;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
  snapshot.settle_angular_speed = 0.04;
  snapshot.stable_duration = 0.099;
  EXPECT_FALSE(scan_planner::populateActiveSensingDiagnostics(
      diagnostics, snapshot));
}

TEST(ActiveSensing, ControllerAcceptanceDeadlinesAreInclusiveOnly)
{
  constexpr std::int64_t publish_ns = 20000000000LL;
  constexpr std::int64_t accept_timeout_ns = 500000000LL;
  constexpr std::int64_t total_timeout_ns = 2750000000LL;

  EXPECT_TRUE(scan_planner::activeSensingAcceptanceIsTimely(
      publish_ns + accept_timeout_ns, publish_ns,
      accept_timeout_ns, total_timeout_ns));
  EXPECT_FALSE(scan_planner::activeSensingAcceptanceIsTimely(
      publish_ns + accept_timeout_ns + 1, publish_ns,
      accept_timeout_ns, total_timeout_ns));

  // 两个截止点必须同时满足；即使接受窗口更长，总时限晚 1 ns 也拒绝。
  EXPECT_FALSE(scan_planner::activeSensingAcceptanceIsTimely(
      publish_ns + total_timeout_ns + 1, publish_ns,
      total_timeout_ns + 100, total_timeout_ns));
  EXPECT_FALSE(scan_planner::activeSensingAcceptanceIsTimely(
      publish_ns - 1, publish_ns,
      accept_timeout_ns, total_timeout_ns));
}
