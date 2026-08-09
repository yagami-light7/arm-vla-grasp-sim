#include <gtest/gtest.h>

#include <plan_manage/planning_status.h>

using scan_planner::GlobalReplanGenerationGate;
using scan_planner::ScanPlanningEvent;
using scan_planner::ScanPlanningState;
using scan_planner::kScanStairExecutionInhibitedReason;
using scan_planner::kScanStairFreezeFrameMismatchFault;
using scan_planner::kScanStairFreezeProtocolFault;
using scan_planner::kScanStairFreezeSnapshotTimeoutFault;
using scan_planner::kScanStairResumeWaitingReason;
using scan_planner::kScanStairStopPublishFault;
using scan_planner::planningStatusPolicy;

TEST(PlanningStatus, EventPolicyDoesNotDependOnReasonText)
{
  const auto normal = planningStatusPolicy(
      ScanPlanningEvent::kTrajectoryPublished, 0, 5, false);
  EXPECT_EQ(normal.state, ScanPlanningState::kTracking);
  EXPECT_FALSE(normal.stop_required);
  EXPECT_FALSE(normal.global_replan_recommended);

  const auto waiting = planningStatusPolicy(
      ScanPlanningEvent::kReferenceAccepted, 0, 5, false);
  EXPECT_EQ(waiting.state, ScanPlanningState::kPlanning);
  EXPECT_TRUE(waiting.stop_required);
  EXPECT_FALSE(waiting.global_replan_recommended);
}

TEST(PlanningStatus, EveryFailureIsVisibleButOnlyThresholdRequiresStop)
{
  const auto fourth = planningStatusPolicy(
      ScanPlanningEvent::kPlanningFailed, 4, 5, true);
  EXPECT_EQ(fourth.state, ScanPlanningState::kPlanning);
  EXPECT_FALSE(fourth.stop_required);
  EXPECT_FALSE(fourth.global_replan_recommended);

  const auto fifth = planningStatusPolicy(
      ScanPlanningEvent::kPlanningFailed, 5, 5, true);
  EXPECT_TRUE(fifth.stop_required);
  EXPECT_TRUE(fifth.global_replan_recommended);

  const auto manual_mode = planningStatusPolicy(
      ScanPlanningEvent::kPlanningFailed, 5, 5, false);
  EXPECT_TRUE(manual_mode.stop_required);
  EXPECT_FALSE(manual_mode.global_replan_recommended);
}

TEST(PlanningStatus, CollisionAndEmergencyUseStructuredReplanLatch)
{
  const auto collision = planningStatusPolicy(
      ScanPlanningEvent::kPredictedCollision, 0, 5, true);
  EXPECT_EQ(collision.state, ScanPlanningState::kEmergencyStop);
  EXPECT_TRUE(collision.stop_required);
  EXPECT_TRUE(collision.global_replan_recommended);

  const auto local_emergency = planningStatusPolicy(
      ScanPlanningEvent::kEmergencyStop, 0, 5, false);
  EXPECT_TRUE(local_emergency.stop_required);
  EXPECT_FALSE(local_emergency.global_replan_recommended);
}

TEST(PlanningStatus, GoalAndStairStatesAlwaysRequireStopWithoutGlobalReplan)
{
  for (const auto event : {
           ScanPlanningEvent::kGoalHold,
           ScanPlanningEvent::kStairInhibited,
           ScanPlanningEvent::kStairResumeWaiting})
  {
    const auto policy = planningStatusPolicy(event, 0, 5, false);
    EXPECT_TRUE(policy.stop_required);
    EXPECT_FALSE(policy.global_replan_recommended);
  }
}

TEST(PlanningStatus, StairReasonContractUsesStableMachineTokens)
{
  EXPECT_STREQ(
      kScanStairExecutionInhibitedReason,
      "scan_stair_execution_inhibited");
  EXPECT_STREQ(kScanStairResumeWaitingReason, "scan_stair_resume_waiting");
  EXPECT_STREQ(
      kScanStairFreezeFrameMismatchFault,
      "scan_stair_freeze_frame_mismatch_fault");
  EXPECT_STREQ(
      kScanStairFreezeProtocolFault,
      "scan_stair_freeze_protocol_fault");
  EXPECT_STREQ(
      kScanStairFreezeSnapshotTimeoutFault,
      "scan_stair_freeze_snapshot_timeout_fault");
  EXPECT_STREQ(
      kScanStairStopPublishFault,
      "scan_stair_stop_publish_fault");
}

TEST(PlanningStatus, GlobalReplanGateRequiresStrictlyNewPathGeneration)
{
  GlobalReplanGenerationGate gate;
  gate.require(100);
  EXPECT_TRUE(gate.required());
  EXPECT_FALSE(gate.isStrictReplacement(99));
  EXPECT_FALSE(gate.isStrictReplacement(100));
  EXPECT_TRUE(gate.isStrictReplacement(101));
  EXPECT_FALSE(gate.clearForStrictReplacement(100));
  EXPECT_TRUE(gate.required());
  EXPECT_TRUE(gate.clearForStrictReplacement(101));
  EXPECT_FALSE(gate.required());
}

TEST(PlanningStatus, RepeatedRequestKeepsHighestBlockedGeneration)
{
  GlobalReplanGenerationGate gate;
  gate.require(100);
  gate.require(90);
  EXPECT_EQ(gate.blockedReferencePathStampNs(), 100);
  gate.require(120);
  EXPECT_EQ(gate.blockedReferencePathStampNs(), 120);
  EXPECT_FALSE(gate.clearForStrictReplacement(120));
  EXPECT_TRUE(gate.clearForStrictReplacement(121));
}
