#include <gtest/gtest.h>

#include <plan_manage/reference_execution.h>

using scan_planner::StairPlanningFreezeGate;
using scan_planner::StairExecutionFreezeSnapshot;
using scan_planner::StairExecutionFreezeMailbox;
using scan_planner::StairExecutionFreezeMailboxMessage;
using scan_planner::StairFreezeFreshnessUpdate;
using scan_planner::StairPlanningFreezeUpdate;
using scan_planner::activeReferenceAvailableForFreezeBinding;
using scan_planner::advanceReferencePathProgressDuringStairFreeze;
using scan_planner::buildReferencePathArcLengths;
using scan_planner::resolveReferencePathBinding;
using scan_planner::trajectoryTimeFrozen;

TEST(StairPlanningFreeze, TrajectoryClockUsesControllerOrStairFreeze)
{
  EXPECT_FALSE(trajectoryTimeFrozen(false, false));
  EXPECT_TRUE(trajectoryTimeFrozen(true, false));
  EXPECT_TRUE(trajectoryTimeFrozen(false, true));
  EXPECT_TRUE(trajectoryTimeFrozen(true, true));
}

TEST(StairPlanningFreeze, PendingValidatedPathOwnsFreezeAndStatusIdentity)
{
  const auto pending = resolveReferencePathBinding(400, true, 300, true);
  EXPECT_TRUE(pending.available);
  EXPECT_TRUE(pending.pending);
  EXPECT_EQ(pending.stamp_ns, 400);

  const auto active = resolveReferencePathBinding(0, false, 300, true);
  EXPECT_TRUE(active.available);
  EXPECT_FALSE(active.pending);
  EXPECT_EQ(active.stamp_ns, 300);

  const auto absent = resolveReferencePathBinding(0, false, 0, false);
  EXPECT_FALSE(absent.available);
  EXPECT_FALSE(absent.pending);
  EXPECT_EQ(absent.stamp_ns, 0);
}

namespace
{
StairExecutionFreezeSnapshot snapshot(
    const bool frozen,
    const std::uint64_t sequence,
    const int64_t header_stamp_ns = 900,
    const int64_t path_stamp_ns = 300,
    const std::string &writer_id = "isaac",
    const std::string &writer_epoch = "epoch-a")
{
  return StairExecutionFreezeSnapshot{
      frozen, header_stamp_ns, path_stamp_ns,
      writer_id, writer_epoch, sequence};
}
} // namespace

TEST(StairPlanningFreeze, FinalHoldRetainsOnlyItsExactActivePathBinding)
{
  EXPECT_TRUE(activeReferenceAvailableForFreezeBinding(true, true, false));
  EXPECT_TRUE(activeReferenceAvailableForFreezeBinding(true, false, true));
  EXPECT_FALSE(activeReferenceAvailableForFreezeBinding(true, false, false));
  EXPECT_FALSE(activeReferenceAvailableForFreezeBinding(false, false, true));

  const auto final_hold = resolveReferencePathBinding(
      0, false, 300,
      activeReferenceAvailableForFreezeBinding(true, false, true));
  ASSERT_TRUE(final_hold.available);
  EXPECT_FALSE(final_hold.pending);
  EXPECT_EQ(final_hold.stamp_ns, 300);

  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(final_hold.stamp_ns);
  EXPECT_EQ(
      gate.updateTyped(
          snapshot(true, 1), 10, 20, final_hold.stamp_ns,
          final_hold.available, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  EXPECT_EQ(
      gate.updateTyped(
          snapshot(true, 2, 910, 301), 10, 20,
          final_hold.stamp_ns, final_hold.available, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_EQ(gate.protocolFault(), "reference_path_identity_mismatch");
}

TEST(StairPlanningFreeze, TypedRefreshIsIdempotentButSequenceMustIncrease)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(
      gate.updateTyped(snapshot(true, 1), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(
      gate.updateTyped(snapshot(true, 2, 910), 100, 200, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kDuplicate);
  EXPECT_TRUE(gate.protocolValid());
  EXPECT_FALSE(gate.resumeWaiting());

  EXPECT_EQ(
      gate.updateTyped(snapshot(false, 3, 920), 11, 21, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kResumeBarrierStarted);
  EXPECT_EQ(gate.odometryBaselineNs(), 11);
  EXPECT_EQ(gate.observationBaselineNs(), 21);
  EXPECT_EQ(gate.referencePathStampNs(), 300);
  EXPECT_EQ(
      gate.updateTyped(snapshot(false, 3, 930), 999, 999, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(gate.protocolFault(), "non_monotonic_sequence");
  EXPECT_EQ(gate.odometryBaselineNs(), 0);
  EXPECT_EQ(gate.observationBaselineNs(), 0);
  EXPECT_EQ(gate.referencePathStampNs(), 0);
}

TEST(StairPlanningFreeze, InitialAuthenticatedFalseDoesNotCreateResumeBarrier)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);

  EXPECT_EQ(
      gate.updateTyped(
          snapshot(false, 1), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInitialUnfrozen);
  EXPECT_TRUE(gate.protocolValid());
  EXPECT_FALSE(gate.frozen());
  EXPECT_FALSE(gate.resumeWaiting());
  EXPECT_FALSE(gate.planningInhibited());
  EXPECT_EQ(gate.odometryBaselineNs(), 0);
  EXPECT_EQ(gate.observationBaselineNs(), 0);
}

TEST(StairPlanningFreeze, PendingFreezeSurvivesSameStampActivation)
{
  StairPlanningFreezeGate gate;
  // 已校验 pending Path 在首帧地图到达前先绑定并接收冻结。
  gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      gate.updateTyped(snapshot(true, 1), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  ASSERT_TRUE(gate.protocolValid());
  ASSERT_TRUE(gate.frozen());

  // pending -> active 使用完全相同的 Path stamp，不得丢失 writer/sequence。
  gate.bindReferenceGeneration(300);
  EXPECT_TRUE(gate.protocolValid());
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(gate.activeReferencePathStampNs(), 300);
  EXPECT_EQ(gate.lastSequence(), 1U);
  EXPECT_EQ(gate.writerId(), "isaac");
  EXPECT_EQ(gate.writerEpoch(), "epoch-a");
}

TEST(StairPlanningFreeze, MissingOrOldPendingIdentityRejectsWithReason)
{
  StairPlanningFreezeGate absent_gate;
  EXPECT_EQ(
      absent_gate.updateTyped(
          snapshot(false, 1), 10, 20, 0, false, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_EQ(
      absent_gate.protocolFault(), "reference_path_identity_mismatch");
  EXPECT_FALSE(absent_gate.frozen());

  StairPlanningFreezeGate pending_gate;
  pending_gate.bindReferenceGeneration(400);
  EXPECT_EQ(
      pending_gate.updateTyped(
          snapshot(false, 1, 900, 300),
          10, 20, 400, true, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_EQ(
      pending_gate.protocolFault(), "reference_path_identity_mismatch");
  EXPECT_TRUE(pending_gate.frozen());
}

TEST(StairPlanningFreeze, ResumeRequiresBothFreshInputsAndSamePathGeneration)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      gate.updateTyped(snapshot(true, 1), 100, 200, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  ASSERT_EQ(
      gate.updateTyped(snapshot(false, 2, 910), 100, 200, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kResumeBarrierStarted);

  EXPECT_FALSE(gate.resumeInputsReady(100, 201, 300, true, true));
  EXPECT_FALSE(gate.resumeInputsReady(101, 200, 300, true, true));
  EXPECT_FALSE(gate.resumeInputsReady(101, 201, 301, true, true));
  EXPECT_FALSE(gate.resumeInputsReady(101, 201, 300, false, true));
  EXPECT_FALSE(gate.resumeInputsReady(101, 201, 300, true, false));
  EXPECT_TRUE(gate.resumeInputsReady(101, 201, 300, true, true));
}

TEST(StairPlanningFreeze, EmptyReferenceCancelsPendingResume)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      gate.updateTyped(snapshot(true, 1), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  ASSERT_EQ(
      gate.updateTyped(snapshot(false, 2, 910), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kResumeBarrierStarted);
  gate.clearReferenceGeneration();
  EXPECT_FALSE(gate.planningInhibited());
  EXPECT_FALSE(gate.resumeInputsReady(11, 21, 30, true, true));
  EXPECT_EQ(gate.activeReferencePathStampNs(), 0);
  EXPECT_EQ(
      gate.refreshFreshness(2000, 200, 50),
      StairFreezeFreshnessUpdate::kNoChange);
}

TEST(StairPlanningFreeze, ReleaseWithoutReferenceDoesNotCreateBarrier)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  EXPECT_EQ(
      gate.updateTyped(snapshot(false, 1), 10, 20, 0, false, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.planningInhibited());
}

TEST(StairPlanningFreeze, RejectsWrongPathAndSecondWriterFalseFailClosed)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      gate.updateTyped(snapshot(false, 1), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInitialUnfrozen);
  gate.clearResumeBarrier();
  ASSERT_FALSE(gate.planningInhibited());

  EXPECT_EQ(
      gate.updateTyped(snapshot(false, 2, 930, 301), 11, 21, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(gate.protocolFault(), "reference_path_identity_mismatch");

  EXPECT_EQ(
      gate.updateTyped(
          snapshot(false, 2, 940, 300, "other", "epoch-b"),
          11, 21, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(gate.protocolFault(), "second_writer_rejected");
  EXPECT_EQ(gate.writerId(), "isaac");
  EXPECT_EQ(gate.writerEpoch(), "epoch-a");
}

TEST(StairPlanningFreeze, StaleFalseAndFreshnessExpiryFailClosed)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  EXPECT_EQ(
      gate.updateTyped(snapshot(false, 1, 700), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.frozen());
  EXPECT_EQ(gate.protocolFault(), "stale_or_invalid_header_stamp");

  ASSERT_EQ(
      gate.updateTyped(snapshot(false, 1, 900), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInitialUnfrozen);
  ASSERT_FALSE(gate.resumeWaiting());
  EXPECT_EQ(
      gate.refreshFreshness(1100, 200, 50),
      StairFreezeFreshnessUpdate::kNoChange);
  EXPECT_EQ(
      gate.refreshFreshness(1101, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutCandidateStarted);
  EXPECT_TRUE(gate.frozen());
  EXPECT_FALSE(gate.resumeWaiting());
  EXPECT_TRUE(gate.protocolValid());
  EXPECT_FALSE(gate.authenticatedSnapshotAvailableForStatus());
  EXPECT_TRUE(gate.protocolFault().empty());
  EXPECT_FALSE(gate.resumeInputsReady(11, 21, 300, true, true));
  EXPECT_TRUE(gate.timeoutCandidateActive());
  EXPECT_FALSE(gate.timeoutFaultConfirmed());
  EXPECT_EQ(gate.timeoutCandidateSequence(), 1U);
  EXPECT_EQ(gate.timeoutCandidateHeaderStampNs(), 900);
  EXPECT_EQ(gate.timeoutCandidateFirstObservedNowNs(), 1101);

  // 同一仿真时刻的重复 timer 和恰好到达宽限边界均不能确认。
  EXPECT_EQ(
      gate.refreshFreshness(1101, 200, 50),
      StairFreezeFreshnessUpdate::kNoChange);
  EXPECT_EQ(
      gate.refreshFreshness(1151, 200, 50),
      StairFreezeFreshnessUpdate::kNoChange);
  EXPECT_EQ(
      gate.refreshFreshness(1152, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutConfirmed);
  EXPECT_TRUE(gate.timeoutFaultConfirmed());
  EXPECT_FALSE(gate.protocolValid());
  EXPECT_FALSE(gate.resumeWaiting());
  EXPECT_EQ(gate.protocolFault(), "stair_freeze_snapshot_timeout");
  EXPECT_EQ(
      gate.refreshFreshness(1200, 200, 50),
      StairFreezeFreshnessUpdate::kNoChange);
}

TEST(StairPlanningFreeze, FreshHigherSequenceCancelsTimeoutCandidate)
{
  StairPlanningFreezeGate frozen_gate;
  frozen_gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      frozen_gate.updateTyped(
          snapshot(true, 1, 900), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInhibited);
  ASSERT_EQ(
      frozen_gate.refreshFreshness(1101, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutCandidateStarted);
  EXPECT_EQ(
      frozen_gate.updateTyped(
          snapshot(true, 2, 1110), 11, 21, 300, true, 1110, 200),
      StairPlanningFreezeUpdate::kDuplicate);
  EXPECT_FALSE(frozen_gate.timeoutCandidateActive());
  EXPECT_FALSE(frozen_gate.timeoutFaultConfirmed());
  EXPECT_TRUE(frozen_gate.protocolValid());
  EXPECT_TRUE(frozen_gate.frozen());

  StairPlanningFreezeGate released_gate;
  released_gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      released_gate.updateTyped(
          snapshot(false, 1, 900), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInitialUnfrozen);
  released_gate.clearResumeBarrier();
  ASSERT_FALSE(released_gate.planningInhibited());
  ASSERT_EQ(
      released_gate.refreshFreshness(1101, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutCandidateStarted);
  EXPECT_TRUE(released_gate.frozen());
  EXPECT_TRUE(released_gate.planningInhibited());
  EXPECT_FALSE(
      released_gate.authenticatedSnapshotAvailableForStatus());
  EXPECT_EQ(
      released_gate.updateTyped(
          snapshot(false, 2, 1110), 11, 21, 300, true, 1110, 200),
      StairPlanningFreezeUpdate::kDuplicate);
  EXPECT_FALSE(released_gate.timeoutCandidateActive());
  EXPECT_TRUE(released_gate.protocolValid());
  EXPECT_TRUE(released_gate.authenticatedSnapshotAvailableForStatus());
  EXPECT_FALSE(released_gate.frozen());
  EXPECT_FALSE(released_gate.resumeWaiting());
  EXPECT_FALSE(released_gate.planningInhibited());
  EXPECT_EQ(released_gate.odometryBaselineNs(), 0);
  EXPECT_EQ(released_gate.observationBaselineNs(), 0);
}

TEST(StairPlanningFreeze, InvalidSnapshotsDoNotCancelTimeoutCandidate)
{
  StairPlanningFreezeGate gate;
  gate.bindReferenceGeneration(300);
  ASSERT_EQ(
      gate.updateTyped(
          snapshot(false, 1, 900), 10, 20, 300, true, 1000, 200),
      StairPlanningFreezeUpdate::kInitialUnfrozen);
  ASSERT_EQ(
      gate.refreshFreshness(1101, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutCandidateStarted);

  EXPECT_EQ(
      gate.updateTyped(
          snapshot(false, 2, 1110, 300, "other", "epoch-b"),
          11, 21, 300, true, 1110, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.timeoutCandidateActive());
  EXPECT_EQ(gate.timeoutCandidateSequence(), 1U);
  EXPECT_EQ(gate.timeoutCandidateHeaderStampNs(), 900);

  EXPECT_EQ(
      gate.updateTyped(
          snapshot(false, 1, 1120),
          12, 22, 300, true, 1120, 200),
      StairPlanningFreezeUpdate::kProtocolRejected);
  EXPECT_TRUE(gate.timeoutCandidateActive());
  EXPECT_EQ(
      gate.refreshFreshness(1152, 200, 50),
      StairFreezeFreshnessUpdate::kTimeoutConfirmed);
  EXPECT_EQ(gate.protocolFault(), "stair_freeze_snapshot_timeout");
}

TEST(StairPlanningFreeze, ProgressAdvancesMonotonicallyAcrossLongFrozenSection)
{
  std::vector<Eigen::Vector3d> points;
  for (int index = 0; index <= 100; ++index)
    points.emplace_back(0.1 * index, 0.0, 0.3);
  const auto arc_lengths = buildReferencePathArcLengths(points);
  ASSERT_EQ(arc_lengths.size(), points.size());

  double progress = 0.0;
  for (int index = 1; index <= 80; ++index)
  {
    const auto projection = advanceReferencePathProgressDuringStairFreeze(
        points, arc_lengths,
        Eigen::Vector3d(0.1 * index, 0.02, 0.3), progress, 0.5);
    ASSERT_TRUE(projection.valid);
    ASSERT_GE(projection.progress_s, progress);
    progress = projection.progress_s;
  }
  EXPECT_NEAR(progress, 8.0, 1.0e-9);

  const auto old_position = advanceReferencePathProgressDuringStairFreeze(
      points, arc_lengths, Eigen::Vector3d(2.0, 0.0, 0.3), progress, 0.5);
  EXPECT_FALSE(old_position.valid);
  EXPECT_NEAR(progress, 8.0, 1.0e-9);

  const auto off_path = advanceReferencePathProgressDuringStairFreeze(
      points, arc_lengths, Eigen::Vector3d(8.2, 1.0, 0.3), progress, 0.5);
  EXPECT_FALSE(off_path.valid);
}

TEST(StairPlanningFreeze, MailboxKeepsLatestSnapshotAndDrainsOnce)
{
  StairExecutionFreezeMailbox mailbox;
  for (std::uint64_t sequence = 1; sequence <= 3; ++sequence)
  {
    StairExecutionFreezeMailboxMessage message;
    message.frame_id = "world";
    message.snapshot.sequence = sequence;
    message.snapshot.frozen = sequence == 2;
    mailbox.push(std::move(message));
  }

  const auto first = mailbox.drain();
  ASSERT_TRUE(first.available);
  EXPECT_EQ(first.message.snapshot.sequence, 3U);
  EXPECT_FALSE(first.message.snapshot.frozen);
  EXPECT_EQ(first.coalesced_count, 2U);

  const auto second = mailbox.drain();
  EXPECT_FALSE(second.available);
  EXPECT_EQ(second.coalesced_count, 0U);
}

TEST(StairPlanningFreeze, MailboxLatestHeartbeatCarriesNewestHeader)
{
  StairExecutionFreezeMailbox mailbox;
  for (std::uint64_t sequence = 1; sequence <= 3; ++sequence)
  {
    StairExecutionFreezeMailboxMessage message;
    message.frame_id = "world";
    message.snapshot.reference_path_stamp_ns = 300;
    message.snapshot.writer_id = "isaac";
    message.snapshot.writer_epoch = "epoch-a";
    message.snapshot.sequence = sequence;
    message.snapshot.header_stamp_ns = 1000 + sequence;
    mailbox.push(std::move(message));
  }

  const auto drain = mailbox.drain();
  ASSERT_TRUE(drain.available);
  EXPECT_EQ(drain.message.snapshot.sequence, 3U);
  EXPECT_EQ(drain.message.snapshot.header_stamp_ns, 1003);
  EXPECT_EQ(drain.coalesced_count, 2U);
}

TEST(StairPlanningFreeze, MailboxNewPathSupersedesQueuedOldGenerationEdges)
{
  StairExecutionFreezeMailbox mailbox;
  for (std::uint64_t sequence = 1; sequence <= 2; ++sequence)
  {
    StairExecutionFreezeMailboxMessage old_message;
    old_message.frame_id = "world";
    old_message.snapshot.reference_path_stamp_ns = 300;
    old_message.snapshot.writer_id = "isaac";
    old_message.snapshot.writer_epoch = "epoch-a";
    old_message.snapshot.sequence = sequence;
    old_message.snapshot.header_stamp_ns = 1000 + sequence;
    old_message.snapshot.frozen = sequence == 2;
    mailbox.push(std::move(old_message));
  }
  StairExecutionFreezeMailboxMessage new_message;
  new_message.frame_id = "world";
  new_message.snapshot.reference_path_stamp_ns = 400;
  new_message.snapshot.writer_id = "isaac";
  new_message.snapshot.writer_epoch = "epoch-a";
  new_message.snapshot.sequence = 3;
  new_message.snapshot.header_stamp_ns = 1003;
  new_message.snapshot.frozen = false;
  mailbox.push(std::move(new_message));

  const auto drain = mailbox.drain();
  ASSERT_TRUE(drain.available);
  EXPECT_EQ(
      drain.message.snapshot.reference_path_stamp_ns, 400);
  EXPECT_EQ(drain.message.snapshot.sequence, 3U);
  EXPECT_EQ(drain.coalesced_count, 2U);
}
