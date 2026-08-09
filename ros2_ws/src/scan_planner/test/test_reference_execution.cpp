#include <gtest/gtest.h>

#include <plan_manage/reference_execution.h>

using scan_planner::ReferenceExecutionAction;
using scan_planner::ReferenceEmergencyRecoveryAction;
using scan_planner::ReferenceTargetSelection;
using scan_planner::FinalHoldControllerAction;
using scan_planner::FinalHoldControllerState;
using scan_planner::FinalHoldLifecycleState;
using scan_planner::FinalHoldStairResumeAction;
using scan_planner::FinalHoldTrajectoryIdentity;
using scan_planner::buildReferencePathArcLengths;
using scan_planner::buildReferencePathCorridorGuide;
using scan_planner::buildReferencePathGuide;
using scan_planner::decideReferenceExecutionAction;
using scan_planner::decideFinalHoldControllerAction;
using scan_planner::decideFinalHoldStairResumeAction;
using scan_planner::decideReferenceEmergencyRecovery;
using scan_planner::finalHoldBlocksStationaryPublication;
using scan_planner::finalHoldRecoveryTrajectoryIdAllowed;
using scan_planner::nonFinalReferenceTargetRequiresTerminalBrake;
using scan_planner::projectReferencePathProgress;
using scan_planner::referenceGoalHoldReady;
using scan_planner::ReferenceGoalHoldDwellState;
using scan_planner::updateReferenceGoalHoldDwell;
using scan_planner::referenceRetryReady;
using scan_planner::rememberPublishedFinalIdentity;
using scan_planner::reboundPlanRejectedBeforeInitialization;
using scan_planner::sampleReferencePathAtArcLength;
using scan_planner::sameFinalHoldIdentity;
using scan_planner::selectedReferenceTargetQualifiesAsTerminalCapture;
using scan_planner::selectReferenceTargetWithFreeRunway;
using scan_planner::selectReferenceTerminalCaptureTarget;
using scan_planner::shortReferenceReplanAllowed;
using scan_planner::shouldAdjustGlobalTargetForLocalOccupancy;
using scan_planner::terminalCaptureTargetFitsLocalWindow;
using scan_planner::transitionFinalHoldLifecycle;
using scan_planner::updateControllerAcceptedFinalIdentity;
using scan_planner::wasFinalIdentityPublished;

TEST(ReferenceExecution, ProjectsAndSamplesDenseStraightPathByArcLength)
{
  std::vector<Eigen::Vector3d> points;
  for (int index = 0; index <= 100; ++index)
    points.emplace_back(0.01 * index, 0.0, 0.2);

  const auto arc_lengths = buildReferencePathArcLengths(points);
  ASSERT_EQ(arc_lengths.size(), points.size());
  const auto projection = projectReferencePathProgress(
      points, arc_lengths, Eigen::Vector3d(0.23, 0.04, 0.2), 0.0, 0.6);
  ASSERT_TRUE(projection.valid);
  EXPECT_NEAR(projection.progress_s, 0.23, 1.0e-9);
  EXPECT_NEAR(projection.distance, 0.04, 1.0e-9);

  Eigen::Vector3d target;
  ASSERT_TRUE(sampleReferencePathAtArcLength(
      points, arc_lengths, projection.progress_s + 0.6, target));
  EXPECT_TRUE(target.isApprox(Eigen::Vector3d(0.83, 0.0, 0.2), 1.0e-9));
}

TEST(ReferenceExecution, ArcLengthTargetDoesNotCutAcrossRightAngle)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.3},
  };

  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto projection = projectReferencePathProgress(
      points, arc_lengths, Eigen::Vector3d(0.8, 0.0, 0.0), 0.7, 1.4);
  ASSERT_TRUE(projection.valid);
  Eigen::Vector3d target;
  ASSERT_TRUE(sampleReferencePathAtArcLength(
      points, arc_lengths, projection.progress_s + 0.6, target));
  EXPECT_NEAR(target.x(), 1.0, 1.0e-9);
  EXPECT_GT(target.y(), 0.35);
  EXPECT_LT(target.y(), 0.45);
}

TEST(ReferenceExecution, ThreeDimensionalProjectionSelectsCorrectFloor)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.0},
      {2.0, 0.0, 0.0},
      {2.0, 1.0, 1.5},
      {0.0, 1.0, 3.0},
      {0.0, 0.0, 3.0},
      {2.0, 0.0, 3.0},
  };

  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto projection = projectReferencePathProgress(
      points, arc_lengths, Eigen::Vector3d(0.5, 0.0, 3.0),
      0.0, arc_lengths.back());
  ASSERT_TRUE(projection.valid);
  EXPECT_EQ(projection.segment_index, 4U);
  EXPECT_NEAR(projection.distance, 0.0, 1.0e-9);
}

TEST(ReferenceExecution, ProjectionNeverRegressesCertifiedProgress)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto projection = projectReferencePathProgress(
      points, arc_lengths, Eigen::Vector3d(0.2, 0.0, 0.0), 0.7, 1.0);
  ASSERT_TRUE(projection.valid);
  EXPECT_NEAR(projection.progress_s, 0.7, 1.0e-9);
  EXPECT_NEAR(projection.distance, 0.5, 1.0e-9);
}

TEST(ReferenceExecution, LocalGuideKeepsInteriorTurnAndHeightAnchors)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.0},
      {0.4, 0.0, 0.0},
      {0.8, 0.0, 0.15},
      {0.8, 0.4, 0.30},
      {0.8, 0.8, 0.30},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto guide = buildReferencePathGuide(
      points, arc_lengths, Eigen::Vector3d(0.2, 0.0, 0.0), 0.2, 1.35);
  ASSERT_GE(guide.size(), 4U);
  EXPECT_TRUE(guide.front().isApprox(Eigen::Vector3d(0.2, 0.0, 0.0)));
  EXPECT_NE(
      std::find_if(
          guide.begin(), guide.end(),
          [](const Eigen::Vector3d &point) {
            return point.isApprox(Eigen::Vector3d(0.8, 0.0, 0.15));
          }),
      guide.end());
}

TEST(ReferenceExecution, CorridorGuideExcludesSyntheticReacquisitionConnector)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.3},
      {0.0, 0.5, 0.3},
      {0.0, 1.0, 0.3},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const Eigen::Vector3d actual_start(0.25, 0.2, 0.3);
  const auto initialization_guide = buildReferencePathGuide(
      points, arc_lengths, actual_start, 0.2, 1.0);
  const auto corridor_guide = buildReferencePathCorridorGuide(
      points, arc_lengths, 0.2, 1.0);

  ASSERT_GE(initialization_guide.size(), 3U);
  ASSERT_GE(corridor_guide.size(), 2U);
  EXPECT_TRUE(initialization_guide.front().isApprox(actual_start, 1.0e-9));
  EXPECT_TRUE(corridor_guide.front().isApprox(
      Eigen::Vector3d(0.0, 0.2, 0.3), 1.0e-9));
  EXPECT_TRUE(corridor_guide.back().isApprox(
      Eigen::Vector3d(0.0, 1.0, 0.3), 1.0e-9));
}

TEST(ReferenceExecution, CorridorGuideKeepsShortTerminalTail)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.3},
      {1.0, 0.0, 0.3},
      {1.02, 0.0, 0.3},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto corridor_guide = buildReferencePathCorridorGuide(
      points, arc_lengths, 1.015, arc_lengths.back());

  ASSERT_EQ(corridor_guide.size(), 2U);
  EXPECT_TRUE(corridor_guide.front().isApprox(
      Eigen::Vector3d(1.015, 0.0, 0.3), 1.0e-9));
  EXPECT_TRUE(corridor_guide.back().isApprox(points.back(), 1.0e-9));
}

TEST(ReferenceExecution, CorridorGuideBackfillsPctAnchorAtExactEndpoint)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.3},
      {1.0, 0.0, 0.3},
      {1.02, 0.0, 0.3},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto corridor_guide = buildReferencePathCorridorGuide(
      points, arc_lengths, arc_lengths.back(), arc_lengths.back());

  ASSERT_EQ(corridor_guide.size(), 2U);
  EXPECT_TRUE(corridor_guide.front().isApprox(points[1], 1.0e-9));
  EXPECT_TRUE(corridor_guide.back().isApprox(points.back(), 1.0e-9));
}

TEST(ReferenceExecution, CorridorGuideUsesOrderedPredecessorNotNearbyBranch)
{
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 0.3},
      {1.0, 0.0, 0.3},
      {1.0, 1.0, 0.3},
      {0.0, 1.0, 0.3},
      {0.0, 0.02, 0.3},
  };
  const auto arc_lengths = buildReferencePathArcLengths(points);
  const auto corridor_guide = buildReferencePathCorridorGuide(
      points, arc_lengths, arc_lengths.back(), arc_lengths.back());

  ASSERT_EQ(corridor_guide.size(), 2U);
  EXPECT_TRUE(corridor_guide.front().isApprox(points[3], 1.0e-9));
  EXPECT_FALSE(corridor_guide.front().isApprox(points.front(), 1.0e-9));
  EXPECT_TRUE(corridor_guide.back().isApprox(points.back(), 1.0e-9));
}

TEST(ReferenceExecution, CorridorGuideRejectsNonfiniteAndDegeneratePath)
{
  const std::vector<Eigen::Vector3d> valid_points = {
      {0.0, 0.0, 0.3},
      {1.0, 0.0, 0.3},
  };
  const auto valid_arc_lengths = buildReferencePathArcLengths(valid_points);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  auto nonfinite_points = valid_points;
  nonfinite_points.back().x() = nan;
  EXPECT_TRUE(buildReferencePathCorridorGuide(
      nonfinite_points, valid_arc_lengths, 0.0, 1.0).empty());

  auto nonfinite_arc_lengths = valid_arc_lengths;
  nonfinite_arc_lengths.back() = nan;
  EXPECT_TRUE(buildReferencePathCorridorGuide(
      valid_points, nonfinite_arc_lengths, 0.0, 1.0).empty());
  EXPECT_TRUE(buildReferencePathCorridorGuide(
      valid_points, valid_arc_lengths, nan, 1.0).empty());

  const std::vector<Eigen::Vector3d> degenerate_points = {
      {0.0, 0.0, 0.3},
      {0.0, 0.0, 0.3},
      {1.0, 0.0, 0.3},
  };
  EXPECT_TRUE(buildReferencePathCorridorGuide(
      degenerate_points, {0.0, 0.0, 1.0}, 0.0, 1.0).empty());
  EXPECT_TRUE(buildReferencePathCorridorGuide(
      valid_points, {0.0, 2.0}, 0.0, 1.0).empty());
}

TEST(ReferenceExecution, TargetRequiresContinuousFreeBsplineTailSupport)
{
  const auto selection = selectReferenceTargetWithFreeRunway(
      4.65, 3.65, 5.20, 0.55, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s < 3.75 || progress_s >= 4.65;
      });

  ASSERT_TRUE(selection.valid);
  EXPECT_EQ(selection.search_direction, 1);
  EXPECT_NEAR(selection.progress_s, 4.75, 1.0e-9);
}

TEST(ReferenceExecution, ForwardRunwayWinsOverCloserObstacleFrontFallback)
{
  const auto selection = selectReferenceTargetWithFreeRunway(
      4.20, 3.65, 5.20, 0.60, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s < 3.75 || progress_s >= 4.65;
      });

  ASSERT_TRUE(selection.valid);
  EXPECT_GT(selection.progress_s, 4.65);
  EXPECT_EQ(selection.search_direction, 1);
}

TEST(ReferenceExecution, SingleFreeVoxelAtPathEndFailsClosed)
{
  const auto selection = selectReferenceTargetWithFreeRunway(
      4.20, 4.00, 4.68, 0.48, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s >= 4.65;
      });

  EXPECT_FALSE(selection.valid);
}

TEST(ReferenceExecution, FreeNominalTargetKeepsItsProgress)
{
  const auto selection = selectReferenceTargetWithFreeRunway(
      0.60, 0.0, 2.0, 0.60, 0.05, 0.025, 0.10,
      [](const double) { return true; });

  ASSERT_TRUE(selection.valid);
  EXPECT_NEAR(selection.progress_s, 0.60, 1.0e-9);
  EXPECT_EQ(selection.search_direction, 0);
}

TEST(ReferenceExecution, TerminalCaptureStopsAtFirstSafePointInsideGoalGate)
{
  const auto selection = selectReferenceTerminalCaptureTarget(
      4.00, 5.00, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s >= 4.70;
      },
      [](const double progress_s) {
        return progress_s < 4.82 || progress_s >= 4.95;
      });

  ASSERT_TRUE(selection.valid);
  EXPECT_EQ(selection.search_direction, -1);
  EXPECT_NEAR(selection.progress_s, 4.70, 1.0e-9);
}

TEST(ReferenceExecution, TerminalCaptureRejectsRunwayAcrossObstacle)
{
  const auto selection = selectReferenceTerminalCaptureTarget(
      4.00, 5.00, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s >= 4.70 && progress_s <= 4.75;
      },
      [](const double progress_s) {
        return progress_s < 4.64 || progress_s > 4.68;
      });

  EXPECT_FALSE(selection.valid);
}

TEST(ReferenceExecution, TerminalCaptureRequiresFiniteOrderedContract)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto always_true = [](const double) { return true; };
  EXPECT_FALSE(selectReferenceTerminalCaptureTarget(
      nan, 1.0, 0.05, 0.025, 0.10,
      always_true, always_true).valid);
  EXPECT_FALSE(selectReferenceTerminalCaptureTarget(
      1.0, 0.5, 0.05, 0.025, 0.10,
      always_true, always_true).valid);
  EXPECT_FALSE(selectReferenceTerminalCaptureTarget(
      0.0, 1.0, 0.0, 0.025, 0.10,
      always_true, always_true).valid);
}

TEST(ReferenceExecution, TerminalCaptureMayEnterBeforeExactGoal)
{
  const auto selection = selectReferenceTerminalCaptureTarget(
      4.40, 5.00, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s >= 4.70;
      },
      [](const double) { return true; });

  ASSERT_TRUE(selection.valid);
  EXPECT_NEAR(selection.progress_s, 4.70, 1.0e-9);
  // 当前进度 4.10、局部视距 0.60：捕获点恰好已进入本次窗口，尽管
  // 精确终点 5.00 尚未进入窗口。
  EXPECT_TRUE(terminalCaptureTargetFitsLocalWindow(
      selection, 4.10, 4.70));
}

TEST(ReferenceExecution, TerminalCaptureCannotExceedNominalLookahead)
{
  const auto selection = selectReferenceTerminalCaptureTarget(
      4.40, 5.00, 0.05, 0.025, 0.10,
      [](const double progress_s) {
        return progress_s >= 4.70;
      },
      [](const double) { return true; });

  ASSERT_TRUE(selection.valid);
  EXPECT_FALSE(terminalCaptureTargetFitsLocalWindow(
      selection, 3.00, 3.60));
  EXPECT_FALSE(terminalCaptureTargetFitsLocalWindow(
      ReferenceTargetSelection{}, 3.00, 3.60));
}

TEST(ReferenceExecution, SelectedFreeTargetMayCloseTerminalSamplingGap)
{
  const ReferenceTargetSelection selected{true, 4.95, -1};
  EXPECT_TRUE(selectedReferenceTargetQualifiesAsTerminalCapture(
      selected, 4.00, 5.00,
      [](const double progress_s) {
        return std::abs(progress_s - 4.95) <= 1.0e-9;
      }));
}

TEST(ReferenceExecution, SelectedFreeTargetStillUsesStrictTerminalGates)
{
  const auto always_capture = [](const double) { return true; };
  EXPECT_FALSE(selectedReferenceTargetQualifiesAsTerminalCapture(
      ReferenceTargetSelection{true, 5.05, -1}, 4.00, 5.00,
      always_capture));
  EXPECT_FALSE(selectedReferenceTargetQualifiesAsTerminalCapture(
      ReferenceTargetSelection{true, 4.95, -1}, 4.00, 5.00,
      [](const double) { return false; }));
  EXPECT_FALSE(selectedReferenceTargetQualifiesAsTerminalCapture(
      ReferenceTargetSelection{}, 4.00, 5.00, always_capture));
}

TEST(ReferenceExecution, NonFinalTargetInsideStrictGoalGateMustBrake)
{
  EXPECT_TRUE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, 0.0502, 0.001, 0.06, 0.12));
  // final 已有独立零末速语义；该辅助门只覆盖不能诚实升级为 final 的候选。
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      true, 0.0502, 0.001, 0.06, 0.12));
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, 0.061, 0.001, 0.06, 0.12));
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, 0.0502, 0.121, 0.06, 0.12));
}

TEST(ReferenceExecution, NonFinalTerminalBrakeRejectsInvalidGeometry)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, nan, 0.0, 0.06, 0.12));
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, 0.05, 0.0, 0.0, 0.12));
  EXPECT_FALSE(nonFinalReferenceTargetRequiresTerminalBrake(
      false, -0.01, 0.0, 0.06, 0.12));
}

TEST(ReferenceExecution, PreservesReferenceGlobalTargetFromLocalOccupancy)
{
  EXPECT_FALSE(shouldAdjustGlobalTargetForLocalOccupancy(true, 1));
  EXPECT_TRUE(shouldAdjustGlobalTargetForLocalOccupancy(false, 1));
  EXPECT_FALSE(shouldAdjustGlobalTargetForLocalOccupancy(false, 0));
  EXPECT_FALSE(shouldAdjustGlobalTargetForLocalOccupancy(false, -1));
}

TEST(ReferenceExecution, AllowsShortFinalSegmentOnlyWithOrderedReferenceGuide)
{
  EXPECT_TRUE(shortReferenceReplanAllowed(0.144, true, 2));
  EXPECT_FALSE(shortReferenceReplanAllowed(0.144, false, 2));
  EXPECT_FALSE(shortReferenceReplanAllowed(0.144, true, 1));
  EXPECT_FALSE(shortReferenceReplanAllowed(0.0, true, 2));
  EXPECT_FALSE(shortReferenceReplanAllowed(
      std::numeric_limits<double>::quiet_NaN(), true, 2));
}

TEST(ReferenceExecution, PreservesCommunityShortDistanceBoundary)
{
  EXPECT_TRUE(reboundPlanRejectedBeforeInitialization(0.199, false, 0));
  EXPECT_FALSE(reboundPlanRejectedBeforeInitialization(0.2, false, 0));
  EXPECT_FALSE(reboundPlanRejectedBeforeInitialization(0.144, true, 2));
  EXPECT_FALSE(reboundPlanRejectedBeforeInitialization(0.081, true, 2));
  EXPECT_FALSE(reboundPlanRejectedBeforeInitialization(0.079, true, 2));
  EXPECT_TRUE(reboundPlanRejectedBeforeInitialization(0.144, true, 1));
  EXPECT_TRUE(reboundPlanRejectedBeforeInitialization(0.0, true, 2));
  EXPECT_TRUE(reboundPlanRejectedBeforeInitialization(
      std::numeric_limits<double>::infinity(), true, 2));
}

TEST(ReferenceExecution, FinalHoldRequiresTrueGoalAndStableOdometry)
{
  // 最后一项是机体系 |wz|，不是包含站立 roll/pitch 微摆的三轴范数。
  EXPECT_TRUE(referenceGoalHoldReady(
      true, true, true,
      0.06, 0.04, 0.18, 0.01, 0.02, 0.03,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
  EXPECT_FALSE(referenceGoalHoldReady(
      true, true, true,
      0.061, 0.04, 0.10, 0.01, 0.02, 0.03,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
  EXPECT_FALSE(referenceGoalHoldReady(
      true, true, true,
      0.05, 0.04, 0.10, 0.06, 0.02, 0.03,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
  EXPECT_FALSE(referenceGoalHoldReady(
      true, false, true,
      0.05, 0.04, 0.10, 0.01, 0.02, 0.03,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
  EXPECT_FALSE(referenceGoalHoldReady(
      true, true, true,
      0.05, 0.04, 0.181, 0.01, 0.02, 0.03,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
  EXPECT_FALSE(referenceGoalHoldReady(
      true, true, true,
      0.05, 0.04, 0.10, 0.01, 0.02, 0.101,
      0.06, 0.12, 0.18, 0.05, 0.05, 0.10));
}

TEST(ReferenceExecution, FinalHoldRequiresContinuousDwellAfterRebound)
{
  ReferenceGoalHoldDwellState state;
  constexpr std::int64_t path_stamp_ns = 1000000000LL;
  constexpr double dwell_sec = 0.50;
  constexpr double maximum_gap_sec = 0.25;

  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 2000000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 2200000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_NEAR(state.stable_duration_sec, 0.20, 1.0e-12);

  // 四足底盘在零命令后回弹一次，已经累计的稳定时间必须全部清零。
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, false, 2300000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_DOUBLE_EQ(state.stable_duration_sec, 0.0);
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 2400000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 2600000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 2800000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
  EXPECT_TRUE(updateReferenceGoalHoldDwell(
      state, true, 2900000000LL, path_stamp_ns,
      dwell_sec, maximum_gap_sec));
}

TEST(ReferenceExecution, FinalHoldDwellResetsOnPathChangeAndInputGap)
{
  ReferenceGoalHoldDwellState state;
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 1000000000LL, 100LL, 0.50, 0.25));
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 1200000000LL, 100LL, 0.50, 0.25));

  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 1300000000LL, 101LL, 0.50, 0.25));
  EXPECT_DOUBLE_EQ(state.stable_duration_sec, 0.0);
  EXPECT_FALSE(updateReferenceGoalHoldDwell(
      state, true, 1700000000LL, 101LL, 0.50, 0.25));
  EXPECT_DOUBLE_EQ(state.stable_duration_sec, 0.0);
}

TEST(ReferenceExecution, FinalHoldOnlyConsumesExactControllerIdentity)
{
  const FinalHoldTrajectoryIdentity expected{100, 200, 200, 27};
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, expected,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kConfirmGoal);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, expected,
          true, true, true, false,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kReplan);

  auto mismatched = expected;
  mismatched.trajectory_id = 28;
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, mismatched,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kIgnore);
  mismatched = expected;
  mismatched.reference_path_stamp_ns = 101;
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, mismatched,
          true, true, true, false,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kIgnore);
  mismatched = expected;
  mismatched.bspline_header_stamp_ns = 201;
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, mismatched,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kIgnore);
  mismatched = expected;
  mismatched.start_stamp_ns = 201;
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, expected, mismatched,
          true, true, true, false,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kIgnore);
}

TEST(ReferenceExecution, MovingFinalConsumesExactGoalButNotTimeout)
{
  const FinalHoldTrajectoryIdentity identity{100, 200, 200, 27};
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          false, true, identity, identity,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kConfirmGoal);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          false, true, identity, identity,
          true, true, true, false,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kIgnore);
}

TEST(ReferenceExecution, RejectedFinalCandidatePreservesAcceptedGoalIdentity)
{
  const FinalHoldTrajectoryIdentity accepted_final{100, 200, 200, 27};
  const FinalHoldTrajectoryIdentity rejected_candidate{100, 300, 300, 28};
  FinalHoldTrajectoryIdentity cached;
  bool have_cached = false;

  updateControllerAcceptedFinalIdentity(
      true, accepted_final,
      true, true, true, false, cached, have_cached);
  ASSERT_TRUE(have_cached);
  EXPECT_TRUE(sameFinalHoldIdentity(cached, accepted_final));

  // controller 拒绝 candidate=28 时，status.identity 仍是已执行的 27。
  updateControllerAcceptedFinalIdentity(
      false, accepted_final,
      true, true, true, false, cached, have_cached);
  ASSERT_TRUE(have_cached);
  EXPECT_TRUE(sameFinalHoldIdentity(cached, accepted_final));
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          false, true, cached, accepted_final,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kConfirmGoal);

  updateControllerAcceptedFinalIdentity(
      true, rejected_candidate,
      true, true, false, false, cached, have_cached);
  EXPECT_FALSE(have_cached);
}

TEST(ReferenceExecution, DelayedAcceptedFinalUsesBoundedPublishedHistory)
{
  const FinalHoldTrajectoryIdentity accepted_final{100, 200, 200, 27};
  const FinalHoldTrajectoryIdentity newer_candidate{100, 300, 300, 28};
  std::vector<FinalHoldTrajectoryIdentity> history;
  rememberPublishedFinalIdentity(history, accepted_final, 2);
  rememberPublishedFinalIdentity(history, newer_candidate, 2);

  ASSERT_TRUE(wasFinalIdentityPublished(history, accepted_final));
  FinalHoldTrajectoryIdentity cached;
  bool have_cached = false;
  updateControllerAcceptedFinalIdentity(
      wasFinalIdentityPublished(history, accepted_final), accepted_final,
      true, true, true, false, cached, have_cached);
  ASSERT_TRUE(have_cached);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          false, true, cached, accepted_final,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kConfirmGoal);

  // 新 Path 激活会清空历史，旧代际终态不能确认新目标。
  history.clear();
  EXPECT_FALSE(wasFinalIdentityPublished(history, accepted_final));
}

TEST(ReferenceExecution, PublishedFinalIdentityHistoryIsDeduplicatedAndBounded)
{
  std::vector<FinalHoldTrajectoryIdentity> history;
  const FinalHoldTrajectoryIdentity first{100, 200, 200, 27};
  const FinalHoldTrajectoryIdentity second{100, 300, 300, 28};
  const FinalHoldTrajectoryIdentity third{100, 400, 400, 29};
  rememberPublishedFinalIdentity(history, first, 2);
  rememberPublishedFinalIdentity(history, first, 2);
  ASSERT_EQ(history.size(), 1U);
  rememberPublishedFinalIdentity(history, second, 2);
  rememberPublishedFinalIdentity(history, third, 2);
  ASSERT_EQ(history.size(), 2U);
  EXPECT_FALSE(wasFinalIdentityPublished(history, first));
  EXPECT_TRUE(wasFinalIdentityPublished(history, second));
  EXPECT_TRUE(wasFinalIdentityPublished(history, third));
}

TEST(ReferenceExecution, FrozenFinalHoldWaitsAcrossResumeBeforeControllerStatus)
{
  const FinalHoldLifecycleState pending{true, false, 0};
  EXPECT_EQ(
      decideFinalHoldStairResumeAction(pending),
      FinalHoldStairResumeAction::kWaitForController);
  EXPECT_TRUE(finalHoldBlocksStationaryPublication(pending));

  const FinalHoldLifecycleState unchanged = transitionFinalHoldLifecycle(
      pending, FinalHoldControllerAction::kIgnore, 27);
  EXPECT_TRUE(unchanged.pending);
  EXPECT_FALSE(unchanged.recovery_required);
  EXPECT_EQ(unchanged.recovery_after_trajectory_id, 0);
}

TEST(ReferenceExecution, FinalHoldTimeoutClearsPendingAndContinuesRecovery)
{
  const FinalHoldLifecycleState pending{true, false, 0};
  const FinalHoldLifecycleState recovery = transitionFinalHoldLifecycle(
      pending, FinalHoldControllerAction::kReplan, 27);

  EXPECT_FALSE(recovery.pending);
  EXPECT_TRUE(recovery.recovery_required);
  EXPECT_EQ(recovery.recovery_after_trajectory_id, 27);
  EXPECT_EQ(
      decideFinalHoldStairResumeAction(recovery),
      FinalHoldStairResumeAction::kContinueTimeoutRecovery);
  EXPECT_TRUE(finalHoldBlocksStationaryPublication(recovery));
}

TEST(ReferenceExecution, FinalHoldTimeoutRecoveryRequiresHigherTrajectoryId)
{
  const FinalHoldLifecycleState recovery{false, true, 27};
  EXPECT_FALSE(finalHoldRecoveryTrajectoryIdAllowed(recovery, 26));
  EXPECT_FALSE(finalHoldRecoveryTrajectoryIdAllowed(recovery, 27));
  EXPECT_TRUE(finalHoldRecoveryTrajectoryIdAllowed(recovery, 28));
  EXPECT_FALSE(finalHoldRecoveryTrajectoryIdAllowed(
      FinalHoldLifecycleState{false, true, 0}, 28));
}

TEST(ReferenceExecution, FinalHoldGoalAckClearsLifecycle)
{
  const FinalHoldLifecycleState pending{true, false, 0};
  const FinalHoldLifecycleState completed = transitionFinalHoldLifecycle(
      pending, FinalHoldControllerAction::kConfirmGoal, 27);
  EXPECT_FALSE(completed.pending);
  EXPECT_FALSE(completed.recovery_required);
  EXPECT_EQ(completed.recovery_after_trajectory_id, 0);
  EXPECT_EQ(
      decideFinalHoldStairResumeAction(completed),
      FinalHoldStairResumeAction::kResumeNormally);
  EXPECT_FALSE(finalHoldBlocksStationaryPublication(completed));
}

TEST(ReferenceExecution, FinalHoldRejectsUnboundOrUnsafeControllerStatus)
{
  const FinalHoldTrajectoryIdentity identity{100, 200, 200, 27};
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          false, true, identity, identity,
          true, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kConfirmGoal);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, false, identity, identity,
          true, true, true, false,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kIgnore);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, identity, identity,
          false, true, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kIgnore);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, identity, identity,
          true, false, true, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kIgnore);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, identity, identity,
          true, true, false, false,
          FinalHoldControllerState::kGoalReached),
      FinalHoldControllerAction::kIgnore);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, identity, identity,
          true, true, true, true,
          FinalHoldControllerState::kTrajectoryTimeout),
      FinalHoldControllerAction::kIgnore);
  EXPECT_EQ(
      decideFinalHoldControllerAction(
          true, true, identity, identity,
          true, true, true, false,
          FinalHoldControllerState::kOther),
      FinalHoldControllerAction::kIgnore);
}

TEST(ReferenceExecution, FailedReferenceRetryIsRateLimited)
{
  EXPECT_FALSE(referenceRetryReady(true, 10.49, 10.50));
  EXPECT_TRUE(referenceRetryReady(true, 10.50, 10.50));
  EXPECT_TRUE(referenceRetryReady(false, 1.0, 10.0));
}

TEST(ReferenceExecution, HoldsUntilOdometryMovesReplanDistance)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          false, true, false, false, false, 1.5, 0.29, 0.30, 0.22),
      ReferenceExecutionAction::kHold);
  EXPECT_EQ(
      decideReferenceExecutionAction(
          false, true, false, false, false, 1.5, 0.31, 0.30, 0.22),
      ReferenceExecutionAction::kReplan);
}

TEST(ReferenceExecution, ReplansExpiredNonFinalSegment)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, true, false, false, true, 1.2, 0.1, 0.30, 0.22),
      ReferenceExecutionAction::kReplan);
}

TEST(ReferenceExecution, ReplansExpiredFinalSegmentWhenRobotStillFar)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, true, true, false, true, 1.31, 0.4, 0.30, 0.22),
      ReferenceExecutionAction::kReplan);
}

TEST(ReferenceExecution, HoldsExpiredFinalSegmentDuringTerminalYawAlignment)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, true, true, true, false, 0.10, 0.4, 0.30, 0.22),
      ReferenceExecutionAction::kHold);
}

TEST(ReferenceExecution, HoldsExpiredFinalSegmentDuringBoundedConvergenceGrace)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, true, true, false, true, 0.10, 0.4, 0.30, 0.22),
      ReferenceExecutionAction::kHold);
}

TEST(ReferenceExecution, ReplansExpiredFinalSegmentAfterControllerHardTimeout)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, true, true, false, false, 0.10, 0.4, 0.30, 0.22),
      ReferenceExecutionAction::kReplan);
  EXPECT_EQ(
      decideReferenceExecutionAction(
          true, false, true, false, false, 0.10, 0.4, 0.30, 0.22),
      ReferenceExecutionAction::kFinish);
}

TEST(ReferenceExecution, HoldsNearGoalForControllerConvergence)
{
  EXPECT_EQ(
      decideReferenceExecutionAction(
          false, true, true, false, false, 0.10, 0.5, 0.30, 0.22),
      ReferenceExecutionAction::kHold);
}

TEST(ReferenceExecution, RetriesStillActiveReferenceAfterEmergencyStop)
{
  EXPECT_EQ(
      decideReferenceEmergencyRecovery(true, true, true),
      ReferenceEmergencyRecoveryAction::kRetryActiveReference);
}

TEST(ReferenceExecution, WaitsWhenReferenceWasCancelledOrTargetWasCleared)
{
  EXPECT_EQ(
      decideReferenceEmergencyRecovery(true, false, true),
      ReferenceEmergencyRecoveryAction::kWaitForNewTarget);
  EXPECT_EQ(
      decideReferenceEmergencyRecovery(true, true, false),
      ReferenceEmergencyRecoveryAction::kWaitForNewTarget);
  EXPECT_EQ(
      decideReferenceEmergencyRecovery(false, true, true),
      ReferenceEmergencyRecoveryAction::kWaitForNewTarget);
}
