#include <array>
#include <utility>

#include <gtest/gtest.h>

#include "plan_manage/local_plan_failure_reason.h"

namespace scan_planner
{
namespace
{

TEST(LocalPlanFailureStateTest, PromotesEveryReboundFailureReason)
{
  constexpr std::array<
      std::pair<ReboundFailureReason, LocalPlanFailureReason>, 4>
      mappings{{
          {ReboundFailureReason::TerminalPointOccupied,
           LocalPlanFailureReason::TerminalPointOccupied},
          {ReboundFailureReason::CollisionSegmentAstarDegenerate,
           LocalPlanFailureReason::CollisionSegmentAstarDegenerate},
          {ReboundFailureReason::StartRegionOccupied,
           LocalPlanFailureReason::StartRegionOccupied},
          {ReboundFailureReason::OptimizationFailed,
           LocalPlanFailureReason::OptimizationFailed},
      }};

  LocalPlanFailureState state;
  for (const auto &mapping : mappings)
  {
    state.beginAttempt();
    state.setFromReboundFailure(mapping.first);
    EXPECT_EQ(state.reason(), mapping.second);
  }
}

TEST(LocalPlanFailureStateTest, AttemptResetAndSuccessHaveExplicitDefaults)
{
  LocalPlanFailureState state;
  EXPECT_EQ(state.reason(), LocalPlanFailureReason::None);

  state.beginAttempt();
  EXPECT_EQ(state.reason(), LocalPlanFailureReason::OptimizationFailed);

  state.set(LocalPlanFailureReason::DynamicFeasibilityFailed);
  EXPECT_EQ(
      state.reason(), LocalPlanFailureReason::DynamicFeasibilityFailed);
  state.beginAttempt();
  EXPECT_EQ(state.reason(), LocalPlanFailureReason::OptimizationFailed);

  state.set(LocalPlanFailureReason::ReferenceCorridorRejected);
  state.completeSuccess();
  EXPECT_EQ(state.reason(), LocalPlanFailureReason::None);
}

TEST(LocalPlanFailureStateTest, MissingLowerLevelReasonFailsClosed)
{
  LocalPlanFailureState state;
  state.beginAttempt();
  state.setFromReboundFailure(ReboundFailureReason::None);
  EXPECT_EQ(state.reason(), LocalPlanFailureReason::OptimizationFailed);
}

}  // namespace
}  // namespace scan_planner
