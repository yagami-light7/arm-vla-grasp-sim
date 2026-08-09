#include <array>

#include <gtest/gtest.h>

#include "bspline_opt/rebound_failure_reason.h"

namespace scan_planner
{
namespace
{

TEST(ReboundFailureStateTest, ResetAndSuccessfulCompletionClearReason)
{
  ReboundFailureState state;
  EXPECT_EQ(state.reason(), ReboundFailureReason::None);

  state.observe(ReboundFailureReason::TerminalPointOccupied);
  ASSERT_EQ(
      state.reason(), ReboundFailureReason::TerminalPointOccupied);
  state.reset();
  EXPECT_EQ(state.reason(), ReboundFailureReason::None);

  state.observe(ReboundFailureReason::StartRegionOccupied);
  state.complete(true);
  EXPECT_EQ(state.reason(), ReboundFailureReason::None);
}

TEST(ReboundFailureStateTest, EveryTypedReasonCanBeRecordedAfterReset)
{
  constexpr std::array<ReboundFailureReason, 4> reasons{
      ReboundFailureReason::TerminalPointOccupied,
      ReboundFailureReason::CollisionSegmentAstarDegenerate,
      ReboundFailureReason::StartRegionOccupied,
      ReboundFailureReason::OptimizationFailed};
  ReboundFailureState state;
  for (const ReboundFailureReason reason : reasons)
  {
    state.reset();
    state.observe(reason);
    EXPECT_EQ(state.reason(), reason);
  }
}

TEST(
    ReboundFailureStateTest,
    DegenerateV9CollisionSegmentSurvivesFinalGenericFailure)
{
  EXPECT_EQ(
      classifyCollisionSegmentAstarPath(0U),
      ReboundFailureReason::CollisionSegmentAstarDegenerate);
  EXPECT_EQ(
      classifyCollisionSegmentAstarPath(1U),
      ReboundFailureReason::CollisionSegmentAstarDegenerate);
  EXPECT_EQ(
      classifyCollisionSegmentAstarPath(2U),
      ReboundFailureReason::None);

  ReboundFailureState state;
  state.observe(classifyCollisionSegmentAstarPath(1U));
  state.complete(false);
  EXPECT_EQ(
      state.reason(),
      ReboundFailureReason::CollisionSegmentAstarDegenerate);
}

TEST(ReboundFailureStateTest, UnclassifiedFailureBecomesOptimizationFailed)
{
  ReboundFailureState state;
  state.complete(false);
  EXPECT_EQ(state.reason(), ReboundFailureReason::OptimizationFailed);
}

}  // namespace
}  // namespace scan_planner
