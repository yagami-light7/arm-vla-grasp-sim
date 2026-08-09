#ifndef SCAN_PLANNER__LOCAL_PLAN_FAILURE_REASON_H_
#define SCAN_PLANNER__LOCAL_PLAN_FAILURE_REASON_H_

#include <cstdint>

#include <bspline_opt/rebound_failure_reason.h>

namespace scan_planner
{

enum class LocalPlanFailureReason : std::uint8_t
{
  None = 0,
  TerminalPointOccupied,
  CollisionSegmentAstarDegenerate,
  StartRegionOccupied,
  OptimizationFailed,
  DynamicFeasibilityFailed,
  ReferenceCorridorRejected,
};

constexpr LocalPlanFailureReason localPlanFailureReasonFromRebound(
    const ReboundFailureReason reason) noexcept
{
  switch (reason)
  {
    case ReboundFailureReason::TerminalPointOccupied:
      return LocalPlanFailureReason::TerminalPointOccupied;
    case ReboundFailureReason::CollisionSegmentAstarDegenerate:
      return LocalPlanFailureReason::CollisionSegmentAstarDegenerate;
    case ReboundFailureReason::StartRegionOccupied:
      return LocalPlanFailureReason::StartRegionOccupied;
    case ReboundFailureReason::OptimizationFailed:
      return LocalPlanFailureReason::OptimizationFailed;
    case ReboundFailureReason::None:
    default:
      return LocalPlanFailureReason::None;
  }
}

class LocalPlanFailureState
{
public:
  constexpr void beginAttempt() noexcept
  {
    // 任意未被更具体分型覆盖的 false 都保守归入优化失败。
    reason_ = LocalPlanFailureReason::OptimizationFailed;
  }

  constexpr void set(const LocalPlanFailureReason reason) noexcept
  {
    reason_ = reason;
  }

  constexpr void setFromReboundFailure(
      const ReboundFailureReason reason) noexcept
  {
    const LocalPlanFailureReason promoted =
        localPlanFailureReasonFromRebound(reason);
    reason_ = promoted == LocalPlanFailureReason::None
                  ? LocalPlanFailureReason::OptimizationFailed
                  : promoted;
  }

  constexpr void completeSuccess() noexcept
  {
    reason_ = LocalPlanFailureReason::None;
  }

  [[nodiscard]] constexpr LocalPlanFailureReason reason() const noexcept
  {
    return reason_;
  }

private:
  LocalPlanFailureReason reason_{LocalPlanFailureReason::None};
};

}  // namespace scan_planner

#endif  // SCAN_PLANNER__LOCAL_PLAN_FAILURE_REASON_H_
