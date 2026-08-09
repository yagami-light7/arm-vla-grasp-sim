#ifndef BSPLINE_OPT__REBOUND_FAILURE_REASON_H_
#define BSPLINE_OPT__REBOUND_FAILURE_REASON_H_

#include <cstddef>
#include <cstdint>

namespace scan_planner
{

enum class ReboundFailureReason : std::uint8_t
{
  None = 0,
  TerminalPointOccupied,
  CollisionSegmentAstarDegenerate,
  StartRegionOccupied,
  OptimizationFailed,
};

constexpr ReboundFailureReason classifyCollisionSegmentAstarPath(
    const std::size_t path_size) noexcept
{
  return path_size < 2U
             ? ReboundFailureReason::CollisionSegmentAstarDegenerate
             : ReboundFailureReason::None;
}

// 单次 rebound 尝试可能依次发现多个问题。终点占据与退化 A* 段比
// 通用求解失败更具体，后续通用失败不得覆盖先前已经取得的因果证据。
constexpr int reboundFailureReasonPriority(
    const ReboundFailureReason reason) noexcept
{
  switch (reason)
  {
    case ReboundFailureReason::TerminalPointOccupied:
      return 4;
    case ReboundFailureReason::CollisionSegmentAstarDegenerate:
      return 3;
    case ReboundFailureReason::StartRegionOccupied:
      return 2;
    case ReboundFailureReason::OptimizationFailed:
      return 1;
    case ReboundFailureReason::None:
    default:
      return 0;
  }
}

class ReboundFailureState
{
public:
  constexpr void reset() noexcept
  {
    reason_ = ReboundFailureReason::None;
  }

  constexpr void observe(const ReboundFailureReason reason) noexcept
  {
    if (reboundFailureReasonPriority(reason) >
        reboundFailureReasonPriority(reason_))
      reason_ = reason;
  }

  constexpr void complete(const bool success) noexcept
  {
    if (success)
      reset();
    else
      observe(ReboundFailureReason::OptimizationFailed);
  }

  [[nodiscard]] constexpr ReboundFailureReason reason() const noexcept
  {
    return reason_;
  }

private:
  ReboundFailureReason reason_{ReboundFailureReason::None};
};

}  // namespace scan_planner

#endif  // BSPLINE_OPT__REBOUND_FAILURE_REASON_H_
