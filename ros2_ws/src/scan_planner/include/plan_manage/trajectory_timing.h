#pragma once

#include <algorithm>
#include <cmath>

#include <bspline_opt/uniform_bspline.h>

namespace scan_planner
{

// 当前在线规划只发布三次三维 B-spline。先验证完整结构再读取 duration，
// 避免畸形 knot 数组令 getTimeSum() 越界，也避免非严格递增 knot 在求导时
// 产生除零和 NaN。
inline bool trajectoryTimingStateIsValid(UniformBspline &trajectory)
{
  const Eigen::MatrixXd control_points = trajectory.getControlPoint();
  const Eigen::VectorXd knots = trajectory.getKnot();
  const double interval = trajectory.getInterval();
  if (control_points.rows() != 3 || control_points.cols() < 4 ||
      knots.size() != control_points.cols() + 4 ||
      !control_points.allFinite() || !knots.allFinite() ||
      !std::isfinite(interval) || interval <= 0.0)
    return false;
  for (Eigen::Index index = 1; index < knots.size(); ++index)
    if (!(knots(index) > knots(index - 1)))
      return false;
  const double duration = trajectory.getTimeSum();
  return std::isfinite(duration) && duration > 0.0;
}

// 连续控制点门必须不宽于最终采样动态门，否则调大相对容差后，速度或
// 加速度窄峰可能落在离散采样之间而被误放行。
inline bool dynamicFeasibilityTolerancesCompatible(
    const double max_velocity,
    const double max_acceleration,
    const double feasibility_tolerance,
    const double velocity_tolerance,
    const double acceleration_tolerance)
{
  if (!std::isfinite(max_velocity) || max_velocity <= 0.0 ||
      !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
      !std::isfinite(feasibility_tolerance) ||
      feasibility_tolerance < 0.0 ||
      !std::isfinite(velocity_tolerance) || velocity_tolerance < 0.0 ||
      !std::isfinite(acceleration_tolerance) ||
      acceleration_tolerance < 0.0)
    return false;
  const double velocity_margin =
      max_velocity * feasibility_tolerance + 1.0e-4;
  const double acceleration_margin =
      max_acceleration * feasibility_tolerance + 1.0e-4;
  return std::isfinite(velocity_margin) &&
         std::isfinite(acceleration_margin) &&
         velocity_margin <= velocity_tolerance &&
         acceleration_margin <= acceleration_tolerance;
}

// rebound/refine 会再次移动控制点，因此它成功返回后仍可能破坏速度或
// 加速度约束。统一缩放全部 knot 可以保持空间曲线和避障净距不变，同时
// 分别按 1/r 与 1/r² 降低速度、加速度。
inline bool rescaleTrajectoryToPhysicalLimits(
    UniformBspline &trajectory,
    const double max_velocity,
    const double max_acceleration,
    const double feasibility_tolerance,
    const int maximum_attempts = 3)
{
  if (!std::isfinite(max_velocity) || max_velocity <= 0.0 ||
      !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
      !std::isfinite(feasibility_tolerance) ||
      feasibility_tolerance < 0.0 || maximum_attempts < 0)
    return false;

  if (!trajectoryTimingStateIsValid(trajectory))
    return false;

  trajectory.setPhysicalLimits(
      max_velocity, max_acceleration, feasibility_tolerance);
  double required_ratio = 1.0;
  for (int attempt = 0; attempt <= maximum_attempts; ++attempt)
  {
    if (trajectory.checkFeasibility(required_ratio, false))
      return true;
    if (attempt == maximum_attempts || !std::isfinite(required_ratio) ||
        required_ratio <= 0.0)
      return false;
    const double applied_ratio =
        std::max(1.01, 1.01 * required_ratio);
    if (!std::isfinite(applied_ratio) || applied_ratio <= 0.0)
      return false;

    // 在原对象上提交缩放前先验证全部乘积，失败时保持原轨迹不变。
    const Eigen::VectorXd knots = trajectory.getKnot();
    const double interval = trajectory.getInterval();
    if (!std::isfinite(interval * applied_ratio))
      return false;
    for (Eigen::Index index = 0; index < knots.size(); ++index)
      if (!std::isfinite(knots(index) * applied_ratio))
        return false;

    trajectory.scaleTimeUniformly(applied_ratio);
    if (!trajectoryTimingStateIsValid(trajectory))
      return false;
  }
  return false;
}

}  // namespace scan_planner
