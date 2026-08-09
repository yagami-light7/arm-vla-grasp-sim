#ifndef SCAN_PLANNER__TRAJECTORY_DIAGNOSTICS_H_
#define SCAN_PLANNER__TRAJECTORY_DIAGNOSTICS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <bspline_opt/uniform_bspline.h>

namespace scan_planner
{

struct BoundedGeometrySamples
{
  bool valid{false};
  std::uint32_t total_count{0};
  bool truncated{false};
  std::vector<Eigen::Vector3d> points;
};

inline std::uint32_t boundedSampleCount(const std::size_t count)
{
  return static_cast<std::uint32_t>(std::min<std::size_t>(
      count, std::numeric_limits<std::uint32_t>::max()));
}

// 对完整轨迹时间域均匀抽样。数组有硬上限，但首尾和中间段均被覆盖，
// 不会因只保留前 N 点而漏掉后半段绕障。
inline BoundedGeometrySamples sampleTrajectoryGeometry(
    UniformBspline trajectory,
    const std::size_t maximum_samples,
    const double dense_interval = 0.01)
{
  BoundedGeometrySamples result;
  const double duration = trajectory.getTimeSum();
  if (!std::isfinite(duration) || duration <= 0.0 ||
      !std::isfinite(dense_interval) || dense_interval <= 0.0 ||
      maximum_samples < 2 || maximum_samples > 64)
    return result;

  const double dense_steps = std::ceil(duration / dense_interval);
  if (!std::isfinite(dense_steps) || dense_steps < 1.0 ||
      dense_steps >=
          static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
    return result;
  const std::uint32_t total_count =
      static_cast<std::uint32_t>(dense_steps) + 1U;
  const std::size_t output_count = std::min<std::size_t>(
      total_count, maximum_samples);
  result.points.reserve(output_count);
  for (std::size_t index = 0; index < output_count; ++index)
  {
    const double ratio = output_count == 1U
                             ? 0.0
                             : static_cast<double>(index) /
                                   static_cast<double>(output_count - 1U);
    const Eigen::Vector3d point =
        trajectory.evaluateDeBoorT(duration * ratio);
    if (!point.allFinite())
      return BoundedGeometrySamples{};
    result.points.push_back(point);
  }
  result.valid = true;
  result.total_count = total_count;
  result.truncated = total_count > output_count;
  return result;
}

// B-spline 曲线始终位于控制点凸包内；其导数同样如此。因此导数
// 控制点最大范数是完整连续时间域上的保守速度上界，而不是离散采样估计。
inline double trajectoryMaximumVelocityUpperBound(
    UniformBspline trajectory)
{
  const Eigen::MatrixXd velocity_control_points =
      trajectory.getDerivative().getControlPoint();
  if (velocity_control_points.rows() != 3 ||
      velocity_control_points.cols() == 0 ||
      !velocity_control_points.allFinite())
    return std::numeric_limits<double>::quiet_NaN();

  double upper_bound = 0.0;
  for (Eigen::Index column = 0;
       column < velocity_control_points.cols(); ++column)
    upper_bound = std::max(
        upper_bound, velocity_control_points.col(column).norm());
  return std::isfinite(upper_bound)
             ? upper_bound
             : std::numeric_limits<double>::quiet_NaN();
}

// 保持 ordered reference 的顺序，并在超限时均匀选取含首尾的锚点。
inline BoundedGeometrySamples sampleOrderedReferenceGeometry(
    const std::vector<Eigen::Vector3d> &guide,
    const std::size_t maximum_samples)
{
  BoundedGeometrySamples result;
  if (guide.empty() || maximum_samples < 2 || maximum_samples > 64)
    return result;
  for (const Eigen::Vector3d &point : guide)
    if (!point.allFinite())
      return result;

  const std::size_t output_count =
      std::min<std::size_t>(guide.size(), maximum_samples);
  result.points.reserve(output_count);
  for (std::size_t index = 0; index < output_count; ++index)
  {
    const double ratio = output_count == 1U
                             ? 0.0
                             : static_cast<double>(index) /
                                   static_cast<double>(output_count - 1U);
    const std::size_t source_index = static_cast<std::size_t>(std::llround(
        ratio * static_cast<double>(guide.size() - 1U)));
    result.points.push_back(guide[source_index]);
  }
  result.valid = true;
  result.total_count = boundedSampleCount(guide.size());
  result.truncated = guide.size() > output_count;
  return result;
}

inline bool trajectoryIsStationary(
    UniformBspline trajectory,
    const double tolerance = 1.0e-9)
{
  if (!std::isfinite(tolerance) || tolerance < 0.0)
    return false;
  const Eigen::MatrixXd control_points = trajectory.getControlPoint();
  if (control_points.rows() != 3 || control_points.cols() == 0 ||
      !control_points.allFinite())
    return false;
  for (Eigen::Index column = 1; column < control_points.cols(); ++column)
    if ((control_points.col(column) - control_points.col(0)).norm() >
        tolerance)
      return false;
  return true;
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__TRAJECTORY_DIAGNOSTICS_H_
