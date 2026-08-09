#ifndef SCAN_PLANNER__REFERENCE_SPLINE_BOUNDARY_H_
#define SCAN_PLANNER__REFERENCE_SPLINE_BOUNDARY_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

namespace scan_planner
{

// community 参数化通过过定最小二乘同时拟合路径点与四个端点导数，
// 因此起点位置和导数并不严格成立，短转弯时会产生厘米级先前进再回摆。
// 对均匀三次 B-spline 的前三个控制点使用解析式恢复精确起点边界；
// 后续碰撞、动态可行性和有序走廊门仍会验证修改后的完整空间曲线。
inline bool enforceCubicBsplineStartBoundary(
    Eigen::MatrixXd &control_points,
    const double interval,
    const Eigen::Vector3d &position,
    const Eigen::Vector3d &velocity,
    const Eigen::Vector3d &acceleration)
{
  if (control_points.rows() != 3 || control_points.cols() < 3 ||
      !control_points.allFinite() || !std::isfinite(interval) ||
      interval <= 0.0 || !position.allFinite() || !velocity.allFinite() ||
      !acceleration.allFinite())
    return false;

  const double interval_squared = interval * interval;
  if (!std::isfinite(interval_squared))
    return false;
  control_points.col(0) =
      position - interval * velocity +
      (interval_squared / 3.0) * acceleration;
  control_points.col(1) =
      position - (interval_squared / 6.0) * acceleration;
  control_points.col(2) =
      position + interval * velocity +
      (interval_squared / 3.0) * acceleration;
  return control_points.leftCols(3).allFinite();
}

// 三次 B-spline 的一阶导数由相邻控制点差的非负基函数组合得到。
// 因此只要控制折线在本轮 start->target 主推进方向上的投影不下降，连续曲线
// 就不会在该方向上产生最小二乘过冲后的短暂倒退。前三点属于精确起点边界，
// 必须原样保留真实的微小落脚漂移；其幅值继续由后续 reverse speed/distance
// 安全门限制。其余点只沿推进方向做最小平移，横向/高度形状仍交给后续
// 碰撞和有序走廊门复核。
inline bool enforceCubicBsplineForwardControlPolygon(
    Eigen::MatrixXd &control_points,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &target)
{
  if (control_points.rows() != 3 || control_points.cols() < 3 ||
      !control_points.allFinite() || !start.allFinite() ||
      !target.allFinite())
    return false;

  Eigen::Vector3d progress_direction = target - start;
  const double planar_distance = progress_direction.head<2>().norm();
  if (planar_distance > 1.0e-9)
    progress_direction.z() = 0.0;
  const double progress_distance = progress_direction.norm();
  if (!std::isfinite(progress_distance) || progress_distance <= 1.0e-9)
    return false;
  progress_direction /= progress_distance;

  constexpr double kProjectionTolerance = 1.0e-12;
  double previous_projection =
      (control_points.col(2) - start).dot(progress_direction);
  if (!std::isfinite(previous_projection))
    return false;
  for (Eigen::Index index = 3; index < control_points.cols(); ++index)
  {
    double projection =
        (control_points.col(index) - start).dot(progress_direction);
    if (!std::isfinite(projection))
      return false;
    if (projection + kProjectionTolerance < previous_projection)
    {
      control_points.col(index) +=
          (previous_projection - projection) * progress_direction;
      projection = previous_projection;
    }
    previous_projection = std::max(previous_projection, projection);
  }
  return control_points.allFinite();
}

// 在前三个控制点解析固定后，只对剩余控制点重新做最小二乘。与“先拟合、
// 再覆盖前三点”相比，这会让后续控制点共同吸收硬边界条件，避免在第一段
// 制造新的折线捷径，同时保留终点速度/加速度和全部参考点拟合项。
inline bool parameterizeCubicBsplineWithExactStartBoundary(
    const double interval,
    const std::vector<Eigen::Vector3d> &point_set,
    const std::vector<Eigen::Vector3d> &start_end_derivatives,
    Eigen::MatrixXd &control_points)
{
  if (!std::isfinite(interval) || interval <= 0.0 ||
      point_set.size() <= 3 || start_end_derivatives.size() != 4)
    return false;
  for (const Eigen::Vector3d &point : point_set)
    if (!point.allFinite())
      return false;
  for (const Eigen::Vector3d &derivative : start_end_derivatives)
    if (!derivative.allFinite())
      return false;

  const Eigen::Index point_count =
      static_cast<Eigen::Index>(point_set.size());
  const Eigen::Index control_point_count = point_count + 2;
  Eigen::MatrixXd system =
      Eigen::MatrixXd::Zero(point_count + 2, control_point_count);
  for (Eigen::Index index = 0; index < point_count; ++index)
  {
    system(index, index) = 1.0 / 6.0;
    system(index, index + 1) = 4.0 / 6.0;
    system(index, index + 2) = 1.0 / 6.0;
  }
  system(point_count, point_count - 1) = -1.0 / (2.0 * interval);
  system(point_count, point_count + 1) = 1.0 / (2.0 * interval);
  system(point_count + 1, point_count - 1) = 1.0 / (interval * interval);
  system(point_count + 1, point_count) = -2.0 / (interval * interval);
  system(point_count + 1, point_count + 1) = 1.0 / (interval * interval);

  control_points = Eigen::MatrixXd::Zero(3, control_point_count);
  if (!enforceCubicBsplineStartBoundary(
          control_points, interval, point_set.front(),
          start_end_derivatives[0], start_end_derivatives[2]))
    return false;

  Eigen::MatrixXd targets(point_count + 2, 3);
  for (Eigen::Index index = 0; index < point_count; ++index)
    targets.row(index) = point_set[static_cast<std::size_t>(index)].transpose();
  targets.row(point_count) = start_end_derivatives[1].transpose();
  targets.row(point_count + 1) = start_end_derivatives[3].transpose();

  const Eigen::MatrixXd reduced_system = system.rightCols(point_count - 1);
  const Eigen::MatrixXd reduced_targets =
      targets - system.leftCols(3) * control_points.leftCols(3).transpose();
  const Eigen::MatrixXd remaining =
      reduced_system.colPivHouseholderQr().solve(reduced_targets);
  if (!remaining.allFinite() || remaining.rows() != point_count - 1 ||
      remaining.cols() != 3)
    return false;
  control_points.rightCols(point_count - 1) = remaining.transpose();
  return enforceCubicBsplineForwardControlPolygon(
      control_points, point_set.front(), point_set.back());
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__REFERENCE_SPLINE_BOUNDARY_H_
