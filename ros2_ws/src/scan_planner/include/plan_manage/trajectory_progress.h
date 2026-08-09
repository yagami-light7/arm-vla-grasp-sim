#ifndef SCAN_PLANNER__TRAJECTORY_PROGRESS_H_
#define SCAN_PLANNER__TRAJECTORY_PROGRESS_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <bspline_opt/uniform_bspline.h>

namespace scan_planner
{

struct TrajectoryProgressCheck
{
  bool safe{false};
  double maximum_reverse_distance{std::numeric_limits<double>::infinity()};
  double minimum_projected_velocity{-std::numeric_limits<double>::infinity()};
};

struct ReferenceCorridorCheck
{
  bool safe{false};
  double maximum_trajectory_deviation{
      std::numeric_limits<double>::infinity()};
  double maximum_guide_anchor_deviation{
      std::numeric_limits<double>::infinity()};
  double initial_guide_progress{
      std::numeric_limits<double>::infinity()};
  double maximum_relative_guide_progress_lead{
      std::numeric_limits<double>::infinity()};
  double maximum_guide_progress_lead{
      std::numeric_limits<double>::infinity()};
};

inline double pointToReferenceGuideDistance(
    const Eigen::Vector3d &point,
    const std::vector<Eigen::Vector3d> &guide)
{
  double minimum_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index + 1 < guide.size(); ++index)
  {
    const Eigen::Vector3d delta = guide[index + 1] - guide[index];
    const double length_squared = delta.squaredNorm();
    if (length_squared <= 1.0e-18)
      continue;
    const double ratio = std::clamp(
        (point - guide[index]).dot(delta) / length_squared, 0.0, 1.0);
    const Eigen::Vector3d nearest = guide[index] + ratio * delta;
    minimum_distance = std::min(minimum_distance, (point - nearest).norm());
  }
  return minimum_distance;
}

// reference 轨迹无论是保留原始 guide 还是动态绕障，都必须双向贴合
// 完整三维折线并按顺序推进。除空间偏差外，guide 弧长进度不得比
// 轨迹已行距离超前过多；这个独立门会拒绝 U 形、平行近邻和自交处
// 几何距离很小但直接跳到后续分支的误捷径。
inline ReferenceCorridorCheck checkTrajectoryReferenceCorridor(
    UniformBspline trajectory,
    const std::vector<Eigen::Vector3d> &spatial_guide,
    const std::vector<Eigen::Vector3d> &progress_guide,
    const double maximum_deviation,
    const double maximum_guide_progress_lead,
    const double requested_sample_interval = 0.01,
    const double initial_progress_tolerance = 0.0,
    const double progress_measurement_tolerance = 0.0)
{
  ReferenceCorridorCheck result;
  const double duration = trajectory.getTimeSum();
  if (!std::isfinite(duration) || duration <= 0.0 ||
      spatial_guide.size() < 2 || progress_guide.size() < 2 ||
      !std::isfinite(maximum_deviation) || maximum_deviation <= 0.0 ||
      !std::isfinite(maximum_guide_progress_lead) ||
      maximum_guide_progress_lead < 0.0 ||
      !std::isfinite(initial_progress_tolerance) ||
      initial_progress_tolerance < 0.0 ||
      !std::isfinite(progress_measurement_tolerance) ||
      progress_measurement_tolerance < 0.0)
    return result;
  for (const Eigen::Vector3d &point : spatial_guide)
    if (!point.allFinite())
      return result;
  for (const Eigen::Vector3d &point : progress_guide)
    if (!point.allFinite())
      return result;

  const double sample_interval =
      std::clamp(requested_sample_interval, 0.005, 0.02);
  const int sample_count =
      std::max(1, static_cast<int>(std::ceil(duration / sample_interval)));
  std::vector<Eigen::Vector3d> trajectory_samples;
  trajectory_samples.reserve(static_cast<std::size_t>(sample_count + 1));
  std::vector<double> guide_arc_lengths(progress_guide.size(), 0.0);
  for (std::size_t index = 1; index < progress_guide.size(); ++index)
  {
    const double length =
        (progress_guide[index] - progress_guide[index - 1]).norm();
    if (!std::isfinite(length) || length <= 1.0e-12)
      return ReferenceCorridorCheck{};
    guide_arc_lengths[index] = guide_arc_lengths[index - 1] + length;
  }

  result.maximum_trajectory_deviation = 0.0;
  result.initial_guide_progress = 0.0;
  result.maximum_relative_guide_progress_lead = 0.0;
  result.maximum_guide_progress_lead = 0.0;
  double minimum_guide_progress = 0.0;
  double guide_origin_progress = 0.0;
  double trajectory_arc_length = 0.0;
  double minimum_progress_balance = 0.0;
  Eigen::Vector3d previous_trajectory_position = Eigen::Vector3d::Zero();
  for (int index = 0; index <= sample_count; ++index)
  {
    const double time =
        duration * static_cast<double>(index) /
        static_cast<double>(sample_count);
    const Eigen::Vector3d position = trajectory.evaluateDeBoorT(time);
    if (!position.allFinite())
      return ReferenceCorridorCheck{};
    if (index > 0)
    {
      const double step_length =
          (position - previous_trajectory_position).norm();
      if (!std::isfinite(step_length))
        return ReferenceCorridorCheck{};
      trajectory_arc_length += step_length;
    }
    previous_trajectory_position = position;
    trajectory_samples.push_back(position);
    const double spatial_distance =
        pointToReferenceGuideDistance(position, spatial_guide);
    if (!std::isfinite(spatial_distance))
      return ReferenceCorridorCheck{};
    double nearest_distance = std::numeric_limits<double>::infinity();
    double nearest_progress = minimum_guide_progress;
    for (std::size_t segment = 0;
         segment + 1 < progress_guide.size(); ++segment)
    {
      const double segment_start = guide_arc_lengths[segment];
      const double segment_end = guide_arc_lengths[segment + 1];
      if (segment_end + 1.0e-12 < minimum_guide_progress)
        continue;
      const Eigen::Vector3d delta =
          progress_guide[segment + 1] - progress_guide[segment];
      const double length = segment_end - segment_start;
      const double minimum_ratio = std::clamp(
          (minimum_guide_progress - segment_start) / length, 0.0, 1.0);
      const double raw_ratio =
          (position - progress_guide[segment]).dot(delta) /
          delta.squaredNorm();
      const double ratio = std::clamp(raw_ratio, minimum_ratio, 1.0);
      const Eigen::Vector3d nearest =
          progress_guide[segment] + ratio * delta;
      const double distance = (position - nearest).norm();
      // 距离并列时保留按 guide 顺序更早的投影，避免在 U 形/楼层重叠处跳段。
      if (distance < nearest_distance - 1.0e-12)
      {
        nearest_distance = distance;
        nearest_progress = segment_start + ratio * length;
      }
    }
    if (!std::isfinite(nearest_distance))
      return ReferenceCorridorCheck{};
    minimum_guide_progress = std::max(
        minimum_guide_progress, nearest_progress);
    if (index == 0)
    {
      // 三次 B-spline 的位置、速度和加速度约束通过过定最小二乘共同
      // 求解，t=0 可能沿 guide 前移少量距离。起点偏置仍受同一个
      // progress-lead 硬门约束，但后续弧长比较必须从该基线重新计零，
      // 否则会把同一拟合偏置重复累计并误拒合法的短折线路径。
      guide_origin_progress = nearest_progress;
      result.initial_guide_progress = nearest_progress;
      result.maximum_guide_progress_lead = nearest_progress;
    }
    result.maximum_trajectory_deviation = std::max(
        result.maximum_trajectory_deviation,
        spatial_distance);
    const double progress_balance =
        (nearest_progress - guide_origin_progress) - trajectory_arc_length;
    // 折返拐点附近的单调投影可能短暂停在前一段：轨迹已经移动、guide
    // 进度尚未跳转，此时负 balance 的绝对值由当前投影距离解释，不能
    // 错当成真实绕远额度。只有扣除这段可恢复距离后仍为负的余额才记入
    // 历史最低值；轨迹真正回到 semantic guide 时距离为零，绕远增量仍
    // 会完整保留并用于约束后续 shortcut。
    minimum_progress_balance = std::min(
        minimum_progress_balance,
        progress_balance + nearest_distance);
    // 绕障或横向回归增加的路程只会降低 balance，不能成为之后抄近路的
    // 永久额度。只限制相对历史最低 balance 的上升，既接受合法绕远，
    // 又会在 U 形、平行近邻或自交跳段时检测到局部 guide 弧长突增。
    const double relative_guide_progress_lead =
        progress_balance - minimum_progress_balance;
    result.maximum_relative_guide_progress_lead = std::max(
        result.maximum_relative_guide_progress_lead,
        relative_guide_progress_lead);
    result.maximum_guide_progress_lead = std::max(
        result.maximum_guide_progress_lead,
        relative_guide_progress_lead);
  }

  result.maximum_guide_anchor_deviation = 0.0;
  std::size_t minimum_sample_index = 0;
  for (const Eigen::Vector3d &anchor : spatial_guide)
  {
    double minimum_distance = std::numeric_limits<double>::infinity();
    std::size_t nearest_sample_index = minimum_sample_index;
    for (std::size_t index = minimum_sample_index;
         index < trajectory_samples.size(); ++index)
    {
      const double distance = (anchor - trajectory_samples[index]).norm();
      if (distance < minimum_distance - 1.0e-12)
      {
        minimum_distance = distance;
        nearest_sample_index = index;
      }
    }
    minimum_sample_index = nearest_sample_index;
    result.maximum_guide_anchor_deviation = std::max(
        result.maximum_guide_anchor_deviation, minimum_distance);
  }
  result.safe =
      result.maximum_trajectory_deviation <= maximum_deviation + 1.0e-9 &&
      result.maximum_guide_anchor_deviation <= maximum_deviation + 1.0e-9 &&
      result.initial_guide_progress <=
      maximum_guide_progress_lead + initial_progress_tolerance + 1.0e-9 &&
      result.maximum_relative_guide_progress_lead <=
      maximum_guide_progress_lead + progress_measurement_tolerance + 1.0e-9;
  return result;
}

// 没有人工回归连接的调用继续让空间、锚点和进度共享同一条 guide。
inline ReferenceCorridorCheck checkTrajectoryReferenceCorridor(
    UniformBspline trajectory,
    const std::vector<Eigen::Vector3d> &guide,
    const double maximum_deviation,
    const double maximum_guide_progress_lead,
    const double requested_sample_interval = 0.01,
    const double initial_progress_tolerance = 0.0,
    const double progress_measurement_tolerance = 0.0)
{
  return checkTrajectoryReferenceCorridor(
      trajectory, guide, guide, maximum_deviation,
      maximum_guide_progress_lead, requested_sample_interval,
      initial_progress_tolerance, progress_measurement_tolerance);
}

// 检查局部轨迹是否持续朝本次滚动窗口的目标推进。
//
// 平面位移有效时只检查 XY 进度，避免斜坡上持续增加的 z 掩盖水平回退；
// 近似竖直的路径才退化为三维方向。允许的微小负值仅用于吸收数值误差。
inline TrajectoryProgressCheck checkTrajectoryForwardProgress(
    UniformBspline trajectory,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &target,
    const double maximum_reverse_distance,
    const double maximum_reverse_speed,
    const double requested_sample_interval = 0.02)
{
  TrajectoryProgressCheck result;
  const double duration = trajectory.getTimeSum();
  if (!std::isfinite(duration) || duration <= 0.0 ||
      !std::isfinite(maximum_reverse_distance) || maximum_reverse_distance < 0.0 ||
      !std::isfinite(maximum_reverse_speed) || maximum_reverse_speed < 0.0)
  {
    return result;
  }

  const Eigen::Vector2d planar_delta = target.head<2>() - start.head<2>();
  const bool use_planar_direction = planar_delta.norm() > 1.0e-6;
  const Eigen::Vector2d planar_direction =
      use_planar_direction ? planar_delta.normalized() : Eigen::Vector2d::Zero();
  const Eigen::Vector3d spatial_delta = target - start;
  if (!use_planar_direction && spatial_delta.norm() <= 1.0e-6)
  {
    return result;
  }
  const Eigen::Vector3d spatial_direction =
      use_planar_direction ? Eigen::Vector3d::Zero() : spatial_delta.normalized();

  const auto project = [&](const Eigen::Vector3d &value) {
    return use_planar_direction
               ? value.head<2>().dot(planar_direction)
               : value.dot(spatial_direction);
  };

  const double sample_interval =
      std::clamp(requested_sample_interval, 0.005, 0.05);
  const int sample_count =
      std::max(1, static_cast<int>(std::ceil(duration / sample_interval)));
  UniformBspline velocity = trajectory.getDerivative();

  double furthest_progress = -std::numeric_limits<double>::infinity();
  result.maximum_reverse_distance = 0.0;
  result.minimum_projected_velocity = std::numeric_limits<double>::infinity();
  for (int index = 0; index <= sample_count; ++index)
  {
    const double time =
        duration * static_cast<double>(index) / static_cast<double>(sample_count);
    const Eigen::Vector3d position = trajectory.evaluateDeBoorT(time);
    const Eigen::Vector3d current_velocity = velocity.evaluateDeBoorT(time);
    if (!position.allFinite() || !current_velocity.allFinite())
    {
      return TrajectoryProgressCheck{};
    }

    const double progress = project(position - start);
    furthest_progress = std::max(furthest_progress, progress);
    result.maximum_reverse_distance =
        std::max(result.maximum_reverse_distance, furthest_progress - progress);
    result.minimum_projected_velocity =
        std::min(result.minimum_projected_velocity, project(current_velocity));
  }

  result.safe =
      result.maximum_reverse_distance <= maximum_reverse_distance + 1.0e-9 &&
      result.minimum_projected_velocity >= -maximum_reverse_speed - 1.0e-9;
  return result;
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__TRAJECTORY_PROGRESS_H_
