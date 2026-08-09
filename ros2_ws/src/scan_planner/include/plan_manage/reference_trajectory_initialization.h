#ifndef SCAN_PLANNER__REFERENCE_TRAJECTORY_INITIALIZATION_H_
#define SCAN_PLANNER__REFERENCE_TRAJECTORY_INITIALIZATION_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace scan_planner
{

struct ReferenceTrajectoryInitialization
{
  // 保留 PCT 转角和高度锚点，专用于碰撞与有序走廊验证。
  std::vector<Eigen::Vector3d> spatial_guide;
  // 在等时间拍上按加速、巡航、减速进度采样的样条拟合点。
  std::vector<Eigen::Vector3d> parameterization_points;
  double time_step{0.0};
  double duration{0.0};
  double peak_speed{0.0};
  double start_speed{0.0};
  double terminal_speed{0.0};
};

inline bool buildDensifiedReferenceGuide(
    const std::vector<Eigen::Vector3d> &raw_guide,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &target,
    const double maximum_segment_length,
    std::vector<Eigen::Vector3d> &spatial_guide)
{
  constexpr std::size_t kMaximumPointCount = 128;
  if (raw_guide.size() < 2 || !start.allFinite() || !target.allFinite() ||
      !std::isfinite(maximum_segment_length) ||
      maximum_segment_length <= 0.0)
    return false;

  std::vector<Eigen::Vector3d> anchors = raw_guide;
  anchors.front() = start;
  anchors.back() = target;
  // Path 投影会在真实 Odometry 起点后保留一个亚栅格回归点。若把毫米级
  // 回归连接当成初切线，实测前进速度会与时间剖面的标量初速互相矛盾，
  // 最终只能把整条 B-spline 统一拉长。小于一个地图采样尺度的锚点
  // 不表达可分辨的转角；精确终点仍在下方单独保留。
  const double minimum_anchor_spacing =
      std::max(1.0e-3, maximum_segment_length);
  std::vector<Eigen::Vector3d> deduplicated;
  deduplicated.reserve(anchors.size());
  deduplicated.push_back(anchors.front());
  for (std::size_t index = 1; index + 1 < anchors.size(); ++index)
  {
    const Eigen::Vector3d &point = anchors[index];
    if (!point.allFinite())
      return false;
    if ((point - deduplicated.back()).norm() > minimum_anchor_spacing)
      deduplicated.push_back(point);
  }
  if (!anchors.back().allFinite())
    return false;
  if ((anchors.back() - deduplicated.back()).norm() >
      minimum_anchor_spacing)
  {
    deduplicated.push_back(anchors.back());
  }
  else if (deduplicated.size() >= 2U)
  {
    deduplicated.back() = anchors.back();
  }
  else if ((anchors.back() - deduplicated.back()).norm() > 1.0e-9)
  {
    // 终点捕获段可能恰好只有一个地图栅格，甚至略短于地图采样尺度。
    // minimum_anchor_spacing 只用于删除不表达可分辨转角的中间锚点，
    // 不能因此删掉唯一的前向终点；短段仍会在下方加密到七个样本，
    // 并继续接受完整碰撞、动力学和有序走廊检查。
    deduplicated.push_back(anchors.back());
  }
  else
  {
    return false;
  }
  if (deduplicated.size() < 2 || deduplicated.size() > kMaximumPointCount)
    return false;

  spatial_guide.clear();
  spatial_guide.reserve(deduplicated.size());
  spatial_guide.push_back(deduplicated.front());
  for (std::size_t index = 0; index + 1 < deduplicated.size(); ++index)
  {
    const Eigen::Vector3d delta =
        deduplicated[index + 1] - deduplicated[index];
    const double length = delta.norm();
    if (!std::isfinite(length) || length <= 1.0e-9)
      return false;
    const std::size_t subdivision_count = static_cast<std::size_t>(
        std::max(1.0, std::ceil(length / maximum_segment_length)));
    if (spatial_guide.size() + subdivision_count > kMaximumPointCount)
      return false;
    for (std::size_t subdivision = 1;
         subdivision <= subdivision_count; ++subdivision)
    {
      const double ratio = static_cast<double>(subdivision) /
          static_cast<double>(subdivision_count);
      spatial_guide.push_back(deduplicated[index] + ratio * delta);
    }
  }

  // 三次 B-spline 参数化至少需要七个样本。只在最长的
  // 原折线边上插入中点，不删除或移动任何 PCT 锚点。
  while (spatial_guide.size() < 7)
  {
    std::size_t longest_index = 0;
    double longest_length = -1.0;
    for (std::size_t index = 0;
         index + 1 < spatial_guide.size(); ++index)
    {
      const double length =
          (spatial_guide[index + 1] - spatial_guide[index]).norm();
      if (length > longest_length)
      {
        longest_length = length;
        longest_index = index;
      }
    }
    if (!std::isfinite(longest_length) || longest_length <= 1.0e-9)
      return false;
    spatial_guide.insert(
        spatial_guide.begin() +
        static_cast<std::ptrdiff_t>(longest_index + 1),
        0.5 * (spatial_guide[longest_index] +
        spatial_guide[longest_index + 1]));
  }
  return spatial_guide.size() <= kMaximumPointCount;
}

inline bool initializeReferenceTrajectory(
    const std::vector<Eigen::Vector3d> &raw_guide,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &target,
    const double maximum_velocity,
    const double maximum_acceleration,
    const double maximum_segment_length,
    const Eigen::Vector3d &start_velocity,
    const Eigen::Vector3d &terminal_velocity,
    ReferenceTrajectoryInitialization &result)
{
  constexpr std::size_t kMaximumPointCount = 128;
  result = ReferenceTrajectoryInitialization{};
  if (!std::isfinite(maximum_velocity) || maximum_velocity <= 0.0 ||
      !std::isfinite(maximum_acceleration) || maximum_acceleration <= 0.0 ||
      !start_velocity.allFinite() || !terminal_velocity.allFinite() ||
      !buildDensifiedReferenceGuide(
          raw_guide, start, target, maximum_segment_length,
          result.spatial_guide))
    return false;

  std::vector<double> cumulative_length(
      result.spatial_guide.size(), 0.0);
  for (std::size_t index = 1;
       index < result.spatial_guide.size(); ++index)
  {
    cumulative_length[index] = cumulative_length[index - 1] +
        (result.spatial_guide[index] -
        result.spatial_guide[index - 1]).norm();
  }
  const double total_length = cumulative_length.back();
  if (!std::isfinite(total_length) || total_length <= 1.0e-6)
    return false;

  const Eigen::Vector3d initial_tangent =
      (result.spatial_guide[1] - result.spatial_guide[0]).normalized();
  const Eigen::Vector3d terminal_tangent =
      (result.spatial_guide.back() -
      result.spatial_guide[result.spatial_guide.size() - 2]).normalized();
  result.start_speed = std::clamp(
      start_velocity.dot(initial_tangent), 0.0, maximum_velocity);
  result.terminal_speed = std::clamp(
      terminal_velocity.dot(terminal_tangent), 0.0, maximum_velocity);

  const double minimum_speed_change_distance = std::abs(
      result.terminal_speed * result.terminal_speed -
      result.start_speed * result.start_speed) /
      (2.0 * maximum_acceleration);
  if (minimum_speed_change_distance > total_length + 1.0e-9)
    return false;

  double acceleration_time = 0.0;
  double cruise_time = 0.0;
  double deceleration_time = 0.0;
  const double full_acceleration_distance =
      (maximum_velocity * maximum_velocity -
      result.start_speed * result.start_speed) /
      (2.0 * maximum_acceleration);
  const double full_deceleration_distance =
      (maximum_velocity * maximum_velocity -
      result.terminal_speed * result.terminal_speed) /
      (2.0 * maximum_acceleration);
  if (full_acceleration_distance + full_deceleration_distance >= total_length)
  {
    result.peak_speed = std::sqrt(
        maximum_acceleration * total_length +
        0.5 * (result.start_speed * result.start_speed +
        result.terminal_speed * result.terminal_speed));
    if (result.peak_speed + 1.0e-9 <
        std::max(result.start_speed, result.terminal_speed))
      return false;
  }
  else
  {
    result.peak_speed = maximum_velocity;
    cruise_time =
        (total_length - full_acceleration_distance -
        full_deceleration_distance) / maximum_velocity;
  }
  acceleration_time =
      (result.peak_speed - result.start_speed) / maximum_acceleration;
  deceleration_time =
      (result.peak_speed - result.terminal_speed) / maximum_acceleration;
  result.duration = acceleration_time + cruise_time + deceleration_time;
  if (!std::isfinite(result.peak_speed) || result.peak_speed <= 0.0 ||
      !std::isfinite(result.duration) || result.duration <= 0.0)
    return false;

  // 等时间样本的最大理论位移不超过地图尺度，并保留
  // 至少七个样本。这个上限与 planner 的输入点资源门一致。
  const std::size_t sample_count = static_cast<std::size_t>(std::max(
      7.0,
      std::ceil(
          result.duration * result.peak_speed /
          maximum_segment_length) + 1.0));
  if (sample_count > kMaximumPointCount)
    return false;
  result.time_step =
      result.duration / static_cast<double>(sample_count - 1);
  if (!std::isfinite(result.time_step) || result.time_step <= 1.0e-4)
    return false;

  result.parameterization_points.reserve(sample_count);
  std::size_t segment_index = 0;
  const double acceleration_distance =
      result.start_speed * acceleration_time +
      0.5 * maximum_acceleration * acceleration_time * acceleration_time;
  for (std::size_t sample = 0; sample < sample_count; ++sample)
  {
    const double time = result.duration * static_cast<double>(sample) /
        static_cast<double>(sample_count - 1);
    double arc_length = 0.0;
    if (time <= acceleration_time)
    {
      arc_length = result.start_speed * time +
          0.5 * maximum_acceleration * time * time;
    }
    else if (time <= acceleration_time + cruise_time)
    {
      arc_length = acceleration_distance +
          result.peak_speed * (time - acceleration_time);
    }
    else
    {
      const double remaining_time = result.duration - time;
      arc_length = total_length -
          (result.terminal_speed * remaining_time +
          0.5 * maximum_acceleration * remaining_time * remaining_time);
    }
    arc_length = std::clamp(arc_length, 0.0, total_length);
    while (segment_index + 1 < cumulative_length.size() - 1 &&
           cumulative_length[segment_index + 1] < arc_length)
      ++segment_index;
    const double segment_start = cumulative_length[segment_index];
    const double segment_finish = cumulative_length[segment_index + 1];
    const double denominator = segment_finish - segment_start;
    if (!std::isfinite(denominator) || denominator <= 1.0e-9)
      return false;
    const double ratio = std::clamp(
        (arc_length - segment_start) / denominator, 0.0, 1.0);
    result.parameterization_points.push_back(
        result.spatial_guide[segment_index] + ratio *
        (result.spatial_guide[segment_index + 1] -
        result.spatial_guide[segment_index]));
  }
  result.parameterization_points.front() = start;
  result.parameterization_points.back() = target;
  return result.parameterization_points.size() >= 7 &&
      result.parameterization_points.size() <= kMaximumPointCount;
}

// 保留旧调用方的静止边界语义；reference 在线巡航应显式传入首末速度。
inline bool initializeReferenceTrajectory(
    const std::vector<Eigen::Vector3d> &raw_guide,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &target,
    const double maximum_velocity,
    const double maximum_acceleration,
    const double maximum_segment_length,
    ReferenceTrajectoryInitialization &result)
{
  return initializeReferenceTrajectory(
      raw_guide, start, target, maximum_velocity, maximum_acceleration,
      maximum_segment_length, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), result);
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__REFERENCE_TRAJECTORY_INITIALIZATION_H_
