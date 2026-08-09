#ifndef SCAN_PLANNER__REFERENCE_VELOCITY_H_
#define SCAN_PLANNER__REFERENCE_VELOCITY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace scan_planner
{

// 四足步态会让单帧机体速度随抬腿/落脚周期明显振荡。滚动重规划若直接取
// 某一帧，可能在机器人持续前进时反复得到近零甚至反向速度。该一阶滤波器
// 只平滑已经由 Odometry 实测的世界系速度，不补速度、不使用控制指令猜测
// 运动状态；时间戳不前进时保持上一结果，避免迟到消息倒推滤波状态。
class ReferenceVelocityLowPassFilter
{
public:
  Eigen::Vector3d update(
      const Eigen::Vector3d &measured_velocity,
      const std::int64_t stamp_ns,
      const double time_constant_sec)
  {
    if (!measured_velocity.allFinite() || stamp_ns <= 0 ||
        !std::isfinite(time_constant_sec) || time_constant_sec < 0.0)
      return initialized_ ? filtered_velocity_ : Eigen::Vector3d::Zero();

    if (!initialized_)
    {
      initialized_ = true;
      last_stamp_ns_ = stamp_ns;
      filtered_velocity_ = measured_velocity;
      return filtered_velocity_;
    }

    if (stamp_ns <= last_stamp_ns_)
      return filtered_velocity_;

    const double dt_sec = static_cast<double>(stamp_ns - last_stamp_ns_) * 1.0e-9;
    last_stamp_ns_ = stamp_ns;
    if (time_constant_sec <= 1.0e-9)
    {
      filtered_velocity_ = measured_velocity;
      return filtered_velocity_;
    }

    // -expm1(-x) 在很小采样周期下比 1-exp(-x) 更稳定。
    const double alpha = -std::expm1(-dt_sec / time_constant_sec);
    filtered_velocity_ += alpha * (measured_velocity - filtered_velocity_);
    return filtered_velocity_;
  }

  void reset()
  {
    initialized_ = false;
    last_stamp_ns_ = 0;
    filtered_velocity_.setZero();
  }

  bool initialized() const { return initialized_; }

  const Eigen::Vector3d &value() const { return filtered_velocity_; }

private:
  bool initialized_{false};
  std::int64_t last_stamp_ns_{0};
  Eigen::Vector3d filtered_velocity_{Eigen::Vector3d::Zero()};
};

// 只保留朝局部参考目标的实测速度分量，避免四足落脚侧滑成为硬边界。
inline Eigen::Vector3d projectVelocityOntoReference(
    const Eigen::Vector3d &measured_velocity,
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &local_target,
    const double maximum_speed)
{
  const Eigen::Vector3d delta = local_target - start;
  if (!measured_velocity.allFinite() || !delta.allFinite() ||
      !std::isfinite(maximum_speed) || maximum_speed <= 0.0 ||
      delta.norm() <= 1.0e-6)
  {
    return Eigen::Vector3d::Zero();
  }

  const Eigen::Vector3d direction = delta.normalized();
  const double forward_speed = std::clamp(
      measured_velocity.dot(direction), 0.0, maximum_speed);
  return direction * forward_speed;
}

// Path 投影后的第一个点可能只是毫米级横向/高度回归点，不能代表机器人的
// 前进切线。沿 guide 寻找第一个达到可分辨前视距离的点，再把实测速度投影
// 到该方向；这与 reference 初始化过滤亚栅格锚点的尺度保持一致。
inline Eigen::Vector3d projectVelocityOntoReferenceGuide(
    const Eigen::Vector3d &measured_velocity,
    const Eigen::Vector3d &start,
    const std::vector<Eigen::Vector3d> &ordered_guide,
    const double minimum_lookahead_distance,
    const double maximum_speed)
{
  if (!measured_velocity.allFinite() || !start.allFinite() ||
      ordered_guide.size() < 2 ||
      !std::isfinite(minimum_lookahead_distance) ||
      minimum_lookahead_distance <= 0.0)
  {
    return Eigen::Vector3d::Zero();
  }
  for (const Eigen::Vector3d &point : ordered_guide)
  {
    if (!point.allFinite())
      return Eigen::Vector3d::Zero();
    if ((point - start).norm() > minimum_lookahead_distance)
    {
      return projectVelocityOntoReference(
          measured_velocity, start, point, maximum_speed);
    }
  }
  return Eigen::Vector3d::Zero();
}

// 为非最终滚动窗口生成沿有序参考折线末切线的连续巡航速度。
// 速度值完全来自 ROS 参数；该 helper 只负责有限性检查和上限裁剪。
inline Eigen::Vector3d referenceCruiseVelocityAlongGuide(
    const std::vector<Eigen::Vector3d> &ordered_guide,
    const double configured_speed,
    const double maximum_speed)
{
  if (ordered_guide.size() < 2 ||
      !std::isfinite(configured_speed) || configured_speed < 0.0 ||
      !std::isfinite(maximum_speed) || maximum_speed <= 0.0 ||
      !ordered_guide.back().allFinite())
  {
    return Eigen::Vector3d::Zero();
  }

  const double speed = std::min(configured_speed, maximum_speed);
  if (speed <= 1.0e-6)
    return Eigen::Vector3d::Zero();

  const Eigen::Vector3d &target = ordered_guide.back();
  for (std::size_t index = ordered_guide.size() - 1; index > 0; --index)
  {
    const Eigen::Vector3d &previous = ordered_guide[index - 1];
    if (!previous.allFinite())
      return Eigen::Vector3d::Zero();
    const Eigen::Vector3d tangent = target - previous;
    if (tangent.norm() > 1.0e-6)
      return tangent.normalized() * speed;
  }
  return Eigen::Vector3d::Zero();
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__REFERENCE_VELOCITY_H_
