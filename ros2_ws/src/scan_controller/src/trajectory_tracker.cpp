#include "scan_controller/trajectory_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace scan_controller
{
namespace
{

constexpr double kEpsilon = 1.0e-9;
constexpr double kStationaryPositionTolerance = 1.0e-6;
constexpr double kActiveSensingHardMaxYawExcursion = 0.22;
constexpr double kActiveSensingHardMaxYawRate = 0.20;

bool finitePositive(double value)
{
  return std::isfinite(value) && value > 0.0;
}

std::int64_t secondsToNanoseconds(double seconds)
{
  if (!finitePositive(seconds)) {
    return 0;
  }
  const long double scaled =
    static_cast<long double>(seconds) * 1000000000.0L;
  if (
    scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    return 0;
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

bool isStationaryTrajectory(const Eigen::MatrixXd & control_points)
{
  const Eigen::Vector3d anchor = control_points.col(0);
  const double tolerance_squared =
    kStationaryPositionTolerance * kStationaryPositionTolerance;
  for (Eigen::Index index = 1; index < control_points.cols(); ++index) {
    if ((control_points.col(index) - anchor).squaredNorm() > tolerance_squared) {
      return false;
    }
  }
  return true;
}

bool hasStrictlyCoincidentControlPoints(
  const Eigen::MatrixXd & control_points)
{
  const Eigen::Vector3d anchor = control_points.col(0);
  for (Eigen::Index index = 1; index < control_points.cols(); ++index) {
    if ((control_points.col(index).array() != anchor.array()).any()) {
      return false;
    }
  }
  return true;
}

}  // 匿名命名空间

TrajectoryTracker::TrajectoryTracker(const TrackerConfig & config)
: config_(config)
{
  validateConfig(config_);
}

void TrajectoryTracker::validateConfig(const TrackerConfig & config)
{
  const double positive_values[] = {
    config.time_forward,
    config.yaw_alignment_min_chord_distance,
    config.heading_error_threshold,
    config.heading_error_release_threshold,
    config.cross_track_alignment_distance,
    config.cross_track_alignment_release_distance,
    config.cross_track_heading_error_threshold,
    config.cross_track_heading_error_release_threshold,
    config.cross_track_recovery_forward_speed,
    config.cross_track_recovery_taper_distance,
    config.cross_track_recovery_lateral_gain,
    config.cross_track_recovery_lateral_speed,
    config.cross_track_heading_assist_gain,
    config.cross_track_heading_assist_max,
    config.turning_yaw_rate_threshold,
    config.turning_max_planar_speed,
    config.stair_heading_lock_half_window_arc,
    config.stair_heading_lock_min_pitch_rad,
    config.reference_path_body_height,
    config.reference_path_min_point_spacing,
    config.reference_path_forward_arc,
    config.kp_position,
    config.kp_yaw,
    config.max_vx,
    config.max_vy,
    config.max_yaw_rate,
    config.active_sensing_max_yaw_excursion,
    config.active_sensing_max_yaw_rate,
    config.active_sensing_yaw_tolerance,
    config.max_ax,
    config.max_ay,
    config.max_yaw_acc,
    config.finish_distance_xy,
    config.finish_distance_z,
    config.terminal_capture_entry_distance_xy,
    config.terminal_capture_zero_hold_distance_xy,
    config.terminal_capture_release_distance_xy,
    config.terminal_capture_brake_hold_sec,
    config.terminal_capture_stable_dwell_sec,
    config.terminal_position_hold_gain,
    config.terminal_position_hold_max_speed,
    config.terminal_approach_min_speed,
    config.terminal_max_yaw_rate,
    config.terminal_max_yaw_acc,
    config.terminal_yaw_control_deadband,
    config.finish_yaw_error,
    config.finish_speed,
    config.finish_vertical_speed,
    config.finish_angular_speed,
    config.finish_yaw_rate,
    config.bspline_timeout_sec,
    config.odom_timeout_sec,
    config.cloud_timeout_sec,
    config.trajectory_expiry_grace_sec,
    config.max_yaw_alignment_freeze_sec,
    config.max_control_dt_sec,
    config.max_start_stamp_skew_sec,
  };
  for (const double value : positive_values) {
    if (!finitePositive(value)) {
      throw std::invalid_argument("控制器参数必须是有限正数");
    }
  }
  if (
    !std::isfinite(config.future_tolerance_sec) ||
    config.future_tolerance_sec < 0.0)
  {
    throw std::invalid_argument("未来时间容差必须是有限非负数");
  }
  if (config.heading_error_threshold > M_PI) {
    throw std::invalid_argument("航向误差阈值不能大于 pi");
  }
  if (
    config.heading_error_release_threshold >=
    config.heading_error_threshold)
  {
    throw std::invalid_argument(
            "普通航向释放阈值必须小于进入阈值");
  }
  if (
    config.cross_track_heading_error_threshold >
    config.heading_error_threshold)
  {
    throw std::invalid_argument(
            "横向偏离时的航向阈值不能大于普通航向阈值");
  }
  if (
    config.cross_track_heading_error_release_threshold >=
    config.cross_track_heading_error_threshold)
  {
    throw std::invalid_argument(
            "横向偏离航向释放阈值必须小于进入阈值");
  }
  if (config.cross_track_recovery_forward_speed > config.max_vx) {
    throw std::invalid_argument(
            "横向恢复前向速度不能大于纵向速度上限");
  }
  if (
    config.cross_track_heading_assist_max >=
    config.cross_track_heading_error_threshold)
  {
    throw std::invalid_argument(
            "横向恢复航向辅助上限必须小于原地对齐阈值");
  }
  if (config.turning_yaw_rate_threshold > config.max_yaw_rate) {
    throw std::invalid_argument(
            "转弯限速触发角速度不能大于角速度硬上限");
  }
  if (
    config.turning_max_planar_speed >
    std::hypot(config.max_vx, config.max_vy))
  {
    throw std::invalid_argument(
            "转弯合成平移速度上限不能大于全局合成平移速度上限");
  }
  if (
    config.cross_track_alignment_release_distance >=
    config.cross_track_alignment_distance)
  {
    throw std::invalid_argument(
            "横向回正释放距离必须小于进入距离");
  }
  if (config.stair_heading_lock_min_pitch_rad >= M_PI_2) {
    throw std::invalid_argument(
            "楼梯航向锁定的最小坡角必须小于 pi/2");
  }
  if (
    !std::isfinite(config.stair_forward_speed_floor) ||
    config.stair_forward_speed_floor < 0.0)
  {
    throw std::invalid_argument(
            "楼梯前向牵引速度下限必须是有限非负数");
  }
  if (config.stair_forward_speed_floor > config.max_vx) {
    throw std::invalid_argument(
            "楼梯前向牵引速度下限不能大于纵向速度上限");
  }
  if (
    config.stair_forward_speed_floor > kEpsilon &&
    !config.stair_heading_lock_enabled)
  {
    throw std::invalid_argument(
            "启用楼梯前向牵引速度前必须启用楼梯航向锁定");
  }
  if (
    !std::isfinite(config.reference_path_backward_arc) ||
    config.reference_path_backward_arc < 0.0)
  {
    throw std::invalid_argument(
            "参考 Path 后向弧长窗口必须是有限非负数");
  }
  if (config.reference_path_max_points < 2) {
    throw std::invalid_argument("参考 Path 点数上限必须至少为 2");
  }
  if (
    config.terminal_capture_entry_distance_xy >=
    config.finish_distance_xy)
  {
    throw std::invalid_argument(
            "终点捕获进入距离必须严格小于完成距离");
  }
  if (
    config.terminal_capture_zero_hold_distance_xy <=
    config.terminal_capture_entry_distance_xy ||
    config.terminal_capture_zero_hold_distance_xy >=
    config.finish_distance_xy)
  {
    throw std::invalid_argument(
            "终点零速保持距离必须严格位于捕获进入距离与完成距离之间");
  }
  if (
    config.terminal_capture_release_distance_xy <=
    config.finish_distance_xy)
  {
    throw std::invalid_argument(
            "终点捕获释放距离必须大于完成距离");
  }
  if (config.terminal_max_yaw_rate > config.max_yaw_rate) {
    throw std::invalid_argument(
            "终点捕获角速度上限不能大于全局角速度上限");
  }
  if (
    config.terminal_position_hold_max_speed > config.max_vx ||
    config.terminal_position_hold_max_speed > config.max_vy)
  {
    throw std::invalid_argument(
            "终点位置保持速度不能大于控制器任一平移速度上限");
  }
  if (
    config.terminal_approach_min_speed >
    std::hypot(config.max_vx, config.max_vy))
  {
    throw std::invalid_argument(
            "终点最低接近速度不能大于控制器合成平移速度上限");
  }
  if (config.active_sensing_max_yaw_rate > config.max_yaw_rate) {
    throw std::invalid_argument(
            "主动观测角速度上限不能大于全局角速度上限");
  }
  if (
    config.active_sensing_max_yaw_excursion >
    kActiveSensingHardMaxYawExcursion ||
    config.active_sensing_max_yaw_excursion >= M_PI)
  {
    throw std::invalid_argument(
            "主动观测最大扫描幅度不能超过代码级 0.22rad 上限");
  }
  if (
    config.active_sensing_max_yaw_rate >
    kActiveSensingHardMaxYawRate)
  {
    throw std::invalid_argument(
            "主动观测角速度不能超过代码级 0.20rad/s 上限");
  }
  if (
    config.active_sensing_yaw_tolerance >=
    config.active_sensing_max_yaw_excursion)
  {
    throw std::invalid_argument(
            "主动观测 yaw 到达容差必须严格小于最大扫描幅度");
  }
  if (config.terminal_max_yaw_acc > config.max_yaw_acc) {
    throw std::invalid_argument(
            "终点捕获角加速度上限不能大于全局角加速度上限");
  }
  if (config.terminal_yaw_control_deadband >= config.finish_yaw_error) {
    throw std::invalid_argument(
            "终点 yaw 控制死区必须严格小于完成 yaw 误差门限");
  }
}

bool TrajectoryTracker::setTrajectory(
  const TrajectoryInput & input, double now_sec, std::string & error)
{
  error.clear();
  if (!finitePositive(now_sec)) {
    error = "当前 ROS 时间必须为正数";
    invalidateTrajectory();
    return false;
  }
  const std::int64_t input_reference_path_stamp_ns =
    input.reference_path_stamp_ns > 0 ?
    input.reference_path_stamp_ns :
    secondsToNanoseconds(input.reference_path_stamp_sec);
  const std::int64_t input_header_stamp_ns =
    input.header_stamp_ns > 0 ?
    input.header_stamp_ns : secondsToNanoseconds(input.header_stamp_sec);
  const std::int64_t input_start_stamp_ns =
    input.start_stamp_ns > 0 ?
    input.start_stamp_ns : secondsToNanoseconds(input.start_stamp_sec);

  // 先按代际和轨迹序号拒绝明确的迟到包。它们不能让一条更新且有效的轨迹
  // 失效；同代内容是否合法只在确认其不是旧包后再做完整校验。
  if (
    have_reference_path_ && input_reference_path_stamp_ns > 0 &&
    input_reference_path_stamp_ns != reference_path_stamp_ns_)
  {
    error = "B-spline 与当前参考 Path 代际不匹配";
    return false;
  }
  const bool same_recorded_path_generation =
    trajectory_reference_path_stamp_ns_ > 0 &&
    input_reference_path_stamp_ns == trajectory_reference_path_stamp_ns_;
  if (
    same_recorded_path_generation && trajectory_id_ > 0 &&
    (input.trajectory_id < trajectory_id_ ||
    (input.trajectory_id > trajectory_id_ &&
    input_header_stamp_ns > 0 && trajectory_header_stamp_ns_ > 0 &&
    input_header_stamp_ns < trajectory_header_stamp_ns_)))
  {
    error = "收到迟于当前状态到达的旧 B-spline identity";
    return false;
  }
  if (input.order < 1 || input.order >= input.control_points.cols()) {
    error = "B-spline 阶数与控制点数量不匹配";
    invalidateTrajectory();
    return false;
  }
  if (
    input.control_points.rows() != 3 ||
    input.control_points.cols() < input.order + 1 ||
    !input.control_points.allFinite())
  {
    error = "B-spline 控制点必须是有限三维点且数量充分";
    invalidateTrajectory();
    return false;
  }
  if (!std::isfinite(input.yaw_dt)) {
    error = "B-spline yaw_dt 必须为有限数";
    invalidateTrajectory();
    return false;
  }
  const bool active_sensing_yaw_only = !input.yaw_points.empty();
  if (active_sensing_yaw_only) {
    const bool yaw_points_are_finite = std::all_of(
      input.yaw_points.begin(), input.yaw_points.end(),
      [](double value) {return std::isfinite(value);});
    if (
      input.yaw_points.size() != 2U || !yaw_points_are_finite ||
      !finitePositive(input.yaw_dt) || input.is_final ||
      input.emergency_stop ||
      !hasStrictlyCoincidentControlPoints(input.control_points))
    {
      error =
        "主动观测 yaw-only 必须为两点、有限正 yaw_dt、严格同点位置且非 final/急停";
      invalidateTrajectory();
      return false;
    }
    // yaw_pts 使用绝对、可连续展开的角度；直接差值保留扫描方向，不能先
    // wrap 后再做幅度认证，否则跨越 pi 的异常载荷会伪装成小角度请求。
    const double signed_excursion =
      input.yaw_points[1] - input.yaw_points[0];
    const double implied_rate = std::abs(signed_excursion) / input.yaw_dt;
    if (
      std::abs(signed_excursion) >
      config_.active_sensing_max_yaw_excursion + kEpsilon ||
      implied_rate > config_.active_sensing_max_yaw_rate + kEpsilon)
    {
      error = "主动观测 yaw-only 超过角度幅度或隐含角速度上限";
      invalidateTrajectory();
      return false;
    }
  }
  const Eigen::Index expected_knot_count =
    input.control_points.cols() + input.order + 1;
  if (
    input.knots.size() != expected_knot_count ||
    !input.knots.allFinite())
  {
    error = "B-spline knot 数量或数值非法";
    invalidateTrajectory();
    return false;
  }
  for (Eigen::Index index = 1; index < input.knots.size(); ++index) {
    if (input.knots(index) + kEpsilon < input.knots(index - 1)) {
      error = "B-spline knot 必须单调不减";
      invalidateTrajectory();
      return false;
    }
  }
  for (
    Eigen::Index index = 0;
    index + 1 < input.control_points.cols();
    ++index)
  {
    const Eigen::Index upper = index + input.order + 1;
    const Eigen::Index lower = index + 1;
    if (input.knots(upper) - input.knots(lower) <= kEpsilon) {
      error = "B-spline 导数 knot 区间必须为正";
      invalidateTrajectory();
      return false;
    }
  }
  if (
    !finitePositive(input.header_stamp_sec) ||
    !finitePositive(input.start_stamp_sec) ||
    !finitePositive(input.reference_path_stamp_sec))
  {
    error = "B-spline 发布时间、起始时间与参考 Path 代际必须为有效正时间";
    invalidateTrajectory();
    return false;
  }
  if (
    input_reference_path_stamp_ns <= 0 || input_header_stamp_ns <= 0 ||
    input_start_stamp_ns <= 0 || input.trajectory_id <= 0)
  {
    error = "B-spline 发布时间、起始时间、Path 代际和轨迹 id 必须为有效正值";
    invalidateTrajectory();
    return false;
  }
  if (
    invalidated_reference_path_stamp_ns_ > 0 &&
    input_reference_path_stamp_ns <= invalidated_reference_path_stamp_ns_)
  {
    error = "B-spline 属于已清除或冲突作废的参考 Path 代际";
    invalidateTrajectory();
    return false;
  }
  if (
    trajectory_finished_ && trajectory_is_final_ &&
    same_recorded_path_generation)
  {
    error = "目标已完成，同一 Path 代际的 B-spline 不能解除完成锁存";
    return false;
  }

  if (
    same_recorded_path_generation && trajectory_id_ > 0 &&
    input.trajectory_id == trajectory_id_)
  {
    if (trajectory_identity_poisoned_) {
      error = "B-spline identity 已因内容冲突作废";
      return false;
    }
    const Eigen::MatrixXd current_control_points =
      position_trajectory_.getControlPoint();
    const Eigen::VectorXd current_knots = position_trajectory_.getKnot();
    const bool exact_duplicate =
      have_trajectory_ && input_header_stamp_ns == trajectory_header_stamp_ns_ &&
      input_start_stamp_ns == trajectory_start_stamp_ns_ &&
      input.order == trajectory_order_ &&
      input.is_final == trajectory_is_final_ &&
      input.emergency_stop == emergency_stop_latched_ &&
      input.control_points.rows() == current_control_points.rows() &&
      input.control_points.cols() == current_control_points.cols() &&
      input.knots.size() == current_knots.size() &&
      input.yaw_points == trajectory_yaw_points_ &&
      input.yaw_dt == trajectory_yaw_dt_ &&
      (input.control_points - current_control_points).cwiseAbs().maxCoeff() <=
      kEpsilon &&
      (input.knots - current_knots).cwiseAbs().maxCoeff() <= kEpsilon;
    if (exact_duplicate) {
      // DDS 重发不得重置执行时间、stationary dwell 或完成生命周期；即使
      // 重发到达时 Header 已变旧，也不能让它打掉当前有效状态。
      return true;
    }
    error = "同一 B-spline identity 出现不同时间戳、控制点或语义";
    trajectory_identity_poisoned_ = true;
    invalidateTrajectory();
    return false;
  }
  if (
    active_sensing_yaw_only &&
    (!have_odometry_ ||
    odom_stamp_sec_ > now_sec + config_.future_tolerance_sec ||
    now_sec - odom_stamp_sec_ > config_.odom_timeout_sec ||
    std::abs(normalizeAngle(input.yaw_points[0] - odom_yaw_)) >
    config_.active_sensing_yaw_tolerance ||
    std::abs(normalizeAngle(input.yaw_points[1] - odom_yaw_)) >
    config_.active_sensing_max_yaw_excursion))
  {
    // 两点自身的小差值不足以证明实际转角安全；第一点必须绑定新鲜
    // Odometry 的当前 yaw。normalizeAngle 保留 ±pi 跨界的等价语义。
    // 此门位于 identity 去重之后，DDS 幂等重发不会因稍后的传感器超时
    // 反向毒化已接受轨迹；真正执行仍由 update() 的 freshness 门停车。
    error = "主动观测 yaw 起点与目标必须匹配新鲜 Odometry 安全转角";
    invalidateTrajectory();
    return false;
  }
  if (
    input.header_stamp_sec > now_sec + config_.future_tolerance_sec ||
    input.start_stamp_sec > now_sec + config_.future_tolerance_sec ||
    input.reference_path_stamp_sec > now_sec + config_.future_tolerance_sec)
  {
    error = "B-spline 时间戳超出允许的未来时间";
    invalidateTrajectory();
    return false;
  }
  if (now_sec - input.header_stamp_sec > config_.bspline_timeout_sec) {
    error = "收到的 B-spline 已超时";
    invalidateTrajectory();
    return false;
  }
  if (
    std::abs(input.header_stamp_sec - input.start_stamp_sec) >
    config_.max_start_stamp_skew_sec)
  {
    error = "B-spline 发布时间与起始时间偏差过大";
    invalidateTrajectory();
    return false;
  }
  const Eigen::Index duration_end = input.control_points.cols();
  const double duration =
    input.knots(duration_end) - input.knots(input.order);
  if (!finitePositive(duration)) {
    error = "B-spline 有效时长必须为有限正数";
    invalidateTrajectory();
    return false;
  }

  scan_planner::UniformBspline position(
    input.control_points, input.order, 0.1);
  position.setKnot(input.knots);
  scan_planner::UniformBspline velocity = position.getDerivative();
  const bool stationary_final_hold =
    input.is_final && !input.emergency_stop &&
    isStationaryTrajectory(input.control_points);
  if (stationary_final_hold && duration <= kEpsilon) {
    error = "stationary final hold 的稳定等待时长必须大于零";
    invalidateTrajectory();
    return false;
  }

  // 普通滚动重规划可能在航向误差门附近发布同 Path 的新轨迹。只要本次
  // 原地对齐尚未通过释放阈值，新 traj_id 就必须继承既有硬截止，不能用
  // 新 start_time 重新获得一整段冻结预算。
  const bool preserve_terminal_capture =
    terminal_capture_active_ && same_recorded_path_generation &&
    input.is_final && !input.emergency_stop && !stationary_final_hold;
  const bool had_terminal_capture = terminal_capture_active_;
  const bool terminal_capture_was_frozen =
    terminal_capture_active_ &&
    (terminal_translation_brake_active_ || yaw_alignment_active_);
  const bool preserve_terminal_capture_deadline =
    preserve_terminal_capture && terminal_capture_was_frozen;
  const double inherited_terminal_capture_hard_expiry =
    terminal_capture_hard_expiry_sec_;
  const bool inherit_yaw_alignment_deadline =
    yaw_alignment_active_ && !input.emergency_stop &&
    yaw_alignment_reference_path_stamp_ns_ == input_reference_path_stamp_ns &&
    !stationary_final_hold &&
    (!had_terminal_capture || input.is_final);
  const double inherited_yaw_alignment_deadline =
    yaw_alignment_hard_expiry_sec_;
  const bool resume_after_expired_trajectory =
    have_reference_path_progress_ &&
    (!have_trajectory_ ||
    (finitePositive(trajectory_expiry_sec_) &&
    now_sec > trajectory_expiry_sec_));

  position_trajectory_ = position;
  velocity_trajectory_ = velocity;
  trajectory_duration_sec_ = duration;
  // 同点 final B-spline 是终点保持请求。其正时长必须从本控制器实际接收后
  // 连续计量，不能用一个已经过去的 start_time 跳过稳定等待。
  execution_time_sec_ = stationary_final_hold ? 0.0 : std::clamp(
    now_sec - input.start_stamp_sec, 0.0, trajectory_duration_sec_);
  trajectory_stamp_sec_ = input.header_stamp_sec;
  trajectory_header_stamp_ns_ = input_header_stamp_ns;
  trajectory_start_stamp_ns_ = input_start_stamp_ns;
  trajectory_reference_path_stamp_sec_ =
    input.reference_path_stamp_sec;
  trajectory_reference_path_stamp_ns_ = input_reference_path_stamp_ns;
  const double effective_start_sec =
    stationary_final_hold ? now_sec : input.start_stamp_sec;
  trajectory_expiry_sec_ = std::max(
    input.header_stamp_sec + config_.bspline_timeout_sec,
    effective_start_sec + duration +
    config_.trajectory_expiry_grace_sec);
  trajectory_hard_expiry_sec_ =
    trajectory_expiry_sec_ + config_.max_yaw_alignment_freeze_sec;
  const double active_hard_expiry = preserve_terminal_capture_deadline ?
    inherited_terminal_capture_hard_expiry : trajectory_hard_expiry_sec_;
  if (preserve_terminal_capture_deadline) {
    // 仍处于终点制动或原地转向时，新 final 只能刷新几何，不能靠滚动重规划
    // 延长冻结预算。否则持续刷新轨迹可永久绕过同一轮终点冻结的硬截止。
    trajectory_expiry_sec_ = std::min(
      trajectory_expiry_sec_, inherited_terminal_capture_hard_expiry);
  } else if (preserve_terminal_capture) {
    // 已经解除零平移并恢复执行 SCAN 轨迹时，上一轮冻结 episode 已结束。
    // 新 B-spline 必须使用自己的软/硬有效期；只保留 terminal orientation
    // mode，避免在终点外圈内把航向切回局部弦。若再次进入内门，新的冻结
    // episode 会从这条已认证轨迹的有限 hard-expiry 开始，仍然不会无限冻结。
    terminal_capture_hard_expiry_sec_ = trajectory_hard_expiry_sec_;
  }
  if (input.emergency_stop) {
    resetYawAlignmentEpisode();
  } else if (inherit_yaw_alignment_deadline) {
    yaw_alignment_hard_expiry_sec_ = std::min(
      inherited_yaw_alignment_deadline, active_hard_expiry);
  } else if (yaw_alignment_active_) {
    // 防御性处理：不同 Path 的轨迹不应通过前述代际门到达这里；若未来
    // 接口顺序改变，也不能把旧 Path 的对齐状态带入新代际。
    resetYawAlignmentEpisode();
  }
  trajectory_id_ = input.trajectory_id;
  trajectory_order_ = input.order;
  trajectory_yaw_points_ = input.yaw_points;
  trajectory_yaw_dt_ = input.yaw_dt;
  active_sensing_yaw_only_ = active_sensing_yaw_only;
  active_sensing_target_yaw_ = active_sensing_yaw_only ?
    normalizeAngle(input.yaw_points[1]) : 0.0;
  trajectory_is_final_ = input.is_final;
  stationary_final_hold_ = stationary_final_hold;
  if (stationary_final_hold) {
    // stationary final 由自身连续驻留语义认证，不能继承运动 final 的捕获、
    // 制动期限或 terminal yaw 对齐 episode。
    resetTerminalCapture();
    resetYawAlignmentEpisode();
  } else if (had_terminal_capture && !preserve_terminal_capture) {
    // 同 Path final→non-final 或急停必须同时清除终点航向；否则后续普通
    // B-spline 会继续追逐旧任务的 terminal yaw。
    resetTerminalCapture();
    resetYawAlignmentEpisode();
  } else if (!input.is_final || input.emergency_stop) {
    resetTerminalCapture();
  }
  emergency_stop_latched_ = input.emergency_stop;
  trajectory_finished_ = false;
  trajectory_identity_poisoned_ = false;
  have_trajectory_ = true;
  if (resume_after_expired_trajectory) {
    // 楼梯 root-lock 会在 SCAN 轨迹停更期间沿同一 Path 向前推进数米。
    // 旧的局部进度窗只有有限前视，恢复后若仍沿用它，会把上层机器人投影
    // 回楼梯前段并要求反向转头。下一次投影只放宽向前上界，向后界仍受
    // backward_arc 约束；这样既能重定位到当前楼层，又不会倒退跳段。
    reference_path_forward_reacquisition_pending_ = true;
    cross_track_alignment_active_ = false;
    resetYawAlignmentEpisode();
  }
  if (active_sensing_yaw_only_) {
    // 主动观测请求切换到严格原地转向，接受当拍即丢弃上一条平移或横摆命令
    // 历史，确保后续任何门控停车与首拍输出都不会泄漏旧轨迹速度。
    previous_vx_ = 0.0;
    previous_vy_ = 0.0;
    previous_wz_ = 0.0;
    resetTerminalCapture();
    resetYawAlignmentEpisode();
  }
  last_update_sec_ = now_sec;
  return true;
}

bool TrajectoryTracker::setOdometry(
  const OdometryInput & input, double now_sec, std::string & error)
{
  error.clear();
  if (
    !finitePositive(now_sec) ||
    !input.position.allFinite() ||
    !std::isfinite(input.yaw) ||
    !std::isfinite(input.planar_speed) ||
    input.planar_speed < 0.0 ||
    !std::isfinite(input.vertical_speed) ||
    input.vertical_speed < 0.0 ||
    !std::isfinite(input.angular_speed) ||
    input.angular_speed < 0.0 ||
    !std::isfinite(input.yaw_rate) ||
    input.yaw_rate < 0.0 ||
    !finitePositive(input.stamp_sec))
  {
    error = "Odometry 数值或时间戳非法";
    invalidateOdometry();
    return false;
  }
  if (input.stamp_sec > now_sec + config_.future_tolerance_sec) {
    error = "Odometry 时间戳超出允许的未来时间";
    invalidateOdometry();
    return false;
  }
  if (now_sec - input.stamp_sec > config_.odom_timeout_sec) {
    error = "收到的 Odometry 已超时";
    invalidateOdometry();
    return false;
  }

  odom_position_ = input.position;
  odom_yaw_ = normalizeAngle(input.yaw);
  odom_planar_speed_ = input.planar_speed;
  odom_vertical_speed_ = input.vertical_speed;
  odom_angular_speed_ = input.angular_speed;
  odom_yaw_rate_ = input.yaw_rate;
  odom_stamp_sec_ = input.stamp_sec;
  have_odometry_ = true;
  return true;
}

bool TrajectoryTracker::setReferencePath(
  const std::vector<Eigen::Vector3d> & points,
  double stamp_sec, double now_sec, std::string & error,
  std::optional<double> terminal_yaw, std::int64_t stamp_ns)
{
  error.clear();
  if (
    !finitePositive(now_sec) || !finitePositive(stamp_sec) ||
    stamp_sec > now_sec + config_.future_tolerance_sec)
  {
    error = "参考 Path 时间戳非法或超出允许的未来时间";
    return false;
  }
  const std::int64_t resolved_stamp_ns =
    stamp_ns > 0 ? stamp_ns : secondsToNanoseconds(stamp_sec);
  if (resolved_stamp_ns <= 0) {
    error = "参考 Path 纳秒代际非法";
    return false;
  }
  if (
    invalidated_reference_path_stamp_ns_ > 0 &&
    resolved_stamp_ns <= invalidated_reference_path_stamp_ns_)
  {
    error = "参考 Path 代际已被清除或因协议冲突作废";
    return false;
  }
  if (
    have_reference_path_ &&
    resolved_stamp_ns < reference_path_stamp_ns_)
  {
    error = "参考 Path 早于当前已接受代际";
    return false;
  }
  if (points.size() < 2U) {
    error = "参考 Path 至少需要两个点";
    return false;
  }
  if (
    points.size() >
    static_cast<std::size_t>(config_.reference_path_max_points))
  {
    error = "参考 Path 超过控制器线性处理点数上限";
    return false;
  }

  std::vector<Eigen::Vector3d> accepted;
  accepted.reserve(points.size());
  for (const auto & point : points) {
    if (!point.allFinite()) {
      error = "参考 Path 含非有限坐标";
      return false;
    }
    if (
      !accepted.empty() &&
      (point - accepted.back()).norm() <
      config_.reference_path_min_point_spacing)
    {
      continue;
    }
    accepted.push_back(point);
  }
  if (accepted.size() < 2U) {
    error = "参考 Path 按最小三维点间距去重后少于两个点";
    return false;
  }

  bool have_planar_segment = false;
  double initial_path_tangent_yaw = 0.0;
  double total_length = 0.0;
  std::vector<double> segment_start_progress;
  std::vector<double> segment_length;
  segment_start_progress.reserve(accepted.size() - 1U);
  segment_length.reserve(accepted.size() - 1U);
  for (std::size_t index = 1; index < accepted.size(); ++index) {
    const Eigen::Vector3d delta = accepted[index] - accepted[index - 1U];
    const double length = delta.norm();
    segment_start_progress.push_back(total_length);
    segment_length.push_back(length);
    total_length += length;
    if (delta.head<2>().squaredNorm() > kEpsilon) {
      if (!have_planar_segment) {
        initial_path_tangent_yaw = std::atan2(delta.y(), delta.x());
      }
      have_planar_segment = true;
    }
  }
  if (!have_planar_segment || total_length <= kEpsilon) {
    error = "参考 Path 必须包含非退化三维路径和非零 XY 线段";
    return false;
  }

  // 终点认证使用上游完整 Path 的原始最后点；即使最后一点因局部投影的
  // 最小间距规则被去重，也不能把倒数第二点误当作全局目标。
  const Eigen::Vector3d raw_goal_ground = points.back();
  double terminal_path_tangent_yaw = 0.0;
  bool have_terminal_planar_segment = false;
  for (std::size_t index = points.size() - 1U; index > 0U; --index) {
    const Eigen::Vector3d delta = points[index] - points[index - 1U];
    if (delta.head<2>().squaredNorm() > kEpsilon) {
      terminal_path_tangent_yaw = std::atan2(delta.y(), delta.x());
      have_terminal_planar_segment = true;
      break;
    }
  }
  if (!have_terminal_planar_segment) {
    error = "参考 Path 缺少可用于终点认证的非退化 XY 线段";
    return false;
  }
  if (terminal_yaw.has_value() && !std::isfinite(*terminal_yaw)) {
    error = "参考 Path 的 terminal yaw 必须为有限值";
    return false;
  }
  const double resolved_terminal_yaw = normalizeAngle(
    terminal_yaw.value_or(terminal_path_tangent_yaw));

  bool geometry_unchanged =
    have_reference_path_ && reference_path_.size() == accepted.size() &&
    (reference_path_goal_ground_ - raw_goal_ground).norm() <= kEpsilon &&
    std::abs(
    normalizeAngle(
      reference_path_terminal_yaw_ - resolved_terminal_yaw)) <= kEpsilon;
  if (geometry_unchanged) {
    for (std::size_t index = 0; index < accepted.size(); ++index) {
      if ((reference_path_[index] - accepted[index]).norm() > kEpsilon) {
        geometry_unchanged = false;
        break;
      }
    }
  }
  if (
    geometry_unchanged && have_reference_path_ &&
    resolved_stamp_ns == reference_path_stamp_ns_)
  {
    // 只有同 stamp、同 payload 才是幂等 DDS 重发；更新 stamp 始终是新代际。
    return true;
  }

  if (
    have_reference_path_ &&
    resolved_stamp_ns == reference_path_stamp_ns_)
  {
    // reference_path_stamp 是现有消息唯一的代际标识。同一 stamp 对应两套
    // 几何或终端 yaw 时，迟到的旧 B-spline 无法与新轨迹区分；必须把该代际
    // 整体作废并等待严格更新的 Path stamp，不能猜测哪一条消息更新。
    error = "同一参考 Path 代际出现不同几何或 terminal yaw";
    invalidateTrajectory();
    invalidateReferencePath(resolved_stamp_ns);
    clearCompletionLatch();
    return false;
  }

  // 已有 Path 的几何或终端朝向一旦变化，旧局部轨迹必须当回调立即失效；
  // future_tolerance 只用于判断时钟，不是 Path 代际宽限。首条 Path 晚于同代
  // B-spline 到达时仅允许原始纳秒 stamp 完全一致。
  if (
    have_reference_path_ ||
    (have_trajectory_ &&
    resolved_stamp_ns != trajectory_reference_path_stamp_ns_))
  {
    invalidateTrajectory();
  }
  reference_path_ = std::move(accepted);
  reference_segment_start_progress_ = std::move(segment_start_progress);
  reference_segment_length_ = std::move(segment_length);
  reference_path_goal_ground_ = raw_goal_ground;
  reference_path_stamp_sec_ = stamp_sec;
  reference_path_stamp_ns_ = resolved_stamp_ns;
  reference_path_total_length_ = total_length;
  reference_path_progress_ = 0.0;
  reference_path_tangent_yaw_ = initial_path_tangent_yaw;
  reference_path_terminal_yaw_ = resolved_terminal_yaw;
  reference_path_correction_world_.setZero();
  have_reference_path_progress_ = false;
  reference_path_forward_reacquisition_pending_ = false;
  cross_track_alignment_active_ = false;
  resetTerminalCapture();
  resetYawAlignmentEpisode();
  emergency_stop_latched_ = false;
  clearCompletionLatch();
  have_reference_path_ = true;
  return true;
}

bool TrajectoryTracker::setCloudObservation(
  double stamp_sec, double now_sec, std::string & error)
{
  error.clear();
  if (!finitePositive(now_sec) || !finitePositive(stamp_sec)) {
    error = "PointCloud2 时间戳非法";
    invalidateCloud();
    return false;
  }
  if (stamp_sec > now_sec + config_.future_tolerance_sec) {
    error = "PointCloud2 时间戳超出允许的未来时间";
    invalidateCloud();
    return false;
  }
  if (now_sec - stamp_sec > config_.cloud_timeout_sec) {
    error = "收到的 PointCloud2 已超时";
    invalidateCloud();
    return false;
  }
  cloud_stamp_sec_ = stamp_sec;
  have_cloud_ = true;
  return true;
}

void TrajectoryTracker::invalidateTrajectory()
{
  const bool had_terminal_capture = terminal_capture_active_;
  have_trajectory_ = false;
  stationary_final_hold_ = false;
  active_sensing_yaw_only_ = false;
  resetTerminalCapture();
  if (had_terminal_capture) {
    resetYawAlignmentEpisode();
  }
  // 已发布的完成事件只由下一条有效轨迹解除，拒绝输入不能制造瞬时 false。
  if (!trajectory_finished_) {
    trajectory_is_final_ = false;
  }
  execution_time_sec_ = 0.0;
}

void TrajectoryTracker::invalidateReferencePath(
  std::int64_t invalidated_through_stamp_ns)
{
  if (invalidated_through_stamp_ns > 0) {
    invalidated_reference_path_stamp_ns_ = std::max(
      invalidated_reference_path_stamp_ns_, invalidated_through_stamp_ns);
  }
  have_reference_path_ = false;
  reference_path_.clear();
  reference_segment_start_progress_.clear();
  reference_segment_length_.clear();
  reference_path_goal_ground_.setZero();
  reference_path_stamp_sec_ = 0.0;
  reference_path_stamp_ns_ = 0;
  reference_path_total_length_ = 0.0;
  reference_path_progress_ = 0.0;
  reference_path_tangent_yaw_ = 0.0;
  reference_path_terminal_yaw_ = 0.0;
  reference_path_correction_world_.setZero();
  have_reference_path_progress_ = false;
  reference_path_forward_reacquisition_pending_ = false;
  cross_track_alignment_active_ = false;
  resetTerminalCapture();
  resetYawAlignmentEpisode();
}

void TrajectoryTracker::invalidateOdometry()
{
  have_odometry_ = false;
  resetTerminalCaptureStability();
}

void TrajectoryTracker::invalidateCloud()
{
  have_cloud_ = false;
  resetTerminalCaptureStability();
}

void TrajectoryTracker::clearCompletionLatch()
{
  const bool had_terminal_capture = terminal_capture_active_;
  trajectory_finished_ = false;
  // 首条 Path 可能与同代 B-spline 跨 topic 乱序到达；若匹配轨迹仍有效，
  // 只清旧完成上升沿，不能抹掉这条轨迹自己的 final 语义。
  if (!have_trajectory_) {
    trajectory_is_final_ = false;
  }
  resetStationaryFinalHoldDwell();
  resetTerminalCapture();
  if (had_terminal_capture) {
    resetYawAlignmentEpisode();
  }
}

void TrajectoryTracker::resetStationaryFinalHoldDwell()
{
  if (stationary_final_hold_ && !trajectory_finished_) {
    execution_time_sec_ = 0.0;
  }
}

void TrajectoryTracker::resetTerminalCaptureStability()
{
  terminal_capture_stable_duration_sec_ = 0.0;
}

void TrajectoryTracker::resetTerminalCapture()
{
  terminal_capture_active_ = false;
  terminal_translation_brake_active_ = false;
  terminal_capture_brake_until_sec_ = 0.0;
  terminal_capture_hard_expiry_sec_ = 0.0;
  resetTerminalCaptureStability();
}

void TrajectoryTracker::resetYawAlignmentEpisode()
{
  yaw_alignment_active_ = false;
  yaw_alignment_reference_path_stamp_ns_ = 0;
  yaw_alignment_hard_expiry_sec_ = 0.0;
  yaw_alignment_target_yaw_ = 0.0;
}

double TrajectoryTracker::normalizeAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double TrajectoryTracker::limitRate(
  double previous, double target, double max_rate, double dt)
{
  const double maximum_delta = max_rate * std::max(0.0, dt);
  return previous +
         std::clamp(target - previous, -maximum_delta, maximum_delta);
}

ControlOutput TrajectoryTracker::stop(
  ControllerState state, double now_sec, bool execution_frozen)
{
  previous_vx_ = 0.0;
  previous_vy_ = 0.0;
  previous_wz_ = 0.0;
  if (finitePositive(now_sec)) {
    last_update_sec_ = now_sec;
  }
  ControlOutput output;
  output.execution_frozen = execution_frozen;
  output.trajectory_finished = trajectory_finished_;
  output.goal_reached = trajectory_finished_ && trajectory_is_final_;
  output.state = state;
  return output;
}

ControlOutput TrajectoryTracker::immediateStop(double now_sec)
{
  ControllerState state = ControllerState::kTracking;
  if (trajectory_finished_) {
    state = trajectory_is_final_ ? ControllerState::kGoalReached :
      ControllerState::kTrajectoryFinished;
  } else if (!have_trajectory_) {
    state = ControllerState::kWaitingForTrajectory;
  } else if (!have_reference_path_) {
    state = ControllerState::kWaitingForReferencePath;
  } else if (!have_odometry_) {
    state = ControllerState::kWaitingForOdometry;
  } else if (!have_cloud_) {
    state = ControllerState::kWaitingForCloud;
  }
  if (emergency_stop_latched_) {
    state = ControllerState::kEmergencyStop;
  }
  return stop(state, now_sec, true);
}

double TrajectoryTracker::referencePathCrossTrackError()
{
  if (!have_reference_path_ || reference_path_.size() < 2U) {
    return 0.0;
  }

  Eigen::Vector3d query = odom_position_;
  query.z() -= config_.reference_path_body_height;
  const double previous_progress = reference_path_progress_;
  const double search_lower = have_reference_path_progress_ ?
    std::max(
    0.0,
    previous_progress - config_.reference_path_backward_arc) :
    0.0;
  const double search_upper =
    have_reference_path_progress_ &&
    !reference_path_forward_reacquisition_pending_ ?
    std::min(
    reference_path_total_length_,
    previous_progress + config_.reference_path_forward_arc) :
    reference_path_total_length_;

  double best_distance_squared = std::numeric_limits<double>::infinity();
  double best_progress = 0.0;
  Eigen::Vector3d best_projection = Eigen::Vector3d::Zero();
  double best_tangent_yaw = reference_path_tangent_yaw_;
  bool best_tangent_valid = false;
  bool have_candidate = false;
  for (std::size_t index = 1; index < reference_path_.size(); ++index) {
    const double length = reference_segment_length_[index - 1U];
    const double segment_start =
      reference_segment_start_progress_[index - 1U];
    const double segment_end = segment_start + length;
    if (
      length <= kEpsilon ||
      segment_end < search_lower ||
      segment_start > search_upper)
    {
      continue;
    }

    const Eigen::Vector3d & start = reference_path_[index - 1U];
    const Eigen::Vector3d delta = reference_path_[index] - start;
    const double lower_fraction = std::clamp(
      (search_lower - segment_start) / length, 0.0, 1.0);
    const double upper_fraction = std::clamp(
      (search_upper - segment_start) / length, 0.0, 1.0);
    const double fraction = std::clamp(
      (query - start).dot(delta) / (length * length),
      lower_fraction, upper_fraction);
    const Eigen::Vector3d projection = start + fraction * delta;
    const double distance_squared = (query - projection).squaredNorm();
    const double candidate_progress = segment_start + fraction * length;

    bool select = !have_candidate ||
      distance_squared < best_distance_squared - 1.0e-12;
    const double tie_tolerance =
      1.0e-12 + 1.0e-10 * std::max(
      std::abs(distance_squared), std::abs(best_distance_squared));
    if (
      have_candidate &&
      std::abs(distance_squared - best_distance_squared) <= tie_tolerance)
    {
      select = have_reference_path_progress_ ?
        std::abs(candidate_progress - previous_progress) <
        std::abs(best_progress - previous_progress) :
        candidate_progress < best_progress;
    }
    if (select) {
      have_candidate = true;
      best_distance_squared = distance_squared;
      best_progress = candidate_progress;
      best_projection = projection;
      if (delta.head<2>().squaredNorm() > kEpsilon) {
        best_tangent_yaw = std::atan2(delta.y(), delta.x());
        best_tangent_valid = true;
      } else {
        best_tangent_valid = false;
      }
    }
  }
  if (!have_candidate) {
    return std::numeric_limits<double>::infinity();
  }

  reference_path_progress_ = have_reference_path_progress_ ?
    std::max(previous_progress, best_progress) :
    best_progress;
  const Eigen::Vector2d projection_correction =
    (best_projection - query).head<2>();
  if (best_tangent_valid) {
    reference_path_tangent_yaw_ = best_tangent_yaw;
    const Eigen::Vector2d path_normal(
      -std::sin(best_tangent_yaw),
      std::cos(best_tangent_yaw));
    reference_path_correction_world_ =
      projection_correction.dot(path_normal) * path_normal;
  } else {
    reference_path_correction_world_ = projection_correction;
  }
  have_reference_path_progress_ = true;
  reference_path_forward_reacquisition_pending_ = false;
  return reference_path_correction_world_.norm();
}

std::optional<double> TrajectoryTracker::referencePathStairHeading() const
{
  if (
    !config_.stair_heading_lock_enabled || !have_reference_path_ ||
    !have_reference_path_progress_ || reference_path_.size() < 2U)
  {
    return std::nullopt;
  }

  const double search_lower = std::max(
    0.0,
    reference_path_progress_ - config_.stair_heading_lock_half_window_arc);
  const double search_upper = std::min(
    reference_path_total_length_,
    reference_path_progress_ + config_.stair_heading_lock_half_window_arc);
  Eigen::Vector2d horizontal_direction = Eigen::Vector2d::Zero();
  bool steep_segment_present = false;

  for (std::size_t index = 1U; index < reference_path_.size(); ++index) {
    const double length = reference_segment_length_[index - 1U];
    const double segment_start =
      reference_segment_start_progress_[index - 1U];
    const double segment_end = segment_start + length;
    const double overlap_start = std::max(search_lower, segment_start);
    const double overlap_end = std::min(search_upper, segment_end);
    const double overlap = overlap_end - overlap_start;
    if (length <= kEpsilon || overlap <= kEpsilon) {
      continue;
    }

    const Eigen::Vector3d delta =
      reference_path_[index] - reference_path_[index - 1U];
    const double planar_length = delta.head<2>().norm();
    const double pitch = std::atan2(std::abs(delta.z()), planar_length);
    if (pitch + kEpsilon >= config_.stair_heading_lock_min_pitch_rad) {
      steep_segment_present = true;
    }
    // 按窗口内实际重叠比例累积水平位移，相当于用完整 Path 在窗口两端
    // 的位移求切向。离散台阶的近竖直段不会放大毫米级 XY 噪声，前后
    // 踏面则共同给出稳定的上楼方向。
    horizontal_direction += (overlap / length) * delta.head<2>();
  }

  if (
    !steep_segment_present ||
    horizontal_direction.squaredNorm() <= kEpsilon)
  {
    return std::nullopt;
  }
  return std::atan2(horizontal_direction.y(), horizontal_direction.x());
}

Eigen::Vector2d TrajectoryTracker::applyStairForwardSpeedFloor(
  const Eigen::Vector2d & velocity_world,
  const std::optional<double> & stair_heading,
  const bool terminal_approach_active) const
{
  if (
    config_.stair_forward_speed_floor <= kEpsilon ||
    !stair_heading.has_value() || cross_track_alignment_active_ ||
    terminal_approach_active)
  {
    return velocity_world;
  }

  const Eigen::Vector2d stair_tangent(
    std::cos(*stair_heading), std::sin(*stair_heading));
  const double planned_forward_speed = velocity_world.dot(stair_tangent);
  if (
    planned_forward_speed <= kEpsilon ||
    planned_forward_speed >= config_.stair_forward_speed_floor)
  {
    // 零速或反向分量可能来自制动、避障或轨迹回退，绝不能被牵引逻辑覆盖。
    return velocity_world;
  }

  // 只补完整 Path 切向分量。SCAN 的世界系横向避障与回线分量保持不变，
  // 后续仍经过机体系 vx/vy 硬限幅和加速度门。
  return velocity_world +
         (config_.stair_forward_speed_floor - planned_forward_speed) *
         stair_tangent;
}

double TrajectoryTracker::estimateTrackingDesiredYaw(
  double trajectory_time,
  const Eigen::Vector3d & tracking_position) const
{
  const double lookahead_time = std::min(
    trajectory_duration_sec_, trajectory_time + config_.time_forward);
  Eigen::Vector3d direction =
    position_trajectory_.evaluateDeBoorT(lookahead_time) -
    tracking_position;
  if (direction.head<2>().squaredNorm() < 1.0e-6) {
    direction = velocity_trajectory_.evaluateDeBoorT(trajectory_time);
  }
  if (direction.head<2>().squaredNorm() < 1.0e-6) {
    return odom_yaw_;
  }
  return std::atan2(direction.y(), direction.x());
}

std::optional<double> TrajectoryTracker::estimateTrajectoryDesiredYaw(
  double trajectory_time) const
{
  const double lookahead_time = std::min(
    trajectory_duration_sec_, trajectory_time + config_.time_forward);
  // 航向来自 B-spline 自身的有向弦，而不是“前视点 - 实际机体位置”。
  // 原地对齐会冻结 trajectory_time，但四足底盘仍有小幅平移漂移；若用
  // 实际位置计算方位，机器人绕过近距离前视点时目标角会持续旋转，形成
  // 追逐移动方位直至硬超时。几何弦在冻结期间保持不变，位置误差仍由
  // 下方闭环速度与完整 Path 横向恢复独立处理。
  Eigen::Vector3d direction =
    position_trajectory_.evaluateDeBoorT(lookahead_time) -
    position_trajectory_.evaluateDeBoorT(trajectory_time);
  const double minimum_chord_squared =
    config_.yaw_alignment_min_chord_distance *
    config_.yaw_alignment_min_chord_distance;
  if (direction.head<2>().squaredNorm() < minimum_chord_squared) {
    // SCAN 滚动轨迹的边界速度可能接近零，短弦会因毫米级优化抖动瞬时
    // 指向后方。此时航向没有物理可信度，不能锁存后冻结执行时间；也不能
    // 回退到同样接近零的瞬时导数。保持普通跟踪，让轨迹先推进出低速段。
    return std::nullopt;
  }
  return std::atan2(direction.y(), direction.x());
}

bool TrajectoryTracker::finalReferencePositionWithin(
  double distance_xy_limit, double distance_z_limit) const
{
  if (!trajectory_is_final_ || !have_reference_path_ ||
    reference_path_.empty() || !have_odometry_ ||
    !finitePositive(distance_xy_limit) ||
    !finitePositive(distance_z_limit))
  {
    return false;
  }
  Eigen::Vector3d final_position = reference_path_goal_ground_;
  final_position.z() += config_.reference_path_body_height;
  return
    (final_position.head<2>() - odom_position_.head<2>()).norm() <=
    distance_xy_limit &&
    std::abs(final_position.z() - odom_position_.z()) <=
    distance_z_limit;
}

Eigen::Vector2d TrajectoryTracker::terminalPositionHoldTargetBody(
  double yaw_error) const
{
  if (
    !terminal_capture_active_ || !terminal_translation_brake_active_ ||
    !trajectory_is_final_ || !have_reference_path_ ||
    reference_path_.empty() || !have_odometry_ ||
    !std::isfinite(yaw_error))
  {
    return Eigen::Vector2d::Zero();
  }

  // 只有仍在收敛 terminal yaw，或真实机体已经漂出零速保持门时，才启用
  // 小幅位置保持。不能只以 8cm 完成外门为停止条件：phase279 seed2 在
  // 约 7.8cm 边缘反复回弹，导致 final B-spline 与零速制动循环。先把
  // 机体拉进 7.5cm 保持门，yaw 与位置均留出余量后再返回严格零命令；
  // 5.5cm 进入门与 7.5cm 保持门之间的迟滞吸收站立摆动，后续物理速度门
  // 和连续驻留仍会如实认证 GOAL_REACHED。
  if (
    std::abs(yaw_error) <= config_.terminal_yaw_control_deadband &&
    finalReferencePositionWithin(
      config_.terminal_capture_zero_hold_distance_xy,
      config_.finish_distance_z))
  {
    return Eigen::Vector2d::Zero();
  }

  Eigen::Vector3d final_position = reference_path_goal_ground_;
  final_position.z() += config_.reference_path_body_height;
  Eigen::Vector2d target_world =
    config_.terminal_position_hold_gain *
    (final_position.head<2>() - odom_position_.head<2>());
  const double target_norm = target_world.norm();
  if (target_norm > config_.terminal_position_hold_max_speed) {
    target_world *= config_.terminal_position_hold_max_speed / target_norm;
  }

  const double cosine = std::cos(odom_yaw_);
  const double sine = std::sin(odom_yaw_);
  return Eigen::Vector2d(
    cosine * target_world.x() + sine * target_world.y(),
    -sine * target_world.x() + cosine * target_world.y());
}

bool TrajectoryTracker::finalReferencePositionSatisfied() const
{
  return finalReferencePositionWithin(
    config_.finish_distance_xy, config_.finish_distance_z);
}

bool TrajectoryTracker::terminalAttitudeAndMotionSatisfied() const
{
  if (
    !trajectory_is_final_ || !have_reference_path_ ||
    reference_path_.empty() || !have_odometry_)
  {
    return false;
  }
  const double finish_yaw_error = std::abs(
    normalizeAngle(
      reference_path_terminal_yaw_ - odom_yaw_));
  // stationary final 与 moving final 的终点捕获阶段都只允许控制机体系 yaw。
  // 四足站立策略会持续产生无法由 cmd_vel 消除的 roll/pitch 微摆；进入捕获
  // 内门并锁住平移后继续使用三轴角速度范数，会让真实静止永远无法累计驻留。
  // 尚未进入终点捕获的运动轨迹仍保留严格三轴角速度门，避免运动中提前完成。
  const bool yaw_only_terminal_hold =
    stationary_final_hold_ || terminal_capture_active_;
  const bool terminal_rotation_is_stable = yaw_only_terminal_hold ?
    odom_yaw_rate_ <= config_.finish_yaw_rate :
    odom_angular_speed_ <= config_.finish_angular_speed;
  return
    finish_yaw_error <= config_.finish_yaw_error &&
    odom_planar_speed_ <= config_.finish_speed &&
    odom_vertical_speed_ <= config_.finish_vertical_speed &&
    terminal_rotation_is_stable;
}

bool TrajectoryTracker::physicalCompletionSatisfied() const
{
  if (!have_trajectory_ || !have_odometry_) {
    return false;
  }
  if (trajectory_is_final_) {
    // final 的位置、高度、末端姿态和真实速度均绑定完整同代 Path；未捕获的
    // moving final 使用三轴角速度，捕获后与 stationary hold 都使用 |wz|。
    return
      finalReferencePositionSatisfied() &&
      terminalAttitudeAndMotionSatisfied();
  }
  Eigen::Vector3d final_position;
  double final_yaw = 0.0;
  bool have_terminal_yaw = false;
  final_position =
    position_trajectory_.evaluateDeBoorT(trajectory_duration_sec_);
  const double terminal_lookback = std::min(
    trajectory_duration_sec_, config_.time_forward);
  const Eigen::Vector3d terminal_direction =
    final_position - position_trajectory_.evaluateDeBoorT(
    trajectory_duration_sec_ - terminal_lookback);
  if (terminal_direction.head<2>().squaredNorm() > 1.0e-6) {
    final_yaw = std::atan2(terminal_direction.y(), terminal_direction.x());
    have_terminal_yaw = true;
  }
  const double finish_xy = (
    final_position.head<2>() - odom_position_.head<2>()).norm();
  const double finish_z =
    std::abs(final_position.z() - odom_position_.z());
  const double finish_yaw_error = have_terminal_yaw ?
    std::abs(
    normalizeAngle(
      final_yaw - odom_yaw_)) :
    0.0;
  return
    finish_xy <= config_.finish_distance_xy &&
    finish_z <= config_.finish_distance_z &&
    finish_yaw_error <= config_.finish_yaw_error &&
    odom_planar_speed_ <= config_.finish_speed &&
    odom_vertical_speed_ <= config_.finish_vertical_speed &&
    odom_angular_speed_ <= config_.finish_angular_speed;
}

bool TrajectoryTracker::completionSatisfied() const
{
  return
    execution_time_sec_ >= trajectory_duration_sec_ - kEpsilon &&
    physicalCompletionSatisfied();
}

ControlOutput TrajectoryTracker::update(double now_sec)
{
  if (!finitePositive(now_sec)) {
    resetStationaryFinalHoldDwell();
    const bool had_terminal_capture = terminal_capture_active_;
    resetTerminalCapture();
    if (had_terminal_capture) {
      resetYawAlignmentEpisode();
    }
    return stop(ControllerState::kInvalidClock, now_sec);
  }
  if (
    last_seen_clock_sec_ > 0.0 &&
    now_sec + config_.future_tolerance_sec < last_seen_clock_sec_)
  {
    invalidateTrajectory();
    emergency_stop_latched_ = false;
    trajectory_finished_ = false;
    trajectory_is_final_ = false;
    invalidateReferencePath();
    // ROS 时钟回拨代表新 epoch；旧 epoch 的纳秒代际上界不能带入新图。
    invalidated_reference_path_stamp_ns_ = 0;
    trajectory_header_stamp_ns_ = 0;
    trajectory_start_stamp_ns_ = 0;
    trajectory_reference_path_stamp_ns_ = 0;
    trajectory_id_ = 0;
    trajectory_order_ = 0;
    trajectory_identity_poisoned_ = false;
    invalidateOdometry();
    invalidateCloud();
    last_seen_clock_sec_ = now_sec;
    return stop(ControllerState::kInvalidClock, now_sec);
  }
  last_seen_clock_sec_ = now_sec;

  if (emergency_stop_latched_) {
    return stop(ControllerState::kEmergencyStop, now_sec, true);
  }
  if (trajectory_finished_) {
    return stop(
      trajectory_is_final_ ? ControllerState::kGoalReached :
      ControllerState::kTrajectoryFinished,
      now_sec);
  }
  if (!have_trajectory_) {
    return stop(ControllerState::kWaitingForTrajectory, now_sec);
  }
  if (!have_reference_path_) {
    resetStationaryFinalHoldDwell();
    resetTerminalCaptureStability();
    return stop(
      ControllerState::kWaitingForReferencePath, now_sec, true);
  }
  if (!have_odometry_) {
    resetStationaryFinalHoldDwell();
    resetTerminalCaptureStability();
    return stop(ControllerState::kWaitingForOdometry, now_sec, true);
  }
  if (!have_cloud_) {
    resetStationaryFinalHoldDwell();
    resetTerminalCaptureStability();
    return stop(ControllerState::kWaitingForCloud, now_sec, true);
  }
  // stationary final hold 可能在楼梯 root-lock 的分阶段解锁期间提前到达。
  // 姿态或速度门的短暂失配会清零连续驻留计时，因此允许它使用已经配置
  // 且仍然有限的 hard-expiry；普通运动轨迹继续遵守更短的软失效时刻。
  const bool terminal_freeze_active =
    terminal_capture_active_ &&
    (terminal_translation_brake_active_ || yaw_alignment_active_);
  const double active_expiry_sec = stationary_final_hold_ ?
    trajectory_hard_expiry_sec_ :
    (terminal_freeze_active ?
    terminal_capture_hard_expiry_sec_ : trajectory_expiry_sec_);
  if (
    trajectory_stamp_sec_ > now_sec + config_.future_tolerance_sec ||
    now_sec > active_expiry_sec)
  {
    // 超时仍发布严格零速，但不能继续要求 planner 顺延已失效轨迹的 start_time；
    // execution_frozen=false 是显式重规划请求，避免 terminal yaw 超过硬截止后
    // planner 永久冻结在旧 final 段。
    const bool had_terminal_capture = terminal_capture_active_;
    resetTerminalCapture();
    if (had_terminal_capture) {
      resetYawAlignmentEpisode();
    }
    return stop(ControllerState::kTrajectoryTimeout, now_sec, false);
  }
  if (
    odom_stamp_sec_ > now_sec + config_.future_tolerance_sec ||
    now_sec - odom_stamp_sec_ > config_.odom_timeout_sec)
  {
    resetStationaryFinalHoldDwell();
    resetTerminalCaptureStability();
    return stop(ControllerState::kOdometryTimeout, now_sec, true);
  }
  const bool cloud_stamp_is_future =
    cloud_stamp_sec_ > now_sec + config_.future_tolerance_sec;
  const bool cloud_has_timed_out =
    now_sec - cloud_stamp_sec_ > config_.cloud_timeout_sec;
  if (cloud_stamp_is_future || cloud_has_timed_out) {
    // 无论是运动 final 还是楼梯的 stationary final hold，都不能用
    // 历史点云认证目标完成。新鲜 Odometry 只能证明机体状态，无法证明
    // 当前局部环境仍无动态障碍；因此点云未来或超时时必须 fail-closed。
    // stationary final hold 还会清零连续驻留时间，恢复新鲜点云后重新计量。
    resetStationaryFinalHoldDwell();
    resetTerminalCaptureStability();
    return stop(ControllerState::kCloudTimeout, now_sec, true);
  }

  const double dt =
    last_update_sec_ > 0.0 ? now_sec - last_update_sec_ : 0.0;
  if (dt < 0.0 || dt > config_.max_control_dt_sec) {
    resetStationaryFinalHoldDwell();
    if (dt < 0.0) {
      const bool had_terminal_capture = terminal_capture_active_;
      resetTerminalCapture();
      if (had_terminal_capture) {
        resetYawAlignmentEpisode();
      }
    } else {
      resetTerminalCaptureStability();
    }
    return stop(ControllerState::kInvalidClock, now_sec);
  }

  if (active_sensing_yaw_only_) {
    // 该分支只能位于 Path identity、轨迹/传感器 freshness 与控制时钟门之后；
    // 任一门失败都必须先由上方 stop() 清空三轴速度。主动观测期间不推进
    // 位置 B-spline，也绝不让 non-final 请求进入 TRAJECTORY_FINISHED。
    const double yaw_error = normalizeAngle(
      active_sensing_target_yaw_ - odom_yaw_);
    if (std::abs(yaw_error) <= config_.active_sensing_yaw_tolerance) {
      previous_vx_ = 0.0;
      previous_vy_ = 0.0;
      previous_wz_ = 0.0;
      last_update_sec_ = now_sec;
      ControlOutput output;
      output.execution_frozen = true;
      output.state = ControllerState::kTracking;
      return output;
    }
    const double target_wz = std::clamp(
      config_.kp_yaw * yaw_error,
      -config_.active_sensing_max_yaw_rate,
      config_.active_sensing_max_yaw_rate);
    ControlOutput output;
    output.vx = 0.0;
    output.vy = 0.0;
    output.wz = std::clamp(
      limitRate(previous_wz_, target_wz, config_.max_yaw_acc, dt),
      -config_.active_sensing_max_yaw_rate,
      config_.active_sensing_max_yaw_rate);
    output.execution_frozen = true;
    output.state = ControllerState::kAligningYaw;
    previous_vx_ = 0.0;
    previous_vy_ = 0.0;
    previous_wz_ = output.wz;
    last_update_sec_ = now_sec;
    return output;
  }

  if (stationary_final_hold_) {
    // final hold 从接收后开始，只有完整 Path 终点的位置、末段航向和三类速度
    // 连续满足时才累计正时长；任一门失配都清零重新等待。整个阶段不经过
    // 速度变化率限制，确保不会残留上一条轨迹的非零命令。
    if (physicalCompletionSatisfied()) {
      execution_time_sec_ = std::min(
        trajectory_duration_sec_, execution_time_sec_ + dt);
    } else {
      execution_time_sec_ = 0.0;
    }
    if (completionSatisfied()) {
      trajectory_finished_ = true;
      return stop(ControllerState::kGoalReached, now_sec);
    }
    return stop(ControllerState::kTracking, now_sec, true);
  }

  // 局部 B-spline 每次从真实 Odometry 起步，不能作为侧偏门的长期基准；
  // 否则每次重规划都会把误差重新归零。这里固定测量到完整 /initial_path
  // 有序 XY 线段的最近距离，坡面高度变化不会掩盖水平滑移。
  const double cross_track_error =
    have_reference_path_ ? referencePathCrossTrackError() : 0.0;
  if (cross_track_alignment_active_) {
    if (
      cross_track_error <=
      config_.cross_track_alignment_release_distance)
    {
      cross_track_alignment_active_ = false;
    }
  } else if (
    cross_track_error > config_.cross_track_alignment_distance)
  {
    cross_track_alignment_active_ = true;
  }
  const double active_heading_threshold =
    cross_track_alignment_active_ ?
    config_.cross_track_heading_error_threshold :
    config_.heading_error_threshold;
  const double active_heading_release_threshold =
    cross_track_alignment_active_ ?
    config_.cross_track_heading_error_release_threshold :
    config_.heading_error_release_threshold;

  const double current_time = std::min(
    execution_time_sec_, trajectory_duration_sec_);
  // moving final 首次进入配置的捕获内门时锁存 terminal orientation mode，
  // 物理完成仍使用独立 0.08m 门。0.12m 只负责彻底退出终端航向；在两门之间若姿态与
  // 平面/垂向速度和导航 yaw 已经稳定，只恢复 SCAN B-spline 的 XY 闭环，
  // 不把航向切回局部弦。
  // 名义时间已经结束时，即使机器人落在 7.5~8cm 的窄环带内，也要先进入
  // 严格全零制动并连续验证真实 Odometry；不能因单个低速采样直接报到达。
  // 提前到达仍使用更小的捕获内门，避免正常跟踪过早冻结。
  // 名义轨迹已结束且机体进入严格 8cm 位置门时，不应继续用最低步态速度
  // 穿越完成圈。这里故意只要求位置合格，姿态、真实速度和连续驻留仍由
  // physicalCompletionSatisfied() 与下方稳定计时独立认证。旧代码调用
  // completionSatisfied()，等价于先要求全部完成才允许开始制动，和上方
  // 注释及四足终点捕获的因果顺序相反。
  const bool nominal_final_inside_completion_position =
    trajectory_is_final_ &&
    execution_time_sec_ >= trajectory_duration_sec_ - kEpsilon &&
    finalReferencePositionWithin(
      config_.finish_distance_xy,
      config_.finish_distance_z);
  const bool terminal_capture_inner_position =
    trajectory_is_final_ && finalReferencePositionWithin(
    config_.terminal_capture_entry_distance_xy,
    config_.finish_distance_z);
  const bool terminal_capture_zero_hold_position =
    trajectory_is_final_ && finalReferencePositionWithin(
    config_.terminal_capture_zero_hold_distance_xy,
    config_.finish_distance_z);
  const bool terminal_capture_entry =
    trajectory_is_final_ && (
    terminal_capture_inner_position ||
    nominal_final_inside_completion_position);
  const bool terminal_capture_retained =
    trajectory_is_final_ && finalReferencePositionWithin(
    config_.terminal_capture_release_distance_xy,
    config_.finish_distance_z);
  bool terminal_translation_brake_entered = false;
  if (terminal_capture_active_ && !terminal_capture_retained) {
    resetTerminalCapture();
    // 只有漂出外圈或高度门时才彻底退出 terminal orientation mode。
    resetYawAlignmentEpisode();
  } else if (!terminal_capture_active_ && terminal_capture_entry) {
    terminal_capture_active_ = true;
    terminal_translation_brake_active_ = true;
    terminal_translation_brake_entered = true;
    terminal_capture_brake_until_sec_ =
      now_sec + config_.terminal_capture_brake_hold_sec;
    terminal_capture_hard_expiry_sec_ = trajectory_hard_expiry_sec_;
    resetTerminalCaptureStability();
    // v15 首次进门时仍有 +0.45rad/s 历史。终端捕获必须丢弃三轴历史，
    // 并在下面保持一段可被 50Hz policy writer 可靠采到的严格全零。
    previous_vx_ = 0.0;
    previous_vy_ = 0.0;
    previous_wz_ = 0.0;
    resetYawAlignmentEpisode();
  } else if (
    terminal_capture_active_ && !terminal_translation_brake_active_ &&
    terminal_capture_inner_position)
  {
    // B-spline 位置回收重新进入严格内门后再次制动；首次捕获的硬截止不刷新。
    terminal_translation_brake_active_ = true;
    terminal_translation_brake_entered = true;
    terminal_capture_brake_until_sec_ =
      now_sec + config_.terminal_capture_brake_hold_sec;
    resetTerminalCaptureStability();
    previous_vx_ = 0.0;
    previous_vy_ = 0.0;
    previous_wz_ = 0.0;
  } else if (
    terminal_capture_active_ && terminal_translation_brake_active_ &&
    !terminal_capture_zero_hold_position && terminal_capture_retained &&
    now_sec >= terminal_capture_brake_until_sec_ &&
    terminalAttitudeAndMotionSatisfied())
  {
    // 捕获零速保持门之外已经完成旋转与动力学制动时解除 XY 冻结，由现有
    // 碰撞安全 B-spline 以最低有效步态收回位置；terminal yaw 模式与固定
    // 硬截止继续保留。5.5~7.5cm 的小幅零命令漂移使用迟滞保持，不应仅因
    // 毫米级站立摆动立即恢复步态；超过 7.5cm 才走这里重新回收。
    terminal_translation_brake_active_ = false;
    resetTerminalCaptureStability();
  }
  if (
    !terminal_translation_brake_active_ &&
    now_sec > trajectory_expiry_sec_)
  {
    // 同一拍退出 0.12m 外圈或恢复 XY 后必须立刻重新服从 soft expiry；
    // 不能先用已经过期的 B-spline 泄漏一拍非零平移。
    const bool had_terminal_capture = terminal_capture_active_;
    resetTerminalCapture();
    if (had_terminal_capture) {
      resetYawAlignmentEpisode();
    }
    return stop(ControllerState::kTrajectoryTimeout, now_sec, false);
  }
  const bool terminal_orientation_mode = terminal_capture_active_;
  const bool terminal_brake_hold_active =
    terminal_translation_brake_active_ &&
    (terminal_translation_brake_entered ||
    now_sec < terminal_capture_brake_until_sec_);
  if (terminal_brake_hold_active) {
    trajectory_expiry_sec_ = std::min(
      terminal_capture_hard_expiry_sec_, trajectory_expiry_sec_ + dt);
    const double terminal_yaw_error = normalizeAngle(
      reference_path_terminal_yaw_ - odom_yaw_);
    return stop(
      std::abs(terminal_yaw_error) >
      config_.terminal_yaw_control_deadband ?
      ControllerState::kAligningYaw : ControllerState::kTracking,
      now_sec, true);
  }
  // 超过跨轨门限后以完整 Path 切线为基准，只加入朝路径中心的小幅航向辅助。
  // v14c 的局部前视点会触发坡面大角度斜转，v17 的纯横移又会引起横摆振荡；
  // 有界辅助让前向牵引承担主要稳定作用，同时通过侧向速度完成闭环回正。
  double tracking_desired_yaw = terminal_orientation_mode ?
    reference_path_terminal_yaw_ :
    estimateTrackingDesiredYaw(current_time, odom_position_);
  const std::optional<double> trajectory_alignment_yaw =
    terminal_orientation_mode ?
    std::optional<double>(reference_path_terminal_yaw_) :
    estimateTrajectoryDesiredYaw(current_time);
  double alignment_candidate_yaw =
    trajectory_alignment_yaw.value_or(tracking_desired_yaw);
  bool alignment_candidate_reliable =
    trajectory_alignment_yaw.has_value();
  const std::optional<double> stair_heading = terminal_orientation_mode ?
    std::nullopt : referencePathStairHeading();
  if (stair_heading.has_value()) {
    // SCAN 的滚动轨迹从真实 Odometry 起步，几厘米横向重接会形成接近
    // 90 度的短弦。楼梯附近若把该短弦解释为机体航向，wz 会反复打满；
    // 这里仅用完整 Path 的楼梯切向约束朝向，B-spline 世界系 XY 速度
    // 仍完整保留，横向重接自然在机体系表现为 vy。
    tracking_desired_yaw = *stair_heading;
    alignment_candidate_yaw = *stair_heading;
    alignment_candidate_reliable = true;
    if (yaw_alignment_active_) {
      // 同一 Path 的滚动轨迹可能在进入楼梯窗口前已经锁存局部短弦；楼梯
      // 语义优先更新目标，但保留原 episode 的硬截止，不能刷新冻结预算。
      yaw_alignment_target_yaw_ = *stair_heading;
    }
  } else if (!terminal_orientation_mode && cross_track_alignment_active_) {
    std::optional<double> final_recovery_yaw;
    if (trajectory_is_final_) {
      // final 在终点外圈发生回弹时，完整 Path 的末段切向可能与当前安全
      // B-spline 的回收方向接近垂直。若仍锁住 Path 切向，世界系回收速度会
      // 几乎全部变成机体系 vy，而 Go2-X5 策略对持续横移的响应明显弱于
      // 前进。终点外圈先朝 SCAN 已碰撞验证的局部弦运动；进入捕获内门后，
      // 上方 terminal orientation mode 再严格切换到任务最终 yaw。
      final_recovery_yaw = trajectory_alignment_yaw;
      if (!final_recovery_yaw.has_value() && have_reference_path_) {
        const Eigen::Vector2d goal_direction =
          reference_path_goal_ground_.head<2>() -
          odom_position_.head<2>();
        const double minimum_chord_squared =
          config_.yaw_alignment_min_chord_distance *
          config_.yaw_alignment_min_chord_distance;
        if (goal_direction.squaredNorm() >= minimum_chord_squared) {
          final_recovery_yaw = std::atan2(
            goal_direction.y(), goal_direction.x());
        }
      }
    }
    if (final_recovery_yaw.has_value()) {
      tracking_desired_yaw = normalizeAngle(*final_recovery_yaw);
      alignment_candidate_yaw = tracking_desired_yaw;
      alignment_candidate_reliable = true;
    } else {
      // non-final 横向恢复仍服从完整 Path 切向，避免局部滚动短弦在普通
      // 导航段反复改变机体朝向；final 的局部弦不可靠时也沿用此安全回退。
      const Eigen::Vector2d path_normal(
        -std::sin(reference_path_tangent_yaw_),
        std::cos(reference_path_tangent_yaw_));
      const double signed_correction =
        reference_path_correction_world_.dot(path_normal);
      const double heading_assist = std::clamp(
        config_.cross_track_heading_assist_gain * signed_correction,
        -config_.cross_track_heading_assist_max,
        config_.cross_track_heading_assist_max);
      tracking_desired_yaw = normalizeAngle(
        reference_path_tangent_yaw_ + heading_assist);
      alignment_candidate_yaw = tracking_desired_yaw;
      alignment_candidate_reliable = true;
    }
  }

  if (
    yaw_alignment_active_ &&
    yaw_alignment_reference_path_stamp_ns_ !=
    trajectory_reference_path_stamp_ns_)
  {
    resetYawAlignmentEpisode();
  }
  const double alignment_yaw_error = normalizeAngle(
    (yaw_alignment_active_ ?
    yaw_alignment_target_yaw_ : alignment_candidate_yaw) - odom_yaw_);
  if (yaw_alignment_active_) {
    if (
      std::abs(alignment_yaw_error) <=
      active_heading_release_threshold)
    {
      resetYawAlignmentEpisode();
    }
  } else if (
    alignment_candidate_reliable &&
    std::abs(alignment_yaw_error) > active_heading_threshold)
  {
    yaw_alignment_active_ = true;
    yaw_alignment_reference_path_stamp_ns_ =
      trajectory_reference_path_stamp_ns_;
    yaw_alignment_hard_expiry_sec_ = terminal_orientation_mode ?
      terminal_capture_hard_expiry_sec_ : trajectory_hard_expiry_sec_;
    // 锁存世界系几何航向。episode 内即使 root 因策略惯性发生 XY 漂移，
    // 或同 Path 滚动轨迹到达，也不能重新追逐近距离前视点的移动方位。
    yaw_alignment_target_yaw_ = alignment_candidate_yaw;
  }

  const double desired_yaw = yaw_alignment_active_ ?
    yaw_alignment_target_yaw_ : tracking_desired_yaw;
  const double yaw_error = normalizeAngle(desired_yaw - odom_yaw_);
  const double active_yaw_rate_limit = terminal_orientation_mode ?
    config_.terminal_max_yaw_rate : config_.max_yaw_rate;
  const double active_yaw_acc_limit = terminal_orientation_mode ?
    config_.terminal_max_yaw_acc : config_.max_yaw_acc;
  // moving final 的控制死区严格窄于完成认证门，使策略稳态先收敛到门内
  // 余量再开始连续驻留；stationary final 已在上方严格零速分支返回。
  const double yaw_feedback =
    terminal_orientation_mode &&
    std::abs(yaw_error) <= config_.terminal_yaw_control_deadband ?
    0.0 : config_.kp_yaw * yaw_error;
  const double target_wz = std::clamp(
    yaw_feedback,
    -active_yaw_rate_limit, active_yaw_rate_limit);
  const Eigen::Vector2d terminal_position_hold_target =
    terminalPositionHoldTargetBody(yaw_error);

  if (terminal_translation_brake_active_) {
    const bool terminal_command_is_zero =
      std::abs(target_wz) <= kEpsilon &&
      terminal_position_hold_target.cwiseAbs().maxCoeff() <= kEpsilon &&
      std::abs(previous_vx_) <= kEpsilon &&
      std::abs(previous_vy_) <= kEpsilon &&
      std::abs(previous_wz_) <= kEpsilon;
    if (physicalCompletionSatisfied() && terminal_command_is_zero) {
      terminal_capture_stable_duration_sec_ += dt;
    } else {
      resetTerminalCaptureStability();
    }
    // moving final 无论是在名义结束前还是结束后进入捕获，都必须连续满足
    // 严格位置、姿态、平面/垂向速度、|wz| 和全零命令驻留；一次偶然低速
    // 采样不能认证
    // GOAL_REACHED，否则四足策略随后回弹时上层已经开始 place。
    if (
      terminal_capture_stable_duration_sec_ + kEpsilon >=
      config_.terminal_capture_stable_dwell_sec)
    {
      trajectory_finished_ = true;
      return stop(ControllerState::kGoalReached, now_sec);
    }
  } else {
    resetTerminalCaptureStability();
  }

  if (
    yaw_alignment_active_ &&
    now_sec > yaw_alignment_hard_expiry_sec_)
  {
    // 同 Path 的滚动重规划只能更新几何参考，不能后移已经开始的航向
    // 对齐 episode 截止。超限后解除 planner 时钟冻结并保持严格零速。
    if (terminal_capture_active_) {
      resetTerminalCapture();
      resetYawAlignmentEpisode();
    }
    return stop(ControllerState::kTrajectoryTimeout, now_sec, false);
  }

  if (yaw_alignment_active_) {
    // 航向对齐阶段既冻结 B-spline 执行时间，也必须等量顺延轨迹失效时刻。
    // 否则长转向会吃掉平移阶段的全部执行预算，刚开始行走便超时停车。
    const double yaw_hard_expiry = terminal_orientation_mode ?
      terminal_capture_hard_expiry_sec_ : trajectory_hard_expiry_sec_;
    trajectory_expiry_sec_ = std::min(
      yaw_hard_expiry, trajectory_expiry_sec_ + dt);
    ControlOutput output;
    // execution_frozen 只冻结 B-spline 名义进度。这里的小幅平移不是继续
    // 导航，而是抵消四足策略在纯 yaw 命令下的真实根位姿漂移，使机体
    // 物理上保持原地；首段严格全零制动已由 terminal_brake_hold 返回。
    output.vx = std::clamp(
      limitRate(
        previous_vx_, terminal_position_hold_target.x(),
        config_.max_ax, dt),
      -config_.terminal_position_hold_max_speed,
      config_.terminal_position_hold_max_speed);
    output.vy = std::clamp(
      limitRate(
        previous_vy_, terminal_position_hold_target.y(),
        config_.max_ay, dt),
      -config_.terminal_position_hold_max_speed,
      config_.terminal_position_hold_max_speed);
    output.wz = limitRate(
      previous_wz_, target_wz, active_yaw_acc_limit, dt);
    output.execution_frozen = true;
    output.state = ControllerState::kAligningYaw;
    previous_vx_ = output.vx;
    previous_vy_ = output.vy;
    previous_wz_ = output.wz;
    last_update_sec_ = now_sec;
    return output;
  }

  if (terminal_translation_brake_active_) {
    // final B-spline 的末切线不等于任务 terminal yaw。进入同代完整 Path
    // 终点位置门后，底盘必须先保持零平移并主动原地收敛到末 Pose 四元数；
    // 否则仍在执行的局部轨迹会把机体带出捕获区，而 desired_yaw 也可能
    // 回退为当前 yaw。提前捕获的完成由上方连续物理驻留认证。
    trajectory_expiry_sec_ = std::min(
      terminal_capture_hard_expiry_sec_, trajectory_expiry_sec_ + dt);
    ControlOutput output;
    output.vx = std::clamp(
      limitRate(
        previous_vx_, terminal_position_hold_target.x(),
        config_.max_ax, dt),
      -config_.terminal_position_hold_max_speed,
      config_.terminal_position_hold_max_speed);
    output.vy = std::clamp(
      limitRate(
        previous_vy_, terminal_position_hold_target.y(),
        config_.max_ay, dt),
      -config_.terminal_position_hold_max_speed,
      config_.terminal_position_hold_max_speed);
    output.wz = std::clamp(
      limitRate(previous_wz_, target_wz, active_yaw_acc_limit, dt),
      -active_yaw_rate_limit, active_yaw_rate_limit);
    output.execution_frozen = true;
    output.state =
      std::abs(yaw_error) > config_.terminal_yaw_control_deadband ?
      ControllerState::kAligningYaw : ControllerState::kTracking;
    previous_vx_ = output.vx;
    previous_vy_ = output.vy;
    previous_wz_ = output.wz;
    last_update_sec_ = now_sec;
    return output;
  }

  // 只有真正的原地航向对齐冻结 B-spline 时间。横向恢复仍推进局部轨迹，
  // 让 SCAN 持续滚动重规划，并避免恢复门无限延长过期轨迹。
  execution_time_sec_ = std::min(
    trajectory_duration_sec_, execution_time_sec_ + dt);
  const Eigen::Vector3d desired_position =
    position_trajectory_.evaluateDeBoorT(execution_time_sec_);
  const Eigen::Vector3d desired_velocity =
    velocity_trajectory_.evaluateDeBoorT(execution_time_sec_);

  const Eigen::Vector2d position_error(
    desired_position.x() - odom_position_.x(),
    desired_position.y() - odom_position_.y());
  const Eigen::Vector2d nominal_velocity_world =
    desired_velocity.head<2>() + config_.kp_position * position_error;
  Eigen::Vector2d velocity_world;
  if (cross_track_alignment_active_) {
    // 滚动 B-spline 总是从当前 Odometry 起步，局部位置误差可能被重置为零；
    // 恢复门保留 SCAN 的制动、后退和绕障法向分量，只限制其切向速度并
    // 叠加完整 Path 回正。这样既提供坡面牵引，也不会强制驶回局部障碍。
    const Eigen::Vector2d path_tangent(
      std::cos(reference_path_tangent_yaw_),
      std::sin(reference_path_tangent_yaw_));
    const Eigen::Vector2d path_normal(
      -path_tangent.y(), path_tangent.x());
    const double remaining_arc = std::max(
      0.0, reference_path_total_length_ - reference_path_progress_);
    const double taper = std::clamp(
      remaining_arc / config_.cross_track_recovery_taper_distance,
      0.0, 1.0);
    const double tangent_limit =
      taper * config_.cross_track_recovery_forward_speed;
    const double tangent_speed = std::clamp(
      nominal_velocity_world.dot(path_tangent),
      -tangent_limit, tangent_limit);
    const double nominal_lateral_speed =
      nominal_velocity_world.dot(path_normal);
    double recovery_lateral_speed = std::clamp(
      config_.cross_track_recovery_lateral_gain *
      reference_path_correction_world_.dot(path_normal),
      -config_.cross_track_recovery_lateral_speed,
      config_.cross_track_recovery_lateral_speed);
    if (nominal_lateral_speed * recovery_lateral_speed < 0.0) {
      // SCAN 明确要求向 Path 外侧绕障时，完整 Path 恢复项不能把它拉回障碍。
      // 这里只抑制新增恢复增量，不裁剪 SCAN 原有法向控制权。
      recovery_lateral_speed = 0.0;
    }
    const double lateral_speed = std::clamp(
      nominal_lateral_speed + recovery_lateral_speed,
      -config_.max_vy, config_.max_vy);
    velocity_world =
      tangent_speed * path_tangent + lateral_speed * path_normal;
  } else {
    velocity_world = nominal_velocity_world;
  }
  // Go2-X5 原始策略在约 0.20m/s 以下的短程斜向命令上响应很弱。终点
  // B-spline 又可能为了连续自由支撑把局部端点留在任务终点前数厘米，
  // 比例控制会因此在进入严格捕获门之前渐近到零。phase242 还证明：机体
  // 首次进入捕获门、终点转向产生漂移并解除平移制动后，同一问题会在回收
  // 阶段再次出现。因此只要当前没有执行严格平移制动，就把同代 final、
  // 完整 Path 终点外圈内的碰撞安全 B-spline 命令抬到最低有效步态速度；
  // 方向、加速度门和各轴硬上限保持不变。最低值约束的是二维合成速度；
  // 斜向命令变换到机体系后仍分别服从 vx/vy 硬上限，因此它可以合法高于
  // 较小的单轴 vy 上限，但不能超过两轴上限的合成模长。
  const double terminal_approach_speed = velocity_world.norm();
  const bool terminal_approach_minimum_active =
    trajectory_is_final_ && !terminal_translation_brake_active_ &&
    !finalReferencePositionWithin(
    config_.terminal_capture_entry_distance_xy,
    config_.finish_distance_z) &&
    finalReferencePositionWithin(
    config_.terminal_capture_release_distance_xy,
    config_.finish_distance_z) &&
    terminal_approach_speed > kEpsilon &&
    terminal_approach_speed < config_.terminal_approach_min_speed;
  if (terminal_approach_minimum_active) {
    velocity_world *=
      config_.terminal_approach_min_speed / terminal_approach_speed;
  }
  const bool terminal_approach_active =
    trajectory_is_final_ && finalReferencePositionWithin(
    config_.terminal_capture_release_distance_xy,
    config_.finish_distance_z);
  velocity_world = applyStairForwardSpeedFloor(
    velocity_world, stair_heading, terminal_approach_active);
  if (
    config_.turning_speed_limit_enabled &&
    std::abs(target_wz) >= config_.turning_yaw_rate_threshold)
  {
    // SCAN 的 B-spline 仍是碰撞安全的；这里约束的是携臂四足对曲线命令的
    // 实际跟踪能力。高角速度时若继续使用直线巡航速度，RL policy 会切向
    // 弯内并越过规划轨迹。只缩放世界系平移向量，不改变方向、yaw 命令、
    // 原地对齐门或直线速度；随后仍经过机体系单轴限幅和变化率限制。
    const double planar_speed = velocity_world.norm();
    if (planar_speed > config_.turning_max_planar_speed) {
      velocity_world *= config_.turning_max_planar_speed / planar_speed;
    }
  }
  const double cosine = std::cos(odom_yaw_);
  const double sine = std::sin(odom_yaw_);
  const double target_vx = std::clamp(
    cosine * velocity_world.x() + sine * velocity_world.y(),
    -config_.max_vx, config_.max_vx);
  const double target_vy = std::clamp(
    -sine * velocity_world.x() + cosine * velocity_world.y(),
    -config_.max_vy, config_.max_vy);

  // final 的完成只允许由上方连续稳定驻留分支认证。普通 non-final 局部轨迹
  // 仍按原合同在名义结束且物理门满足时发布 TRAJECTORY_FINISHED。
  if (!trajectory_is_final_ && completionSatisfied()) {
    trajectory_finished_ = true;
    return stop(ControllerState::kTrajectoryFinished, now_sec);
  }

  ControlOutput output;
  output.vx = std::clamp(
    limitRate(previous_vx_, target_vx, config_.max_ax, dt),
    -config_.max_vx, config_.max_vx);
  output.vy = std::clamp(
    limitRate(previous_vy_, target_vy, config_.max_ay, dt),
    -config_.max_vy, config_.max_vy);
  output.wz = std::clamp(
    limitRate(previous_wz_, target_wz, active_yaw_acc_limit, dt),
    -active_yaw_rate_limit, active_yaw_rate_limit);
  output.execution_frozen = false;
  output.state = ControllerState::kTracking;
  previous_vx_ = output.vx;
  previous_vy_ = output.vy;
  previous_wz_ = output.wz;
  last_update_sec_ = now_sec;
  return output;
}

double TrajectoryTracker::executionTimeSec() const
{
  return execution_time_sec_;
}

double TrajectoryTracker::trajectoryDurationSec() const
{
  return trajectory_duration_sec_;
}

bool TrajectoryTracker::hasTrajectory() const
{
  return have_trajectory_;
}

bool TrajectoryTracker::hasReferencePath() const
{
  return have_reference_path_;
}

double TrajectoryTracker::referencePathStampSec() const
{
  return reference_path_stamp_sec_;
}

const char * TrajectoryTracker::stateName(ControllerState state)
{
  switch (state) {
    case ControllerState::kWaitingForTrajectory:
      return "WAITING_FOR_TRAJECTORY";
    case ControllerState::kWaitingForReferencePath:
      return "WAITING_FOR_REFERENCE_PATH";
    case ControllerState::kWaitingForOdometry:
      return "WAITING_FOR_ODOMETRY";
    case ControllerState::kWaitingForCloud:
      return "WAITING_FOR_CLOUD";
    case ControllerState::kTrajectoryTimeout:
      return "TRAJECTORY_TIMEOUT";
    case ControllerState::kOdometryTimeout:
      return "ODOMETRY_TIMEOUT";
    case ControllerState::kCloudTimeout:
      return "CLOUD_TIMEOUT";
    case ControllerState::kInvalidClock:
      return "INVALID_CLOCK";
    case ControllerState::kEmergencyStop:
      return "EMERGENCY_STOP";
    case ControllerState::kAligningYaw:
      return "ALIGNING_YAW";
    case ControllerState::kTracking:
      return "TRACKING";
    case ControllerState::kTrajectoryFinished:
      return "TRAJECTORY_FINISHED";
    case ControllerState::kGoalReached:
      return "GOAL_REACHED";
  }
  return "UNKNOWN";
}

}  // 命名空间 scan_controller
