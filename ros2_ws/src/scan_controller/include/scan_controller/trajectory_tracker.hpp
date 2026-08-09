#ifndef SCAN_CONTROLLER__TRAJECTORY_TRACKER_HPP_
#define SCAN_CONTROLLER__TRAJECTORY_TRACKER_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "bspline_opt/uniform_bspline.h"

namespace scan_controller
{

struct TrackerConfig
{
  double time_forward{0.60};
  double yaw_alignment_min_chord_distance{0.03};
  double heading_error_threshold{0.70};
  double heading_error_release_threshold{0.55};
  double cross_track_alignment_distance{0.12};
  double cross_track_alignment_release_distance{0.08};
  double cross_track_heading_error_threshold{0.20};
  double cross_track_heading_error_release_threshold{0.18};
  double cross_track_recovery_forward_speed{0.18};
  double cross_track_recovery_taper_distance{0.30};
  double cross_track_recovery_lateral_gain{0.45};
  double cross_track_recovery_lateral_speed{0.08};
  double cross_track_heading_assist_gain{1.00};
  double cross_track_heading_assist_max{0.16};
  bool turning_speed_limit_enabled{false};
  double turning_yaw_rate_threshold{0.35};
  double turning_max_planar_speed{0.30};
  bool stair_heading_lock_enabled{false};
  double stair_heading_lock_half_window_arc{0.45};
  double stair_heading_lock_min_pitch_rad{0.45};
  double stair_forward_speed_floor{0.0};
  double reference_path_body_height{0.30};
  double reference_path_min_point_spacing{0.05};
  double reference_path_backward_arc{1.0};
  double reference_path_forward_arc{3.0};
  int reference_path_max_points{4096};
  double kp_position{0.80};
  double kp_yaw{1.50};
  double max_vx{0.30};
  double max_vy{0.15};
  double max_yaw_rate{0.45};
  double active_sensing_max_yaw_excursion{0.22};
  double active_sensing_max_yaw_rate{0.20};
  double active_sensing_yaw_tolerance{0.02};
  double max_ax{0.50};
  double max_ay{0.40};
  double max_yaw_acc{1.00};
  double finish_distance_xy{0.08};
  double finish_distance_z{0.12};
  double terminal_capture_entry_distance_xy{0.055};
  double terminal_capture_zero_hold_distance_xy{0.075};
  double terminal_capture_release_distance_xy{0.12};
  double terminal_capture_brake_hold_sec{0.06};
  double terminal_capture_stable_dwell_sec{0.50};
  double terminal_position_hold_gain{0.80};
  double terminal_position_hold_max_speed{0.10};
  double terminal_approach_min_speed{0.22};
  double terminal_max_yaw_rate{0.25};
  double terminal_max_yaw_acc{0.50};
  double terminal_yaw_control_deadband{0.18};
  double finish_yaw_error{0.20};
  double finish_speed{0.05};
  double finish_vertical_speed{0.05};
  double finish_angular_speed{0.10};
  double finish_yaw_rate{0.10};
  double bspline_timeout_sec{1.50};
  double odom_timeout_sec{0.30};
  double cloud_timeout_sec{0.50};
  double trajectory_expiry_grace_sec{3.00};
  double max_yaw_alignment_freeze_sec{6.00};
  double max_control_dt_sec{0.20};
  double future_tolerance_sec{0.10};
  double max_start_stamp_skew_sec{0.50};
};

struct TrajectoryInput
{
  Eigen::MatrixXd control_points;
  Eigen::VectorXd knots;
  std::vector<double> yaw_points;
  double yaw_dt{0.0};
  int order{0};
  std::int64_t trajectory_id{0};
  double header_stamp_sec{0.0};
  std::int64_t header_stamp_ns{0};
  double start_stamp_sec{0.0};
  std::int64_t start_stamp_ns{0};
  double reference_path_stamp_sec{0.0};
  // ROS 消息入口必须保留原始纳秒代际，不能用未来时间容差比较 Path 代际。
  std::int64_t reference_path_stamp_ns{0};
  bool is_final{false};
  bool emergency_stop{false};
};

struct OdometryInput
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double planar_speed{0.0};
  double vertical_speed{0.0};
  double angular_speed{0.0};
  double yaw_rate{0.0};
  double stamp_sec{0.0};
};

enum class ControllerState
{
  kWaitingForTrajectory,
  kWaitingForReferencePath,
  kWaitingForOdometry,
  kWaitingForCloud,
  kTrajectoryTimeout,
  kOdometryTimeout,
  kCloudTimeout,
  kInvalidClock,
  kEmergencyStop,
  kAligningYaw,
  kTracking,
  kTrajectoryFinished,
  kGoalReached,
};

struct ControlOutput
{
  double vx{0.0};
  double vy{0.0};
  double wz{0.0};
  bool execution_frozen{false};
  bool trajectory_finished{false};
  bool goal_reached{false};
  ControllerState state{ControllerState::kWaitingForTrajectory};
};

class TrajectoryTracker
{
public:
  explicit TrajectoryTracker(const TrackerConfig & config);

  bool setTrajectory(
    const TrajectoryInput & input, double now_sec, std::string & error);
  bool setReferencePath(
    const std::vector<Eigen::Vector3d> & points,
    double stamp_sec, double now_sec, std::string & error,
    std::optional<double> terminal_yaw = std::nullopt,
    std::int64_t stamp_ns = 0);
  bool setOdometry(
    const OdometryInput & input, double now_sec, std::string & error);
  bool setCloudObservation(
    double stamp_sec, double now_sec, std::string & error);

  void invalidateTrajectory();
  void invalidateReferencePath(
    std::int64_t invalidated_through_stamp_ns = 0);
  void invalidateOdometry();
  void invalidateCloud();
  void clearCompletionLatch();

  ControlOutput immediateStop(double now_sec);
  ControlOutput update(double now_sec);

  double executionTimeSec() const;
  double trajectoryDurationSec() const;
  bool hasTrajectory() const;
  bool hasReferencePath() const;
  double referencePathStampSec() const;

  static const char * stateName(ControllerState state);

private:
  static void validateConfig(const TrackerConfig & config);
  static double normalizeAngle(double angle);
  static double limitRate(
    double previous, double target, double max_rate, double dt);

  ControlOutput stop(
    ControllerState state, double now_sec, bool execution_frozen = false);
  double estimateTrackingDesiredYaw(
    double trajectory_time,
    const Eigen::Vector3d & tracking_position) const;
  std::optional<double> estimateTrajectoryDesiredYaw(
    double trajectory_time) const;
  double referencePathCrossTrackError();
  /**
  - @brief 在当前完整参考 Path 的楼梯窗口内计算稳定机体航向
  - @return 楼梯窗口的水平合成切向；当前不在楼梯附近时返回空值
  */
  std::optional<double> referencePathStairHeading() const;
  /**
  - @brief 在已认证楼梯窗口内补足四足策略需要的前向牵引速度
  - @param velocity_world          B-spline 与位置反馈生成的世界系平面速度
  - @param stair_heading           完整参考 Path 给出的楼梯水平航向
  - @param terminal_approach_active 是否已经进入最终点接近与制动区域
  - @return                        保留横向分量后的安全世界系平面速度
  */
  Eigen::Vector2d applyStairForwardSpeedFloor(
    const Eigen::Vector2d & velocity_world,
    const std::optional<double> & stair_heading,
    bool terminal_approach_active) const;
  bool finalReferencePositionWithin(
    double distance_xy_limit, double distance_z_limit) const;
  Eigen::Vector2d terminalPositionHoldTargetBody(double yaw_error) const;
  bool finalReferencePositionSatisfied() const;
  bool terminalAttitudeAndMotionSatisfied() const;
  bool physicalCompletionSatisfied() const;
  bool completionSatisfied() const;
  void resetStationaryFinalHoldDwell();
  void resetTerminalCaptureStability();
  void resetTerminalCapture();
  void resetYawAlignmentEpisode();

  TrackerConfig config_;
  scan_planner::UniformBspline position_trajectory_;
  scan_planner::UniformBspline velocity_trajectory_;
  std::vector<Eigen::Vector3d> reference_path_;
  std::vector<double> reference_segment_start_progress_;
  std::vector<double> reference_segment_length_;
  Eigen::Vector3d reference_path_goal_ground_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d odom_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector2d reference_path_correction_world_{Eigen::Vector2d::Zero()};
  double odom_yaw_{0.0};
  double odom_planar_speed_{0.0};
  double odom_vertical_speed_{0.0};
  double odom_angular_speed_{0.0};
  double odom_yaw_rate_{0.0};
  double trajectory_duration_sec_{0.0};
  double execution_time_sec_{0.0};
  double trajectory_stamp_sec_{0.0};
  std::int64_t trajectory_header_stamp_ns_{0};
  std::int64_t trajectory_start_stamp_ns_{0};
  double trajectory_reference_path_stamp_sec_{0.0};
  std::int64_t trajectory_reference_path_stamp_ns_{0};
  std::int64_t yaw_alignment_reference_path_stamp_ns_{0};
  double trajectory_expiry_sec_{0.0};
  double trajectory_hard_expiry_sec_{0.0};
  double terminal_capture_brake_until_sec_{0.0};
  double terminal_capture_hard_expiry_sec_{0.0};
  double terminal_capture_stable_duration_sec_{0.0};
  double yaw_alignment_hard_expiry_sec_{0.0};
  double yaw_alignment_target_yaw_{0.0};
  double active_sensing_target_yaw_{0.0};
  double trajectory_yaw_dt_{0.0};
  double reference_path_stamp_sec_{0.0};
  std::int64_t reference_path_stamp_ns_{0};
  std::int64_t invalidated_reference_path_stamp_ns_{0};
  double reference_path_total_length_{0.0};
  double reference_path_progress_{0.0};
  double reference_path_tangent_yaw_{0.0};
  double reference_path_terminal_yaw_{0.0};
  double odom_stamp_sec_{0.0};
  double cloud_stamp_sec_{0.0};
  double last_update_sec_{0.0};
  double last_seen_clock_sec_{0.0};
  double previous_vx_{0.0};
  double previous_vy_{0.0};
  double previous_wz_{0.0};
  std::int64_t trajectory_id_{0};
  int trajectory_order_{0};
  std::vector<double> trajectory_yaw_points_;
  bool have_trajectory_{false};
  bool have_reference_path_{false};
  bool have_reference_path_progress_{false};
  bool reference_path_forward_reacquisition_pending_{false};
  bool cross_track_alignment_active_{false};
  bool yaw_alignment_active_{false};
  bool active_sensing_yaw_only_{false};
  bool emergency_stop_latched_{false};
  bool have_odometry_{false};
  bool have_cloud_{false};
  bool trajectory_is_final_{false};
  bool stationary_final_hold_{false};
  bool terminal_capture_active_{false};
  bool terminal_translation_brake_active_{false};
  bool trajectory_finished_{false};
  bool trajectory_identity_poisoned_{false};
};

}  // 命名空间 scan_controller

#endif  // SCAN_CONTROLLER__TRAJECTORY_TRACKER_HPP_ 头文件保护
