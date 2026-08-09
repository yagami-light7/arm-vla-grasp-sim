#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/controller_status.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>

#include "scan_controller/command_evidence.hpp"
#include "scan_controller/point_cloud_validation.hpp"
#include "scan_controller/scan_controller_node.hpp"
#include "scan_controller/trajectory_tracker.hpp"

namespace scan_controller
{
namespace
{

constexpr int kControllerStatusEvidenceDepth = 64;

bool finiteTwist(const geometry_msgs::msg::Twist & twist)
{
  const double values[] = {
    twist.linear.x, twist.linear.y, twist.linear.z,
    twist.angular.x, twist.angular.y, twist.angular.z,
  };
  return std::all_of(
    std::begin(values), std::end(values),
    [](double value) {return std::isfinite(value);});
}

bool finiteCovariance(const std::array<double, 36> & covariance)
{
  return std::all_of(
    covariance.begin(), covariance.end(),
    [](double value) {return std::isfinite(value);});
}

struct TrajectoryIdentity
{
  builtin_interfaces::msg::Time reference_path_stamp;
  builtin_interfaces::msg::Time bspline_header_stamp;
  builtin_interfaces::msg::Time start_time;
  std::int64_t trajectory_id{0};
  bool is_final{false};
  bool emergency_stop{false};
  bool active_sensing_yaw_only{false};
};

TrajectoryIdentity trajectoryIdentity(
  const scan_planner_msgs::msg::Bspline & message)
{
  TrajectoryIdentity identity;
  identity.reference_path_stamp = message.reference_path_stamp;
  identity.bspline_header_stamp = message.header.stamp;
  identity.start_time = message.start_time;
  identity.trajectory_id = message.traj_id;
  identity.is_final = message.is_final;
  identity.emergency_stop = message.emergency_stop;
  identity.active_sensing_yaw_only = !message.yaw_pts.empty();
  return identity;
}

bool sameTime(
  const builtin_interfaces::msg::Time & left,
  const builtin_interfaces::msg::Time & right)
{
  return left.sec == right.sec && left.nanosec == right.nanosec;
}

bool sameIdentity(
  const TrajectoryIdentity & left,
  const TrajectoryIdentity & right)
{
  return
    left.trajectory_id == right.trajectory_id &&
    left.is_final == right.is_final &&
    left.emergency_stop == right.emergency_stop &&
    sameTime(left.reference_path_stamp, right.reference_path_stamp) &&
    sameTime(left.bspline_header_stamp, right.bspline_header_stamp) &&
    sameTime(left.start_time, right.start_time);
}

std::uint8_t controllerStateCode(ControllerState state)
{
  using Status = scan_planner_msgs::msg::ControllerStatus;
  switch (state) {
    case ControllerState::kWaitingForTrajectory:
      return Status::STATE_WAITING_FOR_TRAJECTORY;
    case ControllerState::kWaitingForReferencePath:
      return Status::STATE_WAITING_FOR_REFERENCE_PATH;
    case ControllerState::kWaitingForOdometry:
      return Status::STATE_WAITING_FOR_ODOMETRY;
    case ControllerState::kWaitingForCloud:
      return Status::STATE_WAITING_FOR_CLOUD;
    case ControllerState::kTrajectoryTimeout:
      return Status::STATE_TRAJECTORY_TIMEOUT;
    case ControllerState::kOdometryTimeout:
      return Status::STATE_ODOMETRY_TIMEOUT;
    case ControllerState::kCloudTimeout:
      return Status::STATE_CLOUD_TIMEOUT;
    case ControllerState::kInvalidClock:
      return Status::STATE_INVALID_CLOCK;
    case ControllerState::kEmergencyStop:
      return Status::STATE_EMERGENCY_STOP;
    case ControllerState::kAligningYaw:
      return Status::STATE_ALIGNING_YAW;
    case ControllerState::kTracking:
      return Status::STATE_TRACKING;
    case ControllerState::kTrajectoryFinished:
      return Status::STATE_TRAJECTORY_FINISHED;
    case ControllerState::kGoalReached:
      return Status::STATE_GOAL_REACHED;
  }
  return Status::STATE_UNKNOWN;
}

}  // 匿名命名空间

class ScanControllerNode : public rclcpp::Node
{
public:
  explicit ScanControllerNode(const rclcpp::NodeOptions & options)
  : Node("scan_controller", options)
  {
    bool use_sim_time = false;
    if (
      !get_parameter("use_sim_time", use_sim_time) ||
      !use_sim_time)
    {
      throw std::runtime_error("scan_controller 必须使用 use_sim_time=true");
    }

    TrackerConfig config;
    config.time_forward =
      declare_parameter<double>("controller.time_forward", 0.60);
    config.yaw_alignment_min_chord_distance =
      declare_parameter<double>(
      "controller.yaw_alignment_min_chord_distance", 0.03);
    config.heading_error_threshold =
      declare_parameter<double>("controller.heading_error_threshold", 0.70);
    config.heading_error_release_threshold =
      declare_parameter<double>(
      "controller.heading_error_release_threshold", 0.55);
    config.cross_track_alignment_distance =
      declare_parameter<double>(
      "controller.cross_track_alignment_distance", 0.12);
    config.cross_track_alignment_release_distance =
      declare_parameter<double>(
      "controller.cross_track_alignment_release_distance", 0.08);
    config.cross_track_heading_error_threshold =
      declare_parameter<double>(
      "controller.cross_track_heading_error_threshold", 0.20);
    config.cross_track_heading_error_release_threshold =
      declare_parameter<double>(
      "controller.cross_track_heading_error_release_threshold", 0.18);
    config.cross_track_recovery_forward_speed =
      declare_parameter<double>(
      "controller.cross_track_recovery_forward_speed", 0.18);
    config.cross_track_recovery_taper_distance =
      declare_parameter<double>(
      "controller.cross_track_recovery_taper_distance", 0.30);
    config.cross_track_recovery_lateral_gain =
      declare_parameter<double>(
      "controller.cross_track_recovery_lateral_gain", 0.45);
    config.cross_track_recovery_lateral_speed =
      declare_parameter<double>(
      "controller.cross_track_recovery_lateral_speed", 0.08);
    config.cross_track_heading_assist_gain =
      declare_parameter<double>(
      "controller.cross_track_heading_assist_gain", 1.00);
    config.cross_track_heading_assist_max =
      declare_parameter<double>(
      "controller.cross_track_heading_assist_max", 0.16);
    config.turning_speed_limit_enabled =
      declare_parameter<bool>(
      "controller.turning_speed_limit_enabled", false);
    config.turning_yaw_rate_threshold =
      declare_parameter<double>(
      "controller.turning_yaw_rate_threshold", 0.35);
    config.turning_max_planar_speed =
      declare_parameter<double>(
      "controller.turning_max_planar_speed", 0.30);
    config.stair_heading_lock_enabled =
      declare_parameter<bool>(
      "controller.stair_heading_lock_enabled", false);
    config.stair_heading_lock_half_window_arc =
      declare_parameter<double>(
      "controller.stair_heading_lock_half_window_arc_m", 0.45);
    config.stair_heading_lock_min_pitch_rad =
      declare_parameter<double>(
      "controller.stair_heading_lock_min_pitch_rad", 0.45);
    config.stair_forward_speed_floor =
      declare_parameter<double>(
      "controller.stair_forward_speed_floor", 0.0);
    config.reference_path_body_height =
      declare_parameter<double>("reference_path.body_height_m", 0.30);
    config.reference_path_min_point_spacing =
      declare_parameter<double>(
      "reference_path.min_point_spacing_m", 0.05);
    config.reference_path_backward_arc =
      declare_parameter<double>(
      "reference_path.backward_arc_m", 1.0);
    config.reference_path_forward_arc =
      declare_parameter<double>(
      "reference_path.forward_arc_m", 3.0);
    config.reference_path_max_points =
      declare_parameter<int>("reference_path.max_points", 4096);
    reference_path_max_points_ = config.reference_path_max_points;
    config.kp_position =
      declare_parameter<double>("controller.kp_position", 0.80);
    config.kp_yaw =
      declare_parameter<double>("controller.kp_yaw", 1.50);
    config.max_vx =
      declare_parameter<double>("limits.max_vx", 0.30);
    config.max_vy =
      declare_parameter<double>("limits.max_vy", 0.15);
    config.max_yaw_rate =
      declare_parameter<double>("limits.max_yaw_rate", 0.45);
    config.active_sensing_max_yaw_excursion =
      declare_parameter<double>(
      "controller.active_sensing_max_yaw_excursion", 0.22);
    config.active_sensing_max_yaw_rate =
      declare_parameter<double>(
      "controller.active_sensing_max_yaw_rate", 0.20);
    config.active_sensing_yaw_tolerance =
      declare_parameter<double>(
      "controller.active_sensing_yaw_tolerance", 0.02);
    config.max_ax =
      declare_parameter<double>("limits.max_ax", 0.50);
    config.max_ay =
      declare_parameter<double>("limits.max_ay", 0.40);
    config.max_yaw_acc =
      declare_parameter<double>("limits.max_yaw_acc", 1.00);
    config.finish_distance_xy =
      declare_parameter<double>("finish.distance_xy", 0.08);
    config.finish_distance_z =
      declare_parameter<double>("finish.distance_z", 0.12);
    config.terminal_capture_entry_distance_xy =
      declare_parameter<double>("finish.capture_entry_distance_xy", 0.055);
    config.terminal_capture_zero_hold_distance_xy =
      declare_parameter<double>(
      "finish.capture_zero_hold_distance_xy", 0.075);
    config.terminal_capture_release_distance_xy =
      declare_parameter<double>(
      "finish.capture_release_distance_xy", 0.12);
    config.terminal_capture_brake_hold_sec =
      declare_parameter<double>("finish.capture_brake_hold_sec", 0.06);
    config.terminal_capture_stable_dwell_sec =
      declare_parameter<double>("finish.capture_stable_dwell_sec", 0.50);
    config.terminal_position_hold_gain =
      declare_parameter<double>("finish.position_hold_gain", 0.80);
    config.terminal_position_hold_max_speed =
      declare_parameter<double>("finish.max_position_hold_speed", 0.10);
    config.terminal_approach_min_speed =
      declare_parameter<double>("finish.min_approach_speed", 0.22);
    config.terminal_max_yaw_rate =
      declare_parameter<double>("finish.max_capture_yaw_rate", 0.25);
    config.terminal_max_yaw_acc =
      declare_parameter<double>("finish.max_capture_yaw_acc", 0.50);
    config.terminal_yaw_control_deadband =
      declare_parameter<double>("finish.yaw_control_deadband", 0.18);
    config.finish_yaw_error =
      declare_parameter<double>("finish.max_yaw_error", 0.20);
    config.finish_speed =
      declare_parameter<double>("finish.max_planar_speed", 0.05);
    config.finish_vertical_speed =
      declare_parameter<double>("finish.max_vertical_speed", 0.05);
    config.finish_angular_speed =
      declare_parameter<double>("finish.max_angular_speed", 0.10);
    config.finish_yaw_rate =
      declare_parameter<double>("finish.max_yaw_rate", 0.10);
    config.bspline_timeout_sec =
      declare_parameter<double>("timeouts.bspline_sec", 1.50);
    config.odom_timeout_sec =
      declare_parameter<double>("timeouts.odom_sec", 0.30);
    config.cloud_timeout_sec =
      declare_parameter<double>("timeouts.cloud_sec", 0.50);
    config.trajectory_expiry_grace_sec =
      declare_parameter<double>(
      "timeouts.trajectory_expiry_grace_sec", 3.00);
    config.max_yaw_alignment_freeze_sec =
      declare_parameter<double>(
      "timeouts.max_yaw_alignment_freeze_sec", 6.00);
    config.max_control_dt_sec =
      declare_parameter<double>("timeouts.max_control_dt_sec", 0.20);
    config.future_tolerance_sec =
      declare_parameter<double>("timestamps.future_tolerance_sec", 0.10);
    config.max_start_stamp_skew_sec =
      declare_parameter<double>("timestamps.max_start_stamp_skew_sec", 0.50);
    tracker_ = std::make_unique<TrajectoryTracker>(config);

    world_frame_id_ =
      declare_parameter<std::string>("frames.world", "world");
    base_frame_id_ =
      declare_parameter<std::string>("frames.base", "base_link");
    bspline_topic_ =
      declare_parameter<std::string>(
      "topics.bspline", "/planning/bspline");
    reference_path_topic_ =
      declare_parameter<std::string>(
      "topics.initial_path", "/initial_path");
    odom_topic_ =
      declare_parameter<std::string>("topics.body_pose", "/body_pose");
    cloud_topic_ =
      declare_parameter<std::string>(
      "topics.cloud", "/cloud_registered");
    cmd_vel_topic_ =
      declare_parameter<std::string>("topics.cmd_vel", "/cmd_vel");
    frozen_topic_ =
      declare_parameter<std::string>(
      "topics.execution_frozen", "/planning/go2_execution_frozen");
    goal_reached_topic_ =
      declare_parameter<std::string>(
      "topics.goal_reached", "/planning/goal_reached");
    trajectory_finished_topic_ =
      declare_parameter<std::string>(
      "topics.trajectory_finished", "/planning/trajectory_finished");
    controller_status_topic_ =
      declare_parameter<std::string>(
      "topics.controller_status", "/planning/controller_status");
    quaternion_norm_tolerance_ =
      declare_parameter<double>("validation.quaternion_norm_tolerance", 0.05);
    const double publish_rate_hz =
      declare_parameter<double>("controller.publish_rate_hz", 100.0);
    const int reference_path_qos_depth =
      declare_parameter<int>("qos.initial_path_depth", 1);
    const int controller_status_qos_depth =
      declare_parameter<int>(
      "qos.controller_status_depth", kControllerStatusEvidenceDepth);

    const std::string required_strings[] = {
      world_frame_id_, base_frame_id_, bspline_topic_,
      reference_path_topic_, odom_topic_,
      cloud_topic_, cmd_vel_topic_, frozen_topic_, goal_reached_topic_,
      trajectory_finished_topic_, controller_status_topic_,
    };
    for (const auto & value : required_strings) {
      if (value.empty()) {
        throw std::runtime_error("控制器 frame 与 topic 参数不能为空");
      }
    }
    if (
      !std::isfinite(quaternion_norm_tolerance_) ||
      quaternion_norm_tolerance_ <= 0.0 ||
      !std::isfinite(publish_rate_hz) ||
      publish_rate_hz <= 0.0 ||
      reference_path_qos_depth <= 0)
    {
      throw std::runtime_error("控制器校验容差与发布频率必须为有限正数");
    }
    if (controller_status_qos_depth != kControllerStatusEvidenceDepth) {
      throw std::runtime_error(
              "qos.controller_status_depth 必须固定为 64，"
              "防止连续 typed 证据被覆盖");
    }

    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, rclcpp::QoS(10).reliable());
    execution_frozen_pub_ = create_publisher<std_msgs::msg::Bool>(
      frozen_topic_, rclcpp::QoS(10).reliable());
    goal_reached_pub_ = create_publisher<std_msgs::msg::Bool>(
      goal_reached_topic_, rclcpp::QoS(1).reliable().transient_local());
    trajectory_finished_pub_ = create_publisher<std_msgs::msg::Bool>(
      trajectory_finished_topic_,
      rclcpp::QoS(1).reliable().transient_local());
    controller_status_pub_ =
      create_publisher<scan_planner_msgs::msg::ControllerStatus>(
      controller_status_topic_,
      rclcpp::QoS(rclcpp::KeepLast(controller_status_qos_depth))
      .reliable().transient_local());

    bspline_sub_ = create_subscription<scan_planner_msgs::msg::Bspline>(
      bspline_topic_,
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
      std::bind(
        &ScanControllerNode::bsplineCallback, this,
        std::placeholders::_1));
    reference_path_sub_ = create_subscription<nav_msgs::msg::Path>(
      reference_path_topic_,
      rclcpp::QoS(rclcpp::KeepLast(reference_path_qos_depth))
      .reliable().transient_local(),
      std::bind(
        &ScanControllerNode::referencePathCallback, this,
        std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(
        &ScanControllerNode::odomCallback, this,
        std::placeholders::_1));
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(
        &ScanControllerNode::cloudCallback, this,
        std::placeholders::_1));

    timer_ = rclcpp::create_timer(
      this, get_clock(),
      rclcpp::Duration::from_seconds(1.0 / publish_rate_hz),
      std::bind(&ScanControllerNode::timerCallback, this));
    RCLCPP_INFO(
      get_logger(),
      "SCAN 闭环控制器已启动，输出为 base_link 机体系速度");
  }

private:
  void publishControllerStatus(
    std::uint8_t event, ControllerState state, const std::string & reason,
    const std::optional<TrajectoryIdentity> & candidate = std::nullopt)
  {
    TrajectoryIdentity identity;
    if (last_accepted_identity_.has_value()) {
      identity = *last_accepted_identity_;
    }
    scan_planner_msgs::msg::ControllerStatus status;
    status.header.stamp = now();
    status.header.frame_id = world_frame_id_;
    status.status_sequence = ++status_sequence_;
    status.acceptance_sequence = acceptance_sequence_;
    status.event = event;
    status.reference_path_stamp = identity.reference_path_stamp;
    status.bspline_header_stamp = identity.bspline_header_stamp;
    status.start_time = identity.start_time;
    status.traj_id = identity.trajectory_id;
    status.accepted = last_accepted_identity_.has_value();
    status.trajectory_valid =
      status.accepted && tracker_->hasTrajectory();
    status.is_final = identity.is_final;
    status.emergency_stop = identity.emergency_stop;
    status.active_sensing_yaw_only =
      status.accepted && identity.active_sensing_yaw_only;
    const CommandAggregate & aggregate = command_evidence_.aggregate();
    status.command_sample_count = aggregate.sample_count;
    status.first_command = aggregate.first_command;
    status.max_abs_vx = aggregate.max_abs_vx;
    status.max_abs_vy = aggregate.max_abs_vy;
    status.max_abs_wz = aggregate.max_abs_wz;
    status.command_violation_count = aggregate.violation_count;
    status.state = controllerStateCode(state);
    status.reason = reason;
    status.candidate_present = candidate.has_value();
    if (candidate.has_value()) {
      status.candidate_reference_path_stamp =
        candidate->reference_path_stamp;
      status.candidate_bspline_header_stamp =
        candidate->bspline_header_stamp;
      status.candidate_start_time = candidate->start_time;
      status.candidate_traj_id = candidate->trajectory_id;
    }
    controller_status_pub_->publish(status);
  }

  void publishActiveTrajectoryStatus(
    ControllerState state, const std::string & reason)
  {
    const std::uint8_t event = status_sequence_ == 0U ?
      scan_planner_msgs::msg::ControllerStatus::EVENT_INITIAL :
      scan_planner_msgs::msg::ControllerStatus::EVENT_STATE_CHANGED;
    publishControllerStatus(
      event, state, reason);
  }

  void publishTrajectoryInvalidated(
    ControllerState state, const std::string & reason)
  {
    if (!last_accepted_identity_.has_value()) {
      return;
    }
    publishControllerStatus(
      scan_planner_msgs::msg::ControllerStatus::EVENT_INVALIDATED,
      state, reason);
    command_evidence_.disable();
  }

  void publishTrajectoryRejected(
    const scan_planner_msgs::msg::Bspline & message,
    ControllerState state, const std::string & reason)
  {
    publishControllerStatus(
      scan_planner_msgs::msg::ControllerStatus::EVENT_REJECTED,
      state, reason, trajectoryIdentity(message));
  }

  void publishTrajectoryReplacementSnapshot(
    const TrajectoryIdentity & candidate)
  {
    if (
      !last_accepted_identity_.has_value() ||
      !command_evidence_.enabled() ||
      sameIdentity(*last_accepted_identity_, candidate))
    {
      return;
    }
    publishControllerStatus(
      scan_planner_msgs::msg::ControllerStatus::EVENT_STATE_CHANGED,
      last_state_, "新 B-spline identity 替换前的旧轨迹命令聚合终态快照");
  }

  void activateAcceptedTrajectory(const TrajectoryIdentity & identity)
  {
    last_accepted_identity_ = identity;
    ++acceptance_sequence_;
    command_evidence_.reset(identity.active_sensing_yaw_only);
  }

  void publishTrajectoryAccepted(bool duplicate)
  {
    publishControllerStatus(
      duplicate ?
      scan_planner_msgs::msg::ControllerStatus::EVENT_DUPLICATE :
      scan_planner_msgs::msg::ControllerStatus::EVENT_ACCEPTED,
      last_state_,
      duplicate ? "B-spline 幂等重发已确认" : "B-spline 已接受");
  }

  void publishOutput(const ControlOutput & output)
  {
    // 聚合先于 DDS publish，保证 accepted status 中的 first_command 就是
    // 主动观测 identity 生效时同步覆盖旧速度的严格零首拍。
    const geometry_msgs::msg::Twist command =
      command_evidence_.filterAndRecord(output);
    cmd_vel_pub_->publish(command);

    std_msgs::msg::Bool state;
    state.data = output.execution_frozen;
    execution_frozen_pub_->publish(state);
    publishCompletionState(output);
  }

  void publishCompletionState(const ControlOutput & output)
  {
    std_msgs::msg::Bool state;
    state.data = output.goal_reached;
    goal_reached_pub_->publish(state);
    state.data = output.trajectory_finished;
    trajectory_finished_pub_->publish(state);
  }

  ControlOutput publishImmediateStop()
  {
    const ControlOutput output = tracker_->immediateStop(now().seconds());
    publishOutput(output);
    return output;
  }

  void bsplineCallback(
    const scan_planner_msgs::msg::Bspline::ConstSharedPtr message)
  {
    // 接受与同步首拍必须共享同一个 ROS 时刻；否则仿真时钟在两次 now()
    // 之间前跳时，可能产生非零 dt，甚至让刚接受的轨迹先失效再发布 accepted。
    const double callback_now_sec = now().seconds();
    if (message->header.frame_id != world_frame_id_) {
      const bool had_trajectory = tracker_->hasTrajectory();
      tracker_->invalidateTrajectory();
      const ControlOutput output = publishImmediateStop();
      const std::string reason =
        "B-spline frame_id 必须为 " + world_frame_id_;
      if (had_trajectory && !tracker_->hasTrajectory()) {
        publishTrajectoryInvalidated(
          output.state, "已接受 B-spline 作废：" + reason);
      }
      publishTrajectoryRejected(*message, output.state, reason);
      RCLCPP_WARN(
        get_logger(), "拒绝 B-spline：frame_id 必须为 %s",
        world_frame_id_.c_str());
      return;
    }
    TrajectoryInput input;
    input.order = message->order;
    input.trajectory_id = message->traj_id;
    input.header_stamp_sec =
      rclcpp::Time(message->header.stamp).seconds();
    input.header_stamp_ns =
      rclcpp::Time(message->header.stamp).nanoseconds();
    input.start_stamp_sec =
      rclcpp::Time(message->start_time).seconds();
    input.start_stamp_ns =
      rclcpp::Time(message->start_time).nanoseconds();
    input.reference_path_stamp_sec =
      rclcpp::Time(message->reference_path_stamp).seconds();
    input.reference_path_stamp_ns =
      rclcpp::Time(message->reference_path_stamp).nanoseconds();
    input.is_final = message->is_final;
    input.emergency_stop = message->emergency_stop;
    input.yaw_points.assign(message->yaw_pts.begin(), message->yaw_pts.end());
    input.yaw_dt = message->yaw_dt;
    input.control_points.resize(3, message->pos_pts.size());
    for (std::size_t index = 0; index < message->pos_pts.size(); ++index) {
      input.control_points.col(index) <<
        message->pos_pts[index].x,
        message->pos_pts[index].y,
        message->pos_pts[index].z;
    }
    input.knots.resize(message->knots.size());
    for (std::size_t index = 0; index < message->knots.size(); ++index) {
      input.knots(index) = message->knots[index];
    }

    std::string error;
    const TrajectoryIdentity candidate_identity = trajectoryIdentity(*message);
    const bool duplicate_identity =
      last_accepted_identity_.has_value() &&
      sameIdentity(*last_accepted_identity_, candidate_identity);
    // 旧 identity 的统计必须先进入 typed 状态流，随后才能让新候选覆盖
    // tracker 与聚合器。候选若被拒绝，此快照只是保守检查点，旧聚合会继续；
    // 候选若被接受，它就是旧 identity 的最终替换前证据。
    publishTrajectoryReplacementSnapshot(candidate_identity);
    const bool had_trajectory = tracker_->hasTrajectory();
    if (!tracker_->setTrajectory(input, callback_now_sec, error)) {
      const ControlOutput output = publishImmediateStop();
      if (had_trajectory && !tracker_->hasTrajectory()) {
        publishTrajectoryInvalidated(
          output.state, "已接受 B-spline 作废：" + error);
      }
      publishTrajectoryRejected(*message, output.state, error);
      RCLCPP_WARN(
        get_logger(), "拒绝 B-spline：%s", error.c_str());
      return;
    }
    if (!duplicate_identity) {
      activateAcceptedTrajectory(candidate_identity);
    }
    if (!duplicate_identity && !input.yaw_points.empty()) {
      // 主动观测 identity 生效后必须在同一订阅回调内先覆盖旧 cmd_vel。
      // setTrajectory 已把三轴历史与 last_update 置于当前拍，因此这里的
      // dt=0 输出严格全零；先更新 last_state_，再发布 accepted typed 状态，
      // 防止新 identity 与上一条轨迹的 TRACKING 状态错误配对。
      const ControlOutput active_observation_stop =
        tracker_->update(callback_now_sec);
      publishOutput(active_observation_stop);
      last_state_ = active_observation_stop.state;
    }
    // 立即覆盖 transient-local 中上一条轨迹的完成状态。
    publishCompletionState(ControlOutput{});
    publishTrajectoryAccepted(duplicate_identity);
    RCLCPP_INFO(
      get_logger(), "接收轨迹 %ld，时长 %.3f 秒，final=%s",
      static_cast<long>(message->traj_id),
      tracker_->trajectoryDurationSec(),
      message->is_final ? "true" : "false");
  }

  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    const auto & pose = message->pose.pose;
    const auto & orientation = pose.orientation;
    const double quaternion_norm = std::sqrt(
      orientation.x * orientation.x +
      orientation.y * orientation.y +
      orientation.z * orientation.z +
      orientation.w * orientation.w);
    const double pose_values[] = {
      pose.position.x, pose.position.y, pose.position.z,
      orientation.x, orientation.y, orientation.z, orientation.w,
    };
    const bool pose_is_finite = std::all_of(
      std::begin(pose_values), std::end(pose_values),
      [](double value) {return std::isfinite(value);});
    if (
      message->header.frame_id != world_frame_id_ ||
      message->child_frame_id != base_frame_id_ ||
      !pose_is_finite ||
      !finiteTwist(message->twist.twist) ||
      !finiteCovariance(message->pose.covariance) ||
      !finiteCovariance(message->twist.covariance) ||
      !std::isfinite(quaternion_norm) ||
      std::abs(quaternion_norm - 1.0) > quaternion_norm_tolerance_)
    {
      tracker_->invalidateOdometry();
      publishImmediateStop();
      RCLCPP_WARN(
        get_logger(),
        "拒绝 Odometry：frame、有限数或单位四元数合同不满足");
      return;
    }

    const double inverse_norm = 1.0 / quaternion_norm;
    const double x = orientation.x * inverse_norm;
    const double y = orientation.y * inverse_norm;
    const double z = orientation.z * inverse_norm;
    const double w = orientation.w * inverse_norm;
    OdometryInput input;
    input.position <<
      pose.position.x, pose.position.y, pose.position.z;
    input.yaw = std::atan2(
      2.0 * (w * z + x * y),
      1.0 - 2.0 * (y * y + z * z));
    input.planar_speed = std::hypot(
      message->twist.twist.linear.x,
      message->twist.twist.linear.y);
    const double world_vertical_speed =
      2.0 * (x * z - w * y) * message->twist.twist.linear.x +
      2.0 * (y * z + w * x) * message->twist.twist.linear.y +
      (1.0 - 2.0 * (x * x + y * y)) *
      message->twist.twist.linear.z;
    input.vertical_speed = std::abs(world_vertical_speed);
    input.angular_speed = std::sqrt(
      message->twist.twist.angular.x * message->twist.twist.angular.x +
      message->twist.twist.angular.y * message->twist.twist.angular.y +
      message->twist.twist.angular.z * message->twist.twist.angular.z);
    // Odometry twist 由 bridge 按 child_frame_id=base_link 发布；导航控制与
    // pipeline 的稳定性合同都把 z 分量解释为机体系 yaw rate。
    input.yaw_rate = std::abs(message->twist.twist.angular.z);
    input.stamp_sec = rclcpp::Time(message->header.stamp).seconds();
    std::string error;
    if (!tracker_->setOdometry(input, now().seconds(), error)) {
      publishImmediateStop();
      RCLCPP_WARN(
        get_logger(), "拒绝 Odometry：%s", error.c_str());
    }
  }

  void referencePathCallback(
    const nav_msgs::msg::Path::ConstSharedPtr message)
  {
    const rclcpp::Time path_stamp(message->header.stamp);
    if (
      message->header.frame_id != world_frame_id_ ||
      path_stamp.nanoseconds() <= 0)
    {
      if (!tracker_->hasReferencePath()) {
        publishImmediateStop();
      }
      RCLCPP_WARN(
        get_logger(),
        "忽略无效参考 Path：frame_id 必须为 %s 且时间戳非零",
        world_frame_id_.c_str());
      return;
    }
    if (
      latest_reference_path_stamp_ns_ > 0 &&
      path_stamp.nanoseconds() < latest_reference_path_stamp_ns_)
    {
      RCLCPP_WARN(
        get_logger(), "忽略早于当前代际的参考 Path");
      return;
    }
    if (message->poses.empty()) {
      const bool had_trajectory = tracker_->hasTrajectory();
      latest_reference_path_stamp_ns_ = path_stamp.nanoseconds();
      tracker_->invalidateReferencePath(path_stamp.nanoseconds());
      tracker_->invalidateTrajectory();
      tracker_->clearCompletionLatch();
      const ControlOutput output = publishImmediateStop();
      if (had_trajectory && !tracker_->hasTrajectory()) {
        publishTrajectoryInvalidated(
          output.state, "已接受 B-spline 作废：收到空参考 Path");
      }
      RCLCPP_WARN(
        get_logger(), "收到空参考 Path，已清除 Path 与局部轨迹代际");
      return;
    }
    if (
      message->poses.size() >
      static_cast<std::size_t>(reference_path_max_points_))
    {
      if (!tracker_->hasReferencePath()) {
        publishImmediateStop();
      }
      RCLCPP_ERROR(
        get_logger(),
        "拒绝参考 Path：%zu 点超过回调预分配上限 %d",
        message->poses.size(), reference_path_max_points_);
      return;
    }

    std::vector<Eigen::Vector3d> points;
    points.reserve(message->poses.size());
    for (
      std::size_t index = 0;
      index < message->poses.size();
      ++index)
    {
      const auto & pose_stamped = message->poses[index];
      const rclcpp::Time pose_stamp(pose_stamped.header.stamp);
      const auto & position = pose_stamped.pose.position;
      const auto & orientation = pose_stamped.pose.orientation;
      const double values[] = {
        position.x, position.y, position.z,
        orientation.x, orientation.y, orientation.z, orientation.w};
      const double quaternion_norm = std::sqrt(
        orientation.x * orientation.x +
        orientation.y * orientation.y +
        orientation.z * orientation.z +
        orientation.w * orientation.w);
      if (
        (
          !pose_stamped.header.frame_id.empty() &&
          pose_stamped.header.frame_id != world_frame_id_) ||
        pose_stamp.nanoseconds() <= 0 ||
        !std::all_of(
          std::begin(values), std::end(values),
          [](double value) {return std::isfinite(value);}) ||
        !std::isfinite(quaternion_norm) ||
        quaternion_norm <= 1.0e-6)
      {
        if (!tracker_->hasReferencePath()) {
          publishImmediateStop();
        }
        RCLCPP_WARN(
          get_logger(),
          "忽略无效参考 Path：第 %zu 个 Pose 的 frame、时间戳、"
          "位置或四元数非法",
          index);
        return;
      }
      points.emplace_back(position.x, position.y, position.z);
    }

    const auto & terminal_orientation =
      message->poses.back().pose.orientation;
    const double terminal_quaternion_norm = std::sqrt(
      terminal_orientation.x * terminal_orientation.x +
      terminal_orientation.y * terminal_orientation.y +
      terminal_orientation.z * terminal_orientation.z +
      terminal_orientation.w * terminal_orientation.w);
    const double qx = terminal_orientation.x / terminal_quaternion_norm;
    const double qy = terminal_orientation.y / terminal_quaternion_norm;
    const double qz = terminal_orientation.z / terminal_quaternion_norm;
    const double qw = terminal_orientation.w / terminal_quaternion_norm;
    const double terminal_heading_x = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double terminal_heading_y = 2.0 * (qx * qy + qw * qz);
    if (std::hypot(terminal_heading_x, terminal_heading_y) <= 1.0e-6) {
      if (!tracker_->hasReferencePath()) {
        publishImmediateStop();
      }
      RCLCPP_WARN(
        get_logger(),
        "忽略无效参考 Path：末 Pose 四元数无法确定 world 平面 terminal yaw");
      return;
    }
    const double terminal_yaw =
      std::atan2(terminal_heading_y, terminal_heading_x);

    std::string error;
    const double stamp_sec =
      rclcpp::Time(message->header.stamp).seconds();
    const bool had_trajectory = tracker_->hasTrajectory();
    if (!tracker_->setReferencePath(
        points, stamp_sec, now().seconds(), error, terminal_yaw,
        path_stamp.nanoseconds()))
    {
      std::optional<ControlOutput> stop_output;
      if (
        !tracker_->hasReferencePath() ||
        (had_trajectory && !tracker_->hasTrajectory()))
      {
        stop_output = publishImmediateStop();
      }
      if (had_trajectory && !tracker_->hasTrajectory()) {
        publishTrajectoryInvalidated(
          stop_output.has_value() ? stop_output->state : last_state_,
          "已接受 B-spline 作废：" + error);
      }
      RCLCPP_WARN(
        get_logger(), "忽略无效参考 Path：%s", error.c_str());
      return;
    }
    if (had_trajectory && !tracker_->hasTrajectory()) {
      // 新 Path 代际在订阅回调内立即停车，不能等下一次 100 Hz 控制定时器。
      const ControlOutput output = publishImmediateStop();
      publishTrajectoryInvalidated(
        output.state, "已接受 B-spline 作废：参考 Path 已切换到新代际");
    }
    latest_reference_path_stamp_ns_ = path_stamp.nanoseconds();
    RCLCPP_INFO(
      get_logger(), "接收参考 Path，共 %zu 个点", points.size());
  }

  void cloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    std::string error;
    if (
      message->header.frame_id != world_frame_id_ ||
      !validPointCloudLayout(*message, error))
    {
      tracker_->invalidateCloud();
      publishImmediateStop();
      if (error.empty()) {
        error = "PointCloud2 frame_id 不匹配";
      }
      RCLCPP_WARN(
        get_logger(), "拒绝 PointCloud2：%s", error.c_str());
      return;
    }
    const double stamp_sec =
      rclcpp::Time(message->header.stamp).seconds();
    if (!tracker_->setCloudObservation(stamp_sec, now().seconds(), error)) {
      publishImmediateStop();
      RCLCPP_WARN(
        get_logger(), "拒绝 PointCloud2：%s", error.c_str());
    }
  }

  void timerCallback()
  {
    const bool had_trajectory = tracker_->hasTrajectory();
    const ControlOutput output = tracker_->update(now().seconds());
    publishOutput(output);
    if (had_trajectory && !tracker_->hasTrajectory()) {
      publishTrajectoryInvalidated(
        output.state, "已接受 B-spline 作废：控制时钟 epoch 已重置");
    }
    if (output.state == last_state_) {
      return;
    }
    last_state_ = output.state;
    const char * state_name = TrajectoryTracker::stateName(output.state);
    publishActiveTrajectoryStatus(
      output.state, std::string("控制状态变化：") + state_name);
    if (
      output.state == ControllerState::kTracking ||
      output.state == ControllerState::kAligningYaw ||
      output.state == ControllerState::kGoalReached ||
      output.state == ControllerState::kTrajectoryFinished)
    {
      RCLCPP_INFO(get_logger(), "控制器状态：%s", state_name);
    } else {
      RCLCPP_WARN(get_logger(), "控制器安全停车：%s", state_name);
    }
  }

  std::unique_ptr<TrajectoryTracker> tracker_;
  std::string world_frame_id_;
  std::string base_frame_id_;
  std::string bspline_topic_;
  std::string reference_path_topic_;
  std::string odom_topic_;
  std::string cloud_topic_;
  std::string cmd_vel_topic_;
  std::string frozen_topic_;
  std::string goal_reached_topic_;
  std::string trajectory_finished_topic_;
  std::string controller_status_topic_;
  double quaternion_norm_tolerance_{0.05};
  int reference_path_max_points_{4096};
  std::int64_t latest_reference_path_stamp_ns_{0};
  std::uint64_t status_sequence_{0};
  std::uint64_t acceptance_sequence_{0};
  std::optional<TrajectoryIdentity> last_accepted_identity_;
  CommandEvidence command_evidence_;
  ControllerState last_state_{ControllerState::kInvalidClock};

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr
    execution_frozen_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_reached_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr
    trajectory_finished_pub_;
  rclcpp::Publisher<scan_planner_msgs::msg::ControllerStatus>::SharedPtr
    controller_status_pub_;
  rclcpp::Subscription<scan_planner_msgs::msg::Bspline>::SharedPtr
    bspline_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr
    reference_path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
    cloud_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

std::shared_ptr<rclcpp::Node> makeScanControllerNode(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<ScanControllerNode>(options);
}

}  // 命名空间 scan_controller
