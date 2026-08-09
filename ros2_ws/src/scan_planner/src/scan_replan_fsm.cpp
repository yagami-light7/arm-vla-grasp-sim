
// 本文件由 ROS 2 community 版本移植，并按本项目路径 QoS 与高度语义修改。
#include <plan_manage/scan_replan_fsm.h>
#include <plan_manage/controller_status_qos.h>
#include <plan_manage/reference_execution.h>
#include <plan_manage/reference_velocity.h>
#include <cmath>
#include <limits>
#include <rclcpp/create_timer.hpp>
#include <stdexcept>

namespace
{
  template <typename T>
  T load_parameter(rclcpp::Node *node, const std::string &name, const T &default_value)
  {
    if (!node->has_parameter(name)) node->declare_parameter<T>(name, default_value);
    return node->get_parameter(name).get_value<T>();
  }

  bool finite_pose(const geometry_msgs::msg::Pose &pose)
  {
    const double values[] = {
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w};
    for (const double value : values)
      if (!std::isfinite(value))
        return false;

    const double quaternion_norm =
        std::sqrt(
            pose.orientation.x * pose.orientation.x +
            pose.orientation.y * pose.orientation.y +
            pose.orientation.z * pose.orientation.z +
            pose.orientation.w * pose.orientation.w);
    return quaternion_norm > 1.0e-6;
  }

  bool finite_twist(const geometry_msgs::msg::Twist &twist)
  {
    const double values[] = {
        twist.linear.x, twist.linear.y, twist.linear.z,
        twist.angular.x, twist.angular.y, twist.angular.z};
    for (const double value : values)
      if (!std::isfinite(value))
        return false;
    return true;
  }

  bool same_waypoints(
      const std::vector<Eigen::Vector3d> &lhs,
      const std::vector<Eigen::Vector3d> &rhs)
  {
    if (lhs.size() != rhs.size())
      return false;
    for (std::size_t index = 0; index < lhs.size(); ++index)
      if ((lhs[index] - rhs[index]).norm() > 1.0e-9)
        return false;
    return true;
  }

  bool same_yaw(const double lhs, const double rhs)
  {
    if (!std::isfinite(lhs) || !std::isfinite(rhs))
      return false;
    const double difference = std::atan2(
        std::sin(lhs - rhs), std::cos(lhs - rhs));
    return std::abs(difference) <= 1.0e-9;
  }
} // namespace

namespace scan_planner
{

  using PlanningStatusMessage =
      scan_planner_msgs::msg::ScanPlanningStatus;
  using ControllerStatusMessage =
      scan_planner_msgs::msg::ControllerStatus;

  static_assert(
      static_cast<std::uint8_t>(ScanPlanningEvent::kInitial) ==
      PlanningStatusMessage::EVENT_INITIAL);
  static_assert(
      static_cast<std::uint8_t>(ScanPlanningEvent::kStairResumeWaiting) ==
      PlanningStatusMessage::EVENT_STAIR_RESUME_WAITING);
  static_assert(
      static_cast<std::uint8_t>(ScanPlanningState::kWaitingForReference) ==
      PlanningStatusMessage::STATE_WAITING_FOR_REFERENCE);
  static_assert(
      static_cast<std::uint8_t>(ScanPlanningState::kUnknown) ==
      PlanningStatusMessage::STATE_UNKNOWN);

  void SCANReplanFSM::init(rclcpp::Node *node)
  {
    node_ = node;
    current_wp_ = 0;
    exec_state_ = FSM_EXEC_STATE::INIT;
    trigger_ = false;
    have_target_ = false;
    have_odom_ = false;
    have_new_target_ = false;
    rviz_height_ready_ = false;
    go2_execution_frozen_ = false;
    flag_escape_emergency_ = true;
    need_hover_stop_ = false;
    replan_fail_count_ = 0;
    latest_reference_path_stamp_ns_ = 0;
    planning_status_sequence_ = 0;
    trajectory_diagnostics_sequence_ = 0;
    initial_planning_status_published_ = false;
    latest_reference_generation_is_empty_ = false;
    recovery_status_pending_ = false;
    pending_emergency_reason_ = "SCAN emergency stop";
    global_replan_generation_gate_.reset();
    have_last_published_trajectory_ = false;
    published_final_trajectory_history_.clear();
    controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
    have_controller_accepted_final_identity_ = false;
    pending_reference_waypoints_.clear();
    active_reference_waypoints_.clear();
    active_reference_arc_lengths_.clear();
    local_reference_guide_.clear();
    local_reference_corridor_guide_.clear();
    reference_progress_s_ = 0.0;
    pending_reference_goal_yaw_ = 0.0;
    active_reference_goal_yaw_ = 0.0;
    pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
    active_reference_path_stamp_ = builtin_interfaces::msg::Time();
    reference_final_hold_lifecycle_ = FinalHoldLifecycleState{};
    reference_goal_hold_dwell_ = ReferenceGoalHoldDwellState{};
    active_sensing_phase_ = ACTIVE_SENSING_IDLE;
    last_local_plan_attempt_reached_manager_ = false;
    active_sensing_consumed_path_stamp_ns_ = 0;
    active_sensing_path_stamp_ns_ = 0;
    active_sensing_publish_stamp_ns_ = 0;
    active_sensing_yaw_stable_since_ns_ = 0;
    active_sensing_observation_baseline_stamp_ns_ = 0;
    active_sensing_fusion_baseline_ = 0;
    active_sensing_fusion_current_ = 0;
    active_sensing_fusion_distinct_ = 0;
    active_sensing_start_yaw_ = 0.0;
    active_sensing_target_yaw_ = 0.0;
    active_sensing_settle_yaw_error_ = 0.0;
    active_sensing_settle_angular_speed_ = 0.0;
    active_sensing_measured_stable_duration_sec_ = 0.0;
    active_sensing_trajectory_duration_sec_ = 0.0;
    active_sensing_start_position_.setZero();
    active_sensing_expected_identity_ = ActiveSensingTrajectoryIdentity{};
    odom_pos_.setZero();
    odom_vel_.setZero();
    filtered_odom_vel_.setZero();
    reference_velocity_filter_.reset();
    odom_acc_.setZero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_angular_speed_ = 0.0;
    odom_yaw_rate_ = 0.0;
    last_freeze_update_time_ = node_->now();
    reference_retry_not_before_ = node_->now();

    /*  fsm param  */
    navi_mode_ = load_parameter<int>(node_, "fsm.navi_mode", -1);
    replan_thresh_ = load_parameter<double>(node_, "fsm.thresh_replan", -1.0);
    no_replan_thresh_ = load_parameter<double>(node_, "fsm.thresh_no_replan", -1.0);
    planning_horizon_ = load_parameter<double>(node_, "fsm.planning_horizon", -1.0);
    emergency_time_ = load_parameter<double>(node_, "fsm.emergency_time", 1.0);
    enable_fail_safe_ = load_parameter<bool>(node_, "fsm.fail_safe", true);
    max_replan_fail_count_ = load_parameter<int>(node_, "fsm.max_replan_fail_count", 1000);
    self_inflation_z_up_ = load_parameter<double>(node_, "grid_map.obstacles_inflation_z_up", 0.0);
    self_inflation_z_down_ = load_parameter<double>(node_, "grid_map.obstacles_inflation_z_down", 0.0);
    self_double_cylinder_radius_ = load_parameter<double>(node_, "grid_map.double_cylinder_radius", 0.0);
    self_double_cylinder_offset_ = load_parameter<double>(node_, "grid_map.double_cylinder_offset", 0.0);
    body_height_ = load_parameter<double>(node_, "grid_map.body_height", 0.30);
    self_inflation_frame_id_ = load_parameter<std::string>(node_, "grid_map.frame_id", "world");
    expected_frame_id_ = load_parameter<std::string>(node_, "grid_map.frame_id", "world");
    expected_base_frame_id_ = load_parameter<std::string>(node_, "grid_map.base_frame_id", "base_link");
    stair_execution_frozen_topic_ = load_parameter<std::string>(
        node_, "topics.stair_execution_frozen",
        "/planning/stair_execution_frozen");
    controller_status_topic_ = load_parameter<std::string>(
        node_, "topics.controller_status", "/planning/controller_status");
    const int controller_status_qos_depth = load_parameter<int>(
        node_, "qos.controller_status_depth",
        kControllerStatusEvidenceDepth);
    planning_status_topic_ = load_parameter<std::string>(
        node_, "topics.planning_status", "/planning/scan_status");
    bspline_diagnostics_topic_ = load_parameter<std::string>(
        node_, "topics.bspline_diagnostics",
        "/planning/bspline_diagnostics");
    trajectory_diagnostic_max_samples_ = load_parameter<int>(
        node_, "fsm.trajectory_diagnostic_max_samples", 64);
    trajectory_diagnostic_history_depth_ = load_parameter<int>(
        node_, "fsm.trajectory_diagnostic_history_depth", 64);
    input_timeout_sec_ = load_parameter<double>(node_, "fsm.input_timeout_sec", 0.5);
    stair_execution_freeze_timeout_sec_ = load_parameter<double>(
        node_, "fsm.stair_execution_freeze_timeout_sec", 0.25);
    stair_execution_freeze_confirmation_sec_ = load_parameter<double>(
        node_, "fsm.stair_execution_freeze_confirmation_sec", 0.05);
    min_path_point_spacing_ = load_parameter<double>(node_, "fsm.min_path_point_spacing", 0.05);
    reference_projection_max_distance_ =
        load_parameter<double>(
            node_, "fsm.reference_projection_max_distance", 0.5);
    reference_target_free_runway_ =
        load_parameter<double>(
            node_, "fsm.reference_target_free_runway", 0.10);
    reference_cruise_speed_ = load_parameter<double>(
        node_, "fsm.reference_cruise_speed", 0.0);
    reference_velocity_filter_time_constant_sec_ = load_parameter<double>(
        node_, "fsm.reference_velocity_filter_time_constant_sec", 0.0);
    max_reference_path_points_ =
        load_parameter<int>(node_, "fsm.max_reference_path_points", 4096);
    reference_retry_period_sec_ =
        load_parameter<double>(node_, "fsm.reference_retry_period_sec", 0.5);
    final_trajectory_convergence_grace_sec_ = load_parameter<double>(
        node_, "fsm.final_trajectory_convergence_grace_sec", 2.0);
    reference_goal_hold_distance_xy_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_distance_xy", 0.04);
    reference_goal_hold_distance_z_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_distance_z", 0.12);
    reference_goal_hold_yaw_error_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_yaw_error", 0.18);
    reference_goal_hold_planar_speed_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_planar_speed", 0.05);
    reference_goal_hold_vertical_speed_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_vertical_speed", 0.05);
    reference_goal_hold_yaw_rate_ =
        load_parameter<double>(node_, "fsm.reference_goal_hold_yaw_rate", 0.10);
    reference_goal_hold_stable_dwell_sec_ = load_parameter<double>(
        node_, "fsm.reference_goal_hold_stable_dwell_sec", 0.75);
    enable_active_sensing_ =
        load_parameter<bool>(node_, "fsm.enable_active_sensing", false);
    active_sensing_yaw_offset_ =
        load_parameter<double>(node_, "fsm.active_sensing_yaw_offset", 0.20);
    active_sensing_yaw_rate_ =
        load_parameter<double>(node_, "fsm.active_sensing_yaw_rate", 0.20);
    active_sensing_max_planar_speed_ = load_parameter<double>(
        node_, "fsm.active_sensing_max_planar_speed", 0.03);
    active_sensing_yaw_error_ = load_parameter<double>(
        node_, "fsm.active_sensing_yaw_error", 0.02);
    active_sensing_max_angular_speed_ = load_parameter<double>(
        node_, "fsm.active_sensing_max_angular_speed", 0.05);
    active_sensing_stable_duration_sec_ = load_parameter<double>(
        node_, "fsm.active_sensing_stable_duration_sec", 0.10);
    active_sensing_max_position_drift_ = load_parameter<double>(
        node_, "fsm.active_sensing_max_position_drift", 0.04);
    active_sensing_accept_timeout_sec_ = load_parameter<double>(
        node_, "fsm.active_sensing_accept_timeout_sec", 0.50);
    active_sensing_observation_timeout_sec_ = load_parameter<double>(
        node_, "fsm.active_sensing_observation_timeout_sec", 1.00);
    active_sensing_safety_margin_sec_ = load_parameter<double>(
        node_, "fsm.active_sensing_safety_margin_sec", 0.25);
    if (expected_frame_id_.empty() || expected_base_frame_id_.empty() ||
        stair_execution_frozen_topic_.empty() || controller_status_topic_.empty() ||
        planning_status_topic_.empty() ||
        bspline_diagnostics_topic_.empty())
      throw std::runtime_error(
          "SCAN 输入 frame_id、base_frame_id 与状态 topic 不能为空");
    validateControllerStatusEvidenceDepth(controller_status_qos_depth);
    // typed 诊断与 Isaac OGN/validator 共用固定 64 点、KeepLast(64)
    // 合同，避免 planner 接受审计端无法解释的另一套有界参数。
    if (trajectory_diagnostic_max_samples_ != 64 ||
        trajectory_diagnostic_history_depth_ != 64)
      throw std::runtime_error(
          "B-spline 诊断样本上限与历史深度必须固定为 64");
    if (input_timeout_sec_ <= 0.0 ||
        !std::isfinite(stair_execution_freeze_timeout_sec_) ||
        stair_execution_freeze_timeout_sec_ <= 0.0 ||
        !std::isfinite(stair_execution_freeze_confirmation_sec_) ||
        stair_execution_freeze_confirmation_sec_ <= 0.0 ||
        !std::isfinite(min_path_point_spacing_) ||
        min_path_point_spacing_ <= 0.0 ||
        !std::isfinite(reference_projection_max_distance_) ||
        reference_projection_max_distance_ <= 0.0 ||
        !std::isfinite(reference_target_free_runway_) ||
        reference_target_free_runway_ < min_path_point_spacing_ ||
        !std::isfinite(reference_cruise_speed_) ||
        reference_cruise_speed_ < 0.0 ||
        !std::isfinite(reference_velocity_filter_time_constant_sec_) ||
        reference_velocity_filter_time_constant_sec_ < 0.0 ||
        reference_velocity_filter_time_constant_sec_ > 1.0 ||
        max_reference_path_points_ < 2 || max_replan_fail_count_ < 1)
      throw std::runtime_error(
          "SCAN 输入/楼梯冻结超时与确认宽限、路径点距、投影距离、"
          "目标自由支撑区、巡航速度、速度滤波时间常数、Path 点数或失败阈值非法");
    if (!std::isfinite(planning_horizon_) || planning_horizon_ <= 0.0 ||
        !std::isfinite(replan_thresh_) || replan_thresh_ <= 0.0 ||
        !std::isfinite(no_replan_thresh_) || no_replan_thresh_ <= 0.0 ||
        replan_thresh_ >= planning_horizon_ ||
        no_replan_thresh_ >= replan_thresh_)
      throw std::runtime_error(
          "SCAN 前视距离必须大于重规划距离，重规划距离必须大于终点保持距离");
    if (reference_retry_period_sec_ <= 0.0 ||
        !std::isfinite(final_trajectory_convergence_grace_sec_) ||
        final_trajectory_convergence_grace_sec_ <= 0.0 ||
        final_trajectory_convergence_grace_sec_ > 3.0 ||
        reference_goal_hold_distance_xy_ <= 0.0 ||
        reference_goal_hold_distance_z_ <= 0.0 ||
        reference_goal_hold_yaw_error_ <= 0.0 ||
        reference_goal_hold_planar_speed_ < 0.0 ||
        reference_goal_hold_vertical_speed_ < 0.0 ||
        reference_goal_hold_yaw_rate_ < 0.0 ||
        !std::isfinite(reference_goal_hold_stable_dwell_sec_) ||
        reference_goal_hold_stable_dwell_sec_ <= 0.0 ||
        reference_goal_hold_stable_dwell_sec_ > 3.0)
      throw std::runtime_error(
          "SCAN reference 重试周期、末段收敛宽限与终点收口门限非法");
    const double active_sensing_yaw_duration =
        std::abs(active_sensing_yaw_offset_) / active_sensing_yaw_rate_;
    const double active_sensing_total_duration =
        active_sensing_accept_timeout_sec_ + active_sensing_yaw_duration +
        active_sensing_observation_timeout_sec_ +
        active_sensing_safety_margin_sec_;
    if (!std::isfinite(active_sensing_yaw_offset_) ||
        std::abs(active_sensing_yaw_offset_) < 0.01 ||
        std::abs(active_sensing_yaw_offset_) >
            kActiveSensingMaximumYawOffset ||
        !std::isfinite(active_sensing_yaw_rate_) ||
        active_sensing_yaw_rate_ < 0.01 ||
        active_sensing_yaw_rate_ > kActiveSensingMaximumYawRate ||
        !std::isfinite(active_sensing_max_planar_speed_) ||
        active_sensing_max_planar_speed_ < 0.0 ||
        active_sensing_max_planar_speed_ > 0.20 ||
        !std::isfinite(active_sensing_yaw_error_) ||
        active_sensing_yaw_error_ <= 0.0 ||
        active_sensing_yaw_error_ >
            kActiveSensingMaximumSettleYawError ||
        active_sensing_yaw_error_ >=
            std::abs(active_sensing_yaw_offset_) ||
        !std::isfinite(active_sensing_max_angular_speed_) ||
        active_sensing_max_angular_speed_ < 0.0 ||
        active_sensing_max_angular_speed_ >
            kActiveSensingMaximumSettleAngularSpeed ||
        !std::isfinite(active_sensing_stable_duration_sec_) ||
        active_sensing_stable_duration_sec_ <
            kActiveSensingMinimumStableDuration ||
        active_sensing_stable_duration_sec_ > 1.0 ||
        !std::isfinite(active_sensing_max_position_drift_) ||
        active_sensing_max_position_drift_ <= 0.0 ||
        active_sensing_max_position_drift_ > 0.20 ||
        !std::isfinite(active_sensing_accept_timeout_sec_) ||
        active_sensing_accept_timeout_sec_ < 0.05 ||
        active_sensing_accept_timeout_sec_ > 2.0 ||
        !std::isfinite(active_sensing_observation_timeout_sec_) ||
        active_sensing_observation_timeout_sec_ < 0.10 ||
        active_sensing_observation_timeout_sec_ > 5.0 ||
        !std::isfinite(active_sensing_safety_margin_sec_) ||
        active_sensing_safety_margin_sec_ < 0.05 ||
        active_sensing_safety_margin_sec_ > 2.0 ||
        !std::isfinite(active_sensing_total_duration) ||
        active_sensing_total_duration <= 0.0 ||
        active_sensing_total_duration > 10.0)
      throw std::runtime_error(
          "SCAN ACTIVE_SENSING yaw、稳定性、漂移或超时参数非法");

    if (navi_mode_ == NAVI_MODE::PRESET_TARGET)
    {
      const auto flat_waypoints = load_parameter<std::vector<double>>(node_, "fsm.waypoints", {});
      if (flat_waypoints.empty() || flat_waypoints.size() % 3 != 0)
        throw std::runtime_error("navi_mode=2 requires non-empty fsm.waypoints with x,y,z triples");
      waypoint_num_ = static_cast<int>(flat_waypoints.size() / 3);
      preset_waypoints_.resize(waypoint_num_);
      for (int i = 0; i < waypoint_num_; i++)
      {
        preset_waypoints_[i] = Eigen::Vector3d(flat_waypoints[3 * i], flat_waypoints[3 * i + 1],
                                               flat_waypoints[3 * i + 2]);
      }
    }

    /* initialize main modules */
    visualization_.reset(new PlanningVisualization(node_));
    planner_manager_.reset(new SCANPlannerManager);
    planner_manager_->initPlanModules(node_, visualization_);
    if (reference_cruise_speed_ > planner_manager_->pp_.max_vel_ + 1.0e-9)
      throw std::runtime_error(
          "fsm.reference_cruise_speed 不能大于 manager.max_vel");
    RCLCPP_INFO(
        node_->get_logger(),
        "SCAN reference 调参：前视 %.2f m，重规划 %.2f m，巡航 %.2f m/s，"
        "实测速度滤波 %.2f s，规划速度上限 %.2f m/s",
        planning_horizon_, replan_thresh_, reference_cruise_speed_,
        reference_velocity_filter_time_constant_sec_,
        planner_manager_->pp_.max_vel_);

    /* callback */
    exec_timer_ = rclcpp::create_timer(
        node_, node_->get_clock(), rclcpp::Duration::from_seconds(0.01),
        std::bind(&SCANReplanFSM::execFSMCallback, this));
    safety_timer_ = rclcpp::create_timer(
        node_, node_->get_clock(), rclcpp::Duration::from_seconds(0.05),
        std::bind(&SCANReplanFSM::checkCollisionCallback, this));
    odometry_callback_group_ = node_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions odometry_options;
    odometry_options.callback_group = odometry_callback_group_;
    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        "body_pose", rclcpp::SensorDataQoS(),
        std::bind(&SCANReplanFSM::odometryCallback, this, std::placeholders::_1),
        odometry_options);
    go2_execution_frozen_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
        "planning/go2_execution_frozen", 10,
        std::bind(&SCANReplanFSM::go2ExecutionFrozenCallback, this, std::placeholders::_1));
    controller_status_sub_ =
        node_->create_subscription<ControllerStatusMessage>(
            controller_status_topic_,
            makeControllerStatusEvidenceQos(),
            std::bind(
                &SCANReplanFSM::controllerStatusCallback, this,
                std::placeholders::_1));
    stair_execution_frozen_callback_group_ = node_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions stair_execution_frozen_options;
    stair_execution_frozen_options.callback_group =
        stair_execution_frozen_callback_group_;
    stair_execution_frozen_sub_ =
        node_->create_subscription<scan_planner_msgs::msg::StairExecutionFreeze>(
            stair_execution_frozen_topic_,
            rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
            std::bind(
                &SCANReplanFSM::stairExecutionFrozenCallback, this,
                std::placeholders::_1),
            stair_execution_frozen_options);

    // 轨迹 payload 与 typed planning status 必须支持 late join 三方对账；
    // 这里只保存当前轨迹，旧代 payload 由严格 identity gate 拒绝。
    bspline_pub_ =
        node_->create_publisher<scan_planner_msgs::msg::Bspline>(
            "planning/bspline",
            rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    bspline_diagnostics_pub_ =
        node_->create_publisher<
            scan_planner_msgs::msg::BsplineDiagnostics>(
            bspline_diagnostics_topic_,
            rclcpp::QoS(rclcpp::KeepLast(
                static_cast<std::size_t>(
                    trajectory_diagnostic_history_depth_)))
                .reliable()
                .transient_local());
    data_disp_pub_ = node_->create_publisher<scan_planner_msgs::msg::DataDisp>("planning/data_display", 100);
    planning_status_pub_ =
        node_->create_publisher<scan_planner_msgs::msg::ScanPlanningStatus>(
            planning_status_topic_,
            rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    self_inflation_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
        "self_inflation", rclcpp::QoS(1).reliable().transient_local());

    if (navi_mode_ == NAVI_MODE::MANUAL_TARGET)
      goal_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
          "move_base_simple/goal", 1,
          std::bind(&SCANReplanFSM::rvizGoalCallback, this, std::placeholders::_1));
    else if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
      path_sub_ = node_->create_subscription<nav_msgs::msg::Path>(
          "initial_path", rclcpp::QoS(1).reliable().transient_local(),
          std::bind(&SCANReplanFSM::pathCallback, this, std::placeholders::_1));
    else if (navi_mode_ == NAVI_MODE::PRESET_TARGET)
      RCLCPP_INFO(node_->get_logger(), "Preset waypoint mode will start after the first odometry message");
    else
      throw std::runtime_error("fsm.navi_mode must be 1, 2, or 3");
  }

  void SCANReplanFSM::planGlobalTrajbyGivenWps()
  {
    std::vector<Eigen::Vector3d> wps = preset_waypoints_;

    for (size_t i = 0; i < wps.size(); i++)
    {
      visualization_->displayGoalPoint(wps[i], Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, i);
    }

    active_waypoints_ = wps;
    current_wp_ = 0;
    trigger_ = true;
    init_pt_ = odom_pos_;

    if (planNextWaypoint())
    {
      changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Unable to generate global trajectory to first preset waypoint");
    }
  }

  void SCANReplanFSM::rvizGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &msg)
  {
    if (!msg)
      return;

    if (!rviz_height_ready_)
    {
      RCLCPP_WARN(node_->get_logger(), "Ignore RViz goal before receiving initial body pose");
      return;
    }

    auto path = std::make_shared<nav_msgs::msg::Path>();
    path->header = msg->header;
    path->poses.push_back(*msg);
    waypointCallback(path);
  }

  void SCANReplanFSM::waypointCallback(const nav_msgs::msg::Path::ConstSharedPtr &msg)
  {
    if (!msg || msg->poses.empty())
    {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                           "Empty waypoint message; ignoring");
      return;
    }

    if (msg->poses[0].pose.position.z < -0.1)
      return;

    cout << "Triggered!" << endl;
    trigger_ = true;
    init_pt_ = odom_pos_;

    bool success = false;
    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, rviz_goal_height_;
    success = planner_manager_->planGlobalTraj(odom_pos_, odom_vel_, Eigen::Vector3d::Zero(), end_pt_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    if (success)
      success = adjustGlobalTargetIfOccupied();

    visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, 0);

    if (success)
    {

      /*** display ***/
      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;

      /*** FSM ***/
      if (exec_state_ == WAIT_TARGET)
        changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
      else if (exec_state_ == EXEC_TRAJ)
        changeFSMExecState(REPLAN_TRAJ, "TRIG");

      // visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(1, 0, 0, 1), 0.3, 0);
      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Unable to generate global trajectory");
    }
  }

  bool SCANReplanFSM::planGlobalTrajByWaypoints(const std::vector<Eigen::Vector3d> &waypoints)
  {
    if (waypoints.empty())
    {
      RCLCPP_WARN(node_->get_logger(), "No waypoint supplied for global trajectory");
      return false;
    }

    end_pt_ = waypoints.back();

    for (size_t i = 0; i < waypoints.size(); i++)
    {
      visualization_->displayGoalPoint(waypoints[i], Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, i);
    }

    bool success = planner_manager_->planGlobalTrajWaypoints(
        odom_pos_,
        odom_vel_,
        Eigen::Vector3d::Zero(),
        waypoints,
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero());

    if (!success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Unable to generate global trajectory from waypoints");
      return false;
    }

    if (!adjustGlobalTargetIfOccupied())
      return false;

    constexpr double step_size_t = 0.1;
    int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
    std::vector<Eigen::Vector3d> gloabl_traj(i_end);
    for (int i = 0; i < i_end; i++)
    {
      gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
    }

    end_vel_.setZero();
    have_target_ = true;
    have_new_target_ = true;
    visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, static_cast<int>(waypoints.size()) - 1);

    return true;
  }

  bool SCANReplanFSM::planNextWaypoint()
  {
    if (current_wp_ < 0 || current_wp_ >= (int)active_waypoints_.size())
    {
      RCLCPP_WARN(node_->get_logger(), "[navi_mode=%d] No active waypoint to plan", navi_mode_);
      return false;
    }

    end_pt_ = active_waypoints_[current_wp_];
    setStartStateFromOdomOrCurrentTraj();

    bool success = planner_manager_->planGlobalTraj(
        start_pt_,
        start_vel_,
        start_acc_,
        end_pt_,
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero());

    if (!success)
    {
      RCLCPP_ERROR(node_->get_logger(), "[navi_mode=%d] Unable to generate trajectory to waypoint %d",
                   navi_mode_, current_wp_ + 1);
      return false;
    }

    if (!adjustGlobalTargetIfOccupied())
      return false;

    constexpr double step_size_t = 0.1;
    int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
    std::vector<Eigen::Vector3d> gloabl_traj(i_end);
    for (int i = 0; i < i_end; i++)
    {
      gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
    }

    end_vel_.setZero();
    have_target_ = true;
    have_new_target_ = true;
    visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, current_wp_);
    RCLCPP_INFO(node_->get_logger(), "[navi_mode=%d] Planning to waypoint %d/%zu: [%.2f, %.2f, %.2f]",
                navi_mode_, current_wp_ + 1, active_waypoints_.size(), end_pt_(0), end_pt_(1), end_pt_(2));

    return true;
  }

  bool SCANReplanFSM::isWaypointSequenceMode() const
  {
    return navi_mode_ == NAVI_MODE::PRESET_TARGET;
  }

  bool SCANReplanFSM::adjustGlobalTargetIfOccupied()
  {
    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
    {
      // 完整参考终点由 PCT 或 /initial_path 所有。局部滑窗既不等于已观测
      // 空间，也无权把跨楼层目标永久截成当前楼梯下方的临时自由点。
      return true;
    }

    auto map = planner_manager_->grid_map_;
    auto &global_data = planner_manager_->global_data_;
    const double duration = global_data.global_duration_;
    if (!map || duration < 1e-3)
      return true;

    constexpr double sample_dt = 0.05;
    const int sample_num = std::max(1, static_cast<int>(std::ceil(duration / sample_dt)));
    const Eigen::Vector3d final_pt = global_data.global_traj_.evaluate(duration);
    const Eigen::Vector3d final_prev = global_data.global_traj_.evaluate(duration * (sample_num - 1) / sample_num);
    const int final_occ = map->getInflateOccupancy(final_pt, estimateYawFromSegment(final_prev, final_pt));
    if (!shouldAdjustGlobalTargetForLocalOccupancy(false, final_occ))
      return true;

    for (int i = sample_num; i >= 0; --i)
    {
      const double t = duration * i / sample_num;
      const double prev_t = duration * std::max(0, i - 1) / sample_num;
      const Eigen::Vector3d pt = global_data.global_traj_.evaluate(t);
      const Eigen::Vector3d prev_pt = global_data.global_traj_.evaluate(prev_t);

      if (map->getInflateOccupancy(pt, estimateYawFromSegment(prev_pt, pt)) == 0)
      {
        const Eigen::Vector3d raw_end = end_pt_;
        end_pt_ = pt;
        global_data.global_duration_ = t;
        global_data.last_progress_time_ = std::min(global_data.last_progress_time_, t);
        RCLCPP_WARN(node_->get_logger(),
                    "Target [%.2f, %.2f, %.2f] is occupied; using [%.2f, %.2f, %.2f]",
                    raw_end(0), raw_end(1), raw_end(2), end_pt_(0), end_pt_(1), end_pt_(2));
        return true;
      }
    }

    RCLCPP_ERROR(node_->get_logger(),
                 "Target is occupied and no collision-free point was found on the global trajectory");
    return false;
  }

  void SCANReplanFSM::pathCallback(const nav_msgs::msg::Path::ConstSharedPtr &msg)
  {
    if (!msg)
      return;
    const int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    if (
      msg->header.frame_id != expected_frame_id_ ||
      stamp_ns <= 0)
    {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                           "忽略 frame 或时间戳无效的 initial_path");
      return;
    }
    if (
      latest_reference_path_stamp_ns_ > 0 &&
      stamp_ns < latest_reference_path_stamp_ns_)
    {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                           "忽略早于当前代际的 initial_path");
      return;
    }
    if (global_replan_generation_gate_.required() &&
        !global_replan_generation_gate_.isStrictReplacement(stamp_ns))
    {
      RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "等待 PCT 严格更新 Path 代际，禁止旧 initial_path 自动恢复");
      return;
    }
    if (msg->poses.empty())
    {
      if (latest_reference_generation_is_empty_ &&
          stamp_ns == latest_reference_path_stamp_ns_)
      {
        RCLCPP_INFO(node_->get_logger(),
                    "忽略同 stamp 的空 initial_path DDS 重发");
        return;
      }
      const bool cleared_global_replan =
          global_replan_generation_gate_.clearForStrictReplacement(stamp_ns);
      if (active_sensing_phase_ != ACTIVE_SENSING_IDLE)
      {
        const std::string active_failure_reason =
            "主动感知期间收到 Path tombstone";
        pending_emergency_reason_ =
            "主动感知期间收到 Path tombstone，先发布严格零速轨迹";
        const ActiveSensingReplacementResult replacement_result =
            terminateActiveSensingBeforeTrajectoryReplacement(
                [this, &active_failure_reason]() {
                  return publishActiveSensingDiagnostics(
                      scan_planner_msgs::msg::BsplineDiagnostics::
                          ACTIVE_SENSING_EVENT_FAILED,
                      active_failure_reason);
                },
                [this]() { resetActiveSensingRuntime(); },
                [this]() { return callEmergencyStop(odom_pos_); });
        if (!replacement_result.failure_snapshot_published)
        {
          RCLCPP_ERROR(
              node_->get_logger(),
              "Path tombstone 前无法发布主动观测 FAILED 快照");
        }
        if (!replacement_result.stop_published)
          RCLCPP_ERROR(node_->get_logger(), "Path tombstone 严格停车失败");
      }
      latest_reference_path_stamp_ns_ = stamp_ns;
      latest_reference_generation_is_empty_ = true;
      pending_reference_waypoints_.clear();
      active_reference_waypoints_.clear();
      active_reference_arc_lengths_.clear();
      local_reference_guide_.clear();
      local_reference_corridor_guide_.clear();
      reference_progress_s_ = 0.0;
      pending_reference_goal_yaw_ = 0.0;
      active_reference_goal_yaw_ = 0.0;
      pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
      active_reference_path_stamp_ = builtin_interfaces::msg::Time();
      resetActiveSensingRuntime();
      active_sensing_consumed_path_stamp_ns_ = 0;
      have_target_ = false;
      trigger_ = false;
      have_new_target_ = false;
      local_target_is_final_ = false;
      reference_final_hold_lifecycle_ = FinalHoldLifecycleState{};
      replan_fail_count_ = 0;
      recovery_status_pending_ = false;
      need_hover_stop_ = false;
      have_last_published_trajectory_ = false;
      published_final_trajectory_history_.clear();
      controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
      have_controller_accepted_final_identity_ = false;
      stair_planning_freeze_gate_.clearReferenceGeneration();
      if (exec_state_ != INIT)
        changeFSMExecState(WAIT_TARGET, "EMPTY_REFERENCE_PATH");
      publishPlanningStatus(
          ScanPlanningEvent::kReferenceCleared,
          cleared_global_replan ?
              "收到严格更新的 Path tombstone，旧重规划代际已清除" :
              "收到 Path tombstone，已清除参考路径与规划目标");
      RCLCPP_WARN(node_->get_logger(),
                  "收到空 initial_path，已清除参考路径与规划目标");
      return;
    }

    if (msg->poses.size() >
        static_cast<std::size_t>(max_reference_path_points_))
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "拒绝包含 %zu 点的 initial_path：超过线性处理上限 %d",
          msg->poses.size(), max_reference_path_points_);
      return;
    }

    std::vector<Eigen::Vector3d> waypoints;
    double goal_yaw = 0.0;
    if (!validateAndConvertReferencePath(*msg, waypoints, goal_yaw))
    {
      RCLCPP_ERROR(node_->get_logger(), "拒绝无效 initial_path");
      return;
    }
    const int64_t pending_stamp_ns =
        rclcpp::Time(pending_reference_path_stamp_).nanoseconds();
    const int64_t active_stamp_ns =
        rclcpp::Time(active_reference_path_stamp_).nanoseconds();
    if (
      (stamp_ns == pending_stamp_ns &&
       same_waypoints(waypoints, pending_reference_waypoints_) &&
       same_yaw(goal_yaw, pending_reference_goal_yaw_)) ||
      (stamp_ns == active_stamp_ns &&
       same_waypoints(waypoints, active_reference_waypoints_) &&
       same_yaw(goal_yaw, active_reference_goal_yaw_)))
    {
      latest_reference_path_stamp_ns_ = stamp_ns;
      RCLCPP_INFO(node_->get_logger(),
                  "忽略同 stamp、同 payload 的 initial_path DDS 重发");
      return;
    }
    if (
      latest_reference_path_stamp_ns_ > 0 &&
      stamp_ns == latest_reference_path_stamp_ns_)
    {
      // reference_path_stamp 是 B-spline 唯一携带的 Path 代际。同一 stamp
      // 出现不同几何或 terminal yaw 时，迟到的旧轨迹无法区分；作废整个
      // 冲突代际并等待严格更新的 Path，不能继续沿任一候选运动。
      if (active_sensing_phase_ != ACTIVE_SENSING_IDLE)
      {
        const std::string active_failure_reason =
            "主动感知期间 Path 同代冲突";
        pending_emergency_reason_ =
            "主动感知期间 Path 同代冲突，先发布严格零速轨迹";
        const ActiveSensingReplacementResult replacement_result =
            terminateActiveSensingBeforeTrajectoryReplacement(
                [this, &active_failure_reason]() {
                  return publishActiveSensingDiagnostics(
                      scan_planner_msgs::msg::BsplineDiagnostics::
                          ACTIVE_SENSING_EVENT_FAILED,
                      active_failure_reason);
                },
                [this]() { resetActiveSensingRuntime(); },
                [this]() { return callEmergencyStop(odom_pos_); });
        if (!replacement_result.failure_snapshot_published)
        {
          RCLCPP_ERROR(
              node_->get_logger(),
              "Path 同代冲突前无法发布主动观测 FAILED 快照");
        }
        if (!replacement_result.stop_published)
          RCLCPP_ERROR(node_->get_logger(), "Path 同代冲突严格停车失败");
      }
      pending_reference_waypoints_.clear();
      active_reference_waypoints_.clear();
      active_reference_arc_lengths_.clear();
      local_reference_guide_.clear();
      local_reference_corridor_guide_.clear();
      reference_progress_s_ = 0.0;
      pending_reference_goal_yaw_ = 0.0;
      active_reference_goal_yaw_ = 0.0;
      pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
      active_reference_path_stamp_ = builtin_interfaces::msg::Time();
      resetActiveSensingRuntime();
      have_target_ = false;
      trigger_ = false;
      have_new_target_ = false;
      local_target_is_final_ = false;
      reference_final_hold_lifecycle_ = FinalHoldLifecycleState{};
      need_hover_stop_ = false;
      latest_reference_generation_is_empty_ = true;
      recovery_status_pending_ = false;
      have_last_published_trajectory_ = false;
      published_final_trajectory_history_.clear();
      controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
      have_controller_accepted_final_identity_ = false;
      stair_planning_freeze_gate_.clearReferenceGeneration();
      if (exec_state_ != INIT)
        changeFSMExecState(WAIT_TARGET, "CONFLICTING_REFERENCE_GENERATION");
      publishPlanningStatus(
          ScanPlanningEvent::kReferenceCleared,
          "同一 Path stamp 出现冲突 payload，已作废整个代际");
      RCLCPP_ERROR(
          node_->get_logger(),
          "同一 initial_path 代际出现不同几何或 terminal yaw；已停车等待新 stamp");
      return;
    }
    latest_reference_path_stamp_ns_ = stamp_ns;
    latest_reference_generation_is_empty_ = false;
    pending_reference_waypoints_ = std::move(waypoints);
    pending_reference_goal_yaw_ = goal_yaw;
    pending_reference_path_stamp_ = msg->header.stamp;
    // pending Path 已完成 frame、时间戳和几何校验，只是等待首帧地图。
    // Isaac 会立即按该最新代际发布楼梯冻结快照，因此此处先绑定精确
    // stamp 并保持 fail-closed；pending 转为 active 时同 stamp 不会清空协议。
    stair_planning_freeze_gate_.bindReferenceGeneration(stamp_ns);
    tryActivatePendingReferencePath();
  }

  bool SCANReplanFSM::validateAndConvertReferencePath(
      const nav_msgs::msg::Path &path,
      std::vector<Eigen::Vector3d> &waypoints,
      double &goal_yaw) const
  {
    if (path.header.frame_id != expected_frame_id_ ||
        rclcpp::Time(path.header.stamp).nanoseconds() <= 0 ||
        path.poses.size() < 2)
      return false;

    waypoints.clear();
    waypoints.reserve(path.poses.size());
    for (const auto &pose_stamped : path.poses)
    {
      const std::string &pose_frame = pose_stamped.header.frame_id;
      if ((!pose_frame.empty() && pose_frame != expected_frame_id_) ||
          rclcpp::Time(pose_stamped.header.stamp).nanoseconds() <= 0 ||
          !finite_pose(pose_stamped.pose))
        return false;

      Eigen::Vector3d waypoint(
          pose_stamped.pose.position.x,
          pose_stamped.pose.position.y,
          // 输入 Path 的 z 是地面高度，本体高度只允许在此处增加一次。
          pose_stamped.pose.position.z + body_height_);
      if (!waypoints.empty() &&
          (waypoint - waypoints.back()).norm() < min_path_point_spacing_)
        continue;
      waypoints.push_back(waypoint);
    }

    if (waypoints.size() < 2)
      return false;

    const auto &orientation = path.poses.back().pose.orientation;
    Eigen::Quaterniond terminal_orientation(
        orientation.w, orientation.x, orientation.y, orientation.z);
    terminal_orientation.normalize();
    const Eigen::Vector3d heading =
        terminal_orientation * Eigen::Vector3d::UnitX();
    if (!heading.allFinite() || heading.head<2>().squaredNorm() <= 1.0e-12)
      return false;
    goal_yaw = std::atan2(heading(1), heading(0));
    return std::isfinite(goal_yaw);
  }

  bool SCANReplanFSM::referenceInputsReady() const
  {
    if (!have_odom_ || last_odom_stamp_ns_ <= 0 ||
        !planner_manager_ || !planner_manager_->grid_map_ ||
        !planner_manager_->grid_map_->observationReady())
      return false;

    const int64_t now_ns = node_->now().nanoseconds();
    const int64_t age_ns = now_ns - last_odom_stamp_ns_;
    const int64_t timeout_ns =
        static_cast<int64_t>(input_timeout_sec_ * 1.0e9);
    return now_ns > 0 && age_ns >= 0 && age_ns <= timeout_ns;
  }

  bool SCANReplanFSM::stairResumeInputsReady() const
  {
    const int64_t reference_stamp_ns =
        rclcpp::Time(active_reference_path_stamp_).nanoseconds();
    const int64_t observation_stamp_ns =
        planner_manager_ && planner_manager_->grid_map_
            ? planner_manager_->grid_map_->observationStampNs()
            : 0;
    return stair_planning_freeze_gate_.resumeInputsReady(
        last_odom_stamp_ns_, observation_stamp_ns, reference_stamp_ns,
        referenceInputsReady(),
        !active_reference_waypoints_.empty() && have_target_);
  }

  void SCANReplanFSM::updateReferenceProgressDuringStairFreeze()
  {
    if (navi_mode_ != NAVI_MODE::REFERENCE_PATH || !have_odom_ ||
        active_reference_waypoints_.size() < 2 ||
        active_reference_arc_lengths_.size() !=
            active_reference_waypoints_.size())
      return;

    const ReferencePathProjection projection =
        advanceReferencePathProgressDuringStairFreeze(
            active_reference_waypoints_, active_reference_arc_lengths_,
            odom_pos_, reference_progress_s_,
            reference_projection_max_distance_);
    if (!projection.valid)
    {
      RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "楼梯暂停期间 Odometry 无法安全投影到剩余 reference Path；"
          "保持已认证进度");
      return;
    }
    reference_progress_s_ = std::max(
        reference_progress_s_, projection.progress_s);
  }

  void SCANReplanFSM::forceReferenceReplanAfterStairResume()
  {
    stair_planning_freeze_gate_.clearResumeBarrier();
    replan_fail_count_ = 0;
    recovery_status_pending_ = true;
    need_hover_stop_ = false;
    flag_escape_emergency_ = true;
    trigger_ = true;
    have_new_target_ = true;
    reference_retry_not_before_ = node_->now();
    changeFSMExecState(GEN_NEW_TRAJ, "STAIR_RESUME_FRESH_INPUTS");
    RCLCPP_INFO(
        node_->get_logger(),
        "楼梯冻结解除后已收到严格更新的 Odometry 与地图观测；"
        "从真实机体位姿强制重规划同代 initial_path");
  }

  void SCANReplanFSM::tryActivatePendingReferencePath()
  {
    if (pending_reference_waypoints_.empty())
      return;
    if (!referenceInputsReady())
    {
      RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "缓存 initial_path，等待同一 world frame 下的新鲜 Odometry 与首帧地图");
      return;
    }

    const std::vector<Eigen::Vector3d> accepted_waypoints =
        pending_reference_waypoints_;
    const builtin_interfaces::msg::Time accepted_stamp =
        pending_reference_path_stamp_;
    const double accepted_goal_yaw = pending_reference_goal_yaw_;
    const std::int64_t accepted_stamp_ns =
        rclcpp::Time(accepted_stamp).nanoseconds();
    const bool global_replan_recovery =
        global_replan_generation_gate_.required();
    if (global_replan_recovery &&
        !global_replan_generation_gate_.isStrictReplacement(
            accepted_stamp_ns))
    {
      RCLCPP_WARN(
          node_->get_logger(),
          "拒绝不严格晚于被阻断代际的 pending initial_path");
      pending_reference_waypoints_.clear();
      pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
      pending_reference_goal_yaw_ = 0.0;
      return;
    }
    const std::vector<double> accepted_arc_lengths =
        buildReferencePathArcLengths(accepted_waypoints);
    if (accepted_arc_lengths.empty())
    {
      pending_reference_waypoints_.clear();
      pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
      pending_reference_goal_yaw_ = 0.0;
      trigger_ = false;
      RCLCPP_ERROR(
          node_->get_logger(),
          "拒绝无法构建三维弧长的 initial_path");
      return;
    }

    pending_reference_waypoints_.clear();
    pending_reference_path_stamp_ = builtin_interfaces::msg::Time();
    pending_reference_goal_yaw_ = 0.0;

    if (active_sensing_phase_ != ACTIVE_SENSING_IDLE)
    {
      const std::string active_failure_reason =
          "主动感知期间收到严格新 Path，旧 identity 失效";
      pending_emergency_reason_ =
          "主动感知期间收到严格新 Path，先发布严格零速轨迹";
      const ActiveSensingReplacementResult replacement_result =
          terminateActiveSensingBeforeTrajectoryReplacement(
              [this, &active_failure_reason]() {
                return publishActiveSensingDiagnostics(
                    scan_planner_msgs::msg::BsplineDiagnostics::
                        ACTIVE_SENSING_EVENT_FAILED,
                    active_failure_reason);
              },
              [this]() { resetActiveSensingRuntime(); },
              [this]() { return callEmergencyStop(odom_pos_); });
      if (!replacement_result.failure_snapshot_published)
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "Path 换代前无法发布主动观测 FAILED 快照");
      }
      if (!replacement_result.stop_published)
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "主动感知 Path 换代时严格零速轨迹发布失败");
      }
    }
    if (active_sensing_consumed_path_stamp_ns_ != accepted_stamp_ns)
      active_sensing_consumed_path_stamp_ns_ = 0;

    // PCT 已经拥有跨楼层全局几何；SCAN 只沿完整折线滚动选择局部目标。
    // 这里禁止再拟合整条高阶多项式，否则楼梯回转和终点短反折会被切角。
    trigger_ = true;
    init_pt_ = odom_pos_;
    reference_retry_not_before_ = node_->now();
    active_reference_waypoints_ = accepted_waypoints;
    active_reference_arc_lengths_ = accepted_arc_lengths;
    local_reference_guide_.clear();
    local_reference_corridor_guide_.clear();
    reference_progress_s_ = 0.0;
    active_reference_goal_yaw_ = accepted_goal_yaw;
    active_reference_path_stamp_ = accepted_stamp;
    if (global_replan_recovery)
    {
      global_replan_generation_gate_.clearForStrictReplacement(
          accepted_stamp_ns);
      recovery_status_pending_ = true;
    }
    replan_fail_count_ = 0;
    reference_final_hold_lifecycle_ = FinalHoldLifecycleState{};
    // 新 Path 代际必须作废上一代的发布证据。否则迟到的旧 GOAL_REACHED
    // 可能在新目标已经激活后被误消费。
    have_last_published_trajectory_ = false;
    published_final_trajectory_history_.clear();
    controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
    have_controller_accepted_final_identity_ = false;
    reference_goal_hold_dwell_ = ReferenceGoalHoldDwellState{};
    stair_planning_freeze_gate_.bindReferenceGeneration(
        rclcpp::Time(active_reference_path_stamp_).nanoseconds());
    end_pt_ = active_reference_waypoints_.back();
    end_vel_.setZero();
    have_target_ = true;
    have_new_target_ = true;

    visualization_->displayGlobalPathList(
        active_reference_waypoints_, 0.1, 0);
    visualization_->displayGoalPoint(
        end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3,
        static_cast<int>(active_reference_waypoints_.size()) - 1);

    if (exec_state_ == WAIT_TARGET || exec_state_ == EMERGENCY_STOP)
      changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
    else if (exec_state_ == EXEC_TRAJ)
      changeFSMExecState(REPLAN_TRAJ, "TRIG");
    else if (exec_state_ == ACTIVE_SENSING)
      changeFSMExecState(GEN_NEW_TRAJ, "ACTIVE_SENSING_NEW_PATH");

    RCLCPP_INFO(
        node_->get_logger(),
        "已接受 %zu 点 initial_path（总弧长 %.3f m），"
        "将沿原始三维折线滚动选择 SCAN 局部目标",
        active_reference_waypoints_.size(),
        active_reference_arc_lengths_.back());

    publishPlanningStatus(
        ScanPlanningEvent::kReferenceAccepted,
        global_replan_recovery ?
            "已接受严格更新的 PCT Path，等待新局部轨迹" :
            "已接受新的 reference Path，等待局部轨迹");

    if (stair_planning_freeze_gate_.planningInhibited())
    {
      updateReferenceProgressDuringStairFreeze();
      // bindReferenceGeneration 会在首条精确快照到达前 fail-closed 冻结。
      // 该采集窗口不是已认证 ACK，不能发布合法 stair reason。若 pending
      // Path 期间已经认证过同 stamp 快照，则保留并重发精确合同 token。
      if (stair_planning_freeze_gate_.authenticatedSnapshotAvailableForStatus())
      {
        publishPlanningStatus(
            stair_planning_freeze_gate_.frozen() ?
                ScanPlanningEvent::kStairInhibited :
                ScanPlanningEvent::kStairResumeWaiting,
            stair_planning_freeze_gate_.frozen() ?
                kScanStairExecutionInhibitedReason :
                kScanStairResumeWaitingReason);
      }
    }
  }

  void SCANReplanFSM::odometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
  {
    odometry_mailbox_.push(msg);
  }

  void SCANReplanFSM::drainOdometryMailbox()
  {
    OdometryMailboxDrain drain = odometry_mailbox_.drain();
    if (drain.message)
      processOdometry(drain.message);
  }

  void SCANReplanFSM::processOdometry(
      const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
  {
    if (!msg)
      return;
    const int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    if (stamp_ns <= 0 ||
        msg->header.frame_id != expected_frame_id_ ||
        msg->child_frame_id != expected_base_frame_id_ ||
        !finite_pose(msg->pose.pose) ||
        !finite_twist(msg->twist.twist))
    {
      RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "拒绝 frame、时间戳、位姿或速度无效的 body_pose");
      return;
    }
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    if (navi_mode_ == NAVI_MODE::MANUAL_TARGET && !rviz_height_ready_)
    {
      rviz_goal_height_ = odom_pos_(2);
      rviz_height_ready_ = true;
      RCLCPP_INFO(node_->get_logger(), "Set RViz goal height from initial body_pose z: %.3f", rviz_goal_height_);
    }

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;
    odom_orient_.normalize();

    // Isaac direct Odometry 的 twist 是 base_link 机体系；SCAN 内部统一使用世界系。
    const Eigen::Vector3d body_velocity(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z);
    odom_vel_ = odom_orient_ * body_velocity;
    filtered_odom_vel_ = reference_velocity_filter_.update(
        odom_vel_, stamp_ns, reference_velocity_filter_time_constant_sec_);
    odom_angular_speed_ = std::sqrt(
        msg->twist.twist.angular.x * msg->twist.twist.angular.x +
        msg->twist.twist.angular.y * msg->twist.twist.angular.y +
        msg->twist.twist.angular.z * msg->twist.twist.angular.z);
    // Odometry twist 由 bridge 按 child_frame_id=base_link 发布；导航终点只
    // 控制机体系 yaw，不能让四足站立时的 roll/pitch 微摆阻断最终驻留。
    odom_yaw_rate_ = std::abs(msg->twist.twist.angular.z);

    have_odom_ = true;
    last_odom_stamp_ns_ = stamp_ns;
    if (stair_planning_freeze_gate_.planningInhibited())
      updateReferenceProgressDuringStairFreeze();
    publishSelfInflationMarker();
    tryActivatePendingReferencePath();
    updateReferenceGoalHoldDwell(stamp_ns);
    if (navi_mode_ == NAVI_MODE::PRESET_TARGET && !preset_started_)
    {
      preset_started_ = true;
      planGlobalTrajbyGivenWps();
    }
  }

  void SCANReplanFSM::go2ExecutionFrozenCallback(const std_msgs::msg::Bool::ConstSharedPtr &msg)
  {
    if (!msg)
      return;
    go2_execution_frozen_ = msg->data;
  }

  void SCANReplanFSM::controllerStatusCallback(
      const ControllerStatusMessage::ConstSharedPtr &msg)
  {
    if (!msg)
      return;
    if (active_sensing_phase_ != ACTIVE_SENSING_IDLE)
    {
      processActiveSensingControllerStatus(*msg);
      return;
    }
    if (!have_last_published_trajectory_ &&
        !have_controller_accepted_final_identity_)
      return;
    if (msg->header.frame_id != expected_frame_id_ ||
        rclcpp::Time(msg->header.stamp).nanoseconds() <= 0)
    {
      RCLCPP_WARN_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 1000,
          "忽略 frame 或时间戳无效的 controller status");
      return;
    }

    const FinalHoldTrajectoryIdentity observed{
        rclcpp::Time(msg->reference_path_stamp).nanoseconds(),
        rclcpp::Time(msg->bspline_header_stamp).nanoseconds(),
        rclcpp::Time(msg->start_time).nanoseconds(),
        msg->traj_id};
    FinalHoldTrajectoryIdentity last_published_identity;
    if (have_last_published_trajectory_)
    {
      last_published_identity = FinalHoldTrajectoryIdentity{
          rclcpp::Time(
              last_published_trajectory_.reference_path_stamp).nanoseconds(),
          rclcpp::Time(
              last_published_trajectory_.header.stamp).nanoseconds(),
          rclcpp::Time(last_published_trajectory_.start_time).nanoseconds(),
          last_published_trajectory_.traj_id};
    }

    // ControllerStatus.identity 始终指向真正执行的轨迹。当 planner 的
    // timer 忙于优化时，ACCEPTED 回调可能晚于下一条 B-spline 发布，所以
    // 使用当前 Path 代际的有界发布历史认证，而不是只比较最新一条。
    // 被 goal latch 拒绝的 candidate 位于独立字段，不能覆盖执行身份。
    const bool observed_was_published =
        wasFinalIdentityPublished(
            published_final_trajectory_history_, observed) ||
        (have_last_published_trajectory_ &&
         sameFinalHoldIdentity(last_published_identity, observed));
    updateControllerAcceptedFinalIdentity(
        observed_was_published, observed,
        msg->accepted, msg->trajectory_valid, msg->is_final,
        msg->emergency_stop, controller_accepted_final_identity_,
        have_controller_accepted_final_identity_);

    FinalHoldTrajectoryIdentity expected;
    if (reference_final_hold_lifecycle_.pending)
    {
      if (!have_last_published_trajectory_)
        return;
      expected = last_published_identity;
    }
    else
    {
      if (!have_controller_accepted_final_identity_)
        return;
      expected = controller_accepted_final_identity_;
    }
    FinalHoldControllerState controller_state =
        FinalHoldControllerState::kOther;
    if (msg->state == ControllerStatusMessage::STATE_GOAL_REACHED)
      controller_state = FinalHoldControllerState::kGoalReached;
    else if (msg->state == ControllerStatusMessage::STATE_TRAJECTORY_TIMEOUT)
      controller_state = FinalHoldControllerState::kTrajectoryTimeout;

    const FinalHoldControllerAction action =
        decideFinalHoldControllerAction(
            reference_final_hold_lifecycle_.pending, have_target_, expected, observed,
            msg->accepted, msg->trajectory_valid, msg->is_final,
            msg->emergency_stop, controller_state);
    if (action == FinalHoldControllerAction::kIgnore)
      return;

    reference_final_hold_lifecycle_ = transitionFinalHoldLifecycle(
        reference_final_hold_lifecycle_, action,
        expected.trajectory_id);

    if (action == FinalHoldControllerAction::kConfirmGoal)
    {
      controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
      have_controller_accepted_final_identity_ = false;
      have_last_published_trajectory_ = false;
      published_final_trajectory_history_.clear();
      have_target_ = false;
      trigger_ = false;
      have_new_target_ = false;
      replan_fail_count_ = 0;
      recovery_status_pending_ = false;
      need_hover_stop_ = false;
      flag_escape_emergency_ = false;
      // controller 已确认最终轨迹到达后，本代 Path 不再有可执行目标。Isaac
      // 会转入机械臂阶段并停止发布楼梯冻结心跳；若继续保留旧代绑定，freshness
      // timer 会把正常的导航结束误报为楼梯快照超时，并向 supervisor 发布一条
      // 缺少可执行 Path stamp 的假急停状态。新 Path 到来时会重新严格绑定。
      stair_planning_freeze_gate_.clearReferenceGeneration();
      changeFSMExecState(WAIT_TARGET, "FINAL_HOLD_CONTROLLER_ACK");
      RCLCPP_INFO(
          node_->get_logger(),
          "收到与 controller 已接受 final 轨迹精确匹配的 "
          "GOAL_REACHED；清除目标");
      return;
    }

    // TIMEOUT 已结束 ACK 等待，但保留 target 与同代 Path。独立 recovery
    // 状态会阻止下一拍刷新 stationary identity，并要求正常恢复轨迹 ID 递增。
    controller_accepted_final_identity_ = FinalHoldTrajectoryIdentity{};
    have_controller_accepted_final_identity_ = false;
    trigger_ = true;
    have_new_target_ = true;
    replan_fail_count_ = 0;
    recovery_status_pending_ = true;
    need_hover_stop_ = false;
    flag_escape_emergency_ = true;
    reference_retry_not_before_ = node_->now();
    changeFSMExecState(REPLAN_TRAJ, "FINAL_HOLD_CONTROLLER_TIMEOUT");
    RCLCPP_WARN(
        node_->get_logger(),
        "stationary final hold 已由 controller 判定超时；保留同代 Path 并重规划");
  }

  void SCANReplanFSM::stairExecutionFrozenCallback(
      const scan_planner_msgs::msg::StairExecutionFreeze::ConstSharedPtr &msg)
  {
    if (!msg)
      return;

    StairExecutionFreezeMailboxMessage message;
    message.frame_id = msg->header.frame_id;
    message.snapshot.frozen = msg->frozen;
    message.snapshot.header_stamp_ns =
        rclcpp::Time(msg->header.stamp).nanoseconds();
    message.snapshot.reference_path_stamp_ns =
        rclcpp::Time(msg->reference_path_stamp).nanoseconds();
    message.snapshot.writer_id = msg->writer_id;
    message.snapshot.writer_epoch = msg->writer_epoch;
    message.snapshot.sequence = msg->sequence;
    stair_execution_frozen_mailbox_.push(std::move(message));
  }

  void SCANReplanFSM::drainStairExecutionFrozenMailbox()
  {
    StairExecutionFreezeMailboxDrain drain =
        stair_execution_frozen_mailbox_.drain();
    if (drain.available)
      processStairExecutionFrozenSnapshot(drain.message);
  }

  void SCANReplanFSM::processStairExecutionFrozenSnapshot(
      const StairExecutionFreezeMailboxMessage &message)
  {
    const StairExecutionFreezeSnapshot &snapshot = message.snapshot;

    const ReferencePathBinding reference_binding =
        resolveReferencePathBinding(
            rclcpp::Time(pending_reference_path_stamp_).nanoseconds(),
            navi_mode_ == NAVI_MODE::REFERENCE_PATH &&
                !pending_reference_waypoints_.empty(),
            rclcpp::Time(active_reference_path_stamp_).nanoseconds(),
            navi_mode_ == NAVI_MODE::REFERENCE_PATH &&
                activeReferenceAvailableForFreezeBinding(
                    !active_reference_waypoints_.empty(), have_target_,
                    reference_final_hold_lifecycle_.pending));
    // Path 已清空或 controller 已确认到达后，publisher 可能在同一仿真拍
    // 仍发出旧楼梯快照。此时没有可执行目标，也没有可认证的绑定；直接忽略
    // 即可。新 Path 到来后会重新 bind，并继续执行严格身份与新鲜度校验。
    if (!reference_binding.available)
      return;
    const int64_t observation_stamp_ns =
        planner_manager_ && planner_manager_->grid_map_
            ? planner_manager_->grid_map_->observationStampNs()
            : 0;
    const int64_t now_ns = node_->now().nanoseconds();
    const int64_t timeout_ns = static_cast<int64_t>(
        stair_execution_freeze_timeout_sec_ * 1.0e9);
    if (message.frame_id != expected_frame_id_)
    {
      stair_planning_freeze_gate_.failClosed(
          "stair_freeze_header_frame_mismatch");
      if (reference_binding.available)
        publishPlanningStatus(
            ScanPlanningEvent::kStairInhibited,
            kScanStairFreezeFrameMismatchFault);
      RCLCPP_ERROR(
          node_->get_logger(),
          "拒绝楼梯冻结消息：Header frame 必须与 active Path 完全一致");
      return;
    }
    const StairPlanningFreezeUpdate update =
        stair_planning_freeze_gate_.updateTyped(
            snapshot, last_odom_stamp_ns_, observation_stamp_ns,
            reference_binding.stamp_ns, reference_binding.available,
            now_ns, timeout_ns);
    if (update == StairPlanningFreezeUpdate::kProtocolRejected)
    {
      if (reference_binding.available)
        publishPlanningStatus(
            ScanPlanningEvent::kStairInhibited,
            kScanStairFreezeProtocolFault);
      RCLCPP_ERROR(
          node_->get_logger(),
          "拒绝楼梯冻结快照并保持冻结：%s",
          stair_planning_freeze_gate_.protocolFault().c_str());
      return;
    }
    if (update == StairPlanningFreezeUpdate::kDuplicate)
      return;

    if (update == StairPlanningFreezeUpdate::kInhibited)
    {
      // 真正进入楼梯冻结时丢弃平地运动历史；冻结期间后续 Odometry 会重新
      // 建立实测均值，解冻后不会把楼梯前的巡航速度带入新局部轨迹。
      reference_velocity_filter_.reset();
      filtered_odom_vel_.setZero();
      updateReferenceProgressDuringStairFreeze();

      // active Path 上可能仍有一条 TRACKING/ALIGNING_YAW 轨迹。仅在策略
      // 写入端输出零速并不能让 controller/supervisor 证明它已经停车，而且
      // 等待旧轨迹自然超时会受剩余时长和 yaw 冻结预算影响。冻结上升沿
      // 因此立即发布同一 active Path 代际下的新急停 B-spline；controller
      // 接受该唯一身份后会报告 EMERGENCY_STOP，root-lock 才能确定性接管。
      // pending Path 尚未成为 active，不能伪造绑定该代际的轨迹；此时
      // supervisor 本来就处于 LOCAL_PLANNING 零速门，由 stair ACK 保持停车。
      if (!reference_binding.pending)
      {
        pending_emergency_reason_ =
            "楼梯执行冻结，发布严格零速安全轨迹";
        if (!callEmergencyStop(odom_pos_))
        {
          publishPlanningStatus(
              ScanPlanningEvent::kStairInhibited,
              kScanStairStopPublishFault);
          RCLCPP_ERROR(
              node_->get_logger(),
              "楼梯执行冻结已生效，但严格零速 B-spline 发布失败");
          return;
        }
      }
      publishPlanningStatus(
          ScanPlanningEvent::kStairInhibited,
          kScanStairExecutionInhibitedReason);
      RCLCPP_INFO(
          node_->get_logger(),
          reference_binding.pending ?
              "收到楼梯执行冻结：pending Path 保持零速并暂停 SCAN 规划" :
              "收到楼梯执行冻结：已发布急停轨迹并暂停 SCAN 规划");
      return;
    }

    if (update == StairPlanningFreezeUpdate::kInitialUnfrozen)
    {
      RCLCPP_INFO(
          node_->get_logger(),
          "当前 Path 首条楼梯状态为未冻结：直接允许 SCAN 规划");
      return;
    }

    if (update == StairPlanningFreezeUpdate::kResumeBarrierStarted)
    {
      // root-lock 上楼期间 GridMap 仍持续接收点云以维持传感器新鲜度，
      // 因而会保留楼梯踏面、立面以及释放点附近的旧膨胀占据。解冻后若直接
      // 复用这份缓存，真实可站立的新起点可能被误判为“机器人在障碍物内”。
      // 这里先清除上一楼梯代次的局部占据；resume barrier 仍要求随后收到
      // 严格晚于释放基线的新 Odometry 和非空地图观测，不能靠空地图放行。
      if (planner_manager_ && planner_manager_->grid_map_)
        planner_manager_->grid_map_->resetBuffer();
      publishPlanningStatus(
          ScanPlanningEvent::kStairResumeWaiting,
          kScanStairResumeWaitingReason);
      RCLCPP_INFO(
          node_->get_logger(),
          "楼梯执行冻结解除：已清除楼梯代次占据缓存，等待 Odometry 与地图观测严格晚于解除基线");
      return;
    }

    RCLCPP_INFO(
        node_->get_logger(),
        "楼梯执行冻结解除，但当前没有可恢复的 active initial_path");
    publishPlanningStatus(
        ScanPlanningEvent::kReferenceCleared,
        "楼梯冻结解除，当前没有 active reference Path");
  }

  void SCANReplanFSM::refreshStairExecutionFreezeFreshness()
  {
    const int64_t timeout_ns = static_cast<int64_t>(
        stair_execution_freeze_timeout_sec_ * 1.0e9);
    const int64_t confirmation_grace_ns = static_cast<int64_t>(
        stair_execution_freeze_confirmation_sec_ * 1.0e9);
    const StairFreezeFreshnessUpdate freshness_update =
        stair_planning_freeze_gate_.refreshFreshness(
            node_->now().nanoseconds(), timeout_ns,
            confirmation_grace_ns);
    if (freshness_update == StairFreezeFreshnessUpdate::kNoChange)
      return;
    updateReferenceProgressDuringStairFreeze();
    if (freshness_update ==
        StairFreezeFreshnessUpdate::kTimeoutCandidateStarted)
    {
      RCLCPP_WARN(
          node_->get_logger(),
          "楼梯冻结快照首次超过 %.3f 秒：已立即暂停 SCAN，等待 %.3f 秒调度确认",
          stair_execution_freeze_timeout_sec_,
          stair_execution_freeze_confirmation_sec_);
      return;
    }
    publishPlanningStatus(
        ScanPlanningEvent::kStairInhibited,
        kScanStairFreezeSnapshotTimeoutFault);
    RCLCPP_ERROR(
        node_->get_logger(),
        "楼梯冻结快照超过 %.3f 秒且在 %.3f 秒确认期内未刷新，已确认暂停 SCAN",
        stair_execution_freeze_timeout_sec_,
        stair_execution_freeze_confirmation_sec_);
  }

  void SCANReplanFSM::updateLocalTrajTimeFreeze()
  {
    const rclcpp::Time now = node_->now();
    double dt = (now - last_freeze_update_time_).seconds();
    last_freeze_update_time_ = now;

    if (dt <= 0.0 || dt > 0.2)
      return;

    LocalTrajData *info = &planner_manager_->local_data_;
    if (trajectoryTimeFrozen(
            go2_execution_frozen_,
            stair_planning_freeze_gate_.frozen()) &&
        info->start_time_.seconds() > 1e-5)
      info->start_time_ += rclcpp::Duration::from_seconds(dt);
  }

  void SCANReplanFSM::publishPlanningStatus(
      const ScanPlanningEvent event,
      const std::string &reason,
      const scan_planner_msgs::msg::Bspline *trajectory)
  {
    if (!planning_status_pub_)
      return;
    const rclcpp::Time stamp = node_->now();
    if (stamp.nanoseconds() <= 0)
      return;

    if (event != ScanPlanningEvent::kInitial &&
        !initial_planning_status_published_)
    {
      publishPlanningStatus(
          ScanPlanningEvent::kInitial,
          "SCAN 已就绪，等待有效 reference Path");
    }
    if (event == ScanPlanningEvent::kInitial &&
        initial_planning_status_published_)
      return;

    const std::uint32_t failure_count =
        static_cast<std::uint32_t>(std::max(replan_fail_count_, 0));
    const std::uint32_t maximum_failures = static_cast<std::uint32_t>(
        std::max(max_replan_fail_count_, 1));
    const ScanPlanningStatusPolicy policy = planningStatusPolicy(
        event, failure_count, maximum_failures,
        global_replan_generation_gate_.required());

    if (planning_status_sequence_ ==
        std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("SCAN planning status_sequence 已耗尽");

    scan_planner_msgs::msg::ScanPlanningStatus status;
    status.header.stamp = stamp;
    status.header.frame_id = expected_frame_id_;
    status.status_sequence = ++planning_status_sequence_;
    status.event = static_cast<std::uint8_t>(event);
    status.state = static_cast<std::uint8_t>(policy.state);
    const ReferencePathBinding reference_binding =
        resolveReferencePathBinding(
            rclcpp::Time(pending_reference_path_stamp_).nanoseconds(),
            navi_mode_ == NAVI_MODE::REFERENCE_PATH &&
                !pending_reference_waypoints_.empty(),
            rclcpp::Time(active_reference_path_stamp_).nanoseconds(),
            navi_mode_ == NAVI_MODE::REFERENCE_PATH &&
                activeReferenceAvailableForFreezeBinding(
                    !active_reference_waypoints_.empty(), have_target_,
                    reference_final_hold_lifecycle_.pending));
    if (reference_binding.available)
      status.reference_path_stamp = reference_binding.pending ?
          pending_reference_path_stamp_ : active_reference_path_stamp_;
    status.consecutive_planning_failures = failure_count;
    status.stop_required = policy.stop_required;
    status.global_replan_recommended =
        policy.global_replan_recommended;
    status.reason = reason;

    if (trajectory != nullptr)
    {
      status.reference_path_stamp = trajectory->reference_path_stamp;
      status.trajectory_present = true;
      status.bspline_header_stamp = trajectory->header.stamp;
      status.trajectory_start_time = trajectory->start_time;
      status.trajectory_id = trajectory->traj_id;
      status.trajectory_is_final = trajectory->is_final;
      status.trajectory_emergency_stop = trajectory->emergency_stop;
    }

    planning_status_pub_->publish(status);
    if (event == ScanPlanningEvent::kInitial)
      initial_planning_status_published_ = true;
  }

  bool SCANReplanFSM::publishBsplineDiagnostics(
      const scan_planner_msgs::msg::Bspline &trajectory,
      const LocalTrajData &trajectory_data,
      const ActiveSensingDiagnosticsSnapshot *active_snapshot)
  {
    if (!bspline_diagnostics_pub_ ||
        trajectory_diagnostics_sequence_ ==
            std::numeric_limits<std::uint64_t>::max())
      return false;

    UniformBspline diagnostic_trajectory =
        trajectory_data.position_traj_;
    const BoundedGeometrySamples trajectory_geometry =
        sampleTrajectoryGeometry(
            diagnostic_trajectory,
            static_cast<std::size_t>(
                trajectory_diagnostic_max_samples_));
    const double trajectory_duration =
        diagnostic_trajectory.getTimeSum();
    const double maximum_velocity_upper_bound =
        trajectoryMaximumVelocityUpperBound(
            diagnostic_trajectory);
    if (!trajectory_geometry.valid ||
        !std::isfinite(trajectory_duration) ||
        trajectory_duration <= 0.0 ||
        !std::isfinite(maximum_velocity_upper_bound) ||
        maximum_velocity_upper_bound < 0.0 ||
        !std::isfinite(self_double_cylinder_radius_) ||
        self_double_cylinder_radius_ <= 0.0 ||
        !std::isfinite(self_double_cylinder_offset_) ||
        self_double_cylinder_offset_ < 0.0)
      return false;

    BoundedGeometrySamples reference_geometry;
    if (trajectory_data.reference_corridor_checked_)
    {
      reference_geometry = sampleOrderedReferenceGeometry(
          trajectory_data.ordered_reference_guide_,
          static_cast<std::size_t>(
              trajectory_diagnostic_max_samples_));
      if (!trajectory_data.reference_corridor_safe_ ||
          !reference_geometry.valid)
        return false;
    }

    scan_planner_msgs::msg::BsplineDiagnostics diagnostics;
    diagnostics.header = trajectory.header;
    diagnostics.start_time = trajectory.start_time;
    diagnostics.reference_path_stamp = trajectory.reference_path_stamp;
    diagnostics.traj_id = trajectory.traj_id;
    diagnostics.is_final = trajectory.is_final;
    diagnostics.emergency_stop = trajectory.emergency_stop;
    diagnostics.stationary =
        trajectoryIsStationary(trajectory_data.position_traj_);
    diagnostics.ordered_reference_checked =
        trajectory_data.reference_corridor_checked_;
    diagnostics.ordered_reference_safe =
        trajectory_data.reference_corridor_safe_;
    diagnostics.maximum_trajectory_deviation =
        trajectory_data.maximum_trajectory_deviation_;
    diagnostics.maximum_guide_anchor_deviation =
        trajectory_data.maximum_guide_anchor_deviation_;
    diagnostics.maximum_guide_progress_lead =
        trajectory_data.maximum_guide_progress_lead_;
    diagnostics.maximum_deviation_limit =
        trajectory_data.reference_corridor_deviation_limit_;
    diagnostics.maximum_progress_lead_limit =
        trajectory_data.reference_corridor_progress_lead_limit_;
    diagnostics.trajectory_duration = trajectory_duration;
    diagnostics.maximum_velocity_upper_bound =
        maximum_velocity_upper_bound;
    diagnostics.double_cylinder_radius =
        self_double_cylinder_radius_;
    diagnostics.double_cylinder_offset =
        self_double_cylinder_offset_;
    diagnostics.trajectory_sample_count_total =
        trajectory_geometry.total_count;
    diagnostics.trajectory_samples_truncated =
        trajectory_geometry.truncated;
    diagnostics.trajectory_samples.reserve(
        trajectory_geometry.points.size());
    for (const Eigen::Vector3d &point : trajectory_geometry.points)
    {
      geometry_msgs::msg::Point sample;
      sample.x = point.x();
      sample.y = point.y();
      sample.z = point.z();
      diagnostics.trajectory_samples.push_back(sample);
    }
    if (reference_geometry.valid)
    {
      diagnostics.ordered_reference_sample_count_total =
          reference_geometry.total_count;
      diagnostics.ordered_reference_samples_truncated =
          reference_geometry.truncated;
      diagnostics.ordered_reference_samples.reserve(
          reference_geometry.points.size());
      for (const Eigen::Vector3d &point : reference_geometry.points)
      {
        geometry_msgs::msg::Point sample;
        sample.x = point.x();
        sample.y = point.y();
        sample.z = point.z();
        diagnostics.ordered_reference_samples.push_back(sample);
      }
    }

    if (active_snapshot != nullptr)
    {
      const ActiveSensingTrajectoryIdentity identity =
          activeSensingIdentityFromBspline(trajectory);
      if (!diagnostics.stationary || diagnostics.is_final ||
          diagnostics.emergency_stop ||
          !sameActiveSensingTrajectoryIdentity(
              active_sensing_expected_identity_, identity) ||
          !populateActiveSensingDiagnostics(
              diagnostics, *active_snapshot))
        return false;
    }
    else
    {
      diagnostics.active_sensing = false;
      diagnostics.active_sensing_event =
          scan_planner_msgs::msg::BsplineDiagnostics::
              ACTIVE_SENSING_EVENT_NONE;
    }

    diagnostics.diagnostic_sequence =
        ++trajectory_diagnostics_sequence_;

    // 先发布诊断再发布 payload；同 identity 的 transient-local 消息已存在，
    // controller 即使立即接受 B-spline，审计端也能完成 join。
    bspline_diagnostics_pub_->publish(diagnostics);
    return true;
  }

  bool SCANReplanFSM::publishActiveSensingDiagnostics(
      const std::uint8_t event, const std::string &reason)
  {
    if (!planner_manager_ || !have_last_published_trajectory_ ||
        !sameActiveSensingTrajectoryIdentity(
            active_sensing_expected_identity_,
            activeSensingIdentityFromBspline(
                last_published_trajectory_)))
      return false;

    ActiveSensingDiagnosticsSnapshot snapshot;
    snapshot.event = event;
    snapshot.start_yaw = active_sensing_start_yaw_;
    snapshot.target_yaw = active_sensing_target_yaw_;
    snapshot.yaw_offset = active_sensing_yaw_offset_;
    snapshot.yaw_rate = active_sensing_yaw_rate_;
    snapshot.settle_stamp_ns =
        active_sensing_observation_baseline_stamp_ns_;
    snapshot.settle_yaw_error = active_sensing_settle_yaw_error_;
    snapshot.settle_angular_speed =
        active_sensing_settle_angular_speed_;
    snapshot.stable_duration =
        active_sensing_measured_stable_duration_sec_;
    snapshot.fusion_baseline = active_sensing_fusion_baseline_;
    snapshot.fusion_current = active_sensing_fusion_current_;
    snapshot.fusion_distinct = active_sensing_fusion_distinct_;
    snapshot.completed =
        event == scan_planner_msgs::msg::BsplineDiagnostics::
            ACTIVE_SENSING_EVENT_COMPLETED;
    snapshot.failed =
        event == scan_planner_msgs::msg::BsplineDiagnostics::
            ACTIVE_SENSING_EVENT_FAILED;
    snapshot.reason = reason;
    return publishBsplineDiagnostics(
        last_published_trajectory_, planner_manager_->local_data_,
        &snapshot);
  }

  void SCANReplanFSM::recordPlanningFailure(const std::string &reason)
  {
    if (replan_fail_count_ < std::numeric_limits<int>::max())
      ++replan_fail_count_;
    if (replan_fail_count_ >= max_replan_fail_count_ &&
        navi_mode_ == NAVI_MODE::REFERENCE_PATH)
    {
      const std::int64_t reference_stamp_ns =
          rclcpp::Time(active_reference_path_stamp_).nanoseconds();
      if (reference_stamp_ns > 0)
        global_replan_generation_gate_.require(reference_stamp_ns);
    }
    publishPlanningStatus(
        ScanPlanningEvent::kPlanningFailed, reason);
  }

  void SCANReplanFSM::publishRecoveredIfPending(
      const std::string &reason)
  {
    const bool recovered = recovery_status_pending_ || replan_fail_count_ > 0;
    replan_fail_count_ = 0;
    recovery_status_pending_ = false;
    if (!recovered || global_replan_generation_gate_.required())
      return;
    publishPlanningStatus(
        ScanPlanningEvent::kRecovered,
        reason,
        have_last_published_trajectory_ ? &last_published_trajectory_ : nullptr);
  }

  double SCANReplanFSM::getOdomYaw() const
  {
    Eigen::Vector3d heading = odom_orient_.toRotationMatrix().col(0);
    if (heading.head<2>().squaredNorm() < 1e-8)
      return 0.0;
    return std::atan2(heading(1), heading(0));
  }

  double SCANReplanFSM::getReferenceGoalYaw() const
  {
    // PCT 最后一条短网格边不代表任务要求的最终朝向；Path 末 Pose 的
    // 四元数才是跨层任务的 terminal_yaw 契约。
    if (!active_reference_waypoints_.empty() &&
        std::isfinite(active_reference_goal_yaw_))
      return active_reference_goal_yaw_;
    return getOdomYaw();
  }

  double SCANReplanFSM::estimateYawFromSegment(const Eigen::Vector3d &from, const Eigen::Vector3d &to) const
  {
    Eigen::Vector2d diff(to(0) - from(0), to(1) - from(1));
    if (diff.squaredNorm() < 1e-8)
      return getOdomYaw();
    return std::atan2(diff(1), diff(0));
  }

  void SCANReplanFSM::publishSelfInflationMarker()
  {
    const double radius = std::max(0.0, self_double_cylinder_radius_);
    const double z_up = std::max(0.0, self_inflation_z_up_);
    const double z_down = std::max(0.0, self_inflation_z_down_);
    const double height = std::max(1e-3, z_up + z_down);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = self_inflation_frame_id_.empty() ? "world" : self_inflation_frame_id_;
    marker.header.stamp = node_->now();
    marker.ns = "self_inflation";
    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 2.0 * radius;
    marker.scale.y = 2.0 * radius;
    marker.scale.z = height;
    marker.color.r = 0.1;
    marker.color.g = 0.6;
    marker.color.b = 1.0;
    marker.color.a = 0.4;
    marker.lifetime = rclcpp::Duration::from_seconds(0.2);

    Eigen::Vector3d center = odom_pos_;
    // GridMap 将障碍向上膨胀 z_up、向下膨胀 z_down；换算到以机器人
    // 中心为原点的实际包络后，下界为 -z_up、上界为 +z_down。
    center(2) += 0.5 * (z_down - z_up);

    Eigen::Vector3d heading(std::cos(getOdomYaw()), std::sin(getOdomYaw()), 0.0);
    Eigen::Vector3d front = center + self_double_cylinder_offset_ * heading;
    Eigen::Vector3d rear = center - self_double_cylinder_offset_ * heading;

    marker.id = 0;
    marker.pose.position.x = front(0);
    marker.pose.position.y = front(1);
    marker.pose.position.z = front(2);
    self_inflation_pub_->publish(marker);

    marker.id = 1;
    marker.pose.position.x = rear(0);
    marker.pose.position.y = rear(1);
    marker.pose.position.z = rear(2);
    self_inflation_pub_->publish(marker);
  }

  void SCANReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call)
  {

    if (new_state == exec_state_)
      continuously_called_times_++;
    else
      continuously_called_times_ = 1;

    static string state_str[7] = {
        "INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ",
        "EXEC_TRAJ", "ACTIVE_SENSING", "EMERGENCY_STOP"};
    int pre_s = int(exec_state_);
    exec_state_ = new_state;
    cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
  }

  std::pair<int, SCANReplanFSM::FSM_EXEC_STATE> SCANReplanFSM::timesOfConsecutiveStateCalls()
  {
    return std::pair<int, FSM_EXEC_STATE>(continuously_called_times_, exec_state_);
  }

  void SCANReplanFSM::printFSMExecState()
  {
    static string state_str[7] = {
        "INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ",
        "EXEC_TRAJ", "ACTIVE_SENSING", "EMERGENCY_STOP"};

    cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
  }

  void SCANReplanFSM::execFSMCallback()
  {
    // 高频 ROS 定时器与规划优化不能阻塞最新 Odometry。先取走独立回调组
    // 合并后的最新帧，再用同一默认互斥组更新全部规划状态。
    drainOdometryMailbox();
    // 独立线程只负责收件；主 FSM 在读取 ROS 时钟和检查 freshness 前，
    // 串行消费全部快照，保证 planner/path/grid 状态仍无并发写入。
    drainStairExecutionFrozenMailbox();
    publishPlanningStatus(
        ScanPlanningEvent::kInitial,
        "SCAN 已就绪，等待有效 reference Path");
    refreshStairExecutionFreezeFreshness();
    updateLocalTrajTimeFreeze();
    tryActivatePendingReferencePath();

    auto publish_data_display = [this]() {
      data_disp_.header.stamp = node_->now();
      data_disp_pub_->publish(data_disp_);
    };

    // 主动感知自身负责逐拍复核 stair/final/replan gate、输入、包络与漂移。
    // 必须先于楼梯早返回执行，才能在异步冻结到达时发布严格零速安全轨迹。
    if (active_sensing_phase_ != ACTIVE_SENSING_IDLE)
    {
      updateActiveSensing();
      publish_data_display();
      return;
    }

    if (stair_planning_freeze_gate_.frozen())
    {
      // 暂停期间只做有序进度投影和终点收口，不进入 optimizer/FSM。
      updateReferenceProgressDuringStairFreeze();
      tryFinishReferenceAtGoal();
      publish_data_display();
      return;
    }

    if (stair_planning_freeze_gate_.resumeWaiting())
    {
      updateReferenceProgressDuringStairFreeze();
      if (!stairResumeInputsReady())
      {
        publish_data_display();
        return;
      }

      // fresh barrier 完成后先复核终点；若尚未到达，再从新鲜 Odometry
      // 强制生成同一 reference_path_stamp 的更高 traj_id 局部轨迹。
      stair_planning_freeze_gate_.clearResumeBarrier();
      const FinalHoldStairResumeAction final_hold_resume_action =
          decideFinalHoldStairResumeAction(
              reference_final_hold_lifecycle_);
      if (final_hold_resume_action ==
          FinalHoldStairResumeAction::kWaitForController)
      {
        // 解冻只恢复时钟和输入，不得在 controller 终态到达前替换 hold。
        publish_data_display();
        return;
      }
      if (final_hold_resume_action ==
          FinalHoldStairResumeAction::kContinueTimeoutRecovery)
      {
        if (exec_state_ != REPLAN_TRAJ)
          changeFSMExecState(
              REPLAN_TRAJ, "FINAL_HOLD_TIMEOUT_AFTER_STAIR_RESUME");
      }
      else
      {
        if (tryFinishReferenceAtGoal())
        {
          publish_data_display();
          return;
        }
        forceReferenceReplanAfterStairResume();
      }
    }

    static int fsm_num = 0;
    fsm_num++;
    if (fsm_num == 100)
    {
      printFSMExecState();
      if (!have_odom_)
        cout << "no odom." << endl;
      if (!trigger_)
        cout << "wait for goal." << endl;
      fsm_num = 0;
    }

    switch (exec_state_)
    {
    case INIT:
    {
      if (!have_odom_)
      {
        return;
      }
      if (!trigger_)
      {
        return;
      }
      changeFSMExecState(WAIT_TARGET, "FSM");
      break;
    }

    case WAIT_TARGET:
    {
      // stationary final 已发布但尚未收到精确 ControllerStatus 终态时，目标
      // 仍有效；不能把 have_target=true 误解为应生成另一条轨迹并重置 dwell。
      if (reference_final_hold_lifecycle_.pending)
        return;
      if (!have_target_)
        return;
      else
      {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }

    case GEN_NEW_TRAJ:
    {
      setStartStateFromOdomOrCurrentTraj();

      if (tryFinishReferenceAtGoal())
        break;
      if (!referenceRetryIsReady())
        break;

      // Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
      // start_yaw_(0)         = atan2(rot_x(1), rot_x(0));
      // start_yaw_(1) = start_yaw_(2) = 0.0;

      bool flag_random_poly_init;
      if (timesOfConsecutiveStateCalls().first == 1)
        flag_random_poly_init = false;
      else
        flag_random_poly_init = true;

      bool success = callReboundReplan(true, flag_random_poly_init);
      if (success)
      {
        publishRecoveredIfPending(
            "局部规划已恢复并发布新的可执行轨迹");
        reference_retry_not_before_ = node_->now();
        changeFSMExecState(EXEC_TRAJ, "FSM");
        flag_escape_emergency_ = true;
      }
      else
      {
        if (!tryStartActiveSensing())
        {
          recordPlanningFailure("生成新局部轨迹失败");
          deferReferenceRetry();
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
        }
      }
      break;
    }

    case REPLAN_TRAJ:
    {

      if (tryFinishReferenceAtGoal())
        break;
      if (!referenceRetryIsReady())
        break;

      if (planFromCurrentTraj())
      {
        publishRecoveredIfPending(
            "局部重规划已恢复并发布新的可执行轨迹");
        reference_retry_not_before_ = node_->now();
        changeFSMExecState(EXEC_TRAJ, "FSM");
      }
      else
      {
        if (!tryStartActiveSensing())
        {
          recordPlanningFailure("滚动局部重规划失败");
          deferReferenceRetry();
          changeFSMExecState(REPLAN_TRAJ, "FSM");
        }
      }

      break;
    }

    case EXEC_TRAJ:
    {
      /* determine if need to replan */
      LocalTrajData *info = &planner_manager_->local_data_;
      rclcpp::Time time_now = node_->now();
      const double raw_t_cur = (time_now - info->start_time_).seconds();
      const double t_cur = min(info->duration_, raw_t_cur);

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t_cur);

      if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
      {
        if (tryFinishReferenceAtGoal())
          break;
        const ReferenceExecutionAction action =
            decideReferenceExecutionAction(
                t_cur > info->duration_ - 1e-2,
                have_target_,
                local_target_is_final_,
                go2_execution_frozen_,
                raw_t_cur <=
                    info->duration_ + final_trajectory_convergence_grace_sec_,
                (end_pt_ - odom_pos_).norm(),
                (info->start_pos_ - odom_pos_).norm(),
                replan_thresh_,
                no_replan_thresh_);
        if (action == ReferenceExecutionAction::kReplan)
        {
          if (t_cur > info->duration_ - 1e-2)
          {
            RCLCPP_INFO(
                node_->get_logger(),
                "局部轨迹已到期但真实 Odometry 尚未到达，继续沿参考路径重规划");
          }
          changeFSMExecState(REPLAN_TRAJ, "FSM");
          return;
        }
        if (action == ReferenceExecutionAction::kFinish)
        {
          have_target_ = false;
          changeFSMExecState(WAIT_TARGET, "FSM");
          return;
        }
        return;
      }

      if (isWaypointSequenceMode() &&
          current_wp_ + 1 < (int)active_waypoints_.size() &&
          (end_pt_ - odom_pos_).norm() < 0.5)
      {
        current_wp_++;
        if (planNextWaypoint())
        {
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
          return;
        }
        recordPlanningFailure("无法切换到下一个预设 waypoint");
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
        return;
      }

      /* && (end_pt_ - pos).norm() < 0.5 */
      if (t_cur > info->duration_ - 1e-2)
      {
        if (have_target_ && !local_target_is_final_)
        {
          // 非最终局部轨迹到期时仍保留全局目标，继续沿参考路径滚动规划。
          RCLCPP_INFO(
              node_->get_logger(),
              "非最终局部轨迹已执行完，继续沿全局参考路径重规划");
          changeFSMExecState(REPLAN_TRAJ, "FSM");
          return;
        }

        if (isWaypointSequenceMode() && current_wp_ + 1 < (int)active_waypoints_.size())
        {
          current_wp_++;
          if (planNextWaypoint())
          {
            changeFSMExecState(GEN_NEW_TRAJ, "FSM");
            return;
          }
          recordPlanningFailure("轨迹结束后无法切换到下一个 waypoint");
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
          return;
        }

        if (isWaypointSequenceMode())
        {
          active_waypoints_.clear();
          current_wp_ = 0;
        }

        have_target_ = false;

        changeFSMExecState(WAIT_TARGET, "FSM");
        return;
      }
      else if ((end_pt_ - pos).norm() < no_replan_thresh_)
      {
        // cout << "near end" << endl;
        return;
      }
      else if ((info->start_pos_ - pos).norm() < replan_thresh_)
      {
        // cout << "near start" << endl;
        return;
      }
      else
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }
      break;
    }

    case ACTIVE_SENSING:
    {
      if (active_sensing_phase_ == ACTIVE_SENSING_IDLE)
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "ACTIVE_SENSING FSM 缺少运行时生命周期，退回正常规划");
        changeFSMExecState(GEN_NEW_TRAJ, "ACTIVE_SENSING_INVALID_PHASE");
      }
      else
      {
        updateActiveSensing();
      }
      break;
    }

    case EMERGENCY_STOP:
    {

      if (tryFinishReferenceAtGoal())
        break;

      if (flag_escape_emergency_) // Avoiding repeated calls
      {
        callEmergencyStop(odom_pos_);
        deferReferenceRetry();
        flag_escape_emergency_ = false;
        break;
      }

      if (global_replan_generation_gate_.required())
        break;

      if (!referenceRetryIsReady())
      {
        flag_escape_emergency_ = false;
        break;
      }
      if (enable_fail_safe_ && !need_hover_stop_ && odom_vel_.norm() < 0.1)
      {
        recovery_status_pending_ = true;
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      else if (enable_fail_safe_ && need_hover_stop_ && odom_vel_.norm() < 0.1)
      {
        need_hover_stop_ = false;
        const ReferenceEmergencyRecoveryAction recovery_action =
            decideReferenceEmergencyRecovery(
                navi_mode_ == NAVI_MODE::REFERENCE_PATH,
                !active_reference_waypoints_.empty(),
                have_target_);
        if (recovery_action ==
            ReferenceEmergencyRecoveryAction::kRetryActiveReference)
        {
          // 仅没有全局重规划锁存时，临时局部失败才允许重试同代 Path。
          trigger_ = true;
          recovery_status_pending_ = true;
          RCLCPP_INFO(
              node_->get_logger(),
              "Exiting EMERGENCY_STOP; retrying the active initial_path from fresh Odometry");
          changeFSMExecState(GEN_NEW_TRAJ, "REFERENCE_RECOVERY");
        }
        else
        {
          RCLCPP_INFO(node_->get_logger(),
                      "Exiting EMERGENCY_STOP; switching to WAIT_TARGET for a new target");
          have_target_ = false;
          trigger_ = false;
          changeFSMExecState(WAIT_TARGET, "EMERGENCY_EXIT");
        }
      }

      flag_escape_emergency_ = false;
      break;
    }
    }

    finishProcess();

    publish_data_display();
  }

  void SCANReplanFSM::finishProcess()
  {
    if (exec_state_ != EMERGENCY_STOP &&
        replan_fail_count_ >= max_replan_fail_count_)
    {
      RCLCPP_WARN(node_->get_logger(),
                  "Replan failed %d times; emergency stop and wait for a new target", replan_fail_count_);
      if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
      {
        const std::int64_t reference_stamp_ns =
            rclcpp::Time(active_reference_path_stamp_).nanoseconds();
        if (reference_stamp_ns > 0)
          global_replan_generation_gate_.require(reference_stamp_ns);
      }
      pending_emergency_reason_ = "连续局部规划失败达到阈值";
      need_hover_stop_ = true;
      flag_escape_emergency_ = true;
      changeFSMExecState(EMERGENCY_STOP, "finishProcess");
    }
  }

  bool SCANReplanFSM::planFromCurrentTraj()
  {
    LocalTrajData *info = &planner_manager_->local_data_;
    rclcpp::Time time_now = node_->now();
    double t_cur = (time_now - info->start_time_).seconds();
    t_cur = std::min(std::max(t_cur, 0.0), info->duration_);

    //cout << "info->velocity_traj_=" << info->velocity_traj_.get_control_points() << endl;

    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
    {
      // 参考路径模式只能更新局部 B-spline，不能把剩余 Path 重建成直达终点的
      // 全局多项式，否则第一次滚动重规划就会切掉楼梯或 90° 折角。四足
      // policy 的实际机体会落后于理论轨迹，因此局部段必须从真实 Odometry
      // 重新初始化，不能继续拼接已经跑到前方的旧 B-spline 尾段。
      start_pt_ = odom_pos_;
      start_vel_ = filtered_odom_vel_;
      start_acc_.setZero();

      bool success = callReboundReplan(true, false);
      if (!success)
      {
        success = callReboundReplan(true, true);
        if (!success)
          return false;
      }
      return true;
    }

    start_pt_ = odom_pos_;
    start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
    start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

    const Eigen::Vector2d to_goal = end_pt_.head<2>() - odom_pos_.head<2>();
    if (to_goal.norm() > 1e-3 && start_vel_.head<2>().dot(to_goal) < 0.0)
    {
      start_vel_.setZero();
      start_acc_.setZero();
    }

    if (!planner_manager_->planGlobalTraj(
            start_pt_,
            start_vel_,
            start_acc_,
            end_pt_,
            Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero()))
    {
      RCLCPP_ERROR(node_->get_logger(),
                   "[navi_mode=%d] Unable to refresh global trajectory from odom to current target", navi_mode_);
      return false;
    }

    if (!adjustGlobalTargetIfOccupied())
      return false;

    bool success = callReboundReplan(true, false);
    if (!success)
    {
      success = callReboundReplan(true, true);
      if (!success)
        return false;
    }

    return true;
  }

  void SCANReplanFSM::setStartStateFromOdomOrCurrentTraj()
  {
    start_pt_ = odom_pos_;
    start_vel_ = navi_mode_ == NAVI_MODE::REFERENCE_PATH ?
        filtered_odom_vel_ : odom_vel_;
    start_acc_.setZero();

    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
      return;

    LocalTrajData *info = &planner_manager_->local_data_;
    if (info->start_time_.seconds() < 1e-5 || info->duration_ <= 1e-5)
      return;

    const double raw_t_cur = (node_->now() - info->start_time_).seconds();
    if (raw_t_cur < -1e-3 || raw_t_cur > info->duration_ + 0.2)
      return;

    const double t_cur = std::min(std::max(raw_t_cur, 0.0), info->duration_);
    start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
    start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

    const Eigen::Vector2d to_goal = end_pt_.head<2>() - odom_pos_.head<2>();
    if (to_goal.norm() > 1e-3 && start_vel_.head<2>().dot(to_goal) < 0.0)
    {
      start_vel_.setZero();
      start_acc_.setZero();
    }
  }

  void SCANReplanFSM::checkCollisionCallback()
  {
    drainOdometryMailbox();
    drainStairExecutionFrozenMailbox();
    updateLocalTrajTimeFreeze();

    // controller 的 go2 freeze 只冻结轨迹时钟；仅楼梯暂停及 fresh-input
    // 恢复屏障能够抑制碰撞线程触发的优化或急停状态迁移。
    if (stair_planning_freeze_gate_.planningInhibited() ||
        global_replan_generation_gate_.required())
      return;

    LocalTrajData *info = &planner_manager_->local_data_;
    auto map = planner_manager_->grid_map_;

    // 只检查正在执行的轨迹。GEN/REPLAN/EMERGENCY 若同时从安全定时器
    // 再次进入规划，会绕过 FSM 的失败节流并形成高频重规划风暴。
    if (exec_state_ != EXEC_TRAJ || info->start_time_.seconds() < 1e-5)
      return;

    /* ---------- check trajectory ---------- */
    constexpr double time_step = 0.01;
    double t_cur = (node_->now() - info->start_time_).seconds();
    double t_2_3 = info->duration_ * 2 / 3;
    for (double t = t_cur; t < info->duration_; t += time_step)
    {
      // reference 模式下 controller 会执行到完整局部末端，动态障碍也必须
      // 立即检查整个剩余时域；其他 community 模式暂保留原 2/3 窗口。
      if (navi_mode_ != NAVI_MODE::REFERENCE_PATH &&
          t_cur < t_2_3 && t >= t_2_3)
        break;

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t);
      Eigen::Vector3d pos_next = info->position_traj_.evaluateDeBoorT(std::min(t + time_step, info->duration_));
      if (map->getInflateOccupancy(pos, estimateYawFromSegment(pos, pos_next)))
      {
        if (planFromCurrentTraj()) // Make a chance
        {
          recovery_status_pending_ = true;
          publishRecoveredIfPending(
              "预测碰撞已由新的局部轨迹解除");
          changeFSMExecState(EXEC_TRAJ, "SAFETY");
          return;
        }
        else
        {
          recordPlanningFailure("预测碰撞后的局部重规划失败");
          if (t - t_cur < emergency_time_) // 0.8s of emergency time
          {
            const std::int64_t reference_stamp_ns =
                rclcpp::Time(active_reference_path_stamp_).nanoseconds();
            if (navi_mode_ == NAVI_MODE::REFERENCE_PATH &&
                reference_stamp_ns > 0)
              global_replan_generation_gate_.require(reference_stamp_ns);
            publishPlanningStatus(
                ScanPlanningEvent::kPredictedCollision,
                "近时域轨迹预测碰撞且局部避障失败");
            RCLCPP_WARN(node_->get_logger(), "Obstacle discovered; emergency stop in %.3fs", t - t_cur);
            pending_emergency_reason_ = "近时域预测碰撞";
            need_hover_stop_ = true;
            changeFSMExecState(EMERGENCY_STOP, "SAFETY");
            // 必须先发布 emergency B-spline，再发布携带相同 identity 的
            // EMERGENCY_STOP 状态；下一拍不得重复制造新轨迹代际。
            callEmergencyStop(odom_pos_);
            deferReferenceRetry();
            flag_escape_emergency_ = false;
          }
          else
          {
            if (replan_fail_count_ >= max_replan_fail_count_)
            {
              pending_emergency_reason_ = "连续局部规划失败达到阈值";
              need_hover_stop_ = true;
              flag_escape_emergency_ = true;
              changeFSMExecState(EMERGENCY_STOP, "SAFETY_FAILURE_LIMIT");
            }
            else
            {
              changeFSMExecState(REPLAN_TRAJ, "SAFETY");
            }
          }
          return;
        }
        break;
      }
    }
  }

  bool SCANReplanFSM::callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj)
  {

    // getLocalTarget 或生命周期门在进入 manager 前失败时，不能沿用上一轮
    // typed failure 误触发主动感知。
    last_local_plan_attempt_reached_manager_ = false;

    if (stair_planning_freeze_gate_.planningInhibited() ||
        global_replan_generation_gate_.required())
      return false;
    if (reference_final_hold_lifecycle_.pending)
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "stationary final hold 等待 ControllerStatus 时禁止规划新轨迹");
      return false;
    }

    if (!getLocalTarget())
      return false;
    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
    {
      // Odometry 的横向侧滑和竖直步态抖动是当前物理状态，不应作为
      // B-spline 的期望硬边界；仅保留有序 guide 可分辨首切线上的前向
      // 分量。毫米级 Path 投影回归点会被跳过，否则时间剖面把它当初切线，
      // 却又强制注入沿主路径的实测速度，触发整段反向拉长。
      const double reference_sample_scale = std::max(
          planner_manager_->grid_map_->getResolution(),
          0.25 * planner_manager_->pp_.ctrl_pt_dist);
      const double minimum_velocity_lookahead = std::max(
          1.0e-3, reference_sample_scale);
      start_vel_ = projectVelocityOntoReferenceGuide(
          start_vel_, start_pt_, local_reference_guide_,
          minimum_velocity_lookahead,
          planner_manager_->pp_.max_vel_);
      start_acc_.setZero();
    }

    last_local_plan_attempt_reached_manager_ = true;
    bool plan_success =
        planner_manager_->reboundReplan(
            start_pt_, start_vel_, start_acc_, local_target_pt_, local_target_vel_,
            (have_new_target_ || flag_use_poly_init), flag_randomPolyTraj,
            navi_mode_ == NAVI_MODE::REFERENCE_PATH,
            local_reference_guide_, local_reference_corridor_guide_);
    have_new_target_ = false;

    cout << "final_plan_success=" << plan_success << endl;

    if (plan_success)
    {

      auto info = &planner_manager_->local_data_;

      /* publish traj */
      scan_planner_msgs::msg::Bspline bspline;
      bspline.header.stamp = node_->now();
      bspline.header.frame_id = expected_frame_id_;
      bspline.order = 3;
      bspline.start_time = info->start_time_;
      bspline.reference_path_stamp = active_reference_path_stamp_;
      bspline.traj_id = info->traj_id_;
      bspline.is_final = local_target_is_final_;
      bspline.emergency_stop = false;

      Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
      bspline.pos_pts.reserve(pos_pts.cols());
      for (int i = 0; i < pos_pts.cols(); ++i)
      {
        geometry_msgs::msg::Point pt;
        pt.x = pos_pts(0, i);
        pt.y = pos_pts(1, i);
        pt.z = pos_pts(2, i);
        bspline.pos_pts.push_back(pt);
      }

      Eigen::VectorXd knots = info->position_traj_.getKnot();
      bspline.knots.reserve(knots.rows());
      for (int i = 0; i < knots.rows(); ++i)
      {
        bspline.knots.push_back(knots(i));
      }

      if (!finalHoldRecoveryTrajectoryIdAllowed(
              reference_final_hold_lifecycle_, bspline.traj_id))
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "final hold 超时恢复必须发布严格更高的 traj_id");
        return false;
      }

      if (!publishBsplineDiagnostics(bspline, *info))
      {
        RCLCPP_ERROR(
            node_->get_logger(),
            "B-spline 诊断构造失败，拒绝发布无法审计的轨迹");
        return false;
      }
      // 诊断已成功构造且更高 identity 即将发布，TIMEOUT recovery 才结束；
      // 规划失败期间继续锁存旧 traj_id，禁止刷新 stationary hold。
      if (reference_final_hold_lifecycle_.recovery_required)
        reference_final_hold_lifecycle_ = FinalHoldLifecycleState{};
      recordPublishedTrajectory(bspline);
      bspline_pub_->publish(bspline);
      publishPlanningStatus(
          ScanPlanningEvent::kTrajectoryPublished,
          "已发布可执行局部 B-spline",
          &last_published_trajectory_);

      visualization_->displayOptimalTraj(info->position_traj_, 0);
    }

    return plan_success;
  }

  bool SCANReplanFSM::tryStartActiveSensing()
  {
    if (!enable_active_sensing_)
      return false;

    const std::int64_t reference_stamp_ns =
        rclcpp::Time(active_reference_path_stamp_).nanoseconds();
    const bool body_rotation_envelope_free =
        planner_manager_ && planner_manager_->grid_map_ &&
        odom_pos_.allFinite() && std::isfinite(getOdomYaw()) &&
        // yaw-only 主动观测必须显式使用双圆柱任意 yaw 扫掠外接圆；当前
        // 配置允许 unknown 通行，因此返回 0 只证明完整旋转包络内无已知
        // 障碍，不声称该包络已被传感器完整观测为 known-free。
        planner_manager_->grid_map_->getRotationSweepOccupancy(
            odom_pos_) == 0;
    const LocalPlanFailureReason failure_reason =
        planner_manager_
            ? planner_manager_->lastLocalPlanFailureReason()
            : LocalPlanFailureReason::None;
    ActiveSensingStartContext context;
    context.failure_reason = failure_reason;
    context.planner_attempted =
        last_local_plan_attempt_reached_manager_;
    context.reference_mode =
        navi_mode_ == NAVI_MODE::REFERENCE_PATH;
    context.reference_path_stamp_ns = reference_stamp_ns;
    context.consumed_reference_path_stamp_ns =
        active_sensing_consumed_path_stamp_ns_;
    context.inputs_fresh = referenceInputsReady();
    context.stair_gate_clear =
        !stair_planning_freeze_gate_.planningInhibited();
    // tryFinishReferenceAtGoal() 已在规划前先收口；这里只禁止打断已经进入的
    // final-hold 生命周期。local_target_is_final_ 仅代表滚动窗口触及终点，
    // 尚未到达时的终点残留占据仍允许触发一次主动复观测。
    context.final_hold_lifecycle_clear =
        !finalHoldBlocksStationaryPublication(
            reference_final_hold_lifecycle_);
    context.global_replan_gate_clear =
        !global_replan_generation_gate_.required();
    context.body_rotation_envelope_free =
        body_rotation_envelope_free;
    context.planar_speed = odom_vel_.head<2>().norm();
    context.maximum_planar_speed = active_sensing_max_planar_speed_;
    if (!have_target_ || active_reference_waypoints_.size() < 2 ||
        !activeSensingMayStart(context))
      return false;

    // 一旦所有启动门满足，先消费该 Path 代际，再构造/发布消息。即使后续
    // publisher、controller 或传感器失败，同一 stamp 也不能再获得第二次
    // 主动观测机会。
    active_sensing_consumed_path_stamp_ns_ = reference_stamp_ns;
    active_sensing_path_stamp_ns_ = reference_stamp_ns;
    active_sensing_start_position_ = odom_pos_;
    active_sensing_start_yaw_ = getOdomYaw();
    active_sensing_target_yaw_ = std::atan2(
        std::sin(
            active_sensing_start_yaw_ + active_sensing_yaw_offset_),
        std::cos(
            active_sensing_start_yaw_ + active_sensing_yaw_offset_));
    active_sensing_phase_ = ACTIVE_SENSING_WAIT_ACCEPTED;
    active_sensing_yaw_stable_since_ns_ = 0;
    active_sensing_observation_baseline_stamp_ns_ = 0;
    active_sensing_fusion_baseline_ = 0;
    active_sensing_fusion_current_ = 0;
    active_sensing_fusion_distinct_ = 0;
    active_sensing_settle_yaw_error_ = 0.0;
    active_sensing_settle_angular_speed_ = 0.0;
    active_sensing_measured_stable_duration_sec_ = 0.0;
    active_sensing_expected_identity_ =
        ActiveSensingTrajectoryIdentity{};
    changeFSMExecState(ACTIVE_SENSING, "ACTIVE_SENSING_START");

    if (!publishActiveSensingTrajectory())
      failActiveSensing("主动感知 yaw-only B-spline 发布失败");
    return true;
  }

  bool SCANReplanFSM::publishActiveSensingTrajectory()
  {
    if (!planner_manager_ ||
        active_sensing_phase_ != ACTIVE_SENSING_WAIT_ACCEPTED ||
        active_sensing_path_stamp_ns_ <= 0 ||
        !active_sensing_start_position_.allFinite())
      return false;

    // 位置轨迹与 emergency/final hold 使用不同发布路径：这里只借用 manager
    // 的严格同点控制点和单调 traj_id，不发布 GOAL_HOLD，也不设置 final 或
    // emergency 标志。
    if (!planner_manager_->EmergencyStop(active_sensing_start_position_))
      return false;
    LocalTrajData *info = &planner_manager_->local_data_;
    const double yaw_duration =
        std::abs(active_sensing_yaw_offset_) /
        active_sensing_yaw_rate_;
    active_sensing_trajectory_duration_sec_ =
        active_sensing_accept_timeout_sec_ + yaw_duration +
        active_sensing_observation_timeout_sec_ +
        active_sensing_safety_margin_sec_;
    const rclcpp::Time publish_time = node_->now();
    const builtin_interfaces::msg::Time publish_stamp = publish_time;
    const auto message = buildActiveSensingTrajectoryMessage(
        expected_frame_id_, publish_stamp,
        active_reference_path_stamp_, info->traj_id_,
        active_sensing_start_position_, active_sensing_start_yaw_,
        active_sensing_yaw_offset_, active_sensing_yaw_rate_,
        active_sensing_trajectory_duration_sec_);
    if (!message.has_value())
      return false;

    Eigen::MatrixXd control_points(
        3, static_cast<Eigen::Index>(message->pos_pts.size()));
    for (std::size_t index = 0; index < message->pos_pts.size(); ++index)
    {
      control_points.col(static_cast<Eigen::Index>(index)) <<
          message->pos_pts[index].x,
          message->pos_pts[index].y,
          message->pos_pts[index].z;
    }
    Eigen::VectorXd knots(
        static_cast<Eigen::Index>(message->knots.size()));
    for (std::size_t index = 0; index < message->knots.size(); ++index)
      knots(static_cast<Eigen::Index>(index)) = message->knots[index];
    UniformBspline stationary(
        control_points, message->order,
        active_sensing_trajectory_duration_sec_ / 3.0);
    stationary.setKnot(knots);
    info->start_time_ = publish_time;
    info->position_traj_ = stationary;
    info->velocity_traj_ = stationary.getDerivative();
    info->acceleration_traj_ = info->velocity_traj_.getDerivative();
    info->start_pos_ = active_sensing_start_position_;
    info->duration_ = stationary.getTimeSum();
    if (!std::isfinite(info->duration_) || info->duration_ <= 0.0 ||
        std::abs(
            info->duration_ - active_sensing_trajectory_duration_sec_) >
            1.0e-9)
      return false;

    active_sensing_expected_identity_ =
        activeSensingIdentityFromBspline(*message);
    active_sensing_publish_stamp_ns_ = publish_time.nanoseconds();
    if (active_sensing_publish_stamp_ns_ <= 0 ||
        !sameActiveSensingTrajectoryIdentity(
            active_sensing_expected_identity_,
            active_sensing_expected_identity_))
      return false;

    ActiveSensingDiagnosticsSnapshot started_snapshot;
    started_snapshot.event =
        scan_planner_msgs::msg::BsplineDiagnostics::
            ACTIVE_SENSING_EVENT_STARTED;
    started_snapshot.start_yaw = active_sensing_start_yaw_;
    started_snapshot.target_yaw = active_sensing_target_yaw_;
    started_snapshot.yaw_offset = active_sensing_yaw_offset_;
    started_snapshot.yaw_rate = active_sensing_yaw_rate_;
    started_snapshot.reason =
        "主动观测 yaw-only B-spline 已发布，等待 controller 接受";
    if (!publishBsplineDiagnostics(
            *message, *info, &started_snapshot))
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "主动感知 B-spline 诊断构造失败，拒绝发布无法审计的轨迹");
      return false;
    }

    recordPublishedTrajectory(*message);
    bspline_pub_->publish(*message);
    publishPlanningStatus(
        ScanPlanningEvent::kTrajectoryPublished,
        "已发布一次性 yaw-only 主动感知 B-spline",
        &last_published_trajectory_);
    RCLCPP_INFO(
        node_->get_logger(),
        "Path stamp=%ld 已消费主动感知机会；等待 controller 精确 ACCEPTED",
        static_cast<long>(active_sensing_path_stamp_ns_));
    return true;
  }

  bool SCANReplanFSM::activeSensingRuntimeSafe(
      std::string &reason) const
  {
    const std::int64_t current_reference_stamp_ns =
        rclcpp::Time(active_reference_path_stamp_).nanoseconds();
    const bool body_rotation_envelope_free =
        planner_manager_ && planner_manager_->grid_map_ &&
        odom_pos_.allFinite() && std::isfinite(getOdomYaw()) &&
        planner_manager_->grid_map_->getRotationSweepOccupancy(
            odom_pos_) == 0;
    const bool latest_identity_matches =
        have_last_published_trajectory_ &&
        sameActiveSensingTrajectoryIdentity(
            active_sensing_expected_identity_,
            activeSensingIdentityFromBspline(
                last_published_trajectory_));
    ActiveSensingRuntimeContext context;
    context.fsm_active =
        exec_state_ == ACTIVE_SENSING &&
        active_sensing_phase_ != ACTIVE_SENSING_IDLE;
    context.expected_reference_path_stamp_ns =
        active_sensing_path_stamp_ns_;
    context.current_reference_path_stamp_ns =
        current_reference_stamp_ns;
    context.path_available =
        active_reference_waypoints_.size() >= 2 && have_target_;
    context.inputs_fresh = referenceInputsReady();
    context.stair_gate_clear =
        !stair_planning_freeze_gate_.planningInhibited();
    context.final_hold_lifecycle_clear =
        !finalHoldBlocksStationaryPublication(
            reference_final_hold_lifecycle_);
    context.global_replan_gate_clear =
        !global_replan_generation_gate_.required();
    context.body_rotation_envelope_free =
        body_rotation_envelope_free;
    context.position_drift =
        (odom_pos_ - active_sensing_start_position_).norm();
    context.maximum_position_drift =
        active_sensing_max_position_drift_;
    context.planar_speed = odom_vel_.head<2>().norm();
    context.maximum_planar_speed = active_sensing_max_planar_speed_;
    context.latest_trajectory_identity_matches =
        latest_identity_matches;
    if (activeSensingMayContinue(context))
      return true;

    if (!context.fsm_active ||
        context.expected_reference_path_stamp_ns <= 0 ||
        context.current_reference_path_stamp_ns !=
            context.expected_reference_path_stamp_ns ||
        !context.path_available)
    {
      reason = "主动感知期间 reference Path identity 失效";
      return false;
    }
    if (!context.inputs_fresh)
    {
      reason = "主动感知期间 Odometry 或点云输入不新鲜";
      return false;
    }
    if (!context.stair_gate_clear)
    {
      reason = "主动感知期间进入楼梯冻结或恢复屏障";
      return false;
    }
    if (!context.global_replan_gate_clear ||
        !context.final_hold_lifecycle_clear)
    {
      reason = "主动感知期间 final/global replan gate 被锁存";
      return false;
    }
    if (!std::isfinite(context.position_drift) ||
        !std::isfinite(context.planar_speed) ||
        !odom_vel_.allFinite() || !std::isfinite(odom_angular_speed_) ||
        context.position_drift > context.maximum_position_drift ||
        context.planar_speed > context.maximum_planar_speed)
    {
      reason = "主动感知期间机体发生超限平移或速度漂移";
      return false;
    }
    if (!context.body_rotation_envelope_free)
    {
      reason = "主动感知期间当前任意 yaw 旋转包络出现已知障碍";
      return false;
    }
    if (!context.latest_trajectory_identity_matches)
    {
      reason = "主动感知轨迹不再是 planner 最新发布 identity";
      return false;
    }
    reason = "主动感知运行时合同出现未知异常";
    return false;
  }

  void SCANReplanFSM::updateActiveSensing()
  {
    std::string unsafe_reason;
    if (!activeSensingRuntimeSafe(unsafe_reason))
    {
      failActiveSensing(unsafe_reason);
      return;
    }
    const std::int64_t now_ns = node_->now().nanoseconds();
    if (now_ns <= 0 || active_sensing_publish_stamp_ns_ <= 0 ||
        now_ns < active_sensing_publish_stamp_ns_)
    {
      failActiveSensing("主动感知 ROS 时钟无效或发生倒退");
      return;
    }
    const double elapsed_sec =
        static_cast<double>(
            now_ns - active_sensing_publish_stamp_ns_) /
        1.0e9;
    if (!std::isfinite(elapsed_sec) ||
        elapsed_sec > active_sensing_trajectory_duration_sec_)
    {
      failActiveSensing("主动感知轨迹总时限已耗尽");
      return;
    }

    if (active_sensing_phase_ == ACTIVE_SENSING_WAIT_ACCEPTED)
    {
      if (elapsed_sec > active_sensing_accept_timeout_sec_)
        failActiveSensing("主动感知等待 controller ACCEPTED 超时");
      return;
    }

    if (active_sensing_phase_ == ACTIVE_SENSING_ROTATING)
    {
      const double yaw_error = std::abs(std::atan2(
          std::sin(active_sensing_target_yaw_ - getOdomYaw()),
          std::cos(active_sensing_target_yaw_ - getOdomYaw())));
      const bool yaw_stable =
          std::isfinite(yaw_error) &&
          yaw_error <= active_sensing_yaw_error_ &&
          odom_angular_speed_ <= active_sensing_max_angular_speed_;
      if (!yaw_stable)
      {
        active_sensing_yaw_stable_since_ns_ = 0;
        const double rotation_deadline =
            active_sensing_accept_timeout_sec_ +
            std::abs(active_sensing_yaw_offset_) /
                active_sensing_yaw_rate_ +
            active_sensing_safety_margin_sec_;
        if (elapsed_sec > rotation_deadline)
          failActiveSensing("主动感知 yaw 或角速度未在时限内稳定");
        return;
      }

      if (active_sensing_yaw_stable_since_ns_ <= 0)
        active_sensing_yaw_stable_since_ns_ = now_ns;
      const double stable_duration =
          static_cast<double>(
              now_ns - active_sensing_yaw_stable_since_ns_) /
          1.0e9;
      if (stable_duration + 1.0e-12 <
          active_sensing_stable_duration_sec_)
        return;

      active_sensing_fusion_baseline_ =
          planner_manager_->grid_map_->fusedObservationSequence();
      active_sensing_fusion_current_ =
          active_sensing_fusion_baseline_;
      active_sensing_fusion_distinct_ = 0U;
      active_sensing_observation_baseline_stamp_ns_ = now_ns;
      active_sensing_settle_yaw_error_ = yaw_error;
      active_sensing_settle_angular_speed_ = odom_angular_speed_;
      active_sensing_measured_stable_duration_sec_ = stable_duration;
      active_sensing_phase_ = ACTIVE_SENSING_WAIT_OBSERVATIONS;
      if (!publishActiveSensingDiagnostics(
              scan_planner_msgs::msg::BsplineDiagnostics::
                  ACTIVE_SENSING_EVENT_YAW_STABLE,
              "yaw 与角速度已连续稳定，建立真实融合基线"))
      {
        failActiveSensing("主动观测 yaw 稳定快照发布失败");
        return;
      }
      RCLCPP_INFO(
          node_->get_logger(),
          "主动感知 yaw 已严格零保持；从 fused sequence=%lu、settle=%ld ns "
          "后等待 3 个不同采集时间戳的真实非空融合",
          static_cast<unsigned long>(
              active_sensing_fusion_baseline_),
          static_cast<long>(
              active_sensing_observation_baseline_stamp_ns_));
      return;
    }

    if (active_sensing_phase_ == ACTIVE_SENSING_WAIT_OBSERVATIONS)
    {
      const FusedObservationEvidence evidence =
          planner_manager_->grid_map_->fusedObservationEvidenceAfter(
              active_sensing_observation_baseline_stamp_ns_,
              active_sensing_fusion_baseline_,
              now_ns, kActiveSensingRequiredFusedObservations);
      if (!evidence.valid)
      {
        failActiveSensing(
            "主动感知 fused observation 时序证据无效、缺失或 ring 被截断");
        return;
      }
      const bool fusion_progressed =
          evidence.current_sequence != active_sensing_fusion_current_ ||
          evidence.distinct_stamp_count !=
              active_sensing_fusion_distinct_;
      active_sensing_fusion_current_ = evidence.current_sequence;
      active_sensing_fusion_distinct_ =
          evidence.distinct_stamp_count;
      if (fusion_progressed &&
          !publishActiveSensingDiagnostics(
              scan_planner_msgs::msg::BsplineDiagnostics::
                  ACTIVE_SENSING_EVENT_FUSION_PROGRESS,
              "主动观测真实非空点云融合进度已更新"))
      {
        failActiveSensing("主动观测融合进度快照发布失败");
        return;
      }
      if (evidence.ready)
      {
        if (!publishActiveSensingDiagnostics(
                scan_planner_msgs::msg::BsplineDiagnostics::
                    ACTIVE_SENSING_EVENT_COMPLETED,
                "主动观测已获得三个新采集时间戳的真实非空融合"))
        {
          failActiveSensing("主动观测完成快照发布失败");
          return;
        }
        resetActiveSensingRuntime();
        trigger_ = true;
        have_new_target_ = true;
        recovery_status_pending_ = true;
        reference_retry_not_before_ = node_->now();
        changeFSMExecState(
            GEN_NEW_TRAJ, "ACTIVE_SENSING_THREE_FUSED_FRAMES");
        RCLCPP_INFO(
            node_->get_logger(),
            "主动感知已获得 %lu 个不同采集时间戳的新真实融合；"
            "仅重试同代 Path 一次",
            static_cast<unsigned long>(
                evidence.distinct_stamp_count));
        return;
      }
      if (active_sensing_observation_baseline_stamp_ns_ <= 0 ||
          now_ns < active_sensing_observation_baseline_stamp_ns_ ||
          static_cast<double>(
              now_ns - active_sensing_observation_baseline_stamp_ns_) /
                  1.0e9 >
              active_sensing_observation_timeout_sec_)
      {
        failActiveSensing(
            "主动感知等待 3 个不同采集时间戳的新真实融合超时；"
            "证据仍不足");
      }
      return;
    }

    failActiveSensing("主动感知内部阶段非法");
  }

  void SCANReplanFSM::processActiveSensingControllerStatus(
      const ControllerStatusMessage &msg)
  {
    if (active_sensing_phase_ == ACTIVE_SENSING_IDLE)
      return;
    if (msg.header.frame_id != expected_frame_id_ ||
        rclcpp::Time(msg.header.stamp).nanoseconds() <= 0)
    {
      failActiveSensing("主动感知收到 frame 或时间戳无效的 controller status");
      return;
    }
    const bool acceptance_observed =
        active_sensing_phase_ != ACTIVE_SENSING_WAIT_ACCEPTED;
    const ActiveSensingControllerAction action =
        decideActiveSensingControllerAction(
            msg, active_sensing_expected_identity_,
            acceptance_observed);
    if (action == ActiveSensingControllerAction::kFailClosed)
    {
      failActiveSensing(
          msg.event == ControllerStatusMessage::EVENT_REJECTED
              ? "主动感知 B-spline 被 controller 拒绝"
              : "主动感知 controller status identity/状态不匹配");
      return;
    }
    if (action == ActiveSensingControllerAction::kAccepted &&
        active_sensing_phase_ == ACTIVE_SENSING_WAIT_ACCEPTED)
    {
      const std::int64_t callback_stamp_ns = node_->now().nanoseconds();
      const std::int64_t accept_timeout_ns =
          rclcpp::Duration::from_seconds(
              active_sensing_accept_timeout_sec_).nanoseconds();
      const std::int64_t total_timeout_ns =
          rclcpp::Duration::from_seconds(
              active_sensing_trajectory_duration_sec_).nanoseconds();
      if (!activeSensingAcceptanceIsTimely(
              callback_stamp_ns, active_sensing_publish_stamp_ns_,
              accept_timeout_ns, total_timeout_ns))
      {
        failActiveSensing(
            "主动感知 controller ACCEPTED 已超过接受或轨迹总截止时间");
        return;
      }
      active_sensing_phase_ = ACTIVE_SENSING_ROTATING;
      active_sensing_yaw_stable_since_ns_ = 0;
      if (!publishActiveSensingDiagnostics(
              scan_planner_msgs::msg::BsplineDiagnostics::
                  ACTIVE_SENSING_EVENT_CONTROLLER_ACCEPTED,
              "controller 已精确接受主动观测 yaw-only identity"))
      {
        failActiveSensing("主动观测 controller 接受快照发布失败");
        return;
      }
      RCLCPP_INFO(
          node_->get_logger(),
          "controller 已精确 ACCEPTED 主动感知 identity；开始原地 yaw-only 旋转");
    }
  }

  void SCANReplanFSM::failActiveSensing(const std::string &reason)
  {
    if (active_sensing_phase_ == ACTIVE_SENSING_IDLE)
      return;
    const std::int64_t failed_path_stamp_ns =
        active_sensing_path_stamp_ns_;
    const bool same_active_generation =
        failed_path_stamp_ns > 0 &&
        rclcpp::Time(active_reference_path_stamp_).nanoseconds() ==
            failed_path_stamp_ns;
    if (!publishActiveSensingDiagnostics(
            scan_planner_msgs::msg::BsplineDiagnostics::
                ACTIVE_SENSING_EVENT_FAILED,
            reason))
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "主动观测失败快照无法绑定原 trajectory identity");
    }
    resetActiveSensingRuntime();

    pending_emergency_reason_ = "主动感知异常，发布严格零速安全轨迹";
    const bool stop_published = callEmergencyStop(odom_pos_);
    if (same_active_generation)
      recordPlanningFailure(reason);
    recovery_status_pending_ = true;
    deferReferenceRetry();

    if (!stop_published)
    {
      need_hover_stop_ = true;
      flag_escape_emergency_ = true;
      changeFSMExecState(
          EMERGENCY_STOP, "ACTIVE_SENSING_STOP_PUBLISH_FAILED");
    }
    else if (global_replan_generation_gate_.required() ||
             replan_fail_count_ >= max_replan_fail_count_)
    {
      need_hover_stop_ = true;
      flag_escape_emergency_ = false;
      changeFSMExecState(
          EMERGENCY_STOP, "ACTIVE_SENSING_FAILURE_LIMIT");
    }
    else if (have_target_ &&
             rclcpp::Time(active_reference_path_stamp_).nanoseconds() > 0)
    {
      trigger_ = true;
      have_new_target_ = true;
      changeFSMExecState(GEN_NEW_TRAJ, "ACTIVE_SENSING_FAIL_CLOSED");
    }
    else
    {
      have_target_ = false;
      trigger_ = false;
      changeFSMExecState(WAIT_TARGET, "ACTIVE_SENSING_PATH_LOST");
    }
    RCLCPP_ERROR(node_->get_logger(), "%s", reason.c_str());
  }

  void SCANReplanFSM::resetActiveSensingRuntime()
  {
    active_sensing_phase_ = ACTIVE_SENSING_IDLE;
    active_sensing_path_stamp_ns_ = 0;
    active_sensing_publish_stamp_ns_ = 0;
    active_sensing_yaw_stable_since_ns_ = 0;
    active_sensing_observation_baseline_stamp_ns_ = 0;
    active_sensing_fusion_baseline_ = 0;
    active_sensing_fusion_current_ = 0;
    active_sensing_fusion_distinct_ = 0;
    active_sensing_start_yaw_ = 0.0;
    active_sensing_target_yaw_ = 0.0;
    active_sensing_settle_yaw_error_ = 0.0;
    active_sensing_settle_angular_speed_ = 0.0;
    active_sensing_measured_stable_duration_sec_ = 0.0;
    active_sensing_trajectory_duration_sec_ = 0.0;
    active_sensing_start_position_.setZero();
    active_sensing_expected_identity_ =
        ActiveSensingTrajectoryIdentity{};
  }

  bool SCANReplanFSM::callEmergencyStop(Eigen::Vector3d stop_pos)
  {

    return publishStationaryTrajectory(stop_pos, false, true);
  }

  void SCANReplanFSM::recordPublishedTrajectory(
      const scan_planner_msgs::msg::Bspline &trajectory)
  {
    last_published_trajectory_ = trajectory;
    have_last_published_trajectory_ = true;
    if (!trajectory.is_final || trajectory.emergency_stop)
      return;
    rememberPublishedFinalIdentity(
        published_final_trajectory_history_,
        FinalHoldTrajectoryIdentity{
            rclcpp::Time(trajectory.reference_path_stamp).nanoseconds(),
            rclcpp::Time(trajectory.header.stamp).nanoseconds(),
            rclcpp::Time(trajectory.start_time).nanoseconds(),
            trajectory.traj_id});
  }

  bool SCANReplanFSM::publishStationaryTrajectory(
      const Eigen::Vector3d &stop_pos,
      const bool is_final,
      const bool emergency_stop)
  {

    planner_manager_->EmergencyStop(stop_pos);

    auto info = &planner_manager_->local_data_;

    /* publish traj */
    scan_planner_msgs::msg::Bspline bspline;
    bspline.header.stamp = node_->now();
    bspline.header.frame_id = expected_frame_id_;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.reference_path_stamp = active_reference_path_stamp_;
    bspline.traj_id = info->traj_id_;
    bspline.is_final = is_final;
    bspline.emergency_stop = emergency_stop;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::msg::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    if (!publishBsplineDiagnostics(bspline, *info))
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "静止 B-spline 诊断构造失败，拒绝发布无法审计的轨迹");
      return false;
    }
    recordPublishedTrajectory(bspline);
    bspline_pub_->publish(bspline);
    publishPlanningStatus(
        emergency_stop ? ScanPlanningEvent::kEmergencyStop :
                         ScanPlanningEvent::kGoalHold,
        emergency_stop ? pending_emergency_reason_ :
                         "完整 reference Path 终点已发布静止 final hold",
        &last_published_trajectory_);

    return true;
  }

  bool SCANReplanFSM::tryFinishReferenceAtGoal()
  {
    if (navi_mode_ != NAVI_MODE::REFERENCE_PATH ||
        active_reference_waypoints_.empty() || !have_target_ ||
        finalHoldBlocksStationaryPublication(
            reference_final_hold_lifecycle_) ||
        !referenceInputsReady())
      return false;

    if (!referenceGoalHoldSampleReady() ||
        reference_goal_hold_dwell_.stable_duration_sec + 1.0e-12 <
        reference_goal_hold_stable_dwell_sec_)
      return false;

    const double distance_xy =
        (end_pt_.head<2>() - odom_pos_.head<2>()).norm();
    const double distance_z = std::abs(end_pt_(2) - odom_pos_(2));
    const double yaw_error = std::abs(std::atan2(
        std::sin(getReferenceGoalYaw() - getOdomYaw()),
        std::cos(getReferenceGoalYaw() - getOdomYaw())));
    const double planar_speed = odom_vel_.head<2>().norm();
    const double vertical_speed = std::abs(odom_vel_(2));
    if (!referenceGoalHoldReady(
            true, true, true,
            distance_xy, distance_z, yaw_error,
            planar_speed, vertical_speed, odom_yaw_rate_,
            reference_goal_hold_distance_xy_,
            reference_goal_hold_distance_z_,
            reference_goal_hold_yaw_error_,
            reference_goal_hold_planar_speed_,
            reference_goal_hold_vertical_speed_,
            reference_goal_hold_yaw_rate_))
      return false;

    // 真实 Odometry 已证明到达目标时，终点完成优先于尚未发出的全局重规划。
    global_replan_generation_gate_.reset();

    // 机器人已由真实 Odometry 证明到达完整 Path 终点。这里发布以当前
    // 机体位置为常值的 final B-spline，只负责让 controller 复核稳定并
    // 锁存 GOAL_REACHED，不会要求机器人穿过终点附近的占据体素。
    if (!publishStationaryTrajectory(odom_pos_, true, false))
      return false;

    reference_final_hold_lifecycle_ = FinalHoldLifecycleState{
        true, false, 0};
    reference_goal_hold_dwell_ = ReferenceGoalHoldDwellState{};
    local_target_pt_ = odom_pos_;
    local_target_vel_.setZero();
    local_target_is_final_ = true;
    // target 与完整 Path 必须保留到 controller 对这条精确 B-spline identity
    // 发布 GOAL_REACHED；否则 hard timeout 后 planner 会困在 WAIT_TARGET。
    have_new_target_ = false;
    replan_fail_count_ = 0;
    need_hover_stop_ = false;
    flag_escape_emergency_ = false;
    reference_retry_not_before_ = node_->now();
    RCLCPP_INFO(
        node_->get_logger(),
        "真实 Odometry 已到达完整 initial_path 终点；发布静止 final hold，"
        "等待匹配 ControllerStatus");
    changeFSMExecState(WAIT_TARGET, "REFERENCE_GOAL_HOLD");
    return true;
  }

  bool SCANReplanFSM::referenceGoalHoldSampleReady() const
  {
    if (navi_mode_ != NAVI_MODE::REFERENCE_PATH ||
        active_reference_waypoints_.empty() || !have_target_ || !have_odom_)
      return false;
    const double distance_xy =
        (end_pt_.head<2>() - odom_pos_.head<2>()).norm();
    const double distance_z = std::abs(end_pt_(2) - odom_pos_(2));
    const double yaw_error = std::abs(std::atan2(
        std::sin(getReferenceGoalYaw() - getOdomYaw()),
        std::cos(getReferenceGoalYaw() - getOdomYaw())));
    return referenceGoalHoldReady(
        true, true, true,
        distance_xy, distance_z, yaw_error,
        odom_vel_.head<2>().norm(), std::abs(odom_vel_(2)),
        odom_yaw_rate_,
        reference_goal_hold_distance_xy_,
        reference_goal_hold_distance_z_,
        reference_goal_hold_yaw_error_,
        reference_goal_hold_planar_speed_,
        reference_goal_hold_vertical_speed_,
        reference_goal_hold_yaw_rate_);
  }

  void SCANReplanFSM::updateReferenceGoalHoldDwell(
      const std::int64_t odom_stamp_ns)
  {
    scan_planner::updateReferenceGoalHoldDwell(
        reference_goal_hold_dwell_, referenceGoalHoldSampleReady(),
        odom_stamp_ns,
        rclcpp::Time(active_reference_path_stamp_).nanoseconds(),
        reference_goal_hold_stable_dwell_sec_, input_timeout_sec_);
  }

  bool SCANReplanFSM::referenceRetryIsReady() const
  {
    return referenceRetryReady(
        navi_mode_ == NAVI_MODE::REFERENCE_PATH,
        node_->now().seconds(),
        reference_retry_not_before_.seconds());
  }

  void SCANReplanFSM::deferReferenceRetry()
  {
    if (navi_mode_ != NAVI_MODE::REFERENCE_PATH)
      return;
    reference_retry_not_before_ =
        node_->now() + rclcpp::Duration::from_seconds(reference_retry_period_sec_);
  }

  bool SCANReplanFSM::getLocalTarget()
  {
    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH)
    {
      local_reference_guide_.clear();
      local_reference_corridor_guide_.clear();
      local_target_pt_ = start_pt_;
      local_target_vel_.setZero();
      local_target_is_final_ = false;
      if (active_reference_waypoints_.size() < 2 ||
          active_reference_arc_lengths_.size() !=
              active_reference_waypoints_.size())
      {
        RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "reference Path 弧长状态无效，拒绝生成局部目标");
        return false;
      }

      const double total_path_length =
          active_reference_arc_lengths_.back();
      // 只在当前单调进度之后的一个局部视距内寻找投影。这样即使上下楼层
      // 的 XY 重叠或路径回转，也不会跳到远处的同名几何段。
      const double projection_window = std::max(
          planning_horizon_,
          std::max(2.0 * std::max(0.0, replan_thresh_),
                   min_path_point_spacing_));
      const ReferencePathProjection projection =
          projectReferencePathProgress(
              active_reference_waypoints_, active_reference_arc_lengths_,
              start_pt_, reference_progress_s_,
              reference_progress_s_ + projection_window);
      if (!projection.valid)
      {
        RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "无法把当前 Odometry 投影到 active reference Path");
        return false;
      }
      if (projection.distance > reference_projection_max_distance_)
      {
        RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "当前 Odometry 距 reference Path %.3f m，超过 %.3f m 安全门；"
            "保持已认证进度并拒绝规划",
            projection.distance, reference_projection_max_distance_);
        return false;
      }
      reference_progress_s_ = std::max(
          reference_progress_s_, projection.progress_s);

      double target_s = std::min(
          total_path_length,
          reference_progress_s_ + planning_horizon_);
      if (!sampleReferencePathAtArcLength(
              active_reference_waypoints_, active_reference_arc_lengths_,
              target_s, local_target_pt_))
      {
        local_target_pt_ = start_pt_;
        RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "无法按弧长采样 active reference Path");
        return false;
      }

      const double map_resolution =
          planner_manager_->grid_map_->getResolution();
      const double candidate_step = std::max(
          min_path_point_spacing_, map_resolution);
      const double runway_sample_step = 0.5 * map_resolution;

      auto targetOccupancyAtArcLength =
          [&](const double candidate_s, const Eigen::Vector3d &candidate) {
            Eigen::Vector3d previous = start_pt_;
            sampleReferencePathAtArcLength(
                active_reference_waypoints_, active_reference_arc_lengths_,
                std::max(reference_progress_s_, candidate_s - candidate_step),
                previous);
            return planner_manager_->grid_map_->getInflateOccupancy(
                candidate, estimateYawFromSegment(previous, candidate));
          };

      const double nominal_target_s = target_s;
      auto reference_is_free_at_progress = [&](const double candidate_s) {
        Eigen::Vector3d candidate;
        return sampleReferencePathAtArcLength(
                   active_reference_waypoints_,
                   active_reference_arc_lengths_, candidate_s,
                   candidate) &&
               targetOccupancyAtArcLength(candidate_s, candidate) == 0;
      };

      // 严格捕获点可能比精确 Path 终点提前数厘米进入当前局部窗口。若仍
      // 只等 nominal_target_s 精确触及终点，本条非 final 轨迹会带着巡航
      // 速度穿过捕获圈，下一轮重规划才制动。始终审计末段候选，但只有
      // 候选不超过本次 nominal 前视进度时才允许提前进入 final。
      const double capture_search_start = std::max(
          reference_progress_s_, total_path_length - planning_horizon_);
      auto reference_terminal_capture_candidate =
          [&](const double candidate_s) {
            Eigen::Vector3d candidate;
            if (!sampleReferencePathAtArcLength(
                    active_reference_waypoints_,
                    active_reference_arc_lengths_, candidate_s,
                    candidate))
              return false;
            const double distance_xy =
                (end_pt_.head<2>() - candidate.head<2>()).norm();
            const double distance_z =
                std::abs(end_pt_(2) - candidate(2));
            return distance_xy <=
                       reference_goal_hold_distance_xy_ + 1.0e-9 &&
                   distance_z <=
                       reference_goal_hold_distance_z_ + 1.0e-9 &&
                   planner_manager_->grid_map_->getInflateOccupancy(
                       candidate, getReferenceGoalYaw()) == 0;
          };
      const ReferenceTargetSelection terminal_capture_selection =
          selectReferenceTerminalCaptureTarget(
              capture_search_start, total_path_length,
              runway_sample_step, runway_sample_step,
              reference_target_free_runway_,
              reference_terminal_capture_candidate,
              reference_is_free_at_progress);

      const bool terminal_capture_target =
          terminalCaptureTargetFitsLocalWindow(
              terminal_capture_selection, reference_progress_s_,
              nominal_target_s);
      const ReferenceTargetSelection target_selection =
          terminal_capture_target ? terminal_capture_selection :
          selectReferenceTargetWithFreeRunway(
              nominal_target_s, reference_progress_s_, total_path_length,
              planning_horizon_, candidate_step, runway_sample_step,
              reference_target_free_runway_,
              reference_is_free_at_progress);
      if (!target_selection.valid ||
          !sampleReferencePathAtArcLength(
              active_reference_waypoints_, active_reference_arc_lengths_,
              target_selection.progress_s, local_target_pt_))
      {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "reference 局部目标及其 B-spline 末端支撑区被占据，"
            "当前前后搜索窗口内没有连续自由区");
        return false;
      }
      target_s = target_selection.progress_s;
      const bool selected_target_terminal_capture =
          !terminal_capture_target &&
          selectedReferenceTargetQualifiesAsTerminalCapture(
              target_selection, reference_progress_s_, nominal_target_s,
              reference_terminal_capture_candidate);
      if (target_selection.search_direction != 0)
      {
        RCLCPP_WARN_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "reference 局部目标已向%s调整到具备 %.2f m 连续自由支撑区的位置",
            target_selection.search_direction > 0 ? "前" : "后",
            reference_target_free_runway_);
      }
      if (selected_target_terminal_capture)
      {
        RCLCPP_INFO(
            node_->get_logger(),
            "普通自由支撑区搜索命中严格终点捕获点；本条轨迹升级为 final");
      }

      local_target_is_final_ = terminal_capture_target ||
          selected_target_terminal_capture ||
          target_s >= total_path_length - 1.0e-6;
      const double selected_target_distance_xy =
          (end_pt_.head<2>() - local_target_pt_.head<2>()).norm();
      const double selected_target_distance_z =
          std::abs(end_pt_(2) - local_target_pt_(2));
      const bool nonfinal_terminal_brake =
          nonFinalReferenceTargetRequiresTerminalBrake(
              local_target_is_final_, selected_target_distance_xy,
              selected_target_distance_z,
              reference_goal_hold_distance_xy_,
              reference_goal_hold_distance_z_);
      local_reference_guide_ = buildReferencePathGuide(
          active_reference_waypoints_, active_reference_arc_lengths_,
          start_pt_, reference_progress_s_, target_s);
      local_reference_corridor_guide_ = buildReferencePathCorridorGuide(
          active_reference_waypoints_, active_reference_arc_lengths_,
          reference_progress_s_, target_s);
      if (local_reference_guide_.size() < 2 ||
          local_reference_corridor_guide_.size() < 2)
      {
        local_reference_guide_.clear();
        local_reference_corridor_guide_.clear();
        RCLCPP_ERROR_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 1000,
            "reference 初始化折线或 PCT semantic 走廊少于两个有效点，拒绝规划");
        return false;
      }
      if (!local_target_is_final_ && !nonfinal_terminal_brake)
      {
        local_target_vel_ = referenceCruiseVelocityAlongGuide(
            local_reference_guide_, reference_cruise_speed_,
            planner_manager_->pp_.max_vel_);
        if (reference_cruise_speed_ > 1.0e-6 &&
            local_target_vel_.norm() <= 1.0e-6)
        {
          RCLCPP_ERROR_THROTTLE(
              node_->get_logger(), *node_->get_clock(), 1000,
              "reference 局部折线没有有效末切线，拒绝生成巡航轨迹");
          return false;
        }
      }
      else
      {
        // final 使用零末速完成终端对齐；若普通搜索退到严格位置门内但
        // terminal yaw 仍被占据，则保持 non-final 语义并同样先制动，禁止
        // 以巡航末速穿过目标。后者只能由真实 Odometry 的 stationary hold
        // 或下一轮安全规划完成，不能借零末速提前报告 GOAL_REACHED。
        local_target_vel_.setZero();
        if (nonfinal_terminal_brake)
        {
          RCLCPP_WARN(
              node_->get_logger(),
              "non-final reference 目标已进入严格终点位置门，但 terminal yaw "
              "尚未通过安全门；本条 B-spline 强制零末速度制动");
        }
      }
      return true;
    }

    const double max_vel = planner_manager_->pp_.max_vel_;
    const double max_acc = planner_manager_->pp_.max_acc_;
    const double duration = planner_manager_->global_data_.global_duration_;
    double t_step = max_vel > 1.0e-6
                        ? planning_horizon_ / 20.0 / max_vel
                        : 0.01;
    t_step = std::max(t_step, 0.01);

    // 先把当前规划起点投影到全局参考轨迹，再按轨迹弧长选择前视点。
    // 仅按欧氏距离选点会在 U 形、楼梯回转和 90° 转弯处跨段抄近路。
    double t_proj = 0.0;
    double min_dist_to_start = std::numeric_limits<double>::max();
    for (double t = 0.0; t < duration; t += t_step)
    {
      const Eigen::Vector3d pos_t =
          planner_manager_->global_data_.getPosition(t);
      const double dist_to_start = (pos_t - start_pt_).norm();
      if (dist_to_start < min_dist_to_start)
      {
        min_dist_to_start = dist_to_start;
        t_proj = t;
      }
    }

    double target_t = duration;
    double total_dist = 0.0;
    bool target_found = false;
    Eigen::Vector3d prev_pos =
        planner_manager_->global_data_.getPosition(t_proj);
    local_target_pt_ = end_pt_;
    local_target_is_final_ = true;

    for (double t = t_proj; t < duration; t += t_step)
    {
      const Eigen::Vector3d pos_t =
          planner_manager_->global_data_.getPosition(t);
      total_dist += (pos_t - prev_pos).norm();
      if (total_dist >= planning_horizon_)
      {
        local_target_pt_ = pos_t;
        target_t = t;
        target_found = true;
        local_target_is_final_ = false;
        break;
      }
      prev_pos = pos_t;
    }
    planner_manager_->global_data_.last_progress_time_ =
        target_found ? target_t : duration;

    auto targetOccupancy = [&](const Eigen::Vector3d &pt) {
      return planner_manager_->grid_map_->getInflateOccupancy(pt, estimateYawFromSegment(odom_pos_, pt));
    };

    if (targetOccupancy(local_target_pt_) != 0)
    {
      bool found_free_target = false;
      double adjusted_t = target_t;

      for (double dt = 0.0; dt <= planner_manager_->global_data_.global_duration_; dt += t_step)
      {
        double t_forward = target_t + dt;
        if (t_forward <= planner_manager_->global_data_.global_duration_)
        {
          Eigen::Vector3d pt = planner_manager_->global_data_.getPosition(t_forward);
          if (targetOccupancy(pt) == 0)
          {
            local_target_pt_ = pt;
            adjusted_t = t_forward;
            found_free_target = true;
            break;
          }
        }

        double t_backward = target_t - dt;
        if (t_backward >= std::max(0.0, t_proj))
        {
          Eigen::Vector3d pt = planner_manager_->global_data_.getPosition(t_backward);
          if (targetOccupancy(pt) == 0)
          {
            local_target_pt_ = pt;
            adjusted_t = t_backward;
            found_free_target = true;
            break;
          }
        }
      }

      if (found_free_target)
      {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                             "Local target was adjusted to a nearby collision-free point");
        target_t = adjusted_t;
        local_target_is_final_ = (end_pt_ - local_target_pt_).norm() < 1.0e-3;
      }
      else
      {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                             "Local target is in collision and no nearby free target was found");
      }
    }

    if (navi_mode_ == NAVI_MODE::REFERENCE_PATH && !local_target_is_final_)
    {
      // 0.60 m 短滚动窗口若继承全局轨迹约 0.50 m/s 的终端速度，
      // community 参数化会为了同时满足长持续时间和高末速而插入回退段。
      // FSM 在到达窗口端点前便触发下一次重规划，因此保守的零终端速度
      // 不会把在线执行变成逐段停车，却能保持局部几何单调向前。
      local_target_vel_ = Eigen::Vector3d::Zero();
    }
    else if ((end_pt_ - local_target_pt_).norm() <
        (max_vel * max_vel) / (2 * std::max(max_acc, 1.0e-6)))
    {
      // local_target_vel_ = (end_pt_ - init_pt_).normalized() * planner_manager_->pp_.max_vel_ * (( end_pt_ - local_target_pt_ ).norm() / ((planner_manager_->pp_.max_vel_*planner_manager_->pp_.max_vel_)/(2*planner_manager_->pp_.max_acc_)));
      // cout << "A" << endl;
      local_target_vel_ = Eigen::Vector3d::Zero();
    }
    else
    {
      local_target_vel_ = planner_manager_->global_data_.getVelocity(target_t);
      if (local_target_vel_.norm() > max_vel && max_vel > 1.0e-6)
        local_target_vel_ = local_target_vel_.normalized() * max_vel;
      // cout << "AA" << endl;
    }
    return true;
  }

} // namespace scan_planner
