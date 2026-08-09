#ifndef _SCAN_REPLAN_FSM_H_
#define _SCAN_REPLAN_FSM_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <builtin_interfaces/msg/time.hpp>
#include <cstdint>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <iostream>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>

#include <bspline_opt/bspline_optimizer.h>
#include <plan_env/grid_map.h>
#include <plan_manage/active_sensing.h>
#include <plan_manage/odometry_mailbox.h>
#include <plan_manage/planning_status.h>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/bspline_diagnostics.hpp>
#include <scan_planner_msgs/msg/controller_status.hpp>
#include <scan_planner_msgs/msg/data_disp.hpp>
#include <scan_planner_msgs/msg/scan_planning_status.hpp>
#include <scan_planner_msgs/msg/stair_execution_freeze.hpp>
#include <plan_manage/planner_manager.h>
#include <plan_manage/reference_execution.h>
#include <plan_manage/reference_velocity.h>
#include <plan_manage/trajectory_diagnostics.h>
#include <traj_utils/planning_visualization.h>

using std::vector;

namespace scan_planner
{

  class SCANReplanFSM
  {

  private:
    /* ---------- flag ---------- */
    enum FSM_EXEC_STATE
    {
      INIT,
      WAIT_TARGET,
      GEN_NEW_TRAJ,
      REPLAN_TRAJ,
      EXEC_TRAJ,
      ACTIVE_SENSING,
      EMERGENCY_STOP
    };
    enum ACTIVE_SENSING_PHASE
    {
      ACTIVE_SENSING_IDLE,
      ACTIVE_SENSING_WAIT_ACCEPTED,
      ACTIVE_SENSING_ROTATING,
      ACTIVE_SENSING_WAIT_OBSERVATIONS,
    };
    enum NAVI_MODE
    {
      MANUAL_TARGET = 1,
      PRESET_TARGET = 2,
      REFERENCE_PATH = 3,
    };

    /* planning utils */
    SCANPlannerManager::Ptr planner_manager_;
    PlanningVisualization::Ptr visualization_;
    scan_planner_msgs::msg::DataDisp data_disp_;

    /* parameters */
    int navi_mode_; // 1 manual select, 2 hard code
    double no_replan_thresh_, replan_thresh_;
    std::vector<Eigen::Vector3d> preset_waypoints_;
    int waypoint_num_;
    double planning_horizon_;
    double emergency_time_;
    double rviz_goal_height_;
    double self_inflation_z_up_, self_inflation_z_down_;
    double self_double_cylinder_radius_, self_double_cylinder_offset_;
    double body_height_;
    double input_timeout_sec_;
    double stair_execution_freeze_timeout_sec_;
    double stair_execution_freeze_confirmation_sec_;
    double min_path_point_spacing_;
    double reference_projection_max_distance_;
    double reference_target_free_runway_;
    double reference_cruise_speed_;
    double reference_velocity_filter_time_constant_sec_;
    int max_reference_path_points_;
    double reference_retry_period_sec_;
    double final_trajectory_convergence_grace_sec_;
    double reference_goal_hold_distance_xy_;
    double reference_goal_hold_distance_z_;
    double reference_goal_hold_yaw_error_;
    double reference_goal_hold_planar_speed_;
    double reference_goal_hold_vertical_speed_;
    double reference_goal_hold_yaw_rate_;
    double reference_goal_hold_stable_dwell_sec_;
    bool enable_active_sensing_;
    double active_sensing_yaw_offset_;
    double active_sensing_yaw_rate_;
    double active_sensing_max_planar_speed_;
    double active_sensing_yaw_error_;
    double active_sensing_max_angular_speed_;
    double active_sensing_stable_duration_sec_;
    double active_sensing_max_position_drift_;
    double active_sensing_accept_timeout_sec_;
    double active_sensing_observation_timeout_sec_;
    double active_sensing_safety_margin_sec_;
    std::string self_inflation_frame_id_;
    std::string expected_frame_id_;
    std::string expected_base_frame_id_;
    std::string stair_execution_frozen_topic_;
    std::string controller_status_topic_;
    std::string planning_status_topic_;
    std::string bspline_diagnostics_topic_;
    int trajectory_diagnostic_max_samples_;
    int trajectory_diagnostic_history_depth_;

    /* planning data */
    bool trigger_, have_target_, have_odom_, have_new_target_;
    bool preset_started_{false};
    bool rviz_height_ready_;
    bool go2_execution_frozen_;
    bool enable_fail_safe_, need_hover_stop_;
    FSM_EXEC_STATE exec_state_;
    int continuously_called_times_{0};
    int replan_fail_count_{0};
    int max_replan_fail_count_{1000};
    int64_t last_odom_stamp_ns_{0};
    int64_t latest_reference_path_stamp_ns_{0};
    std::uint64_t planning_status_sequence_{0};
    std::uint64_t trajectory_diagnostics_sequence_{0};
    bool initial_planning_status_published_{false};
    bool latest_reference_generation_is_empty_{false};
    bool recovery_status_pending_{false};
    std::string pending_emergency_reason_{"SCAN emergency stop"};
    rclcpp::Time last_freeze_update_time_;
    rclcpp::Time reference_retry_not_before_;
    StairPlanningFreezeGate stair_planning_freeze_gate_;
    ReferenceVelocityLowPassFilter reference_velocity_filter_;
    GlobalReplanGenerationGate global_replan_generation_gate_;
    scan_planner_msgs::msg::Bspline last_published_trajectory_;
    bool have_last_published_trajectory_{false};
    std::vector<FinalHoldTrajectoryIdentity>
        published_final_trajectory_history_;
    FinalHoldTrajectoryIdentity controller_accepted_final_identity_;
    bool have_controller_accepted_final_identity_{false};
    std::vector<Eigen::Vector3d> pending_reference_waypoints_;
    std::vector<Eigen::Vector3d> active_reference_waypoints_;
    std::vector<double> active_reference_arc_lengths_;
    std::vector<Eigen::Vector3d> local_reference_guide_;
    std::vector<Eigen::Vector3d> local_reference_corridor_guide_;
    double reference_progress_s_{0.0};
    double pending_reference_goal_yaw_{0.0};
    double active_reference_goal_yaw_{0.0};
    builtin_interfaces::msg::Time pending_reference_path_stamp_;
    builtin_interfaces::msg::Time active_reference_path_stamp_;

    Eigen::Vector3d odom_pos_, odom_vel_, filtered_odom_vel_, odom_acc_; // odometry state
    Eigen::Quaterniond odom_orient_;
    double odom_angular_speed_{0.0};
    double odom_yaw_rate_{0.0};

    Eigen::Vector3d init_pt_, start_pt_, start_vel_, start_acc_, start_yaw_; // start state
    Eigen::Vector3d end_pt_, end_vel_;                                       // goal state
    Eigen::Vector3d local_target_pt_, local_target_vel_;                     // local target state
    bool local_target_is_final_{false};
    FinalHoldLifecycleState reference_final_hold_lifecycle_;
    ReferenceGoalHoldDwellState reference_goal_hold_dwell_;
    ACTIVE_SENSING_PHASE active_sensing_phase_{ACTIVE_SENSING_IDLE};
    bool last_local_plan_attempt_reached_manager_{false};
    std::int64_t active_sensing_consumed_path_stamp_ns_{0};
    std::int64_t active_sensing_path_stamp_ns_{0};
    std::int64_t active_sensing_publish_stamp_ns_{0};
    std::int64_t active_sensing_yaw_stable_since_ns_{0};
    std::int64_t active_sensing_observation_baseline_stamp_ns_{0};
    std::uint64_t active_sensing_fusion_baseline_{0};
    std::uint64_t active_sensing_fusion_current_{0};
    std::uint64_t active_sensing_fusion_distinct_{0};
    double active_sensing_start_yaw_{0.0};
    double active_sensing_target_yaw_{0.0};
    double active_sensing_settle_yaw_error_{0.0};
    double active_sensing_settle_angular_speed_{0.0};
    double active_sensing_measured_stable_duration_sec_{0.0};
    double active_sensing_trajectory_duration_sec_{0.0};
    Eigen::Vector3d active_sensing_start_position_{Eigen::Vector3d::Zero()};
    ActiveSensingTrajectoryIdentity active_sensing_expected_identity_;
    std::vector<Eigen::Vector3d> active_waypoints_;
    int current_wp_;

    bool flag_escape_emergency_;

    /* ROS utils */
    rclcpp::Node *node_{nullptr};
    rclcpp::TimerBase::SharedPtr exec_timer_, safety_timer_;
    rclcpp::CallbackGroup::SharedPtr odometry_callback_group_;
    rclcpp::CallbackGroup::SharedPtr stair_execution_frozen_callback_group_;
    OdometryMailbox odometry_mailbox_;
    StairExecutionFreezeMailbox stair_execution_frozen_mailbox_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr go2_execution_frozen_sub_;
    rclcpp::Subscription<scan_planner_msgs::msg::StairExecutionFreeze>::SharedPtr
        stair_execution_frozen_sub_;
    rclcpp::Subscription<scan_planner_msgs::msg::ControllerStatus>::SharedPtr
        controller_status_sub_;
    rclcpp::Publisher<scan_planner_msgs::msg::Bspline>::SharedPtr bspline_pub_;
    rclcpp::Publisher<scan_planner_msgs::msg::BsplineDiagnostics>::SharedPtr
        bspline_diagnostics_pub_;
    rclcpp::Publisher<scan_planner_msgs::msg::DataDisp>::SharedPtr data_disp_pub_;
    rclcpp::Publisher<scan_planner_msgs::msg::ScanPlanningStatus>::SharedPtr
        planning_status_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr self_inflation_pub_;

    /* helper functions */
    bool callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj); // front-end and back-end method
    bool callEmergencyStop(Eigen::Vector3d stop_pos);                          // front-end and back-end method
    void recordPublishedTrajectory(
        const scan_planner_msgs::msg::Bspline &trajectory);
    bool publishStationaryTrajectory(
        const Eigen::Vector3d &stop_pos, bool is_final, bool emergency_stop);
    bool tryStartActiveSensing();
    bool publishActiveSensingTrajectory();
    void updateActiveSensing();
    void failActiveSensing(const std::string &reason);
    void resetActiveSensingRuntime();
    bool activeSensingRuntimeSafe(std::string &reason) const;
    void processActiveSensingControllerStatus(
        const scan_planner_msgs::msg::ControllerStatus &msg);
    bool tryFinishReferenceAtGoal();
    bool referenceGoalHoldSampleReady() const;
    void updateReferenceGoalHoldDwell(std::int64_t odom_stamp_ns);
    bool referenceRetryIsReady() const;
    void deferReferenceRetry();
    bool planFromCurrentTraj();
    void setStartStateFromOdomOrCurrentTraj();
    void recordPlanningFailure(const std::string &reason);
    void publishRecoveredIfPending(const std::string &reason);
    void publishPlanningStatus(
        ScanPlanningEvent event,
        const std::string &reason,
        const scan_planner_msgs::msg::Bspline *trajectory = nullptr);
    bool publishBsplineDiagnostics(
        const scan_planner_msgs::msg::Bspline &trajectory,
        const LocalTrajData &trajectory_data,
        const ActiveSensingDiagnosticsSnapshot *active_snapshot = nullptr);
    bool publishActiveSensingDiagnostics(
        std::uint8_t event, const std::string &reason);

    /* return value: std::pair< Times of the same state be continuously called, current continuously called state > */
    void changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call);
    std::pair<int, SCANReplanFSM::FSM_EXEC_STATE> timesOfConsecutiveStateCalls();
    void printFSMExecState();

    void planGlobalTrajbyGivenWps();
    bool planGlobalTrajByWaypoints(const std::vector<Eigen::Vector3d> &waypoints);
    bool planNextWaypoint();
    bool isWaypointSequenceMode() const;
    bool adjustGlobalTargetIfOccupied();
    bool referenceInputsReady() const;
    bool stairResumeInputsReady() const;
    void refreshStairExecutionFreezeFreshness();
    void updateReferenceProgressDuringStairFreeze();
    void forceReferenceReplanAfterStairResume();
    void tryActivatePendingReferencePath();
    bool validateAndConvertReferencePath(
        const nav_msgs::msg::Path &path,
        std::vector<Eigen::Vector3d> &waypoints,
        double &goal_yaw) const;
    bool getLocalTarget();
    void finishProcess();
    void publishSelfInflationMarker();
    double getOdomYaw() const;
    double getReferenceGoalYaw() const;
    double estimateYawFromSegment(const Eigen::Vector3d &from, const Eigen::Vector3d &to) const;
    void updateLocalTrajTimeFreeze();

    /* ROS functions */
    void execFSMCallback();
    void checkCollisionCallback();
    void rvizGoalCallback(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &msg);
    void waypointCallback(const nav_msgs::msg::Path::ConstSharedPtr &msg);
    void pathCallback(const nav_msgs::msg::Path::ConstSharedPtr &msg);
    void odometryCallback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg);
    void drainOdometryMailbox();
    void processOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr &msg);
    void go2ExecutionFrozenCallback(const std_msgs::msg::Bool::ConstSharedPtr &msg);
    void controllerStatusCallback(
        const scan_planner_msgs::msg::ControllerStatus::ConstSharedPtr &msg);
    void stairExecutionFrozenCallback(
        const scan_planner_msgs::msg::StairExecutionFreeze::ConstSharedPtr &msg);
    void drainStairExecutionFrozenMailbox();
    void processStairExecutionFrozenSnapshot(
        const StairExecutionFreezeMailboxMessage &message);

    bool checkCollision();

  public:
    SCANReplanFSM(/* args */)
    {
    }
    ~SCANReplanFSM()
    {
    }

    void init(rclcpp::Node *node);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

} // namespace scan_planner

#endif
