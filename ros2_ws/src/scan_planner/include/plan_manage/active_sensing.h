#ifndef SCAN_PLANNER__ACTIVE_SENSING_H_
#define SCAN_PLANNER__ACTIVE_SENSING_H_

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <plan_manage/local_plan_failure_reason.h>
#include <scan_planner_msgs/msg/bspline.hpp>
#include <scan_planner_msgs/msg/bspline_diagnostics.hpp>
#include <scan_planner_msgs/msg/controller_status.hpp>

namespace scan_planner
{

constexpr std::uint64_t kActiveSensingRequiredFusedObservations = 3U;
constexpr double kActiveSensingMaximumYawOffset = 0.22;
constexpr double kActiveSensingMaximumYawRate = 0.20;
constexpr double kActiveSensingMaximumSettleYawError = 0.02;
constexpr double kActiveSensingMaximumSettleAngularSpeed = 0.05;
constexpr double kActiveSensingMinimumStableDuration = 0.10;

struct ActiveSensingDiagnosticsSnapshot
{
  std::uint8_t event{
      scan_planner_msgs::msg::BsplineDiagnostics::
          ACTIVE_SENSING_EVENT_NONE};
  double start_yaw{0.0};
  double target_yaw{0.0};
  double yaw_offset{0.0};
  double yaw_rate{0.0};
  std::int64_t settle_stamp_ns{0};
  double settle_yaw_error{0.0};
  double settle_angular_speed{0.0};
  double stable_duration{0.0};
  std::uint64_t fusion_baseline{0U};
  std::uint64_t fusion_current{0U};
  std::uint64_t fusion_distinct{0U};
  bool completed{false};
  bool failed{false};
  std::string reason;
};

struct ActiveSensingReplacementResult
{
  bool failure_snapshot_published{false};
  bool stop_published{false};
};

template<typename PublishFailure, typename ResetRuntime, typename PublishStop>
inline ActiveSensingReplacementResult
terminateActiveSensingBeforeTrajectoryReplacement(
    PublishFailure publish_failure,
    ResetRuntime reset_runtime,
    PublishStop publish_stop)
{
  // 原 identity 的 FAILED 必须先进入 DDS 历史；随后清空 active runtime，
  // 最后才允许 emergency B-spline 覆盖 last_published_trajectory。
  ActiveSensingReplacementResult result;
  result.failure_snapshot_published = publish_failure();
  reset_runtime();
  result.stop_published = publish_stop();
  return result;
}

inline builtin_interfaces::msg::Time activeSensingTimeFromNanoseconds(
    const std::int64_t stamp_ns) noexcept
{
  builtin_interfaces::msg::Time stamp;
  if (stamp_ns <= 0)
    return stamp;
  constexpr std::int64_t nanoseconds_per_second = 1000000000LL;
  const std::int64_t seconds = stamp_ns / nanoseconds_per_second;
  if (seconds > std::numeric_limits<std::int32_t>::max())
    return stamp;
  stamp.sec = static_cast<std::int32_t>(seconds);
  stamp.nanosec = static_cast<std::uint32_t>(
      stamp_ns % nanoseconds_per_second);
  return stamp;
}

inline bool populateActiveSensingDiagnostics(
    scan_planner_msgs::msg::BsplineDiagnostics &diagnostics,
    const ActiveSensingDiagnosticsSnapshot &snapshot) noexcept
{
  using Diagnostics = scan_planner_msgs::msg::BsplineDiagnostics;
  const bool event_valid =
      snapshot.event >= Diagnostics::ACTIVE_SENSING_EVENT_STARTED &&
      snapshot.event <= Diagnostics::ACTIVE_SENSING_EVENT_FAILED;
  const double expected_target = std::atan2(
      std::sin(snapshot.start_yaw + snapshot.yaw_offset),
      std::cos(snapshot.start_yaw + snapshot.yaw_offset));
  const double target_error = std::abs(std::atan2(
      std::sin(snapshot.target_yaw - expected_target),
      std::cos(snapshot.target_yaw - expected_target)));
  const bool yaw_valid =
      std::isfinite(snapshot.start_yaw) &&
      std::isfinite(snapshot.target_yaw) &&
      std::isfinite(snapshot.yaw_offset) &&
      std::abs(snapshot.yaw_offset) > 1.0e-6 &&
      std::abs(snapshot.yaw_offset) <= kActiveSensingMaximumYawOffset &&
      std::isfinite(snapshot.yaw_rate) && snapshot.yaw_rate > 0.0 &&
      snapshot.yaw_rate <= kActiveSensingMaximumYawRate &&
      target_error <= 1.0e-9;
  const bool terminal_flags_valid =
      snapshot.completed != snapshot.failed ||
      (!snapshot.completed && !snapshot.failed);
  const bool terminal_event_valid =
      snapshot.completed ==
          (snapshot.event == Diagnostics::ACTIVE_SENSING_EVENT_COMPLETED) &&
      snapshot.failed ==
          (snapshot.event == Diagnostics::ACTIVE_SENSING_EVENT_FAILED);
  const bool fusion_valid =
      snapshot.fusion_current >= snapshot.fusion_baseline &&
      snapshot.fusion_distinct <=
          snapshot.fusion_current - snapshot.fusion_baseline &&
      snapshot.fusion_distinct <=
          std::numeric_limits<std::uint32_t>::max();
  const bool settle_values_finite =
      snapshot.settle_stamp_ns >= 0 &&
      std::isfinite(snapshot.settle_yaw_error) &&
      std::isfinite(snapshot.settle_angular_speed) &&
      std::isfinite(snapshot.stable_duration) &&
      snapshot.settle_yaw_error >= 0.0 &&
      snapshot.settle_angular_speed >= 0.0 &&
      snapshot.stable_duration >= 0.0;
  const bool pre_settle = snapshot.settle_stamp_ns == 0;
  const bool pre_settle_values_valid =
      !pre_settle ||
      (snapshot.settle_yaw_error == 0.0 &&
       snapshot.settle_angular_speed == 0.0 &&
       snapshot.stable_duration == 0.0 &&
       snapshot.fusion_baseline == 0U &&
       snapshot.fusion_current == 0U &&
       snapshot.fusion_distinct == 0U);
  const bool early_event =
      snapshot.event == Diagnostics::ACTIVE_SENSING_EVENT_STARTED ||
      snapshot.event ==
          Diagnostics::ACTIVE_SENSING_EVENT_CONTROLLER_ACCEPTED;
  const bool yaw_stable_event =
      snapshot.event == Diagnostics::ACTIVE_SENSING_EVENT_YAW_STABLE;
  const bool fusion_event =
      snapshot.event == Diagnostics::ACTIVE_SENSING_EVENT_FUSION_PROGRESS;
  const bool settled_event =
      yaw_stable_event || fusion_event || snapshot.completed ||
      (snapshot.failed && snapshot.settle_stamp_ns > 0);
  const bool settled_evidence_valid =
      !settled_event ||
      (snapshot.settle_stamp_ns > 0 &&
       snapshot.settle_yaw_error <=
           kActiveSensingMaximumSettleYawError &&
       snapshot.settle_angular_speed <=
           kActiveSensingMaximumSettleAngularSpeed &&
       snapshot.stable_duration + 1.0e-12 >=
           kActiveSensingMinimumStableDuration);
  const bool event_phase_valid =
      (!early_event || pre_settle) &&
      (!yaw_stable_event ||
       (snapshot.settle_stamp_ns > 0 &&
        snapshot.fusion_baseline == snapshot.fusion_current &&
        snapshot.fusion_distinct == 0U)) &&
      (!fusion_event || snapshot.settle_stamp_ns > 0);
  const bool completed_evidence_valid =
      !snapshot.completed ||
      (snapshot.settle_stamp_ns > 0 &&
       snapshot.fusion_distinct >=
           kActiveSensingRequiredFusedObservations);
  if (!event_valid || !yaw_valid || !terminal_flags_valid ||
      !terminal_event_valid || !fusion_valid || !settle_values_finite ||
      !pre_settle_values_valid || !event_phase_valid ||
      !settled_evidence_valid || !completed_evidence_valid ||
      snapshot.reason.empty())
    return false;

  diagnostics.active_sensing = true;
  diagnostics.active_sensing_event = snapshot.event;
  diagnostics.active_sensing_start_yaw = snapshot.start_yaw;
  diagnostics.active_sensing_target_yaw = snapshot.target_yaw;
  diagnostics.active_sensing_yaw_offset = snapshot.yaw_offset;
  diagnostics.active_sensing_yaw_rate = snapshot.yaw_rate;
  diagnostics.active_sensing_settle_stamp =
      activeSensingTimeFromNanoseconds(snapshot.settle_stamp_ns);
  diagnostics.active_sensing_settle_yaw_error =
      snapshot.settle_yaw_error;
  diagnostics.active_sensing_settle_angular_speed =
      snapshot.settle_angular_speed;
  diagnostics.active_sensing_stable_duration = snapshot.stable_duration;
  diagnostics.active_sensing_fusion_baseline = snapshot.fusion_baseline;
  diagnostics.active_sensing_fusion_current = snapshot.fusion_current;
  diagnostics.active_sensing_fusion_distinct =
      static_cast<std::uint32_t>(snapshot.fusion_distinct);
  diagnostics.active_sensing_fusion_required =
      static_cast<std::uint32_t>(
          kActiveSensingRequiredFusedObservations);
  diagnostics.active_sensing_completed = snapshot.completed;
  diagnostics.active_sensing_failed = snapshot.failed;
  diagnostics.active_sensing_reason = snapshot.reason;
  return true;
}

struct ActiveSensingStartContext
{
  LocalPlanFailureReason failure_reason{LocalPlanFailureReason::None};
  bool planner_attempted{false};
  bool reference_mode{false};
  std::int64_t reference_path_stamp_ns{0};
  std::int64_t consumed_reference_path_stamp_ns{0};
  bool inputs_fresh{false};
  bool stair_gate_clear{false};
  // 这里只表示尚未进入 final-hold/GOAL_REACHED 生命周期。滚动窗口已经
  // 触及最终目标（local_target_is_final_）但机器人尚未到达时仍可主动复观测。
  bool final_hold_lifecycle_clear{false};
  bool global_replan_gate_clear{false};
  bool body_rotation_envelope_free{false};
  double planar_speed{std::numeric_limits<double>::infinity()};
  double maximum_planar_speed{0.0};
};

constexpr bool activeSensingFailureIsRecoverable(
    const LocalPlanFailureReason reason) noexcept
{
  // ReferenceCorridorRejected 只表示当前有序 reference 约束拒绝了局部轨迹，
  // 不保证复观测后必然可恢复；但受阻走廊可能因遮挡或局部占据证据不足而
  // 触发，因此允许在其余安全门全部满足时消费该 Path 唯一一次复观测机会。
  return reason == LocalPlanFailureReason::TerminalPointOccupied ||
         reason ==
             LocalPlanFailureReason::CollisionSegmentAstarDegenerate ||
         reason == LocalPlanFailureReason::ReferenceCorridorRejected;
}

inline bool activeSensingMayStart(
    const ActiveSensingStartContext &context) noexcept
{
  return context.planner_attempted && context.reference_mode &&
         activeSensingFailureIsRecoverable(context.failure_reason) &&
         context.reference_path_stamp_ns > 0 &&
         context.reference_path_stamp_ns !=
             context.consumed_reference_path_stamp_ns &&
         context.inputs_fresh && context.stair_gate_clear &&
         context.final_hold_lifecycle_clear &&
         context.global_replan_gate_clear &&
         context.body_rotation_envelope_free &&
         std::isfinite(context.planar_speed) &&
         std::isfinite(context.maximum_planar_speed) &&
         context.maximum_planar_speed >= 0.0 &&
         context.planar_speed >= 0.0 &&
         context.planar_speed <= context.maximum_planar_speed;
}

struct ActiveSensingRuntimeContext
{
  bool fsm_active{false};
  std::int64_t expected_reference_path_stamp_ns{0};
  std::int64_t current_reference_path_stamp_ns{0};
  bool path_available{false};
  bool inputs_fresh{false};
  bool stair_gate_clear{false};
  // 与启动门一致：禁止打断 final-hold 生命周期，不把 final rolling
  // window 本身误判成已经到达目标。
  bool final_hold_lifecycle_clear{false};
  bool global_replan_gate_clear{false};
  bool body_rotation_envelope_free{false};
  double position_drift{std::numeric_limits<double>::infinity()};
  double maximum_position_drift{0.0};
  double planar_speed{std::numeric_limits<double>::infinity()};
  double maximum_planar_speed{0.0};
  bool latest_trajectory_identity_matches{false};
};

inline bool activeSensingMayContinue(
    const ActiveSensingRuntimeContext &context) noexcept
{
  return context.fsm_active &&
         context.expected_reference_path_stamp_ns > 0 &&
         context.current_reference_path_stamp_ns ==
             context.expected_reference_path_stamp_ns &&
         context.path_available && context.inputs_fresh &&
         context.stair_gate_clear &&
         context.final_hold_lifecycle_clear &&
         context.global_replan_gate_clear &&
         context.body_rotation_envelope_free &&
         std::isfinite(context.position_drift) &&
         std::isfinite(context.maximum_position_drift) &&
         context.position_drift >= 0.0 &&
         context.maximum_position_drift > 0.0 &&
         context.position_drift <= context.maximum_position_drift &&
         std::isfinite(context.planar_speed) &&
         std::isfinite(context.maximum_planar_speed) &&
         context.planar_speed >= 0.0 &&
         context.maximum_planar_speed >= 0.0 &&
         context.planar_speed <= context.maximum_planar_speed &&
         context.latest_trajectory_identity_matches;
}

inline std::int64_t activeSensingTimeNanoseconds(
    const builtin_interfaces::msg::Time &stamp) noexcept
{
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U)
    return 0;
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

inline bool activeSensingAcceptanceIsTimely(
    const std::int64_t callback_stamp_ns,
    const std::int64_t publish_stamp_ns,
    const std::int64_t accept_timeout_ns,
    const std::int64_t total_timeout_ns) noexcept
{
  if (callback_stamp_ns <= 0 || publish_stamp_ns <= 0 ||
      callback_stamp_ns < publish_stamp_ns || accept_timeout_ns <= 0 ||
      total_timeout_ns <= 0)
    return false;

  const std::int64_t elapsed_ns = callback_stamp_ns - publish_stamp_ns;
  // 截止点本身仍有效；任一截止点之后哪怕只迟到 1 ns 也 fail-closed。
  return elapsed_ns <= accept_timeout_ns &&
         elapsed_ns <= total_timeout_ns;
}

struct ActiveSensingTrajectoryIdentity
{
  std::int64_t reference_path_stamp_ns{0};
  std::int64_t bspline_header_stamp_ns{0};
  std::int64_t start_stamp_ns{0};
  std::int64_t trajectory_id{0};
};

inline bool sameActiveSensingTrajectoryIdentity(
    const ActiveSensingTrajectoryIdentity &lhs,
    const ActiveSensingTrajectoryIdentity &rhs) noexcept
{
  return lhs.reference_path_stamp_ns > 0 &&
         lhs.bspline_header_stamp_ns > 0 && lhs.start_stamp_ns > 0 &&
         lhs.trajectory_id > 0 &&
         lhs.reference_path_stamp_ns == rhs.reference_path_stamp_ns &&
         lhs.bspline_header_stamp_ns == rhs.bspline_header_stamp_ns &&
         lhs.start_stamp_ns == rhs.start_stamp_ns &&
         lhs.trajectory_id == rhs.trajectory_id;
}

inline ActiveSensingTrajectoryIdentity activeSensingIdentityFromBspline(
    const scan_planner_msgs::msg::Bspline &trajectory) noexcept
{
  return ActiveSensingTrajectoryIdentity{
      activeSensingTimeNanoseconds(trajectory.reference_path_stamp),
      activeSensingTimeNanoseconds(trajectory.header.stamp),
      activeSensingTimeNanoseconds(trajectory.start_time),
      trajectory.traj_id};
}

inline ActiveSensingTrajectoryIdentity activeSensingIdentityFromController(
    const scan_planner_msgs::msg::ControllerStatus &status) noexcept
{
  return ActiveSensingTrajectoryIdentity{
      activeSensingTimeNanoseconds(status.reference_path_stamp),
      activeSensingTimeNanoseconds(status.bspline_header_stamp),
      activeSensingTimeNanoseconds(status.start_time),
      status.traj_id};
}

inline ActiveSensingTrajectoryIdentity
activeSensingCandidateIdentityFromController(
    const scan_planner_msgs::msg::ControllerStatus &status) noexcept
{
  return ActiveSensingTrajectoryIdentity{
      activeSensingTimeNanoseconds(
          status.candidate_reference_path_stamp),
      activeSensingTimeNanoseconds(status.candidate_bspline_header_stamp),
      activeSensingTimeNanoseconds(status.candidate_start_time),
      status.candidate_traj_id};
}

enum class ActiveSensingControllerAction
{
  kAccepted,
  kContinue,
  kFailClosed,
};

inline ActiveSensingControllerAction decideActiveSensingControllerAction(
    const scan_planner_msgs::msg::ControllerStatus &status,
    const ActiveSensingTrajectoryIdentity &expected,
    const bool acceptance_observed) noexcept
{
  using Status = scan_planner_msgs::msg::ControllerStatus;
  if (status.event == Status::EVENT_REJECTED)
  {
    // 被拒候选使用独立 identity；无论是精确拒绝还是错代拒绝，当前主动
    // 观测都不能继续等待并假定 controller 已接管。
    return ActiveSensingControllerAction::kFailClosed;
  }

  const ActiveSensingTrajectoryIdentity observed =
      activeSensingIdentityFromController(status);
  if (!sameActiveSensingTrajectoryIdentity(expected, observed) ||
      status.is_final || status.emergency_stop ||
      !status.active_sensing_yaw_only)
    return ActiveSensingControllerAction::kFailClosed;

  if (status.event == Status::EVENT_ACCEPTED)
  {
    const bool executable_state =
        status.state == Status::STATE_ALIGNING_YAW ||
        status.state == Status::STATE_TRACKING;
    return status.accepted && status.trajectory_valid && executable_state
               ? ActiveSensingControllerAction::kAccepted
               : ActiveSensingControllerAction::kFailClosed;
  }
  if (!acceptance_observed || !status.accepted ||
      !status.trajectory_valid)
    return ActiveSensingControllerAction::kFailClosed;

  if (status.state != Status::STATE_ALIGNING_YAW &&
      status.state != Status::STATE_TRACKING)
    return ActiveSensingControllerAction::kFailClosed;
  return ActiveSensingControllerAction::kContinue;
}

inline std::optional<scan_planner_msgs::msg::Bspline>
buildActiveSensingTrajectoryMessage(
    const std::string &frame_id,
    const builtin_interfaces::msg::Time &publish_stamp,
    const builtin_interfaces::msg::Time &reference_path_stamp,
    const std::int64_t trajectory_id,
    const Eigen::Vector3d &position,
    const double current_yaw,
    const double signed_yaw_offset,
    const double maximum_yaw_rate,
    const double trajectory_duration)
{
  constexpr std::size_t control_point_count = 6U;
  constexpr int order = 3;
  if (frame_id.empty() ||
      activeSensingTimeNanoseconds(publish_stamp) <= 0 ||
      activeSensingTimeNanoseconds(reference_path_stamp) <= 0 ||
      trajectory_id <= 0 || !position.allFinite() ||
      !std::isfinite(current_yaw) ||
      !std::isfinite(signed_yaw_offset) ||
      std::abs(signed_yaw_offset) <= 1.0e-6 ||
      std::abs(signed_yaw_offset) > kActiveSensingMaximumYawOffset ||
      !std::isfinite(maximum_yaw_rate) || maximum_yaw_rate <= 0.0 ||
      maximum_yaw_rate > kActiveSensingMaximumYawRate ||
      !std::isfinite(trajectory_duration) || trajectory_duration <= 0.0)
    return std::nullopt;

  const double yaw_duration =
      std::abs(signed_yaw_offset) / maximum_yaw_rate;
  if (!std::isfinite(yaw_duration) || yaw_duration <= 0.0 ||
      trajectory_duration < yaw_duration)
    return std::nullopt;

  scan_planner_msgs::msg::Bspline trajectory;
  trajectory.header.frame_id = frame_id;
  trajectory.header.stamp = publish_stamp;
  trajectory.order = order;
  trajectory.traj_id = trajectory_id;
  trajectory.start_time = publish_stamp;
  trajectory.reference_path_stamp = reference_path_stamp;
  trajectory.is_final = false;
  trajectory.emergency_stop = false;

  geometry_msgs::msg::Point point;
  point.x = position.x();
  point.y = position.y();
  point.z = position.z();
  trajectory.pos_pts.assign(control_point_count, point);

  // 与 UniformBspline(6 个控制点, 三次) 完全相同的均匀 knot；有效时长
  // 是三个 interval。负的前置 knot 属于现有 SCAN 消息合同。
  const double interval = trajectory_duration / 3.0;
  trajectory.knots.resize(control_point_count + order + 1U);
  for (std::size_t index = 0; index < trajectory.knots.size(); ++index)
  {
    trajectory.knots[index] =
        (static_cast<double>(index) - order) * interval;
  }
  trajectory.yaw_pts = {current_yaw, current_yaw + signed_yaw_offset};
  // yaw_dt 表示完成 signed offset 的时间；由此得到的隐含速度严格不超过
  // controller 的 0.20 rad/s yaw-only 合同。
  trajectory.yaw_dt = yaw_duration;
  return trajectory;
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__ACTIVE_SENSING_H_
