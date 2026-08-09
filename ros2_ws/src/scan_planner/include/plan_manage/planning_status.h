#ifndef SCAN_PLANNER__PLANNING_STATUS_H_
#define SCAN_PLANNER__PLANNING_STATUS_H_

#include <algorithm>
#include <cstdint>

namespace scan_planner
{

// 楼梯冻结状态的 reason 是跨节点安全合同，不使用自然语言。supervisor
// 只把前两个精确 token 视为合法 ACK，其余 fault token 必须 fail-closed。
constexpr const char kScanStairExecutionInhibitedReason[] =
    "scan_stair_execution_inhibited";
constexpr const char kScanStairResumeWaitingReason[] =
    "scan_stair_resume_waiting";
constexpr const char kScanStairFreezeFrameMismatchFault[] =
    "scan_stair_freeze_frame_mismatch_fault";
constexpr const char kScanStairFreezeProtocolFault[] =
    "scan_stair_freeze_protocol_fault";
constexpr const char kScanStairFreezeSnapshotTimeoutFault[] =
    "scan_stair_freeze_snapshot_timeout_fault";
constexpr const char kScanStairStopPublishFault[] =
    "scan_stair_stop_publish_fault";

// 数值与 scan_planner_msgs/msg/ScanPlanningStatus.msg 的常量保持一致。
// 这里不依赖 ROS 2，使安全位与代际规则可以由独立 gtest 直接验证。
enum class ScanPlanningEvent : std::uint8_t
{
  kInitial = 0,
  kReferenceAccepted = 1,
  kReferenceCleared = 2,
  kTrajectoryPublished = 3,
  kPlanningFailed = 4,
  kPredictedCollision = 5,
  kEmergencyStop = 6,
  kRecovered = 7,
  kGoalHold = 8,
  kStairInhibited = 9,
  kStairResumeWaiting = 10,
};

enum class ScanPlanningState : std::uint8_t
{
  kWaitingForReference = 0,
  kPlanning = 1,
  kTracking = 2,
  kEmergencyStop = 3,
  kGoalHold = 4,
  kStairInhibited = 5,
  kUnknown = 255,
};

struct ScanPlanningStatusPolicy
{
  ScanPlanningState state{ScanPlanningState::kUnknown};
  bool stop_required{true};
  bool global_replan_recommended{false};
};

// stop/global-replan 只能由类型化事件、失败计数与锁存状态决定；reason 文本
// 只用于人类诊断，绝不能改变安全行为。
inline ScanPlanningStatusPolicy planningStatusPolicy(
    const ScanPlanningEvent event,
    const std::uint32_t consecutive_failures,
    const std::uint32_t maximum_failures,
    const bool global_replan_latched)
{
  switch (event)
  {
  case ScanPlanningEvent::kInitial:
  case ScanPlanningEvent::kReferenceCleared:
    return {ScanPlanningState::kWaitingForReference, true, false};
  case ScanPlanningEvent::kReferenceAccepted:
    return {ScanPlanningState::kPlanning, true, false};
  case ScanPlanningEvent::kTrajectoryPublished:
  case ScanPlanningEvent::kRecovered:
    return {ScanPlanningState::kTracking, false, false};
  case ScanPlanningEvent::kPlanningFailed:
  {
    const bool threshold_reached =
        maximum_failures > 0 && consecutive_failures >= maximum_failures;
    return {
        ScanPlanningState::kPlanning,
        threshold_reached,
        threshold_reached && global_replan_latched};
  }
  case ScanPlanningEvent::kPredictedCollision:
    return {
        ScanPlanningState::kEmergencyStop,
        true,
        global_replan_latched};
  case ScanPlanningEvent::kEmergencyStop:
    return {
        ScanPlanningState::kEmergencyStop,
        true,
        global_replan_latched};
  case ScanPlanningEvent::kGoalHold:
    return {ScanPlanningState::kGoalHold, true, false};
  case ScanPlanningEvent::kStairInhibited:
  case ScanPlanningEvent::kStairResumeWaiting:
    return {ScanPlanningState::kStairInhibited, true, false};
  }
  return {ScanPlanningState::kUnknown, true, global_replan_latched};
}

// SCAN 请求全局重规划后，旧 Path 的自动恢复必须被锁死。只有严格更大的
// Path stamp（空 tombstone 或非空新代际）才有资格解除该门。
class GlobalReplanGenerationGate
{
public:
  void require(const std::int64_t reference_path_stamp_ns)
  {
    required_ = true;
    blocked_reference_path_stamp_ns_ = std::max(
        blocked_reference_path_stamp_ns_, reference_path_stamp_ns);
  }

  bool required() const { return required_; }

  std::int64_t blockedReferencePathStampNs() const
  {
    return blocked_reference_path_stamp_ns_;
  }

  bool isStrictReplacement(const std::int64_t candidate_stamp_ns) const
  {
    return required_ && candidate_stamp_ns > blocked_reference_path_stamp_ns_;
  }

  bool clearForStrictReplacement(const std::int64_t candidate_stamp_ns)
  {
    if (!isStrictReplacement(candidate_stamp_ns))
      return false;
    required_ = false;
    blocked_reference_path_stamp_ns_ = 0;
    return true;
  }

  void reset()
  {
    required_ = false;
    blocked_reference_path_stamp_ns_ = 0;
  }

private:
  bool required_{false};
  std::int64_t blocked_reference_path_stamp_ns_{0};
};

}  // namespace scan_planner

#endif  // SCAN_PLANNER__PLANNING_STATUS_H_
