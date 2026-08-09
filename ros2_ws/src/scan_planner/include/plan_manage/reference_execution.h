#ifndef SCAN_PLANNER__REFERENCE_EXECUTION_H_
#define SCAN_PLANNER__REFERENCE_EXECUTION_H_

#include <Eigen/Core>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace scan_planner
{

enum class ReferenceExecutionAction
{
  kHold,
  kReplan,
  kFinish,
};

enum class ReferenceEmergencyRecoveryAction
{
  kWaitForNewTarget,
  kRetryActiveReference,
};

enum class FinalHoldControllerAction
{
  kIgnore,
  kConfirmGoal,
  kReplan,
};

enum class FinalHoldControllerState
{
  kOther,
  kGoalReached,
  kTrajectoryTimeout,
};

struct ReferenceGoalHoldDwellState
{
  std::int64_t path_stamp_ns{0};
  std::int64_t stable_since_ns{0};
  std::int64_t last_sample_stamp_ns{0};
  double stable_duration_sec{0.0};
};

enum class FinalHoldStairResumeAction
{
  kResumeNormally,
  kWaitForController,
  kContinueTimeoutRecovery,
};

struct FinalHoldTrajectoryIdentity
{
  int64_t reference_path_stamp_ns{0};
  int64_t bspline_header_stamp_ns{0};
  int64_t start_stamp_ns{0};
  int64_t trajectory_id{0};
};

inline constexpr std::size_t kPublishedFinalIdentityHistoryDepth = 64;

struct FinalHoldLifecycleState
{
  bool pending{false};
  bool recovery_required{false};
  int64_t recovery_after_trajectory_id{0};
};

inline bool sameFinalHoldIdentity(
    const FinalHoldTrajectoryIdentity &lhs,
    const FinalHoldTrajectoryIdentity &rhs)
{
  return
    lhs.reference_path_stamp_ns > 0 &&
    lhs.bspline_header_stamp_ns > 0 &&
    lhs.start_stamp_ns > 0 &&
    lhs.trajectory_id > 0 &&
    lhs.reference_path_stamp_ns == rhs.reference_path_stamp_ns &&
    lhs.bspline_header_stamp_ns == rhs.bspline_header_stamp_ns &&
    lhs.start_stamp_ns == rhs.start_stamp_ns &&
    lhs.trajectory_id == rhs.trajectory_id;
}

// planner 的 timer 可能正忙于局部优化，ControllerStatus 回调会晚于下一条
// B-spline 发布。保留一个与 status QoS 深度相同的有界 final 身份历史，既能
// 认证迟到的 ACCEPTED/GOAL_REACHED，又不会把其他 Path 代际的裸状态当真。
inline void rememberPublishedFinalIdentity(
    std::vector<FinalHoldTrajectoryIdentity> &history,
    const FinalHoldTrajectoryIdentity &identity,
    const std::size_t maximum_depth = kPublishedFinalIdentityHistoryDepth)
{
  if (!sameFinalHoldIdentity(identity, identity) || maximum_depth == 0)
    return;
  const auto duplicate = std::find_if(
      history.begin(), history.end(),
      [&identity](const FinalHoldTrajectoryIdentity &candidate) {
        return sameFinalHoldIdentity(candidate, identity);
      });
  if (duplicate != history.end())
    return;
  if (history.size() >= maximum_depth)
  {
    const std::size_t erase_count =
        history.size() - maximum_depth + 1;
    history.erase(
        history.begin(),
        history.begin() + static_cast<std::ptrdiff_t>(erase_count));
  }
  history.push_back(identity);
}

inline bool wasFinalIdentityPublished(
    const std::vector<FinalHoldTrajectoryIdentity> &history,
    const FinalHoldTrajectoryIdentity &observed)
{
  return std::any_of(
      history.begin(), history.end(),
      [&observed](const FinalHoldTrajectoryIdentity &candidate) {
        return sameFinalHoldIdentity(candidate, observed);
      });
}

// 只有 controller 的当前执行 identity 在 planner 本 Path 代际的有界发布
// 历史中得到精确认证时才更新接受缓存。goal latch 拒绝新 candidate 时，
// status 保留旧执行 identity，因此必须保留旧 final 缓存。
inline void updateControllerAcceptedFinalIdentity(
    const bool observed_was_published,
    const FinalHoldTrajectoryIdentity &observed,
    const bool accepted,
    const bool trajectory_valid,
    const bool is_final,
    const bool emergency_stop,
    FinalHoldTrajectoryIdentity &cached,
    bool &have_cached)
{
  if (!observed_was_published ||
      !accepted || !trajectory_valid)
    return;
  if (is_final && !emergency_stop)
  {
    cached = observed;
    have_cached = true;
    return;
  }
  cached = FinalHoldTrajectoryIdentity{};
  have_cached = false;
}

// planner 只能消费与 controller 已接受 final 轨迹精确同
// identity 的终态。moving final 可直接确认 GOAL_REACHED；只有
// stationary final hold 的 TIMEOUT 可触发 hold 恢复。裸 Bool、候选
// 拒绝状态或其他代际都不能清除目标。
inline FinalHoldControllerAction decideFinalHoldControllerAction(
    const bool final_hold_pending,
    const bool have_target,
    const FinalHoldTrajectoryIdentity &expected,
    const FinalHoldTrajectoryIdentity &observed,
    const bool accepted,
    const bool trajectory_valid,
    const bool is_final,
    const bool emergency_stop,
    const FinalHoldControllerState controller_state)
{
  if (
    !have_target || !accepted ||
    !trajectory_valid || !is_final || emergency_stop ||
    !sameFinalHoldIdentity(expected, observed))
  {
    return FinalHoldControllerAction::kIgnore;
  }
  if (controller_state == FinalHoldControllerState::kGoalReached)
    return FinalHoldControllerAction::kConfirmGoal;
  if (final_hold_pending &&
      controller_state == FinalHoldControllerState::kTrajectoryTimeout)
    return FinalHoldControllerAction::kReplan;
  return FinalHoldControllerAction::kIgnore;
}

// Controller 终态只改变 final-hold 生命周期，不直接改写 Path 或 target。
// TIMEOUT 必须结束 ACK 等待并锁存旧 traj_id，供下一条正常轨迹执行严格递增门。
inline FinalHoldLifecycleState transitionFinalHoldLifecycle(
    const FinalHoldLifecycleState &current,
    const FinalHoldControllerAction action,
    const int64_t current_trajectory_id)
{
  if (action == FinalHoldControllerAction::kConfirmGoal)
    return FinalHoldLifecycleState{};
  if (action == FinalHoldControllerAction::kReplan &&
      current.pending && current_trajectory_id > 0)
  {
    return FinalHoldLifecycleState{
        false, true, current_trajectory_id};
  }
  return current;
}

// 楼梯 fresh-input barrier 只负责恢复规划时钟。若 stationary hold 尚在等待
// controller 终态，解冻不能擅自替换 identity；只有明确 TIMEOUT 才继续恢复。
inline FinalHoldStairResumeAction decideFinalHoldStairResumeAction(
    const FinalHoldLifecycleState &state)
{
  if (state.pending)
    return FinalHoldStairResumeAction::kWaitForController;
  if (state.recovery_required)
    return FinalHoldStairResumeAction::kContinueTimeoutRecovery;
  return FinalHoldStairResumeAction::kResumeNormally;
}

inline bool finalHoldBlocksStationaryPublication(
    const FinalHoldLifecycleState &state)
{
  return state.pending || state.recovery_required;
}

// TIMEOUT 恢复轨迹必须严格晚于失效 hold；非恢复阶段不施加额外 ID 门。
inline bool finalHoldRecoveryTrajectoryIdAllowed(
    const FinalHoldLifecycleState &state,
    const int64_t candidate_trajectory_id)
{
  if (!state.recovery_required)
    return candidate_trajectory_id > 0;
  return state.recovery_after_trajectory_id > 0 &&
         candidate_trajectory_id > state.recovery_after_trajectory_id;
}

struct ReferencePathProjection
{
  bool valid{false};
  double progress_s{0.0};
  double distance{std::numeric_limits<double>::infinity()};
  std::size_t segment_index{0};
};

struct ReferenceTargetSelection
{
  bool valid{false};
  double progress_s{0.0};
  int search_direction{0};
};

enum class StairPlanningFreezeUpdate
{
  kDuplicate,
  kInhibited,
  kInitialUnfrozen,
  kResumeBarrierStarted,
  kReleasedWithoutReference,
  kProtocolRejected,
};

enum class StairFreezeFreshnessUpdate
{
  kNoChange,
  kTimeoutCandidateStarted,
  kTimeoutConfirmed,
};

struct StairExecutionFreezeSnapshot
{
  bool frozen{true};
  int64_t header_stamp_ns{0};
  int64_t reference_path_stamp_ns{0};
  std::string writer_id;
  std::string writer_epoch;
  std::uint64_t sequence{0};
};

struct StairExecutionFreezeMailboxMessage
{
  std::string frame_id;
  StairExecutionFreezeSnapshot snapshot;
};

struct StairExecutionFreezeMailboxDrain
{
  bool available{false};
  StairExecutionFreezeMailboxMessage message;
  std::size_t coalesced_count{0};
};

// DDS 回调只写入该有界 mailbox；所有 Path、地图和 gate 状态仍由 FSM
// callback group 串行处理，避免为解决订阅饥饿而引入规划数据竞争。
class StairExecutionFreezeMailbox
{
public:
  void push(StairExecutionFreezeMailboxMessage message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_latest_)
      ++coalesced_count_;
    // Topic 本身是 KeepLast(1) 的持续状态流；mailbox 同样只保存最新快照，
    // 避免主 FSM 忙于规划后先消费已经过期的旧 Header。
    latest_ = std::move(message);
    has_latest_ = true;
  }

  StairExecutionFreezeMailboxDrain drain()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    StairExecutionFreezeMailboxDrain result;
    result.available = has_latest_;
    if (has_latest_)
    {
      result.message = std::move(latest_);
      has_latest_ = false;
    }
    result.coalesced_count = coalesced_count_;
    coalesced_count_ = 0;
    return result;
  }

private:
  std::mutex mutex_;
  StairExecutionFreezeMailboxMessage latest_;
  bool has_latest_{false};
  std::size_t coalesced_count_{0};
};

struct ReferencePathBinding
{
  int64_t stamp_ns{0};
  bool available{false};
  bool pending{false};
};

// final hold 等待 controller 精确 ACK 时仍保留 target。pending 只描述
// ACK 等待，TIMEOUT recovery 由 have_target 继续绑定同代 Path；不能把任意
// 无目标状态伪装成有效代际。
inline bool activeReferenceAvailableForFreezeBinding(
    const bool has_active_reference_points,
    const bool have_target,
    const bool final_hold_pending)
{
  return has_active_reference_points &&
         (have_target || final_hold_pending);
}

// 已通过几何与 frame 校验但仍在等待首帧地图的 pending Path，已经是当前
// 最新代际。冻结协议必须优先绑定它，不能继续接受旧 active Path 的快照。
inline ReferencePathBinding resolveReferencePathBinding(
    const int64_t pending_stamp_ns,
    const bool pending_available,
    const int64_t active_stamp_ns,
    const bool active_available)
{
  if (pending_available && pending_stamp_ns > 0)
    return ReferencePathBinding{pending_stamp_ns, true, true};
  if (active_available && active_stamp_ns > 0)
    return ReferencePathBinding{active_stamp_ns, true, false};
  return ReferencePathBinding{};
}

// controller 的姿态对齐暂停与楼梯执行暂停都会冻结局部轨迹时钟，但只有
// 楼梯暂停及其恢复屏障能够抑制优化器和 FSM。
inline bool trajectoryTimeFrozen(
    const bool controller_execution_frozen,
    const bool stair_execution_frozen)
{
  return controller_execution_frozen || stair_execution_frozen;
}

// 楼梯 root-lock 快照必须绑定 active 或已校验 pending Path、固定 writer
// epoch、严格递增序列和新鲜 Header。任一身份或新鲜度错误都保持
// fail-closed；合法 false 的下降沿还必须等待 Odometry 与占据地图严格更新后
// 才允许同代 Path 重规划。
class StairPlanningFreezeGate
{
public:
  void bindReferenceGeneration(const int64_t reference_path_stamp_ns)
  {
    if (reference_path_stamp_ns <= 0)
    {
      clearReferenceGeneration();
      return;
    }
    if (active_reference_path_stamp_ns_ == reference_path_stamp_ns)
      return;
    active_reference_path_stamp_ns_ = reference_path_stamp_ns;
    protocol_valid_ = false;
    frozen_ = true;
    protocol_fault_ = "awaiting_exact_path_freeze_snapshot";
    authenticated_snapshot_seen_for_generation_ = false;
    clearResumeBarrier();
    clearTimeoutCandidate();
  }

  void clearReferenceGeneration()
  {
    active_reference_path_stamp_ns_ = 0;
    protocol_valid_ = false;
    frozen_ = false;
    protocol_fault_.clear();
    authenticated_snapshot_seen_for_generation_ = false;
    clearResumeBarrier();
    clearTimeoutCandidate();
  }

  StairPlanningFreezeUpdate updateTyped(
      const StairExecutionFreezeSnapshot &snapshot,
      const int64_t odometry_stamp_ns,
      const int64_t observation_stamp_ns,
      const int64_t current_reference_path_stamp_ns,
      const bool has_bound_reference,
      const int64_t now_ns,
      const int64_t freshness_timeout_ns)
  {
    const bool identity_valid =
        has_bound_reference && current_reference_path_stamp_ns > 0 &&
        active_reference_path_stamp_ns_ == current_reference_path_stamp_ns &&
        snapshot.reference_path_stamp_ns == current_reference_path_stamp_ns;
    const bool timestamp_valid =
        now_ns > 0 && freshness_timeout_ns > 0 &&
        snapshot.header_stamp_ns > 0 && snapshot.header_stamp_ns <= now_ns &&
        now_ns - snapshot.header_stamp_ns <= freshness_timeout_ns &&
        (last_header_stamp_ns_ == 0 ||
         snapshot.header_stamp_ns >= last_header_stamp_ns_);
    const auto has_non_space = [](const std::string &value) {
      return std::any_of(
          value.begin(), value.end(),
          [](const unsigned char character) {
            return !std::isspace(character);
          });
    };
    const bool writer_fields_valid =
        has_non_space(snapshot.writer_id) &&
        has_non_space(snapshot.writer_epoch);
    const bool writer_matches =
        writer_id_.empty() ||
        (snapshot.writer_id == writer_id_ &&
         snapshot.writer_epoch == writer_epoch_);
    const bool sequence_valid =
        snapshot.sequence > 0 && snapshot.sequence > last_sequence_;
    if (!identity_valid || !timestamp_valid || !writer_fields_valid ||
        !writer_matches || !sequence_valid)
    {
      if (!identity_valid)
        failClosed("reference_path_identity_mismatch");
      else if (!timestamp_valid)
        failClosed("stale_or_invalid_header_stamp");
      else if (!writer_fields_valid)
        failClosed("empty_writer_identity");
      else if (!writer_matches)
        failClosed("second_writer_rejected");
      else
        failClosed("non_monotonic_sequence");
      return StairPlanningFreezeUpdate::kProtocolRejected;
    }

    if (writer_id_.empty())
    {
      writer_id_ = snapshot.writer_id;
      writer_epoch_ = snapshot.writer_epoch;
    }
    const bool was_protocol_valid = protocol_valid_;
    const bool was_frozen = frozen_;
    const bool had_authenticated_snapshot =
        authenticated_snapshot_seen_for_generation_;
    last_sequence_ = snapshot.sequence;
    last_header_stamp_ns_ = snapshot.header_stamp_ns;
    protocol_valid_ = true;
    authenticated_snapshot_seen_for_generation_ = true;
    protocol_fault_.clear();
    frozen_ = snapshot.frozen;
    clearTimeoutCandidate();

    if (frozen_)
    {
      clearResumeBarrier();
      return (!was_protocol_valid || !was_frozen) ?
          StairPlanningFreezeUpdate::kInhibited :
          StairPlanningFreezeUpdate::kDuplicate;
    }

    if (!has_bound_reference || current_reference_path_stamp_ns <= 0)
    {
      clearResumeBarrier();
      return StairPlanningFreezeUpdate::kReleasedWithoutReference;
    }

    // 新 Path 的第一条精确绑定快照若明确为 false，说明这一代 Path 从未
    // 进入楼梯冻结；此时没有旧冻结状态需要用 fresh-input barrier 清洗。
    // 若此前见过任何合法快照（包括 true），或协议曾在运行中失效，后续
    // true→false / fault→false 仍走严格恢复屏障。
    if (!had_authenticated_snapshot)
    {
      clearResumeBarrier();
      return StairPlanningFreezeUpdate::kInitialUnfrozen;
    }

    if (was_protocol_valid && !was_frozen)
      return StairPlanningFreezeUpdate::kDuplicate;

    resume_waiting_ = true;
    odometry_baseline_ns_ = odometry_stamp_ns;
    observation_baseline_ns_ = observation_stamp_ns;
    reference_path_stamp_ns_ = current_reference_path_stamp_ns;
    return StairPlanningFreezeUpdate::kResumeBarrierStarted;
  }

  StairFreezeFreshnessUpdate refreshFreshness(
      const int64_t now_ns,
      const int64_t freshness_timeout_ns,
      const int64_t confirmation_grace_ns)
  {
    if (timeout_candidate_active_)
    {
      if (timeout_fault_confirmed_ || now_ns <= 0 ||
          confirmation_grace_ns <= 0 ||
          now_ns <= timeout_candidate_first_observed_now_ns_ ||
          last_sequence_ != timeout_candidate_sequence_ ||
          last_header_stamp_ns_ != timeout_candidate_header_stamp_ns_ ||
          now_ns - timeout_candidate_first_observed_now_ns_ <=
              confirmation_grace_ns)
        return StairFreezeFreshnessUpdate::kNoChange;

      timeout_fault_confirmed_ = true;
      failClosed("stair_freeze_snapshot_timeout");
      return StairFreezeFreshnessUpdate::kTimeoutConfirmed;
    }

    if (active_reference_path_stamp_ns_ <= 0 || !protocol_valid_ ||
        now_ns <= 0 || freshness_timeout_ns <= 0 ||
        confirmation_grace_ns <= 0 || last_header_stamp_ns_ <= 0 ||
        now_ns < last_header_stamp_ns_ ||
        now_ns - last_header_stamp_ns_ <= freshness_timeout_ns)
      return StairFreezeFreshnessUpdate::kNoChange;

    // 首次超时只建立候选并立即暂停规划。调度器可能先消费新的 /clock、
    // 后消费同一批 DDS 心跳；宽限期内若收到更高合法序列，updateTyped
    // 必须透明清除候选并保留上一条快照的逻辑状态，不能把新鲜 false
    // 误判成一次真实解冻而启动恢复屏障。只有确认超时才破坏协议状态。
    timeout_candidate_active_ = true;
    timeout_fault_confirmed_ = false;
    timeout_candidate_sequence_ = last_sequence_;
    timeout_candidate_header_stamp_ns_ = last_header_stamp_ns_;
    timeout_candidate_first_observed_now_ns_ = now_ns;
    return StairFreezeFreshnessUpdate::kTimeoutCandidateStarted;
  }

  void failClosed(const std::string &reason)
  {
    protocol_valid_ = false;
    protocol_fault_ = reason.empty() ? "invalid_stair_freeze_protocol" : reason;
    clearResumeBarrier();
    // 没有任何已绑定 Path 时不能凭一条非法快照制造虚假的 active freeze，
    // 但仍必须保留明确 fault，避免拒绝日志退化为空字符串。
    frozen_ = active_reference_path_stamp_ns_ > 0;
  }

  bool frozen() const { return frozen_ || timeout_candidate_active_; }

  bool resumeWaiting() const { return resume_waiting_; }

  bool planningInhibited() const { return frozen() || resume_waiting_; }

  bool protocolValid() const { return protocol_valid_; }

  bool authenticatedSnapshotAvailableForStatus() const
  {
    return protocol_valid_ && !timeout_candidate_active_;
  }

  const std::string &protocolFault() const { return protocol_fault_; }

  const std::string &writerId() const { return writer_id_; }

  const std::string &writerEpoch() const { return writer_epoch_; }

  std::uint64_t lastSequence() const { return last_sequence_; }

  int64_t lastHeaderStampNs() const { return last_header_stamp_ns_; }

  bool timeoutCandidateActive() const { return timeout_candidate_active_; }

  bool timeoutFaultConfirmed() const { return timeout_fault_confirmed_; }

  std::uint64_t timeoutCandidateSequence() const
  {
    return timeout_candidate_sequence_;
  }

  int64_t timeoutCandidateHeaderStampNs() const
  {
    return timeout_candidate_header_stamp_ns_;
  }

  int64_t timeoutCandidateFirstObservedNowNs() const
  {
    return timeout_candidate_first_observed_now_ns_;
  }

  int64_t activeReferencePathStampNs() const
  {
    return active_reference_path_stamp_ns_;
  }

  int64_t odometryBaselineNs() const { return odometry_baseline_ns_; }

  int64_t observationBaselineNs() const { return observation_baseline_ns_; }

  int64_t referencePathStampNs() const { return reference_path_stamp_ns_; }

  bool resumeInputsReady(
      const int64_t odometry_stamp_ns,
      const int64_t observation_stamp_ns,
      const int64_t reference_path_stamp_ns,
      const bool inputs_ready,
      const bool has_active_reference) const
  {
    return resume_waiting_ && !frozen() && inputs_ready &&
           has_active_reference && reference_path_stamp_ns_ > 0 &&
           reference_path_stamp_ns == reference_path_stamp_ns_ &&
           odometry_stamp_ns > odometry_baseline_ns_ &&
           observation_stamp_ns > observation_baseline_ns_;
  }

  void clearResumeBarrier()
  {
    resume_waiting_ = false;
    odometry_baseline_ns_ = 0;
    observation_baseline_ns_ = 0;
    reference_path_stamp_ns_ = 0;
  }

  void clearTimeoutCandidate()
  {
    timeout_candidate_active_ = false;
    timeout_fault_confirmed_ = false;
    timeout_candidate_sequence_ = 0;
    timeout_candidate_header_stamp_ns_ = 0;
    timeout_candidate_first_observed_now_ns_ = 0;
  }

private:
  bool frozen_{false};
  bool protocol_valid_{false};
  bool authenticated_snapshot_seen_for_generation_{false};
  bool resume_waiting_{false};
  std::string protocol_fault_;
  std::string writer_id_;
  std::string writer_epoch_;
  std::uint64_t last_sequence_{0};
  int64_t last_header_stamp_ns_{0};
  int64_t active_reference_path_stamp_ns_{0};
  int64_t odometry_baseline_ns_{0};
  int64_t observation_baseline_ns_{0};
  int64_t reference_path_stamp_ns_{0};
  bool timeout_candidate_active_{false};
  bool timeout_fault_confirmed_{false};
  std::uint64_t timeout_candidate_sequence_{0};
  int64_t timeout_candidate_header_stamp_ns_{0};
  int64_t timeout_candidate_first_observed_now_ns_{0};
};

// PCT Path 是带顺序的三维折线。SCAN 在 reference 模式下只沿该折线选择
// 滚动局部目标，不再为整条跨楼层路线构造高阶全局多项式。
inline std::vector<double> buildReferencePathArcLengths(
    const std::vector<Eigen::Vector3d> &points,
    const double minimum_segment_length = 1.0e-9)
{
  if (points.size() < 2 || !std::isfinite(minimum_segment_length) ||
      minimum_segment_length < 0.0)
    return {};

  std::vector<double> arc_lengths(points.size(), 0.0);
  for (std::size_t index = 1; index < points.size(); ++index)
  {
    const double segment_length = (points[index] - points[index - 1]).norm();
    if (!std::isfinite(segment_length) ||
        segment_length <= minimum_segment_length)
      return {};
    arc_lengths[index] = arc_lengths[index - 1] + segment_length;
  }
  return arc_lengths;
}

inline bool sampleReferencePathAtArcLength(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<double> &arc_lengths,
    const double requested_s,
    Eigen::Vector3d &sample)
{
  if (points.size() < 2 || arc_lengths.size() != points.size() ||
      !std::isfinite(requested_s) || arc_lengths.back() <= 0.0)
    return false;

  const double sample_s = std::max(0.0, std::min(requested_s, arc_lengths.back()));
  const auto upper = std::upper_bound(
      arc_lengths.begin(), arc_lengths.end(), sample_s);
  if (upper == arc_lengths.begin())
  {
    sample = points.front();
    return true;
  }
  if (upper == arc_lengths.end())
  {
    sample = points.back();
    return true;
  }

  const std::size_t next_index =
      static_cast<std::size_t>(std::distance(arc_lengths.begin(), upper));
  const std::size_t previous_index = next_index - 1;
  const double segment_length =
      arc_lengths[next_index] - arc_lengths[previous_index];
  if (segment_length <= 1.0e-12)
    return false;
  const double ratio =
      (sample_s - arc_lengths[previous_index]) / segment_length;
  sample = points[previous_index] +
           ratio * (points[next_index] - points[previous_index]);
  return sample.allFinite();
}

// 三次 B-spline 参数化不会严格插值末端采样点。局部目标不能只检查一个
// 自由体素，还必须保证其后向支撑区连续自由；否则目标点虽在障碍边界外，
// 实际样条末端仍可能回落到占据体素。先遍历完整前向窗口，保留 SCAN
// 绕过障碍的机会；前方确实没有连续自由区时才退回障碍前安全候选。
inline ReferenceTargetSelection selectReferenceTargetWithFreeRunway(
    const double nominal_progress_s,
    const double minimum_progress_s,
    const double total_progress_s,
    const double maximum_forward_extension_s,
    const double candidate_step_s,
    const double runway_sample_step_s,
    const double free_runway_s,
    const std::function<bool(double)> &is_free_at_progress)
{
  ReferenceTargetSelection selection;
  if (!std::isfinite(nominal_progress_s) ||
      !std::isfinite(minimum_progress_s) ||
      !std::isfinite(total_progress_s) ||
      !std::isfinite(maximum_forward_extension_s) ||
      !std::isfinite(candidate_step_s) ||
      !std::isfinite(runway_sample_step_s) ||
      !std::isfinite(free_runway_s) ||
      minimum_progress_s < 0.0 ||
      total_progress_s < minimum_progress_s ||
      maximum_forward_extension_s < 0.0 ||
      candidate_step_s <= 0.0 || runway_sample_step_s <= 0.0 ||
      free_runway_s < 0.0 ||
      !is_free_at_progress)
    return selection;

  const double nominal = std::max(
      minimum_progress_s,
      std::min(nominal_progress_s, total_progress_s));
  const double maximum = std::min(
      total_progress_s, nominal + maximum_forward_extension_s);
  const int candidate_count = std::max(
      0, static_cast<int>(std::ceil(
             maximum_forward_extension_s / candidate_step_s)));

  auto runway_is_free = [&](const double candidate_s) {
    const double runway_start = std::max(
        minimum_progress_s, candidate_s - free_runway_s);
    if (!is_free_at_progress(candidate_s))
      return false;
    for (double sample_s = candidate_s - runway_sample_step_s;
         sample_s > runway_start + 1.0e-9;
         sample_s -= runway_sample_step_s)
    {
      if (!is_free_at_progress(sample_s))
        return false;
    }
    return is_free_at_progress(runway_start);
  };

  if (runway_is_free(nominal))
  {
    selection.valid = true;
    selection.progress_s = nominal;
    return selection;
  }
  for (int index = 1; index <= candidate_count; ++index)
  {
    const double delta_s = index * candidate_step_s;
    const double forward_s = nominal + delta_s;
    if (forward_s <= maximum + 1.0e-9 && runway_is_free(forward_s))
    {
      selection.valid = true;
      selection.progress_s = std::min(forward_s, maximum);
      selection.search_direction = 1;
      return selection;
    }
  }
  for (int index = 1; index <= candidate_count; ++index)
  {
    const double delta_s = index * candidate_step_s;
    const double backward_s = nominal - delta_s;
    if (backward_s >= minimum_progress_s - 1.0e-9 &&
        runway_is_free(backward_s))
    {
      selection.valid = true;
      selection.progress_s = std::max(backward_s, minimum_progress_s);
      selection.search_direction = -1;
      return selection;
    }
  }
  return selection;
}

// 完整 Path 的精确终点可能位于任务允许误差圈内的静态包络边缘，例如
// 机械臂作业站位前方的桌沿。此时不能让 final B-spline 继续指向占据点，
// 也不能把普通后退候选当作非 final 巡航点反复逼近。仅在终点已经进入
// 当前局部窗口时，沿有序 Path 从当前进度向前寻找第一个同时满足：
// 1) 位于严格终点保持门内；2) 终端 yaw 下自由；3) 后向样条支撑区连续
// 自由的捕获点。返回的进度仍属于原 Path，不改变全局终点身份。
inline ReferenceTargetSelection selectReferenceTerminalCaptureTarget(
    const double minimum_progress_s,
    const double total_progress_s,
    const double candidate_step_s,
    const double runway_sample_step_s,
    const double free_runway_s,
    const std::function<bool(double)> &is_capture_candidate,
    const std::function<bool(double)> &is_free_at_progress)
{
  ReferenceTargetSelection selection;
  if (!std::isfinite(minimum_progress_s) ||
      !std::isfinite(total_progress_s) ||
      !std::isfinite(candidate_step_s) ||
      !std::isfinite(runway_sample_step_s) ||
      !std::isfinite(free_runway_s) ||
      minimum_progress_s < 0.0 ||
      total_progress_s < minimum_progress_s ||
      candidate_step_s <= 0.0 || runway_sample_step_s <= 0.0 ||
      free_runway_s < 0.0 || !is_capture_candidate ||
      !is_free_at_progress)
    return selection;

  auto runway_is_free = [&](const double candidate_s) {
    const double runway_start = std::max(
        minimum_progress_s, candidate_s - free_runway_s);
    if (!is_free_at_progress(candidate_s))
      return false;
    for (double sample_s = candidate_s - runway_sample_step_s;
         sample_s > runway_start + 1.0e-9;
         sample_s -= runway_sample_step_s)
    {
      if (!is_free_at_progress(sample_s))
        return false;
    }
    return is_free_at_progress(runway_start);
  };

  const double span = total_progress_s - minimum_progress_s;
  const int candidate_count = std::max(
      0, static_cast<int>(std::ceil(span / candidate_step_s)));
  for (int index = 0; index <= candidate_count; ++index)
  {
    const double candidate_s = std::min(
        total_progress_s,
        minimum_progress_s + index * candidate_step_s);
    if (is_capture_candidate(candidate_s) && runway_is_free(candidate_s))
    {
      selection.valid = true;
      selection.progress_s = candidate_s;
      selection.search_direction = candidate_s + 1.0e-9 < total_progress_s ?
          -1 : 0;
      return selection;
    }
  }
  return selection;
}

// 终点捕获点可以早于精确 Path 终点进入当前滚动窗口。例如窗口末端距离
// 精确终点仍有 5 cm，但已经落入 6 cm 的严格完成门。此时应把本条轨迹
// 标成 final 并以零终端速度制动；但绝不能为了提前收口选择超过本次
// nominal 前视进度的捕获点，否则会跨过局部规划视距。
inline bool terminalCaptureTargetFitsLocalWindow(
    const ReferenceTargetSelection &selection,
    const double minimum_progress_s,
    const double nominal_target_s)
{
  return selection.valid &&
         std::isfinite(selection.progress_s) &&
         std::isfinite(minimum_progress_s) &&
         std::isfinite(nominal_target_s) &&
         minimum_progress_s >= 0.0 &&
         nominal_target_s >= minimum_progress_s &&
         selection.progress_s >= minimum_progress_s - 1.0e-9 &&
         selection.progress_s <= nominal_target_s + 1.0e-9;
}

// 普通自由支撑区搜索与终点捕获搜索可能使用不同的采样相位。若普通搜索
// 已经选中了当前窗口内的严格终点候选，不能因为另一组离散采样点恰好
// 跨过它而继续发布带巡航末速的非 final 轨迹。该门只复用既有候选判定，
// 不放宽终点距离、终端朝向碰撞或局部前视范围。
inline bool selectedReferenceTargetQualifiesAsTerminalCapture(
    const ReferenceTargetSelection &selection,
    const double minimum_progress_s,
    const double nominal_target_s,
    const std::function<bool(double)> &is_capture_candidate)
{
  return is_capture_candidate &&
         terminalCaptureTargetFitsLocalWindow(
             selection, minimum_progress_s, nominal_target_s) &&
         is_capture_candidate(selection.progress_s);
}

// 普通自由支撑区搜索可能在精确终点被占据时向后退到严格位置门内，但该点
// 在 terminal yaw 下仍不满足 final 碰撞门。此时不能谎报 final，也不能继续
// 携带巡航末速度穿过目标；应把本条 non-final 轨迹的末速度降为零，等待真实
// Odometry 稳定后由 stationary final hold 完成认证，或由后续规划安全恢复。
inline bool nonFinalReferenceTargetRequiresTerminalBrake(
    const bool local_target_is_final,
    const double distance_xy,
    const double distance_z,
    const double hold_distance_xy,
    const double hold_distance_z)
{
  return !local_target_is_final &&
         std::isfinite(distance_xy) && std::isfinite(distance_z) &&
         std::isfinite(hold_distance_xy) &&
         std::isfinite(hold_distance_z) &&
         distance_xy >= 0.0 && distance_z >= 0.0 &&
         hold_distance_xy > 0.0 && hold_distance_z > 0.0 &&
         distance_xy <= hold_distance_xy + 1.0e-9 &&
         distance_z <= hold_distance_z + 1.0e-9;
}

inline ReferencePathProjection projectReferencePathProgress(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<double> &arc_lengths,
    const Eigen::Vector3d &query,
    const double minimum_progress_s,
    const double maximum_progress_s)
{
  ReferencePathProjection result;
  if (points.size() < 2 || arc_lengths.size() != points.size() ||
      !query.allFinite() || !std::isfinite(minimum_progress_s) ||
      !std::isfinite(maximum_progress_s) || arc_lengths.back() <= 0.0)
    return result;

  const double lower =
      std::max(0.0, std::min(minimum_progress_s, arc_lengths.back()));
  const double upper = std::max(
      lower, std::min(maximum_progress_s, arc_lengths.back()));
  constexpr double tie_tolerance = 1.0e-12;

  for (std::size_t index = 0; index + 1 < points.size(); ++index)
  {
    const double segment_start_s = arc_lengths[index];
    const double segment_end_s = arc_lengths[index + 1];
    if (segment_end_s < lower || segment_start_s > upper)
      continue;

    const double segment_length = segment_end_s - segment_start_s;
    if (segment_length <= 1.0e-12)
      continue;
    const Eigen::Vector3d delta = points[index + 1] - points[index];
    const double minimum_ratio =
        std::max(0.0, (lower - segment_start_s) / segment_length);
    const double maximum_ratio =
        std::min(1.0, (upper - segment_start_s) / segment_length);
    if (minimum_ratio > maximum_ratio)
      continue;

    const double raw_ratio =
        (query - points[index]).dot(delta) / delta.squaredNorm();
    const double ratio =
        std::max(minimum_ratio, std::min(raw_ratio, maximum_ratio));
    const Eigen::Vector3d candidate = points[index] + ratio * delta;
    const double distance = (query - candidate).norm();
    // 距离并列时保留按路径顺序遇到的较早点，避免在回转或重叠位置跳段。
    if (!result.valid || distance < result.distance - tie_tolerance)
    {
      result.valid = true;
      result.progress_s = segment_start_s + ratio * segment_length;
      result.distance = distance;
      result.segment_index = index;
    }
  }
  return result;
}

// 楼梯暂停期间不运行占据查询或优化器，只把真实 Odometry 投影到当前认证
// 进度之后的折线。全剩余 Path 搜索可跨越较长 root-lock 段，同时通过下界
// 保证进度永不倒退；偏离安全走廊时保持原进度不变。
inline ReferencePathProjection advanceReferencePathProgressDuringStairFreeze(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<double> &arc_lengths,
    const Eigen::Vector3d &odometry_position,
    const double certified_progress_s,
    const double maximum_projection_distance)
{
  ReferencePathProjection projection;
  if (!std::isfinite(certified_progress_s) ||
      !std::isfinite(maximum_projection_distance) ||
      maximum_projection_distance <= 0.0 || arc_lengths.empty())
    return projection;

  projection = projectReferencePathProgress(
      points, arc_lengths, odometry_position,
      certified_progress_s, arc_lengths.back());
  if (!projection.valid ||
      projection.distance > maximum_projection_distance)
    return ReferencePathProjection();

  projection.progress_s = std::max(
      certified_progress_s, projection.progress_s);
  return projection;
}

inline std::vector<Eigen::Vector3d> buildReferencePathGuide(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<double> &arc_lengths,
    const Eigen::Vector3d &actual_start,
    const double requested_start_s,
    const double requested_end_s)
{
  std::vector<Eigen::Vector3d> guide;
  if (!actual_start.allFinite() || points.size() < 2 ||
      arc_lengths.size() != points.size() ||
      !std::isfinite(requested_start_s) ||
      !std::isfinite(requested_end_s) || arc_lengths.back() <= 0.0)
    return guide;

  const double start_s = std::max(
      0.0, std::min(requested_start_s, arc_lengths.back()));
  const double end_s = std::max(
      start_s, std::min(requested_end_s, arc_lengths.back()));
  Eigen::Vector3d path_start;
  Eigen::Vector3d path_end;
  if (!sampleReferencePathAtArcLength(
          points, arc_lengths, start_s, path_start) ||
      !sampleReferencePathAtArcLength(
          points, arc_lengths, end_s, path_end))
    return {};

  constexpr double duplicate_distance = 1.0e-6;
  guide.push_back(actual_start);
  if ((path_start - guide.back()).norm() > duplicate_distance)
    guide.push_back(path_start);
  for (std::size_t index = 1; index + 1 < points.size(); ++index)
  {
    if (arc_lengths[index] <= start_s + 1.0e-9 ||
        arc_lengths[index] >= end_s - 1.0e-9)
      continue;
    if ((points[index] - guide.back()).norm() > duplicate_distance)
      guide.push_back(points[index]);
  }
  if ((path_end - guide.back()).norm() > duplicate_distance)
    guide.push_back(path_end);
  return guide;
}

// 局部轨迹初始化需要从真实机体位置连接到 PCT Path，但这条横向回归连接
// 不是 PCT 的有序路径语义。受阻绕障后的走廊检查只使用从投影点开始的
// semantic guide，避免把平滑回归连接的折角缩短误判为跨越 PCT 路径锚点。
inline std::vector<Eigen::Vector3d> buildReferencePathCorridorGuide(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<double> &arc_lengths,
    const double requested_start_s,
    const double requested_end_s)
{
  if (points.size() < 2 || arc_lengths.size() != points.size() ||
      !std::isfinite(requested_start_s) ||
      !std::isfinite(requested_end_s))
    return {};

  // semantic guide 只能信任由同一条有限、非退化 PCT 折线得到的弧长。
  // 若调用方传入错位或非单调弧长，后续按序回填锚点可能误接到另一段，
  // 因此这里按原始几何重建并逐项核对，异常时直接拒绝。
  const std::vector<double> verified_arc_lengths =
      buildReferencePathArcLengths(points);
  if (verified_arc_lengths.size() != arc_lengths.size())
    return {};
  for (std::size_t index = 0; index < arc_lengths.size(); ++index)
  {
    if (!std::isfinite(arc_lengths[index]))
      return {};
    const double comparison_scale = std::max(
        1.0, std::max(
                 std::abs(verified_arc_lengths[index]),
                 std::abs(arc_lengths[index])));
    if (std::abs(
            verified_arc_lengths[index] - arc_lengths[index]) >
        1.0e-9 * comparison_scale)
      return {};
  }

  const double total_length = verified_arc_lengths.back();
  const double start_s = std::max(
      0.0, std::min(requested_start_s, total_length));
  const double end_s = std::max(
      start_s, std::min(requested_end_s, total_length));
  Eigen::Vector3d path_start;
  if (!sampleReferencePathAtArcLength(
          points, verified_arc_lengths, start_s, path_start))
    return {};
  std::vector<Eigen::Vector3d> guide = buildReferencePathGuide(
      points, verified_arc_lengths, path_start, start_s, end_s);
  if (guide.size() >= 2)
    return guide;

  // 投影已经位于终点或只剩数值量级的前向余量时，常规 guide 只有一个
  // 点。此处只能沿 PCT 的弧长顺序回填投影点之前最近的真实锚点；不能按
  // 欧氏距离搜索，也不能重新加入“真实机体位置 -> 投影点”的人工连接，
  // 否则 U 形或平行近邻路径可能被拼成跨分支捷径。
  const auto projection_anchor = std::lower_bound(
      verified_arc_lengths.begin(), verified_arc_lengths.end(), start_s);
  if (projection_anchor == verified_arc_lengths.begin())
    return {};
  const std::size_t previous_anchor_index =
      static_cast<std::size_t>(
          std::distance(verified_arc_lengths.begin(), projection_anchor) - 1);

  Eigen::Vector3d path_end;
  if (!sampleReferencePathAtArcLength(
          points, verified_arc_lengths, end_s, path_end))
    return {};
  constexpr double duplicate_distance = 1.0e-6;
  if ((path_end - points[previous_anchor_index]).norm() <=
      duplicate_distance)
    return {};

  guide.clear();
  guide.push_back(points[previous_anchor_index]);
  // previous_anchor 与 end_s 之间若仍有真实折点，必须逐点保留，不能用
  // 一条弦跨过这些语义锚点。
  for (std::size_t index = previous_anchor_index + 1;
       index + 1 < points.size(); ++index)
  {
    if (verified_arc_lengths[index] >= end_s - 1.0e-9)
      break;
    if ((points[index] - guide.back()).norm() > duplicate_distance)
      guide.push_back(points[index]);
  }
  if ((path_end - guide.back()).norm() > duplicate_distance)
    guide.push_back(path_end);
  return guide.size() >= 2 ? guide : std::vector<Eigen::Vector3d>();
}

// PCT 或 /initial_path 拥有参考路径的完整终点；SCAN 只能调整局部前视点。
inline bool shouldAdjustGlobalTargetForLocalOccupancy(
    const bool reference_path_mode,
    const int target_occupancy)
{
  return !reference_path_mode && target_occupancy > 0;
}

// community 模式保留 0.2 m 的短规划拒绝语义；PCT reference 模式必须
// 继续执行仍大于真实到达阈值的末段，但只能在存在可用有序 guide 时放行。
inline bool shortReferenceReplanAllowed(
    const double start_to_target_distance,
    const bool require_forward_progress,
    const std::size_t reference_guide_size)
{
  return std::isfinite(start_to_target_distance) &&
         start_to_target_distance > 1.0e-6 &&
         require_forward_progress && reference_guide_size >= 2;
}

inline bool reboundPlanRejectedBeforeInitialization(
    const double start_to_target_distance,
    const bool require_forward_progress,
    const std::size_t reference_guide_size,
    const double community_minimum_distance = 0.2)
{
  if (!std::isfinite(start_to_target_distance) ||
      start_to_target_distance < 0.0 ||
      !std::isfinite(community_minimum_distance) ||
      community_minimum_distance <= 0.0)
    return true;
  return start_to_target_distance < community_minimum_distance &&
         !shortReferenceReplanAllowed(
             start_to_target_distance, require_forward_progress,
             reference_guide_size);
}

// 到达完整参考路径终点且机体已经稳定时，允许发布不移动的 final hold。
inline bool referenceGoalHoldReady(
    const bool reference_path_mode,
    const bool have_active_reference,
    const bool have_target,
    const double distance_xy,
    const double distance_z,
    const double yaw_error,
    const double planar_speed,
    const double vertical_speed,
    const double yaw_rate,
    const double distance_xy_limit,
    const double distance_z_limit,
    const double yaw_error_limit,
    const double planar_speed_limit,
    const double vertical_speed_limit,
    const double yaw_rate_limit)
{
  const double values[] = {
    distance_xy, distance_z, yaw_error,
    planar_speed, vertical_speed, yaw_rate,
    distance_xy_limit, distance_z_limit,
    yaw_error_limit, planar_speed_limit,
    vertical_speed_limit, yaw_rate_limit};
  for (const double value : values)
    if (!std::isfinite(value) || value < 0.0)
      return false;

  return reference_path_mode && have_active_reference && have_target &&
         distance_xy <= distance_xy_limit &&
         distance_z <= distance_z_limit &&
         yaw_error <= yaw_error_limit &&
         planar_speed <= planar_speed_limit &&
         vertical_speed <= vertical_speed_limit &&
         yaw_rate <= yaw_rate_limit;
}

// stationary final hold 只是 moving final 的兜底，不能由一次偶然的低速
// Odometry 样本触发。这里按同一 Path 代际和严格递增的传感器时间累计连续
// 稳定时长；任一物理门失配、时间回拨或输入长间隔都会从当前样本重新计量。
inline bool updateReferenceGoalHoldDwell(
    ReferenceGoalHoldDwellState &state,
    const bool sample_ready,
    const std::int64_t sample_stamp_ns,
    const std::int64_t path_stamp_ns,
    const double required_dwell_sec,
    const double maximum_sample_gap_sec)
{
  if (sample_stamp_ns <= 0 || path_stamp_ns <= 0 ||
      !std::isfinite(required_dwell_sec) || required_dwell_sec <= 0.0 ||
      !std::isfinite(maximum_sample_gap_sec) ||
      maximum_sample_gap_sec <= 0.0)
  {
    state = ReferenceGoalHoldDwellState{};
    return false;
  }

  const bool generation_changed = state.path_stamp_ns != path_stamp_ns;
  const bool time_rewound =
      state.last_sample_stamp_ns > 0 &&
      sample_stamp_ns < state.last_sample_stamp_ns;
  const double sample_gap_sec = state.last_sample_stamp_ns > 0 ?
      static_cast<double>(sample_stamp_ns - state.last_sample_stamp_ns) *
      1.0e-9 : 0.0;
  const bool sample_gap_exceeded =
      state.last_sample_stamp_ns > 0 &&
      sample_stamp_ns > state.last_sample_stamp_ns &&
      sample_gap_sec > maximum_sample_gap_sec;
  if (generation_changed || time_rewound || sample_gap_exceeded ||
      !sample_ready)
  {
    state = ReferenceGoalHoldDwellState{};
    state.path_stamp_ns = path_stamp_ns;
    state.last_sample_stamp_ns = sample_stamp_ns;
    if (!sample_ready)
      return false;
  }

  state.path_stamp_ns = path_stamp_ns;
  if (state.stable_since_ns <= 0)
    state.stable_since_ns = sample_stamp_ns;
  state.last_sample_stamp_ns = sample_stamp_ns;
  state.stable_duration_sec = std::max(
      0.0,
      static_cast<double>(sample_stamp_ns - state.stable_since_ns) * 1.0e-9);
  return state.stable_duration_sec + 1.0e-12 >= required_dwell_sec;
}

// reference-path 失败重试必须节流；其他模式保留 community 原有时序。
inline bool referenceRetryReady(
    const bool reference_path_mode,
    const double now_sec,
    const double retry_not_before_sec)
{
  if (!reference_path_mode)
    return true;
  return std::isfinite(now_sec) && std::isfinite(retry_not_before_sec) &&
         now_sec >= retry_not_before_sec;
}

// 四足机器人的局部轨迹生命周期必须以真实里程计进度为准。
inline ReferenceExecutionAction decideReferenceExecutionAction(
    const bool trajectory_expired,
    const bool have_target,
    const bool local_target_is_final,
    const bool controller_execution_frozen,
    const bool terminal_convergence_grace_available,
    const double odom_distance_to_goal,
    const double odom_distance_from_trajectory_start,
    const double replan_distance,
    const double no_replan_distance)
{
  if (trajectory_expired)
  {
    if (!have_target)
      return ReferenceExecutionAction::kFinish;
    if (local_target_is_final &&
        odom_distance_to_goal <= no_replan_distance &&
        (controller_execution_frozen || terminal_convergence_grace_available))
      return ReferenceExecutionAction::kHold;
    return ReferenceExecutionAction::kReplan;
  }

  if (odom_distance_to_goal < no_replan_distance ||
      odom_distance_from_trajectory_start < replan_distance)
  {
    return ReferenceExecutionAction::kHold;
  }
  return ReferenceExecutionAction::kReplan;
}

// reference-path 模式的临时局部失败不能丢弃仍然有效的 ROS 2 Path。
inline ReferenceEmergencyRecoveryAction decideReferenceEmergencyRecovery(
    const bool reference_path_mode,
    const bool have_active_reference,
    const bool have_target)
{
  if (reference_path_mode && have_active_reference && have_target)
  {
    return ReferenceEmergencyRecoveryAction::kRetryActiveReference;
  }
  return ReferenceEmergencyRecoveryAction::kWaitForNewTarget;
}

}  // namespace scan_planner

#endif  // SCAN_PLANNER__REFERENCE_EXECUTION_H_
