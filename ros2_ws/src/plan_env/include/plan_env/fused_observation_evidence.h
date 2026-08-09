#ifndef PLAN_ENV__FUSED_OBSERVATION_EVIDENCE_H_
#define PLAN_ENV__FUSED_OBSERVATION_EVIDENCE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

inline constexpr std::size_t kFusedObservationHistoryCapacity = 64U;

struct FusedObservationRecord
{
  std::uint64_t sequence{0U};
  std::int64_t cloud_acquisition_stamp_ns{0};
};

struct FusedObservationEvidence
{
  bool valid{false};
  bool ready{false};
  std::uint64_t distinct_stamp_count{0U};
  std::uint64_t current_sequence{0U};
};

// 只根据完整、连续的融合记录形成主动复观测证据。采集时间相同的重复消息
// 只算一帧；任何记录缺口、未来时间戳或序号倒退都不能推断为新视角。
inline FusedObservationEvidence evaluateFusedObservationEvidence(
    const std::deque<FusedObservationRecord> &history,
    const std::uint64_t current_sequence,
    const std::int64_t settle_stamp_ns,
    const std::uint64_t baseline_sequence,
    const std::int64_t now_ns,
    const std::uint64_t required_observations) noexcept
{
  FusedObservationEvidence evidence;
  evidence.current_sequence = current_sequence;
  if (settle_stamp_ns <= 0 || now_ns < settle_stamp_ns ||
      required_observations == 0U ||
      required_observations > kFusedObservationHistoryCapacity ||
      baseline_sequence > current_sequence ||
      current_sequence == std::numeric_limits<std::uint64_t>::max() ||
      baseline_sequence == std::numeric_limits<std::uint64_t>::max())
    return evidence;

  if (current_sequence == 0U)
  {
    if (!history.empty() || baseline_sequence != 0U)
      return evidence;
    evidence.valid = true;
    return evidence;
  }
  if (history.empty() || history.back().sequence != current_sequence)
    return evidence;

  std::uint64_t expected_sequence = history.front().sequence;
  if (expected_sequence == 0U)
    return evidence;
  for (const auto &record : history)
  {
    if (record.sequence != expected_sequence ||
        record.cloud_acquisition_stamp_ns <= 0)
      return evidence;
    if (expected_sequence == std::numeric_limits<std::uint64_t>::max())
    {
      if (&record != &history.back())
        return evidence;
    }
    else
    {
      ++expected_sequence;
    }
  }

  if (baseline_sequence == current_sequence)
  {
    evidence.valid = true;
    return evidence;
  }
  if (baseline_sequence == std::numeric_limits<std::uint64_t>::max())
    return evidence;
  const std::uint64_t first_required_sequence = baseline_sequence + 1U;
  // 相关区间在有界历史中被截断时，不能用 current-baseline 猜测帧数。
  if (history.front().sequence > first_required_sequence)
    return evidence;

  std::vector<std::int64_t> distinct_stamps;
  distinct_stamps.reserve(std::min<std::size_t>(
      history.size(),
      required_observations > std::numeric_limits<std::size_t>::max()
          ? std::numeric_limits<std::size_t>::max()
          : static_cast<std::size_t>(required_observations)));
  std::uint64_t next_sequence = first_required_sequence;
  for (const auto &record : history)
  {
    if (record.sequence < first_required_sequence)
      continue;
    if (record.sequence != next_sequence ||
        record.cloud_acquisition_stamp_ns > now_ns)
      return evidence;
    if (record.cloud_acquisition_stamp_ns > settle_stamp_ns &&
        std::find(
            distinct_stamps.begin(), distinct_stamps.end(),
            record.cloud_acquisition_stamp_ns) == distinct_stamps.end())
      distinct_stamps.push_back(record.cloud_acquisition_stamp_ns);
    if (next_sequence != current_sequence)
      ++next_sequence;
  }
  if (next_sequence != current_sequence)
    return evidence;

  evidence.valid = true;
  evidence.distinct_stamp_count =
      static_cast<std::uint64_t>(distinct_stamps.size());
  evidence.ready =
      evidence.distinct_stamp_count >= required_observations;
  return evidence;
}

#endif  // PLAN_ENV__FUSED_OBSERVATION_EVIDENCE_H_
