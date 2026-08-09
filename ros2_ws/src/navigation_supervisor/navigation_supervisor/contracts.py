"""导航 supervisor 的消息身份与幂等事务辅助类型。

本模块不依赖 ``rclpy``，因此时间回退、消息代际冲突与固定重试都可以在
普通 pytest 中确定性验证。ROS 2 node 只负责把 wire message 转成这些类型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable


NANOSECONDS_PER_SECOND = 1_000_000_000


def stamp_to_nanoseconds(stamp: object, *, allow_zero: bool = False) -> int:
    """严格读取 ROS ``Time``；非法、负数及默认零时间一律拒绝。"""

    try:
        seconds = int(getattr(stamp, "sec"))
        nanoseconds = int(getattr(stamp, "nanosec"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("时间戳必须包含整数 sec 与 nanosec。") from exc
    if seconds < 0 or not 0 <= nanoseconds < NANOSECONDS_PER_SECOND:
        raise ValueError("时间戳超出 ROS Time 的非负规范范围。")
    result = seconds * NANOSECONDS_PER_SECOND + nanoseconds
    if result == 0 and not allow_zero:
        raise ValueError("关键消息不允许使用零时间戳。")
    return result


def nanoseconds_to_seconds(value_ns: int) -> float:
    """把非负纳秒转换成状态机使用的秒。"""

    if isinstance(value_ns, bool) or not isinstance(value_ns, int):
        raise ValueError("纳秒时间必须是整数。")
    if value_ns < 0:
        raise ValueError("纳秒时间不能为负数。")
    return value_ns / NANOSECONDS_PER_SECOND


def finite_tuple(
    values: Iterable[object],
    *,
    field_name: str,
) -> tuple[float, ...]:
    """把数值序列规范为有限浮点元组，供完整 payload identity 使用。"""

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} 不能包含布尔值。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 必须只包含数值。") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name} 必须只包含有限数值。")
        normalized.append(number)
    return tuple(normalized)


@dataclass(frozen=True)
class GoalIdentity:
    """PCT 活动目标的完整、不可变身份。"""

    goal_id: int
    stamp_ns: int
    frame_id: str
    pose: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.goal_id <= 0 or self.stamp_ns <= 0:
            raise ValueError("goal_id 与目标时间戳必须为正数。")
        if not self.frame_id:
            raise ValueError("目标 frame_id 不能为空。")
        if len(self.pose) != 7 or not all(math.isfinite(v) for v in self.pose):
            raise ValueError("目标 pose 必须是七个有限数值。")


@dataclass(frozen=True)
class PathIdentity:
    """已实际收到的非空全局 Path 身份。"""

    stamp_ns: int
    frame_id: str
    point_count: int
    payload: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if self.stamp_ns <= 0:
            raise ValueError("Path stamp 必须为正数。")
        if not self.frame_id:
            raise ValueError("Path frame_id 不能为空。")
        if self.point_count < 2 or len(self.payload) != self.point_count:
            raise ValueError("有效全局 Path 至少包含两个完整 pose。")


@dataclass(frozen=True)
class TrajectoryIdentity:
    """B-spline 与 ControllerStatus 逐字段对账使用的身份。"""

    reference_path_stamp_ns: int
    bspline_header_stamp_ns: int
    start_time_ns: int
    trajectory_id: int
    is_final: bool
    emergency_stop: bool

    def __post_init__(self) -> None:
        if (
            self.reference_path_stamp_ns <= 0
            or self.bspline_header_stamp_ns <= 0
            or self.start_time_ns <= 0
        ):
            raise ValueError("B-spline 三个关键时间身份必须为正数。")
        if isinstance(self.trajectory_id, bool):
            raise ValueError("trajectory_id 不能是布尔值。")


class SequenceDisposition(str, Enum):
    """单调序列校验结果。"""

    NEW = "new"
    DUPLICATE = "duplicate"
    STALE = "stale"
    CONFLICT = "conflict"


class MonotonicSequence:
    """拒绝乱序消息，并辨别精确重复与同序号冲突。"""

    def __init__(self) -> None:
        self._sequence = -1
        self._signature: Hashable | None = None

    @property
    def latest(self) -> int:
        """返回最近接受的序号；尚未接受时为 ``-1``。"""

        return self._sequence

    def observe(
        self,
        sequence: int,
        signature: Hashable,
    ) -> SequenceDisposition:
        """观察一条消息，但只有更大序号才推进 tracker。"""

        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise ValueError("status_sequence 必须是非负整数。")
        if sequence < 0:
            raise ValueError("status_sequence 不能为负数。")
        if sequence < self._sequence:
            return SequenceDisposition.STALE
        if sequence == self._sequence:
            if signature == self._signature:
                return SequenceDisposition.DUPLICATE
            return SequenceDisposition.CONFLICT
        self._sequence = sequence
        self._signature = signature
        return SequenceDisposition.NEW

    def reset(self) -> None:
        """在 ROS 时钟回退时清除上一时间域的序列。"""

        self._sequence = -1
        self._signature = None


def bspline_valid_until_ns(
    *,
    order: int,
    control_point_count: int,
    knots: Iterable[object],
    start_time_ns: int,
) -> int:
    """按 SCAN 合约计算 B-spline 绝对有效截止时间。"""

    if isinstance(order, bool) or order < 1:
        raise ValueError("B-spline order 必须是正整数。")
    if (
        isinstance(control_point_count, bool)
        or control_point_count < order + 1
    ):
        raise ValueError("B-spline 控制点数不足。")
    normalized_knots = finite_tuple(knots, field_name="knots")
    if len(normalized_knots) <= control_point_count:
        raise ValueError("B-spline knot 数组不能计算有效时长。")
    if any(
        right < left
        for left, right in zip(normalized_knots, normalized_knots[1:])
    ):
        raise ValueError("B-spline knots 必须单调不减。")
    duration_s = (
        normalized_knots[control_point_count] - normalized_knots[order]
    )
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("B-spline 有效时长必须为有限正数。")
    if start_time_ns <= 0:
        raise ValueError("B-spline start_time 必须为正数。")
    duration_ns = int(round(duration_s * NANOSECONDS_PER_SECOND))
    if duration_ns <= 0:
        raise ValueError("B-spline 有效时长低于纳秒分辨率。")
    return start_time_ns + duration_ns


@dataclass
class ReplanTransaction:
    """固定 wire payload 的有界幂等 REPLAN 事务状态。"""

    request_id: int
    core_request_id: int
    goal: GoalIdentity
    expected_path_stamp_ns: int
    request_stamp_ns: int
    reason: str
    max_attempts: int
    retry_period_ns: int
    response_timeout_ns: int
    epoch: int
    attempts: int = 0
    last_sent_ns: int | None = None
    response_deadline_ns: int | None = None
    in_flight: bool = False
    acknowledged: bool = False
    terminal_error: str = ""

    def __post_init__(self) -> None:
        if self.request_id <= 0 or self.core_request_id <= 0:
            raise ValueError("REPLAN request id 必须为正数。")
        if self.expected_path_stamp_ns <= 0 or self.request_stamp_ns <= 0:
            raise ValueError("REPLAN 的 Path 与 header stamp 必须为正数。")
        if not self.reason.strip():
            raise ValueError("REPLAN reason 不能为空。")
        if self.max_attempts < 1:
            raise ValueError("REPLAN max_attempts 必须至少为 1。")
        if self.retry_period_ns <= 0 or self.response_timeout_ns <= 0:
            raise ValueError("REPLAN 重试与响应超时必须为正数。")

    @property
    def exhausted(self) -> bool:
        """事务是否已用完全部发送次数。"""

        return self.attempts >= self.max_attempts

    @property
    def terminal(self) -> bool:
        """事务是否已经 ACK 或确定失败。"""

        return self.acknowledged or bool(self.terminal_error)

    def can_send(self, now_ns: int) -> bool:
        """当前是否允许发送首次请求或完全相同的下一次重试。"""

        if self.terminal or self.in_flight or self.exhausted:
            return False
        if self.last_sent_ns is None:
            return True
        return now_ns - self.last_sent_ns >= self.retry_period_ns

    def mark_sent(self, now_ns: int) -> None:
        """记录一次发送，并启动该次 ACK 响应期限。"""

        if not self.can_send(now_ns):
            raise RuntimeError("当前 REPLAN 事务不能发送。")
        self.attempts += 1
        self.last_sent_ns = now_ns
        self.response_deadline_ns = now_ns + self.response_timeout_ns
        self.in_flight = True

    def response_expired(self, now_ns: int) -> bool:
        """当前异步调用是否已经超过固定响应期限。"""

        return bool(
            self.in_flight
            and self.response_deadline_ns is not None
            and now_ns >= self.response_deadline_ns
        )

    def mark_retryable_failure(self) -> None:
        """清除 in-flight 标记；下一次仍复用原 wire request。"""

        self.in_flight = False
        self.response_deadline_ns = None

    def mark_acknowledged(self) -> None:
        """锁存 typed ACK。"""

        self.in_flight = False
        self.response_deadline_ns = None
        self.acknowledged = True

    def mark_terminal_error(self, reason: str) -> None:
        """锁存 STALE、CONFLICT、REJECTED 或协议错误。"""

        self.in_flight = False
        self.response_deadline_ns = None
        self.terminal_error = str(reason).strip() or "replan_terminal_error"


__all__ = [
    "GoalIdentity",
    "MonotonicSequence",
    "NANOSECONDS_PER_SECOND",
    "PathIdentity",
    "ReplanTransaction",
    "SequenceDisposition",
    "TrajectoryIdentity",
    "bspline_valid_until_ns",
    "finite_tuple",
    "nanoseconds_to_seconds",
    "stamp_to_nanoseconds",
]
