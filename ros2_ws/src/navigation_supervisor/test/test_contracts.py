"""类型化 supervisor 的身份、序列和有界重试合同测试。"""

from dataclasses import dataclass

import pytest

from navigation_supervisor.contracts import (
    GoalIdentity,
    MonotonicSequence,
    ReplanTransaction,
    SequenceDisposition,
    bspline_valid_until_ns,
    stamp_to_nanoseconds,
)


@dataclass
class _Stamp:
    sec: int
    nanosec: int = 0


def _goal() -> GoalIdentity:
    return GoalIdentity(
        goal_id=7,
        stamp_ns=1_000_000_000,
        frame_id="world",
        pose=(0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
    )


def test_stamp_rejects_zero_negative_and_out_of_range_nanoseconds() -> None:
    assert stamp_to_nanoseconds(_Stamp(2, 3)) == 2_000_000_003
    with pytest.raises(ValueError, match="零时间戳"):
        stamp_to_nanoseconds(_Stamp(0, 0))
    with pytest.raises(ValueError, match="规范范围"):
        stamp_to_nanoseconds(_Stamp(-1, 0))
    with pytest.raises(ValueError, match="规范范围"):
        stamp_to_nanoseconds(_Stamp(1, 1_000_000_000))


def test_monotonic_sequence_classifies_all_repeated_inputs() -> None:
    tracker = MonotonicSequence()

    assert tracker.observe(3, ("a",)) is SequenceDisposition.NEW
    assert tracker.observe(3, ("a",)) is SequenceDisposition.DUPLICATE
    assert tracker.observe(2, ("old",)) is SequenceDisposition.STALE
    assert tracker.observe(3, ("different",)) is SequenceDisposition.CONFLICT
    assert tracker.observe(4, ("b",)) is SequenceDisposition.NEW


def test_bspline_valid_until_requires_exact_positive_duration_inputs() -> None:
    result = bspline_valid_until_ns(
        order=3,
        control_point_count=6,
        knots=(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0),
        start_time_ns=10_000_000_000,
    )
    assert result == 13_000_000_000
    with pytest.raises(ValueError, match="单调"):
        bspline_valid_until_ns(
            order=3,
            control_point_count=6,
            knots=(0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 3.0),
            start_time_ns=10_000_000_000,
        )


def test_replan_transaction_is_bounded_and_reuses_fixed_identity() -> None:
    transaction = ReplanTransaction(
        request_id=11,
        core_request_id=2,
        goal=_goal(),
        expected_path_stamp_ns=4_000_000_000,
        request_stamp_ns=5_000_000_000,
        reason="predicted_collision",
        max_attempts=2,
        retry_period_ns=100,
        response_timeout_ns=50,
        epoch=3,
    )

    assert transaction.can_send(5_000_000_000)
    transaction.mark_sent(5_000_000_000)
    assert transaction.response_expired(5_000_000_050)
    transaction.mark_retryable_failure()
    assert not transaction.can_send(5_000_000_099)
    assert transaction.can_send(5_000_000_100)
    transaction.mark_sent(5_000_000_100)
    transaction.mark_retryable_failure()
    assert transaction.exhausted
    assert not transaction.can_send(5_000_000_200)
