from __future__ import annotations

import math

import pytest

from source.navigation.adapters.moe_cts_command_sink import (
    MoeCtsCommandBufferError,
    MoeCtsCommandBufferSink,
)
from source.navigation.cmd_vel_to_policy import (
    CmdVelToPolicyAdapter,
    CmdVelToPolicyConfig,
)


class _FakeRow:
    def __init__(self, owner: "_FakeCommandBuffer") -> None:
        self._owner = owner

    def copy_(self, source: tuple[float, float, float]) -> None:
        if self._owner.fail_copy:
            raise RuntimeError("copy failed")
        self._owner.values[:] = source


class _FakeCommandBuffer:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (1, 3),
        fail_copy: bool = False,
    ) -> None:
        self.shape = shape
        self.fail_copy = fail_copy
        self.values = [9.0, 9.0, 9.0]
        self.new_tensor_inputs: list[tuple[float, float, float]] = []
        self.zero_count = 0

    def __getitem__(self, index: int) -> _FakeRow:
        assert index == 0
        return _FakeRow(self)

    def new_tensor(
        self,
        values: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        normalized = tuple(float(value) for value in values)
        self.new_tensor_inputs.append(normalized)
        return normalized

    def zero_(self) -> None:
        self.zero_count += 1
        self.values[:] = (0.0, 0.0, 0.0)


def test_writes_complete_command_without_importing_torch() -> None:
    buffer = _FakeCommandBuffer()
    sink = MoeCtsCommandBufferSink(buffer)

    sink.apply_base_command(0.4, -0.2, 0.7)

    assert buffer.values == pytest.approx([0.4, -0.2, 0.7])
    assert buffer.new_tensor_inputs == [(0.4, -0.2, 0.7)]
    assert sink.last_command == pytest.approx((0.4, -0.2, 0.7))
    assert sink.write_count == 1


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 4), ()])
def test_rejects_wrong_command_buffer_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(MoeCtsCommandBufferError, match="形状"):
        MoeCtsCommandBufferSink(_FakeCommandBuffer(shape=shape))


@pytest.mark.parametrize(
    "command",
    [
        (True, 0.0, 0.0),
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, object()),
    ],
)
def test_rejects_nonfinite_or_non_numeric_commands(command: tuple[object, ...]) -> None:
    buffer = _FakeCommandBuffer()
    sink = MoeCtsCommandBufferSink(buffer)

    with pytest.raises((TypeError, ValueError)):
        sink.apply_base_command(*command)

    assert buffer.values == [9.0, 9.0, 9.0]
    assert sink.write_count == 0


def test_failed_copy_clears_buffer_and_does_not_count_write() -> None:
    buffer = _FakeCommandBuffer(fail_copy=True)
    sink = MoeCtsCommandBufferSink(buffer)

    with pytest.raises(MoeCtsCommandBufferError, match="写入"):
        sink.apply_base_command(0.2, 0.0, 0.1)

    assert buffer.values == [0.0, 0.0, 0.0]
    assert buffer.zero_count == 1
    assert sink.last_command == (0.0, 0.0, 0.0)
    assert sink.write_count == 0


def test_real_safety_gate_writes_command_then_times_out_to_zero() -> None:
    buffer = _FakeCommandBuffer()
    sink = MoeCtsCommandBufferSink(buffer)
    gate = CmdVelToPolicyAdapter(
        sink,
        CmdVelToPolicyConfig(
            max_vx=0.8,
            max_vy=0.5,
            max_wz=0.8,
            max_vx_rate=100.0,
            max_vy_rate=100.0,
            max_wz_rate=100.0,
            cmd_vel_timeout_s=0.25,
            require_odometry=False,
            require_point_cloud=False,
            require_navigation_status=False,
        ),
        ownership_resource=("test_moe_cts_command_buffer", id(buffer)),
    )

    gate.claim("test_ros_cmd_vel", 1.0)
    try:
        gate.accept_cmd_vel(
            (0.4, -0.1, 0.3),
            owner_id="test_ros_cmd_vel",
            received_at=1.0,
        )
        gate.renew_control_lease("test_ros_cmd_vel", 1.01)
        moving = gate.write(owner_id="test_ros_cmd_vel", now=1.01)
        assert moving.motion_allowed is True
        assert buffer.values == pytest.approx([0.4, -0.1, 0.3])

        gate.renew_control_lease("test_ros_cmd_vel", 1.26)
        stopped = gate.write(owner_id="test_ros_cmd_vel", now=1.26)
        assert stopped.motion_allowed is False
        assert stopped.stop_reasons == ("cmd_vel_timeout",)
        assert buffer.values == [0.0, 0.0, 0.0]
    finally:
        gate.release(owner_id="test_ros_cmd_vel", now=1.26)
