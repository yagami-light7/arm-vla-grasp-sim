"""把安全速度命令写入标准 Go2 的 MoE-CTS command buffer。"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


class MoeCtsCommandBufferError(RuntimeError):
    """表示 command buffer 合同或写入过程不满足安全要求。"""


class MoeCtsCommandBufferSink:
    """实现 ``CmdVelToPolicyAdapter`` 所需的最小 command sink。

    ``go2_rl_robotlab`` 的速度指令保存在形状为 ``[1, 3]`` 的 tensor 中，
    三个分量依次为机体系 ``vx、vy、wz``。本类不导入 Torch，而是复用传入
    tensor 的 ``new_tensor`` 与 ``copy_``，因此不会把 CPU tensor 误写进
    CUDA command buffer。
    """

    EXPECTED_SHAPE = (1, 3)

    def __init__(self, command_buffer: Any) -> None:
        shape = getattr(command_buffer, "shape", None)
        try:
            normalized_shape = tuple(int(value) for value in shape)
        except (TypeError, ValueError):
            normalized_shape = ()
        if normalized_shape != self.EXPECTED_SHAPE:
            raise MoeCtsCommandBufferError(
                "MoE-CTS base_velocity command buffer 形状必须为 "
                f"{self.EXPECTED_SHAPE}，实际为 {normalized_shape or shape!r}。"
            )
        if not callable(getattr(command_buffer, "new_tensor", None)):
            raise MoeCtsCommandBufferError(
                "command buffer 必须提供 new_tensor()，以保持 device 与 dtype。"
            )
        self._command_buffer = command_buffer
        self._last_command = (0.0, 0.0, 0.0)
        self._write_count = 0

    @property
    def last_command(self) -> tuple[float, float, float]:
        """返回最近一次成功写入的 ``vx、vy、wz``。"""

        return self._last_command

    @property
    def write_count(self) -> int:
        """返回成功写入次数，包含 claim/release 触发的零速度。"""

        return self._write_count

    @staticmethod
    def _finite_command(vx: Any, vy: Any, wz: Any) -> tuple[float, float, float]:
        values: list[float] = []
        for name, value in (("vx", vx), ("vy", vy), ("wz", wz)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} 必须是有限实数。")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} 必须是有限实数。")
            values.append(converted)
        return values[0], values[1], values[2]

    def apply_base_command(self, vx: float, vy: float, wz: float) -> None:
        """在原 tensor 的 device/dtype 上一次性覆写三个速度分量。"""

        command = self._finite_command(vx, vy, wz)
        try:
            row = self._command_buffer[0]
            source = self._command_buffer.new_tensor(command)
            row.copy_(source)
        except Exception as exc:
            # copy_ 正常情况下是整行操作；异常时仍尝试清零，禁止保留不完整命令。
            try:
                self._command_buffer.zero_()
            except Exception:
                pass
            raise MoeCtsCommandBufferError(
                "写入 MoE-CTS base_velocity command buffer 失败。"
            ) from exc
        self._last_command = command
        self._write_count += 1


__all__ = [
    "MoeCtsCommandBufferError",
    "MoeCtsCommandBufferSink",
]
