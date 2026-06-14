"""夹爪命令封装。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryGripperController:
    """返回 pipeline/action_applier 可解释的离散夹爪命令。"""

    open_command: str = "open"
    close_command: str = "close"
    hold_command: str = "hold"

    def command_open(self) -> str:
        return self.open_command

    def command_close(self) -> str:
        return self.close_command

    def command_hold(self) -> str:
        return self.hold_command
