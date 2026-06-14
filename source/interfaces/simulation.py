"""Simulation contracts shared by real and dry-run backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .task import EpisodeSpec


@dataclass(frozen=True)
class SimulationState:
    """One immutable observation of the simulation."""

    step_index: int
    timestamp: float
    robot_root_pose: tuple[float, float, float, float, float, float, float]
    robot_root_velocity: tuple[float, float, float, float, float, float]
    joint_positions: tuple[float, ...] = ()
    joint_velocities: tuple[float, ...] = ()
    tcp_pose: tuple[float, float, float, float, float, float, float] | None = None
    object_pose: tuple[float, float, float, float, float, float, float] | None = None
    object_velocity: tuple[float, float, float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotAction:
    """Combined base, arm, and gripper command for one simulation tick."""

    base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_joint_positions: tuple[float, ...] | None = None
    gripper_command: str | None = None
    source: str = "idle"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def idle(cls, *, source: str = "idle") -> "RobotAction":
        return cls(source=source)


class SimulationRuntime(Protocol):
    """Own the stage/world but never decide pipeline state transitions."""

    def build(self, episode_spec: "EpisodeSpec") -> None:
        ...

    def reset(self, episode_spec: "EpisodeSpec", *, seed: int) -> None:
        ...

    def read(self) -> SimulationState:
        ...

    def apply(self, action: RobotAction) -> None:
        ...

    def step(self, *, render: bool) -> None:
        ...

    def prepare_object_for_pick(self, episode_spec: "EpisodeSpec") -> dict[str, Any]:
        """在 pick 规划前恢复任务初始条件并唤醒动态物体。"""

        ...

    def pause(self) -> dict[str, Any]:
        """暂停物理推进，同时保留 stage 和控制目标供 GUI 检查。"""

        ...

    def close(self) -> None:
        ...
