"""Isaac articulation action 构造与下发。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from source.interfaces import RobotAction


DEFAULT_ARM_JOINT_NAMES = (
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
)
DEFAULT_GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")


@dataclass(frozen=True)
class NamedJointActionConfig:
    """关节名到 Isaac DOF index 的固定约定。"""

    arm_joint_names: tuple[str, ...] = DEFAULT_ARM_JOINT_NAMES
    gripper_joint_names: tuple[str, ...] = DEFAULT_GRIPPER_JOINT_NAMES
    gripper_open_position: float = 0.04
    gripper_close_position: float = 0.0


class NamedJointActionApplier:
    """只通过 apply_action 下发 position target，不直接写 articulation state。"""

    def __init__(
        self,
        robot: Any,
        *,
        config: NamedJointActionConfig | None = None,
        articulation_action_factory: Callable[..., Any] | None = None,
    ):
        self.robot = robot
        self.config = config or NamedJointActionConfig()
        self._articulation_action_factory = articulation_action_factory

    def apply(self, action: RobotAction) -> dict[str, Any]:
        target_by_name: dict[str, float] = {}
        report: dict[str, Any] = {
            "available": True,
            "applied": False,
            "source": action.source,
            "arm_targeted": False,
            "gripper_targeted": False,
            "joint_names": [],
            "joint_indices": [],
            "target_positions": [],
            # 这里显式标记：本模块不调用 set_joint_positions，不推进 world.step。
            "uses_direct_joint_state": False,
            "world_step_owned_by_pipeline": True,
        }

        if action.arm_joint_positions is not None:
            arm_report = self._collect_arm_targets(
                action.arm_joint_positions,
                target_by_name,
                action_metadata=action.metadata or {},
            )
            report.update(arm_report)

        gripper_report = self._collect_gripper_targets(action, target_by_name)
        report.update(gripper_report)

        if not target_by_name:
            report["reason"] = "no_joint_targets"
            return report

        dof_names = self._dof_names()
        joint_names = tuple(target_by_name)
        joint_indices = tuple(_index_for_joint(dof_names, name) for name in joint_names)
        target_positions = tuple(target_by_name[name] for name in joint_names)
        articulation_action = self._make_articulation_action(
            joint_positions=target_positions,
            joint_indices=joint_indices,
        )
        self.robot.apply_action(articulation_action)

        report.update(
            {
                "applied": True,
                "joint_names": joint_names,
                "joint_indices": joint_indices,
                "target_positions": target_positions,
            }
        )
        return report

    def _collect_arm_targets(
        self,
        arm_joint_positions: tuple[float, ...],
        target_by_name: dict[str, float],
        *,
        action_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        arm_joint_names = tuple(
            str(name)
            for name in action_metadata.get("arm_joint_names", self.config.arm_joint_names)
        )
        if len(arm_joint_positions) != len(arm_joint_names):
            raise RuntimeError(
                "arm_joint_positions length does not match arm_joint_names: "
                f"{len(arm_joint_positions)} != {len(arm_joint_names)}"
            )
        for name, value in zip(arm_joint_names, arm_joint_positions):
            target_by_name[name] = float(value)
        return {
            "arm_targeted": True,
            "arm_joint_names": arm_joint_names,
            "arm_joint_positions": tuple(float(value) for value in arm_joint_positions),
        }

    def _collect_gripper_targets(
        self,
        action: RobotAction,
        target_by_name: dict[str, float],
    ) -> dict[str, Any]:
        if action.gripper_command is None:
            return {"gripper_targeted": False}

        metadata = action.metadata or {}
        if "gripper_joint_positions" in metadata:
            joint_names = tuple(
                str(name)
                for name in metadata.get("gripper_joint_names", self.config.gripper_joint_names)
            )
            positions = tuple(float(value) for value in metadata["gripper_joint_positions"])
        elif action.gripper_command == "open":
            joint_names = self.config.gripper_joint_names
            positions = tuple(self.config.gripper_open_position for _ in joint_names)
        elif action.gripper_command == "close":
            joint_names = self.config.gripper_joint_names
            positions = tuple(self.config.gripper_close_position for _ in joint_names)
        else:
            return {
                "gripper_targeted": False,
                "gripper_command": action.gripper_command,
                "gripper_reason": "hold_without_explicit_target",
            }

        if len(joint_names) != len(positions):
            raise RuntimeError(
                "gripper_joint_positions length does not match gripper_joint_names: "
                f"{len(positions)} != {len(joint_names)}"
            )
        for name, value in zip(joint_names, positions):
            target_by_name[name] = float(value)
        return {
            "gripper_targeted": True,
            "gripper_command": action.gripper_command,
            "gripper_joint_names": joint_names,
            "gripper_joint_positions": positions,
        }

    def _make_articulation_action(
        self,
        *,
        joint_positions: tuple[float, ...],
        joint_indices: tuple[int, ...],
    ) -> Any:
        factory = self._articulation_action_factory
        if factory is None:
            from isaacsim.core.utils.types import ArticulationAction

            factory = ArticulationAction
        try:
            return factory(joint_positions=joint_positions, joint_indices=joint_indices)
        except TypeError:
            # 部分 Isaac 版本需要 list 而不是 tuple；仍然只走 apply_action。
            return factory(
                joint_positions=list(joint_positions),
                joint_indices=list(joint_indices),
            )

    def _dof_names(self) -> tuple[str, ...]:
        raw_names = getattr(self.robot, "dof_names", None)
        if raw_names is None and hasattr(self.robot, "get_dof_names"):
            raw_names = self.robot.get_dof_names()
        if raw_names is None:
            raise RuntimeError("robot does not expose dof_names")
        return tuple(str(name) for name in raw_names)


def _index_for_joint(dof_names: tuple[str, ...], joint_name: str) -> int:
    try:
        return dof_names.index(joint_name)
    except ValueError as exc:
        raise RuntimeError(f"Isaac articulation does not contain joint: {joint_name}") from exc
