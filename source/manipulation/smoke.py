"""manipulation smoke 使用的确定性分段计划。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from source.interfaces import ArmPlan, EpisodeSpec, ManipulationPlanner, SimulationState


@dataclass(frozen=True)
class SegmentedSmokeManipulationPlanner(ManipulationPlanner):
    """生成短分段计划，用于验证 planner/executor/action 合同。"""

    joint_names: tuple[str, ...] = (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )
    gripper_joint_names: tuple[str, ...] = ("arm_joint7", "arm_joint8")
    open_position: tuple[float, float] = (0.04, 0.04)
    close_position: tuple[float, float] = (0.0, 0.0)

    def plan_pick(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state
        return self._plan(
            operation="pick",
            object_prim_path=episode_spec.object_prim_path,
            segments=(
                self._motion_segment(
                    "approach_to_grasp",
                    (
                        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
                        (0.10, 0.08, 0.06, 0.04, 0.02, 0.01),
                    ),
                ),
                self._gripper_segment("close_gripper", self.close_position),
                self._motion_segment(
                    "lift_object",
                    (
                        (0.10, 0.08, 0.06, 0.04, 0.02, 0.01),
                        (0.16, 0.12, 0.08, 0.06, 0.04, 0.02),
                    ),
                ),
            ),
        )

    def plan_place(self, state: SimulationState, episode_spec: EpisodeSpec) -> ArmPlan:
        del state
        if episode_spec.place_target_pose is None:
            raise RuntimeError("place target pose is missing")
        return self._plan(
            operation="place",
            object_prim_path=episode_spec.object_prim_path,
            segments=(
                self._motion_segment(
                    "approach_to_place",
                    (
                        (0.16, 0.12, 0.08, 0.06, 0.04, 0.02),
                        (0.12, 0.09, 0.06, 0.04, 0.02, 0.01),
                    ),
                ),
                self._gripper_segment("open_gripper", self.open_position),
                self._motion_segment(
                    "retreat_place",
                    (
                        (0.12, 0.09, 0.06, 0.04, 0.02, 0.01),
                        (0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
                    ),
                ),
            ),
            extra_metadata={"target_pose": episode_spec.place_target_pose},
        )

    def _plan(
        self,
        *,
        operation: str,
        object_prim_path: str,
        segments: tuple[dict[str, Any], ...],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ArmPlan:
        joint_trajectory = tuple(
            tuple(float(value) for value in row)
            for segment in segments
            if segment["type"] == "motion"
            for row in segment["trajectory"]["q"]
        )
        return ArmPlan(
            operation=operation,
            joint_trajectory=joint_trajectory,
            metadata={
                "planner": "segmented_smoke",
                "joint_names": self.joint_names,
                "tool_frame": "grasp_tcp_link",
                "object_prim_path": object_prim_path,
                "segments": segments,
                "world_step_owned_by_pipeline": True,
                **dict(extra_metadata or {}),
            },
        )

    def _gripper_segment(
        self,
        name: str,
        target_position: tuple[float, float],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "type": "gripper",
            "joint_names": self.gripper_joint_names,
            "target_position": target_position,
        }

    @staticmethod
    def _motion_segment(name: str, q_rows: tuple[tuple[float, ...], ...]) -> dict[str, Any]:
        return {
            "name": name,
            "type": "motion",
            "target_name": name,
            "timing": {
                "dt": 0.05,
                "duration_s": 0.05 * (len(q_rows) - 1),
                "num_waypoints": len(q_rows),
            },
            "final_error": {"position_m": 0.0, "orientation_deg": 0.0},
            "plan_info": {"planner_success": True, "smoke_plan": True},
            "trajectory": {
                "time_from_start": [0.05 * index for index in range(len(q_rows))],
                "q": [list(row) for row in q_rows],
            },
        }
