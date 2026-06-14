"""真实导航阶段的可达性验证。"""

from __future__ import annotations

import math

from source.interfaces import EpisodeSpec, SimulationState, VerificationResult


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class NavigationEpisodeVerifier:
    """验证 base 是否到达导航目标，并保留 yaw/速度诊断。"""

    def __init__(
        self,
        *,
        position_tolerance: float = 0.18,
        yaw_tolerance: float = 0.18,
        linear_velocity_tolerance: float = 0.06,
        angular_velocity_tolerance: float = 0.20,
        require_yaw_alignment: bool = False,
        require_stable_base: bool = False,
    ):
        self.position_tolerance = float(position_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.linear_velocity_tolerance = float(linear_velocity_tolerance)
        self.angular_velocity_tolerance = float(angular_velocity_tolerance)
        self.require_yaw_alignment = bool(require_yaw_alignment)
        self.require_stable_base = bool(require_stable_base)

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self._verify_goal(
            state,
            goal_x=episode_spec.pick_goal.x,
            goal_y=episode_spec.pick_goal.y,
            goal_yaw=episode_spec.pick_goal.yaw,
            failure_reason="pick_target_unreachable",
        )

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        if episode_spec.place_goal is None:
            return VerificationResult(False, "place_target_unreachable")
        return self._verify_goal(
            state,
            goal_x=episode_spec.place_goal.x,
            goal_y=episode_spec.place_goal.y,
            goal_yaw=episode_spec.place_goal.yaw,
            failure_reason="place_target_unreachable",
        )

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del state, episode_spec
        return VerificationResult(False, "grasp_not_executed")

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        del state, episode_spec
        return VerificationResult(False, "place_not_executed")

    def _verify_goal(
        self,
        state: SimulationState,
        *,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        failure_reason: str,
    ) -> VerificationResult:
        pose = state.robot_root_pose
        yaw = self._yaw_from_quaternion(pose[3:])
        distance = math.hypot(pose[0] - goal_x, pose[1] - goal_y)
        yaw_error = abs(_wrap_angle(yaw - goal_yaw))
        body_velocity = state.metadata.get("body_velocity")
        if body_velocity is None:
            linear_speed = math.hypot(
                state.robot_root_velocity[0],
                state.robot_root_velocity[1],
            )
            angular_speed = abs(state.robot_root_velocity[5])
        else:
            linear_speed = math.hypot(float(body_velocity[0]), float(body_velocity[1]))
            angular_speed = abs(float(body_velocity[2]))
        position_reached = distance <= self.position_tolerance
        yaw_aligned = yaw_error <= self.yaw_tolerance
        base_stable = (
            linear_speed <= self.linear_velocity_tolerance
            and angular_speed <= self.angular_velocity_tolerance
        )
        # 继承 video baseline 的 handoff 语义：导航只要求 base XY 进入容差。
        # yaw 后续由机械臂工作空间和抓取规划吸收，速度稳定性保留为诊断字段。
        success = (
            position_reached
            and (yaw_aligned or not self.require_yaw_alignment)
            and (base_stable or not self.require_stable_base)
        )
        return VerificationResult(
            success=success,
            failure_reason="" if success else failure_reason,
            metadata={
                "goal_distance": distance,
                "yaw_error": yaw_error,
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
                "position_tolerance": self.position_tolerance,
                "yaw_tolerance": self.yaw_tolerance,
                "linear_velocity_tolerance": self.linear_velocity_tolerance,
                "angular_velocity_tolerance": self.angular_velocity_tolerance,
                "position_reached": position_reached,
                "yaw_aligned": yaw_aligned,
                "base_stable": base_stable,
                "yaw_alignment_required": self.require_yaw_alignment,
                "base_stability_required": self.require_stable_base,
                "acceptance_mode": (
                    "xy_yaw_stable"
                    if self.require_yaw_alignment and self.require_stable_base
                    else "xy_only"
                ),
            },
        )

    @staticmethod
    def _yaw_from_quaternion(quaternion_wxyz: tuple[float, ...]) -> float:
        w, x, y, z = quaternion_wxyz
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
