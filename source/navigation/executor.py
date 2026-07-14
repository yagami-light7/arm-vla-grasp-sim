"""逐 tick 输出底盘速度命令的 DWA 导航执行器。"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from source.interfaces.navigation import NavPlan
from source.interfaces.simulation import RobotAction, SimulationState

from .adapters.frame_utils import world_velocity_to_body, wrap_yaw
from .adapters.stall_detector import NavigationStallDetector, StallDiagnostics
from .adapters.yaw_align import (
    TerminalPoseConfig,
    body_goal_components,
    compute_terminal_pose_command,
)
from .navlib import AStarPlanner, DWAConfig, DWAController, DWADebug, OccupancyGridMap


PCT_STAIR_FLOAT_DOG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)
PCT_INFEASIBLE_ROTATION_RECOVERY_HEADING_TOLERANCE = 0.45


class DwaNavExecutor:
    """跟踪 A* 路径，并按配置决定是否做终点 yaw 对齐。"""

    def __init__(
        self,
        map_json: str | Path | None = None,
        local_clearance_radius: float = 0.0,
        dwa_config: DWAConfig | None = None,
        *,
        grid_map: OccupancyGridMap | None = None,
        multifloor_grid_map: OccupancyGridMap | None = None,
        post_stair_grid_map: OccupancyGridMap | None = None,
        multifloor_protected_obstacle_map: OccupancyGridMap | None = None,
        terminal_pose_config: TerminalPoseConfig | None = None,
        terminal_start_distance: float = 0.50,
        position_tolerance: float = 0.15,
        yaw_tolerance: float = 0.15,
        stall_window_steps: int = 120,
        stall_min_progress_m: float = 0.05,
        stall_min_forward_command: float = 0.05,
        stall_min_forward_ratio: float = 0.25,
        stall_recovery_linear_speed_mps: float = 0.08,
        stall_recovery_progress_ratio: float = 0.75,
        pick_near_goal_handoff_tolerance_m: float = 0.32,
        completion_linear_velocity_tolerance: float | None = None,
        completion_angular_velocity_tolerance: float | None = None,
        require_yaw_alignment: bool = True,
        command_recompute_interval_steps: int = 1,
        multifloor_obstacle_inflate_radius: float = 0.0,
        multifloor_route_corridor_radius: float | None = None,
        carry_max_linear_velocity: float | None = None,
        carry_max_angular_velocity: float | None = None,
        carry_max_linear_accel: float | None = None,
        carry_position_tolerance: float | None = None,
        carry_path_deviation_limit: float | None = None,
        carry_initial_alignment_path_deviation_limit: float | None = None,
        carry_path_recovery_deviation_limit: float | None = None,
        carry_max_infeasible_recomputes: int | None = None,
        stair_float_enabled: bool = False,
        stair_float_speed_mps: float = 0.18,
        stair_float_activation_radius_m: float = 0.45,
        stair_float_completion_radius_m: float = 0.25,
        stair_float_min_z_delta_m: float = 0.75,
        stair_float_approach_distance_m: float = 1.80,
        stair_float_exit_distance_m: float = 0.75,
        stair_float_settle_time_s: float = 1.20,
        stair_float_release_settle_time_s: float = 0.0,
        stair_float_yaw_lookahead_m: float = 0.35,
        stair_float_min_root_z_offset_m: float = 0.18,
        stair_float_release_root_z_offset_m: float = 0.36,
    ) -> None:
        if grid_map is not None and map_json is not None:
            raise ValueError("map_json 与 grid_map 只能提供一个。")
        if local_clearance_radius < 0.0:
            raise ValueError("local_clearance_radius 不能为负数。")
        if terminal_start_distance <= 0.0:
            raise ValueError("terminal_start_distance 必须为正数。")
        if position_tolerance < 0.0 or yaw_tolerance < 0.0:
            raise ValueError("位置和朝向容差不能为负数。")
        if command_recompute_interval_steps < 1:
            raise ValueError("DWA 命令重算间隔必须至少为 1 个物理步。")
        if pick_near_goal_handoff_tolerance_m < 0.0:
            raise ValueError("抓取前近目标 handoff 容差不能为负数。")
        if stall_recovery_linear_speed_mps < 0.0:
            raise ValueError("stall 恢复速度阈值不能为负数。")
        if not 0.0 <= stall_recovery_progress_ratio <= 1.0:
            raise ValueError("stall 恢复进度比例必须位于 0 到 1。")
        if multifloor_obstacle_inflate_radius < 0.0:
            raise ValueError("跨楼层障碍膨胀半径不能为负数。")
        if (
            multifloor_route_corridor_radius is not None
            and multifloor_route_corridor_radius < 0.0
        ):
            raise ValueError("跨楼层路径走廊半径不能为负数。")
        if (
            carry_max_infeasible_recomputes is not None
            and carry_max_infeasible_recomputes < 1
        ):
            raise ValueError("携物导航连续无可行轨迹上限必须至少为 1。")
        if carry_position_tolerance is not None and carry_position_tolerance <= 0.0:
            raise ValueError("携物导航位置容差必须为正数。")
        if (
            carry_initial_alignment_path_deviation_limit is not None
            and carry_initial_alignment_path_deviation_limit <= 0.0
        ):
            raise ValueError("携物导航初始转向路径恢复范围必须为正数。")
        if (
            carry_path_recovery_deviation_limit is not None
            and carry_path_recovery_deviation_limit <= 0.0
        ):
            raise ValueError("携物导航普通转弯路径恢复范围必须为正数。")
        if stair_float_speed_mps <= 0.0:
            raise ValueError("楼梯漂移速度必须为正数。")
        if stair_float_activation_radius_m < 0.0:
            raise ValueError("楼梯漂移触发半径不能为负数。")
        if stair_float_completion_radius_m < 0.0:
            raise ValueError("楼梯漂移完成半径不能为负数。")
        if stair_float_min_z_delta_m <= 0.0:
            raise ValueError("楼梯漂移最小跨层高度必须为正数。")
        if stair_float_approach_distance_m < 0.0:
            raise ValueError("楼梯漂移入口前扩展距离不能为负数。")
        if stair_float_exit_distance_m < 0.0:
            raise ValueError("楼梯漂移出口后扩展距离不能为负数。")
        if stair_float_settle_time_s < 0.0:
            raise ValueError("楼梯漂移结束稳定时间不能为负数。")
        if stair_float_release_settle_time_s < 0.0:
            raise ValueError("楼梯漂移解冻稳定时间不能为负数。")
        if stair_float_yaw_lookahead_m < 0.0:
            raise ValueError("楼梯漂移朝向前瞻距离不能为负数。")
        if stair_float_min_root_z_offset_m < 0.0:
            raise ValueError("楼梯漂移 root 最小离地高度不能为负数。")
        if stair_float_release_root_z_offset_m < 0.0:
            raise ValueError("楼梯漂移释放 root 离地高度不能为负数。")

        self.map_json = None if map_json is None else str(Path(map_json).expanduser().resolve())
        self._single_floor_raw_map = grid_map
        self._multifloor_raw_map = multifloor_grid_map
        self._post_stair_raw_map = post_stair_grid_map
        self._multifloor_protected_obstacle_map = (
            multifloor_protected_obstacle_map
        )
        self._raw_map = grid_map
        self.local_clearance_radius = float(local_clearance_radius)
        self.local_map = (
            None
            if self._raw_map is None
            else self._raw_map.inflate(self.local_clearance_radius)
        )
        self.dwa_config = dwa_config or DWAConfig(control_dt=0.05)
        self._active_dwa_config = self.dwa_config
        self.multifloor_obstacle_inflate_radius = float(
            multifloor_obstacle_inflate_radius
        )
        self.multifloor_route_corridor_radius = multifloor_route_corridor_radius
        self.carry_max_linear_velocity = carry_max_linear_velocity
        self.carry_max_angular_velocity = carry_max_angular_velocity
        self.carry_max_linear_accel = carry_max_linear_accel
        self.carry_position_tolerance = (
            None
            if carry_position_tolerance is None
            else float(carry_position_tolerance)
        )
        self.carry_path_deviation_limit = carry_path_deviation_limit
        self.carry_initial_alignment_path_deviation_limit = (
            carry_initial_alignment_path_deviation_limit
        )
        self.carry_path_recovery_deviation_limit = (
            carry_path_recovery_deviation_limit
        )
        self.carry_max_infeasible_recomputes = carry_max_infeasible_recomputes
        self.stair_float_enabled = bool(stair_float_enabled)
        self.stair_float_speed_mps = float(stair_float_speed_mps)
        self.stair_float_activation_radius_m = float(
            stair_float_activation_radius_m
        )
        self.stair_float_completion_radius_m = float(
            stair_float_completion_radius_m
        )
        self.stair_float_min_z_delta_m = float(stair_float_min_z_delta_m)
        self.stair_float_approach_distance_m = float(
            stair_float_approach_distance_m
        )
        self.stair_float_exit_distance_m = float(stair_float_exit_distance_m)
        self.stair_float_settle_time_s = float(stair_float_settle_time_s)
        self.stair_float_release_settle_time_s = float(
            stair_float_release_settle_time_s
        )
        self.stair_float_yaw_lookahead_m = float(stair_float_yaw_lookahead_m)
        self.stair_float_min_root_z_offset_m = float(
            stair_float_min_root_z_offset_m
        )
        self.stair_float_release_root_z_offset_m = float(
            stair_float_release_root_z_offset_m
        )
        self.terminal_start_distance = max(
            float(terminal_start_distance),
            float(self.dwa_config.goal_tolerance),
        )
        self.position_tolerance = float(position_tolerance)
        self._active_position_tolerance = self.position_tolerance
        self.yaw_tolerance = float(yaw_tolerance)
        self.terminal_pose_config = terminal_pose_config or TerminalPoseConfig(
            position_tolerance=min(0.08, self.position_tolerance),
            position_acceptance_tolerance=self.position_tolerance,
            yaw_tolerance=self.yaw_tolerance,
        )
        self.stall_detector = NavigationStallDetector(
            window_steps=int(stall_window_steps),
            min_progress_m=float(stall_min_progress_m),
            min_forward_command=float(stall_min_forward_command),
            min_forward_ratio=float(stall_min_forward_ratio),
        )
        self.stall_recovery_linear_speed_mps = float(
            stall_recovery_linear_speed_mps
        )
        self.stall_recovery_progress_ratio = float(
            stall_recovery_progress_ratio
        )
        self.pick_near_goal_handoff_tolerance_m = float(
            pick_near_goal_handoff_tolerance_m
        )
        self.completion_linear_velocity_tolerance = completion_linear_velocity_tolerance
        self.completion_angular_velocity_tolerance = completion_angular_velocity_tolerance
        self.require_yaw_alignment = bool(require_yaw_alignment)
        self._active_require_yaw_alignment = self.require_yaw_alignment
        self._active_yaw_tolerance = self.yaw_tolerance
        self._active_terminal_pose_config = self.terminal_pose_config
        self.command_recompute_interval_steps = int(command_recompute_interval_steps)

        self.plan: NavPlan | None = None
        self._controller: DWAController | None = None
        self._phase = "idle"
        self._tick_index = 0
        self._done = False
        self._success = False
        self._failure_reason = ""
        self._stall_detected = False
        self._stall_recovery_count = 0
        self._stall_diagnostics = self.stall_detector.diagnostics()
        self._last_command = (0.0, 0.0, 0.0)
        self._last_pose: tuple[float, float, float] | None = None
        self._last_body_velocity = (0.0, 0.0, 0.0)
        self._distance_to_goal: float | None = None
        self._yaw_error: float | None = None
        self._terminal_translation_heading_error: float | None = None
        self._terminal_control_mode: str | None = None
        self._carry_forward_translation_active = False
        self._carry_forward_translation_activation_reason: str | None = None
        self._last_dwa_debug: DWADebug | None = None
        self._dwa_compute_count = 0
        self._dwa_hold_count = 0
        self._last_dwa_compute_duration_s = 0.0
        self._max_dwa_compute_duration_s = 0.0
        self._command_recomputed_this_tick = False
        self._local_refinement_report: dict[str, Any] | None = None
        self._map_selection_report: dict[str, Any] | None = None
        self._carry_mode = False
        self._consecutive_infeasible_recomputes = 0
        self._stair_float_path: tuple[tuple[float, float, float], ...] = ()
        self._stair_float_segment_lengths: tuple[float, ...] = ()
        self._stair_float_total_length = 0.0
        self._stair_float_progress = 0.0
        self._stair_float_active = False
        self._stair_float_done = False
        self._stair_float_started = False
        self._stair_float_last_timestamp: float | None = None
        self._stair_float_root_z_offset = 0.0
        self._stair_float_initial_root_z_offset = 0.0
        self._stair_float_target_root_z_offset = 0.0
        self._stair_float_settling = False
        self._stair_float_settle_remaining_s = 0.0
        self._stair_float_release_settling = False
        self._stair_float_release_settle_remaining_s = 0.0
        self._stair_float_hold_xyzyaw: tuple[float, float, float, float] | None = None
        self._stair_float_report: dict[str, Any] = {"enabled": False}
        self._near_goal_stall_handoff = False
        self._near_goal_stall_handoff_tolerance: float | None = None

    def reset(self, plan: NavPlan) -> None:
        """载入新路径并清空执行、终点和停滞状态。"""

        if len(plan.waypoints) < 2:
            raise ValueError("DWA 路径至少需要两个 waypoint。")
        self._reset_stair_float_state(plan)
        self._select_local_map(plan)
        self._ensure_local_map(plan)
        if self.local_map is None:
            raise RuntimeError("导航执行器没有可用的占用栅格地图。")

        path_world = [
            (float(point[0]), float(point[1]))
            for point in plan.waypoints
        ]
        path_world = self._refine_pct_path_for_local_map(plan, path_world)
        path_world, stair_float_splice = _splice_stair_float_controller_path(
            path_world,
            self._stair_float_path,
        )
        self._stair_float_report["controller_path_splice"] = stair_float_splice
        self._active_dwa_config = self._dwa_config_for_plan(plan)
        self._carry_mode = (
            plan.metadata.get("execution_phase") == "carry_nav_to_place"
        )
        self._active_position_tolerance = (
            float(self.carry_position_tolerance)
            if self._carry_mode and self.carry_position_tolerance is not None
            else self.position_tolerance
        )
        self._active_require_yaw_alignment = bool(
            plan.metadata.get("require_yaw_alignment", self.require_yaw_alignment)
        )
        self._active_yaw_tolerance = float(
            plan.metadata.get("yaw_tolerance", self.yaw_tolerance)
        )
        self._active_terminal_pose_config = replace(
            self.terminal_pose_config,
            yaw_tolerance=self._active_yaw_tolerance,
            prefer_forward_translation=False,
        )
        if self._carry_mode:
            carry_linear_limit = float(self._active_dwa_config.max_linear_velocity)
            self._active_terminal_pose_config = replace(
                self._active_terminal_pose_config,
                max_vx=min(
                    float(self._active_terminal_pose_config.max_vx),
                    carry_linear_limit,
                ),
                max_vy=min(
                    float(self._active_terminal_pose_config.max_vy),
                    carry_linear_limit,
                ),
            )
        self.plan = plan
        self._controller = DWAController(
            path_world=path_world,
            grid_map=self.local_map,
            config=self._active_dwa_config,
        )
        self.stall_detector.reset()
        self._phase = "dwa"
        self._tick_index = 0
        self._done = False
        self._success = False
        self._failure_reason = ""
        self._stall_detected = False
        self._stall_recovery_count = 0
        self._stall_diagnostics = self.stall_detector.diagnostics()
        self._last_command = (0.0, 0.0, 0.0)
        self._last_pose = None
        self._last_body_velocity = (0.0, 0.0, 0.0)
        self._distance_to_goal = None
        self._yaw_error = None
        self._terminal_translation_heading_error = None
        self._terminal_control_mode = None
        self._carry_forward_translation_active = False
        self._carry_forward_translation_activation_reason = None
        self._last_dwa_debug = None
        self._dwa_compute_count = 0
        self._dwa_hold_count = 0
        self._last_dwa_compute_duration_s = 0.0
        self._max_dwa_compute_duration_s = 0.0
        self._command_recomputed_this_tick = False
        self._consecutive_infeasible_recomputes = 0
        self._near_goal_stall_handoff = False
        self._near_goal_stall_handoff_tolerance = None

    def compute_action(self, state: SimulationState) -> RobotAction:
        """根据当前观测计算一个 tick 的 RobotAction，不推进仿真。"""

        self._require_plan()
        if self.is_done(state):
            return self._zero_action()
        if self._phase == "settling":
            # 到达位姿但实测速度尚未衰减时必须持续发零命令；旧逻辑会在
            # 同一 tick 重新进入 terminal controller，把机器人再次推出容差。
            self._last_command = (0.0, 0.0, 0.0)
            return RobotAction(
                base_velocity=self._last_command,
                source="navigation_settling",
                metadata=self._action_metadata(),
            )

        pose = self._pose_xyyaw(state)
        body_velocity = self._body_velocity(state, pose[2])
        distance, yaw_error = self._goal_errors(pose)
        stair_float_action = self._compute_stair_float_action(state, pose)
        if stair_float_action is not None:
            return stair_float_action
        terminal_position_tolerance = float(
            self._active_terminal_pose_config.position_tolerance
        )
        if (
            self._carry_mode
            and not self._carry_forward_translation_active
            and terminal_position_tolerance < distance <= self.terminal_start_distance
            and abs(yaw_error) <= self._active_yaw_tolerance
        ):
            # 保留已验证的大角度 holonomic 对准；只在最终 yaw 已合格、但
            # XY 仍卡在验收半径外时，切换到“朝位置误差转向并前进”。
            self._carry_forward_translation_active = True
            self._carry_forward_translation_activation_reason = (
                "final_yaw_aligned_with_xy_residual"
            )
        elif (
            self._carry_forward_translation_active
            and distance <= terminal_position_tolerance
        ):
            # 位置收敛后交回普通 terminal controller 做最终 yaw polish。
            self._carry_forward_translation_active = False
        next_phase = (
            "terminal_pose"
            if self._active_require_yaw_alignment
            and (
                distance <= self.terminal_start_distance
                or self._carry_forward_translation_active
            )
            else "dwa"
        )
        if next_phase != self._phase:
            self.stall_detector.reset()
            self._stall_diagnostics = self.stall_detector.diagnostics()
        self._phase = next_phase

        if self._phase == "terminal_pose":
            body_goal_x, body_goal_y = body_goal_components(
                pose,
                (self.plan.goal.x, self.plan.goal.y),
            )
            self._terminal_translation_heading_error = math.atan2(
                body_goal_y,
                body_goal_x,
            )
            self._terminal_control_mode = (
                "carry_forward_translation"
                if self._carry_forward_translation_active
                else "final_pose"
            )
            terminal_pose_config = replace(
                self._active_terminal_pose_config,
                prefer_forward_translation=(
                    self._carry_forward_translation_active
                ),
            )
            command = compute_terminal_pose_command(
                body_goal_x=body_goal_x,
                body_goal_y=body_goal_y,
                yaw_error=yaw_error,
                distance_to_goal=distance,
                config=terminal_pose_config,
            )
            self._last_dwa_debug = None
            self._command_recomputed_this_tick = True
        else:
            self._terminal_translation_heading_error = None
            self._terminal_control_mode = None
            if self._controller is None:
                raise RuntimeError("DWA 控制器尚未初始化。")
            should_recompute = (
                self._last_dwa_debug is None
                or self._tick_index % self.command_recompute_interval_steps == 0
            )
            self._command_recomputed_this_tick = should_recompute
            if should_recompute:
                started_at = time.perf_counter()
                raw_command, self._last_dwa_debug = self._controller.compute_command(
                    pose,
                    (body_velocity[0], body_velocity[2]),
                )
                elapsed = time.perf_counter() - started_at
                self._last_dwa_compute_duration_s = elapsed
                self._max_dwa_compute_duration_s = max(
                    self._max_dwa_compute_duration_s,
                    elapsed,
                )
                self._dwa_compute_count += 1
                self._update_infeasible_recomputes()
                command = (
                    float(raw_command[0]),
                    float(raw_command[1]),
                    float(raw_command[2]),
                )
            else:
                # 中间物理步保持上一条命令，避免 DWA 计算阻塞每一次 world.step。
                self._dwa_hold_count += 1
                command = self._last_command

        self._tick_index += 1
        self._last_command = tuple(float(value) for value in command)
        if self._phase == "dwa" and not self._stall_detected:
            self._update_stall(pose, body_velocity, self._last_command)
        if self._done and self._success:
            self._last_command = (0.0, 0.0, 0.0)
            return self._zero_action()
        if self._stall_detected:
            self._last_command = (0.0, 0.0, 0.0)
            return self._zero_action()

        return RobotAction(
            base_velocity=self._last_command,
            source=(
                "navigation_terminal_pose"
                if self._phase == "terminal_pose"
                else "navigation_dwa"
            ),
            metadata=self._action_metadata(),
        )

    def is_done(self, state: SimulationState) -> bool:
        """XY 到达或检测到 stall 时结束；yaw 只在严格模式中参与判定。"""

        if self.plan is None:
            return False
        if self._stall_detected:
            self._done = True
            return True

        pose = self._pose_xyyaw(state)
        body_velocity = self._body_velocity(state, pose[2])
        distance, yaw_error = self._goal_errors(pose)
        yaw_ok = (
            not self._active_require_yaw_alignment
        ) or abs(yaw_error) <= self._active_yaw_tolerance
        if distance <= self._active_position_tolerance and yaw_ok:
            if not self._completion_velocity_is_stable(body_velocity):
                # 严格诊断模式才等待速度衰减；默认导航 handoff 只看 XY。
                self._done = False
                self._success = False
                self._failure_reason = ""
                self._phase = "settling"
                self._last_command = (0.0, 0.0, 0.0)
                return False
            self._done = True
            self._success = True
            self._failure_reason = ""
            self._phase = "completed"
            self._last_command = (0.0, 0.0, 0.0)
        return self._done

    def status(self) -> dict[str, Any]:
        """返回可直接写入 episode summary 的结构化执行状态。"""

        goal = None
        if self.plan is not None:
            goal = (
                float(self.plan.goal.x),
                float(self.plan.goal.y),
                float(self.plan.goal.yaw),
            )
        return {
            "backend": "astar_dwa_tick",
            "phase": self._phase,
            "tick_index": self._tick_index,
            "done": self._done,
            "success": self._success,
            "failed": bool(self._failure_reason),
            "failure_reason": self._failure_reason,
            "goal": goal,
            "pose_xyyaw": self._last_pose,
            "distance_to_goal": self._distance_to_goal,
            "yaw_error": self._yaw_error,
            "terminal_translation_heading_error": (
                self._terminal_translation_heading_error
            ),
            "terminal_control_mode": self._terminal_control_mode,
            "carry_forward_translation_active": (
                self._carry_forward_translation_active
            ),
            "carry_forward_translation_activation_reason": (
                self._carry_forward_translation_activation_reason
            ),
            "position_tolerance": self._active_position_tolerance,
            "configured_position_tolerance": self.position_tolerance,
            "carry_position_tolerance": self.carry_position_tolerance,
            "yaw_tolerance": self._active_yaw_tolerance,
            "yaw_alignment_required": self._active_require_yaw_alignment,
            "acceptance_mode": (
                "xy_yaw"
                if self._active_require_yaw_alignment
                else "xy_only"
            ),
            "completion_linear_velocity_tolerance": self.completion_linear_velocity_tolerance,
            "completion_angular_velocity_tolerance": self.completion_angular_velocity_tolerance,
            "last_command": self._last_command,
            "measured_body_velocity": self._last_body_velocity,
            "stall_detected": self._stall_detected,
            "stall": self._stall_status(self._stall_diagnostics),
            "stall_recovery_count": self._stall_recovery_count,
            "dwa_compute": self._dwa_compute_status(),
            "dwa_limits": self._active_dwa_limits(),
            "local_refinement": self._local_refinement_report,
            "map_selection": self._map_selection_report,
            "near_goal_stall_handoff": self._near_goal_stall_handoff,
            "near_goal_stall_handoff_tolerance": (
                self._near_goal_stall_handoff_tolerance
            ),
            "stair_float": self._stair_float_status(),
            "dwa": (
                None
                if self._last_dwa_debug is None
                else asdict(self._last_dwa_debug)
            ),
        }

    def _ensure_local_map(self, plan: NavPlan) -> None:
        """允许执行器从 planner metadata 延迟加载同一张地图。"""

        if self.local_map is not None:
            return
        map_json = self.map_json or plan.metadata.get("map_json")
        if not map_json:
            raise ValueError("执行器需要 map_json、grid_map 或 plan.metadata['map_json']。")
        self.map_json = str(Path(str(map_json)).expanduser().resolve())
        self._raw_map = OccupancyGridMap.from_meta_file(self.map_json)
        self._single_floor_raw_map = self._raw_map
        self.local_map = self._raw_map.inflate(self.local_clearance_radius)

    def _refine_pct_path_for_local_map(
        self,
        plan: NavPlan,
        path_world: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """用 DWA 本地障碍地图细化 PCT 稀疏路径。"""

        self._local_refinement_report = None
        if plan.metadata.get("planner") != "pct" or self.local_map is None:
            return path_world
        if len(path_world) < 2:
            return path_world
        if _pct_plan_is_multifloor(plan):
            refined = _remove_consecutive_duplicate_waypoints(path_world)
            exact_goal = (float(plan.goal.x), float(plan.goal.y))
            if math.hypot(
                refined[-1][0] - exact_goal[0],
                refined[-1][1] - exact_goal[1],
            ) > 1.0e-3:
                refined.append(exact_goal)
            self._local_refinement_report = {
                "enabled": True,
                "success": True,
                "mode": "pct_multifloor_path_preserved",
                "input_waypoints": len(path_world),
                "output_waypoints": len(refined),
                "multifloor": True,
                "slice_start": plan.metadata.get("slice_start"),
                "slice_end": plan.metadata.get("slice_end"),
            }
            return refined
        exact_goal = (float(plan.goal.x), float(plan.goal.y))
        sim_start = plan.metadata.get("sim_start")
        direct_start = (
            (float(sim_start[0]), float(sim_start[1]))
            if isinstance(sim_start, (list, tuple)) and len(sim_start) >= 2
            else path_world[0]
        )
        direct_free, direct_clearance = _world_segment_clearance(
            direct_start,
            exact_goal,
            self.local_map,
        )
        direct_clearance_required = max(
            0.40,
            2.0 * float(self.local_map.resolution),
        )
        if (
            direct_free
            and direct_clearance is not None
            and direct_clearance >= direct_clearance_required
        ):
            self._local_refinement_report = {
                "enabled": True,
                "success": True,
                "mode": "pct_same_floor_direct",
                "input_waypoints": len(path_world),
                "output_waypoints": 2,
                "direct_clearance_m": direct_clearance,
                "direct_clearance_required_m": direct_clearance_required,
            }
            return [direct_start, exact_goal]
        try:
            result = AStarPlanner().plan(
                self.local_map,
                path_world[0],
                (float(plan.goal.x), float(plan.goal.y)),
                snap_to_free=True,
                max_snap_distance_m=max(0.5, self.local_clearance_radius + 0.5),
            )
        except Exception as exc:
            self._local_refinement_report = {
                "enabled": True,
                "success": False,
                "reason": str(exc),
                "input_waypoints": len(path_world),
            }
            return path_world

        refined = [
            (float(point[0]), float(point[1]))
            for point in result.path_world
        ]
        exact_goal = (float(plan.goal.x), float(plan.goal.y))
        if math.hypot(refined[-1][0] - exact_goal[0], refined[-1][1] - exact_goal[1]) > 1.0e-3:
            refined.append(exact_goal)
        if len(refined) < 2:
            self._local_refinement_report = {
                "enabled": True,
                "success": False,
                "reason": "refined_path_too_short",
                "input_waypoints": len(path_world),
            }
            return path_world
        self._local_refinement_report = {
            "enabled": True,
            "success": True,
            "input_waypoints": len(path_world),
            "output_waypoints": len(refined),
            "raw_grid_waypoints": len(result.raw_path_grid),
            "collinear_waypoints": len(result.path_world),
            "expanded_nodes": int(result.expanded_nodes),
            "cost": float(result.cost),
        }
        return refined

    def _select_local_map(self, plan: NavPlan) -> None:
        """按 PCT 路径是否跨楼层选择对应的二维避障投影。"""

        selected = self._single_floor_raw_map
        multifloor = _pct_plan_is_multifloor(plan)
        self._map_selection_report = {
            "multifloor": multifloor,
            "obstacle_inflate_radius_m": 0.0,
            "route_corridor_radius_m": None,
            "route_cells_cleared": 0,
            "stair_float_route_cells_cleared": 0,
            "protected_cells_preserved": 0,
        }
        if multifloor and self._multifloor_raw_map is not None:
            selected = self._multifloor_raw_map
        if selected is None:
            self._raw_map = None
            self.local_map = None
            return
        if multifloor:
            selected = selected.inflate(self.multifloor_obstacle_inflate_radius)
            self._map_selection_report["obstacle_inflate_radius_m"] = (
                self.multifloor_obstacle_inflate_radius
            )
            if self.multifloor_route_corridor_radius is not None:
                route_paths: list[
                    tuple[str, list[tuple[float, float]], float]
                ] = [
                    (
                        "full_path",
                        [
                            (float(point[0]), float(point[1]))
                            for point in plan.waypoints
                        ],
                        float(self.multifloor_route_corridor_radius),
                    )
                ]
                if not self.stair_float_enabled:
                    route_paths = _pct_no_float_route_corridors(
                        plan,
                        flat_radius_m=float(
                            self.multifloor_route_corridor_radius
                        ),
                        stair_radius_m=min(
                            float(self.multifloor_route_corridor_radius),
                            0.30,
                        ),
                    )
                if self._stair_float_path:
                    stair_route, _ = _splice_stair_float_controller_path(
                        [
                            (float(point[0]), float(point[1]))
                            for point in plan.waypoints
                        ],
                        self._stair_float_path,
                    )
                    route_paths.append(
                        (
                            "stair_float_path",
                            stair_route,
                            float(self.multifloor_route_corridor_radius),
                        )
                    )
                cleared_count = 0
                protected_count = 0
                stair_float_cleared_count = 0
                corridor_reports: list[dict[str, Any]] = []
                for route_name, route_path, route_radius in route_paths:
                    selected, current_cleared, current_protected = (
                        _clear_grid_path_corridor(
                            selected,
                            route_path,
                            radius_m=route_radius,
                            protected_obstacle_map=(
                                self._multifloor_protected_obstacle_map
                            ),
                        )
                    )
                    cleared_count += int(current_cleared)
                    protected_count += int(current_protected)
                    if route_name == "stair_float_path":
                        stair_float_cleared_count += int(current_cleared)
                    corridor_reports.append(
                        {
                            "name": route_name,
                            "radius_m": float(route_radius),
                            "waypoint_count": len(route_path),
                            "cleared_cells": int(current_cleared),
                            "protected_cells": int(current_protected),
                        }
                    )
                self._map_selection_report.update(
                    {
                        "route_corridor_radius_m": float(
                            self.multifloor_route_corridor_radius
                        ),
                        "route_cells_cleared": cleared_count,
                        "stair_float_route_cells_cleared": (
                            stair_float_cleared_count
                        ),
                        "protected_cells_preserved": protected_count,
                        "route_corridors": corridor_reports,
                    }
                )
        self._raw_map = selected
        self.local_map = selected.inflate(self.local_clearance_radius)

    def _dwa_config_for_plan(self, plan: NavPlan) -> DWAConfig:
        """携物导航使用更保守的速度和路径偏离上限。"""

        if plan.metadata.get("execution_phase") != "carry_nav_to_place":
            return self.dwa_config
        updates: dict[str, Any] = {
            # 携物通过窄门时必须停在锐角点并先完成朝向切换，禁止切内角。
            "preserve_sharp_corners": True,
            "corner_angle_threshold": 0.35,
            "corner_waypoint_tolerance": 0.18,
            # 携物导航仍允许大角度原地对齐，但普通门口/房间转角应优先走平滑弧线。
            "rotate_in_place_angle": max(
                1.60,
                float(self.dwa_config.rotate_in_place_angle),
            ),
            "enforce_path_deviation_limit": True,
        }
        if self.carry_max_linear_velocity is not None:
            updates["max_linear_velocity"] = min(
                self.dwa_config.max_linear_velocity,
                float(self.carry_max_linear_velocity),
            )
            updates["min_active_linear_velocity"] = min(
                self.dwa_config.min_active_linear_velocity,
                float(self.carry_max_linear_velocity),
            )
            updates["near_goal_min_active_linear_velocity"] = min(
                self.dwa_config.near_goal_min_active_linear_velocity,
                float(self.carry_max_linear_velocity),
            )
            updates["close_goal_speed_limit"] = min(
                self.dwa_config.close_goal_speed_limit,
                float(self.carry_max_linear_velocity),
            )
        if self.carry_max_angular_velocity is not None:
            updates["max_angular_velocity"] = min(
                self.dwa_config.max_angular_velocity,
                float(self.carry_max_angular_velocity),
            )
        if self.carry_max_linear_accel is not None:
            updates["max_linear_accel"] = min(
                self.dwa_config.max_linear_accel,
                float(self.carry_max_linear_accel),
            )
        if self.carry_path_deviation_limit is not None:
            updates["path_deviation_limit"] = min(
                self.dwa_config.path_deviation_limit,
                float(self.carry_path_deviation_limit),
            )
        if self.carry_initial_alignment_path_deviation_limit is not None:
            updates["initial_alignment_path_deviation_limit"] = max(
                float(updates.get(
                    "path_deviation_limit",
                    self.dwa_config.path_deviation_limit,
                )),
                float(self.carry_initial_alignment_path_deviation_limit),
            )
        if self.carry_path_recovery_deviation_limit is not None:
            updates["path_recovery_deviation_limit"] = max(
                float(updates.get(
                    "path_deviation_limit",
                    self.dwa_config.path_deviation_limit,
                )),
                float(self.carry_path_recovery_deviation_limit),
            )
            updates["near_goal_path_deviation_limit"] = max(
                float(updates["path_recovery_deviation_limit"]),
                0.75,
            )
            updates["near_goal_path_deviation_distance"] = 0.65
        if _pct_plan_is_multifloor(plan) and not self.stair_float_enabled:
            path_limit = float(
                updates.get(
                    "path_deviation_limit",
                    self.dwa_config.path_deviation_limit,
                )
            )
            updates["initial_alignment_path_deviation_limit"] = max(
                path_limit,
                min(
                    float(
                        updates.get(
                            "initial_alignment_path_deviation_limit",
                            path_limit,
                        )
                    ),
                    0.30,
                ),
            )
            updates["path_recovery_deviation_limit"] = max(
                path_limit,
                min(
                    float(
                        updates.get(
                            "path_recovery_deviation_limit",
                            path_limit,
                        )
                    ),
                    0.30,
                ),
            )
        return replace(self.dwa_config, **updates)

    def _update_infeasible_recomputes(self) -> None:
        """连续无可行轨迹时及时停止，避免接触后被墙体持续推出。"""

        debug = self._last_dwa_debug
        limit = self.carry_max_infeasible_recomputes
        if not self._carry_mode or limit is None or debug is None:
            self._consecutive_infeasible_recomputes = 0
            return
        if debug.feasible_candidates > 0:
            self._consecutive_infeasible_recomputes = 0
            return
        if (
            abs(float(debug.heading_error))
            > PCT_INFEASIBLE_ROTATION_RECOVERY_HEADING_TOLERANCE
            and abs(float(debug.best_angular_velocity)) > 1.0e-6
            and abs(float(debug.best_linear_velocity)) <= 1.0e-6
        ):
            # 入口侧偏时 DWA 会先停止前进并原地朝路径回正；这不是持续碰撞命令，
            # 不能用数个重算周期提前终止，否则机器人没有足够时间完成大角度转向。
            self._consecutive_infeasible_recomputes = 0
            return
        self._consecutive_infeasible_recomputes += 1
        if self._consecutive_infeasible_recomputes < int(limit):
            return
        self._stall_detected = True
        self._done = True
        self._success = False
        self._failure_reason = (
            "nav_path_deviation"
            if debug.path_deviation_rejections > debug.collision_rejections
            else "nav_collision"
        )
        self._phase = "stalled"

    def _require_plan(self) -> None:
        if self.plan is None:
            raise RuntimeError("导航执行器尚未 reset(plan)。")

    def _pose_xyyaw(self, state: SimulationState) -> tuple[float, float, float]:
        pose = state.robot_root_pose
        yaw = self._yaw_from_wxyz(pose[3], pose[4], pose[5], pose[6])
        result = (float(pose[0]), float(pose[1]), yaw)
        self._last_pose = result
        return result

    @staticmethod
    def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
        """从项目统一使用的 wxyz 四元数提取平面 yaw。"""

        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm <= 1.0e-12:
            raise ValueError("robot_root_pose 中的四元数范数为零。")
        w, x, y, z = (
            float(w) / norm,
            float(x) / norm,
            float(y) / norm,
            float(z) / norm,
        )
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def _body_velocity(
        self,
        state: SimulationState,
        yaw: float,
    ) -> tuple[float, float, float]:
        velocity = state.robot_root_velocity
        vx_body, vy_body = world_velocity_to_body(
            float(velocity[0]),
            float(velocity[1]),
            yaw,
        )
        result = (vx_body, vy_body, float(velocity[5]))
        self._last_body_velocity = result
        return result

    def _goal_errors(
        self,
        pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        self._require_plan()
        distance = math.hypot(
            float(self.plan.goal.x) - pose[0],
            float(self.plan.goal.y) - pose[1],
        )
        yaw_error = wrap_yaw(float(self.plan.goal.yaw) - pose[2])
        self._distance_to_goal = distance
        self._yaw_error = yaw_error
        return distance, yaw_error

    def _completion_velocity_is_stable(
        self,
        body_velocity: tuple[float, float, float],
    ) -> bool:
        """可选地把“到达目标”收紧为真实底盘速度也已稳定。"""

        if self.completion_linear_velocity_tolerance is not None:
            linear_speed = math.hypot(body_velocity[0], body_velocity[1])
            if linear_speed > float(self.completion_linear_velocity_tolerance):
                return False
        if self.completion_angular_velocity_tolerance is not None:
            if abs(body_velocity[2]) > float(self.completion_angular_velocity_tolerance):
                return False
        return True

    def _update_stall(
        self,
        pose: tuple[float, float, float],
        body_velocity: tuple[float, float, float],
        command: tuple[float, float, float],
    ) -> None:
        stalled, diagnostics = self.stall_detector.update(
            pose[0],
            pose[1],
            command[0],
        )
        self._stall_diagnostics = diagnostics
        if not stalled:
            return
        recovery_speed = math.hypot(body_velocity[0], body_velocity[1])
        recovery_progress = (
            float(self.stall_detector.min_progress_m)
            * self.stall_recovery_progress_ratio
        )
        if (
            self._carry_mode
            and diagnostics.max_displacement_m >= recovery_progress
            and recovery_speed >= self.stall_recovery_linear_speed_mps
        ):
            # 机器人刚从接触中恢复实际位移时重开窗口，避免旧停滞样本误杀脱困动作。
            self.stall_detector.reset()
            self._stall_diagnostics = self.stall_detector.diagnostics()
            self._stall_recovery_count += 1
            return
        if self._accept_near_goal_pick_handoff():
            self._done = True
            self._success = True
            self._failure_reason = ""
            self._phase = "completed_near_goal_stall"
            self._last_command = (0.0, 0.0, 0.0)
            return
        self._stall_detected = True
        self._done = True
        self._success = False
        self._failure_reason = "nav_collision"
        self._phase = "stalled"

    def _accept_near_goal_pick_handoff(self) -> bool:
        """允许 nav_to_pick 在近目标物理停滞时交给抓取规划继续验证。"""

        if self.plan is None:
            return False
        if self.plan.metadata.get("execution_phase") != "nav_to_pick":
            return False
        if self._active_require_yaw_alignment:
            return False
        if self._distance_to_goal is None:
            return False
        tolerance = max(
            float(self._active_position_tolerance),
            float(self.pick_near_goal_handoff_tolerance_m),
        )
        self._near_goal_stall_handoff_tolerance = tolerance
        if float(self._distance_to_goal) > tolerance:
            return False
        debug = self._last_dwa_debug
        if debug is not None and debug.feasible_candidates <= 0:
            return False
        self._near_goal_stall_handoff = True
        return True

    def _zero_action(self) -> RobotAction:
        source = (
            "navigation_stalled"
            if self._stall_detected
            else "navigation_completed"
        )
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source=source,
            metadata=self._action_metadata(),
        )

    def _action_metadata(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "tick_index": self._tick_index,
            "distance_to_goal": self._distance_to_goal,
            "yaw_error": self._yaw_error,
            "terminal_translation_heading_error": (
                self._terminal_translation_heading_error
            ),
            "terminal_control_mode": self._terminal_control_mode,
            "carry_forward_translation_active": (
                self._carry_forward_translation_active
            ),
            "carry_forward_translation_activation_reason": (
                self._carry_forward_translation_activation_reason
            ),
            "measured_body_velocity": self._last_body_velocity,
            "yaw_alignment_required": self._active_require_yaw_alignment,
            "stall_detected": self._stall_detected,
            "failed": bool(self._failure_reason),
            "failure_reason": self._failure_reason,
            "stall": self._stall_status(self._stall_diagnostics),
            "dwa_compute": self._dwa_compute_status(),
            "dwa_limits": self._active_dwa_limits(),
            "local_refinement": self._local_refinement_report,
            "map_selection": self._map_selection_report,
            "near_goal_stall_handoff": self._near_goal_stall_handoff,
            "near_goal_stall_handoff_tolerance": (
                self._near_goal_stall_handoff_tolerance
            ),
            "stair_float": self._stair_float_status(),
            "dwa": (
                None
                if self._last_dwa_debug is None
                else asdict(self._last_dwa_debug)
            ),
        }

    def _dwa_compute_status(self) -> dict[str, Any]:
        return {
            "recompute_interval_steps": self.command_recompute_interval_steps,
            "recomputed_this_tick": self._command_recomputed_this_tick,
            "compute_count": self._dwa_compute_count,
            "held_command_count": self._dwa_hold_count,
            "last_duration_s": self._last_dwa_compute_duration_s,
            "max_duration_s": self._max_dwa_compute_duration_s,
        }

    def _active_dwa_limits(self) -> dict[str, Any]:
        return {
            "lookahead_distance": float(
                self._active_dwa_config.lookahead_distance
            ),
            "max_linear_velocity": float(
                self._active_dwa_config.max_linear_velocity
            ),
            "max_angular_velocity": float(
                self._active_dwa_config.max_angular_velocity
            ),
            "max_linear_accel": float(self._active_dwa_config.max_linear_accel),
            "min_active_linear_velocity": float(
                self._active_dwa_config.min_active_linear_velocity
            ),
            "near_goal_min_active_linear_velocity": float(
                self._active_dwa_config.near_goal_min_active_linear_velocity
            ),
            "close_goal_speed_limit": float(
                self._active_dwa_config.close_goal_speed_limit
            ),
            "path_deviation_limit": float(
                self._active_dwa_config.path_deviation_limit
            ),
            "enforce_path_deviation_limit": bool(
                self._active_dwa_config.enforce_path_deviation_limit
            ),
            "initial_alignment_path_deviation_limit": (
                self._active_dwa_config.initial_alignment_path_deviation_limit
            ),
            "path_recovery_deviation_limit": (
                self._active_dwa_config.path_recovery_deviation_limit
            ),
            "near_goal_path_deviation_limit": (
                self._active_dwa_config.near_goal_path_deviation_limit
            ),
            "near_goal_path_deviation_distance": (
                self._active_dwa_config.near_goal_path_deviation_distance
            ),
            "preserve_sharp_corners": bool(
                self._active_dwa_config.preserve_sharp_corners
            ),
            "corner_angle_threshold": float(
                self._active_dwa_config.corner_angle_threshold
            ),
            "corner_waypoint_tolerance": float(
                self._active_dwa_config.corner_waypoint_tolerance
            ),
            "rotate_in_place_angle": float(
                self._active_dwa_config.rotate_in_place_angle
            ),
            "large_heading_creep_velocity": (
                self._active_dwa_config.large_heading_creep_velocity
            ),
            "consecutive_infeasible_recomputes": int(
                self._consecutive_infeasible_recomputes
            ),
            "max_infeasible_recomputes": self.carry_max_infeasible_recomputes,
        }

    def _stair_float_status(self) -> dict[str, Any]:
        status = dict(self._stair_float_report)
        status.update(
            {
                "active": bool(self._stair_float_active),
                "done": bool(self._stair_float_done),
                "started": bool(self._stair_float_started),
                "release_settling": bool(self._stair_float_release_settling),
                "release_settle_remaining_s": float(
                    self._stair_float_release_settle_remaining_s
                ),
                "progress_m": float(self._stair_float_progress),
                "total_length_m": float(self._stair_float_total_length),
            }
        )
        return status

    def _reset_stair_float_state(self, plan: NavPlan) -> None:
        """为当前 PCT carry plan 准备可选楼梯漂移段。"""

        self._stair_float_path = ()
        self._stair_float_segment_lengths = ()
        self._stair_float_total_length = 0.0
        self._stair_float_progress = 0.0
        self._stair_float_active = False
        self._stair_float_done = False
        self._stair_float_started = False
        self._stair_float_last_timestamp = None
        self._stair_float_root_z_offset = 0.0
        self._stair_float_initial_root_z_offset = 0.0
        self._stair_float_target_root_z_offset = 0.0
        self._stair_float_settling = False
        self._stair_float_settle_remaining_s = 0.0
        self._stair_float_release_settling = False
        self._stair_float_release_settle_remaining_s = 0.0
        self._stair_float_hold_xyzyaw = None
        self._stair_float_report = {
            "enabled": False,
            "reason": "disabled",
        }
        if not self.stair_float_enabled:
            return
        if plan.metadata.get("planner") != "pct":
            self._stair_float_report = {
                "enabled": False,
                "reason": "planner_is_not_pct",
            }
            return
        if plan.metadata.get("execution_phase") != "carry_nav_to_place":
            self._stair_float_report = {
                "enabled": False,
                "reason": "execution_phase_is_not_carry",
            }
            return
        path = _extract_stair_float_path(
            plan,
            min_z_delta_m=self.stair_float_min_z_delta_m,
            approach_distance_m=self.stair_float_approach_distance_m,
            exit_distance_m=self.stair_float_exit_distance_m,
        )
        if len(path) < 2:
            self._stair_float_report = {
                "enabled": False,
                "reason": "no_cross_floor_stair_segment",
            }
            return
        lengths = _segment_lengths_3d(path)
        total = sum(lengths)
        if total <= 1.0e-6:
            self._stair_float_report = {
                "enabled": False,
                "reason": "stair_segment_length_zero",
            }
            return
        z_values = [point[2] for point in path]
        self._stair_float_path = path
        self._stair_float_segment_lengths = tuple(lengths)
        self._stair_float_total_length = float(total)
        self._stair_float_target_root_z_offset = (
            self._desired_stair_float_target_root_z_offset(plan, path)
        )
        self._stair_float_report = {
            "enabled": True,
            "reason": "ready",
            "path_point_count": len(path),
            "start": list(path[0]),
            "end": list(path[-1]),
            "z_min": min(z_values),
            "z_max": max(z_values),
            "speed_mps": self.stair_float_speed_mps,
            "activation_radius_m": self.stair_float_activation_radius_m,
            "completion_radius_m": self.stair_float_completion_radius_m,
            "approach_distance_m": self.stair_float_approach_distance_m,
            "exit_distance_m": self.stair_float_exit_distance_m,
            "settle_time_s": self.stair_float_settle_time_s,
            "release_settle_time_s": self.stair_float_release_settle_time_s,
            "min_root_z_offset_m": self.stair_float_min_root_z_offset_m,
            "release_root_z_offset_m": (
                self.stair_float_release_root_z_offset_m
            ),
            "target_root_z_offset_m": float(
                self._stair_float_target_root_z_offset
            ),
        }

    def _compute_stair_float_action(
        self,
        state: SimulationState,
        pose: tuple[float, float, float],
    ) -> RobotAction | None:
        """在 PCT 楼梯段冻结底盘并沿 3D path 小步推进。"""

        if not self._stair_float_path or self._stair_float_done:
            return None
        if self._stair_float_release_settling:
            return self._compute_stair_float_release_settle_action(state)
        if self._stair_float_settling:
            return self._compute_stair_float_settle_action(state)
        if not self._stair_float_active:
            if not self._stair_float_should_activate(pose):
                return None
            self._activate_stair_float(state, pose)

        dt = self._stair_float_dt(state)
        self._stair_float_progress = min(
            self._stair_float_total_length,
            self._stair_float_progress + self.stair_float_speed_mps * dt,
        )
        path_target = _interpolate_polyline_3d(
            self._stair_float_path,
            self._stair_float_segment_lengths,
            self._stair_float_progress,
        )
        path_lookahead_target = _interpolate_polyline_3d(
            self._stair_float_path,
            self._stair_float_segment_lengths,
            min(
                self._stair_float_total_length,
                self._stair_float_progress + self.stair_float_yaw_lookahead_m,
            ),
        )
        target = self._stair_float_root_pose_target(
            path_target,
            progress_m=self._stair_float_progress,
        )
        lookahead_target = self._stair_float_root_pose_target(
            path_lookahead_target,
            progress_m=min(
                self._stair_float_total_length,
                self._stair_float_progress + self.stair_float_yaw_lookahead_m,
            ),
        )
        yaw = _float_path_yaw(
            target,
            lookahead_target,
            fallback_yaw=pose[2],
        )
        completed = (
            self._stair_float_total_length - self._stair_float_progress
            <= self.stair_float_completion_radius_m
        )
        if completed:
            self._stair_float_progress = self._stair_float_total_length
            target = self._stair_float_root_pose_target(
                self._stair_float_path[-1],
                progress_m=self._stair_float_total_length,
            )
            yaw = _float_path_yaw(
                self._stair_float_root_pose_target(
                    self._stair_float_path[-2],
                    progress_m=max(
                        0.0,
                        self._stair_float_total_length
                        - self._stair_float_segment_lengths[-1],
                    ),
                ),
                target,
                fallback_yaw=yaw,
            )
            self._stair_float_active = False
            self._stair_float_settling = self.stair_float_settle_time_s > 0.0
            self._stair_float_settle_remaining_s = self.stair_float_settle_time_s
            self._stair_float_hold_xyzyaw = (
                float(target[0]),
                float(target[1]),
                float(target[2]),
                float(yaw),
            )
            if self._stair_float_settling:
                self._stair_float_done = False
            elif self.stair_float_release_settle_time_s > 0.0:
                self._begin_stair_float_release_settle()
            else:
                self._stair_float_done = True
            self._sync_controller_after_stair_float(target)
        self._phase = "stair_float_completed" if completed else "stair_float"
        self._last_command = (0.0, 0.0, 0.0)
        self._command_recomputed_this_tick = True
        self._tick_index += 1
        progress_ratio = (
            self._stair_float_progress / self._stair_float_total_length
            if self._stair_float_total_length > 0.0
            else 1.0
        )
        metadata = self._action_metadata()
        metadata.update(
            {
                "navigation_base_pose_lock": True,
                "navigation_base_pose_lock_phase": "pct_stair_float",
                "navigation_base_pose_lock_xyzyaw": (
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    float(yaw),
                ),
                "navigation_support_joint_lock": True,
                "navigation_support_joint_lock_phase": "pct_stair_float",
                "navigation_full_body_joint_lock": True,
                "navigation_full_body_joint_lock_phase": "pct_stair_float",
                "navigation_dog_joint_names": PCT_STAIR_FLOAT_DOG_JOINT_NAMES,
                "navigation_dog_joint_positions": (
                    PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS
                ),
                "navigation_stair_float": True,
                "navigation_carry_object_follow": True,
                "navigation_stair_float_completed": completed,
                "navigation_stair_float_progress_ratio": float(progress_ratio),
            }
        )
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source=(
                "navigation_stair_float_completed"
                if completed
                else "navigation_stair_float"
            ),
            metadata=metadata,
        )

    def _compute_stair_float_settle_action(
        self,
        state: SimulationState,
    ) -> RobotAction:
        """楼梯漂移结束后继续锁住 root 和全身关节，等待物理状态稳定。"""

        if self._stair_float_hold_xyzyaw is None:
            raise RuntimeError("楼梯漂移稳定阶段缺少锁定目标。")
        dt = self._stair_float_dt(state)
        self._stair_float_settle_remaining_s = max(
            0.0,
            self._stair_float_settle_remaining_s - dt,
        )
        completed = self._stair_float_settle_remaining_s <= 0.0
        if completed:
            self._stair_float_settling = False
            if self.stair_float_release_settle_time_s > 0.0:
                self._begin_stair_float_release_settle()
            else:
                self._stair_float_done = True
                self._stair_float_report["reason"] = "completed"
        else:
            self._stair_float_report["reason"] = "settling"
        self._stair_float_report["settle_remaining_s"] = float(
            self._stair_float_settle_remaining_s
        )
        self._phase = "stair_float_completed" if completed else "stair_float_settle"
        self._last_command = (0.0, 0.0, 0.0)
        self._command_recomputed_this_tick = True
        self._tick_index += 1
        metadata = self._action_metadata()
        metadata.update(
            {
                "navigation_base_pose_lock": True,
                "navigation_base_pose_lock_phase": "pct_stair_float_settle",
                "navigation_base_pose_lock_xyzyaw": self._stair_float_hold_xyzyaw,
                "navigation_support_joint_lock": True,
                "navigation_support_joint_lock_phase": "pct_stair_float_settle",
                "navigation_full_body_joint_lock": True,
                "navigation_full_body_joint_lock_phase": "pct_stair_float_settle",
                "navigation_dog_joint_names": PCT_STAIR_FLOAT_DOG_JOINT_NAMES,
                "navigation_dog_joint_positions": (
                    PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS
                ),
                "navigation_stair_float": True,
                "navigation_carry_object_follow": True,
                "navigation_stair_float_settling": not completed,
                "navigation_stair_float_completed": completed,
                "navigation_stair_float_root_only_settle": False,
                "navigation_stair_float_progress_ratio": 1.0,
                "navigation_stair_float_settle_remaining_s": float(
                    self._stair_float_settle_remaining_s
                ),
            }
        )
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            source=(
                "navigation_stair_float_completed"
                if completed
                else "navigation_stair_float_settle"
            ),
            metadata=metadata,
        )

    def _begin_stair_float_release_settle(self) -> None:
        """进入解冻过渡：释放 root/direct joint state 前先记录剩余站稳时间。"""

        self._stair_float_release_settling = True
        self._stair_float_done = False
        self._stair_float_release_settle_remaining_s = (
            self.stair_float_release_settle_time_s
        )
        self._stair_float_report.update(
            {
                "reason": "release_settling",
                "release_settle_remaining_s": float(
                    self._stair_float_release_settle_remaining_s
                ),
            }
        )

    def _compute_stair_float_release_settle_action(
        self,
        state: SimulationState,
    ) -> RobotAction:
        """先解除 direct 关节写入但保持 root 锁，让支撑 target 接管。"""

        if self._stair_float_hold_xyzyaw is None:
            raise RuntimeError("楼梯漂移解冻阶段缺少锁定目标。")

        dt = self._stair_float_dt(state)
        self._stair_float_release_settle_remaining_s = max(
            0.0,
            self._stair_float_release_settle_remaining_s - dt,
        )
        completed = self._stair_float_release_settle_remaining_s <= 0.0
        if completed:
            self._stair_float_release_settling = False
            self._stair_float_done = True
            self._stair_float_report["reason"] = "completed"
        else:
            self._stair_float_report["reason"] = "release_settling"
        self._stair_float_report["release_settle_remaining_s"] = float(
            self._stair_float_release_settle_remaining_s
        )
        self._phase = (
            "stair_float_completed" if completed else "stair_float_release_settle"
        )
        self._last_command = (0.0, 0.0, 0.0)
        self._command_recomputed_this_tick = True
        self._tick_index += 1
        metadata = self._action_metadata()
        metadata.update(
            {
                "navigation_stair_float": True,
                "navigation_stair_float_release_settling": not completed,
                "navigation_stair_float_completed": completed,
                "navigation_stair_float_release_settle_remaining_s": float(
                    self._stair_float_release_settle_remaining_s
                ),
            }
        )
        if not completed:
            metadata.update(
                {
                    "navigation_base_pose_lock": True,
                    "navigation_base_pose_lock_phase": (
                        "pct_stair_float_release_settle"
                    ),
                    "navigation_base_pose_lock_xyzyaw": (
                        self._stair_float_hold_xyzyaw
                    ),
                    "navigation_support_joint_lock": True,
                    "navigation_support_joint_lock_phase": (
                        "pct_stair_float_release_settle"
                    ),
                    "navigation_dog_joint_names": PCT_STAIR_FLOAT_DOG_JOINT_NAMES,
                    "navigation_dog_joint_positions": (
                        PCT_STAIR_FLOAT_DOG_STAND_JOINT_POSITIONS
                    ),
                    "navigation_carry_object_follow": True,
                }
            )
        return RobotAction(
            base_velocity=(0.0, 0.0, 0.0),
            gripper_command="hold",
            source=(
                "navigation_stair_float_release_settle_completed"
                if completed
                else "navigation_stair_float_release_settle"
            ),
            metadata=metadata,
        )

    def _stair_float_should_activate(
        self,
        pose: tuple[float, float, float],
    ) -> bool:
        start = self._stair_float_path[0]
        distance = math.hypot(pose[0] - start[0], pose[1] - start[1])
        self._stair_float_report.update(
            {
                "distance_to_activation_m": float(distance),
                "activation_pose_xyyaw": list(pose),
            }
        )
        return distance <= self.stair_float_activation_radius_m

    def _activate_stair_float(
        self,
        state: SimulationState,
        pose: tuple[float, float, float],
    ) -> None:
        """从机器人当前位置投影到楼梯折线，避免首帧产生明显跳变。"""

        current = (
            float(pose[0]),
            float(pose[1]),
            float(state.robot_root_pose[2]),
        )
        projected = _project_progress_to_polyline_3d(
            self._stair_float_path,
            self._stair_float_segment_lengths,
            current,
        )
        projected_point = _interpolate_polyline_3d(
            self._stair_float_path,
            self._stair_float_segment_lengths,
            projected,
        )
        measured_offset = current[2] - projected_point[2]
        self._stair_float_initial_root_z_offset = max(
            measured_offset,
            self.stair_float_min_root_z_offset_m,
        )
        self._stair_float_target_root_z_offset = max(
            self._stair_float_target_root_z_offset,
            self._stair_float_initial_root_z_offset,
        )
        self._stair_float_root_z_offset = self._stair_float_initial_root_z_offset
        target_distance = _distance_3d(
            self._stair_float_root_pose_target(
                projected_point,
                progress_m=projected,
            ),
            current,
        )
        inserted_current_point = False
        if target_distance > 0.05:
            # 触发半径可能让机器人在楼梯段起点前进入漂移；把当前点作为临时首点，
            # 避免第一帧 root pose 直接跳到 PCT 折线起点。
            current_path_point = (
                current[0],
                current[1],
                current[2] - self._stair_float_root_z_offset,
            )
            self._stair_float_path = (current_path_point, *self._stair_float_path)
            self._stair_float_segment_lengths = tuple(
                _segment_lengths_3d(self._stair_float_path)
            )
            self._stair_float_total_length = float(
                sum(self._stair_float_segment_lengths)
            )
            projected = 0.0
            inserted_current_point = True
        self._stair_float_progress = max(
            0.0,
            min(projected, self._stair_float_total_length),
        )
        self._stair_float_active = True
        self._stair_float_started = True
        self._stair_float_last_timestamp = float(state.timestamp)
        self._stair_float_report.update(
            {
                "reason": "active",
                "activation_step_index": int(state.step_index),
                "activation_progress_m": float(self._stair_float_progress),
                "root_z_offset_m": float(self._stair_float_root_z_offset),
                "initial_root_z_offset_m": float(
                    self._stair_float_initial_root_z_offset
                ),
                "target_root_z_offset_m": float(
                    self._stair_float_target_root_z_offset
                ),
                "measured_root_z_offset_m": float(measured_offset),
                "activation_inserted_current_point": inserted_current_point,
                "activation_target_jump_m": float(target_distance),
                "path_point_count": len(self._stair_float_path),
                "total_length_m": float(self._stair_float_total_length),
                "settle_remaining_s": float(self._stair_float_settle_remaining_s),
            }
        )

    def _stair_float_root_pose_target(
        self,
        path_point: tuple[float, float, float],
        *,
        progress_m: float | None = None,
    ) -> tuple[float, float, float]:
        """把 PCT 楼层高度转换成连续的机器人 root 高度。"""

        offset = self._stair_float_root_z_offset_at(progress_m)
        self._stair_float_root_z_offset = offset
        self._stair_float_report["root_z_offset_m"] = float(offset)
        self._stair_float_report["current_root_z_offset_m"] = float(offset)
        return (
            float(path_point[0]),
            float(path_point[1]),
            float(path_point[2]) + float(offset),
        )

    def _stair_float_root_z_offset_at(
        self,
        progress_m: float | None,
    ) -> float:
        """按漂移进度把 root 高度偏移从实测值平滑过渡到释放值。"""

        if self._stair_float_total_length <= 1.0e-6 or progress_m is None:
            return float(self._stair_float_initial_root_z_offset)
        ratio = min(
            1.0,
            max(0.0, float(progress_m) / float(self._stair_float_total_length)),
        )
        smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return float(self._stair_float_initial_root_z_offset) + (
            float(self._stair_float_target_root_z_offset)
            - float(self._stair_float_initial_root_z_offset)
        ) * smooth_ratio

    def _desired_stair_float_target_root_z_offset(
        self,
        plan: NavPlan,
        path: tuple[tuple[float, float, float], ...],
    ) -> float:
        """返回最低释放偏移，激活时再用机器人实测站立高度抬高。"""

        del plan, path
        return max(
            float(self.stair_float_min_root_z_offset_m),
            float(self.stair_float_release_root_z_offset_m),
        )

    def _stair_float_dt(self, state: SimulationState) -> float:
        timestamp = float(state.timestamp)
        if self._stair_float_last_timestamp is None:
            dt = float(self._active_dwa_config.control_dt)
        else:
            dt = timestamp - self._stair_float_last_timestamp
        self._stair_float_last_timestamp = timestamp
        if dt <= 0.0:
            dt = float(self._active_dwa_config.control_dt)
        return min(max(dt, 1.0e-3), 0.20)

    def _sync_controller_after_stair_float(
        self,
        target: tuple[float, float, float],
    ) -> None:
        """楼梯漂移结束后把 DWA target_index 推进到二楼附近。"""

        if self._replan_controller_on_post_stair_floor(target):
            return
        if self._controller is None:
            return
        path = self._controller.path_world
        if len(path) == 0:
            return
        xy = (float(target[0]), float(target[1]))
        nearest_index = min(
            range(len(path)),
            key=lambda index: math.hypot(
                float(path[index][0]) - xy[0],
                float(path[index][1]) - xy[1],
            ),
        )
        self._controller.target_index = max(
            self._controller.target_index,
            min(len(path) - 1, int(nearest_index)),
        )
        self._controller._initial_alignment_active = False
        self._controller._path_recovery_active = False
        self._stair_float_report.update(
            {
                "reason": "completed",
                "controller_target_index_after_float": int(
                    self._controller.target_index
                ),
            }
        )

    def _replan_controller_on_post_stair_floor(
        self,
        target: tuple[float, float, float],
    ) -> bool:
        """楼梯结束后切换到目标楼层地图，避免跨层清廊抹掉二楼墙体。"""

        report: dict[str, Any] = {
            "applied": False,
            "reason": "unavailable",
        }
        floor_raw_map = (
            self._post_stair_raw_map
            if self._post_stair_raw_map is not None
            else self._single_floor_raw_map
        )
        if self.plan is None or floor_raw_map is None:
            self._stair_float_report["post_stair_floor_replan"] = report
            return False

        floor_map = floor_raw_map.inflate(self.local_clearance_radius)
        start_xy = (float(target[0]), float(target[1]))
        goal_xy = (float(self.plan.goal.x), float(self.plan.goal.y))
        try:
            result = AStarPlanner().plan(
                floor_map,
                start_xy,
                goal_xy,
                snap_to_free=True,
                max_snap_distance_m=max(
                    0.50,
                    float(self.local_clearance_radius) + 0.50,
                ),
            )
        except Exception as exc:
            report.update(
                {
                    "reason": "astar_failed",
                    "failure_reason": str(exc),
                    "start_xy": list(start_xy),
                    "goal_xy": list(goal_xy),
                }
            )
            self._stair_float_report["post_stair_floor_replan"] = report
            return False

        path_world = [start_xy]
        for point in result.path_world:
            xy = (float(point[0]), float(point[1]))
            if math.hypot(
                xy[0] - path_world[-1][0],
                xy[1] - path_world[-1][1],
            ) > 1.0e-5:
                path_world.append(xy)
        if math.hypot(
            goal_xy[0] - path_world[-1][0],
            goal_xy[1] - path_world[-1][1],
        ) > 1.0e-5:
            path_world.append(goal_xy)
        if len(path_world) < 2:
            report.update(
                {
                    "reason": "path_too_short",
                    "start_xy": list(start_xy),
                    "goal_xy": list(goal_xy),
                }
            )
            self._stair_float_report["post_stair_floor_replan"] = report
            return False

        post_stair_dwa_config = self._post_stair_dwa_config()
        self._raw_map = floor_raw_map
        self.local_map = floor_map
        self._active_dwa_config = post_stair_dwa_config
        self._controller = DWAController(
            path_world=path_world,
            grid_map=floor_map,
            config=post_stair_dwa_config,
        )
        self._consecutive_infeasible_recomputes = 0
        self.stall_detector.reset()
        report.update(
            {
                "applied": True,
                "reason": "post_stair_single_floor_astar",
                "start_xy": list(start_xy),
                "goal_xy": list(goal_xy),
                "waypoint_count": len(path_world),
                "raw_grid_waypoint_count": len(result.raw_path_grid),
                "astar_cost": float(result.cost),
                "path_world": [list(point) for point in path_world],
                "max_linear_velocity": float(
                    post_stair_dwa_config.max_linear_velocity
                ),
                "min_active_linear_velocity": float(
                    post_stair_dwa_config.min_active_linear_velocity
                ),
                "near_goal_min_active_linear_velocity": float(
                    post_stair_dwa_config.near_goal_min_active_linear_velocity
                ),
                "close_goal_speed_limit": float(
                    post_stair_dwa_config.close_goal_speed_limit
                ),
                "path_deviation_limit": float(
                    post_stair_dwa_config.path_deviation_limit
                ),
                "path_recovery_deviation_limit": (
                    post_stair_dwa_config.path_recovery_deviation_limit
                ),
                "corner_waypoint_tolerance": float(
                    post_stair_dwa_config.corner_waypoint_tolerance
                ),
                "lookahead_distance": float(
                    post_stair_dwa_config.lookahead_distance
                ),
            }
        )
        self._stair_float_report.update(
            {
                "reason": "completed",
                "controller_target_index_after_float": int(
                    self._controller.target_index
                ),
                "post_stair_floor_replan": report,
            }
        )
        if self._map_selection_report is not None:
            self._map_selection_report.update(
                {
                    "active_map": "post_stair_single_floor",
                    "post_stair_floor_replan": report,
                }
            )
        return True

    def _post_stair_dwa_config(self) -> DWAConfig:
        """为二楼走廊限速，同时保留已验证的平滑转角恢复范围。"""

        config = self._active_dwa_config
        path_deviation_limit = min(
            float(config.path_deviation_limit),
            0.14,
        )
        return replace(
            config,
            lookahead_distance=max(
                float(config.lookahead_distance),
                0.40,
            ),
            max_linear_velocity=min(
                float(config.max_linear_velocity),
                0.25,
            ),
            min_active_linear_velocity=min(
                float(config.min_active_linear_velocity),
                0.22,
            ),
            near_goal_min_active_linear_velocity=max(
                float(config.near_goal_min_active_linear_velocity),
                0.22,
            ),
            close_goal_speed_limit=max(
                float(config.close_goal_speed_limit),
                0.22,
            ),
            speed_bias=min(float(config.speed_bias), 0.75),
            clearance_bias=max(float(config.clearance_bias), 0.75),
            path_bias=max(float(config.path_bias), 1.40),
            trajectory_path_bias=max(
                float(config.trajectory_path_bias),
                1.80,
            ),
            path_deviation_penalty_bias=max(
                float(config.path_deviation_penalty_bias),
                2.40,
            ),
            path_deviation_limit=path_deviation_limit,
            initial_alignment_path_deviation_limit=max(
                path_deviation_limit,
                min(
                    float(config.initial_alignment_path_deviation_limit or 0.40),
                    0.40,
                ),
            ),
            path_recovery_deviation_limit=max(
                path_deviation_limit,
                min(
                    float(config.path_recovery_deviation_limit or 0.50),
                    0.50,
                ),
            ),
            near_goal_path_deviation_limit=max(
                path_deviation_limit,
                min(
                    float(config.near_goal_path_deviation_limit or 0.75),
                    0.75,
                ),
            ),
            corner_waypoint_tolerance=max(
                float(config.corner_waypoint_tolerance),
                0.18,
            ),
        )

    @staticmethod
    def _stall_status(diagnostics: StallDiagnostics) -> dict[str, Any]:
        return {
            "sample_count": int(diagnostics.sample_count),
            "max_displacement_m": float(diagnostics.max_displacement_m),
            "forward_command_ratio": float(diagnostics.forward_command_ratio),
        }


def _pct_plan_is_multifloor(plan: NavPlan) -> bool:
    """根据 slice 元数据和三维轨迹判断 PCT 路径是否跨楼层。"""

    path_3d = plan.metadata.get("path_3d")
    if isinstance(path_3d, (list, tuple)):
        z_values = [
            float(point[2])
            for point in path_3d
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ]
        if z_values:
            return max(z_values) - min(z_values) > 0.35
    slice_start = plan.metadata.get("slice_start")
    slice_end = plan.metadata.get("slice_end")
    if slice_start is None or slice_end is None:
        return False
    try:
        return abs(int(slice_end) - int(slice_start)) >= 2
    except (TypeError, ValueError):
        return False


def _pct_no_float_route_corridors(
    plan: NavPlan,
    *,
    flat_radius_m: float,
    stair_radius_m: float,
) -> list[tuple[str, list[tuple[float, float]], float]]:
    """把跨层路径拆为平层和楼梯清廊，防止宽清廊抹掉扶手。"""

    raw_path = plan.metadata.get("path_3d")
    if not isinstance(raw_path, (list, tuple)):
        return [
            (
                "full_path",
                [(float(point[0]), float(point[1])) for point in plan.waypoints],
                float(flat_radius_m),
            )
        ]
    path = [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in raw_path
        if isinstance(point, (list, tuple)) and len(point) >= 3
    ]
    changing_segments = [
        index
        for index, (start, end) in enumerate(zip(path, path[1:]))
        if abs(float(end[2]) - float(start[2])) > 1.0e-4
    ]
    if not changing_segments:
        return [
            (
                "full_path",
                [(point[0], point[1]) for point in path],
                float(flat_radius_m),
            )
        ]

    stair_start = max(0, changing_segments[0] - 1)
    stair_end = min(len(path) - 1, changing_segments[-1] + 1)
    routes: list[tuple[str, list[tuple[float, float]], float]] = []
    before = [(point[0], point[1]) for point in path[: stair_start + 1]]
    stair = [(point[0], point[1]) for point in path[stair_start : stair_end + 1]]
    after = [(point[0], point[1]) for point in path[stair_end:]]
    if len(before) >= 2:
        routes.append(("flat_before_stair", before, float(flat_radius_m)))
    if len(stair) >= 2:
        routes.append(("stair_centerline", stair, float(stair_radius_m)))
    if len(after) >= 2:
        routes.append(("flat_after_stair", after, float(flat_radius_m)))
    return routes or [
        (
            "full_path",
            [(point[0], point[1]) for point in path],
            float(flat_radius_m),
        )
    ]


def _clear_grid_path_corridor(
    grid_map: OccupancyGridMap,
    path_world: list[tuple[float, float]],
    *,
    radius_m: float,
    protected_obstacle_map: OccupancyGridMap | None = None,
) -> tuple[OccupancyGridMap, int, int]:
    """在保守障碍图中恢复 PCT 已验证中心线附近的窄走廊。"""

    if len(path_world) < 2:
        return grid_map, 0, 0
    if protected_obstacle_map is not None:
        if (
            protected_obstacle_map.shape != grid_map.shape
            or not math.isclose(
                protected_obstacle_map.resolution,
                grid_map.resolution,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
            or any(
                not math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-9)
                for left, right in zip(
                    protected_obstacle_map.origin,
                    grid_map.origin,
                )
            )
        ):
            raise ValueError("受保护硬障碍图与跨楼层局部地图不对齐。")
    occupancy = grid_map.occupancy.copy()
    radius = max(0.0, float(radius_m))
    radius_cells = int(math.ceil(radius / float(grid_map.resolution)))
    sample_spacing = max(0.01, 0.5 * float(grid_map.resolution))
    cleared_count = 0
    protected_count = 0

    def clear_at(x: float, y: float) -> None:
        nonlocal cleared_count, protected_count
        center_row, center_col = grid_map.world_to_grid(x, y)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                row = center_row + dr
                col = center_col + dc
                if not grid_map.in_bounds(row, col):
                    continue
                cell_x, cell_y = grid_map.grid_to_world(row, col)
                if radius > 0.0 and math.hypot(cell_x - x, cell_y - y) > radius:
                    continue
                if occupancy[row, col]:
                    if (
                        protected_obstacle_map is not None
                        and protected_obstacle_map.is_occupied(row, col)
                    ):
                        protected_count += 1
                        continue
                    occupancy[row, col] = False
                    cleared_count += 1

    for start, end in zip(path_world, path_world[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        sample_count = max(1, int(math.ceil(segment_length / sample_spacing)))
        for sample_index in range(sample_count + 1):
            ratio = float(sample_index) / float(sample_count)
            clear_at(
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
    return (
        OccupancyGridMap(
            occupancy=occupancy,
            resolution=grid_map.resolution,
            origin=grid_map.origin,
            image_path=grid_map.image_path,
            meta_path=grid_map.meta_path,
        ),
        cleared_count,
        protected_count,
    )


def _world_segment_clearance(
    start: tuple[float, float],
    end: tuple[float, float],
    grid_map: OccupancyGridMap,
) -> tuple[bool, float | None]:
    """检查同层直线路径，并返回沿线最小障碍净空。"""

    segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
    sample_spacing = max(0.01, 0.5 * float(grid_map.resolution))
    sample_count = max(1, int(math.ceil(segment_length / sample_spacing)))
    minimum_clearance = float("inf")
    for sample_index in range(sample_count + 1):
        alpha = sample_index / sample_count
        x = start[0] + alpha * (end[0] - start[0])
        y = start[1] + alpha * (end[1] - start[1])
        row, col = grid_map.world_to_grid(x, y)
        if grid_map.is_occupied(row, col):
            return False, 0.0
        clearance = grid_map.distance_to_obstacle(row, col)
        if clearance is None:
            return False, None
        minimum_clearance = min(minimum_clearance, float(clearance))
    return True, minimum_clearance


def _remove_consecutive_duplicate_waypoints(
    path_world: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """移除跨 slice gateway 在 XY 投影上的连续重复点。"""

    output: list[tuple[float, float]] = []
    for point in path_world:
        if output and math.hypot(
            point[0] - output[-1][0],
            point[1] - output[-1][1],
        ) <= 1.0e-6:
            continue
        output.append(point)
    return output


def _extract_stair_float_path(
    plan: NavPlan,
    *,
    min_z_delta_m: float,
    approach_distance_m: float = 0.0,
    exit_distance_m: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """从 PCT 3D path 截取楼梯段，并可包含入口和出口缓冲。"""

    path_3d = plan.metadata.get("path_3d")
    if not isinstance(path_3d, (list, tuple)):
        return ()
    points: list[tuple[float, float, float]] = []
    for raw_point in path_3d:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 3:
            continue
        point = (
            float(raw_point[0]),
            float(raw_point[1]),
            float(raw_point[2]),
        )
        if points and _distance_3d(points[-1], point) <= 1.0e-5:
            continue
        points.append(point)
    if len(points) < 2:
        return ()
    z_values = [point[2] for point in points]
    if max(z_values) - min(z_values) < float(min_z_delta_m):
        return ()
    z_trend = 1.0 if points[-1][2] >= points[0][2] else -1.0
    changing_segments = [
        index
        for index in range(len(points) - 1)
        if (points[index + 1][2] - points[index][2]) * z_trend > 0.05
    ]
    if not changing_segments:
        return ()
    start_index = changing_segments[0]
    remaining_approach = max(0.0, float(approach_distance_m))
    while start_index > 0 and remaining_approach > 0.0:
        previous = points[start_index - 1]
        current = points[start_index]
        remaining_approach -= math.hypot(
            current[0] - previous[0],
            current[1] - previous[1],
        )
        start_index -= 1
    end_index = changing_segments[-1] + 1
    remaining_exit = max(0.0, float(exit_distance_m))
    while end_index < len(points) - 1 and remaining_exit > 0.0:
        current = points[end_index]
        next_point = points[end_index + 1]
        remaining_exit -= math.hypot(
            next_point[0] - current[0],
            next_point[1] - current[1],
        )
        end_index += 1
    if end_index < len(points) - 1 and float(exit_distance_m) <= 0.0:
        end_index += 1
    stair_path = tuple(points[start_index : end_index + 1])
    if len(stair_path) < 2:
        return ()
    stair_z_values = [point[2] for point in stair_path]
    if max(stair_z_values) - min(stair_z_values) < float(min_z_delta_m):
        return ()
    return _straighten_stair_float_tail(stair_path, z_trend=z_trend)


def _stair_float_segment_heading(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float | None:
    """返回楼梯漂移段的 XY 朝向；近似竖直段没有可靠朝向。"""

    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) <= 1.0e-6:
        return None
    return math.atan2(dy, dx)


def _splice_stair_float_controller_path(
    path_world: list[tuple[float, float]],
    stair_path: tuple[tuple[float, float, float], ...],
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """把实际楼梯漂移轨迹接入 DWA 路径，并选择前向的二楼合流点。"""

    report: dict[str, Any] = {
        "applied": False,
        "input_waypoint_count": len(path_world),
        "output_waypoint_count": len(path_world),
    }
    if len(path_world) < 2 or len(stair_path) < 2:
        report["reason"] = "insufficient_path"
        return list(path_world), report

    stair_xy: list[tuple[float, float]] = []
    for point in stair_path:
        xy = (float(point[0]), float(point[1]))
        if stair_xy and math.hypot(
            xy[0] - stair_xy[-1][0],
            xy[1] - stair_xy[-1][1],
        ) <= 1.0e-5:
            continue
        stair_xy.append(xy)
    if len(stair_xy) < 2:
        report["reason"] = "degenerate_stair_path"
        return list(path_world), report

    start_xy = stair_xy[0]
    release_xy = stair_xy[-1]
    start_index = min(
        range(len(path_world)),
        key=lambda index: math.hypot(
            float(path_world[index][0]) - start_xy[0],
            float(path_world[index][1]) - start_xy[1],
        ),
    )
    nearest_release_index = min(
        range(start_index, len(path_world)),
        key=lambda index: math.hypot(
            float(path_world[index][0]) - release_xy[0],
            float(path_world[index][1]) - release_xy[1],
        ),
    )
    release_heading = math.atan2(
        release_xy[1] - stair_xy[-2][1],
        release_xy[0] - stair_xy[-2][0],
    )
    heading_x = math.cos(release_heading)
    heading_y = math.sin(release_heading)
    merge_index: int | None = None
    merge_heading_error: float | None = None
    for index in range(nearest_release_index + 1, len(path_world)):
        candidate = path_world[index]
        dx = float(candidate[0]) - release_xy[0]
        dy = float(candidate[1]) - release_xy[1]
        distance = math.hypot(dx, dy)
        if distance < 0.35:
            continue
        forward_progress = dx * heading_x + dy * heading_y
        if forward_progress < 0.20:
            continue
        heading_error = wrap_yaw(math.atan2(dy, dx) - release_heading)
        if abs(heading_error) > math.radians(75.0):
            continue
        merge_index = index
        merge_heading_error = heading_error
        break

    if merge_index is None:
        merge_index = min(len(path_world), nearest_release_index + 1)

    combined = [*path_world[:start_index], *stair_xy, *path_world[merge_index:]]
    output: list[tuple[float, float]] = []
    for point in combined:
        xy = (float(point[0]), float(point[1]))
        if output and math.hypot(
            xy[0] - output[-1][0],
            xy[1] - output[-1][1],
        ) <= 1.0e-5:
            continue
        output.append(xy)
    if len(output) < 2:
        report["reason"] = "splice_collapsed_path"
        return list(path_world), report

    merge_point = (
        output[-1]
        if merge_index >= len(path_world)
        else path_world[merge_index]
    )
    report.update(
        {
            "applied": True,
            "reason": "stair_release_forward_merge",
            "start_index": int(start_index),
            "nearest_release_index": int(nearest_release_index),
            "merge_index": int(merge_index),
            "skipped_waypoint_count": max(
                0,
                int(merge_index - nearest_release_index),
            ),
            "release_xy": [float(release_xy[0]), float(release_xy[1])],
            "merge_xy": [float(merge_point[0]), float(merge_point[1])],
            "merge_heading_error_rad": (
                None
                if merge_heading_error is None
                else float(merge_heading_error)
            ),
            "output_waypoint_count": len(output),
        }
    )
    return output, report


def _straighten_stair_float_tail(
    path: tuple[tuple[float, float, float], ...],
    *,
    z_trend: float,
) -> tuple[tuple[float, float, float], ...]:
    """拉直楼梯尾部异常横切升高段，避免冻结底盘在台阶末端大幅转向。"""

    changing_segments = [
        index
        for index in range(len(path) - 1)
        if (path[index + 1][2] - path[index][2]) * float(z_trend) > 0.05
    ]
    if len(changing_segments) < 2:
        return path
    tail_index = changing_segments[-1]
    z_values = [float(point[2]) for point in path]
    z_range = max(z_values) - min(z_values)
    if z_range <= 1.0e-6:
        return path
    tail_height_ratio = (
        float(path[tail_index][2]) - min(z_values)
    ) / z_range
    if tail_height_ratio < 0.75:
        return path
    tail_heading = _stair_float_segment_heading(
        path[tail_index],
        path[tail_index + 1],
    )
    if tail_heading is None:
        return path
    reference_heading: float | None = None
    for index in range(tail_index - 1, -1, -1):
        heading = _stair_float_segment_heading(path[index], path[index + 1])
        if heading is None:
            continue
        reference_heading = heading
        break
    if reference_heading is None:
        return path
    if abs(wrap_yaw(tail_heading - reference_heading)) <= math.radians(70.0):
        return path
    start = path[tail_index]
    transition_end = path[tail_index + 1]
    transition_horizontal = math.hypot(
        transition_end[0] - start[0],
        transition_end[1] - start[1],
    )
    if transition_horizontal <= 1.0e-6:
        return path
    heading_x = math.cos(reference_heading)
    heading_y = math.sin(reference_heading)
    straight_transition_end = (
        float(start[0]) + heading_x * transition_horizontal,
        float(start[1]) + heading_y * transition_horizontal,
        float(transition_end[2]),
    )
    projected_tail: list[tuple[float, float, float]] = [straight_transition_end]
    accumulated = transition_horizontal
    for index in range(tail_index + 1, len(path) - 1):
        current = path[index]
        next_point = path[index + 1]
        accumulated += math.hypot(
            float(next_point[0]) - float(current[0]),
            float(next_point[1]) - float(current[1]),
        )
        projected_tail.append(
            (
                float(start[0]) + heading_x * accumulated,
                float(start[1]) + heading_y * accumulated,
                float(next_point[2]),
            )
        )
    return (*path[: tail_index + 1], *projected_tail)


def _segment_lengths_3d(
    path: tuple[tuple[float, float, float], ...],
) -> list[float]:
    """计算 3D polyline 每段长度。"""

    return [
        _distance_3d(start, end)
        for start, end in zip(path, path[1:])
    ]


def _interpolate_polyline_3d(
    path: tuple[tuple[float, float, float], ...],
    lengths: tuple[float, ...],
    progress_m: float,
) -> tuple[float, float, float]:
    """按弧长插值 3D polyline。"""

    if not path:
        raise ValueError("楼梯漂移路径不能为空。")
    if len(path) == 1 or not lengths:
        return path[0]
    remaining = max(0.0, float(progress_m))
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            ratio = 0.0 if length <= 1.0e-9 else remaining / length
            ratio = max(0.0, min(1.0, ratio))
            start = path[index]
            end = path[index + 1]
            return (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
                start[2] + ratio * (end[2] - start[2]),
            )
        remaining -= length
    return path[-1]


def _project_progress_to_polyline_3d(
    path: tuple[tuple[float, float, float], ...],
    lengths: tuple[float, ...],
    point: tuple[float, float, float],
) -> float:
    """把当前位置投影到 3D polyline，返回弧长进度。"""

    best_progress = 0.0
    best_distance = float("inf")
    accumulated = 0.0
    for index, length in enumerate(lengths):
        start = path[index]
        end = path[index + 1]
        if length <= 1.0e-9:
            continue
        segment = (
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
        )
        rel = (
            point[0] - start[0],
            point[1] - start[1],
            point[2] - start[2],
        )
        ratio = (
            rel[0] * segment[0]
            + rel[1] * segment[1]
            + rel[2] * segment[2]
        ) / (length * length)
        ratio = max(0.0, min(1.0, ratio))
        projected = (
            start[0] + ratio * segment[0],
            start[1] + ratio * segment[1],
            start[2] + ratio * segment[2],
        )
        distance = _distance_3d(point, projected)
        if distance < best_distance:
            best_distance = distance
            best_progress = accumulated + ratio * length
        accumulated += length
    return best_progress


def _float_path_yaw(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    fallback_yaw: float,
) -> float:
    """根据 3D 路径的 XY 投影计算底盘朝向。"""

    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    if math.hypot(dx, dy) <= 1.0e-6:
        return float(fallback_yaw)
    return math.atan2(dy, dx)


def _distance_3d(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    """计算两个 3D 点的欧氏距离。"""

    return math.sqrt(
        (float(end[0]) - float(start[0])) ** 2
        + (float(end[1]) - float(start[1])) ** 2
        + (float(end[2]) - float(start[2])) ** 2
    )


DWAExecutor = DwaNavExecutor
NavExecutor = DwaNavExecutor


__all__ = [
    "DWAExecutor",
    "DwaNavExecutor",
    "NavExecutor",
]
