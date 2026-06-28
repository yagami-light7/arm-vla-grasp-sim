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
        multifloor_protected_obstacle_map: OccupancyGridMap | None = None,
        terminal_pose_config: TerminalPoseConfig | None = None,
        terminal_start_distance: float = 0.50,
        position_tolerance: float = 0.15,
        yaw_tolerance: float = 0.15,
        stall_window_steps: int = 120,
        stall_min_progress_m: float = 0.05,
        stall_min_forward_command: float = 0.05,
        stall_min_forward_ratio: float = 0.25,
        completion_linear_velocity_tolerance: float | None = None,
        completion_angular_velocity_tolerance: float | None = None,
        require_yaw_alignment: bool = True,
        command_recompute_interval_steps: int = 1,
        multifloor_obstacle_inflate_radius: float = 0.0,
        multifloor_route_corridor_radius: float | None = None,
        carry_max_linear_velocity: float | None = None,
        carry_max_angular_velocity: float | None = None,
        carry_max_linear_accel: float | None = None,
        carry_path_deviation_limit: float | None = None,
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
        if multifloor_obstacle_inflate_radius < 0.0:
            raise ValueError("跨楼层障碍膨胀半径不能为负数。")
        if (
            multifloor_route_corridor_radius is not None
            and multifloor_route_corridor_radius < 0.0
        ):
            raise ValueError("跨楼层路径走廊半径不能为负数。")

        self.map_json = None if map_json is None else str(Path(map_json).expanduser().resolve())
        self._single_floor_raw_map = grid_map
        self._multifloor_raw_map = multifloor_grid_map
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
        self.carry_path_deviation_limit = carry_path_deviation_limit
        self.terminal_start_distance = max(
            float(terminal_start_distance),
            float(self.dwa_config.goal_tolerance),
        )
        self.position_tolerance = float(position_tolerance)
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
        self.completion_linear_velocity_tolerance = completion_linear_velocity_tolerance
        self.completion_angular_velocity_tolerance = completion_angular_velocity_tolerance
        self.require_yaw_alignment = bool(require_yaw_alignment)
        self.command_recompute_interval_steps = int(command_recompute_interval_steps)

        self.plan: NavPlan | None = None
        self._controller: DWAController | None = None
        self._phase = "idle"
        self._tick_index = 0
        self._done = False
        self._success = False
        self._failure_reason = ""
        self._stall_detected = False
        self._stall_diagnostics = self.stall_detector.diagnostics()
        self._last_command = (0.0, 0.0, 0.0)
        self._last_pose: tuple[float, float, float] | None = None
        self._last_body_velocity = (0.0, 0.0, 0.0)
        self._distance_to_goal: float | None = None
        self._yaw_error: float | None = None
        self._last_dwa_debug: DWADebug | None = None
        self._dwa_compute_count = 0
        self._dwa_hold_count = 0
        self._last_dwa_compute_duration_s = 0.0
        self._max_dwa_compute_duration_s = 0.0
        self._command_recomputed_this_tick = False
        self._local_refinement_report: dict[str, Any] | None = None
        self._map_selection_report: dict[str, Any] | None = None

    def reset(self, plan: NavPlan) -> None:
        """载入新路径并清空执行、终点和停滞状态。"""

        if len(plan.waypoints) < 2:
            raise ValueError("DWA 路径至少需要两个 waypoint。")
        self._select_local_map(plan)
        self._ensure_local_map(plan)
        if self.local_map is None:
            raise RuntimeError("导航执行器没有可用的占用栅格地图。")

        path_world = [
            (float(point[0]), float(point[1]))
            for point in plan.waypoints
        ]
        path_world = self._refine_pct_path_for_local_map(plan, path_world)
        self._active_dwa_config = self._dwa_config_for_plan(plan)
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
        self._stall_diagnostics = self.stall_detector.diagnostics()
        self._last_command = (0.0, 0.0, 0.0)
        self._last_pose = None
        self._last_body_velocity = (0.0, 0.0, 0.0)
        self._distance_to_goal = None
        self._yaw_error = None
        self._last_dwa_debug = None
        self._dwa_compute_count = 0
        self._dwa_hold_count = 0
        self._last_dwa_compute_duration_s = 0.0
        self._max_dwa_compute_duration_s = 0.0
        self._command_recomputed_this_tick = False

    def compute_action(self, state: SimulationState) -> RobotAction:
        """根据当前观测计算一个 tick 的 RobotAction，不推进仿真。"""

        self._require_plan()
        if self.is_done(state):
            return self._zero_action()

        pose = self._pose_xyyaw(state)
        body_velocity = self._body_velocity(state, pose[2])
        distance, yaw_error = self._goal_errors(pose)
        next_phase = (
            "terminal_pose"
            if self.require_yaw_alignment and distance <= self.terminal_start_distance
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
            command = compute_terminal_pose_command(
                body_goal_x=body_goal_x,
                body_goal_y=body_goal_y,
                yaw_error=yaw_error,
                distance_to_goal=distance,
                config=self.terminal_pose_config,
            )
            self._last_dwa_debug = None
            self._command_recomputed_this_tick = True
        else:
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
        if self._phase == "dwa":
            self._update_stall(pose, self._last_command)
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
        yaw_ok = (not self.require_yaw_alignment) or abs(yaw_error) <= self.yaw_tolerance
        if distance <= self.position_tolerance and yaw_ok:
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
            "position_tolerance": self.position_tolerance,
            "yaw_tolerance": self.yaw_tolerance,
            "yaw_alignment_required": self.require_yaw_alignment,
            "acceptance_mode": (
                "xy_yaw"
                if self.require_yaw_alignment
                else "xy_only"
            ),
            "completion_linear_velocity_tolerance": self.completion_linear_velocity_tolerance,
            "completion_angular_velocity_tolerance": self.completion_angular_velocity_tolerance,
            "last_command": self._last_command,
            "measured_body_velocity": self._last_body_velocity,
            "stall_detected": self._stall_detected,
            "stall": self._stall_status(self._stall_diagnostics),
            "dwa_compute": self._dwa_compute_status(),
            "dwa_limits": self._active_dwa_limits(),
            "local_refinement": self._local_refinement_report,
            "map_selection": self._map_selection_report,
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
                selected, cleared_count, protected_count = (
                    _clear_grid_path_corridor(
                        selected,
                        [
                            (float(point[0]), float(point[1]))
                            for point in plan.waypoints
                        ],
                        radius_m=float(
                            self.multifloor_route_corridor_radius
                        ),
                        protected_obstacle_map=(
                            self._multifloor_protected_obstacle_map
                        ),
                    )
                )
                self._map_selection_report.update(
                    {
                        "route_corridor_radius_m": float(
                            self.multifloor_route_corridor_radius
                        ),
                        "route_cells_cleared": cleared_count,
                        "protected_cells_preserved": protected_count,
                    }
                )
        self._raw_map = selected
        self.local_map = selected.inflate(self.local_clearance_radius)

    def _dwa_config_for_plan(self, plan: NavPlan) -> DWAConfig:
        """携物导航使用更保守的速度和路径偏离上限。"""

        if plan.metadata.get("execution_phase") != "carry_nav_to_place":
            return self.dwa_config
        updates: dict[str, float] = {}
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
        return replace(self.dwa_config, **updates)

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
        self._stall_detected = True
        self._done = True
        self._success = False
        self._failure_reason = "nav_collision"
        self._phase = "stalled"

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
            "measured_body_velocity": self._last_body_velocity,
            "yaw_alignment_required": self.require_yaw_alignment,
            "stall_detected": self._stall_detected,
            "failed": bool(self._failure_reason),
            "failure_reason": self._failure_reason,
            "stall": self._stall_status(self._stall_diagnostics),
            "dwa_compute": self._dwa_compute_status(),
            "dwa_limits": self._active_dwa_limits(),
            "local_refinement": self._local_refinement_report,
            "map_selection": self._map_selection_report,
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

    def _active_dwa_limits(self) -> dict[str, float]:
        return {
            "max_linear_velocity": float(
                self._active_dwa_config.max_linear_velocity
            ),
            "max_angular_velocity": float(
                self._active_dwa_config.max_angular_velocity
            ),
            "max_linear_accel": float(self._active_dwa_config.max_linear_accel),
            "path_deviation_limit": float(
                self._active_dwa_config.path_deviation_limit
            ),
        }

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


DWAExecutor = DwaNavExecutor
NavExecutor = DwaNavExecutor


__all__ = [
    "DWAExecutor",
    "DwaNavExecutor",
    "NavExecutor",
]
