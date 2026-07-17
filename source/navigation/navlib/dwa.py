from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid_map import OccupancyGridMap


@dataclass(frozen=True)
class DWAConfig:
    """Dynamic Window Approach configuration for vx + wz control."""

    control_dt: float
    lookahead_distance: float = 0.8
    waypoint_tolerance: float = 0.2
    goal_tolerance: float = 0.35
    prediction_horizon: float = 1.8
    integration_dt: float = 0.1
    max_linear_velocity: float = 0.5
    min_linear_velocity: float = 0.0
    min_active_linear_velocity: float = 0.30
    min_active_angular_velocity: float = 0.0
    max_angular_velocity: float = 1.0
    max_linear_accel: float = 2.5
    max_angular_accel: float = 3.0
    linear_samples: int = 7
    angular_samples: int = 13
    clearance_bias: float = 0.18
    heading_bias: float = 0.55
    path_bias: float = 0.9
    trajectory_path_bias: float = 1.1
    path_deviation_penalty_bias: float = 1.6
    progress_bias: float = 1.6
    speed_bias: float = 0.35
    obstacle_distance_cap: float = 0.5
    rotate_in_place_angle: float = 1.05
    # Optional lower release threshold for the rotate-in-place gate.  Keeping
    # the entry and exit thresholds separate prevents a locomotion policy from
    # alternating between turn and forward gait when heading error jitters at
    # one threshold.
    rotate_in_place_exit_angle: float | None = None
    # When configured, emit an explicit zero command after heading alignment
    # until measured angular velocity falls below this threshold.  This resets
    # command-window momentum before forward gait is allowed.
    rotate_in_place_settle_angular_velocity: float | None = None
    close_goal_rotate_in_place_angle: float | None = None
    close_goal_rotate_in_place_distance: float | None = None
    large_heading_creep_velocity: float | None = None
    close_goal_large_heading_creep_velocity: float | None = None
    close_goal_distance: float = 0.45
    close_goal_speed_limit: float = 0.22
    goal_tracking_distance: float = 0.80
    near_goal_min_active_linear_velocity: float = 0.22
    near_goal_force_forward_heading_angle: float = 0.45
    path_sample_spacing: float = 0.05
    path_deviation_limit: float = 0.18
    path_distance_window: int = 80
    use_command_velocity_window: bool = False
    # Some locomotion policies intentionally stand still for small non-zero
    # velocity commands.  When enabled, forward tracking samples either zero or
    # an actually executable gait command instead of lingering in that deadband.
    enforce_min_active_linear_velocity: bool = False
    enforce_min_active_angular_velocity: bool = False
    # Pure rotation may require a minimum command to overcome the locomotion
    # policy deadband, while forward path tracking benefits from continuous
    # small yaw corrections.  This switch separates those two regimes.
    enforce_min_active_angular_velocity_only_during_rotation: bool = False
    enforce_path_deviation_limit: bool = False
    initial_alignment_path_deviation_limit: float | None = None
    path_recovery_deviation_limit: float | None = None
    near_goal_path_deviation_limit: float | None = None
    near_goal_path_deviation_distance: float | None = None
    preserve_sharp_corners: bool = False
    corner_angle_threshold: float = 0.45
    corner_waypoint_tolerance: float = 0.08


@dataclass(frozen=True)
class DWADebug:
    target_index: int
    target_point: tuple[float, float]
    distance_to_target: float
    distance_to_goal: float
    heading_error: float
    clearance: float
    score: float
    reached_goal: bool
    near_goal_tracking: bool
    sampled_candidates: int
    feasible_candidates: int
    collision_rejections: int
    path_deviation_rejections: int
    best_linear_velocity: float
    best_angular_velocity: float
    path_distance: float
    path_deviation_limit_used: float
    initial_alignment_active: bool
    path_recovery_active: bool
    window_linear_velocity: float
    window_angular_velocity: float
    velocity_window_source: str
    rotate_in_place_angle_used: float
    rotate_in_place_exit_angle_used: float
    rotation_gate_active: bool
    rotation_settle_active: bool
    path_anchor_applied: bool = False
    path_anchor_index: int = 0
    path_anchor_distance: float = 0.0
    path_anchor_progress_m: float = 0.0


class DWAController:
    """Local planner that samples dynamically feasible vx + wz commands."""

    def __init__(self, path_world: list[tuple[float, float]], grid_map: OccupancyGridMap, config: DWAConfig):
        if len(path_world) < 2:
            raise ValueError("DWA requires at least two world-frame waypoints.")
        self.reference_path_world = np.asarray(path_world, dtype=np.float64)
        sample_spacing = max(grid_map.resolution, min(config.path_sample_spacing, max(config.lookahead_distance * 0.5, grid_map.resolution)))
        self.path_world = _densify_path(self.reference_path_world, sample_spacing=sample_spacing)
        segment_lengths = np.linalg.norm(
            np.diff(self.path_world, axis=0),
            axis=1,
        )
        self._path_cumulative_lengths = np.concatenate(
            [np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
        )
        self.grid_map = grid_map
        self.config = config
        self.target_index = 1
        self._command_window_velocity: tuple[float, float] | None = None
        self._corner_indices = (
            _find_sharp_corner_indices(
                self.path_world,
                angle_threshold=float(config.corner_angle_threshold),
            )
            if config.preserve_sharp_corners
            else ()
        )
        self._passed_corner_indices: set[int] = set()
        self._initial_alignment_active = (
            config.initial_alignment_path_deviation_limit is not None
        )
        self._rotation_gate_active = False
        self._rotation_settle_active = False
        self._path_recovery_active = False
        self._path_anchor_applied = False
        self._path_anchor_index = 0
        self._path_anchor_distance = 0.0
        self._path_anchor_progress_m = 0.0
        self.grid_map.obstacle_distance_map()

    def compute_command(
        self,
        pose_xyyaw: tuple[float, float, float],
        current_velocity: tuple[float, float],
    ) -> tuple[np.ndarray, DWADebug]:
        x, y, yaw = pose_xyyaw
        current_vx, current_wz = current_velocity
        measured_current_wz = float(current_wz)
        if self.config.use_command_velocity_window:
            # RL policy 对微小速度命令存在响应死区；用上一条高层命令推进窗口，
            # 避免实测速度尚未响应时每次都把加速过程重置到零附近。
            if self._command_window_velocity is None:
                self._command_window_velocity = (
                    float(
                        np.clip(
                            current_vx,
                            self.config.min_linear_velocity,
                            self.config.max_linear_velocity,
                        )
                    ),
                    float(
                        np.clip(
                            current_wz,
                            -self.config.max_angular_velocity,
                            self.config.max_angular_velocity,
                        )
                    ),
                )
            current_vx, current_wz = self._command_window_velocity
        window_vx = float(current_vx)
        window_wz = float(current_wz)
        position = np.array([x, y], dtype=np.float64)

        distance_to_goal = float(np.linalg.norm(self.path_world[-1] - position))
        if distance_to_goal <= self.config.goal_tolerance:
            debug = DWADebug(
                target_index=len(self.path_world) - 1,
                target_point=(float(self.path_world[-1][0]), float(self.path_world[-1][1])),
                distance_to_target=0.0,
                distance_to_goal=distance_to_goal,
                heading_error=0.0,
                clearance=self.config.obstacle_distance_cap,
                score=0.0,
                reached_goal=True,
                near_goal_tracking=False,
                sampled_candidates=0,
                feasible_candidates=0,
                collision_rejections=0,
                path_deviation_rejections=0,
                best_linear_velocity=0.0,
                best_angular_velocity=0.0,
                path_distance=0.0,
                path_deviation_limit_used=float(self.config.path_deviation_limit),
                initial_alignment_active=False,
                path_recovery_active=False,
                window_linear_velocity=window_vx,
                window_angular_velocity=window_wz,
                velocity_window_source=(
                    "command" if self.config.use_command_velocity_window else "measured"
                ),
                rotate_in_place_angle_used=self._rotate_in_place_angle(
                    distance_to_goal
                ),
                rotate_in_place_exit_angle_used=self._rotate_in_place_exit_angle(
                    self._rotate_in_place_angle(distance_to_goal)
                ),
                rotation_gate_active=False,
                rotation_settle_active=False,
                path_anchor_applied=self._path_anchor_applied,
                path_anchor_index=self._path_anchor_index,
                path_anchor_distance=self._path_anchor_distance,
                path_anchor_progress_m=self._path_anchor_progress_m,
            )
            self._command_window_velocity = (0.0, 0.0)
            return np.zeros(3, dtype=np.float32), debug

        near_goal_tracking = False
        self._advance_target(position)
        target_index = self.target_index
        target = self.path_world[target_index]
        delta = target - position
        distance_to_target = float(np.linalg.norm(delta))
        target_heading = math.atan2(delta[1], delta[0])
        heading_error = _wrap_angle(target_heading - yaw)
        rotate_in_place_angle = self._rotate_in_place_angle(distance_to_goal)
        rotate_in_place_exit_angle = self._rotate_in_place_exit_angle(
            rotate_in_place_angle
        )
        self._update_rotation_gate(
            heading_error=heading_error,
            enter_angle=rotate_in_place_angle,
            exit_angle=rotate_in_place_exit_angle,
            measured_angular_velocity=measured_current_wz,
        )
        current_path_distance = float(
            np.min(self._path_distances(position[None, :]))
        )
        if (
            self._initial_alignment_active
            and not self._rotation_gate_active
            and not self._rotation_settle_active
            and current_path_distance <= self.config.path_deviation_limit
        ):
            self._initial_alignment_active = False
            # The close-goal gate is only an initial-alignment policy.  Once
            # that one-shot latch is released, use the normal carry threshold
            # for both candidate sampling and diagnostics in this same tick.
            rotate_in_place_angle = self._rotate_in_place_angle(distance_to_goal)
            rotate_in_place_exit_angle = self._rotate_in_place_exit_angle(
                rotate_in_place_angle
            )
        if (
            not self._initial_alignment_active
            and self.config.path_recovery_deviation_limit is not None
        ):
            recovery_enter_limit = 0.90 * self.config.path_deviation_limit
            recovery_exit_limit = 0.80 * self.config.path_deviation_limit
            if current_path_distance > self.config.path_deviation_limit:
                self._path_recovery_active = True
            elif (
                not self._path_recovery_active
                and self.config.enforce_path_deviation_limit
                and current_path_distance >= recovery_enter_limit
            ):
                # 当前点还没越界，但 rollout 前进后可能被硬偏差阈值全部拒绝；
                # 提前进入恢复模式，避免在边界内侧反复选择零速度。
                self._path_recovery_active = True
            elif (
                self._path_recovery_active
                and current_path_distance <= recovery_exit_limit
            ):
                self._path_recovery_active = False
        path_deviation_limit = float(self.config.path_deviation_limit)
        if self._initial_alignment_active:
            path_deviation_limit = max(
                path_deviation_limit,
                float(self.config.initial_alignment_path_deviation_limit),
            )
        elif self._path_recovery_active:
            path_deviation_limit = max(
                path_deviation_limit,
                float(self.config.path_recovery_deviation_limit),
            )
        if (
            self.config.near_goal_path_deviation_limit is not None
            and self.config.near_goal_path_deviation_distance is not None
            and distance_to_goal <= self.config.near_goal_path_deviation_distance
        ):
            # 近目标阶段优先让机器人收敛到真实 goal，同时仍保留碰撞检查。
            path_deviation_limit = max(
                path_deviation_limit,
                float(self.config.near_goal_path_deviation_limit),
            )

        best_command = np.zeros(3, dtype=np.float32)
        best_score = -float("inf")
        best_clearance = 0.0
        best_path_distance = float("inf")
        sampled_candidates = 0
        feasible_candidates = 0
        collision_rejections = 0
        path_deviation_rejections = 0

        for linear_velocity, angular_velocity in self._sample_velocities(
            current_vx=current_vx,
            current_wz=current_wz,
            distance_to_goal=distance_to_goal,
            heading_error=heading_error,
            rotation_gate_active=self._rotation_gate_active,
            rotation_settle_active=self._rotation_settle_active,
        ):
            sampled_candidates += 1
            trajectory = self._rollout(
                x=x,
                y=y,
                yaw=yaw,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                stop_xy=self.path_world[-1],
                stop_tolerance=self.config.goal_tolerance,
            )
            if trajectory.size == 0:
                continue

            clearance = self._trajectory_clearance(trajectory)
            if clearance <= 0.0:
                collision_rejections += 1
                continue

            end_pose = trajectory[-1]
            score, details = self._score_trajectory(
                start_position=position,
                trajectory=trajectory,
                end_pose=end_pose,
                target=target,
                linear_velocity=linear_velocity,
                clearance=clearance,
            )
            if (
                self.config.enforce_path_deviation_limit
                and details["max_path_distance"]
                > path_deviation_limit
            ):
                path_deviation_rejections += 1
                continue
            feasible_candidates += 1
            if score > best_score:
                best_score = score
                best_clearance = clearance
                best_path_distance = float(details["mean_path_distance"])
                best_command = np.array([linear_velocity, 0.0, angular_velocity], dtype=np.float32)

        if not np.isfinite(best_score):
            dt = max(self.config.control_dt, 1.0e-3)
            angular_lower, angular_upper = _bounded_dynamic_interval(
                current_value=current_wz,
                minimum_value=-self.config.max_angular_velocity,
                maximum_value=self.config.max_angular_velocity,
                max_acceleration=self.config.max_angular_accel,
                dt=dt,
            )
            angular_velocity = np.clip(
                1.5 * heading_error,
                angular_lower,
                angular_upper,
            )
            best_command = np.array([0.0, 0.0, angular_velocity], dtype=np.float32)
            best_score = -1.0
            best_clearance = 0.0
            best_path_distance = float(np.min(self._path_distances(position[None, :])))

        if self.config.use_command_velocity_window:
            self._command_window_velocity = (
                float(best_command[0]),
                float(best_command[2]),
            )
        debug = DWADebug(
            target_index=target_index,
            target_point=(float(target[0]), float(target[1])),
            distance_to_target=distance_to_target,
            distance_to_goal=distance_to_goal,
            heading_error=heading_error,
            clearance=best_clearance,
            score=best_score,
            reached_goal=False,
            near_goal_tracking=near_goal_tracking,
            sampled_candidates=sampled_candidates,
            feasible_candidates=feasible_candidates,
            collision_rejections=collision_rejections,
            path_deviation_rejections=path_deviation_rejections,
            best_linear_velocity=float(best_command[0]),
            best_angular_velocity=float(best_command[2]),
            path_distance=best_path_distance,
            path_deviation_limit_used=path_deviation_limit,
            initial_alignment_active=self._initial_alignment_active,
            path_recovery_active=self._path_recovery_active,
            window_linear_velocity=window_vx,
            window_angular_velocity=window_wz,
            velocity_window_source=(
                "command" if self.config.use_command_velocity_window else "measured"
            ),
            rotate_in_place_angle_used=rotate_in_place_angle,
            rotate_in_place_exit_angle_used=rotate_in_place_exit_angle,
            rotation_gate_active=self._rotation_gate_active,
            rotation_settle_active=self._rotation_settle_active,
            path_anchor_applied=self._path_anchor_applied,
            path_anchor_index=self._path_anchor_index,
            path_anchor_distance=self._path_anchor_distance,
            path_anchor_progress_m=self._path_anchor_progress_m,
        )
        return best_command, debug

    def _rotate_in_place_angle(self, distance_to_goal: float) -> float:
        """Return the heading gate used at the current goal distance.

        Short carry paths need a stricter heading gate than long-route corner
        tracking.  Otherwise the minimum executable forward gait starts while
        the goal is still far off the body x-axis, and the quadruped can arc
        past a nearby goal before terminal pose control becomes active.
        """

        angle = float(self.config.rotate_in_place_angle)
        close_angle = self.config.close_goal_rotate_in_place_angle
        close_distance = self.config.close_goal_rotate_in_place_distance
        if (
            self._initial_alignment_active
            and close_angle is not None
            and close_distance is not None
            and distance_to_goal <= float(close_distance)
        ):
            angle = min(angle, float(close_angle))
        return angle

    def _rotate_in_place_exit_angle(self, enter_angle: float) -> float:
        """Return the lower threshold that releases pure rotation."""

        configured = self.config.rotate_in_place_exit_angle
        if configured is None:
            return float(enter_angle)
        return min(float(enter_angle), max(0.0, float(configured)))

    def _update_rotation_gate(
        self,
        *,
        heading_error: float,
        enter_angle: float,
        exit_angle: float,
        measured_angular_velocity: float,
    ) -> None:
        """Apply hysteresis to heading alignment before forward tracking."""

        absolute_error = abs(float(heading_error))
        settle_threshold = self.config.rotate_in_place_settle_angular_velocity
        if self._rotation_settle_active:
            if (
                settle_threshold is None
                or abs(float(measured_angular_velocity))
                <= max(0.0, float(settle_threshold))
            ):
                self._rotation_settle_active = False
                if absolute_error > float(enter_angle):
                    self._rotation_gate_active = True
            return
        if self._rotation_gate_active:
            if absolute_error <= float(exit_angle):
                self._rotation_gate_active = False
                if settle_threshold is not None:
                    # Always spend at least this command tick braking.  Besides
                    # physical angular momentum, this clears the high previous
                    # command used by ``use_command_velocity_window``.
                    self._rotation_settle_active = True
            return
        if absolute_error > float(enter_angle):
            self._rotation_gate_active = True

    def _advance_target(self, position: np.ndarray):
        if not self._path_anchor_applied:
            self._anchor_target_to_live_position(position)
        next_corner = self._next_unpassed_corner()
        path_slice_start = self.target_index
        search_window = max(
            3,
            int(math.ceil(self.config.lookahead_distance / max(self.grid_map.resolution, 1.0e-3))) + 3,
        )
        path_slice_end = min(len(self.path_world), path_slice_start + search_window)
        path_slice = self.path_world[path_slice_start:path_slice_end]
        nearest_offset = int(np.argmin(np.linalg.norm(path_slice - position, axis=1)))
        nearest_index = min(len(self.path_world) - 1, path_slice_start + nearest_offset)
        if next_corner is not None:
            nearest_index = min(nearest_index, next_corner)
        self.target_index = max(self.target_index, nearest_index)

        while self.target_index < len(self.path_world) - 1:
            target = self.path_world[self.target_index]
            is_corner = self.target_index == next_corner
            tolerance = (
                self.config.corner_waypoint_tolerance
                if is_corner
                else self.config.waypoint_tolerance
            )
            if np.linalg.norm(target - position) > tolerance:
                break
            passed_index = self.target_index
            self.target_index += 1
            if is_corner:
                self._passed_corner_indices.add(passed_index)
                # 切换下一段后先重新计算朝向，禁止一次跳过整个拐角。
                return

        while self.target_index < len(self.path_world) - 1:
            next_corner = self._next_unpassed_corner()
            if next_corner is not None and self.target_index >= next_corner:
                break
            target = self.path_world[self.target_index]
            if np.linalg.norm(target - position) >= self.config.lookahead_distance:
                break
            self.target_index += 1

    def _anchor_target_to_live_position(self, position: np.ndarray) -> None:
        """Project the first live pose onto the route before choosing lookahead.

        A plan may be generated one or more ticks before execution, and global
        planner endpoints may be snapped to a different raster.  Starting from
        hard-coded path index 1 can therefore command the robot back toward an
        already-passed or behind-start waypoint.  The projection is applied once
        and all later target progress remains monotonic.
        """

        starts = self.path_world[:-1]
        segments = self.path_world[1:] - starts
        squared_lengths = np.sum(segments * segments, axis=1)
        relative = position[None, :] - starts
        alphas = np.zeros(len(segments), dtype=np.float64)
        valid = squared_lengths > 1.0e-12
        alphas[valid] = np.clip(
            np.sum(relative[valid] * segments[valid], axis=1)
            / squared_lengths[valid],
            0.0,
            1.0,
        )
        projections = starts + alphas[:, None] * segments
        distances = np.linalg.norm(projections - position[None, :], axis=1)
        minimum_distance = float(np.min(distances))
        # If a route crosses itself, prefer the earliest equal-distance
        # projection.  This prevents an initial jump to a future loop branch.
        candidates = np.flatnonzero(distances <= minimum_distance + 1.0e-9)
        segment_index = int(candidates[0])
        segment_length = math.sqrt(float(squared_lengths[segment_index]))
        progress = float(
            self._path_cumulative_lengths[segment_index]
            + alphas[segment_index] * segment_length
        )
        desired_progress = progress + max(
            float(self.config.lookahead_distance),
            float(self.config.waypoint_tolerance),
        )
        target_index = int(
            np.searchsorted(
                self._path_cumulative_lengths,
                desired_progress,
                side="left",
            )
        )
        target_index = min(
            len(self.path_world) - 1,
            max(segment_index + 1, target_index, 1),
        )
        if self.config.preserve_sharp_corners:
            next_corner = next(
                (
                    index
                    for index in self._corner_indices
                    if self._path_cumulative_lengths[index] >= progress - 1.0e-9
                ),
                None,
            )
            if next_corner is not None:
                target_index = min(target_index, next_corner)
        self.target_index = max(self.target_index, target_index)
        self._passed_corner_indices.update(
            index for index in self._corner_indices if index < self.target_index
        )
        self._path_anchor_applied = True
        self._path_anchor_index = segment_index
        self._path_anchor_distance = minimum_distance
        self._path_anchor_progress_m = progress

    def _next_unpassed_corner(self) -> int | None:
        """返回尚未经过的下一个锐角路径点。"""

        for index in self._corner_indices:
            if index >= self.target_index and index not in self._passed_corner_indices:
                return index
        return None

    def _sample_velocities(
        self,
        current_vx: float,
        current_wz: float,
        distance_to_goal: float,
        heading_error: float,
        rotation_gate_active: bool | None = None,
        rotation_settle_active: bool | None = None,
    ) -> list[tuple[float, float]]:
        if rotation_settle_active:
            return [(0.0, 0.0)]
        dt = max(self.config.control_dt, 1.0e-3)
        linear_cap = self.config.max_linear_velocity
        min_active_linear_velocity = self.config.min_active_linear_velocity
        if distance_to_goal <= self.config.close_goal_distance:
            linear_cap = min(linear_cap, self.config.close_goal_speed_limit)
            min_active_linear_velocity = min(min_active_linear_velocity, linear_cap)
        elif distance_to_goal <= self.config.goal_tracking_distance:
            min_active_linear_velocity = self.config.near_goal_min_active_linear_velocity
            min_active_linear_velocity = min(min_active_linear_velocity, linear_cap)
        linear_lower, linear_upper = _bounded_dynamic_interval(
            current_value=current_vx,
            minimum_value=self.config.min_linear_velocity,
            maximum_value=linear_cap,
            max_acceleration=self.config.max_linear_accel,
            dt=dt,
        )
        angular_lower, angular_upper = _bounded_dynamic_interval(
            current_value=current_wz,
            minimum_value=-self.config.max_angular_velocity,
            maximum_value=self.config.max_angular_velocity,
            max_acceleration=self.config.max_angular_accel,
            dt=dt,
        )

        if rotation_gate_active is None:
            # Preserve the helper's historical standalone behaviour for tests
            # and diagnostics that sample a window without calling
            # ``compute_command`` first.  Normal control passes the hysteretic
            # latch explicitly.
            rotation_gate_active = (
                abs(heading_error)
                > self._rotate_in_place_angle(distance_to_goal)
            )
        if rotation_gate_active:
            configured_creep_velocity = self.config.large_heading_creep_velocity
            if (
                self._initial_alignment_active
                and self.config.close_goal_large_heading_creep_velocity is not None
                and self.config.close_goal_rotate_in_place_distance is not None
                and distance_to_goal
                <= float(self.config.close_goal_rotate_in_place_distance)
            ):
                configured_creep_velocity = (
                    self.config.close_goal_large_heading_creep_velocity
                )
            creep_velocity = (
                max(0.08, 0.5 * min_active_linear_velocity)
                if configured_creep_velocity is None
                else max(0.0, float(configured_creep_velocity))
            )
            creep_cap = min(
                linear_upper,
                max(
                    linear_lower,
                    min(
                        linear_cap,
                        creep_velocity,
                    ),
                ),
            )
            stopped = float(np.clip(0.0, linear_lower, linear_upper))
            linear_values = np.unique(
                np.round(
                    np.array(
                        [
                            stopped,
                            0.5 * (stopped + creep_cap),
                            creep_cap,
                        ],
                        dtype=np.float64,
                    ),
                    decimals=4,
                )
            )
        else:
            if self.config.enforce_min_active_linear_velocity:
                # The high-level velocity is an RL command rather than a direct
                # actuator target.  Commands below the policy's gait threshold
                # produce no translation, so treating them as dynamically
                # feasible only traps the command window in the deadband.  Keep
                # an explicit stop candidate, but evaluate moving candidates at
                # or above the executable threshold with the normal collision
                # rollout and scoring below.
                active_lower = max(linear_lower, min_active_linear_velocity)
                active_upper = max(linear_upper, active_lower)
                active_upper = min(active_upper, linear_cap)
                active_lower = min(active_lower, active_upper)
                linear_values = np.linspace(
                    active_lower,
                    active_upper,
                    num=max(self.config.linear_samples, 2),
                    dtype=np.float64,
                )
                linear_values = np.concatenate(
                    [linear_values, np.array([0.0], dtype=np.float64)]
                )
            else:
                linear_values = np.linspace(
                    linear_lower,
                    linear_upper,
                    num=max(self.config.linear_samples, 2),
                    dtype=np.float64,
                )
                linear_values = np.concatenate(
                    [linear_values, np.array([min_active_linear_velocity], dtype=np.float64)]
                )
                linear_values = np.clip(linear_values, linear_lower, linear_upper)
            linear_values = np.unique(
                np.round(
                    np.concatenate(
                        [
                            linear_values,
                            np.array(
                                [
                                    np.clip(0.0, linear_lower, linear_upper),
                                ]
                            ),
                        ]
                    ),
                    decimals=4,
                )
            )

        angular_values = np.linspace(
            angular_lower,
            angular_upper,
            num=max(self.config.angular_samples, 3),
            dtype=np.float64,
        )
        angular_values = np.unique(
            np.round(
                np.concatenate(
                    [
                        angular_values,
                        np.array(
                            [
                                0.0,
                                -self.config.max_angular_velocity,
                                self.config.max_angular_velocity,
                            ]
                        ),
                    ]
                ),
                decimals=4,
            )
        )
        angular_values = np.clip(
            angular_values,
            angular_lower,
            angular_upper,
        )
        angular_values = np.unique(np.round(angular_values, decimals=4))
        if (
            self.config.enforce_min_active_angular_velocity
            and (
                not self.config.enforce_min_active_angular_velocity_only_during_rotation
                or bool(rotation_gate_active)
            )
        ):
            angular_floor = min(
                max(0.0, self.config.min_active_angular_velocity),
                self.config.max_angular_velocity,
            )
            if angular_floor > 0.0:
                # As with the linear command, preserve an explicit stop while
                # removing angular commands that the policy treats as zero.
                # A sign change must pass through stop: once an active turn is
                # established we only expose its current sign plus zero.
                active_values = angular_values[
                    np.abs(angular_values) >= angular_floor - 1.0e-9
                ]
                if abs(current_wz) < angular_floor:
                    floor_values = np.array(
                        [-angular_floor, 0.0, angular_floor],
                        dtype=np.float64,
                    )
                else:
                    floor_values = np.array(
                        [0.0, math.copysign(angular_floor, current_wz)],
                        dtype=np.float64,
                    )
                angular_values = np.unique(
                    np.round(
                        np.concatenate([active_values, floor_values]),
                        decimals=4,
                    )
                )

        return [(float(v), float(w)) for v in linear_values for w in angular_values]

    def _rollout(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        linear_velocity: float,
        angular_velocity: float,
        stop_xy: np.ndarray | None = None,
        stop_tolerance: float = 0.0,
    ) -> np.ndarray:
        """Predict a command trajectory, stopping once it reaches the goal.

        A grasp base goal is often intentionally close to a table or wall.
        Continuing collision checks past an already-reached goal incorrectly
        rejects valid approach commands because their prediction horizon enters
        the obstacle behind the goal.
        """

        dt = max(self.config.integration_dt, 1.0e-3)
        horizon = max(self.config.prediction_horizon, dt)
        steps = max(1, int(math.ceil(horizon / dt)))
        trajectory = np.zeros((steps, 3), dtype=np.float64)

        sim_x = x
        sim_y = y
        sim_yaw = yaw
        for i in range(steps):
            sim_x += linear_velocity * math.cos(sim_yaw) * dt
            sim_y += linear_velocity * math.sin(sim_yaw) * dt
            sim_yaw = _wrap_angle(sim_yaw + angular_velocity * dt)
            trajectory[i] = (sim_x, sim_y, sim_yaw)
            if stop_xy is not None and math.hypot(sim_x - float(stop_xy[0]), sim_y - float(stop_xy[1])) <= stop_tolerance:
                return trajectory[: i + 1]
        return trajectory

    def _trajectory_clearance(self, trajectory: np.ndarray) -> float:
        min_clearance = self.config.obstacle_distance_cap
        for point in trajectory:
            clearance = self._clearance_at(point[0], point[1])
            if clearance <= 0.0:
                return 0.0
            min_clearance = min(min_clearance, clearance)
        return min_clearance

    def _clearance_at(self, x: float, y: float) -> float:
        row, col = self.grid_map.world_to_grid(x, y)
        distance = self.grid_map.distance_to_obstacle(row, col)
        if distance is not None:
            return min(float(distance), self.config.obstacle_distance_cap)
        return self._clearance_at_slow(row, col)

    def _clearance_at_slow(self, row: int, col: int) -> float:
        if self.grid_map.is_occupied(row, col):
            return 0.0
        cap_cells = max(1, int(math.ceil(self.config.obstacle_distance_cap / self.grid_map.resolution)))
        row_min = max(0, row - cap_cells)
        row_max = min(self.grid_map.height - 1, row + cap_cells)
        col_min = max(0, col - cap_cells)
        col_max = min(self.grid_map.width - 1, col + cap_cells)
        window = self.grid_map.occupancy[row_min : row_max + 1, col_min : col_max + 1]
        occupied = np.argwhere(window)
        if occupied.size == 0:
            return self.config.obstacle_distance_cap

        occupied[:, 0] += row_min
        occupied[:, 1] += col_min
        distances_cells = np.sqrt((occupied[:, 0] - row) ** 2 + (occupied[:, 1] - col) ** 2)
        return float(np.min(distances_cells) * self.grid_map.resolution)

    def _score_trajectory(
        self,
        *,
        start_position: np.ndarray,
        trajectory: np.ndarray,
        end_pose: np.ndarray,
        target: np.ndarray,
        linear_velocity: float,
        clearance: float,
    ) -> tuple[float, dict[str, float]]:
        end_position = end_pose[:2]
        heading_to_target = math.atan2(target[1] - end_position[1], target[0] - end_position[0])
        heading_error = abs(_wrap_angle(heading_to_target - end_pose[2]))

        heading_score = 0.5 * (math.cos(heading_error) + 1.0)
        path_positions = trajectory[:, :2]
        path_distances = self._path_distances(path_positions)
        mean_path_distance = float(np.mean(path_distances))
        max_path_distance = float(np.max(path_distances))
        end_path_distance = float(path_distances[-1])
        path_score = 1.0 / (1.0 + end_path_distance)
        trajectory_path_score = 1.0 / (1.0 + mean_path_distance)
        start_target_distance = float(np.linalg.norm(target - start_position))
        end_target_distance = float(np.linalg.norm(target - end_position))
        progress = max(0.0, start_target_distance - end_target_distance)
        progress_score = progress / max(self.config.max_linear_velocity * self.config.prediction_horizon, 1.0e-6)
        clearance_score = min(clearance / max(self.config.obstacle_distance_cap, 1.0e-6), 1.0)
        speed_score = linear_velocity / max(self.config.max_linear_velocity, 1.0e-6)
        path_deviation_excess = max(0.0, max_path_distance - self.config.path_deviation_limit)
        path_deviation_penalty = path_deviation_excess / max(self.config.path_deviation_limit, 1.0e-6)

        score = (
            self.config.progress_bias * progress_score
            + self.config.heading_bias * heading_score
            + self.config.path_bias * path_score
            + self.config.trajectory_path_bias * trajectory_path_score
            + self.config.clearance_bias * clearance_score
            + self.config.speed_bias * speed_score
            - self.config.path_deviation_penalty_bias * path_deviation_penalty
        )
        return score, {
            "progress": progress,
            "progress_score": progress_score,
            "heading_score": heading_score,
            "path_score": path_score,
            "trajectory_path_score": trajectory_path_score,
            "mean_path_distance": mean_path_distance,
            "max_path_distance": max_path_distance,
            "path_deviation_penalty": path_deviation_penalty,
            "clearance_score": clearance_score,
            "speed_score": speed_score,
        }

    def _path_distances(self, positions: np.ndarray) -> np.ndarray:
        window_start = max(0, self.target_index - 5)
        window_end = min(len(self.path_world), self.target_index + max(1, int(self.config.path_distance_window)))
        if window_end <= window_start:
            window_end = len(self.path_world)
        path_slice = self.path_world[window_start:window_end]
        deltas = positions[:, None, :] - path_slice[None, :, :]
        return np.min(np.linalg.norm(deltas, axis=2), axis=1)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _find_sharp_corner_indices(
    path_world: np.ndarray,
    *,
    angle_threshold: float,
) -> tuple[int, ...]:
    """找出需要显式停靠并转向的锐角路径点。"""

    threshold = max(0.0, float(angle_threshold))
    corners: list[int] = []
    for index in range(1, len(path_world) - 1):
        incoming = path_world[index] - path_world[index - 1]
        outgoing = path_world[index + 1] - path_world[index]
        if np.linalg.norm(incoming) <= 1.0e-9 or np.linalg.norm(outgoing) <= 1.0e-9:
            continue
        incoming_yaw = math.atan2(float(incoming[1]), float(incoming[0]))
        outgoing_yaw = math.atan2(float(outgoing[1]), float(outgoing[0]))
        if abs(_wrap_angle(outgoing_yaw - incoming_yaw)) >= threshold:
            corners.append(index)
    return tuple(corners)


def _bounded_dynamic_interval(
    *,
    current_value: float,
    minimum_value: float,
    maximum_value: float,
    max_acceleration: float,
    dt: float,
) -> tuple[float, float]:
    """把动态窗口与合法命令范围求交；无交集时饱和到最近合法边界。"""

    lower = max(float(minimum_value), float(current_value) - float(max_acceleration) * float(dt))
    upper = min(float(maximum_value), float(current_value) + float(max_acceleration) * float(dt))
    if lower <= upper:
        return lower, upper
    saturated = float(np.clip(current_value, minimum_value, maximum_value))
    return saturated, saturated


def _densify_path(path_world: np.ndarray, sample_spacing: float) -> np.ndarray:
    if len(path_world) < 2:
        return path_world.copy()

    dense_points: list[np.ndarray] = [path_world[0]]
    spacing = max(sample_spacing, 1.0e-3)

    for index in range(1, len(path_world)):
        start = path_world[index - 1]
        end = path_world[index]
        delta = end - start
        segment_length = float(np.linalg.norm(delta))
        if segment_length <= 1.0e-9:
            continue
        samples = max(1, int(math.ceil(segment_length / spacing)))
        for step in range(1, samples + 1):
            alpha = min(1.0, step / samples)
            dense_points.append(start + alpha * delta)

    return np.asarray(dense_points, dtype=np.float64)
