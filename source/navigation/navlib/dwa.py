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
    close_goal_distance: float = 0.45
    close_goal_speed_limit: float = 0.22
    goal_tracking_distance: float = 0.80
    near_goal_min_active_linear_velocity: float = 0.22
    near_goal_force_forward_heading_angle: float = 0.45
    path_sample_spacing: float = 0.05
    path_deviation_limit: float = 0.18


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
    best_linear_velocity: float
    best_angular_velocity: float
    path_distance: float


class DWAController:
    """Local planner that samples dynamically feasible vx + wz commands."""

    def __init__(self, path_world: list[tuple[float, float]], grid_map: OccupancyGridMap, config: DWAConfig):
        if len(path_world) < 2:
            raise ValueError("DWA requires at least two world-frame waypoints.")
        self.reference_path_world = np.asarray(path_world, dtype=np.float64)
        sample_spacing = max(grid_map.resolution, min(config.path_sample_spacing, max(config.lookahead_distance * 0.5, grid_map.resolution)))
        self.path_world = _densify_path(self.reference_path_world, sample_spacing=sample_spacing)
        self.grid_map = grid_map
        self.config = config
        self.target_index = 1

    def compute_command(
        self,
        pose_xyyaw: tuple[float, float, float],
        current_velocity: tuple[float, float],
    ) -> tuple[np.ndarray, DWADebug]:
        x, y, yaw = pose_xyyaw
        current_vx, current_wz = current_velocity
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
                best_linear_velocity=0.0,
                best_angular_velocity=0.0,
                path_distance=0.0,
            )
            return np.zeros(3, dtype=np.float32), debug

        near_goal_tracking = False
        self._advance_target(position)
        target_index = self.target_index
        target = self.path_world[target_index]
        delta = target - position
        distance_to_target = float(np.linalg.norm(delta))
        target_heading = math.atan2(delta[1], delta[0])
        heading_error = _wrap_angle(target_heading - yaw)

        best_command = np.zeros(3, dtype=np.float32)
        best_score = -float("inf")
        best_clearance = 0.0
        best_path_distance = float("inf")
        sampled_candidates = 0
        feasible_candidates = 0
        collision_rejections = 0

        for linear_velocity, angular_velocity in self._sample_velocities(
            current_vx=current_vx,
            current_wz=current_wz,
            distance_to_goal=distance_to_goal,
            heading_error=heading_error,
        ):
            sampled_candidates += 1
            trajectory = self._rollout(x=x, y=y, yaw=yaw, linear_velocity=linear_velocity, angular_velocity=angular_velocity)
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
            feasible_candidates += 1
            if score > best_score:
                best_score = score
                best_clearance = clearance
                best_path_distance = float(details["mean_path_distance"])
                best_command = np.array([linear_velocity, 0.0, angular_velocity], dtype=np.float32)

        if not np.isfinite(best_score):
            angular_velocity = np.clip(1.5 * heading_error, -self.config.max_angular_velocity, self.config.max_angular_velocity)
            best_command = np.array([0.0, 0.0, angular_velocity], dtype=np.float32)
            best_score = -1.0
            best_clearance = 0.0
            best_path_distance = float(np.min(self._path_distances(position[None, :])))

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
            best_linear_velocity=float(best_command[0]),
            best_angular_velocity=float(best_command[2]),
            path_distance=best_path_distance,
        )
        return best_command, debug

    def _advance_target(self, position: np.ndarray):
        path_slice_start = max(0, self.target_index - 1)
        path_slice = self.path_world[path_slice_start:]
        nearest_offset = int(np.argmin(np.linalg.norm(path_slice - position, axis=1)))
        self.target_index = min(len(self.path_world) - 1, path_slice_start + nearest_offset)

        while self.target_index < len(self.path_world) - 1:
            target = self.path_world[self.target_index]
            if np.linalg.norm(target - position) > self.config.waypoint_tolerance:
                break
            self.target_index += 1

        while self.target_index < len(self.path_world) - 1:
            target = self.path_world[self.target_index]
            if np.linalg.norm(target - position) >= self.config.lookahead_distance:
                break
            self.target_index += 1

    def _sample_velocities(
        self,
        current_vx: float,
        current_wz: float,
        distance_to_goal: float,
        heading_error: float,
    ) -> list[tuple[float, float]]:
        dt = max(self.config.control_dt, 1.0e-3)
        linear_lower = max(self.config.min_linear_velocity, current_vx - self.config.max_linear_accel * dt)
        linear_upper = min(self.config.max_linear_velocity, current_vx + self.config.max_linear_accel * dt)
        angular_lower = max(-self.config.max_angular_velocity, current_wz - self.config.max_angular_accel * dt)
        angular_upper = min(self.config.max_angular_velocity, current_wz + self.config.max_angular_accel * dt)

        if abs(heading_error) > self.config.rotate_in_place_angle:
            linear_values = np.array([0.0], dtype=np.float64)
        else:
            linear_values = np.linspace(
                linear_lower,
                linear_upper,
                num=max(self.config.linear_samples, 2),
                dtype=np.float64,
            )
            linear_values = np.concatenate(
                [linear_values, np.array([self.config.min_active_linear_velocity], dtype=np.float64)]
            )
            linear_values = np.clip(linear_values, self.config.min_linear_velocity, self.config.max_linear_velocity)
            linear_values = np.unique(np.round(np.concatenate([linear_values, np.array([0.0])]), decimals=4))

        angular_values = np.linspace(
            angular_lower,
            angular_upper,
            num=max(self.config.angular_samples, 3),
            dtype=np.float64,
        )
        angular_values = np.unique(
            np.round(
                np.concatenate([angular_values, np.array([0.0, -self.config.max_angular_velocity, self.config.max_angular_velocity])]),
                decimals=4,
            )
        )
        angular_values = np.clip(angular_values, -self.config.max_angular_velocity, self.config.max_angular_velocity)

        return [(float(v), float(w)) for v in linear_values for w in angular_values]

    def _rollout(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        linear_velocity: float,
        angular_velocity: float,
    ) -> np.ndarray:
        horizon = max(self.config.prediction_horizon, self.config.integration_dt)
        dt = max(self.config.integration_dt, 1.0e-3)
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
        path_slice = self.path_world[max(0, self.target_index - 2) :]
        deltas = positions[:, None, :] - path_slice[None, :, :]
        return np.min(np.linalg.norm(deltas, axis=2), axis=1)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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
