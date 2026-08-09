#!/usr/bin/env python3
"""复现 phase231 二楼走廊，验收 PCT 后缀可由 SCAN 正常规划。"""

from __future__ import annotations

import json
import math
from pathlib import Path as FilePath
import sys

from rclpy.node import Node
import rclpy

from probe_crossfloor_scan_floor_windows import (
    BODY_HEIGHT_M,
    GOAL_BASE_XYZ,
    GOAL_YAW,
    FloorWindowProbe,
    _cumulative_lengths,
    _evaluate_bspline,
    _production_backend,
    _sample_at_progress,
)


PHASE231_BASE_XYZ = (
    0.20307183265686035,
    4.693713188171387,
    3.383995771408081,
)
EXPECTED_PATH_POINT_COUNT = 31
CORRIDOR_HALF_WIDTH_M = 0.45
MAXIMUM_TRAJECTORY_DURATION_S = 4.0
MINIMUM_VELOCITY_UPPER_BOUND_MPS = 0.35
MINIMUM_TRAJECTORY_PROGRESS_M = 0.30


def _corridor_walls(
    start_ground: tuple[float, float, float],
    travel_yaw: float,
) -> tuple[tuple[float, float, float], ...]:
    """构造与 phase231 最小净空同量级的两侧竖直墙 hit。"""

    tangent = (math.cos(travel_yaw), math.sin(travel_yaw))
    normal = (-tangent[1], tangent[0])
    points: list[tuple[float, float, float]] = []
    for longitudinal_index in range(-8, 37):
        longitudinal = 0.05 * longitudinal_index
        center_x = start_ground[0] + longitudinal * tangent[0]
        center_y = start_ground[1] + longitudinal * tangent[1]
        for side in (-1.0, 1.0):
            wall_x = center_x + side * CORRIDOR_HALF_WIDTH_M * normal[0]
            wall_y = center_y + side * CORRIDOR_HALF_WIDTH_M * normal[1]
            for height_offset in (-0.20, 0.0, 0.20):
                points.append(
                    (
                        wall_x,
                        wall_y,
                        start_ground[2] + BODY_HEIGHT_M + height_offset,
                    )
                )
    return tuple(points)


def main() -> None:
    """发布真实 PCT F2 后缀和确定性窄走廊点云，等待正常 B-spline。"""

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("真实 upstream 扩展和 ROS 2 Humble 要求 Python 3.10")
    project_root = FilePath(__file__).resolve().parents[4]
    backend = _production_backend(project_root)
    plan = backend.plan(
        start_base_xyz=PHASE231_BASE_XYZ,
        goal_base_xyz=GOAL_BASE_XYZ,
        goal_yaw=GOAL_YAW,
    )
    points = tuple(
        tuple(float(value) for value in point)
        for point in plan.points_xyz
    )
    if len(points) != EXPECTED_PATH_POINT_COUNT:
        raise RuntimeError(
            "phase231 F2 PCT Path 点数发生未审计变化："
            f"expected={EXPECTED_PATH_POINT_COUNT}, actual={len(points)}"
        )
    if math.dist(points[0][:2], PHASE231_BASE_XYZ[:2]) > 1.0e-6:
        raise RuntimeError("phase231 F2 PCT Path 没有保持精确起点 XY")

    lengths = _cumulative_lengths(points)
    forward_point, _ = _sample_at_progress(
        points, lengths, min(1.20, lengths[-1])
    )
    travel_yaw = math.atan2(
        forward_point[1] - points[0][1],
        forward_point[0] - points[0][0],
    )
    occupied_points = _corridor_walls(points[0], travel_yaw)

    rclpy.init()
    node = Node("pct_scan_phase231_f2_corridor_probe")
    probe = FloorWindowProbe(
        node,
        occupied_cloud_points=occupied_points,
    )
    try:
        probe.wait_for_graph()
        # Path 发布前先让同一局部 GridMap 确实形成 hit 占据，禁止在障碍
        # 还没融合时抢跑出一条自由空间轨迹。
        probe.run_for(
            0.80,
            ground_point=points[0],
            yaw=travel_yaw,
            forward_speed_mps=0.0,
            path_stamp=None,
        )
        fused_hit_reports = [
            report
            for report in probe.map_diagnostics
            if report.map_fusion_performed
            and report.hit_endpoint_count > 0
            and report.free_to_occupied_transition_count > 0
        ]
        if not fused_hit_reports:
            raise RuntimeError("窄走廊 hit 尚未形成可认证占据，拒绝发布 Path")

        probe.publish_tombstone(points[0], travel_yaw, 0.0)
        _, path_stamp = probe.publish_path(points)
        probe.run_for(
            0.08,
            ground_point=points[0],
            yaw=travel_yaw,
            forward_speed_mps=0.0,
            path_stamp=None,
        )
        spline, diagnostic = probe.wait_for_trajectory(
            path_stamp=path_stamp,
            ground_point=points[0],
            yaw=travel_yaw,
            forward_speed_mps=0.0,
        )
        if (
            not diagnostic.ordered_reference_checked
            or not diagnostic.ordered_reference_safe
            or diagnostic.stationary
            or spline.emergency_stop
        ):
            raise RuntimeError("phase231 F2 窄走廊只产生了不可执行轨迹")
        if diagnostic.trajectory_duration > MAXIMUM_TRAJECTORY_DURATION_S:
            raise RuntimeError(
                "phase231 F2 局部轨迹仍然过慢："
                f"{diagnostic.trajectory_duration:.6f} s"
            )
        if (
            diagnostic.maximum_velocity_upper_bound
            < MINIMUM_VELOCITY_UPPER_BOUND_MPS
        ):
            raise RuntimeError(
                "phase231 F2 局部轨迹速度上界过低："
                f"{diagnostic.maximum_velocity_upper_bound:.6f} m/s"
            )
        trajectory_points = _evaluate_bspline(spline)
        maximum_progress = max(
            math.dist(sample[:2], points[0][:2])
            for sample in trajectory_points
        )
        if maximum_progress < MINIMUM_TRAJECTORY_PROGRESS_M:
            raise RuntimeError(
                "phase231 F2 B-spline 没有产生有效前进距离："
                f"{maximum_progress:.6f} m"
            )

        report = fused_hit_reports[-1]
        print(
            "PHASE231_F2_CORRIDOR_OK "
            + json.dumps(
                {
                    "path_points": len(points),
                    "path_length_m": lengths[-1],
                    "travel_yaw_rad": travel_yaw,
                    "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
                    "hit_endpoint_count": int(report.hit_endpoint_count),
                    "trajectory_duration_s": float(
                        diagnostic.trajectory_duration
                    ),
                    "velocity_upper_bound_mps": float(
                        diagnostic.maximum_velocity_upper_bound
                    ),
                    "maximum_progress_m": maximum_progress,
                    "traj_id": int(spline.traj_id),
                },
                sort_keys=True,
            )
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
