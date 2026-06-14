"""面向纯物理流水线的 A* 全局规划适配器。"""

from __future__ import annotations

import math
from pathlib import Path

from source.interfaces.navigation import NavGoal, NavPlan
from source.interfaces.simulation import SimulationState

from .navlib import AStarPlanner, OccupancyGridMap


class AStarNavPlanner:
    """把纯 Python 占用栅格 A* 适配为流水线导航规划接口。"""

    def __init__(
        self,
        map_json: str | Path | None = None,
        *,
        grid_map: OccupancyGridMap | None = None,
        inflate_radius: float = 0.0,
        allow_diagonal: bool = True,
        heuristic_weight: float = 1.0,
    ) -> None:
        if grid_map is None and map_json is None:
            raise ValueError("必须提供 map_json 或 grid_map。")
        if grid_map is not None and map_json is not None:
            raise ValueError("map_json 与 grid_map 只能提供一个。")
        if inflate_radius < 0.0:
            raise ValueError("inflate_radius 不能为负数。")

        self.map_json = None if map_json is None else str(Path(map_json).expanduser().resolve())
        self.raw_map = grid_map or OccupancyGridMap.from_meta_file(self.map_json)
        self.global_map = self.raw_map.inflate(float(inflate_radius))
        self.inflate_radius = float(inflate_radius)
        self.astar = AStarPlanner(
            allow_diagonal=bool(allow_diagonal),
            heuristic_weight=float(heuristic_weight),
        )

    def plan(self, state: SimulationState, goal: NavGoal) -> NavPlan:
        """从当前世界坐标规划到任务要求的精确目标坐标。"""

        start_xy = (float(state.robot_root_pose[0]), float(state.robot_root_pose[1]))
        goal_xy = (float(goal.x), float(goal.y))
        result = self.astar.plan(
            self.global_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            snap_to_free=False,
        )
        waypoints = self._with_exact_endpoints(
            start_xy,
            goal_xy,
            result.path_world,
        )
        metadata = {
            "planner": "astar",
            "map_json": self.map_json,
            "inflate_radius": self.inflate_radius,
            "allow_diagonal": self.astar.allow_diagonal,
            "heuristic_weight": self.astar.heuristic_weight,
            "cost": float(result.cost),
            "expanded_nodes": int(result.expanded_nodes),
            "raw_waypoint_count": len(result.raw_path_world),
            "pruned_waypoint_count": len(result.path_world),
            "start_grid": tuple(int(value) for value in result.start_grid),
            "goal_grid": tuple(int(value) for value in result.goal_grid),
        }
        return NavPlan(
            goal=goal,
            waypoints=tuple(waypoints),
            metadata=metadata,
        )

    @staticmethod
    def _with_exact_endpoints(
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        path_world: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """保留 A* 栅格路径，同时把首尾替换为真实任务端点。"""

        waypoints: list[tuple[float, float]] = [start_xy]
        for point in path_world:
            candidate = (float(point[0]), float(point[1]))
            if math.dist(waypoints[-1], candidate) > 1.0e-9:
                waypoints.append(candidate)
        if math.dist(waypoints[-1], goal_xy) > 1.0e-9:
            waypoints.append(goal_xy)
        else:
            waypoints[-1] = goal_xy
        if len(waypoints) == 1:
            waypoints.append(goal_xy)
        return waypoints


AStarPlannerAdapter = AStarNavPlanner
NavPlannerAdapter = AStarNavPlanner


__all__ = [
    "AStarNavPlanner",
    "AStarPlannerAdapter",
    "NavPlannerAdapter",
]
