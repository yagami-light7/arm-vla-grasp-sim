"""Unified adapter around the pure A* and DWA navigation modules."""

from __future__ import annotations

from dataclasses import dataclass

from ..navlib import AStarPlanResult, AStarPlanner, DWAConfig, DWAController, DWADebug, OccupancyGridMap


@dataclass(frozen=True)
class NavCommand:
    """One body-frame locomotion command and its DWA diagnostics."""

    vx: float
    vy: float
    wz: float
    debug: DWADebug


class NavPlanner:
    """Load a map, plan a global route, and compute body-frame DWA commands.

    The migrated reference controller is intentionally non-holonomic in v1:
    commands contain ``vy == 0.0``. The three-element API is retained so a
    holonomic local planner can be introduced later without changing callers.
    """

    def __init__(
        self,
        map_json: str,
        inflate_radius: float,
        dwa_config: DWAConfig,
        *,
        local_clearance_radius: float | None = None,
    ):
        self.raw_map = OccupancyGridMap.from_meta_file(map_json)
        self.global_map = self.raw_map.inflate(inflate_radius)
        self.local_map = self.raw_map.inflate(
            inflate_radius if local_clearance_radius is None else local_clearance_radius
        )
        self.dwa_config = dwa_config
        self.astar = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0)
        self.last_global_plan: AStarPlanResult | None = None
        self._controller: DWAController | None = None
        self._controller_path: list[tuple[float, float]] | None = None

    def plan_global_path(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Return a pruned world-frame A* path."""

        self.last_global_plan = self.astar.plan(
            self.global_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            snap_to_free=True,
            max_snap_distance_m=max(0.5, self.global_map.resolution),
        )
        self._controller = None
        self._controller_path = None
        return list(self.last_global_plan.path_world)

    def compute_command(
        self,
        robot_pose_xyyaw: tuple[float, float, float],
        robot_speed: tuple[float, float],
        path_world: list[tuple[float, float]],
    ) -> tuple[float, float, float]:
        """Return the next body-frame command as ``vx, vy, wz``."""

        return self.compute_command_with_debug(robot_pose_xyyaw, robot_speed, path_world)[:3]

    def compute_command_with_debug(
        self,
        robot_pose_xyyaw: tuple[float, float, float],
        robot_speed: tuple[float, float],
        path_world: list[tuple[float, float]],
    ) -> tuple[float, float, float, DWADebug]:
        """Return the next body-frame command and DWA diagnostics."""

        if self._controller is None or self._controller_path != path_world:
            self._controller = DWAController(path_world=path_world, grid_map=self.local_map, config=self.dwa_config)
            self._controller_path = list(path_world)
        command, debug = self._controller.compute_command(robot_pose_xyyaw, robot_speed)
        return float(command[0]), float(command[1]), float(command[2]), debug
