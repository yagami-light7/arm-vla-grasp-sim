"""Map-independent refinement from sparse global paths to DWA controller paths.

The global planner and the local occupancy grid do not necessarily share the
same raster origin or endpoint snapping policy.  Controller paths therefore
must be anchored at the exact live start/goal in world coordinates instead of
blindly following grid-cell centres returned by either planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .astar import AStarPlanner
from .grid_map import OccupancyGridMap


Point2D = tuple[float, float]


@dataclass(frozen=True)
class LocalPathRefinementResult:
    """A controller-ready path and the effective map used to validate it."""

    path_world: tuple[Point2D, ...]
    grid_map: OccupancyGridMap
    report: dict[str, Any]


class LocalPathRefinementError(RuntimeError):
    """Raised when no locally collision-free controller path can be built."""

    def __init__(self, message: str, *, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def refine_same_floor_path(
    *,
    grid_map: OccupancyGridMap,
    global_path_world: Iterable[Point2D],
    live_start_xy: Point2D,
    exact_goal_xy: Point2D,
) -> LocalPathRefinementResult:
    """Build a same-floor DWA path using exact endpoints and local occupancy.

    The occupancy grid is treated as configuration space: obstacle inflation
    and task keepouts belong to map construction, not to a hidden clearance
    threshold in this function.  A collision-free direct segment is therefore
    preferred.  If it is blocked, A* is run from the *live* start cell and the
    resulting cell-centre path is re-anchored to exact world endpoints before
    line-of-sight string pulling.

    When the robot's current cell is marked occupied, only that single cell is
    cleared in the effective local map.  This is the conventional costmap
    recovery for a robot that physically already occupies the cell; no nearby
    obstacle or goal cell is silently cleared.
    """

    live_start = _finite_point(live_start_xy, field_name="live_start_xy")
    exact_goal = _finite_point(exact_goal_xy, field_name="exact_goal_xy")
    global_path = _deduplicate_points(global_path_world)
    report: dict[str, Any] = {
        "enabled": True,
        "success": False,
        "strategy": "exact_endpoint_los_then_local_astar_v1",
        "input_waypoints": len(global_path),
        "live_start_xy": list(live_start),
        "exact_goal_xy": list(exact_goal),
        "grid_resolution_m": float(grid_map.resolution),
        "grid_origin_xyyaw": [float(value) for value in grid_map.origin],
        "clearance_policy": "configuration_space_occupancy",
        "hidden_clearance_threshold_m": None,
    }
    if global_path:
        report["global_path_start_xy"] = list(global_path[0])
        report["global_start_offset_m"] = float(
            math.dist(live_start, global_path[0])
        )

    start_rc = grid_map.world_to_grid(*live_start)
    goal_rc = grid_map.world_to_grid(*exact_goal)
    report["start_grid"] = [int(value) for value in start_rc]
    report["goal_grid"] = [int(value) for value in goal_rc]
    report["start_in_bounds"] = bool(grid_map.in_bounds(*start_rc))
    report["goal_in_bounds"] = bool(grid_map.in_bounds(*goal_rc))
    if not grid_map.in_bounds(*start_rc):
        report["reason"] = "live_start_outside_local_map"
        raise LocalPathRefinementError(
            "live start is outside the local occupancy grid",
            report=report,
        )
    if not grid_map.in_bounds(*goal_rc):
        report["reason"] = "exact_goal_outside_local_map"
        raise LocalPathRefinementError(
            "exact goal is outside the local occupancy grid",
            report=report,
        )

    start_was_occupied = bool(grid_map.is_occupied(*start_rc))
    goal_is_occupied = bool(grid_map.is_occupied(*goal_rc))
    report["start_cell_occupied_before_recovery"] = start_was_occupied
    report["goal_cell_occupied"] = goal_is_occupied
    effective_map = (
        _clear_single_cell(grid_map, start_rc)
        if start_was_occupied
        else grid_map
    )
    report["start_cell_recovered"] = start_was_occupied
    report["start_cell_recovery_policy"] = (
        "clear_live_robot_cell_only" if start_was_occupied else "not_needed"
    )
    if goal_is_occupied:
        report["reason"] = "exact_goal_occupied"
        raise LocalPathRefinementError(
            "exact goal is occupied in the local occupancy grid",
            report=report,
        )

    direct_free, direct_clearance = world_segment_clearance(
        live_start,
        exact_goal,
        effective_map,
    )
    report["direct_path_free"] = bool(direct_free)
    report["direct_clearance_m"] = direct_clearance
    if direct_free:
        path = (live_start, exact_goal)
        report.update(
            {
                "success": True,
                "mode": "pct_same_floor_direct",
                "output_waypoints": 2,
                "raw_grid_waypoints": 0,
                "collinear_waypoints": 0,
                "line_of_sight_waypoints": 2,
            }
        )
        report.update(_path_geometry_report(path, exact_goal=exact_goal))
        return LocalPathRefinementResult(
            path_world=path,
            grid_map=effective_map,
            report=report,
        )

    try:
        astar_result = AStarPlanner().plan(
            effective_map,
            live_start,
            exact_goal,
            snap_to_free=False,
        )
    except Exception as exc:
        report.update(
            {
                "reason": "local_astar_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise LocalPathRefinementError(
            f"local A* failed: {exc}",
            report=report,
        ) from exc

    # Drop the raster cell centres at both ends.  Both exact points are inside
    # those same free cells, and using cell centres as locomotion targets is the
    # source of map-origin-dependent initial turns.
    raw_grid_world = [
        (float(point[0]), float(point[1]))
        for point in astar_result.raw_path_world
    ]
    anchored: list[Point2D] = [live_start]
    for point in raw_grid_world[1:-1]:
        if math.dist(anchored[-1], point) > 1.0e-9:
            anchored.append(point)
    if math.dist(anchored[-1], exact_goal) > 1.0e-9:
        anchored.append(exact_goal)
    elif anchored:
        anchored[-1] = exact_goal
    if len(anchored) < 2:
        anchored.append(exact_goal)

    simplified = simplify_path_line_of_sight(anchored, effective_map)
    invalid_segment = first_blocked_path_segment(simplified, effective_map)
    if invalid_segment is not None:
        report.update(
            {
                "reason": "refined_path_contains_blocked_segment",
                "blocked_segment_index": int(invalid_segment),
            }
        )
        raise LocalPathRefinementError(
            "refined local path contains a blocked segment",
            report=report,
        )

    path = tuple(simplified)
    report.update(
        {
            "success": True,
            "mode": "pct_same_floor_local_astar",
            "output_waypoints": len(path),
            "raw_grid_waypoints": len(astar_result.raw_path_grid),
            "collinear_waypoints": len(astar_result.path_world),
            "anchored_waypoints": len(anchored),
            "line_of_sight_waypoints": len(path),
            "expanded_nodes": int(astar_result.expanded_nodes),
            "cost": float(astar_result.cost),
        }
    )
    report.update(_path_geometry_report(path, exact_goal=exact_goal))
    return LocalPathRefinementResult(
        path_world=path,
        grid_map=effective_map,
        report=report,
    )


def simplify_path_line_of_sight(
    path_world: Iterable[Point2D],
    grid_map: OccupancyGridMap,
) -> list[Point2D]:
    """Greedily string-pull a path while preserving collision-free segments."""

    points = _deduplicate_points(path_world)
    if len(points) <= 2:
        return points
    output = [points[0]]
    anchor_index = 0
    while anchor_index < len(points) - 1:
        chosen_index = anchor_index + 1
        for candidate_index in range(len(points) - 1, anchor_index, -1):
            free, _clearance = world_segment_clearance(
                points[anchor_index],
                points[candidate_index],
                grid_map,
            )
            if free:
                chosen_index = candidate_index
                break
        output.append(points[chosen_index])
        anchor_index = chosen_index
    return _deduplicate_points(output)


def first_blocked_path_segment(
    path_world: Iterable[Point2D],
    grid_map: OccupancyGridMap,
) -> int | None:
    """Return the first blocked segment index, or ``None`` when all are free."""

    points = list(path_world)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        free, _clearance = world_segment_clearance(start, end, grid_map)
        if not free:
            return index
    return None


def world_segment_clearance(
    start: Point2D,
    end: Point2D,
    grid_map: OccupancyGridMap,
) -> tuple[bool, float | None]:
    """Check every grid cell touched by a world-frame line segment.

    A supercover grid traversal is used instead of fixed-distance samples, so
    changing map resolution/origin cannot make a thin occupied corner disappear
    between samples.  Clearance is diagnostic; occupancy alone decides whether
    the segment is feasible because the input map already represents configured
    robot inflation and task keepouts.
    """

    minimum_clearance = float("inf")
    cells = _supercover_segment_cells(start, end, grid_map)
    if not cells:
        return False, 0.0
    for row, col in cells:
        if grid_map.is_occupied(row, col):
            return False, 0.0
        clearance = grid_map.distance_to_obstacle(row, col)
        if clearance is None:
            minimum_clearance = float("nan")
        elif not math.isnan(minimum_clearance):
            minimum_clearance = min(minimum_clearance, float(clearance))
    if math.isnan(minimum_clearance):
        return True, None
    if math.isinf(minimum_clearance):
        return True, None
    return True, minimum_clearance


def _supercover_segment_cells(
    start: Point2D,
    end: Point2D,
    grid_map: OccupancyGridMap,
) -> list[tuple[int, int]]:
    """Enumerate all occupancy cells touched by a segment using 2-D DDA."""

    u0, v0 = _world_to_bottom_grid(start, grid_map)
    u1, v1 = _world_to_bottom_grid(end, grid_map)
    col = int(math.floor(u0))
    row_from_bottom = int(math.floor(v0))
    end_col = int(math.floor(u1))
    end_row_from_bottom = int(math.floor(v1))
    du = u1 - u0
    dv = v1 - v0
    step_u = 0 if abs(du) <= 1.0e-15 else (1 if du > 0.0 else -1)
    step_v = 0 if abs(dv) <= 1.0e-15 else (1 if dv > 0.0 else -1)

    if step_u > 0:
        t_max_u = (math.floor(u0) + 1.0 - u0) / du
        t_delta_u = 1.0 / du
    elif step_u < 0:
        t_max_u = (u0 - math.floor(u0)) / -du
        t_delta_u = 1.0 / -du
    else:
        t_max_u = float("inf")
        t_delta_u = float("inf")
    if step_v > 0:
        t_max_v = (math.floor(v0) + 1.0 - v0) / dv
        t_delta_v = 1.0 / dv
    elif step_v < 0:
        t_max_v = (v0 - math.floor(v0)) / -dv
        t_delta_v = 1.0 / -dv
    else:
        t_max_v = float("inf")
        t_delta_v = float("inf")

    bottom_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(bottom_row: int, current_col: int) -> None:
        key = (bottom_row, current_col)
        if key not in seen:
            seen.add(key)
            bottom_cells.append(key)

    add(row_from_bottom, col)
    max_steps = 2 + abs(end_col - col) + abs(end_row_from_bottom - row_from_bottom)
    for _ in range(max_steps):
        if col == end_col and row_from_bottom == end_row_from_bottom:
            break
        if t_max_u + 1.0e-12 < t_max_v:
            col += step_u
            t_max_u += t_delta_u
            add(row_from_bottom, col)
        elif t_max_v + 1.0e-12 < t_max_u:
            row_from_bottom += step_v
            t_max_v += t_delta_v
            add(row_from_bottom, col)
        else:
            # Crossing a grid corner touches both side-adjacent cells as well
            # as the diagonal destination; include all three conservatively.
            if step_u:
                add(row_from_bottom, col + step_u)
            if step_v:
                add(row_from_bottom + step_v, col)
            col += step_u
            row_from_bottom += step_v
            t_max_u += t_delta_u
            t_max_v += t_delta_v
            add(row_from_bottom, col)

    # A segment exactly on a cell boundary touches both adjacent rows/columns.
    if abs(dv) <= 1.0e-15 and _near_integer(v0):
        for bottom_row, current_col in tuple(bottom_cells):
            add(bottom_row - 1, current_col)
    if abs(du) <= 1.0e-15 and _near_integer(u0):
        for bottom_row, current_col in tuple(bottom_cells):
            add(bottom_row, current_col - 1)

    return [
        (grid_map.height - 1 - bottom_row, current_col)
        for bottom_row, current_col in bottom_cells
    ]


def _world_to_bottom_grid(
    point: Point2D,
    grid_map: OccupancyGridMap,
) -> tuple[float, float]:
    delta_x = float(point[0]) - float(grid_map.origin[0])
    delta_y = float(point[1]) - float(grid_map.origin[1])
    cos_yaw = math.cos(float(grid_map.origin[2]))
    sin_yaw = math.sin(float(grid_map.origin[2]))
    local_x = cos_yaw * delta_x + sin_yaw * delta_y
    local_y = -sin_yaw * delta_x + cos_yaw * delta_y
    return (
        local_x / float(grid_map.resolution),
        local_y / float(grid_map.resolution),
    )


def _clear_single_cell(
    grid_map: OccupancyGridMap,
    cell: tuple[int, int],
) -> OccupancyGridMap:
    occupancy = grid_map.occupancy.copy()
    occupancy[cell] = False
    return OccupancyGridMap(
        occupancy=occupancy,
        resolution=grid_map.resolution,
        origin=grid_map.origin,
        image_path=grid_map.image_path,
        meta_path=grid_map.meta_path,
    )


def _finite_point(point: Point2D, *, field_name: str) -> Point2D:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise ValueError(f"{field_name} must contain x/y")
    value = (float(point[0]), float(point[1]))
    if not all(math.isfinite(component) for component in value):
        raise ValueError(f"{field_name} contains non-finite values")
    return value


def _deduplicate_points(points: Iterable[Point2D]) -> list[Point2D]:
    output: list[Point2D] = []
    for raw_point in points:
        point = _finite_point(raw_point, field_name="path point")
        if output and math.dist(output[-1], point) <= 1.0e-9:
            continue
        output.append(point)
    return output


def _path_geometry_report(
    path: tuple[Point2D, ...],
    *,
    exact_goal: Point2D,
) -> dict[str, Any]:
    lengths = [math.dist(start, end) for start, end in zip(path, path[1:])]
    headings = [
        math.atan2(end[1] - start[1], end[0] - start[0])
        for start, end in zip(path, path[1:])
        if math.dist(start, end) > 1.0e-9
    ]
    turns = [
        abs(_wrap_angle(current - previous))
        for previous, current in zip(headings, headings[1:])
    ]
    goal_vector = (
        exact_goal[0] - path[0][0],
        exact_goal[1] - path[0][1],
    )
    goal_distance = math.hypot(*goal_vector)
    first_progress = 0.0
    first_heading_error = 0.0
    if lengths and goal_distance > 1.0e-9:
        first_vector = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        first_progress = (
            first_vector[0] * goal_vector[0]
            + first_vector[1] * goal_vector[1]
        ) / goal_distance
        first_heading_error = _wrap_angle(
            math.atan2(first_vector[1], first_vector[0])
            - math.atan2(goal_vector[1], goal_vector[0])
        )
    return {
        "path_length_m": float(sum(lengths)),
        "turn_count": len(turns),
        "max_turn_angle_rad": float(max(turns, default=0.0)),
        "total_abs_turn_angle_rad": float(sum(turns)),
        "first_segment_goal_progress_m": float(first_progress),
        "first_segment_heading_error_to_goal_rad": float(first_heading_error),
    }


def _near_integer(value: float) -> bool:
    return abs(value - round(value)) <= 1.0e-12


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "LocalPathRefinementError",
    "LocalPathRefinementResult",
    "first_blocked_path_segment",
    "refine_same_floor_path",
    "simplify_path_line_of_sight",
    "world_segment_clearance",
]
