"""Randomized pick task generation with navigation-map validation."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from source.navigation.adapters.frame_utils import world_to_map_local_xy, wrap_yaw
from source.navigation.navlib import OccupancyGridMap


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RandomTaskGenerationError(RuntimeError):
    """Raised when no valid randomized pick task can be generated."""


@dataclass(frozen=True)
class SpawnRegion:
    """Axis-aligned table sampling region in world coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    table_z: float
    object_z_offset: float = 0.04

    @property
    def object_z(self) -> float:
        return self.table_z + self.object_z_offset

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z": self.table_z,
            "object_z_offset": self.object_z_offset,
        }


@dataclass(frozen=True)
class ObjectPose:
    """World-frame object pose used for task generation."""

    x: float
    y: float
    z: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0
    edge_side: str | None = None
    edge_distance_m: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "roll": self.roll,
            "pitch": self.pitch,
        }

    def edge_report_dict(self) -> dict[str, Any]:
        return {
            "selected_edge_side": self.edge_side,
            "selected_edge_distance_m": self.edge_distance_m,
        }


@dataclass(frozen=True)
class BaseGoalCandidate:
    """Candidate final base pose and validation diagnostics."""

    x: float
    y: float
    yaw: float
    standoff: float
    approach_angle_rad: float
    raw_free: bool
    clearance_free: bool
    boundary_clearance_m: float
    obstacle_clearance_m: float | None
    edge_alignment_score: float
    score: float
    rejection_reason: str = ""

    @property
    def valid(self) -> bool:
        return not self.rejection_reason

    def to_goal_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "standoff": self.standoff,
            "approach_angle_deg": math.degrees(self.approach_angle_rad),
            "raw_free": self.raw_free,
            "clearance_free": self.clearance_free,
            "boundary_clearance_m": self.boundary_clearance_m,
            "obstacle_clearance_m": self.obstacle_clearance_m,
            "edge_alignment_score": self.edge_alignment_score,
            "score": self.score,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


def resolve_project_path(raw_path: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve task-relative paths using the repository root."""

    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _edge_tokens(edge_side: str | None) -> tuple[str, ...]:
    if not edge_side:
        return ()
    tokens = tuple(token for token in edge_side.split("_") if token in {"x", "y", "min", "max"})
    if len(tokens) == 2 and tokens[0] in {"x", "y"} and tokens[1] in {"min", "max"}:
        return (f"{tokens[0]}_{tokens[1]}",)
    if len(tokens) == 4:
        return (f"{tokens[0]}_{tokens[1]}", f"{tokens[2]}_{tokens[3]}")
    raise ValueError(f"unsupported edge side {edge_side!r}; use x_min/x_max/y_min/y_max or corner pairs.")


def _edge_side_from_approach_angle(angle_deg: float) -> str:
    """Return the table edge that best matches the object-to-base direction."""

    angle_rad = math.radians(float(angle_deg))
    base_dx = -math.cos(angle_rad)
    base_dy = -math.sin(angle_rad)
    if abs(base_dx) >= abs(base_dy):
        return "x_max" if base_dx >= 0.0 else "x_min"
    return "y_max" if base_dy >= 0.0 else "y_min"


def derive_edge_sides_from_approach_angles(approach_angles_deg: Sequence[float]) -> list[str]:
    """Derive accessible table edges from base-goal approach directions."""

    sides: list[str] = []
    for angle_deg in approach_angles_deg:
        side = _edge_side_from_approach_angle(float(angle_deg))
        if side not in sides:
            sides.append(side)
    return sides or ["x_max", "y_max"]


def _sample_axis_near_edge(
    rng: random.Random,
    lower: float,
    upper: float,
    edge_name: str,
    *,
    edge_margin: float,
    edge_min_clearance: float,
) -> tuple[float, float]:
    """Sample an axis value near one boundary and return distance to that edge."""

    width = upper - lower
    if width <= 0.0:
        raise ValueError("spawn axis range must have positive width for edge-biased sampling.")
    margin = min(float(edge_margin), width)
    min_clearance = min(max(0.0, float(edge_min_clearance)), margin)
    if margin <= min_clearance:
        distance = min_clearance
    else:
        distance = rng.triangular(min_clearance, margin, min_clearance)
    if edge_name == "min":
        return lower + distance, distance
    if edge_name == "max":
        return upper - distance, distance
    raise ValueError(f"unsupported edge name {edge_name!r}")


def sample_object_pose(
    rng: random.Random,
    spawn_region: SpawnRegion,
    *,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    edge_sides: Sequence[str] | None = None,
    edge_margin: float | None = None,
    edge_min_clearance: float = 0.02,
) -> ObjectPose:
    """Sample one object pose inside the configured table region."""

    yaw_min, yaw_max = yaw_range_deg
    if spawn_region.x_min > spawn_region.x_max:
        raise ValueError("spawn_region x_min must be <= x_max.")
    if spawn_region.y_min > spawn_region.y_max:
        raise ValueError("spawn_region y_min must be <= y_max.")
    if yaw_min > yaw_max:
        raise ValueError("yaw_range_deg min must be <= max.")
    edge_side = None
    edge_distance = None
    if edge_margin is None:
        x = rng.uniform(spawn_region.x_min, spawn_region.x_max)
        y = rng.uniform(spawn_region.y_min, spawn_region.y_max)
    else:
        if edge_margin <= 0.0:
            raise ValueError("edge_margin must be positive when edge-biased sampling is enabled.")
        candidate_sides = list(edge_sides or ("x_max", "y_max"))
        if not candidate_sides:
            raise ValueError("edge_sides must not be empty when edge-biased sampling is enabled.")
        edge_side = rng.choice(candidate_sides)
        tokens = _edge_tokens(edge_side)
        x = rng.uniform(spawn_region.x_min, spawn_region.x_max)
        y = rng.uniform(spawn_region.y_min, spawn_region.y_max)
        distances: list[float] = []
        for token in tokens:
            axis, edge_name = token.split("_", 1)
            if axis == "x":
                x, distance = _sample_axis_near_edge(
                    rng,
                    spawn_region.x_min,
                    spawn_region.x_max,
                    edge_name,
                    edge_margin=edge_margin,
                    edge_min_clearance=edge_min_clearance,
                )
            else:
                y, distance = _sample_axis_near_edge(
                    rng,
                    spawn_region.y_min,
                    spawn_region.y_max,
                    edge_name,
                    edge_margin=edge_margin,
                    edge_min_clearance=edge_min_clearance,
                )
            distances.append(distance)
        edge_distance = min(distances) if distances else None
    return ObjectPose(
        x=x,
        y=y,
        z=spawn_region.object_z,
        yaw=wrap_yaw(math.radians(rng.uniform(yaw_min, yaw_max))),
        roll=0.0,
        pitch=0.0,
        edge_side=edge_side,
        edge_distance_m=edge_distance,
    )


def generate_base_goal_candidates(
    object_pose: ObjectPose,
    *,
    standoff_candidates: Sequence[float],
    approach_angles_deg: Sequence[float],
    grid_map: OccupancyGridMap,
    clearance_map: OccupancyGridMap,
    clearance_radius: float,
    min_boundary_clearance: float,
    start_xy: tuple[float, float] | None = None,
    preferred_edge_side: str | None = None,
) -> list[BaseGoalCandidate]:
    """Create and score base-goal candidates around an object pose."""

    if not standoff_candidates:
        raise ValueError("standoff_candidates must not be empty.")
    if not approach_angles_deg:
        raise ValueError("approach_angles_deg must not be empty.")

    desired_standoff = sorted(float(value) for value in standoff_candidates)[len(standoff_candidates) // 2]
    preferred_edge_tokens = set(_edge_tokens(preferred_edge_side)) if preferred_edge_side else set()
    candidates: list[BaseGoalCandidate] = []
    for angle_deg in approach_angles_deg:
        angle_rad = math.radians(float(angle_deg))
        candidate_edge_tokens = set(_edge_tokens(_edge_side_from_approach_angle(float(angle_deg))))
        edge_alignment_score = (
            len(preferred_edge_tokens & candidate_edge_tokens) / max(1, len(preferred_edge_tokens))
            if preferred_edge_tokens
            else 0.0
        )
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        for standoff in standoff_candidates:
            standoff = float(standoff)
            base_x = object_pose.x - standoff * cos_angle
            base_y = object_pose.y - standoff * sin_angle
            base_yaw = wrap_yaw(math.atan2(object_pose.y - base_y, object_pose.x - base_x))
            row, col = grid_map.world_to_grid(base_x, base_y)
            raw_free = grid_map.in_bounds(row, col) and not grid_map.is_occupied(row, col)
            clearance_free = clearance_map.in_bounds(row, col) and not clearance_map.is_occupied(row, col)
            local_x, local_y = world_to_map_local_xy((base_x, base_y), grid_map.origin)
            boundary_clearance = min(
                local_x,
                local_y,
                grid_map.width * grid_map.resolution - local_x,
                grid_map.height * grid_map.resolution - local_y,
            )
            obstacle_clearance = grid_map.distance_to_obstacle(row, col)

            rejection_reason = ""
            if not math.isfinite(base_yaw):
                rejection_reason = "invalid_yaw"
            elif not raw_free:
                rejection_reason = "raw_map_occupied"
            elif not clearance_free:
                rejection_reason = f"clearance_below_{clearance_radius:.3f}m"
            elif boundary_clearance < min_boundary_clearance:
                rejection_reason = f"boundary_clearance_below_{min_boundary_clearance:.3f}m"

            clearance_score = obstacle_clearance if obstacle_clearance is not None else (clearance_radius if clearance_free else 0.0)
            boundary_score = max(0.0, boundary_clearance)
            standoff_score = -abs(standoff - desired_standoff)
            start_score = 0.0
            if start_xy is not None:
                start_score = -0.05 * math.hypot(base_x - start_xy[0], base_y - start_xy[1])
            score = 2.0 * clearance_score + 0.25 * boundary_score + standoff_score + start_score + 0.50 * edge_alignment_score
            if rejection_reason:
                score -= 1000.0

            candidates.append(
                BaseGoalCandidate(
                    x=base_x,
                    y=base_y,
                    yaw=base_yaw,
                    standoff=standoff,
                    approach_angle_rad=angle_rad,
                    raw_free=raw_free,
                    clearance_free=clearance_free,
                    boundary_clearance_m=boundary_clearance,
                    obstacle_clearance_m=obstacle_clearance,
                    edge_alignment_score=edge_alignment_score,
                    score=score,
                    rejection_reason=rejection_reason,
                )
            )
    return candidates


def select_valid_base_goal(candidates: Sequence[BaseGoalCandidate]) -> BaseGoalCandidate | None:
    """Return the highest-scoring valid base-goal candidate."""

    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate.score)


def generate_random_pick_task(
    base_task: dict[str, Any],
    *,
    seed: int,
    nav_map_path: str | Path | None = None,
    object_prim_path: str | None = None,
    table_prim_path: str = "/World/table",
    spawn_region: SpawnRegion,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    standoff_candidates: Sequence[float] = (0.75, 0.90, 1.05),
    approach_angles_deg: Sequence[float] = (180.0, 210.0, 240.0),
    clearance_radius: float = 0.25,
    min_boundary_clearance: float = 0.25,
    edge_sides: Sequence[str] | None = None,
    edge_margin: float | None = 0.12,
    edge_min_clearance: float = 0.02,
    max_sample_attempts: int = 200,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate a complete randomized task JSON payload."""

    if max_sample_attempts <= 0:
        raise ValueError("max_sample_attempts must be positive.")
    if clearance_radius < 0.0:
        raise ValueError("clearance_radius must be non-negative.")
    if min_boundary_clearance < 0.0:
        raise ValueError("min_boundary_clearance must be non-negative.")
    if edge_margin is not None and edge_margin <= 0.0:
        raise ValueError("edge_margin must be positive when provided.")
    if edge_min_clearance < 0.0:
        raise ValueError("edge_min_clearance must be non-negative.")

    task = copy.deepcopy(base_task)
    pick = dict(task.get("pick") or {})
    task["pick"] = pick

    selected_object_path = object_prim_path or pick.get("object_prim_path")
    if not selected_object_path:
        raise RandomTaskGenerationError("pick.object_prim_path is required.")
    pick["object_prim_path"] = str(selected_object_path)

    selected_nav_map = str(nav_map_path or task.get("nav_map") or "")
    if not selected_nav_map:
        raise RandomTaskGenerationError("nav_map is required.")
    task["nav_map"] = selected_nav_map
    grid_map = OccupancyGridMap.from_meta_file(resolve_project_path(selected_nav_map, project_root=project_root))
    clearance_map = grid_map.inflate(clearance_radius)

    start = task.get("start") or {}
    start_xy = None
    if "x" in start and "y" in start:
        start_xy = (float(start["x"]), float(start["y"]))

    effective_edge_sides = list(edge_sides) if edge_sides is not None else derive_edge_sides_from_approach_angles(approach_angles_deg)
    rng = random.Random(int(seed))
    best_rejection_report: dict[str, Any] | None = None
    for attempt in range(1, max_sample_attempts + 1):
        object_pose = sample_object_pose(
            rng,
            spawn_region,
            yaw_range_deg=yaw_range_deg,
            edge_sides=effective_edge_sides,
            edge_margin=edge_margin,
            edge_min_clearance=edge_min_clearance,
        )
        candidates = generate_base_goal_candidates(
            object_pose,
            standoff_candidates=standoff_candidates,
            approach_angles_deg=approach_angles_deg,
            grid_map=grid_map,
            clearance_map=clearance_map,
            clearance_radius=clearance_radius,
            min_boundary_clearance=min_boundary_clearance,
            start_xy=start_xy,
            preferred_edge_side=object_pose.edge_side,
        )
        selected = select_valid_base_goal(candidates)
        if selected is None:
            best_rejection_report = {
                "attempt": attempt,
                "object_pose_world": object_pose.to_dict(),
                "candidate_rejections": [candidate.to_report_dict() for candidate in candidates],
            }
            continue

        task["task_id"] = int(task.get("task_id", base_task.get("task_id", 0)))
        task["episode_id"] = int(seed)
        pick["object_pose_world"] = object_pose.to_dict()
        pick["base_goal"] = selected.to_goal_dict()
        task["randomization"] = {
            "enabled": True,
            "seed": int(seed),
            "object": Path(str(selected_object_path)).name or str(selected_object_path).rsplit("/", 1)[-1],
            "object_prim_path": str(selected_object_path),
            "table_prim_path": table_prim_path,
            "spawn_region": spawn_region.to_dict(),
            "yaw_range_deg": [float(yaw_range_deg[0]), float(yaw_range_deg[1])],
            "object_edge_sampling": {
                "enabled": edge_margin is not None,
                "edge_sides": list(effective_edge_sides) if edge_margin is not None else [],
                "edge_margin": float(edge_margin) if edge_margin is not None else None,
                "edge_min_clearance": float(edge_min_clearance),
                **object_pose.edge_report_dict(),
            },
            "base_goal_generation": {
                "standoff_candidates": [float(value) for value in standoff_candidates],
                "approach_angle_candidates_deg": [float(value) for value in approach_angles_deg],
                "clearance_radius": float(clearance_radius),
                "min_boundary_clearance": float(min_boundary_clearance),
                "max_sample_attempts": int(max_sample_attempts),
            },
            "attempts_used": attempt,
            "selected_base_goal_candidate": selected.to_report_dict(),
            "candidate_count": len(candidates),
        }
        return task

    detail = (
        f"failed to generate valid object/base_goal after {max_sample_attempts} attempts; "
        f"last_rejections={json.dumps(best_rejection_report, ensure_ascii=False)[:4000]}"
    )
    raise RandomTaskGenerationError(detail)


def load_base_task(path: str | Path) -> dict[str, Any]:
    """Load a base task JSON as a mutable dictionary."""

    task_path = Path(path).expanduser().resolve()
    return json.loads(task_path.read_text(encoding="utf-8"))


def write_random_pick_task(
    *,
    base_task_path: str | Path,
    output_path: str | Path,
    seed: int,
    nav_map_path: str | Path | None = None,
    object_prim_path: str | None = None,
    table_prim_path: str = "/World/table",
    spawn_region: SpawnRegion,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    standoff_candidates: Sequence[float] = (0.75, 0.90, 1.05),
    approach_angles_deg: Sequence[float] = (180.0, 210.0, 240.0),
    clearance_radius: float = 0.25,
    min_boundary_clearance: float = 0.25,
    edge_sides: Sequence[str] | None = None,
    edge_margin: float | None = 0.12,
    edge_min_clearance: float = 0.02,
    max_sample_attempts: int = 200,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate and write a randomized pick task JSON file."""

    task = generate_random_pick_task(
        load_base_task(base_task_path),
        seed=seed,
        nav_map_path=nav_map_path,
        object_prim_path=object_prim_path,
        table_prim_path=table_prim_path,
        spawn_region=spawn_region,
        yaw_range_deg=yaw_range_deg,
        standoff_candidates=standoff_candidates,
        approach_angles_deg=approach_angles_deg,
        clearance_radius=clearance_radius,
        min_boundary_clearance=min_boundary_clearance,
        edge_sides=edge_sides,
        edge_margin=edge_margin,
        edge_min_clearance=edge_min_clearance,
        max_sample_attempts=max_sample_attempts,
        project_root=project_root,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    return task
