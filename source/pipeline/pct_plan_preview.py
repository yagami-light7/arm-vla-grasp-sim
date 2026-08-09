"""PCT 多楼层路线的 Isaac GUI 预览模式。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from source.interfaces import EpisodeSpec
from source.interfaces.simulation import SimulationState
from source.navigation import PCTNavPlanner, PCTPlannerConfig
from source.navigation.executor import (
    optimize_pct_post_stair_plan_path,
    optimize_pct_pre_stair_plan_path,
)

from .config import FullPhysicsConfig
from .navigation_smoke import (
    DEFAULT_PCT_SERVER_SCRIPT,
    _open_pct_local_grid_map,
    _open_pct_post_stair_grid_map,
    _required_pct_asset_path,
)


class PCTPlanPreviewPipeline:
    """只规划并绘制 PCT 路线，不执行 DWA、RL policy 或机械臂。"""

    simulation = None

    def __init__(
        self,
        *,
        config: FullPhysicsConfig,
        episode_spec: EpisodeSpec,
        episode_seed: int,
        episode_dir: str | Path,
        simulation_app: Any,
        project_root: str | Path,
    ) -> None:
        self.config = config
        self.episode_spec = episode_spec
        self.episode_seed = int(episode_seed)
        self.episode_dir = Path(episode_dir)
        self.simulation_app = simulation_app
        self.project_root = Path(project_root).expanduser().resolve()

    def run_episode(self) -> dict[str, Any]:
        started_at = time.time()
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        stage = self._open_stage()
        planner = self._create_planner()
        try:
            segments = self._plan_segments(planner)
        finally:
            planner.close()
        draw_report = draw_pct_plan_preview(stage, segments)
        report = {
            "schema_version": 1,
            "task_id": self.episode_spec.task_id,
            "episode_id": self.episode_spec.episode_id,
            "seed": self.episode_seed,
            "scene_usd": str(self._project_path(self.episode_spec.scene_usd)),
            "segments": segments,
            "draw_report": draw_report,
        }
        report_path = self.episode_dir / "pct_plan_preview.json"
        report_path.write_text(
            json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "episode_id": self.episode_spec.episode_id,
            "task_id": self.episode_spec.task_id,
            "seed": self.episode_seed,
            "success": True,
            "pure_physics_success": False,
            "stable_physics_success": False,
            "physical_navigation_success": False,
            "physical_manipulation_success": False,
            "execution_mode": "pct_plan_preview",
            "failure_reason": "",
            "duration_steps": 0,
            "duration_seconds": time.time() - started_at,
            "state_trace": ["build_stage", "plan_pct_preview", "done"],
            "final_state": "done",
            "pct_plan_preview": str(report_path),
            "latest_planner_result": segments[-1]["metadata"] if segments else None,
        }
        (self.episode_dir / "summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for _ in range(3):
            self.simulation_app.update()
        print(
            "[pct-plan-preview] "
            f"segments={len(segments)} root={draw_report['root_prim_path']} "
            f"report={report_path}",
            flush=True,
        )
        return summary

    def _open_stage(self) -> Any:
        import omni.usd

        scene_usd = self._project_path(self.episode_spec.scene_usd)
        if not scene_usd.is_file():
            raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")
        context = omni.usd.get_context()
        if context.open_stage(str(scene_usd)) is False:
            raise RuntimeError(f"failed to open stage: {scene_usd}")
        for _ in range(30):
            self.simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"stage did not load: {scene_usd}")
        return stage

    def _create_planner(self) -> PCTNavPlanner:
        nav = self.config.navigation
        return PCTNavPlanner(
            PCTPlannerConfig(
                enabled=True,
                planner_root=self._optional_project_path(nav.pct_planner_root),
                server_script=(
                    self._optional_project_path(nav.pct_server_script)
                    or DEFAULT_PCT_SERVER_SCRIPT
                ),
                server_python=self._optional_project_path(nav.pct_server_python),
                tomogram_name=nav.pct_tomogram_name,
                tomogram_path=_required_pct_asset_path(
                    nav.pct_tomogram_path,
                    field_name="pct_tomogram_path",
                ),
                walkable_path=_required_pct_asset_path(
                    nav.pct_walkable_path,
                    field_name="pct_walkable_path",
                ),
                collision_ply_path=_required_pct_asset_path(
                    nav.pct_collision_ply_path,
                    field_name="pct_collision_ply_path",
                ),
                global_vertical_obstacle_min_slices=(
                    nav.pct_global_vertical_obstacle_min_slices
                ),
                cross_floor_vertical_obstacle_min_slices=(
                    nav.pct_cross_floor_vertical_obstacle_min_slices
                ),
                cross_floor_gateway_points=nav.pct_cross_floor_gateway_points,
                cross_floor_stair_exit_points=(
                    nav.pct_cross_floor_stair_exit_points
                ),
                cross_floor_stair_midpoint_points=(
                    nav.pct_cross_floor_stair_midpoint_points
                ),
                cross_floor_gateway_radius_m=nav.pct_cross_floor_gateway_radius_m,
                robot_root_to_floor_m=nav.pct_robot_root_to_floor_m,
                body_obstacle_min_height_m=(
                    nav.pct_body_obstacle_min_height_m
                ),
                body_obstacle_max_height_m=(
                    nav.pct_body_obstacle_max_height_m
                ),
                stair_min_horizontal_per_slice_m=(
                    nav.pct_stair_min_horizontal_per_slice_m
                ),
                stair_max_horizontal_per_slice_m=(
                    nav.pct_stair_max_horizontal_per_slice_m
                ),
                stair_vertical_radius_m=nav.pct_stair_vertical_radius_m,
                stair_progress_tolerance=nav.pct_stair_progress_tolerance,
                stair_progress_cost_weight=nav.pct_stair_progress_cost_weight,
                obstacle_clearance_radius_m=(
                    nav.pct_obstacle_clearance_radius_m
                ),
                obstacle_clearance_cost_weight=(
                    nav.pct_obstacle_clearance_cost_weight
                ),
                coord_mode=nav.pct_coord_mode,
                pct_offset_x=nav.pct_offset_x,
                pct_offset_y=nav.pct_offset_y,
                pct_offset_z=nav.pct_offset_z,
                pct_scale_x=nav.pct_scale_x,
                pct_scale_y=nav.pct_scale_y,
                pct_scale_z=nav.pct_scale_z,
                pct_rotation_x_rad=nav.pct_rotation_x_rad,
                pct_rotation_y_rad=nav.pct_rotation_y_rad,
                pct_rotation_z_rad=nav.pct_rotation_z_rad,
                fallback_to_astar=False,
            )
        )

    def _plan_segments(self, planner: PCTNavPlanner) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        start_state = _state_from_goal(self.episode_spec.start)
        pick_plan = planner.plan(start_state, self.episode_spec.pick_goal)
        segments.append(_segment_report("start_to_pick", pick_plan))
        if self.episode_spec.place_goal is not None:
            place_start = _state_from_goal(self.episode_spec.pick_goal)
            place_plan = planner.plan(place_start, self.episode_spec.place_goal)
            pre_stair_grid_map = _open_pct_local_grid_map(
                self.episode_spec,
                self.config.navigation,
            )
            _optimized_pre_stair_path, pre_stair_optimization_report = (
                optimize_pct_pre_stair_plan_path(
                    place_plan,
                    pre_stair_grid_map=pre_stair_grid_map,
                    clearance_radius_m=(
                        self.config.navigation.local_clearance_radius
                    ),
                    preserve_start_distance_m=1.0,
                )
            )
            place_plan.metadata["pre_stair_path_optimization"] = (
                pre_stair_optimization_report
            )
            post_stair_grid_map = _open_pct_post_stair_grid_map(
                self.episode_spec,
                self.config.navigation,
            )
            if post_stair_grid_map is not None:
                _optimized_path, optimization_report = (
                    optimize_pct_post_stair_plan_path(
                        place_plan,
                        post_stair_grid_map=post_stair_grid_map,
                        min_z_delta_m=(
                            self.config.navigation.pct_stair_float_min_z_delta_m
                        ),
                        approach_distance_m=(
                            self.config.navigation.pct_stair_float_approach_distance_m
                        ),
                        exit_distance_m=(
                            self.config.navigation.pct_stair_float_exit_distance_m
                        ),
                        clearance_radius_m=(
                            self.config.navigation.local_clearance_radius
                        ),
                    )
                )
                place_plan.metadata["post_stair_path_optimization"] = (
                    optimization_report
                )
            segments.append(_segment_report("pick_to_place", place_plan))
        return segments

    def _project_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _optional_project_path(self, raw_path: str | Path | None) -> Path | None:
        if raw_path is None:
            return None
        return self._project_path(raw_path)


def create_pct_plan_preview_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation_app: Any,
    project_root: str | Path,
) -> PCTPlanPreviewPipeline:
    """创建只负责 PCT 路线预览的轻量 pipeline。"""

    return PCTPlanPreviewPipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation_app=simulation_app,
        project_root=project_root,
    )


def draw_pct_plan_preview(
    stage: Any,
    segments: list[dict[str, Any]],
    *,
    root_prim_path: str = "/World/PCTPlanPreview",
) -> dict[str, Any]:
    """在当前 USD stage 中绘制 PCT 路线曲线和 waypoint 球。"""

    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath(root_prim_path).IsValid():
        stage.RemovePrim(root_prim_path)
    UsdGeom.Xform.Define(stage, root_prim_path)
    colors = {
        "start_to_pick": (0.1, 0.8, 1.0),
        "pick_to_place": (1.0, 0.75, 0.1),
    }
    drawn_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        name = str(segment["name"])
        points = [
            tuple(float(value) for value in point[:3])
            for point in segment.get("path_3d", ())
            if isinstance(point, (list, tuple)) and len(point) >= 3
        ]
        if len(points) < 2:
            continue
        color = colors.get(name, (0.8, 0.8, 0.8))
        segment_root = f"{root_prim_path}/{name}"
        UsdGeom.Xform.Define(stage, segment_root)
        lifted = [(x, y, z + 0.12) for x, y, z in points]
        curve_path = f"{segment_root}/path"
        curve = UsdGeom.BasisCurves.Define(stage, curve_path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([len(lifted)])
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in lifted])
        curve.CreateWidthsAttr([0.035] * len(lifted))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        marker_paths: list[str] = []
        for waypoint_index, point in enumerate(lifted):
            radius = 0.09 if waypoint_index in {0, len(lifted) - 1} else 0.055
            marker_path = f"{segment_root}/wp_{waypoint_index:03d}"
            marker = UsdGeom.Sphere.Define(stage, marker_path)
            marker.CreateRadiusAttr(radius)
            marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*point))
            marker_paths.append(marker_path)
        drawn_segments.append(
            {
                "name": name,
                "index": index,
                "curve_prim_path": curve_path,
                "marker_count": len(marker_paths),
                "marker_prim_paths": marker_paths,
                "color_rgb": color,
                "waypoint_count": len(points),
                "slice_start": segment.get("metadata", {}).get("slice_start"),
                "slice_end": segment.get("metadata", {}).get("slice_end"),
                "hard_obstacle_min_slices": segment.get("metadata", {}).get(
                    "hard_obstacle_min_slices"
                ),
            }
        )
    return {
        "root_prim_path": root_prim_path,
        "segments": drawn_segments,
        "legend": "cyan=start_to_pick, yellow=pick_to_place",
    }


def _state_from_goal(goal: Any) -> SimulationState:
    z = 0.0 if goal.z is None else float(goal.z)
    return SimulationState(
        step_index=0,
        timestamp=0.0,
        robot_root_pose=(float(goal.x), float(goal.y), z, 1.0, 0.0, 0.0, 0.0),
        robot_root_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        metadata={
            "floor_id": goal.floor_id,
            "slice_id": goal.slice_id,
            "pct_plan_preview": True,
        },
    )


def _segment_report(name: str, plan: Any) -> dict[str, Any]:
    path_3d = tuple(plan.metadata.get("path_3d") or ())
    return {
        "name": name,
        "goal": {
            "x": float(plan.goal.x),
            "y": float(plan.goal.y),
            "z": plan.goal.z,
            "yaw": float(plan.goal.yaw),
            "floor_id": plan.goal.floor_id,
            "slice_id": plan.goal.slice_id,
        },
        "waypoints_xy": tuple(plan.waypoints),
        "path_3d": path_3d,
        "waypoint_count": len(plan.waypoints),
        "path_3d_count": len(path_3d),
        "metadata": dict(plan.metadata),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
