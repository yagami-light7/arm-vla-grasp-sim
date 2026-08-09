"""PCT grid core 的 ROS 2 进程内调用边界。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import importlib
import math
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Iterator, Protocol, Sequence

from .ground_surface import TriangleGroundProjector


class PCTBackendError(RuntimeError):
    """PCT backend 初始化、输入或执行异常。"""


class PCTNoPathError(PCTBackendError):
    """PCT 正常完成搜索但没有找到可行路径。"""


@dataclass(frozen=True)
class PCTBackendConfig:
    """直接 PCT backend 的所有可配置输入。"""

    project_root: Path
    tomogram_path: Path
    walkable_path: Path
    collision_ply_path: Path | None = None
    backend_kind: str = "compatible"
    upstream_source_root: Path | None = None
    upstream_use_quintic: bool = True
    upstream_max_heading_rate: float = 10.0
    upstream_astar_step_cost_weight: float = 0.20
    upstream_body_clearance_enabled: bool = False
    upstream_body_clearance_radius_m: float = 0.80
    upstream_body_clearance_maximum_cost: float = 20.0
    upstream_body_clearance_power: float = 2.0
    upstream_endpoint_layer_max_z_error_m: float = 0.25
    upstream_same_layer_shortcut_clearance_m: float = 0.27
    upstream_same_layer_shortcut_max_segment_m: float = 10.0
    upstream_stair_profile_path: Path | None = None
    upstream_stair_profile_match_tolerance_m: float = 0.60
    coord_mode: str = "sim_to_pct_180deg"
    pct_offset_x: float = 0.0
    pct_offset_y: float = 0.0
    pct_offset_z: float = 0.0
    pct_scale_x: float = 1.0
    pct_scale_y: float = 1.0
    pct_scale_z: float = 1.0
    pct_rotation_x_rad: float = 0.0
    pct_rotation_y_rad: float = 0.0
    pct_rotation_z_rad: float = 0.0
    slice_query_root_to_floor_m: float = 0.30
    goal_base_to_ground_m: float = 0.30
    global_vertical_obstacle_min_slices: int = 7
    cross_floor_vertical_obstacle_min_slices: int = 9
    cross_floor_gateway_points: tuple[tuple[float, float, float], ...] = (
        (1.5, 5.7, 0.6),
    )
    cross_floor_stair_exit_points: tuple[tuple[float, float, float], ...] = (
        (2.70, 7.05, 3.0),
    )
    cross_floor_stair_midpoint_points: tuple[
        tuple[float, float, float], ...
    ] = (
        (1.51822, 6.27683, 0.29486),
        (2.74512, 9.14634, 1.64666),
        (1.9202, 9.52807, 1.71919),
        (2.69841, 7.79872, 2.61031),
    )
    cross_floor_gateway_radius_m: float = 0.60
    body_obstacle_min_height_m: float = 0.30
    body_obstacle_max_height_m: float = 1.00
    stair_min_horizontal_per_slice_m: float = 0.40
    stair_max_horizontal_per_slice_m: float = 0.90
    stair_vertical_radius_m: float = 0.60
    stair_progress_tolerance: float = 0.35
    stair_progress_cost_weight: float = 20.0
    obstacle_clearance_radius_m: float = 0.60
    obstacle_clearance_cost_weight: float = 2.0
    ground_projection_max_z_error_m: float = 0.60
    start_ground_projection_max_z_error_m: float = 0.08
    terminal_ground_projection_max_z_error_m: float = 0.08
    path_sample_spacing_m: float = 0.20
    maximum_snap_distance_m: float = 0.25
    grid_max_expansions: int = 1_500_000
    grid_compress_max_segment_m: float = 0.80
    grid_timeout_sec: float = 10.0


@dataclass(frozen=True)
class PCTBackendPlan:
    """PCT 输出的仿真世界系地面路径。"""

    points_xyz: tuple[tuple[float, float, float], ...]
    metadata: dict[str, Any]


class GlobalPlannerBackend(Protocol):
    """节点所需的最小同步规划接口。"""

    def plan(
        self,
        *,
        start_base_xyz: Sequence[float],
        goal_base_xyz: Sequence[float],
        goal_yaw: float,
    ) -> PCTBackendPlan:
        """从当前 base 位姿规划到目标 base 位姿，返回地面高度路径。"""


def create_global_planner_backend(
    config: PCTBackendConfig,
) -> GlobalPlannerBackend:
    """按显式类型创建 backend，未知类型或初始化失败时禁止回退。"""

    backend_kind = str(config.backend_kind).strip().lower()
    if backend_kind == "upstream":
        # 延迟导入避免 compatible 单测和离线工具被官方二进制依赖污染。
        from .upstream_backend import UpstreamTomogramBackend

        return UpstreamTomogramBackend(config)
    if backend_kind == "compatible":
        return DirectPCTBackend(config)
    raise ValueError(
        "planner.backend_kind 只允许 upstream 或 compatible，"
        f"收到 {config.backend_kind!r}"
    )


def resolve_project_root(configured: str | Path | None) -> Path:
    """发现包含 PCT core 的仓库根目录，禁止依赖固定绝对路径。"""

    candidates: list[Path] = []
    if configured is not None and str(configured).strip():
        candidates.append(Path(configured).expanduser())
    environment_root = os.environ.get("PCT_SCAN_PROJECT_ROOT")
    if environment_root:
        candidates.append(Path(environment_root).expanduser())
    candidates.extend((Path.cwd(), *Path.cwd().parents))
    module_path = Path(__file__).resolve()
    candidates.extend((module_path.parent, *module_path.parents))

    visited: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        if _looks_like_project_root(resolved):
            return resolved
    raise FileNotFoundError(
        "找不到 pct_scan 仓库根目录；请设置 planner.project_root 或 "
        "PCT_SCAN_PROJECT_ROOT"
    )


def resolve_project_path(
    project_root: Path,
    value: str | Path,
    *,
    field_name: str,
) -> Path:
    """解析仓库相对资产路径并要求目标存在。"""

    raw_path = Path(value).expanduser()
    resolved = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (project_root / raw_path).resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} 不存在: {resolved}")
    return resolved


def xyz_triples(
    values: Sequence[object],
    *,
    field_name: str,
) -> tuple[tuple[float, float, float], ...]:
    """把 ROS 展平数组转换为有限 xyz 点列。"""

    raw = tuple(values)
    if len(raw) % 3 != 0:
        raise ValueError(f"{field_name} 必须由完整 xyz 三元组组成")
    points: list[tuple[float, float, float]] = []
    for index in range(0, len(raw), 3):
        point = tuple(float(raw[index + axis]) for axis in range(3))
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"{field_name} 不能包含 NaN 或 Inf")
        points.append(point)
    return tuple(points)


class _DirectGridClient:
    """用函数调用替代历史 PCT stdin/stdout JSON 子进程。"""

    def __init__(
        self,
        grid_module: ModuleType,
        state: object,
    ) -> None:
        self._grid_module = grid_module
        self._state = state
        self._cancel_event = threading.Event()
        self.last_response: dict[str, Any] | None = None

    def plan(
        self,
        *,
        start: Sequence[float],
        end: Sequence[float],
    ) -> dict[str, Any]:
        self._cancel_event.clear()
        response = self._grid_module.plan_request(
            self._state,
            start=start,
            end=end,
            cancel_check=self._cancel_event.is_set,
        )
        if not isinstance(response, dict):
            raise PCTBackendError("PCT grid core 返回了非对象结果")
        self.last_response = response
        status = str(response.get("status", ""))
        if status == "ok" and isinstance(response.get("traj"), list):
            return response
        if status == "no_path":
            raise PCTNoPathError(_response_message(response, "PCT 未找到路径"))
        raise PCTBackendError(_response_message(response, "PCT 规划异常"))

    def cancel(self) -> None:
        """请求 grid core 尽快结束当前搜索。"""

        self._cancel_event.set()


class DirectPCTBackend:
    """在 ROS 2 节点进程内直接复用现有 PCT grid core。"""

    def __init__(
        self,
        config: PCTBackendConfig,
        *,
        planner_module: ModuleType | None = None,
        grid_module: ModuleType | None = None,
    ) -> None:
        self.config = _validated_config(config)
        if planner_module is None or grid_module is None:
            loaded_planner, loaded_grid = _load_project_modules(
                self.config.project_root
            )
            planner_module = planner_module or loaded_planner
            grid_module = grid_module or loaded_grid
        self._planner_module = planner_module
        self._grid_module = grid_module
        self._legacy_config = _legacy_planner_config(
            planner_module,
            self.config,
        )
        if self.config.collision_ply_path is None:
            raise ValueError(
                "ROS 2 PCT Path 必须配置 collision PLY，"
                "用于把粗 slice z 投影为真实地面高度"
            )

        # grid core 目前在加载时读取环境参数；这里只做一次性配置
        # 转换，不实例化旧 JSON client，也不在 plan 阶段改动环境。
        pct_environment = _grid_load_environment(
            planner_module,
            self.config,
        )
        with _temporary_environment(pct_environment):
            state = grid_module.load_state_from_environment()
        self._client = _DirectGridClient(grid_module, state)
        self._planner = planner_module.PCTNavPlanner(
            self._legacy_config,
            client=self._client,
        )
        support_module = importlib.import_module("source.scene.placement_support")
        vertices, faces = support_module.load_binary_triangle_ply(
            self.config.collision_ply_path
        )
        self._ground_projector = TriangleGroundProjector(
            vertices,
            faces,
            maximum_hint_error_m=(
                self.config.ground_projection_max_z_error_m
            ),
        )

    def plan(
        self,
        *,
        start_base_xyz: Sequence[float],
        goal_base_xyz: Sequence[float],
        goal_yaw: float,
    ) -> PCTBackendPlan:
        """同步执行 PCT；输入是 base 高度，输出保持地面高度。"""

        start = _finite_xyz(start_base_xyz, field_name="start_base_xyz")
        goal = _finite_xyz(goal_base_xyz, field_name="goal_base_xyz")
        yaw = float(goal_yaw)
        if not math.isfinite(yaw):
            raise ValueError("goal_yaw 必须是有限数值")
        exact_start_ground_hint = (
            start[0],
            start[1],
            start[2] - self.config.goal_base_to_ground_m,
        )
        exact_goal_ground_hint = (
            goal[0],
            goal[1],
            goal[2] - self.config.goal_base_to_ground_m,
        )

        def to_pct(point: Sequence[float]) -> tuple[float, float, float]:
            return self._planner_module.sim_to_pct_xyz(
                point,
                coord_mode=self.config.coord_mode,
                pct_offset_x=self.config.pct_offset_x,
                pct_offset_y=self.config.pct_offset_y,
                pct_offset_z=self.config.pct_offset_z,
                pct_scale_x=self.config.pct_scale_x,
                pct_scale_y=self.config.pct_scale_y,
                pct_scale_z=self.config.pct_scale_z,
                pct_rotation_x_rad=self.config.pct_rotation_x_rad,
                pct_rotation_y_rad=self.config.pct_rotation_y_rad,
                pct_rotation_z_rad=self.config.pct_rotation_z_rad,
            )

        start_pct = to_pct(exact_start_ground_hint)
        goal_pct = to_pct(exact_goal_ground_hint)
        try:
            start_projection = self._ground_projector.project(
                x=start_pct[0],
                y=start_pct[1],
                z_hint=start_pct[2],
            )
            terminal_projection = self._ground_projector.project(
                x=goal_pct[0],
                y=goal_pct[1],
                z_hint=goal_pct[2],
            )
        except ValueError as exc:
            raise PCTBackendError(f"PCT 端点地面投影失败：{exc}") from exc
        if (
            start_projection.hint_error_m
            > self.config.start_ground_projection_max_z_error_m
        ):
            raise PCTBackendError(
                "PCT start z 与 collision PLY 支撑面不一致："
                f"base-to-ground 后误差 {start_projection.hint_error_m:.3f} m，"
                "超过 "
                f"{self.config.start_ground_projection_max_z_error_m:.3f} m"
            )
        if (
            terminal_projection.hint_error_m
            > self.config.terminal_ground_projection_max_z_error_m
        ):
            raise PCTBackendError(
                "PCT goal z 与 collision PLY 支撑面不一致："
                f"base-to-ground 后误差 {terminal_projection.hint_error_m:.3f} m，"
                "超过 "
                f"{self.config.terminal_ground_projection_max_z_error_m:.3f} m"
            )
        state = self._planner_module.SimulationState(
            step_index=0,
            timestamp=0.0,
            robot_root_pose=(start[0], start[1], start[2], 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
        )
        goal_message = self._planner_module.NavGoal(
            x=goal[0],
            y=goal[1],
            z=goal[2],
            yaw=yaw,
        )
        try:
            result = self._planner.plan(state, goal_message)
        except Exception as exc:
            typed_cause = _find_typed_cause(exc)
            if isinstance(typed_cause, PCTNoPathError):
                # 不反向把原 cause 对象挂回包装异常，避免形成循环异常链。
                raise PCTNoPathError(str(typed_cause)) from exc
            if typed_cause is not None:
                raise PCTBackendError(str(typed_cause)) from exc
            raise PCTBackendError(str(exc)) from exc

        _enforce_snap_distance(
            result.metadata,
            maximum_snap_distance_m=self.config.maximum_snap_distance_m,
        )
        _enforce_snap_slice_identity(result.metadata)
        _enforce_exact_goal_cell(result.metadata)
        raw_points = result.metadata.get("path_3d")
        if not isinstance(raw_points, (tuple, list)):
            raise PCTBackendError("PCT 结果缺少 path_3d")
        coarse_sim_points = tuple(
            _finite_xyz(point, field_name=f"path_3d[{index}]")
            for index, point in enumerate(raw_points)
        )
        if len(coarse_sim_points) < 2:
            raise PCTBackendError("PCT 路径少于 2 个点")
        goal_appended = (
            math.dist(coarse_sim_points[-1][:2], goal[:2]) > 1.0e-9
        )
        if goal_appended:
            coarse_sim_points = (*coarse_sim_points, exact_goal_ground_hint)
        else:
            # 即使 XY 已经对齐，也必须用请求目标的 base-to-ground
            # 语义选择末端楼层，不能盲信粗粒度 slice z。
            coarse_sim_points = (
                *coarse_sim_points[:-1],
                exact_goal_ground_hint,
            )
        sampled_sim_points = _sample_polyline(
            coarse_sim_points,
            spacing_m=self.config.path_sample_spacing_m,
        )
        coarse_pct_points = tuple(
            self._planner_module.sim_to_pct_xyz(
                point,
                coord_mode=self.config.coord_mode,
                pct_offset_x=self.config.pct_offset_x,
                pct_offset_y=self.config.pct_offset_y,
                pct_offset_z=self.config.pct_offset_z,
                pct_scale_x=self.config.pct_scale_x,
                pct_scale_y=self.config.pct_scale_y,
                pct_scale_z=self.config.pct_scale_z,
                pct_rotation_x_rad=self.config.pct_rotation_x_rad,
                pct_rotation_y_rad=self.config.pct_rotation_y_rad,
                pct_rotation_z_rad=self.config.pct_rotation_z_rad,
            )
            for point in sampled_sim_points
        )
        try:
            projected_pct_points, projection_reports = (
                self._ground_projector.project_path(coarse_pct_points)
            )
        except ValueError as exc:
            raise PCTBackendError(f"collision PLY 地面投影失败：{exc}") from exc
        points = tuple(
            self._planner_module.pct_to_sim_xyz(
                point,
                coord_mode=self.config.coord_mode,
                pct_offset_x=self.config.pct_offset_x,
                pct_offset_y=self.config.pct_offset_y,
                pct_offset_z=self.config.pct_offset_z,
                pct_scale_x=self.config.pct_scale_x,
                pct_scale_y=self.config.pct_scale_y,
                pct_scale_z=self.config.pct_scale_z,
                pct_rotation_x_rad=self.config.pct_rotation_x_rad,
                pct_rotation_y_rad=self.config.pct_rotation_y_rad,
                pct_rotation_z_rad=self.config.pct_rotation_z_rad,
            )
            for point in projected_pct_points
        )
        metadata = dict(result.metadata)
        metadata["coarse_slice_path_3d"] = result.metadata.get("path_3d")
        metadata["path_3d"] = points
        metadata["height_semantics"] = "ground_height"
        metadata["slice_query_root_to_floor_m"] = (
            self.config.slice_query_root_to_floor_m
        )
        metadata["goal_base_to_ground_m"] = self.config.goal_base_to_ground_m
        metadata["start_ground_projection_error_m"] = (
            start_projection.hint_error_m
        )
        metadata["start_ground_projection_face_index"] = (
            start_projection.face_index
        )
        metadata["terminal_ground_projection_error_m"] = (
            terminal_projection.hint_error_m
        )
        metadata["terminal_ground_projection_face_index"] = (
            terminal_projection.face_index
        )
        metadata["ground_projection_source"] = "collision_ply_vertical_intersection"
        maximum_projection_index = max(
            range(len(projection_reports)),
            key=lambda index: projection_reports[index].hint_error_m,
        )
        maximum_projection = projection_reports[maximum_projection_index]
        metadata["ground_projection_max_hint_error_m"] = (
            maximum_projection.hint_error_m
        )
        metadata["ground_projection_max_hint_error_index"] = (
            maximum_projection_index
        )
        metadata["ground_projection_max_hint_input_pct_xyz"] = (
            coarse_pct_points[maximum_projection_index]
        )
        metadata["ground_projection_max_hint_output_pct_xyz"] = (
            projected_pct_points[maximum_projection_index]
        )
        metadata["ground_projection_max_hint_face_index"] = (
            maximum_projection.face_index
        )
        metadata["ground_projection_point_count"] = len(points)
        metadata["requested_goal_appended"] = goal_appended
        metadata["transport"] = "direct_in_process_ros2"
        return PCTBackendPlan(points_xyz=points, metadata=metadata)

    def cancel_current_plan(self) -> None:
        """供 ROS goal 抢占与节点退出请求中止当前 A* 搜索。"""

        self._client.cancel()


def _looks_like_project_root(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in (
            path / "project.json",
            path / "source/navigation/pct_adapter.py",
            path / "scripts/navigation/pct_grid_server.py",
        )
    )


def _load_project_coordinate_module(project_root: Path) -> ModuleType:
    """只加载 pct_scan 统一坐标模块，不连带加载 compatible grid core。"""

    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    planner_module = importlib.import_module("source.navigation.pct_adapter")
    module_file = Path(str(planner_module.__file__)).resolve()
    if project_root not in module_file.parents:
        raise ImportError(
            "PCT planner adapter 来自错误仓库: "
            f"{module_file}，预期位于 {project_root}"
        )
    return planner_module


def _load_project_modules(project_root: Path) -> tuple[ModuleType, ModuleType]:
    planner_module = _load_project_coordinate_module(project_root)
    grid_module = importlib.import_module("scripts.navigation.pct_grid_server")
    module_file = Path(str(grid_module.__file__)).resolve()
    if project_root not in module_file.parents:
        raise ImportError(
            f"PCT grid core 来自错误仓库: {module_file}，"
            f"预期位于 {project_root}"
        )
    return planner_module, grid_module


def _legacy_planner_config(
    planner_module: ModuleType,
    config: PCTBackendConfig,
) -> object:
    return planner_module.PCTPlannerConfig(
        enabled=True,
        planner_root=config.project_root,
        server_script=(
            config.project_root / "scripts/navigation/pct_grid_server.py"
        ),
        tomogram_path=config.tomogram_path,
        walkable_path=config.walkable_path,
        collision_ply_path=config.collision_ply_path,
        global_vertical_obstacle_min_slices=(
            config.global_vertical_obstacle_min_slices
        ),
        cross_floor_vertical_obstacle_min_slices=(
            config.cross_floor_vertical_obstacle_min_slices
        ),
        cross_floor_gateway_points=config.cross_floor_gateway_points,
        cross_floor_stair_exit_points=config.cross_floor_stair_exit_points,
        cross_floor_stair_midpoint_points=(
            config.cross_floor_stair_midpoint_points
        ),
        cross_floor_gateway_radius_m=config.cross_floor_gateway_radius_m,
        robot_root_to_floor_m=config.slice_query_root_to_floor_m,
        body_obstacle_min_height_m=config.body_obstacle_min_height_m,
        body_obstacle_max_height_m=config.body_obstacle_max_height_m,
        stair_min_horizontal_per_slice_m=(
            config.stair_min_horizontal_per_slice_m
        ),
        stair_max_horizontal_per_slice_m=(
            config.stair_max_horizontal_per_slice_m
        ),
        stair_vertical_radius_m=config.stair_vertical_radius_m,
        stair_progress_tolerance=config.stair_progress_tolerance,
        stair_progress_cost_weight=config.stair_progress_cost_weight,
        obstacle_clearance_radius_m=config.obstacle_clearance_radius_m,
        obstacle_clearance_cost_weight=config.obstacle_clearance_cost_weight,
        coord_mode=config.coord_mode,
        pct_offset_x=config.pct_offset_x,
        pct_offset_y=config.pct_offset_y,
        pct_offset_z=config.pct_offset_z,
        pct_scale_x=config.pct_scale_x,
        pct_scale_y=config.pct_scale_y,
        pct_scale_z=config.pct_scale_z,
        pct_rotation_x_rad=config.pct_rotation_x_rad,
        pct_rotation_y_rad=config.pct_rotation_y_rad,
        pct_rotation_z_rad=config.pct_rotation_z_rad,
        fallback_to_astar=False,
    )


def _grid_load_environment(
    planner_module: ModuleType,
    config: PCTBackendConfig,
) -> dict[str, str]:
    """生成 grid core 一次性加载参数，不经过 JSON client。"""

    collision_ply_path = config.collision_ply_path
    if collision_ply_path is None:
        raise ValueError("grid core 必须配置 collision_ply_path")

    def to_pct(point: Sequence[float]) -> tuple[float, float, float]:
        return planner_module.sim_to_pct_xyz(
            point,
            coord_mode=config.coord_mode,
            pct_offset_x=config.pct_offset_x,
            pct_offset_y=config.pct_offset_y,
            pct_offset_z=config.pct_offset_z,
            pct_scale_x=config.pct_scale_x,
            pct_scale_y=config.pct_scale_y,
            pct_scale_z=config.pct_scale_z,
            pct_rotation_x_rad=config.pct_rotation_x_rad,
            pct_rotation_y_rad=config.pct_rotation_y_rad,
            pct_rotation_z_rad=config.pct_rotation_z_rad,
        )

    return {
        "PCT_PLANNER_ROOT": os.fspath(config.project_root),
        "PCT_TOMOGRAM_PATH": os.fspath(config.tomogram_path),
        "PCT_WALKABLE_PATH": os.fspath(config.walkable_path),
        "PCT_COLLISION_PLY_PATH": os.fspath(collision_ply_path),
        "PCT_GLOBAL_VERTICAL_OBSTACLE_MIN_SLICES": str(
            config.global_vertical_obstacle_min_slices
        ),
        "PCT_CROSS_FLOOR_VERTICAL_OBSTACLE_MIN_SLICES": str(
            config.cross_floor_vertical_obstacle_min_slices
        ),
        "PCT_CROSS_FLOOR_GATEWAYS_PCT": _encode_xyz_points(
            tuple(to_pct(point) for point in config.cross_floor_gateway_points)
        ),
        "PCT_CROSS_FLOOR_STAIR_EXITS_PCT": _encode_xyz_points(
            tuple(
                to_pct(point)
                for point in config.cross_floor_stair_exit_points
            )
        ),
        "PCT_CROSS_FLOOR_STAIR_MIDPOINTS_PCT": _encode_xyz_points(
            tuple(
                to_pct(point)
                for point in config.cross_floor_stair_midpoint_points
            )
        ),
        "PCT_CROSS_FLOOR_GATEWAY_RADIUS_M": str(
            config.cross_floor_gateway_radius_m
        ),
        "PCT_ROBOT_ROOT_TO_FLOOR_M": str(
            config.slice_query_root_to_floor_m
        ),
        "PCT_BODY_OBSTACLE_MIN_HEIGHT_M": str(
            config.body_obstacle_min_height_m
        ),
        "PCT_BODY_OBSTACLE_MAX_HEIGHT_M": str(
            config.body_obstacle_max_height_m
        ),
        "PCT_STAIR_MIN_HORIZONTAL_PER_SLICE_M": str(
            config.stair_min_horizontal_per_slice_m
        ),
        "PCT_STAIR_MAX_HORIZONTAL_PER_SLICE_M": str(
            config.stair_max_horizontal_per_slice_m
        ),
        "PCT_STAIR_VERTICAL_RADIUS_M": str(config.stair_vertical_radius_m),
        "PCT_STAIR_PROGRESS_TOLERANCE": str(
            config.stair_progress_tolerance
        ),
        "PCT_STAIR_PROGRESS_COST_WEIGHT": str(
            config.stair_progress_cost_weight
        ),
        "PCT_OBSTACLE_CLEARANCE_RADIUS_M": str(
            config.obstacle_clearance_radius_m
        ),
        "PCT_OBSTACLE_CLEARANCE_COST_WEIGHT": str(
            config.obstacle_clearance_cost_weight
        ),
        "PCT_GRID_MAX_EXPANSIONS": str(config.grid_max_expansions),
        "PCT_GRID_COMPRESS_MAX_SEGMENT_M": str(
            config.grid_compress_max_segment_m
        ),
        "PCT_GRID_TIMEOUT_SEC": str(config.grid_timeout_sec),
    }


def _encode_xyz_points(points: Sequence[Sequence[float]]) -> str:
    """使用 grid core 原生分号格式编码点列，避免 JSON 传输。"""

    return ";".join(
        ",".join(format(float(value), ".17g") for value in point)
        for point in points
    )


def _validated_config(config: PCTBackendConfig) -> PCTBackendConfig:
    project_root = Path(config.project_root).expanduser().resolve()
    tomogram_path = _resolved_config_path(project_root, config.tomogram_path)
    walkable_path = _resolved_config_path(project_root, config.walkable_path)
    collision_ply_path = (
        None
        if config.collision_ply_path is None
        else _resolved_config_path(project_root, config.collision_ply_path)
    )
    upstream_stair_profile_path = (
        None
        if config.upstream_stair_profile_path is None
        else _resolved_config_path(
            project_root,
            config.upstream_stair_profile_path,
        )
    )
    upstream_source_root = (
        (project_root / "external/PCT_planner").resolve()
        if config.upstream_source_root is None
        else _resolved_config_path(project_root, config.upstream_source_root)
    )
    normalized = replace(
        config,
        project_root=project_root,
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        collision_ply_path=collision_ply_path,
        upstream_stair_profile_path=upstream_stair_profile_path,
        backend_kind=str(config.backend_kind).strip().lower(),
        upstream_source_root=upstream_source_root,
        cross_floor_gateway_points=_normalized_xyz_points(
            config.cross_floor_gateway_points,
            field_name="cross_floor_gateway_points",
        ),
        cross_floor_stair_exit_points=_normalized_xyz_points(
            config.cross_floor_stair_exit_points,
            field_name="cross_floor_stair_exit_points",
        ),
        cross_floor_stair_midpoint_points=_normalized_xyz_points(
            config.cross_floor_stair_midpoint_points,
            field_name="cross_floor_stair_midpoint_points",
        ),
    )
    if not _looks_like_project_root(project_root):
        raise FileNotFoundError(
            f"planner.project_root 不是有效 pct_scan 根目录: {project_root}"
        )
    for label, path in (
        ("tomogram_path", normalized.tomogram_path),
        ("walkable_path", normalized.walkable_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在: {path}")
    if (
        normalized.collision_ply_path is not None
        and not normalized.collision_ply_path.is_file()
    ):
        raise FileNotFoundError(
            f"collision_ply_path 不存在: {normalized.collision_ply_path}"
        )
    if (
        normalized.upstream_stair_profile_path is not None
        and not normalized.upstream_stair_profile_path.is_file()
    ):
        raise FileNotFoundError(
            "upstream_stair_profile_path 不存在: "
            f"{normalized.upstream_stair_profile_path}"
        )
    finite_values = (
        normalized.pct_offset_x,
        normalized.pct_offset_y,
        normalized.pct_offset_z,
        normalized.pct_scale_x,
        normalized.pct_scale_y,
        normalized.pct_scale_z,
        normalized.pct_rotation_x_rad,
        normalized.pct_rotation_y_rad,
        normalized.pct_rotation_z_rad,
        normalized.slice_query_root_to_floor_m,
        normalized.goal_base_to_ground_m,
        normalized.body_obstacle_min_height_m,
        normalized.body_obstacle_max_height_m,
        normalized.cross_floor_gateway_radius_m,
        normalized.stair_min_horizontal_per_slice_m,
        normalized.stair_max_horizontal_per_slice_m,
        normalized.stair_vertical_radius_m,
        normalized.stair_progress_tolerance,
        normalized.stair_progress_cost_weight,
        normalized.obstacle_clearance_radius_m,
        normalized.obstacle_clearance_cost_weight,
        normalized.ground_projection_max_z_error_m,
        normalized.start_ground_projection_max_z_error_m,
        normalized.terminal_ground_projection_max_z_error_m,
        normalized.path_sample_spacing_m,
        normalized.maximum_snap_distance_m,
        normalized.grid_compress_max_segment_m,
        normalized.grid_timeout_sec,
        normalized.upstream_max_heading_rate,
        normalized.upstream_astar_step_cost_weight,
        normalized.upstream_body_clearance_radius_m,
        normalized.upstream_body_clearance_maximum_cost,
        normalized.upstream_body_clearance_power,
        normalized.upstream_endpoint_layer_max_z_error_m,
        normalized.upstream_same_layer_shortcut_clearance_m,
        normalized.upstream_same_layer_shortcut_max_segment_m,
        normalized.upstream_stair_profile_match_tolerance_m,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("PCT 坐标和高度参数必须是有限数值")
    if normalized.coord_mode not in {"sim_to_pct_180deg", "identity"}:
        raise ValueError(f"不支持的 planner.coord_mode: {normalized.coord_mode}")
    if normalized.backend_kind not in {"upstream", "compatible"}:
        raise ValueError(
            "planner.backend_kind 只允许 upstream 或 compatible，"
            f"收到 {normalized.backend_kind!r}"
        )
    if normalized.upstream_max_heading_rate <= 0.0:
        raise ValueError("planner.upstream_max_heading_rate 必须为正数")
    if normalized.upstream_astar_step_cost_weight <= 0.0:
        raise ValueError(
            "planner.upstream_astar_step_cost_weight 必须为正数"
        )
    if normalized.upstream_body_clearance_radius_m <= 0.0:
        raise ValueError(
            "planner.upstream_body_clearance_radius_m 必须为正数"
        )
    if not (
        0.0
        < normalized.upstream_body_clearance_maximum_cost
        <= 20.0
    ):
        raise ValueError(
            "planner.upstream_body_clearance_maximum_cost 必须位于 (0, 20]"
        )
    if normalized.upstream_body_clearance_power <= 0.0:
        raise ValueError(
            "planner.upstream_body_clearance_power 必须为正数"
        )
    if normalized.upstream_endpoint_layer_max_z_error_m <= 0.0:
        raise ValueError(
            "planner.upstream_endpoint_layer_max_z_error_m 必须为正数"
        )
    if normalized.upstream_same_layer_shortcut_clearance_m < 0.0:
        raise ValueError(
            "planner.upstream_same_layer_shortcut_clearance_m 不能为负数"
        )
    if normalized.upstream_same_layer_shortcut_max_segment_m <= 0.0:
        raise ValueError(
            "planner.upstream_same_layer_shortcut_max_segment_m 必须为正数"
        )
    if normalized.upstream_stair_profile_match_tolerance_m <= 0.0:
        raise ValueError(
            "planner.upstream_stair_profile_match_tolerance_m 必须为正数"
        )
    if any(
        scale == 0.0
        for scale in (
            normalized.pct_scale_x,
            normalized.pct_scale_y,
            normalized.pct_scale_z,
        )
    ):
        raise ValueError("PCT 坐标 scale 不能为零")
    if normalized.slice_query_root_to_floor_m < 0.0:
        raise ValueError("planner.slice_query_root_to_floor_m 不能为负数")
    if normalized.goal_base_to_ground_m <= 0.0:
        raise ValueError("planner.goal_base_to_ground_m 必须为正数")
    if (
        normalized.backend_kind == "upstream"
        and not math.isclose(
            normalized.slice_query_root_to_floor_m,
            normalized.goal_base_to_ground_m,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(
            "upstream backend 要求 slice_query_root_to_floor_m 与 "
            "goal_base_to_ground_m 使用同一 body_height，禁止重复高度转换"
        )
    if normalized.body_obstacle_min_height_m < 0.0:
        raise ValueError("planner.body_obstacle_min_height_m 不能为负数")
    if (
        normalized.body_obstacle_max_height_m
        <= normalized.body_obstacle_min_height_m
    ):
        raise ValueError("身体障碍高度上限必须大于下限")
    if normalized.cross_floor_gateway_radius_m < 0.0:
        raise ValueError("planner.cross_floor_gateway_radius_m 不能为负数")
    if normalized.stair_min_horizontal_per_slice_m <= 0.0:
        raise ValueError("楼梯每 slice 最小水平行程必须为正数")
    if (
        normalized.stair_max_horizontal_per_slice_m
        < normalized.stair_min_horizontal_per_slice_m
    ):
        raise ValueError("楼梯每 slice 最大水平行程不能小于最小值")
    if normalized.stair_vertical_radius_m <= 0.0:
        raise ValueError("planner.stair_vertical_radius_m 必须为正数")
    if not 0.0 <= normalized.stair_progress_tolerance <= 1.0:
        raise ValueError("planner.stair_progress_tolerance 必须位于 [0, 1]")
    if normalized.stair_progress_cost_weight < 0.0:
        raise ValueError("planner.stair_progress_cost_weight 不能为负数")
    if normalized.obstacle_clearance_radius_m < 0.0:
        raise ValueError("planner.obstacle_clearance_radius_m 不能为负数")
    if normalized.obstacle_clearance_cost_weight < 0.0:
        raise ValueError("planner.obstacle_clearance_cost_weight 不能为负数")
    if normalized.ground_projection_max_z_error_m <= 0.0:
        raise ValueError("planner.ground_projection_max_z_error_m 必须为正数")
    if normalized.start_ground_projection_max_z_error_m <= 0.0:
        raise ValueError(
            "planner.start_ground_projection_max_z_error_m 必须为正数"
        )
    if normalized.terminal_ground_projection_max_z_error_m <= 0.0:
        raise ValueError(
            "planner.terminal_ground_projection_max_z_error_m 必须为正数"
        )
    if (
        normalized.start_ground_projection_max_z_error_m
        > normalized.ground_projection_max_z_error_m
    ):
        raise ValueError("PCT 起点投影门限不能大于路径投影门限")
    if (
        normalized.terminal_ground_projection_max_z_error_m
        > normalized.ground_projection_max_z_error_m
    ):
        raise ValueError("PCT 终点投影门限不能大于路径投影门限")
    if normalized.path_sample_spacing_m <= 0.0:
        raise ValueError("planner.path_sample_spacing_m 必须为正数")
    if normalized.maximum_snap_distance_m < 0.0:
        raise ValueError("planner.maximum_snap_distance_m 不能为负数")
    if normalized.grid_max_expansions < 1:
        raise ValueError("planner.grid_max_expansions 必须至少为 1")
    if normalized.grid_compress_max_segment_m <= 0.0:
        raise ValueError("planner.grid_compress_max_segment_m 必须为正数")
    if normalized.grid_timeout_sec <= 0.0:
        raise ValueError("planner.grid_timeout_sec 必须为正数")
    for field_name, value in (
        (
            "global_vertical_obstacle_min_slices",
            normalized.global_vertical_obstacle_min_slices,
        ),
        (
            "cross_floor_vertical_obstacle_min_slices",
            normalized.cross_floor_vertical_obstacle_min_slices,
        ),
    ):
        if int(value) < 1:
            raise ValueError(f"planner.{field_name} 必须至少为 1")
    return normalized


def _resolved_config_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _normalized_xyz_points(
    points: Sequence[Sequence[float]],
    *,
    field_name: str,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        _finite_xyz(point, field_name=f"{field_name}[{index}]")
        for index, point in enumerate(points)
    )


def _finite_xyz(
    values: Sequence[float],
    *,
    field_name: str,
) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{field_name} 必须包含 3 个坐标")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{field_name} 不能包含 NaN 或 Inf")
    return point


def _response_message(response: dict[str, Any], fallback: str) -> str:
    message = response.get("msg")
    if isinstance(message, str) and message.strip():
        return message.strip()
    status = str(response.get("status", "unknown"))
    return f"{fallback}（status={status}）"


def _enforce_snap_distance(
    metadata: dict[str, Any],
    *,
    maximum_snap_distance_m: float,
) -> None:
    """拒绝把远距离栅格吸附点伪装成用户请求终点。"""

    limit = float(maximum_snap_distance_m)
    for field_name in ("snap_start_distance_m", "snap_end_distance_m"):
        raw_value = metadata.get(field_name)
        if raw_value is None:
            raise PCTBackendError(f"PCT 结果缺少 {field_name}")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise PCTBackendError(f"PCT {field_name} 非法: {raw_value!r}")
        if value > limit:
            raise PCTNoPathError(
                f"PCT {field_name}={value:.3f} m 超过 "
                f"{limit:.3f} m 安全门限"
            )


def _enforce_exact_goal_cell(metadata: dict[str, Any]) -> None:
    """只有请求目标所在栅格本身可走时，才允许接回精确目标坐标。"""

    raw_distance = metadata.get("snap_end_dist")
    if raw_distance is None:
        raise PCTBackendError("PCT 结果缺少 snap_end_dist")
    distance = float(raw_distance)
    if not math.isfinite(distance) or distance < 0.0:
        raise PCTBackendError(f"PCT snap_end_dist 非法: {raw_distance!r}")
    if distance > 0.0:
        raise PCTNoPathError(
            "PCT 请求目标所在栅格不可走，禁止从吸附点直接接回精确目标："
            f"snap_end_dist={distance:g}"
        )


def _enforce_snap_slice_identity(metadata: dict[str, Any]) -> None:
    """防御性确认 grid 吸附点没有落到请求之外的楼层 slice。"""

    for endpoint in ("start", "end"):
        requested_name = f"slice_{endpoint}"
        snapped_name = f"snapped_{endpoint}_slice"
        delta_name = f"snap_{endpoint}_slice_delta"
        values = {
            name: metadata.get(name)
            for name in (requested_name, snapped_name, delta_name)
        }
        if any(value is None for value in values.values()):
            missing = next(
                name for name, value in values.items() if value is None
            )
            raise PCTBackendError(f"PCT 结果缺少 {missing}")
        try:
            requested = int(values[requested_name])
            snapped = int(values[snapped_name])
            delta = int(values[delta_name])
        except (TypeError, ValueError) as exc:
            raise PCTBackendError(
                f"PCT {endpoint} snap slice 元数据非法"
            ) from exc
        if snapped != requested or delta != 0:
            raise PCTNoPathError(
                f"PCT {endpoint} snap 跨层：requested={requested}, "
                f"snapped={snapped}, delta={delta}"
            )


def _sample_polyline(
    points: Sequence[Sequence[float]],
    *,
    spacing_m: float,
) -> tuple[tuple[float, float, float], ...]:
    """按三维弧长重采样最终路径，确保每段都经过真实支撑面投影。"""

    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_m 必须是有限正数")
    coordinates = tuple(
        _finite_xyz(point, field_name=f"polyline[{index}]")
        for index, point in enumerate(points)
    )
    if len(coordinates) < 2:
        raise ValueError("polyline 至少需要 2 个点")
    output: list[tuple[float, float, float]] = [coordinates[0]]
    for start, end in zip(coordinates, coordinates[1:]):
        distance = math.dist(start, end)
        if distance <= 1.0e-9:
            continue
        step_count = max(1, int(math.ceil(distance / spacing)))
        for step in range(1, step_count + 1):
            alpha = step / step_count
            point = tuple(
                start[axis] + alpha * (end[axis] - start[axis])
                for axis in range(3)
            )
            if math.dist(output[-1], point) > 1.0e-9:
                output.append(point)
    if len(output) < 2:
        raise ValueError("polyline 重采样后少于 2 个点")
    return tuple(output)


def _find_typed_cause(exc: BaseException) -> PCTBackendError | None:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, PCTBackendError):
            return current
        current = current.__cause__ or current.__context__
    return None


@contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    """仅在同步地图加载期间设置 PCT 环境参数，退出后完整恢复。"""

    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in updates}
        try:
            os.environ.update(updates)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


_ENVIRONMENT_LOCK = threading.RLock()
