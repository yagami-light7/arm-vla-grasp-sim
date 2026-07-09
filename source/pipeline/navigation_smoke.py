"""真实 Isaac Lab 物理导航 smoke 组装器。"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from source.diagnostics import NavigationEpisodeVerifier
from source.interfaces import EpisodeSpec, NavGoal
from source.manipulation.dry_run import (
    DryRunArmExecutor,
    DryRunGripperController,
    DryRunManipulationPlanner,
)
from source.navigation import AStarNavPlanner, DwaNavExecutor, PCTNavPlanner, PCTPlannerConfig
from source.navigation.adapters.yaw_align import TerminalPoseConfig
from source.navigation.navlib import DWAConfig, OccupancyGridMap
from source.navigation.pct_local_map import (
    add_circular_keepouts,
    load_pct_route_local_map,
    load_pct_slice_local_map,
)
from source.recording import JsonlEpisodeRecorder
from source.simulation import IsaacLabNavigationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PCT_SERVER_SCRIPT = PROJECT_ROOT / "scripts/navigation/pct_grid_server.py"
DEFAULT_PCT_TOMOGRAM_PATH = PROJECT_ROOT / "source/scene/multifloor/mutifloor.pickle"
DEFAULT_PCT_WALKABLE_PATH = PROJECT_ROOT / "source/scene/multifloor/mutifloor_ply_walkable.npy"
DEFAULT_PCT_COLLISION_PLY_PATH = PROJECT_ROOT / "source/scene/multifloor/ply/3dgs_collision.ply"


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _optional_project_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    return _project_path(raw_path)


def create_navigation_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """创建只执行 nav_to_pick 的真实物理导航 smoke pipeline。"""

    return _create_navigation_pipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation=simulation,
    )


def create_navigation_carry_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """从抓取后稳定底盘位姿 reset，验证 home arm + close gripper 的 nav_to_place。"""

    if episode_spec.place_goal is None:
        raise ValueError("navigation carry smoke requires an enabled place base goal")
    carry_start, reset_start_source = _navigation_carry_smoke_start(episode_spec)
    raw_task = {
        **episode_spec.raw_task,
        "runtime_override": {
            "mode": "navigation_carry_smoke",
            "reset_start_source": reset_start_source,
            "object_carry_verified": False,
        },
    }
    carry_spec = replace(
        episode_spec,
        start=carry_start,
        raw_task=raw_task,
    )
    return _create_navigation_pipeline(
        config=config,
        episode_spec=carry_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation=simulation,
    )


def _navigation_carry_smoke_start(episode_spec: EpisodeSpec) -> tuple[NavGoal, str]:
    """读取任务级抓取后底盘位姿，未配置时保持旧的 pick goal 行为。"""

    raw_carry = episode_spec.raw_task.get("carry")
    if not isinstance(raw_carry, dict):
        return episode_spec.pick_goal, "pick.base_goal"
    raw_start = raw_carry.get("smoke_start")
    if not isinstance(raw_start, dict):
        return episode_spec.pick_goal, "pick.base_goal"

    missing = [name for name in ("x", "y", "yaw") if name not in raw_start]
    if missing:
        raise ValueError(
            "carry.smoke_start requires x, y and yaw; "
            f"missing={','.join(missing)}"
        )

    pick_goal = episode_spec.pick_goal
    raw_slice_id = raw_start.get("slice_id", pick_goal.slice_id)
    return (
        NavGoal(
            x=float(raw_start["x"]),
            y=float(raw_start["y"]),
            yaw=float(raw_start["yaw"]),
            z=(
                pick_goal.z
                if raw_start.get("z") is None
                else float(raw_start["z"])
            ),
            floor_id=raw_start.get("floor_id", pick_goal.floor_id),
            slice_id=None if raw_slice_id is None else int(raw_slice_id),
        ),
        "carry.smoke_start",
    )


def _create_navigation_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    planner, executor, verifier = create_navigation_components(
        config=config,
        episode_spec=episode_spec,
    )
    gripper = DryRunGripperController()
    return FullPhysicsPipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        simulation=simulation,
        nav_planner=planner,
        nav_executor=executor,
        # navigation-smoke 会在 pick reachable 后退出，manipulation 组件只占位。
        manipulation_planner=DryRunManipulationPlanner(),
        arm_executor=DryRunArmExecutor(gripper),
        gripper=gripper,
        verifier=verifier,
        recorder=JsonlEpisodeRecorder(episode_dir),
    )


def create_navigation_components(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
):
    """创建可被独立 smoke 和连续联调共同复用的导航组件。"""

    nav = config.navigation
    nav_map = _project_path(episode_spec.nav_map) if episode_spec.nav_map else None
    nav_map_exists = nav_map is not None and nav_map.is_file()
    astar_planner = None
    pct_fallback_to_astar = bool(nav.pct_fallback_to_astar and nav_map_exists)
    if nav.global_planner == "astar" or pct_fallback_to_astar:
        if not nav_map_exists:
            raise ValueError(
                "A* planner/fallback requires an existing nav_map; "
                "provide a valid task nav_map or disable PCT fallback with --pct-no-fallback."
            )
        astar_planner = AStarNavPlanner(
            nav_map,
            inflate_radius=nav.global_inflate_radius,
        )
    elif nav.global_planner == "pct" and nav.pct_fallback_to_astar:
        print(
            "[navigation] 任务缺少可用 flat nav_map，已关闭 PCT 到 A* 的 fallback；"
            "后续 PCT 失败会直接报错。",
            flush=True,
        )
    if nav.global_planner == "astar":
        if astar_planner is None:
            raise RuntimeError("A* planner was not initialized")
        planner = astar_planner
    elif nav.global_planner == "pct":
        pct_server_script = _optional_project_path(nav.pct_server_script) or DEFAULT_PCT_SERVER_SCRIPT
        pct_tomogram_path = _optional_project_path(nav.pct_tomogram_path) or DEFAULT_PCT_TOMOGRAM_PATH
        pct_walkable_path = _optional_project_path(nav.pct_walkable_path) or DEFAULT_PCT_WALKABLE_PATH
        pct_collision_ply_path = (
            _optional_project_path(nav.pct_collision_ply_path)
            or DEFAULT_PCT_COLLISION_PLY_PATH
        )
        planner = PCTNavPlanner(
            PCTPlannerConfig(
                enabled=nav.pct_enabled or nav.global_planner == "pct",
                planner_root=_optional_project_path(nav.pct_planner_root),
                server_script=pct_server_script,
                server_python=nav.pct_server_python,
                tomogram_name=nav.pct_tomogram_name,
                tomogram_path=pct_tomogram_path,
                walkable_path=pct_walkable_path,
                collision_ply_path=pct_collision_ply_path,
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
                pct_scale_x=nav.pct_scale_x,
                pct_scale_y=nav.pct_scale_y,
                fallback_to_astar=pct_fallback_to_astar,
            ),
            fallback_planner=astar_planner if pct_fallback_to_astar else None,
        )
    else:
        raise ValueError(f"unsupported global planner: {nav.global_planner}")
    dwa_config = _build_dwa_config(
        nav,
        policy_profile=config.locomotion.policy_profile,
    )
    executor_map_kwargs = (
        {
            "grid_map": _open_pct_local_grid_map(episode_spec, nav),
            "multifloor_grid_map": _open_pct_multifloor_grid_map(
                episode_spec,
                nav,
            ),
            # 全局 PCT 已按路径所在 slice 校验 3D 墙体；旧二维投影会让其他楼层
            # 的墙体错误封住合法楼梯中心线，因此这里不再重复保护。
            "multifloor_protected_obstacle_map": None,
        }
        if nav.global_planner == "pct" and not nav_map_exists
        else {"map_json": nav_map}
    )
    executor_local_clearance_radius = nav.local_clearance_radius
    if nav.global_planner == "pct" and config.locomotion.policy_profile == "pct_multifloor":
        # PCT 0.20m 栅格的窄通道不能再做硬膨胀；墙体由 collision PLY 层和 DWA clearance 软约束处理。
        executor_local_clearance_radius = 0.0
    executor = DwaNavExecutor(
        local_clearance_radius=executor_local_clearance_radius,
        dwa_config=dwa_config,
        **executor_map_kwargs,
        terminal_pose_config=TerminalPoseConfig(
            position_tolerance=min(0.08, nav.final_position_tolerance),
            position_acceptance_tolerance=nav.final_position_tolerance,
            yaw_tolerance=nav.final_yaw_tolerance,
            # 保留轻微前向步态激活，避免 Go2 policy 在纯原地转末段站住。
            gait_activation_vx=0.08,
            recovery_gait_vx=0.08,
            yaw_polish_gait_vx=0.08,
            yaw_min_wz=0.55,
            yaw_max_wz=dwa_config.max_angular_velocity,
            yaw_polish_min_wz=0.55,
            yaw_polish_max_wz=dwa_config.max_angular_velocity,
        ),
        terminal_start_distance=nav.terminal_start_distance,
        position_tolerance=nav.final_position_tolerance,
        yaw_tolerance=nav.final_yaw_tolerance,
        stall_window_steps=nav.stall_window_steps,
        stall_min_progress_m=nav.stall_min_progress,
        command_recompute_interval_steps=nav.dwa_replan_interval_steps,
        completion_linear_velocity_tolerance=(
            nav.stable_linear_velocity if nav.require_stable_base else None
        ),
        completion_angular_velocity_tolerance=(
            nav.stable_angular_velocity if nav.require_stable_base else None
        ),
        require_yaw_alignment=nav.require_yaw_alignment,
        multifloor_obstacle_inflate_radius=(
            nav.pct_multifloor_obstacle_inflate_radius
            if nav.global_planner == "pct"
            else 0.0
        ),
        multifloor_route_corridor_radius=(
            nav.pct_multifloor_route_corridor_radius
            if nav.global_planner == "pct"
            else None
        ),
        carry_max_linear_velocity=(
            nav.pct_carry_max_linear_velocity
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_max_angular_velocity=(
            nav.pct_carry_max_angular_velocity
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_max_linear_accel=(
            nav.pct_carry_max_linear_accel
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_path_deviation_limit=(
            nav.pct_carry_path_deviation_limit
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_initial_alignment_path_deviation_limit=(
            nav.pct_carry_initial_alignment_path_deviation_limit
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_path_recovery_deviation_limit=(
            nav.pct_carry_path_recovery_deviation_limit
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        carry_max_infeasible_recomputes=(
            nav.pct_carry_max_infeasible_recomputes
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else None
        ),
        stair_float_enabled=(
            bool(nav.pct_stair_float_enabled)
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else False
        ),
        stair_float_speed_mps=nav.pct_stair_float_speed_mps,
        stair_float_activation_radius_m=nav.pct_stair_float_activation_radius_m,
        stair_float_completion_radius_m=nav.pct_stair_float_completion_radius_m,
        stair_float_min_z_delta_m=nav.pct_stair_float_min_z_delta_m,
        stair_float_approach_distance_m=nav.pct_stair_float_approach_distance_m,
        stair_float_exit_distance_m=nav.pct_stair_float_exit_distance_m,
        stair_float_settle_time_s=nav.pct_stair_float_settle_time_s,
        stair_float_release_settle_time_s=(
            nav.pct_stair_float_release_settle_time_s
        ),
        stair_float_yaw_lookahead_m=nav.pct_stair_float_yaw_lookahead_m,
        stair_float_min_root_z_offset_m=(
            nav.pct_stair_float_min_root_z_offset_m
        ),
    )
    verifier = NavigationEpisodeVerifier(
        position_tolerance=nav.final_position_tolerance,
        yaw_tolerance=nav.final_yaw_tolerance,
        linear_velocity_tolerance=nav.stable_linear_velocity,
        angular_velocity_tolerance=nav.stable_angular_velocity,
        require_yaw_alignment=nav.require_yaw_alignment,
        require_stable_base=nav.require_stable_base,
        goal_z_tolerance=nav.goal_z_tolerance,
    )
    return planner, executor, verifier


def _open_local_grid_map(episode_spec: EpisodeSpec) -> OccupancyGridMap:
    """为尚未生成 2D local map 的 PCT 场景创建过渡开放地图。"""

    goals = [episode_spec.start, episode_spec.pick_goal]
    if episode_spec.place_goal is not None:
        goals.append(episode_spec.place_goal)
    xs = [float(goal.x) for goal in goals]
    ys = [float(goal.y) for goal in goals]
    margin_m = 10.0
    resolution_m = 0.20
    min_x = min(xs) - margin_m
    max_x = max(xs) + margin_m
    min_y = min(ys) - margin_m
    max_y = max(ys) + margin_m
    width = max(10, int(math.ceil((max_x - min_x) / resolution_m)))
    height = max(10, int(math.ceil((max_y - min_y) / resolution_m)))
    occupancy = np.zeros((height, width), dtype=bool)
    return OccupancyGridMap(
        occupancy,
        resolution_m,
        (min_x, min_y, 0.0),
    )


def _open_pct_local_grid_map(episode_spec: EpisodeSpec, nav) -> OccupancyGridMap:
    """从 PCT walkable slice 创建 DWA 局部避障地图。"""

    tomogram_path = _optional_project_path(nav.pct_tomogram_path) or DEFAULT_PCT_TOMOGRAM_PATH
    walkable_path = _optional_project_path(nav.pct_walkable_path) or DEFAULT_PCT_WALKABLE_PATH
    collision_ply_path = (
        _optional_project_path(nav.pct_collision_ply_path)
        or DEFAULT_PCT_COLLISION_PLY_PATH
    )
    if not tomogram_path.is_file() or not walkable_path.is_file():
        raise ValueError(
            "PCT DWA 局部地图需要 tomogram 和 walkable 资产；"
            "请先运行 scripts/navigation/build_pct_multifloor_assets.py。"
        )
    z = _local_map_reference_z(episode_spec)
    grid_map = load_pct_slice_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        z=z,
        coord_mode=nav.pct_coord_mode,
        pct_offset_x=nav.pct_offset_x,
        pct_offset_y=nav.pct_offset_y,
        pct_scale_x=nav.pct_scale_x,
        pct_scale_y=nav.pct_scale_y,
        slice_neighbor_radius=1,
        free_dilation_radius_cells=0,
        collision_ply_path=collision_ply_path,
        vertical_obstacle_min_slices=nav.pct_vertical_obstacle_min_slices,
        vertical_obstacle_dilation_radius_cells=(
            nav.pct_vertical_obstacle_dilation_radius_cells
        ),
        robot_root_to_floor_m=nav.pct_robot_root_to_floor_m,
        body_obstacle_min_height_m=nav.pct_body_obstacle_min_height_m,
        body_obstacle_max_height_m=nav.pct_body_obstacle_max_height_m,
    )
    keepout_centers: list[tuple[float, float]] = []
    if episode_spec.object_initial_pose is not None:
        keepout_centers.append(
            (
                float(episode_spec.object_initial_pose[0]),
                float(episode_spec.object_initial_pose[1]),
            )
        )
    if keepout_centers:
        grid_map = add_circular_keepouts(
            grid_map,
            centers_xy=keepout_centers,
            radius_m=float(nav.pct_task_object_keepout_radius),
        )
    return grid_map


def _open_pct_multifloor_grid_map(
    episode_spec: EpisodeSpec,
    nav,
) -> OccupancyGridMap | None:
    """为跨楼层 PCT 路径合并高度切片，并叠加跨层墙体。"""

    z_values = _multifloor_route_z_values(episode_spec)
    if z_values is None:
        return None
    tomogram_path = _optional_project_path(nav.pct_tomogram_path) or DEFAULT_PCT_TOMOGRAM_PATH
    walkable_path = _optional_project_path(nav.pct_walkable_path) or DEFAULT_PCT_WALKABLE_PATH
    collision_ply_path = (
        _optional_project_path(nav.pct_collision_ply_path)
        or DEFAULT_PCT_COLLISION_PLY_PATH
    )
    if not collision_ply_path.is_file():
        raise ValueError(
            "PCT 跨楼层 carry 局部避障需要 collision PLY："
            f"{collision_ply_path}"
        )
    grid_map = load_pct_route_local_map(
        tomogram_path=tomogram_path,
        walkable_path=walkable_path,
        z_values=z_values,
        coord_mode=nav.pct_coord_mode,
        pct_offset_x=nav.pct_offset_x,
        pct_offset_y=nav.pct_offset_y,
        pct_scale_x=nav.pct_scale_x,
        pct_scale_y=nav.pct_scale_y,
        slice_neighbor_radius=1,
        free_dilation_radius_cells=0,
        # route slice 并集会把低层墙体误当成高层自由空间；使用跨多个高度
        # 都存在的 PLY 单元恢复墙体，合法楼梯中心线稍后由 executor 保护。
        collision_ply_path=collision_ply_path,
        vertical_obstacle_min_slices=(
            nav.pct_multifloor_vertical_obstacle_min_slices
        ),
        vertical_obstacle_dilation_radius_cells=0,
        robot_root_to_floor_m=nav.pct_robot_root_to_floor_m,
        body_obstacle_min_height_m=nav.pct_body_obstacle_min_height_m,
        body_obstacle_max_height_m=nav.pct_body_obstacle_max_height_m,
    )
    if episode_spec.object_initial_pose is not None:
        grid_map = add_circular_keepouts(
            grid_map,
            centers_xy=(
                (
                    float(episode_spec.object_initial_pose[0]),
                    float(episode_spec.object_initial_pose[1]),
                ),
            ),
            radius_m=float(nav.pct_task_object_keepout_radius),
        )
    return grid_map


def _multifloor_route_z_values(
    episode_spec: EpisodeSpec,
) -> tuple[float, ...] | None:
    """按任务起终楼层生成覆盖中间 slice 的 z 采样。"""

    if episode_spec.place_goal is None or episode_spec.place_goal.z is None:
        return None
    start_z = (
        float(episode_spec.pick_goal.z)
        if episode_spec.pick_goal.z is not None
        else float(episode_spec.start.z or 0.0)
    )
    end_z = float(episode_spec.place_goal.z)
    if abs(end_z - start_z) <= 0.35:
        return None
    z_min = min(start_z, end_z)
    z_max = max(start_z, end_z)
    values = [z_min]
    while values[-1] + 0.5 < z_max:
        values.append(values[-1] + 0.5)
    values.append(z_max)
    return tuple(values)


def _local_map_reference_z(episode_spec: EpisodeSpec) -> float:
    """选择当前导航 smoke 最相关的楼层高度。"""

    runtime_override = episode_spec.raw_task.get("runtime_override", {})
    if (
        isinstance(runtime_override, dict)
        and runtime_override.get("mode") == "navigation_carry_smoke"
        and episode_spec.place_goal is not None
        and episode_spec.place_goal.z is not None
    ):
        return float(episode_spec.place_goal.z)
    if episode_spec.pick_goal.z is not None:
        return float(episode_spec.pick_goal.z)
    if episode_spec.start.z is not None:
        return float(episode_spec.start.z)
    return 0.0


def _build_dwa_config(nav, *, policy_profile: str = "flat") -> DWAConfig:
    """复用旧 baseline 的 brisk-nav / fast-dwa 速度配置。"""

    lookahead_distance = nav.lookahead_distance
    prediction_horizon = nav.prediction_horizon
    integration_dt = nav.control_dt
    config_kwargs = {}
    if nav.fast_dwa:
        # fast-dwa 是旧 baseline 的低计算量 profile：缩短预测、减少采样点。
        lookahead_distance = min(lookahead_distance, 0.30)
        prediction_horizon = min(prediction_horizon, 0.45)
        integration_dt = max(integration_dt, 0.05)
        config_kwargs.update(
            {
                "linear_samples": 3,
                "angular_samples": 7,
                "path_sample_spacing": 0.08,
                "path_distance_window": 80,
            }
        )

    max_linear_velocity = nav.max_linear_velocity
    min_active_linear_velocity = nav.min_active_linear_velocity
    near_goal_min_active_linear_velocity = nav.near_goal_min_active_linear_velocity
    close_goal_speed_limit = nav.close_goal_speed_limit
    speed_bias = nav.speed_bias
    max_linear_accel = nav.max_linear_accel
    if nav.brisk_nav:
        # 使用本地已验证稳定的 brisk 参数，不采用后续未稳定的 1.0 m/s profile。
        max_linear_velocity = max(max_linear_velocity, 0.80)
        min_active_linear_velocity = max(min_active_linear_velocity, 0.55)
        near_goal_min_active_linear_velocity = max(near_goal_min_active_linear_velocity, 0.38)
        close_goal_speed_limit = max(close_goal_speed_limit, 0.35)
        speed_bias = max(speed_bias, 1.10)
        max_linear_accel = max(max_linear_accel, 4.5)

    max_angular_velocity = nav.max_angular_velocity
    if policy_profile == "pct_multifloor":
        # 多楼层场景通道较窄，缩短预测距离以减少前方墙体对合法转弯的误杀。
        prediction_horizon = min(prediction_horizon, 0.35)
        integration_dt = min(integration_dt, 0.05)
        # 实测路线没有 DWA collision rejection；提高巡航速度但仍低于 checkpoint
        # 训练快照的 vx=0.55 m/s、wz=0.60 rad/s 上限。
        max_linear_velocity = min(max_linear_velocity, 0.45)
        min_active_linear_velocity = min(min_active_linear_velocity, 0.25)
        near_goal_min_active_linear_velocity = min(
            near_goal_min_active_linear_velocity,
            0.12,
        )
        close_goal_speed_limit = min(close_goal_speed_limit, 0.18)
        max_angular_velocity = min(max_angular_velocity, 0.50)
        max_linear_accel = min(max_linear_accel, 2.5)
        speed_bias = min(speed_bias, 0.90)
        config_kwargs.update(
            {
                "lookahead_distance": 0.12,
                "waypoint_tolerance": 0.05,
                "obstacle_distance_cap": 1.00,
                "clearance_bias": 0.55,
                "path_deviation_limit": 0.30,
                "use_command_velocity_window": True,
            }
        )

    return DWAConfig(
        control_dt=nav.control_dt,
        lookahead_distance=config_kwargs.pop("lookahead_distance", lookahead_distance),
        prediction_horizon=prediction_horizon,
        integration_dt=integration_dt,
        goal_tolerance=nav.goal_tolerance,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
        min_active_linear_velocity=min_active_linear_velocity,
        near_goal_min_active_linear_velocity=near_goal_min_active_linear_velocity,
        close_goal_speed_limit=close_goal_speed_limit,
        speed_bias=speed_bias,
        max_linear_accel=max_linear_accel,
        **config_kwargs,
    )
