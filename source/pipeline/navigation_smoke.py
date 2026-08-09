"""真实 Isaac Lab 物理导航 smoke 组装器。"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from source.diagnostics import NavigationEpisodeVerifier
from source.interfaces import EpisodeSpec, NavGoal
from source.manipulation.dry_run import (
    DryRunArmExecutor,
    DryRunGripperController,
    DryRunManipulationPlanner,
)
from source.navigation import (
    AStarNavPlanner,
    FixedCommandStairProbeConfig,
    FixedCommandStairProbeExecutor,
    FixedCommandStairProbePlanner,
    PCTNavPlanner,
    PCTPlannerConfig,
    ScanStairFreezeConfig,
    load_scan_reference_path,
)
from source.navigation.executor import DwaNavExecutor
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


def _requires_extended_pct_navigation_limits(
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
) -> bool:
    """按任务声明或起终楼层关系决定是否使用跨楼层长时限。"""

    if config.locomotion.policy_profile != "pct_multifloor":
        return False
    raw_execution = episode_spec.raw_task.get("navigation_execution") or {}
    if not isinstance(raw_execution, dict):
        raise ValueError("task.navigation_execution 必须是对象")
    explicit = raw_execution.get("extended_state_limits")
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise ValueError(
                "task.navigation_execution.extended_state_limits 必须是布尔值"
            )
        return explicit
    start_floor = (episode_spec.raw_task.get("start") or {}).get("floor_id")
    place_floor = (
        ((episode_spec.raw_task.get("place") or {}).get("base_goal") or {}).get(
            "floor_id"
        )
    )
    if start_floor is not None and place_floor is not None:
        return start_floor != place_floor
    # 旧任务没有 scene_profile / navigation_execution 声明。保留其原有长时限，
    # 新增场景则要求在 task 中显式声明能力，避免再按场景名称分支。
    return "scene_profile" not in episode_spec.raw_task


def _navigation_settings_for_episode(settings: Any, episode_spec: EpisodeSpec):
    """把任务声明的终点交接精度统一映射到所有物理导航入口。"""

    raw_config = episode_spec.raw_task.get("navigation_execution")
    if raw_config is None:
        return settings
    if not isinstance(raw_config, dict):
        raise ValueError("task.navigation_execution 必须是对象")

    numeric_fields = (
        "final_position_tolerance",
        "place_position_tolerance",
        "final_yaw_tolerance",
        "stable_linear_velocity",
        "stable_angular_velocity",
    )
    boolean_fields = ("require_yaw_alignment", "require_stable_base")
    updates: dict[str, Any] = {}
    for field_name in numeric_fields:
        if field_name not in raw_config:
            continue
        value = float(raw_config[field_name])
        if value <= 0.0:
            raise ValueError(f"task.navigation_execution.{field_name} 必须大于零")
        updates[field_name] = value
    for field_name in boolean_fields:
        if field_name not in raw_config:
            continue
        value = raw_config[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"task.navigation_execution.{field_name} 必须是布尔值")
        updates[field_name] = value
    return replace(settings, **updates)


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _optional_project_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    return _project_path(raw_path)


def _scan_reference_path_for_episode(episode_spec: EpisodeSpec):
    """读取任务声明的手工 Path；未声明时保留动态 ROS Path 模式。"""

    notes = episode_spec.raw_task.get("notes")
    if notes is None:
        return None
    if not isinstance(notes, dict):
        raise ValueError("task.notes 必须是对象。")
    raw_path = notes.get("online_reference_path")
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("task.notes.online_reference_path 必须是非空路径。")
    return load_scan_reference_path(_project_path(raw_path))


def _episode_declares_scan_reference_path(episode_spec: EpisodeSpec) -> bool:
    """返回任务是否声明由 ROS 手工发布器使用的参考路径。"""

    notes = episode_spec.raw_task.get("notes")
    return isinstance(notes, dict) and notes.get("online_reference_path") is not None


def _scan_stair_freeze_config(nav: Any) -> ScanStairFreezeConfig:
    """把 pipeline 配置转换为与传输层无关的 SCAN 楼梯冻结参数。"""

    return ScanStairFreezeConfig(
        enabled=nav.scan_stair_freeze_enabled,
        speed_mps=nav.scan_stair_freeze_speed_mps,
        activation_radius_m=nav.scan_stair_freeze_activation_radius_m,
        min_component_z_delta_m=(
            nav.scan_stair_freeze_min_component_z_delta_m
        ),
        min_step_z_delta_m=nav.scan_stair_freeze_min_step_z_delta_m,
        min_step_grade=nav.scan_stair_freeze_min_step_grade,
        min_riser_grade_variation=(
            nav.scan_stair_freeze_min_riser_grade_variation
        ),
        max_inter_step_gap_m=nav.scan_stair_freeze_max_inter_step_gap_m,
        approach_distance_m=nav.scan_stair_freeze_approach_distance_m,
        exit_distance_m=nav.scan_stair_freeze_exit_distance_m,
        activation_lookahead_m=(
            nav.scan_stair_freeze_activation_lookahead_m
        ),
        activation_timeout_s=nav.scan_stair_freeze_activation_timeout_s,
        activation_passed_margin_m=(
            nav.scan_stair_freeze_activation_passed_margin_m
        ),
        full_lock_settle_time_s=(
            nav.scan_stair_freeze_full_lock_settle_time_s
        ),
        root_release_settle_time_s=(
            nav.scan_stair_freeze_root_release_settle_time_s
        ),
        post_release_stable_time_s=(
            nav.scan_stair_freeze_post_release_stable_time_s
        ),
        post_release_stabilization_timeout_s=(
            nav.scan_stair_freeze_post_release_stabilization_timeout_s
        ),
        resume_wait_fresh_cmd_timeout_s=(
            nav.scan_stair_freeze_resume_wait_fresh_cmd_timeout_s
        ),
        terminal_goal_hold_timeout_s=(
            nav.scan_stair_freeze_terminal_goal_hold_timeout_s
        ),
        post_release_max_linear_speed_mps=(
            nav.scan_stair_freeze_post_release_max_linear_speed_mps
        ),
        post_release_max_angular_speed_rps=(
            nav.scan_stair_freeze_post_release_max_angular_speed_rps
        ),
        post_release_max_z_error_m=(
            nav.scan_stair_freeze_post_release_max_z_error_m
        ),
        post_release_max_tilt_rad=(
            nav.scan_stair_freeze_post_release_max_tilt_rad
        ),
        yaw_lookahead_m=nav.scan_stair_freeze_yaw_lookahead_m,
        body_height_m=nav.navigation_body_height_m,
        terminal_goal_xy_tolerance_m=(
            nav.scan_stair_freeze_terminal_goal_xy_tolerance_m
        ),
        terminal_goal_z_tolerance_m=(
            nav.scan_stair_freeze_terminal_goal_z_tolerance_m
        ),
        terminal_goal_yaw_tolerance_rad=(
            nav.scan_stair_freeze_terminal_goal_yaw_tolerance_rad
        ),
        min_measured_body_height_m=(
            nav.scan_stair_freeze_min_measured_body_height_m
        ),
        max_measured_body_height_m=(
            nav.scan_stair_freeze_max_measured_body_height_m
        ),
        certified_progress_m=nav.scan_stair_freeze_certified_progress_m,
        require_supervisor_sensor_status=(
            nav.scan_stair_freeze_require_supervisor_sensor_status
        ),
        supervisor_sensor_status_timeout_s=(
            nav.scan_stair_freeze_supervisor_sensor_status_timeout_s
        ),
        default_control_dt_s=nav.control_dt,
        max_control_dt_s=nav.scan_stair_freeze_max_control_dt_s,
    )


def _required_pct_asset_path(
    raw_path: str | Path | None,
    *,
    field_name: str,
) -> Path:
    """解析场景 profile 提供的 PCT 资产，禁止静默使用其他场景地图。"""

    path = _optional_project_path(raw_path)
    if path is None:
        raise ValueError(
            f"PCT 场景缺少 {field_name}；请在 configs/scenes/<scene>.json "
            "中声明对应路径，或通过 CLI 显式传入。"
        )
    return path


def create_navigation_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """创建只执行 nav_to_pick 的真实物理导航 smoke pipeline。"""

    smoke_config = replace(
        config,
        navigation=_navigation_settings_for_episode(
            config.navigation,
            episode_spec,
        ),
    )
    if _requires_extended_pct_navigation_limits(config, episode_spec):
        smoke_config = replace(
            smoke_config,
            limits=replace(
                smoke_config.limits,
                navigation=max(smoke_config.limits.navigation, 12000),
                episode=max(smoke_config.limits.episode, 15000),
            ),
        )
    return _create_navigation_pipeline(
        config=smoke_config,
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
    carry_config = replace(
        config,
        navigation=_navigation_settings_for_episode(
            config.navigation,
            carry_spec,
        ),
    )
    if config.locomotion.policy_profile == "pct_multifloor":
        carry_config = replace(
            carry_config,
            limits=replace(
                carry_config.limits,
                navigation=max(carry_config.limits.navigation, 12000),
                episode=max(carry_config.limits.episode, 15000),
            ),
        )
    return _create_navigation_pipeline(
        config=carry_config,
        episode_spec=carry_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation=simulation,
    )


def create_stair_locomotion_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """验证 PCT→SCAN 楼梯冻结主链；固定速度分支仅作隔离诊断。"""

    fixed_command_probe = bool(
        config.navigation.stair_fixed_command_probe
    )
    if fixed_command_probe:
        stair_spec = _stair_fixed_command_probe_spec(config, episode_spec)
    elif _episode_declares_scan_reference_path(episode_spec):
        # 手工 SCAN 楼梯任务已经提供精确起终点和地面高度 Path，不能再用
        # 旧 PCT gateway 覆盖它们。
        runtime_override = dict(
            episode_spec.raw_task.get("runtime_override") or {}
        )
        runtime_override.update(
            {
                "mode": "scan_stair_freeze_smoke",
                "controller": "scan_stair_freeze",
                "global_planner": "external_ros2_path",
                "scan_enabled": True,
                "pct_enabled": False,
                "dwa_enabled": False,
                "base_pose_lock": True,
                "pure_physics": False,
            }
        )
        stair_spec = replace(
            episode_spec,
            place_goal=None,
            object_prim_path=None,
            object_initial_pose=None,
            place_target_pose=None,
            raw_task={
                **episode_spec.raw_task,
                "runtime_override": runtime_override,
            },
        )
    else:
        stair_spec = _stair_locomotion_smoke_spec(config, episode_spec)
        runtime_override = dict(
            stair_spec.raw_task.get("runtime_override") or {}
        )
        runtime_override.update(
            {
                "controller": "scan_stair_freeze",
                "global_planner": "external_ros2_path",
                "scan_enabled": True,
                "pct_enabled": False,
                "dwa_enabled": False,
                "base_pose_lock": True,
                "pure_physics": False,
            }
        )
        stair_spec = replace(
            stair_spec,
            raw_task={
                **stair_spec.raw_task,
                "runtime_override": runtime_override,
            },
        )
    stair_navigation = replace(
        config.navigation,
        global_planner=("bypassed" if fixed_command_probe else "pct"),
        pct_enabled=False,
        pct_fallback_to_astar=False,
        pct_stair_float_enabled=False,
        body_height_calibration_enabled=bool(
            not fixed_command_probe
            and not _episode_declares_scan_reference_path(episode_spec)
        ),
    )
    stair_config = replace(
        config,
        navigation_smoke=False,
        navigation_carry_smoke=False,
        stair_locomotion_smoke=True,
        full_physics=False,
        navigation=stair_navigation,
        limits=replace(
            config.limits,
            navigation=max(config.limits.navigation, 6000),
            episode=max(config.limits.episode, 8000),
        ),
    )
    if fixed_command_probe:
        planner = FixedCommandStairProbePlanner()
        executor = FixedCommandStairProbeExecutor(
            FixedCommandStairProbeConfig(
                forward_velocity_mps=(
                    config.navigation.stair_probe_forward_velocity_mps
                ),
                drive_duration_s=(
                    config.navigation.stair_probe_drive_duration_s
                ),
            )
        )
    else:
        planner, executor, _ = create_navigation_components(
            config=stair_config,
            episode_spec=stair_spec,
        )
    verifier = NavigationEpisodeVerifier(
        position_tolerance=0.25,
        yaw_tolerance=config.navigation.final_yaw_tolerance,
        linear_velocity_tolerance=config.navigation.stable_linear_velocity,
        angular_velocity_tolerance=config.navigation.stable_angular_velocity,
        require_yaw_alignment=False,
        require_stable_base=True,
        goal_z_tolerance=config.navigation.goal_z_tolerance,
    )
    return _create_navigation_pipeline(
        config=stair_config,
        episode_spec=stair_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation=simulation,
        components=(planner, executor, verifier),
    )


def _stair_fixed_command_probe_spec(
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
) -> EpisodeSpec:
    """保留任务精确起终点，记录固定速度低层探针合同。"""

    if episode_spec.start.z is None or episode_spec.pick_goal.z is None:
        raise ValueError("楼梯固定速度探针要求任务提供 start/pick base z。")
    runtime_override = dict(
        episode_spec.raw_task.get("runtime_override") or {}
    )
    runtime_override.update(
        {
            "mode": "stair_fixed_command_probe",
            "controller": "fixed_body_velocity_probe",
            "global_planner": "bypassed",
            "scan_enabled": False,
            "pct_enabled": False,
            "float_enabled": False,
            "base_pose_lock": False,
            "requested_command_vx_vy_wz": [
                config.navigation.stair_probe_forward_velocity_mps,
                0.0,
                0.0,
            ],
            "warmup_duration_s": 1.0,
            "drive_duration_s": (
                config.navigation.stair_probe_drive_duration_s
            ),
            "start_xyz_yaw": [
                episode_spec.start.x,
                episode_spec.start.y,
                episode_spec.start.z,
                episode_spec.start.yaw,
            ],
            "goal_xyz_yaw": [
                episode_spec.pick_goal.x,
                episode_spec.pick_goal.y,
                episode_spec.pick_goal.z,
                episode_spec.pick_goal.yaw,
            ],
        }
    )
    return replace(
        episode_spec,
        instruction=(
            "固定初始航向并以恒定机体系前进速度验证低层 locomotion "
            "checkpoint 的真实楼梯响应。"
        ),
        place_goal=None,
        object_prim_path=None,
        object_initial_pose=None,
        place_target_pose=None,
        raw_task={
            **episode_spec.raw_task,
            "runtime_override": runtime_override,
        },
    )


def _stair_locomotion_smoke_spec(
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
) -> EpisodeSpec:
    """用 PCT 楼梯入口和出口构造只包含楼梯路段的 episode。"""

    nav = config.navigation
    if not nav.pct_cross_floor_gateway_points:
        raise ValueError("stair locomotion smoke requires a PCT stair gateway")
    if not nav.pct_cross_floor_stair_exit_points:
        raise ValueError("stair locomotion smoke requires a PCT stair exit")
    if episode_spec.place_goal is None or episode_spec.place_goal.z is None:
        raise ValueError("stair locomotion smoke requires an upstairs place goal z")

    gateway = nav.pct_cross_floor_gateway_points[0]
    stair_exit = nav.pct_cross_floor_stair_exit_points[0]
    if not nav.pct_cross_floor_stair_midpoint_points:
        raise ValueError("stair locomotion smoke requires PCT stair midpoint points")
    start_root_z = float(
        episode_spec.pick_goal.z
        if episode_spec.pick_goal.z is not None
        else episode_spec.start.z
        if episode_spec.start.z is not None
        else 0.0
    )
    end_root_z = float(episode_spec.place_goal.z)
    entry_reference = min(
        nav.pct_cross_floor_stair_midpoint_points,
        key=lambda point: abs(float(point[2]) - float(gateway[2])),
    )
    exit_reference = min(
        nav.pct_cross_floor_stair_midpoint_points,
        key=lambda point: abs(float(point[2]) - float(stair_exit[2])),
    )
    start_heading = math.atan2(
        float(entry_reference[1]) - float(gateway[1]),
        float(entry_reference[0]) - float(gateway[0]),
    )
    end_heading = math.atan2(
        float(stair_exit[1]) - float(exit_reference[1]),
        float(stair_exit[0]) - float(exit_reference[0]),
    )
    exit_extension_m = float(nav.pct_stair_locomotion_exit_extension_m)
    terminal_x = float(stair_exit[0]) + exit_extension_m * math.cos(end_heading)
    terminal_y = float(stair_exit[1]) + exit_extension_m * math.sin(end_heading)
    start = NavGoal(
        x=float(gateway[0]),
        y=float(gateway[1]),
        z=start_root_z,
        yaw=start_heading,
        floor_id=episode_spec.pick_goal.floor_id,
        slice_id=episode_spec.pick_goal.slice_id,
    )
    goal = NavGoal(
        x=terminal_x,
        y=terminal_y,
        z=end_root_z,
        yaw=end_heading,
        floor_id=episode_spec.place_goal.floor_id,
        slice_id=episode_spec.place_goal.slice_id,
    )
    raw_task = {
        **episode_spec.raw_task,
        "runtime_override": {
            "mode": "stair_locomotion_smoke",
            "controller": "stair_heading_tracker",
            "global_planner": "pct",
            "global_path": "pct_online_path_3d",
            "manual_centerline": False,
            "float_enabled": False,
            "stair_exit_xyz": [
                float(stair_exit[0]),
                float(stair_exit[1]),
                float(stair_exit[2]),
            ],
            "exit_extension_m": exit_extension_m,
            "start_xyz_yaw": [start.x, start.y, start.z, start.yaw],
            "goal_xyz_yaw": [goal.x, goal.y, goal.z, goal.yaw],
        },
    }
    return replace(
        episode_spec,
        instruction="测试低层 locomotion policy 能否沿 PCT 在线规划路径完成上楼。",
        start=start,
        pick_goal=goal,
        place_goal=None,
        object_prim_path=None,
        object_initial_pose=None,
        place_target_pose=None,
        raw_task=raw_task,
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
    components=None,
) -> FullPhysicsPipeline:
    if components is None:
        planner, executor, verifier = create_navigation_components(
            config=config,
            episode_spec=episode_spec,
        )
    else:
        planner, executor, verifier = components
    config = enable_production_pct_goal_body_height_calibration(
        config,
        planner=planner,
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
        recorder=JsonlEpisodeRecorder(
            episode_dir,
            diagnostic_frame_stride=config.diagnostic_frame_stride,
        ),
    )


def enable_production_pct_goal_body_height_calibration(
    config: FullPhysicsConfig,
    *,
    planner: Any,
) -> FullPhysicsConfig:
    """所有在线 PCT goal 在发布前都必须完成统一机体高度投影。"""

    if getattr(planner, "publish_pct_goal", False) is not True:
        return config
    if config.navigation.body_height_calibration_enabled:
        return config
    return replace(
        config,
        navigation=replace(
            config.navigation,
            body_height_calibration_enabled=True,
        ),
    )


def _create_navigation_planner(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
) -> AStarNavPlanner | PCTNavPlanner:
    """创建完整 pipeline 与楼梯 smoke 共用的全局规划器。"""

    nav = config.navigation
    if nav.pct_fallback_to_astar:
        raise ValueError("pct-scan 分支禁止 PCT→A* fallback")
    nav_map = _project_path(episode_spec.nav_map) if episode_spec.nav_map else None
    nav_map_exists = nav_map is not None and nav_map.is_file()
    astar_planner = None
    if nav.global_planner == "astar":
        if not nav_map_exists:
            raise ValueError(
                "A* planner requires an existing nav_map; "
                "provide a valid task nav_map."
            )
        astar_planner = AStarNavPlanner(
            nav_map,
            inflate_radius=nav.global_inflate_radius,
        )
    if nav.global_planner == "astar":
        if astar_planner is None:
            raise RuntimeError("A* planner was not initialized")
        planner = astar_planner
    elif nav.global_planner == "pct":
        pct_server_script = _optional_project_path(nav.pct_server_script) or DEFAULT_PCT_SERVER_SCRIPT
        pct_tomogram_path = _required_pct_asset_path(
            nav.pct_tomogram_path,
            field_name="pct_tomogram_path",
        )
        pct_walkable_path = _required_pct_asset_path(
            nav.pct_walkable_path,
            field_name="pct_walkable_path",
        )
        pct_collision_ply_path = _required_pct_asset_path(
            nav.pct_collision_ply_path,
            field_name="pct_collision_ply_path",
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
                pct_offset_z=nav.pct_offset_z,
                pct_scale_x=nav.pct_scale_x,
                pct_scale_y=nav.pct_scale_y,
                pct_scale_z=nav.pct_scale_z,
                pct_rotation_x_rad=nav.pct_rotation_x_rad,
                pct_rotation_y_rad=nav.pct_rotation_y_rad,
                pct_rotation_z_rad=nav.pct_rotation_z_rad,
                fallback_to_astar=False,
            ),
        )
    else:
        raise ValueError(f"unsupported global planner: {nav.global_planner}")
    return planner


def create_navigation_components(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
):
    """为 pct-scan 主 pipeline 创建唯一的 ROS 2 SCAN 导航组件。"""

    from source.navigation.scan_ros2_executor import (
        ScanRos2LifecyclePlanner,
        ScanRos2NavExecutor,
        ScanRos2NavExecutorConfig,
    )

    nav = config.navigation
    reference_path = _scan_reference_path_for_episode(episode_spec)
    return (
        ScanRos2LifecyclePlanner(
            reference_path=reference_path,
            publish_pct_goal=reference_path is None,
        ),
        ScanRos2NavExecutor(
            ScanRos2NavExecutorConfig(
                require_live_reference_path=True,
            ),
            stair_freeze_config=_scan_stair_freeze_config(nav),
            allow_carry_object_follow=bool(episode_spec.object_prim_path),
        ),
        _create_navigation_verifier(nav),
    )


def _create_legacy_navigation_components_for_tests(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
):
    """只为历史隔离单测创建旧 PCT/A* + DWA 组件，主 pipeline 禁止调用。"""

    nav = config.navigation
    nav_map = _project_path(episode_spec.nav_map) if episode_spec.nav_map else None
    nav_map_exists = nav_map is not None and nav_map.is_file()
    planner = _create_navigation_planner(
        config=config,
        episode_spec=episode_spec,
    )
    dwa_config = _build_dwa_config(
        nav,
        policy_profile=config.locomotion.policy_profile,
    )
    executor_map_kwargs = (
        {
            "grid_map": _open_pct_local_grid_map(episode_spec, nav),
            "carry_grid_map": _open_pct_carry_local_grid_map(
                episode_spec,
                nav,
            ),
            "multifloor_grid_map": _open_pct_multifloor_grid_map(
                episode_spec,
                nav,
            ),
            "post_stair_grid_map": _open_pct_post_stair_grid_map(
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
        stall_min_forward_command=nav.stall_min_forward_command,
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
        multifloor_no_float_clearance_radius=(
            # 无漂移或仅在台阶处漂移时，F1 都按 Go2 落足扫掠范围保留墙柱净距。
            nav.local_clearance_radius
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
                and (
                    not nav.pct_stair_float_enabled
                    or nav.pct_stair_float_approach_distance_m <= 1.0e-6
                )
            )
            else 0.0
        ),
        post_stair_clearance_radius=(
            # float 释放后在目标楼层恢复真实步态，必须按底盘足迹
            # 优化 PCT 尾段，不能继续使用贴障碍的点机器最短路。
            nav.local_clearance_radius
            if (
                nav.global_planner == "pct"
                and config.locomotion.policy_profile == "pct_multifloor"
            )
            else 0.0
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
        carry_position_tolerance=nav.place_position_tolerance,
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
        stair_float_release_root_z_offset_m=(
            nav.pct_stair_float_release_root_z_offset_m
        ),
    )
    verifier = _create_navigation_verifier(nav)
    return planner, executor, verifier


def _create_navigation_verifier(nav: Any) -> NavigationEpisodeVerifier:
    """创建 SCAN 与旧离线测试共用的最终位姿验收器。"""

    return NavigationEpisodeVerifier(
        position_tolerance=nav.final_position_tolerance,
        place_position_tolerance=nav.place_position_tolerance,
        yaw_tolerance=nav.final_yaw_tolerance,
        linear_velocity_tolerance=nav.stable_linear_velocity,
        angular_velocity_tolerance=nav.stable_angular_velocity,
        require_yaw_alignment=nav.require_yaw_alignment,
        require_stable_base=nav.require_stable_base,
        goal_z_tolerance=nav.goal_z_tolerance,
    )


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

    return _open_pct_grid_map_at_z(
        episode_spec,
        nav,
        z=_local_map_reference_z(episode_spec),
        include_task_object_keepout=True,
        navigation_phase="nav_to_pick",
    )


def _open_pct_carry_local_grid_map(
    episode_spec: EpisodeSpec,
    nav,
) -> OccupancyGridMap:
    """创建夹持导航地图，移除已被抓走目标的过期 keepout。"""

    return _open_pct_grid_map_at_z(
        episode_spec,
        nav,
        z=_local_map_reference_z(episode_spec),
        include_task_object_keepout=False,
        navigation_phase="nav_to_place",
    )


def _open_pct_post_stair_grid_map(
    episode_spec: EpisodeSpec,
    nav,
) -> OccupancyGridMap | None:
    """跨楼层任务按 place 高度创建解冻后的目标楼层地图。"""

    if episode_spec.place_goal is None or episode_spec.place_goal.z is None:
        return None
    z_values = _multifloor_route_z_values(episode_spec)
    if z_values is None:
        return None
    return _open_pct_grid_map_at_z(
        episode_spec,
        nav,
        z=float(episode_spec.place_goal.z),
        include_task_object_keepout=False,
        navigation_phase="nav_to_place",
    )


def _task_navigation_keepouts(
    episode_spec: EpisodeSpec,
    *,
    navigation_phase: str,
) -> tuple[dict[str, Any], ...]:
    """读取任务动态桌子等局部导航障碍，支持每个障碍独立半径。"""

    raw_keepouts = episode_spec.raw_task.get("navigation_dynamic_keepouts")
    if raw_keepouts is None:
        return ()
    if not isinstance(raw_keepouts, list):
        raise ValueError("task.navigation_dynamic_keepouts 必须是数组")
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_keepout in enumerate(raw_keepouts):
        field = f"task.navigation_dynamic_keepouts[{index}]"
        if not isinstance(raw_keepout, dict):
            raise ValueError(f"{field} 必须是对象")
        keepout_id = str(raw_keepout.get("id") or index)
        if keepout_id in seen_ids:
            raise ValueError(f"{field}.id 重复: {keepout_id}")
        seen_ids.add(keepout_id)
        phases = raw_keepout.get("phases", ["nav_to_pick", "nav_to_place"])
        if not isinstance(phases, list) or not all(
            isinstance(value, str) for value in phases
        ):
            raise ValueError(f"{field}.phases 必须是字符串数组")
        if navigation_phase not in phases:
            continue
        center = raw_keepout.get("center_xy")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError(f"{field}.center_xy 必须包含两个数值")
        try:
            center_xy = (float(center[0]), float(center[1]))
            radius_m = float(raw_keepout.get("radius_m"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 的 center_xy/radius_m 必须是数值") from exc
        if not all(math.isfinite(value) for value in (*center_xy, radius_m)):
            raise ValueError(f"{field} 包含非有限数值")
        if radius_m <= 0.0:
            raise ValueError(f"{field}.radius_m 必须大于零")
        output.append(
            {
                **raw_keepout,
                "id": keepout_id,
                "center_xy": center_xy,
                "radius_m": radius_m,
            }
        )
    return tuple(output)


def _open_pct_grid_map_at_z(
    episode_spec: EpisodeSpec,
    nav,
    *,
    z: float,
    include_task_object_keepout: bool = True,
    navigation_phase: str = "nav_to_pick",
) -> OccupancyGridMap:
    """按指定世界高度创建 PCT 单楼层局部避障地图。"""

    tomogram_path = _required_pct_asset_path(
        nav.pct_tomogram_path,
        field_name="pct_tomogram_path",
    )
    walkable_path = _required_pct_asset_path(
        nav.pct_walkable_path,
        field_name="pct_walkable_path",
    )
    collision_ply_path = _required_pct_asset_path(
        nav.pct_collision_ply_path,
        field_name="pct_collision_ply_path",
    )
    if not tomogram_path.is_file() or not walkable_path.is_file():
        raise ValueError(
            "PCT DWA 局部地图需要 tomogram 和 walkable 资产；"
            "请先运行 scripts/navigation/build_pct_multifloor_assets.py。"
        )
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
    if include_task_object_keepout and episode_spec.object_initial_pose is not None:
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
    for keepout in _task_navigation_keepouts(
        episode_spec,
        navigation_phase=navigation_phase,
    ):
        grid_map = add_circular_keepouts(
            grid_map,
            centers_xy=[keepout["center_xy"]],
            radius_m=float(keepout["radius_m"]),
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
    tomogram_path = _required_pct_asset_path(
        nav.pct_tomogram_path,
        field_name="pct_tomogram_path",
    )
    walkable_path = _required_pct_asset_path(
        nav.pct_walkable_path,
        field_name="pct_walkable_path",
    )
    collision_ply_path = _required_pct_asset_path(
        nav.pct_collision_ply_path,
        field_name="pct_collision_ply_path",
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
    # 跨层地图只用于 pick 完成后的 carry 导航。目标物已随机器人
    # 移动，不能再把它的初始位置保留为静态 keepout。
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
        # 低于 0.25 m/s 的命令落入当前 policy 的站立死区，会造成临近目标时
        # 先停住再偶发迈步。保持有效步态速度，进入验收半径后由 executor 直接切零。
        near_goal_min_active_linear_velocity = 0.30
        close_goal_speed_limit = 0.30
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
                "enforce_min_active_linear_velocity": True,
                "min_active_angular_velocity": 0.30,
                "enforce_min_active_angular_velocity": True,
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
