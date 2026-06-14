"""真实 Isaac Lab 物理导航 smoke 组装器。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from source.diagnostics import NavigationEpisodeVerifier
from source.interfaces import EpisodeSpec
from source.manipulation.dry_run import (
    DryRunArmExecutor,
    DryRunGripperController,
    DryRunManipulationPlanner,
)
from source.navigation import AStarNavPlanner, DwaNavExecutor
from source.navigation.adapters.yaw_align import TerminalPoseConfig
from source.navigation.navlib import DWAConfig
from source.recording import JsonlEpisodeRecorder
from source.simulation import IsaacLabNavigationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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
    """从 pick base goal reset，验证 home arm + close gripper 的 nav_to_place。"""

    if episode_spec.place_goal is None:
        raise ValueError("navigation carry smoke requires an enabled place base goal")
    raw_task = {
        **episode_spec.raw_task,
        "runtime_override": {
            "mode": "navigation_carry_smoke",
            "reset_start_source": "pick.base_goal",
            "object_carry_verified": False,
        },
    }
    carry_spec = replace(
        episode_spec,
        start=episode_spec.pick_goal,
        raw_task=raw_task,
    )
    return _create_navigation_pipeline(
        config=config,
        episode_spec=carry_spec,
        episode_seed=episode_seed,
        episode_dir=episode_dir,
        simulation=simulation,
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
    nav_map = _project_path(episode_spec.nav_map)
    planner = AStarNavPlanner(
        nav_map,
        inflate_radius=nav.global_inflate_radius,
    )
    dwa_config = _build_dwa_config(nav)
    executor = DwaNavExecutor(
        nav_map,
        nav.local_clearance_radius,
        dwa_config,
        terminal_pose_config=TerminalPoseConfig(
            position_tolerance=min(0.08, nav.final_position_tolerance),
            position_acceptance_tolerance=nav.final_position_tolerance,
            yaw_tolerance=nav.final_yaw_tolerance,
            # 保留轻微前向步态激活，避免 Go2 policy 在纯原地转末段站住。
            gait_activation_vx=0.08,
            recovery_gait_vx=0.08,
            yaw_polish_gait_vx=0.08,
            yaw_min_wz=0.55,
            yaw_max_wz=1.00,
            yaw_polish_min_wz=0.55,
            yaw_polish_max_wz=1.00,
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
    )
    verifier = NavigationEpisodeVerifier(
        position_tolerance=nav.final_position_tolerance,
        yaw_tolerance=nav.final_yaw_tolerance,
        linear_velocity_tolerance=nav.stable_linear_velocity,
        angular_velocity_tolerance=nav.stable_angular_velocity,
        require_yaw_alignment=nav.require_yaw_alignment,
        require_stable_base=nav.require_stable_base,
    )
    return planner, executor, verifier


def _build_dwa_config(nav) -> DWAConfig:
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

    return DWAConfig(
        control_dt=nav.control_dt,
        lookahead_distance=lookahead_distance,
        prediction_horizon=prediction_horizon,
        integration_dt=integration_dt,
        goal_tolerance=nav.goal_tolerance,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=nav.max_angular_velocity,
        min_active_linear_velocity=min_active_linear_velocity,
        near_goal_min_active_linear_velocity=near_goal_min_active_linear_velocity,
        close_goal_speed_limit=close_goal_speed_limit,
        speed_bias=speed_bias,
        max_linear_accel=max_linear_accel,
        **config_kwargs,
    )
