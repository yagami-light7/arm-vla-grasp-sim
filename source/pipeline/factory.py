"""Full-physics nav-pick-place pipeline factory."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from source.diagnostics import FullPhysicsVerifier
from source.interfaces import EpisodeSpec
from source.manipulation import (
    BinaryGripperController,
    CurrentStateCuroboPlanner,
    CurrentStateCuroboPlannerConfig,
    SegmentedArmExecutor,
    SegmentedArmExecutorConfig,
)
from source.recording import JsonlEpisodeRecorder, LeRobotRecordingConfig
from source.simulation import IsaacLabNavigationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline
from .navigation_smoke import create_navigation_components


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


def _navigation_settings_for_episode(settings, episode_spec: EpisodeSpec):
    """把任务声明的终点交接精度映射到现有 NavigationSettings。"""

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
    updates = {}
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


def _manipulation_settings_for_episode(settings, episode_spec: EpisodeSpec):
    """把任务声明的抓放执行约束映射到现有 ManipulationSettings。"""

    raw_config = episode_spec.raw_task.get("manipulation_execution")
    if raw_config is None:
        return settings
    if not isinstance(raw_config, dict):
        raise ValueError("task.manipulation_execution 必须是对象")
    updates = {}
    for field_name in ("reuse_pick_grasp_orientation_for_place",):
        if field_name not in raw_config:
            continue
        value = raw_config[field_name]
        if not isinstance(value, bool):
            raise ValueError(
                f"task.manipulation_execution.{field_name} 必须是布尔值"
            )
        updates[field_name] = value
    for field_name in ("place_release_object_tcp_offset_tolerance_m",):
        if field_name not in raw_config:
            continue
        value = float(raw_config[field_name])
        if value < 0.0:
            raise ValueError(
                f"task.manipulation_execution.{field_name} 必须为非负数"
            )
        updates[field_name] = value
    return replace(settings, **updates)


def create_full_physics_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """在同一 IsaacLab runtime 中连续调度 nav、在线 cuRobo、arm 和 gripper。"""

    if config.pick_plan_json is not None or config.place_plan_json is not None:
        raise ValueError("full-physics 模式禁止使用离线 pick/place plan JSON。")
    # 完整 pipeline 在 pick 后必须先回到抓取起始姿态，才能释放 root/support lock
    # 并进入 carry nav。这里使用规划 motion 的反向轨迹，不生成无避障 all-zero 直连。
    pct_multifloor = config.locomotion.policy_profile == "pct_multifloor"
    extended_pct_navigation_limits = _requires_extended_pct_navigation_limits(
        config,
        episode_spec,
    )
    navigation_settings = _navigation_settings_for_episode(
        config.navigation,
        episode_spec,
    )
    manipulation_settings = _manipulation_settings_for_episode(
        config.manipulation,
        episode_spec,
    )
    full_physics_config = replace(
        config,
        navigation=navigation_settings,
        limits=(
            replace(
                config.limits,
                navigation=max(config.limits.navigation, 12000),
                episode=max(config.limits.episode, 24000),
            )
            if extended_pct_navigation_limits
            else config.limits
        ),
        manipulation=replace(
            manipulation_settings,
            return_home_after_pick=True,
            settle_object_before_navigation=(
                config.manipulation.settle_object_before_navigation
                or pct_multifloor
            ),
            settle_base_before_navigation=(
                config.manipulation.settle_base_before_navigation
                or pct_multifloor
            ),
        ),
    )
    manipulation_planner = CurrentStateCuroboPlanner(
        simulation=simulation,
        config=CurrentStateCuroboPlannerConfig(
            output_dir=Path(episode_dir) / "current_state_curobo",
            project_root=Path(__file__).resolve().parents[2],
            place_plan_json=None,
            # 该开关只约束 side grasp；top-down 会按目标语义执行竖直 lift。
            side_grasp_plan_vertical_lift=False,
            side_grasp_fallback_retreat=False,
            side_grasp_retreat_to_pregrasp=False,
            split_pregrasp_motion=True,
            reuse_pick_grasp_orientation_for_place=(
                full_physics_config.manipulation.reuse_pick_grasp_orientation_for_place
            ),
        )
    )

    nav_planner, nav_executor, nav_verifier = create_navigation_components(
        config=full_physics_config,
        episode_spec=episode_spec,
    )
    gripper = BinaryGripperController()
    post_motion_hold_duration = (
        full_physics_config.manipulation.arm_post_motion_hold_duration_s
    )
    post_motion_joint_error_tolerance = (
        full_physics_config.manipulation.arm_post_motion_joint_error_tolerance
    )
    if full_physics_config.locomotion.policy_profile == "pct_multifloor":
        # DogOnly 多楼层策略不直接暴露机械臂 action 槽位，真实跟踪会通过
        # 独立 position target 收敛；这里仅放宽 PCT profile 的终端收敛判定。
        # top-down 接触后各关节的独立 position target 会保留小量稳态残差；门禁使用
        # 六关节 L2 norm（不是单关节 max），因此 0.070 rad 仍对应更小的逐关节误差。
        post_motion_hold_duration = max(post_motion_hold_duration, 1.50)
        post_motion_joint_error_tolerance = max(post_motion_joint_error_tolerance, 0.070)
    return FullPhysicsPipeline(
        config=full_physics_config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        simulation=simulation,
        nav_planner=nav_planner,
        nav_executor=nav_executor,
        manipulation_planner=manipulation_planner,
        # executor 只逐 tick 产生命令，唯一 physics step 仍由 pipeline 持有。
        arm_executor=SegmentedArmExecutor(
            gripper,
            config=SegmentedArmExecutorConfig(
                # 2 倍速后每个 50 Hz 仿真步都更新关节目标，避免低频命令产生大跳变。
                arm_command_dt=0.02,
                motion_time_scale=full_physics_config.manipulation.arm_motion_time_scale,
                pick_approach_motion_time_scale=(
                    full_physics_config.manipulation.pick_approach_motion_time_scale
                ),
                place_move_to_pre_place_motion_time_scale=(
                    full_physics_config.manipulation.place_move_to_pre_place_motion_time_scale
                ),
                place_approach_motion_time_scale=(
                    full_physics_config.manipulation.place_approach_motion_time_scale
                ),
                place_retreat_motion_time_scale=(
                    full_physics_config.manipulation.place_retreat_motion_time_scale
                ),
                # 相邻切片来自同一条 baseline 轨迹；实际已到切分点时跳过重复 settle，
                # 只有 tracking 落后时才由严格 post-motion hold 等待到位。
                settle_to_segment_start_skip_error_tolerance=0.005,
                post_motion_hold_duration=post_motion_hold_duration,
                post_motion_joint_error_tolerance=post_motion_joint_error_tolerance,
                place_release_joint_error_tolerance=(
                    full_physics_config.manipulation.place_release_joint_error_tolerance
                ),
                place_release_joint_velocity_tolerance=(
                    full_physics_config.manipulation.place_release_joint_velocity_tolerance
                ),
                place_release_stability_window_duration=(
                    full_physics_config.manipulation.place_release_stability_window_duration_s
                ),
                fail_on_strict_post_motion_state_unavailable=True,
                require_close_progress_for_motion=True,
                post_open_release_settle_duration=(
                    full_physics_config.manipulation.place_release_settle_duration_s
                ),
                # 对齐 baseline 的渐进开合时长，避免瞬间张开夹爪把苹果弹滚。
                gripper_move_duration=0.70,
                gripper_hold_duration=0.30,
            ),
        ),
        gripper=gripper,
        verifier=FullPhysicsVerifier(nav_verifier),
        recorder=JsonlEpisodeRecorder(
            episode_dir,
            lerobot_config=LeRobotRecordingConfig(
                enabled=full_physics_config.recording.enabled,
                control_dt=full_physics_config.navigation.control_dt,
                dataset_fps=full_physics_config.recording.dataset_fps,
                image_height=full_physics_config.recording.image_height,
                image_width=full_physics_config.recording.image_width,
                jpeg_quality=full_physics_config.recording.jpeg_quality,
                chunks_size=full_physics_config.recording.chunks_size,
                camera_keys=full_physics_config.recording.camera_keys,
                primary_camera_key=full_physics_config.recording.primary_camera_key,
                save_raw_images=full_physics_config.recording.save_raw_images,
                debug_per_episode_lerobot=(
                    full_physics_config.recording.debug_per_episode_lerobot
                ),
                unified_dataset=full_physics_config.recording.unified_dataset,
                validate_export=full_physics_config.recording.validate_export,
            ),
        ),
    )
