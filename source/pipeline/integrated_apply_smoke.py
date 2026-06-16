"""单 stage full-physics nav-pick-place 组装器。"""

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


def create_integrated_apply_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacLabNavigationRuntime,
) -> FullPhysicsPipeline:
    """旧 integrated smoke 已取消，避免继续误用离线 place plan。"""

    raise RuntimeError("--integrated-apply-smoke 已取消，请改用 --full-physics。")


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
    full_physics_config = replace(
        config,
        manipulation=replace(
            config.manipulation,
            return_home_after_pick=True,
        ),
    )
    manipulation_planner = CurrentStateCuroboPlanner(
        simulation=simulation,
        config=CurrentStateCuroboPlannerConfig(
            output_dir=Path(episode_dir) / "current_state_curobo",
            project_root=Path(__file__).resolve().parents[2],
            place_plan_json=None,
            # 对齐 random nav-pick-place baseline：侧抓后沿 approach 原路撤回，
            # 避免 vertical lift 及随后 reverse lift-down 与桌面剐蹭。
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
                place_approach_motion_time_scale=(
                    full_physics_config.manipulation.place_approach_motion_time_scale
                ),
                # 相邻切片来自同一条 baseline 轨迹；实际已到切分点时跳过重复 settle，
                # 只有 tracking 落后时才由严格 post-motion hold 等待到位。
                settle_to_segment_start_skip_error_tolerance=0.005,
                post_motion_hold_duration=(
                    full_physics_config.manipulation.arm_post_motion_hold_duration_s
                ),
                post_motion_joint_error_tolerance=0.030,
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
