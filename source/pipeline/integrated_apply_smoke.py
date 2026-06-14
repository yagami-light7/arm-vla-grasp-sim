"""单 stage full-physics nav-pick-place 组装器。"""

from __future__ import annotations

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
from source.recording import JsonlEpisodeRecorder
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
    manipulation_planner = CurrentStateCuroboPlanner(
        simulation=simulation,
        config=CurrentStateCuroboPlannerConfig(
            output_dir=Path(episode_dir) / "current_state_curobo",
            project_root=Path(__file__).resolve().parents[2],
            place_plan_json=None,
            # full_physics 先沿 approach 反向退到 pregrasp，再由状态机受控回 home。
            # 不能反向回到低位 home，否则 TCP 会下探并拖拽苹果碰桌。
            side_grasp_plan_vertical_lift=False,
            side_grasp_fallback_retreat=False,
            side_grasp_retreat_to_pregrasp=True,
        )
    )

    nav_planner, nav_executor, nav_verifier = create_navigation_components(
        config=config,
        episode_spec=episode_spec,
    )
    gripper = BinaryGripperController()
    return FullPhysicsPipeline(
        config=config,
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
                motion_time_scale=config.manipulation.arm_motion_time_scale,
                place_approach_motion_time_scale=(
                    config.manipulation.place_approach_motion_time_scale
                ),
                post_motion_hold_duration=(
                    config.manipulation.arm_post_motion_hold_duration_s
                ),
                post_open_release_settle_duration=(
                    config.manipulation.place_release_settle_duration_s
                ),
                # 对齐 baseline 的渐进开合时长，避免瞬间张开夹爪把苹果弹滚。
                gripper_move_duration=0.70,
                gripper_hold_duration=0.45,
            ),
        ),
        gripper=gripper,
        verifier=FullPhysicsVerifier(nav_verifier),
        recorder=JsonlEpisodeRecorder(episode_dir),
    )
