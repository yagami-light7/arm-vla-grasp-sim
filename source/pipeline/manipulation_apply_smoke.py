"""真实 Isaac arm/gripper apply smoke 组装器。"""

from __future__ import annotations

from pathlib import Path

from source.diagnostics import ManipulationApplySmokeVerifier
from source.interfaces import EpisodeSpec
from source.manipulation import (
    BinaryGripperController,
    CuroboJsonManipulationPlanner,
    SegmentedArmExecutor,
    SegmentedArmExecutorConfig,
    SegmentedSmokeManipulationPlanner,
)
from source.navigation.dry_run import DryRunNavExecutor, DryRunNavPlanner
from source.recording import JsonlEpisodeRecorder
from source.simulation import IsaacSimulationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline


def create_manipulation_apply_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacSimulationRuntime,
) -> FullPhysicsPipeline:
    """创建真实 Isaac 中只验证 arm/gripper apply_action 的 pipeline。"""

    gripper = BinaryGripperController()
    if config.pick_plan_json is not None and config.place_plan_json is not None:
        # 这里仅替换 planner 输入来源；执行和 world.step 仍由 pipeline 主循环统一控制。
        manipulation_planner = CuroboJsonManipulationPlanner(
            pick_plan_json=config.pick_plan_json,
            place_plan_json=config.place_plan_json,
        )
    else:
        manipulation_planner = SegmentedSmokeManipulationPlanner()
    return FullPhysicsPipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        simulation=simulation,
        # 本 smoke 不执行 base 导航；导航组件只满足 pipeline 依赖注入协议。
        nav_planner=DryRunNavPlanner(),
        nav_executor=DryRunNavExecutor(),
        manipulation_planner=manipulation_planner,
        arm_executor=SegmentedArmExecutor(
            gripper,
            config=SegmentedArmExecutorConfig(
                sim_dt=0.05,
                gripper_move_duration=0.10,
                gripper_hold_duration=0.05,
            ),
        ),
        gripper=gripper,
        verifier=ManipulationApplySmokeVerifier(),
        recorder=JsonlEpisodeRecorder(episode_dir),
    )
