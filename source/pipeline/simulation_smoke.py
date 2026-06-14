"""真实 Isaac 场景构建与 reset 冒烟组装器。"""

from __future__ import annotations

from pathlib import Path

from source.diagnostics import DryRunEpisodeVerifier
from source.interfaces import EpisodeSpec
from source.manipulation.dry_run import (
    DryRunArmExecutor,
    DryRunGripperController,
    DryRunManipulationPlanner,
)
from source.navigation.dry_run import DryRunNavExecutor, DryRunNavPlanner
from source.recording import JsonlEpisodeRecorder
from source.simulation import IsaacSimulationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline


def create_simulation_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
    simulation: IsaacSimulationRuntime,
) -> FullPhysicsPipeline:
    """创建在真实场景 reset 成功后立即结束的 pipeline。"""

    gripper = DryRunGripperController()
    return FullPhysicsPipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        simulation=simulation,
        # smoke 在 reset 后结束，以下组件不会被调用。
        nav_planner=DryRunNavPlanner(),
        nav_executor=DryRunNavExecutor(),
        manipulation_planner=DryRunManipulationPlanner(),
        arm_executor=DryRunArmExecutor(gripper),
        gripper=gripper,
        verifier=DryRunEpisodeVerifier(),
        recorder=JsonlEpisodeRecorder(episode_dir),
    )
