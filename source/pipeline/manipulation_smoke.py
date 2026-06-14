"""分段 manipulation action 合同 smoke 组装器。"""

from __future__ import annotations

from pathlib import Path

from source.diagnostics import DryRunEpisodeVerifier
from source.interfaces import EpisodeSpec
from source.manipulation import (
    BinaryGripperController,
    SegmentedArmExecutor,
    SegmentedArmExecutorConfig,
    SegmentedSmokeManipulationPlanner,
)
from source.navigation.dry_run import DryRunNavExecutor, DryRunNavPlanner
from source.recording import JsonlEpisodeRecorder
from source.simulation import InMemorySimulationRuntime

from .config import FullPhysicsConfig
from .full_physics_pipeline import FullPhysicsPipeline


def create_manipulation_smoke_pipeline(
    *,
    config: FullPhysicsConfig,
    episode_spec: EpisodeSpec,
    episode_seed: int,
    episode_dir: str | Path,
) -> FullPhysicsPipeline:
    """创建跳过导航、只验证分段机械臂与夹爪 action 合同的 pipeline。"""

    gripper = BinaryGripperController()
    return FullPhysicsPipeline(
        config=config,
        episode_spec=episode_spec,
        episode_seed=episode_seed,
        simulation=InMemorySimulationRuntime(),
        # manipulation smoke 从 reset 后直接进入 PLAN_PICK，导航组件只作协议占位。
        nav_planner=DryRunNavPlanner(),
        nav_executor=DryRunNavExecutor(),
        manipulation_planner=SegmentedSmokeManipulationPlanner(),
        arm_executor=SegmentedArmExecutor(
            gripper,
            config=SegmentedArmExecutorConfig(
                sim_dt=0.05,
                gripper_move_duration=0.10,
                gripper_hold_duration=0.05,
            ),
        ),
        gripper=gripper,
        verifier=DryRunEpisodeVerifier(),
        recorder=JsonlEpisodeRecorder(episode_dir),
    )
