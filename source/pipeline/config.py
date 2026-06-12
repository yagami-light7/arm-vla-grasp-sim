"""Configuration for the full-physics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StateLimits:
    """Maximum ticks allowed in each type of pipeline phase."""

    build_stage: int = 5
    reset_episode: int = 20
    planning: int = 20
    navigation: int = 5000
    manipulation: int = 3000
    verification: int = 20
    export: int = 20
    cleanup: int = 20
    episode: int = 15000


@dataclass(frozen=True)
class FullPhysicsConfig:
    """Runtime configuration kept intentionally smaller than legacy CLIs."""

    task_json: Path
    output_dir: Path
    num_episodes: int = 1
    seed: int = 0
    headless: bool = True
    enable_debug_vis: bool = False
    save_video: bool = False
    dry_run: bool = False
    limits: StateLimits = field(default_factory=StateLimits)

    def __post_init__(self) -> None:
        if self.num_episodes < 1:
            raise ValueError("num_episodes must be at least 1")
        if self.limits.episode < 1:
            raise ValueError("episode tick limit must be positive")

    @property
    def render(self) -> bool:
        return not self.headless

    def episode_seed(self, episode_index: int) -> int:
        return self.seed + episode_index
