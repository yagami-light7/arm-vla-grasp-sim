"""导航与 manipulation action apply 的组合验证器。"""

from __future__ import annotations

from source.interfaces import EpisodeSpec, SimulationState, VerificationResult

from .manipulation_apply import ManipulationApplySmokeVerifier
from .navigation import NavigationEpisodeVerifier


class IntegratedApplySmokeVerifier:
    """组合真实导航可达性与关节 action apply 验证，不声明物体抓放成功。"""

    def __init__(self, navigation: NavigationEpisodeVerifier):
        self.navigation = navigation
        self.manipulation = ManipulationApplySmokeVerifier()

    def verify_pick_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.navigation.verify_pick_reachable(state, episode_spec)

    def verify_pick_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.manipulation.verify_pick_success(state, episode_spec)

    def verify_place_reachable(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.navigation.verify_place_reachable(state, episode_spec)

    def verify_place_success(
        self,
        state: SimulationState,
        episode_spec: EpisodeSpec,
    ) -> VerificationResult:
        return self.manipulation.verify_place_success(state, episode_spec)
