"""State definitions for one full nav-pick-place episode."""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    BUILD_STAGE = "build_stage"
    RESET_EPISODE = "reset_episode"
    PLAN_NAV_TO_PICK = "plan_nav_to_pick"
    EXEC_NAV_TO_PICK = "exec_nav_to_pick"
    VERIFY_PICK_REACHABLE = "verify_pick_reachable"
    PLAN_PICK = "plan_pick"
    EXEC_PICK = "exec_pick"
    VERIFY_PICK_SUCCESS = "verify_pick_success"
    PLAN_NAV_TO_PLACE = "plan_nav_to_place"
    EXEC_NAV_TO_PLACE = "exec_nav_to_place"
    VERIFY_PLACE_REACHABLE = "verify_place_reachable"
    PLAN_PLACE = "plan_place"
    EXEC_PLACE = "exec_place"
    VERIFY_PLACE_SUCCESS = "verify_place_success"
    EXPORT_LEROBOT = "export_lerobot"
    CLEANUP_EPISODE = "cleanup_episode"
    DONE = "done"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {PipelineState.DONE, PipelineState.FAILED}
