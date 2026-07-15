from __future__ import annotations

from pathlib import Path

import pytest

from source.pipeline import (
    FullPhysicsConfig,
    LocomotionPolicySettings,
    NavigationSettings,
    PCT_MULTIFLOOR_LOCOMOTION_TASK,
)
from source.pipeline.navigation_smoke import _build_dwa_config
from scripts.pipeline.run_full_physics_pipeline import _navigation_visual_runtime_kwargs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_apple_contact.json"


def test_pct_multifloor_profile_requires_locomotion_task(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake checkpoint")

    with pytest.raises(ValueError, match="locomotion_task"):
        FullPhysicsConfig(
            task_json=TASK_PATH,
            output_dir=tmp_path,
            dry_run=True,
            locomotion=LocomotionPolicySettings(
                policy_profile="pct_multifloor",
                locomotion_checkpoint=checkpoint,
            ),
        )


def test_pct_multifloor_profile_accepts_dog_only_task(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fake checkpoint")

    config = FullPhysicsConfig(
        task_json=TASK_PATH,
        output_dir=tmp_path,
        dry_run=True,
        locomotion=LocomotionPolicySettings(
            policy_profile="pct_multifloor",
            locomotion_task=PCT_MULTIFLOOR_LOCOMOTION_TASK,
            locomotion_checkpoint=checkpoint,
        ),
    )

    assert config.locomotion.locomotion_task == PCT_MULTIFLOOR_LOCOMOTION_TASK


def test_pct_robot_config_isolated_from_default_single_floor_asset() -> None:
    asset_source = (
        PROJECT_ROOT / "source/robot_lab/robot_lab/assets/go2_x5.py"
    ).read_text(encoding="utf-8")
    task_source = (
        PROJECT_ROOT
        / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity"
        / "config/quadruped/go2_x5/train_route_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "GO2_X5_PCT_DOG_ONLY_CFG = GO2_X5_CFG.replace(" in asset_source
    assert "GO2_X5_PCT_USD" not in asset_source
    assert "spawn=sim_utils.UsdFileCfg" not in asset_source
    pct_asset_source = asset_source.split(
        "GO2_X5_PCT_DOG_ONLY_CFG = GO2_X5_CFG.replace(", 1
    )[1]
    pct_arm_source = pct_asset_source.split("# 主 pipeline 后续仍要抓取", 1)[0]
    assert '"arm": ImplicitActuatorCfg(' in pct_arm_source
    assert '"arm": DCMotorCfg(' not in pct_arm_source
    assert 'joint_names_expr=["arm_joint[1-6]"]' in pct_arm_source
    assert "effort_limit_sim=100.0" in pct_arm_source
    assert "velocity_limit_sim=10.0" in pct_arm_source
    assert "stiffness=1000.0" in pct_arm_source
    assert "damping=50.0" in pct_arm_source
    assert "self.scene.robot = GO2_X5_PCT_DOG_ONLY_CFG.replace(" in task_source
    assert "self.scene.height_scanner.offset.pos = (0.0, 0.0, 1.0)" in task_source
    assert "self.scene.height_scanner_base.offset.pos = (0.0, 0.0, 1.0)" in task_source
    assert "class _Go2X5LeggedBaseEnvCfg" in task_source
    assert 'self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")' in task_source


def test_pct_multifloor_dwa_commands_stay_inside_checkpoint_training_range() -> None:
    nav = NavigationSettings(brisk_nav=True)

    assert nav.pct_vertical_obstacle_min_slices == 0
    assert nav.pct_global_vertical_obstacle_min_slices == 7
    assert nav.pct_cross_floor_vertical_obstacle_min_slices == 9
    assert nav.pct_cross_floor_gateway_points == ((1.5, 5.7, 0.6),)
    assert nav.pct_cross_floor_stair_exit_points == ((2.90, 7.05, 3.0),)
    assert nav.pct_cross_floor_stair_midpoint_points == (
        (1.51822, 6.27683, 0.29486),
        (2.94512, 9.14634, 1.64666),
        (1.9202, 9.52807, 1.71919),
        (2.89841, 7.79872, 2.61031),
    )
    assert nav.pct_cross_floor_gateway_radius_m == 0.6
    assert nav.pct_robot_root_to_floor_m == 0.45
    assert nav.pct_body_obstacle_min_height_m == 0.30
    assert nav.pct_body_obstacle_max_height_m == 1.0
    assert nav.pct_stair_vertical_radius_m == pytest.approx(0.60)
    assert nav.pct_stair_progress_tolerance == pytest.approx(0.35)
    assert nav.pct_stair_progress_cost_weight == pytest.approx(20.0)
    assert nav.pct_multifloor_vertical_obstacle_min_slices == 5
    assert nav.pct_multifloor_obstacle_inflate_radius == pytest.approx(0.12)
    assert nav.pct_multifloor_route_corridor_radius == pytest.approx(0.45)
    assert nav.pct_carry_max_linear_velocity == pytest.approx(0.25)
    assert nav.pct_carry_max_angular_velocity == pytest.approx(0.30)
    assert nav.pct_carry_max_linear_accel == pytest.approx(1.00)
    assert nav.pct_carry_initial_alignment_path_deviation_limit == pytest.approx(0.40)
    assert nav.pct_carry_path_recovery_deviation_limit == pytest.approx(0.50)
    assert nav.pct_carry_max_infeasible_recomputes == 8
    assert nav.pct_stair_float_enabled is False
    assert nav.pct_stair_float_speed_mps == pytest.approx(0.18)
    assert nav.pct_stair_float_activation_radius_m == pytest.approx(0.45)
    assert nav.pct_stair_float_completion_radius_m == pytest.approx(0.25)
    assert nav.pct_stair_float_min_z_delta_m == pytest.approx(0.75)
    assert nav.pct_stair_float_approach_distance_m == pytest.approx(6.00)
    assert nav.pct_stair_float_exit_distance_m == pytest.approx(1.40)
    assert nav.pct_stair_float_settle_time_s == pytest.approx(1.20)
    assert nav.pct_stair_float_yaw_lookahead_m == pytest.approx(0.35)
    assert nav.pct_stair_float_min_root_z_offset_m == pytest.approx(0.18)
    assert nav.pct_stair_float_release_root_z_offset_m == pytest.approx(0.36)
    flat = _build_dwa_config(nav, policy_profile="flat")
    pct = _build_dwa_config(nav, policy_profile="pct_multifloor")

    assert flat.max_linear_velocity == pytest.approx(0.80)
    assert flat.max_angular_velocity == pytest.approx(1.00)
    assert pct.max_linear_velocity == pytest.approx(0.45)
    assert pct.max_angular_velocity == pytest.approx(0.50)
    assert pct.max_linear_accel == pytest.approx(2.50)
    assert pct.min_active_linear_velocity == pytest.approx(0.25)
    assert pct.speed_bias == pytest.approx(0.90)
    assert pct.lookahead_distance == pytest.approx(0.12)
    assert pct.waypoint_tolerance == pytest.approx(0.05)
    assert pct.prediction_horizon == pytest.approx(0.35)
    assert pct.integration_dt == pytest.approx(0.05)
    assert pct.obstacle_distance_cap == pytest.approx(1.00)
    assert pct.clearance_bias == pytest.approx(0.55)
    assert pct.path_deviation_limit == pytest.approx(0.30)
    assert pct.close_goal_speed_limit == pytest.approx(0.30)
    assert pct.near_goal_min_active_linear_velocity == pytest.approx(0.30)
    assert pct.enforce_min_active_linear_velocity is True
    assert pct.enforce_min_active_angular_velocity is True
    assert pct.min_active_angular_velocity == pytest.approx(0.30)
    assert pct.close_goal_speed_limit >= pct.min_active_linear_velocity
    assert pct.min_active_linear_velocity <= pct.max_linear_velocity


def test_navigation_visual_auto_uses_collision_for_pct_only() -> None:
    pct = _navigation_visual_runtime_kwargs("pct_multifloor", "auto")
    flat = _navigation_visual_runtime_kwargs("flat", "auto")

    assert pct == {
        "enable_scene_visual": False,
        "hide_navigation_collision_visual": False,
        "hide_object_collision_visual": False,
    }
    assert flat == {
        "enable_scene_visual": True,
        "hide_navigation_collision_visual": True,
        "hide_object_collision_visual": True,
    }


def test_navigation_visual_auto_uses_full_scene_for_recorded_rgb() -> None:
    recorded = _navigation_visual_runtime_kwargs(
        "pct_multifloor",
        "auto",
        recording_visual_required=True,
    )

    assert recorded == {
        "enable_scene_visual": True,
        "hide_navigation_collision_visual": True,
        "hide_object_collision_visual": True,
    }
