"""Task-bound scene runtime settings must preserve legacy profiles safely."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from source.simulation.isaac_runtime import IsaacSimulationRuntime
from source.simulation.object_initialization import (
    evaluate_object_initialization_pose,
    resolve_object_initialization_policy,
)
from source.simulation.scene_runtime import resolve_scene_runtime_settings
from source.simulation.task_scene_pose import (
    resolve_task_pick_support_pose,
    resolve_task_receptacle_pose,
)
from source.simulation.receptacle_support import (
    _validate_task_support_proxy,
    inspect_task_receptacle_support_stage,
    resolve_task_receptacle_support_settings,
)
from source.recording.training_action import (
    physical_execution_success_verified,
    task_requires_mesh_truth_manipulation_targets,
    task_requires_wrist_camera_object_clearance,
    training_mesh_truth_manipulation_targets_verified,
    training_quality_success_verified,
    training_receptacle_support_verified,
    training_wrist_camera_object_clearance_verified,
)
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_box_task_requires_supported_upright_object_initialization() -> None:
    task = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
    ).raw_task

    policy = resolve_object_initialization_policy(task)

    assert policy == {
        "enabled": True,
        "mode": "supported_upright_v1",
        "restore_pose_after_runtime_reset": True,
        "stabilize_xy_and_orientation_during_settle": True,
        "required_for_episode": True,
        "max_horizontal_displacement_m": 0.02,
        "max_vertical_displacement_m": 0.02,
        "max_orientation_error_rad": 0.1,
        "dynamic_settle_steps_before_sleep": 8,
        "source": "task.object_initialization",
    }


def test_seed7_historical_toppled_cola_is_rejected_by_initialization_gate() -> None:
    policy = resolve_object_initialization_policy(
        {
            "object_initialization": {
                "enabled": True,
                "mode": "supported_upright_v1",
                "max_horizontal_displacement_m": 0.02,
                "max_vertical_displacement_m": 0.02,
                "max_orientation_error_rad": 0.1,
            }
        }
    )

    report = evaluate_object_initialization_pose(
        policy=policy,
        requested_position_xyz=(
            -0.5937452709771781,
            6.458868795734489,
            0.1978932363673036,
        ),
        requested_quaternion_wxyz=(
            0.6917987507794513,
            0.6917987507794512,
            -0.14633690040448005,
            -0.1463369004044801,
        ),
        actual_pose_xyz_wxyz=(
            -0.513323962688446,
            6.43435525894165,
            0.167711079120636,
            0.43538790941238403,
            0.4164320230484009,
            0.22225421667099,
            -0.7665668725967407,
        ),
    )

    assert report["verified"] is False
    assert report["orientation_error_deg"] == pytest.approx(96.02923481327791)
    assert report["horizontal_displacement_m"] == pytest.approx(
        0.0840743736995116
    )
    assert set(report["violations"]) == {
        "horizontal_displacement",
        "vertical_displacement",
        "orientation_error",
    }


def test_normal_box_cola_settle_pose_passes_initialization_gate() -> None:
    task = JsonTaskProvider().load(
        PROJECT_ROOT / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
    ).raw_task
    policy = resolve_object_initialization_policy(task)

    report = evaluate_object_initialization_pose(
        policy=policy,
        requested_position_xyz=(
            -0.5014635713088045,
            6.657266206183525,
            0.1978932363673036,
        ),
        requested_quaternion_wxyz=(
            0.6829847067909134,
            0.6829847067909136,
            0.18311714908694363,
            0.18311714908694363,
        ),
        actual_pose_xyz_wxyz=(
            -0.50146484375,
            6.6566901206970215,
            0.19323338568210602,
            0.6798391938209534,
            0.6864492297172546,
            0.18179567158222198,
            0.18318438529968262,
        ),
    )

    assert report["verified"] is True
    assert report["violations"] == []


def test_scene_runtime_uses_legacy_defaults_when_task_has_no_override() -> None:
    settings = resolve_scene_runtime_settings(
        {},
        default_collision_prim_path="/World/scene_collision",
        default_visual_prim_path="/World/gauss",
        default_collision_floor_proxy_profile="yinluyuan_f2",
    )

    assert settings == {
        "collision_prim_path": "/World/scene_collision",
        "visual_prim_path": "/World/gauss",
        "collision_floor_proxy_profile": "yinluyuan_f2",
        "source": "runtime_defaults",
        "task_override_present": False,
    }


def test_liangzhu_scene_runtime_overrides_prims_and_disables_old_floor_proxy() -> None:
    settings = resolve_scene_runtime_settings(
        {
            "scene_runtime": {
                "collision_prim_path": (
                    "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
                ),
                "visual_prim_path": "/World/VisualScene/GaussianScene",
                "collision_floor_proxy_profile": None,
            }
        },
        default_collision_prim_path="/World/scene_collision",
        default_visual_prim_path="/World/gauss",
        default_collision_floor_proxy_profile="yinluyuan_f2",
    )

    assert settings["collision_prim_path"] == (
        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
    )
    assert settings["visual_prim_path"] == "/World/VisualScene/GaussianScene"
    assert settings["collision_floor_proxy_profile"] is None
    assert settings["source"] == "task.scene_runtime"
    assert settings["task_override_present"] is True


@pytest.mark.parametrize(
    "field,value",
    (
        ("collision_prim_path", "World/collision"),
        ("visual_prim_path", "VisualScene/GaussianScene"),
    ),
)
def test_scene_runtime_rejects_relative_prim_paths(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="绝对 USD prim path"):
        resolve_scene_runtime_settings(
            {"scene_runtime": {field: value}},
            default_collision_prim_path="/World/scene_collision",
            default_visual_prim_path="/World/gauss",
            default_collision_floor_proxy_profile=None,
        )


def test_simulation_smoke_does_not_clear_asset_units_resolve_xform() -> None:
    """轻量 smoke 重置不能删除可乐资产的 Rotate X=90° 模型转换。"""

    implementation = inspect.getsource(
        IsaacSimulationRuntime._apply_initial_object_pose
    )

    assert "ClearXformOpOrder" not in implementation
    assert "RemoveProperty" not in implementation
    assert "unitsResolve" in implementation


def test_simulation_smoke_inspects_task_receptacle_before_world_creation() -> None:
    """轻量 scene smoke 也必须在 PhysX 初始化前拒绝无碰撞垫子。"""

    inspect_source = inspect.getsource(IsaacSimulationRuntime._inspect_stage)
    build_source = inspect.getsource(IsaacSimulationRuntime.build)

    assert "inspect_task_receptacle_support_stage" in inspect_source
    assert '"task_receptacle_support_runtime_stage_report"' in build_source
    assert build_source.index("stage_report = self._inspect_stage") < build_source.index(
        "world.initialize_physics()"
    )


def test_vla_training_gate_requires_synchronized_rgb_after_physical_success() -> None:
    summary = {
        "success": True,
        "failure_reason": None,
        "success_semantics": "strict_physical_execution",
        "execution_provenance_verified": True,
        "task_config": {"training_action": {"enabled": True}},
        "training_visual_source_verified": False,
    }

    assert physical_execution_success_verified(summary) is True
    assert training_quality_success_verified(summary) is False

    summary["training_visual_source_verified"] = True
    assert training_quality_success_verified(summary) is True


def test_vla_training_gate_rejects_wrist_camera_object_near_plane_intersection() -> None:
    clearance_config = {
        "enabled": True,
        "required_for_training": True,
        "shape": "cylinder_local_z",
    }
    summary = {
        "success": True,
        "failure_reason": None,
        "success_semantics": "strict_physical_execution",
        "execution_provenance_verified": True,
        "task_config": {
            "training_action": {"enabled": True},
            "recording": {"wrist_camera_object_clearance": clearance_config},
        },
        "training_visual_source_verified": True,
        "simulation_report": {
            "wrist_camera_object_clearance_report": {
                **clearance_config,
                "camera_extrinsics_source": (
                    "hand_eye_calibration_with_visual_alignment_v3"
                ),
                "considered_sample_count": 12,
                "violation_count": 1,
                "verified": False,
            }
        },
    }

    assert task_requires_wrist_camera_object_clearance(summary["task_config"])
    assert training_wrist_camera_object_clearance_verified(summary) is False
    assert training_quality_success_verified(summary) is False

    report = summary["simulation_report"]["wrist_camera_object_clearance_report"]
    report["violation_count"] = 0
    report["verified"] = True
    assert training_wrist_camera_object_clearance_verified(summary) is True
    assert training_quality_success_verified(summary) is True


def test_rejected_wrist_visual_alignment_v2_is_never_training_eligible() -> None:
    summary = {
        "task_config": {},
        "simulation_report": {
            "wrist_camera_report": {
                "source": "hand_eye_calibration_with_visual_alignment_v2"
            }
        },
    }

    assert training_wrist_camera_object_clearance_verified(summary) is False


def test_liangzhu_task_receptacle_pose_is_resolved_before_physics() -> None:
    """地垫 episode 位姿必须通过统一解析器进入 runtime。"""

    task = json.loads(
        (PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json").read_text(
            encoding="utf-8"
        )
    )

    settings = resolve_task_receptacle_pose(task)

    assert settings["configured"] is True
    assert settings["prim_path"] == "/World/carpet"
    assert settings["pose_world"]["z"] == -0.1509331168865564
    build_source = inspect.getsource(IsaacSimulationRuntime.build)
    assert build_source.index("apply_task_receptacle_pose") < build_source.index(
        "world.initialize_physics()"
    )


def test_box_pair_support_poses_preserve_authored_orientation_and_scale() -> None:
    """双箱 episode 只能覆盖 translate，且 box2 在物理启动前补齐静态碰撞。"""

    task = JsonTaskProvider().load(
        PROJECT_ROOT
        / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
    ).raw_task
    pick = resolve_task_pick_support_pose(task)
    place = resolve_task_receptacle_pose(task)

    assert pick["prim_path"] == "/World/box1"
    assert place["prim_path"] == "/World/box2"
    for settings, collision_path in (
        (pick, "/World/box1/node_0"),
        (place, "/World/box2/node_0"),
    ):
        assert settings["translation_only"] is True
        assert settings["ensure_static_mesh_collision"] is True
        assert settings["collision_prim_path"] == collision_path
        assert settings["expected_support_bbox_dims_xyz"] is not None


def test_task_receptacle_pose_rejects_relative_prim_path() -> None:
    """任务场景姿态不允许把相对路径写入错误 prim。"""

    with pytest.raises(ValueError, match="绝对 USD prim path"):
        resolve_task_receptacle_pose(
            {
                "place": {
                    "receptacle_pose_world": {
                        "prim_path": "World/carpet",
                    }
                }
            }
        )


def test_liangzhu_task_requires_static_runtime_receptacle_support() -> None:
    task = json.loads(
        (PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json").read_text(
            encoding="utf-8"
        )
    )

    settings = resolve_task_receptacle_support_settings(task)

    assert settings["configured"] is True
    assert settings["runtime_validation_required"] is True
    assert settings["target_receptacle_prim_path"] == "/World/carpet"
    assert settings["target_support_prim_path"] == "/World/carpet/material"
    assert settings["support_expected_static"] is True


def test_receptacle_support_settings_reject_support_outside_receptacle() -> None:
    with pytest.raises(ValueError, match="必须位于 target receptacle 下"):
        resolve_task_receptacle_support_settings(
            {
                "place": {
                    "enabled": True,
                    "target_receptacle_prim_path": "/World/carpet",
                    "target_support_prim_path": "/World/other/material",
                    "support_runtime_validation_required": True,
                }
            }
        )


def test_training_gate_requires_runtime_receptacle_collision_report() -> None:
    summary = {
        "success": True,
        "failure_reason": None,
        "success_semantics": "strict_physical_execution",
        "execution_provenance_verified": True,
        "training_visual_source_verified": True,
        "task_config": {
            "place": {
                "enabled": True,
                "support_runtime_validation_required": True,
                "support_expected_static": True,
                "placement_region": {"frame": "world"},
                "curobo_world_collision": {"required": True},
            }
        },
        "simulation_report": {},
    }

    assert training_receptacle_support_verified(summary) is False
    assert training_quality_success_verified(summary) is False

    summary["simulation_report"][
        "task_receptacle_support_runtime_stage_report"
    ] = {
        "configured": True,
        "geometry_verified": True,
        "mesh_count": 1,
        "collision_enabled_count": 1,
        "static_support_verified": True,
        "placement_region_report": {"verified": True},
        "task_support_proxy_report": {"verified": True},
    }
    assert training_receptacle_support_verified(summary) is True
    assert training_quality_success_verified(summary) is True

    summary["simulation_report"][
        "task_receptacle_support_runtime_stage_report"
    ]["mesh_count"] = None
    assert training_receptacle_support_verified(summary) is False
    assert training_quality_success_verified(summary) is False


def test_training_gate_requires_runtime_mesh_truth_pick_and_place_exports() -> None:
    task = json.loads(
        (PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json").read_text(
            encoding="utf-8"
        )
    )
    assert task_requires_mesh_truth_manipulation_targets(task) is True
    derived_pose = task["place"]["place_pose_world"]
    summary = {
        "success": True,
        "failure_reason": None,
        "success_semantics": "strict_physical_execution",
        "execution_provenance_verified": True,
        "training_visual_source_verified": True,
        "task_config": task,
        "simulation_report": {
            "task_receptacle_support_runtime_stage_report": {
                "configured": True,
                "geometry_verified": True,
                "mesh_count": 1,
                "collision_enabled_count": 1,
                "static_support_verified": True,
                "placement_region_report": {"verified": True},
                "task_support_proxy_report": {"verified": True},
            }
        },
    }

    assert training_mesh_truth_manipulation_targets_verified(summary) is False
    assert training_quality_success_verified(summary) is False

    summary["simulation_report"]["last_current_state_curobo_pick_export"] = {
        "mesh_truth_pick_target_report": {
            "required": True,
            "verified": True,
            "visual_localization_used": False,
            "pick_tcp_source": "runtime_live_object_bbox",
            "resolved_grasp_mode": "top_down",
            "target_source_type": "sim_object_bbox_top_down",
            "bbox_center_source": "live_physx_object_pose",
        }
    }
    summary["simulation_report"]["last_current_state_curobo_place_export"] = {
        "desired_final_object_center_world": [
            derived_pose["x"],
            derived_pose["y"],
            derived_pose["z"],
        ],
        "mesh_truth_place_target_report": {
            "required": True,
            "verified": True,
            "visual_localization_used": False,
            "xyz_source": "runtime_mesh_truth",
            "support_geometry_verified": True,
            "object_extent_consistency_verified": True,
            "configured_pose_consistency_verified": True,
            "current_object_center_live_verified": True,
            "place_tcp_source": (
                "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
            ),
            "current_tcp_offset_source": (
                "runtime_current_tcp_and_live_object_center"
            ),
            "derived_place_pose_world": derived_pose,
            "configured_pose_consistency_tolerance_m": 1.0e-6,
        },
    }

    assert training_mesh_truth_manipulation_targets_verified(summary) is True
    assert training_quality_success_verified(summary) is True

    summary["simulation_report"]["last_current_state_curobo_place_export"][
        "mesh_truth_place_target_report"
    ]["visual_localization_used"] = True
    assert training_mesh_truth_manipulation_targets_verified(summary) is False
    assert training_quality_success_verified(summary) is False


def test_receptacle_support_gate_rejects_stale_curobo_bbox_proxy() -> None:
    task = json.loads(
        (PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json").read_text(
            encoding="utf-8"
        )
    )
    source = task["place"]["curobo_world_collision"]["cuboids_world"][0][
        "source"
    ]
    bbox_min = tuple(source["world_bbox_min_xyz"])
    bbox_max = tuple(source["world_bbox_max_xyz"])

    report = _validate_task_support_proxy(
        task,
        support_path="/World/carpet/material",
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        tolerance_m=1.0e-6,
    )
    assert report["verified"] is True
    assert report["max_abs_error_m"] == 0.0

    stale_task = deepcopy(task)
    stale_task["place"]["curobo_world_collision"]["cuboids_world"][0][
        "source"
    ]["world_bbox_max_xyz"][0] += 0.01
    with pytest.raises(RuntimeError, match="geometry drifted"):
        _validate_task_support_proxy(
            stale_task,
            support_path="/World/carpet/material",
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            tolerance_m=1.0e-6,
        )


def test_receptacle_support_stage_default_tolerance_covers_usd_roundoff() -> None:
    """运行时默认容差应覆盖数微米 USD bbox 舍入，同时远小于真实几何漂移。"""

    default_tolerance = inspect.signature(
        inspect_task_receptacle_support_stage
    ).parameters["tolerance_m"].default
    assert default_tolerance == 5.0e-6
    assert 2.7983789412378e-6 <= default_tolerance < 1.0e-4
