"""校验 Liangzhu Phase-0 固定任务和运行时 Mesh-truth 操作目标。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from source.manipulation.current_state_curobo import pose_to_matrix
from source.navigation.pct_adapter import pct_to_sim_xyz, sim_to_pct_xyz
from source.pipeline.factory import _requires_extended_pct_navigation_limits
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json"
TARGET_PATH = PROJECT_ROOT / "tasks/liangzhu_mat_placement_target_legacy.json"
ASSET_MANIFEST_PATH = (
    PROJECT_ROOT / "source/scene/liangzhu/runtime_asset_manifest.json"
)


def test_liangzhu_fixed_task_loads_through_existing_episode_spec() -> None:
    """新任务必须由 JsonTaskProvider 加载，不能绕过现有 NavPickTask schema。"""

    episode = JsonTaskProvider().load(TASK_PATH)

    assert episode.task_id == 2001
    assert episode.object_prim_path == "/World/cola"
    assert episode.scene_usd == "source/scene/liangzhu/liangzhu.usda"
    assert episode.nav_map == ""
    assert episode.place_goal is not None
    assert episode.place_target_pose == (
        -0.4375161874288581,
        5.111811405946711,
        -0.09712594670872055,
        0.0,
        0.0,
        0.0,
    )

    config = SimpleNamespace(
        locomotion=SimpleNamespace(policy_profile="pct_multifloor")
    )
    assert _requires_extended_pct_navigation_limits(config, episode) is False
    assert episode.instruction == (
        "Pick up the coke can on the floor in front of the robot and place it "
        "on the designated mouse mat on the floor."
    )
    assert episode.raw_task["target_receptacle_id"] == "mat_carpet_01"
    assert episode.raw_task["target_receptacle_aliases"] == [
        "mouse_mat",
        "mouse_pad",
        "mat",
        "carpet",
    ]
    assert episode.raw_task["target_receptacle_prim_path"] == "/World/carpet"
    assert episode.raw_task["global_planner"] == "pct"
    assert episode.raw_task["policy_profile"] == "pct_multifloor"
    assert episode.raw_task["perception_mode"] == "sim_ground_truth"
    assert episode.raw_task["randomization"]["forward_sector"][
        "robot_yaw_range_deg"
    ] == [-180.0, 180.0]
    assert (
        episode.raw_task["place"]["mesh_truth_target"]["object_extent_tolerance_m"]
        == 0.01
    )
    assert episode.raw_task["navigation_execution"] == {
        "final_position_tolerance": 0.1,
        "place_position_tolerance": 0.15,
        "final_yaw_tolerance": 0.15,
        "stable_linear_velocity": 0.06,
        "stable_angular_velocity": 0.2,
        "require_yaw_alignment": True,
        "require_stable_base": True,
    }
    assert episode.raw_task["manipulation_execution"] == {
        "reuse_pick_grasp_orientation_for_place": True,
    }
    assert episode.raw_task["mesh_truth_manipulation_targets"] == {
        "required": True,
        "visual_localization_required": False,
        "pick_tcp_source": "runtime_live_object_bbox",
        "place_tcp_source": (
            "runtime_receptacle_bbox_plus_pick_object_bbox_plus_current_tcp_offset"
        ),
    }
    assert episode.raw_task["scene_runtime"] == {
        "collision_prim_path": (
            "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
        ),
        "visual_prim_path": "/World/VisualScene/GaussianScene",
        "collision_floor_proxy_profile": None,
    }
    assert episode.raw_task["training_action"] == {
        "enabled": True,
        "schema": "base_xyyaw_tcp_base_rpy_gripper_v1",
        "dimension": 10,
        "base_pose_frame": "world",
        "tcp_pose_frame": "base_frame",
        "tcp_euler_order": "roll_pitch_yaw",
        "angle_unit": "rad",
        "position_unit": "m",
        "gripper_range": [0.0, 1.0],
        "gripper_closed_value": 0.0,
        "gripper_open_value": 1.0,
        "source_gripper_joint_range_m": [0.0, 0.04],
        "action_alignment": "next_sample_executed_pose",
        "terminal_action": "hold_current_pose",
    }
    assert episode.raw_task["subtask_segmentation"] == {
        "enabled": True,
        "schema": "nav_straight_turn_stop__arm_approach_contact_retreat_v1",
        "directory_export": True,
        "output_layout": "episodes_task_episode_subtask_front_wrist_v3",
        "min_segment_frames": 3,
        "hysteresis_frames": 2,
        "navigation": {
            "stop_command_linear_max_mps": 0.03,
            "stop_command_angular_max_rps": 0.08,
            "stop_measured_linear_max_mps": 0.08,
            "stop_measured_angular_max_rps": 0.20,
            "turn_command_angular_min_rps": 0.12,
            "turn_measured_angular_min_rps": 0.25,
            "turn_yaw_delta_min_rad": 0.03,
        },
        "contact_label_source": "heuristic_action_and_kinematics",
    }


def test_liangzhu_task_place_pose_matches_mat_geometry_target() -> None:
    """任务中的标准 place_pose_world 不能与垫子几何目标发生漂移。"""

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))

    assert task["place"]["place_pose_world"] == target["task_schema_fragment"][
        "place"
    ]["place_pose_world"]
    assert task["place"]["target_receptacle_id"] == target[
        "target_receptacle_id"
    ]


def test_liangzhu_phase0_pick_pose_is_runtime_mesh_truth_with_audited_spawn() -> None:
    """抓取使用运行时 bbox，固定 spawn 仍保持可追溯的地面支撑半高。"""

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    pick = task["pick"]
    support = pick["support_geometry"]

    assert pick["grasp_mode"] == "top_down"
    assert pick["pose_source"] == "runtime_live_object_bbox_top_down_grasp"
    assert task["notes"]["pick_pose_status"] == (
        "runtime_live_mesh_bbox_top_down_truth"
    )
    assert abs(
        pick["object_pose_world"]["z"] - support["collision_support_z"]
        - support["center_to_support_m"]
    ) < 1.0e-12


def test_liangzhu_fixed_pick_and_place_face_target_from_front() -> None:
    """固定 baseline 的可乐和垫子都必须位于底盘正前方。"""

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    targets = {
        "pick": task["pick"]["object_pose_world"],
        "place": task["place"]["place_pose_world"],
    }
    for phase, target in targets.items():
        goal = task[phase]["base_goal"]
        target_bearing = float(
            np.arctan2(target["y"] - goal["y"], target["x"] - goal["x"])
        )
        relative_bearing = float(
            np.arctan2(
                np.sin(target_bearing - goal["yaw"]),
                np.cos(target_bearing - goal["yaw"]),
            )
        )
        assert abs(relative_bearing) < 1.0e-12
        assert goal["target_region_in_base"] == "front"
        assert goal["final_alignment_mode"] == "face_target"


def test_liangzhu_phase0_coke_is_in_forward_sector_and_both_targets_are_grounded() -> None:
    """固定 baseline 必须位于前向扇区，并让可乐与垫子都受地面支撑。"""

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    preconditions = task["phase0_spatial_preconditions"]
    start = task["start"]
    object_pose = task["pick"]["object_pose_world"]
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    bearing = float(
        np.arctan2(
            object_pose["y"] - start["y"],
            object_pose["x"] - start["x"],
        )
    )
    heading_error = float(
        np.arctan2(
            np.sin(bearing - start["yaw"]),
            np.cos(bearing - start["yaw"]),
        )
    )

    forward = preconditions["object_in_front_of_robot"]
    assert abs(heading_error) <= forward["sector_half_angle_rad"]
    assert abs(heading_error - forward["heading_error_rad"]) < 1.0e-15
    assert [start[key] for key in ("x", "y", "z")] == [
        -1.4849319648011197,
        5.126136502764003,
        0.29281728532721385,
    ]
    assert preconditions["object_on_floor"]["offline_geometry_verified"] is True
    assert abs(
        object_pose["z"]
        - task["pick"]["support_geometry"]["collision_support_z"]
        - task["pick"]["support_geometry"]["center_to_support_m"]
    ) < 1.0e-12
    mat_ground = preconditions["receptacle_on_floor"]
    assert mat_ground["offline_geometry_verified"] is True
    assert abs(
        mat_ground["top_above_scene_floor_m"]
        - target["support_geometry"]["mat_top_above_scene_floor_m"]
    ) < 1.0e-12
    assert task["place"]["support_runtime_validation_required"] is True
    assert task["place"]["support_expected_static"] is True
    assert task["randomization"]["forward_sector"][
        "placement_region_half_extent_xy_m"
    ] == [0.04, 0.04]
    assert task["place"]["place_xy_tolerance"] == 0.04
    mesh_target = task["place"]["mesh_truth_target"]
    assert mesh_target["enabled"] is True
    assert mesh_target["visual_localization_required"] is False
    assert mesh_target["target_xy_source"] == "runtime_placement_region_center"
    assert mesh_target["support_surface_source"] == (
        "runtime_target_support_bbox_top"
    )
    assert mesh_target["object_support_extent_source"] == (
        "pick_live_object_bbox_center_to_min_z"
    )
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    assert abs(
        mesh_target["expected_object_bbox_center_to_min_z_m"]
        - target["target_object_mesh_geometry"]["bbox_center_to_min_z_m"]
    ) < 1.0e-15
    assert abs(
        task["pick"]["support_geometry"]["center_to_support_m"]
        - mesh_target["expected_object_bbox_center_to_min_z_m"]
    ) > 1.0e-5
    assert task["notes"]["runtime_mesh_truth_place_target"] is True
    assert task["notes"]["fixed_place_target"] is False


def test_liangzhu_curobo_support_proxies_match_pick_floor_and_mat() -> None:
    """pick 使用 PLY 地面，place 使用实际垫子碰撞包围盒。"""

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    for phase in ("pick", "place"):
        config = task[phase]["curobo_world_collision"]
        assert config["enabled"] is True
        assert config["required"] is True
        assert len(config["cuboids_world"]) == 1
        assert config["cuboids_world"][0]["padding_mode"] == "preserve_top"

    pick_proxy = task["pick"]["curobo_world_collision"]["cuboids_world"][0]
    pick_source = pick_proxy["source"]
    pick_transform = pose_to_matrix(
        pick_proxy["center_xyz"],
        pick_proxy["quaternion_wxyz"],
    )
    pick_top_center = (
        pick_transform[:3, 3]
        + pick_transform[:3, 2] * float(pick_proxy["dims_xyz"][2]) * 0.5
    )
    np.testing.assert_allclose(
        pick_top_center,
        pick_source["support_point_xyz"],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert pick_source["collision_ply_sha256"] == (
        target["support_geometry"]["scene_floor_collision_ply_sha256"]
    )
    assert pick_source["collision_face_index"] > 0

    place_proxy = task["place"]["curobo_world_collision"]["cuboids_world"][0]
    place_source = place_proxy["source"]
    place_transform = pose_to_matrix(
        place_proxy["center_xyz"],
        place_proxy["quaternion_wxyz"],
    )
    place_top_center = (
        place_transform[:3, 3]
        + place_transform[:3, 2] * float(place_proxy["dims_xyz"][2]) * 0.5
    )
    assert place_proxy["semantic_role"] == "mat_support"
    assert place_proxy["source_prim_path"] == "/World/carpet/material"
    np.testing.assert_allclose(
        place_top_center[:2],
        target["placement_region"]["center_xyz"][:2],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert abs(place_top_center[2] - place_source["support_surface_z"]) < 1.0e-12
    assert place_source["asset_sha256"] == target["support_geometry"][
        "receptacle_asset_sha256"
    ]


def test_liangzhu_runtime_manifest_uses_identity_pct_frame() -> None:
    """同坐标 collision PLY 必须显式使用 identity，不能继承旧多楼层取反。"""

    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    calibration = manifest["frame_calibration"]
    box2 = manifest["receptacles"]["box2_01"]
    point = tuple(box2["nominal_support_center_xy"]) + (
        box2["support_surface_z"],
    )

    pct_point = sim_to_pct_xyz(point, coord_mode="identity")
    restored = pct_to_sim_xyz(pct_point, coord_mode="identity")

    assert pct_point == point
    assert restored == point
    assert calibration["coord_mode"] == "identity"
    assert manifest["pct"]["coord_mode"] == "identity"
    assert manifest["pct"]["randomized_seed_sweep_count"] == 20
    assert manifest["pct"]["randomized_plan_request_count"] == 40
    assert manifest["pct"]["randomized_seed_sweep_all_ok"] is True
    assert manifest["pct"]["dynamic_box_keepouts_local_map_verified"] is True
    assert manifest["randomization"]["robot_yaw_range_deg"] == [-180.0, 180.0]
    assert manifest["randomization"]["placement_region_half_extent_xy_m"] == [
        0.1,
        0.05,
    ]
    assert manifest["randomization"]["box1_xy_only"] is True
    assert manifest["randomization"]["box2_xy_only"] is True
    assert manifest["randomization"]["box_orientation_z_scale_preserved"] is True
    assert calibration["box_orientation_policy"] == "preserve_authored_xform_ops"
    assert calibration["box_position_policy"] == "root_translate_xy_only"
    assert calibration["runtime_path_overlay_verified"] is False


def test_liangzhu_runtime_manifest_records_real_box_pair_validation() -> None:
    """真实 seed=5000 已覆盖场景、导航、操作和训练数据门禁。"""

    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))

    scene_runtime = manifest["scene_runtime"]
    assert scene_runtime["collision_prim_path"] == (
        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
    )
    assert scene_runtime["visual_prim_path"] == "/World/VisualScene/GaussianScene"
    assert scene_runtime["collision_floor_proxy_profile"] is None
    assert scene_runtime["legacy_yinluyuan_f2_floor_proxy_disabled"] is True
    assert scene_runtime["default_navigation_visual_mode"] == "collision"
    assert scene_runtime["overview_camera_prim_path"] == "/World/overview"
    assert scene_runtime["box_pair_openusd_composition_verified"] is True
    assert scene_runtime["full_physics_runtime_verified"] is True

    assert manifest["scene_collision"]["runtime_load_verified"] is True
    assert manifest["scene_collision"]["payload_mesh_marker_verified"] is True
    assert manifest["visual_scene"]["loaded_by_default"] is False
    assert manifest["visual_scene"]["explicit_enable_mode"] == (
        "--navigation-visual-mode full"
    )
    assert manifest["visual_scene"]["runtime_load_verified"] is False
    assert manifest["robot"]["asset_exists"] is True
    assert manifest["robot"]["runtime_articulation_verified"] is True
    assert manifest["robot"]["checkpoint_runtime_load_verified"] is True
    cola = manifest["objects"]["cola_01"]
    assert cola["prim_path"] == "/World/cola"
    assert cola["stage_visual_mesh_count"] == 1
    assert cola["stage_collision_api_count"] == 1
    assert cola["stage_rigid_body_api_count"] == 1
    assert cola["runtime_rigid_body_verified"] is True
    assert manifest["pct"]["runtime_navigation_verified"] is True
    mesh_targets = manifest["manipulation_targets"]
    assert mesh_targets["mode"] == "sim_mesh_truth"
    assert mesh_targets["visual_localization_required"] is False
    assert mesh_targets["pick_grasp_mode"] == "top_down"
    assert mesh_targets["place_reuses_pick_top_down_orientation"] is True
    assert mesh_targets["mesh_truth_derivation_unit_verified"] is True
    assert mesh_targets["runtime_pick_export_verified"] is True
    assert mesh_targets["runtime_place_export_verified"] is True
    collision = manifest["collision_relationship"]
    proxies = collision["task_local_curobo_support_proxies"]
    assert collision["active_pick_support"] == "box1_01"
    assert collision["active_target_receptacle"] == "box2_01"
    assert proxies["openusd_bbox_verified"] is True
    assert proxies["runtime_export_verified"] is True
    assert proxies["curobo_plan_avoidance_verified"] is True
    box1 = manifest["receptacles"]["box1_01"]
    box2 = manifest["receptacles"]["box2_01"]
    assert box1["translation_only_episode_override"] is True
    assert box1["physx_contact_verified"] is True
    assert box2["translation_only_episode_override"] is True
    assert box2["collision_runtime_policy"] == (
        "apply_static_mesh_collision_before_physics"
    )
    assert box2["physx_contact_verified"] is True
    assert box2["release_stability_verified"] is True
    randomization = manifest["randomization"]
    assert randomization["phase"] == 1
    assert randomization["mode"] == "liangzhu_box_pair_xy_v1"
    assert randomization["robot_spawn_policy"] == (
        "between_box_centers_with_lateral_offset"
    )
    assert randomization["cola_xy_policy"] == "box1_center_local_safe_region"
    assert randomization["cola_yaw_range_deg"] == [-180.0, 180.0]
    assert randomization["dynamic_navigation_keepouts"] == ["box1", "box2"]
    assert randomization["offline_seed_sweep_verified"] is True
    assert randomization["runtime_randomized_episode_verified"] is True
    assert manifest["data_export"]["training_action_dimension"] == 10
    assert manifest["data_export"]["control_action_dimension"] == 11
    assert manifest["data_export"]["schema_version"] == (
        "full_physics_lerobot_v2.2.0"
    )
    assert manifest["data_export"]["subtask_directory_layout"] == (
        "episodes_task_episode_subtask_front_wrist_v3"
    )
    assert manifest["data_export"]["subtask_directory_count_per_collected_episode"] == 6
    assert list(
        manifest["data_export"]["subtask_directory_index_mapping"].values()
    ) == [
        "nav_straight",
        "nav_turn",
        "nav_stop",
        "arm_approach",
        "arm_contact",
        "arm_retreat",
    ]
    assert manifest["data_export"]["instruction_annotation_schema"] == (
        "relative_direction_segment_instruction_v1"
    )
    assert manifest["data_export"]["instruction_language"] == "en"
    assert manifest["data_export"]["instruction_direction_labels"] == [
        "front",
        "front-left",
        "left",
        "back-left",
        "back",
        "back-right",
        "right",
        "front-right",
    ]
    assert manifest["data_export"]["per_frame_instruction_parquet_columns_added"] is True
    assert manifest["data_export"]["per_frame_instruction_subtask_csv_columns_added"] is True
    assert manifest["data_export"]["real_episode_export_verified"] is True
    assert (
        manifest["data_export"]["real_episode_training_eligible_verified"]
        is True
    )
    validation = manifest["real_validation"]
    assert validation["seed"] == 5000
    assert validation["success"] is True
    assert validation["final_state"] == "done"
    assert validation["lerobot_frames"] == 283
    assert validation["lerobot_validation_errors"] == 0
    assert validation["subtask_directory_count"] == 6
