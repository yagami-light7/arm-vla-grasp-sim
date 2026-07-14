"""校验 Liangzhu Phase-0 固定任务和运行时 Mesh-truth 操作目标。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from source.manipulation.current_state_curobo import pose_to_matrix
from source.navigation.pct_adapter import pct_to_sim_xyz, sim_to_pct_xyz
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json"
TARGET_PATH = PROJECT_ROOT / "tasks/liangzhu_placement_target.json"
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
    assert episode.raw_task["navigation_execution"] == {
        "final_position_tolerance": 0.05,
        "place_position_tolerance": 0.12,
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
    mat_evidence = next(
        item
        for item in calibration["evidence"]
        if item["name"] == "mat_carpet_place_target"
    )
    point = tuple(mat_evidence["world_xy"]) + (
        mat_evidence["scene_floor_support_z"],
    )

    pct_point = sim_to_pct_xyz(point, coord_mode="identity")
    restored = pct_to_sim_xyz(pct_point, coord_mode="identity")

    assert pct_point == point
    assert restored == point
    assert manifest["pct"]["coord_mode"] == "identity"
    assert calibration["runtime_path_overlay_verified"] is False


def test_liangzhu_runtime_manifest_keeps_unverified_runtime_gates_false() -> None:
    """离线地图成功不能冒充 Isaac 场景、checkpoint 或真实导航已验收。"""

    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))

    scene_runtime = manifest["scene_runtime"]
    assert scene_runtime["collision_prim_path"] == (
        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
    )
    assert scene_runtime["visual_prim_path"] == "/World/VisualScene/GaussianScene"
    assert scene_runtime["collision_floor_proxy_profile"] is None
    assert scene_runtime["legacy_yinluyuan_f2_floor_proxy_disabled"] is True
    assert scene_runtime["default_navigation_visual_mode"] == "collision"
    assert scene_runtime["gaussian_scene_loaded_by_default"] is False
    assert scene_runtime["overview_camera_prim_path"] == "/World/overview"
    assert scene_runtime["configuration_unit_verified"] is True
    assert scene_runtime["preflight_text_marker_verified"] is True
    assert scene_runtime["runtime_resolution_verified"] is False

    assert manifest["scene_collision"]["runtime_load_verified"] is False
    assert manifest["scene_collision"]["payload_mesh_marker_verified"] is True
    assert manifest["visual_scene"]["runtime_load_verified"] is False
    assert manifest["visual_scene"]["package_members_verified"] == [
        "default.usda",
        "gauss.usda",
        "liangzhu.nurec",
    ]
    assert manifest["robot"]["asset_exists"] is True
    assert manifest["robot"]["runtime_articulation_verified"] is False
    cola = manifest["objects"]["cola_01"]
    assert cola["prim_path"] == "/World/cola"
    assert cola["asset_and_texture_exist"] is True
    assert cola["runtime_rigid_body_verified"] is False
    assert manifest["pct"]["offline_probe"]["runtime_navigation_verified"] is False
    assert manifest["locomotion"]["checkpoint_runtime_load_verified"] is False
    mesh_targets = manifest["manipulation_targets"]
    assert mesh_targets["mode"] == "sim_mesh_truth"
    assert mesh_targets["visual_localization_required"] is False
    assert mesh_targets["pick_grasp_mode"] == "top_down"
    assert mesh_targets["top_down_reverse_approach_lift"] is True
    assert mesh_targets["top_down_curobo_plan_probe_verified"] is True
    assert mesh_targets["place_reuses_pick_top_down_orientation"] is True
    assert mesh_targets["mesh_truth_derivation_unit_verified"] is True
    assert mesh_targets["runtime_pick_export_verified"] is False
    assert mesh_targets["runtime_place_export_verified"] is False
    collision = manifest["collision_relationship"]
    proxies = collision["task_local_curobo_support_proxies"]
    assert collision["physics_collision_layout"] == "single_merged_collision_prim"
    assert collision["semantic_mat_prim_verified"] is True
    assert proxies["configuration_unit_verified"] is True
    assert proxies["runtime_export_verified"] is False
    assert proxies["curobo_plan_avoidance_verified"] is False
    randomization = manifest["randomization"]
    assert randomization["phase"] == 1
    assert randomization["mode"] == "robot_forward_sector_v1"
    assert randomization["target_randomization_enabled"] is True
    assert randomization["cola_mat_initial_overlap_rejected"] is True
    assert randomization["dynamic_proxy_regeneration_implemented"] is True
    assert randomization["pick_base_standoff_range_m"] == [0.35, 0.39]
    assert randomization["place_base_standoff_range_m"] == [0.35, 0.39]
    assert randomization["target_region_in_base"] == "front"
    assert randomization["place_approach_origin"] == "pick_base_goal"
    assert randomization["offline_seed_sweep_verified"] is True
    assert randomization["runtime_randomized_episode_verified"] is False
    assert manifest["data_export"]["training_action_dimension"] == 10
    assert manifest["data_export"]["control_action_dimension"] == 11
    assert manifest["data_export"]["gaussian_scene_required_for_training"] is False
    assert (
        manifest["data_export"]["synchronized_rgb_camera_gate_unit_verified"]
        is True
    )
    assert manifest["data_export"]["collision_mode_rgb_training_eligible"] is True
    assert manifest["data_export"]["overview_camera_prim_path"] == "/World/overview"
    assert (
        manifest["data_export"]["receptacle_runtime_support_training_gate_added"]
        is True
    )
    assert (
        manifest["data_export"][
            "receptacle_runtime_support_training_gate_unit_verified"
        ]
        is True
    )
    assert (
        manifest["data_export"][
            "mesh_truth_manipulation_target_training_gate_unit_verified"
        ]
        is True
    )
    assert manifest["data_export"]["real_episode_export_verified"] is False
    assert (
        manifest["data_export"]["real_episode_training_eligible_verified"]
        is False
    )
