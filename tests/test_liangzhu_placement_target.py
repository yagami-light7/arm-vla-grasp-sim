"""校验 Liangzhu 可乐到垫子的固定放置目标。"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = PROJECT_ROOT / "tasks/liangzhu_placement_target.json"
SCENE_PATH = PROJECT_ROOT / "source/scene/liangzhu/liangzhu.usda"
MAT_ASSET_PATH = PROJECT_ROOT / "source/scene/objects/carpet.usd"


def test_liangzhu_mat_target_matches_scene_receptacle_transform() -> None:
    """固定目标必须绑定场景中的垫子，而不是旧桌面标注点。"""

    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    scene_text = SCENE_PATH.read_text(encoding="utf-8")
    pose = target["placement_pose_world"]

    assert target["target_object_prim_path"] == "/World/cola"
    assert target["target_receptacle_id"] == "mat_carpet_01"
    assert target["target_receptacle_aliases"] == [
        "mouse_mat",
        "mouse_pad",
        "mat",
        "carpet",
    ]
    assert target["target_receptacle_prim_path"] == "/World/carpet"
    assert target["target_support_prim_path"] == "/World/carpet/material"
    assert target["support_runtime_validation_required"] is True
    assert target["support_expected_static"] is True
    assert pose["position_xyz"] == [
        -0.4375161874288581,
        5.111811405946711,
        -0.09712594670872055,
    ]
    assert pose["orientation_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert target["task_schema_fragment"]["place"]["place_pose_world"] == {
        "x": -0.4375161874288581,
        "y": 5.111811405946711,
        "z": -0.09712594670872055,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    assert 'def "carpet" (' in scene_text
    assert "prepend payload = @../objects/carpet.usd@" in scene_text
    assert (
        "double3 xformOp:translate = "
        "(-0.438209287414772, 5.042934329921819, -0.13822162094644663)"
    ) in scene_text
    assert "float3 xformOp:scale = (0.2, 0.15, 0.005)" in scene_text
    assert MAT_ASSET_PATH.is_file()


def test_liangzhu_mat_target_geometry_is_self_consistent() -> None:
    """目标中心、安全区域和碰撞包围盒必须使用同一组几何。"""

    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    pose = target["placement_pose_world"]["position_xyz"]
    region = target["placement_region"]
    support = target["support_geometry"]
    bbox_min = support["world_bbox_min_xyz"]
    bbox_max = support["world_bbox_max_xyz"]
    bbox_center = support["world_bbox_center_xyz"]
    bbox_dims = support["world_bbox_dims_xyz"]

    for axis in range(3):
        assert abs(bbox_center[axis] - (bbox_min[axis] + bbox_max[axis]) * 0.5) < 1.0e-12
        assert abs(bbox_dims[axis] - (bbox_max[axis] - bbox_min[axis])) < 1.0e-12
    assert abs(pose[0] - bbox_center[0]) < 1.0e-12
    assert abs(pose[1] - bbox_center[1]) < 1.0e-12
    assert abs(
        pose[2]
        - support["support_surface_z"]
        - support["object_bbox_center_to_min_z_m"]
    ) < 1.0e-12
    assert region["x_min"] < pose[0] < region["x_max"]
    assert region["y_min"] < pose[1] < region["y_max"]
    assert abs(region["z_surface"] - support["support_surface_z"]) < 1.0e-12
    assert abs(
        support["mat_top_above_scene_floor_m"]
        - support["world_bbox_dims_xyz"][2]
        - support["mat_bottom_clearance_above_highest_floor_m"]
    ) < 1.0e-15
    assert support["mat_bottom_clearance_above_highest_floor_m"] > 0.0
    assert support["receptacle_asset_sha256"] == (
        "817ff5412fedeb712ad23e8f137d698d3247f2416831bbfebfe376ab7b8cfd04"
    )


def test_liangzhu_mat_target_uses_actual_coke_mesh_half_height() -> None:
    """放置半高必须来自组合可乐 Mesh，不能复用 spawn 到地面的距离。"""

    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    geometry = target["target_object_mesh_geometry"]
    bbox_min = geometry["authored_world_bbox_min_xyz"]
    bbox_max = geometry["authored_world_bbox_max_xyz"]
    bbox_center = geometry["authored_world_bbox_center_xyz"]
    half_height = bbox_center[2] - bbox_min[2]

    assert abs(bbox_center[2] - (bbox_min[2] + bbox_max[2]) * 0.5) < 1.0e-15
    assert abs(half_height - 0.05364498018521855) < 1.0e-15
    assert abs(
        target["support_geometry"]["object_bbox_center_to_min_z_m"]
        - half_height
    ) < 1.0e-15
    assert abs(
        target["placement_pose_world"]["position_xyz"][2]
        - target["support_geometry"]["support_surface_z"]
        - half_height
    ) < 1.0e-15


def test_liangzhu_mat_target_keeps_runtime_validation_pending() -> None:
    """离线垫子几何成功不能冒充真实释放稳定性已验收。"""

    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    validation = target["validation"]

    assert validation["receptacle_scene_transform_matches"] is False
    assert validation["receptacle_task_pose_override_verified"] is True
    assert validation["mat_collision_api_offline_verified"] is True
    assert validation["mat_bbox_geometry_verified"] is True
    assert validation["mat_ground_clearance_geometry_verified"] is True
    assert validation["object_footprint_fits_safe_region"] is True
    assert validation["collision_support_runtime_verified"] is False
    assert validation["release_stability_verified"] is False
