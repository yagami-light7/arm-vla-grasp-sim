"""校验 Liangzhu box1 到 box2 的集中式任务标注。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.recording.subtask_segmentation import (
    INSTRUCTION_ANNOTATION_SCHEMA,
    RELATIVE_DIRECTION_LABELS,
    SUBTASK_DIRECTORY_LAYOUT,
    SUBTASK_SCHEMA_VERSION,
)
from source.pipeline.state_machine import _navigation_plan_execution_metadata
from source.tasks import JsonTaskProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = PROJECT_ROOT / "tasks/liangzhu_placement_target.json"
TASK_PATH = (
    PROJECT_ROOT
    / "tasks/nav_pick_place_cola_box1_to_box2_liangzhu_pct.json"
)
SCENE_PATH = PROJECT_ROOT / "source/scene/liangzhu/liangzhu.usda"


def _annotation() -> dict:
    return json.loads(TARGET_PATH.read_text(encoding="utf-8"))


def test_box_pair_annotation_is_the_task_specific_single_source() -> None:
    annotation = _annotation()
    episode = JsonTaskProvider().load(TASK_PATH)
    task = episode.raw_task

    assert annotation["schema_version"] == "liangzhu_box_pair_task_annotation_v1"
    assert annotation["annotation_id"] == "liangzhu_box1_cola_to_box2"
    assert episode.task_id == 2002
    assert episode.instruction == (
        "Pick up the coke can on box1 and place it on box2."
    )
    assert task["annotation_config_report"]["task_overrides_applied"] is True
    assert task["source_receptacle_id"] == "box1_01"
    assert task["target_receptacle_id"] == "box2_01"
    assert task["pick"]["target_support_prim_path"] == "/World/box1/node_0"
    assert task["place"]["target_support_prim_path"] == "/World/box2/node_0"
    assert task["navigation_execution"]["place_position_tolerance"] == 0.08
    alignment = task["navigation_execution"]["same_floor_alignment"]
    assert alignment["rotate_in_place_enter_angle_rad"] == 0.45
    assert alignment["rotate_in_place_exit_angle_rad"] == 0.2
    assert alignment["rotation_settle_angular_velocity_rps"] == 0.12
    assert alignment["large_heading_creep_velocity_mps"] == 0.0
    assert task["navigation_execution"]["carry_departure"]["enabled"] is True
    assert task["place"]["place_linear_velocity_tolerance_mps"] == 0.1
    assert task["place"]["place_angular_velocity_tolerance_rps"] == 2.0
    assert task["randomization"]["box_pair"][
        "place_base_standoff_range_m"
    ] == [0.48, 0.51]
    assert task["place"]["retreat_clearance"] == task["place"][
        "pre_place_clearance"
    ]


def test_carry_departure_clearance_is_derived_from_live_support_geometry() -> None:
    task = JsonTaskProvider().load(TASK_PATH).raw_task

    metadata = _navigation_plan_execution_metadata(
        task,
        include_carry_departure=True,
    )

    departure = metadata["carry_departure"]
    bbox = task["pick"]["support_geometry"]
    dims = bbox["world_bbox_dims_xyz"]
    expected_half_diagonal = 0.5 * (dims[0] ** 2 + dims[1] ** 2) ** 0.5
    assert departure["source_support_center_xy"] == pytest.approx(
        [
            0.5 * (bbox["world_bbox_min_xyz"][0] + bbox["world_bbox_max_xyz"][0]),
            0.5 * (bbox["world_bbox_min_xyz"][1] + bbox["world_bbox_max_xyz"][1]),
        ]
    )
    assert departure["source_support_half_diagonal_m"] == pytest.approx(
        expected_half_diagonal
    )
    assert departure["required_center_clearance_m"] == pytest.approx(
        expected_half_diagonal + 0.5 + 0.08
    )


def test_scene_contains_both_box_assets_and_coke() -> None:
    scene_text = SCENE_PATH.read_text(encoding="utf-8")

    assert 'def "box1" (' in scene_text
    assert 'def "box2" (' in scene_text
    assert 'def "cola" (' in scene_text
    assert "prepend payload = @../objects/box/box.usd@" in scene_text
    assert "prepend payload = @../objects/box2/box2.usd@" in scene_text


def test_box_randomization_changes_only_root_xy() -> None:
    task = _annotation()["task_overrides"]
    config = task["randomization"]
    box_pair = config["box_pair"]

    assert config["mode"] == "liangzhu_box_pair_xy_v1"
    assert box_pair["robot_yaw_range_deg"] == [-180.0, 180.0]
    assert box_pair["robot_segment_fraction_range"] == [0.4, 0.6]
    for name, prim_path in (("box1", "/World/box1"), ("box2", "/World/box2")):
        box = box_pair[name]
        support_pose = task["pick" if name == "box1" else "place"][
            "support_pose_world" if name == "box1" else "receptacle_pose_world"
        ]
        assert box["root_prim_path"] == prim_path
        assert len(box["center_x_offset_range_m"]) == 2
        assert len(box["center_y_offset_range_m"]) == 2
        assert support_pose["translation_only"] is True
        assert support_pose["ensure_static_mesh_collision"] is True
        assert support_pose["z"] == box["root_translate_xyz"][2]


def test_cola_and_place_regions_fit_inside_box_surfaces() -> None:
    task = _annotation()["task_overrides"]
    config = task["randomization"]["box_pair"]
    cola_half = config["cola_center_region_half_extent_xy_m"]
    place_half = config["placement_region_half_extent_xy_m"]
    radius = config["cola_footprint_radius_m"]

    for half, box_name, clearance_key in (
        (cola_half, "box1", "cola_table_edge_clearance_m"),
        (place_half, "box2", "placement_edge_clearance_m"),
    ):
        dims = config[box_name]["support_dims_xyz"]
        clearance = config[clearance_key]
        assert half[0] + radius + clearance <= dims[0] * 0.5
        assert half[1] + radius + clearance <= dims[1] * 0.5

    object_half_height = config["cola_bbox_center_to_min_z_m"]
    assert abs(
        task["pick"]["object_pose_world"]["z"]
        - config["box1"]["support_top_z"]
        - object_half_height
    ) < 1.0e-12
    assert abs(
        task["place"]["place_pose_world"]["z"]
        - config["box2"]["support_top_z"]
        - object_half_height
    ) < 1.0e-12


def test_subtask_and_english_instruction_annotations_are_centralized() -> None:
    task = _annotation()["task_overrides"]
    segmentation = task["subtask_segmentation"]
    instruction = task["instruction_annotation"]

    assert segmentation["schema"] == SUBTASK_SCHEMA_VERSION
    assert segmentation["output_layout"] == SUBTASK_DIRECTORY_LAYOUT
    assert instruction["schema"] == INSTRUCTION_ANNOTATION_SCHEMA
    assert instruction["language"] == "en"
    assert tuple(instruction["direction_labels"]) == RELATIVE_DIRECTION_LABELS
    assert instruction["templates"] == {
        "find_pick_box": (
            "Turn toward your {direction} to find the box with the coke can."
        ),
        "pick_from_front_box": (
            "Pick up the coke can from the box in front of you."
        ),
        "find_place_box": (
            "Turn toward your {direction} to find the box where you can place "
            "the coke can."
        ),
        "place_on_front_box": (
            "Place the coke can on the box in front of you."
        ),
    }


def test_box_pair_annotation_records_real_pipeline_and_dataset_validation() -> None:
    audit = _annotation()["geometry_audit"]
    validation = audit["full_pipeline_validation"]

    assert audit["full_pipeline_success_verified"] is True
    assert validation["seed"] == 5002
    assert validation["final_state"] == "done"
    assert validation["physical_navigation_success"] is True
    assert validation["physical_manipulation_success"] is True
    assert validation["stable_physics_success"] is True
    assert validation["pure_physics_success"] is False
    assert validation["training_eligible"] is True
    assert validation["lerobot_rows"] == 196
    assert {
        validation["front_frames"],
        validation["wrist_frames"],
        validation["overview_frames"],
    } == {196}
    assert validation["validation_errors"] == 0
    assert validation["validation_warnings"] == 0
