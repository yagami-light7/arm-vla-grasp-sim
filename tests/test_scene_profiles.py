"""场景 profile 注册表与任务绑定测试。"""

from __future__ import annotations

import ast
from collections import Counter
import json
from dataclasses import replace
from pathlib import Path

from source.scene.profiles import (
    SceneUsdAssetBinding,
    apply_scene_profile_defaults,
    check_scene_profile_assets,
    list_scene_profiles,
    load_scene_profile,
)
from source.scene.runtime_assets import materialize_scene_asset_bindings
from source.tasks.task_loader import JsonTaskProvider
from scripts.pipeline.run_full_physics_pipeline import (
    _scene_isaac_app_overrides,
    _scene_isaac_kit_args,
    _scene_isaac_runtime_overrides,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scene_profile_registry_discovers_supported_scenes() -> None:
    profiles = list_scene_profiles(PROJECT_ROOT)

    assert [profile.name for profile in profiles] == [
        "liangzhu",
        "multi_floor",
        "ramp_validation",
    ]
    assert load_scene_profile("liangzhu_single_floor", PROJECT_ROOT).name == "liangzhu"
    assert load_scene_profile("villa", PROJECT_ROOT).name == "multi_floor"
    assert load_scene_profile("scan_ramp", PROJECT_ROOT).name == "ramp_validation"
    assert not load_scene_profile("liangzhu", PROJECT_ROOT).supports(
        "stair_locomotion_smoke"
    )
    assert load_scene_profile("multi_floor", PROJECT_ROOT).supports(
        "stair_locomotion_smoke"
    )
    assert load_scene_profile("liangzhu", PROJECT_ROOT).supports("nurec_visual")
    assert not load_scene_profile("multi_floor", PROJECT_ROOT).supports(
        "nurec_visual"
    )
    assert load_scene_profile("ramp_validation", PROJECT_ROOT).supports(
        "slope_locomotion_smoke"
    )


def test_scene_profile_registry_discovers_new_json_without_python_enum(
    tmp_path: Path,
) -> None:
    """新增普通场景只需增加 JSON，不应修改 Python 场景枚举。"""

    config_dir = tmp_path / "scene_profiles"
    config_dir.mkdir()
    profile_path = config_dir / "annotated_lab.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "annotated_lab",
                "aliases": ["lab_v1"],
                "description": "带已标注 pick/place/nav 交接点的测试场景。",
                "capabilities": ["navigation", "manipulation", "data_export"],
                "task_scene_profile": "annotated_lab_v1",
                "runtime_asset_manifest": (
                    "source/scene/annotated_lab/runtime_asset_manifest.json"
                ),
                "defaults": {
                    "task_json": "tasks/annotated_lab.json",
                    "global_planner": "pct",
                    "pct_coord_mode": "identity",
                },
                "mode_defaults": {},
                "usd_asset_bindings": [],
                "required_assets": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profiles = list_scene_profiles(PROJECT_ROOT, config_dir=config_dir)
    loaded = load_scene_profile(
        "lab_v1",
        PROJECT_ROOT,
        config_dir=config_dir,
    )

    assert [profile.name for profile in profiles] == ["annotated_lab"]
    assert loaded.name == "annotated_lab"
    assert loaded.task_scene_profile == "annotated_lab_v1"
    assert loaded.defaults["task_json"] == "tasks/annotated_lab.json"


def test_liangzhu_nurec_profile_forces_single_gpu_taa_before_rendering() -> None:
    liangzhu = load_scene_profile("liangzhu", PROJECT_ROOT)
    multi_floor = load_scene_profile("multi_floor", PROJECT_ROOT)

    assert _scene_isaac_app_overrides(liangzhu) == {
        "multi_gpu": False,
        "anti_aliasing": 1,
    }
    assert "--/renderer/multiGpu/enabled=false" in _scene_isaac_kit_args(
        liangzhu
    )
    assert "--/rtx/post/aa/op=1" in _scene_isaac_kit_args(liangzhu)
    assert "--/rtx-defaults/post/aa/op=1" in _scene_isaac_kit_args(liangzhu)
    assert _scene_isaac_app_overrides(multi_floor) == {}
    assert _scene_isaac_kit_args(multi_floor) == ()
    assert _scene_isaac_runtime_overrides(liangzhu) == {
        "render_antialiasing_mode": "TAA"
    }
    assert _scene_isaac_runtime_overrides(multi_floor) == {}


def test_scene_profiles_bind_matching_task_scene_names() -> None:
    for name in ("liangzhu", "multi_floor"):
        profile = load_scene_profile(name, PROJECT_ROOT)
        task_path = PROJECT_ROOT / str(profile.defaults["task_json"])
        task = json.loads(task_path.read_text(encoding="utf-8"))

        assert task["scene_profile"] == profile.task_scene_profile
        assert (PROJECT_ROOT / profile.runtime_asset_manifest).is_file()
        assert task["training_action"]["schema"] == (
            "base_xyyaw_tcp_base_rpy_gripper_v1"
        )
        assert task["training_action"]["dimension"] == 10
        assert task["recording"]["front_camera"] is True
        assert task["recording"]["wrist_camera"] is True
        assert task["subtask_segmentation"]["output_layout"] == (
            "episodes_task_episode_subtask_front_wrist_v3"
        )


def test_ramp_validation_profile_binds_minimal_scene_task_and_path_contract() -> None:
    """斜坡资产必须由统一 profile、task 和手工 Path 同时描述。"""

    profile = load_scene_profile("ramp_validation", PROJECT_ROOT)
    task_path = PROJECT_ROOT / str(profile.defaults["task_json"])
    task = json.loads(task_path.read_text(encoding="utf-8"))
    episode = JsonTaskProvider().load(task_path)
    manifest = json.loads(
        (PROJECT_ROOT / profile.runtime_asset_manifest).read_text(
            encoding="utf-8"
        )
    )
    path_file = (
        PROJECT_ROOT
        / "ros2_ws/src/scan_navigation_tools/config/validation_ramp_path.yaml"
    )
    path_values = [
        float(line.strip()[2:])
        for line in path_file.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]
    points = tuple(
        tuple(path_values[index : index + 3])
        for index in range(0, len(path_values), 3)
    )
    geometry = manifest["geometry_contract"]
    scene_text = (PROJECT_ROOT / task["scene_usd"]).read_text(encoding="utf-8")
    point_blocks = scene_text.split("point3f[] points = [")
    assert len(point_blocks) == 3
    visual_points = ast.literal_eval(
        "[" + point_blocks[1].split("]", 1)[0] + "]"
    )
    collision_points = ast.literal_eval(
        "[" + point_blocks[2].split("]", 1)[0] + "]"
    )
    assert task["scene_profile"] == profile.task_scene_profile
    assert profile.defaults["global_planner"] == "pct"
    assert profile.defaults["policy_profile"] == "pct_multifloor"
    assert profile.defaults["randomize_task"] is False
    assert profile.defaults["randomize_base_goal"] is False
    assert profile.defaults["navigation_visual_mode"] == "collision"
    assert task["scene_runtime"] == {
        "collision_prim_path": "/World/CollisionScene/RampSurface",
        "visual_prim_path": "/World/VisualScene/RampSurface",
        "collision_floor_proxy_profile": None,
    }
    assert (PROJECT_ROOT / task["scene_usd"]).is_file()
    assert task["nav_map"] == ""
    assert episode.start.z == 0.3
    assert episode.pick_goal.z == 0.48
    assert manifest["runtime_contract"]["pipeline_entry"] == (
        "scripts/pipeline/run_full_physics_pipeline.py"
    )
    assert manifest["runtime_contract"]["requires_navigation_ros2_bridge"] is True
    assert manifest["runtime_contract"]["policy_profile"] == "pct_multifloor"

    assert len(points) == 9
    assert points[0] == tuple(geometry["path_start_xyz"])
    assert points[2] == tuple(geometry["ramp_start_xyz"])
    assert points[6] == tuple(geometry["ramp_end_xyz"])
    assert points[-1] == tuple(geometry["path_goal_xyz"])
    assert geometry["path_flat_approach_m"] == 0.5
    assert geometry["ramp_run_m"] == 1.2
    assert geometry["ramp_rise_m"] == 0.18
    assert geometry["path_top_platform_m"] == 0.5
    assert geometry["surface_width_m"] >= 1.2
    assert visual_points == collision_points
    assert 'quatd xformOp:orient = (0.8660254037844386, 0.5, 0, 0)' in scene_text
    assert 'double3 xformOp:scale = (1, 1, 1)' in scene_text
    assert '"xformOp:rotateXYZ"' not in scene_text
    assert scene_text.index('"xformOp:translate"') < scene_text.index(
        '"xformOp:orient"'
    ) < scene_text.index('"xformOp:scale"')
    assert min(point[0] for point in collision_points) == -1.3
    assert max(point[0] for point in collision_points) == 2.5
    assert (
        max(point[1] for point in collision_points)
        - min(point[1] for point in collision_points)
        == geometry["surface_width_m"]
    )
    assert geometry["task_base_height_m"] == 0.3
    assert geometry["collision_solid_thickness_m"] == 0.12
    assert geometry["support_padding_behind_start_m"] == 0.8
    assert geometry["support_padding_beyond_goal_m"] == 0.8
    assert episode.start.z - points[0][2] == geometry["task_base_height_m"]
    assert episode.pick_goal.z - points[-1][2] == geometry["task_base_height_m"]

    face_count_blocks = scene_text.split("int[] faceVertexCounts = [")
    face_index_blocks = scene_text.split("int[] faceVertexIndices = [")
    assert len(face_count_blocks) == 3
    assert len(face_index_blocks) == 3
    collision_face_counts = ast.literal_eval(
        "[" + face_count_blocks[2].split("]", 1)[0] + "]"
    )
    collision_face_indices = ast.literal_eval(
        "[" + face_index_blocks[2].split("]", 1)[0] + "]"
    )
    visual_face_counts = ast.literal_eval(
        "[" + face_count_blocks[1].split("]", 1)[0] + "]"
    )
    visual_face_indices = ast.literal_eval(
        "[" + face_index_blocks[1].split("]", 1)[0] + "]"
    )
    # Isaac Lab RayCaster 将索引流按每三个顶点解释为一个三角形，
    # 所以运行时碰撞网格不能依赖 USD 的多边形面计数。
    assert set(collision_face_counts) == {3}
    assert visual_face_counts == collision_face_counts
    assert visual_face_indices == collision_face_indices
    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_edge_counts: Counter[tuple[int, int]] = Counter()
    signed_volume_times_six = 0.0
    surface_area_times_two = 0.0
    cursor = 0
    for face_count in collision_face_counts:
        face = collision_face_indices[cursor : cursor + face_count]
        cursor += face_count
        for index, vertex in enumerate(face):
            next_vertex = face[(index + 1) % face_count]
            edge = tuple(sorted((vertex, next_vertex)))
            edge_counts[edge] += 1
            directed_edge_counts[(vertex, next_vertex)] += 1
        point_a, point_b, point_c = (
            collision_points[vertex] for vertex in face
        )
        edge_ab = tuple(
            point_b[index] - point_a[index] for index in range(3)
        )
        edge_ac = tuple(
            point_c[index] - point_a[index] for index in range(3)
        )
        cross = (
            edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
            edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
            edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
        )
        cross_norm = sum(component * component for component in cross) ** 0.5
        assert cross_norm > 1.0e-12
        surface_area_times_two += cross_norm
        signed_volume_times_six += sum(
            point_a[index] * (
                point_b[(index + 1) % 3] * point_c[(index + 2) % 3]
                - point_b[(index + 2) % 3] * point_c[(index + 1) % 3]
            )
            for index in range(3)
        )
    assert cursor == len(collision_face_indices)
    assert edge_counts
    assert set(edge_counts.values()) == {2}
    assert all(
        directed_edge_counts[(edge[0], edge[1])] == 1
        and directed_edge_counts[(edge[1], edge[0])] == 1
        for edge in edge_counts
    )
    assert abs(signed_volume_times_six / 6.0 - 1.2768) < 1.0e-12
    assert abs(surface_area_times_two / 2.0 - 14.449479847951) < 1.0e-12
    for x, y, z in points[2:7]:
        assert y == 0.0
        assert abs(z - 0.15 * x) < 1.0e-12


def test_multi_floor_stair_mode_overrides_only_unset_cli_fields() -> None:
    class Namespace:
        output_dir = None
        pct_stair_float = None
        show_planned_trajectories = None
        overview_camera_prim_path = "/World/UserCamera"
        overview_camera_schedule = None
        video_mode = None
        task_json = None
        global_planner = None
        pct_server_script = None
        pct_tomogram_path = None
        pct_walkable_path = None
        pct_collision_ply_path = None
        pct_no_fallback = None
        pct_coord_mode = None
        pct_cross_floor_gateway = None
        pct_cross_floor_stair_exit = None
        pct_cross_floor_stair_midpoint = None
        policy_profile = None
        locomotion_task = None
        locomotion_checkpoint = None
        randomize_task = None
        randomize_base_goal = None
        navigation_visual_mode = None
        overview_camera_mode = None

    namespace = Namespace()
    profile = load_scene_profile("multi_floor", PROJECT_ROOT)
    applied = apply_scene_profile_defaults(
        namespace,
        profile,
        mode="stair_locomotion_smoke",
    )

    assert namespace.output_dir == "outputs/multi_floor_stair_locomotion_smoke"
    assert namespace.pct_stair_float is False
    assert namespace.overview_camera_prim_path == "/World/UserCamera"
    assert "overview_camera_prim_path" not in applied


def test_scene_asset_binding_uses_environment_without_modifying_source(
    tmp_path: Path,
) -> None:
    source_scene = tmp_path / "source.usda"
    source_scene.write_text(
        (
            '#usda 1.0\n(defaultPrim = "World")\n'
            'def Xform "World"\n{\n'
            '    def Xform "VisualScene"\n    {\n'
            '        def Xform "GaussianScene" (\n'
            f'            references = @{tmp_path / "missing_visual.usdz"}[gauss.usda]@\n'
            '        ) {}\n    }\n'
            '    def Xform "PhysicsScene"\n    {\n'
            '        def Xform "CollisionScene" (\n'
            f'            payload = @{tmp_path / "missing_collision.usda"}@\n'
            '        ) {}\n    }\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    visual = tmp_path / "portable_visual.usdz"
    collision = tmp_path / "portable_collision.usda"
    visual.write_bytes(b"visual-placeholder")
    collision.write_text("#usda 1.0\n", encoding="utf-8")

    profile = replace(
        load_scene_profile("liangzhu", PROJECT_ROOT),
        required_assets=(),
        usd_asset_bindings=(
            SceneUsdAssetBinding(
                name="visual",
                prim_path="/World/VisualScene/GaussianScene",
                arc_type="reference",
                environment_variable="TEST_VISUAL_USDZ",
                fallback_path=str(tmp_path / "missing_visual.usdz"),
                package_member="gauss.usda",
            ),
            SceneUsdAssetBinding(
                name="collision",
                prim_path="/World/PhysicsScene/CollisionScene",
                arc_type="payload",
                environment_variable="TEST_COLLISION_USD",
                fallback_path=str(tmp_path / "missing_collision.usda"),
            ),
        ),
    )
    environment = {
        "TEST_VISUAL_USDZ": str(visual),
        "TEST_COLLISION_USD": str(collision),
    }

    asset_check = check_scene_profile_assets(
        profile,
        PROJECT_ROOT,
        environ=environment,
    )
    output = tmp_path / "runtime" / "bound.usda"
    report = materialize_scene_asset_bindings(
        profile,
        source_scene,
        output,
        project_root=PROJECT_ROOT,
        environ=environment,
    )

    text = output.read_text(encoding="utf-8")
    assert asset_check.success
    assert set(asset_check.available) == {visual.resolve(), collision.resolve()}
    assert report["materialized"] is True
    assert {item["selected_source"] for item in report["bindings"]} == {
        "environment"
    }
    assert f"references = @{visual.resolve().as_posix()}[gauss.usda]@" in text
    assert f"payload = @{collision.resolve().as_posix()}@" in text
    assert "missing_visual.usdz" in source_scene.read_text(encoding="utf-8")
