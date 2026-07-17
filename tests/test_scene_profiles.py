"""场景 profile 注册表与任务绑定测试。"""

from __future__ import annotations

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
from scripts.pipeline.run_full_physics_pipeline import (
    _scene_isaac_app_overrides,
    _scene_isaac_kit_args,
    _scene_isaac_runtime_overrides,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scene_profile_registry_discovers_both_supported_scenes() -> None:
    profiles = list_scene_profiles(PROJECT_ROOT)

    assert [profile.name for profile in profiles] == ["liangzhu", "multi_floor"]
    assert load_scene_profile("liangzhu_single_floor", PROJECT_ROOT).name == "liangzhu"
    assert load_scene_profile("villa", PROJECT_ROOT).name == "multi_floor"
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
