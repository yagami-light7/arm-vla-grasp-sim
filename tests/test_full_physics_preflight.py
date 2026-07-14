"""校验 full-physics 运行前检查不会把缺失资产误判为可运行。"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.pipeline.preflight_full_physics as preflight_module
from scripts.pipeline.preflight_full_physics import (
    PROJECT_ROOT,
    PreflightOptions,
    _scene_marker_report,
    _task_collision_proxy_report,
    build_preflight_report,
)
from source.tasks import JsonTaskProvider


SCENE_PATH = PROJECT_ROOT / "source/scene/liangzhu/liangzhu.usda"
TASK_PATH = PROJECT_ROOT / "tasks/nav_pick_place_cola_liangzhu_pct.json"


def test_liangzhu_scene_marker_report_finds_cola_and_mat_collision() -> None:
    """文本 USDA 必须同时声明可乐、垫子和垫子碰撞 Mesh。"""

    report = _scene_marker_report(
        SCENE_PATH,
        required_prim_paths=("/World/cola", "/World/carpet"),
        collision_prim_path="/World/carpet/material",
    )

    assert report["mode"] == "usda_text_marker"
    assert report["required_prim_paths"]["/World/cola"]["status"] == "present_text_marker"
    assert report["required_prim_paths"]["/World/carpet"]["status"] == (
        "present_text_marker"
    )
    assert report["collision_prim"]["status"] == "present_text_marker"


def test_liangzhu_preflight_parses_required_curobo_support_proxies() -> None:
    """真实运行前检查必须同时看到地面和垫子局部碰撞代理。"""

    report = _task_collision_proxy_report(JsonTaskProvider().load(TASK_PATH))

    assert report["phases"]["pick"]["required_count"] == 1
    assert report["phases"]["pick"]["semantic_roles"] == ["floor_support"]
    assert report["phases"]["place"]["required_count"] == 1
    assert report["phases"]["place"]["semantic_roles"] == ["mat_support"]
    assert report["phases"]["place"]["padding_modes"] == ["preserve_top"]
    assert report["phases"]["place"]["source_collision_ply_sha256"] == []
    assert report["phases"]["place"]["source_prim_paths"] == [
        "/World/carpet/material"
    ]
    assert report["phases"]["place"]["source_assets"] == [
        {
            "path": "source/scene/objects/carpet.usd",
            "sha256": (
                "817ff5412fedeb712ad23e8f137d698d3247f2416831bbfebfe376ab7b8cfd04"
            ),
        }
    ]


def test_liangzhu_preflight_uses_task_scene_runtime_and_rejects_cli_mismatch() -> None:
    """task prim 是唯一真值；CLI 不能把 Liangzhu 静默指回旧 collision。"""

    report = build_preflight_report(
        PreflightOptions(
            task_json=TASK_PATH,
            global_planner="pct",
            policy_profile="pct_multifloor",
            collision_prim_path="/World/scene_collision",
            require_cuda=False,
            minimum_free_gb=0.0,
        )
    )

    scene_runtime = report["task"]["scene_runtime"]
    assert scene_runtime["collision_prim_path"] == (
        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
    )
    assert scene_runtime["visual_prim_path"] == (
        "/World/VisualScene/GaussianScene"
    )
    assert scene_runtime["collision_floor_proxy_profile"] is None
    assert report["checks"]["scene_structure"]["collision_prim"]["status"] == (
        "present_text_marker"
    )
    assert report["checks"]["scene_structure"]["required_prim_paths"][
        "/World/VisualScene/GaussianScene"
    ]["status"] == "present_text_marker"
    assert report["checks"]["scene_structure"]["required_prim_paths"][
        "/World/carpet"
    ]["status"] == "present_text_marker"
    assert report["checks"]["scene_structure"]["required_prim_paths"][
        "/World/carpet/material"
    ]["status"] == "present_text_marker"
    support_settings = report["checks"]["task_receptacle_support"]
    assert support_settings["configured"] is True
    assert support_settings["runtime_validation_required"] is True
    assert support_settings["support_expected_static"] is True
    assert support_settings["target_support_prim_path"] == "/World/carpet/material"
    assert report["task"]["receptacle_support_runtime_validation"] == {
        "configured": True,
        "required": True,
        "support_expected_static": True,
    }
    mat_asset = str((PROJECT_ROOT / "source/scene/objects/carpet.usd").resolve())
    assert report["checks"]["task_collision_source_assets"][mat_asset][
        "sha256_matches"
    ] is True
    assert any(
        "CLI collision prim 与 task.scene_runtime 不一致" in item
        for item in report["errors"]
    )


def test_preflight_reports_missing_runtime_assets_without_starting_sim(tmp_path: Path) -> None:
    """缺少 PCT 运行资产时应阻塞报告，而不是启动仿真或静默 fallback。"""

    task_path = tmp_path / "liangzhu_preflight_task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": 2001,
                "episode_id": 1,
                "instruction": "将可乐放到指定垫子上。",
                "scene_usd": str(SCENE_PATH),
                "nav_map": "",
                "start": {"x": -1.25, "y": 4.49, "z": 0.29, "yaw": 0.0},
                "pick": {
                    "base_goal": {"x": -1.0, "y": 5.0, "z": 0.29, "yaw": 0.0},
                    "object_prim_path": "/World/cola",
                    "object_pose_world": {
                        "x": -0.64,
                        "y": 7.63,
                        "z": 0.20,
                        "yaw": 0.0,
                    },
                },
                "place": {
                    "enabled": True,
                    "base_goal": {"x": -0.9, "y": 7.1, "z": 0.29, "yaw": 0.0},
                    "place_pose_world": {
                        "x": -0.4375161874288581,
                        "y": 5.111811405946711,
                        "z": -0.08431263665173105,
                        "yaw": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_preflight_report(
        PreflightOptions(
            task_json=task_path,
            global_planner="pct",
            policy_profile="flat",
            require_cuda=False,
            minimum_free_gb=0.0,
        )
    )

    assert report["ready_for_real_full_physics"] is False
    assert report["task"]["object_prim_path"] == "/World/cola"
    assert (
        report["checks"]["scene_structure"]["required_prim_paths"]["/World/cola"]["status"]
        == "present_text_marker"
    )
    assert any("PCT server_script 不存在" in item for item in report["errors"])


def test_preflight_idle_gate_separates_ready_assets_from_launch_safety(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """资产可用时，其他 runtime 仍必须单独阻止新 Isaac 启动。"""

    server = tmp_path / "pct_grid_server.py"
    tomogram = tmp_path / "map.pickle"
    walkable = tmp_path / "walkable.npy"
    checkpoint = tmp_path / "model.pt"
    for path in (server, tomogram, walkable, checkpoint):
        path.write_bytes(b"preflight-test")
    monkeypatch.setattr(
        preflight_module,
        "_background_process_report",
        lambda: {
            "available": True,
            "blocking_process_count": 1,
            "processes": [
                {
                    "pid": 12345,
                    "name": "python",
                    "cmdline": "python run_full_physics_pipeline.py",
                    "cwd": "/tmp/other_worktree",
                    "category": "isaac_full_physics_pipeline",
                }
            ],
        },
    )

    report = build_preflight_report(
        PreflightOptions(
            task_json=TASK_PATH,
            global_planner="pct",
            policy_profile="pct_multifloor",
            locomotion_checkpoint=checkpoint,
            pct_server_script=server,
            pct_tomogram_path=tomogram,
            pct_walkable_path=walkable,
            require_cuda=False,
            require_idle_runtime=True,
            minimum_free_gb=0.0,
        )
    )

    assert report["asset_checks_passed"] is True
    assert report["runtime_launch_safe"] is False
    assert report["ready_for_real_full_physics"] is False
    assert any("禁止并发启动新仿真" in item for item in report["errors"])


def test_preflight_allows_compatible_shared_curobo_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """单个能力兼容的常驻 CuRobo 服务可复用，不应冒充 Isaac 并发冲突。"""

    server = tmp_path / "pct_grid_server.py"
    tomogram = tmp_path / "map.pickle"
    walkable = tmp_path / "walkable.npy"
    checkpoint = tmp_path / "model.pt"
    for path in (server, tomogram, walkable, checkpoint):
        path.write_bytes(b"preflight-test")
    monkeypatch.setattr(
        preflight_module,
        "_background_process_report",
        lambda: {
            "available": True,
            "blocking_process_count": 1,
            "processes": [
                {
                    "pid": 28191,
                    "name": "python",
                    "cmdline": "python grasp_planner_server.py",
                    "cwd": "/tmp/shared_worktree",
                    "category": "curobo_server",
                }
            ],
        },
    )
    monkeypatch.setattr(preflight_module, "planner_server_ping", lambda: True)
    monkeypatch.setattr(
        preflight_module,
        "planner_server_supports_required_features",
        lambda: True,
    )

    report = build_preflight_report(
        PreflightOptions(
            task_json=TASK_PATH,
            global_planner="pct",
            policy_profile="pct_multifloor",
            locomotion_checkpoint=checkpoint,
            pct_server_script=server,
            pct_tomogram_path=tomogram,
            pct_walkable_path=walkable,
            require_cuda=False,
            require_idle_runtime=True,
            minimum_free_gb=0.0,
        )
    )

    runtime = report["checks"]["background_processes"]
    assert report["asset_checks_passed"] is True
    assert report["runtime_launch_safe"] is True
    assert report["ready_for_real_full_physics"] is True
    assert runtime["blocking_process_count"] == 0
    assert runtime["allowed_process_count"] == 1
    assert runtime["shared_curobo_server"]["allowed"] is True


def test_preflight_rejects_incompatible_shared_curobo_server(
    monkeypatch,
) -> None:
    """端口存活但能力不足的 CuRobo 服务仍必须保持原进程并阻塞启动。"""

    monkeypatch.setattr(preflight_module, "planner_server_ping", lambda: True)
    monkeypatch.setattr(
        preflight_module,
        "planner_server_supports_required_features",
        lambda: False,
    )
    report = preflight_module._apply_runtime_process_gate(
        {
            "available": True,
            "processes": [
                {
                    "pid": 12345,
                    "category": "curobo_server",
                }
            ],
        }
    )

    assert report["blocking_process_count"] == 1
    assert report["allowed_process_count"] == 0
    assert report["shared_curobo_server"]["allowed"] is False
