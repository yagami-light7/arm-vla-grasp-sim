"""固定提交的官方 PCT backend 与 selector 边界测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import pickle
import struct
import threading
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from pct_ros2_adapter import backend as backend_module
from pct_ros2_adapter.backend import (
    PCTBackendConfig,
    PCTBackendError,
    PCTNoPathError,
    create_global_planner_backend,
)
from pct_ros2_adapter.upstream_backend import (
    UPSTREAM_PCT_ARCHIVE_SHA256,
    UPSTREAM_PCT_COMMIT,
    UPSTREAM_PCT_LICENSE,
    UPSTREAM_PCT_PATCH_ID,
    UPSTREAM_PCT_PATCH_SHA256,
    UPSTREAM_PCT_REPOSITORY,
    _StairApproachClearanceContract,
    UpstreamTomogramBackend,
    _StairCenterlineProfile,
    _UpstreamTomogramIndex,
    _audit_stair_approach_clearance,
    _build_body_clearance_overlay,
    _runtime_stair_profile_anchors,
    _splice_stair_centerline,
    _validate_pinned_source,
)
from pct_ros2_adapter import upstream_backend as upstream_backend_module
from source.navigation import pct_adapter as coordinate_module


PROJECT_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_ROOT = PROJECT_ROOT / "external/PCT_planner"
SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "upstream/PCT_PLANNER_SOURCE.json"
)


def _write_plane_ply(path: Path) -> None:
    vertices = np.asarray(
        [
            (-1.0, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 2.0, 0.0),
            (-1.0, 2.0, 0.0),
        ],
        dtype="<f4",
    )
    faces = ((0, 1, 2), (0, 2, 3))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    payload = bytearray(header)
    payload.extend(vertices.tobytes())
    for face in faces:
        payload.extend(struct.pack("<Biii", 3, *face))
    path.write_bytes(payload)


def _config(tmp_path: Path, **overrides: Any) -> PCTBackendConfig:
    tomogram = tmp_path / "map.pickle"
    walkable = tmp_path / "walkable.npy"
    collision = tmp_path / "collision.ply"
    data = np.zeros((5, 3, 20, 20), dtype=np.float16)
    data[0, 0] = 50.0
    data[0, 1:] = 1.0
    data[3, 0] = -0.2
    data[3, 1] = 0.0
    data[3, 2] = 1.7
    data[4] = data[3] + 1.0
    with tomogram.open("wb") as stream:
        pickle.dump(
            {
                "data": data,
                "resolution": 0.1,
                "center": np.asarray((0.5, 0.5), dtype=np.float64),
                "slice_h0": -1.0,
                "slice_dh": 0.5,
                "pct_scan_asset_kind": "synthetic_upstream_test",
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    walkable.write_bytes(b"unused-by-upstream")
    _write_plane_ply(collision)
    values: dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "tomogram_path": tomogram,
        "walkable_path": walkable,
        "collision_ply_path": collision,
        "backend_kind": "upstream",
        "upstream_source_root": UPSTREAM_ROOT,
        "coord_mode": "identity",
        "cross_floor_gateway_points": (),
        "cross_floor_stair_exit_points": (),
        "cross_floor_stair_midpoint_points": (),
        "path_sample_spacing_m": 0.20,
    }
    values.update(overrides)
    return PCTBackendConfig(**values)


def test_cross_layer_stair_profile_splice_preserves_floor_route() -> None:
    profile = _StairCenterlineProfile(
        path=Path("stair_profile.json"),
        sha256="profile-hash",
        asset_kind="test-profile",
        anchors_sim_ground_xyz=(
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.8),
            (1.0, 1.0, 1.6),
            (1.0, 2.0, 3.0),
        ),
    )
    raw_path = (
        (-2.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.1, 0.1, 0.0),
        (0.2, 1.1, 0.9),
        (0.8, 1.2, 1.7),
        (1.1, 2.1, 3.0),
        (2.0, 2.0, 3.0),
    )

    refined, report = _splice_stair_centerline(
        raw_path,
        profile=profile,
        match_tolerance_m=0.25,
    )

    assert report["applied"] is True
    assert report["reason"] == "calibrated_stair_centerline"
    assert refined[:2] == raw_path[:2]
    assert refined[2:6] == profile.anchors_sim_ground_xyz
    assert refined[-1] == raw_path[-1]
    assert report["refined_profile_start_index"] == 2
    assert report["refined_profile_end_index"] == 5


def test_cross_layer_stair_profile_splice_preserves_calibrated_approach() -> None:
    profile = _StairCenterlineProfile(
        path=Path("stair_profile.json"),
        sha256="profile-hash",
        asset_kind="test-profile",
        anchors_sim_ground_xyz=(
            (1.5, 5.7, 0.0),
            (1.6, 6.3, 0.3),
            (2.0, 7.0, 1.0),
        ),
        lower_floor_approach_anchors_sim_ground_xyz=(
            (0.7, 4.9, 0.0),
            (1.1, 5.0, 0.0),
            (1.5, 5.7, 0.0),
        ),
    )
    raw_path = (
        (0.0, 4.8, 0.0),
        (0.7, 4.8, 0.0),
        (1.5, 5.8, 0.0),
        (1.6, 6.2, 0.3),
        (2.0, 7.1, 1.0),
        (2.5, 7.0, 1.0),
    )

    refined, report = _splice_stair_centerline(
        raw_path,
        profile=profile,
        match_tolerance_m=0.20,
    )

    runtime_anchors = _runtime_stair_profile_anchors(profile)
    assert report["applied"] is True
    assert report["lower_floor_approach_anchor_count"] == 3
    assert report["stair_anchor_count"] == 3
    assert report["profile_anchor_count"] == len(runtime_anchors)
    assert refined[1 : 1 + len(runtime_anchors)] == runtime_anchors


def test_stair_approach_clearance_audit_rejects_old_late_turn() -> None:
    # 用两片很小的水平三角面复现 live 中窄口上下两侧的地形表面。
    vertices = np.asarray(
        (
            (1.20, 4.58, 0.30),
            (1.36, 4.58, 0.30),
            (1.28, 4.68, 0.30),
            (0.75, 5.25, 0.30),
            (0.90, 5.25, 0.30),
            (0.82, 5.33, 0.30),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    contract = _StairApproachClearanceContract(
        double_cylinder_radius_m=0.27,
        double_cylinder_offset_m=0.16,
        obstacle_minimum_z_m=0.05,
        obstacle_maximum_z_m=0.75,
        sample_spacing_m=0.01,
        maximum_yaw_step_rad=0.02,
        minimum_surface_clearance_m=0.285,
    )
    old_late_turn = (
        (0.70, 4.86, 0.0),
        (1.10, 4.86, 0.0),
        (1.30, 5.06, 0.0),
        (1.50, 5.70, 0.0),
    )
    calibrated_approach = (
        (0.70, 4.86, 0.0),
        (0.85, 4.88, 0.0),
        (0.99, 4.92, 0.0),
        (1.11, 4.97, 0.0),
        (1.22, 5.05, 0.0),
        (1.30, 5.15, 0.0),
        (1.39, 5.31, 0.0),
        (1.47, 5.51, 0.0),
        (1.50, 5.70, 0.0),
    )

    old_report = _audit_stair_approach_clearance(
        approach_anchors=old_late_turn,
        collision_vertices_sim=vertices,
        collision_faces=faces,
        contract=contract,
    )
    calibrated_report = _audit_stair_approach_clearance(
        approach_anchors=calibrated_approach,
        collision_vertices_sim=vertices,
        collision_faces=faces,
        contract=contract,
    )

    assert old_report["collision_free"] is False
    assert old_report["minimum_surface_clearance_m"] < 0.27
    assert calibrated_report["collision_free"] is True
    assert calibrated_report["minimum_surface_clearance_m"] >= 0.285


def test_cross_layer_stair_profile_splice_supports_descending_route() -> None:
    profile = _StairCenterlineProfile(
        path=Path("stair_profile.json"),
        sha256="profile-hash",
        asset_kind="test-profile",
        anchors_sim_ground_xyz=((0.0, 0.0, 0.0), (1.0, 1.0, 3.0)),
    )
    raw_path = (
        (2.0, 1.0, 3.0),
        (1.1, 1.0, 3.0),
        (0.1, 0.1, 0.0),
        (-1.0, 0.0, 0.0),
    )

    refined, report = _splice_stair_centerline(
        raw_path,
        profile=profile,
        match_tolerance_m=0.20,
    )

    assert report["applied"] is True
    assert report["ascending"] is False
    assert refined[1:3] == tuple(reversed(profile.anchors_sim_ground_xyz))


def test_cross_layer_stair_profile_splice_rejects_unmatched_path() -> None:
    profile = _StairCenterlineProfile(
        path=Path("stair_profile.json"),
        sha256="profile-hash",
        asset_kind="test-profile",
        anchors_sim_ground_xyz=((0.0, 0.0, 0.0), (1.0, 1.0, 3.0)),
    )

    original = ((5.0, 5.0, 0.0), (6.0, 6.0, 3.0))
    refined, report = _splice_stair_centerline(
        original,
        profile=profile,
        match_tolerance_m=0.20,
    )

    assert refined == original
    assert report["applied"] is False
    assert report["reason"] == "profile_endpoints_not_matched"


def test_cross_layer_stair_profile_never_replaces_requested_endpoints() -> None:
    profile = _StairCenterlineProfile(
        path=Path("stair_profile.json"),
        sha256="profile-hash",
        asset_kind="test-profile",
        anchors_sim_ground_xyz=((0.0, 0.0, 0.0), (1.0, 1.0, 3.0)),
    )
    requested_start = (0.02, 0.0, 0.0)
    requested_goal = (1.02, 1.0, 3.0)

    refined, report = _splice_stair_centerline(
        (requested_start, (0.5, 0.5, 1.5), requested_goal),
        profile=profile,
        match_tolerance_m=0.10,
    )

    assert report["applied"] is True
    assert refined[0] == requested_start
    assert refined[-1] == requested_goal
    assert profile.anchors_sim_ground_xyz[0] in refined
    assert profile.anchors_sim_ground_xyz[-1] in refined


def _fake_planner_module(
    observations: dict[str, Any],
    trajectory: object,
) -> ModuleType:
    module = ModuleType("fake_pinned_upstream_planner")

    class FakePathFinder:
        def get_result_matrix(self) -> object:
            return trajectory

    class FakeNativePlanner:
        def request_cancel(self) -> None:
            observations["native_cancel_count"] = int(
                observations.get("native_cancel_count", 0)
            ) + 1
            observations["native_cancelled"] = True

        def reset_cancellation(self) -> None:
            observations["native_reset_count"] = int(
                observations.get("native_reset_count", 0)
            ) + 1
            observations["native_cancelled"] = False

        def was_cancelled(self) -> bool:
            return bool(observations.get("native_cancelled", False))

        def get_last_search_status(self) -> int:
            return 0

        def get_expanded_node_count(self) -> int:
            return 0

        def plan(
            self,
            start_idx: object,
            goal_idx: object,
            optimize: bool,
        ) -> bool:
            observations["native_start_idx"] = tuple(
                int(value) for value in np.asarray(start_idx)
            )
            observations["native_goal_idx"] = tuple(
                int(value) for value in np.asarray(goal_idx)
            )
            observations["native_optimize"] = bool(optimize)
            return trajectory is not None

        def get_path_finder(self) -> FakePathFinder:
            return FakePathFinder()

    class FakeTomogramPlanner:
        def __init__(self, config: object) -> None:
            observations["planner_instances"] = int(
                observations.get("planner_instances", 0)
            ) + 1
            observations["config"] = config
            self.start_idx = np.zeros(3, dtype=np.int32)
            self.end_idx = np.zeros(3, dtype=np.int32)
            self.n_slice = None
            self.slice_h0 = None
            self.slice_dh = None
            self.planner = FakeNativePlanner()

        def loadTomogram(self, stem: str) -> None:
            observations["tomogram_stem"] = stem
            self.n_slice = 3
            self.slice_h0 = -1.0
            self.slice_dh = 0.5

        def initPlanner(
            self,
            trav: object,
            trav_gx: object,
            trav_gy: object,
            elev_g: object,
            elev_c: object,
            gateway_trav: object | None = None,
        ) -> None:
            observations["overlay_init"] = {
                "trav": np.asarray(trav).copy(),
                "trav_gx": np.asarray(trav_gx).copy(),
                "trav_gy": np.asarray(trav_gy).copy(),
                "elev_g": np.asarray(elev_g).copy(),
                "elev_c": np.asarray(elev_c).copy(),
                "gateway_trav": np.asarray(gateway_trav).copy(),
            }

        def plan(self, _start_xy: object, _goal_xy: object) -> object:
            raise AssertionError("upstream backend 不得调用 GPMP wrapper.plan")

    module.TomogramPlanner = FakeTomogramPlanner
    return module


def _stateful_fake_planner_module(
    observations: dict[str, Any],
    outcomes: tuple[object, ...],
) -> ModuleType:
    """让每个新 planner 实例消费一个结果，用于验证失败后重建。"""

    module = ModuleType("stateful_fake_pinned_upstream_planner")

    class FakePathFinder:
        def __init__(self, outcome: object) -> None:
            self._outcome = outcome

        def get_result_matrix(self) -> object:
            return self._outcome

    class FakeNativePlanner:
        def __init__(self, outcome: object, instance_index: int) -> None:
            self._outcome = outcome
            self._instance_index = instance_index

        def request_cancel(self) -> None:
            observations.setdefault("native_cancel_instances", []).append(
                self._instance_index
            )

        def reset_cancellation(self) -> None:
            observations.setdefault("native_reset_instances", []).append(
                self._instance_index
            )

        def was_cancelled(self) -> bool:
            return False

        def get_last_search_status(self) -> int:
            return 0

        def get_expanded_node_count(self) -> int:
            return 0

        def plan(
            self,
            _start_idx: object,
            _goal_idx: object,
            _optimize: bool,
        ) -> bool:
            observations.setdefault("native_plan_instances", []).append(
                self._instance_index
            )
            if isinstance(self._outcome, Exception):
                raise self._outcome
            return self._outcome is not None

        def get_path_finder(self) -> FakePathFinder:
            return FakePathFinder(self._outcome)

    class FakeTomogramPlanner:
        def __init__(self, _config: object) -> None:
            instance_index = int(observations.get("planner_instances", 0))
            if instance_index >= len(outcomes):
                raise AssertionError("测试没有为新 planner 提供结果")
            observations["planner_instances"] = instance_index + 1
            self.start_idx = np.zeros(3, dtype=np.int32)
            self.end_idx = np.zeros(3, dtype=np.int32)
            self.n_slice = None
            self.slice_h0 = None
            self.slice_dh = None
            self.planner = FakeNativePlanner(
                outcomes[instance_index],
                instance_index,
            )

        def loadTomogram(self, stem: str) -> None:
            observations.setdefault("tomogram_loads", []).append(stem)
            self.n_slice = 3
            self.slice_h0 = -1.0
            self.slice_dh = 0.5

        def plan(self, _start_xy: object, _goal_xy: object) -> object:
            raise AssertionError("upstream backend 不得调用 GPMP wrapper.plan")

    module.TomogramPlanner = FakeTomogramPlanner
    return module


def test_upstream_backend_calls_pinned_core_and_publishes_ground_height(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    planner_module = _fake_planner_module(
        observations,
        np.asarray(
            ((1, 8, 6), (1, 12, 6), (1, 14, 6)),
            dtype=np.float64,
        ),
    )
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=planner_module,
        coordinate_module=coordinate_module,
    )

    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.30),
        goal_base_xyz=(0.9, 0.1, 0.30),
        goal_yaw=0.5,
    )

    assert observations["tomogram_stem"] == "map"
    configured_parent = (
        UPSTREAM_ROOT
        / observations["config"].wrapper.tomo_dir.lstrip("/")
    ).resolve()
    assert configured_parent == tmp_path.resolve()
    assert observations["config"].planner.use_quintic is True
    assert observations["config"].planner.max_heading_rate == pytest.approx(
        10.0
    )
    assert observations[
        "config"
    ].planner.astar_step_cost_weight == pytest.approx(0.20)
    assert observations["native_start_idx"] == (1, 6, 6)
    assert observations["native_goal_idx"] == (1, 6, 14)
    assert observations["native_optimize"] is False
    assert observations["native_reset_count"] == 1
    assert plan.points_xyz[0] == pytest.approx((0.1, 0.1, 0.0))
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))
    assert all(point[2] == pytest.approx(0.0) for point in plan.points_xyz)
    assert max(
        math.dist(start, end)
        for start, end in zip(plan.points_xyz, plan.points_xyz[1:])
    ) <= 0.20 + 1.0e-9
    assert plan.metadata["backend_kind"] == "upstream"
    assert plan.metadata["height_semantics"] == "ground_height"
    assert plan.metadata["requested_start_prepended"] is True
    assert plan.metadata["requested_goal_appended"] is False
    assert plan.metadata["transport"] == "direct_in_process_ros2"
    assert plan.metadata["upstream_output_ground_normalization"] == (
        "native_astar_tomogram_logical_layer_ground"
    )
    assert plan.metadata["upstream_core_mode"] == (
        "offline_ele_planner_native_astar_ground"
    )
    assert plan.metadata["upstream_astar_step_cost_weight"] == pytest.approx(
        0.20
    )
    assert plan.metadata["upstream_patch_id"] == UPSTREAM_PCT_PATCH_ID
    assert (
        plan.metadata["upstream_patch_sha256"]
        == UPSTREAM_PCT_PATCH_SHA256
    )
    assert plan.metadata["upstream_native_cancel_supported"] is True
    assert plan.metadata["upstream_native_gil_released"] is True
    assert plan.metadata["start_layer_height_error_m"] == pytest.approx(0.0)
    shortcut = plan.metadata["upstream_same_layer_shortcut_report"]
    assert shortcut["applied"] is True
    assert shortcut["raw_point_count"] == 4
    assert shortcut["shortcut_point_count"] == 2
    np.testing.assert_allclose(
        plan.metadata["upstream_shortcut_path_3d_pct"],
        ((0.1, 0.1, 0.0), (0.9, 0.1, 0.0)),
    )


def test_body_clearance_overlay_penalizes_body_height_obstacle_only(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    index = _UpstreamTomogramIndex(config.tomogram_path)
    vertices = np.asarray(
        (
            (0.50, 0.50, 0.35),
            (0.50, 0.60, 0.80),
            (0.50, 0.50, 0.80),
        ),
        dtype=np.float64,
    )
    overlay = _build_body_clearance_overlay(
        tomogram_index=index,
        collision_vertices=vertices,
        collision_faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        profile=None,
        sim_to_pct=lambda point: tuple(float(value) for value in point),
        minimum_height_m=0.30,
        maximum_height_m=1.00,
        radius_m=0.40,
        maximum_cost=20.0,
        power=2.0,
    )

    center = index.grid_dim_x // 2, index.grid_dim_y // 2
    assert overlay.traversability[1, center[0], center[1]] == pytest.approx(
        20.0
    )
    assert overlay.traversability[1, 0, 0] == pytest.approx(1.0)
    # 同一几何在更高逻辑层位于地面以下，不能被重复写成身体障碍。
    assert overlay.traversability[2, center[0], center[1]] == pytest.approx(
        1.0
    )
    assert overlay.report["changed_cell_count"] > 0
    assert overlay.report["gateway_source"] == "original_traversability"


def test_upstream_backend_reinitializes_overlay_with_original_gateway(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    backend = UpstreamTomogramBackend(
        _config(
            tmp_path,
            upstream_body_clearance_enabled=True,
            upstream_body_clearance_radius_m=0.40,
        ),
        planner_module=_fake_planner_module(
            observations,
            np.asarray(((1, 6, 6), (1, 6, 14)), dtype=np.float64),
        ),
        coordinate_module=coordinate_module,
    )

    initialized = observations["overlay_init"]
    np.testing.assert_array_equal(
        initialized["gateway_trav"],
        backend._tomogram_index.traversability,
    )
    assert initialized["trav"].shape == initialized["gateway_trav"].shape
    assert backend._body_clearance_report["enabled"] is True


def test_upstream_backend_preserves_no_path_and_rejects_malformed_output(
    tmp_path: Path,
) -> None:
    no_path = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_fake_planner_module({}, None),
        coordinate_module=coordinate_module,
    )
    with pytest.raises(PCTNoPathError, match="未找到路径"):
        no_path.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )

    malformed = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_fake_planner_module({}, ((0.1, 0.2), (0.3, 0.4))),
        coordinate_module=coordinate_module,
    )
    with pytest.raises(PCTBackendError, match="N×3"):
        malformed.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )


def test_upstream_backend_reloads_map_after_native_exception(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    valid_trajectory = np.asarray(
        ((1, 8, 6), (1, 12, 6), (1, 14, 6)),
        dtype=np.float64,
    )
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_stateful_fake_planner_module(
            observations,
            (RuntimeError("native state corrupted"), valid_trajectory),
        ),
        coordinate_module=coordinate_module,
    )

    with pytest.raises(PCTBackendError, match="执行失败"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )

    # 失败请求只标记旧对象；下一请求才用已重新加载地图的新对象替换。
    assert observations["planner_instances"] == 1
    assert observations["tomogram_loads"] == ["map"]

    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.3),
        goal_base_xyz=(0.9, 0.1, 0.3),
        goal_yaw=0.0,
    )

    assert observations["planner_instances"] == 2
    assert observations["tomogram_loads"] == ["map", "map"]
    assert observations["native_plan_instances"] == [0, 1]
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))


def test_upstream_backend_reuses_same_planner_after_no_path(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    valid_trajectory = np.asarray(
        ((1, 8, 6), (1, 12, 6), (1, 14, 6)),
        dtype=np.float64,
    )
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_fake_planner_module(observations, valid_trajectory),
        coordinate_module=coordinate_module,
    )
    native_planner = backend._planner.planner
    original_plan = native_planner.plan
    plan_call_count = 0

    def no_path_once(
        start_idx: object,
        goal_idx: object,
        optimize: bool,
    ) -> bool:
        nonlocal plan_call_count
        plan_call_count += 1
        if plan_call_count == 1:
            return False
        return bool(original_plan(start_idx, goal_idx, optimize))

    native_planner.plan = no_path_once
    with pytest.raises(PCTNoPathError, match="未找到路径"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )

    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.3),
        goal_base_xyz=(0.9, 0.1, 0.3),
        goal_yaw=0.0,
    )

    assert observations["planner_instances"] == 1
    assert observations["native_reset_count"] == 2
    assert backend._planner is not None
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))


def test_upstream_backend_reuses_same_planner_when_cancel_arrives_inside_native_call(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    valid_trajectory = np.asarray(
        ((1, 8, 6), (1, 12, 6), (1, 14, 6)),
        dtype=np.float64,
    )
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_stateful_fake_planner_module(
            observations,
            (valid_trajectory,),
        ),
        coordinate_module=coordinate_module,
    )
    original_native_plan = backend._planner.planner.plan

    def cancel_and_return_no_path(
        _start_idx: object,
        _goal_idx: object,
        _optimize: bool,
    ) -> bool:
        backend.cancel_current_plan()
        return False

    backend._planner.planner.plan = cancel_and_return_no_path
    with pytest.raises(PCTBackendError, match="新代际取消"):
        backend.plan(
            start_base_xyz=(0.1, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )

    assert observations["planner_instances"] == 1
    assert observations["native_cancel_instances"] == [0]
    assert observations["native_reset_instances"] == [0]
    backend._planner.planner.plan = original_native_plan
    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.3),
        goal_base_xyz=(0.9, 0.1, 0.3),
        goal_yaw=0.0,
    )

    assert observations["planner_instances"] == 1
    assert observations["tomogram_loads"] == ["map"]
    assert observations["native_reset_instances"] == [0, 0]
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))


def test_prepare_plan_reissues_cancel_arriving_inside_native_reset(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_fake_planner_module(
            observations,
            np.asarray(((1, 8, 6), (1, 14, 6)), dtype=np.float64),
        ),
        coordinate_module=coordinate_module,
    )
    reset_started = threading.Event()
    release_reset = threading.Event()
    job_cancel = threading.Event()

    def reset_with_barrier() -> None:
        observations["native_reset_count"] = int(
            observations.get("native_reset_count", 0)
        ) + 1
        reset_started.set()
        if not release_reset.wait(timeout=2.0):
            raise TimeoutError("测试没有释放 native reset barrier")
        # 精确模拟第一次 request_cancel 被稍后的 reset 清除。
        observations["native_cancelled"] = False

    backend._planner.planner.reset_cancellation = reset_with_barrier
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(backend.prepare_plan, job_cancel)
        assert reset_started.wait(timeout=1.0)
        job_cancel.set()
        backend.cancel_current_plan()
        assert observations["native_cancelled"] is True
        release_reset.set()
        with pytest.raises(PCTBackendError, match="准备 native core"):
            future.result(timeout=1.0)

    assert observations["native_cancel_count"] == 2
    assert observations["native_cancelled"] is True
    assert backend._cancel_event.is_set()
    assert backend._plan_prepared is False
    assert "native_start_idx" not in observations


def test_invalid_direct_plan_consumes_prepared_state_before_validation(
    tmp_path: Path,
) -> None:
    observations: dict[str, Any] = {}
    backend = UpstreamTomogramBackend(
        _config(tmp_path),
        planner_module=_fake_planner_module(
            observations,
            np.asarray(((1, 8, 6), (1, 14, 6)), dtype=np.float64),
        ),
        coordinate_module=coordinate_module,
    )
    backend.prepare_plan()

    with pytest.raises(ValueError, match="start_base_xyz"):
        backend.plan(
            start_base_xyz=(math.nan, 0.1, 0.3),
            goal_base_xyz=(0.9, 0.1, 0.3),
            goal_yaw=0.0,
        )

    assert backend._plan_prepared is False
    plan = backend.plan(
        start_base_xyz=(0.1, 0.1, 0.3),
        goal_base_xyz=(0.9, 0.1, 0.3),
        goal_yaw=0.0,
    )
    assert observations["planner_instances"] == 1
    assert observations["native_reset_count"] == 2
    assert plan.points_xyz[-1] == pytest.approx((0.9, 0.1, 0.0))


@pytest.mark.parametrize(
    "missing_method",
    (
        "request_cancel",
        "reset_cancellation",
        "was_cancelled",
        "get_last_search_status",
        "get_expanded_node_count",
    ),
)
def test_upstream_backend_rejects_native_without_cancellation_contract(
    tmp_path: Path,
    missing_method: str,
) -> None:
    module = _fake_planner_module({}, None)
    original_class = module.TomogramPlanner

    class MissingCancellationContractPlanner(original_class):
        def __init__(self, config: object) -> None:
            super().__init__(config)
            setattr(self.planner, missing_method, None)

    module.TomogramPlanner = MissingCancellationContractPlanner

    with pytest.raises(PCTBackendError, match=missing_method):
        UpstreamTomogramBackend(
            _config(tmp_path),
            planner_module=module,
            coordinate_module=coordinate_module,
        )


def test_endpoint_layer_uses_xy_ground_and_cost_not_linear_slice_formula(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    index = _UpstreamTomogramIndex(config.tomogram_path)

    match = index.endpoint_layer(
        point_xyz=(0.1, 0.1, 0.0),
        maximum_height_error_m=0.25,
        label="start",
    )

    # 线性旧公式会得到 2；真实 layers_g/cost 查询必须选择 layer 1。
    assert match.layer == 1
    assert match.ground_z == pytest.approx(0.0)
    assert match.cost == pytest.approx(1.0)
    with pytest.raises(PCTNoPathError, match="匹配地面层"):
        index.endpoint_layer(
            point_xyz=(0.1, 0.1, 3.0),
            maximum_height_error_m=0.25,
            label="goal",
        )


def test_same_layer_shortcut_removes_clear_grid_tie_bends(
    tmp_path: Path,
) -> None:
    index = _UpstreamTomogramIndex(_config(tmp_path).tomogram_path)

    shortcut, report = index.shortcut_same_layer_path(
        (
            (0.1, 0.3, 0.0),
            (0.1, 0.8, 0.0),
            (0.5, 0.8, 0.0),
            (0.9, 0.8, 0.0),
        ),
        layer=1,
        clearance_m=0.05,
        maximum_segment_m=2.0,
    )

    np.testing.assert_allclose(
        shortcut,
        ((0.1, 0.3, 0.0), (0.9, 0.8, 0.0)),
    )
    assert report["applied"] is True
    assert report["reason"] == "clearance_verified_line_of_sight"


def test_same_layer_shortcut_preserves_bend_around_blocked_cell_cover(
    tmp_path: Path,
) -> None:
    index = _UpstreamTomogramIndex(_config(tmp_path).tomogram_path)
    # 该阻塞单元中心位于起终点直线附近；半栅格对角线余量要求保留折角。
    index.traversability[1, 10, 10] = 50.0

    shortcut, report = index.shortcut_same_layer_path(
        (
            (0.1, 0.3, 0.0),
            (0.1, 0.55, 0.0),
            (0.1, 0.8, 0.0),
            (0.5, 0.8, 0.0),
            (0.9, 0.8, 0.0),
        ),
        layer=1,
        clearance_m=0.05,
        maximum_segment_m=2.0,
    )

    assert len(shortcut) == 3
    np.testing.assert_allclose(shortcut[0], (0.1, 0.3, 0.0))
    np.testing.assert_allclose(shortcut[-1], (0.9, 0.8, 0.0))
    direct_safe, _ = index._same_layer_segment_is_clear(
        shortcut[0], shortcut[-1], layer=1, clearance_m=0.05
    )
    assert direct_safe is False
    assert all(
        index._same_layer_segment_is_clear(
            start, end, layer=1, clearance_m=0.05
        )[0]
        for start, end in zip(shortcut, shortcut[1:])
    )
    assert report["applied"] is True
    assert float(report["minimum_blocked_cell_center_distance_m"]) >= (
        0.05 + index.resolution / math.sqrt(2.0) - 1.0e-9
    )


def test_shortcut_uses_body_obstacle_surface_cells(
    tmp_path: Path,
) -> None:
    index = _UpstreamTomogramIndex(_config(tmp_path).tomogram_path)
    overlay = np.asarray(index.traversability, dtype=np.float32).copy()
    overlay[1, 10, 10] = 20.0
    index.configure_shortcut_body_obstacles(
        overlay,
        obstacle_surface_cost=20.0,
    )

    shortcut, report = index.shortcut_same_layer_path(
        (
            (0.1, 0.3, 0.0),
            (0.1, 0.8, 0.0),
            (0.9, 0.8, 0.0),
        ),
        layer=1,
        clearance_m=0.05,
        maximum_segment_m=2.0,
    )

    assert len(shortcut) == 3
    assert report["blocked_cell_source"] == (
        "native_traversability_plus_body_obstacle_surface"
    )
    direct_safe, _ = index._same_layer_segment_is_clear(
        shortcut[0], shortcut[-1], layer=1, clearance_m=0.05
    )
    assert direct_safe is False


def test_cross_layer_shortcut_only_reduces_floor_segments(
    tmp_path: Path,
) -> None:
    index = _UpstreamTomogramIndex(_config(tmp_path).tomogram_path)
    points = (
        (0.1, 0.1, 0.0),
        (0.1, 0.4, 0.0),
        (0.1, 0.8, 0.0),
        (0.5, 0.8, 0.8),
        (0.9, 0.8, 1.6),
        (0.9, 0.4, 1.6),
        (0.9, 0.1, 1.6),
    )

    shortcut, report = index.shortcut_cross_layer_floor_segments(
        points,
        profile_start_index=2,
        profile_end_index=4,
        start_layer=1,
        goal_layer=2,
        clearance_m=0.05,
        maximum_segment_m=2.0,
    )

    np.testing.assert_allclose(
        shortcut,
        (
            points[0],
            points[2],
            points[3],
            points[4],
            points[6],
        ),
    )
    assert report["applied"] is True
    assert report["reason"] == (
        "cross_layer_floor_segments_clearance_shortcut"
    )
    assert report["profile_point_count"] == 3
    assert report["input_profile_start_index"] == 2
    assert report["input_profile_end_index"] == 4
    assert report["profile_start_index"] == 1
    assert report["profile_end_index"] == 3
    assert report["start_floor_report"]["shortcut_point_count"] == 2
    assert report["goal_floor_report"]["shortcut_point_count"] == 2


def test_cross_layer_asset_without_gateway_fails_before_native_core(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    index = _UpstreamTomogramIndex(config.tomogram_path)
    start = index.endpoint_layer(
        point_xyz=(0.1, 0.1, 0.0),
        maximum_height_error_m=0.25,
        label="start",
    )
    goal = index.endpoint_layer(
        point_xyz=(0.1, 0.1, 1.7),
        maximum_height_error_m=0.25,
        label="goal",
    )

    with pytest.raises(PCTNoPathError, match="没有向上 gateway"):
        index.validate_cross_layer_gateway(start, goal)


def test_selector_never_falls_back_when_upstream_extensions_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenCompatibleBackend:
        def __init__(self, _config: object) -> None:
            raise AssertionError("upstream 失败时不得构造 compatible backend")

    def fail_upstream_loader(_root: Path) -> ModuleType:
        raise PCTBackendError("forced upstream extension failure")

    monkeypatch.setattr(
        backend_module,
        "DirectPCTBackend",
        ForbiddenCompatibleBackend,
    )
    monkeypatch.setattr(
        upstream_backend_module,
        "_load_upstream_planner_module",
        fail_upstream_loader,
    )
    with pytest.raises(PCTBackendError, match="forced upstream"):
        create_global_planner_backend(_config(tmp_path))


def test_selector_requires_explicit_known_backend_kind(tmp_path: Path) -> None:
    config = _config(tmp_path, backend_kind="typo")
    with pytest.raises(ValueError, match="upstream 或 compatible"):
        create_global_planner_backend(config)


def test_upstream_selector_requires_one_shared_body_height(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        slice_query_root_to_floor_m=0.31,
        goal_base_to_ground_m=0.30,
    )
    with pytest.raises(ValueError, match="同一 body_height"):
        create_global_planner_backend(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"upstream_same_layer_shortcut_clearance_m": -0.01},
            "shortcut_clearance_m",
        ),
        (
            {"upstream_same_layer_shortcut_max_segment_m": 0.0},
            "shortcut_max_segment_m",
        ),
    ),
)
def test_upstream_selector_rejects_invalid_shortcut_contract(
    tmp_path: Path,
    overrides: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_global_planner_backend(_config(tmp_path, **overrides))


def test_source_manifest_matches_runtime_pin_and_license_notice() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["repository"] == UPSTREAM_PCT_REPOSITORY
    assert manifest["commit"] == UPSTREAM_PCT_COMMIT
    assert manifest["archive_sha256"] == UPSTREAM_PCT_ARCHIVE_SHA256
    assert manifest["license"] == UPSTREAM_PCT_LICENSE
    assert manifest["schema_version"] == 2
    assert manifest["source"]["repository"] == UPSTREAM_PCT_REPOSITORY
    assert manifest["source"]["commit"] == UPSTREAM_PCT_COMMIT
    assert (
        manifest["source"]["archive_sha256"]
        == UPSTREAM_PCT_ARCHIVE_SHA256
    )
    assert manifest["patches"][0]["id"] == UPSTREAM_PCT_PATCH_ID
    assert (
        manifest["patches"][0]["sha256"]
        == UPSTREAM_PCT_PATCH_SHA256
    )
    assert _validate_pinned_source(UPSTREAM_ROOT) == UPSTREAM_ROOT.resolve()


def test_source_pin_rejects_missing_or_modified_tree(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="必需文件"):
        _validate_pinned_source(tmp_path)
