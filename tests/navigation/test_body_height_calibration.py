from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from source.navigation.body_height_calibration import (
    BodyHeightCalibrationConfig,
    BodyHeightCalibrationSample,
    GroundSurfaceProjectionError,
    LiveBodyHeightCalibrator,
    _load_collision_triangle_cache,
)


def _write_collision_ply(
    path: Path,
    *,
    surface_heights: tuple[float, ...] = (0.0,),
) -> None:
    """写入覆盖 PCT (-1, -2) 的多层 binary triangle PLY。"""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for height in surface_heights:
        offset = len(vertices)
        vertices.extend(
            (
                (-2.0, -3.0, height),
                (0.0, -3.0, height),
                (-1.0, -1.0, height),
            )
        )
        faces.append((offset, offset + 1, offset + 2))
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
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in vertices:
            stream.write(struct.pack("<fff", *vertex))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


def _write_split_plane_ply(path: Path) -> None:
    """写入由两个共面三角形组成的方形支撑面。"""

    vertices = (
        (-2.0, -3.0, 0.0),
        (0.0, -3.0, 0.0),
        (0.0, -1.0, 0.0),
        (-2.0, -1.0, 0.0),
    )
    faces = ((0, 1, 2), (0, 2, 3))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 4\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in vertices:
            stream.write(struct.pack("<fff", *vertex))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *face))


@pytest.fixture
def collision_ply(tmp_path: Path) -> Path:
    path = tmp_path / "collision.ply"
    _write_collision_ply(path)
    return path


def _config(
    collision_ply: Path,
    **overrides: object,
) -> BodyHeightCalibrationConfig:
    values: dict[str, object] = {
        "collision_ply": collision_ply,
        "configured_body_height_hint_m": 0.30,
        "arm_stow_joint_positions": (0.1, -0.2, 0.3),
        "minimum_consecutive_samples": 3,
        "minimum_stable_duration_s": 0.2,
        "maximum_ground_hint_error_m": 0.10,
        "maximum_linear_speed_mps": 0.02,
        "maximum_angular_speed_rps": 0.05,
        "maximum_tilt_rad": 0.08,
        "maximum_arm_stow_error_rad": 0.04,
        "maximum_body_height_mad_m": 0.01,
        "maximum_body_height_p95_p05_m": 0.02,
    }
    values.update(overrides)
    return BodyHeightCalibrationConfig(**values)


def _sample(
    step: int,
    *,
    timestamp: float | None = None,
    write_sequence: int | None = None,
    root_z: float = 0.30,
    **overrides: object,
) -> BodyHeightCalibrationSample:
    values: dict[str, object] = {
        "step_index": step,
        "timestamp_s": step * 0.1 if timestamp is None else timestamp,
        "policy_write_sequence": step if write_sequence is None else write_sequence,
        "written_command": (0.0, 0.0, 0.0),
        "root_position_sim_xyz": (1.0, 2.0, root_z),
        "root_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "root_linear_velocity_xyz": (0.0, 0.0, 0.0),
        "root_angular_velocity_xyz": (0.0, 0.0, 0.0),
        "arm_joint_positions": (0.1, -0.2, 0.3),
        "base_lock_active": False,
        "support_joint_lock_active": False,
        "full_body_joint_lock_active": False,
        "object_follow_active": False,
    }
    values.update(overrides)
    return BodyHeightCalibrationSample(**values)


def test_live_calibration_outputs_complete_auditable_statistics(
    collision_ply: Path,
) -> None:
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))

    first = calibrator.observe(_sample(10, root_z=0.300))
    second = calibrator.observe(_sample(11, root_z=0.302))
    complete = calibrator.observe(_sample(12, root_z=0.301))

    assert first.accepted is True
    assert first.reason == "collecting"
    assert first.selected_ground_z_m == pytest.approx(0.0)
    assert second.consecutive_sample_count == 2
    assert complete.reason == "calibration_complete"
    assert complete.result is not None
    result = complete.result
    assert result.sample_count == 3
    assert result.sample_duration_s == pytest.approx(0.2)
    assert result.body_height_median_m == pytest.approx(0.301)
    assert result.body_height_mad_m == pytest.approx(0.001)
    assert result.body_height_p05_m == pytest.approx(0.3001)
    assert result.body_height_p95_m == pytest.approx(0.3019)
    assert result.body_height_p95_p05_m == pytest.approx(0.0018)
    assert result.ground_surface_z_median_m == pytest.approx(0.0)
    assert result.ground_face_index == 0
    assert result.ground_face_indices == (0,)
    assert result.first_step_index == 10
    assert result.last_step_index == 12
    assert result.first_policy_write_sequence == 10
    assert result.last_policy_write_sequence == 12
    assert result.certification_mode == "full_window"
    assert result.certification_minimum_samples == 3
    assert result.certification_minimum_duration_s == pytest.approx(0.2)
    assert result.configured_body_height_error_m == pytest.approx(0.001)
    assert result.quick_fallback_reason is None
    assert result.collision_ply_sha256 == hashlib.sha256(
        collision_ply.read_bytes()
    ).hexdigest()
    report = result.to_dict()
    assert report["height_semantics"] == "live_root_z_minus_collision_support_z"
    assert report["raw_task_z_used"] is False


def test_sim_xy_is_transformed_and_height_hint_selects_nearest_surface(
    tmp_path: Path,
) -> None:
    collision_ply = tmp_path / "stacked.ply"
    _write_collision_ply(collision_ply, surface_heights=(0.0, 3.2))
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            minimum_consecutive_samples=2,
            minimum_stable_duration_s=0.1,
        )
    )

    first = calibrator.observe(_sample(1, root_z=3.50))
    complete = calibrator.observe(_sample(2, root_z=3.50))

    assert first.selected_ground_z_m == pytest.approx(3.2)
    assert first.selected_ground_face_index == 1
    assert complete.result is not None
    assert complete.result.body_height_median_m == pytest.approx(0.30)
    assert complete.result.ground_face_index == 1
    assert "task" not in BodyHeightCalibrationSample.__dataclass_fields__


def test_read_only_projection_reports_both_frames_and_projected_base(
    tmp_path: Path,
) -> None:
    collision_ply = tmp_path / "stacked_projection.ply"
    _write_collision_ply(collision_ply, surface_heights=(0.0, 3.2))
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))

    projection = calibrator.project_ground_surface((1.0, 2.0, 3.18))

    assert projection.query_sim_xyz == pytest.approx((1.0, 2.0, 3.18))
    assert projection.query_pct_xyz == pytest.approx((-1.0, -2.0, 3.18))
    assert projection.ground_surface_pct_xyz == pytest.approx((-1.0, -2.0, 3.2))
    assert projection.ground_surface_sim_xyz == pytest.approx((1.0, 2.0, 3.2))
    assert projection.projected_base_sim_xyz == pytest.approx((1.0, 2.0, 3.5))
    assert projection.ground_face_index == 1
    assert projection.hint_error_m == pytest.approx(0.02, abs=1.0e-6)
    assert projection.to_dict()["z_hint_semantics"] == "floor_disambiguation_only"
    assert projection.to_dict()["raw_task_z_used_as_height_evidence"] is False


def test_read_only_projection_rejects_wrong_floor_hint(collision_ply: Path) -> None:
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))

    with pytest.raises(GroundSurfaceProjectionError, match="偏差超过安全门"):
        calibrator.project_ground_surface((1.0, 2.0, 2.0))


def test_coplanar_face_change_does_not_reset_stable_window(tmp_path: Path) -> None:
    collision_ply = tmp_path / "split_plane.ply"
    _write_split_plane_ply(collision_ply)
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            minimum_consecutive_samples=2,
            minimum_stable_duration_s=0.1,
        )
    )

    first = calibrator.observe(
        _sample(1, root_position_sim_xyz=(0.99, 2.01, 0.30))
    )
    complete = calibrator.observe(
        _sample(2, root_position_sim_xyz=(1.01, 1.99, 0.30))
    )

    assert first.selected_ground_face_index == 0
    assert complete.selected_ground_face_index == 1
    assert complete.result is not None
    assert complete.result.ground_face_index == 0
    assert complete.result.ground_face_indices == (0, 1)


def test_failure_after_completion_invalidates_latched_result(
    collision_ply: Path,
) -> None:
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            minimum_consecutive_samples=2,
            minimum_stable_duration_s=0.1,
        )
    )
    calibrator.observe(_sample(1))
    assert calibrator.observe(_sample(2)).result is not None

    rejected = calibrator.observe(
        _sample(3, written_command=(0.0, 0.0, 1.0e-12))
    )

    assert rejected.reason == "written_command_not_exact_zero"
    assert rejected.window_reset is True
    assert calibrator.result is None


@pytest.mark.parametrize(
    ("bad_sample", "reason"),
    (
        (_sample(10, written_command=(1.0e-15, 0.0, 0.0)), "written_command_not_exact_zero"),
        (_sample(10, base_lock_active=True), "base_lock_active_must_be_false"),
        (_sample(10, support_joint_lock_active=True), "support_joint_lock_active_must_be_false"),
        (_sample(10, full_body_joint_lock_active=True), "full_body_joint_lock_active_must_be_false"),
        (_sample(10, object_follow_active=True), "object_follow_active_must_be_false"),
        (_sample(10, root_linear_velocity_xyz=(0.021, 0.0, 0.0)), "linear_speed_exceeded"),
        (_sample(10, root_angular_velocity_xyz=(0.0, 0.0, 0.051)), "angular_speed_exceeded"),
        (
            _sample(
                10,
                root_orientation_wxyz=(math.cos(0.10), math.sin(0.10), 0.0, 0.0),
            ),
            "tilt_exceeded",
        ),
        (_sample(10, arm_joint_positions=(0.1, -0.2, 0.341)), "arm_not_stowed"),
    ),
)
def test_every_physical_gate_failure_clears_the_window(
    collision_ply: Path,
    bad_sample: BodyHeightCalibrationSample,
    reason: str,
) -> None:
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))
    assert calibrator.observe(_sample(1)).consecutive_sample_count == 1

    update = calibrator.observe(bad_sample)

    assert update.accepted is False
    assert update.reason == reason
    assert update.window_reset is True
    assert update.consecutive_sample_count == 0
    assert calibrator.result is None


@pytest.mark.parametrize(
    ("bad_sample", "reason"),
    (
        (_sample(1, timestamp=0.2, write_sequence=2), "step_index_not_strictly_increasing"),
        (_sample(2, timestamp=0.0, write_sequence=2), "timestamp_not_strictly_increasing"),
        (_sample(2, timestamp=0.2, write_sequence=1), "write_sequence_not_strictly_increasing"),
    ),
)
def test_step_time_and_policy_sequence_must_all_strictly_increase(
    collision_ply: Path,
    bad_sample: BodyHeightCalibrationSample,
    reason: str,
) -> None:
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))
    calibrator.observe(_sample(1, timestamp=0.1, write_sequence=1))

    update = calibrator.observe(bad_sample)

    assert update.reason == reason
    assert update.window_reset is True
    assert calibrator.consecutive_sample_count == 0


def test_ground_hint_mismatch_fails_closed(collision_ply: Path) -> None:
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            configured_body_height_hint_m=0.50,
            maximum_ground_hint_error_m=0.05,
        )
    )

    update = calibrator.observe(_sample(1, root_z=0.30))

    assert update.accepted is False
    assert update.reason == "ground_surface_hint_error_exceeded"
    assert update.window_reset is True


def test_excessive_height_dispersion_clears_completed_candidate(
    collision_ply: Path,
) -> None:
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            maximum_ground_hint_error_m=0.20,
            maximum_body_height_mad_m=0.001,
            maximum_body_height_p95_p05_m=0.005,
        )
    )
    calibrator.observe(_sample(1, root_z=0.30))
    calibrator.observe(_sample(2, root_z=0.34))

    update = calibrator.observe(_sample(3, root_z=0.30))

    assert update.accepted is False
    assert update.reason == "body_height_dispersion_exceeded"
    assert calibrator.consecutive_sample_count == 0
    assert calibrator.result is None


def test_strict_quick_window_completes_before_full_window(
    collision_ply: Path,
) -> None:
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            minimum_consecutive_samples=5,
            minimum_stable_duration_s=0.4,
            quick_minimum_consecutive_samples=3,
            quick_minimum_stable_duration_s=0.2,
            quick_maximum_body_height_mad_m=0.002,
            quick_maximum_body_height_p95_p05_m=0.004,
            quick_maximum_configured_height_error_m=0.01,
        )
    )

    calibrator.observe(_sample(1, root_z=0.3000))
    calibrator.observe(_sample(2, root_z=0.3002))
    completed = calibrator.observe(_sample(3, root_z=0.3001))

    assert completed.reason == "quick_calibration_complete"
    assert completed.quick_candidate_evaluated is True
    assert completed.quick_rejection_reason is None
    assert completed.result is not None
    result = completed.result
    assert result.sample_count == 3
    assert result.sample_duration_s == pytest.approx(0.2)
    assert result.certification_mode == "quick_window"
    assert result.certification_minimum_samples == 3
    assert result.certification_minimum_duration_s == pytest.approx(0.2)
    assert result.quick_fallback_reason is None


def test_quick_dispersion_failure_falls_back_without_resetting_window(
    collision_ply: Path,
) -> None:
    calibrator = LiveBodyHeightCalibrator(
        _config(
            collision_ply,
            minimum_consecutive_samples=5,
            minimum_stable_duration_s=0.4,
            quick_minimum_consecutive_samples=3,
            quick_minimum_stable_duration_s=0.2,
            quick_maximum_body_height_mad_m=0.0001,
            quick_maximum_body_height_p95_p05_m=0.0005,
            quick_maximum_configured_height_error_m=0.01,
        )
    )

    calibrator.observe(_sample(1, root_z=0.300))
    calibrator.observe(_sample(2, root_z=0.302))
    quick_rejected = calibrator.observe(_sample(3, root_z=0.300))

    assert quick_rejected.accepted is True
    assert quick_rejected.window_reset is False
    assert quick_rejected.reason == "collecting_full_window"
    assert quick_rejected.quick_candidate_evaluated is True
    assert quick_rejected.quick_rejection_reason == (
        "quick_body_height_dispersion_exceeded"
    )
    assert quick_rejected.consecutive_sample_count == 3

    fallback_continues = calibrator.observe(_sample(4, root_z=0.301))
    completed = calibrator.observe(_sample(5, root_z=0.300))

    assert fallback_continues.reason == "collecting_full_window"
    assert fallback_continues.quick_candidate_evaluated is False
    assert fallback_continues.quick_rejection_reason == (
        "quick_body_height_dispersion_exceeded"
    )
    assert completed.reason == "full_calibration_complete"
    assert completed.result is not None
    result = completed.result
    assert result.sample_count == 5
    assert result.certification_mode == "full_window"
    assert result.quick_fallback_reason == (
        "quick_body_height_dispersion_exceeded"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"quick_minimum_consecutive_samples": 2},
            "必须同时提供或同时省略",
        ),
        (
            {
                "quick_minimum_consecutive_samples": 3,
                "quick_minimum_stable_duration_s": 0.3,
                "quick_maximum_body_height_mad_m": 0.002,
                "quick_maximum_body_height_p95_p05_m": 0.004,
                "quick_maximum_configured_height_error_m": 0.01,
            },
            "不能超过完整窗口",
        ),
        (
            {
                "quick_minimum_consecutive_samples": 3,
                "quick_minimum_stable_duration_s": 0.2,
                "quick_maximum_body_height_mad_m": 0.02,
                "quick_maximum_body_height_p95_p05_m": 0.03,
                "quick_maximum_configured_height_error_m": 0.01,
            },
            "离散度门不能宽于",
        ),
    ),
)
def test_quick_window_configuration_must_be_complete_and_stricter(
    collision_ply: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(collision_ply, **overrides)


def test_collision_triangles_are_cached_by_asset_fingerprint(
    collision_ply: Path,
) -> None:
    _load_collision_triangle_cache.cache_clear()
    config = _config(collision_ply)

    first = LiveBodyHeightCalibrator(config)
    second = LiveBodyHeightCalibrator(config)

    assert first._mesh is second._mesh
    assert _load_collision_triangle_cache.cache_info().misses == 1
    assert _load_collision_triangle_cache.cache_info().hits == 1


def test_malformed_runtime_values_fail_closed_and_reset(collision_ply: Path) -> None:
    calibrator = LiveBodyHeightCalibrator(_config(collision_ply))
    calibrator.observe(_sample(1))
    malformed = replace(
        _sample(2),
        root_position_sim_xyz=(1.0, 2.0, float("nan")),
    )

    update = calibrator.observe(malformed)

    assert update.reason == "root_position_invalid"
    assert update.window_reset is True
    assert calibrator.consecutive_sample_count == 0
