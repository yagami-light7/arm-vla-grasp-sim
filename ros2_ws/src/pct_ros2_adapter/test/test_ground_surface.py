import numpy as np
import pytest

from pct_ros2_adapter.ground_surface import TriangleGroundProjector


def _two_layer_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
            (1.0, 1.0, 3.0),
            (0.0, 1.0, 3.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)],
        dtype=np.int32,
    )
    return vertices, faces


def test_project_selects_nearest_support_in_double_layer_mesh() -> None:
    vertices, faces = _two_layer_mesh()
    projector = TriangleGroundProjector(
        vertices,
        faces,
        maximum_hint_error_m=0.40,
    )

    lower = projector.project(x=0.25, y=0.25, z_hint=0.18)
    upper = projector.project(x=0.25, y=0.25, z_hint=2.82)

    assert lower.z == pytest.approx(0.0)
    assert lower.face_index in {0, 1}
    assert lower.hint_error_m == pytest.approx(0.18)
    assert upper.z == pytest.approx(3.0)
    assert upper.face_index in {2, 3}
    assert upper.hint_error_m == pytest.approx(0.18)


def test_project_path_preserves_xy_and_reports_each_selected_layer() -> None:
    vertices, faces = _two_layer_mesh()
    projector = TriangleGroundProjector(
        vertices,
        faces,
        maximum_hint_error_m=0.30,
    )

    points, reports = projector.project_path(
        ((0.2, 0.3, 0.1), (0.8, 0.7, 2.9))
    )

    assert np.asarray(points) == pytest.approx(
        np.asarray(((0.2, 0.3, 0.0), (0.8, 0.7, 3.0)))
    )
    assert tuple(report.z for report in reports) == pytest.approx((0.0, 3.0))


def test_project_rejects_missing_surface_and_wrong_floor_hint() -> None:
    vertices, faces = _two_layer_mesh()
    projector = TriangleGroundProjector(
        vertices,
        faces,
        maximum_hint_error_m=0.20,
    )

    with pytest.raises(ValueError, match="\u6ca1\u6709\u652f\u6491\u9762"):
        projector.project(x=2.0, y=2.0, z_hint=0.0)
    with pytest.raises(ValueError, match="\u8d85\u8fc7"):
        projector.project(x=0.5, y=0.5, z_hint=1.5)


def test_project_path_error_contains_point_index_and_xyz() -> None:
    vertices, faces = _two_layer_mesh()
    projector = TriangleGroundProjector(
        vertices,
        faces,
        maximum_hint_error_m=0.20,
    )

    with pytest.raises(
        ValueError,
        match=r"points_pct_xyz\[1\]=\(0\.500000,0\.500000,1\.500000\)",
    ):
        projector.project_path(((0.2, 0.2, 0.0), (0.5, 0.5, 1.5)))


@pytest.mark.parametrize(
    ("vertices", "faces", "message"),
    [
        (
            np.asarray([(0.0, 0.0, 0.0)] * 3),
            np.asarray([[0.0, 1.0, 2.0]]),
            "face index",
        ),
        (
            np.asarray(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
            ),
            np.asarray([[0, 1, 3]], dtype=np.int32),
            "\u8d85\u51fa vertex",
        ),
        (
            np.asarray(
                [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
            ),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            "\u5782\u76f4\u6295\u5f71",
        ),
    ],
)
def test_projector_rejects_invalid_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TriangleGroundProjector(
            vertices,
            faces,
            maximum_hint_error_m=0.4,
        )
