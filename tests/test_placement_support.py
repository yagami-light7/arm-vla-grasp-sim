"""测试固定 place target 的 collision PLY 垂直支撑检查。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts.scene.validate_placement_support import main
from source.scene.placement_support import (
    inspect_nearest_ground_support,
    inspect_placement_support,
)


def _write_triangle_ply(path: Path) -> None:
    """写入一个地面三角面和一个顶面三角面的最小二进制 PLY。"""

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 6\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertices = (
        -1.0,
        -1.0,
        0.5,
        1.0,
        -1.0,
        0.5,
        0.0,
        1.0,
        0.5,
        -1.0,
        -1.0,
        2.0,
        1.0,
        -1.0,
        2.0,
        0.0,
        1.0,
        2.0,
    )
    faces = struct.pack("<BiiiBiii", 3, 0, 1, 2, 3, 3, 4, 5)
    path.write_bytes(header + struct.pack("<18f", *vertices) + faces)


def test_inspect_placement_support_selects_highest_surface_below_target(
    tmp_path: Path,
) -> None:
    """顶面交点不能覆盖目标下方真正的支撑面。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)

    result = inspect_placement_support(
        collision_ply,
        (0.0, 0.0, 0.56),
        minimum_clearance_m=0.01,
        maximum_clearance_m=0.10,
    )

    assert [round(item.z, 6) for item in result.intersections] == [0.5, 2.0]
    assert result.support is not None
    assert result.support.z == 0.5
    assert result.center_to_support_m == 0.06000000000000005
    assert result.geometry_verified is True


def test_inspect_placement_support_rejects_implausible_center_clearance(
    tmp_path: Path,
) -> None:
    """物体中心离支撑面过高时不能标记为已验证。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)

    result = inspect_placement_support(
        collision_ply,
        (0.0, 0.0, 0.80),
        minimum_clearance_m=0.01,
        maximum_clearance_m=0.10,
    )

    assert result.support is not None
    assert result.center_to_support_m == 0.30000000000000004
    assert result.geometry_verified is False


def test_inspect_nearest_ground_support_uses_layer_hint(tmp_path: Path) -> None:
    """多楼层查询必须选最近楼层，不能固定取最高或最低交点。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)

    lower = inspect_nearest_ground_support(
        collision_ply,
        (0.0, 0.0, 0.62),
        maximum_hint_error_m=0.20,
    )
    upper = inspect_nearest_ground_support(
        collision_ply,
        (0.0, 0.0, 1.92),
        maximum_hint_error_m=0.20,
    )

    assert lower.support.z == pytest.approx(0.5)
    assert lower.hint_error_m == pytest.approx(0.12)
    assert upper.support.z == pytest.approx(2.0)
    assert upper.hint_error_m == pytest.approx(0.08)


def test_inspect_nearest_ground_support_fails_closed(tmp_path: Path) -> None:
    """无交点、非有限提示和超限楼层提示都不能静默猜测。"""

    collision_ply = tmp_path / "collision.ply"
    _write_triangle_ply(collision_ply)

    with pytest.raises(ValueError, match="没有支撑面"):
        inspect_nearest_ground_support(collision_ply, (2.0, 2.0, 0.5))
    with pytest.raises(ValueError, match="NaN 或 Inf"):
        inspect_nearest_ground_support(collision_ply, (0.0, 0.0, float("nan")))
    with pytest.raises(ValueError, match="超过"):
        inspect_nearest_ground_support(
            collision_ply,
            (0.0, 0.0, 1.25),
            maximum_hint_error_m=0.20,
        )


def test_validate_placement_support_cli_writes_traceable_report(tmp_path: Path) -> None:
    """CLI 报告必须保留目标实例、支撑面和资产哈希。"""

    collision_ply = tmp_path / "collision.ply"
    target_json = tmp_path / "target.json"
    output_json = tmp_path / "report.json"
    _write_triangle_ply(collision_ply)
    target_json.write_text(
        json.dumps(
            {
                "target_object_id": "cola_01",
                "target_object_prim_path": "/World/cola",
                "placement_pose_world": {"position_xyz": [0.0, 0.0, 0.56]},
            }
        ),
        encoding="utf-8",
    )

    return_code = main(
        [
            "--target-json",
            str(target_json),
            "--collision-ply",
            str(collision_ply),
            "--maximum-clearance",
            "0.10",
            "--include-sha256",
            "--output-json",
            str(output_json),
        ]
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert return_code == 0
    assert report["target_object_id"] == "cola_01"
    assert report["target_object_prim_path"] == "/World/cola"
    assert report["support_surface"]["z"] == 0.5
    assert report["geometry_verified"] is True
    assert len(report["collision_ply_sha256"]) == 64
