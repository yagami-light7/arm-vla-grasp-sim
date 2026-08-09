from __future__ import annotations

import math

import pytest

from source.navigation.pct_adapter import pct_to_sim_xyz, sim_to_pct_xyz


def test_default_sim_to_pct_uses_180deg_xy_flip() -> None:
    pct = sim_to_pct_xyz((1.25, -2.5, 0.75))

    assert pct == pytest.approx((-1.25, 2.5, 0.75))


def test_sim_pct_roundtrip_with_offset_and_scale() -> None:
    sim = (1.25, -2.5, 0.75)
    pct = sim_to_pct_xyz(
        sim,
        pct_offset_x=10.0,
        pct_offset_y=-3.0,
        pct_scale_x=2.0,
        pct_scale_y=0.5,
    )
    restored = pct_to_sim_xyz(
        pct,
        pct_offset_x=10.0,
        pct_offset_y=-3.0,
        pct_scale_x=2.0,
        pct_scale_y=0.5,
    )

    assert restored == pytest.approx(sim)


def test_identity_roundtrip_with_offset_and_scale() -> None:
    sim = (-4.0, 3.0, 1.2)
    pct = sim_to_pct_xyz(
        sim,
        coord_mode="identity",
        pct_offset_x=-1.0,
        pct_offset_y=2.0,
        pct_scale_x=0.25,
        pct_scale_y=4.0,
    )
    restored = pct_to_sim_xyz(
        pct,
        coord_mode="identity",
        pct_offset_x=-1.0,
        pct_offset_y=2.0,
        pct_scale_x=0.25,
        pct_scale_y=4.0,
    )

    assert restored == pytest.approx(sim)


def test_fixed_xyz_rotation_order_is_rx_then_ry_then_rz() -> None:
    pct = sim_to_pct_xyz(
        (1.0, 0.0, 0.0),
        coord_mode="identity",
        pct_rotation_z_rad=math.pi / 2.0,
    )

    assert pct == pytest.approx((0.0, 1.0, 0.0), abs=1.0e-12)

    composite = sim_to_pct_xyz(
        (0.0, 1.0, 0.0),
        coord_mode="identity",
        pct_rotation_x_rad=math.pi / 2.0,
        pct_rotation_y_rad=math.pi / 2.0,
    )
    assert composite == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-12)


@pytest.mark.parametrize("coord_mode", ("identity", "sim_to_pct_180deg"))
def test_full_xyz_transform_is_reversible(coord_mode: str) -> None:
    sim = (1.25, -2.5, 0.75)
    transform = {
        "coord_mode": coord_mode,
        "pct_offset_x": 10.0,
        "pct_offset_y": -3.0,
        "pct_offset_z": 0.4,
        "pct_scale_x": 2.0,
        "pct_scale_y": -0.5,
        "pct_scale_z": 1.25,
        "pct_rotation_x_rad": 0.31,
        "pct_rotation_y_rad": -0.27,
        "pct_rotation_z_rad": 0.63,
    }

    pct = sim_to_pct_xyz(sim, **transform)
    restored = pct_to_sim_xyz(pct, **transform)

    assert restored == pytest.approx(sim, abs=1.0e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pct_scale_x", 0.0),
        ("pct_scale_y", 0.0),
        ("pct_scale_z", 0.0),
        ("pct_offset_z", math.inf),
        ("pct_rotation_x_rad", math.nan),
        ("pct_rotation_y_rad", math.inf),
        ("pct_rotation_z_rad", -math.inf),
        ("coord_mode", "unsupported"),
    ),
)
def test_invalid_or_singular_transform_fails_closed(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        sim_to_pct_xyz((1.0, 2.0, 3.0), **{field: value})


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_nonfinite_coordinate_fails_closed(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        sim_to_pct_xyz((1.0, invalid, 3.0))
