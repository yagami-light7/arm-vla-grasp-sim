from __future__ import annotations

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
