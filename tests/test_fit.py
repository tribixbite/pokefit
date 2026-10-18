import numpy as np
import pytest
import trimesh

from pokefit.ball import Ball, Cavity
from pokefit.fit import annulus_margin, best_offset, check_fit, max_scale, prepare_points
from pokefit.washer import Washer

THIN = Washer("t", "test", 10.0, 2.0, 0.5)


def plain_ball(r: float = 30.0) -> Ball:
    """A ball without connector rings: only the inner sphere limits."""
    return Ball(r, r + 2, [0.0, 1.0], [1e6, 1e6], 1.0, 1e6, "test")


def test_prepare_points_reduces() -> None:
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    hp = prepare_points(m.vertices)
    assert len(hp.xy) < len(m.vertices)
    assert hp.height == pytest.approx(2.0, abs=1e-6)
    assert hp.z_lo.min() == 0.0


def test_sphere_analytic_max_scale() -> None:
    cav = Cavity(plain_ball(30.0), 1.0, THIN, clearance=0.0)
    unit = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    res = max_scale(prepare_points(unit.vertices), cav)
    assert res is not None
    # A sphere standing on the floor touches the top first: diameter = R - floor.
    # Slab hull points are tested at their tighter slab end, so the solver is
    # conservative by up to ~1 % near the steep top of the cavity.
    assert res.max_scale == pytest.approx((30.0 - cav.floor_z) / 2, rel=1e-2)
    assert res.max_scale <= (30.0 - cav.floor_z) / 2
    assert res.offset == pytest.approx((0.0, 0.0), abs=0.05)


def test_offcentre_figure_gets_recentred() -> None:
    cav = Cavity(plain_ball(30.0), 1.0, THIN, clearance=1.0)
    unit = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
    unit.apply_translation([5.0, 3.0, 0.0])
    res = max_scale(prepare_points(unit.vertices), cav)
    assert res is not None
    s = res.max_scale
    assert res.offset == pytest.approx((-5 * s, -3 * s), abs=0.05 * s)
    assert res.check.margin == pytest.approx(0.0, abs=0.05)
    assert check_fit(prepare_points(unit.vertices), cav, s * 1.05, res.offset).margin < 0


def test_too_big_returns_none() -> None:
    cav = Cavity(plain_ball(10.0), 1.0, THIN, clearance=1.0)
    huge = trimesh.creation.box([10000, 10000, 1])
    assert max_scale(prepare_points(huge.vertices), cav) is None


def test_annulus_forces_offset() -> None:
    cav = Cavity(plain_ball(60.0), 1.0, THIN, clearance=1.0)
    figure = trimesh.creation.box([12, 4, 4])
    figure.apply_translation([13, 10, 2])         # footprint centre (13, 10)
    hp = prepare_points(figure.vertices)
    pockets = np.array([[10.0, 10.0], [16.0, 10.0]])
    lo, hi = 2.0, 4.0
    off, margin = best_offset(hp, cav, 1.0, pockets, (lo, hi))
    assert margin >= 0
    assert annulus_margin(pockets, 1.0, off, (lo, hi)) >= 0


def test_annulus_reduces_scale() -> None:
    cav = Cavity(plain_ball(60.0), 1.0, THIN, clearance=1.0)
    figure = trimesh.creation.box([12, 4, 4])
    hp = prepare_points(figure.vertices)
    pockets = np.array([[-4.0, 0.0], [4.0, 0.0]])
    free = max_scale(hp, cav)
    held = max_scale(hp, cav, pockets, (1.0, 4.5))
    assert free is not None and held is not None
    assert held.max_scale < free.max_scale
    assert held.max_scale * 4 <= 4.5 + 1e-6
