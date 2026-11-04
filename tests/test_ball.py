from pathlib import Path

import numpy as np
import pytest

from pokefit.ball import SEAT_MARGIN, Cavity, load_ball
from pokefit.shell import cap_height
from pokefit.steel import CATALOGUE, get_steel

from conftest import BALL_3MF as BALL_JSON, BALL_MODEL as BALL

needs_model = pytest.mark.skipif(not BALL.exists(), reason="third-party pokeball.3mf not present")


@pytest.fixture(scope="module")
def ball():
    return load_ball(BALL_JSON)


@needs_model
def test_bundled_measurements_match_model(tmp_path: Path, ball) -> None:
    derived = load_ball(BALL, tmp_path)
    # Same model (hash), same measurements to float rounding (numpy versions differ in the last digit).
    assert derived.source_sha256 == ball.source_sha256
    for name in ("r_inner", "r_outer", "ring_span", "aperture"):
        assert getattr(derived, name) == pytest.approx(getattr(ball, name), rel=1e-9), name
    assert np.allclose(derived.ring_h, ball.ring_h) and np.allclose(derived.ring_r, ball.ring_r)


def test_ball_geometry(ball) -> None:
    assert ball.r_inner == pytest.approx(36.806, abs=0.05)
    assert ball.r_outer == pytest.approx(38.740, abs=0.05)
    assert 15.5 <= ball.ring_span <= 17.0
    assert 28.5 <= ball.aperture <= 29.5
    assert all(r < ball.r_inner for r in ball.ring_r)


@needs_model
def test_cache_roundtrip(tmp_path: Path) -> None:
    a = load_ball(BALL, tmp_path)
    assert len(list(tmp_path.glob("ball_*.json"))) == 1
    b = load_ball(BALL, tmp_path)
    assert a == b


def test_cavity_floor_and_limits(ball) -> None:
    c = Cavity(ball, 0.6, CATALOGUE["M6"], clearance=1.0)
    ri = ball.r_inner * 0.6
    assert c.floor_z == pytest.approx(-cap_height(ri, 18 + SEAT_MARGIN) + 1.6)
    # Equator: ring-limited.
    assert c.allowed_radius(np.array([0.0]))[0] == pytest.approx(ball.ring_radius(np.array([0.0]))[0] * 0.6 - 1.0)
    # Beyond the ring span: sphere or aperture, whichever is smaller.
    z = -(ball.ring_span + 2) * 0.6
    expect = min(np.sqrt(ri ** 2 - z ** 2), ball.aperture * 0.6) - 1.0
    assert c.allowed_radius(np.array([z]))[0] == pytest.approx(expect)
    assert c.allowed_radius(np.array([c.top_z + 0.01]))[0] < 0


def test_custom_steel() -> None:
    w = get_steel("20,6,1.5")
    assert (w.od, w.id, w.thickness) == (20, 6, 1.5)
    lo, hi = w.annulus(4.18)
    assert lo == pytest.approx(3 + 2.09) and hi == pytest.approx(10 - 2.09)
    with pytest.raises(ValueError):
        get_steel("bogus")


def test_a_solid_disc_holds_magnets_anywhere_on_it_and_sits_lower() -> None:
    """The default steel is the user's thin disc (0.8" x 0.01"): no hole, so a magnet may lie at its middle; its
    floor is the seat plus 0.25 mm. Custom "OD,0,T" is a disc too."""
    d = CATALOGUE["D20"]
    assert d.solid and (d.od, d.thickness) == (pytest.approx(20.32), pytest.approx(0.254))
    assert d.annulus(4.18) == (0.0, pytest.approx(20.32 / 2 - 2.09))
    assert CATALOGUE["D15"].od == pytest.approx(14.99) and CATALOGUE["D15"].thickness == pytest.approx(0.508)
    custom = get_steel("22,0,0.3")
    assert custom.solid and "disc" in custom.label and custom.annulus(4)[0] == 0.0
    assert not get_steel("M6").solid and get_steel("M6").to_dict()["solid"] is False
    cav = Cavity(load_ball(BALL_JSON), 0.6, d, 1.0)
    assert cav.floor_z == pytest.approx(cav.seat_z + 0.254)
