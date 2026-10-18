from pathlib import Path

import numpy as np
import pytest

from pokefit.ball import SEAT_MARGIN, Cavity, load_ball
from pokefit.shell import cap_height
from pokefit.washer import CATALOGUE, get_washer

from conftest import BALL_3MF as BALL_JSON, BALL_MODEL as BALL

needs_model = pytest.mark.skipif(not BALL.exists(), reason="third-party pokeball.3mf not present")


@pytest.fixture(scope="module")
def ball():
    return load_ball(BALL_JSON)


@needs_model
def test_bundled_measurements_match_model(tmp_path: Path, ball) -> None:
    assert load_ball(BALL, tmp_path) == ball


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


def test_custom_washer() -> None:
    w = get_washer("20,6,1.5")
    assert (w.od, w.id, w.thickness) == (20, 6, 1.5)
    lo, hi = w.annulus(4.18)
    assert lo == pytest.approx(3 + 2.09) and hi == pytest.approx(10 - 2.09)
    with pytest.raises(ValueError):
        get_washer("bogus")
