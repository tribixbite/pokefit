"""The flat floor printed into the bottom shell: chosen per figure (`Session.floor`, `fit.best_floor`)."""
from pathlib import Path

import pytest
import trimesh

from conftest import BALL_3MF
from test_conform import sitting_figure
from pokefit.ball import SEAT_MARGIN, Cavity, load_ball
from pokefit.pipeline import Params, analyse, open_session
from pokefit.steel import get_steel


def _session(tmp_path: Path, mesh: trimesh.Trimesh, name: str):
    stl = tmp_path / f"{name}.stl"
    mesh.export(stl)
    return open_session(stl, BALL_3MF, None, tmp_path / f"w_{name}")


def test_a_wide_figure_gets_a_wider_floor_and_fits_larger(tmp_path: Path) -> None:
    """A sitting figure with its legs out on the thin disc's own seat (a low, narrow floor) fits at 0.61; on the
    floor chosen for it (wider, higher, where the sphere is wider) it fits much larger, without bending."""
    s = _session(tmp_path, sitting_figure(), "sit")
    seat, _ = analyse(s, Params(magnets=0, floor_diameter=0))
    auto, _ = analyse(s, Params(magnets=0))
    disc = get_steel("D20")
    assert seat.ball["floor_diameter"] == pytest.approx(disc.od + SEAT_MARGIN)
    assert auto.ball["floor_diameter"] > disc.od + SEAT_MARGIN + 2
    assert auto.fit["max_scale"] > seat.fit["max_scale"] * 1.3
    assert auto.ball["floor_z"] > seat.ball["floor_z"]          # a wider floor sits higher
    # A typed floor is used as it is; one smaller than the steel's seat is the seat.
    typed, _ = analyse(s, Params(magnets=0, floor_diameter=24))
    assert typed.ball["floor_diameter"] == pytest.approx(24)
    small, _ = analyse(s, Params(magnets=0, floor_diameter=5))
    assert small.ball["floor_diameter"] == pytest.approx(disc.od + SEAT_MARGIN)
    with pytest.raises(ValueError):
        Params(floor_diameter=80)


def test_a_tall_narrow_figure_keeps_a_low_floor(tmp_path: Path) -> None:
    """A thin column is limited by the ball's height, not its width: the lowest floor (the steel's seat) wins."""
    s = _session(tmp_path, trimesh.creation.box([4, 4, 60]).apply_translation([0, 0, 30]), "col")
    rep, _ = analyse(s, Params(magnets=0))
    assert rep.ball["floor_diameter"] == pytest.approx(get_steel("D20").od + SEAT_MARGIN, abs=0.6)


def test_the_floor_edge_is_where_the_figure_may_reach() -> None:
    """`Cavity.floor_radius` is the fit's own allowed radius at the floor."""
    import numpy as np

    c = Cavity(load_ball(BALL_3MF), 0.6, get_steel("D20"), 1.0, 26.0)
    assert c.seat_diameter == 26.0
    assert c.floor_radius == pytest.approx(float(c.allowed_radius(np.array([c.floor_z]))[0]))
