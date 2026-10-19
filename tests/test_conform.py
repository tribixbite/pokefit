"""Conform a figure to the ball's floor: bend outstretched limbs, flatten the base (pokefit/conform.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pokefit.ball import Cavity, load_ball
from pokefit.conform import FILL_NAME, ConformSpec, conform, lift_curve
from pokefit.model3mf import Part
from pokefit.pipeline import DEFAULT_BALL, Params, analyse, open_session
from pokefit.washer import get_washer

CAVITY = Cavity(load_ball(DEFAULT_BALL), 0.6, get_washer("M6"), 1.0)


def _part(mesh: trimesh.Trimesh, name: str = "fig", extra: dict[int, str] | None = None) -> Part:
    v = np.asarray(mesh.vertices, np.float64).copy()
    v[:, 2] -= v[:, 2].min()
    return Part(name, 1, v, np.asarray(mesh.faces, np.int64), extra or {}, "normal_part")


def sitting_figure() -> trimesh.Trimesh:
    """A round body with two legs lying flat out to the sides (mm, ~ a sitting Squirtle)."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=7.0)
    body.apply_translation([0, 0, 9.0])
    legs = []
    for side in (-1, 1):
        leg = trimesh.creation.box((12.0, 4.0, 4.0))
        leg = leg.subdivide().subdivide().subdivide()      # vertices along the leg so it can bend
        leg.apply_translation([side * 10.0, 0, 2.0])
        legs.append(leg)
    m = trimesh.util.concatenate([body, *legs])
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_lift_starts_flat_and_eases_in() -> None:
    r, lift = lift_curve(CAVITY, 9.0, 20.0)
    assert np.all(lift[r <= 9.0] == 0)
    slope = np.diff(lift) / np.diff(r)
    assert slope[(r[:-1] > 9.0) & (r[:-1] < 9.5)].max() < 0.2        # eased, not a kink
    assert lift[np.searchsorted(r, 14.0)] > 1.0                      # rises like the sphere beyond


def test_limbs_bend_up_and_the_figure_fits_larger() -> None:
    m = sitting_figure()
    extra = {0: 'paint_color="8"', 5: 'paint_color="C"'}
    part = _part(m, extra=extra)
    out, info = conform([part], CAVITY, ConformSpec(bend=None, flat_diameter=18.0, fill=0.0))
    assert info.auto and info.limb_share > 0.8 and info.bend > 0
    assert info.scale_after > info.scale_before * 1.02
    assert info.lift_mm > 0.5
    bent = out[0]
    # Only vertices moved: same faces in the same order, paint untouched.
    assert np.array_equal(bent.faces, part.faces) and bent.tri_extra == extra
    assert np.allclose(bent.vertices[:, :2], part.vertices[:, :2])
    tips = np.abs(part.vertices[:, 0]) > 15.0
    assert bent.vertices[tips, 2].min() > part.vertices[tips, 2].min() + 0.5   # leg tips raised


def test_a_solid_body_is_not_bent() -> None:
    box = trimesh.creation.box((30.0, 30.0, 20.0)).subdivide().subdivide()
    _, info = conform([_part(box)], CAVITY, ConformSpec(bend=None, flat_diameter=18.0))
    assert info.limb_share < 0.2 and info.bend == 0
    assert info.flat_depth_mm == 0 and info.fill_mm2 == 0           # already flat


def test_round_underside_is_pressed_and_filled_over_the_disc() -> None:
    ball = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    out, info = conform([_part(ball)], CAVITY, ConformSpec(bend=0.0, flat_diameter=18.0, max_depth=1.0, fill=2.5))
    assert 0.5 < info.flat_depth_mm <= 1.0 + 1e-6
    names = [p.name for p in out]
    assert names == ["fig", FILL_NAME]
    base = out[1].mesh()
    assert base.is_watertight and base.volume > 0
    s = info.scale_after
    centre = out[0].vertices[:, :2].mean(axis=0)
    r = np.linalg.norm(base.vertices[:, :2] - centre, axis=1) * s
    assert r.max() <= 9.0 + 0.5                                        # within the washer disc
    assert abs(base.vertices[:, 2].min()) < 1e-9                       # stands on the bed
    assert info.fill_mm2 > 50


def test_off_leaves_the_geometry_alone(tmp_path: Path) -> None:
    stl = tmp_path / "fig.stl"
    sitting_figure().export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    off = Params(bend=0, flat_base=0, magnets=0)
    assert s.conform_spec(off) is None
    plain = s.unit_parts(off)
    shaped = s.unit_parts(Params(magnets=0))
    assert len(shaped) == 2 and shaped[1].name == FILL_NAME and len(plain) == 1
    rep, _ = analyse(s, Params(magnets=0))
    assert "conform" in rep.fit and any(c["id"] == "conform" for c in rep.to_dict()["checks"])
    rep_off, _ = analyse(s, off)
    assert rep.fit["max_scale"] > rep_off.fit["max_scale"]


def test_bad_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="bend"):
        Params(bend=2)
    with pytest.raises(ValueError, match="flat_fill"):
        Params(flat_fill=-1)
