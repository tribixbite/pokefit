"""Conform a figure to the ball's floor: bend outstretched limbs, flatten the base (pokefit/conform.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import CLEAN, edge_stats
from pokefit.ball import Cavity, load_ball
from pokefit.conform import FILL_NAME, ConformSpec, conform, lift_curve
from pokefit.model3mf import Part
from pokefit.pipeline import DEFAULT_BALL, Params, analyse, open_session, pick, preview_faces
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


def test_default_fill_is_raised_when_no_magnet_fits(tmp_path: Path) -> None:
    # A body on four thin legs (an Eevee): the legs are too thin for a walled pocket and the belly is ~2.8 mm
    # up at the fit size, so with the default 2.5 mm fill no magnet fits; the default is raised to 4 mm.
    parts = [trimesh.creation.box((14, 9, 10)).apply_translation([0, 0, 7.0])]
    for x in (-5.5, 5.5):
        for y in (-3, 3):
            parts.append(trimesh.creation.box((2.4, 2.4, 2.5)).apply_translation([x, y, 1.25]))
    stl = tmp_path / "legs.stl"
    trimesh.util.concatenate(parts).export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    rep, _ = analyse(s, Params())
    d = rep.to_dict()
    assert d["fit"]["conform"]["fill_mm"] == 4.0 and len(d["pockets"]) == 2
    assert "with less, no magnet pocket fits" in next(c["message"] for c in d["checks"] if c["id"] == "conform")
    # a fill the user chose is kept, even when no pocket fits with it
    rep2, _ = analyse(s, Params(flat_fill=2.4))
    d2 = rep2.to_dict()
    assert d2["fit"]["conform"]["fill_mm"] == 2.4 and not d2["pockets"]
    # no magnets asked for: nothing to make room for
    rep3, _ = analyse(s, Params(magnets=0))
    assert rep3.to_dict()["fit"]["conform"]["fill_mm"] == 2.5


def test_the_flat_base_cannot_be_painted_and_shows_as_its_own_region(tmp_path: Path) -> None:
    """The conform step's flat base is not one of the file's parts: a tap on it says so (it used to
    fail with an index error), and "show regions" greys it out instead of failing too."""
    stl = tmp_path / "fig.stl"
    sitting_figure().export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    p = Params(magnets=0)
    parts = s.unit_parts(p)
    assert parts[-1].name == FILL_NAME
    fb = parts[-1]
    tri = fb.vertices[fb.faces[np.argmin(fb.vertices[fb.faces][:, :, 2].max(axis=1))]]
    with pytest.raises(ValueError, match="flat base"):
        pick(s, p, tri.mean(axis=0).tolist(), [0.0, 0.0, -1.0], "part", 15.0)
    faces = preview_faces(s, p, ("part", 15.0))
    import base64
    base = np.frombuffer(base64.b64decode(faces["parts"][-1]), np.uint8)
    assert len(base) == len(fb.faces) and set(base.tolist()) == {len(faces["colours"]) - 1}


def _four_legs(tmp_path: Path) -> Path:
    """A block body on four thin legs (an Eevee): 2.5 mm of air under the belly between the legs."""
    parts = [trimesh.creation.box((14, 9, 10)).apply_translation([0, 0, 7.0])]
    for x in (-5.5, 5.5):
        for y in (-3, 3):
            parts.append(trimesh.creation.box((2.4, 2.4, 2.5)).apply_translation([x, y, 1.25]))
    stl = tmp_path / "legs.stl"
    trimesh.util.concatenate(parts).export(stl)
    return stl


def test_a_fill_circle_fills_only_inside_it_wherever_it_is_placed(tmp_path: Path) -> None:
    s = open_session(_four_legs(tmp_path), work_dir=tmp_path / "w")
    auto = [p for p in s.unit_parts(Params(magnets=0, flat_fill=8)) if p.name == FILL_NAME][0]
    assert np.ptp(auto.vertices[:, 0]) > 10                         # round the washer: out to the legs
    for cx in (0.0, 2.0):
        params = Params(magnets=0, flat_fill=8, fill_circle={"x": cx, "y": 0.0, "d": 6.0})
        base = [p for p in s.unit_parts(params) if p.name == FILL_NAME][0]
        v = base.vertices
        assert 4.5 < np.ptp(v[:, 0]) <= 6.2 and 4.5 < np.ptp(v[:, 1]) <= 6.2       # the circle, not the legs
        assert abs((v[:, 0].min() + v[:, 0].max()) / 2 - cx) < 0.5
        rep, _ = analyse(s, params)
        assert rep.to_dict()["fit"]["conform"]["fill_mm2"] > 0
    with pytest.raises(ValueError, match="fill_circle"):
        Params(fill_circle={"x": 0, "y": 0, "d": 0})
    with pytest.raises(ValueError, match="fill_circle"):
        Params(fill_circle={"x": 0})


def test_a_height_asked_for_is_printed_unless_it_does_not_fit(tmp_path: Path) -> None:
    s = open_session(_four_legs(tmp_path), work_dir=tmp_path / "w")
    largest = analyse(s, Params())[0].to_dict()["fit"]["height_mm"]
    rep = analyse(s, Params(figure_height=round(largest * 0.8, 1)))[0].to_dict()
    assert rep["fit"]["height_mm"] == pytest.approx(round(largest * 0.8, 1), abs=0.01)
    assert not any(c["id"] == "size:height" for c in rep["checks"])
    big = analyse(s, Params(figure_height=largest * 3))[0].to_dict()
    assert big["fit"]["height_mm"] == pytest.approx(largest, abs=0.01) and big["status"] != "fail"
    assert any(c["id"] == "size:height" for c in big["checks"])
    with pytest.raises(ValueError, match="figure_height"):
        Params(figure_height=0)


def test_magnets_on_the_washer_get_pads_so_they_are_walled_from_the_washer_up(tmp_path: Path) -> None:
    """2.5 mm of air under the belly and a tiny fill circle: the pockets get pads filling that gap round them."""
    s = open_session(_four_legs(tmp_path), work_dir=tmp_path / "w")
    params = Params(magnets=2, flat_fill=4, fill_circle={"x": 0.0, "y": 0.0, "d": 1.0})
    rep, final = analyse(s, params)
    d = rep.to_dict()
    pocket_checks = [c for c in d["checks"] if c["id"].startswith("pocket:")]
    assert len(d["pockets"]) == 2 and all(p["z"] == 0.0 for p in d["pockets"])
    assert all(c["status"] == "pass" and "Sits on the washer" in c["message"] for c in pocket_checks)
    assert not any("glue" in c["message"] for c in pocket_checks)
    pad = s.pads(params)
    assert pad is not None and any(p.name == FILL_NAME for p in s.figure_parts(params))
    scale = d["fit"]["scale"]
    for p in d["pockets"]:                       # the pad lies under each pocket, from the bed up
        near = np.hypot(pad.vertices[:, 0] * scale - p["x"], pad.vertices[:, 1] * scale - p["y"]) < 3.5
        assert near.any() and pad.vertices[near, 2].min() == pytest.approx(0.0, abs=1e-6)
    # The raised seat puts the pockets up into the figure instead: no pads.
    raised = Params(magnets=2, flat_fill=4, fill_circle={"x": 0.0, "y": 0.0, "d": 1.0}, pocket_seat="raised")
    assert s.pads(raised) is None
    assert all(p["z"] > 0 for p in analyse(s, raised)[0].to_dict()["pockets"])


def test_exported_parts_have_no_non_manifold_edges(tmp_path: Path) -> None:
    """Pockets cut beside a pressed base, pads and a drawn fill (grid cells meeting at corners): every exported
    part's edges join exactly two faces, also after welding by position (Bambu Studio's check)."""
    s = open_session(_four_legs(tmp_path), work_dir=tmp_path / "w")
    for params in (Params(magnets=2, flat_fill=4, fill_circle={"x": 0.0, "y": 0.0, "d": 1.0}),
                   Params(magnets=2, flat_fill=3, fill_area="shape",
                          fill_shape=[[-5.0, -3.5], [5.0, -3.2], [0.4, 3.5]])):
        _, final = analyse(s, params)
        for p in final:
            assert edge_stats(p.vertices, p.faces) == {"index": CLEAN, "welded": CLEAN}, p.name


def test_fill_everywhere_and_a_drawn_shape(tmp_path: Path) -> None:
    s = open_session(_four_legs(tmp_path), work_dir=tmp_path / "w")
    every = [p for p in s.unit_parts(Params(magnets=0, flat_fill=3, fill_area="all")) if p.name == FILL_NAME][0]
    assert np.ptp(every.vertices[:, 0]) > 13 and np.ptp(every.vertices[:, 1]) > 8      # under the whole belly
    assert [p for p in s.unit_parts(Params(magnets=0, flat_fill=2, fill_area="all")) if p.name == FILL_NAME] == []
    tri = [[-5.0, -3.5], [5.0, -3.5], [0.0, 3.5]]
    shaped = [p for p in s.unit_parts(Params(magnets=0, flat_fill=3, fill_area="shape", fill_shape=tri))
              if p.name == FILL_NAME][0]
    v = shaped.vertices
    assert 8 < np.ptp(v[:, 0]) <= 10.2 and 5.5 < np.ptp(v[:, 1]) <= 7.2
    top = v[v[:, 1] > 2.0]                          # narrow near the triangle's apex
    assert len(top) and np.ptp(top[:, 0]) < 3.5
    with pytest.raises(ValueError, match="fill_area"):
        Params(fill_area="moon")
    with pytest.raises(ValueError, match="fill_shape"):
        Params(fill_area="shape")
    with pytest.raises(ValueError, match="fill_shape"):
        Params(fill_area="shape", fill_shape=[[0, 0], [1, 1]])
    assert Params(fill_circle={"x": 0, "y": 0, "d": 4}).fill_area == "circle"   # older clients
