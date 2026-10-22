"""Strengthen thin parts (pokefit/strengthen.py): necks thicken to the target, free features taper to their tips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pokefit.model3mf import Part
from pokefit.pipeline import Params, analyse, open_session
from pokefit.strengthen import strengthen


def _neck_figure() -> trimesh.Trimesh:
    """A body, a 3 mm tail, a 0.7 mm neck 3 mm long, then a big flame ball (a Charmander's tail at print size)."""
    body = trimesh.creation.icosphere(subdivisions=4, radius=6).apply_translation([0, 0, 6])
    tail = trimesh.creation.cylinder(radius=1.5, height=6, sections=48).apply_translation([0, 0, 14])
    stem = trimesh.creation.cylinder(radius=0.35, height=3.0, sections=48).apply_translation([0, 0, 18.0])
    flame = trimesh.creation.icosphere(subdivisions=4, radius=3).apply_translation([0, 0, 22.2])
    m = trimesh.boolean.union([body, tail, stem, flame], engine="manifold").subdivide_to_size(0.25)
    assert isinstance(m, trimesh.Trimesh)
    return m


def _fin_on_block() -> trimesh.Trimesh:
    """A 10 mm block with a thin fin (0.6 x 8 x 6 mm) standing on top: an ear with a free end."""
    block = trimesh.creation.box((10, 10, 6)).apply_translation([0, 0, 3])
    fin = trimesh.creation.box((0.6, 8, 6)).apply_translation([0, 0, 8.9])
    m = trimesh.boolean.union([block, fin], engine="manifold").subdivide_to_size(0.3)
    assert isinstance(m, trimesh.Trimesh)
    return m


def _width(m: trimesh.Trimesh, z: float, axis: int = 0) -> float:
    """Extent along `axis` of the horizontal section at z (near the middle of the other axis)."""
    sec = m.section([0, 0, 1], [0, 0, z])
    assert sec is not None
    p = np.asarray(sec.vertices)
    p = p[np.abs(p[:, 1 - axis]) < 2]
    return float(np.ptp(p[:, axis]))


def _strengthened(m: trimesh.Trimesh, target: float, extra: dict[int, str] | None = None) -> tuple[Part, Part]:
    part = Part("fig", 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), extra or {}, "normal_part")
    (out,), _ = strengthen([part], 1.0, target)
    return part, out


def test_a_neck_between_two_thick_parts_thickens_to_the_target() -> None:
    m = _neck_figure()
    part, out = _strengthened(m, 1.6, {0: 'paint_color="8"'})
    after = trimesh.Trimesh(out.vertices, out.faces, process=False)
    for z in (17.6, 18.3, 19.0):                               # (17.0 = the tail top: it rises into a fillet)
        assert _width(m, z) == pytest.approx(0.7, abs=0.02)
        assert 1.45 <= _width(after, z) <= 2.0                    # its middle too: a neck has no free end
    assert out.faces is part.faces and out.tri_extra == part.tri_extra   # vertices move only: paint carries over
    assert out.vertices[:, 2].min() >= part.vertices[:, 2].min() - 1e-9  # nothing below the bed
    # The body's far side does not move.
    low = part.vertices[:, 2] < 3
    assert np.abs(out.vertices[low] - part.vertices[low]).max() < 1e-6


def test_a_free_fin_tapers_from_its_join_to_its_tip() -> None:
    m = _fin_on_block()
    _, out = _strengthened(m, 1.6)
    after = trimesh.Trimesh(out.vertices, out.faces, process=False)
    near, mid, tip = _width(after, 7.0), _width(after, 9.5), _width(after, 11.6)
    assert near >= 1.3 and near > mid > tip > 0.6 + 0.05       # thicker than before everywhere, most at the join


def test_thick_figures_are_left_alone_and_off_means_off() -> None:
    box = trimesh.creation.box((10, 10, 10)).apply_translation([0, 0, 5]).subdivide_to_size(1.0)
    part = Part("fig", 1, np.asarray(box.vertices, float), np.asarray(box.faces, np.int64), {}, "normal_part")
    (out,), info = strengthen([part], 1.0, 1.6)
    assert info.thin_share < 0.01 and info.max_push_mm < 0.05      # sharp corners measure a little thin at most
    same, _ = strengthen([part], 1.0, 0.0)
    assert same[0] is part


def test_the_pipeline_strengthens_on_request(tmp_path: Path) -> None:
    stl = tmp_path / "neck.stl"
    _neck_figure().export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    plain = s.unit_parts(Params(magnets=0))
    strong = s.unit_parts(Params(magnets=0, strengthen=1.6))
    assert len(plain) == len(strong) and not np.array_equal(plain[0].vertices, strong[0].vertices)
    rep, _ = analyse(s, Params(magnets=0, strengthen=1.6))
    d = rep.to_dict()
    assert d["fit"]["strengthen"]["target_mm"] == 1.6
    assert any(c["id"] == "strengthen" for c in d["checks"])
    with pytest.raises(ValueError, match="strengthen"):
        Params(strengthen=6)          # 0..5 mm
