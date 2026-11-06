"""Strengthen thin parts (pokefit/strengthen.py): necks thicken to the target, free features taper to their tips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pokefit.model3mf import Part
from pokefit.pipeline import Params, analyse, open_session, pick
from pokefit.strengthen import _marching_tets, strengthen


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
    assert after.is_watertight and after.is_winding_consistent and after.volume > m.volume
    assert 'paint_color="8"' in out.tri_extra.values()                   # paint carries over to the new triangles
    assert out.vertices[:, 2].min() >= part.vertices[:, 2].min() - 1e-6  # nothing below the bed
    # The body's far side is the file's own surface still.
    from scipy.spatial import cKDTree
    low = part.vertices[part.vertices[:, 2] < 3]
    assert cKDTree(out.vertices).query(low)[0].max() < 1e-5


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


def test_marching_tetrahedra_close_a_solid_facing_out() -> None:
    """A sphere's distance field -> a closed, outward-wound mesh of the sphere's volume."""
    h = 0.1
    ax = np.arange(-1.5, 1.5 + h / 2, h) + 0.0123
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    phi = np.sqrt(x * x + y * y + z * z) - 1.0
    n = np.array(phi.shape)
    cubes = np.flatnonzero(np.ones(n - 1, bool))
    ci, cj, ck = np.unravel_index(cubes, n - 1)
    v, f = _marching_tets(phi, (ci * n[1] + cj) * n[2] + ck, np.array([ax[0]] * 3), h)
    m = trimesh.Trimesh(v, f, process=False)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume == pytest.approx(4 / 3 * np.pi, rel=0.02)          # positive: wound outward


def _grooved_fin() -> trimesh.Trimesh:
    """A 0.7 mm fin on a block with an incised line 0.15 mm wide, 0.2 mm deep down its face (an ear's panel edge)."""
    block = trimesh.creation.box((10, 10, 6)).apply_translation([0, 0, 3])
    fin = trimesh.creation.box((0.7, 8, 6)).apply_translation([0, 0, 8.9])
    groove = trimesh.creation.box((0.4, 0.15, 5)).apply_translation([-0.35, 1.0, 9.0])
    m = trimesh.boolean.difference([trimesh.boolean.union([block, fin], engine="manifold"), groove],
                                   engine="manifold").subdivide_to_size(0.15)
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_an_incised_line_fills_instead_of_crumpling() -> None:
    """Moving vertices crossed a groove's sides and the panel beside it (scaly ears): the solid offset fills it, so
    the sheet is as thick across the line as beside it, and the result is one clean closed surface."""
    m = _grooved_fin()
    _, out = _strengthened(m, 2.0)
    after = trimesh.Trimesh(out.vertices, out.faces, process=False)
    assert after.is_watertight and after.is_winding_consistent

    def across(y: float, z: float) -> float:
        sec = after.section([0, 0, 1], [0, 0, z])
        assert sec is not None
        pts = np.asarray(sec.vertices)
        pts = pts[np.abs(pts[:, 1] - y) < 0.08]
        return float(np.ptp(pts[:, 0])) if len(pts) else 0.0

    for z in (7.5, 9.0):
        at_line, beside = across(1.0, z), across(2.5, z)
        assert at_line > 1.3 and at_line == pytest.approx(beside, abs=0.15), (z, at_line, beside)


def test_paint_names_the_file_triangles_and_carries_onto_the_thickening(tmp_path: Path) -> None:
    """Strengthen re-meshes what it thickens; taps and paint steps still name the file's triangles (`base_parts`),
    and the paint shows on the thickened shape (`figure_parts`)."""
    stl = tmp_path / "neck.stl"
    _neck_figure().export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    plain, strong = Params(magnets=0), Params(magnets=0, strengthen=1.6)
    assert np.array_equal(s.base_parts(strong)[0].faces, s.unit_parts(plain)[0].faces)
    assert len(s.unit_parts(strong)[0].faces) != len(s.unit_parts(plain)[0].faces)
    top = s.base_parts(strong)[0].vertices.max(axis=0)
    tap = pick(s, strong, [0.0, 0.0, float(top[2])], [0.0, 0.0, 1.0], mode="face")
    painted = Params(magnets=0, strengthen=1.6, colours=["#ffffff", "#ff0000"],
                     paint=[{"part": tap["part"], "face": tap["face"], "filament": 2, "mode": "part", "angle": 30}])
    fig = s.figure_parts(painted)[0]
    assert len(fig.faces) == len(s.unit_parts(strong)[0].faces)
    assert any('paint_color="8"' in a for a in fig.tri_extra.values())    # filament 2 shows on the thickened part


def _cone_on_block() -> trimesh.Trimesh:
    """A pointed ear: a cone (1.6 mm across at its base, 5 mm tall) on a block."""
    block = trimesh.creation.box((10, 10, 6)).apply_translation([0, 0, 3])
    cone = trimesh.creation.cone(radius=0.8, height=5.0, sections=48).apply_translation([0, 0, 5.9])
    m = trimesh.boolean.union([block, cone], engine="manifold").subdivide_to_size(0.2)
    assert isinstance(m, trimesh.Trimesh)
    return m


@pytest.mark.parametrize("target", [1.6, 3.0, 4.0])
def test_a_pointed_end_keeps_its_shape(target: float) -> None:
    """No ball on the end of an ear or a nose: the point stays where it was and the feature only narrows towards
    it (each section no wider than the one below), while its base still thickens."""
    m = _cone_on_block()
    _, out = _strengthened(m, target)
    after = trimesh.Trimesh(out.vertices, out.faces, process=False)
    # The point stays (up 0.26 mm at 4 mm, 0.11 at 3, 0 at 1.6: an incised line's gate is closed over one edge, which
    # reaches a point's own low cap too; `strengthen._fill_grooves`).
    assert after.vertices[:, 2].max() == pytest.approx(m.vertices[:, 2].max(), abs=0.3)
    # Above the first ~1.5 mm: this join is a sharp 90-degree corner, which reads thick there (strengthen() TODO).
    widths = [_width(after, z) for z in (7.5, 8.5, 9.5, 10.3)]
    assert all(a >= b - 0.02 for a, b in zip(widths, widths[1:])), widths
    assert widths[0] > _width(m, 7.5) + 0.2
