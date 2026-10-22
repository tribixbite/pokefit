"""Import-time simplification (pokefit/simplify.py): shape kept, triangles cut, paint and glTF colour carried."""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

from conftest import write_inline_3mf
from pokefit.colorize import paint_code
from pokefit.model3mf import read_project
from pokefit.paint import decode_majority
from pokefit.pipeline import Params, analyse, open_session
from pokefit.simplify import budgets, clean, simplify_indices, simplify_part
from pokefit.model3mf import Part


def _sphere(sub: int = 6, r: float = 10.0) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=sub, radius=r)


def _edges(m: trimesh.Trimesh) -> tuple[int, int]:
    """(border edges, non-manifold edges) after welding by position."""
    m = trimesh.Trimesh(m.vertices, m.faces, process=False)
    m.merge_vertices()
    _, c = np.unique(np.sort(m.edges, axis=1), axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum())


def test_a_dense_sphere_keeps_its_shape_and_stays_closed() -> None:
    ball = _sphere()                                                  # 81,920 triangles
    f, rel = simplify_indices(ball.vertices, ball.faces, 5_000, error=1.0)
    f = clean(f)
    assert 4_000 <= len(f) <= 5_000
    out = trimesh.Trimesh(ball.vertices, f, process=False)
    assert _edges(out) == (0, 0)
    # Only original vertices are used, so every corner is on the sphere; the faces between them stay close.
    r = np.linalg.norm(out.triangles_center, axis=1)
    assert r.min() > 9.9 and abs(out.volume - ball.volume) / ball.volume < 0.01
    assert 0 < rel < 0.01


def test_the_error_limit_stops_it_before_the_target() -> None:
    """At the default 0.1 % of the size, a sphere cannot go down to 500 triangles: it stops where the shape would
    move further."""
    ball = _sphere()
    f, rel = simplify_indices(ball.vertices, ball.faces, 500)
    assert len(f) > 500 and rel <= 1e-3 + 1e-6


def test_folds_and_degenerate_triangles_are_removed() -> None:
    f = np.array([[0, 1, 2], [2, 1, 0], [0, 1, 3], [0, 1, 3], [4, 4, 5], [1, 2, 3]])
    assert clean(f).tolist() == [[0, 1, 3], [1, 2, 3]]   # fin pair gone, a repeat kept once, degenerate gone


def test_budgets_share_the_target_and_leave_small_parts() -> None:
    assert budgets([1_000, 100_000], 200_000) == [None, None]          # already within the target
    b = budgets([500, 60_000, 20_000, 0], 40_000)
    assert b[0] is None and b[3] is None                                # small and empty parts kept
    assert b[1] == 29_625 and b[2] == 9_875                             # (40,000 - 500) shared 3:1


def test_paint_follows_the_nearest_old_triangle_as_one_colour() -> None:
    ball = _sphere(5)
    c = ball.triangles_center
    extra = {i: f'paint_color="{paint_code(2)}"' for i in np.nonzero(c[:, 2] > 3)[0].tolist()}
    part = Part("ball", 1, ball.vertices, np.asarray(ball.faces, np.int64), extra)
    out, err = simplify_part(part, 2_000, error=1.0)
    assert len(out.faces) <= 2_000 and err > 0
    painted = np.array(sorted(out.tri_extra))
    cz = out.vertices[out.faces].mean(axis=1)[:, 2]
    assert len(painted) and cz[painted].min() > 1.5                     # the cap, give or take a triangle
    assert (cz[np.setdiff1d(np.arange(len(out.faces)), painted)] < 4.5).all()
    assert {decode_majority(a.split('"')[1]) for a in out.tri_extra.values()} == {2}


def test_a_file_loads_simplified_and_the_session_reloads_on_change(tmp_path: Path) -> None:
    stl = tmp_path / "ball.stl"
    _sphere().export(stl)
    proj = read_project(stl, 20_000)
    obj = proj.default_object()
    assert len(obj.parts[0].faces) <= 20_000 and obj.simplified.faces_before == 81_920
    s = open_session(stl, work_dir=tmp_path / "w")
    rep, _ = analyse(s, Params(magnets=0))
    assert rep.object["simplified"] is None and rep.object["parts"][0]["faces"] == 81_920
    rep2, _ = analyse(s, Params(magnets=0, simplify=10_000))
    assert rep2.object["parts"][0]["faces"] <= 10_000
    assert rep2.object["simplified"]["faces_before"] == 81_920 and rep2.object["simplified"]["error"] < 0.05
    assert abs(rep2.fit["max_scale"] - rep.fit["max_scale"]) / rep.fit["max_scale"] < 0.01
    rep3, _ = analyse(s, Params(magnets=0))                             # back to the file's own triangles
    assert rep3.object["parts"][0]["faces"] == 81_920


def test_a_dense_mesh_suggests_simplify(tmp_path: Path) -> None:
    stl = tmp_path / "dense.stl"
    _sphere(7).export(stl)                                             # 327,680 triangles
    s = open_session(stl, work_dir=tmp_path / "w")
    rep, _ = analyse(s, Params(magnets=0))
    assert any(c["id"] == "dense" for c in rep.checks)
    rep2, _ = analyse(s, Params(magnets=0, simplify=50_000))
    assert not any(c["id"] == "dense" for c in rep2.checks)


def test_a_painted_3mf_keeps_its_colours(tmp_path: Path) -> None:
    ball = _sphere(5)
    c = ball.triangles_center
    paint = {i: paint_code(3) for i in np.nonzero(c[:, 0] > 0)[0].tolist()}
    f = write_inline_3mf(tmp_path / "p.3mf", ball, np.eye(4), paint)
    part = read_project(f, 3_000).default_object().parts[0]
    assert len(part.faces) < 5_000                                      # of 20,480 (the 0.1 % limit stops it)
    cx = part.vertices[part.faces].mean(axis=1)[:, 0]
    painted = np.array(sorted(part.tri_extra))
    assert cx[painted].min() > -1.0 and (cx > 1.0).sum() <= len(painted)


def test_glb_texture_is_sampled_on_the_simplified_triangles(tmp_path: Path) -> None:
    """A textured sphere (red x < 0, blue x > 0) cut from 20,480 triangles (to ~4,100: the 0.1 % limit) keeps both
    colours, on their own sides: the UVs come with the kept vertices, and seams stay matched."""
    m = trimesh.creation.icosphere(subdivisions=5, radius=0.05)
    n = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    uv = np.stack([0.5 + 0.5 * n[:, 0], 0.5 + 0.5 * n[:, 1]], axis=1)
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = (220, 30, 30)
    img[:, 32:] = (30, 30, 220)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=Image.fromarray(img)))
    glb = tmp_path / "s.glb"
    glb.write_bytes(trimesh.Scene(m).export(file_type="glb"))
    proj = read_project(glb, 3_000)
    obj = proj.default_object()
    part = obj.parts[0]
    assert len(part.faces) < 5_000 and obj.simplified.faces_before == 20_480
    assert len(proj.colours) == 2
    red = next(i for i, h in enumerate(proj.colours) if int(h[1:3], 16) > 128) + 1
    state = np.full(len(part.faces), part.extruder)
    for i, a in part.tri_extra.items():
        state[i] = decode_majority(a.split('paint_color="')[1].split('"')[0])
    x = part.vertices[part.faces].mean(axis=1)[:, 0]
    assert (state[x < -5] == red).mean() > 0.97 and (state[x > 5] != red).mean() > 0.97
    border, nonmanifold = _edges(trimesh.Trimesh(part.vertices, part.faces, process=False))
    assert (border, nonmanifold) == (0, 0)
