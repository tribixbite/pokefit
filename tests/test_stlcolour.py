import numpy as np
import pytest
import trimesh

from conftest import CLEAN, edge_stats
from pokefit.colorize import PaintOp, TopologyCache, apply_paint, pick_face
from pokefit.model3mf import Part
from pokefit.stlcolour import coloured_triangles, read_coloured_stl, stl_colour_word, write_coloured_stl


def _welded(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A triangle soup welded by position, as a slicer reads an STL."""
    pts = tris.reshape(-1, 3).round(6)
    v, inv = np.unique(pts, axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def _painted_ball() -> tuple[Part, trimesh.Trimesh]:
    ball = trimesh.creation.icosphere(subdivisions=3, radius=10)
    part = Part("body", 1, np.asarray(ball.vertices, float), np.asarray(ball.faces, np.int64), {})
    pi, fi, _ = pick_face([part], np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 1.0]))
    assert pi == 0
    (out,) = apply_paint([part], [PaintOp(0, fi, 2, "brush", radius=3.0)], TopologyCache([part]))
    return out, ball


def test_colour_word_is_viscam() -> None:
    assert stl_colour_word("#FF0000") == 0x8000 | 31 << 10
    assert stl_colour_word("#00FF00") == 0x8000 | 31 << 5
    assert stl_colour_word("#0000FF") == 0x8000 | 31
    assert stl_colour_word("#000000") == 0x8000


def test_soft_paint_is_written_as_closed_sub_triangles(tmp_path) -> None:
    """A soft brush dab (split paint) comes out as its sub-triangles; the neighbours are fanned through the new
    corners on their sides, so the STL is closed (every edge on two triangles) and holds the same volume; the
    painted area is the dab's."""
    part, ball = _painted_ball()
    assert any("paint_color" in a for a in part.tri_extra.values())
    tris, ext = coloured_triangles(part, 1)
    assert len(tris) > len(part.faces) and set(ext.tolist()) == {1, 2}
    v, f = _welded(tris)
    assert edge_stats(v, f) == {"index": CLEAN, "welded": CLEAN}
    m = trimesh.Trimesh(v, f, process=False)
    assert m.volume == pytest.approx(ball.volume, rel=1e-9)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
    assert area[ext == 2].sum() == pytest.approx(np.pi * 9.0, rel=0.03)

    path, note = write_coloured_stl(tmp_path / "ball.stl", [part], 1, ["#E05028", "#161616"])
    assert note == ""
    back, hexes = read_coloured_stl(path)
    assert np.allclose(back, tris, atol=1e-5)
    assert set(hexes.tolist()) == {"#E05028", "#101010"}            # 5 bits a channel: #161616 reads #101010
    assert path.read_bytes()[:20] == b"pokefit coloured STL"
    loaded = trimesh.load(path, force="mesh")
    assert loaded.is_watertight and len(loaded.faces) == len(tris)


def test_unpainted_parts_and_modifiers(tmp_path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    a = Part("a", 0, np.asarray(box.vertices, float), np.asarray(box.faces, np.int64), {})
    mod = Part("m", 0, np.asarray(box.vertices, float) + 20, np.asarray(box.faces, np.int64), {}, "modifier_part")
    path, _ = write_coloured_stl(tmp_path / "box.stl", [a, mod], 3, ["#111111", "#222222"])
    back, hexes = read_coloured_stl(path)
    assert len(back) == 12                                          # the modifier is left out
    assert set(hexes.tolist()) == {"#808080"}                       # filament 3 is past the palette: grey


def test_parts_are_joined_into_one_solid(tmp_path) -> None:
    """A figure standing on a flat base it overlaps: the STL is their union (closed, no edge on four triangles, the
    union's volume), each triangle in its own part's colour, the painted dab's area unchanged."""
    part, ball = _painted_ball()
    base = trimesh.creation.box([30, 30, 4]).apply_translation([0, 0, -10])     # cuts into the ball's bottom
    bp = Part("flat base", 3, np.asarray(base.vertices, float), np.asarray(base.faces, np.int64), {})
    path, note = write_coloured_stl(tmp_path / "joined.stl", [part, bp], 1, ["#E05028", "#161616", "#2850E0"])
    assert note == ""
    tris, hexes = read_coloured_stl(path)
    v, f = _welded(tris)
    assert edge_stats(v, f) == {"index": CLEAN, "welded": CLEAN}
    m = trimesh.Trimesh(v, f, process=False)
    cap = np.pi * 2.0 ** 2 * (3 * 10 - 2.0) / 3                                  # ball cap below z = -8 inside the box
    assert m.volume == pytest.approx(ball.volume + base.volume - cap, rel=0.01)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
    assert area[hexes == "#101010"].sum() == pytest.approx(np.pi * 9.0, rel=0.03)
    blue = hexes == "#2850E0"
    assert blue.any() and np.allclose(tris[blue][:, :, 2].max(), -8.0, atol=1e-4)  # the base, up to its top
