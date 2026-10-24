"""Mesh repair (pokefit/repair.py): every edge joins exactly two faces afterwards, as Bambu Studio's check wants."""
import numpy as np
import trimesh

from conftest import CLEAN, edge_stats
from pokefit.model3mf import Part
from pokefit.repair import NUDGE, repair, repair_part, weld


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    return float(trimesh.Trimesh(v, f, process=False).volume)


def test_a_closed_mesh_is_left_as_it_is() -> None:
    ball = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    part = Part("ball", 1, ball.vertices, ball.faces)
    out, info = repair_part(part)
    assert out is part and not info


def _touching_cubes() -> tuple[np.ndarray, np.ndarray]:
    a = trimesh.creation.box((10, 10, 10))
    b = a.copy()
    b.apply_translation((10, 10, 0))                     # shares the vertical edge x = y = 5 with a
    return np.vstack([a.vertices, b.vertices]), np.vstack([a.faces, b.faces + len(a.vertices)])


def test_two_cubes_touching_along_an_edge_are_moved_apart() -> None:
    """Sound by index (a boolean's result): the topology stays, the two shared corners move apart."""
    v, f = _touching_cubes()
    assert edge_stats(v, f)["welded"]["over"] >= 1      # the problem Bambu reports
    v2, f2, src, turned, info = repair(v, f)
    assert info.pinched == 2 and not info.split and not info.filled and not turned.any()
    assert (f2 == f).all() and (src == np.arange(len(f))).all()
    assert edge_stats(v2, f2) == {"index": CLEAN, "welded": CLEAN}
    assert abs(_volume(v2, f2) - 2000.0) < 0.5      # a corner moved 1 µm drags 10 mm triangles
    moved = np.linalg.norm(v2 - v, axis=1)
    assert (moved > 0).sum() == 4 and moved.max() <= NUDGE * 1.0001


def test_welded_cubes_sharing_an_edge_four_ways_are_separated() -> None:
    """Welded (one vertex per position, as an STL loads): the edge on four faces is split into two pairs."""
    v, f = weld(*_touching_cubes())
    assert edge_stats(v, f)["index"]["over"] == 1
    v2, f2, src, turned, info = repair(v, f)
    assert info.split == 1 and info.pinched == 2 and not info.filled and not turned.any()
    assert edge_stats(v2, f2) == {"index": CLEAN, "welded": CLEAN}
    assert abs(_volume(v2, f2) - 2000.0) < 0.5
    # Two separate closed boxes again: shells by shared vertices.
    m = trimesh.Trimesh(v2, f2, process=False)
    assert len(trimesh.graph.connected_components(m.face_adjacency, min_len=1)) == 2


def test_faces_turned_the_wrong_way_are_turned_back() -> None:
    ball = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    f = ball.faces.copy()
    bad = np.arange(0, len(f), 7)
    f[bad] = f[bad][:, ::-1]
    v2, f2, src, turned, info = repair(ball.vertices, f)
    assert info.flipped == len(bad) and set(np.nonzero(turned)[0]) == set(bad.tolist())
    assert edge_stats(v2, f2)["index"] == CLEAN
    assert abs(_volume(v2, f2) - ball.volume) < 1e-6


def test_an_inside_out_shell_is_turned() -> None:
    ball = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    small = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    small.apply_translation((20, 0, 0))
    v = np.vstack([ball.vertices, small.vertices])
    f = np.vstack([ball.faces, small.faces[:, ::-1] + len(ball.vertices)])   # the small ball inside out
    v2, f2, _, turned, info = repair(v, f)
    assert info.flipped == len(small.faces) and turned[len(ball.faces):].all() and not turned[:len(ball.faces)].any()
    assert abs(_volume(v2, f2) - ball.volume - small.volume) < 1e-6


def test_holes_are_closed() -> None:
    ball = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    # One missing triangle (a three-edge hole) and a missing fan round a vertex (a hole of five or six edges).
    fan = np.nonzero((ball.faces == 100).any(axis=1))[0]
    drop = np.concatenate([[0], fan])
    f = np.delete(ball.faces, drop, axis=0)
    v2, f2, src, _, info = repair(ball.vertices, f)
    assert info.filled == 2 and info.open_edges == 3 + len(fan)
    assert edge_stats(v2, f2) == {"index": CLEAN, "welded": CLEAN}
    assert (src == -1).sum() == 1 + len(fan)
    assert abs(_volume(v2, f2) - ball.volume) / ball.volume < 0.01


def test_paint_follows_its_triangle_and_turns_with_it() -> None:
    ball = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    f = ball.faces.copy()
    f[3] = f[3][::-1]
    extra = {3: 'paint_color="8"', 10: 'paint_color="C"'}
    out, info = repair_part(Part("ball", 1, ball.vertices, f, extra))
    assert info.flipped == 1
    assert out.tri_extra[10] == 'paint_color="C"'       # untouched
    assert out.tri_extra[3] == 'paint_color="8"'        # whole-triangle paint reads the same in any corner order


def test_soup_with_duplicate_vertices_only_is_left_as_it_is() -> None:
    ball = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    v = ball.vertices[ball.faces].reshape(-1, 3)        # STL-style: every triangle its own corners
    f = np.arange(len(v)).reshape(-1, 3)
    out, info = repair_part(Part("soup", 1, v, f))
    assert not info and out.vertices is v


def test_a_stray_fin_on_an_edge_is_dropped_and_its_hole_closed() -> None:
    box = trimesh.creation.box((10, 10, 10))
    e = box.edges_unique[0]
    tip = box.vertices[e].mean(axis=0) + [0, 0, 0] + 7 * np.sign(box.vertices[e].mean(axis=0) + 1e-9)
    v = np.vstack([box.vertices, tip])
    f = np.vstack([box.faces, [[e[0], e[1], len(box.vertices)]]])     # a third face on one of the box's edges
    assert edge_stats(v, f)["index"]["over"] == 1
    v2, f2, src, _, info = repair(v, f)
    assert info.split == 1 and info.removed == 1
    assert edge_stats(v2, f2) == {"index": CLEAN, "welded": CLEAN}
    assert abs(_volume(v2, f2) - 1000.0) < 1e-6


def test_a_boolean_result_touching_itself_comes_out_clean() -> None:
    """manifold3d's union of two boxes touching along an edge is sound by index, but welding it by position (as
    trimesh's `process` did, and as a slicer may) fuses the edge onto four faces: the export showed it in Bambu."""
    from pokefit.repair import repaired
    from pokefit.shell import from_manifold, to_manifold

    a = trimesh.creation.box((4, 4, 4))
    b = a.copy()
    b.apply_translation((4, 4, 0))
    m = from_manifold(to_manifold(a) + to_manifold(b))
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    stats = edge_stats(v, f)
    assert stats["index"] == CLEAN and stats["welded"]["over"] == 1
    v2, f2 = repaired(v, f)
    assert edge_stats(v2, f2) == {"index": CLEAN, "welded": CLEAN}
