import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import BALL_3MF, two_footed, write_inline_3mf
from pokefit.colorize import PaintOp, TopologyCache, apply_paint, paint_code, pick_face, region_faces, topology
from pokefit.model3mf import Part, read_project
from pokefit.paint import attr_paint, decode_majority, leaves, parse_tree
from pokefit.pipeline import BUNDLED_REFERENCE, Params, analyse, export, open_session, pick, preview_faces, preview_glb
from pokefit.slicer import resize_filaments


def _part(m: trimesh.Trimesh, extra: dict[int, str] | None = None) -> Part:
    return Part("body", 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), dict(extra or {}))


def _two_boxes() -> trimesh.Trimesh:
    """Two separate cubes (two pieces, as an STL with loose eyes would have)."""
    a = trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])
    b = trimesh.creation.box([4, 4, 4]).apply_translation([20, 0, 2])
    m = trimesh.util.concatenate([a, b])
    assert isinstance(m, trimesh.Trimesh)
    return m


def _face_at(part: Part, point: list[float], normal: list[float]) -> int:
    pi, fi, d = pick_face([part], np.array(point, float), np.array(normal, float))
    assert pi == 0 and d < 1e-6
    return fi


def test_paint_code_round_trips() -> None:
    """Whole-triangle codes decode to the same filament, including 0xF continuation (18+)."""
    assert [paint_code(k) for k in (1, 2, 3, 4)] == ["4", "8", "0C", "1C"]
    for k in range(1, 40):
        assert decode_majority(paint_code(k)) == k


def test_region_piece_and_face() -> None:
    part = _part(_two_boxes())
    topo = TopologyCache([part])
    f = _face_at(part, [-5, 0, 5], [-1, 0, 0])               # big cube, -x side
    assert len(region_faces(topo, 0, f, "region", 30)) == 2    # stops at the 90° edges
    assert len(region_faces(topo, 0, f, "region", 95)) == 12   # nothing bends more than 90°
    assert len(region_faces(topo, 0, f, "piece", 30)) == 12    # the big cube, not the small one
    assert region_faces(topo, 0, f, "face", 30).tolist() == [f]


def test_part_mode_stops_at_valleys_not_ridges() -> None:
    """A block with a smaller block on top (a "tail" joined on): the joint is a concave valley, so a part
    tap on the top block paints just its 5 visible sides; every convex 90° box edge is crossed."""
    body = trimesh.creation.box([20, 20, 10]).apply_translation([0, 0, 5])
    tail = trimesh.creation.box([6, 6, 6]).apply_translation([0, 0, 12.9])
    m = trimesh.boolean.union([body, tail], engine="manifold")
    part = _part(m)
    topo = TopologyCache([part])
    f_tail = _face_at(part, [0, 0, 15.9], [0, 0, 1])
    tail_faces = region_faces(topo, 0, f_tail, "part", 15)
    cz = m.triangles_center[tail_faces][:, 2]
    assert cz.min() > 10 - 1e-6 and len(tail_faces) == len(np.nonzero(m.triangles_center[:, 2] > 10 + 1e-6)[0])
    f_body = _face_at(part, [-10, 0, 5], [-1, 0, 0])
    body_faces = region_faces(topo, 0, f_body, "part", 15)
    assert len(body_faces) + len(tail_faces) == len(m.faces)
    assert len(region_faces(topo, 0, f_body, "region", 15)) == 2   # crease rule stops at every box edge


def test_brush_paints_a_disc_of_connected_surface() -> None:
    ball = trimesh.creation.icosphere(subdivisions=4, radius=10)
    part = _part(ball)
    topo = TopologyCache([part])
    f = _face_at(part, [0, 0, 10], [0, 0, 1])
    faces = region_faces(topo, 0, f, "brush", 15, radius=3)
    c = ball.triangles_center
    d = np.linalg.norm(c[faces] - c[f], axis=1)
    assert d.max() <= 3 + 1e-9 and len(faces) == int(np.sum(np.linalg.norm(c - c[f], axis=1) <= 3))
    with pytest.raises(ValueError):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "brush"})


def _ring_stroke(part: Part, r: float, n: int) -> list[int]:
    """Faces under n points on a circle of radius r around the top of a sphere (a drawn outline)."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.c_[r * np.cos(a), r * np.sin(a), np.sqrt(100 - r * r) * np.ones_like(a)]
    return [pick_face([part], p, p / 10)[1] for p in pts]


def test_a_fill_stops_at_a_painted_outline_and_a_stroke_has_no_gaps() -> None:
    """A smooth sphere has no crease, so a part fill takes all of it; outlined by a brush stroke whose samples
    are 2.6 radii apart (a fast drag), the fill takes only the cap inside, and erasing it again works the same way."""
    ball = trimesh.creation.icosphere(subdivisions=5, radius=10)
    part = _part(ball)
    topo = TopologyCache([part])
    top = _face_at(part, [0, 0, 10], [0, 0, 1])
    assert len(region_faces(topo, 0, top, "part", 15)) == len(ball.faces)
    stroke = _ring_stroke(part, 5, 24)                     # steps of ~1.3 mm, brush radius 0.5 mm
    outline = PaintOp(0, stroke[0], 2, "brush", radius=0.5, path=tuple(stroke[1:]))
    fill = PaintOp(0, top, 3, "part")
    (out,) = apply_paint([part], [outline, fill], topo)
    states = {i: decode_majority(a.split('paint_color="')[1].split('"')[0]) for i, a in out.tri_extra.items()}
    c = ball.triangles_center
    filled = np.array([f for f, s in states.items() if s == 3])
    assert len(filled) and np.hypot(c[filled, 0], c[filled, 1]).max() < 5 and c[filled, 2].min() > 0
    assert sum(1 for s in states.values() if s == 2) > 0       # the outline keeps its own colour
    (erased,) = apply_paint([part], [outline, fill, PaintOp(0, top, 0, "part")], topo)
    # The fill is gone and the outline (its soft edge: triangles split along it) is as it was.
    held = {i: attr_paint(a).states for i, a in erased.tri_extra.items() if attr_paint(a) is not None}
    before = {i: attr_paint(a).states for i, a in out.tri_extra.items() if attr_paint(a) is not None}
    assert held and all(3 not in st for st in held.values())
    assert {i for i, st in held.items() if 2 in st} == {i for i, st in before.items() if 2 in st}
    # Without the joins a 24-point stroke leaves gaps and the fill leaks over the whole sphere.
    assert PaintOp.from_dict(outline.to_dict()) == outline
    with pytest.raises(ValueError):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "part", "path": [1, 2]})


def test_topology_merges_stl_soup() -> None:
    """An STL soup (no shared vertices) still has edge neighbours."""
    m = trimesh.creation.box([10, 10, 10])
    soup = Part("soup", 1, np.asarray(m.vertices[m.faces].reshape(-1, 3), float),
                np.arange(len(m.faces) * 3).reshape(-1, 3))
    assert len(topology(soup).pairs) == 18                     # 12 triangles × 3 edges / 2


def test_pick_skips_the_far_side_of_a_thin_plate() -> None:
    plate = trimesh.creation.box([20, 20, 0.4]).apply_translation([0, 0, 0.2])
    part = _part(plate)
    _, f, _ = pick_face([part], np.array([0.0, 0.0, 0.4]), np.array([0.0, 0.0, 1.0]))
    assert plate.face_normals[f][2] > 0.99
    _, f, _ = pick_face([part], np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    assert plate.face_normals[f][2] < -0.99


def test_apply_paint_order_erase_and_other_attributes() -> None:
    part = _part(_two_boxes(), {0: 'paint_supports="4"'})
    topo = TopologyCache([part])
    f_big = 0
    f_small = _face_at(part, [22, 0, 2], [1, 0, 0])
    ops = [PaintOp(0, f_big, 2, "piece"), PaintOp(0, f_small, 3, "piece"), PaintOp(0, f_small, 0, "face")]
    (out,) = apply_paint([part], ops, topo)
    states = {i: decode_majority(a.split('paint_color="')[1].split('"')[0])
              for i, a in out.tri_extra.items() if "paint_color" in a}
    assert sum(1 for s in states.values() if s == 2) == 12
    assert sum(1 for s in states.values() if s == 3) == 11      # 12 painted, then one erased
    assert f_small not in states
    assert 'paint_supports="4"' in out.tri_extra[0]             # other per-triangle data kept
    assert part.tri_extra == {0: 'paint_supports="4"'}          # the source is not modified


def test_paint_op_validation() -> None:
    with pytest.raises(ValueError):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "blob"})
    with pytest.raises(ValueError):
        Params(colours=["red"])
    with pytest.raises(ValueError):
        Params(paint=[{"part": 0, "face": 0, "filament": 99}])


def test_resize_filaments_extends_per_filament_arrays() -> None:
    st = {"filament_settings_id": ["PLA a", "PETG b"], "filament_type": ["PLA", "PETG"],
          "nozzle_temperature": ["220", "220", "250", "250"], "filament_colour": ["#FF0000", "#00FF00"],
          "different_settings_to_system": ["p", "fa", "fb", "pr"], "layer_height": "0.2"}
    out = resize_filaments(st, ["#FF0000", "#00FF00", "#0000FF"])
    assert out["filament_type"] == ["PLA", "PETG", "PETG"]
    assert out["nozzle_temperature"] == ["220", "220", "250", "250", "250", "250"]
    assert out["different_settings_to_system"] == ["p", "fa", "fb", "fb", "pr"]
    assert out["filament_colour"] == ["#FF0000", "#00FF00", "#0000FF"]
    assert out["layer_height"] == "0.2"


@pytest.fixture(scope="module")
def critter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_inline_3mf(tmp_path_factory.mktemp("in") / "critter.3mf", two_footed(), np.eye(4))


def test_pipeline_paint_survives_the_pocket_cut(critter: Path, tmp_path: Path) -> None:
    """Pick the body's top in the preview frame, paint its region, export: the pocketed 3MF carries it."""
    s = open_session(critter, BALL_3MF, None, tmp_path / "w")
    base = Params(washer="US1/4x1")
    top_z = float(np.vstack([p.vertices for p in s.unit_parts(base)])[:, 2].max())
    hit = pick(s, base, [0.0, 0.0, top_z], [0.0, 0.0, 1.0], "region", 30.0)
    assert hit["faces"] >= 2 and hit["distance"] < 1e-6
    params = Params(washer="US1/4x1", colours=["#E05028", "#161616"],
                    paint=[{"part": hit["part"], "face": hit["face"], "filament": 2, "mode": "region", "angle": 30}])
    rep, _ = analyse(s, params)
    assert rep.object["colours"] == ["#E05028", "#161616"]
    assert rep.object["parts"][0]["painted"] == hit["faces"]
    assert any(c["id"].startswith("cut:") for c in rep.to_dict()["checks"])
    out = export(s, params, tmp_path / "out")
    proj = read_project(out["figure_3mf"])
    (part,) = proj.objects[0].parts
    painted = [decode_majority(a.split('paint_color="')[1].split('"')[0])
               for a in part.tri_extra.values() if "paint_color" in a]
    assert painted and set(painted) == {2}
    # The painted area is the top: every painted triangle faces up.
    up = [i for i, a in part.tri_extra.items() if "paint_color" in a]
    n = trimesh.Trimesh(part.vertices, part.faces, process=False).face_normals[up]
    assert np.all(n[:, 2] > 0.99)
    glb = preview_glb(s, params, ("region", 30.0))
    assert glb[:4] == b"glTF"
    json.dumps(rep.to_dict())
    # The face colours carry "Prints from" (the slot match of every used colour), as the analysis has it, so a paint
    # step updates it without an analysis; the regions view leaves it out.
    assert preview_faces(s, params)["import_map"] == rep.to_dict()["import_map"] == []     # no reference project
    sr = open_session(critter, BALL_3MF, BUNDLED_REFERENCE, tmp_path / "w2")
    faces = preview_faces(sr, params)
    assert faces["import_map"] == analyse(sr, params)[0].to_dict()["import_map"] and len(faces["import_map"]) == 2
    assert "import_map" not in preview_faces(sr, params, ("region", 30.0))


def test_a_paint_edit_recolours_the_same_preview_and_reuses_the_shape_work(critter: Path, tmp_path: Path) -> None:
    """Colorize paint keeps the geometry: the preview's signature stays (the view recolours in place),
    the painted faces get the new colour, and the analysis (fit, pockets, checks reused from the
    unpainted run) matches a fresh session's."""
    import base64

    s = open_session(critter, BALL_3MF, None, tmp_path / "w")
    base = Params(washer="US1/4x1")
    analyse(s, base)
    before = preview_faces(s, base)
    top_z = float(np.vstack([p.vertices for p in s.unit_parts(base)])[:, 2].max())
    hit = pick(s, base, [0.0, 0.0, top_z], [0.0, 0.0, 1.0], "region", 30.0)
    params = Params(washer="US1/4x1", colours=["#E05028", "#161616"],
                    paint=[{"part": hit["part"], "face": hit["face"], "filament": 2, "mode": "region", "angle": 30}])
    after = preview_faces(s, params)
    assert after["signature"] == before["signature"] and after["colours"] == ["#E05028", "#161616"]
    idx = np.frombuffer(base64.b64decode(after["parts"][0]), np.uint8)
    assert int((idx == 1).sum()) == hit["faces"]
    rep, _ = analyse(s, params)
    fresh, _ = analyse(open_session(critter, BALL_3MF, None, tmp_path / "w2"), params)
    a, b = rep.to_dict(), fresh.to_dict()
    for d in (a, b):
        d.pop("timings", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _states(out: Part) -> dict[int, int]:
    return {i: decode_majority(a.split('paint_color="')[1].split('"')[0])
            for i, a in out.tri_extra.items() if "paint_color" in a}


def test_fill_stops_only_at_colour() -> None:
    """A box has 90° creases a part fill stops at; a fill crosses them and stops only at the painted outline."""
    box = trimesh.creation.box([20, 20, 20]).subdivide().subdivide().subdivide()
    part = _part(box)
    topo = TopologyCache([part])
    top = _face_at(part, [0, 0, 10], [0, 0, 1])
    assert len(region_faces(topo, 0, top, "fill", 15)) == len(box.faces)       # nothing painted: the whole piece
    c = box.triangles_center
    band = np.nonzero(np.abs(c[:, 2]) < 1.5)[0]                                # a belt round the middle
    ring = PaintOp(0, int(band[0]), 2, "brush", radius=1.5, path=tuple(int(f) for f in band[1:]))
    (out,) = apply_paint([part], [ring, PaintOp(0, top, 3, "fill")], topo)
    s = _states(out)
    filled = np.array([f for f, v in s.items() if v == 3])
    assert len(filled) and c[filled, 2].min() > 0                              # the upper half, all its sides
    assert {round(float(n), 3) for n in box.face_normals[filled][:, 0]} >= {-1.0, 0.0, 1.0}


def _painted_area(out: Part, state: int) -> float:
    """Area (mm²) painted `state`, split paint counted by its sub-triangles."""
    total = 0.0
    for i, a in out.tri_extra.items():
        fp = attr_paint(a)
        if fp is None:
            continue
        for tri, st in leaves(parse_tree(fp.code), list(out.vertices[out.faces[i]])):
            if st == state:
                total += float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))) / 2
    return total


def test_grow_and_shrink_move_a_soft_edge_exactly() -> None:
    """A brush dab of 3 mm on a 10 mm sphere covers a cap of pi r² (r measured straight from its centre); grown by
    1 mm it is 4 mm, shrunk 2 mm. The edge cuts through triangles, so the areas match to a few per cent."""
    ball = trimesh.creation.icosphere(subdivisions=5, radius=10)
    part = _part(ball)
    topo = TopologyCache([part])
    top = _face_at(part, [0, 0, 10], [0, 0, 1])
    for grow, r in ((0.0, 3.0), (1.0, 4.0), (-1.0, 2.0)):
        (out,) = apply_paint([part], [PaintOp(0, top, 2, "brush", radius=3.0, grow=grow)], topo)
        assert _painted_area(out, 2) == pytest.approx(np.pi * r * r, rel=0.03)
    # Growing a region found from the shape (a crease-bounded box face) moves its edge the same way.
    box = trimesh.creation.box([10, 10, 10]).subdivide_to_size(0.5)
    bp = _part(box)
    bt = TopologyCache([bp])
    f = _face_at(bp, [0, 0, 5], [0, 0, 1])
    (grown,) = apply_paint([bp], [PaintOp(0, f, 2, "region", 30, grow=1.0)], bt)
    assert _painted_area(grown, 2) == pytest.approx(100 + 4 * 10 * 1.0, rel=0.04)    # the top and 1 mm down its sides
    assert PaintOp.from_dict(PaintOp(0, 1, 2, grow=-0.5).to_dict()).grow == -0.5
    assert "grow" not in PaintOp(0, 1, 2).to_dict() and "mirror" not in PaintOp(0, 1, 2).to_dict()
    with pytest.raises(ValueError):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "grow": 99})


def _two_eared() -> trimesh.Trimesh:
    """A head with two ears and a nose, mirror-symmetric across x = 0 only (turned 0.3 rad about z, so the plane is
    not an axis of the file)."""
    head = trimesh.creation.icosphere(subdivisions=4, radius=10)
    ears = [trimesh.creation.icosphere(subdivisions=3, radius=3).apply_translation([s * 7, 0, 7]) for s in (-1, 1)]
    nose = trimesh.creation.icosphere(subdivisions=3, radius=2.5).apply_translation([0, -9.5, 0])
    m = trimesh.boolean.union([head, *ears, nose], engine="manifold")
    m.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
    return m


def test_mirror_paints_the_other_side_too() -> None:
    m = _two_eared()
    part = _part(m)
    topo = TopologyCache([part])
    plane = topo.mirror()
    expect = np.array([np.cos(0.3), np.sin(0.3), 0.0])
    assert abs(abs(plane.normal @ expect) - 1) < 2e-3 and plane.error < 0.01
    _, f, _ = pick_face([part], np.array([7 * np.cos(0.3), 7 * np.sin(0.3), 7.0]) + 3 * np.array([0, 0, 1.0]),
                        np.array([0, 0, 1.0]))
    (out,) = apply_paint([part], [PaintOp(0, f, 2, "brush", radius=1.5, mirror=True)], topo)
    c = m.triangles_center
    lr = c @ expect
    painted = np.array(sorted(_states(out)))
    assert (lr[painted] > 3).sum() > 5 and (lr[painted] < -3).sum() > 5         # both ears
    assert abs(int((lr[painted] > 0).sum()) - int((lr[painted] < 0).sum())) <= 0.25 * len(painted)
    # A lopsided shape has no plane: mirror paints one side only.
    lop = trimesh.boolean.union([trimesh.creation.box([20, 10, 10]),
                                 trimesh.creation.box([6, 6, 14]).apply_translation([7, 2, 6]),
                                 trimesh.creation.icosphere(3, 4).apply_translation([-8, -3, -4])], engine="manifold")
    lp = _part(lop)
    lt = TopologyCache([lp])
    assert lt.mirrored(0, 0) is None or lt.mirror().error <= 0.02


def test_specks_take_the_colour_round_them() -> None:
    ball = trimesh.creation.icosphere(subdivisions=5, radius=10)
    part = _part(ball)
    topo = TopologyCache([part])
    top = _face_at(part, [0, 0, 10], [0, 0, 1])
    side = _face_at(part, [10, 0, 0], [1, 0, 0])
    ops = [PaintOp(0, top, 2, "brush", radius=0.4), PaintOp(0, side, 3, "brush", radius=3.0)]
    (kept,) = apply_paint([part], ops, topo)
    assert {2, 3} <= set(_states(kept).values()) <= {0, 2, 3}           # 0: edge triangles mostly unpainted
    (clean,) = apply_paint([part], ops, topo, min_area=2.0)                  # the 0.4 mm dab is ~0.5 mm²
    assert 2 not in set(_states(clean).values()) and 3 in set(_states(clean).values())
    # A speck of the part's own colour inside paint is filled in too.
    hole = [PaintOp(0, side, 3, "brush", radius=3.0), PaintOp(0, side, 0, "face")]
    (filled,) = apply_paint([part], hole, topo, min_area=2.0)
    assert side in _states(filled)


def test_pick_reports_the_colour_under_the_tap_and_the_mirror(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path / "w")
    base = Params(washer="US1/4x1")
    top_z = float(np.vstack([p.vertices for p in s.unit_parts(base)])[:, 2].max())
    hit = pick(s, base, [0.0, 0.0, top_z], [0.0, 0.0, 1.0], "region", 30.0)
    assert hit["filament"] == 1 and hit["mirrored"] is False
    painted = Params(washer="US1/4x1", colours=["#E05028", "#161616"],
                     paint=[{"part": hit["part"], "face": hit["face"], "filament": 2, "mode": "region", "angle": 30}])
    assert pick(s, painted, [0.0, 0.0, top_z], [0.0, 0.0, 1.0], "region", 30.0)["filament"] == 2
    # A head with ears mirrors across its middle: an ear's tap has a twin, and counts both.
    head = open_session(write_inline_3mf(tmp_path / "head.3mf", _two_eared(), np.eye(4)), BALL_3MF, None,
                        tmp_path / "w2")
    v = np.vstack([p.vertices for p in head.unit_parts(base)])
    ear = np.array([7 * np.cos(0.3), 7 * np.sin(0.3), 0.0])       # the unit frame is centred on x/y like the file
    top = float(v[:, 2].max())
    one = pick(head, base, [*ear[:2].tolist(), top], [0.0, 0.0, 1.0], "brush", 15.0, 1.0)
    both = pick(head, base, [*ear[:2].tolist(), top], [0.0, 0.0, 1.0], "brush", 15.0, 1.0, mirror=True)
    assert both["mirrored"] is True and one["mirrored"] is False
    assert 1.6 * one["faces"] <= both["faces"] <= 2.4 * one["faces"]


def _dimpled_head() -> trimesh.Trimesh:
    """A 10 mm sphere with two soft oval dimples ("eyes", 0.3 mm deep, edges smoothed over ~0.4 mm) and noise like a
    sculpt's (0.01 mm), front at -y."""
    m = trimesh.creation.icosphere(subdivisions=6, radius=10)
    v = m.vertices.copy()
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    for sx in (-1, 1):
        c = np.array([sx * 3.5, -9.0, 2.0])
        c /= np.linalg.norm(c)
        # elliptical distance on the sphere (taller than wide), smoothstep edge
        d = n - c
        e = np.sqrt((d[:, 0] / 0.17) ** 2 + (d[:, 2] / 0.24) ** 2 + (d[:, 1] / 0.2) ** 2)
        depth = 0.3 * np.clip((1.2 - e) / 0.4, 0, 1) ** 2 * (3 - 2 * np.clip((1.2 - e) / 0.4, 0, 1))
        v -= n * depth[:, None]
    # A snout bump at the front: a bare sphere is mirror-symmetric front to back as well, and the mirror plane
    # would be ambiguous (a figure's face settles it).
    snout = np.exp(-np.sum((n - np.array([0.0, -0.94, -0.34])) ** 2, axis=1) / 0.05)
    v += n * (1.5 * snout)[:, None]
    v += np.random.default_rng(3).normal(0, 0.01, v.shape)
    return trimesh.Trimesh(v, m.faces, process=False)


def test_wand_finds_a_soft_eye_and_steps_through_sizes() -> None:
    m = _dimpled_head()
    part = _part(m)
    topo = TopologyCache([part])
    c = np.array([3.5, -9.0, 2.0]) / np.linalg.norm([3.5, -9.0, 2.0]) * 10
    f = pick_face([part], c, c / 10)[1]
    eye = topo.part(0).wand(f)
    cen = m.triangles_center / np.linalg.norm(m.triangles_center, axis=1, keepdims=True)
    cdir = c / 10
    d = cen - cdir
    e = np.sqrt((d[:, 0] / 0.17) ** 2 + (d[:, 2] / 0.24) ** 2 + (d[:, 1] / 0.2) ** 2)
    truth = np.nonzero(e < 1.0)[0]                       # the dimple's floor up to its mid-slope
    inter = len(np.intersect1d(eye, truth))
    assert inter / len(np.union1d(eye, truth)) > 0.6, (len(eye), len(truth), inter)
    sizes = [len(r) for _, _, r in topo.part(0).wand_candidates(f)]
    assert sizes == sorted(sizes)
    smaller = topo.part(0).wand(f, level=-1)
    assert len(smaller) <= len(eye)
    # Through the cache and paint ops: a wand op paints the eye; mirror paints the other one too.
    (out,) = apply_paint([part], [PaintOp(0, f, 2, "wand", mirror=True)], topo)
    s = _states(out)
    xs = m.triangles_center[np.array(sorted(s))][:, 0]
    assert (xs > 1).sum() > 0.3 * len(eye) and (xs < -1).sum() > 0.3 * len(eye)
    assert PaintOp.from_dict(PaintOp(0, f, 2, "wand", level=-1).to_dict()).level == -1
    with pytest.raises(ValueError):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "wand", "level": 20})


def test_wand_inside_a_painted_patch_does_not_return_the_whole_patch() -> None:
    m = _dimpled_head()
    part = _part(m)
    topo = TopologyCache([part])
    c = np.array([3.5, -9.0, 2.0]) / np.linalg.norm([3.5, -9.0, 2.0]) * 10
    f = pick_face([part], c, c / 10)[1]
    eye = topo.part(0).wand(f)
    allowed = np.zeros(topo.n_faces(0), bool)
    allowed[eye] = True
    for _, _, reg in topo.part(0).wand_candidates(f, allowed):
        assert topo.part(0).areas[reg].sum() < 0.85 * topo.part(0).areas[eye].sum()


def test_wand_takes_a_flame_off_its_tail() -> None:
    """A flame ball on a thin tail from a big body (a valley where flame and tail meet, as the Part mode sees
    it): the wand's answer is the whole flame."""
    tail = trimesh.creation.cylinder(radius=1.2, height=10, sections=48).subdivide().subdivide()
    flame = trimesh.creation.icosphere(subdivisions=4, radius=3).apply_translation([0, 0, 7.5])
    body = trimesh.creation.icosphere(subdivisions=4, radius=8).apply_translation([0, 0, -11])
    m = trimesh.boolean.union([tail, flame, body], engine="manifold")
    part = _part(m)
    topo = TopologyCache([part])
    f = pick_face([part], np.array([0, 0, 10.5]), np.array([0, 0, 1.0]))[1]
    reg = topo.part(0).wand(f)
    z = m.triangles_center[reg][:, 2]
    assert z.min() > 4.0 and len(reg) > 0.5 * (m.triangles_center[:, 2] > 5.5).sum()


def test_a_stamp_paints_an_ellipse_across_big_triangles() -> None:
    """A decal on a coarse box face (two triangles per side): the painted area is the ellipse's, pi a b, cut through
    the triangles; nothing on the sides it does not face."""
    box = trimesh.creation.box([20, 20, 20])
    part = _part(box)
    topo = TopologyCache([part])
    f = _face_at(part, [1.0, 1.0, 10.0], [0, 0, 1])
    stamp = (1.0, 1.0, 10.0, 4.0, 0.0, 0.0, 0.0, 2.5, 0.0)                    # 4 x 2.5 mm semi-axes, seen from +z
    (out,) = apply_paint([part], [PaintOp(0, f, 2, "stamp", stamp=stamp)], topo)
    assert _painted_area(out, 2) == pytest.approx(np.pi * 4.0 * 2.5, rel=0.03)
    painted = [i for i, a in out.tri_extra.items() if 2 in attr_paint(a).states]
    assert all(box.face_normals[i][2] > 0.99 for i in painted)
    op = PaintOp(0, f, 2, "stamp", stamp=stamp, grow=0.5)
    assert PaintOp.from_dict(op.to_dict()) == op
    (grown,) = apply_paint([part], [op], topo)
    assert _painted_area(grown, 2) == pytest.approx(np.pi * 4.5 * 3.0, rel=0.03)
    with pytest.raises(ValueError, match="stamp"):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "stamp"})
    with pytest.raises(ValueError, match="right angles"):
        PaintOp.from_dict({"part": 0, "face": 0, "filament": 1, "mode": "stamp", "stamp": [0, 0, 0, 1, 0, 0, 1, 1, 0]})


def test_a_soft_edge_fill_stops_at_an_earlier_soft_edge() -> None:
    """A brush dab painted soft, then the wand / bucket of the surrounding colour: the second paint fills up to the
    first one's curved edge, inside split triangles too, and never over it."""
    ball = trimesh.creation.icosphere(subdivisions=4, radius=10)
    part = _part(ball)
    topo = TopologyCache([part])
    top = _face_at(part, [0, 0, 10], [0, 0, 1])
    dab = PaintOp(0, top, 2, "brush", radius=2.5)
    (one,) = apply_paint([part], [dab], topo)
    side = _face_at(part, [10, 0, 0], [1, 0, 0])
    (two,) = apply_paint([part], [dab, PaintOp(0, side, 3, "fill")], topo)
    assert _painted_area(two, 2) == pytest.approx(_painted_area(one, 2), rel=1e-9)        # the dab untouched
    total = float(ball.area)
    assert _painted_area(two, 3) + _painted_area(two, 2) == pytest.approx(total, rel=0.002)  # the rest, all of it


def test_the_shape_map_marks_creases_and_relief() -> None:
    """A ball with a bump and a dent: the bump reads raised, the dent sunk, smooth surface level; the dent's rim
    is a crease (a valley), the smooth surface is not."""
    ball = trimesh.creation.icosphere(subdivisions=5, radius=10)
    bump = trimesh.creation.icosphere(subdivisions=3, radius=2.0).apply_translation([0, 0, 10.8])
    dent = trimesh.creation.icosphere(subdivisions=3, radius=2.0).apply_translation([10.8, 0, 0])
    m = trimesh.boolean.difference([trimesh.boolean.union([ball, bump], engine="manifold"), dent], engine="manifold")
    part = _part(m)
    t = TopologyCache([part]).part(0)
    sm = t.shape_map().astype(float)
    v = t.verts
    top = (np.hypot(v[:, 0], v[:, 1]) < 0.8) & (v[:, 2] > 11)
    pit = (np.hypot(v[:, 1], v[:, 2]) < 0.8) & (v[:, 0] > 8) & (v[:, 0] < 9.6)
    plain = np.abs(v[:, 1]) > 9.0
    assert sm[top, 1].mean() > sm[plain, 1].mean() + 25 and sm[pit, 1].mean() < sm[plain, 1].mean() - 25
    rim = (np.abs(np.hypot(v[:, 1], v[:, 2]) - 1.5) < 0.4) & (v[:, 0] > 9)
    assert sm[rim, 0].mean() > sm[plain, 0].mean() + 40


def test_the_mirror_plane_prefers_an_upright_one() -> None:
    """An egg lying on its side mirrors across a level plane and, nearly as well, an upright one: with the figure's
    up given, the upright plane is taken."""
    from pokefit.colorize import mirror_plane, surface_points

    egg = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    egg.apply_scale([10.0, 6.0, 5.0])
    bump = trimesh.creation.icosphere(subdivisions=2, radius=0.8).apply_translation([3.0, 6.0, 0.0])
    m = trimesh.boolean.union([egg, bump], engine="manifold")      # breaks the x = 0 and y = 0 planes only
    rng = np.random.default_rng(1)
    tri = m.triangles
    pts, dense = surface_points(tri, 3000, rng), surface_points(tri, 150_000, rng)
    plain = mirror_plane(pts, dense)
    assert abs(plain.normal[2]) > 0.95                                 # level (z = 0) is the best fit
    upright = mirror_plane(pts, dense, up=np.array([0.0, 0.0, 1.0]))
    assert abs(upright.normal[2]) < 0.3 and upright.error <= 1.5 * plain.error + 1e-9 or upright.normal[2] > 0.95
