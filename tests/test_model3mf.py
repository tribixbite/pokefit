import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import REAL_DRAGON, write_inline_3mf
from pokefit.model3mf import Part, read_project, remesh_part, world_parts, write_project
from pokefit.paint import encode_tree, leaves, parse_tree


def _rot_scale(deg: float, s: float, t: tuple[float, float, float]) -> np.ndarray:
    m = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])
    m[:3, :3] *= s
    m[:3, 3] = t
    return m


def test_build_transform_baked(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 20, 30])
    xf = _rot_scale(90, 0.5, (100, 50, 7.5))
    proj = read_project(write_inline_3mf(tmp_path / "a.3mf", box, xf))
    (part,) = world_parts(proj.default_object())
    ext = np.ptp(part.vertices, axis=0)
    assert ext == pytest.approx([10, 5, 15], abs=1e-6)       # rotated 90°, half size
    assert part.vertices.mean(axis=0) == pytest.approx([100, 50, 7.5], abs=1e-6)
    assert part.mesh().volume == pytest.approx(10 * 20 * 30 * 0.125)


def test_mirror_keeps_outward_normals(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    xf = np.diag([-1.0, 1, 1, 1])
    proj = read_project(write_inline_3mf(tmp_path / "m.3mf", box, xf))
    (part,) = world_parts(proj.default_object())
    assert part.mesh().volume > 0


def test_paint_roundtrip(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    paint = {0: "8", 5: "0C", 11: "481"}
    proj = read_project(write_inline_3mf(tmp_path / "p.3mf", box, np.eye(4), paint))
    obj = proj.default_object()
    assert {k: v for k, v in obj.parts[0].tri_extra.items()} == {k: f'paint_color="{v}"' for k, v in paint.items()}
    out = write_project(proj, "box", 1, obj.parts, tmp_path / "out.3mf", settings={"printable_area": ["0x0", "200x0", "200x200", "0x200"]})
    back = read_project(out).default_object()
    assert back.parts[0].tri_extra == obj.parts[0].tri_extra
    assert back.transform[:3, 3] == pytest.approx([100, 100, 0])
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("Metadata/project_settings.config"))["printable_area"][1] == "200x0"


def test_paint_roundtrip_after_cut() -> None:
    box = trimesh.creation.box([20, 20, 10])
    box.apply_translation([0, 0, 5])
    painted = {i: f'paint_color="{"8" if i % 2 else "4"}"' for i in range(len(box.faces))}
    part = Part("b", 1, np.asarray(box.vertices), np.asarray(box.faces), painted)
    hole = trimesh.creation.cylinder(radius=2, height=4, sections=32)  # centred at z=0: cuts the bottom
    cut = trimesh.boolean.difference([box, hole], engine="manifold")
    new = remesh_part(part, cut)
    assert len(new.faces) > len(box.faces)
    # Every new triangle carries a paint attribute (all originals were painted).
    assert len(new.tri_extra) == len(new.faces)
    # Top face triangles are untouched by the cut: exact attribute match.
    top_old = {tuple(sorted(map(tuple, box.vertices[f].round(5)))): painted[i]
               for i, f in enumerate(box.faces) if box.vertices[f][:, 2].min() > 9.99}
    for i, f in enumerate(new.faces):
        key = tuple(sorted(map(tuple, new.vertices[f].round(5))))
        if key in top_old:
            assert new.tri_extra[i] == top_old[key]


def test_multiple_instances(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    proj = read_project(write_inline_3mf(tmp_path / "b.3mf", box, np.eye(4)))
    out = write_project(proj, "box", 1, proj.default_object().parts, tmp_path / "x.3mf",
                        positions=[(20, 20), (40, 20), (60, 20)])
    with zipfile.ZipFile(out) as z:
        root = z.read("3D/3dmodel.model").decode()
        cfg = z.read("Metadata/model_settings.config").decode()
    assert root.count("<item ") == 3
    assert cfg.count("<model_instance>") == 3


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DRAGON.exists(), reason="real dragon 3MF not present")
def test_real_dragon() -> None:
    proj = read_project(REAL_DRAGON)
    obj = proj.default_object()
    assert obj.name == "Dragon ornaments.stl"
    assert [p.extruder for p in obj.parts] == [2, 2, 1]
    v = np.vstack([p.vertices for p in world_parts(obj)])
    # Native ~80 x 80 x 120 mm placed at build scale 0.3.
    assert np.ptp(v[:, 2]) == pytest.approx(36.0, abs=0.1)
    assert sum(len(p.tri_extra) for p in obj.parts) == 1306


def test_stl_is_merged(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    f = tmp_path / "box.stl"
    box.export(f)
    part = read_project(f).default_object().parts[0]
    assert len(part.vertices) == 8


def _paint_areas(part: Part) -> set[tuple[frozenset, int]]:
    """Every painted leaf of every triangle as (corner positions, state): independent of corner order."""
    out = set()
    for i, attrs in part.tri_extra.items():
        code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
        corners = [part.vertices[j] for j in part.faces[i]]
        for v, st in leaves(parse_tree(code), corners):
            out.add((frozenset(tuple(np.round(c, 6)) for c in v), st))
    return out


def test_split_paint_follows_corner_order() -> None:
    # One triangle with a 3-way split whose children differ, plus a 1-way split one.
    v = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0], [8, 8, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    painted = Part("p", 1, v, f, {0: f'paint_color="{encode_tree((3, 0, [1, 2, 3, 4]))}"',
                                  1: f'paint_color="{encode_tree((1, 1, [2, (3, 0, [1, 1, 4, 3])]))}"'})
    before = _paint_areas(painted)
    rolled = trimesh.Trimesh(v, np.array([[1, 2, 0], [2, 1, 3]]), process=False)   # same triangles, rotated
    assert _paint_areas(remesh_part(painted, rolled)) == before
    mirror = np.diag([-1.0, 1, 1, 1])
    back = painted.transformed(mirror).transformed(mirror)        # each mirror reverses the corners
    assert _paint_areas(painted.transformed(mirror)) == {(frozenset((-x, y, z) for x, y, z in c), s)
                                                         for c, s in before}
    assert _paint_areas(back) == before


def test_fast_parser_handles_bambu_layout(tmp_path: Path) -> None:
    """The regex path must accept what Bambu writes (it once never did: the
    "<triangles>" tag was counted as a triangle, so every file fell back to lxml)."""
    from pokefit.model3mf import _OBJ_RE, _parse_mesh_fast

    box = trimesh.creation.box([10, 10, 10])
    proj = read_project(write_inline_3mf(tmp_path / "b.3mf", box, np.eye(4), {0: "8"}))
    out = write_project(proj, "box", 1, proj.default_object().parts, tmp_path / "o.3mf")
    with zipfile.ZipFile(out) as z:
        body = _OBJ_RE.search(z.read("3D/Objects/object_1.model")).group(2)  # type: ignore[union-attr]
    fast = _parse_mesh_fast(body)
    assert fast is not None
    assert len(fast[1]) == 12 and fast[2] == {0: 'paint_color="8"'}


def test_objects_parse_lazily(tmp_path: Path) -> None:
    box = trimesh.creation.box([10, 10, 10])
    proj = read_project(write_inline_3mf(tmp_path / "b.3mf", box, np.eye(4)))
    obj = proj.default_object()
    assert obj.face_count == 12 and obj._parts is None      # counted, not parsed
    assert len(obj.parts[0].faces) == 12 and obj._parts is not None
