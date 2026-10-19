from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

from pokefit.gltf import leaf_barycentrics, tree_of
from pokefit.model3mf import Part, read_project
from pokefit.paint import decode_majority, leaves, parse_tree, split_children
from pokefit.pipeline import Params, analyse, open_session, preview_glb


def _states(part: Part) -> np.ndarray:
    """Main filament per face (paint wins over the part's extruder)."""
    out = np.full(len(part.faces), part.extruder)
    for i, a in part.tri_extra.items():
        out[i] = decode_majority(a.split('paint_color="', 1)[1].split('"', 1)[0])
    return out


def _glb_scene() -> trimesh.Scene:
    """A glTF-style scene in metres, Y-up: a red PBR box (2 nodes, one mirrored) and a blue
    vertex-coloured box on top of it."""
    red = trimesh.creation.box([0.02, 0.01, 0.02])
    red.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=[200, 20, 20, 255]))
    blue = trimesh.creation.box([0.01, 0.01, 0.01])
    blue.visual = trimesh.visual.ColorVisuals(blue, face_colors=[20, 20, 200, 255])
    s = trimesh.Scene()
    s.add_geometry(red, node_name="red", transform=trimesh.transformations.translation_matrix([0, 0.005, 0]))
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    mirror[:3, 3] = [0.03, 0.005, 0]
    s.add_geometry(red.copy(), node_name="red_mirrored", transform=mirror)
    s.add_geometry(blue, node_name="blue", transform=trimesh.transformations.translation_matrix([0, 0.015, 0]))
    return s


def _textured_sphere(emissive: bool = False) -> trimesh.Scene:
    """A 10 cm sphere textured red where x < 0 and blue where x > 0 (u = 0.5 at x = 0), with a
    small black dot (an "eye") at the +x pole. Optionally in the emissive slot with a black
    base colour, as some AI generators export."""
    m = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    n = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    uv = np.stack([0.5 + 0.5 * n[:, 0], 0.5 + 0.5 * n[:, 1]], axis=1)
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, :128] = (220, 30, 30)
    img[:, 128:] = (30, 30, 220)
    img[118:138, 240:256] = (10, 10, 10)                       # +x pole: u ~ 1, v ~ 0.5
    tex = Image.fromarray(img)
    mat = (PBRMaterial(baseColorFactor=[0, 0, 0, 255], emissiveTexture=tex, emissiveFactor=[1.0, 1.0, 1.0])
           if emissive else PBRMaterial(baseColorTexture=tex))
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return trimesh.Scene(m)


def test_glb_colours_become_paint(tmp_path: Path) -> None:
    f = tmp_path / "toy.glb"
    f.write_bytes(_glb_scene().export(file_type="glb"))
    proj = read_project(f)
    assert proj.kind == "mesh" and len(proj.objects) == 1
    assert proj.colours == ["#C81414", "#1414C8"]          # by area: 2 red boxes, 1 blue
    obj = proj.default_object()
    (part,) = obj.parts                                     # one solid part, colour as paint
    assert part.extruder == obj.extruder == 1
    st = _states(part)
    assert (st == 2).sum() == 12 and (st == 1).sum() == 24   # the blue box's 12 faces are painted 2
    assert all(isinstance(parse_tree(a.split('"')[1]), int) for a in part.tri_extra.values())  # flat: no splits
    # Metres -> mm, Y-up -> Z-up: the stack is 20 mm tall in Z; glTF's depth (its Z) becomes Y.
    v = part.vertices
    assert np.allclose(v.max(axis=0) - v.min(axis=0), [50, 20, 20], atol=1e-4)   # glTF stores float32
    blue = part.vertices[np.unique(part.faces[st == 2])]
    assert np.isclose(blue[:, 2].min(), 10, atol=1e-4) and np.isclose(blue[:, 2].max(), 20, atol=1e-4)
    m = part.mesh()                                         # seams merged, normals outward (mirror too)
    assert m.is_watertight and m.volume > 0 and len(m.split(only_watertight=False)) == 3


def test_glb_in_mm_keeps_units(tmp_path: Path) -> None:
    s = trimesh.Scene(trimesh.creation.box([40, 30, 20]))   # > 10 units: not metres
    f = tmp_path / "big.glb"
    f.write_bytes(s.export(file_type="glb"))
    v = read_project(f).default_object().parts[0].vertices
    assert np.allclose(v.max(axis=0) - v.min(axis=0), [40, 20, 30])


def test_glb_unreadable(tmp_path: Path) -> None:
    f = tmp_path / "broken.glb"
    f.write_bytes(b"glTF\x02\x00\x00\x00junk")
    with pytest.raises(ValueError, match="export a plain .glb"):
        read_project(f)


def test_leaf_layout_matches_split_tree() -> None:
    # The sampled leaves (leaf_barycentrics) are the leaves a depth-2 full split tree draws.
    b = leaf_barycentrics(2)
    tree = tree_of(np.arange(16) + 1)                       # every leaf its own state: never merged
    drawn = leaves(tree, list(np.eye(3)))
    assert [st for _, st in drawn] == list(range(1, 17))
    assert all(np.allclose(np.array(v), b[i]) for i, (v, _) in enumerate(drawn))
    assert tree_of(np.full(16, 3)) == 3                     # uniform: one leaf
    assert len(split_children(list(np.eye(3)), 3, 0)) == 4


@pytest.mark.parametrize("emissive", [False, True])
def test_texture_becomes_split_paint(tmp_path: Path, emissive: bool) -> None:
    f = tmp_path / "ball.glb"
    f.write_bytes(_textured_sphere(emissive).export(file_type="glb"))
    proj = read_project(f)
    assert len(proj.colours) == 3                           # red, blue and the black dot
    rgb = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in proj.colours]
    red = next(i for i, c in enumerate(rgb) if c[0] > 150) + 1
    blue = next(i for i, c in enumerate(rgb) if c[2] > 150) + 1
    black = next(i for i, c in enumerate(rgb) if max(c) < 60) + 1
    part = proj.default_object().parts[0]
    centre = part.vertices[part.faces].mean(axis=1)
    st = _states(part)
    far = np.abs(centre[:, 0]) > 5                          # mm (x is kept by the Y-up turn)
    pole = centre[:, 0] > 45
    assert (st[far & (centre[:, 0] < 0)] == red).all()
    assert (st[far & (centre[:, 0] > 0) & ~pole] == blue).mean() > 0.97
    assert (st[pole] == black).any()                        # the dot survives as its own colour
    # Triangles across x = 0 are split along it: their leaves are red on one side, blue on the other.
    split = [i for i, a in part.tri_extra.items() if not isinstance(parse_tree(a.split('"')[1]), int)]
    assert split
    for i in split:
        for v, s in leaves(parse_tree(part.tri_extra[i].split('"')[1]), list(part.vertices[part.faces[i]])):
            x = np.mean([p[0] for p in v])
            if 1.5 < abs(x) and x < 40:                    # clear of the boundary and of the dot at +x
                assert s == (red if x < 0 else blue)


def test_colour_count_and_preview(tmp_path: Path) -> None:
    f = tmp_path / "ball.glb"
    f.write_bytes(_textured_sphere().export(file_type="glb"))
    s = open_session(f, work_dir=tmp_path / "w")
    rep, _ = analyse(s, Params())
    info = rep.to_dict()["object"]["texture_colours"]
    assert info == {"auto": 3, "count": 3, "max": 16}
    two = Params(texture_colours=2)                         # the dot joins its nearer colour
    rep2, _ = analyse(s, two)
    o = rep2.to_dict()["object"]
    assert len(o["colours"]) == 2 and o["texture_colours"]["count"] == 2
    st = _states(s.figure_parts(two)[0])
    assert set(np.unique(st).tolist()) == {1, 2}
    # The preview draws split paint as sub-triangles: more triangles than the mesh has.
    glb = trimesh.load(trimesh.util.wrap_as_stream(preview_glb(s, Params())), file_type="glb")
    assert sum(len(g.faces) for g in glb.geometry.values()) > len(s.unit_parts(Params())[0].faces)
    with pytest.raises(ValueError):
        Params(texture_colours=0)


def _glb_vertex_colours(vertices: np.ndarray, faces: np.ndarray, linear: np.ndarray) -> bytes:
    """A minimal GLB like ComfyUI's MeshToFile3D writes: float COLOR_0 (linear) under a white PBR material.
    (trimesh's own exporter writes a COLOR_0 accessor with the wrong count for this case.)"""
    import json as _json
    import struct
    pos = np.asarray(vertices, np.float32).tobytes()
    col = np.asarray(linear, np.float32).tobytes()
    idx = np.asarray(faces, np.uint32).tobytes()
    blob = idx + pos + col
    n = len(vertices)
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "COLOR_0": 2}, "indices": 0, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0}}],
           "buffers": [{"byteLength": len(blob)}],
           "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(idx)},
                           {"buffer": 0, "byteOffset": len(idx), "byteLength": len(pos)},
                           {"buffer": 0, "byteOffset": len(idx) + len(pos), "byteLength": len(col)}],
           "accessors": [{"bufferView": 0, "componentType": 5125, "count": int(np.size(faces)), "type": "SCALAR"},
                         {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3",
                          "min": np.min(vertices, 0).tolist(), "max": np.max(vertices, 0).tolist()},
                         {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"}]}
    js = _json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_vertex_colours_under_a_pbr_material_are_linear(tmp_path: Path) -> None:
    """COLOR_0 under a material (ComfyUI's PaintMesh writes this) multiplies the base colour and is LINEAR."""
    box = trimesh.creation.box((0.02, 0.02, 0.02)).subdivide()
    lin = np.where(box.vertices[:, :1] > 0, [0.2, 0.2, 0.9], [0.9, 0.05, 0.05])     # two halves, linear RGB
    path = tmp_path / "paint.glb"
    path.write_bytes(_glb_vertex_colours(box.vertices, box.faces, lin))
    proj = read_project(path)
    srgb = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in proj.colours}
    # linear 0.2 -> sRGB 124, 0.9 -> 243, 0.05 -> 63 (not 51 / 230 / 13 as plain 8-bit would give)
    assert any(abs(r - 124) < 8 and abs(b - 243) < 8 for r, _, b in srgb), srgb
    assert any(abs(r - 243) < 8 and abs(g - 63) < 10 for r, g, _ in srgb), srgb


def test_small_palettes_keep_large_distinct_colours() -> None:
    """Merging for a chosen count weighs area (Ward-like): a large cream belly survives at 3 colours
    while a tiny dark detail merges; closest-first merged the belly into the blue body."""
    from pokefit.gltf import TextureColours, _lab
    rgb = np.array([[140, 200, 215], [150, 205, 220], [200, 190, 160], [165, 100, 65], [30, 30, 25]], float)
    weight = np.array([0.40, 0.10, 0.25, 0.23, 0.02])
    tex = TextureColours(np.zeros((1, 1), np.intp), _lab(rgb), rgb, weight, np.ones(1), np.zeros((0, 2), np.intp), False)
    three = [sorted(g) for g in tex.groups(3)]
    assert [0, 1] in three                            # the two blues merge
    assert not any(2 in g and (0 in g or 1 in g) for g in three)   # cream never joins the blue body
    assert any(4 in g and len(g) > 1 for g in three)  # the tiny dark joins a neighbour
    four = [sorted(g) for g in tex.groups(4)]
    assert [4] in four                                # at 4 the dark detail gets its own filament
