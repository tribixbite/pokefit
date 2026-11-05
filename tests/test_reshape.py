"""Reshape a limb (reshape.py): tap a tail, resize or bend it at the joint; the body and the paint stay."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import BALL_3MF, write_inline_3mf
from pokefit.colorize import TopologyCache, pick_face
from pokefit.model3mf import Part
from pokefit.pipeline import Params, analyse, limb_region, open_session
from pokefit.reshape import LimbOp, apply_limbs, find_limb, limb_axes


def _tailed() -> trimesh.Trimesh:
    """A 20 x 10 x 10 block (z 0..10) with a round tail 15 mm long sticking out along +x at mid-height."""
    body = trimesh.creation.box([20, 10, 10]).apply_translation([0, 0, 5])
    tail = trimesh.creation.cylinder(radius=2.0, height=16.0, sections=32)
    tail.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    tail.apply_translation([10 + 7, 0, 5])          # from x = 9 (inside the block) to x = 25
    m = trimesh.boolean.union([body, tail], engine="manifold")
    return m.subdivide_to_size(1.0)


def _part(m: trimesh.Trimesh, extra: dict[int, str] | None = None) -> Part:
    return Part("body", 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), dict(extra or {}))


def _tail_face(part: Part) -> int:
    _, f, d = pick_face([part], np.array([24.0, 0.0, 7.0]), np.array([0.0, 0.0, 1.0]))
    assert d < 1e-6
    return f


def test_a_tap_on_the_tail_finds_the_tail_up_to_the_joint() -> None:
    m = _tailed()
    part = _part(m)
    topo = TopologyCache([part])
    limb = find_limb(topo, 0, _tail_face(part))
    c = m.triangles_center[limb.faces]
    assert c[:, 0].min() > 10 - 1e-6                       # nothing of the block
    assert len(limb.faces) == int((m.triangles_center[:, 0] > 10 + 1e-6).sum())   # all of the tail
    # Its length is the longest walk from the joint: 15 mm along the tail and on across its end to the middle.
    assert len(limb.joint) > 8 and 15 <= limb.length <= 17.5
    assert limb.weight.min() == 0 and limb.weight.max() == 1
    with pytest.raises(ValueError, match="too much of the figure"):
        find_limb(topo, 0, int(np.argmin(m.triangles_center[:, 0])))       # the block itself


def test_bend_lifts_the_tip_and_leaves_the_body_and_paint() -> None:
    m = _tailed()
    extra = {0: 'paint_color="8"', 7: 'paint_color="C"'}
    part = _part(m, extra)
    topo = TopologyCache([part])
    f = _tail_face(part)
    (bent,) = apply_limbs([part], [LimbOp(0, f, bend=45.0)], topo)
    assert np.array_equal(bent.faces, part.faces) and bent.tri_extra == extra
    v0, v1 = part.vertices, bent.vertices
    body = v0[:, 0] < 10 - 1e-6
    assert np.allclose(v1[body], v0[body])                 # the block is untouched
    tip = np.argmax(v0[:, 0])
    rise = v1[tip, 2] - v0[tip, 2]
    assert 6.0 < rise < 15 * np.sin(np.radians(45)) + 0.5  # up, by about the length past the blend x sin 45
    assert v1[tip, 0] < v0[tip, 0]                         # and in: it turned, not stretched
    # The surface stays in one piece: every edge keeps its length within a sane factor (no tearing).
    e = bent.mesh().edges_unique
    ratio = np.linalg.norm(v1[e[:, 0]] - v1[e[:, 1]], axis=1) / np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    assert ratio.max() < 2.0 and ratio.min() > 0.4
    (down,) = apply_limbs([part], [LimbOp(0, f, bend=-30.0)], topo)
    assert down.vertices[tip, 2] < v0[tip, 2] - 3


def test_resize_grows_the_tail_from_its_joint_and_swing_turns_it_sideways() -> None:
    part = _part(_tailed())
    topo = TopologyCache([part])
    f = _tail_face(part)
    v0 = part.vertices
    tip = np.argmax(v0[:, 0])
    (big,) = apply_limbs([part], [LimbOp(0, f, scale=1.5)], topo)
    pivot_x = 10.0
    grown = (big.vertices[tip, 0] - pivot_x) / (v0[tip, 0] - pivot_x)
    assert 1.4 < grown <= 1.5 + 1e-9                     # the tip moves out by nearly half the length again
    assert big.vertices[tip, 2] - 5 == pytest.approx((v0[tip, 2] - 5) * 1.5, abs=0.05)     # thicker too
    (turned,) = apply_limbs([part], [LimbOp(0, f, swing=40.0)], topo)
    assert turned.vertices[tip, 1] > v0[tip, 1] + 5       # counter-clockwise seen from above: toward +y
    assert turned.vertices[tip, 2] == pytest.approx(v0[tip, 2], abs=0.6)


def test_limb_axes_follow_the_figure() -> None:
    bend, swing = limb_axes(np.array([1.0, 0, 0]))
    assert np.allclose(bend, [0, -1, 0]) and np.allclose(swing, [0, 0, 1])
    bend, _ = limb_axes(np.array([0.0, 0, 1]))          # an ear pointing up tips to the front (-y)
    tip = np.array([0.0, 0, 1])
    turned = tip * np.cos(0.3) + np.cross(bend, tip) * np.sin(0.3)
    assert turned[1] < 0


def test_limbs_in_the_pipeline_change_the_fit_and_keep_the_paint(tmp_path: Path) -> None:
    m = _tailed()
    top = {int(i): "8" for i in np.nonzero(m.face_normals[:, 2] > 0.99)[0][:20]}
    s = open_session(write_inline_3mf(tmp_path / "tail.3mf", m, np.eye(4), top), BALL_3MF, None, tmp_path / "w")
    base = Params(magnets=0)
    hit = limb_region(s, base, [24.0 - 12.5, 0.0, 7.0], [0.0, 0.0, 1.0])     # unit frame: centred on x/y
    assert hit["joined"] and 0.05 < hit["share"] < 0.6 and hit["length_mm"] > 5
    flat, _ = analyse(s, base)
    up = Params(magnets=0, limbs=[{"part": hit["part"], "face": hit["face"], "bend": 60.0}])
    bent, _ = analyse(s, up)
    # The tail no longer sticks out flat: the footprint is narrower and the figure fits larger.
    w0 = np.ptp(np.vstack([p.vertices for p in s.unit_parts(base)])[:, 0])
    w1 = np.ptp(np.vstack([p.vertices for p in s.unit_parts(up)])[:, 0])
    assert w1 < w0 - 3 and bent.fit["max_scale"] > flat.fit["max_scale"]
    assert s.figure_parts(up)[0].tri_extra.keys() == s.figure_parts(base)[0].tri_extra.keys()
    assert Params.from_dict(up.__dict__).limbs == up.limbs
    with pytest.raises(ValueError):
        Params(limbs=[{"part": 0, "face": 0, "scale": 9}])


def test_a_tap_on_a_shallow_joint_steps_down_to_a_smaller_angle() -> None:
    """Asked for an angle the joint is too shallow for (100 degrees here, the joint being 90: a sculpt's valleys are a
    few degrees), the search runs on into the body; `find_limb_at` steps down until the tail stands alone, and
    says which angle it used."""
    from pokefit.reshape import find_limb_at

    part = _part(_tailed())
    topo = TopologyCache([part])
    f = _tail_face(part)
    with pytest.raises(ValueError):
        find_limb(topo, 0, f, 100.0)                   # past the 90-degree joint: the body
    limb, used = find_limb_at(topo, 0, f, 100.0)
    assert used < 90 and len(limb.joint) > 8


def _radius(v: np.ndarray, x0: float, x1: float) -> float:
    """Mean distance from the tail's axis (y = 0, z = 5) of the vertices with x0 <= x < x1."""
    sel = (v[:, 0] >= x0) & (v[:, 0] < x1)
    return float(np.hypot(v[sel, 1], v[sel, 2] - 5).mean())


def test_thicken_widens_the_base_only_and_tapers_along_the_tail() -> None:
    """Base width x2 over the first 40 % of the tail: the root's radius doubles, the middle grows less, the far
    half and the body stay as they were, and paint and triangles carry over."""
    extra = {3: 'paint_color="8"'}
    part = _part(_tailed(), extra)
    topo = TopologyCache([part])
    f = _tail_face(part)
    v0 = part.vertices
    (wide,) = apply_limbs([part], [LimbOp(0, f, thicken=2.0, reach=0.4, steep=1.0)], topo)
    v1 = wide.vertices
    assert np.array_equal(wide.faces, part.faces) and wide.tri_extra == extra
    root0, root1 = _radius(v0, 10.2, 11.0), _radius(v1, 10.2, 11.0)
    mid0, mid1 = _radius(v0, 13.0, 14.0), _radius(v1, 13.0, 14.0)
    assert root1 / root0 == pytest.approx(1.95, abs=0.1)    # x2 at the joint ring, a little less 1 mm along
    assert 1.05 < mid1 / mid0 < root1 / root0              # tapering
    far = v0[:, 0] > 10 + 0.4 * 17.5 + 1                   # past the reach (of the longest walk) nothing moves
    assert np.allclose(v1[far], v0[far])
    inner = v0[:, 0] < 10 - 1e-6
    assert np.allclose(v1[inner], v0[inner])              # the body's own vertices stay (the joint ring is the limb's)
    # A steeper taper keeps less of the width 3 mm out; a gentler one more.
    (steep,) = apply_limbs([part], [LimbOp(0, f, thicken=2.0, reach=0.4, steep=3.0)], topo)
    (soft,) = apply_limbs([part], [LimbOp(0, f, thicken=2.0, reach=0.4, steep=0.4)], topo)
    assert _radius(steep.vertices, 13.0, 14.0) < mid1 < _radius(soft.vertices, 13.0, 14.0)
    with pytest.raises(ValueError, match="base width"):
        LimbOp.from_dict({"part": 0, "face": f, "thicken": 5.0})
