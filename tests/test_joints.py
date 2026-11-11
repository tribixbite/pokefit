"""Weak points (joints.py): thin features graded by the push that snaps them; single features thickened
(Params.strengthen_at); reinforcement modifier volumes in the export."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pokefit import joints
from pokefit.model3mf import Part, read_project
from pokefit.pipeline import Params, export, open_session, weak_points
from pokefit.strengthen import Local, strengthen


def _rods() -> trimesh.Trimesh:
    """A 20 x 12 x 10 mm body with two horizontal rods 10 mm long, Ø1.5 mm (+x side) and Ø2.2 mm (-x side), at
    mid-height: cantilevers whose breaking push is sigma x pi d³ / 32 / L."""
    body = trimesh.creation.box((20, 12, 10)).apply_translation([0, 0, 5])
    thin = trimesh.creation.cylinder(radius=0.75, height=10.5, sections=48)
    thin.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    thin.apply_translation([10 + 10.5 / 2 - 0.5, -3, 5])
    thick = trimesh.creation.cylinder(radius=1.1, height=10.5, sections=48)
    thick.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    thick.apply_translation([-(10 + 10.5 / 2 - 0.5), 3, 5])
    m = trimesh.boolean.union([body, thin, thick], engine="manifold").subdivide_to_size(0.3)
    assert isinstance(m, trimesh.Trimesh)
    return m


def _part(m: trimesh.Trimesh) -> Part:
    return Part("fig", 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), {}, "normal_part")


def _inside(part: Part, p: np.ndarray) -> bool:
    """Ray parity along +z (no rtree): whether `p` is inside the closed mesh."""
    tri = part.vertices[part.faces]
    a, b, c = tri[:, 0, :2] - p[:2], tri[:, 1, :2] - p[:2], tri[:, 2, :2] - p[:2]

    def side(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    hit = ((s1 > 0) & (s2 > 0) & (s3 > 0)) | ((s1 < 0) & (s2 < 0) & (s3 < 0))
    w = np.c_[s2, s3, s1][hit] / (s1 + s2 + s3)[hit, None]           # barycentric weights of the z = column point
    z = (w * tri[hit][:, :, 2]).sum(axis=1)
    return bool((z > p[2]).sum() % 2)


def _cantilever(d: float, length: float) -> float:
    """Push (N) at the end of a horizontal round rod that reaches the bending strength along the layers."""
    return joints.SIGMA_XY * np.pi * d ** 3 / 32 / length


def test_a_thin_rod_is_graded_by_the_push_that_snaps_it_at_its_root() -> None:
    found = joints.weak_points([_part(_rods())], 1.0)
    rods = [w for w in found if w.volume_mm3 > 5]
    assert len(rods) == 2, [w.to_dict() for w in found]
    thin = min(rods, key=lambda w: w.width_mm)
    thick = max(rods, key=lambda w: w.width_mm)
    # Measured near the root (the body's face is at x = 10), across the rod, upright section: stress along layers.
    assert thin.centre[0] == pytest.approx(10.5, abs=1.0) and thin.flat < 0.1
    assert thin.width_mm == pytest.approx(1.5, abs=0.3) and thin.lever_mm == pytest.approx(10, abs=1.2)
    for w, d in ((thin, 1.5), (thick, 2.2)):
        assert w.force_n == pytest.approx(_cantilever(d, 10), rel=0.4), (d, w.to_dict())
    assert thin.grade == "D" and thick.grade == "C"
    # The piece that breaks off is the rod's own triangles, not the body's.
    tri = _rods().triangles_center[thin.faces]
    assert (tri[:, 0] > 9.5).all()


def test_a_straight_rod_gets_stronger_towards_its_tip_at_every_section(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every section listed (none merged into the weakest): the push that snaps a uniform rod grows as its lever
    # shortens. Counting slab voxels read some sections 30 % small (a dip mid-rod that was not in the rod).
    monkeypatch.setattr(joints, "NEST", 0.0)
    monkeypatch.setattr(joints, "MAX_REPORTED", 999)
    rod = sorted((w for w in joints.weak_points([_part(_rods())], 1.0) if w.centre[0] > 10 and w.centre[1] < 0),
                 key=lambda w: w.centre[0])
    assert len(rod) > 20
    forces = np.array([w.force_n for w in rod])
    assert (forces[1:] >= forces[:-1] * 0.97).all(), forces.round(2)
    assert all(w.area_mm2 == pytest.approx(np.pi * 0.75 ** 2, rel=0.2) for w in rod)


def test_nothing_is_reported_without_thin_features() -> None:
    block = trimesh.creation.box((10, 10, 10)).apply_translation([0, 0, 5]).subdivide_to_size(0.5)
    assert joints.weak_points([_part(block)], 1.0) == []
    assert joints.grade(100.0, 0.5) == "F"                  # narrower than two extrusion lines: F whatever its force


def test_one_feature_thickened_and_the_other_left_as_it_is() -> None:
    m = _rods()
    part = _part(m)
    thin = min((w for w in joints.weak_points([part], 1.0) if w.volume_mm3 > 5), key=lambda w: w.width_mm)
    cen = part.vertices[part.faces].mean(axis=1)
    root = int(np.argmin(((cen - np.asarray(thin.centre)) ** 2).sum(axis=1)))
    face = int(thin.faces[np.argmin(((cen[thin.faces] - np.asarray(thin.end)) ** 2).sum(axis=1))])
    (out,), info = strengthen([part], 1.0, 0.0, [Local(0, face, 2.5, root)])
    assert info.features == 1 and info.max_push_mm > 0.2
    # Stretched (strengthen._stretch): the same triangles, so paint and relief stay; only vertices move.
    assert np.array_equal(out.faces, part.faces) and out.tri_extra == part.tri_extra
    after = joints.weak_points([out], 1.0)
    rods = {("thin" if w.centre[0] > 0 else "thick"): w for w in after if w.volume_mm3 > 5}
    assert rods["thin"].force_n > 2.5 * thin.force_n                # D -> C or better, all along the rod
    # Thickest at its root, tapering (the moment falls towards the tip), the tip itself as it was.
    v = out.vertices
    def radius(x: float) -> float:
        sel = (np.abs(v[:, 0] - x) < 0.2) & (v[:, 1] < 0)
        return float(np.median(np.linalg.norm(v[sel][:, 1:] - [-3.0, 5.0], axis=1)))
    assert radius(11.5) > radius(15) > radius(19) == pytest.approx(0.75, abs=0.02)
    assert radius(11.5) > 1.0
    # The other rod (x < -10) and the body are untouched: every vertex there is one of the original's.
    from scipy.spatial import cKDTree

    other = out.vertices[out.vertices[:, 0] < -1]
    d, _ = cKDTree(part.vertices).query(other)
    assert d.max() < 1e-6


def _curved_fin() -> trimesh.Trimesh:
    """A 20 x 12 x 10 mm body with a bent fin 10 mm long out of its +x face: the lower part of a pipe along x (inner
    radius 2.5, 0.6 mm wall, axis at y = 0, z = 7) below z = 6, a sheet curved like an ear (concave face up)."""
    body = trimesh.creation.box((20, 12, 10)).apply_translation([0, 0, 5])
    pipe = trimesh.creation.annulus(r_min=2.5, r_max=3.1, height=10.5, sections=96)
    pipe.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    pipe.apply_translation([10 + 10.5 / 2 - 0.5, 0, 7])
    lower = trimesh.creation.box((12, 16, 6)).apply_translation([15, 0, 3])
    fin = trimesh.boolean.intersection([pipe, lower], engine="manifold")
    m = trimesh.boolean.union([body, fin], engine="manifold").subdivide_to_size(0.3)
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_a_bent_sheet_thickens_on_its_convex_side_and_keeps_its_concave_face() -> None:
    # Pushing a concave face out along its normals shrinks it: an ear's painted inner panel lost 11 % of its area, this
    # fin's 4.5 % (the outer face grew 4 %).
    # strengthen._even_sheet puts the thickening on the convex side instead; the concave face keeps its size.
    part = _part(_curved_fin())
    thin = min((w for w in joints.weak_points([part], 1.0) if w.centre[0] > 10), key=lambda w: w.force_n)
    cen = part.vertices[part.faces].mean(axis=1)
    root = int(np.argmin(((cen - np.asarray(thin.centre)) ** 2).sum(axis=1)))
    face = int(thin.faces[np.argmin(((cen[thin.faces] - np.asarray(thin.end)) ** 2).sum(axis=1))])
    (out,), _ = strengthen([part], 1.0, 0.0, [Local(0, face, 2.0, root)])
    assert np.array_equal(out.faces, part.faces)
    # The fin's faces (past the body), split by their distance to the pipe's axis: inner (concave) and outer.
    r = np.linalg.norm(cen[:, 1:] - [0.0, 7.0], axis=1)
    fin = (cen[:, 0] > 10.3) & (cen[:, 2] < 5.9) & (np.abs(r - 2.8) < 0.35)
    inner, outer = fin & (r < 2.8), fin & (r > 2.8)
    def area(v: np.ndarray, sel: np.ndarray) -> float:
        t = v[part.faces[sel]]
        return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)
    assert area(out.vertices, inner) == pytest.approx(area(part.vertices, inner), rel=0.02)
    assert area(out.vertices, outer) > 1.05 * area(part.vertices, outer)
    after = [w for w in joints.weak_points([out], 1.0) if w.centre[0] > 10]
    assert not after or min(w.force_n for w in after) > 1.4 * thin.force_n


def _flame(d: float) -> trimesh.Trimesh:
    """A 20 x 12 x 10 mm body with an upright neck Ø`d` from its top (z 10 to 15) holding a ball Ø6 (centre z 17.5):
    a flame on a tail. Its sections lie flat (the layers pull apart) and the lever reaches the ball's top, z 20.5."""
    body = trimesh.creation.box((20, 12, 10)).apply_translation([0, 0, 5])
    neck = trimesh.creation.cylinder(radius=d / 2, height=6.0, sections=48).apply_translation([0, 0, 12.5])
    ball = trimesh.creation.icosphere(subdivisions=4, radius=3.0).apply_translation([0, 0, 17.5])
    m = trimesh.boolean.union([body, neck, ball], engine="manifold").subdivide_to_size(0.3)
    assert isinstance(m, trimesh.Trimesh)
    return m


def _handle(part: Part, w: joints.WeakPoint) -> Local:
    """What the app hands `strengthen_at` for a weak point (pipeline.weak_points: root and face triangles, probe)."""
    cen = part.vertices[part.faces].mean(axis=1)
    root = int(np.argmin(((cen - np.asarray(w.centre)) ** 2).sum(axis=1)))
    face = int(w.faces[np.argmin(((cen[w.faces] - np.asarray(w.end)) ** 2).sum(axis=1))])
    return Local(0, face, 3.6, root, w.probe)


def test_a_neck_holding_a_thick_piece_is_graded_by_the_lever_to_its_far_end() -> None:
    # The ball fits a PROBE ball, so it is thick; taken as a body of its own, the neck between it and the body was a
    # short piece with a ~1 mm lever (graded A; Charmander's flame neck, which snapped in print).
    found = [w for w in joints.weak_points([_part(_flame(2.2))], 1.0) if w.volume_mm3 > 100]   # the ball goes with it
    assert len(found) == 1, [w.to_dict() for w in found]     # found at both probes, listed once (the weaker reading)
    neck = found[0]
    assert neck.flat > 0.9 and neck.centre[2] == pytest.approx(10.5, abs=1.5)
    assert neck.lever_mm == pytest.approx(10.0, abs=1.5)
    assert neck.force_n == pytest.approx(joints.SIGMA_Z * np.pi * 2.2 ** 3 / 32 / 10.0, rel=0.4)
    assert neck.grade in "DF" and neck.probe in (joints.PROBE, joints.NECK_PROBE)
    # A neck a little thicker than PROBE (a ball fits round its axis: "body" at PROBE) is found with NECK_PROBE.
    wide = [w for w in joints.weak_points([_part(_flame(3.4))], 1.0) if w.volume_mm3 > 100]
    assert len(wide) == 1 and wide[0].probe == joints.NECK_PROBE and wide[0].grade in "CD"


def test_a_neck_holding_a_thick_piece_widens_and_the_piece_keeps_its_shape() -> None:
    # strengthen._stretch read the neck as thick as the ball beside it (+0.02 mm on Charmander): `_widen_neck` widens it
    # about its centre line, each ring to a soft maximum of its width and the target, so the ball stays round.
    m = _flame(2.2)
    part = _part(m)
    neck = min(joints.weak_points([part], 1.0), key=lambda w: w.force_n)
    (out,), info = strengthen([part], 1.0, 0.0, [_handle(part, neck)])
    assert info.features == 1 and np.array_equal(out.faces, part.faces)
    v0, v1 = part.vertices, out.vertices

    def radius(v: np.ndarray, z: float) -> float:
        sel = (np.abs(v0[:, 2] - z) < 0.15) & (np.linalg.norm(v0[:, :2], axis=1) < 2.0)
        return float(np.median(np.linalg.norm(v[sel][:, :2], axis=1)))
    for z in (11.5, 12.5, 13.5):
        assert 2 * radius(v1, z) > 0.85 * 3.6, z                   # 2.2 mm -> near the 3.6 mm target
    ball = np.linalg.norm(v0 - [0, 0, 17.5], axis=1) < 3.05
    top = ball & (v0[:, 2] > 17.5)
    assert np.abs(np.linalg.norm(v1[top] - [0, 0, 17.5], axis=1) - 3.0).max() < 0.03   # the ball's top half as it was
    body = v0[:, 2] < 9.0
    assert np.abs(v1[body] - v0[body]).max() < 1e-9
    after = min(joints.weak_points([out], 1.0), key=lambda w: w.force_n)
    assert after.force_n > 3.0 * neck.force_n


def test_params_take_single_features_and_check_them() -> None:
    p = Params(strengthen_at=[{"part": 0, "face": 3, "mm": 2.5, "root": 7}])
    assert p.strengthens and p.local_strengthen() == [Local(0, 3, 2.5, 7)]
    assert Params(strengthen_at=[{"part": 0, "face": 3, "mm": 2.5, "root": 7, "probe": 4}]).local_strengthen() == \
        [Local(0, 3, 2.5, 7, 4.0)]
    assert not Params().strengthens
    with pytest.raises(ValueError, match="strengthen_at"):
        Params(strengthen_at=[{"part": 0, "face": 3, "mm": 6}])
    with pytest.raises(ValueError, match="probe"):
        Params(strengthen_at=[{"part": 0, "face": 3, "mm": 2, "probe": 40}])


def test_reinforcement_volumes_cover_weak_features_and_add_only_what_the_process_lacks(tmp_path: Path) -> None:
    part = _part(_rods())
    mods, held = joints.reinforcements([part], 1.0)
    assert mods and all(m.subtype == "modifier_part" and m.settings == joints.REINFORCE for m in mods)
    assert all(len(m.faces) <= joints.REINFORCE_FACES * 1.3 for m in mods)
    assert {w.grade for w in held} <= set(joints.REINFORCE_GRADES)
    # The thin rod's root (x = 10.5, y = -3, z = 5) is inside a volume.
    assert any(_inside(m, np.array([10.5, -3.0, 5.0])) for m in mods)
    assert joints.stronger({"sparse_infill_density": "100%", "wall_loops": "2"}) == {"wall_loops": "6"}
    assert joints.stronger({"sparse_infill_density": "100%", "wall_loops": "6"}) == {}
    assert joints.stronger({"sparse_infill_density": "15%", "wall_loops": ["3"]}) == joints.REINFORCE


def test_the_export_writes_reinforcement_volumes_and_reopening_drops_them(tmp_path: Path) -> None:
    stl = tmp_path / "rods.stl"
    _rods().export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    p = Params(magnets=0, flat_base=0, figure_height=10, reinforce=True)
    rep = weak_points(s, p)
    assert rep["points"] and all(w["face"] >= 0 and w["root"] >= 0 and w["view_faces"] for w in rep["points"])
    res = export(s, p, tmp_path / "out")
    assert res["reinforced"]
    with zipfile.ZipFile(res["figure_3mf"]) as z:
        cfg = z.read("Metadata/model_settings.config").decode()
    if res["reinforce_volumes"]:
        assert 'subtype="modifier_part"' in cfg and 'key="wall_loops"' in cfg
    again = read_project(Path(res["figure_3mf"])).default_object()
    assert [q.subtype for q in again.parts] == ["normal_part"]
