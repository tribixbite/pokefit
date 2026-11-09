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
    after = joints.weak_points([out], 1.0)
    rods = {("thin" if w.centre[0] > 0 else "thick"): w for w in after if w.volume_mm3 > 5}
    assert rods["thin"].force_n > 2 * thin.force_n                  # D -> C or better
    # The other rod (x < -10) and the body are untouched: every vertex there is one of the original's.
    from scipy.spatial import cKDTree

    other = out.vertices[out.vertices[:, 0] < -1]
    d, _ = cKDTree(part.vertices).query(other)
    assert d.max() < 1e-6


def test_params_take_single_features_and_check_them() -> None:
    p = Params(strengthen_at=[{"part": 0, "face": 3, "mm": 2.5, "root": 7}])
    assert p.strengthens and p.local_strengthen() == [Local(0, 3, 2.5, 7)]
    assert not Params().strengthens
    with pytest.raises(ValueError, match="strengthen_at"):
        Params(strengthen_at=[{"part": 0, "face": 3, "mm": 6}])


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
