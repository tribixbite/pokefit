"""Pockets of an earlier export (pokefit/oldpockets.py): found from their shape, filled, the flat base dropped."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from conftest import CLEAN, edge_stats
from pokefit.magnets import Pocket, PocketSpec, cut_pockets
from pokefit.model3mf import Part, read_project
from pokefit.oldpockets import find_pockets, strip_pockets
from pokefit.pipeline import Params, analyse, export, open_session
from pokefit.repair import repair_parts


def _part(m: trimesh.Trimesh, name: str = "figure") -> Part:
    return Part(name, 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), {}, "normal_part")


def test_a_cut_pocket_is_found_filled_and_its_flat_base_dropped() -> None:
    body = _part(trimesh.creation.box((16, 12, 10)).apply_translation([0, 0, 5]))
    base = _part(trimesh.creation.box((20, 14, 1.0)).apply_translation([0, 0, 0.5]), "flat base")
    spec = PocketSpec()
    cut, _ = cut_pockets([body, base], [Pocket(3.0, -2.0), Pocket(-4.0, 1.5)], spec)
    shift = np.eye(4)
    shift[:3, 3] = [120.0, 100.0, 0.0]                         # a build item's place on the plate
    parts, found, bases = strip_pockets(cut, shift)
    assert bases == 1 and [p.name for p in parts] == ["figure"]
    got = sorted((round(p.x - 120, 2), round(p.y - 100, 2)) for p in found)
    assert got == [(-4.0, 1.5), (3.0, -2.0)]
    assert all(abs(2 * p.radius - spec.diameter) < 0.05 and abs(p.z + p.depth - spec.depth) < 0.02 for p in found)
    fixed, _ = repair_parts(parts)
    m = trimesh.Trimesh(fixed[0].vertices, fixed[0].faces, process=False)
    assert edge_stats(fixed[0].vertices, fixed[0].faces) == {"index": CLEAN, "welded": CLEAN}
    assert abs(m.volume - 16 * 12 * 10) < 1e-3                 # solid again, as before the cut


def test_other_round_holes_are_left_alone() -> None:
    """A hole with a domed top, and a figure with no pocket: nothing is found."""
    body = trimesh.creation.box((16, 12, 10)).apply_translation([0, 0, 5])
    dome = trimesh.creation.capsule(height=2.0, radius=2.0, count=[32, 16])   # rounded at both ends
    holed = trimesh.boolean.difference([body, dome], engine="manifold")
    assert find_pockets([(np.asarray(holed.vertices), np.asarray(holed.faces))]) == []
    plain = _part(body)
    parts, found, bases = strip_pockets([plain], np.eye(4))
    assert parts[0] is plain and found == [] and bases == 0


def test_an_export_reopens_with_its_pockets_and_reads_as_written_for_tools(tmp_path: Path) -> None:
    """Round trip through `export`: opened again, the pockets are found where they were cut and the flat base is
    gone (made again from the settings); read `as_written` (render/verify tools) the file is untouched."""
    parts = [trimesh.creation.box((14, 9, 10)).apply_translation([0, 0, 7.0])]
    for x in (-5.5, 5.5):
        for y in (-3, 3):
            parts.append(trimesh.creation.box((2.4, 2.4, 2.5)).apply_translation([x, y, 1.25]))
    stl = tmp_path / "legs.stl"
    trimesh.util.concatenate(parts).export(stl)
    s = open_session(stl, work_dir=tmp_path / "w")
    out = export(s, Params(magnets=2), tmp_path / "out")
    placed = analyse(s, Params(magnets=2))[0].pockets
    assert len(placed) == 2
    again = read_project(out["figure_3mf"]).default_object()
    assert all(p.name != "flat base" for p in again.parts) and again.old_bases >= 1
    assert len(again.old_pockets) == 2
    raw = read_project(out["figure_3mf"], as_written=True).default_object()
    assert any(p.name == "flat base" for p in raw.parts) and raw.old_pockets == []
    # Found where they were cut: the same spacing as placed (the plate frame is only shifted and the figure scaled).
    gap = float(np.hypot(placed[0]["x"] - placed[1]["x"], placed[0]["y"] - placed[1]["y"]))
    a, b = again.old_pockets
    assert abs(float(np.hypot(a.x - b.x, a.y - b.y)) - gap) < 0.01
