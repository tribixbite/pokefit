"""Pockets of an earlier export (pokefit/oldpockets.py): found from their shape, filled, the flat base dropped."""
from __future__ import annotations

import numpy as np
import trimesh

from conftest import CLEAN, edge_stats
from pokefit.magnets import Pocket, PocketSpec, cut_pockets
from pokefit.model3mf import Part
from pokefit.oldpockets import find_pockets, strip_pockets
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
