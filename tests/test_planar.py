"""Filled regions of mesh sections (pokefit/planar.py): nesting depth, crossing loops, many bodies."""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon, box

from pokefit import planar
from pokefit.planar import _nested, layer_regions, regions


class _Section:
    """Stand-in for a trimesh Path2D: just its closed loops."""

    def __init__(self, loops: list[Polygon]) -> None:
        self.polygons_closed = loops


def test_nesting_depth_makes_holes_and_islands() -> None:
    # outer square, a hole in it, an island in the hole
    loops = [box(0, 0, 10, 10), box(2, 2, 8, 8), box(4, 4, 6, 6)]
    got = regions(_Section(loops))
    areas = sorted(round(p.area, 6) for p in got)
    assert areas == [4.0, 64.0]                    # the island 2x2, the ring 100-36
    ring = max(got, key=lambda p: p.area)
    assert len(ring.interiors) == 1
    assert _nested(loops) is not None              # the fast path handled it


def test_crossing_loops_fall_back_to_xor() -> None:
    loops = [box(0, 0, 4, 4), box(2, 2, 6, 6)]     # overlapping, crossing boundaries
    assert _nested(loops) is None
    got = regions(_Section(loops))
    assert abs(sum(p.area for p in got) - (16 + 16 - 2 * 4)) < 1e-9   # even-odd: the overlap is out


def test_overlapping_bodies_are_sectioned_one_by_one(monkeypatch) -> None:
    # Many separate, overlapping closed boxes in one mesh (a LEGO set): every layer's loops cross.
    boxes = [trimesh.creation.box((4, 4, 4)).apply_translation([x, 0.0, 2.0]) for x in np.arange(0, 12, 1.5)]
    mesh = trimesh.util.concatenate(boxes)
    heights = [0.5, 1.5, 2.5, 3.5]
    monkeypatch.setattr(planar, "CROSSING_LAYERS", 1)
    per = layer_regions(mesh, heights)
    # Each layer holds every body's own square (8 of them), not an even-odd tangle of the overlaps.
    assert all(len(r) == 8 for r in per)
    assert all(abs(sum(p.area for p in r) - 8 * 16) < 1e-6 for r in per)
    single = layer_regions(trimesh.creation.box((4, 4, 4)).apply_translation([0, 0, 2]), heights)
    assert all(len(r) == 1 and abs(r[0].area - 16) < 1e-6 for r in single)
