"""Filled regions of planar mesh sections, without trimesh's rtree-based `polygons_full`.

A section of one closed solid is a set of non-crossing closed loops; nesting
alternates material and hole, so the region is the even-odd (XOR) combination
of the loops. This needs only shapely, which keeps the engine runnable in the
browser (Pyodide has no rtree). Call it per solid part: two overlapping parts
XORed together would turn their overlap into a hole.
"""
from __future__ import annotations

from functools import reduce

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def regions(path: object) -> list[Polygon]:
    """The filled polygons (with holes) of a trimesh `Path2D` section."""
    loops = [p for p in getattr(path, "polygons_closed", []) if p is not None and not p.is_empty]
    loops = [p if p.is_valid else p.buffer(0) for p in loops]
    loops = [p for p in loops if not p.is_empty and p.area > 1e-12]
    if not loops:
        return []
    filled: BaseGeometry = reduce(lambda a, b: a.symmetric_difference(b), loops)
    if isinstance(filled, Polygon):
        return [filled]
    if isinstance(filled, MultiPolygon):
        return list(filled.geoms)
    return [g for g in getattr(filled, "geoms", []) if isinstance(g, Polygon)]
