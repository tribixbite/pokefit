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
from shapely import make_valid, set_precision
from shapely.errors import GEOSException

GRID = 1e-6                             # mm: snap grid for a retry when GEOS finds unnoded crossings


def _xor(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """Even-odd combination of two regions. Loops of a messy (non-manifold, self-touching) mesh can
    meet at near-coincident points that GEOS rejects as a non-noded intersection; snapping both to
    a micrometre grid resolves those without visibly moving anything."""
    try:
        return a.symmetric_difference(b)
    except GEOSException:
        pass
    # Repair both sides (keeping only areas: a repair can leave lines and points), then
    # retry exactly and on the grid. Loops that still conflict are self-intersecting
    # shells of a non-manifold mesh, where even-odd has no clear meaning: take the union.
    a, b = _areas(make_valid(a)), _areas(make_valid(b))
    for grid in (0.0, GRID):
        try:
            return a.symmetric_difference(b, grid_size=grid or None)
        except GEOSException:
            continue
    return _areas(set_precision(a, GRID)).union(_areas(set_precision(b, GRID)), grid_size=GRID)


def _areas(g: BaseGeometry) -> BaseGeometry:
    """The polygonal part of a geometry (a snapped region can also hold lines and points)."""
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    polys = [x for x in getattr(g, "geoms", []) if isinstance(x, (Polygon, MultiPolygon))]
    flat = [q for x in polys for q in (x.geoms if isinstance(x, MultiPolygon) else [x])]
    return MultiPolygon(flat) if flat else Polygon()


def regions(path: object) -> list[Polygon]:
    """The filled polygons (with holes) of a trimesh `Path2D` section."""
    loops = [p for p in getattr(path, "polygons_closed", []) if p is not None and not p.is_empty]
    loops = [p if p.is_valid else p.buffer(0) for p in loops]
    loops = [p for p in loops if not p.is_empty and p.area > 1e-12]
    if not loops:
        return []
    filled: BaseGeometry = reduce(_xor, loops)
    if isinstance(filled, Polygon):
        return [filled]
    if isinstance(filled, MultiPolygon):
        return list(filled.geoms)
    return [g for g in getattr(filled, "geoms", []) if isinstance(g, Polygon)]
