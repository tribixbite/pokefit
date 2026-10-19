"""Filled regions of planar mesh sections, without trimesh's rtree-based `polygons_full`.

A section of one closed solid is a set of non-crossing closed loops; nesting
alternates material and hole, so the region is the even-odd (XOR) combination
of the loops. This needs only shapely, which keeps the engine runnable in the
browser (Pyodide has no rtree). Call it per solid part: two overlapping parts
XORed together would turn their overlap into a hole.
"""
from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
import shapely
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


def _nested(loops: list[BaseGeometry]) -> list[Polygon] | None:
    """Even-odd fill of loops that do not cross, by nesting depth: a loop inside an odd number of others is
    a hole of the innermost one. One spatial-index query instead of a running XOR, whose cost grows with
    every loop folded in (a 684k-triangle LEGO set: 241 s of `symmetric_difference` over 55 slices).
    None when any two loops cross (`overlaps`: interiors meet, neither holds the other); then the robust
    XOR decides."""
    pieces = [q for g in loops for q in (g.geoms if isinstance(g, MultiPolygon) else [g]) if isinstance(q, Polygon)]
    pieces = [Polygon(q.exterior) for q in pieces if not q.is_empty and q.area > 1e-12]
    if len(pieces) < 2:
        return pieces
    geoms = np.array(pieces, dtype=object)
    tree = shapely.STRtree(geoms)
    if tree.query(geoms, predicate="overlaps").size:
        return None                                    # crossing loops: no clean nesting
    area = shapely.area(geoms)
    points = shapely.point_on_surface(geoms)
    inner, outer = tree.query(points, predicate="within")   # point of loop i inside loop j
    keep = (inner != outer) & (area[outer] >= area[inner])
    inner, outer = inner[keep], outer[keep]
    depth = np.bincount(inner, minlength=len(pieces))
    # Innermost container (smallest area) of each nested loop: its parent.
    parent = np.full(len(pieces), -1)
    best = np.full(len(pieces), np.inf)
    for i, j in zip(inner.tolist(), outer.tolist()):
        if area[j] < best[i]:
            best[i], parent[i] = area[j], j
    out: list[Polygon] = []
    for i in np.nonzero(depth % 2 == 0)[0].tolist():
        holes = [pieces[c] for c in np.nonzero(parent == i)[0].tolist()]
        region = pieces[i] if not holes else pieces[i].difference(shapely.union_all(holes))
        out += [g for g in getattr(region, "geoms", [region]) if isinstance(g, Polygon) and not g.is_empty]
    return out


def _loops(path: object) -> list[BaseGeometry]:
    loops = [p for p in getattr(path, "polygons_closed", []) if p is not None and not p.is_empty]
    loops = [p if p.is_valid else p.buffer(0) for p in loops]
    return [p for p in loops if not p.is_empty and p.area > 1e-12]


def nested_regions(path: object) -> list[Polygon] | None:
    """`regions` by nesting depth only; None when the section's loops cross (see `_nested`)."""
    loops = _loops(path)
    return _nested(loops) if loops else []


CROSSING_LAYERS = 3     # layer_regions: more layers than this with crossing loops -> section body by body


def layer_regions(mesh: Any, heights: list[float]) -> list[list[Polygon]]:
    """Filled regions of horizontal sections of `mesh` at `heights` (z), one list per height.

    A mesh made of many overlapping closed shells (a LEGO set: 16k bricks in one part) has sections
    whose loops cross, where even-odd is meaningless and the running XOR is slow (197 s for 40
    layers). If more than CROSSING_LAYERS layers cross, every connected body is sectioned on its own
    (its loops nest) and a layer's regions are all bodies' regions: 9 s there.
    """
    sections = mesh.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=list(heights))
    out: list[list[Polygon] | None] = [nested_regions(s) if s is not None else [] for s in sections]
    if sum(r is None for r in out) > CROSSING_LAYERS:
        bodies = mesh.split(only_watertight=False)
        if len(bodies) > 1:
            per: list[list[Polygon]] = [[] for _ in heights]
            z = np.asarray(heights, np.float64)
            for b in bodies:
                lo, hi = float(b.bounds[0][2]), float(b.bounds[1][2])
                idx = np.nonzero((z > lo) & (z < hi))[0]
                if not len(idx):
                    continue
                for i, sec in zip(idx.tolist(), b.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1],
                                                                     heights=z[idx].tolist())):
                    if sec is not None:
                        per[i] += regions(sec)
            return per
    return [r if r is not None else regions(s) for r, s in zip(out, sections)]


def regions(path: object) -> list[Polygon]:
    """The filled polygons (with holes) of a trimesh `Path2D` section."""
    loops = _loops(path)
    if not loops:
        return []
    fast = _nested(loops)
    if fast is not None:
        return fast
    filled: BaseGeometry = reduce(_xor, loops)
    if isinstance(filled, Polygon):
        return [filled]
    if isinstance(filled, MultiPolygon):
        return list(filled.geoms)
    return [g for g in getattr(filled, "geoms", []) if isinstance(g, Polygon)]
