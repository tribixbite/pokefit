"""Magnet pockets in a figure exported by pokefit before, found again when it is opened for another edit.

An export has its pockets cut (`magnets.pocket_solid`: a vertical cylinder from below the bed up to a flat ceiling)
and carries the conform step's flat base and the magnet pads as parts named "flat base". Opened again as it is, the
next export would cut new pockets beside the old holes and add a second flat base. So, when a file loads:

  * each pocket is found from its shape: a flat, round, downward-facing ceiling above the bed (radius 0.75-5 mm,
    its outline a circle; where a pocket was cut through the flat base into the figure its ceiling is split between
    the two parts, so ceilings are joined across parts as 2-D shapes), ringed all the way round by vertical walls
    that face its axis;
  * its ceiling and walls are taken out and the opening is closed (the load repair's hole fill, `repair._fill`:
    flat where the underside round it was flat, as it is under a pocket), so the figure is solid again;
  * the "flat base" parts are dropped, as the settings make them again;
  * the pockets' centres are kept (`FigureObject.old_pockets`), and the UI places the magnets there, as pockets
    placed by hand: they can be dragged, removed, or replaced by automatic placement.

Only pokefit's own pocket shape is matched; a round recess with a rounded top, sloped walls or no full ring of wall
is left alone (checked on 4 original figures: none found).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import Polygon, unary_union

from .model3mf import Part

FILL_NAME = "flat base"      # conform.FILL_NAME (not imported: conform imports more than a loader needs)
R_MIN, R_MAX = 0.75, 5.0     # pocket radius, mm (default 4.18 mm across; smaller or larger magnets set by hand)
FLAT = 0.999                 # a ceiling triangle faces down within ~2.5 degrees
UPRIGHT = 0.02               # a wall triangle is vertical within ~1 degree
ROUND = 0.06                 # ceiling outline and walls lie this close (share of the radius) to the circle
ABOVE_BED = 0.3              # a ceiling at least this high above the figure's lowest point (not the bed contact)
LEVEL = 0.01                 # ceiling triangles within this height of each other are one level, mm
RING_BINS = 24               # the walls must face the axis from every one of these directions
MIN_DEPTH = 0.5              # mm from the opening to the ceiling


@dataclass(frozen=True)
class OldPocket:
    """A pocket found in a loaded figure, in the file's plate frame (z up, mm)."""
    x: float
    y: float
    z: float           # where it opens on the underside
    radius: float
    depth: float       # opening to ceiling

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3),
                "diameter": round(2 * self.radius, 3), "depth": round(self.depth, 3)}


def _normals(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    out = np.zeros_like(n)
    ok = a > 1e-12
    out[ok] = n[ok] / a[ok, None]
    return out


def find_pockets(meshes: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[OldPocket, list[np.ndarray]]]:
    """Pockets in closed meshes (vertices in a z-up frame), each with the faces it is made of (ceiling and walls),
    per mesh."""
    tris = [v[f] for v, f in meshes]
    if not tris or not sum(len(t) for t in tris):
        return []
    z0 = min(float(t[:, :, 2].min()) for t in tris if len(t))
    normals = [_normals(t) for t in tris]
    # Downward faces above the bed, from every mesh, sorted by height and cut into levels.
    down = [np.flatnonzero((n[:, 2] < -FLAT) & (t[:, :, 2].min(axis=1) > z0 + ABOVE_BED)) for t, n in zip(tris, normals)]
    mesh_id = np.concatenate([np.full(len(d), i) for i, d in enumerate(down)]).astype(np.int64)
    face_id = np.concatenate(down).astype(np.int64)
    if not len(face_id):
        return []
    zs = np.asarray([tris[m][fi, :, 2].mean() for m, fi in zip(mesh_id.tolist(), face_id.tolist())])
    order = np.argsort(zs)
    cuts = np.flatnonzero(np.diff(zs[order]) > LEVEL) + 1
    out: list[tuple[OldPocket, list[np.ndarray]]] = []
    for level in np.split(order, cuts):
        if len(level) < 4:
            continue
        top = float(zs[level].mean())
        polys = [Polygon(tris[m][fi, :, :2]) for m, fi in zip(mesh_id[level].tolist(), face_id[level].tolist())]
        merged = unary_union([p.buffer(1e-4) for p in polys if p.is_valid and p.area > 0])
        for disc in getattr(merged, "geoms", [merged]):
            if disc.is_empty or len(disc.interiors):
                continue
            c = np.asarray(disc.centroid.coords[0])
            rim = np.hypot(*(np.asarray(disc.exterior.coords) - c).T)
            r = float(rim.max())
            if not (R_MIN <= r <= R_MAX) or abs(disc.area - np.pi * r * r) > 0.06 * np.pi * r * r:
                continue
            if np.abs(rim - r).max() > ROUND * r:
                continue
            found = _walls(tris, normals, c, r, top)
            if found is None:
                continue
            bottom, walls = found
            faces = []
            for i, (t, d) in enumerate(zip(tris, down)):
                inside = np.hypot(*(t[d].mean(axis=1)[:, :2] - c).T) < r
                ceiling = d[inside & (np.abs(t[d, :, 2].mean(axis=1) - top) <= LEVEL)]
                faces.append(np.concatenate([ceiling, walls[i]]))
            out.append((OldPocket(float(c[0]), float(c[1]), bottom, r, top - bottom), faces))
    return out


def _walls(tris: list[np.ndarray], normals: list[np.ndarray], c: np.ndarray, r: float,
           top: float) -> tuple[float, list[np.ndarray]] | None:
    """(opening height, wall faces per mesh) of the cylinder of radius `r` round `c` under a ceiling at `top`:
    vertical faces on the circle, facing its axis, below the ceiling, in every direction round it; None if not."""
    walls: list[np.ndarray] = []
    seen = np.zeros(RING_BINS, bool)
    bottom = top
    for t, n in zip(tris, normals):
        d = t.mean(axis=1)[:, :2] - c
        ring = ((np.abs(n[:, 2]) < UPRIGHT) & (np.abs(np.hypot(*d.T) - r) < 2 * ROUND * r)
                & (np.einsum("ij,ij->i", n[:, :2], -d) > 0) & (t[:, :, 2].max(axis=1) <= top + LEVEL))
        if ring.any():
            bottom = min(bottom, float(t[ring, :, 2].min()))
            ang = np.arctan2(d[ring, 1], d[ring, 0])
            seen[((ang + np.pi) / (2 * np.pi) * RING_BINS).astype(int) % RING_BINS] = True
        walls.append(np.flatnonzero(ring))
    if not seen.all() or top - bottom < MIN_DEPTH:
        return None
    return bottom, walls


def strip_pockets(parts: list[Part], transform: np.ndarray) -> tuple[list[Part], list[OldPocket], int]:
    """(parts with old pockets taken out and "flat base" parts dropped, the pockets in the plate frame given by
    `transform` (the build item's), how many flat-base parts were dropped). The holes left are closed by the
    load repair that runs next. Parts are returned unchanged when no pocket is found."""
    solid = [i for i, p in enumerate(parts) if p.subtype == "normal_part" and len(p.faces)]
    world = [(parts[i].vertices @ transform[:3, :3].T + transform[:3, 3], parts[i].faces) for i in solid]
    hits = find_pockets(world)
    if not hits:
        return parts, [], 0
    drop = {i: np.zeros(len(parts[i].faces), bool) for i in solid}
    for _, faces in hits:
        for i, fs in zip(solid, faces):
            drop[i][fs] = True
    new: list[Part] = []
    for i, p in enumerate(parts):
        if i not in drop or not drop[i].any():
            new.append(p)
            continue
        keep = np.flatnonzero(~drop[i])
        extra = {j: p.tri_extra[k] for j, k in enumerate(keep.tolist()) if k in p.tri_extra}
        new.append(Part(p.name, p.extruder, p.vertices, p.faces[keep], extra, p.subtype))
    # The flat base and pads of that export: the settings make them again (with the pockets where they are now).
    bases = [p for p in new if p.name == FILL_NAME]
    if bases and len(bases) < len(new):
        new = [p for p in new if p.name != FILL_NAME]
    else:
        bases = []
    return new, [pk for pk, _ in hits], len(bases)
