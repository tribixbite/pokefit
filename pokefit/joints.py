"""Weak points: where a thin feature (an ear, a tail, a claw, a leaf on a stalk) would snap off, and how hard.

Why: a figure scaled into a ~27-34 mm ball has ears and tails a millimetre or two across. "Thin" alone does not say
what breaks: a thin flat leaf held along its whole edge is sturdy, a thicker tail on a long lever is not. What breaks
a feature is bending at its narrowest section, under a push at its far end.

How (all parts' solid together, on a VOXEL grid at print size: `strengthen.voxels`):

1. **Thin features.** The body is what a ball PROBE mm across fits inside (a morphological opening: two distance
   transforms); the rest of the solid falls into thin features (connected voxels), each joined to the body at its
   root.
2. **Sections.** Each feature's geodesic distance from its root (Dijkstra over voxel neighbours, inside the feature)
   cuts it into slabs STEP thick. A slab's voxels projected onto the plane across the feature (normal: from the slab
   to the next one), one per voxel-sized cell, are the cross-section there: its area, second moments and narrowest
   width. (Volume / STEP, the coarea formula, was used first: how many voxel layers a slab holds varies with where
   it starts, and some sections of a straight rod read 30 % small.)
3. **Breaking force.** A push F at the far end bends the section by F x lever: the stress at its edge is
   F x lever / Z (Z = I / c, the section modulus about its weak axis). It breaks at the material's strength:
   F = sigma x Z / lever. The lever is the farthest point of the piece beyond the section (the voxels past it that
   it connects to), measured across the weak axis. sigma is SIGMA_XY when the section is upright (the stress runs
   along the layers) down to SIGMA_Z when it lies flat (the stress pulls layers apart: an upright ear snaps between
   two layers).
4. **One entry per break.** The weakest section of each feature is reported. A later (stronger) section whose piece
   overlaps one already reported is the same break and is skipped, unless its piece is NEST x bigger: a different
   one (a whole ear behind its fragile tip). The stretch past a reported section that is still under GOAL newtons
   (`_weak_stretch`) is what a single-feature thickening covers (strengthen.Local).

Grades (`GRADES`) by that force: A holds about 2 kg pushed at its tip, D under 300 g, F under 100 g, or a section under
MIN_WIDTH across (two extrusion lines: the slicer may leave gaps, and it snaps taking supports off). The strengths
are typical PETG/PLA values for well-tuned prints (bending ~40 MPa along the layers, about half across them): the
forces are estimates for comparing features and choosing what to thicken, not a guarantee. Pieces under MIN_VOLUME
(a fine point, a serration) are not listed: losing one is cosmetic.

Limits: a feature thicker than PROBE is not examined (a 3 mm rod 15 mm long already holds ~70 N); a neck (thin between
two thick parts, e.g. a thin leg) is measured as the piece between its ends, so its lever is short; sections are
measured to a voxel (VOXEL mm), so widths under ~0.5 mm are rough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .model3mf import REINFORCE_NAME, Part
from .strengthen import Voxels, voxels

PROBE = 3.0          # mm at print size: what a ball this wide fits in is the body; the rest is thin features
VOXEL = 0.15         # mm at print size
STEP = 0.3           # mm at print size between sections along a feature (at least)
MAX_SECTIONS = 40    # ... and at most this many per feature (a long tail: wider steps, then refined)
REFINE = 3           # coarse levels refined to STEP round them, the weakest first
MIN_LENGTH = 1.0     # mm at print size: shorter features (bumps, rims, relief) are not examined
MIN_VOLUME = 1.0     # mm³ at print size: smaller pieces (points, serrations) are not listed
SIGMA_XY = 40.0      # MPa (N/mm²): bending strength with the stress along the layers
SIGMA_Z = 20.0       # MPa: across the layers (a section lying flat)
MIN_WIDTH = 0.8      # mm: a section narrower than this is grade F whatever its force
GRADES = ((20.0, "A"), (8.0, "B"), (3.0, "C"), (1.0, "D"))   # N at the far end: at least this, else the next; F below
ORDER = "FDCBA"      # weakest first
NEST = 3.0           # a piece this many times bigger than an overlapping one already listed is a different break
MAX_REPORTED = 16
GOAL = GRADES[2][0]  # N: a feature is thickened up to where it already holds this (grade C) (`_weak_stretch`)
_NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                        if (i, j, k) > (0, 0, 0)])        # 13 of the 26: each neighbour pair once
_CUBE = np.ones((3, 3, 3), bool)


@dataclass
class WeakPoint:
    """The weakest section of one thin feature. Lengths in mm at print size, positions in the figure frame (scale 1:
    the parts' own coordinates)."""
    part: int                   # index into the parts measured (the one most of the piece is in)
    grade: str                  # A..F (`GRADES`)
    force_n: float              # estimated push at the far end that breaks it
    width_mm: float             # the section's narrowest width (across its weak axis)
    depth_mm: float             # ... and its widest
    area_mm2: float
    lever_mm: float             # from the section to the farthest point beyond it, across the weak axis
    length_mm: float            # how far the feature reaches past the section
    volume_mm3: float           # the piece that breaks off
    flat: float                 # 0 = the section is upright (stress along the layers) .. 1 = flat (across them)
    centre: list[float]         # the section's centre
    normal: list[float]         # across the section, toward the far end
    face: int                   # a triangle of the piece (of `part`): a handle for per-feature settings
    faces: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.int64))   # the piece's, of `part`
    cells: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.int64))   # its voxels (flat)
    end: list[float] = field(default_factory=list)   # the centre of the last section past it still under GOAL N
    weak_mm: float = 0.0        # ... that far past it along the feature (0: only this section is under GOAL)

    def to_dict(self) -> dict[str, Any]:
        return {"part": self.part, "grade": self.grade, "force_n": round(self.force_n, 2),
                "width_mm": round(self.width_mm, 2), "depth_mm": round(self.depth_mm, 2),
                "area_mm2": round(self.area_mm2, 2), "lever_mm": round(self.lever_mm, 2),
                "length_mm": round(self.length_mm, 2), "volume_mm3": round(self.volume_mm3, 2),
                "flat": round(self.flat, 2), "centre": [round(float(x), 4) for x in self.centre],
                "normal": [round(float(x), 4) for x in self.normal], "face": self.face, "faces": int(len(self.faces)),
                "end": [round(float(x), 4) for x in self.end], "weak_mm": round(self.weak_mm, 2)}


def grade(force: float, width: float) -> str:
    if width < MIN_WIDTH:
        return "F"
    for at_least, g in GRADES:
        if force >= at_least:
            return g
    return "F"


@dataclass
class _Section:
    area: float                 # mm²
    centre: np.ndarray          # print mm
    normal: np.ndarray
    width: float
    depth: float
    z_weak: float               # section moduli (mm³)
    z_strong: float
    thin: np.ndarray            # in-plane directions (3D): across the narrow way, and along the long way
    long: np.ndarray


def _section(pts: np.ndarray, normal: np.ndarray, h: float) -> _Section | None:
    """The cross-section the slab voxels `pts` (print mm, centres, `h` apart) make across `normal`."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    # The slab's voxels projected onto the section plane, one per h x h cell: how many voxel layers a geodesic slab
    # holds varies with where it starts (26-neighbour steps), and counting voxels (n h³ / thickness) read some
    # sections 30 % small (a dip in a straight rod's strength that was not in the rod).
    r0 = pts - pts.mean(axis=0)
    cells = np.unique(np.round(np.c_[r0 @ u, r0 @ w] / h).astype(np.int64), axis=0)
    xy = cells * h
    centre = pts.mean(axis=0) + (xy.mean(axis=0) @ np.vstack([u, w]))
    xy = xy - xy.mean(axis=0)
    area = h * h * len(cells)
    m = h * h * (xy.T @ xy) + np.eye(2) * area * h * h / 12           # each cell's own spread
    vals, vecs = np.linalg.eigh(m)
    if vals[0] <= 0:
        return None
    across, along = xy @ vecs[:, 0], xy @ vecs[:, 1]
    c_thin, c_long = np.abs(across).max() + h / 2, np.abs(along).max() + h / 2
    return _Section(area, centre, n, float(np.ptp(across) + h), float(np.ptp(along) + h), float(vals[0] / c_thin),
                    float(vals[1] / c_long), vecs[0, 0] * u + vecs[1, 0] * w, vecs[0, 1] * u + vecs[1, 1] * w)


def _strength(sec: _Section, beyond: np.ndarray) -> tuple[float, float, float]:
    """(force N, lever mm, flatness) for a section and the points (print mm) of the piece beyond it."""
    flat = float(sec.normal[2] ** 2)
    sigma = SIGMA_XY + (SIGMA_Z - SIGMA_XY) * flat
    r = beyond - sec.centre
    # Bending the weak way (about the long axis) is done by a push whose lever is the distance across that axis.
    lever_weak = float(np.linalg.norm(r - np.outer(r @ sec.long, sec.long), axis=1).max(initial=0.0))
    lever_strong = float(np.linalg.norm(r - np.outer(r @ sec.thin, sec.thin), axis=1).max(initial=0.0))
    f_weak = sigma * sec.z_weak / max(lever_weak, 1e-6)
    f_strong = sigma * sec.z_strong / max(lever_strong, 1e-6)
    return (f_weak, lever_weak, flat) if f_weak <= f_strong else (f_strong, lever_strong, flat)


def _geodesic(mask: np.ndarray, roots: np.ndarray, h: float) -> np.ndarray:
    """Per voxel of `mask` (a crop), its distance (mm) from the nearest of `roots` through `mask`'s voxels (26
    neighbours); inf where none reaches, and outside the mask."""
    ids = np.full(mask.shape, -1, np.int64)
    n = int(mask.sum())
    ids[mask] = np.arange(n)
    rows, cols, wts = [], [], []
    for off in _NEIGHBOURS:
        a = tuple(slice(max(0, -o), mask.shape[ax] - max(0, o)) for ax, o in enumerate(off))
        b = tuple(slice(max(0, o), mask.shape[ax] - max(0, -o)) for ax, o in enumerate(off))
        both = mask[a] & mask[b]
        rows.append(ids[a][both])
        cols.append(ids[b][both])
        wts.append(np.full(int(both.sum()), h * float(np.linalg.norm(off))))
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    g = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    dist = np.full(mask.shape, np.inf)
    src = ids[roots & mask]
    if len(src):
        dist[mask] = dijkstra(g, directed=False, indices=src, min_only=True)
    return dist


def thin_parts(grid: Voxels, probe: float) -> tuple[np.ndarray, np.ndarray]:
    """The grid's thin features (module doc, 1): labels (0 = body or outside, 1.. one per connected feature) and
    the feature voxels touching the body (their roots). `probe` in the grid's units."""
    inside = grid.inside
    radius = probe / 2 / grid.h                                    # in voxels
    eroded = ndimage.distance_transform_edt(inside) > radius
    if not eroded.any():                                           # nothing is thick: no body to break off from
        return np.zeros(inside.shape, np.int32), np.zeros(inside.shape, bool)
    body = inside & (ndimage.distance_transform_edt(~eroded) <= radius)
    thin = inside & ~body
    labels, _ = ndimage.label(thin, _CUBE)
    return labels, ndimage.binary_dilation(body, _CUBE) & thin


def _grown(sl: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
    """A crop one voxel bigger on every side (inside the grid)."""
    return tuple(slice(max(0, s.start - 1), min(dim, s.stop + 1)) for s, dim in zip(sl, shape))


def cell_under(grid: Voxels, points: np.ndarray, normals: np.ndarray, within: np.ndarray) -> np.ndarray:
    """Per surface point (with its outward normal), the flat index of the first cell of `within` (a bool grid) a
    half, one and a half or two and a half cells under it; -1 where none is."""
    cells = np.full(len(points), -1, np.int64)
    for depth in (0.5, 1.5, 2.5):
        idx = grid.index(points - normals * depth * grid.h)
        ok = (cells < 0) & within[idx]
        cells[ok] = np.ravel_multi_index(idx, within.shape)[ok]
    return cells


def feature_along(grid: Voxels, probe: float, face: tuple[np.ndarray, np.ndarray],
                  root: tuple[np.ndarray, np.ndarray], back: float) -> np.ndarray | None:
    """The thin feature (`thin_parts` at `probe`) holding the surface point `face` (point, outward normal), from
    `back` (grid units) behind the section through the surface point `root` outward: per cell, its distance past that
    section along the feature (negative behind it), NaN elsewhere. The same feature `weak_points` names by its
    triangles (`face` of the piece, `root` on the section), found again in any orientation; only the branch `face` is
    on (a fork's other branches past the same distance are left out). None when `face` is not on a thin feature."""
    labels, roots = thin_parts(grid, probe)
    fc = cell_under(grid, face[0][None], face[1][None], labels > 0)[0]
    if fc < 0:
        return None
    lab = labels.flat[fc]
    sl = _grown(ndimage.find_objects((labels == lab).astype(np.int8))[0], labels.shape)
    mask = labels[sl] == lab
    dist = _geodesic(mask, roots[sl] & mask, grid.h)
    start = np.array([q.start for q in sl])

    def in_crop(cell: int) -> tuple[int, ...]:
        return tuple(int(x) for x in np.array(np.unravel_index(cell, labels.shape)) - start)

    # The section's distance from the root: where the root triangle is (its feature's cell under it; none: the root).
    rc = cell_under(grid, root[0][None], root[1][None], labels == lab)[0]
    s = float(dist[in_crop(rc)]) if rc >= 0 else 0.0
    s = s if np.isfinite(s) else 0.0
    keep = mask & (dist >= s - back)
    pieces, _ = ndimage.label(keep, _CUBE)
    pid = pieces[in_crop(fc)]
    out = np.full(labels.shape, np.nan, np.float32)
    sel = pieces == pid if pid > 0 else keep
    crop = out[sl]
    crop[sel] = dist[sel] - s
    return out


def weak_points(parts: list[Part], scale: float, probe: float = PROBE) -> list[WeakPoint]:
    """The weakest section of each thin feature of the solid `parts` (figure frame, scale 1) printed at `scale`,
    weakest first (module doc). Empty when nothing is thinner than `probe` mm."""
    return _graded(parts, scale, probe)[1]


def _graded(parts: list[Part], scale: float, probe: float) -> tuple[Voxels | None, list[WeakPoint]]:
    """`weak_points` and the grid (print mm) their `cells` index."""
    solid = [i for i, p in enumerate(parts) if p.subtype == "normal_part" and len(p.faces)]
    if scale <= 0 or not solid:
        return None, []
    grid = voxels([parts[i].vertices * scale for i in solid], [parts[i].faces for i in solid], VOXEL)
    labels, roots = thin_parts(grid, probe)
    cands: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sl = _grown(sl, labels.shape)
        mask = labels[sl] == lab
        root = roots[sl] & mask
        if not root.any():
            continue                                                 # loose: the islands check's business
        for key, cand in _feature(grid, sl, mask, _geodesic(mask, root, grid.h)):
            cand["feature"] = lab
            cands.append((key, cand))
    cands.sort(key=lambda t: t[0])
    found: list[dict[str, Any]] = []
    for _, c in cands:
        # Weakest first, so a later section overlapping a listed piece is stronger: the same break unless its piece is
        # NEST x bigger (the whole ear behind its fragile tip); a smaller piece further out on it says nothing new.
        size = len(c["cells"])
        if any(np.intersect1d(c["cells"], t["cells"], assume_unique=True).size and size < NEST * len(t["cells"])
               for t in found):
            continue
        found.append(c)
        if len(found) >= MAX_REPORTED:
            break
    for c in found:
        c["end"] = _weak_stretch(c, [d for _, d in cands if d["feature"] == c["feature"]])
    return grid, _with_faces(found, parts, solid, grid, scale)


def _weak_stretch(c: dict[str, Any], same: list[dict[str, Any]]) -> dict[str, Any]:
    """The farthest section past `c` along its branch (sections of the same feature whose piece is inside c's) that
    is still under GOAL newtons: thickening from c's section to there brings the whole stretch up, and nothing past it
    (the rest is strong already: a tail's leaf past its thin stalk keeps its shape). `c` itself when none is."""
    end = c
    for d in same:
        if d["s"] > end["s"] and d["force"] < GOAL and np.isin(d["cells"], c["cells"], assume_unique=True).all():
            end = d
    return end


def _feature(grid: Voxels, sl: tuple[slice, ...], mask: np.ndarray, dist: np.ndarray) -> list[tuple[Any, dict]]:
    """Every section of one feature (a crop `sl` of the grid, its voxels `mask`, their distance from its root) with
    its strength and the piece beyond it."""
    h = grid.h
    reach = float(dist[mask & np.isfinite(dist)].max(initial=0.0))
    if reach < MIN_LENGTH:
        return []
    offset = np.array([s.start for s in sl])
    slab_w = STEP                                                    # each section's slab is STEP thick

    def sections(s: float) -> list[tuple[Any, dict]]:
        """The sections at distance `s` from the root (one per slab: a fork has one per branch)."""
        got: list[tuple[Any, dict]] = []
        beyond = mask & (dist >= s)
        pieces, _ = ndimage.label(beyond, _CUBE)
        slab = beyond & (dist < s + slab_w)
        nxt = beyond & (dist >= s + slab_w) & (dist < s + 2 * slab_w)
        # One section per piece: a piece held by several slabs (a loop, two legs touching) is held by all of them
        # together, so their cells make one section (one slab alone read as a one-voxel break).
        for piece_id in np.unique(pieces[slab]):
            cells = np.argwhere(slab & (pieces == piece_id))
            piece = np.argwhere(pieces == piece_id)
            volume = len(piece) * h ** 3
            if volume < MIN_VOLUME:
                continue
            pts = grid.centres(tuple((cells + offset).T))
            ahead = np.argwhere(nxt & (pieces == piece_id))
            if len(ahead):
                normal = grid.centres(tuple((ahead + offset).T)).mean(axis=0) - pts.mean(axis=0)
            else:                                                    # the tip: the slab's thinnest direction
                normal = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))[1][:, 0]
            if np.linalg.norm(normal) < 1e-9:
                continue
            sec = _section(pts, normal, h)
            if sec is None:
                continue
            force, lever, flat = _strength(sec, grid.centres(tuple((piece + offset).T)))
            g = grade(force, sec.width)
            flat_cells = np.ravel_multi_index(tuple((piece + offset).T), grid.inside.shape)
            # Narrower than MIN_WIDTH ranks first among the F grades, the narrowest first.
            key = (ORDER.index(g), force if sec.width >= MIN_WIDTH else -1.0 / max(sec.width, 1e-6))
            got.append((key, {"grade": g, "force": force, "sec": sec, "lever": lever, "flat": flat, "s": s,
                              "length": float(dist[tuple(piece.T)].max()) - s, "volume": volume,
                              "cells": np.sort(flat_cells)}))
        return got

    # Coarse (at most MAX_SECTIONS levels), then every STEP round the REFINE weakest coarse levels.
    step = max(STEP, reach / MAX_SECTIONS)
    out: list[tuple[Any, dict]] = []
    for s in np.arange(step, reach - MIN_LENGTH / 2, step):
        out.extend(sections(float(s)))
    if step > STEP:
        done = {round(c["s"], 6) for _, c in out}
        weakest = sorted({c["s"] for _, c in sorted(out, key=lambda t: t[0])[:REFINE]})
        for s0 in weakest:
            for s in np.arange(s0 - step + STEP, s0 + step - STEP / 2, STEP):
                if STEP <= s < reach - MIN_LENGTH / 2 and round(float(s), 6) not in done:
                    done.add(round(float(s), 6))
                    out.extend(sections(float(s)))
    return out


def _with_faces(found: list[dict[str, Any]], parts: list[Part], solid: list[int], grid: Voxels,
                scale: float) -> list[WeakPoint]:
    """WeakPoints with their pieces' triangles: a triangle is in a piece when the voxel just under its centre is."""
    face_cells: dict[int, np.ndarray] = {}
    for i in solid:
        tri = parts[i].vertices[parts[i].faces] * scale
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        face_cells[i] = cell_under(grid, tri.mean(axis=1), nrm, grid.inside)
    out: list[WeakPoint] = []
    for c in found:
        hits = {i: np.flatnonzero(np.isin(fc, c["cells"])) for i, fc in face_cells.items()}
        part = max(hits, key=lambda i: len(hits[i]))
        pf = hits[part]
        sec = c["sec"]
        end = c["end"]["sec"]
        out.append(WeakPoint(part, c["grade"], c["force"], sec.width, sec.depth, sec.area, c["lever"], c["length"],
                             c["volume"], c["flat"], [float(x) for x in sec.centre / scale],
                             [float(x) for x in sec.normal], int(pf[0]) if len(pf) else -1, pf, c["cells"],
                             [float(x) for x in end.centre / scale], float(c["end"]["s"] - c["s"])))
    return out


# Modifier volumes over weak features (`reinforcements`): Bambu prints the solid inside them with these settings. Thin
# features are mostly walls already; past two walls a side (~1.7 mm) the rest is sparse infill, which these fill.
REINFORCE = {"wall_loops": "6", "sparse_infill_density": "100%"}
REINFORCE_GRADES = "FDC"     # grades reinforced
REINFORCE_BACK = 1.5         # mm the volume reaches past the piece (over its weak section, into the body round the join)
REINFORCE_FACES = 4000       # triangles per volume (its exact shape does not matter: it only marks a region)
REINFORCE_ERROR = 0.1        # mm the simplified volume may move


def stronger(process: dict[str, Any] | None) -> dict[str, str]:
    """The REINFORCE settings that print more solid than the project's own `process` (Bambu settings): none when it
    already prints 100 % infill with as many walls. A Leafeon at 34 mm printed with 100 % infill sliced to the same
    grams with or without them (2026-11-09; a 10 % infill box over it did take 0.4 g off, so modifiers do apply)."""
    if not process:
        return dict(REINFORCE)

    def num(v: Any) -> float:
        try:
            return float(str(v[0] if isinstance(v, list) else v).rstrip("%"))
        except ValueError:
            return 0.0

    return {k: v for k, v in REINFORCE.items() if num(process.get(k, 0)) < num(v)}


def reinforcements(parts: list[Part], scale: float = 1.0, probe: float = PROBE,
                   process: dict[str, Any] | None = None) -> tuple[list[Part], list[WeakPoint]]:
    """Bambu modifier parts (figure frame, scale 1: the parts' own) that print every weak feature graded
    REINFORCE_GRADES solid (the REINFORCE settings the `process` does not reach already: `stronger`): the piece that
    would break off and REINFORCE_BACK mm round it, so its weak section and the join are inside; overlapping pieces
    make one volume. The shape is unchanged. With the weak points they cover (also when no setting is stronger and no
    volume is made)."""
    grid, points = _graded(parts, scale, probe)
    chosen = [w for w in points if w.grade in REINFORCE_GRADES]
    settings = stronger(process)
    if grid is None or not chosen or not settings:
        return [], chosen
    cells = np.zeros(grid.inside.shape, bool)
    for w in chosen:
        cells.flat[w.cells] = True
    grow = REINFORCE_BACK / grid.h                                  # voxels
    labels, n = ndimage.label(cells, _CUBE)
    out: list[Part] = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        pad = int(np.ceil(grow)) + 2
        shape: tuple[int, ...] = cells.shape
        sl = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad)) for s, dim in zip(sl, shape))
        mask = labels[sl] == k
        # Signed distance (voxels) to the piece grown by `grow`: negative inside, for the tetrahedra mesher.
        phi = ndimage.distance_transform_edt(~mask) - grow
        phi = np.pad(phi, 1, constant_values=float(grow + pad))      # closed: the volume never touches the crop's edge
        v, f = _mesh(phi.transpose(1, 0, 2), np.array([grid.xs[sl[1].start], grid.ys[sl[0].start],
                                                      grid.zs[sl[2].start]]) - grid.h, grid.h)
        if len(f) > REINFORCE_FACES:
            from .simplify import clean, compact, simplify_indices

            extent = float(np.ptp(v, axis=0).max())
            cut, _ = simplify_indices(v, f, REINFORCE_FACES, REINFORCE_ERROR / max(extent, 1e-9))
            v, f = compact(v, clean(cut))
        if len(f):
            out.append(Part(f"{REINFORCE_NAME} {len(out) + 1}", 0, v / scale, f, {}, "modifier_part", dict(settings)))
    return out, chosen


def _mesh(phi: np.ndarray, origin: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """The closed surface phi = 0 of a grid in x, y, z order (`strengthen._marching_tets` over every cube)."""
    from .strengthen import _marching_tets

    nx, ny, nz = phi.shape
    ix, iy, iz = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    cubes = ((ix * ny + iy) * nz + iz).ravel()
    return _marching_tets(np.ascontiguousarray(phi), cubes, origin, h)
