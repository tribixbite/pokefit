"""Colorize: paint parts of a figure that has no (or too few) colours, picked by geometry.

A paint operation names one triangle (part index + face index in the source
object, so it survives rotation and scaling) and a filament. It paints the
triangles around it that its mode selects:

  * "part" (default): spread over the surface, stopping only at concave
    valleys sharper than `angle` degrees: where a tail, ear or paw meets the
    body, the rim of an eye socket. Parts of a sculpted figure meet along
    such valleys (the "minima rule" of shape segmentation); convex edges (an
    ear tip, a fin) do not stop it. On a smooth 10k-triangle Eevee, 15–20°
    gives ears, tail, legs, bib and eyes as separate parts where plain
    creases give four lumps;
  * "region": stop at any crease, convex or concave, sharper than `angle`:
    the same rule as Bambu Studio's smart fill, for hard-surface models;
  * "piece": every triangle connected to it (a separate shell: loose eyes,
    a hat modelled as its own body);
  * "fill": a bucket fill: the connected surface of the tapped colour,
    stopped only where the colour changes (inside an outline drawn with the
    brush, whatever the shape does);
  * "brush": the connected surface within `radius` mm (file units) of the
    triangle, or of any triangle on its `path` (a stroke dragged across the
    figure): spots, stripes and outlines on a smooth sculpt that has no edge
    to follow;
  * "face": that triangle only;
  * "stamp": an ellipse laid on the surface like a decal (dragged on the view), for an eye or a spot the shape
    does not outline (a decimated mesh whose eye creases were flattened away).

"part" and "region" fills also stop where the visible colour changes (bucket
fill): a sculpt whose eyes or belly have no crease all the way round (a
generated or smoothed figure) gets them by outlining with a brush stroke and
tapping inside.

Any operation can also
  * `mirror`: paint the same spot on the other side of the figure, across its
    plane of mirror symmetry (found from the shape: `MirrorPlane`), so both
    eyes, cheeks or paws take one tap or one stroke;
  * `grow`: widen (mm > 0) or narrow (mm < 0) the painted area: an eye fill
    that stops short of its socket's rim grows into it.

The wand, brush and stamp (and anything grown) paint with a soft edge: a
smooth line through the triangles (`Field`), written as split paint, so an
eye is an oval and not the stepped outline of the triangles it covers.

`despeckle` (run after all operations) gives a painted patch smaller than a
set area the colour around it: a stray dab on a claw would otherwise cost a
filament change on every layer it spans.

Filament 0 removes paint (the part's own filament prints there). The result
is written as Bambu/Prusa per-triangle paint (`paint_color`: whole-triangle
states, split triangles along soft edges), so it opens in Bambu Studio as
ordinary painted colour and survives the pocket cut (`remesh_part` carries
per-triangle attributes over, split paint remapped).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .model3mf import Part

MODES = ("wand", "part", "region", "piece", "fill", "brush", "face", "stamp")
DEFAULT_ANGLE = 15.0        # degrees; "part" mode (Bambu's smart fill, the "region" rule, defaults to 30)
MAX_PATH = 4000             # triangles on one brush stroke's path
MAX_GROW = 20.0             # mm (file units): the most an operation's area grows or shrinks
JOIN_STEPS = 8              # a stroke step up to this many brush radii long is filled in; longer ones jumped
BOUNDED = ("wand", "part", "region", "fill")  # fills that stop where the colour changes
MAX_LEVEL = 8              # wand: furthest step from its first answer to a smaller or larger one
WAND_CONVEX = 0.5           # a ridge counts this much of its bend (a feature's edge is mostly a groove or a step's foot)
WAND_MIN_FACES = 12         # smallest region the wand returns
WAND_MAX_SHARE = 0.4        # largest, as a share of the piece's area
WAND_RADII = 9              # loop candidates: maximum radii, geometric from WAND_R0 edge lengths to WAND_R1 of the size
WAND_R0 = 5.0
WAND_R1 = 0.3
WAND_ANGLES = (3.0, 6.0, 10.0, 15.0, 25.0, 40.0)   # "part" candidates (limbs, flames: they wrap round, no loop fits)
WAND_TURN = 180             # loop: angle steps
WAND_RINGS = 96             # loop: radius steps
WAND_TURN_COST = 0.15       # loop: cost of moving one ring in or out per angle step (bins cost 0.1 on a wall .. 1)
WAND_EASE = 7               # loop radius smoothing: running mean over this many angle steps
WAND_OPEN = 0.25            # a loop running along its maximum radius for more than this share of the turn is open
WAND_REACH = 2              # rings round an edge searched for the wall it follows
WAND_QUANTILE = 25          # edge strength = this percentile of the wall along it
WAND_WHOLE = 0.85           # inside a painted patch, candidates this close to the whole patch are left out
WAND_PASSES = 1             # majority passes on a candidate's edge, after snapping it
WAND_SNAP = 2               # rings either side of a candidate's edge it may move to meet the wall ridge (`_snap`)
WAND_SAME = 0.8             # candidates overlapping this much (IoU) are one; the better scored is kept
MIRROR_SAMPLES = 3000       # surface points the symmetry search reflects
MIRROR_COARSE = 800         # ... of which this many score the first scan
MIRROR_TARGET = 150_000     # surface points voxelised into the distance field (several per voxel)
MIRROR_GRID = 160           # distance-field voxels along the figure's longest side
MIRROR_DIRECTIONS = 600     # plane normals tried before refining (a Fibonacci hemisphere, ~4 degrees apart)
MIRROR_KEEP = 0.75          # the score averages the best-matching 75 % of points (`mirror_plane`)
MIRROR_STARTS = 4           # directions refined
MIRROR_APART = 15.0         # ... at least this many degrees apart
MIRROR_MAX_ERROR = 0.01     # a mirror plane fits if reflected points land this close (share of the size) on average
MIRROR_LEVEL = 0.3          # a plane whose normal is this close to level (|n . up|) is upright
MIRROR_UPRIGHT = 1.5        # ... and wins over a better plane that is not, if no more than this much worse
MIRROR_MISS = 0.04          # ... and a mirrored spot within this share of the size of the surface
SLIVER_RINGS = 8            # a bucket fill follows a soft edge's split triangles this many rings out of its region
_FILL_CACHE = 64            # colour-bounded fills kept per object (keyed by the paint state they saw)
_PAINT_RE = re.compile(r'\s*paint_color="[^"]*"')
BIG = 1e9                   # `Field.phi` far from the edge (inside: -BIG)
SOFT_DEPTH = 4              # split levels below a triangle a soft edge is drawn to (1/16 of its sides)
MAX_SOFT_DEPTH = 8          # ... at most, for a stamp much smaller than its triangles (1/256)
STAMP_FACING = 0.35         # a stamp paints triangles turned less than ~70 degrees from the view
STAMP_STEPS = 24            # a stamp's edge is drawn in sub-triangles at most 1/24 of its smaller semi-axis
WAND_SMOOTH = 4             # wand: averaging passes rounding its triangle-stepped outline (`Topology.outline`)
SHAPE_SCALE = 0.05          # "Shape map": relief is height over the surface smoothed over this share of the size
SHAPE_PASSES = 400          # ... in at most this many averaging passes (a dense mesh is smoothed less far)


@dataclass(frozen=True)
class PaintOp:
    part: int                 # index into the object's parts
    face: int                 # triangle index in that part
    filament: int             # 1-based filament number; 0 = remove paint
    mode: str = "part"
    angle: float = DEFAULT_ANGLE
    radius: float = 0.0       # "brush": mm in the file's units (the UI converts from print mm)
    path: tuple[int, ...] = ()  # "brush": more triangles of the same part along a dragged stroke
    mirror: bool = False      # also paint the mirror image across the figure's symmetry plane
    grow: float = 0.0         # widen (> 0) or narrow (< 0) the painted area by this many mm (file units)
    level: int = 0            # "wand": 0 its best answer, -1 / +1 the next smaller / larger (Topology.wand)
    # "stamp": an ellipse laid on the surface like a decal, (centre, semi-axis u, semi-axis v) as 9 numbers in the
    # topology's frame (file mm; `Topology.stamp_field`); `face` is the triangle under its centre.
    stamp: tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaintOp":
        op = cls(int(d["part"]), int(d["face"]), int(d["filament"]), str(d.get("mode", "part")),
                 float(d.get("angle", DEFAULT_ANGLE)), float(d.get("radius", 0.0)),
                 tuple(int(f) for f in d.get("path") or ()), bool(d.get("mirror", False)),
                 float(d.get("grow", 0.0)), int(d.get("level", 0)),
                 tuple(float(x) for x in d.get("stamp") or ()))
        if op.mode not in MODES:
            raise ValueError(f"paint mode must be one of {', '.join(MODES)}, not {op.mode!r}")
        if not (0 <= op.filament <= 64):
            raise ValueError(f"filament must be 0–64, not {op.filament}")
        if not (0.0 < op.angle <= 180.0):
            raise ValueError(f"angle must be in (0, 180], not {op.angle}")
        if op.mode == "brush" and not (op.radius > 0):
            raise ValueError("a brush needs a radius > 0")
        if op.path and op.mode != "brush":
            raise ValueError("only a brush stroke has a path")
        if len(op.path) > MAX_PATH:
            raise ValueError(f"a stroke has at most {MAX_PATH} points")
        if abs(op.level) > MAX_LEVEL:
            raise ValueError(f"level must be within ±{MAX_LEVEL}")
        if not (abs(op.grow) <= MAX_GROW and np.isfinite(op.grow)):
            raise ValueError(f"grow must be within ±{MAX_GROW:g} mm")
        if op.mode == "stamp":
            st = np.asarray(op.stamp, float)
            if len(st) != 9 or not np.isfinite(st).all() or min(np.linalg.norm(st[3:6]), np.linalg.norm(st[6:9])) <= 0:
                raise ValueError("a stamp needs its centre and two semi-axes (9 numbers, axes longer than 0)")
            if abs(float(st[3:6] @ st[6:9])) > 1e-3 * np.linalg.norm(st[3:6]) * np.linalg.norm(st[6:9]):
                raise ValueError("a stamp's two semi-axes must be at right angles")
        elif op.stamp:
            raise ValueError("only a stamp has a stamp shape")
        return op

    def to_dict(self) -> dict[str, Any]:
        """The UI's form: optional fields only when set (older paint lists stay byte-identical)."""
        d = asdict(self)
        if self.path:
            d["path"] = list(self.path)
        else:
            del d["path"]
        if not self.mirror:
            del d["mirror"]
        if not self.grow:
            del d["grow"]
        if not self.level:
            del d["level"]
        if self.stamp:
            d["stamp"] = list(self.stamp)
        else:
            del d["stamp"]
        return d


@dataclass
class Field:
    """A painted area with a smooth edge: `phi` per merged vertex of the part's `Topology` (<= 0 inside; near the
    edge about the distance to it, mm), read linearly across each triangle, so the edge is the zero line of `phi`
    and cuts through triangles (`paint.overlay` writes it as split paint). Only `faces` are touched."""
    faces: np.ndarray
    phi: np.ndarray
    # The field at any points (k, 3) -> (k,), when it is known between the vertices too (a stamp's ellipse, a
    # distance to a stroke or an outline): triangles it crosses are then cut along the true curve, not along the
    # straight line between their corners' values (which bends an ellipse on a big low-poly triangle into a spike).
    fn: Callable[[np.ndarray], np.ndarray] | None = None
    small: bool = False         # the area can lie inside one triangle (a small stamp on a low-poly mesh)
    detail: float | None = None  # mm: split crossed triangles finer than SOFT_DEPTH until this fine (`_depth`)

    def corners(self, topo: "Topology", faces: np.ndarray | None = None) -> np.ndarray:
        """(k, 3) `phi` at the corners of `faces` (default: all of `self.faces`), in each triangle's corner order."""
        return np.asarray(self.phi[topo.vfaces[self.faces if faces is None else faces]])

    def inside(self, topo: "Topology") -> np.ndarray:
        """The faces mostly inside (centre <= 0): the area as whole triangles (counts, areas, mirror matching)."""
        return np.asarray(self.faces[self.corners(topo).mean(axis=1) <= 0])

    def restricted(self, allowed: np.ndarray) -> "Field":
        """Only the faces in `allowed` (a mask): another colour's triangles are never touched."""
        return Field(self.faces[allowed[self.faces]], self.phi, self.fn, self.small, self.detail)


@dataclass
class Topology:
    """Which triangles share an edge, how sharply the surface bends there, and which way."""
    n_faces: int
    pairs: np.ndarray          # (k, 2) adjacent face indices
    angles: np.ndarray         # (k,) degrees between the two face normals (0 = flat)
    convex: np.ndarray         # (k,) the edge bends outward (a ridge), not into a valley
    centroids: np.ndarray      # (n, 3) triangle centres, file mm
    areas: np.ndarray          # (n,) triangle areas, file mm²
    normals: np.ndarray        # (n, 3) unit face normals
    # Merged vertices (positions, each face's corners in the part's own corner order) and the edge each adjacent
    # pair shares: soft edges (`Field`) are drawn across triangles on these.
    verts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vfaces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    pair_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    _tree: Any = None          # KD-tree of `centroids`, built on first use (`tree`)
    _wall: Any = None          # `wall`, built on first use
    _vgraph: Any = None        # `_vertex_graph`, built on first use

    @property
    def tree(self) -> Any:
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.centroids)
        return self._tree

    def labels(self, mode: str, angle: float) -> np.ndarray:
        """Region id per triangle for a seed-independent mode (all but "brush"). A "fill" spans the whole
        piece; the colour bounds it (`bounded`)."""
        if mode == "face":
            return np.arange(self.n_faces)
        if mode in ("piece", "fill"):
            keep = np.ones(len(self.pairs), bool)
        elif mode == "part":
            keep = self.convex | (self.angles < angle)
        else:
            keep = self.angles < angle
        return self._components(keep)

    def brush(self, seeds: list[int], radius: float) -> np.ndarray:
        """Faces within `radius` of a seed's centre and connected to a seed inside those balls: one dab, or
        a stroke (a dab per point along it; a thin ridge between two dabs is not bridged)."""
        seed = np.asarray(seeds, np.int64)
        centres = self.centroids[seed]
        if len(seed) > 1:
            # Join consecutive samples (a fast drag leaves gaps a fill would leak through): points along each
            # step at most half a radius apart. A step longer than JOIN_STEPS radii jumped to another surface
            # (the drag crossed a silhouette) and is not bridged.
            a, b = centres[:-1], centres[1:]
            n = np.ceil(np.linalg.norm(b - a, axis=1) / (radius * 0.5)).astype(np.int64)
            n[n > JOIN_STEPS * 2] = 1
            k = np.repeat(np.arange(len(a)), n.astype(np.intp))
            t = (np.arange(len(k)) - np.repeat(np.cumsum(n) - n, n.astype(np.intp))) / n[k]
            centres = np.vstack([a[k] + (b[k] - a[k]) * t[:, None], centres[-1:]])
        near = self._near(centres, radius)
        near[seed] = True
        lab = self._components(near[self.pairs[:, 0]] & near[self.pairs[:, 1]])
        return np.nonzero(near & np.isin(lab, lab[seed]))[0]

    def _near(self, points: np.ndarray, radius: float) -> np.ndarray:
        """Triangles whose centre lies within `radius` of any of `points` (a mask)."""
        near = np.zeros(self.n_faces, bool)
        hits = self.tree.query_ball_point(points, r=radius) if len(points) else []
        if len(hits):
            near[np.concatenate([np.asarray(h, np.int64) for h in hits])] = True
        return near

    def _rim(self, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(faces of a region on its edge, faces outside it along that edge)."""
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        cut = inside[a] != inside[b]
        ins = np.where(inside[a[cut]], a[cut], b[cut])
        out = np.where(inside[a[cut]], b[cut], a[cut])
        return np.unique(ins), np.unique(out)

    # ------------------------------------------------------------------ soft edges (sub-triangle paint)

    def edge_length(self) -> float:
        """Median distance between neighbouring triangle centres (the mesh's resolution), file mm."""
        if not len(self.pairs):
            return 1.0
        d = np.linalg.norm(self.centroids[self.pairs[:, 0]] - self.centroids[self.pairs[:, 1]], axis=1)
        return float(np.median(d)) or 1.0

    def vertex_graph(self) -> Any:
        """Merged-vertex adjacency weighted by edge length (CSR), for walks along the surface."""
        if self._vgraph is None:
            f = self.vfaces
            e = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=0)
            w = np.linalg.norm(self.verts[e[:, 0]] - self.verts[e[:, 1]], axis=1) + 1e-12
            n = len(self.verts)
            self._vgraph = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                                      shape=(n, n)).tocsr()
        return self._vgraph

    def _reach(self, start: np.ndarray, limit: float) -> np.ndarray:
        """Merged vertices within `limit` mm of any of `start` walking along mesh edges (a mask)."""
        from scipy.sparse.csgraph import dijkstra

        out = np.zeros(len(self.verts), bool)
        if len(start):
            d = dijkstra(self.vertex_graph(), directed=False, indices=np.unique(start), limit=limit, min_only=True)
            out[np.isfinite(d)] = True
        return out

    def outline(self, inside: np.ndarray, smooth: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """The edge of a face region as chains of (points, per-segment normal of the inside triangle), each run with
        the region on its left (seen from outside). The edge steps from triangle to triangle; `smooth` passes of
        averaging each point with its two neighbours round it off (a closed chain stays closed; an open one, where
        the region meets a hole in the mesh, keeps its ends)."""
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        cut = inside[a] != inside[b]
        if not cut.any():
            return []
        fin = np.where(inside[a[cut]], a[cut], b[cut])
        fout = np.where(inside[a[cut]], b[cut], a[cut])
        uv = self.pair_edges[cut].copy()
        corners = self.vfaces[fin]
        # Run each edge the way its inside triangle does: then the region lies on the left of it, whichever way the
        # triangle is wound (its normal turns with its winding, and the side test uses both).
        at = np.argmax(corners == uv[:, :1], axis=1)
        flip = corners[np.arange(len(fin)), (at + 1) % 3] != uv[:, 1]
        uv[flip] = uv[flip][:, ::-1]
        # The side test's "up": both triangles on the edge (one of them alone tips over on a sharp fold, a brow's
        # crease), the outside one turned to agree with the inside one's winding.
        n_in, n_out = self.normals[fin], self.normals[fout]
        nrm = n_in + n_out * np.where((n_in * n_out).sum(axis=1) < 0, -1.0, 1.0)[:, None]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        succ: dict[int, list[int]] = {}
        for k, u in enumerate(uv[:, 0].tolist()):
            succ.setdefault(u, []).append(k)
        used = np.zeros(len(uv), bool)
        chains: list[tuple[np.ndarray, np.ndarray]] = []
        for k0 in range(len(uv)):
            if used[k0]:
                continue
            ks = [k0]
            used[k0] = True
            first, cur = int(uv[k0, 0]), int(uv[k0, 1])
            while cur != first:
                nxt = next((k for k in succ.get(cur, []) if not used[k]), None)
                if nxt is None:
                    break
                used[nxt] = True
                ks.append(nxt)
                cur = int(uv[nxt, 1])
            closed = cur == first
            pts = self.verts[uv[ks, 0]]
            if not closed:
                pts = np.vstack([pts, self.verts[uv[ks[-1], 1]]])
            for _ in range(smooth):
                if closed and len(pts) >= 3:
                    pts = 0.5 * pts + 0.25 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
                elif len(pts) >= 3:
                    pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
            if closed:
                pts = np.vstack([pts, pts[:1]])
            chains.append((pts, nrm[ks]))
        return chains

    def soft(self, inside: np.ndarray, smooth: int, offset: float = 0.0) -> "Field":
        """A face region (a mask) as a `Field` with a smooth edge: the signed distance (inside < 0) of every vertex
        near its outline (`outline`, rounded by `smooth` passes) to that outline, minus `offset` mm (> 0 widens the
        area, < 0 narrows it, that far from the edge measured straight, within a walk along the surface)."""
        from scipy.spatial import cKDTree

        h = self.edge_length()
        n = len(self.verts)
        vin = np.zeros(n, bool)
        vin[self.vfaces[inside]] = True
        phi = np.where(vin, -BIG, BIG)
        chains = self.outline(inside, smooth)
        if not chains:
            return Field(np.nonzero(inside)[0], phi - offset)
        # Dense samples along the outline, each with its direction and the inside triangle's normal.
        pts, tan, nrm = [], [], []
        for c, cn in chains:
            seg = c[1:] - c[:-1]
            ln = np.linalg.norm(seg, axis=1)
            k = np.maximum(1, np.ceil(ln / (h / 4))).astype(np.intp)
            t = (np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
            i = np.repeat(np.arange(len(seg)), k)
            pts.append(c[i] + seg[i] * t[:, None])
            tan.append(seg[i] / (ln[i, None] + 1e-12))
            nrm.append(cn[i])
        q, tq, nq = np.vstack(pts), np.vstack(tan), np.vstack(nrm)
        # Only vertices reachable along the surface within the band: a distance across a gap (a finger beside the
        # body) is not a distance along the figure.
        cut = inside[self.pairs[:, 0]] != inside[self.pairs[:, 1]]
        band = self._reach(self.pair_edges[cut].ravel(), abs(offset) * 1.3 + 3 * h)
        idx = np.nonzero(band)[0]
        d, j = cKDTree(q).query(self.verts[idx])
        left = np.cross(nq[j], tq[j])
        side = ((self.verts[idx] - q[j]) * left).sum(axis=1) > 0
        # Which side a vertex is on: its own triangles say, unless it has triangles on both sides (on the stepped
        # outline) or lies by the smoothed one; only there is the side test needed (and right).
        vout = np.zeros(n, bool)
        vout[self.vfaces[~inside]] = True
        sure = (vin[idx] != vout[idx]) & (d > 1.5 * h)
        side = np.where(sure, vin[idx], side)
        phi[idx] = np.where(side, -d, d)
        faces = np.nonzero(inside | band[self.vfaces].any(axis=1))[0]
        tree = cKDTree(q)

        def exact(pts: np.ndarray) -> np.ndarray:
            dd, jj = tree.query(pts)
            inner = ((pts - q[jj]) * np.cross(nq[jj], tq[jj])).sum(axis=1) > 0
            return np.asarray(np.where(inner, -dd, dd) - offset)
        return Field(faces, phi - offset, exact)

    def brush_field(self, seeds: list[int], radius: float) -> "Field":
        """A brush dab or stroke with a round edge: distance from each vertex to the stroke (the seeds' centres,
        joined) minus `radius`, over the triangles `brush` covers and the ring round them."""
        faces = self.brush(seeds, radius)
        centres = self.centroids[np.asarray(seeds, np.int64)]
        if len(centres) > 1:
            a, b = centres[:-1], centres[1:]
            ln = np.linalg.norm(b - a, axis=1)
            k = np.maximum(1, np.ceil(ln / (radius / 8))).astype(np.intp)
            k[ln > JOIN_STEPS * radius] = 1          # a jump to another surface (the drag crossed a silhouette)
            t = (np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
            i = np.repeat(np.arange(len(a)), k)
            centres = np.vstack([a[i] + (b[i] - a[i]) * t[:, None], centres[-1:]])
        return self._distance_field(faces, centres, radius)

    def _distance_field(self, faces: np.ndarray, centres: np.ndarray, radius: float) -> "Field":
        """`Field` of the points within `radius` of `centres`, limited to `faces` and the ring of faces round them."""
        from scipy.spatial import cKDTree

        ring = np.zeros(self.n_faces, bool)
        ring[faces] = True
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        grown = ring.copy()
        grown[a[ring[b]]] = True
        grown[b[ring[a]]] = True
        cand = np.nonzero(grown)[0]
        vid = np.unique(self.vfaces[cand])
        phi = np.full(len(self.verts), BIG)
        tree = cKDTree(centres)
        d, _ = tree.query(self.verts[vid])
        phi[vid] = d - radius

        def exact(pts: np.ndarray) -> np.ndarray:
            return np.asarray(tree.query(pts)[0] - radius)
        return Field(cand, phi, exact)

    def stamp_field(self, face: int, centre: np.ndarray, u: np.ndarray, v: np.ndarray) -> "Field":
        """An ellipse projected onto the surface like a decal: `centre`, semi-axes `u` and `v` (vectors, file mm),
        seen along u x v. Covers the triangles joined to `face` that face that way, whose centre falls inside the
        ellipse and lies within its size in front of or behind the centre (not a surface further back)."""
        a, b = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        uh, vh = u / (a + 1e-12), v / (b + 1e-12)
        w = np.cross(uh, vh)
        w /= np.linalg.norm(w) + 1e-12
        reach = max(a, b)

        def q(p: np.ndarray) -> np.ndarray:
            d = p - centre
            return np.asarray(np.sqrt(((d @ uh) / a) ** 2 + ((d @ vh) / b) ** 2))

        h = self.edge_length()
        near = np.asarray(self.tree.query_ball_point(centre, reach * 1.5 + 2 * h), np.int64)
        cen = self.centroids[near]
        ok = np.zeros(self.n_faces, bool)
        # Only surfaces facing the view: on a wall seen nearly edge-on (a crease's side) the ellipse's edge projects
        # into long slivers (seen beside a low-poly eye). The depth window keeps the far side of a thin part out.
        ok[near] = (((self.normals[near] @ w) > STAMP_FACING) & (np.abs((cen - centre) @ w) <= reach)
                    & (q(cen) <= 1.0 + 2 * h / min(a, b)))
        ok[face] = True
        lab = self._components(ok[self.pairs[:, 0]] & ok[self.pairs[:, 1]])
        faces = np.nonzero(ok & (lab == lab[face]))[0]
        vid = np.unique(self.vfaces[faces])
        phi = np.full(len(self.verts), BIG)
        phi[vid] = (q(self.verts[vid]) - 1.0) * min(a, b)       # about the distance to the edge, near it
        return Field(faces, phi, lambda pts: (q(pts) - 1.0) * min(a, b), small=True, detail=min(a, b) / STAMP_STEPS)

    def bend(self) -> np.ndarray:
        """Signed bend per face, degrees (> 0 concave): the mean turn of the face normals across its edges after one
        averaging pass over edge neighbours. Per-edge crease angles of a smooth sculpt are mostly noise (a soft
        sculpted step bends each of its triangles a degree or two, like the noise); one pass of averaging keeps the
        step and drops the noise (measured on the 303k Charmander: closed rings round both eyes)."""
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        n = self.n_faces
        w = self.normals * self.areas[:, None]
        sm = w.copy()
        np.add.at(sm, a, w[b])
        np.add.at(sm, b, w[a])
        sm /= np.linalg.norm(sm, axis=1, keepdims=True) + 1e-12
        turn = np.degrees(np.arccos(np.clip((sm[a] * sm[b]).sum(1), -1.0, 1.0)))
        d = self.centroids[b] - self.centroids[a]
        concave = (d * sm[a]).sum(1) - (d * sm[b]).sum(1) > 0      # each centre above the other's tangent plane
        signed = np.where(concave, turn, -turn)
        face = np.bincount(a, signed, n) + np.bincount(b, signed, n)
        return np.asarray(face / np.maximum(np.bincount(np.r_[a, b], minlength=n), 1))

    def shape_map(self) -> np.ndarray:
        """(merged vertices, 2) uint8 for the "Shape map" view: how much each spot is a crease (the wand's `wall`,
        0 = smooth .. 255 = among the strongest 2 %), and its relief (128 = level with the surface round it, lower =
        sunk, higher = raised: the height over a copy of the surface smoothed over SHAPE_SCALE of its size, scaled by
        its 98th percentile). Lines of equal relief are drawn from it like a topographic map."""
        n = len(self.verts)
        f = self.vfaces
        w = np.repeat(self.areas, 3)
        acc = np.bincount(f.ravel(), weights=np.repeat(self.wall(), 3) * w, minlength=n)
        tot = np.bincount(f.ravel(), weights=w, minlength=n) + 1e-12
        crease = acc / tot
        crease = np.clip(crease / (np.percentile(crease, 98) + 1e-9), 0, 1)
        g = self.vertex_graph().copy()
        edge = float(np.mean(g.data)) or 1.0
        g.data[:] = 1.0
        deg = np.asarray(g.sum(axis=1)).ravel() + 1e-12
        smooth = self.verts.copy()
        # Smoothed over about SHAPE_SCALE of the figure's size, whatever its triangles: a pass spreads about one edge.
        reach = SHAPE_SCALE * float(np.linalg.norm(np.ptp(self.verts, axis=0)))
        passes = int(np.clip((reach / edge) ** 2 / 0.6, 10, SHAPE_PASSES))
        # Plain averaging (Taubin's shrink-free smoothing keeps any bump wider than a few edges, the very features
        # wanted here); it shrinks a rounded body a little, an offset that varies slowly and is taken off below.
        for _ in range(passes):
            smooth = smooth + 0.5 * (g @ smooth / deg[:, None] - smooth)
        vn = np.zeros((n, 3))
        sm_tri = smooth[f]
        fn_ = np.cross(sm_tri[:, 1] - sm_tri[:, 0], sm_tri[:, 2] - sm_tri[:, 0])
        for k in range(3):
            np.add.at(vn, f[:, k], fn_)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        relief = ((self.verts - smooth) * vn).sum(axis=1)
        trend = relief.copy()
        for _ in range(4 * passes):                     # the shrink offset: the relief smoothed twice as far
            trend = trend + 0.5 * (g @ trend / deg - trend)
        relief = relief - trend
        for _ in range(4):                              # a decimated sculpt's facets are noise at this scale
            relief = 0.5 * relief + 0.5 * (g @ relief) / deg
        relief = np.clip(relief / (np.percentile(np.abs(relief), 98) + 1e-12), -1, 1)
        return np.stack([np.round(crease * 255), np.round(128 + relief * 127)], axis=1).astype(np.uint8)

    def wall(self) -> np.ndarray:
        """How much each face is a feature edge, degrees: concave bend, plus WAND_CONVEX of convex bend."""
        if self._wall is None:
            bd = self.bend()
            self._wall = np.maximum(bd, 0) + WAND_CONVEX * np.maximum(-bd, 0)
        return self._wall

    def wand_candidates(self, seed: int, allowed: np.ndarray | None = None) -> list[tuple[float, int, np.ndarray]]:
        """Regions the magic wand offers for a tap, smallest first, as (score, support, faces): the score is the low
        quartile of `wall` along its edge over the mesh's noise (median wall); support counts the generators that
        found it. Two kinds: closed loops round the tap at a series of maximum radii
        (`_loop`: eyes, spots, cheeks - anything roughly convex seen from the tap), and the tap's "part" region at
        several angles (limbs, ears, flames, which wrap round and fit no flat loop). `allowed` keeps it in the
        tap's colour."""
        ok = np.ones(self.n_faces, bool) if allowed is None else allowed.copy()
        ok[seed] = True
        wall = self.wall()
        noise = float(np.median(wall)) + 1e-3
        piece = self.labels("piece", 0.0)
        mine = piece == piece[seed]
        piece_area = float(self.areas[mine].sum())
        size = float(np.linalg.norm(np.ptp(self.centroids[mine], axis=0))) or 1.0
        edge = float(np.median(np.linalg.norm(self.centroids[self.pairs[:, 0]] - self.centroids[self.pairs[:, 1]],
                                              axis=1))) or 1.0
        own = self.labels("piece", 0.0) if allowed is None else self._components(ok[self.pairs[:, 0]] & ok[self.pairs[:, 1]])
        own_area = float(self.areas[(own == own[seed]) & ok].sum())
        raw: list[np.ndarray] = []
        for rmax in np.geomspace(WAND_R0 * edge, WAND_R1 * size, WAND_RADII):
            reg = self._loop(seed, float(rmax), ok, wall, noise)
            if reg is not None:
                raw.append(reg)
        for ang in WAND_ANGLES:
            lab = self.labels("part", ang)
            raw.append(np.nonzero((lab == lab[seed]) & ok)[0])
        out: list[tuple[float, int, np.ndarray]] = []
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        # The strongest wall within WAND_REACH rings of each face: an edge drawn a triangle inside or outside the
        # ring it follows still counts as on it.
        near_wall = wall.copy()
        for _ in range(WAND_REACH):
            grown = near_wall.copy()
            np.maximum.at(grown, a, near_wall[b])
            np.maximum.at(grown, b, near_wall[a])
            near_wall = grown
        near_noise = float(np.median(near_wall)) + 1e-3
        for reg in raw:
            if float(self.areas[reg].sum()) > 1.1 * WAND_MAX_SHARE * piece_area:
                continue                # too big before cleaning (holes only add): skip the work
            reg = self._tidy(reg, seed, ok)
            area = float(self.areas[reg].sum())
            if len(reg) < WAND_MIN_FACES or area > WAND_MAX_SHARE * piece_area:
                continue
            if allowed is not None and area >= WAND_WHOLE * own_area:
                continue        # all of the tapped colour's patch: that is a fill, not a feature inside it
            inside = np.zeros(self.n_faces, bool)
            inside[reg] = True
            cut = inside[a] != inside[b]
            rim = np.unique(np.r_[a[cut], b[cut]])
            # The low quartile, not the mean: a feature's edge is strong all the way round; a big region that borrows
            # one deep groove (a snout) for part of its edge and runs through smooth surface elsewhere is not one.
            score = float(np.percentile(near_wall[rim], WAND_QUANTILE)) / near_noise if len(rim) else 0.0
            dup = next((i for i, (_, _, r) in enumerate(out) if _iou(r, reg) >= WAND_SAME), None)
            if dup is None:
                out.append((score, 1, reg))
            else:
                sc, sup, old = out[dup]
                out[dup] = (score, sup + 1, reg) if score > sc else (sc, sup + 1, old)
        out.sort(key=lambda t: float(self.areas[t[2]].sum()))
        return out

    def wand(self, seed: int, allowed: np.ndarray | None = None, level: int = 0) -> np.ndarray:
        """Magic wand: the best-scoring `wand_candidates` region (level 0), or the next smaller (< 0) / larger (> 0)
        one; just the tapped triangle when nothing qualifies."""
        cands = self.wand_candidates(seed, allowed)
        if not cands:
            return np.asarray([seed])
        best = max(range(len(cands)), key=lambda i: cands[i][0])
        return cands[min(max(best + int(level), 0), len(cands) - 1)][2]

    def _loop(self, seed: int, rmax: float, ok: np.ndarray, wall: np.ndarray, noise: float) -> np.ndarray | None:
        """Faces inside the cheapest closed loop round `seed` within `rmax` (mm), or None if the loop is open (runs
        along `rmax`). The faces near the tap that face its way are laid flat on its tangent plane in polar bins
        (WAND_TURN x WAND_RINGS); a bin costs less the higher the wall in it; dynamic programming over two turns
        (the radius may change one bin per step, so the loop is smooth and a gap in a ring costs only its width)
        gives the loop, its second turn taken as closed."""
        near = np.asarray(self.tree.query_ball_point(self.centroids[seed], rmax * 1.02), np.int64)
        if len(near) < WAND_MIN_FACES:
            return None
        nrm = (self.normals[near] * self.areas[near, None]).sum(0)
        nrm /= np.linalg.norm(nrm) + 1e-12
        near = near[(self.normals[near] @ nrm) > 0.2]
        u = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        d = self.centroids[near] - self.centroids[seed]
        x, y = d @ u, d @ v
        r = np.hypot(x, y)
        ti = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * WAND_TURN).astype(np.int64) % WAND_TURN
        ri = np.clip((r / rmax * WAND_RINGS).astype(np.int64), 0, WAND_RINGS - 1)
        grid = np.zeros((WAND_TURN, WAND_RINGS))
        np.maximum.at(grid, (ti, ri), np.where(ok[near], wall[near], 0.0))
        cost = 1.0 / (1.0 + (grid / (3.0 * noise)) ** 2)
        cost[:, :2] = 10.0                                           # never through the tap itself
        steps = 2 * WAND_TURN
        acc = cost[0].copy()
        back = np.zeros((steps, WAND_RINGS), np.int64)
        idx = np.arange(WAND_RINGS)
        down = np.full(WAND_RINGS, np.inf)          # from one ring further in
        up = np.full(WAND_RINGS, np.inf)            # from one ring further out
        for t in range(1, steps):
            down[1:] = acc[:-1]
            up[:-1] = acc[1:]
            # A ring change costs WAND_TURN_COST: the loop follows a ridge, not every bump of the noise.
            dn, upc = down + WAND_TURN_COST, up + WAND_TURN_COST
            k = np.where(dn < acc, np.where(dn <= upc, 0, 2), np.where(acc <= upc, 1, 2))
            acc = np.minimum(np.minimum(dn, acc), upc) + cost[t % WAND_TURN]
            back[t] = idx + k - 1
        path = np.zeros(steps, np.int64)
        path[-1] = int(np.argmin(acc))
        for t in range(steps - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        loop = np.empty(WAND_TURN, np.int64)
        loop[np.arange(WAND_TURN, steps) % WAND_TURN] = path[WAND_TURN:]
        if float(np.mean(loop >= WAND_RINGS - 3)) > WAND_OPEN:
            return None
        # The path steps a bin at a time and zigzags on noise: smooth its radius round the circle (a running mean
        # over WAND_EASE angle steps), then read it between bins for each face's exact angle.
        k = np.ones(WAND_EASE) / WAND_EASE
        rad = (np.convolve(np.r_[loop[-WAND_EASE:], loop, loop[:WAND_EASE]], k, mode="same")[WAND_EASE:-WAND_EASE]
               + 0.5) / WAND_RINGS * rmax
        th = (np.arctan2(y, x) + np.pi) / (2 * np.pi) * WAND_TURN - 0.5
        lim = np.interp(th, np.arange(-1, WAND_TURN + 1), np.r_[rad[-1], rad, rad[0]])
        return np.asarray(near[r <= lim])

    def _tidy(self, reg: np.ndarray, seed: int, ok: np.ndarray) -> np.ndarray:
        """A candidate made clean: in the allowed colour, enclosed holes filled (smaller than a quarter of it),
        WAND_PASSES majority passes on its edge (bins and facets leave notches), only the part joined to the tap.
        Worked on the faces round the candidate only (a ball 15 % past its farthest face): over the whole mesh,
        15 candidates a tap took 0.5 s of 0.95 on 100k triangles."""
        cen = self.centroids[reg]
        mid = cen.mean(axis=0)
        reach = float(np.linalg.norm(cen - mid, axis=1).max()) * 1.15 + 1e-6
        local = np.union1d(np.asarray(self.tree.query_ball_point(mid, reach), np.int64), np.r_[reg, seed])
        n = len(local)
        pos = np.full(self.n_faces, -1, np.int64)
        pos[local] = np.arange(n)
        pa, pb = pos[self.pairs[:, 0]], pos[self.pairs[:, 1]]
        both = (pa >= 0) & (pb >= 0)
        pairs = np.stack([pa[both], pb[both]], axis=1)
        # Faces on the ball's surface (an edge leaves it) count as "outside": holes are what they do not reach.
        border = np.zeros(n, bool)
        border[pa[(pa >= 0) & (pb < 0)]] = True
        border[pb[(pb >= 0) & (pa < 0)]] = True
        lok = ok[local]
        area = self.areas[local]
        s = pos[seed]
        inside = np.zeros(n, bool)
        inside[pos[reg]] = True
        inside &= lok
        inside[s] = True
        rest = lok & ~inside
        lab = _labels(pairs, n, rest[pairs[:, 0]] & rest[pairs[:, 1]])
        outer = np.unique(lab[border & rest])
        holes = rest & ~np.isin(lab, outer)
        if holes.any():
            sizes = np.bincount(lab[holes], weights=area[holes], minlength=int(lab.max()) + 1)
            inside |= holes & (sizes[lab] < 0.25 * area[inside].sum())
        inside = _snap(pairs, n, inside, lok, self.wall()[local], WAND_SNAP)
        for _ in range(WAND_PASSES):
            cut = inside[pairs[:, 0]] != inside[pairs[:, 1]]
            band = np.zeros(n, bool)
            band[pairs[cut].ravel()] = True
            inside = _majority(pairs, n, inside, band & lok)
        inside[s] = True
        lab = _labels(pairs, n, inside[pairs[:, 0]] & inside[pairs[:, 1]])
        return np.asarray(local[inside & (lab == lab[s])])

    def specks(self, colour: np.ndarray, min_area: float) -> tuple[np.ndarray, np.ndarray]:
        """(faces, new colour) for every one-colour patch smaller than `min_area` (file mm²): each takes the
        colour it shares the most edges with. A patch with no other colour round it (a whole separate piece)
        is kept."""
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        same = colour[a] == colour[b]
        lab = self._components(same)
        area = np.bincount(lab, weights=self.areas, minlength=int(lab.max()) + 1 if len(lab) else 0)
        small = area < min_area
        if not small[lab].any():
            return np.zeros(0, np.int64), np.zeros(0, np.int32)
        # Edges from a small patch into another colour: count them per (patch, colour); the most wins.
        e = ~same
        pa = np.concatenate([lab[a[e]], lab[b[e]]])
        nc = np.concatenate([colour[b[e]], colour[a[e]]]).astype(np.int64)
        sel = small[pa]
        if not sel.any():
            return np.zeros(0, np.int64), np.zeros(0, np.int32)
        pa, nc = pa[sel], nc[sel]
        key = pa.astype(np.int64) * (int(nc.max()) + 1) + nc
        k, cnt = np.unique(key, return_counts=True)
        kp, kc = k // (int(nc.max()) + 1), k % (int(nc.max()) + 1)
        order = np.lexsort((-cnt, kp))                     # per patch, the colour with the most edges first
        first = np.ones(len(order), bool)
        first[1:] = kp[order][1:] != kp[order][:-1]
        new = np.full(len(area), -1, np.int64)
        new[kp[order][first]] = kc[order][first]
        faces = np.nonzero(new[lab] >= 0)[0]
        return faces, new[lab[faces]].astype(np.int32)

    def bounded(self, region: np.ndarray, face: int, colour: np.ndarray) -> np.ndarray:
        """The part of `region` (a fill's faces) connected to `face` through faces of `face`'s colour."""
        inside = np.zeros(self.n_faces, bool)
        inside[region] = True
        inside &= colour == colour[face]
        lab = self._components(inside[self.pairs[:, 0]] & inside[self.pairs[:, 1]])
        return np.nonzero(inside & (lab == lab[face]))[0]

    def _components(self, keep: np.ndarray) -> np.ndarray:
        p = self.pairs[keep]
        g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(self.n_faces, self.n_faces))
        _, lab = connected_components(g, directed=False)
        return np.asarray(lab)


def topology(part: Part) -> Topology:
    """Edge adjacency of a part. Vertices are merged by position first (STL soups repeat them)."""
    m = trimesh.Trimesh(part.vertices, part.faces, process=False)
    m.merge_vertices()          # keeps the face count and order
    pairs = np.asarray(m.face_adjacency, np.int64).reshape(-1, 2)
    if len(pairs):
        angles = np.degrees(np.asarray(m.face_adjacency_angles, np.float64))
        convex = np.asarray(m.face_adjacency_convex, bool)
    else:
        angles, convex = np.zeros(0), np.zeros(0, bool)
    edges = np.asarray(m.face_adjacency_edges, np.int64).reshape(-1, 2) if len(pairs) else np.zeros((0, 2), np.int64)
    return Topology(len(part.faces), pairs, angles, convex, np.asarray(m.triangles_center, np.float64),
                    np.asarray(m.area_faces, np.float64), np.asarray(m.face_normals, np.float64),
                    np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), edges)


def _labels(pairs: np.ndarray, n: int, keep: np.ndarray) -> np.ndarray:
    """Connected component per node of the graph `pairs` with only the `keep` edges."""
    p = pairs[keep]
    g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    return np.asarray(connected_components(g, directed=False)[1])


def _snap(pairs: np.ndarray, n: int, inside: np.ndarray, ok: np.ndarray, wall: np.ndarray, rings: int) -> np.ndarray:
    """`inside` with its edge moved onto the wall ridge near it (marker watershed): the faces within `rings` of the
    edge are undecided; the region left inside and the surface left outside then flood into them together, the
    lowest wall first, and meet on the ridge. A loop's edge is only good to a bin or two; the ridge is the outline.
    Faces not `ok` (another colour) stay outside."""
    import heapq

    a, b = pairs[:, 0], pairs[:, 1]
    undecided = np.zeros(n, bool)
    cut = inside[a] != inside[b]
    undecided[pairs[cut].ravel()] = True
    for _ in range(rings - 1):
        grow = undecided.copy()
        grow[a[undecided[b]]] = True
        grow[b[undecided[a]]] = True
        undecided = grow
    undecided &= ok
    label = np.where(undecided, -1, np.where(inside & ok, 1, 0)).astype(np.int64)
    if not undecided.any():
        return label == 1
    # Adjacency lists of the graph (CSR), for the flood.
    order = np.argsort(np.r_[a, b], kind="stable")
    nbr = np.r_[b, a][order]
    start = np.searchsorted(np.r_[a, b][order], np.arange(n + 1))
    heap: list[tuple[float, int, int]] = []
    for f in np.nonzero(undecided)[0].tolist():
        for g in nbr[start[f]:start[f + 1]].tolist():
            if label[g] >= 0:
                heapq.heappush(heap, (float(wall[f]), f, int(label[g])))
                break
    while heap:
        _, f, lab = heapq.heappop(heap)
        if label[f] >= 0:
            continue
        label[f] = lab
        for g in nbr[start[f]:start[f + 1]].tolist():
            if label[g] < 0:
                heapq.heappush(heap, (float(wall[g]), g, lab))
    return label == 1


def _majority(pairs: np.ndarray, n: int, mask: np.ndarray, where: np.ndarray) -> np.ndarray:
    """One majority pass on a graph given as pairs: each node in `where` takes the state of most of its neighbours
    (a triangle has three), which removes the one-triangle notches whole-triangle paint leaves on an edge."""
    a, b = pairs[:, 0], pairs[:, 1]
    on = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    out = mask.copy()
    sel = where & (deg > 0)
    out[sel] = on[sel] * 2 > deg[sel]
    tie = sel & (on * 2 == deg)
    out[tie] = mask[tie]
    return out


@lru_cache(maxsize=None)
def _grid(depth: int) -> tuple[np.ndarray, np.ndarray]:
    """A triangle's split points down to `depth` levels (every corner of its sub-triangles) as (i, j) on the
    barycentric lattice of 2**depth steps, and their barycentric coordinates."""
    n = 1 << depth
    ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], np.intp)
    return ij, np.column_stack([n - ij.sum(axis=1), ij]).astype(float) / n


def _lattice(fn: Callable[[np.ndarray], np.ndarray], corners: np.ndarray, depth: int) -> list[list[float]]:
    """`fn` at a triangle's split points in one call, as `paint.overlay`'s table (`table[i][j]`)."""
    ij, bary = _grid(depth)
    n = 1 << depth
    out = np.full((n + 1, n + 1), np.nan)
    out[ij[:, 0], ij[:, 1]] = fn(bary @ corners)
    return list(out.tolist())


def _depth(corners: np.ndarray, detail: float | None) -> int:
    """Split levels for a triangle: SOFT_DEPTH, or more (up to MAX_SOFT_DEPTH) until its sub-triangles are no
    longer than `detail` mm (a small stamp on a big low-poly triangle)."""
    if detail is None or detail <= 0:
        return SOFT_DEPTH
    longest = float(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).max())
    return int(np.clip(np.ceil(np.log2(max(longest / detail, 1.0))), SOFT_DEPTH, MAX_SOFT_DEPTH))


def _iou(x: np.ndarray, y: np.ndarray) -> float:
    inter = len(np.intersect1d(x, y, assume_unique=True))
    return inter / max(1, len(x) + len(y) - inter)


def paint_code(filament: int) -> str:
    """TriangleSelector string of a whole triangle in one state (see paint.py): nibbles read from the end."""
    from .paint import _leaf_nibbles

    return "".join(f"{n:X}" for n in reversed(_leaf_nibbles(filament)))


@lru_cache(maxsize=1 << 12)
def _with_paint(attrs: str | None, code: str | None) -> str | None:
    """A triangle's extra attributes with its paint replaced (None = paint removed)."""
    rest = _PAINT_RE.sub("", attrs or "").strip()
    if code is not None:
        rest = (rest + f' paint_color="{code}"').strip()
    return rest or None


class PaintState:
    """One part's per-triangle attributes as an index per triangle into a list of distinct strings, so a
    paint operation updates hundreds of thousands of triangles with a few array operations."""

    def __init__(self, part: Part):
        self.extruder = part.extruder
        self.values: list[str | None] = [None]
        self._ids: dict[str | None, int] = {None: 0}
        self.state = np.zeros(len(part.faces), np.int32)
        if part.tri_extra:
            idx = np.fromiter(part.tri_extra.keys(), np.int64, len(part.tri_extra))
            self.state[idx] = [self._id(v) for v in part.tri_extra.values()]
        self._colours: np.ndarray | None = None

    def _id(self, value: str | None) -> int:
        if value not in self._ids:
            self._ids[value] = len(self.values)
            self.values.append(value)
        return self._ids[value]

    def paint(self, faces: np.ndarray, code: str | None) -> None:
        """Give `faces` the paint `code` (None = remove it), keeping their other attributes."""
        uniq, inv = np.unique(self.state[faces], return_inverse=True)
        new = np.asarray([self._id(_with_paint(self.values[u], code)) for u in uniq.tolist()], np.int32)
        self.state[faces] = new[inv.reshape(-1)]
        self._colours = None

    def paint_field(self, topo: Topology, fld: Field, filament: int, over: int | None = None) -> None:
        """Paint where `fld` is <= 0 with `filament` (0 = remove paint): triangles wholly inside take it whole; the
        ones its edge crosses are split along it (`paint.overlay`, `SOFT_DEPTH` levels), keeping what they showed
        on the other side, including split paint they had. With `over` (a filament), only what shows that colour
        is painted, inside split triangles too (a fill bounded by colour)."""
        from .paint import attr_paint, encode_tree, overlay, parse_tree

        ph = fld.corners(topo)
        neg = ph <= 0
        whole = neg.all(axis=1)
        mixed = neg.any(axis=1) & ~whole
        if fld.fn is not None and fld.small:
            # The true edge can cross a triangle whose corners are all outside (a small stamp inside a big one).
            tri = topo.verts[topo.vfaces[fld.faces]]
            longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
            mixed |= ~whole & (ph.min(axis=1) < longest)
        only = None
        if over is not None:
            # Whole triangles of another colour are left alone; a split one may hold some of the tapped colour (the
            # other side of an earlier soft edge): it is overlaid leaf by leaf, only where it shows that colour.
            per_value = [attr_paint(v or "") for v in self.values]
            split = np.array([fp is not None and fp.whole is None for fp in per_value], bool)[self.state[fld.faces]]
            same = self.colours()[fld.faces] == over
            mixed = (mixed | (whole & split)) & (same | split)
            whole &= ~split & same
            ext = self.extruder

            def only(st: int) -> bool:
                return (st or ext) == over
        self.paint(fld.faces[whole], paint_code(filament) if filament else None)
        for f, p3 in zip(fld.faces[mixed].tolist(), ph[mixed].tolist()):
            value = self.values[int(self.state[f])]
            fp = attr_paint(value or "")
            tree = parse_tree(fp.code) if fp is not None else 0
            corners = topo.verts[topo.vfaces[f]]
            depth = _depth(corners, fld.detail)
            table = _lattice(fld.fn, corners, depth) if fld.fn is not None else None
            new = overlay(tree, (p3[0], p3[1], p3[2]), filament, depth, only, table, fld.small)
            code = None if new == 0 else paint_code(new) if isinstance(new, int) else encode_tree(new)
            self.state[f] = self._id(_with_paint(value, code))
        self._colours = None

    def has_splits(self, faces: np.ndarray) -> bool:
        """Some of `faces` hold split paint (a soft edge)."""
        from .paint import attr_paint

        split = [i for i, v in enumerate(self.values) if (fp := attr_paint(v or "")) is not None and fp.whole is None]
        return bool(split) and bool(np.isin(self.state[faces], split).any())

    def colours(self) -> np.ndarray:
        """The filament each triangle shows most (its dominant paint, else the part's own)."""
        if self._colours is None:
            from .paint import attr_paint

            per_value = [self.extruder]
            for v in self.values[1:]:
                fp = attr_paint(v or "")
                per_value.append(fp.majority if fp is not None and fp.majority > 0 else self.extruder)
            self._colours = np.asarray(per_value, np.int32)[self.state]
        return self._colours

    def shown(self) -> list[frozenset[int]]:
        """Per distinct value (`state` indexes it), the filaments a triangle with it shows anywhere: its paint, both
        sides of a soft edge (an unpainted side is the part's own filament)."""
        from .paint import attr_paint, decode_weights

        out = []
        for v in self.values:
            fp = attr_paint(v or "")
            if fp is None:
                out.append(frozenset({self.extruder}))
            elif fp.whole is not None:
                out.append(frozenset({fp.whole or self.extruder}))
            else:
                out.append(frozenset(st or self.extruder for st in decode_weights(fp.code)))
        return out

    def holds(self, faces: np.ndarray, filament: int) -> np.ndarray:
        """Which of `faces` show `filament` somewhere (whole, or on one side of a soft edge)."""
        per_value = np.asarray([filament in s for s in self.shown()], bool)
        return per_value[self.state[faces]]

    def split(self) -> np.ndarray:
        """Per triangle: a soft edge between two colours crosses it."""
        per_value = np.asarray([len(s) > 1 for s in self.shown()], bool)
        return per_value[self.state]

    def key(self) -> str:
        """A digest of the per-triangle colours (what a colour-bounded fill depends on)."""
        import hashlib

        return hashlib.blake2b(self.colours().tobytes(), digest_size=16).hexdigest()

    def extra(self) -> dict[int, str]:
        """Back to `Part.tri_extra` form."""
        idx = np.nonzero(self.state)[0]
        vals = self.values
        return {f: v for f, s in zip(idx.tolist(), self.state[idx].tolist()) if (v := vals[s]) is not None}


    def recolour(self, faces: np.ndarray, colours: np.ndarray) -> None:
        """Give each of `faces` its filament in `colours` (the part's own filament = no paint)."""
        for c in np.unique(colours).tolist():
            self.paint(faces[colours == c], None if c == self.extruder else paint_code(int(c)))


@dataclass(frozen=True)
class MirrorPlane:
    """A figure's plane of mirror symmetry: unit `normal` through `point` (file mm)."""
    normal: np.ndarray
    point: np.ndarray
    error: float            # mean distance from a reflected surface point to the surface, share of `size`
    size: float             # the figure's bounding-box diagonal, mm

    def reflect(self, p: np.ndarray) -> np.ndarray:
        return p - 2.0 * np.outer((p - self.point) @ self.normal, self.normal)

    def reflect_dir(self, d: np.ndarray) -> np.ndarray:
        return d - 2.0 * np.outer(d @ self.normal, self.normal)


def surface_points(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` points uniformly over the triangles `tri` (k, 3, 3): a triangle by area, a point in it by area."""
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cum = np.cumsum(area)
    pick = np.minimum(np.searchsorted(cum, rng.random(n) * cum[-1]), len(tri) - 1)
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    t = tri[pick]
    return np.asarray(t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0]))


def mirror_plane(points: np.ndarray, surface: np.ndarray | None = None, up: np.ndarray | None = None) -> MirrorPlane:
    """The plane `points` (surface samples) mirror best across: every direction of a Fibonacci hemisphere through
    the centre is scored by how far the reflected points land from `surface` (a dense sampling of it; default
    `points`), then the best few are refined by coordinate descent on direction and offset (halving steps).

    Distances come from a distance field of the surface (voxels of 1/MIRROR_GRID of the size, Euclidean distance
    transform, read with trilinear interpolation): a thousand planes cost a lookup each, where nearest-neighbour
    searches took 4-13 s. Against sparse samples instead of the surface every plane would score about their
    spacing, and the right one would not stand out."""
    from scipy.ndimage import distance_transform_edt, map_coordinates

    p = np.asarray(points, np.float64)
    target = p if surface is None else np.asarray(surface, np.float64)
    centre = target.mean(axis=0)
    size = float(np.linalg.norm(np.ptp(target, axis=0))) or 1.0
    cap = MIRROR_MISS * size
    vox = float(np.ptp(target, axis=0).max()) / MIRROR_GRID or 1.0
    lo = target.min(axis=0) - cap
    shape = np.ceil((target.max(axis=0) + cap - lo) / vox).astype(np.intp) + 1
    occupied = np.zeros(tuple(shape.tolist()), bool)
    occupied[tuple(np.round((target - lo) / vox).astype(np.intp).T)] = True
    field = (distance_transform_edt(~occupied) * vox).astype(np.float32)
    coarse = p[:: max(1, len(p) // MIRROR_COARSE)]

    def score(n: np.ndarray, c: np.ndarray, q: np.ndarray = p) -> float:
        # Trimmed: the best-matching MIRROR_KEEP of the points. A pose is rarely symmetric all over (a tail curled
        # to one side, a turned head); the plane is the one the rest of the figure agrees on. Points off the grid
        # count as a miss (`cap`).
        g = (q - 2.0 * np.outer((q - c) @ n, n) - lo) / vox
        d = np.minimum(map_coordinates(field, g.T, order=1, mode="constant", cval=cap), cap)
        k = max(1, int(len(d) * MIRROR_KEEP))
        return float(np.partition(d, k - 1)[:k].mean())

    k = np.arange(MIRROR_DIRECTIONS) + 0.5
    z = k / MIRROR_DIRECTIONS                                    # upper hemisphere: a plane's two normals agree
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z * z)
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    errs = np.array([score(n, centre, coarse) for n in dirs])
    # Refine the best few directions at least MIRROR_APART apart: with a coarse probe the right plane is often only
    # second or third (a boxy body mirrors across several planes about as well).
    starts: list[np.ndarray] = []
    for i in np.argsort(errs):
        if all(abs(float(dirs[i] @ d)) < np.cos(np.radians(MIRROR_APART)) for d in starts):
            starts.append(dirs[i])
        if len(starts) == MIRROR_STARTS:
            break
    found = [_refine(score, n, centre, size) for n in starts]
    n, c, best = min(found, key=lambda t: t[2])
    if up is not None:
        # A figure stands: its plane of symmetry is upright (left | right). A round body with no face mirrors about
        # as well across a level plane (top | bottom); an upright plane nearly as good as the best wins.
        upright = [t for t in found if abs(float(t[0] @ up)) < MIRROR_LEVEL]
        if upright:
            un, uc, ub = min(upright, key=lambda t: t[2])
            if ub <= MIRROR_UPRIGHT * best + 1e-9:
                n, c, best = un, uc, ub
    return MirrorPlane(n, c, best / size, size)


def _refine(score: Any, n: np.ndarray, c: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Coordinate descent on a plane's direction and offset, halving the steps until 0.25 degrees."""
    best = score(n, c)
    step, shift = np.radians(4.0), size * 0.02
    while step > np.radians(0.25):
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        moved = False
        for cand_n, cand_c in ((n + np.tan(step) * u, c), (n - np.tan(step) * u, c), (n + np.tan(step) * v, c),
                               (n - np.tan(step) * v, c), (n, c + shift * n), (n, c - shift * n)):
            cand_n = cand_n / np.linalg.norm(cand_n)
            e = score(cand_n, cand_c)
            if e < best - 1e-12:
                n, c, best, moved = cand_n, cand_c, e, True
                break
        if not moved:
            step, shift = step / 2, shift / 2
    return n, c, best


def apply_paint(parts: list[Part], ops: list[PaintOp], topo: "TopologyCache", min_area: float = 0.0) -> list[Part]:
    """Copies of `parts` with the paint operations applied in order (later ones win), then (`min_area` > 0, file
    mm²) painted patches smaller than that given the colour around them (`Topology.specks`)."""
    states: dict[int, PaintState] = {}
    for op in ops:
        if not (0 <= op.part < min(len(parts), len(topo))) or not (0 <= op.face < len(parts[op.part].faces)):
            continue                       # from another object or orientation of the file: ignore
        code = paint_code(op.filament) if op.filament else None
        # Every side's region is found against the paint before this operation, then all are painted.
        for cov in op_cover(topo, op, lambda i: states.setdefault(i, PaintState(parts[i]))):
            st = states.setdefault(cov.part, PaintState(parts[cov.part]))
            if cov.field is None and cov.over is not None:
                # Every corner inside: the whole triangle, but split ones only where they show the tapped colour.
                t = topo.part(cov.part)
                st.paint_field(t, Field(cov.faces, np.full(len(t.verts), -1.0)), op.filament, cov.over)
            elif cov.field is None:
                st.paint(cov.faces, code)
            else:
                st.paint_field(topo.part(cov.part), cov.field, op.filament, cov.over)
    if min_area > 0:
        for pi, st in states.items():
            faces, colours = topo.part(pi).specks(st.colours(), min_area)
            if len(faces):
                st.recolour(faces, colours)
    return [Part(p.name, p.extruder, p.vertices, p.faces, states[i].extra() if i in states else dict(p.tri_extra),
                 p.subtype) for i, p in enumerate(parts)]


@dataclass
class Cover:
    """What one side of a paint operation covers: whole `faces`, or (with `field`) the area of a soft edge."""
    part: int
    faces: np.ndarray
    field: Field | None = None
    over: int | None = None          # a fill bounded by colour: the soft edge paints only over this filament

    def inside(self, topo: "TopologyCache") -> np.ndarray:
        """The covered faces as whole triangles (a soft area: the ones mostly inside it)."""
        return self.field.inside(topo.part(self.part)) if self.field is not None else self.faces


def op_faces(topo: "TopologyCache", op: PaintOp, state: Any) -> list[tuple[int, np.ndarray]]:
    """(part, faces) an operation paints, as whole triangles (`op_cover`)."""
    return [(c.part, c.inside(topo)) for c in op_cover(topo, op, state)]


def op_cover(topo: "TopologyCache", op: PaintOp, state: Any) -> list[Cover]:
    """What an operation paints: its region, and with `mirror` the mirror image's (a spot with no match on the
    other side is painted on one side only). `state(i)` gives part i's current paint (fills stop at it).

    The wand, brush and stamp, and any operation grown or shrunk, paint with a soft edge (`Field`): a smooth line
    through the triangles, not their stepped outline. The other modes follow the mesh's own edges (creases,
    separate pieces) and paint whole triangles."""
    n = topo.n_faces(op.part)
    sides = [(op.part, op.face, tuple(f for f in op.path if 0 <= f < n))]
    if op.mirror:
        twin = topo.mirrored(op.part, op.face, sides[0][2])
        if twin is not None and twin[:2] != (op.part, op.face):
            sides.append(twin)
    out: list[Cover] = []
    for k, (pi, fi, path) in enumerate(sides):
        t = topo.part(pi)
        if op.mode == "stamp":
            st = np.asarray(op.stamp, float)
            centre, u, v = st[0:3], st[3:6], st[6:9]
            if k == 1:
                # Reflected, the axes' cross product (the way it is seen) would point into the surface: v turns.
                plane = topo.mirror()
                centre = plane.reflect(centre[None])[0]
                u, v = plane.reflect_dir(u[None])[0], -plane.reflect_dir(v[None])[0]
            if op.grow:
                a, b = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                u, v = u * max(a + op.grow, 1e-3) / a, v * max(b + op.grow, 1e-3) / b
            out.append(Cover(pi, np.zeros(0, np.int64), t.stamp_field(fi, centre, u, v)))
            continue
        if op.mode == "brush":
            fld = t.brush_field([fi, *path], max(op.radius + op.grow, 1e-3))
            out.append(Cover(pi, fld.faces, fld))
            continue
        if op.mode == "wand" and k == 1:
            # The twin's candidates differ a little from the tapped side's, so a step number would not pick the
            # same feature there: take the twin candidate nearest the tapped side's area.
            area = float(topo.part(out[0].part).areas[out[0].inside(topo)].sum())
            faces = topo.wand_like(pi, fi, area, state(pi))
        else:
            faces = region_faces(topo, pi, fi, op.mode, op.angle, op.radius, path, state(pi), op.level)
        if op.mode != "wand" and not op.grow:
            paint = state(pi)
            over = None
            if paint is not None and op.mode == "fill":
                # A bucket fill also takes the tapped colour's side of the soft edges round it: a triangle a soft edge
                # crosses that shows mostly the other colour is not in the region, yet holds a sliver of this one
                # (a soft edge can run through several such triangles in a row: followed while they hold it).
                c = int(paint.colours()[fi])
                inside = np.zeros(t.n_faces, bool)
                inside[faces] = True
                sliver = paint.split() & paint.holds(np.arange(t.n_faces), c) & ~inside
                pa, pb = t.pairs[:, 0], t.pairs[:, 1]
                grown = inside.copy()
                for _ in range(SLIVER_RINGS):
                    ring = np.zeros(t.n_faces, bool)
                    ring[pa[grown[pb]]] = True
                    ring[pb[grown[pa]]] = True
                    ring &= sliver & ~grown
                    if not ring.any():
                        break
                    grown |= ring
                if (grown & ~inside).any():
                    faces = np.flatnonzero(grown)
            if paint is not None and op.mode in BOUNDED and paint.has_splits(faces):
                over = int(paint.colours()[fi])
            out.append(Cover(pi, faces, None, over))
            continue
        mask = np.zeros(t.n_faces, bool)
        mask[faces] = True
        paint = state(pi)
        over = None
        if paint is not None and op.mode in BOUNDED:
            colour = paint.colours()
            other = colour != colour[fi]
            if other.any():
                # Bounded by colour: where the area meets another colour, its edge is that colour's own (painted
                # before, maybe soft). The field reaches two rings into the other colour there and paints only over
                # the tapped colour, so it fills up to that edge instead of drawing a second line beside it.
                over = int(colour[fi])
                pa, pb = t.pairs[:, 0], t.pairs[:, 1]
                for _ in range(2):
                    wider = mask.copy()
                    wider[pa[mask[pb] & other[pa]]] = True
                    wider[pb[mask[pa] & other[pb]]] = True
                    mask = wider
        fld = t.soft(mask, WAND_SMOOTH, op.grow)
        out.append(Cover(pi, fld.faces, fld, over))
    return out


class TopologyCache:
    """Topology and region labels of one object's parts, computed on first use."""

    def __init__(self, parts: list[Part]):
        self._parts = parts
        self._topo: dict[int, Topology] = {}
        self._labels: dict[tuple[int, str, float], np.ndarray] = {}
        self._fills: dict[tuple[Any, ...], np.ndarray] = {}
        self._mirror: MirrorPlane | None = None
        self._trees: list[Any] | None = None
        self._up: tuple[float, float, float] | None = None

    def set_up(self, up: np.ndarray) -> None:
        """The figure's up direction in these parts' frame (its rotation); the mirror plane prefers upright planes."""
        key = tuple(round(float(x), 6) for x in up)
        if key != self._up:
            self._up = (key[0], key[1], key[2])
            self._mirror = None

    def __len__(self) -> int:
        """Parts it covers: the file's own (not the conform step's flat base, which taps cannot paint)."""
        return len(self._parts)

    def n_faces(self, i: int) -> int:
        return len(self._parts[i].faces)

    def mirror(self) -> MirrorPlane:
        """The figure's mirror plane (all its printed parts together, so a left eye modelled as its own part
        finds the right one)."""
        if self._mirror is None:
            tri = np.vstack([p.vertices[p.faces] for p in self._parts if p.subtype == "normal_part" and len(p.faces)])
            # Points spread evenly over the surface (dense detail must not outweigh broad surfaces, and a low-poly
            # mesh's few triangle centres would be too sparse): a few to reflect, many to land on.
            rng = np.random.default_rng(1)
            self._mirror = mirror_plane(surface_points(tri, MIRROR_SAMPLES, rng),
                                        surface_points(tri, MIRROR_TARGET, rng),
                                        None if self._up is None else np.asarray(self._up))
        return self._mirror

    def mirrored(self, part: int, face: int, path: tuple[int, ...] = ()) -> tuple[int, int, tuple[int, ...]] | None:
        """The (part, face, path) across the mirror plane: the triangle facing the same way nearest the
        reflected centre (path triangles: nearest centre, on that part). None when the figure is not symmetric
        enough, or nothing lies near the reflected spot (a tail curled to one side)."""
        plane = self.mirror()
        if plane.error > MIRROR_MAX_ERROR:
            return None
        if self._trees is None:
            self._trees = centroid_trees(self._parts)
        tri = self._parts[part].vertices[self._parts[part].faces[face]]
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        q = plane.reflect(tri.mean(axis=0)[None])[0]
        mp, mf, dist = pick_face(self._parts, q, plane.reflect_dir(nrm[None])[0], trees=self._trees)
        if mp < 0 or dist > MIRROR_MISS * plane.size:
            return None
        mpath: tuple[int, ...] = ()
        if path:
            # A stroke's other points: the nearest triangle centre on the mirrored triangle's part (a brush is a
            # radius around each, so the nearest centre is close enough), in stroke order, without repeats.
            cen = self._parts[part].vertices[self._parts[part].faces[np.asarray(path)]].mean(axis=1)
            _, idx = self.part(mp).tree.query(plane.reflect(cen))
            mpath = tuple(dict.fromkeys(int(i) for i in np.atleast_1d(idx) if int(i) != mf))
        return mp, mf, mpath

    def part(self, i: int) -> Topology:
        if i not in self._topo:
            self._topo[i] = topology(self._parts[i])
        return self._topo[i]

    def labels(self, i: int, mode: str, angle: float) -> np.ndarray:
        if mode in ("piece", "fill", "face"):     # no angle; a fill's region is its piece
            mode, angle = ("piece" if mode == "fill" else mode), 0.0
        key = (i, mode, round(float(angle), 3))
        if key not in self._labels:
            self._labels[key] = self.part(i).labels(mode, angle)
        return self._labels[key]


    def wand(self, i: int, face: int, level: int, paint: PaintState | None) -> np.ndarray:
        """`Topology.wand` inside the tapped colour, cached by the paint it saw (an analysis replays every step)."""
        colour = paint.colours() if paint is not None else None
        key = (i, face, int(level), paint.key() if paint is not None else None, "wand")
        if key not in self._fills:
            if len(self._fills) >= _FILL_CACHE:
                self._fills.pop(next(iter(self._fills)))
            allowed = None if colour is None or (colour == colour[face]).all() else colour == colour[face]
            self._fills[key] = self.part(i).wand(face, allowed, level)
        return self._fills[key]

    def wand_like(self, i: int, face: int, area: float, paint: PaintState | None) -> np.ndarray:
        """The wand candidate at `face` whose area is nearest `area` (a mirrored wand step's other side)."""
        colour = paint.colours() if paint is not None else None
        allowed = None if colour is None or (colour == colour[face]).all() else colour == colour[face]
        t = self.part(i)
        cands = t.wand_candidates(face, allowed)
        if not cands:
            return np.asarray([face])
        return min(cands, key=lambda c: abs(float(t.areas[c[2]].sum()) - area))[2]

    def fill(self, i: int, face: int, mode: str, angle: float, paint: PaintState) -> np.ndarray:
        """A "part"/"region" fill that also stops where the colour changes (see `region_faces`)."""
        lab = self.labels(i, mode, angle)
        region = np.nonzero(lab == lab[face])[0]
        colour = paint.colours()
        if (colour[region] == colour[face]).all():
            return region                  # one colour throughout (the usual case): no extra work
        key = (i, face, mode, round(float(angle), 3), paint.key())
        if key not in self._fills:
            if len(self._fills) >= _FILL_CACHE:
                self._fills.pop(next(iter(self._fills)))
            self._fills[key] = self.part(i).bounded(region, face, colour)
        return self._fills[key]


def region_faces(topo: TopologyCache, part: int, face: int, mode: str, angle: float,
                 radius: float = 0.0, path: tuple[int, ...] = (), paint: PaintState | None = None,
                 level: int = 0) -> np.ndarray:
    """The faces a paint operation covers. With `paint` (the part's current paint), the wand and the "part",
    "region" and "fill" fills stop where the visible colour changes, like a bucket fill."""
    if mode == "brush":
        return topo.part(part).brush([face, *path], radius)
    if mode == "wand":
        return topo.wand(part, face, level, paint)
    if paint is not None and mode in BOUNDED:
        return topo.fill(part, face, mode, angle, paint)
    lab = topo.labels(part, mode, angle)
    return np.nonzero(lab == lab[face])[0]


@dataclass(frozen=True)
class Piece:
    """One connected patch of a colour: a "fill" step tapped at (`part`, `face`) covers it."""
    part: int
    face: int
    area: float          # file mm²
    faces: int


def colour_pieces(topo: TopologyCache, parts: list[Part], filament: int, default_extruder: int = 1) -> list[Piece]:
    """Every connected patch of the figure showing `filament` (each triangle by its dominant colour; patches meeting
    only at a corner are separate), largest first: white eyes and a white tail are three. Each seed is the patch's
    largest triangle showing the filament whole, so a "fill" step from it recolours exactly that patch. A part with
    no filament of its own (0) prints in `default_extruder`, the object's."""
    out: list[Piece] = []
    for i in range(min(len(parts), len(topo))):
        st = PaintState(parts[i])
        colour = st.colours()
        sel = (colour == filament) | ((colour == 0) & (filament == default_extruder))
        if not sel.any():
            continue
        t = topo.part(i)
        lab = t._components(sel[t.pairs[:, 0]] & sel[t.pairs[:, 1]])
        idx = np.flatnonzero(sel)
        labs = lab[idx]
        area = np.bincount(labs, weights=t.areas[idx])
        count = np.bincount(labs)
        # Seed per patch: whole (not split) triangles first, then the largest.
        order = np.lexsort((-t.areas[idx], st.split()[idx], labs))
        first = np.ones(len(order), bool)
        first[1:] = labs[order][1:] != labs[order][:-1]
        for f in idx[order][first].tolist():
            out.append(Piece(i, int(f), float(area[lab[f]]), int(count[lab[f]])))
    return sorted(out, key=lambda pc: -pc.area)


def centroid_trees(parts: list[Part]) -> list[Any]:
    """Per part, a KD-tree of its triangle centroids for `pick_face` (None for parts taps skip)."""
    from scipy.spatial import cKDTree

    return [cKDTree(p.vertices[p.faces].mean(axis=1)) if len(p.faces) and p.subtype == "normal_part" else None
            for p in parts]


def pick_face(parts: list[Part], point: np.ndarray, normal: np.ndarray | None = None,
              candidates: int = 24, trees: list[Any] | None = None) -> tuple[int, int, float]:
    """(part index, face index, distance) of the triangle nearest `point`.

    Exact point–triangle distances over the triangles with the nearest
    centroids (pure numpy + a KD-tree, so it runs in the browser too). With
    `normal` (the view's surface normal), triangles facing away are skipped, so
    a tap on a thin ear picks its near side. Ties go to the earlier part, which
    is the one the slicer prints where parts overlap. `trees` (`centroid_trees(parts)`) saves
    building them per tap (a 300k-triangle part: 0.2 s).
    """
    trees = trees if trees is not None else centroid_trees(parts)
    best = (-1, -1, np.inf)
    p = np.asarray(point, np.float64)
    for pi, (part, tree) in enumerate(zip(parts, trees)):
        if tree is None:
            continue
        k = min(candidates, len(part.faces))
        _, idx = tree.query(p, k=k)
        idx = np.atleast_1d(idx)
        tri = part.vertices[part.faces[idx]]
        if normal is not None:
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            facing = (n @ np.asarray(normal, np.float64)) > 0
            if facing.any():
                idx, tri = idx[facing], tri[facing]
        cp = trimesh.triangles.closest_point(tri, np.repeat(p[None], len(idx), axis=0))
        d = np.linalg.norm(cp - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < best[2] - 1e-9:
            best = (pi, int(idx[j]), float(d[j]))
    return best


def region_colours(n: int, seed: int = 7) -> np.ndarray:
    """Distinct pastel RGB (0–255) per region id, for the "show regions" preview (golden-ratio hues)."""
    import colorsys

    start = float(np.random.default_rng(seed).random())
    rgb = [colorsys.hsv_to_rgb((start + k * 0.618033988749895) % 1.0, 0.45, 0.92) for k in range(n)]
    return (np.asarray(rgb, np.float64).reshape(-1, 3) * 255).astype(np.uint8)
