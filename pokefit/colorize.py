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
  * "face": that triangle only.

"part" and "region" fills also stop where the visible colour changes (bucket
fill): a sculpt whose eyes or belly have no crease all the way round (a
generated or smoothed figure) gets them by outlining with a brush stroke and
tapping inside.

Any operation can also
  * `mirror`: paint the same spot on the other side of the figure, across its
    plane of mirror symmetry (found from the shape: `MirrorPlane`), so both
    eyes, cheeks or paws take one tap or one stroke;
  * `grow`: widen (mm > 0) or narrow (mm < 0) the painted area along the
    surface: an eye fill that stops short of its socket's rim grows into it.

`despeckle` (run after all operations) gives a painted patch smaller than a
set area the colour around it: a stray dab on a claw would otherwise cost a
filament change on every layer it spans.

Filament 0 removes paint (the part's own filament prints there). The result
is written as Bambu/Prusa per-triangle paint (`paint_color`, whole-triangle
states), so it opens in Bambu Studio as ordinary painted colour and survives
the pocket cut (`remesh_part` carries per-triangle attributes over).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .model3mf import Part

MODES = ("wand", "part", "region", "piece", "fill", "brush", "face")
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
MIRROR_MISS = 0.04          # ... and a mirrored spot within this share of the size of the surface
_FILL_CACHE = 64            # colour-bounded fills kept per object (keyed by the paint state they saw)
_PAINT_RE = re.compile(r'\s*paint_color="[^"]*"')


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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaintOp":
        op = cls(int(d["part"]), int(d["face"]), int(d["filament"]), str(d.get("mode", "part")),
                 float(d.get("angle", DEFAULT_ANGLE)), float(d.get("radius", 0.0)),
                 tuple(int(f) for f in d.get("path") or ()), bool(d.get("mirror", False)),
                 float(d.get("grow", 0.0)), int(d.get("level", 0)))
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
        return d


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
    _tree: Any = None          # KD-tree of `centroids`, built on first use (`tree`)
    _wall: Any = None          # `wall`, built on first use

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

    def grown(self, faces: np.ndarray, mm: float) -> np.ndarray:
        """`faces` widened along the surface by `mm` (> 0) or narrowed (< 0): the triangles whose centre lies within
        that distance of the region's edge, walking from triangle to neighbouring triangle (Dijkstra over centre-to-
        centre steps, so the band follows the surface and never jumps a gap), then one smoothing pass on the new
        edge. Measured as the union of balls round the edge's triangles instead, the edge came out zigzagged."""
        if not mm or not len(faces):
            return faces
        from scipy.sparse.csgraph import dijkstra

        inside = np.zeros(self.n_faces, bool)
        inside[faces] = True
        rim_in, rim_out = self._rim(inside)
        start = rim_in if mm > 0 else rim_out
        # Walk only on the side that changes: outward when growing, inward when shrinking.
        side = ~inside if mm > 0 else inside
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        walk = (side[a] | np.isin(a, start)) & (side[b] | np.isin(b, start))
        w = np.linalg.norm(self.centroids[a[walk]] - self.centroids[b[walk]], axis=1) + 1e-9
        g = coo_matrix((w, (a[walk], b[walk])), shape=(self.n_faces, self.n_faces)).tocsr()
        dist = dijkstra(g, directed=False, indices=start, limit=abs(mm), min_only=True)
        band = np.isfinite(dist) & side
        out = inside | band if mm > 0 else inside & ~band
        return np.nonzero(self._smooth(out, band))[0]

    def _smooth(self, mask: np.ndarray, where: np.ndarray) -> np.ndarray:
        """One majority pass over the triangles in `where`: each takes the state of most of its edge neighbours (a
        triangle has three), which removes the one-triangle notches whole-triangle paint leaves on an edge."""
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        on = np.bincount(a, weights=mask[b], minlength=self.n_faces) + np.bincount(b, weights=mask[a], minlength=self.n_faces)
        deg = np.bincount(a, minlength=self.n_faces) + np.bincount(b, minlength=self.n_faces)
        out = mask.copy()
        sel = where & (deg > 0)
        out[sel] = on[sel] * 2 > deg[sel]
        tie = sel & (on * 2 == deg)
        out[tie] = mask[tie]
        return out

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
    return Topology(len(part.faces), pairs, angles, convex, np.asarray(m.triangles_center, np.float64),
                    np.asarray(m.area_faces, np.float64), np.asarray(m.face_normals, np.float64))


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
    """One majority pass (see `Topology._smooth`) on a graph given as pairs."""
    a, b = pairs[:, 0], pairs[:, 1]
    on = np.bincount(a, weights=mask[b], minlength=n) + np.bincount(b, weights=mask[a], minlength=n)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    out = mask.copy()
    sel = where & (deg > 0)
    out[sel] = on[sel] * 2 > deg[sel]
    tie = sel & (on * 2 == deg)
    out[tie] = mask[tie]
    return out


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


def mirror_plane(points: np.ndarray, surface: np.ndarray | None = None) -> MirrorPlane:
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
        for pi, faces in op_faces(topo, op, lambda i: states.setdefault(i, PaintState(parts[i]))):
            states[pi].paint(faces, code)
    if min_area > 0:
        for pi, st in states.items():
            faces, colours = topo.part(pi).specks(st.colours(), min_area)
            if len(faces):
                st.recolour(faces, colours)
    return [Part(p.name, p.extruder, p.vertices, p.faces, states[i].extra() if i in states else dict(p.tri_extra),
                 p.subtype) for i, p in enumerate(parts)]


def op_faces(topo: "TopologyCache", op: PaintOp, state: Any) -> list[tuple[int, np.ndarray]]:
    """(part, faces) an operation paints: its region, and with `mirror` the mirror image's (a spot with no match
    on the other side is painted on one side only). `state(i)` gives part i's current paint (fills stop at it)."""
    n = topo.n_faces(op.part)
    sides = [(op.part, op.face, tuple(f for f in op.path if 0 <= f < n))]
    if op.mirror:
        twin = topo.mirrored(op.part, op.face, sides[0][2])
        if twin is not None and twin[:2] != (op.part, op.face):
            sides.append(twin)
    out: list[tuple[int, np.ndarray]] = []
    for k, (pi, fi, path) in enumerate(sides):
        if op.mode == "wand" and k == 1:
            # The twin's candidates differ a little from the tapped side's, so a step number would not pick the
            # same feature there: take the twin candidate nearest the tapped side's area.
            area = float(topo.part(out[0][0]).areas[out[0][1]].sum())
            faces = topo.wand_like(pi, fi, area, state(pi))
        else:
            faces = region_faces(topo, pi, fi, op.mode, op.angle, op.radius, path, state(pi), op.level)
        out.append((pi, topo.part(pi).grown(faces, op.grow) if op.grow else faces))
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
                                        surface_points(tri, MIRROR_TARGET, rng))
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
