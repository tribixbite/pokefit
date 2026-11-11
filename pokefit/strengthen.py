"""Strengthen thin parts: thicken whatever is thinner than a minimum at the print size (a flame's thin
tips, toes and claws, an ear, and the narrow join where a small tail tip meets a big flame), blending
the change into the surface around it so a join gets a fillet-like taper rather than a step.

Why: a figure scaled down to fit a ~27 mm ball has features a few tenths of a millimetre thick at
print size. Such a join holds only a few extrusion lines and snaps off (a Charmander's tail flame,
an Eevee's ears, a Pikachu's tail).

How:

1. **Local thickness.** The solid parts are voxelised at VOXEL mm (print size) by ray parity along z
   (`magnets._crossings`: every triangle crossing a column, as an entry or an exit), and a Euclidean
   distance transform gives each inside voxel its distance to the surface. A vertex's local radius
   is the largest distance found along its inward normal within the target thickness: about half the
   thickness of a sheet, the radius of a rod, and "thick enough" for a body.
2. **Taper.** Each thin feature (connected vertices thinner than the target) gets the full target
   where it meets thick material (its root: the join that breaks) and TIP x target at its far end,
   in proportion to the distance over its surface from the root. A fine tip stays fine; its base
   gets stronger. A neck (thin between two thick parts) has no free end and gets the full target.
3. **Push size.** A thin vertex's surface is to move out by half its shortfall from that (both sides of a sheet
   move, so the sheet ends at the target), spread over BLEND x target mm of surface around it, fading (each pass
   a vertex keeps the larger of its own push and SPREAD x its neighbours' mean), then smoothed: the thickened
   part meets its body with a taper, which is where a join is weakest.
4. **Offset as a solid** (`_shell`, `with_shells`). The surface is not moved vertex by vertex: the part gains the
   solid inside its offset surface, {x : sdf(x) < push at the nearest surface point, push >= FADE},
   built on a SHELL_VOXEL grid (distance to dense samples of the surface, inside by ray parity) as a closed mesh
   (`_marching_tets`) and united with the part (manifold3d). Paint and texture colour carry over to the new
   triangles from the nearest old one (`model3mf.remesh_part`). An offset of a distance field is smooth however
   fine the relief under it: moving vertices instead pushed an ear's raised rim and the recessed panel inside it
   (0.7 mm Jolteon ear, 2 mm target) into each other and left the surface scaly; a groove narrower than the push
   (an incised line) now fills instead of closing with a ragged seam.

Only across: a vertex is pushed only where its inward ray leaves the solid again within about its own thickness
(ACROSS), so the push widens the feature across its thin direction. On a rim, an edge or a point (an ear's tip,
the point of a lightning-bolt tail, the edge of a flat fin) "outward" runs along the feature: the ray runs down
it and leaves far away. Pushing those grew the point outward and rounded it into a knob (worst at 2.5-4 mm);
they now keep their place, and the sheet behind them still thickens. Not into a gap (`clear`): the surface moves
under half the open space in front of it (a claw against the body).

Single features (`Local`, from joints.py's weak points) are stretched instead (`_stretch`): vertex moves only, so the
triangles, their paint, the outline and the relief stay. Per vertex: the feature's thin directions (the second moments
of the normals over STRETCH_AXES targets of surface: n n^T is the same on both faces of a sheet, so a sheet has one,
a rod two, and the length none), the feature's thickness across them (chords from its faces, not its rim), and a move
of (k - 1) x half that thickness along P n (the smoothed normal's part across the feature: a sheet's faces move straight
out, its rim barely, a rod all round), k = the thickness needed / the thickness there. The thickness needed tapers as
sqrt(the length left to the tip) (bending moment linear, strength ~ thickness squared). On a sheet (`_even_sheet`) a
bent one is thickened on its convex side only (a concave face pushed out along its normal shrinks: an ear's painted
inner panel lost 11 % of its area) and the sideways part is spread linearly across the faces from the rim (a panel
widens with its outline); Leafeon's ear panels then lose about 4 % of their area instead of 11 % (a test's bent fin
keeps its concave face within 2 %). The moves are smoothed over STRETCH_SMOOTH targets with the cotangent Laplacian
(`_Graph.cot_lap`) in STRETCH_STEPS heat steps, and where two features' moves overlap they are blended by length^
OVERLAP_POWER (`_stretched`). A shell union (below) instead filled a leaf tail's veins, crumpled its blade and swelled an
ear's rims and shrank the panel painted inside it (Leafeon, 2026-11-09); stretched, the veins and panel stay, and the
grade came out at least as strong (graded with a 6 mm probe, before joints.py read sections by projected area: tail
1.6 -> 3.0 N vs 2.4 N, ears 2.2 -> 8.3-8.6 N vs 8.8 N; with it and `_even_sheet`, probe 3: ears 2.0-2.4 -> 5.8 N). Moving vertices along plain normals was tried before for the whole figure and left scaly relief (see 4): the
stretch's moves are smooth fields and only across the feature.

Limits: the thickness is measured at one print size (the height asked for, else the largest that fits); a smaller
print has thinner features. The shell is a SHELL_VOXEL mesh (0.1 mm facets at print size) and it fades in from
FADE mm, a step under a layer. Relief finer than the push under it is filled, not kept (that is what makes it
thicker). Nothing grows below the bed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, diags

from .magnets import _crossings
import trimesh
import trimesh.smoothing

from .model3mf import Part, digest, remesh_part, row_ids

VOXEL = 0.15         # mm at print size
SAMPLES = 12         # steps along the inward normal when measuring a vertex's local radius
ACROSS = (1.5, 3.0)  # full push where the inward ray leaves the solid within 1.5 x the thickness, none past 3 x
EXIT_SLACK = 2       # ... plus this many voxels (`_across`)
GROOVE_ROUNDS = 1    # passes filling the gate's narrow valleys (`_fill_grooves`)
GROOVE_DEG = 25.0    # ... between neighbours whose smoothed normals are this close
REACH = 1.5          # ...up to this many targets deep (a 90-degree corner's distance grows only as depth / sqrt 3)
BLEND = 1.0          # the push blends out over this many target thicknesses of surface
SPREAD = 0.9         # per pass, a vertex takes this share of its neighbours' mean push (if larger than its own)
SMOOTH = 6           # fewest smoothing passes over the push
TIP = 0.4            # a thin feature's far end is thickened to this share of the target (a taper)
ROOT_MIN = 6         # vertices a join needs to count (fewer: measuring noise)
NORMAL_ROUNDS = 80   # most smoothing passes over the push direction and the push
MAX_VOXELS = 30_000_000  # coarser voxels beyond this (a very large figure or a fine target)
GAP_SHARE = 0.45     # the surface moves at most this share of the open space in front of it (`clear`)
SHELL_VOXEL = 0.1    # mm at print size: the grid the gained solid is built on (`_shell`)
FADE = 0.05          # mm: the shell starts where the push reaches this (under a layer: no visible step)
TIDY_ROUNDS = 6      # Taubin smoothing passes over the shell's grid noise (`_tidy`)
TIDY_ERROR = 0.01    # mm the shell may move when its triangles are thinned
MAX_SHELL_NODES = 16_000_000   # a coarser shell grid beyond this (memory: the browser engine is 32-bit)
DEPTH = 2            # the shell reaches its push plus this many grid steps under the surface (`_shell`)
NEAREST = 4          # the samples whose triangles give a grid point's exact distance (`_shell`)
BALL = 0.1           # mm: samples pushed this far or more also grow in front of them (`_shell`: concave joins)
BALLS = 8            # ... the candidates tried per grid point
GROW_DEG = 45.0      # ... by their full push within this angle of their normal
FACE_DEG = 80.0      # ... fading to nothing at this angle
COARSE = 2           # the shell's first pass fills every COARSE-th grid point (`_shell`)
# A single feature (`Local`, `_stretch`): in full to a quarter target past its last weak section, eased (smoothstep) to
# nothing a target further on.
LOCAL_REACH = (0.25, 1.25)
STRETCH_AXES = 1.0   # its thin directions: the normals' second moments smoothed over this many targets of surface
STRETCH_SMOOTH = 0.5  # the move smoothed over this many targets (no creases where the thin directions turn; 1.0 took
#                       13 % off a rod's gain: the peak at its weak section flattened)
STRETCH_SHIFT_SMOOTH = 1.0  # a bent sheet's shift to its convex side smoothed over this many targets first (at 0.5 it
#                             left a 0.1 mm ripple on an ear's back; a shift adds no thickness, so this costs none)
STRETCH_FACE = 4     # the local thickness is a mean of chords weighted by |P n|^this: a sheet's faces, not its rim
STRETCH_KNOWN = 0.02  # ... measured only where that weight averages at least this (else noise: no move)
STRETCH_MIN = 1e-4   # moves under this share of a target are dropped (the smoothing's far tail: the rest stays exact)
STRETCH_SHEET = 0.5  # a feature is a sheet where its second thin direction is under this share of the first
STRETCH_RIM = 0.5    # a sheet vertex whose move is over this share sideways is on its rim (`_even_sheet`)
STRETCH_CURVE = (0.05, 0.2)  # targets / bend radius: a sheet thickened half each side .. all on its convex side
OVERLAP_POWER = 4    # overlapping features' moves weigh in by their length to this power (`_stretched`)
# A neck holding a thick piece (`_widen_neck`): its centre line from bands NECK_BAND targets apart (at least 2 voxels)
# of at least NECK_RING_MIN voxels, carried on NECK_BACK targets behind the feature's start; each ring's width from a
# flat cut of the solid within NECK_REACH targets of the line, the narrowest over NECK_DIRS directions; grown to a soft
# maximum (NECK_SOFT targets soft) of its width and the target, eased in over the line's first NECK_EASE targets,
# smoothed over NECK_SMOOTH targets (the stretch's 0.5 flattened the neck's peak).
NECK_BAND = 0.08
NECK_RING_MIN = 6
NECK_BACK = 1.0
NECK_REACH = 1.0
NECK_DIRS = 36
NECK_SOFT = 0.1
NECK_EASE = 0.25
NECK_SMOOTH = 0.15
COT_FLOOR = 0.01    # `_Graph`'s smallest edge weight (a cotangent weight under it, from an obtuse sliver, is raised)
STRETCH_STEPS = 3   # the final smoothing of the moves in this many heat steps (one step's kernel has a cusp: a jump in
#                      the moves, e.g. where the thickness is first measured, came out as a 0.05 mm crease on a forehead)
# The shell's search for samples pushed in front of a grid point (4D power distance) returns candidates within
# (1 + KD_EPS) of the nearest (scipy `eps`). The Vaporeon at 2 mm: 47 -> 35 s single-threaded (as in the browser engine),
# its surface within 0.08 mm of the exact build's (0.07 between two exact builds, by sampling); 1.0 moved 0.13. The same on
# the 3D distance search (47 -> 27 s) left a thin fin unthickened (tests/test_strengthen): not there.
KD_EPS = 0.5


@dataclass
class StrengthenInfo:
    target_mm: float = 0.0          # minimum thickness asked for
    scale: float = 0.0              # print scale it was measured at
    thin_share: float = 0.0         # share of the surface (vertices) thinner than the target
    max_push_mm: float = 0.0        # largest outward move
    voxel_mm: float = 0.0
    features: int = 0               # thin features thickened to their own minimum (`Local`)

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class Voxels:
    """Solid voxels of closed meshes: `inside[i, j, k]` is the cell centred at (xs[j], ys[i], zs[k]), `h` apart."""
    inside: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    h: float

    def index(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(i, j, k) of the cells holding points `p` (clipped to the grid)."""
        j = np.clip(np.round((p[:, 0] - self.xs[0]) / self.h).astype(np.intp), 0, len(self.xs) - 1)
        i = np.clip(np.round((p[:, 1] - self.ys[0]) / self.h).astype(np.intp), 0, len(self.ys) - 1)
        k = np.clip(np.round((p[:, 2] - self.zs[0]) / self.h).astype(np.intp), 0, len(self.zs) - 1)
        return i, j, k

    def centres(self, idx: tuple[np.ndarray, ...]) -> np.ndarray:
        """Centres of cells (i, j, k)."""
        return np.c_[self.xs[idx[1]], self.ys[idx[0]], self.zs[idx[2]]]


def voxels(meshes_v: list[np.ndarray], meshes_f: list[np.ndarray], voxel: float) -> Voxels:
    """The meshes' solid (ray parity along z: `magnets._crossings`) on a `voxel` grid, coarser when it would hold
    over MAX_VOXELS cells, with two empty cells round it."""
    tri = np.vstack([v[f] for v, f in zip(meshes_v, meshes_f)]).astype(np.float64)
    lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
    h = voxel
    while np.prod((hi - lo) / h + 5) > MAX_VOXELS:
        h *= 1.25
    pad = 2 * h
    # Irrational offsets keep cell centres off mesh edges (as in magnets.height_field).
    xs = np.arange(lo[0] - pad, hi[0] + pad, h) + 0.1234567 * h
    ys = np.arange(lo[1] - pad, hi[1] + pad, h) + 0.3183099 * h
    zs = np.arange(lo[2] - pad, hi[2] + pad, h) + 0.2718281 * h
    cell, z, sign = _crossings(tri, xs, ys)
    acc = np.zeros((len(ys) * len(xs), len(zs) + 1), np.int16)
    np.add.at(acc, (cell, np.searchsorted(zs, z)), sign.astype(np.int16))
    inside = (np.cumsum(acc, axis=1)[:, :-1] > 0).reshape(len(ys), len(xs), len(zs))
    return Voxels(inside, xs, ys, zs, h)


def local_radius(meshes_v: list[np.ndarray], meshes_f: list[np.ndarray], points: np.ndarray,
                 inward: np.ndarray, reach: float, voxel: float,
                 grid: Voxels | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point (on the surface): the largest distance to the surface found along `inward` up to `reach` (about
    half the local thickness, capped near `reach` for thick material), how far along `inward` the solid ends
    (`reach` when it does not end within it), and how far OUT (against `inward`) the next solid is (`reach` when
    none is within it: open space). All in the same units. `grid`: the meshes' `voxels` at `voxel`, if made already."""
    grid = grid if grid is not None else voxels(meshes_v, meshes_f, voxel)
    inside, xs, ys, zs, h = grid.inside, grid.xs, grid.ys, grid.zs, grid.h
    dist = ndimage.distance_transform_edt(inside, sampling=h).astype(np.float32)
    r = np.zeros(len(points))
    exit_at = np.full(len(points), reach)
    entered = np.zeros(len(points), bool)
    # Steps of one voxel (at least SAMPLES, at most 64): where the far side is must be known to a voxel or two.
    for t in np.linspace(h, reach, int(np.clip(np.ceil(reach / h), SAMPLES, 64))):
        p = points - inward * t
        j = np.clip(np.round((p[:, 0] - xs[0]) / h).astype(np.intp), 0, len(xs) - 1)
        i = np.clip(np.round((p[:, 1] - ys[0]) / h).astype(np.intp), 0, len(ys) - 1)
        k = np.clip(np.round((p[:, 2] - zs[0]) / h).astype(np.intp), 0, len(zs) - 1)
        r = np.maximum(r, dist[i, j, k])
        ins = inside[i, j, k]
        # The first sample outside after one inside: the far side (a first sample just outside is the surface).
        leave = entered & ~ins & (exit_at >= reach)
        exit_at[leave] = t
        entered |= ins
    # Outward: the first solid sample after one outside (the first sample may still be inside the own surface).
    clear = np.full(len(points), reach)
    out_seen = np.zeros(len(points), bool)
    for t in np.linspace(h, reach, int(np.clip(np.ceil(reach / h), SAMPLES, 64))):
        p = points + inward * t
        j = np.clip(np.round((p[:, 0] - xs[0]) / h).astype(np.intp), 0, len(xs) - 1)
        i = np.clip(np.round((p[:, 1] - ys[0]) / h).astype(np.intp), 0, len(ys) - 1)
        k = np.clip(np.round((p[:, 2] - zs[0]) / h).astype(np.intp), 0, len(zs) - 1)
        ins = inside[i, j, k]
        hit = out_seen & ins & (clear >= reach)
        clear[hit] = t
        out_seen |= ~ins
    return r, exit_at, clear


class _Graph:
    """A part's surface as a vertex graph: coincident vertices merged (STL soups repeat a vertex per triangle, and
    a push must move them together)."""

    def __init__(self, v: np.ndarray, f: np.ndarray):
        q = np.round(v / (np.ptp(v, axis=0).max() * 1e-7 + 1e-12)).astype(np.int64)
        uniq, ids = np.unique(row_ids(q), return_inverse=True)
        self.ids = ids.reshape(-1)               # merged id per source vertex
        self.n = n = len(uniq)
        self.pos = np.zeros((n, 3))
        self.pos[self.ids] = v
        fi = self.ids[f]
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])     # length = 2 x area: area-weighted
        nrm = np.zeros((n, 3))
        for c in range(3):
            np.add.at(nrm, fi[:, c], fn)
        self.normals = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        e = np.unique(np.sort(np.vstack([fi[:, [0, 1]], fi[:, [1, 2]], fi[:, [2, 0]]]), axis=1), axis=0)
        self.edges = e[e[:, 0] != e[:, 1]]
        a = coo_matrix((np.ones(2 * len(self.edges)), (np.r_[self.edges[:, 0], self.edges[:, 1]],
                                                        np.r_[self.edges[:, 1], self.edges[:, 0]])), shape=(n, n)).tocsr()
        self.mean = diags(1.0 / np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1)) @ a   # neighbours' mean
        self.elen = np.linalg.norm(self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]], axis=1)
        self.edge = float(np.median(self.elen)) if len(self.elen) else 0.0
        # For `smooth`: edge weights 1 / length^2 and each vertex's share deg / 4 (on an even mesh 4 / deg x the sum
        # of (x_j - x_i) / h^2 is the Laplacian), so a radius is the same mm on any even triangles.
        self.w = 1.0 / np.maximum(self.elen, 1e-6 * max(self.edge, 1e-12)) ** 2
        self.share = np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1) / 4.0
        self._plain: Any = None
        self._fi = fi
        self._cot: Any = None
        self.area = np.zeros(0)

    def cot_lap(self) -> Any:
        """The cotangent Laplacian (use with `smooth(..., area=True)`): the surface's own Laplace-Beltrami operator on
        any triangles, for smoothing that becomes a shape. `plain_lap`'s weights tie a vertex to its short edges'
        neighbours only: on a simplified sculpt mixing 0.02 and 0.3 mm edges in one triangle a smoothed move stayed
        0.05 mm uneven across single edges (creases on Leafeon's forehead). The measuring steps keep `plain_lap`:
        the thickening's strength was tuned on it (a rod's gain fell from 2.8x to 2.3x with this one everywhere)."""
        if self._cot is None:
            w, self.area = self._cotangent(self._fi, self.edges, self.n)
            self._cot = self._laplacian(w)
        return self._cot

    def _cotangent(self, fi: np.ndarray, edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Per edge of `edges` its cotangent weight (half the sum of the cotangents of the angles facing it), clamped
        to COT_FLOOR (an obtuse sliver's long edge keeps a small positive weight: a weighted average, connected); per
        vertex a third of its triangles' area (merged ids `fi`)."""
        p = self.pos[fi]
        area = np.zeros(n)
        weight = np.zeros(len(edges))
        if not len(fi) or not len(edges):
            return weight, np.maximum(area, 1e-12)
        cross = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        for c in range(3):
            np.add.at(area, fi[:, c], cross / 6.0)
        scale_sq = float(np.median(self.elen)) ** 2 if len(self.elen) else 1.0
        ok = cross > 1e-12 * max(scale_sq, 1e-30)                 # degenerate triangles add nothing
        keys = edges[:, 0].astype(np.int64) * n + edges[:, 1]
        order = np.argsort(keys)
        for c in range(3):
            i, j, k = fi[:, c], fi[:, (c + 1) % 3], fi[:, (c + 2) % 3]
            u, v = self.pos[j] - self.pos[i], self.pos[k] - self.pos[i]   # the angle at i faces edge (j, k)
            cot = np.einsum("fi,fi->f", u, v) / np.maximum(cross, 1e-30)
            a_, b_ = np.minimum(j, k), np.maximum(j, k)
            sel = ok & (a_ != b_)
            pos = np.searchsorted(keys, a_[sel].astype(np.int64) * n + b_[sel], sorter=order)
            np.add.at(weight, order[np.minimum(pos, len(order) - 1)], 0.5 * cot[sel])
        weight = np.maximum(weight, COT_FLOOR)
        mean_area = float(np.mean(area[area > 0])) if (area > 0).any() else 1.0
        return weight, np.maximum(area, 1e-9 * mean_area)

    def _laplacian(self, w: np.ndarray) -> Any:
        """D - W for symmetric edge weights `w` (one per edge)."""
        n, e = self.n, self.edges
        ww = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        return diags(np.asarray(ww.sum(axis=1)).ravel()) - ww

    def plain_lap(self) -> Any:
        """The Laplacian in mm (`smooth`), every edge counting."""
        if self._plain is None:
            self._plain = self._laplacian(self.w)
        return self._plain

    def smooth(self, x: np.ndarray, radius: float, lap: Any, steps: int = 1, area: bool = False) -> np.ndarray:
        """`x` (per vertex, one or more columns) diffused over about `radius` of surface: implicit heat steps,
        (M + tau L) y = M x with tau = radius^2 / 8 in all (what (radius / edge)^2 passes of 0.5 x + 0.5 x the
        neighbours' mean do on an even mesh), by conjugate gradients. One step's kernel, exp(-|x| / sqrt(tau)), has a
        cusp: a jump in `x` comes out as a crease. `steps` > 1 splits tau over that many steps (same spread), whose
        kernel is smooth at its centre: use it where the result is a shape. `area`: `lap` is `cot_lap()`, whose mass
        is each vertex's area (else `plain_lap()`'s, deg / 4)."""
        from scipy.sparse.linalg import cg

        if radius <= 0 or self.n == 0:
            return x
        cols = x.reshape(len(x), -1).astype(np.float64)
        out = cols.copy()
        # Only vertices with an edge in `lap` change (a vertex without one keeps its value): a feature's surface
        # (`_stretch`'s sub-graph) is a few % of a 300k figure, and every CG step otherwise ran over all of it.
        live = np.flatnonzero(lap.diagonal() > 0)
        if not len(live):
            return x
        if len(live) < self.n:
            lap = lap.tocsr()[live][:, live]
        share = (self.area if area else self.share)[live]
        steps = max(int(steps), 1)
        a = diags(share) + (radius * radius / 8.0 / steps) * lap
        pre = diags(1.0 / a.diagonal())
        for j in range(cols.shape[1]):
            y = cols[live, j]
            for _ in range(steps):
                b = share * y
                try:
                    y, _ = cg(a, b, x0=y, rtol=1e-6, maxiter=500, M=pre)
                except TypeError:                 # scipy < 1.12 names it `tol`
                    y, _ = cg(a, b, x0=y, tol=1e-6, maxiter=500, M=pre)
            out[live, j] = y
        return out.reshape(x.shape)

    def smooth_normals(self, radius: float) -> np.ndarray:
        """Normals averaged over about `radius` of surface (a noisy sculpt's own normals make lumpy pushes)."""
        rounds = int(np.clip((radius / max(self.edge, 1e-9)) ** 2, 1, NORMAL_ROUNDS))
        nrm = self.normals
        for _ in range(rounds):
            nrm = 0.5 * nrm + 0.5 * (self.mean @ nrm)
        return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    def root_distance(self, thin: np.ndarray) -> np.ndarray:
        """Per vertex of the thin set, the distance over the thin surface to where it meets thick material (its
        root); 0 for a thin piece that meets none (a loose thin shell is all root), inf off the thin set."""
        from scipy.sparse.csgraph import dijkstra

        e = self.edges[thin[self.edges[:, 0]] & thin[self.edges[:, 1]]]
        length = np.linalg.norm(self.pos[e[:, 0]] - self.pos[e[:, 1]], axis=1) + 1e-12
        g = coo_matrix((length, (e[:, 0], e[:, 1])), shape=(self.n, self.n)).tocsr()
        border = self.edges[thin[self.edges[:, 0]] != thin[self.edges[:, 1]]]
        roots = np.unique(border[thin[border]])
        d = np.full(self.n, np.inf)
        if len(roots):
            d = np.asarray(dijkstra(g, directed=False, indices=roots, min_only=True), np.float64)
        orphan = thin & ~np.isfinite(d)
        d[orphan] = 0.0
        d[~thin] = np.inf
        return d


_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
"""A cube's corners (bit 0 = +x, 1 = +y, 2 = +z) in six tetrahedra round its main diagonal (Freudenthal): every
cube splits its faces along the same diagonals, so neighbouring cubes' tetrahedra meet face to face."""


def _across(thick: np.ndarray, exits: np.ndarray, vox: float) -> np.ndarray:
    """Per vertex, how much of its push it gets (module doc, "only across"): 1 where its inward ray leaves the solid
    within ACROSS[0] x its thickness, 0 past ACROSS[1] x. Both thresholds allow EXIT_SLACK voxels more: the exit is
    found to a voxel or two, which on a very thin spot (the 0.2 mm floor of an incised line in an ear) is several
    times its thickness, and the line came out unpushed between pushed sides (a ridge along it)."""
    slack = EXIT_SLACK * vox
    return np.asarray(np.clip((ACROSS[1] * thick + slack - exits) / ((ACROSS[1] - ACROSS[0]) * thick + 1e-12),
                              0.0, 1.0))


def _fill_grooves(g: _Graph, across: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """`across` with its narrow valleys closed (a grey closing: GROOVE_ROUNDS of neighbour max, then as many of
    neighbour min) over edges whose ends face the same way (smoothed normals `nrm` within GROOVE_DEG). The floor of
    an incised line inside a sheet measures as if its ray ran along the sheet and was gated out between pushed sides,
    leaving a ridge along the line: closed. A point's gate only rises away from it (no valley: kept), and the strip
    round a sheet's edge, whose smoothed normals turn away from both faces, is not joined to them."""
    e = g.edges
    e = e[(nrm[e[:, 0]] * nrm[e[:, 1]]).sum(axis=1) >= np.cos(np.radians(GROOVE_DEG))]
    a, b = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    out = np.asarray(across, np.float64).copy()
    for pick in (np.maximum, np.minimum):
        for _ in range(GROOVE_ROUNDS):
            nxt = out.copy()
            pick.at(nxt, a, out[b])
            out = nxt
    return out


def _push_sizes(g: _Graph, thick: np.ndarray, across: np.ndarray, clear: np.ndarray, t_unit: float,
                vox: float) -> np.ndarray:
    """How far (scale-1 units) the surface at each merged vertex of a part is offset outward: half its shortfall
    from the tapered target (module doc, 2-4), gated to across a feature, blended into its surroundings and capped
    under half the open space in front of it. All zeros when nothing is thin."""
    thin = thick < t_unit
    if not thin.any():
        return np.zeros(g.n)
    from scipy.sparse.csgraph import connected_components

    # Taper: the full target where a thin feature joins the figure, TIP x target at its far end.
    d = g.root_distance(thin)
    e = g.edges[thin[g.edges[:, 0]] & thin[g.edges[:, 1]]]
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(g.n, g.n)),
                                   directed=False)
    reach = np.zeros(comp.max() + 1)
    np.maximum.at(reach, comp[thin], d[thin])
    frac = np.where(thin, d / np.maximum(reach[comp], 1e-12), 0.0)
    # A neck (a thin stretch joined to thick material at two or more places, like a small tail tip under a big
    # flame) has no free end: its middle is its weakest point, so it gets the full target throughout. The
    # band within half a target of a join falls into one piece per join.
    band = thin & (d <= t_unit / 2)
    eb = g.edges[band[g.edges[:, 0]] & band[g.edges[:, 1]]]
    _, joins = connected_components(coo_matrix((np.ones(len(eb)), (eb[:, 0], eb[:, 1])), shape=(g.n, g.n)),
                                    directed=False)
    sizes = np.bincount(joins[band], minlength=joins.max() + 1)
    big = band & (sizes[joins] >= ROOT_MIN)
    pairs = np.unique(np.c_[comp[big], joins[big]], axis=0)
    necks = np.bincount(pairs[:, 0], minlength=comp.max() + 1) >= 2 if len(pairs) else np.zeros(comp.max() + 1, bool)
    frac[necks[comp]] = 0.0
    want = t_unit * (1 - (1 - TIP) * np.clip(frac, 0, 1))
    # The across share varies from vertex to vertex near a rim: averaged over about a quarter target of surface
    # first, or neighbours pushed and not pushed would leave spikes.
    raw = across
    for _ in range(int(np.clip((t_unit / 4 / max(g.edge, 1e-9)) ** 2, 2, NORMAL_ROUNDS))):
        across = 0.5 * across + 0.5 * (g.mean @ across)
    across = np.minimum(across, raw + 0.25)          # ... but a fine point keeps (nearly) its own: no push
    gate = np.where(thin, across, 1.0)
    push = np.where(thin, np.clip((want - thick) / 2, 0.0, t_unit / 2) * across, 0.0)
    # Denoise over about half a target of surface (a noisy sculpt measures unevenly; passes grow with the
    # mesh's density), then spread it over ~BLEND targets, fading: a spread keeps its peaks, so a join gets
    # its full push and a fillet round it. A few last passes round off the spread's steps.
    # Averaged among thin vertices only: with the thick body's zeros it fell short of the target right where a
    # feature joins the body (a neck's ends), which is where it breaks.
    w = thin.astype(np.float64)
    for _ in range(int(np.clip((t_unit / 2 / max(g.edge, 1e-9)) ** 2, SMOOTH, NORMAL_ROUNDS))):
        push = np.where(thin, 0.5 * push + 0.5 * (g.mean @ (push * w)) / np.maximum(g.mean @ w, 1e-12), 0.0)
    push *= gate                                       # denoising carried it onto rims and points too
    for _ in range(int(np.clip(round(BLEND * t_unit / max(g.edge, 1e-9)), 1, 60))):
        push = np.maximum(push, SPREAD * (g.mean @ push) * gate)   # not out onto rims and points
    for _ in range(SMOOTH):
        push = 0.5 * push + 0.5 * (g.mean @ push)
    push *= gate                                       # ... and so did the last rounding passes
    # The passes above count in edges (capped): on a dense sculpt they reach a fraction of a mm. One last smoothing
    # over an eighth of a target in mm (`_Graph.smooth`) evens the gates' steps on any mesh.
    push = np.maximum(g.smooth(push, t_unit / 8, g.plain_lap()), 0.0)
    # Not into a narrow gap (a claw against the body): under half the open space in front.
    return np.asarray(np.minimum(push, np.maximum(clear - vox, 0.0) * GAP_SHARE))


def _inside_grid(tri: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Whether each grid point (y, x, z order) is inside the triangles' solid, by ray parity along z."""
    cell, z, sign = _crossings(tri, xs, ys)
    acc = np.zeros((len(ys) * len(xs), len(zs) + 1), np.int16)
    np.add.at(acc, (cell, np.searchsorted(zs, z)), sign.astype(np.int16))
    return np.asarray((np.cumsum(acc, axis=1)[:, :-1] > 0).reshape(len(ys), len(xs), len(zs)))


def _samples(v: np.ndarray, f: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points over triangles `f` about `spacing` apart at most: (points, triangle row of `f` per point, barycentric
    weights per point). Each triangle is split n x n, n from its longest edge."""
    tri = v[f]
    longest = np.max(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2), axis=1)
    n_of = np.clip(np.ceil(longest / spacing), 1, 64).astype(np.intp)
    pts, owner, bary = [], [], []
    for n in np.unique(n_of).tolist():
        sel = np.flatnonzero(n_of == n)
        a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = a + b <= n
        w = np.stack([a[keep], b[keep], n - a[keep] - b[keep]], axis=1) / n          # (k, 3) lattice weights
        pts.append(np.einsum("kj,fjc->fkc", w, tri[sel]).reshape(-1, 3))
        owner.append(np.repeat(sel, len(w)))
        bary.append(np.tile(w, (len(sel), 1)))
    return np.vstack(pts), np.concatenate(owner), np.vstack(bary)


def _marching_tets(phi: np.ndarray, cubes: np.ndarray, origin: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """The surface phi = 0 through the given cubes of a grid of values (x, y, z order, spacing h from `origin`;
    negative inside) as (vertices, faces), wound outward. `cubes`: flat ids of the cubes' lowest corners; the
    surface is closed when every cube it passes through is listed. Each cube is cut into six tetrahedra (`_TETS`),
    each tetrahedron linearly: unlike marching cubes there are no ambiguous cases, so every edge of the result
    joins exactly two triangles (a manifold mesh, which the union needs)."""
    nx, ny, nz = phi.shape
    flat = phi.ravel()
    bits = [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]
    off = np.array([(dx * ny + dy) * nz + dz for dx, dy, dz in bits], np.int64)
    corner_ids = cubes.astype(np.int64)[:, None] + off[None, :]
    neg = np.asarray(flat[corner_ids] < 0)
    cut = neg.any(axis=1) & ~neg.all(axis=1)
    if not cut.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tv = corner_ids[cut][:, _TETS].reshape(-1, 4)                                 # node ids per tetrahedron
    # Each template tetrahedron's handedness (the sign of its corners' determinant).
    cpos = np.array(bits, float)[_TETS]
    hand = np.tile(np.sign(np.linalg.det(cpos[:, 1:] - cpos[:, :1])), int(cut.sum()))
    ins = flat[tv] < 0
    cnt = ins.sum(axis=1)
    use = (cnt > 0) & (cnt < 4)
    tv, ins, cnt, hand = tv[use], ins[use], cnt[use], hand[use]
    # Inside corners first; that reordering's parity and the handedness say which way the triangles face (for a
    # right-handed a, b, c, d with a inside, the cut near a in the order b, c, d faces away from a).
    order = np.argsort(~ins, axis=1, kind="stable")
    tv = np.take_along_axis(tv, order, axis=1)
    inv = sum((order[:, i] > order[:, j]).astype(np.intp) for i in range(4) for j in range(i + 1, 4))
    flip = (hand < 0) ^ (inv % 2 == 1)
    a, b, c, d = tv.T
    one, three, two = cnt == 1, cnt == 3, cnt == 2
    # (corner, corner) pairs per triangle corner: the triangle's corners are the cut points of those grid edges.
    tris = [(np.c_[a, a, a][one], np.c_[b, c, d][one], flip[one]),            # a lone inside corner a
            (np.c_[a, b, c][three], np.c_[d, d, d][three], flip[three]),      # a lone outside corner d
            # two inside (a, b), two outside (c, d): the quad ac, ad, bd, bc as two triangles
            (np.c_[a, a, b][two], np.c_[c, d, d][two], flip[two]),
            (np.c_[a, b, b][two], np.c_[c, d, c][two], flip[two])]
    lo = np.vstack([t[0] for t in tris])
    hi = np.vstack([t[1] for t in tris])
    fl = np.concatenate([t[2] for t in tris])
    # One vertex per cut grid edge, shared by every triangle round it.
    n_nodes = np.int64(nx * ny * nz)
    key = np.minimum(lo, hi) * n_nodes + np.maximum(lo, hi)
    uniq, inverse = np.unique(key.ravel(), return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    faces[fl] = faces[fl][:, ::-1]
    e0, e1 = uniq // n_nodes, uniq % n_nodes

    def xyz(ids: np.ndarray) -> np.ndarray:
        return np.asarray(origin + h * np.stack([ids // (ny * nz), (ids // nz) % ny, ids % nz], axis=1))

    p0, p1 = flat[e0].astype(np.float64), flat[e1].astype(np.float64)
    t = (p0 / (p0 - p1))[:, None]
    return xyz(e0) + t * (xyz(e1) - xyz(e0)), faces


def _shell(v: np.ndarray, f: np.ndarray, push: np.ndarray, unit: float) -> tuple[np.ndarray, np.ndarray] | None:
    """What a part gains (module doc): the solid within `push` (per vertex) of its surface where the push is FADE or
    more, outside and in (inside down to DEPTH grid steps below that: the union hides it), as a closed mesh; None
    when nothing is pushed that far.
    The distance to the surface is measured to dense samples of it, the side by ray parity; `unit` = scale-1 units
    per print mm. The grid is filled in two passes: every COARSE-th point near the pushed surface first, then every
    point only in the coarse cells the surface can pass through."""
    from scipy.spatial import cKDTree

    fade, h = FADE * unit, SHELL_VOXEL * unit
    pmax = float(push.max(initial=0.0))
    if pmax < fade:
        return None
    moved = push >= fade
    span_lo, span_hi = v[moved].min(axis=0), v[moved].max(axis=0)
    while True:
        reach = pmax + 3 * h                    # the farthest a grid point in the shell can be from a pushed sample
        lo = np.maximum(span_lo - reach, v.min(axis=0)) - 3 * h
        hi = np.minimum(span_hi + reach, v.max(axis=0) + pmax) + 3 * h
        if np.prod((hi - lo) / h + 1) <= MAX_SHELL_NODES:
            break
        h *= 1.2
    # Irrational offsets keep grid points off mesh edges (as in `local_radius`); a whole number of coarse cells.
    n = (np.ceil((hi - lo) / (COARSE * h)).astype(np.intp) + 1) * COARSE + 1
    xs, ys, zs = (lo[k] + h * (np.arange(n[k]) + o) for k, o in ((0, 0.1234567), (1, 0.3183099), (2, 0.2718281)))
    org = np.array([xs[0], ys[0], zs[0]])
    tv = v[f]
    near = np.flatnonzero((tv.max(axis=1) >= lo - reach).all(axis=1) & (tv.min(axis=1) <= hi + reach).all(axis=1))
    pts, owner, bary = _samples(v, f[near], h)
    tri_of = tv[near][owner]                                         # each sample's triangle
    d_at = (bary * push[f[near][owner]]).sum(axis=1)                 # the push at each sample, from its corners
    inside = _inside_grid(tv, xs, ys, zs).transpose(1, 0, 2)        # -> x, y, z order
    tree = cKDTree(pts)
    big_d = d_at >= BALL * unit
    dd = d_at[big_d]
    balls = cKDTree(np.c_[pts[big_d], np.sqrt(np.maximum(pmax * pmax - dd * dd, 0.0))]) if big_d.any() else None
    bpts = pts[big_d]
    fn = np.cross(tri_of[:, 1] - tri_of[:, 0], tri_of[:, 2] - tri_of[:, 0])[big_d]
    bnrm = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)     # each sample's outward normal
    z_bed = float(v[:, 2].min())
    slope = max(1.0, 3 * COARSE * h / fade)

    def field(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        """The shell's value at grid points (negative inside the shell)."""
        q = np.stack([xs[i], ys[j], zs[k]], axis=1)
        dist_k, idx_k = tree.query(q, k=NEAREST, distance_upper_bound=reach, workers=-1)
        dist, idx = dist_k[:, 0], idx_k[:, 0]
        far = ~np.isfinite(dist)
        dist = np.where(far, reach, dist)
        # Exactly to the nearest samples' triangles (the samples are a grid step apart: their own distance is up to
        # half a step long beside the surface, and the nearest one's triangle is not always the nearest triangle;
        # an error of a few hundredths there let the original surface show through the shell's thin edge).
        got = np.flatnonzero(~far)
        if len(got):
            ks = np.isfinite(dist_k[got])
            rows, cols = np.nonzero(ks)
            foot = trimesh.triangles.closest_point(tri_of[idx_k[got][rows, cols]], q[got][rows])
            exact = np.full(ks.shape, np.inf)
            exact[rows, cols] = np.linalg.norm(foot - q[got][rows], axis=1)
            dist[got] = np.minimum(dist[got], exact.min(axis=1))
        sdf = np.where(inside[i, j, k], -dist, dist)
        d = np.where(far, 0.0, d_at[np.minimum(idx, len(d_at) - 1)])
        # Under the offset of the nearest surface point, where it moves FADE or more. That second bound is steep
        # (`slope`): left at its own size it is a plateau a few hundredths of a mm above zero all round the pushed
        # area, which the fine pass would take for the surface's neighbourhood and fill.
        off = np.maximum(sdf - d, (fade - d) * slope)
        # ... or in front of any sample pushed BALL or more, within its push: at a concave join (a neck under a flame)
        # the nearest surface is often the unpushed body, which would hide the neck's own push where the neck meets
        # it. In front only: a ball all round a sample just under a point grows the point (a knob), so the radius
        # fades from the push within GROW_DEG of the sample's normal to 0 at FACE_DEG. The candidates are the BALLS
        # nearest by power distance |x - p|^2 - d(p)^2: a nearest-neighbour query on points lifted to 4D,
        # (p, sqrt(D^2 - d(p)^2)) against (x, 0), whose squared distance is that plus D^2.
        ask = np.flatnonzero(off > -2 * h) if balls is not None else np.zeros(0, np.intp)   # (deep inside: no matter)
        if balls is not None and len(ask):
            d4, jb = balls.query(np.c_[q[ask], np.zeros(len(ask))], k=BALLS,
                                 eps=KD_EPS, distance_upper_bound=float(np.hypot(pmax, reach)), workers=-1)
            hit = np.isfinite(d4)
            jb = np.minimum(jb, len(dd) - 1)
            u = q[ask][:, None, :] - bpts[jb]
            r = np.linalg.norm(u, axis=2)
            cos = (u * bnrm[jb]).sum(axis=2) / np.maximum(r, 1e-12)
            lo_c, hi_c = np.cos(np.radians(FACE_DEG)), np.cos(np.radians(GROW_DEG))
            reach_k = dd[jb] * np.clip((cos - lo_c) / (hi_c - lo_c), 0.0, 1.0)
            # |x - p| - (its reach that way): negative in front of a sample within its push.
            ball = np.where(hit, r - reach_k, np.inf).min(axis=1)
            off[ask] = np.minimum(off[ask], ball)
        # In the shell: in the offset, no deeper under the surface than its push and DEPTH grid steps (a thin feature
        # pushed from both sides is solid through; in a thick body the field jumps where the nearest surface turns
        # from pushed to unpushed, which the coarse pass could miss) and above the bed.
        return np.asarray(np.maximum.reduce([off, -(d + DEPTH * h) - sdf, z_bed - zs[k]]))

    # Coarse pass: grid points every COARSE steps within `reach` of a pushed sample (the rest are outside).
    nc = (n - 1) // COARSE + 1
    pc = np.clip(np.round((pts[d_at >= fade] - org) / (COARSE * h)).astype(np.intp), 0, nc - 1)
    mark = np.zeros(tuple(nc), bool)
    mark[pc[:, 0], pc[:, 1], pc[:, 2]] = True
    mark = ndimage.binary_dilation(mark, np.ones((3, 3, 3), bool), iterations=int(np.ceil(reach / (COARSE * h))) + 1)
    mark[[0, -1]] = False
    mark[:, [0, -1]] = False
    mark[:, :, [0, -1]] = False
    coarse = np.full(tuple(nc), 4 * reach, np.float32)        # far outside: never refined
    ci, cj, ck = np.nonzero(mark)
    coarse[ci, cj, ck] = field(ci * COARSE, cj * COARSE, ck * COARSE)
    # Fine pass: the coarse cells the surface may cross (a sign change, or a corner within a cell's diagonal of it).
    cells = [coarse[dx:nc[0] - 1 + dx, dy:nc[1] - 1 + dy, dz:nc[2] - 1 + dz]
             for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    lo_c, hi_c = np.minimum.reduce(cells), np.maximum.reduce(cells)
    near_c = (lo_c < 0) & (hi_c > 0) | (np.minimum(np.abs(lo_c), np.abs(hi_c)) < 1.2 * np.sqrt(3) * COARSE * h)
    ri, rj, rk = np.nonzero(near_c)
    if not len(ri):
        return None
    # Every fine point of those cells (shared corners once); the rest take their coarse cell's side.
    side = (np.sign(coarse) * h).astype(np.float32)
    phi = np.ascontiguousarray(side[np.ix_(*(np.minimum(np.arange(n[k]) // COARSE, nc[k] - 1) for k in range(3)))])
    step = np.arange(COARSE + 1)
    fi = (ri[:, None] * COARSE + step[None, :])
    fj = (rj[:, None] * COARSE + step[None, :])
    fk = (rk[:, None] * COARSE + step[None, :])
    ids = ((fi[:, :, None, None] * n[1] + fj[:, None, :, None]) * n[2] + fk[:, None, None, :]).reshape(-1)
    seen = np.zeros(int(np.prod(n)), bool)                            # shared corners once (np.unique: 10x slower)
    seen[ids] = True
    ids = np.flatnonzero(seen)
    del seen
    gi, gj, gk = ids // (n[1] * n[2]), (ids // n[2]) % n[1], ids % n[2]
    val = field(gi, gj, gk)
    val = np.where(np.abs(val) < 1e-6 * h, 1e-6 * h, val)              # no value exactly on the level
    # Closed: the grid's border is outside (where the box cuts through the part, that cap is inside it: hidden).
    edge = (gi == 0) | (gj == 0) | (gk == 0) | (gi == n[0] - 1) | (gj == n[1] - 1) | (gk == n[2] - 1)
    val[edge] = h
    phi.ravel()[ids] = val
    # The fine cubes of those cells (their lowest corners).
    cube = np.arange(COARSE)
    cid = ((((ri[:, None] * COARSE + cube)[:, :, None, None]) * n[1] + (rj[:, None] * COARSE + cube)[:, None, :, None])
           * n[2] + (rk[:, None] * COARSE + cube)[:, None, None, :]).reshape(-1)
    sv, sf = _marching_tets(phi, cid, org, h)
    return _tidy(sv, sf, unit) if len(sf) else None


def _tidy(v: np.ndarray, f: np.ndarray, unit: float) -> tuple[np.ndarray, np.ndarray]:
    """A marching-tetrahedra mesh with its grid noise smoothed out (Taubin: no shrinking) and its triangles thinned
    where the surface is flat (meshoptimizer, moving it under TIDY_ERROR mm). Either step is skipped when it leaves
    something that is not a closed solid."""
    from .shell import to_manifold
    from .simplify import clean, compact, simplify_indices

    m = trimesh.Trimesh(v, f, process=False)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=TIDY_ROUNDS)
    best = (np.asarray(m.vertices), f)
    try:
        to_manifold(m)
    except RuntimeError:
        best = (v, f)
    extent = float(np.ptp(best[0], axis=0).max())
    if extent <= 0:
        return best
    thin, _ = simplify_indices(best[0], best[1], len(best[1]) // 4, TIDY_ERROR * unit / extent)
    sv, sf = compact(best[0], clean(thin))
    try:
        to_manifold(trimesh.Trimesh(sv, sf, process=False))
    except RuntimeError:
        return best
    return sv, sf


def _triangle(part: Part, face: int) -> tuple[np.ndarray, np.ndarray]:
    """A triangle's centre and outward unit normal."""
    a, b, c = part.vertices[part.faces[face]]
    n = np.cross(b - a, c - a)
    return (a + b + c) / 3, n / (np.linalg.norm(n) + 1e-12)


@dataclass
class _Profile:
    """Where a single feature (`Local`) is thickened: per merged vertex the share of the change it takes (`weight`),
    and with a root section its distance past that section along the feature (`along`, NaN off it) and how far the
    feature reaches past it (`length`)."""
    weight: np.ndarray
    along: np.ndarray | None = None
    length: float = 0.0
    heavy: bool = False      # the feature holds a thick piece past a thin neck (`joints.thick_within`): `_widen_neck`
    cells: np.ndarray | None = None   # with a root section: per voxel of the grid, `along` (NaN off the feature)


def _local_profile(g: _Graph, part: Part, q: Local, thick: Any, t_unit: float, grid: Voxels,
                   scale: float) -> _Profile:
    """Where the feature `q` is thickened (`_Profile`, `Local`); `thick` is a function returning the vertices' local
    thickness (only measured when needed).

    With a root triangle: the volumetric thin feature `joints.weak_points` found it in (`joints.feature_along`: the
    thin part holding `q.face`, from the section through `q.root` outward and one target back toward the body), the
    surface over it, eased in over a quarter target. Without one, or when that triangle is not on a thin feature: the
    connected surface thinner than `t_unit` round `q.face`, and the band its push blends out over (BLEND + 1 targets),
    fading over half a target past that. A thin surface can run on from one feature to others over the bed (a tail
    lying beside the legs), so the first is preferred: those are separate thin features there."""
    from scipy.sparse.csgraph import connected_components, dijkstra

    from .joints import PROBE, cell_under, feature_along, thick_within

    corners = part.faces[q.face]
    if 0 <= q.root < len(part.faces):
        probe = (q.probe or PROBE) / scale
        along = feature_along(grid, probe, _triangle(part, q.face), _triangle(part, q.root), t_unit)
        if along is not None:
            cells = cell_under(grid, g.pos, g.normals, np.isfinite(along))
            d = np.full(g.n, np.nan)
            d[cells >= 0] = along.flat[cells[cells >= 0]]
            # Full from a target behind the section to `face` (the last weak section: joints.GOAL) and LOCAL_REACH[0]
            # targets past it, eased to nothing at LOCAL_REACH[1]: the weak stretch gets stronger; the rest of the
            # feature (a leaf's blade past its stalk, an ear's serrated edge) keeps its shape.
            fp, fn = _triangle(part, q.face)
            fc = cell_under(grid, fp[None], fn[None], np.isfinite(along))[0]
            far = float(along.flat[fc]) if fc >= 0 else 0.0
            lo, hi = far + LOCAL_REACH[0] * t_unit, far + LOCAL_REACH[1] * t_unit
            x = np.where(np.isfinite(d), np.clip((hi - np.nan_to_num(d, nan=hi)) / (hi - lo), 0.0, 1.0), 0.0)
            w = x * x * (3.0 - 2.0 * x)
            # Eased out over about half a target onto the body round the feature's join (its cells stop at the
            # body: cut off there, the change's fillet into the body would end in a step).
            w = np.clip(np.maximum(w, 2.0 * g.smooth(w, t_unit / 2, g.plain_lap())), 0.0, 1.0)
            return _Profile(np.asarray(w), d, float(np.nanmax(along)), thick_within(grid, probe, along), along)
    thin = thick() < t_unit
    seeds = np.unique(g.ids[corners])
    seeds = seeds[thin[seeds]]
    if not len(seeds):
        return _Profile(np.zeros(g.n))
    e = g.edges[thin[g.edges[:, 0]] & thin[g.edges[:, 1]]]
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(g.n, g.n)),
                                   directed=False)
    feature = thin & np.isin(comp, comp[seeds])
    reach = (BLEND + 1.5) * t_unit
    w = coo_matrix((g.elen, (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n)).tocsr()
    d = np.asarray(dijkstra(w, directed=False, indices=np.flatnonzero(feature), min_only=True, limit=reach),
                   np.float64)
    return _Profile(np.asarray(np.clip((reach - d) / (t_unit / 2), 0.0, 1.0)))


def _stretch(g: _Graph, part: Part, q: Local, prof: _Profile, t_q: float, grid: Voxels,
             nrm: np.ndarray) -> np.ndarray:
    """Per merged vertex of `part`, the move (figure units) that thickens the feature `q` to `t_q` by stretching it
    across its thin directions (module doc, "Single features"); zero off it."""
    move = np.zeros((g.n, 3))
    near = g.smooth(prof.weight, t_q, g.plain_lap()) > STRETCH_MIN
    on = prof.weight > STRETCH_MIN
    if not on.any():
        return move
    # Only the surface round the feature takes part in the smoothing below (same merged ids: same vertices).
    sub = _Graph(part.vertices, part.faces[near[g.ids[part.faces]].all(axis=1)])
    lap = sub.plain_lap()
    # Thin directions: the normals' second moments over a target of surface. Both faces of a sheet agree (n n^T is the
    # same for n and -n): one direction for a sheet, two for a rod; along the feature none.
    nn = (g.normals[:, :, None] * g.normals[:, None, :]).reshape(-1, 9)[:, [0, 1, 2, 4, 5, 8]]
    six = sub.smooth(np.where(near[:, None], nn, 0.0), STRETCH_AXES * t_q, lap)[on]
    ten = six[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    lam, vec = np.linalg.eigh(ten)                        # ascending: the last is the thinnest way
    rel = np.clip(lam / np.maximum(lam[:, 2:3], 1e-12), 0.0, 1.0)
    rel[:, 0] = 0.0                                       # never along the feature
    # P n: the smoothed normal's part across the feature, weighted by how thin each direction is. A sheet's faces
    # move straight out (|P n| ~ 1), its rim barely (n lies along the sheet there), a rod all round.
    d = np.einsum("vij,vj,vkj,vk->vi", vec, rel, vec, nrm[on])
    mag = np.linalg.norm(d, axis=1)
    _, chord, clear = local_radius([part.vertices], [part.faces], g.pos[on], d / np.maximum(mag[:, None], 1e-9),
                                   REACH * t_q, grid.h, grid)
    # The feature's thickness round each vertex: the chords across it from its faces (the rim's run along it).
    wt, ch = np.zeros(g.n), np.zeros(g.n)
    wt[on] = mag ** STRETCH_FACE * (chord < REACH * t_q)
    ch[on] = chord
    sm = sub.smooth(np.c_[ch * wt, wt], t_q, lap)[on]
    # Measured only where a sheet's faces or a rod's sides are round it (the ratio of two near-zero sums is noise,
    # even negative); elsewhere nothing moves (k = 1 below).
    known = sm[:, 1] > STRETCH_KNOWN
    t_loc = np.where(known, sm[:, 0] / np.where(known, sm[:, 1], 1.0), np.inf)
    t_loc = np.maximum(t_loc, grid.h)
    # What it takes: the bending moment falls off linearly to the tip and a section's strength goes with its
    # thickness squared, so the thickness needed for the same strength falls as sqrt(the length left).
    need = np.full(len(t_loc), t_q)
    if prof.along is not None and prof.length > 0:
        past = np.nan_to_num(prof.along[on], nan=0.0)
        need *= np.sqrt(np.clip((prof.length - past) / prof.length, 0.0, 1.0))
    # Stretched about the middle: each face moves out by half the shortfall, (k - 1) x half the thickness; never
    # past GAP_SHARE of the open space in front of it.
    short = np.where(np.isfinite(t_loc), np.maximum(need - t_loc, 0.0), 0.0)
    amount = np.clip(prof.weight[on] * short / 2.0, 0.0, GAP_SHARE * clear)
    move[on] = d * amount[:, None]
    # Which way each bit of a sheet curves: smoothing over a target pulls the surface toward its concave side. Each
    # face only sees its own side, so a lens-shaped blade (a leaf tail) reads convex on both and a cupped ear the same
    # way on both: the sheet's bend is the mean of a vertex's and the far face's (the nearest vertex where the chord
    # across leaves the solid): cancelled on the lens, kept on the cup.
    from scipy.spatial import cKDTree

    bow = sub.smooth(g.pos, t_q, lap)[on] - g.pos[on]
    across = chord < REACH * t_q
    far = g.pos[on] - d / np.maximum(mag[:, None], 1e-9) * np.where(across, chord, 0.0)[:, None]
    _, opp = cKDTree(g.pos[on]).query(far)
    bow = 0.5 * (bow + bow[np.where(across, opp, np.arange(len(bow)))])
    move = _even_sheet(move, on, vec[:, :, 2], rel[:, 1], d, bow, t_q, GAP_SHARE * clear, sub, lap)
    # The shape: smoothed with the surface's own Laplacian (`cot_lap`), in steps (no cusp).
    move = sub.smooth(move, STRETCH_SMOOTH * t_q, sub.cot_lap(), STRETCH_STEPS, area=True)
    move[np.linalg.norm(move, axis=1) < STRETCH_MIN * t_q] = 0.0
    return move


def _widen_neck(g: _Graph, part: Part, prof: _Profile, t_q: float, grid: Voxels) -> np.ndarray:
    """Per merged vertex, the move (figure units) that widens a neck holding a thick piece (`_Profile.heavy`: a flame
    on a thin tail) to `t_q` across, about its own centre line; zero off it.

    `_stretch` measures a feature's thickness as chords averaged over a target of surface. A neck about one target
    long between a thin stalk and a thick piece read as thick as the piece (Charmander's 2.3 mm flame neck: +0.02 mm
    at a 3.8 mm target). Here:

    - The centre line: the centroids of the feature's voxels in bands of equal distance along it (`prof.cells`,
      NECK_BAND targets apart; voxels are dense and even, a sculpt's few vertices round a neck are not), carried on
      straight NECK_BACK targets behind where the thin feature starts (the stalk below a neck is often a little
      thicker than the probe, so "body", and needs widening too), each ring there re-centred on its own cut.
    - Each ring's narrowest width: a flat cut across the line through the ring's centre, of the whole solid
      (`grid.inside`) within NECK_REACH targets of the line. A band of equal distance curves like a cap in a narrowing
      neck: its projection read the neck as wide as the flame above it. Where the line runs into the body, the cut is
      wide and nothing grows: the widening fades out by itself.
    - Each vertex takes the ring at its nearest point on the line and moves out from the line by a soft maximum
      (NECK_SOFT targets) of the ring's width and what it needs (the target, falling as sqrt(length left) like
      `_stretch`) over that width: rings already wider (the thick piece's round underside) keep their shape; a hard
      factor carried along the line turned a flame's underside into a straight wall."""
    from scipy.spatial import cKDTree

    move = np.zeros((g.n, 3))
    if prof.along is None or prof.cells is None:
        return move
    idx = np.nonzero(np.isfinite(prof.cells))
    if len(idx[0]) < 8:
        return move
    h = grid.h
    ca = prof.cells[idx].astype(np.float64)
    cp = grid.centres(idx)
    band = max(2.0 * h, NECK_BAND * t_q)
    k = np.floor((ca - ca.min()) / band).astype(np.int64)
    nb = int(k.max()) + 1
    count = np.bincount(k, minlength=nb)
    sums = np.stack([np.bincount(k, weights=cp[:, j], minlength=nb) for j in range(3)], axis=1)
    mean_a = np.bincount(k, weights=ca, minlength=nb) / np.maximum(count, 1)
    rings = np.flatnonzero(count >= NECK_RING_MIN)
    if len(rings) < 3:
        return move
    cent, at = sums[rings] / count[rings, None], mean_a[rings]
    # Behind the feature's start: straight on along its first rings' direction (over a quarter target), every band.
    run = np.cumsum(np.linalg.norm(np.diff(cent, axis=0), axis=1))
    first = min(int(np.searchsorted(run, t_q / 4)) + 1, len(cent) - 1)
    back_dir = cent[0] - cent[first]
    back_dir /= max(float(np.linalg.norm(back_dir)), 1e-12)
    steps = np.arange(int(np.ceil(NECK_BACK * t_q / band)), 0, -1) * band
    cent = np.vstack([cent[0] + steps[:, None] * back_dir, cent])
    at = np.r_[at[0] - steps, at]
    inside = np.argwhere(grid.inside)
    tree = cKDTree(grid.centres(tuple(inside.T)))
    reach = NECK_REACH * t_q
    dirs = np.linspace(0.0, np.pi, NECK_DIRS, endpoint=False)

    def axis_of(c: np.ndarray) -> np.ndarray:
        ax = np.gradient(c, at, axis=0)
        ax = np.stack([np.convolve(np.pad(ax[:, j], 1, mode="edge"), [0.25, 0.5, 0.25], "valid") for j in range(3)], 1)
        return np.asarray(ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12))

    def cut(c: np.ndarray, ax: np.ndarray) -> np.ndarray:
        """The solid's voxels (positions) in a flat cut through `c` across `ax`, within `reach` of `c`."""
        near = tree.query_ball_point(c, reach)
        q = grid.centres(tuple(inside[near].T)) if len(near) else np.zeros((0, 3))
        return np.asarray(q[np.abs((q - c) @ ax) <= 0.75 * h])

    axis = axis_of(cent)
    for j in range(len(steps)):                             # re-centred on their own cut (the stalk may curve)
        q = cut(cent[j], axis[j])
        if len(q) >= NECK_RING_MIN:
            cent[j] = q.mean(axis=0)
    axis = axis_of(cent)
    width = np.full(len(cent), np.inf)
    for j in range(len(cent)):
        q = cut(cent[j], axis[j]) - cent[j]
        if len(q) < NECK_RING_MIN:
            continue
        u = np.cross(axis[j], [1.0, 0.0, 0.0] if abs(axis[j][0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(axis[j], u)
        xy = np.unique(np.round(np.c_[q @ u, q @ w] / h), axis=0) * h
        width[j] = min(float(np.ptp(xy @ np.array([np.cos(t), np.sin(t)]))) for t in dirs) + h
    # Each vertex near the line: its nearest point on the line (between ring centres) and the ring there.
    cand = np.union1d(np.flatnonzero(prof.weight > STRETCH_MIN), np.flatnonzero(cKDTree(cent).query(g.pos)[0] <= reach))
    p = g.pos[cand]
    seg = cent[1:] - cent[:-1]
    ll = np.maximum((seg * seg).sum(axis=1), 1e-12)
    best = np.full(len(p), np.inf)
    s_at = np.zeros(len(p))
    foot = np.zeros_like(p)
    for i in range(len(seg)):
        tt = np.clip(((p - cent[i]) @ seg[i]) / ll[i], 0.0, 1.0)
        f = cent[i] + tt[:, None] * seg[i]
        dist = np.linalg.norm(p - f, axis=1)
        closer = dist < best
        best[closer], foot[closer], s_at[closer] = dist[closer], f[closer], i + tt[closer]
    lo = np.floor(s_at).astype(np.int64).clip(0, len(cent) - 1)
    hi = np.minimum(lo + 1, len(cent) - 1)
    fr = s_at - lo
    a = at[lo] * (1 - fr) + at[hi] * fr
    known = np.isfinite(width[lo]) & np.isfinite(width[hi])
    w_at = np.where(known, np.where(known, width[lo], 0.0) * (1 - fr) + np.where(known, width[hi], 0.0) * fr, np.inf)
    ax_at = axis[lo] * (1 - fr)[:, None] + axis[hi] * fr[:, None]
    ax_at /= np.maximum(np.linalg.norm(ax_at, axis=1, keepdims=True), 1e-12)
    need = np.full(len(a), t_q)
    if prof.length > 0:
        need *= np.sqrt(np.clip((prof.length - a) / prof.length, 0.0, 1.0))
    soft = NECK_SOFT * t_q
    w_fin = np.where(known, w_at, need)
    want = 0.5 * (w_fin + need + np.sqrt((w_fin - need) ** 2 + soft * soft))
    grow = np.where(known, want / np.maximum(w_fin, h) - 1.0, 0.0)
    # Only the surface round the line (a vertex much farther out than its ring's half width is something else: the
    # body beside the stalk); eased out past the last weak section by the profile's weight, and in over the line's
    # first NECK_EASE targets.
    grow[best > 0.75 * w_fin + h] = 0.0
    past = a > 0
    grow[past] *= prof.weight[cand][past]
    x = np.clip((a - at[0]) / (NECK_EASE * t_q), 0.0, 1.0)
    grow *= x * x * (3.0 - 2.0 * x)
    r = p - foot
    r -= np.einsum("vi,vi->v", r, ax_at)[:, None] * ax_at
    move[cand] = r * grow[:, None]
    # TODO: not capped by the open space in front (`_stretch`'s GAP_SHARE x clear): a neck close beside the body could
    # widen into it; the necks seen so far stand free.
    moved = np.linalg.norm(move, axis=1) > 0
    if not moved.any():
        return move
    near = g.smooth(moved.astype(np.float64), t_q, g.plain_lap()) > STRETCH_MIN
    sub = _Graph(part.vertices, part.faces[near[g.ids[part.faces]].all(axis=1)])
    move = sub.smooth(move, NECK_SMOOTH * t_q, sub.cot_lap(), STRETCH_STEPS, area=True)
    move[np.linalg.norm(move, axis=1) < STRETCH_MIN * t_q] = 0.0
    return move


def _even_sheet(move: np.ndarray, on: np.ndarray, thin_axis: np.ndarray, rel2: np.ndarray, d: np.ndarray,
                bow: np.ndarray, t_q: float, cap: np.ndarray, sub: _Graph, lap: Any) -> np.ndarray:
    """`_stretch`'s moves on a sheet (leaf, ear) made to keep what is painted or carved on its faces. Two things
    shrank an ear's painted inner panel (by 11 % of its area on Leafeon) while its outline grew:

    - Curvature. Each face moved out along its normal by half the shortfall, and a concave face moved outward
      shrinks (a convex one grows). Where the sheet curves, the thickening goes to its convex side instead: the
      concave face stays, the convex one moves the whole shortfall (same thickness gained). By STRETCH_CURVE: from a
      bend radius of 1 / STRETCH_CURVE[0] targets (flat: half each, as before) to 1 / STRETCH_CURVE[1] (all to the
      convex side). Never past `cap` (its share of the open space in front, per vertex).
    - Width. A face's vertices moved only through the sheet and only the rim, whose normals lie in it, moved
      sideways: the outline widened round a panel of the same size. The sideways part is kept as it is on the rim
      (in-plane share over STRETCH_RIM) and off the feature (none), and on the faces is the harmonic blend between
      those: linear across a strip, so the whole section widens by one factor and a panel with its outline.

    `on`: the vertices `_stretch` measured; per one of them, `thin_axis` (the thinnest direction), `rel2` (how thin
    the second direction is, relative to the first: near 1 on a rod, whose moves are left as they are), `d` (the
    move's direction), `bow` (the surface smoothed over a target `t_q`, minus where it is, averaged with the far
    face's: toward the sheet's concave side).
    `sub`, `lap`: the feature's surface and its Laplacian (vertices off it have no edges)."""
    from scipy.sparse.linalg import spsolve

    sheet = rel2 < STRETCH_SHEET
    if not sheet.any():
        return move
    m = move[on]
    along_thin = np.einsum("vi,vi->v", m, thin_axis)
    through = along_thin[:, None] * thin_axis
    side = m - through
    # The bend: one implicit heat step over radius r moves a surface of mean curvature H by about r^2 H / 4 toward
    # its centre; a sheet bent like a cylinder of radius R has H = 1 / 2R, so 1 / R ~ 8 |bow . axis| / r^2.
    toward = np.einsum("vi,vi->v", bow, thin_axis)
    bend = 8.0 * np.abs(toward) / (t_q * t_q) * t_q          # targets / bend radius
    x = np.clip((bend - STRETCH_CURVE[0]) / (STRETCH_CURVE[1] - STRETCH_CURVE[0]), 0.0, 1.0)
    shift = np.where(sheet, x * x * (3.0 - 2.0 * x), 0.0)
    convex = -np.sign(toward)[:, None] * thin_axis            # away from the centre of the bend
    # Concave face (moving toward the centre): + shift x its move back out; convex face: + the same again outward.
    # Both faces move the same way, so it is a shift of the whole section, not thickness: smoothed wider than the
    # move, as a mean over the faces only (the rim moves sideways, so its own shift is ~0: counted, it thinned the
    # shift near a panel's edge and the panel kept 5 % less of its area), and the rim takes the faces' shift.
    share = np.linalg.norm(d - np.einsum("vi,vi->v", d, thin_axis)[:, None] * thin_axis, axis=1) / \
        np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    faces = sheet & (share < STRETCH_RIM)
    src = np.zeros((len(move), 4))
    src[on, :3] = np.where(faces, shift * np.abs(along_thin), 0.0)[:, None] * convex
    src[on, 3] = faces
    mean = sub.smooth(src, STRETCH_SHIFT_SMOOTH * t_q, lap)[on]
    seen = mean[:, 3] > STRETCH_KNOWN
    spread = np.where((seen & sheet)[:, None], mean[:, :3] / np.where(seen, mean[:, 3], 1.0)[:, None], 0.0)
    gained = through + spread
    size = np.linalg.norm(gained, axis=1)
    through = gained * np.minimum(1.0, cap / np.maximum(size, 1e-12))[:, None]
    move = move.copy()
    move[on] = through + side
    free = np.zeros(len(move), bool)
    free[np.flatnonzero(on)[faces]] = True
    if not free.any():
        return move
    side_all = np.zeros_like(move)
    side_all[on] = side
    fixed = ~free
    lap = lap.tocsr()
    a = lap[free][:, free]
    rhs = -(lap[free][:, fixed] @ side_all[fixed])
    # A free patch that touches nothing fixed (or a vertex with no edges) would make the system singular: a tiny
    # pull to zero on every free vertex settles those at no sideways move and changes the rest by ~1e-9.
    a = a + diags(np.full(a.shape[0], 1e-9 * max(float(np.abs(a.diagonal()).mean()), 1e-12)))
    blend = spsolve(a.tocsc(), rhs)
    out = move.copy()
    idx = np.flatnonzero(free)
    out[idx] = (through[np.searchsorted(np.flatnonzero(on), idx)] + np.asarray(blend).reshape(len(idx), 3))
    return out


@dataclass
class Thickening:
    """What `thicken` found: per part (in the order given), its vertices' moves (single features stretched: `_stretch`)
    and the solid it gains (`_shell`), or None."""
    shells: list[tuple[np.ndarray, np.ndarray] | None]
    info: StrengthenInfo
    moves: list[np.ndarray | None] = field(default_factory=list)


@dataclass(frozen=True)
class Local:
    """A thin feature thickened to its own minimum: the one holding triangle `face` of part `part`, to `mm`. With a
    `root` triangle (on the section where it would break: joints.py), only that feature from the section outward
    (and one target back toward the body) is stretched (`_local_profile`, `_stretch`). `probe`: the ball (mm at print
    size) `joints.weak_points` found it with (its `probe`); 0 = joints.PROBE."""
    part: int
    face: int
    mm: float
    root: int = -1
    probe: float = 0.0


def thicken(parts: list[Part], scale: float, target: float, local: list[Local] | None = None) -> Thickening:
    """How figure-frame parts (scale 1) are thickened at print scale `scale`: each `local` feature stretched to its
    own minimum (`_stretch`: vertex moves), then the solids that thicken the (moved) parts to `target` mm where thin
    features join the figure, tapering to TIP x target at their far ends (module doc). Only solid parts change."""
    info = StrengthenInfo(target_mm=target, scale=scale)
    shells: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(parts)
    moves: list[np.ndarray | None] = [None] * len(parts)
    solid = [i for i, p in enumerate(parts) if p.subtype == "normal_part" and len(p.faces)]
    local = [q for q in (local or []) if q.mm > 0 and q.part in solid and 0 <= q.face < len(parts[q.part].faces)]
    if (target <= 0 and not local) or scale <= 0 or not solid:
        return Thickening(shells, info, moves)
    vox = VOXEL / scale
    info.voxel_mm = VOXEL
    info.features = len(local)
    move_max = 0.0
    if local:
        moves, move_max = _stretched(parts, solid, local, scale, vox)
        parts = [replace(p, vertices=p.vertices + m) if m is not None else p for p, m in zip(parts, moves)]
    push_max = 0.0
    if target > 0:
        shells, info.thin_share, push_max = _shells(parts, solid, target / scale, vox, scale)
    info.max_push_mm = max(move_max, push_max) * scale
    return Thickening(shells, info, moves)


def _stretched(parts: list[Part], solid: list[int], local: list[Local], scale: float,
               vox: float) -> tuple[list[np.ndarray | None], float]:
    """Per part, its vertices' moves for the `local` features on it (`_stretch`, or `_widen_neck` for a neck under a
    thick piece; where two overlap, blended by length: OVERLAP_POWER), and the longest move (figure units)."""
    moves: list[np.ndarray | None] = [None] * len(parts)
    grid = voxels([parts[i].vertices for i in solid], [parts[i].faces for i in solid], vox)
    longest = 0.0
    for i in sorted({q.part for q in local}):
        g = _Graph(parts[i].vertices, parts[i].faces)
        qs = [q for q in local if q.part == i]
        widest = max(q.mm for q in qs) / scale
        nrm = g.smooth_normals(widest / 2)
        measured: dict[str, np.ndarray] = {}

        def thick() -> np.ndarray:
            """Local thickness per merged vertex (only for a feature named without a root section)."""
            if "t" not in measured:
                r, _, _ = local_radius([parts[i].vertices], [parts[i].faces], g.pos, nrm, REACH * widest, vox, grid)
                measured["t"] = 2 * r
            return measured["t"]

        # Where two features' moves overlap (their eases onto the body meet), each weighs in by its length to the power
        # OVERLAP_POWER: the longer one wins where they differ, smoothly. Picking the longer outright switched
        # between two moves of about the same length but different ways along a line: a crease (Leafeon's forehead,
        # between an ear's ease and the forehead leaf's).
        num, den = np.zeros((g.n, 3)), np.zeros(g.n)
        for q in qs:
            t_q = q.mm / scale
            prof = _local_profile(g, parts[i], q, thick, t_q, grid, scale)
            m = (_widen_neck(g, parts[i], prof, t_q, grid) if prof.heavy else
                 _stretch(g, parts[i], q, prof, t_q, grid, nrm))
            w = (np.linalg.norm(m, axis=1) * scale) ** OVERLAP_POWER       # print mm: only the ratios count
            num += m * w[:, None]
            den += w
        total = np.where(den[:, None] > 0, num / np.where(den > 0, den, 1.0)[:, None], 0.0)
        if total.any():
            moves[i] = total[g.ids]
            longest = max(longest, float(np.linalg.norm(total, axis=1).max()))
    return moves, longest


def _shells(parts: list[Part], solid: list[int], t_unit: float, vox: float,
            scale: float) -> tuple[list[tuple[np.ndarray, np.ndarray] | None], float, float]:
    """Per part, the solid that thickens its thin features to `t_unit` (figure units; module doc 1-4), the share of
    the surface thinner than that, and the largest push; `scale` is the print scale."""
    shells: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(parts)
    graphs = {i: _Graph(parts[i].vertices, parts[i].faces) for i in solid}
    normals = {i: g.smooth_normals(t_unit / 2) for i, g in graphs.items()}
    # One measurement over all solid parts together: an eye set into a head is as thick as the head. Along smoothed
    # normals: a vertex's own normal beside a sharp convex edge runs along the wall and reads a solid box as thin.
    # TODO: a sharp 90-degree join (a flat fin on a box) then reads thick for its first ~half target (smoothed
    # normals tilt into the box there); sculpted figures join smoothly, so this is left as is.
    grid = voxels([parts[i].vertices for i in solid], [parts[i].faces for i in solid], vox)
    radius, exits, clears = local_radius([parts[i].vertices for i in solid], [parts[i].faces for i in solid],
                                         np.vstack([graphs[i].pos for i in solid]),
                                         np.vstack([normals[i] for i in solid]), REACH * t_unit, vox, grid)
    start, thin_n, total, push_max = 0, 0, 0, 0.0
    for i in solid:
        g = graphs[i]
        sl = slice(start, start + g.n)
        start += g.n
        thick = 2 * radius[sl]
        # Pushed only across the feature (module doc): 1 where the solid ends within ACROSS[0] x its thickness
        # along the inward ray, 0 past ACROSS[1] x (a rim, an edge or a point: the ray runs along the feature).
        across = _fill_grooves(g, _across(thick, exits[sl], vox), normals[i])
        thin_n += int((thick < t_unit).sum())
        total += g.n
        push = _push_sizes(g, thick, across, clears[sl], t_unit, vox)
        if push.max(initial=0.0) <= 0:
            continue
        shells[i] = _shell(parts[i].vertices, parts[i].faces, push[g.ids], 1.0 / scale)
        if shells[i] is not None:
            push_max = max(push_max, float(push.max()))
    return shells, thin_n / max(total, 1), push_max


# Unions by geometry (part, shell): a paint edit re-applies the same shells to the same shapes, and only the paint
# is carried over again (`remesh_part`, whose triangle map is cached by geometry too).
_UNIONS: dict[bytes, tuple[np.ndarray, np.ndarray] | None] = {}
UNION_CACHE = 16


def _union(part: Part, shell: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """The part's solid and its shell as one closed mesh (manifold3d), or None when either is not a valid solid."""
    from .shell import from_manifold, to_manifold

    key = digest(part.vertices, part.faces, *shell)
    if key not in _UNIONS:
        try:
            m = from_manifold(to_manifold(part.mesh()) + to_manifold(trimesh.Trimesh(shell[0], shell[1],
                                                                                     process=False)))
            got: tuple[np.ndarray, np.ndarray] | None = (np.asarray(m.vertices), np.asarray(m.faces))
        except RuntimeError:
            got = None
        while len(_UNIONS) >= UNION_CACHE:
            _UNIONS.pop(next(iter(_UNIONS)))
        _UNIONS[key] = got
    return _UNIONS[key]


def with_shells(parts: list[Part], thickening: Thickening) -> list[Part]:
    """`parts` (the ones `thicken` measured, or the same shapes painted: same order and vertices) with their vertices
    moved (single features: same triangles, so paint stays as it is) and united with their shells, paint carried to
    the new triangles from the nearest old one (`remesh_part`). A part that is not a valid solid (the union refuses it)
    is left without its shell."""
    out = list(parts)
    for i, move in enumerate(thickening.moves):
        if move is not None and i < len(out) and len(move) == len(out[i].vertices):
            out[i] = replace(out[i], vertices=out[i].vertices + move)
    for i, shell in enumerate(thickening.shells):
        if shell is None or i >= len(out):
            continue
        got = _union(out[i], shell)
        if got is not None:
            out[i] = remesh_part(out[i], trimesh.Trimesh(got[0], got[1], process=False))
    return out


def strengthen(parts: list[Part], scale: float, target: float,
               local: list[Local] | None = None) -> tuple[list[Part], StrengthenInfo]:
    """Figure-frame parts (scale 1) with thin features thickened to `target` mm at print scale `scale`, and `local`
    features to their own (`thicken` then `with_shells`)."""
    th = thicken(parts, scale, target, local)
    return with_shells(parts, th), th.info
