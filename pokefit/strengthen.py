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

Limits: the thickness is measured at one print size (the height asked for, else the largest that fits); a smaller
print has thinner features. The shell is a SHELL_VOXEL mesh (0.1 mm facets at print size) and it fades in from
FADE mm, a step under a layer. Relief finer than the push under it is filled, not kept (that is what makes it
thicker). Nothing grows below the bed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
LOCAL_REACH = (0.25, 1.0)  # a single feature (`Local`): in full to a quarter target past its last weak section, to 0 at 1
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
        # of (x_j - x_i) / h^2 is the Laplacian), so a radius is the same mm on any triangles.
        self.w = 1.0 / np.maximum(self.elen, 1e-6 * max(self.edge, 1e-12)) ** 2
        self.share = np.maximum(np.asarray(a.sum(axis=1)).ravel(), 1) / 4.0
        self._plain: Any = None

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

    def smooth(self, x: np.ndarray, radius: float, lap: Any) -> np.ndarray:
        """`x` (per vertex, one or more columns) diffused over about `radius` of surface: one implicit heat step,
        (M + tau L) y = M x with tau = radius^2 / 8 (what (radius / edge)^2 passes of 0.5 x + 0.5 x the neighbours'
        mean do on an even mesh), by conjugate gradients."""
        from scipy.sparse.linalg import cg

        if radius <= 0 or self.n == 0:
            return x
        a = diags(self.share) + (radius * radius / 8.0) * lap
        pre = diags(1.0 / a.diagonal())
        cols = x.reshape(len(x), -1).astype(np.float64)
        out = np.empty_like(cols)
        for j in range(cols.shape[1]):
            b = self.share * cols[:, j]
            try:
                y, _ = cg(a, b, x0=cols[:, j], rtol=1e-6, maxiter=500, M=pre)
            except TypeError:                     # scipy < 1.12 names it `tol`
                y, _ = cg(a, b, x0=cols[:, j], tol=1e-6, maxiter=500, M=pre)
            out[:, j] = y
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


def _local_weight(g: _Graph, part: Part, q: Local, thick: np.ndarray, t_unit: float, grid: Voxels,
                  scale: float) -> np.ndarray:
    """Per merged vertex of `part`, how much of a `t_unit` push the feature `q` takes (`Local`).

    With a root triangle: the volumetric thin feature `joints.weak_points` found it in (`joints.feature_cells`: the
    thin part holding `q.face`, from the section through `q.root` outward and one target back toward the body), the
    surface over it, eased in over a quarter target. Without one, or when that triangle is not on a thin feature: the
    connected surface thinner than `t_unit` round `q.face`, and the band its push blends out over (BLEND + 1 targets),
    fading over half a target past that. A thin surface can run on from one feature to others over the bed (a tail
    lying beside the legs), so the first is preferred: those are separate thin features there."""
    from scipy.sparse.csgraph import connected_components, dijkstra

    from .joints import PROBE, cell_under, feature_along

    corners = part.faces[q.face]
    if 0 <= q.root < len(part.faces):
        along = feature_along(grid, PROBE / scale, _triangle(part, q.face), _triangle(part, q.root), t_unit)
        if along is not None:
            cells = cell_under(grid, g.pos, g.normals, np.isfinite(along))
            d = np.full(g.n, np.nan)
            d[cells >= 0] = along.flat[cells[cells >= 0]]
            # Full from a target behind the section to `face` (the last weak section: joints.GOAL) and LOCAL_REACH[0]
            # targets past it, none past LOCAL_REACH[1]: the weak stretch gets stronger; the rest of the feature (a
            # leaf's blade past its stalk, an ear's serrated edge) keeps its shape.
            fp, fn = _triangle(part, q.face)
            fc = cell_under(grid, fp[None], fn[None], np.isfinite(along))[0]
            far = float(along.flat[fc]) if fc >= 0 else 0.0
            lo, hi = far + LOCAL_REACH[0] * t_unit, far + LOCAL_REACH[1] * t_unit
            w = np.where(np.isfinite(d), np.clip((hi - np.nan_to_num(d, nan=hi)) / (hi - lo), 0.0, 1.0), 0.0)
            # Eased out over about half a target onto the body round the feature's join (its cells stop at the
            # body: cut off there, the push's fillet into the body would end in a step).
            return np.asarray(np.clip(np.maximum(w, 2.0 * g.smooth(w, t_unit / 2, g.plain_lap())), 0.0, 1.0))
    thin = thick < t_unit
    seeds = np.unique(g.ids[corners])
    seeds = seeds[thin[seeds]]
    if not len(seeds):
        return np.zeros(g.n)
    e = g.edges[thin[g.edges[:, 0]] & thin[g.edges[:, 1]]]
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(g.n, g.n)),
                                   directed=False)
    feature = thin & np.isin(comp, comp[seeds])
    reach = (BLEND + 1.5) * t_unit
    w = coo_matrix((g.elen, (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n)).tocsr()
    d = np.asarray(dijkstra(w, directed=False, indices=np.flatnonzero(feature), min_only=True, limit=reach),
                   np.float64)
    return np.asarray(np.clip((reach - d) / (t_unit / 2), 0.0, 1.0))


@dataclass
class Thickening:
    """What `thicken` found: per part (in the order given), the solid it gains, or None."""
    shells: list[tuple[np.ndarray, np.ndarray] | None]
    info: StrengthenInfo


@dataclass(frozen=True)
class Local:
    """A thin feature thickened to its own minimum: the one holding triangle `face` of part `part`, to `mm`. With a
    `root` triangle (on the section where it would break: joints.py), only that feature from the section outward
    (and one target back toward the body) gains (`_local_weight`)."""
    part: int
    face: int
    mm: float
    root: int = -1


def thicken(parts: list[Part], scale: float, target: float, local: list[Local] | None = None) -> Thickening:
    """The solids that thicken figure-frame parts (scale 1) to `target` mm at print scale `scale` where thin
    features join the figure, tapering to TIP x target at their far ends (module doc), and each `local` feature
    to its own minimum (`_local_weight`). Only solid parts gain."""
    info = StrengthenInfo(target_mm=target, scale=scale)
    shells: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(parts)
    solid = [i for i, p in enumerate(parts) if p.subtype == "normal_part" and len(p.faces)]
    local = [q for q in (local or []) if q.mm > 0 and q.part in solid and 0 <= q.face < len(parts[q.part].faces)]
    widest = max([target, *(q.mm for q in local)])
    if widest <= 0 or scale <= 0 or not solid:
        return Thickening(shells, info)
    # Measured to the widest target asked for (the same measurement as before when there are no local ones).
    t_unit, vox = widest / scale, VOXEL / scale
    info.voxel_mm = VOXEL
    info.features = len(local)
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
        thin_n += int((thick < target / scale).sum())
        total += g.n
        push = (_push_sizes(g, thick, across, clears[sl], target / scale, vox) if target > 0 else np.zeros(g.n))
        sizes: dict[float, np.ndarray] = {}          # `_push_sizes` per local target (several features share one)
        for q in local:
            if q.part != i:
                continue
            t_q = q.mm / scale
            if t_q not in sizes:
                sizes[t_q] = _push_sizes(g, thick, across, clears[sl], t_q, vox)
            push = np.maximum(push, sizes[t_q] * _local_weight(g, parts[i], q, thick, t_q, grid, scale))
        if push.max(initial=0.0) <= 0:
            continue
        shells[i] = _shell(parts[i].vertices, parts[i].faces, push[g.ids], 1.0 / scale)
        if shells[i] is not None:
            push_max = max(push_max, float(push.max()))
    info.thin_share = thin_n / max(total, 1)
    info.max_push_mm = push_max * scale
    return Thickening(shells, info)


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
    """`parts` (the ones `thicken` measured, or the same shapes painted: same order) united with their shells,
    paint carried to the new triangles from the nearest old one (`remesh_part`). A part that is not a valid solid
    (the union refuses it) is left as it is."""
    out = list(parts)
    for i, shell in enumerate(thickening.shells):
        if shell is None or i >= len(parts):
            continue
        got = _union(parts[i], shell)
        if got is not None:
            out[i] = remesh_part(parts[i], trimesh.Trimesh(got[0], got[1], process=False))
    return out


def strengthen(parts: list[Part], scale: float, target: float,
               local: list[Local] | None = None) -> tuple[list[Part], StrengthenInfo]:
    """Figure-frame parts (scale 1) with thin features thickened to `target` mm at print scale `scale`, and `local`
    features to their own (`thicken` then `with_shells`)."""
    th = thicken(parts, scale, target, local)
    return with_shells(parts, th), th.info
