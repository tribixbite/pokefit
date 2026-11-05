"""Strengthen thin parts: thicken whatever is thinner than a minimum at the print size (a flame's thin
tips, toes and claws, an ear, and the narrow join where a small tail tip meets a big flame), blending
the change into the surface around it so a join gets a fillet-like taper rather than a step.

Why: a figure scaled down to fit a ~27 mm ball has features a few tenths of a millimetre thick at
print size. Such a join holds only a few extrusion lines and snaps off (a Charmander's tail flame,
an Eevee's ears, a Pikachu's tail).

How (vertices only move, faces and their order are kept, so paint and texture colour carry over,
like conform.py):

1. **Local thickness.** The solid parts are voxelised at VOXEL mm (print size) by ray parity along z
   (`magnets._crossings`: every triangle crossing a column, as an entry or an exit), and a Euclidean
   distance transform gives each inside voxel its distance to the surface. A vertex's local radius
   is the largest distance found along its inward normal within the target thickness: about half the
   thickness of a sheet, the radius of a rod, and "thick enough" for a body.
2. **Taper.** Each thin feature (connected vertices thinner than the target) gets the full target
   where it meets thick material (its root: the join that breaks) and TIP × target at its far end,
   in proportion to the distance over its surface from the root. A fine tip stays fine; its base
   gets stronger. A neck (thin between two thick parts) has no far end and gets the full target.
3. **Push.** A thin vertex moves out along its (smoothed) normal by half its shortfall from that
   (both sides of a sheet move, so the sheet ends at the target).
4. **Blend.** The push spreads over BLEND × target mm of surface around it, fading (each pass a vertex
   keeps the larger of its own push and SPREAD × its neighbours' mean), then a few smoothing passes:
   the thickened part meets its body with a taper, which is where a join is weakest.

Only across: a vertex is pushed only where its inward ray leaves the solid again within about its own thickness
(ACROSS), so the push widens the feature across its thin direction. On a rim, an edge or a point (an ear's tip,
the point of a lightning-bolt tail, the edge of a flat fin) "outward" runs along the feature: the ray runs down
it and leaves far away. Pushing those grew the point outward and rounded it into a knob (worst at 2.5-4 mm);
they now keep their place, and the sheet behind them still thickens.

Even where the surface curves: a push 2-3 edges long along normals that differ by a few degrees between
neighbours (fur spikes, the hollow of an ear, every curve of a dense sculpt) moved neighbours apart sideways, so
the triangles tilted into wrinkles, and turned 500-1,000 triangles over on a Jolteon at any simplify level: what
looked like the model's flaws magnified (most on a simplified mesh, whose long uneven triangles the smoothing
passes, counted in edges, reach unevenly). So the moves end up smooth by construction:
- **Sizes**: after the passes, one implicit smoothing over an eighth of a target in mm evens the gates' steps;
- **Directions** (`_Graph.smooth`, `crease_lap`): the move vectors are diffused over a quarter target of surface
  (one implicit heat step with edge weights in mm, so the radius is the same on a dense sculpt and on a simplified
  mesh of long, uneven triangles; edges count by how much their normals agree, so the two sides of a crease or of a
  thin sheet's rim do not mix), and each keeps its own length;
- **Gaps** (`clear`): a vertex moves under half the open space in front of it (a claw against the body);
- **No folds** (`_unfold`): a well-shaped triangle the moves turn over has its corners' moves halved until none is.

Measured on a 300k / 60k / 20k-triangle Jolteon at 2 mm: triangles turned over 1015 / 472 / 972 -> 90 / 35 / 82.

Limits: the thickness is measured at one print size (the height asked for, else the largest that
fits); a smaller print has thinner features. A groove narrower than a voxel (an incised line, like the one round
an ear's inner panel) is not seen as a gap: its sides still close it and cross (the slicer unions them; the view
shows a ragged seam there). Nothing moves below the bed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, diags

from .magnets import _crossings
from .model3mf import Part, row_ids

VOXEL = 0.15         # mm at print size
SAMPLES = 12         # steps along the inward normal when measuring a vertex's local radius
ACROSS = (1.5, 3.0)  # full push where the inward ray leaves the solid within 1.5 x the thickness, none past 3 x
REACH = 1.5          # ...up to this many targets deep (a 90-degree corner's distance grows only as depth / sqrt 3)
BLEND = 1.0          # the push blends out over this many target thicknesses of surface
SPREAD = 0.9         # per pass, a vertex takes this share of its neighbours' mean push (if larger than its own)
SMOOTH = 6           # fewest smoothing passes over the push
TIP = 0.4            # a thin feature's far end is thickened to this share of the target (a taper)
ROOT_MIN = 6         # vertices a join needs to count (fewer: measuring noise)
NORMAL_ROUNDS = 80   # most smoothing passes over the push direction and the push
MAX_VOXELS = 30_000_000  # coarser voxels beyond this (a very large figure or a fine target)
GAP_SHARE = 0.45     # a vertex moves at most this share of the open space in front of it (`clear`)
FOLD_DEG = 90.0      # a triangle whose normal turns more than this has turned over (`_unfold`)
UNFOLD_ROUNDS = 12   # halvings at most
SLIVER = 0.05        # ... of triangles at least this well shaped


@dataclass
class StrengthenInfo:
    target_mm: float = 0.0          # minimum thickness asked for
    scale: float = 0.0              # print scale it was measured at
    thin_share: float = 0.0         # share of the surface (vertices) thinner than the target
    max_push_mm: float = 0.0        # largest outward move
    voxel_mm: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def local_radius(meshes_v: list[np.ndarray], meshes_f: list[np.ndarray], points: np.ndarray,
                 inward: np.ndarray, reach: float, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point (on the surface): the largest distance to the surface found along `inward` up to `reach` (about
    half the local thickness, capped near `reach` for thick material), how far along `inward` the solid ends
    (`reach` when it does not end within it), and how far OUT (against `inward`) the next solid is (`reach` when
    none is within it: open space). All in the same units."""
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
        self._crease: Any = None
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

    def crease_lap(self) -> Any:
        """The Laplacian with each edge weighted by how much its ends' normals agree (0 from a right angle on):
        smoothing with it does not mix the two sides of a crease or of a thin sheet's rim."""
        if self._crease is None:
            e = self.edges
            agree = np.clip((self.normals[e[:, 0]] * self.normals[e[:, 1]]).sum(axis=1), 0.0, 1.0) ** 2
            self._crease = self._laplacian(self.w * agree)
        return self._crease

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


def _unfold(pos: np.ndarray, faces: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """`disp` (per vertex moves) with the corners of every triangle it would turn over (normal turning more than
    FOLD_DEG) halved, repeated until none does (at most UNFOLD_ROUNDS times; then those corners do not move).
    Merged-vertex frame."""
    def normals(v: np.ndarray) -> np.ndarray:
        n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
        return np.asarray(n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30))

    before = normals(pos)
    # Slivers (a long edge over a tiny height) turn over at the least move and show nothing: only well-shaped ones
    # count (twice the area over the longest edge squared: 0.87 for an equilateral triangle).
    e0, e1, e2 = (pos[faces[:, k]] - pos[faces[:, (k + 1) % 3]] for k in range(3))
    area2 = np.linalg.norm(np.cross(e0, e1), axis=1)
    longest = np.maximum.reduce([np.einsum("ij,ij->i", e, e) for e in (e0, e1, e2)])
    ok = area2 > SLIVER * longest
    limit = np.cos(np.radians(FOLD_DEG))
    out = disp.copy()
    for k in range(UNFOLD_ROUNDS + 1):
        bad = ok & ((before * normals(pos + out)).sum(axis=1) < limit)
        if not bad.any():
            break
        corners = np.unique(faces[bad])
        out[corners] = 0.0 if k == UNFOLD_ROUNDS else out[corners] / 2
    return out


def strengthen(parts: list[Part], scale: float, target: float) -> tuple[list[Part], StrengthenInfo]:
    """Figure-frame parts (scale 1) with thin features thickened to `target` mm at print scale `scale` where they
    join the figure, tapering to TIP x target at their far ends (module doc). Only solid parts move."""
    info = StrengthenInfo(target_mm=target, scale=scale)
    solid = [i for i, p in enumerate(parts) if p.subtype == "normal_part" and len(p.faces)]
    if target <= 0 or scale <= 0 or not solid:
        return parts, info
    t_unit, vox = target / scale, VOXEL / scale
    info.voxel_mm = VOXEL
    graphs = {i: _Graph(parts[i].vertices, parts[i].faces) for i in solid}
    normals = {i: g.smooth_normals(t_unit / 2) for i, g in graphs.items()}
    # One measurement over all solid parts together: an eye set into a head is as thick as the head. Along smoothed
    # normals: a vertex's own normal beside a sharp convex edge runs along the wall and reads a solid box as thin.
    # TODO: a sharp 90-degree join (a flat fin on a box) then reads thick for its first ~half target (smoothed
    # normals tilt into the box there); sculpted figures join smoothly, so this is left as is.
    radius, exits, clears = local_radius([parts[i].vertices for i in solid], [parts[i].faces for i in solid],
                                 np.vstack([graphs[i].pos for i in solid]), np.vstack([normals[i] for i in solid]),
                                 REACH * t_unit, vox)
    out = list(parts)
    start, thin_n, total, push_max = 0, 0, 0, 0.0
    for i in solid:
        g, nrm = graphs[i], normals[i]
        thick = 2 * radius[start:start + g.n]
        # Pushed only across the feature (module doc): 1 where the solid ends within ACROSS[0] x its thickness
        # along the inward ray, 0 past ACROSS[1] x (a rim, an edge or a point: the ray runs along the feature).
        clear = clears[start:start + g.n]
        across = np.clip((ACROSS[1] * thick - exits[start:start + g.n]) / ((ACROSS[1] - ACROSS[0]) * thick + 1e-12),
                         0.0, 1.0)
        start += g.n
        thin = thick < t_unit
        thin_n += int(thin.sum())
        total += g.n
        if not thin.any():
            continue
        # Taper: the full target where a thin feature joins the figure, TIP x target at its far end.
        d = g.root_distance(thin)
        from scipy.sparse.csgraph import connected_components

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
        for _ in range(int(np.clip((t_unit / 2 / max(g.edge, 1e-9)) ** 2, SMOOTH, NORMAL_ROUNDS))):
            push = 0.5 * push + 0.5 * (g.mean @ push)
        push *= gate                                       # denoising carried it onto rims and points too
        for _ in range(int(np.clip(round(BLEND * t_unit / max(g.edge, 1e-9)), 1, 60))):
            push = np.maximum(push, SPREAD * (g.mean @ push) * gate)   # not out onto rims and points
        for _ in range(SMOOTH):
            push = 0.5 * push + 0.5 * (g.mean @ push)
        push *= gate                                       # ... and so did the last rounding passes
        # The passes above count in edges (capped): on a dense sculpt they reach a fraction of a mm and leave the
        # gates' per-vertex steps, which the push raised as wrinkles. One last smoothing over an eighth of a target
        # in mm (`_Graph.smooth`) evens them on any mesh.
        push = np.maximum(g.smooth(push, t_unit / 8, g.plain_lap()), 0.0)
        # Even where the surface curves (module doc): not into a narrow gap, smoothed directions
        # (each move keeps its length: round a rod thinner than the radius, opposite sides' moves partly cancel;
        # where they cancel outright, the vertex's own normal), then no triangle turned over.
        push = np.minimum(push, np.maximum(clear - vox, 0.0) * GAP_SHARE)
        way = g.smooth(nrm * push[:, None], t_unit / 4, g.crease_lap())
        size = np.linalg.norm(way, axis=1, keepdims=True)
        way = np.where(size > 1e-3 * push[:, None], way / np.maximum(size, 1e-30), nrm)
        disp = _unfold(g.pos, g.ids[parts[i].faces], way * push[:, None])
        v = parts[i].vertices + disp[g.ids]
        v[:, 2] = np.maximum(v[:, 2], parts[i].vertices[:, 2].min())     # nothing below the bed
        push_max = max(push_max, float(np.linalg.norm(disp, axis=1).max()))
        q = parts[i]
        out[i] = Part(q.name, q.extruder, v, q.faces, q.tri_extra, q.subtype)
    info.thin_share = thin_n / max(total, 1)
    info.max_push_mm = push_max * scale
    return out, info
