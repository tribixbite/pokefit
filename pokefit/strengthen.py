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

Limits: the thickness is measured at one print size (the height asked for, else the largest that
fits); a smaller print has thinner features. Pushing out the two sides of a narrow gap (a claw
against the body) can make them touch; the slicer unions them. Nothing moves below the bed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, diags

from .magnets import _crossings
from .model3mf import Part, row_ids

VOXEL = 0.15         # mm at print size
SAMPLES = 12         # steps along the inward normal when measuring a vertex's local radius
REACH = 1.5          # ...up to this many targets deep (a 90-degree corner's distance grows only as depth / sqrt 3)
BLEND = 1.0          # the push blends out over this many target thicknesses of surface
SPREAD = 0.9         # per pass, a vertex takes this share of its neighbours' mean push (if larger than its own)
SMOOTH = 6           # fewest smoothing passes over the push
TIP = 0.4            # a thin feature's far end is thickened to this share of the target (a taper)
ROOT_MIN = 6         # vertices a join needs to count (fewer: measuring noise)
NORMAL_ROUNDS = 80   # most smoothing passes over the push direction and the push
MAX_VOXELS = 30_000_000  # coarser voxels beyond this (a very large figure or a fine target)


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
                 inward: np.ndarray, reach: float, voxel: float) -> np.ndarray:
    """Per point (on the surface), the largest distance to the surface found along `inward` up to `reach`
    (all in the same units): about half the local thickness, capped near `reach` for thick material."""
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
    for t in np.linspace(h, reach, SAMPLES):
        p = points - inward * t
        j = np.clip(np.round((p[:, 0] - xs[0]) / h).astype(np.intp), 0, len(xs) - 1)
        i = np.clip(np.round((p[:, 1] - ys[0]) / h).astype(np.intp), 0, len(ys) - 1)
        k = np.clip(np.round((p[:, 2] - zs[0]) / h).astype(np.intp), 0, len(zs) - 1)
        r = np.maximum(r, dist[i, j, k])
    return r


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
        self.edge = float(np.median(np.linalg.norm(self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]], axis=1)))

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
    radius = local_radius([parts[i].vertices for i in solid], [parts[i].faces for i in solid],
                          np.vstack([graphs[i].pos for i in solid]), np.vstack([normals[i] for i in solid]),
                          REACH * t_unit, vox)
    out = list(parts)
    start, thin_n, total, push_max = 0, 0, 0, 0.0
    for i in solid:
        g, nrm = graphs[i], normals[i]
        thick = 2 * radius[start:start + g.n]
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
        push = np.where(thin, np.clip((want - thick) / 2, 0.0, t_unit / 2), 0.0)
        # Denoise over about half a target of surface (a noisy sculpt measures unevenly; passes grow with the
        # mesh's density), then spread it over ~BLEND targets, fading: a spread keeps its peaks, so a join gets
        # its full push and a fillet round it. A few last passes round off the spread's steps.
        for _ in range(int(np.clip((t_unit / 2 / max(g.edge, 1e-9)) ** 2, SMOOTH, NORMAL_ROUNDS))):
            push = 0.5 * push + 0.5 * (g.mean @ push)
        for _ in range(int(np.clip(round(BLEND * t_unit / max(g.edge, 1e-9)), 1, 60))):
            push = np.maximum(push, SPREAD * (g.mean @ push))
        for _ in range(SMOOTH):
            push = 0.5 * push + 0.5 * (g.mean @ push)
        v = parts[i].vertices + (nrm * push[:, None])[g.ids]
        v[:, 2] = np.maximum(v[:, 2], parts[i].vertices[:, 2].min())     # nothing below the bed
        push_max = max(push_max, float(push.max()))
        q = parts[i]
        out[i] = Part(q.name, q.extruder, v, q.faces, q.tri_extra, q.subtype)
    info.thin_share = thin_n / max(total, 1)
    info.max_push_mm = push_max * scale
    return out, info
