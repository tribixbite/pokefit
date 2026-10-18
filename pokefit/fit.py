"""Largest scale (and XY offset) at which a figure fits the ball cavity.

The figure stands on the cavity floor (washer top). At figure scale `s` and XY
offset `o` (final mm), a figure point q = (x, y, z) — figure frame, lowest
point at z = 0 — lands at (s·x + o_x, s·y + o_y, floor + s·z). It fits when
every point's distance from the ball axis is within the cavity's allowed
radius at its height.

Only the 2D convex hull of each thin z-slab can be limiting (the allowed
radius is a disc at each height), so the vertex cloud is reduced to per-slab
hull points; each keeps its slab's z-range and is tested against the tighter
end, which keeps the reduction conservative.

margin(s, o) = min_i [allowed(z_i) − |s·q_i + o|] is concave in o, so the best
offset is a convex problem; Nelder–Mead with a few starts is plenty in 2D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

from .ball import Cavity

N_SLABS = 200


@dataclass
class HullPoints:
    xy: np.ndarray        # (n, 2) figure frame, scale 1
    z_lo: np.ndarray      # (n,) slab bottom, figure frame (lowest point = 0)
    z_hi: np.ndarray      # (n,)
    height: float         # figure height at scale 1
    xy_centre: np.ndarray  # bbox centre of the footprint, scale 1


def prepare_points(vertices: np.ndarray, n_slabs: int = N_SLABS) -> HullPoints:
    """Reduce a vertex cloud (any frame, z up) to per-slab 2D hull points with z = 0 at the bottom."""
    v = np.asarray(vertices, np.float64)
    z = v[:, 2] - v[:, 2].min()
    height = float(z.max())
    edges = np.linspace(0.0, height, n_slabs + 1)
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n_slabs - 1)
    xs, lo, hi = [], [], []
    order = np.argsort(idx, kind="stable")
    bounds = np.searchsorted(idx[order], np.arange(n_slabs + 1))
    for k in range(n_slabs):
        sel = order[bounds[k]:bounds[k + 1]]
        if len(sel) == 0:
            continue
        pts = v[sel, :2]
        if len(pts) >= 3:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except QhullError:  # collinear / duplicate points: keep them all
                pass
        xs.append(pts)
        lo.append(np.full(len(pts), edges[k]))
        hi.append(np.full(len(pts), edges[k + 1]))
    xy = np.vstack(xs)
    mn, mx = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    return HullPoints(xy, np.concatenate(lo), np.concatenate(hi), height, (mn + mx) / 2)


@dataclass
class FitCheck:
    margin: float                 # mm; >= 0 fits
    point: tuple[float, float, float]  # limiting point, final mm, ball frame (z=0 at centre)
    kind: str                     # what limits there: "ring", "shell", "top", "aperture"


@dataclass
class FitResult:
    max_scale: float
    offset: tuple[float, float]   # best XY offset at max_scale, final mm
    check: FitCheck               # at max_scale
    height_mm: float              # figure height at max_scale


def _allowed(points: HullPoints, cavity: Cavity, s: float) -> np.ndarray:
    lo = cavity.allowed_radius(cavity.floor_z + s * points.z_lo)
    hi = cavity.allowed_radius(cavity.floor_z + s * points.z_hi)
    return np.minimum(lo, hi)


def _limit_kind(cavity: Cavity, z: float) -> str:
    s = cavity.scale
    if z > cavity.top_z:
        return "top"
    sphere = np.sqrt(max(cavity.r_inner ** 2 - z ** 2, 0.0))
    ring = float(cavity.ball.ring_radius(np.array([z / s]))[0]) * s
    if np.isinf(ring):
        return "aperture" if cavity.ball.aperture * s < sphere else "shell"
    return "ring" if ring < sphere else "shell"


def check_fit(points: HullPoints, cavity: Cavity, s: float, offset: tuple[float, float]) -> FitCheck:
    allowed = _allowed(points, cavity, s)
    pos = s * points.xy + np.asarray(offset)
    m = allowed - np.linalg.norm(pos, axis=1)
    i = int(np.argmin(m))
    z = cavity.floor_z + s * float(points.z_hi[i])
    if cavity.floor_z + s * points.height > cavity.top_z:
        return FitCheck(cavity.top_z - (cavity.floor_z + s * points.height),
                        (float(pos[i, 0]), float(pos[i, 1]), cavity.floor_z + s * points.height), "top")
    return FitCheck(float(m[i]), (float(pos[i, 0]), float(pos[i, 1]), z), _limit_kind(cavity, z))


def annulus_margin(pockets_unit: np.ndarray, s: float, offset: np.ndarray, annulus: tuple[float, float]) -> float:
    """min over pockets of the distance inside [lo, hi] (negative = off the steel)."""
    r = np.linalg.norm(s * pockets_unit + offset, axis=1)
    lo, hi = annulus
    return float(np.min(np.minimum(r - lo, hi - r)))


def best_offset(points: HullPoints, cavity: Cavity, s: float,
                pockets_unit: np.ndarray | None = None,
                annulus: tuple[float, float] | None = None) -> tuple[np.ndarray, float]:
    """XY offset maximising min(fit margin, annulus margin); returns (offset, margin)."""
    allowed = _allowed(points, cavity, s)
    q = s * points.xy

    def neg_margin(o: np.ndarray) -> float:
        m = float(np.min(allowed - np.linalg.norm(q + o, axis=1)))
        if pockets_unit is not None and annulus is not None:
            m = min(m, annulus_margin(pockets_unit, s, o, annulus))
        return -m

    starts = [-s * points.xy_centre]
    if pockets_unit is not None:
        starts.append(-s * pockets_unit.mean(axis=0))
    # Centre of the hull points' extremes is a good second start for lopsided figures.
    starts.append(-(q.min(axis=0) + q.max(axis=0)) / 2)
    best = None
    for o0 in starts:
        r = minimize(neg_margin, o0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    assert best is not None
    return np.asarray(best.x), -float(best.fun)


def max_scale(points: HullPoints, cavity: Cavity, pockets_unit: np.ndarray | None = None,
              annulus: tuple[float, float] | None = None, tol: float = 1e-3) -> FitResult | None:
    """Largest figure scale that fits (bisection to `tol` relative). None if even 5 % does not fit.

    The cavity fit is monotone in scale, so it is bisected alone. Keeping the
    pockets on the washer's steel ring is not (pockets move apart as the figure
    grows), so when the fit-optimal scale leaves them off the steel the scale
    steps down 0.5 % at a time until both hold. If none does, the fit-only
    result is returned and the caller reports the annulus failure.
    """
    room = cavity.top_z - cavity.floor_z
    hi = room / points.height
    lo = 0.05 if 0.05 < hi else hi * 0.5

    def ok(s: float, with_annulus: bool = False) -> bool:
        if cavity.floor_z + s * points.height > cavity.top_z + 1e-9:
            return False
        if with_annulus:
            return best_offset(points, cavity, s, pockets_unit, annulus)[1] >= 0
        return best_offset(points, cavity, s)[1] >= 0

    if not ok(lo):
        return None
    if ok(hi):
        lo = hi
    while (hi - lo) / lo > tol:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    if pockets_unit is not None and annulus is not None:
        s = lo
        while s >= 0.05 and not ok(s, with_annulus=True):
            s *= 0.995
        if s >= 0.05:
            lo = s
        else:
            pockets_unit = annulus = None
    off, _ = best_offset(points, cavity, lo, pockets_unit, annulus)
    return FitResult(lo, (float(off[0]), float(off[1])), check_fit(points, cavity, lo, (off[0], off[1])),
                     lo * points.height)
