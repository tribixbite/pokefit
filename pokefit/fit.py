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
HINT_SPAN = 0.15      # max_scale(hint=...): bracket searched around the hint first


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


class _Enough(Exception):
    """Raised inside the search once an offset reaches the margin asked for (`best_offset(enough=...)`)."""

    def __init__(self, x: np.ndarray, f: float) -> None:
        super().__init__()
        self.x, self.f = x, f


def best_offset(points: HullPoints, cavity: Cavity, s: float,
                pockets_unit: np.ndarray | None = None,
                annulus: tuple[float, float] | None = None,
                enough: float | None = None) -> tuple[np.ndarray, float]:
    """XY offset maximising min(fit margin, annulus margin); returns (offset, margin).

    `enough`: stop as soon as an offset with at least this margin is found (a yes/no fit test needs no
    more), returning that offset and margin instead of the best.

    The fit margin alone is concave in the offset, but the annulus term
    (|p + o| >= inner radius) is not, so a coarse grid over plausible offsets
    seeds Nelder-Mead from the three best cells.
    """
    allowed = _allowed(points, cavity, s)
    q = s * points.xy
    use_annulus = pockets_unit is not None and annulus is not None

    def margins(offs: np.ndarray) -> np.ndarray:
        """Combined margin for each offset row of `offs` (k, 2)."""
        d = np.linalg.norm(q[None, :, :] + offs[:, None, :], axis=2)          # (k, n)
        m = np.min(allowed[None, :] - d, axis=1)
        if use_annulus:
            assert pockets_unit is not None and annulus is not None
            r = np.linalg.norm(s * pockets_unit[None, :, :] + offs[:, None, :], axis=2)
            m = np.minimum(m, np.min(np.minimum(r - annulus[0], annulus[1] - r), axis=1))
        return m

    qx, qy = np.ascontiguousarray(q[:, 0]), np.ascontiguousarray(q[:, 1])

    def neg_margin(o: np.ndarray) -> float:
        """One offset (Nelder–Mead calls this ~10⁴ times per fit): hypot on columns, no (k, n, 2) array."""
        if use_annulus:
            f = -float(margins(o[None, :])[0])
        else:
            f = -float(np.min(allowed - np.hypot(qx + o[0], qy + o[1])))
        if enough is not None and -f >= enough:
            raise _Enough(np.array(o, float), f)
        return f

    centre = -(q.min(axis=0) + q.max(axis=0)) / 2
    starts = [centre]                      # concave case: any start reaches the optimum
    if use_annulus:
        assert pockets_unit is not None
        half = max(float(np.max(allowed, initial=0.0)), 1.0)
        g = np.linspace(-half, half, 17)
        grid = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2) + centre
        gm = np.concatenate([margins(grid[i:i + 64]) for i in range(0, len(grid), 64)])
        starts += [grid[i] for i in np.argsort(-gm)[:3]] + [-s * pockets_unit.mean(axis=0)]
    best_x, best_f = None, np.inf
    try:
        for o0 in starts:
            x, f = _nelder_mead_restarts(neg_margin, np.asarray(o0, float))
            if f < best_f:
                best_x, best_f = x, f
    except _Enough as hit:
        return hit.x, -hit.f
    assert best_x is not None
    return best_x, -float(best_f)


def _nelder_mead_restarts(f, x0: np.ndarray, step: float = 1.0, max_restarts: int = 8) -> tuple[np.ndarray, float]:
    """Nelder–Mead restarted from a fresh simplex until it stops improving.

    The margin is a nonsmooth max–min; a single Nelder–Mead run stalls on its
    ridges (measured: −0.002 mm where the true optimum is +0.51 mm). Restarting
    with a new `step`-sized simplex at each result reaches the grid-search
    optimum within 0.005 mm.
    """
    x, fx = x0, f(x0)
    for _ in range(max_restarts):
        simplex = np.array([x, x + [step, 0.0], x + [0.0, step]])
        r = minimize(f, x, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 1500, "initial_simplex": simplex})
        if r.fun >= fx - 1e-7:
            break
        x, fx = np.asarray(r.x), float(r.fun)
    return x, fx


def max_scale(points: HullPoints, cavity: Cavity, pockets_unit: np.ndarray | None = None,
              annulus: tuple[float, float] | None = None, tol: float = 1e-3,
              hint: float | None = None) -> FitResult | None:
    """Largest figure scale that fits (bisection to `tol` relative). None if even 5 % does not fit.

    `hint`: a scale the answer is probably near (a previous fit of a slightly changed figure); when it
    brackets the answer within ±HINT_SPAN the bisection starts there (about half the steps).

    The cavity fit is bisected first. If the pockets (scaled with the figure)
    cannot also sit on the washer's steel ring there, the combined constraint
    is bisected below that scale: shrinking the figure only brings the pockets
    closer, and two pockets any distance apart up to the ring's diameter can
    both sit on it, so the combined test is monotone in practice. If not even
    5 % works, the fit-only result is returned and the caller reports it.
    """
    room = cavity.top_z - cavity.floor_z
    hi = room / points.height
    lo = 0.05 if 0.05 < hi else hi * 0.5

    def ok(s: float, with_annulus: bool = False) -> bool:
        if cavity.floor_z + s * points.height > cavity.top_z + 1e-9:
            return False
        # A yes/no test: the search stops at the first offset that fits (enough=0).
        if with_annulus:
            return best_offset(points, cavity, s, pockets_unit, annulus, enough=0.0)[1] >= 0
        return best_offset(points, cavity, s, enough=0.0)[1] >= 0

    if not ok(lo):
        return None
    if hint is not None and lo < hint < hi:
        a, b = max(lo, hint * (1 - HINT_SPAN)), min(hi, hint * (1 + HINT_SPAN))
        if ok(a) and not ok(b):
            lo, hi = a, b
    if ok(hi):
        lo = hi
    while (hi - lo) / lo > tol:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    if pockets_unit is not None and annulus is not None and not ok(lo, with_annulus=True):
        if ok(0.05, with_annulus=True):
            a, b = 0.05, lo
            while (b - a) / a > tol:
                mid = (a + b) / 2
                if ok(mid, with_annulus=True):
                    a = mid
                else:
                    b = mid
            lo = a
        else:
            pockets_unit = annulus = None
    off, _ = best_offset(points, cavity, lo, pockets_unit, annulus)
    return FitResult(lo, (float(off[0]), float(off[1])), check_fit(points, cavity, lo, (off[0], off[1])),
                     lo * points.height)
