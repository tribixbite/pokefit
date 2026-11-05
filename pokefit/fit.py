"""Largest scale (and XY offset) at which a figure fits the ball cavity.

The figure stands on the cavity floor (the steel's top). At figure scale `s` and XY
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
from typing import Callable

import numpy as np
import shapely
from scipy.optimize import minimize

from .ball import Cavity

N_SLABS = 200
HINT_SPAN = 0.15      # max_scale(hint=...): bracket searched around the hint first
MIN_SCALE = 0.05      # max_scale: smallest scale tried, as a share of the height-limited scale (or of 1)
ACTIVE = 64           # best_offset: points in the active set it searches on (more join when they limit)
FLOOR_STEP = 2.0      # best_floor: coarse steps of the flat floor's diameter, mm ...
FLOOR_FINE = 0.5      # ... then this fine round the best
FLOOR_MAX = 0.85      # ... up to this share of the ball's inner diameter
FLOOR_TIE = 0.003     # a smaller floor within this share of the best fit is taken (less filled into the shell)


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
    # Per slab, its 2D convex hull's corners (the fit's margins are extremes, met at hull corners; a
    # collinear slab keeps its end points). One vectorised GEOS call: scipy's qhull per slab cost a
    # temporary file each on Windows (~0.4 s per figure).
    present, slab = np.unique(idx, return_inverse=True)
    slab = slab.reshape(-1)
    order = np.argsort(slab, kind="stable")          # shapely wants the points grouped by slab
    order = order[_maybe_on_hull(v[order, :2], slab[order])]
    hulls = shapely.convex_hull(shapely.multipoints(v[order, :2], indices=slab[order]))
    xy, which = shapely.get_coordinates(hulls, return_index=True)
    k = present[which]
    mn, mx = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    return HullPoints(np.asarray(xy, np.float64), edges[k], edges[k + 1], height, (mn + mx) / 2)


_DIRS = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])   # by angle: hull order


def _maybe_on_hull(xy: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Mask of the points (grouped: `group` sorted) that may be corners of their group's 2D convex hull.

    Akl–Toussaint: each group's extreme points in 8 directions span an octagon inside its hull, and a point strictly
    inside it cannot be a hull corner. Building shapely points for every vertex was most of `prepare_points` (0.16 of
    0.28 s for 300k vertices, of which ~6,000 are hull corners); this keeps a few per cent of them. The hulls of what is
    kept are exactly the hulls of all."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, bool)
    starts = np.r_[0, np.nonzero(np.diff(group))[0] + 1]
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    proj = xy @ _DIRS.T                                      # (n, 8)
    keep = np.zeros(n, bool)
    corners = np.empty((len(starts), 8, 2))
    for k in range(8):
        best = np.maximum.reduceat(proj[:, k], starts)
        # The first point of each group reaching its maximum: the group's extreme point in direction k.
        at = np.where(proj[:, k] >= best[gid], np.arange(n), n)
        first = np.minimum.reduceat(at, starts)
        keep[first] = True
        corners[:, k] = xy[first]
    x, y = xy[:, 0], xy[:, 1]
    inside = np.ones(n, bool)
    for k in range(8):
        a, b = corners[:, k], corners[:, (k + 1) % 8]
        ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        # cross(e, p - a) = ex·y − ey·x + c per group; strictly left of every edge is strictly inside. A repeated
        # corner (no edge) says nothing: its line is made to pass every point.
        c = np.where((ex == 0) & (ey == 0), 1.0, ey * a[:, 0] - ex * a[:, 1])
        inside &= ex[gid] * y - ey[gid] * x + c[gid] > 1e-12
    return keep | ~inside


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

    Only the few points nearest the cavity's wall limit the margin, so the search runs on an active set (the
    ACTIVE tightest points at the start) and every answer is checked against all points: a point that turns out
    tighter there joins the set and the search goes on from that offset. The optimum over a subset is at least the
    optimum over all; where all points agree with the subset at its optimum, that offset is the optimum over all.
    """
    allowed = _allowed(points, cavity, s)
    q = s * points.xy
    use_annulus = pockets_unit is not None and annulus is not None
    qx_all, qy_all = np.ascontiguousarray(q[:, 0]), np.ascontiguousarray(q[:, 1])

    def full_margin(o: np.ndarray) -> float:
        """The figure's margin at offset `o` over every point."""
        return float(np.min(allowed - np.hypot(qx_all + o[0], qy_all + o[1])))

    def pocket_margin(offs: np.ndarray) -> np.ndarray:
        assert pockets_unit is not None and annulus is not None
        r = np.linalg.norm(s * pockets_unit[None, :, :] + offs[:, None, :], axis=2)
        return np.asarray(np.min(np.minimum(r - annulus[0], annulus[1] - r), axis=1))

    centre = -(q.min(axis=0) + q.max(axis=0)) / 2
    starts = [centre]                      # concave case: any start reaches the optimum
    if use_annulus:
        assert pockets_unit is not None
        half = max(float(np.max(allowed, initial=0.0)), 1.0)
        g = np.linspace(-half, half, 17)
        grid = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2) + centre
        gm = np.empty(len(grid))
        for i in range(0, len(grid), 16):
            d = np.hypot(qx_all[None, :] + grid[i:i + 16, 0:1], qy_all[None, :] + grid[i:i + 16, 1:2])
            gm[i:i + 16] = np.min(allowed[None, :] - d, axis=1)
        gm = np.minimum(gm, pocket_margin(grid))
        starts += [grid[i] for i in np.argsort(-gm)[:3]] + [-s * pockets_unit.mean(axis=0)]

    def tightest(o: np.ndarray, k: int) -> np.ndarray:
        m = allowed - np.hypot(qx_all + o[0], qy_all + o[1])
        return np.argpartition(m, k - 1)[:k] if k < len(m) else np.arange(len(m))

    active = np.unique(np.concatenate([tightest(np.asarray(o0, float), ACTIVE) for o0 in starts]))
    while True:
        qx, qy, al = qx_all[active], qy_all[active], allowed[active]

        def neg_margin(o: np.ndarray) -> float:
            """One offset (Nelder–Mead calls this ~10⁴ times per fit): hypot on columns, no (k, n, 2) array."""
            f = -float(np.min(al - np.hypot(qx + o[0], qy + o[1])))
            if use_annulus:
                f = max(f, -float(pocket_margin(o[None, :])[0]))
            if enough is not None and -f >= enough:
                raise _Enough(np.array(o, float), f)
            return f

        best_x: np.ndarray | None = None
        best_f = np.inf
        try:
            for o0 in starts:
                x, f = _nelder_mead_restarts(neg_margin, np.asarray(o0, float))
                if f < best_f:
                    best_x, best_f = x, f
        except _Enough as hit:
            best_x, best_f = hit.x, hit.f
        assert best_x is not None
        sub = -float(best_f)
        real = full_margin(best_x)
        if use_annulus:
            real = min(real, float(pocket_margin(best_x[None, :])[0]))
        if real >= sub - 1e-9 or (enough is not None and real >= enough):
            return best_x, real
        # A point outside the set is tighter here: add the tightest ones and go on from this offset.
        more = tightest(best_x, ACTIVE)
        grown = np.union1d(active, more)
        if len(grown) == len(active):          # nothing new (cannot happen with a strictly tighter point): stop
            return best_x, real
        active = grown
        starts = [best_x]


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
    """Largest figure scale that fits (bisection to `tol` relative). None if even the smallest scale tried
    (MIN_SCALE of the height-limited scale, or of 1) does not fit.

    `hint`: a scale the answer is probably near (a previous fit of a slightly changed figure); when it
    brackets the answer within ±HINT_SPAN the bisection starts there (about half the steps).

    The cavity fit is bisected first. If the pockets (scaled with the figure)
    cannot also sit on the steel there, the combined constraint
    is bisected below that scale: shrinking the figure only brings the pockets
    closer, and two pockets any distance apart up to the ring's diameter can
    both sit on it, so the combined test is monotone in practice. If not even
    that smallest scale works, the fit-only result is returned and the caller reports it.
    """
    room = cavity.top_z - cavity.floor_z
    hi = room / points.height
    # The smallest scale tried: 5 % of the size the height allows, or of the file's own size when that is
    # smaller (a file in metres read as mm, like generated GLBs ~700 units tall, fits at ~4 %).
    floor = lo = MIN_SCALE * min(1.0, hi)

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
        if ok(floor, with_annulus=True):
            a, b = floor, lo
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


def best_floor(points: HullPoints, cavity: Callable[[float], Cavity], d_min: float, d_max: float) -> tuple[float, float]:
    """(floor diameter, its largest fit scale): the flat floor filled into the bottom shell that lets the figure fit
    largest. A wider floor sits higher, where the sphere is wider (room for a wide base, less for height); a thin
    steel disc can lie on any of them. `cavity(d)` builds the cavity for floor diameter d; d_min (the steel's own
    seat) .. d_max. Coarse steps, then fine ones round the best; the smallest diameter within FLOOR_TIE of the
    best wins (the fit curve is flat near its top)."""
    found: dict[float, float] = {}
    hint: float | None = None

    def scale(d: float) -> float:
        nonlocal hint
        d = round(float(np.clip(d, d_min, d_max)), 3)
        if d not in found:
            fit = max_scale(points, cavity(d), hint=hint)
            found[d] = fit.max_scale if fit is not None else 0.0
            hint = found[d] or hint
        return found[d]

    for d in [*np.arange(d_min, d_max, FLOOR_STEP), d_max]:
        scale(d)
    top = max(found, key=lambda d: found[d])
    for d in np.arange(top - FLOOR_STEP + FLOOR_FINE, top + FLOOR_STEP, FLOOR_FINE):
        scale(d)
    best = max(found.values())
    pick = min(d for d, v in found.items() if v >= best * (1 - FLOOR_TIE))
    return pick, found[pick]
