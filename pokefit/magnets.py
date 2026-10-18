"""Magnet pockets in the figure's underside.

Pockets are open-bottomed blind holes for disc magnets (default Ø4.18 × 2.20
mm: the pockets of the Zelda pot set, sized for 4 × 2 mm magnets). All
coordinates here are final print mm in the figure frame: XY as the figure
will be printed, z = 0 on the bed (= the plane the figure stands on).

The underside does not have to be flat. It is described by a height field
(`HeightField`): for each XY grid point, where material starts going up
(z1) and where that first solid span ends (z2). A pocket at (x, y) is cut
from below up to its ceiling at z_top = z + depth, where z is its seat (the
pocket's open end) and is chosen as low as possible such that

  * z >= 0: the magnet never sticks out below the bed plane, so the figure
    still stands on its original underside;
  * at least `grip` mm of the pocket is surrounded by `wall` mm of material
    (z1 <= z_top - grip all over the walled disc), so a magnet pressed in
    is held; on a slope it may stand proud of the surface below that;
  * the walls reach the ceiling (z2 >= z_top over the walled disc) and
    `ceiling` mm of material covers it (z2 >= z_top + ceiling over the disc);
  * (automatic placement) z <= `max_seat`: higher up, the magnet is too far
    from the steel to hold much.

Up to three pockets are placed automatically: spread as far apart as possible
(stability), deep inside the valid area (thicker walls), and low (a magnet
any distance above the steel holds much less). Nothing is cut where a rule
fails; the failure is reported with numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh
from scipy import ndimage
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .geom import mf
from .model3mf import Part, remesh_part
from .report import Check, Status
from .shell import from_manifold, to_manifold

Z_PROBE = 0.3        # height of the "feet" section above the bed, mm (underside outline)
SEGMENTS = 48        # polygon sides of the pocket cylinder (as in the Zelda pot)
MIN_CUT_WALL = 0.2   # mm; thinner than this the pocket would break through the side, so it is not cut
GRID_CELLS = 400     # height-field cells across the figure's larger footprint side
LOW_SEAT = 0.5       # mm above the washer: higher than this, the hold is noticeably weaker
# Placement score, in mm: spread + DEPTH_WEIGHT × min(extra wall, DEPTH_CAP) − SEAT_PENALTY × seat.
# The first mm of wall beyond the minimum is worth 2 mm of spread (pockets centre
# in small feet); past that spread wins. 1 mm of seat height costs 5 mm of spread.
DEPTH_WEIGHT = 2.0
DEPTH_CAP = 1.0
SEAT_PENALTY = 5.0
MAX_CANDIDATES = {1: 4000, 2: 400, 3: 120}   # thinned candidate centres per pocket count


@dataclass(frozen=True)
class PocketSpec:
    diameter: float = 4.18
    depth: float = 2.20
    wall: float = 0.8
    ceiling: float = 0.6
    grip: float = 1.2          # min. height of the pocket surrounded by wall
    max_seat: float = 2.0      # auto placement: highest pocket opening above the washer (the hold fades fast)


@dataclass
class Pocket:
    x: float
    y: float
    z: float = 0.0             # seat: height of the pocket's open end above the bed

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ------------------------------------------------------------- cross-sections


def section(meshes: list[trimesh.Trimesh], z: float) -> MultiPolygon | Polygon:
    """Union of the parts' cross-sections at height z (XY plane)."""
    polys: list[Polygon] = []
    for m in meshes:
        if not (m.bounds[0][2] < z < m.bounds[1][2]):
            continue
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        planar, _ = sec.to_2D(to_2D=np.linalg.inv(trimesh.transformations.translation_matrix([0, 0, z])),
                              check=False)
        polys.extend(p for p in planar.polygons_full if p.is_valid and p.area > 1e-6)
    return unary_union(polys) if polys else Polygon()


def _parts(geom: Polygon | MultiPolygon) -> list[Polygon]:
    if geom.is_empty:
        return []
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def find_feet(meshes: list[trimesh.Trimesh], z_probe: float = Z_PROBE) -> list[Polygon]:
    """Islands of the cross-section just above the bed, largest first."""
    return sorted(_parts(section(meshes, z_probe)), key=lambda p: -p.area)


# --------------------------------------------------------------- height field


@dataclass
class HeightField:
    """Underside of a solid on a regular XY grid (row i ↔ y0 + i·step, column j ↔ x0 + j·step).

    z1: height where material starts going up (+inf where there is none);
    z2: top of that first solid span (−inf where there is none).
    """
    x0: float
    y0: float
    step: float
    z1: np.ndarray
    z2: np.ndarray

    def scaled(self, s: float) -> "HeightField":
        """The same underside after scaling the solid by s about the origin."""
        return HeightField(self.x0 * s, self.y0 * s, self.step * s, self.z1 * s, self.z2 * s)

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.z1.shape
        return self.x0 + np.arange(nx) * self.step, self.y0 + np.arange(ny) * self.step

    def window(self, x: float, y: float, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(distance to (x, y), z1, z2) of the cells within `radius` (cells off the grid count as empty)."""
        n = int(np.ceil(radius / self.step)) + 1
        j0, i0 = int(round((x - self.x0) / self.step)), int(round((y - self.y0) / self.step))
        ii, jj = np.mgrid[i0 - n:i0 + n + 1, j0 - n:j0 + n + 1]
        d = np.hypot(self.x0 + jj * self.step - x, self.y0 + ii * self.step - y)
        ny, nx = self.z1.shape
        on = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        z1 = np.full(ii.shape, np.inf)
        z2 = np.full(ii.shape, -np.inf)
        z1[on], z2[on] = self.z1[ii[on], jj[on]], self.z2[ii[on], jj[on]]
        keep = d <= radius
        return d[keep], z1[keep], z2[keep]


def height_field(meshes: list[trimesh.Trimesh], cells: int = GRID_CELLS, margin: float = 0.0) -> HeightField:
    """Rasterise the parts onto an XY grid and record the first solid span above each cell.

    Every triangle is tested against the cell centres inside its XY bounding
    box (pure numpy, so it runs the same natively and in the browser). A
    triangle seen from below (projected winding clockwise) is an entry, one
    seen from above an exit. Parts may overlap (an eye inside a head): the span
    ends where the running entry count first returns to zero. Cell centres are
    offset by irrational fractions of a cell so they do not land on mesh edges.
    """
    m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    assert isinstance(m, trimesh.Trimesh)
    lo, hi = m.bounds
    step = float(max(hi[0] - lo[0], hi[1] - lo[1])) / cells or 0.1
    pad = margin + 2 * step
    # Different offsets in x and y: equal ones would put cell centres on every x = y diagonal edge.
    xs = np.arange(lo[0] - pad, hi[0] + pad + step, step) + 0.1234567 * step
    ys = np.arange(lo[1] - pad, hi[1] + pad + step, step) + 0.3183099 * step
    cell, z, sign = _crossings(np.asarray(m.triangles, np.float64), xs, ys)
    z1, z2 = _first_spans(cell, z, sign, len(xs) * len(ys))
    return HeightField(float(xs[0]), float(ys[0]), step, z1.reshape(len(ys), len(xs)),
                       z2.reshape(len(ys), len(xs)))


def _crossings(tri: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               chunk: int = 4_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cell index, z, +1 entry / −1 exit) of every triangle-over-cell-centre crossing."""
    step = xs[1] - xs[0] if len(xs) > 1 else 1.0
    nx, ny = len(xs), len(ys)
    gx = (tri[:, :, 0] - xs[0]) / step
    gy = (tri[:, :, 1] - ys[0]) / step
    j0 = np.clip(np.ceil(gx.min(axis=1)), 0, nx).astype(np.int64)
    j1 = np.clip(np.floor(gx.max(axis=1)), -1, nx - 1).astype(np.int64)
    i0 = np.clip(np.ceil(gy.min(axis=1)), 0, ny).astype(np.int64)
    i1 = np.clip(np.floor(gy.max(axis=1)), -1, ny - 1).astype(np.int64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    w = np.maximum(j1 - j0 + 1, 0)
    h = np.maximum(i1 - i0 + 1, 0)
    count = w * h
    use = np.nonzero((count > 0) & (np.abs(area2) > 1e-14))[0]
    out_c, out_z, out_s = [], [], []
    # Process triangles in batches so the expanded (triangle, cell) list stays bounded.
    csum = np.cumsum(count[use])
    start = 0
    while start < len(use):
        stop = int(np.searchsorted(csum, (csum[start - 1] if start else 0) + chunk, side="right"))
        stop = max(stop, start + 1)
        t = use[start:stop]
        n = count[t]
        tid = np.repeat(t, n)
        k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        ii = i0[tid] + k // w[tid]
        jj = j0[tid] + k % w[tid]
        px, py = xs[jj], ys[ii]
        ax, ay, bx, by, cx, cy = a[tid, 0], a[tid, 1], b[tid, 0], b[tid, 1], c[tid, 0], c[tid, 1]
        d = area2[tid]
        l0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / d
        l1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / d
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        tz = l0 * a[tid, 2] + l1 * b[tid, 2] + l2 * c[tid, 2]
        out_c.append((ii * nx + jj)[inside])
        out_z.append(tz[inside])
        out_s.append(np.where(d[inside] < 0, 1, -1))      # clockwise from above = facing down = entry
        start = stop
    if not out_c:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    return np.concatenate(out_c), np.concatenate(out_z), np.concatenate(out_s)


def _first_spans(cell: np.ndarray, z: np.ndarray, sign: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: z where material first starts and where that span ends (inf / −inf where none)."""
    z1 = np.full(n, np.inf)
    z2 = np.full(n, -np.inf)
    if not len(cell):
        return z1, z2
    order = np.lexsort((z, cell))
    cell, z, sign = cell[order], z[order], sign[order]
    starts = np.r_[0, np.nonzero(np.diff(cell))[0] + 1]
    lengths = np.diff(np.r_[starts, len(cell)])
    cs = np.cumsum(sign)
    before = np.r_[0, cs[:-1]][starts]                       # running count before each cell's first hit
    depth = cs - np.repeat(before, lengths)
    idx = np.arange(len(cell))
    closed = np.where((depth <= 0) & (idx > np.repeat(starts, lengths)), idx, len(cell))
    first_close = np.minimum.reduceat(closed, starts)
    last = starts + lengths - 1
    end = np.where(first_close < len(cell), first_close, last)
    c0 = cell[starts]
    z1[c0] = z[starts]
    z2[c0] = np.where(end > starts, z[end], z[starts])
    return z1, z2


def _disc_filter(a: np.ndarray, radius_px: float, kind: str, fill: float) -> np.ndarray:
    """Min or max of `a` over a disc of `radius_px` cells around every cell.

    Decomposed into one 1-D filter per row offset (van Herk, O(n) each), much
    faster than a 2-D footprint filter of this size. Cells off the grid take `fill`.
    """
    n = int(np.floor(radius_px))
    fn = ndimage.maximum_filter1d if kind == "max" else ndimage.minimum_filter1d
    pad = np.pad(a, n + 1, constant_values=fill)
    rows: dict[int, np.ndarray] = {}
    h, w = a.shape
    out = np.full(a.shape, -np.inf if kind == "max" else np.inf)
    for dy in range(-n, n + 1):
        half = int(np.floor(np.sqrt(max(radius_px * radius_px - dy * dy, 0.0))))
        if half not in rows:
            rows[half] = fn(pad, size=2 * half + 1, axis=1, mode="constant", cval=fill)
        r = rows[half][n + 1 + dy:n + 1 + dy + h, n + 1:n + 1 + w]
        out = np.maximum(out, r) if kind == "max" else np.minimum(out, r)
    return out


@dataclass
class PocketMap:
    """Where a pocket centre may go on a height field, and how it would sit there."""
    field: HeightField
    seat: np.ndarray       # pocket seat height per cell (inf where no material)
    valid: np.ndarray      # a pocket centred here satisfies every rule
    room: np.ndarray       # ... every rule but `max_seat`
    depth: np.ndarray      # mm from the cell to the nearest invalid one


def pocket_map(f: HeightField, spec: PocketSpec) -> PocketMap:
    r = spec.diameter / 2 / f.step
    rw = (spec.diameter / 2 + spec.wall) / f.step
    z_hi = _disc_filter(f.z1, rw, "max", np.inf)              # highest underside point under the walled disc
    seat = np.maximum(0.0, z_hi + spec.grip - spec.depth)
    top = seat + spec.depth
    walls = _disc_filter(f.z2, rw, "min", -np.inf)            # walls reach the ceiling
    cover = _disc_filter(f.z2, r, "min", -np.inf)             # material over the ceiling
    eps = 1e-6
    room = np.isfinite(z_hi) & (walls >= top - eps) & (cover >= top + spec.ceiling - eps)
    valid = room & (seat <= spec.max_seat + eps)
    depth = ndimage.distance_transform_edt(valid) * f.step
    return PocketMap(f, seat, valid, room, depth)


Ring = tuple[float, float, float, float]   # (centre x, centre y, inner r, outer r) in the figure frame


def ring_polygon(ring: Ring) -> Polygon:
    cx, cy, lo, hi = ring
    return Point(cx, cy).buffer(hi, quad_segs=32).difference(Point(cx, cy).buffer(lo, quad_segs=32))


def auto_pockets(meshes: list[trimesh.Trimesh], spec: PocketSpec, ring: Ring | None = None,
                 count: int = 2) -> tuple[list[Pocket], list[Check]]:
    """Choose `count` pocket positions; with `ring`, all must lie on the washer's steel ring."""
    f = height_field(meshes, margin=spec.diameter / 2 + spec.wall)
    return choose_pockets(pocket_map(f, spec), spec, count, ring)


def _no_room(pm: PocketMap, spec: PocketSpec) -> Check:
    """Why no cell is valid: the largest circle inside material thick enough for pocket + ceiling."""
    f = pm.field
    if pm.room.any():
        # Rounded UP to 0.1 mm, so a limit set to this value admits the lowest spot.
        low = float(np.ceil(pm.seat[pm.room].min() * 10 - 1e-9) / 10)
        return Check("pockets:feet", "Underside", "fail",
                     f"The underside is at least {low:.1f} mm above the washer everywhere a pocket would fit; "
                     f"a magnet that high holds little (limit {spec.max_seat:.1f} mm).", value=low)
    thick = np.isfinite(f.z1) & (f.z2 - f.z1 >= spec.depth + spec.ceiling)
    largest = float(ndimage.distance_transform_edt(thick).max() * f.step) if thick.any() else 0.0
    need = spec.diameter / 2 + spec.wall
    return Check("pockets:feet", "Underside", "fail",
                 f"No spot on the underside holds a Ø{spec.diameter:.2f} mm pocket with {spec.wall:.1f} mm walls "
                 f"and {spec.ceiling:.1f} mm ceiling (needs an inscribed circle of r {need:.2f} mm in material "
                 f"≥ {spec.depth + spec.ceiling:.1f} mm thick; the largest has r {largest:.2f} mm).",
                 value=round(largest, 2))


def choose_pockets(pm: PocketMap, spec: PocketSpec, count: int = 2, ring: Ring | None = None,
                   allow_fewer: bool = True) -> tuple[list[Pocket], list[Check]]:
    """Up to `count` (1–3) pocket positions from a pocket map.

    Score (mm): spread (1 pocket: closeness to the footprint centre; 2: their
    distance; 3: 2·sqrt(triangle area)) + capped, weighted extra wall of each −
    SEAT_PENALTY × seat height of each (see the constants). Pockets keep a wall between them. With
    `allow_fewer`, fewer pockets are returned (with a warning) when `count`
    do not fit.
    """
    count = max(1, min(3, int(count)))
    if not pm.valid.any():
        return [], [_no_room(pm, spec)]
    f = pm.field
    xs, ys = f.xy()
    cand = pm.valid.copy()
    if ring is not None:
        cx, cy, lo, hi = ring
        d = np.hypot(xs[None, :] - cx, ys[:, None] - cy)
        cand &= (d >= lo) & (d <= hi)
    if not cand.any():
        return [], [Check("pockets:feet", "Underside", "fail",
                          "The underside has room for pockets, but none of it lies over the washer's steel ring "
                          "with the figure placed to fit the ball.")]
    full = np.argwhere(cand)
    all_pts = np.c_[xs[full[:, 1]], ys[full[:, 0]]]
    all_seat = pm.seat[full[:, 0], full[:, 1]]
    d_all = pm.depth[full[:, 0], full[:, 1]]
    # Capped, weighted extra wall − seat penalty; + 1 % of the raw depth to break ties toward the middle.
    all_base = DEPTH_WEIGHT * np.minimum(d_all, DEPTH_CAP) + 0.01 * d_all - SEAT_PENALTY * all_seat
    mat = np.isfinite(f.z1)
    centre = np.array([xs[mat.any(axis=0)].mean(), ys[mat.any(axis=1)].mean()])
    d_min = spec.diameter + spec.wall

    def spread(q: np.ndarray) -> float:
        """Stability term of a set of pocket centres (rows of q), mm."""
        if len(q) == 1:
            return -0.3 * float(np.hypot(*(q[0] - centre)))
        if len(q) == 2:
            return float(np.hypot(*(q[0] - q[1])))
        e1, e2 = q[1] - q[0], q[2] - q[0]
        return 2 * float(np.sqrt(0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])))

    # Search on a thinned lattice (pairs/triples grow fast), then refine each pocket at full resolution.
    k = int(np.ceil(np.sqrt(len(full) / MAX_CANDIDATES[count])))
    sel = np.arange(len(full))
    if k > 1:
        thin = sel[(full[:, 0] % k == 0) & (full[:, 1] % k == 0)]
        sel = thin if len(thin) >= count else sel
    pts, base = all_pts[sel], all_base[sel]
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    ok = dist >= d_min

    def pick(c: int) -> list[int] | None:
        """Indices into the thinned set."""
        if c == 1:
            return [int(np.argmax(base - 0.3 * np.hypot(*(pts - centre).T)))]
        if c == 2:
            score = np.where(ok, dist + base[:, None] + base[None, :], -np.inf)
            a, b = np.unravel_index(int(np.argmax(score)), score.shape)
            return [int(a), int(b)] if np.isfinite(score[a, b]) else None
        best: list[int] | None = None
        best_s = -np.inf
        for i in range(len(pts)):
            e = pts - pts[i]
            area = 0.5 * np.abs(e[:, None, 0] * e[None, :, 1] - e[:, None, 1] * e[None, :, 0])
            good = ok[i][:, None] & ok[i][None, :] & ok
            score = np.where(good, 2 * np.sqrt(area) + base[i] + base[:, None] + base[None, :], -np.inf)
            j, kk = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[j, kk] > best_s:
                best, best_s = [i, int(j), int(kk)], float(score[j, kk])
        return best

    def refine(idx: list[int]) -> list[int]:
        """Move each pocket to the best full-resolution cell within ~1.5 lattice steps (others fixed)."""
        idx = list(idx)
        reach = 1.5 * k * f.step
        for n in range(len(idx)):
            near = np.nonzero(np.hypot(*(all_pts - all_pts[idx[n]]).T) <= reach)[0]
            others = [all_pts[m] for o, m in enumerate(idx) if o != n]
            best_m, best_s = idx[n], -np.inf
            for m in near.tolist():
                if any(np.hypot(*(all_pts[m] - q)) < d_min for q in others):
                    continue
                q = np.array([all_pts[m] if o == n else all_pts[mm] for o, mm in enumerate(idx)])
                sc = spread(q) + float(sum(all_base[mm] for mm in idx if mm != idx[n]) + all_base[m])
                if sc > best_s:
                    best_m, best_s = m, sc
            idx[n] = best_m
        return idx

    for c in (range(count, 0, -1) if allow_fewer else [count]):
        chosen = pick(c)
        if chosen is None:
            continue
        idx = refine([int(sel[i]) for i in chosen]) if k > 1 else [int(sel[i]) for i in chosen]
        pockets = [Pocket(float(all_pts[i, 0]), float(all_pts[i, 1]), float(all_seat[i])) for i in idx]
        if c < count:
            return pockets, [Check("pockets:feet", "Underside", "warn",
                                   f"Only room for {c} of {count} pockets (each needs a {spec.wall:.1f} mm wall "
                                   f"to the next); placed {c}.")]
        gaps = [float(np.hypot(*(all_pts[a] - all_pts[b]))) for n, a in enumerate(idx) for b in idx[n + 1:]]
        msg = ("One pocket." if c == 1 else
               f"{c} pockets, {min(gaps):.1f} mm apart (centre to centre{', closest pair' if c == 3 else ''}).")
        return pockets, [Check("pockets:feet", "Underside", "pass", msg)]
    return [], [Check("pockets:feet", "Underside", "fail", f"Room for fewer than {count} pockets.")]


def seat_of(f: HeightField, x: float, y: float, spec: PocketSpec) -> float:
    """Lowest seat for a pocket at (x, y) that keeps `grip` mm of it walled (≥ 0; inf without material)."""
    _, z1, _ = f.window(x, y, spec.diameter / 2 + spec.wall)
    z1 = z1[np.isfinite(z1)]
    return float(max(0.0, z1.max() + spec.grip - spec.depth)) if len(z1) else float("inf")


def seat_pockets(f: HeightField, pockets: list[Pocket], spec: PocketSpec) -> list[Pocket]:
    """The same XY positions with their seat heights."""
    return [Pocket(p.x, p.y, seat_of(f, p.x, p.y, spec)) for p in pockets]


def validate_pockets(source: HeightField | list[trimesh.Trimesh], pockets: list[Pocket],
                     spec: PocketSpec) -> list[Check]:
    """Measured wall, ceiling and seat of each pocket, plus the spacing between them."""
    f = source if isinstance(source, HeightField) else height_field(source, margin=spec.diameter / 2 + spec.wall)
    r = spec.diameter / 2
    checks = []
    for i, p in enumerate(pockets):
        seat = seat_of(f, p.x, p.y, spec)
        top = seat + spec.depth
        search = r + spec.wall + 1.0
        d, z1, z2 = f.window(p.x, p.y, search)
        bad = ~np.isfinite(z1) | (z1 > top - spec.grip + 1e-6) | (z2 < top - 1e-6)
        wall = float((d[bad].min() if bad.any() else search) - r)
        inner = d <= r
        solid = inner & np.isfinite(z1)
        covered = float(np.mean(solid[inner] & (z2[inner] >= top + spec.ceiling - 1e-6))) if inner.any() else 0.0
        ok_ceil = covered > 0.999
        status: Status
        if wall < MIN_CUT_WALL or not ok_ceil or not np.isfinite(seat):
            status = "fail"            # breaks through the side or the top: not cut
        elif wall < spec.wall - f.step / 2 or seat > LOW_SEAT:
            status = "warn"            # cut, but the wall is thin or the magnet sits high
        else:
            status = "pass"
        if not np.isfinite(seat) or wall + r <= 0:
            msg = "Pocket centre is not under the figure."
        else:
            msg = (f"Breaks through the side by {-wall:.2f} mm, " if wall < 0 else f"Wall {wall:.2f} mm (min {spec.wall:.1f}), ")
            msg = (msg +
                   f"ceiling {'covered' if ok_ceil else f'only {covered:.0%} covered'}. ")
            msg += ("Sits on the washer." if seat < 0.05 else
                    f"Opening {seat:.1f} mm above the washer" +
                    (" (too far to hold much)." if seat > spec.max_seat else
                     " (weaker hold)." if seat > LOW_SEAT else "."))
            proud = float(z1[solid].max() - seat) if solid.any() else 0.0
            if proud > 0.05:
                msg += (f" On a slope: the magnet stands up to {min(proud, spec.depth - spec.grip):.1f} mm "
                        "proud of the surface.")
        if status == "fail":
            msg += " Not cut."
        checks.append(Check(f"pocket:{i + 1}", f"Pocket {i + 1}", status, msg, round(wall, 2),
                            [[p.x, p.y, seat if np.isfinite(seat) else 0.0]]))
    for a in range(len(pockets)):
        for b in range(a + 1, len(pockets)):
            gap = float(np.hypot(pockets[a].x - pockets[b].x, pockets[a].y - pockets[b].y)) - spec.diameter
            if gap < spec.wall:
                checks.append(Check("pocket:gap", "Pocket spacing", "fail",
                                    f"Pockets {a + 1} and {b + 1} are {gap:.2f} mm apart "
                                    f"(need ≥ {spec.wall:.1f} mm wall between)."))
    return checks


def pocket_solid(pockets: list[Pocket], spec: PocketSpec) -> mf.Manifold:
    """Cutting tool: cylinders from below the bed up to each pocket's ceiling."""
    r = spec.diameter / 2
    tool = None
    for p in pockets:
        c = mf.Manifold.cylinder(p.z + spec.depth + 1.0, r, r, SEGMENTS).translate((p.x, p.y, -1.0))
        tool = c if tool is None else tool + c
    assert tool is not None
    return tool


def cut_pockets(parts: list[Part], pockets: list[Pocket], spec: PocketSpec) -> tuple[list[Part], list[Check]]:
    """Subtract the pockets from every part they touch; returns new parts + per-part results."""
    if not pockets:
        return parts, []
    tool = pocket_solid(pockets, spec)
    tb = np.asarray(tool.bounding_box()).reshape(2, 3)
    out, checks = [], []
    for part in parts:
        lo, hi = part.vertices.min(axis=0), part.vertices.max(axis=0)
        if part.subtype != "normal_part" or np.any(hi < tb[0]) or np.any(lo > tb[1]):
            out.append(part)
            continue
        try:
            mesh = part.mesh()
            mesh.merge_vertices()      # STL soups and some 3MF writers duplicate shared vertices
            body = to_manifold(mesh)
        except RuntimeError as exc:
            checks.append(Check(f"cut:{part.name}", f"Cut {part.name}", "fail",
                                f"Cannot cut pockets: part is not a closed solid ({exc}). Repair it first."))
            out.append(part)
            continue
        cut = body - tool
        if abs(cut.volume() - body.volume()) < 1e-6:
            out.append(part)
            continue
        removed = body.volume() - cut.volume()
        out.append(remesh_part(part, from_manifold(cut)))
        checks.append(Check(f"cut:{part.name}", f"Cut {part.name}", "pass",
                            f"Removed {removed:.1f} mm³ for {len(pockets)} pocket(s).", round(removed, 2)))
    return out, checks


def required_washer(pockets: list[Pocket], spec: PocketSpec) -> tuple[float, float]:
    """(min OD, max ID) for every pocket to sit on steel, the washer centred on the pockets' centroid."""
    if len(pockets) < 2:
        return spec.diameter, 0.0
    xy = np.array([[p.x, p.y] for p in pockets])
    r = np.hypot(*(xy - xy.mean(axis=0)).T)
    return float(2 * r.max() + spec.diameter), float(max(0.0, 2 * (r.min() - spec.diameter / 2)))


