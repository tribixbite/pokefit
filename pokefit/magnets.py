"""Magnet pockets in the figure's feet.

Pockets are open-bottomed blind holes for disc magnets (default Ø4.18 × 2.20
mm: the pockets of the Zelda pot set, sized for 4 × 2 mm magnets). All
coordinates here are final print mm in the figure frame: XY as the figure
will be printed, z = 0 on the bed (= the figure's underside).

Placement rules (spec "Magnet pockets"):
  * feet = islands of the cross-section just above the bed;
  * the pocket disc grown by `wall` must lie inside the figure's
    cross-section at the bottom, middle and top of the pocket;
  * `ceiling` mm of material must cover the pocket;
  * two feet → one pocket per foot at its most-inland point;
    one foot → two pockets on its long axis.
Nothing is cut when a rule fails; the failure is reported with numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import manifold3d as mf
import shapely
import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import polylabel, unary_union

from .model3mf import Part, remesh_part
from .report import Check, Status
from .shell import from_manifold, to_manifold

Z_PROBE = 0.3        # height of the "feet" section above the bed, mm
SEGMENTS = 48        # polygon sides of the pocket cylinder (as in the Zelda pot)
MIN_CUT_WALL = 0.2   # mm; thinner than this the pocket would break through the foot, so it is not cut


@dataclass(frozen=True)
class PocketSpec:
    diameter: float = 4.18
    depth: float = 2.20
    wall: float = 0.8
    ceiling: float = 0.6


@dataclass
class Pocket:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


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


def _levels(spec: PocketSpec) -> tuple[list[float], list[float]]:
    """Heights where the walled disc must fit, and where the ceiling must cover it."""
    walls = [0.1, spec.depth / 2, spec.depth - 0.05]
    ceil = [spec.depth + 0.05, spec.depth + spec.ceiling - 0.05]
    return walls, ceil


def feasible_region(meshes: list[trimesh.Trimesh], spec: PocketSpec) -> Polygon | MultiPolygon:
    """Where a pocket centre may go: walled disc inside the walls, disc under material."""
    r = spec.diameter / 2
    walls, ceil = _levels(spec)
    region = None
    for z in walls:
        a = section(meshes, z).buffer(-(r + spec.wall))
        region = a if region is None else region.intersection(a)
    for z in ceil:
        region = region.intersection(section(meshes, z).buffer(-r))  # type: ignore[union-attr]
    return region if region is not None else Polygon()


def _inscribed_radius(poly: Polygon) -> float:
    c = polylabel(poly, tolerance=0.01)
    return float(poly.exterior.distance(c)) if poly.contains(c) else 0.0


Ring = tuple[float, float, float, float]   # (centre x, centre y, inner r, outer r) in the figure frame


def ring_polygon(ring: Ring) -> Polygon:
    cx, cy, lo, hi = ring
    return Point(cx, cy).buffer(hi, quad_segs=32).difference(Point(cx, cy).buffer(lo, quad_segs=32))


def auto_pockets(meshes: list[trimesh.Trimesh], spec: PocketSpec,
                 ring: Ring | None = None) -> tuple[list[Pocket], list[Check]]:
    """Choose two pocket positions; with `ring`, both must lie on the washer's steel ring."""
    return choose_pockets(find_feet(meshes), feasible_region(meshes, spec), spec, ring)


def choose_pockets(feet: list[Polygon], region: Polygon | MultiPolygon, spec: PocketSpec,
                   ring: Ring | None = None, allow_single: bool = True) -> tuple[list[Pocket], list[Check]]:
    """Pocket positions from precomputed feet and feasible region (see `feasible_region`).

    Two separate usable areas → one pocket at the most inland point of each.
    One area → the pair of valid centres farthest apart (steadier), deeper
    inside on ties. `allow_single` returns one pocket when two do not fit.
    """
    r = spec.diameter / 2
    need = r + spec.wall
    if not feet:
        return [], [Check("pockets:feet", "Feet", "fail",
                          f"No cross-section {Z_PROBE} mm above the bed: the figure has no flat underside.")]
    if region.is_empty:
        largest = max(_inscribed_radius(f) for f in feet[:4])
        return [], [Check("pockets:feet", "Feet", "fail",
                          f"No spot in the feet holds a Ø{spec.diameter:.2f} mm pocket with {spec.wall:.1f} mm "
                          f"walls (needs an inscribed circle of r {need:.2f} mm, largest foot has r {largest:.2f} mm).",
                          value=round(largest, 2))]
    cand = region if ring is None else region.intersection(ring_polygon(ring))
    comps = sorted((g for g in _parts(cand) if g.area > 1e-4), key=lambda g: -g.area)
    if not comps:
        return [], [Check("pockets:feet", "Feet", "fail",
                          "The feet have room for pockets, but none of it lies over the washer's steel ring "
                          "with the figure placed to fit the ball.")]

    by_foot: list[Polygon] = []
    for foot in feet:
        inside = [g for g in comps if foot.intersects(g)]
        if inside:
            by_foot.append(max(inside, key=lambda g: g.area))
        if len(by_foot) == 2:
            break
    if len(by_foot) == 2:
        pockets = []
        for g in by_foot:
            c = polylabel(g, tolerance=0.01)
            if not g.contains(c):
                c = g.representative_point()
            pockets.append(Pocket(float(c.x), float(c.y)))
        return pockets, [Check("pockets:feet", "Feet", "pass",
                               f"Two feet found; one pocket in each ({len(feet)} contact islands).")]

    pair = _farthest_pair(cand, region, spec.diameter + spec.wall)
    if pair is None:
        if not allow_single:
            return [], [Check("pockets:feet", "Feet", "fail", "Room for only one pocket.")]
        c0 = polylabel(comps[0], tolerance=0.01)
        if not comps[0].contains(c0):
            c0 = comps[0].representative_point()
        return [Pocket(float(c0.x), float(c0.y))], [Check(
            "pockets:feet", "Feet", "warn",
            "Only room for ONE pocket (too small for two with a wall between); placed one.")]
    a, b = pair
    d = float(np.hypot(*(a - b)))
    return [Pocket(float(a[0]), float(a[1])), Pocket(float(b[0]), float(b[1]))], [Check(
        "pockets:feet", "Feet", "pass", f"One contact area; two pockets {d:.1f} mm apart (centre to centre).")]


def _farthest_pair(cand: Polygon | MultiPolygon, region: Polygon | MultiPolygon,
                   d_min: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Two points of `cand` at least d_min apart, as far apart as possible.

    Grid search (~60 cells across); ties prefer points deeper inside `region`
    (i.e. thicker walls).
    """
    from scipy.spatial.distance import pdist, squareform
    from shapely import contains_xy, distance

    x0, y0, x1, y1 = cand.bounds
    step = max(x1 - x0, y1 - y0) / 60 or 0.05
    gx, gy = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step))
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[contains_xy(cand, pts[:, 0], pts[:, 1])]
    if len(pts) < 2:
        return None
    depth = distance(region.boundary, shapely.points(pts))
    dist = squareform(pdist(pts))
    valid = dist >= d_min
    if not valid.any():
        return None
    score = np.where(valid, dist + 1.0 * (depth[:, None] + depth[None, :]), -np.inf)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return pts[i], pts[j]


def validate_pockets(meshes: list[trimesh.Trimesh], pockets: list[Pocket], spec: PocketSpec) -> list[Check]:
    """Measured wall around and material above each pocket."""
    r = spec.diameter / 2
    walls, ceil = _levels(spec)
    secs = [section(meshes, z) for z in walls]
    ceil_secs = [section(meshes, z) for z in ceil]
    checks = []
    for i, p in enumerate(pockets):
        pt = Point(p.x, p.y)
        wall = min((s.boundary.distance(pt) - r) if s.contains(pt) else -r for s in secs) if secs else -r
        disc = pt.buffer(r, quad_segs=12)
        covered = min((s.intersection(disc).area / disc.area) for s in ceil_secs) if ceil_secs else 0.0
        ok_ceil = covered > 0.999
        status: Status
        if wall < MIN_CUT_WALL or not ok_ceil:
            status = "fail"            # breaks through the side or the top: not cut
        elif wall < spec.wall - 1e-3:
            status = "warn"            # cut, but the wall is thin
        else:
            status = "pass"
        msg = f"Wall {wall:.2f} mm (min {spec.wall:.1f}), ceiling {'covered' if ok_ceil else f'only {covered:.0%} covered'}."
        if wall < 0:
            msg = "Pocket centre is outside the foot."
        if status == "fail":
            msg += " Not cut."
        checks.append(Check(f"pocket:{i + 1}", f"Pocket {i + 1}", status, msg, round(float(wall), 2),
                            [[p.x, p.y, 0.0]]))
    if len(pockets) == 2:
        gap = float(np.hypot(pockets[0].x - pockets[1].x, pockets[0].y - pockets[1].y)) - spec.diameter
        if gap < spec.wall:
            checks.append(Check("pocket:gap", "Pocket spacing", "fail",
                                f"Pockets are {gap:.2f} mm apart (need ≥ {spec.wall:.1f} mm wall between)."))
    return checks


def pocket_solid(pockets: list[Pocket], spec: PocketSpec) -> mf.Manifold:
    """Cutting tool: cylinders from below the bed up to the pocket ceiling."""
    r = spec.diameter / 2
    tool = None
    for p in pockets:
        c = mf.Manifold.cylinder(spec.depth + 1.0, r, r, SEGMENTS).translate((p.x, p.y, -1.0))
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
            body = to_manifold(part.mesh())
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
    """(min OD, max ID) for both pockets to sit on steel with the pair centred on the washer."""
    if len(pockets) < 2:
        return spec.diameter, 0.0
    half = float(np.hypot(pockets[0].x - pockets[1].x, pockets[0].y - pockets[1].y)) / 2
    return 2 * half + spec.diameter, max(0.0, 2 * (half - spec.diameter / 2))

