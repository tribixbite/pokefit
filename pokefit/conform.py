"""Conform a figure to the ball's floor: bend outstretched legs and tails up along the sphere, and
press the base flat over the washer.

Why: the cavity floor is flat only over the washer (an M6 DIN 9021 is 18 mm across); beyond it the
sphere rises (60 % ball: 2 mm up at 13.7 mm from the axis, 4 mm at 15.7 mm). Feet or a tail lying
straight out reach that slope first, which caps the figure's size; and a rounded underside rests on
a point, which leaves the magnets little to sit on.

Both steps only MOVE vertices. Faces and their order are unchanged, so paint, a glTF's texture
colour and the tap-to-paint topology all carry over exactly.

**Bend**: z += k · lift(r) · w(z), in final millimetres.
- r is the distance from the ball axis.
- lift(r) is the sphere's own floor curve, starting at the flat disc's edge. It eases in over
  EASE mm, so a leg curves rather than kinks, and its slope is capped at MAX_SLOPE.
- w is 1 in the lowest BAND of the figure's height and fades to 0 by BAND + FADE, so arms and the
  head are untouched. A foot moves up as a whole, so it bends rather than squashes.
- k is the strength. Auto mode sets it from two things:
  - the "limb share": the fraction of the low footprint beyond the disc whose first solid span
    ends below LIMB_TOP of the height, i.e. something with free space above it (a foot, a tail),
    not a column of body (a box's corner);
  - the low footprint's aspect ratio: wide, flat-lying figures bend more, upright ones not at all.
  Auto mode keeps the bend only if the figure then fits at least MIN_GAIN larger.
- The lift depends on the print size, which depends on the bend, so the fit is redone until the
  scale settles (ROUNDS).

**Flatten**: vertices below the plane z = t are lifted onto it, and then the figure drops by t.
This gives a flat contact patch on the washer. t is the smallest depth at which FLAT_TARGET of
the figure's footprint over the disc touches the bed, capped at `max_depth` mm. A base that is
already flat needs t ≈ 0 and is left alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from .ball import Cavity
from .fit import max_scale, prepare_points
from .repair import repaired
from .magnets import HeightField, height_field
from .model3mf import Part

BAND = 0.30          # lowest fraction of the height that a bend lifts fully
FADE = 0.20          # ...fading to no lift over this fraction above it
EASE = 3.0           # mm over which the lift's slope eases in past the disc edge
MAX_SLOPE = np.tan(np.radians(40.0))
LIMB_TOP = 0.5       # a low part whose first solid span ends below this fraction of the height is a limb
MIN_GAIN = 0.01      # auto bend is kept only if the figure fits at least 1 % larger
ROUNDS = 3
FIT_TOL = 2e-3       # relative tolerance of the conform step's fits (rounds settle at 0.5 %)
FLAT_TARGET = 0.6    # share of the footprint over the disc the flat base aims to touch
MIN_DEPTH = 0.02     # mm; shallower presses are skipped (the base is flat already)
DEFAULT_MAX_DEPTH = 1.2
DEFAULT_FILL = 2.5   # mm: largest gap under the figure the flat base fills
FILL_CELL = 0.4      # mm: grid of the fill solid
FILL_OVERLAP = 0.4   # mm the fill reaches into the figure (the slicer unions overlapping parts)
FILL_NAME = "flat base"
OUTLINE_ROUNDS = 3   # Laplacian passes rounding the fill's grid-stepped outline


@dataclass
class ConformSpec:
    bend: float | None = None       # None = automatic, 0 = off, else strength 0..1
    flat_diameter: float = 18.0     # mm, the disc pressed flat (default: the washer); 0 = off
    max_depth: float = DEFAULT_MAX_DEPTH  # mm, deepest press for the flat base
    fill: float = DEFAULT_FILL      # mm, largest gap under the figure filled by the flat base (0 = off)
    scale: float | None = None      # print scale to design for (None = the largest that fits)
    # The fill over a circle the user placed (x, y, diameter; figure frame, scale 1) instead of the flat disc
    fill_circle: tuple[float, float, float] | None = None
    # ... or inside an outline the user drew (figure frame, scale 1; first wins of shape, circle, everywhere)
    fill_shape: tuple[tuple[float, float], ...] | None = None
    # ... or under the whole figure: every part hovering at most `fill` mm above the bed is filled down to it
    fill_all: bool = False


@dataclass
class ConformInfo:
    bend: float = 0.0               # strength used
    auto: bool = True
    limb_share: float = 0.0
    lift_mm: float = 0.0            # largest lift of any vertex
    flat_depth_mm: float = 0.0
    flat_share: float = 0.0         # share of the footprint over the disc on the bed after pressing
    fill_mm2: float = 0.0           # area of the flat base added under the figure
    scale_before: float = 0.0       # largest fit scale before / after (no pockets)
    scale_after: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, float | bool | str]:
        d = asdict(self)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


def lift_curve(cavity: Cavity, r_flat: float, r_max: float, step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(r, lift) in final mm: the sphere's floor curve beyond the flat disc, eased in, slope capped."""
    big_r = cavity.r_inner - cavity.clearance
    r = np.arange(0.0, max(r_max, r_flat) + 2 * step, step)
    inside = np.clip(big_r ** 2 - r ** 2, 1e-9, None)
    slope = np.minimum(r / np.sqrt(inside), MAX_SLOPE)
    slope = np.where(r < big_r, slope, MAX_SLOPE)
    slope *= np.clip((r - r_flat) / EASE, 0.0, 1.0)
    lift = np.concatenate([[0.0], np.cumsum((slope[1:] + slope[:-1]) / 2 * step)])
    return r, lift


def band_weight(z: np.ndarray, height: float) -> np.ndarray:
    """1 in the lowest BAND of the height, smoothly 0 from BAND + FADE up."""
    u = np.clip((z / max(height, 1e-9) - BAND) / FADE, 0.0, 1.0)
    return 1.0 - u * u * (3 - 2 * u)


def _field(v: np.ndarray, parts: list[Part]) -> HeightField:
    meshes, start = [], 0
    for p in parts:
        n = len(p.vertices)
        meshes.append(Part(p.name, p.extruder, v[start:start + n], p.faces, {}, p.subtype).mesh())
        start += n
    return height_field(meshes)


def _disc_cells(field: HeightField, axis: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance from axis, z1, z2) of the footprint cells (unit scale)."""
    xs, ys = field.xy()
    gx, gy = np.meshgrid(xs, ys)
    d = np.hypot(gx - axis[0], gy - axis[1])
    on = np.isfinite(field.z1) & (d <= radius)
    return d[on], field.z1[on], field.z2[on]


def limb_share(field: HeightField, axis: np.ndarray, r_flat_unit: float, height: float) -> float:
    """Share of the low footprint beyond the disc that is a limb (free space above its first span)."""
    d, z1, z2 = _disc_cells(field, axis, np.inf)
    low = (d > r_flat_unit) & (z1 <= BAND * height)
    if not low.any():
        return 0.0
    return float(np.mean(z2[low] < LIMB_TOP * height))


def low_aspect(v: np.ndarray, height: float) -> float:
    """Horizontal extent of the lowest BAND of the figure over its height."""
    low = v[v[:, 2] <= BAND * height]
    if len(low) < 3:
        return 0.0
    return float(np.ptp(low[:, :2], axis=0).max() / max(height, 1e-9))


def flat_depth(field: HeightField, axis: np.ndarray, r_disc: float, cap: float) -> tuple[float, float]:
    """(t, share on the bed): smallest press depth (unit scale) reaching FLAT_TARGET of the
    footprint over the disc, capped at `cap`."""
    _, z1, _ = _disc_cells(field, axis, r_disc)
    if len(z1) == 0:
        return 0.0, 0.0
    z1 = np.sort(z1 - min(0.0, float(z1.min())))
    t = float(z1[min(len(z1) - 1, int(np.ceil(FLAT_TARGET * len(z1))) - 1)])
    t = min(t, cap)
    return t, float(np.mean(z1 <= t + 1e-9))


def _with_vertices(parts: list[Part], v: np.ndarray) -> list[Part]:
    out, start = [], 0
    for p in parts:
        n = len(p.vertices)
        out.append(Part(p.name, p.extruder, v[start:start + n], p.faces, p.tri_extra, p.subtype))
        start += n
    return out


def _fit(v: np.ndarray, cavity: Cavity, hint: float | None = None) -> tuple[float, np.ndarray] | None:
    """(largest scale, figure-frame XY of the ball axis) — no pockets. To FIT_TOL: the rounds stop at
    0.5 % change; `hint` (the previous round's scale) narrows the search."""
    res = max_scale(prepare_points(v), cavity, tol=FIT_TOL, hint=hint)
    if res is None:
        return None
    return res.max_scale, -np.asarray(res.offset) / res.max_scale


@dataclass
class _Shaped:
    v: np.ndarray
    depth: float        # press depth, unit scale
    share: float        # share of the disc footprint on the bed
    axis: np.ndarray    # figure-frame XY of the ball axis it was designed for


def _shape(v0: np.ndarray, parts: list[Part], axis: np.ndarray, s: float, k: float, cavity: Cavity,
           spec: ConformSpec, height: float, field0: HeightField | None = None) -> _Shaped:
    """Bend (strength k) and press the disc flat, designed for print scale s with the ball axis at `axis`.

    The press lowers only the disc: parts beyond it (inside the bend band) are raised by the
    same depth first, so pressing cannot push feet or a tail into the sphere's slope.
    `field0` is v0's height field when the caller has it (unbent, v0 is pressed as it is).
    """
    r_flat = spec.flat_diameter / 2 if spec.flat_diameter > 0 else 0.0
    rr = np.linalg.norm(v0[:, :2] - axis, axis=1) * s
    w = band_weight(v0[:, 2], height)
    v = v0.copy()
    if k > 0:
        r, lift = lift_curve(cavity, max(r_flat, 1e-6), float(rr.max()))
        v[:, 2] += k * np.interp(rr, r, lift) / s * w
    v[:, 2] -= v[:, 2].min()
    if r_flat <= 0 or spec.max_depth <= 0:
        return _Shaped(v, 0.0, 0.0, axis)
    field = field0 if k <= 0 and field0 is not None else _field(v, parts)
    t, share = flat_depth(field, axis, r_flat / s, spec.max_depth / s)
    if t * s < MIN_DEPTH:
        return _Shaped(v, 0.0, share, axis)
    u = np.clip((rr - r_flat) / EASE, 0.0, 1.0)
    v[:, 2] += t * (u * u * (3 - 2 * u)) * w
    v[:, 2] = np.maximum(v[:, 2], t) - t
    return _Shaped(v, t, share, axis)


def conform(parts: list[Part], cavity: Cavity, spec: ConformSpec) -> tuple[list[Part], ConformInfo]:
    """Figure-frame parts (scale 1, lowest point z = 0) bent and flattened for the ball (module doc)."""
    info = ConformInfo(auto=spec.bend is None)
    v0 = np.vstack([p.vertices for p in parts]).astype(np.float64)
    v0[:, 2] -= v0[:, 2].min()
    height = float(np.ptp(v0[:, 2]))
    first = _fit(v0, cavity)
    if first is None or height <= 0:
        info.note = "does not fit; left as is"
        return parts, info
    info.scale_before = first[0]
    s0 = float(spec.scale) if spec.scale else first[0]
    r_flat = spec.flat_diameter / 2 if spec.flat_diameter > 0 else 0.0
    field0 = _field(v0, parts)
    info.limb_share = limb_share(field0, first[1], max(r_flat, 1e-6) / s0, height)
    k = spec.bend if spec.bend is not None else info.limb_share * float(
        np.clip(2 * (low_aspect(v0, height) - 0.5), 0.0, 1.0))
    k = float(np.clip(k, 0.0, 1.0))

    def solve(strength: float) -> tuple[_Shaped, float]:
        """Shape, refitting until the design scale settles; returns (shape, its largest fit scale)."""
        s, axis = s0, first[1]
        shaped = _shape(v0, parts, axis, s, strength, cavity, spec, height, field0)
        got = _fit(shaped.v, cavity, hint=s)
        for _ in range(ROUNDS if strength > 0 else 0):
            if got is None:
                break
            settled = abs(got[0] - s) / s < 0.005
            s, axis = (s0 if spec.scale else got[0]), got[1]
            if settled or spec.scale:
                break
            shaped = _shape(v0, parts, axis, s, strength, cavity, spec, height, field0)
            got = _fit(shaped.v, cavity, hint=s)
        return shaped, (got[0] if got else 0.0)

    flat, flat_scale = solve(0.0)
    best, best_scale = flat, flat_scale
    if k > 0:
        bent, bent_scale = solve(k)
        gain = bent_scale / max(flat_scale, 1e-12) - 1
        if spec.bend is None and gain < MIN_GAIN:
            info.note = f"bend skipped: the figure would fit only {max(gain, 0.0) * 100:.1f} % larger"
        else:
            best, best_scale, info.bend = bent, bent_scale, k
    info.scale_after = best_scale
    info.flat_depth_mm = best.depth * (s0 if spec.scale else best_scale)
    info.flat_share = best.share
    info.lift_mm = float(np.max(best.v[:, 2] - v0[:, 2]) * (s0 if spec.scale else best_scale))
    s_final = s0 if spec.scale else best_scale
    out = parts if best.depth == 0 and info.bend == 0 else _with_vertices(parts, best.v)
    base: Part | None = None
    if spec.fill > 0 and s_final > 0:
        # A region the user chose fills every gap in it, joined to the figure's footing or not; the default disc
        # over the washer only the patches joined to where the figure already stands.
        region: Region | None = None
        if spec.fill_shape is not None:
            region = shape_region(spec.fill_shape)
        elif spec.fill_circle is not None:
            cx, cy, d = spec.fill_circle
            region = disc_region([(cx, cy)], d / 2)
        elif spec.fill_all:
            region = everywhere
        if region is not None:
            base = base_fill(out, region, spec.fill / s_final, FILL_CELL / s_final, FILL_OVERLAP / s_final,
                             parts[0].extruder, connected=False)
        elif r_flat > 0:
            base = base_fill(out, disc_region([tuple(best.axis)], r_flat / s_final), spec.fill / s_final,
                             FILL_CELL / s_final, FILL_OVERLAP / s_final, parts[0].extruder)
    if base is not None:
        out = [*out, base]
        info.fill_mm2 = float(base.mesh().area_faces[base.mesh().face_normals[:, 2] < -0.9].sum()) * s_final ** 2
    return out, info


def smooth_outline(px: np.ndarray, py: np.ndarray, walls: list[tuple[int, int]], rounds: int = OUTLINE_ROUNDS
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Round off the flat base's grid-stepped outline: corners on its walls move toward the mean of their
    two outline neighbours (Laplacian, `rounds` times). Corners with other than two outline neighbours
    (a pinch) stay put. Top and bottom share these XY, so walls stay vertical and the solid closed."""
    nbr: dict[int, list[int]] = {}
    for e0, e1 in walls:
        nbr.setdefault(e0, []).append(e1)
        nbr.setdefault(e1, []).append(e0)
    ring = np.array([k for k, v in nbr.items() if len(v) == 2], np.int64)
    if not len(ring):
        return px, py
    a = np.array([nbr[k][0] for k in ring], np.int64)
    b = np.array([nbr[k][1] for k in ring], np.int64)
    x, y = px.astype(np.float64).copy(), py.astype(np.float64).copy()
    for _ in range(rounds):
        nx = 0.5 * x[ring] + 0.25 * (x[a] + x[b])
        ny = 0.5 * y[ring] + 0.25 * (y[a] + y[b])
        x[ring], y[ring] = nx, ny
    return x, y


# Which grid cells (centres gx, gy; cell size) a fill may cover: whole cells inside its region.
Region = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def disc_region(centres: list[tuple[float, float]], radius: float) -> Region:
    """Cells within `radius` of any of `centres` (the washer's disc, a fill circle, the pads under pockets)."""
    def inside(gx: np.ndarray, gy: np.ndarray, step: float) -> np.ndarray:
        out = np.zeros(gx.shape, bool)
        for cx, cy in centres:
            out |= np.hypot(gx - cx, gy - cy) <= radius - step / 2
        return out
    return inside


def shape_region(points: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> Region:
    """Cells inside an outline the user drew (a freehand loop may cross itself: its even-odd area counts)."""
    import shapely
    from shapely.geometry import Polygon

    poly = shapely.make_valid(Polygon(points))

    def inside(gx: np.ndarray, gy: np.ndarray, step: float) -> np.ndarray:
        core = poly.buffer(-step / 2) if poly.area > 0 else poly
        return np.asarray(shapely.contains_xy(core, gx, gy), bool) if not core.is_empty else np.zeros(gx.shape, bool)
    return inside


def everywhere(gx: np.ndarray, gy: np.ndarray, step: float) -> np.ndarray:
    """No limit: every cell under the figure (gaps under it up to the fill height)."""
    return np.ones(gx.shape, bool)


def base_fill(parts: list[Part], region: Region, max_gap: float, cell: float, overlap: float,
              extruder: int, connected: bool = True) -> Part | None:
    """A solid filling the gap between the bed and the figure's underside over `region`.

    Built on a coarse grid (`cell`) from the underside height field: flat bottom at z = 0, top at
    the underside (the lowest point in each cell) + `overlap`, vertical walls round its outline.
    Only cells where the gap is at most `max_gap` and fully under the figure are filled, and with
    `connected` only those joined to where the figure already touches the bed (or the largest
    patch); a region the user chose (`connected=False`) fills all of them. Unit-scale frame.
    """
    from scipy import ndimage

    solid = [p for p in parts if p.subtype == "normal_part"]
    field = height_field([p.mesh() for p in solid])
    b = max(1, int(round(cell / field.step)))
    ny, nx = field.z1.shape
    cy, cx = ny // b, nx // b
    if cy < 2 or cx < 2:
        return None
    blocks = field.z1[:cy * b, :cx * b].reshape(cy, b, cx, b)
    zmin = blocks.min(axis=(1, 3))
    under = np.isfinite(blocks).all(axis=(1, 3))
    # The top reaches the HIGHEST underside point of its cell (+ overlap), so no gap is left where the
    # underside slopes (a Pikachu's feet: 0.3-1 mm gaps when capped two cells above the lowest), but stays
    # below the top of that first solid span. Points of the cell more than `max_gap` above its lowest are not
    # underside the base fills (a cell on the footprint's edge that "sees" the chin far up).
    low = np.isfinite(blocks) & (blocks <= zmin[:, None, :, None] + max_gap)
    zmax = np.where(under, np.where(low, blocks, -np.inf).max(axis=(1, 3)), np.inf)
    span_top = field.z2[:cy * b, :cx * b].reshape(cy, b, cx, b).min(axis=(1, 3))
    step = field.step * b
    xs = field.x0 + (np.arange(cx) * b + (b - 1) / 2) * field.step
    ys = field.y0 + (np.arange(cy) * b + (b - 1) / 2) * field.step
    gx, gy = np.meshgrid(xs, ys)
    inside = region(gx, gy, step)
    mask = ndimage.binary_opening(under & inside & (zmin <= max_gap))
    if not mask.any():
        return None
    if connected:
        labels, n = ndimage.label(mask)
        touching = set(np.unique(labels[mask & (zmin <= overlap)])) - {0}
        if not touching:
            sizes = ndimage.sum(mask, labels, range(1, n + 1))
            touching = {int(np.argmax(sizes)) + 1}
        mask = np.isin(labels, list(touching))
    if not (mask & (zmin > overlap * 0.25)).any():
        return None                         # nothing to fill: the figure already sits flat here
    top = np.where(mask, np.minimum(zmax + overlap, np.maximum(span_top - overlap, zmax)), -np.inf)
    # Corner (i, j) is shared by cells (i-1..i, j-1..j); its height is the highest of the filled ones, so the
    # top meets the underside across every cell (it may reach further into the figure: the slicer unions it).
    pad = np.pad(top, 1, constant_values=-np.inf)
    corner = np.maximum.reduce([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]])
    used = np.isfinite(corner)
    idx = -np.ones(corner.shape, np.int64)
    idx[used] = np.arange(int(used.sum()))
    ci, cj = np.nonzero(used)
    px = field.x0 + (cj * b - 0.5) * field.step
    py = field.y0 + (ci * b - 0.5) * field.step
    n_c = len(ci)
    faces: list[list[int]] = []
    walls: list[tuple[int, int]] = []
    for i, j in zip(*np.nonzero(mask)):
        a, bb, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]   # CCW from above
        faces += [[a, bb, c], [a, c, d], [a + n_c, c + n_c, bb + n_c], [a + n_c, d + n_c, c + n_c]]
        for (ni, nj), (e0, e1) in (((i - 1, j), (a, bb)), ((i, j + 1), (bb, c)), ((i + 1, j), (c, d)),
                                   ((i, j - 1), (d, a))):
            if not (0 <= ni < cy and 0 <= nj < cx and mask[ni, nj]):
                faces += [[e0 + n_c, e1 + n_c, e1], [e0 + n_c, e1, e0]]       # outward wall
                walls.append((int(e0), int(e1)))
    px, py = smooth_outline(px, py, walls)
    verts = np.vstack([np.column_stack([px, py, corner[used]]), np.column_stack([px, py, np.zeros(n_c)])])
    import trimesh
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    mesh.fix_normals()
    # Cells meeting only at a corner share a wall edge four ways (non-manifold): separate them (repair.py).
    v, f = repaired(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))
    return Part(FILL_NAME, extruder, v, f, {}, "normal_part")
