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

import numpy as np

from .ball import Cavity
from .fit import max_scale, prepare_points
from .magnets import HeightField, height_field
from .model3mf import Part

BAND = 0.30          # lowest fraction of the height that a bend lifts fully
FADE = 0.20          # ...fading to no lift over this fraction above it
EASE = 3.0           # mm over which the lift's slope eases in past the disc edge
MAX_SLOPE = np.tan(np.radians(40.0))
LIMB_TOP = 0.5       # a low part whose first solid span ends below this fraction of the height is a limb
MIN_GAIN = 0.01      # auto bend is kept only if the figure fits at least 1 % larger
ROUNDS = 3
FLAT_TARGET = 0.6    # share of the footprint over the disc the flat base aims to touch
MIN_DEPTH = 0.02     # mm; shallower presses are skipped (the base is flat already)
DEFAULT_MAX_DEPTH = 1.2
DEFAULT_FILL = 2.5   # mm: largest gap under the figure the flat base fills
FILL_CELL = 0.4      # mm: grid of the fill solid
FILL_OVERLAP = 0.4   # mm the fill reaches into the figure (the slicer unions overlapping parts)
FILL_NAME = "flat base"


@dataclass
class ConformSpec:
    bend: float | None = None       # None = automatic, 0 = off, else strength 0..1
    flat_diameter: float = 18.0     # mm, the disc pressed flat (default: the washer); 0 = off
    max_depth: float = DEFAULT_MAX_DEPTH  # mm, deepest press for the flat base
    fill: float = DEFAULT_FILL      # mm, largest gap under the figure filled by the flat base (0 = off)
    scale: float | None = None      # print scale to design for (None = the largest that fits)


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


def _fit(v: np.ndarray, cavity: Cavity) -> tuple[float, np.ndarray] | None:
    """(largest scale, figure-frame XY of the ball axis) — no pockets."""
    res = max_scale(prepare_points(v), cavity)
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
           spec: ConformSpec, height: float) -> _Shaped:
    """Bend (strength k) and press the disc flat, designed for print scale s with the ball axis at `axis`.

    The press lowers only the disc: parts beyond it (inside the bend band) are raised by the
    same depth first, so pressing cannot push feet or a tail into the sphere's slope.
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
    t, share = flat_depth(_field(v, parts), axis, r_flat / s, spec.max_depth / s)
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
    info.limb_share = limb_share(_field(v0, parts), first[1], max(r_flat, 1e-6) / s0, height)
    k = spec.bend if spec.bend is not None else info.limb_share * float(
        np.clip(2 * (low_aspect(v0, height) - 0.5), 0.0, 1.0))
    k = float(np.clip(k, 0.0, 1.0))

    def solve(strength: float) -> tuple[_Shaped, float]:
        """Shape, refitting until the design scale settles; returns (shape, its largest fit scale)."""
        s, axis = s0, first[1]
        shaped = _shape(v0, parts, axis, s, strength, cavity, spec, height)
        got = _fit(shaped.v, cavity)
        for _ in range(ROUNDS if strength > 0 else 0):
            if got is None:
                break
            settled = abs(got[0] - s) / s < 0.005
            s, axis = (s0 if spec.scale else got[0]), got[1]
            if settled or spec.scale:
                break
            shaped = _shape(v0, parts, axis, s, strength, cavity, spec, height)
            got = _fit(shaped.v, cavity)
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
    if r_flat > 0 and spec.fill > 0 and s_final > 0:
        base = base_fill(out, best.axis, r_flat / s_final, spec.fill / s_final, FILL_CELL / s_final,
                         FILL_OVERLAP / s_final, parts[0].extruder)
        if base is not None:
            out = [*out, base]
            info.fill_mm2 = float(base.mesh().area_faces[base.mesh().face_normals[:, 2] < -0.9].sum()) * s_final ** 2
    return out, info


def base_fill(parts: list[Part], axis: np.ndarray, radius: float, max_gap: float, cell: float, overlap: float,
              extruder: int) -> Part | None:
    """A solid filling the gap between the bed and the figure's underside over the disc.

    Built on a coarse grid (`cell`) from the underside height field: flat bottom at z = 0, top at
    the underside (the lowest point in each cell) + `overlap`, vertical walls round its outline.
    Only cells where the gap is at most `max_gap`, connected to where the figure already touches
    the bed (or the largest patch), and fully under the figure are filled. Unit-scale frame.
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
    # The top reaches the HIGHEST underside point of its cell (+ overlap), so no hairline gap is left
    # where the underside slopes, but stays below the top of that first solid span.
    zmax = np.where(under, np.where(np.isfinite(blocks), blocks, -np.inf).max(axis=(1, 3)), np.inf)
    span_top = field.z2[:cy * b, :cx * b].reshape(cy, b, cx, b).min(axis=(1, 3))
    step = field.step * b
    xs = field.x0 + (np.arange(cx) * b + (b - 1) / 2) * field.step
    ys = field.y0 + (np.arange(cy) * b + (b - 1) / 2) * field.step
    gx, gy = np.meshgrid(xs, ys)
    inside = np.hypot(gx - axis[0], gy - axis[1]) <= radius - step / 2
    mask = ndimage.binary_opening(under & inside & (zmin <= max_gap))
    if not mask.any():
        return None
    labels, n = ndimage.label(mask)
    touching = set(np.unique(labels[mask & (zmin <= overlap)])) - {0}
    if not touching:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        touching = {int(np.argmax(sizes)) + 1}
    mask = np.isin(labels, list(touching))
    if not (mask & (zmin > overlap * 0.25)).any():
        return None                         # nothing to fill: the figure already sits flat here
    # Highest point, but never more than a steep slope (2 cells) above the lowest: a cell on the edge
    # of the footprint can "see" something far up (the chin), which would raise a spike to it.
    reach = np.minimum(zmax, zmin + 2 * step)
    top = np.where(mask, np.minimum(reach + overlap, np.maximum(span_top - overlap, reach)), np.inf)
    # Corner (i, j) is shared by cells (i-1..i, j-1..j); its height is the lowest of theirs (a gap of
    # at most the slope across one cell; the overlap covers it).
    pad = np.pad(top, 1, constant_values=np.inf)
    corner = np.minimum.reduce([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]])
    used = np.isfinite(corner)
    idx = -np.ones(corner.shape, np.int64)
    idx[used] = np.arange(int(used.sum()))
    ci, cj = np.nonzero(used)
    px = field.x0 + (cj * b - 0.5) * field.step
    py = field.y0 + (ci * b - 0.5) * field.step
    n_c = len(ci)
    verts = np.vstack([np.column_stack([px, py, corner[used]]), np.column_stack([px, py, np.zeros(n_c)])])
    faces: list[list[int]] = []
    for i, j in zip(*np.nonzero(mask)):
        a, bb, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]   # CCW from above
        faces += [[a, bb, c], [a, c, d], [a + n_c, c + n_c, bb + n_c], [a + n_c, d + n_c, c + n_c]]
        for (ni, nj), (e0, e1) in (((i - 1, j), (a, bb)), ((i, j + 1), (bb, c)), ((i + 1, j), (c, d)),
                                   ((i, j - 1), (d, a))):
            if not (0 <= ni < cy and 0 <= nj < cx and mask[ni, nj]):
                faces += [[e0 + n_c, e1 + n_c, e1], [e0 + n_c, e1, e0]]       # outward wall
    import trimesh
    mesh = trimesh.Trimesh(verts, np.array(faces), process=True)
    mesh.fix_normals()
    return Part(FILL_NAME, extruder, np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), {},
                "normal_part")
