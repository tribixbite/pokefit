"""Reshape a limb: resize or bend one part of the figure (a tail, an ear, a leg), tapped on the 3D view.

The limb is found from the shape like paint's "part" mode (`colorize.region_faces`): the surface up to the valley
where it meets the body, across convex edges, stopping at concave ones sharper than `angle`. Where its triangles
meet the rest of the part is the joint; the pivot is the joint's middle (a separate piece, with no joint, turns
and grows about its own middle).

Only vertices move, faces keep their order, so paint, a glTF's texture colour and the tap-to-paint topology carry
over exactly, as in the conform step. Every limb vertex moves by a weight w that is 0 on the joint and eases up to
1 at BLEND of the limb's length from it (walking along the surface), so the body is untouched and the limb bends
in a smooth curve instead of a kink:

    v' = pivot + R(swing * w) R(bend * w) ((v - pivot) * (1 + (scale - 1) * w))

Directions are the figure's (after its rotation; z up, the front at -y): `bend` > 0 lifts the tip (a limb pointing
up or down tips toward the front instead), `swing` turns it round the axis at right angles to both (for a limb lying
flat: round the vertical, counter-clockwise seen from above). Nothing stops a bent limb running into the body:
the view shows it, and a larger joint angle, a smaller bend or a size change avoids it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .colorize import DEFAULT_ANGLE, TopologyCache, region_faces
from .model3mf import Part

BLEND = 0.35        # share of the limb's length (from the joint, along the surface) over which a change eases in
MAX_SHARE = 0.4     # a "limb" with more of its part's surface (area) than this is the body: refused (tails, ears, legs: 2-25 %)
SCALE_RANGE = (0.3, 3.0)
MAX_ANGLE = 150.0   # degrees, |bend| and |swing|
UPRIGHT = 0.8       # a limb whose direction is within ~37 degrees of vertical bends toward the front, not up
MAX_LIMBS = 24
# A tap that finds the body at the angle asked for tries these smaller ones (sculpts have shallow joints: a
# Charmander's tail is 5-7 % of it from 2 to 8 degrees, the whole body from 15).
FALLBACK_ANGLES = (10.0, 8.0, 6.0, 4.0, 3.0, 2.0)


@dataclass(frozen=True)
class LimbOp:
    """One reshaped limb: the triangle tapped (`part`, `face`, as paint operations name them) and the change."""
    part: int
    face: int
    angle: float = DEFAULT_ANGLE   # the "part" search's crease angle (smaller: shallower valleys end a limb too)
    scale: float = 1.0             # size about the joint
    bend: float = 0.0              # degrees, tip up (> 0) or down
    swing: float = 0.0             # degrees, sideways

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LimbOp":
        op = cls(int(d["part"]), int(d["face"]), float(d.get("angle", DEFAULT_ANGLE)), float(d.get("scale", 1.0)),
                 float(d.get("bend", 0.0)), float(d.get("swing", 0.0)))
        if op.part < 0 or op.face < 0:
            raise ValueError("a limb's part and face must be >= 0")
        if not 0.0 < op.angle <= 180.0:
            raise ValueError(f"a limb's joint angle must be in (0, 180], not {op.angle}")
        if not SCALE_RANGE[0] <= op.scale <= SCALE_RANGE[1]:
            raise ValueError(f"a limb's size must be {SCALE_RANGE[0]:g}..{SCALE_RANGE[1]:g}, not {op.scale}")
        if not (abs(op.bend) <= MAX_ANGLE and abs(op.swing) <= MAX_ANGLE):
            raise ValueError(f"a limb's bend and swing must be within ±{MAX_ANGLE:g} degrees")
        return op

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def changes(self) -> bool:
        return self.scale != 1.0 or self.bend != 0.0 or self.swing != 0.0


@dataclass
class Limb:
    """A limb's surface on one part, in the topology's merged vertices."""
    part: int
    faces: np.ndarray      # its triangles
    verts: np.ndarray      # its merged vertices
    weight: np.ndarray     # per entry of `verts`: 0 on the joint, easing to 1 at BLEND of its length
    joint: np.ndarray      # merged vertices it shares with the rest of the part (empty: a separate piece)
    length: float          # longest walk from the joint along it, file mm (0: a separate piece)


def find_limb(topo: TopologyCache, part: int, face: int, angle: float = DEFAULT_ANGLE) -> Limb:
    """The limb at triangle `face` of `part` (module doc). Raises ValueError when the tap found the body (most
    of the part) instead."""
    from scipy.sparse.csgraph import dijkstra

    if not 0 <= part < len(topo) or not 0 <= face < topo.n_faces(part):
        raise ValueError("no such triangle")
    t = topo.part(part)
    faces = np.asarray(region_faces(topo, part, face, "part", angle), np.int64)
    if float(t.areas[faces].sum()) > MAX_SHARE * float(t.areas.sum()):
        raise ValueError("That takes in too much of the figure to be a limb: tap a tail, ear, leg or arm where it meets the "
                         "body in a valley. If it still takes the body, lower the Joint angle (more valleys then count as joints).")
    inside = np.zeros(t.n_faces, bool)
    inside[faces] = True
    verts = np.unique(t.vfaces[faces])
    joint = np.intersect1d(verts, np.unique(t.vfaces[~inside]), assume_unique=True)
    if not len(joint):
        return Limb(part, faces, verts, np.ones(len(verts)), joint, 0.0)
    graph = t.vertex_graph()[verts][:, verts]
    starts = np.searchsorted(verts, joint)
    dist = np.asarray(dijkstra(graph, indices=starts, min_only=True))
    reach = np.isfinite(dist)
    length = float(dist[reach].max()) if reach.any() else 0.0
    if length <= 0:
        return Limb(part, faces, verts, np.zeros(len(verts)), joint, 0.0)
    u = np.clip(np.where(reach, dist, length) / (BLEND * length), 0.0, 1.0)
    return Limb(part, faces, verts, u * u * (3 - 2 * u), joint, length)


def find_limb_at(topo: TopologyCache, part: int, face: int, angle: float = DEFAULT_ANGLE) -> tuple[Limb, float]:
    """`find_limb` at `angle`, or, when that finds the body, at the first smaller FALLBACK_ANGLES that finds a limb;
    returns it with the angle used (a `LimbOp` stores that angle, so the change replays exactly)."""
    try:
        return find_limb(topo, part, face, angle), angle
    except ValueError:
        for a in FALLBACK_ANGLES:
            if a < angle:
                try:
                    return find_limb(topo, part, face, a), a
                except ValueError:
                    continue
        raise


def _merged_index(part: Part, topo_vfaces: np.ndarray, n_merged: int) -> np.ndarray:
    """Per vertex of `part`, its merged vertex in the topology (-1 for one no triangle uses). The parts keep the
    file's triangles in order and their corners in order (only proper rotations and shifts are applied)."""
    m = np.full(len(part.vertices), -1, np.int64)
    m[part.faces.ravel()] = topo_vfaces.ravel()
    return np.where(m < n_merged, m, -1)


def _rotate(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues: each row of `v` turned about unit `axis` by its own `angle` (radians)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(axis, v) * s + axis * (v @ axis)[:, None] * (1 - c)


def limb_axes(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bend axis, swing axis) for a limb pointing along `direction` (figure frame): turning about the bend axis
    by a positive angle moves the tip up (an upright limb: to the front, -y); the swing axis is at right angles to
    the limb and the bend axis."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    ref = np.array([0.0, -1.0, 0.0]) if abs(d[2]) > UPRIGHT else np.array([0.0, 0.0, 1.0])
    bend = np.cross(d, ref)
    bend /= np.linalg.norm(bend) + 1e-12
    swing = np.cross(bend, d)
    swing /= np.linalg.norm(swing) + 1e-12
    return bend, swing


def apply_limbs(parts: list[Part], ops: list[LimbOp], topo: TopologyCache) -> list[Part]:
    """`parts` (figure frame, the same triangles as `topo`'s parts) with each limb resized and bent, in order (a
    later one sees the earlier ones' result). Operations on parts that are not there are skipped."""
    out = list(parts)
    for op in ops:
        if not op.changes or op.part >= len(out) or op.part >= len(topo):
            continue
        limb = find_limb(topo, op.part, op.face, op.angle)
        t = topo.part(op.part)
        p = out[op.part]
        m = _merged_index(p, t.vfaces, len(t.verts))
        pos = np.zeros((len(t.verts), 3))
        used = m >= 0
        pos[m[used]] = p.vertices[used]
        tri = pos[t.vfaces[limb.faces]]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        centre = (tri.mean(axis=1) * area[:, None]).sum(axis=0) / max(float(area.sum()), 1e-12)
        pivot = pos[limb.joint].mean(axis=0) if len(limb.joint) else centre
        direction = centre - pivot
        if np.linalg.norm(direction) < 1e-9:
            far = pos[limb.verts][np.argmax(np.linalg.norm(pos[limb.verts] - pivot, axis=1))]
            direction = far - pivot if np.linalg.norm(far - pivot) > 1e-9 else np.array([1.0, 0.0, 0.0])
        bend_axis, swing_axis = limb_axes(direction)
        w = limb.weight
        rel = (pos[limb.verts] - pivot) * (1 + (op.scale - 1) * w)[:, None]
        rel = _rotate(rel, bend_axis, np.radians(op.bend) * w)
        rel = _rotate(rel, swing_axis, np.radians(op.swing) * w)
        moved = pos.copy()
        moved[limb.verts] = pivot + rel
        inside = np.zeros(len(t.verts), bool)
        inside[limb.verts] = True
        hit = used.copy()
        hit[used] = inside[m[used]]
        v = p.vertices.copy()
        v[hit] = moved[m[hit]]
        out[op.part] = Part(p.name, p.extruder, v, p.faces, p.tri_extra, p.subtype)
    return out


def limb_ops(raw: list[dict[str, Any]] | None) -> list[LimbOp]:
    """`Params.limbs` as operations (validated)."""
    ops = [LimbOp.from_dict(d) for d in raw or []]
    if len(ops) > MAX_LIMBS:
        raise ValueError(f"at most {MAX_LIMBS} reshaped limbs")
    return ops
