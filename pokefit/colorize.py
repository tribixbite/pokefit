"""Colorize: paint parts of a figure that has no (or too few) colours, picked by geometry.

A paint operation names one triangle (part index + face index in the source
object, so it survives rotation and scaling) and a filament. It paints the
triangles around it that its mode selects:

  * "part" (default): spread over the surface, stopping only at concave
    valleys sharper than `angle` degrees: where a tail, ear or paw meets the
    body, the rim of an eye socket. Parts of a sculpted figure meet along
    such valleys (the "minima rule" of shape segmentation); convex edges (an
    ear tip, a fin) do not stop it. On a smooth 10k-triangle Eevee, 15–20°
    gives ears, tail, legs, bib and eyes as separate parts where plain
    creases give four lumps;
  * "region": stop at any crease, convex or concave, sharper than `angle`:
    the same rule as Bambu Studio's smart fill, for hard-surface models;
  * "piece": every triangle connected to it (a separate shell: loose eyes,
    a hat modelled as its own body);
  * "brush": the connected surface within `radius` mm (file units) of the
    triangle: spots and stripes on a smooth sculpt that has no edge to follow;
  * "face": that triangle only.

Filament 0 removes paint (the part's own filament prints there). The result
is written as Bambu/Prusa per-triangle paint (`paint_color`, whole-triangle
states), so it opens in Bambu Studio as ordinary painted colour and survives
the pocket cut (`remesh_part` carries per-triangle attributes over).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .model3mf import Part

MODES = ("part", "region", "piece", "brush", "face")
DEFAULT_ANGLE = 15.0        # degrees; "part" mode (Bambu's smart fill, the "region" rule, defaults to 30)
_PAINT_RE = re.compile(r'\s*paint_color="[^"]*"')


@dataclass(frozen=True)
class PaintOp:
    part: int                 # index into the object's parts
    face: int                 # triangle index in that part
    filament: int             # 1-based filament number; 0 = remove paint
    mode: str = "part"
    angle: float = DEFAULT_ANGLE
    radius: float = 0.0       # "brush": mm in the file's units (the UI converts from print mm)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaintOp":
        op = cls(int(d["part"]), int(d["face"]), int(d["filament"]), str(d.get("mode", "part")),
                 float(d.get("angle", DEFAULT_ANGLE)), float(d.get("radius", 0.0)))
        if op.mode not in MODES:
            raise ValueError(f"paint mode must be one of {', '.join(MODES)}, not {op.mode!r}")
        if not (0 <= op.filament <= 64):
            raise ValueError(f"filament must be 0–64, not {op.filament}")
        if not (0.0 < op.angle <= 180.0):
            raise ValueError(f"angle must be in (0, 180], not {op.angle}")
        if op.mode == "brush" and not (op.radius > 0):
            raise ValueError("a brush needs a radius > 0")
        return op

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Topology:
    """Which triangles share an edge, how sharply the surface bends there, and which way."""
    n_faces: int
    pairs: np.ndarray          # (k, 2) adjacent face indices
    angles: np.ndarray         # (k,) degrees between the two face normals (0 = flat)
    convex: np.ndarray         # (k,) the edge bends outward (a ridge), not into a valley
    centroids: np.ndarray      # (n, 3) triangle centres, file mm

    def labels(self, mode: str, angle: float) -> np.ndarray:
        """Region id per triangle for a seed-independent mode (all but "brush")."""
        if mode == "face":
            return np.arange(self.n_faces)
        if mode == "piece":
            keep = np.ones(len(self.pairs), bool)
        elif mode == "part":
            keep = self.convex | (self.angles < angle)
        else:
            keep = self.angles < angle
        return self._components(keep)

    def brush(self, face: int, radius: float) -> np.ndarray:
        """Faces within `radius` of `face`'s centre and connected to it inside that ball."""
        near = np.linalg.norm(self.centroids - self.centroids[face], axis=1) <= radius
        near[face] = True
        lab = self._components(near[self.pairs[:, 0]] & near[self.pairs[:, 1]])
        return np.nonzero(lab == lab[face])[0]

    def _components(self, keep: np.ndarray) -> np.ndarray:
        p = self.pairs[keep]
        g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(self.n_faces, self.n_faces))
        _, lab = connected_components(g, directed=False)
        return np.asarray(lab)


def topology(part: Part) -> Topology:
    """Edge adjacency of a part. Vertices are merged by position first (STL soups repeat them)."""
    m = trimesh.Trimesh(part.vertices, part.faces, process=False)
    m.merge_vertices()          # keeps the face count and order
    pairs = np.asarray(m.face_adjacency, np.int64).reshape(-1, 2)
    if len(pairs):
        angles = np.degrees(np.asarray(m.face_adjacency_angles, np.float64))
        convex = np.asarray(m.face_adjacency_convex, bool)
    else:
        angles, convex = np.zeros(0), np.zeros(0, bool)
    return Topology(len(part.faces), pairs, angles, convex, np.asarray(m.triangles_center, np.float64))


def paint_code(filament: int) -> str:
    """TriangleSelector string of a whole triangle in one state (see paint.py): nibbles read from the end."""
    from .paint import _leaf_nibbles

    return "".join(f"{n:X}" for n in reversed(_leaf_nibbles(filament)))


def _with_paint(attrs: str | None, code: str | None) -> str | None:
    """A triangle's extra attributes with its paint replaced (None = paint removed)."""
    rest = _PAINT_RE.sub("", attrs or "").strip()
    if code is not None:
        rest = (rest + f' paint_color="{code}"').strip()
    return rest or None


def apply_paint(parts: list[Part], ops: list[PaintOp], topo: "TopologyCache") -> list[Part]:
    """Copies of `parts` with the paint operations applied in order (later ones win)."""
    extras = [dict(p.tri_extra) for p in parts]
    for op in ops:
        if not (0 <= op.part < len(parts)) or not (0 <= op.face < len(parts[op.part].faces)):
            continue                       # from another object or orientation of the file: ignore
        faces = region_faces(topo, op.part, op.face, op.mode, op.angle, op.radius)
        code = paint_code(op.filament) if op.filament else None
        ex = extras[op.part]
        for f in faces.tolist():
            v = _with_paint(ex.get(f), code)
            if v is None:
                ex.pop(f, None)
            else:
                ex[f] = v
    return [Part(p.name, p.extruder, p.vertices, p.faces, ex, p.subtype) for p, ex in zip(parts, extras)]


class TopologyCache:
    """Topology and region labels of one object's parts, computed on first use."""

    def __init__(self, parts: list[Part]):
        self._parts = parts
        self._topo: dict[int, Topology] = {}
        self._labels: dict[tuple[int, str, float], np.ndarray] = {}

    def part(self, i: int) -> Topology:
        if i not in self._topo:
            self._topo[i] = topology(self._parts[i])
        return self._topo[i]

    def labels(self, i: int, mode: str, angle: float) -> np.ndarray:
        key = (i, mode, round(float(angle), 3))
        if key not in self._labels:
            self._labels[key] = self.part(i).labels(mode, angle)
        return self._labels[key]


def region_faces(topo: TopologyCache, part: int, face: int, mode: str, angle: float,
                 radius: float = 0.0) -> np.ndarray:
    if mode == "brush":
        return topo.part(part).brush(face, radius)
    lab = topo.labels(part, mode, angle)
    return np.nonzero(lab == lab[face])[0]


def pick_face(parts: list[Part], point: np.ndarray, normal: np.ndarray | None = None,
              candidates: int = 24) -> tuple[int, int, float]:
    """(part index, face index, distance) of the triangle nearest `point`.

    Exact point–triangle distances over the triangles with the nearest
    centroids (pure numpy + a KD-tree, so it runs in the browser too). With
    `normal` (the view's surface normal), triangles facing away are skipped, so
    a tap on a thin ear picks its near side. Ties go to the earlier part, which
    is the one the slicer prints where parts overlap.
    """
    from scipy.spatial import cKDTree

    best = (-1, -1, np.inf)
    p = np.asarray(point, np.float64)
    for pi, part in enumerate(parts):
        if not len(part.faces) or part.subtype != "normal_part":
            continue
        tri = part.vertices[part.faces]
        k = min(candidates, len(tri))
        _, idx = cKDTree(tri.mean(axis=1)).query(p, k=k)
        idx = np.atleast_1d(idx)
        if normal is not None:
            n = np.cross(tri[idx, 1] - tri[idx, 0], tri[idx, 2] - tri[idx, 0])
            facing = (n @ np.asarray(normal, np.float64)) > 0
            if facing.any():
                idx = idx[facing]
        cp = trimesh.triangles.closest_point(tri[idx], np.repeat(p[None], len(idx), axis=0))
        d = np.linalg.norm(cp - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < best[2] - 1e-9:
            best = (pi, int(idx[j]), float(d[j]))
    return best


def region_colours(n: int, seed: int = 7) -> np.ndarray:
    """Distinct pastel RGB (0–255) per region id, for the "show regions" preview (golden-ratio hues)."""
    import colorsys

    start = float(np.random.default_rng(seed).random())
    rgb = [colorsys.hsv_to_rgb((start + k * 0.618033988749895) % 1.0, 0.45, 0.92) for k in range(n)]
    return (np.asarray(rgb, np.float64).reshape(-1, 3) * 255).astype(np.uint8)
