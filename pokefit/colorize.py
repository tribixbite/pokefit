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
    triangle, or of any triangle on its `path` (a stroke dragged across the
    figure): spots, stripes and outlines on a smooth sculpt that has no edge
    to follow;
  * "face": that triangle only.

"part" and "region" fills also stop where the visible colour changes (bucket
fill): a sculpt whose eyes or belly have no crease all the way round (a
generated or smoothed figure) gets them by outlining with a brush stroke and
tapping inside.

Filament 0 removes paint (the part's own filament prints there). The result
is written as Bambu/Prusa per-triangle paint (`paint_color`, whole-triangle
states), so it opens in Bambu Studio as ordinary painted colour and survives
the pocket cut (`remesh_part` carries per-triangle attributes over).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .model3mf import Part

MODES = ("part", "region", "piece", "brush", "face")
DEFAULT_ANGLE = 15.0        # degrees; "part" mode (Bambu's smart fill, the "region" rule, defaults to 30)
MAX_PATH = 4000             # triangles on one brush stroke's path
JOIN_STEPS = 8              # a stroke step up to this many brush radii long is filled in; longer ones jumped
BOUNDED = ("part", "region")  # fills that stop where the colour changes
_FILL_CACHE = 64            # colour-bounded fills kept per object (keyed by the paint state they saw)
_PAINT_RE = re.compile(r'\s*paint_color="[^"]*"')


@dataclass(frozen=True)
class PaintOp:
    part: int                 # index into the object's parts
    face: int                 # triangle index in that part
    filament: int             # 1-based filament number; 0 = remove paint
    mode: str = "part"
    angle: float = DEFAULT_ANGLE
    radius: float = 0.0       # "brush": mm in the file's units (the UI converts from print mm)
    path: tuple[int, ...] = ()  # "brush": more triangles of the same part along a dragged stroke

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaintOp":
        op = cls(int(d["part"]), int(d["face"]), int(d["filament"]), str(d.get("mode", "part")),
                 float(d.get("angle", DEFAULT_ANGLE)), float(d.get("radius", 0.0)),
                 tuple(int(f) for f in d.get("path") or ()))
        if op.mode not in MODES:
            raise ValueError(f"paint mode must be one of {', '.join(MODES)}, not {op.mode!r}")
        if not (0 <= op.filament <= 64):
            raise ValueError(f"filament must be 0–64, not {op.filament}")
        if not (0.0 < op.angle <= 180.0):
            raise ValueError(f"angle must be in (0, 180], not {op.angle}")
        if op.mode == "brush" and not (op.radius > 0):
            raise ValueError("a brush needs a radius > 0")
        if op.path and op.mode != "brush":
            raise ValueError("only a brush stroke has a path")
        if len(op.path) > MAX_PATH:
            raise ValueError(f"a stroke has at most {MAX_PATH} points")
        return op

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.path:
            d["path"] = list(self.path)
        else:
            del d["path"]
        return d


@dataclass
class Topology:
    """Which triangles share an edge, how sharply the surface bends there, and which way."""
    n_faces: int
    pairs: np.ndarray          # (k, 2) adjacent face indices
    angles: np.ndarray         # (k,) degrees between the two face normals (0 = flat)
    convex: np.ndarray         # (k,) the edge bends outward (a ridge), not into a valley
    centroids: np.ndarray      # (n, 3) triangle centres, file mm
    _tree: Any = None          # KD-tree of `centroids`, built by the first brush

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

    def brush(self, seeds: list[int], radius: float) -> np.ndarray:
        """Faces within `radius` of a seed's centre and connected to a seed inside those balls: one dab, or
        a stroke (a dab per point along it; a thin ridge between two dabs is not bridged)."""
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.centroids)
        seed = np.asarray(seeds, np.int64)
        centres = self.centroids[seed]
        if len(seed) > 1:
            # Join consecutive samples (a fast drag leaves gaps a fill would leak through): points along each
            # step at most half a radius apart. A step longer than JOIN_STEPS radii jumped to another surface
            # (the drag crossed a silhouette) and is not bridged.
            a, b = centres[:-1], centres[1:]
            n = np.ceil(np.linalg.norm(b - a, axis=1) / (radius * 0.5)).astype(np.int64)
            n[n > JOIN_STEPS * 2] = 1
            k = np.repeat(np.arange(len(a)), n.astype(np.intp))
            t = (np.arange(len(k)) - np.repeat(np.cumsum(n) - n, n.astype(np.intp))) / n[k]
            centres = np.vstack([a[k] + (b[k] - a[k]) * t[:, None], centres[-1:]])
        near = np.zeros(self.n_faces, bool)
        for idx in self._tree.query_ball_point(centres, r=radius):
            near[idx] = True
        near[seed] = True
        lab = self._components(near[self.pairs[:, 0]] & near[self.pairs[:, 1]])
        return np.nonzero(near & np.isin(lab, lab[seed]))[0]

    def bounded(self, region: np.ndarray, face: int, colour: np.ndarray) -> np.ndarray:
        """The part of `region` (a fill's faces) connected to `face` through faces of `face`'s colour."""
        inside = np.zeros(self.n_faces, bool)
        inside[region] = True
        inside &= colour == colour[face]
        lab = self._components(inside[self.pairs[:, 0]] & inside[self.pairs[:, 1]])
        return np.nonzero(inside & (lab == lab[face]))[0]

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


@lru_cache(maxsize=1 << 12)
def _with_paint(attrs: str | None, code: str | None) -> str | None:
    """A triangle's extra attributes with its paint replaced (None = paint removed)."""
    rest = _PAINT_RE.sub("", attrs or "").strip()
    if code is not None:
        rest = (rest + f' paint_color="{code}"').strip()
    return rest or None


class PaintState:
    """One part's per-triangle attributes as an index per triangle into a list of distinct strings, so a
    paint operation updates hundreds of thousands of triangles with a few array operations."""

    def __init__(self, part: Part):
        self.extruder = part.extruder
        self.values: list[str | None] = [None]
        self._ids: dict[str | None, int] = {None: 0}
        self.state = np.zeros(len(part.faces), np.int32)
        if part.tri_extra:
            idx = np.fromiter(part.tri_extra.keys(), np.int64, len(part.tri_extra))
            self.state[idx] = [self._id(v) for v in part.tri_extra.values()]
        self._colours: np.ndarray | None = None

    def _id(self, value: str | None) -> int:
        if value not in self._ids:
            self._ids[value] = len(self.values)
            self.values.append(value)
        return self._ids[value]

    def paint(self, faces: np.ndarray, code: str | None) -> None:
        """Give `faces` the paint `code` (None = remove it), keeping their other attributes."""
        uniq, inv = np.unique(self.state[faces], return_inverse=True)
        new = np.asarray([self._id(_with_paint(self.values[u], code)) for u in uniq.tolist()], np.int32)
        self.state[faces] = new[inv.reshape(-1)]
        self._colours = None

    def colours(self) -> np.ndarray:
        """The filament each triangle shows most (its dominant paint, else the part's own)."""
        if self._colours is None:
            from .paint import attr_paint

            per_value = [self.extruder]
            for v in self.values[1:]:
                fp = attr_paint(v or "")
                per_value.append(fp.majority if fp is not None and fp.majority > 0 else self.extruder)
            self._colours = np.asarray(per_value, np.int32)[self.state]
        return self._colours

    def key(self) -> str:
        """A digest of the per-triangle colours (what a colour-bounded fill depends on)."""
        import hashlib

        return hashlib.blake2b(self.colours().tobytes(), digest_size=16).hexdigest()

    def extra(self) -> dict[int, str]:
        """Back to `Part.tri_extra` form."""
        idx = np.nonzero(self.state)[0]
        vals = self.values
        return {f: v for f, s in zip(idx.tolist(), self.state[idx].tolist()) if (v := vals[s]) is not None}


def apply_paint(parts: list[Part], ops: list[PaintOp], topo: "TopologyCache") -> list[Part]:
    """Copies of `parts` with the paint operations applied in order (later ones win)."""
    states: dict[int, PaintState] = {}
    for op in ops:
        if not (0 <= op.part < min(len(parts), len(topo))) or not (0 <= op.face < len(parts[op.part].faces)):
            continue                       # from another object or orientation of the file: ignore
        n = len(parts[op.part].faces)
        path = tuple(f for f in op.path if 0 <= f < n)
        st = states.setdefault(op.part, PaintState(parts[op.part]))
        faces = region_faces(topo, op.part, op.face, op.mode, op.angle, op.radius, path, st)
        st.paint(faces, paint_code(op.filament) if op.filament else None)
    return [Part(p.name, p.extruder, p.vertices, p.faces, states[i].extra() if i in states else dict(p.tri_extra),
                 p.subtype) for i, p in enumerate(parts)]


class TopologyCache:
    """Topology and region labels of one object's parts, computed on first use."""

    def __init__(self, parts: list[Part]):
        self._parts = parts
        self._topo: dict[int, Topology] = {}
        self._labels: dict[tuple[int, str, float], np.ndarray] = {}
        self._fills: dict[tuple[Any, ...], np.ndarray] = {}

    def __len__(self) -> int:
        """Parts it covers: the file's own (not the conform step's flat base, which taps cannot paint)."""
        return len(self._parts)

    def part(self, i: int) -> Topology:
        if i not in self._topo:
            self._topo[i] = topology(self._parts[i])
        return self._topo[i]

    def labels(self, i: int, mode: str, angle: float) -> np.ndarray:
        key = (i, mode, round(float(angle), 3))
        if key not in self._labels:
            self._labels[key] = self.part(i).labels(mode, angle)
        return self._labels[key]


    def fill(self, i: int, face: int, mode: str, angle: float, paint: PaintState) -> np.ndarray:
        """A "part"/"region" fill that also stops where the colour changes (see `region_faces`)."""
        lab = self.labels(i, mode, angle)
        region = np.nonzero(lab == lab[face])[0]
        colour = paint.colours()
        if (colour[region] == colour[face]).all():
            return region                  # one colour throughout (the usual case): no extra work
        key = (i, face, mode, round(float(angle), 3), paint.key())
        if key not in self._fills:
            if len(self._fills) >= _FILL_CACHE:
                self._fills.pop(next(iter(self._fills)))
            self._fills[key] = self.part(i).bounded(region, face, colour)
        return self._fills[key]


def region_faces(topo: TopologyCache, part: int, face: int, mode: str, angle: float,
                 radius: float = 0.0, path: tuple[int, ...] = (), paint: PaintState | None = None) -> np.ndarray:
    """The faces a paint operation covers. With `paint` (the part's current paint), "part" and "region"
    fills stop where the visible colour changes, like a bucket fill."""
    if mode == "brush":
        return topo.part(part).brush([face, *path], radius)
    if paint is not None and mode in BOUNDED:
        return topo.fill(part, face, mode, angle, paint)
    lab = topo.labels(part, mode, angle)
    return np.nonzero(lab == lab[face])[0]


def centroid_trees(parts: list[Part]) -> list[Any]:
    """Per part, a KD-tree of its triangle centroids for `pick_face` (None for parts taps skip)."""
    from scipy.spatial import cKDTree

    return [cKDTree(p.vertices[p.faces].mean(axis=1)) if len(p.faces) and p.subtype == "normal_part" else None
            for p in parts]


def pick_face(parts: list[Part], point: np.ndarray, normal: np.ndarray | None = None,
              candidates: int = 24, trees: list[Any] | None = None) -> tuple[int, int, float]:
    """(part index, face index, distance) of the triangle nearest `point`.

    Exact point–triangle distances over the triangles with the nearest
    centroids (pure numpy + a KD-tree, so it runs in the browser too). With
    `normal` (the view's surface normal), triangles facing away are skipped, so
    a tap on a thin ear picks its near side. Ties go to the earlier part, which
    is the one the slicer prints where parts overlap. `trees` (`centroid_trees(parts)`) saves
    building them per tap (a 300k-triangle part: 0.2 s).
    """
    trees = trees if trees is not None else centroid_trees(parts)
    best = (-1, -1, np.inf)
    p = np.asarray(point, np.float64)
    for pi, (part, tree) in enumerate(zip(parts, trees)):
        if tree is None:
            continue
        k = min(candidates, len(part.faces))
        _, idx = tree.query(p, k=k)
        idx = np.atleast_1d(idx)
        tri = part.vertices[part.faces[idx]]
        if normal is not None:
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            facing = (n @ np.asarray(normal, np.float64)) > 0
            if facing.any():
                idx, tri = idx[facing], tri[facing]
        cp = trimesh.triangles.closest_point(tri, np.repeat(p[None], len(idx), axis=0))
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
