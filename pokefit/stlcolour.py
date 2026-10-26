"""Coloured STL: the exported figure as one binary STL with a colour per triangle.

STL has no standard colour. The common extension (VisCAM, SolidView; MeshLab and other viewers read it) keeps a
15-bit colour in each triangle's 2-byte "attribute byte count": bit 15 set = the colour is valid, bits 10-14 red,
5-9 green, 0-4 blue (5 bits each). Materialise Magics uses the same slot with the channels and the valid bit the
other way round plus a `COLOR=` header; it is not written here, since one file cannot carry both.

Soft paint edges (Bambu split paint, `paint.leaves`) are written as their sub-triangles, so a colour boundary
follows the stroke instead of the mesh's triangles. Splitting a triangle puts new corners on its sides; the
neighbour across such a side would then meet them in the middle of its own edge (a T-junction: open edges to a
slicer). Every triangle with such a point on a side is cut into a fan round its centre through those points, so
the file stays closed wherever the mesh was.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model3mf import Part
from .paint import attr_paint, leaves, parse_tree

HEADER = b"pokefit coloured STL: VisCAM/SolidView 15-bit RGB per triangle"
_EYE = np.eye(3)
_STL = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attr", "<u2")])

# A point of the output, named so that two triangles naming it get the very same coordinates:
#   ("v", g)          an original vertex
#   ("e", lo, hi, t)  a point at t along the original edge lo -> hi (lo < hi, vertex ids)
#   ("f", face, b)    a point inside an original face, barycentric b (only that face's sub-triangles share it)
Point = tuple


def _name(face: int, corners: np.ndarray, b: np.ndarray) -> Point:
    """The name of the point at barycentric `b` of the face with vertex ids `corners`."""
    zero = np.flatnonzero(b == 0.0)
    if len(zero) == 2:
        return ("v", int(corners[int(np.flatnonzero(b)[0])]))
    if len(zero) == 1:
        j, k = (int(zero[0]) + 1) % 3, (int(zero[0]) + 2) % 3
        lo, hi = (j, k) if corners[j] < corners[k] else (k, j)
        return ("e", int(corners[lo]), int(corners[hi]), float(b[hi]))
    return ("f", face, (float(b[0]), float(b[1]), float(b[2])))


def _between(p: Point, q: Point, on_edge: dict[tuple[int, int], list[float]]) -> list[Point]:
    """Points other triangles put strictly inside the segment p -> q (both on one original edge), from p to q."""
    def at(x: Point, lo: int, hi: int) -> float | None:
        if x[0] == "v":
            return 0.0 if x[1] == lo else 1.0 if x[1] == hi else None
        if x[0] == "e" and (x[1], x[2]) == (lo, hi):
            return float(x[3])
        return None

    ids = {x[1] for x in (p, q) if x[0] == "v"} | {y for x in (p, q) if x[0] == "e" for y in x[1:3]}
    if len(ids) != 2 or p[0] == "f" or q[0] == "f":
        return []
    lo, hi = sorted(ids)
    ts = on_edge.get((lo, hi))
    if not ts:
        return []
    tp, tq = at(p, lo, hi), at(q, lo, hi)
    if tp is None or tq is None:
        return []
    inner = [t for t in ts if min(tp, tq) < t < max(tp, tq)]
    inner.sort(reverse=tp > tq)
    return [("e", lo, hi, t) for t in inner]


def coloured_triangles(part: Part, default_extruder: int) -> tuple[np.ndarray, np.ndarray]:
    """(triangles (k, 3, 3), filament per triangle) of a part: split paint as its sub-triangles, every triangle
    meeting a split point mid-side fanned through it (no T-junctions)."""
    v = np.asarray(part.vertices, np.float64)
    f = np.asarray(part.faces, np.int64)
    base = part.extruder or default_extruder
    ext = np.full(len(f), base, np.int64)
    split: dict[int, str] = {}
    for i, attrs in part.tri_extra.items():
        fp = attr_paint(attrs)
        if fp is None:
            continue
        if fp.whole is None:
            split[i] = fp.code
        elif fp.whole > 0:
            ext[i] = fp.whole
    if not split:
        return v[f], ext

    # Sub-triangles of every split face, their corners named, and the points they add on original edges. Inside a
    # face, leaves split to different depths meet the same way (a small leaf's corner halfway along a big leaf's
    # side): `inner[k]` lists the face's own points on side k of a leaf, in order.
    pieces: list[tuple[list[Point], int, list[list[Point]]]] = []
    on_edge: dict[tuple[int, int], set[float]] = {}
    for i, code in split.items():
        lv = leaves(parse_tree(code), [_EYE[0], _EYE[1], _EYE[2]])
        pts = np.unique(np.asarray([b for bary, _ in lv for b in bary], np.float64), axis=0)
        for bary, state in lv:
            corners = [np.asarray(b, np.float64) for b in bary]
            names = [_name(i, f[i], b) for b in corners]
            for n in names:
                if n[0] == "e":
                    on_edge.setdefault((n[1], n[2]), set()).add(n[3])
            inner: list[list[Point]] = []
            for k in range(3):
                a, b = corners[k], corners[(k + 1) % 3]
                if ((a == 0.0) & (b == 0.0)).any():
                    inner.append([])               # on the face's own side: the points of both faces, below
                    continue
                d = b - a
                t = (pts - a) @ d / float(d @ d)
                off = np.abs(pts - a - t[:, None] * d).max(axis=1)
                on = (off < 1e-12) & (t > 1e-12) & (t < 1.0 - 1e-12)
                inner.append([_name(i, f[i], pts[j]) for j in np.flatnonzero(on)[np.argsort(t[on])].tolist()])
            pieces.append((names, int(state) or base, inner))
    edges = {k: sorted(t) for k, t in on_edge.items()}

    # Whole faces with a split point on a side join the fan treatment; the rest go out as they are.
    whole = np.ones(len(f), bool)
    whole[list(split)] = False
    lo = np.minimum(f, np.roll(f, -1, axis=1))
    hi = np.maximum(f, np.roll(f, -1, axis=1))
    key = lo * len(v) + hi
    touched = np.isin(key, np.fromiter((a * len(v) + b for a, b in edges), np.int64, len(edges))).any(axis=1)
    for i in np.flatnonzero(whole & touched).tolist():
        pieces.append(([("v", int(g)) for g in f[i]], int(ext[i]), [[], [], []]))
    plain = whole & ~touched

    def position(n: Point) -> np.ndarray:
        if n[0] == "v":
            return v[n[1]]
        if n[0] == "e":
            return v[n[1]] * (1.0 - n[3]) + v[n[2]] * n[3]
        return np.asarray(n[2]) @ v[f[n[1]]]

    tris: list[np.ndarray] = [v[f[plain]]]
    exts: list[np.ndarray] = [ext[plain]]
    extra: list[np.ndarray] = []
    extra_ext: list[int] = []
    for names, state, inner in pieces:
        ring: list[Point] = []
        for k in range(3):
            ring.append(names[k])
            ring.extend(inner[k] or _between(names[k], names[(k + 1) % 3], edges))
        corner = [position(n) for n in ring]
        if len(ring) == 3:
            extra.append(np.asarray(corner))
            extra_ext.append(state)
            continue
        centre = np.mean(corner, axis=0)
        for k in range(len(corner)):
            extra.append(np.asarray([centre, corner[k], corner[(k + 1) % len(corner)]]))
            extra_ext.append(state)
    if extra:
        tris.append(np.asarray(extra))
        exts.append(np.asarray(extra_ext, np.int64))
    return np.concatenate(tris), np.concatenate(exts)


def stl_colour_word(hex_colour: str) -> int:
    """The VisCAM/SolidView 16-bit word for '#RRGGBB': valid bit, then 5 bits each of red, green, blue."""
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (1, 3, 5))
    return 0x8000 | (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)


def united(pieces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Several coloured bodies ((triangles (k, 3, 3), filament per triangle) each) as one solid: a 3MF keeps the
    flat base and the figure as parts of one object, but in a single STL their touching and overlapping surfaces
    would be edges on four triangles and faces inside the solid. Each body is welded and the bodies are joined
    (manifold3d); every triangle of the result keeps the filament of the input triangle it lies in (`face_map`).
    Raises RuntimeError when manifold3d rejects a body (not a closed solid)."""
    from .model3mf import face_map
    from .repair import repaired, weld
    from .shell import from_manifold, to_manifold

    import trimesh

    vs: list[np.ndarray] = []
    fs: list[np.ndarray] = []
    total = None
    n = 0
    for tris, _ in pieces:
        v, f = weld(tris.reshape(-1, 3), np.arange(len(tris) * 3, dtype=np.int64).reshape(-1, 3))
        man = to_manifold(trimesh.Trimesh(v, f, process=False))
        total = man if total is None else total + man
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    if total is None:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int64)
    out = from_manifold(total)
    v2, f2 = repaired(np.asarray(out.vertices, np.float64), np.asarray(out.faces, np.int64))
    ext = np.concatenate([e for _, e in pieces])
    fm = face_map(np.vstack(vs), np.vstack(fs), v2, f2)
    new = np.where(fm.match >= 0, ext[np.maximum(fm.match, 0)], 0)
    new[fm.unmatched] = ext[fm.nearest]
    return v2[f2], new


def write_coloured_stl(path: str | Path, parts: list[Part], default_extruder: int,
                       colours: list[str]) -> tuple[Path, str]:
    """Write `parts` (normal parts; modifiers are left out) as one binary STL coloured by filament
    (`colours[n - 1]` for filament n; a filament past the palette is grey), joined into one solid (`united`).
    Returns the path and a note ("" when all went well)."""
    bodies = [coloured_triangles(p, default_extruder) for p in parts
              if p.subtype == "normal_part" and len(p.faces)]
    note = ""
    if len(bodies) > 1:
        try:
            bodies = [united(bodies)]
        except RuntimeError as exc:
            # TODO: a body manifold3d rejects (an open mesh the repair could not close) is written beside the others.
            note = f"coloured STL: parts written separately, not joined into one solid ({exc})"
    tris = np.concatenate([t for t, _ in bodies]) if bodies else np.zeros((0, 3, 3))
    exts = np.concatenate([e for _, e in bodies]) if bodies else np.zeros(0, np.int64)
    words = np.asarray([stl_colour_word(c) for c in colours] + [stl_colour_word("#808080")], np.uint16)
    idx = np.where((exts >= 1) & (exts <= len(colours)), exts - 1, len(colours))
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(tris), _STL)
    rec["normal"] = n
    rec["corners"] = tris
    rec["attr"] = words[idx]
    path = Path(path)
    with path.open("wb") as fh:
        fh.write(HEADER.ljust(80, b" "))
        fh.write(np.uint32(len(rec)).tobytes())
        fh.write(rec.tobytes())
    return path, note


def read_coloured_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(triangles (k, 3, 3), '#RRGGBB' per triangle, '' where none) of a VisCAM-coloured binary STL (tests)."""
    data = Path(path).read_bytes()
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    rec = np.frombuffer(data[84:84 + n * _STL.itemsize], _STL)
    w = rec["attr"].astype(np.int64)
    hexes = np.asarray([f"#{(x >> 10 & 31) << 3:02X}{(x >> 5 & 31) << 3:02X}{(x & 31) << 3:02X}" if x & 0x8000
                        else "" for x in w.tolist()])
    return rec["corners"].astype(np.float64), hexes
