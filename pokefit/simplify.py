"""Simplify dense meshes at import: fewer triangles, the same shape.

Sculpts and scans arrive with hundreds of thousands to millions of triangles,
far more than a 25-40 mm print shows (a 0.4 mm nozzle draws 0.16 mm layers).
Every later step (fit, conform, the checks, paint taps, the preview, Bambu
Studio) costs time per triangle, and a dense sculpt's tiny noisy triangles
break paint regions into fragments. Like Bambu Studio's "Simplify model", this
collapses edges by quadric error (meshoptimizer's `meshopt_simplify`, keeping
triangles of even size: `OPTIONS`) down to a triangle budget, but never moves the surface by more than `ERROR` of the
model's size (0.1 %: 0.03 mm on a 30 mm figure), so it stops early where the
shape needs the triangles.

The simplifier is meshoptimizer's own WebAssembly build (MIT, vendored in
`vendor/`, extracted from the `meshoptimizer` npm package 1.3.0). The same
binary runs locally (wasmtime) and in the browser (Pyodide: the worker's
WebAssembly), so both engines produce identical triangles and paint steps
refer to the same faces.

Per-triangle attributes (Bambu paint) go to each new triangle from the old
triangle nearest its centre, reduced to one colour per triangle (a split
triangle's pattern would land on the wrong areas). glTF colour is sampled
after simplifying (gltf.py), so it is not coarsened that way.
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .model3mf import Part

WASM = Path(__file__).resolve().parent / "vendor" / "meshopt_simplifier.wasm"
ERROR = 1e-3            # largest surface change allowed, as a fraction of the model's largest extent
MIN_FACES = 2_000       # parts smaller than this are kept as they are (eyes, a nose)
CHOICES = (25_000, 50_000, 100_000, 200_000, 400_000)   # triangle budgets the UI offers
# meshopt_SimplifyRegularizeLight: keep triangles of even size. Plain quadric collapse leaves long triangles on flat
# areas, and paint covers whole triangles, so painted edges there came out jagged. Measured on the 1.45M-triangle
# Eevee at 100k: longest edge p95 0.93 -> 0.52 mm, p99 1.54 -> 0.60 mm, surface change 0.007 -> 0.021 mm (of 40).
OPTIONS = 64
MISS_SHARE = 0.1      # ... unless it stops this far above the target (`simplify_indices`)
# A simplify budget (Params.simplify): a triangle count (>= 1,000), or a share of the file's triangles (0 < b < 1),
# so a level such as "medium" means the same on a 30k and a 1.4M mesh.
Budget = float


class _Wasm:
    """meshopt_simplifier.wasm: exports `memory`, `sbrk` and the `meshopt_*` functions; no imports."""

    def alloc(self, n: int) -> int:
        """Bump-allocate `n` bytes (sbrk), 16-byte aligned; returns the address."""
        n = (n + 15) & ~15
        return int(self.call("sbrk", n))

    def release(self, ptr: int) -> None:
        """Free everything allocated since `ptr` (the allocator is a stack)."""
        self.call("sbrk", ptr - int(self.call("sbrk", 0)))

    def write(self, ptr: int, data: bytes) -> None:
        raise NotImplementedError

    def read(self, ptr: int, n: int) -> bytes:
        raise NotImplementedError

    def call(self, name: str, *args: Any) -> Any:
        raise NotImplementedError


class _Wasmtime(_Wasm):
    """CPython: the wasmtime runtime."""

    def __init__(self, data: bytes):
        import wasmtime

        self._store = wasmtime.Store()
        inst = wasmtime.Instance(self._store, wasmtime.Module(self._store.engine, data), [])
        self._ex = inst.exports(self._store)
        mem = self._ex["memory"]
        assert isinstance(mem, wasmtime.Memory)
        self._mem = mem
        self._func = wasmtime.Func
        self.call("__wasm_call_ctors")

    def write(self, ptr: int, data: bytes) -> None:
        self._mem.write(self._store, data, ptr)

    def read(self, ptr: int, n: int) -> bytes:
        return bytes(self._mem.read(self._store, ptr, ptr + n))

    def call(self, name: str, *args: Any) -> Any:
        fn = self._ex[name]
        assert isinstance(fn, self._func)
        return fn(self._store, *args)


class _Browser(_Wasm):
    """Pyodide: the JavaScript engine's own WebAssembly (synchronous compile is allowed in a worker)."""

    def __init__(self, data: bytes):
        from js import Object, Uint8Array, WebAssembly

        buf = Uint8Array.new(len(data))
        buf.assign(data)
        self._inst = WebAssembly.Instance.new(WebAssembly.Module.new(buf), Object.new())
        self._u8 = Uint8Array
        self.call("__wasm_call_ctors")

    def _view(self, ptr: int, n: int) -> Any:
        # The memory may have grown (and its buffer been replaced) since the last call: a fresh view each time.
        return self._u8.new(self._inst.exports.memory.buffer, ptr, n)

    def write(self, ptr: int, data: bytes) -> None:
        self._view(ptr, len(data)).assign(data)

    def read(self, ptr: int, n: int) -> bytes:
        return bytes(self._view(ptr, n).to_bytes())

    def call(self, name: str, *args: Any) -> Any:
        return getattr(self._inst.exports, name)(*args)


@lru_cache(maxsize=1)
def _runtime() -> _Wasm:
    data = WASM.read_bytes()
    return _Browser(data) if sys.platform == "emscripten" else _Wasmtime(data)


def simplify_indices(vertices: np.ndarray, faces: np.ndarray, target: int, error: float = ERROR
                     ) -> tuple[np.ndarray, float]:
    """(faces, relative error): `faces` cut to about `target` triangles over the SAME vertex array (a subset of it
    is used), stopping early where the surface would move by more than `error` x the model's extent. Vertices with
    the same position are a seam (a glTF splits them at UV seams) and stay matched, so no crack opens.

    Triangles are kept even in size (`OPTIONS`) unless that stops more than MISS_SHARE above the target: on a mesh
    that is even already (a uniform sphere) the evening spends the error allowance and stops early; there plain
    collapse gets further (and leaves even triangles anyway)."""
    v = np.ascontiguousarray(vertices, np.float32)
    f = np.ascontiguousarray(faces, np.uint32)
    out, rel = _simplify(v, f, target, error, OPTIONS)
    if len(out) > target * (1 + MISS_SHARE):
        plain, rel0 = _simplify(v, f, target, error, 0)
        if len(plain) < len(out):
            return plain, rel0
    return out, rel


def _simplify(v: np.ndarray, f: np.ndarray, target: int, error: float, options: int) -> tuple[np.ndarray, float]:
    """One `meshopt_simplify` call: float32 positions, uint32 indices."""
    n_idx = f.size
    rt = _runtime()
    base = rt.alloc(0)
    try:
        err_p = rt.alloc(4)
        dst = rt.alloc(n_idx * 4)
        src = rt.alloc(n_idx * 4)
        pos = rt.alloc(v.nbytes)
        rt.write(src, f.tobytes())
        rt.write(pos, v.tobytes())
        target_idx = max(3, min(n_idx, int(target) * 3))
        count = int(rt.call("meshopt_simplify", dst, src, n_idx, pos, len(v), 12, target_idx, float(error), options,
                            err_p)) & 0xFFFFFFFF
        out = np.frombuffer(rt.read(dst, count * 4), np.uint32).astype(np.int64).reshape(-1, 3)
        rel = struct.unpack("<f", rt.read(err_p, 4))[0]
    finally:
        rt.release(base)
    return out, float(rel)


def clean(faces: np.ndarray) -> np.ndarray:
    """Faces without degenerate triangles and without folds (`clean_index`)."""
    return faces[clean_index(faces)]


def clean_index(faces: np.ndarray) -> np.ndarray:
    """Indices (ascending) of the faces to keep: no degenerate triangles and no folds. An edge collapse can leave
    two triangles on the same three corners (a zero-volume fin, measured: 2-9 non-manifold edges per 100k
    triangles); both copies of a reversed pair go, one of a repeated one stays."""
    ok = np.nonzero((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))[0]
    f = faces[ok]
    if not len(f):
        return ok
    group, first, counts = group_rows(np.sort(f, axis=1))
    order = np.argsort(f, axis=1)
    # Winding as a parity (the permutation that sorts the corners is even or odd), as in gltf.two_sided.
    parity = ((order[:, 0] > order[:, 1]).astype(np.int64) + (order[:, 0] > order[:, 2]) + (order[:, 1] > order[:, 2])) % 2
    odd = np.bincount(group, weights=parity, minlength=len(counts))
    mixed = (odd > 0) & (odd < counts)                      # both windings present: a fin
    keep = np.zeros(len(f), bool)
    keep[first[~mixed]] = True
    return ok[np.sort(np.nonzero(keep)[0])]


def group_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group, first, counts) of equal integer rows, as `np.unique(rows, axis=0, return_inverse, return_index,
    return_counts)` gives them (groups in sorted order, first = lowest index), by one lexsort: several times faster
    than `unique` on rows (1.5 s -> 0.3 s for 1.4M triangles)."""
    n = len(rows)
    if not n:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    # Fold neighbouring columns into one int64 key while their ranges fit 62 bits: sorting one key is ~7x faster
    # than lexsort over three (triangle corners always fold into one; positions on a 1e-6 grid into two).
    lo = rows.min(axis=0)
    span = rows.max(axis=0) - lo + 1
    keys: list[np.ndarray] = []
    cur, size = np.zeros(n, np.int64), 1
    for c in range(rows.shape[1]):
        if size * int(span[c]) >= 1 << 62:
            keys.append(cur)
            cur, size = np.zeros(n, np.int64), 1
        cur = cur * int(span[c]) + (rows[:, c] - lo[c])
        size *= int(span[c])
    keys.append(cur)
    order = np.argsort(keys[0], kind="stable") if len(keys) == 1 else np.lexsort(keys[::-1])
    new = np.empty(n, bool)
    new[0] = True
    new[1:] = False
    for k in keys:
        srt = k[order]
        new[1:] |= srt[1:] != srt[:-1]
    group = np.empty(n, np.int64)
    group[order] = np.cumsum(new) - 1
    starts = np.nonzero(new)[0]
    return group, order[starts], np.diff(np.append(starts, n))


def compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices no face uses (face order kept)."""
    used, inv = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[used], inv.reshape(-1, 3).astype(np.int64)


@dataclass(frozen=True)
class SimplifyInfo:
    faces_before: int
    faces_after: int
    error: float          # largest surface change, file mm (meshoptimizer's estimate)

    def to_dict(self) -> dict[str, Any]:
        return {"faces_before": self.faces_before, "faces_after": self.faces_after, "error": self.error}


def resolve(budget: Budget, faces: int) -> int:
    """Triangles a budget asks for on a mesh of `faces` triangles (a share of them, or the count itself)."""
    return max(1, round(budget * faces)) if budget < 1 else int(budget)


def budgets(counts: list[int], budget: Budget) -> list[int | None]:
    """Each part's triangle budget for `budget` over all of them (None = leave the part as it is): shared in
    proportion to size among the parts above MIN_FACES; nothing changes when the total is within the target."""
    target = resolve(budget, sum(counts))
    big = [n if n >= MIN_FACES else 0 for n in counts]
    small = sum(counts) - sum(big)
    if sum(counts) <= target or not sum(big):
        return [None] * len(counts)
    room = max(target - small, MIN_FACES * sum(1 for n in big if n))
    return [max(MIN_FACES, round(room * n / sum(big))) if n else None for n in big]


def simplify_part(part: "Part", target: int, error: float = ERROR) -> tuple["Part", float]:
    """(part with about `target` triangles, its surface change in file mm). Paint and other per-triangle
    attributes come from the old triangle nearest each new one's centre, one state per triangle."""
    from scipy.spatial import cKDTree

    from .model3mf import Part, whole_paint_attrs

    faces, rel = simplify_indices(part.vertices, part.faces, target, error)
    faces = clean(faces)
    if not len(faces):
        return part, 0.0
    v, f = compact(part.vertices, faces)
    extra: dict[int, str] = {}
    if part.tri_extra:
        old = part.vertices[part.faces].mean(axis=1)
        _, src = cKDTree(old).query(v[f].mean(axis=1))
        for i, s in enumerate(np.asarray(src).tolist()):
            a = part.tri_extra.get(s)
            if a:
                extra[i] = whole_paint_attrs(a)
    extent = float(np.ptp(part.vertices, axis=0).max()) if len(part.vertices) else 0.0
    return Part(part.name, part.extruder, v, f, extra, part.subtype), rel * extent


def simplify_parts(parts: list["Part"], budget: Budget, error: float = ERROR) -> tuple[list["Part"], SimplifyInfo]:
    """`parts` cut to `budget` triangles in all (modifier and negative parts, and small parts, unchanged)."""
    counts = [len(p.faces) if p.subtype == "normal_part" else 0 for p in parts]
    want = budgets(counts, budget)
    out: list[Part] = []
    worst = 0.0
    for p, b in zip(parts, want):
        if b is None or b >= len(p.faces):
            out.append(p)
            continue
        q, e = simplify_part(p, b, error)
        out.append(q)
        worst = max(worst, e)
    before = sum(len(p.faces) for p in parts)
    return out, SimplifyInfo(before, sum(len(p.faces) for p in out), worst)
