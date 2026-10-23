"""Decode Bambu/Prusa per-triangle multi-material paint (`paint_color` attribute).

The attribute is PrusaSlicer's TriangleSelector serialisation: a hex string
whose digits are read from the END of the string, one nibble per step.
For each node nibble `d`:
  * `d & 3` = number of split sides (0 = leaf). Children = split sides + 1.
  * leaf: state = `d >> 2`; if that is 3, the state is 3 + the following
    nibbles: each 0xF adds 15, and the first nibble below 0xF ends it (Bambu
    TriangleSelector::serialize; filaments 18+ need the 0xF continuation).
  * split: `d >> 2` is the special side, then the children follow (depth first).
State 0 means "not painted" (use the part/object extruder), state k ≥ 1 means
extruder k. Examples: "4" → 1, "8" → 2, "0C" → 3, "1C" → 4.

Previews decode; `remap_states` rewrites the filament numbers exactly (for
an export that is imported into another project with other filament slots).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np


def _nibbles(code: str) -> list[int]:
    return [int(c, 16) for c in reversed(code.strip())]


def decode_weights(code: str) -> dict[int, float]:
    """Area fraction per state (each child of a split takes an equal share)."""
    nib = _nibbles(code)
    weights: dict[int, float] = defaultdict(float)
    pos = 0

    def node(share: float) -> None:
        nonlocal pos
        if pos >= len(nib):
            return
        d = nib[pos]
        pos += 1
        split = d & 3
        if split == 0:
            state = d >> 2
            if state == 3:
                state = 3
                while pos < len(nib) and nib[pos] == 0xF:
                    state += 15
                    pos += 1
                state += nib[pos] if pos < len(nib) else 0
                pos += 1
            weights[state] += share
            return
        n = split + 1
        for _ in range(n):
            node(share / n)

    node(1.0)
    return dict(weights)


def decode_majority(code: str) -> int:
    """Dominant state of a painted triangle (0 = unpainted)."""
    w = decode_weights(code)
    return max(w.items(), key=lambda kv: kv[1])[0] if w else 0


@dataclass(frozen=True)
class FacePaint:
    """One triangle's `paint_color`, decoded."""
    code: str
    whole: int | None                 # its state when the triangle is not split, else None
    majority: int                     # dominant state (0 = unpainted)
    states: tuple[int, ...]           # painted states (> 0) it holds, ascending


@lru_cache(maxsize=1 << 16)
def attr_paint(attrs: str) -> FacePaint | None:
    """The `paint_color` in a triangle's extra attributes (None when it has none).

    Colorize paint gives hundreds of thousands of triangles the same few strings, and
    previews, checks and the colour-change estimate each read every triangle: decoding
    once per distinct string took a 300k-triangle figure's paint pass from seconds to
    tens of milliseconds.
    """
    if 'paint_color="' not in attrs:
        return None
    code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
    w = decode_weights(code)
    tree = parse_tree(code)
    return FacePaint(code, tree if isinstance(tree, int) else None,
                     max(w.items(), key=lambda kv: kv[1])[0] if w else 0, tuple(sorted(k for k in w if k > 0)))


def _leaf_nibbles(state: int) -> list[int]:
    if state < 3:
        return [state << 2]
    n = state - 3
    out = [0b1100]
    while n >= 15:
        out.append(0xF)
        n -= 15
    return out + [n]


def remap_states(code: str, mapping: dict[int, int]) -> str:
    """The same paint with each painted state k replaced by mapping.get(k, k); splits are kept exactly."""
    nib = _nibbles(code)
    out: list[int] = []
    pos = 0

    def node() -> None:
        nonlocal pos
        if pos >= len(nib):
            return
        d = nib[pos]
        pos += 1
        split = d & 3
        if split == 0:
            state = d >> 2
            if state == 3:
                state = 3
                while pos < len(nib) and nib[pos] == 0xF:
                    state += 15
                    pos += 1
                state += nib[pos] if pos < len(nib) else 0
                pos += 1
            out.extend(_leaf_nibbles(mapping.get(state, state) if state else 0))
            return
        out.append(d)
        for _ in range(split + 1):
            node()

    node()
    return "".join("0123456789ABCDEF"[n] for n in reversed(out))


# ------------------------------------------------------------ split geometry
# A painted triangle is a tree: a leaf is a state; a split node is (split sides, special
# side, children). Children are kept here in Bambu's own index order (children[0..n]);
# the string stores them last-first (TriangleSelector::serialize, "reverse order for
# compatibility with PrusaSlicer 2.3.1"), and a reader assigns the first one it reads to
# children[n]. Child geometry is TriangleSelector::perform_split's, verified against
# BambuStudio src/libslic3r/TriangleSelector.cpp (2026-10).
Tree = int | tuple[int, int, list["Tree"]]


def parse_tree(code: str) -> Tree:
    """The split tree of one `paint_color` string (an empty string reads as state 0)."""
    nib = _nibbles(code)
    pos = 0

    def node() -> Tree:
        nonlocal pos
        if pos >= len(nib):
            return 0
        d = nib[pos]
        pos += 1
        split = d & 3
        if split == 0:
            state = d >> 2
            if state == 3:
                while pos < len(nib) and nib[pos] == 0xF:
                    state += 15
                    pos += 1
                state += nib[pos] if pos < len(nib) else 0
                pos += 1
            return state
        read = [node() for _ in range(split + 1)]
        return split, d >> 2, read[::-1]

    return node()


def encode_tree(tree: Tree) -> str:
    """`paint_color` string of a split tree (the inverse of `parse_tree`)."""
    out: list[int] = []

    def node(t: Tree) -> None:
        if isinstance(t, int):
            out.extend(_leaf_nibbles(t))
            return
        split, special, children = t
        out.append(split | special << 2)
        for c in reversed(children):
            node(c)

    node(tree)
    return "".join("0123456789ABCDEF"[n] for n in reversed(out))


def split_children(v: list, split: int, special: int) -> list[list]:
    """Corner lists of the children of a triangle with corners `v` (any vector type that
    supports + and * by a float), in children[] order (TriangleSelector::perform_split)."""
    a, b, c = v[special], v[(special + 1) % 3], v[(special + 2) % 3]

    def mid(p: Any, q: Any) -> Any:
        return (p + q) * 0.5

    if split == 1:                     # the side opposite the special corner is halved
        m = mid(c, b)
        return [[a, b, m], [m, c, a]]
    if split == 2:                     # both sides at the special corner are halved
        mab, mac = mid(b, a), mid(a, c)
        return [[a, mab, mac], [mab, b, mac], [b, c, mac]]
    mab, mbc, mca = mid(b, a), mid(c, b), mid(a, c)   # split == 3 (special side is 0)
    return [[a, mab, mca], [mab, b, mbc], [mbc, c, mca], [mab, mbc, mca]]


def leaves(tree: Tree, v: list) -> list[tuple[list, int]]:
    """(corners, state) of every leaf of a tree laid on the triangle with corners `v`."""
    if isinstance(tree, int):
        return [(v, tree)]
    split, special, children = tree
    out: list[tuple[list, int]] = []
    for cv, ct in zip(split_children(v, split, special), children):
        out.extend(leaves(ct, cv))
    return out


def remap_tree(tree: Tree, perm: tuple[int, int, int]) -> Tree:
    """The same paint for the same triangle with its corners listed in another order:
    new corner j = old corner perm[j] (a rotation, or a reversal for a mirrored triangle).

    Each node keeps its split; its special corner is renumbered, the children it then
    makes are matched to the old children by their corners, and each child is remapped
    by the corner order it ends up with (children's corner orders follow the parent's in
    a different way for each child, so this is solved geometrically, not by a rule).

    Exact for rotations (what re-meshing produces). A mirrored order turns a 2-side split's
    quad diagonal the other way, so those two children cannot be kept: they become one
    state, the larger of the two.
    """
    if isinstance(tree, int) or tuple(perm) == (0, 1, 2):
        return tree
    split, special, children = tree
    basis = [np.eye(3)[i] for i in range(3)]
    old = split_children(basis, split, special)
    new_special = special if split == 3 else list(perm).index(special)
    new = split_children([basis[k] for k in perm], split, new_special)
    key = [{tuple(np.round(c, 6)) for c in tri} for tri in old]
    matches = [{tuple(np.round(c, 6)) for c in tri} for tri in new]
    unmatched = [i for i, k in enumerate(key) if k not in matches]
    rest: Tree = 0
    if unmatched:                                       # mirrored 2-side split: the flipped quad
        area: dict[int, float] = defaultdict(float)
        for i in unmatched:
            for st, w in decode_weights(encode_tree(children[i])).items():
                area[st] += w
        rest = max(area.items(), key=lambda kv: kv[1])[0]
    out: list[Tree] = []
    for tri, m in zip(new, matches):
        if m not in key:
            out.append(rest)
            continue
        i = key.index(m)
        o = [tuple(np.round(c, 6)) for c in old[i]]
        sub = tuple(o.index(tuple(np.round(c, 6))) for c in tri)
        out.append(remap_tree(children[i], sub))       # type: ignore[arg-type]
    return split, new_special, out


def remap_code(code: str, perm: tuple[int, int, int]) -> str:
    """`remap_tree` on a `paint_color` string (unsplit paint is returned unchanged)."""
    tree = parse_tree(code)
    return code if isinstance(tree, int) else encode_tree(remap_tree(tree, perm))


def majority_tree(tree: Tree) -> int:
    """The state covering most of a tree's area."""
    return decode_majority(encode_tree(tree)) if not isinstance(tree, int) else tree


def _collapse(split: int, special: int, kids: list[Tree]) -> Tree:
    """A split node, or the one state all its children hold."""
    if all(isinstance(k, int) for k in kids) and len({k for k in kids if isinstance(k, int)}) == 1:
        return kids[0]
    return split, special, kids


def overlay(tree: Tree, phi: tuple[float, float, float], state: int, depth: int,
            only: Callable[[int], bool] | None = None, table: list[list[float]] | None = None,
            inner: bool = False) -> Tree:
    """`tree` (one triangle's paint) with the part where a field is <= 0 painted `state`; with `only`, just its
    leaves whose state passes (a fill bounded by colour must not run over a neighbour's split edge).

    The field is `phi` at the triangle's three corners (its order in the mesh), read linearly in between, or,
    with `table`, its value at every split point down to `depth` levels: `table[i][j]` at barycentric coordinates
    ((L - i - j) / L, i / L, j / L), L = 2 ** depth (points off that grid, inside an older deeper split, read
    `phi` linearly). With `inner`, the area may lie wholly inside a (sub-)triangle whose corners are all outside
    (a small stamp on a big triangle): such a triangle is still split while a table point inside it is <= 0.
    The painted edge is the field's zero line. An existing split keeps its structure (each of
    its leaves is overlaid in turn); a leaf the line crosses is split into four (TriangleSelector's 3-side
    split) down to `depth` levels below the triangle, the last level taking the state of its centre.
    """
    big = float(1 << depth)
    p0, p1, p2 = (float(x) for x in phi)

    # A point is a complex number i + j*1j in lattice units: the split helper averages corners, and complex
    # numbers average exactly (dyadic), cheaply, with no arrays.
    def value(c: complex) -> float:
        i, j = c.real, c.imag
        if table is not None and i == int(i) and j == int(j):
            return table[int(i)][int(j)]
        return ((big - i - j) * p0 + i * p1 + j * p2) / big

    negs = [complex(i, j) for i, row in enumerate(table) for j, v in enumerate(row) if v <= 0] \
        if inner and table is not None else []

    def holds(bc: list[complex]) -> bool:
        """A negative table point lies in the triangle `bc` (edges included)."""
        a, b, c = bc
        den = (b - a).real * (c - a).imag - (b - a).imag * (c - a).real
        if abs(den) < 1e-12:
            return False
        for p in negs:
            u = ((p - a).real * (c - a).imag - (p - a).imag * (c - a).real) / den
            v = ((b - a).real * (p - a).imag - (b - a).imag * (p - a).real) / den
            if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
                return True
        return False

    def rec(t: Tree, bc: list[complex], d: int) -> Tree:
        ph = [value(b) for b in bc]
        if isinstance(t, int):
            if only is not None and not only(t):
                return t
            if max(ph) <= 0:
                return state
            if min(ph) > 0 and not (negs and d < depth and holds(bc)):
                return t
            if d >= depth:
                return state if value((bc[0] + bc[1] + bc[2]) / 3) <= 0 else t
            return _collapse(3, 0, [rec(t, c, d + 1) for c in split_children(bc, 3, 0)])
        split, special, children = t
        return _collapse(split, special, [rec(c, cb, d + 1)
                                          for c, cb in zip(children, split_children(bc, split, special))])

    return rec(tree, [0j, complex(big, 0), complex(0, big)], 0)
