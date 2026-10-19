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
from typing import Any

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
