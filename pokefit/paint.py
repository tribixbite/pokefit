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
