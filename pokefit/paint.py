"""Decode Bambu/Prusa per-triangle multi-material paint (`paint_color` attribute).

The attribute is PrusaSlicer's TriangleSelector serialisation: a hex string
whose digits are read from the END of the string, one nibble per step.
For each node nibble `d`:
  * `d & 3` = number of split sides (0 = leaf). Children = split sides + 1.
  * leaf: state = `d >> 2`; if that is 3, the next nibble + 3 is the state.
  * split: `d >> 2` is the special side, then the children follow (depth first).
State 0 means "not painted" (use the part/object extruder), state k ≥ 1 means
extruder k. Examples: "4" → 1, "8" → 2, "0C" → 3, "1C" → 4.

Only previews need decoding: the output 3MF copies paint strings verbatim.
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
                state = (nib[pos] if pos < len(nib) else 0) + 3
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
