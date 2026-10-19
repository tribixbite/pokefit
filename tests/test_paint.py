import itertools
import random

import numpy as np

from pokefit.paint import (Tree, decode_majority, decode_weights, encode_tree, leaves, parse_tree, remap_code,
                           remap_tree, split_children)


def test_leaf_states() -> None:
    assert decode_majority("4") == 1
    assert decode_majority("8") == 2
    assert decode_majority("0C") == 3
    assert decode_majority("1C") == 4
    assert decode_majority("0") == 0


def test_split_weights() -> None:
    # Read from the end: '1' = split into 2 children (special side 0),
    # then child '8' (state 2) and child '4' (state 1), half each.
    w = decode_weights("481")
    assert w == {2: 0.5, 1: 0.5}
    # Three children: two of state 2, one unpainted.
    assert decode_majority("0882") == 2


def test_real_dragon_codes_decode() -> None:
    for code in ["4002", "0442", "400A", "04006044A2", "40446400AA"]:
        w = decode_weights(code)
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_remap_leaves() -> None:
    from pokefit.paint import remap_states

    assert remap_states("4", {1: 2}) == "8"
    assert remap_states("4", {1: 4}) == "1C"
    assert remap_states("8", {2: 1}) == "4"
    assert remap_states("0", {1: 5}) == "0"                  # unpainted stays unpainted
    # 18 = 3 + 15: Bambu writes 0b1100, then 0xF (+15), then 0.
    assert remap_states("4", {1: 18}) == "0FC" and decode_majority("0FC") == 18
    assert remap_states("0FC", {18: 1}) == "4"


def test_remap_keeps_splits() -> None:
    from pokefit.paint import remap_states

    codes = ["481", "0882", "4002", "0442", "400A", "04006044A2", "40446400AA", "1C0C2C0C1C13"]
    for code in codes:
        assert remap_states(code, {}) == code                 # exact round trip
        w = decode_weights(code)
        m = remap_states(code, {1: 7, 2: 9})
        assert decode_weights(m) == {({1: 7, 2: 9}.get(k, k)): v for k, v in w.items()}


def _random_tree(rng: random.Random, depth: int) -> Tree:
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([0, 1, 2, 3, 4, 17, 20])        # includes multi-nibble states
    split = rng.choice([1, 2, 3])
    special = 0 if split == 3 else rng.randrange(3)
    return split, special, [_random_tree(rng, depth - 1) for _ in range(split + 1)]


def _painted(tree: Tree, corners: list[np.ndarray]) -> set[tuple[frozenset, int]]:
    """Each leaf as (its corner set, state): what the paint covers, independent of corner order."""
    return {(frozenset(tuple(np.round(c, 9)) for c in v), st) for v, st in leaves(tree, corners)}


def test_tree_roundtrip() -> None:
    rng = random.Random(3)
    for _ in range(200):
        t = _random_tree(rng, 4)
        assert parse_tree(encode_tree(t)) == t
    for code in ("4", "8", "0C", "1C", "481", "84443", "2FC"):
        assert encode_tree(parse_tree(code)) == code


def test_split_geometry_matches_bambu() -> None:
    a, b, c = (np.array(p, float) for p in ([0, 0], [4, 0], [0, 4]))
    kids = split_children([a, b, c], 3, 0)                 # (v0,m01,m20) (m01,v1,m12) (m12,v2,m20) centre
    assert [[tuple(p) for p in k] for k in kids] == [
        [(0, 0), (2, 0), (0, 2)], [(2, 0), (4, 0), (2, 2)], [(2, 2), (0, 4), (0, 2)], [(2, 0), (2, 2), (0, 2)]]
    one = split_children([a, b, c], 1, 0)                  # halves the side opposite corner 0
    assert [[tuple(p) for p in k] for k in one] == [[(0, 0), (4, 0), (2, 2)], [(2, 2), (0, 4), (0, 0)]]
    # Serialised last child first: "481" = split 1, then children[1] = state 2, children[0] = state 1.
    assert parse_tree("481") == (1, 0, [1, 2])


def test_remap_paints_the_same_area() -> None:
    rng = random.Random(5)
    v = [np.array([0.0, 0, 0]), np.array([7.0, 1, 0]), np.array([2.0, 5, 1])]
    for _ in range(60):
        t = _random_tree(rng, 3)
        for perm in itertools.permutations(range(3)):      # rotations and mirrored orders
            moved = remap_tree(t, perm)
            mirrored = perm not in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            if mirrored and "(2," in repr(t):              # a mirrored 2-split loses its quad detail
                assert set(decode_weights(encode_tree(moved))) <= set(decode_weights(encode_tree(t)))
                continue
            assert _painted(moved, [v[k] for k in perm]) == _painted(t, v)
    assert remap_code("4", (1, 2, 0)) == "4"
    # A mirrored 2-split keeps its special corner child; the flipped quad takes its larger state.
    assert remap_tree((2, 0, [3, 1, 1]), (0, 2, 1)) == (2, 0, [3, 1, 1])
