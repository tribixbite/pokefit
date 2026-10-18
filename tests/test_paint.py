from pokefit.paint import decode_majority, decode_weights


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
