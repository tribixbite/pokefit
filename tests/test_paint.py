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
