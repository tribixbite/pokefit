import numpy as np
import pytest

from pokefit.model3mf import Part
from pokefit.paint import decode_weights
from pokefit.remap import delta_e, match_slots, remap_parts, used_filaments


def _part(extruder: int, paint: dict[int, str] | None = None) -> Part:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    return Part("p", extruder, v, np.array([[0, 1, 2]]), {i: f'paint_color="{c}"' for i, c in (paint or {}).items()})


def test_delta_e() -> None:
    assert delta_e("#161616", "#161616") == pytest.approx(0.0)
    assert delta_e("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.1)


def test_used_filaments_includes_paint_and_inherited() -> None:
    parts = [_part(0, {0: "8"}), _part(3)]
    assert used_filaments(parts, object_extruder=1) == [1, 2, 3]


def test_match_nearest_and_new_slot() -> None:
    targets = ["#FFFFFF", "#161616", "#F72323"]
    m = match_slots([1, 2, 3], ["#E05028", "#101010", "#2850E0"], targets)
    assert [(x.source, x.target, x.new) for x in m] == [(1, 3, False), (2, 2, False), (3, 4, True)]
    assert m[1].delta_e < 3


def test_manual_override() -> None:
    m = match_slots([1], ["#E05028"], ["#FFFFFF", "#F72323"], manual={1: 1})
    assert m[0].target == 1 and m[0].manual and m[0].target_colour == "#FFFFFF"


def test_remap_parts() -> None:
    parts = [_part(0, {0: "481"}), _part(2)]
    m = match_slots([1, 2], ["#E05028", "#161616"], ["#161616", "#FFFFFF", "#F72323"])
    new, ext = remap_parts(parts, 1, m)
    assert ext == 3 and [p.extruder for p in new] == [3, 1]
    code = new[0].tri_extra[0].split('"')[1]
    assert decode_weights(code) == {1: 0.5, 3: 0.5}               # 2 -> 1 (black), 1 -> 3 (red)
