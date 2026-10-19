"""Colour changes on one or two nozzles (pokefit/toolchange.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from pokefit.model3mf import Part
from pokefit.pipeline import BUNDLED_REFERENCE, Params, analyse, open_session
from pokefit.slicer import reference_settings
from pokefit.toolchange import best_value_count, flush_matrix, grams, layer_use, nozzle_count, plan, simulate


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float], ext: int, name: str) -> Part:
    m = trimesh.creation.box(bounds=[lo, hi])
    return Part(name, ext, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), {}, "normal_part")


def test_layers_know_their_filaments() -> None:
    low = _box((0, 0, 0), (10, 10, 2.0), 1, "low")
    high = _box((0, 0, 2.0), (10, 10, 4.0), 2, "high")
    # One face of the high box painted filament 3 (a whole-triangle paint state).
    high.tri_extra[0] = 'paint_color="C"'
    use = layer_use([low, high], 1, 3, 0.2, 0.2)
    assert use.shape == (20, 3)
    assert use[:9, 0].all() and not use[:9, 1].any()            # below z 2: only the low box
    assert use[11:, 1].all() and not use[11:, 0].any()          # above: only the high box
    assert use[:, 2].any()                                       # the painted face's layers


def test_two_nozzles_keep_a_colour_each() -> None:
    layers = 50
    use = np.ones((layers, 2), bool)                             # two colours on every layer
    flush = np.array([[0, 100.0], [100.0, 0]])
    one = simulate(use, [1, 1], flush)
    two = simulate(use, [1, 2], flush)
    # One nozzle: change every layer; it starts each layer with what it ended on.
    assert one.changes == layers and one.switches == 0 and one.purge_mm3 == 100.0 * layers
    assert two.changes == 0 and two.switches == layers and two.purge_mm3 == 0
    best, single = plan(use, flush)
    assert best.nozzle_of == [1, 2] and single.nozzle_of == [1, 1] and best.seconds < single.seconds
    assert grams(single) > grams(best)


def test_plan_respects_allowed_nozzles_and_third_colours() -> None:
    use = np.zeros((30, 3), bool)
    use[:, 0] = True                     # base everywhere
    use[:10, 1] = True                   # a colour low down
    use[20:, 2] = True                   # another colour high up
    flush = np.full((3, 3), 120.0)
    np.fill_diagonal(flush, 0)
    best, _ = plan(use, flush)
    # 1 and 2 never share a layer with each other, only with the base: they can share the other nozzle.
    assert best.nozzle_of[0] != best.nozzle_of[1] and best.nozzle_of[0] != best.nozzle_of[2]
    assert best.changes == 1
    pinned, _ = plan(use, flush, allowed=[{1}, {1}, {1, 2}])      # e.g. TPU only on the direct-drive nozzle
    assert pinned.nozzle_of[:2] == [1, 1] and pinned.nozzle_of[2] == 2


def test_settings_helpers() -> None:
    ref = reference_settings(BUNDLED_REFERENCE)
    assert nozzle_count(ref) == 2                                 # the X2D
    m = flush_matrix(ref, 3)
    assert m.shape == (3, 3) and np.all(np.diag(m) == 0) and m[0, 1] > 0
    assert flush_matrix({}, 2)[0, 1] == 150.0


def test_report_has_colour_change_estimates(tmp_path: Path) -> None:
    # Two coloured boxes, side by side and stacked so both colours share layers.
    s = trimesh.Scene()
    for i, (c, x) in enumerate((((200, 20, 20, 255), -6.0), ((20, 20, 200, 255), 6.0))):
        b = trimesh.creation.box((10.0, 10.0, 16.0))
        b.apply_translation([x, 0, 8.0])
        b.visual.face_colors = c
        s.add_geometry(b, node_name=f"b{i}")
    path = tmp_path / "two.glb"
    path.write_bytes(s.export(file_type="glb"))
    sess = open_session(path, reference_path=BUNDLED_REFERENCE, work_dir=tmp_path / "w")
    rep, _ = analyse(sess, Params(magnets=0, bend=0, flat_base=0))
    tc = rep.toolchange
    assert tc["filaments"] == 2 and tc["nozzles"] == 2
    assert tc["changes"] == 0 and tc["switches"] > 10                         # one colour per nozzle
    assert tc["one_nozzle"]["changes"] > 10                                    # one nozzle must swap
    counts = {r["count"]: r for r in tc["by_count"]}
    assert counts[1]["changes"] == 0 and counts[2]["changes"] == 0


def test_best_value_is_the_count_before_the_biggest_jump() -> None:
    # The Manex Squirtle's table: the 4th colour (black eyes) adds 130 changes.
    rows = [{"count": i + 1, "changes": c} for i, c in enumerate([0, 0, 29, 159, 180, 287, 287])]
    assert best_value_count(rows) == 3
    assert best_value_count([{"count": 1, "changes": 0}, {"count": 2, "changes": 10}]) is None
    assert best_value_count([]) is None


def test_settings_without_nozzles_mean_one() -> None:
    assert nozzle_count({}) == 1 and nozzle_count({"nozzle_diameter": "0.4"}) == 1
    use = np.ones((5, 2), bool)
    best, single = plan(use, flush_matrix({}, 2), n_nozzles=nozzle_count({}))
    assert best.nozzle_of == single.nozzle_of == [1, 1] and best.changes == 5
