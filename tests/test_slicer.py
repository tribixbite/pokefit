import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import write_inline_3mf
from pokefit.model3mf import Part, read_project, write_project
from pokefit.slicer import (adapt_settings, default_slot, find_exe, grid_positions, read_result, reference_settings,
                            slice_3mf)

REF = Path(__file__).resolve().parent.parent / "pokeball60goodPrinted.3mf"


def fake_ref(n_ref: int = 3, n_ext: int = 2) -> dict:
    return {
        "filament_settings_id": ["PLA a", "PETG b", "PLA c"][:n_ref],
        "filament_type": ["PLA", "PETG", "PLA"][:n_ref],
        "filament_colour": ["#FF0000", "#00FF00", "#0000FF"][:n_ref],
        "nozzle_temperature": ["220", "250", "221"][:n_ref],
        "filament_extruder_variant": [f"f{i}v{j}" for i in range(n_ref) for j in range(2)],
        "different_settings_to_system": ["print"] + [f"fil{i}" for i in range(n_ref)] + ["printer"],
        "flush_volumes_matrix": [str(e * 1000 + 10 * a + b) for e in range(n_ext) for a in range(n_ref) for b in range(n_ref)],
        "flush_multiplier": ["1"] * n_ext,
        "printable_area": ["0x0", "256x0", "256x256", "0x256"],
        "layer_height": "0.16",
        "enable_support": "0",
        "support_type": "normal(auto)",
    }


def test_adapt_settings_shapes() -> None:
    ref = fake_ref()
    out = adapt_settings(ref, ["#FF1010", "#0000EE"], slot=default_slot(ref), supports=True)
    assert out["filament_settings_id"] == ["PETG b", "PETG b"]
    assert out["nozzle_temperature"] == ["250", "250"]
    assert out["filament_extruder_variant"] == ["f1v0", "f1v1", "f1v0", "f1v1"]
    assert out["different_settings_to_system"] == ["print", "fil1", "fil1", "printer"]
    assert out["filament_colour"] == ["#FF1010", "#0000EE"]
    assert out["printable_area"] == ref["printable_area"] and out["layer_height"] == "0.16"
    # Flush: 2 extruders x 2 x 2; red-ish -> slot 0, blue-ish -> slot 2.
    m = np.array(out["flush_volumes_matrix"], float).reshape(2, 2, 2)
    assert m[0, 0, 1] == 2 and m[0, 1, 0] == 20 and m[1, 0, 1] == 1002
    assert m[0, 0, 0] == 0 and m[1, 1, 1] == 0
    assert out["enable_support"] == "1" and out["support_type"] == "normal(auto)"   # auto kept as chosen
    manual = adapt_settings({**ref, "support_type": "normal(manual)"}, ["#FF1010"], 1, supports=True)
    assert manual["support_type"] == "tree(auto)"
    assert adapt_settings(ref, ["#FF1010"], 1, supports=False)["enable_support"] == "0"


@pytest.mark.skipif(not REF.exists(), reason="reference project not present")
def test_adapt_real_reference_lengths() -> None:
    ref = reference_settings(REF)
    n_ref = len(ref["filament_settings_id"])
    out = adapt_settings(ref, ["#E05028", "#161616"], default_slot(ref), supports=False)
    for k, v in ref.items():
        if isinstance(v, list) and k not in ("different_settings_to_system", "inherits_group",
                                             "flush_volumes_matrix") and len(v) % n_ref == 0 and len(v) >= n_ref:
            assert len(out[k]) == len(v) // n_ref * 2, k
    n_ext = len(ref["flush_volumes_matrix"]) // (n_ref * n_ref)
    assert len(out["flush_volumes_matrix"]) == n_ext * 4


def test_missing_exe_warns(tmp_path: Path) -> None:
    r = slice_3mf(tmp_path / "x.3mf", exe=tmp_path / "nope.exe")
    assert not r.ok and "not found" in r.error


def test_read_result(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(json.dumps({
        "return_code": 0, "error_string": "Success.",
        "sliced_plates": [{"total_predication": 100.5, "warning_message": "w",
                           "feature_type_times": {"Support": 7.0, "Support interface": 3.0, "Outer wall": 50},
                           "filaments": [{"total_used_g": 1.5}, {"total_used_g": 0.5}]}]}))
    r = read_result(tmp_path, 1)
    assert r.ok and r.seconds == 100.5 and r.grams == 2.0 and r.support_seconds == 10.0 and r.warnings == ["w"]
    (tmp_path / "result.json").write_text(json.dumps({"return_code": -50, "error_string": "Nothing to be sliced"}))
    bad = read_result(tmp_path, 1)
    assert not bad.ok and "Nothing" in bad.error


def test_grid_positions() -> None:
    pos = grid_positions((30, 30), 10, (256, 256))
    assert pos is not None and len(pos) == 10
    xs = sorted({round(x, 3) for x, _ in pos})
    assert min(np.diff(xs)) >= 38 - 1e-9
    assert all(15 <= x <= 241 and 15 <= y <= 241 for x, y in pos)
    assert grid_positions((200, 200), 10, (256, 256)) is None


@pytest.mark.slow
@pytest.mark.skipif(find_exe() is None or not REF.exists(), reason="Bambu Studio or reference not present")
def test_slice_t_shape_supports(tmp_path: Path) -> None:
    stem = trimesh.creation.box([6, 6, 12]).apply_translation([0, 0, 6])
    bar = trimesh.creation.box([30, 6, 3]).apply_translation([0, 0, 13.4])
    t = trimesh.boolean.union([stem, bar], engine="manifold")
    proj = read_project(write_inline_3mf(tmp_path / "t.3mf", t, np.eye(4)))
    ref = reference_settings(REF)
    part = Part("t", 0, np.asarray(t.vertices), np.asarray(t.faces))
    res = {}
    for sup in (False, True):
        f = write_project(proj, "t", 1, [part], tmp_path / f"t_{sup}.3mf",
                          settings=adapt_settings(ref, ["#FF0000"], default_slot(ref), sup))
        res[sup] = slice_3mf(f, work_dir=tmp_path / f"out_{sup}")
        assert res[sup].ok, res[sup].error
    assert res[False].support_seconds == 0
    assert res[True].support_seconds > 0
    assert res[True].seconds > res[False].seconds
