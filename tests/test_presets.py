"""Bambu preset exports (.bbscfg / .bbsflmt / presets .zip / .json) added to the user's settings."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from pokefit.pipeline import BUNDLED_REFERENCE, Params, app_config, export_settings, open_session
from pokefit.presets import (PresetLibrary, SystemProfiles, add_filament, classify, load_settings_file,
                             material_of, printer_code, read_presets, resolve)
from pokefit.slicer import reference_settings

REF = reference_settings(BUNDLED_REFERENCE)          # X2D, 0.4 nozzle, 17 filament slots

FILAMENT = {"filament_settings_id": ["eSUN PETG Basic"], "from": "User",
            "inherits": "Generic PETG @BBL X2D 0.4 nozzle", "name": "eSUN PETG Basic",
            "nozzle_temperature": ["250", "250", "nil", "250", "250", "nil"],
            "filament_extruder_variant": ["Direct Drive Standard", "Direct Drive High Flow",
                                          "Direct Drive E3D High Flow", "Bowden Standard", "Bowden High Flow",
                                          "Bowden E3D High Flow"],
            "filament_density": ["1.29"], "filament_vendor": ["eSUN"], "version": "1.10.0.69"}
PROCESS = {"print_settings_id": "0.12 fine", "from": "User", "inherits": "0.12mm Fine @BBL X2D",
           "name": "0.12 fine", "layer_height": "0.12", "wall_loops": "3"}
OTHER_PROCESS = {**PROCESS, "name": "a1 draft", "print_settings_id": "a1 draft", "inherits": "0.28mm Draft @BBL A1"}
PRINTER = {"printer_settings_id": "my x2d", "from": "User", "inherits": "Bambu Lab X2D 0.4 nozzle",
           "name": "my x2d", "printer_notes": "tuned"}
OTHER_PRINTER = {**PRINTER, "name": "my a1", "printer_settings_id": "my a1", "inherits": "Bambu Lab A1 0.4 nozzle"}


def _zip(members: dict[str, object]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, json.dumps(data))
    return buf.getvalue()


def test_reads_every_export_kind() -> None:
    bbscfg = _zip({"bundle_structure.json": {"bundle_type": "printer config bundle", "printer_config": ["printer/p.json"],
                                             "filament_config": ["filament/f.json"],
                                             "process_config": ["process/q.json"]},
                   "printer/p.json": PRINTER, "filament/f.json": FILAMENT, "process/q.json": PROCESS})
    got = {(p.kind, p.name) for p in read_presets("My X2D.bbscfg", bbscfg)}
    assert got == {("printer", "my x2d"), ("filament", "eSUN PETG Basic"), ("process", "0.12 fine")}

    bbsflmt = _zip({"bundle_structure.json": {"bundle_type": "filament config bundle",
                                              "filament_vendor": [{"vendor": "eSUN",
                                                                   "filament_path": ["eSUN/eSUN PETG Basic.json"]}]},
                    "eSUN/eSUN PETG Basic.json": FILAMENT})
    assert [(p.kind, p.name) for p in read_presets("eSUN.bbsflmt", bbsflmt)] == [("filament", "eSUN PETG Basic")]

    plain = _zip({"0.12 fine.json": PROCESS, "eSUN PETG Basic.json": FILAMENT})
    assert sorted(p.kind for p in read_presets("Process presets.zip", plain)) == ["filament", "process"]
    assert read_presets("x.json", json.dumps(PRINTER).encode())[0].kind == "printer"


def test_rejects_non_presets() -> None:
    with pytest.raises(ValueError, match="not a Bambu preset"):
        read_presets("x.zip", b"not a zip")
    with pytest.raises(ValueError, match="No Bambu Studio presets"):
        read_presets("x.zip", _zip({"readme.json": {"hello": 1}}))
    with pytest.raises(ValueError, match="whole project"):
        read_presets("p.json", json.dumps(REF).encode())


def test_settings_file_routes_projects_and_presets(tmp_path: Path) -> None:
    lib = PresetLibrary()
    proj = tmp_path / "project.json"
    proj.write_text(json.dumps(REF), encoding="utf-8")
    assert load_settings_file(proj, lib) == REF and not lib.presets
    pre = tmp_path / "Filament presets.zip"
    pre.write_bytes(_zip({"f.json": FILAMENT}))
    assert load_settings_file(pre, lib) is None and lib.version == 1 and len(lib.of("filament")) == 1
    with pytest.raises(ValueError, match="expected a Bambu project"):
        load_settings_file(tmp_path / "x.stl", lib)


def test_filament_preset_becomes_a_slot() -> None:
    n = len(REF["filament_settings_id"])
    base = REF["filament_settings_id"].index("Generic PETG @BBL X2D 0.4 nozzle")
    out = add_filament(REF, FILAMENT, "eSUN PETG Basic", base)
    assert out["filament_settings_id"][-1] == "eSUN PETG Basic" and len(out["filament_settings_id"]) == n + 1
    k = len(REF["nozzle_temperature"]) // n
    base_temps = REF["nozzle_temperature"][base * k:(base + 1) * k]
    # "nil" keeps the base slot's value; the rest come from the preset (matched by extruder variant).
    assert out["nozzle_temperature"][-k:] == ["250", "250", base_temps[2], "250", "250", base_temps[5]]
    assert out["filament_density"][-1] == "1.29" and out["filament_vendor"][-1] == "eSUN"
    assert out["filament_type"][-1] == REF["filament_type"][base]
    for key, val in REF.items():          # every per-filament array grew by exactly one block
        if isinstance(val, list) and len(val) % n == 0 and len(val) >= n and key not in (
                "flush_volumes_matrix", "different_settings_to_system", "inherits_group"):
            assert len(out[key]) == len(val) // n * (n + 1), key
    e = len(REF["flush_volumes_matrix"]) // (n * n)
    assert len(out["flush_volumes_matrix"]) == e * (n + 1) ** 2
    m = [out["flush_volumes_matrix"][r * (n + 1):(r + 1) * (n + 1)] for r in range(n + 1)]
    assert m[n][n] == "0" and m[0][n] == m[0][base] or base == 0


def test_library_merges_and_skips_other_printers() -> None:
    lib = PresetLibrary()
    lib.add(read_presets("all.zip", _zip({"f.json": FILAMENT, "p.json": PROCESS, "o.json": OTHER_PROCESS,
                                          "m.json": PRINTER, "a.json": OTHER_PRINTER})))
    assert lib.processes(REF) == ["0.12 fine"]
    merged, notes = lib.merged(REF, "0.12 fine")
    by = {(x.kind, x.name): x for x in notes}
    assert merged["layer_height"] == "0.12" and merged["print_settings_id"] == "0.12 fine"
    assert merged["printer_settings_id"] == "my x2d" and merged["printer_notes"] == "tuned"
    assert by[("process", "a1 draft")].status == "skipped" and "A1" in by[("process", "a1 draft")].note
    assert by[("printer", "my a1")].status == "skipped"
    assert by[("filament", "eSUN PETG Basic")].status == "approximate"      # no system profiles here
    unchanged, _ = lib.merged(REF)                                           # no process chosen
    assert unchanged["layer_height"] == REF["layer_height"]


def test_inherits_resolve_against_system_profiles(tmp_path: Path) -> None:
    fil = tmp_path / "filament"
    fil.mkdir()
    (fil / "Generic PETG @BBL X2D 0.4 nozzle.json").write_text(json.dumps(
        {"type": "filament", "name": "Generic PETG @BBL X2D 0.4 nozzle", "inherits": "fdm_filament_pet",
         "hot_plate_temp": ["75"]}), encoding="utf-8")
    (fil / "fdm_filament_pet.json").write_text(json.dumps(
        {"type": "filament", "name": "fdm_filament_pet", "filament_type": ["PETG"], "hot_plate_temp": ["70"],
         "cool_plate_temp": ["0"]}), encoding="utf-8")
    lib = PresetLibrary(system=SystemProfiles(tmp_path))
    lib.add(read_presets("f.json", json.dumps(FILAMENT).encode()))
    values, exact = resolve(lib.of("filament")[0], lib.system)
    assert exact and values["hot_plate_temp"] == ["75"] and values["filament_type"] == ["PETG"]
    assert values["filament_density"] == ["1.29"]
    merged, notes = lib.merged(REF)
    assert notes[0].status == "used" and merged["hot_plate_temp"][-1] == "75"


def test_name_helpers() -> None:
    assert printer_code("Generic PLA @BBL X2D 0.4 nozzle") == "X2D"
    assert printer_code("Bambu Lab X1 Carbon 0.4 nozzle") == "X1 Carbon"
    assert material_of({}, "Generic PETG HF @BBL X2D 0.4 nozzle") == "PETG"
    assert material_of({}, "Bambu PLA Basic @BBL X2D") == "PLA"
    assert material_of({"filament_type": ["TPU"]}, "x") == "TPU"
    assert classify({"type": "machine"}) == "printer" and classify({}, "process/x.json") == "process"


def test_session_exports_with_an_imported_preset(tmp_path: Path) -> None:
    import trimesh
    stl = tmp_path / "box.stl"
    trimesh.creation.box((20, 20, 20)).export(stl)
    s = open_session(stl, reference_path=BUNDLED_REFERENCE, work_dir=tmp_path / "w")
    lib = PresetLibrary()
    lib.add(read_presets("all.zip", _zip({"f.json": FILAMENT, "p.json": PROCESS})))
    s.presets = lib
    cfg = app_config(None, REF, "ref.json", lib)
    imported = [x for x in cfg["slots"] if x.get("imported")]
    assert [x["name"] for x in imported] == ["eSUN PETG Basic"] and cfg["processes"][-1] == "0.12 fine"
    p = Params(slot=int(imported[0]["slot"]), process="0.12 fine")
    st = export_settings(s, p)
    assert st is not None and st["filament_settings_id"] == ["eSUN PETG Basic"] and st["layer_height"] == "0.12"
    assert s.slot_matches(p) == [] or all(m.target <= 17 for m in s.slot_matches(p))


def test_plate_the_filament_cannot_use_is_reported() -> None:
    from pokefit.presets import plate_problem
    no_supertack = {**FILAMENT, "name": "no st", "supertack_plate_temp": ["0"]}
    out = add_filament({**REF, "curr_bed_type": "Supertack Plate"}, no_supertack, "no st", 0)
    assert "Supertack Plate" in plate_problem(out, len(out["filament_settings_id"]) - 1)
    ok = add_filament({**REF, "curr_bed_type": "Supertack Plate"}, {**FILAMENT, "supertack_plate_temp": ["35"]}, "ok", 0)
    assert plate_problem(ok, len(ok["filament_settings_id"]) - 1) == ""
    assert plate_problem({**ok, "curr_bed_type": "Unknown"}, 0) == ""
