"""The printer's loaded filaments via beambam (pokefit/printer.py), against a fake `beambam status`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import trimesh

from pokefit.pipeline import BUNDLED_REFERENCE, Params, analyse, open_session
from pokefit.printer import PrinterError, ams_slots, beambam_command, read_state, track_switch

STATE = {"print": {
    "ams": {"ams": [
        {"id": "1", "info": "10001103", "tray": [{"id": "0", "tray_type": "PLA", "tray_color": "FFFFFFFF", "state": 11}]},
        {"id": "0", "info": "10001003", "tray": [
            {"id": "1", "tray_type": "PETG", "tray_color": "1414C8FF", "tray_sub_brands": "PETG HF", "state": 11},
            {"id": "0", "tray_type": "PETG", "tray_color": "C81414FF", "state": 11},
            {"id": "2", "state": 0},                                   # empty
            {"id": "3", "tray_type": "", "tray_color": "", "state": 11},   # no filament reported
        ]},
    ]},
    "vt_tray": {"id": "254", "tray_type": "TPU", "tray_color": "000000FF"},
}}


def test_slots_in_ams_order_with_labels() -> None:
    got = [(a.label, a.slot, a.type, a.colour) for a in ams_slots(STATE)]
    assert got == [("A1", 0, "PETG", "#C81414"), ("A2", 1, "PETG", "#1414C8"), ("B1", 4, "PLA", "#FFFFFF"),
                   ("Ext", 254, "TPU", "#000000")]


def test_nozzle_routing_from_the_unit_info() -> None:
    # Unit 0 feeds nozzle 0, unit 1 nozzle 1, the external spool is not an AMS unit.
    assert [a.extruder for a in ams_slots(STATE)] == [0, 0, 1, None]
    assert not track_switch(STATE)
    # With a Filament Track Switch (X2D report): bits 8-11 read 0xE and `device.fila_switch` is present.
    fts = json.loads(json.dumps(STATE))
    for unit in fts["print"]["ams"]["ams"]:
        unit["info"] = "10001E03"
    fts["print"]["device"] = {"fila_switch": {"in": [-1, -1], "info": 0, "out": [1, 0], "stat": 0}}
    assert [a.extruder for a in ams_slots(fts)] == [None] * 4
    assert track_switch(fts)


def _fake(tmp_path: Path, body: str, code: int = 0) -> str:
    script = tmp_path / "fake_beambam.py"
    script.write_text(f"import sys\nprint({body!r})\nsys.exit({code})\n", encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


def test_read_state_runs_beambam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake(tmp_path, "connecting...\n" + json.dumps(STATE)))
    assert beambam_command() is not None
    assert ams_slots(read_state())[0].label == "A1"
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake(tmp_path, "auth failed", 2))
    with pytest.raises(PrinterError, match="failed"):
        read_state()
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake(tmp_path, "no json here"))
    with pytest.raises(PrinterError, match="no JSON"):
        read_state()


def test_figure_colours_match_the_loaded_filaments(tmp_path: Path) -> None:
    s = trimesh.Scene()
    for i, (c, x) in enumerate((((200, 20, 20, 255), -6.0), ((20, 20, 200, 255), 6.0))):
        b = trimesh.creation.box((10.0, 10.0, 16.0))
        b.apply_translation([x, 0, 8.0])
        b.visual.face_colors = c
        s.add_geometry(b, node_name=f"b{i}")
    path = tmp_path / "two.glb"
    path.write_bytes(s.export(file_type="glb"))
    sess = open_session(path, reference_path=BUNDLED_REFERENCE, work_dir=tmp_path / "w")
    sess.ams = ams_slots(STATE)
    rep, _ = analyse(sess, Params(magnets=0, bend=0, flat_base=0))
    by_colour = {m["colour"].upper(): m for m in rep.import_map}
    # Red goes to A1 (target 1), blue to A2 (target 2): the AMS order, not the project's slots.
    assert by_colour["#C81414"]["target"] == 1 and by_colour["#1414C8"]["target"] == 2
