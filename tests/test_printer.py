"""The printer's loaded filaments via beambam (pokefit/printer.py), against a fake `beambam status`."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import trimesh

from pokefit.pipeline import BUNDLED_REFERENCE, Params, analyse, open_session
from pokefit.printer import (CameraProxy, PrinterError, ams_slots, beambam_command, print_file_check,
                             print_file_summary,
                             print_progress, read_state, send_print, track_switch)

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


def test_print_progress_unpacks_both_nozzles() -> None:
    report = {"print": {
        "gcode_state": "RUNNING", "subtask_name": "squirtle", "mc_percent": 42, "layer_num": 80,
        "total_layer_num": 190, "mc_remaining_time": 72, "bed_temper": 69.5, "bed_target_temper": 70.0,
        "print_error": 0,
        # X2D: current temperature in bits 0-15, target in 16-31.
        "device": {"extruder": {"info": [{"id": 1, "temp": 27}, {"id": 0, "temp": (250 << 16) | 248}]}},
    }}
    p = print_progress(report)
    assert (p.state, p.job, p.percent, p.layer, p.layers, p.remaining_min) == ("RUNNING", "squirtle", 42, 80, 190, 72)
    assert [(n.id, n.temp, n.target) for n in p.nozzles] == [(0, 248.0, 250.0), (1, 27.0, 0.0)]
    assert (p.bed, p.bed_target, p.error) == (69.5, 70.0, 0)
    # A single-nozzle report (no device.extruder): the top-level nozzle fields.
    one = print_progress({"print": {"gcode_state": "IDLE", "nozzle_temper": 30, "nozzle_target_temper": 0}})
    assert [(n.temp, n.target) for n in one.nozzles] == [(30.0, 0.0)] and one.layers == 0


def _alive(pid: int) -> bool:
    if sys.platform == "win32":     # (os.kill on Windows would TERMINATE the process)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_camera_proxy_starts_and_kills_its_whole_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake `beambam camera --bind HOST:PORT` that, like the real one, has a child (ffmpeg) and listens.
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "fake_camera.py"
    script.write_text(
        "import socket, subprocess, sys, time\n"
        "host, port = sys.argv[sys.argv.index('--bind') + 1].split(':')\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
        "s = socket.socket(); s.bind((host, int(port))); s.listen()\n"
        "time.sleep(120)\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{script}"')
    cam = CameraProxy(_free_port())
    url = cam.start(timeout=20)
    assert url == f"http://127.0.0.1:{cam.port}/cam.mjpeg"
    assert cam.start() == url                       # already running: reused
    child = int(pidfile.read_text())
    assert _alive(child)
    cam.stop()
    t = time.time()
    while _alive(child) and time.time() - t < 10:
        time.sleep(0.2)
    assert not _alive(child)                        # the grandchild (ffmpeg) went too
    assert not cam._listening()


def test_camera_proxy_reports_a_failed_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # beambam exits through sys.exit(message): the message goes to stderr.
    script = tmp_path / "fake_fail.py"
    script.write_text("import sys\nsys.exit('credentials missing - provide --ip --code --serial')\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{script}"')
    with pytest.raises(PrinterError, match="credentials missing"):
        CameraProxy(_free_port()).start(timeout=20)


def _print_file(path: Path) -> Path:
    import zipfile
    ps = {"curr_bed_type": "Supertack Plate", "supertack_plate_temp_initial_layer": ["45", "70"],
          "filament_type": ["PLA", "PETG"], "filament_colour": ["#8ECAD8", "#B49478"],
          "filament_settings_id": ["Generic PLA @BBL X2D 0.4 nozzle", "Generic PETG @BBL X2D 0.4 nozzle"]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Metadata/project_settings.config", json.dumps(ps))
    return path


def test_print_file_summary(tmp_path: Path) -> None:
    got = print_file_summary(_print_file(tmp_path / "f.gcode.3mf"))
    assert got["plate"] == "Supertack Plate"
    assert [(f["filament"], f["type"], f["colour"], f["bed_temp"]) for f in got["filaments"]] == [
        (1, "PLA", "#8ECAD8", "45"), (2, "PETG", "#B49478", "70")]


def test_send_print_passes_one_slot_per_filament(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake beambam that records its arguments (no printer is touched).
    log = tmp_path / "args.json"
    script = tmp_path / "fake_print.py"
    script.write_text("import json, sys\n"
                      f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                      "print('start_print queued: f.gcode.3mf (slot=8,10)')\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{script}"')
    f = _print_file(tmp_path / "f.gcode.3mf")
    out = send_print(f, [8, 10])
    assert json.loads(log.read_text()) == ["print", str(f), "--slot", "8,10"]
    assert "queued" in out
    with pytest.raises(PrinterError, match="0..15"):
        send_print(f, [8, 99])
    fail = tmp_path / "fake_refuse.py"
    fail.write_text("import sys\nsys.exit('AMS slot 10 is empty')\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{fail}"')
    with pytest.raises(PrinterError, match="slot 10 is empty"):
        send_print(f, [8, 10])


def test_print_file_check_reads_beambams_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake `beambam analyze --json` (the real one reads only the file; nothing here touches a printer).
    log = tmp_path / "args.json"
    report = {"plate": {"predicted_seconds": 6030, "weight_g": 19.62},
              "filaments": [{"id": 1, "nozzle": 0}, {"id": 2, "nozzle": 1}],
              "totals": {"nozzle_swaps": 0, "tool_calls_total": 150, "flush_volume_g": 0.0},
              "warnings": ["purge over 10 g"], "hints": ["put filament 2 on the other nozzle"]}
    script = tmp_path / "fake_analyze.py"
    script.write_text("import json, sys\n"
                      f"open({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                      f"print({json.dumps(json.dumps(report))})\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{script}"')
    f = _print_file(tmp_path / "f.gcode.3mf")
    got = print_file_check(f)
    assert json.loads(log.read_text()) == ["analyze", "--json", str(f)]
    assert got == {"seconds": 6030, "grams": 19.62, "purge_g": 0.0, "nozzle_swaps": 0, "tool_changes": 150,
                   "nozzles": {"1": 0, "2": 1}, "warnings": ["purge over 10 g"],
                   "hints": ["put filament 2 on the other nozzle"]}
    fail = tmp_path / "fake_fail.py"
    fail.write_text("import sys\nsys.exit('not a .gcode.3mf')\n", encoding="utf-8")
    monkeypatch.setenv("POKEFIT_BEAMBAM", f'"{sys.executable}" "{fail}"')
    with pytest.raises(PrinterError, match="not a .gcode.3mf"):
        print_file_check(f)
