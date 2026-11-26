"""Printing through beambam: its daemon's HTTP API (`POKEFIT_BEAMBAM_URL`) and the CLI's `ams map` / `--yes`.

A fake daemon (same routes and replies as beambam's `serve_http`: `/state`, `/print/plan`, `/print`,
`/print/jobs/<id>`, `/print/upload`) and a fake CLI; no printer, and nothing here needs trimesh.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pokefit import printer
from pokefit.printer import PrinterError, print_plan, read_state, send_print

STATE = {"print": {"gcode_state": "IDLE", "ams": {"ams": []}}}
PLAN_WARN = {"filaments": [{"filament": 1, "type": "PLA", "colour": "#00AE42", "nozzle": "left",
                            "tray": {"tray_id": 13, "label": "D2 (slot 13)", "type": "PLA", "colour": "#057748"},
                            "errors": [], "warnings": ["colour #00AE42 vs loaded #057748 (ΔE 38)"]}],
             "errors": [], "warnings": ["filament 1: colour #00AE42 vs loaded #057748 (ΔE 38)"], "notes": [],
             "ok": True, "needs_confirmation": True}


class _Daemon(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, object]] = []
    job_state = "accepted"

    def log_message(self, *_: object) -> None:
        return

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.seen.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/state":
            self._send(200, STATE)
        elif self.path == "/print/jobs/j1":
            self._send(200, {"id": "j1", "state": self.job_state, "done": True,
                             "message": "the printer accepted the print" if self.job_state == "accepted"
                             else "the printer rejected the print: bad file"})
        else:
            self._send(404, {"error": "no such route"})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/print/upload":
            self.seen.append(("POST", self.path, self.headers.get("X-Filename")))
            self._send(200, {"ok": True, "file": "/remote/uploads/1_" + str(self.headers.get("X-Filename"))})
            return
        body = json.loads(raw or b"{}")
        self.seen.append(("POST", self.path, body))
        if self.path == "/print/plan":
            self._send(200, getattr(self, "plan", PLAN_WARN))
        elif self.path == "/print":
            if not body.get("accept_warnings"):
                self._send(409, {"ok": False, "needs_confirmation": True,
                                 "error": "the mapping has warnings the user has not accepted"})
            else:
                self._send(202, {"ok": True, "job": {"id": "j1"}, "poll": "/print/jobs/j1"})
        else:
            self._send(404, {"error": "no such route"})


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Daemon.seen = []
    _Daemon.job_state = "accepted"
    monkeypatch.setenv("POKEFIT_BEAMBAM_URL", f"http://127.0.0.1:{srv.server_address[1]}/")
    monkeypatch.setenv("POKEFIT_BEAMBAM_TOKEN", "t0k")
    monkeypatch.setattr(printer, "POLL_S", 0.01)
    yield _Daemon
    srv.shutdown()


def test_state_comes_from_the_daemon(daemon) -> None:
    assert read_state() == STATE
    assert daemon.seen == [("GET", "/state", "Bearer t0k")]


def test_plan_then_print_with_the_users_ok(daemon, tmp_path: Path) -> None:
    f = tmp_path / "fig_auto.gcode.3mf"
    f.write_bytes(b"PK")
    plan = print_plan(f, [13])
    assert plan["warnings"] and not plan["errors"]
    # Without the user's OK beambam refuses; the daemon's reason reaches the user.
    with pytest.raises(PrinterError, match="warnings the user has not accepted"):
        send_print(f, [13])
    assert send_print(f, [13], accept_warnings=True) == "the printer accepted the print"
    posts = [b for m, p, b in daemon.seen if p == "/print"]
    assert posts[-1] == {"file": str(f.resolve()), "slot": [13], "confirm": True, "accept_warnings": True}


def test_a_rejected_print_raises(daemon, tmp_path: Path) -> None:
    f = tmp_path / "fig.gcode.3mf"
    f.write_bytes(b"PK")
    daemon.job_state = "rejected"
    with pytest.raises(PrinterError, match="rejected"):
        send_print(f, [0], accept_warnings=True)


def test_a_remote_daemon_gets_the_file_uploaded(daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(printer, "_daemon_is_local", lambda: False)
    f = tmp_path / "fig.gcode.3mf"
    f.write_bytes(b"PK\x03\x04")
    print_plan(f, [0])
    assert ("POST", "/print/upload", "fig.gcode.3mf") in daemon.seen
    plan_req = next(b for m, p, b in daemon.seen if p == "/print/plan")
    assert plan_req["file"] == "/remote/uploads/1_fig.gcode.3mf"


def test_unreachable_daemon_says_how_to_start_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POKEFIT_BEAMBAM_URL", "http://127.0.0.1:9")
    with pytest.raises(PrinterError, match="beambam daemon --http"):
        read_state(timeout=2)


def _fake_cli(tmp_path: Path, stdout: str, code: int) -> str:
    """A fake `beambam` that records its argv and prints `stdout`."""
    log = tmp_path / "argv.json"
    script = tmp_path / "fake_beambam.py"
    script.write_text("import json, sys\n"
                      f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                      f"print({stdout!r})\nsys.exit({code})\n", encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


def _argvs(tmp_path: Path) -> list[list[str]]:
    return [json.loads(x) for x in (tmp_path / "argv.json").read_text().splitlines()]


def test_cli_plan_is_ams_map_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEFIT_BEAMBAM_URL", raising=False)
    old = {k: v for k, v in PLAN_WARN.items() if k not in ("errors", "warnings")}   # beambam ≤ 1.6.0 shape
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake_cli(tmp_path, json.dumps(old), 2))  # exit 2 = warnings
    f = tmp_path / "fig.gcode.3mf"
    plan = print_plan(f, [13, 4])
    assert _argvs(tmp_path)[0] == ["ams", "map", str(f), "--slot", "13,4", "--json"]
    assert plan["warnings"] == ["filament 1: colour #00AE42 vs loaded #057748 (ΔE 38)"]


def test_cli_plan_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEFIT_BEAMBAM_URL", raising=False)
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake_cli(tmp_path, "ams map: fig.gcode.3mf has no sliced plate 1", 1))
    with pytest.raises(PrinterError, match="no sliced plate"):
        print_plan(tmp_path / "fig.gcode.3mf", [0])


def test_cli_print_passes_yes_only_with_the_users_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEFIT_BEAMBAM_URL", raising=False)
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake_cli(tmp_path, "print accepted by the printer", 0))
    f = tmp_path / "fig.gcode.3mf"
    send_print(f, [1, 2])
    send_print(f, [1, 2], accept_warnings=True)
    a, b = _argvs(tmp_path)
    assert a == ["print", str(f), "--slot", "1,2"] and b == a + ["--yes"]


def _sliced_file(path: Path, groups: dict[int, str]) -> Path:
    """A minimal .gcode.3mf: project settings for len(groups) filaments, slice_info with each one's group_id."""
    import zipfile
    n = len(groups)
    ps = {"curr_bed_type": "Supertack Plate", "filament_type": ["PLA"] * n, "filament_colour": ["#FFFFFF"] * n,
          "filament_settings_id": ["Generic PLA"] * n, "filament_map": ["1"] * n}
    info = "<config><plate>" + "".join(f'<filament id="{i}" type="PLA" group_id="{g}"/>' for i, g in groups.items()) + \
        "</plate></config>"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Metadata/project_settings.config", json.dumps(ps))
        z.writestr("Metadata/slice_info.config", info)
    return path


def test_summary_names_the_nozzles_the_file_was_sliced_on(tmp_path: Path) -> None:
    # Bambu's own split lands in slice_info's group_id (0 = left); project_settings' filament_map stays all 1.
    got = printer.print_file_summary(_sliced_file(tmp_path / "f.gcode.3mf", {1: "0", 2: "1", 3: "0,1"}))
    assert [f["nozzle"] for f in got["filaments"]] == [1, 2, None]


def test_a_changed_plan_is_refused_before_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI's `ams map` has no digest: pokefit makes one, and confirm_plan plans again and compares.
    monkeypatch.delenv("POKEFIT_BEAMBAM_URL", raising=False)
    f = _sliced_file(tmp_path / "fig.gcode.3mf", {1: "0"})
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake_cli(tmp_path, json.dumps(PLAN_WARN), 2))
    shown = print_plan(f, [13])
    assert isinstance(shown["digest"], str) and shown["digest"] == print_plan(f, [13])["digest"]
    assert printer.confirm_plan(f, [13], shown["digest"])["warnings"] == shown["warnings"]
    assert print_plan(f, [12])["digest"] != shown["digest"]              # other slots, another plan
    # A warning that was not on screen (the plate on the printer changed since): refused, with the new plan.
    later = {**PLAN_WARN, "warnings": [*PLAN_WARN["warnings"], "the printer has a Cool Plate, the file is for Supertack"]}
    monkeypatch.setenv("POKEFIT_BEAMBAM", _fake_cli(tmp_path, json.dumps(later), 2))
    with pytest.raises(printer.PlanChanged) as e:
        printer.confirm_plan(f, [13], shown["digest"])
    assert e.value.plan["warnings"] == later["warnings"]


def test_a_local_daemon_checks_its_own_digest_at_start(daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Daemon, "plan", {**PLAN_WARN, "digest": "beambam1"}, raising=False)
    f = tmp_path / "fig.gcode.3mf"
    f.write_bytes(b"PK")
    plan = print_plan(f, [13])
    assert plan["beambam_digest"] == "beambam1" and plan["digest"] != "beambam1"
    send_print(f, [13], accept_warnings=True, beambam_digest=plan["beambam_digest"])
    assert [b for m, p, b in daemon.seen if p == "/print"][-1]["plan_digest"] == "beambam1"
    # A remote daemon reads a fresh upload each time (its digest names that path): not handed back.
    monkeypatch.setattr(printer, "_daemon_is_local", lambda: False)
    assert "beambam_digest" not in print_plan(f, [13])
