import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import two_footed, write_inline_3mf
from pokefit import server


@pytest.fixture(scope="module")
def base_url():
    httpd = server.make_server("127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _req(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    r = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(base: str, path: str, obj: dict[str, Any]) -> tuple[int, Any]:
    code, body = _req(base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return code, json.loads(body)


@pytest.fixture(scope="module")
def loaded(base_url: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    f = write_inline_3mf(tmp_path_factory.mktemp("srv") / "critter.3mf",
                         two_footed(foot=9.0, gap=8.0, leg_h=12.0, body_h=18.0), np.eye(4))
    code, body = _req(base_url + "/api/load", f.read_bytes(), {"X-Filename": "critter.3mf"})
    assert code == 200, body
    return json.loads(body)


def test_config(base_url: str) -> None:
    code, body = _req(base_url + "/api/config")
    cfg = json.loads(body)
    assert code == 200 and cfg["default_steel"] == "D20"
    assert {w["key"] for w in cfg["steels"]} >= {"D20", "D15", "M6", "M8"}


def test_load_returns_report(loaded: dict[str, Any]) -> None:
    rep = loaded["report"]
    assert rep["fit"]["scale"] > 0 and len(rep["ball"]["profile_z"]) == 160
    assert loaded["file"] == "critter.3mf"


def test_analyse_and_preview(base_url: str, loaded: dict[str, Any]) -> None:
    code, rep = _post(base_url, "/api/analyse", {"session": loaded["session"],
                                                 "params": {"figure_scale": 0.4, "magnets": False}})
    assert code == 200 and rep["fit"]["scale"] == 0.4 and rep["pockets"] == []
    code, glb = _req(f"{base_url}/api/preview.glb?session={loaded['session']}")
    assert code == 200 and glb[:4] == b"glTF"


def test_export_and_download(base_url: str, loaded: dict[str, Any]) -> None:
    code, res = _post(base_url, "/api/export", {"session": loaded["session"], "params": {"steel": "US1/4x1"}})
    assert code == 200, res
    names = [f["name"] for f in res["files"]]
    assert "critter_pokefit_60.3mf" in names
    url = next(f["url"] for f in res["files"] if f["name"].endswith(".3mf"))
    code, data = _req(base_url + url)
    assert code == 200 and data[:2] == b"PK"


def test_errors(base_url: str) -> None:
    code, body = _post(base_url, "/api/analyse", {"session": "nope", "params": {}})
    assert code == 404 and "load a file" in body["error"]
    code, body = _req(base_url + "/api/load", b"not a zip", {"X-Filename": "x.3mf"})
    assert code == 400 and "Cannot read" in json.loads(body)["error"]
    code, body = _req(base_url + "/api/load", b"abc", {"X-Filename": "x.txt"})
    assert code == 400
    code, _ = _req(base_url + "/api/download?file=../pyproject.toml")
    assert code == 404


def test_slice_job_reports(base_url: str, loaded: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The slice endpoint runs as a job; a missing slicer is reported, not raised."""
    monkeypatch.setattr("pokefit.slicer.DEFAULT_EXE", Path("C:/nonexistent/bambu.exe"))
    monkeypatch.setattr("pokefit.slicer.shutil.which", lambda _: None)
    code, res = _post(base_url, "/api/slice", {"session": loaded["session"], "params": {"magnets": False}, "x10": False})
    assert code == 200
    for _ in range(600):
        code, job = _req(f"{base_url}/api/job?id={res['job']}")
        job = json.loads(job)
        if job["state"] != "running":
            break
        time.sleep(0.2)
    assert job["state"] == "done"
    assert job["result"]["off"]["ok"] is False and "not found" in job["result"]["off"]["error"]


def test_reference_project_sets_slots(base_url: str, tmp_path: Path) -> None:
    """Uploading the user's own Bambu project makes its filament slots the import targets."""
    import trimesh

    from pokefit.model3mf import read_project, write_project

    box = trimesh.creation.box([10, 10, 10])
    src = read_project(write_inline_3mf(tmp_path / "b.3mf", box, np.eye(4)))
    settings = {"filament_colour": ["#112233", "#445566"], "filament_settings_id": ["A", "B"],
                "filament_type": ["PLA", "PETG"], "printer_settings_id": "My printer"}
    proj = write_project(src, "box", 1, src.default_object().parts, tmp_path / "mine.3mf", settings=settings)
    try:
        code, body = _req(base_url + "/api/reference", proj.read_bytes(), {"X-Filename": "mine.3mf"})
        assert code == 200
        cfg = json.loads(body)
        assert cfg["reference"] == "mine.3mf" and [s["colour"] for s in cfg["slots"]] == ["#112233", "#445566"]
        assert cfg["printer"] == "My printer" and cfg["engine"] == "local"
        code, _ = _req(base_url + "/api/reference", b"not a zip", {"X-Filename": "junk.3mf"})
        assert code == 400
    finally:
        server._reference, server._reference_name = None, ""      # module state: do not leak into other tests


def test_start_print_refuses_a_plan_that_changed_since_it_was_shown(base_url: str,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    # The page sends the digest of the plan on screen; the server plans again just before the start, so accepting
    # beambam's warnings covers only the ones the user saw. No printer: beambam is faked.
    import zipfile
    from pokefit import printer
    d = server.OUT_DIR / "_test_digest"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "fig.gcode.3mf"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("Metadata/project_settings.config", json.dumps({
            "curr_bed_type": "Supertack Plate", "filament_type": ["PLA"], "filament_colour": ["#FFFFFF"],
            "filament_settings_id": ["Generic PLA"]}))
    plans = [{"filaments": [{"filament": 1, "nozzle": "left", "tray": {"tray_id": 3}}], "errors": [],
              "warnings": ["filament 1: colour"]}]

    def plan(path: Path, slots: list[int], timeout: float = 0) -> dict[str, Any]:
        p = json.loads(json.dumps(plans[-1]))
        p["digest"] = printer.plan_digest(p, path, slots)
        return p
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(printer, "print_plan", plan)
    monkeypatch.setattr(server, "print_plan", plan)
    monkeypatch.setattr(server, "print_file_check", lambda p: {})
    monkeypatch.setattr(server, "send_print", lambda *a, **k: sent.append(k) or "accepted")
    rel = "_test_digest/fig.gcode.3mf"
    try:
        code, prev = _post(base_url, "/api/printer/print", {"file": rel, "slots": [3]})
        assert code == 200, prev
        digest = prev["mapping"]["digest"]
        start = {"file": rel, "slots": [3], "confirm": True, "accept_warnings": True}
        code, body = _post(base_url, "/api/printer/print", start)
        assert code == 400 and "plan_digest" in body["error"]
        plans.append({**plans[0], "warnings": [*plans[0]["warnings"], "the printer has a Cool Plate"]})
        code, body = _post(base_url, "/api/printer/print", {**start, "plan_digest": digest})
        assert code == 409 and "changed since this check" in body["error"] and not sent
        plans.pop()
        code, body = _post(base_url, "/api/printer/print", {**start, "plan_digest": digest})
        assert code == 200 and body["job"]
        for _ in range(100):
            if sent:
                break
            time.sleep(0.02)
        assert sent and sent[0]["accept_warnings"] is True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
