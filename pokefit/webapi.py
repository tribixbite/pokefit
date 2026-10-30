"""The pokefit engine as called from the browser (Pyodide in a Web Worker).

Mirrors the local server's JSON API (server.py) without HTTP: the worker
passes JSON strings and file bytes in, and gets JSON strings and bytes back.
Sessions live in memory; files are written to Pyodide's in-memory filesystem.
Bambu slicing needs the desktop slicer and is only available in the local
version (`python -m pokefit serve`).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .pipeline import (Params, Session, analyse, app_config, export, open_session, pick, pieces_of_colour,
                       prepare_paint, preview_faces, preview_glb, safe_stem, washer_table)
from .presets import PresetLibrary, load_settings_file

WORK = Path("/tmp/pokefit")
_sessions: dict[str, Session] = {}
_reference: dict | None = None          # the user's own Bambu project settings, if they loaded one
_reference_name = ""
_presets = PresetLibrary()              # imported Bambu presets (no system profiles in the browser)
_ball_model: Path | None = None         # the user's copy of the ball model (for the washer-seat shell)


def _dump(o: Any) -> str:
    import numpy as np

    def default(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"not JSON serialisable: {type(x).__name__}")

    return json.dumps(o, default=default)


def _raw(data: Any) -> bytes:
    """Bytes from a JS Uint8Array / ArrayBuffer proxy (or Python bytes in tests)."""
    return data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)


def _session(sid: str) -> Session:
    if sid not in _sessions:
        raise KeyError("Unknown session — load a file first.")
    return _sessions[sid]


def config() -> str:
    return _dump({**app_config(None, _reference, _reference_name, _presets), "engine": "browser"})


def load(name: str, data: Any, params_json: str = "{}", quick: bool = False) -> str:
    """Open an uploaded .3mf / .stl / .obj / .glb (bytes) and analyse it with the UI's settings (simplified as
    they ask), so the first analysis is the one shown. `quick`: without the printability checks (`analyse`)."""
    params = Params.from_dict(json.loads(params_json))
    folder = WORK / uuid.uuid4().hex[:8]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(name).name
    path.write_bytes(_raw(data))
    s = open_session(path, work_dir=folder / "work", simplify=params.simplify)
    if _reference is not None:
        s.reference = _reference
    s.presets = _presets
    _sessions[s.id] = s
    rep, _ = analyse(s, params, printing=not quick)
    return _dump({"session": s.id, "file": path.name, "report": rep.to_dict()})


def set_reference(name: str, data: Any) -> str:
    """Use the user's own Bambu project (filament slots, printer settings) or add exported presets."""
    global _reference, _reference_name
    path = WORK / "reference" / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(data))
    ref = load_settings_file(path, _presets)
    if ref is not None:
        _reference, _reference_name = ref, path.name
        for s in _sessions.values():
            s.reference = _reference
    return config()


def set_ball_model(name: str, data: Any) -> str:
    """Accept the user's copy of the ball model so exports include the washer-seat shell."""
    global _ball_model
    from .pipeline import _shell_mesh
    path = WORK / "ball" / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw(data))
    _shell_mesh(path)                     # raises if it has no 0_ball_shell object
    _ball_model = path
    return _dump({"ok": True, "name": path.name})


def run_analyse(sid: str, params_json: str, quick: bool = False) -> str:
    """Analyse with the UI's settings; `quick`: the printability checks only if cached (`analyse(printing=)`)."""
    rep, _ = analyse(_session(sid), Params.from_dict(json.loads(params_json)), printing=not quick)
    return _dump(rep.to_dict())


def washers(sid: str, params_json: str) -> str:
    return _dump(washer_table(_session(sid), Params.from_dict(json.loads(params_json))))


def _regions(regions_json: str) -> tuple[str, float] | None:
    reg = json.loads(regions_json)
    return (str(reg.get("mode", "part")), float(reg.get("angle", 15.0))) if isinstance(reg, dict) else None


def preview(sid: str, params_json: str, regions_json: str = "null") -> bytes:
    """GLB of the figure (one uncoloured mesh per part); `regions_json` {mode, angle}: the unclipped parts."""
    return preview_glb(_session(sid), Params.from_dict(json.loads(params_json)), _regions(regions_json))


def faces(sid: str, params_json: str, regions_json: str = "null") -> str:
    """The preview's face colours (`preview_faces`): {signature, colours, parts}."""
    return _dump(preview_faces(_session(sid), Params.from_dict(json.loads(params_json)), _regions(regions_json)))


def pick_face(sid: str, params_json: str, req_json: str) -> str:
    """The triangle under a tap: req = {point, normal, mode, angle, radius, stroke?, mirror?, grow?, level?,
    stamp_axes?} (figure frame, scale 1)."""
    req = json.loads(req_json)
    return _dump(pick(_session(sid), Params.from_dict(json.loads(params_json)), req["point"], req.get("normal"),
                      str(req.get("mode", "part")), float(req.get("angle", 15.0)), float(req.get("radius", 0.0)),
                      req.get("stroke"), bool(req.get("mirror", False)), float(req.get("grow", 0.0)),
                      int(req.get("level", 0)),
                      [float(v) for v in req["stamp_axes"]] if req.get("stamp_axes") else None))


def paint_prepare(sid: str, params_json: str, req_json: str) -> str:
    """Build the tap structures (`pipeline.prepare_paint`) when painting starts: req = {mode, angle, mirror?}."""
    req = json.loads(req_json)
    return _dump(prepare_paint(_session(sid), Params.from_dict(json.loads(params_json)),
                               str(req.get("mode", "part")), float(req.get("angle", 15.0)),
                               bool(req.get("mirror", False))))


def colour_pieces(sid: str, params_json: str, req_json: str) -> str:
    """The connected patches of one colour (`pipeline.pieces_of_colour`): req = {filament}."""
    req = json.loads(req_json)
    return _dump(pieces_of_colour(_session(sid), Params.from_dict(json.loads(params_json)), int(req["filament"])))


def run_export(sid: str, params_json: str) -> tuple[str, list[tuple[str, bytes]]]:
    """(result JSON, [(file name, bytes)]) — the files are offered as downloads by the page."""
    s = _session(sid)
    out = WORK / "out" / uuid.uuid4().hex[:8] / safe_stem(s.project.path.name)
    res = export(s, Params.from_dict(json.loads(params_json)), out, ball_model=_ball_model)
    files = [Path(p) for p in (res["figure_3mf"], res["import_3mf"], res["colour_stl"], res["shell_stl"], res["report"]) if p]
    return _dump(res), [(f.name, f.read_bytes()) for f in files]
