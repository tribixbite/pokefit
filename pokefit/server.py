"""Local web server: JSON API over the pipeline + the static web UI (web/dist).

stdlib only (ThreadingHTTPServer). Sessions live in memory; uploads and
outputs go under .pokefit_cache/ and pokefit_out/ next to the repo.

API (all JSON unless noted):
  GET  /api/config                         washers, filament slots, defaults, slicer availability
  POST /api/load        body = file bytes, header X-Filename      -> {session, report}
  POST /api/load_path   {path}                                     -> {session, report}
  POST /api/analyse     {session, params}                          -> report
  POST /api/washers     {session, params}                          -> [washer rows]
  GET  /api/preview.glb ?session&object&rx&ry                      -> GLB (figure at scale 1)
  POST /api/preview     {session, params, regions?: {mode, angle}} -> GLB (with colorize paint / regions)
  POST /api/pick        {session, params, point, normal, mode, angle} -> {part, face, faces, ...}
  POST /api/slice       {session, params, x10}                     -> {job}
  GET  /api/job         ?id                                        -> {state, result|error}
  POST /api/export      {session, params}                          -> {files: [{name, url}], ...}
  GET  /api/download    ?file                                      -> file bytes
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .pipeline import (CACHE_DIR, DEFAULT_REFERENCE, ROOT, Params, Session, analyse, app_config, export,
                       input_checks,
                       open_session, pick, preview_glb, safe_stem, slice_variants, washer_table)
from .gltf import GLTF_SUFFIXES
from .presets import PresetLibrary, default_system_profiles, load_settings_file
from .slicer import find_exe
from .washer import CATALOGUE, DEFAULT_WASHER

WEB_DIST = ROOT / "web" / "dist"
OUT_DIR = ROOT / "pokefit_out"
UPLOAD_DIR = CACHE_DIR / "uploads"
MAX_UPLOAD = 300 * 1024 * 1024

_sessions: dict[str, Session] = {}
_locks: dict[str, threading.Lock] = {}
_jobs: dict[str, dict[str, Any]] = {}
_registry_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _session(sid: str | None) -> tuple[Session, threading.Lock]:
    if not sid or sid not in _sessions:
        raise ApiError(404, "Unknown session — load a file first.")
    return _sessions[sid], _locks[sid]


def _register(s: Session) -> None:
    with _registry_lock:
        _sessions[s.id] = s
        _locks[s.id] = threading.Lock()


def _params(body: dict[str, Any]) -> Params:
    try:
        return Params.from_dict(body.get("params") or {})
    except (TypeError, ValueError) as exc:
        raise ApiError(400, f"bad params: {exc}") from exc


def _load(path: Path) -> dict[str, Any]:
    try:
        s = open_session(path)
    except Exception as exc:  # noqa: BLE001 — bad zip, bad XML, no objects...: all are "unreadable file"
        traceback.print_exc()
        raise ApiError(400, f"Cannot read {path.name}: {type(exc).__name__}: {exc}") from exc
    if _reference is not None:
        s.reference = _reference
    s.presets = _presets
    _register(s)
    with _locks[s.id]:
        rep, _ = analyse(s, Params())
    return {"session": s.id, "file": path.name, "report": rep.to_dict()}


# The user's own Bambu project (filament slots + printer settings), once they load one.
_reference: dict | None = None
_reference_name = ""
# Bambu presets the user imported (.bbscfg / .bbsflmt / presets .zip); inherits resolve against the
# installed Bambu Studio's system profiles.
_presets = PresetLibrary(system=default_system_profiles())


def _config() -> dict[str, Any]:
    exe = find_exe()
    from .imageto3d import Comfy
    return {**app_config(str(exe) if exe else None, _reference, _reference_name, _presets), "engine": "local",
            "photos": Comfy(COMFY_URL).alive(timeout=1.5)}


def _run_job(job_id: str, fn: Any) -> None:
    try:
        _jobs[job_id].update(state="done", result=fn())
    except Exception as exc:  # noqa: BLE001 — reported to the UI, never crashes the server
        _jobs[job_id].update(state="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _start_job(fn: Any, jid: str | None = None) -> str:
    jid = jid or uuid.uuid4().hex[:10]
    _jobs.setdefault(jid, {}).update(state="running")
    threading.Thread(target=_run_job, args=(jid, fn), daemon=True).start()
    return jid


# Photos -> 3D: a local ComfyUI with Pixal3D (pokefit/imageto3d.py, AGENTS.md "Photos to 3D").
COMFY_URL = os.environ.get("POKEFIT_COMFY", "http://127.0.0.1:8188")


class Handler(BaseHTTPRequestHandler):
    server_version = "pokefit/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console: errors only
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(fmt, *args)

    # -------------------------------------------------------------- helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, default=_jsonable).encode("utf-8"), "application/json")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, f"File too large ({n / 1e6:.0f} MB > {MAX_UPLOAD / 1e6:.0f} MB).")
        return self.rfile.read(n) if n else b""

    def _json_body(self) -> dict[str, Any]:
        try:
            data = json.loads(self._body() or b"{}")
        except json.JSONDecodeError as exc:
            raise ApiError(400, f"bad JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "JSON object expected")
        return data

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path.startswith("/api/"):
                handler = getattr(self, f"{method}_{url.path[5:].replace('.', '_').replace('/', '_')}", None)
                if handler is None:
                    raise ApiError(404, f"no API {method.upper()} {url.path}")
                handler(q)
            elif method == "get":
                self._static(url.path)
            else:
                raise ApiError(405, "method not allowed")
        except ApiError as exc:
            self._json({"error": str(exc)}, exc.status)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("get")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("post")

    # -------------------------------------------------------------- static
    def _static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        f = (WEB_DIST / rel).resolve()
        if not str(f).startswith(str(WEB_DIST.resolve())) or not f.is_file():
            if "." not in rel.rsplit("/", 1)[-1]:
                f = WEB_DIST / "index.html"
            else:
                raise ApiError(404, "not found")
        if not f.exists():
            raise ApiError(500, "web UI not built: run `bun install && bun run build` in web/")
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        self._send(200, f.read_bytes(), ctype)

    # -------------------------------------------------------------- API
    def get_config(self, q: dict[str, str]) -> None:
        self._json(_config())

    def post_load(self, q: dict[str, str]) -> None:
        name = Path(unquote(self.headers.get("X-Filename") or "upload.3mf")).name
        if Path(name).suffix.lower() not in (".3mf", ".stl", ".obj", *GLTF_SUFFIXES):
            raise ApiError(400, "Supported files: .3mf, .stl, .obj, .glb, .gltf")
        data = self._body()
        if not data:
            raise ApiError(400, "empty upload")
        d = UPLOAD_DIR / uuid.uuid4().hex[:8]
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_bytes(data)
        self._json(_load(path))

    def post_reference(self, q: dict[str, str]) -> None:
        """Use the user's own Bambu project (filament slots, printer settings) or add exported presets."""
        global _reference, _reference_name
        name = Path(unquote(self.headers.get("X-Filename") or "project.3mf")).name
        d = UPLOAD_DIR / "reference"
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_bytes(self._body())
        try:
            ref = load_settings_file(path, _presets)
        except Exception as exc:  # noqa: BLE001 — any unreadable project is a 400
            raise ApiError(400, f"Cannot read {name}: {exc}") from exc
        if ref is not None:
            _reference, _reference_name = ref, name
            with _registry_lock:
                for s in _sessions.values():
                    s.reference = ref
        self._json(_config())

    def post_photos(self, q: dict[str, str]) -> None:
        """Photos of a figure (base64, by view) -> a coloured GLB via ComfyUI, then loaded like a file.

        Runs as a job (minutes); its `progress` says what it is doing."""
        from .imageto3d import VIEWS, photos_to_glb
        body = self._json_body()
        views = body.get("views") or {}
        if not isinstance(views, dict) or not views.get("front"):
            raise ApiError(400, "a front photo is required")
        d = UPLOAD_DIR / f"photos_{uuid.uuid4().hex[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        photos: dict[str, Path] = {}
        for v in VIEWS:
            if views.get(v):
                raw = str(views[v]).split(",", 1)[-1]          # a data: URL or bare base64
                try:
                    (d / f"{v}.png").write_bytes(base64.b64decode(raw))
                except ValueError as exc:
                    raise ApiError(400, f"{v} photo is not base64 image data") from exc
                photos[v] = d / f"{v}.png"
        name = safe_stem(str(body.get("name") or "figure")) or "figure"
        jid = uuid.uuid4().hex[:10]
        _jobs[jid] = {"state": "running", "progress": "starting"}

        def log(text: str) -> None:
            _jobs[jid]["progress"] = text.strip()

        def run() -> dict[str, Any]:
            rep = photos_to_glb(photos, d / f"{name}.glb", url=COMFY_URL, seed=int(body.get("seed") or 0),
                                mirror_right=bool(body.get("mirror_right")), log=log)
            log("fitting the figure")
            return {**_load(d / f"{name}.glb"), "photos": rep}
        self._json({"job": _start_job(run, jid)})

    def post_load_path(self, q: dict[str, str]) -> None:
        p = Path(str(self._json_body().get("path", ""))).expanduser()
        if not p.is_file():
            raise ApiError(404, f"no such file: {p}")
        self._json(_load(p))

    def post_analyse(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        with lock:
            rep, _ = analyse(s, params)
        self._json(rep.to_dict())

    def post_washers(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        with lock:
            input_checks(s, params)   # warm the per-object caches; the table itself only reads them
            s.unit_parts(params)
        self._json(washer_table(s, params))

    def get_preview_glb(self, q: dict[str, str]) -> None:
        s, lock = _session(q.get("session"))
        try:
            params = Params(object_id=q.get("object") or None, rotate_x=float(q.get("rx", 0)),
                            rotate_y=float(q.get("ry", 0)), rotate_z=float(q.get("rz", 0)))
        except ValueError as exc:
            raise ApiError(400, f"bad rotation: {exc}") from exc
        with lock:
            glb = preview_glb(s, params)
        self._send(200, glb, "model/gltf-binary")

    def post_preview(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        reg = body.get("regions")
        regions = (str(reg.get("mode", "part")), float(reg.get("angle", 15.0))) if isinstance(reg, dict) else None
        with lock:
            glb = preview_glb(s, params, regions)
        self._send(200, glb, "model/gltf-binary")

    def post_pick(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        try:
            point = [float(v) for v in body["point"]][:3]
            normal = [float(v) for v in body["normal"]][:3] if body.get("normal") else None
            mode, angle = str(body.get("mode", "part")), float(body.get("angle", 15.0))
            with lock:
                res = pick(s, params, point, normal, mode, angle, float(body.get("radius", 0.0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(400, f"bad pick: {exc}") from exc
        self._json(res)

    def post_slice(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        x10 = bool(body.get("x10", True))

        with lock:
            input_checks(s, params)
            s.unit_parts(params)

        def run() -> dict[str, Any]:
            return slice_variants(s, params, x10=x10)
        self._json({"job": _start_job(run)})

    def get_job(self, q: dict[str, str]) -> None:
        job = _jobs.get(q.get("id", ""))
        if job is None:
            raise ApiError(404, "unknown job")
        self._json(job)

    def post_export(self, q: dict[str, str]) -> None:
        body = self._json_body()
        s, lock = _session(body.get("session"))
        params = _params(body)
        out = OUT_DIR / safe_stem(s.project.path.name)
        with lock:
            res = export(s, params, out)
        files = [Path(p) for p in (res["figure_3mf"], res["import_3mf"], res["shell_stl"], res["report"]) if p]
        res["files"] = [{"name": f.name, "path": str(f),
                         "url": "/api/download?file=" + quote(str(f.relative_to(OUT_DIR)).replace("\\", "/"))}
                        for f in files]
        self._json(res)

    def get_download(self, q: dict[str, str]) -> None:
        f = (OUT_DIR / q.get("file", "")).resolve()
        if not str(f).startswith(str(OUT_DIR.resolve())) or not f.is_file():
            raise ApiError(404, "no such output file")
        self._send(200, f.read_bytes(), "application/octet-stream",
                   {"Content-Disposition": f'attachment; filename="{f.name}"'})


def _jsonable(o: Any) -> Any:
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and
    # requests then reach whichever one Windows picks: refuse instead ("address in use").
    allow_reuse_address = sys.platform != "win32"


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return _Server((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    httpd = make_server(host, port)
    url = f"http://{host}:{port}/"
    print(f"pokefit UI at {url}  (Ctrl+C to stop)")
    if not (WEB_DIST / "index.html").exists():
        print("note: web UI not built yet — run `bun install && bun run build` in web/")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
