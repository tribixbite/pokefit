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
  POST /api/slice       {session, params, x10}                     -> {job}
  GET  /api/job         ?id                                        -> {state, result|error}
  POST /api/export      {session, params}                          -> {files: [{name, url}], ...}
  GET  /api/download    ?file                                      -> file bytes
"""
from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .pipeline import (CACHE_DIR, DEFAULT_REFERENCE, ROOT, Params, Session, analyse, export, input_checks,
                       open_session, preview_glb, safe_stem, slice_variants, washer_table)
from .slicer import default_slot, filament_slots, find_exe
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
    except TypeError as exc:
        raise ApiError(400, f"bad params: {exc}") from exc


def _load(path: Path) -> dict[str, Any]:
    try:
        s = open_session(path)
    except Exception as exc:  # noqa: BLE001 — bad zip, bad XML, no objects...: all are "unreadable file"
        traceback.print_exc()
        raise ApiError(400, f"Cannot read {path.name}: {type(exc).__name__}: {exc}") from exc
    _register(s)
    with _locks[s.id]:
        rep, _ = analyse(s, Params())
    return {"session": s.id, "file": path.name, "report": rep.to_dict()}


def _config() -> dict[str, Any]:
    from .slicer import reference_settings
    ref = reference_settings(DEFAULT_REFERENCE) if DEFAULT_REFERENCE.exists() else None
    return {
        "washers": [w.to_dict() for w in CATALOGUE.values()],
        "default_washer": DEFAULT_WASHER,
        "ball_scales": [0.6, 1.0],
        "params": Params().__dict__,
        "slots": filament_slots(ref) if ref else [],
        "default_slot": default_slot(ref) if ref else 0,
        "reference": DEFAULT_REFERENCE.name,
        "printer": (ref or {}).get("printer_settings_id", ""),
        "process": (ref or {}).get("print_settings_id", ""),
        "slicer": str(find_exe()) if find_exe() else None,
    }


def _run_job(job_id: str, fn: Any) -> None:
    try:
        _jobs[job_id].update(state="done", result=fn())
    except Exception as exc:  # noqa: BLE001 — reported to the UI, never crashes the server
        _jobs[job_id].update(state="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _start_job(fn: Any) -> str:
    jid = uuid.uuid4().hex[:10]
    _jobs[jid] = {"state": "running"}
    threading.Thread(target=_run_job, args=(jid, fn), daemon=True).start()
    return jid


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
        name = Path(self.headers.get("X-Filename") or "upload.3mf").name
        if Path(name).suffix.lower() not in (".3mf", ".stl", ".obj"):
            raise ApiError(400, "Supported files: .3mf, .stl, .obj")
        data = self._body()
        if not data:
            raise ApiError(400, "empty upload")
        d = UPLOAD_DIR / uuid.uuid4().hex[:8]
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_bytes(data)
        self._json(_load(path))

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
        params = Params(object_id=q.get("object") or None, rotate_x=int(q.get("rx", 0)), rotate_y=int(q.get("ry", 0)))
        with lock:
            glb = preview_glb(s, params)
        self._send(200, glb, "model/gltf-binary")

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
        files = [Path(p) for p in (res["figure_3mf"], res["shell_stl"], res["report"]) if p]
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


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


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
