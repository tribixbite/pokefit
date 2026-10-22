"""The user's printer, read through beambam (github.com/tribixbite/beambam): which filaments are loaded.

beambam talks to Bambu printers over the LAN (signed MQTT, FTPS). Both projects are AGPL-3.0-or-later, but
pokefit does not import it (beambam lives in its own venv, and pokefit's engine also runs in the browser): it
runs its CLI as a separate program (`beambam status` prints the printer's full state report as JSON) and
reads the AMS block. Local version only; set `POKEFIT_BEAMBAM` to the beambam
command (e.g. `C:/git/beambam/.venv/Scripts/beambam.exe`) if it is not on PATH, and configure the
printer for beambam itself (`beambam init` / `~/.x2d/credentials`).

The loaded trays, in order — AMS A slots 1–4, B 1–4, … then the external spool — are what Bambu
Studio's "sync AMS" makes a project's filament list, so they serve as the slots the figure's colours
are matched to (`remap.match_slots`), and their colours can replace the figure's palette.

Dual-nozzle printers (H2D, X2D): each AMS unit's `info` hex word holds, in bits 8-11, the nozzle it is
wired to (0 right, 1 left), or 0xE when a Filament Track Switch is installed and routes any unit to either
nozzle (`device.fila_switch` is then present; bambuddy's reading, matching the user's X2D: 4 units, all 0xE).
With the switch the filament-to-nozzle split is free, which is what `toolchange.plan` assumes; the CLI slice
is identical with `has_filament_switcher` 0 or 1 (Manex Squirtle, 7 colours: 10.19 h / 145.7 g both).
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any

TIMEOUT_S = 30.0


@dataclass
class AmsSlot:
    slot: int          # global AMS slot index (unit × 4 + tray), the firmware's ams_mapping number; 254 = external
    label: str         # "A1".."D4", "Ext"
    colour: str        # "#RRGGBB"
    type: str          # "PETG", "PLA", ...
    name: str          # brand / sub-brand if the printer knows it
    extruder: int | None = None   # nozzle the unit feeds (firmware id: 0 right, 1 left); None = either / unknown
    filament_id: str = ""         # Bambu filament id of the spool (tray_info_idx, e.g. GFL99 = Generic PLA)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrinterError(RuntimeError):
    pass


def beambam_command() -> list[str] | None:
    """How to run beambam: `POKEFIT_BEAMBAM` (a command line) or `beambam` on PATH."""
    env = os.environ.get("POKEFIT_BEAMBAM")
    if env:
        if os.name != "nt":
            return shlex.split(env)
        # Windows: keep backslashes (non-POSIX split), then drop the quotes it leaves on each token.
        return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(env, posix=False)]
    found = shutil.which("beambam")
    return [found] if found else None


def read_state(timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """The printer's state report (`beambam status`)."""
    cmd = beambam_command()
    if cmd is None:
        raise PrinterError("beambam not found: install it (pip install -e <beambam>) or set POKEFIT_BEAMBAM")
    try:
        # UTF-8 both ways: Windows would otherwise decode beambam's output as cp1252 (mangled dashes).
        out = subprocess.run([*cmd, "status"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrinterError(f"beambam status failed: {exc}") from exc
    if out.returncode != 0:
        raise PrinterError(f"beambam status failed ({out.returncode}): {(out.stderr or out.stdout).strip()[-400:]}")
    text = out.stdout
    start = text.find("{")                   # log lines may come first
    try:
        data = json.loads(text[start:]) if start >= 0 else None
    except ValueError as exc:
        raise PrinterError(f"beambam status printed no JSON state: {exc}") from exc
    if not isinstance(data, dict):
        raise PrinterError("beambam status printed no JSON state")
    return data


def _colour(raw: Any) -> str:
    c = str(raw or "").lstrip("#")
    return f"#{c[:6].upper()}" if len(c) >= 6 else ""


def _unit_extruder(info: Any) -> int | None:
    """The nozzle an AMS unit feeds, from its `info` hex word (bits 8-11); None if routed by a track switch."""
    try:
        nib = (int(str(info), 16) >> 8) & 0xF
    except ValueError:
        return None
    return nib if nib in (0, 1) else None


# TODO: without a switch, pass each colour's allowed nozzle (its matched slot's `extruder`) to
# `toolchange.plan(allowed=...)`; needs the firmware-id -> Bambu `filament_map` (1-based) side mapping verified.
def track_switch(state: dict[str, Any]) -> bool:
    """Whether a Filament Track Switch is installed (any AMS unit can feed either nozzle)."""
    pr = state.get("print") or state
    return isinstance((pr.get("device") or {}).get("fila_switch"), dict)


def ams_slots(state: dict[str, Any]) -> list[AmsSlot]:
    """Loaded trays in the printer's report, AMS units in order, then the external spool(s)."""
    pr = state.get("print") or state
    out: list[AmsSlot] = []
    for unit in sorted(((pr.get("ams") or {}).get("ams") or []), key=lambda u: int(u.get("id", 0))):
        u = int(unit.get("id", 0))
        ext = _unit_extruder(unit.get("info"))
        for tray in sorted(unit.get("tray") or [], key=lambda t: int(t.get("id", 0))):
            colour = _colour(tray.get("tray_color"))
            if not colour or not tray.get("tray_type") or int(tray.get("state", 11) or 0) == 0:
                continue                      # empty slot
            t = int(tray.get("id", 0))
            out.append(AmsSlot(u * 4 + t, f"{chr(ord('A') + u)}{t + 1}", colour, str(tray.get("tray_type")),
                               str(tray.get("tray_sub_brands") or tray.get("tray_id_name") or ""), ext,
                               str(tray.get("tray_info_idx") or "")))
    ext = pr.get("vt_tray")
    for tray in ext if isinstance(ext, list) else [ext] if isinstance(ext, dict) else []:
        colour = _colour(tray.get("tray_color"))
        if colour and tray.get("tray_type"):
            out.append(AmsSlot(254, "Ext", colour, str(tray.get("tray_type")), str(tray.get("tray_sub_brands") or ""),
                               None, str(tray.get("tray_info_idx") or "")))
    return out


@dataclass
class Nozzle:
    id: int            # firmware extruder id: 0 right, 1 left (single-nozzle printers: 0)
    temp: float        # °C now
    target: float      # °C set (0 = off)


@dataclass
class PrintProgress:
    """What the printer is doing, from its state report (read-only; for the local "Printer" card)."""
    state: str         # gcode_state: IDLE, PREPARE, RUNNING, PAUSE, FINISH, FAILED, ...
    job: str           # the plate / project being (or last) printed
    percent: int
    layer: int
    layers: int
    remaining_min: int
    nozzles: list[Nozzle]
    bed: float
    bed_target: float
    error: int         # print_error (0 = none)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def print_progress(state: dict[str, Any]) -> PrintProgress:
    """Progress, temperatures and errors from a state report (`beambam status`)."""
    pr = state.get("print") or state
    nozzles: list[Nozzle] = []
    # Dual-nozzle printers: device.extruder.info[].temp packs current (bits 0-15) and target (16-31) °C
    # (Bambu Fleet Hub HTTP API v1.0.0, "device.extruder.info.temp").
    for e in ((pr.get("device") or {}).get("extruder") or {}).get("info") or []:
        packed = int(_num(e.get("temp")))
        nozzles.append(Nozzle(int(_num(e.get("id"))), float(packed & 0xFFFF), float((packed >> 16) & 0xFFFF)))
    if not nozzles and "nozzle_temper" in pr:
        nozzles.append(Nozzle(0, _num(pr.get("nozzle_temper")), _num(pr.get("nozzle_target_temper"))))
    return PrintProgress(
        state=str(pr.get("gcode_state") or "UNKNOWN"), job=str(pr.get("subtask_name") or ""),
        percent=int(_num(pr.get("mc_percent"))), layer=int(_num(pr.get("layer_num"))),
        layers=int(_num(pr.get("total_layer_num"))), remaining_min=int(_num(pr.get("mc_remaining_time"))),
        nozzles=sorted(nozzles, key=lambda n: n.id), bed=_num(pr.get("bed_temper")),
        bed_target=_num(pr.get("bed_target_temper")), error=int(_num(pr.get("print_error"))))


CAMERA_PORT = int(os.environ.get("POKEFIT_CAMERA_PORT", "8766"))   # beambam camera's default


class CameraProxy:
    """`beambam camera` as a child process: the printer's RTSPS feed re-served as MJPEG on
    http://127.0.0.1:<port>/cam.mjpeg (ffmpeg runs only while someone watches). Read-only.

    If something already answers on the port (a `beambam camera` the user started), it is used as is
    and left running. Stopping kills the whole process tree: on Windows the `beambam.exe` launcher runs
    Python, which runs ffmpeg, and terminating the launcher alone would orphan both.
    """

    def __init__(self, port: int = CAMERA_PORT) -> None:
        self.port = port
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/cam.mjpeg"

    def _listening(self) -> bool:
        import socket
        with socket.socket() as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self, timeout: float = 20.0) -> str:
        """Start (or reuse) the proxy; returns the MJPEG URL. Raises PrinterError if it does not come up."""
        if self._proc is not None and self._proc.poll() is None or self._listening():
            return self.url
        cmd = beambam_command()
        if cmd is None:
            raise PrinterError("beambam not found: install it (pip install -e <beambam>) or set POKEFIT_BEAMBAM")
        kwargs: dict[str, Any] = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt"
                                  else {"start_new_session": True})
        self._proc = subprocess.Popen([*cmd, "camera", "--bind", f"127.0.0.1:{self.port}"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
        import time
        t = time.time()
        while time.time() - t < timeout:
            if self._listening():
                return self.url
            if self._proc.poll() is not None:
                assert self._proc.stderr is not None
                err = self._proc.stderr.read().decode("utf-8", "replace").strip()[-400:]
                self._proc = None
                raise PrinterError(f"beambam camera stopped: {err or 'no output'}")
            time.sleep(0.2)
        self.stop()
        raise PrinterError(f"beambam camera did not start within {timeout:.0f} s")

    def stop(self) -> None:
        """Stop the proxy this object started (with its children). A proxy it did not start is left alone."""
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        if sys.platform == "win32":      # (sys.platform, not os.name: mypy narrows on it for os.killpg)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


# Bambu plate name -> its settings-key prefix (`<prefix>_plate_temp_initial_layer`), as beambam maps them.
PLATE_KEYS = {"Cool Plate": "cool", "Engineering Plate": "eng", "High Temp Plate": "hot",
              "Textured PEI Plate": "textured", "Supertack Plate": "supertack"}
PRINT_TIMEOUT_S = 600.0    # upload (FTPS) + live AMS check + start; a 20 MB file uploads in well under this


def print_file_summary(path: Path) -> dict[str, Any]:
    """What a sliced print file (.gcode.3mf) will print: per filament its preset, material and colour, and
    the plate with its first-layer bed temperature per filament (what beambam derives and checks too)."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        ps = json.loads(z.read("Metadata/project_settings.config"))
    bed = str(ps.get("curr_bed_type", ""))
    temps = ps.get(f"{PLATE_KEYS.get(bed, '')}_plate_temp_initial_layer") or []
    names, types, cols = (list(ps.get(k) or []) for k in ("filament_settings_id", "filament_type", "filament_colour"))
    return {"plate": bed, "filaments": [
        {"filament": i + 1, "preset": names[i] if i < len(names) else "", "type": types[i] if i < len(types) else "",
         "colour": cols[i] if i < len(cols) else "", "bed_temp": temps[i] if i < len(temps) else ""}
        for i in range(len(types))]}


ANALYZE_TIMEOUT_S = 60.0


def print_file_check(path: Path, timeout: float = ANALYZE_TIMEOUT_S) -> dict[str, Any]:
    """beambam's own reading of a sliced print file (`beambam analyze --json`: the G-code's tool changes, purge,
    nozzle swaps, its warnings and hints). Reads only the file: it never contacts the printer. The same
    analysis `beambam print --dry-run` gates on."""
    cmd = beambam_command()
    if cmd is None:
        raise PrinterError("beambam not found: install it (pip install -e <beambam>) or set POKEFIT_BEAMBAM")
    try:
        out = subprocess.run([*cmd, "analyze", "--json", str(path)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrinterError(f"beambam analyze failed: {exc}") from exc
    if out.returncode != 0:
        raise PrinterError(f"beambam analyze failed ({out.returncode}): {(out.stderr or out.stdout).strip()[-400:]}")
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise PrinterError(f"beambam analyze printed no JSON: {exc}") from exc
    plate, totals = d.get("plate") or {}, d.get("totals") or {}
    return {"seconds": plate.get("predicted_seconds"), "grams": plate.get("weight_g"),
            "purge_g": totals.get("flush_volume_g"), "nozzle_swaps": totals.get("nozzle_swaps"),
            "tool_changes": totals.get("tool_calls_total"),
            "nozzles": {str(f.get("id")): f.get("nozzle") for f in d.get("filaments") or []},
            "warnings": [str(w) for w in d.get("warnings") or []], "hints": [str(h) for h in d.get("hints") or []]}


def send_print(path: Path, slots: list[int], timeout: float = PRINT_TIMEOUT_S) -> str:
    """Upload `path` (a sliced .gcode.3mf) to the printer and start it, filament i from AMS slot slots[i]:
    `beambam print <file> --slot a,b,...`. CHANGES THE PRINTER: call only on the user's explicit action.
    beambam re-checks each slot's spool against the file (empty slot / wrong material refuse to start)."""
    cmd = beambam_command()
    if cmd is None:
        raise PrinterError("beambam not found: install it (pip install -e <beambam>) or set POKEFIT_BEAMBAM")
    if not slots or any(not 0 <= s <= 15 for s in slots):
        raise PrinterError(f"AMS slots must be 0..15, one per filament: {slots}")
    try:
        out = subprocess.run([*cmd, "print", str(path), "--slot", ",".join(map(str, slots))], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=timeout,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PrinterError(f"beambam print failed: {exc}") from exc
    text = (out.stdout + "\n" + out.stderr).strip()
    if out.returncode != 0:
        raise PrinterError(f"beambam print refused or failed ({out.returncode}): {text[-600:]}")
    return text[-1500:]
