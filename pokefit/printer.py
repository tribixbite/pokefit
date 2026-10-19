"""The user's printer, read through beambam (github.com/tribixbite/beambam): which filaments are loaded.

beambam talks to Bambu printers over the LAN (signed MQTT, FTPS). It is AGPL-licensed, so pokefit does
not import it: it runs its CLI as a separate program (`beambam status` prints the printer's full
state report as JSON) and reads the AMS block. Local version only; set `POKEFIT_BEAMBAM` to the beambam
command (e.g. `C:/git/beambam/.venv/Scripts/beambam.exe`) if it is not on PATH, and configure the
printer for beambam itself (`beambam init` / `~/.x2d/credentials`).

The loaded trays, in order — AMS A slots 1–4, B 1–4, … then the external spool — are what Bambu
Studio's "sync AMS" makes a project's filament list, so they serve as the slots the figure's colours
are matched to (`remap.match_slots`), and their colours can replace the figure's palette.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
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


def ams_slots(state: dict[str, Any]) -> list[AmsSlot]:
    """Loaded trays in the printer's report, AMS units in order, then the external spool(s)."""
    pr = state.get("print") or state
    out: list[AmsSlot] = []
    for unit in sorted(((pr.get("ams") or {}).get("ams") or []), key=lambda u: int(u.get("id", 0))):
        u = int(unit.get("id", 0))
        for tray in sorted(unit.get("tray") or [], key=lambda t: int(t.get("id", 0))):
            colour = _colour(tray.get("tray_color"))
            if not colour or not tray.get("tray_type") or int(tray.get("state", 11) or 0) == 0:
                continue                      # empty slot
            t = int(tray.get("id", 0))
            out.append(AmsSlot(u * 4 + t, f"{chr(ord('A') + u)}{t + 1}", colour, str(tray.get("tray_type")),
                               str(tray.get("tray_sub_brands") or tray.get("tray_id_name") or "")))
    ext = pr.get("vt_tray")
    for tray in ext if isinstance(ext, list) else [ext] if isinstance(ext, dict) else []:
        colour = _colour(tray.get("tray_color"))
        if colour and tray.get("tray_type"):
            out.append(AmsSlot(254, "Ext", colour, str(tray.get("tray_type")), str(tray.get("tray_sub_brands") or "")))
    return out
