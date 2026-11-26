"""The user's project against the printer: offer to bring its filament list (and plate) in line with what is loaded.

Bambu Studio offers the same after it connects to a printer ("Sync filaments from AMS"): the project's filament i
becomes the AMS tray in position i (A1, A2, … D4), with the spool's preset, material and colour. pokefit matches
each figure colour to a loaded spool already when it slices (`Session.print_settings`), but the project itself —
its slot list (the export's "Import geometry only" file, the material picker) and its plate — stays as loaded
until the user takes the printer's.

Pure functions; the engines (webapi for the site, server for the local version) keep the reference project and
the spools and call these.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .printer import AmsSlot
from .remap import delta_e

# Bambu Studio's `curr_bed_type` values (PrintConfig.cpp), in its own order.
BED_TYPES = ("Cool Plate", "Engineering Plate", "High Temp Plate", "Textured PEI Plate", "Supertack Plate")
EXTERNAL = 254           # tray ids from here on are the external spool holders
SAME_COLOUR = 6.0       # ΔE under which two colours are "the same" for this comparison (spool readings vary)


def compare(ref: dict[str, Any] | None, ams: list[AmsSlot] | None) -> dict[str, Any] | None:
    """Project slot i against the i-th loaded spool: {"differs", "rows", "same", "project", "printer", "plate"},
    or None without a project or spools. A row: {"index", "project": {type, colour, name} | None,
    "spool": {label, type, colour, name} | None, "same"}."""
    if not ref or not ams:
        return None
    names = [str(x) for x in ref.get("filament_settings_id", [])]
    types = [str(x) for x in ref.get("filament_type", [""] * len(names))]
    cols = [str(x) for x in ref.get("filament_colour", [""] * len(names))]
    spools = ams_trays(ams)
    rows = []
    for i in range(max(len(names), len(spools))):
        p = {"type": types[i] if i < len(types) else "", "colour": cols[i] if i < len(cols) else "",
             "name": names[i]} if i < len(names) else None
        s = spools[i] if i < len(spools) else None
        sd = {"label": s.label, "type": s.type, "colour": s.colour, "name": s.name} if s else None
        same = bool(p and s and p["type"].upper() == s.type.upper()
                    and _close(p["colour"], s.colour))
        rows.append({"index": i + 1, "project": p, "spool": sd, "same": same})
    # Slots past the spools are left alone by `adopt`, so they never count as a difference.
    compared = rows[:len(spools)]
    return {"differs": any(not r["same"] for r in compared), "rows": rows,
            "same": sum(r["same"] for r in compared), "project": len(names), "printer": len(spools),
            "plate": ref.get("curr_bed_type") or None}


def ams_trays(ams: list[AmsSlot]) -> list[AmsSlot]:
    """The AMS trays in order (A1 … D4, then AMS HT): the external spools (254 / 255) are no project slot."""
    return sorted((a for a in ams if a.slot < EXTERNAL), key=lambda a: a.slot)


def _close(a: str, b: str) -> bool:
    if not (a.startswith("#") and b.startswith("#")) or len(a) < 7 or len(b) < 7:
        return False
    try:
        return delta_e(a[:7], b[:7]) <= SAME_COLOUR
    except ValueError:
        return False


def adopt(ref: dict[str, Any], ams: list[AmsSlot], system: Any = None, bed_type: str | None = None) -> dict[str, Any]:
    """`ref` with filament i = the i-th loaded spool (its preset, material and colour); slots past the spools stay
    as they are, so objects assigned to them still have a filament. `bed_type`: also the plate (BED_TYPES)."""
    from .presets import ams_filaments
    from .slicer import _filaments_from, default_slot

    if bed_type is not None and bed_type not in BED_TYPES:
        raise ValueError(f"plate must be one of {', '.join(BED_TYPES)}")
    spools = ams_trays(ams)
    out, src, _names = ams_filaments(ref, system, [(a.filament_id, a.type) for a in spools], default_slot(ref))
    n_ref = len(ref.get("filament_settings_id", []))
    # Keep the project's own slots past the spools (their index in `out` is unchanged: ams_filaments only appends).
    src += list(range(len(spools), n_ref))
    colours = [a.colour for a in spools] + [str(c) for c in list(ref.get("filament_colour", []))[len(spools):n_ref]]
    st = _filaments_from(out, src, colours)
    if bed_type is not None:
        st["curr_bed_type"] = bed_type
    return st


def updated_project(original: Path | None, settings: dict[str, Any], name: str) -> tuple[str, bytes]:
    """The user's project file with `settings` (a copy: its objects, plates and everything else as they were);
    without a .3mf to copy, the settings as a project .json pokefit and Bambu Studio's presets loader read back."""
    stem = Path(name).stem or "project"
    if original is not None and original.suffix.lower() == ".3mf" and original.is_file():
        import io
        buf = io.BytesIO()
        with zipfile.ZipFile(original) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "Metadata/project_settings.config":
                    data = json.dumps(settings, indent=4).encode()
                zout.writestr(item, data)
        return f"{stem}_printer.3mf", buf.getvalue()
    return f"{stem}_printer.json", json.dumps({**settings, "from": "project"}, indent=1).encode()
