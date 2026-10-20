"""Bambu Studio preset exports: filament, process and printer presets added to the user's settings.

Bambu Studio's File > Export > Export Preset Bundle writes five kinds of file (verified against
BambuStudio `src/slic3r/GUI/CreatePresetsDialog.cpp`, `ExportConfigsDialog::archive_*`):

- ``<printer>.bbscfg`` — printer config bundle: a zip with ``bundle_structure.json``
  (``bundle_type: "printer config bundle"``, ``printer_config`` / ``filament_config`` /
  ``process_config`` lists of member paths under ``printer/``, ``filament/``, ``process/``).
- ``<filament>.bbsflmt`` — filament config bundle: ``bundle_structure.json`` with
  ``filament_vendor: [{vendor, filament_path: ["<vendor>/<name>.json", ...]}]``.
- ``Printer presets.zip`` / ``Filament presets.zip`` / ``Process presets.zip`` — plain zips of
  ``<name>.json`` presets.

A user preset is a DIFF: it names a system preset in ``inherits`` and holds only the keys the user
changed. The system presets ship with Bambu Studio (``resources/profiles/BBL/{filament,process,machine}``);
when they are installed (the local version) the chain is resolved exactly, otherwise (the browser) the
preset is laid over the closest slot of the user's project and reported as approximate.

How each kind is used:
- filament: appended to the project's filament list, so it can be picked as the material
  ("Filament for every colour"). The import file's slot mapping still uses only the project's real
  slots — a preset is not a spool in the AMS.
- process: selectable (``Params.process``); its keys replace the project's print settings.
- printer: applied when it is for the project's printer model; another printer's preset is skipped
  (a project 3MF saved for that printer is the way to switch printers).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PRESET_SUFFIXES = (".bbscfg", ".bbsflmt", ".zip", ".json")
KINDS = ("filament", "process", "printer")
BUNDLE_STRUCTURE = "bundle_structure.json"
# Identity/bookkeeping keys of a preset file: never copied into project settings.
META_KEYS = {"name", "inherits", "from", "version", "type", "instantiation", "setting_id", "is_custom_defined",
             "compatible_printers", "compatible_printers_condition", "compatible_prints",
             "compatible_prints_condition", "filament_id", "base_id", "user_id", "updated_time"}
# Material families, longest first so "PETG-CF" wins over "PETG" and "PETG" over "PET".
MATERIALS = ("PETG-CF", "PETG", "PLA-CF", "PLA", "PET-CF", "PET", "ABS", "ASA", "TPU", "PA-CF", "PA", "PC",
             "PVA", "HIPS", "PPS", "PP", "PE")


@dataclass
class Preset:
    kind: str                  # filament | process | printer
    name: str
    data: dict[str, Any]       # the file as exported (a diff over `inherits`)
    source: str                # file it came from

    @property
    def inherits(self) -> str:
        return str(self.data.get("inherits", "") or "")


@dataclass
class PresetNote:
    """What became of one preset, for the UI."""
    kind: str
    name: str
    status: str                # "used" | "approximate" | "skipped"
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def is_project_settings(data: dict[str, Any]) -> bool:
    """A full project config (Metadata/project_settings.config or a saved copy), not a preset diff."""
    return data.get("from") == "project" or (
        "print_settings_id" in data and "printer_settings_id" in data and isinstance(
            data.get("filament_settings_id"), list) and len(data) > 200)


def classify(data: dict[str, Any], member: str = "") -> str | None:
    """Kind of a preset dict: its `type`, else the *_settings_id key it carries, else its bundle folder."""
    t = str(data.get("type", ""))
    if t in ("filament", "process"):
        return t
    if t in ("machine", "printer"):
        return "printer"
    for key, kind in (("filament_settings_id", "filament"), ("print_settings_id", "process"),
                      ("printer_settings_id", "printer")):
        if key in data:
            return kind
    top = member.replace("\\", "/").split("/", 1)[0].lower()
    return {"filament": "filament", "process": "process", "printer": "printer", "machine": "printer"}.get(top)


def _preset(data: Any, member: str, source: str, kind: str | None = None) -> Preset | None:
    if not isinstance(data, dict):
        return None
    kind = kind or classify(data, member)
    if kind is None:
        return None
    name = str(data.get("name") or Path(member).stem)
    return Preset(kind, name, data, source)


def read_presets(name: str, raw: bytes) -> list[Preset]:
    """Every preset in an exported file (.bbscfg / .bbsflmt / presets .zip / one .json).

    Raises ValueError when the file holds none (or is a full project, which belongs elsewhere).
    """
    suffix = Path(name).suffix.lower()
    out: list[Preset] = []
    if suffix == ".json":
        data = json.loads(raw.decode("utf-8-sig"))
        if is_project_settings(data):
            raise ValueError(f"{name} is a whole project's settings, not a preset")
        p = _preset(data, name, name)
        if p:
            out.append(p)
    else:
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{name} is not a Bambu preset export (not a zip)") from exc
        with z:
            members = {i.filename: i for i in z.infolist() if not i.is_dir()}
            listed: list[tuple[str, str | None]] = []
            if BUNDLE_STRUCTURE in members:
                st = json.loads(z.read(BUNDLE_STRUCTURE).decode("utf-8-sig"))
                for key, folder_kind in (("printer_config", "printer"), ("filament_config", "filament"),
                                         ("process_config", "process")):
                    listed += [(m, folder_kind) for m in st.get(key, []) or []]
                for vendor in st.get("filament_vendor", []) or []:
                    listed += [(m, "filament") for m in vendor.get("filament_path", []) or []]
            else:
                listed = [(m, None) for m in members if m.lower().endswith(".json")]
            for member, kind in listed:
                if member not in members:
                    continue
                try:
                    data = json.loads(z.read(member).decode("utf-8-sig"))
                except ValueError:
                    continue
                p = _preset(data, member, name, kind)
                if p:
                    out.append(p)
    if not out:
        raise ValueError(f"No Bambu Studio presets found in {name}")
    return out


class SystemProfiles:
    """Bambu Studio's shipped system presets (resources/profiles/BBL), indexed by name, for `inherits`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._index: dict[str, dict[str, Path]] | None = None
        self._by_id: dict[str, list[str]] | None = None

    def _build(self) -> dict[str, dict[str, Path]]:
        idx: dict[str, dict[str, Path]] = {k: {} for k in KINDS}
        for folder, kind in (("filament", "filament"), ("process", "process"), ("machine", "printer")):
            d = self.root / folder
            if d.is_dir():
                for f in d.rglob("*.json"):
                    idx[kind].setdefault(f.stem, f)
        return idx

    def path(self, kind: str, name: str) -> Path | None:
        if self._index is None:
            self._index = self._build()
        p = self._index[kind].get(name)
        if p is not None:
            return p
        # A file whose stem differs from its preset name: fall back to reading names (rare).
        return None

    def load(self, kind: str, name: str) -> dict[str, Any] | None:
        p = self.path(kind, name)
        return _read_json(str(p)) if p else None

    def filaments_by_id(self) -> dict[str, list[str]]:
        """Selectable system filament presets by Bambu filament id ("GFL99" = Generic PLA, the id an AMS
        reports per spool as `tray_info_idx`). The id is set on a family's `@base` preset and inherited."""
        if self._by_id is None:
            if self._index is None:
                self._index = self._build()
            by_id: dict[str, list[str]] = {}
            for name in self._index["filament"]:
                d = self.load("filament", name) or {}
                if str(d.get("instantiation", "")).lower() != "true":
                    continue
                fid, parent, seen = d.get("filament_id"), str(d.get("inherits", "") or ""), set()
                while not fid and parent and parent not in seen:
                    seen.add(parent)
                    up = self.load("filament", parent) or {}
                    fid, parent = up.get("filament_id"), str(up.get("inherits", "") or "")
                if fid:
                    by_id.setdefault(str(fid), []).append(name)
            self._by_id = by_id
        return self._by_id


@lru_cache(maxsize=4096)
def _read_json_cached(path: str, mtime: float) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _read_json(path: str) -> dict[str, Any]:
    return _read_json_cached(path, Path(path).stat().st_mtime)


def default_system_profiles() -> SystemProfiles | None:
    """The installed Bambu Studio's BBL profiles (next to the CLI the slicer uses), if any."""
    from .slicer import find_exe
    exe = find_exe()
    if exe is None:
        return None
    root = exe.parent / "resources" / "profiles" / "BBL"
    return SystemProfiles(root) if root.is_dir() else None


def resolve(preset: Preset, system: SystemProfiles | None) -> tuple[dict[str, Any], bool]:
    """The preset's full values: its `inherits` chain (system presets) overlaid by its own keys.

    Returns (values, exact); exact is False when part of the chain could not be found, so the
    values are only the keys the chain that was found sets.
    """
    chain: list[dict[str, Any]] = [preset.data]
    parent = preset.inherits
    exact = True
    seen: set[str] = set()
    while parent:
        if parent in seen:
            break
        seen.add(parent)
        d = system.load(preset.kind, parent) if system else None
        if d is None:
            exact = False
            break
        chain.append(d)
        parent = str(d.get("inherits", "") or "")
    values: dict[str, Any] = {}
    for d in reversed(chain):
        values.update(d)
    return values, exact


def material_of(values: dict[str, Any], *names: str) -> str:
    """Material family: the preset's filament_type, else the first family named in the given names."""
    t = values.get("filament_type")
    if isinstance(t, list):
        t = t[0] if t else ""
    if t:
        return str(t)
    for n in names:
        up = n.upper()
        for m in MATERIALS:
            if re.search(rf"(?<![A-Z]){re.escape(m)}(?![A-Z])", up):
                return m
    return ""


def printer_code(name: str) -> str:
    """'Generic PLA @BBL X2D 0.4 nozzle' -> 'X2D'; 'Bambu Lab X2D 0.4 nozzle' -> 'X2D'; else ''."""
    m = re.search(r"@BBL\s+(\S+)", name) or re.search(r"Bambu Lab\s+(.+?)(?:\s+\d+(?:\.\d+)?\s+nozzle)?$", name)
    return m.group(1).strip() if m else ""


def nozzle_of(name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s+nozzle", name)
    return m.group(1) if m else ""


def _ref_printer(ref: dict[str, Any]) -> tuple[str, str]:
    pid = str(ref.get("printer_settings_id", ""))
    model = str(ref.get("printer_model", "")) or pid
    code = printer_code(model) or printer_code(pid)
    noz = ref.get("nozzle_diameter", [""])
    return code, str(noz[0] if isinstance(noz, list) and noz else noz)


def compatible(values: dict[str, Any], preset: Preset, ref: dict[str, Any]) -> tuple[bool, str]:
    """Is a process/printer/filament preset made for the project's printer (model and nozzle)?"""
    code, noz = _ref_printer(ref)
    printers = values.get("compatible_printers") or []
    pid = str(ref.get("printer_settings_id", ""))
    if isinstance(printers, list) and printers and pid and pid not in printers:
        return False, f"made for {', '.join(map(str, printers[:2]))}"
    for n in (preset.name, preset.inherits):
        c = printer_code(n)
        if c and code and c != code:
            return False, f"made for the {c}, your project is the {code}"
        nz = nozzle_of(n)
        if nz and noz and float(nz) != float(noz):
            return False, f"made for a {nz} mm nozzle, your project uses {noz} mm"
    return True, ""


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, list) else [v]


def _fit(new: Any, old: list[Any], new_variants: list[Any] | None = None,
         old_variants: list[Any] | None = None) -> list[Any] | None:
    """Value `new` reshaped to the length of `old`, or None when it cannot be.

    "nil" entries mean "not set" in Bambu presets and keep the old value. Per-variant arrays
    (one entry per extruder variant, `filament_extruder_variant`) are matched by variant name.
    """
    vals = _as_list(new)
    k = len(old)
    if len(vals) == k:
        out = vals
    elif len(vals) == 1:
        out = vals * k
    elif new_variants and old_variants and len(vals) == len(new_variants) and len(old_variants) == k:
        by = dict(zip(map(str, new_variants), vals))
        out = [by.get(str(v), "nil") for v in old_variants]
    else:
        return None
    return [o if str(n) == "nil" else n for n, o in zip(out, old)]


def _per_filament(ref: dict[str, Any], key: str, n: int) -> bool:
    val = ref[key]
    return (isinstance(val, list) and len(val) >= n and len(val) % n == 0
            and key not in ("flush_volumes_matrix", "different_settings_to_system", "inherits_group"))


def add_filament(ref: dict[str, Any], values: dict[str, Any], name: str, base: int) -> dict[str, Any]:
    """Settings with one more filament: a copy of slot `base` overlaid by `values`, named `name`."""
    n = len(ref["filament_settings_id"])
    out = dict(ref)
    variants_new = values.get("filament_extruder_variant")
    vk = len(ref.get("filament_extruder_variant", [])) // n if ref.get("filament_extruder_variant") else 0
    variants_old = ref["filament_extruder_variant"][base * vk:(base + 1) * vk] if vk else None
    for key, val in ref.items():
        if not _per_filament(ref, key, n):
            continue
        k = len(val) // n
        block = list(val[base * k:(base + 1) * k])
        if key in values and key not in META_KEYS and key not in ("filament_extruder_variant", "filament_self_index"):
            fitted = _fit(values[key], block, _as_list(variants_new) if variants_new else None, variants_old)
            if fitted is not None:
                block = fitted
        if key == "filament_self_index":
            block = [str(n + 1)] * k
        out[key] = list(val) + block
    out["filament_settings_id"] = list(ref["filament_settings_id"]) + [name]
    colour = values.get("default_filament_colour")
    colour = colour[0] if isinstance(colour, list) and colour else colour
    out["filament_colour"] = list(ref.get("filament_colour", [])) + [
        colour if colour else ref.get("filament_colour", ["#808080"] * n)[base]]
    for key in ("different_settings_to_system", "inherits_group"):
        v = ref.get(key)
        if isinstance(v, list) and len(v) == n + 2:
            extra = values.get("inherits", "") if key == "inherits_group" else ""
            out[key] = v[:-1] + [extra] + v[-1:]
    mat = ref.get("flush_volumes_matrix")
    if isinstance(mat, list) and mat and len(mat) % (n * n) == 0:
        e = len(mat) // (n * n)
        new: list[Any] = []
        for x in range(e):
            m = [list(mat[x * n * n + r * n:x * n * n + (r + 1) * n]) for r in range(n)]
            # The new filament purges like its base slot: row/column `base` copied, then the pair
            # (base, new) — same material, maybe another colour — gets the typical volume.
            rows = [row + [row[base]] for row in m] + [m[base] + ["0"]]
            rows[base][n] = rows[n][base] = _typical(m)
            new += [v for row in rows for v in row]
        out["flush_volumes_matrix"] = new
    return out


def _typical(m: list[list[Any]]) -> str:
    """Median off-diagonal purge volume: used between a slot and its own preset copy."""
    off = sorted(float(m[i][j]) for i in range(len(m)) for j in range(len(m)) if i != j)
    return f"{off[len(off) // 2]:g}" if off else "0"


def apply_keys(ref: dict[str, Any], values: dict[str, Any], id_key: str, name: str) -> tuple[dict[str, Any], int]:
    """Settings with `values` replacing the keys they share with `ref` (shape-checked); returns (settings, count)."""
    out = dict(ref)
    n = len(ref.get("filament_settings_id", [])) or 1
    count = 0
    for key, v in values.items():
        if key in META_KEYS or key not in ref or key == id_key:
            continue
        old = ref[key]
        if isinstance(old, list):
            if _per_filament(ref, key, n) and n > 1:
                continue            # filament keys come from filament presets
            fitted = _fit(v, old)
            if fitted is None:
                continue
            out[key] = fitted
        else:
            vv = _as_list(v)
            if len(vv) != 1:
                continue
            out[key] = old if str(vv[0]) == "nil" else vv[0]
        count += 1
    out[id_key] = name
    return out, count


@dataclass
class PresetLibrary:
    """Presets the user imported this session (latest file wins for a repeated name)."""
    presets: dict[tuple[str, str], Preset] = field(default_factory=dict)
    system: SystemProfiles | None = None
    version: int = 0           # bumped on every change (settings caches key on it)

    def add(self, items: list[Preset]) -> None:
        for p in items:
            self.presets[(p.kind, p.name)] = p
        self.version += 1

    def of(self, kind: str) -> list[Preset]:
        return [p for (k, _), p in self.presets.items() if k == kind]

    def merged(self, ref: dict[str, Any], process: str | None = None) -> tuple[dict[str, Any], list[PresetNote]]:
        """`ref` with the printer preset (if it fits), every filament preset appended and process `process`."""
        notes: list[PresetNote] = []
        out = dict(ref)
        printers = []
        for p in self.of("printer"):
            values, exact = resolve(p, self.system)
            ok, why = compatible(values, p, ref)
            if ok:
                printers.append((p, values, exact))
            else:
                notes.append(PresetNote("printer", p.name, "skipped", why + " — load a project saved for it instead"))
        if len(printers) == 1:
            p, values, exact = printers[0]
            out, count = apply_keys(out, values, "printer_settings_id", p.name)
            notes.append(PresetNote("printer", p.name, "used" if exact else "approximate",
                                    f"{count} printer settings" + ("" if exact else "; base preset not installed")))
        else:
            notes += [PresetNote("printer", p.name, "skipped", "several printer presets: kept your project's")
                      for p, _, _ in printers]
        for p in self.of("filament"):
            values, exact = resolve(p, self.system)
            ok, why = compatible(values, p, ref)
            mat = material_of(values, p.inherits, p.name)
            base = _base_slot(ref, p.inherits, mat)
            out = add_filament(out, {**values, "filament_type": [mat] if mat else values.get("filament_type")},
                               p.name, base)
            note = f"{mat or 'unknown material'}, from slot {base + 1}"
            if not ok:
                note += f"; {why}"
            plate = plate_problem(out, len(out["filament_settings_id"]) - 1)
            if plate:
                note += f"; {plate}"
            if not exact:
                note += "; base preset not installed, unchanged settings copied from your project"
            notes.append(PresetNote("filament", p.name, "used" if exact and ok else "approximate", note))
        for p in self.of("process"):
            values, exact = resolve(p, self.system)
            ok, why = compatible(values, p, ref)
            if not ok:
                notes.append(PresetNote("process", p.name, "skipped", why))
                continue
            if p.name == process:
                out, count = apply_keys(out, values, "print_settings_id", p.name)
                notes.append(PresetNote("process", p.name, "used" if exact else "approximate",
                                        f"{count} print settings" + ("" if exact else "; base preset not installed")))
            else:
                notes.append(PresetNote("process", p.name, "available"))
        return out, notes

    def processes(self, ref: dict[str, Any]) -> list[str]:
        """Process presets that fit the project's printer, for the UI's picker."""
        out = []
        for p in self.of("process"):
            values, _ = resolve(p, self.system)
            if compatible(values, p, ref)[0]:
                out.append(p.name)
        return out


# Bambu's bed types -> the filament key holding that plate's temperature (0 = cannot print on it).
PLATE_KEYS = {"Supertack Plate": "supertack_plate_temp", "Cool Plate": "cool_plate_temp",
              "Engineering Plate": "eng_plate_temp", "High Temp Plate": "hot_plate_temp",
              "Textured PEI Plate": "textured_plate_temp", "Textured Cool Plate": "textured_cool_plate_temp"}


def plate_problem(settings: dict[str, Any], slot: int) -> str:
    """Why filament `slot` cannot print on the project's plate ("" if it can).

    Bambu refuses to slice ("Filaments are not compatible with the plate type") when the
    filament's temperature for the project's plate (`curr_bed_type`) is 0.
    """
    plate = str(settings.get("curr_bed_type", ""))
    key = PLATE_KEYS.get(plate)
    vals = settings.get(key) if key else None
    if not isinstance(vals, list) or slot >= len(vals):
        return ""
    try:
        ok = float(vals[slot]) > 0
    except ValueError:
        return ""
    return "" if ok else f"not for your project's {plate} (its {plate} temperature is 0): pick another plate in Bambu Studio"


def preset_for_filament_id(system: SystemProfiles | None, filament_id: str,
                           ref: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """(name, full values) of the installed system filament preset with this Bambu filament id made for
    the project's printer and nozzle (e.g. GFL96 on an X2D 0.4 -> "Generic PLA Silk @BBL X2D 0.4 nozzle"),
    or None. The most specific name wins: one naming the nozzle over the printer-wide one."""
    if system is None or not filament_id:
        return None
    best: tuple[int, str, dict[str, Any]] | None = None
    for name in system.filaments_by_id().get(filament_id, []):
        data = system.load("filament", name) or {}
        values, _ = resolve(Preset("filament", name, data, name), system)
        ok, _why = compatible(values, Preset("filament", name, data, name), ref)
        if not ok or not printer_code(name):
            continue
        rank = 2 if nozzle_of(name) else 1
        if best is None or rank > best[0]:
            best = (rank, name, values)
    return (best[1], best[2]) if best else None


def generic_preset(system: SystemProfiles | None, material: str,
                   ref: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The system "Generic <material>" preset for the project's printer (a spool whose own preset is not
    installed for this printer, e.g. eSUN PLA+ on the X2D, prints as Generic PLA)."""
    if system is None or not material:
        return None
    if system._index is None:
        system._index = system._build()
    best: tuple[int, str, dict[str, Any]] | None = None
    for name in system._index["filament"]:
        if not name.startswith(f"Generic {material} @BBL"):
            continue
        data = system.load("filament", name) or {}
        values, _ = resolve(Preset("filament", name, data, name), system)
        if not compatible(values, Preset("filament", name, data, name), ref)[0]:
            continue
        rank = 2 if nozzle_of(name) else 1
        if best is None or rank > best[0]:
            best = (rank, name, values)
    return (best[1], best[2]) if best else None


def ams_filaments(ref: dict[str, Any], system: SystemProfiles | None, spools: list[tuple[str, str]],
                  fallback: int) -> tuple[dict[str, Any], list[int], list[str]]:
    """Settings that hold a preset for each of `spools` ((Bambu filament id, material) of the AMS spools the
    figure's filaments map to), and for each the slot index to copy: (settings, src, preset names). The
    spool's own system preset for this printer, else the Generic preset of its material; one already a
    slot of the project is reused, others are added. With neither, slot `fallback` (the export's material)."""
    out = ref
    src: list[int] = []
    names: list[str] = []
    for fid, material in spools:
        found = preset_for_filament_id(system, fid, out) or generic_preset(system, material, out)
        if found is None:
            src.append(fallback)
            names.append(str(out["filament_settings_id"][fallback]))
            continue
        name, values = found
        ids = list(out["filament_settings_id"])
        if name not in ids:
            mat = material_of(values, name)
            out = add_filament(out, values, name, _base_slot(out, str(values.get("inherits", "")), mat))
            ids = list(out["filament_settings_id"])
        src.append(ids.index(name))
        names.append(name)
    return out, src, names


def _base_slot(ref: dict[str, Any], inherits: str, material: str) -> int:
    """The project slot a filament preset starts from: the one with the same parent, else the same material."""
    ids = list(ref.get("filament_settings_id", []))
    if inherits in ids:
        return ids.index(inherits)
    types = list(ref.get("filament_type", []))
    if material in types:
        return types.index(material)
    fam = material.split("-")[0]
    for i, t in enumerate(types):
        if str(t).split("-")[0] == fam:
            return i
    return 0


def load_settings_file(path: str | Path, library: PresetLibrary) -> dict[str, Any] | None:
    """Read a file the user gave as "their settings".

    A Bambu project (.3mf) or a saved project config (.json with "from": "project") returns its
    settings (the new reference). Preset exports (.bbscfg / .bbsflmt / presets .zip / a preset
    .json) are added to `library` and None is returned. Raises ValueError for anything else.
    """
    from .slicer import reference_settings
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".3mf":
        return reference_settings(path)
    if suffix not in PRESET_SUFFIXES:
        raise ValueError(f"{path.name}: expected a Bambu project (.3mf) or a preset export "
                         "(.bbscfg, .bbsflmt, .zip, .json)")
    raw = path.read_bytes()
    if suffix == ".json":
        try:
            data = json.loads(raw.decode("utf-8-sig"))
        except ValueError as exc:
            raise ValueError(f"{path.name} is not JSON: {exc}") from exc
        if isinstance(data, dict) and is_project_settings(data):
            return data
    library.add(read_presets(path.name, raw))
    return None
