"""Bambu Studio command-line slicing: print time, filament, supports.

Verified on Bambu Studio 02.08.02.61 (Windows):
  bambu-studio.exe --slice 0 [--arrange 1] --outputdir <dir> <in.3mf>
slices every plate of the 3MF with the settings embedded in it, writes
plate_N.gcode and result.json into <dir> — and ALSO a result.json into the
current working directory, so it always runs with cwd = <dir>.
result.json: return_code (0 = ok), error_string, and per plate
total_predication (s), filaments[].total_used_g, warning_message.

Settings: the figure is sliced with the user's own project settings (a
reference 3MF, default pokeball60goodPrinted.3mf: Bambu X2D, PETG, 0.16 mm)
rather than whatever printer the figure's author used. Its per-filament
arrays (17 AMS slots there) are reduced to the figure's filament count, each
figure filament taking the chosen slot's material.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EXE = Path(os.environ.get("POKEFIT_BAMBU", r"C:\Program Files\Bambu Studio\bambu-studio.exe"))
TIMEOUT_S = 300
RETRIES = 1          # a run that dies without writing result.json is retried once


@dataclass
class SliceResult:
    ok: bool
    seconds: float = 0.0           # Bambu's total print time prediction
    grams: float = 0.0             # total filament
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    copies: int = 1
    wall_time: float = 0.0         # how long slicing took
    support_seconds: float = 0.0   # time Bambu spends printing support features
    grams_per_filament: list[float] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)   # seconds per feature type (all plates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_exe(exe: str | Path | None = None) -> Path | None:
    p = Path(exe) if exe else DEFAULT_EXE
    if p.exists():
        return p
    found = shutil.which("bambu-studio")
    return Path(found) if found else None


def reference_settings(path: str | Path) -> dict:
    """Project settings from a Bambu 3MF (Metadata/project_settings.config) or a saved JSON copy."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("Metadata/project_settings.config"))


def filament_slots(ref: dict) -> list[dict[str, str]]:
    """The reference project's filament slots, for the UI's material picker."""
    ids = ref.get("filament_settings_id", [])
    types = ref.get("filament_type", [""] * len(ids))
    cols = ref.get("filament_colour", [""] * len(ids))
    return [{"slot": str(i), "name": n, "type": t, "colour": c} for i, (n, t, c) in enumerate(zip(ids, types, cols))]


def default_slot(ref: dict, prefer: str = "PETG") -> int:
    for i, t in enumerate(ref.get("filament_type", [])):
        if t == prefer:
            return i
    return 0


def _hex_rgb(c: str) -> np.ndarray:
    c = c.lstrip("#")[:6].ljust(6, "0")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)


def adapt_settings(ref: dict, colours: list[str], slot: int, supports: bool) -> dict:
    """Reference settings reduced to len(colours) filaments, all using material `slot`.

    The flush matrix (extruders × n_ref × n_ref) keeps the reference's purge
    volume between the slots whose colours are closest to each figure colour.
    """
    out = _filaments_from(ref, [slot] * max(1, len(colours)), colours)
    out["enable_support"] = "1" if supports else "0"
    if supports and not str(out.get("support_type", "")).endswith("(auto)"):
        out["support_type"] = "tree(auto)"
    return out


def resize_filaments(settings: dict, colours: list[str]) -> dict:
    """A project's own settings with len(colours) filaments: filament i keeps its material, filaments
    added beyond the project's copy its last one (colorize added colours to a file)."""
    n_ref = len(settings.get("filament_settings_id", [])) or 1
    return _filaments_from(settings, [min(i, n_ref - 1) for i in range(max(1, len(colours)))], colours)


def _filaments_from(ref: dict, src: list[int], colours: list[str]) -> dict:
    """Settings whose filament i is a copy of `ref`'s filament src[i], coloured colours[i].

    Per-filament arrays are filament-major blocks: length n_ref·k → n·k.
    `different_settings_to_system` / `inherits_group` are [print, filaments..., printer].
    """
    n_ref = len(ref["filament_settings_id"])
    n = len(src)
    out: dict = {}
    for key, val in ref.items():
        if not isinstance(val, list):
            out[key] = val
        elif key in ("different_settings_to_system", "inherits_group") and len(val) == n_ref + 2:
            out[key] = [val[0]] + [val[1 + i] for i in src] + [val[-1]]
        elif key == "flush_volumes_matrix":
            out[key] = val            # rebuilt below
        elif len(val) >= n_ref and len(val) % n_ref == 0:
            k = len(val) // n_ref
            out[key] = [x for i in src for x in val[i * k:(i + 1) * k]]
        else:
            out[key] = val

    ref_cols = np.array([_hex_rgb(c) for c in ref.get("filament_colour", ["#808080"] * n_ref)])
    near = [int(np.argmin(np.linalg.norm(ref_cols - _hex_rgb(c), axis=1))) for c in colours] or src[:1]
    mat = ref.get("flush_volumes_matrix", [])
    n_ext = max(1, len(mat) // (n_ref * n_ref)) if mat else 0
    if n_ext:
        m = np.array(mat, float).reshape(n_ext, n_ref, n_ref)
        off = m[:, ~np.eye(n_ref, dtype=bool)]
        typical = float(np.median(off)) if off.size else 0.0
        new = np.zeros((n_ext, n, n))
        for e in range(n_ext):
            for a in range(n):
                for b in range(n):
                    if a != b:
                        v = m[e, near[a], near[b]]
                        new[e, a, b] = v if near[a] != near[b] else typical
        out["flush_volumes_matrix"] = [f"{v:g}" for v in new.ravel()]
    out["filament_colour"] = list(colours) if colours else out.get("filament_colour", [])[:n]
    return out


def read_result(out_dir: Path, copies: int) -> SliceResult:
    rj = out_dir / "result.json"
    if not rj.exists():
        return SliceResult(False, error="Bambu Studio wrote no result.json", copies=copies)
    data = json.loads(rj.read_text(encoding="utf-8"))
    plates = data.get("sliced_plates", [])
    warnings = [p["warning_message"] for p in plates if p.get("warning_message")]
    if data.get("return_code", -1) != 0 or not plates:
        return SliceResult(False, warnings=warnings, copies=copies,
                           error=f"{data.get('error_string', 'slice failed')} (code {data.get('return_code')})")
    secs = sum(float(p.get("total_predication", 0)) for p in plates)
    per = [float(f.get("total_used_g", 0)) for p in plates for f in p.get("filaments", [])]
    features: dict[str, float] = {}
    for p in plates:
        for k, t in p.get("feature_type_times", {}).items():
            features[k] = features.get(k, 0.0) + float(t)
    support = sum(t for k, t in features.items() if "support" in k.lower())
    return SliceResult(True, secs, sum(per), warnings, copies=copies, support_seconds=support,
                       grams_per_filament=per, features=features)


def slice_3mf(path: str | Path, exe: str | Path | None = None, copies: int = 1, arrange: bool = False,
              work_dir: str | Path | None = None, timeout: float = TIMEOUT_S) -> SliceResult:
    """Slice every plate of a 3MF; never raises for slicer problems (they come back as `error`)."""
    exe_path = find_exe(exe)
    if exe_path is None:
        return SliceResult(False, error=f"Bambu Studio not found at {exe or DEFAULT_EXE}", copies=copies)
    # Absolute: the CLI runs with cwd = out_dir, so a relative --outputdir would land one level deeper.
    out_dir = (Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="pokefit_slice_"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe_path), "--slice", "0"]
    if arrange:
        cmd += ["--arrange", "1"]
    cmd += ["--outputdir", str(out_dir), str(Path(path).resolve())]
    t0 = time.time()
    rc: int | None = None
    for attempt in range(RETRIES + 1):
        (out_dir / "result.json").unlink(missing_ok=True)
        try:
            rc = _run_until_result(cmd, out_dir, timeout)
        except OSError as exc:
            return SliceResult(False, error=f"cannot run Bambu Studio: {exc}", copies=copies)
        if rc is not None or _result_complete(out_dir):
            break
        if attempt == RETRIES:
            return SliceResult(False, error=f"slicing timed out after {timeout:.0f} s ({RETRIES + 1} attempts)",
                               copies=copies)
    res = read_result(out_dir, copies)
    res.wall_time = time.time() - t0
    if not res.ok and rc and not res.error:
        res.error = f"exit code {rc}"
    return res


def _result_complete(out_dir: Path) -> bool:
    try:
        return "return_code" in json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def _run_until_result(cmd: list[str], out_dir: Path, timeout: float, grace: float = 5.0) -> int | None:
    """Run the CLI; return its exit code, or None if it had to be killed.

    The CLI sometimes finishes slicing, writes result.json and the G-code, and
    then hangs during shutdown with a single idle thread (seen 2 times in ~12
    runs on 02.08.02.61). Once result.json is complete the process gets
    `grace` seconds to exit on its own before it is killed; the result stands.
    """
    proc = subprocess.Popen(cmd, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = time.time() + timeout
    done_at: float | None = None
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        now = time.time()
        if done_at is None and _result_complete(out_dir):
            done_at = now
        if (done_at is not None and now - done_at > grace) or now > deadline:
            proc.kill()
            proc.wait(10)
            return None
        time.sleep(0.25)


def grid_positions(footprint: tuple[float, float], copies: int, plate: tuple[float, float],
                   gap: float = 8.0, margin: float = 15.0) -> list[tuple[float, float]] | None:
    """Row-major grid of `copies` footprints centred on the plate; None if they do not fit."""
    w, h = footprint[0] + gap, footprint[1] + gap
    cols = max(1, int((plate[0] - 2 * margin) // w))
    rows = -(-copies // cols)
    if rows * h > plate[1] - 2 * margin:
        return None
    x0 = plate[0] / 2 - (min(cols, copies) - 1) * w / 2
    y0 = plate[1] / 2 - (rows - 1) * h / 2
    return [(x0 + (i % cols) * w, y0 + (i // cols) * h) for i in range(copies)]
