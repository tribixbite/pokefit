"""Check an exported pokefit 3MF: its parts, paint and filament settings, then slice it with Bambu Studio.

    python scripts/verify_export.py pokefit_out/<name>/<name>_pokefit_60.3mf [--out DIR] [--no-slice]

Prints the parts (name, extruder, faces), how many triangles carry each paint state (main state of
split paint), whether split paint is present, the filament colours in the embedded settings, and
the Bambu CLI result: time, grams, colour changes and the purge share (Flush + Prime tower time).
The figure's own filament is `main g`; the rest of `total g` is purge. Needs Bambu Studio installed
(`pokefit.slicer.DEFAULT_EXE` or `bambu-studio` on PATH) unless --no-slice.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from pokefit.model3mf import read_project
from pokefit.paint import decode_majority, parse_tree
from pokefit.slicer import slice_3mf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="exported .3mf")
    ap.add_argument("--out", type=Path, help="slice output folder (default: a new temp folder)")
    ap.add_argument("--no-slice", action="store_true", help="only inspect the file")
    a = ap.parse_args()

    proj = read_project(a.file, as_written=True)
    obj = proj.objects[0]
    states: Counter[int] = Counter()
    split = 0
    for p in obj.parts:
        for attrs in p.tri_extra.values():
            if 'paint_color="' in attrs:
                code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
                states[decode_majority(code)] += 1
                split += not isinstance(parse_tree(code), int)
    print("parts:", [(p.name, p.extruder, len(p.faces)) for p in obj.parts], "object extruder", obj.extruder)
    print(f"paint (main state -> triangles): {dict(sorted(states.items()))}; split triangles: {split}")
    with zipfile.ZipFile(a.file) as z:
        settings = json.loads(z.read("Metadata/project_settings.config"))
    print("filament_colour:", settings.get("filament_colour"), "| settings ids:", len(settings.get("filament_settings_id", [])))
    if a.no_slice:
        return 0

    out = a.out or Path(tempfile.mkdtemp(prefix="pokefit_verify_"))
    res = slice_3mf(a.file, work_dir=out)
    if not res.ok:
        print("slice FAILED:", res.error)
        return 1
    print(f"slice ok: {res.seconds / 3600:.2f} h, {res.grams:.1f} g, per filament {[round(g, 2) for g in res.grams_per_filament]}")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    for plate in result.get("sliced_plates", []):
        feats = plate.get("feature_type_times", {})
        purge = feats.get("Flush", 0.0) + feats.get("Prime tower", 0.0)
        main = sum(f.get("main_used_g", 0.0) for f in plate.get("filaments", []))
        total = sum(f.get("total_used_g", 0.0) for f in plate.get("filaments", []))
        print(f"plate {plate.get('id')}: colour changes {plate.get('filament_change_times')}, "
              f"purge {purge / 3600:.2f} h, main g {main:.2f}, total g {total:.1f}")
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
