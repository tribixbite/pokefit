# pokefit × beambam × craftmatic — one pipeline from model to printed figure

Investigated 2026-10-19 (pokefit `main`, beambam `1c040c9`, craftmatic `9eda695f`). This is the plan; the
"Built" column is what exists in pokefit today.

## The pipeline

| Stage | What happens | Owner | Built |
|---|---|---|---|
| 1. Create | Photos → 3D (Pixal3D in a local ComfyUI); LEGO / schematic / voxel build → GLB; any 3MF/STL/OBJ/GLB | pokefit `imageto3d.py`; craftmatic `web/src/viewer/exporter.ts` | ✔ photos, ✔ file formats, ✔ craftmatic "Fit in a Pokéball" (both sides) |
| 2. Rework | Fit inside the ball, bend limbs, flat base, magnet pockets, printability checks | pokefit `pipeline`, `conform`, `magnets`, `checks` | ✔ |
| 3. Colour | Texture/vertex colour → filaments, palette size, tap-to-paint, Bambu split paint | pokefit `gltf`, `colorize`, `paint` | ✔ |
| 4. Plan | Materials (presets), nozzle split and colour-change cost on the X2D, slice | pokefit `presets`, `toolchange`, `slicer` | ✔ estimates + Bambu CLI slices |
| 5. Map | Figure colours → filaments loaded in the 4 AMS units | pokefit `printer.py` (reads via beambam) + `remap.match_slots` | ✔ read + match + recolour; verified on the user's X2D (16 spools, track switch) |
| 6. Print | Upload, start with the AMS mapping, watch live, change filaments | beambam (`ftps`, `print_job`, `cam`, `ams`) | ✗ next (see below) |

## Boundaries (why not one codebase)

- **Both are AGPL-3.0-or-later** (pokefit since 2026-10-19, the user's choice), so pokefit *may* import
  beambam. It still runs beambam's CLI as a separate program (`pokefit/printer.py`: `beambam status` →
  JSON): beambam is a large local-only dependency with its own venv, and the browser engine (Pyodide) cannot
  use it. Importing `beambam.printer.Printer` directly is an option if the subprocess becomes a bottleneck.
- **craftmatic is TypeScript (MIT), pokefit's engine is Python** (also in the browser via Pyodide). Sharing
  code means a port; sharing **files** is free. So the seam is a handoff of the model:
  - craftmatic exports a GLB with one material per colour (`exporter.ts`) → pokefit reads it as filaments
    (`gltf.py`), fits it, paints it and writes the Bambu 3MF. This makes craftmatic's unbuilt
    `docs/specs/bambu-3mf-export.md` unnecessary: don't write a second Bambu writer in TS.
  - pokefit side: `web/src/handoff.ts`. craftmatic opens `https://pokefit.party/` and, on
    `pokefit:ready`, posts `{type: "pokefit:load", name: "build.glb", data: ArrayBuffer}`; origins
    craftmatic.click / tribixbite.github.io / localhost are trusted. `?url=<model>` also works (CORS).
  - craftmatic side (commits `343cd48a`, `d4182af6`, requirement `VIEW-19`; local, not pushed yet):
    `web/src/viewer/pokefit-handoff.ts` + "Fit in a Pokéball" in both Download menus and the LEGO export
    select. Verified 2026-10-19 against the live pokefit.party from a localhost page with the same protocol:
    the tab took a 4.8 MB GLB, fitted it (30.4 mm, 2 pockets) and painted it.
  - Worth sharing later as a tiny TS package: craftmatic's build stamp (`scripts/pipeline-stamp.ts`) vs
    pokefit's `src/version.ts`, and its Lab palette code vs pokefit's ΔE matching — only if both keep
    growing; today the duplication is small.

## X2D specifics (two nozzles, 4 AMS, track switch)

Measured with the Bambu CLI on the user's X2D project (PETG, 0.16 mm, prime tower), details in NOTES.md:
- Every colour change on one nozzle costs ≈ 105 s and ≈ 0.4 g (flush included); a nozzle switch ≈ 18 s.
- Two nozzles turn two colours into **zero** filament changes. Manex Squirtle with today's palettes
  (Ward merging, flat base), Bambu slices 2026-10-19: 1 / 2 / 3 / 4 colours = 0.67 / 1.76 / 5.32 / 6.35 h,
  0 / 0 / 120 / 151 changes, 2.8 / 19.8 / 70.9 / 88.9 g; pokefit's estimate 0 / 120 / 161 changes,
  +46 / +255 / +327 min (sliced +65 / +279 / +341). Which colour is the expensive one depends on where it
  sits in height: here the 3rd (brown shell, spanning most layers). "Auto" now picks the count before the
  biggest jump (`toolchange.best_value_count`, the ★ in the Colours card): 2 for this figure.
- Nozzle assignment: the slicer CLI takes it from the **plate's** `filament_map_mode` / `filament_maps` in
  `model_settings.config` (not `project_settings.config`). Bambu's "Auto For Flush" matched or beat
  pokefit's own planner (equal on 5 colours, 9 min better on 7), so exports keep Bambu's auto mode; all 16
  assignments of a 5-colour figure ranged 6.94–10.79 h, one nozzle being the worst.
- Track switch: the user's X2D has it (`device.fila_switch` in the status report; every AMS unit's `info`
  reads 0xE in bits 8-11 = routed by the switch, not wired to one nozzle), so any spool can feed either
  nozzle — what pokefit's planner assumes. Bambu's `has_filament_switcher` setting made no difference to the
  CLI slice (7 colours: 10.19 h / 145.7 g / 425 changes / same nozzle split with 0 and 1), so exports leave
  the project's value alone. Without a switch each unit feeds one nozzle; pokefit shows (R)/(L) per spool
  but does not yet constrain the planner (`printer.py` TODO).

## Next steps (need the user / the printer)

1. Done 2026-10-19: credentials in `~/.x2d/credentials` (outside the repo); "Use my printer's AMS" reads the
   X2D's 4 AMS units (16 spools) through `POKEFIT_BEAMBAM=C:/git/beambam/.venv/Scripts/beambam.exe`.
2. **Send to printer**: the local Slice step now writes the print file (`…_auto.gcode.3mf` /
   `…_off.gcode.3mf`, with the plate thumbnails the touchscreen needs) and offers it as a download; next is
   `beambam print <file> --ams-mapping …` from the AMS match. Needs the per-install signing key (`~/.x2d/printer_sign_key.pem`,
   extracted from Bambu Handy via adb, per beambam's docs). **Blocker in beambam**: the dual-nozzle
   `nozzle_mapping` in `project_file` is not reproduced yet (err 84033544) — capture one real dual-nozzle
   print start (Bambu Studio → printer) and add it to beambam first. Until then: upload with
   `beambam push` and start on the touchscreen (needs the plate thumbnails beambam's
   `inject_thumbnails.py` adds).
3. **Live view**: `beambam camera` proxies the X2D's RTSPS feed to MJPEG on 127.0.0.1:8766; the local UI
   can embed it in a "Printer" card next to the AMS slots (print progress from `beambam status`). Tried
   2026-10-19: on Windows its ffmpeg pump dies at `select.select()` on a pipe (`beambam/cli/daemon.py:433`,
   WinError 10038) — a beambam fix (reader thread) comes first.
4. **Filament control**: `beambam ams-load / ams-unload` and `print.ams_filament_setting` (set a tray's
   type/colour) — a per-slot menu in that card. Every printer-changing action is user-confirmed.
5. **Auto colour optimisation** across the loop: palette size from the cost table (done: "auto"), snap colours
   to loaded spools ("Use these filaments' colours"), and suggest which spools to load (the colours the
   figure needs that the AMS lacks).
