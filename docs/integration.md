# pokefit × beambam × craftmatic — one pipeline from model to printed figure

Investigated 2026-10-19 (pokefit `main`, beambam `1c040c9`, craftmatic `9eda695f`). This is the plan; the
"Built" column is what exists in pokefit today.

## The pipeline

| Stage | What happens | Owner | Built |
|---|---|---|---|
| 1. Create | Photos → 3D (Pixal3D in a local ComfyUI); LEGO / schematic / voxel build → GLB; any 3MF/STL/OBJ/GLB | pokefit `imageto3d.py`; craftmatic `web/src/viewer/exporter.ts` | ✔ photos, ✔ file formats; craftmatic handoff: pokefit side ✔ |
| 2. Rework | Fit inside the ball, bend limbs, flat base, magnet pockets, printability checks | pokefit `pipeline`, `conform`, `magnets`, `checks` | ✔ |
| 3. Colour | Texture/vertex colour → filaments, palette size, tap-to-paint, Bambu split paint | pokefit `gltf`, `colorize`, `paint` | ✔ |
| 4. Plan | Materials (presets), nozzle split and colour-change cost on the X2D, slice | pokefit `presets`, `toolchange`, `slicer` | ✔ estimates + Bambu CLI slices |
| 5. Map | Figure colours → filaments loaded in the 4 AMS units | pokefit `printer.py` (reads via beambam) + `remap.match_slots` | ✔ read + match + recolour (needs beambam credentials) |
| 6. Print | Upload, start with the AMS mapping, watch live, change filaments | beambam (`ftps`, `print_job`, `cam`, `ams`) | ✗ next (see below) |

## Boundaries (why not one codebase)

- **beambam is AGPL-3.0**; pokefit (which serves pokefit.party) has no licence yet. Importing beambam would
  put pokefit under AGPL, including §13 for the website. So pokefit **runs beambam's CLI as a separate
  program** (`pokefit/printer.py`: `beambam status` → JSON) and never imports it. Same for any later step
  (upload, print, camera). If you want them merged, that is a licence decision for you (e.g. license
  pokefit AGPL too) — then `beambam.printer.Printer` could be imported directly.
- **craftmatic is TypeScript (MIT), pokefit's engine is Python** (also in the browser via Pyodide). Sharing
  code means a port; sharing **files** is free. So the seam is a handoff of the model:
  - craftmatic exports a GLB with one material per colour (`exporter.ts`) → pokefit reads it as filaments
    (`gltf.py`), fits it, paints it and writes the Bambu 3MF. This makes craftmatic's unbuilt
    `docs/specs/bambu-3mf-export.md` unnecessary: don't write a second Bambu writer in TS.
  - pokefit side (built): `web/src/handoff.ts`. craftmatic opens `https://pokefit.party/` and, on
    `pokefit:ready`, posts `{type: "pokefit:load", name: "build.glb", data: ArrayBuffer}`; origins
    craftmatic.click / tribixbite.github.io / localhost are trusted. `?url=<model>` also works (CORS).
  - craftmatic side (to add, ~15 lines in its export menu):
    ```ts
    const w = window.open("https://pokefit.party/");
    addEventListener("message", (e) => {
      if (e.source === w && e.data?.type === "pokefit:ready")
        w!.postMessage({ type: "pokefit:load", name: `${title}.glb`, data: glb }, "https://pokefit.party");
    });
    ```
  - Worth sharing later as a tiny TS package: craftmatic's build stamp (`scripts/pipeline-stamp.ts`) vs
    pokefit's `src/version.ts`, and its Lab palette code vs pokefit's ΔE matching — only if both keep
    growing; today the duplication is small.

## X2D specifics (two nozzles, 4 AMS, track switch)

Measured with the Bambu CLI on the user's X2D project (PETG, 0.16 mm, prime tower), details in NOTES.md:
- Every colour change on one nozzle costs ≈ 105 s and ≈ 0.4 g (flush included); a nozzle switch ≈ 18 s.
- Two nozzles turn two colours into **zero** filament changes: Manex Squirtle 1 / 2 / 3 / 4 / 7 colours =
  0.67 / 1.57 / 2.35 / 5.45 / 10.19 h and 2.8 / 17.3 / 28.9 / 73.8 / 145.7 g. The 3rd colour is cheap
  (it shares a nozzle with the rarest), the 4th is where it explodes. pokefit now shows this per palette
  size (Colours card) and in the Print card.
- Nozzle assignment: the slicer CLI takes it from the **plate's** `filament_map_mode` / `filament_maps` in
  `model_settings.config` (not `project_settings.config`). Bambu's "Auto For Flush" matched or beat
  pokefit's own planner (equal on 5 colours, 9 min better on 7), so exports keep Bambu's auto mode; all 16
  assignments of a 5-colour figure ranged 6.94–10.79 h, one nozzle being the worst.
- Track switch: Bambu's `has_filament_switcher` (printer setting) lifts the per-nozzle AMS-capacity limit
  in the slicer's filament grouping (BambuStudio `Print.cpp`). The user's saved project has it **off**:
  turn it on in Bambu Studio's printer settings if the switch is fitted; pokefit carries the project's
  value through.

## Next steps (need the user / the printer)

1. **beambam credentials on this PC** (`~/.x2d/credentials`: ip, access code, serial) → the "Use my
   printer's AMS" button works (it is hidden unless `beambam` is found; set `POKEFIT_BEAMBAM` to
   `C:/git/beambam/.venv/Scripts/beambam.exe`, which this session installed).
2. **Send to printer**: slice to a `.gcode.3mf` (Bambu CLI `--export-3mf`), then `beambam print <file>
   --ams-mapping …` from the AMS match. Needs the per-install signing key (`~/.x2d/printer_sign_key.pem`,
   extracted from Bambu Handy via adb, per beambam's docs). **Blocker in beambam**: the dual-nozzle
   `nozzle_mapping` in `project_file` is not reproduced yet (err 84033544) — capture one real dual-nozzle
   print start (Bambu Studio → printer) and add it to beambam first. Until then: upload with
   `beambam push` and start on the touchscreen (needs the plate thumbnails beambam's
   `inject_thumbnails.py` adds).
3. **Live view**: `beambam camera` proxies the X2D's RTSPS feed to MJPEG on 127.0.0.1:8766; the local UI
   can embed it in a "Printer" card next to the AMS slots (print progress from `beambam status`).
4. **Filament control**: `beambam ams-load / ams-unload` and `print.ams_filament_setting` (set a tray's
   type/colour) — a per-slot menu in that card. Every printer-changing action is user-confirmed.
5. **Auto colour optimisation** across the loop: pick the palette size from the cost table, snap colours
   to loaded spools ("Use these filaments' colours"), and suggest which spools to load (the colours the
   figure needs that the AMS lacks).
