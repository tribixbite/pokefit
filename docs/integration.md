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
| 6. Print | Upload, start with the AMS mapping, watch live, change filaments | beambam (`ftps`, `print_job`, `cam`, `ams`) | ✔ watch (Printer card: camera + progress); ✔ send with beambam's spool check (CLI or daemon, below; no real print yet); ✗ filament control |

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
  CLI slice (7 colours: 10.19 h / 145.7 g / 425 changes / same nozzle split with 0 and 1) — but the Bambu CLI resets it to 0 unless passed as `--has-filament-switcher=1` (BambuStudio.cpp, found 2026-11-25), so both runs were probably sliced without the switch
  (TODO: re-measure). Exports leave the project's value alone; a print project made after reading the
  printer records the printer's switch (`pipeline._slice_settings`), and beambam's slicer passes it on. Without a switch each unit feeds one nozzle; pokefit shows (R)/(L) per spool
  but does not yet constrain the planner (`printer.py` TODO).

## Control path: CLI or daemon (2026-11-25)

pokefit's local server talks to beambam one of two ways (`pokefit/printer.py`):

| | CLI (default) | Daemon (`POKEFIT_BEAMBAM_URL`) |
|---|---|---|
| Start | nothing to run | `beambam daemon --http 127.0.0.1:8765` (keep it running) |
| State | `beambam status`, new MQTT login each time (≈ 2.6 s) | `GET /state` from the daemon's cache (0.06 s on the X2D) |
| Spool check | `beambam ams map FILE --slot … --json` | `POST /print/plan {file, slot}` |
| Start print | `beambam print FILE --slot … [--yes]` | `POST /print {file, slot, confirm, accept_warnings}` → poll `/print/jobs/<id>` |
| Daemon elsewhere | — | files go up with `POST /print/upload` first; set `POKEFIT_BEAMBAM_TOKEN` |

Either way the page shows beambam's plan before "Start print"; errors (empty slot, other material, a tray that
cannot reach the nozzle) disable it and warnings are accepted by the click.

### The site (pokefit.party) on a device that runs beambam

The site cannot slice or reach the printer, so in the browser engine the page itself talks to a beambam daemon on
the same device (`web/src/beambam.ts`; opt-in under Print → "Connect to beambam", remembered in the browser):

1. `beambam daemon --http 127.0.0.1:8765 --cors-origin https://pokefit.party` (beambam ≥ 1.6.2: earlier versions drop the track switch when slicing). beambam refuses
   POSTs from any web origin not named there, so other pages cannot drive the printer. The site calls
   `http://127.0.0.1` from https without a mixed-content block (loopback counts as secure); Chrome asks once to
   let it reach apps on the device, and the daemon answers the Private Network Access preflight.
2. Connect: `GET /state` → the engine (`webapi.set_printer`) reads the spools, as "Use my printer's AMS" does
   locally, so the colours match the loaded spools. Without system profiles in the browser each spool prints with
   the reference project's slot of its filament id, else of its material (`presets._project_slot`).
3. Slice: the engine writes the print project (`pipeline.print_project`: the local Slice's auto-supports variant,
   with the Print card's "Print settings" — supports where needed / from the plate only / off, layer height, walls,
   infill, brim: `Params.supports` … `brim`, written by `print_overrides`, also into exports; "Print weak parts
   solid" adds its modifier volumes as the export does) and the page sends it to `POST /slice`; the daemon's
   headless Bambu Studio slices it on the device.
4. Send to printer: the confirm panel (`web/src/printconfirm.ts`, shared with the local version) shows each
   filament → spool with beambam's `/print/plan` verdict, reads the spools every 4 s (6 s locally) and checks again
   when they change; "Start print" → `POST /print` → poll `/print/jobs/<id>`.
5. Nozzles (two-nozzle printers): each filament shows the nozzle Bambu sliced it for, as Left | Right. Picking the
   other one is a new print (tool order, purge, time), so the page slices again with `Params.nozzles` — Bambu's
   "Manual" grouping (`filament_map_mode` = Manual, `filament_map` 1 = left / 2 = right; the CLI reads both from
   the project) — and the panel's time, weight and spool check follow the new file. "Let Bambu choose" goes back
   to "Auto For Flush". With a Filament Track Switch every AMS spool reaches either nozzle (beambam
   `read_topology`); without one beambam's check refuses a spool its filament's nozzle cannot reach.

6. Project vs printer (`pokefit/projectsync.py`, `web/src/projectsync.ts`): once the spools are read (here, or
   "Use my printer's AMS" locally) the Print card compares the project's filament list slot by slot with the AMS
   trays in order (A1 … D4; the external spools are no slot) and offers "Use the printer's filaments" (filament i
   becomes tray i: its preset, material, colour; the project's slots past the spools stay, so objects assigned to
   them still have a filament) and "Download the updated project" (a copy of the user's .3mf with the new
   settings, else a project .json). The plate is Print settings → Plate (`Params.bed_type` → `curr_bed_type`);
   the printer's plate marker (`device.plate.cur_id`) is named when it is a confirmed one (`printer.PLATE_MARKERS`:
   P0301 = SuperTack, the same table as beambam's), and a project set up for another plate is flagged; taking the
   printer's filaments takes its plate too. On the site, with beambam's "project-sync", the slots get the spools'
   system presets from beambam (POST /project/update), which has the printer's profiles.

The site probes nothing until the user connects: probing 127.0.0.1 for every visitor would show them Chrome's
prompt. With beambam's "camera" capability the Printer card shows the daemon's camera relay (GET /camera.mjpeg)
there too, read with fetch (`web/src/mjpeg.ts`, a port of beambam's site reader) so it carries the token and
Local Network Access; without it, status only. A daemon on another machine needs `--auth-token` (beambam refuses a
LAN bind without one): the Print card asks for it when the address is not this device, keeps it with the link in
this browser, and sends it on every call (`Authorization: Bearer`).

**On a computer (anyone's, 2026-11-26).** beambam ≥ 1.9.0 slices off the phone: `POST /slice` uses the official
Bambu Studio `beambam slicer install` downloads (CI slices with it on Linux, macOS and Windows), or an installed one.
With [uv](https://docs.astral.sh/uv/):

```bash
uvx beambam init               # once: finds the printer, asks for its access code
uvx beambam slicer install     # once: Bambu Studio for slicing (~200-430 MB)
uvx beambam boo --http 127.0.0.1:8765 --cors-origin https://pokefit.party --camera
```

then Print → Connect to beambam at `127.0.0.1:8765` in Chrome or Edge (allow the local-network prompt). The
camera needs `ffmpeg`. Start needs the printer to take beambam's commands: the key from `beambam key`, or the
printer's Developer LAN Mode; without either, everything before Start works. To use the phone's beambam from a
laptop instead (the phone slices), start it on the LAN with a token (the Print card shows the command for an
address that is not this device) and enter the phone's address and the token.

## Next steps (need the user / the printer)

1. Done 2026-10-19: credentials in `~/.x2d/credentials` (outside the repo); "Use my printer's AMS" reads the
   X2D's 4 AMS units (16 spools) through `POKEFIT_BEAMBAM=C:/git/beambam/.venv/Scripts/beambam.exe`.
2. **Send to printer**: the local Slice step now writes the print file (`…_auto.gcode.3mf` /
   `…_off.gcode.3mf`, with the plate thumbnails the touchscreen needs) and offers it as a download; next is
   `beambam print <file> --ams-mapping …` from the AMS match. Needs the per-install signing key
   (`~/.x2d/printer_sign_key.pem` + `printer_cert_id.txt`, extracted from Bambu Handy on a phone via adb,
   `beambam key`): not on this PC yet — the user's call. beambam's `docs/COMPARISON.md` now records a pure-LAN
   print start verified on the X2D (err 0, after the `url_enc` and `ftp:///cache/` fixes), while a docstring
   in `print_job.py` still says the dual-nozzle `project_file` returns 84033544: confirm with one real
   (user-approved) print before building the button.
3. Done 2026-10-19: **Printer card** (local): "Watch the printer" starts `beambam camera` as a child
   (`printer.CameraProxy`, MJPEG on 127.0.0.1:8766, process tree killed on stop / server exit) and polls
   `beambam status` every 15 s while the page is visible (`print_progress`: state, %, layers, time left,
   both nozzles' packed temperatures, bed). Read-only. Needed beambam `ebd1a6b` (local, not pushed): its
   camera pump used `select()` on a pipe, which Windows refuses (WinError 10038).
4. **Filament control**: `beambam ams-load / ams-unload` and `print.ams_filament_setting` (set a tray's
   type/colour) — a per-slot menu in that card. Every printer-changing action is user-confirmed.
5. **Auto colour optimisation** across the loop: palette size from the cost table (done: "auto"), snap colours
   to loaded spools ("Use these filaments' colours"), and suggest which spools to load (the colours the
   figure needs that the AMS lacks).
