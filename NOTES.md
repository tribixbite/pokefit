# Pokeball / pokefit — working tracker

Agents: read `AGENTS.md` first (rules, setup, architecture, gotchas). This file holds open work and the
measurements decisions still need. Paths under `~/Downloads`, `%APPDATA%` or `C:\git\pokeball_backup_*`
exist only on the original Windows machine (ask the user for the files).

## To do (prioritised)
1. **Printer link next steps** (`docs/integration.md` "Next steps"; needs the user): beambam credentials on
   this PC (`~/.x2d/credentials`) to try "Use my printer's AMS" for real; then send-to-printer (the print
   file `.gcode.3mf` now comes out of Slice; next `beambam print … --ams-mapping`) — blocked in beambam by the
   dual-nozzle `nozzle_mapping` print start (err 84033544); live view (`beambam camera` MJPEG on :8766) and
   AMS load/unload in a local "Printer" card. Ask before any action that changes the printer.
2. **Palette choice** (`pokefit/gltf.py`): a chosen count now merges by a softened Ward cost (3 colours keep
   blue / cream / brown; black eyes come in at 4). "Auto" still picks by colour distance (7 for the
   Squirtles) and the Colours card stars the best-value count; ask the user whether auto should default to
   the starred count instead (it trades detail for hours of print time).
3. **Photos -> 3D colour, what is left** (`pokefit/imageto3d.py`): photo colours now replace the model's
   (eyes, iris, shell plates right); left: glossy highlights in the photos become light specks (a few
   survive despeckling on the head). Single photo: the photographed side now takes the photo's colours
   (21 % of the Squirtle's vertices from its side photo). Auto-turn a single-photo result to face front
   (now a 90° turn by hand for a side photo).
4. **Profile views from ordinary photos**: one photo from any angle now works (single-view Pixal3D + MoGe
   FOV). Not done: combining several arbitrary-angle photos (would need their camera poses — e.g. estimate
   azimuths, or synthesise the rig views with SV3D / StableZero123, which ComfyUI ships natively).
5. **Conform limits**: designed for the largest-fit scale and the default washer's floor (a smaller chosen
   size keeps the same shape); the flat base's walls are vertical (a pedestal between the legs).
6. **Big meshes** (200k-face photo Squirtle: first analyse 18.7 s = conform 6.1 s, colour-change table
   4.5 s, pockets 2.5, printability 2.5; 684k-triangle LEGO GLB (16k overlapping bricks): 65-69 s, was 289 s
   under shapely 2.2 — islands now section each body when loops cross). Next: conform's height fields
   (7 per run, 2.4 s), `prepare_points` (1.4 s), the LEGO's islands sections (~35-49 s, per body).
7. User's call, ask before doing: LICENSE file (public repo is all-rights-reserved by default; note beambam
   is AGPL, so pokefit only calls it as a program); Cloudflare "Always Use HTTPS" (off; the page redirects
   itself); turning on `has_filament_switcher` (track switch) in the Bambu project (it is off there).
8. craftmatic side of the handoff (its repo, ~15 lines, snippet in `docs/integration.md`): ask before
   changing that repo (it has an unpushed local commit).

## Mesh repair (SketchUp 2017 → STL exports; files on the original machine only)
SketchUp exports here are at **16× working scale** and often non-manifold.
- Check: `python scripts/diag_stl.py <file.stl>` (needs `pip install trimesh pymeshfix manifold3d`)
- Fix: `python scripts/repair_shell.py <in.stl> <out.stl> --scale 16 --flat-z <bed plane z in source units>`
  (`--no-trim` if the part has no flat bed spot). Output is mm, min corner at origin.
- `redsmaller.stl` is repaired (watertight, 46.51 × 45.02 × 21.06 mm, 3350.9 mm³).
  The unrepaired export is in commit `6fe8dd9`.

## pokeball.3mf (MakerWorld "functional Pokéball - no glue", Bambu project)
- Parts: `python scripts/extract_3mf.py pokeball.3mf parts` → one STL per object in
  `parts/` (git-ignored, derivable). Shell = `0_ball_shell_2x` (ids 2 and 8 are the same
  mesh, printed twice). Native shell: outer R 38.74, inner R 36.81, wall 1.93 mm;
  ×0.6 = the 60% size of red.stl. Model's assemble_item transforms are an exploded
  layout, not the assembled ball.
- Flat bottom: `python scripts/flat_spot.py parts/0_ball_shell_2x_id2.stl <out> --scale 0.6 --disk 8 --fill 10`
  (final-mm diameters). Outputs `shell_flatbottom_60.stl` / `_100.stl`: 8 mm outer disk,
  10 mm inner flat ceiling, pole wall 1.39 mm @60% (orig 1.16) / 2.07 @100% (orig 1.93).
  Print ONE of these (bottom half) + one original shell. Rim on z=0 like the 3MF.
- In SketchUp 2017 a 3MF opens only via Extensions → Universal Importer (installed);
  for one part, import the extracted STL instead.

## Avoiding broken exports (SketchUp 2017 Make — no built-in Solid Tools)
- Defects in both redsmaller/redsphere sit at the flat-spot cut: leftover sphere cap below
  the plane, coincident stray faces, holes ⇒ the flat spot was made with a plane +
  Intersect Faces + manual delete. Cut with **Eneroth Solid Tools** instead (installed
  2026-10-07 from github.com/Eneroth3/eneroth-solid-tools 3.0.1; Tools → Eneroth Solid
  Tools → Union/Subtract/Trim/Intersect, plus a toolbar). Click target first, then
  modifier; Subtract deletes the modifier, Trim keeps it. Both must be solid groups.
  Its menu.rb hides the tools on Make (`if Sketchup.is_pro?`, an Extension Warehouse
  hosting rule; the booleans are pure Ruby). Installed copy is patched; patched package at
  `%USERPROFILE%\Downloads\ene_solids_v3.0.1_make.rbz`. After any reinstall/update:
  `python scripts/patch_ene_solids.py "%APPDATA%/SketchUp/SketchUp 2017/SketchUp/Plugins/ene_solids/menu.rb"`.
- Entity Info must read "Solid Group" (with a volume) before export; if not, run
  Tools → Solid Inspector² (installed 2026-10-07 from github.com/thomthom/solid-inspector
  v2.5.1, no TT_Lib² needed).
- Vertices ~0.02 units apart on the rim at 16× ⇒ still near SketchUp's merge tolerance;
  a larger working scale (model mm as metres, ×1000) gives more margin.

## pokefit — fit a figure in the ball, magnet pockets, printability
Spec `docs/superpowers/specs/2026-10-16-pokefit-design.md`, plan `docs/superpowers/plans/2026-10-17-pokefit.md`.
- **Live: https://pokefit.party** (public repo github.com/tribixbite/pokefit, GitHub Pages via
  Actions, custom domain from `web/public/CNAME`; DNS is Cloudflare-proxied CNAME @ →
  tribixbite.github.io, so HTTPS is Cloudflare's — GitHub's own cert stays unissued). Every push
  to `main` runs `.github/workflows/deploy.yml` (mypy, pytest, typecheck, bun test, build,
  Pyodide smoke test) and redeploys. Watch with `gh run list -R tribixbite/pokefit`.
- The site runs the same Python engine in a Web Worker (Pyodide 0.29.5 from jsDelivr,
  manifold-3d WASM for booleans via `pokefit/geom.py` → `_manifold_js.py`, vendored
  trimesh 5.1.1 / networkx 3.4.2 wheels, `pokefit.zip`, build-time `config.json`). `api.ts`
  uses the local server when the page is served by it, else the worker. No Bambu slicing on the
  site. Browser timings: dragon load+analyse 24 s (native 12), re-analyse 13 s.
- Public-repo hygiene: history was rewritten 2026-10-18 (author → tribixbite noreply; the
  third-party `pokeball.3mf`, `*.skp` and derived STLs removed). Original history + files are
  in `C:\git\pokeball_backup_20261018\` (bundle verified). Model files are git-ignored; the
  ball ships only as measurements (`pokefit/profiles/ball_functional_pokeball.json`).
- Setup once: `pip install -e .` (repo root) and `cd web && bun install && bun run build`.
- UI: `python -m pokefit serve --open` → http://127.0.0.1:8765/ (drop a .3mf/.stl/.obj/.glb, or
  `?path=C:/path/to/file.3mf` with `+` written as `%2B`). Outputs land in `pokefit_out/<name>/`.
- CLI: `python -m pokefit fit <figure.3mf> [--ball 0.6 | --diameter 44] --washer M6 [--scale 85%] [--magnets 0-3]
  [--rotate-x/-y/-z DEG]
  [--washers] [--out DIR] [--slice] [--x10]`; `python -m pokefit ball` prints the cavity profile.
- Magnets (1–3): placed on a rasterised height field of the underside (`magnets.HeightField`), so
  sloped/raised undersides work. Seat mode `pocket_seat` (CLI `--magnet-seat`):
  `washer` (default, user's request 2026-10-18): every opening at z 0, magnet rests on the steel;
  under floating parts it shows below them; ≥ 0.5 mm (`MIN_CUT_GRIP`) must be walled, less than
  `grip` (1.2) is cut with a "glue it" warning; scoring penalises showing magnet 8 mm/mm.
  `raised`: opening as low as possible with ≥ `grip` walled, ≤ `max_seat` (2 mm). Eevee
  (Downloads/eevee+v3 (1).3mf): washer → 2 pockets between belly and front legs, 0.8 mm walled
  (paws are ~4 mm wide, too narrow for a walled Ø4.18 pocket); raised → one in the tail at 1.2 mm.
- Islands/overhangs are "info" details; "Supports needed" is the single amber warning.
- Colorize (`pokefit/colorize.py`, UI `web/src/colorize.ts`): `Params.colours` (palette) + `Params.paint`
  (PaintOp list: part, face, filament, mode part|region|piece|brush|face, angle, radius in file mm), applied
  by `Session.figure_parts` on top of `unit_parts`, written as whole-triangle `paint_color`. "part" = spread
  except across concave edges sharper than angle (default 15°); on the plain Eevee STL
  (Downloads/eevee+v3_stls.zip) it isolates ears, eyes (67 tris each), nose, bib (631), body. Taps: three.js
  raycast -> `/api/pick` (KD-tree + exact point-triangle, normal-facing filter). Verified: exported 3MF sliced by
  Bambu CLI with 3 filaments (9.1 / 1.35 / 2.8 g). Settings: `adapt_settings` (reference) or `resize_filaments`
  (file's own settings) grow to the palette length.
- Export also writes `<name>_pokefit_<pct>_import.3mf` for dragging into an open Bambu project
  ("Import geometry only" keeps part filaments + paint, which then mean that project's slots):
  filaments renumbered to the closest-colour slot of the reference project (CIELAB ΔE ≤ 25, else
  a new slot), overridable per colour in the UI. Bambu 2.8.2 shows "The 3mf file has invalid
  config, load geometry data only" on EVERY geometry import (bambulab/BambuStudio#11927, fixed
  upstream Sep 2026) — harmless.
- Multi-figure 3MFs: picker card at the top of the panel; meshes are parsed per object on demand
  (14 × 500k-face file opens in ~6 s; was 99 s because the regex fast path never matched).
- Gates and end-to-end checks: see `AGENTS.md` and `.claude/skills/pokefit-verify`. Export check:
  `python scripts/verify_export.py <export.3mf>` (paint summary + Bambu slice incl. colour changes/purge).
- Social preview: `cd web && bun run og`, open http://127.0.0.1:8791/ → writes public/og.gif + og.jpg
  (original creature; no third-party model). Icons: `python scripts/make_icons.py`.
- Slicing uses the user's X2D/PETG/0.16 project settings (`pokeball60goodPrinted.3mf`, or the
  bundled copy `pokefit/profiles/x2d_petg_printed_ball_shells.json`), reduced to the figure's
  filament count. Bambu CLI facts: needs `Application=BambuStudio-…` in 3dmodel.model or it
  ignores embedded settings; writes result.json to its cwd; sometimes hangs AFTER writing
  result.json (killed after a 5 s grace).
- Dragon (`~/Downloads/dragonwithegg+eyes+multicolor.3mf`) at the 60 % ball, M6: 94.8 % =
  34.1 mm, pockets on steel, Bambu 1× 1h04 / 10.8 g (prime tower 19 min of it), 10× 4h16 /
  38.9 g, no supports generated. Washer table: US 1/4″×1″ fender 34.3 mm, M6 34.1, M8 33.9,
  US #10×3/4″ 33.6, M5 off steel.

## Open
- Site assets are cache-busted per build (`web/src/version.ts`, `?v=<hash of inputs>` from build.ts): Cloudflare
  rewrites .js/.wasm/.zip to a 4 h browser cache. Discord caches link embeds per URL; an old embed of
  pokefit.party (GIF + JPEG) can persist until Discord refetches — only og.gif is declared now.
- pokefit known limits (not bugs): ring radius is the minimum over all angles (the latch
  notch sets it, so a figure could be a little bigger turned away from the latch — would need
  yaw-dependent ring profile); manual pockets stay at absolute mm when the size changes;
  the height field sees only the FIRST solid span above each point (material above an air gap
  is ignored — conservative); the import file's slot match uses the reference project
  (`pokeball60goodPrinted.3mf`), not whichever project is open in Bambu; the actual GUI
  drag-in import was not automated (CLI slice of the import file verified: 1h05, 10.7 g);
  analysis ~6–10 s per change on a 350k-face figure (islands section ~2.4 s, mesh checks
  4.4 s once per file).
- GLB/glTF colour (`pokefit/gltf.py`, doc at top): one part, colour as split Bambu paint (`paint.encode_tree`,
  child geometry/order from BambuStudio `TriangleSelector::perform_split`/`serialize`). `Params.texture_colours`
  (UI "Filaments" slider, CLI `--colours N`) re-cuts the palette (~0.3 s; drops user paint). Split paint now
  survives re-meshing and mirrors (`paint.remap_tree`; a mirrored 2-side split's quad collapses to its larger
  state) and shows in the preview (`pipeline.painted_mesh`). Squirtle test file (Manex3D AI GLB, colour in the
  emissive texture; copy in scratchpad `manex/manex.glb`, not committed): 19.8k tris, depth-3 samples, auto 7
  colours; Bambu CLI slice 60% ball: auto 7 = 233 changes / 129 g / 9.0 h, 4 colours = 127 / 74 g / 5.4 h
  (figure itself ~3 g; the rest is purge). Opens in ~15 s in the browser engine, ~5 s locally.
  Its base sits 1.7 mm above the washer except a tiny bed contact → "raised" magnet seat needed.
  Verified live 2026-10-19 (browser engine): opens, slider re-cuts to 4, no console errors; Colours card fits
  at 390 px. The browser's palette differs slightly from the local one (8,715 vs 8,481 painted triangles at
  auto 7; colour 6 grey-green vs dark brown): Pyodide's scipy k-means lands differently with the same seed.
  Each engine is self-consistent (export uses the same engine); make it deterministic only if users compare.
- Toolchange model (`pokefit/toolchange.py`), Bambu CLI on the X2D project. Manex Squirtle palette sizes
  1/2/3/4/7: 0.67/1.57/2.35/5.45/10.19 h, 2.8/17.3/28.9/73.8/145.7 g (fit: 104.6 s and 0.43 g per filament
  change incl. flush, 18 s per nozzle switch; predictions within 0.4 h / 12 g). Photo Squirtle, 5 colours, all
  16 nozzle splits: 6.94 h (Bambu's auto 1,2,2,2,2 = the planner's pick) to 10.79 h / 156.9 g (one nozzle).
  Manex 7 colours: Bambu auto 1,1,1,2,2,1,2 = 10.19 h beat the planner's 1,2,1,2,2,2,2 = 10.34 h, so exports
  keep Bambu's Auto For Flush. Sweep script: scratchpad `nozzle_sweep.py` (plate `filament_maps`, Manual).
- Bambu presets (`pokefit/presets.py`): user's `printer_config/Filament presets.zip` (17 X2D presets, git-ignored)
  resolves exactly against `C:\Program Files\Bambu Studio\resources\profiles\BBL`; in the browser it is
  "approximate" (keys not in the preset come from the project's closest slot). 10 of the 17 set 0 °C for the
  Supertack plate the user's project uses, and Bambu refuses them (code -61, verified with eSUN PETG Basic);
  `redpetg` slices (box: 0.40 h, 1.7 g). Limits: a printer preset for another printer is skipped (no
  project is built from presets alone); with several printer presets none is applied.
- Conform defaults on (Params `bend` auto, `flat_base` washer OD, `flat_depth` 1.2, `flat_fill` 2.5). Results:
  Manex Squirtle +7.8 % (29.6 mm), 2 pockets on the steel (were 0–1 in the tail); photo Squirtle +7.1 %
  (31.6 mm), 2 pockets; dragon and Eevee: bend skipped (< 1 % gain); Eevee gets a 1.2 mm press only.
- Photos -> 3D: ComfyUI v0.42.1 at `C:\git\comfyui-3d` (own venv, torch 2.14.1+cu130); the older
  `C:\git\ComfyUI` (Mar 2025) is the user's, left untouched. First run 203 s for 3 views while the user's
  llama-server (Qwen 27B, port 8080) held 12 GB of the 16 GB VRAM. Squirtle photos -> 200k faces, watertight
  (solid remesh), 8 auto colours, slices in Bambu. Photos are in `imageto3d/` (git-ignored).
- glTF not supported: Draco/meshopt compression, .gltf with external .bin/textures. `planar._xor`
  repairs/snaps loops GEOS rejects (DamagedHelmet.glb crashed islands_check).
- (SketchUp, original machine) **CleanUp³ needs TT_Lib² ≥ 2.11** — CleanUp³ 3.4.3 files are in
  `%APPDATA%\SketchUp\SketchUp 2017\SketchUp\Plugins\tt_cleanup`, but TT_Lib² is not
  installed. Not on GitHub (`thomthom/tt-lib` is the different tt-lib³; `thomthom/cleanup`
  master 3.4.6 still requires TT_Lib²), no Wayback copy. Only source: SketchUp's
  Window → Extension Warehouse (free Trimble ID sign-in, installs directly) or
  sketchucation.com/plugin/726-tt_lib.
- `redsphere.stl` (untracked, 16×): 12 holes (11 around the flat spot at z=-93.22, one
  3.5×2.2-unit hole on the outer wall at y=-357, z=158), 15 non-manifold edges, 10 loose
  zero-thickness fragments on the flat-spot plane, sphere cap dipping 4 units below it.
  Fix if wanted: commit it, then
  `python scripts/repair_shell.py redsphere.stl redsphere.stl --flat-z -93.22`.
- `redflat.stl`, `red2.stl` also non-manifold; not analysed.
- `redsmaller.stl` bed contact is a 35 mm² disc (~6.7 mm) — consider a brim.
