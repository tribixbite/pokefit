# Pokeball / pokefit — working tracker

Agents: read `AGENTS.md` first (rules, setup, architecture, gotchas). This file holds open work and the
measurements decisions still need. Paths under `~/Downloads`, `%APPDATA%` or `C:\git\pokeball_backup_*`
exist only on the original Windows machine (ask the user for the files).

## To do (prioritised)
1. **First real print** (Send to printer is built, `fffd03c`: with the AMS read, Slice -> "Send to printer…" -> confirm
   panel (filament -> spool, plate, bed temp) -> "Start print" = `beambam print <file> --slot …`; credentials and signing
   key in `~/.x2d`, never in a repo). "Start print" has never been pressed. **The first real print needs the user's OK in
   chat, every later one their click.** It also settles whether a dual-nozzle start still fails (beambam notes disagree,
   err 84033544). Then: AMS load/unload. `beambam print --dry-run` (purge check only, touches no printer) is safe to use.
2. **Palettes on generated figures** (the open quality gap; shapes are good). Auto = best value, capped at 2 h of colour
   changes (`toolchange.AUTO_MAX_MIN`); "Keep details" (`texture_details`, off by default) adds filaments on top of the
   auto count. Weak spots seen in the 12-figure batch (below): Pikachu's red cheeks (~1 % of the surface, close to its
   dark brown) merge into brown below 5 colours and +1 detail picks a 0.08 % beige; Squirtle at auto 2 merges belly,
   shell and eyes into one tan; Mew finds no detail. Ideas: score detail candidates by hue distance and share, not
   only ΔE to the group mean; ask the user whether "Keep details +1" should be the default (pupils on Squirtle cost
   1.98 -> 2.70 h, 20.9 -> 34.1 g).
3. **Generated-figure batch** (2026-10-21; photos in `imageto3d/<name>/`, git-ignored, Takara Tomy MonColle studio shots
   + a Jazwares Togepi; scratchpad `batch_figs.py` = generate + fit + slice, `sheet.py` = painted renders,
   results `batch3/summary.jsonl`, sheets `figures_1.png`/`figures_2.png`). Pixal3D multi-view 96-235 s each; Mew via
   `--clean` (3 photos) 231 s. Defaults, Bambu slices (auto | auto + 1 detail):
   | figure | mm | magnets | auto | h / g | +1 detail h / g |
   |---|---|---|---|---|---|
   | Squirtle (gloss-fixed) | 31.6 | 2 | 2 | 1.98 / 20.9 | 2.70 / 34.1 |
   | Charmander, `--pose` | 33.8 | 2 | 3 | 2.83 / 35.5 | 3.67 / 44.8 |
   | Pikachu | 36.0 | 2 | 2 | 2.01 / 22.8 | 2.33 / 28.6 |
   | Eevee | 29.8 | 1 (fill 4 mm) | 2 | 1.38 / 15.3 | 1.73 / 22.3 |
   | Jigglypuff | 29.8 | 2 | 4 | 2.23 / 28.8 | 2.42 / 28.9 |
   | Bulbasaur | 22.1 | 2 | 2 | 1.48 / 15.6 | 2.25 / 27.2 |
   | Charmander (arms up) | 25.0 | 2 | 3 | 2.00 / 26.0 | 2.34 / 30.6 |
   | Snorlax | 28.3 | 2 | 2 | 1.76 / 20.2 | 2.15 / 26.6 |
   | Psyduck | 33.0 | 2 | 3 | 1.91 / 24.1 | 3.27 / 42.6 |
   | Togepi | 30.7 | 2 | 2 | 1.63 / 16.5 | 3.11 / 38.7 |
   | Piplup | 37.8 | 1 (fill 6 mm) | 2 | 1.37 / 12.9 | 1.75 / 19.5 |
   | Mew | 39.6 | 0: FAIL | 3 | 3.54 / 47.1 | (no detail) |
   All need supports (chins, arms). Open: Mew (thin looping tail and feet: no pocket, small bed contact; glue it or a
   stand); Bulbasaur only 22 mm (long and wide); several "magnets overhang the steel" warnings (a larger washer).
4. **Photo pipeline**: Pixal3D multi-view + mirrored right is the default. `--pose` (OpenPose skeleton reference, arms
   down) turned the arms-up Charmander into a sitting, arms-down one: 33.8 mm vs 32.8, 2 magnets. Gloss
   (`imageto3d.gloss_free`) clears streaks on head and shell; a faint darker blotchiness remains on the Squirtle's head
   side where reflections were replaced. Side-view naming confirmed on a second figure (Mew). In-browser TRELLIS.2: only
   after the user switches Chrome to the high-performance GPU. llama-server is stopped (restart command in memory
   `local-installs`).
5. **Paint depth**: `mmu_segmented_region_interlocking_depth` 0 / 0.4 / 0.8 mm (max width 1.6) = 666.6 / 664.6 /
   665.8 min, 284 changes each (8-colour Charmander): no time effect; strength needs a physical test. Not exposed.
6. **Send to printer** now shows beambam's file check (`printer.print_file_check` = `beambam analyze --json`, file only).
7. **MakerWorld URLs**: not supported, by design (no public API; files need a Bambu Cloud login; Terms s.9 forbid
   automated download). The user downloads, then opens the file.
8. **Conform limits**: designed for the largest-fit scale and the default washer's floor; the flat base's walls are
   vertical (a pedestal between the legs). Fill where: washer / everywhere / circle / drawn outline (`Params.fill_area`,
   `fill_shape`). Magnet pads (`Session.pads`): on the washer seat each pocket's walled disc gets the gap fill too, so
   magnets are walled from the washer up; Downloads Charmander with a 10.7 mm fill circle: both pockets now pass
   ("Sits on the washer"; before: "only 0.5-0.7 mm walled, glue it"). The raised seat lifts each pocket only as far as
   its own spot needs (one over the flat base stays on the washer): by design, explained in the seat hint. Pads stay
   within the cavity's flat floor radius; not checked against a hand-placed pocket outside it (clipped there).
9. **Big meshes**: Simplify (`pokefit/simplify.py`, `Params.simplify`, Figure card "Simplify mesh", off by default,
   remembered; the analysis shows a "Dense mesh" hint above 300k triangles). UI: Low / Medium / High = 7 / 20 / 50 %
   of the file's triangles (`web/src/simplify.ts`; `Params.simplify` < 1 is a share), log slider 1-100 %, typed count.
   Mesh repair (`pokefit/repair.py`) runs on every load and boolean: the user's site export of MINI EEVEE at 100k had
   65 edges on >2 faces after welding (Bambu offered to repair it), every level now 0 open / 0 over / 0 flipped in the
   exported 3MF (scratchpad `manifold_check.py`); 100k export slices (0.42 h, 1.3 g at 25.4 mm). DamagedHelmet.glb:
   fully closed after load repair (was "not a valid solid"). Repair cost: 2.2 s on the 1.45M-triangle mesh unsimplified
   (0.15 s at 100k) on top of loading; not timed in Pyodide (smoke test passes). 1.45M-triangle MINI EEVEE at 100k,
   local engine (scratchpad `bench_simplify.py`): open + analyse 39.5 -> 11.5 s, prepare paint (mirror) 5.4 -> 1.0 s,
   tap 0.80 -> 0.03 s, re-analysis after a paint step 3.7 -> 0.3 s, export 13.5 -> 1.4 s (19.3 -> 2.5 MB); surface
   change 0.017 mm at 25.4 mm tall (exact closest-point check, rtree: max 0.024 mm of 40 before RegularizeLight).
   Watertight in, watertight out (fold cleanup). Local and Pyodide give identical triangles (same WASM).
   Open: (a) should Simplify default to on (e.g. 200k)? The user's call - it changes face numbers, so a session's
   paint steps belong to one setting; (b) of the remaining 11.5 s, parsing the 1.45M-triangle 3MF XML is the bulk
   (not measured separately); (c) a GLB loads its texture again on every Simplify change (k-means, a few s).
10. **Strengthen thin parts** (`pokefit/strengthen.py`, `Params.strengthen`, UI "Strengthen thin parts", off by default).
   Charmander at 26.9 mm, 1.6 mm: 5-6 % of the surface thin (flame tips, flame/tail join, toes, claws), up to
   0.37-0.45 mm out, +2.5 s local; smooth result (scratchpad renders `char_flame_zoom.png`). Synthetic checks
   (tests/test_strengthen.py): a 0.7 mm neck between a tail and a flame ball -> ~1.6 mm all along; a 0.6 mm free
   fin -> 1.3-1.45 mm near its join tapering to ~0.8 mm at the tip. Open: (a) never printed: needs a physical
   test (does 1.6 mm at the join stop the flame snapping off? PETG at 0.16 mm layers); (b) a sharp 90-degree join
   (a flat fin on a box) is under-thickened in its first ~half target (smoothed normals tilt into the box; TODO in
   `strengthen()`); (c) low-poly meshes move only their few vertices (the smoke critter reads 33 % "thin": box
   corners), so it suits dense sculpts; (d) measured at the asked height or the largest, not at a slider scale.
11. **Paint tools** (colorize). Wand (default; `Topology.wand`, ~0.4 s a tap on 100k triangles + ~0.1 s for the soft
   edge), Stamp (drag a box: ellipse decal), Brush, Fill, Mirror, Grow/Shrink, Pick colour, Undo/Redo, "Drop specks
   under", "Shape map" view (crease shading + relief contours). Wand, brush, stamp and grown steps paint soft edges
   (`colorize.Field` -> split paint via `paint.overlay`, depth 4, a stamp on big triangles up to 8); a fill bounded
   by colour fills up to an earlier soft edge. Grow/shrink: within 3-4 % of the asked mm (tests).
   - Dense Charmander (`printed/obj_2_Charmander mini.stl` at 100k): wand eyes are smooth ovals now (scratchpad
     `soft3/eyeL.png`); the white disc fills up to the blue's edge. Belly still runs over the claws (left off).
   - Downloads `Charmander mini.stl` (6.7k triangles): stamps give both eyes + white discs (mirrored), the flame is
     wand level +4 from the front, +4 from the back and two side stamps. Export `pokefit_out/charmander_lowpoly_stamp/`:
     1.80 h, 25.3 g, 21 colour changes, 137 split triangles, sliced by Bambu CLI. Open: (a) a crease's inner wall
     inside an eye's outline faces the view and takes the stamp; from some angles it shows as blue streaks beside the
     mirrored eye (`scripts/render_export.py --zoom=-3.0,20.5,8`); (b) the wand on this decimated flame is unstable
     (grow changed which candidate the back tap took).
   - MINI EEVEE with the wand (`pokefit_out/eevee_wand/`, 100k): one tap each (with Mirror) for ears, eyes, eye
     highlights, collar; the tail tip has no crease, so a stamp. 1.61 h, 20.4 g, 7 colour changes (the old part/brush
     job: 1.59 h, 20.1 g, 7).
   - Open: (c) mirror now prefers an upright plane within 1.5x of the best (`MIRROR_UPRIGHT`; the figure's up from its
     rotation), so a rotation after painting can change the plane on an ambiguous shape; (d) the Shape map view
     replaces the paint colours while on; (e) a soft edge's split triangles change the preview's geometry, so a wand
     tap re-sends the mesh to the view (not only colours).
12. User's call: Cloudflare "Always Use HTTPS" (dashboard → pokefit.party → SSL/TLS → Edge Certificates). Off now: an
   `http://` visit gets the page over HTTP and redirects itself with a script. Explained 2026-10-19; their account.

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
- Auto palette = best value (`toolchange.best_value_count`, decided at the fit limit; the user chose this
  2026-10-19). Manex Squirtle with today's palettes, Bambu slices: 1/2/3/4 colours = 0.67/1.76/5.32/6.35 h,
  0/0/120/151 changes; estimate 0/0/120/161 changes, +46/+255/+327 min vs sliced +65/+279/+341 -> auto 2.
  `has_filament_switcher` 0 vs 1: identical slice (7 colours 10.19 h / 145.7 g / 425 changes).
- Toolchange model (`pokefit/toolchange.py`), Bambu CLI on the X2D project. Manex Squirtle palette sizes
  (older palettes) 1/2/3/4/7: 0.67/1.57/2.35/5.45/10.19 h, 2.8/17.3/28.9/73.8/145.7 g (fit: 104.6 s and 0.43 g per filament
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
