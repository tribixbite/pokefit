# Pokeball (SketchUp 2017 → STL)

## Mesh repair
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
- Gates: `python -m mypy`, `python -m pytest` (86 incl. slow real-file/Bambu tests, ~1 min),
  `cd web && bun run typecheck && bun test && bun run build && bun run smoke` (engine under Node).
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
- Site: GitHub Actions warns that checkout/setup-python/deploy-pages@v4 run on Node 20 (forced to 24);
  bump the action majors when convenient. No LICENSE file yet (public repo = all rights reserved
  by default) — the user's call. Cloudflare "Always Use HTTPS" is off (the page redirects itself).
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
  Open: auto count favours fidelity over purge (could weigh area vs. change cost); the 3-colour cut merges
  cream into blue (closest-first merge) — a hue-aware merge might pick better small palettes. Its base sits
  1.7 mm above the washer except a tiny bed contact → "raised" magnet seat needed.
- Big glTF scenes are slow: a 600k-triangle LEGO GLB (Downloads/41714-1-dbix_conv_v3.glb) took 82 s to open
  locally. Not supported: Draco/meshopt compression, .gltf with external .bin/textures. `planar._xor`
  repairs/snaps loops GEOS rejects (DamagedHelmet.glb crashed islands_check).
- **CleanUp³ needs TT_Lib² ≥ 2.11** — CleanUp³ 3.4.3 files are in
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
