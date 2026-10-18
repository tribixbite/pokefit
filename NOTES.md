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
- Setup once: `pip install -e .` (repo root) and `cd web && bun install && bun run build`.
- UI: `python -m pokefit serve --open` → http://127.0.0.1:8765/ (drop a .3mf/.stl/.obj, or
  `?path=C:/path/to/file.3mf` with `+` written as `%2B`). Outputs land in `pokefit_out/<name>/`.
- CLI: `python -m pokefit fit <figure.3mf> --ball 0.6 --washer M6 [--scale 85%] [--magnets 0-3]
  [--washers] [--out DIR] [--slice] [--x10]`; `python -m pokefit ball` prints the cavity profile.
- Magnets (1–3): placed on a ray-cast height field of the underside (`magnets.HeightField`), so
  sloped/raised undersides work. Each pocket opens as low as possible, never below the bed
  plane, with ≥ `grip` (1.2 mm) of it walled; auto placement keeps openings ≤ 2 mm above the
  washer (UI offers raising it when nothing lower fits). Pocket z = opening height in reports.
- Export also writes `<name>_pokefit_<pct>_import.3mf` for dragging into an open Bambu project
  ("Import geometry only" keeps part filaments + paint, which then mean that project's slots):
  filaments renumbered to the closest-colour slot of the reference project (CIELAB ΔE ≤ 25, else
  a new slot), overridable per colour in the UI. Bambu 2.8.2 shows "The 3mf file has invalid
  config, load geometry data only" on EVERY geometry import (bambulab/BambuStudio#11927, fixed
  upstream Sep 2026) — harmless.
- Multi-figure 3MFs: picker card at the top of the panel; meshes are parsed per object on demand
  (14 × 500k-face file opens in ~6 s; was 99 s because the regex fast path never matched).
- Gates: `python -m mypy`, `python -m pytest` (82 incl. slow real-file/Bambu tests, ~1 min),
  `cd web && bun run typecheck && bun test && bun run build`.
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
- pokefit known limits (not bugs): ring radius is the minimum over all angles (the latch
  notch sets it, so a figure could be a little bigger turned away from the latch — would need
  yaw-dependent ring profile); manual pockets stay at absolute mm when the size changes;
  the height field sees only the FIRST solid span above each point (material above an air gap
  is ignored — conservative); the import file's slot match uses the reference project
  (`pokeball60goodPrinted.3mf`), not whichever project is open in Bambu; the actual GUI
  drag-in import was not automated (CLI slice of the import file verified: 1h05, 10.7 g);
  analysis ~6–10 s per change on a 350k-face figure (islands section ~2.4 s, mesh checks
  4.4 s once per file).
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
