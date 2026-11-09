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
   stand); Bulbasaur only 22 mm (long and wide); several "magnets overhang the steel" warnings (measured with washers, before steel discs).
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
8. **Conform limits**: designed for the largest-fit scale and the figure's floor (`Session.floor`); the flat base's walls are
   vertical (a pedestal between the legs). Fill where: steel / everywhere / circle / drawn outline (`Params.fill_area`,
   `fill_shape`). Magnet pads (`Session.pads`): on the steel seat each pocket's walled disc gets the gap fill too, so
   magnets are walled from the steel up; Downloads Charmander with a 10.7 mm fill circle: both pockets now pass
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
   2026-10-27, user saw knobs on Pikachu (`Downloads/Pikachu_3MF(2)+gjhkddkhhfjljgmkdgfhjk.3mf`): the code had not
   changed since it was added (bit-identical output at 2e0c3c1, 35dc094 and HEAD, scratchpad `strengthen_cmp.py`);
   the knob was the push growing points and rims outward (lightning-bolt tail point from 1.6 mm, ear tips at 3 mm),
   worse at the 2.5-4 mm options added on request. Fix: push only across (`ACROSS`, exit distance along the inward
   ray), gated through the blend. Pikachu 3 mm: ear-tip move 0.89 -> 0.26 mm, surface moved 28 -> 25 %, largest push
   1.08 -> 0.87 mm; renders scratchpad `pikachu_ears.png`, `strengthen_before_after.png`. Open: (e) a fine point
   still rises up to 0.22 mm at 4 mm (tests/test_strengthen cone); (f) the Eevee nose grows a little at 3 mm.
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
12. **Split by piece + coloured STL** (2026-10-26). Split: `colorize.colour_pieces` -> `pipeline.pieces_of_colour` ->
   `POST /api/colour_pieces` / worker `colourPieces`; the UI (`colorize.ts splitColour`) appends one "fill" step per
   extra patch (shared `group`, undone/redone as one, palette restored) with the colour most unlike the palette
   (`colour.distinctColour`). A bucket fill now also takes its colour's slivers in split triangles round it
   (`SLIVER_RINGS`), so a split patch moves whole. Measured (scratchpad `real_split_stl.py`): Eevee multicolour 3MF
   black = 5 patches over 0.5 mm², each new filament's area equals its patch to 0.001 mm²; Manex GLB grey 11 patches;
   MINI EEVEE (single colour) 2 shells. Coloured STL (`pokefit/stlcolour.py`, `<stem>_colour.stl` in every export):
   parts joined (manifold union), 0 open / 0 over edges with exact float32 matching on all three files, Bambu CLI
   slices it; export +~5 s on the 1M-triangle Eevee (888k STL triangles). Open: (a) the STL's colours are read by
   MeshLab/VisCAM/SolidView, not by Bambu/Orca/Prusa (they want the 3MF); a coloured OBJ would carry colour into
   Bambu Studio 2.x / PrusaSlicer if the user wants that; (b) not opened in MeshLab here (decoder in
   `stlcolour.read_coloured_stl` + trimesh only); (c) split steps are "fill" steps at a triangle: a later Simplify
   change drops them like any paint. UI: every panel card collapses (`collapse.ts`, remembered); the busy pill is
   fixed at the top on phones (desktop: in the view, which never scrolls).
13. **Remove colour + re-opening exports** (2026-10-27). Remove colour: `Params.colour_map` (file filament n ->
   colour_map[n-1], applied in `Session.mapped_parts` before paint; `Session.extruder` = renumbered object filament
   everywhere a file is written), UI `colorize.ts removeColour` (closest by CIE76, later ones move down, paint steps
   and "Prints from" choices renumbered; Undo and Redo while nothing changed since). Old pockets:
   `pokefit/oldpockets.py`; round trips (export with pockets, reopen) on 7 figures found every pocket within 0.001 mm,
   the refilled meshes closed; the user's old exports: Eevee_pokefit_60 (2), MINI EEVEE 30k (1), Manex (3 at 2.09 mm);
   no false finds on 5 originals. Open: (a) a pocket cut by an older pokefit with a different shape (none known)
   would not be found; (b) a found pocket's diameter sets the pocket size for the session (the Manex export used
   2.09 mm); (c) more than 3 found pockets: the first 3 are placed. Busy pill: a child of `body`, `position: fixed`
   at every width (style.css `.busy`).
14. **Magnet count with press + fill** (fixed 563734b; details in its commit). Open: the fill also builds up over
   paws inside the fill area ("fill gaps up to X" literally: the multicolour Eevee stands on a ~6 mm block at the
   default); not yet looked at in Bambu Studio on the user's figures.
15. **Paint dock, "Prints from" fold, faster loads** (2026-10-30, bc6645b; details in its commit). Open: on a phone the
   open dock is ~205-246 px of a 460 px view. Load hotspots (CPython, Little+Pikachu 8.7 MB, before this round's fit
   speed-up): conform 5.5 s, height fields 3.3 s, 3MF parse 3 s, mesh checks 1.7 s, pocket cut 1.5 s, pads 1.2 s;
   engine import ~4 s in the browser (networkx cannot be dropped: trimesh path traversal). Scratchpad `prof_load.py`.
16. **Steel discs, flat floor, reshape a part, fine brush** (2026-11-04, b64a5a1; details in its commit). Thin discs
   D20 (0.8" x 0.01", default) / D15; washers kept. Auto flat floor per figure (`Session.floor`, `fit.best_floor`);
   `conform.lift_start` = floor reach - 2.1 mm (0.2 mm off cut a bend gain from 8 % to 0.4 %). Open: (a) a reshaped
   limb is not checked for running into the body; (b) the auto floor changes the shell per figure (a user reusing a
   printed shell must type its floor); (c) thin-steel hold strength not measured; (d) `Steel` labels show 20.32 x
   0.254; (e) Bambu 2.8.2.61 "invalid config, load geometry data only" on every geometry-only import cannot be
   avoided from the file (fixed in 2.9.1); the slicer CLI was not retested on 2.9.1.
17. **Crease noise, paint lag, export links/names** (2026-11-04, 1af978a; details in its commit). Open: a stroke with
   soft edges re-sends the whole preview mesh and rebuilds the BVH (main-thread time not measured); the rest of a
   paint step (~1.2 s in the browser engine on the 300k Jolteon) is mostly `paint.overlay` (Python recursion).
18. **Fill height, faster analysis, base widening, smoother thickening** (2026-11-05). Test figure: the Jolteon
   (`~/Downloads/Eevee+Evolutions.3mf`, simplify 300k), scripts in the session scratchpad (`jolt_fill2.py`,
   `jolt_edits.py`, `strength_probe.py`, `strength_ear.py`).
   - "Fill gaps up to" made the base taller than the gap: a cell was admitted by its lowest underside point and built
     to its highest. 0.5 mm fill: base 1.39 mm tall, edges under the belly 2.8-4.9 mm up; 2.5 mm: 5.04 mm tall.
     Now a cell needs ALL its underside within the fill: 0.9 mm (0.5 + 0.4 overlap) and 2.9 mm. Open: the default
     still rises to 4 / 6 mm when the magnets asked for do not fit (`_auto_fill`, reported in the conform check).
   - Analysis, first result after an edit (CPython, s): open 13.3 -> 9.3, fill 9.2 -> 6.2, rotate 10.0 -> 7.6,
     clearance 10.1 -> 8.5. How: conform's bend+press cached apart from its fill (`Session._shapes`); quick analysis
     leaves the pocket cut and the self-intersection check (pymeshlab, 2.5 s, local only) to the full one; cut
     parts' paint re-laid per distinct string and cached (`magnets._REMESHED`); `fit.prepare_points` pre-filters
     hull candidates (`_maybe_on_hull`, 0.28 -> 0.16 s). The full (second) analysis is slower by what moved (open:
     6.5 -> 11 s). Tried and dropped: SLSQP/secant fit search (no faster: the margin is kinked, `max_scale` is
     ~0.1 s already) and a narrower floor-search bracket (0.3 s saved, but it picked another floor: 29.72 -> 30.22).
     Open: `best_floor` (~2 s, 17 fits) and the auto-bend trial (~1 s, rejected for the Jolteon) dominate an open.
   - Reshape a part: Base width (x1-3) / Reach / Taper widen a limb only round its joint (`reshape.thickened`).
     Checked in the local UI on the Jolteon's ear (Joint 10 deg: 5.2 mm, 4 % of the surface). A tap on the inner ear
     takes only the panel inside its incised line (deeper than 60 deg); tap the rim.
   - Thicken thin parts is volumetric now (2026-11-05, user: 2 mm made the Jolteon's smooth ears scaly). The push
     sizes are measured as before; the part then gains a solid (`strengthen._shell`: offset of the surface by the
     push of the nearest point, plus each sample pushed >= 0.1 mm grown in front of it within its push, so a concave
     join is not shadowed by its unpushed body; 0.1 mm grid, two passes, marching tetrahedra, Taubin + meshopt tidy)
     united with it (manifold). Paint/taps name `Session.base_parts` (before strengthen); `figure_parts` = painted
     base parts + the same shells (`Session.thickened`), paint carried by `remesh_part`. The incised line round the
     ear's panel was gated out (its rays read "along the sheet") and left a seam: `_fill_grooves` closes the gate
     over one edge between vertices whose smoothed normals agree. User's Jolteon (object "2" of
     `~/Downloads/Eevee_Evolutions_pokefit_60_import__2_pokefit_60_import.3mf`) at 2 mm: ear smooth front and back
     (scratchpad `ear_vol.py OUT 2.0`, renders `v11/`); 26 s locally (44 % of the surface thin; threads in cKDTree
     queries; Pyodide single-threaded, not timed), 296k -> 439k triangles. Open: (a) where many fine fur spikes meet
     the head at the ear base, spike fragments poke through the shell's fading edge (ragged look, sub-0.1 mm);
     (b) a cone's point rises 0.26 mm at 4 mm (0.11 at 3, 0 at 1.6; the groove closing reaches a point's cap);
     (c) thin spikes become rounded sausages (that is what 2 mm on a 0.5 mm spike is); (d) printed black showing
     through the ear = paint depth (`mmu_segmented_region_max_width` 0 = unlimited): use Paint depth 0.4 mm (not
     sliced or printed to confirm).
   - 2026-11-07 (user: "thicken thin parts doesn't work anymore"): it worked, but ran inside the preview call with
     no busy pill: 57 s of nothing on the Vaporeon in the browser engine (then 10 s of analysis). Now the pill shows at
     once ("Thickening thin parts…"); the 4D ball search takes `KD_EPS` 0.5 (single-threaded 47 -> 35 s; the same on
     the 3D search left a fin unthickened); it is no longer remembered (`prefs.ts`): a remembered 2 mm ran on every
     figure opened. Open: still ~45 s in the browser engine on a 300k figure (`_shell` KD queries, 780k fine points).
   - Paint (2026-11-05): "Drop specks under" cleared an Umbreon's wand-painted eye ovals (0.3-0.6 mm², under the
     0.5 default at print size): a patch that is at least half of the paint step that last covered it is now kept
     (`PaintState.owner`/`step_area`/`deliberate`); only leftovers of a step and the file's own specks go. Grow /
     Shrink step 0.1 mm. Reshape's highlight showed specks on the head: it indexed the preview mesh, whose split
     paint is drawn as sub-triangles, by the part's triangle numbers; `limb_region` now answers `view_faces`
     (`PreviewMeshes.sources`), checked in the local UI on the Jolteon's ear.
19. **Eevee evolutions batch** (2026-11-06, fourth pass; user's request: official colours, 300k, 1-2 pockets as large
   as fit, like `printed/umbreon3.3mf`; no flat feet, no gap fill / filler; paint on the model's lines; inspect each
   in pokefit.party). Source `~/Downloads/Eevee+Evolutions.3mf`; results `~/Downloads/eevee_evolutions_pokefit_300k/`
   (`<name>_pokefit_60.3mf`, `_import.3mf`, `review_<name>.png`; older versions with pads in
   `superseded_2026-11-06_with_pads/`). Colours from official Pokedex art (assets.pokemon.com .../full/<n>.png).
   Scripts in the session scratchpad: `fig2.py NAME [export]` (`FIG_NOPAD=1` = `flat_base` 0 and `flat_fill` 0, to
   `batch/export3_nopad/`), `insp.py NAME` close-ups, `gx.py` region ids, `live_shots.py OUT FILES` (Playwright +
   real Chrome on pokefit.party: 8 orbit shots per figure; `sheet.py` contact sheet).
   - The filler the user saw in pokefit was the APP: reopening an export re-applied the default press + gap fill.
     Fixed in 6ba9b4b (exports record `Metadata/pokefit.json`, report `object.made_with`, UI applies it, not
     remembered). Live-checked: all 8 open with flat base 0 / fill 0.
   - pokefit.party auto-simplifies a 300k file to 200k on open (paint resampled: edges look ragged that are clean in
     the file). Inspect with Simplify off (prefs `{simplify: 0}`).
   - Paint helpers added this pass: `band` (mask + smooth offset band via a smoothed per-vertex distance field: eyes,
     inner ears, gem, head leaf), straight soft cuts across ribbon ends (Sylveon stripe/fork/pink band), Glaceon tail
     crystal facets at 1.2 degrees (front diamond + back facets). Nearest-face and ray "through the blade" guesses
     failed (5 of 99 mm2): the tip is a 4-sided crystal, not a blade.
   | figure | mm | pockets (no pads) |
   |---|---|---|
   | Espeon | 32.6 | 1 |
   | Flareon | 31.6 | 2 |
   | Glaceon | 34.5 | 2 |
   | Jolteon | 31.6 | 1 |
   | Leafeon | 34.3 | 1 |
   | Sylveon | 38.5 | 1 |
   | Umbreon | 38.5 | 2 |
   | Vaporeon | 27.9 | 1 |
   2026-11-07 (user: Umbreon's rings warped and moved, one tail ring missing, the other partial; check the others):
   the Umbreon came from the user's `umbreon3.3mf` paint per triangle (stepped, ~10 % off each engraved band) and one
   hand-placed tail band. Now every ring is the model's band between two grooves (`gregions` 2.5 deg for ears, forehead,
   legs: the user's patches only say which; 0.8 deg for the two tail bands) painted with the app's Cover line
   (`line_width` medians 0.15-0.22 mm); under the tail, where the grooves fade on the bed, each band is closed between
   planes fitted to its two grooves (`groove_planes`), `bottoms()` skips those faces (`Fig.keep_bottom`). Check tool:
   scratchpad `linecheck.py EXPORT OUT [TH] [SIZE] [x,y,z,span] [yaw:pitch;...]` (split paint as written, engraved lines
   in blue): the other 6 follow their lines, except Leafeon's tail stem, whose green ran ~6 mm past the join along the
   foot of the rump (no line there): now a straight soft cut square to the stem at the join. Both re-exported to
   Downloads (old ones in `superseded_2026-11-06_rings_stem/`). Open: a ~0.3 mm jagged notch where each Umbreon tail
   band meets the inside of the curl, on the bed side.
   Open: not re-sliced this pass (user: don't worry about slicing; third pass sliced fine with pads, ~1.5-5 h);
   pockets without pads are walled only 0.5-1.2 mm of the 2 mm magnet on most figures; pupils/highlights not
   modelled on Leafeon/Flareon/Jolteon/Umbreon/Vaporeon (one colour per eye); Vaporeon fins under 0.4 mm in places;
   `band` is now the app's "Cover line" (item 22); straight soft cuts are not a paint mode.
21. **Colours from a picture** (2026-11-06; user: make colouring like the Eevee batch easy in the app; test with
   Ninetales). `web/src/reference.ts` (+ `reference.test.ts`), Colours card fold "Colours from a picture": official art
   by name/number via PokeAPI (species list `pokeapi.co/api/v2/pokemon-species?limit=2000`, art
   `raw.githubusercontent.com/PokeAPI/sprites/.../official-artwork/<id>.png`, both CORS *), name suggested from the
   file/figure name (`guessName`, typos: "Ninetails" -> ninetales); or open / drop on the view / paste a picture. Tap
   picture or chip = selected filament's colour; "Match the palette" = per filament the mode of the picture's pixels
   within 15 ΔE of its closest one (none within 45: kept). Engine: a pokefit export never auto-simplifies (`auto_simplify`).
   Test file: user's `ninetales-50mbpart2.3mf` (231,650 triangles, painted by its designer in 5 filaments; copy in the
   session scratchpad `nt/`, scripts `nt_ui.py` (local UI, Playwright + Chrome), `nt_drop.py` (drop at 390 px),
   `nt_export.py OUT [base]`). Match gave black #010101, tips #E3A94D, body #F1E5A2, eyes #C26955; the 5th filament
   (#898989, the designer's plate, unused on the figure) also moved to #A2927B. Exports in `~/Downloads/ninetales_pokefit/`:
   no base/fill = 21.1 mm, **0 pockets** (paws and tail curls too thin, raised seat too); with the default base
   (`with_base_2_pockets/`) = 19.9 mm, 2 pockets, a 199 mm² filled block under the tails. User to choose.
   Open: (a) match also recolours filaments unused on the figure (no usage per filament in the report);
   (b) a file chosen within ~1 s of page load, before `/api/config` answers, loads in the browser engine while later
   calls go to the local server ("Unknown session"; api.ts `mode` set after config) — seen only from automation;
   (c) live since f3cf1ec, checked on pokefit.party with `nt_ui.py live https://pokefit.party/` (same palette as
   locally; console: only the usual local-engine probes).
22. **Cover line** (2026-11-06; user: "add the engraved lines feat"). `PaintOp.line` (wand/part/region/fill),
   UI "Cover line" button + L key on the last step (`colorize.ts lineLast`). Engine: `Topology.line_depth` (relief over
   the surface smoothed ~LINE_REACH edges, grey closings at LINE_RINGS 2/4/8/16 rings), `Topology.line_width` (depth
   profile vs straight distance from the outline; smallest scale whose peak is >= 0.75 of the next; none -> smallest
   (domes); per 6-edge stretch of outline, Gaussian-blended) -> `soft(widen=)`. Also fixed `Topology.outline`: the
   outside normal was flipped on any fold over 90 degrees (a groove's rim), which put the surface beyond the rim inside.
   Checked (scratchpad `line_diag.py NAME EYE MODE`, renders in `linechk/`): Leafeon eyes widths 0.08-0.38 mm (median
   0.18; the hand-tuned band was 0.15), Espeon sunk eye 0.07-0.25 (reaches its wall), synthetic ring 0.89 (to the far
   rim, < 0.05 mm past it on average); 0.13 s a tap warm on 300k. UI: `nt/line_ui.py` (Ninetales, button + key, no errors).
   Open: (a) a step (block on a slab, Part mode) gets a ~0.6 mm band at its foot (TODO in `line_depth`; capping by
   high ground outside the region did not help); (b) Leafeon runs ~1 edge onto the face below the eye; Espeon's narrow
   tip stops ~1 edge short (Grow/Shrink fix both); (c) live since 706b97b: `nt/line_ui.py OUT https://pokefit.party/`
   passes in the browser engine (live `pokefit.zip` has `line_width`; console: only the usual local-engine probes).
23. **Opening an export** (2026-11-07; user: loading an Eevee evolution takes forever). Browser engine, Vaporeon export:
   38 s -> 14 s. It was analysed three times: with the app's defaults (incl. gap fill and pads, 6 s), again with the
   export's own base settings, again with its old pockets placed. `pipeline.reopened` applies both before the first
   analysis (UI takes the pockets from the report: `main.ts load`); old pockets are placed at the size it fits at
   (`max_scale`, as manual pockets are analysed; the UI used the auto-placement's scale). Fixed on the way:
   `Session.old_pockets` was empty until something else read the parts. Calls now: load 9.2 s, faces 0.6, full analysis
   5.9 (scratchpad `nt/str_ui.py OUT FILE [URL]`: load + strengthen timings per worker call, live or local).
24. **Weak points, per-feature thickening, reinforcement, Changes to your model** (2026-11-09; user: Leafeon's ears and
   tail look brittle; a tool grading thin connections; thicken tail + ears; more walls/infill inside; flag settings
   that reshape the model). `pokefit/joints.py` (module doc), `Params.strengthen_at` / `reinforce`, UI `web/src/weak.ts`,
   `web/src/changes.ts`. Measured: rod 1.5 x 10 mm 1.34 N vs 1.33 N beam formula (tests/test_joints.py). Leafeon
   at 34.3 mm (grader after the projected-area fix): tail 1.5 N (D), ears 2.0-2.4 N (D), ear tips F. Reinforcement:
   Bambu honours the modifiers (10 % box: -0.4 g), but the X2D 0.20 Standard process is already 100 % infill.
   **Stretch** (2026-11-09, second pass): single features are stretched across their thin directions
   (`strengthen._stretch`, module doc), replacing the shell union for `Local`: veins, outline and the ears' brown
   panels stay. Leafeon tail 4.5 / ears 3.0: ears 6.3-6.4 N (C); the tail is no longer a thin feature at the 3 mm
   probe; at a 6 mm probe its weakest is 2.6 N at the stalk's join to the body (behind the weak section, not stretched).
   Rods test: 1.3 N -> >= 3.6 N along the whole rod, triangles unchanged. Browser engine, 300k Leafeon: grade 15-18 s,
   thicken the tail 39 s (local UI ~12 s / ~30 s). Files: Downloads `leafeon_pokefit_60_stretched_tail4.5_ears3*.3mf`,
   `review_leafeon_stretched_{tail,ears}.png` (rows: current / old shell union / stretch); export in scratchpad
   `batch/export_stretch/leafeon`, script `stretch.py` (prototype) and `stretch_grade.py` (per-feature grades, PROBE env).
   Open: (a) slight lumps on the lower tail stalk and fine creases where the ears meet the head (the ease onto the body);
   (b) the forehead leaf is still D (2.9 N): thicken it the same way; (c) the stalk's join behind the weak section
   (probe 6 shows it): `REINFORCE_BACK`-like reach back into the join for the stretch?; (d) the user picks a Leafeon
   variant (current / stretched).
25. **beambam / phone printing**: audit (76 findings, 7 P0) + design + phased task list in
   `C:\git\beambam\docs\plan-2026-11\` (TASKS.md first). Not committed there: security findings in a public repo, a
   personal git identity in that clone, and someone's uncommitted edits (`beambam/ams.py`, `doctor.py`, 3 tests).
   Next: Phase 0 there (S6 runner, port 8780, one AMS parser), then pokefit task 0.4 (`printer.py` reads it).
20. User's call: Cloudflare "Always Use HTTPS" (dashboard → pokefit.party → SSL/TLS → Edge Certificates). Off now: an
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
- CLI: `python -m pokefit fit <figure.3mf> [--ball 0.6 | --diameter 44] --steel D20 [--floor auto] [--scale 85%] [--magnets 0-3]
  [--rotate-x/-y/-z DEG]
  [--steels] [--out DIR] [--slice] [--x10]`; `python -m pokefit ball` prints the cavity profile.
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
