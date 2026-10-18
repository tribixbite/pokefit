# pokefit — fit a figure inside a scaled Pokéball, add magnet pockets, verify printability

Date: 2026-10-16 · Status: implemented 2026-10-18 (see "Implementation notes" at the end)

## Goal

Given a 3MF (or STL/OBJ) of a Pokémon figure, quickly:

1. scale it to the largest size (or a chosen size) that fits inside one of the user's scaled
   Pokéballs (`pokeball.3mf`, "functional Pokéball – no glue"),
2. cut two push-in pockets in its feet for 4 × 2 mm disc magnets that grip a steel fender
   washer glued to the ball's inner floor,
3. verify mesh soundness and printability, including whether supports are needed and the print
   time (PETG, Bambu printer), and
4. write a print-ready 3MF that keeps the original parts, filament assignments and paint.

Success: for a typical figure the user gets a fit verdict, preview and output 3MF in under a
minute (Bambu slices run on demand and may take longer), and the printed figure fits the
assembled ball and stays put on the washer.

## Fixed facts (measured, not assumed)

| Item | Value | Source |
|---|---|---|
| Ball shell inner sphere | R 36.806 mm (native), outer R 38.740 | sphere fit of `0_ball_shell_2x` |
| Connector rings | 2 rings meeting at the equator, each lining its shell for 16.7 mm toward the pole; inner radius ≈ 30.1–32.5 (min 28.76 at the latch) | `1_inner_black` radial profile |
| User's ball scales | 0.6 (`pokeball60.3mf`), 1.0 | build transforms |
| Magnet pocket | Ø 4.18 mm (48-gon), 2.20 mm deep, open-ended blind hole → 4 × 2 mm magnets | all six Zelda pot STLs |
| Default washer | DIN 9021 M6: OD 18, ID 6.4, 1.6 mm thick | recommendation (user has not bought one) |

## Geometry

### Cavity model (`pokefit/ball.py`)
- Derived from `pokeball.3mf` at runtime (cached as JSON keyed by file hash): shell inner
  sphere radius, ring inner radius profile `r_ring(h)` (h = distance from equator), ring span.
- Ring profile uses the **minimum over all angles** at each height (conservative; the latch
  defines it near the button).
- Allowed radius at height z (z = 0 at ball centre): `min(sqrt(Ri² − z²), r_ring(|z|))` inside
  the ring span, `sqrt(Ri² − z²)` outside it. Scaled by ball scale, then reduced by the
  clearance margin (default 1.0 mm, applies radially and at the top).
- Floor = top face of the washer. The washer seat is the shell's inner flat made by
  `flat_spot` with fill diameter = washer OD, so the floor height is
  `−Ri·s + seat_depth(OD) + washer_thickness` (≈ 3.5 mm above the original inner bottom for
  M6 at 0.6).

### Fit (`pokefit/fit.py`)
- Figure is z-up (as in its 3MF); UI offers 90° rotations about X/Y and free yaw.
- Figure's lowest point sits on the floor. Feasibility: every vertex within the allowed radius
  at its height and below the top limit, for a given figure scale and XY offset.
- Max scale by bisection (tolerance 0.1 %). Report the limiting vertex/region
  (e.g. "ear tip touches ring at 87 %") — highlighted in the preview.
- User may choose any scale ≤ max; the tool always re-verifies the chosen scale.

### Magnet pockets (`pokefit/magnets.py`)
- Auto-detect feet: section at `bottom + 0.3 mm`; the two largest islands are the feet. Pocket
  centre = pole of inaccessibility of each island (`shapely.ops.polylabel`). One island only →
  two pockets placed along its major axis, as far apart as wall rules allow.
- Pocket: cylinder Ø 4.18 × 2.20 mm (configurable), open at the bottom, cut with manifold3d
  from the part(s) it intersects. Dimensions are in final (printed) mm.
- Validity: pocket disk buffered by the wall (default 0.8 mm) must lie inside the figure's
  cross-section at the bottom, mid-depth and top of the pocket; ≥ 0.6 mm material above the
  pocket ceiling. Failure is reported, never silently cut through.
- Steel coverage: both pocket centres must lie on the washer annulus
  `ID/2 + magnet_r ≤ r ≤ OD/2 − magnet_r`. The solver shifts the figure in XY to satisfy this
  while keeping the fit; if impossible (feet spacing > annulus chord), report the required
  washer size. In the UI the user can drag pockets.

## Checks (`pokefit/checks.py`, `pokefit/slicer.py`)

Each check returns `{id, status: pass|warn|fail, value, message, highlight?}`.

Mesh soundness (input and output, per part):
- watertight (every edge used by exactly 2 faces), manifold3d accepts, consistent winding,
  no degenerate faces, stray components (count + volume), self-intersections (pymeshlab).
- Optional auto-repair per part (pymeshfix, as in `scripts/repair_shell.py`) with surface
  deviation report (median / p99 / max mm).

Printability at the chosen scale:
- Wall thickness: sampled inward ray thickness; warn < 0.8 mm, fail < 0.4 mm; heatmap
  highlight.
- Overhangs: area of down-facing faces steeper than 45° that are not on the bed; islands that
  start above the bed (layer-sliced at 0.2 mm). Gives the **supports needed: yes/no** verdict.
- Bed contact area after pockets; warn below a threshold and suggest a brim (PETG).
- Pocket walls and steel coverage (above).

Bambu Studio slice (`C:\Program Files\Bambu Studio\bambu-studio.exe`, headless CLI):
- Slice the output 3MF twice: supports off and auto supports on. Report errors/warnings
  (e.g. floating regions), print time and filament for each → supports verdict confirmed and
  the time cost of supports.
- 10× estimate (best-effort): one plate with 10 instances, auto-arranged. If the CLI cannot do
  this reliably it is reported as unavailable, not estimated by multiplication.
- CLI flag behaviour must be verified on this install first; an unavailable slicer is a
  reported `warn`, never a silent skip.

## Output (`pokefit/model3mf.py`)
- `<name>_pokefit_<ballpct>.3mf`: a copy of the input 3MF with scale baked into vertices,
  pockets cut, everything else (objects, parts, part names, extruder assignment, plate
  settings, metadata) kept. Per-triangle `paint_color` is preserved for unchanged triangles;
  triangles in modified parts get the paint of the nearest original triangle.
- Optional per-part STLs.
- Matching bottom shell `shell_washerseat_<ballpct>_<washer>.stl` via the shared flat-spot
  code (8 mm outer disk, inner flat = washer OD).
- STL/OBJ input → plain 3MF output.

## Architecture

```
pokefit/
  model3mf.py   read/write 3MF (objects, components, build items, paint, Bambu config)
  ball.py       cavity model from pokeball.3mf (+ JSON cache)
  shell.py      sphere fitting + flat spot / washer seat (moved from scripts/flat_spot.py)
  fit.py        feasibility + max-scale bisection + XY offset
  magnets.py    foot detection, pocket placement/validation, cutting
  checks.py     mesh + printability checks
  slicer.py     Bambu Studio CLI wrapper
  cli.py        `python -m pokefit fit|serve|ball`
  server.py     stdlib ThreadingHTTPServer: JSON API + static web/dist
web/            TypeScript + three.js (bun build → web/dist)
tests/          pytest
```
- `scripts/flat_spot.py` becomes a thin wrapper over `pokefit.shell` (no duplicate code).
- Server API: `POST /api/load` (upload → session id, parts, auto-fit result),
  `POST /api/fit` (scale, rotation, offset, pocket positions → checks + preview meshes),
  `POST /api/slice` (Bambu slice, long-running), `GET /api/export` (3MF download).
  Preview meshes sent as binary (GLB) for three.js.
- Web UI (dark, mobile-friendly, touch): 3D view with translucent ball/rings/washer and the
  figure; scale slider showing max-fit marker; pocket markers draggable on a bottom view;
  checklist panel; slice button with 1×/10× time and supports verdict; export button.

## Error handling
- Unreadable/unsupported file, figure that cannot fit at any scale ≥ 5 %, feet too small for
  pockets, washer too small for feet spacing, slicer missing/failing: each is a clear check
  result with the measured numbers, not an exception trace.
- Long operations (slice, thickness) report progress; the UI stays responsive.

## Testing
- pytest on synthetic meshes with known answers: two-footed box (pocket positions, walls),
  sphere bigger than the cavity (max scale analytic), cube with a removed face (watertight
  fail), two overlapping parts, a part with paint attributes (paint preserved round-trip).
- Cavity model: allowed radius at known heights vs hand-computed values from the fits above.
- Real files: `dragonwithegg+eyes+multicolor.3mf` (multicolour figure), Zelda pot parts.
- Round-trip: output 3MF re-opened, sliced by Bambu CLI.
- Web UI verified with screenshots at desktop and phone widths.

## Out of scope
- Orientation optimisation for minimum supports, hollowing, auto-splitting a figure,
  modifying shells beyond the washer seat.

## Implementation notes (2026-10-18)
Decisions made while building, where the code differs from or adds to the text above:
- Output 3MF holds only the chosen figure object (one plate), not the other objects of the
  source project (e.g. the egg halves); settings default to the user's printer profile
  (`settings_source="reference"`), the source project's settings with `"original"`.
- Washer seat diameter = washer OD + 0.4 mm so the washer drops in.
- Beyond the ring span the allowed radius is also capped by the ring aperture (the figure
  must pass through the ring when the ball closes).
- Pockets are placed inside (walled foot area ∩ washer steel ring) at the fit offset or any
  offset within the fit margin; the scale only steps down (to ≥ 60 % of the fit size) when
  no such pair exists; then pocket centres on steel (magnets about half over it) is a warning.
- Wall-thickness `fail` only when > 5 % of the surface is under 0.4 mm (Arachne prints
  isolated thin features); otherwise `warn`.
- Supports verdict: geometric estimate, replaced by Bambu's auto-support result after slicing.
- Overlapping parts are clipped in list order for the preview (as the slicers do).
