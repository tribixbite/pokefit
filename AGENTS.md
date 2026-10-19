# AGENTS.md — handoff guide for coding agents

Read this first, then `NOTES.md` (the working tracker: open items, measurements, decisions).
`README.md` is the user-facing description; `docs/superpowers/` holds the original spec and plan.

## What this is

**pokefit** (live at https://pokefit.party) fits any 3D-printable figure (3MF / STL / OBJ / GLB) inside a
printed Pokéball (MakerWorld "functional Pokéball - no glue"), scaled to the largest size that fits; cuts
1–3 pockets for 4 × 2 mm disc magnets so it snaps onto a steel washer glued in the ball; colours
single-colour models (tap-to-paint) and turns a GLB's texture into filament paint; checks printability;
exports Bambu Studio 3MFs and (locally) slices them with the Bambu Studio CLI.

One Python engine (`pokefit/`) runs two ways:
- **Locally**: `python -m pokefit serve` (HTTP server + the web UI), or the CLI `python -m pokefit fit …`.
- **In the browser**: the same package zipped into Pyodide in a Web Worker (`web/src/worker.ts`,
  `web/src/engine/boot.ts`); manifold-3d WASM does the booleans (`pokefit/geom.py` → `_manifold_js.py`).
  `web/src/api.ts` uses the local server when the page is served by it, else the worker. No slicing on
  the site.

## Hard rules (the user's, non-negotiable)

- **The repo is PUBLIC.** Commit as `tribixbite <381345+tribixbite@users.noreply.github.com>`. On a new
  clone set it locally first:
  `git config user.name tribixbite && git config user.email 381345+tribixbite@users.noreply.github.com`.
  No personal emails, names, home paths or private data in commits.
- **Never commit model files**: `*.3mf *.stl *.skp *.skb *.dae`, `original/`, `printed/` (third-party
  designs and personal models; all git-ignored). The ball ships only as measurements
  (`pokefit/profiles/ball_functional_pokeball.json`).
- **Every push to `main` redeploys the live site** (no tags/releases). Push only when the user asks or the
  round's work is done and verified; then watch CI: `gh run list -L 2` and `gh run watch <id> --exit-status`.
- Never `git reset/revert/rebase/force-push` or rewrite history without explicit permission. Never run a
  recursive delete (`rm -rf`, `Remove-Item -Recurse`); use a new directory name instead. Never create
  `foo_fixed.py`-style copies; edit the existing file.
- Never post on GitHub/anywhere (issues, comments) without per-instance approval.
- Conventional commits, signed with an em dash + model name (e.g. `— Opus 5.5`), no "Claude" mention,
  no co-author trailer. After each round: commit, and update `NOTES.md` (prune finished items — history
  lives in `git log`).
- Use **bun** (never npm/npx) for the web app; TypeScript over JS. Dark, mobile-friendly UI.
- Run the gates before committing (below). Report compromises and anything unverified honestly.
- Screenshots shared with the user: under 2000 px per side and under 4 MB.

## Setup on a fresh clone

```bash
pip install -e ".[dev]"            # Python ≥ 3.11; add ".[full]" for pymeshlab self-intersection checks
cd web && bun install && bun run build && cd ..
python -m pokefit serve --open     # http://127.0.0.1:8765/
```
Optional, for slicing: Bambu Studio installed (Windows default path, or set `POKEFIT_BAMBU` to the
executable). Without it, slice features report "not found" and the slicer tests skip.

Files NOT in the repo (tests needing them skip; get them from the user if needed):
- `pokeball.3mf` (repo root): the ball model, MakerWorld #2045372. Needed to export the washer-seat shell
  STL and for `tests/test_ball.py`.
- `pokeball60goodPrinted.3mf` (repo root): the user's real Bambu project (X2D, 0.4 nozzle, PETG, 0.16 mm,
  process "printed_ball_shells"). Without it the bundled copy `pokefit/profiles/x2d_petg_printed_ball_shells.json`
  is used. Don't use a figure file's own printer settings for time estimates (MakerWorld files carry
  someone else's printer).
- Real test figures from the original machine's `~/Downloads`: `dragonwithegg+eyes+multicolor.3mf`
  (painted multi-colour 3MF), `eevee+v3 (1).3mf` and `eevee+v3_stls.zip` (floating tail, colorize),
  a Manex3D AI Squirtle GLB (colour in the *emissive* texture), LEGO GLBs (`41714-1-dbix_conv_v3.glb`,
  600k triangles), `DamagedHelmet.glb` (non-manifold).

## Gates (run all before committing; CI runs the same in `.github/workflows/deploy.yml`)

```bash
python -m mypy                     # strict-ish, must be clean
python -m pytest -q                # ~135 tests, ~100 s; real-file/Bambu tests skip without their files
cd web && bun run typecheck && bun test && bun run build
node scripts/engine-smoke.ts       # the whole engine under Pyodide in Node 24 (~25 s): load, analyse,
                                   # magnets, colorize, export, GLB flat + textured colour
```
End-to-end checks worth doing for engine changes (see `.claude/skills/pokefit-verify`):
open a real file in the local UI, export, and `python scripts/verify_export.py <export.3mf>` (paint
summary + Bambu slice: time, grams, colour changes, purge).

## Architecture map

| Python (`pokefit/`) | Role |
|---|---|
| `pipeline.py` | `Params` (every UI setting), `Session` (per-file caches: orientation frames, height fields, topology, paint), `analyse`, `export`, `preview_glb`, `pick`. Start here. |
| `model3mf.py` | 3MF read (lazy per object, regex fast path) / write (Bambu flavour, settings carried); STL/OBJ; `remesh_part` (carries per-triangle attributes after a boolean, incl. split paint via `paint.remap_tree`). |
| `conform.py` | Shape for the ball: bend limbs lying out past the washer up along the sphere, press/fill a flat base over the washer. Vertex moves only (paint survives) + a separate "flat base" part. |
| `presets.py` | Bambu preset exports (.bbscfg / .bbsflmt / preset zips): read, resolve `inherits` against installed system profiles, merge into the reference settings; plate-compatibility check. |
| `toolchange.py` | Colour-change cost: per-layer filament use from paint, per-nozzle simulation, best X2D nozzle split, cost per palette size (report `toolchange`). Calibrated on Bambu slices. |
| `printer.py` | The printer's loaded AMS filaments via beambam's CLI (subprocess: beambam is AGPL, never imported); colours matched with `remap.match_slots`. |
| `imageto3d.py` | Photos (front/left/back[/right]) -> coloured GLB via a local ComfyUI running Pixal3D (`python -m pokefit photos`, UI "Make one from photos", `POST /api/photos`). |
| `gltf.py` | GLB/glTF → one part with colour as split Bambu paint; `TextureColours` re-cuts the palette (`Params.texture_colours`). Module doc explains the algorithm. |
| `paint.py` | Bambu `paint_color` (TriangleSelector) format: decode, remap states, split trees (`parse_tree`/`encode_tree`/`split_children`/`leaves`/`remap_tree`). |
| `colorize.py` | Tap-to-paint: topology (convex/concave edges), regions by mode (part/region/piece/brush/face), `apply_paint`. |
| `fit.py`, `ball.py`, `shell.py` | Cavity model of the ball, max-scale fit (Nelder–Mead with restarts), washer-seat shell. |
| `magnets.py` | Height field of the underside, pocket maps, auto/manual placement, seat modes `washer` (default) / `raised`. |
| `checks.py`, `planar.py` | Printability checks (watertight, thickness, overhangs, islands, bed contact, supports verdict); section regions without rtree. |
| `slicer.py`, `remap.py`, `washer.py` | Bambu CLI slicing + settings adaptation; import-file slot matching; washer catalogue. |
| `server.py`, `webapi.py`, `cli.py` | Local HTTP API; the Pyodide-facing API (same functions); CLI. |

| Web (`web/`) | Role |
|---|---|
| `src/main.ts` | UI state (`state.params` mirrors `Params`), cards, analyse/export flow. |
| `src/scene.ts` | three.js viewer, ball/figure/pockets, tap picking. |
| `src/colorize.ts` | Colours card: palette, paint modes, GLB "Filaments" slider. |
| `src/api.ts`, `src/worker.ts`, `src/engine/boot.ts` | Local-server vs. Pyodide worker transport. |
| `build.ts`, `pack.ts`, `src/version.ts` | Bundles, `pokefit.zip`, build-hash `?v=` stamps on every asset URL. |
| `src/handoff.ts` | Models from other apps (craftmatic): `pokefit:load` postMessage from trusted origins, `?url=`. |
| `public/` | Static files: CNAME, og.gif, icons, `llms.txt` (keep it in sync with features). |

## Gotchas that cost real time (keep them in mind)

- **Bambu `paint_color` split trees**: hex nibbles read from the END; children are serialised
  last-first; a 3-side split's children are (v0,m01,m20),(m01,v1,m12),(m12,v2,m20),centre, each with a
  different corner order — corner rotations/mirrors are remapped geometrically (`paint.remap_tree`), never
  by a fixed rule. Verified against BambuStudio `src/libslic3r/TriangleSelector.cpp`.
- **Bambu CLI nozzle map** (X2D/H2D): the CLI takes `filament_map_mode` / `filament_maps` from the PLATE's
  metadata in `Metadata/model_settings.config`, not from `project_settings.config` (a "Manual" map there
  is silently replaced by its auto choice).
- **Bambu CLI**: a 3MF must carry `<metadata name="Application">BambuStudio-…` or embedded settings are
  ignored; it writes `result.json` into its cwd; it sometimes hangs after writing it (handled: poll + kill).
- **Cloudflare** in front of GitHub Pages keeps .js/.wasm/.zip for 4 h while HTML is 600 s: every asset
  URL is stamped `?v=<build hash>` (build.ts). Keep new assets stamped (`versioned()` in `src/version.ts`).
- **Pyodide**: `np.intp` is int32 (cast repeat counts); no rtree/embree/manifold3d/pymeshlab; trimesh and
  networkx are vendored wheels installed with `deps=False`. Pillow and scipy are available.
- **Windows dev box quirks**: Python `ThreadingHTTPServer` can double-bind a port on Windows (server sets
  `allow_reuse_address` off there); `Path.read_text()` defaults to cp1252 — pass `encoding="utf-8"`;
  heredocs with apostrophes/backslashes break in the Bash tool — write scripts to files. The repo is LF
  (`.gitattributes`).
- **Dark Reader** in the user's Chrome re-tints pages; `index.html` has `<meta name="darkreader-lock">`.
- Real figures are messy (non-manifold, floating parts). Checks report, they don't block; `planar._xor`
  repairs section loops GEOS rejects.

## Photos to 3D (local only, needs an NVIDIA GPU with 16 GB)

`pokefit/imageto3d.py` drives a **ComfyUI** server running **Pixal3D multi-view** (Tencent ARC, MIT, Sep 2026;
ComfyUI's native nodes in `comfy_extras/nodes_trellis2.py`, pure PyTorch, so no custom CUDA builds on Windows /
Blackwell). Chosen 2026-10-19 over Hunyuan3D-2mv (non-commercial in EU/UK/KR, custom rasteriser builds) and
TRELLIS.2 (24 GB, Linux). A web page cannot run it (~9 GB of weights, sparse 3D convolutions).

Setup (verified on the original machine: RTX 5080 Laptop 16 GB, driver CUDA 13.3, Python 3.11):
```bash
git clone --depth 1 --branch v0.42.1 https://github.com/Comfy-Org/ComfyUI.git ../comfyui-3d
cd ../comfyui-3d && py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
# weights (8.8 GB + 0.44 GB), from https://huggingface.co/Comfy-Org/Pixal3D and .../Comfy-Org/BiRefNet:
#   models/diffusion_models/pixal3d_multiview_int8_convrot.safetensors
#   models/clip_vision/dino_v3_L_naf_fp32.safetensors
#   models/vae/trellis_2_shape_vae_bf16.safetensors, models/vae/trellis_2_texture_vae_bf16.safetensors
#   models/background_removal/birefnet.safetensors
.venv/Scripts/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch
```
Then `python -m pokefit photos --front f.png --left l.png --back b.png --out fig.glb` (or the UI button, which
appears when the local server reaches ComfyUI at `POKEFIT_COMFY`, default http://127.0.0.1:8188).
- Views: the figure's own left (camera at +X when it faces -Y = it faces the image's left). Photos are framed
  by pokefit, not ComfyUI: one scale for all views (same height), largest extent at 1024/1.1 px, centred, on
  black (`frame_views`). `--mirror-right` adds the mirrored left photo as the right view.
- RemeshMesh must use `sign_mode: "solid"`: the template's `udf` gives a double-walled skin (92 % of the surface
  "under 0.4 mm thick", not manifold). Colour comes back as `COLOR_0` (linear) under a white PBR material.
- Measured: ~3.4 min for 3 views (int8 weights, 200k faces) while another app held 12 GB of the VRAM
  (ComfyUI streams weights). Squirtle photos -> watertight solid, 8 colours, slices in Bambu.

## Ecosystem: beambam and craftmatic

Plan and boundaries: `docs/integration.md`. beambam (printer control, AGPL) is used only as a subprocess
(`POKEFIT_BEAMBAM`, default `beambam` on PATH; a venv is at `C:/git/beambam/.venv` on the original machine).
craftmatic (TS) hands models over as files (`web/src/handoff.ts`).

## Testing the UI with a browser agent

Serve locally (`python -m pokefit serve`), open http://127.0.0.1:8765/, upload with the browser tool's
file-upload on the `Open figure` input (don't click it — it opens a native dialog). The "Cut" view hides
the ball shell for checking colours. Phone layout: emulate ~390 × 844 (a same-origin iframe works when
the window can't be resized). Avoid browser alert dialogs. Check the console for errors after each flow.

## Where things stand

See `NOTES.md` → "To do" for prioritised open work and "Open" for known limits and measurements.
