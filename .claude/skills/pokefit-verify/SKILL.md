---
name: pokefit-verify
description: Verify a pokefit change end to end — gates, the in-browser engine smoke test, a real figure in the local UI, and a Bambu Studio slice of the export. Use after any engine (pokefit/*.py) or web UI change, before committing.
---

# Verify a pokefit change

Work from the repo root. Stop and report at the first failure, with its output.

## 1. Gates (always)

```bash
# In the CI-mirroring venv (.venv, see AGENTS.md "Gates"): CI installs the latest numpy/Pillow.
.venv/Scripts/python -m mypy                     # must print "Success"
.venv/Scripts/python -m pytest -q                # all pass; note how many skipped (missing private files)
cd web && bun run typecheck && bun test && bun run build && node scripts/engine-smoke.ts && cd ..
```
`engine-smoke.ts` runs the engine in Pyodide under Node 24 and must end with "engine smoke test passed".
It is the only check of the browser engine: a feature that works locally can still break there
(int32 `np.intp`, missing packages). When you add an engine feature, add a line to the smoke test.

Make sure a new test can fail: break the code it covers once (e.g. monkeypatch the function to a no-op)
and see it fail — a guarded fast path or a lenient assertion can pass on broken code.

## 2. A real figure in the local UI (engine or UI changes)

1. Restart the server on the new code (an old one keeps serving stale code; on Windows check no second
   `python -m pokefit serve` is bound to the port):
   `python -m pokefit serve --port 8765` (run in the background).
2. In the browser agent: get tab context, open a new tab at http://127.0.0.1:8765/, find the
   "Open figure" file input and upload a figure with the file-upload tool (never click the input).
3. Screenshot; switch to "Cut" view to see colours without the shell; zoom on details.
4. Exercise the changed controls; read the console filtered to errors; close the tab when done.
Real files are not in the repo (see AGENTS.md → "Files NOT in the repo"); use what the user provides,
or synthesise one with trimesh (the tests show how: boxes, textured icospheres, GLB scenes).

## 3. Export and slice (paint, colour, settings or geometry-cut changes)

```bash
python -m pokefit fit <figure> [--colours N] --out pokefit_out/check     # or Export in the UI
python scripts/verify_export.py pokefit_out/check/<name>_pokefit_60.3mf --out <new folder>
```
Read: parts and extruders, paint states (and split triangles), `filament_colour` length = palette size,
`slice ok`, colour changes and purge vs. `main g` (the figure's own filament). Needs Bambu Studio
(`POKEFIT_BAMBU` or the default Windows path). Use a NEW `--out` folder each run.

## 4. Report

State what ran and what it showed (numbers), what was skipped and why, and any compromise. Update
`NOTES.md` with measurements a later decision needs.
