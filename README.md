# pokefit

**https://pokefit.party** — fit any 3D-printable figure inside a printed Pokéball, with magnets.

![pokefit preview](web/public/og.jpg)

pokefit takes a figure (3MF, STL or OBJ) and:

- scales it to the **largest size that fits** the ball's inner sphere, connector rings and ring opening
  (60 % or 100 % ball, or any inner diameter);
- cuts **1–3 magnet pockets** (4 × 2 mm discs) into its underside — flat feet or not — so it snaps onto a
  **steel washer** glued inside the ball (magnets rest on the washer; optionally raised into floating parts),
  and compares washers;
- **colours single-colour models**: tap the 3D view and each tap fills a part found from the shape (ears, tail,
  eyes, bib…), a separate piece or a brush patch, exported as Bambu paint;
- **checks printability**: closed mesh, wall thickness, overhangs, floating layers, bed contact;
- **exports a Bambu Studio 3MF** that keeps painted colours, plus a variant renumbered to your own project's
  filament slots for *Import geometry only*.

The website runs the whole engine **in your browser** (Python → WebAssembly via Pyodide); files are never
uploaded. The local version adds **Bambu Studio slicing** (print time and filament for 1 and 10 copies, supports).

## Local use

Needs Python 3.11+ and [bun](https://bun.sh); Bambu Studio for slicing.

```sh
pip install -e .            # add ".[full]" for the self-intersection check
cd web && bun install && bun run build && cd ..
python -m pokefit serve --open                       # web UI at http://127.0.0.1:8765
python -m pokefit fit figure.3mf --diameter 44 --magnets 2 --washers --out out/ --slice --x10
```

The ball's measurements ship with the package. The ball model itself (MakerWorld
[#2045372](https://makerworld.com/en/models/2045372), not redistributed here) is only needed for the
washer-seat bottom shell: put `pokeball.3mf` in the repo folder, or load it on the website.

## Development

| | |
|---|---|
| Python engine | `pokefit/` — `pipeline.py` (analysis), `fit.py`, `magnets.py`, `checks.py`, `model3mf.py`, `webapi.py` (browser entry) |
| Web UI | `web/src/` — `main.ts`, `scene.ts` (three.js), `underside.ts`, `api.ts` (local server or worker), `worker.ts` (Pyodide) |
| Gates | `python -m mypy && python -m pytest`, `cd web && bun run typecheck && bun test && bun run build && bun run smoke` |
| Social preview | `cd web && bun run og`, open the printed URL (writes `public/og.gif`, `public/og.jpg`) |

Every push to `main` runs the gates and deploys `web/dist` to GitHub Pages (`.github/workflows/deploy.yml`).

pokefit is a fan-made tool, not affiliated with or endorsed by Nintendo, Game Freak, Creatures or The Pokémon
Company. Pokémon and Poké Ball are trademarks of their respective owners.
