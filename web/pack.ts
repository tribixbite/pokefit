/** Zip the `pokefit` Python package for the browser engine (used by build.ts and the smoke test). */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { zipSync } from "fflate";

/** Python sources, bundled JSON profiles and WebAssembly (the mesh simplifier) under `<repo>/pokefit`, as a zip. */
export function pokefitZip(repoRoot: string): Uint8Array {
  const pkg = join(repoRoot, "pokefit");
  const files: Record<string, Uint8Array> = {};
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      if (statSync(p).isDirectory()) {
        if (name !== "__pycache__") walk(p);
      } else if (name.endsWith(".py") || name.endsWith(".json") || name.endsWith(".wasm")) {
        files[relative(repoRoot, p).replace(/\\/g, "/")] = readFileSync(p);
      }
    }
  };
  walk(pkg);
  return zipSync(files, { level: 9 });
}
