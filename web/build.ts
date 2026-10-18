/**
 * Build the static site into dist/: the UI bundle, the engine worker and everything
 * the in-browser engine loads (manifold.wasm, pokefit.zip, vendored wheels,
 * config.json), plus the static files in public/ (SEO, social preview, icons).
 */
import { copyFileSync, cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PY_WHEELS, wheelName } from "./src/engine/boot";
import { pokefitZip } from "./pack";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dist = join(here, "dist");
mkdirSync(dist, { recursive: true });

const result = await Bun.build({
  entrypoints: [join(here, "src", "main.ts"), join(here, "src", "worker.ts")],
  outdir: dist,
  target: "browser",
  minify: true,
  sourcemap: "linked",
  naming: "[name].js",
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

for (const f of ["index.html", "style.css"]) copyFileSync(join(here, f), join(dist, f));
if (existsSync(join(here, "public"))) cpSync(join(here, "public"), dist, { recursive: true });
copyFileSync(join(here, "node_modules", "manifold-3d", "manifold.wasm"), join(dist, "manifold.wasm"));
writeFileSync(join(dist, "pokefit.zip"), pokefitZip(root));

// Wheels: cached in .wheel-cache so offline rebuilds work after the first.
const cache = join(here, ".wheel-cache");
mkdirSync(cache, { recursive: true });
mkdirSync(join(dist, "wheels"), { recursive: true });
for (const url of PY_WHEELS) {
  const name = wheelName(url);
  const cached = join(cache, name);
  if (!existsSync(cached)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    writeFileSync(cached, new Uint8Array(await res.arrayBuffer()));
  }
  copyFileSync(cached, join(dist, "wheels", name));
}

// The engine's start-up config, so the static page is usable before the engine has loaded.
const py = Bun.spawnSync(["python", "-c",
  "import json; from pokefit.pipeline import app_config; print(json.dumps({**app_config(None), 'engine': 'browser'}))"],
  { cwd: root, stderr: "inherit" });
if (py.exitCode !== 0) throw new Error("python could not produce config.json (is pokefit installed? pip install -e .)");
writeFileSync(join(dist, "config.json"), py.stdout);

const size = result.outputs.reduce((n, o) => n + o.size, 0) + readFileSync(join(dist, "pokefit.zip")).length;
console.log(`built ${result.outputs.length} bundles + engine files, ${(size / 1024).toFixed(0)} KB -> ${dist}`);
