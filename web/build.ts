/**
 * Build the static site into dist/: the UI bundle, the engine worker and everything
 * the in-browser engine loads (manifold.wasm, pokefit.zip, vendored wheels,
 * config.json), plus the static files in public/ (SEO, social preview, icons).
 */
import { createHash } from "node:crypto";
import { copyFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PY_WHEELS, wheelName } from "./src/engine/boot";
import { pokefitZip } from "./pack";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dist = join(here, "dist");
mkdirSync(dist, { recursive: true });

/** Every file under `dir`, sorted (for a stable hash). */
function files(dir: string): string[] {
  return readdirSync(dir).sort().flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : [p];
  });
}

// Build stamp: a hash of everything the bundles and the engine are made from. Asset URLs carry it
// (?v=), so a deploy is never run as new HTML against scripts the CDN told browsers to keep for hours.
const engineZip = pokefitZip(root);
const stamp = createHash("sha256");
for (const f of [...files(join(here, "src")), join(here, "style.css"), join(here, "bun.lock")]) stamp.update(readFileSync(f));
stamp.update(engineZip);
const build = stamp.digest("hex").slice(0, 12);

const result = await Bun.build({
  entrypoints: [join(here, "src", "main.ts"), join(here, "src", "worker.ts")],
  outdir: dist,
  target: "browser",
  minify: true,
  sourcemap: "linked",
  naming: "[name].js",
  define: { __BUILD__: JSON.stringify(build) },
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

copyFileSync(join(here, "style.css"), join(dist, "style.css"));
const html = readFileSync(join(here, "index.html"), "utf8");
const stamped = html.replace('src="main.js"', `src="main.js?v=${build}"`).replace('href="style.css"', `href="style.css?v=${build}"`);
if (stamped === html || !stamped.includes(`main.js?v=${build}`) || !stamped.includes(`style.css?v=${build}`)) {
  throw new Error("index.html: main.js / style.css references not found for the build stamp");
}
writeFileSync(join(dist, "index.html"), stamped);
if (existsSync(join(here, "public"))) cpSync(join(here, "public"), dist, { recursive: true });
copyFileSync(join(here, "node_modules", "manifold-3d", "manifold.wasm"), join(dist, "manifold.wasm"));
writeFileSync(join(dist, "pokefit.zip"), engineZip);

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
console.log(`built ${result.outputs.length} bundles + engine files, ${(size / 1024).toFixed(0)} KB, build ${build} -> ${dist}`);
