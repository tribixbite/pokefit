/**
 * Run the browser engine (Pyodide + manifold-3d WASM) under Node on a small
 * synthetic figure, the way the Web Worker does. CI runs this before deploying.
 *
 *   node scripts/engine-smoke.ts [figure.3mf|.stl]
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Module from "manifold-3d";
import { loadPyodide } from "pyodide";
import { bootPokefit, PYODIDE_CDN } from "../src/engine/boot.ts";
import { pokefitZip } from "../pack.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const t0 = performance.now();
const secs = (): string => ((performance.now() - t0) / 1000).toFixed(1);

// Node keeps downloaded Pyodide packages in a local cache folder (git-ignored).
const pyodide = await loadPyodide({ packageBaseUrl: PYODIDE_CDN, packageCacheDir: join(here, "..", ".pyodide-cache") });
const manifold = await Module();
manifold.setup();
const zip = pokefitZip(root);
const api = await bootPokefit({ pyodide, manifold, pokefitZip: zip.buffer as ArrayBuffer, onStatus: (s) => console.log(`[${secs()} s] ${s}`) });

// Two-footed block figure (same as the Python tests' critter): STL written by trimesh inside Pyodide.
const file = process.argv[2];
let name = "critter.stl";
let data: Uint8Array;
if (file) {
  name = file.split(/[\\/]/).pop()!;
  data = new Uint8Array(readFileSync(file));
} else {
  pyodide.runPython(`
import trimesh
def box(size, at):
    return trimesh.creation.box(size).apply_translation(at)
legs = [box([9, 9, 12], [s * 8.5, 0, 6]) for s in (-1, 1)]
body = box([26, 9, 18], [0, 0, 12 + 9 - 0.01])
m = trimesh.util.concatenate(legs + [body])
m.export("/tmp/critter.stl")
`);
  data = pyodide.FS.readFile("/tmp/critter.stl");
}

const loaded = JSON.parse(api.load(name, data));
const rep = loaded.report;
console.log(`[${secs()} s] loaded ${loaded.file}: scale ${(rep.fit.scale * 100).toFixed(1)} %, ${rep.fit.height_mm.toFixed(1)} mm, ` +
  `${rep.pockets.length} pockets, status ${rep.status}`);
for (const c of rep.checks) if (c.status !== "pass") console.log(`  ${c.status}: ${c.title} — ${c.message}`);

const params = JSON.stringify({ ...rep.params, magnets: 3, rotate_z: 45, ball_diameter: 50 });
const rep2 = JSON.parse(api.run_analyse(loaded.session, params));
console.log(`[${secs()} s] 3 magnets, 45°, Ø50: scale ${(rep2.fit.scale * 100).toFixed(1)} %, ${rep2.pockets.length} pockets`);

const glb = api.preview(loaded.session, params);
const bytes = glb.toJs();
glb.destroy();
const [resJson, files] = api.run_export(loaded.session, params).toJs({ create_proxies: false });
const res = JSON.parse(resJson);
console.log(`[${secs()} s] preview ${bytes.length} B; export ${files.map(([n, b]) => `${n} (${b.length} B)`).join(", ")}`);

const problems: string[] = [];
if (!(rep.fit.scale > 0)) problems.push("figure does not fit");
if (rep.pockets.length !== 2) problems.push(`expected 2 pockets, got ${rep.pockets.length}`);
if (rep2.ball.inner_diameter.toFixed(1) !== "50.0") problems.push(`diameter ${rep2.ball.inner_diameter}`);
if (String.fromCharCode(...bytes.slice(0, 4)) !== "glTF") problems.push("preview is not a GLB");
if (!files.some(([n]) => n.endsWith(".3mf"))) problems.push("no 3MF exported");
if (!rep.checks.some((c: { id: string }) => c.id.startsWith("cut:"))) problems.push("pockets were not cut");
if (res.status === undefined) problems.push("export result missing status");
if (problems.length) {
  console.error(`FAILED: ${problems.join("; ")}`);
  process.exit(1);
}
console.log(`engine smoke test passed in ${secs()} s`);
