/**
 * Time the browser engine (Pyodide + manifold-3d WASM, under Node) through the steps the UI runs:
 * open a figure, preview it, tap to paint, recolour, re-analyse. For finding what makes the site slow.
 *
 *   node scripts/engine-bench.ts figure.stl|.3mf|.glb
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Module from "manifold-3d";
import { loadPyodide } from "pyodide";
import { bootPokefit, fetchWheels, PYODIDE_CDN } from "../src/engine/boot.ts";
import { pokefitZip } from "../pack.ts";

const here = dirname(fileURLToPath(import.meta.url));
const file = process.argv[2];
if (!file) throw new Error("usage: node scripts/engine-bench.ts <figure>");

/** Run `fn`, print how long it took, return its result. */
function step<T>(label: string, fn: () => T): T {
  const t = performance.now();
  const out = fn();
  console.log(`${label.padEnd(34)} ${((performance.now() - t) / 1000).toFixed(2).padStart(7)} s`);
  return out;
}

let t = performance.now();
const pyodide = await loadPyodide({ packageBaseUrl: PYODIDE_CDN, packageCacheDir: join(here, "..", ".pyodide-cache") });
const manifold = await Module();
manifold.setup();
const api = await bootPokefit({ pyodide, manifold, pokefitZip: pokefitZip(join(here, "..", "..")).buffer as ArrayBuffer, wheels: await fetchWheels() });
console.log(`${"engine boot".padEnd(34)} ${((performance.now() - t) / 1000).toFixed(2).padStart(7)} s`);

const name = file.split(/[\\/]/).pop()!;
const loaded = step("open + first analysis", () => JSON.parse(api.load(name, new Uint8Array(readFileSync(file)))));
const sid: string = loaded.session;
const params = loaded.report.params;
const plain = JSON.stringify(params);
const glbBytes = step("preview glTF", () => {
  const p = api.preview(sid, plain, "null");
  const n = p.toJs().length;
  p.destroy();
  return n;
});
const faces = step("preview face colours", () => api.faces(sid, plain, "null"));
console.log(`  glTF ${(glbBytes / 1e6).toFixed(1)} MB, face colours ${(faces.length / 1e6).toFixed(2)} MB`);

// Tap the figure's front, half way up (figure frame, scale 1): the nearest triangle.
const h = loaded.report.fit.unit_height as number;
const tap = (mode: string) => JSON.stringify({ point: [0, -1000, h / 2], normal: null, mode, angle: 15, radius: 0 });
const hit = step("first tap (topology)", () => JSON.parse(api.pick_face(sid, plain, tap("part"))));
step("second tap", () => JSON.parse(api.pick_face(sid, plain, tap("region"))));
const colours = [...loaded.report.object.colours, "#E05028"];
const painted = JSON.stringify({ ...params, colours,
  paint: [{ part: hit.part, face: hit.face, filament: colours.length, mode: "part", angle: 15, radius: 0 }] });
step("recolour after the tap", () => api.faces(sid, painted, "null"));
step("analysis after painting", () => api.run_analyse(sid, painted));
const again = JSON.stringify({ ...JSON.parse(painted), colours: [...colours.slice(0, -1), "#2050E0"] });
step("palette edit: recolour", () => api.faces(sid, again, "null"));
step("palette edit: analysis", () => api.run_analyse(sid, again));
step("show regions", () => api.faces(sid, plain, JSON.stringify({ mode: "part", angle: 15 })));
