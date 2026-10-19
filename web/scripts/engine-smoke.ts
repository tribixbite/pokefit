/**
 * Run the browser engine (Pyodide + manifold-3d WASM) under Node on a small
 * synthetic figure, the way the Web Worker does. CI runs this before deploying.
 *
 *   node scripts/engine-smoke.ts [figure.3mf|.stl]
 */
import { existsSync, readFileSync } from "node:fs";
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

// With the (third-party, not in the repo) ball model present, exports must include the shell.
const ballModel = join(root, "pokeball.3mf");
const haveBall = existsSync(ballModel);
if (haveBall) console.log(`[${secs()} s] ball model: ${api.set_ball_model("pokeball.3mf", new Uint8Array(readFileSync(ballModel)))}`);

// Colorize: tap the left leg's outer side (figure frame, file size); a region stops at the box edges,
// a piece is the whole leg. Paint the leg with a second, added colour.
const plain = JSON.stringify(rep.params);
const side = JSON.parse(api.pick_face(loaded.session, plain,
  JSON.stringify({ point: [-13, 0, 6], normal: [-1, 0, 0], mode: "region", angle: 30 })));
const leg = JSON.parse(api.pick_face(loaded.session, plain,
  JSON.stringify({ point: [-13, 0, 6], normal: [-1, 0, 0], mode: "piece", angle: 30 })));
const painted = JSON.stringify({ ...rep.params, colours: ["#E05028", "#161616"],
  paint: [{ part: leg.part, face: leg.face, filament: 2, mode: "piece", angle: 30 }] });
const rep3 = JSON.parse(api.run_analyse(loaded.session, painted));
const regionGlb = api.preview(loaded.session, painted, JSON.stringify({ mode: "region", angle: 30 }));
const regionBytes = regionGlb.toJs().length;
regionGlb.destroy();
console.log(`[${secs()} s] colorize: region ${side.faces} faces, piece ${leg.faces} faces; ` +
  `painted ${rep3.object.parts.map((x: { painted: number }) => x.painted).join("+")} faces; regions preview ${regionBytes} B`);
const [paintedJson] = api.run_export(loaded.session, painted).toJs({ create_proxies: false });

const glb = api.preview(loaded.session, params, "null");
const bytes = glb.toJs();
glb.destroy();
const [resJson, files] = api.run_export(loaded.session, params).toJs({ create_proxies: false });
const res = JSON.parse(resJson);
console.log(`[${secs()} s] preview ${bytes.length} B; export ${files.map(([n, b]) => `${n} (${b.length} B)`).join(", ")}`);

// GLB: the critter as a glTF scene (Y-up, metres) with red legs and a (larger, so first) blue body,
// and a sphere textured red/blue: both open as one part with their colours painted on, in mm.
pyodide.runPython(`
import numpy as np, trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
def box(size, at, rgb):
    m = trimesh.creation.box([s / 1000 for s in size]).apply_translation([a / 1000 for a in at])
    m.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=[*rgb, 255]))
    return m
y_up = trimesh.transformations.rotation_matrix(-1.5707963267948966, [1, 0, 0])
s = trimesh.Scene()
for i, m in enumerate([box([9, 9, 12], [x * 8.5, 0, 6], (200, 20, 20)) for x in (-1, 1)] +
                      [box([26, 9, 18], [0, 0, 20.99], (20, 20, 200))]):
    s.add_geometry(m.apply_transform(y_up), node_name=f"n{i}")
open("/tmp/critter.glb", "wb").write(s.export(file_type="glb"))
ball = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
n = ball.vertices / np.linalg.norm(ball.vertices, axis=1, keepdims=True)
img = np.zeros((64, 64, 3), np.uint8); img[:, :32] = (220, 30, 30); img[:, 32:] = (30, 30, 220)
ball.visual = trimesh.visual.TextureVisuals(uv=np.stack([0.5 + 0.5 * n[:, 0], 0.5 + 0.5 * n[:, 1]], 1),
                                            material=PBRMaterial(baseColorTexture=Image.fromarray(img)))
open("/tmp/ball.glb", "wb").write(trimesh.Scene(ball).export(file_type="glb"))
`);
const glbLoaded = JSON.parse(api.load("critter.glb", pyodide.FS.readFile("/tmp/critter.glb")));
const glbObj = glbLoaded.report.object;
const ballLoaded = JSON.parse(api.load("ball.glb", pyodide.FS.readFile("/tmp/ball.glb")));
const ballObj = ballLoaded.report.object;
console.log(`[${secs()} s] GLB: ${glbObj.parts.length} part, colours ${glbObj.colours.join(" ")}, ` +
  `${glbObj.parts[0].painted} faces painted, ${glbLoaded.report.fit.height_mm.toFixed(1)} mm at ` +
  `${(glbLoaded.report.fit.scale * 100).toFixed(1)} %; textured: ${ballObj.colours.join(" ")}, ` +
  `${ballObj.parts[0].painted} painted, texture_colours ${JSON.stringify(ballObj.texture_colours)}`);

// Bambu preset export (a "Filament presets.zip"): its filament becomes a selectable slot.
pyodide.runPython(`
import io, json, zipfile
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("eSUN PETG Basic.json", json.dumps({"filament_settings_id": ["eSUN PETG Basic"], "from": "User",
        "inherits": "Generic PETG @BBL X2D 0.4 nozzle", "name": "eSUN PETG Basic", "nozzle_temperature": ["250"]}))
open("/tmp/presets.zip", "wb").write(buf.getvalue())
`);
const presetCfg = JSON.parse(api.set_reference("Filament presets.zip", pyodide.FS.readFile("/tmp/presets.zip")));
const imported = presetCfg.slots.filter((x: { imported?: boolean }) => x.imported);
console.log(`[${secs()} s] presets: ${imported.map((x: { name: string }) => x.name).join(", ")} ` +
  `(${presetCfg.presets.map((x: { status: string }) => x.status).join(", ")})`);

// Conform: a round body with legs lying flat out bends its legs up and gets a flat base (scipy in Pyodide).
pyodide.runPython(`
import trimesh
body = trimesh.creation.icosphere(subdivisions=3, radius=7.0); body.apply_translation([0, 0, 9.0])
legs = [trimesh.creation.box((12.0, 4.0, 4.0)).subdivide().subdivide().subdivide() for _ in (0, 1)]
for leg, side in zip(legs, (-1, 1)): leg.apply_translation([side * 10.0, 0, 2.0])
trimesh.util.concatenate([body, *legs]).export("/tmp/sitting.stl")
`);
const sitting = JSON.parse(api.load("sitting.stl", pyodide.FS.readFile("/tmp/sitting.stl")));
const shaped = sitting.report.checks.find((c: { id: string }) => c.id === "conform");
console.log(`[${secs()} s] conform: ${shaped?.message ?? "none"} parts ${sitting.report.object.parts.length}`);

const problems: string[] = [];
if (!shaped || !(sitting.report.fit.conform?.bend > 0) || sitting.report.object.parts.length !== 2) {
  problems.push("conform (bend + flat base)");
}
if (imported.length !== 1 || imported[0].name !== "eSUN PETG Basic") problems.push("preset import");
const tcv = glbLoaded.report.toolchange;
console.log(`[${secs()} s] toolchange: ${JSON.stringify({ ...tcv, by_count: undefined })}`);
if (tcv?.nozzles !== 2 || tcv.filaments !== 2 || !Array.isArray(tcv.by_count)) problems.push("toolchange estimate");
if (glbObj.parts.length !== 1 || glbObj.colours.join() !== "#1414C8,#C81414" || glbObj.parts[0].painted < 12) {
  problems.push("GLB colours");
}
if (ballObj.colours.length !== 2 || !(ballObj.parts[0].painted > 0) || ballObj.texture_colours?.auto !== 2) {
  problems.push("GLB texture colours");
}
if (Math.abs(glbLoaded.report.fit.scale - rep.fit.scale) > 0.01) problems.push("GLB scale differs from the STL's");
if (!(rep.fit.scale > 0)) problems.push("figure does not fit");
if (side.faces !== 2) problems.push(`region pick: expected 2 faces (one box side), got ${side.faces}`);
if (leg.faces !== 12) problems.push(`piece pick: expected 12 faces (the leg box), got ${leg.faces}`);
if (!rep3.object.parts.some((x: { painted: number }) => x.painted >= 12)) problems.push("paint was not applied");
if (JSON.parse(paintedJson).status === undefined) problems.push("painted export failed");
if (rep.pockets.length !== 2) problems.push(`expected 2 pockets, got ${rep.pockets.length}`);
if (rep2.ball.inner_diameter.toFixed(1) !== "50.0") problems.push(`diameter ${rep2.ball.inner_diameter}`);
if (String.fromCharCode(...bytes.slice(0, 4)) !== "glTF") problems.push("preview is not a GLB");
if (!files.some(([n]) => n.endsWith(".3mf"))) problems.push("no 3MF exported");
if (!rep.checks.some((c: { id: string }) => c.id.startsWith("cut:"))) problems.push("pockets were not cut");
if (res.status === undefined) problems.push("export result missing status");
if (haveBall && !files.some(([n]) => n.startsWith("shell_washerseat"))) problems.push(`no shell exported: ${res.note}`);
if (problems.length) {
  console.error(`FAILED: ${problems.join("; ")}`);
  process.exit(1);
}
console.log(`engine smoke test passed in ${secs()} s`);
