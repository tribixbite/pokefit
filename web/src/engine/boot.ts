/**
 * Start the Python engine under Pyodide: shared by the browser Web Worker and the
 * Node smoke test (scripts/engine-smoke.ts), so both exercise the same set-up.
 */
import type { PyodideInterface } from "pyodide";

/** Exact versions: the engine is tested against these. */
export const PYODIDE_VERSION = "0.29.5";
export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
/** Pyodide-built packages the engine imports (manifold3d is replaced by manifold-3d's WASM build). */
export const PY_PACKAGES = ["numpy", "scipy", "shapely", "lxml", "pillow", "contourpy", "micropip"];
/**
 * Pure-Python wheels installed by URL without their dependencies: Pyodide's
 * networkx recipe would pull in matplotlib (~10 MB) that trimesh's path code
 * never uses. build.ts copies them into the site (`wheels/`) so the live app
 * does not depend on PyPI; the Node smoke test fetches them from PyPI.
 */
export const PY_WHEELS = [
  "https://files.pythonhosted.org/packages/b9/54/dd730b32ea14ea797530a4479b2ed46a6fb250f682a9cfb997e968bf0261/networkx-3.4.2-py3-none-any.whl",
  "https://files.pythonhosted.org/packages/8f/f6/fb68b8c08f864ecee8f7f7ee2f1203ed5cba2335215de69e4f621eb47db5/trimesh-5.1.1-py3-none-any.whl",
];

/** A wheel's file name (the last URL segment). */
export const wheelName = (url: string): string => url.slice(url.lastIndexOf("/") + 1);

/** The `pokefit.webapi` module as seen from JS. */
export interface WebApi {
  config(): string;
  /** `params`: the UI's settings as JSON (the first analysis uses them; default settings when left out). */
  load(name: string, data: Uint8Array, params?: string): string;
  set_reference(name: string, data: Uint8Array): string;
  set_ball_model(name: string, data: Uint8Array): string;
  run_analyse(sid: string, params: string): string;
  washers(sid: string, params: string): string;
  preview(sid: string, params: string, regions: string): { toJs(): Uint8Array; destroy(): void };
  faces(sid: string, params: string, regions: string): string;
  pick_face(sid: string, params: string, req: string): string;
  paint_prepare(sid: string, params: string, req: string): string;
  colour_pieces(sid: string, params: string, req: string): string;
  run_export(sid: string, params: string): { toJs(o?: object): [string, [string, Uint8Array][]]; destroy(): void };
}

export interface BootOptions {
  pyodide: PyodideInterface;
  /** The initialised manifold-3d module (after `setup()`). */
  manifold: unknown;
  /** Zip of the `pokefit` package (see build.ts). */
  pokefitZip: ArrayBuffer;
  /** Where the vendored wheels are served (ends in "/"); default: PyPI. */
  wheelBase?: string;
  onStatus?: (text: string) => void;
}

export async function bootPokefit(o: BootOptions): Promise<WebApi> {
  const py = o.pyodide;
  const status = o.onStatus ?? (() => {});
  // pokefit/_manifold_js.py reads it from the JS global scope.
  (globalThis as unknown as { manifoldWasm: unknown }).manifoldWasm = o.manifold;
  status("Loading numpy, scipy, shapely…");
  await py.loadPackage(PY_PACKAGES);
  status("Loading trimesh, networkx…");
  const micropip = py.pyimport("micropip");
  const wheels = PY_WHEELS.map((u) => (o.wheelBase ? o.wheelBase + wheelName(u) : u));
  await micropip.install.callKwargs(wheels, { deps: false });
  micropip.destroy();
  py.unpackArchive(o.pokefitZip, "zip", { extractDir: "/home/pyodide" });
  status("Starting pokefit…");
  return py.pyimport("pokefit.webapi") as unknown as WebApi;
}
