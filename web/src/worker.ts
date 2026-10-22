/**
 * Web Worker running the pokefit Python engine in the browser (Pyodide), so the
 * static site (pokefit.party) needs no server. Messages:
 *   in:  { id, fn, args }        fn ∈ config | load | setReference | setBallModel | analyse | washers | preview | faces | pick | paintPrepare | export
 *   out: { id, ok: true, result } | { id, ok: false, error } | { status: string }
 */
import Module from "manifold-3d";
import { bootPokefit, PYODIDE_CDN, type WebApi } from "./engine/boot";
import { versioned } from "./version";

/** The worker's global scope (typed here: the project compiles against the DOM lib, not WebWorker). */
const scope = globalThis as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent) => void) | null;
  location: { href: string };
};

const status = (text: string): void => scope.postMessage({ status: text });

async function start(): Promise<WebApi> {
  status("Downloading the engine (first visit: ~30 MB, then cached)…");
  // Pyodide is loaded from its CDN at run time (not bundled): the URL is not a literal on purpose.
  const pyodideUrl = `${PYODIDE_CDN}pyodide.mjs`;
  const { loadPyodide } = (await import(/* @vite-ignore */ pyodideUrl)) as typeof import("pyodide");
  const [pyodide, manifold, zip] = await Promise.all([
    loadPyodide({ indexURL: PYODIDE_CDN }),
    Module({ locateFile: () => versioned(new URL("manifold.wasm", scope.location.href)) }).then((m) => {
      m.setup();
      return m;
    }),
    fetch(versioned(new URL("pokefit.zip", scope.location.href))).then((r) => {
      if (!r.ok) throw new Error(`pokefit.zip: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  const api = await bootPokefit({
    pyodide, manifold, pokefitZip: zip, onStatus: status,
    wheelBase: new URL("wheels/", scope.location.href).href,
  });
  status("ready");
  return api;
}

const ready = start();
ready.catch((e: unknown) => status(`failed: ${e instanceof Error ? e.message : String(e)}`));

type Call = { id: number; fn: string; args: unknown[] };

scope.onmessage = async (ev: MessageEvent<Call>): Promise<void> => {
  const { id, fn, args } = ev.data;
  try {
    const api = await ready;
    let result: unknown;
    const transfer: Transferable[] = [];
    switch (fn) {
      case "config": result = api.config(); break;
      case "load": result = api.load(args[0] as string, args[1] as Uint8Array, args[2] as string); break;
      case "setReference": result = api.set_reference(args[0] as string, args[1] as Uint8Array); break;
      case "setBallModel": result = api.set_ball_model(args[0] as string, args[1] as Uint8Array); break;
      case "analyse": result = api.run_analyse(args[0] as string, args[1] as string); break;
      case "washers": result = api.washers(args[0] as string, args[1] as string); break;
      case "preview": {
        const p = api.preview(args[0] as string, args[1] as string, (args[2] as string | undefined) ?? "null");
        const bytes = p.toJs();
        p.destroy();
        transfer.push(bytes.buffer);
        result = bytes;
        break;
      }
      case "faces": result = api.faces(args[0] as string, args[1] as string, (args[2] as string | undefined) ?? "null"); break;
      case "pick": result = api.pick_face(args[0] as string, args[1] as string, args[2] as string); break;
      case "paintPrepare": result = api.paint_prepare(args[0] as string, args[1] as string, args[2] as string); break;
      case "export": {
        const p = api.run_export(args[0] as string, args[1] as string);
        const [json, files] = p.toJs({ create_proxies: false });
        p.destroy();
        for (const [, b] of files) transfer.push(b.buffer);
        result = [json, files];
        break;
      }
      default: throw new Error(`unknown call ${fn}`);
    }
    scope.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    // Python exceptions arrive with the full traceback; the last line is the useful message.
    const text = e instanceof Error ? e.message : String(e);
    const last = text.trim().split("\n").filter(Boolean).pop() ?? text;
    console.error(text);
    scope.postMessage({ id, ok: false, error: last.replace(/^\w+Error: /, "") });
  }
};
