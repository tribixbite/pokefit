/**
 * Typed client for the pokefit engine. Two back ends, same calls:
 *   - "local":   the Python server (`python -m pokefit serve`), with Bambu slicing;
 *   - "browser": the same Python engine in a Web Worker (Pyodide), as on pokefit.party.
 * The local server is used whenever the page is served by it (/api/config answers JSON).
 * Every call rejects with the engine's error text.
 */
import type { Config, ExportResponse, Job, LoadResponse, Params, Report, SliceVariants, WasherRow } from "./types";

export type Mode = "local" | "browser";
let mode: Mode = "browser";

// ------------------------------------------------------------------ local server (HTTP)

async function check(res: Response): Promise<Response> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) msg = body.error;
    } catch {
      /* non-JSON error body: keep the status text */
    }
    throw new Error(msg);
  }
  return res;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await check(
    await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  );
  return (await res.json()) as T;
}

// ------------------------------------------------------------------ browser engine (Web Worker)

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
const statusListeners = new Set<(text: string) => void>();
let lastStatus = "";

/** Follow the browser engine's start-up ("ready" once it can take calls). */
export function onEngineStatus(cb: (text: string) => void): void {
  statusListeners.add(cb);
  if (lastStatus) cb(lastStatus);
}

function startWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("worker.js", location.href), { type: "module" });
  worker.onmessage = (ev: MessageEvent): void => {
    const d = ev.data as { id?: number; ok?: boolean; result?: unknown; error?: string; status?: string };
    if (d.status !== undefined) {
      lastStatus = d.status;
      statusListeners.forEach((cb) => cb(d.status!));
      return;
    }
    const p = pending.get(d.id!);
    if (!p) return;
    pending.delete(d.id!);
    if (d.ok) p.resolve(d.result);
    else p.reject(new Error(d.error ?? "engine error"));
  };
  worker.onerror = (ev): void => {
    const msg = `engine crashed: ${ev.message || "unknown error"}`;
    lastStatus = `failed: ${msg}`;
    statusListeners.forEach((cb) => cb(lastStatus));
    pending.forEach((p) => p.reject(new Error(msg)));
    pending.clear();
  };
  return worker;
}

function call<T>(fn: string, args: unknown[] = [], transfer: Transferable[] = []): Promise<T> {
  const w = startWorker();
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    w.postMessage({ id, fn, args }, transfer);
  });
}

const json = <T>(s: unknown): T => JSON.parse(s as string) as T;

// ------------------------------------------------------------------ public API

export function engineMode(): Mode {
  return mode;
}

/** Engine configuration: washers, filament slots, slicer availability. Decides local vs browser. */
export async function getConfig(): Promise<Config> {
  try {
    const res = await fetch("/api/config", { headers: { Accept: "application/json" } });
    if (res.ok && (res.headers.get("content-type") ?? "").includes("json")) {
      mode = "local";
      return (await res.json()) as Config;
    }
  } catch {
    /* no server: static hosting */
  }
  mode = "browser";
  startWorker();                     // warm the engine up while the user picks a file
  // The same config the engine would return, generated at build time (build.ts), so the
  // page is usable at once instead of after the ~30 MB engine download.
  return (await (await check(await fetch("config.json"))).json()) as Config;
}

/** Open a .3mf/.stl/.obj; the engine analyses it with default parameters. */
export async function loadFile(file: File): Promise<LoadResponse> {
  if (mode === "local") {
    const res = await check(
      await fetch("/api/load", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file }),
    );
    return (await res.json()) as LoadResponse;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  return json<LoadResponse>(await call("load", [file.name, bytes], [bytes.buffer]));
}

/** Load a file already on the server's disk (local engine only). */
export async function loadPath(path: string): Promise<LoadResponse> {
  if (mode !== "local") throw new Error("Opening a path needs the local version; drop the file instead.");
  return postJson<LoadResponse>("/api/load_path", { path });
}

/** Use the filament slots of the user's own Bambu project (for the import file). */
export async function setReference(file: File): Promise<Config> {
  if (mode === "local") {
    const res = await check(
      await fetch("/api/reference", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file }),
    );
    return (await res.json()) as Config;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  return json<Config>(await call("setReference", [file.name, bytes], [bytes.buffer]));
}

export async function analyse(session: string, params: Params): Promise<Report> {
  if (mode === "local") return postJson<Report>("/api/analyse", { session, params });
  return json<Report>(await call("analyse", [session, JSON.stringify(params)]));
}

export async function washers(session: string, params: Params): Promise<WasherRow[]> {
  if (mode === "local") return postJson<WasherRow[]>("/api/washers", { session, params });
  return json<WasherRow[]>(await call("washers", [session, JSON.stringify(params)]));
}

let lastPreview = "";

/** A URL of the figure preview (GLB) for three.js: a server URL, or a blob URL from the worker. */
export async function previewSource(session: string, params: Params): Promise<string> {
  if (mode === "local") {
    const q = new URLSearchParams({
      session, object: params.object_id ?? "",
      rx: String(params.rotate_x), ry: String(params.rotate_y), rz: String(params.rotate_z),
    });
    return `/api/preview.glb?${q}`;
  }
  const bytes = await call<Uint8Array>("preview", [session, JSON.stringify(params)]);
  if (lastPreview) URL.revokeObjectURL(lastPreview);
  lastPreview = URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
  return lastPreview;
}

export async function exportFiles(session: string, params: Params): Promise<ExportResponse> {
  if (mode === "local") return postJson<ExportResponse>("/api/export", { session, params });
  const [resJson, files] = await call<[string, [string, Uint8Array][]]>("export", [session, JSON.stringify(params)]);
  const res = JSON.parse(resJson) as ExportResponse;
  res.files = files.map(([name, bytes]) => ({
    name, path: "",
    url: URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" })),
  }));
  return res;
}

/** Start the Bambu slices and poll until they finish (local engine only). */
export async function slice(
  session: string,
  params: Params,
  x10: boolean,
  onTick?: (seconds: number) => void,
): Promise<SliceVariants> {
  if (mode !== "local") throw new Error("Slicing needs Bambu Studio: run pokefit locally.");
  const { job } = await postJson<{ job: string }>("/api/slice", { session, params, x10 });
  const t0 = performance.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    onTick?.((performance.now() - t0) / 1000);
    const j = (await (await check(await fetch(`/api/job?id=${job}`))).json()) as Job<SliceVariants>;
    if (j.state === "done" && j.result) return j.result;
    if (j.state === "error") throw new Error(j.error ?? "slicing failed");
  }
}
