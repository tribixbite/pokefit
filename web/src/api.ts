/**
 * Typed client for the pokefit engine. Two back ends, same calls:
 *   - "local":   the Python server (`python -m pokefit serve`), with Bambu slicing;
 *   - "browser": the same Python engine in a Web Worker (Pyodide), as on pokefit.party.
 * The local server is used whenever the page is served by it (/api/config answers JSON).
 * Every call rejects with the engine's error text.
 */
import type {
  ColourPiece, Config, ExportResponse, Job, LoadResponse, Params, PickRequest, PickResult, PrepareResult, PreviewFaces, PrintPlan, PrintProgress, RegionView, Report, SliceVariants, WasherRow,
} from "./types";
import { versioned } from "./version";

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
  worker = new Worker(versioned(new URL("worker.js", location.href)), { type: "module" });
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
  return (await (await check(await fetch(versioned("config.json")))).json()) as Config;
}

/** Open a .3mf/.stl/.obj/.glb; the engine simplifies it as `params` ask and analyses it with them. */
export async function loadFile(file: File, params: Params): Promise<LoadResponse> {
  if (mode === "local") {
    const res = await check(
      await fetch("/api/load", {
        method: "POST", body: file,
        headers: { "X-Filename": encodeURIComponent(file.name), "X-Params": encodeURIComponent(JSON.stringify(params)) },
      }),
    );
    return (await res.json()) as LoadResponse;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  return json<LoadResponse>(await call("load", [file.name, bytes, JSON.stringify(params)], [bytes.buffer]));
}

/** Load a file already on the server's disk (local engine only). */
export async function loadPath(path: string, params: Params): Promise<LoadResponse> {
  if (mode !== "local") throw new Error("Opening a path needs the local version; drop the file instead.");
  return postJson<LoadResponse>("/api/load_path", { path, params });
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

/** Browser engine: the user's copy of the ball model, so exports include the washer-seat shell. */
export async function setBallModel(file: File): Promise<string> {
  if (mode === "local") throw new Error("The local version reads pokeball.3mf from its folder.");
  const bytes = new Uint8Array(await file.arrayBuffer());
  return (json<{ name: string }>(await call("setBallModel", [file.name, bytes], [bytes.buffer]))).name;
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

/**
 * A blob URL of the figure preview (GLB) for three.js, with the colorize paint applied.
 * With `regions`, the figure is coloured by the regions a colorize tap would paint.
 */
export async function previewSource(session: string, params: Params, regions: RegionView | null = null): Promise<string> {
  let bytes: Uint8Array;
  if (mode === "local") {
    const r = await check(await fetch("/api/preview", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session, params, regions }),
    }));
    bytes = new Uint8Array(await r.arrayBuffer());
  } else {
    bytes = await call<Uint8Array>("preview", [session, JSON.stringify(params), JSON.stringify(regions)]);
  }
  if (lastPreview) URL.revokeObjectURL(lastPreview);
  lastPreview = URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
  return lastPreview;
}

/** Colours of the preview's faces; while `signature` matches the shown glTF, only these need to change. */
export async function previewFaces(session: string, params: Params, regions: RegionView | null = null): Promise<PreviewFaces> {
  if (mode === "local") return postJson<PreviewFaces>("/api/preview_faces", { session, params, regions });
  return json<PreviewFaces>(await call("faces", [session, JSON.stringify(params), JSON.stringify(regions)]));
}

/** The triangle under a tap on the preview (figure frame, scale 1) and how big its region is. */
export async function pick(session: string, params: Params, req: PickRequest): Promise<PickResult> {
  if (mode === "local") return postJson<PickResult>("/api/pick", { session, params, ...req });
  return json<PickResult>(await call("pick", [session, JSON.stringify(params), JSON.stringify(req)]));
}

/** Build what taps need (pick trees, edge topology, the mode's regions; the mirror plane) before the first tap. */
export async function preparePaint(session: string, params: Params,
  req: { mode: string; angle: number; mirror?: boolean }): Promise<PrepareResult> {
  if (mode === "local") return postJson<PrepareResult>("/api/paint_prepare", { session, params, ...req });
  return json<PrepareResult>(await call("paintPrepare", [session, JSON.stringify(params), JSON.stringify(req)]));
}

/** The connected patches of filament `filament`, largest first ("Split by piece"). */
export async function colourPieces(session: string, params: Params, filament: number): Promise<ColourPiece[]> {
  if (mode === "local") {
    return (await postJson<{ pieces: ColourPiece[] }>("/api/colour_pieces", { session, params, filament })).pieces;
  }
  return json<{ pieces: ColourPiece[] }>(await call("colourPieces", [session, JSON.stringify(params),
    JSON.stringify({ filament })])).pieces;
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

/** Read (or with `clear`, forget) the printer's loaded AMS filaments through beambam (local only). */
export async function readPrinter(clear = false): Promise<Config> {
  if (mode !== "local") throw new Error("Reading the printer needs the local version with beambam.");
  return postJson<Config>("/api/printer", { clear });
}

/** What the printer is doing (read-only, local only). */
export async function printerStatus(): Promise<PrintProgress> {
  if (mode !== "local") throw new Error("Watching the printer needs the local version with beambam.");
  return (await postJson<{ progress: PrintProgress }>("/api/printer/status", {})).progress;
}

/** Start (`on`) or stop the printer's live view; returns its MJPEG URL when on. */
export async function printerCamera(on: boolean): Promise<string | null> {
  if (mode !== "local") throw new Error("Watching the printer needs the local version with beambam.");
  return (await postJson<{ url: string | null }>("/api/printer/camera", { on })).url;
}

/** Pixal3D generation settings (pokefit/imageto3d.py GenOptions; the same defaults). */
export interface GenOptions {
  seed: number;
  faces: number;
  detail: number;
  remesh: number;
  smooth: number;
  steps: number;
  /** 0 = per-vertex colour; else the size of a baked colour texture. */
  texture: number;
  /** One photo: "pixal3d" or "trellis2". */
  model: string;
  /** CFG of the shape samplers (1-15) and of the colour sampler (1-10). */
  guidance: number;
  tex_guidance: number;
  /** Structure sampler shift (1-10). */
  shift: number;
  /** KSampler sampler of every stage (imageto3d.SAMPLERS). */
  sampler: string;
  /** Loose pieces under this share of the largest are dropped (0-0.2). */
  min_piece: number;
}

export const GEN_DEFAULTS: GenOptions = {
  seed: 0, faces: 200_000, detail: 1536, remesh: 768, smooth: 20, steps: 1, texture: 0, model: "pixal3d",
  guidance: 7.5, tex_guidance: 1, shift: 5, sampler: "euler", min_piece: 0.01,
};

/** Photos of a figure (by view) -> a coloured 3D figure via the local ComfyUI, loaded like a file. */
export async function photosToFigure(
  photos: Partial<Record<"front" | "left" | "back" | "right" | "photo", File>>,
  mirrorRight: boolean,
  onTick?: (seconds: number, progress: string) => void,
  options: GenOptions = GEN_DEFAULTS,
  photoColour = true,
  /** Ordinary photos (1-3) to redraw as clean views first (pokefit/photoclean.py), and whether to re-pose. */
  clean: File[] = [],
  pose = false,
): Promise<LoadResponse> {
  if (mode !== "local") throw new Error("Photos to 3D runs on your computer: install pokefit and ComfyUI (see About).");
  const dataUrl = (f: File): Promise<string> => new Promise((ok, fail) => {
    const r = new FileReader();
    r.onload = () => ok(String(r.result));
    r.onerror = () => fail(r.error ?? new Error(`cannot read ${f.name}`));
    r.readAsDataURL(f);
  });
  const views: Record<string, string> = {};
  for (const [v, f] of Object.entries(photos)) if (f) views[v] = await dataUrl(f);
  const noisy = await Promise.all(clean.slice(0, 3).map(dataUrl));
  const name = (photos.front?.name ?? clean[0]?.name ?? "figure").replace(/\.[^.]+$/, "").replace(/[-_ ]?front$/i, "") || "figure";
  const { job } = await postJson<{ job: string }>("/api/photos", { views, mirror_right: mirrorRight, name, options,
    // Redrawn views keep the model's own colours (the server's default for them; see post_photos).
    ...(noisy.length ? {} : { photo_colour: photoColour }), clean: noisy, pose });
  return waitJob<LoadResponse>(job, 2000, onTick, "photos to 3D failed");
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
  return waitJob<SliceVariants>(job, 1000, onTick, "slicing failed");
}

/** Poll a local-server job (GET /api/job) every `everyMs` until it is done; `onTick` gets the seconds so
 * far and the job's progress text. */
async function waitJob<T>(job: string, everyMs: number, onTick: ((seconds: number, progress: string) => void) | undefined,
  failed: string): Promise<T> {
  const t0 = performance.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, everyMs));
    const j = (await (await check(await fetch(`/api/job?id=${job}`))).json()) as Job<T>;
    onTick?.((performance.now() - t0) / 1000, j.progress ?? "");
    if (j.state === "done" && j.result) return j.result;
    if (j.state === "error") throw new Error(j.error ?? failed);
  }
}

/** What a sliced print file would print, each filament from its AMS slot (no printer action). */
export async function printPlan(file: string, slots: number[]): Promise<PrintPlan> {
  if (mode !== "local") throw new Error("Printing needs the local version with beambam.");
  return postJson<PrintPlan>("/api/printer/print", { file, slots });
}

/** Upload the print file and START THE PRINT (the user's explicit "Start print"). Returns beambam's output. */
export async function startPrint(file: string, slots: number[], onTick?: (seconds: number, progress: string) => void,
): Promise<string> {
  if (mode !== "local") throw new Error("Printing needs the local version with beambam.");
  const { job } = await postJson<{ job: string }>("/api/printer/print", { file, slots, confirm: true });
  return (await waitJob<{ output: string }>(job, 1500, onTick, "the print did not start")).output;
}
