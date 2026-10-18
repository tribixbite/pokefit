/** Thin typed client for the pokefit server. Every call rejects with the server's error text. */
import type { Config, ExportResponse, Job, LoadResponse, Params, Report, SliceVariants, WasherRow } from "./types";

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

/** Server configuration: washers, filament slots, slicer availability. */
export async function getConfig(): Promise<Config> {
  return (await (await check(await fetch("/api/config"))).json()) as Config;
}

/** Upload a .3mf/.stl/.obj; the server analyses it with default parameters. */
export async function loadFile(file: File): Promise<LoadResponse> {
  const res = await check(
    await fetch("/api/load", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file }),
  );
  return (await res.json()) as LoadResponse;
}

/** Load a file already on the server's disk (local use). */
export async function loadPath(path: string): Promise<LoadResponse> {
  return postJson<LoadResponse>("/api/load_path", { path });
}

export async function analyse(session: string, params: Params): Promise<Report> {
  return postJson<Report>("/api/analyse", { session, params });
}

export async function washers(session: string, params: Params): Promise<WasherRow[]> {
  return postJson<WasherRow[]>("/api/washers", { session, params });
}

export function previewUrl(session: string, params: Params): string {
  const q = new URLSearchParams({
    session,
    object: params.object_id ?? "",
    rx: String(params.rotate_x),
    ry: String(params.rotate_y),
  });
  return `/api/preview.glb?${q}`;
}

export async function exportFiles(session: string, params: Params): Promise<ExportResponse> {
  return postJson<ExportResponse>("/api/export", { session, params });
}

/** Start the Bambu slices and poll until they finish. */
export async function slice(
  session: string,
  params: Params,
  x10: boolean,
  onTick?: (seconds: number) => void,
): Promise<SliceVariants> {
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
