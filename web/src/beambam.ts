/**
 * The beambam daemon, called straight from the page: pokefit.party open in the browser of a phone or
 * computer that runs beambam (github.com/tribixbite/beambam) on the printer's network. The site cannot slice
 * or reach the printer itself; the daemon can:
 *
 *   GET  /state            the printer's report (loaded spools) — the page passes it to the engine
 *   POST /slice            a project .3mf -> job -> the .gcode.3mf on the daemon's machine
 *   POST /print/plan       beambam's filament -> tray check of that file against the live AMS
 *   POST /print            {confirm: true} after the user pressed Start print -> job -> accepted / rejected
 *
 * The daemon refuses POSTs from web pages it was not told about, so it must be started with this site's
 * origin: `beambam daemon --http 127.0.0.1:8765 --cors-origin https://pokefit.party` (`startCommand`). Chrome
 * also asks the user once to let the site reach apps on the device (Local Network Access); the daemon
 * answers the older Private Network Access preflight. The link is opt-in (`readLink`): a page that probed
 * 127.0.0.1 for every visitor would show them that prompt.
 */
import type { PrefStore } from "./prefs";
import type { PrintMapping } from "./types";

export const DEFAULT_URL = "http://127.0.0.1:8765";
const KEY = "pokefit.beambam";
const TIMEOUT_MS = 8000;
const SLICE_POLL_MS = 1000;
const PRINT_POLL_MS = 1500;
/** Uploading, signing and the printer's answer: `beambam print` waits 15 s for it. */
const PRINT_TIMEOUT_MS = 180_000;
const SLICE_TIMEOUT_MS = 20 * 60_000;

/** Where the daemon is, and whether the user turned the link on (remembered in this browser). */
export interface Link {
  url: string;
  enabled: boolean;
}

export function readLink(store: PrefStore | null): Link {
  try {
    const raw = store?.getItem(KEY);
    const v = raw ? (JSON.parse(raw) as Partial<Link>) : {};
    return { url: normaliseUrl(v.url ?? DEFAULT_URL), enabled: v.enabled === true };
  } catch {
    return { url: DEFAULT_URL, enabled: false };
  }
}

export function writeLink(store: PrefStore | null, link: Link): void {
  try {
    store?.setItem(KEY, JSON.stringify(link));
  } catch { /* storage blocked: the link lasts for this visit */ }
}

/** "127.0.0.1:8765/" -> "http://127.0.0.1:8765"; throws on something that is not an http(s) address. */
export function normaliseUrl(raw: string): string {
  const t = raw.trim().replace(/\/+$/, "");
  const u = new URL(/^https?:\/\//i.test(t) ? t : `http://${t}`);
  if (u.protocol !== "http:" && u.protocol !== "https:") throw new Error(`not a web address: ${raw}`);
  return `${u.protocol}//${u.host}`;
}

/** The command that starts a daemon this page may use. */
export function startCommand(origin: string, url: string = DEFAULT_URL): string {
  const host = new URL(url).host;
  return `beambam daemon --http ${host} --cors-origin ${origin}`;
}

/** A daemon failure; `unreachable` when no answer came at all (not running, blocked, or wrong origin). */
export class DaemonError extends Error {
  constructor(message: string, readonly unreachable = false) {
    super(message);
  }
}

type Json = Record<string, unknown>;

async function request(link: Link, path: string, init: RequestInit = {}, timeoutMs = TIMEOUT_MS): Promise<Json> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(link.url + path, { ...init, signal: ctl.signal, cache: "no-store" });
  } catch (e) {
    // A refused connection, a blocked one and a CORS refusal all look the same to the page.
    const why = (e as Error).name === "AbortError" ? "no answer" : "no connection";
    throw new DaemonError(`beambam is not reachable at ${link.url} (${why}). Start it on this device with ` +
      `\`${startCommand(location.origin, link.url)}\`, and allow pokefit to reach apps on this device if ` +
      "the browser asks.", true);
  } finally {
    clearTimeout(timer);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch { /* not JSON: reported below */ }
  const reply = (body && typeof body === "object" ? body : {}) as Json;
  // A plan with errors is still a plan: the panel shows why the print is refused.
  if (!res.ok && !(Array.isArray(reply.filaments) && (res.status === 409 || res.status === 422))) {
    throw new DaemonError(`beambam: ${typeof reply.error === "string" ? reply.error : `HTTP ${res.status}`}`);
  }
  return reply;
}

const postJson = (link: Link, path: string, body: unknown, timeoutMs?: number): Promise<Json> =>
  request(link, path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    timeoutMs);

/** What the daemon says it can do (`/healthz` capabilities); empty for a beambam that predates them. A 503 (the
 * printer is quiet) still carries them. */
export async function capabilities(link: Link): Promise<string[]> {
  try {
    const res = await fetch(`${link.url}/healthz`, { cache: "no-store" });
    const h = (await res.json()) as Json;
    return Array.isArray(h.capabilities) ? h.capabilities.map(String) : [];
  } catch {
    return [];        // unreachable or an older daemon: assume nothing beyond the basics
  }
}

/**
 * The chamber camera, live through beambam (`GET /camera.mjpeg`, capability "camera"), for an <img>: the
 * daemon relays its camera proxy's MJPEG. `n` makes each (re)connect a new URL.
 */
export function cameraUrl(link: Link, n: number = Date.now()): string {
  return `${link.url}/camera.mjpeg?n=${n}`;
}

/** Why the live view does not show, in beambam's words (its 503 text), or null when a frame comes through now. */
export async function cameraProblem(link: Link): Promise<string | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 25_000);
  try {
    const res = await fetch(`${link.url}/snapshot.jpg`, { signal: ctl.signal, cache: "no-store" });
    if (res.ok) return null;
    return (await res.text()).trim() || `beambam answered HTTP ${res.status}`;
  } catch (e) {
    return `beambam did not answer (${(e as Error).message})`;
  } finally {
    clearTimeout(timer);
  }
}

/** The printer's state report (the daemon's cached copy: milliseconds). */
export async function readState(link: Link): Promise<Json> {
  const st = await request(link, "/state");
  if (!st.print || typeof st.print !== "object") {
    throw new DaemonError("beambam has no report from the printer yet: is the printer on and on this network?");
  }
  return st;
}

/** One project filament's spool for beambam's `spools` option (null id / type: keep the slot's preset). */
export interface SpoolOption {
  id: string | null;
  type: string | null;
  colour: string;
}

/**
 * The project with filament i set to `spools[i]` (the spool's Bambu system preset, material and colour) and, with
 * `bedType`, its plate: beambam's POST /project/update ("project-sync"), which has the printer's bundled profiles.
 */
export async function updateProject(link: Link, name: string, bytes: Uint8Array, spools: SpoolOption[],
  bedType = ""): Promise<{ blob: Blob; warnings: string[] }> {
  const options: Record<string, unknown> = { spools };
  if (bedType) options.bed_type = bedType;
  let res: Response;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 180_000);
  try {
    res = await fetch(`${link.url}/project/update`, {
      method: "POST", cache: "no-store", body: bytes as BodyInit, signal: ctl.signal,
      headers: { "Content-Type": "application/octet-stream", "X-Filename": headerName(name),
        "X-Slice-Options": JSON.stringify(options) },
    });
  } catch {
    throw new DaemonError(`beambam is not reachable at ${link.url}`, true);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let said = "";
    try {
      const j = (await res.json()) as Json;
      said = typeof j.error === "string" ? j.error : "";
    } catch { /* not JSON */ }
    throw new DaemonError(`beambam could not update the project: ${said || `HTTP ${res.status}`}`);
  }
  // What the spools could not change (a material without a bundled profile, another nozzle size).
  let warnings: string[] = [];
  try {
    const w: unknown = JSON.parse(res.headers.get("X-Beambam-Warnings") ?? "[]");
    if (Array.isArray(w)) warnings = w.map(String);
  } catch { /* none */ }
  return { blob: await res.blob(), warnings };
}

/** The AMS trays in order (A1 … D4, then AMS HT) as `spools`: what "Use the printer's filaments" takes. */
export function traySpools(ams: { slot: number; type: string; colour: string; filament_id?: string }[]): SpoolOption[] {
  return ams.filter((a) => a.slot < 254).sort((a, b) => a.slot - b.slot)
    .map((a) => ({ id: a.filament_id || null, type: a.type || null, colour: a.colour.slice(0, 7).toUpperCase() }));
}

/** A finished slice on the daemon's machine. */
export interface DaemonSlice {
  /** The .gcode.3mf's path on the daemon's machine (what /print/plan and /print take). */
  file: string;
  seconds: number;
  grams: number;
  grams_per_filament: number[];
  warnings: string[];
}

/** Failed polls in a row before a print job is given up on. */
const POLL_RETRIES = 4;

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** An ASCII file name for the X-Filename header (header values cannot carry other characters). */
export function headerName(name: string): string {
  return name.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120) || "figure.3mf";
}

/** Send a project .3mf to the daemon's slicer and wait for the print file. */
export async function slice(link: Link, name: string, bytes: Uint8Array,
  onTick?: (seconds: number, what: string) => void): Promise<DaemonSlice> {
  const t0 = performance.now();
  const started = await request(link, "/slice", {
    method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Filename": headerName(name) },
    body: bytes as BodyInit,
  }, 120_000);
  const poll = String(started.poll ?? "");
  if (!poll) throw new DaemonError("beambam did not start the slice");
  let failures = 0;
  for (;;) {
    await sleep(SLICE_POLL_MS);
    let job: Json;
    try {
      job = await request(link, poll);
      failures = 0;
    } catch (e) {
      if (++failures < POLL_RETRIES) continue;      // a blip: the slice goes on on beambam
      throw e;
    }
    onTick?.((performance.now() - t0) / 1000, String(job.message ?? job.state ?? ""));
    if (job.done) {
      if (job.state === "done") return job as unknown as DaemonSlice;
      throw new DaemonError(`Slicing failed: ${String(job.message ?? "unknown error")}`);
    }
    if (performance.now() - t0 > SLICE_TIMEOUT_MS) throw new DaemonError("Slicing is taking too long: check beambam");
  }
}

/** beambam's filament -> tray plan for `file`: `slots` one tray per filament in file order, or null for
 * beambam's own choice (closest loaded spool of the right material). Touches nothing. */
export async function plan(link: Link, file: string, slots: number[] | null): Promise<PrintMapping> {
  return await postJson(link, "/print/plan", slots ? { file, slot: slots } : { file }) as unknown as PrintMapping;
}

/** Start the print (the user pressed Start print, having seen the plan); resolves once the printer accepted it. */
export async function print(link: Link, file: string, slots: number[], acceptWarnings: boolean,
  onTick?: (seconds: number, what: string) => void, digest?: string): Promise<string> {
  const t0 = performance.now();
  // `plan_digest` (beambam ≥ 1.9): start only the plan on screen; a changed one answers 409 with the new plan.
  const req: Json = { file, slot: slots, confirm: true, accept_warnings: acceptWarnings };
  if (digest) req.plan_digest = digest;
  const reply = await postJson(link, "/print", req);
  if (reply.plan_changed === true) {
    throw new DaemonError("The spools, the plate or the file changed since this check: look at it again, then Start.");
  }
  if (reply.ok === false) throw new DaemonError(String(reply.error ?? "beambam refused the print"));
  const poll = String(reply.poll ?? "");
  if (!poll) throw new DaemonError("beambam did not start a print job");
  let failures = 0;
  for (;;) {
    await sleep(PRINT_POLL_MS);
    let job: Json;
    try {
      job = await request(link, poll);
      failures = 0;
    } catch {
      // beambam took the print: a lost poll must not read as "start it again".
      if (++failures < POLL_RETRIES) continue;
      throw new DaemonError("Lost contact with beambam after it took the print: it may have started. " +
        "Check the printer before trying again.");
    }
    onTick?.((performance.now() - t0) / 1000, String(job.message ?? job.state ?? ""));
    if (job.done) {
      if (job.state === "accepted" || job.state === "sent") return String(job.message ?? job.state);
      throw new DaemonError(`The print did not start (${String(job.state)}): ${String(job.message ?? "")}`);
    }
    if (performance.now() - t0 > PRINT_TIMEOUT_MS) {
      throw new DaemonError("No answer from the printer yet: check it before trying again");
    }
  }
}
