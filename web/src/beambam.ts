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

/** What the daemon says it can do (`/healthz` capabilities); empty for a beambam that predates them. */
export async function capabilities(link: Link): Promise<string[]> {
  try {
    const h = await request(link, "/healthz");
    return Array.isArray(h.capabilities) ? h.capabilities.map(String) : [];
  } catch {
    return [];        // unhealthy (503) or an older daemon: assume nothing beyond the basics
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

/** A finished slice on the daemon's machine. */
export interface DaemonSlice {
  /** The .gcode.3mf's path on the daemon's machine (what /print/plan and /print take). */
  file: string;
  seconds: number;
  grams: number;
  grams_per_filament: number[];
  warnings: string[];
}

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
  for (;;) {
    await sleep(SLICE_POLL_MS);
    const job = await request(link, poll);
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
  onTick?: (seconds: number, what: string) => void): Promise<string> {
  const t0 = performance.now();
  const reply = await postJson(link, "/print", { file, slot: slots, confirm: true, accept_warnings: acceptWarnings });
  if (reply.ok === false) throw new DaemonError(String(reply.error ?? "beambam refused the print"));
  const poll = String(reply.poll ?? "");
  if (!poll) throw new DaemonError("beambam did not start a print job");
  for (;;) {
    await sleep(PRINT_POLL_MS);
    const job = await request(link, poll);
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
