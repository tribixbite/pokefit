/**
 * Remembered settings: the values a user changes once and wants on every figure (ball, washer,
 * magnets, pocket sizes, shape for the ball, print height). Kept in this browser's localStorage,
 * read at start-up as the defaults, so they can also be set before a file is opened.
 *
 * Per-file settings (size as a scale, rotation, pockets placed by hand, fill circle, paint,
 * colours, the figure picked in a multi-figure file) are never remembered.
 */
import type { Params } from "./types";

const KEY = "pokefit.prefs.v1";

/** Remembered params and the type each must have (null allowed where the param itself allows it). */
const REMEMBERED = {
  ball_scale: "number", ball_diameter: "number|null", washer: "string", clearance: "number",
  figure_height: "number|null", magnets: "number", pocket_diameter: "number", pocket_depth: "number",
  pocket_wall: "number", pocket_ceiling: "number", pocket_grip: "number", pocket_max_seat: "number",
  pocket_seat: "string", settings_source: "string", bend: "number|null", flat_base: "number|null",
  flat_depth: "number", flat_fill: "number", paint_depth: "number|null", strengthen: "number",
} as const satisfies Partial<Record<keyof Params, string>>;

export type RememberedKey = keyof typeof REMEMBERED;
/** Every remembered param name. */
export const REMEMBERED_KEYS = Object.keys(REMEMBERED) as RememberedKey[];
export type Prefs = Partial<Pick<Params, RememberedKey>>;

/** Minimal storage interface (localStorage, or a stub in tests). */
export interface PrefStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** The storage to use, or null where it is unavailable (private mode, blocked site data). */
export function browserStore(): PrefStore | null {
  try {
    const s = globalThis.localStorage;
    return s ?? null;
  } catch {
    return null;
  }
}

function valid(key: RememberedKey, v: unknown): boolean {
  const want = REMEMBERED[key];
  if (v === null) return want.endsWith("|null");
  if (want.startsWith("number")) return typeof v === "number" && Number.isFinite(v);
  return typeof v === "string" && v.length > 0 && v.length < 64;
}

/** The remembered values in `params`, where they differ from `defaults` (only what the user changed). */
export function pickPrefs(params: Params, defaults: Params): Prefs {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(REMEMBERED) as RememberedKey[]) {
    if (params[k] !== defaults[k] && valid(k, params[k])) out[k] = params[k];
  }
  return out as Prefs;
}

/** Stored prefs, with anything malformed or unknown dropped (a newer or older version's, or edited by hand). */
export function readPrefs(store: PrefStore | null): Prefs {
  if (!store) return {};
  try {
    const raw: unknown = JSON.parse(store.getItem(KEY) ?? "{}");
    if (!raw || typeof raw !== "object") return {};
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(raw)) {
      if (k in REMEMBERED && valid(k as RememberedKey, v)) out[k] = v;
    }
    return out as Prefs;
  } catch {
    return {};
  }
}

/** Remember what in `params` differs from `defaults` (an empty set forgets everything). */
export function writePrefs(store: PrefStore | null, params: Params, defaults: Params): void {
  if (!store) return;
  try {
    const p = pickPrefs(params, defaults);
    if (Object.keys(p).length) store.setItem(KEY, JSON.stringify(p));
    else store.removeItem(KEY);
  } catch {
    // Storage full or blocked: the settings still apply to this visit.
  }
}

/** Forget every remembered setting. */
export function clearPrefs(store: PrefStore | null): void {
  try {
    store?.removeItem(KEY);
  } catch {
    // ignore: nothing was remembered
  }
}
