/**
 * "Changes to your model": every setting that is changing the figure's shape right now (simplified, limbs reshaped,
 * legs bent, base pressed or filled, parts thickened), where each came from, and the values that undo it.
 *
 * Why: settings are remembered from one figure to the next (prefs.ts), and the engine's defaults press and fill the
 * base. A bend or a fill chosen for the last figure can quietly reshape the next one; listing them with an Undo, and
 * marking their controls, keeps the user in charge of what happens to their model.
 */
import type { Params, Report } from "./types";

/** Where a change's setting came from. */
export type ChangeOrigin = "remembered" | "default" | "this figure" | "file";

export interface ModelChange {
  /** Stable id (one per kind of change). */
  id: string;
  /** What happened to the model, e.g. "Legs/tail bent up". */
  label: string;
  /** How much, e.g. "up to 2.1 mm". */
  detail: string;
  origin: ChangeOrigin;
  /** Params that undo it ("leave the model as it is"). */
  undo: Partial<Params>;
  /** The id of the control that sets it (marked while the change is on). */
  control: string;
}

/** Settings remembered between figures that change the shape (prefs.ts REMEMBERED). */
const SHAPE_KEYS = ["bend", "flat_depth", "flat_fill", "flat_base", "fill_area", "simplify"] as const;
type ShapeKey = (typeof SHAPE_KEYS)[number];

/**
 * Where the current value of `keys` came from: a value equal to the remembered one (and not the engine's default)
 * came from the last figure; equal to the default, from the defaults; else it was set for this figure.
 */
function origin(keys: ShapeKey[], params: Params, defaults: Params, remembered: Partial<Params>): ChangeOrigin {
  const r = remembered as Record<string, unknown>;
  if (keys.some((k) => k in r && r[k] === params[k] && params[k] !== defaults[k])) return "remembered";
  if (keys.every((k) => params[k] === defaults[k])) return "default";
  return "this figure";
}

function mm(v: number): string {
  return `${v.toFixed(v < 1 ? 2 : 1)} mm`;
}

/**
 * The changes `report` (the last analysis of `params`) shows the engine made to the figure. `remembered`: the values
 * the session started from (prefs), to tell a carried-over setting from one set for this figure.
 */
export function modelChanges(params: Params, report: Report | null, defaults: Params,
  remembered: Partial<Params>): ModelChange[] {
  const out: ModelChange[] = [];
  const simp = report?.object.simplified;
  if (simp) {
    out.push({ id: "simplify", label: "Simplified", detail:
      `${simp.faces_before.toLocaleString()} → ${simp.faces_after.toLocaleString()} triangles`,
    origin: params.simplify === null ? "file" : origin(["simplify"], params, defaults, remembered),
    undo: { simplify: null, simplify_auto: false }, control: "simplifyLevel" });
  }
  const limbs = params.limbs?.length ?? 0;
  if (limbs) {
    out.push({ id: "limbs", label: `Reshaped ${limbs === 1 ? "a part" : `${limbs} parts`}`, detail: "size, bend or base width",
      origin: "this figure", undo: { limbs: null }, control: "limbList" });
  }
  const c = report?.fit.conform;
  if (c && c.lift_mm > 0.01) {
    out.push({ id: "bend", label: "Legs/tail bent up", detail: `up to ${mm(c.lift_mm)}${params.bend === null ? " (auto)" : ""}`,
      origin: origin(["bend"], params, defaults, remembered), undo: { bend: 0 }, control: "bendSelect" });
  }
  if (c && c.flat_depth_mm > 0.01) {
    out.push({ id: "press", label: "Base pressed flat", detail: `by up to ${mm(c.flat_depth_mm)}`,
      origin: origin(["flat_depth", "flat_base"], params, defaults, remembered), undo: { flat_depth: 0 }, control: "flatDepth" });
  }
  if (c && c.fill_mm2 > 0.5) {
    out.push({ id: "fill", label: "Flat base added", detail: `${Math.round(c.fill_mm2)} mm² under the figure, up to ${mm(c.fill_mm ?? params.flat_fill)} high`,
      origin: origin(["flat_fill", "fill_area"], params, defaults, remembered), undo: { flat_fill: 0 }, control: "flatFill" });
  }
  if (params.strengthen > 0) {
    out.push({ id: "strengthen", label: "Thin parts thickened", detail: `to ${params.strengthen} mm where they join`,
      origin: "this figure", undo: { strengthen: 0 }, control: "strengthenSelect" });
  }
  const at = params.strengthen_at?.length ?? 0;
  if (at) {
    out.push({ id: "strengthen_at", label: `${at === 1 ? "A weak part" : `${at} weak parts`} thickened`,
      detail: params.strengthen_at!.map((q) => `${q.mm} mm`).join(", "), origin: "this figure",
      undo: { strengthen_at: null }, control: "weakList" });
  }
  return out;
}

/** The changes carried over from the last figure: worth telling the user about as a file opens. */
export function carriedOver(changes: ModelChange[]): ModelChange[] {
  return changes.filter((c) => c.origin === "remembered");
}
