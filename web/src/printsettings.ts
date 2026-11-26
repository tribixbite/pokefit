/**
 * Print settings in the Print card: supports, layer height, walls, infill and brim (pipeline `Params.supports` …
 * `brim`, applied by `print_overrides` to every slice and export). "project's" (null) keeps the reference project's
 * value, shown beside it when the engine reported it (`Config.print_defaults`).
 */
import type { BrimType, Config, Params, SupportMode } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

export interface PrintSettingsHost {
  params(): Params;
  config(): Config | null;
  /** A setting changed: `reanalyse` when the checks read it (the layer height: islands, colour changes). */
  changed(reanalyse: boolean): void;
}

const BRIM_NAMES: Record<string, string> = { auto_brim: "auto", outer_only: "outside only", no_brim: "none" };
const SUPPORT_NAMES: Record<SupportMode, string> = { auto: "supports where needed", plate: "supports from the plate", off: "no supports" };

/** "supports where needed · 0.16 mm · 2 walls · 15 % infill" for the closed section's summary. */
export function settingsSummary(p: Params, cfg: Config | null): string {
  const d = cfg?.print_defaults;
  const lh = p.layer_height ?? d?.layer_height;
  const walls = p.wall_loops ?? d?.wall_loops;
  const infill = p.infill ?? d?.infill;
  const brim = p.brim ?? d?.brim;
  return [
    SUPPORT_NAMES[p.supports ?? "auto"],
    lh != null ? `${lh} mm` : "",
    walls != null ? `${walls} wall${walls === 1 ? "" : "s"}` : "",
    infill != null ? `${infill} % infill` : "",
    brim ? `brim ${BRIM_NAMES[brim] ?? brim}` : "",
  ].filter(Boolean).join(" · ");
}

/** Label each select's "project's" option with the project's value. */
function labelDefaults(cfg: Config | null): void {
  const d = cfg?.print_defaults;
  const set = (id: string, v: string | null): void => {
    const o = $<HTMLSelectElement>(id).querySelector<HTMLOptionElement>('option[value=""]');
    if (o) o.textContent = v ? `project's (${v})` : "project's";
  };
  set("layerSelect", d?.layer_height != null ? `${d.layer_height} mm` : null);
  set("wallSelect", d?.wall_loops != null ? String(d.wall_loops) : null);
  set("infillSelect", d?.infill != null ? `${d.infill} %` : null);
  set("brimSelect", d?.brim ? (BRIM_NAMES[d.brim] ?? d.brim) : null);
  // Bambu refuses a line no wider than the layer: the engine widens those lines (pipeline `thin_lines`); say so.
  const thin = d?.min_line_width ?? null;
  for (const o of $<HTMLSelectElement>("layerSelect").options) {
    if (o.value === "") continue;
    const v = +o.value;
    o.textContent = thin !== null && v >= thin ? `${o.value} mm *` : `${o.value} mm`;
  }
}

/** Under the controls: what a layer as tall as the project's narrowest line does to those lines. */
function layerNote(p: Params, cfg: Config | null): void {
  const thin = cfg?.print_defaults?.min_line_width ?? null;
  const lh = p.layer_height;
  const note = $("layerNote");
  note.hidden = !(thin !== null && lh != null && lh >= thin);
  if (!note.hidden) {
    note.textContent = `* At ${lh} mm, the project's ${thin} mm lines print ${(lh! + 0.02).toFixed(2)} mm wide: ` +
      "Bambu needs every line wider than the layer.";
  }
}

/** A select's value for a number param: "" for null, else the option whose number equals it. */
function numberValue(sel: HTMLSelectElement, v: number | null | undefined): string {
  if (v == null) return "";
  const o = [...sel.options].find((x) => x.value !== "" && +x.value === v);
  return o ? o.value : "";
}

/** Show `p` in the controls. */
export function showPrintSettings(p: Params, cfg: Config | null): void {
  labelDefaults(cfg);
  $<HTMLSelectElement>("supportSelect").value = p.supports ?? "auto";
  for (const [id, v] of [["layerSelect", p.layer_height], ["wallSelect", p.wall_loops], ["infillSelect", p.infill]] as const) {
    const sel = $<HTMLSelectElement>(id);
    sel.value = numberValue(sel, v);
  }
  $<HTMLSelectElement>("brimSelect").value = p.brim ?? "";
  $("printSettingsSum").textContent = settingsSummary(p, cfg);
  layerNote(p, cfg);
}

/** Wire the controls to the params. */
export function bindPrintSettings(host: PrintSettingsHost): void {
  const on = (id: string, apply: (value: string) => boolean): void => {
    $<HTMLSelectElement>(id).addEventListener("change", (e) => {
      const reanalyse = apply((e.target as HTMLSelectElement).value);
      $("printSettingsSum").textContent = settingsSummary(host.params(), host.config());
      layerNote(host.params(), host.config());
      host.changed(reanalyse);
    });
  };
  const num = (v: string): number | null => (v === "" ? null : +v);
  on("supportSelect", (v) => { host.params().supports = v as SupportMode; return false; });
  on("layerSelect", (v) => { host.params().layer_height = num(v); return true; });
  on("wallSelect", (v) => { host.params().wall_loops = num(v); return false; });
  on("infillSelect", (v) => { host.params().infill = num(v); return false; });
  on("brimSelect", (v) => { host.params().brim = v === "" ? null : v as BrimType; return false; });
}
