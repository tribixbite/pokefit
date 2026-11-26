/**
 * "Your project and your printer differ" (Print card): once the printer's spools are read (beambam on the site,
 * "Use my printer's AMS" locally), compare the project's filament list, slot by slot, with the AMS trays in order
 * (the engine's projectsync.compare) and offer what Bambu Studio offers after it connects to a printer: take the
 * printer's filaments as the project's (its slices and exports use them; slots past the spools stay), and
 * download the updated project to keep. The plate the project prints on is compared with the plate on the
 * printer when its marker is a known one (printer.PLATE_MARKERS), else the marker is shown; Print settings → Plate
 * changes it, and taking the printer's filaments takes its known plate too.
 */
import { textOn } from "./colour";
import type { Config, ProjectSync } from "./types";

/** "SuperTack plate", "Textured PEI plate": a Bambu plate name in its maker's words. */
export function plateText(name: string): string {
  return name.replace(/^Supertack/, "SuperTack").replace(/ Plate$/, " plate");
}

/** The plate line: the project's plate against the printer's (named when known, else its marker). */
export function plateLine(project: string | null, printerName: string | null | undefined, marker: string | null | undefined): { text: string; differs: boolean } {
  const differs = !!project && !!printerName && project !== printerName;
  const head = project ? `The project prints on the ${plateText(project)}` : "The project names no plate";
  const tail = printerName ? `; the printer has the ${plateText(printerName)} on`
    : marker ? `; the printer reports plate marker ${marker} (not known yet)` : "";
  return { text: `${head}${tail}.${differs || !printerName ? " Change it under Print settings → Plate if that is not the plate on the printer." : ""}`, differs };
}

export interface ProjectSyncHost {
  /** Take the printer's filaments (and the plate chosen in Print settings, if any) as the project's. */
  adopt(): Promise<void>;
  /** Download the project as now in use. */
  download(): Promise<void>;
}

/** "3 of 16 slots already match the printer" and friends: the one-line verdict. */
export function syncSummary(s: ProjectSync): string {
  if (!s.differs) return `Your project's first ${s.printer} filaments match the printer's spools.`;
  const rest = s.project - s.printer;
  const extra = rest === 1 ? " (its last slot stays as it is)" : rest > 1 ? ` (its other ${rest} slots stay as they are)` : "";
  return `Your project's filaments differ from the printer's: ${s.same} of ${s.printer} slots match${extra}.`;
}

function swatch(colour: string): HTMLSpanElement {
  const e = document.createElement("span");
  e.className = "sw";
  e.style.background = colour || "transparent";
  if (/^#[0-9a-f]{6}/i.test(colour)) e.style.color = textOn(colour.slice(0, 7));
  return e;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

/** Fill `box` from the config; empty (hidden) without both a project and the printer's spools. */
export function renderProjectSync(box: HTMLElement, cfg: Config, host: ProjectSyncHost): void {
  const s = cfg.project_sync;
  box.hidden = !s;
  if (!s) {
    box.replaceChildren();
    return;
  }
  const head = el("div", s.differs ? "warn-text" : "muted", syncSummary(s));
  const parts: HTMLElement[] = [head];
  if (s.differs) {
    const list = el("details", "sync-rows");
    list.append(el("summary", "", "Slot by slot"));
    for (const r of s.rows.slice(0, Math.max(s.printer, 1))) {
      const row = el("div", `sync-row${r.same ? "" : " differs"}`);
      row.append(el("span", "muted", String(r.index)));
      if (r.project) row.append(swatch(r.project.colour), el("span", "", `${r.project.type} · ${r.project.name}`));
      else row.append(el("span", "muted", "no slot"));
      row.append(el("span", "muted", r.same ? "=" : "→"));
      if (r.spool) row.append(swatch(r.spool.colour), el("span", "", `${r.spool.label} ${r.spool.type}${r.spool.name ? ` · ${r.spool.name}` : ""}`));
      list.append(row);
    }
    parts.push(list);
  }
  const plate = plateLine(s.plate, cfg.printer_plate_name, cfg.printer_plate);
  parts.push(el("div", plate.differs ? "warn-text" : "muted", plate.text));
  const buttons = el("div", "row");
  if (s.differs || plate.differs) {
    const use = el("button", "btn", s.differs ? "Use the printer's filaments" : `Use the ${plateText(cfg.printer_plate_name ?? "")}`);
    use.type = "button";
    use.title = (s.differs ? "The project's filament i becomes the printer's tray i (A1, A2, …): its material, preset and colour"
      : "The project prints on the plate that is on the printer") +
      (cfg.printer_plate_name ? `; the plate becomes the ${plateText(cfg.printer_plate_name)}` : "");
    use.addEventListener("click", () => void run(use, host.adopt));
    buttons.append(use);
  }
  const dl = el("button", "btn", "Download the updated project");
  dl.type = "button";
  dl.title = "Your project file with the filament list and plate now in use (everything else as it was)";
  dl.addEventListener("click", () => void run(dl, host.download));
  buttons.append(dl);
  parts.push(buttons);
  box.replaceChildren(...parts);
}

/** Disable `btn` while `fn` runs (it reports its own errors). */
async function run(btn: HTMLButtonElement, fn: () => Promise<void>): Promise<void> {
  btn.disabled = true;
  try {
    await fn();
  } finally {
    btn.disabled = false;
  }
}
