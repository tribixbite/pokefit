/**
 * "Send to printer": the filament -> spool confirm before a print starts, kept in step with the printer.
 *
 * One row per filament of the print file: its colour and material -> the AMS spool it prints from (a slot
 * picker of the spools loaded now, closest colour first) and beambam's verdict for that pair. While the
 * panel is open the printer's spools are read again every few seconds; when they change (a spool loaded,
 * swapped or run out) beambam checks the plan again and the rows update, so what is on screen is what
 * "Start print" sends. beambam's errors (empty tray, other material, a nozzle the tray cannot reach) disable
 * Start print; its warnings (brand, profile, colour) stay visible and pressing Start print accepts them.
 *
 * On a two-nozzle printer each filament also shows the nozzle it was sliced for, and the user can move it to the
 * other one: that is a different print (Bambu's tool order, purge and time all change), so the source slices again
 * (`renozzle`) and the header, time and spool check follow the new print file. With a Filament Track Switch any
 * spool reaches either nozzle; without one beambam's check refuses a spool that cannot reach its filament's nozzle.
 *
 * Back-end neutral: `ConfirmSource` is the local server (`/api/printer/*`) or the beambam daemon the site
 * is connected to (main.ts builds both).
 */
import { textOn } from "./colour";
import { closeSlotPicker, slotPicker, type SlotChoice } from "./slotpicker";
import type { AmsSlot, PrintMapping } from "./types";

export interface ConfirmSource {
  /** beambam's plan with `slots` (one tray per filament, file order), or its own choice for null. */
  plan(slots: number[] | null): Promise<PrintMapping>;
  /** The spools loaded now. */
  trays(): Promise<AmsSlot[]>;
  /** Start the print (the user pressed Start print). Resolves with beambam's last word. */
  start(slots: number[], acceptWarnings: boolean, onTick: (seconds: number, what: string) => void,
    digest?: string): Promise<string>;
  /** How often to read the spools again while the panel is open. */
  pollMs: number;
  /** A line under filament `filament` (1-based; e.g. the preset it was sliced with). */
  note?(filament: number): string;
  /** Slice again with filament i on nozzle nozzles[i - 1] (1 = left, 2 = right), or null for Bambu's own choice;
   *  resolves with the new header (time, weight). Absent: the nozzles cannot be changed here. */
  renozzle?(nozzles: number[] | null, onTick: (seconds: number, what: string) => void): Promise<(string | HTMLElement)[]>;
  /** The nozzles the user chose for the current print file (null: Bambu chose them). */
  nozzles?(): number[] | null;
}

/** The nozzle number (1 = left, 2 = right) beambam's plan names, or null ("-": one nozzle; "both": either). */
export function nozzleNumber(name: string): 1 | 2 | null {
  return name === "left" ? 1 : name === "right" ? 2 : null;
}

/** Every filament's nozzle (index 0 = filament 1) as the plan has it, with filament `filament` moved to `to`.
 *  Filaments the plan does not list (unused) and "both" print on the left, Bambu's default. */
export function nozzlesWith(m: PrintMapping, filament: number, to: 1 | 2): number[] {
  const n = Math.max(filament, ...m.filaments.map((f) => f.filament));
  const out = new Array<number>(n).fill(1);
  for (const f of m.filaments) out[f.filament - 1] = nozzleNumber(f.nozzle) ?? 1;
  out[filament - 1] = to;
  return out;
}

type Filament = PrintMapping["filaments"][number];

/** What makes two readings of the spools different for printing: tray, material, colour, name. */
export function traySignature(trays: AmsSlot[]): string {
  return trays.map((t) => `${t.slot}:${t.type}:${t.colour.toUpperCase()}:${t.name}`).sort().join("|");
}

/** The trays beambam assigned, file order; null when one has none (it cannot print yet). */
export function slotsFromPlan(m: PrintMapping, base: number[] | null = null): number[] | null {
  if (m.filaments.some((f) => typeof f.tray?.tray_id !== "number")) return null;
  // Filament-indexed (index 0 = filament 1): beambam's plan lists the used filaments only, so filaments the
  // plate does not use keep their entry from `base` (the trays the print was sliced for: one per project
  // filament, which the local server requires), else -1. beambam reads either form by filament number.
  const n = Math.max(base?.length ?? 0, ...m.filaments.map((f) => f.filament));
  const out = Array.from({ length: n }, (_, i) => base?.[i] ?? -1);
  for (const f of m.filaments) out[f.filament - 1] = f.tray!.tray_id;
  return out;
}

/** Picker choices: the loaded spools, plus the filament's current tray if it is no longer loaded. */
export function trayChoices(trays: AmsSlot[], f: Filament, current: number | undefined): SlotChoice[] {
  const out: SlotChoice[] = trays.map((t) => ({
    value: t.slot, label: t.label, colour: t.colour, detail: [t.type, t.name].filter(Boolean).join(" · "),
  }));
  if (current === undefined) {
    out.push({ value: -1, label: "No spool", colour: null, detail: "pick the spool to print it from" });
  } else if (!trays.some((t) => t.slot === current)) {
    out.push({ value: current, label: f.tray?.label.replace(/ \(slot \d+\)$/, "") ?? `slot ${current}`,
      colour: null, detail: "not loaded now" });
  }
  return out;
}

/** "12:03:04" */
const clock = (d: Date): string => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function swatch(colour: string, text = ""): HTMLSpanElement {
  const s = el("span", "fig-sw", text);
  s.style.background = colour || "transparent";
  if (/^#[0-9a-f]{6}/i.test(colour)) s.style.color = textOn(colour.slice(0, 7));
  return s;
}

/**
 * Fill `box` with the confirm panel. `header`: what the file is (name, time, weight, beambam's file check).
 * `initial`: the trays the print was sliced for, or null to start from beambam's choice. Returns a stop
 * function (also called by Cancel, and when `box` leaves the page).
 */
export function openPrintConfirm(box: HTMLElement, src: ConfirmSource, initial: number[] | null,
  header: (string | HTMLElement)[], notes: string[] = []): () => void {
  let slots = initial;
  let trays: AmsSlot[] = [];
  let traySig = "";
  let mapping: PrintMapping | null = null;
  let starting = false;
  let reslicing = false;
  let stopped = false;
  let timer: number | null = null;
  let replanning: Promise<void> | null = null;
  /** Spool checks asked for and not answered yet (Start print stays off meanwhile). */
  let pending = 0;
  const retryButton = (): HTMLButtonElement => {
    const b = el("button", "btn", "Check again");
    b.type = "button";
    b.addEventListener("click", () => void replan(""));
    return b;
  };

  const panel = el("div", "print-plan print-confirm");
  const head = el("div", "pc-head");
  head.append(...header.map((x) => (typeof x === "string" ? el("div", "", x) : x)));
  const nozzleInfo = el("div", "pc-nozzles");
  const sync = el("div", "muted pc-sync");
  sync.setAttribute("aria-live", "polite");
  const rows = el("div", "pc-rows");
  rows.setAttribute("role", "list");
  const check = el("div", "print-check");
  const startBtn = el("button", "btn primary", "Start print");
  const cancelBtn = el("button", "btn", "Cancel");
  const status = el("div", "muted");
  status.setAttribute("aria-live", "polite");
  const buttons = el("div", "row");
  buttons.append(startBtn, cancelBtn);
  panel.append(head, nozzleInfo, sync, rows, check,
    el("div", "muted", "Check the plate is clean and the spools are right: this starts the printer."), buttons, status);
  box.replaceChildren(panel);

  const stop = (): void => {
    stopped = true;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    document.removeEventListener("visibilitychange", onVisible);
  };

  const render = (): void => {
    closeSlotPicker();
    const m = mapping;
    if (!m) return;
    rows.replaceChildren(...m.filaments.map((f) => {
      const row = el("div", `slot-row pc-row${f.errors.length ? " bad" : f.warnings.length ? " warn" : ""}`);
      row.setAttribute("role", "listitem");
      const what = el("span", "pc-fil");
      what.append(swatch(f.colour, String(f.filament)),
        el("span", "", `${f.type}${f.nozzle && !(src.renozzle && nozzleNumber(f.nozzle)) ? ` · ${f.nozzle} nozzle` : ""} prints from`));
      const current = f.tray?.tray_id;
      const picker = slotPicker({
        choices: trayChoices(trays, f, current),
        value: current ?? -1,
        figureColour: f.colour.slice(0, 7),
        ariaLabel: `Spool for filament ${f.filament}`,
        onChange: (v) => {
          if (v < 0 || reslicing || starting) return;
          const base = [...(slotsFromPlan(m, slots ?? initial) ?? slots ?? initial ?? [])];
          while (base.length < f.filament) base.push(-1);
          base[f.filament - 1] = v;
          slots = base;
          void replan("");
        },
      });
      const verdict = el("span", "pc-verdict", f.errors.length ? "✗" : f.warnings.length ? "⚠" : "✓");
      verdict.title = [...f.errors, ...f.warnings].join("\n") || "matches";
      // Filament and verdict on one line, the spool picker under it: a phone-width card has no room for
      // all three side by side.
      row.append(what, verdict, picker);
      const lines = el("div", "pc-lines");
      const nz = nozzleNumber(f.nozzle);
      if (src.renozzle && nz !== null) lines.append(nozzleSwitch(m, f.filament, nz));
      const note = src.note?.(f.filament);
      if (note) lines.append(el("div", "muted", note));
      for (const e of f.errors) lines.append(el("div", "err", `✗ ${e}`));
      for (const w of f.warnings) lines.append(el("div", "warn-text", `⚠ ${w}`));
      const wrap = el("div", "pc-item");
      wrap.append(row, lines);
      return wrap;
    }));
    const verdict = m.errors.length ? "beambam will not start this print: fix the ✗ rows (pick another spool, or load one)."
      : m.warnings.length ? "Check (beambam): Start print accepts the ⚠ warnings."
        : "Spool check (beambam): every spool matches.";
    check.replaceChildren(el("div", "", verdict), ...printWarnings(m).map((w) => el("div", "warn-text", `⚠ ${w}`)),
      ...m.notes.map((n) => el("div", "muted", n)));
    startBtn.disabled = starting || reslicing || pending > 0 || m.errors.length > 0 || !slotsFromPlan(m, slots);
    showNozzleInfo();
  };

  /** "Left | Right" for one filament: picking the other nozzle slices the print again. */
  const nozzleSwitch = (m: PrintMapping, filament: number, current: 1 | 2): HTMLElement => {
    const wrap = el("div", "pc-nozzle");
    const seg = el("div", "seg");
    seg.setAttribute("role", "radiogroup");
    seg.setAttribute("aria-label", `Nozzle for filament ${filament}`);
    for (const [n, name] of [[1, "Left"], [2, "Right"]] as const) {
      const b = el("button", "", name);
      b.type = "button";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(n === current));
      b.disabled = reslicing || starting;
      b.addEventListener("click", () => {
        if (n !== current) void renozzle(nozzlesWith(m, filament, n), `filament ${filament} on the ${name.toLowerCase()} nozzle`);
      });
      seg.append(b);
    }
    wrap.append(el("span", "muted", "nozzle"), seg);
    return wrap;
  };

  /** Who chose the nozzles, with a way back to Bambu's choice. */
  const showNozzleInfo = (): void => {
    const twoNozzles = mapping?.filaments.some((f) => nozzleNumber(f.nozzle) !== null) ?? false;
    if (!src.renozzle || !twoNozzles) {
      // e.g. "Update beambam to choose each filament's nozzle here." (an older daemon)
      nozzleInfo.replaceChildren(...(twoNozzles ? notes.map((n) => el("div", "muted", n)) : []));
      return;
    }
    const mine = src.nozzles?.() ?? null;
    if (!mine) {
      nozzleInfo.replaceChildren(el("div", "muted",
        "Nozzles: Bambu's choice (fewest filament changes). Pick Left or Right under a filament to slice it again that way."));
      return;
    }
    const reset = el("button", "btn small", "Let Bambu choose");
    reset.type = "button";
    reset.disabled = reslicing || starting;
    reset.addEventListener("click", () => void renozzle(null, "Bambu's choice of nozzles"));
    nozzleInfo.replaceChildren(el("div", "muted", "Nozzles: your choice (the time above is for it)."), reset);
  };

  /** Slice again for other nozzles; the header and the spool check follow the new print file. */
  const renozzle = async (nozzles: number[] | null, what: string): Promise<void> => {
    if (!src.renozzle || reslicing || starting) return;
    reslicing = true;
    closeSlotPicker();
    render();
    status.textContent = `Slicing again with ${what}…`;
    try {
      const next = await src.renozzle(nozzles, (sec, step) => {
        status.textContent = `Slicing again with ${what}: ${step || "slicing"}… ${Math.round(sec)} s`;
      });
      if (stopped) return;
      head.replaceChildren(...next.map((x) => (typeof x === "string" ? el("div", "", x) : x)));
      reslicing = false;
      await replan("");
      // In the status line, not the spool line: the next spool read (every few seconds) rewrites that one.
      if (!stopped) status.textContent = `Sliced again with ${what} at ${clock(new Date())}: the time above is for it.`;
    } catch (e) {
      if (stopped) return;
      reslicing = false;
      status.textContent = `Could not slice again: ${(e as Error).message}`;
      render();
    }
  };

  const replan = (reason: string): Promise<void> => {
    // Until this check answers, Start print would send the previous spools: keep it off and say so.
    pending++;
    startBtn.disabled = true;
    check.replaceChildren(el("div", "muted", "Checking the spools with beambam…"));
    const run = async (): Promise<void> => {
      try {
        mapping = await src.plan(slots);
        if (stopped) return;
        slots = slotsFromPlan(mapping, slots ?? initial) ?? slots;
        pending--;
        render();
        if (reason) sync.textContent = reason;
      } catch (e) {
        pending--;
        if (stopped) return;
        // Keep the last plan's rows on screen and offer a retry (a one-off failure must not kill the panel).
        check.replaceChildren(el("div", "err", `Spool check failed: ${(e as Error).message}`),
          retryButton());
        startBtn.disabled = true;
      }
    };
    replanning = (replanning ?? Promise.resolve()).then(run);
    return replanning;
  };

  /** Read the spools; re-check the plan when they changed. Runs every `pollMs` while visible. */
  const tick = async (): Promise<void> => {
    timer = null;
    if (stopped) return;
    if (!box.isConnected || !box.contains(panel)) { stop(); return; }
    // An open spool list or a print being sent: read again later rather than redraw under the finger.
    if (!document.hidden && !starting && !reslicing && !document.querySelector(".slot-list")) {
      try {
        const now = await src.trays();
        if (stopped) return;
        const sig = traySignature(now);
        const changed = traySig !== "" && sig !== traySig;
        trays = now;
        traySig = sig;
        if (changed) await replan(`The printer's spools changed at ${clock(new Date())}: checked again.`);
        else sync.textContent = `Spools read from the printer at ${clock(new Date())} (every ${Math.round(src.pollMs / 1000)} s).`;
      } catch (e) {
        if (!stopped) sync.textContent = `Could not read the printer's spools: ${(e as Error).message}`;
      }
    }
    if (!stopped && !document.hidden) timer = window.setTimeout(() => void tick(), src.pollMs);
  };
  const onVisible = (): void => {
    if (!document.hidden && timer === null && !stopped) void tick();
  };
  document.addEventListener("visibilitychange", onVisible);

  startBtn.addEventListener("click", async () => {
    const m = mapping;
    const s = m ? slotsFromPlan(m, slots ?? initial) : null;
    if (!m || !s || m.errors.length || reslicing || pending > 0) return;
    starting = true;
    startBtn.disabled = true;
    cancelBtn.disabled = true;
    status.textContent = "Sending to the printer…";
    try {
      // The warnings on screen when Start print was pressed are the user's OK; beambam checks the spools again.
      const said = await src.start(s, m.warnings.length > 0, (sec, what) => {
        status.textContent = `${what || "Sending"}… ${Math.round(sec)} s`;
      }, m.digest);
      const last = said.split("\n").filter(Boolean).slice(-1)[0] ?? "";
      status.textContent = last ? `Started: ${last}.` : "Started.";
      stop();
    } catch (e) {
      status.textContent = (e as Error).message;
      starting = false;
      cancelBtn.disabled = false;
      render();
      if (timer === null && !stopped) void tick();
    }
  });
  cancelBtn.addEventListener("click", () => {
    stop();
    closeSlotPicker();
    box.replaceChildren();
  });

  sync.textContent = "Reading the printer's spools…";
  startBtn.disabled = true;
  void (async () => {
    try {
      trays = await src.trays();
      traySig = traySignature(trays);
      sync.textContent = `Spools read from the printer at ${clock(new Date())} (every ${Math.round(src.pollMs / 1000)} s).`;
    } catch (e) {
      sync.textContent = `Could not read the printer's spools: ${(e as Error).message}`;
    }
    await replan("");
    if (!stopped) timer = window.setTimeout(() => void tick(), src.pollMs);
  })();
  return stop;
}

/**
 * The plan's warnings about the print as a whole (the plate on the printer is not the one the file was sliced for),
 * not shown on a filament's row: beambam's flat list names a filament's own as "filament N: …".
 */
export function printWarnings(m: PrintMapping): string[] {
  return m.warnings.filter((w) => !/^filament \d+: /.test(w));
}
