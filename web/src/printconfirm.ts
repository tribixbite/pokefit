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
  start(slots: number[], acceptWarnings: boolean, onTick: (seconds: number, what: string) => void): Promise<string>;
  /** How often to read the spools again while the panel is open. */
  pollMs: number;
  /** A line under a filament (e.g. the preset it was sliced with). */
  note?(index: number): string;
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
export function slotsFromPlan(m: PrintMapping): number[] | null {
  const ids = m.filaments.map((f) => f.tray?.tray_id);
  return ids.every((x): x is number => typeof x === "number") ? ids : null;
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
  header: (string | HTMLElement)[]): () => void {
  let slots = initial;
  let trays: AmsSlot[] = [];
  let traySig = "";
  let mapping: PrintMapping | null = null;
  let starting = false;
  let reslicing = false;
  let stopped = false;
  let timer: number | null = null;
  let replanning: Promise<void> | null = null;

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
    rows.replaceChildren(...m.filaments.map((f, i) => {
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
          if (v < 0 || reslicing) return;
          const base = slots ?? slotsFromPlan(m) ?? m.filaments.map((x) => x.tray?.tray_id ?? -1);
          slots = base.map((s, k) => (k === i ? v : s));
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
      const note = src.note?.(i);
      if (note) lines.append(el("div", "muted", note));
      for (const e of f.errors) lines.append(el("div", "err", `✗ ${e}`));
      for (const w of f.warnings) lines.append(el("div", "warn-text", `⚠ ${w}`));
      const wrap = el("div", "pc-item");
      wrap.append(row, lines);
      return wrap;
    }));
    const verdict = m.errors.length ? "beambam will not start this print: fix the ✗ rows (pick another spool, or load one)."
      : m.warnings.length ? "Spool check (beambam): Start print accepts the ⚠ differences."
        : "Spool check (beambam): every spool matches.";
    check.replaceChildren(el("div", "", verdict), ...m.notes.map((n) => el("div", "muted", n)));
    startBtn.disabled = starting || reslicing || m.errors.length > 0 || !slotsFromPlan(m);
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
    if (!src.renozzle || !mapping?.filaments.some((f) => nozzleNumber(f.nozzle) !== null)) {
      nozzleInfo.replaceChildren();
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
    const run = async (): Promise<void> => {
      try {
        mapping = await src.plan(slots);
        if (stopped) return;
        slots = slotsFromPlan(mapping) ?? slots;
        render();
        if (reason) sync.textContent = reason;
      } catch (e) {
        if (stopped) return;
        mapping = null;
        rows.replaceChildren(el("div", "err", (e as Error).message));
        check.replaceChildren();
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
    const s = m ? slotsFromPlan(m) : null;
    if (!m || !s || m.errors.length || reslicing) return;
    starting = true;
    startBtn.disabled = true;
    cancelBtn.disabled = true;
    status.textContent = "Sending to the printer…";
    try {
      // The warnings on screen when Start print was pressed are the user's OK; beambam checks the spools again.
      const said = await src.start(s, m.warnings.length > 0, (sec, what) => {
        status.textContent = `${what || "Sending"}… ${Math.round(sec)} s`;
      });
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
