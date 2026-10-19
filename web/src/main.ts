/** pokefit web UI: wires the panel controls, the 3D view and the underside editor to the API. */
import * as api from "./api";
import { ColorizePanel } from "./colorize";
import { formatDuration, pct } from "./geometry";
import { Viewer } from "./scene";
import type { Check, Config, Params, Report, SliceResult, SliceVariants, WasherRow } from "./types";
import { UndersideEditor } from "./underside";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const e = document.getElementById(id);
  if (!e) throw new Error(`#${id} missing`);
  return e as T;
};

interface State {
  config: Config | null;
  session: string | null;
  params: Params;
  report: Report | null;
  focus: Check | null;
  seq: number;
}

const state: State = {
  config: null,
  session: null,
  params: {
    ball_scale: 0.6, ball_diameter: null, washer: "M6", clearance: 1, figure_scale: null,
    rotate_x: 0, rotate_y: 0, rotate_z: 0, magnets: 2,
    pockets: null, pocket_diameter: 4.18, pocket_depth: 2.2, pocket_wall: 0.8, pocket_ceiling: 0.6,
    pocket_grip: 1.2, pocket_max_seat: 2.0, pocket_seat: "washer", object_id: null, slot: null, settings_source: "reference",
    filament_map: null, colours: null, paint: null, texture_colours: null,
  },
  report: null,
  focus: null,
  seq: 0,
};

const viewer = new Viewer($("viewport"));
const underside = new UndersideEditor($("underside"), (pockets) => {
  state.params.pockets = pockets;
  void runAnalyse();
});
/** Paint/colours changed since the last analysis (its import-file colour map is stale). */
let paintDirty = false;
const colorize = new ColorizePanel({
  params: () => state.params,
  report: () => state.report,
  // Paint changes colours only: the figure keeps its place, so show it again at once.
  refresh: async (regions) => {
    await refreshFigure(regions);
    if (state.report) viewer.update(state.report, state.focus);
  },
  pick: (hit, mode, angle, radius) =>
    api.pick(state.session ?? "", state.params, { point: hit.point, normal: hit.normal, mode, angle, radius }),
  setPicking: (fn) => viewer.setPickHandler(fn),
  changed: () => { paintDirty = true; },
  settled: () => { if (paintDirty) void runAnalyse(); },
  toast: (msg) => toast(msg),
});

// ------------------------------------------------------------------ helpers

function toast(msg: string): void {
  console.warn("[pokefit]", msg);
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  window.clearTimeout((t as HTMLElement & { _h?: number })._h);
  (t as HTMLElement & { _h?: number })._h = window.setTimeout(() => (t.hidden = true), 6000);
}

function busy(on: boolean, text = "Analysing…"): void {
  $("busy").hidden = !on;
  $("busyText").textContent = text;
}

function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { cls?: string } = {},
  ...kids: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  const { cls, ...rest } = props;
  if (cls) e.className = cls;
  Object.assign(e, rest);
  e.append(...kids);
  return e;
}

// ------------------------------------------------------------------ loading

async function init(): Promise<void> {
  api.onEngineStatus(engineStatus);
  try {
    state.config = await api.getConfig();
  } catch (e) {
    toast(`Engine not reachable: ${(e as Error).message}`);
    return;
  }
  const cfg = state.config;
  if (cfg.engine === "local") engineStatus("local");
  state.params.washer = cfg.default_washer;
  state.params.slot = cfg.default_slot;
  const ballSel = $<HTMLSelectElement>("ballSelect");
  for (const s of cfg.ball_scales) {
    const d = 2 * cfg.ball_r_inner * s;
    ballSel.append(h("option", { value: String(s), textContent: `${Math.round(s * 100)} % · Ø ${d.toFixed(1)} mm` }));
  }
  ballSel.append(h("option", { value: "custom", textContent: "Inner diameter…" }));
  const ws = $<HTMLSelectElement>("washerSelect");
  for (const w of cfg.washers) ws.append(h("option", { value: w.key, textContent: w.label }));
  ws.append(h("option", { value: "custom", textContent: "Custom size…" }));
  ws.value = cfg.default_washer;
  const ss = $<HTMLSelectElement>("slotSelect");
  // One entry per distinct filament profile (the reference project lists AMS slots).
  const seen = new Set<string>();
  for (const s of cfg.slots) {
    if (seen.has(s.name)) continue;
    seen.add(s.name);
    ss.append(h("option", { value: s.slot, textContent: s.name }));
  }
  ss.value = String(cfg.default_slot);
  renderPrinter(cfg);

  const path = new URLSearchParams(location.search).get("path");
  if (path) await load(() => api.loadPath(path), path.split(/[\\/]/).pop() ?? path);
}

/** Print card: Bambu slicing needs the desktop slicer, so it only exists in the local version. */
function renderPrinter(cfg: Config): void {
  $("printerInfo").textContent = cfg.engine === "browser"
    ? "Print time and filament come from Bambu Studio itself, which only runs on your computer: install pokefit " +
      "locally (see About) to slice. Everything else runs here in your browser."
    : cfg.slicer
      ? `Sliced with your ${cfg.printer} / ${cfg.process} settings (from ${cfg.reference}).`
      : "Bambu Studio not found — slicing unavailable; geometry checks still run.";
  ($("sliceBtn") as HTMLButtonElement).disabled = !cfg.slicer;
  // Slicing controls only where a slicer exists (the local version).
  $("slotRow").hidden = $("sliceRow").hidden = cfg.engine === "browser";
  $("engineWhere").textContent = cfg.engine === "browser" ? "in your browser" : "local";
  // The bundled default project is a JSON profile: name it as such rather than by file name.
  $("refInfo").textContent = !cfg.reference ? "" : cfg.reference.endsWith(".json")
    ? "default slots — load your project to match its filaments" : `slots from ${cfg.reference}`;
  $("ballModelRow").hidden = cfg.engine !== "browser";
}

/** Header chip: where the engine runs and, in the browser, its start-up progress. */
function engineStatus(text: string): void {
  const chip = $("engineChip");
  chip.hidden = false;
  /** Long label for wide screens, short one for phones (CSS picks). */
  const label = (long: string, short: string): void =>
    chip.replaceChildren(h("span", { cls: "long", textContent: long }), h("span", { cls: "short", textContent: short }));
  if (text === "local") {
    chip.className = "engine-chip ready";
    label("local engine · Bambu slicing", "local");
    return;
  }
  const failed = text.startsWith("failed");
  chip.className = `engine-chip ${text === "ready" ? "ready" : failed ? "failed" : "loading"}`;
  if (text === "ready") label("runs in your browser", "ready");
  else if (failed) label("engine failed to start", "failed");
  else label("starting engine…", "starting…");
  chip.title = text === "ready" ? "Your files never leave this device." : text;
  if (!$("busy").hidden && text !== "ready" && !failed) $("busyText").textContent = text;
  if (failed) toast(`The engine could not start (${text.slice(8)}). Reload to retry.`);
}

async function load(fn: () => Promise<import("./types").LoadResponse>, name: string): Promise<void> {
  busy(true, `Loading ${name}…`);
  try {
    const res = await fn();
    state.session = res.session;
    colorize.reset();
    state.params = { ...state.params, figure_scale: null, pockets: null, rotate_x: 0, rotate_y: 0, rotate_z: 0,
      object_id: null, filament_map: null, colours: null, paint: null, texture_colours: null };
    $("fileName").textContent = res.file;
    $("drop").classList.add("hidden");
    $("sliceOut").replaceChildren();
    $("exportOut").replaceChildren();
    $("washerTable").replaceChildren();
    // Initial analysis used server defaults; re-run if the UI's settings differ.
    const r = res.report;
    const differs = r.params.washer !== state.params.washer || r.params.ball_scale !== state.params.ball_scale ||
      r.params.ball_diameter !== state.params.ball_diameter ||
      r.params.clearance !== state.params.clearance || r.params.magnets !== state.params.magnets;
    await viewer.setFigure(await api.previewSource(res.session, state.params));
    if (differs) await runAnalyse();
    else show(r);
    viewer.view("iso");
  } catch (e) {
    toast((e as Error).message);
  } finally {
    busy(false);
  }
}

// ------------------------------------------------------------------ analysis

/**
 * Reload the 3D preview for the current object/orientation/paint (seconds in the browser engine).
 * `regions` colours it by colorize region instead of by filament.
 */
async function refreshFigure(regions: import("./types").RegionView | null = null): Promise<void> {
  if (!state.session) return;
  busy(true, "Preparing the preview…");
  try {
    await viewer.setFigure(await api.previewSource(state.session, state.params, regions));
  } catch (e) {
    toast((e as Error).message);
  } finally {
    busy(false);
  }
}

async function runAnalyse(): Promise<void> {
  if (!state.session) return;
  const seq = ++state.seq;
  paintDirty = false;
  busy(true);
  try {
    const rep = await api.analyse(state.session, state.params);
    if (seq !== state.seq) return;          // a newer request superseded this one
    show(rep);
  } catch (e) {
    if (seq === state.seq) toast((e as Error).message);
  } finally {
    if (seq === state.seq) busy(false);
  }
}

function show(rep: Report): void {
  state.report = rep;
  state.focus = null;
  const f = rep.fit;
  const fitCheck = rep.checks.find((c) => c.id === "fit");

  $("sumHeight").textContent = f.scale > 0 ? f.height_mm.toFixed(1) : "–";
  $("sumScale").textContent = f.scale > 0
    ? `${pct(f.scale)} of the file · max ${pct(f.max_scale)} (${f.max_height_mm.toFixed(1)} mm)`
    : "does not fit";
  $("sumLimit").textContent = f.limit_kind ? `Tightest at the ${limitText(f.limit_kind)}` : "";
  renderStatus(rep);

  const pill = $("fitPill");
  pill.hidden = false;
  pill.className = `fit-pill ${fitCheck?.status ?? "fail"}`;
  const spare = f.margin ?? 0;
  pill.textContent = fitCheck?.status !== "pass" ? "Does not fit"
    : spare < 0.05 ? `Fits · touches the ${limitText(f.limit_kind ?? "")} clearance`
      : `Fits · ${spare.toFixed(1)} mm to spare`;

  // Slider in % of the file's size, up to 125 % of the max, with a marker at the max.
  const slider = $<HTMLInputElement>("scaleSlider");
  const maxPct = f.max_scale * 100;
  slider.min = "5";
  slider.max = String(Math.max(10, Math.ceil(maxPct * 1.25)));
  slider.value = String((f.scale || f.max_scale) * 100);
  const mark = $("maxMark");
  const frac = (maxPct - +slider.min) / (+slider.max - +slider.min);
  mark.style.left = `calc(${(frac * 100).toFixed(2)}% + ${(8 - frac * 16).toFixed(1)}px)`;
  $<HTMLInputElement>("heightInput").value = f.height_mm.toFixed(1);

  const preset = !rep.params.ball_diameter && (state.config?.ball_scales.includes(rep.params.ball_scale) ?? false);
  $<HTMLSelectElement>("ballSelect").value = preset ? String(rep.params.ball_scale) : "custom";
  $("ballCustomRow").hidden = preset;
  if (!preset) $<HTMLInputElement>("ballCustom").value = rep.ball.inner_diameter.toFixed(1);
  $<HTMLInputElement>("rotX").value = String(rep.params.rotate_x);
  $<HTMLInputElement>("rotY").value = String(rep.params.rotate_y);
  $<HTMLInputElement>("rotZ").value = String(rep.params.rotate_z);
  $("rotLabel").textContent = "";

  renderObjects(rep);
  renderChecks(rep);
  renderMagnets(rep);
  renderImportMap(rep);
  colorize.render();
  $("timing").textContent = `analysed in ${rep.timings.total?.toFixed(1)} s`;
  viewer.update(rep);
}

function renderStatus(rep: Report): void {
  const chip = $("sumStatus");
  chip.className = `status-chip ${rep.status}`;
  chip.textContent = { pass: "ready", warn: "check warnings", fail: "problems", info: "info" }[rep.status];
}

function limitText(kind: string): string {
  return ({ ring: "connector ring", shell: "shell wall", top: "top of the ball", aperture: "ring opening" } as Record<string, string>)[kind] ?? kind;
}

function renderObjects(rep: Report): void {
  const o = rep.object;
  $("objectCard").hidden = o.objects.length < 2;
  $("objectCount").textContent = `${o.objects.length} figures in this file — pick one`;
  const sel = $<HTMLSelectElement>("objectSelect");
  sel.replaceChildren(...o.objects.map((x) => h("option", { value: x.id, textContent: `${x.name} (${x.faces.toLocaleString()} faces)` })));
  sel.value = o.id;
  $("partsList").replaceChildren(...o.parts.map((p) => {
    const col = o.colours[(p.extruder - 1 + o.colours.length) % o.colours.length] ?? "#888";
    const sw = h("span", { cls: "sw" });
    sw.style.background = col;
    return h("div", {}, sw, `${p.name} — ${p.faces.toLocaleString()} faces${p.painted ? `, ${p.painted} painted` : ""}`);
  }));
}

/** Pocket fields of the "Pocket size and limits" form, by element id. */
const POCKET_INPUTS: [string, "pocket_diameter" | "pocket_depth" | "pocket_wall" | "pocket_ceiling" | "pocket_grip" | "pocket_max_seat"][] = [
  ["pkD", "pocket_diameter"], ["pkDepth", "pocket_depth"], ["pkWall", "pocket_wall"],
  ["pkCeil", "pocket_ceiling"], ["pkGrip", "pocket_grip"], ["pkMaxSeat", "pocket_max_seat"],
];

function renderMagnets(rep: Report): void {
  const n = rep.params.magnets;
  document.querySelectorAll<HTMLButtonElement>("#magnetCount button").forEach((b) =>
    b.setAttribute("aria-checked", String(+(b.dataset.n ?? -1) === n)));
  for (const [id, key] of POCKET_INPUTS) $<HTMLInputElement>(id).value = String(rep.params[key]);
  const raised = rep.params.pocket_seat === "raised";
  document.querySelectorAll<HTMLButtonElement>("#seatMode button").forEach((b) =>
    b.setAttribute("aria-checked", String(b.dataset.seat === rep.params.pocket_seat)));
  $("seatHint").textContent = raised
    ? "Under a part that floats above the bed, the pocket opening moves up into it: the magnet is hidden but sits above the steel, so it holds less."
    : "Every magnet rests on the washer. Under a part that floats above the bed, the pocket only goes as far into it as the magnet reaches, and the magnet shows below it.";
  $<HTMLInputElement>("pkMaxSeat").disabled = !raised;
  $("maxSeatLabel").classList.toggle("off", !raised);
  for (const e of [$("seatMode").closest<HTMLElement>(".seat-row"), $("seatHint")]) if (e) e.hidden = n === 0;
  $("underside").hidden = n === 0;
  underside.render(rep.underside, rep.pockets);
  $("pocketInfo").textContent = !n ? "no magnets" : rep.params.pockets ? "placed by hand"
    : rep.pockets.length ? "automatic" : "";
  $("autoPockets").hidden = n === 0;
  const under = rep.checks.find((x) => x.id === "pockets:feet");
  if (n && !rep.pockets.length && under) {
    // Nothing placed: say why, and offer the one fix that is a single number.
    const items: HTMLElement[] = [h("li", { cls: under.status }, h("span", { textContent: under.message }))];
    const low = typeof under.value === "number" ? under.value : NaN;
    if (!raised && /Raised into the figure/.test(under.message)) {
      const b = h("button", { cls: "btn", textContent: "Raise the pockets into the figure" });
      b.addEventListener("click", () => {
        state.params.pocket_seat = "raised";
        state.params.pocket_max_seat = Math.max(state.params.pocket_max_seat, low);
        void runAnalyse();
      });
      items.push(h("li", {}, b));
    }
    if (raised && /above the washer everywhere/.test(under.message) && low > rep.params.pocket_max_seat) {
      const allow = low;                    // already rounded up to 0.1 mm by the server
      const b = h("button", { cls: "btn", textContent: `Allow pockets up to ${allow.toFixed(1)} mm high` });
      b.addEventListener("click", () => { state.params.pocket_max_seat = allow; void runAnalyse(); });
      items.push(h("li", {}, b));
    }
    $("pocketList").replaceChildren(...items);
    return;
  }
  // One line per pocket: how high its opening is and its check status.
  $("pocketList").replaceChildren(...rep.pockets.map((p, i) => {
    const c = rep.checks.find((x) => x.id === `pocket:${i + 1}`);
    let seat = p.z < 0.05 ? "on the washer" : `${p.z.toFixed(1)} mm above the washer`;
    const shows = c?.message.match(/shows up to ([\d.]+) mm/);
    const short = c?.message.match(/Only ([\d.]+) mm of the pocket is walled/);
    if (short) seat += ` · only ${short[1]} mm walled, glue it`;
    else if (shows) seat += ` · shows ${shows[1]} mm`;
    return h("li", { cls: c?.status ?? "" }, h("span", { cls: "n", textContent: String(i + 1) }),
      h("span", { textContent: c?.status === "fail" ? c.message : seat }));
  }));
}

/** Figure colour -> project slot table; each row's picker overrides the closest-colour choice. */
function renderImportMap(rep: Report): void {
  const slots = state.config?.slots ?? [];
  $("importMapBox").hidden = rep.import_map.length === 0;
  const swatch = (c: string): HTMLElement => {
    const sw = h("span", { cls: "sw" });
    sw.style.background = c || "transparent";
    return sw;
  };
  $("importMap").replaceChildren(
    h("thead", {}, h("tr", {}, ...["figure", "→ your slot", ""].map((t) => h("th", { textContent: t })))),
    h("tbody", {}, ...rep.import_map.map((m) => {
      const sel = h("select");
      sel.setAttribute("aria-label", `Slot for figure colour ${m.source}`);
      slots.forEach((s, i) => sel.append(h("option", { value: String(i + 1), textContent: `${i + 1} · ${s.type} ${s.colour}`, title: s.name })));
      const extra = Math.max(slots.length + 1, m.new ? m.target : 0);
      sel.append(h("option", { value: String(extra), textContent: `${extra} · new slot (set its colour after import)` }));
      sel.value = String(m.target);
      sel.addEventListener("change", () => {
        state.params.filament_map = { ...(state.params.filament_map ?? {}), [String(m.source)]: +sel.value };
        void runAnalyse();
      });
      const note = m.new ? "no close colour in your project" : m.delta_e > 10 ? `colour differs (ΔE ${m.delta_e.toFixed(0)})` : "";
      return h("tr", {}, h("td", {}, swatch(m.colour), `${m.source}`),
        h("td", {}, swatch(m.target_colour), sel), h("td", { cls: "muted", textContent: note }));
    })));
}

function renderChecks(rep: Report): void {
  const order = { fail: 0, warn: 1, info: 2, pass: 3 } as const;
  const list = [...rep.checks].sort((a, b) => order[a.status] - order[b.status]);
  $("checks").replaceChildren(...list.map((c) => {
    const icon = { pass: "✓", warn: "!", fail: "✕", info: "i" }[c.status];
    const li = h("li", { cls: c.status, tabIndex: 0, title: c.highlight.length ? "Show in 3D" : "" },
      h("span", { cls: "ic", textContent: icon }), h("span", { cls: "t", textContent: c.title }),
      h("span", { cls: "m", textContent: c.message }));
    const toggle = (): void => {
      const on = state.focus?.id !== c.id;
      state.focus = on ? c : null;
      document.querySelectorAll(".checks li").forEach((x) => x.classList.remove("focus"));
      if (on) li.classList.add("focus");
      if (state.report) viewer.update(state.report, state.focus);
    };
    li.addEventListener("click", toggle);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    return li;
  }));
}

// ------------------------------------------------------------------ slicing / export / washers

function sliceLine(label: string, r: SliceResult | undefined): HTMLElement {
  if (!r) return h("div");
  if (!r.ok) return h("div", { cls: "err", textContent: `${label}: unavailable — ${r.error}` });
  const tower = r.features["Prime tower"] ?? 0;
  const extra = [
    r.support_seconds ? `supports ${formatDuration(r.support_seconds)}` : "",
    tower ? `prime tower ${formatDuration(tower)}` : "",
  ].filter(Boolean).join(", ");
  return h("tr", {},
    h("td", { textContent: label }),
    h("td", { textContent: formatDuration(r.seconds) }),
    h("td", { textContent: `${r.grams.toFixed(1)} g` }),
    h("td", { cls: "muted", textContent: extra }));
}

async function doSlice(): Promise<void> {
  if (!state.session) return;
  const btn = $<HTMLButtonElement>("sliceBtn");
  const out = $("sliceOut");
  btn.disabled = true;
  out.replaceChildren(h("div", { cls: "muted", textContent: "Slicing with Bambu Studio…" }));
  try {
    const x10 = $<HTMLInputElement>("x10Toggle").checked;
    const res: SliceVariants = await api.slice(state.session, state.params, x10, (s) => {
      out.firstElementChild!.textContent = `Slicing with Bambu Studio… ${Math.round(s)} s`;
    });
    const needs = res.supports_needed;
    const verdict = h("div", {
      cls: `verdict ${needs ? "yes" : "no"}`,
      textContent: needs === null ? "Supports: unknown" : needs ? "Supports needed" : "No supports needed",
    });
    const per = res.x10?.ok ? h("div", { cls: "muted", textContent:
      `10× plate: ${formatDuration(res.x10.seconds / 10)} and ${(res.x10.grams / 10).toFixed(1)} g per figure` }) : "";
    const tbl = h("table", { cls: "tbl" },
      h("thead", {}, h("tr", {}, ...["", "time", "filament", ""].map((t) => h("th", { textContent: t })))),
      h("tbody", {},
        sliceLine("1× no supports", res.off),
        sliceLine("1× auto supports", res.auto),
        res.x10 ? sliceLine(`10× ${needs ? "with" : "no"} supports`, res.x10) : h("tr")));
    applySlicerVerdict(res);
    const warn = [...res.off.warnings, ...res.auto.warnings].filter(Boolean);
    out.replaceChildren(verdict, tbl, per, ...warn.map((w) => h("div", { cls: "muted", textContent: `Bambu: ${w}` })));
  } catch (e) {
    out.replaceChildren(h("div", { cls: "err", textContent: (e as Error).message }));
  } finally {
    btn.disabled = !state.config?.slicer;
  }
}

/** The slicer has the final word on supports: replace the geometric estimate in the checklist. */
function applySlicerVerdict(res: SliceVariants): void {
  const rep = state.report;
  if (!rep || res.supports_needed === null) return;
  const c = rep.checks.find((x) => x.id === "supports");
  if (!c) return;
  const estimate = c.message;
  c.status = res.supports_needed ? "warn" : "pass";
  c.value = res.supports_needed ? "yes" : "no";
  c.message = res.supports_needed
    ? `Yes — Bambu auto supports add ${formatDuration(res.auto.support_seconds)} of support printing.`
    : `No — Bambu's auto supports generated none. (Geometry estimate: ${estimate.replace(/\.$/, "")}.)`;
  renderChecks(rep);
  rep.status = rep.checks.some((x) => x.status === "fail") ? "fail"
    : rep.checks.some((x) => x.status === "warn") ? "warn" : "pass";
  renderStatus(rep);
}

async function doExport(): Promise<void> {
  if (!state.session) return;
  const btn = $<HTMLButtonElement>("exportBtn");
  btn.disabled = true;
  busy(true, "Exporting…");
  try {
    const res = await api.exportFiles(state.session, state.params);
    $("exportOut").replaceChildren(
      ...res.files.map((f) => h("a", { href: f.url, textContent: `⬇ ${f.name}`, download: f.name })),
      h("div", { cls: "muted", textContent: res.files[0]?.path
        ? `Also saved in ${res.files[0].path.replace(/[\\/][^\\/]+$/, "")}`
        : "Made in your browser: click each file to save it." }),
      ...(res.note ? [h("div", { cls: "err", textContent: res.note })] : []),
    );
  } catch (e) {
    $("exportOut").replaceChildren(h("div", { cls: "err", textContent: (e as Error).message }));
  } finally {
    btn.disabled = false;
    busy(false);
  }
}

async function compareWashers(): Promise<void> {
  if (!state.session) return;
  const btn = $<HTMLButtonElement>("washerCompare");
  btn.disabled = true;
  $("washerTable").replaceChildren(h("div", { cls: "muted", textContent:
    api.engineMode() === "browser" ? "Trying every washer (about a minute in the browser)…" : "Trying every washer (≈ 30 s)…" }));
  try {
    const rows: WasherRow[] = await api.washers(state.session, state.params);
    const best = rows.find((r) => r.pockets_on_steel)?.key;
    $("washerTable").replaceChildren(h("table", { cls: "tbl" },
      h("thead", {}, h("tr", {}, ...["washer", "figure", "magnets", ""].map((t) => h("th", { textContent: t })))),
      h("tbody", {}, ...rows.map((r) => {
        const use = h("button", { cls: "btn", textContent: "Use" });
        use.addEventListener("click", () => {
          $<HTMLSelectElement>("washerSelect").value = r.key;
          state.params.washer = r.key;
          state.params.pockets = null;
          void runAnalyse();
        });
        const tr = h("tr", { cls: `${r.key === best ? "best" : ""} ${r.key === state.params.washer ? "current" : ""}` },
          h("td", { textContent: r.key }), h("td", { textContent: `${r.height_mm.toFixed(1)} mm` }),
          h("td", { textContent: { full: "on steel", partial: "half on steel", off: "off steel" }[r.steel] }),
          h("td", {}, use));
        tr.title = r.label;
        return tr;
      }))));
  } catch (e) {
    $("washerTable").replaceChildren(h("div", { cls: "err", textContent: (e as Error).message }));
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ events

function bind(): void {
  const fileInput = $<HTMLInputElement>("fileInput");
  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (f) void load(() => api.loadFile(f), f.name);
    fileInput.value = "";
  });
  const vp = $("viewport");
  const drop = $("drop");
  vp.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.remove("hidden"); drop.classList.add("over"); });
  vp.addEventListener("dragleave", () => { drop.classList.remove("over"); if (state.session) drop.classList.add("hidden"); });
  vp.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f) void load(() => api.loadFile(f), f.name);
    else if (state.session) drop.classList.add("hidden");
  });

  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((b) =>
    b.addEventListener("click", () => viewer.view(b.dataset.view as "iso")));
  const clip = $<HTMLButtonElement>("clipBtn");
  clip.addEventListener("click", () => {
    const on = clip.getAttribute("aria-pressed") !== "true";
    clip.setAttribute("aria-pressed", String(on));
    viewer.setClip(on);
  });
  const space = $<HTMLButtonElement>("spaceBtn");
  space.addEventListener("click", () => {
    const on = space.getAttribute("aria-pressed") !== "true";
    space.setAttribute("aria-pressed", String(on));
    viewer.setAllowedVisible(on);
  });

  const slider = $<HTMLInputElement>("scaleSlider");
  slider.addEventListener("input", () => {
    viewer.previewScale(+slider.value / 100);
    if (state.report) $("sumHeight").textContent = ((+slider.value / 100) * state.report.fit.unit_height).toFixed(1);
  });
  slider.addEventListener("change", () => {
    state.params.figure_scale = +slider.value / 100;
    void runAnalyse();
  });
  $<HTMLInputElement>("heightInput").addEventListener("change", (e) => {
    const mm = +(e.target as HTMLInputElement).value;
    if (!state.report || !(mm > 0)) return;
    state.params.figure_scale = mm / state.report.fit.unit_height;
    void runAnalyse();
  });
  $("maxBtn").addEventListener("click", () => {
    state.params.figure_scale = null;
    state.params.pockets = null;
    void runAnalyse();
  });

  /** A preset ball (print scale) or a custom inner diameter in mm. */
  const setBall = (scale: number, diameter: number | null): void => {
    state.params.ball_scale = scale;
    state.params.ball_diameter = diameter;
    state.params.figure_scale = null;
    state.params.pockets = null;
    void runAnalyse();
  };
  $<HTMLSelectElement>("ballSelect").addEventListener("change", (e) => {
    const sel = e.target as HTMLSelectElement;
    const custom = sel.value === "custom";
    $("ballCustomRow").hidden = !custom;
    if (custom) {
      const inp = $<HTMLInputElement>("ballCustom");
      inp.value = (state.report?.ball.inner_diameter ?? 44.2).toFixed(1);
      inp.focus();
    } else setBall(+sel.value, null);
  });
  $<HTMLInputElement>("ballCustom").addEventListener("change", (e) => {
    const v = +(e.target as HTMLInputElement).value;
    if (v >= 15 && v <= 300) setBall(state.params.ball_scale, v);
    else toast("Inner diameter must be between 15 and 300 mm.");
  });
  $<HTMLSelectElement>("washerSelect").addEventListener("change", (e) => {
    const sel = e.target as HTMLSelectElement;
    $("washerCustomRow").hidden = sel.value !== "custom";
    if (sel.value === "custom") return;
    state.params.washer = sel.value;
    state.params.pockets = null;
    void runAnalyse();
  });
  $("wApply").addEventListener("click", () => {
    const od = +$<HTMLInputElement>("wOd").value;
    const id = +$<HTMLInputElement>("wId").value;
    const t = +$<HTMLInputElement>("wT").value;
    if (!(od > id && id >= 0 && t > 0)) { toast("Washer needs OD > hole ≥ 0 and thickness > 0."); return; }
    const key = `${od},${id},${t}`;
    const sel = $<HTMLSelectElement>("washerSelect");
    if (![...sel.options].some((o) => o.value === key)) {
      sel.insertBefore(h("option", { value: key, textContent: `custom ${od} × ${id} × ${t} mm` }), sel.lastElementChild);
    }
    sel.value = key;
    $("washerCustomRow").hidden = true;
    state.params.washer = key;
    state.params.pockets = null;
    void runAnalyse();
  });
  $<HTMLInputElement>("clearanceInput").addEventListener("change", (e) => {
    const v = +(e.target as HTMLInputElement).value;
    if (v >= 0) { state.params.clearance = v; state.params.figure_scale = null; void runAnalyse(); }
  });
  $<HTMLSelectElement>("objectSelect").addEventListener("change", async (e) => {
    state.params.object_id = (e.target as HTMLSelectElement).value;
    colorize.reset();                       // paint refers to triangles of the previous figure
    state.params.figure_scale = null;
    state.params.pockets = null;
    state.params.filament_map = null;
    await refreshFigure();
    void runAnalyse();
  });
  /** New orientation: reset size and pockets, reload the preview, re-analyse. */
  const rotate = async (x: number, y: number, z: number): Promise<void> => {
    const norm = (a: number): number => +(((a % 360) + 360) % 360).toFixed(3);
    state.params.rotate_x = norm(x);
    state.params.rotate_y = norm(y);
    state.params.rotate_z = norm(z);
    state.params.figure_scale = null;
    state.params.pockets = null;
    await refreshFigure();
    void runAnalyse();
  };
  document.querySelectorAll<HTMLButtonElement>("[data-rot]").forEach((b) =>
    b.addEventListener("click", () => {
      const p = state.params;
      const ax = b.dataset.rot;
      void rotate(p.rotate_x + (ax === "x" ? 90 : 0), p.rotate_y + (ax === "y" ? 90 : 0), p.rotate_z + (ax === "z" ? 90 : 0));
    }));
  $("rotReset").addEventListener("click", () => void rotate(0, 0, 0));
  for (const id of ["rotX", "rotY", "rotZ"]) {
    $<HTMLInputElement>(id).addEventListener("change", () => {
      const v = (k: string): number => +$<HTMLInputElement>(k).value || 0;
      void rotate(v("rotX"), v("rotY"), v("rotZ"));
    });
  }
  $<HTMLInputElement>("ballModelInput").addEventListener("change", async (e) => {
    const inp = e.target as HTMLInputElement;
    const f = inp.files?.[0];
    inp.value = "";
    if (!f) return;
    busy(true, `Reading ${f.name}…`);
    try {
      $("ballModelInfo").textContent = `✓ ${await api.setBallModel(f)} — exports include the shell`;
    } catch (err) {
      toast(`Not the ball model: ${(err as Error).message}`);
    } finally {
      busy(false);
    }
  });
  $<HTMLInputElement>("refInput").addEventListener("change", async (e) => {
    const inp = e.target as HTMLInputElement;
    const f = inp.files?.[0];
    inp.value = "";
    if (!f) return;
    busy(true, `Reading ${f.name}…`);
    try {
      state.config = await api.setReference(f);
      renderPrinter(state.config);
      state.params.filament_map = null;
      if (state.session) await runAnalyse();
    } catch (err) {
      toast((err as Error).message);
    } finally {
      busy(false);
    }
  });
  document.querySelectorAll<HTMLButtonElement>("#magnetCount button").forEach((b) =>
    b.addEventListener("click", () => {
      const n = +(b.dataset.n ?? 2);
      if (n === state.params.magnets) return;
      state.params.magnets = n;
      state.params.pockets = null;          // a new count is placed automatically
      void runAnalyse();
    }));
  document.querySelectorAll<HTMLButtonElement>("#seatMode button").forEach((b) =>
    b.addEventListener("click", () => {
      const seat = b.dataset.seat === "raised" ? "raised" : "washer";
      if (seat === state.params.pocket_seat) return;
      state.params.pocket_seat = seat;
      void runAnalyse();                    // hand-placed pockets keep their XY; their seats follow the mode
    }));
  for (const [id, key] of POCKET_INPUTS) {
    $<HTMLInputElement>(id).addEventListener("change", (e) => {
      const inp = e.target as HTMLInputElement;
      const v = +inp.value;
      if (!(v >= +inp.min && v <= +inp.max)) { toast(`${key.replace("pocket_", "").replace("_", " ")} must be ${inp.min}–${inp.max} mm.`); return; }
      state.params[key] = v;
      void runAnalyse();
    });
  }
  $("autoPockets").addEventListener("click", () => { state.params.pockets = null; void runAnalyse(); });
  $<HTMLSelectElement>("slotSelect").addEventListener("change", (e) => {
    state.params.slot = +(e.target as HTMLSelectElement).value;
  });
  // About panel: ✕, Escape or a click outside the text closes it; /#about opens it directly.
  const about = $("about");
  const showAbout = (on: boolean): void => {
    about.hidden = !on;
    $("aboutBtn").setAttribute("aria-expanded", String(on));
    if (on) $("aboutClose").focus();
  };
  $("aboutBtn").addEventListener("click", () => showAbout(about.hidden));
  $("aboutClose").addEventListener("click", () => showAbout(false));
  about.addEventListener("click", (e) => { if (e.target === about) showAbout(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !about.hidden) showAbout(false); });
  if (location.hash === "#about") showAbout(true);
  $("sliceBtn").addEventListener("click", () => void doSlice());
  $("exportBtn").addEventListener("click", () => void doExport());
  $("washerCompare").addEventListener("click", () => void compareWashers());
}

bind();
colorize.render();
void init();
