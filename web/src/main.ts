/** pokefit web UI: wires the panel controls, the 3D view and the underside editor to the API. */
import * as api from "./api";
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
    ball_scale: 0.6, washer: "M6", clearance: 1, figure_scale: null, rotate_x: 0, rotate_y: 0, magnets: true,
    pockets: null, pocket_diameter: 4.18, pocket_depth: 2.2, pocket_wall: 0.8, pocket_ceiling: 0.6,
    object_id: null, slot: null, settings_source: "reference",
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
  try {
    state.config = await api.getConfig();
  } catch (e) {
    toast(`Server not reachable: ${(e as Error).message}`);
    return;
  }
  const cfg = state.config;
  state.params.washer = cfg.default_washer;
  state.params.slot = cfg.default_slot;
  const ballSel = $<HTMLSelectElement>("ballSelect");
  for (const s of cfg.ball_scales) ballSel.append(h("option", { value: String(s), textContent: `${Math.round(s * 100)} % ball` }));
  ballSel.append(h("option", { value: "custom", textContent: "Other…" }));
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
  $("printerInfo").textContent = cfg.slicer
    ? `Sliced with your ${cfg.printer} / ${cfg.process} settings (from ${cfg.reference}).`
    : "Bambu Studio not found — slicing unavailable; geometry checks still run.";
  ($("sliceBtn") as HTMLButtonElement).disabled = !cfg.slicer;

  const path = new URLSearchParams(location.search).get("path");
  if (path) await load(() => api.loadPath(path), path.split(/[\\/]/).pop() ?? path);
}

async function load(fn: () => Promise<import("./types").LoadResponse>, name: string): Promise<void> {
  busy(true, `Loading ${name}…`);
  try {
    const res = await fn();
    state.session = res.session;
    state.params = { ...state.params, figure_scale: null, pockets: null, rotate_x: 0, rotate_y: 0, object_id: null };
    $("fileName").textContent = res.file;
    $("drop").classList.add("hidden");
    $("sliceOut").replaceChildren();
    $("exportOut").replaceChildren();
    $("washerTable").replaceChildren();
    // Initial analysis used server defaults; re-run if the UI's settings differ.
    const r = res.report;
    const differs = r.params.washer !== state.params.washer || r.params.ball_scale !== state.params.ball_scale ||
      r.params.clearance !== state.params.clearance || r.params.magnets !== state.params.magnets;
    await viewer.setFigure(api.previewUrl(res.session, state.params));
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

async function runAnalyse(): Promise<void> {
  if (!state.session) return;
  const seq = ++state.seq;
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

  const preset = state.config?.ball_scales.includes(rep.params.ball_scale) ?? false;
  $<HTMLSelectElement>("ballSelect").value = preset ? String(rep.params.ball_scale) : "custom";
  $("ballCustomRow").hidden = preset;
  if (!preset) $<HTMLInputElement>("ballCustom").value = String(Math.round(rep.params.ball_scale * 100));
  $("rotLabel").textContent = rep.params.rotate_x || rep.params.rotate_y
    ? `X ${rep.params.rotate_x}° · Y ${rep.params.rotate_y}°` : "";

  renderObjects(rep);
  renderChecks(rep);
  underside.render(rep.underside, rep.pockets);
  $("pocketInfo").textContent = rep.params.pockets ? "placed by hand" : rep.pockets.length ? "automatic" : "";
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
  const row = $("objectRow");
  row.hidden = o.objects.length < 2;
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
      h("div", { cls: "muted", textContent: `Also saved in ${res.files[0]?.path.replace(/[\\/][^\\/]+$/, "") ?? ""}` }),
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
  $("washerTable").replaceChildren(h("div", { cls: "muted", textContent: "Trying every washer (≈ 30 s)…" }));
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

  const setBall = (v: number): void => {
    state.params.ball_scale = v;
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
      inp.value = String(Math.round(state.params.ball_scale * 100));
      inp.focus();
    } else setBall(+sel.value);
  });
  $<HTMLInputElement>("ballCustom").addEventListener("change", (e) => {
    const v = +(e.target as HTMLInputElement).value;
    if (v >= 10 && v <= 300) setBall(v / 100);
    else toast("Ball scale must be between 10 and 300 %.");
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
    state.params.figure_scale = null;
    state.params.pockets = null;
    if (state.session) await viewer.setFigure(api.previewUrl(state.session, state.params));
    void runAnalyse();
  });
  document.querySelectorAll<HTMLButtonElement>("[data-rot]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (b.dataset.rot === "x") state.params.rotate_x = (state.params.rotate_x + 90) % 360;
      else state.params.rotate_y = (state.params.rotate_y + 90) % 360;
      state.params.figure_scale = null;
      state.params.pockets = null;
      if (state.session) await viewer.setFigure(api.previewUrl(state.session, state.params));
      void runAnalyse();
    }));
  $<HTMLInputElement>("magnetToggle").addEventListener("change", (e) => {
    state.params.magnets = (e.target as HTMLInputElement).checked;
    state.params.pockets = null;
    void runAnalyse();
  });
  $("autoPockets").addEventListener("click", () => { state.params.pockets = null; void runAnalyse(); });
  $<HTMLSelectElement>("slotSelect").addEventListener("change", (e) => {
    state.params.slot = +(e.target as HTMLSelectElement).value;
  });
  $("sliceBtn").addEventListener("click", () => void doSlice());
  $("exportBtn").addEventListener("click", () => void doExport());
  $("washerCompare").addEventListener("click", () => void compareWashers());
}

bind();
void init();
