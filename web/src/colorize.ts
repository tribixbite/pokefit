/**
 * Colorize panel: choose a filament colour, then tap the 3D figure to paint the
 * region around the tap. The engine finds the region from the geometry (up to
 * the nearest crease, the whole connected piece, or one triangle) and writes it
 * as Bambu paint, so a single-colour file can come out multi-colour.
 *
 * Also: a bucket "Fill" bounded only by colour, Mirror (the same spot on the
 * other side of the figure), Pick colour (eyedropper), Grow/Shrink of the last
 * step, Undo/Redo, and keyboard shortcuts.
 *
 * State lives in the shared `Params` (`colours`, `paint`), so analysis and
 * export see exactly what the preview shows.
 */
import { textOn } from "./colour";
import { formatDuration } from "./geometry";
import type { FigureHit } from "./scene";
import type { PaintMode, PaintOp, Params, PickResult, PrepareResult, RegionView, Report } from "./types";

export interface ColorizeHost {
  params(): Params;
  report(): Report | null;
  /** Reload the 3D preview; `regions` colours it by what a tap would paint. */
  refresh(regions: RegionView | null): Promise<void>;
  /** The triangle under a tap and the size of its region; `stroke`: the rest of a brush drag. */
  pick(hit: FigureHit, mode: PaintMode, angle: number, radius: number, stroke?: FigureHit[], mirror?: boolean,
       stampAxes?: number[]): Promise<PickResult>;
  /** Build the engine's tap structures now (painting started), so the first tap is as quick as the rest. */
  prepare(mode: PaintMode, angle: number, mirror: boolean): Promise<PrepareResult>;
  /** Turn tap-to-pick on the 3D view on or off. */
  setPicking(fn: ((hit: FigureHit) => void) | null): void;
  /** Brush: a drag that starts on the figure paints a stroke (null = drags turn the view). */
  setStroking(fn: ((hits: FigureHit[]) => void) | null): void;
  /** Stamp: a drag on the figure draws an ellipse's box (null = drags turn the view). */
  setStamping(fn: ((hit: FigureHit, axes: number[]) => void) | null): void;
  /** Called after paint or colours changed (the import-file colour map is stale until re-analysed). */
  changed(): void;
  /** Painting stopped: a good moment to re-analyse if anything changed. */
  settled(): void;
  /** A colour was added or changed: which slot each colour prints from must be matched again. */
  coloursChanged(): void;
  toast(msg: string): void;
  /** The slot (AMS tray or project slot) filament `n` prints from, per the last analysis; null = unknown. */
  slotOf(n: number): { label: string; colour: string | null } | null;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

/** Colour offered for a newly added filament: the first of these not already in the palette. */
const NEW_COLOURS = ["#161616", "#FFFFFF", "#E05028", "#2850E0", "#0ACC38", "#FFF144", "#A03CF7", "#F98C36", "#F5A3C7", "#7A4A2A"];
/** Grow / Shrink step, print mm along the surface. */
const GROW_STEP = 0.2;
/** The engine's limit on an operation's grow, file mm (colorize.MAX_GROW). */
const MAX_GROW = 20;
/** Wand: the furthest step from its first answer (colorize.MAX_LEVEL). */
const MAX_LEVEL = 8;
/** Modes whose angle slider applies. */
const ANGLED: readonly PaintMode[] = ["part", "region"];

export class ColorizePanel {
  private filament = 1;            // selected filament (1-based; 0 = erase)
  private mode: PaintMode = "wand";
  /** Crease angle per mode: valleys separate parts at a lower angle than Bambu's 30° smart fill. */
  private angles: Partial<Record<PaintMode, number>> = { part: 15, region: 30 };
  private brushMm = 1.5;           // brush radius in print mm
  private painting = false;
  private regions = false;
  /** The shape map (creases and relief, what the wand follows) is shown. */
  private shapeMap = false;
  private busy = false;
  /** Paint the mirror image too (the figure's symmetry plane, found by the engine). */
  private mirror = false;
  /** The next tap picks the colour under it instead of painting. */
  private sampling = false;
  /** Undone steps, last undone at the end (Redo puts them back; a new step clears them). */
  private undone: PaintOp[] = [];

  constructor(private host: ColorizeHost) {
    $("paintToggle").addEventListener("click", () => this.setPainting(!this.painting));
    $("regionsBtn").addEventListener("click", () => {
      this.regions = !this.regions;
      if (this.regions) this.shapeMap = false;
      void this.reload();
      this.render();
    });
    $("shapeMapBtn").addEventListener("click", () => {
      this.shapeMap = !this.shapeMap;
      if (this.shapeMap) this.regions = false;
      void this.reload();
      this.render();
    });
    $("paintUndo").addEventListener("click", () => this.undo());
    $("paintRedo").addEventListener("click", () => this.redo());
    $("paintClear").addEventListener("click", () => {
      const ops = this.host.params().paint ?? [];
      this.undone.push(...[...ops].reverse());          // Redo brings them back one by one, first step first
      this.edit(() => []);
    });
    $("paintGrow").addEventListener("click", () => this.growLast(GROW_STEP));
    $("paintSmaller").addEventListener("click", () => this.stepLast(-1));
    $("paintLarger").addEventListener("click", () => this.stepLast(1));
    $("paintShrink").addEventListener("click", () => this.growLast(-GROW_STEP));
    $("paintMirror").addEventListener("click", () => void this.setMirror(!this.mirror));
    $("paintSample").addEventListener("click", () => {
      this.sampling = !this.sampling;
      if (this.sampling && !this.painting) this.setPainting(true);
      this.render();
    });
    document.querySelectorAll<HTMLButtonElement>("#paintMode button").forEach((b) =>
      b.addEventListener("click", () => this.setMode((b.dataset.mode as PaintMode) ?? "part")));
    const slider = $<HTMLInputElement>("paintAngle");
    slider.addEventListener("input", () => { this.angles[this.mode] = +slider.value; this.render(); });
    slider.addEventListener("change", () => { if (this.regions) void this.reload(); });
    const brush = $<HTMLInputElement>("brushSize");
    brush.addEventListener("input", () => { this.brushMm = +brush.value; this.render(); });
    $<HTMLInputElement>("paintColour").addEventListener("input", (e) => {
      if (this.filament < 1) return;
      const cols = this.colours();
      cols[this.filament - 1] = (e.target as HTMLInputElement).value.toUpperCase();
      this.host.params().colours = cols;
      this.render();
    });
    $<HTMLInputElement>("paintColour").addEventListener("change", () => {
      this.host.changed();
      void this.reload();
      this.host.coloursChanged();
    });
    const tex = $<HTMLInputElement>("texCol");
    tex.addEventListener("input", () => { $("texColVal").textContent = String(tex.value); });
    tex.addEventListener("change", () => this.setTextureColours(+tex.value));
    $("texColAuto").addEventListener("click", () => this.setTextureColours(null));
    const details = $<HTMLSelectElement>("texDetails");
    details.addEventListener("change", () => this.setTextureDetails(+details.value));
    document.addEventListener("keydown", (e) => this.onKey(e));
  }

  /**
   * Re-cut a GLB's own colours to `n` filaments (null = automatic). The palette and the
   * filament numbers change, so the user's palette edits and paint steps are dropped.
   */
  private setTextureColours(n: number | null): void {
    const p = this.host.params();
    const auto = this.host.report()?.object.texture_colours?.auto;
    const next = n === auto ? null : n;
    if (next === p.texture_colours) { this.render(); return; }
    this.recut(() => { p.texture_colours = next; });
  }

  /** Keep `d` small, high-contrast features of a GLB as their own filaments (Params.texture_details). */
  private setTextureDetails(d: number): void {
    const p = this.host.params();
    if (d === p.texture_details) { this.render(); return; }
    this.recut(() => { p.texture_details = d; });
  }

  /** Apply a change to how a GLB's colours are cut into filaments: palette edits and paint steps go. */
  private recut(change: () => void): void {
    const p = this.host.params();
    if (p.paint?.length) this.host.toast("Paint steps cleared: the filaments were re-cut from the file's colours.");
    change();
    p.colours = null;
    p.paint = null;
    this.undone = [];
    this.filament = 1;
    this.host.changed();
    void this.reload();
    this.host.settled();
    this.render();
  }

  /** The palette in use: the user's, else the file's. */
  private colours(): string[] {
    const p = this.host.params();
    return [...(p.colours ?? this.host.report()?.object.colours ?? ["#E05028"])];
  }

  /** Forget paint and colours (a new file or another figure of the file: face numbers differ). */
  reset(): void {
    const p = this.host.params();
    p.paint = null;
    p.colours = null;
    p.texture_colours = null;
    p.texture_details = 0;
    this.filament = 1;
    this.regions = false;
    this.shapeMap = false;
    this.mirror = false;
    this.sampling = false;
    this.undone = [];
    this.setPainting(false);
  }

  /** Paint steps name triangles; when the triangles change (Simplify), the steps and their redo history go. */
  dropPaint(): void {
    this.host.params().paint = null;
    this.undone = [];
    this.render();
  }

  render(): void {
    const p = this.host.params();
    const cols = this.colours();
    const tex = this.host.report()?.object.texture_colours ?? null;
    $("texColBox").hidden = !tex;
    if (tex) {
      const n = p.texture_colours ?? tex.auto;
      const slider = $<HTMLInputElement>("texCol");
      slider.max = String(tex.max);
      slider.value = String(n);
      const toolchange = this.host.report()?.toolchange;
      const pick = toolchange?.best_count ?? null;
      // "auto" is the engine's best-value count (★) when the cost table has a jump, else the file's palette.
      $("texColVal").textContent = p.texture_colours === null ? `${n} (auto${n === pick ? " ★" : ""})` : String(n);
      $("texColAuto").setAttribute("aria-pressed", String(p.texture_colours === null));
      $<HTMLSelectElement>("texDetails").value = String(p.texture_details);
      if (tex.details) $("texColVal").textContent += ` + ${tex.details} detail${tex.details > 1 ? "s" : ""}`;
      // What each palette size costs in filament changes (estimated); tap one to use it.
      // With details on, each row is that count plus the detail filaments ("2+1").
      const rows = toolchange?.by_count ?? [];
      const plus = tex.details ? `+${tex.details}` : "";
      $("texCost").replaceChildren(...rows.map((r) => {
        const b = document.createElement("button");
        b.setAttribute("aria-current", String(r.count === n));
        b.title = `${r.count}${plus} filament${r.count === 1 && !plus ? "" : "s"}: about ${r.changes} filament changes, ` +
          `${formatDuration(r.minutes * 60)} and ${r.purge_g} g of purge on top of the print`;
        const extra = r.minutes > 0 ? ` · +${formatDuration(r.minutes * 60)}` : "";
        b.innerHTML = `<b>${r.count}${plus}</b> · ${r.changes ? `${r.changes} changes` : "no changes"}${extra}` +
          (r.count === pick ? " ★" : "");
        if (r.count === pick) b.title += ". ★ Best value: the next colour would add many more filament changes (auto adds at most 2 h of them).";
        b.addEventListener("click", () => this.setTextureColours(r.count));
        return b;
      }));
    }
    if (this.filament > cols.length) this.filament = 1;
    const pal = $("palette");
    pal.replaceChildren(...cols.map((c, i) => this.swatch(i + 1, c)), this.swatch(0, ""), this.addButton());
    $<HTMLInputElement>("paintColour").value = this.filament ? cols[this.filament - 1].slice(0, 7) : "#000000";
    $("paintColourRow").hidden = this.filament === 0;
    // "Prints from": mark the selected colour's row.
    document.querySelectorAll<HTMLElement>("#importMap .slot-row").forEach((r) =>
      r.classList.toggle("sel", r.dataset.n === String(this.filament)));
    $("paintColourLabel").textContent = this.filament ? `Filament ${this.filament}` : "";
    document.querySelectorAll<HTMLButtonElement>("#paintMode button").forEach((b) =>
      b.setAttribute("aria-checked", String(b.dataset.mode === this.mode)));
    const angled = ANGLED.includes(this.mode);
    $("paintAngleRow").hidden = !angled;
    $("paintAngleVal").textContent = `${this.angle}°`;
    $<HTMLInputElement>("paintAngle").value = String(this.angle);
    $("brushRow").hidden = this.mode !== "brush";
    $("brushVal").textContent = `${this.brushMm.toFixed(1)} mm`;
    $<HTMLInputElement>("brushSize").value = String(this.brushMm);
    ($("regionsBtn") as HTMLButtonElement).disabled = !angled && this.mode !== "piece";
    const t = $("paintToggle");
    t.setAttribute("aria-pressed", String(this.painting));
    t.textContent = this.painting ? "Done painting" : "Paint";
    $("regionsBtn").setAttribute("aria-pressed", String(this.regions));
    $("shapeMapBtn").setAttribute("aria-pressed", String(this.shapeMap));
    $("paintMirror").setAttribute("aria-pressed", String(this.mirror));
    $("paintSample").setAttribute("aria-pressed", String(this.sampling));
    const ops = p.paint ?? [];
    const n = ops.length;
    ($("paintUndo") as HTMLButtonElement).disabled = n === 0;
    ($("paintRedo") as HTMLButtonElement).disabled = this.undone.length === 0;
    ($("paintClear") as HTMLButtonElement).disabled = n === 0;
    ($("paintGrow") as HTMLButtonElement).disabled = n === 0 || ops[n - 1].filament === 0;
    $("levelGroup").hidden = !(n > 0 && ops[n - 1].mode === "wand");
    ($("paintShrink") as HTMLButtonElement).disabled = n === 0 || ops[n - 1].filament === 0;
    if (!this.busy && !$("paintInfo").dataset.sticky) {
      $("paintInfo").textContent = this.sampling ? "Tap the figure to pick its colour there."
        : !this.painting ? (n ? `${n} paint step${n > 1 ? "s" : ""}` : "")
          : this.mode === "brush"
            ? "Tap or drag on the figure to paint. Turn the view by dragging beside it, with two fingers or the right mouse button."
            : this.mode === "fill" ? "Tap inside an outline to fill it."
              : this.mode === "wand" ? "Tap a feature (an eye, a flame, a toe pad). Smaller / Larger if it is not quite it."
                : this.mode === "stamp"
                  ? "Drag a box over the spot (an eye): the ellipse in it is painted, seen as you look at it. Turn the view by dragging beside the figure."
                  : "Tap the figure to paint.";
    }
  }

  private swatch(filament: number, colour: string): HTMLElement {
    const b = document.createElement("button");
    b.className = `swatch${filament === 0 ? " erase" : ""}`;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(filament === this.filament));
    b.title = filament ? `Filament ${filament} (${colour})` : "Erase paint (back to the part's own colour)";
    b.setAttribute("aria-label", b.title);
    if (filament) {
      b.style.background = colour;
      b.textContent = String(filament);
      b.style.color = textOn(colour);
      // Which spool/slot it prints from: a corner dot in that slot's colour, its label in the tooltip.
      const slot = this.host.slotOf(filament);
      if (slot) {
        const dot = document.createElement("span");
        dot.className = `slot-dot${slot.colour ? "" : " none"}`;
        if (slot.colour) { dot.style.background = slot.colour; dot.style.color = textOn(slot.colour); }
        dot.textContent = slot.label;
        b.append(dot);
        b.title += slot.label === "?" ? " · no loaded spool is close"
          : slot.label === "+" ? " · prints from a new slot (Bambu adds it on import)" : ` · prints from ${slot.label}`;
        b.setAttribute("aria-label", b.title);
      }
    } else b.textContent = "⌫";
    b.addEventListener("click", () => { this.filament = filament; this.render(); });
    return b;
  }

  private addButton(): HTMLElement {
    const b = document.createElement("button");
    b.className = "swatch add";
    b.textContent = "+";
    b.title = "Add a filament colour";
    b.setAttribute("aria-label", b.title);
    b.addEventListener("click", () => {
      const cols = this.colours();
      if (cols.length >= 16) { this.host.toast("16 colours at most."); return; }
      const have = new Set(cols.map((c) => c.toUpperCase().slice(0, 7)));
      cols.push(NEW_COLOURS.find((c) => !have.has(c)) ?? "#808080");
      this.host.params().colours = cols;
      this.filament = cols.length;
      this.host.changed();
      this.render();
      this.host.coloursChanged();
    });
    return b;
  }

  private setMode(mode: PaintMode): void {
    this.mode = mode;
    if (this.painting) this.setPainting(true);     // brush: drags paint strokes
    this.render();
    if (this.regions) void this.reload();
  }

  private setPainting(on: boolean): void {
    this.painting = on;
    if (!on) this.sampling = false;
    this.host.setPicking(on ? (hit) => {
      if (this.mode === "stamp" && !this.sampling) this.host.toast("Drag a box over the spot: the ellipse in it is painted.");
      else void this.paintAt([hit]);
    } : null);
    this.host.setStroking(on && this.mode === "brush" ? (hits) => void this.paintAt(hits) : null);
    this.host.setStamping(on && this.mode === "stamp" ? (hit, axes) => void this.paintAt([hit], axes) : null);
    // The engine caches these per figure and orientation, so asking again is cheap; failures surface on the tap.
    if (on) void this.host.prepare(this.mode, this.angle, this.mirror).catch(() => undefined);
    if (!on && this.regions) { this.regions = false; void this.reload(); }
    if (!on) this.host.settled();
    this.render();
  }

  /** Mirror on: the engine finds the figure's mirror plane first and says if there is none. */
  private async setMirror(on: boolean): Promise<void> {
    this.mirror = on;
    this.render();
    if (!on) return;
    $("paintInfo").textContent = "Finding the mirror plane…";
    try {
      const res = await this.host.prepare(this.mode, this.angle, true);
      if (res.mirror && !res.mirror.ok) {
        this.mirror = false;
        this.host.toast("This figure is not mirror-symmetric enough to paint both sides at once.");
      }
    } catch (e) {
      this.mirror = false;
      this.host.toast((e as Error).message);
    }
    this.render();
  }

  /** Paint at a tap, along a brush stroke (`hits` = the drag's samples, first = where it started), or a stamp
   * (`axes`: the ellipse's semi-axes, figure frame). */
  private async paintAt(hits: FigureHit[], axes?: number[]): Promise<void> {
    if (this.busy || !hits.length) return;
    if (this.sampling) { await this.sampleAt(hits[0]); return; }
    this.busy = true;
    $("paintInfo").textContent = "Finding the region…";
    try {
      const radius = this.radiusFile();
      const stroke = this.mode === "brush" ? hits.slice(1) : [];
      const res = await this.host.pick(hits[0], this.mode, this.angle, radius, stroke, this.mirror, axes);
      const p = this.host.params();
      if (!p.colours) p.colours = this.colours();      // pin the palette the paint refers to
      p.paint = [...(p.paint ?? []), {
        part: res.part, face: res.face, filament: this.filament, mode: this.mode, angle: this.angle, radius,
        ...(res.path?.length ? { path: res.path } : {}),
        ...(this.mirror ? { mirror: true } : {}),
        ...(res.stamp ? { stamp: res.stamp } : {}),
      }];
      this.undone = [];
      const what = this.mode === "face" ? "1 triangle" : `${res.faces.toLocaleString()} triangles`;
      const both = this.mirror ? (res.mirrored ? " on both sides" : " (no matching spot on the other side)") : "";
      $("paintInfo").textContent = `${this.filament ? "Painted" : "Erased"} ${what}${both}` +
        (res.share > 0.9 && ANGLED.includes(this.mode)
          ? " — nearly everything. No crease goes all the way round here: outline it with the brush (drag), " +
            "then Fill inside"
          : "");
      $("paintInfo").dataset.sticky = "1";
      this.host.changed();
      await this.reload();
    } catch (e) {
      this.host.toast((e as Error).message);
      $("paintInfo").textContent = "";
    } finally {
      this.busy = false;
      this.render();
      delete $("paintInfo").dataset.sticky;
    }
  }

  /** Pick colour: select the filament shown under the tap, then go back to painting. */
  private async sampleAt(hit: FigureHit): Promise<void> {
    this.busy = true;
    try {
      const res = await this.host.pick(hit, "face", 15, 0);
      if (res.filament >= 1 && res.filament <= this.colours().length) this.filament = res.filament;
      else this.host.toast(`That prints in filament ${res.filament}, which is not in this palette.`);
    } catch (e) {
      this.host.toast((e as Error).message);
    } finally {
      this.busy = false;
      this.sampling = false;
      this.render();
    }
  }

  /** Crease angle of the current mode (the brush, fill and piece modes ignore it). */
  private get angle(): number {
    return this.angles[this.mode] ?? 15;
  }

  /** The figure's print scale: file mm × scale = print mm. */
  private scale(): number {
    return this.host.report()?.fit.scale || this.host.report()?.fit.max_scale || 1;
  }

  /** Brush radius in the file's mm: the figure prints at `fit.scale` of the file. */
  private radiusFile(): number {
    if (this.mode !== "brush") return 0;
    return +(this.brushMm / this.scale()).toFixed(4);
  }

  /** Widen (`mm` > 0) or narrow the last paint step along the surface, `mm` at the print size. */
  private growLast(mm: number): void {
    const ops = this.host.params().paint ?? [];
    if (!ops.length || ops[ops.length - 1].filament === 0) return;
    const s = this.scale();
    const last = ops[ops.length - 1];
    const grow = Math.max(-MAX_GROW, Math.min(MAX_GROW, +((last.grow ?? 0) + mm / s).toFixed(4)));
    const next: PaintOp = { ...last, grow };
    if (Math.abs(grow * s) < 1e-3) delete next.grow;
    this.edit((p) => [...p.slice(0, -1), next]);
    const g = (next.grow ?? 0) * s;
    $("paintInfo").textContent = g ? `Last step ${g > 0 ? "grown" : "shrunk"} by ${Math.abs(g).toFixed(1)} mm` : "Last step as painted";
    $("paintInfo").dataset.sticky = "1";
    window.setTimeout(() => { delete $("paintInfo").dataset.sticky; }, 2500);
  }

  /** Wand: replace the last step's region with the next smaller (-1) or larger (+1) one the wand found. */
  private stepLast(dir: number): void {
    const ops = this.host.params().paint ?? [];
    const last = ops[ops.length - 1];
    if (!last || last.mode !== "wand") return;
    const level = Math.max(-MAX_LEVEL, Math.min(MAX_LEVEL, (last.level ?? 0) + dir));
    const next: PaintOp = { ...last, level };
    if (!level) delete next.level;
    this.edit((p) => [...p.slice(0, -1), next]);
  }

  private undo(): void {
    const ops = this.host.params().paint ?? [];
    if (!ops.length) return;
    this.undone.push(ops[ops.length - 1]);
    this.edit((p) => p.slice(0, -1));
  }

  private redo(): void {
    const op = this.undone.pop();
    if (op) this.edit((p) => [...p, op]);
  }

  /** Keyboard shortcuts while a figure is open (ignored while typing in a field). */
  private onKey(e: KeyboardEvent): void {
    const el = e.target as HTMLElement | null;
    if (el && (el.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName))) return;
    if (!this.host.report() || e.altKey) return;
    const k = e.key;
    const mod = e.ctrlKey || e.metaKey;
    let done = true;
    if (mod && (k === "z" || k === "Z")) { if (e.shiftKey) this.redo(); else this.undo(); }
    else if (mod && (k === "y" || k === "Y")) this.redo();
    else if (mod) done = false;
    else if (/^[1-9]$/.test(k) && +k <= this.colours().length) { this.filament = +k; this.render(); }
    else if (k === "0") { this.filament = 0; this.render(); }
    else if (k === "p" || k === "P") this.setPainting(!this.painting);
    else if (k === "Escape" && this.painting) this.setPainting(false);
    else if (k === "m" || k === "M") void this.setMirror(!this.mirror);
    else if (k === "i" || k === "I") { this.sampling = !this.sampling; if (this.sampling && !this.painting) this.setPainting(true); this.render(); }
    else if (k === "+" || k === "=") this.growLast(GROW_STEP);
    else if (k === "-" || k === "_") this.growLast(-GROW_STEP);
    else if (k === "[" || k === "]") this.nudge(k === "]" ? 1 : -1);
    else if (k === "," || k === ".") this.stepLast(k === "." ? 1 : -1);
    else done = false;
    if (done) e.preventDefault();
  }

  /** [ / ]: the brush radius (0.1 mm steps) or the crease angle (1° steps) down / up. */
  private nudge(dir: number): void {
    if (this.mode === "brush") this.brushMm = Math.max(0.3, Math.min(8, +(this.brushMm + dir * 0.1).toFixed(1)));
    else if (ANGLED.includes(this.mode)) this.angles[this.mode] = Math.max(3, Math.min(90, this.angle + dir));
    this.render();
  }

  private edit(fn: (ops: NonNullable<Params["paint"]>) => NonNullable<Params["paint"]>): void {
    const p = this.host.params();
    const next = fn(p.paint ?? []);
    p.paint = next.length ? next : null;
    this.host.changed();
    void this.reload();
    this.render();
  }

  private async reload(): Promise<void> {
    await this.host.refresh(this.regions ? { mode: this.mode, angle: this.angle }
      : this.shapeMap ? { mode: "shape", angle: 0 } : null);
  }
}
