/**
 * Colorize panel: choose a filament colour, then tap the 3D figure to paint the
 * region around the tap. The engine finds the region from the geometry (up to
 * the nearest crease, the whole connected piece, or one triangle) and writes it
 * as Bambu paint, so a single-colour file can come out multi-colour.
 *
 * State lives in the shared `Params` (`colours`, `paint`), so analysis and
 * export see exactly what the preview shows.
 */
import { textOn } from "./colour";
import { formatDuration } from "./geometry";
import type { FigureHit } from "./scene";
import type { Params, PaintMode, PickResult, RegionView, Report } from "./types";

export interface ColorizeHost {
  params(): Params;
  report(): Report | null;
  /** Reload the 3D preview; `regions` colours it by what a tap would paint. */
  refresh(regions: RegionView | null): Promise<void>;
  /** The triangle under a tap and the size of its region. */
  pick(hit: FigureHit, mode: PaintMode, angle: number, radius: number): Promise<PickResult>;
  /** Turn tap-to-pick on the 3D view on or off. */
  setPicking(fn: ((hit: FigureHit) => void) | null): void;
  /** Called after paint or colours changed (the import-file colour map is stale until re-analysed). */
  changed(): void;
  /** Painting stopped: a good moment to re-analyse if anything changed. */
  settled(): void;
  toast(msg: string): void;
  /** The slot (AMS tray or project slot) filament `n` prints from, per the last analysis; null = unknown. */
  slotOf(n: number): { label: string; colour: string | null } | null;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

/** Colour offered for a newly added filament: the first of these not already in the palette. */
const NEW_COLOURS = ["#161616", "#FFFFFF", "#E05028", "#2850E0", "#0ACC38", "#FFF144", "#A03CF7", "#F98C36", "#F5A3C7", "#7A4A2A"];

export class ColorizePanel {
  private filament = 1;            // selected filament (1-based; 0 = erase)
  private mode: PaintMode = "part";
  /** Crease angle per mode: valleys separate parts at a lower angle than Bambu's 30° smart fill. */
  private angles: Partial<Record<PaintMode, number>> = { part: 15, region: 30 };
  private brushMm = 1.5;           // brush radius in print mm
  private painting = false;
  private regions = false;
  private busy = false;

  constructor(private host: ColorizeHost) {
    $("paintToggle").addEventListener("click", () => this.setPainting(!this.painting));
    $("regionsBtn").addEventListener("click", () => { this.regions = !this.regions; void this.reload(); this.render(); });
    $("paintUndo").addEventListener("click", () => this.edit((p) => p.slice(0, -1)));
    $("paintClear").addEventListener("click", () => this.edit(() => []));
    document.querySelectorAll<HTMLButtonElement>("#paintMode button").forEach((b) =>
      b.addEventListener("click", () => {
        this.mode = (b.dataset.mode as PaintMode) ?? "part";
        this.render();
        if (this.regions) void this.reload();
      }));
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
    $<HTMLInputElement>("paintColour").addEventListener("change", () => { this.host.changed(); void this.reload(); });
    const tex = $<HTMLInputElement>("texCol");
    tex.addEventListener("input", () => { $("texColVal").textContent = String(tex.value); });
    tex.addEventListener("change", () => this.setTextureColours(+tex.value));
    $("texColAuto").addEventListener("click", () => this.setTextureColours(null));
    const details = $<HTMLSelectElement>("texDetails");
    details.addEventListener("change", () => this.setTextureDetails(+details.value));
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
    this.setPainting(false);
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
    $("paintColourLabel").textContent = this.filament ? `Filament ${this.filament}` : "";
    document.querySelectorAll<HTMLButtonElement>("#paintMode button").forEach((b) =>
      b.setAttribute("aria-checked", String(b.dataset.mode === this.mode)));
    const angled = this.mode === "part" || this.mode === "region";
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
    const n = p.paint?.length ?? 0;
    ($("paintUndo") as HTMLButtonElement).disabled = n === 0;
    ($("paintClear") as HTMLButtonElement).disabled = n === 0;
    if (!this.busy && !$("paintInfo").dataset.sticky) {
      $("paintInfo").textContent = this.painting ? "Tap the figure to paint." : n ? `${n} paint step${n > 1 ? "s" : ""}` : "";
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
    });
    return b;
  }

  private setPainting(on: boolean): void {
    this.painting = on;
    this.host.setPicking(on ? (hit) => void this.paintAt(hit) : null);
    if (!on && this.regions) { this.regions = false; void this.reload(); }
    if (!on) this.host.settled();
    this.render();
  }

  private async paintAt(hit: FigureHit): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    $("paintInfo").textContent = "Finding the region…";
    try {
      const radius = this.radiusFile();
      const res = await this.host.pick(hit, this.mode, this.angle, radius);
      const p = this.host.params();
      if (!p.colours) p.colours = this.colours();      // pin the palette the paint refers to
      p.paint = [...(p.paint ?? []), {
        part: res.part, face: res.face, filament: this.filament, mode: this.mode, angle: this.angle, radius,
      }];
      const what = this.mode === "face" ? "1 triangle" : `${res.faces.toLocaleString()} triangles`;
      const angled = this.mode === "part" || this.mode === "region";
      $("paintInfo").textContent = `${this.filament ? "Painted" : "Erased"} ${what}` +
        (res.share > 0.9 && angled ? " — nearly everything: lower the angle to stop sooner, or use the brush" : "");
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

  /** Crease angle of the current mode (the brush and piece modes ignore it). */
  private get angle(): number {
    return this.angles[this.mode] ?? 15;
  }

  /** Brush radius in the file's mm: the figure prints at `fit.scale` of the file. */
  private radiusFile(): number {
    if (this.mode !== "brush") return 0;
    const s = this.host.report()?.fit.scale || this.host.report()?.fit.max_scale || 1;
    return +(this.brushMm / s).toFixed(4);
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
    await this.host.refresh(this.regions ? { mode: this.mode, angle: this.angle } : null);
  }
}

