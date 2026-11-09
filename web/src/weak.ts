/**
 * "Weak points" (Figure card): the engine grades every thin feature (an ear, a tail, a claw) by the push at its far
 * end that would snap it at its weakest section (pokefit/joints.py); the list shows the weakest first, each can be
 * shown on the view and thickened on its own (`Params.strengthen_at`: only that feature, from its weak section to
 * where it is strong enough). "Print weak parts solid" (`Params.reinforce`) adds Bambu modifier volumes over them in
 * the export instead: the shape stays as it is.
 */
import type { Params, StrengthenAt, WeakPoint, WeakReport } from "./types";

export interface WeakHost {
  params(): Params;
  /** Grade the figure as it is now (`api.weakPoints`). */
  grade(): Promise<WeakReport>;
  /** Highlight a piece's preview triangles on the view (null: none). */
  highlight(part: number, faces: number[] | null): void;
  /** The shape changed: show it and analyse again. */
  apply(): void;
  busy(on: boolean, text?: string): void;
  toast(msg: string): void;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

/** Thickness choices (mm) for one feature, as the engine allows (0 < mm <= 5). */
export const THICKNESSES = [2, 2.5, 3, 3.5, 4, 4.5, 5];
/** The push (N) grade C holds: what a suggested thickness aims for (joints.GOAL). */
const GOAL_N = 3;

/** One newton is about 102 grams pushing. */
export function grams(newtons: number): number {
  return Math.round(newtons * 101.97);
}

/**
 * A thickness (mm, one of THICKNESSES) that should bring a feature to at least grade C: a section's strength grows
 * with its thickness squared across its weak way, so the width it needs is its width x sqrt(goal / force), and at
 * least half a millimetre more than it is.
 */
export function suggestThickness(w: WeakPoint): number {
  const need = Math.max(w.width_mm * Math.sqrt(GOAL_N / Math.max(w.force_n, 0.01)), w.width_mm + 0.5);
  return THICKNESSES.find((t) => t >= need - 1e-9) ?? THICKNESSES[THICKNESSES.length - 1];
}

/** One line about a weak point: what breaks it, how big its section is, and why. */
export function describeWeak(w: WeakPoint): string {
  const layers = w.flat > 0.5 ? "; its layers pull apart there" : "";
  return `Snaps at ≈ ${w.force_n.toFixed(1)} N (${grams(w.force_n)} g) pushed at its end · ` +
    `${w.width_mm.toFixed(1)} × ${w.depth_mm.toFixed(1)} mm across, ${w.lever_mm.toFixed(0)} mm lever${layers}`;
}

export class WeakPanel {
  private report: WeakReport | null = null;
  private shown = -1;
  /** The figure the list was graded on changed since: it is shown as out of date. */
  private stale = false;

  constructor(private host: WeakHost) {
    $("weakFind").addEventListener("click", () => void this.find());
    $<HTMLInputElement>("reinforce").addEventListener("change", (e) => {
      this.host.params().reinforce = (e.target as HTMLInputElement).checked;
    });
  }

  /** A new figure: forget the list. */
  reset(): void {
    this.report = null;
    this.shown = -1;
    this.stale = false;
    this.host.highlight(0, null);
    this.render();
  }

  /** The figure's shape changed (another analysis): the grades are of the old one. */
  invalidate(): void {
    if (!this.report || this.stale) return;
    this.stale = true;
    this.shown = -1;
    this.host.highlight(0, null);
    this.render();
  }

  sync(): void {
    $<HTMLInputElement>("reinforce").checked = !!this.host.params().reinforce;
    this.render();
  }

  private async find(): Promise<void> {
    this.host.busy(true, "Grading thin parts…");
    try {
      this.report = await this.host.grade();
      this.stale = false;
      this.shown = -1;
      this.host.highlight(0, null);
      this.render();
      if (!this.report.points.length) {
        this.host.toast(`Nothing is thinner than ${this.report.probe_mm} mm at this size: no weak points.`);
      }
    } catch (e) {
      this.host.toast((e as Error).message);
    } finally {
      this.host.busy(false);
    }
  }

  private show(i: number): void {
    const w = this.report?.points[i];
    this.shown = this.shown === i || !w ? -1 : i;
    this.host.highlight(w?.part ?? 0, this.shown >= 0 && w ? w.view_faces : null);
    this.render();
  }

  private thicken(w: WeakPoint, mm: number): void {
    const p = this.host.params();
    const at: StrengthenAt = { part: w.part, face: w.face, root: w.root, mm };
    // The same weak section thickened again replaces its earlier thickness.
    p.strengthen_at = [...(p.strengthen_at ?? []).filter((q) => !(q.part === at.part && q.root === at.root)), at];
    this.host.toast(`Thickening it to ${mm} mm where it is weak (about a minute on a large figure in the browser).`);
    this.host.apply();
  }

  private undoAll(): void {
    this.host.params().strengthen_at = null;
    this.host.apply();
  }

  render(): void {
    const list = $("weakList");
    const info = $("weakInfo");
    const at = this.host.params().strengthen_at ?? [];
    const rows: HTMLElement[] = [];
    if (at.length) {
      const row = document.createElement("div");
      row.className = "weak-done";
      row.append(`${at.length === 1 ? "1 part" : `${at.length} parts`} thickened (${at.map((q) => `${q.mm} mm`).join(", ")}) `);
      const undo = document.createElement("button");
      undo.className = "btn small";
      undo.textContent = "Undo";
      undo.title = "Put every thickened weak part back as it was";
      undo.addEventListener("click", () => this.undoAll());
      row.append(undo);
      rows.push(row);
    }
    const r = this.report;
    info.textContent = !r ? "" : this.stale ? "The figure changed: find again for its new grades."
      : r.points.length ? `${r.points.length} thin part${r.points.length === 1 ? "" : "s"} at ${r.height_mm.toFixed(1)} mm tall, ` +
        `weakest first (${r.seconds.toFixed(0)} s)` : "No thin parts.";
    if (r && !this.stale) {
      r.points.forEach((w, i) => rows.push(this.item(w, i)));
    }
    list.replaceChildren(...rows);
  }

  private item(w: WeakPoint, i: number): HTMLElement {
    const el = document.createElement("div");
    el.className = "weak-item";
    const badge = document.createElement("span");
    badge.className = `grade grade-${w.grade}`;
    badge.textContent = w.grade;
    badge.title = "A holds ~2 kg pushed at its end, B ~800 g, C ~300 g, D ~100 g; F less, or under 0.8 mm across";
    const text = document.createElement("span");
    text.className = "weak-text";
    text.textContent = describeWeak(w);
    const showBtn = document.createElement("button");
    showBtn.className = "btn small";
    showBtn.textContent = this.shown === i ? "Hide" : "Show";
    showBtn.setAttribute("aria-pressed", String(this.shown === i));
    showBtn.title = "Tint the piece that would break off on the 3D view";
    showBtn.addEventListener("click", () => this.show(i));
    const sel = document.createElement("select");
    sel.title = "Thickness its weak stretch gets (where it is thinner): the rest of it stays as it is";
    sel.setAttribute("aria-label", "Thicken to");
    const pick = suggestThickness(w);
    for (const t of THICKNESSES) {
      const o = document.createElement("option");
      o.value = String(t);
      o.textContent = `${t} mm`;
      o.selected = t === pick;
      sel.append(o);
    }
    const go = document.createElement("button");
    go.className = "btn small";
    go.textContent = "Thicken";
    go.disabled = w.face < 0 || w.root < 0;
    go.addEventListener("click", () => this.thicken(w, +sel.value));
    const actions = document.createElement("span");
    actions.className = "weak-actions";
    actions.append(showBtn, sel, go);
    el.append(badge, text, actions);
    return el;
  }
}
