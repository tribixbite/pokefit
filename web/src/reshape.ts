/**
 * "Reshape a part" (Figure card): tap a tail, ear, leg or arm on the 3D view, then resize it or bend it at the joint
 * where it meets the body (the engine's reshape.py; the limb is found like paint's Part mode). Each reshaped limb is
 * a `LimbOp` in `Params.limbs`, named by the tapped triangle like a paint step; the view highlights the limb being
 * changed. Sliders apply on release (the figure is re-fitted then), so a drag shows its value without waiting.
 */
import type { FigureHit } from "./scene";
import type { LimbOp, LimbRegion, Params } from "./types";

export interface ReshapeHost {
  params(): Params;
  /** The limb under a tap, or at a known triangle (a list entry). */
  limb(req: { hit: FigureHit } | { part: number; face: number }, angle: number): Promise<LimbRegion>;
  /** Turn tap-to-pick on the 3D view on or off. */
  setPicking(fn: ((hit: FigureHit) => void) | null): void;
  /** Highlight a limb's triangles on the view (null: none). */
  highlight(part: number, faces: number[] | null): void;
  /** Stop the other tools that take taps (painting). */
  stopOthers(): void;
  /** The shape changed: analyse again. */
  apply(): void;
  toast(msg: string): void;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const DEFAULT_ANGLE = 15;
/** Base widening defaults (reshape.py `LimbOp`): fades over half the part, linearly. */
const DEFAULT_REACH = 0.5;
const DEFAULT_STEEP = 1;

/** A short description of a change ("120 %, bend 30°"). */
export function describeLimb(op: LimbOp): string {
  const bits: string[] = [];
  if (op.scale !== 1) bits.push(`${Math.round(op.scale * 100)} %`);
  if (op.bend) bits.push(`bend ${op.bend > 0 ? "+" : ""}${op.bend}°`);
  if (op.swing) bits.push(`swing ${op.swing > 0 ? "+" : ""}${op.swing}°`);
  if ((op.thicken ?? 1) !== 1) bits.push(`base ×${+(op.thicken ?? 1).toFixed(2)}`);
  return bits.join(", ") || "unchanged";
}

/** `ops` with `op` put in place of the one naming the same limb (`index`), or added; dropped when it changes nothing. */
export function withLimb(ops: LimbOp[], index: number, op: LimbOp): LimbOp[] {
  const out = [...ops];
  const changes = op.scale !== 1 || op.bend !== 0 || op.swing !== 0 || (op.thicken ?? 1) !== 1;
  if (index >= 0 && index < out.length) {
    if (changes) out[index] = op;
    else out.splice(index, 1);
  } else if (changes) out.push(op);
  return out;
}

export class Reshape {
  private picking = false;
  private busy = false;
  /** The limb being edited: its region and its place in `params.limbs` (-1: not reshaped yet). */
  private current: { region: LimbRegion; index: number } | null = null;
  /** Base widening fade for a part not widened yet (a widened one keeps its own in its `LimbOp`). */
  private fade = { reach: DEFAULT_REACH, steep: DEFAULT_STEEP };

  constructor(private host: ReshapeHost) {
    $("limbPick").addEventListener("click", () => this.setPicking(!this.picking));
    $("limbDone").addEventListener("click", () => this.close());
    $("limbReset").addEventListener("click", () => this.change({ scale: 1, bend: 0, swing: 0, thicken: 1 }));
    const sliders: [string, keyof LimbOp, (v: number) => number][] = [
      ["limbScale", "scale", (v) => v / 100], ["limbBend", "bend", (v) => v], ["limbSwing", "swing", (v) => v],
      ["limbThicken", "thicken", (v) => v / 100], ["limbReach", "reach", (v) => v / 100],
      ["limbSteep", "steep", (v) => v],
    ];
    for (const [id, key, conv] of sliders) {
      const el = $<HTMLInputElement>(id);
      el.addEventListener("input", () => this.showValues());
      el.addEventListener("change", () => this.change({ [key]: conv(+el.value) }));
    }
    const angle = $<HTMLInputElement>("limbAngle");
    angle.addEventListener("input", () => this.showValues());
    angle.addEventListener("change", () => void this.reangle(+angle.value));
    this.render();
  }

  /** Stop picking (another tool took the taps). */
  stop(): void {
    if (!this.picking) return;
    this.picking = false;
    this.render();
  }

  /** Forget the edit in progress (a new file or figure: triangle numbers differ). */
  reset(): void {
    this.picking = false;
    this.current = null;
    this.host.highlight(0, null);
    this.host.setPicking(null);
    this.render();
  }

  private setPicking(on: boolean): void {
    if (on) this.host.stopOthers();
    this.picking = on;
    this.host.setPicking(on ? (hit) => void this.pickAt(hit) : null);
    this.render();
  }

  private async pickAt(hit: FigureHit): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    $("limbInfo").textContent = "Finding the part…";
    try {
      const ops = this.host.params().limbs ?? [];
      const region = await this.host.limb({ hit }, +$<HTMLInputElement>("limbAngle").value || DEFAULT_ANGLE);
      // Tapping a limb already reshaped edits that one (its own triangle names it, and its own joint angle).
      const index = ops.findIndex((op) => op.part === region.part && region.faces.includes(op.face));
      const found = index >= 0 && ops[index].angle !== region.angle
        ? await this.host.limb({ part: ops[index].part, face: ops[index].face }, ops[index].angle) : region;
      if (index < 0) $<HTMLInputElement>("limbAngle").value = String(found.angle);   // a smaller one, if it had to
      this.open(found, index);
    } catch (e) {
      this.host.toast((e as Error).message);
      $("limbInfo").textContent = "";
    } finally {
      this.busy = false;
    }
  }

  private open(region: LimbRegion, index: number): void {
    this.current = { region, index };
    this.host.highlight(region.part, region.view_faces);
    const op = this.op();
    $<HTMLInputElement>("limbScale").value = String(Math.round(op.scale * 100));
    $<HTMLInputElement>("limbBend").value = String(op.bend);
    $<HTMLInputElement>("limbSwing").value = String(op.swing);
    $<HTMLInputElement>("limbThicken").value = String(Math.round((op.thicken ?? 1) * 100));
    $<HTMLInputElement>("limbReach").value = String(Math.round((op.reach ?? DEFAULT_REACH) * 100));
    $<HTMLInputElement>("limbSteep").value = String(op.steep ?? DEFAULT_STEEP);
    $<HTMLInputElement>("limbAngle").value = String(op.angle);
    this.render();
  }

  /** The operation being edited, as stored (or unchanged, for a limb not reshaped yet). */
  private op(): LimbOp {
    const c = this.current;
    const fade = { reach: this.fade.reach, steep: this.fade.steep };
    if (!c) return { part: 0, face: 0, angle: DEFAULT_ANGLE, scale: 1, bend: 0, swing: 0, thicken: 1, ...fade };
    const ops = this.host.params().limbs ?? [];
    return c.index >= 0 && ops[c.index]
      ? ops[c.index]
      : { part: c.region.part, face: c.region.face, angle: c.region.angle, scale: 1, bend: 0, swing: 0, thicken: 1,
          ...fade };
  }

  private change(delta: Partial<LimbOp>): void {
    const c = this.current;
    if (!c) return;
    const p = this.host.params();
    const ops = p.limbs ?? [];
    const next = { ...this.op(), ...delta };
    // Reach and taper are kept for a part not widened yet (set them first, then the base width).
    if (delta.reach !== undefined) this.fade.reach = delta.reach;
    if (delta.steep !== undefined) this.fade.steep = delta.steep;
    const out = withLimb(ops, c.index, next);
    // Where it now sits in the list (added at the end, kept, or gone).
    c.index = out.length > ops.length ? out.length - 1 : out.length < ops.length ? -1 : c.index;
    const same = JSON.stringify(out) === JSON.stringify(ops);
    p.limbs = out.length ? out : null;
    this.open(c.region, c.index);
    if (same) return;               // a fade setting on a part not widened: the shape is the same
    p.figure_scale = null;          // the largest that fits changes with the shape
    p.pockets = null;
    this.host.apply();
  }

  /** A new joint angle: the limb is found again from the same triangle (it may now end at another valley). */
  private async reangle(angle: number): Promise<void> {
    const c = this.current;
    if (!c || this.busy) return;          // nothing picked yet: the next tap uses the new angle
    this.busy = true;
    try {
      const op = this.op();
      const region = await this.host.limb({ part: op.part, face: op.face }, angle);
      this.current = { region, index: c.index };
      if (c.index >= 0) this.change({ angle });
      else this.open(region, -1);
    } catch (e) {
      this.host.toast((e as Error).message);
      $<HTMLInputElement>("limbAngle").value = String(c.region.angle);
    } finally {
      this.busy = false;
    }
  }

  private close(): void {
    this.current = null;
    this.host.highlight(0, null);
    this.setPicking(false);
  }

  private showValues(): void {
    $("limbScaleVal").textContent = `${$<HTMLInputElement>("limbScale").value} %`;
    $("limbBendVal").textContent = `${$<HTMLInputElement>("limbBend").value}°`;
    $("limbSwingVal").textContent = `${$<HTMLInputElement>("limbSwing").value}°`;
    $("limbThickenVal").textContent = `×${(+$<HTMLInputElement>("limbThicken").value / 100).toFixed(2)}`;
    $("limbReachVal").textContent = `${$<HTMLInputElement>("limbReach").value} %`;
    $("limbSteepVal").textContent = (+$<HTMLInputElement>("limbSteep").value).toFixed(2);
    $("limbAngleVal").textContent = `${$<HTMLInputElement>("limbAngle").value}°`;
  }

  render(): void {
    const btn = $<HTMLButtonElement>("limbPick");
    btn.setAttribute("aria-pressed", String(this.picking));
    btn.textContent = this.picking ? "Tap a part…" : "Pick a part";
    const c = this.current;
    $("limbEditor").hidden = !c;
    $("limbAngleRow").hidden = !c && !this.picking;
    if (c) {
      $("limbInfo").textContent = `${c.region.joined ? "From its joint" : "A separate piece"}: ` +
        `${c.region.length_mm.toFixed(1)} mm long, ${Math.round(c.region.share * 100)} % of the surface` +
        (c.index >= 0 ? ` · ${describeLimb(this.op())}` : "");
    } else if (!this.busy) {
      $("limbInfo").textContent = this.picking ? "Tap a tail, ear, leg or arm on the 3D view." : "";
    }
    this.showValues();
    const ops = this.host.params().limbs ?? [];
    $("limbList").replaceChildren(...ops.map((op, i) => {
      const row = document.createElement("div");
      row.className = "limb-row";
      const edit = document.createElement("button");
      edit.className = "linkbtn";
      edit.textContent = `Part ${i + 1}: ${describeLimb(op)}`;
      edit.title = "Edit this part";
      edit.addEventListener("click", () => void this.host.limb({ part: op.part, face: op.face }, op.angle)
        .then((r) => this.open(r, i)).catch((e: Error) => this.host.toast(e.message)));
      const del = document.createElement("button");
      del.className = "linkbtn";
      del.textContent = "×";
      del.title = "Put this part back as it was";
      del.setAttribute("aria-label", del.title);
      del.addEventListener("click", () => {
        const p = this.host.params();
        const rest = ops.filter((_, k) => k !== i);
        p.limbs = rest.length ? rest : null;
        p.figure_scale = null;
        p.pockets = null;
        if (this.current) this.close();
        this.render();
        this.host.apply();
      });
      row.append(edit, del);
      return row;
    }));
  }
}
