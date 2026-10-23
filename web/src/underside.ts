/**
 * Underside view (looking up at the figure's base, figure frame, final mm) with
 * draggable magnet pockets. Shows the underside heights (dark = on the bed,
 * lighter = higher), the outline of what touches the bed, where a pocket centre
 * keeps its walls (green), and the washer's steel ring as the ball will hold it.
 * The gap fill's circle can be dragged, and its outline drawn freehand (`setDrawing`).
 */
import { bounds, insideRings, onSteel } from "./geometry";
import type { Underside } from "./types";

const NS = "http://www.w3.org/2000/svg";
/** `dragging` value for the fill circle (pocket indices are >= 0). */
const CIRCLE = -2;

export type Pocket = { x: number; y: number };
/** The flat base's fill circle as drawn: centre and radius in final mm (figure frame). */
export type FillCircle = { x: number; y: number; r: number };
/** A drawn outline's points are kept at least this far apart (mm). */
const DRAW_STEP = 0.25;
/** ...and at most this many (the engine's limit is 400). */
const MAX_POINTS = 300;

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function ringsPath(rings: number[][][]): string {
  return rings.map((r) => "M" + r.map((p) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`).join("L") + "Z").join("");
}

export class UndersideEditor {
  readonly svg: SVGSVGElement;
  private data: Underside | null = null;
  private pockets: Pocket[] = [];
  private circle: FillCircle | null = null;
  private shape: Pocket[] | null = null;
  /** Pocket index being dragged, CIRCLE for the fill circle, -1 = none. */
  private dragging = -1;
  /** Drawing the fill outline: the next drag on the view draws it. */
  private drawMode = false;
  /** The outline being drawn (figure frame, final mm) and its line on screen. */
  private drawn: Pocket[] | null = null;
  private drawLine: SVGPolylineElement | null = null;
  private group: SVGGElement | null = null;

  /** `onChange` fires when a pocket drag ends, with all pocket positions; `onCircle` when the fill circle moved;
   * `onShape` when a fill outline was drawn (final mm, figure frame). */
  constructor(host: HTMLElement, private onChange: (p: Pocket[]) => void,
              private onCircle: (c: Pocket) => void = () => undefined,
              private onShape: (points: Pocket[]) => void = () => undefined) {
    this.svg = el("svg", { class: "underside", role: "img", "aria-label": "Figure underside with magnet pockets" });
    host.appendChild(this.svg);
    this.svg.addEventListener("pointerdown", (e) => this.startDraw(e));
    this.svg.addEventListener("pointermove", (e) => this.move(e));
    this.svg.addEventListener("pointerup", (e) => this.drop(e));
    this.svg.addEventListener("pointercancel", (e) => this.drop(e));
  }

  /** Draw mode on/off: a drag on the view then draws the gap fill's outline (released = closed). */
  setDrawing(on: boolean): void {
    this.drawMode = on;
    this.svg.classList.toggle("drawing", on);
  }

  get drawing(): boolean {
    return this.drawMode;
  }

  render(data: Underside, pockets: Pocket[], circle: FillCircle | null = null, shape: Pocket[] | null = null): void {
    this.data = data;
    this.pockets = pockets.map((p) => ({ ...p }));
    this.circle = circle ? { ...circle } : null;
    this.shape = shape ? shape.map((p) => ({ ...p })) : null;
    const s = this.svg;
    s.replaceChildren();
    const ring = data.ring;
    const extra = [
      [ring.x - ring.outer, ring.y - ring.outer],
      [ring.x + ring.outer, ring.y + ring.outer],
      ...pockets.map((p) => [p.x, p.y]),
      ...(circle ? [[circle.x - circle.r, circle.y - circle.r], [circle.x + circle.r, circle.y + circle.r]] : []),
      ...(shape ?? []).map((p) => [p.x, p.y]),
    ];
    const ext = data.extent;
    const [x0, y0, x1, y1] = bounds(data.feet, [...extra, [ext[0], ext[1]], [ext[2], ext[3]]]);
    const pad = 2;
    // SVG y grows downward; flip so +y is up. This is the view from below,
    // so mirror x as well: what is on the figure's right appears on the left.
    s.setAttribute("viewBox", `${-(x1 + pad)} ${-(y1 + pad)} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`);
    const g = el("g", { transform: "scale(-1,-1)" });
    s.appendChild(g);
    this.group = g;

    // Height image: row 0 is the grid's lowest y, which this mirrored group draws at the bottom.
    g.appendChild(el("image", { href: data.height_png, x: ext[0], y: ext[1], width: ext[2] - ext[0],
      height: ext[3] - ext[1], preserveAspectRatio: "none", class: "u-height" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: ring.washer_od / 2, class: "u-washer" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: ring.washer_id / 2, class: "u-hole" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: (ring.inner + ring.outer) / 2, class: "u-band",
      "stroke-width": ring.outer - ring.inner }));
    g.appendChild(el("path", { d: ringsPath(data.feet), class: "u-feet", "fill-rule": "evenodd" }));
    // What the gap fill added: its own colour, so a fill or circle change is visible here at once.
    if (data.fill?.length) g.appendChild(el("path", { d: ringsPath(data.fill), class: "u-base", "fill-rule": "evenodd" }));
    if (data.valid.length) g.appendChild(el("path", { d: ringsPath(data.valid), class: "u-valid", "fill-rule": "evenodd" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: 0.6, class: "u-centre" }));
    if (this.circle) {
      const c = el("circle", { cx: this.circle.x, cy: this.circle.y, r: this.circle.r, class: "u-fill", tabindex: 0,
        "data-fill": 1, "aria-label": "Flat base fill circle: drag to move" });
      c.addEventListener("pointerdown", (e) => {
        this.dragging = CIRCLE;
        (e.target as Element).setPointerCapture(e.pointerId);
        e.preventDefault();
      });
      c.addEventListener("keydown", (e) => this.nudge(e, CIRCLE));
      g.appendChild(c);
    }

    if (this.shape) {
      g.appendChild(el("polygon", { points: this.shape.map((p) => `${p.x},${p.y}`).join(" "), class: "u-fill u-shape",
        "aria-label": "Gap fill outline" }));
    }

    this.pockets.forEach((p, i) => {
      const ok = this.pocketOk(p);
      const c = el("circle", { cx: p.x, cy: p.y, r: data.pocket_d / 2, class: `u-pocket ${ok ? "ok" : "bad"}`,
        tabindex: 0, "data-i": i });
      c.addEventListener("pointerdown", (e) => {
        this.dragging = i;
        (e.target as Element).setPointerCapture(e.pointerId);
        e.preventDefault();
      });
      c.addEventListener("keydown", (e) => this.nudge(e, i));
      g.appendChild(c);
    });
    // Pocket numbers (outside the mirrored group so the digits read normally).
    this.pockets.forEach((p, i) => {
      const t = el("text", { x: -p.x, y: -p.y, class: "u-label", "data-l": i, "font-size": data.pocket_d * 0.55,
        "text-anchor": "middle", "dominant-baseline": "central" });
      t.textContent = String(i + 1);
      s.appendChild(t);
    });
  }

  private pocketOk(p: Pocket): boolean {
    if (!this.data) return false;
    const inValid = this.data.valid.length === 0 || insideRings(p.x, p.y, this.data.valid);
    return inValid && onSteel(p.x, p.y, this.data.ring).ok;
  }

  private toFigure(e: PointerEvent): Pocket {
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(this.svg.getScreenCTM()!.inverse());
    return { x: -pt.x, y: -pt.y };
  }

  /** In draw mode, a press on the view (not on a pocket or the circle) starts the outline. */
  private startDraw(e: PointerEvent): void {
    if (!this.drawMode || this.dragging !== -1 || !this.group) return;
    this.drawn = [this.toFigure(e)];
    this.drawLine = el("polyline", { points: "", class: "u-fill u-drawing" });
    this.group.appendChild(this.drawLine);
    try { this.svg.setPointerCapture(e.pointerId); } catch { /* synthetic events have no capturable pointer */ }
    e.preventDefault();
  }

  private move(e: PointerEvent): void {
    if (this.drawn && this.drawLine) {
      const p = this.toFigure(e);
      const last = this.drawn[this.drawn.length - 1];
      if (Math.hypot(p.x - last.x, p.y - last.y) >= DRAW_STEP) {
        this.drawn.push(p);
        this.drawLine.setAttribute("points", this.drawn.map((q) => `${q.x},${q.y}`).join(" "));
      }
      return;
    }
    if (this.dragging === CIRCLE && this.circle) {
      const p = this.toFigure(e);
      this.circle = { ...this.circle, x: p.x, y: p.y };
      const c = this.svg.querySelector<SVGCircleElement>("circle[data-fill]");
      c?.setAttribute("cx", String(p.x));
      c?.setAttribute("cy", String(p.y));
      return;
    }
    if (this.dragging < 0 || !this.data) return;
    this.pockets[this.dragging] = this.toFigure(e);
    const c = this.svg.querySelector<SVGCircleElement>(`circle[data-i="${this.dragging}"]`);
    if (c) {
      const p = this.pockets[this.dragging];
      c.setAttribute("cx", String(p.x));
      c.setAttribute("cy", String(p.y));
      c.setAttribute("class", `u-pocket ${this.pocketOk(p) ? "ok" : "bad"}`);
    }
    const t = this.svg.querySelector<SVGTextElement>(`text[data-l="${this.dragging}"]`);
    if (t) {
      t.setAttribute("x", String(-this.pockets[this.dragging].x));
      t.setAttribute("y", String(-this.pockets[this.dragging].y));
    }
  }

  private drop(e: PointerEvent): void {
    if (this.drawn) {
      const pts = this.drawn;
      this.drawn = null;
      if (e.type === "pointerup") pts.push(this.toFigure(e));
      this.drawLine?.remove();
      this.drawLine = null;
      if (pts.length >= 3 && Math.abs(polygonArea(pts)) >= 1) {
        this.setDrawing(false);
        this.onShape(thin(pts, MAX_POINTS).map((p) => ({ x: +p.x.toFixed(3), y: +p.y.toFixed(3) })));
      }
      return;
    }
    if (this.dragging === CIRCLE && this.circle) {
      this.dragging = -1;
      this.onCircle({ x: +this.circle.x.toFixed(3), y: +this.circle.y.toFixed(3) });
      return;
    }
    if (this.dragging < 0) return;
    this.dragging = -1;
    this.onChange(this.pockets.map((p) => ({ x: +p.x.toFixed(3), y: +p.y.toFixed(3) })));
  }

  /** Arrow keys move a focused pocket (or the fill circle) by 0.25 mm (in the underside view's directions). */
  private nudge(e: KeyboardEvent, i: number): void {
    const step = e.shiftKey ? 1 : 0.25;
    const d: Record<string, [number, number]> = {
      ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
    };
    const v = d[e.key];
    if (!v) return;
    e.preventDefault();
    if (i === CIRCLE) {
      if (this.circle) this.onCircle({ x: +(this.circle.x + v[0]).toFixed(3), y: +(this.circle.y + v[1]).toFixed(3) });
      return;
    }
    this.pockets[i] = { x: this.pockets[i].x + v[0], y: this.pockets[i].y + v[1] };
    this.onChange(this.pockets.map((p) => ({ x: +p.x.toFixed(3), y: +p.y.toFixed(3) })));
  }
}

/** Shoelace area of a closed outline (mm²; the sign gives its direction). */
export function polygonArea(pts: Pocket[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

/** At most `n` of `pts`, evenly spaced along the list (a long slow drag records many). */
export function thin(pts: Pocket[], n: number): Pocket[] {
  if (pts.length <= n) return pts;
  return Array.from({ length: n }, (_, i) => pts[Math.floor((i * pts.length) / n)]);
}
