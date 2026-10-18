/**
 * Underside view (looking up at the figure's base, figure frame, final mm) with
 * draggable magnet pockets. Shows the foot outlines, where a pocket centre keeps
 * its walls (green), and the washer's steel ring as the ball will hold it.
 */
import { bounds, insideRings, onSteel } from "./geometry";
import type { Underside } from "./types";

const NS = "http://www.w3.org/2000/svg";

export type Pocket = { x: number; y: number };

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
  private dragging = -1;

  /** `onChange` fires when a drag ends, with all pocket positions. */
  constructor(host: HTMLElement, private onChange: (p: Pocket[]) => void) {
    this.svg = el("svg", { class: "underside", role: "img", "aria-label": "Figure underside with magnet pockets" });
    host.appendChild(this.svg);
    this.svg.addEventListener("pointermove", (e) => this.move(e));
    this.svg.addEventListener("pointerup", (e) => this.drop(e));
    this.svg.addEventListener("pointercancel", (e) => this.drop(e));
  }

  render(data: Underside, pockets: Pocket[]): void {
    this.data = data;
    this.pockets = pockets.map((p) => ({ ...p }));
    const s = this.svg;
    s.replaceChildren();
    const ring = data.ring;
    const extra = [
      [ring.x - ring.outer, ring.y - ring.outer],
      [ring.x + ring.outer, ring.y + ring.outer],
      ...pockets.map((p) => [p.x, p.y]),
    ];
    const [x0, y0, x1, y1] = bounds(data.feet, extra);
    const pad = 2;
    // SVG y grows downward; flip so +y is up. This is the view from below,
    // so mirror x as well: what is on the figure's right appears on the left.
    s.setAttribute("viewBox", `${-(x1 + pad)} ${-(y1 + pad)} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`);
    const g = el("g", { transform: "scale(-1,-1)" });
    s.appendChild(g);

    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: ring.washer_od / 2, class: "u-washer" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: ring.washer_id / 2, class: "u-hole" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: (ring.inner + ring.outer) / 2, class: "u-band",
      "stroke-width": ring.outer - ring.inner }));
    g.appendChild(el("path", { d: ringsPath(data.feet), class: "u-feet", "fill-rule": "evenodd" }));
    if (data.valid.length) g.appendChild(el("path", { d: ringsPath(data.valid), class: "u-valid", "fill-rule": "evenodd" }));
    g.appendChild(el("circle", { cx: ring.x, cy: ring.y, r: 0.6, class: "u-centre" }));

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

  private move(e: PointerEvent): void {
    if (this.dragging < 0 || !this.data) return;
    this.pockets[this.dragging] = this.toFigure(e);
    const c = this.svg.querySelector<SVGCircleElement>(`circle[data-i="${this.dragging}"]`);
    if (c) {
      const p = this.pockets[this.dragging];
      c.setAttribute("cx", String(p.x));
      c.setAttribute("cy", String(p.y));
      c.setAttribute("class", `u-pocket ${this.pocketOk(p) ? "ok" : "bad"}`);
    }
  }

  private drop(_e: PointerEvent): void {
    if (this.dragging < 0) return;
    this.dragging = -1;
    this.onChange(this.pockets.map((p) => ({ x: +p.x.toFixed(3), y: +p.y.toFixed(3) })));
  }

  /** Arrow keys move a focused pocket by 0.25 mm (in the underside view's directions). */
  private nudge(e: KeyboardEvent, i: number): void {
    const step = e.shiftKey ? 1 : 0.25;
    const d: Record<string, [number, number]> = {
      ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
    };
    const v = d[e.key];
    if (!v) return;
    e.preventDefault();
    this.pockets[i] = { x: this.pockets[i].x + v[0], y: this.pockets[i].y + v[1] };
    this.onChange(this.pockets.map((p) => ({ x: +p.x.toFixed(3), y: +p.y.toFixed(3) })));
  }
}
