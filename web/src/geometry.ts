/** Pure helpers (no three.js / DOM) so they can be unit-tested with `bun test`. */
import type { BallInfo } from "./types";

export interface RZ {
  r: number;
  z: number;
}

/**
 * Inner surface of the assembled ball: the shell's inner sphere, narrowed by the
 * connector rings within `ring_span` of the equator. Ball frame, final mm.
 */
export function innerWallProfile(ball: BallInfo, steps = 160): RZ[] {
  const ri = ball.r_inner;
  const pts: RZ[] = [];
  for (let i = 0; i <= steps; i++) {
    const z = -ri + (2 * ri * i) / steps;
    let r = Math.sqrt(Math.max(ri * ri - z * z, 0));
    const h = Math.abs(z);
    if (h <= ball.ring_span) r = Math.min(r, interp(h, ball.ring_h, ball.ring_r));
    pts.push({ r, z });
  }
  return pts;
}

/** The space the figure may use (clearance applied), floor to top, closed on the axis. */
export function allowedProfile(ball: BallInfo): RZ[] {
  const pts: RZ[] = [{ r: 0, z: ball.floor_z }];
  ball.profile_z.forEach((z, i) => pts.push({ r: Math.max(ball.profile_r[i], 0), z }));
  pts.push({ r: 0, z: ball.top_z });
  return pts;
}

/** Linear interpolation with clamping, xs ascending. */
export function interp(x: number, xs: number[], ys: number[]): number {
  if (xs.length === 0) return Infinity;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

/** "1h 04m", "25 min", "45 s". */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const totalMin = Math.round(seconds / 60);
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m} min`;
}

export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Distance from (x, y) to the ring centre, and whether a pocket there is fully on steel. */
export function onSteel(
  x: number,
  y: number,
  ring: { x: number; y: number; inner: number; outer: number },
): { r: number; ok: boolean } {
  const r = Math.hypot(x - ring.x, y - ring.y);
  return { r, ok: r >= ring.inner - 1e-6 && r <= ring.outer + 1e-6 };
}

/** Even-odd point-in-polygon over a set of rings (exteriors and holes alike). */
export function insideRings(x: number, y: number, rings: number[][][]): boolean {
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** Bounding box of rings plus extra points: [minX, minY, maxX, maxY]. */
export function bounds(rings: number[][][], extra: number[][] = []): [number, number, number, number] {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of [...rings.flat(), ...extra]) {
    x0 = Math.min(x0, p[0]);
    y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]);
    y1 = Math.max(y1, p[1]);
  }
  if (!Number.isFinite(x0)) return [-10, -10, 10, 10];
  return [x0, y0, x1, y1];
}
