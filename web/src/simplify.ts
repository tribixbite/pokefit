/**
 * The Simplify control: Low / Medium / High levels (a share of the file's own triangle count, so a level means
 * the same on a 30k and a 1.4M mesh), a slider over that share on a log scale, and a box to type any triangle
 * count. `Params.simplify` holds either a share (0 < x < 1) or a count (>= 1,000); null = as in the file.
 */

/** The levels, as shares of the file's triangles. Low on the 1.45M-triangle MINI EEVEE is ~100k, the count
 *  that made its painting and analysis several times faster with the surface moving < 0.03 mm. */
export const LEVELS = [
  { name: "Low", share: 0.07 },
  { name: "Medium", share: 0.2 },
  { name: "High", share: 0.5 },
] as const;
/** The slider's left end (share); its right end is 1 = as in the file. */
export const MIN_SHARE = 0.01;
/** Slider positions (integer steps, for a smooth drag on a log scale). */
export const SLIDER_MAX = 1000;
/** Within this many slider steps of a level, the slider settles on the level. */
const SNAP = 15;
/** The engine's smallest triangle count (`Params.simplify`). */
export const MIN_TRIANGLES = 1000;

/** Slider position for a share (1 = as in the file). */
export function shareToSlider(share: number): number {
  const s = Math.min(1, Math.max(MIN_SHARE, share));
  return Math.round((SLIDER_MAX * Math.log(s / MIN_SHARE)) / Math.log(1 / MIN_SHARE));
}

/** Share for a slider position, settling on a level (or on "as in the file") when close to it. */
export function sliderToShare(pos: number): number {
  if (pos >= SLIDER_MAX - SNAP / 2) return 1;
  for (const l of LEVELS) if (Math.abs(pos - shareToSlider(l.share)) <= SNAP) return l.share;
  return Number((MIN_SHARE * Math.pow(1 / MIN_SHARE, pos / SLIDER_MAX)).toFixed(4));
}

/** The share a budget means on a file of `faces` triangles (1 = as in the file; unknown file size: a count
 *  shows as the right end). */
export function budgetShare(budget: number | null, faces: number | null): number {
  if (!budget) return 1;
  if (budget < 1) return budget;
  return faces ? Math.min(1, budget / faces) : 1;
}

/** Triangles a budget keeps on a file of `faces` triangles (null when the file size is unknown). */
export function budgetCount(budget: number | null, faces: number | null): number | null {
  if (!faces) return budget && budget >= 1 ? budget : null;
  if (!budget) return faces;
  return Math.min(faces, budget < 1 ? Math.max(1, Math.round(budget * faces)) : budget);
}

/** The budget for a slider position (null = as in the file). */
export function sliderBudget(pos: number): number | null {
  const share = sliderToShare(pos);
  return share >= 1 ? null : share;
}

/** The budget for a typed triangle count: the count itself (at least MIN_TRIANGLES), or null when it keeps
 *  every triangle of the file. */
export function countBudget(count: number, faces: number | null): number | null {
  if (!Number.isFinite(count) || count <= 0) return null;
  if (faces && count >= faces) return null;
  return Math.max(MIN_TRIANGLES, Math.round(count));
}

/** The level a budget is ("Off" for none), or null when it is in between. */
export function levelName(budget: number | null): string | null {
  if (!budget) return "Off";
  return LEVELS.find((l) => Math.abs(l.share - budget) < 1e-6)?.name ?? null;
}
