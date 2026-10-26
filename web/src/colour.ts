/** Colour helpers shared by the Colours card and the slot pickers. */

/** Relative luminance (0–1) of a #RRGGBB colour, to pick readable text on a swatch. */
export function luminance(hex: string): number {
  const v = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}

/** Text colour that reads on a swatch of `hex`. */
export function textOn(hex: string): string {
  return luminance(hex) > 0.55 ? "#111" : "#fff";
}

/** sRGB hex (#RRGGBB[AA]) -> CIELAB (D65); the same formula as pokefit/remap.py `_lab`. */
function lab(hex: string): [number, number, number] {
  const c = hex.replace(/^#/, "").slice(0, 6).padEnd(6, "0");
  const lin = [0, 2, 4].map((i) => {
    const v = parseInt(c.slice(i, i + 2), 16) / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  const xyz = [
    (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047,
    0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2],
    (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883,
  ];
  const f = xyz.map((t) => (t > (6 / 29) ** 3 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29));
  return [116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])];
}

/** CIE76 colour difference, as the engine reports it (`SlotMatch.delta_e`). */
export function deltaE(a: string, b: string): number {
  const p = lab(a), q = lab(b);
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

/** How well a slot's colour matches a figure colour; the thresholds follow pokefit/remap.py
 * (above 10 the import table flags it, above NEW_SLOT_DELTA_E = 25 no slot counts as the same colour). */
export type MatchLevel = "good" | "close" | "far";

export function matchLevel(de: number): MatchLevel {
  return de <= 10 ? "good" : de <= 25 ? "close" : "far";
}

/** The candidate that stands out most from every colour in `palette` (largest smallest ΔE), for a new filament:
 * a colour only slightly unlike one already in use (near-black beside black) would hide what it marks. */
export function distinctColour(palette: string[], candidates: readonly string[]): string {
  let best = candidates[0];
  let score = -1;
  for (const c of candidates) {
    const d = palette.length ? Math.min(...palette.map((p) => deltaE(c, p))) : Infinity;
    if (d > score) { best = c; score = d; }
  }
  return best;
}
