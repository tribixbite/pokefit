import { describe, expect, it } from "bun:test";
import { deltaE, matchLevel, textOn } from "./colour";

describe("colour", () => {
  it("matches the engine's CIE76 ΔE (pokefit/remap.py delta_e)", () => {
    // Reference values printed by pokefit.remap.delta_e.
    expect(deltaE("#FF0000", "#00FF00")).toBeCloseTo(170.5842, 3);
    expect(deltaE("#F98C36", "#FF8C00")).toBeCloseTo(14.5971, 3);
    expect(deltaE("#161616", "#000000")).toBeCloseTo(7.2473, 3);
    expect(deltaE("#123456", "#123456ff")).toBe(0);           // alpha is ignored
  });

  it("grades matches with the engine's thresholds", () => {
    expect(matchLevel(4)).toBe("good");
    expect(matchLevel(10)).toBe("good");
    expect(matchLevel(18)).toBe("close");
    expect(matchLevel(25.1)).toBe("far");
  });

  it("picks readable text on light and dark swatches", () => {
    expect(textOn("#FFFFFF")).toBe("#111");
    expect(textOn("#161616")).toBe("#fff");
  });
});
