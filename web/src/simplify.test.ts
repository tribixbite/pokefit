import { describe, expect, it } from "bun:test";
import {
  budgetCount, budgetShare, countBudget, LEVELS, levelName, MIN_SHARE, MIN_TRIANGLES, shareToSlider,
  SLIDER_MAX, sliderBudget, sliderToShare,
} from "./simplify";

describe("simplify control", () => {
  it("maps shares to slider positions on a log scale and back", () => {
    expect(shareToSlider(MIN_SHARE)).toBe(0);
    expect(shareToSlider(1)).toBe(SLIDER_MAX);
    expect(shareToSlider(0.1)).toBe(SLIDER_MAX / 2);            // halfway between 1 % and 100 %
    expect(sliderToShare(SLIDER_MAX / 4)).toBeCloseTo(Math.sqrt(0.1) / 10, 4);
  });

  it("settles on a level or on 'as in the file' near it, and nowhere else", () => {
    for (const l of LEVELS) {
      const at = shareToSlider(l.share);
      expect(sliderToShare(at + 10)).toBe(l.share);
      expect(sliderToShare(at - 10)).toBe(l.share);
      expect(levelName(sliderBudget(at))).toBe(l.name);
    }
    expect(sliderBudget(SLIDER_MAX)).toBeNull();
    expect(sliderBudget(SLIDER_MAX - 3)).toBeNull();
    expect(levelName(sliderBudget(SLIDER_MAX))).toBe("Off");
    expect(levelName(sliderBudget(100))).toBeNull();
  });

  it("reads a budget as a share and a count of the file's triangles", () => {
    expect(budgetShare(0.2, 1_000_000)).toBe(0.2);
    expect(budgetShare(100_000, 1_000_000)).toBe(0.1);
    expect(budgetShare(2_000_000, 1_000_000)).toBe(1);           // more than the file has: as in the file
    expect(budgetShare(null, 1_000_000)).toBe(1);
    expect(budgetCount(0.2, 1_445_902)).toBe(289_180);
    expect(budgetCount(100_000, 1_445_902)).toBe(100_000);
    expect(budgetCount(null, 6_762)).toBe(6_762);
    expect(budgetCount(0.2, null)).toBeNull();                  // no file open yet
  });

  it("turns a typed count into a budget", () => {
    expect(countBudget(123_456.4, 1_445_902)).toBe(123_456);
    expect(countBudget(10, 1_445_902)).toBe(MIN_TRIANGLES);
    expect(countBudget(1_445_902, 1_445_902)).toBeNull();
    expect(countBudget(Number.NaN, 1_445_902)).toBeNull();
    expect(countBudget(50_000, null)).toBe(50_000);
  });
});
