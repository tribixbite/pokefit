import { describe, expect, test } from "bun:test";
import { bounds, formatDuration, innerWallProfile, insideRings, interp, onSteel } from "./geometry";
import type { BallInfo } from "./types";

const ball = {
  r_inner: 22,
  ring_span: 9,
  ring_h: [0, 9],
  ring_r: [17, 18],
} as unknown as BallInfo;

describe("geometry", () => {
  test("interp clamps and interpolates", () => {
    expect(interp(-1, [0, 10], [1, 2])).toBe(1);
    expect(interp(5, [0, 10], [1, 2])).toBeCloseTo(1.5);
    expect(interp(20, [0, 10], [1, 2])).toBe(2);
  });

  test("inner wall is the ring near the equator, the sphere at the poles", () => {
    const p = innerWallProfile(ball, 44);
    const eq = p.find((q) => Math.abs(q.z) < 1e-9)!;
    expect(eq.r).toBeCloseTo(17);
    const pole = p[0];
    expect(pole.z).toBeCloseTo(-22);
    expect(pole.r).toBeCloseTo(0);
    const z15 = p.find((q) => Math.abs(q.z - 15) < 1e-9)!;
    expect(z15.r).toBeCloseTo(Math.sqrt(22 * 22 - 15 * 15));
  });

  test("formatDuration", () => {
    expect(formatDuration(45)).toBe("45 s");
    expect(formatDuration(1500)).toBe("25 min");
    expect(formatDuration(3851)).toBe("1h 04m");
    expect(formatDuration(NaN)).toBe("–");
  });

  test("onSteel", () => {
    const ring = { x: 1, y: 0, inner: 5, outer: 7 };
    expect(onSteel(7, 0, ring).ok).toBe(true);
    expect(onSteel(1, 0, ring).ok).toBe(false);
    expect(onSteel(9, 0, ring).ok).toBe(false);
  });

  test("insideRings handles holes", () => {
    const sq = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
    const hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]];
    expect(insideRings(2, 2, [sq, hole])).toBe(true);
    expect(insideRings(5, 5, [sq, hole])).toBe(false);
    expect(insideRings(12, 5, [sq, hole])).toBe(false);
  });

  test("bounds", () => {
    expect(bounds([[[0, 1], [2, 3]]], [[-1, 5]])).toEqual([-1, 1, 2, 5]);
    expect(bounds([])).toEqual([-10, -10, 10, 10]);
  });
});
