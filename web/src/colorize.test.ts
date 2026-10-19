import { describe, expect, test } from "bun:test";
import { bestValueCount } from "./colorize";

describe("palette cost", () => {
  test("best value is the count before the biggest jump", () => {
    const rows = [0, 0, 29, 159, 180, 287, 287].map((changes, i) => ({ count: i + 1, changes }));
    expect(bestValueCount(rows)).toBe(3);
    expect(bestValueCount([{ count: 1, changes: 0 }, { count: 2, changes: 10 }])).toBeNull();
  });
});
