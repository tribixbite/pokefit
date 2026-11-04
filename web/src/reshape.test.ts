import { describe, expect, it } from "bun:test";
import { describeLimb, withLimb } from "./reshape";
import type { LimbOp } from "./types";

const op = (o: Partial<LimbOp>): LimbOp => ({ part: 0, face: 5, angle: 15, scale: 1, bend: 0, swing: 0, ...o });

describe("reshaped limbs", () => {
  it("describes a change briefly", () => {
    expect(describeLimb(op({ scale: 1.2, bend: 30 }))).toBe("120 %, bend +30°");
    expect(describeLimb(op({ swing: -15 }))).toBe("swing -15°");
    expect(describeLimb(op({}))).toBe("unchanged");
  });

  it("adds, replaces and drops a limb's step", () => {
    const a = op({ bend: 20 });
    let ops = withLimb([], -1, a);
    expect(ops).toEqual([a]);
    ops = withLimb(ops, 0, { ...a, bend: 40 });
    expect(ops).toEqual([{ ...a, bend: 40 }]);
    expect(withLimb(ops, -1, op({}))).toEqual(ops);          // a new limb left as it is adds nothing
    expect(withLimb(ops, 0, op({}))).toEqual([]);            // put back as it was: the step goes
  });
});
