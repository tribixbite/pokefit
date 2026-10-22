import { describe, expect, it } from "bun:test";
import { base64Bytes, groupFaces } from "./faces";

describe("preview face colours", () => {
  it("decodes base64 bytes", () => {
    expect([...base64Bytes(btoa(String.fromCharCode(0, 2, 255)))]).toEqual([0, 2, 255]);
  });

  it("groups triangles by colour, keeping their corners and order within a colour", () => {
    // Four triangles coloured 1, 0, 1, 2 (three colours).
    const order = new Uint16Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    const { index, groups } = groupFaces(order, new Uint8Array([1, 0, 1, 2]), 3);
    expect(index).toBeInstanceOf(Uint16Array);
    expect([...index]).toEqual([3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]);
    expect(groups).toEqual([
      { start: 0, count: 3, materialIndex: 0 },
      { start: 3, count: 6, materialIndex: 1 },
      { start: 9, count: 3, materialIndex: 2 },
    ]);
  });

  it("drops faces whose colour index has no material and skips empty colours", () => {
    const order = new Uint32Array([0, 1, 2, 3, 4, 5]);
    const { index, groups } = groupFaces(order, new Uint8Array([7, 2]), 3);
    expect([...index]).toEqual([3, 4, 5]);
    expect(groups).toEqual([{ start: 0, count: 3, materialIndex: 2 }]);
  });
});
