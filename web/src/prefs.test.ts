import { describe, expect, it } from "bun:test";
import { clearPrefs, pickPrefs, readPrefs, writePrefs, type PrefStore } from "./prefs";
import type { Params } from "./types";

const DEFAULTS = {
  ball_scale: 0.6, ball_diameter: null, washer: "M6", clearance: 1, figure_scale: null, figure_height: null,
  rotate_x: 0, rotate_y: 0, rotate_z: 0, magnets: 2, pockets: null, pocket_diameter: 4.18, pocket_depth: 2.2,
  pocket_wall: 0.8, pocket_ceiling: 0.6, pocket_grip: 1.2, pocket_max_seat: 2, pocket_seat: "washer",
  object_id: null, slot: null, settings_source: "reference", process: null, bend: null, flat_base: null,
  flat_depth: 1.2, flat_fill: 2.5, fill_circle: null, paint_depth: null, filament_map: null, colours: null,
  paint: null, texture_colours: null, texture_details: 0, strengthen: 0,
} as Params;

function memory(): PrefStore & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v), removeItem: (k) => void data.delete(k) };
}

describe("remembered settings", () => {
  it("keeps only what the user changed, never per-file values", () => {
    const p = { ...DEFAULTS, figure_height: 25.4, flat_fill: 0, rotate_x: 90, figure_scale: 0.5, washer: "M8",
      fill_circle: { x: 0, y: 0, d: 10 } } as Params;
    expect(pickPrefs(p, DEFAULTS)).toEqual({ figure_height: 25.4, flat_fill: 0, washer: "M8" });
  });

  it("round-trips through storage and forgets when back at the defaults", () => {
    const s = memory();
    writePrefs(s, { ...DEFAULTS, figure_height: 25.4, flat_fill: 0 }, DEFAULTS);
    expect(readPrefs(s)).toEqual({ figure_height: 25.4, flat_fill: 0 });
    writePrefs(s, DEFAULTS, DEFAULTS);
    expect(s.data.size).toBe(0);
    writePrefs(s, { ...DEFAULTS, magnets: 1 }, DEFAULTS);
    clearPrefs(s);
    expect(readPrefs(s)).toEqual({});
  });

  it("drops malformed, unknown and wrongly typed entries", () => {
    const s = memory();
    s.setItem("pokefit.prefs.v1", JSON.stringify({ flat_fill: "lots", magnets: 3, paint: [1], figure_height: null, bend: null, washer: "" }));
    expect(readPrefs(s)).toEqual({ magnets: 3, figure_height: null, bend: null });
    s.setItem("pokefit.prefs.v1", "{not json");
    expect(readPrefs(s)).toEqual({});
    expect(readPrefs(null)).toEqual({});
  });

  it("survives storage that throws (blocked site data)", () => {
    const bad: PrefStore = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("full"); }, removeItem: () => { throw new Error("x"); } };
    expect(readPrefs(bad)).toEqual({});
    expect(() => writePrefs(bad, { ...DEFAULTS, magnets: 1 }, DEFAULTS)).not.toThrow();
    expect(() => clearPrefs(bad)).not.toThrow();
  });
});
