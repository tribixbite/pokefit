import { describe, expect, it } from "bun:test";
import { carriedOver, modelChanges } from "./changes";
import type { Params, Report, WeakPoint } from "./types";
import { describeWeak, grams, suggestThickness } from "./weak";

const DEFAULTS = {
  ball_scale: 0.6, ball_diameter: null, steel: "D20", floor_diameter: null, clearance: 1, figure_scale: null, figure_height: null,
  rotate_x: 0, rotate_y: 0, rotate_z: 0, magnets: 2, pockets: null, pocket_diameter: 4.18, pocket_depth: 2.2,
  pocket_wall: 0.8, pocket_ceiling: 0.6, pocket_grip: 1.2, pocket_max_seat: 2, pocket_seat: "steel",
  object_id: null, slot: null, settings_source: "reference", process: null, bend: null, flat_base: null,
  flat_depth: 1.2, flat_fill: 2.5, fill_circle: null, fill_area: "steel", fill_shape: null, paint_depth: null, core_filament: null, filament_map: null, colours: null,
  paint: null, texture_colours: null, texture_details: 0, strengthen: 0, simplify: null, despeckle: 0.5, simplify_auto: true,
} as Params;

/** A report whose conform step bent, pressed and filled, and whose file was simplified. */
function report(conform: Partial<NonNullable<Report["fit"]["conform"]>>, simplified = false): Report {
  return {
    fit: { conform: { bend: 0.66, auto: false, lift_mm: 0, flat_depth_mm: 0, fill_mm2: 0, scale_before: 1, scale_after: 1, note: "",
      fill_mm: 2.5, ...conform } },
    object: { simplified: simplified ? { faces_before: 900000, faces_after: 200000, error: 0.01 } : null },
  } as unknown as Report;
}

describe("changes to your model", () => {
  it("lists what the engine reports it did, with the value that undoes each", () => {
    const p = { ...DEFAULTS, bend: 0.66, strengthen: 2, limbs: [{ part: 0, face: 1, scale: 1.2 }] } as Params;
    const c = modelChanges(p, report({ lift_mm: 2.1, flat_depth_mm: 0.8, fill_mm2: 40 }, true), DEFAULTS, {});
    expect(c.map((x) => x.id)).toEqual(["simplify", "limbs", "bend", "press", "fill", "strengthen"]);
    expect(Object.fromEntries(c.map((x) => [x.id, x.undo]))).toEqual({
      simplify: { simplify: null, simplify_auto: false }, limbs: { limbs: null }, bend: { bend: 0 }, press: { flat_depth: 0 },
      fill: { flat_fill: 0 }, strengthen: { strengthen: 0 },
    });
  });

  it("lists nothing when the settings leave the model as it is", () => {
    expect(modelChanges(DEFAULTS, report({}), DEFAULTS, {})).toEqual([]);
    expect(modelChanges(DEFAULTS, null, DEFAULTS, {})).toEqual([]);
  });

  it("tells a setting carried over from the last figure from a default or one set now", () => {
    const rep = report({ lift_mm: 2.1, fill_mm2: 40 });
    const carriedBend = { ...DEFAULTS, bend: 0.66 } as Params;
    const remembered = { bend: 0.66 };
    const c = modelChanges(carriedBend, rep, DEFAULTS, remembered);
    expect(c.find((x) => x.id === "bend")?.origin).toBe("remembered");
    expect(c.find((x) => x.id === "fill")?.origin).toBe("default");
    expect(carriedOver(c).map((x) => x.id)).toEqual(["bend"]);
    // Changed again for this figure: no longer the remembered value.
    const now = modelChanges({ ...carriedBend, bend: 1 }, rep, DEFAULTS, remembered);
    expect(now.find((x) => x.id === "bend")?.origin).toBe("this figure");
  });

  it("lists single weak parts thickened", () => {
    const p = { ...DEFAULTS, strengthen_at: [{ part: 0, face: 3, root: 9, mm: 3.5 }, { part: 0, face: 4, root: 8, mm: 3 }] } as Params;
    const c = modelChanges(p, report({}), DEFAULTS, {});
    expect(c).toHaveLength(1);
    expect(c[0]).toMatchObject({ id: "strengthen_at", label: "2 weak parts thickened", detail: "3.5 mm, 3 mm",
      undo: { strengthen_at: null } });
  });
});

describe("weak points", () => {
  const tail = { force_n: 1.8, width_mm: 2.44, depth_mm: 2.71, lever_mm: 24.6, flat: 0.55 } as WeakPoint;

  it("suggests a thickness that should reach grade C", () => {
    // Strength grows with the thickness squared: 2.44 x sqrt(3 / 1.8) = 3.15 -> 3.5 mm.
    expect(suggestThickness(tail)).toBe(3.5);
    expect(suggestThickness({ ...tail, force_n: 20, width_mm: 1.2 })).toBe(2);       // strong already: +0.5 mm
    expect(suggestThickness({ ...tail, force_n: 0.01, width_mm: 3 })).toBe(5);       // capped at the engine's 5 mm
  });

  it("describes the push in newtons and grams", () => {
    expect(grams(1)).toBe(102);
    expect(describeWeak(tail)).toBe("Snaps at ≈ 1.8 N (184 g) pushed at its end · 2.4 × 2.7 mm across, 25 mm lever; its layers pull apart there");
  });
});
