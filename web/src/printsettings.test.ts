/** The Print card's "project's (…)" values (printsettings.ts): the 2026-11-26 audit, finding 19. */
import { describe, expect, test } from "bun:test";
import { defaultsFor, settingsSummary } from "./printsettings";
import type { Config, Params } from "./types";

const cfg = {
  print_defaults: { layer_height: 0.16, wall_loops: 2, infill: 15, brim: null, min_line_width: 0.2 },
  process_defaults: { "0.12 fine": { layer_height: 0.12, wall_loops: 3, infill: 15, brim: null, min_line_width: 0.15 } },
} as unknown as Config;
const params = (process: string | null) => ({ process, supports: "auto" } as unknown as Params);

describe("print settings defaults", () => {
  test("an imported process preset brings its own values", () => {
    expect(defaultsFor(params(null), cfg)?.layer_height).toBe(0.16);
    expect(defaultsFor(params("0.12 fine"), cfg)?.layer_height).toBe(0.12);
    expect(defaultsFor(params("0.12 fine"), cfg)?.min_line_width).toBe(0.15);
    // A preset the engine did not report (an older engine): the project's.
    expect(defaultsFor(params("other"), cfg)?.layer_height).toBe(0.16);
    expect(settingsSummary(params("0.12 fine"), cfg)).toContain("0.12 mm · 3 walls");
  });
});
