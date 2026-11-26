import { describe, expect, test } from "bun:test";
import { syncSummary } from "./projectsync";
import type { ProjectSync } from "./types";

const sync = (o: Partial<ProjectSync>): ProjectSync =>
  ({ differs: true, same: 1, project: 17, printer: 16, plate: "Supertack Plate", rows: [], ...o });

describe("project vs printer", () => {
  test("the verdict says how many slots match and what stays", () => {
    expect(syncSummary(sync({}))).toBe("Your project's filaments differ from the printer's: 1 of 16 slots match " +
      "(its last slot stays as it is).");
    expect(syncSummary(sync({ project: 20 }))).toContain("(its other 4 slots stay as they are)");
    expect(syncSummary(sync({ project: 4, printer: 16 }))).toBe("Your project's filaments differ from the printer's: 1 of 16 slots match.");
    expect(syncSummary(sync({ differs: false, same: 16 }))).toBe("Your project's first 16 filaments match the printer's spools.");
  });
});
