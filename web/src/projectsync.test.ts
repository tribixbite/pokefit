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

describe("the plate line", () => {
  test("names the printer's plate when known, else its marker; flags a difference", async () => {
    const { plateLine, plateText } = await import("./projectsync");
    expect(plateText("Supertack Plate")).toBe("SuperTack plate");
    const d = plateLine("Textured PEI Plate", "Supertack Plate", "P0301");
    expect(d.differs).toBe(true);
    expect(d.text).toContain("prints on the Textured PEI plate; the printer has the SuperTack plate on");
    expect(plateLine("Supertack Plate", "Supertack Plate", "P0301")).toEqual({ differs: false,
      text: "The project prints on the SuperTack plate; the printer has the SuperTack plate on." });
    expect(plateLine("Cool Plate", null, "P0999").text).toContain("plate marker P0999 (not known yet)");
  });
});
