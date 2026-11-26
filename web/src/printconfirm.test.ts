import { describe, expect, test } from "bun:test";
import { nozzleNumber, nozzlesWith, planNamesSlicedNozzles, remapFilamentMap, slotsFromPlan, trayChoices,
  traySignature } from "./printconfirm";
import type { AmsSlot, PrintMapping } from "./types";

const tray = (slot: number, label: string, type: string, colour: string, name = ""): AmsSlot =>
  ({ slot, label, type, colour, name });
const fil = (n: number, trayId: number | null): PrintMapping["filaments"][number] => ({
  filament: n, type: "PLA", colour: "#FFFFFF", nozzle: "left",
  tray: trayId === null ? null : { tray_id: trayId, label: `B${trayId - 3} (slot ${trayId})`, type: "PLA", colour: "#FFFFFF" },
  errors: [], warnings: [],
});
const mapping = (...trays: (number | null)[]): PrintMapping =>
  ({ filaments: trays.map((t, i) => fil(i + 1, t)), errors: [], warnings: [], notes: [] });

describe("print confirm", () => {
  test("a spool change changes the signature; order and colour case do not", () => {
    const a = [tray(0, "A1", "PLA", "#ffffff"), tray(4, "B1", "PETG", "#F72323")];
    expect(traySignature(a)).toBe(traySignature([tray(4, "B1", "PETG", "#F72323"), tray(0, "A1", "PLA", "#FFFFFF")]));
    expect(traySignature(a)).not.toBe(traySignature([tray(0, "A1", "PLA", "#FFFFFF"), tray(4, "B1", "PLA", "#F72323")]));
    expect(traySignature(a)).not.toBe(traySignature([tray(0, "A1", "PLA", "#FFFFFF")]));
  });

  test("a plate change changes the signature too", () => {
    const a = [tray(0, "A1", "PLA", "#FFFFFF")];
    expect(traySignature(a, "P0301")).toBe(traySignature(a, "P0301"));
    expect(traySignature(a, "P0301")).not.toBe(traySignature(a, "P0201"));
    expect(traySignature(a)).not.toBe(traySignature(a, "P0301"));
  });

  test("the plan vouches for the sliced nozzles only when it matches the file's", () => {
    const m = (n1: string, n2: string): PrintMapping =>
      ({ ...mapping(4, 7), filaments: mapping(4, 7).filaments.map((f, i) => ({ ...f, nozzle: i ? n2 : n1 })) });
    // Bambu put filament 2 on the right; an older beambam reports the map asked for (all left).
    expect(planNamesSlicedNozzles(m("left", "left"), [{ filament: 1, nozzle: 1 }, { filament: 2, nozzle: 2 }])).toBe(false);
    expect(planNamesSlicedNozzles(m("left", "right"), [{ filament: 1, nozzle: 1 }, { filament: 2, nozzle: 2 }])).toBe(true);
    // A file whose nozzles are unknown cannot vouch; one nozzle ("-") or "both" needs nothing.
    expect(planNamesSlicedNozzles(m("left", "right"), [{ filament: 1, nozzle: 1 }, { filament: 2, nozzle: null }])).toBe(false);
    expect(planNamesSlicedNozzles(m("-", "both"), [])).toBe(true);
  });

  test("a manual colour -> slot choice stays on its tray across a new reading of the spools", () => {
    const before = [tray(0, "A1", "PLA", "#FFFFFF"), tray(1, "A2", "PLA", "#000000"), tray(4, "B1", "PETG", "#F72323")];
    // A2 ran out: B1 moves from position 3 to 2; the choice of A2 has no tray left; a "new slot" keeps its distance.
    const after = [tray(0, "A1", "PLA", "#FFFFFF"), tray(4, "B1", "PETG", "#F72323")];
    expect(remapFilamentMap({ "1": 3, "2": 2, "3": 4 }, before, after)).toEqual({ "1": 2, "3": 3 });
    expect(remapFilamentMap({ "2": 2 }, before, after)).toBeNull();
    expect(remapFilamentMap(null, before, after)).toBeNull();
    expect(remapFilamentMap({ "1": 1 }, before, before)).toEqual({ "1": 1 });
  });

  test("trays come from the plan, or none while a filament has no spool", () => {
    expect(slotsFromPlan(mapping(4, 0))).toEqual([4, 0]);
    expect(slotsFromPlan(mapping(4, null))).toBeNull();
  });

  test("the picker lists loaded spools, keeps an unloaded current tray, offers none for no spool", () => {
    const loaded = [tray(0, "A1", "PLA", "#FFFFFF", "Bambu PLA Basic")];
    expect(trayChoices(loaded, fil(1, 0), 0)).toEqual([
      { value: 0, label: "A1", colour: "#FFFFFF", detail: "PLA · Bambu PLA Basic" }]);
    const gone = trayChoices(loaded, fil(1, 7), 7);
    expect(gone[1]).toEqual({ value: 7, label: "B4", colour: null, detail: "not loaded now" });
    expect(trayChoices(loaded, fil(1, null), undefined)[1].value).toBe(-1);
  });

  test("moving one filament to the other nozzle keeps the others where the plan has them", () => {
    const m = mapping(4, 0, 9);
    m.filaments[1].nozzle = "right";
    m.filaments[2].nozzle = "both";              // the switch's dynamic mode: prints on the left once pinned
    expect(nozzleNumber("left")).toBe(1);
    expect(nozzleNumber("right")).toBe(2);
    expect(nozzleNumber("-")).toBeNull();
    expect(nozzlesWith(m, 1, 2)).toEqual([2, 2, 1]);
    expect(nozzlesWith(m, 2, 1)).toEqual([1, 1, 1]);
    // A plan that lists only filaments 2 and 4 (1 and 3 unused): every filament up to the highest gets a nozzle.
    const sparse: PrintMapping = { filaments: [{ ...fil(2, 0), nozzle: "right" }, fil(4, 4)], errors: [], warnings: [], notes: [] };
    expect(nozzlesWith(sparse, 4, 2)).toEqual([1, 2, 1, 2]);
  });
});

describe("print-wide warnings", () => {
  test("the plate warning is shown apart from the filament rows' own", async () => {
    const { printWarnings } = await import("./printconfirm");
    const m = { filaments: [], errors: [], notes: [], warnings: [
      "this file was sliced for the Textured PEI Plate, but the printer has the Supertack Plate on: swap the plate, or slice again for the plate that is on",
      "filament 1: colour differs (ΔE 12)"] };
    expect(printWarnings(m as never)).toEqual([m.warnings[0]]);
  });
});

describe("slots by filament number", () => {
  test("unused filaments keep the sliced-for tray (the local server wants one per project filament)", () => {
    const m = { filaments: [{ filament: 2, tray: { tray_id: 9 } }], errors: [], warnings: [], notes: [] };
    expect(slotsFromPlan(m as never, [4, 7, 0])).toEqual([4, 9, 0]);
    expect(slotsFromPlan(m as never)).toEqual([-1, 9]);           // beambam reads a list with -1 by number
  });
});
