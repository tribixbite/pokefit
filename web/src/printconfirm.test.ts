import { describe, expect, test } from "bun:test";
import { slotsFromPlan, trayChoices, traySignature } from "./printconfirm";
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
});
