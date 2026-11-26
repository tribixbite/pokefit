/**
 * The confirm panel's state machine in a DOM (happy-dom): re-slicing for other nozzles, cancelling meanwhile, a
 * failed re-slice, an older daemon without nozzle choice. The pure helpers are in printconfirm.test.ts.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import type { AmsSlot, PrintMapping } from "./types";

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const trays: AmsSlot[] = [
  { slot: 4, label: "B1", type: "PETG", colour: "#F4EE2A", name: "" },
  { slot: 7, label: "B4", type: "PLA", colour: "#FFFFFF", name: "PLA Pure" },
];

/** A plan for two filaments on the given nozzles ("left" / "right"). */
function plan(n1: string, n2: string, errors: string[] = []): PrintMapping {
  const f = (filament: number, nozzle: string, tray: AmsSlot) => ({
    filament, type: tray.type, colour: tray.colour, nozzle, errors: filament === 1 ? errors : [], warnings: [],
    tray: { tray_id: tray.slot, label: tray.label, type: tray.type, colour: tray.colour },
  });
  return { filaments: [f(1, n1, trays[0]), f(2, n2, trays[1])], errors, warnings: [], notes: [] };
}

/** A promise the test resolves or rejects by hand. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const panelText = (box: HTMLElement) => box.textContent ?? "";
const startButton = (box: HTMLElement) =>
  [...box.querySelectorAll("button")].find((b) => b.textContent === "Start print") as HTMLButtonElement;
const nozzleButton = (box: HTMLElement, filament: number, name: string) =>
  [...box.querySelectorAll(".pc-item")[filament - 1].querySelectorAll(".pc-nozzle button")]
    .find((b) => b.textContent === name) as HTMLButtonElement;

async function open(over: Record<string, unknown> = {}, notes: string[] = []) {
  const { openPrintConfirm } = await import("./printconfirm");
  const box = document.createElement("div");
  document.body.append(box);
  let current = plan("left", "right");
  let mine: number[] | null = null;
  const started: number[][] = [];
  const src = {
    pollMs: 60_000,
    plan: async () => current,
    trays: async () => trays,
    start: async (slots: number[]) => { started.push(slots); return "the printer accepted the print"; },
    nozzles: () => mine,
    ...over,
  };
  const stop = openPrintConfirm(box, src as never, [4, 7], ["file A: 2h 09m"], notes);
  await tick(5);
  return { box, stop, started, setPlan: (m: PrintMapping) => { current = m; }, setMine: (n: number[] | null) => { mine = n; } };
}

describe("confirm panel in a DOM", () => {
  test("picking the other nozzle re-slices: Start print waits, then the header and status follow", async () => {
    const slice = deferred<string[]>();
    const asked: (number[] | null)[] = [];
    const p = await open({ renozzle: (n: number[] | null) => { asked.push(n); return slice.promise; } });
    expect(nozzleButton(p.box, 1, "Left").getAttribute("aria-checked")).toBe("true");
    nozzleButton(p.box, 1, "Right").click();
    await tick();
    expect(asked).toEqual([[2, 2]]);
    expect(startButton(p.box).disabled).toBe(true);           // never start the file being replaced
    expect(nozzleButton(p.box, 2, "Left").disabled).toBe(true);
    p.setPlan(plan("right", "right"));
    p.setMine([2, 2]);
    slice.resolve(["file B: 2h 44m"]);
    await tick(5);
    expect(panelText(p.box)).toContain("file B: 2h 44m");
    expect(panelText(p.box)).not.toContain("file A");
    expect(panelText(p.box)).toContain("Sliced again with filament 1 on the right nozzle");
    expect(panelText(p.box)).toContain("Let Bambu choose");
    expect(startButton(p.box).disabled).toBe(false);
    startButton(p.box).click();
    await tick(5);
    expect(p.started).toEqual([[4, 7]]);
    p.stop();
  });

  test("a failed re-slice says so and leaves the panel usable", async () => {
    const p = await open({ renozzle: async () => { throw new Error("beambam: the slicer timed out"); } });
    nozzleButton(p.box, 2, "Left").click();
    await tick(5);
    expect(panelText(p.box)).toContain("Could not slice again: beambam: the slicer timed out");
    expect(panelText(p.box)).toContain("file A");
    expect(startButton(p.box).disabled).toBe(false);
    p.stop();
  });

  test("cancel during a re-slice: the late result does not draw into the emptied box", async () => {
    const slice = deferred<string[]>();
    const p = await open({ renozzle: () => slice.promise });
    nozzleButton(p.box, 1, "Right").click();
    await tick();
    ([...p.box.querySelectorAll("button")].find((b) => b.textContent === "Cancel") as HTMLButtonElement).click();
    expect(p.box.childElementCount).toBe(0);
    slice.resolve(["file B"]);
    await tick(5);
    expect(p.box.childElementCount).toBe(0);
  });

  test("an older daemon: no nozzle switch, the panel says why", async () => {
    const p = await open({}, ["Update beambam to choose each filament's nozzle here."]);
    expect(p.box.querySelector(".pc-nozzle")).toBeNull();
    expect(panelText(p.box)).toContain("Update beambam");
    expect(panelText(p.box)).toContain("right nozzle");          // the plan's nozzle still shown as text
    p.stop();
  });

  test("an error in the plan keeps Start print disabled", async () => {
    const p = await open({ plan: async () => plan("left", "right", ["tray B1 is empty"]) });
    expect(startButton(p.box).disabled).toBe(true);
    expect(panelText(p.box)).toContain("tray B1 is empty");
    p.stop();
  });
});
