/**
 * The confirm panel's state machine in a DOM (happy-dom): re-slicing for other nozzles, cancelling meanwhile, a
 * failed re-slice, an older daemon without nozzle choice. The pure helpers are in printconfirm.test.ts.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import type { ConfirmSource } from "./printconfirm";
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

/* From the 2026-11-26 audit: a pick's pending check, and a check that fails once. */
const auditTrays: AmsSlot[] = [
  { slot: 4, label: "B1", type: "PETG", colour: "#F4EE2A", name: "" },
  { slot: 7, label: "B4", type: "PETG", colour: "#FFFFFF", name: "" },
  { slot: 9, label: "C2", type: "PETG", colour: "#F0E020", name: "" },
];
const bySlot = (s: number): AmsSlot => auditTrays.find((t) => t.slot === s)!;
function planFor(slots: number[], nozzles = ["left", "right"]): PrintMapping {
  return { filaments: slots.map((s, i) => ({ filament: i + 1, type: "PETG", colour: bySlot(s).colour, nozzle: nozzles[i],
    errors: [], warnings: [], tray: { tray_id: s, label: bySlot(s).label, type: "PETG", colour: bySlot(s).colour } })),
    errors: [], warnings: [], notes: [] } as unknown as PrintMapping;
}
function later<T>() {
  let resolve!: (v: T) => void; let reject!: (e: Error) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}
const wait = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const button = (box: ParentNode, text: string) => [...box.querySelectorAll("button")].find((b) => b.textContent === text) as HTMLButtonElement;


describe("spool checks in a DOM", () => {
  test("A: Start print pressed while the check of a newly picked spool is still running sends the picked spool (or waits)", async () => {
    const { openPrintConfirm } = await import("./printconfirm");
    const box = document.createElement("div");
    document.body.append(box);
    const started: number[][] = [];
    let slow: ReturnType<typeof later<PrintMapping>> | null = null;
    const src: ConfirmSource = {
      pollMs: 60_000,
      plan: async (slots) => (slow ? slow.promise : planFor(slots ?? [4, 7])),
      trays: async () => auditTrays,
      start: async (slots) => { started.push(slots); return "accepted"; },
    };
    openPrintConfirm(box, src, [4, 7], ["file A"]);
    await wait(5);
    // The user opens filament 1's spool list and picks C2 (slot 9); the plan for it is still being fetched
    // (`beambam ams map` takes seconds in the local version).
    slow = later<PrintMapping>();
    (box.querySelector(".slot-picker") as HTMLButtonElement).click();
    const opt = [...document.querySelectorAll(".slot-opt")].find((o) => o.textContent?.includes("C2")) as HTMLElement;
    opt.click();
    await wait();
    const start = button(box, "Start print");
    const wasEnabled = !start.disabled;
    start.click();
    await wait(5);
    slow.resolve(planFor([9, 7]));
    await wait(5);
    expect(wasEnabled).toBe(false);                    // Start waits for the check
    // Expected: nothing sent with the old spool.
    expect(started.some((s) => s[0] === 4)).toBe(false);
  });

  test("C: a spool check that fails once (network) leaves a way to check again", async () => {
    const { openPrintConfirm } = await import("./printconfirm");
    const box = document.createElement("div");
    document.body.append(box);
    let fail = false; let calls = 0;
    const src: ConfirmSource = {
      pollMs: 20,
      plan: async (slots) => { calls++; if (fail) { fail = false; throw new Error("beambam is not reachable"); } return planFor(slots ?? [4, 7]); },
      trays: async () => auditTrays,
      start: async () => "accepted",
    };
    const stop = openPrintConfirm(box, src, [4, 7], ["file A"]);
    await wait(5);
    fail = true;
    (box.querySelector(".slot-picker") as HTMLButtonElement).click();
    ([...document.querySelectorAll(".slot-opt")].find((o) => o.textContent?.includes("C2")) as HTMLElement).click();
    await wait(120);                                   // several polls later
    const text = box.textContent ?? "";
    expect(button(box, "Check again")).toBeTruthy();
    console.log("C: plan calls:", calls, "| rows:", box.querySelectorAll(".pc-item").length, "| Start disabled:", button(box, "Start print").disabled, "| text has error:", text.includes("not reachable"));
    stop();
    expect(box.querySelectorAll(".pc-item").length).toBe(2);
  });
});

/* The 2026-11-26 audit, findings 16, 18 and 21: nozzle switches only when the plan names the sliced nozzles, one
 * poll loop, inert orphaned panels, and the plate in what re-checks a plan. */
describe("poll loop and orphaned panels in a DOM", () => {
  test("a failed Start does not start a second poll loop, and checks the plan again", async () => {
    const { openPrintConfirm } = await import("./printconfirm");
    const box = document.createElement("div");
    document.body.append(box);
    let reads = 0;
    let plans = 0;
    let gate: ReturnType<typeof later<AmsSlot[]>> | null = null;
    const src: ConfirmSource = {
      pollMs: 30,
      plan: async (slots) => { plans++; return planFor(slots ?? [4, 7]); },
      trays: async () => { reads++; return gate ? gate.promise : auditTrays; },
      start: async () => { throw new Error("The spools, the plate or the file changed since this check"); },
    };
    const stop = openPrintConfirm(box, src, [4, 7], ["file A"]);
    await wait(5);
    // A spool read hangs (the local `beambam status` takes seconds); Start fails meanwhile.
    gate = later<AmsSlot[]>();
    await wait(40);                                  // the poll's tick is now awaiting the read
    const plansBefore = plans;
    button(box, "Start print").click();
    await wait(5);
    expect(plans).toBe(plansBefore + 1);             // checked again after the refusal
    const pending = gate;
    gate = null;
    pending.resolve(auditTrays);
    reads = 0;
    await wait(100);                                 // ~3 poll periods
    stop();
    // One loop reads about every 30 ms; two would read twice as often.
    expect(reads).toBeGreaterThan(0);
    expect(reads).toBeLessThanOrEqual(4);
  });

  test("an orphaned panel's late answer does not close the spool list open in the new one", async () => {
    const { openPrintConfirm } = await import("./printconfirm");
    const page = document.createElement("div");
    document.body.append(page);
    const boxA = document.createElement("div");
    page.append(boxA);
    const sending = later<string>();
    const src: ConfirmSource = {
      pollMs: 60_000,
      plan: async (slots) => planFor(slots ?? [4, 7]),
      trays: async () => auditTrays,
      start: () => sending.promise,
    };
    openPrintConfirm(boxA, src, [4, 7], ["file A"]);
    await wait(5);
    button(boxA, "Start print").click();             // sending...
    await wait();
    // The slice went stale meanwhile: its output (and this panel) left the page; a new slice's panel opens.
    const boxB = document.createElement("div");
    page.replaceChildren(boxB);
    const stopNew = openPrintConfirm(boxB, { ...src, start: async () => "accepted" }, [4, 7], ["file B"]);
    await wait(5);
    (boxB.querySelector(".slot-picker") as HTMLButtonElement).click();
    expect(document.querySelector(".slot-list")).not.toBeNull();
    sending.reject(new Error("beambam refused the print"));   // the orphaned panel's late answer
    await wait(5);
    expect(document.querySelector(".slot-list")).not.toBeNull();
    (document.querySelector(".slot-opt") as HTMLElement | null)?.click();
    stopNew();
  });

  test("no nozzle switches when the plan does not name the sliced nozzles", async () => {
    const p = await open({ renozzle: async () => ["file B"], nozzlesKnown: () => false },
      ["Update beambam to choose each filament's nozzle here."]);
    expect(p.box.querySelector(".pc-nozzle")).toBeNull();
    expect(panelText(p.box)).toContain("Update beambam");
    expect(panelText(p.box)).toContain("right nozzle");
    p.stop();
  });

  test("a plate change on the printer checks the plan again", async () => {
    const { openPrintConfirm } = await import("./printconfirm");
    const box = document.createElement("div");
    document.body.append(box);
    let plate = "P0301";
    let plans = 0;
    const stop = openPrintConfirm(box, {
      pollMs: 20,
      plan: async (slots) => { plans++; return planFor(slots ?? [4, 7]); },
      trays: async () => auditTrays,
      plate: () => plate,
      start: async () => "accepted",
    }, [4, 7], ["file A"]);
    await wait(50);
    const before = plans;
    plate = "P0201";
    await wait(50);
    stop();
    expect(plans).toBe(before + 1);                  // once: the next reads see the same plate
  });

  const connectButton = (box: HTMLElement) =>
    [...box.querySelectorAll("button")].find((b) => (b.textContent ?? "").startsWith("Send with Bambu Connect")) as
      HTMLButtonElement | undefined;

  test("no Bambu Connect section without a connect source (the usual case)", async () => {
    const p = await open();
    expect(connectButton(p.box)).toBeUndefined();
    expect(p.box.querySelector(".pc-connect")).toBeNull();
    p.stop();
  });

  test("Send with Bambu Connect hands the file over and reports the outcome", async () => {
    const calls: number[] = [];
    const p = await open({
      connect: async () => { calls.push(1); return "bambu-connect://import-file?path=%2Ftmp%2Fa.gcode.3mf"; },
      signingNote: "no signing key on this beambam yet",
    });
    expect(panelText(p.box)).toContain("cannot sign a print start yet (no signing key on this beambam yet)");
    expect(panelText(p.box)).toContain("Bambu Connect can send it with your Bambu login instead");
    const btn = connectButton(p.box)!;
    expect(btn).toBeDefined();
    btn.click();
    await tick(5);
    expect(calls).toEqual([1]);
    expect(panelText(p.box)).toContain("Opened in Bambu Connect");
    expect(btn.disabled).toBe(false);                 // usable again
    p.stop();
  });

  test("a Bambu Connect failure is shown, the button stays usable", async () => {
    const p = await open({
      connect: async () => { throw new Error("beambam connect: Bambu Connect is not installed"); },
    });
    // No signingNote: the button still appears (beambam offered it), just no note above it.
    expect(panelText(p.box)).not.toContain("cannot sign a print start");
    const btn = connectButton(p.box)!;
    btn.click();
    await tick(5);
    expect(panelText(p.box)).toContain("Bambu Connect is not installed");
    expect(btn.disabled).toBe(false);
    p.stop();
  });

  test("only the note shows when Bambu Connect is not available (no connect source)", async () => {
    const p = await open({ signingNote: "Bambu Connect has no build for this system" });
    expect(panelText(p.box)).toContain("cannot sign a print start yet");
    expect(panelText(p.box)).not.toContain("Bambu Connect can send it with your Bambu login");
    expect(connectButton(p.box)).toBeUndefined();
    p.stop();
  });
});
