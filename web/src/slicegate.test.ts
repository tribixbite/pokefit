/** SliceGate (main.ts's Slice button and slice currency): the 2026-11-26 audit, findings 14 and 15. */
import { describe, expect, test } from "bun:test";
import { SliceGate } from "./slicegate";

/** A promise the test resolves by hand. */
function later<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((a) => { resolve = a; });
  return { promise, resolve };
}

describe("slice gate", () => {
  test("through beambam, Slice stays off while a dropped slice is still running there", async () => {
    const g = new SliceGate();
    const gen = g.begin();
    const job = later<string>();
    const run = g.daemonSlice(() => job.promise);
    expect(g.button(false, true)).toEqual({ disabled: true, waiting: false });     // the current slice runs
    g.stale();                                         // settings changed: its result will be dropped
    expect(g.isCurrent(gen)).toBe(false);
    // The daemon is still slicing it and would answer 409 to another: off, and the page says why.
    expect(g.button(false, true)).toEqual({ disabled: true, waiting: true });
    job.resolve("old file");
    await run;
    g.end(gen);
    expect(g.button(false, true)).toEqual({ disabled: false, waiting: false });
  });

  test("locally a new Slice may start while a dropped one finishes, which then does not turn Slice on", () => {
    const g = new SliceGate();
    const first = g.begin();
    g.stale();
    expect(g.button(true, false).disabled).toBe(false);
    const second = g.begin();
    g.end(first);                                      // the dropped slice ends while the new one runs
    expect(g.button(true, false).disabled).toBe(true);
    g.end(second);
    expect(g.button(true, false).disabled).toBe(false);
    expect(g.button(false, false).disabled).toBe(true);  // no slicer at all
  });

  test("a panel re-slice keeps Slice off through beambam until it ends", async () => {
    const g = new SliceGate();
    g.end(g.begin());
    const job = later<void>();
    const run = g.daemonSlice(() => job.promise);
    expect(g.button(false, true)).toEqual({ disabled: true, waiting: true });
    job.resolve();
    await run;
    expect(g.button(false, true).disabled).toBe(false);
  });

  test("a failed re-slice restores the nozzle choice only while its slice is still current", () => {
    const g = new SliceGate();
    g.end(g.begin());
    const p = { nozzles: [2, 1] as number[] | null };
    let gen = g.gen;
    p.nozzles = [2, 2];                                // the re-slice asked for this, then failed
    g.restoreNozzles(p, [2, 1], gen);
    expect(p.nozzles).toEqual([2, 1]);
    gen = g.gen;
    p.nozzles = [1, 2];
    g.begin();                                         // a newer Slice meanwhile (it clears the choice itself)
    p.nozzles = null;
    g.restoreNozzles(p, [2, 1], gen);
    expect(p.nozzles).toBeNull();                      // not the stale map
  });
});
