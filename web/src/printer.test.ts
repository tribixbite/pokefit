import { describe, expect, test } from "bun:test";
import { progressText } from "./printer";
import type { PrintProgress } from "./types";

const base: PrintProgress = {
  state: "RUNNING", job: "squirtle", percent: 42, layer: 80, layers: 190, remaining_min: 72,
  nozzles: [{ id: 0, temp: 248, target: 250 }, { id: 1, temp: 27, target: 0 }], bed: 69.6, bed_target: 70, error: 0,
};

describe("printer card text", () => {
  test("a running print: progress, both nozzles (R / L), bed", () => {
    expect(progressText(base)).toBe(
      "Printing: squirtle · 42 % · layer 80/190 · 1h 12m left · nozzles R 248/250 °C, L 27 °C · bed 70/70 °C");
  });

  test("finished or idle: no progress figures; one nozzle unlabelled; errors in hex", () => {
    const done = { ...base, state: "FINISH", nozzles: [{ id: 0, temp: 27, target: 0 }], bed: 25, bed_target: 0 };
    expect(progressText(done)).toBe("Finished: squirtle · nozzle 27 °C · bed 25 °C");
    expect(progressText({ ...done, job: "", state: "IDLE", error: 0x0500_4038 })).toBe(
      "Idle · nozzle 27 °C · bed 25 °C · error 5004038");
  });
});
