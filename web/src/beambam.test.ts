import { afterEach, describe, expect, test } from "bun:test";
import { DaemonError, headerName, normaliseUrl, plan, print, readLink, slice, startCommand, writeLink } from "./beambam";
import type { PrefStore } from "./prefs";

const store = (): PrefStore & { data: Map<string, string> } => {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v), removeItem: (k) => void data.delete(k) };
};

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

/** A fake daemon: answers each path from `routes` (status, body) and records the requests. */
function fakeDaemon(routes: Record<string, (body: unknown) => [number, unknown]>): { seen: [string, string, unknown][] } {
  const seen: [string, string, unknown][] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : init?.body ? "<bytes>" : null;
    seen.push([init?.method ?? "GET", url.pathname, body]);
    const route = routes[url.pathname];
    if (!route) return new Response(JSON.stringify({ error: "no such route" }), { status: 404 });
    const [status, reply] = route(body);
    return new Response(JSON.stringify(reply), { status, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  return { seen };
}

const link = { url: "http://127.0.0.1:8765", enabled: true };

describe("link settings", () => {
  test("default is off at the loopback daemon; remembered once set", () => {
    const s = store();
    expect(readLink(s)).toEqual({ url: "http://127.0.0.1:8765", enabled: false });
    writeLink(s, { url: "http://192.168.0.20:8765", enabled: true });
    expect(readLink(s)).toEqual({ url: "http://192.168.0.20:8765", enabled: true });
    s.data.set("pokefit.beambam", "{not json");
    expect(readLink(s).enabled).toBe(false);
  });

  test("addresses are normalised; the start command names this site", () => {
    expect(normaliseUrl("127.0.0.1:8765/")).toBe("http://127.0.0.1:8765");
    expect(normaliseUrl(" https://printer.lan:9000 ")).toBe("https://printer.lan:9000");
    expect(startCommand("https://pokefit.party")).toBe("beambam daemon --http 127.0.0.1:8765 --cors-origin https://pokefit.party");
    expect(headerName("Jolteon (v2) é.3mf")).toBe("Jolteon_v2_.3mf");
  });
});

describe("daemon calls", () => {
  test("a refused plan is still a plan; other refusals raise with beambam's words", async () => {
    const bad = { filaments: [{ filament: 1, errors: ["tray A4 is empty"], warnings: [] }], errors: ["filament 1: tray A4 is empty"], warnings: [], notes: [] };
    fakeDaemon({
      "/print/plan": (b) => ((b as { slot?: number[] }).slot ? [409, bad] : [422, { ok: false, error: "no sliced plate 1" }]),
    });
    expect((await plan(link, "/x.gcode.3mf", [3])).errors).toEqual(["filament 1: tray A4 is empty"]);
    await expect(plan(link, "/x.gcode.3mf", null)).rejects.toThrow("beambam: no sliced plate 1");
  });

  test("slice: upload, poll, the daemon-side print file", async () => {
    let polls = 0;
    const { seen } = fakeDaemon({
      "/slice": () => [202, { ok: true, poll: "/slice/jobs/s1" }],
      "/slice/jobs/s1": () => (++polls < 2 ? [200, { state: "slicing", message: "slicing", done: false }]
        : [200, { state: "done", done: true, file: "/home/u/.x2d/uploads/sliced/s1/fig.gcode.3mf", seconds: 60, grams: 2,
          grams_per_filament: [2], warnings: [] }]),
    });
    const ticks: string[] = [];
    const res = await slice(link, "fig.3mf", new Uint8Array([80, 75]), (_s, what) => ticks.push(what));
    expect(res.file).toEndWith("fig.gcode.3mf");
    expect(seen[0]).toEqual(["POST", "/slice", "<bytes>"]);
    expect(ticks).toEqual(["slicing", "done"]);
  });

  test("print sends the user's OK and waits for the printer", async () => {
    const { seen } = fakeDaemon({
      "/print": () => [202, { ok: true, poll: "/print/jobs/p1" }],
      "/print/jobs/p1": () => [200, { state: "rejected", done: true, message: "the printer rejected the print: bad file" }],
    });
    await expect(print(link, "/f.gcode.3mf", [4, 0], true)).rejects.toThrow("rejected");
    expect(seen[0]).toEqual(["POST", "/print", { file: "/f.gcode.3mf", slot: [4, 0], confirm: true, accept_warnings: true }]);
  });

  test("no daemon: says how to start it", async () => {
    globalThis.fetch = (async () => { throw new TypeError("Failed to fetch"); }) as unknown as typeof fetch;
    (globalThis as { location?: unknown }).location ??= { origin: "https://pokefit.party" };
    const err = await plan(link, "/f", null).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(DaemonError);
    expect((err as DaemonError).unreachable).toBe(true);
    expect((err as Error).message).toContain("--cors-origin");
  });
});
