import { describe, expect, it } from "bun:test";
import { cardKey, type CollapseStore, loadCollapsed, saveCollapsed, sectionToggle } from "./collapse";

function memory(): CollapseStore & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v) };
}

describe("collapsible cards", () => {
  it("keys a card by its id, else its heading", () => {
    expect(cardKey("colorCard", "Colours")).toBe("colorCard");
    expect(cardKey("", "Ball & washer")).toBe("ball-washer");
    expect(cardKey("", "Print (Bambu Studio)")).toBe("print-bambu-studio");
  });

  it("remembers collapsed cards and survives bad or missing storage", () => {
    const s = memory();
    saveCollapsed(s, new Set(["size", "checks"]));
    expect([...loadCollapsed(s)].sort()).toEqual(["checks", "size"]);
    s.data.set("pokefit.collapsed.v1", "{not json");
    expect(loadCollapsed(s).size).toBe(0);
    s.data.set("pokefit.collapsed.v1", JSON.stringify(["a", 3, null]));
    expect([...loadCollapsed(s)]).toEqual(["a"]);
    expect(loadCollapsed(null).size).toBe(0);
    const broken: CollapseStore = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    expect(loadCollapsed(broken).size).toBe(0);
    expect(() => saveCollapsed(broken, new Set(["x"]))).not.toThrow();
  });

  it("folds a section, remembering only a state that differs from its default", () => {
    const s = memory();
    (globalThis as { localStorage?: CollapseStore }).localStorage = s;
    // A stand-in for the toggle button: attributes and one click listener.
    const attrs = new Map<string, string>();
    let click = (): void => {};
    const btn = {
      setAttribute: (k: string, v: string) => void attrs.set(k, v),
      getAttribute: (k: string) => attrs.get(k) ?? null,
      addEventListener: (_: string, f: () => void) => { click = f; },
    } as unknown as HTMLButtonElement;
    const seen: boolean[] = [];
    const open = sectionToggle(btn, "prints-from", false, (o) => seen.push(o));
    expect(seen).toEqual([false]);                         // collapsed by default, nothing stored
    expect(attrs.get("aria-expanded")).toBe("false");
    click();
    expect(seen.at(-1)).toBe(true);
    expect([...loadCollapsed(s)]).toEqual(["prints-from:open"]);
    open(false);
    expect(loadCollapsed(s).size).toBe(0);
    // Open by default: the key marks it collapsed, and a new toggle reads it back.
    const dock: boolean[] = [];
    sectionToggle(btn, "paint-dock", true, (o) => dock.push(o));
    click();
    expect([...loadCollapsed(s)]).toEqual(["paint-dock"]);
    sectionToggle(btn, "paint-dock", true, (o) => dock.push(o));
    expect(dock).toEqual([true, false, false]);
    delete (globalThis as { localStorage?: CollapseStore }).localStorage;
  });
});
