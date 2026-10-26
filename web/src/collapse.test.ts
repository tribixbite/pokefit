import { describe, expect, it } from "bun:test";
import { cardKey, type CollapseStore, loadCollapsed, saveCollapsed } from "./collapse";

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
});
