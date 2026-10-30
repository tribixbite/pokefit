/**
 * Collapsible cards: every panel card with a heading gets a toggle in its heading; a collapsed card shows only its
 * heading row (controls beside the heading, e.g. the magnet count, stay usable). Which cards are collapsed is
 * remembered in this browser (`pokefit.collapsed.v1`; storage may be unavailable, then nothing is remembered).
 */

const KEY = "pokefit.collapsed.v1";

/** Minimal storage interface (localStorage, or a stub in tests). */
export interface CollapseStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** The remembered collapsed card keys. */
export function loadCollapsed(store: CollapseStore | null): Set<string> {
  try {
    const raw = store?.getItem(KEY);
    const list: unknown = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(list) ? list.filter((x): x is string => typeof x === "string") : []);
  } catch {
    return new Set();
  }
}

export function saveCollapsed(store: CollapseStore | null, keys: Set<string>): void {
  try {
    store?.setItem(KEY, JSON.stringify([...keys].sort()));
  } catch {
    // Private window or blocked storage: the state just isn't remembered.
  }
}

/** A stable key for a card: its id, else its heading as a slug ("Ball & washer" -> "ball-washer"). */
export function cardKey(id: string, heading: string): string {
  return id || heading.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function store(): CollapseStore | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

/**
 * A section inside a card opened and closed by `btn` (its `aria-expanded`); `apply(open)` shows or hides the body.
 * Remembered with the cards: `key` in the set means collapsed for a section open by default, and `key:open` means
 * opened for one collapsed by default (`defaultOpen` false). Returns a setter for code that opens it.
 */
export function sectionToggle(btn: HTMLButtonElement, key: string, defaultOpen: boolean,
  apply: (open: boolean) => void): (open: boolean) => void {
  const mark = defaultOpen ? key : `${key}:open`;
  const set = (open: boolean, save: boolean): void => {
    btn.setAttribute("aria-expanded", String(open));
    apply(open);
    if (!save) return;
    const keys = loadCollapsed(store());
    if (open !== defaultOpen) keys.add(mark); else keys.delete(mark);
    saveCollapsed(store(), keys);
  };
  set(loadCollapsed(store()).has(mark) ? !defaultOpen : defaultOpen, false);
  btn.addEventListener("click", () => set(btn.getAttribute("aria-expanded") !== "true", true));
  return (open) => set(open, true);
}

/** Make every `.card` in `panel` with an `h2` collapsible. */
export function makeCollapsible(panel: HTMLElement): void {
  const collapsed = loadCollapsed(store());
  panel.querySelectorAll<HTMLElement>(":scope > .card").forEach((card) => {
    const h = card.querySelector<HTMLElement>(":scope > h2, :scope > .h-row > h2");
    if (!h) return;
    const key = cardKey(card.id, h.textContent ?? "");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "card-toggle";
    const chev = document.createElement("span");
    chev.className = "chev";
    chev.setAttribute("aria-hidden", "true");
    btn.append(chev, ...Array.from(h.childNodes));
    h.replaceChildren(btn);
    const set = (shut: boolean): void => {
      card.classList.toggle("collapsed", shut);
      btn.setAttribute("aria-expanded", String(!shut));
      btn.title = shut ? "Show this section" : "Hide this section";
    };
    set(collapsed.has(key));
    btn.addEventListener("click", () => {
      const shut = !card.classList.contains("collapsed");
      set(shut);
      if (shut) collapsed.add(key); else collapsed.delete(key);
      saveCollapsed(store(), collapsed);
    });
  });
}
