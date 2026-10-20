/**
 * A dropdown of filament slots that shows each slot's colour (a native <select> cannot): a button with the
 * chosen slot's swatch, and a list of every slot, closest colour to the figure colour first, each with its
 * swatch, label, material and colour difference. Keyboard: arrows/Home/End move, Enter/Space pick, Esc
 * closes; any outside tap closes it. One list is open at a time.
 */
import { deltaE, matchLevel, textOn, type MatchLevel } from "./colour";

export interface SlotChoice {
  /** 1-based slot number sent back to the engine (`Params.filament_map`). */
  value: number;
  /** Short label: the AMS tray ("A3") or the project's slot number. */
  label: string;
  /** Material and name, e.g. "PLA · Bambu PLA Basic". */
  detail: string;
  /** #RRGGBB, or null for a slot that does not exist yet ("new slot"). */
  colour: string | null;
}

export interface SlotPickerOptions {
  choices: SlotChoice[];
  value: number;
  /** The figure colour the choices are ranked against. */
  figureColour: string;
  ariaLabel: string;
  onChange: (value: number) => void;
}

let open: { close: () => void } | null = null;

/** Close whichever slot list is open (e.g. before the rows are rebuilt). */
export function closeSlotPicker(): void {
  open?.close();
}

function swatchEl(colour: string | null, text = ""): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = `sp-sw${colour ? "" : " none"}`;
  if (colour) {
    s.style.background = colour;
    s.style.color = textOn(colour);
  }
  s.textContent = text;
  return s;
}

/** "ΔE 12" with its level, or "new" for a slot Bambu adds on import. */
function matchText(c: SlotChoice, figure: string): { text: string; level: MatchLevel | "new" } {
  if (!c.colour) return { text: "new slot", level: "new" };
  const de = deltaE(c.colour, figure);
  return { text: de < 1 ? "same colour" : `ΔE ${de.toFixed(0)}`, level: matchLevel(de) };
}

function choiceBody(c: SlotChoice, figure: string): HTMLElement[] {
  const m = matchText(c, figure);
  const main = document.createElement("span");
  main.className = "sp-main";
  const top = document.createElement("span");
  top.className = "sp-label";
  top.textContent = c.label;
  const sub = document.createElement("span");
  sub.className = "sp-detail";
  sub.textContent = c.detail;
  main.append(top, sub);
  const tag = document.createElement("span");
  tag.className = `sp-match ${m.level}`;
  tag.textContent = m.text;
  return [swatchEl(c.colour), main, tag];
}

/** Build a slot picker button; the list opens below it (above, when there is no room below). */
export function slotPicker(o: SlotPickerOptions): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "slot-picker";
  btn.setAttribute("aria-haspopup", "listbox");
  btn.setAttribute("aria-expanded", "false");
  btn.setAttribute("aria-label", o.ariaLabel);
  const current = o.choices.find((c) => c.value === o.value) ?? o.choices[0];
  btn.append(...choiceBody(current, o.figureColour));
  const caret = document.createElement("span");
  caret.className = "sp-caret";
  caret.textContent = "▾";
  btn.append(caret);

  // Closest colour first; a slot that does not exist yet goes last.
  const ranked = [...o.choices].sort((a, b) =>
    (a.colour ? deltaE(a.colour, o.figureColour) : Infinity) - (b.colour ? deltaE(b.colour, o.figureColour) : Infinity));

  btn.addEventListener("click", () => {
    if (btn.getAttribute("aria-expanded") === "true") { open?.close(); return; }
    open?.close();
    const list = document.createElement("div");
    list.className = "slot-list";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", o.ariaLabel);
    list.tabIndex = -1;
    const items = ranked.map((c, i) => {
      const it = document.createElement("div");
      it.className = "slot-opt";
      it.setAttribute("role", "option");
      it.id = `slot-opt-${i}`;
      it.setAttribute("aria-selected", String(c.value === o.value));
      it.append(...choiceBody(c, o.figureColour));
      if (i === 0 && c.colour) it.querySelector(".sp-label")?.append(" ", Object.assign(document.createElement("em"),
        { textContent: "closest" }));
      it.addEventListener("click", () => pick(c.value));
      return it;
    });
    list.append(...items);
    document.body.append(list);

    let active = Math.max(0, ranked.findIndex((c) => c.value === o.value));
    const focus = (i: number): void => {
      active = (i + items.length) % items.length;
      items.forEach((it, k) => it.classList.toggle("active", k === active));
      list.setAttribute("aria-activedescendant", items[active].id);
      items[active].scrollIntoView({ block: "nearest" });
    };
    const place = (): void => {
      const r = btn.getBoundingClientRect();
      const w = Math.min(Math.max(r.width, 260), window.innerWidth - 16);
      const left = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
      const below = window.innerHeight - r.bottom - 8, above = r.top - 8;
      const up = below < 240 && above > below;
      list.style.width = `${w}px`;
      list.style.left = `${left}px`;
      list.style.maxHeight = `${Math.min(360, up ? above : below)}px`;
      list.style.top = up ? "" : `${r.bottom + 4}px`;
      list.style.bottom = up ? `${window.innerHeight - r.top + 4}px` : "";
    };
    const outside = (e: PointerEvent): void => {
      if (!list.contains(e.target as Node) && !btn.contains(e.target as Node)) close();
    };
    const key = (e: KeyboardEvent): void => {
      if (e.key === "ArrowDown") focus(active + 1);
      else if (e.key === "ArrowUp") focus(active - 1);
      else if (e.key === "Home") focus(0);
      else if (e.key === "End") focus(items.length - 1);
      else if (e.key === "Enter" || e.key === " ") pick(ranked[active].value);
      else if (e.key === "Escape" || e.key === "Tab") { close(); if (e.key === "Tab") return; }
      else return;
      e.preventDefault();
    };
    function close(): void {
      list.remove();
      btn.setAttribute("aria-expanded", "false");
      document.removeEventListener("pointerdown", outside, true);
      window.removeEventListener("resize", close);
      document.querySelector(".panel")?.removeEventListener("scroll", place);
      open = null;
      btn.focus({ preventScroll: true });
    }
    function pick(v: number): void {
      close();
      if (v !== o.value) o.onChange(v);
    }
    place();
    focus(active);
    btn.setAttribute("aria-expanded", "true");
    document.addEventListener("pointerdown", outside, true);
    window.addEventListener("resize", close);
    document.querySelector(".panel")?.addEventListener("scroll", place, { passive: true });
    list.addEventListener("keydown", key);
    list.focus({ preventScroll: true });
    open = { close };
  });
  return btn;
}
