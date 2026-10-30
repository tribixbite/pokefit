/**
 * Paint dock: while painting, the Colours card's painting controls (palette, tap modes, brush / angle, Mirror,
 * Pick colour, Show regions, Shape map, Undo / Redo and the hint line) move onto the 3D view, so they sit beside
 * the figure instead of a scroll away; when painting stops they go back to their places in the card. The same
 * elements move (not copies), so their listeners and state stay as they are. The dock can be folded to its
 * palette row (remembered with the cards' collapsed state, collapse.ts).
 */
import { sectionToggle } from "./collapse";

export class PaintDock {
  /** Where each moved element came from: a marker left in its place in the card. */
  private homes = new Map<HTMLElement, Comment>();
  private shown = false;

  /**
   * `dock`: the overlay on the 3D view; `body`: its container for the moved controls; `ids`: the controls, in
   * dock order; `fold`: the button folding the dock to its first control (the palette); `cover(px)`: how much of
   * the view's bottom the dock covers now (0 when hidden), so the view can frame the figure above it.
   */
  constructor(private dock: HTMLElement, private body: HTMLElement, private ids: string[], fold: HTMLButtonElement,
    cover: (px: number) => void) {
    const bottom = (): number => parseFloat(getComputedStyle(dock).bottom) || 0;
    new ResizeObserver(() => cover(dock.hidden ? 0 : dock.offsetHeight + bottom())).observe(dock);
    sectionToggle(fold, "paint-dock", true, (open) => {
      dock.classList.toggle("folded", !open);
      fold.title = open ? "Fold the paint tools to the colours" : "Show all the paint tools";
      fold.setAttribute("aria-label", fold.title);
    });
  }

  get visible(): boolean {
    return this.shown;
  }

  /** Move the controls onto the view (`on`) or back into the card. */
  show(on: boolean): void {
    if (on === this.shown) return;
    this.shown = on;
    for (const id of this.ids) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (on) {
        const mark = document.createComment(`paint dock: ${id}`);
        el.replaceWith(mark);
        this.homes.set(el, mark);
        this.body.append(el);
      } else {
        const mark = this.homes.get(el);
        if (mark) mark.replaceWith(el);
        this.homes.delete(el);
      }
    }
    this.dock.hidden = !on;
    // The view's own overlays at the bottom (fit pill, legend) make way; on a phone the view stays on screen.
    document.body.classList.toggle("painting", on);
  }
}
