/**
 * Which slice is current, and whether Slice can run now (main.ts).
 *
 * Every Slice and every change that makes a slice out of date (`stale`) moves `gen` on: a slice still running
 * when it moved was made from something else, and its result is dropped. The local server slices each job in its
 * own directory, so a new Slice may start while a dropped one finishes. The beambam daemon slices one at a time and
 * answers 409 "a slice is already running" to another, so while any daemon slice runs (a dropped one, or the
 * confirm panel's re-slice for other nozzles) Slice stays off, with a word on why.
 */
export class SliceGate {
  /** Bumped by every Slice and every change that makes a slice in flight out of date. */
  gen = 0;
  /** The `gen` of the Slice in flight (null: none; another value than `gen`: it was dropped). */
  private slicing: number | null = null;
  /** Slices the beambam daemon is running for this page. */
  private daemon = 0;

  /** `changed`: the button state may differ now (main.ts redraws it). */
  constructor(private readonly changed: () => void = () => {}) {}

  /** The figure, its project or its settings changed: a slice in flight is out of date. */
  stale(): void {
    this.gen++;
    this.changed();
  }

  /** A Slice starts: its `gen`, to hand back to `end` and compare with `isCurrent`. */
  begin(): number {
    this.slicing = ++this.gen;
    this.changed();
    return this.slicing;
  }

  /** The Slice of `gen` ended. Only the current one's end turns Slice back on: a dropped one ending must not
   *  while a newer one runs. */
  end(gen: number): void {
    if (this.slicing === gen) this.slicing = null;
    this.changed();
  }

  /** Nothing made the slice of `gen` out of date since it started. */
  isCurrent(gen: number): boolean {
    return gen === this.gen;
  }

  /** Run one beambam slice (a Slice or the panel's re-slice), counted while it runs. */
  async daemonSlice<T>(run: () => Promise<T>): Promise<T> {
    this.daemon++;
    this.changed();
    try {
      return await run();
    } finally {
      this.daemon--;
      this.changed();
    }
  }

  /** Slice's state: off without a slicer, while the current slice runs, and (through beambam) while the daemon is
   *  still busy with one. `waiting`: off for that last reason only, which the page explains under the button. */
  button(hasSlicer: boolean, viaDaemon: boolean): { disabled: boolean; waiting: boolean } {
    const busy = viaDaemon && this.daemon > 0;
    const running = this.slicing !== null && this.slicing === this.gen;
    return { disabled: (!hasSlicer && !viaDaemon) || running || busy, waiting: busy && !running };
  }

  /**
   * After a failed re-slice that started at `gen`: put back the nozzle choice it replaced, unless the slice it
   * belonged to was superseded meanwhile (a newer Slice, or a change that made it stale, which also cleared the
   * choice). Restoring then would pin the next slice to a map made for another print file.
   */
  restoreNozzles(p: { nozzles?: number[] | null }, before: number[] | null, gen: number): void {
    if (gen === this.gen) p.nozzles = before;
  }
}
