/**
 * The "Printer" card (local version with beambam): the printer's live camera and what it is doing.
 * Read-only — nothing here sends a command to the printer. pokefit/printer.py `print_progress` /
 * `CameraProxy`, served at POST /api/printer/status and /api/printer/camera.
 */
import * as api from "./api";
import { formatDuration } from "./geometry";
import type { PrintProgress } from "./types";

/** Status poll interval while watching: `beambam status` takes ~2.6 s, the X2D reports every few seconds. */
export const POLL_MS = 15_000;

const STATES: Record<string, string> = {
  IDLE: "Idle", PREPARE: "Preparing", RUNNING: "Printing", PAUSE: "Paused", FINISH: "Finished",
  FAILED: "Failed", SLICING: "Slicing", INIT: "Starting",
};

/** "248/250 °C" while heating or holding, "27 °C" when off. */
function temp(now: number, target: number): string {
  return target > 0 ? `${Math.round(now)}/${Math.round(target)} °C` : `${Math.round(now)} °C`;
}

/** One line for the card: state, job, progress and temperatures. */
export function progressText(p: PrintProgress): string {
  const state = STATES[p.state] ?? p.state.toLowerCase();
  const parts = [p.job ? `${state}: ${p.job}` : state];
  if (p.state === "RUNNING" || p.state === "PAUSE" || p.state === "PREPARE") {
    parts.push(`${p.percent} %`);
    if (p.layers) parts.push(`layer ${p.layer}/${p.layers}`);
    if (p.remaining_min > 0) parts.push(`${formatDuration(p.remaining_min * 60)} left`);
  }
  // Firmware extruder ids: 0 is the right nozzle, 1 the left (dual-nozzle printers).
  const nz = p.nozzles.map((n) => (p.nozzles.length > 1 ? `${n.id === 1 ? "L" : "R"} ` : "") + temp(n.temp, n.target));
  if (nz.length) parts.push(`nozzle${nz.length > 1 ? "s" : ""} ${nz.join(", ")}`);
  parts.push(`bed ${temp(p.bed, p.bed_target)}`);
  if (p.error) parts.push(`error ${p.error.toString(16).toUpperCase()}`);
  return parts.join(" · ");
}

/** Watch / stop watching: camera on, status polled while the page is visible. */
export class PrinterCard {
  private timer: number | null = null;
  private watching = false;

  constructor(
    private readonly button: HTMLButtonElement,
    private readonly status: HTMLElement,
    private readonly img: HTMLImageElement,
  ) {
    button.addEventListener("click", () => void (this.watching ? this.stop() : this.start()));
    document.addEventListener("visibilitychange", () => {
      if (!this.watching) return;
      if (document.hidden) this.clearTimer();
      else void this.poll();
    });
    // Leaving the page stops the camera child so ffmpeg does not keep pulling the printer's stream.
    addEventListener("pagehide", () => { if (this.watching) navigator.sendBeacon?.("/api/printer/camera", JSON.stringify({ on: false })); });
  }

  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }

  private async poll(): Promise<void> {
    this.clearTimer();
    try {
      this.status.textContent = progressText(await api.printerStatus());
    } catch (err) {
      this.status.textContent = (err as Error).message;
    }
    if (this.watching && !document.hidden) this.timer = window.setTimeout(() => void this.poll(), POLL_MS);
  }

  async start(): Promise<void> {
    this.watching = true;
    this.button.textContent = "Stop watching";
    this.status.textContent = "Connecting to the printer…";
    void this.poll();
    try {
      const url = await api.printerCamera(true);
      if (!this.watching || !url) return;
      this.img.src = url;
      this.img.hidden = false;
    } catch (err) {
      this.img.hidden = true;
      this.status.textContent = `Camera: ${(err as Error).message}`;
    }
  }

  async stop(): Promise<void> {
    this.watching = false;
    this.clearTimer();
    this.button.textContent = "Watch the printer";
    this.img.hidden = true;
    this.img.removeAttribute("src");          // closes the MJPEG connection
    try {
      await api.printerCamera(false);
    } catch { /* the server is gone: nothing to stop */ }
  }
}
