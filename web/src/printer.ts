/**
 * The "Printer" card: the printer's live camera and what it is doing, through beambam. Read-only — nothing here
 * sends a command to the printer. Local version: pokefit/printer.py `print_progress` / `CameraProxy`, served at
 * POST /api/printer/status and /api/printer/camera. On the site (browser engine linked to the beambam daemon): the
 * daemon's /state through the engine's `print_progress`, and its camera relay (GET /camera.mjpeg), read with fetch
 * (`readMjpeg`) so it carries beambam's access token and Local Network Access like every other call.
 */
import * as api from "./api";
import { formatDuration } from "./geometry";
import { newestFrame, readMjpeg } from "./mjpeg";
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
  /** The current stream was already restarted once after an error. */
  private retried = false;
  /** The stream is being closed on purpose: its error event is not a camera failure. */
  private dropping = false;
  /** Through beambam: the running camera fetch (aborting it closes beambam's relay at once). */
  private stream: AbortController | null = null;

  constructor(
    private readonly button: HTMLButtonElement,
    private readonly status: HTMLElement,
    private readonly img: HTMLImageElement,
  ) {
    button.addEventListener("click", () => void (this.watching ? this.stop() : this.start()));
    // Local version only: there the <img> streams from pokefit's own camera proxy (same origin).
    img.addEventListener("error", () => { if (!api.viaDaemon()) void this.cameraFailed(); });
    document.addEventListener("visibilitychange", () => {
      if (!this.watching) return;
      if (document.hidden) {
        this.clearTimer();
        this.dropStream();                  // no ~1.5 MB/s camera stream for a tab nobody sees
      } else {
        void this.poll();
        void this.showCamera();
      }
    });
    // Leaving the page stops the camera child so ffmpeg does not keep pulling the printer's stream (local version;
    // through beambam the closed page closes the stream, and beambam's proxy stops its pump on its own).
    addEventListener("pagehide", () => {
      if (this.watching && !api.viaDaemon()) navigator.sendBeacon?.("/api/printer/camera", JSON.stringify({ on: false }));
    });
  }

  /** Local version: the stream failed. One quiet reconnect (the proxy restarted), then say so. */
  private async cameraFailed(): Promise<void> {
    if (!this.watching || this.dropping || !this.img.getAttribute("src")) return;
    if (!this.retried) {
      this.retried = true;
      const url = await api.printerCamera(true).catch(() => null);
      if (url && this.watching) this.img.src = url;
      return;
    }
    this.img.hidden = true;
    this.img.removeAttribute("src");
    this.status.textContent = "Camera: the live view stopped; press Stop watching, then Watch again";
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
    await this.showCamera();
  }

  /** Start the live view: through beambam a fetch-read stream, locally the image on the proxy's URL. */
  private async showCamera(): Promise<void> {
    if (api.viaDaemon()) {
      this.streamFromDaemon();
      return;
    }
    try {
      const url = await api.printerCamera(true);
      if (!this.watching || !url || document.hidden) return;
      this.retried = false;
      this.img.src = url;
      this.img.hidden = false;
    } catch (err) {
      this.img.hidden = true;
      this.status.textContent = `Camera: ${(err as Error).message}`;
    }
  }

  /**
   * beambam's relay, read with fetch: frames go into the image as they decode (`newestFrame`). A stream that ends
   * (the camera proxy restarted) reconnects once quietly; a refusal shows beambam's words in the status line.
   */
  private streamFromDaemon(): void {
    this.stream?.abort();
    if (!this.watching || document.hidden) return;
    const ctl = new AbortController();
    this.stream = ctl;
    const onFrame = newestFrame((url) => {
      if (url) {
        this.img.src = url;
        this.img.hidden = false;
      } else {
        this.img.removeAttribute("src");
      }
    }, ctl.signal);
    void (async () => {
      let ended = 0;
      while (!ctl.signal.aborted) {
        try {
          const res = await api.daemonCamera(ctl.signal);
          await readMjpeg(res, (jpeg) => {
            ended = 0;
            onFrame(jpeg);
          });
          if (++ended > 1) throw new Error("the live view stopped; press Stop watching, then Watch again");
        } catch (err) {
          if (ctl.signal.aborted) return;
          this.img.hidden = true;
          this.status.textContent = `Camera: ${(err as Error).message}`;
          return;
        }
      }
    })();
  }

  /** Close the MJPEG connection (a multipart <img> keeps streaming unless its src is cleared; a fetch is aborted). */
  private dropStream(): void {
    this.stream?.abort();
    this.stream = null;
    this.dropping = true;
    this.img.removeAttribute("src");
    this.img.hidden = true;
    queueMicrotask(() => { this.dropping = false; });
  }

  async stop(): Promise<void> {
    this.watching = false;
    this.clearTimer();
    this.button.textContent = "Watch the printer";
    this.dropStream();
    try {
      await api.printerCamera(false);
    } catch { /* the server is gone: nothing to stop */ }
  }
}
