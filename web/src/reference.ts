/**
 * Colours from a reference picture (Colours card, "Colours from a picture"): load the official art of a Pokémon
 * (by name or number, from PokeAPI's copy of the official artwork) or any picture (open, drop or paste), then
 *  - tap the picture: the selected filament takes the colour there;
 *  - tap one of its main colours (chips under it): the same;
 *  - "Match the palette": every filament takes the picture's own colour of its family (the file's cream becomes the
 *    art's cream, its red the art's eye red), so colours follow the art instead of being guessed.
 *
 * A picked colour is the most common exact colour of the tone, not an average: art is shaded and anti-aliased, and
 * an average of a lit and a shaded cream (or of a red and its dark outline) is a colour the art never uses.
 *
 * The pure functions (sampling, main colours, matching, name guessing) are exported for tests.
 */
import { deltaE, labOf, textOn, toHex } from "./colour";

/** An RGBA picture (ImageData's shape). */
export interface Pixels {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** A main colour of the picture: its most common exact colour and its share of the picture's opaque pixels. */
export interface RefColour {
  hex: string;
  share: number;
}

/** Opaque pixels of a picture, sampled evenly, as sRGB and CIELAB (3 numbers each). */
export interface Samples {
  rgb: Uint8Array;
  lab: Float32Array;
  n: number;
}

/** A pixel counts if its alpha is at least this (anti-aliased borders against a transparent background do not). */
const OPAQUE = 250;
/** Most pixels sampled (the rest are skipped evenly): main colours and matching stay well under 100 ms. */
const MAX_SAMPLES = 60_000;
/** k-means clusters for the main colours: enough that a small saturated feature (an eye) gets its own. */
const CLUSTERS = 12;
/** k-means rounds. */
const ROUNDS = 15;
/** A main colour must be at least this share of the picture (Ninetales' eyes: 0.03 %) and this many samples. */
const MIN_SHARE = 0.0002;
const MIN_SAMPLES = 8;
/** Main colours closer than this (ΔE) are one. */
const MERGE_DE = 6;
/** Matching: the pixels within this ΔE of the closest one are the colour's family; the family's mode is the match. */
const FAMILY_DE = 15;
/** Matching: a filament with nothing in the picture closer than this keeps its colour (no family there). */
const MAX_MATCH_DE = 45;
/** Channel bin for the "most common exact colour" (8 levels of 256: anti-aliasing noise falls in one bin). */
const BIN = 8;

/** Sample the opaque pixels (evenly, at most MAX_SAMPLES). */
export function samplePixels(px: Pixels): Samples {
  const total = px.width * px.height;
  let opaque = 0;
  for (let i = 0; i < total; i++) if (px.data[4 * i + 3] >= OPAQUE) opaque++;
  const step = Math.max(1, opaque / MAX_SAMPLES);
  const cap = Math.min(opaque, MAX_SAMPLES);
  const rgb = new Uint8Array(3 * cap);
  const lab = new Float32Array(3 * cap);
  let n = 0;
  let seen = 0;
  let next = 0;
  for (let i = 0; i < total && n < cap; i++) {
    if (px.data[4 * i + 3] < OPAQUE) continue;
    if (seen++ < next) continue;
    next += step;
    const r = px.data[4 * i], g = px.data[4 * i + 1], b = px.data[4 * i + 2];
    rgb.set([r, g, b], 3 * n);
    lab.set(labOf(r, g, b), 3 * n);
    n++;
  }
  return { rgb, lab, n };
}

/** The most common exact colour of the samples `idx` (by BIN-wide channel bins; the bin's mean). */
function modeOf(s: Samples, idx: ArrayLike<number>): string {
  const count = new Map<number, number>();
  let best = -1, most = 0;
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j];
    const key = ((s.rgb[3 * i] / BIN) | 0) * 1024 + ((s.rgb[3 * i + 1] / BIN) | 0) * 32 + ((s.rgb[3 * i + 2] / BIN) | 0);
    const c = (count.get(key) ?? 0) + 1;
    count.set(key, c);
    if (c > most) { most = c; best = key; }
  }
  const sum = [0, 0, 0];
  let k = 0;
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j];
    const key = ((s.rgb[3 * i] / BIN) | 0) * 1024 + ((s.rgb[3 * i + 1] / BIN) | 0) * 32 + ((s.rgb[3 * i + 2] / BIN) | 0);
    if (key !== best) continue;
    for (let c = 0; c < 3; c++) sum[c] += s.rgb[3 * i + c];
    k++;
  }
  return k ? toHex(sum[0] / k, sum[1] / k, sum[2] / k) : "#000000";
}

function dist2(a: Float32Array, i: number, c: number[]): number {
  const d0 = a[3 * i] - c[0], d1 = a[3 * i + 1] - c[1], d2 = a[3 * i + 2] - c[2];
  return d0 * d0 + d1 * d1 + d2 * d2;
}

/**
 * The picture's main colours, most common first: k-means in CIELAB seeded by farthest points (so a small,
 * saturated feature such as an eye gets a cluster of its own), each cluster named by its most common exact colour.
 */
export function mainColours(s: Samples, k: number = CLUSTERS): RefColour[] {
  if (!s.n) return [];
  k = Math.min(k, s.n);
  // Seeds: the mean, then repeatedly the sample farthest from every seed so far (deterministic).
  const mean = [0, 0, 0];
  for (let i = 0; i < s.n; i++) for (let c = 0; c < 3; c++) mean[c] += s.lab[3 * i + c] / s.n;
  const centres: number[][] = [mean];
  const near = new Float32Array(s.n).fill(Infinity);
  while (centres.length < k) {
    const last = centres[centres.length - 1];
    let far = 0;
    for (let i = 0; i < s.n; i++) {
      near[i] = Math.min(near[i], dist2(s.lab, i, last));
      if (near[i] > near[far]) far = i;
    }
    if (near[far] === 0) break;
    centres.push([s.lab[3 * far], s.lab[3 * far + 1], s.lab[3 * far + 2]]);
  }
  const label = new Int32Array(s.n);
  for (let round = 0; round < ROUNDS; round++) {
    const sum = centres.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < s.n; i++) {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < centres.length; c++) {
        const d = dist2(s.lab, i, centres[c]);
        if (d < bd) { bd = d; bi = c; }
      }
      label[i] = bi;
      const t = sum[bi];
      t[0] += s.lab[3 * i]; t[1] += s.lab[3 * i + 1]; t[2] += s.lab[3 * i + 2]; t[3]++;
    }
    centres.forEach((c, j) => { if (sum[j][3]) for (let q = 0; q < 3; q++) c[q] = sum[j][q] / sum[j][3]; });
  }
  const members: number[][] = centres.map(() => []);
  for (let i = 0; i < s.n; i++) members[label[i]].push(i);
  const found = members.filter((m) => m.length >= Math.max(MIN_SAMPLES, MIN_SHARE * s.n))
    .sort((a, b) => b.length - a.length)
    .map((m) => ({ hex: modeOf(s, m), share: m.length / s.n }));
  // Two clusters of one tone (a large flat area split in two) show as one colour.
  const out: RefColour[] = [];
  for (const c of found) {
    const same = out.find((o) => deltaE(o.hex, c.hex) < MERGE_DE);
    if (same) same.share += c.share;
    else out.push({ ...c });
  }
  return out;
}

/**
 * The picture's own colour for `hex`: among the pixels within FAMILY_DE of the closest one (that colour's family in
 * the picture: the art's cream for a file's cream, the eye's red for a red), the most common exact colour. Null when
 * nothing in the picture is within MAX_MATCH_DE (the colour is not in the picture: a filament for a base, say).
 */
export function matchColour(hex: string, s: Samples): { hex: string; de: number } | null {
  if (!s.n) return null;
  const c = labOf(parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16));
  const d = new Float32Array(s.n);
  let min = Infinity;
  for (let i = 0; i < s.n; i++) { d[i] = Math.sqrt(dist2(s.lab, i, c)); if (d[i] < min) min = d[i]; }
  if (min > MAX_MATCH_DE) return null;
  const family: number[] = [];
  for (let i = 0; i < s.n; i++) if (d[i] <= min + FAMILY_DE) family.push(i);
  const out = modeOf(s, family);
  return { hex: out, de: deltaE(hex, out) };
}

/** The colour at pixel (x, y): the most common exact colour of the opaque pixels within `r` (null: transparent). */
export function colourAt(px: Pixels, x: number, y: number, r = 2): string | null {
  const rgb: number[] = [];
  for (let v = Math.max(0, y - r); v <= Math.min(px.height - 1, y + r); v++) {
    for (let u = Math.max(0, x - r); u <= Math.min(px.width - 1, x + r); u++) {
      const i = 4 * (v * px.width + u);
      if (px.data[i + 3] >= OPAQUE) rgb.push(px.data[i], px.data[i + 1], px.data[i + 2]);
    }
  }
  if (!rgb.length) return null;
  const s: Samples = { rgb: Uint8Array.from(rgb), lab: new Float32Array(rgb.length), n: rgb.length / 3 };
  // Prefer the tapped pixel's own bin when it is opaque (a tap on a thin line takes the line).
  const own = 4 * (y * px.width + x);
  if (px.data[own + 3] >= OPAQUE) {
    const near = [];
    for (let i = 0; i < s.n; i++) {
      if (Math.abs(s.rgb[3 * i] - px.data[own]) + Math.abs(s.rgb[3 * i + 1] - px.data[own + 1])
        + Math.abs(s.rgb[3 * i + 2] - px.data[own + 2]) <= 3 * BIN) near.push(i);
    }
    return modeOf(s, near);
  }
  return modeOf(s, Array.from({ length: s.n }, (_, i) => i));
}

/** Edit distance (insertions, deletions, substitutions). */
export function levenshtein(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}

/**
 * The Pokémon a file is named after: a word of `text` (file name, figure name) equal to a species name, or within
 * a typo of one (1 edit for 5-6 letters, 2 from 7: "ninetails" -> "ninetales"). Two words joined count too
 * ("mr mime" -> "mr-mime"). Null if none.
 */
export function guessName(text: string, names: readonly string[]): string | null {
  const words = text.toLowerCase().replace(/é/g, "e").split(/[^a-z0-9]+/).filter((w) => w.length >= 3 || /^mr|jr$/.test(w));
  const tries = [...words, ...words.slice(1).map((w, i) => `${words[i]}-${w}`)].filter((w) => /[a-z]/.test(w));
  const plain = names.map((n) => n.replace(/-/g, ""));
  let best: string | null = null;
  let bestScore = Infinity;
  for (const w of tries) {
    const flat = w.replace(/-/g, "");
    const allowed = flat.length >= 7 ? 2 : flat.length >= 5 ? 1 : 0;
    for (let i = 0; i < names.length; i++) {
      if (Math.abs(plain[i].length - flat.length) > allowed) continue;
      const d = flat === plain[i] ? 0 : levenshtein(flat, plain[i]);
      // Exact first, then fewer edits, then the longer name (a more specific match).
      const score = d * 100 - plain[i].length;
      if (d <= allowed && score < bestScore) { bestScore = score; best = names[i]; }
    }
  }
  return best;
}

/** PokeAPI: every species name (index + 1 = national number), and the official artwork by number. */
const SPECIES_URL = "https://pokeapi.co/api/v2/pokemon-species?limit=2000";
const ART_URL = (id: number): string =>
  `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/${id}.png`;

/** What the panel needs from the Colours card. */
export interface ReferenceHost {
  /** The palette in use (filament n = index n - 1). */
  palette(): string[];
  /** The selected filament (1-based; 0 = the eraser). */
  selected(): number;
  /** Use these colours for the palette (same length). */
  setColours(cols: string[]): void;
  /** The open figure's file and object names (to guess the Pokémon). */
  names(): string[];
  toast(msg: string): void;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

export class ReferencePanel {
  private pixels: Pixels | null = null;
  private samples: Samples | null = null;
  /** The picture's main colours (computed once per picture). */
  private chips: RefColour[] = [];
  private species: string[] | null = null;
  /** The palette before the last "Match the palette" (Undo match), while unchanged since. */
  private beforeMatch: { before: string[]; after: string } | null = null;
  /** The picture's label (shown under it). */
  private source = "";

  constructor(private host: ReferenceHost) {
    $("picFetch").addEventListener("click", () => void this.fetchArt($<HTMLInputElement>("picName").value));
    $("picName").addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); void this.fetchArt($<HTMLInputElement>("picName").value); }
    });
    $("picOpen").addEventListener("click", () => $("picFile").click());
    $<HTMLInputElement>("picFile").addEventListener("change", (e) => {
      const input = e.target as HTMLInputElement;
      const f = input.files?.[0];
      if (f) void this.loadBlob(f, f.name);
      input.value = "";
    });
    $("picMatch").addEventListener("click", () => this.matchPalette());
    $("picUndo").addEventListener("click", () => this.undoMatch());
    const canvas = $<HTMLCanvasElement>("picCanvas");
    canvas.addEventListener("click", (e) => this.pickAt(e));
    canvas.addEventListener("pointermove", (e) => this.hover(e));
    canvas.addEventListener("pointerleave", () => { $("picHover").hidden = true; });
    // A picture pasted anywhere (not into a text field) while the box is open becomes the reference.
    document.addEventListener("paste", (e) => {
      const el = e.target as HTMLElement | null;
      if (!($<HTMLDetailsElement>("picBox")).open || (el && ["INPUT", "TEXTAREA"].includes(el.tagName))) return;
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
      const f = item?.getAsFile();
      if (f) { e.preventDefault(); void this.loadBlob(f, "pasted picture"); }
    });
    $("picBox").addEventListener("toggle", () => { if ($<HTMLDetailsElement>("picBox").open) void this.suggest(); });
    this.render();
  }

  /** A picture file dropped on the page (not a figure): open the box and use it. */
  async useFile(f: File): Promise<void> {
    $<HTMLDetailsElement>("picBox").open = true;
    $("picBox").scrollIntoView({ block: "nearest" });
    await this.loadBlob(f, f.name);
  }

  /** A new figure: suggest its Pokémon's name (the picture stays: the same Pokémon is often opened again). */
  figureOpened(): void {
    $<HTMLInputElement>("picName").value = "";
    if ($<HTMLDetailsElement>("picBox").open) void this.suggest();
  }

  private async speciesNames(): Promise<string[]> {
    if (!this.species) {
      const res = await fetch(SPECIES_URL);
      if (!res.ok) throw new Error(`PokeAPI answered ${res.status}`);
      const body = await res.json() as { results: { name: string; url: string }[] };
      // Ordered by national number; the number is also the last part of each url.
      const names: string[] = [];
      for (const r of body.results) {
        const id = +(r.url.match(/\/(\d+)\/?$/)?.[1] ?? 0);
        if (id > 0) names[id - 1] = r.name;
      }
      this.species = names;
    }
    return this.species;
  }

  /** Fill the name box from the figure's file name when it names a Pokémon. */
  private async suggest(): Promise<void> {
    const box = $<HTMLInputElement>("picName");
    if (box.value.trim()) return;
    try {
      const guess = guessName(this.host.names().join(" "), await this.speciesNames());
      if (guess && !box.value.trim()) box.value = guess;
    } catch { /* offline: the user types a name */ }
  }

  /** Official artwork for a name or national number. */
  private async fetchArt(query: string): Promise<void> {
    const q = query.trim().toLowerCase().replace(/\s+/g, "-").replace(/é/g, "e").replace(/[.'’]/g, "");
    if (!q) { this.host.toast("Type a Pokémon's name or number first."); return; }
    this.info("Getting the official art…");
    try {
      let id = /^\d+$/.test(q) ? +q : 0;
      if (!id) {
        const names = await this.speciesNames();
        id = names.indexOf(q) + 1;
        if (!id) {
          const near = guessName(q, names);
          if (!near) { this.info(`No Pokémon called “${query.trim()}”.`); return; }
          id = names.indexOf(near) + 1;
          $<HTMLInputElement>("picName").value = near;
        }
      }
      const name = this.species?.[id - 1] ?? `#${id}`;
      await this.loadUrl(ART_URL(id), `Official art: ${name[0].toUpperCase()}${name.slice(1)} (#${id}, via PokeAPI)`);
    } catch (e) {
      this.info(`Could not get the art (${(e as Error).message}). Open or paste a picture instead.`);
    }
  }

  private async loadBlob(blob: Blob, label: string): Promise<void> {
    if (!blob.type.startsWith("image/")) { this.host.toast("That is not a picture."); return; }
    const url = URL.createObjectURL(blob);
    try { await this.loadUrl(url, label); } finally { URL.revokeObjectURL(url); }
  }

  private async loadUrl(url: string, label: string): Promise<void> {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    await img.decode().catch(() => { throw new Error("the picture did not load"); });
    // Work on at most 1024 px a side (sampling caps the cost anyway; this caps the memory).
    const f = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
    const w = Math.max(1, Math.round(img.naturalWidth * f)), h = Math.max(1, Math.round(img.naturalHeight * f));
    const work = document.createElement("canvas");
    work.width = w; work.height = h;
    const ctx = work.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("no 2D canvas");
    ctx.drawImage(img, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h);
    this.pixels = { data: data.data, width: w, height: h };
    this.samples = samplePixels(this.pixels);
    this.chips = mainColours(this.samples);
    this.source = label;
    const shown = $<HTMLCanvasElement>("picCanvas");
    shown.width = w; shown.height = h;
    shown.getContext("2d")?.putImageData(data, 0, 0);
    if (!this.samples.n) this.info("The picture is all transparent.");
    else this.info("");
    this.render();
  }

  /** The picture pixel under a pointer event on the shown canvas. */
  private pixelAt(e: MouseEvent): [number, number] | null {
    if (!this.pixels) return null;
    const c = $<HTMLCanvasElement>("picCanvas").getBoundingClientRect();
    const x = Math.floor((e.clientX - c.left) / c.width * this.pixels.width);
    const y = Math.floor((e.clientY - c.top) / c.height * this.pixels.height);
    return x >= 0 && y >= 0 && x < this.pixels.width && y < this.pixels.height ? [x, y] : null;
  }

  private hover(e: PointerEvent): void {
    const at = this.pixelAt(e);
    const hex = at && this.pixels ? colourAt(this.pixels, at[0], at[1]) : null;
    const tip = $("picHover");
    tip.hidden = !hex;
    if (hex) { tip.style.background = hex; tip.style.color = textOn(hex); tip.textContent = hex; }
  }

  private pickAt(e: MouseEvent): void {
    const at = this.pixelAt(e);
    const hex = at && this.pixels ? colourAt(this.pixels, at[0], at[1]) : null;
    if (hex) this.setSelected(hex);
  }

  /** The selected filament takes `hex`. */
  private setSelected(hex: string): void {
    const n = this.host.selected();
    if (n < 1) { this.host.toast("Select a filament (not the eraser) first, then tap the picture."); return; }
    const cols = this.host.palette();
    cols[n - 1] = hex;
    this.beforeMatch = null;
    this.host.setColours(cols);
    this.info(`Filament ${n} is now ${hex}.`);
    this.render();
  }

  /** Every filament takes the picture's colour of its family (`matchColour`); those with none keep theirs. */
  private matchPalette(): void {
    if (!this.samples) return;
    const before = this.host.palette();
    const after = before.map((c) => matchColour(c, this.samples as Samples)?.hex ?? c);
    const changed = after.filter((c, i) => c !== before[i]).length;
    const kept = before.filter((c) => !matchColour(c, this.samples as Samples)).length;
    if (!changed) { this.info("The palette already uses the picture's colours."); return; }
    this.host.setColours(after);
    this.beforeMatch = { before, after: after.join() };
    this.info(`${changed} filament${changed > 1 ? "s" : ""} now use${changed > 1 ? "" : "s"} the picture's colours` +
      (kept ? `; ${kept} with nothing like ${kept > 1 ? "them" : "it"} in the picture kept ${kept > 1 ? "theirs" : "its own"}` : "") +
      ". Check each: tap the picture to change one.");
    this.render();
  }

  private undoMatch(): void {
    if (!this.beforeMatch || this.host.palette().join() !== this.beforeMatch.after) return;
    this.host.setColours(this.beforeMatch.before);
    this.beforeMatch = null;
    this.info("The colours before the match are back.");
    this.render();
  }

  private info(text: string): void {
    $("picInfo").textContent = text;
    $("picInfo").hidden = !text;
  }

  /** Show the picture's state: canvas, chips, buttons. Called by the Colours card when the palette changes too. */
  render(): void {
    const has = this.chips.length > 0;
    $("picPicture").hidden = !this.pixels;
    $("picSource").textContent = this.source;
    ($("picMatch") as HTMLButtonElement).disabled = !has;
    $("picUndo").hidden = !(this.beforeMatch && this.host.palette().join() === this.beforeMatch.after);
    $("picChips").replaceChildren(...this.chips.map((c) => {
      const b = document.createElement("button");
      b.className = "ref-chip";
      b.type = "button";
      b.style.background = c.hex;
      b.style.color = textOn(c.hex);
      b.title = `${c.hex}: ${c.share >= 0.01 ? Math.round(c.share * 100) : (c.share * 100).toFixed(2)} % of the picture. ` +
        "Tap: the selected filament takes this colour.";
      b.setAttribute("aria-label", b.title);
      b.addEventListener("click", () => this.setSelected(c.hex));
      return b;
    }));
  }
}
