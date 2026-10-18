/** Shapes of the pokefit JSON API (see pokefit/server.py and pokefit/pipeline.py). */

export type Status = "pass" | "warn" | "fail" | "info";

export interface Check {
  id: string;
  title: string;
  status: Status;
  message: string;
  value: number | string | null;
  /** Points in the figure frame (final mm, z = 0 on the bed). */
  highlight: number[][];
}

export interface Params {
  /** Print scale of the ball model (0.6 = the 60 % ball). */
  ball_scale: number;
  /** Inner diameter of the ball in mm; overrides ball_scale when set. */
  ball_diameter: number | null;
  washer: string;
  clearance: number;
  figure_scale: number | null;
  /** Degrees, any angle; applied about X, then Y, then Z. */
  rotate_x: number;
  rotate_y: number;
  rotate_z: number;
  /** Number of magnet pockets, 0–3. */
  magnets: number;
  pockets: { x: number; y: number }[] | null;
  pocket_diameter: number;
  pocket_depth: number;
  pocket_wall: number;
  pocket_ceiling: number;
  /** Min. pocket height surrounded by wall (on a sloped underside the magnet may stand proud below it). */
  pocket_grip: number;
  /** Automatic placement: highest pocket opening above the washer. */
  pocket_max_seat: number;
  /** Import file: figure filament -> project slot (1-based). */
  filament_map: Record<string, number> | null;
  object_id: string | null;
  slot: number | null;
  settings_source: "reference" | "original";
}

export interface Washer {
  key: string;
  label: string;
  od: number;
  id: number;
  thickness: number;
}

export interface BallInfo {
  scale: number;
  inner_diameter: number;
  r_inner: number;
  r_outer: number;
  floor_z: number;
  seat_z: number;
  top_z: number;
  ring_span: number;
  ring_h: number[];
  ring_r: number[];
  aperture: number;
  profile_z: number[];
  profile_r: number[];
  washer: Washer;
  annulus: [number, number];
  clearance: number;
}

export interface FitInfo {
  max_scale: number;
  scale: number;
  offset: [number, number];
  height_mm: number;
  max_height_mm: number;
  unit_height: number;
  unit_extents: number[];
  margin?: number;
  limit_kind?: string;
  limit_point?: number[];
}

export interface ObjectInfo {
  id: string;
  name: string;
  faces: number;
  parts: { name: string; extruder: number; faces: number; painted: number }[];
  colours: string[];
  objects: { id: string; name: string; faces: number }[];
}

export interface Underside {
  feet: number[][][];
  valid: number[][][];
  ring: { x: number; y: number; inner: number; outer: number; washer_od: number; washer_id: number };
  pocket_d: number;
  /** Underside heights as an image (dark = on the bed), drawn over `extent` [x0, y0, x1, y1]. */
  height_png: string;
  extent: [number, number, number, number];
  /** Height (mm) shown as the lightest shade. */
  height_span: number;
}

/** A magnet pocket: figure-frame XY and the height of its opening above the bed (mm). */
export interface PocketPos {
  x: number;
  y: number;
  z: number;
}

export interface WasherRow {
  key: string;
  label: string;
  max_scale: number;
  height_mm: number;
  pockets_on_steel: boolean;
  /** full = magnets wholly on steel, partial = centres on steel, off = not on steel. */
  steel: "full" | "partial" | "off";
  floor_z: number;
}

export interface Report {
  params: Params;
  object: ObjectInfo;
  ball: BallInfo;
  fit: FitInfo;
  pockets: PocketPos[];
  import_map: SlotMatch[];
  checks: Check[];
  status: Status;
  washers: WasherRow[];
  timings: Record<string, number>;
  underside: Underside;
}

export interface LoadResponse {
  session: string;
  file: string;
  report: Report;
}

export interface Slot {
  slot: string;
  name: string;
  type: string;
  colour: string;
}

export interface Config {
  washers: Washer[];
  default_washer: string;
  ball_scales: number[];
  params: Params;
  slots: Slot[];
  default_slot: number;
  reference: string;
  printer: string;
  process: string;
  slicer: string | null;
  /** "local": the Python server with Bambu slicing; "browser": the in-page engine. */
  engine: "local" | "browser";
  /** Native inner radius of the ball model (mm): inner diameter = 2 · this · ball scale. */
  ball_r_inner: number;
}

export interface SliceResult {
  ok: boolean;
  seconds: number;
  grams: number;
  warnings: string[];
  error: string;
  copies: number;
  wall_time: number;
  support_seconds: number;
  grams_per_filament: number[];
  features: Record<string, number>;
}

export interface SliceVariants {
  off: SliceResult;
  auto: SliceResult;
  x10?: SliceResult;
  supports_needed: boolean | null;
}

export interface Job<T> {
  state: "running" | "done" | "error";
  result?: T;
  error?: string;
}

export interface ExportResponse {
  files: { name: string; path: string; url: string }[];
  status: Status;
  note: string;
  import_3mf: string | null;
}

/** How one figure filament maps onto a slot of the user's project (for the import file). */
export interface SlotMatch {
  source: number;
  colour: string;
  target: number;
  target_colour: string;
  delta_e: number;
  /** Beyond the project's slots: Bambu adds a filament on import; set its colour there. */
  new: boolean;
  manual: boolean;
}
