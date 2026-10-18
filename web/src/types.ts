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
  ball_scale: number;
  washer: string;
  clearance: number;
  figure_scale: number | null;
  rotate_x: number;
  rotate_y: number;
  magnets: boolean;
  pockets: { x: number; y: number }[] | null;
  pocket_diameter: number;
  pocket_depth: number;
  pocket_wall: number;
  pocket_ceiling: number;
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
}

export interface WasherRow {
  key: string;
  label: string;
  max_scale: number;
  height_mm: number;
  pockets_on_steel: boolean;
  floor_z: number;
}

export interface Report {
  params: Params;
  object: ObjectInfo;
  ball: BallInfo;
  fit: FitInfo;
  pockets: { x: number; y: number }[];
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
}
