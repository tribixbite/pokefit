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
  /** "washer": every opening on the washer; "raised": openings move up into parts above the bed. */
  pocket_seat: "washer" | "raised";
  /** Import file: figure filament -> project slot (1-based). */
  filament_map: Record<string, number> | null;
  /** Colorize: filament colours (index 0 = filament 1); null = the file's. */
  colours: string[] | null;
  /** Colorize paint operations, applied in order. */
  paint: PaintOp[] | null;
  /** GLB/glTF colour (texture, vertex colours): filaments to reduce it to; null = automatic. */
  texture_colours: number | null;
  /** GLB: extra filaments for small, high-contrast details (eyes), 0..3 (gltf.TextureColours._details). */
  texture_details: number;
  object_id: string | null;
  slot: number | null;
  settings_source: "reference" | "original";
  /** An imported process preset's name; null = the project's own process. */
  process: string | null;
  /** Conform to the ball floor: bend strength (null = auto, 0 = off), flat base Ø (null = washer), press, fill (mm). */
  bend: number | null;
  flat_base: number | null;
  flat_depth: number;
  flat_fill: number;
  /** Fill the flat base over this circle instead of round the washer: figure frame at scale 1 (file mm). */
  fill_circle: { x: number; y: number; d: number } | null;
  /** Where gaps are filled: round the washer, under the whole figure, in `fill_circle`, or in `fill_shape`. */
  fill_area: FillArea;
  /** An outline drawn in the underside view, [[x, y], ...] at scale 1 like `fill_circle`. */
  fill_shape: number[][] | null;
  /** Print height in mm when figure_scale is null (capped at the largest that fits); null = the largest. */
  figure_height: number | null;
  /** Strengthen thin parts: minimum thickness (mm) at their joins, tapering to their tips; 0 = off. */
  strengthen: number;
  /** How far paint reaches into the figure, mm (Bambu mmu_segmented_region_max_width); null = the project's, 0 = unlimited. */
  paint_depth: number | null;
  /** Simplify the mesh to about this many triangles when it loads (shape within 0.1 % of its size); null = as in the file. */
  simplify: number | null;
  /** Painted patches smaller than this (mm² at the print size) take the colour around them; 0 = keep all. */
  despeckle: number;
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
  /** The conform step (pokefit/conform.py ConformInfo). */
  conform?: { bend: number; auto: boolean; lift_mm: number; flat_depth_mm: number; fill_mm2: number;
    scale_before: number; scale_after: number; note: string;
    /** Largest gap the flat base fills: the setting, or more when no magnet fitted with the default. */
    fill_mm: number };
  limit_point?: number[];
}

export interface ObjectInfo {
  id: string;
  name: string;
  faces: number;
  /** Triangles before / after the import's simplify step and the largest surface change (file mm); null = not simplified. */
  simplified: { faces_before: number; faces_after: number; error: number } | null;
  parts: { name: string; extruder: number; faces: number; painted: number }[];
  colours: string[];
  /** The file's own colours (before colorize changes). */
  file_colours: string[];
  /** GLB/glTF colour: the automatic filament count, the count in use and the most allowed. */
  texture_colours: { auto: number; count: number; max: number; details: number; max_details: number } | null;
  objects: { id: string; name: string; faces: number }[];
}

/** part: up to concave valleys; region: up to any crease; piece: connected shell; brush: a disc; face: one triangle. */
export type FillArea = "washer" | "all" | "circle" | "shape";

export type PaintMode = "wand" | "part" | "region" | "piece" | "fill" | "brush" | "face" | "stamp";

/** Paint the region (by `mode`, crease `angle` in degrees) of one source triangle with a filament (0 = erase). */
export interface PaintOp {
  part: number;
  face: number;
  filament: number;
  mode: PaintMode;
  angle: number;
  /** Brush radius in the file's mm (print mm ÷ figure scale). */
  radius: number;
  /** Brush stroke: more triangles of the same part along the drag. */
  path?: number[];
  /** Also paint the mirror image across the figure's symmetry plane. */
  mirror?: boolean;
  /** Widen (> 0) or narrow (< 0) the painted area, file mm. */
  grow?: number;
  /** Wand: 0 its best answer, -1 / +1 the next smaller / larger region. */
  level?: number;
  /** Stamp: the ellipse (centre, semi-axis u, semi-axis v; 9 numbers in the file's frame), from `PickResult.stamp`. */
  stamp?: number[];
}

/** Colour the preview by what a tap would paint. */
export interface RegionView {
  /** A paint mode: colour by what a tap would fill; "shape": the shape map (creases and relief). */
  mode: PaintMode | "shape";
  angle: number;
}

/** Colours of the preview glTF's faces (engine `preview_faces`). */
export interface PreviewFaces {
  /** Names the preview geometry: equal to the loaded glTF's = recolour it in place. */
  signature: string;
  /** "#rrggbb" per colour index. */
  colours: string[];
  /** Per mesh `p<i>`: base64, one colour index byte per face. */
  parts: string[];
  /** Which slot each used figure colour prints from (absent for the regions view). */
  import_map?: SlotMatch[];
  /** Shape map view: per mesh, base64 of two bytes per vertex (crease 0..255, relief with 128 = level). */
  shape?: string[];
}

export interface PickRequest {
  point: number[];
  normal: number[] | null;
  mode: PaintMode;
  angle: number;
  radius: number;
  /** Brush stroke: the rest of the drag after `point` (figure frame, scale 1). */
  stroke?: { point: number[]; normal: number[] | null }[];
  mirror?: boolean;
  grow?: number;
  /** Stamp: the ellipse's semi-axes u then v (6 numbers, figure frame, scale 1), seen along u × v. */
  stamp_axes?: number[];
}

/** `prepare_paint`: with `mirror` asked, whether the figure has a mirror plane to paint across. */
export interface PrepareResult {
  parts: number;
  mirror?: { ok: boolean; error: number; normal: number[] };
}

export interface PickResult {
  part: number;
  face: number;
  /** Brush stroke: the triangles under the rest of the drag. */
  path: number[];
  distance: number;
  faces: number;
  share: number;
  area_mm2: number;
  /** With mirror: a matching spot on the other side was painted too. */
  mirrored: boolean;
  /** The filament shown under the tap (the colour picker). */
  filament: number;
  /** Stamp: the ellipse in the paint operation's frame. */
  stamp?: number[];
}

export interface Underside {
  feet: number[][][];
  /** The flat base the gap fill added (absent from older reports). */
  fill?: number[][][];
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
  /** Estimated colour changes (pokefit/toolchange.py); empty for a one-colour figure. */
  toolchange: Partial<ToolchangeInfo>;
}

export interface ToolchangeCost { changes: number; purge_g: number; minutes: number }

export interface ToolchangeInfo extends ToolchangeCost {
  filaments: number;
  nozzles: number;
  switches: number;
  /** 1-based nozzle per filament in the best split (Bambu's Auto For Flush finds the same or better). */
  nozzle_of: number[];
  one_nozzle: ToolchangeCost;
  /** glTF colour: the cost of each palette size. */
  by_count?: (ToolchangeCost & { count: number })[];
  /** Best-value palette size (before the biggest jump in changes; pokefit/toolchange.best_value_count). */
  best_count?: number | null;
}

export interface LoadResponse {
  session: string;
  file: string;
  report: Report;
  /** Photos -> 3D only (pokefit/imageto3d.py report): a one-photo model's turn to face front. */
  photos?: { facing?: { turn_deg: number; front_sure: boolean; note: string } | null };
}

export interface Slot {
  slot: string;
  name: string;
  type: string;
  colour: string;
  /** A filament preset the user imported (not one of the project's AMS slots). */
  imported?: boolean;
  /** Why an imported preset cannot print with the project as set up (e.g. its plate). */
  warning?: string;
}

/** A filament loaded in the printer (pokefit/printer.py). */
export interface AmsSlot {
  slot: number;
  label: string;
  colour: string;
  type: string;
  name: string;
  /** Nozzle the AMS unit feeds (firmware id: 0 right, 1 left); null = either (track switch) or unknown. */
  extruder?: number | null;
}

/** What the printer is doing (pokefit/printer.py PrintProgress). */
export interface PrintProgress {
  state: string;
  job: string;
  percent: number;
  layer: number;
  layers: number;
  remaining_min: number;
  /** Firmware extruder ids: 0 right, 1 left. */
  nozzles: { id: number; temp: number; target: number }[];
  bed: number;
  bed_target: number;
  error: number;
}

/** What a print file will print (POST /api/printer/print preview). */
export interface PrintPlan {
  plate: string;
  name: string;
  file: string;
  filaments: { filament: number; preset: string; type: string; colour: string; bed_temp: string; slot: number;
    spool: AmsSlot | null }[];
  /** beambam's reading of the G-code (`beambam analyze`, file only); null with `check_error` when it failed. */
  check: PrintCheck | null;
  check_error: string;
}

/** `printer.print_file_check`: what the sliced G-code does, as beambam reads it. */
export interface PrintCheck {
  seconds: number | null;
  grams: number | null;
  purge_g: number | null;
  nozzle_swaps: number | null;
  tool_changes: number | null;
  /** Filament id -> nozzle index (0 / 1). */
  nozzles: Record<string, number | null>;
  warnings: string[];
  hints: string[];
}

/** What became of one imported Bambu preset. */
export interface PresetNote {
  kind: "filament" | "process" | "printer";
  name: string;
  status: "used" | "approximate" | "skipped" | "available";
  note: string;
}

export interface Config {
  washers: Washer[];
  default_washer: string;
  ball_scales: number[];
  params: Params;
  slots: Slot[];
  default_slot: number;
  /** The project's own process first, then imported process presets for the same printer. */
  processes: string[];
  presets: PresetNote[];
  reference: string;
  printer: string;
  process: string;
  slicer: string | null;
  /** "local": the Python server with Bambu slicing; "browser": the in-page engine. */
  engine: "local" | "browser";
  /** Native inner radius of the ball model (mm): inner diameter = 2 · this · ball scale. */
  ball_r_inner: number;
  /** Local engine with a reachable ComfyUI: "From photos" can make a figure. */
  photos?: boolean;
  /** Local engine with beambam installed: the printer's loaded filaments can be read. */
  printer_link?: boolean;
  /** The printer has a Filament Track Switch: any AMS spool can feed either nozzle. */
  track_switch?: boolean;
  /** The printer's loaded AMS filaments (once read): the slots colours are matched to. */
  ams: AmsSlot[];
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
  /** The sliced print file (.gcode.3mf) of the recommended variant, ready for the printer. */
  print_file?: { name: string; path: string; url: string; file: string } | null;
  /** AMS slot each filament of the print file is sliced for (once the printer's AMS was read). */
  ams_slots?: number[] | null;
}

export interface Job<T> {
  state: "running" | "done" | "error";
  result?: T;
  error?: string;
  /** What a long job is doing now (photos -> 3D). */
  progress?: string;
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
  /** The figure prints with it (false: a palette colour not painted yet). */
  used?: boolean;
  manual: boolean;
}
