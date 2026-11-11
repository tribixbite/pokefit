"""One analysis pipeline shared by the CLI and the web server.

Frames:
  * figure frame — final print mm, XY centred on the figure's footprint
    bounding box, z = 0 on the bed. Pockets, check highlights and exported
    meshes use it.
  * ball frame — z = 0 at the ball centre, ball axis at XY 0. The figure frame
    sits in it at (offset_x, offset_y, cavity.floor_z).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .ball import SEAT_MARGIN, Ball, Cavity, load_ball
from .checks import (bed_contact_check, merged, islands_check, mesh_checks, overhang_check, printable_solid,
                     self_intersections, supports_verdict, thickness_check)
from .conform import (DEFAULT_FILL, DEFAULT_MAX_DEPTH, FILL_CELL, FILL_NAME, FILL_OVERLAP, ConformInfo, ConformSpec,
                      Shaped, base_fill, conform_fill, conform_shape, disc_region)
from .strengthen import Local, StrengthenInfo, Thickening, thicken, with_shells
from .colorize import (BOUNDED, MAX_PATH, MIRROR_MAX_ERROR, WAND_ANGLES, PaintOp, PaintState, TopologyCache,
                       PaintReplay, apply_paint, centroid_trees, colour_pieces, op_faces, pick_face, region_colours)
from .fit import FLOOR_MAX, HullPoints, annulus_margin, best_floor, best_offset, check_fit, max_scale, prepare_points
from .magnets import (HeightField, Pocket, PocketMap, PocketSpec, choose_pockets, cut_pockets, find_feet,
                      height_field, pocket_map, pockets_cut_cached, required_steel, seat_pockets, validate_pockets)
from .model3mf import (FigureObject, Part, Project, digest, parts_digest, plate_size, read_project,
                       remesh_part, world_parts, write_project)
from .gltf import MAX_DETAILS, MAX_PICK
from .paint import attr_paint, distinct_attrs, leaves, parse_tree, remap_states
from .presets import PresetLibrary, PresetNote, ams_filaments
from .printer import AmsSlot
from .remap import SlotMatch, match_slots, remap_parts, used_filaments
from .report import Check, worst
from . import toolchange as tc
from .shell import flat_spot, from_manifold, to_manifold
from .repair import repaired
from .stlcolour import write_coloured_stl
from .slicer import (SliceResult, adapt_settings, default_slot, grid_positions, reference_settings,
                     resize_filaments, slice_3mf)
from .reshape import apply_limbs, find_limb, find_limb_at, limb_ops
from .steel import CATALOGUE, DEFAULT_STEEL, Steel, get_steel

ROOT = Path(__file__).resolve().parent.parent
# The ball's derived measurements (radii, ring profile) ship with the package; the
# model itself (MakerWorld #2045372) is third-party and only needed locally for
# the steel seat shell export.
BUNDLED_BALL = Path(__file__).resolve().parent / "profiles" / "ball_functional_pokeball.json"
BALL_MODEL = ROOT / "pokeball.3mf"
DEFAULT_BALL = BUNDLED_BALL
# The user's own printer/process/filament settings: their printed-shells project
# if present, else the copy of its settings bundled with the package.
BUNDLED_REFERENCE = Path(__file__).resolve().parent / "profiles" / "x2d_petg_printed_ball_shells.json"
DEFAULT_REFERENCE = ROOT / "pokeball60goodPrinted.3mf" if (ROOT / "pokeball60goodPrinted.3mf").exists() else BUNDLED_REFERENCE
CACHE_DIR = ROOT / ".pokefit_cache"
SHELL_DISK = 8.0      # outer flat disk of the steel seat shell, final mm
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
SHAPE_CACHE = 4             # fit / underside / printability results kept per session (recent shapes)
# Params that change only colours, never the figure's shape, fit or pockets: editing them reuses those results.
COLOUR_ONLY = frozenset({"slot", "settings_source", "process", "filament_map", "colours", "paint", "colour_map",
                         "paint_depth", "simplify_auto",
                         "texture_colours", "texture_details", "despeckle"})
MIN_SCALE_FOR_MAGNETS = 0.6   # never shrink a figure below this fraction of its fit size for the magnets
DENSE_FACES = 300_000         # more triangles than this: suggest Simplify (`_dense_hint`) ...
AUTO_SIMPLIFY = 200_000       # ... or, with `Params.simplify_auto`, load it simplified to this many (`auto_simplify`)
SIMPLIFY_LOW = 0.07           # the UI's "Low" Simplify level (web/src/simplify.ts LEVELS), a share of the file
AUTO_FILLS = (4.0, 6.0)        # mm: deeper flat-base fills tried, in order, when no magnet pocket fits at the default
FILL_AREAS = ("steel", "all", "circle", "shape")
MAX_SHAPE_POINTS = 400        # points of a drawn fill outline
SHAPE_COLOUR = "#d9dde4"       # "Shape map" view: the figure's colour under its crease and relief shading
UNIT_STAMP = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # a stamp shape for checking the other fields
PAD_MARGIN = 0.4              # mm a pocket's pad reaches past its walled disc


@dataclass
class Params:
    ball_scale: float = 0.6                      # print scale of the ball model (0.6 = the 60 % ball)
    ball_diameter: float | None = None           # inner diameter in mm; overrides ball_scale when set
    steel: str = DEFAULT_STEEL
    # Diameter of the flat floor filled into the bottom shell (the steel is glued on it), mm: None = the one that lets
    # this figure fit largest (`Session.floor`), 0 = just the steel's own seat. The exported shell is cut to match.
    floor_diameter: float | None = None
    clearance: float = 1.0
    figure_scale: float | None = None            # None = largest that fits
    # A height in mm to print at (when figure_scale is None): kept across files by the UI, so it is a height, not
    # a scale. Capped at the largest that fits (with the magnets on the steel), with a note.
    figure_height: float | None = None
    rotate_x: float = 0.0                        # degrees, any angle; applied X, then Y, then Z
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    magnets: int = 2                             # number of magnet pockets, 0–3
    pockets: list[dict[str, float]] | None = None  # None = automatic; figure frame, final mm (x, y)
    pocket_diameter: float = 4.18
    pocket_depth: float = 2.20
    pocket_wall: float = 0.8
    pocket_ceiling: float = 0.6
    pocket_grip: float = 1.2                     # min. pocket height surrounded by wall (sloped undersides)
    pocket_max_seat: float = 2.0                 # raised, auto placement: highest pocket opening above the steel
    pocket_seat: str = "steel"                  # "steel": openings on the steel disc; "raised": into floating parts
    object_id: str | None = None
    slot: int | None = None                      # reference filament slot for slicing/export
    settings_source: str = "reference"           # "reference" (user's printer) or "original"
    process: str | None = None                   # an imported process preset's name (None = the project's own)
    # Import file: figure filament -> slot of the user's project (1-based; JSON keys are strings).
    filament_map: dict[str, int] | None = None
    # Colorize: the filament colours (index 0 = filament 1; None = the file's) and paint operations
    # (colorize.PaintOp dicts, applied in order).
    colours: list[str] | None = None
    paint: list[dict[str, Any]] | None = None
    # Colours removed from the palette: the file's own filament n (its paint, part and object filaments; for a glTF,
    # at the chosen count) prints as filament colour_map[n - 1]. Applied before `paint`, whose steps use the new
    # numbers. None = as in the file.
    colour_map: list[int] | None = None
    # glTF colour (texture, vertex colours, materials): how many filaments to reduce it to
    # (None = automatic, see gltf.py). Ignored for other files.
    texture_colours: int | None = None
    # ... plus up to this many "detail" filaments: small features that contrast most with the colour they would
    # merge into (black pupils on a cream head) keep their own (gltf.TextureColours._details). 0..MAX_DETAILS.
    texture_details: int = 0
    # Conform to the ball's floor (conform.py): bend legs/tails that reach past the steel up along the
    # sphere (None = automatic, 0 = off, else strength 0..1) and make the base flat over a disc of
    # `flat_base` mm (None = the steel's diameter, 0 = off): press it by at most `flat_depth` mm,
    # then fill gaps up to `flat_fill` mm under the figure with a solid "flat base" part.
    bend: float | None = None
    flat_base: float | None = None
    flat_depth: float = DEFAULT_MAX_DEPTH
    flat_fill: float = DEFAULT_FILL
    # The flat base's fill over a circle the user places instead of round the steel: {x, y, d} in the figure
    # frame at scale 1 (file mm, like the figure itself, so it stays put under the figure at any size). Inside
    # it every gap under the figure up to `flat_fill` is filled, connected or not (None = automatic).
    fill_circle: dict[str, float] | None = None
    # Where gaps are filled: "steel" (round the steel disc, joined to where the figure stands), "all" (every part
    # hovering up to `flat_fill` above the bed, anywhere under the figure), "circle" (`fill_circle`) or "shape"
    # (`fill_shape`: an outline the user drew, [[x, y], ...] in the same frame as `fill_circle`).
    fill_area: str = "steel"
    fill_shape: list[list[float]] | None = None
    # Strengthen thin parts (strengthen.py): thin features get this minimum thickness in mm where they join the
    # figure, tapering toward their tips (0 = off). Measured at `figure_height` if set, else the largest size.
    strengthen: float = 0.0
    # Single features thickened to their own minimum (with `strengthen` for the rest, or alone): [{"part": i,
    # "face": f, "mm": t, "root": r}] = the thin feature holding triangle f of `base_parts`' part i gets t mm where it
    # joins, on the far side of triangle r (where it would break; -1 = anywhere) (strengthen.Local). The UI's
    # weak-points list (`weak_points`, joints.py) fills it. None / [] = none.
    strengthen_at: list[dict[str, Any]] | None = None
    # How far painted colours reach into the figure, in mm (Bambu `mmu_segmented_region_max_width`; deeper is
    # the object's own filament). None = the project's value (0 there = unlimited: paint fills inward until it
    # meets other paint). Measured on an 8-colour figure: time and colour changes stay the same (the colours
    # still appear on the same layers); only how much of each colour, and of the base, is used.
    paint_depth: float | None = None
    # Simplify dense meshes when the file loads (simplify.py; None = as in the file): to about this many triangles
    # (>= 1,000), or to this share of the file's triangles (0 < x < 1: the UI's low / medium / high levels).
    # The shape moves by at most 0.1 % of the figure's size; fewer triangles make every later step faster.
    simplify: float | None = None
    # Opening a file: a figure of more than DENSE_FACES triangles, with `simplify` unset, loads simplified to
    # AUTO_SIMPLIFY (`auto_simplify`; the UI then shows that setting, and Off keeps every triangle).
    simplify_auto: bool = True
    # Limbs resized or bent (reshape.py `LimbOp` dicts: part, face, angle, scale, bend, swing), applied in order to
    # the figure as rotated, before the floor is chosen and the conform step runs. None / [] = as in the file.
    limbs: list[dict[str, Any]] | None = None
    # Painted patches smaller than this (mm² at the print size) take the colour around them: a stray dab costs a
    # filament change on every layer it spans (colorize.Topology.specks). 0 = keep every patch.
    despeckle: float = 0.5
    # Print weak features solid (joints.reinforcements): the export gets Bambu modifier volumes over every thin feature
    # graded C or worse (more walls, 100 % infill inside). The shape is unchanged.
    reinforce: bool = False

    @property
    def strengthens(self) -> bool:
        """Whether thin parts are thickened at all (everywhere, or single features)."""
        return self.strengthen > 0 or bool(self.strengthen_at)

    def local_strengthen(self) -> list[Local]:
        return [Local(q["part"], q["face"], q["mm"], q.get("root", -1), q.get("probe", 0.0))
                for q in self.strengthen_at or []]

    def __post_init__(self) -> None:
        # Older clients send magnets as a switch: on = the default two pockets.
        if isinstance(self.magnets, bool):
            self.magnets = 2 if self.magnets else 0
        self.magnets = max(0, min(3, int(self.magnets)))
        # Before steel discs (2026-11-04) the steel was always a washer: the old names still mean the same.
        if self.pocket_seat == "washer":
            self.pocket_seat = "steel"
        if self.fill_area == "washer":
            self.fill_area = "steel"
        if self.pocket_seat not in ("steel", "raised"):
            raise ValueError(f"pocket_seat must be 'steel' or 'raised', not {self.pocket_seat!r}")
        if self.colours is not None:
            if not self.colours or not all(isinstance(c, str) and _HEX.fullmatch(c) for c in self.colours):
                raise ValueError("colours must be a non-empty list of #RRGGBB strings")
            if len(self.colours) > 32:
                raise ValueError("at most 32 colours")
        for d in self.paint or []:
            PaintOp.from_dict(d)          # raises ValueError on a malformed operation
        if self.colour_map is not None:
            if not isinstance(self.colour_map, list) or len(self.colour_map) > 64:
                raise ValueError("colour_map must be a list of at most 64 filament numbers")
            self.colour_map = [int(n) for n in self.colour_map]
            if not all(1 <= n <= 64 for n in self.colour_map):
                raise ValueError("colour_map entries must be filament numbers 1-64")
        if self.texture_colours is not None:
            self.texture_colours = int(self.texture_colours)
            if not 1 <= self.texture_colours <= MAX_PICK:
                raise ValueError(f"texture_colours must be 1..{MAX_PICK}")
        self.texture_details = int(self.texture_details)
        if not 0 <= self.texture_details <= MAX_DETAILS:
            raise ValueError(f"texture_details must be 0..{MAX_DETAILS}")
        if self.bend is not None and not 0 <= float(self.bend) <= 1:
            raise ValueError("bend must be 0..1 (or null for automatic)")
        if self.limbs:
            self.limbs = [op.to_dict() for op in limb_ops(self.limbs)]
        if self.floor_diameter is not None and not 0 <= float(self.floor_diameter) <= 60:
            raise ValueError("floor_diameter must be 0..60 mm (or null for the best for the figure)")
        if self.flat_base is not None and not 0 <= float(self.flat_base) <= 60:
            raise ValueError("flat_base must be 0..60 mm (or null for the steel's diameter)")
        if not 0 <= float(self.flat_depth) <= 5:
            raise ValueError("flat_depth must be 0..5 mm")
        if not 0 <= float(self.flat_fill) <= 10:
            raise ValueError("flat_fill must be 0..10 mm")
        self.strengthen = float(self.strengthen)
        if not 0 <= self.strengthen <= 5:
            raise ValueError("strengthen must be 0..5 mm (0 = off)")
        if self.strengthen_at:
            at = [{"part": int(q["part"]), "face": int(q["face"]), "mm": round(float(q["mm"]), 3),
                   "root": int(q.get("root", -1)), **({"probe": float(q["probe"])} if q.get("probe") else {})}
                  for q in self.strengthen_at]
            if any(not 0 < q["mm"] <= 5 or q["part"] < 0 or q["face"] < 0 for q in at):
                raise ValueError("strengthen_at: each needs part >= 0, face >= 0 and 0 < mm <= 5")
            if any(not 1 <= q.get("probe", 3.0) <= 10 for q in at):
                raise ValueError("strengthen_at: probe must be 1..10 mm (the weak point's probe_mm)")
            self.strengthen_at = at
        if self.figure_height is not None:
            self.figure_height = float(self.figure_height)
            if not 1 <= self.figure_height <= 1000:
                raise ValueError("figure_height must be 1..1000 mm (or null for the largest that fits)")
        if self.fill_area not in FILL_AREAS:
            raise ValueError(f"fill_area must be one of {', '.join(FILL_AREAS)}, not {self.fill_area!r}")
        if self.fill_circle is not None and self.fill_area == "steel":
            self.fill_area = "circle"           # older clients: a circle alone meant "fill inside it"
        if self.fill_shape is not None:
            try:
                pts = [(float(x), float(y)) for x, y in self.fill_shape]
            except (TypeError, ValueError) as exc:
                raise ValueError("fill_shape must be a list of [x, y] points in mm") from exc
            if not 3 <= len(pts) <= MAX_SHAPE_POINTS or not np.isfinite(pts).all():
                raise ValueError(f"fill_shape needs 3..{MAX_SHAPE_POINTS} finite points")
            self.fill_shape = [[x, y] for x, y in pts]
        if self.fill_area == "shape" and self.fill_shape is None:
            raise ValueError("fill_area 'shape' needs fill_shape")
        if self.fill_circle is not None:
            try:
                self.fill_circle = {k: float(self.fill_circle[k]) for k in ("x", "y", "d")}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("fill_circle must be {x, y, d} in mm") from exc
            if not self.fill_circle["d"] > 0 or not all(np.isfinite(v) for v in self.fill_circle.values()):
                raise ValueError("fill_circle needs a diameter > 0")
        if self.paint_depth is not None:
            self.paint_depth = float(self.paint_depth)
            if not 0 <= self.paint_depth <= 10:
                raise ValueError("paint_depth must be 0..10 mm (0 = unlimited, or null for the project's)")
        self.despeckle = float(self.despeckle)
        if not 0 <= self.despeckle <= 20:
            raise ValueError("despeckle must be 0..20 mm² (0 = off)")
        if self.simplify is not None:
            b = float(self.simplify)
            if 0 < b < 1:
                self.simplify = round(b, 4)               # a share of the file's triangles
            elif b in (0, 1):
                self.simplify = None                      # 0 % / 100 %: as in the file
            elif 1_000 <= b <= 10_000_000 and b == int(b):
                self.simplify = int(b)
            else:
                raise ValueError("simplify must be a share of the file's triangles (0..1), 1,000..10,000,000 "
                                 "triangles, or null for the file's own")

    @property
    def paint_ops(self) -> list[PaintOp]:
        return [PaintOp.from_dict(d) for d in self.paint or []]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        if "washer" in d and "steel" not in d:
            d = {**d, "steel": d["washer"]}       # older clients and remembered settings
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def spec(self) -> PocketSpec:
        # On the steel, the gap fill also pads each pocket: small gaps under its walled disc are filled, so the
        # magnet resting on the steel is walled from the steel up (`pocket_pads`).
        flush = self.pocket_seat == "steel"
        return PocketSpec(self.pocket_diameter, self.pocket_depth, self.pocket_wall, self.pocket_ceiling,
                          self.pocket_grip, self.pocket_max_seat, flush, float(self.flat_fill) if flush else 0.0)


@dataclass
class Session:
    id: str
    project: Project
    work_dir: Path
    ball: Ball
    reference: dict | None
    presets: PresetLibrary | None = None        # Bambu presets the user imported (presets.py)
    simplify: float | None = None               # triangle budget `project` was loaded with (Params.simplify)
    # Filaments loaded in the user's printer (printer.py, via beambam): when set, the figure's colours are
    # matched to these slots instead of the reference project's.
    ams: list[AmsSlot] | None = None
    input_checks: dict[str, list[Check]] = field(default_factory=dict)   # object id -> checks
    _frames: dict[tuple, list[Part]] = field(default_factory=dict)           # orientation + conform -> parts
    _oriented: dict[tuple, list[Part]] = field(default_factory=dict)         # orientation only
    _conform: dict[tuple, ConformInfo] = field(default_factory=dict)
    _strength: dict[tuple, StrengthenInfo] = field(default_factory=dict)       # same key -> strengthen.py result
    # Same key -> the conform step's parts before strengthen (the triangles paint steps name) and what strengthen adds
    # to them (`unit_parts` = the two united; paint is applied to the first, then united: `figure_parts`).
    _base: dict[tuple, list[Part]] = field(default_factory=dict)
    _thick: dict[tuple, Thickening] = field(default_factory=dict)
    _fills: dict[tuple, float] = field(default_factory=dict)                 # `_auto_fill` decisions
    _points: dict[tuple[str, float, float, float], HullPoints] = field(default_factory=dict)
    _fields: dict[tuple[str, float, float, float], HeightField] = field(default_factory=dict)  # unit-scale undersides
    # Auto-placement results per configuration (the figure-scale-independent search).
    _auto: dict[tuple, Any] = field(default_factory=dict)
    _topo: dict[str, TopologyCache] = field(default_factory=dict)       # object id -> colorize topology
    _painted: dict[tuple, list[Part]] = field(default_factory=dict)     # (orientation, paint) -> parts
    _replay: PaintReplay = field(default_factory=PaintReplay)          # paint after recent paint lists
    _mapped: dict[tuple, list[Part]] = field(default_factory=dict)      # (orientation, colour_map) -> parts
    _used: dict[str, list[int]] = field(default_factory=dict)   # object id -> filament numbers it uses
    _textured: dict[tuple, list[Part]] = field(default_factory=dict)    # (orientation, count) -> parts
    _settings: dict[tuple, dict] = field(default_factory=dict)          # (presets version, process) -> settings
    _tc_table: dict[tuple, list[dict[str, Any]]] = field(default_factory=dict)   # palette-size cost tables
    _auto_count: dict[tuple, int | None] = field(default_factory=dict)     # glTF auto palette size per setup
    _printing: dict[tuple, list[Check]] = field(default_factory=dict)      # geometry -> printability checks
    _solved: dict[str, tuple] = field(default_factory=dict)                # shape_key -> `_solve` result
    _undersides: dict[str, dict[str, Any]] = field(default_factory=dict)   # shape_key -> pocket editor data
    _preview: dict[str, Any] = field(default_factory=dict)                 # last `_preview_meshes` result
    _pads: dict[str, Part | None] = field(default_factory=dict)            # shape_key -> `pocket_pads` (unit scale)
    _pick_trees: dict[tuple, list[Any]] = field(default_factory=dict)      # figure frame -> `centroid_trees`
    _paint_scale: dict[tuple, float] = field(default_factory=dict)         # `paint_scale` per setup
    _floors: dict[tuple, float] = field(default_factory=dict)              # `floor` (auto) per setup
    _shaped: dict[tuple, list[Part]] = field(default_factory=dict)          # orientation + limbs -> parts
    _selfint: dict[str, list[Check]] = field(default_factory=dict)          # object id -> self-intersection checks
    _shapes: dict[tuple, Shaped] = field(default_factory=dict)              # conform step before its fill

    def obj(self, params: Params) -> FigureObject:
        return self.project.object(params.object_id) if params.object_id else self.project.default_object()

    def use_simplify(self, target: float | None) -> None:
        """Load the file again simplified to `target` triangles (None = as in the file) if it was loaded with
        another budget. Every cached result belongs to the old triangles, so all are dropped (paint steps name
        triangles too: the UI clears them when this setting changes)."""
        if (target or None) == self.simplify:
            return
        self.project = read_project(self.project.path, target or None)
        self.simplify = target or None
        self.input_checks.clear()
        for f in fields(self):
            if f.name.startswith("_"):
                getattr(self, f.name).clear()

    def _orientation(self, params: Params) -> tuple[str, float, float, float]:
        """Object id and the three angles (degrees, mod 360)."""
        return (self.obj(params).id, round(params.rotate_x % 360, 6), round(params.rotate_y % 360, 6),
                round(params.rotate_z % 360, 6))

    def _key(self, params: Params) -> tuple:
        """Cache key of the figure-frame geometry: the orientation, plus what the conform step
        depends on when it is on (its settings, the ball size and the clearance; the steel only
        through the flat disc's diameter, see `unit_parts`)."""
        return self._spec_key(params, self.conform_spec(params))

    def _figure_key(self, params: Params) -> tuple:
        """The figure before the conform step: its orientation and the limbs reshaped."""
        okey = self._orientation(params)
        return okey + (json.dumps(params.limbs, sort_keys=True),) if params.limbs else okey

    def _spec_key(self, params: Params, spec: ConformSpec | None) -> tuple:
        on = params.strengthens
        strong = (round(params.strengthen, 3), params.figure_height if on else None,
                  round(self.ball_scale(params), 6) if on else None, params.clearance if on else None,
                  json.dumps(params.strengthen_at) if params.strengthen_at else None)
        if spec is None:
            return self._figure_key(params) + strong + (self._floor_key(params) if on else ())
        return self._figure_key(params) + (spec.bend, spec.flat_diameter, spec.max_depth, spec.fill,
                                             spec.fill_circle, spec.fill_shape, spec.fill_all,
                                             round(self.ball_scale(params), 6), params.clearance) + strong + \
            self._floor_key(params)

    def _floor_key(self, params: Params) -> tuple:
        """What of the floor the conform and strengthen steps see: its height and width."""
        c = self.cavity(params)
        return (round(c.seat_diameter, 3), round(c.steel.thickness, 4))

    def floor(self, params: Params, steel: Steel | None = None) -> float:
        """Diameter of the flat floor in the bottom shell: the one asked for (at least the steel's seat), else the
        one that lets the figure as loaded (rotated, before the conform step) fit largest (`fit.best_floor`).
        Decided on the unshaped figure, so the conform step can be designed against it."""
        st = steel or get_steel(params.steel)
        seat = st.od + SEAT_MARGIN
        if params.floor_diameter is not None:
            return max(seat, float(params.floor_diameter))
        key = (self._figure_key(params), round(self.ball_scale(params), 6), st.od, st.thickness, params.clearance)
        if key not in self._floors:
            raw = self._shaped_parts(params)
            hp = prepare_points(np.vstack([q.vertices for q in raw]))
            ball_s = self.ball_scale(params)
            d_max = max(seat, FLOOR_MAX * 2 * self.ball.r_inner * ball_s)
            d, _ = best_floor(hp, lambda d: Cavity(self.ball, ball_s, st, params.clearance, d), seat, d_max)
            self._floors[key] = d
        return self._floors[key]

    def cavity(self, params: Params, steel: Steel | None = None) -> Cavity:
        """The ball's cavity for these settings: its scale, the steel (`steel` overrides the setting's) on its floor
        (`floor`), the clearance."""
        st = steel or get_steel(params.steel)
        return Cavity(self.ball, self.ball_scale(params), st, params.clearance, self.floor(params, st))

    def conform_spec(self, params: Params) -> ConformSpec | None:
        """The conform step's settings, or None when bend and flat base are both off. The default flat-base
        fill is raised (`AUTO_FILLS`) when no magnet pocket fits under the figure with it (`_auto_fill`)."""
        flat = get_steel(params.steel).od if params.flat_base is None else float(params.flat_base)
        bend = None if params.bend is None else float(params.bend)
        area = params.fill_area
        circle = params.fill_circle if area == "circle" else None
        shape = tuple((float(x), float(y)) for x, y in params.fill_shape) if area == "shape" and params.fill_shape else None
        pressed = flat > 0 and params.flat_depth > 0
        filled = params.flat_fill > 0 and (flat > 0 or area != "steel")
        if bend == 0 and not pressed and not filled:
            return None
        spec = ConformSpec(bend, flat, float(params.flat_depth), float(params.flat_fill),
                           fill_circle=(circle["x"], circle["y"], circle["d"]) if circle else None,
                           fill_shape=shape, fill_all=area == "all")
        if spec.fill == DEFAULT_FILL and flat > 0 and params.magnets and params.pockets is None and area == "steel":
            spec = replace(spec, fill=self._auto_fill(params, spec))
        return spec

    def _auto_fill(self, params: Params, spec: ConformSpec) -> float:
        """The flat-base fill to use at the default setting: DEFAULT_FILL, or the first of AUTO_FILLS under
        which all `params.magnets` pockets fit at the largest size (a figure on four thin legs, like an Eevee,
        has room for none until the base fills the gap under its belly). When no fill holds them all, the
        first that holds the most (none: DEFAULT_FILL)."""
        key = self._spec_key(params, spec) + (params.spec, params.magnets)
        if key not in self._fills:
            self._fills[key] = spec.fill
            most = 0
            for fill in (spec.fill, *AUTO_FILLS):
                parts = self._conformed(params, replace(spec, fill=fill))
                cavity = self.cavity(params)
                free = max_scale(self._points[self._spec_key(params, replace(spec, fill=fill))], cavity)
                if free is None:
                    break
                fkey = self._spec_key(params, replace(spec, fill=fill))
                if fkey not in self._fields:     # the same underside `underside` needs later
                    self._fields[fkey] = height_field([p.mesh() for p in parts if p.subtype == "normal_part"])
                field = self._fields[fkey]
                pspec = replace(params.spec, pad=fill) if params.spec.pad > 0 else params.spec   # pads fill as deep
                pockets, _ = choose_pockets(pocket_map(field.scaled(free.max_scale), pspec), pspec, params.magnets)
                if len(pockets) > most:          # a deeper fill only where it places more pockets
                    self._fills[key], most = fill, len(pockets)
                if most >= params.magnets:
                    break
        return self._fills[key]

    def pocket_spec(self, params: Params) -> PocketSpec:
        """`params.spec` with the pads filling as deep as the flat base does: the fill raised by `_auto_fill`
        (typed 2.5 mm, filled 6) pads the pockets 6 mm deep too, as a typed 6 does."""
        spec = params.spec
        cspec = self.conform_spec(params) if spec.pad > 0 else None
        return replace(spec, pad=cspec.fill) if cspec is not None and cspec.fill != spec.pad else spec

    def conform_info(self, params: Params) -> ConformInfo | None:
        self.unit_parts(params)
        return self._conform.get(self._key(params))

    def ball_scale(self, params: Params) -> float:
        """Print scale of the ball: from the inner diameter when one is given."""
        if params.ball_diameter:
            return float(params.ball_diameter) / (2 * self.ball.r_inner)
        return params.ball_scale

    def unit_parts(self, params: Params) -> list[Part]:
        """Figure-frame parts at scale 1 (rotated, XY-centred, on z = 0), thin parts strengthened: the shape that is
        fitted, cut and printed. Strengthen re-meshes what it thickens, so paint steps name `base_parts`' triangles."""
        return self._conformed(params, self.conform_spec(params))

    def base_parts(self, params: Params) -> list[Part]:
        """`unit_parts` before strengthen (the same parts when it is off): the triangles paint steps, taps, the
        topology and a glTF's colour name."""
        self.unit_parts(params)
        return self._base[self._key(params)]

    def _conformed(self, params: Params, spec: ConformSpec | None) -> list[Part]:
        """`unit_parts` with the conform step's settings given (cached under the same key)."""
        key = self._spec_key(params, spec)
        if key not in self._frames:
            parts = self._shaped_parts(params)
            if spec is not None:
                # Designed against this setup's floor (its height and width: `_floor_key` is in the key).
                cavity = self.cavity(params)
                # Vertices move, faces keep their order: paint and texture colour carry over. The bend and press
                # do not depend on the fill settings: a fill edit (or `_auto_fill` trying deeper fills) re-fills.
                skey = self._spec_key(params, replace(spec, fill=0.0, fill_circle=None, fill_shape=None,
                                                      fill_all=False))
                if skey not in self._shapes:
                    if len(self._shapes) > 4:
                        self._shapes.clear()
                    self._shapes[skey] = conform_shape(parts, cavity, spec)
                parts, self._conform[key] = conform_fill(self._shapes[skey], spec)
            self._base[key] = parts
            if params.strengthens:
                parts = self._strengthened(params, parts, key)
            self._frames[key] = parts
            self._points[key] = prepare_points(np.vstack([p.vertices for p in parts]))
        return self._frames[key]

    def _oriented_parts(self, params: Params) -> list[Part]:
        """The object's parts rotated, XY-centred and on z = 0, before the conform step (cached)."""
        okey = self._orientation(params)
        if okey not in self._oriented:
            parts = world_parts(self.obj(params))
            rot = (trimesh.transformations.rotation_matrix(np.radians(okey[3]), [0, 0, 1]) @
                   trimesh.transformations.rotation_matrix(np.radians(okey[2]), [0, 1, 0]) @
                   trimesh.transformations.rotation_matrix(np.radians(okey[1]), [1, 0, 0]))
            parts = [p.transformed(rot) for p in parts]
            v = np.vstack([p.vertices for p in parts])
            mn, mx = v.min(axis=0), v.max(axis=0)
            shift = trimesh.transformations.translation_matrix([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]])
            self._oriented[okey] = [p.transformed(shift) for p in parts]
        return self._oriented[okey]

    def _shaped_parts(self, params: Params) -> list[Part]:
        """`_oriented_parts` with the limbs resized and bent (`Params.limbs`, reshape.py), back on z = 0 (a limb bent
        down may now be the lowest point). Same triangles in the same order: paint still applies."""
        if not params.limbs:
            return self._oriented_parts(params)
        key = self._figure_key(params)
        if key not in self._shaped:
            if len(self._shaped) > 8:
                self._shaped.clear()
            parts = apply_limbs(self._oriented_parts(params), limb_ops(params.limbs), self.topology(params))
            low = float(np.vstack([q.vertices for q in parts])[:, 2].min())
            self._shaped[key] = [q.transformed(trimesh.transformations.translation_matrix([0, 0, -low]))
                                 for q in parts] if abs(low) > 1e-9 else parts
        return self._shaped[key]

    def _strengthened(self, params: Params, parts: list[Part], key: tuple) -> list[Part]:
        """`parts` with thin features thickened (strengthen.py), measured at the height asked for or else the
        largest that fits (no magnets); the conform step's flat base is solid already and is left alone. What is
        added is kept (`_thick`) for the painted parts (`figure_parts`)."""
        cavity = self.cavity(params)
        v = np.vstack([p.vertices for p in parts])
        free = max_scale(prepare_points(v), cavity)
        s = free.max_scale if free is not None else 0.0
        height = float(np.ptp(v[:, 2]))
        if params.figure_height and height > 0:
            s = min(s, params.figure_height / height) if s > 0 else params.figure_height / height
        # The flat base gains nothing (no shell), so the shells line up with `parts` (and the painted ones) by index.
        th = thicken([p if p.name != FILL_NAME else replace(p, faces=p.faces[:0]) for p in parts], s,
                     params.strengthen, params.local_strengthen())
        self._thick[key], self._strength[key] = th, th.info
        return with_shells(parts, th)

    def thickened(self, params: Params, parts: list[Part]) -> list[Part]:
        """`parts` (`base_parts` order and shapes, e.g. painted) with strengthen's additions (`unit_parts`' shape),
        paint carried to the new triangles; `parts` themselves when strengthen is off."""
        self.unit_parts(params)
        th = self._thick.get(self._key(params)) if params.strengthens else None
        return with_shells(parts, th) if th is not None else parts

    def strengthen_info(self, params: Params) -> StrengthenInfo | None:
        self.unit_parts(params)
        return self._strength.get(self._key(params))

    def colours(self, params: Params) -> list[str]:
        """Filament colours (index 0 = filament 1): the user's palette, else the file's (for a
        glTF, the palette of the chosen colour count)."""
        if params.colours:
            return list(params.colours)
        n = self.texture_count(params)
        if self.project.texture is not None and n is not None:
            return list(self.project.texture.paint(n, params.texture_details)[0])
        return list(self.project.colours)

    def texture_count(self, params: Params) -> int | None:
        """A glTF's filament count: the chosen one, else the automatic one (`auto_texture_count`); None for
        other files, or for the file's own automatic palette."""
        if self.project.texture is None:
            return None
        if params.texture_colours is not None:
            return params.texture_colours
        return self.auto_texture_count(params)

    def auto_texture_count(self, params: Params) -> int | None:
        """Automatic filament count of a glTF: the best value of the colour-change cost table at the largest
        size that fits (`toolchange.best_value_count`: the count before the biggest jump in filament changes, capped
        at AUTO_MAX_MIN minutes of colour changes),
        so "auto" trades detail for hours of print time only where a colour is expensive. None (the file's
        own palette, by colour distance) when no added colour is a jump, or no settings / fit are known.
        At the fit limit rather than the chosen size, so resizing the figure does not change its palette."""
        if params.texture_details:
            # "Keep details" adds filaments on top of the automatic count; it never changes it (counted with
            # the details, a Charmander's best value dropped from 3 colours to 2 and lost its yellow belly).
            return self.auto_texture_count(replace(params, texture_details=0))
        key = (self._key(params), round(self.ball_scale(params), 6), params.steel, params.floor_diameter,
               params.clearance, params.slot,
               params.process, self.presets.version if self.presets else 0)
        if key not in self._auto_count:
            if len(self._auto_count) > 16:
                self._auto_count.clear()
            best = None
            free = max_scale(self.points(params), self.cavity(params))
            if free is not None and free.max_scale > 0:
                best = tc.best_value_count(colour_rows(self, params, free.max_scale))
            self._auto_count[key] = best
        return self._auto_count[key]

    def textured_parts(self, params: Params) -> list[Part]:
        """`unit_parts` with a glTF's colour reduced to `texture_count` filaments (the file loads with
        its own automatic palette; geometry and face order are identical)."""
        parts = self.base_parts(params)
        n = self.texture_count(params)
        if self.project.texture is None or n is None:
            return parts
        key = (self._key(params), n, params.texture_details)
        if key not in self._textured:
            if len(self._textured) > 8:
                self._textured.clear()
            _, extra, base = self.project.texture.paint(n, params.texture_details)
            p0 = parts[0]
            # The conform step's flat base prints in the base filament, which depends on the count.
            rest = [Part(p.name, base, p.vertices, p.faces, p.tri_extra, p.subtype) if p.name == FILL_NAME else p
                    for p in parts[1:]]
            self._textured[key] = [Part(p0.name, base, p0.vertices, p0.faces, dict(extra), p0.subtype), *rest]
        return self._textured[key]

    def topology(self, params: Params) -> TopologyCache:
        obj = self.obj(params)
        if obj.id not in self._topo:
            # Plate frame (build transform applied, not rotated): brush radii are in the units the fit scales.
            self._topo[obj.id] = TopologyCache(world_parts(obj))
        # Up in the file's frame: the figure is turned by rx, ry, rz, so the file's up is that turn undone on z.
        _, rx, ry, rz = self._orientation(params)
        rot = (trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]) @
               trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]) @
               trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))[:3, :3]
        self._topo[obj.id].set_up(rot.T @ np.array([0.0, 0.0, 1.0]))
        return self._topo[obj.id]

    def old_pockets(self, params: Params) -> list[dict[str, float]]:
        """An earlier export's pockets (`FigureObject.old_pockets`) in the figure frame at scale 1: the plate frame
        turned like the figure and moved like `unit_parts`. None while the figure is turned about X or Y (its
        underside is somewhere else then)."""
        obj = self.obj(params)
        obj.parts                    # read on first use, which is when its pockets are found (before: none yet)
        if not obj.old_pockets or params.rotate_x % 360 or params.rotate_y % 360:
            return []
        okey = self._orientation(params)
        self.unit_parts(params)
        world = world_parts(obj)[0].vertices[0]
        a = np.radians(okey[3])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        shift = self._oriented[okey][0].vertices[0][:2] - rot @ world[:2]
        out = []
        for pk in obj.old_pockets:
            x, y = rot @ np.array([pk.x, pk.y]) + shift
            out.append({"x": round(float(x), 3), "y": round(float(y), 3), "diameter": round(2 * pk.radius, 3)})
        return out

    def extruder(self, params: Params) -> int:
        """The object's own filament, renumbered by `Params.colour_map` (removed colours)."""
        return map_filament(self.obj(params).extruder or 1, params.colour_map)

    def mapped_parts(self, params: Params) -> list[Part]:
        """`textured_parts` with the file's filaments renumbered by `Params.colour_map` (`remap_filaments`)."""
        parts = self.textured_parts(params)
        if not params.colour_map:
            return parts
        key = (self._key(params), self.texture_count(params), params.texture_details, tuple(params.colour_map))
        if key not in self._mapped:
            if len(self._mapped) > 8:
                self._mapped.clear()
            self._mapped[key] = remap_filaments(parts, params.colour_map, self.obj(params).extruder or 1)
        return self._mapped[key]

    def figure_parts(self, params: Params) -> list[Part]:
        """`unit_parts` with the file's colour (glTF: at the chosen count, renumbered by `colour_map`) and the
        colorize paint applied, specks under `despeckle` mm² at the print size cleared, and the pads."""
        return self._with_pads(params, self.thickened(params, self.painted_parts(params)))

    def painted_parts(self, params: Params) -> list[Part]:
        """`base_parts` (the triangles paint names) with the file's colour and the colorize paint applied."""
        parts = self.mapped_parts(params)
        if not params.paint:
            return parts
        s = self.paint_scale(params) if params.despeckle > 0 else 0.0
        key = (self._key(params), self.texture_count(params), params.texture_details, json.dumps(params.colour_map),
               json.dumps(params.paint, sort_keys=True), params.despeckle, round(s, 4))
        if key not in self._painted:
            if len(self._painted) > 8:          # one figure's recent edits; paint is cheap to redo
                self._painted.clear()
            min_area = params.despeckle / (s * s) if s > 0 else 0.0
            self._painted[key] = self._replay.apply(key[:4], parts, params.paint_ops, self.topology(params),
                                                    min_area)
        return self._painted[key]

    def _with_pads(self, params: Params, parts: list[Part]) -> list[Part]:
        """`parts` plus the pads under the magnet pockets (`pocket_pads`), in the flat base's filament."""
        pad = self.pads(params)
        if pad is None:
            return parts
        base = next((p.extruder for p in parts if p.name == FILL_NAME), parts[0].extruder if parts else 1)
        return [*parts, Part(pad.name, base, pad.vertices, pad.faces, {}, pad.subtype)]

    def pads(self, params: Params) -> Part | None:
        """Pads under the pockets on the steel (unit scale): every gap up to `flat_fill` under a pocket's walled
        disc filled to the bed, so a magnet resting on the steel is held all round (the pocket map counts
        on them: `PocketSpec.pad`). None without magnets, on the raised seat, with the fill off, or where the
        figure already sits flat over every pocket."""
        spec = self.pocket_spec(params)
        if not (params.magnets and spec.flush and spec.pad > 0):
            return None
        key = shape_key(params)
        if key not in self._pads:
            _evict(self._pads)
            self._pads[key] = None
            scale, _, pockets, _, off = solved(self, params)
            if scale > 0 and pockets:
                cavity = self.cavity(params)
                # Only where the cavity's floor is flat (it rises past it): a pad there could not collide.
                reach = cavity.floor_radius
                centre = (-off[0] / scale, -off[1] / scale)
                discs = disc_region([(p.x / scale, p.y / scale) for p in pockets],
                                    (spec.diameter / 2 + spec.wall + PAD_MARGIN) / scale)
                floor = disc_region([centre], reach / scale)
                self._pads[key] = base_fill(self.unit_parts(params),
                                            lambda gx, gy, step: discs(gx, gy, step) & floor(gx, gy, step),
                                            spec.pad / scale, FILL_CELL / scale, FILL_OVERLAP / scale, 1,
                                            connected=False)
        return self._pads[key]

    def paint_scale(self, params: Params) -> float:
        """The print scale paint sizes are judged at: the scale chosen, else the height asked for (capped at the
        largest that fits), else the largest that fits (0 when nothing fits)."""
        key = (self._key(params), params.figure_scale, params.figure_height, round(self.ball_scale(params), 6),
               params.steel, params.floor_diameter, params.clearance)
        if key not in self._paint_scale:
            if params.figure_scale:
                s = float(params.figure_scale)
            else:
                hp = self.points(params)
                free = max_scale(hp, self.cavity(params))
                s = free.max_scale if free is not None else 0.0
                if params.figure_height and hp.height > 0:
                    s = min(s, params.figure_height / hp.height) if s > 0 else params.figure_height / hp.height
            self._paint_scale[key] = s
        return self._paint_scale[key]

    def points(self, params: Params) -> HullPoints:
        self.unit_parts(params)
        obj = self.obj(params)
        return self._points[self._key(params)]

    def underside(self, params: Params, s: float) -> HeightField:
        """Height field of the figure's underside at figure scale s (ray-cast once per orientation)."""
        key = self._key(params)
        if key not in self._fields:
            solid = [p.mesh() for p in self.unit_parts(params) if p.subtype == "normal_part"]
            self._fields[key] = height_field(solid)
        return self._fields[key].scaled(s)

    def ams_spools(self, params: Params) -> list[AmsSlot | None]:
        """The AMS spool each figure filament (1..n) prints from, once the printer was read: the user's
        choice (`filament_map`) or the closest colour; None where no loaded spool is close."""
        if not self.ams:
            return []
        colours = self.colours(params)
        manual = {int(k): int(v) for k, v in (params.filament_map or {}).items()}
        ms = match_slots(list(range(1, len(colours) + 1)), colours, [a.colour for a in self.ams], manual)
        return [self.ams[m.target - 1] if not m.new and 1 <= m.target <= len(self.ams) else None for m in ms]

    def print_settings(self, params: Params, supports: bool) -> tuple[dict, list[str]] | None:
        """The reference settings cut to the figure's filaments (`adapt_settings`), and the preset each
        filament prints with. With the printer's AMS read, filament i takes the preset of its spool
        (`presets.ams_filaments`: PLA spools slice as PLA); else every filament the material of `slot`."""
        ref = self.settings(params)
        if not ref:
            return None
        colours = self.colours(params)
        spools = self.ams_spools(params)
        if not spools:
            st = adapt_settings(ref, colours, self.slot(params), supports)
            return st, list(st.get("filament_settings_id", []))
        system = self.presets.system if self.presets else None
        ref2, src, names = ams_filaments(ref, system, [(a.filament_id, a.type) if a else ("", "") for a in spools],
                                         self.slot(params))
        return adapt_settings(ref2, colours, self.slot(params), supports, src), names

    def slot_matches(self, params: Params) -> list[SlotMatch]:
        """How the figure's filaments map onto the user's project slots (for the import file): the
        printer's loaded AMS trays when they were read, else the reference project's filaments."""
        targets = [a.colour for a in self.ams] if self.ams else list((self.reference or {}).get("filament_colour", []))
        if not targets:
            return []
        obj = self.obj(params)
        ukey = (f"{obj.id}:{self.texture_count(params)}:{params.texture_details}:{params.despeckle}:"
                f"{json.dumps(params.colour_map)}:{json.dumps(params.paint, sort_keys=True)}")
        if ukey not in self._used:
            parts = self.figure_parts(params)
            self._used[ukey] = used_filaments(parts, self.extruder(params))
        manual = {int(k): int(v) for k, v in (params.filament_map or {}).items()}
        used = self._used[ukey]
        colours = self.colours(params)
        # Every palette colour gets a row (a colour added but not painted yet can be given its slot first); the used
        # ones are matched first, so new slots the import file adds are numbered as before.
        spare = [k for k in range(1, len(colours) + 1) if k not in used]
        out = match_slots(used, colours, targets, manual)
        extra = max([len(targets)] + [m.target for m in out if m.new])
        for m in match_slots(spare, colours, targets, manual):
            if m.new and not m.manual:
                extra += 1
                m = replace(m, target=extra)
            out.append(replace(m, used=False))
        return sorted(out, key=lambda m: m.source)

    def slot(self, params: Params) -> int:
        if params.slot is not None:
            return params.slot
        return default_slot(self.reference) if self.reference else 0

    def settings(self, params: Params) -> dict | None:
        """The settings exports and slices use: the reference project plus the imported presets.

        Imported filaments are appended after the project's own slots (so `slot` can pick them);
        `slot_matches` keeps using the project's real slots, which are what is loaded in the AMS.
        """
        if not self.reference or self.presets is None or not self.presets.presets:
            return self.reference
        key = (id(self.reference), self.presets.version, params.process)
        if key not in self._settings:
            self._settings[key] = self.presets.merged(self.reference, params.process)[0]
        return self._settings[key]


def scaled(parts: list[Part], s: float) -> list[Part]:
    return [p.transformed(np.diag([s, s, s, 1.0])) for p in parts]


def open_session(path: str | Path, ball_path: str | Path = DEFAULT_BALL,
                 reference_path: str | Path | None = DEFAULT_REFERENCE, work_dir: str | Path | None = None,
                 simplify: float | None = None) -> Session:
    path = Path(path)
    proj = read_project(path, simplify)
    sid = uuid.uuid4().hex[:12]
    wd = Path(work_dir) if work_dir else CACHE_DIR / "sessions" / sid
    wd.mkdir(parents=True, exist_ok=True)
    ref = reference_settings(reference_path) if reference_path and Path(reference_path).exists() else None
    return Session(sid, proj, wd, load_ball(ball_path, CACHE_DIR), ref, simplify=simplify)


def auto_simplify(session: Session, params: Params) -> Params:
    """`params` for a file just opened: with `simplify_auto` and no Simplify setting, a figure denser than
    DENSE_FACES triangles is simplified to AUTO_SIMPLIFY (MINI EEVEE, 1.45M: open and analyse 39.5 -> 11.5 s, the
    surface moving 0.02 mm at 25 mm tall). A 3MF reads its meshes on demand, so counting them first costs little.
    A pokefit export (`Project.pokefit` set) opens as it is: it was made at the density its maker chose, and
    simplifying would resample its paint (clean soft edges came out ragged at 200k of a 300k export)."""
    if not params.simplify_auto or params.simplify is not None or session.project.pokefit is not None:
        return params
    if session.obj(params).face_count <= DENSE_FACES:
        return params
    return replace(params, simplify=float(AUTO_SIMPLIFY))


OLD_POCKETS_MAX = 3            # magnets put back from an earlier export (the UI's limit)


def reopened(session: Session, params: Params) -> Params:
    """`params` for a pokefit export just opened, as the UI goes on to use them: the base settings it was made with
    (`made_with`: its mesh has that base already) and its old pockets as pockets placed by hand (`old_pockets`, at
    the size it fits at; the magnets' own diameter). Applied before the first analysis, so the UI's next one is that
    same analysis, cached: a Vaporeon export opened in 40 s in the browser engine, re-analysed with each in turn.
    The settings unchanged for any other file."""
    made = _made_with(session)
    if made:
        params = replace(params, **made)
    old = session.old_pockets(params)[:OLD_POCKETS_MAX]
    if not old or params.pockets is not None:
        return params
    scale = params.figure_scale
    if scale is None:
        free = max_scale(session.points(params), session.cavity(params))
        if free is None:
            return params
        scale = free.max_scale
    d = old[0]["diameter"]
    return replace(params, magnets=len(old),
                   pockets=[{"x": round(q["x"] * scale, 3), "y": round(q["y"] * scale, 3)} for q in old],
                   pocket_diameter=round(d, 2) if abs(d - params.pocket_diameter) > 0.05 else params.pocket_diameter)


def app_config(slicer: str | None, reference: dict | None = None, reference_name: str = "",
               presets: PresetLibrary | None = None, ams: list[AmsSlot] | None = None) -> dict[str, Any]:
    """What the UI needs before a file is loaded: the steel discs and washers, defaults, the reference printer's filament slots.

    `reference` is a project the user loaded (their own filament slots); default: DEFAULT_REFERENCE.
    `presets` are Bambu presets the user imported: their filaments are listed after the project's
    slots (flagged `imported`), their process presets in `processes`.
    """
    from .slicer import filament_slots
    ref = reference if reference is not None else (
        reference_settings(DEFAULT_REFERENCE) if DEFAULT_REFERENCE.exists() else None)
    notes: list[PresetNote] = []
    slots: list[dict[str, Any]] = list(filament_slots(ref)) if ref else []
    processes = [str(ref.get("print_settings_id", ""))] if ref else []
    if ref and presets is not None and presets.presets:
        merged, notes = presets.merged(ref)
        n_real = len(slots)
        from .presets import plate_problem
        slots = [{**s, "imported": i >= n_real, "warning": plate_problem(merged, i) if i >= n_real else ""}
                 for i, s in enumerate(filament_slots(merged))]
        processes += presets.processes(ref)
    ball = load_ball(DEFAULT_BALL)
    return {
        "steels": [w.to_dict() for w in CATALOGUE.values()],
        "default_steel": DEFAULT_STEEL,
        "ball_scales": [0.6, 1.0],
        "ball_r_inner": ball.r_inner,             # native: inner diameter = 2 · this · scale
        "params": asdict(Params()),
        "slots": slots,
        "processes": processes,
        "presets": [n.to_dict() for n in notes],
        "default_slot": default_slot(ref) if ref else 0,
        "reference": reference_name if reference is not None else DEFAULT_REFERENCE.name,
        "ams": [a.to_dict() for a in ams or []],
        "printer": (ref or {}).get("printer_settings_id", ""),
        "process": (ref or {}).get("print_settings_id", ""),
        "slicer": slicer,
    }


def input_checks(session: Session, params: Params, printing: bool = True) -> list[Check]:
    """Mesh soundness of the source parts (scale-independent, computed once per object). The self-intersection
    checks (the slow part: pymeshlab, local only) come with the printability checks: `printing=False` leaves them
    out unless they were made already."""
    obj = session.obj(params)
    if printing and obj.id not in session._selfint:
        session._selfint[obj.id] = [self_intersections(p) for p in obj.parts]
    if obj.id not in session.input_checks:
        checks: list[Check] = []
        for p in obj.parts:
            checks.extend(mesh_checks(p, intersections=False))
        if obj.old_pockets:
            n = len(obj.old_pockets)
            base = (f" Its flat base ({obj.old_bases} part{'s' if obj.old_bases > 1 else ''}) was taken off too: the "
                    "settings make it again." if obj.old_bases else "")
            checks.insert(0, Check("old_pockets", "Pockets from an earlier export", "info",
                                   f"This figure has {n} magnet pocket{'s' if n > 1 else ''} from an earlier export: "
                                   f"{'they were' if n > 1 else 'it was'} filled in and the magnets placed there again "
                                   "(Magnets card: drag to move, change the count, or Auto place)." + base, n))
        if obj.repaired:
            # What loading fixed (repair.py), as Bambu Studio reports its own repairs.
            sound = all(c.status == "pass" for c in checks if c.id.startswith("watertight:"))
            checks.insert(0, Check("repaired", "Mesh repaired", "info",
                                   f"Repaired as it loaded (like Bambu Studio's Fix model): {obj.repaired.text()}. "
                                   + ("Every edge now joins exactly two faces." if sound else
                                      "Some problems are left (below)."), obj.repaired.split))
        session.input_checks[obj.id] = checks
    return [*session.input_checks[obj.id], *session._selfint.get(obj.id, [])]


@dataclass
class Report:
    params: dict[str, Any]
    object: dict[str, Any]
    ball: dict[str, Any]
    fit: dict[str, Any]
    pockets: list[dict[str, float]]
    checks: list[dict[str, Any]]
    status: str
    steels: list[dict[str, Any]]
    timings: dict[str, float]
    underside: dict[str, Any] = field(default_factory=dict)   # 2D outlines for the pocket editor
    import_map: list[dict[str, Any]] = field(default_factory=list)   # figure filament -> project slot
    toolchange: dict[str, Any] = field(default_factory=dict)         # estimated colour changes (toolchange.py)
    # The printability checks were left out (`analyse(printing=False)`): ask again with them.
    printing_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _steel_band(steel: Steel, diameter: float, relaxed: bool) -> tuple[float, float]:
    """Radii allowed for pocket centres: magnet fully on steel, or (relaxed) centre on steel."""
    return (steel.id / 2, steel.od / 2) if relaxed else steel.annulus(diameter)


def _pocket_map(session: Session, params: Params, s: float) -> PocketMap:
    return pocket_map(session.underside(params, s), session.pocket_spec(params))


def _place(session: Session, params: Params, cavity: Cavity, steel: Steel, s: float, relaxed: bool = False
           ) -> tuple[list[Pocket], list[Check], tuple[float, float]]:
    """`params.magnets` pockets at figure scale `s` that sit on the steel while the figure fits.

    The figure frame's origin is at offset o in the ball frame, so the steel disc
    (on the ball axis) is centred at -o in the figure frame. The fit margin is
    1-Lipschitz in o, so every offset within the best margin m* of the best
    offset o* still fits; o* and 36 such offsets are tried.
    """
    spec, n = session.pocket_spec(params), params.magnets
    pm = _pocket_map(session, params, s)
    o_star, m_star = best_offset(session.points(params), cavity, s)
    o0 = (float(o_star[0]), float(o_star[1]))
    if m_star < 0:
        return [], [Check("pockets:feet", "Underside", "fail", "Figure does not fit at this size.")], o0
    lo, hi = _steel_band(steel, spec.diameter, relaxed)
    cands = [o_star] + [o_star + m_star * f * np.array([np.cos(a), np.sin(a)])
                        for f in (0.33, 0.66, 0.95) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    first: list[Check] = []
    for o in cands:
        pockets, checks = choose_pockets(pm, spec, n, (-float(o[0]), -float(o[1]), lo, hi), allow_fewer=False)
        if len(pockets) == n:
            return pockets, checks, (float(o[0]), float(o[1]))
        first = first or checks
    return [], first, o0


def _offset_for_pockets(session: Session, params: Params, cavity: Cavity, steel: Steel, s: float,
                        pockets: list[Pocket]) -> tuple[float, float]:
    """Offset that keeps the figure fitting and puts the given pockets as well on the steel as it can.

    The fit comes first: only offsets within the best fit margin m* of the best
    offset o* are considered (the margin is 1-Lipschitz in the offset).
    """
    o_star, m_star = best_offset(session.points(params), cavity, s)
    if m_star <= 0:
        return float(o_star[0]), float(o_star[1])
    g = np.linspace(-m_star, m_star, 21)
    grid = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= m_star] + o_star
    pxy = np.array([[p.x, p.y] for p in pockets])
    lo, hi = steel.annulus(params.pocket_diameter)
    r = np.linalg.norm(pxy[None, :, :] + grid[:, None, :], axis=2)
    score = np.min(np.minimum(r - lo, hi - r), axis=1)
    best = grid[int(np.argmax(score))]
    return float(best[0]), float(best[1])


def _largest_with_pockets(session: Session, params: Params, cavity: Cavity, steel: Steel, s_fit: float,
                          relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
    """Largest scale <= s_fit where the pockets fit the underside AND sit on the steel.

    Not monotone at the small end (walls are absolute mm), so scan down in 4 %
    steps from the fit limit, then bisect between the last failure and the first success.
    The scan stops at 60 % of the fit size: a steel disc that needs the figure
    smaller than that is the wrong size, and the steel check says so.
    """
    best = None
    prev_fail = None
    s = s_fit
    while s >= MIN_SCALE_FOR_MAGNETS * s_fit:      # (no absolute floor: generated GLBs fit at ~4 %)
        pk, ch, off = _place(session, params, cavity, steel, s, relaxed)
        if pk:
            best = (s, pk, ch, off)
            break
        prev_fail, s = s, s * 0.96
    if best is not None and prev_fail is not None:
        a, b = best[0], prev_fail
        for _ in range(6):
            mid = (a + b) / 2
            pk, ch, off = _place(session, params, cavity, steel, mid, relaxed)
            if pk:
                best, a = (mid, pk, ch, off), mid
            else:
                b = mid
    return best


def shape_key(params: Params) -> str:
    """`params` without the colour-only settings (`COLOUR_ONLY`): equal keys give the same fit and pockets."""
    return json.dumps({k: v for k, v in asdict(params).items() if k not in COLOUR_ONLY}, sort_keys=True)


def _evict(cache: dict) -> None:
    """Make room in a small per-session cache (oldest entry out)."""
    while len(cache) >= SHAPE_CACHE:
        cache.pop(next(iter(cache)))


def _solve(session: Session, params: Params, cavity: Cavity, steel: Steel
           ) -> tuple[float, float, list[Pocket], list[Check], tuple[float, float]]:
    """(scale, max scale, pockets in the figure frame, pocket checks, XY offset)."""
    hp = session.points(params)
    if params.figure_scale is None and params.figure_height is not None:
        # A height asked for: that size, unless it is larger than the largest that fits (with the magnets).
        auto = _solve(session, replace(params, figure_height=None), cavity, steel)
        want = params.figure_height / hp.height
        if auto[0] <= 0 or want < auto[0] * (1 - 1e-4):
            return _solve(session, replace(params, figure_height=None, figure_scale=want), cavity, steel)
        note = Check("size:height", "Size", "info",
                     f"{params.figure_height:g} mm was asked for; the largest that fits"
                     f"{' with the magnets on the steel' if params.magnets else ''} is {auto[0] * hp.height:.1f} mm, "
                     "so it prints at that.")
        return auto[0], auto[1], auto[2], [*auto[3], note], auto[4]
    spec, n = session.pocket_spec(params), params.magnets
    free = max_scale(hp, cavity)
    if free is None:
        return 0.0, 0.0, [], [], (0.0, 0.0)
    s_fit = free.max_scale

    def fit_offset(s: float) -> tuple[float, float]:
        o, _ = best_offset(hp, cavity, s)
        return float(o[0]), float(o[1])

    if not n:
        scale = params.figure_scale or s_fit
        return scale, s_fit, [], [], fit_offset(scale)

    if params.pockets is not None:
        # Manual pockets: figure-frame XY as placed by the user; seats follow the underside.
        scale = params.figure_scale or s_fit
        manual = seat_pockets(session.underside(params, scale),
                              [Pocket(float(p["x"]), float(p["y"])) for p in params.pockets], spec)
        if not manual:
            return scale, s_fit, [], [], fit_offset(scale)
        return (scale, s_fit, manual, [Check("pockets:feet", "Underside", "info", "Pockets placed by hand.")],
                _offset_for_pockets(session, params, cavity, steel, scale, manual))

    # Shrinking the figure only shrinks the area that can hold a walled pocket
    # (walls are absolute mm). If the pockets do not fit even without the steel
    # ring at the size being asked for (or the largest that fits), skip the search
    # and show as many as do fit.
    probe = params.figure_scale or s_fit
    free_pk, free_ch = choose_pockets(_pocket_map(session, params, probe), spec, n)
    if not free_pk:
        return probe, s_fit, free_pk, free_ch + _more_room(session, params), fit_offset(probe)
    if len(free_pk) < n:
        # Fewer fit than asked: place those like a full set, on the steel (a lone pocket placed freely sat on a
        # Pikachu's centre, over a washer's hole; on a solid disc the steel is a whole disc), keeping the "only room
        # for k" warning.
        scale, top, pk, ch, off = _solve(session, replace(params, magnets=len(free_pk)), cavity, steel)
        return scale, top, pk, free_ch + _more_room(session, params) + [c for c in ch if c.id != "pockets:feet"], off

    def largest(relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
        key = (session._key(params), session.ball_scale(params), steel.key, round(cavity.seat_diameter, 3),
               params.clearance, spec, n, relaxed)
        if key not in session._auto:
            session._auto[key] = _largest_with_pockets(session, params, cavity, steel, s_fit, relaxed)
        return session._auto[key]

    if params.figure_scale is not None:
        sc = params.figure_scale
        for relaxed in (False, True):
            pk, ch, off = _place(session, params, cavity, steel, sc, relaxed)
            if pk:
                best = largest(relaxed)
                return sc, best[0] if best else s_fit, pk, ch, off
        # Nothing on steel: show the best unconstrained spots so the user sees why.
        return sc, s_fit, free_pk, free_ch, fit_offset(sc)

    for relaxed in (False, True):
        best = largest(relaxed)
        if best is not None:
            scale, pk, ch, off = best
            if scale < s_fit * 0.999:
                ch = ch + [Check("pockets:scale", "Size vs. magnets", "info",
                                 f"Reduced from {s_fit:.1%} to {scale:.1%} so the magnets sit on the steel; "
                                 f"a larger disc may allow more (Compare steel sizes).")]
            return scale, scale, pk, ch, off
    return s_fit, s_fit, free_pk, free_ch, fit_offset(s_fit)


def _more_room(session: Session, params: Params) -> list[Check]:
    """When fewer pockets fit than asked: the settings that can make room under the figure (none on the raised seat,
    which uses the figure's own underside, or when the fill is already the deepest offered)."""
    cspec = session.conform_spec(params)
    if (params.pocket_seat != "steel" or cspec is None or cspec.fill >= AUTO_FILLS[-1]
            or not (cspec.flat_diameter > 0 or cspec.fill_all)):          # no flat base: the fill does nothing
        return []
    return [Check("pockets:room", "More magnets", "info",
                  f"More may fit with a deeper flat base: \"Fill gaps up to\" {AUTO_FILLS[-1]:g} mm (now "
                  f"{cspec.fill:g}){'' if cspec.fill_all else ', or Fill where: everywhere under the figure'}.",
                  AUTO_FILLS[-1])]


def solved(session: Session, params: Params) -> tuple[float, float, list[Pocket], list[Check], tuple[float, float]]:
    """`_solve` for these settings, cached per shape (`shape_key`): a colour edit reuses it."""
    skey = shape_key(params)
    if skey not in session._solved:
        _evict(session._solved)
        cavity = session.cavity(params)
        session._solved[skey] = _solve(session, params, cavity, cavity.steel)
    return session._solved[skey]


def steel_table(session: Session, params: Params) -> list[dict[str, Any]]:
    """Max figure scale with each catalogue steel disc and washer (magnets full / partial / off steel), best first.

    They are evaluated in parallel threads: the work is numpy/shapely/
    manifold3d, which release the GIL for most of it. The session's caches are
    warmed first so the threads only read them (each writes its own _auto key).
    """
    session.use_simplify(params.simplify)
    from concurrent.futures import ThreadPoolExecutor

    session.unit_parts(params)
    height = session.points(params).height
    floor = session.floor(params)

    def row(key: str, w: Steel) -> dict[str, Any]:
        # The same figure (its shape as conformed for the current steel and floor) on the same printed floor (at
        # least this steel's seat): only the steel differs, so the rows compare the steel, not reshaped figures.
        cav = Cavity(session.ball, session.ball_scale(params), w, params.clearance, floor)
        p = replace(params, figure_scale=None, figure_height=None, pockets=None)
        _, s_max, pockets, _, off = _solve(session, p, cav, w)
        on = "off"
        if pockets and len(pockets) == p.magnets:
            pxy = np.array([[q.x, q.y] for q in pockets])
            if annulus_margin(pxy, 1.0, np.array(off), w.annulus(p.pocket_diameter)) >= -1e-6:
                on = "full"
            elif annulus_margin(pxy, 1.0, np.array(off), (w.id / 2, w.od / 2)) >= -1e-6:
                on = "partial"
        return {"key": key, "label": w.label, "max_scale": round(s_max, 4), "height_mm": round(s_max * height, 2),
                "pockets_on_steel": on == "full", "steel": on, "floor_z": round(cav.floor_z, 2), "solid": w.solid,
                "floor_diameter": round(cav.seat_diameter, 1)}

    if sys.platform == "emscripten":      # the browser build has no threads
        rows = [row(*kv) for kv in CATALOGUE.items()]
    else:
        with ThreadPoolExecutor(max_workers=len(CATALOGUE)) as pool:
            rows = list(pool.map(lambda kv: row(*kv), CATALOGUE.items()))
    rank = {"full": 0, "partial": 1, "off": 2}
    rows.sort(key=lambda r: (rank[r["steel"]], -float(r["max_scale"])))
    return rows


def analyse(session: Session, params: Params, recommend: bool = False,
            printing: bool = True) -> tuple[Report, list[Part]]:
    """Fit, pockets, checks. Returns the report and the final (cut) figure-frame parts.

    `printing=False`: leave the printability checks (thickness, overhangs, islands, bed contact: about a quarter
    of a first analysis) out unless they are cached for this geometry, marking the report `printing_pending`
    (its status then counts only the checks made), so the UI shows the fit and the pockets first and asks again."""
    session.use_simplify(params.simplify)
    t: dict[str, float] = {}
    t0 = time.time()
    cavity = session.cavity(params)
    steel = cavity.steel
    hp = session.points(params)
    unit = session.figure_parts(params)
    obj = session.obj(params)
    checks: list[Check] = list(input_checks(session, params, printing))
    if (dense := _dense_hint(obj, params)) is not None:
        checks.append(dense)
    t["input_checks"] = time.time() - t0

    t1 = time.time()
    skey = shape_key(params)
    scale, s_max, pockets, pchecks, off = solved(session, params)
    t["fit"] = time.time() - t1
    fit_dict: dict[str, Any] = {"max_scale": s_max, "scale": scale, "offset": off,
                                "height_mm": scale * hp.height, "max_height_mm": s_max * hp.height,
                                "unit_height": hp.height, "unit_extents": _extents(unit)}
    if scale <= 0:
        checks.append(Check("fit", "Fits in the ball", "fail",
                            "Does not fit even when very small — check the orientation (z up) and the file's units."))
        rep = Report(asdict(params), _obj_info(session, obj, params), _ball_info(cavity, steel), fit_dict, [],
                     [c.to_dict() for c in checks], "fail", [], t)
        return rep, unit

    final = scaled(unit, scale)
    units = _units_hint(hp.height)
    if units:
        checks.append(units)
    fc = check_fit(hp, cavity, scale, off)
    limit = [fc.point[0] - off[0], fc.point[1] - off[1], fc.point[2] - cavity.floor_z]
    fit_dict.update({"margin": fc.margin, "limit_kind": fc.kind, "limit_point": limit})
    if fc.margin >= -1e-3:
        checks.append(Check("fit", "Fits in the ball", "pass",
                            f"{scale * hp.height:.1f} mm tall ({scale:.1%} of the file); largest that fits is "
                            f"{s_max * hp.height:.1f} mm ({s_max:.1%}), limited by the {_kind_text(fc.kind)}.",
                            round(fc.margin, 2), [limit]))
    else:
        checks.append(Check("fit", "Fits in the ball", "fail",
                            f"Pokes {-fc.margin:.1f} mm into the {_kind_text(fc.kind)} at this size; "
                            f"max is {s_max:.1%} ({s_max * hp.height:.1f} mm).", round(fc.margin, 2), [limit]))

    info = session.conform_info(params)
    if info is not None:
        fit_dict["conform"] = info.to_dict()
        spec = session.conform_spec(params)
        fit_dict["conform"]["fill_mm"] = spec.fill if spec else 0.0
        raised = spec.fill if spec and spec.fill > params.flat_fill and info.fill_mm2 > 0 else None
        conform_check = _conform_check(info, raised)
        if conform_check:
            checks.append(conform_check)
    sinfo = session.strengthen_info(params)
    if sinfo is not None:
        fit_dict["strengthen"] = sinfo.to_dict()
        tall = f"{sinfo.scale * hp.height:.1f} mm tall"
        each = (f" {sinfo.features} feature{'s' if sinfo.features != 1 else ''} thickened to {'their' if sinfo.features != 1 else 'its'} "
                f"own minimum." if sinfo.features else "")
        if sinfo.target_mm <= 0:
            text = f"At {tall}:{each} (up to {sinfo.max_push_mm:.2f} mm out)"
        elif sinfo.thin_share > 0:
            text = (f"{sinfo.thin_share:.1%} of the surface was thinner than {sinfo.target_mm:g} mm at {tall}: thickened "
                    f"at the joins, tapering to the tips (up to {sinfo.max_push_mm:.2f} mm out).{each}")
        else:
            text = f"Nothing was thinner than {sinfo.target_mm:g} mm at {tall}.{each}"
        checks.append(Check("strengthen", "Thin parts strengthened", "info", text, round(sinfo.thin_share, 4)))

    t2 = time.time()
    uncut = False                         # pockets left uncut by a quick analysis (the full one cuts them)
    pspec = session.pocket_spec(params)
    meshes = [p.mesh() for p in final]
    if skey not in session._undersides:
        _evict(session._undersides)
        fill = [m for p, m in zip(final, meshes) if p.name == FILL_NAME]
        session._undersides[skey] = _underside(session, params, meshes, scale, steel, off, fill)
    underside = session._undersides[skey]
    if params.magnets:
        checks.extend(pchecks)
        if pockets:
            field_s = session.underside(params, scale)
            pocket_checks = validate_pockets(field_s, pockets, pspec)  # pocket:N (+ pocket:gap)
            checks.extend(pocket_checks)
            lo, hi = steel.annulus(params.pocket_diameter)
            pxy = np.array([[p.x, p.y] for p in pockets])
            am = annulus_margin(pxy, 1.0, np.array(off), (lo, hi))
            am_centre = annulus_margin(pxy, 1.0, np.array(off), (steel.id / 2, steel.od / 2))
            if am >= -1e-6:
                where = (f"steel disc (magnet centres up to {hi:.1f} mm from its centre)" if steel.solid else
                         f"washer's steel ring ({lo:.1f}–{hi:.1f} mm from the centre)")
                checks.append(Check("steel", "Magnets on the steel", "pass",
                                    f"{'The magnet lies' if len(pockets) == 1 else 'All magnets lie'} fully on the "
                                    f"{where}."))
            elif am_centre >= -1e-6:
                checks.append(Check("steel", "Magnets on the steel", "warn",
                                    f"{'The pocket centre is' if len(pockets) == 1 else 'Pocket centres are'} on "
                                    "the steel but the magnets overhang its edge "
                                    f"by up to {-am:.1f} mm, so they hold with roughly half force. "
                                    "A larger disc avoids this (Compare steel sizes).", round(am, 2)))
            else:
                od, idd = required_steel(pockets, pspec)
                need = (f"a disc of Ø ≥ {od:.1f} mm" if steel.solid else
                        f"a washer of OD ≥ {od:.1f} mm and hole ≤ {idd:.1f} mm, or a solid disc of Ø ≥ {od:.1f} mm")
                checks.append(Check("steel", "Magnets on the steel", "fail",
                                    f"Pockets miss the steel by {-am:.1f} mm with this {'disc' if steel.solid else 'washer'}"
                                    f"; needs {need}.", round(am, 2)))
            # Per-pocket checks come first, in pocket order; a failed one is not cut.
            cuttable = [p for p, c in zip(pockets, pocket_checks) if c.status != "fail"]
            # The quick analysis shows the fit and where the pockets go (the preview draws them on the uncut figure):
            # cutting them (~2 s on 300k triangles) waits for the full one unless that cut was made already.
            if printing or pockets_cut_cached(final, cuttable, pspec):
                final, cut_checks = cut_pockets(final, cuttable, pspec)
                checks.extend(cut_checks)
            else:
                uncut = True
    t["pockets"] = time.time() - t2

    t3 = time.time()
    st = session.settings(params) or {}
    layer = float(st.get("layer_height", 0.2) or 0.2)
    angle = float(st.get("support_threshold_angle", 30) or 30)
    ignore = [(p.x, p.y, params.pocket_diameter / 2, p.z + params.pocket_depth) for p in pockets] if params.magnets else []
    # Geometry only: a colour edit re-analyses the same shape, so the checks are reused.
    pkey = (parts_digest(final), layer, angle, repr(ignore))
    pending = not printing and (uncut or pkey not in session._printing)
    if pending:
        checks.append(Check("printing", "Print checks", "info",
                            ("Cutting the magnet pockets, checking" if uncut else "Checking")
                            + " wall thickness, overhangs, layers that start in mid-air and bed contact…"))
    elif pkey not in session._printing:
        solid = printable_solid(final)
        found = [thickness_check(solid), overhang_check(solid, angle, ignore),
                 islands_check(solid, 2 * layer), bed_contact_check(solid)]
        found.append(supports_verdict(found))
        _evict(session._printing)
        session._printing[pkey] = found
    if not pending:
        checks.extend(session._printing[pkey])
    t["printability"] = time.time() - t3

    t4 = time.time()
    toolchange = toolchange_info(session, params, final, scale)
    t["toolchange"] = time.time() - t4

    steels = steel_table(session, params) if recommend else []
    t["total"] = time.time() - t0
    rep = Report(asdict(params), _obj_info(session, obj, params), _ball_info(cavity, steel), fit_dict,
                 [p.to_dict() for p in pockets], [c.to_dict() for c in checks], worst(checks), steels,
                 {k: round(v, 2) for k, v in t.items()}, underside,
                 [m.to_dict() for m in session.slot_matches(params)], toolchange, pending)
    return rep, final


def _tc_estimate(parts: list[Part], object_extruder: int, colours: list[str], settings: dict) -> dict[str, Any]:
    """Colour changes for these (final-mm) parts: the best nozzle split (what Bambu's Auto For Flush
    finds, within the model's error) and, on a two-nozzle printer, everything on one nozzle."""
    n = max(1, len(colours))
    lh = float(settings.get("layer_height", 0.2) or 0.2)
    first = float(settings.get("initial_layer_print_height", lh) or lh)
    use = tc.layer_use(parts, object_extruder, n, lh, first)
    nozzles = tc.nozzle_count(settings)
    best, single = tc.plan(use, tc.flush_matrix(settings, n), n_nozzles=nozzles)
    return {"filaments": int(use.any(axis=0).sum()), "nozzles": nozzles, "changes": best.changes,
            "switches": best.switches, "purge_g": round(tc.grams(best), 1),
            "minutes": round(best.seconds / 60), "nozzle_of": best.nozzle_of,
            "one_nozzle": {"changes": single.changes, "purge_g": round(tc.grams(single), 1),
                           "minutes": round(single.seconds / 60)}}


def toolchange_info(session: Session, params: Params, final: list[Part], scale: float) -> dict[str, Any]:
    """Estimated colour changes of the figure as exported (module toolchange.py); for a glTF, also per
    palette size, so the Filaments slider can show what each extra colour costs."""
    st = export_settings(session, params) or {}
    colours = session.colours(params)
    if len(colours) < 2 or scale <= 0:
        return {}
    obj = session.obj(params)
    info = _tc_estimate(final, session.extruder(params), colours, st)
    rows = colour_rows(session, params, scale)
    if rows:
        info["by_count"] = rows
        info["best_count"] = tc.best_value_count(rows)
    return info


def colour_rows(session: Session, params: Params, scale: float) -> list[dict[str, Any]]:
    """A glTF's estimated colour-change cost per palette size at figure scale `scale`: rows {count, changes,
    purge_g, minutes} (empty for other files or without printer settings). From the texture's colour spans
    (TextureColours.colour_spans): no paint is built, so all sizes cost about one paint (30 s -> 3 s on
    200k faces). The two most recent scales are kept (the fit limit for "auto", the chosen size)."""
    tex = session.project.texture
    ref = session.settings(params)
    if tex is None or not ref or scale <= 0:
        return []
    key = (session._key(params), round(scale, 5), params.slot, params.process,
           session.presets.version if session.presets else 0, params.texture_details)
    if key not in session._tc_table:
        rows = []
        unit = session.base_parts(params)
        p0 = unit[0]
        corner_z = p0.vertices[p0.faces][:, :, 2] * scale
        lh = float(ref.get("layer_height", 0.2) or 0.2)
        first = float(ref.get("initial_layer_print_height", lh) or lh)
        for k in range(1, min(MAX_PICK, max(tex.auto_count + 2, 6)) + 1):
            cols, spans, base = tex.colour_spans(k, corner_z, params.texture_details)
            use = tc.layer_use_arrays(spans[:, 2], spans[:, 3], spans[:, 1].astype(np.int64), len(cols), lh, first)
            rest = [Part(p.name, base, p.vertices, p.faces, {}, p.subtype) for p in unit[1:]]
            if rest:                         # the flat base prints in the base filament
                extra_use = tc.layer_use(scaled(rest, scale), base, len(cols), lh, first)
                rows_n = max(len(use), len(extra_use))
                use = np.pad(use, ((0, rows_n - len(use)), (0, 0))) | np.pad(extra_use, ((0, rows_n - len(extra_use)), (0, 0)))
            st_k = adapt_settings(ref, list(cols), session.slot(params), False)
            best, _ = tc.plan(use, tc.flush_matrix(st_k, len(cols)), n_nozzles=tc.nozzle_count(st_k))
            rows.append({"count": k, "changes": best.changes, "purge_g": round(tc.grams(best), 1),
                         "minutes": round(best.seconds / 60)})
        if len(session._tc_table) > 1:
            session._tc_table.pop(next(iter(session._tc_table)))
        session._tc_table[key] = rows
    return session._tc_table[key]



def _conform_check(info: ConformInfo, raised_fill: float | None = None) -> Check | None:
    """One line on what the conform step did (bend, flat base), or None if it changed nothing. `raised_fill`:
    the default fill was raised to this (`Session._auto_fill`)."""
    said = []
    if info.bend > 0:
        gain = info.scale_after / info.scale_before - 1 if info.scale_before else 0.0
        said.append(f"legs/tail beyond the steel bent up by up to {info.lift_mm:.1f} mm"
                    f"{' (automatic)' if info.auto else ''}, so it fits {gain:+.1%} larger")
    if info.flat_depth_mm > 0:
        said.append(f"base pressed {info.flat_depth_mm:.1f} mm flat over the steel")
    if info.fill_mm2 > 0:
        said.append(f"a {info.fill_mm2:.0f} mm² flat base fills the gap under it"
                    + (f" (gaps up to {raised_fill:g} mm: with less, no magnet pocket fits)" if raised_fill else ""))
    if not said:
        return None
    text = "; ".join(said)
    return Check("conform", "Shaped for the ball", "info", text[0].upper() + text[1:] + ".",
                 round(info.lift_mm, 2))


def _rings(geom: Any) -> list[list[list[float]]]:
    """Polygon / MultiPolygon -> list of closed rings (exteriors and holes), simplified to 0.05 mm."""
    if geom is None or geom.is_empty:
        return []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out: list[list[list[float]]] = []
    for poly in polys:
        if poly.geom_type != "Polygon":
            continue
        poly = poly.simplify(0.05)
        out.append(np.round(np.asarray(poly.exterior.coords), 3).tolist())
        out.extend(np.round(np.asarray(i.coords), 3).tolist() for i in poly.interiors)
    return out


def _underside(session: Session, params: Params, meshes: list[trimesh.Trimesh], scale: float, steel: Steel,
               off: tuple[float, float], fill: list[trimesh.Trimesh] | None = None) -> dict[str, Any]:
    """Figure-frame 2D data for the pocket editor.

    feet: outline of what touches the bed; fill: outline of the flat base the conform step added (the gap fill:
    what "Fill gaps" changed); valid: outline of where a pocket centre satisfies every rule; height: the underside
    as an image (darker = lower, transparent = no material) over `extent` [x0, y0, x1, y1]; ring: where the
    magnets lie fully on the steel (a disc, or a washer's ring) as the ball holds it.
    """
    lo, hi = steel.annulus(params.pocket_diameter)
    feet = find_feet(meshes)
    base = find_feet(fill) if fill else []
    f = session.underside(params, scale)
    out: dict[str, Any] = {"feet": _rings(unary_union(feet) if feet else None),
                           "fill": _rings(unary_union(base) if base else None), "valid": [],
                           "ring": {"x": -off[0], "y": -off[1], "inner": lo, "outer": hi,
                                    "steel_od": steel.od, "steel_id": steel.id},
                           "pocket_d": params.pocket_diameter}
    out.update(_height_image(f))
    if params.magnets:
        out["valid"] = _mask_rings(pocket_map(f, session.pocket_spec(params)).valid, f)
    return out


def _mask_rings(mask: np.ndarray, f: HeightField) -> list[list[list[float]]]:
    """Outlines of a boolean grid mask as figure-frame rings (even-odd: holes come out as rings too)."""
    import contourpy

    if not mask.any():
        return []
    padded = np.pad(mask.astype(float), 1)
    rings = []
    # contourpy: small, numpy-only, in Pyodide (scikit-image would pull in matplotlib there).
    for line in contourpy.contour_generator(z=padded, line_type="Separate").lines(0.5):
        c = np.asarray(line, np.float64)                  # (n, 2): column, row
        if len(c) < 4:
            continue
        xy = np.c_[f.x0 + (c[:, 0] - 1) * f.step, f.y0 + (c[:, 1] - 1) * f.step]
        poly = Polygon(xy).simplify(f.step / 2)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        rings.append(np.round(np.asarray(poly.exterior.coords), 3).tolist())
    return rings


def _height_image(f: HeightField) -> dict[str, Any]:
    """The underside heights as a PNG data URL: black at the bed, lighter higher up, transparent = none."""
    import base64
    import io

    from PIL import Image

    z = f.z1
    have = np.isfinite(z)
    top = float(z[have].max()) if have.any() else 1.0
    span = max(min(top, 6.0), 0.5)            # shade the lowest 6 mm; anything higher is the lightest
    t = np.clip(np.where(have, z, 0) / span, 0, 1)
    rgba = np.zeros(z.shape + (4,), np.uint8)
    rgba[..., 0] = (40 + 150 * t).astype(np.uint8)
    rgba[..., 1] = (50 + 150 * t).astype(np.uint8)
    rgba[..., 2] = (70 + 160 * t).astype(np.uint8)
    rgba[..., 3] = np.where(have, 255, 0)
    buf = io.BytesIO()
    # Row 0 = y0: the editor draws it inside its own figure-frame group (which mirrors it).
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    ny, nx = z.shape
    return {"height_png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "extent": [f.x0 - f.step / 2, f.y0 - f.step / 2, f.x0 + (nx - 0.5) * f.step, f.y0 + (ny - 0.5) * f.step],
            "height_span": round(span, 2)}


def _dense_hint(obj: FigureObject, params: Params) -> Check | None:
    """A mesh far denser than a print shows, not simplified: say what Simplify would save."""
    if params.simplify or obj.face_count <= DENSE_FACES:
        return None
    low = round(obj.face_count * SIMPLIFY_LOW)
    return Check("dense", "Dense mesh", "info",
                 f"{obj.face_count:,} triangles, far more than a print this size shows. Simplify (Model card) at "
                 f"Low (about {low:,} triangles) makes painting, the analysis and Bambu Studio several times faster; "
                 "the surface moves by at most 0.1 % of the figure's size (about 0.03 mm).", obj.face_count)


def _units_hint(height: float) -> Check | None:
    """STL/OBJ carry no units: a figure a few mm or several metres tall as saved was probably not modelled in mm.

    The fitted size in mm is right either way; only the "% of the file" figures look odd.
    """
    if 2.0 <= height <= 2000.0:
        return None
    guess = "metres or centimetres" if height < 2.0 else "micrometres"
    return Check("units", "File units", "info",
                 f"The figure is {height:.3g} mm tall as saved, so the file was probably modelled in {guess}. "
                 "The fitted size in mm is still right; only the percentages of the file look unusual.",
                 round(height, 4))


def _kind_text(kind: str) -> str:
    return {"ring": "connector ring", "shell": "shell wall", "top": "top of the ball",
            "aperture": "ring opening (it must pass through when closing)"}.get(kind, kind)


def _extents(parts: list[Part]) -> list[float]:
    v = np.vstack([p.vertices for p in parts])
    return np.ptp(v, axis=0).round(3).tolist()


def _made_with(session: Session) -> dict[str, Any] | None:
    """The recorded base settings of a reopened export (only known keys, values checked by `Params.from_dict`)."""
    made = session.project.pokefit
    saved = made.get("params") if made else None
    if not isinstance(saved, dict):
        return None
    keep = {k: saved[k] for k in MADE_WITH_KEYS if k in saved}
    try:
        Params.from_dict(keep)
    except (TypeError, ValueError):
        return None
    return keep or None


def _obj_info(session: Session, obj: FigureObject, params: Params) -> dict[str, Any]:
    return {"id": obj.id, "name": obj.name, "faces": obj.face_count,
            # Triangles in the file itself: the UI's Simplify levels are shares of this.
            "file_faces": obj.simplified.faces_before if obj.simplified else obj.face_count,
            # Triangles before and after the import's simplify step, and how far the surface moved (file mm).
            "simplified": obj.simplified.to_dict() if obj.simplified else None,
            # What the import's repair fixed (repair.RepairInfo), None when the mesh was sound.
            "repaired": obj.repaired.to_dict() if obj.repaired else None,
            # Pockets of an earlier export, filled as it loaded: centres in the figure frame at scale 1 (x, y; the UI
            # places the magnets there, times the scale). Empty when the figure is turned off its base.
            "old_pockets": session.old_pockets(params),
            # The base settings a pokefit export was made with (`made_with`): the UI opens it with them. None otherwise.
            "made_with": _made_with(session),
            "parts": [{"name": p.name, "extruder": p.extruder or session.extruder(params), "faces": len(p.faces),
                       "painted": len(p.tri_extra)} for p in session.figure_parts(params)],
            "colours": session.colours(params), "file_colours": session.project.colours,
            # glTF colour: the automatic filament count and the count in use (None for other files).
            # "auto": the count used when none is chosen (best value, else the file's own palette).
            "texture_colours": None if session.project.texture is None else {
                "auto": _auto_count(session, params),
                "count": params.texture_colours or _auto_count(session, params), "max": MAX_PICK,
                "details": params.texture_details, "max_details": MAX_DETAILS},
            "objects": [{"id": o.id, "name": o.name, "faces": o.face_count} for o in session.project.objects]}


def _auto_count(session: Session, params: Params) -> int:
    assert session.project.texture is not None
    return session.auto_texture_count(params) or session.project.texture.auto_count


def _ball_info(cavity: Cavity, steel: Steel) -> dict[str, Any]:
    z, r = cavity.profile(160)
    s = cavity.scale
    return {"scale": s, "r_inner": cavity.r_inner, "r_outer": cavity.ball.r_outer * s,
            "inner_diameter": 2 * cavity.r_inner, "native_r_inner": cavity.ball.r_inner,
            "floor_z": cavity.floor_z, "seat_z": cavity.seat_z, "top_z": cavity.top_z,
            "ring_span": cavity.ball.ring_span * s, "ring_h": [h * s for h in cavity.ball.ring_h],
            "ring_r": [x * s for x in cavity.ball.ring_r], "aperture": cavity.ball.aperture * s,
            "profile_z": z.tolist(), "profile_r": r.tolist(), "steel": steel.to_dict(),
            "floor_diameter": cavity.seat_diameter, "floor_radius": cavity.floor_radius,
            "annulus": list(steel.annulus(4.18)), "clearance": cavity.clearance}


# ------------------------------------------------------------------ outputs


def face_colours(part: Part, colours: list[str], default_extruder: int) -> np.ndarray:
    """Extruder index (1-based) per face: paint wins, else the part's, else the object's."""
    base = part.extruder or default_extruder
    ext = np.full(len(part.faces), base, np.int64)
    if part.tri_extra:
        faces, ids, uniq = distinct_attrs(part.tri_extra)
        maj = np.asarray([fp.majority if (fp := attr_paint(a)) is not None else 0 for a in uniq], np.int64)
        own = maj[ids] > 0
        ext[faces[own]] = maj[ids][own]
    return ext


def map_filament(n: int, colour_map: list[int] | None) -> int:
    """Filament `n` of the file as it prints after colours were removed (`Params.colour_map`)."""
    return colour_map[n - 1] if colour_map and 1 <= n <= len(colour_map) else n


def remap_filaments(parts: list[Part], colour_map: list[int], default_extruder: int) -> list[Part]:
    """Copies of `parts` with every filament number n (paint states, split paint exactly; the part's filament,
    made explicit) replaced by `colour_map[n - 1]`."""
    mapping = {n: m for n, m in enumerate(colour_map, start=1) if m != n}
    out = []
    for p in parts:
        ext = map_filament(p.extruder or default_extruder, colour_map)
        extra: dict[int, str] = {}
        done: dict[str, str] = {}
        for i, attrs in p.tri_extra.items():
            if attrs not in done:
                fp = attr_paint(attrs)
                done[attrs] = attrs if fp is None or not mapping else attrs.replace(
                    f'paint_color="{fp.code}"', f'paint_color="{remap_states(fp.code, mapping)}"')
            extra[i] = done[attrs]
        out.append(Part(p.name, ext, p.vertices, p.faces, extra, p.subtype))
    return out


def painted_mesh(part: Part, default_extruder: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, extruder per face, source triangle per face) with split paint drawn as its sub-triangles,
    so the preview shows brush strokes and texture-derived colour edges as the slicer will print them. The drawn
    triangles are not the part's (unsplit ones first, then each split one's sub-triangles): `source` maps them back."""
    ext = face_colours(part, [], default_extruder)
    split: list[int] = []
    if part.tri_extra:
        faces, ids, uniq = distinct_attrs(part.tri_extra)
        is_split = np.asarray([(fp := attr_paint(a)) is not None and fp.whole is None for a in uniq], bool)
        split = faces[is_split[ids]].tolist()
    if not split:
        return part.vertices, part.faces, ext, np.arange(len(part.faces), dtype=np.int64)
    base = part.extruder or default_extruder
    keep = np.ones(len(part.faces), bool)
    keep[split] = False
    tris, states = [], []
    if len(_LEAVES) > MAX_LEAVES:
        _LEAVES.clear()
    for i in split:
        code = part.tri_extra[i].split('paint_color="', 1)[1].split('"', 1)[0]
        corners = part.vertices[part.faces[i]]
        key = (code, corners.tobytes())
        if key not in _LEAVES:
            lv = leaves(parse_tree(code), list(corners))
            _LEAVES[key] = (np.asarray([t for t, _ in lv], np.float64).reshape(-1, 3, 3),
                            np.asarray([st for _, st in lv], np.int64))
        t, st = _LEAVES[key]
        tris.append(t)
        states.append(st)
    sub = np.concatenate(tris).reshape(-1, 3)
    n = len(part.vertices)
    sub_faces = np.arange(n, n + len(sub), dtype=part.faces.dtype).reshape(-1, 3)
    sub_ext = np.concatenate(states)
    source = np.concatenate([np.flatnonzero(keep), np.repeat(np.asarray(split, np.int64), [len(s) for s in states])])
    return (np.vstack([part.vertices, sub]), np.vstack([part.faces[keep], sub_faces]),
            np.concatenate([ext[keep], np.where(sub_ext == 0, base, sub_ext)]), source)


# Sub-triangles of split paint (`painted_mesh`) by (paint code, the triangle's corners): each paint step redraws the
# preview, and decoding every split triangle of every earlier stroke again made each step slower than the last.
_LEAVES: dict[tuple[str, bytes], tuple[np.ndarray, np.ndarray]] = {}
MAX_LEAVES = 200_000


# Overlap clipping by part geometry (`parts_digest`): a colour edit clips the same shapes again.
_CLIPS: dict[bytes, list[tuple[np.ndarray, np.ndarray] | None]] = {}


def _clipped_geometry(parts: list[Part]) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Per part, its (vertices, faces) after `clip_overlaps`, or None when clipping leaves it as it is."""
    key = parts_digest(parts)
    if key in _CLIPS:
        return _CLIPS[key]
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    done: list[tuple[Part, Any]] = []
    for part in parts:
        res = None
        lo, hi = part.vertices.min(axis=0), part.vertices.max(axis=0)
        if part.subtype == "normal_part":
            try:
                base = to_manifold(merged(part))
                man = base
                for prev, pman in done:
                    plo, phi = prev.vertices.min(axis=0), prev.vertices.max(axis=0)
                    if np.all(phi >= lo) and np.all(plo <= hi):
                        man = man - pman
                if abs(man.volume() - base.volume()) > 1e-6:
                    m = from_manifold(man)
                    res = repaired(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
                done.append((part, base))
            except RuntimeError:
                pass
        out.append(res)
    _evict(_CLIPS)
    _CLIPS[key] = out
    return out


def clip_overlaps(parts: list[Part]) -> list[Part]:
    """What the slicer prints for overlapping parts: each part minus all earlier ones.

    Bambu/Prusa clip multi-part objects in list order — the 2nd part is clipped
    by the 1st, the 3rd by the 1st and 2nd — so an eye listed before the body
    keeps its colour even though the body mesh encloses it. Parts that are not
    valid solids are left as they are.
    """
    return [part if g is None else remesh_part(part, trimesh.Trimesh(g[0], g[1], process=False))
            for part, g in zip(parts, _clipped_geometry(parts))]


REGION_PALETTE = 64          # "show regions" colours, reused cyclically (a figure can have thousands of regions)
REGION_OTHER = "#5a6270"     # parts taps cannot paint (the conform step's flat base)


@dataclass
class PreviewMeshes:
    """What the 3D preview draws: per part (vertices, faces) and a colour key per face, plus a geometry hash."""
    meshes: list[tuple[np.ndarray, np.ndarray]]
    keys: list[np.ndarray]           # per face: 1-based filament, or (regions) region id, -1 = REGION_OTHER
    signature: str
    # Per drawn face, the part's own triangle it shows (split paint is drawn as sub-triangles, overlapping parts are
    # clipped): what a highlight of the part's triangles (`view_faces`) draws.
    sources: list[np.ndarray]


def _preview_meshes(session: Session, params: Params, regions: tuple[str, float] | None) -> PreviewMeshes:
    """The preview's parts at scale 1 in the figure frame, cached for the last setting (the
    glTF and the face colours of one view are asked for one after the other)."""
    pad = None if regions else session.pads(params)
    key = json.dumps([session._key(params), session.texture_count(params), params.texture_details, params.paint,
                      regions, digest(pad.vertices, pad.faces).hex() if pad is not None else None],
                     sort_keys=True, default=str)
    hit = session._preview.get(key)
    if hit is not None:
        return hit
    obj = session.obj(params)
    unit = session.base_parts(params) if regions else session.figure_parts(params)
    named = session.painted_parts(params)             # the triangles a highlight names (`sources`)
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    keys: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    if regions:
        # Overlap clipping re-meshes parts and loses the face order the region labels need: skip it here.
        topo = session.topology(params)
        mode = "part" if regions[0] in ("brush", "stamp") else regions[0]   # no fixed regions: show parts
        for pi, part in enumerate(unit):
            meshes.append((part.vertices, part.faces))
            sources.append(np.arange(len(part.faces), dtype=np.int64))
            if mode == "shape":                 # the shape map is drawn per vertex (`preview_faces`)
                keys.append(np.where(np.arange(len(part.faces)) >= 0, 0 if pi < len(topo) else -1, 0))
            else:
                keys.append(topo.labels(pi, mode, regions[1]) if pi < len(topo)
                            else np.full(len(part.faces), -1, np.int64))
    else:
        for pi, clipped in enumerate(clip_overlaps(unit)):
            pv, pf, ext, src = painted_mesh(clipped, session.extruder(params))
            own = named[pi] if pi < len(named) else unit[pi]
            if clipped is not own:
                src = _nearest_faces(own, clipped)[src]           # re-meshed (clip, strengthen): by position
            meshes.append((pv, pf))
            keys.append(ext)
            sources.append(src)
    sig = hashlib.blake2b(b"".join(digest(v, f) for v, f in meshes), digest_size=12).hexdigest()
    out = PreviewMeshes(meshes, keys, sig, sources)
    session._preview.clear()
    session._preview[key] = out
    return out


def _nearest_faces(part: Part, other: Part) -> np.ndarray:
    """Per triangle of `other` (a re-meshed copy of `part`), the triangle of `part` whose centre is nearest its own."""
    from scipy.spatial import cKDTree

    if not len(part.faces):
        return np.full(len(other.faces), -1, np.int64)
    _, idx = cKDTree(part.vertices[part.faces].mean(axis=1)).query(other.vertices[other.faces].mean(axis=1))
    return np.asarray(idx, np.int64)


def view_faces(session: Session, params: Params, part: int, faces: np.ndarray) -> list[int]:
    """The preview's triangles (`preview_glb` without regions, mesh `p<part>`) that show `faces` of that part."""
    pm = _preview_meshes(session, params, None)
    if not 0 <= part < len(pm.sources):
        return []
    src = pm.sources[part]
    mask = np.zeros(int(max(src.max(initial=-1), faces.max(initial=-1))) + 1, bool)
    mask[faces] = True
    return np.flatnonzero((src >= 0) & mask[np.maximum(src, 0)]).tolist()


def preview_glb(session: Session, params: Params, regions: tuple[str, float] | None = None) -> bytes:
    """Figure at scale 1 in the figure frame: one uncoloured mesh per part, named `p<i>`, faces in
    the order `preview_faces` colours them. Split paint is drawn as its sub-triangles, so brush
    strokes and texture colour edges show as the slicer prints them.

    With `regions` (colorize mode, angle) the parts are the unclipped ones whose
    face order the region labels follow.
    """
    session.use_simplify(params.simplify)
    pm = _preview_meshes(session, params, regions)
    scene = trimesh.Scene()
    for pi, (v, f) in enumerate(pm.meshes):
        if len(f):
            scene.add_geometry(trimesh.Trimesh(v, f, process=False), node_name=f"p{pi}", geom_name=f"p{pi}")
    out = scene.export(file_type="glb")
    assert isinstance(out, bytes)
    return out


def preview_faces(session: Session, params: Params, regions: tuple[str, float] | None = None) -> dict[str, Any]:
    """Colours of `preview_glb`'s triangles: {signature, colours, parts}.

    `parts[i]` is base64 of one byte per face of mesh `p<i>`: an index into `colours` ("#rrggbb").
    `signature` names the geometry: while it equals the loaded glTF's, a paint or palette edit
    only recolours the view (a few hundred kB) instead of re-sending the whole mesh.
    With `regions`, colours are the regions a tap would paint (cycling REGION_PALETTE colours).
    """
    session.use_simplify(params.simplify)
    import base64

    pm = _preview_meshes(session, params, regions)
    if regions and regions[0] == "shape":
        # Shape map: one neutral colour; per vertex, crease and relief bytes the viewer shades and draws lines of.
        topo = session.topology(params)
        shape = []
        for pi, (v, f) in enumerate(pm.meshes):
            per = np.full((len(v), 2), (0, 128), np.uint8)
            if pi < len(topo):
                t = topo.part(pi)
                per[f.ravel()] = t.shape_map()[t.vfaces.ravel()]
            shape.append(base64.b64encode(per.tobytes()).decode("ascii"))
        return {"signature": pm.signature, "colours": [SHAPE_COLOUR, REGION_OTHER],
                "parts": [base64.b64encode(np.where(k < 0, 1, 0).astype(np.uint8).tobytes()).decode("ascii")
                          for k in pm.keys], "shape": shape}
    if regions:
        colours = ["#%02x%02x%02x" % tuple(int(c) for c in rgb) for rgb in region_colours(REGION_PALETTE)]
        colours.append(REGION_OTHER)
        idx = [np.where(k < 0, REGION_PALETTE, k % REGION_PALETTE) for k in pm.keys]
    else:
        colours = [c[:7] for c in session.colours(params)] or ["#cccccc"]
        idx = [(k - 1) % len(colours) for k in pm.keys]
    out: dict[str, Any] = {"signature": pm.signature, "colours": colours,
                           "parts": [base64.b64encode(np.asarray(i, np.uint8).tobytes()).decode("ascii") for i in idx]}
    if regions is None:
        # Which slot each used colour prints from: cheap, and a paint step changes which colours are used, so the
        # UI's "Prints from" follows every tap without waiting for the next analysis.
        out["import_map"] = [m.to_dict() for m in session.slot_matches(params)]
    return out


def _pick_trees_for(session: Session, params: Params) -> list[Any]:
    key = session._key(params)
    if key not in session._pick_trees:
        session._pick_trees.clear()                 # one orientation at a time: trees are large
        session._pick_trees[key] = centroid_trees(session.base_parts(params))
    return session._pick_trees[key]


def prepare_paint(session: Session, params: Params, mode: str = "part", angle: float = 15.0,
                  mirror: bool = False) -> dict[str, Any]:
    """Build what taps need (pick trees, edge topology, the mode's regions; with `mirror`, the symmetry plane)
    before the first tap: the UI calls it when painting starts, so the first tap on a big mesh answers as fast as
    the rest. Says whether the figure is symmetric enough to paint both sides at once."""
    session.use_simplify(params.simplify)
    _pick_trees_for(session, params)
    topo = session.topology(params)
    for i in range(len(topo)):
        t = topo.part(i)
        if mode in ("wand", "brush", "stamp"):  # soft edges walk the vertex graph; the stamp and wand use the KD-tree
            t.vertex_graph()
            _ = t.tree
        if mode == "wand":                     # its bend field and the "part" regions it also offers
            t.wall()
            for a in WAND_ANGLES:
                topo.labels(i, "part", a)
        elif mode not in ("brush", "face", "stamp"):
            topo.labels(i, mode, angle)
    out: dict[str, Any] = {"parts": len(topo)}
    if mirror:
        out["mirror"] = _mirror_info(topo)
    return out


def pieces_of_colour(session: Session, params: Params, filament: int) -> dict[str, Any]:
    """The connected patches of one colour (`colorize.colour_pieces`), largest first, for "Split by piece": the UI
    keeps the largest in the colour and gives each other one a new filament with a "fill" step at its seed.
    Areas are at the print size (mm²; the file's mm when nothing fits)."""
    session.use_simplify(params.simplify)
    if not 1 <= filament <= 64:
        raise ValueError(f"filament must be 1-64, not {filament}")
    s = session.paint_scale(params) or 1.0
    found = colour_pieces(session.topology(params), session.painted_parts(params), filament,
                          session.extruder(params))
    return {"filament": filament, "pieces": [{"part": p.part, "face": p.face, "area_mm2": round(p.area * s * s, 3),
                                              "faces": p.faces} for p in found]}


def limb_region(session: Session, params: Params, point: list[float] | None, normal: list[float] | None,
                angle: float = 15.0, at: tuple[int, int] | None = None) -> dict[str, Any]:
    """The limb under a tap on the 3D preview (figure frame, scale 1), or at triangle `at` = (part, face), for
    "Reshape a part": the triangle that names it in a `reshape.LimbOp` (part, face), its triangles (the view
    highlights them), its length along the surface at the print size and whether it meets the body (a separate
    piece turns about its own middle)."""
    session.use_simplify(params.simplify)
    if at is not None:
        pi, fi = int(at[0]), int(at[1])
    elif point is not None:
        parts = session.base_parts(params)
        pi, fi, _ = pick_face(parts, np.asarray(point, float)[:3],
                              None if normal is None else np.asarray(normal, float)[:3],
                              trees=_pick_trees_for(session, params))
    else:
        raise ValueError("a limb needs a point or a triangle")
    if pi < 0:
        raise ValueError("no triangle near that point")
    topo = session.topology(params)
    if pi >= len(topo):
        raise ValueError("That is the flat base under the figure, not a part of it.")
    # A tap (not a stored limb's own triangle and angle) steps down to a smaller angle if this one takes the body.
    limb, angle = find_limb_at(topo, pi, fi, angle) if at is None else (find_limb(topo, pi, fi, angle), angle)
    s = session.paint_scale(params) or 1.0
    return {"part": pi, "face": fi, "angle": angle, "faces": limb.faces.tolist(),
            "view_faces": view_faces(session, params, pi, limb.faces),
            "share": round(float(topo.part(pi).areas[limb.faces].sum() / max(topo.part(pi).areas.sum(), 1e-12)), 4),
            "length_mm": round(limb.length * s, 2), "joined": bool(len(limb.joint))}


def weak_points(session: Session, params: Params) -> dict[str, Any]:
    """Thin features graded by the push at their far end that would snap them (joints.py), weakest first, measured
    on the figure as it prints (strengthened, at the fitted size). Each names a triangle of `base_parts` (`part`,
    `face`: what `Params.strengthen_at` takes) and carries the preview triangles that show the piece that would
    break off (`view_faces`)."""
    from . import joints

    session.use_simplify(params.simplify)
    scale = solved(session, params)[0]
    unit, base = session.unit_parts(params), session.base_parts(params)
    t0 = time.time()
    found = joints.weak_points(unit, scale) if scale > 0 else []
    out = []
    for w in found:
        d = w.to_dict()
        faces = w.faces
        if unit[w.part] is not base[w.part] and len(faces):         # strengthened: a re-meshed copy of the part
            faces = np.unique(_nearest_faces(base[w.part], replace(unit[w.part], faces=unit[w.part].faces[faces])))
        # `strengthen_at`'s triangles (strengthen.Local): `root` nearest the section's centre, `face` the piece's
        # triangle nearest the last section past it still under joints.GOAL: thickening covers the stretch between.
        bp = base[w.part]
        cen = bp.vertices[bp.faces].mean(axis=1)
        d["root"] = int(np.argmin(((cen - np.asarray(w.centre)) ** 2).sum(axis=1))) if len(bp.faces) else -1
        d["face"] = int(faces[np.argmin(((cen[faces] - np.asarray(w.end)) ** 2).sum(axis=1))]) if len(faces) else -1
        d["view_faces"] = view_faces(session, params, w.part, faces) if len(faces) else []
        out.append(d)
    return {"scale": scale, "height_mm": round(scale * session.points(params).height, 2), "points": out,
            "grades": {g: lim for lim, g in joints.GRADES}, "probe_mm": joints.PROBE,
            "seconds": round(time.time() - t0, 2)}


def _mirror_info(topo: TopologyCache) -> dict[str, Any]:
    plane = topo.mirror()
    return {"ok": plane.error <= MIRROR_MAX_ERROR, "error": round(plane.error, 4),
            "normal": [round(float(x), 4) for x in plane.normal]}


def pick(session: Session, params: Params, point: list[float], normal: list[float] | None,
         mode: str = "part", angle: float = 15.0, radius: float = 0.0,
         stroke: list[dict[str, Any]] | None = None, mirror: bool = False, grow: float = 0.0,
         level: int = 0, stamp_axes: list[float] | None = None) -> dict[str, Any]:
    """The triangle under a tap on the 3D preview (figure frame, scale 1) and the size of its region.

    A stamp (`mode` "stamp") is an ellipse centred on `point` with semi-axes `stamp_axes` = [u, v] (6 numbers,
    figure frame, scale 1, seen along u x v); the answer's `stamp` carries it in the paint operation's own frame.

    The answer's part/face go into a colorize paint operation, which then
    paints the same region on every later analysis and export. `stroke` (brush
    only): further {point, normal} samples of a drag; their triangles on the
    same part come back as `path`. Fills are counted against the current
    paint, so a fill that stops at a painted outline reports the faces inside it.
    With `mirror` the count includes the other side (`mirrored`: one was found).
    `filament` is the colour shown under the tap (the UI's colour picker).
    """
    session.use_simplify(params.simplify)
    PaintOp.from_dict({"part": 0, "face": 0, "filament": 0, "mode": mode, "angle": angle, "radius": radius,
                       "grow": grow, "level": level, **({"stamp": UNIT_STAMP} if mode == "stamp" else {})})
    if stroke and mode != "brush":
        raise ValueError("only the brush paints strokes")
    if (mode == "stamp") != (stamp_axes is not None):
        raise ValueError("a stamp needs its two semi-axes, and only a stamp has them")
    parts = session.base_parts(params)
    trees = _pick_trees_for(session, params)

    def at(pt: list[float], nrm: list[float] | None) -> tuple[int, int, float]:
        return pick_face(parts, np.asarray(pt, float)[:3], None if nrm is None else np.asarray(nrm, float)[:3],
                         trees=trees)

    pi, fi, dist = at(point, normal)
    if pi < 0:
        raise ValueError("no triangle near that point")
    topo = session.topology(params)
    if pi >= len(topo):
        raise ValueError("That is the flat base under the figure: it prints in the figure's main colour.")
    path: list[int] = []
    # A stroke's own points, in the frame paint operations live in: the brush follows where the drag went.
    points: list[float] = []
    if mode == "brush" and stroke:
        points.extend(_file_point(session, params, pi, fi, np.asarray(point, float)[:3]))
    for sample in (stroke or [])[:MAX_PATH]:
        sp, sf, _ = at(sample["point"], sample.get("normal"))
        if sp != pi:
            continue
        points.extend(_file_point(session, params, sp, sf, np.asarray(sample["point"], float)[:3]))
        if sf != fi and (not path or path[-1] != sf):
            path.append(sf)
    figure = session.painted_parts(params)
    states: dict[int, PaintState] = {}

    def state(i: int) -> PaintState | None:
        if mode not in BOUNDED or not params.paint:
            return None
        if i not in states:
            states[i] = PaintState(figure[i])
        return states[i]

    stamp: tuple[float, ...] = ()
    if stamp_axes is not None:
        stamp = _stamp_frame(session, params, pi, fi, np.asarray(point, float)[:3], np.asarray(stamp_axes, float))
    op = PaintOp.from_dict(PaintOp(pi, fi, 0, mode, angle, radius, tuple(path), mirror, grow, level,
                                   stamp, tuple(points)).to_dict())
    sides = op_faces(topo, op, state)
    n = sum(len(f) for _, f in sides)
    area = 0.0
    for sp_, f in sides:
        tri = parts[sp_].vertices[parts[sp_].faces[f]]
        area += float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    attrs = figure[pi].tri_extra.get(fi)
    fp = attr_paint(attrs) if attrs else None
    shown = fp.majority if fp is not None and fp.majority > 0 else (figure[pi].extruder or session.extruder(params))
    return {"part": pi, "face": fi, "distance": dist, "faces": int(n), "path": path,
            "share": round(len(sides[0][1]) / max(1, len(parts[pi].faces)), 4), "area_mm2": round(area, 2),
            "mirrored": len(sides) > 1, "filament": int(shown), **({"stamp": list(stamp)} if stamp else {}),
            **({"points": [round(x, 5) for x in points]} if points else {})}


def _file_point(session: Session, params: Params, pi: int, fi: int, point: np.ndarray) -> tuple[float, float, float]:
    """A point on triangle `fi` of part `pi` in the figure frame (scale 1), at the same spot of the same triangle in
    the frame paint operations live in (the file's, before rotation and the conform step)."""
    fig_tri = session.base_parts(params)[pi]
    src = session.topology(params)._parts[pi]
    a, b, c = fig_tri.vertices[fig_tri.faces[fi]]
    # Barycentric coordinates of the point in the figure-frame triangle, applied to the same triangle in the file.
    m = np.column_stack([b - a, c - a])
    uv, *_ = np.linalg.lstsq(m, point - a, rcond=None)
    sa, sb, sc = src.vertices[src.faces[fi]]
    q = sa + uv[0] * (sb - sa) + uv[1] * (sc - sa)
    return float(q[0]), float(q[1]), float(q[2])


def _stamp_frame(session: Session, params: Params, pi: int, fi: int, point: np.ndarray, axes: np.ndarray
                 ) -> tuple[float, ...]:
    """A stamp drawn in the figure frame (centre `point` on triangle `fi`, semi-axes `axes` = u, v), in the frame
    paint operations live in (the file's, before rotation): the centre at the same spot of the same triangle,
    the axes turned back by the figure's rotation."""
    if len(axes) != 6 or not np.isfinite(axes).all():
        raise ValueError("stamp_axes must be 6 numbers (two semi-axes)")
    centre = np.asarray(_file_point(session, params, pi, fi, point))
    _, rx, ry, rz = session._orientation(params)
    rot = (trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]) @
           trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]) @
           trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))[:3, :3]
    u, v = rot.T @ axes[:3], rot.T @ axes[3:]
    v = v - u * (u @ v) / max(float(u @ u), 1e-12)        # exactly at right angles
    return tuple(float(x) for x in (*centre, *u, *v))


def with_paint_depth(settings: dict | None, params: Params) -> dict | None:
    """`settings` with the chosen paint depth (a copy; unchanged when `params.paint_depth` is None)."""
    if settings is None or params.paint_depth is None:
        return settings
    return {**settings, "mmu_segmented_region_max_width": f"{params.paint_depth:g}"}


def export_settings(session: Session, params: Params, supports: bool = False) -> dict | None:
    if params.settings_source == "reference":
        built = session.print_settings(params, supports)
        if built is not None:
            return with_paint_depth(built[0], params)
    if params.colours and session.project.settings:
        return with_paint_depth(resize_filaments(session.project.settings, session.colours(params)), params)
    return with_paint_depth(session.project.settings, params)


def safe_stem(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(name).stem)
    return keep.strip("_") or "figure"


def export_stem(session: Session, params: Params) -> str:
    """The start of exported file names: the figure's own name in a file of several ("jolteon" from the object
    "jolteon.stl" in "Eevee+Evolutions.3mf"), else the file's name."""
    if len(session.project.objects) > 1:
        name = session.obj(params).name
        if name and safe_stem(name) != "figure":
            return safe_stem(name)
    return safe_stem(session.project.path.name)


# Params an export's shape was made with that opening it again must reuse: its base (press, gap fill, pads) is in the
# mesh already, and the app's defaults would press and fill a figure that was exported without them.
MADE_WITH_KEYS = ("flat_base", "flat_depth", "flat_fill", "fill_area")


def made_with(params: Params) -> dict[str, Any]:
    """What an export records about how it was made (model3mf.POKEFIT_META; report `object.made_with` on reopening)."""
    return {"version": 1, "params": {k: getattr(params, k) for k in MADE_WITH_KEYS}}


def export(session: Session, params: Params, out_dir: str | Path, ball_model: Path | None = None) -> dict[str, Any]:
    """Write the figure 3MF, its import variant, a coloured STL (`stlcolour`), the steel seat shell STL and a
    JSON report.

    `<stem>_pokefit_<pct>.3mf` opens as a project (File > Open). `..._import.3mf`
    is for dragging into an open project with "Import geometry only": its
    filament numbers are the user's project slots of the closest colour.
    """
    session.use_simplify(params.simplify)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep, final = analyse(session, params)
    obj = session.obj(params)
    ball_s = session.ball_scale(params)
    # File tag: the ball's print percentage, or its inner diameter when that was given.
    pct = f"d{params.ball_diameter:.0f}mm" if params.ball_diameter else str(int(round(ball_s * 100)))
    stem = export_stem(session, params)
    # Weak features printed solid (joints.reinforcements): Bambu modifier volumes, written to the 3MFs only.
    settings = export_settings(session, params)
    mods, held = reinforced(final, settings) if params.reinforce else ([], [])
    f3 = write_project(session.project, obj.name, session.extruder(params), final + mods,
                       out_dir / f"{stem}_pokefit_{pct}.3mf", settings=settings, made=made_with(params))
    shell_path = out_dir / f"shell_floor{session.cavity(params).seat_diameter:.1f}mm_{pct}.stl"
    shell_note = ""
    try:
        shell = _shell_mesh(ball_model or BALL_MODEL)
        fs = flat_spot(shell, ball_s, SHELL_DISK, session.cavity(params).seat_diameter)   # the floor the fit used
        fs.mesh.export(shell_path)
    except (OSError, ValueError, RuntimeError) as exc:
        shell_path, shell_note = None, f"steel seat shell not written: {exc}"  # type: ignore[assignment]
    # For "Import geometry only" into the user's own project: filaments renumbered to its slots.
    imp = None
    matches = [m for m in session.slot_matches(params) if m.used]
    if matches and session.reference:
        parts, ext = remap_parts(final + mods, session.extruder(params), matches)
        imp = write_project(session.project, obj.name, ext, parts, out_dir / f"{stem}_pokefit_{pct}_import.3mf",
                            settings=session.reference, made=made_with(params))
    # The same figure as one STL with a colour per triangle, for tools that take STL (Bambu Studio ignores it).
    cstl, cnote = write_coloured_stl(out_dir / f"{stem}_pokefit_{pct}_colour.stl", final, session.extruder(params),
                                     session.colours(params))
    shell_note = "; ".join(n for n in (shell_note, cnote) if n)
    rpath = out_dir / f"{stem}_pokefit_{pct}.json"
    rpath.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return {"figure_3mf": str(f3), "import_3mf": str(imp) if imp else None, "colour_stl": str(cstl),
            "shell_stl": str(shell_path) if shell_path else None, "report": str(rpath),
            "note": shell_note, "status": rep.status, "import_map": [m.to_dict() for m in matches],
            "reinforced": [w.to_dict() for w in held], "reinforce_volumes": len(mods)}


def reinforced(final: list[Part], process: dict | None) -> tuple[list[Part], list[Any]]:
    """Modifier parts printing the weak features of `final` (print size) more solid than `process` does, and the
    weak points graded for it (joints.reinforcements)."""
    from .joints import reinforcements

    return reinforcements(final, 1.0, process=process)


def _shell_mesh(model: Path) -> trimesh.Trimesh:
    """The ball's shell mesh from its model (third-party: MakerWorld #2045372, not redistributed)."""
    if not model.exists():
        raise ValueError("it is cut from the ball model, which pokefit cannot redistribute: load your copy of "
                         "the MakerWorld model #2045372 (.3mf) to get it")
    proj = read_project(model)
    shells = [o for o in proj.objects if o.name.startswith("0_ball_shell")]
    if not shells:
        raise ValueError("pokeball.3mf has no 0_ball_shell object")
    m = trimesh.util.concatenate([p.mesh() for p in shells[0].parts])
    assert isinstance(m, trimesh.Trimesh)
    return m


def slice_variants(session: Session, params: Params, x10: bool = True,
                   work_root: str | Path | None = None) -> dict[str, Any]:
    """Bambu slices: supports off, auto supports, and (best-effort) 10 copies with supports as needed."""
    session.use_simplify(params.simplify)
    _, final = analyse(session, params)
    obj = session.obj(params)
    root = Path(work_root) if work_root else session.work_dir / f"slice_{int(time.time() * 1000)}"
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    stem = export_stem(session, params)
    for name, sup in (("off", False), ("auto", True)):
        f = write_project(session.project, obj.name, session.extruder(params), final, root / f"{name}.3mf",
                          settings=_slice_settings(session, params, sup))
        out[name] = slice_3mf(f, work_dir=root / name, export_name=f"{stem}_{name}.gcode.3mf").to_dict()
    needs = out["auto"].get("support_seconds", 0) > 0
    out["supports_needed"] = needs if out["auto"].get("ok") else None
    # The print file of the variant the slicer says to print (supports only if it generated any).
    chosen = out["auto"] if needs else out["off"]
    out["print_file"] = chosen.get("print_file") or None
    # The AMS slot each filament of the print file is sliced for (send-to-printer passes them on).
    spools = session.ams_spools(params)
    loaded = [a for a in spools if a is not None]
    out["ams_slots"] = [a.slot for a in loaded] if spools and len(loaded) == len(spools) else None
    if x10:
        v = np.vstack([p.vertices for p in final])
        ext = np.ptp(v, axis=0)
        st = _slice_settings(session, params, needs)
        pos = grid_positions((float(ext[0]), float(ext[1])), 10, plate_size(st))
        if pos is None:
            out["x10"] = SliceResult(False, error="10 copies do not fit on one plate", copies=10).to_dict()
        else:
            f = write_project(session.project, obj.name, session.extruder(params), final, root / "x10.3mf",
                              positions=pos, settings=st)
            out["x10"] = slice_3mf(f, copies=10, work_dir=root / "x10").to_dict()
    return out


def _slice_settings(session: Session, params: Params, supports: bool) -> dict | None:
    built = session.print_settings(params, supports)
    if built is not None:
        return with_paint_depth(built[0], params)
    st = dict(export_settings(session, params) or {})
    if st:
        st["enable_support"] = "1" if supports else "0"
    return st or None

