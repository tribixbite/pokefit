"""One analysis pipeline shared by the CLI and the web server.

Frames:
  * figure frame — final print mm, XY centred on the figure's footprint
    bounding box, z = 0 on the bed. Pockets, check highlights and exported
    meshes use it.
  * ball frame — z = 0 at the ball centre, ball axis at XY 0. The figure frame
    sits in it at (offset_x, offset_y, cavity.floor_z).
"""
from __future__ import annotations

import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .ball import Ball, Cavity, load_ball
from .checks import (bed_contact_check, merged, islands_check, mesh_checks, overhang_check, printable_solid,
                     supports_verdict, thickness_check)
from .conform import DEFAULT_FILL, DEFAULT_MAX_DEPTH, FILL_NAME, ConformInfo, ConformSpec, conform
from .colorize import PaintOp, TopologyCache, apply_paint, pick_face, region_colours, region_faces
from .fit import HullPoints, annulus_margin, best_offset, check_fit, max_scale, prepare_points
from .magnets import (HeightField, Pocket, PocketMap, PocketSpec, choose_pockets, cut_pockets, find_feet,
                      height_field, pocket_map, required_washer, seat_pockets, validate_pockets)
from .model3mf import (FigureObject, Part, Project, plate_size, read_project, remesh_part, world_parts,
                       write_project)
from .gltf import MAX_DETAILS, MAX_PICK
from .paint import decode_majority, leaves, parse_tree
from .presets import PresetLibrary, PresetNote, ams_filaments
from .printer import AmsSlot
from .remap import SlotMatch, match_slots, remap_parts, used_filaments
from .report import Check, worst
from . import toolchange as tc
from .shell import flat_spot, from_manifold, to_manifold
from .slicer import (SliceResult, adapt_settings, default_slot, grid_positions, reference_settings,
                     resize_filaments, slice_3mf)
from .washer import CATALOGUE, DEFAULT_WASHER, Washer, get_washer

ROOT = Path(__file__).resolve().parent.parent
# The ball's derived measurements (radii, ring profile) ship with the package; the
# model itself (MakerWorld #2045372) is third-party and only needed locally for
# the washer-seat shell export.
BUNDLED_BALL = Path(__file__).resolve().parent / "profiles" / "ball_functional_pokeball.json"
BALL_MODEL = ROOT / "pokeball.3mf"
DEFAULT_BALL = BUNDLED_BALL
# The user's own printer/process/filament settings: their printed-shells project
# if present, else the copy of its settings bundled with the package.
BUNDLED_REFERENCE = Path(__file__).resolve().parent / "profiles" / "x2d_petg_printed_ball_shells.json"
DEFAULT_REFERENCE = ROOT / "pokeball60goodPrinted.3mf" if (ROOT / "pokeball60goodPrinted.3mf").exists() else BUNDLED_REFERENCE
CACHE_DIR = ROOT / ".pokefit_cache"
SHELL_DISK = 8.0      # outer flat disk of the washer-seat shell, final mm
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")
MIN_SCALE_FOR_MAGNETS = 0.6   # never shrink a figure below this fraction of its fit size for the magnets
AUTO_FILLS = (4.0, 6.0)        # mm: deeper flat-base fills tried, in order, when no magnet pocket fits at the default


@dataclass
class Params:
    ball_scale: float = 0.6                      # print scale of the ball model (0.6 = the 60 % ball)
    ball_diameter: float | None = None           # inner diameter in mm; overrides ball_scale when set
    washer: str = DEFAULT_WASHER
    clearance: float = 1.0
    figure_scale: float | None = None            # None = largest that fits
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
    pocket_max_seat: float = 2.0                 # raised, auto placement: highest pocket opening above the washer
    pocket_seat: str = "washer"                  # "washer": openings on the washer; "raised": into floating parts
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
    # glTF colour (texture, vertex colours, materials): how many filaments to reduce it to
    # (None = automatic, see gltf.py). Ignored for other files.
    texture_colours: int | None = None
    # ... plus up to this many "detail" filaments: small features that contrast most with the colour they would
    # merge into (black pupils on a cream head) keep their own (gltf.TextureColours._details). 0..MAX_DETAILS.
    texture_details: int = 0
    # Conform to the ball's floor (conform.py): bend legs/tails that reach past the washer up along the
    # sphere (None = automatic, 0 = off, else strength 0..1) and make the base flat over a disc of
    # `flat_base` mm (None = the washer's diameter, 0 = off): press it by at most `flat_depth` mm,
    # then fill gaps up to `flat_fill` mm under the figure with a solid "flat base" part.
    bend: float | None = None
    flat_base: float | None = None
    flat_depth: float = DEFAULT_MAX_DEPTH
    flat_fill: float = DEFAULT_FILL
    # How far painted colours reach into the figure, in mm (Bambu `mmu_segmented_region_max_width`; deeper is
    # the object's own filament). None = the project's value (0 there = unlimited: paint fills inward until it
    # meets other paint). Measured on an 8-colour figure: time and colour changes stay the same (the colours
    # still appear on the same layers); only how much of each colour, and of the base, is used.
    paint_depth: float | None = None

    def __post_init__(self) -> None:
        # Older clients send magnets as a switch: on = the default two pockets.
        if isinstance(self.magnets, bool):
            self.magnets = 2 if self.magnets else 0
        self.magnets = max(0, min(3, int(self.magnets)))
        if self.pocket_seat not in ("washer", "raised"):
            raise ValueError(f"pocket_seat must be 'washer' or 'raised', not {self.pocket_seat!r}")
        if self.colours is not None:
            if not self.colours or not all(isinstance(c, str) and _HEX.fullmatch(c) for c in self.colours):
                raise ValueError("colours must be a non-empty list of #RRGGBB strings")
            if len(self.colours) > 32:
                raise ValueError("at most 32 colours")
        for d in self.paint or []:
            PaintOp.from_dict(d)          # raises ValueError on a malformed operation
        if self.texture_colours is not None:
            self.texture_colours = int(self.texture_colours)
            if not 1 <= self.texture_colours <= MAX_PICK:
                raise ValueError(f"texture_colours must be 1..{MAX_PICK}")
        self.texture_details = int(self.texture_details)
        if not 0 <= self.texture_details <= MAX_DETAILS:
            raise ValueError(f"texture_details must be 0..{MAX_DETAILS}")
        if self.bend is not None and not 0 <= float(self.bend) <= 1:
            raise ValueError("bend must be 0..1 (or null for automatic)")
        if self.flat_base is not None and not 0 <= float(self.flat_base) <= 60:
            raise ValueError("flat_base must be 0..60 mm (or null for the washer's diameter)")
        if not 0 <= float(self.flat_depth) <= 5:
            raise ValueError("flat_depth must be 0..5 mm")
        if not 0 <= float(self.flat_fill) <= 10:
            raise ValueError("flat_fill must be 0..10 mm")
        if self.paint_depth is not None:
            self.paint_depth = float(self.paint_depth)
            if not 0 <= self.paint_depth <= 10:
                raise ValueError("paint_depth must be 0..10 mm (0 = unlimited, or null for the project's)")

    @property
    def paint_ops(self) -> list[PaintOp]:
        return [PaintOp.from_dict(d) for d in self.paint or []]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def spec(self) -> PocketSpec:
        return PocketSpec(self.pocket_diameter, self.pocket_depth, self.pocket_wall, self.pocket_ceiling,
                          self.pocket_grip, self.pocket_max_seat, self.pocket_seat == "washer")


@dataclass
class Session:
    id: str
    project: Project
    work_dir: Path
    ball: Ball
    reference: dict | None
    presets: PresetLibrary | None = None        # Bambu presets the user imported (presets.py)
    # Filaments loaded in the user's printer (printer.py, via beambam): when set, the figure's colours are
    # matched to these slots instead of the reference project's.
    ams: list[AmsSlot] | None = None
    input_checks: dict[str, list[Check]] = field(default_factory=dict)   # object id -> checks
    _frames: dict[tuple, list[Part]] = field(default_factory=dict)           # orientation + conform -> parts
    _oriented: dict[tuple, list[Part]] = field(default_factory=dict)         # orientation only
    _conform: dict[tuple, ConformInfo] = field(default_factory=dict)
    _fills: dict[tuple, float] = field(default_factory=dict)                 # `_auto_fill` decisions
    _points: dict[tuple[str, float, float, float], HullPoints] = field(default_factory=dict)
    _fields: dict[tuple[str, float, float, float], HeightField] = field(default_factory=dict)  # unit-scale undersides
    # Auto-placement results per configuration (the figure-scale-independent search).
    _auto: dict[tuple, Any] = field(default_factory=dict)
    _topo: dict[str, TopologyCache] = field(default_factory=dict)       # object id -> colorize topology
    _painted: dict[tuple, list[Part]] = field(default_factory=dict)     # (orientation, paint) -> parts
    _used: dict[str, list[int]] = field(default_factory=dict)   # object id -> filament numbers it uses
    _textured: dict[tuple, list[Part]] = field(default_factory=dict)    # (orientation, count) -> parts
    _settings: dict[tuple, dict] = field(default_factory=dict)          # (presets version, process) -> settings
    _tc_table: dict[tuple, list[dict[str, Any]]] = field(default_factory=dict)   # palette-size cost tables
    _auto_count: dict[tuple, int | None] = field(default_factory=dict)     # glTF auto palette size per setup

    def obj(self, params: Params) -> FigureObject:
        return self.project.object(params.object_id) if params.object_id else self.project.default_object()

    def _orientation(self, params: Params) -> tuple[str, float, float, float]:
        """Object id and the three angles (degrees, mod 360)."""
        return (self.obj(params).id, round(params.rotate_x % 360, 6), round(params.rotate_y % 360, 6),
                round(params.rotate_z % 360, 6))

    def _key(self, params: Params) -> tuple:
        """Cache key of the figure-frame geometry: the orientation, plus what the conform step
        depends on when it is on (its settings, the ball size and the clearance; the washer only
        through the flat disc's diameter, see `unit_parts`)."""
        return self._spec_key(params, self.conform_spec(params))

    def _spec_key(self, params: Params, spec: ConformSpec | None) -> tuple:
        if spec is None:
            return self._orientation(params)
        return self._orientation(params) + (spec.bend, spec.flat_diameter, spec.max_depth, spec.fill,
                                             round(self.ball_scale(params), 6), params.clearance)

    def conform_spec(self, params: Params) -> ConformSpec | None:
        """The conform step's settings, or None when bend and flat base are both off. The default flat-base
        fill is raised (`AUTO_FILLS`) when no magnet pocket fits under the figure with it (`_auto_fill`)."""
        flat = get_washer(params.washer).od if params.flat_base is None else float(params.flat_base)
        bend = None if params.bend is None else float(params.bend)
        if bend == 0 and (flat <= 0 or (params.flat_depth <= 0 and params.flat_fill <= 0)):
            return None
        spec = ConformSpec(bend, flat, float(params.flat_depth), float(params.flat_fill))
        if spec.fill == DEFAULT_FILL and flat > 0 and params.magnets and params.pockets is None:
            spec = replace(spec, fill=self._auto_fill(params, spec))
        return spec

    def _auto_fill(self, params: Params, spec: ConformSpec) -> float:
        """The flat-base fill to use at the default setting: DEFAULT_FILL, or the first of AUTO_FILLS under
        which a magnet pocket fits at the largest size (a figure on four thin legs, like an Eevee, has room
        for none until the base fills the gap under its belly). Decided against the default washer, like the
        conform shape itself, so every washer in the table sees the same figure."""
        key = self._spec_key(params, spec) + (params.spec, params.magnets)
        if key not in self._fills:
            self._fills[key] = spec.fill
            for fill in (spec.fill, *AUTO_FILLS):
                parts = self._conformed(params, replace(spec, fill=fill))
                cavity = Cavity(self.ball, self.ball_scale(params), get_washer(DEFAULT_WASHER), params.clearance)
                free = max_scale(self._points[self._spec_key(params, replace(spec, fill=fill))], cavity)
                if free is None:
                    break
                field = height_field([p.mesh() for p in parts if p.subtype == "normal_part"])
                pockets, _ = choose_pockets(pocket_map(field.scaled(free.max_scale), params.spec), params.spec,
                                            params.magnets)
                if pockets:
                    self._fills[key] = fill
                    break
        return self._fills[key]

    def conform_info(self, params: Params) -> ConformInfo | None:
        self.unit_parts(params)
        return self._conform.get(self._key(params))

    def ball_scale(self, params: Params) -> float:
        """Print scale of the ball: from the inner diameter when one is given."""
        if params.ball_diameter:
            return float(params.ball_diameter) / (2 * self.ball.r_inner)
        return params.ball_scale

    def unit_parts(self, params: Params) -> list[Part]:
        """Figure-frame parts at scale 1 (rotated, XY-centred, on z = 0)."""
        return self._conformed(params, self.conform_spec(params))

    def _conformed(self, params: Params, spec: ConformSpec | None) -> list[Part]:
        """`unit_parts` with the conform step's settings given (cached under the same key)."""
        obj = self.obj(params)
        key = self._spec_key(params, spec)
        if key not in self._frames:
            okey = self._orientation(params)
            if okey not in self._oriented:
                parts = world_parts(obj)
                rot = (trimesh.transformations.rotation_matrix(np.radians(okey[3]), [0, 0, 1]) @
                       trimesh.transformations.rotation_matrix(np.radians(okey[2]), [0, 1, 0]) @
                       trimesh.transformations.rotation_matrix(np.radians(okey[1]), [1, 0, 0]))
                parts = [p.transformed(rot) for p in parts]
                v = np.vstack([p.vertices for p in parts])
                mn, mx = v.min(axis=0), v.max(axis=0)
                shift = trimesh.transformations.translation_matrix([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2,
                                                                    -mn[2]])
                self._oriented[okey] = [p.transformed(shift) for p in parts]
            parts = self._oriented[okey]
            if spec is not None:
                # Designed against the default washer's floor, so the shape (and every cache keyed
                # on it) is the same whatever washer is compared; the lift itself depends only on the
                # sphere and the flat disc.
                cavity = Cavity(self.ball, self.ball_scale(params), get_washer(DEFAULT_WASHER), params.clearance)
                # Vertices move, faces keep their order: paint and texture colour carry over.
                parts, self._conform[key] = conform(parts, cavity, spec)
            self._frames[key] = parts
            self._points[key] = prepare_points(np.vstack([p.vertices for p in parts]))
        return self._frames[key]

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
        key = (self._key(params), round(self.ball_scale(params), 6), params.washer, params.clearance, params.slot,
               params.process, self.presets.version if self.presets else 0)
        if key not in self._auto_count:
            if len(self._auto_count) > 16:
                self._auto_count.clear()
            best = None
            cavity = Cavity(self.ball, self.ball_scale(params), get_washer(params.washer), params.clearance)
            free = max_scale(self.points(params), cavity)
            if free is not None and free.max_scale > 0:
                best = tc.best_value_count(colour_rows(self, params, free.max_scale))
            self._auto_count[key] = best
        return self._auto_count[key]

    def textured_parts(self, params: Params) -> list[Part]:
        """`unit_parts` with a glTF's colour reduced to `texture_count` filaments (the file loads with
        its own automatic palette; geometry and face order are identical)."""
        parts = self.unit_parts(params)
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
        return self._topo[obj.id]

    def figure_parts(self, params: Params) -> list[Part]:
        """`unit_parts` with the file's colour (glTF: at the chosen count) and the colorize paint
        applied (geometry identical)."""
        parts = self.textured_parts(params)
        if not params.paint:
            return parts
        key = (self._key(params), self.texture_count(params), params.texture_details,
               json.dumps(params.paint, sort_keys=True))
        if key not in self._painted:
            if len(self._painted) > 8:          # one figure's recent edits; paint is cheap to redo
                self._painted.clear()
            self._painted[key] = apply_paint(parts, params.paint_ops, self.topology(params))
        return self._painted[key]

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
        ukey = (f"{obj.id}:{self.texture_count(params)}:{params.texture_details}:"
                f"{json.dumps(params.paint, sort_keys=True)}")
        if ukey not in self._used:
            parts = self.figure_parts(params)
            self._used[ukey] = used_filaments(parts, obj.extruder)
        manual = {int(k): int(v) for k, v in (params.filament_map or {}).items()}
        return match_slots(self._used[ukey], self.colours(params), targets, manual)

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
                 reference_path: str | Path | None = DEFAULT_REFERENCE, work_dir: str | Path | None = None) -> Session:
    path = Path(path)
    proj = read_project(path)
    sid = uuid.uuid4().hex[:12]
    wd = Path(work_dir) if work_dir else CACHE_DIR / "sessions" / sid
    wd.mkdir(parents=True, exist_ok=True)
    ref = reference_settings(reference_path) if reference_path and Path(reference_path).exists() else None
    return Session(sid, proj, wd, load_ball(ball_path, CACHE_DIR), ref)


def app_config(slicer: str | None, reference: dict | None = None, reference_name: str = "",
               presets: PresetLibrary | None = None, ams: list[AmsSlot] | None = None) -> dict[str, Any]:
    """What the UI needs before a file is loaded: washers, defaults, the reference printer's filament slots.

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
        "washers": [w.to_dict() for w in CATALOGUE.values()],
        "default_washer": DEFAULT_WASHER,
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


def input_checks(session: Session, params: Params) -> list[Check]:
    """Mesh soundness of the source parts (scale-independent, computed once per object)."""
    obj = session.obj(params)
    if obj.id not in session.input_checks:
        checks: list[Check] = []
        for p in obj.parts:
            checks.extend(mesh_checks(p))
        session.input_checks[obj.id] = checks
    return session.input_checks[obj.id]


@dataclass
class Report:
    params: dict[str, Any]
    object: dict[str, Any]
    ball: dict[str, Any]
    fit: dict[str, Any]
    pockets: list[dict[str, float]]
    checks: list[dict[str, Any]]
    status: str
    washers: list[dict[str, Any]]
    timings: dict[str, float]
    underside: dict[str, Any] = field(default_factory=dict)   # 2D outlines for the pocket editor
    import_map: list[dict[str, Any]] = field(default_factory=list)   # figure filament -> project slot
    toolchange: dict[str, Any] = field(default_factory=dict)         # estimated colour changes (toolchange.py)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _steel_band(washer: Washer, diameter: float, relaxed: bool) -> tuple[float, float]:
    """Radii allowed for pocket centres: magnet fully on steel, or (relaxed) centre on steel."""
    return (washer.id / 2, washer.od / 2) if relaxed else washer.annulus(diameter)


def _pocket_map(session: Session, params: Params, s: float) -> PocketMap:
    return pocket_map(session.underside(params, s), params.spec)


def _place(session: Session, params: Params, cavity: Cavity, washer: Washer, s: float, relaxed: bool = False
           ) -> tuple[list[Pocket], list[Check], tuple[float, float]]:
    """`params.magnets` pockets at figure scale `s` that sit on the steel ring while the figure fits.

    The figure frame's origin is at offset o in the ball frame, so the washer
    (on the ball axis) is centred at -o in the figure frame. The fit margin is
    1-Lipschitz in o, so every offset within the best margin m* of the best
    offset o* still fits; o* and 36 such offsets are tried.
    """
    spec, n = params.spec, params.magnets
    pm = _pocket_map(session, params, s)
    o_star, m_star = best_offset(session.points(params), cavity, s)
    o0 = (float(o_star[0]), float(o_star[1]))
    if m_star < 0:
        return [], [Check("pockets:feet", "Underside", "fail", "Figure does not fit at this size.")], o0
    lo, hi = _steel_band(washer, spec.diameter, relaxed)
    cands = [o_star] + [o_star + m_star * f * np.array([np.cos(a), np.sin(a)])
                        for f in (0.33, 0.66, 0.95) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    first: list[Check] = []
    for o in cands:
        pockets, checks = choose_pockets(pm, spec, n, (-float(o[0]), -float(o[1]), lo, hi), allow_fewer=False)
        if len(pockets) == n:
            return pockets, checks, (float(o[0]), float(o[1]))
        first = first or checks
    return [], first, o0


def _offset_for_pockets(session: Session, params: Params, cavity: Cavity, washer: Washer, s: float,
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
    lo, hi = washer.annulus(params.pocket_diameter)
    r = np.linalg.norm(pxy[None, :, :] + grid[:, None, :], axis=2)
    score = np.min(np.minimum(r - lo, hi - r), axis=1)
    best = grid[int(np.argmax(score))]
    return float(best[0]), float(best[1])


def _largest_with_pockets(session: Session, params: Params, cavity: Cavity, washer: Washer, s_fit: float,
                          relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
    """Largest scale <= s_fit where the pockets fit the underside AND sit on the steel.

    Not monotone at the small end (walls are absolute mm), so scan down in 4 %
    steps from the fit limit, then bisect between the last failure and the first success.
    The scan stops at 60 % of the fit size: a washer that needs the figure
    smaller than that is the wrong washer, and the steel check says so.
    """
    best = None
    prev_fail = None
    s = s_fit
    while s >= MIN_SCALE_FOR_MAGNETS * s_fit:      # (no absolute floor: generated GLBs fit at ~4 %)
        pk, ch, off = _place(session, params, cavity, washer, s, relaxed)
        if pk:
            best = (s, pk, ch, off)
            break
        prev_fail, s = s, s * 0.96
    if best is not None and prev_fail is not None:
        a, b = best[0], prev_fail
        for _ in range(6):
            mid = (a + b) / 2
            pk, ch, off = _place(session, params, cavity, washer, mid, relaxed)
            if pk:
                best, a = (mid, pk, ch, off), mid
            else:
                b = mid
    return best


def _solve(session: Session, params: Params, cavity: Cavity, washer: Washer
           ) -> tuple[float, float, list[Pocket], list[Check], tuple[float, float]]:
    """(scale, max scale, pockets in the figure frame, pocket checks, XY offset)."""
    hp = session.points(params)
    spec, n = params.spec, params.magnets
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
                _offset_for_pockets(session, params, cavity, washer, scale, manual))

    # Shrinking the figure only shrinks the area that can hold a walled pocket
    # (walls are absolute mm). If the pockets do not fit even without the steel
    # ring at the size being asked for (or the largest that fits), skip the search
    # and show as many as do fit.
    probe = params.figure_scale or s_fit
    free_pk, free_ch = choose_pockets(_pocket_map(session, params, probe), spec, n)
    if not free_pk:
        return probe, s_fit, free_pk, free_ch, fit_offset(probe)
    if len(free_pk) < n:
        # Fewer fit than asked: place those like a full set, on the steel ring (a lone pocket placed freely sat
        # on a Pikachu's centre, over the washer's hole), keeping the "only room for k" warning.
        scale, top, pk, ch, off = _solve(session, replace(params, magnets=len(free_pk)), cavity, washer)
        return scale, top, pk, free_ch + [c for c in ch if c.id != "pockets:feet"], off

    def largest(relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
        key = (session._key(params), session.ball_scale(params), washer.key, params.clearance, spec, n, relaxed)
        if key not in session._auto:
            session._auto[key] = _largest_with_pockets(session, params, cavity, washer, s_fit, relaxed)
        return session._auto[key]

    if params.figure_scale is not None:
        sc = params.figure_scale
        for relaxed in (False, True):
            pk, ch, off = _place(session, params, cavity, washer, sc, relaxed)
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
                                 f"Reduced from {s_fit:.1%} to {scale:.1%} so the magnets sit on the washer's "
                                 f"steel; a larger washer may allow more (see the washer table).")]
            return scale, scale, pk, ch, off
    return s_fit, s_fit, free_pk, free_ch, fit_offset(s_fit)


def washer_table(session: Session, params: Params) -> list[dict[str, Any]]:
    """Max figure scale with each catalogue washer (magnets full / partial / off steel), best first.

    Washers are evaluated in parallel threads: the work is numpy/shapely/
    manifold3d, which release the GIL for most of it. The session's caches are
    warmed first so the threads only read them (each writes its own _auto key).
    """
    from concurrent.futures import ThreadPoolExecutor

    session.unit_parts(params)
    height = session.points(params).height

    def row(key: str, w: Washer) -> dict[str, Any]:
        cav = Cavity(session.ball, session.ball_scale(params), w, params.clearance)
        # The same figure shape for every washer (flat disc pinned to the current one's).
        spec = session.conform_spec(params)
        p = Params.from_dict({**asdict(params), "washer": key, "figure_scale": None, "pockets": None,
                              "flat_base": spec.flat_diameter if spec else params.flat_base})
        _, s_max, pockets, _, off = _solve(session, p, cav, w)
        steel = "off"
        if pockets and len(pockets) == p.magnets:
            pxy = np.array([[q.x, q.y] for q in pockets])
            if annulus_margin(pxy, 1.0, np.array(off), w.annulus(p.pocket_diameter)) >= -1e-6:
                steel = "full"
            elif annulus_margin(pxy, 1.0, np.array(off), (w.id / 2, w.od / 2)) >= -1e-6:
                steel = "partial"
        return {"key": key, "label": w.label, "max_scale": round(s_max, 4), "height_mm": round(s_max * height, 2),
                "pockets_on_steel": steel == "full", "steel": steel, "floor_z": round(cav.floor_z, 2)}

    if sys.platform == "emscripten":      # the browser build has no threads
        rows = [row(*kv) for kv in CATALOGUE.items()]
    else:
        with ThreadPoolExecutor(max_workers=len(CATALOGUE)) as pool:
            rows = list(pool.map(lambda kv: row(*kv), CATALOGUE.items()))
    rank = {"full": 0, "partial": 1, "off": 2}
    rows.sort(key=lambda r: (rank[r["steel"]], -float(r["max_scale"])))
    return rows


def analyse(session: Session, params: Params, recommend: bool = False) -> tuple[Report, list[Part]]:
    """Fit, pockets, checks. Returns the report and the final (cut) figure-frame parts."""
    t: dict[str, float] = {}
    t0 = time.time()
    washer = get_washer(params.washer)
    cavity = Cavity(session.ball, session.ball_scale(params), washer, params.clearance)
    hp = session.points(params)
    unit = session.figure_parts(params)
    obj = session.obj(params)
    checks: list[Check] = list(input_checks(session, params))
    t["input_checks"] = time.time() - t0

    t1 = time.time()
    scale, s_max, pockets, pchecks, off = _solve(session, params, cavity, washer)
    t["fit"] = time.time() - t1
    fit_dict: dict[str, Any] = {"max_scale": s_max, "scale": scale, "offset": off,
                                "height_mm": scale * hp.height, "max_height_mm": s_max * hp.height,
                                "unit_height": hp.height, "unit_extents": _extents(unit)}
    if scale <= 0:
        checks.append(Check("fit", "Fits in the ball", "fail",
                            "Does not fit even when very small — check the orientation (z up) and the file's units."))
        rep = Report(asdict(params), _obj_info(session, obj, params), _ball_info(cavity, washer), fit_dict, [],
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

    t2 = time.time()
    meshes = [p.mesh() for p in final]
    underside = _underside(session, params, meshes, scale, washer, off)
    if params.magnets:
        checks.extend(pchecks)
        if pockets:
            field_s = session.underside(params, scale)
            pocket_checks = validate_pockets(field_s, pockets, params.spec)  # pocket:N (+ pocket:gap)
            checks.extend(pocket_checks)
            lo, hi = washer.annulus(params.pocket_diameter)
            pxy = np.array([[p.x, p.y] for p in pockets])
            am = annulus_margin(pxy, 1.0, np.array(off), (lo, hi))
            am_centre = annulus_margin(pxy, 1.0, np.array(off), (washer.id / 2, washer.od / 2))
            if am >= -1e-6:
                checks.append(Check("steel", "Magnets on the washer", "pass",
                                    f"{'The pocket sits' if len(pockets) == 1 else 'All pockets sit'} on the "
                                    f"steel ring ({lo:.1f}–{hi:.1f} mm from the centre)."))
            elif am_centre >= -1e-6:
                checks.append(Check("steel", "Magnets on the washer", "warn",
                                    f"{'The pocket centre is' if len(pockets) == 1 else 'Pocket centres are'} on "
                                    "the steel but the magnets overhang its edge "
                                    f"by up to {-am:.1f} mm, so they hold with roughly half force. "
                                    "A larger washer avoids this (Compare washers).", round(am, 2)))
            else:
                od, idd = required_washer(pockets, params.spec)
                checks.append(Check("steel", "Magnets on the washer", "fail",
                                    f"Pockets miss the steel by {-am:.1f} mm with this washer; needs OD ≥ {od:.1f} mm "
                                    f"and hole ≤ {idd:.1f} mm.", round(am, 2)))
            # Per-pocket checks come first, in pocket order; a failed one is not cut.
            cuttable = [p for p, c in zip(pockets, pocket_checks) if c.status != "fail"]
            final, cut_checks = cut_pockets(final, cuttable, params.spec)
            checks.extend(cut_checks)
    t["pockets"] = time.time() - t2

    t3 = time.time()
    solid = printable_solid(final)
    st = session.settings(params) or {}
    layer = float(st.get("layer_height", 0.2) or 0.2)
    angle = float(st.get("support_threshold_angle", 30) or 30)
    ignore = [(p.x, p.y, params.pocket_diameter / 2, p.z + params.pocket_depth) for p in pockets] if params.magnets else []
    printing = [thickness_check(solid), overhang_check(solid, angle, ignore),
                islands_check(solid, 2 * layer), bed_contact_check(solid)]
    printing.append(supports_verdict(printing))
    checks.extend(printing)
    t["printability"] = time.time() - t3

    t4 = time.time()
    toolchange = toolchange_info(session, params, final, scale)
    t["toolchange"] = time.time() - t4

    washers = washer_table(session, params) if recommend else []
    t["total"] = time.time() - t0
    rep = Report(asdict(params), _obj_info(session, obj, params), _ball_info(cavity, washer), fit_dict,
                 [p.to_dict() for p in pockets], [c.to_dict() for c in checks], worst(checks), washers,
                 {k: round(v, 2) for k, v in t.items()}, underside,
                 [m.to_dict() for m in session.slot_matches(params)], toolchange)
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
    info = _tc_estimate(final, obj.extruder, colours, st)
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
        unit = session.unit_parts(params)
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
        said.append(f"legs/tail beyond the washer bent up by up to {info.lift_mm:.1f} mm"
                    f"{' (automatic)' if info.auto else ''}, so it fits {gain:+.1%} larger")
    if info.flat_depth_mm > 0:
        said.append(f"base pressed {info.flat_depth_mm:.1f} mm flat over the washer")
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


def _underside(session: Session, params: Params, meshes: list[trimesh.Trimesh], scale: float, washer: Washer,
               off: tuple[float, float]) -> dict[str, Any]:
    """Figure-frame 2D data for the pocket editor.

    feet: outline of what touches the bed; valid: outline of where a pocket
    centre satisfies every rule; height: the underside as an image (darker =
    lower, transparent = no material) over `extent` [x0, y0, x1, y1]; ring: the
    washer's steel ring as the ball holds it.
    """
    lo, hi = washer.annulus(params.pocket_diameter)
    feet = find_feet(meshes)
    f = session.underside(params, scale)
    out: dict[str, Any] = {"feet": _rings(unary_union(feet) if feet else None), "valid": [],
                           "ring": {"x": -off[0], "y": -off[1], "inner": lo, "outer": hi,
                                    "washer_od": washer.od, "washer_id": washer.id},
                           "pocket_d": params.pocket_diameter}
    out.update(_height_image(f))
    if params.magnets:
        out["valid"] = _mask_rings(pocket_map(f, params.spec).valid, f)
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


def _obj_info(session: Session, obj: FigureObject, params: Params) -> dict[str, Any]:
    return {"id": obj.id, "name": obj.name, "faces": obj.face_count,
            "parts": [{"name": p.name, "extruder": p.extruder or obj.extruder, "faces": len(p.faces),
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


def _ball_info(cavity: Cavity, washer: Washer) -> dict[str, Any]:
    z, r = cavity.profile(160)
    s = cavity.scale
    return {"scale": s, "r_inner": cavity.r_inner, "r_outer": cavity.ball.r_outer * s,
            "inner_diameter": 2 * cavity.r_inner, "native_r_inner": cavity.ball.r_inner,
            "floor_z": cavity.floor_z, "seat_z": cavity.seat_z, "top_z": cavity.top_z,
            "ring_span": cavity.ball.ring_span * s, "ring_h": [h * s for h in cavity.ball.ring_h],
            "ring_r": [x * s for x in cavity.ball.ring_r], "aperture": cavity.ball.aperture * s,
            "profile_z": z.tolist(), "profile_r": r.tolist(), "washer": washer.to_dict(),
            "annulus": list(washer.annulus(4.18)), "clearance": cavity.clearance}


# ------------------------------------------------------------------ outputs


def face_colours(part: Part, colours: list[str], default_extruder: int) -> np.ndarray:
    """Extruder index (1-based) per face: paint wins, else the part's, else the object's."""
    base = part.extruder or default_extruder
    ext = np.full(len(part.faces), base, np.int64)
    for i, attrs in part.tri_extra.items():
        if "paint_color=" in attrs:
            code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
            st = decode_majority(code)
            if st > 0:
                ext[i] = st
    return ext


def painted_mesh(part: Part, default_extruder: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, extruder per face) with split paint drawn as its sub-triangles, so the
    preview shows brush strokes and texture-derived colour edges as the slicer will print them."""
    ext = face_colours(part, [], default_extruder)
    split = [i for i, a in part.tri_extra.items() if "paint_color=" in a and not _whole_paint(a)]
    if not split:
        return part.vertices, part.faces, ext
    base = part.extruder or default_extruder
    keep = np.ones(len(part.faces), bool)
    keep[split] = False
    verts, faces, exts = [part.vertices], [part.faces[keep]], [ext[keep]]
    n = len(part.vertices)
    for i in split:
        code = part.tri_extra[i].split('paint_color="', 1)[1].split('"', 1)[0]
        corners = list(part.vertices[part.faces[i]])
        for tri, st in leaves(parse_tree(code), corners):
            verts.append(np.array(tri))
            faces.append(np.array([[n, n + 1, n + 2]]))
            exts.append(np.array([st or base]))
            n += 3
    return np.vstack(verts), np.vstack(faces), np.concatenate(exts)


def _whole_paint(attrs: str) -> bool:
    """A paint_color that is one state for the whole triangle (no split)."""
    code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
    return isinstance(parse_tree(code), int)


def clip_overlaps(parts: list[Part]) -> list[Part]:
    """What the slicer prints for overlapping parts: each part minus all earlier ones.

    Bambu/Prusa clip multi-part objects in list order — the 2nd part is clipped
    by the 1st, the 3rd by the 1st and 2nd — so an eye listed before the body
    keeps its colour even though the body mesh encloses it. Parts that are not
    valid solids are left as they are.
    """
    out: list[Part] = []
    done: list[tuple[Part, Any]] = []
    for part in parts:
        lo, hi = part.vertices.min(axis=0), part.vertices.max(axis=0)
        cur = part
        if part.subtype == "normal_part":
            try:
                man = to_manifold(merged(part))
                before = man.volume()
                for prev, pman in done:
                    plo, phi = prev.vertices.min(axis=0), prev.vertices.max(axis=0)
                    if np.all(phi >= lo) and np.all(plo <= hi):
                        man = man - pman
                if abs(man.volume() - before) > 1e-6:
                    cur = remesh_part(part, from_manifold(man))
                done.append((part, to_manifold(merged(part))))
            except RuntimeError:
                pass
        out.append(cur)
    return out


def preview_glb(session: Session, params: Params, regions: tuple[str, float] | None = None) -> bytes:
    """Figure at scale 1 in the figure frame, one mesh per (part, filament) for compact colours.

    With `regions` (colorize mode, angle), each part is coloured by the regions a
    tap would paint instead, so the user sees what the geometry detection found.
    """
    obj = session.obj(params)
    cols = session.colours(params) or ["#cccccc"]
    scene = trimesh.Scene()
    unit = session.unit_parts(params) if regions else session.figure_parts(params)
    # Overlap clipping re-meshes parts and loses the face order the region labels need: skip it there.
    shown = unit if regions else clip_overlaps(unit)
    for pi, part in enumerate(shown):
        if regions:
            # A brush has no fixed regions: show the parts it would paint within.
            mode = "part" if regions[0] == "brush" else regions[0]
            lab = session.topology(params).labels(pi, mode, regions[1])
            ext = lab + 1
            palette = region_colours(int(lab.max()) + 1 if len(lab) else 0)
            pv, pf = part.vertices, part.faces
        else:
            pv, pf, ext = painted_mesh(part, obj.extruder)
        for e in np.unique(ext):
            sub = trimesh.Trimesh(pv, pf[ext == e], process=False)
            sub.remove_unreferenced_vertices()
            if regions:
                rgba = [int(c) for c in palette[int(e) - 1]] + [255]
            else:
                hexc = cols[(int(e) - 1) % len(cols)].lstrip("#")[:6]
                rgba = [int(hexc[i:i + 2], 16) for i in (0, 2, 4)] + [255]
            sub.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.75))
            scene.add_geometry(sub, geom_name=f"p{pi}_e{int(e)}")
    out = scene.export(file_type="glb")
    assert isinstance(out, bytes)
    return out


def pick(session: Session, params: Params, point: list[float], normal: list[float] | None,
         mode: str = "part", angle: float = 15.0, radius: float = 0.0) -> dict[str, Any]:
    """The triangle under a tap on the 3D preview (figure frame, scale 1) and the size of its region.

    The answer's part/face go into a colorize paint operation, which then
    paints the same region on every later analysis and export.
    """
    PaintOp.from_dict({"part": 0, "face": 0, "filament": 0, "mode": mode, "angle": angle, "radius": radius})
    parts = session.unit_parts(params)
    pi, fi, dist = pick_face(parts, np.asarray(point, float), None if normal is None else np.asarray(normal, float))
    if pi < 0:
        raise ValueError("no triangle near that point")
    faces = region_faces(session.topology(params), pi, fi, mode, angle, radius)
    area = float(trimesh.Trimesh(parts[pi].vertices, parts[pi].faces[faces], process=False).area)
    return {"part": pi, "face": fi, "distance": dist, "faces": int(len(faces)),
            "share": round(len(faces) / max(1, len(parts[pi].faces)), 4), "area_mm2": round(area, 2)}


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


def export(session: Session, params: Params, out_dir: str | Path, ball_model: Path | None = None) -> dict[str, Any]:
    """Write the figure 3MF, its import variant, the washer-seat shell STL and a JSON report.

    `<stem>_pokefit_<pct>.3mf` opens as a project (File > Open). `..._import.3mf`
    is for dragging into an open project with "Import geometry only": its
    filament numbers are the user's project slots of the closest colour.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep, final = analyse(session, params)
    obj = session.obj(params)
    ball_s = session.ball_scale(params)
    # File tag: the ball's print percentage, or its inner diameter when that was given.
    pct = f"d{params.ball_diameter:.0f}mm" if params.ball_diameter else str(int(round(ball_s * 100)))
    stem = safe_stem(session.project.path.name)
    f3 = write_project(session.project, obj.name, obj.extruder, final, out_dir / f"{stem}_pokefit_{pct}.3mf",
                       settings=export_settings(session, params))
    shell_path = out_dir / f"shell_washerseat_{pct}_{safe_stem(params.washer)}.stl"
    shell_note = ""
    try:
        shell = _shell_mesh(ball_model or BALL_MODEL)
        fs = flat_spot(shell, ball_s, SHELL_DISK, get_washer(params.washer).od + 0.4)
        fs.mesh.export(shell_path)
    except (OSError, ValueError, RuntimeError) as exc:
        shell_path, shell_note = None, f"washer-seat shell not written: {exc}"  # type: ignore[assignment]
    # For "Import geometry only" into the user's own project: filaments renumbered to its slots.
    imp = None
    matches = session.slot_matches(params)
    if matches and session.reference:
        parts, ext = remap_parts(final, obj.extruder, matches)
        imp = write_project(session.project, obj.name, ext, parts, out_dir / f"{stem}_pokefit_{pct}_import.3mf",
                            settings=session.reference)
    rpath = out_dir / f"{stem}_pokefit_{pct}.json"
    rpath.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return {"figure_3mf": str(f3), "import_3mf": str(imp) if imp else None,
            "shell_stl": str(shell_path) if shell_path else None, "report": str(rpath),
            "note": shell_note, "status": rep.status, "import_map": [m.to_dict() for m in matches]}


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
    _, final = analyse(session, params)
    obj = session.obj(params)
    root = Path(work_root) if work_root else session.work_dir / f"slice_{int(time.time() * 1000)}"
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    stem = safe_stem(session.project.path.name)
    for name, sup in (("off", False), ("auto", True)):
        f = write_project(session.project, obj.name, obj.extruder, final, root / f"{name}.3mf",
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
            f = write_project(session.project, obj.name, obj.extruder, final, root / "x10.3mf",
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

