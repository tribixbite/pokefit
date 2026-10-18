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
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from shapely.ops import unary_union

from .ball import Ball, Cavity, load_ball
from .checks import (bed_contact_check, merged, islands_check, mesh_checks, overhang_check, printable_solid,
                     supports_verdict, thickness_check)
from .fit import HullPoints, annulus_margin, best_offset, check_fit, max_scale, prepare_points
from .magnets import (Pocket, PocketSpec, auto_pockets, choose_pockets, cut_pockets, feasible_region, find_feet,
                      required_washer, validate_pockets)
from .model3mf import (FigureObject, Part, Project, plate_size, read_project, remesh_part, world_parts,
                       write_project)
from .paint import decode_majority
from .report import Check, worst
from .shell import flat_spot, from_manifold, to_manifold
from .slicer import SliceResult, adapt_settings, default_slot, grid_positions, reference_settings, slice_3mf
from .washer import CATALOGUE, DEFAULT_WASHER, Washer, get_washer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BALL = ROOT / "pokeball.3mf"
# The user's own printer/process/filament settings: their printed-shells project
# if present, else the copy of its settings bundled with the package.
BUNDLED_REFERENCE = Path(__file__).resolve().parent / "profiles" / "x2d_petg_printed_ball_shells.json"
DEFAULT_REFERENCE = ROOT / "pokeball60goodPrinted.3mf" if (ROOT / "pokeball60goodPrinted.3mf").exists() else BUNDLED_REFERENCE
CACHE_DIR = ROOT / ".pokefit_cache"
SHELL_DISK = 8.0      # outer flat disk of the washer-seat shell, final mm


@dataclass
class Params:
    ball_scale: float = 0.6
    washer: str = DEFAULT_WASHER
    clearance: float = 1.0
    figure_scale: float | None = None            # None = largest that fits
    rotate_x: int = 0                            # degrees, multiples of 90
    rotate_y: int = 0
    magnets: bool = True
    pockets: list[dict[str, float]] | None = None  # None = automatic; figure frame, final mm
    pocket_diameter: float = 4.18
    pocket_depth: float = 2.20
    pocket_wall: float = 0.8
    pocket_ceiling: float = 0.6
    object_id: str | None = None
    slot: int | None = None                      # reference filament slot for slicing/export
    settings_source: str = "reference"           # "reference" (user's printer) or "original"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Params":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def spec(self) -> PocketSpec:
        return PocketSpec(self.pocket_diameter, self.pocket_depth, self.pocket_wall, self.pocket_ceiling)


@dataclass
class Session:
    id: str
    project: Project
    work_dir: Path
    ball: Ball
    reference: dict | None
    input_checks: dict[str, list[Check]] = field(default_factory=dict)   # object id -> checks
    _frames: dict[tuple[str, int, int], list[Part]] = field(default_factory=dict)
    _points: dict[tuple[str, int, int], HullPoints] = field(default_factory=dict)
    # Auto-placement results per configuration (the figure-scale-independent search).
    _auto: dict[tuple, Any] = field(default_factory=dict)

    def obj(self, params: Params) -> FigureObject:
        return self.project.object(params.object_id) if params.object_id else self.project.default_object()

    def unit_parts(self, params: Params) -> list[Part]:
        """Figure-frame parts at scale 1 (rotated, XY-centred, on z = 0)."""
        obj = self.obj(params)
        key = (obj.id, params.rotate_x % 360, params.rotate_y % 360)
        if key not in self._frames:
            parts = world_parts(obj)
            rot = (trimesh.transformations.rotation_matrix(np.radians(key[2]), [0, 1, 0]) @
                   trimesh.transformations.rotation_matrix(np.radians(key[1]), [1, 0, 0]))
            parts = [p.transformed(rot) for p in parts]
            v = np.vstack([p.vertices for p in parts])
            mn, mx = v.min(axis=0), v.max(axis=0)
            shift = trimesh.transformations.translation_matrix([-(mn[0] + mx[0]) / 2, -(mn[1] + mx[1]) / 2, -mn[2]])
            self._frames[key] = [p.transformed(shift) for p in parts]
            self._points[key] = prepare_points(np.vstack([p.vertices for p in self._frames[key]]))
        return self._frames[key]

    def points(self, params: Params) -> HullPoints:
        self.unit_parts(params)
        obj = self.obj(params)
        return self._points[(obj.id, params.rotate_x % 360, params.rotate_y % 360)]

    def slot(self, params: Params) -> int:
        if params.slot is not None:
            return params.slot
        return default_slot(self.reference) if self.reference else 0


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _steel_band(washer: Washer, diameter: float, relaxed: bool) -> tuple[float, float]:
    """Radii allowed for pocket centres: magnet fully on steel, or (relaxed) centre on steel."""
    return (washer.id / 2, washer.od / 2) if relaxed else washer.annulus(diameter)


def _place(session: Session, params: Params, cavity: Cavity, washer: Washer, s: float, relaxed: bool = False
           ) -> tuple[list[Pocket], list[Check], tuple[float, float]]:
    """Pockets at figure scale `s` that sit on the steel ring while the figure fits.

    The figure frame's origin is at offset o in the ball frame, so the washer
    (on the ball axis) is centred at -o in the figure frame. The fit margin is
    1-Lipschitz in o, so every offset within the best margin m* of the best
    offset o* still fits; o* and 36 such offsets are tried.
    """
    spec = params.spec
    meshes = [p.mesh() for p in scaled(session.unit_parts(params), s)]
    feet, region = find_feet(meshes), feasible_region(meshes, spec)
    o_star, m_star = best_offset(session.points(params), cavity, s)
    o0 = (float(o_star[0]), float(o_star[1]))
    if m_star < 0:
        return [], [Check("pockets:feet", "Feet", "fail", "Figure does not fit at this size.")], o0
    lo, hi = _steel_band(washer, spec.diameter, relaxed)
    cands = [o_star] + [o_star + m_star * f * np.array([np.cos(a), np.sin(a)])
                        for f in (0.33, 0.66, 0.95) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    first: list[Check] = []
    for o in cands:
        pockets, checks = choose_pockets(feet, region, spec, (-float(o[0]), -float(o[1]), lo, hi),
                                         allow_single=False)
        if len(pockets) == 2:
            return pockets, checks, (float(o[0]), float(o[1]))
        first = first or checks
    return [], first, o0


def _largest_with_pockets(session: Session, params: Params, cavity: Cavity, washer: Washer, s_fit: float,
                          relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
    """Largest scale <= s_fit where two pockets fit the feet AND sit on the steel.

    Not monotone at the small end (walls are absolute mm), so scan down in 4 %
    steps from the fit limit, then bisect between the last failure and the first success.
    """
    best = None
    prev_fail = None
    s = s_fit
    while s >= max(0.05, 0.25 * s_fit):
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
    spec = params.spec
    annulus = washer.annulus(spec.diameter)
    free = max_scale(hp, cavity)
    if free is None:
        return 0.0, 0.0, [], [], (0.0, 0.0)
    s_fit = free.max_scale

    def fit_offset(s: float) -> tuple[float, float]:
        o, _ = best_offset(hp, cavity, s)
        return float(o[0]), float(o[1])

    if not params.magnets:
        scale = params.figure_scale or s_fit
        return scale, s_fit, [], [], fit_offset(scale)

    if params.pockets is not None:
        # Manual pockets: figure-frame mm as placed by the user.
        scale = params.figure_scale or s_fit
        manual = [Pocket(float(p["x"]), float(p["y"])) for p in params.pockets]
        if not manual:
            return scale, s_fit, [], [], fit_offset(scale)
        o, _ = best_offset(hp, cavity, scale, np.array([[p.x, p.y] for p in manual]) / scale, annulus)
        return scale, s_fit, manual, [Check("pockets:feet", "Feet", "info", "Pockets placed by hand.")],             (float(o[0]), float(o[1]))

    def largest(relaxed: bool) -> tuple[float, list[Pocket], list[Check], tuple[float, float]] | None:
        key = (session.obj(params).id, params.rotate_x % 360, params.rotate_y % 360, params.ball_scale,
               washer.key, params.clearance, spec, relaxed)
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
        meshes = [p.mesh() for p in scaled(session.unit_parts(params), sc)]
        pk1, ch1 = auto_pockets(meshes, spec)
        return sc, s_fit, pk1, ch1, fit_offset(sc)

    for relaxed in (False, True):
        best = largest(relaxed)
        if best is not None:
            scale, pk, ch, off = best
            if scale < s_fit * 0.999:
                ch = ch + [Check("pockets:scale", "Size vs. magnets", "info",
                                 f"Reduced from {s_fit:.1%} to {scale:.1%} so both magnets sit on the washer's "
                                 f"steel; a larger washer may allow more (see the washer table).")]
            return scale, scale, pk, ch, off
    meshes = [p.mesh() for p in scaled(session.unit_parts(params), s_fit)]
    pk1, ch1 = auto_pockets(meshes, spec)
    return s_fit, s_fit, pk1, ch1, fit_offset(s_fit)


def washer_table(session: Session, params: Params) -> list[dict[str, Any]]:
    """Max figure scale with each catalogue washer (pockets on steel), best first."""
    rows: list[dict[str, Any]] = []
    for key, w in CATALOGUE.items():
        cav = Cavity(session.ball, params.ball_scale, w, params.clearance)
        p = Params.from_dict({**asdict(params), "washer": key, "figure_scale": None, "pockets": None})
        _, s_max, pockets, _, off = _solve(session, p, cav, w)
        steel = "off"
        if len(pockets) == 2:
            pxy = np.array([[q.x, q.y] for q in pockets])
            if annulus_margin(pxy, 1.0, np.array(off), w.annulus(p.pocket_diameter)) >= -1e-6:
                steel = "full"
            elif annulus_margin(pxy, 1.0, np.array(off), (w.id / 2, w.od / 2)) >= -1e-6:
                steel = "partial"
        rows.append({"key": key, "label": w.label, "max_scale": round(s_max, 4),
                     "height_mm": round(s_max * session.points(params).height, 2),
                     "pockets_on_steel": steel == "full", "steel": steel, "floor_z": round(cav.floor_z, 2)})
    rank = {"full": 0, "partial": 1, "off": 2}
    rows.sort(key=lambda r: (rank[r["steel"]], -float(r["max_scale"])))
    return rows


def analyse(session: Session, params: Params, recommend: bool = False) -> tuple[Report, list[Part]]:
    """Fit, pockets, checks. Returns the report and the final (cut) figure-frame parts."""
    t: dict[str, float] = {}
    t0 = time.time()
    washer = get_washer(params.washer)
    cavity = Cavity(session.ball, params.ball_scale, washer, params.clearance)
    hp = session.points(params)
    unit = session.unit_parts(params)
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
                            "Does not fit even at 5 % — check the orientation (z up) and the file's units."))
        rep = Report(asdict(params), _obj_info(session, obj), _ball_info(cavity, washer), fit_dict, [],
                     [c.to_dict() for c in checks], "fail", [], t)
        return rep, unit

    final = scaled(unit, scale)
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

    t2 = time.time()
    meshes = [p.mesh() for p in final]
    underside = _underside(meshes, params, washer, off)
    if params.magnets:
        checks.extend(pchecks)
        if pockets:
            checks.extend(validate_pockets(meshes, pockets, params.spec))
            lo, hi = washer.annulus(params.pocket_diameter)
            pxy = np.array([[p.x, p.y] for p in pockets])
            am = annulus_margin(pxy, 1.0, np.array(off), (lo, hi))
            am_centre = annulus_margin(pxy, 1.0, np.array(off), (washer.id / 2, washer.od / 2))
            if am >= -1e-6:
                checks.append(Check("steel", "Magnets on the washer", "pass",
                                    f"Both pockets sit on the steel ring ({lo:.1f}–{hi:.1f} mm from the centre)."))
            elif am_centre >= -1e-6:
                checks.append(Check("steel", "Magnets on the washer", "warn",
                                    "Pocket centres are on the steel but the magnets overhang its edge "
                                    f"by up to {-am:.1f} mm, so they hold with roughly half force. "
                                    "A larger washer avoids this (Compare washers).", round(am, 2)))
            else:
                od, idd = required_washer(pockets, params.spec)
                checks.append(Check("steel", "Magnets on the washer", "fail",
                                    f"Pockets miss the steel by {-am:.1f} mm with this washer; needs OD ≥ {od:.1f} mm "
                                    f"and hole ≤ {idd:.1f} mm.", round(am, 2)))
            final, cut_checks = cut_pockets(final, pockets, params.spec)
            checks.extend(cut_checks)
    t["pockets"] = time.time() - t2

    t3 = time.time()
    solid = printable_solid(final)
    layer = float((session.reference or {}).get("layer_height", 0.2) or 0.2)
    angle = float((session.reference or {}).get("support_threshold_angle", 30) or 30)
    ignore = [(p.x, p.y, params.pocket_diameter / 2) for p in pockets] if params.magnets else []
    printing = [thickness_check(solid), overhang_check(solid, angle, ignore, params.pocket_depth),
                islands_check(solid, 2 * layer), bed_contact_check(solid)]
    printing.append(supports_verdict(printing))
    checks.extend(printing)
    t["printability"] = time.time() - t3

    washers = washer_table(session, params) if recommend else []
    t["total"] = time.time() - t0
    rep = Report(asdict(params), _obj_info(session, obj), _ball_info(cavity, washer), fit_dict,
                 [p.to_dict() for p in pockets], [c.to_dict() for c in checks], worst(checks), washers,
                 {k: round(v, 2) for k, v in t.items()}, underside)
    return rep, final


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


def _underside(meshes: list[trimesh.Trimesh], params: Params, washer: Washer,
               off: tuple[float, float]) -> dict[str, Any]:
    """Figure-frame 2D data: foot outlines, where pocket centres are valid, and the steel ring."""
    lo, hi = washer.annulus(params.pocket_diameter)
    feet = find_feet(meshes)
    return {"feet": _rings(unary_union(feet) if feet else None),
            "valid": _rings(feasible_region(meshes, params.spec)) if params.magnets else [],
            "ring": {"x": -off[0], "y": -off[1], "inner": lo, "outer": hi,
                     "washer_od": washer.od, "washer_id": washer.id},
            "pocket_d": params.pocket_diameter}


def _kind_text(kind: str) -> str:
    return {"ring": "connector ring", "shell": "shell wall", "top": "top of the ball",
            "aperture": "ring opening (it must pass through when closing)"}.get(kind, kind)


def _extents(parts: list[Part]) -> list[float]:
    v = np.vstack([p.vertices for p in parts])
    return np.ptp(v, axis=0).round(3).tolist()


def _obj_info(session: Session, obj: FigureObject) -> dict[str, Any]:
    return {"id": obj.id, "name": obj.name, "faces": obj.face_count,
            "parts": [{"name": p.name, "extruder": p.extruder or obj.extruder, "faces": len(p.faces),
                       "painted": len(p.tri_extra)} for p in obj.parts],
            "colours": session.project.colours,
            "objects": [{"id": o.id, "name": o.name, "faces": o.face_count} for o in session.project.objects]}


def _ball_info(cavity: Cavity, washer: Washer) -> dict[str, Any]:
    z, r = cavity.profile(160)
    s = cavity.scale
    return {"scale": s, "r_inner": cavity.r_inner, "r_outer": cavity.ball.r_outer * s,
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


def preview_glb(session: Session, params: Params) -> bytes:
    """Figure at scale 1 in the figure frame, one mesh per (part, filament) for compact colours."""
    obj = session.obj(params)
    cols = session.project.colours or ["#cccccc"]
    scene = trimesh.Scene()
    for pi, part in enumerate(clip_overlaps(session.unit_parts(params))):
        ext = face_colours(part, cols, obj.extruder)
        for e in np.unique(ext):
            sub = trimesh.Trimesh(part.vertices, part.faces[ext == e], process=False)
            sub.remove_unreferenced_vertices()
            hexc = cols[(int(e) - 1) % len(cols)].lstrip("#")[:6]
            rgba = [int(hexc[i:i + 2], 16) for i in (0, 2, 4)] + [255]
            sub.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.75))
            scene.add_geometry(sub, geom_name=f"p{pi}_e{int(e)}")
    out = scene.export(file_type="glb")
    assert isinstance(out, bytes)
    return out


def export_settings(session: Session, params: Params, supports: bool = False) -> dict | None:
    if params.settings_source == "reference" and session.reference:
        return adapt_settings(session.reference, session.project.colours, session.slot(params), supports)
    return session.project.settings


def safe_stem(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(name).stem)
    return keep.strip("_") or "figure"


def export(session: Session, params: Params, out_dir: str | Path) -> dict[str, Any]:
    """Write the figure 3MF, the washer-seat shell STL and a JSON report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep, final = analyse(session, params)
    obj = session.obj(params)
    pct = int(round(params.ball_scale * 100))
    stem = safe_stem(session.project.path.name)
    f3 = write_project(session.project, obj.name, obj.extruder, final, out_dir / f"{stem}_pokefit_{pct}.3mf",
                       settings=export_settings(session, params))
    shell_path = out_dir / f"shell_washerseat_{pct}_{safe_stem(params.washer)}.stl"
    shell_note = ""
    try:
        shell = _shell_mesh()
        fs = flat_spot(shell, params.ball_scale, SHELL_DISK, get_washer(params.washer).od + 0.4)
        fs.mesh.export(shell_path)
    except (OSError, ValueError, RuntimeError) as exc:
        shell_path, shell_note = None, f"washer-seat shell not written: {exc}"  # type: ignore[assignment]
    rpath = out_dir / f"{stem}_pokefit_{pct}.json"
    rpath.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return {"figure_3mf": str(f3), "shell_stl": str(shell_path) if shell_path else None, "report": str(rpath),
            "note": shell_note, "status": rep.status}


def _shell_mesh() -> trimesh.Trimesh:
    proj = read_project(DEFAULT_BALL)
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
    for name, sup in (("off", False), ("auto", True)):
        f = write_project(session.project, obj.name, obj.extruder, final, root / f"{name}.3mf",
                          settings=_slice_settings(session, params, sup))
        out[name] = slice_3mf(f, work_dir=root / name).to_dict()
    needs = out["auto"].get("support_seconds", 0) > 0
    out["supports_needed"] = needs if out["auto"].get("ok") else None
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
    if session.reference:
        return adapt_settings(session.reference, session.project.colours, session.slot(params), supports)
    st = dict(session.project.settings or {})
    if st:
        st["enable_support"] = "1" if supports else "0"
    return st or None

