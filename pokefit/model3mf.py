"""Read and write 3MF projects (Bambu Studio flavour); also read glTF/GLB scenes and plain STL/OBJ meshes.

Reading keeps what a print-ready output needs: per-object parts (Bambu
"volumes") with names, extruder assignment and every extra per-triangle
attribute verbatim (`paint_color`, `paint_supports`, `paint_seam`, ...).
Writing produces a single-object Bambu project whose mesh vertices are in final
print coordinates (all transforms baked in), with the source project's
settings, filament config and auxiliaries carried over.

3MF layout reminder (core 2015/02 + production 2015/06 extension):
  3D/3dmodel.model           build items (object id + 3x4 transform) and
                             top-level objects whose <components> point at
  3D/Objects/object_N.model  meshes (p:path + objectid + transform)
  Metadata/model_settings.config   Bambu: names, extruders, plates
  Metadata/project_settings.config Bambu: full print/printer/filament config
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import quoteattr

import numpy as np
import trimesh
from lxml import etree

from .paint import encode_tree, majority_tree, parse_tree, remap_code

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS = {"m": NS_CORE, "p": NS_PROD}
P_PATH = f"{{{NS_PROD}}}path"

DEFAULT_COLOURS = ["#E05028", "#161616", "#FFFFFF", "#2850E0", "#0ACC38", "#FFF144", "#A03CF7", "#F98C36"]

# Fast path for Bambu/Prusa-written meshes: fixed attribute order.
_VERTEX_RE = re.compile(rb'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"\s*/>')
_TRI_RE = re.compile(rb'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"([^>]*?)/>')
_OBJ_RE = re.compile(rb'<object\s[^>]*?\bid="(\d+)"[^>]*>(.*?)</object>', re.S)


@dataclass
class Part:
    """One mesh volume of an object, in object coordinates."""

    name: str
    extruder: int                       # 1-based; 0 = inherit the object's
    vertices: np.ndarray                # (n, 3) float64
    faces: np.ndarray                   # (m, 3) int64
    tri_extra: dict[int, str] = field(default_factory=dict)  # face index -> raw extra attrs
    subtype: str = "normal_part"

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(self.vertices, self.faces, process=False)

    def transformed(self, m: np.ndarray) -> "Part":
        """Copy with a 4x4 affine applied; mirrored transforms flip the winding back
        (reversing each triangle's corners, so split paint is re-laid to match)."""
        v = self.vertices @ m[:3, :3].T + m[:3, 3]
        if np.linalg.det(m[:3, :3]) < 0:
            extra = {i: remap_paint_attrs(a, (2, 1, 0)) for i, a in self.tri_extra.items()}
            return Part(self.name, self.extruder, v, self.faces[:, ::-1].copy(), extra, self.subtype)
        return Part(self.name, self.extruder, v, self.faces.copy(), dict(self.tri_extra), self.subtype)


_PAINT_ATTR_RE = re.compile(r'(paint_\w+)="([0-9A-Fa-f]+)"')


@lru_cache(maxsize=65536)
def _remap_cached(code: str, perm: tuple[int, int, int]) -> str:
    return remap_code(code, perm)


@lru_cache(maxsize=1 << 16)
def remap_paint_attrs(attrs: str, perm: tuple[int, int, int]) -> str:
    """A triangle's extra attributes for its corners listed in a new order (new j = old perm[j]):
    split paint (colour, supports, seams) is re-laid on the same areas."""
    if perm == (0, 1, 2) or "paint_" not in attrs:
        return attrs
    return _PAINT_ATTR_RE.sub(lambda mm: f'{mm.group(1)}="{_remap_cached(mm.group(2).upper(), perm)}"', attrs)


@lru_cache(maxsize=1 << 16)
def whole_paint_attrs(attrs: str) -> str:
    """Attributes with every split paint reduced to its main state, for a triangle whose
    shape is new (its splits would land on the wrong areas)."""
    if "paint_" not in attrs:
        return attrs
    return _PAINT_ATTR_RE.sub(
        lambda mm: f'{mm.group(1)}="{encode_tree(majority_tree(parse_tree(mm.group(2).upper())))}"', attrs)


@dataclass
class FigureObject:
    id: str
    name: str
    extruder: int
    _parts: list[Part] | None           # None until `loader` has run
    transform: np.ndarray               # build item transform (4x4), first instance
    loader: Callable[[], list[Part]] | None = None
    n_faces: int | None = None          # known without parsing (counted in the XML)
    simplify: float | None = None       # triangle budget (simplify.Budget) the parts are cut to when loaded
    simplified: Any = None              # simplify.SimplifyInfo once that happened (None: nothing to cut)
    repaired: Any = None                # repair.RepairInfo when loading fixed the mesh (None: nothing to fix)
    # Magnet pockets of an earlier pokefit export, taken out as it loaded (oldpockets.OldPocket, plate frame), and
    # how many of its "flat base" parts were dropped with them.
    old_pockets: list[Any] = field(default_factory=list)
    old_bases: int = 0
    as_written: bool = False            # True: the parts exactly as in the file (no pocket strip, no repair: tools)

    @property
    def parts(self) -> list[Part]:
        """Meshes are parsed on first use: a file of 14 figures only parses the one being fitted."""
        if self._parts is None:
            assert self.loader is not None
            parts = self.loader()
            if self.simplify:
                from .simplify import simplify_parts

                cut, info = simplify_parts(parts, self.simplify)
                if info.faces_after < info.faces_before:
                    parts, self.simplified = cut, info
            if self.as_written:
                self._parts, self.loader = parts, None
                return parts
            # An earlier export's pockets and flat base out (the repair below closes the holes), so they can be
            # placed again.
            from .oldpockets import strip_pockets

            parts, self.old_pockets, self.old_bases = strip_pockets(parts, self.transform)
            # Like Bambu Studio's "Fix model": every edge joins exactly two faces before anything is cut or painted.
            from .repair import repair_parts

            parts, fixed = repair_parts(parts)
            self.repaired = fixed or None
            self._parts, self.loader = parts, None
        return self._parts

    @property
    def face_count(self) -> int:
        if self.n_faces is None:
            self.n_faces = sum(len(p.faces) for p in self.parts)
        return self.n_faces


@dataclass
class Project:
    path: Path
    kind: str                           # "3mf" or "mesh"
    objects: list[FigureObject]
    colours: list[str]                  # filament colours, index 0 = extruder 1
    settings: dict | None               # Bambu project_settings.config (parsed JSON)
    texture: Any = None                 # gltf.TextureColours: palettes of other sizes for a glTF

    def object(self, oid: str) -> FigureObject:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(f"object {oid!r} not in {self.path.name}: {[o.id for o in self.objects]}")

    def default_object(self) -> FigureObject:
        """The largest object by triangle count (the figure, not a stand or an egg)."""
        return max(self.objects, key=lambda o: o.face_count)


def parse_transform(s: str | None) -> np.ndarray:
    """3MF 3x4 row-major affine ("m00 m01 m02 m10 ... m32") -> 4x4 matrix."""
    m = np.eye(4)
    if s:
        v = np.array([float(x) for x in s.split()]).reshape(4, 3)
        m[:3, :3] = v[:3].T
        m[:3, 3] = v[3]
    return m


def format_transform(m: np.ndarray) -> str:
    v = np.vstack([m[:3, :3].T, m[:3, 3]])
    return " ".join(f"{x:.9g}" for x in v.ravel())


def _parse_mesh_fast(body: bytes) -> tuple[np.ndarray, np.ndarray, dict[int, str]] | None:
    """Regex parse of one <object> body; None when the layout is not the fast-path one."""
    # "<triangle" also prefixes the "<triangles>" container tag.
    nv, nt = body.count(b"<vertex"), body.count(b"<triangle") - body.count(b"<triangles")
    verts = _VERTEX_RE.findall(body)
    tris = _TRI_RE.findall(body)
    if len(verts) != nv or len(tris) != nt:
        return None
    v = np.array(verts, dtype=np.float64).reshape(-1, 3) if verts else np.zeros((0, 3))
    f = np.array([t[:3] for t in tris], dtype=np.int64).reshape(-1, 3)
    extra = {i: t[3].decode().strip() for i, t in enumerate(tris) if t[3].strip()}
    return v, f, extra


def _parse_mesh_xml(obj: etree._Element) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Generic (slow) parse for 3MF writers with other attribute orders."""
    v = np.array([[float(e.get(a)) for a in "xyz"] for e in obj.iterfind(".//m:vertex", NS)]).reshape(-1, 3)
    f_list, extra = [], {}
    for i, e in enumerate(obj.iterfind(".//m:triangle", NS)):
        f_list.append([int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))])
        rest = {k: val for k, val in e.attrib.items() if k not in ("v1", "v2", "v3")}
        if rest:
            extra[i] = " ".join(f"{k}={quoteattr(val)}" for k, val in rest.items())
    return v, np.array(f_list, dtype=np.int64).reshape(-1, 3), extra


class _ModelFile:
    """Lazy access to the meshes of one .model part file."""

    def __init__(self, data: bytes):
        self.data = data
        self._bodies: dict[str, bytes] | None = None
        self._tree: etree._Element | None = None

    def _body(self, oid: str) -> bytes | None:
        if self._bodies is None:
            self._bodies = {m.group(1).decode(): m.group(2) for m in _OBJ_RE.finditer(self.data)}
        return self._bodies.get(oid)

    def face_count(self, oid: str) -> int:
        if self._bodies is None and self.data.count(b"<object") == 1 and b"<components" not in self.data:
            # The usual Bambu layout (one mesh per file): count without splitting into objects.
            return self.data.count(b"<triangle") - self.data.count(b"<triangles")
        body = self._body(oid)
        if body is None:
            return len(self.mesh(oid)[1])
        return body.count(b"<triangle") - body.count(b"<triangles")

    def mesh(self, oid: str) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
        body = self._body(oid)
        if body is not None:
            fast = _parse_mesh_fast(body)
            if fast is not None:
                return fast
        if self._tree is None:
            self._tree = etree.fromstring(self.data, parser=etree.XMLParser(huge_tree=True))
        obj = self._tree.find(f".//m:object[@id='{oid}']", NS)
        if obj is None:
            raise KeyError(f"mesh object {oid} not found")
        return _parse_mesh_xml(obj)

    def components(self, oid: str) -> list[etree._Element]:
        if b"<components" not in self.data:
            return []
        body = self._body(oid)
        if body is not None and b"<components" not in body:
            return []                  # a mesh object: no need to build the (huge) XML tree
        if self._tree is None:
            self._tree = etree.fromstring(self.data, parser=etree.XMLParser(huge_tree=True))
        obj = self._tree.find(f".//m:object[@id='{oid}']", NS)
        return [] if obj is None else list(obj.iterfind("m:components/m:component", NS))


def _model_settings(z: zipfile.ZipFile) -> dict[str, dict]:
    """object id -> {name, extruder, parts: {part id -> {name, extruder, subtype}}}."""
    try:
        root = etree.fromstring(z.read("Metadata/model_settings.config"))
    except KeyError:
        return {}
    out: dict[str, dict] = {}
    for obj in root.iterfind("object"):
        meta = {m.get("key"): m.get("value") for m in obj.iterfind("metadata") if m.get("key")}
        parts = {}
        for p in obj.iterfind("part"):
            pm = {m.get("key"): m.get("value") for m in p.iterfind("metadata") if m.get("key")}
            parts[p.get("id")] = {"name": pm.get("name", ""), "extruder": int(pm.get("extruder", 0) or 0),
                                  "subtype": p.get("subtype", "normal_part")}
        out[obj.get("id")] = {"name": meta.get("name", ""), "extruder": int(meta.get("extruder", 1) or 1),
                              "parts": parts}
    return out


def read_project(path: str | Path, simplify: float | None = None, as_written: bool = False) -> Project:
    """Load a 3MF project, a glTF/GLB scene (one part, its colours as paint; see gltf.py)
    or an STL/OBJ mesh (one object, one part). `simplify`: cut each figure to about this many triangles
    when its meshes load (simplify.py; None = as in the file). `as_written`: a 3MF's parts exactly as stored (an
    export's pockets and flat base kept, no repair), for tools that check or render a file."""
    path = Path(path)
    if path.suffix.lower() in (".glb", ".gltf"):
        from .gltf import read_gltf                 # imports this module: load it on use

        return read_gltf(path, simplify)
    if path.suffix.lower() != ".3mf":
        m = trimesh.load(path, force="mesh", process=False)
        assert isinstance(m, trimesh.Trimesh)
        m.merge_vertices()             # STL stores every triangle's corners separately
        part = Part(path.stem, 0, np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
        obj = FigureObject("1", path.stem, 1, None, np.eye(4), lambda: [part], len(part.faces), simplify)
        return Project(path, "mesh", [obj], DEFAULT_COLOURS[:1], None)

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        root_data = z.read("3D/3dmodel.model")
        settings_meta = _model_settings(z)
        settings = None
        if "Metadata/project_settings.config" in names:
            try:
                settings = json.loads(z.read("Metadata/project_settings.config"))
            except json.JSONDecodeError:
                settings = None
        root = etree.fromstring(root_data, parser=etree.XMLParser(huge_tree=True))
        files: dict[str, _ModelFile] = {"3D/3dmodel.model": _ModelFile(root_data)}

        # Pass 1 (every object): walk the component tree to its mesh leaves and
        # count their triangles. Meshes themselves are parsed later, per object.
        Leaf = tuple[str, str, np.ndarray]                 # (model file, mesh object id, transform)

        def leaves(mf_path: str, oid: str, xf: np.ndarray, out: list[Leaf]) -> None:
            mf_path = mf_path.lstrip("/")
            if mf_path not in files:
                if mf_path not in names:
                    raise ValueError(f"{path.name}: component file {mf_path} missing")
                files[mf_path] = _ModelFile(z.read(mf_path))
            comps = files[mf_path].components(oid)
            if not comps:
                out.append((mf_path, oid, xf))
                return
            for c in comps:
                leaves(c.get(P_PATH) or mf_path, c.get("objectid"), xf @ parse_transform(c.get("transform")), out)

        build: dict[str, np.ndarray] = {}
        for item in root.iterfind("m:build/m:item", NS):
            build.setdefault(item.get("objectid"), parse_transform(item.get("transform")))

        found: list[tuple[str, str, int, dict, list[Leaf], int]] = []
        for obj in root.iterfind("m:resources/m:object", NS):
            oid = obj.get("id")
            if oid not in build or obj.get("type", "model") != "model":
                continue
            meta = settings_meta.get(oid, {})
            lv: list[Leaf] = []
            leaves("3D/3dmodel.model", oid, np.eye(4), lv)
            n = sum(files[f].face_count(i) for f, i, _ in lv)
            found.append((oid, meta.get("name") or obj.get("name") or f"object_{oid}", meta.get("extruder", 1),
                          meta.get("parts", {}), lv, n))
        files.clear()                  # release the XML; loaders re-read their files

    def loader(lv: list[Leaf], part_meta: dict) -> Callable[[], list[Part]]:
        def load() -> list[Part]:
            out: list[Part] = []
            cache: dict[str, _ModelFile] = {}
            with zipfile.ZipFile(path) as zz:
                for f, i, xf in lv:
                    if f not in cache:
                        cache[f] = _ModelFile(zz.read(f))
                    v, fc, extra = cache[f].mesh(i)
                    m = part_meta.get(i, {})
                    out.append(Part(m.get("name") or f"part_{i}", m.get("extruder", 0), v, fc, extra,
                                    m.get("subtype", "normal_part")).transformed(xf))
            return out
        return load

    objects = [FigureObject(oid, name, ext, None, build[oid], loader(lv, pm), n, simplify, as_written=as_written)
               for oid, name, ext, pm, lv, n in found]

    if not objects:
        raise ValueError(f"{path.name}: no printable objects")
    colours = list(settings.get("filament_colour", [])) if settings else []
    if not colours:
        # From the metadata, so no mesh has to be parsed for it.
        n = max([1] + [ext for _, _, ext, _, _, _ in found] +
                [m.get("extruder", 0) for _, _, _, pm, _, _ in found for m in pm.values()])
        colours = (DEFAULT_COLOURS * 4)[:n]
    return Project(path, "3mf", objects, colours, settings)


def world_parts(obj: FigureObject) -> list[Part]:
    """The object's parts with its build transform applied (plate coordinates)."""
    return [p.transformed(obj.transform) for p in obj.parts]


def row_ids(rows: np.ndarray) -> np.ndarray:
    """An id per row of an (n, k) integer array, equal ids for equal rows (`np.unique(axis=0)`'s inverse,
    in some other order).

    `np.unique(axis=0)` sorts the rows as opaque byte strings, about 1 s per million rows. Here the
    columns are packed into one int64 key, so a single 1-D unique does it (3-10x faster): as they
    are when their value ranges fit in 63 bits together (vertex ids), else as per-column ranks.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] == 0 or len(rows) == 0:
        return np.zeros(len(rows), np.int64)
    lo = rows.min(axis=0).astype(np.int64)
    span = rows.max(axis=0).astype(np.int64) - lo
    cols: list[np.ndarray] = []
    bits = [max(1, int(s).bit_length()) for s in span.tolist()]
    if sum(bits) <= 63:
        cols = [rows[:, j].astype(np.int64) - lo[j] for j in range(rows.shape[1])]
    else:
        ranks = [np.unique(col, return_inverse=True) for col in rows.T]
        bits = [max(1, int(len(u) - 1).bit_length()) for u, _ in ranks]
        if sum(bits) > 63:
            return np.unique(rows, axis=0, return_inverse=True)[1].reshape(-1).astype(np.int64)
        cols = [inv.reshape(-1).astype(np.int64) for _, inv in ranks]
    key = np.zeros(len(rows), np.int64)
    for col, b in zip(cols, bits):
        key = (key << b) | col
    return np.unique(key, return_inverse=True)[1].reshape(-1).astype(np.int64)


def digest(*arrays: np.ndarray) -> bytes:
    """A short hash of arrays' shapes, types and contents, for caches keyed on geometry
    (about 10 ms for a 300k-triangle part)."""
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(a.data)
    return h.digest()


def parts_digest(parts: list[Part]) -> bytes:
    """`digest` of every part's geometry and kind (not its paint)."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.subtype.encode())
        h.update(digest(p.vertices, p.faces))
    return h.digest()


@dataclass(frozen=True)
class FaceMap:
    """Where each triangle of a re-meshed part came from (`face_map`)."""
    match: np.ndarray        # new triangle -> original triangle with the same corners, or -1
    shift: np.ndarray        # corner rotation of a matched triangle: new corner j = old corner (j + shift)
    unmatched: np.ndarray    # new triangles with no match
    nearest: np.ndarray      # per unmatched triangle: the original triangle with the nearest centroid


_FACE_MAPS: dict[bytes, FaceMap] = {}
FACE_MAP_CACHE = 4           # re-meshes remembered: a paint edit re-cuts the same geometry


def face_map(old_v: np.ndarray, old_f: np.ndarray, new_v: np.ndarray, new_f: np.ndarray) -> FaceMap:
    """Triangle correspondence between a part and its re-meshed geometry (see `remesh_part`).

    It depends on geometry only, so it is cached: a colour edit re-runs the pocket cut on
    the same shapes and then only carries the new paint over.
    """
    key = digest(old_v, old_f, new_v, new_f)
    if key in _FACE_MAPS:
        return _FACE_MAPS[key]
    # Quantised vertex positions -> shared integer ids across both meshes, then per corner.
    vid = row_ids(np.round(np.vstack([old_v, new_v]) / 1e-5).astype(np.int64))
    corner = np.vstack([vid[:len(old_v)][old_f], vid[len(old_v):][new_f]])
    # Canonical corner rotation: start at the smallest id (keeps orientation).
    k = np.argmin(corner, axis=1)
    rolled = corner[np.arange(len(corner))[:, None], (np.arange(3) + k[:, None]) % 3]
    tri_id = row_ids(rolled)
    n_old = len(old_f)
    lookup = np.full(tri_id.max() + 1, -1, np.int64)
    lookup[tri_id[:n_old]] = np.arange(n_old)
    match = lookup[tri_id[n_old:]]
    shift = (k[:n_old][np.maximum(match, 0)] - k[n_old:]) % 3
    unmatched = np.nonzero(match < 0)[0]
    nearest = np.zeros(0, np.int64)
    if len(unmatched):
        from scipy.spatial import cKDTree

        tree = cKDTree(old_v[old_f].mean(axis=1))
        nearest = np.asarray(tree.query(new_v[new_f[unmatched]].mean(axis=1))[1], np.int64)
    fm = FaceMap(match, shift, unmatched, nearest)
    while len(_FACE_MAPS) >= FACE_MAP_CACHE:
        _FACE_MAPS.pop(next(iter(_FACE_MAPS)))
    _FACE_MAPS[key] = fm
    return fm


def remesh_part(original: Part, mesh: trimesh.Trimesh) -> Part:
    """New geometry for a part (e.g. after a boolean cut), carrying per-triangle attributes over.

    Triangles whose three corners match an original triangle (same positions to
    1e-5 mm, any rotation of the corner order) keep that triangle's attributes
    verbatim; new triangles (pocket walls, re-triangulated areas) take the
    attributes of the nearest original triangle centroid.
    """
    v = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    if not original.tri_extra:
        return Part(original.name, original.extruder, v, f, {}, original.subtype)
    fm = face_map(original.vertices, original.faces, v, f)
    extra: dict[int, str] = {}
    old_extra = original.tri_extra
    match = fm.match.tolist()
    shift = fm.shift.tolist()
    for i in np.nonzero(fm.match >= 0)[0].tolist():
        ex = old_extra.get(match[i])
        if ex:
            r = shift[i]
            extra[i] = remap_paint_attrs(ex, (r, (r + 1) % 3, (r + 2) % 3))
    for i, j in zip(fm.unmatched.tolist(), fm.nearest.tolist()):
        if j in old_extra:
            extra[i] = whole_paint_attrs(old_extra[j])
    return Part(original.name, original.extruder, v, f, extra, original.subtype)


# ---------------------------------------------------------------- writing


def _mesh_xml(oid: int, part: Part) -> str:
    v = part.vertices
    vx = "\n".join(f'     <vertex x="{a:.6f}" y="{b:.6f}" z="{c:.6f}"/>' for a, b, c in v.tolist())
    rows = []
    for i, (a, b, c) in enumerate(part.faces.tolist()):
        ex = part.tri_extra.get(i)
        rows.append(f'     <triangle v1="{a}" v2="{b}" v3="{c}" {ex}/>' if ex else
                    f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>')
    return (f'  <object id="{oid}" type="model">\n   <mesh>\n    <vertices>\n{vx}\n    </vertices>\n'
            f'    <triangles>\n' + "\n".join(rows) + '\n    </triangles>\n   </mesh>\n  </object>\n')


_SKIP_PREFIXES = ("3D/", "Metadata/plate_", "Metadata/top_", "Metadata/pick_")
_SKIP_FILES = {"Metadata/model_settings.config", "Metadata/slice_info.config", "Metadata/cut_information.xml",
               "Metadata/filament_sequence.json", "[Content_Types].xml", "_rels/.rels"}

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>
"""
_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""
_MODEL_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def write_project(src: Project, name: str, extruder: int, parts: list[Part], out_path: str | Path,
                  positions: list[tuple[float, float]] | None = None,
                  settings: dict | None = None) -> Path:
    """Write a single-object Bambu 3MF.

    `parts` are in final print coordinates (object origin = XY centre, z = 0 on
    the bed). `positions` are plate XY positions of each instance (default: one
    copy at the plate centre). `settings` replaces the project settings.
    """
    out_path = Path(out_path)
    settings = settings if settings is not None else src.settings
    if positions is None:
        positions = [plate_centre(settings)]
        settings = front_wipe_tower(settings)
    obj_id = len(parts) + 1

    # Part meshes: one sub-model file, ids 1..k; the top-level object composes them.
    sub = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}" '
           f'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="{NS_PROD}" requiredextensions="p">',
           ' <metadata name="BambuStudio:3mfVersion">1</metadata>', ' <resources>']
    sub += [_mesh_xml(i + 1, p) for i, p in enumerate(parts)]
    sub += [' </resources>', ' <build/>', '</model>', '']

    comps = "\n".join(f'    <component p:path="/3D/Objects/object_1.model" objectid="{i + 1}" '
                      f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>' for i in range(len(parts)))
    items = "\n".join(f'  <item objectid="{obj_id}" transform="1 0 0 0 1 0 0 0 1 {x:.4f} {y:.4f} 0" printable="1"/>'
                      for x, y in positions)
    root = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}" '
            f'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="{NS_PROD}" requiredextensions="p">\n'
            # Bambu Studio only loads Metadata/project_settings.config from files
            # whose Application starts with "BambuStudio-"; anything else is
            # treated as a bare-geometry 3MF and sliced with defaults.
            f' <metadata name="Application">BambuStudio-{_esc(str((settings or {}).get("version", "02.00.00.00")))}</metadata>\n'
            f' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            f' <metadata name="Title">{_esc(name)}</metadata>\n'
            f' <resources>\n  <object id="{obj_id}" type="model">\n   <components>\n{comps}\n'
            f'   </components>\n  </object>\n </resources>\n <build>\n{items}\n </build>\n</model>\n')

    part_cfg = "\n".join(
        f'    <part id="{i + 1}" subtype="{p.subtype}">\n'
        f'      <metadata key="name" value={quoteattr(p.name)}/>\n'
        f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        + (f'      <metadata key="extruder" value="{p.extruder}"/>\n' if p.extruder else "")
        + f'      <mesh_stat face_count="{len(p.faces)}" edges_fixed="0" degenerate_facets="0" '
          f'facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n    </part>'
        for i, p in enumerate(parts))
    instances = "\n".join(f'    <model_instance>\n      <metadata key="object_id" value="{obj_id}"/>\n'
                          f'      <metadata key="instance_id" value="{i}"/>\n'
                          f'      <metadata key="identify_id" value="{100 + i}"/>\n    </model_instance>'
                          for i in range(len(positions)))
    model_cfg = (f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="{obj_id}">\n'
                 f'    <metadata key="name" value={quoteattr(name)}/>\n'
                 f'    <metadata key="extruder" value="{extruder}"/>\n{part_cfg}\n  </object>\n'
                 f'  <plate>\n    <metadata key="plater_id" value="1"/>\n'
                 f'    <metadata key="plater_name" value=""/>\n    <metadata key="locked" value="false"/>\n'
                 f'{instances}\n  </plate>\n</config>\n')

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zout.writestr("_rels/.rels", _ROOT_RELS)
        zout.writestr("3D/3dmodel.model", root)
        zout.writestr("3D/_rels/3dmodel.model.rels", _MODEL_RELS)
        zout.writestr("3D/Objects/object_1.model", "\n".join(sub))
        zout.writestr("Metadata/model_settings.config", model_cfg)
        if settings is not None:
            zout.writestr("Metadata/project_settings.config", json.dumps(settings, indent=4))
        if src.kind == "3mf":
            with zipfile.ZipFile(src.path) as zin:
                for info in zin.infolist():
                    n = info.filename
                    if n in _SKIP_FILES or n.startswith(_SKIP_PREFIXES) or n == "Metadata/project_settings.config":
                        continue
                    zout.writestr(info, zin.read(n))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.getvalue())
    return out_path


def plate_centre(settings: dict | None) -> tuple[float, float]:
    """Centre of the printable area (Bambu `printable_area` = ["0x0", "256x0", ...])."""
    try:
        pts = np.array([[float(c) for c in p.split("x")] for p in settings["printable_area"]])  # type: ignore[index]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    except (TypeError, KeyError, ValueError):
        return 128.0, 128.0


# One figure at the plate centre: plate 1's prime tower moves to the front of the bed (y TOWER_FRONT_Y), where
# it can grow as deep as it needs. The reference project keeps it at y 200 of a 256 mm bed, which leaves 56 mm:
# 9 filaments outgrew that and Bambu refused the slice ("G-code outside of the printable area", code -104).
# x stays (40 in the reference): on the X2D the second nozzle cannot reach x < 20.5 (`extruder_printable_area`;
# a tower at x 15 failed with code -102).
TOWER_FRONT_Y = 15.0


def front_wipe_tower(settings: dict | None) -> dict | None:
    """`settings` with plate 1's prime tower (`wipe_tower_y`, one entry per plate) at the front of the bed."""
    if not settings or not isinstance(settings.get("wipe_tower_y"), list) or not settings["wipe_tower_y"]:
        return settings
    out = dict(settings)
    out["wipe_tower_y"] = [f"{TOWER_FRONT_Y:g}", *settings["wipe_tower_y"][1:]]
    return out


def plate_size(settings: dict | None) -> tuple[float, float]:
    try:
        pts = np.array([[float(c) for c in p.split("x")] for p in settings["printable_area"]])  # type: ignore[index]
        return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))
    except (TypeError, KeyError, ValueError):
        return 256.0, 256.0


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
