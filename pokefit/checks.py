"""Mesh soundness and printability checks.

Mesh checks run per part (a part is what the slicer treats as one volume).
Printability checks run on the printable solid: the union of all parts at the
final scale, pockets cut, bed at z = 0.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .model3mf import Part
from .planar import regions
from .report import Check, Status
from .shell import from_manifold, to_manifold

MAX_HIGHLIGHTS = 300
THIN_WARN = 0.8          # mm, below this walls print with fewer perimeters
THIN_FAIL = 0.4          # mm, below the nozzle width: detail is lost
THIN_FAIL_FRACTION = 0.05    # fail when more than 5 % of the surface is that thin (Arachne walls
                             # print isolated features down to ~0.1 mm, so small amounts only warn)
OVERHANG_MIN_AREA = 2.0      # mm²; smaller down-facing patches bridge or sag harmlessly
BED_LAYER = 0.2


def _sample(points: np.ndarray, n: int = MAX_HIGHLIGHTS) -> list[list[float]]:
    if len(points) > n:
        points = points[np.linspace(0, len(points) - 1, n).astype(int)]
    return np.round(points, 3).tolist()


def merged(part: Part) -> trimesh.Trimesh:
    """Part mesh with coincident vertices merged (STL soups become indexed)."""
    m = trimesh.Trimesh(part.vertices, part.faces, process=False)
    m.merge_vertices()
    return m


def mesh_checks(part: Part) -> list[Check]:
    """Watertight, manifold, winding, degenerate faces, stray shells, self-intersections."""
    m = merged(part)
    name, pid = part.name, part.name
    out: list[Check] = []

    edges = m.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_e, nonman = int((counts == 1).sum()), int((counts > 2).sum())
    if open_e or nonman:
        # Highlight midpoints of the offending edges.
        uniq, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
        bad = uniq[cnt != 2]
        mids = m.vertices[bad].mean(axis=1)
        out.append(Check(f"watertight:{pid}", f"{name}: closed surface", "fail",
                         f"{open_e} open edges and {nonman} edges shared by more than two faces.",
                         open_e + nonman, _sample(mids)))
    else:
        out.append(Check(f"watertight:{pid}", f"{name}: closed surface", "pass",
                         "Every edge joins exactly two faces."))

    if not m.is_winding_consistent:
        out.append(Check(f"winding:{pid}", f"{name}: face orientation", "fail",
                         "Neighbouring faces point opposite ways (inconsistent winding)."))
    elif m.is_watertight and m.volume < 0:
        out.append(Check(f"winding:{pid}", f"{name}: face orientation", "fail",
                         "The surface is inside-out (negative volume)."))
    else:
        out.append(Check(f"winding:{pid}", f"{name}: face orientation", "pass", "Consistent, outward-facing."))

    areas = m.area_faces
    degen = int((areas < 1e-10).sum())
    out.append(Check(f"degenerate:{pid}", f"{name}: degenerate faces", "warn" if degen else "pass",
                     f"{degen} zero-area faces." if degen else "None.", degen,
                     _sample(m.triangles_center[areas < 1e-10]) if degen else []))

    try:
        man = to_manifold(m)
        out.append(Check(f"manifold:{pid}", f"{name}: solid", "pass",
                         f"Valid solid, {man.volume():.1f} mm³."))
    except RuntimeError as exc:
        out.append(Check(f"manifold:{pid}", f"{name}: solid", "fail",
                         f"Not a valid solid for booleans ({exc}); pockets cannot be cut into it."))

    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        vols = sorted((abs(b.volume) if b.is_watertight else 0.0 for b in bodies), reverse=True)
        stray = [v for v in vols[1:] if v < 0.01 * max(vols[0], 1e-9)]
        status: Status = "warn" if stray else "info"
        out.append(Check(f"shells:{pid}", f"{name}: separate shells", status,
                         f"{len(bodies)} shells; {len(stray)} tiny (< 1 % of the largest)." if stray else
                         f"{len(bodies)} separate shells (intended pieces such as eyes are normal).",
                         len(bodies)))

    out.append(_self_intersections(m, name, pid))
    return out


def _self_intersections(m: trimesh.Trimesh, name: str, pid: str) -> Check:
    try:
        import pymeshlab
    except ImportError:
        return Check(f"selfint:{pid}", f"{name}: self-intersections", "info",
                     "Not checked (needs pymeshlab, which the local version installs).")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    cur = ms.current_mesh()
    n = int(cur.selected_face_number())
    if not n:
        return Check(f"selfint:{pid}", f"{name}: self-intersections", "pass", "None.")
    sel = cur.face_selection_array()
    pts = m.triangles_center[np.asarray(sel, bool)[:len(m.faces)]]
    return Check(f"selfint:{pid}", f"{name}: self-intersections", "warn",
                 f"{n} faces cut through other faces of the same part; slicers usually cope, "
                 f"but booleans may fail there.", n, _sample(pts))


def printable_solid(parts: list[Part]) -> trimesh.Trimesh:
    """Union of the parts (what gets printed); falls back to concatenation if a part is not a solid."""
    meshes = [merged(p) for p in parts if p.subtype == "normal_part"]
    try:
        man = to_manifold(meshes[0])
        for m in meshes[1:]:
            man = man + to_manifold(m)
        return from_manifold(man)
    except RuntimeError:
        cat = trimesh.util.concatenate(meshes)
        assert isinstance(cat, trimesh.Trimesh)
        return cat


THICK_RANGE = 1.0   # mm: rays look this far for the opposite surface (thin walls only matter < 0.8 mm)


def ray_distances(m: trimesh.Trimesh, origins: np.ndarray, dirs: np.ndarray, max_dist: float) -> np.ndarray:
    """Distance along each ray to the first triangle within `max_dist` (inf if none).

    Short rays only touch triangles near them, so candidates come from a KD-tree
    of triangle centres, searched in small spheres strung along each ray (large
    triangles are tested against every ray). Pure numpy/scipy: runs the same
    natively and in the browser.
    """
    from scipy.spatial import cKDTree

    tri = np.asarray(m.triangles, np.float64)
    cen = tri.mean(axis=1)
    rad = np.linalg.norm(tri - cen[:, None, :], axis=2).max(axis=1)
    # Spheres of radius seg/2 + r centred every `seg` mm along a ray cover every triangle of
    # circumradius <= r that can meet it. Triangles are grouped by size (powers of two) so
    # each group is searched with its own r: a few big faces do not inflate the search for all.
    n_seg = max(1, int(np.ceil(max_dist / 0.25)))
    seg = max_dist / n_seg
    ts = (np.arange(n_seg) + 0.5) * seg
    probes = (origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]).reshape(-1, 3)
    size_class = np.ceil(np.log2(np.maximum(rad, 1e-6) / 0.05)).clip(0, 30).astype(np.int64)
    rays, cands = [], []
    for k in np.unique(size_class):
        idx = np.nonzero(size_class == k)[0]
        r_max = float(rad[idx].max())
        found = cKDTree(cen[idx]).query_ball_point(probes, seg / 2 + r_max)
        lens = np.fromiter((len(f) for f in found), np.intp, len(found))   # platform index type (int32 in WASM)
        if lens.sum():
            rays.append(np.repeat(np.arange(len(probes)) // n_seg, lens))
            cands.append(idx[np.concatenate([np.asarray(f, np.int64) for f in found if len(f)])])
    ray = np.concatenate(rays) if rays else np.zeros(0, np.int64)
    cand = np.concatenate(cands) if cands else np.zeros(0, np.int64)
    # One test per (ray, triangle) even if it is near two probes; int64 so the key cannot overflow in WASM.
    key = np.unique(ray.astype(np.int64) * len(tri) + cand)
    ray, cand = (key // max(len(tri), 1)).astype(np.intp), (key % max(len(tri), 1)).astype(np.intp)
    out = np.full(len(origins), np.inf)
    if not len(ray):
        return out
    # Möller–Trumbore, vectorised over (ray, triangle) pairs.
    o, d = origins[ray], dirs[ray]
    v0, e1, e2 = tri[cand, 0], tri[cand, 1] - tri[cand, 0], tri[cand, 2] - tri[cand, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - v0
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t <= max_dist)
    np.minimum.at(out, ray[hit], t[hit])
    return out


def thickness_check(m: trimesh.Trimesh, samples: int = 4000, seed: int = 0) -> Check:
    """Local wall thickness: distance along the inward normal to the opposite surface."""
    pts, fidx = trimesh.sample.sample_surface(m, samples, seed=seed)
    normals = m.face_normals[fidx]
    origins = pts - normals * 1e-4
    thick = ray_distances(m, origins, -normals, THICK_RANGE)
    fail = thick < THIN_FAIL
    warn = (thick < THIN_WARN) & ~fail
    f_fail, f_warn = fail.mean(), warn.mean()
    hi = _sample(np.vstack([pts[fail], pts[warn]]))
    # Thickness beyond THICK_RANGE is not measured: count it as THICK_RANGE.
    p01 = float(np.percentile(np.minimum(thick, THICK_RANGE), 1))
    if f_fail > THIN_FAIL_FRACTION:
        return Check("thickness", "Wall thickness", "fail",
                     f"{f_fail:.1%} of the surface is under {THIN_FAIL} mm thick (nozzle width): large areas "
                     f"that thin print weak or not at all. Print larger.", round(p01, 2), hi)
    if f_fail > 0 or f_warn > 0.01:
        return Check("thickness", "Wall thickness", "warn",
                     f"{f_fail:.1%} of the surface under {THIN_FAIL} mm, {f_warn:.1%} under {THIN_WARN} mm "
                     f"(thin tips and edges print fragile or lose detail).", round(p01, 2), hi)
    return Check("thickness", "Wall thickness", "pass",
                 f"Thinnest 1 % of the surface is {p01:.2f} mm." if p01 < THICK_RANGE else
                 f"Thinnest 1 % of the surface is at least {THICK_RANGE:.0f} mm.", round(p01, 2))


def overhang_check(m: trimesh.Trimesh, threshold_deg: float,
                   ignore_xy: list[tuple[float, float, float, float]] | None = None) -> Check:
    """Down-facing surfaces flatter than the slicer's support threshold, above the bed.

    `threshold_deg` is Bambu's support_threshold_angle: a face whose angle to
    the horizontal is below it needs support. `ignore_xy` (x, y, r, z_top)
    cylinders exclude the magnet pocket ceilings (short bridges).
    """
    nz = m.face_normals[:, 2]
    tilt = np.degrees(np.arccos(np.clip(-nz, -1, 1)))      # 0 = flat ceiling, 90 = vertical wall
    centres = m.triangles_center
    over = (nz < 0) & (tilt < threshold_deg) & (centres[:, 2] > BED_LAYER)
    for x, y, r, z_top in ignore_xy or []:
        inside = (np.hypot(centres[:, 0] - x, centres[:, 1] - y) <= r + 0.05) & (centres[:, 2] <= z_top + 0.05)
        over &= ~inside
    idx = np.nonzero(over)[0]
    if not len(idx):
        return Check("overhang", "Overhangs", "pass", f"No surface flatter than {threshold_deg:.0f}° faces down.", 0.0)
    # Group into connected patches; ignore patches too small to matter.
    adj = m.face_adjacency
    keep = over[adj[:, 0]] & over[adj[:, 1]]
    groups = trimesh.graph.connected_components(adj[keep], nodes=idx, min_len=1)
    areas = m.area_faces
    big = [g for g in groups if areas[g].sum() >= OVERHANG_MIN_AREA]
    total = float(sum(areas[g].sum() for g in big))
    if not big:
        return Check("overhang", "Overhangs", "pass",
                     f"Only tiny down-facing patches (each < {OVERHANG_MIN_AREA} mm²).", 0.0)
    pts = np.vstack([centres[g] for g in big])
    lowest = min(float(centres[g, 2].min()) for g in big)
    return Check("overhang", "Overhangs", "warn",
                 f"{len(big)} down-facing area(s), {total:.0f} mm² flatter than {threshold_deg:.0f}° "
                 f"(lowest at z {lowest:.1f} mm) — needs support unless they bridge.", round(total, 1), _sample(pts))


def islands_check(m: trimesh.Trimesh, layer: float = 0.2) -> Check:
    """Layer regions with nothing below them (start in mid-air): these always need support."""
    zmax = m.bounds[1][2]
    heights = np.arange(layer / 2, zmax, layer)
    sections = m.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=list(heights))
    from shapely.ops import unary_union

    prev = None
    found: list[list[float]] = []
    for z, sec in zip(heights, sections):
        polys = regions(sec) if sec is not None else []
        cur = unary_union(polys) if polys else None
        if prev is not None and cur is not None:
            for p in polys:
                if p.area > 0.05 and not p.intersects(prev):
                    c = p.representative_point()
                    found.append([float(c.x), float(c.y), float(z)])
        if cur is not None:
            prev = cur
    if not found:
        return Check("islands", "Floating islands", "pass", "Every layer region rests on the one below.", 0)
    return Check("islands", "Floating islands", "fail",
                 f"{len(found)} region(s) start in mid-air (first at z {found[0][2]:.1f} mm): "
                 f"they need support.", len(found), found[:MAX_HIGHLIGHTS])


def bed_contact_check(m: trimesh.Trimesh) -> Check:
    """Area touching the bed vs. height: small footprints on tall figures want a brim."""
    sec = m.section(plane_origin=[0, 0, 0.1], plane_normal=[0, 0, 1])
    area = 0.0
    if sec is not None:
        planar, _ = sec.to_2D(to_2D=np.linalg.inv(trimesh.transformations.translation_matrix([0, 0, 0.1])),
                              check=False)
        area = float(sum(p.area for p in regions(planar)))
    h = float(m.extents[2])
    ratio = h / np.sqrt(area) if area > 0 else np.inf
    if area <= 0:
        return Check("bed", "Bed contact", "fail", "Nothing touches the bed.", 0.0)
    if area < 30 or ratio > 6:
        return Check("bed", "Bed contact", "warn",
                     f"{area:.0f} mm² on the bed for {h:.0f} mm of height — add a brim (PETG lifts easily).",
                     round(area, 1))
    return Check("bed", "Bed contact", "pass", f"{area:.0f} mm² on the bed.", round(area, 1))


def supports_verdict(checks: list[Check]) -> Check:
    """Combine overhang + island findings into the yes/no answer."""
    by = {c.id: c for c in checks}
    reasons = []
    if "islands" in by and by["islands"].status == "fail":
        reasons.append(f"{by['islands'].value} floating region(s)")
    if "overhang" in by and by["overhang"].status != "pass":
        reasons.append(f"{by['overhang'].value} mm² of steep overhang")
    if reasons:
        return Check("supports", "Supports needed", "warn", "Yes — " + " and ".join(reasons) + ".", "yes")
    return Check("supports", "Supports needed", "pass", "No — prints without supports.", "no")
