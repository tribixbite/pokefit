"""Repair redsmaller.stl (red pokeball half-shell) into a printable solid.

Defects in the SketchUp export (16x working scale):
  * 6 loose triangles floating near the flat spot,
  * 11 open boundary loops (one 121-unit sliver along the rim, the rest in the
    flat spot), 8 non-manifold edges, inconsistent winding,
  * the flat spot at z=3.464 has vertices dipping to z=0.44, so it is not
    actually planar on the bed.

Usage:
  python scripts/repair_shell.py <in.stl> <out.stl> [--scale 16] [--flat-z 3.464]
  --flat-z is the bed-face plane in SOURCE units; omit it to skip the trim.

Pipeline: MeshFix (keeps largest shell, fills holes, removes
self-intersections) -> manifold3d planar trim at the flat-spot plane (gives
an exact flat bed face) -> scale 1/16 to match red.stl's mm size -> sit on z=0.
"""
import argparse

import manifold3d as mf
import numpy as np
import pymeshfix
import trimesh

# Defaults measured on redsmaller.stl.
WORK_SCALE = 16.0  # redsmaller extents / red.stl extents = 744.096 / 46.51
FLAT_Z = 3.464  # the flat spot's plane, in source units (86 verts sit exactly here)


def to_manifold(m: trimesh.Trimesh) -> mf.Manifold:
    mesh = mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                   tri_verts=np.asarray(m.faces, np.uint32))
    out = mf.Manifold(mesh)
    if out.status() != mf.Error.NoError:
        raise RuntimeError(f"manifold3d rejected mesh: {out.status()}")
    return out


def from_manifold(man: mf.Manifold) -> trimesh.Trimesh:
    mesh = man.to_mesh()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=True)


def repair(src: str, dst: str, work_scale: float = WORK_SCALE,
           flat_z: float | None = FLAT_Z) -> trimesh.Trimesh:
    """Repair `src`, write a watertight mm-scale STL to `dst`, return the mesh."""
    raw = trimesh.load(src, force="mesh")
    # Work in float64 and recentre near the origin: coordinates around x=-1100
    # waste float32 precision once manifold3d takes over.
    offset = raw.bounds.mean(axis=0) * [1, 1, 0]
    verts = raw.vertices - offset

    fixer = pymeshfix.MeshFix(verts, raw.faces)
    # joincomp=False + remove_smallest_components=True drops the stray triangles.
    fixer.repair(joincomp=False, remove_smallest_components=True)
    fixed = trimesh.Trimesh(fixer.points, fixer.faces, process=True)
    trimesh.repair.fix_normals(fixed)
    print(f"meshfix: faces {len(raw.faces)} -> {len(fixed.faces)}, watertight={fixed.is_watertight}")

    man = to_manifold(fixed)
    if flat_z is not None:
        # Keep everything above the flat-spot plane; trim_by_plane caps the cut.
        man = man.trim_by_plane((0.0, 0.0, 1.0), flat_z)
    man = man.scale((1 / work_scale,) * 3)
    out = from_manifold(man)
    out.apply_translation(-out.bounds[0] * [0, 0, 1])  # bed face at z=0
    out.apply_translation(-out.bounds[0] * [1, 1, 0])  # match red.stl: min corner at origin
    out.export(dst)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=WORK_SCALE, help="divide by this (SketchUp working scale)")
    ap.add_argument("--flat-z", type=float, default=FLAT_Z, help="bed-face plane height in source units")
    ap.add_argument("--no-trim", action="store_true", help="skip the planar trim")
    a = ap.parse_args()
    repair(a.src, a.dst, a.scale, None if a.no_trim else a.flat_z)
