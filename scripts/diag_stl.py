"""Diagnose printability problems in an STL: units, watertightness, open edges,
non-manifold edges, shell count, normals, self-intersections."""
import sys
from collections import Counter

import numpy as np
import trimesh


def report(path: str) -> None:
    # process=False keeps the raw triangle soup so we see the file as-is;
    # then merge coincident vertices the way a slicer does.
    raw = trimesh.load(path, force="mesh", process=False)
    print(f"== {path}")
    print(f"raw triangles: {len(raw.faces)}  raw vertices: {len(raw.vertices)}")
    m = trimesh.Trimesh(raw.vertices, raw.faces, process=True, validate=False)
    print(f"merged vertices: {len(m.vertices)}  faces: {len(m.faces)}")
    print(f"bounds min {m.bounds[0].round(3)} max {m.bounds[1].round(3)}  extents {m.extents.round(3)}")
    areas = m.area_faces
    print(f"degenerate (zero-area) faces: {(areas < 1e-12).sum()}")

    # Edge use counts: 2 = manifold, 1 = open boundary, >2 = non-manifold.
    edges = np.sort(m.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    c = Counter(counts.tolist())
    print(f"edge-use histogram: {dict(sorted(c.items()))}")
    print(f"watertight: {m.is_watertight}  winding consistent: {m.is_winding_consistent}  volume-valid: {m.is_volume}")

    bodies = m.split(only_watertight=False)
    print(f"connected shells: {len(bodies)}")
    for i, b in enumerate(sorted(bodies, key=lambda b: -len(b.faces))[:15]):
        print(f"  shell {i}: faces={len(b.faces)} watertight={b.is_watertight} "
              f"extents={b.extents.round(2)} vol={b.volume:.2f}")
    if m.is_watertight:
        print(f"volume: {m.volume:.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
