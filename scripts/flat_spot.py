"""Add a flat resting spot to a pokeball half-shell, and fill the inside of the
pole so the wall stays thick and the tiny pole layers become one flat area.

Works on the shell in its native frame (dome up, pole on +z), e.g. the
0_ball_shell_2x object extracted from pokeball.3mf by extract_3mf.py.

  * Outer: the dome is cut by a horizontal plane so the cut face is a disk of
    `--disk` mm diameter (final print size).
  * Inner: the cavity above a horizontal plane is filled, giving an inner flat
    ceiling of `--fill` mm diameter. Keep --fill >= --disk so the wall at the
    pole is never thinner than the original wall.

Both spheres are least-squares fitted from the mesh, so nothing is hard-coded
to one model. Output is scaled by --scale, rim on z=0 (same print orientation
as the 3MF), centred on the pole axis.

Usage:
  python scripts/flat_spot.py <shell.stl> <out.stl> [--scale 0.6] [--disk 8] [--fill 10]
"""
import argparse

import manifold3d as mf
import numpy as np
import trimesh


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic least-squares sphere fit -> (center, radius)."""
    a = np.c_[2 * points, np.ones(len(points))]
    x = np.linalg.lstsq(a, (points ** 2).sum(axis=1), rcond=None)[0]
    center = x[:3]
    return center, float(np.sqrt(x[3] + center @ center))


def shell_spheres(m: trimesh.Trimesh) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
    """Fit outer and inner spheres from the upper dome (away from rim features).

    Outer-surface vertices have normals pointing away from the ball centre,
    inner-surface ones point towards it; a rough centre guess on the axis below
    the rim is enough to tell them apart.
    """
    v, n = m.vertices, m.vertex_normals
    guess = np.array([*v[:, :2].mean(axis=0), m.bounds[0][2]])
    radial = (v - guess) / np.linalg.norm(v - guess, axis=1)[:, None]
    facing = np.einsum("ij,ij->i", n, radial)
    upper = v[:, 2] > (m.bounds[0][2] + m.bounds[1][2]) / 2
    outer = fit_sphere(v[upper & (facing > 0.95)])
    inner = fit_sphere(v[upper & (facing < -0.95)])
    return outer, inner


def to_manifold(m: trimesh.Trimesh) -> mf.Manifold:
    man = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                              tri_verts=np.asarray(m.faces, np.uint32)))
    if man.status() != mf.Error.NoError:
        raise RuntimeError(f"manifold3d rejected mesh: {man.status()}")
    return man


def cap_height(radius: float, chord_diameter: float) -> float:
    """Height of the plane above the sphere centre where the chord has this diameter."""
    half = chord_diameter / 2
    if half >= radius:
        raise ValueError(f"diameter {chord_diameter:.2f} exceeds sphere diameter {2 * radius:.2f}")
    return float(np.sqrt(radius ** 2 - half ** 2))


def flat_spot(src: str, dst: str, scale: float, disk: float, fill: float) -> trimesh.Trimesh:
    shell = trimesh.load(src, force="mesh")
    (c_out, r_out), (c_in, r_in) = shell_spheres(shell)
    center = (c_out + c_in) / 2
    print(f"outer R={r_out:.3f}  inner R={r_in:.3f}  wall={r_out - r_in:.3f} mm (native)  "
          f"centre offset={np.linalg.norm(c_out - c_in):.3f}")

    # Diameters are final-print mm; convert to native mm before building geometry.
    z_flat = center[2] + cap_height(r_out, disk / scale)
    z_fill = center[2] + cap_height(r_in, fill / scale)

    # Recentre on the pole axis so the cylinder/sphere primitives sit at the origin.
    shift = -np.array([center[0], center[1], 0.0])
    shell.apply_translation(shift)
    body = to_manifold(shell)

    # Inner fill: a sphere reaching mid-wall, clipped to everything above z_fill.
    # Growing it only to r_in leaves a sliver void: the mesh's faceted inner
    # pole sits ~0.13 mm off the fitted sphere. Mid-wall overlaps any such
    # deviation while staying inside the outer surface.
    ball = mf.Manifold.sphere((r_in + r_out) / 2, 256).translate((0.0, 0.0, float(center[2])))
    plug = ball.trim_by_plane((0.0, 0.0, 1.0), float(z_fill))
    body = body + plug

    # Outer flat: discard everything above z_flat (the cut face is capped).
    body = body.trim_by_plane((0.0, 0.0, -1.0), -float(z_flat))
    body = body.scale((scale, scale, scale))

    mesh = body.to_mesh()
    out = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=True)
    out.apply_translation([0.0, 0.0, -out.bounds[0][2]])  # rim on the bed

    pole_wall = (z_flat - z_fill) * scale
    print(f"disk {disk:.2f} mm, inner flat {fill:.2f} mm, wall at pole {pole_wall:.2f} mm "
          f"(original {(r_out - r_in) * scale:.2f} mm), height {out.extents[2]:.2f} mm")
    print(f"watertight={out.is_watertight} volume={out.volume:.1f} mm^3 "
          f"(added {out.volume - shell.volume * scale ** 3:+.1f})")
    out.export(dst)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=1.0, help="print scale, e.g. 0.6 for the 60%% ball")
    ap.add_argument("--disk", type=float, default=8.0, help="outer flat disk diameter, final mm")
    ap.add_argument("--fill", type=float, default=10.0, help="inner flat ceiling diameter, final mm")
    a = ap.parse_args()
    if a.fill < a.disk:
        print("warning: --fill < --disk thins the wall at the pole")
    flat_spot(a.src, a.dst, a.scale, a.disk, a.fill)
