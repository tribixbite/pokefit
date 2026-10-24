"""Pokéball shell geometry: sphere fitting and the flat-bottom / washer-seat cut.

The shell is a dome in its native frame (pole on +z), e.g. the
`0_ball_shell_2x` object of pokeball.3mf.

* Outer: the dome is cut by a horizontal plane so the cut face is a disk of
  `disk` mm diameter (final print size) — the ball rests on it.
* Inner: the cavity above a horizontal plane is filled, giving an inner flat
  of `fill` mm diameter. Used as the washer seat (fill = washer OD) and to keep
  the pole wall thick. Keep fill >= disk so the pole wall never thins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .geom import mf

Sphere = tuple[np.ndarray, float]


def fit_sphere(points: np.ndarray) -> Sphere:
    """Algebraic least-squares sphere fit -> (center, radius)."""
    a = np.c_[2 * points, np.ones(len(points))]
    x = np.linalg.lstsq(a, (points ** 2).sum(axis=1), rcond=None)[0]
    center = x[:3]
    return center, float(np.sqrt(x[3] + center @ center))


def shell_spheres(m: trimesh.Trimesh) -> tuple[Sphere, Sphere]:
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
    """trimesh -> manifold3d; raises if manifold3d rejects the mesh."""
    man = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                              tri_verts=np.asarray(m.faces, np.uint32)))
    if man.status() != mf.Error.NoError:
        raise RuntimeError(f"manifold3d rejected mesh: {man.status()}")
    return man


def from_manifold(man: mf.Manifold) -> trimesh.Trimesh:
    """manifold3d -> trimesh, its topology as it is: every edge joins two faces by index. Welding by position
    (trimesh's `process`) would fuse surfaces a boolean leaves touching with no thickness between them into edges
    on four faces (repair.py moves such vertices apart instead)."""
    mesh = man.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties[:, :3], np.float64),
                           np.asarray(mesh.tri_verts, np.int64), process=False)


def cap_height(radius: float, chord_diameter: float) -> float:
    """Height of the plane above the sphere centre where the chord has this diameter."""
    half = chord_diameter / 2
    if half >= radius:
        raise ValueError(f"diameter {chord_diameter:.2f} exceeds sphere diameter {2 * radius:.2f}")
    return float(np.sqrt(radius ** 2 - half ** 2))


@dataclass
class FlatSpotResult:
    mesh: trimesh.Trimesh
    r_outer: float          # native mm
    r_inner: float          # native mm
    pole_wall: float        # final mm
    original_wall: float    # final mm


def flat_spot(shell: trimesh.Trimesh, scale: float, disk: float, fill: float) -> FlatSpotResult:
    """Cut the outer flat disk and fill the inner pole; returns the scaled part, rim on z=0."""
    shell = shell.copy()
    (c_out, r_out), (c_in, r_in) = shell_spheres(shell)
    center = (c_out + c_in) / 2

    # Diameters are final-print mm; convert to native mm before building geometry.
    z_flat = center[2] + cap_height(r_out, disk / scale)
    z_fill = center[2] + cap_height(r_in, fill / scale)

    # Recentre on the pole axis so the sphere primitive sits at the origin.
    shell.apply_translation(-np.array([center[0], center[1], 0.0]))
    body = to_manifold(shell)

    # Inner fill: a sphere reaching mid-wall, clipped to everything above z_fill.
    # Growing it only to r_in leaves a sliver void: the mesh's faceted inner
    # pole sits ~0.13 mm off the fitted sphere. Mid-wall overlaps any such
    # deviation while staying inside the outer surface.
    ball = mf.Manifold.sphere((r_in + r_out) / 2, 256).translate((0.0, 0.0, float(center[2])))
    body = body + ball.trim_by_plane((0.0, 0.0, 1.0), float(z_fill))

    # Outer flat: discard everything above z_flat (the cut face is capped).
    body = body.trim_by_plane((0.0, 0.0, -1.0), -float(z_flat)).scale((scale, scale, scale))
    out = from_manifold(body)
    out.apply_translation([0.0, 0.0, -out.bounds[0][2]])  # rim on the bed
    return FlatSpotResult(out, r_out, r_in, (z_flat - z_fill) * scale, (r_out - r_in) * scale)


def washer_seat_depth(r_inner_final: float, washer_od: float) -> float:
    """How far above the inner pole the washer seat (inner flat of diameter OD) sits, final mm."""
    return r_inner_final - cap_height(r_inner_final, washer_od)
