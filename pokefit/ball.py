"""Cavity model of the assembled Pokéball, derived from pokeball.3mf.

Assembled ball (native mm, z = 0 at the ball centre / equator plane):
  * two shells, inner sphere radius Ri, rims ~3.3 mm short of the equator;
  * two connector rings ("1_inner_black"), each lining one shell from the
    equator toward its pole over `ring_span` mm; their outer faces sit on the
    shell's inner sphere and they meet at the equator.

The figure may occupy, at height z, the disc of radius
  sphere(z) = sqrt(Ri² − z²)                   everywhere
  ring(|z|)                                    inside the ring span
  aperture (= min ring radius)                 beyond the span: anything there
                                               must pass through a ring when the
                                               ball is closed
all scaled by the ball scale and reduced by the clearance. The ring radius at
each depth is the minimum over all angles (conservative: the latch notch sets
it near the button).

The floor is the top of the steel washer resting on the washer seat (inner
flat of diameter OD + seat_margin filled into the bottom shell's pole).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from .model3mf import Part, read_project
from .shell import cap_height, fit_sphere, shell_spheres
from .washer import Washer

SHELL_NAME = "0_ball_shell"
RING_NAME = "1_inner_black"
SEAT_MARGIN = 0.4   # washer seat diameter = OD + this, so the washer drops in (final mm)
RING_STEP = 0.25    # profile sampling step along the axis, native mm


@dataclass
class Ball:
    """Native (unscaled) ball geometry."""

    r_inner: float
    r_outer: float
    ring_h: list[float]      # depth below the equator, native mm (ascending)
    ring_r: list[float]      # min inner radius of either ring at that depth
    ring_span: float
    aperture: float
    source_sha256: str

    def ring_radius(self, h: np.ndarray) -> np.ndarray:
        """Ring inner radius at depth |h| (inf outside the ring span)."""
        h = np.abs(np.asarray(h, dtype=np.float64))
        r = np.interp(h, self.ring_h, self.ring_r)
        return np.where(h <= self.ring_span, r, np.inf)


def _merged(parts: list[Part]) -> trimesh.Trimesh:
    m = trimesh.util.concatenate([p.mesh() for p in parts])
    assert isinstance(m, trimesh.Trimesh)
    return m


def _ring_profile(ring: trimesh.Trimesh) -> tuple[float, np.ndarray, np.ndarray]:
    """Equator z and (depth, min radius) profile of one ring, native mm, ring axis vertical.

    The equator is the centre of the sphere fitted to the ring's outward-facing,
    sloped faces (the part that lies against the shell), ignoring the top 3 mm
    where the twist-lock lip sits.
    """
    cx, cy = ring.bounds.mean(axis=0)[:2]
    v, n = ring.vertices, ring.vertex_normals
    radial = np.c_[v[:, :2] - [cx, cy], np.zeros(len(v))]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    lining = (np.einsum("ij,ij->i", n, radial) > 0.3) & (np.abs(n[:, 2]) > 0.15) & (v[:, 2] < ring.bounds[1, 2] - 3)
    centre, _ = fit_sphere(v[lining])
    equator = float(centre[2])
    # The pole-ward end is the end further from the equator.
    toward_pole = -1.0 if abs(ring.bounds[0, 2] - equator) > abs(ring.bounds[1, 2] - equator) else 1.0
    zs = np.arange(ring.bounds[0, 2] + 0.05, ring.bounds[1, 2] - 0.05, RING_STEP)
    sections = ring.section_multiplane(plane_origin=[cx, cy, 0], plane_normal=[0, 0, 1], heights=list(zs))
    hs, rs = [], []
    for z, sec in zip(zs, sections):
        if sec is None:
            continue
        pts = np.vstack([np.asarray(p.exterior.coords) for p in sec.polygons_full] +
                        [np.asarray(i.coords) for p in sec.polygons_full for i in p.interiors])
        # section_multiplane returns coordinates relative to plane_origin.
        rs.append(float(np.linalg.norm(pts, axis=1).min()))
        hs.append((z - equator) * toward_pole)
    order = np.argsort(hs)
    return equator, np.asarray(hs)[order], np.asarray(rs)[order]


def derive_ball(path: str | Path) -> Ball:
    path = Path(path)
    proj = read_project(path)
    shells = [o for o in proj.objects if o.name.startswith(SHELL_NAME)]
    rings_obj = [o for o in proj.objects if o.name.startswith(RING_NAME)]
    if not shells or not rings_obj:
        raise ValueError(f"{path.name}: needs objects named {SHELL_NAME}* and {RING_NAME}*")
    (_, r_out), (_, r_in) = shell_spheres(_merged(shells[0].parts))
    rings = [b for b in _merged(rings_obj[0].parts).split(only_watertight=False) if len(b.faces) > 1000]
    grid = np.arange(0.0, 40.0, RING_STEP)
    r_min = np.full(grid.shape, np.inf)
    span = 0.0
    for ring in rings:
        _, h, r = _ring_profile(ring)
        # Depths above the equator (lip of the twist lock) fold onto |h| as well.
        h_abs = np.abs(h)
        span = max(span, float(h.max()))
        for hv, rv in zip(h_abs, r):
            i = int(round(hv / RING_STEP))
            if i < len(r_min):
                r_min[i] = min(r_min[i], rv)
    keep = np.isfinite(r_min) & (grid <= span)
    ring_h, ring_r = grid[keep], r_min[keep]
    return Ball(r_in, r_out, ring_h.tolist(), ring_r.tolist(), span, float(ring_r.min()),
                hashlib.sha256(path.read_bytes()).hexdigest())


def load_ball(path: str | Path, cache_dir: str | Path | None = None) -> Ball:
    """derive_ball with a JSON cache keyed by the file's SHA-256."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = Path(cache_dir) / f"ball_{sha[:16]}.json" if cache_dir else None
    if cache and cache.exists():
        return Ball(**json.loads(cache.read_text(encoding="utf-8")))
    ball = derive_ball(path)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(asdict(ball)), encoding="utf-8")
    return ball


@dataclass
class Cavity:
    """The space a figure may occupy, final mm, z = 0 at the ball centre."""

    ball: Ball
    scale: float
    washer: Washer
    clearance: float = 1.0

    @property
    def r_inner(self) -> float:
        return self.ball.r_inner * self.scale

    @property
    def seat_diameter(self) -> float:
        return self.washer.od + SEAT_MARGIN

    @property
    def seat_z(self) -> float:
        """Top of the washer seat (inner flat filled into the bottom pole)."""
        return -cap_height(self.r_inner, self.seat_diameter)

    @property
    def floor_z(self) -> float:
        """Top face of the washer: where the figure stands."""
        return self.seat_z + self.washer.thickness

    @property
    def top_z(self) -> float:
        return self.r_inner - self.clearance

    def allowed_radius(self, z: np.ndarray) -> np.ndarray:
        """Max distance from the ball axis at height z (negative = nothing fits)."""
        z = np.asarray(z, dtype=np.float64)
        s = self.scale
        sphere = np.sqrt(np.clip(self.r_inner ** 2 - z ** 2, 0, None))
        ring = self.ball.ring_radius(z / s) * s
        outside = np.isinf(ring)
        ring = np.where(outside, self.ball.aperture * s, ring)
        r = np.minimum(sphere, ring) - self.clearance
        return np.where(z > self.top_z, -1.0, r)

    def profile(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """(z, allowed radius) from floor to top, for plots and the UI."""
        z = np.linspace(self.floor_z, self.top_z, n)
        return z, self.allowed_radius(z)
