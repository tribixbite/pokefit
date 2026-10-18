"""The subset of the `manifold3d` Python API pokefit uses, backed by manifold's WebAssembly build.

In the browser (Pyodide) the native `manifold3d` wheel is unavailable; the web
worker loads the official `manifold-3d` npm package, calls its `setup()`, and
exposes it as `globalThis.manifoldWasm`. This module mirrors exactly the calls
made in shell.py / magnets.py / pipeline.py / checks.py:

  Manifold(Mesh(vert_properties=..., tri_verts=...)), .status(), Error.NoError,
  .to_mesh().vert_properties / .tri_verts, .volume(), .bounding_box(),
  Manifold.cylinder(h, r_low, r_high, segments), Manifold.sphere(r, segments),
  .translate((x, y, z)), .scale((x, y, z)), .trim_by_plane((nx, ny, nz), offset),
  a + b (union), a - b (difference).

Each wrapper owns its WebAssembly object and frees it when collected.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from js import Float32Array, Object, Uint32Array, manifoldWasm
from pyodide.ffi import to_js


class Error:
    NoError = "NoError"


def _typed(arr: np.ndarray, kind: Any, dtype: type) -> Any:
    """numpy -> JS typed array (one copy)."""
    return kind.new(to_js(np.ascontiguousarray(arr, dtype).ravel()))


class Mesh:
    def __init__(self, vert_properties: np.ndarray, tri_verts: np.ndarray):
        self.vert_properties = np.asarray(vert_properties, np.float32)
        self.tri_verts = np.asarray(tri_verts, np.uint32)


class Manifold:
    def __init__(self, mesh: Mesh | None = None, *, _js: Any = None):
        if _js is None:
            assert mesh is not None
            opts = to_js({"numProp": 3,
                          "vertProperties": _typed(mesh.vert_properties[:, :3], Float32Array, np.float32),
                          "triVerts": _typed(mesh.tri_verts, Uint32Array, np.uint32)},
                         dict_converter=Object.fromEntries)
            # No merge(): the native path does not weld either; callers merge vertices first.
            _js = manifoldWasm.Manifold.new(manifoldWasm.Mesh.new(opts))
        self._js = _js

    def __del__(self) -> None:
        try:
            self._js.delete()
        except Exception:  # noqa: BLE001 — interpreter shutdown / already freed
            pass

    # constructors
    @staticmethod
    def cylinder(height: float, radius_low: float, radius_high: float = -1.0, circular_segments: int = 0,
                 center: bool = False) -> "Manifold":
        return Manifold(_js=manifoldWasm.Manifold.cylinder(height, radius_low, radius_high, circular_segments, center))

    @staticmethod
    def sphere(radius: float, circular_segments: int = 0) -> "Manifold":
        return Manifold(_js=manifoldWasm.Manifold.sphere(radius, circular_segments))

    # queries
    def status(self) -> str:
        return str(self._js.status())

    def volume(self) -> float:
        return float(self._js.volume())

    def bounding_box(self) -> tuple[float, float, float, float, float, float]:
        b = self._js.boundingBox()
        lo, hi = list(b.min), list(b.max)
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def to_mesh(self) -> Mesh:
        m = self._js.getMesh()
        n = int(m.numProp)
        verts = np.asarray(m.vertProperties.to_py(), np.float32).reshape(-1, n)
        tris = np.asarray(m.triVerts.to_py(), np.uint32).reshape(-1, 3)
        return Mesh(verts, tris)

    # transforms / booleans
    def translate(self, v: tuple[float, float, float]) -> "Manifold":
        return Manifold(_js=self._js.translate(to_js(list(map(float, v)))))

    def scale(self, v: tuple[float, float, float]) -> "Manifold":
        return Manifold(_js=self._js.scale(to_js(list(map(float, v)))))

    def trim_by_plane(self, normal: tuple[float, float, float], offset: float) -> "Manifold":
        return Manifold(_js=self._js.trimByPlane(to_js(list(map(float, normal))), float(offset)))

    def __add__(self, other: "Manifold") -> "Manifold":
        return Manifold(_js=self._js.add(other._js))

    def __sub__(self, other: "Manifold") -> "Manifold":
        return Manifold(_js=self._js.subtract(other._js))
