"""Boolean/mesh kernel: native `manifold3d`, or its WebAssembly build in the browser."""
from __future__ import annotations

import sys

if sys.platform == "emscripten":         # Pyodide: no native wheel; the web worker provides manifold-3d
    from . import _manifold_js as mf
else:
    import manifold3d as mf

__all__ = ["mf"]
