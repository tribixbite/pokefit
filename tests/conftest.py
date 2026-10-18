"""Shared synthetic fixtures."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

REAL_DRAGON = Path.home() / "Downloads" / "dragonwithegg+eyes+multicolor.3mf"
REAL_ZELDA = Path.home() / "Downloads" / "Zelda+Pot_stls.zip"
BALL_3MF = Path(__file__).resolve().parent.parent / "pokeball.3mf"


def two_footed(foot: float = 8.0, gap: float = 10.0, leg_h: float = 6.0, body_h: float = 10.0) -> trimesh.Trimesh:
    """Two square feet (side `foot`, centres ±(foot+gap)/2 on x) under a body block."""
    off = (foot + gap) / 2
    legs = [trimesh.creation.box([foot, foot, leg_h]).apply_translation([s * off, 0, leg_h / 2]) for s in (-1, 1)]
    body = trimesh.creation.box([2 * foot + gap, foot, body_h]).apply_translation([0, 0, leg_h + body_h / 2 - 0.01])
    return trimesh.boolean.union(legs + [body], engine="manifold")


def write_inline_3mf(path: Path, mesh: trimesh.Trimesh, transform: np.ndarray,
                     paint: dict[int, str] | None = None) -> Path:
    """Minimal core-spec 3MF: one inline mesh object, one build item with `transform`."""
    paint = paint or {}
    verts = "".join(f'<vertex x="{a}" y="{b}" z="{c}"/>' for a, b, c in mesh.vertices.tolist())
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"' + (f' paint_color="{paint[i]}"' if i in paint else "") + "/>"
                   for i, (a, b, c) in enumerate(mesh.faces.tolist()))
    m = transform
    tf = " ".join(f"{x:.9g}" for x in np.vstack([m[:3, :3].T, m[:3, 3]]).ravel())
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'
             f'<object id="1" type="model"><mesh><vertices>{verts}</vertices><triangles>{tris}</triangles>'
             f'</mesh></object></resources><build><item objectid="1" transform="{tf}"/></build></model>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", model)
    return path


@pytest.fixture
def box_feet() -> trimesh.Trimesh:
    return two_footed()
