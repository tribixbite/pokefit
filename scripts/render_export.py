"""Render a pokefit export (or any 3MF / STL / GLB) with its painted colours, from five sides, to a PNG.

    python scripts/render_export.py pokefit_out/<name>/<name>_pokefit_60.3mf out.png [--size 400]
        [--zoom X,Z,SPAN]      # a front close-up (and two 30-degree turns) round X, Z (figure frame, mm)

What the file holds, exactly: every triangle is rasterised into a z-buffer (pixels whose centre it covers),
coloured by its main paint state (`pipeline.painted_mesh`) and shaded by its normal. A cheaper splat render
(a few points per triangle) let the far side show through as dark speckles and was taken for paint defects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pokefit.model3mf import read_project, world_parts
from pokefit.pipeline import painted_mesh

BG = 28


def view_matrix(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """World -> camera (x right, y up, z towards the viewer); yaw 0 = front (camera at -Y), Z up."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    front = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    rz = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return np.asarray(rx @ front @ rz)


def render(v: np.ndarray, f: np.ndarray, rgb: np.ndarray, view: np.ndarray, size: int,
           span: float | None = None, centre: np.ndarray | None = None) -> np.ndarray:
    """Orthographic z-buffer render, (size, size, 3) uint8; `span` mm across, centred on `centre` (camera frame)."""
    tri = (v @ view.T)[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.32 + 0.68 * np.clip(n[:, 2] * 0.8 + n[:, 1] * 0.3 + 0.05, 0, 1)
    col = np.clip(rgb * shade[:, None], 0, 255).astype(np.uint8)
    flat = tri.reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    span = span or float((hi - lo)[:2].max() * 1.06)
    c = centre if centre is not None else (hi + lo) / 2
    px = ((tri[..., 0] - c[0]) / span + 0.5) * size - 0.5          # pixel centres at integers
    py = (0.5 - (tri[..., 1] - c[1]) / span) * size - 0.5
    pz = tri[..., 2]
    x0 = np.clip(np.ceil(px.min(1)), 0, size).astype(np.int64)
    x1 = np.clip(np.floor(px.max(1)), -1, size - 1).astype(np.int64)
    y0 = np.clip(np.ceil(py.min(1)), 0, size).astype(np.int64)
    y1 = np.clip(np.floor(py.max(1)), -1, size - 1).astype(np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    zbuf = np.full(size * size, -np.inf)
    img = np.full((size * size, 3), BG, np.uint8)
    idx = np.nonzero((w > 0) & (h > 0))[0]
    for chunk in np.array_split(idx, max(1, len(idx) // 200_000)):
        cnt = w[chunk] * h[chunk]
        t = np.repeat(chunk, cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        xx, yy = x0[t] + k % w[t], y0[t] + k // w[t]
        ax, ay, bx, by, cx, cy = px[t, 0], py[t, 0], px[t, 1], py[t, 1], px[t, 2], py[t, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        l1 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / den
        l2 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / den
        l3 = 1 - l1 - l2
        ok = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        t, xx, yy = t[ok], xx[ok], yy[ok]
        z = l1[ok] * pz[t, 0] + l2[ok] * pz[t, 1] + l3[ok] * pz[t, 2]
        pix = yy * size + xx
        order = np.lexsort((-z, pix))
        pix, z, t = pix[order], z[order], t[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, z, t = pix[first], z[first], t[first]
        better = z > zbuf[pix]
        zbuf[pix[better]] = z[better]
        img[pix[better]] = col[t[better]]
    return img.reshape(size, size, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--zoom", help="X,Z,SPAN: a front close-up round X, Z (mm), SPAN mm across")
    a = ap.parse_args()
    proj = read_project(a.file)
    obj = proj.default_object()
    cols = proj.colours
    vs, fs, cs, base = [], [], [], 0
    for p in world_parts(obj):
        if p.subtype != "normal_part":
            continue
        pv, pf, ext = painted_mesh(p, p.extruder or obj.extruder or 1)
        rgb = np.array([[int(cols[(e - 1) % len(cols)].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for e in ext],
                       float)
        vs.append(pv)
        fs.append(pf + base)
        cs.append(rgb)
        base += len(pv)
    v, f, c = np.vstack(vs), np.vstack(fs), np.vstack(cs)
    v = v - [(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()]
    if a.zoom:
        x, z, span = (float(t) for t in a.zoom.split(","))
        views = [view_matrix(yw, pt) for yw, pt in ((0, 3), (-30, 5), (30, 5))]
        tiles = [render(v, f, c, m, a.size, span, np.array([x, 0.0, z]) @ m.T) for m in views]
    else:
        tiles = [render(v, f, c, view_matrix(yw, pt), a.size) for yw, pt in ((0, 8), (90, 8), (180, 8), (270, 8), (35, 25))]
    sheet = Image.new("RGB", (a.size * len(tiles), a.size))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (a.size * i, 0))
    sheet.save(a.out)
    print("saved", a.out, sheet.size)


if __name__ == "__main__":
    main()
