"""Draw the site's PNG icons (apple-touch-icon, manifest icons) into web/public/.

The same ball as the inline SVG favicon in web/index.html: red top, white
bottom, dark band and button. Drawn at 4x and downsampled for smooth edges.

    python scripts/make_icons.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "web" / "public"
BG = (13, 17, 23, 255)          # --bg
RED = (229, 72, 77, 255)        # --accent
WHITE = (232, 236, 244, 255)
DARK = (22, 26, 34, 255)


def ball(size: int, maskable: bool = False) -> Image.Image:
    s = size * 4
    im = Image.new("RGBA", (s, s), BG if maskable else (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # Maskable icons keep the art inside the central 80 % safe zone.
    pad = s * (0.16 if maskable else 0.04)
    box = (pad, pad, s - pad, s - pad)
    d.ellipse(box, fill=WHITE)
    d.pieslice(box, 180, 360, fill=RED)
    c, r = s / 2, (s - 2 * pad) / 2
    band = r * 0.11
    d.rectangle((pad, c - band, s - pad, c + band), fill=DARK)
    d.ellipse((c - r * 0.36, c - r * 0.36, c + r * 0.36, c + r * 0.36), fill=DARK)
    d.ellipse((c - r * 0.2, c - r * 0.2, c + r * 0.2, c + r * 0.2), fill=WHITE)
    return im.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ball(180, maskable=True).convert("RGB").save(OUT / "apple-touch-icon.png", optimize=True)
    ball(192).save(OUT / "icon-192.png", optimize=True)
    ball(512).save(OUT / "icon-512.png", optimize=True)
    ball(512, maskable=True).save(OUT / "icon-maskable-512.png", optimize=True)
    ball(32).save(OUT / "favicon-32.png", optimize=True)
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
