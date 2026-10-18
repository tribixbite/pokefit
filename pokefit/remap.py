"""Map a figure's filaments onto another project's filament slots.

Bambu Studio's "Import geometry only" (dragging a 3MF into an open project)
keeps every part's filament number and the per-triangle paint, but those
numbers then mean the open project's slots: filament 2 of the figure prints
with whatever is loaded as filament 2 there. So an export meant for importing
renumbers the figure's filaments to the project's slots of the closest colour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model3mf import Part
from .paint import decode_weights, remap_states

NEW_SLOT_DELTA_E = 25.0     # CIE76 ΔE above which no slot counts as "the same colour"


@dataclass
class SlotMatch:
    source: int             # figure filament number (1-based)
    colour: str             # its colour
    target: int             # project filament number (1-based) it is mapped to
    target_colour: str      # colour of that slot ("" for a slot the import will add)
    delta_e: float          # colour difference (CIE76); large = not really the same colour
    new: bool = False       # beyond the project's slots: Bambu adds a filament, set its colour by hand
    manual: bool = False    # chosen by the user

    def to_dict(self) -> dict:
        return asdict(self)


def _lab(hex_colour: str) -> np.ndarray:
    """sRGB hex (#RRGGBB[AA]) -> CIELAB (D65)."""
    c = hex_colour.lstrip("#")[:6].ljust(6, "0")
    rgb = np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float) / 255
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: str, b: str) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b)))


def used_filaments(parts: list[Part], object_extruder: int) -> list[int]:
    """Filament numbers the parts print with: part/object assignment plus painted states."""
    used: set[int] = set()
    for p in parts:
        used.add(p.extruder or object_extruder)
        for attrs in p.tri_extra.values():
            if 'paint_color="' in attrs:
                code = attrs.split('paint_color="', 1)[1].split('"', 1)[0]
                used.update(k for k in decode_weights(code) if k > 0)
    return sorted(used)


def match_slots(used: list[int], colours: list[str], targets: list[str],
                manual: dict[int, int] | None = None) -> list[SlotMatch]:
    """Closest target slot per used filament; colours beyond NEW_SLOT_DELTA_E get new slots after the last.

    `manual` (figure filament -> project filament, 1-based) overrides the choice;
    a number beyond the project's slots is a new slot.
    """
    manual = manual or {}
    out: list[SlotMatch] = []
    extra = len(targets)
    for k in used:
        col = colours[(k - 1) % len(colours)] if colours else "#808080"
        if k in manual:
            t = int(manual[k])
            tc = targets[t - 1] if 1 <= t <= len(targets) else ""
            out.append(SlotMatch(k, col, t, tc, delta_e(col, tc) if tc else 0.0, not tc, True))
            continue
        if targets:
            d = [delta_e(col, t) for t in targets]
            j = int(np.argmin(d))
            if d[j] <= NEW_SLOT_DELTA_E:
                out.append(SlotMatch(k, col, j + 1, targets[j], round(d[j], 1)))
                continue
        extra += 1
        out.append(SlotMatch(k, col, extra, "", 0.0, True))
    return out


def remap_parts(parts: list[Part], object_extruder: int, matches: list[SlotMatch]) -> tuple[list[Part], int]:
    """Parts (and the object's default filament) renumbered per `matches`; paint splits are kept exactly."""
    m = {s.source: s.target for s in matches}
    out = []
    for p in parts:
        extra = {}
        for i, attrs in p.tri_extra.items():
            if 'paint_color="' in attrs:
                head, rest = attrs.split('paint_color="', 1)
                code, tail = rest.split('"', 1)
                attrs = f'{head}paint_color="{remap_states(code, m)}"{tail}'
            extra[i] = attrs
        # A part without its own filament inherits the object's; give it the mapped one explicitly.
        ext = m.get(p.extruder or object_extruder, p.extruder or object_extruder)
        out.append(Part(p.name, ext, p.vertices, p.faces, extra, p.subtype))
    return out, m.get(object_extruder, object_extruder)
