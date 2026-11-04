"""The steel the figure's magnets grip, glued to the ball's inner floor: a solid steel disc (the default) or a
washer (a disc with a hole).

Values are nominal catalogue dimensions in mm. Measure the steel you have and pass a custom one if it differs:
the thickness sets the floor height; the diameter (and a washer's hole) sets where the magnets may sit. On a
solid disc a magnet holds anywhere it lies fully on the steel, its middle included; a washer's hole has none.
Stainless (A2/304) is not magnetic: buy plain or zinc-plated steel.

Thin discs (0.25-0.5 mm, sold as "metal discs for magnets") keep the floor low, so the figure fits a little
larger; a magnet holds less on very thin steel than on a thick washer (the steel saturates), but a figure is
light.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

INCH = 25.4


@dataclass(frozen=True)
class Steel:
    """A flat steel disc or washer: outer diameter, hole diameter (0 = solid disc) and thickness, mm."""

    key: str
    label: str
    od: float
    id: float
    thickness: float

    @property
    def solid(self) -> bool:
        """A disc with no hole: magnets may sit anywhere on it, its middle too."""
        return self.id <= 0

    def annulus(self, magnet_d: float) -> tuple[float, float]:
        """Radii between which a magnet of diameter `magnet_d` lies fully on steel (from 0 on a solid disc)."""
        r_m = magnet_d / 2
        return (0.0 if self.solid else self.id / 2 + r_m), self.od / 2 - r_m

    def to_dict(self) -> dict[str, float | str | bool]:
        return {**asdict(self), "solid": self.solid}


def _disc(key: str, d_in: float, t_in: float) -> Steel:
    """A steel disc sold in inches (diameter, thickness)."""
    d, t = round(d_in * INCH, 2), round(t_in * INCH, 3)
    return Steel(key, f"Steel disc {d:g} × {t:g} mm ({d_in:g}″ × {t_in:g}″)", d, 0.0, t)


# Thin steel discs (the user's, 2026-11-04: 0.8" x 0.01" and 0.59" x 0.02"); then DIN 9021 ("large outer
# diameter") metric washers and two common US fender washers (US hole/thickness are typical values: listings
# give OD and a nominal 1/4" or #10 bolt size; Kent lists 0.06" thickness for 1/4" x 1").
CATALOGUE: dict[str, Steel] = {w.key: w for w in [
    _disc("D20", 0.8, 0.01),
    _disc("D15", 0.59, 0.02),
    Steel("M5", "Washer DIN 9021 M5 (15 × 5.3 × 1.2)", 15.0, 5.3, 1.2),
    Steel("M6", "Washer DIN 9021 M6 (18 × 6.4 × 1.6)", 18.0, 6.4, 1.6),
    Steel("M8", "Washer DIN 9021 M8 (24 × 8.4 × 2.0)", 24.0, 8.4, 2.0),
    Steel("US10x3/4", "Washer US fender #10 × 3/4\" (19.1 × 5.6 × 1.2)", 19.05, 5.56, 1.24),
    Steel("US1/4x1", "Washer US fender 1/4\" × 1\" (25.4 × 7.1 × 1.5)", 25.4, 7.14, 1.52),
]}

DEFAULT_STEEL = "D20"


def get_steel(key_or_spec: str) -> Steel:
    """Catalogue key ("D20", "M6") or a custom "OD,ID,T" string in mm (ID 0 = a solid disc)."""
    if key_or_spec in CATALOGUE:
        return CATALOGUE[key_or_spec]
    try:
        od, id_, t = (float(x) for x in key_or_spec.split(","))
    except ValueError as exc:
        raise ValueError(f"unknown steel {key_or_spec!r}: use one of {list(CATALOGUE)} or 'OD,ID,T'") from exc
    if not (od > id_ >= 0 and t > 0):
        raise ValueError(f"steel {key_or_spec!r}: need OD > ID >= 0 and T > 0")
    if id_ > 0:
        return Steel(key_or_spec, f"custom washer ({od:g} × {id_:g} × {t:g})", od, id_, t)
    return Steel(key_or_spec, f"custom disc ({od:g} × {t:g})", od, 0.0, t)
