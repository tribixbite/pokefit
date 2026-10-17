"""Steel washers the figure's magnets can grip, glued to the ball's inner floor.

Values are nominal catalogue dimensions in mm. Measure the washer you buy and
pass a custom one if it differs: thickness sets the floor height, OD/ID set
where the magnets may sit. Stainless (A2/304) washers are not magnetic — buy
zinc-plated or plain steel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Washer:
    """A flat washer: outer/inner diameter and thickness, mm."""

    key: str
    label: str
    od: float
    id: float
    thickness: float

    def annulus(self, magnet_d: float) -> tuple[float, float]:
        """Radii between which a magnet of diameter `magnet_d` lies fully on steel."""
        r_m = magnet_d / 2
        return self.id / 2 + r_m, self.od / 2 - r_m

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


# DIN 9021 ("large outer diameter") metric washers, and two common US fender
# washers (US ID/thickness are typical values: listings give OD and a nominal
# 1/4" or #10 bolt size; Kent lists 0.06" thickness for 1/4" x 1").
CATALOGUE: dict[str, Washer] = {w.key: w for w in [
    Washer("M5", "DIN 9021 M5 (15 × 5.3 × 1.2)", 15.0, 5.3, 1.2),
    Washer("M6", "DIN 9021 M6 (18 × 6.4 × 1.6)", 18.0, 6.4, 1.6),
    Washer("M8", "DIN 9021 M8 (24 × 8.4 × 2.0)", 24.0, 8.4, 2.0),
    Washer("US10x3/4", "US fender #10 × 3/4\" (19.1 × 5.6 × 1.2)", 19.05, 5.56, 1.24),
    Washer("US1/4x1", "US fender 1/4\" × 1\" (25.4 × 7.1 × 1.5)", 25.4, 7.14, 1.52),
]}

DEFAULT_WASHER = "M6"


def get_washer(key_or_spec: str) -> Washer:
    """Catalogue key ("M6") or a custom "OD,ID,T" string in mm."""
    if key_or_spec in CATALOGUE:
        return CATALOGUE[key_or_spec]
    try:
        od, id_, t = (float(x) for x in key_or_spec.split(","))
    except ValueError as exc:
        raise ValueError(f"unknown washer {key_or_spec!r}: use one of {list(CATALOGUE)} or 'OD,ID,T'") from exc
    if not (od > id_ >= 0 and t > 0):
        raise ValueError(f"washer {key_or_spec!r}: need OD > ID >= 0 and T > 0")
    return Washer(key_or_spec, f"custom ({od:g} × {id_:g} × {t:g})", od, id_, t)
