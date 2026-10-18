"""Check results shared by every analysis step (and serialised to the UI)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["pass", "warn", "fail", "info"]
_RANK = {"info": 0, "pass": 0, "warn": 1, "fail": 2}


@dataclass
class Check:
    id: str                       # stable machine id, e.g. "watertight:body"
    title: str                    # short human label
    status: Status
    message: str                  # one sentence with the measured numbers
    value: float | str | None = None
    highlight: list[list[float]] = field(default_factory=list)  # points (final mm, figure frame) to mark

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def worst(checks: list[Check]) -> Status:
    return max((c.status for c in checks), key=lambda s: _RANK[s], default="pass")
