"""Plan which nozzle prints each filament on a two-nozzle printer (Bambu X2D / H2D) so that colour
changes — and the purge each one costs — are as few as possible.

On a single nozzle every colour change unloads the filament, loads the next one and flushes the old colour
out (the prime tower): on the X2D that is about two minutes and a few grams, every time. With two nozzles
a filament that keeps its own nozzle never needs flushing: switching nozzles only parks one and wakes the
other. The X2D's track switch lets any AMS slot feed either nozzle, so the assignment is free; Bambu
Studio's "Auto For Flush" mode picks one heuristically. This module scores assignments directly from the
figure's paint:

1. `layer_use`: which filaments each layer prints — every (sub-)triangle of the paint marks the layers it
   spans (split paint is expanded to its leaves).
2. `simulate`: per nozzle, walk the layers keeping the loaded filament: a layer whose set on that nozzle
   is S costs |S| − 1 changes if the loaded filament is in S (it prints first), else |S|; the layer ends
   on a filament the next layer also needs when there is one. Each change costs the flush volume between
   the two colours (the project's flush matrix). A layer that uses both nozzles costs a nozzle switch.
3. `plan`: every assignment (exhaustive up to 12 filaments, else greedy + single moves), lowest
   estimated time = changes × CHANGE_S + switches × SWITCH_S, then lowest flush volume.

Bambu Studio's own "Auto For Flush" assignment is as good or better than this plan (measured: equal on
a 5-colour figure, 9 min better on 7 colours, within this model's error), so exports keep it; the plan
is for estimates — what each colour costs, one nozzle against two.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .model3mf import Part
from .paint import leaves, parse_tree

# Cost model fitted to Bambu CLI slices on the user's X2D project (PETG, 0.16 mm, prime tower):
# the Manex Squirtle at 1, 2, 3, 4 and 7 colours (extra time / grams over the 1-colour print; predictions
# within 0.4 h and 12 g) — NOTES.md "Toolchange model". Changes and flush volume rise together, so a
# change's flush time is folded into CHANGE_S; grams keep the colour-pair flush volumes.
CHANGE_S = 104.6            # one AMS filament change on a nozzle, flush included
SWITCH_S = 18.0             # one nozzle switch (incl. its prime-tower share)
PURGE_G_PER_MM3 = 0.00133   # grams per mm³ of flush volume (1.217 x the matrix volume at 1.09 g/cm³)
SWITCH_G = 0.092            # grams per nozzle switch (prime tower)
EXHAUSTIVE = 12
BEST_JUMP = 50              # best_value_count: an added colour costing this many more changes is a jump


def best_value_count(rows: list[dict], min_jump: int = BEST_JUMP) -> int | None:
    """The palette size just before the biggest jump in filament changes (the knee of the cost curve), from
    rows {count, changes} in count order; None when no added colour costs at least `min_jump` more changes.

    Manex Squirtle: 0, 0, 29, 159, 180, 287, 287 changes for 1-7 colours -> 3 (blue / cream / brown, 2.35 h;
    the 4th colour, the black eyes, adds 130 changes and ~3 h)."""
    best: int | None = None
    jump = min_jump - 1
    for prev, cur in zip(rows, rows[1:]):
        d = int(cur["changes"]) - int(prev["changes"])
        if d > jump:
            jump, best = d, int(prev["count"])
    return best


def layer_use(parts: list[Part], object_extruder: int, n_filaments: int, layer_height: float,
              first_layer: float | None = None) -> np.ndarray:
    """(layers, n_filaments) bool: which filaments each layer prints (final-mm parts, bed at z = 0)."""
    lo: list[np.ndarray] = []
    hi: list[np.ndarray] = []
    st: list[np.ndarray] = []
    for p in parts:
        if p.subtype != "normal_part":
            continue
        base = p.extruder or object_extruder
        tri = p.vertices[p.faces]
        state = np.full(len(p.faces), base, np.int64)
        split: list[int] = []
        for i, attrs in p.tri_extra.items():
            if 'paint_color="' not in attrs:
                continue
            tree = parse_tree(attrs.split('paint_color="', 1)[1].split('"', 1)[0])
            if isinstance(tree, int):
                state[i] = tree or base
            else:
                split.append(i)
        keep = np.ones(len(p.faces), bool)
        keep[split] = False
        lo.append(tri[keep, :, 2].min(axis=1))
        hi.append(tri[keep, :, 2].max(axis=1))
        st.append(state[keep])
        for i in split:
            code = p.tri_extra[i].split('paint_color="', 1)[1].split('"', 1)[0]
            for corners, s in leaves(parse_tree(code), list(tri[i])):
                z = np.asarray(corners)[:, 2]
                lo.append(np.array([z.min()]))
                hi.append(np.array([z.max()]))
                st.append(np.array([s or base]))
    return layer_use_arrays(np.concatenate(lo), np.concatenate(hi), np.concatenate(st), n_filaments, layer_height,
                            first_layer)


def layer_use_arrays(zlo: np.ndarray, zhi: np.ndarray, state: np.ndarray, n_filaments: int, layer_height: float,
                     first_layer: float | None = None) -> np.ndarray:
    """`layer_use` from (z bottom, z top, 1-based filament) per printed (sub-)triangle."""
    first = layer_height if first_layer is None else first_layer
    top = float(zhi.max())
    n_layers = max(1, int(np.ceil((top - first) / layer_height)) + 1)

    def layer(z: np.ndarray) -> np.ndarray:
        return np.clip(np.where(z <= first, 0, np.ceil((z - first) / layer_height)), 0, n_layers - 1).astype(int)

    a, b = layer(zlo), layer(zhi)
    use = np.zeros((n_layers + 1, n_filaments), np.int64)
    ok = (state >= 1) & (state <= n_filaments)
    # Difference array: +1 at a triangle's first layer, -1 after its last.
    np.add.at(use, (a[ok], state[ok] - 1), 1)
    np.add.at(use, (b[ok] + 1, state[ok] - 1), -1)
    return np.cumsum(use, axis=0)[:n_layers] > 0


@dataclass
class Estimate:
    nozzle_of: list[int]              # 1-based nozzle per filament (Bambu filament_map)
    changes: int                      # AMS filament changes (flushes)
    switches: int                     # nozzle switches
    purge_mm3: float
    seconds: float                    # estimated time those cost
    per_nozzle: list[int] = field(default_factory=list)   # changes per nozzle

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["purge_mm3"] = round(self.purge_mm3)
        d["seconds"] = round(self.seconds)
        return d


def layer_lists(use: np.ndarray) -> list[tuple[int, ...]]:
    """The filaments (0-based, ascending) each layer of `use` prints: computed once per `plan`, so the many
    assignments it simulates walk plain tuples instead of indexing numpy rows per layer."""
    return [tuple(int(f) for f in np.flatnonzero(row)) for row in np.asarray(use, dtype=bool)]


def simulate(use: np.ndarray, nozzle_of: list[int], flush: np.ndarray, n_nozzles: int = 2,
             layers: list[tuple[int, ...]] | None = None) -> Estimate:
    """Changes, nozzle switches and purge for one assignment (module doc, step 2).

    `flush[a, b]` is the purge volume (mm³) going from filament a to b (0-based); `layers` = `layer_lists(use)`
    when the caller has it."""
    layers = layer_lists(use) if layers is None else layers
    per = [0] * n_nozzles
    purge = 0.0
    for n in range(n_nozzles):
        mine = [[f for f in layer if nozzle_of[f] == n + 1] for layer in layers]
        loaded = -1
        for li, here in enumerate(mine):
            if not here:
                continue
            nxt = mine[li + 1] if li + 1 < len(mine) else []
            order = sorted(here, key=lambda f: (f != loaded, f in nxt))   # loaded first, a carry-over last
            for f in order:
                if f != loaded:
                    if loaded >= 0:
                        per[n] += 1
                        purge += float(flush[loaded, f])
                    loaded = f
    # A nozzle switch on every layer that prints from more than one nozzle.
    switches = sum(1 for layer in layers if len({nozzle_of[f] for f in layer}) > 1)
    changes = sum(per)
    secs = changes * CHANGE_S + switches * SWITCH_S
    return Estimate(list(nozzle_of), changes, switches, purge, secs, per)


def _better(a: Estimate, b: Estimate) -> bool:
    """Less estimated time; at equal time, less flush volume."""
    return (round(a.seconds, 3), round(a.purge_mm3, 3)) < (round(b.seconds, 3), round(b.purge_mm3, 3))


def grams(est: Estimate) -> float:
    """Estimated filament spent on colour changes (flushes + prime tower share)."""
    return est.purge_mm3 * PURGE_G_PER_MM3 + est.switches * SWITCH_G


def plan(use: np.ndarray, flush: np.ndarray, n_nozzles: int = 2,
         allowed: list[set[int]] | None = None) -> tuple[Estimate, Estimate]:
    """(best assignment, everything on nozzle 1) — `allowed[f]` limits filament f to some nozzles."""
    n = use.shape[1]
    allowed = allowed or [set(range(1, n_nozzles + 1)) for _ in range(n)]
    layers = layer_lists(use)
    single = simulate(use, [min(a) for a in allowed], flush, n_nozzles, layers)
    best = single
    if n <= EXHAUSTIVE:
        for combo in itertools.product(*[sorted(a) for a in allowed]):
            if combo[0] != 1 and all(1 in a for a in allowed):
                continue                       # mirror images of an assignment cost the same
            est = simulate(use, list(combo), flush, n_nozzles, layers)
            if _better(est, best):
                best = est
    else:
        cur = list(single.nozzle_of)
        improved = True
        while improved:
            improved = False
            for f in range(n):
                for nz in sorted(allowed[f]):
                    if nz == cur[f]:
                        continue
                    trial = cur[:f] + [nz] + cur[f + 1:]
                    est = simulate(use, trial, flush, n_nozzles, layers)
                    if _better(est, best):
                        best, cur, improved = est, trial, True
    return best, single


def flush_matrix(settings: dict, n: int) -> np.ndarray:
    """(n, n) purge volumes (mm³) from a Bambu project's flush_volumes_matrix (first extruder's block)."""
    mat = settings.get("flush_volumes_matrix") or []
    if len(mat) >= n * n:
        m = np.array([float(x) for x in mat[:n * n]]).reshape(n, n)
    else:
        m = np.full((n, n), 150.0)
    mult = settings.get("flush_multiplier") or ["1"]
    np.fill_diagonal(m, 0.0)
    return m * float(mult[0] if isinstance(mult, list) else mult)


def nozzle_count(settings: dict) -> int:
    """Nozzles of the project's printer (2 on an X2D / H2D); 1 when the settings do not say."""
    noz = settings.get("nozzle_diameter") or []
    return max(1, len(noz)) if isinstance(noz, list) else 1
