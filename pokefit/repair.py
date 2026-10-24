"""Repair meshes so every edge joins exactly two faces, as Bambu Studio's "Fix model" does.

Bambu Studio flags a part whose edges are shared by more than two triangles
("non-manifold edges") or by only one (open edges), and offers to repair it.
pokefit's own steps can make such edges too: a boolean result welded where two
surfaces touch along a line (a pocket's rim, a pressed base meeting the flat
base), grid cells of the flat base that touch only at a corner, a simplified
mesh folding onto itself. Files often arrive with them as well. So every part
is repaired as the file loads and after every boolean, and the report says what
was fixed:

1. weld: vertices at the same position (to 1e-6 mm, the 3MF writer's
   precision) are one vertex;
2. drop degenerate triangles and folds (two triangles on the same corners,
   `simplify.clean_index`);
3. separate: around an edge shared by 2k triangles, pair each triangle with its
   neighbour across the solid between them (sorted by angle about the edge), so
   two solids touching along a line become two closed surfaces; a vertex where
   surfaces only touch gets one copy per surface, moved `NUDGE` toward its own
   triangles so welding by position keeps them apart;
4. winding: neighbours whose shared edge runs the same way are turned to agree
   (the majority of each shell keeps its order), and a shell enclosing negative
   volume is turned inside out;
5. holes: each loop of open edges is closed by a fan from its centre.

Per-triangle attributes (Bambu paint) stay with their triangles; a turned
triangle's split paint is re-laid for its new corner order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .simplify import clean_index, group_rows

if TYPE_CHECKING:
    from .model3mf import Part

WELD = 6          # decimals of a mm: vertices this close are one (the 3MF writer prints 6)
NUDGE = 1e-3      # mm a separated vertex copy moves toward its own triangles


@dataclass(frozen=True)
class RepairInfo:
    removed: int = 0      # degenerate and folded triangles dropped
    split: int = 0        # edges shared by more than two triangles, separated
    flipped: int = 0      # triangles turned to face the same way as their neighbours / the outside
    filled: int = 0       # holes closed
    open_edges: int = 0   # open edges found (before filling)
    pinched: int = 0      # places where separate surfaces met at the same point, moved apart

    def __bool__(self) -> bool:
        return bool(self.removed or self.split or self.flipped or self.filled or self.pinched)

    def __add__(self, other: "RepairInfo") -> "RepairInfo":
        return RepairInfo(self.removed + other.removed, self.split + other.split, self.flipped + other.flipped,
                          self.filled + other.filled, self.open_edges + other.open_edges,
                          self.pinched + other.pinched)

    def to_dict(self) -> dict[str, Any]:
        return {"removed": self.removed, "split": self.split, "flipped": self.flipped, "filled": self.filled,
                "open_edges": self.open_edges, "pinched": self.pinched}

    def text(self) -> str:
        """What was fixed, as a list for a sentence ("" when nothing)."""
        bits = []
        if self.split:
            bits.append(f"separated {self.split:,} edges shared by more than two triangles")
        if self.filled:
            bits.append(f"closed {self.filled:,} holes ({self.open_edges:,} open edges)")
        if self.pinched:
            bits.append(f"moved apart {self.pinched:,} points where separate surfaces touched")
        if self.flipped:
            bits.append(f"turned {self.flipped:,} inside-out triangles")
        if self.removed:
            bits.append(f"removed {self.removed:,} degenerate or folded triangles")
        return ", ".join(bits)


def weld(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices at the same position (to WELD decimals) merged; each keeps its first occurrence's exact position."""
    if not len(vertices):
        return vertices, faces
    group, first, _ = group_rows(_grid(vertices))
    return vertices[first], group[faces]


def _grid(vertices: np.ndarray) -> np.ndarray:
    """Positions as integers of 10^-WELD mm: equal rows = the same place."""
    return np.round(np.asarray(vertices, np.float64) * 10.0 ** WELD).astype(np.int64)


def _pairs(v: np.ndarray, f: np.ndarray, he: np.ndarray, u: np.ndarray, w: np.ndarray
           ) -> tuple[list[tuple[int, int]], list[int]]:
    """Pairs of half-edges (indices into the 3F list) around one edge shared by more than two triangles.

    About the edge lo->hi, a triangle running lo->hi has the solid at smaller angles, one running hi->lo at larger
    angles; so the solid lies between a hi->lo triangle and the next lo->hi one counter-clockwise, and those two
    are paired. Returns the pairs and the half-edges left over (an odd count, or a mixed-up winding)."""
    lo = min(int(u[he[0]]), int(w[he[0]]))
    hi = max(int(u[he[0]]), int(w[he[0]]))
    d = v[hi] - v[lo]
    d = d / (np.linalg.norm(d) or 1.0)
    third = f.reshape(-1)[3 * (he // 3) + (he % 3 + 2) % 3]
    r = v[third] - v[lo]
    r -= np.outer(r @ d, d)
    e1 = r[0] / (np.linalg.norm(r[0]) or 1.0)
    e2 = np.cross(d, e1)
    ang = np.arctan2(r @ e2, r @ e1)
    forward = u[he] == lo
    order = np.argsort(ang, kind="stable")
    ring = [(int(he[i]), bool(forward[i])) for i in order]
    n = len(ring)
    out: list[tuple[int, int]] = []
    used = [False] * n
    for i in range(n):
        h, fwd = ring[i]
        if used[i] or fwd:
            continue
        for s in range(1, n):
            j = (i + s) % n
            if not used[j] and ring[j][1]:
                used[i] = used[j] = True
                out.append((h, ring[j][0]))
                break
    return out, [ring[i][0] for i in range(n) if not used[i]]


def _links(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """(links, same, split, open half-edges, odd faces, surplus faces): pairs of half-edges (h = 3 * face + k runs
    from corner k to corner k + 1) joined across an edge, whether each pair runs the same way (one face is turned),
    how many edges are shared by more than two faces (paired by `_pairs`), the faces left unpaired at such an edge,
    and those plus the faces of every pair but the first there."""
    nf, nv = len(f), len(v)
    u = f.reshape(-1)
    w = f[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(u, w) * nv + np.maximum(u, w)
    order = np.argsort(key)
    srt = key[order]
    start = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1]])
    count = np.diff(np.append(start, len(srt)))
    two = start[count == 2]
    links = [np.column_stack([order[two], order[two + 1]])]
    split = 0
    odd: list[int] = []
    surplus: list[int] = []
    for s, c in zip(start[count > 2].tolist(), count[count > 2].tolist()):
        split += 1
        pairs, left = _pairs(v, f, order[s:s + c], u, w)
        odd += [h // 3 for h in left]
        surplus += [h // 3 for pr in pairs[1:] for h in pr]
        if pairs:
            links.append(np.asarray(pairs, np.int64))
    link = np.concatenate(links)
    return (link, u[link[:, 0]] == u[link[:, 1]], split, 3 * nf - 2 * len(link),
            np.unique(np.asarray(odd, np.int64)), np.unique(np.asarray(odd + surplus, np.int64)))


def repair(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             RepairInfo]:
    """(vertices, faces, src, turned, info): the repaired mesh, the input face each output face came from (-1 for
    a face closing a hole) and whether its corner order was reversed.

    A mesh whose indices are already sound (a boolean's result: every edge on two faces) keeps its topology; only
    vertices that share a position (surfaces touching with no thickness between them) are moved apart. Otherwise
    (a triangle soup, a broken file) vertices are welded by position first and every step runs."""
    v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    keep = clean_index(f)
    f, src = f[keep], keep.astype(np.int64)
    removed = len(faces) - len(keep)
    if not len(f):
        return v, f, src, np.zeros(0, bool), RepairInfo(removed=removed)
    link, same, split, n_open, _, _ = _links(v, f)
    if split or n_open:
        v, f = weld(v, f)
        keep = clean_index(f)
        f, src = f[keep], src[keep]
        link, same, split, n_open, odd, _ = _links(v, f)
        found = split
        if len(odd):
            # A face left over at an edge of three or five (a stray sheet): drop it, its hole is closed below.
            f, src = _without(f, src, odd)
            link, same, split, n_open, _, _ = _links(v, f)
        if split or n_open:
            v, f = _separate(v, f, link, same)
            link, same, split, n_open, _, surplus = _links(v, f)
            if split:
                # Still on four faces after separating: a sheet folding back to touch itself, so both ends stay
                # shared. Keep one pair there and drop the rest (their holes are closed below).
                f, src = _without(f, src, surplus)
                v, f = _separate(v, f, *_links(v, f)[:2])
                link, same, split, n_open, _, _ = _links(v, f)
        split = max(split, found)
        removed = len(faces) - len(f)
    flip = _consistent(f, link, same)
    if flip.any():
        f[flip] = f[flip][:, [0, 2, 1]]
    filled = 0
    if n_open:
        v, f, src, flip, filled = _fill(v, f, src, flip)
    out = _inside_out(v, f, link, len(f) - len(src[src >= 0]))
    if out.any():
        f[out] = f[out][:, [0, 2, 1]]
        flip ^= out
    v, pinched = _unpinch(v, f)
    return v, f, src, flip, RepairInfo(removed, split, int(flip.sum()), filled, n_open, pinched)


def _without(f: np.ndarray, src: np.ndarray, drop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.setdiff1d(np.arange(len(f)), drop)
    return f[keep], src[keep]


def _separate(v: np.ndarray, f: np.ndarray, link: np.ndarray, same: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One vertex per fan of linked triangles: corners at the same vertex are joined across each linked edge; a
    vertex whose corners fall in several groups gets a copy per group (at the same place: `_unpinch` moves them)."""
    nf = len(f)
    u = f.reshape(-1)
    nxt = 3 * (np.arange(3 * nf) // 3) + (np.arange(3 * nf) % 3 + 1) % 3
    a, b = link[:, 0], link[:, 1]
    # Corners joined across a link: a's start with b's end (opposite runs) or start (same run), and likewise.
    ca = np.concatenate([a, nxt[a]])
    cb = np.concatenate([np.where(same, b, nxt[b]), np.where(same, nxt[b], b)])
    g = coo_matrix((np.ones(len(ca), np.int8), (ca, cb)), shape=(3 * nf, 3 * nf))
    _, label = connected_components(g, directed=False)
    _, first, corner_group = np.unique(label, return_index=True, return_inverse=True)
    return v[u[first]], corner_group.reshape(-1, 3).astype(np.int64)


def _unpinch(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, int]:
    """(vertices, places): distinct vertices at the same position (to WELD decimals) moved NUDGE toward their own
    triangles, so a slicer that welds by position keeps the surfaces apart; any still together (their triangles
    lie the same way) are stepped apart along their normal."""
    inv, _, cnt = group_rows(_grid(v))
    multi = cnt[inv] > 1
    if not multi.any():
        return v, 0
    v = v.copy()
    tri = v[f]
    centre = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros_like(v)
    nrm = np.zeros_like(v)
    np.add.at(acc, f.reshape(-1), np.repeat(centre, 3, axis=0))
    np.add.at(nrm, f.reshape(-1), np.repeat(normal, 3, axis=0))
    uses = np.bincount(f.reshape(-1), minlength=len(v)).astype(np.float64)
    toward = acc[multi] / np.maximum(uses[multi], 1.0)[:, None] - v[multi]
    length = np.linalg.norm(toward, axis=1)
    v[multi] += toward * (NUDGE / np.where(length > 0, length, 1.0))[:, None]
    inv2, _, cnt2 = group_rows(_grid(v))
    still = np.nonzero(cnt2[inv2] > 1)[0]
    if len(still):
        order = still[np.argsort(inv2[still], kind="stable")]
        group = inv2[order]
        rank = np.arange(len(order)) - np.searchsorted(group, group)          # 0, 1, ... within each place
        n = nrm[order] / np.maximum(np.linalg.norm(nrm[order], axis=1), 1e-30)[:, None]
        v[order] -= n * (rank * NUDGE)[:, None]                               # inward, one step per copy
    return v, int((cnt > 1).sum())


def _consistent(f: np.ndarray, link: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Which faces to turn so that neighbours agree; each shell keeps the order most of its faces have."""
    nf = len(f)
    flip = np.zeros(nf, bool)
    if not same.any():
        return flip
    fa, fb = link[:, 0] // 3, link[:, 1] // 3
    _, shell = connected_components(coo_matrix((np.ones(len(fa)), (fa, fb)), shape=(nf, nf)), directed=False)
    # Two-colour the face graph: across a same-way link the faces differ. BFS from a virtual root joined to one
    # face of each shell (weight 3 = no constraint), so one traversal covers every shell.
    _, heads = np.unique(shell, return_index=True)
    root = nf
    rows = np.concatenate([fa, fb, np.full(len(heads), root)])
    cols = np.concatenate([fb, fa, heads])
    data = np.concatenate([same + 1, same + 1, np.full(len(heads), 3)]).astype(np.int8)
    g = coo_matrix((data, (rows, cols)), shape=(nf + 1, nf + 1)).tocsr()
    order, pred = breadth_first_order(g, root, directed=False, return_predecessors=True)
    order = order[1:]
    p = pred[order]
    rel = np.asarray(g[p, order]).ravel()
    state = np.zeros(nf + 1, bool)
    for node, parent, r in zip(order.tolist(), p.tolist(), rel.tolist()):
        state[node] = state[parent] ^ (r == 2)
    flip = state[:nf]
    turned = np.bincount(shell, weights=flip)
    size = np.bincount(shell)
    return flip ^ (turned * 2 > size)[shell]


def _inside_out(v: np.ndarray, f: np.ndarray, link: np.ndarray, n_fill: int) -> np.ndarray:
    """Faces of shells (faces joined across linked edges; the last `n_fill` faces, closing holes, join the shell
    of a face sharing their first corner) that enclose a negative volume."""
    nf = len(f)
    fa, fb = link[:, 0] // 3, link[:, 1] // 3
    rows, cols = [fa], [fb]
    if n_fill:
        extra = np.arange(nf - n_fill, nf)
        first_face = np.full(len(v), -1, np.int64)
        first_face[f[:nf - n_fill].reshape(-1)] = np.repeat(np.arange(nf - n_fill), 3)
        near = first_face[f[extra, 0]]
        rows.append(extra[near >= 0])
        cols.append(near[near >= 0])
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, shell = connected_components(coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(nf, nf)), directed=False)
    tri = v[f]
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    return (np.bincount(shell, weights=vol6) < 0)[shell]


def _fill(v: np.ndarray, f: np.ndarray, src: np.ndarray, flip: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Close each loop of open edges (a -> b with no b -> a) with a fan from its centre (one triangle for a
    three-edge loop)."""
    nv = len(v)
    a = f.reshape(-1)
    b = f[:, [1, 2, 0]].reshape(-1)
    open_ = ~np.isin(a * nv + b, b * nv + a)
    a, b = a[open_], b[open_]
    # The hole runs the other way round: b -> a. Next hole edge starts where this one ends (at a).
    out_of = {int(s): int(t) for s, t in zip(b.tolist(), a.tolist())}
    seen: set[int] = set()
    new_v: list[np.ndarray] = []
    new_f: list[list[int]] = []
    loops = 0
    for s0 in out_of:
        if s0 in seen:
            continue
        loop = [s0]
        seen.add(s0)
        cur = out_of[s0]
        while cur != s0 and cur in out_of and cur not in seen and len(loop) <= len(out_of):
            loop.append(cur)
            seen.add(cur)
            cur = out_of[cur]
        if cur != s0 or len(loop) < 3:
            continue                        # not a closed loop (a tangle the steps above could not sort out)
        loops += 1
        if len(loop) == 3:
            new_f.append(loop)
            continue
        c = nv + len(new_v)
        new_v.append(v[loop].mean(axis=0))
        new_f += [[loop[i], loop[(i + 1) % len(loop)], c] for i in range(len(loop))]
    if not new_f:
        return v, f, src, flip, 0
    v2 = np.vstack([v, *new_v]) if new_v else v
    f2 = np.vstack([f, np.asarray(new_f, np.int64)])
    return (v2, f2, np.concatenate([src, np.full(len(new_f), -1)]), np.concatenate([flip, np.zeros(len(new_f), bool)]),
            loops)


def repaired(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Just the repaired (vertices, faces), for a mesh with no per-triangle attributes (a boolean's result)."""
    v, f, _, _, _ = repair(vertices, faces)
    return v, f


def repair_part(part: "Part") -> tuple["Part", RepairInfo]:
    """`part` repaired (unchanged when it needs nothing beyond welding), its paint carried with its triangles."""
    from .model3mf import Part, remap_paint_attrs

    v, f, src, turned, info = repair(part.vertices, part.faces)
    if not info:
        return part, info
    extra: dict[int, str] = {}
    if part.tri_extra:
        for i, (s, t) in enumerate(zip(src.tolist(), turned.tolist())):
            a = part.tri_extra.get(s) if s >= 0 else None
            if a:
                extra[i] = remap_paint_attrs(a, (0, 2, 1)) if t else a
    return Part(part.name, part.extruder, v, f, extra, part.subtype), info


def repair_parts(parts: list["Part"]) -> tuple[list["Part"], RepairInfo]:
    """Every printing part repaired (modifiers and negative parts as they are)."""
    out: list[Part] = []
    total = RepairInfo()
    for p in parts:
        if p.subtype != "normal_part" or not len(p.faces):
            out.append(p)
            continue
        q, info = repair_part(p)
        out.append(q)
        total = total + info
    return out, total
