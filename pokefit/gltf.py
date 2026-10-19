"""Read glTF / GLB scenes as one figure, with their colours as Bambu paint.

Colour in a glTF lives in a texture (most scanned and AI-generated models), in
vertex colours, or in flat materials. All three become per-triangle
multi-material paint, so the figure stays one solid part and slices as
multi-colour:

1. Every triangle is cut (in thought, not in the mesh) into 4^D equal
   sub-triangles exactly as Bambu's paint brush splits it (paint.split_children,
   3-side splits), and the colour under each one's centre is sampled. D is
   chosen so the whole model has at most LEAF_BUDGET samples.
2. The samples are over-clustered (k-means, OVER clusters) in CIELAB with
   lightness down-weighted, so baked-in shading does not count as a colour.
   `TextureColours` keeps those fine labels; a palette of any size is then the
   fine clusters merged closest-first: "auto" merges until all are at least
   MERGE_DE apart (at most MAX_COLOURS), a count merges until that many remain.
   Small but distinct areas (eyes, a mouth) survive because they are far from
   everything else.
3. Specks are removed: a colour that covers less than SPECK_SHARE of the surface
   in one connected patch joins the colour around it (each such patch costs a
   filament change on every layer it spans, for a fleck nobody would miss).
4. Each triangle gets a split tree (paint.encode_tree) whose identical
   sub-trees are merged, so a one-colour triangle is a single leaf and only
   triangles on a colour boundary are split.

glTF is Y-up and in metres; prints are Z-up in mm. The axes are turned (+90°
about X), and a scene no larger than 10 units is taken as metres (x1000). Many
exporters ignore the metre convention (LEGO exports are in mm-ish units), so a
larger scene keeps its units; the fit rescales the figure either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from .model3mf import DEFAULT_COLOURS, FigureObject, Part, Project
from .paint import Tree, encode_tree, split_children

GLTF_SUFFIXES = (".glb", ".gltf")
MAX_COLOURS = 8           # cap of the "auto" palette
MAX_PICK = 16             # most colours a user may ask for
LEAF_BUDGET = 1_500_000   # colour samples over the whole model (sets the split depth)
MAX_DEPTH = 3             # at most 64 sub-triangles per triangle (Bambu's brush goes finer)
OVER = 24                 # fine clusters
MERGE_DE = 15.0           # auto: merge clusters closer than this (CIELAB, lightness x L_WEIGHT)
L_WEIGHT = 0.5            # shading changes lightness more than hue: count it half
FIT_SAMPLES = 200_000     # samples the clustering is fitted on (area-weighted draw)
SPECK_SHARE = 2e-4        # a one-colour patch under this share of the surface is a speck

Sampler = Callable[[np.ndarray, np.ndarray], np.ndarray]   # (faces, barycentric (n,3)) -> sRGB (n,3)


def _flat(rgb: tuple[int, int, int]) -> Sampler:
    c = np.array(rgb, np.float64)
    return lambda fi, b: np.broadcast_to(c, (len(fi), 3))


def _image_rgb(img: Any) -> np.ndarray:
    """A texture as (H, W, 3) uint8, lightly median-filtered: JPEG noise would otherwise
    scatter stray samples into wrong clusters along colour edges."""
    from PIL import ImageFilter

    return np.asarray(img.convert("RGB").filter(ImageFilter.MedianFilter(3)))


def _sampler(mesh: trimesh.Trimesh) -> tuple[Sampler, bool]:
    """How to read this primitive's colour at points on its faces, and whether it varies
    across a face (texture or vertex colours) or is one colour per primitive."""
    vis = mesh.visual
    mat = getattr(vis, "material", None)
    uv = getattr(vis, "uv", None)
    if mat is not None:
        base = getattr(mat, "baseColorFactor", None)
        factor = np.asarray(base if base is not None else [255, 255, 255, 255], np.float64)[:3] / 255.0
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None and float(factor.max()) < 0.02:
            # Unlit exports (some AI generators): base colour black, colour in the emissive texture.
            emissive = getattr(mat, "emissiveTexture", None)
            if emissive is not None:
                ef = getattr(mat, "emissiveFactor", None)
                tex, factor = emissive, np.asarray(ef if ef is not None else [1, 1, 1], np.float64)[:3]
        if tex is not None and uv is not None and len(uv) == len(mesh.vertices):
            img = _image_rgb(tex).astype(np.float64) * np.clip(factor, 0, 1)
            h, w = img.shape[:2]
            uvf = np.asarray(uv, np.float64)
            faces = np.asarray(mesh.faces)

            def sample(fi: np.ndarray, b: np.ndarray) -> np.ndarray:
                p = np.einsum("ij,ijk->ik", b, uvf[faces[fi]])
                x = (np.mod(p[:, 0], 1.0) * (w - 1)).round().astype(np.intp)       # repeat wrap
                y = ((1.0 - np.mod(p[:, 1], 1.0)) * (h - 1)).round().astype(np.intp)  # glTF v is down
                return img[y, x]

            return sample, True
        try:
            c = np.asarray(mat.main_color, np.float64)[:3]   # factor, or a texture's mean without UVs
            return _flat((int(c[0]), int(c[1]), int(c[2]))), False
        except Exception:                                  # a material with neither
            pass
    if isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "vertex":
        vc = np.asarray(vis.vertex_colors, np.float64)[:, :3]
        faces = np.asarray(mesh.faces)
        return (lambda fi, b: np.einsum("ij,ijk->ik", b, vc[faces[fi]])), True
    if isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "face":
        fc = np.asarray(vis.face_colors, np.float64)[:, :3]
        return (lambda fi, b: fc[fi]), False
    hexc = DEFAULT_COLOURS[0].lstrip("#")
    return _flat((int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16))), False


def leaf_barycentrics(depth: int) -> np.ndarray:
    """(4^depth, 3, 3) corners, in barycentric coordinates, of the sub-triangles of a fully
    3-side-split tree of that depth, depth first in Bambu's children order."""
    tris = [list(np.eye(3))]
    for _ in range(depth):
        tris = [c for t in tris for c in split_children(t, 3, 0)]
    return np.array([np.array(t) for t in tris])


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB (D65), lightness scaled by L_WEIGHT."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([(116 * f[:, 1] - 16) * L_WEIGHT, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _nearest(lab: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest centre for each (weighted CIELAB) sample, in chunks."""
    out = np.empty(len(lab), np.intp)
    for s in range(0, len(lab), 262_144):
        out[s:s + 262_144] = np.argmin(((lab[s:s + 262_144, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    return out


def tree_of(labels: np.ndarray) -> Tree:
    """Split tree for one triangle's leaf labels (4^D, Bambu depth-first order; label = state)."""
    if (labels == labels[0]).all():
        return int(labels[0])
    q = len(labels) // 4
    return 3, 0, [tree_of(labels[i * q:(i + 1) * q]) for i in range(4)]


@dataclass
class TextureColours:
    """Fine colour clusters of a glTF's surface, from which a palette of any size is cut."""

    labels: np.ndarray        # (faces, leaves) fine cluster per sub-triangle (Bambu leaf order)
    lab: np.ndarray           # (k0, 3) weighted-CIELAB centre per fine cluster
    rgb: np.ndarray           # (k0, 3) mean sRGB per fine cluster
    weight: np.ndarray        # (k0,) surface share per fine cluster
    area: np.ndarray          # (faces,) triangle areas (file units)
    adjacency: np.ndarray     # (n, 2) face pairs sharing an edge
    exact: bool               # the fine clusters are the file's own flat colours
    _cache: dict[int | None, tuple[list[str], dict[int, str], int]] = field(default_factory=dict)

    @property
    def auto_count(self) -> int:
        return len(self.groups(None))

    def groups(self, count: int | None) -> list[list[int]]:
        """Fine clusters merged closest-first, largest group first: until `count` remain, or
        (count None) until all are MERGE_DE apart and at most MAX_COLOURS remain. Exact flat
        colours are only merged to meet a count or the cap."""
        groups = [[i] for i in range(len(self.lab)) if self.weight[i] > 0]
        centres = [self.lab[g[0]] for g in groups]
        weights = [float(self.weight[g[0]]) for g in groups]
        limit = MAX_COLOURS if count is None else max(1, min(count, MAX_PICK))
        while len(groups) > 1:
            d = np.linalg.norm(np.array(centres)[:, None] - np.array(centres)[None], axis=2)
            np.fill_diagonal(d, np.inf)
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            if len(groups) <= limit and (count is not None or self.exact or d[a, b] >= MERGE_DE):
                break
            wa, wb = weights[a], weights[b]
            centres[a] = (centres[a] * wa + centres[b] * wb) / (wa + wb)
            weights[a] = wa + wb
            groups[a] = groups[a] + groups[b]
            del groups[b], centres[b], weights[b]
        order = np.argsort(-np.array(weights), kind="stable")
        return [groups[i] for i in order]

    def paint(self, count: int | None = None) -> tuple[list[str], dict[int, str], int]:
        """(palette hex colours, tri_extra paint per face, base filament) for a palette size."""
        if count not in self._cache:
            groups = self.groups(count)
            to_group = np.zeros(len(self.lab), np.intp)
            for gi, g in enumerate(groups):
                to_group[g] = gi
            leaf = to_group[self.labels]
            leaf = self._despeckle(leaf, len(groups))
            n_leaf = leaf.shape[1]
            share = np.bincount(leaf.ravel(), weights=np.repeat(self.area / n_leaf, n_leaf), minlength=len(groups))
            base = int(np.argmax(share))
            extra = {i: f'paint_color="{encode_tree(tree_of(leaf[i] + 1))}"'
                     for i in np.nonzero((leaf != base).any(axis=1))[0].tolist()}
            colours = []
            for g in groups:
                w = self.weight[g]
                c = (self.rgb[g] * w[:, None]).sum(axis=0) / w.sum()
                colours.append("#{:02X}{:02X}{:02X}".format(*(int(round(x)) for x in np.clip(c, 0, 255))))
            self._cache[count] = (colours, extra, base + 1)
        return self._cache[count]

    def _despeckle(self, leaf: np.ndarray, n: int) -> np.ndarray:
        """Specks join the colour around them (module doc, step 3)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        leaf = leaf.copy()
        n_faces, n_leaf = leaf.shape
        total = float(self.area.sum())
        a, b = self.adjacency[:, 0], self.adjacency[:, 1]
        for _ in range(3):                  # a removed speck can leave a smaller one beside it
            major = np.array([np.bincount(r, minlength=n).argmax() for r in leaf]) if n_leaf > 1 else leaf[:, 0]
            same = major[a] == major[b]
            graph = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(n_faces, n_faces))
            _, comp = connected_components(graph, directed=False)
            comp_area = np.bincount(comp, weights=self.area)
            small = comp_area[comp] < SPECK_SHARE * total
            if not small.any():
                break
            # Each small patch takes the colour bordering it over the most area.
            border = comp[a] != comp[b]
            xs = np.concatenate([a[border], b[border]])
            ys = np.concatenate([b[border], a[border]])
            keep = small[xs]
            key = comp[xs[keep]].astype(np.int64) * n + major[ys[keep]]
            uniq, inv = np.unique(key, return_inverse=True)
            votes = np.bincount(inv.ravel(), weights=self.area[ys[keep]])
            order = np.lexsort((-votes, uniq // n))        # per patch, the strongest colour first
            first = np.ones(len(order), bool)
            first[1:] = (uniq[order] // n)[1:] != (uniq[order] // n)[:-1]
            winner = {int(k // n): int(k % n) for k in uniq[order][first].tolist()}
            changed = False
            for f in np.nonzero(small)[0].tolist():
                new = winner.get(int(comp[f]))
                if new is not None and new != major[f]:
                    leaf[f][leaf[f] == major[f]] = new
                    changed = True
            if not changed:
                break
        # Sub-triangle flecks: a colour inside a triangle that is no neighbour's (or its own)
        # main colour does not continue anywhere; give it the triangle's main colour.
        if n_leaf > 1:
            major = np.array([np.bincount(r, minlength=n).argmax() for r in leaf])
            near: list[set[int]] = [{int(m)} for m in major]
            for x, y in zip(a.tolist(), b.tolist()):
                near[x].add(int(major[y]))
                near[y].add(int(major[x]))
            for f in np.nonzero((leaf != major[:, None]).any(axis=1))[0].tolist():
                stray = ~np.isin(leaf[f], list(near[f]))
                leaf[f][stray] = major[f]
        return leaf


def _cluster(rgb: np.ndarray, weight: np.ndarray, seed: int = 1) -> tuple[np.ndarray, np.ndarray, bool]:
    """(fine label per sample, weighted-CIELAB centres, exact) — exact when the samples are
    a few flat colours, which are then kept as they are."""
    distinct, inverse = np.unique(np.round(rgb).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    if len(distinct) <= MAX_PICK:
        return inverse, _lab(distinct.astype(np.float64)), True
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rgb), min(FIT_SAMPLES, len(rgb)), p=weight / weight.sum())
    centres, _ = kmeans2(_lab(rgb[pick]), min(OVER, len(distinct)), minit="++", seed=seed)
    return _nearest(_lab(rgb), centres), centres, False


def read_gltf(path: Path) -> Project:
    """A glTF/GLB scene as one single-part figure whose colours are paint (module doc).
    The project's `texture` re-cuts the palette to another size (TextureColours.paint)."""
    try:
        scene = trimesh.load(path, force="scene", process=False)
    except Exception as e:                          # Draco/meshopt compression, external buffers, ...
        raise ValueError(f"{path.name}: cannot read this glTF ({e}). Draco- or meshopt-compressed files "
                         "and .gltf files with separate .bin/texture files are not supported; "
                         "export a plain .glb.") from e
    assert isinstance(scene, trimesh.Scene)        # force="scene" always returns one
    meshes: list[tuple[np.ndarray, np.ndarray, Sampler, bool]] = []
    for node in scene.graph.nodes_geometry:
        xf, gname = scene.graph[node]
        g = scene.geometry.get(gname)
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue                                # points, lines, empty primitives
        v = np.asarray(g.vertices, np.float64) @ xf[:3, :3].T + xf[:3, 3]
        f = np.asarray(g.faces, np.int64)
        sample, varies = _sampler(g)
        if np.linalg.det(xf[:3, :3]) < 0:           # mirrored node: keep normals outward
            # The sampler indexes the primitive's own faces; reversing a face's corners
            # reverses its barycentric order, so samples read the barycentrics back to front.
            f = f[:, ::-1]
            sample = (lambda s: (lambda fi, b: s(fi, b[:, ::-1])))(sample)
        meshes.append((v, f, sample, varies))
    if not meshes:
        raise ValueError(f"{path.name}: no triangle meshes in this glTF")

    n_faces = sum(len(f) for _, f, _, _ in meshes)
    depth = 0
    if any(varies for *_, varies in meshes):
        while depth < MAX_DEPTH and n_faces * 4 ** (depth + 1) <= LEAF_BUDGET:
            depth += 1
    bary = leaf_barycentrics(depth).mean(axis=1)    # (L, 3) leaf centres
    n_leaf = len(bary)

    rgb = np.empty((n_faces, n_leaf, 3), np.float64)
    area = np.empty(n_faces, np.float64)
    at = 0
    for v, f, sample, varies in meshes:
        fi = np.arange(len(f))
        if varies:
            rgb[at:at + len(f)] = sample(np.repeat(fi, n_leaf), np.tile(bary, (len(f), 1))).reshape(-1, n_leaf, 3)
        else:
            rgb[at:at + len(f)] = sample(fi, np.full((len(f), 3), 1 / 3))[:, None, :]
        tri = v[f]
        area[at:at + len(f)] = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        at += len(f)

    # One part: all nodes concatenated (face order kept), then turned Z-up and scaled to mm.
    offsets = np.cumsum([0] + [len(v) for v, _, _, _ in meshes[:-1]])
    allv = np.vstack([v for v, _, _, _ in meshes])
    allf = np.vstack([f + o for (_, f, _, _), o in zip(meshes, offsets)])
    m = trimesh.Trimesh(allv, allf, process=False)
    m.merge_vertices()                              # glTF splits vertices at UV and normal seams

    flat = rgb.reshape(-1, 3)
    weight = np.repeat(area / n_leaf, n_leaf) + 1e-12
    fine, lab, exact = _cluster(flat, weight)
    k0 = len(lab)
    share = np.bincount(fine, weights=weight, minlength=k0)
    means = np.array([np.average(flat[fine == i], axis=0, weights=weight[fine == i]) if share[i] > 0
                      else np.zeros(3) for i in range(k0)])
    texture = TextureColours(fine.reshape(n_faces, n_leaf).astype(np.uint8), lab, means, share / share.sum(),
                             area, np.asarray(m.face_adjacency, np.int64), exact)
    colours, extra, base = texture.paint(None)

    lo, hi = allv.min(axis=0), allv.max(axis=0)
    unit = 1000.0 if float((hi - lo).max()) <= 10.0 else 1.0
    y_up = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float64)
    part = Part(path.stem, base, np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64),
                extra).transformed(y_up @ np.diag([unit, unit, unit, 1.0]))
    return Project(path, "mesh", [FigureObject("1", path.stem, base, [part], np.eye(4))], colours, None,
                   texture=texture)
