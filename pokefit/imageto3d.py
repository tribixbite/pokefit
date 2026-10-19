"""Photos of a figure (front / left / back, optionally right) -> a coloured GLB, via a local ComfyUI.

The model is Tencent ARC's **Pixal3D** multi-view (MIT, SIGGRAPH 2026; weights repackaged for
ComfyUI as `Comfy-Org/Pixal3D`), run by ComfyUI's native nodes (`comfy_extras/nodes_trellis2.py`,
ComfyUI >= 0.42): pure PyTorch, so it works on Windows / Blackwell without custom CUDA builds.
It cannot run in a web page (≈ 9 GB of weights and sparse 3D convolutions), so this is a local tool;
see AGENTS.md "Photos to 3D" for the setup.

Pipeline:
1. Background masks: one ComfyUI prompt runs BiRefNet (`RemoveBackground`) on every photo.
2. Framing (here, numpy): Pixal3D's rig puts the object, normalised to a unit cube at the origin, at
   1/1.1 of the frame in every view (20° FOV orbit cameras at azimuth 0/90/180/270, z up). So each
   photo is cropped to its mask, scaled so the figure has the SAME height in every view (the vertical
   extent is the same from any side), the largest extent over all views spans 1024/1.1 px, the mask's
   box is centred, and the result is composited on black. Per-view crop-to-mask framing (the
   template's `ImageCropToMask`) would give each view its own scale.
3. Generation: the template "Pixal3D: Multi Views to 3D" (Comfy-Org/workflow_templates
   `3d_pixal3d_multi_views.json`) with its sampler settings, then remesh (UDF), decimate and
   `PaintMesh` (vertex colours) -> GLB. pokefit reads vertex colours as filament paint (gltf.py).

"left" is the figure's own left side (camera at +X when the figure faces -Y), which is the view where
it faces the image's left.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

VIEWS = ("front", "left", "back", "right")      # Pixal3D's order; the first given view is the front
FRAME = 1024                                    # Pixal3D conditions on 1024 x 1024 views
PAD = 1.1                                       # the unit cube spans FRAME / PAD pixels
MASK_THRESHOLD = 0.5
DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_FACES = 200_000                         # plenty for a ~25 mm figure; pokefit stays fast
# Weights, by ComfyUI/models/<folder>: Hugging Face Comfy-Org/Pixal3D and Comfy-Org/BiRefNet.
MODEL = "pixal3d_multiview_int8_convrot.safetensors"     # diffusion_models/ (int8: fits 16 GB)
MODEL_SINGLE = "pixal3d_int8_convrot.safetensors"        # diffusion_models/ (one photo, any angle)
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"       # geometry_estimation/ (Comfy-Org/MoGe): photo FOV
CLIP = "dino_v3_L_naf_fp32.safetensors"                  # clip_vision/
VAE_SHAPE = "trellis_2_shape_vae_bf16.safetensors"       # vae/
VAE_TEXTURE = "trellis_2_texture_vae_bf16.safetensors"   # vae/
BG_MODEL = "birefnet.safetensors"                        # background_removal/


class ComfyError(RuntimeError):
    pass


@dataclass
class Framed:
    """One prepared view and how it was framed (for the report)."""
    name: str
    image: Image.Image          # FRAME x FRAME RGB, figure on black
    scale: float                # photo pixels -> frame pixels
    box: tuple[int, int, int, int]   # mask bounding box in the photo (x0, y0, x1, y1)
    mask: np.ndarray | None = None   # FRAME x FRAME bool: the figure's pixels in `image`


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1), exclusive ends, of mask > MASK_THRESHOLD."""
    ys, xs = np.nonzero(mask > MASK_THRESHOLD)
    if len(xs) == 0:
        raise ValueError("no figure found in the photo (empty background mask)")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def frame_views(photos: dict[str, Image.Image], masks: dict[str, np.ndarray]) -> dict[str, Framed]:
    """Frame every photo like Pixal3D's rig: same figure height in all views, largest extent at
    FRAME / PAD, mask box centred, background black (RGB × mask)."""
    boxes = {v: mask_box(masks[v]) for v in photos}
    # Every view at the common height H; the widest view's width (or H) must fit FRAME / PAD.
    ratio = max(max(b[2] - b[0], b[3] - b[1]) / (b[3] - b[1]) for b in boxes.values())
    height = FRAME / PAD / ratio
    out: dict[str, Framed] = {}
    for v, photo in photos.items():
        x0, y0, x1, y1 = boxes[v]
        s = height / (y1 - y0)
        rgb = np.asarray(photo.convert("RGB"), np.float32) / 255.0
        alpha = np.clip(masks[v].astype(np.float32), 0.0, 1.0)[..., None]
        on_black = Image.fromarray((rgb * alpha * 255.0 + 0.5).astype(np.uint8))
        w, h = max(1, round(photo.width * s)), max(1, round(photo.height * s))
        scaled = on_black.resize((w, h), Image.Resampling.LANCZOS)
        cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s
        canvas = Image.new("RGB", (FRAME, FRAME))
        at = (round(FRAME / 2 - cx), round(FRAME / 2 - cy))
        canvas.paste(scaled, at)
        m = Image.fromarray((np.clip(masks[v], 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        mcanvas = Image.new("L", (FRAME, FRAME))
        mcanvas.paste(m, at)
        out[v] = Framed(v, canvas, s, boxes[v], np.asarray(mcanvas) > 127)
    return out


# ---------------------------------------------------------------------------------------------------------
# Photo colours back onto the mesh. Pixal3D's mesh sits in the rig's frame (silhouettes overlap the framed
# photos with IoU 0.94-0.98 on the Squirtle), so each vertex can take the photos' colour where a view sees
# it. The model's own colour is lighting-free albedo, darker than the photos (blue #6DAAB2 vs ~#8ECAD8),
# and its small details (eye highlights, iris) are blurred; the photos have both right.

VIEW_AZIMUTH = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
FOV_DEG = 20.0
MIN_COS = 0.45          # a view must face the surface at least this squarely (cos of the angle)
EDGE_PX = 4             # photo pixels this close to the silhouette are not sampled (the mesh is a few px off)
DEPTH_TOL = 0.012       # rig units (the figure spans ~1): how far behind the nearest surface still counts as seen


@dataclass(frozen=True)
class Camera:
    """Pixal3D's view cameras: on a circle round the origin, z up, looking at it; `pad` = how far the unit
    cube is framed (the multi-view rig: FOV 20°, 1.1; the single-image node: MoGe's FOV, 1.0 — its
    camera-to-world is `build_proj_transform_matrix`, the rig's front view at distance 0.5/tan(fov/2))."""
    fov: float = FOV_DEG
    pad: float = PAD


RIG = Camera()


def rig_view(name: str, cam: Camera = RIG) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(camera position, right, up, forward) of a view; z up, the figure facing -y (ComfyUI's
    `_orbit_camera_to_world`: azimuth 0 = front, 90 = the figure's left)."""
    a = np.radians(VIEW_AZIMUTH[name])
    back = np.array([np.sin(a), -np.cos(a), 0.0])
    right = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.cross(back, right)
    dist = cam.pad * 0.5 / np.tan(np.radians(cam.fov) / 2)
    return back * dist, right, up, -back


def project(points: np.ndarray, name: str, cam: Camera = RIG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel x, y (FRAME frame) and depth of rig-frame points in a view."""
    pos, right, up, fwd = rig_view(name, cam)
    rel = points - pos
    depth = rel @ fwd
    k = (FRAME / 2) / (np.maximum(depth, 1e-9) * np.tan(np.radians(cam.fov) / 2))
    return FRAME / 2 + (rel @ right) * k, FRAME / 2 - (rel @ up) * k, depth


def photo_colours(vertices: np.ndarray, normals: np.ndarray, base: np.ndarray,
                  views: dict[str, Framed], cam: Camera = RIG) -> tuple[np.ndarray, float]:
    """Vertex colours (sRGB 0..255) from the framed photos; (colours, share of vertices a photo saw).

    Per view: vertices facing it (cos > MIN_COS) and not behind the nearest surface (a z-buffer of vertex
    splats, min-filtered to close the gaps between them) sample the photo bilinearly inside its mask
    (eroded by EDGE_PX: along the silhouette the mesh and photo disagree by a few pixels), weighted by cos². Vertices no view sees keep the model's colour, corrected by the per-channel gain and
    offset that maps the model's colours to the photos' on the vertices that were seen.
    """
    from scipy import ndimage

    n = len(vertices)
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)
    for name, fr in views.items():
        if fr.mask is None or name not in VIEW_AZIMUTH:
            continue
        pos, *_ = rig_view(name, cam)
        px, py, depth = project(vertices, name, cam)
        to_cam = pos - vertices
        cos = np.einsum("ij,ij->i", normals, to_cam) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-12)
        ix, iy = np.round(px).astype(int), np.round(py).astype(int)
        inside = (ix >= 0) & (ix < FRAME) & (iy >= 0) & (iy < FRAME)
        zbuf = np.full((FRAME, FRAME), np.inf)
        np.minimum.at(zbuf, (iy[inside], ix[inside]), depth[inside])
        zbuf = ndimage.minimum_filter(zbuf, size=5)
        seen = inside & (cos > MIN_COS)
        seen[seen] &= depth[seen] <= zbuf[iy[seen], ix[seen]] + DEPTH_TOL
        inner = ndimage.binary_erosion(fr.mask, iterations=EDGE_PX)
        seen[seen] &= inner[iy[seen], ix[seen]]
        if not seen.any():
            continue
        img = np.asarray(fr.image.convert("RGB"), np.float64)
        x, y = np.clip(px[seen], 0, FRAME - 1.001), np.clip(py[seen], 0, FRAME - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        sample = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
                  + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        w = cos[seen] ** 2
        acc[seen] += sample * w[:, None]
        wsum[seen] += w
    got = wsum > 0
    out = np.asarray(base, np.float64).copy()
    out[got] = acc[got] / wsum[got, None]
    if got.sum() > 10 and (~got).any():
        for c in range(3):                     # model colour -> photo colour, fitted where both exist
            a, b = np.polyfit(base[got, c], out[got, c], 1)
            out[~got, c] = a * base[~got, c] + b
    return np.clip(out, 0, 255), float(got.mean())


def vertex_colour_glb(vertices: np.ndarray, faces: np.ndarray, srgb: np.ndarray) -> bytes:
    """A GLB with float COLOR_0 (LINEAR, per the glTF spec) under a white PBR material, as ComfyUI writes."""
    import struct

    lin = trimesh.visual.color.srgb_to_linear(np.clip(np.asarray(srgb, np.float64) / 255.0, 0, 1))
    pos = np.asarray(vertices, np.float32)
    chunks = [np.asarray(faces, np.uint32).tobytes(), pos.tobytes(), np.asarray(lin, np.float32).tobytes()]
    offsets = np.cumsum([0] + [len(c) for c in chunks[:-1]]).tolist()
    blob = b"".join(chunks)
    n = len(pos)
    doc = {"asset": {"version": "2.0", "generator": "pokefit imageto3d"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "COLOR_0": 2}, "indices": 0, "material": 0}]}],
           "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
                                                   "roughnessFactor": 1.0}, "doubleSided": True}],
           "buffers": [{"byteLength": len(blob)}],
           "bufferViews": [{"buffer": 0, "byteOffset": int(o), "byteLength": len(c)} for o, c in zip(offsets, chunks)],
           "accessors": [{"bufferView": 0, "componentType": 5125, "count": int(np.size(faces)), "type": "SCALAR"},
                         {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3",
                          "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
                         {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += bytes(-len(blob) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN" + bytes(1)) + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _glb_vertex_colours(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, sRGB 0..255 vertex colours) of a GLB, node transforms applied. Read from the
    scene's geometry directly: `Scene.dump()` copies drop COLOR_0 kept as a vertex attribute."""
    sc = trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb", force="scene", process=False)
    assert isinstance(sc, trimesh.Scene)        # force="scene" always returns one
    verts, faces, cols, base = [], [], [], 0
    for node in sc.graph.nodes_geometry:
        transform, name = sc.graph[node]
        g = sc.geometry[name]
        if not isinstance(g, trimesh.Trimesh):
            continue
        v = np.asarray(g.vertices, np.float64)
        verts.append(v @ transform[:3, :3].T + transform[:3, 3])
        faces.append(np.asarray(g.faces, np.int64) + base)
        base += len(v)
        raw = getattr(g.visual, "vertex_attributes", {}).get("color")
        if raw is not None and len(raw) == len(v):
            lin = np.asarray(raw, np.float64)[:, :3]
            lin = lin / 255.0 if lin.max() > 1.0 else lin
            cols.append(trimesh.visual.color.linear_to_srgb(lin) * 255.0)
        elif isinstance(g.visual, trimesh.visual.ColorVisuals):
            cols.append(np.asarray(g.visual.vertex_colors, np.float64)[:, :3])
        else:
            cols.append(np.full((len(v), 3), 128.0))     # no colour: the photos decide
    if not verts:
        raise ComfyError("the generated GLB has no mesh")
    return np.vstack(verts), np.vstack(faces), np.vstack(cols)


def recolour_glb(data: bytes, views: dict[str, Framed], cam: Camera = RIG) -> tuple[bytes, float]:
    """The generated GLB with its vertex colours replaced by the photos' (see `photo_colours`)."""
    vertices, faces, base = _glb_vertex_colours(data)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    # glTF is y-up; the rig is z-up with the figure facing -y: (x, y, z) -> (x, -z, y)
    to_rig = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
    v_rig = vertices @ to_rig.T
    n_rig = np.asarray(mesh.vertex_normals, np.float64) @ to_rig.T
    colours, share = photo_colours(v_rig, n_rig, base, views, cam)
    return vertex_colour_glb(vertices, faces, colours), share


class Comfy:
    """Minimal ComfyUI HTTP client: upload, queue a prompt, wait, download outputs."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 3600.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            data: bytes = r.read()
            return data

    def get_json(self, path: str) -> Any:
        return json.loads(self._get(path))

    def alive(self, timeout: float = 5.0) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/system_stats", timeout=timeout) as r:
                r.read()
            return True
        except (OSError, ValueError):
            return False

    def upload(self, name: str, png: bytes) -> str:
        """Upload an image to ComfyUI's input folder; returns the name LoadImage uses."""
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for field, value in (("overwrite", "true"), ("type", "input")):
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                   f"Content-Type: image/png\r\n\r\n".encode())
        body.write(png)
        body.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(self.url + "/upload/image", data=body.getvalue(), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())
        return str(res["name"]) if not res.get("subfolder") else f"{res['subfolder']}/{res['name']}"

    def run(self, prompt: dict[str, Any], progress: Any = None) -> dict[str, Any]:
        """Queue a prompt and wait for it; returns its outputs (node id -> output dict)."""
        req = urllib.request.Request(self.url + "/prompt", method="POST",
                                     data=json.dumps({"prompt": prompt, "client_id": self.client_id}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                pid = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as exc:
            raise ComfyError(f"ComfyUI rejected the prompt: {exc.read().decode(errors='replace')[:2000]}") from exc
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            hist = self.get_json(f"/history/{pid}")
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1] if msgs else status
                    raise ComfyError(f"ComfyUI failed: {json.dumps(detail)[:2000]}")
                if status.get("completed", True):
                    out: dict[str, Any] = entry.get("outputs", {})
                    return out
            if progress:
                progress(time.time() - t0)
            time.sleep(2.0)
        raise ComfyError(f"ComfyUI did not finish within {self.timeout:.0f} s")

    def fetch(self, item: dict[str, Any]) -> bytes:
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        return self._get(f"/view?{q}")


def mask_prompt(images: dict[str, str]) -> dict[str, Any]:
    """BiRefNet masks of uploaded photos, saved as greyscale PNGs (one SaveImage per view)."""
    p: dict[str, Any] = {"bg": {"class_type": "LoadBackgroundRemovalModel",
                                "inputs": {"bg_removal_name": BG_MODEL}}}
    for v, name in images.items():
        p[f"load_{v}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p[f"mask_{v}"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg", 0],
                                                                        "image": [f"load_{v}", 0]}}
        p[f"img_{v}"] = {"class_type": "MaskToImage", "inputs": {"mask": [f"mask_{v}", 0]}}
        p[f"save_{v}"] = {"class_type": "SaveImage", "inputs": {"images": [f"img_{v}", 0],
                                                                 "filename_prefix": f"pokefit/mask_{v}"}}
    return p


def _pixal3d_chain(p: dict[str, Any], model: str, seed: int, faces: int, prefix: str) -> dict[str, Any]:
    """Add the Pixal3D sampling chain after a conditioning node `cond` (positive 0, negative 1): the
    templates' sampler settings (seeds offset by `seed`), then RemeshMesh (solid, 768) ->
    DecimateMesh(`faces`) -> PaintMesh -> GLB (node `save`)."""
    def ks(model_node: str, pos: tuple[str, int], neg: tuple[str, int], latent: tuple[str, int], s: int,
           steps: int, cfg: float, scheduler: str) -> dict[str, Any]:
        return {"class_type": "KSampler", "inputs": {
            "model": [model_node, 0], "seed": s, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": scheduler, "positive": list(pos), "negative": list(neg), "latent_image": list(latent),
            "denoise": 1.0}}

    p.update({
        "clip": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP}},
        "vae_shape": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_SHAPE}},
        "vae_tex": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_TEXTURE}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        # CFG override and rescale match the original pipeline's defaults (template note).
        "cfg_struct": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0,
                                                               "start_percent": 0.667, "end_percent": 1.0}},
        "cfg_shape": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0,
                                                              "start_percent": 0.769, "end_percent": 1.0}},
        "rescale_struct": {"class_type": "RescaleCFG", "inputs": {"model": ["cfg_struct", 0], "multiplier": 0.7}},
        "rescale_shape": {"class_type": "RescaleCFG", "inputs": {"model": ["cfg_shape", 0], "multiplier": 0.5}},
        "sampling": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["rescale_struct", 0], "shift": 5.0}},
        "empty": {"class_type": "EmptyTrellis2LatentStructure", "inputs": {"batch_size": 1}},
    })
    p["s_struct"] = ks("sampling", ("cond", 0), ("cond", 1), ("empty", 0), 56 + seed, 12, 7.5, "normal")
    p["structure"] = {"class_type": "VaeDecodeStructureTrellis2",
                      "inputs": {"samples": ["s_struct", 0], "vae": ["vae_shape", 0], "resolution": "32"}}
    p["shape_stage"] = {"class_type": "Trellis2ShapeStage",
                        "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "voxel": ["structure", 0]}}
    p["s_shape"] = ks("rescale_shape", ("shape_stage", 0), ("shape_stage", 1), ("shape_stage", 2), 42 + seed, 20,
                      7.5, "normal")
    p["upsample"] = {"class_type": "Trellis2UpsampleStage",
                     "inputs": {"positive": ["shape_stage", 0], "negative": ["shape_stage", 1],
                                "shape_latent": ["s_shape", 0], "vae": ["vae_shape", 0], "target_resolution": 1536}}
    p["s_up"] = ks("rescale_shape", ("upsample", 0), ("upsample", 1), ("upsample", 2), 42 + seed, 12, 7.5, "simple")
    p["shape"] = {"class_type": "VaeDecodeShapeTrellis", "inputs": {"samples": ["s_up", 0], "vae": ["vae_shape", 0]}}
    p["tex_stage"] = {"class_type": "Trellis2TextureStage",
                      "inputs": {"positive": ["upsample", 0], "negative": ["upsample", 1], "shape_latent": ["s_up", 0]}}
    p["s_tex"] = ks("unet", ("tex_stage", 0), ("tex_stage", 1), ("tex_stage", 2), 43 + seed, 12, 1.0, "normal")
    p["texture"] = {"class_type": "VaeDecodeTextureTrellis",
                    "inputs": {"samples": ["s_tex", 0], "vae": ["vae_tex", 0], "shape_subdivides": ["shape", 1]}}
    # "solid" (not the template's "udf"): a global inside/outside cut gives ONE watertight solid. UDF
    # leaves a double-walled skin (an outer and an inner shell 0.1 mm apart), useless for printing.
    p["remesh"] = {"class_type": "RemeshMesh", "inputs": {
        "mesh": ["shape", 0], "resolution": 768, "sign_mode": "solid", "sign_mode.fill": 20.0,
        "sign_mode.qef": False, "band": 1.0, "project_back": 0.0, "fix_poles": False, "smooth_iters": 20,
        "drop_small_components": 0.01, "precluster_max_verts": 20_000_000}}
    p["decimate"] = {"class_type": "DecimateMesh",
                     "inputs": {"mesh": ["remesh", 0], "target_face_count": faces, "placement_mode": "midpoint"}}
    p["paint"] = {"class_type": "PaintMesh", "inputs": {"mesh": ["decimate", 0], "voxel_colors": ["texture", 0]}}
    p["glb"] = {"class_type": "MeshToFile3D", "inputs": {"mesh": ["paint", 0]}}
    p["save"] = {"class_type": "Save3DAdvanced", "inputs": {"model_3d": ["glb", 0], "filename_prefix": prefix,
                                                            "viewport_state": "", "width": 1024, "height": 1024}}
    return p


def generate_prompt(views: dict[str, str], seed: int = 0, faces: int = DEFAULT_FACES,
                    prefix: str = "pokefit/figure", fov: float = FOV_DEG) -> dict[str, Any]:
    """The Pixal3D multi-view template ("Multi Views to 3D") as an API prompt: framed rig views -> GLB."""
    p: dict[str, Any] = {"cond": {"class_type": "Pixal3DMultiViewConditioning",
                                  "inputs": {"clip_vision_model": ["clip", 0], "fov": fov}}}
    for v, name in views.items():
        p[f"view_{v}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p["cond"]["inputs"][v] = [f"view_{v}", 0]
    return _pixal3d_chain(p, MODEL, seed, faces, prefix)


def single_prompt(photo: str, seed: int = 0, faces: int = DEFAULT_FACES,
                  prefix: str = "pokefit/figure") -> dict[str, Any]:
    """One photo from any angle -> GLB, as the template "Pixal3D & TRELLIS.2: Image to Model": BiRefNet
    mask, crop to it (pad 1.1), MoGe-2 estimates the crop's horizontal FOV, single-view Pixal3D. The mesh is
    posed as the photo shows it (turn it upright in pokefit's Figure card if the photo was angled)."""
    p: dict[str, Any] = {
        "load": {"class_type": "LoadImage", "inputs": {"image": photo}},
        "bg": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BG_MODEL}},
        "mask": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg", 0], "image": ["load", 0]}},
        "crop": {"class_type": "ImageCropToMask", "inputs": {"images": ["load", 0], "masks": ["mask", 0],
                                                             "width": FRAME, "height": FRAME, "pad_factor": PAD,
                                                             "grow_mask": 0, "background": "#000000"}},
        "moge": {"class_type": "LoadMoGeModel", "inputs": {"model_name": MOGE_MODEL}},
        "geometry": {"class_type": "MoGeInference", "inputs": {
            "moge_model": ["moge", 0], "image": ["crop", 0], "resolution_level": 9, "fov_x_degrees": 0.0,
            "batch_size": 4, "force_projection": True, "apply_mask": True, "refine_steps": 3}},
        "fov": {"class_type": "MoGeGeometryToFOV", "inputs": {"moge_geometry": ["geometry", 0],
                                                              "axis": "horizontal", "unit": "degrees"}},
        "cond": {"class_type": "Pixal3DConditioning", "inputs": {"clip_vision_model": ["clip", 0],
                                                                 "image": ["crop", 0], "camera_angle_x": ["fov", 0]}},
        # What photo colours need back: the crop the model saw, its mask, and the FOV.
        "mask_img": {"class_type": "MaskToImage", "inputs": {"mask": ["mask", 0]}},
        "mask_crop": {"class_type": "ImageCropToMask", "inputs": {"images": ["mask_img", 0], "masks": ["mask", 0],
                                                                  "width": FRAME, "height": FRAME, "pad_factor": PAD,
                                                                  "grow_mask": 0, "background": "#000000"}},
        "save_crop": {"class_type": "SaveImage", "inputs": {"images": ["crop", 0], "filename_prefix": f"{prefix}_crop"}},
        "save_mask": {"class_type": "SaveImage", "inputs": {"images": ["mask_crop", 0],
                                                            "filename_prefix": f"{prefix}_mask"}},
        "show_fov": {"class_type": "PreviewAny", "inputs": {"source": ["fov", 0]}},
    }
    return _pixal3d_chain(p, MODEL_SINGLE, seed, faces, prefix)


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _first_file(outputs: dict[str, Any], node: str) -> dict[str, Any]:
    out = outputs.get(node, {})
    for key in ("3d", "images", "result", "model_3d", "files"):
        items = out.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict) and "filename" in items[0]:
            return dict(items[0])
    for items in out.values():               # any list of {filename, ...}
        if isinstance(items, list) and items and isinstance(items[0], dict) and "filename" in items[0]:
            return dict(items[0])
    raise ComfyError(f"node {node} produced no file: {json.dumps(out)[:500]}")


def photo_to_glb(photo: Path, out: Path, url: str = DEFAULT_URL, seed: int = 0, faces: int = DEFAULT_FACES,
                 log: Any = print, photo_colour: bool = True) -> dict[str, Any]:
    """One photo of the figure from any angle -> GLB (single-view Pixal3D, `single_prompt`). The side the
    photo shows takes the photo's colours (the camera is the single-image node's: MoGe's FOV); the rest
    keeps the model's, gain-corrected to match."""
    comfy = Comfy(url)
    if not comfy.alive():
        raise ComfyError(f"ComfyUI is not running at {url} (see AGENTS.md: Photos to 3D)")
    tag = uuid.uuid4().hex[:8]
    name = comfy.upload(f"pokefit_{tag}_photo.png", _png(Image.open(photo).convert("RGB")))
    stage = "generating with Pixal3D (one photo, a few minutes)"
    log(stage)
    t0 = time.time()
    outs = comfy.run(single_prompt(name, seed, faces, prefix=f"pokefit/{out.stem}_{tag}"),
                     progress=lambda s: log(f"{stage}: {s:.0f} s"))
    data = comfy.fetch(_first_file(outs, "save"))
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = None
    fov = None
    if photo_colour:
        fov = float(outs.get("show_fov", {}).get("text", ["nan"])[0])
        crop = Image.open(io.BytesIO(comfy.fetch(_first_file(outs, "save_crop")))).convert("RGB")
        mimg = Image.open(io.BytesIO(comfy.fetch(_first_file(outs, "save_mask")))).convert("L")
        if np.isfinite(fov) and crop.size == (FRAME, FRAME):
            (out.parent / f"{out.stem}_model_colours.glb").write_bytes(data)
            log("putting the photo's colours on the model")
            view = Framed("front", crop, 1.0, (0, 0, FRAME, FRAME), np.asarray(mimg.resize(crop.size)) > 127)
            data, seen = recolour_glb(data, {"front": view}, Camera(fov=fov, pad=1.0))
    out.write_bytes(data)
    return {"glb": str(out), "views": ["photo"], "generate_seconds": round(time.time() - t0, 1), "bytes": len(data),
            "photo_colour_share": None if seen is None else round(seen, 3), "fov": fov}


def photos_to_glb(photos: dict[str, Path], out: Path, url: str = DEFAULT_URL, seed: int = 0,
                  faces: int = DEFAULT_FACES, mirror_right: bool = False, work: Path | None = None,
                  log: Any = print, photo_colour: bool = True) -> dict[str, Any]:
    """Run the whole pipeline; writes `out` (.glb) and the framed views next to it. Returns a report."""
    names = [v for v in VIEWS if v in photos]
    if not names or names[0] != "front":
        raise ValueError("a front photo is required (and is the pose the figure gets)")
    comfy = Comfy(url)
    if not comfy.alive():
        raise ComfyError(f"ComfyUI is not running at {url} (see AGENTS.md: Photos to 3D)")
    work = work or out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    images: dict[str, Image.Image] = {v: Image.open(photos[v]) for v in names}
    t0 = time.time()
    uploaded = {v: comfy.upload(f"pokefit_{tag}_{v}.png", _png(images[v].convert("RGB"))) for v in names}
    log(f"masks: BiRefNet on {', '.join(names)}")
    outs = comfy.run(mask_prompt(uploaded))
    masks = {}
    for v in names:
        m = Image.open(io.BytesIO(comfy.fetch(_first_file(outs, f"save_{v}")))).convert("L")
        masks[v] = np.asarray(m.resize(images[v].size), np.float32) / 255.0
    if mirror_right and "left" in names and "right" not in names:
        images["right"] = images["left"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        masks["right"] = masks["left"][:, ::-1].copy()
        names.append("right")
    framed = frame_views({v: images[v] for v in names}, masks)
    views = {}
    for v in names:
        framed[v].image.save(work / f"view_{v}.png")
        views[v] = comfy.upload(f"pokefit_{tag}_view_{v}.png", _png(framed[v].image))
    t1 = time.time()
    stage = f"generating with Pixal3D ({len(names)} views, a few minutes)"
    log(stage)
    outs = comfy.run(generate_prompt(views, seed, faces, prefix=f"pokefit/{out.stem}_{tag}"),
                     progress=lambda s: log(f"{stage}: {s:.0f} s"))
    data = comfy.fetch(_first_file(outs, "save"))
    seen = None
    if photo_colour:
        (work / f"{out.stem}_model_colours.glb").write_bytes(data)     # the model's own colours, kept
        log("putting the photos' colours on the model")
        data, seen = recolour_glb(data, framed)
    out.write_bytes(data)
    return {"glb": str(out), "views": names, "mask_seconds": round(t1 - t0, 1),
            "generate_seconds": round(time.time() - t1, 1), "bytes": len(data),
            "photo_colour_share": None if seen is None else round(seen, 3),
            "framing": {v: {"scale": round(f.scale, 4), "box": f.box} for v, f in framed.items()}}
