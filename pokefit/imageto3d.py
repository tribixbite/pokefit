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
from PIL import Image

VIEWS = ("front", "left", "back", "right")      # Pixal3D's order; the first given view is the front
FRAME = 1024                                    # Pixal3D conditions on 1024 x 1024 views
PAD = 1.1                                       # the unit cube spans FRAME / PAD pixels
MASK_THRESHOLD = 0.5
DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_FACES = 200_000                         # plenty for a ~25 mm figure; pokefit stays fast
# Weights, by ComfyUI/models/<folder>: Hugging Face Comfy-Org/Pixal3D and Comfy-Org/BiRefNet.
MODEL = "pixal3d_multiview_int8_convrot.safetensors"     # diffusion_models/ (int8: fits 16 GB)
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
        scaled = on_black.resize((w, h), Image.LANCZOS)
        cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s
        canvas = Image.new("RGB", (FRAME, FRAME))
        canvas.paste(scaled, (round(FRAME / 2 - cx), round(FRAME / 2 - cy)))
        out[v] = Framed(v, canvas, s, boxes[v])
    return out


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


def generate_prompt(views: dict[str, str], seed: int = 0, faces: int = DEFAULT_FACES,
                    prefix: str = "pokefit/figure", fov: float = 20.0) -> dict[str, Any]:
    """The Pixal3D multi-view template as an API prompt, from framed views to a vertex-coloured GLB.

    Node settings are the template's (seeds offset by `seed`); the end of the chain is
    RemeshMesh (UDF, 768) -> DecimateMesh(`faces`) -> PaintMesh -> GLB.
    """
    def ks(model: str, pos: tuple[str, int], neg: tuple[str, int], latent: tuple[str, int], s: int, steps: int,
           cfg: float, scheduler: str) -> dict[str, Any]:
        return {"class_type": "KSampler", "inputs": {
            "model": [model, 0], "seed": s, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": scheduler, "positive": list(pos), "negative": list(neg), "latent_image": list(latent),
            "denoise": 1.0}}

    p: dict[str, Any] = {
        "clip": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP}},
        "vae_shape": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_SHAPE}},
        "vae_tex": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_TEXTURE}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        # CFG override and rescale match the original pipeline's defaults (template note).
        "cfg_struct": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0,
                                                               "start_percent": 0.667, "end_percent": 1.0}},
        "cfg_shape": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0,
                                                              "start_percent": 0.769, "end_percent": 1.0}},
        "rescale_struct": {"class_type": "RescaleCFG", "inputs": {"model": ["cfg_struct", 0], "multiplier": 0.7}},
        "rescale_shape": {"class_type": "RescaleCFG", "inputs": {"model": ["cfg_shape", 0], "multiplier": 0.5}},
        "sampling": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["rescale_struct", 0], "shift": 5.0}},
        "cond": {"class_type": "Pixal3DMultiViewConditioning",
                 "inputs": {"clip_vision_model": ["clip", 0], "fov": fov}},
        "empty": {"class_type": "EmptyTrellis2LatentStructure", "inputs": {"batch_size": 1}},
    }
    for v, name in views.items():
        p[f"view_{v}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p["cond"]["inputs"][v] = [f"view_{v}", 0]
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


def photos_to_glb(photos: dict[str, Path], out: Path, url: str = DEFAULT_URL, seed: int = 0,
                  faces: int = DEFAULT_FACES, mirror_right: bool = False, work: Path | None = None,
                  log: Any = print) -> dict[str, Any]:
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
    images = {v: Image.open(photos[v]) for v in names}
    t0 = time.time()
    uploaded = {v: comfy.upload(f"pokefit_{tag}_{v}.png", _png(images[v].convert("RGB"))) for v in names}
    log(f"masks: BiRefNet on {', '.join(names)}")
    outs = comfy.run(mask_prompt(uploaded))
    masks = {}
    for v in names:
        m = Image.open(io.BytesIO(comfy.fetch(_first_file(outs, f"save_{v}")))).convert("L")
        masks[v] = np.asarray(m.resize(images[v].size), np.float32) / 255.0
    if mirror_right and "left" in names and "right" not in names:
        images["right"] = images["left"].transpose(Image.FLIP_LEFT_RIGHT)
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
    out.write_bytes(data)
    return {"glb": str(out), "views": names, "mask_seconds": round(t1 - t0, 1),
            "generate_seconds": round(time.time() - t1, 1), "bytes": len(data),
            "framing": {v: {"scale": round(f.scale, 4), "box": f.box} for v, f in framed.items()}}
