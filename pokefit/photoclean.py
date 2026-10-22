"""Noisy photos of a figure -> clean front / left / back views for `imageto3d`, via a local ComfyUI.

Stock or product photos rarely suit Pixal3D: the figure sits in an open ball or on a stand, the shots are
off-angle, glossy and cropped, and a leftover background turns into a slab of the mesh. This step redraws
them with **Qwen-Image-Edit-2511** (Alibaba Qwen, Apache-2.0; ComfyUI's native `TextEncodeQwenImageEditPlus`
nodes and the `image_qwen_image_edit_2511_int8` template's settings):

1. Clean front: one edit of up to three of the photos together ("Picture 1..3" references): the figure
   alone, facing the camera, on plain white, even light. `pose=True` re-poses it with arms down at its sides
   (a better fit in a ball): the last reference is then an OpenPose keypoint map (`pose_skeleton`), which
   2509+ reads natively. Words alone ("arms tucked in") did not move a Charmander's raised arms; the skeleton
   did, though it also stood the figure up, so check the result.
2. Other views: one edit per view from the clean front with fal's Multiple-Angles LoRA (Apache-2.0, trained
   on 2511; its prompt grammar is "<sks> <azimuth> <elevation> <distance>"), same seed for every view.
   Separate calls, not one turnaround sheet: sheet LoRAs make characters narrower.
Both use the Lightning 8-step LoRA (lightx2v, Apache-2.0): 8 steps at CFG 1 instead of 40 at CFG 3.

Weights (ComfyUI/models/<folder>), from Hugging Face:
  diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors   Comfy-Org/Qwen-Image-Edit_ComfyUI (19.5 GB)
  text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors              Comfy-Org/Qwen-Image_ComfyUI (8.9 GB)
  vae/qwen_image_vae.safetensors                                   Comfy-Org/Qwen-Image_ComfyUI
  loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors lightx2v/Qwen-Image-Edit-2511-Lightning
  loras/qwen-image-edit-2511-multiple-angles-lora.safetensors      fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
"""
from __future__ import annotations

import io
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .imageto3d import DEFAULT_URL, Comfy, ComfyError, _first_file, _png

EDIT_MODEL = "qwen_image_edit_2511_int8_convrot.safetensors"
EDIT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
EDIT_VAE = "qwen_image_vae.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
MAX_REFS = 3                       # TextEncodeQwenImageEditPlus takes image1..image3

CLEAN_PROMPT = (
    "Show only the toy figure from these pictures, alone, standing on nothing, seen straight from the front at "
    "eye level, the whole figure in view and centred, on a plain pure white background, soft even studio light, "
    "no shadows, a matte finish with no shiny highlights, reflections or glare. Remove everything that is not part of the figure: the ball, the stand, "
    "the base and any loose effect pieces around it. Keep the figure's exact shape, proportions, colours, "
    "eyes and markings.")
# With a pose skeleton as the last reference picture (`pose_skeleton`): Qwen-Image-Edit 2509+ reads OpenPose
# keypoint maps natively, so the pose is drawn rather than described.
POSE_SKELETON_PROMPT = (
    " The last picture is a stick-figure pose skeleton, not part of the figure: pose the figure like it, sitting "
    "flat on its bottom with both arms down at its sides, close to the body, and the legs forward.")

# OpenPose COCO-18 skeleton (controlnet_aux's drawing: limb pairs, one colour per limb and joint).
_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17))
_POSE_COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
                 (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                 (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85))
# A chibi figure (head ~40 % of the height) sitting, facing the camera, arms down at its sides, legs forward:
# (x, y) as shares of the figure's box. Index order: nose, neck, R shoulder/elbow/wrist, L shoulder/elbow/wrist,
# R hip/knee/ankle, L hip/knee/ankle, R eye, L eye, R ear, L ear (R = the figure's right = image left).
_SITTING = ((0.50, 0.27), (0.50, 0.45), (0.36, 0.49), (0.31, 0.60), (0.30, 0.70), (0.64, 0.49), (0.69, 0.60),
            (0.70, 0.70), (0.42, 0.76), (0.29, 0.86), (0.24, 0.93), (0.58, 0.76), (0.71, 0.86), (0.76, 0.93),
            (0.43, 0.21), (0.57, 0.21), (0.30, 0.23), (0.70, 0.23))


def pose_skeleton(width: int, height: int, box: tuple[float, float, float, float] = (0.2, 0.08, 0.8, 0.95),
                  points: tuple[tuple[float, float], ...] = _SITTING) -> Image.Image:
    """An OpenPose-style keypoint map (black background, coloured limbs and joints) of `points` placed in
    `box` = (left, top, right, bottom) as shares of the image."""
    import math

    from PIL import ImageDraw

    l, t, r, b = box
    pts = [((l + x * (r - l)) * width, (t + y * (b - t)) * height) for x, y in points]
    img = Image.new("RGB", (width, height), (0, 0, 0))
    stick = max(2.0, 4.0 * min(width, height) / 512)
    layer = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for (a, c), colour in zip(_LIMBS, _POSE_COLOURS):
        (x0, y0), (x1, y1) = pts[a], pts[c]
        length, angle = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # The limb as a polygon approximating controlnet_aux's ellipse (cv2.ellipse2Poly).
        poly = [(mx + math.cos(angle) * length / 2 * math.cos(s) - math.sin(angle) * stick * math.sin(s),
                 my + math.sin(angle) * length / 2 * math.cos(s) + math.cos(angle) * stick * math.sin(s))
                for s in (k * math.pi / 18 for k in range(36))]
        draw.polygon(poly, fill=tuple(int(v * 0.6) for v in colour))
    img = Image.blend(img, layer, 1.0)
    draw = ImageDraw.Draw(img)
    rad = stick
    for (x, y), colour in zip(pts, _POSE_COLOURS):
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=colour)
    return img


# The figure's own left = the view where it faces the image's left (imageto3d's convention). fal's LoRA names
# them the other way round: on the Charmander photos "<sks> left side view" turned it to face the image's
# RIGHT (tail on the left), i.e. its own right side; so the rig's left view asks for "right side view".
VIEW_AZIMUTH = {"left": "right side view", "back": "back view", "right": "left side view"}


def view_prompt(view: str) -> str:
    """fal Multiple-Angles prompt for a rig view: eye level, medium shot (the whole figure)."""
    return f"<sks> {VIEW_AZIMUTH[view]} eye-level shot medium shot"


@dataclass
class CleanOptions:
    """Settings of the clean-up edits."""
    seed: int = 0
    steps: int = 8                  # Lightning 8-step LoRA
    pose: bool = False              # also re-pose: arms down at the sides (an OpenPose skeleton reference)
    angle_strength: float = 0.9     # Multiple-Angles LoRA strength (fal: 0.8-1.0)
    views: tuple[str, ...] = ("left", "back")   # generated from the clean front; "right" too for lopsided figures
    view_refs: bool = True          # also give each view edit the original photos (after the clean front), so
                                    # parts the front hides (a tail flame) come from them, not from a guess

    def __post_init__(self) -> None:
        self.seed, self.steps, self.angle_strength = int(self.seed), int(self.steps), float(self.angle_strength)
        self.views = tuple(self.views)
        if not 4 <= self.steps <= 40:
            raise ValueError("steps must be 4..40")
        if not 0.0 <= self.angle_strength <= 1.5:
            raise ValueError("angle_strength must be 0..1.5")
        bad = [v for v in self.views if v not in VIEW_AZIMUTH]
        if bad:
            raise ValueError(f"unknown views {bad}: use {', '.join(VIEW_AZIMUTH)}")


def edit_prompt(images: list[str], prompt: str, opts: CleanOptions, angles: bool = False,
                prefix: str = "pokefit/clean") -> dict[str, Any]:
    """ComfyUI API prompt: one Qwen-Image-Edit-2511 edit of 1-3 uploaded images (the first sets the output
    size) -> PNG (node `save`). `angles` adds the Multiple-Angles LoRA."""
    if not 1 <= len(images) <= MAX_REFS:
        raise ValueError(f"1..{MAX_REFS} images per edit")
    p: dict[str, Any] = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": EDIT_MODEL, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": EDIT_CLIP, "type": "qwen_image",
                                                        "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": EDIT_VAE}},
        "lightning": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0],
                                                                     "lora_name": LIGHTNING_LORA,
                                                                     "strength_model": 1.0}},
    }
    model = "lightning"
    if angles:
        p["angles"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["lightning", 0],
                                                                      "lora_name": ANGLES_LORA,
                                                                      "strength_model": opts.angle_strength}}
        model = "angles"
    p["sampling"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": [model, 0], "shift": 3.1}}
    p["cfgnorm"] = {"class_type": "CFGNorm", "inputs": {"model": ["sampling", 0], "strength": 1.0}}
    refs: dict[str, Any] = {}
    for i, name in enumerate(images, 1):
        p[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        p[f"scale{i}"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": [f"load{i}", 0]}}
        refs[f"image{i}"] = [f"scale{i}", 0]
    for k, text in (("pos", prompt), ("neg", "")):
        p[f"enc_{k}"] = {"class_type": "TextEncodeQwenImageEditPlus",
                         "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **refs}}
        p[k] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": [f"enc_{k}", 0], "reference_latents_method": "index_timestep_zero"}}
    p["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale1", 0], "vae": ["vae", 0]}}
    p["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["cfgnorm", 0], "seed": opts.seed, "steps": opts.steps, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
        "denoise": 1.0}}
    p["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    p["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return p


def clean_photos(photos: list[Path], out_dir: Path, url: str = DEFAULT_URL, opts: CleanOptions | None = None,
                 log: Any = print) -> dict[str, Path]:
    """Redraw noisy photos as clean views: {"front": ..., "left": ..., "back": ...} PNGs in `out_dir`
    (`clean_<view>.png`), ready for `imageto3d.photos_to_glb`. The first photo matters most."""
    if not photos:
        raise ValueError("at least one photo is needed")
    opts = opts or CleanOptions()
    comfy = Comfy(url)
    if not comfy.alive():
        raise ComfyError(f"ComfyUI is not running at {url} (see AGENTS.md: Photos to 3D)")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]

    def edit(images: list[str], prompt: str, angles: bool, view: str) -> Path:
        t = time.time()
        try:
            outs = comfy.run(edit_prompt(images, prompt, opts, angles, prefix=f"pokefit/clean_{tag}_{view}"),
                             progress=lambda s: log(f"cleaning photos ({view} view): {s:.0f} s"))
        except ComfyError:
            comfy.free()
            raise
        path = out_dir / f"clean_{view}.png"
        Image.open(io.BytesIO(comfy.fetch(_first_file(outs, "save")))).convert("RGB").save(path)
        log(f"cleaned the {view} view in {time.time() - t:.0f} s")
        return path

    n_photos = MAX_REFS - 1 if opts.pose else MAX_REFS       # the pose skeleton takes the last reference
    refs = [comfy.upload(f"pokefit_{tag}_ref{i}.png", _png(Image.open(p).convert("RGB")))
            for i, p in enumerate(photos[:n_photos])]
    if len(photos) > n_photos:
        log(f"cleaning photos: using the first {n_photos} of {len(photos)} photos")
    front_refs, prompt = refs, CLEAN_PROMPT
    if opts.pose:
        size = Image.open(photos[0]).size
        front_refs = [*refs, comfy.upload(f"pokefit_{tag}_pose.png", _png(pose_skeleton(*size)))]
        prompt += POSE_SKELETON_PROMPT
    views = {"front": edit(front_refs, prompt, False, "front")}
    front = comfy.upload(f"pokefit_{tag}_clean_front.png", views["front"].read_bytes())
    extra = refs[:MAX_REFS - 1] if opts.view_refs else []
    for v in opts.views:
        views[v] = edit([front, *extra], view_prompt(v), True, v)
    comfy.free()            # the edit model (~20 GB) must not crowd Pixal3D, which runs next
    return views


def report(opts: CleanOptions, views: dict[str, Path]) -> dict[str, Any]:
    """What the clean-up did, for the generation report."""
    return {"options": asdict(opts), "views": {v: str(p) for v, p in views.items()}}
