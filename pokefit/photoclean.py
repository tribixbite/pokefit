"""Noisy photos of a figure -> clean front / left / back views for `imageto3d`, via a local ComfyUI.

Stock or product photos rarely suit Pixal3D: the figure sits in an open ball or on a stand, the shots are
off-angle, glossy and cropped, and a leftover background turns into a slab of the mesh. This step redraws
them with **Qwen-Image-Edit-2511** (Alibaba Qwen, Apache-2.0; ComfyUI's native `TextEncodeQwenImageEditPlus`
nodes and the `image_qwen_image_edit_2511_int8` template's settings):

1. Clean front: one edit of up to three of the photos together ("Picture 1..3" references): the figure
   alone, facing the camera, on plain white, even light. `pose=True` also asks it to sit flat on its base
   with limbs and tail tucked close to the body (a better fit in a ball; edit models follow pose requests
   only loosely, so check the result).
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
POSE_PROMPT = (
    " Pose it sitting flat on its bottom, compact: legs, arms and tail tucked close to the body instead of "
    "sticking out.")
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
    pose: bool = False              # also re-pose: sit flat, limbs and tail tucked in
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

    refs = [comfy.upload(f"pokefit_{tag}_ref{i}.png", _png(Image.open(p).convert("RGB")))
            for i, p in enumerate(photos[:MAX_REFS])]
    if len(photos) > MAX_REFS:
        log(f"cleaning photos: using the first {MAX_REFS} of {len(photos)} photos")
    views = {"front": edit(refs, CLEAN_PROMPT + (POSE_PROMPT if opts.pose else ""), False, "front")}
    front = comfy.upload(f"pokefit_{tag}_clean_front.png", views["front"].read_bytes())
    extra = refs[:MAX_REFS - 1] if opts.view_refs else []
    for v in opts.views:
        views[v] = edit([front, *extra], view_prompt(v), True, v)
    comfy.free()            # the edit model (~20 GB) must not crowd Pixal3D, which runs next
    return views


def report(opts: CleanOptions, views: dict[str, Path]) -> dict[str, Any]:
    """What the clean-up did, for the generation report."""
    return {"options": asdict(opts), "views": {v: str(p) for v, p in views.items()}}
