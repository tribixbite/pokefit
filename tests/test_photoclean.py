"""Photo clean-up graphs (pokefit.photoclean): node wiring and options, without ComfyUI."""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from PIL import Image

from pokefit.photoclean import (ANGLES_LORA, LIGHTNING_LORA, POSE_PROMPT, CleanOptions, clean_photos, edit_prompt,
                                view_prompt)


def _links_ok(prompt: dict[str, Any]) -> None:
    """Every [node, slot] input names a node of the prompt."""
    for nid, node in prompt.items():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int):
                assert v[0] in prompt, (nid, k, v)


def test_clean_edit_takes_up_to_three_references() -> None:
    g = edit_prompt(["a.png", "b.png", "c.png"], "clean it", CleanOptions(seed=5))
    _links_ok(g)
    enc = g["enc_pos"]["inputs"]
    assert [enc[f"image{i}"] for i in (1, 2, 3)] == [["scale1", 0], ["scale2", 0], ["scale3", 0]]
    assert enc["prompt"] == "clean it" and g["enc_neg"]["inputs"]["prompt"] == ""
    assert g["lightning"]["inputs"]["lora_name"] == LIGHTNING_LORA and "angles" not in g
    assert g["sampling"]["inputs"]["model"] == ["lightning", 0]
    s = g["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["seed"]) == (8, 1.0, 5)
    assert g["latent"]["inputs"]["pixels"] == ["scale1", 0]       # the first photo sets the output size
    with pytest.raises(ValueError):
        edit_prompt([f"{i}.png" for i in range(4)], "x", CleanOptions())
    with pytest.raises(ValueError):
        edit_prompt([], "x", CleanOptions())


def test_view_edit_adds_the_angles_lora() -> None:
    g = edit_prompt(["front.png"], view_prompt("back"), CleanOptions(angle_strength=0.8), angles=True)
    _links_ok(g)
    assert g["angles"]["inputs"]["lora_name"] == ANGLES_LORA and g["angles"]["inputs"]["strength_model"] == 0.8
    assert g["sampling"]["inputs"]["model"] == ["angles", 0]
    assert g["enc_pos"]["inputs"]["prompt"] == "<sks> back view eye-level shot medium shot"
    # fal's side names are swapped against the rig's (measured: its "left side view" faces the image's right)
    assert view_prompt("left") == "<sks> right side view eye-level shot medium shot"


def test_clean_options_are_checked() -> None:
    for bad in ({"steps": 2}, {"angle_strength": 3}, {"views": ("top",)}):
        with pytest.raises(ValueError):
            CleanOptions(**bad)


class _FakeComfy(BaseHTTPRequestHandler):
    """Enough of ComfyUI's API for clean_photos: every SaveImage returns a small grey PNG."""
    prompts: list[dict[str, Any]] = []
    uploads: list[str] = []
    freed = 0

    def log_message(self, *a: Any) -> None:
        pass

    def _send(self, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path.startswith("/history/"):
            pid = u.path.rsplit("/", 1)[1]
            outs = {"save": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}
            return self._send(json.dumps({pid: {"status": {"status_str": "success", "completed": True},
                                                "outputs": outs}}).encode())
        if u.path == "/view":
            buf = io.BytesIO()
            Image.new("RGB", (64, 48), (128, 128, 128)).save(buf, "PNG")
            return self._send(buf.getvalue(), "image/png")
        return self._send(b"{}")             # /system_stats

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/prompt":
            type(self).prompts.append(json.loads(body)["prompt"])
            return self._send(json.dumps({"prompt_id": str(len(self.prompts) - 1)}).encode())
        if self.path == "/upload/image":
            name = f"up{len(self.uploads)}.png"
            type(self).uploads.append(name)
            return self._send(json.dumps({"name": name, "subfolder": ""}).encode())
        if self.path == "/free":
            type(self).freed += 1
        return self._send(b"{}")


def test_clean_photos_runs_front_then_views(tmp_path: Path) -> None:
    photos = []
    for i in range(4):
        photos.append(tmp_path / f"p{i}.png")
        Image.new("RGB", (80, 60), (200, 100, 0)).save(photos[-1])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeComfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        views = clean_photos(photos, tmp_path / "out", url=f"http://127.0.0.1:{srv.server_port}",
                             opts=CleanOptions(pose=True), log=lambda s: None)
    finally:
        srv.shutdown()
    assert sorted(views) == ["back", "front", "left"] and all(p.exists() for p in views.values())
    front, left, back = _FakeComfy.prompts
    assert "angles" not in front and front["enc_pos"]["inputs"]["prompt"].endswith(POSE_PROMPT)
    assert [front[f"load{i}"]["inputs"]["image"] for i in (1, 2, 3)] == ["up0.png", "up1.png", "up2.png"]
    # views: the clean front (upload 3) first, then the first two original photos
    assert [left[f"load{i}"]["inputs"]["image"] for i in (1, 2, 3)] == ["up3.png", "up0.png", "up1.png"]
    assert back["enc_pos"]["inputs"]["prompt"] == view_prompt("back") and "angles" in back
    assert _FakeComfy.freed == 1
