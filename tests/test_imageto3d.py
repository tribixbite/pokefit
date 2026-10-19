"""Photos -> GLB via ComfyUI (pokefit/imageto3d.py): framing, the prompt graphs, and the client against a fake
ComfyUI (the real one needs ~9 GB of weights and a GPU; see AGENTS.md "Photos to 3D")."""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import trimesh
from PIL import Image

from pokefit.imageto3d import FRAME, PAD, ComfyError, frame_views, generate_prompt, mask_prompt, photos_to_glb


def _photo(size: tuple[int, int], box: tuple[int, int, int, int], colour: tuple[int, int, int]) -> tuple[Image.Image, np.ndarray]:
    """A white photo with a coloured rectangle at `box` (x0, y0, x1, y1) and its mask."""
    img = np.full((size[1], size[0], 3), 250, np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = colour
    mask = np.zeros((size[1], size[0]), np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return Image.fromarray(img), mask


def _box(img: Image.Image) -> tuple[int, int, int, int]:
    a = np.asarray(img).max(axis=2) > 8
    ys, xs = np.nonzero(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_views_get_one_scale_and_are_centred() -> None:
    # The same figure: 200 wide x 400 tall from the front, 300 deep from the side, shot at other zooms.
    front, fm = _photo((800, 800), (100, 100, 300, 500), (40, 120, 200))      # 200 x 400 px
    left, lm = _photo((1000, 700), (500, 50, 950, 650), (200, 80, 40))        # 450 x 600 px (1.5x zoom)
    out = frame_views({"front": front, "left": left}, {"front": fm, "left": lm})
    fb, lb = _box(out["front"].image), _box(out["left"].image)
    hf, hl = fb[3] - fb[1], lb[3] - lb[1]
    assert abs(hf - hl) <= 2                                                   # same height in every view
    widest = max(fb[2] - fb[0], lb[2] - lb[0], hf, hl)
    assert abs(widest - FRAME / PAD) <= 3                                      # largest extent at FRAME / PAD
    for b in (fb, lb):
        assert abs((b[0] + b[2]) / 2 - FRAME / 2) <= 2 and abs((b[1] + b[3]) / 2 - FRAME / 2) <= 2
    assert out["front"].image.size == (FRAME, FRAME)
    assert np.asarray(out["front"].image)[5, 5].max() == 0                    # background black


def test_empty_mask_is_an_error() -> None:
    img, _ = _photo((64, 64), (0, 0, 1, 1), (0, 0, 0))
    with pytest.raises(ValueError, match="no figure"):
        frame_views({"front": img}, {"front": np.zeros((64, 64), np.float32)})


def _links_ok(prompt: dict[str, Any]) -> None:
    for nid, node in prompt.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in prompt, f"{nid} links to missing node {v[0]}"


def test_prompts_are_wired() -> None:
    g = generate_prompt({"front": "f.png", "back": "b.png"}, seed=3, faces=1234)
    _links_ok(g)
    cond = g["cond"]["inputs"]
    assert cond["front"] == ["view_front", 0] and cond["back"] == ["view_back", 0] and "left" not in cond
    assert g["decimate"]["inputs"]["target_face_count"] == 1234
    assert g["remesh"]["inputs"]["sign_mode"] == "solid"                       # one watertight solid
    assert g["s_struct"]["inputs"]["seed"] == 59
    m = mask_prompt({"front": "f.png", "left": "l.png"})
    _links_ok(m)
    assert {k for k in m if k.startswith("save_")} == {"save_front", "save_left"}


class FakeComfy(BaseHTTPRequestHandler):
    """Enough of ComfyUI's API for photos_to_glb: masks are the photo's non-white pixels."""
    uploads: dict[str, bytes] = {}
    prompts: list[dict[str, Any]] = []
    glb = b""

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
        if u.path == "/system_stats":
            return self._send(b"{}")
        if u.path.startswith("/history/"):
            pid = u.path.rsplit("/", 1)[1]
            prompt = self.prompts[int(pid)]
            outs: dict[str, Any] = {}
            for nid, node in prompt.items():
                if node["class_type"] == "SaveImage":
                    outs[nid] = {"images": [{"filename": f"{pid}_{nid}.png", "subfolder": "", "type": "output"}]}
                if node["class_type"] == "Save3DAdvanced":
                    outs[nid] = {"3d": [{"filename": "fig.glb", "subfolder": "pokefit", "type": "output"}]}
            return self._send(json.dumps({pid: {"status": {"status_str": "success", "completed": True},
                                                "outputs": outs}}).encode())
        if u.path == "/view":
            name = parse_qs(u.query)["filename"][0]
            if name.endswith(".glb"):
                return self._send(self.glb, "model/gltf-binary")
            pid, node = name[:-4].split("_", 1)
            view = node.removeprefix("save_")
            src = self.prompts[int(pid)][f"load_{view}"]["inputs"]["image"]
            photo = np.asarray(Image.open(io.BytesIO(self.uploads[src])).convert("RGB"))
            buf = io.BytesIO()
            Image.fromarray(((photo.min(axis=2) < 200) * 255).astype(np.uint8)).save(buf, "PNG")
            return self._send(buf.getvalue(), "image/png")
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/upload/image":
            head, rest = body.split(b'filename="', 1)
            name = rest.split(b'"', 1)[0].decode()
            data = rest.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            self.uploads[name] = data
            return self._send(json.dumps({"name": name, "subfolder": "", "type": "input"}).encode())
        if self.path == "/prompt":
            self.prompts.append(json.loads(body)["prompt"])
            return self._send(json.dumps({"prompt_id": str(len(self.prompts) - 1)}).encode())
        self.send_error(404)


@pytest.fixture
def fake_comfy() -> Any:
    FakeComfy.uploads, FakeComfy.prompts = {}, []
    FakeComfy.glb = trimesh.Scene(trimesh.creation.box((0.5, 0.5, 0.5))).export(file_type="glb")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_photos_to_glb_against_a_fake_comfy(tmp_path: Path, fake_comfy: str) -> None:
    paths = {}
    for v, box in (("front", (20, 10, 60, 90)), ("left", (30, 20, 80, 100))):
        img, _ = _photo((120, 120), box, (30, 90, 200))
        paths[v] = tmp_path / f"{v}.png"
        img.save(paths[v])
    out = tmp_path / "fig.glb"
    rep = photos_to_glb(paths, out, url=fake_comfy, mirror_right=True, log=lambda *_: None)
    assert out.read_bytes() == FakeComfy.glb and rep["views"] == ["front", "left", "right"]
    gen = FakeComfy.prompts[1]
    assert set(gen["cond"]["inputs"]) >= {"front", "left", "right"} and "back" not in gen["cond"]["inputs"]
    framed = Image.open(tmp_path / "fig" / "view_right.png")
    left = Image.open(tmp_path / "fig" / "view_left.png")
    assert np.array_equal(np.asarray(framed), np.asarray(left)[:, ::-1])       # mirrored left


def test_needs_a_front_and_a_running_comfy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="front"):
        photos_to_glb({"left": tmp_path / "l.png"}, tmp_path / "x.glb")
    with pytest.raises(ComfyError, match="not running"):
        photos_to_glb({"front": tmp_path / "f.png"}, tmp_path / "x.glb", url="http://127.0.0.1:9")
