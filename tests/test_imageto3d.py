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

from pokefit.imageto3d import (FRAME, PAD, ComfyError, Framed, frame_views, generate_prompt, mask_prompt,
                               single_prompt,
                               photo_colours, photos_to_glb, project, recolour_glb, vertex_colour_glb, Camera)
from pokefit.model3mf import read_project


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
    one = single_prompt("p.png", faces=999)
    _links_ok(one)
    assert one["cond"]["class_type"] == "Pixal3DConditioning" and one["cond"]["inputs"]["camera_angle_x"] == ["fov", 0]
    assert one["unet"]["inputs"]["unet_name"].startswith("pixal3d_int8") and one["decimate"]["inputs"]["target_face_count"] == 999
    assert g["unet"]["inputs"]["unet_name"].startswith("pixal3d_multiview")
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
    rep = photos_to_glb(paths, out, url=fake_comfy, mirror_right=True, log=lambda *_: None, photo_colour=False)
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


@pytest.mark.parametrize("cam", [None, Camera(fov=30.0, pad=1.0)], ids=["rig", "single-image camera"])
def test_photo_colours_land_where_the_views_see(cam: Camera | None) -> None:
    """A sphere in the rig, a front photo red on the image's left and blue on its right: front vertices
    take those colours by side; vertices facing away keep (a correction of) the model colour. Also with
    the single-image node's camera (MoGe's FOV, unit cube unpadded)."""
    cam = cam or Camera()
    ball = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    v, n = np.asarray(ball.vertices), np.asarray(ball.vertex_normals)
    img = np.zeros((FRAME, FRAME, 3), np.uint8)
    img[:, : FRAME // 2] = (220, 30, 30)
    img[:, FRAME // 2:] = (30, 30, 220)
    px, py, _ = project(v, "front", cam)
    mask = np.zeros((FRAME, FRAME), bool)
    r = int(np.ptp(px) / 2) + 2
    yy, xx = np.mgrid[:FRAME, :FRAME]
    mask[(xx - FRAME / 2) ** 2 + (yy - FRAME / 2) ** 2 <= r * r] = True
    base = np.column_stack([v[:, 0] * 50 + 100, v[:, 1] * 50 + 100, v[:, 2] * 50 + 100])   # varied model colour
    out, share = photo_colours(v, n, base, {"front": Framed("front", Image.fromarray(img), 1.0, (0, 0, 0, 0), mask)}, cam)
    front = n[:, 1] < -0.8                                     # facing the front camera (at -y)
    left, right = front & (v[:, 0] < -0.05), front & (v[:, 0] > 0.05)
    assert np.all(out[left, 0] > 180) and np.all(out[left, 2] < 80)
    assert np.all(out[right, 2] > 180) and np.all(out[right, 0] < 80)
    away = n[:, 1] > 0.5
    assert 0.15 < share < 0.6 and not np.any(np.all(out[away] == img[0, 0], axis=1))


def test_vertex_colour_glb_round_trips(tmp_path: Path) -> None:
    box = trimesh.creation.box((0.02, 0.02, 0.02)).subdivide()
    srgb = np.where(box.vertices[:, :1] > 0, [200, 40, 40], [40, 40, 200]).astype(float)
    path = tmp_path / "c.glb"
    path.write_bytes(vertex_colour_glb(box.vertices, box.faces, srgb))
    cols = {tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in read_project(path).colours}
    assert any(abs(r - 200) < 6 and b < 60 for r, _, b in cols) and any(abs(b - 200) < 6 and r < 60 for r, _, b in cols)
    # recolour_glb keeps the geometry and writes colour (a view that sees nothing leaves the model colour)
    empty = Framed("front", Image.new("RGB", (FRAME, FRAME)), 1.0, (0, 0, 0, 0), np.zeros((FRAME, FRAME), bool))
    data, share = recolour_glb(path.read_bytes(), {"front": empty})
    assert share == 0.0 and data[:4] == b"glTF"
