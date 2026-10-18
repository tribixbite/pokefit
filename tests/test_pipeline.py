import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import BALL_3MF, REAL_DRAGON, two_footed, write_inline_3mf
from pokefit.model3mf import read_project
from pokefit.pipeline import Params, analyse, export, open_session, preview_glb


@pytest.fixture(scope="module")
def critter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-footed block figure 30 mm tall, painted on the top face."""
    m = two_footed(foot=9.0, gap=8.0, leg_h=12.0, body_h=18.0)
    top = {int(i): "8" for i in np.nonzero(m.face_normals[:, 2] > 0.99)[0]}
    return write_inline_3mf(tmp_path_factory.mktemp("in") / "critter.3mf", m, np.eye(4), top)


def _by_id(rep) -> dict:
    return {c["id"]: c for c in rep.checks}


def test_analyse_two_footed(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    rep, final = analyse(s, Params(ball_scale=0.6, washer="US1/4x1"))
    c = _by_id(rep)
    assert c["fit"]["status"] == "pass"
    assert 0 < rep.fit["scale"] <= rep.fit["max_scale"] + 1e-9
    assert len(rep.pockets) == 2
    assert c["steel"]["status"] == "pass"
    assert all(c[f"pocket:{i}"]["status"] == "pass" for i in (1, 2))
    # Pockets were cut: volume dropped by about two pockets.
    before = sum(trimesh.Trimesh(p.vertices, p.faces).volume for p in s.unit_parts(Params())) * rep.fit["scale"] ** 3
    after = sum(trimesh.Trimesh(p.vertices, p.faces).volume for p in final)
    assert before - after == pytest.approx(2 * np.pi * 2.09 ** 2 * 2.2, rel=0.05)
    # Fits the cavity within the clearance: the limiting margin is ~0 at max scale.
    assert rep.fit["margin"] == pytest.approx(0.0, abs=0.05)


def test_fixed_scale_and_no_magnets(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    rep, final = analyse(s, Params(figure_scale=0.5, magnets=False))
    assert rep.fit["scale"] == 0.5 and rep.pockets == []
    assert np.ptp(np.vstack([p.vertices for p in final])[:, 2]) == pytest.approx(0.5 * rep.fit["unit_height"], abs=1e-6)


def test_too_big_scale_fails(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    rep, _ = analyse(s, Params(figure_scale=3.0, magnets=False))
    assert _by_id(rep)["fit"]["status"] == "fail" and rep.status == "fail"


def test_export_files(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path / "w")
    out = export(s, Params(washer="US1/4x1"), tmp_path / "out")
    f3 = Path(out["figure_3mf"])
    assert f3.name == "critter_pokefit_60.3mf"
    back = read_project(f3).default_object()
    # Paint survived: the top face triangles are still painted.
    assert sum(len(p.tri_extra) for p in back.parts) >= 2
    shell = trimesh.load(out["shell_stl"])
    assert shell.is_watertight
    rep = json.loads(Path(out["report"]).read_text(encoding="utf-8"))
    assert rep["fit"]["scale"] > 0


def test_preview_glb(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    glb = preview_glb(s, Params())
    assert glb[:4] == b"glTF"
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    assert len(scene.geometry) == 2          # unpainted + painted (extruder 2)


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DRAGON.exists(), reason="real dragon 3MF not present")
def test_real_dragon_end_to_end(tmp_path: Path) -> None:
    s = open_session(REAL_DRAGON, BALL_3MF, None, tmp_path)
    rep, _ = analyse(s, Params(ball_scale=0.6, washer="M6"))
    c = _by_id(rep)
    assert c["fit"]["status"] == "pass"
    assert rep.fit["height_mm"] > 30
    assert len(rep.pockets) == 2 and c["steel"]["status"] == "pass"
    out = export(s, Params(ball_scale=0.6, washer="M6"), tmp_path / "out")
    with zipfile.ZipFile(out["figure_3mf"]) as z:
        assert "3D/Objects/object_1.model" in z.namelist()
