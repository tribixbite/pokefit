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


def test_manual_pockets_keep_fit_and_bad_pocket_not_cut(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    auto, _ = analyse(s, Params(washer="US1/4x1"))
    good = auto.pockets[0]
    rep, final = analyse(s, Params(washer="US1/4x1", figure_scale=auto.fit["scale"],
                                   pockets=[good, {"x": 40.0, "y": 0.0}]))
    c = _by_id(rep)
    assert c["fit"]["status"] == "pass"                      # the fit is never traded for the steel ring
    assert c["pocket:2"]["status"] == "fail" and "Not cut" in c["pocket:2"]["message"]
    removed = c[next(k for k in c if k.startswith("cut:"))]["value"]
    assert removed == pytest.approx(np.pi * 2.09 ** 2 * 2.2, rel=0.05)   # only one pocket cut


def test_magnet_count(critter: Path, tmp_path: Path) -> None:
    assert Params(magnets=True).magnets == 2 and Params(magnets=False).magnets == 0
    assert Params.from_dict({"magnets": 5}).magnets == 3
    s = open_session(critter, BALL_3MF, None, tmp_path)
    # Each 9 mm foot holds one pocket; a third would go under the belly 11 mm up (over max_seat).
    for n, placed in ((1, 1), (3, 2)):
        rep, _ = analyse(s, Params(washer="US1/4x1", magnets=n))
        c = _by_id(rep)
        assert len(rep.pockets) == placed, (n, c["pockets:feet"]["message"])
        assert c["pockets:feet"]["status"] == ("pass" if n == placed else "warn")
        assert all(c[f"pocket:{i + 1}"]["status"] != "fail" for i in range(placed))
        assert all(p["z"] == 0 for p in rep.pockets)              # flat feet: magnets sit on the washer
        assert rep.underside["height_png"].startswith("data:image/png;base64,")
        assert len(rep.underside["extent"]) == 4 and rep.underside["valid"]


def test_export_import_variant(critter: Path, tmp_path: Path) -> None:
    """With the user's project settings, export also writes a file for "Import geometry only"."""
    from pokefit.paint import decode_weights
    from pokefit.pipeline import BUNDLED_REFERENCE

    s = open_session(critter, BALL_3MF, BUNDLED_REFERENCE, tmp_path / "w")
    out = export(s, Params(washer="US1/4x1"), tmp_path / "out")
    imp = read_project(out["import_3mf"])
    targets = s.reference["filament_colour"]
    assert imp.colours == targets                                    # carries the project's own slots
    m = {x["source"]: x["target"] for x in out["import_map"]}
    assert set(m) == {1, 2}                                          # body + painted top
    obj = imp.default_object()
    assert {p.extruder for p in obj.parts} == {m[1]}
    painted = {k for p in obj.parts for a in p.tri_extra.values() for k in decode_weights(a.split('"')[1])}
    assert painted == {m[2]}


def test_inner_diameter_and_free_rotation(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    by_scale, _ = analyse(s, Params(ball_scale=0.6, magnets=0))
    by_diam, _ = analyse(s, Params(ball_diameter=by_scale.ball["inner_diameter"], magnets=0))
    assert by_diam.ball["scale"] == pytest.approx(0.6)
    assert by_diam.fit["max_scale"] == pytest.approx(by_scale.fit["max_scale"], rel=1e-6)
    tilted, _ = analyse(s, Params(rotate_z=45, magnets=0))
    # The block is 26 x 9 mm in plan: turned 45° about z its footprint is (26 + 9) / sqrt(2) square.
    ext = tilted.fit["unit_extents"]
    assert ext[0] == pytest.approx(ext[1], abs=1e-6) and ext[0] == pytest.approx(35 / np.sqrt(2), abs=1e-3)
    lean, _ = analyse(s, Params(rotate_x=30, magnets=0))
    assert lean.fit["scale"] > 0
