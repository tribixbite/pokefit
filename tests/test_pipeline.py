import base64
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import BALL_3MF, REAL_DRAGON, two_footed, write_inline_3mf
from pokefit.model3mf import read_project
from pokefit.pipeline import Params, analyse, export, open_session, preview_faces, preview_glb


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
    rep, final = analyse(s, Params(ball_scale=0.6, steel="US1/4x1"))
    c = _by_id(rep)
    assert c["fit"]["status"] == "pass"
    assert 0 < rep.fit["scale"] <= rep.fit["max_scale"] + 1e-9
    assert len(rep.pockets) == 2
    assert c["steel"]["status"] == "pass"
    assert all(c[f"pocket:{i}"]["status"] == "pass" for i in (1, 2))
    # Pockets were cut: volume dropped by about two pockets.
    before = sum(trimesh.Trimesh(p.vertices, p.faces).volume
                 for p in s.unit_parts(Params(ball_scale=0.6, steel="US1/4x1"))) * rep.fit["scale"] ** 3
    after = sum(trimesh.Trimesh(p.vertices, p.faces).volume for p in final)
    assert before - after == pytest.approx(2 * np.pi * 2.09 ** 2 * 2.2, rel=0.05)
    # Fits the cavity within the clearance: the limiting margin is ~0 at max scale.
    assert rep.fit["margin"] == pytest.approx(0.0, abs=0.05)


def test_quick_analysis_leaves_the_printability_checks_for_a_second_call(critter: Path, tmp_path: Path) -> None:
    """`printing=False` (the UI's first call): fit and pockets as in a full analysis, the printability checks
    pending; the full call adds them, and a quick call after that has them from the cache."""
    s = open_session(critter, BALL_3MF, None, tmp_path)
    p = Params(ball_scale=0.6)
    quick, _ = analyse(s, p, printing=False)
    printing_ids = {"thickness", "overhang", "islands", "bed", "supports"}
    assert quick.printing_pending and "printing" in _by_id(quick)
    assert not printing_ids & set(_by_id(quick))
    full, _ = analyse(s, p)
    assert not full.printing_pending and "printing" not in _by_id(full)
    assert printing_ids <= set(_by_id(full)), sorted(_by_id(full))
    assert quick.fit == full.fit and quick.pockets == full.pockets
    again, _ = analyse(s, p, printing=False)
    assert not again.printing_pending and _by_id(again) == _by_id(full)


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
    out = export(s, Params(steel="US1/4x1"), tmp_path / "out")
    f3 = Path(out["figure_3mf"])
    assert f3.name == "critter_pokefit_60.3mf"
    back = read_project(f3).default_object()
    # Paint survived: the top face triangles are still painted.
    assert sum(len(p.tri_extra) for p in back.parts) >= 2
    # The steel-seat shell is cut from the third-party ball model, present only locally.
    from pokefit.pipeline import BALL_MODEL
    if BALL_MODEL.exists():
        assert trimesh.load(out["shell_stl"]).is_watertight
    else:
        assert out["shell_stl"] is None and "ball model" in out["note"]
    rep = json.loads(Path(out["report"]).read_text(encoding="utf-8"))
    assert rep["fit"]["scale"] > 0


def test_paint_depth_reaches_the_exported_settings(critter: Path, tmp_path: Path) -> None:
    from pokefit.pipeline import BUNDLED_REFERENCE
    s = open_session(critter, BALL_3MF, BUNDLED_REFERENCE, tmp_path / "w")
    for depth, want in ((0.8, "0.8"), (0, "0")):
        out = export(s, Params(steel="US1/4x1", paint_depth=depth), tmp_path / f"out{depth}")
        with zipfile.ZipFile(out["figure_3mf"]) as z:
            st = json.loads(z.read("Metadata/project_settings.config"))
        assert st["mmu_segmented_region_max_width"] == want
    # None keeps the project's own value (the bundled reference has "0").
    out = export(s, Params(steel="US1/4x1"), tmp_path / "outnone")
    with zipfile.ZipFile(out["figure_3mf"]) as z:
        st = json.loads(z.read("Metadata/project_settings.config"))
    assert st.get("mmu_segmented_region_max_width") == "0"
    for bad in (-1, 11):
        with pytest.raises(ValueError):
            Params(paint_depth=bad)


def test_preview_glb(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    glb = preview_glb(s, Params())
    assert glb[:4] == b"glTF"
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    parts = s.unit_parts(Params())
    assert sorted(scene.geometry) == [f"p{i}" for i in range(len(parts))]     # one mesh per part
    faces = preview_faces(s, Params(colours=["#cccccc", "#e05028"]))
    assert faces["colours"] == ["#cccccc", "#e05028"]
    idx = np.frombuffer(base64.b64decode(faces["parts"][0]), np.uint8)
    assert len(idx) == len(scene.geometry["p0"].faces) and set(idx.tolist()) == {0, 1}   # unpainted + painted


@pytest.mark.slow
@pytest.mark.skipif(not REAL_DRAGON.exists(), reason="real dragon 3MF not present")
def test_real_dragon_end_to_end(tmp_path: Path) -> None:
    s = open_session(REAL_DRAGON, BALL_3MF, None, tmp_path)
    rep, _ = analyse(s, Params(ball_scale=0.6, steel="M6"))
    c = _by_id(rep)
    assert c["fit"]["status"] == "pass"
    assert rep.fit["height_mm"] > 30
    assert len(rep.pockets) == 2 and c["steel"]["status"] == "pass"
    out = export(s, Params(ball_scale=0.6, steel="M6"), tmp_path / "out")
    with zipfile.ZipFile(out["figure_3mf"]) as z:
        assert "3D/Objects/object_1.model" in z.namelist()


def test_manual_pockets_keep_fit_and_bad_pocket_not_cut(critter: Path, tmp_path: Path) -> None:
    s = open_session(critter, BALL_3MF, None, tmp_path)
    auto, _ = analyse(s, Params(steel="US1/4x1"))
    good = auto.pockets[0]
    rep, final = analyse(s, Params(steel="US1/4x1", figure_scale=auto.fit["scale"],
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
        rep, _ = analyse(s, Params(steel="US1/4x1", magnets=n))
        c = _by_id(rep)
        assert len(rep.pockets) == placed, (n, c["pockets:feet"]["message"])
        assert c["pockets:feet"]["status"] == ("pass" if n == placed else "warn")
        assert all(c[f"pocket:{i + 1}"]["status"] != "fail" for i in range(placed))
        assert all(p["z"] == 0 for p in rep.pockets)              # flat feet: magnets sit on the steel
        assert rep.underside["height_png"].startswith("data:image/png;base64,")
        assert len(rep.underside["extent"]) == 4 and rep.underside["valid"]


def test_export_import_variant(critter: Path, tmp_path: Path) -> None:
    """With the user's project settings, export also writes a file for "Import geometry only"."""
    from pokefit.paint import decode_weights
    from pokefit.pipeline import BUNDLED_REFERENCE

    s = open_session(critter, BALL_3MF, BUNDLED_REFERENCE, tmp_path / "w")
    out = export(s, Params(steel="US1/4x1"), tmp_path / "out")
    imp = read_project(out["import_3mf"], as_written=True)
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


def test_units_hint_for_tiny_files(tmp_path: Path) -> None:
    """A figure 0.9 mm tall as saved was modelled in metres/cm: an info check says so; mm figures get none."""
    tiny = write_inline_3mf(tmp_path / "tiny.3mf", trimesh.creation.box([0.6, 0.4, 0.9]), np.eye(4))
    rep, _ = analyse(open_session(tiny, BALL_3MF, None, tmp_path / "w"), Params(magnets=0))
    c = _by_id(rep)
    assert c["units"]["status"] == "info" and "metres" in c["units"]["message"]
    assert rep.fit["height_mm"] > 10                     # still fitted to the ball in mm


def test_older_washer_names_still_mean_the_steel() -> None:
    """Settings saved before steel discs: "washer" was the steel's key, and the value of the magnet seat and the
    fill area; they map onto the new names (the remembered washer is kept)."""
    p = Params.from_dict({"washer": "M8", "pocket_seat": "washer", "fill_area": "washer"})
    assert (p.steel, p.pocket_seat, p.fill_area) == ("M8", "steel", "steel")
    assert Params.from_dict({"washer": "M8", "steel": "D15"}).steel == "D15"
    assert Params().steel == "D20"


def test_a_dense_figure_opens_simplified_unless_asked_otherwise(tmp_path: Path) -> None:
    """Over DENSE_FACES triangles with Simplify unset (and `simplify_auto` on), the file opens at AUTO_SIMPLIFY; a
    Simplify setting of the user's, or `simplify_auto` off, is kept; a small figure is left alone."""
    from pokefit.pipeline import AUTO_SIMPLIFY, DENSE_FACES, auto_simplify

    dense = trimesh.creation.icosphere(subdivisions=7, radius=10)           # 327,680 triangles
    assert len(dense.faces) > DENSE_FACES
    stl = tmp_path / "dense.stl"
    dense.export(stl)
    s = open_session(stl, BALL_3MF, None, tmp_path / "w")
    p = auto_simplify(s, Params())
    assert p.simplify == AUTO_SIMPLIFY
    s.use_simplify(p.simplify)
    loaded = sum(len(q.faces) for q in s.unit_parts(p))
    assert abs(loaded - AUTO_SIMPLIFY) < 0.05 * AUTO_SIMPLIFY
    assert auto_simplify(s, Params(simplify=0.5)).simplify == 0.5
    assert auto_simplify(s, Params(simplify_auto=False)).simplify is None
    small = tmp_path / "small.stl"
    trimesh.creation.icosphere(subdivisions=3).export(small)
    assert auto_simplify(open_session(small, BALL_3MF, None, tmp_path / "w2"), Params()).simplify is None
    # A pokefit export of the same dense figure opens with every triangle (its paint is not resampled).
    s.project.pokefit = {"version": 1, "params": {"flat_base": 0.0}}
    assert auto_simplify(s, Params()).simplify is None


def test_exports_from_a_file_of_several_figures_are_named_after_the_figure(tmp_path: Path) -> None:
    """A 3MF holding several figures ("Eevee+Evolutions.3mf"): the files are named after the chosen one."""
    def mesh_xml(m: trimesh.Trimesh) -> str:
        v = "".join(f'<vertex x="{a}" y="{b}" z="{c}"/>' for a, b, c in m.vertices.tolist())
        f = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces.tolist())
        return f"<mesh><vertices>{v}</vertices><triangles>{f}</triangles></mesh>"

    a = trimesh.creation.box([10, 10, 20]).apply_translation([0, 0, 10])
    b = trimesh.creation.icosphere(subdivisions=2, radius=8).apply_translation([30, 0, 8])
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'
             f'<object id="1" type="model" name="Vaporeon.stl">{mesh_xml(a)}</object>'
             f'<object id="2" type="model" name="jolteon.stl">{mesh_xml(b)}</object>'
             '</resources><build><item objectid="1"/><item objectid="2"/></build></model>')
    path = tmp_path / "Eevee+Evolutions.3mf"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", model)
    s = open_session(path, BALL_3MF, None, tmp_path / "w")
    jolt = next(o.id for o in s.project.objects if o.name == "jolteon.stl")
    out = export(s, Params(object_id=jolt, magnets=0), tmp_path / "out")
    assert Path(out["figure_3mf"]).name.startswith("jolteon_pokefit_")
    assert Path(out["report"]).name.startswith("jolteon_pokefit_")


def test_core_filament_keeps_the_surface_and_prints_the_inside_in_it() -> None:
    from pokefit.model3mf import Part
    from pokefit.paint import decode_weights, encode_tree
    from pokefit.pipeline import cored, face_colours

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    split = encode_tree((1, 0, [1, 0]))  # a split triangle: one half filament 1, the other unpainted (state 0)
    assert decode_weights(split).get(0, 0) > 0
    part = Part("body", 0, v, f, {1: 'paint_color="8"', 2: f'paint_supports="4" paint_color="{split}"',
                                  3: 'paint_supports="4"'})
    mod = Part("solid", 0, v, f, {}, "modifier_part", {"wall_loops": "6"})
    out, ext = cored([part, mod], 3, 4)
    assert ext == 4 and out[0].extruder == 4 and out[1] is mod
    # Every triangle is painted now, the colour it showed (face 0 and 3 were the object's 3).
    assert len(out[0].tri_extra) == 4 and all('paint_color="' in a for a in out[0].tri_extra.values())
    # (face 2 is half and half: its majority is a tie, so compare its pieces below)
    assert face_colours(out[0], [], 4)[[0, 1, 3]].tolist() == face_colours(part, [], 3)[[0, 1, 3]].tolist() == [3, 2, 3]
    assert 'paint_supports="4"' in out[0].tri_extra[2] and 'paint_supports="4"' in out[0].tri_extra[3]
    old, new = decode_weights(split), decode_weights(out[0].tri_extra[2].split('paint_color="')[1].split('"')[0])
    assert 0 not in new and new[3] == pytest.approx(old[0]) and new[1] == pytest.approx(old[1])
    # Nothing to do: no core, or the core is what everything already inherits.
    assert cored([part], 3, None) == ([part], 3) and cored([part], 3, 3) == ([part], 3)
    for bad in (0, 65):
        with pytest.raises(ValueError):
            Params(core_filament=bad)


def test_core_filament_reaches_the_export(critter: Path, tmp_path: Path) -> None:
    import re

    from pokefit.pipeline import BUNDLED_REFERENCE

    s = open_session(critter, BALL_3MF, BUNDLED_REFERENCE, tmp_path / "w")

    def written(**kw: object) -> tuple[list[str], list[str]]:
        """The exported figure's triangles' paint (raw XML: reading it back strips the pockets) and filaments."""
        out = export(s, Params(steel="US1/4x1", paint_depth=1.6, **kw), tmp_path / f"out{len(kw)}")
        with zipfile.ZipFile(out["figure_3mf"]) as z:
            model = "".join(z.read(n).decode() for n in z.namelist() if n.endswith(".model"))
            cfg = z.read("Metadata/model_settings.config").decode()
        tris = re.findall(r"<triangle [^>]*/>", model)
        return ([m.group(1) if (m := re.search(r'paint_color="([0-9A-F]*)"', tri)) else "" for tri in tris],
                re.findall(r'key="extruder" value="(\d+)"', cfg))
    plain, plain_ext = written()
    core, core_ext = written(core_filament=2)
    assert set(plain_ext) == {"1"} and set(core_ext) == {"2"}
    assert len(core) == len(plain) and "" in plain and "" not in core
    # The surface is the same: what was unpainted (filament 1, inherited) is now painted 1 ("4").
    assert [c or "4" for c in plain] == core
