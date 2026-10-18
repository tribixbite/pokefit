import numpy as np
import trimesh

from pokefit.checks import (bed_contact_check, islands_check, mesh_checks, overhang_check, printable_solid,
                            supports_verdict, thickness_check)
from pokefit.model3mf import Part


def _part(m: trimesh.Trimesh, name: str = "p") -> Part:
    return Part(name, 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))


def _by_id(checks):
    return {c.id.split(":")[0]: c for c in checks}


def test_cube_is_sound() -> None:
    c = _by_id(mesh_checks(_part(trimesh.creation.box([10, 10, 10]))))
    assert c["watertight"].status == "pass"
    assert c["winding"].status == "pass"
    assert c["manifold"].status == "pass"
    # The self-intersection check needs pymeshlab (an optional extra; absent in CI and in the browser).
    try:
        import pymeshlab  # noqa: F401
        assert c["selfint"].status == "pass"
    except ImportError:
        assert c["selfint"].status == "info"
    assert c["degenerate"].status == "pass"


def test_open_cube_fails() -> None:
    box = trimesh.creation.box([10, 10, 10])
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-1], process=False)
    c = _by_id(mesh_checks(_part(open_box)))
    assert c["watertight"].status == "fail"
    assert c["watertight"].value == 3
    assert c["manifold"].status == "fail"
    assert len(c["watertight"].highlight) == 3


def test_inverted_cube_fails() -> None:
    box = trimesh.creation.box([10, 10, 10])
    inv = trimesh.Trimesh(box.vertices, box.faces[:, ::-1], process=False)
    assert _by_id(mesh_checks(_part(inv)))["winding"].status == "fail"


def test_stl_soup_is_merged() -> None:
    box = trimesh.creation.box([10, 10, 10])
    soup = trimesh.Trimesh(box.vertices[box.faces].reshape(-1, 3), np.arange(36).reshape(-1, 3), process=False)
    assert _by_id(mesh_checks(_part(soup)))["watertight"].status == "pass"


def test_thin_plate_fails_thickness() -> None:
    plate = trimesh.creation.box([20, 20, 0.3]).apply_translation([0, 0, 0.15])
    assert thickness_check(plate).status == "fail"
    assert thickness_check(trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])).status == "pass"


def test_t_shape_needs_supports() -> None:
    stem = trimesh.creation.box([4, 4, 10]).apply_translation([0, 0, 5])
    bar = trimesh.creation.box([20, 4, 3]).apply_translation([0, 0, 11.4])
    t = printable_solid([_part(stem), _part(bar)])
    oh = overhang_check(t, 30)
    assert oh.status == "warn" and oh.value > 60          # 2 x 8 x 4 mm² under the arms
    assert supports_verdict([oh, islands_check(t)]).value == "yes"


def test_floating_island_detected() -> None:
    base = trimesh.creation.box([10, 10, 2]).apply_translation([0, 0, 1])
    floating = trimesh.creation.box([3, 3, 3]).apply_translation([20, 0, 6])
    m = printable_solid([_part(base), _part(floating)])
    isl = islands_check(m)
    assert isl.status == "fail" and isl.value == 1
    assert abs(isl.highlight[0][0] - 20) < 1.6


def test_cube_needs_no_supports() -> None:
    cube = trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])
    checks = [overhang_check(cube, 30), islands_check(cube)]
    assert supports_verdict(checks).value == "no"


def test_pocket_ceiling_ignored() -> None:
    cube = trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])
    hole = trimesh.creation.cylinder(radius=2.09, height=3.2, sections=48).apply_translation([0, 0, 0.6])
    cut = cube.difference(hole, engine="manifold")
    assert overhang_check(cut, 30).status == "warn"
    assert overhang_check(cut, 30, ignore_xy=[(0, 0, 2.09, 2.2)]).status == "pass"


def test_bed_contact() -> None:
    needle = trimesh.creation.box([2, 2, 30]).apply_translation([0, 0, 15])
    assert bed_contact_check(needle).status == "warn"
    block = trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])
    assert bed_contact_check(block).status == "pass"
