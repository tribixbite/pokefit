import numpy as np
import pytest
import trimesh

from conftest import two_footed
from pokefit.magnets import (Pocket, PocketSpec, auto_pockets, cut_pockets, find_feet, required_washer,
                             validate_pockets)
from pokefit.model3mf import Part

SPEC = PocketSpec()


def _part(m: trimesh.Trimesh) -> Part:
    return Part("body", 1, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))


def test_two_feet_found(box_feet: trimesh.Trimesh) -> None:
    feet = find_feet([box_feet])
    assert len(feet) == 2
    assert sorted(round(f.centroid.x, 3) for f in feet) == [-9.0, 9.0]


def test_two_footed_pockets_centred(box_feet: trimesh.Trimesh) -> None:
    pockets, checks = auto_pockets([box_feet], SPEC)
    assert [c.status for c in checks] == ["pass"]
    xs = sorted(p.x for p in pockets)
    assert xs == pytest.approx([-9.0, 9.0], abs=0.05)
    assert all(abs(p.y) < 0.05 for p in pockets)
    assert all(c.status == "pass" for c in validate_pockets([box_feet], pockets, SPEC))


def test_single_foot_two_pockets_far_apart() -> None:
    slab = trimesh.creation.box([30, 10, 8]).apply_translation([0, 0, 4])
    pockets, checks = auto_pockets([slab], SPEC)
    assert len(pockets) == 2 and checks[0].status == "pass"
    # Valid centres form a 24.2 x 4.2 rectangle (30/10 minus 2 x (2.09 + 0.8)).
    assert np.hypot(pockets[0].x - pockets[1].x, pockets[0].y - pockets[1].y) >= 22.0
    assert all(c.status == "pass" for c in validate_pockets([slab], pockets, SPEC))


def test_pockets_constrained_to_ring() -> None:
    slab = trimesh.creation.box([40, 10, 8]).apply_translation([0, 0, 4])
    ring = (3.0, 0.0, 5.29, 6.91)
    pockets, checks = auto_pockets([slab], SPEC, ring=ring)
    assert len(pockets) == 2 and checks[0].status == "pass"
    for p in pockets:
        assert 5.29 - 0.05 <= np.hypot(p.x - 3.0, p.y) <= 6.91 + 0.05


def test_ring_unreachable_fails() -> None:
    slab = trimesh.creation.box([40, 10, 8]).apply_translation([0, 0, 4])
    pockets, checks = auto_pockets([slab], SPEC, ring=(0.0, 60.0, 5.29, 6.91))
    assert pockets == [] and "steel ring" in checks[0].message


def test_small_feet_fail() -> None:
    small = two_footed(foot=4.0, gap=6.0)                 # 4 mm feet < 4.18 + 2 * 0.8
    pockets, checks = auto_pockets([small], SPEC)
    assert pockets == []
    assert checks[0].status == "fail"
    assert "inscribed" in checks[0].message


def test_low_ceiling_rejected() -> None:
    plate = trimesh.creation.box([20, 20, 2.5]).apply_translation([0, 0, 1.25])   # 2.2 deep + 0.6 > 2.5
    pockets, checks = auto_pockets([plate], SPEC)
    assert pockets == [] and checks[0].status == "fail"
    bad = validate_pockets([plate], [Pocket(0, 0)], SPEC)
    assert bad[0].status == "fail" and "covered" in bad[0].message


def test_cut_volume(box_feet: trimesh.Trimesh) -> None:
    pockets, _ = auto_pockets([box_feet], SPEC)
    (new,), checks = cut_pockets([_part(box_feet)], pockets, SPEC)
    m = new.mesh()
    assert m.is_watertight
    r = SPEC.diameter / 2
    polygon_area = 0.5 * 48 * r * r * np.sin(2 * np.pi / 48)
    assert box_feet.volume - m.volume == pytest.approx(2 * polygon_area * SPEC.depth, rel=0.01)
    assert checks[0].status == "pass"


def test_cut_skips_untouched_parts(box_feet: trimesh.Trimesh) -> None:
    far = trimesh.creation.box([2, 2, 2]).apply_translation([0, 30, 20])
    parts = [_part(box_feet), _part(far)]
    pockets, _ = auto_pockets([box_feet], SPEC)
    new, _ = cut_pockets(parts, pockets, SPEC)
    assert new[1] is parts[1]


def test_required_washer() -> None:
    od, id_ = required_washer([Pocket(-10, 0), Pocket(10, 0)], SPEC)
    assert od == pytest.approx(20 + 4.18)
    assert id_ == pytest.approx(2 * (10 - 2.09))


def test_cut_unmerged_soup(box_feet: trimesh.Trimesh) -> None:
    """A triangle soup (as read from STL) is merged before the boolean cut."""
    soup = Part("soup", 1, box_feet.vertices[box_feet.faces].reshape(-1, 3),
                np.arange(len(box_feet.faces) * 3).reshape(-1, 3))
    pockets, _ = auto_pockets([box_feet], SPEC)
    (new,), checks = cut_pockets([soup], pockets, SPEC)
    assert checks[0].status == "pass" and new.mesh().is_watertight
