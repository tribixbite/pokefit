import numpy as np
import pytest
import trimesh

from conftest import two_footed
from pokefit.magnets import (Pocket, PocketSpec, auto_pockets, cut_pockets, find_feet, height_field, required_washer,
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
    # Valid centres on each foot are a 2.2 mm square; every spot with ≥ 1 mm of
    # extra wall scores the same for walls (DEPTH_CAP), so spread may pull a
    # pocket ~0.2 mm toward that plateau's corner.
    assert xs == pytest.approx([-9.0, 9.0], abs=0.25)
    assert all(abs(p.y) < 0.25 for p in pockets)
    assert all(p.z == 0 for p in pockets)
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


def _raised_body(post: float = 2.0, lift: float = 1.5) -> trimesh.Trimesh:
    """A 20 x 20 x 8 block held `lift` mm above the bed by one small corner post."""
    body = trimesh.creation.box([20, 20, 8]).apply_translation([0, 0, lift + 4])
    leg = trimesh.creation.box([post, post, lift + 0.5]).apply_translation([8, 8, (lift + 0.5) / 2])
    return trimesh.boolean.union([body, leg], engine="manifold")


def test_pocket_in_raised_underside() -> None:
    """No flat foot: the pocket goes into the raised belly and stands proud, but never below the bed."""
    m = _raised_body()
    pockets, checks = auto_pockets([m], SPEC, count=1)
    assert len(pockets) == 1 and checks[0].status == "pass"
    (p,) = pockets
    # Seat = belly height + grip − depth = 1.5 + 1.2 − 2.2.
    assert p.z == pytest.approx(0.5, abs=0.02)
    (v,) = validate_pockets([m], pockets, SPEC)
    assert v.status == "pass" and "proud" in v.message
    (new,), _ = cut_pockets([_part(m)], pockets, SPEC)
    assert new.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)   # still stands on the same plane
    r = SPEC.diameter / 2
    area = 0.5 * 48 * r * r * np.sin(2 * np.pi / 48)
    assert m.volume - new.mesh().volume == pytest.approx(area * (p.z + SPEC.depth - 1.5), rel=0.02)


def test_high_belly_warns() -> None:
    m = _raised_body(lift=3.0)
    pockets, _ = auto_pockets([m], SPEC, count=1)
    assert pockets[0].z == pytest.approx(2.0, abs=0.02)
    (v,) = validate_pockets([m], pockets, SPEC)
    assert v.status == "warn" and "weaker hold" in v.message


def test_sloped_underside() -> None:
    """A wedge rising 1 mm per 10 mm: seat follows the highest point under the walled disc."""
    pts = np.array([[x, y, z] for x in (-15, 15) for y in (-10, 10) for z in (0, 10)], float)
    pts[pts[:, 2] == 0, 2] = (pts[pts[:, 2] == 0, 0] + 15) / 10        # underside z = (x + 15) / 10
    wedge = trimesh.convex.convex_hull(pts)
    wedge.apply_translation([0, 0, -wedge.bounds[0][2]])
    pockets, checks = auto_pockets([wedge], SPEC, count=1)
    (p,) = pockets
    need = SPEC.diameter / 2 + SPEC.wall
    assert p.z == pytest.approx(max(0.0, (p.x + need + 15) / 10 + SPEC.grip - SPEC.depth), abs=0.03)
    assert p.z >= 0


def test_three_pockets() -> None:
    slab = trimesh.creation.box([30, 30, 8]).apply_translation([0, 0, 4])
    pockets, checks = auto_pockets([slab], SPEC, count=3)
    assert len(pockets) == 3 and checks[0].status == "pass"
    xy = np.array([[p.x, p.y] for p in pockets])
    d = [np.hypot(*(xy[a] - xy[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
    assert min(d) >= 15                                      # spread over the slab, not bunched
    assert all(c.status == "pass" for c in validate_pockets([slab], pockets, SPEC))


def test_fewer_pockets_when_crowded() -> None:
    slab = trimesh.creation.box([8, 8, 8]).apply_translation([0, 0, 4])   # room for one walled pocket
    pockets, checks = auto_pockets([slab], SPEC, count=3)
    assert len(pockets) == 1 and checks[0].status == "warn" and "1 of 3" in checks[0].message


def test_height_field_overlapping_parts() -> None:
    """An inner part (an eye inside a head) does not end the first solid span early."""
    outer = trimesh.creation.box([10, 10, 10]).apply_translation([0, 0, 5])
    inner = trimesh.creation.box([4, 4, 4]).apply_translation([0, 0, 5])
    f = height_field([outer, inner])
    i, j = np.unravel_index(np.argmin(np.hypot(*np.meshgrid(*[a for a in f.xy()]))), f.z1.shape)
    assert f.z1[i, j] == pytest.approx(0.0, abs=1e-6) and f.z2[i, j] == pytest.approx(10.0, abs=1e-6)


def test_too_high_everywhere_fails() -> None:
    m = _raised_body(lift=4.0)                       # seat 3.0 mm > max_seat 2.0
    pockets, checks = auto_pockets([m], SPEC, count=1)
    assert pockets == [] and checks[0].status == "fail" and "3.0 mm above the washer" in checks[0].message
