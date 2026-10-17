import numpy as np
import pytest
import trimesh

from pokefit.shell import cap_height, fit_sphere, flat_spot, washer_seat_depth


def test_fit_sphere_exact() -> None:
    rng = np.random.default_rng(1)
    d = rng.normal(size=(500, 3))
    p = d / np.linalg.norm(d, axis=1)[:, None] * 7.5 + [1, -2, 3]
    c, r = fit_sphere(p)
    assert np.allclose(c, [1, -2, 3], atol=1e-9)
    assert r == pytest.approx(7.5, abs=1e-9)


def test_cap_height() -> None:
    assert cap_height(10, 12) == pytest.approx(8.0)
    assert washer_seat_depth(10, 12) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cap_height(5, 10)


def _hemishell(r_out: float = 20.0, r_in: float = 18.0) -> trimesh.Trimesh:
    """Dome shell, pole on +z, rim on z=0."""
    outer = trimesh.creation.icosphere(subdivisions=5, radius=r_out)
    inner = trimesh.creation.icosphere(subdivisions=5, radius=r_in)
    shell = outer.difference(inner, engine="manifold")
    return shell.slice_plane([0, 0, 0], [0, 0, 1], cap=True)


def test_flat_spot_hemishell() -> None:
    res = flat_spot(_hemishell(), scale=1.0, disk=8, fill=10)
    m = res.mesh
    assert m.is_watertight and len(m.split(only_watertight=False)) == 1
    assert m.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    # Pole wall: outer cut at cap_height(20, 8), inner flat at cap_height(18, 10).
    assert res.pole_wall == pytest.approx(cap_height(res.r_outer, 8) - cap_height(res.r_inner, 10), abs=1e-6)
    assert res.pole_wall > res.original_wall
