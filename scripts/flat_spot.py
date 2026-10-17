"""Add a flat resting spot to a pokeball half-shell, and fill the inside of the
pole so the wall stays thick and the tiny pole layers become one flat area.

Works on the shell in its native frame (dome up, pole on +z), e.g. the
0_ball_shell_2x object extracted from pokeball.3mf by extract_3mf.py.
Geometry lives in pokefit.shell; this is the command-line wrapper.

  * Outer: the dome is cut by a horizontal plane so the cut face is a disk of
    `--disk` mm diameter (final print size).
  * Inner: the cavity above a horizontal plane is filled, giving an inner flat
    ceiling of `--fill` mm diameter. Keep --fill >= --disk so the wall at the
    pole is never thinner than the original wall.

Usage:
  python scripts/flat_spot.py <shell.stl> <out.stl> [--scale 0.6] [--disk 8] [--fill 10]
"""
import argparse
import sys
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pokefit.shell import flat_spot  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=float, default=1.0, help="print scale, e.g. 0.6 for the 60%% ball")
    ap.add_argument("--disk", type=float, default=8.0, help="outer flat disk diameter, final mm")
    ap.add_argument("--fill", type=float, default=10.0, help="inner flat ceiling diameter, final mm")
    a = ap.parse_args()
    if a.fill < a.disk:
        print("warning: --fill < --disk thins the wall at the pole")
    shell = trimesh.load(a.src, force="mesh")
    r = flat_spot(shell, a.scale, a.disk, a.fill)
    print(f"outer R={r.r_outer:.3f}  inner R={r.r_inner:.3f} (native)")
    print(f"disk {a.disk:.2f} mm, inner flat {a.fill:.2f} mm, wall at pole {r.pole_wall:.2f} mm "
          f"(original {r.original_wall:.2f} mm), height {r.mesh.extents[2]:.2f} mm")
    print(f"watertight={r.mesh.is_watertight} volume={r.mesh.volume:.1f} mm^3 "
          f"(added {r.mesh.volume - shell.volume * a.scale ** 3:+.1f})")
    r.mesh.export(a.dst)
