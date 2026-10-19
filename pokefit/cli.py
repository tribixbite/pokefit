"""Command line: `python -m pokefit fit|serve|ball`.

  python -m pokefit fit figure.3mf [--ball 0.6 | --diameter 44] [--washer M6] [--scale 85%] [--magnets 2] [--out DIR] [--slice] [--x10]
  python -m pokefit serve [--port 8765] [--open]
  python -m pokefit ball [--ball 0.6] [--washer M6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .washer import CATALOGUE, DEFAULT_WASHER

ICON = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}


def _scale(s: str) -> float | None:
    """'max', '85%', or a factor like 0.85."""
    if s in ("max", ""):
        return None
    return float(s[:-1]) / 100 if s.endswith("%") else float(s)


def _fmt_time(sec: float) -> str:
    h, m = divmod(int(round(sec / 60)), 60)
    return f"{h}h {m:02d}m" if h else f"{m} min"


def cmd_fit(a: argparse.Namespace) -> int:
    from .pipeline import Params, analyse, export, open_session, slice_variants

    params = Params(ball_scale=a.ball, ball_diameter=a.diameter, washer=a.washer, clearance=a.clearance,
                    figure_scale=_scale(a.scale), magnets=0 if a.no_magnets else a.magnets, object_id=a.object,
                    rotate_x=a.rotate_x, rotate_y=a.rotate_y, rotate_z=a.rotate_z, pocket_seat=a.magnet_seat,
                    texture_colours=a.texture_colours)
    s = open_session(a.input)
    rep, _ = analyse(s, params, recommend=a.washers)
    f = rep.fit
    print(f"{rep.object['name']}  ({rep.object['faces']} faces, {len(rep.object['parts'])} parts)")
    print(f"ball {rep.ball['scale']:.0%} (inner Ø {rep.ball['inner_diameter']:.1f} mm): floor z "
          f"{rep.ball['floor_z']:.1f}, washer {rep.ball['washer']['label']}")
    if f["scale"] > 0:
        print(f"scale {f['scale']:.1%} -> {f['height_mm']:.1f} mm tall (max {f['max_scale']:.1%} = "
              f"{f['max_height_mm']:.1f} mm), offset ({f['offset'][0]:.1f}, {f['offset'][1]:.1f}) mm")
    for p in rep.pockets:
        print(f"pocket at ({p['x']:.2f}, {p['y']:.2f}) mm, figure frame, opening {p['z']:.2f} mm above the bed")
    for c in rep.checks:
        print(f"  [{ICON[c['status']]}] {c['title']}: {c['message']}")
    if rep.washers:
        print("washer options (largest figure first):")
        for w in rep.washers:
            print(f"  {w['key']:9s} {w['height_mm']:5.1f} mm  {'on steel' if w['pockets_on_steel'] else 'pockets OFF steel'}"
                  f"  {w['label']}")
    print(f"overall: {rep.status.upper()}  ({rep.timings['total']:.1f} s)")
    if a.out:
        out = export(s, params, a.out)
        print(json.dumps(out, indent=1))
    if a.slice:
        res = slice_variants(s, params, x10=a.x10)
        for k in ("off", "auto", "x10"):
            r = res.get(k)
            if not r:
                continue
            if r["ok"]:
                print(f"slice {k:4s}: {_fmt_time(r['seconds'])}, {r['grams']:.1f} g"
                      + (f", supports {_fmt_time(r['support_seconds'])}" if r["support_seconds"] else ""))
            else:
                print(f"slice {k:4s}: unavailable — {r['error']}")
        if res.get("supports_needed") is not None:
            print(f"supports needed (Bambu auto): {'YES' if res['supports_needed'] else 'no'}")
    return 0 if rep.status != "fail" else 2


def cmd_ball(a: argparse.Namespace) -> int:
    from .ball import Cavity, load_ball
    from .pipeline import CACHE_DIR, DEFAULT_BALL
    from .washer import get_washer

    b = load_ball(DEFAULT_BALL, CACHE_DIR)
    s = a.diameter / (2 * b.r_inner) if a.diameter else a.ball
    c = Cavity(b, s, get_washer(a.washer), a.clearance)
    print(f"inner R {b.r_inner:.3f} (x{s:.3f} = {c.r_inner:.2f}, inner Ø {2 * c.r_inner:.2f}), "
          f"ring span {b.ring_span:.2f}, aperture {b.aperture:.2f}")
    print(f"floor z {c.floor_z:.2f}, top z {c.top_z:.2f}, room {c.top_z - c.floor_z:.2f} mm")
    z, r = c.profile(12)
    for zi, ri in zip(z, r):
        print(f"  z {zi:7.2f}  r {ri:6.2f}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve

    serve(a.host, a.port, open_browser=a.open)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pokefit", description="Fit a figure inside a scaled Pokéball.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ball", type=float, default=0.6, help="ball print scale (0.6 = the 60%% ball)")
        p.add_argument("--diameter", type=float, help="ball inner diameter in mm (overrides --ball)")
        p.add_argument("--washer", default=DEFAULT_WASHER, help=f"{', '.join(CATALOGUE)} or 'OD,ID,T' mm")
        p.add_argument("--clearance", type=float, default=1.0, help="gap to the ball walls, mm")

    f = sub.add_parser("fit", help="analyse a figure, optionally export and slice")
    f.add_argument("input", type=Path)
    common(f)
    f.add_argument("--scale", default="max", help="'max', '85%%' or 0.85 (of the file's size)")
    f.add_argument("--object", help="object id when the 3MF holds several (default: largest)")
    f.add_argument("--rotate-x", type=float, default=0.0, help="degrees, any angle (applied X, Y, then Z)")
    f.add_argument("--rotate-y", type=float, default=0.0)
    f.add_argument("--rotate-z", type=float, default=0.0)
    f.add_argument("--magnets", type=int, default=2, choices=[0, 1, 2, 3], help="number of magnet pockets")
    f.add_argument("--no-magnets", action="store_true", help="same as --magnets 0")
    f.add_argument("--magnet-seat", choices=["washer", "raised"], default="washer",
                   help="washer: magnets rest on the washer (default); raised: pockets move up into parts "
                        "that float above the bed")
    f.add_argument("--colours", type=int, metavar="N", dest="texture_colours",
                   help="GLB/glTF: reduce the file's colours (texture, vertex colours) to N filaments "
                        "(default: as many as are clearly distinct, up to 8)")
    f.add_argument("--washers", action="store_true", help="compare every catalogue washer (slower)")
    f.add_argument("--out", type=Path, help="write <name>_pokefit_<ball%%>.3mf, shell STL and report here")
    f.add_argument("--slice", action="store_true", help="Bambu Studio time/filament, supports off vs auto")
    f.add_argument("--x10", action="store_true", help="also slice a plate of 10")
    f.set_defaults(fn=cmd_fit)

    b = sub.add_parser("ball", help="print the cavity profile")
    common(b)
    b.set_defaults(fn=cmd_ball)

    s = sub.add_parser("serve", help="web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", help="open the browser")
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    return int(a.fn(a))
