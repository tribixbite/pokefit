# Pokeball (SketchUp 2017 → STL)

## Mesh repair
SketchUp exports here are at **16× working scale** and often non-manifold.
- Check: `python scripts/diag_stl.py <file.stl>` (needs `pip install trimesh pymeshfix manifold3d`)
- Fix: `python scripts/repair_shell.py <in.stl> <out.stl> --scale 16 --flat-z <bed plane z in source units>`
  (`--no-trim` if the part has no flat bed spot). Output is mm, min corner at origin.
- `redsmaller.stl` is repaired (watertight, 46.51 × 45.02 × 21.06 mm, 3350.9 mm³).
  The unrepaired export is in commit `6fe8dd9`.

## Open
- **CleanUp³ needs TT_Lib² ≥ 2.11** — CleanUp³ 3.4.3 files are in
  `%APPDATA%\SketchUp\SketchUp 2017\SketchUp\Plugins\tt_cleanup` and match the rbz, but
  TT_Lib² is not installed (download is login-gated: Extension Warehouse
  `extensions.sketchup.com/extension/c03a2b93-3365-4ef1-95f4-f35158757622/tt-lib`, or
  sketchucation.com/plugin/726-tt_lib). Install via Window → Extension Manager → Install
  Extension, then restart SketchUp.
- Other untracked exports are also non-manifold (from `diag_stl.py`): `redsphere.stl`
  (43 open / 15 non-manifold edges), `redflat.stl`, `red2.stl`. Not repaired.
- `redsmaller.stl` bed contact is a 35 mm² disc (~6.7 mm) — consider a brim.
