"""Extract each mesh object from a (Bambu Studio) 3MF into its own STL.

Bambu stores each object in 3D/Objects/object_N.model (3MF Production
extension). Meshes are written in their own local coordinates (the component
transform is applied; the plate placement transform is not), in millimetres.

Usage: python scripts/extract_3mf.py <in.3mf> <out_dir>
"""
import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
      "p": "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"}
P_PATH = "{%s}path" % NS["p"]


def parse_transform(s: str | None) -> np.ndarray:
    """3MF 3x4 row-major affine ("m00 m01 m02 m10 ... m32") -> 4x4 matrix."""
    m = np.eye(4)
    if s:
        v = np.array([float(x) for x in s.split()]).reshape(4, 3)
        m[:3, :3] = v[:3].T
        m[:3, 3] = v[3]
    return m


def mesh_from_model(xml: bytes, object_id: str) -> trimesh.Trimesh:
    root = ET.fromstring(xml)
    obj = root.find(f".//m:object[@id='{object_id}']", NS)
    verts = np.array([[float(v.get(a)) for a in "xyz"] for v in obj.iterfind(".//m:vertex", NS)])
    tris = np.array([[int(t.get(a)) for a in ("v1", "v2", "v3")] for t in obj.iterfind(".//m:triangle", NS)])
    return trimesh.Trimesh(verts, tris, process=True)


def object_names(z: zipfile.ZipFile) -> dict[str, str]:
    """Map top-level object id -> name from Bambu's model_settings.config."""
    try:
        cfg = z.read("Metadata/model_settings.config").decode()
    except KeyError:
        return {}
    return dict(re.findall(r'<object id="(\d+)">\s*<metadata key="name" value="([^"]+)"', cfg))


def extract(src: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        root = ET.fromstring(z.read("3D/3dmodel.model"))
        names = object_names(z)
        for obj in root.iterfind(".//m:resources/m:object", NS):
            oid = obj.get("id")
            parts = []
            for comp in obj.iterfind(".//m:component", NS):
                sub = mesh_from_model(z.read(comp.get(P_PATH).lstrip("/")), comp.get("objectid"))
                sub.apply_transform(parse_transform(comp.get("transform")))
                parts.append(sub)
            if not parts:  # plain 3MF: mesh inline in the root model
                parts.append(mesh_from_model(z.read("3D/3dmodel.model"), oid))
            mesh = trimesh.util.concatenate(parts)
            stem = Path(names.get(oid, f"object_{oid}")).stem
            dst = out / f"{stem}_id{oid}.stl"
            mesh.export(dst)
            print(f"{dst.name}: faces={len(mesh.faces)} extents={mesh.extents.round(2)} "
                  f"watertight={mesh.is_watertight}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    extract(a.src, a.out_dir)
