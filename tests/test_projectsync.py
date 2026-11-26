"""The project's filament list against the printer's spools (projectsync)."""
import json
import zipfile
from pathlib import Path

import pytest

from pokefit.printer import AmsSlot
from pokefit.projectsync import adopt, compare, updated_project


def _ref(n: int = 3) -> dict:
    types = ["PETG", "PLA", "PLA", "PETG"][:n]
    return {"filament_settings_id": [f"Generic {t} @BBL X2D 0.4 nozzle" for t in types],
            "filament_type": types, "filament_colour": ["#F4EE2A", "#FFFFFF", "#161616", "#FF0000"][:n],
            "filament_ids": ["GFG99" if t == "PETG" else "GFL99" for t in types],
            "nozzle_temperature": [str(240 if t == "PETG" else 220) for t in types],
            "curr_bed_type": "Supertack Plate"}


SPOOLS = [AmsSlot(4, "B1", "#F4EE2A", "PETG", "", filament_id="GFG99"),
          AmsSlot(0, "A1", "#FFFFFF", "PLA", "", filament_id="GFL99"),
          AmsSlot(255, "Ext", "#898989", "PLA", "", filament_id="GFL99")]


def test_compare_slot_by_slot_against_the_trays_in_order():
    s = compare(_ref(), SPOOLS)
    assert s is not None
    # Trays in order A1, B1 (the external spool is no project slot): slot 1 PETG yellow vs A1 white PLA.
    assert (s["printer"], s["project"], s["same"], s["differs"]) == (2, 3, 0, True)
    assert [r["spool"]["label"] if r["spool"] else None for r in s["rows"]] == ["A1", "B1", None]
    assert s["plate"] == "Supertack Plate"
    same = compare({**_ref(2), "filament_type": ["PLA", "PETG"], "filament_colour": ["#FEFEFE", "#F4EE2A"]}, SPOOLS)
    assert same and not same["differs"] and same["same"] == 2
    assert compare(None, SPOOLS) is None and compare(_ref(), []) is None


def test_adopt_takes_the_trays_and_keeps_the_slots_past_them():
    out = adopt(_ref(3), SPOOLS, None, "Textured PEI Plate")
    assert out["filament_type"] == ["PLA", "PETG", "PLA"]            # A1, B1, then the project's own slot 3
    assert out["filament_colour"] == ["#FFFFFF", "#F4EE2A", "#161616"]
    assert out["nozzle_temperature"] == ["220", "240", "220"]        # per-filament arrays follow their slot
    assert out["curr_bed_type"] == "Textured PEI Plate"
    assert compare(out, SPOOLS)["differs"] is False
    with pytest.raises(ValueError):
        adopt(_ref(), SPOOLS, None, "Glass Plate")


def test_updated_project_copies_the_users_file_with_the_new_settings(tmp_path: Path):
    src = tmp_path / "shells.3mf"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("Metadata/project_settings.config", json.dumps(_ref()))
        z.writestr("3D/3dmodel.model", "<model/>")
    new = adopt(_ref(), SPOOLS)
    name, data = updated_project(src, new, "shells.3mf")
    out = tmp_path / name
    out.write_bytes(data)
    assert name == "shells_printer.3mf"
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("Metadata/project_settings.config")) == new
        assert z.read("3D/3dmodel.model") == b"<model/>"
    name, data = updated_project(None, new, "x2d_petg_printed_ball_shells.json")
    assert name == "x2d_petg_printed_ball_shells_printer.json" and json.loads(data)["from"] == "project"
