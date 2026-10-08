"""Remove the Extension-Warehouse-only Pro gate from Eneroth Solid Tools menu.rb.

The tools implement booleans in Ruby (intersect_with), not via the Pro-only
Group#union/subtract API, so they run on SketchUp Make. MIT licensed.
Byte-level edit so the file's CRLF endings are preserved.
"""
import sys
from pathlib import Path

OLD = b"    if Sketchup.is_pro?\r\n"
NEW = (b"    # Patched for SketchUp Make: Pro gate removed (tools don't use Pro-only API).\r\n"
       b"    if true\r\n")

path = Path(sys.argv[1])
data = path.read_bytes()
if NEW in data:
    print("already patched")
    sys.exit(0)
if data.count(OLD) != 1:
    sys.exit(f"expected exactly one gate line, found {data.count(OLD)}")
path.write_bytes(data.replace(OLD, NEW))
print("patched")
