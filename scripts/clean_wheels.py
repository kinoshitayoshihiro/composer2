# scripts/clean_wheels.py
import re
from pathlib import Path

ROOT = Path("wheelhouse")  # ← プロジェクト直下の wheelhouse
KEEP_RE = re.compile(r"cp312.*manylinux_2_17_x86_64")

for whl in ROOT.glob("*.whl"):
    if not KEEP_RE.search(whl.name):
        print("Remove", whl.name)
        whl.unlink()
