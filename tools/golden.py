from __future__ import annotations

import os
from pathlib import Path

import base64
import tempfile

from utilities.golden import compare_midi as _compare_midi


def _decode_temp(path: Path) -> Path:
    data = base64.b64decode(path.read_text())
    tmp = Path(tempfile.mkstemp(suffix=".mid")[1])
    tmp.write_bytes(data)
    return tmp


def compare_midi(golden: str | Path, generated: str | Path) -> str:
    """Return diff string if MIDI files differ, else empty string."""
    golden_tmp = _decode_temp(Path(golden))
    same = _compare_midi(golden_tmp, Path(generated))
    golden_tmp.unlink()
    return "" if same else "MIDI files differ"


def update_golden(src: str | Path, dst: str | Path) -> None:
    """Encode ``src`` MIDI to base64 and write to ``dst``."""
    data = Path(src).read_bytes()
    Path(dst).write_text(base64.b64encode(data).decode("ascii"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("generated")
    ns = ap.parse_args()

    if os.getenv("UPDATE_GOLDENS") == "1":
        update_golden(ns.generated, ns.golden)
    else:
        diff = compare_midi(ns.golden, ns.generated)
        if diff:
            print(diff)
