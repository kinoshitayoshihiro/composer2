import pathlib
import subprocess
from typing import List, Tuple

import mido
import pytest


def _read_events(path: pathlib.Path) -> List[Tuple[str, int, int | None, int | None, int | None]]:
    mid = mido.MidiFile(str(path))
    events: List[Tuple[str, int, int | None, int | None, int | None]] = []
    for track in mid.tracks:
        for msg in track:
            note = getattr(msg, "note", None)
            vel = getattr(msg, "velocity", None)
            ch = getattr(msg, "channel", None)
            events.append((msg.type, msg.time, note, vel, ch))
    return events


def test_golden_demo(tmp_path: pathlib.Path, request: pytest.FixtureRequest) -> None:
    out = tmp_path / "demo.mid"
    subprocess.run(
        [
            "python",
            "-m",
            "modular_composer.cli",
            "demo",
            "-o",
            str(out),
            "--tempo-curve",
            str(pathlib.Path("data/tempo_curve.json")),
        ],
        check=True,
    )

    golden = pathlib.Path(__file__).resolve().parent / "golden_midi" / "expected_demo.mid"
    if request.config.getoption("--update-golden"):
        golden.write_bytes(out.read_bytes())
        pytest.skip("golden regenerated")

    if not golden.exists():
        pytest.skip("Golden MIDI missing")

    assert _read_events(out) == _read_events(golden)

