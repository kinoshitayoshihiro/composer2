import pathlib
import subprocess
import sys
import pytest

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

import mido


def _read_events(path: pathlib.Path) -> list[tuple[str, int, int | None, int | None, int | None]]:
    mid = mido.MidiFile(str(path))
    events: list[tuple[str, int, int | None, int | None, int | None]] = []
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
            sys.executable,
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

    if not golden.exists() or golden.stat().st_size == 0:
        pytest.skip("Golden MIDI missing")

    assert _read_events(out) == _read_events(golden)

