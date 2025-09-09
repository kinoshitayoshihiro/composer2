import subprocess
import sys
from pathlib import Path


def test_cli_fx_render(tmp_path: Path) -> None:
    midi = tmp_path / "demo.mid"
    midi.write_bytes(
        b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xFF\x2F\x00"
    )
    out = tmp_path / "out.wav"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "modcompose"),
        "fx",
        "render",
        str(midi),
        "-o",
        str(out),
        "--preset",
        "clean",
    ]
    subprocess.check_call(cmd)
    assert out.exists()
