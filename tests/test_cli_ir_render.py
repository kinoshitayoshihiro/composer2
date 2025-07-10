import sys
from pathlib import Path
import importlib.util
import types
import numpy as np

if importlib.util.find_spec("soundfile") is None:
    sf = types.SimpleNamespace(
        write=lambda p, d, sr: Path(p).write_bytes(b""),
    )
else:
    import soundfile as sf
import importlib

import modular_composer.cli as cli


def _write_midi(path: Path) -> None:
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xFF\x2F\x00")


def test_cli_ir_render_modes(tmp_path, monkeypatch):
    midi = tmp_path / "a.mid"
    _write_midi(midi)
    ir = tmp_path / "ir.wav"
    sf.write(ir, [1.0], 44100)
    out = tmp_path / "out.wav"

    monkeypatch.setattr(cli, "has_fluidsynth", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)

    captured = []

    def fake_render_wav(m, i, o, **kw):
        captured.append(kw["quality"])
        Path(o).write_text("ok")

    monkeypatch.setattr(cli, "render_wav", fake_render_wav)

    for q in ["fast", "high", "ultra"]:
        monkeypatch.setattr(sys, "argv", [
            "modcompose",
            "ir-render",
            str(midi),
            str(ir),
            "-o",
            str(out),
            "--quality",
            q,
        ])
        cli.main()
        assert out.is_file()
        out.unlink()
    assert captured == ["fast", "high", "ultra"]


def test_cli_ir_render_dither_warning(tmp_path, monkeypatch, capsys):
    midi = tmp_path / "a.mid"
    _write_midi(midi)
    ir = tmp_path / "ir.wav"
    sf.write(ir, [1.0], 44100)
    out = tmp_path / "out.wav"

    monkeypatch.setattr(cli, "has_fluidsynth", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(cli, "render_wav", lambda *a, **k: Path(out).write_text("ok"))

    cli.main([
        "ir-render",
        str(midi),
        str(ir),
        "-o",
        str(out),
        "--no-normalize",
        "--dither",
    ])
    captured = capsys.readouterr()
    assert "Dither disabled because normalization is off" in captured.err
