import sys
from pathlib import Path
import modular_composer.cli as cli


def test_render_lufs_fallback(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.yml"
    spec.write_text('tempo_curve: [{"bpm": 120}]\ndrum_pattern: []\n')
    out = tmp_path / "out.mid"

    monkeypatch.setattr(cli, "has_fluidsynth", lambda: True)
    monkeypatch.setattr(cli.synth, "render_midi", lambda m, o, sf2_path=None: Path(o).write_bytes(b""))

    monkeypatch.setitem(sys.modules, "pyloudnorm", None)
    if "utilities.loudness_normalizer" in sys.modules:
        del sys.modules["utilities.loudness_normalizer"]

    cli.main([
        "render",
        str(spec),
        "-o",
        str(out),
        "--soundfont",
        "dummy.sf2",
        "--normalize-lufs",
        "-14",
    ])
    log = out.with_suffix(".wav")
    assert log.exists()  # file produced
    captured = capsys.readouterr()
    assert "pyloudnorm not installed" in captured.err
    
