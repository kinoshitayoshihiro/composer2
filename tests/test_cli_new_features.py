import json
from pathlib import Path

import pretty_midi
from click.testing import CliRunner

import modular_composer.cli as cli
from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_cli_export_midi(tmp_path: Path) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    _make_loop(loops / "a.mid")
    model = gs.train(loops, order=1)
    gs.save(model, tmp_path / "model.pkl")
    out = tmp_path / "out.mid"
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["export-midi", str(tmp_path / "model.pkl"), str(out)])
    assert res.exit_code == 0
    assert out.exists()


def test_cli_render_audio(tmp_path: Path, monkeypatch) -> None:
    midi = tmp_path / "a.mid"
    midi.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xFF\x2F\x00")
    out = tmp_path / "out.wav"
    runner = CliRunner()
    monkeypatch.setattr(cli, "has_fluidsynth", lambda: True)
    monkeypatch.setattr(cli.synth, "export_audio", lambda *a, **k: out.write_bytes(b""))
    res = runner.invoke(
        cli.cli,
        ["render-audio", str(midi), "-o", str(out), "--use-default-sf2"],
    )
    assert res.exit_code == 0
    assert out.exists()


def test_cli_evaluate(tmp_path: Path) -> None:
    ref = tmp_path / "ref.mid"
    gen = tmp_path / "gen.mid"
    _make_loop(ref)
    _make_loop(gen)
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["evaluate", str(gen), "--ref", str(ref)])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert "swing_score" in data
    assert "blec" in data


def test_cli_visualize(tmp_path: Path) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    _make_loop(loops / "a.mid")
    model = gs.train(loops, order=1)
    gs.save(model, tmp_path / "model.pkl")
    out = tmp_path / "plot.png"
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["visualize", str(tmp_path / "model.pkl"), "--out", str(out)])
    assert res.exit_code == 0
    assert out.exists()
