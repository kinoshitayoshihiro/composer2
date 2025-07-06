import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utilities.audio_env import has_fluidsynth

sf = pytest.importorskip("soundfile")
pyln = pytest.importorskip("pyloudnorm")

if not has_fluidsynth():
    pytest.skip("fluidsynth missing", allow_module_level=True)

from utilities.convolver import convolve_ir, render_with_ir


def _sine(sr, dur=1.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return 0.1 * np.sin(2 * np.pi * 440 * t)


def test_render_mono_stereo(tmp_path):
    inp = tmp_path / "in.wav"
    sf.write(inp, np.column_stack([_sine(48000), _sine(48000)]), 48000)

    ir_mono = tmp_path / "ir_mono.wav"
    imp = np.zeros(44100)
    imp[0] = 1.0
    sf.write(ir_mono, imp, 44100)
    out1 = tmp_path / "out1.wav"
    render_with_ir(inp, ir_mono, out1, lufs_target=-14, progress=False)
    assert out1.is_file()

    ir_st = tmp_path / "ir_stereo.wav"
    imp2 = np.zeros((48000, 2))
    imp2[0, :] = 1.0
    sf.write(ir_st, imp2, 48000)
    out2 = tmp_path / "out2.wav"
    render_with_ir(inp, ir_st, out2, lufs_target=-14, progress=False)
    assert out2.is_file()


def test_lufs_target(tmp_path):
    sr = 44100
    inp = tmp_path / "in.wav"
    sf.write(inp, _sine(sr), sr)
    ir = tmp_path / "ir.wav"
    imp = np.zeros(sr)
    imp[0] = 1.0
    sf.write(ir, imp, sr)
    out = tmp_path / "out.wav"
    render_with_ir(inp, ir, out, lufs_target=-14, progress=False)
    data, sr = sf.read(out)
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(data)
    assert abs(loud + 14) < 0.5


def test_batch_render_dry_run(tmp_path):
    score = tmp_path / "test.mid"
    pretty = pytest.importorskip("pretty_midi")
    pm = pretty.PrettyMIDI()
    inst = pretty.Instrument(program=0)
    note = pretty.Note(velocity=64, pitch=60, start=0, end=1)
    inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(str(score))

    ir_dir = tmp_path / "irs"
    ir_dir.mkdir()
    ir = ir_dir / "crunch.wav"
    sf.write(ir, [1.0], 44100)

    script = Path("scripts/modcompose_batch_render.py").resolve()
    res = subprocess.run(
        [
            sys.executable,
            str(script),
            str(score),
            "--ir-folder",
            str(ir_dir),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert res.returncode == 0
    assert not any(tmp_path.glob("*.wav"))


@pytest.mark.audio
def test_convolve_length():
    audio = np.zeros(50, dtype=np.float32)
    ir = np.zeros(10, dtype=np.float32)
    ir[0] = 1.0
    out = convolve_ir(audio, ir)
    assert len(out) == len(audio)
