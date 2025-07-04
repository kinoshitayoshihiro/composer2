import numpy as np
import pretty_midi
import soundfile as sf
from pathlib import Path

from utilities.ir_renderer import render_ir


def test_ir_render(tmp_path: Path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    midi.instruments.append(inst)
    midi_path = tmp_path / "in.mid"
    midi.write(str(midi_path))
    ir_path = tmp_path / "ir.wav"
    sf.write(ir_path, np.zeros(44100, dtype=np.float32), 44100)
    out = tmp_path / "out.wav"
    render_ir(str(midi_path), str(out), ir_path=str(ir_path))
    data, sr = sf.read(out)
    assert len(data) == 44100
