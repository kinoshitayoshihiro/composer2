import random
import pretty_midi
from pathlib import Path
from utilities import groove_sampler


def _write_loop(path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    beat_sec = 0.5
    for bar in range(4):
        for beat in range(4):
            start = (bar * 4 + beat) * beat_sec
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.1))
    pm.instruments.append(inst)
    out = path / "loop.mid"
    pm.write(str(out))
    return out


def test_generate_bar_from_model(tmp_path: Path):
    _write_loop(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    events = groove_sampler.generate_bar([], model, random.Random(0))
    assert any(e.get('instrument') for e in events)
