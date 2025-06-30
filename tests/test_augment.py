from pathlib import Path

import pretty_midi

from data_ops.augment import swing


def _make_midi(path: Path) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=False)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.25, end=0.5))
    pm.instruments.append(inst)
    pm.write(str(path))
    return pm


def test_swing(tmp_path: Path) -> None:
    midi = tmp_path / "a.mid"
    pm = _make_midi(midi)
    swung = swing(pm, 50)
    beat = 60.0 / 120
    tick = beat / 480
    expected = 0.25 + 0.125
    assert abs(swung.instruments[0].notes[1].start - expected) < 2 * tick
