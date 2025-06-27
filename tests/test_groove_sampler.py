from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_train_and_sample(tmp_path: Path) -> None:
    for i in range(4):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path)
    events = groove_sampler_ngram.sample(model, bars=1, seed=0)
    assert events
