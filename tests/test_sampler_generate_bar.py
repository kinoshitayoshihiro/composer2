import random
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_generate_bar_history_and_deterministic(tmp_path: Path) -> None:
    for i in range(2):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path, order=2)
    history: list[groove_sampler_ngram.State] = []
    events1, history = groove_sampler_ngram.generate_bar(history, model, rng=random.Random(0))
    assert events1
    assert len(history) <= model["order"] - 1
    events_a, _ = groove_sampler_ngram.generate_bar(history.copy(), model, temperature=0, rng=random.Random(1))
    events_b, _ = groove_sampler_ngram.generate_bar(history.copy(), model, temperature=0, rng=random.Random(2))
    first_a = (int(round(events_a[0]["offset"] * 4)), events_a[0]["instrument"])
    first_b = (int(round(events_b[0]["offset"] * 4)), events_b[0]["instrument"])
    assert first_a == first_b


def test_gaussian_fallback(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_ngram.train(tmp_path, order=1)
    model["micro_offsets"] = {}
    events, _ = groove_sampler_ngram.generate_bar(None, model, humanize_micro=True, rng=random.Random(0))
    assert events
    step_ticks = groove_sampler_ngram.PPQ // 4
    found = False
    for ev in events:
        off_ticks = round(ev["offset"] * groove_sampler_ngram.PPQ)
        step = round(ev["offset"] * 4)
        micro = off_ticks - step * step_ticks
        if micro != 0:
            assert -45 <= micro <= 45
            found = True
    assert found
