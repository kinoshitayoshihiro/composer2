import random
import weakref
from pathlib import Path
import pretty_midi

from utilities import groove_sampler_ngram as gs


def _loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


class CountingCache(weakref.WeakValueDictionary):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def get(self, key, default=None):  # type: ignore[override]
        self.calls += 1
        return super().get(key, default)


def test_bar_cache_reduces_lin_prob(tmp_path: Path, monkeypatch) -> None:
    _loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=2)
    orig = gs._sample_next

    cd = CountingCache()
    monkeypatch.setattr(gs, "_lin_prob", cd, raising=False)

    def no_cache(history, model_arg, rng, **kwargs):
        kwargs["cache"] = None
        return orig(history, model_arg, rng, **kwargs)

    monkeypatch.setattr(gs, "_sample_next", no_cache)
    gs.generate_bar([], model=model)
    no_cache_calls = cd.calls

    cd2 = CountingCache()
    monkeypatch.setattr(gs, "_lin_prob", cd2, raising=False)
    monkeypatch.setattr(gs, "_sample_next", orig)
    gs.generate_bar([], model=model)
    with_cache_calls = cd2.calls

    assert with_cache_calls < no_cache_calls


def test_sample_consistency_no_cache(tmp_path: Path) -> None:
    _loop(tmp_path / "b.mid")
    model = gs.train(tmp_path, order=1)
    ev_cache = gs.sample(model, bars=2, seed=0, use_bar_cache=True)
    ev_no = gs.sample(model, bars=2, seed=0, use_bar_cache=False)
    key = lambda e: (e["offset"], e["instrument"])
    assert sorted(ev_cache, key=key) == sorted(ev_no, key=key)
