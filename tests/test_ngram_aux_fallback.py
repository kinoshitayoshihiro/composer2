import warnings
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path, pitch: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_aux_fallback(tmp_path: Path) -> None:
    _make_loop(tmp_path / "verse.mid", 36)
    _make_loop(tmp_path / "chorus.mid", 38)
    aux_map = {
        "verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"},
        "chorus.mid": {"section": "chorus", "heat_bin": 0, "intensity": "mid"},
    }
    model = groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)
    with warnings.catch_warnings(record=True) as rec:
        groove_sampler_ngram.sample(model, bars=1, cond={"section": "bridge"})
    assert any("unknown aux" in str(w.message).lower() for w in rec)


def test_partial_condition(tmp_path: Path) -> None:
    _make_loop(tmp_path / "verse.mid", 36)
    aux_map = {"verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"}}
    model = groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)
    ev = groove_sampler_ngram.sample(model, bars=1, cond={"heat_bin": 0})
    assert ev
