import warnings
import random
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path, pitch: int = 36) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_generate_bar_aux_fallback(tmp_path: Path) -> None:
    _make_loop(tmp_path / "verse.mid")
    aux_map = {"verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"}}
    model = gs.train(tmp_path, aux_map=aux_map, order=1)
    with warnings.catch_warnings(record=True) as rec:
        ev = gs.generate_bar(None, model=model, cond={"section": "bridge"})
    assert ev
    assert any("unknown aux" in str(w.message).lower() for w in rec)


def test_topk_temp_zero_equivalent(tmp_path: Path) -> None:
    for i in range(2):
        _make_loop(tmp_path / f"{i}.mid", pitch=36 + i)
    model = gs.train(tmp_path, order=2)
    hist_a: list[gs.State] = []
    ev_a = gs.generate_bar(hist_a, model=model, temperature=0.0, top_k=1)
    hist_b: list[gs.State] = []
    ev_b = gs.generate_bar(hist_b, model=model, temperature=0.0)
    a_first = (round(ev_a[0]["offset"] * 4), ev_a[0]["instrument"])
    b_first = (round(ev_b[0]["offset"] * 4), ev_b[0]["instrument"])
    assert a_first == b_first
