import pytest

pytest.importorskip("scipy.stats")
from scipy.stats import ks_2samp
from pathlib import Path

import numpy as np
import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path, pitch: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start,
                end=start + 0.05,
            )
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_aux_conditioning(tmp_path: Path) -> None:
    verse = tmp_path / "verse.mid"
    chorus = tmp_path / "chorus.mid"
    _make_loop(verse, 36)
    _make_loop(chorus, 38)

    aux_map = {
        "verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"},
        "chorus.mid": {"section": "chorus", "heat_bin": 1, "intensity": "high"},
    }

    model = groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)

    def _sample(section: str) -> list[str]:
        names: list[str] = []
        for seed in range(20):
            events = groove_sampler_ngram.sample(
                model,
                bars=32,
                seed=seed,
                cond={"section": section},
            )
            names.extend(ev["instrument"] for ev in events)
        return names

    names_chorus = _sample("chorus")
    names_verse = _sample("verse")

    inst_map = {n: i for i, n in enumerate(sorted(set(names_chorus + names_verse)))}
    arr_chorus = [inst_map[n] for n in names_chorus]
    arr_verse = [inst_map[n] for n in names_verse]

    _stat, p = ks_2samp(arr_chorus, arr_verse)

    assert p < 0.01

