from collections import Counter
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

    def _sample(section: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for seed in range(20):
            events = groove_sampler_ngram.sample(
                model,
                bars=32,
                seed=seed,
                cond={"section": section},
            )
            counts.update(ev["instrument"] for ev in events)
        return counts

    c_chorus = _sample("chorus")
    c_verse = _sample("verse")

    instruments = sorted(set(c_chorus) | set(c_verse))
    obs = np.array(
        [
            [c_chorus.get(i, 0) for i in instruments],
            [c_verse.get(i, 0) for i in instruments],
        ]
    )

    row_tot = obs.sum(axis=1, keepdims=True)
    col_tot = obs.sum(axis=0, keepdims=True)
    expected = row_tot * col_tot / obs.sum()
    chi2 = float(((obs - expected) ** 2 / expected).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    critical = {1: 3.841, 2: 5.991, 3: 7.815}.get(dof, 0.0)
    p = 0.0 if chi2 > critical else 1.0

    assert p < 0.05

