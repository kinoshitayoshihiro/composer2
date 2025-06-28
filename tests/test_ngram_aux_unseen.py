import pretty_midi
from pathlib import Path

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_unseen_aux_fallback(tmp_path: Path) -> None:
    loop = tmp_path / "verse.mid"
    _make_loop(loop)
    aux_map = {"verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"}}
    model = groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)
    events = groove_sampler_ngram.sample(model, bars=1, cond={"section": "bridge"})
    assert events
