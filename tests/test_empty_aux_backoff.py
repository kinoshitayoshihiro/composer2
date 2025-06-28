from pathlib import Path
import pretty_midi

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_unknown_aux_fallback(tmp_path: Path) -> None:
    _make_loop(tmp_path / "verse.mid")
    aux_map = {"verse.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"}}
    model = gs.train(tmp_path, aux_map=aux_map, order=1)
    events = gs.sample(model, bars=1, cond={"section": "bridge"})
    assert events

