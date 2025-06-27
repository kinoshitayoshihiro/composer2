from pathlib import Path

import pretty_midi
import pytest

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_validate_aux_map_missing(tmp_path: Path) -> None:
    loop = tmp_path / "a.mid"
    _make_loop(loop)
    aux_map = {"b.mid": {"section": "verse", "heat_bin": 0, "intensity": "mid"}}
    with pytest.raises(ValueError):
        groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)


def test_validate_aux_map_invalid(tmp_path: Path) -> None:
    loop = tmp_path / "x.mid"
    _make_loop(loop)
    aux_map = {"x.mid": {"section": "!bad", "heat_bin": 20, "intensity": "loud"}}
    with pytest.raises(ValueError):
        groove_sampler_ngram.train(tmp_path, aux_map=aux_map, order=1)

