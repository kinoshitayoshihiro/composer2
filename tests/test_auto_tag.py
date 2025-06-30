from pathlib import Path

import pretty_midi
import pytest

from data_ops.auto_tag import auto_tag


def _make_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(8):
        inst.notes.append(
            pretty_midi.Note(velocity=80, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


@pytest.mark.data_ops
def test_auto_tag(tmp_path: Path) -> None:
    midi = tmp_path / "a.mid"
    _make_midi(midi)
    meta = auto_tag(tmp_path)
    assert meta[midi.name][0]["intensity"] and meta[midi.name][0]["section"]
