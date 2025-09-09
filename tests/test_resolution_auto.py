from pathlib import Path

import pretty_midi

from utilities import groove_sampler


def _loop_3_4(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(12):
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=36,
                start=i * 0.25,
                end=i * 0.25 + 0.05,
            )
        )
    pm.instruments.append(inst)
    pm.write(str(path / "three.mid"))


def test_resolution_auto(tmp_path: Path):
    _loop_3_4(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, resolution="auto")
    assert model["resolution"] in {3, 6, 12, 24}


def test_resolution_triplet(tmp_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    sec_per_beat = 0.5
    for i in range(12):
        start = (i / 3) * sec_per_beat
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    path = tmp_path / "tri.mid"
    pm.write(str(path))
    model = groove_sampler.load_grooves(tmp_path, resolution="auto")
    assert model["resolution"] == 12
