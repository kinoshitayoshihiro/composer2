from pathlib import Path

import pretty_midi

from utilities.midi_export import export_song


def stub_generator(bars: int, tempo: float) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.5, pitch=36, velocity=100))
    pm.instruments.append(inst)
    return pm


def test_export_song(tmp_path: Path) -> None:
    tempo_map = [(0, 120), (8, 90), (16, 140)]
    out = tmp_path / "song.mid"
    pm = export_song(
        32, tempo_map=tempo_map, generators={"drum": stub_generator}, out_path=out
    )
    assert out.exists()
    assert len(pm.instruments) == 1
    tempi, _times = pm.get_tempo_changes()
    assert list(tempi) == [120, 90, 140]
