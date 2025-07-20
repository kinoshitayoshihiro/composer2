from pathlib import Path

import mido
import pretty_midi

from utilities.articulation_csv import extract_from_midi
from utilities.time_utils import seconds_to_qlen


def test_onset_tempo_change(tmp_path: Path) -> None:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=mid.ticks_per_beat * 4))
    track.append(mido.Message("note_on", note=60, velocity=100, time=mid.ticks_per_beat * 2))
    track.append(mido.Message("note_off", note=60, velocity=0, time=mid.ticks_per_beat // 2))
    midi = tmp_path / "tempo.mid"
    mid.save(midi)
    pm = pretty_midi.PrettyMIDI(str(midi))

    df = extract_from_midi(midi)
    onset = float(df.loc[0, "onset"])
    assert onset == 6.0
    assert seconds_to_qlen(pm, 4.0) == 6.0
