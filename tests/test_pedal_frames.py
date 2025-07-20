import pytest

pytest.importorskip("pandas")
pretty_midi = pytest.importorskip("pretty_midi")

from utilities.pedal_frames import HOP_LENGTH, SR, extract_from_midi


def create_midi() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.5, pitch=60, velocity=100))
    inst.notes.append(pretty_midi.Note(start=1.0, end=1.5, pitch=62, velocity=100))
    inst.control_changes.append(
        pretty_midi.ControlChange(number=64, value=127, time=0.0)
    )
    inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=0.8))
    inst.control_changes.append(
        pretty_midi.ControlChange(number=64, value=127, time=1.2)
    )
    pm.instruments.append(inst)
    return pm


def test_extract_from_midi() -> None:
    pm = create_midi()
    df = extract_from_midi(pm)
    assert set(df.columns).issuperset(
        {"track_id", "frame_id", "rel_release", "pedal_state", "chroma_0"}
    )
    hop = HOP_LENGTH / SR
    idx_down = int(0.1 / hop)
    idx_up = int(0.9 / hop)
    states = (
        df[df.track_id == 0]
        .set_index("frame_id")
        .loc[[idx_down, idx_up], "pedal_state"]
        .tolist()
    )
    assert states[0] == 1 and states[1] == 0
