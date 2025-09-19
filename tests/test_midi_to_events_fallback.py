import pytest

pytest.importorskip("pretty_midi")
import pretty_midi  # noqa: E402

from utilities.groove_sampler_v2 import midi_to_events  # noqa: E402


def _pm(notes, *, is_drum: bool) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=is_drum)
    for t, pitch in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=float(t),
                end=float(t) + 0.1,
            )
        )
    pm.instruments.append(inst)
    return pm


def test_midi_to_events_drum_priority():
    pm_drum = _pm([(0.0, 36)], is_drum=True)
    pm_pitched = _pm([(1.0, 60)], is_drum=False)
    pm = pretty_midi.PrettyMIDI()
    pm.instruments = pm_drum.instruments + pm_pitched.instruments

    events = midi_to_events(pm, 120.0)
    assert events, "drum events should be collected"
    assert all(pitch == 36 for _, pitch in events), (
        "when drums exist, only drum pitches should be returned"
    )


def test_midi_to_events_pitched_fallback():
    pm = _pm([(0.0, 60), (1.0, 62)], is_drum=False)
    events = midi_to_events(pm, 120.0)
    assert [pitch for _, pitch in events] == [60, 62], (
        "with no drums, pitched notes should be used in order"
    )


def test_midi_to_events_no_fallback():
    pm = _pm([(0.0, 60)], is_drum=False)
    assert midi_to_events(pm, 120.0, allow_pitched_fallback=False) == [], (
        "when fallback disabled and no drums exist, no events should be returned"
    )
