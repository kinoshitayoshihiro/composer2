import random

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from tests._stubs import pretty_midi  # type: ignore

from tools.ujam_bridge import utils


def _single_note(start_tick: int) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(program=0)
    start = start_tick / (pm.resolution * 2)
    note = pretty_midi.Note(velocity=100, pitch=60, start=start, end=start + 0.5)
    inst.notes.append(note)
    pm.instruments.append(inst)
    return pm


def test_quantize_and_swing() -> None:
    pm = _single_note(90)
    utils.quantize(pm, grid=120, swing=0.0)
    tick = pm.time_to_tick(pm.instruments[0].notes[0].start)
    assert tick == 120

    pm = _single_note(120)
    utils.quantize(pm, grid=120, swing=0.5)
    tick = pm.time_to_tick(pm.instruments[0].notes[0].start)
    assert tick == 150


def test_chordify_and_groove() -> None:
    chord = utils.chordify([36, 40, 43], (40, 64))
    assert chord == [48, 55]

    pm = _single_note(240)
    profile = {"0": 0.0, "0.5": 0.1, "meta_grid_step": 0.5}
    utils.apply_groove_profile(pm, profile)
    tick = pm.time_to_tick(pm.instruments[0].notes[0].start)
    assert tick == 288

    pm = _single_note(240)
    utils.apply_groove_profile(pm, profile, max_ms=10)
    tick = pm.time_to_tick(pm.instruments[0].notes[0].start)
    assert round(tick) == 250


def test_humanize() -> None:
    pm = _single_note(0)
    utils.humanize(pm, 0.0, rng=random.Random(0))
    assert pm.time_to_tick(pm.instruments[0].notes[0].start) == 0

    pm = _single_note(0)
    utils.humanize(pm, 1.0, rng=random.Random(0))
    assert pm.time_to_tick(pm.instruments[0].notes[0].start) != 0
