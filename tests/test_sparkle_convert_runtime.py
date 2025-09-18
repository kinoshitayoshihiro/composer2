import yaml
from pathlib import Path

try:  # pragma: no cover - use stub if pretty_midi missing
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

from ujam import sparkle_convert as sc


def _pm(length: float = 4.0, bar_dur: float = 2.0):
    class Dummy:
        def __init__(self, length: float, bar_dur: float) -> None:
            self._length = length
            self._bar = bar_dur
            inst = pretty_midi.Instrument(0)
            inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=length))
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            step = 0.5
            n = int(self._length / step) + 1
            return [i * step for i in range(n)]

        def get_downbeats(self):
            n = int(self._length / self._bar) + 1
            return [i * self._bar for i in range(n)]

        def get_end_time(self):
            return self._length

        def get_tempo_changes(self):
            return [0.0], [120.0]

        def write(self, path: str) -> None:  # pragma: no cover
            Path(path).write_bytes(b"")

    return Dummy(length, bar_dur)


def _phrase_inst(pm: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    return next(i for i in pm.instruments if i.name == sc.PHRASE_INST_NAME)


def test_12_8_swing_pulses_monotonic() -> None:
    pm = _pm(6.0, 3.0)
    chords = [sc.ChordSpan(0, 3, 0, "maj"), sc.ChordSpan(3, 6, 0, "maj")]
    mapping = yaml.safe_load(sc.generate_mapping_template(False))
    mapping["cycle_phrase_notes"] = [36]
    for sw in (0.0, 1 / 12, 2 / 12, 4 / 12):
        stats: dict = {"_legacy_bar_pulses_grid": True}
        out = sc.build_sparkle_midi(
            pm,
            chords,
            mapping,
            0.5,
            "bar",
            0.0,
            0,
            "flat",
            120.0,
            sw,
            0.5,
            stats=stats,
        )
        counts = [len(stats["bar_pulses"].get(i, [])) for i in range(2)]
        assert counts == [24, 24]
        starts = [n.start for n in _phrase_inst(out).notes]
        assert all(b >= a for a, b in zip(starts, starts[1:]))


def test_phrase_hold_modes_no_negative_duration() -> None:
    pm = _pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = yaml.safe_load(sc.generate_mapping_template(False))
    mapping["cycle_phrase_notes"] = [36]
    for hold in ("bar", "chord"):
        mapping["phrase_hold"] = hold
        for mode in ("first", "mean", "max"):
            mapping["held_vel_mode"] = mode
            out = sc.build_sparkle_midi(
                pm,
                chords,
                mapping,
                0.5,
                "bar",
                5.0,
                4,
                "flat",
                120.0,
                0.0,
                0.5,
            )
            notes = _phrase_inst(out).notes
            assert notes
            for n in notes:
                assert n.end > n.start and n.start >= 0


def test_cross_bar_merge() -> None:
    pm = _pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = yaml.safe_load(sc.generate_mapping_template(False))
    mapping["cycle_phrase_notes"] = [36]
    mapping["phrase_length_beats"] = 0.5
    mapping["phrase_merge_gap"] = 0.05
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120.0,
        0.0,
        0.5,
    )
    assert _phrase_inst(out).notes

