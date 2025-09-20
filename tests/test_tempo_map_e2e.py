from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from utilities import beat_to_seconds as _beat_to_seconds  # noqa: F401
from utilities.midi_export import export_song


def stub_gen(
    section: dict, tempo: float, *, vocal_metrics=None
) -> pretty_midi.PrettyMIDI:
    bars = section.get("bars", 4)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)
    qlen = 60.0 / tempo
    for b in range(bars):
        start = b * 4 * qlen
        for pitch in (60, 64, 67):
            inst.notes.append(
                pretty_midi.Note(
                    velocity=90, pitch=pitch, start=start, end=start + qlen
                )
            )
    # ensure section covers full bar length
    end_start = bars * 4 * qlen - qlen
    for pitch in (60, 64, 67):
        inst.notes.append(
            pretty_midi.Note(
                velocity=90, pitch=pitch, start=end_start, end=end_start + qlen
            )
        )
    pm.instruments.append(inst)
    return pm


def test_tempo_map_e2e(tmp_path: Path) -> None:
    sec1 = {"bars": 4, "tempo_map": [(0, 120)]}
    sec2 = {"bars": 4, "tempo_map": [(0, 90)]}
    sec3 = {"bars": 4, "tempo_map": [(0, 140)]}
    out = tmp_path / "out.mid"
    pm = export_song(
        0,
        generators={"piano": stub_gen},
        fixed_tempo=120,
        sections=[sec1, sec2, sec3],
        out_path=out,
    )
    assert out.exists()
    times, bpms = pm.get_tempo_changes()
    times_list = times.tolist() if hasattr(times, "tolist") else list(times)
    assert times_list[0] == pytest.approx(0.0)
    assert all(t >= 0 for t in times_list)
    assert [round(b) for b in bpms] == [120, 90, 140]
    # Reload file so tempo map affects timing
    pm2 = pretty_midi.PrettyMIDI(str(out))
    expected_length = pm2.get_end_time()
    assert pm2.get_end_time() == pytest.approx(expected_length, rel=0.05)
