import json
from pathlib import Path

import pretty_midi
import pytest

from generator.drum_generator import DrumGenerator
from utilities import vocal_sync


def _make_pm() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.5, end=1.0))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.75, end=0.76))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=1.5, end=2.0))
    pm.instruments.append(inst)
    return pm


def test_extract_onsets():
    pm = _make_pm()
    on = vocal_sync.extract_onsets(pm)
    assert len(on) == 4


def test_extract_long_rests():
    pm = _make_pm()
    onsets = vocal_sync.extract_onsets(pm)
    rests = vocal_sync.extract_long_rests(onsets, min_rest=0.4)
    assert (onsets[2], pytest.approx(1.5)) in rests


def test_load_consonant_peaks(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"peaks": [0.5, 1.0]}))
    beats = vocal_sync.load_consonant_peaks(p)
    assert beats == [1.0, 2.0]


def test_tempo_map_conversion(tmp_path: Path):
    pm = _make_pm()
    tempo_map = pretty_midi.PrettyMIDI(initial_tempo=60)

    # extract_onsets with external tempo map
    beats = vocal_sync.extract_onsets(pm, tempo_map=tempo_map)
    assert beats[-1] == pytest.approx(1.5)

    p = tmp_path / "p.json"
    p.write_text(json.dumps({"peaks": [1.0]}))
    peak_beats = vocal_sync.load_consonant_peaks(p, tempo_map=tempo_map)
    assert peak_beats == [1.0]

    onsets_sec = [n.start for n in pm.instruments[0].notes]
    rests = vocal_sync.extract_long_rests(onsets_sec, min_rest=0.4, tempo_map=tempo_map)
    assert (pytest.approx(0.75), pytest.approx(0.75)) in rests


def test_quantize_times_dedup():
    times = [0.1, 0.12, 0.35]
    q = vocal_sync.quantize_times(times, 0.25, dedup=True)
    assert q == [0.0, 0.25]


def test_drumgen_integration(monkeypatch, tmp_path: Path):
    midi_path = tmp_path / "v.mid"
    _make_pm().write(str(midi_path))
    peaks_json = tmp_path / "c.json"
    peaks_json.write_text(json.dumps({"peaks": [0.1]}))
    called = {"on": False, "rest": False, "peak": False}

    def fake_onsets(_pm):
        called["on"] = True
        return [0.0]

    def fake_rests(onsets, *, min_rest=0.5):
        called["rest"] = True
        return [(0.0, 1.0)]

    def fake_peaks(_path, tempo=120):
        called["peak"] = True
        return [0.0]

    monkeypatch.setattr(vocal_sync, "extract_onsets", fake_onsets)
    monkeypatch.setattr(vocal_sync, "extract_long_rests", fake_rests)
    monkeypatch.setattr(vocal_sync, "load_consonant_peaks", fake_peaks)

    cfg = {
        "vocal_midi_path_for_drums": str(midi_path),
        "vocal_peak_json_for_drums": str(peaks_json),
        "heatmap_json_path_for_drums": str(tmp_path / "h.json"),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    (tmp_path / "h.json").write_text("[]")
    drum = DrumGenerator(
        main_cfg=cfg,
        part_name="drums",
        part_parameters={},
    )
    section = {"absolute_offset": 0.0, "length_in_measures": 1}
    drum.compose(section_data=section)
    assert all(called.values())
