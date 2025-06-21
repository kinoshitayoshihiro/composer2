import json
import random
from pathlib import Path

import pytest
from music21 import stream, note
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP
from utilities import humanizer


def _cfg(tmp_path: Path, extra_global=None):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heat.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    if extra_global:
        cfg["global_settings"] = extra_global
    return cfg


def test_open_hat_pedal_choke(tmp_path: Path):
    cfg = _cfg(tmp_path, {"open_hat_choke_prob": 1.0})
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = [{"instrument": "ohh", "offset": 0.0, "velocity": 100}]
    drum._apply_pattern(
        part,
        events,
        0.0,
        1.0,
        100,
        "eighth",
        0.5,
        drum.global_ts,
        {},
    )
    notes = sorted(part.flatten().notes, key=lambda n: n.offset)
    assert len(notes) == 2
    ohh_midi = GM_DRUM_MAP["ohh"][1]
    pedal_midi = GM_DRUM_MAP["hh_pedal"][1]
    assert notes[0].pitch.midi == ohh_midi
    assert notes[1].pitch.midi == pedal_midi
    lag = (notes[1].offset - notes[0].offset) * 60 / drum.global_tempo
    assert 0.16 <= lag <= 0.28
    ratio = notes[1].volume.velocity / notes[0].volume.velocity
    assert 0.6 <= ratio <= 0.9


def test_cymbal_articulations(tmp_path: Path):
    cfg = _cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = [
        {"instrument": "ride", "offset": 0.0, "velocity": 90, "articulation": "bell"},
        {"instrument": "crash", "offset": 1.0, "velocity": 80, "articulation": "splash", "duration": 1.0},
        {"instrument": "crash", "offset": 2.0, "velocity": 70, "articulation": "choke"},
    ]
    drum._apply_pattern(
        part,
        events,
        0.0,
        4.0,
        100,
        "eighth",
        0.5,
        drum.global_ts,
        {},
    )
    notes = sorted(part.flatten().notes, key=lambda n: n.offset)
    bell = notes[0]
    splash = notes[1]
    choke_main = next(n for n in notes if n.offset == pytest.approx(2.0))
    choke_off = next(n for n in notes if n.volume.velocity == 0)

    assert bell.pitch.midi == GM_DRUM_MAP["ride_bell"][1]
    assert bell.volume.velocity > 90

    assert splash.pitch.midi == GM_DRUM_MAP["splash"][1]
    dur_sec = float(splash.duration.quarterLength * 60 / drum.global_tempo)
    assert dur_sec <= 0.3 + 1e-6

    assert choke_main.pitch.midi == GM_DRUM_MAP["crash_choke"][1]
    lag = (choke_off.offset - choke_main.offset) * 60 / drum.global_tempo
    assert pytest.approx(lag, abs=0.02) == 0.2


from hypothesis import given, strategies as st


@given(st.integers(min_value=60, max_value=120))
def test_ghost_jitter_range(v):
    rng = random.Random(0)
    shifts = []
    for _ in range(100):
        n = note.Note()
        n.pitch.midi = GM_DRUM_MAP["snare"][1]
        n.duration.quarterLength = 0.25
        n.volume.velocity = v
        n.offset = 0.0
        humanizer.apply_ghost_jitter(n, rng, tempo_bpm=120)
        shift = float(n.offset * 60 / 120)
        assert -0.015 <= shift <= 0.015
        shifts.append(shift)
    mean = sum(shifts) / len(shifts)
    assert -0.007 <= mean <= -0.001
