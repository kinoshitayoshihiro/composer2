import json
from pathlib import Path
from music21 import stream
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP
from utilities.override_loader import PartOverride


def _cfg(tmp_path: Path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }


def test_articulation_variants(tmp_path: Path):
    cfg = _cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = [
        {"instrument": "kick", "offset": 0.0, "velocity": 90},
        {"instrument": "chh", "offset": 1.0, "velocity": 80},
        {"instrument": "chh", "offset": 2.0, "velocity": 40},
        {"instrument": "chh", "offset": 3.0, "velocity": 80, "pedal": True},
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
    pitches = {n.pitch.midi for n in part.flatten().notes}
    expected = {
        GM_DRUM_MAP["kick"][1],
        GM_DRUM_MAP["chh"][1],
        GM_DRUM_MAP["hh_edge"][1],
        GM_DRUM_MAP["hh_pedal"][1],
    }
    assert pitches == expected


def test_velocity_random_walk(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["random_walk_step"] = 4
    cfg["rng_seed"] = 0
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = [{"instrument": "snare", "offset": i} for i in range(32)]
    drum._apply_pattern(
        part,
        events,
        0.0,
        32.0,
        80,
        "eighth",
        0.5,
        drum.global_ts,
        {},
    )
    notes = list(part.flatten().notes)
    assert len({n.volume.velocity for n in notes}) > 4


def test_brush_velocity_scaling(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["drum_brush"] = True
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    hit = drum._make_hit("snare", 100, 0.5)
    assert hit.volume.velocity < 70


def test_brush_override_scaling(tmp_path: Path):
    cfg = _cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    drum.overrides = PartOverride(drum_brush=True)
    part = stream.Part(id="drums")
    events = [
        {"instrument": "snare", "offset": 0.0, "velocity": 90},
        {"instrument": "snare", "offset": 1.0, "velocity": 100},
    ]
    orig_vel = [e["velocity"] for e in events]
    drum._apply_pattern(
        part,
        events,
        0.0,
        2.0,
        80,
        "eighth",
        0.5,
        drum.global_ts,
        {},
    )
    velocities = [n.volume.velocity for n in part.flatten().notes]
    assert velocities and all(v < o for v, o in zip(velocities, orig_vel))


def test_intro_ride_notes(tmp_path: Path):
    """Dummy Intro section should map 'ride' instrument correctly."""
    cfg = _cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = [
        {"instrument": "ride", "offset": 0.0, "velocity": 90},
        {"instrument": "ride", "offset": 1.0, "velocity": 90},
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
    notes = list(part.flatten().notes)
    assert notes
    assert all(p.pitch.midi == GM_DRUM_MAP["ride"][1] for p in notes)
