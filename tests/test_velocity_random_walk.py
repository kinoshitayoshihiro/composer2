import random

from utilities.accent_mapper import AccentMapper
from utilities.velocity_smoother import EMASmoother
from generator.drum_generator import DrumGenerator, RESOLUTION, INTENSITY_FACTOR
from music21 import stream

def test_velocity_random_walk_per_bar():
    rng = random.Random(0)
    heatmap = {0: 10}
    settings = {
        "accent_threshold": 0.0,
        "ghost_density_range": (0.3, 0.8),
        "random_walk_step": 8,
    }
    mapper = AccentMapper(heatmap, settings, rng=rng)
    base = 80
    velocities = []
    for _ in range(8):
        mapper.begin_bar()
        velocities.append(mapper.accent(0, base, apply_walk=True))
    step_range = settings["random_walk_step"]
    base_after_accent = int(round(base * 1.2))
    diffs = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
    assert all(
        base_after_accent - step_range <= v <= base_after_accent + step_range
        for v in velocities
    )
    assert all(d <= 2 * step_range for d in diffs)


def test_walk_after_ema_order_change():
    heatmap = {0: 0}
    base_settings = {
        "accent_threshold": 0.0,
        "ghost_density_range": (0.3, 0.8),
        "random_walk_step": 4,
    }
    rng_seed = 1
    a = AccentMapper(
        heatmap,
        {**base_settings, "walk_after_ema": False},
        rng=random.Random(rng_seed),
        ema_smoother=EMASmoother(),
        use_velocity_ema=True,
        walk_after_ema=False,
    )
    b = AccentMapper(
        heatmap,
        {**base_settings, "walk_after_ema": True},
        rng=random.Random(rng_seed),
        ema_smoother=EMASmoother(),
        use_velocity_ema=True,
        walk_after_ema=True,
    )
    vals_a = []
    vals_b = []
    for _ in range(6):
        a.begin_bar()
        b.begin_bar()
        vals_a.append(a.accent(0, 80))
        vals_b.append(b.accent(0, 80))
    assert vals_a != vals_b


def test_intensity_step_scaling(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "h.json"
    import json
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "global_settings": {"random_walk_step": 8},
        "paths": {"drum_pattern_files": []},
        "rng_seed": 0,
    }
    pattern = {"main": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0, "velocity_base": 80}}
    gen = DrumGenerator(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern,
        global_settings=cfg["global_settings"],
    )
    part_low = stream.Part(id="drums")
    low_step = int(8 * INTENSITY_FACTOR["low"])
    for bar in range(8):
        gen.accent_mapper.begin_bar(bar * 4.0, low_step)
        gen._apply_pattern(
            part_low,
            pattern["main"]["pattern"],
            bar * 4.0,
            4.0,
            80,
            "eighth",
            0.5,
            gen.global_ts,
            {"musical_intent": {"intensity": "low"}},
        )
    v_low = [n.volume.velocity for n in part_low.flatten().notes]
    range_low = max(v_low) - min(v_low)

    part_high = stream.Part(id="drums")
    high_step = int(8 * INTENSITY_FACTOR["high"])
    for bar in range(8):
        gen.accent_mapper.begin_bar(bar * 4.0, high_step)
        gen._apply_pattern(
            part_high,
            pattern["main"]["pattern"],
            bar * 4.0,
            4.0,
            80,
            "eighth",
            0.5,
            gen.global_ts,
            {"musical_intent": {"intensity": "high"}},
        )
    v_high = [n.volume.velocity for n in part_high.flatten().notes]
    range_high = max(v_high) - min(v_high)

    assert range_high > range_low


def test_random_walk_cc_export(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    heatmap_path = tmp_path / "h.json"
    import json
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "global_settings": {
            "random_walk_step": 4,
            "export_random_walk_cc": True,
        },
        "paths": {"drum_pattern_files": []},
        "rng_seed": 0,
    }
    pattern = {"main": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0, "velocity_base": 80}}
    gen = DrumGenerator(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern,
        global_settings=cfg["global_settings"],
    )
    part = stream.Part(id="drums")
    mid_step = int(4 * INTENSITY_FACTOR["medium"])
    for bar in range(2):
        gen.accent_mapper.begin_bar(bar * 4.0, mid_step)
        gen._apply_pattern(
            part,
            pattern["main"]["pattern"],
            bar * 4.0,
            4.0,
            80,
            "eighth",
            0.5,
            gen.global_ts,
            {"musical_intent": {"intensity": "medium"}},
        )
    cc20 = [c for c in getattr(part, "extra_cc", []) if c["number"] == 20]
    assert len(cc20) == 2


def test_begin_bar_once_per_bar(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(RESOLUTION)]
    h = tmp_path / "hm.json"
    import json
    with open(h, "w") as f:
        json.dump(heatmap, f)

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(h),
        "global_settings": {"random_walk_step": 4},
        "paths": {"drum_pattern_files": []},
        "rng_seed": 0,
    }
    pattern = {"main": {"pattern": [{"instrument": "snare", "offset": 0.0}], "length_beats": 4.0, "velocity_base": 80}}
    gen = DrumGenerator(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=pattern,
        global_settings=cfg["global_settings"],
    )
    section = {
        "section_name": "X",
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "musical_intent": {},
        "part_params": {},
    }
    from unittest.mock import patch

    with patch.object(gen.accent_mapper, "begin_bar", wraps=gen.accent_mapper.begin_bar) as mock_b:
        gen.compose(section_data=section)
        assert mock_b.call_count == 2


def test_drum_global_settings_persist():
    dg = DrumGenerator(main_cfg={}, part_name="drums", part_parameters={}, global_settings={"random_walk_step": 6})
    assert dg.global_settings["random_walk_step"] == 6


def test_begin_bar_records_offset():
    am = AccentMapper({}, {"random_walk_step": 4})
    am.begin_bar(8.0)
    assert am.debug_rw_values[-1][0] == 8.0


def test_step_range_updates_immediately():
    am = AccentMapper({}, {"random_walk_step": 8})
    low_step = int(8 * INTENSITY_FACTOR["low"])
    high_step = int(8 * INTENSITY_FACTOR["high"])
    am.begin_bar(0.0, low_step)
    assert am._step_range == low_step
    am.begin_bar(4.0, high_step)
    assert am._step_range == high_step

