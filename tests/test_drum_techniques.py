import json
from music21 import note
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP


class SimpleDrum(DrumGenerator):
    def _resolve_style_key(self, mi, ov, section_data=None):
        return "flam_test"


def _make_cfg(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {},
    }


pattern_lib = {
    "flam_test": {
        "pattern": [{"instrument": "snare", "offset": 0.5, "type": "flam"}],
        "length_beats": 1.0,
    },
    "ghost_pat": {
        "pattern": [{"instrument": "snare", "offset": 0.0, "type": "ghost"}],
        "length_beats": 1.0,
    },
    "legato_fill": {
        "pattern": [
            {"instrument": "snare", "offset": 3.0},
            {"instrument": "snare", "offset": 3.5},
        ],
        "length_beats": 4.0,
        "legato": True,
    },
    "main": {
        "pattern": [{"instrument": "kick", "offset": 0.0}],
        "length_beats": 4.0,
        "fill_patterns": ["legato_fill"],
        "preferred_fill_positions": [1],
    },
}


def test_flam_insertion(tmp_path):
    cfg = _make_cfg(tmp_path)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {"absolute_offset": 0.0, "q_length": 1.0, "part_params": {}}
    part = gen.compose(section_data=section)
    hits = list(part.flatten().notes)
    assert len(hits) == 2
    assert hits[1].offset > hits[0].offset
    assert hits[1].volume.velocity > hits[0].volume.velocity


def test_ghost_note_velocity(tmp_path):
    cfg = _make_cfg(tmp_path)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    gen._resolve_style_key = lambda mi, ov, section_data=None: "ghost_pat"
    section = {
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "part_params": {"drums": {"velocity": 100}},
    }
    part = gen.compose(section_data=section)
    hit = list(part.flatten().notes)[0]
    # velocity may fluctuate due to accent boost and random walk
    assert 16 <= hit.volume.velocity <= 32
    assert hit.duration.quarterLength <= 0.1


def test_legato_ties_in_fill(tmp_path):
    cfg = _make_cfg(tmp_path)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    gen._resolve_style_key = lambda mi, ov, section_data=None: "main"
    section = {
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "length_in_measures": 1,
        "part_params": {},
    }
    part = gen.compose(section_data=section)
    ties = [n.tie for n in part.flatten().notes if n.tie]
    assert any(t.tieType == "hold" for t in ties)


def test_brush_mode_switches_map(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg["global_settings"]["drum_brush"] = True
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    n = gen._make_hit("snare", 80, 0.5)
    assert n.pitch.midi == GM_DRUM_MAP["snare_brush"][1]
    assert n.volume.velocity == int(80 * 0.6)


def test_syncopation_tag_selection():
    gen = DrumGenerator(
        main_cfg={},
        part_name="drums",
        part_parameters={
            "base": {"pattern": [], "length_beats": 4.0},
            "off": {"pattern": [], "length_beats": 4.0, "tags": ["offbeat"]},
        },
    )
    key = gen._choose_pattern_key("default", "medium", {"syncopation": True})
    assert key == "off"
