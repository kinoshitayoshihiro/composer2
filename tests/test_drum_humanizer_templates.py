import json
from unittest.mock import patch
from pathlib import Path
from music21 import volume
from generator.drum_generator import DrumGenerator


def _basic_cfg(tmp_path: Path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }


pattern_lib = {
    "main": {
        "pattern": [
            {
                "instrument": "kick",
                "offset": 0.0,
                "humanize_template": "flam_legato_ghost",
            }
        ],
        "length_beats": 1.0,
    }
}

pattern_lib_multi = {
    "main": {
        "pattern": [
            {
                "instrument": "kick",
                "offset": 0.0,
                "humanize_templates": ["drum_tight", "flam_legato_ghost"],
            }
        ],
        "length_beats": 1.0,
    }
}

pattern_lib_random = {
    "main": {
        "pattern": [
            {
                "instrument": "kick",
                "offset": 0.0,
                "humanize_templates": ["drum_tight", "flam_legato_ghost"],
                "humanize_templates_mode": "random",
            }
        ],
        "length_beats": 1.0,
    }
}


@patch("generator.drum_generator.apply_humanization_to_element")
def test_flam_legato_ghost_template_applied(mock_apply, tmp_path: Path):
    def modify(n, template_name=None, custom_params=None):
        n.offset += 0.001
        n.duration.quarterLength += 0.05
        if n.volume and n.volume.velocity is not None:
            n.volume.velocity += 1
        else:
            n.volume = volume.Volume(velocity=1)
        return n

    mock_apply.side_effect = modify
    cfg = _basic_cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    section = {
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "part_params": {"drums": {"final_style_key_for_render": "main"}},
    }
    part = drum.compose(section_data=section)

    mock_apply.assert_called()
    note_obj = list(part.flatten().notes)[0]
    assert note_obj.duration.quarterLength > 0.125
    assert note_obj.volume.velocity > 1


@patch("generator.drum_generator.apply_humanization_to_element")
def test_multiple_templates_sequential(mock_apply, tmp_path: Path):
    call_order = []

    def record(n, template_name=None, custom_params=None):
        call_order.append(template_name)
        return n

    mock_apply.side_effect = record
    cfg = _basic_cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib_multi)
    section = {
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "part_params": {"drums": {"final_style_key_for_render": "main"}},
    }
    drum.compose(section_data=section)

    assert call_order == ["drum_tight", "flam_legato_ghost"]


@patch("generator.drum_generator.apply_humanization_to_element")
def test_multiple_templates_random(mock_apply, tmp_path: Path):
    chosen_templates = []

    def record(n, template_name=None, custom_params=None):
        chosen_templates.append(template_name)
        return n

    mock_apply.side_effect = record
    cfg = _basic_cfg(tmp_path)
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib_random)
    section = {
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "part_params": {"drums": {"final_style_key_for_render": "main"}},
    }
    drum.compose(section_data=section)

    assert len(chosen_templates) == 1
    assert chosen_templates[0] in {"drum_tight", "flam_legato_ghost"}
