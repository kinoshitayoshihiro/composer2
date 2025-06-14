import json
from pathlib import Path

from generator.drum_generator import DrumGenerator
from modular_composer import compose

from utilities.rhythm_library_loader import load_rhythm_library

class EightKickDrum(DrumGenerator):
    def _resolve_style_key(self, musical_intent, overrides, section_data=None):
        return "eight_kick"


def test_drum_kick_offsets(tmp_path):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    heatmap_path = tmp_path / "h.json"
    with open(heatmap_path, "w") as f:
        json.dump(heatmap, f)

    pattern = [{"offset": i * 0.5, "instrument": "kick"} for i in range(8)]
    lib = {"eight_kick": {"pattern": pattern, "length_beats": 4.0}}
    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heatmap_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    drum = EightKickDrum(
        main_cfg=cfg,
        part_name="drums",
        part_parameters=lib,
        global_time_signature="4/4",
        global_tempo=120,
    )
    section = {"absolute_offset": 0.0, "q_length": 4.0, "musical_intent": {}, "part_params": {}}
    drum.compose(section_data=section, shared_tracks={})
    assert len(drum.get_kick_offsets()) == 8


def test_compose_shared_kicks(tmp_path):
    rhythm_lib = load_rhythm_library(Path("data/rhythm_library.yml"))
    main_cfg = {
        "global_settings": {"time_signature": "4/4", "tempo_bpm": 120},
        "sections_to_generate": ["A"],
        "part_defaults": {"drums": {"role": "drums"}, "bass": {"role": "bass"}},
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    chordmap = {
        "sections": {
            "A": {
                "processed_chord_events": [
                    {
                        "absolute_offset_beats": 0.0,
                        "humanized_duration_beats": 4.0,
                        "chord_symbol_for_voicing": "C",
                    }
                ],
                "musical_intent": {"emotion": "neutral"},
                "expression_details": {},
            }
        }
    }
    score, sections = compose(main_cfg, chordmap, rhythm_lib)
    assert sections[0]["shared_tracks"]["kick_offsets"]
