import json
from pathlib import Path

import pretty_midi
import pytest
from generator.drum_generator import DrumGenerator
from utilities import groove_sampler_ngram as gs


def _mk_loop(path: Path, pitch: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=i * 0.25, end=i * 0.25 + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_generator_ngram_integration(tmp_path: Path, rhythm_library) -> None:
    _mk_loop(tmp_path / "a.mid", 36)
    _mk_loop(tmp_path / "b.mid", 38)
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")

    heat = [{"grid_index": i, "count": 0} for i in range(gs.RESOLUTION)]
    heat_path = tmp_path / "heat.json"
    heat_path.write_text(json.dumps(heat))

    cfg = {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(heat_path),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {
            "groove_strength": 1.0,
            "humanize_profile": "some",
            "groove_temperature": 1.0,
        },
    }
    drum = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    drum.groove_model = gs.load(tmp_path / "m.pkl")
    section = {
        "absolute_offset": 0.0,
        "length_in_measures": 4,
        "musical_intent": {"section": "verse", "intensity": "mid"},
    }
    part = drum.compose(section_data=section)
    notes = list(part.flatten().notes)
    assert notes
    velocities = [n.volume.velocity for n in notes]
    humanised = sum(v != 100 for v in velocities)
    assert humanised / len(velocities) >= 0.5
