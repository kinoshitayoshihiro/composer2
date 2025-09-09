import json
from pathlib import Path

import pretty_midi

from generator.drum_generator import DrumGenerator
from utilities import groove_sampler_ngram as gs


def _loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def _cfg(tmp: Path) -> dict:
    heat = [{"grid_index": i, "count": 0} for i in range(gs.RESOLUTION)]
    hp = tmp / "heat.json"
    with hp.open("w") as f:
        json.dump(heat, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {"rng_seed": 0},
    }


def test_sampler_fallback(tmp_path: Path) -> None:
    for i in range(2):
        _loop(tmp_path / f"{i}.mid")
    model = gs.train(tmp_path, order=2)
    cfg = _cfg(tmp_path)
    pattern_lib = {"main": {"pattern": [], "length_beats": 4.0}}
    gen = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters=pattern_lib)
    gen.groove_model = model
    section = {
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "part_params": {"drums": {"final_style_key_for_render": "main"}},
    }
    part = gen.compose(section_data=section)
    assert len(list(part.flatten().notes)) > 0
