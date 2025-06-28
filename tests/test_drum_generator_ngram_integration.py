import json
from pathlib import Path

import pretty_midi

from generator.drum_generator import DrumGenerator
from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def _cfg(tmp_path: Path) -> dict:
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "global_settings": {"rng_seed": 0},
    }


def test_drum_generator_ngram_integration(tmp_path: Path) -> None:
    for i in range(2):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path, order=2)
    cfg = _cfg(tmp_path)
    gen = DrumGenerator(main_cfg=cfg, part_name="drums", part_parameters={})
    gen.groove_model = model
    section = {"absolute_offset": 0.0, "q_length": 16.0, "part_params": {}}
    part = gen.compose(section_data=section)
    assert len(list(part.flatten().notes)) >= 1
