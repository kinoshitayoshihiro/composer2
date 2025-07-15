import tempfile
from pathlib import Path
import pretty_midi

from utilities import perc_sampler_v1
from modular_composer.perc_generator import PercGenerator
from generator.drum_generator import DrumGenerator


def _create_conga_shaker(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(8):
        start = i * 0.5
        inst.notes.append(pretty_midi.Note(80, 39, start, start + 0.1))
        inst.notes.append(pretty_midi.Note(70, 70, start + 0.25, start + 0.35))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_perc_generator(tmp_path: Path) -> None:
    midi = tmp_path / "loop.mid"
    _create_conga_shaker(midi)
    model_path = tmp_path / "model.pkl"
    model = perc_sampler_v1.train(tmp_path, auto_res=True)
    perc_sampler_v1.save(model, model_path)

    gen = PercGenerator(model_path)
    events = []
    for _ in range(4):
        events.extend(gen.generate_bar())
    assert any(e["instrument"] == "conga_cl" for e in events)
    assert any(e["instrument"] == "shaker" for e in events)

    merged = DrumGenerator.merge_perc_events([
        {"instrument": "kick", "offset": 0.0},
        {"instrument": "snare", "offset": 0.5},
    ], events)

    for ev in merged:
        for d in ["kick", "snare"]:
            if ev["instrument"] == d:
                assert not any(
                    abs(ev["offset"] - p["offset"]) < 1e-6 and p["instrument"] != d
                    for p in merged
                )

