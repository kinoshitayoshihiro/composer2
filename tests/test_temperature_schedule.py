from pathlib import Path
from utilities import groove_sampler_v2
import pretty_midi


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_temperature_schedule(tmp_path: Path, monkeypatch) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_v2.train(tmp_path)
    temps: list[float] = []
    orig = groove_sampler_v2.sample_next

    def wrapper(*args, temperature: float, **kwargs):
        temps.append(temperature)
        return orig(*args, temperature=temperature, **kwargs)

    monkeypatch.setattr(groove_sampler_v2, "sample_next", wrapper)
    groove_sampler_v2.generate_events(model, bars=1, temperature=1.0, temperature_end=0.5, seed=0)
    assert temps[0] == 1.0
    assert temps[-1] <= 0.55

