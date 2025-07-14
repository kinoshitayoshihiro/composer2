import random
import pytest
from generator.drum_generator import DrumGenerator

@pytest.fixture()
def drum():
    cfg = {"global_settings": {"tempo_bpm": 120}}
    return DrumGenerator(main_cfg=cfg, part_name="drums")

@pytest.mark.parametrize(
    "intensity,expected",
    [
        (0.0, 0.05),
        (0.3, 0.15),
        (0.75, 0.44375),
        (1.0, 0.60),
    ],
)
def test_fill_density_interp(drum, intensity, expected):
    assert abs(drum.fill_density(intensity) - expected) < 1e-3


def test_emotional_fill_calls(monkeypatch):
    cfg = {"rng_seed": 0}
    d = DrumGenerator(main_cfg=cfg, part_name="drums")
    calls = []
    def fake_insert(self, part, section, fill_key=None):
        calls.append(1)

    monkeypatch.setattr(
        type(d.fill_inserter), "insert", fake_insert, raising=False
    )
    section = {
        "absolute_offset": 0.0,
        "q_length": 8.0,
        "length_in_measures": 2,
        "musical_intent": {"emotion_intensity": 0.2},
        "part_params": {},
    }
    d.rng.seed(0)
    for _ in range(10):
        d.compose(section_data=section)
    low = len(calls)
    calls.clear()
    section["musical_intent"] = {"emotion_intensity": 0.8}
    d.rng.seed(0)
    for _ in range(10):
        d.compose(section_data=section)
    high = len(calls)
    assert high > low
