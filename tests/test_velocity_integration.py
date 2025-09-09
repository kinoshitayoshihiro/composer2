import time
from unittest.mock import patch

import numpy as np
import pytest
from music21 import instrument

from generator.bass_generator import BassGenerator
from generator.piano_generator import PianoGenerator
from utilities.kde_velocity import KDEVelocityModel
from utilities.velocity_model import pretty_midi
from utilities.velocity_utils import scale_velocity

# 依存ライブラリがない場合はスキップ
pytest.importorskip("pretty_midi")

# pretty_midi がロードできないとき、Bass/Piano テストはスキップ
pytestmark = pytest.mark.skipif(
    pretty_midi is None,
    reason="pretty_midi not installed for bass/piano tests",
)


@pytest.fixture()
def velocity_model() -> KDEVelocityModel:
    return KDEVelocityModel(np.array([64.0], dtype=np.float32))


def test_kde_velocity_accuracy(velocity_model: KDEVelocityModel) -> None:
    ctx = np.random.rand(32, 1).astype(np.float32)
    preds = velocity_model.predict(ctx)
    mse = np.mean((preds - 64) ** 2)
    assert mse < 0.02


def test_kde_velocity_speed(velocity_model: KDEVelocityModel) -> None:
    ctx = np.random.rand(32, 1).astype(np.float32)
    start = time.time()
    for _ in range(100):
        velocity_model.predict(ctx)
    avg_ms = (time.time() - start) / 100 * 1000
    assert avg_ms < 50


class DummyVelocity:
    def __init__(self, val: int) -> None:
        self.val = val

    def sample(self, part: str, pos_beat: float) -> int:
        return self.val


def make_bass_gen():
    return BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={"global_settings": {"key_tonic": "C", "key_mode": "major"}},
        velocity_model=DummyVelocity(80),
    )


def make_piano_gen(patterns):
    class SimplePiano(PianoGenerator):
        def _get_pattern_keys(self, mi, ov):
            return "rh", "lh"

    return SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
        velocity_model=DummyVelocity(70),
    )


def test_bass_velocity_model():
    gen = make_bass_gen()
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {},
        "tonic_of_section": "C",
        "mode": "major",
    }
    # Patch the symbol used inside BasePartGenerator
    with patch("generator.base_part_generator.scale_velocity", wraps=scale_velocity) as spy:
        part = gen.compose(section_data=section)
        assert spy.called
    for n in part.flatten().notes:
        assert n.volume.velocity == 80


def test_piano_velocity_model():
    patterns = {
        "rh": {
            "pattern": [{"offset": 0, "duration": 1, "type": "chord"}],
            "length_beats": 1.0,
        },
        "lh": {
            "pattern": [{"offset": 0, "duration": 1, "type": "root"}],
            "length_beats": 1.0,
        },
    }
    gen = make_piano_gen(patterns)
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {},
        "part_params": {},
    }
    # Patch the symbol used inside BasePartGenerator
    with patch("generator.base_part_generator.scale_velocity", wraps=scale_velocity) as spy:
        parts = gen.compose(section_data=section)
        assert spy.called
        for p in parts.values():
            for n in p.flatten().notes:
                assert n.volume.velocity == 70
