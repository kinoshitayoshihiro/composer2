import numpy as np
import pytest
import soundfile as sf

from utilities.audio_env import has_fluidsynth

if not has_fluidsynth():
    pytest.skip("fluidsynth missing", allow_module_level=True)

from utilities.tone_shaper import ToneShaper


@pytest.mark.fx
def test_guitar_generator_cc31(_basic_gen) -> None:
    gen = _basic_gen()
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "musical_intent": {"intensity": "low"},
        "part_params": {"g": {"amp_preset": "clean", "fx_preset_intensity": "low"}},
    }
    part = gen.compose(section_data=sec)
    cc31 = [c for c in part.extra_cc if c["cc"] == 31]
    assert cc31 and cc31[0]["val"] == 20
    assert hasattr(part.metadata, "extra_cc")
    assert {"time": 0.0, "cc": 31, "val": 20} in part.metadata.extra_cc


@pytest.mark.fx
def test_ir_render(tmp_path):
    ir = tmp_path / "ir.wav"
    sf.write(ir, np.zeros(10), 44100)
    ts = ToneShaper({"clean": {"amp": 20}}, {"clean": ir})
    mix = tmp_path / "mix.wav"
    sf.write(mix, np.zeros(10), 44100)
    out = tmp_path / "out.wav"
    ts.render_with_ir(mix, "clean", out)
    assert out.read_bytes().startswith(b"RIFF")
