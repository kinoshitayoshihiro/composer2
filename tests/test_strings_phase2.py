import importlib.util
import statistics
import sys
from pathlib import Path

import pytest
from music21 import instrument

ROOT = Path(__file__).resolve().parents[1]
pkg = type(sys)("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

_MOD_PATH = ROOT / "generator" / "strings_generator.py"
spec = importlib.util.spec_from_file_location("generator.strings_generator", _MOD_PATH)
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def _gen(**kwargs):
    return StringsGenerator(
        global_settings={},
        default_instrument=instrument.Violin(),
        part_name="strings",
        global_tempo=kwargs.pop("global_tempo", 120),
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs,
    )


def test_velocity_curve_mapping():
    curve = [30, 80, 110]
    gen = _gen(default_velocity_curve=curve)
    assert gen.default_velocity_curve[0] == 30
    assert gen.default_velocity_curve[64] == 80
    assert gen.default_velocity_curve[127] == 109
    # midpoint checks
    assert gen.default_velocity_curve[32] == 55
    assert gen.default_velocity_curve[96] == 95


def test_timing_jitter_bpm_scale():
    sec = _basic_section()
    sec["events"] = [{"duration": 1.0} for _ in range(8)]
    gen_lo = _gen(global_tempo=60, timing_jitter_ms=30, timing_jitter_scale_mode="bpm_relative")
    gen_hi = _gen(global_tempo=180, timing_jitter_ms=30, timing_jitter_scale_mode="bpm_relative")
    gen_lo.rng.seed(1)
    gen_hi.rng.seed(1)
    notes_lo = gen_lo.compose(section_data=sec)["violin_i"].flatten().notes
    notes_hi = gen_hi.compose(section_data=sec)["violin_i"].flatten().notes
    diffs_lo = [float(n.offset) - i for i, n in enumerate(notes_lo)]
    diffs_hi = [float(n.offset) - i for i, n in enumerate(notes_hi)]
    ratio = statistics.pstdev(diffs_lo) / statistics.pstdev(diffs_hi)
    assert ratio == pytest.approx(3.0, rel=0.2)


def test_bow_position_meta():
    gen = _gen()
    sec = _basic_section()
    sec["bow_position"] = "tasto"
    parts = gen.compose(section_data=sec)
    n = list(parts["violin_i"].flatten().notes)[0]
    meta = getattr(n.style, "bowPosition", getattr(n.style, "other", None))
    assert meta == strings_module.BowPosition.TASTO.value


def test_balance_scale_effect():
    gen_hi = _gen()
    gen_lo = _gen(balance_scale=0.5)
    sec = _basic_section()
    parts_hi = gen_hi.compose(section_data=sec)
    parts_lo = gen_lo.compose(section_data=sec)
    v_hi = parts_hi["violin_i"].flatten().notes[0].volume.velocity
    v_lo = parts_lo["violin_i"].flatten().notes[0].volume.velocity
    assert v_lo < v_hi and v_lo >= 20


def test_balance_scale_high_clamp():
    gen_hi = _gen(balance_scale=1.5)
    sec = _basic_section()
    parts = gen_hi.compose(section_data=sec)
    vel = parts["violin_i"].flatten().notes[0].volume.velocity
    assert 1 <= vel <= 127
    gen_def = _gen()
    vel_def = gen_def.compose(section_data=sec)["violin_i"].flatten().notes[0].volume.velocity
    assert vel > vel_def


def test_jitter_bar_boundary():
    sec = _basic_section()
    sec["events"] = [{"duration": 1.0} for _ in range(4)]
    gen = _gen(global_tempo=60, timing_jitter_ms=100)
    gen.rng.seed(2)
    parts = gen.compose(section_data=sec)
    starts = [float(n.offset) for n in parts["violin_i"].flatten().notes]
    assert all(s < sec["q_length"] for s in starts)
