import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument, articulations, pitch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

_MOD_PATH = ROOT / "generator" / "strings_generator.py"
spec = importlib.util.spec_from_file_location("generator.strings_generator", _MOD_PATH)
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator
EXEC_STYLE_TRILL = strings_module.EXEC_STYLE_TRILL
EXEC_STYLE_TREMOLO = strings_module.EXEC_STYLE_TREMOLO


def _gen(**kwargs):
    return StringsGenerator(
        global_settings={},
        default_instrument=instrument.Violin(),
        part_name="strings",
        global_tempo=kwargs.pop("tempo", 120),
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs,
    )


def test_trill_pattern_notes():
    gen = _gen()
    base_p = pitch.Pitch("C4")
    notes = gen._create_notes_from_event(
        base_p,
        0.5,
        "violin_i",
        None,
        80,
        1.0,
        None,
        {"pattern_type": EXEC_STYLE_TRILL, "interval": 1, "rate_hz": 10},
    )
    assert len(notes) > 1
    mids = [n.pitch.midi for n in notes]
    assert mids[0] != mids[1]
    assert abs(notes[1].offset - notes[0].offset - 0.05) < 0.01


def test_tremolo_pattern_articulation():
    gen = _gen()
    base_p = pitch.Pitch("C4")
    notes = gen._create_notes_from_event(
        base_p,
        0.3,
        "violin_i",
        None,
        80,
        1.0,
        None,
        {"pattern_type": EXEC_STYLE_TREMOLO, "rate_hz": 10},
    )
    assert len(notes) > 1
    for n in notes:
        assert any(isinstance(a, articulations.Tremolo) for a in n.articulations)
