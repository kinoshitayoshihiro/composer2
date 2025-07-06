import importlib.util
import sys
import types
from pathlib import Path
from music21 import instrument, articulations, spanner, expressions

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


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 2.0,
        "humanized_duration_beats": 2.0,
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
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs,
    )


def _sig(parts):
    result = []
    for name in sorted(parts.keys()):
        seq = []
        for n in parts[name].flatten().notes:
            if hasattr(n, "pitches"):
                seq.append(tuple(p.midi for p in n.pitches))
            else:
                seq.append(n.pitch.midi)
        result.append((name, tuple(seq)))
    return tuple(result)


def test_event_articulations_propagate():
    gen = _gen()
    sec = _basic_section()
    sec["events"] = [{"duration": 2.0, "articulations": ["staccato"]}]
    parts = gen.compose(section_data=sec)
    note_obj = list(parts["violin_i"].flatten().notes)[0]
    assert any(isinstance(a, articulations.Staccato) for a in note_obj.articulations)


def test_legato_spanner_count():
    gen = _gen()
    sec = _basic_section()
    sec["events"] = [
        {"duration": 1.0, "articulations": ["legato"]},
        {"duration": 1.0, "articulations": ["legato"]},
    ]
    parts = gen.compose(section_data=sec)
    slurs = [s for s in parts["violin_i"].spanners if isinstance(s, spanner.Slur)]
    assert len(slurs) == 1


def test_section_default_articulations():
    gen = _gen()
    sec = _basic_section()
    sec["events"] = [{"duration": 2.0}]
    sec["part_params"] = {"strings": {"default_articulations": ["pizz"]}}
    parts = gen.compose(section_data=sec)
    for p in parts.values():
        n = list(p.flatten().notes)[0]
        assert any(
            isinstance(e, expressions.TextExpression) and e.content == "pizz."
            for e in n.expressions
        )


def test_no_articulation_regression():
    gen = _gen()
    sec = _basic_section()
    base = gen.compose(section_data=sec)

    sec2 = _basic_section()
    sec2["events"] = [{"duration": 2.0}]
    parts = gen.compose(section_data=sec2)
    assert _sig(parts) == _sig(base)
