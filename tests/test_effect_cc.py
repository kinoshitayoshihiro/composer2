import pytest
import json
from pathlib import Path

import pytest
from music21 import instrument
from generator.guitar_generator import GuitarGenerator


@pytest.fixture
def _gen():
    def factory():
        return GuitarGenerator(
            global_settings={},
            default_instrument=instrument.Guitar(),
            part_name="g",
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
        )

    return factory


def test_fx_cc_injection(_gen):
    gen = _gen()
    gen.part_parameters["pat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pat"}},
        "musical_intent": {"intensity": "medium"},
        "fx_params": {"reverb_send": 80, "chorus_send": 60},
    }
    part = gen.compose(section_data=sec)
    ccs = {(e.get("cc"), e.get("val")) for e in part.extra_cc}
    assert (91, 80) in ccs
    assert (93, 60) in ccs
    assert any(e[0] == 31 for e in ccs)
    times = [e["time"] for e in part.extra_cc]
    assert times == sorted(times)


def test_effect_envelope_increasing(_gen):
    gen = _gen()
    gen.part_parameters["pat"] = {
        "pattern": [{"offset": 0.0, "duration": 4.0}],
        "reference_duration_ql": 4.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pat"}},
        "fx_envelope": {
            0.0: {"cc": 91, "start_val": 0, "end_val": 100, "duration_ql": 4.0, "shape": "lin"}
        },
    }
    part = gen.compose(section_data=sec)
    events = [e for e in part.extra_cc if e.get("cc") == 91]
    events.sort(key=lambda x: x["time"])
    vals = [e["val"] for e in events if e["time"] > 0.0]
    assert vals == sorted(vals) and vals[0] < vals[-1]


def test_export_audio_realtime(monkeypatch, tmp_path, _gen):
    gen = _gen()
    p = gen.compose(
        section_data={
            "section_name": "A",
            "q_length": 1.0,
            "humanized_duration_beats": 1.0,
            "original_chord_label": "C",
            "chord_symbol_for_voicing": "C",
            "part_params": {"g": {"guitar_rhythm_key": "pat"}},
            "fx_params": {"reverb_send": 80},
        }
    )
    midi = tmp_path / "in.mid"
    wav = tmp_path / "out.wav"
    midi.write_text("dummy")

    calls = []

    class Dummy:
        tempo = type("T", (), {"events": [{"beat": 0.0, "bpm": 120.0}]})()

        def send_cc(self, cc, value, time):
            calls.append((cc, value))

    gen.export_audio_old(midi, wav, realtime=True, streamer=Dummy())
    assert calls


def test_export_audio_realtime_fx_env(monkeypatch, tmp_path, _gen):
    gen = _gen()
    gen.part_parameters["pat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pat"}},
        "fx_envelope": {0.0: {"cc": 91, "start_val": 0, "end_val": 100, "duration_ql": 1.0}},
    }
    gen.compose(section_data=sec)
    midi = tmp_path / "i.mid"
    wav = tmp_path / "o.wav"
    midi.write_text("m")

    recorded = []

    class Dummy:
        tempo = type("T", (), {"events": [{"beat": 0.0, "bpm": 120.0}]})()

        def send_cc(self, cc, value, time):
            recorded.append(cc)

    gen.export_audio_old(midi, wav, realtime=True, streamer=Dummy())
    assert 91 in recorded


def test_mix_metadata_json(tmp_path, monkeypatch, _gen):
    gen = _gen()
    gen.tone_shaper.ir_map["clean"] = "ir.wav"
    gen.part_parameters["pat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pat"}},
        "fx_params": {"reverb_send": 80},
    }
    part = gen.compose(section_data=sec)
    midi = tmp_path / "i.mid"
    wav = tmp_path / "o.wav"
    midi.write_text("m")

    def fake_export(mp, ow, part=None, **kw):
        Path(ow).touch()
        return Path(ow)

    import utilities.synth as synth

    monkeypatch.setattr(synth, "export_audio", fake_export)

    gen.export_audio_old(midi, wav, write_mix_json=True)
    meta = json.loads(wav.with_suffix(".json").read_text())
    key = part.id or "part"
    assert "ir_file" in meta[key]
    assert meta[key]["extra_cc"]
