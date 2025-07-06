from pathlib import Path

import pytest

from utilities.audio_env import has_fluidsynth

sf = pytest.importorskip("soundfile")

if not has_fluidsynth():
    pytest.skip("fluidsynth missing", allow_module_level=True)

from generator.guitar_generator import GuitarGenerator


def _dummy_gen():
    from music21 import instrument

    return GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_export_audio_ir(tmp_path, monkeypatch):
    gen = _dummy_gen()
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
    }
    part = gen.compose(section_data=sec)
    if part.metadata is None:
        from music21 import metadata as m21metadata

        part.metadata = m21metadata.Metadata()
    ir = tmp_path / "ir.wav"
    sf.write(ir, [1.0], 44100)
    part.metadata.ir_file = ir

    midi = tmp_path / "in.mid"
    midi.write_text("dummy")
    out = tmp_path / "out.wav"

    calls = {}

    def fake_export(mp, ow, part=None, **kw):
        Path(ow).write_bytes(b"RIFF0000")
        calls["export"] = True
        return Path(ow)

    def fake_conv(iw, irw, ow, gain_db=0.0):
        Path(ow).write_bytes(b"RIFF1111")
        calls["conv"] = True

    import utilities.convolver as conv
    import utilities.synth as synth

    monkeypatch.setattr(synth, "export_audio", fake_export)
    monkeypatch.setattr(conv, "render_with_ir", fake_conv)

    gen.export_audio_old(midi, out)

    assert calls.get("conv")
    assert out.read_bytes().startswith(b"RIFF")


def test_export_audio_missing_ir(tmp_path, monkeypatch):
    gen = _dummy_gen()
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
    }
    part = gen.compose(section_data=sec)
    if part.metadata is None:
        from music21 import metadata as m21metadata

        part.metadata = m21metadata.Metadata()
    part.metadata.ir_file = tmp_path / "missing.wav"

    midi = tmp_path / "in.mid"
    midi.write_text("dummy")
    out = tmp_path / "out.wav"

    called = False

    def fake_export(mp, ow, part=None, **kw):
        Path(ow).write_bytes(b"RIFF0000")
        return Path(ow)

    def fake_conv(*a, **k):
        nonlocal called
        called = True

    import utilities.convolver as conv
    import utilities.synth as synth

    monkeypatch.setattr(synth, "export_audio", fake_export)
    monkeypatch.setattr(conv, "render_with_ir", fake_conv)

    gen.export_audio_old(midi, out)

    assert not called
    assert out.read_bytes().startswith(b"RIFF")


@pytest.mark.audio
def test_export_audio_basic(tmp_path):
    gen = _dummy_gen()
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
    }
    part = gen.compose(section_data=sec)
    if part.metadata is None:
        from music21 import metadata as m21metadata

        part.metadata = m21metadata.Metadata()
    ir = tmp_path / "ir.wav"
    sf.write(ir, [1.0], 44100)
    out = tmp_path / "out.wav"
    result = gen.export_audio(ir_name=str(ir), out_path=out)
    assert isinstance(result, Path) and result.is_file()
    data, sr = sf.read(result)
    assert abs(len(data) / sr - 0.5) < 0.1
