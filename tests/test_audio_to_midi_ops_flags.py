import argparse
import math
import struct
import warnings
import wave

import pytest

np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")

from utilities.apply_controls import apply_controls  # noqa: E402
from utilities.audio_to_midi_batch import build_control_curves_for_stem  # noqa: E402


def _write_wav(path, data, sr=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767)) for x in data
        )
        wf.writeframes(frames)


class TestOpsFlags:
    def test_emit_cc11(self, tmp_path):
        audio = np.ones(16000)
        wav = tmp_path / "a.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        args = argparse.Namespace(
            emit_cc11=True,
            emit_cc64=False,
            cc_strategy="energy",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: {"cc11": curves["cc11"]}}
        apply_controls(
            pm,
            curves_by_ch,
            bend_range_semitones=2.0,
            sample_rate_hz={"cc": args.controls_sample_rate_hz},
        )
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        assert any(cc.number == 11 for cc in inst0.control_changes)

    def test_no_emit_cc11(self, tmp_path):
        audio = np.ones(16000)
        wav = tmp_path / "a.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        args = argparse.Namespace(
            emit_cc11=False,
            emit_cc64=False,
            cc_strategy="energy",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        assert "cc11" not in curves

    def test_emit_cc64_piano(self, tmp_path):
        audio = np.ones(16000)
        wav = tmp_path / "p.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0, name="Piano")
        inst.notes.append(pretty_midi.Note(60, 0.0, 1.0))
        inst.notes.append(pretty_midi.Note(62, 0.5, 1.5))
        args = argparse.Namespace(
            emit_cc11=False,
            emit_cc64=True,
            cc_strategy="none",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: {"cc64": curves["cc64"]}}
        apply_controls(pm, curves_by_ch, bend_range_semitones=2.0)
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        vals = [cc.value for cc in inst0.control_changes if cc.number == 64]
        assert 127 in vals and 0 in vals

    def test_beats_domain_tempo_map(self, tmp_path):
        audio = np.ones(16000)
        wav = tmp_path / "b.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        args = argparse.Namespace(
            emit_cc11=True,
            emit_cc64=False,
            cc_strategy="energy",
            controls_domain="beats",
            controls_sample_rate_hz=2.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args, tempo=120.0)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: {"cc11": curves["cc11"]}}
        tempo_map = [(0, 120), (1, 60)]
        apply_controls(
            pm,
            curves_by_ch,
            bend_range_semitones=2.0,
            sample_rate_hz={"cc": args.controls_sample_rate_hz},
            tempo_map=tempo_map,
        )
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        last_t = max(cc.time for cc in inst0.control_changes if cc.number == 11)
        assert math.isclose(last_t, 1.5, rel_tol=0.1)

    def test_max_events_limits(self, tmp_path):
        audio = np.ones(32000)
        wav = tmp_path / "c.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        for i in range(50):
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=i * 100, time=i / 50))
        args = argparse.Namespace(
            emit_cc11=True,
            emit_cc64=False,
            cc_strategy="energy",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: curves}
        apply_controls(
            pm,
            curves_by_ch,
            bend_range_semitones=2.0,
            sample_rate_hz={
                "cc": args.controls_sample_rate_hz,
                "bend": args.controls_sample_rate_hz,
            },
            max_events={"cc11": 10, "bend": 10},
        )
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        assert sum(1 for cc in inst0.control_changes if cc.number == 11) <= 10
        assert len(inst0.pitch_bends) <= 10

    def test_write_rpn_once(self, tmp_path):
        audio = np.ones(32000)
        wav = tmp_path / "d.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=100, time=0.1))
        args = argparse.Namespace(
            emit_cc11=False,
            emit_cc64=False,
            cc_strategy="none",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: curves}
        apply_controls(
            pm,
            curves_by_ch,
            bend_range_semitones=2.0,
            write_rpn=True,
        )
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        numbers = [cc.number for cc in inst0.control_changes[:4]]
        assert numbers == [101, 100, 6, 38]
        first_cc_time = inst0.control_changes[0].time
        assert first_cc_time <= inst0.pitch_bends[0].time
        assert inst0.control_changes == sorted(
            inst0.control_changes, key=lambda c: (c.time, c.number, c.value)
        )
        assert inst0.pitch_bends == sorted(
            inst0.pitch_bends, key=lambda b: (b.time, b.pitch)
        )

    def test_controls_res_hz_alias(self, tmp_path):
        audio = np.ones(16000)
        wav = tmp_path / "e.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            args = argparse.Namespace(
                emit_cc11=True,
                emit_cc64=False,
                cc_strategy="energy",
                controls_domain="time",
                controls_sample_rate_hz=40.0,
                controls_res_hz=20.0,
            )
            curves = build_control_curves_for_stem(wav, inst, args)
        msgs = [str(wr.message) for wr in w]
        assert len(msgs) == 1
        assert "deprecated" in msgs[0]
        assert curves["cc11"].sample_rate_hz == 40.0

    def test_total_max_events(self, tmp_path):
        audio = np.ones(32000)
        wav = tmp_path / "tot.wav"
        _write_wav(wav, audio)
        inst = pretty_midi.Instrument(program=0)
        for i in range(100):
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=i * 50, time=i / 100))
        args = argparse.Namespace(
            emit_cc11=True,
            emit_cc64=False,
            cc_strategy="energy",
            controls_domain="time",
            controls_sample_rate_hz=50.0,
        )
        curves = build_control_curves_for_stem(wav, inst, args)
        pm = pretty_midi.PrettyMIDI()
        curves_by_ch = {0: curves}
        apply_controls(
            pm,
            curves_by_ch,
            bend_range_semitones=2.0,
            sample_rate_hz={"cc": 50.0, "bend": 50.0},
            max_events={"cc11": 100, "bend": 100},
            total_max_events=40,
        )
        total = sum(len(i.control_changes) + len(i.pitch_bends) for i in pm.instruments)
        assert total <= 40
        inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
        assert math.isclose(inst0.pitch_bends[0].time, 0.0, abs_tol=1e-9)
        assert math.isclose(inst0.pitch_bends[-1].time, 0.99, rel_tol=1e-2)
