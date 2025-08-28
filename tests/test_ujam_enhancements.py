from types import SimpleNamespace
from pathlib import Path

from tests import _stubs
from tools.ujam_bridge import ujam_map, utils, validate
import pretty_midi
import pytest
mido = pytest.importorskip("mido")


def _make_pm(starts):
    pm = _stubs.PrettyMIDI()
    inst = _stubs.Instrument(program=0, name="Guitar")
    for s in starts:
        inst.notes.append(_stubs.Note(velocity=100, pitch=60, start=s, end=s + 0.5))
    pm.instruments.append(inst)
    pm.time_signature_changes = [SimpleNamespace(numerator=4)]
    return pm


def _run_convert(monkeypatch, starts, **kwargs):
    pm_in = _make_pm(starts)
    holder = {}

    class PM(_stubs.PrettyMIDI):
        def __init__(self, path=None, resolution=480):
            if isinstance(path, str):
                self.__dict__ = pm_in.__dict__
            else:
                super().__init__()
                self.resolution = resolution
                holder['pm'] = self
        def write(self, path):
            holder['pm'] = self

    monkeypatch.setattr(ujam_map, 'pretty_midi', SimpleNamespace(
        PrettyMIDI=PM,
        Note=_stubs.Note,
        Instrument=_stubs.Instrument,
    ))
    class MF:
        def __init__(self, path):
            self.tracks = [[
                SimpleNamespace(type="track_name", name="Keyswitches"),
                SimpleNamespace(type="note_on", channel=0),
                SimpleNamespace(type="note_off", channel=0),
            ]]
        def save(self, path):
            holder['mf'] = self
    monkeypatch.setattr(ujam_map, 'mido', SimpleNamespace(MidiFile=MF))
    args = {
        'plugin': 'vg_iron2',
        'mapping': 'tools/ujam_bridge/configs/vg_iron2.yaml',
        'in_midi': 'in.mid',
        'out_midi': 'out.mid',
        'tags': None,
        'section_aware': 'off',
        'quant': 120,
        'swing': 0.0,
        'humanize': 0.0,
        'use_groove_profile': None,
        'dry_run': False,
        'ks_lead': 60.0,
        'no_redundant_ks': False,
        'periodic_ks': 0,
        'ks_headroom': 10.0,
        'ks_channel': 16,
        'ks_vel': 100,
        'groove_clip_head': 10.0,
        'groove_clip_other': 35.0,
    }
    args.update(kwargs)
    ujam_map.convert(SimpleNamespace(**args))
    return holder['pm']


def test_ks_lead_redundant_periodic(monkeypatch):
    pm = _run_convert(monkeypatch, [0.0, 2.0, 4.0], no_redundant_ks=True, periodic_ks=2)
    ks_notes = pm.instruments[0].notes
    assert len(ks_notes) == 2
    assert abs(ks_notes[1].start - 3.92) < 1e-6
    pm2 = _run_convert(monkeypatch, [0.0, 2.0, 4.0], no_redundant_ks=True, periodic_ks=0)
    assert len(pm2.instruments[0].notes) == 1


def test_groove_caps(monkeypatch):
    pm = _make_pm([0.0, 1.0])
    monkeypatch.setattr(utils, 'gp', SimpleNamespace(apply_groove=lambda b, p: b + 0.1))
    utils.apply_groove_profile(pm, {}, beats_per_bar=4, clip_head_ms=10, clip_other_ms=35)
    assert abs(pm.instruments[0].notes[0].start - 0.01) < 1e-6
    assert abs(pm.instruments[0].notes[1].start - 1.035) < 1e-6


def test_pattern_inference(monkeypatch):
    pm = _run_convert(monkeypatch, [0.0, 1.0])
    pitches = [n.pitch for n in pm.instruments[0].notes]
    assert pitches == [36, 36]


def test_validator(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text((Path('tools/ujam_bridge/configs/vg_iron2.yaml').read_text()))
    bad = tmp_path / "bad.yaml"
    bad.write_text("keyswitch:\n  a: 1\nsection_styles:\n  sec: [b]\n")
    assert validate.validate(good) == []
    problems = validate.validate(bad)
    assert problems and 'undefined keyswitch' in problems[0]


def test_validator_range(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text("play_range:\n  low:40\n  high:88\nkeyswitch:\n  a:60\n  b:200\n")
    issues = validate.validate(bad)
    assert any('out of range' in s for s in issues)
    assert any('overlaps play range' in s for s in issues)


def test_ks_channel_written(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Guitar")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    in_path = tmp_path / "in.mid"
    out_path = tmp_path / "out.mid"
    pm.write(in_path)
    args = SimpleNamespace(
        plugin='vg_iron2', mapping='tools/ujam_bridge/configs/vg_iron2.yaml',
        in_midi=str(in_path), out_midi=str(out_path), tags=None, section_aware='off',
        quant=120, swing=0.0, humanize=0.0, use_groove_profile=None, dry_run=False,
        ks_lead=60.0, no_redundant_ks=False, periodic_ks=0, ks_headroom=10.0,
        ks_channel=10, ks_vel=100, groove_clip_head=10.0, groove_clip_other=35.0)
    ujam_map.convert(args)
    mf = mido.MidiFile(out_path)
    channels = {msg.channel for msg in mf.tracks[0] if msg.type == 'note_on'}
    assert channels == {9}


def test_time_signature_change(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Guitar")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=2.0, end=2.5))
    pm.instruments.append(inst)
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(4, 4, 0),
        pretty_midi.TimeSignature(3, 4, 2.0),
    ]
    in_path = tmp_path / "ts.mid"
    out_path = tmp_path / "ts_out.mid"
    pm.write(in_path)
    args = SimpleNamespace(
        plugin='vg_iron2', mapping='tools/ujam_bridge/configs/vg_iron2.yaml',
        in_midi=str(in_path), out_midi=str(out_path), tags=None, section_aware='off',
        quant=120, swing=0.0, humanize=0.0, use_groove_profile=None, dry_run=False,
        ks_lead=60.0, no_redundant_ks=False, periodic_ks=0, ks_headroom=10.0,
        ks_channel=16, ks_vel=100, groove_clip_head=10.0, groove_clip_other=35.0)
    ujam_map.convert(args)
    out_pm = pretty_midi.PrettyMIDI(str(out_path))
    ks_notes = out_pm.instruments[0].notes
    assert len(ks_notes) == 2
    assert abs(ks_notes[1].start - 1.92) < 1e-3
