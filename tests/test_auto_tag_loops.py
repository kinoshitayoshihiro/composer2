import subprocess
import sys
import json
import csv
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
pm = pretty_midi
yaml = pytest.importorskip("yaml")

from tools.auto_tag_loops import _load_thresholds, DEFAULT_THRESHOLDS

def make_drum_midi(path: Path) -> None:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.is_drum = True
    inst.notes.append(pm.Note(velocity=100, pitch=36, start=0.0, end=0.5))
    inst.notes.append(pm.Note(velocity=100, pitch=38, start=0.5, end=1.0))
    m.instruments.append(inst)
    m.write(str(path))


def make_poly_midi(path: Path) -> None:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.notes.append(pm.Note(velocity=80, pitch=90, start=0.0, end=1.0))
    inst.notes.append(pm.Note(velocity=80, pitch=94, start=0.0, end=1.0))
    m.instruments.append(inst)
    m.write(str(path))


def make_ts_midi(path: Path, num: int, den: int, offset: float = 0.0) -> None:
    m = pm.PrettyMIDI()
    m.time_signature_changes.append(pm.TimeSignature(num, den, 0))
    inst = pm.Instrument(program=0)
    inst.is_drum = True
    inst.notes.append(pm.Note(velocity=100, pitch=36, start=offset, end=offset + 0.3))
    m.instruments.append(inst)
    m.write(str(path))


def make_named_drum_midi(path: Path) -> None:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="Percussion")
    inst.notes.append(pm.Note(velocity=100, pitch=100, start=0.0, end=0.5))
    m.instruments.append(inst)
    m.write(str(path))


def run_tool(ind: Path, out_yaml: Path, thresholds: Path | None = None, extra: list[str] | None = None) -> dict:
    cmd = [
        sys.executable,
        '-m',
        'tools.auto_tag_loops',
        '--in', str(ind),
        '--out-combined', str(out_yaml),
        '--report', str(ind / 'r.csv'),
        '--overwrite',
        '--num-workers', '1',
        '--min-notes', '1'
    ]
    if thresholds:
        cmd.extend(['--thresholds', str(thresholds)])
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True)
    return yaml.safe_load(out_yaml.read_text()) if out_yaml.exists() else {}


def run_tool_sections(ind: Path, out_sec: Path, out_mood: Path, extra: list[str] | None = None) -> tuple[dict, dict]:
    cmd = [
        sys.executable,
        '-m',
        'tools.auto_tag_loops',
        '--in', str(ind),
        '--out-sections', str(out_sec),
        '--out-mood', str(out_mood),
        '--report', str(ind / 'r.csv'),
        '--overwrite',
        '--num-workers', '1',
        '--min-notes', '1'
    ]
    if extra:
        cmd.extend(extra)
    subprocess.run(cmd, check=True)
    return yaml.safe_load(out_sec.read_text()), yaml.safe_load(out_mood.read_text())


def test_drum_mode(tmp_path: Path) -> None:
    mid = tmp_path / 'drum.mid'
    make_drum_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml')
    entry = data['drum.mid']
    assert entry['mode'] == 'drums'
    assert entry['section'] in {"chorus", "verse", "intro", "fill", "bridge", "outro"}
    assert entry['mood'] in {"energetic", "relaxed", "aggressive", "melancholic", "neutral"}
    for key in ['confidence_section', 'confidence_mood', 'bpm', 'time_signature', 'bars']:
        assert key in entry


def test_poly_mode(tmp_path: Path) -> None:
    mid = tmp_path / 'poly.mid'
    make_poly_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml')
    entry = data['poly.mid']
    assert entry['mode'] == 'poly'
    assert all(k in entry for k in ['section', 'mood', 'confidence_section', 'confidence_mood'])


def test_time_signature_and_offset(tmp_path: Path) -> None:
    make_ts_midi(tmp_path / 'ts34.mid', 3, 4)
    make_ts_midi(tmp_path / 'ts68.mid', 6, 8)
    make_ts_midi(tmp_path / 'offset.mid', 4, 4, offset=0.5)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml')
    assert data['ts34.mid']['section']
    assert data['ts68.mid']['section']
    assert data['offset.mid']['section']


def test_track_name_detection(tmp_path: Path) -> None:
    mid = tmp_path / 'named.mid'
    make_named_drum_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml')
    assert data['named.mid']['mode'] == 'drums'


def test_thresholds_yaml(tmp_path: Path) -> None:
    th_file = tmp_path / 'th.yaml'
    th_file.write_text('drums:\n  section:\n    fill: 0.5\n')
    loaded = _load_thresholds(th_file)
    th = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    def deep_update(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v
    deep_update(th, loaded)
    assert th['drums']['section']['fill'] == 0.5


def test_csv_columns(tmp_path: Path) -> None:
    make_drum_midi(tmp_path / 'd.mid')
    make_poly_midi(tmp_path / 'p.mid')
    run_tool(tmp_path, tmp_path / 'tags.yaml')
    header = next(csv.reader((tmp_path / 'r.csv').open()))
    for col in ['fill_ratio', 'crash_rate', 'sync_score', 'poly_sustain_ratio', 'range_semitones', 'change_rate',
                'detect_channel10', 'detect_name', 'detect_gm_ratio', 'detect_score',
                'section_decision_rule', 'section_rule_score', 'mood_decision_rule', 'mood_rule_score']:
        assert col in header


def test_detection_weights(tmp_path: Path) -> None:
    mid = tmp_path / 'dr.mid'
    make_drum_midi(mid)
    th = tmp_path / 'th.yaml'
    th.write_text('drums:\n  detect:\n    channel10_weight: 0\n    name_weight: 0\n    gm_hits_weight: 0\n')
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', thresholds=th)
    assert data['dr.mid']['mode'] == 'poly'


def test_dry_run_summary(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    summary = tmp_path / 's.json'
    run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--dry-run', '--summary', str(summary)])
    assert summary.exists()
    assert not (tmp_path / 'tags.yaml').exists()
    assert not (tmp_path / 'r.csv').exists()


def test_min_notes_filter(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--min-notes', '10'])
    assert data == {}


def test_min_bars_filter(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--min-bars', '10'])
    assert data == {}


def test_mode_override(tmp_path: Path) -> None:
    mid = tmp_path / 'p.mid'
    make_poly_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--mode-override', '*.mid:drums'])
    assert data['p.mid']['mode'] == 'drums'


def test_confidence_threshold(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--min-conf-section', '1', '--min-conf-mood', '1'])
    assert data['d.mid']['section'] == 'unknown'
    assert data['d.mid']['mood'] == 'unknown'


def test_limit_and_glob(tmp_path: Path) -> None:
    make_drum_midi(tmp_path / 'a.mid')
    make_poly_midi(tmp_path / 'b.mid')
    make_drum_midi(tmp_path / 'c.mid')
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--glob', 'a.mid,b.mid', '--limit', '1'])
    assert list(data.keys()) == ['a.mid']


def test_split_output(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    sec = tmp_path / 's.yaml'
    mood = tmp_path / 'm.yaml'
    run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--split-output', '--out-sections', str(sec), '--out-mood', str(mood)])
    assert sec.exists() and mood.exists()
    sec_data = yaml.safe_load(sec.read_text())
    mood_data = yaml.safe_load(mood.read_text())
    assert 'd.mid' in sec_data and 'section' in sec_data['d.mid']
    assert 'd.mid' in mood_data and 'mood' in mood_data['d.mid']


def test_executor_switch(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    run_tool(tmp_path, tmp_path / 't_thread.yaml', extra=['--executor', 'thread'])
    run_tool(tmp_path, tmp_path / 't_proc.yaml', extra=['--executor', 'process'])


def make_empty_midi(path: Path) -> None:
    m = pm.PrettyMIDI()
    m.instruments.append(pm.Instrument(program=0))
    m.write(str(path))


def test_empty_and_tempo_fallback(tmp_path: Path) -> None:
    mid = tmp_path / 'e.mid'
    make_empty_midi(mid)
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--min-notes', '0'])
    entry = data['e.mid']
    assert entry['tempo_fallback'] and entry['ts_fallback']


def test_very_short_loop(tmp_path: Path) -> None:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.is_drum = True
    inst.notes.append(pm.Note(velocity=100, pitch=36, start=0.0, end=0.05))
    m.instruments.append(inst)
    m.write(str(tmp_path / 'short.mid'))
    data = run_tool(tmp_path, tmp_path / 'tags.yaml')
    assert 'short.mid' in data


def test_update_and_dedupe(tmp_path: Path) -> None:
    mid1 = tmp_path / 'a.mid'
    mid2 = tmp_path / 'b.mid'
    make_drum_midi(mid1)
    run_tool(tmp_path, tmp_path / 'tags.yaml')
    make_drum_midi(mid2)
    data2 = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--update', '--dedupe-strategy', 'skip'])
    assert set(data2.keys()) == {'a.mid', 'b.mid'}


def test_compat_simple(tmp_path: Path) -> None:
    mid = tmp_path / 'd.mid'
    make_drum_midi(mid)
    sec, mood = run_tool_sections(tmp_path, tmp_path / 's.yaml', tmp_path / 'm.yaml', extra=['--compat', 'simple'])
    assert list(sec['d.mid'].keys()) == ['section']
    assert list(mood['d.mid'].keys()) == ['mood']


def test_dedupe_suffix(tmp_path: Path) -> None:
    mid = tmp_path / 'a.mid'
    make_drum_midi(mid)
    run_tool(tmp_path, tmp_path / 'tags.yaml')
    data = run_tool(tmp_path, tmp_path / 'tags.yaml', extra=['--update', '--dedupe-strategy', 'suffix'])
    assert set(data.keys()) == {'a.mid', 'a.mid_1'}
