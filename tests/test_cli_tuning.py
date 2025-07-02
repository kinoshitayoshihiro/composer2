import sys
from pathlib import Path
import yaml
from music21 import stream
import modular_composer
from utilities.generator_factory import GenFactory


def _write_cfg(tmp_path: Path) -> Path:
    chordmap = {
        'sections': {
            'A': {
                'processed_chord_events': [
                    {
                        'absolute_offset_beats': 0.0,
                        'humanized_duration_beats': 4.0,
                        'chord_symbol_for_voicing': 'C',
                    }
                ],
                'musical_intent': {'emotion': 'neutral'},
                'expression_details': {},
            }
        }
    }
    cm_path = tmp_path / "cm.yml"
    cm_path.write_text(yaml.safe_dump(chordmap))
    cfg = {
        'global_settings': {
            'time_signature': '4/4',
            'tempo_bpm': 120,
            'key_tonic': 'C',
            'key_mode': 'major',
        },
        'sections_to_generate': ['A'],
        'paths': {
            'chordmap_path': str(cm_path),
            'rhythm_library_path': str(Path('data/rhythm_library.yml').resolve()),
            'output_dir': str(tmp_path),
        },
        'part_defaults': {'guitar': {}, 'rhythm': {}},
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def _run_cli(tmp_path: Path, tuning_value, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    captured = {}

    def fake_build_from_config(cfg, rl, tempo_map=None):
        captured['cfg'] = cfg
        return {}

    monkeypatch.setattr(GenFactory, 'build_from_config', fake_build_from_config)
    monkeypatch.setattr(modular_composer, 'compose', lambda *a, **k: (stream.Score(), []))
    monkeypatch.setattr(stream.Score, 'write', lambda self, fmt, fp: None)

    argv = ['modcompose', '--main-cfg', str(cfg_path), '--tuning', tuning_value, '--dry-run']
    monkeypatch.setattr(sys, 'argv', argv)
    modular_composer.main_cli()
    return captured['cfg']['part_defaults']['guitar']['tuning']


def test_cli_tuning_preset(tmp_path, monkeypatch):
    val = _run_cli(tmp_path, 'drop_d', monkeypatch)
    assert val == 'drop_d'


def test_cli_tuning_offsets(tmp_path, monkeypatch):
    val = _run_cli(tmp_path, '0,-2,0,0,0,0', monkeypatch)
    assert val == [0, -2, 0, 0, 0, 0]
