import json
from pathlib import Path
from tools.corpus_to_phrase_csv import list_instruments


def make_jsonl(path: Path, entries):
    with path.open('w') as f:
        for obj in entries:
            f.write(json.dumps(obj) + '\n')


def test_list_instruments_counts(tmp_path, capsys):
    train = tmp_path / 'train'
    valid = tmp_path / 'valid'
    train.mkdir()
    valid.mkdir()
    make_jsonl(
        train / 'samples.jsonl',
        [
            {"instrument": "piano", "track_name": "Main", "program": 0, "path": "piano.mid"},
            {"meta": {"instrument": "bass", "track_name": "BassLine", "program": 32, "source_path": "bass.mid"}},
        ],
    )
    make_jsonl(
        valid / 'samples.jsonl',
        [
            {"meta": {"instrument": "piano", "track_name": "Pno2", "program": 0, "source_path": "piano2.mid"}}
        ],
    )
    out_json = tmp_path / 'stats.json'
    list_instruments(tmp_path, min_count=1, stats_json=out_json, examples_per_key=1)
    out = capsys.readouterr().out
    assert 'instrument:' in out
    assert 'piano: 2' in out
    assert 'bass: 1' in out
    data = json.loads(out_json.read_text())
    assert data['instrument']['piano'] == 2
    assert data['instrument']['bass'] == 1
    assert data['program']['32'] == 1
