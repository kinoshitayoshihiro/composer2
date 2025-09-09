import json

from utilities import user_history


def test_record_and_load(tmp_path, monkeypatch):
    hist_file = tmp_path / "hist.jsonl"
    monkeypatch.setattr(user_history, "_HISTORY_FILE", hist_file)
    user_history.record_generate({"bpm": 120}, [{"instrument": "bass"}])
    lines = hist_file.read_text().splitlines()
    data = [json.loads(lines[0])]
    assert data[0]["config"]["bpm"] == 120
    loaded = user_history.load_history()
    assert loaded == data
