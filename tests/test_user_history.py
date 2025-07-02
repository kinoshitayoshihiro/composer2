import json
from utilities import user_history


def test_record_and_load(tmp_path, monkeypatch):
    hist_file = tmp_path / "hist.json"
    monkeypatch.setattr(user_history, "_HISTORY_FILE", hist_file)
    user_history.record_generate({"bpm": 120}, [{"instrument": "bass"}])
    data = json.loads(hist_file.read_text())
    assert data[0]["config"]["bpm"] == 120
    loaded = user_history.load_history()
    assert loaded == data
