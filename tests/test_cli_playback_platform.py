import sys
import types
from utilities import cli_playback

def test_find_player_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(cli_playback.shutil, "which", lambda _cmd: None)
    assert cli_playback.find_player() is None

def test_find_player_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.setattr(cli_playback.shutil, "which", lambda _cmd: None)
    assert cli_playback.find_player() is None
