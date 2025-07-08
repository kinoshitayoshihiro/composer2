import json
import sys
import logging
from pathlib import Path

import pytest

from modular_composer import cli
from generator import vocal_generator


def test_cli_vocal_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dict_path = tmp_path / "d.json"
    dict_path.write_text(json.dumps({"あ": "X"}))
    model = tmp_path / "m.mid"
    model.write_text("m")

    captured: dict[str, object] = {}
    orig_compose = vocal_generator.VocalGenerator.compose

    def capture(self, *a, **k):
        part = orig_compose(self, *a, **k)
        captured["dict"] = self.phoneme_dict
        return part

    monkeypatch.setattr(vocal_generator.VocalGenerator, "compose", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modcompose",
            "sample",
            str(model),
            "--backend",
            "vocal",
            "--phoneme-dict",
            str(dict_path),
        ],
    )
    cli.main()

    assert [p for p, _, _ in vocal_generator.text_to_phonemes("あ", captured["dict"])] == [
        "X"
    ]


def test_cli_vocal_dict_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "missing.json"
    model = tmp_path / "m.mid"
    model.write_text("m")

    captured: dict[str, object] = {}
    orig_compose = vocal_generator.VocalGenerator.compose

    def capture(self, *a, **k):
        part = orig_compose(self, *a, **k)
        captured["dict"] = self.phoneme_dict
        return part

    monkeypatch.setattr(vocal_generator.VocalGenerator, "compose", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modcompose",
            "sample",
            str(model),
            "--backend",
            "vocal",
            "--phoneme-dict",
            str(missing),
        ],
    )

    with caplog.at_level(logging.WARNING):
        cli.main()

    assert any("Phoneme dictionary not found" in r.message for r in caplog.records)
    assert captured["dict"] == vocal_generator.PHONEME_DICT
