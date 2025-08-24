import os
import subprocess
from pathlib import Path

import pytest

from tests.test_corpus_to_phrase_csv import make_midi


def test_train_phrase_viz(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    m1 = tmp_path / "a.mid"
    m2 = tmp_path / "b.mid"
    make_midi(m1, [60, 62])
    make_midi(m2, [65])
    train_csv = tmp_path / "train.csv"
    valid_csv = tmp_path / "valid.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--in",
            str(tmp_path),
            "--out-train",
            str(train_csv),
            "--out-valid",
            str(valid_csv),
        ],
        check=True,
    )
    ckpt = tmp_path / "model.ckpt"
    subprocess.run(
        [
            "python",
            "scripts/train_phrase.py",
            str(train_csv),
            str(valid_csv),
            "--epochs",
            "1",
            "--out",
            str(ckpt),
            "--min-f1",
            "-1",
            "--batch-size",
            "1",
            "--d_model",
            "32",
            "--max-len",
            "32",
            "--viz",
        ],
        check=True,
        env={**os.environ, "ALLOW_LOCAL_IMPORT": "1"},
    )
    assert (tmp_path / "pr_curve_ep1.png").is_file()
    assert (tmp_path / "confusion_matrix_ep1.png").is_file()
