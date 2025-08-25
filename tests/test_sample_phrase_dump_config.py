import json
import sys
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from models.phrase_transformer import PhraseTransformer


def test_dump_config(tmp_path: Path) -> None:
    ckpt = tmp_path / "toy.ckpt"
    model = PhraseTransformer(d_model=16, max_len=8, nhead=2, num_layers=1, pitch_vocab_size=128)
    meta = {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "max_len": 8,
        "duv_mode": "reg",
        "vel_bins": 0,
        "dur_bins": 0,
        "vocab_pitch": 128,
        "vocab": {},
    }
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt)
    out_csv = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        "-m",
        "scripts.sample_phrase",
        "--ckpt",
        str(ckpt),
        "--length",
        "4",
        "--out-csv",
        str(out_csv),
        "--seed",
        "123",
        "--topk",
        "5",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    cfg_path = out_csv.with_suffix(out_csv.suffix + ".json")
    assert cfg_path.is_file()
    data = json.loads(cfg_path.read_text())
    assert data["topk"] == 5
    assert data["seed"] == 123
    assert "temperature_start" in data
    assert data.get("temperature_schedule") == "linear"
