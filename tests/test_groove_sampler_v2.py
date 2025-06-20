import json
import subprocess
import time
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_v2


def _make_full_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_train_speed_large(tmp_path: Path) -> None:
    for i in range(1000):
        _make_full_loop(tmp_path / f"{i}.mid")
    t0 = time.perf_counter()
    groove_sampler_v2.train(tmp_path, n=4)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0


def test_coarse_bucket_count(tmp_path: Path) -> None:
    _make_full_loop(tmp_path / "loop.mid")
    model = groove_sampler_v2.train(tmp_path, coarse=True)
    buckets = {st[1] for st in model.idx_to_state}
    assert len(buckets) == model.resolution // 4


def test_sample_cli_json_sorted(tmp_path: Path) -> None:
    _make_full_loop(tmp_path / "loop.mid")
    model = groove_sampler_v2.train(tmp_path)
    model_path = tmp_path / "model.pkl"
    groove_sampler_v2.save(model, model_path)
    result = subprocess.run(
        [
            "python",
            "-m",
            "utilities.groove_sampler_v2",
            "sample",
            str(model_path),
            "-l",
            "1",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    offsets = [ev["offset"] for ev in data]
    assert offsets == sorted(offsets)

