import logging
from pathlib import Path
from unittest import mock

import numpy as np
import pretty_midi
import pytest

import subprocess

from utilities import groove_sampler_v2, memmap_utils
import sys


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=36,
                start=start,
                end=start + 0.05,
            )
        )
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=38,
                start=start,
                end=start + 0.05,
            )
        )
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=42,
                start=start,
                end=start + 0.05,
            )
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_parallel_invocation(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    with (
        mock.patch("utilities.groove_sampler_v2.Parallel") as par,
        mock.patch(
            "utilities.groove_sampler_v2.delayed",
            side_effect=lambda f: (lambda p: lambda: f(p)),
        ),
    ):
        par.return_value.side_effect = lambda funcs: [f() for f in funcs]
        groove_sampler_v2.train(tmp_path, n_jobs=2)
    par.assert_called_once()
    assert par.call_args.kwargs == {
        "n_jobs": 2,
        "prefer": "threads",
        "batch_size": 1,
        "verbose": 0,
    }


def test_parallel_invocation_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _make_loop(tmp_path / "err.mid")
    with (
        mock.patch("utilities.groove_sampler_v2.Parallel") as par,
        mock.patch(
            "utilities.groove_sampler_v2.delayed",
            side_effect=lambda f: (lambda p: lambda: f(p)),
        ),
        caplog.at_level(logging.WARNING),
    ):
        par.side_effect = RuntimeError("parallel boom")
        with pytest.raises(RuntimeError):
            groove_sampler_v2.train(tmp_path, n_jobs=2)
    par.assert_called_once()
    assert par.call_args.kwargs == {
        "n_jobs": 2,
        "prefer": "threads",
        "batch_size": 1,
        "verbose": 0,
    }
    assert "Parallel groove loading failed" in caplog.text


def test_memmap_creation(tmp_path: Path) -> None:
    _make_loop(tmp_path / "b.mid")
    model = groove_sampler_v2.train(tmp_path, memmap_dir=tmp_path)
    path = tmp_path / "prob_order0.mmap"
    assert path.exists()
    mm = memmap_utils.load_memmap(path, shape=(len(model.ctx_maps[0]), len(model.idx_to_state)))
    assert mm.dtype == np.float32
    assert mm.shape == (len(model.ctx_maps[0]), len(model.idx_to_state))


def test_collision_no_kick_snare_same_tick(tmp_path: Path) -> None:
    _make_loop(tmp_path / "c.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    for off in {ev["offset"] for ev in events}:
        insts = [e["instrument"] for e in events if e["offset"] == off]
        assert not ("kick" in insts and "snare" in insts)


def test_velocity_condition_soft(tmp_path: Path) -> None:
    _make_loop(tmp_path / "d.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, cond_velocity="soft")
    assert all(ev["velocity_factor"] <= 0.8 for ev in events)


def test_train_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        groove_sampler_v2.train(tmp_path)


def _make_wav(path: Path) -> None:
    # create an invalid WAV to force a conversion failure
    path.write_bytes(b"invalid")


@pytest.mark.parametrize(
    "cli_args,scanned,skipped",
    [([], 2, 1), (["--no-audio"], 1, 0)],
)
def test_stats_audio_skip(tmp_path: Path, cli_args: list[str], scanned: int, skipped: int) -> None:
    loops = tmp_path / "loops"
    loops.mkdir()
    _make_loop(loops / "a.mid")
    _make_wav(loops / "a.wav")
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.groove_sampler_v2",
            "-q",
            "train",
            str(loops),
            "-o",
            str(model_path),
            *cli_args,
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.groove_sampler_v2",
            "stats",
            str(model_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Scanned {scanned} files" in result.stdout
    assert f"skipped {skipped}" in result.stdout
